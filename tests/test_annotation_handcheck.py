"""Silver hand-check: marks saved by the app are read by `silver.handcheck_accuracy`.

The hand-check file is made by the real `filter_silver` and `export_handcheck` from
invented sentences.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

from explain_to_learn.annotation.handcheck import OUTPUT_COLUMNS, HandcheckTask
from explain_to_learn.edits import Edit
from explain_to_learn.silver import export_handcheck, filter_silver, handcheck_accuracy

ROOT = Path(__file__).parents[1]


def make_edit(idx: int) -> Edit:
    return Edit(
        edit_id=f"wi-{idx}-0-0",
        corpus="wi",
        essay_id=str(idx),
        split="train",
        cefr_level="B1",
        sent_idx=0,
        original=f"She do n't like tea {idx} .",
        corrected=f"She does n't like tea {idx} .",
        start=1,
        end=2,
        original_text="do",
        corrected_text="does",
        error_type="R:VERB:SVA",
        operation="R",
        error_category="VERB:SVA",
    )


EDITS = {edit.edit_id: edit for edit in map(make_edit, range(5))}


def teacher_row(edit_id: str, explanation: str) -> dict:
    """A teacher output as `scripts/make_silver.py generate` writes it."""
    return {
        "edit_id": edit_id,
        "level": "B1",
        "mode": "explain",
        "parsed": {"concept": "Subject-verb agreement", "explanation": explanation, "hint": ""},
        "concept_id": "subject_verb_agreement",
        "concept_name": "Subject-verb agreement",
    }


@pytest.fixture
def handcheck_file(tmp_path):
    rows = [teacher_row(edit_id, 'Use "does", not "do", after "she".') for edit_id in EDITS]
    kept, _ = filter_silver(rows, EDITS)
    path = tmp_path / "silver" / "silver_handcheck.csv"
    assert export_handcheck(kept, EDITS, path, n=4, seed=0) == 4
    return path


def test_marks_round_trip_through_handcheck_accuracy(handcheck_file, tmp_path):
    task = HandcheckTask(handcheck_file, tmp_path / "annotations")
    assert len(task) == 4 and task.accuracy() is None
    task.save(0, "1", "")
    task.save(1, "0", "  The rule is too general.  ")
    task.save(2, "1", "")

    path = tmp_path / "annotations" / "silver_handcheck.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert [row["correct"] for row in rows] == ["1", "0", "1"]
    assert rows[1]["note"] == "The rule is too general."
    result = handcheck_accuracy(path)
    assert (result["rated"], result["correct"], result["unrated"]) == (3, 2, 0)
    assert "accuracy so far 66.7% (95% CI" in task.status(3)


def test_make_silver_score_reads_the_marks(handcheck_file, tmp_path, monkeypatch, capsys):
    task = HandcheckTask(handcheck_file, tmp_path / "annotations")
    for index in range(len(task)):
        task.save(index, "1", "")
    spec = importlib.util.spec_from_file_location(
        "make_silver", ROOT / "scripts" / "make_silver.py"
    )
    make_silver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_silver)

    marks = tmp_path / "annotations" / "silver_handcheck.csv"
    make_silver.main(["--handcheck", str(marks), "--results", str(tmp_path / "results"), "score"])

    assert "Hand-check accuracy 100.0%" in capsys.readouterr().out
    assert (tmp_path / "results" / "silver_handcheck.json").exists()


def test_view_values_and_resume(handcheck_file, tmp_path):
    task = HandcheckTask(handcheck_file, tmp_path)
    view = task.view(0)[0]
    assert "**Explanation**" in view and '> Use "does", not "do", after "she".' in view
    assert "**Edit:** do → does" in view
    task.save(0, "0", "Wrong concept.")

    resumed = HandcheckTask(handcheck_file, tmp_path)
    assert resumed.start_index() == 1
    assert resumed.values(0) == ["0", "Wrong concept."]
    assert resumed.values(1) == [None, ""]
    with pytest.raises(ValueError, match="correct or not"):
        resumed.save(1, None, "No mark.")


def test_a_changed_sample_is_refused(handcheck_file, tmp_path):
    HandcheckTask(handcheck_file, tmp_path).save(0, "1", "")
    with handcheck_file.open(newline="") as file:
        rows = list(csv.DictReader(file))
    rows[0]["explanation"] = "A new text the teacher never saw."
    with handcheck_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="no longer in"):
        HandcheckTask(handcheck_file, tmp_path)


def test_missing_or_repeated_items_are_errors(handcheck_file, tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        HandcheckTask(handcheck_file, handcheck_file.parent)
    with pytest.raises(ValueError, match="make_silver.py filter"):
        HandcheckTask(tmp_path / "missing.csv", tmp_path)
    lines = handcheck_file.read_text().splitlines()
    handcheck_file.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(ValueError, match="more than once"):
        HandcheckTask(handcheck_file, tmp_path)
