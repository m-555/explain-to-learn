"""The GEE loader and the comparison script, on an invented fixture in GEE's format.

`fixtures/gee/en_anonymized.json` copies the structure of GEE's English annotation file;
its sentences and explanations are invented for the tests.
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from explain_to_learn.edits import Edit, write_jsonl
from explain_to_learn.gee import (
    match_edits,
    parse_atomic_edit,
    parse_record,
    read_gee,
    reference_row,
    text_key,
)

ROOT = Path(__file__).parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "gee" / "en_anonymized.json"


def make_edit(essay, sent, original, start, end, correction, error_type, corrected, idx=0):
    """An official edit of the edit table; `original` and `corrected` are M2 tokens."""
    tokens = original.split()
    operation, category = error_type.split(":", 1)
    return Edit(
        edit_id=f"wi-{essay}-{sent}-{idx}",
        corpus="wi",
        essay_id=essay,
        split="dev",
        cefr_level="B1",
        sent_idx=sent,
        original=original,
        corrected=corrected,
        start=start,
        end=end,
        original_text=" ".join(tokens[start:end]),
        corrected_text=correction,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


APPLES = "She like apples and he play football ."
APPLES_FIXED = "She likes apples and he plays football ."
PLAN, PLAN_FIXED = "We discussed about the new plan", "We discussed the new plan ."
LEFT = "However we left early so we missed it ."
LEFT_FIXED = "However , we left early , so we missed it ."
EDITS = [
    # The sentence of record 1 occurs twice; only E1's correction is GEE's target.
    make_edit(
        "E9", 0, APPLES, 1, 2, "liked", "R:VERB:TENSE", "She liked apples and he play football ."
    ),
    make_edit("E1", 0, APPLES, 1, 2, "likes", "R:VERB:SVA", APPLES_FIXED),
    make_edit("E1", 0, APPLES, 5, 6, "plays", "R:VERB:SVA", APPLES_FIXED, idx=1),
    make_edit("E2", 3, PLAN, 2, 3, "", "U:PREP", PLAN_FIXED),
    make_edit("E2", 3, PLAN, 6, 6, ".", "M:PUNCT", PLAN_FIXED, idx=1),
    make_edit("E4", 0, "I always am late .", 1, 3, "am always", "R:WO", "I am always late ."),
    make_edit("E6", 1, LEFT, 1, 1, ",", "M:PUNCT", LEFT_FIXED),
    make_edit("E6", 1, LEFT, 4, 4, ",", "M:PUNCT", LEFT_FIXED, idx=1),
    make_edit(
        "E7", 2, "He do n't know it .", 1, 3, "does n't", "R:VERB:SVA", "He does n't know it ."
    ),
]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Parsing ---------------------------------------------------------------------------


def test_parse_atomic_edit_reads_python_style_lists():
    assert parse_atomic_edit("['replace', 'ditch', 'ditches']") == ("replace", "ditch", "ditches")
    assert parse_atomic_edit("['insert', '', ',']") == ("insert", "", ",")
    assert parse_atomic_edit("['replace', \"don't\", \"didn't\"]") == ("replace", "don't", "didn't")
    for bad in ("['insert', '', ''s']", "['replace', 'for', 'towards', ',']", "['swap', 'a', 'b']"):
        with pytest.raises(ValueError):
            parse_atomic_edit(bad)


def test_parse_record_pairs_explanations_with_edits_in_order():
    records = json.loads(FIXTURE.read_text())
    first = parse_record(records[0])

    assert first.gee_id == "1"
    assert first.source == "She like apples and he play football."
    assert first.target == "She likes apples and he plays football."
    assert [(e.operation, e.original, e.corrected) for e in first.edits] == [
        ("replace", "like", "likes"),
        ("replace", "play", "plays"),
    ]
    assert first.edits[1].explanation.startswith("The word 'play' is replaced by 'plays'")
    assert first.edits[0].error_type == "person and number"
    assert first.all_correct


def test_annotator_flags_go_to_the_explanation_they_highlight():
    # Record 2 flags its second explanation; its "Missing error" span is in the target.
    second = parse_record(json.loads(FIXTURE.read_text())[1])

    assert [edit.flags for edit in second.edits] == [(), ("Wrong error explanation",)]
    assert not second.all_correct


def test_read_gee_counts_the_records_it_cannot_use():
    records, counts = read_gee(FIXTURE)

    assert [r.gee_id for r in records] == ["1", "2", "4", "5", "6", "7"]
    assert counts == {
        "records used": 6,
        "skipped: explanations and edits differ in number": 1,
        "skipped: unreadable atomic edit": 1,
        "skipped: no edits": 1,
    }


# --- Matching with the edit table -----------------------------------------------------


def test_text_key_matches_m2_tokens_with_written_text():
    assert text_key("do n't") == text_key("don't")
    assert text_key("It ’s") == text_key("It's")
    assert text_key("a b") != text_key("a c")


def test_match_edits_pairs_atomic_edits_with_official_edits():
    records, _ = read_gee(FIXTURE)
    matches, counts = match_edits(records, EDITS)

    assert [(m.gee_id, m.edit.edit_id) for m in matches] == [
        ("1", "wi-E1-0-0"),  # E1's correction is GEE's target, E9's is not
        ("1", "wi-E1-0-1"),
        ("2", "wi-E2-3-0"),
        ("2", "wi-E2-3-1"),
        ("6", "wi-E6-1-0"),  # the two inserted commas, in sentence order
        ("6", "wi-E6-1-1"),
        ("7", "wi-E7-2-0"),  # "don't" -> "doesn't" against the M2 tokens "do n't"
    ]
    assert counts == {
        "atomic edits matched": 7,
        "atomic edits dropped: relocate with no official edit": 1,
        "sentences not in the edit table": 1,
        "atomic edits dropped: sentence not in the edit table": 1,
    }
    row = reference_row(matches[3])
    assert row["flags"] == ["Wrong error explanation"] and row["error_type"] == "punctuation"
    assert row["all_correct"] is False


# --- The comparison script -------------------------------------------------------------


class FakeGraph:
    def concept_for(self, error_type):
        return None


class FakeModel:
    """Names both sides of the edit when it can find them, and is vague otherwise."""

    def complete(self, messages, decoding):
        user = messages[1]["content"]
        correction = next(line for line in user.splitlines() if line.startswith("Correction:"))
        if "replace" in correction:
            words = correction.split('"')
            text = f'Use "{words[3]}", not "{words[1]}".'
        else:
            text = "Look at this part of the sentence."
        return json.dumps({"concept": "Grammar", "explanation": text, "hint": ""})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    (tmp_path / "gold.txt").write_text("wi-E7-2-0\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return tmp_path


def test_prepare_run_and_compare(workspace, monkeypatch):
    pytest.importorskip("yaml")
    gee = load_script("evaluate_on_gee")
    common = ["--edits", "edits.jsonl", "--gee-dir", "gee", "--outputs", "outputs"]

    gee.main([*common, "prepare", "--gee", str(FIXTURE), "--gold-ids", "gold.txt"])
    ids = Path("gee/edit_ids.txt").read_text().split()
    assert len(ids) == 7
    counts = json.loads(Path("gee/prepare_counts.json").read_text())
    assert counts["matched edits in the gold set"] == 1 and counts["records used"] == 6

    import run_system

    monkeypatch.setattr(run_system, "make_backend", lambda system: FakeModel())
    monkeypatch.setattr(run_system, "open_graph", lambda directory: FakeGraph())
    config = str(ROOT / "configs" / "systems.yaml")
    gee.main([*common, "run", "--system", "D", "--config", config])
    rows = [json.loads(line) for line in Path("outputs/D/gee.jsonl").read_text().splitlines()]
    assert [row["edit_id"] for row in rows] == ids
    assert {row["mode"] for row in rows} == {"explain"}

    gee.main(
        [*common, "compare", "--systems", "D", "C", "--out-dir", "results", "--resamples", "200"]
    )
    results = json.loads(Path("results/gee_comparison.json").read_text())
    gee_scores, d_scores = results["sources"]["GEE"], results["sources"]["D"]
    assert "C" not in results["sources"]  # no outputs for C
    assert gee_scores["covered"]["share"] == 1.0 and gee_scores["faithful"]["share"] == 1.0
    # D names both sides of the three replacements and none of the four insertions or deletions.
    assert d_scores["faithful"]["share"] == pytest.approx(3 / 7)
    assert results["differences"]["D - GEE"]["faithful"]["diff"] == pytest.approx(-4 / 7)
    assert results["flagged_by_gee_annotator"] == 1

    report = Path("results/gee_comparison.md").read_text()
    assert "| GEE explanation | 7 |" in report and "| System D | 7 |" in report
    assert "Concept accuracy" in report and "apples" not in report and "'like'" not in report

    with Path("gee/side_by_side.csv").open(newline="") as lines:
        sheet = list(csv.DictReader(lines))
    assert len(sheet) == 7
    assert sheet[0]["gee_explanation"].startswith("The word 'like'")
    assert sheet[0]["system_D"] == 'Use "likes", not "like".'
    assert sheet[0]["sentence"] == "She like apples and he play football."
    assert sheet[0]["preferred"] == "" and "system_C" not in sheet[0]


def test_compare_needs_the_prepare_step(workspace):
    gee = load_script("evaluate_on_gee")

    with pytest.raises(SystemExit, match="prepare"):
        gee.main(["--gee-dir", "gee", "compare"])
