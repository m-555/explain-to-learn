import csv
import json

import pytest

from explain_to_learn.edits import Edit
from explain_to_learn.silver import (
    assign_modes,
    counts_table,
    export_handcheck,
    filter_silver,
    handcheck_accuracy,
    rejection_reason,
    select_training_edits,
    wilson_interval,
)


def make_edit(edit_id, essay_id, original, start, end, corrected_text, error_type):
    """A synthetic edit; the sentences are invented for the tests."""
    tokens = original.split()
    operation, category = error_type.split(":", 1)
    return Edit(
        edit_id=edit_id,
        corpus="wi",
        essay_id=essay_id,
        split="train",
        cefr_level="B1",
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=" ".join(tokens[start:end]),
        corrected_text=corrected_text,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


SVA = make_edit("wi-1-0-0", "1", "She do n't like tea .", 1, 2, "does", "R:VERB:SVA")
DET = make_edit("wi-1-0-1", "1", "I have cat .", 2, 2, "a", "M:DET")
EDITS = {edit.edit_id: edit for edit in (SVA, DET)}
SVA_LABELS = {"subject_verb_agreement", "Subject-verb agreement"}


def reply(concept, explanation="", hint=""):
    return {"concept": concept, "explanation": explanation, "hint": hint}


def row(edit_id, mode, parsed, concept_id="subject_verb_agreement"):
    name = {"subject_verb_agreement": "Subject-verb agreement", "articles": "Articles"}
    return {
        "edit_id": edit_id,
        "level": "B1",
        "mode": mode,
        "parsed": parsed,
        "concept_id": concept_id,
        "concept_name": name.get(concept_id),
    }


def test_select_training_edits_filters_and_is_seeded():
    edits = [
        make_edit(f"wi-{essay}-0-{i}", str(essay), "a b .", 0, 1, "c", "R:NOUN")
        for essay in (1, 2)
        for i in range(10)
    ]
    train = {"wi-1"}
    first = select_training_edits(edits, train, n=5, seed=3, exclude_ids={"wi-1-0-0"})

    assert first == select_training_edits(edits, train, n=5, seed=3, exclude_ids={"wi-1-0-0"})
    assert len(first) == 5
    assert all(e.essay_id == "1" and e.edit_id != "wi-1-0-0" for e in first)
    assert [e.edit_id for e in first] == sorted(e.edit_id for e in first)
    assert len(select_training_edits(edits, train, n=50)) == 10
    assert select_training_edits(edits, train, n=50, has_concept=lambda t: False) == []


def test_assign_modes_share_and_determinism():
    ids = [f"e{i}" for i in range(2000)]
    modes = assign_modes(ids, hint_share=0.25, seed=1)

    assert modes == assign_modes(reversed(ids), hint_share=0.25, seed=1)
    assert 0.2 < sum(m == "hint" for m in modes.values()) / len(ids) < 0.3


@pytest.mark.parametrize(
    ("edit", "mode", "parsed", "labels", "reason"),
    [
        (SVA, "explain", None, SVA_LABELS, "invalid_json"),
        (SVA, "explain", reply("Subject-verb agreement", hint="Look."), SVA_LABELS, "empty_text"),
        (SVA, "explain", reply("Agreement", 'Use "does".'), None, "no_concept"),
        (SVA, "explain", reply("Verb tense", 'Use "does", not "do".'), SVA_LABELS, "wrong_concept"),
        (
            SVA,
            "explain",
            reply("Subject-verb agreement", 'Use "does".'),
            SVA_LABELS,
            "missing_original",
        ),
        (DET, "explain", reply("Articles", "Add an article."), {"articles"}, "missing_correction"),
        (
            SVA,
            "hint",
            reply("Subject-verb agreement", hint='Change "do" to "does".'),
            SVA_LABELS,
            "reveals_correction",
        ),
        (
            SVA,
            "explain",
            reply("subject_verb_agreement", 'Use "does", not "do".'),
            SVA_LABELS,
            None,
        ),
    ],
)
def test_rejection_reasons(edit, mode, parsed, labels, reason):
    assert rejection_reason(edit, mode, parsed, labels) == reason


def test_filter_silver_keeps_good_items_with_graph_names():
    rows = [
        row("wi-1-0-0", "explain", reply("subject verb agreement", 'Use "does", not "do".', "x")),
        row("wi-1-0-0", "hint", reply("Subject-verb agreement", hint='Change "do" to "does".')),
        row(
            "wi-1-0-1", "hint", reply("Articles", "y", 'What is missing before "cat"?'), "articles"
        ),
        row("wi-1-0-1", "explain", reply("Articles", 'Add "a".'), None),
        row("wi-1-0-1", "explain", None, "articles"),
    ]
    kept, counts = filter_silver(rows, EDITS)

    assert counts == {"kept": 2, "reveals_correction": 1, "no_concept": 1, "invalid_json": 1}
    assert kept[0]["concept"] == "Subject-verb agreement"
    assert kept[0]["hint"] == ""
    assert kept[1]["explanation"] == "" and kept[1]["mode"] == "hint"
    assert json.loads(kept[0]["target"]) == {
        "concept": "Subject-verb agreement",
        "explanation": 'Use "does", not "do".',
        "hint": "",
    }
    table = counts_table(counts)
    assert "| kept | 2 | 40.0% |" in table and "| total | 5 | |" in table


def test_handcheck_export_and_accuracy(tmp_path):
    kept, _ = filter_silver(
        [row("wi-1-0-0", "explain", reply("Subject-verb agreement", 'Use "does", not "do".'))],
        EDITS,
    )
    path = tmp_path / "handcheck.csv"
    assert export_handcheck(kept * 5, EDITS, path, n=3, seed=0) == 3

    with path.open(newline="") as lines:
        rows = list(csv.DictReader(lines))
    assert rows[0]["sentence"] == "She don't like tea."
    assert rows[0]["correct"] == ""

    for mark, item in zip(["yes", "0", ""], rows, strict=True):
        item["correct"] = mark
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = handcheck_accuracy(path)
    assert (result["rated"], result["correct"], result["unrated"]) == (2, 1, 1)
    assert result["accuracy"] == 0.5
    assert result["low"] < 0.5 < result["high"]


def test_handcheck_rejects_unknown_marks(tmp_path):
    path = tmp_path / "handcheck.csv"
    path.write_text("edit_id,correct\ne1,maybe\n")
    with pytest.raises(ValueError):
        handcheck_accuracy(path)


def test_wilson_interval():
    low, high = wilson_interval(180, 200)
    assert 0.85 < low < 0.9 < high < 0.94
    assert wilson_interval(0, 0) == (0.0, 1.0)
