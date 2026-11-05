"""Gold explanations: item view, concept from the ERRANT mapping, drafts and export."""

import json
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("yaml")

from explain_to_learn.annotation.gold import GoldTask, read_drafts  # noqa: E402
from explain_to_learn.edits import Edit, write_jsonl  # noqa: E402

GRAPH_DIR = Path(__file__).parent / "fixtures" / "graph"
FIELDS = {
    "edit_id",
    "level",
    "error_type",
    "concept_id",
    "concept",
    "explanation",
    "hint",
    "not_rule_explainable",
    "notes",
    "from_draft",
}


def make_edit(edit_id, original, start, end, corrected_text, error_type, level, split="dev"):
    """A synthetic edit; the sentences are invented for the tests."""
    tokens = original.split()
    operation, category = error_type.split(":", 1)
    corrected = [*tokens[:start], *corrected_text.split(), *tokens[end:]]
    return Edit(
        edit_id=edit_id,
        corpus="wi",
        essay_id=edit_id.split("-")[1],
        split=split,
        cefr_level=level,
        sent_idx=0,
        original=original,
        corrected=" ".join(corrected),
        start=start,
        end=end,
        original_text=" ".join(tokens[start:end]),
        corrected_text=corrected_text,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


EDITS = [
    make_edit("wi-1-0-0", "She do n't like tea .", 1, 2, "does", "R:VERB:SVA", "A2"),
    make_edit("wi-1-0-1", "The life is hard .", 0, 1, "", "U:DET", "B1"),
    make_edit("locness-2-0-0", "I bought book .", 2, 2, "a", "M:DET", "native"),
    make_edit("wi-3-0-0", "My cat sleep .", 2, 3, "sleeps", "R:VERB:SVA", "B1", "train"),
]
GOLD_IDS = ["wi-1-0-0", "wi-1-0-1", "locness-2-0-0"]


def draft(edit_id, mode, explanation="", hint=""):
    """A generation record as `scripts/run_system.py` writes it."""
    parsed = {"concept": "Agreement", "explanation": explanation, "hint": hint}
    return {"edit_id": edit_id, "system": "R", "mode": mode, "parsed": parsed}


@pytest.fixture
def files(tmp_path):
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    (tmp_path / "gold.txt").write_text("\n".join(GOLD_IDS) + "\n")
    runs = {
        "explain.jsonl": [
            draft("wi-1-0-0", "explain", explanation='Use "does" after "she".'),
            {"edit_id": "wi-1-0-1", "mode": "explain", "parsed": None},
        ],
        "hint.jsonl": [draft("wi-1-0-0", "hint", hint='Look at "do".')],
    }
    for name, rows in runs.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def gold_task(files, drafts=()):
    return GoldTask(
        files / "gold.txt",
        files / "edits.jsonl",
        GRAPH_DIR,
        files / "annotations",
        [files / name for name in drafts],
    )


def test_view_marks_the_edit_and_prefills_the_mapped_concept(files):
    task = gold_task(files)

    assert len(task) == 3
    item, concept, draft_panel = task.view(0)
    assert "She [**do → does**]n't like tea." in item
    assert "**Corrected:** She doesn't like tea." in item
    assert "`R:VERB:SVA`" in item and "**Level:** A2" in item
    assert "**Rule for A2:** With he, she or it, add -s to the verb." in concept
    assert "wrong *He like tea.*, right *He likes tea.*" in concept
    assert draft_panel == ""
    assert task.values(0) == ["subject_verb_agreement", "", "", "", False, False]
    assert task.values(1)[0] == "zero_article"  # the U:DET override
    assert "**Level:** native (write for C2)" in task.view(2)[0]
    assert "**Rule for C2:** Use the for something specific" in task.view(2)[1]


def test_concept_view_follows_the_choice_and_flags_missing_content(files):
    task = gold_task(files)

    other = task.concept_view(0, "noun_plurals")
    assert "**Rule for A2:** Add -s to a noun" in other
    assert "The ERRANT mapping gives `subject_verb_agreement`" in other
    empty = task.concept_view(1, "zero_article")
    assert "*not written yet*" in empty
    assert ("Singular and plural nouns (noun_plurals)", "noun_plurals") in task.concept_choices()


def test_drafts_merge_explain_and_hint_runs_and_fill_only_empty_boxes(files):
    assert read_drafts([files / "explain.jsonl", files / "hint.jsonl"])["wi-1-0-0"] == {
        "concept": "Agreement",
        "explanation": 'Use "does" after "she".',
        "hint": 'Look at "do".',
    }
    task = gold_task(files, drafts=("explain.jsonl", "hint.jsonl"))

    assert "**Model draft** (concept: Agreement)" in task.view(0)[2]
    assert task.view(1)[2] == "*No model draft for this edit.*"
    assert task.fill_from_draft(0, "", "My hint.", False) == (
        'Use "does" after "she".',
        "My hint.",
        True,
    )
    assert task.fill_from_draft(0, "Mine.", "Also mine.", False) == ("Mine.", "Also mine.", False)
    assert task.fill_from_draft(1, "", "", False) == ("", "", False)


def test_export_format_and_resume(files):
    task = gold_task(files)
    task.save(
        1, "zero_article", ' Remove "the". ', "Do you need a word before life?", "", False, True
    )
    task.save(0, "subject_verb_agreement", 'Use "does".', 'Look at "do".', "check", False, False)

    path = files / "annotations" / "gold_explanations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["edit_id"] for row in rows] == ["wi-1-0-0", "wi-1-0-1"]
    assert all(set(row) == FIELDS for row in rows)
    assert rows[1] == {
        "edit_id": "wi-1-0-1",
        "level": "B1",
        "error_type": "U:DET",
        "concept_id": "zero_article",
        "concept": "No article for general meaning",
        "explanation": 'Remove "the".',
        "hint": "Do you need a word before life?",
        "not_rule_explainable": False,
        "notes": "",
        "from_draft": True,
    }

    resumed = gold_task(files)
    assert resumed.start_index() == 2
    assert resumed.values(0) == [
        "subject_verb_agreement",
        'Use "does".',
        'Look at "do".',
        "check",
        False,
        False,
    ]


def test_save_checks_the_texts_and_the_rule_flag(files):
    task = gold_task(files)

    with pytest.raises(ValueError, match="explanation and the hint"):
        task.save(0, "subject_verb_agreement", "Text.", "", "", False, False)
    with pytest.raises(ValueError, match="notes"):
        task.save(0, None, "", "", "", True, False)
    with pytest.raises(ValueError, match="Unknown concept"):
        task.save(0, "no_such_concept", "Text.", "Hint.", "", False, False)

    task.save(2, None, "", "", "Only style; no rule.", True, False)
    row = json.loads((files / "annotations" / "gold_explanations.jsonl").read_text())
    assert (row["level"], row["concept_id"], row["not_rule_explainable"]) == ("C2", "", True)


def test_a_gold_edit_missing_from_the_edit_table_is_an_error(files):
    (files / "gold.txt").write_text("wi-1-0-0\nwi-9-0-0\n")

    with pytest.raises(ValueError, match="1 gold edits are not in"):
        gold_task(files)
