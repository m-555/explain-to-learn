"""Preference pairs: blind side-by-side view and the export format for DPO data."""

import json
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("yaml")

from explain_to_learn.annotation.preferences import (  # noqa: E402
    PreferenceTask,
    build_pairs,
    training_target,
)
from explain_to_learn.edits import Edit, write_jsonl  # noqa: E402
from explain_to_learn.generate import edit_item, parse_output  # noqa: E402
from explain_to_learn.graph import load_graph  # noqa: E402

GRAPH_DIR = Path(__file__).parent / "fixtures" / "graph"
FIELDS = {
    "edit_id",
    "level",
    "mode",
    "prompt_messages",
    "chosen",
    "rejected",
    "source_systems",
    "judgement",
}


def make_edit(idx: int) -> Edit:
    """A synthetic edit; the sentences are invented for the tests."""
    return Edit(
        edit_id=f"wi-{idx}-0-0",
        corpus="wi",
        essay_id=str(idx),
        split="train",
        cefr_level="B1",
        sent_idx=0,
        original=f"He go to school {idx} .",
        corrected=f"He goes to school {idx} .",
        start=1,
        end=2,
        original_text="go",
        corrected_text="goes",
        error_type="R:VERB:SVA",
        operation="R",
        error_category="VERB:SVA",
    )


EDITS = [make_edit(i) for i in range(8)]
# Each source writes its own wording, so the tests can tell the outputs apart.
WORDING = {"SOURCE-ONE": "wording x", "SOURCE-TWO": "wording y", "A": "a", "B": "b"}


def record(edit, source, mode="explain", parsed=True, text=None):
    """A generation record as `scripts/run_system.py` writes it, for a grounded system."""
    text = text or f"Answer {edit.essay_id} in {WORDING[source]}: use goes after he."
    output = {
        "concept": "Subject-verb agreement",
        "explanation": text if mode == "explain" else "unused",
        "hint": text if mode == "hint" else "unused",
    }
    return {
        "edit_id": edit.edit_id,
        "system": source,
        "level": "B1",
        "mode": mode,
        "grounded": True,
        "parsed": output if parsed else None,
    }


@pytest.fixture
def files(tmp_path):
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    first = [record(edit, "SOURCE-ONE") for edit in EDITS[:6]]
    second = [record(edit, "SOURCE-TWO") for edit in EDITS[:4]]
    second.append(record(EDITS[4], "SOURCE-TWO", text=first[4]["parsed"]["explanation"]))
    second.append(record(EDITS[5], "SOURCE-TWO", parsed=False))
    for name, rows in (("one.jsonl", first), ("two.jsonl", second)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def preference_task(files, order=("SOURCE-TWO", "SOURCE-ONE")):
    paths = {"SOURCE-ONE": files / "one.jsonl", "SOURCE-TWO": files / "two.jsonl"}
    return PreferenceTask(
        [(name, paths[name]) for name in order],
        files / "edits.jsonl",
        GRAPH_DIR,
        files / "annotations",
        seed=3,
    )


def read_output(files):
    path = files / "annotations" / "preferences.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pairs_need_two_different_readable_texts_in_the_same_mode():
    first = {e.edit_id: record(e, "A") for e in EDITS[:4]}
    second = {
        EDITS[0].edit_id: record(EDITS[0], "B"),
        EDITS[1].edit_id: record(EDITS[1], "B", mode="hint"),
        EDITS[2].edit_id: record(EDITS[2], "B", parsed=False),
    }

    pairs, skipped = build_pairs(first, second)
    assert [pair["edit_id"] for pair in pairs] == [EDITS[0].edit_id]
    assert skipped == {"different_mode": 1, "no_readable_output": 1, "only_in_one_file": 1}


def test_view_is_blind_and_the_order_is_fixed_per_edit(files):
    task = preference_task(files)
    again = preference_task(files, order=("SOURCE-ONE", "SOURCE-TWO"))

    assert task.names == ("SOURCE-ONE", "SOURCE-TWO")
    assert len(task) == 4 and task.skipped == {"same_text": 1, "no_readable_output": 1}
    flips = [task.swapped(index) for index in range(len(task))]
    assert flips == [again.swapped(index) for index in range(len(task))]
    for index in range(len(task)):
        item, panel_1, panel_2 = task.view(index)
        assert "He [**go → goes**] to school" in item
        assert panel_1.startswith("**Explanation 1** · concept: Subject-verb agreement")
        text = "\n".join([task.status(index), item, panel_1, panel_2])
        assert "wording x" in text and "wording y" in text
        assert "SOURCE" not in text
        assert "unused" not in text  # only the text for the mode is shown


def test_a_choice_is_saved_as_chosen_and_rejected_training_targets(files):
    task = preference_task(files)
    task.save(0, "1")
    task.save(1, "tie")
    task.save(2, "both_bad")

    rows = read_output(files)
    assert [row["edit_id"] for row in rows] == [pair["edit_id"] for pair in task.pairs[:3]]
    assert all(set(row) == FIELDS for row in rows)
    first = rows[0]
    shown_first = task.pairs[0]["outputs"][task.source_at(0, 0)]
    assert first["chosen"] == training_target(shown_first, "explain")
    assert json.loads(first["chosen"])["hint"] == ""  # unused field left empty
    assert parse_output(first["rejected"])["explanation"].startswith("Answer 0")
    assert first["chosen"] != first["rejected"]
    assert first["source_systems"] == ["SOURCE-ONE", "SOURCE-TWO"]
    winner = "SOURCE-TWO" if task.swapped(0) else "SOURCE-ONE"
    assert first["judgement"] == ("first", "second")[first["source_systems"].index(winner)]
    assert WORDING[winner] in first["chosen"]
    assert (first["level"], first["mode"]) == ("B1", "explain")
    for row in rows[1:]:
        assert (row["chosen"], row["rejected"]) == (None, None)
    assert [row["judgement"] for row in rows[1:]] == ["tie", "both_bad"]


def test_prompt_messages_are_the_grounded_inference_prompt(files):
    task = preference_task(files)
    task.save(0, "2")

    row = read_output(files)[0]
    edit = next(e for e in EDITS if e.edit_id == row["edit_id"])
    graph = load_graph(GRAPH_DIR / "concepts.yaml", GRAPH_DIR / "errant_map.yaml")
    assert row["prompt_messages"] == edit_item(edit, "explain", graph, grounded=True)[1]
    assert "Grammar rule (from the grammar graph)" in row["prompt_messages"][1]["content"]


def test_resume_shows_the_saved_choice_at_its_screen_position(files):
    task = preference_task(files)
    task.save(0, "2")
    task.save(1, "tie")

    resumed = preference_task(files, order=("SOURCE-ONE", "SOURCE-TWO"))
    assert resumed.start_index() == 2
    assert [resumed.values(i) for i in range(3)] == [["2"], ["tie"], [None]]
    with pytest.raises(ValueError, match="Choose"):
        resumed.save(2, None)


def test_sources_must_be_two_different_names(files):
    with pytest.raises(ValueError, match="exactly two"):
        PreferenceTask([("D", files / "one.jsonl")], files / "edits.jsonl", GRAPH_DIR, files)
    with pytest.raises(ValueError, match="exactly two"):
        PreferenceTask(
            [("D", files / "one.jsonl"), ("D", files / "two.jsonl")],
            files / "edits.jsonl",
            GRAPH_DIR,
            files,
        )
