"""SFT example building with fake prompt builders and a fake grammar graph."""

import dataclasses
import json
import sys

import pytest

from explain_to_learn import graph, prompts, train_data
from explain_to_learn.edits import Edit
from explain_to_learn.train_data import (
    assign_modes,
    build_examples,
    edit_from_row,
    read_jsonl,
    target_level,
    validation_essays,
)


def fake_build_messages(edit, level, mode, snippet):
    """Chat messages that show every input, so tests can see what was passed."""
    user = f"{edit.original} | {edit.original_text} -> {edit.corrected_text} | {level} | {mode}"
    if snippet is not None:
        user += f" | {snippet}"
    return [{"role": "system", "content": "Explain the edit."}, {"role": "user", "content": user}]


def fake_build_target(concept, explanation, hint):
    return json.dumps({"concept": concept, "explanation": explanation, "hint": hint})


class FakeGraph:
    """Maps one error type to a concept; every other type is unmapped."""

    def concept_for(self, error_type):
        return {"R:VERB:SVA": "subject_verb_agreement"}.get(error_type)

    def snippet(self, concept_id, level):
        return f"[{concept_id} rule for {level}]"


@pytest.fixture(autouse=True)
def fake_prompts(monkeypatch):
    monkeypatch.setattr(prompts, "build_messages", fake_build_messages, raising=False)
    monkeypatch.setattr(prompts, "build_target", fake_build_target, raising=False)


def silver_row(essay: str, idx: int, error_type: str = "R:VERB:SVA", level: str = "B1") -> dict:
    """A synthetic silver row: an edit plus a teacher explanation."""
    edit = Edit(
        edit_id=f"wi-{essay}-0-{idx}",
        corpus="wi",
        essay_id=essay,
        split="train",
        cefr_level=level,
        sent_idx=0,
        original="He go home .",
        corrected="He goes home .",
        start=1,
        end=2,
        original_text="go",
        corrected_text="goes",
        error_type=error_type,
        operation=error_type.split(":", 1)[0],
        error_category=error_type.split(":", 1)[1],
    )
    return dataclasses.asdict(edit) | {
        "concept": "subject_verb_agreement",
        "explanation": "With he, the verb takes -s.",
        "hint": "Look at the verb after he.",
    }


def silver_rows() -> list[dict]:
    """Twenty essays with three edits each; one edit per essay has an unmapped type."""
    return [
        silver_row(f"E{essay}", idx, "R:VERB:SVA" if idx < 2 else "R:PREP", level)
        for essay, level in zip(range(20), ["A1", "B1", "C2", "B2"] * 5, strict=True)
        for idx in range(3)
    ]


def test_target_level_maps_to_graph_levels():
    levels = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]

    assert [target_level(level) for level in levels] == ["A2", "A2", "B1", "B2", "B2", "B2", "B2"]


def test_c_and_d_differ_only_in_grounding():
    rows = silver_rows()
    system_c = build_examples(rows, "c")
    system_d = build_examples(rows, "d", FakeGraph())

    rows_by_id = {row["edit_id"]: row for row in rows}
    for split in ("train", "validation"):
        assert [e["edit_id"] for e in system_c[split]] == [e["edit_id"] for e in system_d[split]]
        for c, d in zip(system_c[split], system_d[split], strict=True):
            assert (c["mode"], c["level"], c["completion"]) == (
                d["mode"],
                d["level"],
                d["completion"],
            )
            edit = edit_from_row(rows_by_id[c["edit_id"]])
            snippet = None
            if edit.error_type == "R:VERB:SVA":
                snippet = f"[subject_verb_agreement rule for {c['level']}]"
            assert c["prompt"] == fake_build_messages(edit, c["level"], c["mode"], None)
            assert d["prompt"] == fake_build_messages(edit, d["level"], d["mode"], snippet)
            assert not c["grounded"] and d["grounded"] == (snippet is not None)
    assert any(e["grounded"] for e in system_d["train"])
    assert not all(e["grounded"] for e in system_d["train"])


def test_completion_is_the_target_json():
    example = build_examples([silver_row("E1", 0)], "c")["validation"][0]

    assert example["completion"] == [
        {
            "role": "assistant",
            "content": fake_build_target(
                "subject_verb_agreement",
                "With he, the verb takes -s.",
                "Look at the verb after he.",
            ),
        }
    ]


def test_modes_are_mixed_and_reproducible():
    ids = [f"wi-E{i}-0-0" for i in range(41)]
    modes = assign_modes(ids, seed=1)

    assert sorted(set(modes.values())) == ["explain", "hint"]
    assert abs(list(modes.values()).count("explain") - list(modes.values()).count("hint")) <= 1
    assert assign_modes(reversed(ids), seed=1) == modes
    assert assign_modes(ids, seed=2) != modes


def test_levels_follow_the_essay_or_the_row():
    rows = [silver_row("E1", 0, level="C1"), silver_row("E2", 0, level="C1") | {"level": "A2"}]
    examples = build_examples(rows, "c", share=0.5)
    levels = {e["edit_id"]: e["level"] for split in examples.values() for e in split}

    assert levels == {"wi-E1-0-0": "B2", "wi-E2-0-0": "A2"}


def test_split_is_by_essay_and_reproducible():
    rows = silver_rows()
    examples = build_examples(rows, "c", share=0.25, seed=3)
    train_essays = {e["essay_key"] for e in examples["train"]}
    validation = {e["essay_key"] for e in examples["validation"]}

    assert not train_essays & validation
    assert len(validation) == 5
    assert len(examples["train"]) + len(examples["validation"]) == len(rows)
    assert validation == validation_essays(train_essays | validation, 0.25, 3)
    assert build_examples(list(reversed(rows)), "c", share=0.25, seed=3) == examples


def test_system_d_needs_the_graph():
    with pytest.raises(ValueError, match="graph"):
        build_examples([silver_row("E1", 0)], "d")


def test_main_writes_both_systems(tmp_path, monkeypatch):
    silver = tmp_path / "silver.jsonl"
    train_data.write_jsonl(silver_rows(), silver)
    out = tmp_path / "sft"
    monkeypatch.setattr(graph, "load_graph", FakeGraph, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_data", "--silver", str(silver), "--out", str(out)])

    train_data.main()

    for system in ("c", "d"):
        train = read_jsonl(out / f"system_{system}" / "train.jsonl")
        validation = read_jsonl(out / f"system_{system}" / "validation.jsonl")
        assert len(train) + len(validation) == 60
        assert {e["system"] for e in train} == {system}
