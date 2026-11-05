"""SFT example building from synthetic silver rows, with a fake grammar graph.

The prompts come from the real `prompts` and `generate` modules, so the tests check
that training inputs are exactly the inputs the systems get at inference time.
"""

import json

import pytest

from explain_to_learn import train_data
from explain_to_learn.edits import Edit, write_jsonl
from explain_to_learn.prompts import build_messages, build_target, format_snippet
from explain_to_learn.train_data import build_examples, read_rows, validation_essays


class FakeGraph:
    """Maps R:VERB:SVA to a concept; every other error type has none."""

    def concept_for(self, error_type):
        return "subject_verb_agreement" if error_type == "R:VERB:SVA" else None

    def snippet(self, concept_id, level):
        return {
            "concept_id": concept_id,
            "name": "Subject-verb agreement",
            "rule": f"Rule worded for {level}.",
            "example": {"wrong": "He go.", "right": "He goes."},
            "prerequisites": [],
        }


def make_edit(essay: str, idx: int, error_type: str, level: str) -> Edit:
    return Edit(
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
        operation="R",
        error_category=error_type.split(":", 1)[1],
    )


def silver_row(edit: Edit, mode: str) -> dict:
    """A kept silver row as written by `silver.filter_silver`."""
    explanation = 'Use "goes", not "go", after "he".' if mode == "explain" else ""
    hint = 'Look at the verb after "he".' if mode == "hint" else ""
    return {
        "edit_id": edit.edit_id,
        "level": edit.cefr_level,
        "mode": mode,
        "concept_id": "subject_verb_agreement",
        "concept": "Subject-verb agreement",
        "explanation": explanation,
        "hint": hint,
        "target": build_target("Subject-verb agreement", explanation, hint),
    }


def silver_data(n_essays: int = 20) -> tuple[list[dict], dict[str, Edit]]:
    """Three edits per essay; the third has an error type without a concept."""
    edits = [
        make_edit(f"E{essay}", idx, "R:VERB:SVA" if idx < 2 else "R:PREP", level)
        for essay, level in zip(range(n_essays), ["A1", "B1", "C2", "B2"] * 5, strict=False)
        for idx in range(3)
    ]
    rows = [silver_row(edit, "hint" if i % 4 == 0 else "explain") for i, edit in enumerate(edits)]
    return rows, {edit.edit_id: edit for edit in edits}


def test_c_and_d_differ_only_in_grounding():
    rows, edits = silver_data()
    system_c = build_examples(rows, edits, "c", FakeGraph())
    system_d = build_examples(rows, edits, "d", FakeGraph())

    for split in ("train", "validation"):
        assert [e["edit_id"] for e in system_c[split]] == [e["edit_id"] for e in system_d[split]]
        for c, d in zip(system_c[split], system_d[split], strict=True):
            edit = edits[c["edit_id"]]
            assert (c["mode"], c["level"], c["completion"]) == (
                d["mode"],
                d["level"],
                d["completion"],
            )
            assert c["prompt"] == build_messages(edit, c["level"], c["mode"], None)
            assert c["prompt"][0] == d["prompt"][0]
            if edit.error_type == "R:VERB:SVA":
                snippet = format_snippet(FakeGraph().snippet("subject_verb_agreement", c["level"]))
                assert d["prompt"][1]["content"] == c["prompt"][1]["content"] + "\n" + snippet
                assert d["grounded"] and not c["grounded"]
            else:
                assert d["prompt"] == c["prompt"] and not d["grounded"]


def test_mode_level_and_target_come_from_the_silver_row():
    rows, edits = silver_data(4)
    examples = build_examples(rows, edits, "d", FakeGraph(), share=0.25)
    by_id = {e["edit_id"]: e for split in examples.values() for e in split}

    for row in rows:
        example = by_id[row["edit_id"]]
        assert example["mode"] == row["mode"]
        assert example["completion"] == [{"role": "assistant", "content": row["target"]}]
    assert {e["level"] for e in by_id.values()} == {"A1", "B1", "C2", "B2"}
    assert {e["mode"] for e in by_id.values()} == {"explain", "hint"}


def test_split_is_by_essay_and_reproducible():
    rows, edits = silver_data()
    examples = build_examples(rows, edits, "c", FakeGraph(), share=0.25, seed=3)
    train_essays = {e["essay_key"] for e in examples["train"]}
    validation = {e["essay_key"] for e in examples["validation"]}

    assert not train_essays & validation
    assert len(validation) == 5
    assert len(examples["train"]) + len(examples["validation"]) == len(rows)
    assert validation == validation_essays(train_essays | validation, 0.25, 3)
    assert build_examples(list(reversed(rows)), edits, "c", FakeGraph(), 0.25, 3) == examples
    assert build_examples(rows, edits, "c", FakeGraph(), 0.25, 4) != examples


def test_unknown_edits_and_systems_are_errors():
    rows, edits = silver_data(2)

    with pytest.raises(ValueError, match="not in the edit table"):
        build_examples(rows, {}, "c", FakeGraph())
    with pytest.raises(ValueError, match="Unknown system"):
        build_examples(rows, edits, "a", FakeGraph())


def test_main_writes_both_systems(tmp_path, monkeypatch):
    graph = pytest.importorskip("explain_to_learn.graph")
    rows, edits = silver_data()
    silver, edits_path, out = tmp_path / "kept.jsonl", tmp_path / "edits.jsonl", tmp_path / "sft"
    silver.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_jsonl(edits.values(), edits_path)
    monkeypatch.setattr(graph, "load_graph", lambda *paths: FakeGraph())

    train_data.main(["--silver", str(silver), "--edits", str(edits_path), "--out", str(out)])

    for system in ("c", "d"):
        train = read_rows(out / f"system_{system}" / "train.jsonl")
        validation = read_rows(out / f"system_{system}" / "validation.jsonl")
        assert len(train) + len(validation) == 60
        assert {e["system"] for e in train} == {system}
