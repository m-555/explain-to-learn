import dataclasses
from collections import Counter

import pytest

from explain_to_learn.edits import Edit, build_edits
from explain_to_learn.splits import allocate, essay_splits, sample_gold, write_split_files


def make_edit(essay_id: str, split: str, category: str, level: str, idx: int = 0) -> Edit:
    """A minimal synthetic edit row."""
    return Edit(
        edit_id=f"wi-{essay_id}-0-{idx}",
        corpus="wi",
        essay_id=essay_id,
        split=split,
        cefr_level=level,
        sent_idx=0,
        original="He go home .",
        corrected="He goes home .",
        start=1,
        end=2,
        original_text="go",
        corrected_text="goes",
        error_type=f"R:{category}",
        operation="R",
        error_category=category,
    )


def test_essay_splits_from_official_split(raw_dir):
    splits = essay_splits(build_edits(raw_dir))

    assert set(splits) == {"train", "dev", "test"}
    assert not splits["train"] & splits["dev"]
    assert not splits["train"] & splits["test"]
    assert not splits["dev"] & splits["test"]


def test_essay_splits_detects_leakage():
    edits = [
        make_edit("E1", "train", "VERB:SVA", "A2"),
        make_edit("E1", "dev", "VERB:SVA", "A2", 1),
    ]

    with pytest.raises(ValueError, match="wi-E1"):
        essay_splits(edits)


def test_write_split_files(raw_dir, tmp_path):
    write_split_files(build_edits(raw_dir), tmp_path / "splits")
    lines = (tmp_path / "splits" / "dev_essays.txt").read_text().splitlines()

    assert lines == sorted(lines)
    assert "wi-3-ID-1" in lines
    assert "wi-0-ID-1" not in lines


def test_allocate_is_proportional_and_capped():
    sizes = {("A", "B1"): 60, ("B", "B1"): 30, ("C", "B1"): 9, ("D", "B1"): 1}

    assert allocate(sizes, 10) == {("A", "B1"): 6, ("B", "B1"): 3, ("C", "B1"): 1, ("D", "B1"): 0}
    assert allocate(sizes, 1000) == sizes
    assert sum(allocate(sizes, 37).values()) == 37


def gold_pool() -> list[Edit]:
    """Dev edits in three strata of different sizes, plus train and FCE edits."""
    edits = []
    for category, level, count in [("VERB:SVA", "A2", 40), ("DET", "B1", 15), ("SPELL", "C1", 5)]:
        edits += [make_edit(f"{category}-{i}", "dev", category, level) for i in range(count)]
    edits += [make_edit(f"T{i}", "train", "DET", "B1") for i in range(50)]
    edits += [
        dataclasses.replace(make_edit(f"F{i}", "dev", "DET", "B2"), corpus="fce") for i in range(50)
    ]
    return edits


def test_sample_gold_is_deterministic_and_stratified():
    first = sample_gold(gold_pool(), n=12, seed=7)
    second = sample_gold(list(reversed(gold_pool())), n=12, seed=7)
    other_seed = sample_gold(gold_pool(), n=12, seed=8)

    assert [e.edit_id for e in first] == [e.edit_id for e in second]
    assert [e.edit_id for e in first] != [e.edit_id for e in other_seed]
    assert all(e.split == "dev" and e.corpus == "wi" for e in first)
    assert Counter(e.error_category for e in first) == {"VERB:SVA": 8, "DET": 3, "SPELL": 1}


def test_sample_gold_returns_all_when_pool_is_small():
    assert len(sample_gold(gold_pool(), n=500)) == 60
