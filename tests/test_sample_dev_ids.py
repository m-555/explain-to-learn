"""The dev sample of runbook step 4 (scripts/sample_dev_ids.py). Sentences are invented."""

import importlib.util
import random
from pathlib import Path

from explain_to_learn.edits import Edit, write_jsonl

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "sample_dev_ids", ROOT / "scripts" / "sample_dev_ids.py"
)
sample_dev_ids = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample_dev_ids)


def make_edit(corpus: str, essay: str, index: int, split: str) -> Edit:
    return Edit(
        edit_id=f"{corpus}-{essay}-0-{index}",
        corpus=corpus,
        essay_id=essay,
        split=split,
        cefr_level="B1",
        sent_idx=0,
        original="She like tea .",
        corrected="She likes tea .",
        start=1,
        end=2,
        original_text="like",
        corrected_text="likes",
        error_type="R:VERB:SVA",
        operation="R",
        error_category="VERB:SVA",
    )


EDITS = (
    [make_edit("wi", "1-1", i, "dev") for i in range(20)]
    + [make_edit("locness", "7-1", i, "dev") for i in range(5)]
    + [make_edit("wi", "1-2", i, "train") for i in range(5)]
    + [make_edit("fce", "TR1-1", i, "dev") for i in range(5)]
)
GOLD = {"wi-1-1-0-0", "wi-1-1-0-1", "locness-7-1-0-0"}


def test_sample_is_dev_wi_and_locness_outside_the_gold_set():
    ids = sample_dev_ids.sample_dev_ids(EDITS, GOLD, n=10)

    assert len(ids) == 10 and ids == sorted(ids)
    assert not set(ids) & GOLD
    assert all(i.startswith(("wi-1-1-", "locness-")) for i in ids)
    assert ids == sample_dev_ids.sample_dev_ids(EDITS, GOLD, n=10)  # seeded


def test_sample_matches_the_earlier_inline_runbook_code():
    """The runbook drew the sample with this code before the script existed."""
    pool = sorted(
        e.edit_id
        for e in EDITS
        if e.split == "dev" and e.corpus in ("wi", "locness") and e.edit_id not in GOLD
    )
    assert sample_dev_ids.sample_dev_ids(EDITS, GOLD, n=10) == sorted(
        random.Random(0).sample(pool, 10)
    )


def test_a_small_pool_gives_all_its_edits(tmp_path):
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    (tmp_path / "gold.txt").write_text("".join(f"{i}\n" for i in sorted(GOLD)), encoding="utf-8")
    out = tmp_path / "dev_ids.txt"

    sample_dev_ids.main(
        ["--n", "500", "--edits", str(tmp_path / "edits.jsonl")]
        + ["--gold-ids", str(tmp_path / "gold.txt"), "--out", str(out)]
    )

    assert len(out.read_text(encoding="utf-8").split()) == 25 - len(GOLD)
