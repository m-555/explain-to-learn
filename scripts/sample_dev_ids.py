"""Sample the dev edits for the first runs of the prompted systems (runbook step 4).

The first runs of systems A and B, and the comparison of the LoRA variants, use W&I+LOCNESS
dev edits that are not in the gold set, so that prompts, parsing and metrics are checked
without looking at the gold edits. This draws a seeded sample of them and writes one edit
ID per line, sorted. The same edit table, gold IDs, size and seed give the same sample.

Usage:
    python scripts/sample_dev_ids.py --n 500
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.splits import GOLD_CORPORA, write_ids


def sample_dev_ids(edits: list[Edit], gold_ids: set[str], n: int, seed: int = 0) -> list[str]:
    """`n` dev edit IDs of W&I+LOCNESS outside the gold set (all of them if fewer), sorted."""
    pool = sorted(
        edit.edit_id
        for edit in edits
        if edit.split == "dev" and edit.corpus in GOLD_CORPORA and edit.edit_id not in gold_ids
    )
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample dev edits outside the gold set.")
    parser.add_argument("--n", type=int, default=500, help="number of edits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/dev_ids.txt"))
    args = parser.parse_args(argv)

    gold = set(args.gold_ids.read_text(encoding="utf-8").split())
    ids = sample_dev_ids(read_jsonl(args.edits), gold, args.n, args.seed)
    write_ids(ids, args.out)
    print(f"Wrote {len(ids)} dev edit IDs to {args.out}")


if __name__ == "__main__":
    main()
