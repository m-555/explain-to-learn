"""Essay-level splits and the stratified gold-set sample.

Train and dev come from the official BEA-2019 split, which is already by essay.
`essay_splits` checks that no essay appears in two splits, and `write_split_files`
writes one file of essay keys per split (`train_essays.txt`, `dev_essays.txt`, and
`test_essays.txt` for the FCE test set). An essay key is `{corpus}-{essay_id}`, the
prefix of every edit ID of that essay. Only essays with at least one edit are listed.

`sample_gold` draws the gold evaluation set from W&I+LOCNESS dev edits, stratified
by error category and CEFR level.

Usage:
    python -m explain_to_learn.splits --edits data/processed/edits.jsonl --out data/splits
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from explain_to_learn.edits import Edit, read_jsonl

GOLD_CORPORA = ("wi", "locness")
GOLD_SEED = 2019


def essay_splits(edits: Iterable[Edit]) -> dict[str, set[str]]:
    """Group essay keys by split; raise `ValueError` if an essay is in two splits."""
    split_of: dict[str, str] = {}
    for edit in edits:
        known = split_of.setdefault(edit.essay_key, edit.split)
        if known != edit.split:
            raise ValueError(f"Essay {edit.essay_key} is in both {known} and {edit.split}")
    splits: dict[str, set[str]] = defaultdict(set)
    for essay, split in split_of.items():
        splits[split].add(essay)
    return dict(splits)


def write_ids(ids: Iterable[str], path: str | Path) -> None:
    """Write IDs to a text file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def write_split_files(edits: Iterable[Edit], out_dir: str | Path) -> dict[str, set[str]]:
    """Write `{split}_essays.txt` (sorted essay keys) for every split."""
    splits = essay_splits(edits)
    for split, essays in splits.items():
        write_ids(sorted(essays), Path(out_dir) / f"{split}_essays.txt")
    return splits


def allocate(sizes: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """Share `n` draws between strata in proportion to their sizes.

    Each stratum first gets the whole part of its proportional share; the draws left
    over go to the strata with the largest fractional parts (ties broken by stratum
    name). A stratum never gets more draws than it has edits, so a small stratum gets
    one draw only when its share is large enough, and none otherwise.
    """
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    shares = {key: n * size / total for key, size in sizes.items()}
    counts = {key: int(share) for key, share in shares.items()}
    left = n - sum(counts.values())
    by_remainder = sorted(sizes, key=lambda key: (counts[key] - shares[key], key))
    for key in by_remainder[:left]:
        counts[key] += 1
    return counts


def sample_gold(edits: Iterable[Edit], n: int = 300, seed: int = GOLD_SEED) -> list[Edit]:
    """Sample `n` W&I+LOCNESS dev edits, stratified by error category and CEFR level.

    The result is deterministic for a given edit table and seed, and sorted by edit ID.
    """
    strata: dict[tuple[str, str], list[Edit]] = defaultdict(list)
    for edit in edits:
        if edit.split == "dev" and edit.corpus in GOLD_CORPORA:
            strata[(edit.error_category, edit.cefr_level)].append(edit)

    counts = allocate({key: len(group) for key, group in strata.items()}, n)
    rng = random.Random(seed)
    sample = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda e: e.edit_id)
        sample.extend(rng.sample(group, counts[key]))
    return sorted(sample, key=lambda e: e.edit_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write essay split files and gold edit IDs.")
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--n-gold", type=int, default=300)
    args = parser.parse_args()

    edits = read_jsonl(args.edits)
    for split, essays in sorted(write_split_files(edits, args.out).items()):
        print(f"{split}: {len(essays)} essays")
    gold = sample_gold(edits, n=args.n_gold)
    write_ids([e.edit_id for e in gold], args.out / "gold_edit_ids.txt")
    print(f"gold: {len(gold)} edits")


if __name__ == "__main__":
    main()
