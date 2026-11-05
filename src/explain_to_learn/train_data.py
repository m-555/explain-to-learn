"""Supervised fine-tuning (SFT) examples for systems C and D, built from silver data.

The silver step (`scripts/make_silver.py filter`) keeps teacher outputs in
`outputs/silver/kept.jsonl`, one row per edit with its `edit_id`, `mode`, `level`,
the graph's `concept` name and the `explanation` (explain mode) or `hint` (hint
mode). Each row becomes one chat example in TRL's prompt-completion format:

- `prompt`: the messages from `generate.edit_item`, the same function that builds the
  inputs at inference time;
- `completion`: one assistant message with `prompts.build_target(concept,
  explanation, hint)`.

System C gets the ungrounded prompt and system D the prompt with the grammar graph
snippet (none for an error type without a concept, as at inference time). The mode,
level and target of a row are the same for both systems, so C and D differ only in
grounding. The silver step already mixes the modes (a seeded share of edits in hint
mode) and keeps each learner's level.

Train and validation are split by essay: a seeded share of the essays goes to
validation, so all edits of one essay are on the same side.

The examples quote learner sentences, so they are written to `outputs/sft/`, which is
git-ignored.

Usage:
    python -m explain_to_learn.train_data --silver outputs/silver/kept.jsonl \
        --edits data/processed/edits.jsonl --out outputs/sft
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from explain_to_learn import prompts
from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.generate import edit_item
from explain_to_learn.io import open_graph

SYSTEMS = ("c", "d")
SEED = 42
VALIDATION_SHARE = 0.05


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    with Path(path).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dictionaries to a JSON Lines file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def validation_essays(
    essay_keys: Iterable[str], share: float = VALIDATION_SHARE, seed: int = SEED
) -> set[str]:
    """Pick a seeded share of the essays (at least one) for validation."""
    essays = sorted(set(essay_keys))
    random.Random(seed).shuffle(essays)
    return set(essays[: max(1, round(len(essays) * share))])


def make_example(row: dict[str, Any], edit: Edit, system: str, grammar_graph: Any) -> dict:
    """One prompt-completion example for system C (ungrounded) or D (grounded)."""
    record, messages = edit_item(edit, row["mode"], grammar_graph, grounded=system == "d")
    target = prompts.build_target(row["concept"], row["explanation"], row["hint"])
    return {
        "edit_id": edit.edit_id,
        "essay_key": edit.essay_key,
        "system": system,
        "mode": record["mode"],
        "level": record["level"],
        "grounded": record["grounded"],
        "prompt": messages,
        "completion": [{"role": "assistant", "content": target}],
    }


def build_examples(
    rows: list[dict[str, Any]],
    edits: Mapping[str, Edit],
    system: str,
    grammar_graph: Any,
    share: float = VALIDATION_SHARE,
    seed: int = SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Examples for one system, sorted by edit ID and split by essay."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose from {SYSTEMS}")
    missing = [row["edit_id"] for row in rows if row["edit_id"] not in edits]
    if missing:
        raise ValueError(
            f"{len(missing)} silver edits are not in the edit table, e.g. {missing[0]}"
        )
    held_out = validation_essays((edits[row["edit_id"]].essay_key for row in rows), share, seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for row in sorted(rows, key=lambda r: r["edit_id"]):
        example = make_example(row, edits[row["edit_id"]], system, grammar_graph)
        splits["validation" if example["essay_key"] in held_out else "train"].append(example)
    return splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build SFT examples for systems C and D.")
    parser.add_argument("--silver", type=Path, default=Path("outputs/silver/kept.jsonl"))
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rows = read_rows(args.silver)
    edits = {edit.edit_id: edit for edit in read_jsonl(args.edits)}
    grammar_graph = open_graph(args.graph)  # needs the `graph` extra
    for system in SYSTEMS:
        splits = build_examples(
            rows, edits, system, grammar_graph, args.validation_share, args.seed
        )
        for split, examples in splits.items():
            write_rows(examples, args.out / f"system_{system}" / f"{split}.jsonl")
            grounded = sum(example["grounded"] for example in examples)
            print(f"system {system.upper()} {split}: {len(examples)} examples, {grounded} grounded")


if __name__ == "__main__":
    main()
