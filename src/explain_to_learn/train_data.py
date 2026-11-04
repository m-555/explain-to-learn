"""Supervised fine-tuning (SFT) examples for systems C and D, built from silver rows.

A silver row is one edit from the edit table (the fields of `edits.Edit`) plus the
teacher model's `concept`, `explanation` and `hint`, as kept in
`data/processed/silver_kept.jsonl`. Each row becomes one chat example in TRL's
prompt-completion format:

- `prompt`: the messages from `prompts.build_messages(edit, level, mode, snippet)`;
- `completion`: one assistant message with `prompts.build_target(concept, explanation, hint)`.

System C gets no snippet. System D gets the grammar graph snippet of the concept that
the graph maps the edit's error type to, exactly as at inference time. The mode, level
and target of a row are the same for both systems, so C and D differ only in grounding.

- Mode: a seeded shuffle puts half of the edits in explain mode and half in hint mode.
- Level: the row's `level` if the silver data has one, otherwise the essay's CEFR level
  mapped to the nearest level the graph is worded for (A1 -> A2; C1, C2, native -> B2).
- Split: a seeded share of the essays goes to validation, so all edits of one essay end
  up on the same side.

The examples contain learner sentences, so they are written under `data/processed/`,
which is git-ignored.

Usage:
    python -m explain_to_learn.train_data --silver data/processed/silver_kept.jsonl \
        --out data/processed/sft
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from explain_to_learn import graph, prompts
from explain_to_learn.edits import Edit

GRAPH_LEVELS = ("A2", "B1", "B2")
MODES = ("explain", "hint")
SYSTEMS = ("c", "d")
EDIT_FIELDS = tuple(field.name for field in dataclasses.fields(Edit))
SEED = 42
VALIDATION_SHARE = 0.05


def target_level(cefr_level: str) -> str:
    """Map a CEFR level to the nearest level the grammar graph is worded for."""
    if cefr_level in GRAPH_LEVELS:
        return cefr_level
    return "A2" if cefr_level == "A1" else "B2"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    with Path(path).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dictionaries to a JSON Lines file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def edit_from_row(row: dict[str, Any]) -> Edit:
    """The edit-table fields of a silver row as an `Edit`."""
    return Edit(**{name: row[name] for name in EDIT_FIELDS})


def assign_modes(edit_ids: Iterable[str], seed: int = SEED) -> dict[str, str]:
    """Put a seeded half of the edits in explain mode and the rest in hint mode."""
    ids = sorted(set(edit_ids))
    random.Random(seed).shuffle(ids)
    half = (len(ids) + 1) // 2
    return {edit_id: MODES[0] if i < half else MODES[1] for i, edit_id in enumerate(ids)}


def validation_essays(
    essay_keys: Iterable[str], share: float = VALIDATION_SHARE, seed: int = SEED
) -> set[str]:
    """Pick a seeded share of the essays (at least one) for validation."""
    essays = sorted(set(essay_keys))
    random.Random(seed).shuffle(essays)
    return set(essays[: max(1, round(len(essays) * share))])


def make_example(
    row: dict[str, Any], system: str, mode: str, grammar_graph: Any = None
) -> dict[str, Any]:
    """One prompt-completion example for system C (no snippet) or D (graph snippet)."""
    edit = edit_from_row(row)
    level = row.get("level") or target_level(edit.cefr_level)
    snippet = None
    if system == "d":
        concept_id = grammar_graph.concept_for(edit.error_type)
        snippet = grammar_graph.snippet(concept_id, level) if concept_id else None
    target = prompts.build_target(row["concept"], row["explanation"], row["hint"])
    return {
        "edit_id": edit.edit_id,
        "essay_key": edit.essay_key,
        "system": system,
        "mode": mode,
        "level": level,
        "grounded": snippet is not None,
        "prompt": prompts.build_messages(edit, level, mode, snippet),
        "completion": [{"role": "assistant", "content": target}],
    }


def build_examples(
    rows: list[dict[str, Any]],
    system: str,
    grammar_graph: Any = None,
    share: float = VALIDATION_SHARE,
    seed: int = SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Examples for one system, split into `train` and `validation` by essay."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose from {SYSTEMS}")
    if system == "d" and grammar_graph is None:
        raise ValueError("System D needs the grammar graph")
    modes = assign_modes((row["edit_id"] for row in rows), seed)
    held_out = validation_essays((edit_from_row(row).essay_key for row in rows), share, seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for row in sorted(rows, key=lambda r: r["edit_id"]):
        example = make_example(row, system, modes[row["edit_id"]], grammar_graph)
        splits["validation" if example["essay_key"] in held_out else "train"].append(example)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT examples for systems C and D.")
    parser.add_argument("--silver", type=Path, default=Path("data/processed/silver_kept.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = read_jsonl(args.silver)
    grammar_graph = graph.load_graph()
    for system in SYSTEMS:
        splits = build_examples(rows, system, grammar_graph, args.validation_share, args.seed)
        for split, examples in splits.items():
            path = args.out / f"system_{system}" / f"{split}.jsonl"
            write_jsonl(examples, path)
            grounded = sum(example["grounded"] for example in examples)
            print(f"system {system.upper()} {split}: {len(examples)} examples, {grounded} grounded")


if __name__ == "__main__":
    main()
