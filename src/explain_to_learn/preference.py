"""Preference pairs for DPO, from the teachers' pairwise judgements.

The annotation app writes `outputs/annotations/preferences.jsonl`, one judgement per line:

    {"edit_id", "level", "mode", "prompt_messages", "chosen", "rejected", "source_systems"}

`prompt_messages` are the chat messages the systems were given (`generate.edit_item`),
`chosen` and `rejected` the preferred and the other output, and `source_systems` the
systems that wrote them. A tie or "both bad" has `chosen` and `rejected` set to null and
a `judgement` field. Neither says which output is better, so both are skipped.

Each remaining pair becomes one example in TRL's conversational preference format:

    {"prompt": [...messages], "chosen": [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}

An output may be the reply text or the parsed JSON object (`concept`, `explanation`,
`hint`); a parsed object is written back as the training target with
`prompts.build_target`, the same form as the SFT targets.

Train and validation are split by essay, as for SFT, so that pairs of the same essay are
on the same side. The examples quote learner sentences, so they go to the git-ignored
`outputs/dpo/`.

Usage:
    python -m explain_to_learn.preference \
        --preferences outputs/annotations/preferences.jsonl --out outputs/dpo
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from explain_to_learn.generate import OUTPUT_FIELDS
from explain_to_learn.io import read_rows, write_rows
from explain_to_learn.prompts import build_target
from explain_to_learn.train_data import validation_essays

SEED = 42
# Preference data is small, so a larger share than for SFT keeps the validation set usable.
VALIDATION_SHARE = 0.1
GROUNDING_HEADER = "Grammar rule (from the grammar graph):"  # see prompts.format_snippet


def essay_key(edit_id: str) -> str:
    """The essay part of an edit ID: `wi-1-352420-3-0` -> `wi-1-352420`."""
    return edit_id.rsplit("-", 2)[0]


def assistant_message(output: Any) -> list[dict[str, str]]:
    """An output as a one-message assistant turn.

    Accepts the reply text, the parsed JSON object, or the assistant messages themselves.
    """
    if isinstance(output, str):
        return [{"role": "assistant", "content": output}]
    if isinstance(output, dict) and all(key in output for key in OUTPUT_FIELDS):
        target = build_target(output["concept"], output["explanation"], output["hint"])
        return [{"role": "assistant", "content": target}]
    if isinstance(output, list) and output and all(m.get("role") == "assistant" for m in output):
        return output
    raise ValueError(f"cannot read output {output!r}")


def skip_reason(row: dict[str, Any]) -> str | None:
    """Why a judgement gives no training pair, or None if it does."""
    if row.get("chosen") is None or row.get("rejected") is None:
        return row.get("judgement") or "missing output"
    if assistant_message(row["chosen"]) == assistant_message(row["rejected"]):
        return "same output"
    return None


def to_example(row: dict[str, Any]) -> dict[str, Any]:
    """One preference example in TRL's conversational format, with its metadata."""
    return {
        "edit_id": row["edit_id"],
        "essay_key": essay_key(row["edit_id"]),
        "level": row["level"],
        "mode": row["mode"],
        "source_systems": row.get("source_systems"),
        "prompt": row["prompt_messages"],
        "chosen": assistant_message(row["chosen"]),
        "rejected": assistant_message(row["rejected"]),
    }


def build_preferences(
    rows: Iterable[dict[str, Any]], share: float = VALIDATION_SHARE, seed: int = SEED
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Train and validation examples, and counts of the pairs used and skipped."""
    examples, counts = [], Counter()
    for row in rows:
        reason = skip_reason(row)
        if reason is not None:
            counts[f"skipped: {reason}"] += 1
            continue
        examples.append(to_example(row))
        counts["pairs"] += 1
    counts["grounded prompts"] = sum(
        any(GROUNDING_HEADER in m["content"] for m in example["prompt"]) for example in examples
    )
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    if examples:
        held_out = validation_essays((e["essay_key"] for e in examples), share, seed)
        for example in sorted(examples, key=lambda e: e["edit_id"]):
            split = "validation" if example["essay_key"] in held_out else "train"
            splits[split].append(example)
    return splits, counts


def main(argv: list[str] | None = None) -> None:
    """Build the DPO pairs and write `train.jsonl` and `validation.jsonl` to `--out`."""
    parser = argparse.ArgumentParser(description="Build DPO pairs from teacher preferences.")
    parser.add_argument(
        "--preferences", type=Path, default=Path("outputs/annotations/preferences.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/dpo"))
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not args.preferences.is_file():
        raise SystemExit(f"{args.preferences} not found; export the preference annotations first")
    splits, counts = build_preferences(
        read_rows(args.preferences), args.validation_share, args.seed
    )
    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")
    if not counts["pairs"]:
        raise SystemExit("No preference pairs to train on; nothing written")
    for split, examples in splits.items():
        write_rows(examples, args.out / f"{split}.jsonl")
        print(f"{split}: {len(examples)} pairs -> {args.out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
