"""Make the blind rating sheet: gold edits x systems, shuffled, system names hidden.

Samples `--n-edits` edits (default 100) from the gold edit IDs and takes each system's
output for them from a `scripts/run_system.py` run file (`outputs/<system>/<run>.jsonl`).
All items are shuffled with a seed and numbered in that order. The sheet shows only
what a rater needs (`eval/rubric.md`): the level, the mode, the sentence, the edit and
the feedback the learner would see (the explanation, or the hint in hint mode). An
output that failed to parse is shown as empty feedback and rated as such.

A separate key maps each item ID to its system, edit ID, error category, level and
mode; keep it closed until rating is done. A random subset of `--n-second` items
(default 50) is written as a second sheet for the second rater.

All outputs quote learner text and go to `outputs/human_eval/` (git-ignored). Existing
sheets are not overwritten unless `--force` is given, so ratings in progress are safe.

Usage:
    python scripts/make_rating_sheet.py \
        --system A=outputs/A/gold-explain.jsonl --system B=outputs/B/gold-explain.jsonl \
        --system C=outputs/C/gold-explain.jsonl --system D=outputs/D/gold-explain.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.human_eval import KEY_COLUMNS, RATING_COLUMNS, SHEET_COLUMNS, write_csv
from explain_to_learn.metrics import shown_text
from explain_to_learn.prompts import detokenize, prompt_level

SEED = 2025


def read_outputs(path: str | Path) -> dict[str, dict[str, Any]]:
    """One system's generation records by edit ID."""
    with Path(path).open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return {row["edit_id"]: row for row in rows}


def build_items(
    edit_ids: list[str], edits: dict[str, Edit], outputs: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One item per edit and system, with the sheet fields and the key fields."""
    items = []
    for edit_id in edit_ids:
        edit = edits[edit_id]
        for system, rows in sorted(outputs.items()):
            if edit_id not in rows:
                raise ValueError(f"System {system} has no output for {edit_id}")
            row = rows[edit_id]
            mode = row.get("mode", "explain")
            items.append(
                {
                    "system": system,
                    "edit_id": edit_id,
                    "error_category": edit.error_category,
                    "level": row.get("level") or prompt_level(edit.cefr_level),
                    "mode": mode,
                    "sentence": detokenize(edit.original),
                    "corrected": detokenize(edit.corrected),
                    "edit": f"{detokenize(edit.original_text) or '-'} -> "
                    f"{detokenize(edit.corrected_text) or '-'}",
                    "feedback": shown_text(row.get("parsed"), mode),
                }
            )
    return items


def make_sheets(
    gold_ids: list[str],
    edits: dict[str, Edit],
    outputs: dict[str, dict[str, dict[str, Any]]],
    n_edits: int,
    n_second: int,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the rating sheet, the second rater's sheet and the key.

    The same inputs and seed always give the same sheets. Item IDs follow the shuffled
    order, so they carry no information about the system.
    """
    rng = random.Random(seed)
    edit_ids = sorted(rng.sample(sorted(gold_ids), n_edits))
    items = build_items(edit_ids, edits, outputs)
    rng.shuffle(items)
    width = len(str(len(items)))
    for number, item in enumerate(items, start=1):
        item["item_id"] = f"item-{number:0{width}d}"
    sheet = [{column: item[column] for column in SHEET_COLUMNS} for item in items]
    second_ids = set(rng.sample([row["item_id"] for row in sheet], n_second))
    second = [row for row in sheet if row["item_id"] in second_ids]
    key = [{column: item[column] for column in KEY_COLUMNS} for item in items]
    return sheet, second, key


def parse_system(value: str) -> tuple[str, Path]:
    """Read a `NAME=PATH` argument."""
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the blind rating sheets and the key.")
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    parser.add_argument("--n-edits", type=int, default=100)
    parser.add_argument("--n-second", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/human_eval"))
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = parser.parse_args()

    paths = {
        "sheet": args.out_dir / "rating_sheet.csv",
        "second": args.out_dir / "rating_sheet_second_rater.csv",
        "key": args.out_dir / "key.csv",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.force:
        parser.error(f"{', '.join(existing)} already exist; use --force to overwrite")

    gold_ids = args.gold_ids.read_text(encoding="utf-8").split()
    edits = {edit.edit_id: edit for edit in read_jsonl(args.edits)}
    outputs = {name: read_outputs(path) for name, path in args.system}
    sheet, second, key = make_sheets(
        gold_ids, edits, outputs, args.n_edits, args.n_second, args.seed
    )
    columns = (*SHEET_COLUMNS, *RATING_COLUMNS)
    write_csv(sheet, paths["sheet"], columns)
    write_csv(second, paths["second"], columns)
    write_csv(key, paths["key"], KEY_COLUMNS)
    print(f"{len(sheet)} items -> {paths['sheet']}")
    print(f"{len(second)} items for the second rater -> {paths['second']}")
    print(f"Key (keep closed until rating is done) -> {paths['key']}")


if __name__ == "__main__":
    main()
