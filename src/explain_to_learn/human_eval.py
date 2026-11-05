"""Human evaluation: rubric scales, rating files and summaries per system.

The rubric (`eval/rubric.md`) has four dimensions on a no / partly / yes scale (stored
as 0 / 1 / 2) and a yes / no flag for invented rules. `scripts/make_rating_sheet.py`
writes the blind rating sheet and a separate key, `scripts/agreement.py` compares two
sets of ratings, and `scripts/llm_judge.py` rates the sheet with a model. This module
holds what they share, and the summary step:

    python -m explain_to_learn.human_eval --ratings outputs/human_eval/ratings_rater1.csv \
        --key outputs/human_eval/key.csv --out-dir eval/results --name human_eval_rater1

The summary joins the ratings with the key (item ID -> system, edit ID, error category,
level) and writes mean scores per system, per error category and per level. It writes
only counts and means, never learner text, so its output can be committed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from explain_to_learn.io import read_csv, write_csv

DIMENSIONS = ("rule_correct", "error_located", "helps_fix", "fits_level")
FLAG = "invented_rule"
SCALE = {"no": 0, "partly": 1, "yes": 2}
FLAG_VALUES = {"no": 0, "yes": 1}
RATING_COLUMNS = (*DIMENSIONS, FLAG, "comment")
SHEET_COLUMNS = ("item_id", "level", "mode", "sentence", "corrected", "edit", "feedback")
KEY_COLUMNS = ("item_id", "system", "edit_id", "error_category", "level", "mode")
GROUPINGS = {
    "system": ("system",),
    "category": ("error_category", "system"),
    "level": ("level", "system"),
}


def parse_rating(value: str | None, scale: dict[str, int]) -> int | None:
    """Read one rating as a number: a word of the scale or its number; empty is None."""
    text = (value or "").strip().lower()
    if not text:
        return None
    if text in scale:
        return scale[text]
    if text.isdigit() and int(text) in scale.values():
        return int(text)
    raise ValueError(f"{value!r} is not one of {', '.join(scale)}")


def read_ratings(path: str | Path) -> dict[str, dict[str, int | None]]:
    """Ratings by item ID: the four dimensions (0-2) and the flag (0-1), None if empty."""
    ratings = {}
    for row in read_csv(path):
        scores: dict[str, int | None] = {
            dim: parse_rating(row.get(dim), SCALE) for dim in DIMENSIONS
        }
        scores[FLAG] = parse_rating(row.get(FLAG), FLAG_VALUES)
        ratings[row["item_id"]] = scores
    return ratings


def join_key(
    ratings: dict[str, dict[str, int | None]], key_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Add the system, edit ID, error category and level from the key to each rated item."""
    key = {row["item_id"]: row for row in key_rows}
    missing = sorted(set(ratings) - set(key))
    if missing:
        raise ValueError(f"{len(missing)} rated items are not in the key, e.g. {missing[0]}")
    return [key[item_id] | scores for item_id, scores in sorted(ratings.items())]


def summarise(rows: list[dict[str, Any]], by: Sequence[str]) -> list[list[Any]]:
    """Per group: item count, mean of each dimension and share of invented rules.

    Empty ratings are left out of each mean, so a mean can rest on fewer items than the
    count. A mean is None when no item of the group has that rating.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in by)].append(row)
    table = []
    for group in sorted(groups):
        items = groups[group]
        cells: list[Any] = [*group, len(items)]
        for column in (*DIMENSIONS, FLAG):
            values = [item[column] for item in items if item[column] is not None]
            cells.append(mean(values) if values else None)
        table.append(cells)
    return table


def header(by: Sequence[str]) -> list[str]:
    """Column names of a summary table."""
    return [*by, "items", *DIMENSIONS, f"{FLAG}_share"]


def markdown_table(by: Sequence[str], table: list[list[Any]]) -> str:
    """Format a summary as Markdown: means with two decimals, the flag as a percentage."""
    names = header(by)
    alignment = ["---"] * len(by) + ["---:"] * (len(names) - len(by))
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(alignment) + " |"]
    for row in table:
        means = ["–" if value is None else f"{value:.2f}" for value in row[len(by) + 1 : -1]]
        flag = "–" if row[-1] is None else f"{row[-1]:.0%}"
        lines.append("| " + " | ".join([*map(str, row[: len(by) + 1]), *means, flag]) + " |")
    return "\n".join(lines)


def write_summaries(rows: list[dict[str, Any]], out_dir: str | Path, name: str) -> Path:
    """Write the three summaries as one Markdown file and one CSV file per grouping."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        f"# Human evaluation: {name}",
        "",
        "Mean ratings on the rubric's 0-2 scale (no = 0, partly = 1, yes = 2) and the share",
        f"of items flagged for an invented rule. {len(rows)} rated items.",
    ]
    for grouping, by in GROUPINGS.items():
        table = summarise(rows, by)
        sections += ["", f"## By {grouping}", "", markdown_table(by, table)]
        csv_rows = [dict(zip(header(by), row, strict=True)) for row in table]
        write_csv(csv_rows, out_dir / f"{name}_by_{grouping}.csv", header(by))
    markdown_path = out_dir / f"{name}.md"
    markdown_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return markdown_path


def main() -> None:
    """Join a ratings file with the key and write the summaries to `--out-dir`."""
    parser = argparse.ArgumentParser(description="Summarise ratings per system, category, level.")
    parser.add_argument("--ratings", type=Path, required=True, help="a filled rating sheet")
    parser.add_argument("--key", type=Path, default=Path("outputs/human_eval/key.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--name", default="human_eval", help="prefix of the output files")
    args = parser.parse_args()

    rows = join_key(read_ratings(args.ratings), read_csv(args.key))
    print(f"Wrote {write_summaries(rows, args.out_dir, args.name)}")


if __name__ == "__main__":
    main()
