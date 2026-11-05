"""Agreement between two sets of ratings of the same items, per rubric dimension.

For each dimension it reports the number of items both sets rated, the share of exact
agreement, Cohen's kappa and, for the three-point dimensions, quadratic-weighted kappa.

Cohen's kappa corrects agreement for chance: kappa = 1 - D_observed / D_expected, where
D is the mean disagreement weight over item pairs, observed in the ratings or expected
if the two sets were independent with the same label frequencies. Plain kappa gives
every disagreement weight 1. Quadratic weights, ((i - j) / (k - 1))^2 on a k-point
scale, make a no-vs-yes disagreement count four times as much as an off-by-one one.

Use it for the second rater against the first (50 items), and for the LLM judge
against the first rater (all items).

Usage:
    python scripts/agreement.py --first outputs/human_eval/ratings_rater1.csv \
        --second outputs/human_eval/ratings_rater2.csv --title "Rater 1 vs rater 2" \
        --out eval/results/agreement_raters.md
    python scripts/agreement.py --first outputs/human_eval/ratings_rater1.csv \
        --second outputs/human_eval/ratings_llm_judge.csv --title "LLM judge vs rater 1" \
        --out eval/results/agreement_judge.md

`scripts/make_report_assets.py` reads both tables from these default paths.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from explain_to_learn.human_eval import DIMENSIONS, FLAG, FLAG_VALUES, SCALE, read_ratings

Ratings = dict[str, dict[str, int | None]]


def cohen_kappa(
    first: Sequence[int], second: Sequence[int], categories: Sequence[int], quadratic: bool = False
) -> float:
    """Cohen's kappa of two raters; quadratic-weighted if `quadratic` is true.

    Returns NaN when it is undefined: no items, or both raters always used one label.
    """
    n, k = len(first), len(categories)
    if n == 0 or len(second) != n:
        return math.nan
    position = {category: i for i, category in enumerate(categories)}

    def weight(i: int, j: int) -> float:
        return ((i - j) / (k - 1)) ** 2 if quadratic else float(i != j)

    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(first, second, strict=True):
        observed[position[a]][position[b]] += 1 / n
    first_share = [sum(row) for row in observed]
    second_share = [sum(row[j] for row in observed) for j in range(k)]
    cells = [(i, j) for i in range(k) for j in range(k)]
    disagreement = sum(weight(i, j) * observed[i][j] for i, j in cells)
    expected = sum(weight(i, j) * first_share[i] * second_share[j] for i, j in cells)
    return math.nan if expected == 0 else 1 - disagreement / expected


def paired(first: Ratings, second: Ratings, column: str) -> tuple[list[int], list[int]]:
    """Ratings of one column for the items that both sets rated."""
    items = sorted(set(first) & set(second))
    pairs = [(first[i][column], second[i][column]) for i in items]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return [a for a, _ in pairs], [b for _, b in pairs]


def agreement_rows(first: Ratings, second: Ratings) -> list[list[object]]:
    """One row per dimension: items, exact agreement, kappa, weighted kappa (or None)."""
    rows = []
    for column in (*DIMENSIONS, FLAG):
        a, b = paired(first, second, column)
        scale = sorted((SCALE if column in DIMENSIONS else FLAG_VALUES).values())
        exact = sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else math.nan
        weighted = cohen_kappa(a, b, scale, quadratic=True) if column in DIMENSIONS else None
        rows.append([column, len(a), exact, cohen_kappa(a, b, scale), weighted])
    return rows


def number(value: object) -> str:
    """Format a statistic with two decimals; NaN as 'n/a' and None as '–'."""
    if value is None:
        return "–"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.2f}"
    return str(value)


def markdown(title: str, rows: list[list[object]]) -> str:
    """The agreement table as Markdown."""
    lines = [
        f"# {title}",
        "",
        "| Dimension | Items | Exact agreement | Cohen's kappa | Quadratic-weighted kappa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += ["| " + " | ".join(number(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement between two sets of ratings.")
    parser.add_argument("--first", type=Path, required=True, help="reference ratings (CSV)")
    parser.add_argument("--second", type=Path, required=True, help="ratings to compare (CSV)")
    parser.add_argument("--title", default="Agreement")
    parser.add_argument("--out", type=Path, help="write the table here (Markdown)")
    args = parser.parse_args()

    table = markdown(
        args.title, agreement_rows(read_ratings(args.first), read_ratings(args.second))
    )
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
