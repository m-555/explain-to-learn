"""Aggregate statistics of the edit table, written as Markdown tables.

Only counts are written, never learner text, so the output can be committed.

Usage:
    python scripts/data_stats.py --edits data/processed/edits.jsonl --out eval/results/data_stats.md
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from explain_to_learn.edits import LEVELS, Edit, read_jsonl

SPLITS = ["train", "dev", "test"]
TOP_N = 15


def table(header: list[str], rows: list[list[object]]) -> str:
    """Format a Markdown table; columns after the first are right-aligned."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- |" + " ---: |" * (len(header) - 1),
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def corpus_table(edits: list[Edit]) -> str:
    """Essays and edits per corpus and split."""
    essays = Counter((c, s) for c, s, _ in {(e.corpus, e.split, e.essay_key) for e in edits})
    edit_counts = Counter((e.corpus, e.split) for e in edits)
    rows = [[f"{c} / {s}", essays[c, s], edit_counts[c, s]] for c, s in sorted(edit_counts)]
    rows.append(["total", sum(essays.values()), len(edits)])
    return table(["Corpus / split", "Essays", "Edits"], rows)


def level_table(edits: list[Edit]) -> str:
    """Edits per CEFR level and split."""
    counts = Counter((e.cefr_level, e.split) for e in edits)
    rows = [
        [level, *(counts[level, s] for s in SPLITS), sum(counts[level, s] for s in SPLITS)]
        for level in LEVELS
    ]
    return table(["Level", *SPLITS, "total"], rows)


def category_level_table(edits: list[Edit]) -> str:
    """Error category x CEFR level counts, most frequent categories first."""
    counts = Counter((e.error_category, e.cefr_level) for e in edits)
    totals = Counter(e.error_category for e in edits)
    rows = [
        [category, *(counts[category, level] for level in LEVELS), total]
        for category, total in totals.most_common()
    ]
    return table(["Category", *LEVELS, "total"], rows)


def top_categories_table(edits: list[Edit]) -> str:
    """The most frequent error categories with their share of all edits."""
    totals = Counter(e.error_category for e in edits)
    rows = [
        [category, count, f"{100 * count / len(edits):.1f}%"]
        for category, count in totals.most_common(TOP_N)
    ]
    return table(["Category", "Edits", "Share"], rows)


def operation_table(edits: list[Edit]) -> str:
    """Edits per ERRANT operation: missing, replacement, unnecessary."""
    counts = Counter(e.operation for e in edits)
    names = {"M": "M (missing)", "R": "R (replacement)", "U": "U (unnecessary)"}
    return table(["Operation", "Edits"], [[names[op], counts[op]] for op in names])


def stats_markdown(edits: list[Edit]) -> str:
    """The full statistics report."""
    dev = [e for e in edits if e.split == "dev" and e.corpus in ("wi", "locness")]
    sections = [
        "# Data statistics",
        "Edits in the edit table built from the BEA-2019 releases of W&I+LOCNESS and FCE "
        "(gold ERRANT edits, annotator 0, `noop` and `UNK` edits excluded). FCE essays have "
        "no individual CEFR rating and are counted as B2, the level of the FCE exam. "
        "Essays with no edits are not counted.",
        "## Essays and edits per corpus and split",
        corpus_table(edits),
        "## Edits per CEFR level",
        level_table(edits),
        "## Edits per operation",
        operation_table(edits),
        f"## Top {TOP_N} error categories (all corpora)",
        top_categories_table(edits),
        "## Error category by CEFR level (all corpora)",
        category_level_table(edits),
        "## Error category by CEFR level (W&I+LOCNESS dev, the gold-set pool)",
        category_level_table(dev),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    """Write the statistics of the edit table to `--out`."""
    parser = argparse.ArgumentParser(description="Write aggregate edit statistics.")
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/data_stats.md"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(stats_markdown(read_jsonl(args.edits)), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
