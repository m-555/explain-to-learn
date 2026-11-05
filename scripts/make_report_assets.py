"""Figures for the README and the report, from aggregate counts only.

    data     Charts of the edit table in report/figures/data_*.png: error categories by
             CEFR level, edits per essay by level, the operation mix (M/R/U) by level and
             the most frequent categories per level. Only counts and shares are drawn,
             never learner text.

Usage:
    python scripts/make_report_assets.py data --edits data/processed/edits.jsonl
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from explain_to_learn.edits import Edit, read_jsonl

LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2", "native")
OPERATIONS = {"M": "Missing", "R": "Replacement", "U": "Unnecessary"}

# Colours from the reference palette of the project's figures (see draw_graph.py).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": BASELINE,
        "ytick.color": BASELINE,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 8,
        "axes.titlesize": 9,
        "legend.frameon": False,
    }
)


# --- Data statistics ---------------------------------------------------------------


def present_levels(edits: Iterable[Edit]) -> list[str]:
    """The CEFR levels that occur in `edits`, in level order."""
    found = {edit.cefr_level for edit in edits}
    return [level for level in LEVELS if level in found]


def category_level_shares(edits: Sequence[Edit]) -> tuple[list[str], list[str], np.ndarray]:
    """Share of each level's edits in each error category.

    Returns the categories (most frequent first), the levels and a matrix with one row
    per category and one column per level; each column sums to 1.
    """
    levels = present_levels(edits)
    categories = [c for c, _ in Counter(e.error_category for e in edits).most_common()]
    counts = Counter((e.error_category, e.cefr_level) for e in edits)
    level_totals = Counter(e.cefr_level for e in edits)
    matrix = np.array(
        [[counts[c, level] / level_totals[level] for level in levels] for c in categories]
    )
    return categories, levels, matrix


def edits_per_essay(edits: Iterable[Edit]) -> dict[str, list[int]]:
    """Number of edits in each essay, grouped by the essay's level."""
    per_essay = Counter(edit.essay_key for edit in edits)
    level_of = {edit.essay_key: edit.cefr_level for edit in edits}
    by_level: dict[str, list[int]] = defaultdict(list)
    for essay, count in sorted(per_essay.items()):
        by_level[level_of[essay]].append(count)
    return {level: by_level[level] for level in LEVELS if level in by_level}


def operation_shares(edits: Sequence[Edit]) -> dict[str, dict[str, float]]:
    """Share of missing, replacement and unnecessary edits within each level."""
    counts = Counter((e.cefr_level, e.operation) for e in edits)
    totals = Counter(e.cefr_level for e in edits)
    return {
        level: {op: counts[level, op] / totals[level] for op in OPERATIONS}
        for level in present_levels(edits)
    }


def top_categories(edits: Sequence[Edit], n: int = 5) -> dict[str, list[tuple[str, float]]]:
    """The `n` most frequent error categories of each level, with their share."""
    by_level: dict[str, Counter] = defaultdict(Counter)
    for edit in edits:
        by_level[edit.cefr_level][edit.error_category] += 1
    return {
        level: [(c, k / by_level[level].total()) for c, k in by_level[level].most_common(n)]
        for level in present_levels(edits)
    }


def plot_category_heatmap(edits: Sequence[Edit]) -> Figure:
    """Heatmap: share of each level's edits in each error category."""
    categories, levels, matrix = category_level_shares(edits)
    fig, ax = plt.subplots(figsize=(4.2, 0.18 * len(categories) + 1.2))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    image = ax.imshow(100 * matrix, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(categories)), categories)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Error categories by CEFR level", loc="left")
    bar = fig.colorbar(image, ax=ax, shrink=0.6)
    bar.set_label("Share of the level's edits (%)")
    bar.outline.set_visible(False)
    return fig


def plot_edits_per_essay(edits: Sequence[Edit]) -> Figure:
    """Box plot of the number of edits per essay, by level."""
    by_level = edits_per_essay(edits)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.boxplot(
        list(by_level.values()),
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": SERIES[0] + "1a", "edgecolor": SERIES[0], "linewidth": 1},
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": SERIES[0], "linewidth": 1},
        capprops={"color": SERIES[0], "linewidth": 1},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTED, "alpha": 0.5},
    )
    labels = [f"{level}\n({len(counts)})" for level, counts in by_level.items()]
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Edits per essay")
    ax.set_xlabel("CEFR level (number of essays)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Edits per essay by level", loc="left")
    return fig


def plot_operation_mix(edits: Sequence[Edit]) -> Figure:
    """Stacked horizontal bars: missing, replacement and unnecessary edits per level."""
    shares = operation_shares(edits)
    levels = list(shares)
    fig, ax = plt.subplots(figsize=(4.2, 0.32 * len(levels) + 1.0))
    left = np.zeros(len(levels))
    for colour, (op, name) in zip(SERIES, OPERATIONS.items(), strict=False):
        widths = np.array([100 * shares[level][op] for level in levels])
        bars = ax.barh(
            levels,
            widths,
            left=left,
            height=0.6,
            color=colour,
            label=name,
            edgecolor=SURFACE,
            linewidth=1,
        )
        for bar, width in zip(bars, widths, strict=True):
            if width >= 8:
                x = bar.get_x() + 1.5
                ax.text(x, bar.get_y() + 0.3, f"{width:.0f}%", va="center", color="white")
        left += widths
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the level's edits (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1)
    ax.set_title("Operation mix by level", loc="left", pad=18)
    return fig


def plot_top_categories(edits: Sequence[Edit], n: int = 5) -> Figure:
    """Small multiples: the `n` most frequent error categories of each level."""
    top = top_categories(edits, n)
    columns = 4
    rows = math.ceil(len(top) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.9, 1.5 * rows + 0.4), squeeze=False)
    for ax, (level, pairs) in zip(axes.flat, top.items(), strict=False):
        names = [name for name, _ in pairs]
        values = [100 * share for _, share in pairs]
        ax.barh(names, values, height=0.6, color=SERIES[0])
        for y, value in enumerate(values):
            ax.text(value + 1, y, f"{value:.0f}%", va="center", color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.35)
        ax.set_xticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(level, loc="left")
    for ax in list(axes.flat)[len(top) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Top {n} error categories by level (share of the level's edits)", x=0.01, ha="left"
    )
    fig.tight_layout()
    return fig


def save(fig: Figure, path: Path) -> Path:
    """Save a figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


DATA_FIGURES = {
    "data_categories_by_level.png": plot_category_heatmap,
    "data_edits_per_essay.png": plot_edits_per_essay,
    "data_operations_by_level.png": plot_operation_mix,
    "data_top_categories_by_level.png": plot_top_categories,
}


def make_data_figures(edits: Sequence[Edit], out_dir: Path) -> list[Path]:
    """Draw and save every data statistics figure; returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, plot in DATA_FIGURES.items():
        written.append(save(plot(edits), out_dir / name))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Figures for the README and report.")
    parser.add_argument("--figures-dir", type=Path, default=Path("report/figures"))
    steps = parser.add_subparsers(dest="step", required=True)

    data = steps.add_parser("data", help="Data statistics figures from the edit table.")
    data.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    args = parser.parse_args(argv)

    if not args.edits.is_file():
        print(f"{args.edits} not found; build the edit table first (see README).")
        return
    for path in make_data_figures(read_jsonl(args.edits), args.figures_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
