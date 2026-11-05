"""Figures and tables for the README and the report, from aggregate files only.

Two steps. Both write counts, means and intervals, never learner text.

    data     Charts of the edit table in report/figures/data_*.png: error categories by
             CEFR level, edits per essay by level, the operation mix (M/R/U) by level and
             the most frequent categories per level.
    results  The README results table (eval/results/results_table.md), LaTeX tables
             (report/tables/*.tex) and figures (report/figures/*.png), from the files the
             evaluation scripts write:

             eval/results/metrics_<run>.json      scripts/compute_metrics.py
             outputs/annotations/ratings_rater1.csv   the first rater's ratings, saved
                                                   by the annotation app
             outputs/human_eval/key.csv            the key from scripts/make_rating_sheet.py
             outputs/human_eval/ratings_llm_judge.csv   scripts/llm_judge.py
             eval/results/agreement_raters.md
             eval/results/agreement_judge.md       scripts/agreement.py

             Only the rating columns of the sheets are read, never the sentences or the
             feedback. Missing files are skipped; with none of them nothing is written.

Intervals are 95% percentile bootstrap intervals: over gold edits for the automatic
metrics and over rated items for the human ratings. Differences between systems are
paired: the same edits are resampled for both systems (`metrics.paired_bootstrap`).

Usage:
    python scripts/make_report_assets.py data --edits data/processed/edits.jsonl
    python scripts/make_report_assets.py results --run gold-explain
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from explain_to_learn.edits import LEVELS, Edit, read_jsonl
from explain_to_learn.human_eval import DIMENSIONS, FLAG, join_key, read_ratings
from explain_to_learn.io import read_csv
from explain_to_learn.metrics import paired_bootstrap

OPERATIONS = {"M": "Missing", "R": "Replacement", "U": "Unnecessary"}
SYSTEM_ORDER = ("A", "B", "C", "D", "R")
COMPARISONS = (("B", "A"), ("C", "A"), ("D", "C"), ("D", "B"))

# Automatic metrics: key in the metrics summary, label, key of its number of edits.
AUTOMATIC = (
    ("covered", "Coverage", "covered_n"),
    ("concept_correct", "Concept accuracy", "concept_correct_n"),
    ("faithful", "Faithfulness", "faithful_n"),
    ("within_length", "Length fit", "within_length_n"),
    ("with_off_level_term", "Off-level terms", "within_length_n"),
)
LOWER_IS_BETTER = {"with_off_level_term", FLAG}
HUMAN_LABELS = {
    "rule_correct": "Rule correct",
    "error_located": "Error located",
    "helps_fix": "Helps fix",
    "fits_level": "Fits level",
    FLAG: "Invented rule",
}

# Colours from the reference palette of the project's figures (see draw_graph.py).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = ("o", "s", "^", "D", "v")
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SYSTEM_COLOURS = dict(zip(SYSTEM_ORDER, SERIES, strict=True))
SYSTEM_MARKERS = dict(zip(SYSTEM_ORDER, MARKERS, strict=True))

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


# --- Estimates with bootstrap intervals -------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A mean with its bootstrap interval; `low` and `high` are None without one."""

    value: float | None
    low: float | None = None
    high: float | None = None
    n: int = 0

    def scaled(self, factor: float) -> Estimate:
        """The same estimate in other units, e.g. a share in percent."""

        def scale(x: float | None) -> float | None:
            return None if x is None else x * factor

        return Estimate(scale(self.value), scale(self.low), scale(self.high), self.n)


def share_interval(
    share: float | None, n: int, n_resamples: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> Estimate:
    """Bootstrap interval of a 0/1 score's mean, from the mean and the number of edits.

    Resampling n edits with replacement from edits of which a share p score 1 gives a
    number of ones that follows Binomial(n, p). So the bootstrap distribution of the
    mean can be drawn from the aggregate file, without the per-edit scores.
    """
    if share is None or n == 0:
        return Estimate(None, n=n)
    means = np.random.default_rng(seed).binomial(n, share, size=n_resamples) / n
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(share, float(low), float(high), n)


def mean_interval(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> Estimate:
    """Percentile bootstrap interval of the mean of `values`."""
    if not values:
        return Estimate(None)
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_resamples, len(data))).mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(data.mean()), float(low), float(high), len(data))


def summary_estimates(summary: dict[str, Any], **bootstrap: Any) -> dict[str, Estimate]:
    """Estimates of the automatic metrics from one summary of `metrics.summarize`."""
    return {
        key: share_interval(summary.get(key), summary.get(n_key, 0), **bootstrap)
        for key, _, n_key in AUTOMATIC
    }


def human_estimates(rows: Sequence[dict[str, Any]], **bootstrap: Any) -> dict[str, Estimate]:
    """Mean rating of each dimension and the invented-rule rate, over rated items."""
    return {
        column: mean_interval([r[column] for r in rows if r[column] is not None], **bootstrap)
        for column in (*DIMENSIONS, FLAG)
    }


def group_rows(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    """Rated items grouped by the values of `keys`."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def human_differences(
    rows: Sequence[dict[str, Any]], a: str, b: str, **bootstrap: Any
) -> dict[str, dict[str, Any]]:
    """Paired bootstrap of system `a` minus system `b`, over edits both were rated on."""
    by_system = group_rows(rows, "system")
    scores = {
        system: {column: {r["edit_id"]: r[column] for r in items} for column in (*DIMENSIONS, FLAG)}
        for (system,), items in by_system.items()
    }
    if a not in scores or b not in scores:
        return {}
    return {
        column: paired_bootstrap(scores[a][column], scores[b][column], **bootstrap)
        for column in (*DIMENSIONS, FLAG)
    }


def difference_estimate(comparison: dict[str, Any]) -> Estimate:
    """An `Estimate` from a result of `metrics.paired_bootstrap`."""
    return Estimate(comparison["diff"], comparison["low"], comparison["high"], comparison["n"])


# --- Reading result files ----------------------------------------------------------


def load_metrics(path: Path) -> dict[str, Any] | None:
    """The JSON written by scripts/compute_metrics.py, or None if it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_ratings(ratings: Path, key: Path) -> list[dict[str, Any]] | None:
    """Rated items joined with the key (system, edit, category, level), or None."""
    if not (ratings.is_file() and key.is_file()):
        return None
    return join_key(read_ratings(ratings), read_csv(key))


def read_agreement(path: Path) -> dict[str, dict[str, float | None]] | None:
    """The table written by scripts/agreement.py, by dimension; None if no file.

    Each dimension has `items`, `exact`, `kappa` and `weighted_kappa`; statistics shown
    as "n/a" (undefined) or "–" (not computed) are None.
    """
    if not path.is_file():
        return None
    columns = ("items", "exact", "kappa", "weighted_kappa")
    table: dict[str, dict[str, float | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in HUMAN_LABELS:
            continue
        values = [None if cell in ("n/a", "–", "-") else float(cell) for cell in cells[1:]]
        table[cells[0]] = dict(zip(columns, values, strict=True))
    return table


def ordered_systems(names: Iterable[str]) -> list[str]:
    """System names in the order A, B, C, D, R, then any others alphabetically."""
    names = set(names)
    return [s for s in SYSTEM_ORDER if s in names] + sorted(names - set(SYSTEM_ORDER))


# --- Tables ------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One table row: a metric's estimate for each system."""

    label: str
    section: str  # "automatic" or "human"
    percent: bool
    cells: dict[str, Estimate]


def results_rows(
    metrics: dict[str, Any] | None, human: Sequence[dict[str, Any]] | None, **bootstrap: Any
) -> list[Row]:
    """Rows of the main results table: automatic metrics, then the human ratings."""
    rows = []
    if metrics:
        estimates = {
            system: summary_estimates(result["overall"], **bootstrap)
            for system, result in metrics["systems"].items()
        }
        for key, label, _ in AUTOMATIC:
            suffix = ", lower is better" if key in LOWER_IS_BETTER else ""
            cells = {system: estimates[system][key] for system in estimates}
            rows.append(Row(f"{label} (%{suffix})", "automatic", True, cells))
    if human:
        estimates = {
            system: human_estimates(items, **bootstrap)
            for (system,), items in group_rows(human, "system").items()
        }
        for column in (*DIMENSIONS, FLAG):
            percent = column == FLAG
            unit = "%, lower is better" if percent else "0-2"
            cells = {system: estimates[system][column] for system in estimates}
            rows.append(Row(f"{HUMAN_LABELS[column]} ({unit})", "human", percent, cells))
    return rows


def format_estimate(estimate: Estimate | None, percent: bool, latex: bool = False) -> str:
    """`85.3 (81.0–89.3)` for a share in percent, `1.62 (1.48–1.75)` for a rating."""
    if estimate is None or estimate.value is None:
        return "--" if latex else "–"
    if percent:
        estimate = estimate.scaled(100)
    digits = 1 if percent else 2
    value = f"{estimate.value:.{digits}f}"
    if estimate.low is None:
        return value
    interval = f"{estimate.low:.{digits}f}–{estimate.high:.{digits}f}"
    if latex:
        return f"{value} {{\\scriptsize ({interval.replace('–', '--')})}}"
    return f"{value} ({interval})"


def format_difference(estimate: Estimate, percent: bool, latex: bool = False) -> str:
    """A signed difference with its interval, e.g. `+3.2 [-1.0, 7.5]`."""
    if estimate.value is None:
        return "--" if latex else "–"
    if percent:
        estimate = estimate.scaled(100)
    digits = 1 if percent else 2
    text = f"{estimate.value:+.{digits}f} [{estimate.low:.{digits}f}, {estimate.high:.{digits}f}]"
    return text.replace("-", "$-$") if latex else text


def results_caption(metrics: dict[str, Any] | None, human: Sequence[dict] | None) -> str:
    """What the main table shows and on how many edits and items."""
    parts = []
    if metrics:
        parts.append(f"automatic metrics on {metrics['edits']} gold edits")
    if human:
        items = Counter(row["system"] for row in human)
        parts.append(f"mean teacher ratings on {max(items.values())} rated items per system")
    return (
        "Systems A to D and the teacher reference R: "
        + " and ".join(parts)
        + ", with 95% bootstrap intervals in parentheses."
    )


def markdown_results(rows: Sequence[Row], systems: Sequence[str], caption: str) -> str:
    """The README results table: one row per metric, one column per system."""
    lines = [
        "| Metric | " + " | ".join(systems) + " |",
        "| --- |" + " ---: |" * len(systems),
    ]
    for row in rows:
        cells = [format_estimate(row.cells.get(system), row.percent) for system in systems]
        lines.append(f"| {row.label} | " + " | ".join(cells) + " |")
    notes = [
        "",
        caption,
        "Automatic metrics: `scripts/compute_metrics.py`; ratings: blind teacher ratings on the "
        "rubric in `eval/rubric.md` (no = 0, partly = 1, yes = 2). A dash means not measured.",
    ]
    return "\n".join(lines + notes) + "\n"


def latex_escape(text: str) -> str:
    """Escape the LaTeX special characters that occur in labels."""
    for char in ("\\", "%", "&", "_", "#"):
        text = text.replace(char, "\\" + char)
    return text


def latex_table(
    header: Sequence[str],
    body: Sequence[Sequence[str]],
    caption: str,
    label: str,
    breaks: Sequence[int] = (),
    wide: bool = True,
    top: Sequence[str] = (),
) -> str:
    """A booktabs table.

    `breaks` are body row indices that get a `\\midrule` before them, and `top` holds
    LaTeX lines placed above the header, such as column group titles.
    """
    environment = "table*" if wide else "table"
    lines = [
        f"\\begin{{{environment}}}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        "\\toprule",
        *top,
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for index, cells in enumerate(body):
        if index in breaks:
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\end{{{environment}}}",
    ]
    return "\n".join(lines) + "\n"


def latex_results(rows: Sequence[Row], systems: Sequence[str], caption: str) -> str:
    """The main results table for the report."""
    body = [
        [latex_escape(row.label)]
        + [format_estimate(row.cells.get(s), row.percent, latex=True) for s in systems]
        for row in rows
    ]
    breaks = [i for i in range(1, len(rows)) if rows[i].section != rows[i - 1].section]
    return latex_table(["Metric", *systems], body, caption, "tab:results", breaks)


def difference_rows(
    metrics: dict[str, Any] | None, human: Sequence[dict[str, Any]] | None, **bootstrap: Any
) -> list[Row]:
    """Paired differences B-A, C-A, D-C and D-B, one row per metric (cells by comparison)."""
    rows = []
    if metrics:
        comparisons = metrics.get("comparisons", {})
        for key, label, _ in AUTOMATIC:
            cells = {}
            for a, b in COMPARISONS:
                result = comparisons.get(f"{a} - {b}", {}).get(key)
                if result and result["diff"] is not None:
                    cells[f"{a}-{b}"] = difference_estimate(result)
            if cells:
                rows.append(Row(f"{label} (pp)", "automatic", True, cells))
    if human:
        differences = {
            f"{a}-{b}": human_differences(human, a, b, **bootstrap) for a, b in COMPARISONS
        }
        for column in (*DIMENSIONS, FLAG):
            cells = {
                name: difference_estimate(result[column])
                for name, result in differences.items()
                if result and result[column]["diff"] is not None
            }
            if cells:
                unit = "pp" if column == FLAG else "rating points"
                rows.append(Row(f"{HUMAN_LABELS[column]} ({unit})", "human", column == FLAG, cells))
    return rows


def latex_differences(rows: Sequence[Row]) -> str:
    """Paired differences with their intervals, one column per comparison."""
    names = [f"{a}-{b}" for a, b in COMPARISONS]
    body = [
        [latex_escape(row.label)]
        + [
            format_difference(row.cells[n], row.percent, latex=True) if n in row.cells else "--"
            for n in names
        ]
        for row in rows
    ]
    breaks = [i for i in range(1, len(rows)) if rows[i].section != rows[i - 1].section]
    caption = (
        "Paired differences between systems (first minus second) with 95% paired bootstrap "
        "intervals: percentage points (pp) for shares, points on the 0-2 scale for ratings."
    )
    header = ["Metric", *(n.replace("-", "$-$") for n in names)]
    return latex_table(header, body, caption, "tab:differences", breaks)


AGREEMENT_SOURCES = {"raters": "Rater 2 vs rater 1", "judge": "LLM judge vs rater 1"}


def latex_agreement(agreement: dict[str, dict[str, dict[str, float | None]]]) -> str:
    """Agreement per dimension: items, exact agreement, kappa and weighted kappa."""

    def number(value: float | None, digits: int = 2) -> str:
        return "--" if value is None else f"{value:.{digits}f}".replace("-", "$-$")

    header = ["Dimension"] + ["Items", "Exact", "$\\kappa$", "$\\kappa_w$"] * len(agreement)
    titles = [f"\\multicolumn{{4}}{{c}}{{{AGREEMENT_SOURCES[source]}}}" for source in agreement]
    rules = [f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(agreement))]
    top = [" & ".join(["", *titles]) + " \\\\", " ".join(rules)]
    body = []
    for column in (*DIMENSIONS, FLAG):
        cells = [HUMAN_LABELS[column]]
        for table in agreement.values():
            stats = table.get(column, {})
            items = stats.get("items")
            cells += [
                "--" if items is None else f"{items:.0f}",
                number(stats.get("exact")),
                number(stats.get("kappa")),
                number(stats.get("weighted_kappa")),
            ]
        body.append(cells)
    caption = (
        "Agreement with the first rater: share of exact agreement, Cohen's kappa and, for the "
        "three-point dimensions, quadratic-weighted kappa."
    )
    return latex_table(header, body, caption, "tab:agreement", top=top)


# --- Figures -----------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One small multiple: estimates by group (a row) and series (a marker)."""

    title: str
    data: dict[str, dict[str, Estimate]]
    xlabel: str = ""
    limits: tuple[float | None, float | None] = (None, None)  # x range; None is automatic


SHARE_LIMITS = (-2.0, 102.0)  # percent, with room for markers at 0 and 100
RATING_LIMITS = (-0.05, 2.05)  # the rubric's 0-2 scale


def plot_intervals(
    panels: Sequence[Panel],
    groups: Sequence[str],
    series: Sequence[str],
    title: str,
    zero_line: bool = False,
) -> Figure:
    """Points with interval whiskers: one panel per metric, one row per group.

    Each series keeps its colour and marker shape (systems by name, so a missing system
    does not repaint the others), and a legend names them when there is more than one.
    """
    colours = {name: SYSTEM_COLOURS.get(name, SERIES[i]) for i, name in enumerate(series)}
    markers = {name: SYSTEM_MARKERS.get(name, MARKERS[i]) for i, name in enumerate(series)}
    width = min(6.9, 1.6 * len(panels) + 1.2)
    height = 0.12 * len(groups) * max(len(series), 2) + 1.1
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True, squeeze=False)
    step = 0.6 / len(series)  # series of one group stay closer than neighbouring groups
    offsets = [(i - (len(series) - 1) / 2) * step for i in range(len(series))]
    for ax, panel in zip(axes[0], panels, strict=True):
        if zero_line:
            ax.axvline(0, color=BASELINE, linewidth=1)
        for offset, name in zip(offsets, series, strict=True):
            for row, group in enumerate(groups):
                estimate = panel.data.get(group, {}).get(name)
                if estimate is None or estimate.value is None:
                    continue
                y = row + offset
                if estimate.low is not None:
                    ax.hlines(y, estimate.low, estimate.high, color=colours[name], linewidth=1.5)
                ax.plot(
                    estimate.value,
                    y,
                    marker=markers[name],
                    markersize=5,
                    color=colours[name],
                    markeredgecolor=SURFACE,
                    markeredgewidth=1,
                    linestyle="none",
                )
        ax.set_title(panel.title, loc="left")
        ax.set_xlabel(panel.xlabel)
        ax.set_xlim(*panel.limits)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0][0].set_yticks(range(len(groups)), groups)
    axes[0][0].set_ylim(len(groups) - 0.5, -0.5)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    if len(series) > 1:
        handles = [
            Line2D([], [], marker=markers[n], color=colours[n], linestyle="none", markersize=5)
            for n in series
        ]
        fig.legend(handles, series, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(series))
    return fig


def breakdown_panels_automatic(
    metrics: dict[str, Any], key: str, min_edits: int, **bootstrap: Any
) -> tuple[list[Panel], list[str]]:
    """Panels of the automatic metrics by level (`key="by_level"`) or error category.

    Groups with fewer than `min_edits` edits are left out; categories are sorted by
    their number of edits.
    """
    systems = metrics["systems"]
    sizes = Counter()
    for result in systems.values():
        for group, summary in result[key].items():
            sizes[group] = max(sizes[group], summary["edits"])
    groups = [g for g, n in sizes.most_common() if n >= min_edits]
    if key == "by_level":
        groups = [level for level in LEVELS if level in groups]
    panels = []
    for metric, label, _ in AUTOMATIC[:4]:
        data = {
            group: {
                system: summary_estimates(result[key][group], **bootstrap)[metric].scaled(100)
                for system, result in systems.items()
                if group in result[key]
            }
            for group in groups
        }
        panels.append(Panel(label, data, "%", SHARE_LIMITS))
    return panels, groups


def breakdown_panels_human(
    human: Sequence[dict[str, Any]], key: str, min_items: int, **bootstrap: Any
) -> tuple[list[Panel], list[str]]:
    """Panels of the mean ratings by `level` or `error_category` and system."""
    by_group = group_rows(human, key, "system")
    sizes = Counter()
    for (group, _), items in by_group.items():
        sizes[group] = max(sizes[group], len(items))
    groups = [g for g, n in sizes.most_common() if n >= min_items]
    if key == "level":
        groups = [level for level in LEVELS if level in groups]
    estimates = {
        (group, system): human_estimates(items, **bootstrap)
        for (group, system), items in by_group.items()
        if group in groups
    }
    panels = []
    for column in DIMENSIONS:
        data: dict[str, dict[str, Estimate]] = defaultdict(dict)
        for (group, system), values in estimates.items():
            data[group][system] = values[column]
        panels.append(Panel(HUMAN_LABELS[column], dict(data), "mean rating (0-2)", RATING_LIMITS))
    return panels, groups


def difference_panels(rows: Sequence[Row]) -> list[Panel]:
    """One panel per metric with the paired differences by comparison."""
    panels = []
    for row in rows:
        unit = "percentage points" if row.percent else "rating points"
        data = {
            name: {"difference": e.scaled(100 if row.percent else 1)}
            for name, e in row.cells.items()
        }
        panels.append(Panel(row.label.split(" (")[0], data, unit))
    return panels


def agreement_panels(agreement: dict[str, dict[str, dict[str, float | None]]]) -> list[Panel]:
    """Cohen's kappa and quadratic-weighted kappa per dimension, for each comparison."""
    panels = []
    for stat, title in (("kappa", "Cohen's kappa"), ("weighted_kappa", "Weighted kappa")):
        data: dict[str, dict[str, Estimate]] = defaultdict(dict)
        for source, table in agreement.items():
            for column, stats in table.items():
                if stats.get(stat) is not None:
                    data[HUMAN_LABELS[column]][AGREEMENT_SOURCES[source]] = Estimate(stats[stat])
        panels.append(Panel(title, dict(data), "kappa", (None, 1.02)))
    return panels


def judge_panels(
    human: Sequence[dict[str, Any]], judge: Sequence[dict[str, Any]], **bootstrap: Any
) -> tuple[list[Panel], list[str]]:
    """Mean rating per system from the first rater and from the LLM judge."""
    sources = {"Rater 1": human, "LLM judge": judge}
    estimates = {
        (name, system): human_estimates(items, **bootstrap)
        for name, rows in sources.items()
        for (system,), items in group_rows(rows, "system").items()
    }
    systems = ordered_systems(system for _, system in estimates)
    panels = []
    for column in DIMENSIONS:
        data: dict[str, dict[str, Estimate]] = defaultdict(dict)
        for (name, system), values in estimates.items():
            data[system][name] = values[column]
        panels.append(Panel(HUMAN_LABELS[column], dict(data), "mean rating (0-2)", RATING_LIMITS))
    return panels, systems


# --- Results step ------------------------------------------------------------------


def write_text(text: str, path: Path) -> Path:
    """Write a text file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_results(args: argparse.Namespace) -> list[Path]:
    """Write every table and figure that the available result files allow."""
    metrics = load_metrics(args.metrics)
    human = load_ratings(args.ratings, args.key)
    judge = load_ratings(args.judge_ratings, args.key)
    agreement = {
        source: table
        for source, path in (("raters", args.agreement_raters), ("judge", args.agreement_judge))
        if (table := read_agreement(path))
    }
    if not (metrics or human or agreement):
        print("No result files are available, so nothing was written. Looked for:")
        for path in (args.metrics, args.ratings, args.agreement_raters, args.agreement_judge):
            print(f"  {path}")
        return []

    bootstrap = {"n_resamples": args.resamples, "seed": args.seed}
    figures, tables = args.figures_dir, args.tables_dir
    written = []
    rows = results_rows(metrics, human, **bootstrap)
    if rows:
        systems = ordered_systems(name for row in rows for name in row.cells)
        caption = results_caption(metrics, human)
        written.append(write_text(markdown_results(rows, systems, caption), args.table))
        latex = latex_results(rows, systems, caption)
        written.append(write_text(latex, tables / "results_main.tex"))

    differences = difference_rows(metrics, human, **bootstrap)
    if differences:
        written.append(
            write_text(latex_differences(differences), tables / "paired_differences.tex")
        )
        comparisons = [f"{a}-{b}" for a, b in COMPARISONS]
        for section in ("automatic", "human"):
            section_rows = [row for row in differences if row.section == section]
            if section_rows:
                what = "metrics" if section == "automatic" else "ratings"
                title = f"Paired differences ({section} {what})"
                fig = plot_intervals(
                    difference_panels(section_rows), comparisons, ["difference"], title, True
                )
                written.append(save(fig, figures / f"paired_differences_{section}.png"))

    breakdowns = []
    if metrics:
        for key, name in (("by_level", "level"), ("by_error_category", "category")):
            panels, groups = breakdown_panels_automatic(metrics, key, args.min_group, **bootstrap)
            title = f"Automatic metrics by {'CEFR level' if name == 'level' else 'error category'}"
            breakdowns.append((panels, groups, title, f"results_by_{name}.png"))
    if human:
        for key, name in (("level", "level"), ("error_category", "category")):
            panels, groups = breakdown_panels_human(human, key, args.min_group, **bootstrap)
            title = f"Teacher ratings by {'CEFR level' if name == 'level' else 'error category'}"
            breakdowns.append((panels, groups, title, f"human_by_{name}.png"))
    for panels, groups, title, filename in breakdowns:
        if not groups:
            print(f"Skipped {filename}: no group has {args.min_group} or more items")
            continue
        series = ordered_systems(s for p in panels for cells in p.data.values() for s in cells)
        written.append(save(plot_intervals(panels, groups, series, title), figures / filename))

    if agreement:
        written.append(write_text(latex_agreement(agreement), tables / "agreement.tex"))
        series = [AGREEMENT_SOURCES[source] for source in agreement]
        groups = [HUMAN_LABELS[c] for c in (*DIMENSIONS, FLAG)]
        fig = plot_intervals(agreement_panels(agreement), groups, series, "Agreement", True)
        written.append(save(fig, figures / "judge_agreement.png"))
    if human and judge:
        panels, systems = judge_panels(human, judge, **bootstrap)
        title = "Mean rating per system: first rater and LLM judge"
        fig = plot_intervals(panels, systems, ["Rater 1", "LLM judge"], title)
        written.append(save(fig, figures / "judge_vs_rater_means.png"))
    return written


def main(argv: list[str] | None = None) -> list[Path]:
    """Run one step; returns the paths written (none when its inputs are missing)."""
    parser = argparse.ArgumentParser(description="Figures and tables for the README and report.")
    parser.add_argument("--figures-dir", type=Path, default=Path("report/figures"))
    steps = parser.add_subparsers(dest="step", required=True)

    data = steps.add_parser("data", help="Data statistics figures from the edit table.")
    data.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))

    results = steps.add_parser("results", help="Results table, LaTeX tables and figures.")
    results.add_argument("--run", default="gold-explain", help="run name of the metrics file")
    results.add_argument("--metrics", type=Path, help="default: eval/results/metrics_<run>.json")
    human_dir = Path("outputs/human_eval")
    # Where the annotation app saves the first rater's ratings (annotation/ratings.py).
    results.add_argument(
        "--ratings", type=Path, default=Path("outputs/annotations/ratings_rater1.csv")
    )
    results.add_argument("--key", type=Path, default=human_dir / "key.csv")
    results.add_argument("--judge-ratings", type=Path, default=human_dir / "ratings_llm_judge.csv")
    results.add_argument(
        "--agreement-raters", type=Path, default=Path("eval/results/agreement_raters.md")
    )
    results.add_argument(
        "--agreement-judge", type=Path, default=Path("eval/results/agreement_judge.md")
    )
    results.add_argument("--table", type=Path, default=Path("eval/results/results_table.md"))
    results.add_argument("--tables-dir", type=Path, default=Path("report/tables"))
    results.add_argument("--min-group", type=int, default=10, help="smallest group to plot")
    results.add_argument("--resamples", type=int, default=10_000)
    results.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.step == "data":
        if not args.edits.is_file():
            print(f"{args.edits} not found; build the edit table first (see README).")
            return []
        written = make_data_figures(read_jsonl(args.edits), args.figures_dir)
    else:
        args.metrics = args.metrics or Path(f"eval/results/metrics_{args.run}.json")
        written = make_results(args)
    for path in written:
        print(f"Wrote {path}")
    return written


if __name__ == "__main__":
    main()
