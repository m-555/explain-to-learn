"""Draw the grammar graph's prerequisite links as a figure.

Concepts are placed in columns by depth: a concept sits one column to the right of
its deepest prerequisite, so links run left to right. Reviewed concepts are filled
blue, draft concepts are hollow orange.

Usage:
    python scripts/draw_graph.py [--out report/figures/grammar_graph.png]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explain_to_learn.graph import (  # noqa: E402
    DEFAULT_CONCEPTS_PATH,
    DEFAULT_MAP_PATH,
    GrammarGraph,
    load_graph,
)

REVIEWED_COLOUR = "#2a78d6"
DRAFT_COLOUR = "#eb6834"
TEXT_COLOUR = "#0b0b0b"
EDGE_COLOUR = "#b8b7b0"
BACKGROUND = "#fcfcfb"


def layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Place each concept in a column by depth.

    The first column is sorted by name. In later columns a concept is placed near the
    average height of its prerequisites, which reduces crossing links.
    """
    positions: dict[str, tuple[float, float]] = {}
    for column, layer in enumerate(nx.topological_generations(graph)):

        def height(node: str) -> float:
            parents = list(graph.predecessors(node))
            return -sum(positions[p][1] for p in parents) / len(parents) if parents else 0.0

        for row, node in enumerate(sorted(layer, key=lambda n: (height(n), n))):
            positions[node] = (float(column), -float(row))
    return positions


def draw(grammar: GrammarGraph, out_path: str | Path) -> Path:
    """Draw the prerequisite graph to `out_path` and return the path."""
    graph = grammar.graph
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the prerequisite graph has a cycle; run the graph validator")
    positions = layout(graph)
    columns = 1 + max(x for x, _ in positions.values())
    rows = 1 - min(y for _, y in positions.values())

    fig, ax = plt.subplots(figsize=(4.2 * columns, 0.55 * rows + 1.5))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color=EDGE_COLOUR,
        width=1.0,
        arrowsize=10,
        node_size=120,
        connectionstyle="arc3,rad=0.08",
    )
    reviewed = [n for n in graph if graph.nodes[n]["status"] == "reviewed"]
    draft = [n for n in graph if n not in reviewed]
    nx.draw_networkx_nodes(
        graph, positions, nodelist=reviewed, ax=ax, node_size=120, node_color=REVIEWED_COLOUR
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=draft,
        ax=ax,
        node_size=120,
        node_color=BACKGROUND,
        edgecolors=DRAFT_COLOUR,
        linewidths=2.0,
    )
    for node, (x, y) in positions.items():
        label = textwrap.fill(graph.nodes[node]["name"] or node, width=28)
        ax.text(
            x + 0.06,
            y,
            label,
            fontsize=9,
            color=TEXT_COLOUR,
            va="center",
            ha="left",
            bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )

    legend = [
        Line2D([], [], marker="o", ls="", color=REVIEWED_COLOUR, markersize=8, label="Reviewed"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor=BACKGROUND,
            markeredgecolor=DRAFT_COLOUR,
            markeredgewidth=2,
            markersize=8,
            label="Draft",
        ),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=10)
    ax.set_title(
        "Grammar graph: prerequisite links (prerequisite → concept)",
        loc="left",
        fontsize=13,
        color=TEXT_COLOUR,
    )
    ax.set_xlim(-0.2, columns - 0.15)
    ax.set_ylim(-rows + 0.5, 0.8)
    ax.axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Draw the prerequisite graph to `--out`; returns the exit code."""
    parser = argparse.ArgumentParser(description="Draw the grammar graph's prerequisite links.")
    parser.add_argument("--concepts", default=DEFAULT_CONCEPTS_PATH)
    parser.add_argument("--map", default=DEFAULT_MAP_PATH)
    parser.add_argument("--out", default="report/figures/grammar_graph.png")
    args = parser.parse_args(argv)
    path = draw(load_graph(args.concepts, args.map), args.out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
