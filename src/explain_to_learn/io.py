"""File helpers shared by the package, the scripts, the annotation app and the demo.

`open_graph` loads the grammar graph from its folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explain_to_learn.graph import GrammarGraph


def open_graph(directory: str | Path) -> GrammarGraph:
    """Load the grammar graph from a folder with `concepts.yaml` and `errant_map.yaml`.

    Needs the `graph` extra. The graph module is imported only here, so code that uses the
    graph for some of its work (an ungrounded system, for example) runs without the extra.
    """
    from explain_to_learn.graph import load_graph

    directory = Path(directory)
    return load_graph(directory / "concepts.yaml", directory / "errant_map.yaml")
