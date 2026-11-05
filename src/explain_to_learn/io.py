"""File helpers shared by the package, the scripts, the annotation app and the demo.

- ID files (`data/splits/*.txt`, `outputs/dev_ids.txt`): one edit ID or essay key per
  line; `read_ids` and `write_ids`.
- JSON Lines files (run files, silver data, SFT and DPO examples): one JSON object per
  line; `read_rows` and `write_rows`, and `read_run` for a run file by edit ID.
- CSV files with a header row (rating sheets, the key, ratings, the hand-check):
  `read_csv` and `write_csv`.

`open_graph` loads the grammar graph from its folder.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from explain_to_learn.graph import GrammarGraph


def read_ids(path: str | Path) -> list[str]:
    """IDs from a text file with one ID per line, in file order; blank lines are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def write_ids(ids: Iterable[str], path: str | Path) -> None:
    """Write IDs to a text file, one per line, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries; blank lines are skipped."""
    with Path(path).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dictionaries to a JSON Lines file, one per line, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_run(path: str | Path) -> dict[str, dict[str, Any]]:
    """The generation records of a run file (`scripts/run_system.py`) by edit ID.

    If an edit was run more than once, its last record wins.
    """
    return {row["edit_id"]: row for row in read_rows(path)}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file with a header row into a list of dictionaries."""
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path, columns: Sequence[str]) -> None:
    """Write dictionaries to a CSV file with the given columns, in that order.

    Keys of a row that are not in `columns` are left out. The folder is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_graph(directory: str | Path) -> GrammarGraph:
    """Load the grammar graph from a folder with `concepts.yaml` and `errant_map.yaml`.

    Needs the `graph` extra. The graph module is imported only here, so code that uses the
    graph for some of its work (an ungrounded system, for example) runs without the extra.
    """
    from explain_to_learn.graph import load_graph

    directory = Path(directory)
    return load_graph(directory / "concepts.yaml", directory / "errant_map.yaml")
