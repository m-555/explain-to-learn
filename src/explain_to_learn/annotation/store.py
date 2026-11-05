"""Annotation files: one per task, written in full after every saved item.

A store keeps one record per item, keyed by the item's ID. `Store.save` adds or replaces
a record and writes the whole file again, in the order of the items, through a temporary
file that then replaces the old one. A crash or a closed laptop can lose at most the
item being saved, never the rest of the file. The files hold a few hundred rows, so
writing them in full each time is fast.

Before writing, the store reads the file again, so two browser tabs open on the same
task add to each other's work instead of overwriting it.

Files ending in `.csv` are written with a fixed list of columns (the format that the
evaluation scripts read); all other files are JSON Lines.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any

Record = dict[str, Any]
Key = Callable[[Record], Hashable]

_WRITE_LOCK = threading.Lock()


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file in the same folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def read_records(path: Path) -> list[Record]:
    """Read a CSV file with a header row, or a JSON Lines file; [] if it does not exist."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        if path.suffix == ".csv":
            return list(csv.DictReader(file))
        return [json.loads(line) for line in file if line.strip()]


def format_records(records: Sequence[Record], path: Path, columns: Sequence[str] | None) -> str:
    """The file text for `records`: CSV with `columns` for a .csv path, else JSON Lines."""
    if path.suffix == ".csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(columns or ()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        return buffer.getvalue()
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


class Store:
    """The saved records of one task, by item key, written to `path` after every save.

    `key` gives a record's item key, and `order` lists the item keys in the order the
    items are shown. Records for keys that are not in `order` (for example from an older
    item list) are kept at the end of the file, never dropped.
    """

    def __init__(
        self,
        path: str | Path,
        key: Key,
        order: Sequence[Hashable],
        columns: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.suffix == ".csv" and not columns:
            raise ValueError("a CSV store needs its list of columns")
        self.key = key
        self.order = list(order)
        self.columns = tuple(columns) if columns else None
        self.records = self._read()

    def _read(self) -> dict[Hashable, Record]:
        return {self.key(record): record for record in read_records(self.path)}

    def get(self, item_key: Hashable) -> Record | None:
        """The saved record of an item, or None."""
        return self.records.get(item_key)

    def save(self, record: Record) -> None:
        """Add or replace the record of one item and write the file."""
        with _WRITE_LOCK:
            self.records = self._read()
            self.records[self.key(record)] = record
            position = {item_key: i for i, item_key in enumerate(self.order)}
            rows = sorted(
                self.records.values(),
                key=lambda row: position.get(self.key(row), len(position)),
            )
            write_atomic(self.path, format_records(rows, self.path, self.columns))

    def done_count(self) -> int:
        """Number of items that have a saved record."""
        return sum(item_key in self.records for item_key in self.order)

    def first_open(self) -> int:
        """Position of the first item without a record; the last item if all are done."""
        for index, item_key in enumerate(self.order):
            if item_key not in self.records:
                return index
        return max(len(self.order) - 1, 0)

    def progress(self, index: int) -> str:
        """One line for the interface, e.g. `Item 12 of 400 · 11 of 400 saved`."""
        total, done = len(self.order), self.done_count()
        current = " (saved)" if self.order and self.order[index] in self.records else ""
        line = f"**Item {index + 1} of {total}**{current} · {done} of {total} saved"
        if total and done == total:
            line += " · all items are saved; you can still go back and change any of them"
        return line
