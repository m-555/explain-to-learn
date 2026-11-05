"""What every annotation task shares: items in a fixed order, a store and navigation.

The interface (`app.py`) only calls the methods below, so each task can be tested
without a browser: `view` gives the Markdown panels for an item, `values` the form
values (the saved ones, or defaults for a new item), and `save` checks the form values
and stores them. `save` raises `ValueError` with a message for the annotator when the
values are not complete.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from explain_to_learn.annotation.store import Record, Store


class Task:
    """Base class of the annotation tasks."""

    store: Store

    def __len__(self) -> int:
        return len(self.store.order)

    def key(self, index: int) -> Hashable:
        """The key of the item at `index`."""
        return self.store.order[index]

    def saved(self, index: int) -> Record | None:
        """The saved record of the item at `index`, or None."""
        return self.store.get(self.key(index))

    def start_index(self) -> int:
        """Where to resume: the first item that has not been saved."""
        return self.store.first_open()

    def status(self, index: int) -> str:
        """The progress line shown above the item."""
        return self.store.progress(index)

    def view(self, index: int) -> list[str]:
        """The Markdown panels that show the item."""
        raise NotImplementedError

    def values(self, index: int) -> list[Any]:
        """The form values for the item: the saved ones, or defaults."""
        raise NotImplementedError

    def save(self, index: int, *values: Any) -> None:
        """Check the form values and save them as the item's record."""
        raise NotImplementedError
