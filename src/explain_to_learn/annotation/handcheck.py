"""Hand-check of silver training explanations.

`scripts/make_silver.py filter` writes a random sample of kept silver items to
`outputs/silver_handcheck.csv` with an empty `correct` column. This task shows the items
one by one, and the teacher marks each one correct or not, with an optional note.

Marks go to `<out-dir>/silver_handcheck.csv`: the hand-check columns plus `note`, with
`correct` set to `1` or `0`, for the items checked so far. `silver.handcheck_accuracy`
reads this file, so the task shows the accuracy as it goes, and
`python scripts/make_silver.py score --handcheck outputs/annotations/silver_handcheck.csv`
reports it at the end.

The marks live in their own file because running the filter step again rewrites
`outputs/silver_handcheck.csv`. If an item that was already marked now has a different
text, or is no longer in the sample, the task stops instead of keeping a mark for a
text the teacher never saw.
"""

from __future__ import annotations

from operator import itemgetter
from pathlib import Path
from typing import Any

from explain_to_learn.annotation.display import escape, quote
from explain_to_learn.annotation.store import Store, read_records
from explain_to_learn.annotation.task import Task
from explain_to_learn.silver import HANDCHECK_FIELDS, handcheck_accuracy

OUTPUT_FILE = "silver_handcheck.csv"
OUTPUT_COLUMNS = (*HANDCHECK_FIELDS, "note")
CONTENT_FIELDS = tuple(field for field in HANDCHECK_FIELDS if field != "correct")
# Each option label starts with its keyboard shortcut; values are what
# `handcheck_accuracy` reads as correct (1) and not correct (0).
MARK_CHOICES = [("1 correct", "1"), ("0 not correct", "0")]
CRITERION = (
    "Mark an item **correct** if you would give this feedback to a learner at this level "
    "as it is: the rule is true and is the reason for this correction, the concept fits, "
    "and the text does what its mode asks (an explanation names the words and says why; "
    "a hint points to the error without giving the answer)."
)


def read_handcheck(path: str | Path) -> list[dict[str, str]]:
    """Read the hand-check CSV written by `silver.export_handcheck`."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} not found; run scripts/make_silver.py filter first")
    rows = read_records(path)
    if not rows:
        raise ValueError(f"{path} has no items")
    missing = [field for field in CONTENT_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"{path} is not a hand-check file: no column {', '.join(missing)}")
    ids = [row["edit_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists an edit more than once")
    return [{field: row[field] or "" for field in CONTENT_FIELDS} for row in rows]


class HandcheckTask(Task):
    """Mark a sample of silver items correct or not."""

    def __init__(self, handcheck_path: str | Path, out_dir: str | Path) -> None:
        if (Path(out_dir) / OUTPUT_FILE).resolve() == Path(handcheck_path).resolve():
            raise ValueError(
                "The marks would overwrite the hand-check file; use another --out-dir."
            )
        self.items = read_handcheck(handcheck_path)
        order = [item["edit_id"] for item in self.items]
        self.store = Store(
            Path(out_dir) / OUTPUT_FILE, itemgetter("edit_id"), order, OUTPUT_COLUMNS
        )
        by_id = {item["edit_id"]: item for item in self.items}
        for edit_id, record in self.store.records.items():
            item = by_id.get(edit_id)
            if item is None or any(record[field] != item[field] for field in CONTENT_FIELDS):
                raise ValueError(
                    f"{self.store.path} has marks for items that are no longer in "
                    f"{handcheck_path} as they were checked (first: {edit_id}). Move the "
                    "old marks file away to check the new sample."
                )

    def accuracy(self) -> dict[str, Any] | None:
        """`silver.handcheck_accuracy` of the marks so far; None before the first mark."""
        if not self.store.records:
            return None
        return handcheck_accuracy(self.store.path)

    def status(self, index: int) -> str:
        """The progress line, with the accuracy of the marks saved so far."""
        line = super().status(index)
        result = self.accuracy()
        if result and result["accuracy"] is not None:
            line += (
                f" · accuracy so far {result['accuracy']:.1%} "
                f"(95% CI {result['low']:.1%}–{result['high']:.1%})"
            )
        return line

    def view(self, index: int) -> list[str]:
        item = self.items[index]
        mode = item["mode"]
        text = item["hint"] if mode == "hint" else item["explanation"]
        change = f"{escape(item['original_text']) or '–'} → {escape(item['corrected_text']) or '–'}"
        return [
            "\n".join(
                [
                    f"**Level** {escape(item['level'])} · **Mode** {escape(mode)} · "
                    f"**Error type** `{item['error_type']}`",
                    "",
                    f"**Sentence:** {escape(item['sentence'])}",
                    "",
                    f"**Edit:** {change}",
                    "",
                    f"**Concept:** {escape(item['concept'])}",
                    "",
                    f"**{'Hint' if mode == 'hint' else 'Explanation'}**",
                    "",
                    quote(text),
                ]
            )
        ]

    def values(self, index: int) -> list[str | None]:
        saved = self.saved(index) or {}
        return [saved.get("correct") or None, saved.get("note", "")]

    def save(self, index: int, correct: str | None, note: str) -> None:
        if correct not in ("1", "0"):
            raise ValueError("Mark the item correct or not correct.")
        item = self.items[index]
        self.store.save({**item, "correct": correct, "note": (note or "").strip()})
