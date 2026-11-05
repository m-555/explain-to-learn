"""Blind ratings of a rating sheet, by the first rater and by the second rater.

The sheet comes from `scripts/make_rating_sheet.py`. Only its sheet columns are read
(item ID, level, mode, sentence, corrected sentence, edit and feedback). A file with
any other column, such as the key's `system`, is refused, so the view cannot show which
system wrote an item. The key file is never opened.

Ratings go to `<out-dir>/ratings_<rater>.csv` with the columns `item_id`, the four
dimensions, `invented_rule` and `comment`: the same columns as the file that
`scripts/llm_judge.py` writes, so `scripts/agreement.py` and
`explain_to_learn.human_eval` read it directly. Ratings are stored as the rubric's
words (`no`, `partly`, `yes`). Only saved items are written, in sheet order.

A small `.json` file next to the ratings records the rater's name and a fingerprint
of the sheet's items. If the sheet changes after rating has started (for example when
it is made again with `--force`), the task refuses to continue, instead of attaching
old ratings to new items.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path

from explain_to_learn.annotation.display import escape, quote
from explain_to_learn.annotation.store import Record, Store, write_atomic
from explain_to_learn.annotation.task import Task
from explain_to_learn.human_eval import (
    DIMENSIONS,
    FLAG,
    FLAG_VALUES,
    RATING_COLUMNS,
    SCALE,
    SHEET_COLUMNS,
    parse_rating,
)

OUTPUT_COLUMNS = ("item_id", *RATING_COLUMNS)
# Each option label starts with its keyboard shortcut, which is the rubric's number.
SCALE_CHOICES = [("0 no", "no"), ("1 partly", "partly"), ("2 yes", "yes")]
FLAG_CHOICES = [("0 no", "no"), ("1 yes", "yes")]
QUESTIONS = {
    "rule_correct": "Is the grammar it states true, and is it the reason for this correction?",
    "error_located": "Does it show the learner where the error is?",
    "helps_fix": "Would a learner at this level know what to do after reading it?",
    "fits_level": "Can a learner at this level understand it?",
    "invented_rule": "Does it state a grammar rule that does not exist in English?",
}
LABELS = {
    "rule_correct": "Rule correct",
    "error_located": "Error located",
    "helps_fix": "Helps the learner fix it",
    "fits_level": "Fits the level",
    "invented_rule": "Invented rule (flag)",
}
MODE_NOTES = {
    "explain": "an explanation, which should name the original and corrected words and say why",
    "hint": "a hint, which should point to the error and the rule without giving the answer",
}


def read_sheet(path: str | Path) -> list[dict[str, str]]:
    """Read a blind rating sheet and keep only the columns a rater may see.

    Raises `ValueError` for a file with columns that a rating sheet does not have
    (such as the key), a missing sheet column, no items or a repeated item ID.
    """
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    extra = [column for column in columns if column not in (*SHEET_COLUMNS, *RATING_COLUMNS)]
    if extra:
        raise ValueError(
            f"{path} has columns that a blind rating sheet does not have "
            f"({', '.join(extra)}). Open the rating sheet, never the key."
        )
    missing = [column for column in SHEET_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{path} is not a rating sheet: no column {', '.join(missing)}")
    if not rows:
        raise ValueError(f"{path} has no items")
    ids = [row["item_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists an item ID more than once")
    return [{column: row[column] or "" for column in SHEET_COLUMNS} for row in rows]


def sheet_fingerprint(items: list[dict[str, str]]) -> str:
    """A SHA-256 fingerprint of the sheet's items, the same for any copy of the sheet."""
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


def rater_slug(name: str) -> str:
    """The rater's name as a file-name part: `Anna B.` -> `anna_b`."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Enter a rater name made of letters or digits, for example rater2.")
    return slug


def ratings_path(out_dir: str | Path, rater: str) -> Path:
    """Where one rater's ratings are saved."""
    return Path(out_dir) / f"ratings_{rater_slug(rater)}.csv"


def read_rubric(path: str | Path) -> str:
    """The rubric text, or a note if the file is missing."""
    path = Path(path)
    if not path.is_file():
        return f"The rubric was not found at `{escape(str(path))}`."
    return path.read_text(encoding="utf-8")


def item_markdown(item: dict[str, str]) -> str:
    """What the rater sees of one item: the sheet fields only."""
    mode = item["mode"]
    return "\n".join(
        [
            f"**Level** {escape(item['level'])} · **Mode** {escape(mode)}: "
            f"{MODE_NOTES.get(mode, '')}",
            "",
            f"**Sentence:** {escape(item['sentence'])}",
            "",
            f"**Corrected:** {escape(item['corrected'])}",
            "",
            f"**Edit:** {escape(item['edit'])}",
            "",
            "**Feedback**",
            "",
            quote(item["feedback"]),
        ]
    )


def as_word(value: str | None, scale: dict[str, int]) -> str | None:
    """A stored rating (word or number) as the scale's word; None if empty."""
    number = parse_rating(value, scale)
    if number is None:
        return None
    return next(word for word, n in scale.items() if n == number)


def make_rating(
    item_id: str, ratings: Sequence[str | None], flag: str | None, comment: str | None
) -> Record:
    """One output row. A rating left open needs a comment that says why (rubric step 3)."""
    scores = {dim: as_word(value, SCALE) for dim, value in zip(DIMENSIONS, ratings, strict=True)}
    flag = as_word(flag, FLAG_VALUES)
    comment = (comment or "").strip()
    open_ratings = [dim for dim in DIMENSIONS if scores[dim] is None]
    if flag is None:
        open_ratings.append(FLAG)
    if open_ratings and not comment:
        names = ", ".join(LABELS[name].lower() for name in open_ratings)
        raise ValueError(f"Rate {names}, or say in the comment why you cannot decide.")
    return {
        "item_id": item_id,
        **{dim: scores[dim] or "" for dim in DIMENSIONS},
        FLAG: flag or "",
        "comment": comment,
    }


class RatingTask(Task):
    """Rate every item of one sheet, saved under one rater's name."""

    def __init__(self, sheet_path: str | Path, out_dir: str | Path, rater: str) -> None:
        self.rater = rater.strip()
        self.sheet_path = Path(sheet_path)
        self.items = read_sheet(sheet_path)
        self.fingerprint = sheet_fingerprint(self.items)
        path = ratings_path(out_dir, rater)
        self.info_path = path.with_suffix(".json")
        order = [item["item_id"] for item in self.items]
        self.store = Store(path, itemgetter("item_id"), order, OUTPUT_COLUMNS)
        self._check_same_sheet(set(order))

    def _check_same_sheet(self, item_ids: set[str]) -> None:
        """Refuse to continue ratings that were made on a different sheet."""
        changed = f"{self.store.path} holds ratings of a different rating sheet."
        advice = " Use another rater name for this sheet, or move the old file away."
        if self.info_path.is_file():
            info = json.loads(self.info_path.read_text(encoding="utf-8"))
            if info.get("sheet_fingerprint") != self.fingerprint:
                raise ValueError(changed + advice)
        if any(item_id not in item_ids for item_id in self.store.records):
            raise ValueError(changed + advice)

    def view(self, index: int) -> list[str]:
        return [item_markdown(self.items[index])]

    def values(self, index: int) -> list[str | None]:
        saved = self.saved(index) or {}
        ratings = [as_word(saved.get(dim), SCALE) for dim in DIMENSIONS]
        return [*ratings, as_word(saved.get(FLAG), FLAG_VALUES), saved.get("comment", "")]

    def save(self, index: int, *values: str | None) -> None:
        *ratings, flag, comment = values
        self.store.save(make_rating(self.items[index]["item_id"], ratings, flag, comment))
        info = {
            "rater": self.rater,
            "sheet": str(self.sheet_path),
            "sheet_fingerprint": self.fingerprint,
            "items": len(self),
            "saved": self.store.done_count(),
        }
        write_atomic(self.info_path, json.dumps(info, indent=2) + "\n")
