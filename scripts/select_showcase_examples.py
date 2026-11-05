"""Propose candidates for the README's five good and five bad outputs.

Joins the first rater's blind ratings with the answer key and the rating sheet that was
rated, adds each edit's position from the edit table, and writes candidate examples to
`outputs/showcase_candidates.md` for the teacher to choose from and comment on.

- Good: every dimension rated "yes" and no invented rule. System D comes first.
- Bad: an invented rule, a wrong rule, an error that is not located, or feedback that
  does not help the learner fix the error. The more serious the failure, the higher
  the candidate ranks: an invented rule, then a wrong rule, then unhelpful feedback.
- Candidates are spread over error categories first. A system-D candidate is shown
  next to system A's output for the same edit and mode when that was rated too.
- Only W&I edits are used: the W&I licence allows published excerpts of under 100
  words with a citation, while LOCNESS allows no redistribution. Each example shows at
  most `--window` words of context around the edit.

The output quotes learner text, so it stays in `outputs/` (git-ignored). The teacher
decides what, if anything, appears in the README.

Usage:
    python scripts/select_showcase_examples.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.human_eval import DIMENSIONS, FLAG, join_key, read_ratings
from explain_to_learn.io import read_csv
from explain_to_learn.prompts import detokenize

SHOWCASE_CORPUS = "wi"
SCALE_NAMES = {0: "no", 1: "partly", 2: "yes"}
FLAG_NAMES = {0: "no", 1: "yes"}
DIMENSION_LABELS = {
    "rule_correct": "rule correct",
    "error_located": "error located",
    "helps_fix": "helps fix",
    "fits_level": "fits level",
}

Item = dict[str, Any]


def is_complete(item: Item) -> bool:
    """True if every dimension and the invented-rule flag were rated."""
    return all(item[name] is not None for name in (*DIMENSIONS, FLAG))


def is_good(item: Item) -> bool:
    """Every dimension rated "yes" and no invented rule."""
    return all(item[name] == 2 for name in DIMENSIONS) and item[FLAG] == 0


def badness(item: Item) -> int:
    """How serious an item's failure is; 0 means none of the failures counted here."""
    score = 0
    if item[FLAG] == 1:
        score += 8
    if item["rule_correct"] == 0:
        score += 4
    if item["helps_fix"] == 0:
        score += 2
    if item["error_located"] == 0:
        score += 1
    return score


def spread(items: Sequence[Item], n: int) -> list[Item]:
    """Up to `n` items in the given order, first one per error category, then the rest."""
    chosen: list[Item] = []
    categories: set[str] = set()
    for item in items:
        if len(chosen) < n and item["error_category"] not in categories:
            chosen.append(item)
            categories.add(item["error_category"])
    for item in items:
        if len(chosen) < n and item not in chosen:
            chosen.append(item)
    return chosen


def window(edit: Edit, size: int) -> str:
    """The edit marked as [original → corrected], with at most `size` words around it."""
    tokens = edit.original.split()
    original = detokenize(edit.original_text) or "∅"
    corrected = detokenize(edit.corrected_text) or "∅"
    left = max(edit.start - size // 2, 0)
    right = min(edit.end + (size - (edit.start - left)), len(tokens))
    text = detokenize([*tokens[left : edit.start], "\0", *tokens[edit.end : right]])
    text = text.replace("\0", f"[{original} → {corrected}]")
    return ("… " if left > 0 else "") + text + (" …" if right < len(tokens) else "")


def ratings_line(item: Item) -> str:
    """The ratings of an item in words."""
    parts = [f"{DIMENSION_LABELS[name]} {SCALE_NAMES[item[name]]}" for name in DIMENSIONS]
    parts.append(f"invented rule {FLAG_NAMES[item[FLAG]]}")
    return " · ".join(parts)


def load_items(ratings: Path, key: Path, sheet: Path) -> list[Item]:
    """Complete ratings joined with the key and the feedback shown on the rating sheet."""
    shown = {row["item_id"]: row for row in read_csv(sheet)}
    items = join_key(read_ratings(ratings), read_csv(key))
    return [
        item | {"feedback": shown[item["item_id"]]["feedback"]}
        for item in items
        if is_complete(item) and item["item_id"] in shown
    ]


def select(items: Iterable[Item], edits: dict[str, Edit], n: int) -> tuple[list, list]:
    """Good and bad candidates among the W&I items, `n` of each at most."""
    usable = [i for i in items if i["edit_id"] in edits and i["edit_id"].startswith("wi-")]
    good = sorted(
        (i for i in usable if is_good(i)), key=lambda i: (i["system"] != "D", i["edit_id"])
    )
    bad = sorted(
        (i for i in usable if badness(i) > 0),
        key=lambda i: (-badness(i), i["system"] != "D", i["edit_id"]),
    )
    return spread(good, n), spread(bad, n)


def contrast(item: Item, items: Iterable[Item]) -> Item | None:
    """System A's rated output for the same edit and mode as a system-D item."""
    if item["system"] != "D":
        return None
    for other in items:
        if other["system"] == "A" and (other["edit_id"], other["mode"]) == (
            item["edit_id"],
            item["mode"],
        ):
            return other
    return None


def entry(number: int, item: Item, edit: Edit, items: list[Item], size: int) -> str:
    """One candidate as Markdown, with an empty field for the teacher's comment."""
    lines = [
        f"### {number}. {item['error_category']} · {item['level']} · {item['mode']} · "
        f"system {item['system']} (`{item['edit_id']}`)",
        "",
        f"> {window(edit, size)}",
        "",
        f"**Output:** {item['feedback']}",
        "",
        f"**Ratings:** {ratings_line(item)}",
        "",
    ]
    other = contrast(item, items)
    if other is not None:
        lines += [
            f"**System A on the same edit:** {other['feedback']}",
            "",
            f"**System A ratings:** {ratings_line(other)}",
            "",
        ]
    lines += ["**Teacher comment:**", ""]
    return "\n".join(lines)


def report(good: list[Item], bad: list[Item], items: list[Item], edits, size: int) -> str:
    """The candidates file."""
    parts = [
        "# Showcase candidates",
        "",
        "Proposed from the blind ratings. Choose five good and five bad outputs for the "
        "README and write a comment on each. W&I edits only: the W&I licence allows "
        "excerpts of under 100 words with a citation of Yannakoudakis et al. (2018); "
        "LOCNESS allows no redistribution. This file quotes learner text: keep it local.",
        "",
        f"## Good candidates ({len(good)})",
        "",
    ]
    parts += [entry(n, item, edits[item["edit_id"]], items, size) for n, item in enumerate(good, 1)]
    parts += [f"## Bad candidates ({len(bad)})", ""]
    parts += [entry(n, item, edits[item["edit_id"]], items, size) for n, item in enumerate(bad, 1)]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratings", type=Path, default=Path("outputs/annotations/ratings_rater1.csv")
    )
    parser.add_argument("--key", type=Path, default=Path("outputs/human_eval/key.csv"))
    parser.add_argument("--sheet", type=Path, default=Path("outputs/human_eval/rating_sheet.csv"))
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/showcase_candidates.md"))
    parser.add_argument("--n", type=int, default=10, help="candidates of each kind")
    parser.add_argument("--window", type=int, default=12, help="words of context")
    args = parser.parse_args(argv)

    if not args.ratings.is_file():
        print(f"No ratings yet ({args.ratings}); rate the sheet in the annotation app first.")
        return 0
    for path, hint in ((args.key, "make_rating_sheet.py"), (args.sheet, "make_rating_sheet.py")):
        if not path.is_file():
            raise SystemExit(f"{path} not found; it is written by scripts/{hint}")
    if not args.edits.is_file():
        raise SystemExit(f"{args.edits} not found; build it with `explain_to_learn.edits build`")

    items = load_items(args.ratings, args.key, args.sheet)
    edits = {e.edit_id: e for e in read_jsonl(args.edits) if e.corpus == SHOWCASE_CORPUS}
    good, bad = select(items, edits, args.n)
    if not good and not bad:
        print(f"No complete W&I ratings in {args.ratings} yet; nothing written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(good, bad, items, edits, args.window), encoding="utf-8")
    print(f"Wrote {len(good)} good and {len(bad)} bad candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
