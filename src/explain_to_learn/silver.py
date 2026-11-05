"""Silver training explanations: select training edits, filter teacher outputs, hand-check.

The teacher model (system R, with the grammar graph) writes an explanation or hint
for 5,000 to 10,000 training edits. An output is kept only if:

1. it parses (`invalid_json`) and the text for its mode is not empty (`empty_text`);
2. the edit's error type has a graph concept (`no_concept`) and the output names
   that concept (`wrong_concept`);
3. it names the edited words as the mode requires (`missing_original`,
   `missing_correction`, `reveals_correction`; see `metrics.faithfulness`).

The first failed check is the rejection reason. A random sample of kept items is
exported for a teacher to mark as correct or not, and `handcheck_accuracy` reads the
marks back.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from explain_to_learn.edits import Edit
from explain_to_learn.io import read_csv, write_csv
from explain_to_learn.metrics import concept_matches, faithfulness, shown_text
from explain_to_learn.prompts import build_target, detokenize

REJECTION_REASONS = (
    "invalid_json",
    "empty_text",
    "no_concept",
    "wrong_concept",
    "missing_original",
    "missing_correction",
    "reveals_correction",
)
HANDCHECK_FIELDS = (
    "edit_id",
    "level",
    "mode",
    "error_type",
    "sentence",
    "original_text",
    "corrected_text",
    "concept",
    "explanation",
    "hint",
    "correct",
)
YES = {"1", "y", "yes", "true", "correct"}
NO = {"0", "n", "no", "false", "incorrect", "wrong"}


def select_training_edits(
    edits: Iterable[Edit],
    train_essays: set[str],
    n: int,
    seed: int = 0,
    exclude_ids: Iterable[str] = (),
    has_concept: Callable[[str], bool] | None = None,
) -> list[Edit]:
    """A seeded random sample of `n` edits from training essays, sorted by edit ID.

    `has_concept(error_type)` can drop edits whose type has no graph concept, since
    their outputs would be rejected anyway.
    """
    exclude = set(exclude_ids)
    pool = [
        edit
        for edit in edits
        if edit.essay_key in train_essays
        and edit.edit_id not in exclude
        and (has_concept is None or has_concept(edit.error_type))
    ]
    pool.sort(key=lambda edit: edit.edit_id)
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    return sorted(sample, key=lambda edit: edit.edit_id)


def assign_modes(edit_ids: Iterable[str], hint_share: float = 0.25, seed: int = 0) -> dict:
    """Give each edit the mode `hint` with probability `hint_share`, else `explain`."""
    rng = random.Random(seed)
    return {
        edit_id: "hint" if rng.random() < hint_share else "explain" for edit_id in sorted(edit_ids)
    }


def rejection_reason(
    edit: Edit, mode: str, parsed: dict | None, concept_labels: Iterable[str] | None
) -> str | None:
    """Why a teacher output is rejected, or None if it is kept."""
    if parsed is None:
        return "invalid_json"
    text = shown_text(parsed, mode)
    if not text:
        return "empty_text"
    if concept_labels is None:
        return "no_concept"
    if not concept_matches(parsed["concept"], concept_labels):
        return "wrong_concept"
    return faithfulness(edit, mode, text).reason


def filter_silver(rows: Iterable[dict], edits_by_id: Mapping[str, Edit]) -> tuple[list, Counter]:
    """Split teacher outputs into kept training items and rejection counts.

    Each row is a generation record with `edit_id`, `mode`, `parsed`, and the
    graph's `concept_id` and `concept_name` (None if the type has no concept). Kept
    items use the graph's concept name and an empty string for the unused field, and
    carry the training `target` built with `prompts.build_target`.
    """
    kept, counts = [], Counter()
    for row in rows:
        edit, mode, parsed = edits_by_id[row["edit_id"]], row["mode"], row["parsed"]
        labels = None
        if row.get("concept_id"):
            labels = {row["concept_id"], row.get("concept_name") or row["concept_id"]}
        reason = rejection_reason(edit, mode, parsed, labels)
        counts[reason or "kept"] += 1
        if reason is not None:
            continue
        concept = row.get("concept_name") or row["concept_id"]
        explanation = parsed["explanation"] if mode == "explain" else ""
        hint = parsed["hint"] if mode == "hint" else ""
        kept.append(
            {
                "edit_id": edit.edit_id,
                "level": row.get("level", edit.cefr_level),
                "mode": mode,
                "concept_id": row["concept_id"],
                "concept": concept,
                "explanation": explanation,
                "hint": hint,
                "target": build_target(concept, explanation, hint),
            }
        )
    return kept, counts


def counts_table(counts: Counter) -> str:
    """Markdown table of kept and rejected counts (no learner text)."""
    total = sum(counts.values())
    lines = [
        "| Outcome | Items | Share |",
        "| --- | ---: | ---: |",
    ]
    for outcome in ("kept", *REJECTION_REASONS):
        share = f"{100 * counts[outcome] / total:.1f}%" if total else "-"
        lines.append(f"| {outcome} | {counts[outcome]} | {share} |")
    lines.append(f"| total | {total} | |")
    return "\n".join(lines) + "\n"


def export_handcheck(
    kept: list[dict],
    edits_by_id: Mapping[str, Edit],
    path: str | Path,
    n: int = 200,
    seed: int = 0,
) -> int:
    """Write a random sample of kept items to CSV with an empty `correct` column.

    The CSV quotes learner text, so it belongs in the git-ignored `outputs/`.
    Returns the number of rows written.
    """
    sample = random.Random(seed).sample(kept, min(n, len(kept)))
    rows = []
    for item in sample:
        edit = edits_by_id[item["edit_id"]]
        rows.append(
            {
                "edit_id": item["edit_id"],
                "level": item["level"],
                "mode": item["mode"],
                "error_type": edit.error_type,
                "sentence": detokenize(edit.original),
                "original_text": detokenize(edit.original_text),
                "corrected_text": detokenize(edit.corrected_text),
                "concept": item["concept"],
                "explanation": item["explanation"],
                "hint": item["hint"],
                "correct": "",
            }
        )
    write_csv(rows, path, HANDCHECK_FIELDS)
    return len(sample)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def handcheck_accuracy(path: str | Path) -> dict:
    """Share of rated items marked correct, with a Wilson 95% interval.

    The `correct` column accepts 1/0, yes/no, y/n or true/false; blank rows are
    counted as unrated. Any other value raises `ValueError`.
    """
    rated = correct = unrated = 0
    for row in read_csv(path):
        mark = (row.get("correct") or "").strip().lower()
        if not mark:
            unrated += 1
        elif mark in YES:
            rated, correct = rated + 1, correct + 1
        elif mark in NO:
            rated += 1
        else:
            raise ValueError(
                f"Unknown mark {mark!r} for {row['edit_id']}; mark items 1 (correct) or 0"
            )
    low, high = wilson_interval(correct, rated)
    return {
        "rated": rated,
        "correct": correct,
        "unrated": unrated,
        "accuracy": correct / rated if rated else None,
        "low": low,
        "high": high,
    }
