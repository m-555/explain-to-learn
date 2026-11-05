"""GEE's English explanations as edits of this project, for a side-by-side comparison.

Source
------
GEE (Song et al., 2024, "GEE! Grammar Error Explanation with Large Language Models",
Findings of NAACL) publishes its data at https://github.com/Yixiao-Song/GEE-with-LLMs,
which `scripts/download_data.sh` clones into `data/raw/gee/`. That repository has no
licence file, so none of its data is committed here: what quotes it stays in the
git-ignored `data/raw/` and `outputs/`, and only counts and shares are published.

The English explanations are in `data/human_annotation_data/en_anonymized.json`, a JSON
array with one record per sentence pair. The fields read here:

- `id`: the record's number.
- `text`: HTML with the source sentence, the target (corrected) sentence and one
  explanation per atomic edit, each followed by a line `Error type: <type>`:
  `<b>Source:</b> ...<br><b>Target:</b> ...<br><br><b>Explanation:</b><br>...`.
- `edits`: the atomic edits, separated by `<br>`, each a Python-style list
  `[operation, original words, corrected words]`, where the operation is `replace`,
  `insert`, `delete` or `relocate` and an empty string marks the missing side.
- `all_correct` (optional): present when the annotator found no error in the
  explanations.
- `label` (optional): spans the annotator highlighted, with `labels` such as
  "Wrong error explanation" or "Hallucinated error" and their position in `globalOffsets`
  (characters of `text` without tags, counting each `<br>` as one newline). Spans in the
  target sentence ("Missing error") are about edits, not explanations.

The explanations were written by the GEE pipeline (GPT-4, one call per atomic edit, in
the order of the edits; the prompt is in GEE's `code/prompts/en_prompt_utils.py`) and then
judged by one annotator. They are a comparison point, not a gold standard.

Conversion
----------
1. `parse_record` pairs the i-th explanation with the i-th atomic edit. A record whose
   numbers of explanations and edits differ cannot be paired safely and is skipped.
2. `match_edits` looks up the source sentence in the edit table. The English GEE
   sentences come from the W&I+LOCNESS development set, so the official BEA-2019 edits,
   ERRANT error types and the learner's CEFR level are already there, the same
   annotation that the rest of the project uses. Each atomic edit is paired with the
   official edit of that sentence with the same original and corrected words (ignoring
   spaces). GEE and ERRANT sometimes cut a change into different spans (GEE's
   `'' -> 'up'` and `'' -> ','` against ERRANT's `'' -> 'up ,'`, or any `relocate` edit);
   those atomic edits have no official counterpart and are dropped and counted.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from explain_to_learn.edits import PUNCT_NORMALISATION, Edit

GEE_FILE = Path("data/raw/gee/data/human_annotation_data/en_anonymized.json")
OPERATIONS = ("replace", "insert", "delete", "relocate")


@dataclass(frozen=True)
class GEEEdit:
    """One atomic edit of a GEE record with GEE's explanation of it."""

    operation: str
    original: str
    corrected: str
    explanation: str
    error_type: str  # GEE's own label, e.g. "noun number"
    flags: tuple[str, ...] = ()  # the annotator's labels on this explanation


@dataclass(frozen=True)
class GEERecord:
    """One GEE sentence pair with its explained atomic edits."""

    gee_id: str
    source: str
    target: str
    edits: tuple[GEEEdit, ...]
    all_correct: bool  # the annotator found no error in any of its explanations


@dataclass(frozen=True)
class GEEMatch:
    """An official edit from the edit table and the GEE edit that matches it."""

    edit: Edit
    gee_id: str
    reference: GEEEdit
    all_correct: bool


def parse_atomic_edit(text: str) -> tuple[str, str, str]:
    """Read `['replace', 'ditch', 'ditches']`; raises `ValueError` if it is malformed."""
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"unreadable atomic edit {text!r}") from error
    if (
        not isinstance(value, list | tuple)
        or len(value) != 3
        or not all(isinstance(part, str) for part in value)
        or value[0] not in OPERATIONS
    ):
        raise ValueError(f"unreadable atomic edit {text!r}")
    return value[0], value[1], value[2]


def plain_lines(text: str) -> list[tuple[int, str]]:
    """The lines of a record's `text` without tags, each with its start offset.

    The offsets count characters the way GEE's `globalOffsets` do: tags removed and each
    `<br>` counted as one newline.
    """
    plain = re.sub(r"</?b>", "", text.replace("<br>", "\n"))
    lines, offset = [], 0
    for line in plain.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1
    return lines


def parse_record(record: dict[str, Any]) -> GEERecord:
    """Read one record of the annotation file; raises `ValueError` if it cannot be used."""
    lines = plain_lines(record["text"])
    if len(lines) < 2:
        raise ValueError("no source and target sentence")
    source = lines[0][1].removeprefix("Source:").strip()
    target = lines[1][1].removeprefix("Target:").strip()

    explanations: list[list[Any]] = []  # [start offset, explanation, error type]
    started = False
    for offset, line in lines[2:]:
        line = line.strip()
        if line == "Explanation:":
            started = True
        elif started and line.startswith("Error type:"):
            if explanations and not explanations[-1][2]:
                explanations[-1][2] = line.removeprefix("Error type:").strip()
        elif started and line:
            explanations.append([offset, line, ""])

    atomic = [parse_atomic_edit(part) for part in record["edits"].split("<br>") if part.strip()]
    if not atomic:
        raise ValueError("no edits")
    if len(atomic) != len(explanations):
        raise ValueError("explanations and edits differ in number")

    flags: list[list[str]] = [[] for _ in explanations]
    for label in record.get("label", []):
        start = label["globalOffsets"]["start"]
        owners = [i for i, (offset, _, _) in enumerate(explanations) if offset <= start]
        if owners:  # spans before the first explanation are in the sentences
            flags[owners[-1]].extend(label["labels"])

    edits = tuple(
        GEEEdit(operation, original, corrected, text, error_type, tuple(flag))
        for (operation, original, corrected), (_, text, error_type), flag in zip(
            atomic, explanations, flags, strict=True
        )
    )
    return GEERecord(str(record["id"]), source, target, edits, "all_correct" in record)


def read_gee(path: str | Path = GEE_FILE) -> tuple[list[GEERecord], Counter]:
    """All usable records of the annotation file, and counts of the skipped ones."""
    records, counts = [], Counter()
    for raw in json.loads(Path(path).read_text(encoding="utf-8")):
        try:
            records.append(parse_record(raw))
        except ValueError as error:
            reason = str(error)
            counts["skipped: " + ("unreadable atomic edit" if "atomic" in reason else reason)] += 1
    counts["records used"] = len(records)
    return records, counts


def text_key(text: str) -> str:
    """Text as compared when matching: normalised punctuation and no whitespace.

    This is the comparison `edits.align_essays` uses, so `do n't` (M2 tokens) and
    `don't` (GEE) are the same.
    """
    return re.sub(r"\s", "", text.translate(PUNCT_NORMALISATION))


def match_edits(
    records: Iterable[GEERecord], edits: Iterable[Edit]
) -> tuple[list[GEEMatch], Counter]:
    """Pair GEE's atomic edits with the official edits of the same sentences.

    A sentence that occurs more than once in the edit table is taken from the essay whose
    correction equals GEE's target, otherwise from the first edit ID. Within a sentence,
    official edits are used once each, in sentence order, so two inserted commas match
    the two official comma edits in turn.
    """
    sentences: dict[str, dict[tuple[str, int], list[Edit]]] = defaultdict(lambda: defaultdict(list))
    for edit in edits:
        sentences[text_key(edit.original)][(edit.essay_key, edit.sent_idx)].append(edit)

    matches, counts, used = [], Counter(), set()
    for record in records:
        candidates = sentences.get(text_key(record.source))
        if not candidates:
            counts["sentences not in the edit table"] += 1
            counts["atomic edits dropped: sentence not in the edit table"] += len(record.edits)
            continue
        rows = min(
            candidates.values(),
            key=lambda rows: (
                text_key(rows[0].corrected) != text_key(record.target),
                rows[0].edit_id,
            ),
        )
        official = sorted(rows, key=lambda edit: (edit.start, edit.end, edit.edit_id))
        for atomic in record.edits:
            pair = (text_key(atomic.original), text_key(atomic.corrected))
            found = next(
                (
                    edit
                    for edit in official
                    if edit.edit_id not in used
                    and (text_key(edit.original_text), text_key(edit.corrected_text)) == pair
                ),
                None,
            )
            if found is None:
                counts[f"atomic edits dropped: {atomic.operation} with no official edit"] += 1
                continue
            used.add(found.edit_id)
            matches.append(GEEMatch(found, record.gee_id, atomic, record.all_correct))
            counts["atomic edits matched"] += 1
    return matches, counts


def reference_row(match: GEEMatch) -> dict[str, Any]:
    """One line of `outputs/gee/references.jsonl`."""
    reference = match.reference
    return {
        "edit_id": match.edit.edit_id,
        "gee_id": match.gee_id,
        "operation": reference.operation,
        "original": reference.original,
        "corrected": reference.corrected,
        "explanation": reference.explanation,
        "error_type": reference.error_type,
        "flags": list(reference.flags),
        "all_correct": match.all_correct,
    }
