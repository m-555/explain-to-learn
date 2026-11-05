"""Teacher summary of one text: its edits grouped by grammar concept, in a teaching order.

For one learner text or essay, `summarize` groups the edits by the graph concept of
their ERRANT error type, with a count and the position of each edit in the text, and
lists the edits whose type has no concept. Examples are the learner's own edits, given
by position (sentence and word numbers) with the edited words; nothing new is written.

Teaching order
--------------
Concepts are taken by priority: more edits first, then the concept whose first edit
comes earlier in the text, then the concept ID. Just before each concept come the
concepts it builds on (directly or through a chain of prerequisites in the graph) that
also have errors in this text, in the same priority order. So a concept always comes
after its prerequisites with errors, and the most frequent problems come early. The
order is fixed for a given text and graph, and no model is involved.

`markdown` formats the summary for a teacher.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from explain_to_learn.edits import Edit
from explain_to_learn.prompts import detokenize


def position(edit: Edit) -> dict:
    """Where an edit is: sentence and word numbers counted from 1, and its words."""
    return {
        "edit_id": edit.edit_id,
        "sentence": edit.sent_idx + 1,
        "start": edit.start,
        "end": edit.end,
        "original": detokenize(edit.original_text),
        "correction": detokenize(edit.corrected_text),
        "error_type": edit.error_type,
    }


def ancestors(graph: Any, concept_id: str) -> set[str]:
    """Every concept that `concept_id` builds on, directly or through prerequisites."""
    found: set[str] = set()
    stack = list(graph.concepts[concept_id].prerequisites)
    while stack:
        current = stack.pop()
        if current in found or current not in graph.concepts:
            continue
        found.add(current)
        stack.extend(graph.concepts[current].prerequisites)
    return found


def teaching_order(groups: dict[str, list[Edit]], graph: Any) -> list[str]:
    """Concept IDs in the suggested teaching order (see the module docstring)."""

    def priority(concept_id: str) -> tuple:
        first = min((e.sent_idx, e.start) for e in groups[concept_id])
        return (-len(groups[concept_id]), first, concept_id)

    before = {c: ancestors(graph, c) & groups.keys() for c in groups}
    order: list[str] = []

    def place(concept_id: str, visiting: set[str]) -> None:
        # `visiting` stops a prerequisite cycle, which graph validation reports.
        if concept_id in order or concept_id in visiting:
            return
        visiting.add(concept_id)
        for prerequisite in sorted(before[concept_id], key=priority):
            place(prerequisite, visiting)
        order.append(concept_id)

    for concept_id in sorted(groups, key=priority):
        place(concept_id, set())
    return order


def summarize(
    edits: Sequence[Edit], graph: Any, level: str | None = None, max_examples: int = 5
) -> dict:
    """The teacher summary of one text's edits as a dictionary.

    `graph` is a `GrammarGraph`. With a `level`, each concept also carries its rule
    worded for that level. At most `max_examples` positions are listed per concept.
    """
    groups: dict[str, list[Edit]] = defaultdict(list)
    without_concept: list[Edit] = []
    for edit in sorted(edits, key=lambda e: (e.sent_idx, e.start, e.end)):
        concept_id = graph.concept_for(edit.error_type)
        if concept_id:
            groups[concept_id].append(edit)
        else:
            without_concept.append(edit)

    concepts = []
    for step, concept_id in enumerate(teaching_order(groups, graph), start=1):
        concept = graph.concepts[concept_id]
        group = groups[concept_id]
        builds_on = sorted(ancestors(graph, concept_id) & groups.keys())
        concepts.append(
            {
                "step": step,
                "concept_id": concept_id,
                "name": concept.name or concept_id,
                "count": len(group),
                "error_types": sorted({e.error_type for e in group}),
                "builds_on": [graph.concepts[c].name or c for c in builds_on],
                "rule": graph.snippet(concept_id, level)["rule"] if level else "",
                "examples": [position(e) for e in group[:max_examples]],
            }
        )
    return {
        "level": level,
        "edits": len(edits),
        "sentences_with_edits": len({e.sent_idx for e in edits}),
        "concepts": concepts,
        "without_concept": [position(e) for e in without_concept],
    }


def describe_position(example: dict) -> str:
    """`sentence 2, words 3-4: *do* → *does*` (or `add` / `remove`)."""
    start, end = example["start"], example["end"]
    if start == end:
        where = f"before word {start + 1}"
    elif end - start == 1:
        where = f"word {start + 1}"
    else:
        where = f"words {start + 1}-{end}"
    original, correction = example["original"], example["correction"]
    if not original:
        change = f"add *{correction}*"
    elif not correction:
        change = f"remove *{original}*"
    else:
        change = f"*{original}* → *{correction}*"
    return f"sentence {example['sentence']}, {where}: {change}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def markdown(summary: dict) -> str:
    """The summary as short Markdown for a teacher."""
    concepts, unmapped = summary["concepts"], summary["without_concept"]
    header = [_plural(summary["edits"], "edit"), _plural(len(concepts), "concept")]
    if summary["level"]:
        header.insert(0, f"Level {summary['level']}")
    lines = ["# Teacher summary", "", " · ".join(header), ""]
    if not summary["edits"]:
        return "\n".join([*lines, "No corrections in this text."]) + "\n"
    if concepts:
        lines += [
            "## Suggested teaching order",
            "",
            "A concept comes after the concepts it builds on that also have errors here; "
            "otherwise concepts with more errors come first.",
            "",
        ]
    for concept in concepts:
        title = f"{concept['step']}. **{concept['name']}** · {_plural(concept['count'], 'edit')}"
        lines.append(title)
        if concept["builds_on"]:
            lines.append(f"   - Builds on: {', '.join(concept['builds_on'])}")
        if concept["rule"]:
            lines.append(f"   - Rule ({summary['level']}): {concept['rule']}")
        lines += [f"   - {describe_position(example)}" for example in concept["examples"]]
        hidden = concept["count"] - len(concept["examples"])
        if hidden:
            lines.append(f"   - and {_plural(hidden, 'more edit')}")
    if unmapped:
        lines += ["", "## Edits without a grammar concept", ""]
        lines += [f"- {describe_position(e)} ({e['error_type']})" for e in unmapped]
    return "\n".join(lines) + "\n"


def essay_edits(edits: Iterable[Edit], essay_key: str) -> list[Edit]:
    """The edits of one essay of the edit table, e.g. `wi-1-352420`."""
    return [edit for edit in edits if edit.essay_key == essay_key]
