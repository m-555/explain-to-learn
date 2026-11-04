"""Prompt templates for the explanation model.

Builds the model input from a sentence, one edit, its error type, the grammar graph
snippet (systems B and D only), the learner level and the mode (explain or hint).

The same functions build the inputs and targets for LoRA training, so the grounded
and ungrounded systems differ only in whether a snippet is passed.

Output format
-------------
The model replies with one JSON object with three string fields:

- `concept`: the name of the grammar concept (the graph's name when grounded).
- `explanation`: in explain mode, one or two sentences that name the original and
  corrected words and say why; in hint mode, an empty string.
- `hint`: in hint mode, one sentence that points to the error without giving the
  corrected words; in explain mode, an empty string.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from explain_to_learn.edits import Edit

Mode = Literal["explain", "hint"]
MODES: tuple[str, ...] = ("explain", "hint")

SYSTEM_PROMPT = """\
You are an English teacher who explains grammar corrections to learners of English.

You get a sentence written by a learner, one correction to it, its ERRANT error type, \
the learner's CEFR level and a mode. You may also get a rule from the teacher's \
grammar graph.

Reply with one JSON object and nothing else, in this form:
{"concept": "...", "explanation": "...", "hint": "..."}

Rules:
- "concept": a short name of the grammar concept. If a grammar rule is given, copy \
its concept name exactly.
- Mode "explain": write the explanation in one or two short sentences. Quote the \
original words and the corrected words (for a missing word, the word to add; for an \
unnecessary word, the word to remove) and say why the correction is right. Set \
"hint" to "".
- Mode "hint": write one short sentence that points the learner to the error and the \
rule, without giving the corrected words. Set "explanation" to "".
- Suit the wording to the level. A1 and A2: short sentences, everyday words, very \
few grammar terms. B1 and B2: common grammar terms. C1 and C2: precise grammar terms \
are fine.
- If a grammar rule is given, follow it and do not contradict it. Never invent a rule."""

# M2 tokens that attach to the previous token when text is shown to a reader.
_ATTACH_LEFT = re.compile(r" (n't|'s|'re|'ve|'ll|'m|'d|[.,!?;:%)\]}])")
_ATTACH_RIGHT = re.compile(r"([(\[{$]) ")

_OPERATION_VERBS = {"R": "replace", "M": "add", "U": "remove"}


def detokenize(tokens: str | Sequence[str]) -> str:
    """Join M2 tokens into readable text: `do n't go .` -> `don't go.`."""
    text = tokens if isinstance(tokens, str) else " ".join(tokens)
    text = _ATTACH_LEFT.sub(r"\1", " ".join(text.split()))
    return _ATTACH_RIGHT.sub(r"\1", text)


def prompt_level(cefr_level: str) -> str:
    """Level shown to the model; LOCNESS writers (`native`) are treated as C2."""
    return "C2" if cefr_level == "native" else cefr_level


def describe_edit(edit: Edit) -> str:
    """One line describing the edit, e.g. `replace "do" with "does" (tokens 1-2)`."""
    original = detokenize(edit.original_text)
    corrected = detokenize(edit.corrected_text)
    if edit.operation == "M":
        action = f'add "{corrected}"'
    elif edit.operation == "U":
        action = f'remove "{original}"'
    else:
        action = f'replace "{original}" with "{corrected}"'
    return f"{action} (tokens {edit.start}-{edit.end})"


def mark_edit(edit: Edit) -> str:
    """The original sentence with this one edit marked in brackets."""
    tokens = edit.original.split()
    if edit.operation == "M":
        mark = f"[+ {edit.corrected_text}]"
    elif edit.operation == "U":
        mark = f"[- {edit.original_text}]"
    else:
        mark = f"[{edit.original_text} -> {edit.corrected_text}]"
    return detokenize([*tokens[: edit.start], mark, *tokens[edit.end :]])


def format_snippet(snippet: dict) -> str:
    """Grammar graph snippet as plain text lines for the user message."""
    lines = [
        "Grammar rule (from the grammar graph):",
        f"  Concept: {snippet['name']}",
        f"  Rule: {snippet['rule']}",
    ]
    example = snippet.get("example") or {}
    if example:
        lines.append(f'  Example: wrong "{example["wrong"]}", right "{example["right"]}"')
    if snippet.get("prerequisites"):
        lines.append(f"  Builds on: {', '.join(snippet['prerequisites'])}")
    return "\n".join(lines)


def build_user_message(edit: Edit, level: str, mode: Mode, snippet: dict | None) -> str:
    """User message with the sentence, the edit, its type, level, mode and snippet."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    lines = [
        f"Learner sentence: {detokenize(edit.original)}",
        f"Correction: {describe_edit(edit)}",
        f"In context: {mark_edit(edit)}",
        f"Error type: {edit.error_type}",
        f"Learner level: {level}",
        f"Mode: {mode}",
    ]
    if snippet is not None:
        lines.append(format_snippet(snippet))
    return "\n".join(lines)


def build_messages(edit: Edit, level: str, mode: Mode, snippet: dict | None = None) -> list[dict]:
    """Chat messages for one edit; `snippet=None` gives the ungrounded input (A, C)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(edit, level, mode, snippet)},
    ]


def build_target(concept: str, explanation: str, hint: str) -> str:
    """The JSON string the model should output, used as the training target."""
    return json.dumps(
        {"concept": concept, "explanation": explanation, "hint": hint}, ensure_ascii=False
    )
