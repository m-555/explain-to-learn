"""Markdown helpers for the item views."""

from __future__ import annotations

import re

from explain_to_learn.edits import Edit
from explain_to_learn.prompts import detokenize

_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~])")
_PLACEHOLDER = "\x00"


def escape(text: str) -> str:
    """Escape Markdown characters, so that text is shown exactly as written."""
    return _SPECIAL.sub(r"\\\1", text)


def quote(text: str, empty: str = "*(empty)*") -> str:
    """Text as a Markdown block quote, with a marker if it is empty."""
    lines = escape(text).splitlines() or [empty]
    return "\n".join(f"> {line}" for line in lines)


def edit_change(edit: Edit) -> str:
    """The change of one edit: `do → does`, `+ a` (missing) or `− very` (unnecessary)."""
    original = escape(detokenize(edit.original_text))
    corrected = escape(detokenize(edit.corrected_text))
    if edit.operation == "M":
        return f"+ {corrected}"
    if edit.operation == "U":
        return f"− {original}"
    return f"{original} → {corrected}"


def marked_sentence(edit: Edit) -> str:
    """The original sentence as readable text, with the edit in bold brackets."""
    tokens = edit.original.split()
    text = detokenize([*tokens[: edit.start], _PLACEHOLDER, *tokens[edit.end :]])
    return escape(text).replace(_PLACEHOLDER, f"[**{edit_change(edit)}**]")
