"""Stubs for the demo's preview mode (`--fake`): no GPU, model files or ERRANT needed.

They stand in for the corrector, the ERRANT annotator and the explanation model with
the interfaces of `explain_to_learn.pipeline`, so the interface and the teacher summary
run on the real grammar graph while every model is replaced:

- `StubCorrector` fixes a few common errors with regular expressions;
- `StubAnnotator` splits sentences at end punctuation, takes the token differences
  between a sentence and its correction, and types them from a small table;
- `StubBackend` answers every prompt with the same placeholder JSON.
"""

from __future__ import annotations

import difflib
import re

from explain_to_learn.generate import Decoding
from explain_to_learn.pipeline import Span
from explain_to_learn.prompts import build_target

RULES = [
    (r"\b(he|she|it) don't\b", r"\1 doesn't"),
    (r"\bto going\b", "to go"),
    (r"\b(we|you|they) was\b", r"\1 were"),
    (r"\bI has\b", "I have"),
    (r"\ba (?=[aeiou])", "an "),
]
TYPES = {
    ("do", "does"): "R:VERB:SVA",
    ("was", "were"): "R:VERB:SVA",
    ("has", "have"): "R:VERB:SVA",
    ("going", "go"): "R:VERB:FORM",
    ("a", "an"): "R:DET",
}
TOKEN = re.compile(r"\w+(?=n't)|n't|\w+|[^\w\s]")
SENTENCE = re.compile(r"[^.!?]+[.!?]*")


class StubCorrector:
    """Fixes a few common errors with regular expressions."""

    def correct(self, sentences: list[str]) -> list[str]:
        corrected = []
        for sentence in sentences:
            for pattern, replacement in RULES:
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
            corrected.append(sentence)
        return corrected


class StubAnnotator:
    """Token differences between a sentence and its correction, typed from `TYPES`."""

    def sentences(self, text: str) -> list[str]:
        return [" ".join(s.split()) for s in SENTENCE.findall(text) if s.strip()]

    def annotate(self, original: str, corrected: str) -> tuple[list[str], list[str], list[Span]]:
        tokens, corrected_tokens = TOKEN.findall(original), TOKEN.findall(corrected)
        matcher = difflib.SequenceMatcher(a=tokens, b=corrected_tokens, autojunk=False)
        spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            blocks = [(i1, i2, j1, j2)]
            if tag == "replace" and i2 - i1 == j2 - j1:  # one edit per replaced token
                blocks = [
                    (i, i + 1, j, j + 1) for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                ]
            operation = {"replace": "R", "insert": "M", "delete": "U"}[tag]
            for start, end, c_start, c_end in blocks:
                before = " ".join(tokens[start:end])
                after = " ".join(corrected_tokens[c_start:c_end])
                error_type = TYPES.get((before.lower(), after.lower()), f"{operation}:OTHER")
                spans.append(Span(start, end, after, error_type))
        return tokens, corrected_tokens, spans


class StubBackend:
    """Answers every prompt with the same placeholder JSON."""

    def complete(self, messages: list[dict], decoding: Decoding) -> str:
        return build_target(
            "Stub concept",
            "Stub explanation: start the demo without --fake to see the model's feedback.",
            "Stub hint: start the demo without --fake to see the model's hint.",
        )
