"""The demo pipeline: correction, ERRANT edits, grammar graph lookup, explanation.

1. A corrector rewrites the learner's sentence: an open grammatical error correction
   (GEC) model, or the explanation model's backend prompted to correct.
2. ERRANT aligns the sentence with its correction and splits the change into typed
   edits.
3. For each edit, `generate.edit_item` looks up the grammar graph concept and snippet
   and builds the prompt, exactly as for the evaluated systems.
4. The explanation model (system D by default: the LoRA-tuned model with the graph
   snippet) answers with JSON, and the learner sees the explanation or the hint.

Every step is a small object with one method, so the demo runs either with real
models or, with `fake_pipeline()`, with stubs that need no GPU, model files or ERRANT.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from explain_to_learn.edits import Edit
from explain_to_learn.generate import Backend, Decoding, edit_item, generate
from explain_to_learn.metrics import shown_text
from explain_to_learn.prompts import build_target, detokenize

GEC_MODEL = "grammarly/coedit-large"
GEC_INSTRUCTION = "Fix grammatical errors in this sentence: "
CORRECTION_PROMPT = (
    "Correct the grammar of the learner's sentence. Change as little as possible and keep "
    "the learner's words where they are correct. Reply with the corrected sentence only."
)


class Corrector(Protocol):
    def correct(self, sentence: str) -> str: ...


class EditExtractor(Protocol):
    def edits(self, sentence: str, corrected: str, level: str) -> list[Edit]: ...


def make_edits(
    tokens: list[str],
    corrected_tokens: list[str],
    spans: list[tuple[int, int, str, str]],
    level: str,
) -> list[Edit]:
    """Edit rows (as in the edit table) for one sentence.

    `spans` holds (start, end, correction, error type) for each edit, with token offsets
    into `tokens`. The learner's level is stored as `cefr_level`, which is where
    `generate.edit_item` reads it from.
    """
    edits = []
    for idx, (start, end, correction, error_type) in enumerate(spans):
        operation, _, category = error_type.partition(":")
        edits.append(
            Edit(
                edit_id=f"demo-0-{idx}",
                corpus="demo",
                essay_id="0",
                split="demo",
                cefr_level=level,
                sent_idx=0,
                original=" ".join(tokens),
                corrected=" ".join(corrected_tokens),
                start=start,
                end=end,
                original_text=" ".join(tokens[start:end]),
                corrected_text=correction,
                error_type=error_type,
                operation=operation,
                error_category=category,
            )
        )
    return edits


class GecCorrector:
    """An open sequence-to-sequence GEC model; by default CoEdIT large (CC BY-NC 4.0)."""

    def __init__(self, model_name: str = GEC_MODEL, instruction: str = GEC_INSTRUCTION):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    def correct(self, sentence: str) -> str:
        inputs = self.tokenizer(self.instruction + sentence, return_tensors="pt")
        output = self.model.generate(
            **inputs.to(self.model.device), max_new_tokens=128, num_beams=4
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


@dataclass
class ModelCorrector:
    """Ask a chat model (for example the explanation model's backend) to correct."""

    backend: Backend
    decoding: Decoding = field(default_factory=Decoding)

    def correct(self, sentence: str) -> str:
        messages = [
            {"role": "system", "content": CORRECTION_PROMPT},
            {"role": "user", "content": sentence},
        ]
        return self.backend.complete(messages, self.decoding).strip() or sentence


class ErrantExtractor:
    """ERRANT edits between a sentence and its correction (needs the `data` extra)."""

    def __init__(self, language: str = "en"):
        import errant

        self.annotator = errant.load(language)

    def edits(self, sentence: str, corrected: str, level: str) -> list[Edit]:
        original = self.annotator.parse(sentence, tokenise=True)
        correction = self.annotator.parse(corrected, tokenise=True)
        spans = [
            (e.o_start, e.o_end, e.c_str, e.type)
            for e in self.annotator.annotate(original, correction)
            if e.type not in ("noop", "UNK")
        ]
        return make_edits([t.text for t in original], [t.text for t in correction], spans, level)


@dataclass
class Feedback:
    """What the learner sees for one edit."""

    original_text: str
    corrected_text: str
    error_type: str
    concept: str
    text: str  # the explanation, or the hint in hint mode; empty if unreadable
    grounded: bool


@dataclass
class Result:
    corrected: str
    feedback: list[Feedback]


@dataclass
class Pipeline:
    corrector: Corrector
    extractor: EditExtractor
    graph: Any
    backend: Backend
    grounded: bool = True
    decoding: Decoding = field(default_factory=Decoding)

    def run(self, sentence: str, level: str, mode: str) -> Result:
        """Correct the sentence and give feedback on each edit at the learner's level."""
        sentence = " ".join(sentence.split())
        if not sentence:
            return Result("", [])
        corrected = self.corrector.correct(sentence)
        feedback = []
        for edit in self.extractor.edits(sentence, corrected, level):
            record, messages = edit_item(edit, mode, self.graph, grounded=self.grounded)
            parsed = generate(self.backend, messages, self.decoding).parsed
            feedback.append(
                Feedback(
                    original_text=detokenize(edit.original_text),
                    corrected_text=detokenize(edit.corrected_text),
                    error_type=edit.error_type,
                    concept=record["concept_name"] or (parsed or {}).get("concept", ""),
                    text=shown_text(parsed, mode),
                    grounded=record["grounded"],
                )
            )
        return Result(corrected, feedback)


def format_feedback(result: Result) -> str:
    """The feedback as Markdown, one numbered block per edit."""
    if not result.corrected:
        return ""
    if not result.feedback:
        return "No corrections needed."
    blocks = []
    for number, item in enumerate(result.feedback, start=1):
        if not item.original_text:
            change = f"add **{item.corrected_text}**"
        elif not item.corrected_text:
            change = f"remove ~~{item.original_text}~~"
        else:
            change = f"~~{item.original_text}~~ → **{item.corrected_text}**"
        label = item.concept or item.error_type
        text = item.text or "*The model's answer could not be read.*"
        blocks.append(f"**{number}. {change}** · {label}\n\n{text}")
    return "\n\n".join(blocks)


# --- Stubs for running the interface without models ------------------------------------

FAKE_RULES = [
    (r"\b(he|she|it) don't\b", r"\1 doesn't"),
    (r"\bto going\b", "to go"),
    (r"\b(we|you|they) was\b", r"\1 were"),
    (r"\bI has\b", "I have"),
    (r"\ba (?=[aeiou])", "an "),
]
FAKE_TYPES = {
    ("do", "does"): "R:VERB:SVA",
    ("was", "were"): "R:VERB:SVA",
    ("has", "have"): "R:VERB:SVA",
    ("going", "go"): "R:VERB:FORM",
    ("a", "an"): "R:DET",
}
FAKE_CONCEPTS = {
    "R:VERB:SVA": "Subject-verb agreement",
    "R:VERB:FORM": "Verb form",
    "R:DET": "Articles",
}
TOKEN = re.compile(r"\w+(?=n't)|n't|\w+|[^\w\s]")


class FakeCorrector:
    """Fixes a few common errors with regular expressions."""

    def correct(self, sentence: str) -> str:
        for pattern, replacement in FAKE_RULES:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        return sentence


class FakeExtractor:
    """Token differences between the sentences, typed from a small table."""

    def edits(self, sentence: str, corrected: str, level: str) -> list[Edit]:
        tokens, corrected_tokens = TOKEN.findall(sentence), TOKEN.findall(corrected)
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
                error_type = FAKE_TYPES.get((before.lower(), after.lower()), f"{operation}:OTHER")
                spans.append((start, end, after, error_type))
        return make_edits(tokens, corrected_tokens, spans, level)


class FakeGraph:
    """A grammar graph with three concepts."""

    def concept_for(self, error_type: str) -> str | None:
        return error_type if error_type in FAKE_CONCEPTS else None

    def snippet(self, concept_id: str, level: str) -> dict:
        return {
            "concept_id": concept_id,
            "name": FAKE_CONCEPTS[concept_id],
            "rule": f"(Stub rule for {level}.)",
            "example": {},
            "prerequisites": [],
        }


class FakeBackend:
    """Answers every prompt with the same placeholder JSON."""

    def complete(self, messages: list[dict], decoding: Decoding) -> str:
        return build_target(
            "Stub concept",
            "Stub explanation: start the demo without --fake to see the model's feedback.",
            "Stub hint: start the demo without --fake to see the model's hint.",
        )


def fake_pipeline() -> Pipeline:
    """A pipeline of stubs, for trying the interface without a GPU or model files."""
    return Pipeline(FakeCorrector(), FakeExtractor(), FakeGraph(), FakeBackend())
