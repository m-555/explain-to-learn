"""End-to-end pipeline: free text -> correction -> ERRANT edits -> grammar graph -> feedback.

Two entry points share one explanation step (`explain_edits`):

- `explain_text` takes a learner's free text. It splits the text into sentences,
  corrects each sentence (or takes a correction the user gives), aligns each sentence
  with its correction using ERRANT, and turns every ERRANT edit into an `Edit` record
  with the same fields as a row of the edit table (`edits.py`). Each edit is then
  explained.
- `explain_gold_edits` takes rows of the edit table, whose corrections are gold, as
  in the main evaluation.

The explanation step uses the same functions as `scripts/run_system.py`:
`generate.edit_item` looks up the graph concept and builds the prompt, and
`generate.generate` calls the model and parses its JSON. `ExplainedEdit.to_row()`
returns the fields of a run_system output line, so the metrics and rating-sheet
scripts read pipeline output unchanged.

Every step is an object with one or two methods (`Corrector`, `Annotator`, the
backend inside `Explainer`), so tests and the demo's preview mode can replace any of
them with a stub.

Correctors
----------
- `GecCorrector`: an open sequence-to-sequence grammatical error correction model
  through transformers, by default Unbabel/gec-t5_small (T5-small, 60M parameters,
  Apache-2.0). It makes minimal edits and runs on the GPU or the CPU.
- `EndpointCorrector`: a chat model on an OpenAI-compatible endpoint (by default the
  explanation model's base model on the vLLM server), prompted to correct.

Both are configured under `correctors` in `configs/systems.yaml`.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from explain_to_learn.edits import SKIPPED_TYPES, Edit
from explain_to_learn.generate import (
    Backend,
    Decoding,
    Generation,
    OpenAIBackend,
    edit_item,
    generate,
    make_backend,
)
from explain_to_learn.metrics import shown_text
from explain_to_learn.prompts import detokenize

LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
GEC_MODEL = "Unbabel/gec-t5_small"
GEC_PREFIX = "gec: "
CORRECTION_PROMPT = (
    "Correct the grammar, spelling and punctuation of the learner's sentence. Change as "
    "little as possible: keep every word that is already correct and do not rephrase for "
    "style. Reply with the corrected sentence only, or with the sentence unchanged if it "
    "is already correct."
)


class Span(NamedTuple):
    """One ERRANT edit: token offsets in the original sentence, correction, error type."""

    start: int
    end: int
    correction: str
    error_type: str


class Corrector(Protocol):
    """Anything that corrects a list of sentences, returning one sentence for each."""

    def correct(self, sentences: list[str]) -> list[str]: ...


class Annotator(Protocol):
    """Sentence splitting, tokenisation and typed edits between two sentences."""

    def sentences(self, text: str) -> list[str]: ...

    def annotate(self, original: str, corrected: str) -> tuple[list[str], list[str], list[Span]]:
        """Return the original tokens, the corrected tokens and the edits."""
        ...


# --- Annotation --------------------------------------------------------------------


class ErrantAnnotator:
    """ERRANT with spaCy's `en_core_web_sm` (the `data` extra and the spaCy model).

    The model's parser splits sentences, and its tokeniser is the one the BEA-2019 M2
    files were made with, so edits look like the edit table's.
    """

    def __init__(self, language: str = "en") -> None:
        import errant

        self.errant = errant.load(language)

    def sentences(self, text: str) -> list[str]:
        """The sentences of `text`, with whitespace inside each one normalised."""
        sentences = (" ".join(s.text.split()) for s in self.errant.nlp(text).sents)
        return [s for s in sentences if s]

    def annotate(self, original: str, corrected: str) -> tuple[list[str], list[str], list[Span]]:
        orig = self.errant.parse(original, tokenise=True)
        cor = self.errant.parse(corrected, tokenise=True)
        spans = [
            Span(e.o_start, e.o_end, e.c_str, e.type)
            for e in self.errant.annotate(orig, cor)
            if e.type not in SKIPPED_TYPES
        ]
        return [t.text for t in orig], [t.text for t in cor], spans


@lru_cache(maxsize=1)
def default_annotator() -> ErrantAnnotator:
    """One ERRANT annotator per process; loading spaCy takes a second or two."""
    return ErrantAnnotator()


def check_level(level: str) -> str:
    """Return the CEFR level in upper case; raise ValueError if it is not A1 to C2."""
    if level.strip().upper() not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
    return level.strip().upper()


def make_edits(
    tokens: list[str],
    corrected_tokens: list[str],
    spans: Sequence[Span],
    level: str,
    sent_idx: int = 0,
    text_id: str = "input",
) -> list[Edit]:
    """Edit-table rows for one sentence of a free text.

    The text gets the corpus name `text`, so edit IDs follow the edit table's form
    `{corpus}-{essay_id}-{sent_idx}-{edit_idx}`. The learner's level is stored as
    `cefr_level`, which is where `generate.edit_item` reads it from.
    """
    edits = []
    for edit_idx, span in enumerate(spans):
        operation, _, category = span.error_type.partition(":")
        edits.append(
            Edit(
                edit_id=f"text-{text_id}-{sent_idx}-{edit_idx}",
                corpus="text",
                essay_id=text_id,
                split="",
                cefr_level=level,
                sent_idx=sent_idx,
                original=" ".join(tokens),
                corrected=" ".join(corrected_tokens),
                start=span.start,
                end=span.end,
                original_text=" ".join(tokens[span.start : span.end]),
                corrected_text=span.correction,
                error_type=span.error_type,
                operation=operation,
                error_category=category,
            )
        )
    return edits


def pair_sentences(original: str, corrected: str, annotator: Annotator) -> list[tuple[str, str]]:
    """Pair the sentences of a text with the sentences of its given correction.

    If both have the same number of sentences they are paired in order. Otherwise the
    correction joined or split sentences, and the whole text is aligned as one unit.
    """
    originals = annotator.sentences(original)
    corrections = annotator.sentences(corrected)
    if not originals:
        return []
    if len(originals) == len(corrections):
        return list(zip(originals, corrections, strict=True))
    return [(" ".join(originals), " ".join(corrections))]


def text_edits(
    text: str,
    level: str,
    corrector: Corrector | None = None,
    annotator: Annotator | None = None,
    corrected: str | None = None,
    text_id: str = "input",
) -> tuple[list[tuple[str, str]], list[Edit]]:
    """Sentence pairs (original, corrected) and their edits for a free text.

    The correction comes from `corrector`, or from `corrected` when the user gives
    it, in which case no corrector is needed.
    """
    level = check_level(level)
    annotator = annotator or default_annotator()
    if corrected is not None:
        if not corrected.strip() and text.strip():
            raise ValueError("the corrected text is empty")
        pairs = pair_sentences(text, corrected, annotator)
    else:
        if corrector is None:
            raise ValueError("give a corrector or the corrected text")
        sentences = annotator.sentences(text)
        pairs = list(zip(sentences, corrector.correct(sentences), strict=True))
    edits = []
    for sent_idx, (original, correction) in enumerate(pairs):
        tokens, corrected_tokens, spans = annotator.annotate(original, correction)
        edits += make_edits(tokens, corrected_tokens, spans, level, sent_idx, text_id)
    return pairs, edits


# --- Correction --------------------------------------------------------------------


class GecCorrector:
    """An open seq2seq GEC model through transformers (the `model` extra and PyTorch).

    It runs on the GPU if PyTorch sees one, and on the CPU otherwise. The default
    model is small enough for interactive use on a CPU.
    """

    def __init__(
        self,
        model: str = GEC_MODEL,
        prefix: str = GEC_PREFIX,
        num_beams: int = 5,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.prefix, self.num_beams, self.batch_size = prefix, num_beams, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def correct(self, sentences: list[str]) -> list[str]:
        corrected: list[str] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = [self.prefix + s for s in sentences[i : i + self.batch_size]]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            # Enough room for the correction to be somewhat longer than the input.
            limit = int(1.5 * inputs["input_ids"].shape[1]) + 10
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, num_beams=self.num_beams, max_new_tokens=limit
                )
            corrected += self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [c.strip() or s for c, s in zip(corrected, sentences, strict=True)]


def clean_correction(reply: str, sentence: str) -> str:
    """The corrected sentence from a chat model's reply; the sentence if there is none.

    Keeps the first non-empty line and removes quotes around it that the learner's
    sentence did not have.
    """
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return sentence
    text = lines[0]
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'" and sentence[:1] != text[0]:
        text = text[1:-1].strip()
    return text or sentence


@dataclass
class EndpointCorrector:
    """A chat model prompted to correct each sentence, usually through `OpenAIBackend`."""

    backend: Backend
    decoding: Decoding = field(default_factory=lambda: Decoding(max_new_tokens=256))
    workers: int = 1

    def messages(self, sentence: str) -> list[dict]:
        return [
            {"role": "system", "content": CORRECTION_PROMPT},
            {"role": "user", "content": sentence},
        ]

    def correct(self, sentences: list[str]) -> list[str]:
        def one(sentence: str) -> str:
            return clean_correction(
                self.backend.complete(self.messages(sentence), self.decoding), sentence
            )

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, sentences))


def load_correctors(path: str | Path = "configs/systems.yaml") -> dict[str, dict]:
    """The corrector settings under `correctors` in the systems config."""
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(config.get("correctors") or {})


def make_corrector(config: dict) -> Corrector:
    """Build the corrector described by one entry of `load_correctors`."""
    kind = config["kind"]
    if kind == "gec":
        return GecCorrector(
            config.get("model", GEC_MODEL),
            config.get("prefix", GEC_PREFIX),
            config.get("num_beams", 5),
            config.get("batch_size", 32),
        )
    if kind == "endpoint":
        backend = OpenAIBackend(
            model=config["model"],
            base_url=config["base_url"],
            api_key=os.environ.get("API_KEY", "none"),
            timeout_s=config.get("timeout_s", 120.0),
        )
        return EndpointCorrector(
            backend, Decoding(**config.get("decoding", {})), config.get("workers", 1)
        )
    raise ValueError(f"Unknown corrector kind {kind!r}; use 'gec' or 'endpoint'")


# --- Explanation -------------------------------------------------------------------


@dataclass
class Explainer:
    """The explanation model of one system: its backend, grounding and decoding."""

    backend: Backend
    grounded: bool = True
    decoding: Decoding = field(default_factory=Decoding)
    retries: int = 1
    system: str = ""

    @classmethod
    def from_system(cls, name: str, system: dict) -> Explainer:
        """Load a system defined in `configs/systems.yaml` (see `generate.load_systems`)."""
        return cls(
            make_backend(system),
            system["grounded"],
            Decoding(**system["decoding"]),
            system.get("retries", 1),
            name,
        )


@dataclass
class ExplainedEdit:
    """One edit with its generation record (`generate.edit_item`) and the model's answer."""

    edit: Edit
    record: dict
    generation: Generation

    @property
    def mode(self) -> str:
        return self.record["mode"]

    @property
    def text(self) -> str:
        """What the learner sees: the explanation, or the hint in hint mode."""
        return shown_text(self.generation.parsed, self.mode)

    @property
    def concept(self) -> str:
        """The graph concept's name, or the model's concept if the type has none."""
        return self.record["concept_name"] or (self.generation.parsed or {}).get("concept", "")

    def to_row(self) -> dict:
        """The fields of one output line of `scripts/run_system.py`."""
        return {**self.record, **dataclasses.asdict(self.generation)}


@dataclass
class TextResult:
    """The result for one free text: its sentence pairs and one result per edit."""

    pairs: list[tuple[str, str]]
    explained: list[ExplainedEdit]

    @property
    def edits(self) -> list[Edit]:
        return [item.edit for item in self.explained]

    @property
    def corrected_text(self) -> str:
        return " ".join(corrected for _, corrected in self.pairs)

    def to_dict(self) -> dict:
        """A JSON-ready view: the sentences and, per edit, its fields and the answer."""
        return {
            "sentences": [{"original": o, "corrected": c} for o, c in self.pairs],
            "results": [
                {**dataclasses.asdict(item.edit), **item.to_row(), "text": item.text}
                for item in self.explained
            ],
        }


def explain_edits(
    edits: Sequence[Edit],
    mode: str,
    explainer: Explainer,
    graph: Any,
    workers: int = 1,
    **extra: Any,
) -> list[ExplainedEdit]:
    """Explain each edit at its own level (`cefr_level`); one result per edit, in order.

    `graph` is a `GrammarGraph` (or any object with `concept_for` and `snippet`). It
    names the concept of every edit; only a grounded explainer gets the snippet in its
    prompt. `extra` fields (for example `run`) are added to each record. Use
    `workers > 1` only with a server backend.
    """
    items = [
        edit_item(edit, mode, graph, explainer.grounded, system=explainer.system, **extra)
        for edit in edits
    ]

    def run(item: tuple[dict, list[dict]]) -> Generation:
        return generate(explainer.backend, item[1], explainer.decoding, explainer.retries)

    with ThreadPoolExecutor(workers) as pool:
        generations = list(pool.map(run, items))
    return [
        ExplainedEdit(edit, record, generation)
        for edit, (record, _), generation in zip(edits, items, generations, strict=True)
    ]


def explain_gold_edits(
    edits: Sequence[Edit],
    mode: str,
    explainer: Explainer,
    graph: Any,
    workers: int = 1,
    **extra: Any,
) -> list[ExplainedEdit]:
    """Explain edit-table rows (gold corrections): the evaluation path.

    The same code as for free text, with each edit's level from the edit table.
    `[item.to_row() for item in ...]` gives the lines that `scripts/run_system.py`
    writes for a grounded system.
    """
    return explain_edits(edits, mode, explainer, graph, workers, **extra)


def explain_text(
    text: str,
    level: str,
    mode: str,
    corrector: Corrector | None,
    explainer: Explainer,
    graph: Any,
    annotator: Annotator | None = None,
    corrected: str | None = None,
    text_id: str = "input",
) -> TextResult:
    """Correct a free text, split the correction into edits and explain each edit.

    Pass `corrected` to use a given correction instead of `corrector`. `annotator`
    defaults to ERRANT with spaCy's English model.
    """
    pairs, edits = text_edits(text, level, corrector, annotator, corrected, text_id)
    return TextResult(pairs, explain_edits(edits, mode, explainer, graph))


def describe_change(original: str, corrected: str) -> str:
    """One edit as Markdown: `~~do~~ → **does**`, `add **a**` or `remove ~~very~~`."""
    original, corrected = detokenize(original), detokenize(corrected)
    if not original:
        return f"add **{corrected}**"
    if not corrected:
        return f"remove ~~{original}~~"
    return f"~~{original}~~ → **{corrected}**"


def feedback_markdown(result: TextResult) -> str:
    """The feedback as Markdown, one numbered block per edit."""
    if not result.pairs:
        return ""
    if not result.explained:
        return "No corrections needed."
    blocks = []
    for number, item in enumerate(result.explained, start=1):
        change = describe_change(item.edit.original_text, item.edit.corrected_text)
        label = item.concept or item.edit.error_type
        text = item.text or "*The model's answer could not be read.*"
        blocks.append(f"**{number}.** {change} · *{label}*\n\n{text}")
    return "\n\n".join(blocks)
