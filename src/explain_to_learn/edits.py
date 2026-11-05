"""Edit table: one row per gold ERRANT edit, with its essay ID and learner level.

Sources
-------
The BEA-2019 releases of W&I+LOCNESS (v2.1) and FCE (v2.1) come in two formats:

- JSON: one essay per line, with its `id`, CEFR level (`cefr`, W&I and LOCNESS only),
  raw `text` and character-offset `edits`.
- M2: tokenised sentences with typed ERRANT edits, but no essay IDs.

The project splits and samples by essay, so every edit needs its essay ID and level.

Approach: align the official M2 sentences back to the JSON essays
------------------------------------------------------------------
The official M2 files were made by the `json_to_m2.py` script shipped with each
corpus. It reads the JSON file in order, normalises some punctuation, splits each
essay into paragraphs and sentences, and writes the sentences one after another. A
tokeniser only splits text and never changes characters, so the characters of an
essay (after the same punctuation normalisation, ignoring whitespace) are exactly the
characters of its consecutive M2 sentences. `align_essays` walks through both files
in order and takes M2 sentences for each essay until the characters match, and stops
with an error if they ever differ.

Why this approach: the edits and error types are exactly the official BEA-2019 gold
annotation that other work reports on, and building the table needs no spaCy model.
Re-running ERRANT on the JSON text with a current spaCy would give different
sentence splits, tokens and some different error types.

Caveats
-------
- The alignment depends on the M2 file and the JSON file being the same release
  and in the same order. Any mismatch raises `ValueError`; nothing is guessed.
- Edits that `json_to_m2.py` dropped (edits across a sentence or paragraph boundary)
  are not in the M2 files and so are not in this table.
- `noop` edits (no change) are skipped. `UNK` edits are skipped too: in FCE they mark
  errors that were detected but not corrected, so they have no correction to explain.
- Only annotator 0 is used. All BEA-2019 W&I+LOCNESS and FCE files have one annotator.
- `cefr_level` for W&I is the essay's CEFR level with its sub-level dropped
  (`A2.ii` -> `A2`, `C2+` -> `C2`). LOCNESS essays are `native`. FCE essays have no
  individual level; they are answers to a B2-level exam (First Certificate in
  English), so they are labelled `B2`. That is the exam's target level, not a
  rating of the writer.
- FCE `id` identifies an exam script with two answers, so the FCE essay ID is
  `{id}-{q}` (script ID and question number).
- `split` is the official split. W&I+LOCNESS has train and dev (its test set has no
  public corrections); FCE has train, dev and test.

Edit IDs have the form `{corpus}-{essay_id}-{sent_idx}-{edit_idx}`, where `sent_idx`
is the sentence's position in the essay and `edit_idx` the edit's position in the
M2 sentence block, both counted from 0. They depend only on the official files, so
they can be published and the table rebuilt from them.

Usage:
    python -m explain_to_learn.edits build --raw data/raw --out data/processed/edits.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Punctuation normalisation applied by the official json_to_m2.py before tokenising.
PUNCT_NORMALISATION = str.maketrans(
    {
        "’": "'",
        "´": "'",
        "‘": "'",
        "′": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "˝": '"',
        "¨": '"',
        "„": '"',
        "『": '"',
        "』": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "¬": "-",
        "、": ",",
        "，": ",",
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "ِ": " ",
        "​": " ",
    }
)

# (corpus, split, JSON file, M2 file), paths relative to the raw data directory.
CORPUS_FILES = [
    *[
        (
            "wi",
            split,
            f"wi+locness/json/{band}.{split}.json",
            f"wi+locness/m2/{band}.{split}.gold.bea19.m2",
        )
        for split in ("train", "dev")
        for band in ("A", "B", "C")
    ],
    ("locness", "dev", "wi+locness/json/N.dev.json", "wi+locness/m2/N.dev.gold.bea19.m2"),
    *[
        ("fce", split, f"fce/json/fce.{split}.json", f"fce/m2/fce.{split}.gold.bea19.m2")
        for split in ("train", "dev", "test")
    ],
]

FCE_LEVEL = "B2"
# Every value of `cefr_level` in the edit table, in level order (`native` is LOCNESS).
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2", "native")
SKIPPED_TYPES = {"noop", "UNK"}


@dataclass(frozen=True)
class M2Edit:
    """One edit line of an M2 file."""

    start: int
    end: int
    error_type: str
    correction: str
    annotator: int


@dataclass(frozen=True)
class M2Sentence:
    """One M2 block: an original tokenised sentence and its edits."""

    tokens: tuple[str, ...]
    edits: tuple[M2Edit, ...]


@dataclass(frozen=True)
class Essay:
    """One essay from a BEA-2019 JSON file."""

    essay_id: str
    cefr_level: str
    text: str


@dataclass(frozen=True)
class Edit:
    """One row of the edit table."""

    edit_id: str
    corpus: str
    essay_id: str
    split: str
    cefr_level: str
    sent_idx: int
    original: str
    corrected: str
    start: int
    end: int
    original_text: str
    corrected_text: str
    error_type: str
    operation: str
    error_category: str

    @property
    def essay_key(self) -> str:
        """Essay identifier that is unique across corpora, e.g. `wi-1-352420`."""
        return f"{self.corpus}-{self.essay_id}"


def parse_m2(path: str | Path) -> list[M2Sentence]:
    """Read an M2 file into a list of sentences with their edits."""
    sentences = []
    tokens: list[str] | None = None
    edits: list[M2Edit] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("S "):
            if tokens is not None:
                sentences.append(M2Sentence(tuple(tokens), tuple(edits)))
            tokens, edits = line[2:].split(), []
        elif line.startswith("A "):
            span, error_type, correction, _, _, annotator = line[2:].split("|||")
            start, end = span.split()
            if correction == "-NONE-":
                correction = ""
            edits.append(M2Edit(int(start), int(end), error_type, correction, int(annotator)))
    if tokens is not None:
        sentences.append(M2Sentence(tuple(tokens), tuple(edits)))
    return sentences


def apply_edits(tokens: Iterable[str], edits: Iterable[M2Edit]) -> list[str]:
    """Return the corrected tokens after applying non-overlapping edits."""
    corrected = list(tokens)
    offset = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.error_type == "noop":
            continue
        new_tokens = edit.correction.split()
        corrected[edit.start + offset : edit.end + offset] = new_tokens
        offset += len(new_tokens) - (edit.end - edit.start)
    return corrected


def cefr_level(raw: str) -> str:
    """Map a W&I CEFR label to a coarse level: `A2.ii` -> `A2`, `N` -> `native`."""
    if raw == "N":
        return "native"
    level = raw[:2]
    if level not in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        raise ValueError(f"Unknown CEFR label: {raw!r}")
    return level


def read_essays(path: str | Path, corpus: str) -> list[Essay]:
    """Read a BEA-2019 JSON file (one essay per line) in file order."""
    essays = []
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if corpus == "fce":
                essays.append(Essay(f"{row['id']}-{row['q']}", FCE_LEVEL, row["text"]))
            else:
                essays.append(Essay(row["id"], cefr_level(row["cefr"]), row["text"]))
    return essays


def _characters(text: str) -> str:
    """Characters compared during alignment: normalised punctuation, no whitespace."""
    return re.sub(r"\s", "", text.translate(PUNCT_NORMALISATION))


def align_essays(
    essays: list[Essay], sentences: list[M2Sentence]
) -> Iterator[tuple[Essay, list[M2Sentence]]]:
    """Pair each essay with its consecutive M2 sentences (see module docstring)."""
    position = 0
    for essay in essays:
        target = _characters(essay.text)
        taken = ""
        essay_sentences = []
        while len(taken) < len(target) and position < len(sentences):
            sentence = sentences[position]
            taken += _characters("".join(sentence.tokens))
            essay_sentences.append(sentence)
            position += 1
        if taken != target:
            raise ValueError(f"M2 sentences do not match the text of essay {essay.essay_id}")
        yield essay, essay_sentences
    if position != len(sentences):
        raise ValueError(f"{len(sentences) - position} M2 sentences left after the last essay")


def essay_edits(corpus: str, split: str, essay: Essay, sentences: list[M2Sentence]) -> list[Edit]:
    """Build the edit rows for one essay from its M2 sentences."""
    rows = []
    for sent_idx, sentence in enumerate(sentences):
        sentence_edits = [e for e in sentence.edits if e.annotator == 0]
        original = " ".join(sentence.tokens)
        corrected = " ".join(apply_edits(sentence.tokens, sentence_edits))
        for edit_idx, edit in enumerate(sentence_edits):
            if edit.error_type in SKIPPED_TYPES:
                continue
            operation, error_category = edit.error_type.split(":", 1)
            rows.append(
                Edit(
                    edit_id=f"{corpus}-{essay.essay_id}-{sent_idx}-{edit_idx}",
                    corpus=corpus,
                    essay_id=essay.essay_id,
                    split=split,
                    cefr_level=essay.cefr_level,
                    sent_idx=sent_idx,
                    original=original,
                    corrected=corrected,
                    start=edit.start,
                    end=edit.end,
                    original_text=" ".join(sentence.tokens[edit.start : edit.end]),
                    corrected_text=edit.correction,
                    error_type=edit.error_type,
                    operation=operation,
                    error_category=error_category,
                )
            )
    return rows


def build_edits(raw_dir: str | Path) -> list[Edit]:
    """Build the edit table from the extracted corpora in `raw_dir`."""
    raw_dir = Path(raw_dir)
    rows = []
    for corpus, split, json_file, m2_file in CORPUS_FILES:
        json_path, m2_path = raw_dir / json_file, raw_dir / m2_file
        for path in (json_path, m2_path):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; run scripts/download_data.sh")
        essays = read_essays(json_path, corpus)
        for essay, sentences in align_essays(essays, parse_m2(m2_path)):
            rows.extend(essay_edits(corpus, split, essay, sentences))
    return rows


def write_jsonl(edits: Iterable[Edit], path: str | Path) -> None:
    """Write edits to a JSON Lines file, one edit per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for edit in edits:
            out.write(json.dumps(dataclasses.asdict(edit), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Edit]:
    """Read edits written by `write_jsonl`."""
    with Path(path).open(encoding="utf-8") as lines:
        return [Edit(**json.loads(line)) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the edit table.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build the edit table from the raw corpora.")
    build.add_argument("--raw", type=Path, default=Path("data/raw"))
    build.add_argument("--out", type=Path, default=Path("data/processed/edits.jsonl"))
    args = parser.parse_args()

    edits = build_edits(args.raw)
    write_jsonl(edits, args.out)
    print(f"Wrote {len(edits)} edits to {args.out}")


if __name__ == "__main__":
    main()
