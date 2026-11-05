"""Automatic metrics for generated explanations.

Coverage, concept accuracy, faithfulness (the explanation names the edited words and
does not contradict the graph rule) and level fit, with paired bootstrap intervals.

Every metric is first scored per edit by `score_item`, so that systems can be
compared on the same edits with `paired_bootstrap`, and broken down by error
category or CEFR level with `breakdown`.

Definitions
-----------
The text scored is the explanation in explain mode and the hint in hint mode.

- Coverage: the reply parsed and the text is not empty.
- Concept accuracy: the predicted concept equals the graph concept for the edit's
  ERRANT type (its ID or its name, ignoring case, spaces, `_` and `-`). Edits whose
  type has no concept are left out; replies that did not parse count as wrong.
- Faithfulness (`faithfulness`): explain mode must name the original words (R and U
  edits) and the corrected words (R and M edits). Hint mode must locate the error
  (the original words, or for an M edit a word next to the gap) and must not give
  the corrected words. Uncovered edits count as not faithful. See `mentions` for
  the matching rules.
- Rule contradiction: an optional check, off by default. Pass a callable
  `(text, snippet) -> bool` such as `nli_contradiction_check()` or an LLM judge.
- Level fit: the text is within the word limit for the level and has at most two
  sentences (one for a hint), and the share of grammar terms used that are not in
  the graph's term list for the level. Computed on covered edits only.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from explain_to_learn.edits import Edit
from explain_to_learn.prompts import detokenize

# Maximum words in an explanation or hint, per learner level.
LENGTH_LIMITS = {"A1": 20, "A2": 25, "B1": 30, "B2": 35, "C1": 40, "C2": 40}
MAX_SENTENCES = {"explain": 2, "hint": 1}

PUNCTUATION_NAMES = {
    ",": ("comma",),
    ".": ("full stop", "period", "dot"),
    "?": ("question mark",),
    "!": ("exclamation mark", "exclamation point"),
    "'": ("apostrophe",),
    '"': ("quotation mark", "quote"),
    ";": ("semicolon", "semi-colon"),
    ":": ("colon",),
    "-": ("hyphen", "dash"),
    "(": ("bracket", "parenthesis", "parentheses"),
    ")": ("bracket", "parenthesis", "parentheses"),
}
CASE_TERMS = ("capital", "lower case", "lowercase", "upper case", "uppercase", "small letter")

# Words of three letters or fewer ("a", "the", "to") turn up in any sentence, so they
# only count as named when they are quoted.
SHORT_WORD = 3

_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"'})
_WORD = re.compile(r"\w+(?:['-]\w+)*")
_QUOTED = re.compile(r'"([^"]+)"|(?<!\w)\'(.+?)\'(?!\w)|\*([^*]+)\*')
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
]

ContradictionCheck = Callable[[str, dict], bool]


# --- Text matching -----------------------------------------------------------------


def _words(text: str, strict: bool = False) -> list[str]:
    """Word tokens of `text`; lower-cased with contractions expanded unless strict."""
    text = text.translate(_QUOTES)
    if not strict:
        text = text.lower()
        for pattern, expansion in _CONTRACTIONS:
            text = pattern.sub(expansion, text)
    return _WORD.findall(text)


def _contains(haystack: list[str], needle: list[str]) -> bool:
    """True if `needle` occurs as a contiguous run in `haystack`."""
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def quoted_spans(text: str) -> list[str]:
    """Text inside double quotes, single quotes or asterisks."""
    return ["".join(groups) for groups in _QUOTED.findall(text.translate(_QUOTES))]


def mentions(text: str, phrase: str, strict: bool = False) -> bool:
    """True if `text` names `phrase` (M2 tokens, e.g. `do n't`).

    - Punctuation is ignored around words; a phrase of punctuation only is named by
      its name ("comma") or by the mark in quotes.
    - Matching ignores case and treats "don't" and "do not" alike, unless `strict`.
    - A phrase of short words only ("a", "to the") must appear inside quotes.
    """
    phrase = detokenize(phrase)
    needle = _words(phrase, strict)
    if not needle:
        marks = [mark for mark in phrase.translate(_QUOTES) if not mark.isspace()]
        if not marks:
            return False
        names = PUNCTUATION_NAMES.get(marks[0], ())
        return any(_contains(_words(text), _words(name)) for name in names) or any(
            marks[0] in span for span in quoted_spans(text)
        )
    if all(len(word) <= SHORT_WORD for word in needle):
        return any(_contains(_words(span, strict), needle) for span in quoted_spans(text))
    return _contains(_words(text, strict), needle)


@dataclass(frozen=True)
class Faithfulness:
    """Which parts of an edit a text names; None means the check does not apply."""

    names_original: bool | None
    names_corrected: bool | None
    reveals_correction: bool | None

    @property
    def faithful(self) -> bool:
        return (
            self.names_original is not False
            and self.names_corrected is not False
            and not self.reveals_correction
        )

    @property
    def reason(self) -> str | None:
        """The first failed check, as used for silver-data rejection counts."""
        if self.names_original is False:
            return "missing_original"
        if self.names_corrected is False:
            return "missing_correction"
        if self.reveals_correction:
            return "reveals_correction"
        return None


def _neighbours(edit: Edit) -> list[str]:
    """Words next to the edit in the original sentence (for locating a gap)."""
    tokens = edit.original.split()
    return [t for t in tokens[max(edit.start - 1, 0) : edit.end + 1] if _WORD.search(t)]


def faithfulness(edit: Edit, mode: str, text: str) -> Faithfulness:
    """Check that `text` names the edited words as the mode requires."""
    original, corrected = edit.original_text, edit.corrected_text
    # When the two sides differ only in case or contraction, compare them exactly.
    strict = bool(original and corrected) and _words(original) == _words(corrected)
    case_only = strict and original.lower() == corrected.lower()

    def names(phrase: str) -> bool:
        return mentions(text, phrase, strict)

    if mode == "explain":
        names_original = None
        if edit.operation in ("R", "U"):
            names_original = names(original) or (
                case_only and any(term in text.lower() for term in CASE_TERMS)
            )
        names_corrected = names(corrected) if edit.operation in ("R", "M") else None
        return Faithfulness(names_original, names_corrected, None)

    if edit.operation == "M":
        located = any(mentions(text, word) for word in _neighbours(edit))
    else:
        located = names(original)
    reveals = None
    if edit.operation in ("R", "M"):
        # A correction that is already in the sentence cannot give the answer away.
        in_sentence = _contains(_words(edit.original, strict), _words(corrected, strict))
        reveals = names(corrected) and not in_sentence
    return Faithfulness(located, None, reveals)


def normalize_concept(name: str) -> str:
    """Lower-case and keep letters and digits only (`Verb_tense` -> `verbtense`)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def concept_matches(predicted: str, labels: Iterable[str]) -> bool:
    """True if the predicted concept equals one of the accepted labels."""
    target = normalize_concept(predicted)
    return bool(target) and any(target == normalize_concept(label) for label in labels)


# --- Level fit ---------------------------------------------------------------------


def load_terms(path: str | Path = "configs/grammar_terms.txt") -> list[str]:
    """Read the grammar term list: one term per line, `#` starts a comment."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.split("#")[0].strip().lower() for line in lines if line.split("#")[0].strip()]


def _term_variants(term: str) -> list[tuple[str, ...]]:
    """The term and its plural forms, as word tuples."""
    words = _words(term)
    last = words[-1]
    plurals = [last + "s", last + "es"]
    if last.endswith("y"):
        plurals.append(last[:-1] + "ies")
    return [tuple(words)] + [tuple(words[:-1] + [p]) for p in plurals]


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Grammar terms used in `text`, longest match first ("past participle", not "participle")."""
    lookup = {variant: term for term in terms for variant in _term_variants(term)}
    longest = max((len(v) for v in lookup), default=0)
    words, found, i = _words(text), [], 0
    while i < len(words):
        for size in range(min(longest, len(words) - i), 0, -1):
            term = lookup.get(tuple(words[i : i + size]))
            if term is not None:
                found.append(term)
                i += size
                break
        else:
            i += 1
    return found


def count_sentences(text: str) -> int:
    """Number of sentences, ignoring full stops inside quotes."""
    unquoted = _QUOTED.sub("Q", text.translate(_QUOTES))
    return max(len(re.findall(r"[.!?]+(?=\s|$)", unquoted.strip())), 1 if text.strip() else 0)


def shown_text(parsed: dict | None, mode: str) -> str:
    """The field the learner sees: the explanation, or the hint in hint mode."""
    if parsed is None:
        return ""
    return parsed.get("hint" if mode == "hint" else "explanation", "")


# --- Per-edit scores ---------------------------------------------------------------


def score_item(
    edit: Edit,
    mode: str,
    parsed: dict | None,
    concept_labels: Iterable[str] | None,
    level_terms: Iterable[str],
    grammar_terms: Iterable[str],
    level: str | None = None,
    snippet: dict | None = None,
    contradiction_check: ContradictionCheck | None = None,
    length_limits: Mapping[str, int] = LENGTH_LIMITS,
) -> dict:
    """All per-edit scores for one reply; None marks a score that does not apply.

    `concept_labels` are the accepted labels for the edit's concept (its ID and name),
    or None if the error type has no concept. `level_terms` are the grammar terms the
    graph allows at the learner's level. `level` is the level the text was written
    for (default: the edit's level); breakdowns use the edit's own level.
    """
    level = level or edit.cefr_level
    text = shown_text(parsed, mode)
    covered = bool(text)
    item: dict = {
        "edit_id": edit.edit_id,
        "error_category": edit.error_category,
        "operation": edit.operation,
        "level": edit.cefr_level,
        "mode": mode,
        "covered": float(covered),
        "concept_correct": None,
        "faithful": float(covered and faithfulness(edit, mode, text).faithful),
        "contradicts_rule": None,
        "within_length": None,
        "terms": None,
        "off_level_terms": None,
    }
    if concept_labels is not None:
        item["concept_correct"] = float(
            parsed is not None and concept_matches(parsed["concept"], concept_labels)
        )
    if covered:
        limit = length_limits.get(level, max(length_limits.values()))
        item["within_length"] = float(
            len(_words(text)) <= limit and count_sentences(text) <= MAX_SENTENCES[mode]
        )
        allowed = {term.lower() for term in level_terms}
        used = find_terms(text, grammar_terms)
        item["terms"] = len(used)
        item["off_level_terms"] = sum(term not in allowed for term in used)
        if contradiction_check is not None and snippet is not None:
            item["contradicts_rule"] = float(contradiction_check(text, snippet))
    return item


SCORE_KEYS = ("covered", "concept_correct", "faithful", "contradicts_rule", "within_length")


def summarize(items: Sequence[dict]) -> dict:
    """Mean of each per-edit score over the edits where it applies, with counts."""
    summary: dict = {"edits": len(items)}
    for key in SCORE_KEYS:
        values = [item[key] for item in items if item[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
        summary[f"{key}_n"] = len(values)
    scored = [item for item in items if item["terms"] is not None]
    terms = sum(item["terms"] for item in scored)
    off = sum(item["off_level_terms"] for item in scored)
    summary["off_level_term_share"] = off / terms if terms else None
    summary["terms_used"] = terms
    summary["with_off_level_term"] = (
        sum(item["off_level_terms"] > 0 for item in scored) / len(scored) if scored else None
    )
    return summary


def breakdown(items: Sequence[dict], key: str) -> dict[str, dict]:
    """`summarize` per value of `key`, e.g. `error_category` or `level`."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(item[key])].append(item)
    return {group: summarize(groups[group]) for group in sorted(groups)}


# --- Paired bootstrap --------------------------------------------------------------


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence."""
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_bootstrap(
    scores_a: Mapping[str, float | None],
    scores_b: Mapping[str, float | None],
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Confidence interval for mean(A) - mean(B) on the edits both systems have.

    Edits are resampled with replacement; each resample keeps the A and B scores of
    the same edits together. The p-value is two-sided: twice the share of resampled
    differences on the other side of zero from the observed difference.
    """
    ids = sorted(k for k in scores_a if scores_a[k] is not None and scores_b.get(k) is not None)
    result = {"n": len(ids), "mean_a": None, "mean_b": None, "diff": None}
    result.update({"low": None, "high": None, "p_value": None})
    if not ids:
        return result
    diffs = [scores_a[k] - scores_b[k] for k in ids]
    n = len(ids)
    observed = sum(diffs) / n
    rng = random.Random(seed)
    resampled = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    alpha = 1 - confidence
    if observed > 0:
        other_side = sum(d <= 0 for d in resampled)
    elif observed < 0:
        other_side = sum(d >= 0 for d in resampled)
    else:
        other_side = n_resamples
    result.update(
        mean_a=sum(scores_a[k] for k in ids) / n,
        mean_b=sum(scores_b[k] for k in ids) / n,
        diff=observed,
        low=_quantile(resampled, alpha / 2),
        high=_quantile(resampled, 1 - alpha / 2),
        p_value=min(1.0, 2 * other_side / n_resamples),
    )
    return result


# --- Optional rule-contradiction check ---------------------------------------------


def nli_contradiction_check(
    model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    threshold: float = 0.5,
) -> ContradictionCheck:
    """An NLI model as the rule-contradiction check (needs the `model` extra).

    The premise is the graph rule with its example and the hypothesis is the
    explanation; the check fires when the contradiction probability exceeds
    `threshold`. This is a rough signal: explanations mention the learner's words,
    which the rule does not, so most pairs come out neutral. An LLM judge with the
    rubric can be passed instead, as any callable `(text, snippet) -> bool`.
    """
    from transformers import pipeline

    classify = pipeline("text-classification", model=model, top_k=None)

    def check(text: str, snippet: dict) -> bool:
        example = snippet.get("example") or {}
        premise = snippet["rule"]
        if example:
            premise += f' Wrong: "{example["wrong"]}" Right: "{example["right"]}"'
        scores = classify({"text": premise, "text_pair": text})
        return any(
            s["label"].lower().startswith("contra") and s["score"] > threshold for s in scores
        )

    return check


# --- Correction F0.5 (end-to-end demo pipeline) ------------------------------------


def parse_errant_compare(output: str) -> dict:
    """Read TP, FP, FN, precision, recall and F0.5 from `errant_compare` output."""
    lines = [line.split() for line in output.splitlines()]
    for header, values in zip(lines, lines[1:], strict=False):
        if header[:3] == ["TP", "FP", "FN"]:
            tp, fp, fn = (int(v) for v in values[:3])
            precision, recall, f05 = (float(v) for v in values[3:6])
            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f0.5": f05,
            }
    raise ValueError("No score table in errant_compare output")


def _run_errant(command: str, *args: str) -> subprocess.CompletedProcess:
    """Run an ERRANT command (`parallel_to_m2` or `compare_m2`) with this Python.

    Calling the command module's `main()` through `sys.executable` works whether or
    not the environment's scripts folder is on PATH (for example on Windows without
    an activated virtual environment), unlike calling `errant_parallel` by name.
    """
    code = f"from errant.commands.{command} import main; main()"
    return subprocess.run(
        [sys.executable, "-c", code, *args], check=True, capture_output=True, text=True
    )


def errant_parallel(orig: str | Path, hyp: str | Path, out_m2: str | Path) -> None:
    """Write the M2 file of corrected sentences with ERRANT's `errant_parallel`.

    `orig` and `hyp` are text files with one tokenised sentence per line (original
    and system-corrected). Needs the `data` extra (ERRANT and a spaCy English model).
    """
    _run_errant("parallel_to_m2", "-orig", str(orig), "-cor", str(hyp), "-out", str(out_m2))


def errant_compare(hyp_m2: str | Path, ref_m2: str | Path) -> dict:
    """Span-based correction scores of a hypothesis M2 file against a reference M2 file."""
    result = _run_errant("compare_m2", "-hyp", str(hyp_m2), "-ref", str(ref_m2))
    return parse_errant_compare(result.stdout)


def correction_f05(orig: str | Path, hyp: str | Path, ref_m2: str | Path) -> dict:
    """Span-based correction F0.5 of corrected sentences against a reference M2 file.

    `orig` and `hyp` are as for `errant_parallel`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        hyp_m2 = Path(tmp) / "hyp.m2"
        errant_parallel(orig, hyp, hyp_m2)
        return errant_compare(hyp_m2, ref_m2)


# --- Report ------------------------------------------------------------------------


def _cell(value: float | None, percent: bool = True) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f}" if percent else f"{value:.3f}"


def markdown_report(results: dict) -> str:
    """Aggregate tables (no learner text) for `eval/results/metrics_<run>.md`."""
    lines = [f"# Automatic metrics: {results['run']}", ""]
    lines += [
        "Scores in percent. Coverage, concept accuracy and faithfulness are over all "
        "edits (concept accuracy over edits with a mapped concept); length fit and "
        "off-level terms are over covered edits. Off-level terms: share of grammar "
        "terms used that are not in the graph's term list for the learner's level.",
        "",
    ]
    columns = [
        ("covered", "Coverage"),
        ("concept_correct", "Concept acc."),
        ("faithful", "Faithful"),
        ("contradicts_rule", "Contradicts rule"),
        ("within_length", "Length fit"),
        ("off_level_term_share", "Off-level terms"),
    ]
    header = "| System | Edits | " + " | ".join(title for _, title in columns) + " |"
    rule = "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |"

    def table(rows: dict[str, dict], first: str) -> list[str]:
        out = [header.replace("System", first, 1), rule]
        for name, summary in rows.items():
            cells = " | ".join(_cell(summary[key]) for key, _ in columns)
            out.append(f"| {name} | {summary['edits']} | {cells} |")
        return out

    lines += ["## Overall", ""]
    lines += table({s: r["overall"] for s, r in results["systems"].items()}, "System")
    for key, title in (("level", "CEFR level"), ("error_category", "error category")):
        lines += ["", f"## By {title}", ""]
        for system, result in results["systems"].items():
            lines += [f"### System {system}", ""]
            lines += table(result["by_" + key], title[0].upper() + title[1:])
            lines.append("")
    if results.get("comparisons"):
        lines += ["## Paired differences", ""]
        lines += [
            f"Mean difference in percentage points with a "
            f"{results.get('confidence', 0.95):.0%} paired bootstrap interval "
            f"({results.get('n_resamples', 10_000)} resamples of edits).",
            "",
            "| Comparison | Metric | Edits | Difference | Interval | p |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
        for name, metrics in results["comparisons"].items():
            for metric, c in metrics.items():
                if c["diff"] is None:
                    continue
                interval = f"[{_cell(c['low'])}, {_cell(c['high'])}]"
                lines.append(
                    f"| {name} | {metric} | {c['n']} | {_cell(c['diff'])} | {interval} | "
                    f"{c['p_value']:.3f} |"
                )
    if results.get("correction"):
        c = results["correction"]
        lines += [
            "",
            "## Correction (ERRANT, span-based)",
            "",
            "| TP | FP | FN | Precision | Recall | F0.5 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.4f} | "
            f"{c['recall']:.4f} | {c['f0.5']:.4f} |",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(results: dict, out_dir: str | Path = "eval/results") -> tuple[Path, Path]:
    """Write `metrics_<run>.md` and `metrics_<run>.json`; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"metrics_{results['run']}.md"
    json_path = out_dir / f"metrics_{results['run']}.json"
    md_path.write_text(markdown_report(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path
