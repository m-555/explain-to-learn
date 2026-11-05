"""Correction F0.5 of the pipeline's correction step on the W&I+LOCNESS dev set.

1. Read every W&I+LOCNESS dev sentence from the official BEA-2019 M2 files, with its
   essay's CEFR level, using the same alignment that builds the edit table
   (`edits.align_essays`). The edit table lists only sentences with edits, but scoring
   needs all of them: a change to a correct sentence is a false positive.
2. Correct each sentence, detokenised as the learner wrote it, with a corrector from
   `configs/systems.yaml`, and tokenise the output with spaCy, as the M2 files are.
3. Write the hypothesis M2 file with ERRANT (`errant_parallel`) and score it against the
   official dev M2 files with `errant_compare`: span-based correction precision, recall
   and F0.5, overall and per CEFR level.

Outputs
-------
- `outputs/correction/<run>/` (git-ignored, quotes learner text): `orig.txt`, `hyp.txt`
  (corrected, tokenised, one sentence per line), `hyp.m2`, `ref.m2`, and the reference
  and hypothesis M2 subsets per level. A run resumes from the sentences already in
  `hyp.txt` unless `--overwrite` is given.
- `eval/results/correction_<run>.md` and `.json`: aggregate scores only.

The installed spaCy version may tokenise a few sentences differently from the one that
made the official files; such differences count as edits for every corrector alike.

Usage:
    python scripts/evaluate_correction.py --run gec-t5-small --corrector gec
    python scripts/evaluate_correction.py --run qwen3-4b-prompted --corrector llm
    python scripts/evaluate_correction.py --run smoke --limit 50
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from explain_to_learn.edits import CORPUS_FILES, LEVELS, align_essays, parse_m2, read_essays
from explain_to_learn.metrics import errant_compare, errant_parallel
from explain_to_learn.pipeline import Corrector, load_correctors, make_corrector
from explain_to_learn.prompts import detokenize
from explain_to_learn.splits import GOLD_CORPORA

CHUNK = 64


@dataclass(frozen=True)
class DevSentence:
    """One dev sentence: its M2 tokens, its essay's level and its reference M2 block."""

    tokens: str
    level: str
    reference: str


def m2_blocks(path: str | Path) -> list[str]:
    """The sentence blocks of an M2 file, in order, without the blank lines between them."""
    text = Path(path).read_text(encoding="utf-8")
    return [block.strip("\n") for block in text.split("\n\n") if block.strip()]


def write_m2(blocks: list[str], path: Path) -> None:
    """Write M2 sentence blocks to a file, separated by blank lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def dev_sentences(raw_dir: Path) -> list[DevSentence]:
    """Every W&I+LOCNESS dev sentence in the order of the official M2 files."""
    sentences = []
    for corpus, split, json_file, m2_file in CORPUS_FILES:
        if split != "dev" or corpus not in GOLD_CORPORA:
            continue
        json_path, m2_path = raw_dir / json_file, raw_dir / m2_file
        for path in (json_path, m2_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found; download the corpora with: "
                    "bash scripts/download_data.sh --accept-licence"
                )
        parsed, blocks = parse_m2(m2_path), m2_blocks(m2_path)
        if len(parsed) != len(blocks):
            raise ValueError(
                f"{m2_path}: {len(parsed)} sentences but {len(blocks)} blocks; "
                "download the corpora again"
            )
        position = 0
        for essay, essay_sentences in align_essays(read_essays(json_path, corpus), parsed):
            for sentence in essay_sentences:
                tokens = " ".join(sentence.tokens)
                sentences.append(DevSentence(tokens, essay.cefr_level, blocks[position]))
                position += 1
    return sentences


def spacy_tokeniser() -> Callable[[str], str]:
    """Tokenise a sentence with spaCy's English model, as the M2 files are tokenised."""
    from explain_to_learn.pipeline import default_annotator

    errant = default_annotator().errant

    def tokenise(text: str) -> str:
        doc = errant.parse(" ".join(text.split()), tokenise=True)
        return " ".join(token.text for token in doc if not token.is_space)

    return tokenise


def correct_all(
    sentences: list[DevSentence],
    corrector: Corrector,
    tokenise: Callable[[str], str],
    hyp_path: Path,
    chunk: int = CHUNK,
) -> list[str]:
    """Correct the sentences not yet in `hyp_path`, appending one line per sentence."""
    done = hyp_path.read_text(encoding="utf-8").splitlines() if hyp_path.exists() else []
    if len(done) > len(sentences):
        raise SystemExit(f"{hyp_path} has more lines than sentences; use --overwrite")
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    with hyp_path.open("a", encoding="utf-8") as out:
        for start in range(len(done), len(sentences), chunk):
            batch = sentences[start : start + chunk]
            corrected = corrector.correct([detokenize(s.tokens) for s in batch])
            out.writelines(tokenise(line) + "\n" for line in corrected)
            out.flush()
            print(f"{start + len(batch)}/{len(sentences)} sentences corrected")
    return hyp_path.read_text(encoding="utf-8").splitlines()


def score(sentences: list[DevSentence], run_dir: Path) -> dict:
    """Overall and per-level scores of `run_dir/hyp.txt` against the references."""
    hyp_m2, ref_m2 = run_dir / "hyp.m2", run_dir / "ref.m2"
    errant_parallel(run_dir / "orig.txt", run_dir / "hyp.txt", hyp_m2)
    hyp_blocks = m2_blocks(hyp_m2)
    if len(hyp_blocks) != len(sentences):
        raise ValueError(
            f"{hyp_m2} has {len(hyp_blocks)} sentences, expected {len(sentences)}; "
            "correct them again with --overwrite"
        )
    write_m2([s.reference for s in sentences], ref_m2)
    results = {"overall": {"sentences": len(sentences), **errant_compare(hyp_m2, ref_m2)}}
    results["by_level"] = {}
    for level in LEVELS:
        index = [i for i, s in enumerate(sentences) if s.level == level]
        if not index:
            continue
        level_hyp, level_ref = (
            run_dir / "levels" / f"{level}.hyp.m2",
            run_dir / "levels" / f"{level}.ref.m2",
        )
        write_m2([hyp_blocks[i] for i in index], level_hyp)
        write_m2([sentences[i].reference for i in index], level_ref)
        results["by_level"][level] = {
            "sentences": len(index),
            **errant_compare(level_hyp, level_ref),
        }
    return results


def markdown_report(results: dict) -> str:
    """Aggregate tables (no learner text) for `eval/results/correction_<run>.md`."""
    corrector = results["corrector"]
    lines = [
        f"# Correction: {results['run']}",
        "",
        f"Span-based correction scores (ERRANT `errant_compare`) of the `{corrector['name']}` "
        f"corrector (`{corrector.get('model', '')}`) on {results['overall']['sentences']} "
        "W&I+LOCNESS dev sentences, against the official BEA-2019 dev M2 files.",
    ]
    if results.get("limit"):
        lines.append(f"Only the first {results['limit']} sentences were corrected (a smoke test).")
    lines += [
        "",
        "| Level | Sentences | TP | FP | FN | Precision | Recall | F0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [("overall", results["overall"]), *results["by_level"].items()]
    for name, r in rows:
        lines.append(
            f"| {name} | {r['sentences']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f0.5']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Correct the dev sentences (resuming), score them and write the result files."""
    parser = argparse.ArgumentParser(description="Correction F0.5 on W&I+LOCNESS dev.")
    parser.add_argument("--run", required=True, help="Run name, e.g. gec-t5-small")
    parser.add_argument("--corrector", default="gec", help="Corrector in the config")
    parser.add_argument("--config", type=Path, default=Path("configs/systems.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/correction"))
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    parser.add_argument("--limit", type=int, help="Only the first N sentences (smoke test)")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="Sentences per corrector call")
    parser.add_argument("--overwrite", action="store_true", help="Correct every sentence again")
    args = parser.parse_args(argv)

    sentences = dev_sentences(args.raw)[: args.limit]
    run_dir = args.out_dir / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "orig.txt").write_text("".join(s.tokens + "\n" for s in sentences), "utf-8")
    if args.overwrite:
        (run_dir / "hyp.txt").unlink(missing_ok=True)
    correctors = load_correctors(args.config)
    if args.corrector not in correctors:
        parser.error(
            f"no corrector {args.corrector!r} in {args.config}; "
            f"choose from {', '.join(sorted(correctors))}"
        )
    settings = correctors[args.corrector]
    hyp_path = run_dir / "hyp.txt"
    done = len(hyp_path.read_text(encoding="utf-8").splitlines()) if hyp_path.exists() else 0
    if done != len(sentences):  # load the model only if there is something to correct
        correct_all(sentences, make_corrector(settings), spacy_tokeniser(), hyp_path, args.chunk)
    results = {
        "run": args.run,
        "corrector": {"name": args.corrector, **settings},
        "limit": args.limit,
        **score(sentences, run_dir),
    }

    args.results.mkdir(parents=True, exist_ok=True)
    md_path = args.results / f"correction_{args.run}.md"
    md_path.write_text(markdown_report(results), encoding="utf-8")
    md_path.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", "utf-8")
    overall = results["overall"]
    print(f"F0.5 {overall['f0.5']:.4f} on {overall['sentences']} sentences; wrote {md_path}")


if __name__ == "__main__":
    main()
