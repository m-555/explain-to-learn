"""Compute the automatic metrics for each system's run on the gold edits.

Reads `outputs/<system>/<run>.jsonl` for each system, scores every edit in the ID
list (an edit with no output counts as not covered), and writes aggregate tables,
with no learner text, to `eval/results/metrics_<run>.md` and `.json`.

Usage:
    python scripts/compute_metrics.py --run gold-explain --systems A B C D R
    python scripts/compute_metrics.py --run gold-explain --compare D:B D:C
    python scripts/compute_metrics.py --run demo --systems D \\
        --correction data/demo/orig.txt outputs/demo/corrected.txt data/demo/ref.m2

The optional rule-contradiction check (`--contradiction nli`) loads an NLI model
and needs the `model` extra; it is off by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explain_to_learn.edits import read_jsonl
from explain_to_learn.metrics import (
    breakdown,
    correction_f05,
    load_terms,
    nli_contradiction_check,
    paired_bootstrap,
    score_item,
    summarize,
    write_report,
)
from explain_to_learn.prompts import prompt_level

COMPARED_METRICS = ("covered", "concept_correct", "faithful", "within_length")
DEFAULT_COMPARISONS = ("B:A", "C:A", "D:B", "D:C")


def open_graph(directory: Path):
    """Load the grammar graph from its YAML files (see `explain_to_learn.graph`)."""
    from explain_to_learn.graph import GrammarGraph

    return GrammarGraph.load(directory)


def read_outputs(path: Path) -> dict[str, dict]:
    """Output rows by edit ID; the last row wins if an edit was run twice."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return {row["edit_id"]: row for row in rows}


def score_system(outputs, edits, graph, terms, mode, contradiction_check=None) -> list[dict]:
    """Per-edit scores for one system; `edits` fixes the denominator."""
    items = []
    for edit in edits:
        row = outputs.get(edit.edit_id)
        level = prompt_level(edit.cefr_level)
        concept_id = graph.concept_for(edit.error_type)
        snippet = graph.snippet(concept_id, level) if concept_id else None
        labels = {concept_id, snippet["name"]} if snippet else None
        items.append(
            score_item(
                edit,
                row["mode"] if row else mode,
                row["parsed"] if row else None,
                labels,
                graph.level_terms(level),
                terms,
                level=level,
                snippet=snippet,
                contradiction_check=contradiction_check,
            )
        )
    return items


def compute(args: argparse.Namespace, graph) -> dict:
    """All systems' summaries, breakdowns and paired comparisons."""
    all_edits = {edit.edit_id: edit for edit in read_jsonl(args.edits)}
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    edits = [all_edits[i] for i in ids]
    terms = load_terms(args.terms)
    check = nli_contradiction_check() if args.contradiction == "nli" else None

    items = {}
    for system in args.systems:
        outputs = read_outputs(args.outputs / system / f"{args.run}.jsonl")
        if not outputs:
            print(f"Warning: no outputs for system {system}; all its edits count as uncovered")
        items[system] = score_system(outputs, edits, graph, terms, args.mode, check)

    results: dict = {
        "run": args.run,
        "edits": len(edits),
        "n_resamples": args.resamples,
        "confidence": 0.95,
        "seed": args.seed,
        "systems": {
            system: {
                "overall": summarize(scored),
                "by_level": breakdown(scored, "level"),
                "by_error_category": breakdown(scored, "error_category"),
            }
            for system, scored in items.items()
        },
        "comparisons": {},
    }
    for pair in args.compare:
        a, b = pair.split(":")
        if a not in items or b not in items:
            continue
        results["comparisons"][f"{a} - {b}"] = {
            metric: paired_bootstrap(
                {item["edit_id"]: item[metric] for item in items[a]},
                {item["edit_id"]: item[metric] for item in items[b]},
                n_resamples=args.resamples,
                seed=args.seed,
            )
            for metric in COMPARED_METRICS
        }
    if args.correction:
        results["correction"] = correction_f05(*args.correction)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Automatic metrics per system.")
    parser.add_argument("--run", required=True, help="Run name, e.g. gold-explain")
    parser.add_argument("--systems", nargs="+", default=["A", "B", "C", "D", "R"])
    parser.add_argument(
        "--compare",
        nargs="*",
        default=list(DEFAULT_COMPARISONS),
        help="Pairs X:Y, reported as X minus Y",
    )
    parser.add_argument(
        "--mode",
        choices=("explain", "hint"),
        default="explain",
        help="Mode assumed for edits with no output",
    )
    parser.add_argument("--ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    parser.add_argument("--terms", type=Path, default=Path("configs/grammar_terms.txt"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contradiction", choices=("off", "nli"), default="off")
    parser.add_argument(
        "--correction",
        nargs=3,
        type=Path,
        metavar=("ORIG", "HYP", "REF_M2"),
        help="Also report correction F0.5 with ERRANT",
    )
    args = parser.parse_args(argv)

    results = compute(args, open_graph(args.graph))
    md_path, json_path = write_report(results, args.out_dir)
    print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
