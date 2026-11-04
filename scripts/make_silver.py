"""Generate, filter and hand-check silver training explanations.

Steps:
    generate  Run the teacher (system R, with the graph) on a seeded sample of
              training edits. Output: outputs/silver/teacher.jsonl (resumable).
    filter    Keep outputs that parse, name the graph concept and name the edited
              words. Output: outputs/silver/kept.jsonl, kept and rejected counts in
              eval/results/silver_filter.md and .json, and a random sample of kept
              items in outputs/silver_handcheck.csv with an empty `correct` column.
    score     Read the teacher's marks in the hand-check CSV and report accuracy.

Usage:
    python scripts/make_silver.py generate --n 8000 --workers 16
    python scripts/make_silver.py filter
    python scripts/make_silver.py score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explain_to_learn.edits import read_jsonl
from explain_to_learn.generate import (
    Decoding,
    done_ids,
    edit_item,
    load_systems,
    make_backend,
    run_batch,
)
from explain_to_learn.silver import (
    assign_modes,
    counts_table,
    export_handcheck,
    filter_silver,
    handcheck_accuracy,
    select_training_edits,
)

SILVER_DIR = Path("outputs/silver")
RESULTS_DIR = Path("eval/results")


def open_graph(directory: Path):
    """Load the grammar graph from its YAML files (see `explain_to_learn.graph`)."""
    from explain_to_learn.graph import load_graph

    return load_graph(directory / "concepts.yaml", directory / "errant_map.yaml")


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def generate(args: argparse.Namespace) -> None:
    system = load_systems(args.config)[args.system]
    graph = open_graph(args.graph)
    edits = read_jsonl(args.edits)
    selected = select_training_edits(
        edits,
        set(read_lines(args.train_essays)),
        n=args.n,
        seed=args.seed,
        exclude_ids=read_lines(args.exclude) if args.exclude else (),
        has_concept=lambda error_type: graph.concept_for(error_type) is not None,
    )
    modes = assign_modes([e.edit_id for e in selected], args.hint_share, args.seed)
    done = done_ids(args.out)
    items = [
        edit_item(edit, modes[edit.edit_id], graph, grounded=True, system=args.system)
        for edit in selected
        if edit.edit_id not in done
    ]
    print(f"Selected {len(selected)} training edits; {len(items)} to run, {len(done)} done")
    written = run_batch(
        make_backend(system),
        items,
        args.out,
        Decoding(**system["decoding"]),
        retries=system.get("retries", 1),
        workers=args.workers,
    )
    print(f"Wrote {written} teacher outputs to {args.out}")


def filter_outputs(args: argparse.Namespace) -> None:
    edits = {edit.edit_id: edit for edit in read_jsonl(args.edits)}
    kept, counts = filter_silver(read_rows(args.raw), edits)

    args.kept.parent.mkdir(parents=True, exist_ok=True)
    with args.kept.open("w", encoding="utf-8") as out:
        for item in kept:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"total": sum(counts.values()), **dict(counts)}
    (args.results / "silver_filter.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = (
        "# Silver data filter\n\n"
        "Teacher outputs kept and rejected, with the first failed check as the "
        "reason (see `src/explain_to_learn/silver.py`).\n\n" + counts_table(counts)
    )
    (args.results / "silver_filter.md").write_text(report, encoding="utf-8")
    n = export_handcheck(kept, edits, args.handcheck, n=args.handcheck_n, seed=args.seed)
    print(f"Kept {len(kept)} of {summary['total']}; wrote {args.kept}")
    print(f"Wrote {n} items for hand-checking to {args.handcheck}")


def score(args: argparse.Namespace) -> None:
    result = handcheck_accuracy(args.handcheck)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "silver_handcheck.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["accuracy"] is None:
        print("No rated items yet")
        return
    print(
        f"Hand-check accuracy {100 * result['accuracy']:.1f}% "
        f"(95% CI {100 * result['low']:.1f}-{100 * result['high']:.1f}) on "
        f"{result['rated']} rated items; {result['unrated']} unrated"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Silver training explanations.")
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--handcheck", type=Path, default=Path("outputs/silver_handcheck.csv"))
    steps = parser.add_subparsers(dest="step", required=True)

    gen = steps.add_parser("generate", help="Run the teacher on training edits.")
    gen.add_argument("--n", type=int, default=8000, help="Training edits to sample")
    gen.add_argument("--hint-share", type=float, default=0.25)
    gen.add_argument("--system", default="R")
    gen.add_argument("--config", type=Path, default=Path("configs/systems.yaml"))
    gen.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    gen.add_argument("--train-essays", type=Path, default=Path("data/splits/train_essays.txt"))
    gen.add_argument("--exclude", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    gen.add_argument("--out", type=Path, default=SILVER_DIR / "teacher.jsonl")
    gen.add_argument("--workers", type=int, default=16, help="Parallel requests")
    gen.set_defaults(func=generate)

    filt = steps.add_parser("filter", help="Filter teacher outputs, export the hand-check.")
    filt.add_argument("--raw", type=Path, default=SILVER_DIR / "teacher.jsonl")
    filt.add_argument("--kept", type=Path, default=SILVER_DIR / "kept.jsonl")
    filt.add_argument("--handcheck-n", type=int, default=200)
    filt.set_defaults(func=filter_outputs)

    sc = steps.add_parser("score", help="Accuracy of the filled-in hand-check CSV.")
    sc.set_defaults(func=score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
