"""Run one system (A, B, C, D or R) over a list of edit IDs.

Writes one JSON line per edit to `outputs/<system>/<run>.jsonl` (git-ignored, since
replies quote learner text) with the edit ID, level, mode, raw reply, parsed JSON,
attempts, latency and any error. A run resumes where it stopped unless
`--overwrite` is given.

Usage:
    python scripts/run_system.py --system B --run gold-explain \\
        --ids data/splits/gold_edit_ids.txt --mode explain
"""

from __future__ import annotations

import argparse
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


def open_graph(directory: Path):
    """Load the grammar graph from its YAML files (see `explain_to_learn.graph`)."""
    from explain_to_learn.graph import GrammarGraph

    return GrammarGraph.load(directory)


def read_ids(path: Path) -> list[str]:
    """Edit IDs, one per line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a system over a list of edits.")
    parser.add_argument("--system", required=True, help="System name in the config, e.g. B")
    parser.add_argument("--run", required=True, help="Run name, e.g. gold-explain")
    parser.add_argument("--mode", choices=("explain", "hint"), default="explain")
    parser.add_argument("--ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/systems.yaml"))
    parser.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests (servers)")
    parser.add_argument("--limit", type=int, help="Only the first N edits (for a smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="Start the run again")
    args = parser.parse_args(argv)

    system = load_systems(args.config)[args.system]
    edits = {edit.edit_id: edit for edit in read_jsonl(args.edits)}
    ids = read_ids(args.ids)[: args.limit]
    missing = [edit_id for edit_id in ids if edit_id not in edits]
    if missing:
        raise SystemExit(f"{len(missing)} edit IDs not in {args.edits}, e.g. {missing[0]}")

    out_path = args.out_dir / args.system / f"{args.run}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = done_ids(out_path)
    graph = open_graph(args.graph) if system["grounded"] else None
    items = [
        edit_item(edits[i], args.mode, graph, system["grounded"], system=args.system, run=args.run)
        for i in ids
        if i not in done
    ]
    print(f"System {args.system}: {len(items)} edits to run, {len(done)} already done")

    backend = make_backend(system)
    failures = 0

    def count(row: dict) -> None:
        nonlocal failures
        failures += row["parsed"] is None

    written = run_batch(
        backend,
        items,
        out_path,
        Decoding(**system["decoding"]),
        retries=system.get("retries", 1),
        workers=args.workers,
        on_result=count,
    )
    print(f"Wrote {written} outputs to {out_path} ({failures} without valid JSON)")


if __name__ == "__main__":
    main()
