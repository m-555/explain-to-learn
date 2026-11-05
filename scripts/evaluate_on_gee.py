"""Compare the systems with GEE's English explanations (Song et al., 2024) on the same edits.

Steps:
    prepare  Read GEE's English explanations (data/raw/gee/, cloned by
             scripts/download_data.sh) and pair each atomic edit with the official
             W&I+LOCNESS edit of the same sentence (see explain_to_learn.gee). Writes
             outputs/gee/edit_ids.txt and outputs/gee/references.jsonl.
    run      Run one configured system on those edits in explain mode, at each learner's
             level, with scripts/run_system.py: outputs/<system>/gee.jsonl.
    compare  Score each system and GEE's explanations on the same edits, and write
             aggregate tables to eval/results/gee_comparison.md and .json, and a sheet
             with GEE's and every system's explanation side by side, for a teacher, to
             outputs/gee/side_by_side.csv.

Metrics that apply to both GEE and the systems (explain mode):
- coverage: there is a non-empty explanation;
- faithfulness: it names the original and the corrected words (`metrics.faithfulness`).

Metrics that do not apply:
- concept accuracy: GEE labels each edit with its own free-text error type ("noun
  number", "collocation"), not with a grammar graph concept, so its explanations have
  no concept to score;
- level fit: GEE's explanations were not written for a learner level;
- hint mode: GEE has explanations only;
- rule contradiction: an optional check against the graph rule, which GEE did not see.

GEE's repository has no licence file. Its data is not committed: everything written here
that quotes it goes to outputs/ (git-ignored), and eval/results/ gets only counts and
shares.

Usage:
    python scripts/evaluate_on_gee.py prepare
    python scripts/evaluate_on_gee.py run --system D
    python scripts/evaluate_on_gee.py compare --systems A B C D R
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.gee import GEE_FILE, match_edits, read_gee, reference_row
from explain_to_learn.human_eval import write_csv
from explain_to_learn.io import read_ids, write_ids
from explain_to_learn.metrics import faithfulness, paired_bootstrap, shown_text
from explain_to_learn.prompts import detokenize, prompt_level
from explain_to_learn.silver import wilson_interval

RUN = "gee"
GEE_DIR = Path("outputs/gee")
METRICS = (("covered", "Coverage"), ("faithful", "Faithfulness"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    """A JSON Lines file as a list of dictionaries; empty if there is no file."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def prepare(args: argparse.Namespace) -> None:
    """Pair GEE's atomic edits with the edit table and write the IDs and references."""
    records, counts = read_gee(args.gee)
    matches, match_counts = match_edits(records, read_jsonl(args.edits))
    counts.update(match_counts)
    gold = set(read_ids(args.gold_ids)) if args.gold_ids.is_file() else set()
    counts["matched edits in the gold set"] = sum(m.edit.edit_id in gold for m in matches)

    write_ids((m.edit.edit_id for m in matches), args.gee_dir / "edit_ids.txt")
    with (args.gee_dir / "references.jsonl").open("w", encoding="utf-8") as out:
        for match in matches:
            out.write(json.dumps(reference_row(match), ensure_ascii=False) + "\n")
    (args.gee_dir / "prepare_counts.json").write_text(
        json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8"
    )
    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")
    print(f"Wrote {len(matches)} edit IDs and references to {args.gee_dir}")


def run(args: argparse.Namespace) -> None:
    """Run one system on the GEE edits with scripts/run_system.py."""
    import run_system  # scripts/run_system.py, next to this script

    ids = args.gee_dir / "edit_ids.txt"
    if not ids.is_file():
        raise SystemExit(f"{ids} not found; run the prepare step first")
    run_system.main(
        ["--system", args.system, "--run", RUN, "--mode", "explain", "--ids", str(ids)]
        + ["--edits", str(args.edits), "--config", str(args.config), "--graph", str(args.graph)]
        + ["--out-dir", str(args.outputs), "--workers", str(args.workers)]
        + (["--overwrite"] if args.overwrite else [])
    )


def score(edit: Edit, text: str) -> dict[str, float]:
    """Coverage and faithfulness of one explanation of an edit."""
    covered = bool(text)
    return {
        "covered": float(covered),
        "faithful": float(covered and faithfulness(edit, "explain", text).faithful),
    }


def summary(scores: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Share of each metric with its 95% Wilson interval."""
    result: dict[str, Any] = {"edits": len(scores)}
    for key, _ in METRICS:
        hits = round(sum(s[key] for s in scores))
        low, high = wilson_interval(hits, len(scores))
        result[key] = {"share": hits / len(scores) if scores else None, "low": low, "high": high}
    return result


def compare_scores(
    references: list[dict[str, Any]],
    edits: dict[str, Edit],
    outputs: dict[str, dict[str, dict[str, Any]]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summaries for GEE and each system, and paired differences of each system minus GEE.

    An edit with no output from a system counts as not covered, as in compute_metrics.py.
    """
    scores = {
        "GEE": {r["edit_id"]: score(edits[r["edit_id"]], r["explanation"]) for r in references}
    }
    for system, rows in outputs.items():
        scores[system] = {
            r["edit_id"]: score(
                edits[r["edit_id"]],
                shown_text(rows.get(r["edit_id"], {}).get("parsed"), "explain"),
            )
            for r in references
        }
    results: dict[str, Any] = {
        "edits": len(references),
        "flagged_by_gee_annotator": sum(bool(r["flags"]) for r in references),
        "from_records_marked_all_correct": sum(r["all_correct"] for r in references),
        "sources": {name: summary(list(values.values())) for name, values in scores.items()},
        "differences": {},
    }
    for system in outputs:
        results["differences"][f"{system} - GEE"] = {
            key: paired_bootstrap(
                {i: s[key] for i, s in scores[system].items()},
                {i: s[key] for i, s in scores["GEE"].items()},
                n_resamples=n_resamples,
                seed=seed,
            )
            for key, _ in METRICS
        }
    return results


def percent(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def markdown_report(results: dict[str, Any], counts: dict[str, int]) -> str:
    """The comparison as Markdown: counts and shares only, no text from GEE or learners."""
    lines = [
        "# Comparison with GEE (English)",
        "",
        f"{results['edits']} edits from the English data of GEE (Song et al., 2024) that match an "
        "official W&I+LOCNESS edit of the same sentence. The systems explain each edit at the "
        "learner's CEFR level; GEE's explanations were written by the GEE pipeline. "
        f"The GEE annotator flagged {results['flagged_by_gee_annotator']} of these GEE "
        "explanations as wrong. Shares in percent with 95% Wilson intervals.",
        "",
        "| Source | Edits | " + " | ".join(f"{title} (%)" for _, title in METRICS) + " |",
        "| --- | ---: |" + " ---: |" * len(METRICS),
    ]
    for name, result in results["sources"].items():
        label = "GEE explanation" if name == "GEE" else f"System {name}"
        cells = [
            f"{percent(result[key]['share'])} ({percent(result[key]['low'])}–"
            f"{percent(result[key]['high'])})"
            for key, _ in METRICS
        ]
        lines.append(f"| {label} | {result['edits']} | " + " | ".join(cells) + " |")
    if results["differences"]:
        lines += [
            "",
            "## Paired differences (system minus GEE)",
            "",
            "Percentage points with a 95% paired bootstrap interval over edits.",
            "",
            "| Comparison | Metric | Edits | Difference | Interval | p |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
        for name, metrics in results["differences"].items():
            for key, title in METRICS:
                c = metrics[key]
                if c["diff"] is None:
                    continue
                lines.append(
                    f"| {name} | {title} | {c['n']} | {percent(c['diff'])} | "
                    f"[{percent(c['low'])}, {percent(c['high'])}] | {c['p_value']:.3f} |"
                )
    lines += ["", "## Edits used", "", "| Count | Value |", "| --- | ---: |"]
    lines += [f"| {name} | {value} |" for name, value in sorted(counts.items())]
    lines += [
        "",
        "## Not compared",
        "",
        "- Concept accuracy: GEE labels edits with its own free-text error types, not with "
        "grammar graph concepts.",
        "- Level fit: GEE's explanations were not written for a learner level.",
        "- Hint mode: GEE has explanations only.",
    ]
    return "\n".join(lines) + "\n"


def side_by_side(
    references: list[dict[str, Any]],
    edits: dict[str, Edit],
    outputs: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows and columns of the teacher's sheet: GEE's and each system's explanation."""
    columns = [
        "edit_id",
        "level",
        "sentence",
        "corrected",
        "edit",
        "errant_type",
        "gee_error_type",
        "gee_annotator_flags",
        "gee_explanation",
        *(f"system_{system}" for system in outputs),
        "preferred",
        "comment",
    ]
    rows = []
    for reference in references:
        edit = edits[reference["edit_id"]]
        row = {
            "edit_id": edit.edit_id,
            "level": prompt_level(edit.cefr_level),
            "sentence": detokenize(edit.original),
            "corrected": detokenize(edit.corrected),
            "edit": f"{detokenize(edit.original_text) or '-'} -> "
            f"{detokenize(edit.corrected_text) or '-'}",
            "errant_type": edit.error_type,
            "gee_error_type": reference["error_type"],
            "gee_annotator_flags": "; ".join(reference["flags"]),
            "gee_explanation": reference["explanation"],
            "preferred": "",
            "comment": "",
        }
        for system, system_rows in outputs.items():
            parsed = system_rows.get(edit.edit_id, {}).get("parsed")
            row[f"system_{system}"] = shown_text(parsed, "explain")
        rows.append(row)
    return rows, columns


def compare(args: argparse.Namespace) -> None:
    """Score GEE and the systems, write the aggregate report and the teacher's sheet."""
    references = read_rows(args.gee_dir / "references.jsonl")
    if not references:
        raise SystemExit(f"No references in {args.gee_dir}; run the prepare step first")
    wanted = {r["edit_id"] for r in references}
    edits = {e.edit_id: e for e in read_jsonl(args.edits) if e.edit_id in wanted}
    outputs = {}
    for system in args.systems:
        rows = read_rows(args.outputs / system / f"{RUN}.jsonl")
        if rows:
            outputs[system] = {row["edit_id"]: row for row in rows}
        else:
            print(
                f"No GEE outputs for system {system}; run: evaluate_on_gee.py run --system {system}"
            )

    results = compare_scores(references, edits, outputs, args.resamples, args.seed)
    counts_path = args.gee_dir / "prepare_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8")) if counts_path.is_file() else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gee_comparison.json").write_text(
        json.dumps(results | {"counts": counts}, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "gee_comparison.md").write_text(
        markdown_report(results, counts), encoding="utf-8"
    )
    rows, columns = side_by_side(references, edits, outputs)
    write_csv(rows, args.gee_dir / "side_by_side.csv", columns)
    print(f"Wrote {args.out_dir / 'gee_comparison.md'} and {args.gee_dir / 'side_by_side.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the systems with GEE's explanations.")
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--gee-dir", type=Path, default=GEE_DIR, help="IDs, references, sheet")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="system outputs")
    steps = parser.add_subparsers(dest="step", required=True)

    prep = steps.add_parser("prepare", help="Pair GEE's edits with the edit table.")
    prep.add_argument("--gee", type=Path, default=GEE_FILE)
    prep.add_argument("--gold-ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    prep.set_defaults(func=prepare)

    gen = steps.add_parser("run", help="Run one system on the GEE edits.")
    gen.add_argument("--system", required=True)
    gen.add_argument("--config", type=Path, default=Path("configs/systems.yaml"))
    gen.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=run)

    comp = steps.add_parser("compare", help="Score GEE and the systems on the same edits.")
    comp.add_argument("--systems", nargs="+", default=["A", "B", "C", "D", "R"])
    comp.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    comp.add_argument("--resamples", type=int, default=10_000)
    comp.add_argument("--seed", type=int, default=0)
    comp.set_defaults(func=compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
