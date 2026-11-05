"""Run registry: one table of every training and evaluation run, from the files they wrote.

Reads
- `outputs/adapters/*/run_info.json`, written by `train/train_lora.py` and
  `train/train_dpo.py` (date, git commit, settings, losses, peak GPU memory, time);
- `eval/results/metrics_<run>.json`, written by `scripts/compute_metrics.py`;
- `eval/results/correction_<run>.json`, written by `scripts/evaluate_correction.py`;

and writes `eval/results/runs.md` with three tables: training runs, evaluation runs (one
row per run and system) and correction runs. A training row also shows the dev metrics
of its adapter when the runbook's naming was followed: adapter `<system>-<variant>`
scored as system `<SYSTEM>` in run `dev-explain-<variant>` (runbook step 7).

Nothing is estimated or filled in: a value a run did not record is shown as "–", and
with no runs the tables have headers only. All inputs hold aggregate numbers and
settings, never learner text, so the table can be committed. Run it again after each run.

Usage:
    python scripts/collect_runs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_ORDER = ("A", "B", "C", "D", "R")
MISSING = "–"

TRAINING_HEADER = (
    "Date",
    "Commit",
    "Run",
    "System",
    "Variant",
    "Settings",
    "Examples (train/val)",
    "Train loss",
    "Val loss",
    "Peak GPU memory (GB)",
    "Training time (min)",
    "Dev concept acc.",
    "Dev faithful",
)
EVALUATION_HEADER = (
    "Run",
    "System",
    "Edits",
    "Coverage",
    "Concept acc.",
    "Faithful",
    "Length fit",
    "Off-level terms",
)
CORRECTION_HEADER = ("Run", "Corrector", "Model", "Sentences", "Precision", "Recall", "F0.5")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def table(header: tuple[str, ...], rows: list[list[str]], text_columns: int) -> str:
    """A Markdown table: the first `text_columns` left-aligned, the numbers right-aligned."""
    alignment = ["---"] * text_columns + ["---:"] * (len(header) - text_columns)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(alignment) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def percent(value: float | None) -> str:
    """A share as a percentage with one decimal."""
    return MISSING if value is None else f"{100 * value:.1f}"


def number(value: float | None, digits: int = 3) -> str:
    """A number with a fixed number of decimals."""
    return MISSING if value is None else f"{value:.{digits}f}"


def short_commit(commit: str | None) -> str:
    """The first 7 characters of a commit hash, keeping a `-dirty` mark."""
    if not commit:
        return MISSING
    sha, dirty, _ = commit.partition("-dirty")
    return sha[:7] + ("-dirty" if dirty else "")


def settings(info: dict[str, Any]) -> str:
    """The settings that tell training runs apart, from the config in run_info.json."""
    config = info.get("config", {})
    training, lora, model = (config.get(key, {}) for key in ("training", "lora", "model"))
    parts = []
    if "adapter_trained_from" in info:
        parts.append(f"DPO from {info['adapter_trained_from']}, beta {training.get('beta')}")
    if lora:
        parts.append(f"rank {lora.get('rank')}, alpha {lora.get('alpha')}")
    if model:
        parts.append("4-bit base" if model.get("load_in_4bit") else "bf16 base")
    if training:
        batch = training.get("batch_size", 0) * training.get("gradient_accumulation", 1)
        parts.append(
            f"lr {training.get('learning_rate')}, {training.get('epochs')} epochs, batch {batch}"
        )
    return "; ".join(parts) or MISSING


def system_of(info: dict[str, Any]) -> str:
    """The system an adapter belongs to; DPO always continues training system D."""
    system = info.get("system") or ("d" if "adapter_trained_from" in info else "")
    return system.upper() or MISSING


def dev_scores(info: dict[str, Any], metrics: dict[str, dict]) -> tuple[str, str]:
    """Concept accuracy and faithfulness of the adapter on its dev run, if there is one."""
    run = metrics.get(f"dev-explain-{info.get('variant')}", {})
    overall = run.get("systems", {}).get(system_of(info), {}).get("overall")
    if not overall:
        return MISSING, MISSING
    return percent(overall.get("concept_correct")), percent(overall.get("faithful"))


def training_rows(adapters: Path, metrics: dict[str, dict]) -> list[list[str]]:
    """One row per adapter folder with a run_info.json, oldest first."""
    infos = [read_json(path) for path in sorted(adapters.glob("*/run_info.json"))]
    infos.sort(key=lambda info: (info.get("date") or "", info.get("run") or ""))
    rows = []
    for info in infos:
        examples = info.get("examples") or info.get("pairs") or {}
        memory = info.get("peak_gpu_memory_gb", {})
        rows.append(
            [
                (info.get("date") or MISSING)[:10],
                short_commit(info.get("git_commit")),
                info.get("run", MISSING),
                system_of(info),
                info.get("variant", MISSING),
                settings(info),
                f"{examples.get('train', MISSING)}/{examples.get('validation', MISSING)}",
                number(info.get("train_loss")),
                number(info.get("eval_loss")),
                number(memory.get("reserved"), 2),
                number(info.get("train_minutes"), 1),
                *dev_scores(info, metrics),
            ]
        )
    return rows


def evaluation_rows(metrics: dict[str, dict]) -> list[list[str]]:
    """One row per run and system of the automatic metrics."""
    rows = []
    for run, result in sorted(metrics.items()):
        systems = result.get("systems", {})
        order = [s for s in SYSTEM_ORDER if s in systems] + sorted(set(systems) - set(SYSTEM_ORDER))
        for system in order:
            overall = systems[system]["overall"]
            rows.append(
                [
                    run,
                    system,
                    str(overall.get("edits", MISSING)),
                    percent(overall.get("covered")),
                    percent(overall.get("concept_correct")),
                    percent(overall.get("faithful")),
                    percent(overall.get("within_length")),
                    percent(overall.get("off_level_term_share")),
                ]
            )
    return rows


def correction_rows(results: Path) -> list[list[str]]:
    """One row per correction run, with its overall ERRANT scores."""
    rows = []
    for path in sorted(results.glob("correction_*.json")):
        result = read_json(path)
        corrector, overall = result.get("corrector", {}), result.get("overall", {})
        rows.append(
            [
                result.get("run", MISSING),
                corrector.get("name", MISSING),
                corrector.get("model", MISSING),
                str(overall.get("sentences", MISSING)),
                number(overall.get("precision"), 4),
                number(overall.get("recall"), 4),
                number(overall.get("f0.5"), 4),
            ]
        )
    return rows


def read_metrics(results: Path) -> dict[str, dict]:
    """The automatic metrics files by run name."""
    return {
        path.stem.removeprefix("metrics_"): read_json(path)
        for path in results.glob("metrics_*.json")
    }


def markdown(training: list[list[str]], evaluation: list[list[str]], correction: list) -> str:
    """The run registry as Markdown."""
    sections = [
        "# Run registry",
        "Every training and evaluation run, collected by `scripts/collect_runs.py` from the "
        "files the runs wrote: `outputs/adapters/*/run_info.json` and "
        "`eval/results/metrics_*.json` and `correction_*.json`. Scores are percentages; a "
        "dash means the run did not record the value. Do not edit by hand: run "
        "`python scripts/collect_runs.py` (or `make runs`) after each run.",
        "## Training runs",
        "Peak GPU memory is the memory PyTorch reserved. The dev columns are the adapter's "
        "scores in run `dev-explain-<variant>` (runbook step 7).",
        table(TRAINING_HEADER, training, 6),
        "## Evaluation runs",
        table(EVALUATION_HEADER, evaluation, 2),
        "## Correction runs",
        "Span-based ERRANT scores on the W&I+LOCNESS dev sentences.",
        table(CORRECTION_HEADER, correction, 3),
    ]
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect every run into eval/results/runs.md.")
    parser.add_argument("--adapters", type=Path, default=Path("outputs/adapters"))
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/runs.md"))
    args = parser.parse_args(argv)

    metrics = read_metrics(args.results)
    training = training_rows(args.adapters, metrics)
    evaluation = evaluation_rows(metrics)
    correction = correction_rows(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(training, evaluation, correction), encoding="utf-8")
    print(
        f"Wrote {args.out}: {len(training)} training, {len(evaluation)} evaluation and "
        f"{len(correction)} correction rows"
    )


if __name__ == "__main__":
    main()
