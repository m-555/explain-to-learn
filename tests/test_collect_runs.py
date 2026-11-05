"""The run registry (scripts/collect_runs.py) on run files in the formats the runs write."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from explain_to_learn.metrics import summarize, write_report

ROOT = Path(__file__).parents[1]


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


collect_runs = load(ROOT / "scripts" / "collect_runs.py")


def item(covered: float, concept: float, faithful: float) -> dict:
    """One per-edit score, with the keys of metrics.score_item."""
    return {
        "covered": covered,
        "concept_correct": concept,
        "faithful": faithful,
        "contradicts_rule": None,
        "within_length": covered or None,
        "terms": 1 if covered else None,
        "off_level_terms": 0 if covered else None,
    }


def write_metrics(results: Path, run: str, scores: dict[str, list[dict]]) -> None:
    """A metrics file as scripts/compute_metrics.py writes it."""
    systems = {
        system: {"overall": summarize(items), "by_level": {}, "by_error_category": {}}
        for system, items in scores.items()
    }
    write_report({"run": run, "edits": 4, "systems": systems, "comparisons": {}}, results)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    pytest.importorskip("yaml")
    train_lora = load(ROOT / "train" / "train_lora.py")
    adapter = tmp_path / "adapters" / "d-r16"
    adapter.mkdir(parents=True)
    # The keys train/train_lora.py writes.
    info = {
        "run": "d-r16",
        "date": "2026-11-05T10:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567-dirty",
        "system": "d",
        "variant": "r16",
        "backend": "unsloth",
        "config": train_lora.load_config(train_lora.CONFIG_PATH, "r16"),
        "examples": {"train": 7600, "validation": 400, "longer_than_max_seq_length": 0},
        "train_loss": 0.41234,
        "eval_loss": 0.45678,
        "train_minutes": 23.4,
        "peak_gpu_memory_gb": {"allocated": 14.2, "reserved": 16.75},
        "gpu": "NVIDIA GeForce RTX 5090",
        "versions": {"torch": "2.14.1"},
    }
    (adapter / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    dpo = tmp_path / "adapters" / "d-dpo-beta01"
    dpo.mkdir()
    # The keys train/train_dpo.py writes, from a run made before date and commit were recorded.
    dpo_info = {
        "run": "d-dpo-beta01",
        "variant": "beta01",
        "adapter_trained_from": "outputs/adapters/d-r16",
        "config": {"training": {"beta": 0.1, "learning_rate": 5e-06, "epochs": 2}},
        "pairs": {"train": 180, "validation": 20},
        "train_loss": 0.6,
        "eval_loss": 0.62,
        "train_minutes": 4.0,
        "peak_gpu_memory_gb": {"allocated": 20.0, "reserved": 22.5},
    }
    (dpo / "run_info.json").write_text(json.dumps(dpo_info), encoding="utf-8")

    results = tmp_path / "results"
    good, poor = [item(1.0, 1.0, 1.0)] * 3 + [item(0.0, 0.0, 0.0)], [item(1.0, 0.0, 0.0)] * 4
    write_metrics(results, "dev-explain-r16", {"C": poor, "D": good})
    write_metrics(results, "gold-explain", {"R": good, "A": poor})
    correction = {
        "run": "gec-t5-small",
        "corrector": {"name": "gec", "kind": "gec", "model": "Unbabel/gec-t5_small"},
        "limit": None,
        "overall": {"sentences": 4384, "tp": 1, "fp": 1, "fn": 2, "precision": 0.5},
        "by_level": {},
    }
    correction["overall"] |= {"recall": 1 / 3, "f0.5": 0.4545}
    (results / "correction_gec-t5-small.json").write_text(json.dumps(correction))
    return tmp_path


def rows_of(markdown: str, section: str) -> list[list[str]]:
    """The body rows of the table under a `## section` heading."""
    text = markdown.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in text.splitlines() if line.startswith("| ")][2:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def test_registry_rows_from_the_run_files(runs, capsys):
    out = runs / "runs.md"
    collect_runs.main(
        ["--adapters", str(runs / "adapters"), "--results", str(runs / "results")]
        + ["--out", str(out)]
    )
    markdown = out.read_text(encoding="utf-8")

    training = rows_of(markdown, "Training runs")
    assert [row[2] for row in training] == ["d-dpo-beta01", "d-r16"]  # undated first
    dpo, lora = training
    assert lora[:5] == ["2026-11-05", "0123456-dirty", "d-r16", "D", "r16"]
    assert lora[5] == "rank 16, alpha 32; bf16 base; lr 0.0002, 2 epochs, batch 32"
    assert lora[6:] == ["7600/400", "0.412", "0.457", "16.75", "23.4", "75.0", "75.0"]
    assert dpo[:4] == ["–", "–", "d-dpo-beta01", "D"]
    assert dpo[5].startswith("DPO from outputs/adapters/d-r16, beta 0.1")
    assert dpo[6] == "180/20" and dpo[-2:] == ["–", "–"]

    evaluation = rows_of(markdown, "Evaluation runs")
    assert [row[:2] for row in evaluation] == [
        ["dev-explain-r16", "C"],
        ["dev-explain-r16", "D"],
        ["gold-explain", "A"],
        ["gold-explain", "R"],
    ]
    assert evaluation[1][2:6] == ["4", "75.0", "75.0", "75.0"]

    assert rows_of(markdown, "Correction runs") == [
        ["gec-t5-small", "gec", "Unbabel/gec-t5_small", "4384", "0.5000", "0.3333", "0.4545"]
    ]
    assert "2 training, 4 evaluation and 1 correction rows" in capsys.readouterr().out


def test_without_runs_the_tables_have_headers_only(tmp_path):
    out = tmp_path / "runs.md"
    collect_runs.main(
        ["--adapters", str(tmp_path / "none"), "--results", str(tmp_path / "none")]
        + ["--out", str(out)]
    )
    markdown = out.read_text(encoding="utf-8")

    for section in ("Training runs", "Evaluation runs", "Correction runs"):
        assert rows_of(markdown, section) == []


def test_committed_registry_is_up_to_date():
    """eval/results/runs.md is what the script writes from the committed result files."""
    committed = (ROOT / "eval" / "results" / "runs.md").read_text(encoding="utf-8")
    metrics = collect_runs.read_metrics(ROOT / "eval" / "results")
    evaluation = collect_runs.evaluation_rows(metrics)
    correction = collect_runs.correction_rows(ROOT / "eval" / "results")
    expected = collect_runs.markdown(rows_of(committed, "Training runs"), evaluation, correction)

    assert committed == expected
