"""End-to-end runs of the scripts with a fake graph and a fake model."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from explain_to_learn.edits import Edit, write_jsonl

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGraph:
    """The GrammarGraph contract with two concepts."""

    concepts = {"R:VERB:SVA": "subject_verb_agreement", "M:DET": "articles"}
    names = {"subject_verb_agreement": "Subject-verb agreement", "articles": "Articles"}

    def concept_for(self, error_type):
        return self.concepts.get(error_type)

    def snippet(self, concept_id, level):
        return {
            "concept_id": concept_id,
            "name": self.names[concept_id],
            "rule": "A rule.",
            "example": {"wrong": "x", "right": "y"},
            "prerequisites": [],
        }

    def level_terms(self, level):
        return {"verb"}


class FakeModel:
    """Answers well for subject-verb agreement edits and badly for the rest."""

    def complete(self, messages, decoding):
        user = messages[1]["content"]
        if "R:VERB:SVA" in user:
            hint = "Mode: hint" in user
            return json.dumps(
                {
                    "concept": "Subject-verb agreement",
                    "explanation": "" if hint else 'With "she", use "does", not "do".',
                    "hint": 'Look at "do" after "she".' if hint else "",
                }
            )
        return "I am not sure."


def make_edit(edit_id, essay, split, error_type, level="B1"):
    """Synthetic edits; the sentences are invented for the tests."""
    operation, category = error_type.split(":", 1)
    if error_type == "M:DET":
        original, start, end, old, new = "I have cat .", 2, 2, "", "a"
    else:
        original, start, end, old, new = "She do n't like tea .", 1, 2, "do", "does"
    return Edit(
        edit_id=edit_id,
        corpus="wi",
        essay_id=essay,
        split=split,
        cefr_level=level,
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=old,
        corrected_text=new,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    edits = [
        make_edit("wi-1-0-0", "1", "dev", "R:VERB:SVA", "A2"),
        make_edit("wi-1-0-1", "1", "dev", "M:DET", "B2"),
        *[make_edit(f"wi-2-0-{i}", "2", "train", "R:VERB:SVA") for i in range(6)],
        *[make_edit(f"wi-2-1-{i}", "2", "train", "M:DET") for i in range(4)],
        make_edit("wi-2-2-0", "2", "train", "R:OTHER"),
    ]
    write_jsonl(edits, tmp_path / "edits.jsonl")
    (tmp_path / "gold.txt").write_text("wi-1-0-0\nwi-1-0-1\n")
    (tmp_path / "train.txt").write_text("wi-2\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def patch(module, monkeypatch):
    monkeypatch.setattr(module, "open_graph", lambda directory: FakeGraph())
    if hasattr(module, "make_backend"):
        monkeypatch.setattr(module, "make_backend", lambda system: FakeModel())
    return module


def test_run_system_then_compute_metrics(workspace, monkeypatch):
    pytest.importorskip("yaml")
    run_system = patch(load_script("run_system"), monkeypatch)
    compute_metrics = patch(load_script("compute_metrics"), monkeypatch)
    common = ["--edits", "edits.jsonl", "--ids", "gold.txt", "--run", "gold"]
    config = ["--config", str(ROOT / "configs" / "systems.yaml")]

    for system in ("A", "B"):
        run_system.main(["--system", system, *common, *config])
    run_system.main(["--system", "B", *common, *config])  # resumes: nothing new

    rows = [json.loads(line) for line in Path("outputs/B/gold.jsonl").read_text().splitlines()]
    assert [row["edit_id"] for row in rows] == ["wi-1-0-0", "wi-1-0-1"]
    assert rows[0]["grounded"] and rows[0]["parsed"]["concept"] == "Subject-verb agreement"
    assert rows[1]["parsed"] is None and rows[1]["attempts"] == 2

    compute_metrics.main(
        [*common, "--systems", "A", "B", "C", "--compare", "B:A", "--resamples", "200"]
        + ["--terms", str(ROOT / "configs" / "grammar_terms.txt"), "--out-dir", "results"]
    )
    results = json.loads(Path("results/metrics_gold.json").read_text())
    overall = results["systems"]["B"]["overall"]
    assert (overall["covered"], overall["concept_correct"], overall["faithful"]) == (0.5, 0.5, 0.5)
    assert results["systems"]["C"]["overall"]["covered"] == 0.0
    assert set(results["systems"]["B"]["by_level"]) == {"A2", "B2"}
    assert results["comparisons"]["B - A"]["faithful"]["diff"] == 0.0
    report = Path("results/metrics_gold.md").read_text()
    assert "## Paired differences" in report and "tea" not in report


def test_make_silver_generate_filter_and_score(workspace, monkeypatch):
    pytest.importorskip("yaml")
    make_silver = patch(load_script("make_silver"), monkeypatch)
    base = ["--edits", "edits.jsonl", "--results", "results", "--handcheck", "check.csv"]

    make_silver.main(
        [*base, "generate", "--n", "100", "--train-essays", "train.txt", "--exclude", "gold.txt"]
        + ["--config", str(ROOT / "configs" / "systems.yaml"), "--out", "silver/raw.jsonl"]
    )
    raw = [json.loads(line) for line in Path("silver/raw.jsonl").read_text().splitlines()]
    assert len(raw) == 10  # the R:OTHER edit has no concept, gold edits are excluded

    make_silver.main([*base, "filter", "--raw", "silver/raw.jsonl", "--kept", "silver/kept.jsonl"])
    counts = json.loads(Path("results/silver_filter.json").read_text())
    assert counts == {"total": 10, "kept": 6, "invalid_json": 4}
    assert len(Path("silver/kept.jsonl").read_text().splitlines()) == 6

    with Path("check.csv").open(newline="") as lines:
        rows = list(csv.DictReader(lines))
    assert len(rows) == 6
    for i, row in enumerate(rows):
        row["correct"] = "1" if i else "0"
    with Path("check.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    make_silver.main([*base, "score"])
    score = json.loads(Path("results/silver_handcheck.json").read_text())
    assert (score["rated"], score["correct"]) == (6, 5)
