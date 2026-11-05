"""The report assets script on a synthetic results fixture and on synthetic edits.

The fixture is written by the project's own writers (compute_metrics.py, human_eval and
agreement.py), so the tests fail if those formats change. All sentences and numbers are
invented for the tests.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("yaml")

from explain_to_learn.edits import Edit, build_edits, write_jsonl  # noqa: E402
from explain_to_learn.human_eval import (  # noqa: E402
    KEY_COLUMNS,
    RATING_COLUMNS,
    read_ratings,
    write_csv,
)

ROOT = Path(__file__).parents[1]
SYSTEMS = ("A", "B", "C", "D", "R")
QUALITY = {"A": 0.3, "B": 0.5, "C": 0.6, "D": 0.8, "R": 0.9}
LEVELS = ("A2", "B1", "C1")
CATEGORIES = ("VERB:SVA", "DET")
PNG = b"\x89PNG"


def load_script(name, directory="scripts"):
    spec = importlib.util.spec_from_file_location(name, ROOT / directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


assets = load_script("make_report_assets")


class FakeGraph:
    """The GrammarGraph methods that compute_metrics.py calls."""

    def concept_for(self, error_type):
        return {"R:VERB:SVA": "agreement", "R:DET": "articles"}.get(error_type)

    def snippet(self, concept_id, level):
        name = {"agreement": "Subject-verb agreement", "articles": "Articles"}[concept_id]
        return {"concept_id": concept_id, "name": name, "rule": "", "example": {}}

    def level_terms(self, level):
        return {"verb"}


def make_edit(index: int) -> Edit:
    level, category = LEVELS[index % 3], CATEGORIES[index % 2]
    original, new = ("zebra walk", "walks") if category == "VERB:SVA" else ("zebra a", "the")
    return Edit(
        edit_id=f"wi-E{index}-0-0",
        corpus="wi",
        essay_id=f"E{index}",
        split="dev",
        cefr_level=level,
        sent_idx=0,
        original=f"The {original} .",
        corrected=f"The {original} .",
        start=2,
        end=3,
        original_text=original.split()[1],
        corrected_text=new,
        error_type=f"R:{category}",
        operation="R",
        error_category=category,
    )


def good(system: str, index: int) -> bool:
    """A fixed pattern of good and bad items, more good ones for better systems."""
    return (index * 7 + SYSTEMS.index(system)) % 20 < 20 * QUALITY[system]


def output_row(system: str, edit: Edit, index: int) -> dict:
    """A run_system.py record: faithful for good items, vague or missing otherwise."""
    if index % 10 == 9 and not good(system, index):
        return {"edit_id": edit.edit_id, "mode": "explain", "parsed": None}
    concept = FakeGraph().snippet(FakeGraph().concept_for(edit.error_type), "B1")["name"]
    text = f'Use "{edit.corrected_text}", not "{edit.original_text}".'
    if not good(system, index):
        concept, text = "Grammar", "Check this word."
    parsed = {"concept": concept, "explanation": text, "hint": ""}
    return {"edit_id": edit.edit_id, "mode": "explain", "parsed": parsed}


def ratings_for(system: str, index: int, shift: int = 0) -> dict:
    high = good(system, index + shift)
    flag = "yes" if not high and index % 5 == 0 else "no"
    values = ("yes", "yes", "partly", "yes") if high else ("partly", "no", "no", "partly")
    dimensions = ("rule_correct", "error_located", "helps_fix", "fits_level")
    return dict(zip(dimensions, values, strict=True)) | {"invented_rule": flag, "comment": ""}


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Result files in the formats of the evaluation scripts, under tmp_path."""
    monkeypatch.chdir(tmp_path)
    edits = [make_edit(i) for i in range(60)]
    write_jsonl(edits, tmp_path / "edits.jsonl")
    (tmp_path / "gold.txt").write_text("".join(f"{e.edit_id}\n" for e in edits))
    for system in SYSTEMS:
        out = tmp_path / "outputs" / system / "gold.jsonl"
        out.parent.mkdir(parents=True)
        rows = [output_row(system, edit, i) for i, edit in enumerate(edits)]
        out.write_text("".join(json.dumps(row) + "\n" for row in rows))

    compute_metrics = load_script("compute_metrics")
    monkeypatch.setattr(compute_metrics, "open_graph", lambda directory: FakeGraph())
    compute_metrics.main(
        ["--run", "gold", "--edits", "edits.jsonl", "--ids", "gold.txt", "--resamples", "200"]
        + ["--terms", str(ROOT / "configs" / "grammar_terms.txt"), "--out-dir", "results"]
    )

    key, rater1, rater2, judge = [], [], [], []
    for number, (index, system) in enumerate(
        [(i, s) for i in range(40) for s in SYSTEMS[:4]], start=1
    ):
        item_id = f"item-{number:03d}"
        edit = edits[index]
        key.append(
            {
                "item_id": item_id,
                "system": system,
                "edit_id": edit.edit_id,
                "error_category": edit.error_category,
                "level": edit.cefr_level,
                "mode": "explain",
            }
        )
        rater1.append({"item_id": item_id} | ratings_for(system, index))
        judge.append({"item_id": item_id} | ratings_for(system, index, shift=number % 3))
        if number % 3 == 0:
            rater2.append({"item_id": item_id} | ratings_for(system, index, shift=number % 2))
    human = tmp_path / "outputs" / "human_eval"
    write_csv(key, human / "key.csv", KEY_COLUMNS)
    for name, rows in (("rater1", rater1), ("rater2", rater2), ("llm_judge", judge)):
        write_csv(rows, human / f"ratings_{name}.csv", ("item_id", *RATING_COLUMNS))

    agreement = load_script("agreement")
    first = read_ratings(human / "ratings_rater1.csv")
    for name, second in (("raters", "rater2"), ("judge", "llm_judge")):
        rows = agreement.agreement_rows(first, read_ratings(human / f"ratings_{second}.csv"))
        (tmp_path / "results" / f"agreement_{name}.md").write_text(agreement.markdown(name, rows))
    return tmp_path


def results_args(root: Path) -> list[str]:
    human = root / "outputs" / "human_eval"
    return [
        "--figures-dir",
        str(root / "figures"),
        "results",
        "--metrics",
        str(root / "results" / "metrics_gold.json"),
        "--ratings",
        str(human / "ratings_rater1.csv"),
        "--key",
        str(human / "key.csv"),
        "--judge-ratings",
        str(human / "ratings_llm_judge.csv"),
        "--agreement-raters",
        str(root / "results" / "agreement_raters.md"),
        "--agreement-judge",
        str(root / "results" / "agreement_judge.md"),
        "--table",
        str(root / "assets" / "results_table.md"),
        "--tables-dir",
        str(root / "assets" / "tables"),
        "--min-group",
        "5",
        "--resamples",
        "300",
    ]


def test_no_result_files_writes_nothing(tmp_path, capsys):
    assert assets.main(results_args(tmp_path)) == []

    assert "No result files are available" in capsys.readouterr().out
    assert not (tmp_path / "assets").exists()
    assert not (tmp_path / "figures").exists()


def test_results_table_tables_and_figures_from_the_fixture(results):
    written = assets.main(results_args(results))

    metrics = json.loads((results / "results" / "metrics_gold.json").read_text())
    table = (results / "assets" / "results_table.md").read_text()
    lines = table.splitlines()
    assert lines[0] == "| Metric | A | B | C | D | R |"
    coverage = next(line for line in lines if line.startswith("| Coverage"))
    for system in SYSTEMS:
        value = 100 * metrics["systems"][system]["overall"]["covered"]
        assert f"| {value:.1f} (" in coverage or f"| {value:.1f} |" in coverage
    rule = next(line for line in lines if line.startswith("| Rule correct"))
    assert rule.endswith("| – |")  # R is not in the human evaluation
    assert "60 gold edits" in table and "40 rated items per system" in table

    tables = results / "assets" / "tables"
    main_table = (tables / "results_main.tex").read_text()
    assert "\\toprule" in main_table and "\\bottomrule" in main_table
    assert main_table.count("\\midrule") == 2  # after the header and between the sections
    differences = (tables / "paired_differences.tex").read_text()
    assert "B$-$A & C$-$A & D$-$C & D$-$B" in differences
    assert "Cohen's kappa" in (tables / "agreement.tex").read_text()

    figures = {path.name for path in (results / "figures").iterdir()}
    assert figures == {
        "paired_differences_automatic.png",
        "paired_differences_human.png",
        "results_by_level.png",
        "results_by_category.png",
        "human_by_level.png",
        "human_by_category.png",
        "judge_agreement.png",
        "judge_vs_rater_means.png",
    }
    for name in figures:
        assert (results / "figures" / name).read_bytes().startswith(PNG)
    assert len(written) == len(figures) + 4  # the Markdown table and three LaTeX tables
    for path in [results / "assets" / "results_table.md", *tables.iterdir()]:
        assert "zebra" not in path.read_text()


def test_metrics_alone_give_the_automatic_rows(results):
    args = results_args(results)
    for option in ("--ratings", "--agreement-raters", "--agreement-judge"):
        args[args.index(option) + 1] = str(results / "missing")
    assets.main(args)

    table = (results / "assets" / "results_table.md").read_text()
    assert "| Coverage (%) |" in table and "Rule correct" not in table
    figures = {path.name for path in (results / "figures").iterdir()}
    assert figures == {
        "paired_differences_automatic.png",
        "results_by_level.png",
        "results_by_category.png",
    }


def test_agreement_alone_gives_no_results_table(results):
    args = results_args(results)
    for option in ("--metrics", "--ratings"):
        args[args.index(option) + 1] = str(results / "missing")
    assets.main(args)

    assert not (results / "assets" / "results_table.md").exists()
    agreement = (results / "assets" / "tables" / "agreement.tex").read_text()
    assert "\\multicolumn{4}{c}{Rater 2 vs rater 1}" in agreement
    assert "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}" in agreement
    assert {path.name for path in (results / "figures").iterdir()} == {"judge_agreement.png"}


def test_human_differences_are_paired_by_edit(results):
    rows = assets.load_ratings(
        results / "outputs" / "human_eval" / "ratings_rater1.csv",
        results / "outputs" / "human_eval" / "key.csv",
    )
    result = assets.human_differences(rows, "D", "A", n_resamples=200)["rule_correct"]
    means = {
        system: sum(r["rule_correct"] for r in rows if r["system"] == system) / 40
        for system in ("A", "D")
    }

    assert result["n"] == 40
    assert result["diff"] == pytest.approx(means["D"] - means["A"])
    assert result["low"] <= result["diff"] <= result["high"]


def test_share_interval_matches_resampling_the_scores():
    from_share = assets.share_interval(0.3, 200, n_resamples=20_000)
    from_scores = assets.mean_interval([1] * 60 + [0] * 140, n_resamples=20_000)

    assert from_share.value == pytest.approx(from_scores.value)
    assert from_share.low == pytest.approx(from_scores.low, abs=0.01)
    assert from_share.high == pytest.approx(from_scores.high, abs=0.01)
    assert assets.share_interval(None, 0).value is None


def test_estimates_are_formatted_with_their_intervals():
    estimate = assets.Estimate(0.853, 0.81, 0.893, 300)

    assert assets.format_estimate(estimate, percent=True) == "85.3 (81.0–89.3)"
    assert assets.format_estimate(assets.Estimate(1.625, 1.5, 1.75), False) == "1.62 (1.50–1.75)"
    assert assets.format_estimate(None, percent=True) == "–"
    latex = assets.format_estimate(estimate, percent=True, latex=True)
    assert latex == "85.3 {\\scriptsize (81.0--89.3)}"
    assert assets.format_difference(assets.Estimate(-0.02, -0.05, 0.01), True) == (
        "-2.0 [-5.0, 1.0]"
    )


def test_read_agreement_parses_the_agreement_table(tmp_path):
    path = tmp_path / "agreement.md"
    path.write_text(
        "# Rater 1 vs rater 2\n\n"
        "| Dimension | Items | Exact agreement | Cohen's kappa | Quadratic-weighted kappa |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        "| rule_correct | 50 | 0.80 | 0.61 | 0.72 |\n"
        "| error_located | 0 | n/a | n/a | n/a |\n"
        "| invented_rule | 50 | 0.96 | 0.48 | – |\n"
    )
    table = assets.read_agreement(path)

    assert table["rule_correct"] == {
        "items": 50,
        "exact": 0.8,
        "kappa": 0.61,
        "weighted_kappa": 0.72,
    }
    assert table["error_located"]["kappa"] is None
    assert table["invented_rule"]["weighted_kappa"] is None
    assert assets.read_agreement(tmp_path / "missing.md") is None


def test_data_statistics_and_figures(raw_dir, tmp_path):
    # conftest.raw_dir: six W&I files (an A2 essay with four edits and a C2 essay with
    # one), one LOCNESS file (one native edit) and three FCE files (two B2 essays each).
    edits = build_edits(raw_dir)

    assert assets.edits_per_essay(edits) == {
        "A2": [4] * 6,
        "B2": [1] * 6,
        "C2": [1] * 6,
        "native": [1],
    }
    shares = assets.operation_shares(edits)
    assert shares["A2"] == {"M": 0.0, "R": 0.75, "U": 0.25}
    assert shares["B2"] == {"M": 0.0, "R": 0.5, "U": 0.5}
    assert assets.top_categories(edits, n=2)["A2"] == [("VERB:SVA", 0.5), ("VERB:FORM", 0.25)]
    categories, levels, matrix = assets.category_level_shares(edits)
    assert levels == ["A2", "B2", "C2", "native"]
    assert categories[0] == "VERB:SVA"
    assert matrix.sum(axis=0) == pytest.approx([1.0] * 4)

    edits_path = tmp_path / "edits.jsonl"
    write_jsonl(edits, edits_path)
    assets.main(["--figures-dir", str(tmp_path / "figures"), "data", "--edits", str(edits_path)])
    written = sorted(path.name for path in (tmp_path / "figures").iterdir())
    assert written == sorted(assets.DATA_FIGURES)
    for name in written:
        assert (tmp_path / "figures" / name).read_bytes().startswith(PNG)


def test_data_step_without_an_edit_table_writes_nothing(tmp_path, capsys):
    assets.main(["--figures-dir", str(tmp_path / "figures"), "data", "--edits", "missing.jsonl"])

    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "figures").exists()
