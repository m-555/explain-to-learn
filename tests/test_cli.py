"""The `explain-to-learn` command with a fake corrector, annotator and model.

The models are replaced where the command loads them, so the tests need neither a GPU
nor ERRANT. All sentences are invented.
"""

import json
import tomllib
from pathlib import Path

import pytest
from test_pipeline import RecordingModel, TableCorrector, WordAnnotator

from explain_to_learn import cli
from explain_to_learn.edits import write_jsonl
from explain_to_learn.generate import OpenAIBackend
from explain_to_learn.pipeline import Explainer, Span, make_edits

pytest.importorskip("networkx")
pytest.importorskip("yaml")

ROOT = Path(__file__).parents[1]
GRAPH = ["--graph", str(Path(__file__).parent / "fixtures" / "graph")]


@pytest.fixture
def fakes(monkeypatch):
    """Replace the model loaders; returns the models so tests can inspect them."""
    loaded = {
        "model": RecordingModel(),
        "corrector": TableCorrector({"She like tea.": "She likes tea."}),
    }

    def explainer(config, name, backend=None):
        loaded["system"] = (name, backend)
        return Explainer(loaded["model"], grounded=True, system=name)

    def corrector(config, name):
        loaded["corrector_name"] = name
        return loaded["corrector"]

    monkeypatch.setattr(cli, "load_explainer", explainer)
    monkeypatch.setattr(cli, "load_corrector", corrector)
    monkeypatch.setattr(cli, "load_annotator", WordAnnotator)
    return loaded


def test_explain_prints_markdown_feedback(fakes, capsys):
    assert cli.main(["explain", "She like tea.", "--level", "b1", "--mode", "hint", *GRAPH]) == 0

    out = capsys.readouterr().out
    assert out.startswith("**Corrected:** She likes tea.\n\n1. ~~like~~ → **likes**")
    assert "Look at the verb." in out
    assert fakes["system"] == ("D", None) and fakes["corrector_name"] == "gec"
    assert "Learner level: B1" in fakes["model"].prompts[0][1]["content"]


def test_explain_json_with_a_given_correction(fakes, capsys):
    args = ["explain", "I bought book.", "--level", "A2", "--corrected", "I bought a book."]
    cli.main([*args, "--system", "B", "--backend", "openai", "--json", *GRAPH])

    result = json.loads(capsys.readouterr().out)
    assert result["sentences"] == [{"original": "I bought book.", "corrected": "I bought a book."}]
    (row,) = result["results"]
    assert (row["error_type"], row["concept_name"], row["system"]) == (
        "M:DET",
        "Articles a, an and the",
        "B",
    )
    assert fakes["system"] == ("B", "openai")
    assert fakes["corrector"].calls == 0 and "corrector_name" not in fakes


def test_explain_reports_backend_errors(fakes, capsys):
    fakes["model"].reply = "not json"
    cli.main(["explain", "She like tea.", "--level", "C1", *GRAPH])
    captured = capsys.readouterr()
    assert "could not be read" in captured.out
    assert "warning: 1 edits got no valid answer: invalid_json" in captured.err


def test_summary_of_a_text_file(fakes, capsys, tmp_path):
    text, corrected = tmp_path / "text.txt", tmp_path / "corrected.txt"
    text.write_text("She like tea. I have two cat.", encoding="utf-8")
    corrected.write_text("She likes tea. I have two cats.", encoding="utf-8")

    cli.main(["summary", str(text), "--level", "A2", "--corrected", str(corrected), *GRAPH])
    out = capsys.readouterr().out
    assert out.startswith("# Teacher summary\n\nLevel A2 · 2 edits · 2 concepts")
    assert out.index("Singular and plural nouns") < out.index("Subject-verb agreement")
    assert "corrector_name" not in fakes

    cli.main(["summary", str(text), "--level", "A2", "--json", *GRAPH])
    result = json.loads(capsys.readouterr().out)
    assert [c["concept_id"] for c in result["concepts"]] == ["subject_verb_agreement"]
    assert fakes["corrector_name"] == "gec"


def test_summary_of_an_essay_in_the_edit_table(capsys, tmp_path):
    edits = make_edits(
        ["I", "has", "a", "cat", "."],
        ["I", "have", "a", "cat", "."],
        [Span(1, 2, "have", "R:VERB:SVA")],
        "native",
        text_id="9",
    )
    write_jsonl(edits, tmp_path / "edits.jsonl")
    args = ["summary", "--essay", "text-9", "--edits", str(tmp_path / "edits.jsonl"), *GRAPH]

    cli.main(args)
    out = capsys.readouterr().out
    assert "Level native · 1 edit · 1 concept" in out
    assert "sentence 1, word 2: *has* → *have*" in out
    with pytest.raises(SystemExit, match="no edits for essay text-1"):
        cli.main(["summary", "--essay", "text-1", "--edits", str(tmp_path / "edits.jsonl"), *GRAPH])


@pytest.mark.parametrize(
    "args",
    [
        ["summary", "--level", "B1"],
        ["summary", "file.txt"],
        ["summary", "file.txt", "--essay", "wi-1", "--level", "B1"],
        ["explain", "She like tea.", "--level", "B3"],
    ],
)
def test_bad_arguments_are_rejected(args):
    with pytest.raises(SystemExit):
        cli.main(args)


def test_models_and_endpoints_come_from_the_config():
    explainer = cli.load_explainer(ROOT / "configs" / "systems.yaml", "D", backend="openai")
    assert isinstance(explainer.backend, OpenAIBackend)
    assert explainer.backend.model == "system-d"
    assert explainer.backend.base_url == "http://localhost:8000/v1"
    assert explainer.grounded and explainer.system == "D"
    assert explainer.decoding.temperature == 0.0
    with pytest.raises(SystemExit, match="no system 'E'"):
        cli.load_explainer(ROOT / "configs" / "systems.yaml", "E")
    with pytest.raises(SystemExit, match="no corrector 'magic'"):
        cli.load_corrector(ROOT / "configs" / "systems.yaml", "magic")


def test_console_script_is_registered():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"] == {"explain-to-learn": "explain_to_learn.cli:main"}
