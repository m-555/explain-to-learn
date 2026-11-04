"""The demo pipeline in fake mode, and the app's settings, without models or a GPU."""

import importlib
import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "demo"


@pytest.fixture
def pipeline(monkeypatch):
    """demo/pipeline.py, imported the way app.py imports it."""
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    yield importlib.import_module("pipeline")
    for name in ("pipeline", "app"):
        sys.modules.pop(name, None)


@pytest.fixture
def app(pipeline):
    return importlib.import_module("app")


class RecordingBackend:
    """Remembers the prompts it gets and answers with a fixed hint."""

    def __init__(self, reply: str | None = None):
        self.prompts = []
        self.reply = reply or json.dumps(
            {"concept": "Subject-verb agreement", "explanation": "", "hint": "Look at the verb."}
        )

    def complete(self, messages, decoding):
        self.prompts.append(messages)
        return self.reply


def test_fake_pipeline_corrects_and_explains_each_edit(pipeline):
    result = pipeline.fake_pipeline().run("She don't  like to going.", "A2", "explain")

    assert result.corrected == "She doesn't like to go."
    assert [(f.original_text, f.corrected_text, f.error_type) for f in result.feedback] == [
        ("do", "does", "R:VERB:SVA"),
        ("going", "go", "R:VERB:FORM"),
    ]
    assert [f.concept for f in result.feedback] == ["Subject-verb agreement", "Verb form"]
    assert all(f.grounded and f.text.startswith("Stub explanation") for f in result.feedback)


def test_hint_mode_shows_the_hint(pipeline):
    result = pipeline.fake_pipeline().run("I has a apple.", "A1", "hint")

    assert result.corrected == "I have an apple."
    assert len(result.feedback) == 2
    assert all(f.text.startswith("Stub hint") for f in result.feedback)


def test_model_gets_the_grounded_prompt_at_the_chosen_level(pipeline):
    backend = RecordingBackend()
    demo = pipeline.Pipeline(
        pipeline.FakeCorrector(), pipeline.FakeExtractor(), pipeline.FakeGraph(), backend
    )

    result = demo.run("We was happy.", "B1", "hint")

    user_message = backend.prompts[0][1]["content"]
    assert "Learner level: B1" in user_message and "Mode: hint" in user_message
    assert "Concept: Subject-verb agreement" in user_message
    assert 'replace "was" with "were"' in user_message
    assert result.feedback[0].text == "Look at the verb."


def test_ungrounded_pipeline_sends_no_snippet(pipeline):
    backend = RecordingBackend()
    demo = pipeline.Pipeline(
        pipeline.FakeCorrector(), pipeline.FakeExtractor(), pipeline.FakeGraph(), backend, False
    )

    result = demo.run("We was happy.", "B1", "explain")

    assert "grammar graph" not in backend.prompts[0][1]["content"]
    assert not result.feedback[0].grounded


def test_unreadable_answers_and_unchanged_sentences(pipeline):
    demo = pipeline.Pipeline(
        pipeline.FakeCorrector(),
        pipeline.FakeExtractor(),
        pipeline.FakeGraph(),
        RecordingBackend(reply="not json"),
    )

    result = demo.run("She don't know.", "B2", "explain")
    assert result.feedback[0].text == ""
    assert "could not be read" in pipeline.format_feedback(result)
    assert pipeline.format_feedback(demo.run("She knows.", "B2", "explain")) == (
        "No corrections needed."
    )
    assert pipeline.format_feedback(demo.run("   ", "B2", "explain")) == ""


def test_format_feedback_for_each_operation(pipeline):
    feedback = [
        pipeline.Feedback("go", "goes", "R:VERB:SVA", "Agreement", "Use goes.", True),
        pipeline.Feedback("", "the", "M:DET", "", "Add the.", False),
        pipeline.Feedback("very", "", "U:ADV", "Adverbs", "Remove very.", True),
    ]

    text = pipeline.format_feedback(pipeline.Result("x", feedback))
    assert "**1. ~~go~~ → **goes**** · Agreement\n\nUse goes." in text
    assert "**2. add **the**** · M:DET" in text
    assert "**3. remove ~~very~~** · Adverbs" in text


def test_make_edits_fills_the_edit_table_fields(pipeline):
    tokens = ["She", "do", "n't", "like", "it", "."]
    corrected = ["She", "does", "n't", "like", "it", "."]

    (edit,) = pipeline.make_edits(tokens, corrected, [(1, 2, "does", "R:VERB:SVA")], "C1")
    assert (edit.original_text, edit.corrected_text, edit.cefr_level) == ("do", "does", "C1")
    assert (edit.operation, edit.error_category) == ("R", "VERB:SVA")
    assert edit.corrected == "She does n't like it ."


def test_app_settings_come_from_arguments_or_the_environment(app, monkeypatch):
    assert app.parse_args([]).fake is False
    assert app.parse_args(["--fake"]).fake is True
    monkeypatch.setenv("DEMO_FAKE", "1")
    monkeypatch.setenv("DEMO_ADAPTER", "user/explain-to-learn-d-r16")
    args = app.parse_args([])
    assert args.fake and args.adapter == "user/explain-to-learn-d-r16"
    assert args.model == "Qwen/Qwen3-4B-Instruct-2507"


def test_app_builds_the_interface_in_fake_mode(app):
    pytest.importorskip("gradio")
    args = app.parse_args(["--fake"])

    demo = app.build_ui(app.build_pipeline(args), fake=args.fake)
    assert demo.title == "Explain-to-Learn"
