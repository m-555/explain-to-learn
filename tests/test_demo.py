"""The demo in preview mode, its settings and its interface, without models or a GPU.

Preview mode replaces the corrector, ERRANT and the explanation model with the stubs in
`demo/stubs.py` and runs the rest of `explain_to_learn.pipeline` on the real grammar
graph. All sentences are invented.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("yaml")

ROOT = Path(__file__).parents[1]
DEMO_DIR = ROOT / "demo"


@pytest.fixture
def app(monkeypatch):
    """demo/app.py, imported the way `python demo/app.py` imports it."""
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.chdir(ROOT)
    yield importlib.import_module("app")
    for name in ("app", "stubs"):
        sys.modules.pop(name, None)


@pytest.fixture
def demo(app):
    return app.build_demo(app.parse_args(["--fake"]))


class RecordingBackend:
    """Remembers the prompts it gets and answers with a fixed hint."""

    def __init__(self):
        self.prompts = []

    def complete(self, messages, decoding):
        self.prompts.append(messages)
        return json.dumps({"concept": "Agreement", "explanation": "", "hint": "Look at the verb."})


def test_preview_corrects_and_explains_each_edit(demo):
    corrected, feedback = demo.feedback("She don't  like to going.", True, "", "A2", "explain")

    assert corrected == "She doesn't like to go."
    assert "**1.** ~~do~~ → **does** · *Subject-verb agreement*" in feedback
    assert "**2.** ~~going~~ → **go** · *Base form after to*" in feedback
    assert feedback.count("Stub explanation") == 2


def test_hint_mode_and_several_sentences(demo):
    corrected, feedback = demo.feedback("I has a apple. We was happy!", True, "", "A1", "hint")

    assert corrected == "I have an apple. We were happy!"
    assert feedback.count("Stub hint") == 3


def test_with_correction_off_the_users_correction_is_used(demo):
    corrected, feedback = demo.feedback("He go home.", False, "He goes home.", "B1", "explain")
    assert corrected == "He goes home."
    assert "~~go~~ → **goes**" in feedback
    assert demo.feedback("He go home.", False, " ", "B1", "explain")[1].startswith("Enter the")
    assert demo.feedback("She knows.", True, "", "B2", "explain") == (
        "She knows.",
        "No corrections needed.",
    )
    assert demo.feedback("   ", True, "", "B2", "explain") == ("", "")


def test_model_gets_the_grounded_prompt_at_the_chosen_level(app, demo):
    backend = RecordingBackend()
    demo.explainer = app.Explainer(backend, system="D")

    _, feedback = demo.feedback("We was happy.", True, "", "B1", "hint")
    user_message = backend.prompts[0][1]["content"]
    assert "Learner level: B1" in user_message and "Mode: hint" in user_message
    assert "Concept: Subject-verb agreement" in user_message
    assert 'replace "was" with "were"' in user_message
    assert "Look at the verb." in feedback


def test_teacher_summary_tab(demo):
    text = "I has a apple. She don't like it."
    summary = demo.summary(text, True, "", "A2")

    assert summary.startswith("# Teacher summary\n\nLevel A2 · 3 edits · 2 concepts")
    assert summary.index("Subject-verb agreement") < summary.index("Articles a, an and the")
    assert "sentence 2, word 2: *do* → *does*" in summary
    assert demo.summary(text, False, "", "A2").startswith("Enter the")


def test_settings_come_from_the_config_arguments_and_environment(app, monkeypatch):
    args = app.parse_args([])
    assert (args.fake, args.system, args.corrector) == (False, "D", "gec")
    system = app.system_settings(args)
    assert system["adapter"] == "outputs/adapters/d-r16" and system["backend"] == "transformers"

    monkeypatch.setenv("DEMO_FAKE", "1")
    monkeypatch.setenv("DEMO_ADAPTER", "user/explain-to-learn-d-r16")
    args = app.parse_args([])
    assert args.fake and app.system_settings(args)["adapter"] == "user/explain-to-learn-d-r16"

    served = app.system_settings(
        app.parse_args(["--backend", "openai", "--base-url", "http://x/v1"])
    )
    assert (served["served_model"], served["base_url"]) == ("system-d", "http://x/v1")
    renamed = app.system_settings(app.parse_args(["--backend", "openai", "--model", "d"]))
    assert renamed["served_model"] == "d" and renamed["model"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_app_builds_the_interface_in_preview_mode(app, demo):
    pytest.importorskip("gradio")

    interface = app.build_ui(demo, fake=True)
    assert interface.title == "Explain-to-Learn"
