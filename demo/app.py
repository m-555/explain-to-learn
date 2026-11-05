"""Gradio demo: feedback on a learner's text, and a summary for teachers.

Run locally (see demo/README.md):
    python demo/app.py --fake                     # stubs, no GPU or model files needed
    python demo/app.py                            # system D and the GEC model on the GPU
    python demo/app.py --backend openai           # system D on the vLLM server

The models and endpoints come from `configs/systems.yaml` (`--system`, `--corrector`);
`--backend`, `--model`, `--adapter` and `--base-url` override the system's settings.
Every option can also be set with an environment variable (DEMO_FAKE=1, DEMO_ADAPTER,
...), which is how a Hugging Face Space configures the app.

Both tabs run `explain_to_learn.pipeline`. With automatic correction on, the
correction model corrects the text; with it off, the user also enters the corrected
text and no correction model is used.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stubs import StubAnnotator, StubBackend, StubCorrector

from explain_to_learn.generate import load_systems
from explain_to_learn.io import open_graph
from explain_to_learn.pipeline import (
    LEVELS,
    Annotator,
    Corrector,
    ErrantAnnotator,
    Explainer,
    explain_text,
    feedback_markdown,
    load_correctors,
    make_corrector,
    text_edits,
)
from explain_to_learn.summary import markdown, summarize

if TYPE_CHECKING:
    import gradio as gr

try:  # On a ZeroGPU Space, a GPU is attached only while a decorated function runs.
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:

    def gpu(function: Callable[..., Any]) -> Callable[..., Any]:
        """Run the function as it is when ZeroGPU's `spaces` package is not installed."""
        return function


MODES = [("Explanation", "explain"), ("Hint", "hint")]
MAX_CHARS = 2000
EXAMPLES = [
    ["She don't like to going to school.", "A2", "explain"],
    ["We was very happy when we see the results.", "B1", "hint"],
    ["I has a apple in my bag. My brother don't like apples.", "A1", "explain"],
]
SUMMARY_EXAMPLE = (
    "Last weekend we was at the beach. My sister don't like swimming, so she read a book. "
    "I has a ice cream and we want to going back next year."
)
DESCRIPTION = (
    "Write a few sentences in English, choose your level and the kind of feedback. The "
    "demo corrects the text, splits each correction into edits and explains each edit, "
    "using the rules of a teacher-written grammar graph. A hint points to the error "
    "without giving the answer."
)
SUMMARY_DESCRIPTION = (
    "Paste a learner's text to see its corrections grouped by grammar concept, in a "
    "suggested teaching order: a concept comes after the concepts it builds on that also "
    "have errors. No explanation model is used."
)
FAKE_NOTICE = (
    "**Preview mode:** the correction and the edits come from simple rules and the "
    "feedback is placeholder text. The grammar graph and the teacher summary are real."
)
MISSING_CORRECTION = "Enter the corrected text, or turn on automatic correction."


def env(name: str, default: str | None = None) -> str | None:
    """Read DEMO_<NAME> from the environment."""
    return os.environ.get(f"DEMO_{name}", default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The demo's settings: command-line options, defaulting to DEMO_<NAME> variables."""
    parser = argparse.ArgumentParser(description="Explain-to-Learn demo.")
    parser.add_argument("--fake", action="store_true", default=env("FAKE") == "1")
    parser.add_argument("--config", default=env("CONFIG", "configs/systems.yaml"))
    parser.add_argument("--system", default=env("SYSTEM", "D"), help="explanation system")
    parser.add_argument("--backend", choices=("transformers", "openai"), default=env("BACKEND"))
    parser.add_argument("--model", default=env("MODEL"), help="base model, or served model name")
    parser.add_argument("--adapter", default=env("ADAPTER"), help="LoRA adapter path or Hub ID")
    parser.add_argument("--base-url", default=env("BASE_URL"), help="server address for openai")
    parser.add_argument("--corrector", default=env("CORRECTOR", "gec"), help="gec or llm")
    parser.add_argument("--graph", default=env("GRAPH", "grammar_graph"))
    parser.add_argument("--port", type=int, default=int(env("PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    return parser.parse_args(argv)


def system_settings(args: argparse.Namespace) -> dict:
    """The explanation system from the config, with the command-line overrides."""
    systems = load_systems(args.config)
    if args.system not in systems:
        raise SystemExit(
            f"error: no system {args.system!r} in {args.config}; "
            f"choose from {', '.join(sorted(systems))}"
        )
    system = dict(systems[args.system])
    if args.backend:
        system["backend"] = args.backend
    if args.model:
        system["served_model" if system["backend"] == "openai" else "model"] = args.model
    if args.adapter:
        system["adapter"] = args.adapter
    if args.base_url:
        system["base_url"] = args.base_url
    return system


@dataclass
class Demo:
    """The pipeline steps behind both tabs."""

    corrector: Corrector
    annotator: Annotator
    explainer: Explainer
    graph: Any

    def feedback(
        self, text: str, auto_correct: bool, corrected: str, level: str, mode: str
    ) -> tuple[str, str]:
        """The corrected text and the feedback as Markdown."""
        if not text.strip():
            return "", ""
        if not auto_correct and not corrected.strip():
            return "", MISSING_CORRECTION
        result = explain_text(
            text,
            level,
            mode,
            self.corrector,
            self.explainer,
            self.graph,
            annotator=self.annotator,
            corrected=None if auto_correct else corrected,
        )
        return result.corrected_text, feedback_markdown(result)

    def summary(self, text: str, auto_correct: bool, corrected: str, level: str) -> str:
        """The teacher summary as Markdown."""
        if not text.strip():
            return ""
        if not auto_correct and not corrected.strip():
            return MISSING_CORRECTION
        _, edits = text_edits(
            text,
            level,
            self.corrector,
            self.annotator,
            corrected=None if auto_correct else corrected,
        )
        return markdown(summarize(edits, self.graph, level))


def build_demo(args: argparse.Namespace) -> Demo:
    """Load the models named in the arguments; with --fake, use stubs."""
    graph = open_graph(args.graph)
    if args.fake:
        return Demo(StubCorrector(), StubAnnotator(), Explainer(StubBackend()), graph)
    correctors = load_correctors(args.config)
    if args.corrector not in correctors:
        raise SystemExit(
            f"error: no corrector {args.corrector!r} in {args.config}; "
            f"choose from {', '.join(sorted(correctors))}"
        )
    corrector = make_corrector(correctors[args.corrector])
    explainer = Explainer.from_system(args.system, system_settings(args))
    return Demo(corrector, ErrantAnnotator(), explainer, graph)


def build_ui(demo: Demo, fake: bool = False) -> gr.Blocks:
    """The Gradio interface: a feedback tab for learners and a summary tab for teachers."""
    import gradio as gr

    @gpu
    def feedback(
        text: str, auto_correct: bool, corrected: str, level: str, mode: str
    ) -> tuple[str, str]:
        return demo.feedback(text, auto_correct, corrected, level, mode)

    @gpu
    def summary(text: str, auto_correct: bool, corrected: str, level: str) -> str:
        return demo.summary(text, auto_correct, corrected, level)

    def show_correction(auto_correct: bool) -> gr.Textbox:
        return gr.Textbox(visible=not auto_correct)

    def text_inputs(label: str) -> tuple[gr.Textbox, gr.Checkbox, gr.Textbox]:
        text = gr.Textbox(label=label, lines=4, max_length=MAX_CHARS)
        auto_correct = gr.Checkbox(value=True, label="Correct the text automatically")
        corrected = gr.Textbox(
            label="Corrected text (used when automatic correction is off)",
            lines=4,
            max_length=MAX_CHARS,
            visible=False,
        )
        auto_correct.change(show_correction, auto_correct, corrected)
        return text, auto_correct, corrected

    with gr.Blocks(title="Explain-to-Learn") as app:
        gr.Markdown("# Explain-to-Learn")
        if fake:
            gr.Markdown(FAKE_NOTICE)
        with gr.Tab("Feedback"):
            gr.Markdown(DESCRIPTION)
            text, auto_correct, corrected = text_inputs("Your text")
            with gr.Row():
                level = gr.Radio(list(LEVELS), value="B1", label="Your level (CEFR)")
                mode = gr.Radio(MODES, value="explain", label="Feedback")
            button = gr.Button("Check", variant="primary")
            corrected_out = gr.Textbox(label="Corrected text", interactive=False)
            feedback_out = gr.Markdown()
            gr.Examples(EXAMPLES, inputs=[text, level, mode])
            inputs = [text, auto_correct, corrected, level, mode]
            button.click(feedback, inputs, [corrected_out, feedback_out])
        with gr.Tab("Teacher summary"):
            gr.Markdown(SUMMARY_DESCRIPTION)
            s_text, s_auto_correct, s_corrected = text_inputs("Learner's text")
            s_level = gr.Radio(list(LEVELS), value="B1", label="Learner's level (CEFR)")
            s_button = gr.Button("Summarise", variant="primary")
            summary_out = gr.Markdown()
            gr.Examples([[SUMMARY_EXAMPLE, "A2"]], inputs=[s_text, s_level])
            s_button.click(summary, [s_text, s_auto_correct, s_corrected, s_level], summary_out)
    return app


def main() -> None:
    """Start the demo on `--port`, with the models or, with --fake, the stubs."""
    args = parse_args()
    build_ui(build_demo(args), args.fake).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
