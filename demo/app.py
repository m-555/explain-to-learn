"""Gradio demo: feedback on a learner's sentence at the learner's CEFR level.

Run locally (see demo/README.md):
    python demo/app.py --fake                                   # stubs, no GPU needed
    python demo/app.py --adapter outputs/adapters/d-r16         # system D on the local GPU
    python demo/app.py --backend openai --model system-d        # system D on a vLLM server

Every option can also be set with an environment variable (DEMO_FAKE=1, DEMO_ADAPTER,
...), which is how a Hugging Face Space configures the app.
"""

from __future__ import annotations

import argparse
import os

from pipeline import (
    GEC_MODEL,
    ErrantExtractor,
    GecCorrector,
    ModelCorrector,
    Pipeline,
    fake_pipeline,
    format_feedback,
)

from explain_to_learn.generate import OpenAIBackend, TransformersBackend

try:  # On a ZeroGPU Space, a GPU is attached only while a decorated function runs.
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:

    def gpu(function):
        return function


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
MODES = [("Explanation", "explain"), ("Hint", "hint")]
EXAMPLES = [
    ["She don't like to going to school.", "A2", "explain"],
    ["We was very happy when we see the results.", "B1", "hint"],
    ["I has a apple in my bag.", "A1", "explain"],
]
DESCRIPTION = (
    "Write a sentence in English, choose your level and the kind of feedback. The demo "
    "corrects the sentence, splits the correction into edits and explains each edit, "
    "using the rules of a teacher-written grammar graph. A hint points to the error "
    "without giving the answer."
)
FAKE_NOTICE = (
    "**Preview mode:** every step is a stub, so the corrections are rule-based and the "
    "feedback is placeholder text."
)


def env(name: str, default: str | None = None) -> str | None:
    """Read DEMO_<NAME> from the environment."""
    return os.environ.get(f"DEMO_{name}", default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain-to-Learn demo.")
    parser.add_argument("--fake", action="store_true", default=env("FAKE") == "1")
    parser.add_argument(
        "--backend", choices=("transformers", "openai"), default=env("BACKEND", "transformers")
    )
    parser.add_argument(
        "--model", default=env("MODEL", BASE_MODEL), help="base model, or served model name"
    )
    parser.add_argument("--adapter", default=env("ADAPTER"), help="LoRA adapter path or Hub ID")
    parser.add_argument("--base-url", default=env("BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--corrector", choices=("gec", "model"), default=env("CORRECTOR", "gec"))
    parser.add_argument("--gec-model", default=env("GEC_MODEL", GEC_MODEL))
    parser.add_argument("--graph", default=env("GRAPH", "grammar_graph"))
    parser.add_argument("--port", type=int, default=int(env("PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    return parser.parse_args(argv)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Load the models named in the arguments; with --fake, use stubs."""
    if args.fake:
        return fake_pipeline()
    from explain_to_learn.graph import load_graph

    graph = load_graph(f"{args.graph}/concepts.yaml", f"{args.graph}/errant_map.yaml")
    if args.backend == "openai":
        backend = OpenAIBackend(args.model, args.base_url, os.environ.get("API_KEY", "none"))
    else:
        backend = TransformersBackend(args.model, args.adapter)
    corrector = GecCorrector(args.gec_model) if args.corrector == "gec" else ModelCorrector(backend)
    return Pipeline(corrector, ErrantExtractor(), graph, backend)


def build_ui(pipeline: Pipeline, fake: bool = False):
    """The Gradio interface around one pipeline."""
    import gradio as gr

    @gpu
    def check(sentence: str, level: str, mode: str) -> tuple[str, str]:
        result = pipeline.run(sentence, level, mode)
        return result.corrected, format_feedback(result)

    with gr.Blocks(title="Explain-to-Learn") as demo:
        gr.Markdown("# Explain-to-Learn\n\n" + DESCRIPTION)
        if fake:
            gr.Markdown(FAKE_NOTICE)
        sentence = gr.Textbox(label="Your sentence", lines=2, max_length=300)
        with gr.Row():
            level = gr.Radio(list(LEVELS), value="B1", label="Your level (CEFR)")
            mode = gr.Radio(MODES, value="explain", label="Feedback")
        button = gr.Button("Check", variant="primary")
        corrected = gr.Textbox(label="Corrected sentence", interactive=False)
        feedback = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[sentence, level, mode])
        button.click(check, [sentence, level, mode], [corrected, feedback])
        sentence.submit(check, [sentence, level, mode], [corrected, feedback])
    return demo


def main() -> None:
    args = parse_args()
    build_ui(build_pipeline(args), args.fake).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
