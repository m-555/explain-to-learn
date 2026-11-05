"""Command-line tool: feedback on a learner's text, and the teacher summary of a text.

    explain-to-learn explain "She don't like to going." --level B1 --mode hint
    explain-to-learn explain "She don't like to going." --level B1 \
        --corrected "She doesn't like to go."
    explain-to-learn summary essay.txt --level B1
    explain-to-learn summary essay.txt --level B1 --corrected essay_corrected.txt
    explain-to-learn summary --essay wi-1-352420 --edits data/processed/edits.jsonl

Models and endpoints come from `configs/systems.yaml`: `--system` picks the
explanation model (system D by default) and `--corrector` the correction step (`gec`
or `llm`). `--backend openai` sends the explanation requests to the system's server
instead of loading the model. With `--corrected`, no correction model is loaded, and
`summary` then needs no model at all. Both commands need ERRANT and spaCy's English
model (the `data` extra) to split the text into edits; `summary --essay` reads the
edits from the edit table instead.

Output is Markdown, or JSON with `--json`. It quotes the learner's text, so save it
under `outputs/` (git-ignored) rather than in the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from explain_to_learn.edits import read_jsonl
from explain_to_learn.generate import load_systems
from explain_to_learn.pipeline import (
    LEVELS,
    Annotator,
    Corrector,
    Explainer,
    default_annotator,
    explain_text,
    feedback_markdown,
    load_correctors,
    make_corrector,
    text_edits,
)
from explain_to_learn.prompts import MODES
from explain_to_learn.summary import essay_edits, markdown, summarize

DEFAULT_CONFIG = Path("configs/systems.yaml")
DEFAULT_GRAPH = Path("grammar_graph")


def open_graph(directory: Path) -> Any:
    """Load the grammar graph from its folder (needs the `graph` extra)."""
    from explain_to_learn.graph import load_graph

    return load_graph(directory / "concepts.yaml", directory / "errant_map.yaml")


def load_explainer(config: Path, name: str, backend: str | None = None) -> Explainer:
    """The explanation model of system `name`, optionally on another backend."""
    systems = load_systems(config)
    if name not in systems:
        raise SystemExit(f"error: no system {name!r} in {config}; choose from {sorted(systems)}")
    system = dict(systems[name])
    if backend:
        system["backend"] = backend
    return Explainer.from_system(name, system)


def load_corrector(config: Path, name: str) -> Corrector:
    """The corrector `name` from the `correctors` section of the config."""
    correctors = load_correctors(config)
    if name not in correctors:
        raise SystemExit(
            f"error: no corrector {name!r} in {config}; choose from {sorted(correctors)}"
        )
    return make_corrector(correctors[name])


def load_annotator() -> Annotator:
    """ERRANT with spaCy's English model."""
    return default_annotator()


def read_text(value: str) -> str:
    """The argument itself, or standard input for `-`."""
    return sys.stdin.read() if value == "-" else value


def explain(args: argparse.Namespace) -> str:
    """Markdown (or JSON) feedback on each edit of the text."""
    graph = open_graph(args.graph)
    corrector = None if args.corrected is not None else load_corrector(args.config, args.corrector)
    explainer = load_explainer(args.config, args.system, args.backend)
    result = explain_text(
        read_text(args.text),
        args.level,
        args.mode,
        corrector,
        explainer,
        graph,
        annotator=load_annotator(),
        corrected=args.corrected,
    )
    errors = [item.generation.error for item in result.explained if item.generation.error]
    if errors:
        print(f"warning: {len(errors)} edits got no valid answer: {errors[0]}", file=sys.stderr)
    if args.json:
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return f"**Corrected:** {result.corrected_text}\n\n{feedback_markdown(result)}"


def summary(args: argparse.Namespace) -> str:
    """Markdown (or JSON) teacher summary of a text file or of an essay's gold edits."""
    graph = open_graph(args.graph)
    level = args.level
    if args.essay:
        edits = essay_edits(read_jsonl(args.edits), args.essay)
        if not edits:
            raise SystemExit(f"error: no edits for essay {args.essay} in {args.edits}")
        level = level or edits[0].cefr_level
    else:
        corrected = args.corrected.read_text(encoding="utf-8") if args.corrected else None
        corrector = None if corrected is not None else load_corrector(args.config, args.corrector)
        text = args.file.read_text(encoding="utf-8")
        _, edits = text_edits(text, level, corrector, load_annotator(), corrected)
    result = summarize(edits, graph, level)
    if args.json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    return markdown(result)


def parser() -> argparse.ArgumentParser:
    """The command-line arguments of both commands."""
    main_parser = argparse.ArgumentParser(
        prog="explain-to-learn", description=__doc__.splitlines()[0]
    )
    commands = main_parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="grammar graph folder")
    common.add_argument(
        "--corrector", default="gec", help="corrector in the config: gec or llm (default gec)"
    )
    common.add_argument("--json", action="store_true", help="print JSON instead of Markdown")

    run = commands.add_parser(
        "explain", parents=[common], help="explain each correction in a learner's text"
    )
    run.add_argument("text", help="the learner's text, or - to read standard input")
    run.add_argument("--level", required=True, choices=LEVELS, type=str.upper)
    run.add_argument("--mode", choices=MODES, default="explain")
    run.add_argument("--corrected", help="the corrected text, instead of a correction model")
    run.add_argument("--system", default="D", help="explanation system in the config")
    run.add_argument("--backend", choices=("transformers", "openai"), help="override")
    run.set_defaults(func=explain)

    report = commands.add_parser(
        "summary", parents=[common], help="teacher summary of a text, by grammar concept"
    )
    report.add_argument("file", type=Path, nargs="?", help="text file with the learner's text")
    report.add_argument("--level", choices=LEVELS, type=str.upper)
    report.add_argument("--corrected", type=Path, help="file with the corrected text")
    report.add_argument("--essay", help="essay key in the edit table, e.g. wi-1-352420")
    report.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    report.set_defaults(func=summary)
    return main_parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `explain-to-learn` command."""
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "summary":
        if (args.file is None) == (args.essay is None):
            cli.error("summary needs either a text FILE or --essay")
        if args.file is not None and args.level is None:
            cli.error("summary of a text FILE needs --level")
    try:
        print(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
