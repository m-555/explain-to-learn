"""Start the annotation app on this computer.

Usage:
    python -m explain_to_learn.annotation                       # every task
    python -m explain_to_learn.annotation --task second-rater \
        --second-sheet rating_sheet_second_rater.csv            # the second rater's laptop

The app listens on 127.0.0.1 only, so the learner text it shows stays on this computer.
See docs/annotation_guide.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from explain_to_learn.io import name_and_path

TASKS = ("gold", "ratings", "second-rater", "silver", "preferences")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m explain_to_learn.annotation",
        description="Local annotation app: gold explanations, ratings, hand-check, preferences.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="show only this task; repeat for more (default: all tasks)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/annotations"))

    gold = parser.add_argument_group("gold explanations")
    gold.add_argument("--gold-ids", type=Path, default=Path("data/splits/gold_edit_ids.txt"))
    gold.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    gold.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    gold.add_argument(
        "--drafts",
        type=Path,
        action="append",
        default=[],
        help="a run file of scripts/run_system.py to start from; repeat for an explain and "
        "a hint run",
    )

    ratings = parser.add_argument_group("blind ratings")
    ratings.add_argument("--sheet", type=Path, default=Path("outputs/human_eval/rating_sheet.csv"))
    ratings.add_argument(
        "--second-sheet",
        type=Path,
        default=Path("outputs/human_eval/rating_sheet_second_rater.csv"),
    )
    ratings.add_argument("--rater", default="rater1", help="name of the first rater")
    ratings.add_argument("--second-rater", default="", help="name of the second rater")
    ratings.add_argument("--rubric", type=Path, default=Path("eval/rubric.md"))

    silver = parser.add_argument_group("silver hand-check")
    silver.add_argument("--handcheck", type=Path, default=Path("outputs/silver_handcheck.csv"))

    preferences = parser.add_argument_group("preference pairs")
    preferences.add_argument(
        "--pair",
        type=name_and_path,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a run file to compare; give exactly two",
    )
    preferences.add_argument("--seed", type=int, default=0, help="seed of the left/right order")

    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    args.task = args.task or list(TASKS)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from explain_to_learn.annotation.app import build_app  # needs the `demo` extra (Gradio)

    build_app(args).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
