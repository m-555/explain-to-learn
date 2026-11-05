"""Command-line options of the annotation app, and building its interface."""

from pathlib import Path

import pytest

from explain_to_learn.annotation.__main__ import TASKS, parse_args


def test_defaults_point_at_the_project_files():
    args = parse_args([])

    assert args.task == list(TASKS)
    assert args.out_dir == Path("outputs/annotations")
    assert args.sheet == Path("outputs/human_eval/rating_sheet.csv")
    assert args.second_sheet == Path("outputs/human_eval/rating_sheet_second_rater.csv")
    assert args.handcheck == Path("outputs/silver_handcheck.csv")
    assert (args.rater, args.second_rater, args.drafts, args.pair) == ("rater1", "", [], [])


def test_tasks_drafts_and_pairs_can_be_chosen():
    args = parse_args(
        ["--task", "second-rater", "--drafts", "a.jsonl", "--drafts", "b.jsonl"]
        + ["--pair", "D-1=one.jsonl", "--pair", "D-2=two.jsonl"]
    )

    assert args.task == ["second-rater"]
    assert args.drafts == [Path("a.jsonl"), Path("b.jsonl")]
    assert args.pair == [("D-1", Path("one.jsonl")), ("D-2", Path("two.jsonl"))]
    with pytest.raises(SystemExit):
        parse_args(["--pair", "no-name"])


def test_the_app_builds_without_any_data_files(tmp_path, monkeypatch):
    gradio = pytest.importorskip("gradio")
    from explain_to_learn.annotation.app import build_app

    monkeypatch.chdir(tmp_path)  # no corpus, sheets or graph here
    tabs = [
        block.label
        for block in build_app(parse_args([])).blocks.values()
        if isinstance(block, gradio.Tab)
    ]
    assert tabs == [
        "Gold explanations",
        "Blind ratings",
        "Second rater",
        "Silver hand-check",
        "Preference pairs",
    ]
    only_second = build_app(parse_args(["--task", "second-rater"]))
    assert [b.label for b in only_second.blocks.values() if isinstance(b, gradio.Tab)] == [
        "Second rater"
    ]
