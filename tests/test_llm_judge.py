"""LLM judge prompts and strict answer checking, with a fake endpoint."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "llm_judge.py"


def load_script():
    spec = importlib.util.spec_from_file_location("llm_judge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm_judge = load_script()
FLAG = "invented_rule"

ITEM = {
    "item_id": "item-001",
    "level": "A2",
    "sentence": "She don't like coffee .",
    "corrected": "She doesn't like coffee .",
    "mode": "explain",
    "edit": "don't -> doesn't",
    "feedback": "With she, use doesn't.",
}
VALID = {
    "rule_correct": "yes",
    "error_located": "Yes",
    "helps_fix": "partly",
    "fits_level": "no",
    "invented_rule": "no",
}


def test_messages_hold_the_rubric_and_the_item_but_not_its_id():
    messages = llm_judge.judge_messages("RUBRIC TEXT", ITEM)

    assert "RUBRIC TEXT" in messages[0]["content"]
    assert '"invented_rule"' in messages[0]["content"]
    assert "sentence: She don't like coffee ." in messages[1]["content"]
    assert "item-001" not in messages[1]["content"]


def test_parse_judgement_accepts_only_allowed_values():
    assert llm_judge.parse_judgement(json.dumps(VALID))["error_located"] == "yes"
    with pytest.raises(ValueError):
        llm_judge.parse_judgement("rule_correct: yes")
    with pytest.raises(ValueError):
        llm_judge.parse_judgement(json.dumps(["yes"]))
    with pytest.raises(ValueError, match="fits_level"):
        llm_judge.parse_judgement(json.dumps(VALID | {"fits_level": "maybe"}))
    with pytest.raises(ValueError, match="invented_rule"):
        llm_judge.parse_judgement(json.dumps({k: v for k, v in VALID.items() if k != FLAG}))


def test_judge_item_retries_after_a_bad_answer():
    answers = iter(["not json", json.dumps(VALID)])

    result = llm_judge.judge_item(lambda messages: next(answers), "rubric", ITEM, retries=2)

    assert result["rule_correct"] == "yes" and result["comment"] == ""


def test_judge_item_gives_up_with_empty_ratings():
    result = llm_judge.judge_item(lambda messages: "{}", "rubric", ITEM, retries=1)

    assert result["rule_correct"] == "" and result["invented_rule"] == ""
    assert result["comment"].startswith("judge failed")
