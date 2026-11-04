import json

import pytest

from explain_to_learn.edits import Edit
from explain_to_learn.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_target,
    describe_edit,
    detokenize,
    mark_edit,
    prompt_level,
)

SNIPPET = {
    "concept_id": "subject_verb_agreement",
    "name": "Subject-verb agreement",
    "rule": "With he, she or it, the verb in the present simple takes -s.",
    "example": {"wrong": "She like tea.", "right": "She likes tea."},
    "prerequisites": ["Present simple"],
}


def make_edit(original, start, end, corrected_text, error_type, level="B1"):
    """A synthetic edit; the sentences are invented for the tests."""
    tokens = original.split()
    operation, category = error_type.split(":", 1)
    return Edit(
        edit_id="wi-test-0-0",
        corpus="wi",
        essay_id="test",
        split="dev",
        cefr_level=level,
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=" ".join(tokens[start:end]),
        corrected_text=corrected_text,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


SVA = make_edit("She do n't like tea .", 1, 2, "does", "R:VERB:SVA")
MISSING = make_edit("I have cat .", 2, 2, "a", "M:DET")
UNNECESSARY = make_edit("We were very very happy .", 3, 4, "", "U:ADV")


def test_detokenize_joins_clitics_and_punctuation():
    assert detokenize("She do n't like tea .") == "She don't like tea."
    assert detokenize(["It", "'s", "( very )", "good", "!"]) == "It's (very) good!"


def test_describe_and_mark_each_operation():
    assert describe_edit(SVA) == 'replace "do" with "does" (tokens 1-2)'
    assert describe_edit(MISSING) == 'add "a" (tokens 2-2)'
    assert describe_edit(UNNECESSARY) == 'remove "very" (tokens 3-4)'
    assert mark_edit(SVA) == "She [do -> does]n't like tea."
    assert mark_edit(MISSING) == "I have [+ a] cat."
    assert mark_edit(UNNECESSARY) == "We were very [- very] happy."


def test_ungrounded_messages_have_no_rule():
    messages = build_messages(SVA, "B1", "explain", None)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Learner sentence: She don't like tea." in user
    assert "Error type: R:VERB:SVA" in user
    assert "Learner level: B1" in user
    assert "Mode: explain" in user
    assert "Grammar rule" not in user


def test_grounded_messages_differ_only_by_snippet():
    plain = build_messages(SVA, "A2", "hint", None)
    grounded = build_messages(SVA, "A2", "hint", SNIPPET)

    assert grounded[0] == plain[0]
    assert grounded[1]["content"].startswith(plain[1]["content"] + "\n")
    rule = grounded[1]["content"][len(plain[1]["content"]) :]
    assert "Concept: Subject-verb agreement" in rule
    assert 'wrong "She like tea.", right "She likes tea."' in rule
    assert "Builds on: Present simple" in rule
    assert "Mode: hint" in plain[1]["content"]


def test_snippet_without_example_or_prerequisites():
    snippet = {**SNIPPET, "example": {}, "prerequisites": []}
    user = build_messages(SVA, "B2", "explain", snippet)[1]["content"]

    assert "Example:" not in user
    assert "Builds on:" not in user


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_messages(SVA, "B1", "answer", None)


def test_build_target_is_json_with_three_fields():
    target = build_target("Subject-verb agreement", 'Use "does", not "do".', "")

    assert json.loads(target) == {
        "concept": "Subject-verb agreement",
        "explanation": 'Use "does", not "do".',
        "hint": "",
    }


def test_native_writers_are_prompted_as_c2():
    assert prompt_level("native") == "C2"
    assert prompt_level("A2") == "A2"
