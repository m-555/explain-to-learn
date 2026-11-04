import json
from pathlib import Path

import pytest

from explain_to_learn.generate import (
    Decoding,
    OpenAIBackend,
    done_ids,
    generate,
    load_systems,
    make_backend,
    parse_output,
    run_batch,
)

GOOD = {"concept": "Articles", "explanation": 'Add "a" before "cat".', "hint": ""}
GOOD_TEXT = json.dumps(GOOD)
MESSAGES = [{"role": "user", "content": "Explain."}]


class ScriptedBackend:
    """Returns the scripted replies in order; an Exception in the list is raised."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, decoding):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize(
    "text",
    [
        GOOD_TEXT,
        f"```json\n{GOOD_TEXT}\n```",
        f"Sure! Here is the JSON:\n{GOOD_TEXT}\nHope this helps.",
        f"<think>The learner wrote {{cat}}. {{\"concept\": 1}}</think>\n{GOOD_TEXT}",
        f"I should check the rule {{not json}}.</think>{GOOD_TEXT}",
        f"<|channel|>analysis<|message|>Think.<|end|><|start|>assistant"
        f"<|channel|>final<|message|>{GOOD_TEXT}<|return|>",
        '{"concept": "Articles", "explanation": "Add \\"a\\" before \\"cat\\".", '
        '"hint": "", "level": "B1"}',
        '{"concept": " Articles ", "explanation": "Add \\"a\\" before \\"cat\\".", "hint": null}',
    ],
)
def test_parse_output_handles_messy_replies(text):
    assert parse_output(text) == GOOD


def test_parse_output_skips_invalid_objects_before_a_valid_one():
    text = '{"concept": "Articles"} and then ' + GOOD_TEXT
    assert parse_output(text) == GOOD


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "The answer is to add an article.",
        '{"concept": "Articles", "explanation": "Add a."',  # truncated
        '{"concept": "Articles", "explanation": "Add a."}',  # missing hint
        '{"concept": "", "explanation": "Add a.", "hint": ""}',  # empty concept
        '{"concept": "Articles", "explanation": "", "hint": ""}',  # nothing to show
        '{"concept": ["Articles"], "explanation": "Add a.", "hint": ""}',
        "[1, 2, 3]",
    ],
)
def test_parse_output_rejects_invalid_replies(text):
    assert parse_output(text) is None


def test_generate_returns_parsed_output_on_first_try():
    result = generate(ScriptedBackend([GOOD_TEXT]), MESSAGES)

    assert result.parsed == GOOD
    assert result.attempts == 1
    assert result.error is None


def test_generate_retries_once_with_a_follow_up_message():
    backend = ScriptedBackend(["not json", GOOD_TEXT])
    result = generate(backend, MESSAGES)

    assert result.parsed == GOOD
    assert result.attempts == 2
    assert backend.calls[1][:1] == MESSAGES
    assert backend.calls[1][1] == {"role": "assistant", "content": "not json"}


def test_generate_records_invalid_json_after_retry():
    result = generate(ScriptedBackend(["no", "still no"]), MESSAGES)

    assert result.parsed is None
    assert result.raw == "still no"
    assert result.attempts == 2
    assert result.error == "invalid_json"


def test_generate_records_backend_errors_without_raising():
    result = generate(ScriptedBackend([TimeoutError("slow"), TimeoutError("slow")]), MESSAGES)

    assert result.parsed is None
    assert result.error == "TimeoutError: slow"


def test_generate_recovers_after_a_backend_error():
    result = generate(ScriptedBackend([ConnectionError("reset"), GOOD_TEXT]), MESSAGES)

    assert result.parsed == GOOD
    assert result.error is None


def test_run_batch_writes_records_and_resumes(tmp_path):
    out = tmp_path / "out" / "run.jsonl"
    items = [({"edit_id": "e1", "mode": "explain"}, MESSAGES), ({"edit_id": "e2"}, MESSAGES)]

    written = run_batch(ScriptedBackend([GOOD_TEXT, "bad", "bad"]), items, out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == 2
    assert rows[0]["edit_id"] == "e1" and rows[0]["mode"] == "explain"
    assert rows[0]["parsed"] == GOOD
    assert rows[1]["parsed"] is None and rows[1]["error"] == "invalid_json"
    assert set(rows[0]) >= {"raw", "parsed", "attempts", "latency_s", "error"}
    assert done_ids(out) == {"e1", "e2"}
    assert done_ids(tmp_path / "missing.jsonl") == set()


def test_openai_request_body_uses_decoding_and_extra_body():
    backend = OpenAIBackend("m", extra_body={"reasoning_effort": "low"})
    body = backend.request_body(MESSAGES, Decoding(max_new_tokens=99))

    assert body["model"] == "m"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 99
    assert body["reasoning_effort"] == "low"


def test_make_backend_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_backend({"backend": "magic", "model": "m"})


def test_systems_config_defines_all_systems():
    pytest.importorskip("yaml")
    systems = load_systems(Path(__file__).parents[1] / "configs" / "systems.yaml")

    assert set(systems) == {"A", "B", "C", "D", "R"}
    assert [systems[s]["grounded"] for s in "ABCDR"] == [False, True, False, True, True]
    assert systems["A"]["model"] == systems["C"]["model"]
    assert not systems["A"].get("adapter") and not systems["B"].get("adapter")
    assert systems["C"]["adapter"] and systems["C"]["adapter"] == systems["D"]["adapter"]
    for system in systems.values():
        assert system["backend"] in {"openai", "transformers"}
        assert system["decoding"]["temperature"] == 0.0
        Decoding(**system["decoding"])
