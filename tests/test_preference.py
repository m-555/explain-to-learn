"""Conversion of the teachers' preference judgements into TRL DPO pairs."""

import json

import pytest

from explain_to_learn.preference import (
    assistant_message,
    build_preferences,
    essay_key,
    main,
    skip_reason,
)
from explain_to_learn.prompts import build_target

PROMPT = [
    {"role": "system", "content": "You are an English teacher."},
    {"role": "user", "content": "Learner sentence: She like tea.\nMode: explain"},
]
GROUNDED = [
    PROMPT[0],
    {"role": "user", "content": PROMPT[1]["content"] + "\nGrammar rule (from the grammar graph):"},
]
GOOD = build_target("Subject-verb agreement", 'Use "likes" after "she".', "")
VAGUE = build_target("Verbs", "Check the verb.", "")


def judgement(edit_id, chosen=GOOD, rejected=VAGUE, prompt=GROUNDED, **extra):
    """One line of preferences.jsonl as the annotation app writes it."""
    return {
        "edit_id": edit_id,
        "level": "B1",
        "mode": "explain",
        "prompt_messages": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source_systems": ["D", "B"],
        **extra,
    }


def test_essay_key_drops_the_sentence_and_edit_numbers():
    assert essay_key("wi-1-352420-3-0") == "wi-1-352420"
    assert essay_key("fce-TR1*0100*2000*01-1-3-0") == "fce-TR1*0100*2000*01-1"


def test_outputs_become_assistant_messages():
    parsed = {"concept": "Subject-verb agreement", "explanation": 'Use "likes".', "hint": ""}

    assert assistant_message(GOOD) == [{"role": "assistant", "content": GOOD}]
    assert assistant_message(parsed)[0]["content"] == build_target(
        "Subject-verb agreement", 'Use "likes".', ""
    )
    assert assistant_message([{"role": "assistant", "content": "x"}]) == [
        {"role": "assistant", "content": "x"}
    ]
    with pytest.raises(ValueError):
        assistant_message({"text": "no output fields"})


def test_ties_both_bad_and_identical_outputs_are_skipped():
    assert skip_reason(judgement("wi-1-0-0")) is None
    assert skip_reason(judgement("wi-1-0-0", None, None, judgement="tie")) == "tie"
    assert skip_reason(judgement("wi-1-0-0", None, None, judgement="both_bad")) == "both_bad"
    assert skip_reason(judgement("wi-1-0-0", GOOD, None)) == "missing output"
    assert skip_reason(judgement("wi-1-0-0", GOOD, GOOD)) == "same output"


def test_build_preferences_keeps_pairs_and_counts_the_rest():
    rows = [
        judgement("wi-1-0-0"),
        judgement("wi-1-0-1", None, None, judgement="tie"),
        judgement("wi-2-0-0", None, None, judgement="both_bad"),
        judgement("wi-2-1-0", prompt=PROMPT),
        judgement("wi-3-0-0", GOOD, GOOD),
    ]
    splits, counts = build_preferences(rows, share=0.0)

    assert counts == {
        "pairs": 2,
        "grounded prompts": 1,
        "skipped: tie": 1,
        "skipped: both_bad": 1,
        "skipped: same output": 1,
    }
    examples = splits["train"] + splits["validation"]
    assert sorted(e["edit_id"] for e in examples) == ["wi-1-0-0", "wi-2-1-0"]
    first = next(e for e in examples if e["edit_id"] == "wi-1-0-0")
    assert first["prompt"] == GROUNDED
    assert first["chosen"] == [{"role": "assistant", "content": GOOD}]
    assert first["rejected"] == [{"role": "assistant", "content": VAGUE}]
    assert first["essay_key"] == "wi-1" and first["source_systems"] == ["D", "B"]


def test_validation_split_keeps_each_essay_on_one_side():
    rows = [judgement(f"wi-{essay}-{sent}-0") for essay in range(20) for sent in range(3)]
    splits, _ = build_preferences(rows, share=0.1, seed=1)

    train = {e["essay_key"] for e in splits["train"]}
    validation = {e["essay_key"] for e in splits["validation"]}
    assert len(validation) == 2 and not train & validation
    assert len(splits["validation"]) == 6
    assert build_preferences(rows, share=0.1, seed=1)[0] == splits  # the same split again


def test_main_writes_train_and_validation_files(tmp_path, capsys):
    path = tmp_path / "preferences.jsonl"
    rows = [judgement(f"wi-{essay}-0-0") for essay in range(10)]
    rows.append(judgement("wi-99-0-0", None, None, judgement="tie"))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    main(["--preferences", str(path), "--out", str(tmp_path / "dpo")])

    lines = {
        split: (tmp_path / "dpo" / f"{split}.jsonl").read_text().splitlines()
        for split in ("train", "validation")
    }
    assert len(lines["train"]) + len(lines["validation"]) == 10
    assert set(json.loads(lines["train"][0])) >= {"prompt", "chosen", "rejected"}
    assert "skipped: tie: 1" in capsys.readouterr().out


def test_main_without_pairs_writes_nothing(tmp_path):
    path = tmp_path / "preferences.jsonl"
    path.write_text(json.dumps(judgement("wi-1-0-0", None, None, judgement="tie")) + "\n")

    with pytest.raises(SystemExit, match="No preference pairs"):
        main(["--preferences", str(path), "--out", str(tmp_path / "dpo")])
    assert not (tmp_path / "dpo").exists()
