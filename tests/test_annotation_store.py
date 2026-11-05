"""The annotation store (save, load, resume) and the Markdown helpers."""

import csv
import json
from operator import itemgetter

import pytest

from explain_to_learn.annotation.display import escape, marked_sentence
from explain_to_learn.annotation.store import Store
from explain_to_learn.edits import Edit


def jsonl_store(path, order=("a", "b", "c")):
    return Store(path, itemgetter("id"), order)


def test_save_writes_records_in_item_order_and_a_new_store_resumes(tmp_path):
    path = tmp_path / "out" / "notes.jsonl"
    store = jsonl_store(path)
    assert (store.done_count(), store.first_open()) == (0, 0)

    store.save({"id": "b", "text": "second"})
    store.save({"id": "a", "text": "first"})
    store.save({"id": "a", "text": "first, changed"})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"id": "a", "text": "first, changed"}, {"id": "b", "text": "second"}]
    resumed = jsonl_store(path)
    assert resumed.get("b") == {"id": "b", "text": "second"}
    assert resumed.get("c") is None
    assert (resumed.done_count(), resumed.first_open()) == (2, 2)
    assert list(path.parent.iterdir()) == [path]  # no temporary file left behind


def test_progress_and_resume_when_every_item_is_saved(tmp_path):
    store = jsonl_store(tmp_path / "notes.jsonl")
    assert store.progress(0) == "**Item 1 of 3** · 0 of 3 saved"

    for item_id in "abc":
        store.save({"id": item_id})

    assert store.first_open() == 2
    assert store.progress(1).startswith("**Item 2 of 3** (saved) · 3 of 3 saved · all items")


def test_two_stores_on_one_file_keep_each_others_records(tmp_path):
    path = tmp_path / "notes.jsonl"
    first, second = jsonl_store(path), jsonl_store(path)

    first.save({"id": "a"})
    second.save({"id": "b"})

    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_records_for_unknown_items_are_kept_at_the_end(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text(json.dumps({"id": "old"}) + "\n")

    store = jsonl_store(path, order=("a",))
    store.save({"id": "a"})

    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["a", "old"]


def test_csv_store_writes_its_columns_only(tmp_path):
    path = tmp_path / "marks.csv"
    store = Store(path, itemgetter("id"), ["x", "y"], columns=("id", "mark"))

    store.save({"id": "y", "mark": "1", "scratch": "not written"})

    with path.open(newline="") as file:
        assert list(csv.DictReader(file)) == [{"id": "y", "mark": "1"}]
    assert Store(path, itemgetter("id"), ["x", "y"], ("id", "mark")).first_open() == 0
    with pytest.raises(ValueError, match="columns"):
        Store(path, itemgetter("id"), ["x"])


def make_edit(original, start, end, original_text, corrected_text, operation):
    return Edit(
        edit_id="wi-1-0-0",
        corpus="wi",
        essay_id="1",
        split="dev",
        cefr_level="B1",
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=original_text,
        corrected_text=corrected_text,
        error_type=f"{operation}:OTHER",
        operation=operation,
        error_category="OTHER",
    )


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            make_edit("She do n't like it .", 1, 2, "do", "does", "R"),
            "She [**do → does**]n't like it.",
        ),
        (make_edit("I have cat .", 2, 2, "", "a", "M"), "I have [**+ a**] cat."),
        (
            make_edit("It is very very good .", 3, 4, "very", "", "U"),
            "It is very [**− very**] good.",
        ),
        (make_edit("A *star* is me .", 2, 3, "is", "am", "R"), r"A \*star\* [**is → am**] me."),
    ],
)
def test_marked_sentence_shows_the_edit_in_readable_text(edit, expected):
    assert marked_sentence(edit) == expected


def test_escape_keeps_learner_text_literal():
    assert escape("a_b *c* [d] <e> #f") == r"a\_b \*c\* \[d\] \<e\> \#f"
