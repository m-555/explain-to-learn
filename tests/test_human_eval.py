"""Rating sheet building, the key, and the per-system summaries, on synthetic items."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from explain_to_learn.edits import Edit
from explain_to_learn.human_eval import (
    DIMENSIONS,
    KEY_COLUMNS,
    SHEET_COLUMNS,
    join_key,
    markdown_table,
    parse_rating,
    read_csv,
    read_ratings,
    summarise,
    write_csv,
    write_summaries,
)
from explain_to_learn.prompts import detokenize

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_rating_sheet.py"
SYSTEMS = ("A", "B", "C", "D")


def load_script():
    spec = importlib.util.spec_from_file_location("make_rating_sheet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_rating_sheet = load_script()


def make_edit(idx: int) -> Edit:
    category = ["VERB:SVA", "DET", "PREP"][idx % 3]
    return Edit(
        edit_id=f"wi-E{idx}-0-0",
        corpus="wi",
        essay_id=f"E{idx}",
        split="dev",
        cefr_level=["A2", "B1", "C1"][idx % 3],
        sent_idx=0,
        original=f"He go home {idx} .",
        corrected=f"He goes home {idx} .",
        start=1,
        end=2,
        original_text="go",
        corrected_text="goes",
        error_type=f"R:{category}",
        operation="R",
        error_category=category,
    )


def generation(system: str, edit: Edit, mode: str = "explain") -> dict:
    """A generation record as written by `scripts/run_system.py`."""
    parsed = {"concept": "Agreement", "explanation": f"{system} explains", "hint": f"{system} hint"}
    return {"edit_id": edit.edit_id, "level": edit.cefr_level, "mode": mode, "parsed": parsed}


EDITS = {edit.edit_id: edit for edit in map(make_edit, range(12))}
# Each system's feedback names the system, so the tests can check the key joins back.
FEEDBACK = {
    system: {edit_id: generation(system, edit) for edit_id, edit in EDITS.items()}
    for system in SYSTEMS
}


def sheets(seed: int = 7):
    return make_rating_sheet.make_sheets(list(EDITS), EDITS, FEEDBACK, 8, 10, seed)


def test_sheet_has_every_edit_for_every_system_and_hides_the_system():
    sheet, _, key = sheets()

    assert len(sheet) == len(key) == 8 * len(SYSTEMS)
    assert tuple(sheet[0]) == SHEET_COLUMNS
    assert "system" not in sheet[0]
    assert [row["item_id"] for row in sheet] == [f"item-{i:02d}" for i in range(1, 33)]
    systems_in_order = [row["system"] for row in key]
    assert systems_in_order != sorted(systems_in_order)
    assert {row["edit_id"] for row in key} <= set(EDITS)
    assert all(sum(r["edit_id"] == row["edit_id"] for r in key) == 4 for row in key)


def test_sheet_is_reproducible_and_depends_on_the_seed():
    assert sheets(seed=7) == sheets(seed=7)
    assert sheets(seed=7)[0] != sheets(seed=8)[0]


def test_key_joins_back_to_each_systems_feedback():
    sheet, _, key = sheets()
    key_by_item = {row["item_id"]: row for row in key}

    for row in sheet:
        entry = key_by_item[row["item_id"]]
        assert row["feedback"] == f"{entry['system']} explains"
        assert row["level"] == entry["level"]
        assert detokenize(EDITS[entry["edit_id"]].original) == row["sentence"]


def test_second_rater_sheet_is_a_subset_in_sheet_order():
    sheet, second, _ = sheets()

    assert len(second) == 10
    assert [row for row in sheet if row in second] == second


def test_sheet_shows_readable_text_and_the_feedback_for_the_mode():
    edit = dataclasses.replace(
        make_edit(0), original="She do n't like it .", original_text="do", corrected_text="does"
    )
    outputs = {
        "A": {edit.edit_id: generation("A", edit, mode="hint")},
        "B": {edit.edit_id: generation("B", edit) | {"parsed": None}},
    }
    sheet, _, key = make_rating_sheet.make_sheets(
        [edit.edit_id], {edit.edit_id: edit}, outputs, 1, 1
    )
    by_system = {k["system"]: row for k, row in zip(key, sheet, strict=True)}

    assert by_system["A"]["sentence"] == "She don't like it."
    assert by_system["A"]["edit"] == "do -> does"
    assert (by_system["A"]["mode"], by_system["A"]["feedback"]) == ("hint", "A hint")
    assert by_system["B"]["feedback"] == ""
    assert by_system["A"]["level"] == "A2"


def test_missing_feedback_is_an_error():
    feedback = {**FEEDBACK, "D": {}}

    with pytest.raises(ValueError, match="System D"):
        make_rating_sheet.make_sheets(list(EDITS), EDITS, feedback, 8, 10, 7)


def test_parse_rating_accepts_words_and_numbers():
    scale = {"no": 0, "partly": 1, "yes": 2}

    assert [parse_rating(v, scale) for v in ["no", " Partly", "YES", "0", "2", ""]] == [
        0,
        1,
        2,
        0,
        2,
        None,
    ]
    with pytest.raises(ValueError):
        parse_rating("maybe", scale)
    with pytest.raises(ValueError):
        parse_rating("3", scale)


def filled_ratings(tmp_path, key):
    """A filled sheet: system A gets 'yes' everywhere, the others 'partly'."""
    rows = []
    for entry in key:
        score = "yes" if entry["system"] == "A" else "partly"
        rows.append(
            {"item_id": entry["item_id"]}
            | {dim: score for dim in DIMENSIONS}
            | {"invented_rule": "yes" if entry["system"] == "D" else "no"}
        )
    path = tmp_path / "ratings.csv"
    write_csv(rows, path, ["item_id", *DIMENSIONS, "invented_rule", "comment"])
    return path


def test_summary_per_system_category_and_level(tmp_path):
    _, _, key = sheets()
    write_csv(key, tmp_path / "key.csv", KEY_COLUMNS)
    rows = join_key(read_ratings(filled_ratings(tmp_path, key)), read_csv(tmp_path / "key.csv"))

    by_system = summarise(rows, ("system",))
    assert by_system[0] == ["A", 8, 2, 2, 2, 2, 0]
    assert by_system[3] == ["D", 8, 1, 1, 1, 1, 1]
    by_category = summarise(rows, ("error_category", "system"))
    assert sum(row[2] for row in by_category) == 32

    markdown_path = write_summaries(rows, tmp_path / "results", "human_eval")
    text = markdown_path.read_text()
    assert "## By system" in text and "## By category" in text and "## By level" in text
    assert "| D | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 100% |" in text
    assert "He go" not in text
    assert (tmp_path / "results" / "human_eval_by_level.csv").exists()


def test_empty_ratings_are_left_out_of_the_means():
    rows = [
        {"system": "A", "rule_correct": 2, "error_located": None, "helps_fix": 1, "fits_level": 0}
        | {"invented_rule": None},
        {"system": "A", "rule_correct": 0, "error_located": None, "helps_fix": 1, "fits_level": 2}
        | {"invented_rule": 1},
    ]

    table = summarise(rows, ("system",))
    assert table == [["A", 2, 1, None, 1, 1, 1]]
    assert "| A | 2 | 1.00 | – | 1.00 | 1.00 | 100% |" in markdown_table(("system",), table)


def test_join_key_rejects_unknown_items():
    with pytest.raises(ValueError, match="not in the key"):
        join_key({"item-99": {}}, [{"item_id": "item-01", "system": "A"}])
