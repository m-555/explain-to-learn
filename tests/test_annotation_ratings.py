"""Blind and second-rater ratings: the views stay blind, and the ratings file is read by
the real readers in `human_eval` and `scripts/agreement.py`.

Sheets are built with the real `scripts/make_rating_sheet.py` from invented sentences.
"""

import builtins
import importlib.util
import io
import json
from pathlib import Path

import pytest

from explain_to_learn.annotation.ratings import RatingTask, read_sheet
from explain_to_learn.edits import Edit
from explain_to_learn.human_eval import (
    KEY_COLUMNS,
    RATING_COLUMNS,
    SHEET_COLUMNS,
    join_key,
    read_csv,
    read_ratings,
    summarise,
    write_csv,
)

ROOT = Path(__file__).parents[1]
SYSTEMS = ("SYSTEM-ALPHA", "SYSTEM-BRAVO", "SYSTEM-CHARLIE", "SYSTEM-DELTA")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_rating_sheet = load_script("make_rating_sheet")
agreement = load_script("agreement")


def make_edit(idx: int) -> Edit:
    return Edit(
        edit_id=f"wi-E{idx}-0-0",
        corpus="wi",
        essay_id=f"E{idx}",
        split="dev",
        cefr_level=["A2", "B1", "C1"][idx % 3],
        sent_idx=0,
        original=f"He go to school {idx} .",
        corrected=f"He goes to school {idx} .",
        start=1,
        end=2,
        original_text="go",
        corrected_text="goes",
        error_type="R:VERB:SVA",
        operation="R",
        error_category="VERB:SVA",
    )


@pytest.fixture
def sheets(tmp_path):
    """The rating sheet, the second rater's sheet and the key, as the script writes them."""
    edits = {edit.edit_id: edit for edit in map(make_edit, range(6))}
    outputs = {
        system: {
            edit_id: {
                "edit_id": edit_id,
                "level": edit.cefr_level,
                "mode": "explain",
                "parsed": {
                    "concept": "Agreement",
                    "explanation": f'Use "goes" after "he" ({number}).',
                    "hint": "",
                },
            }
            for edit_id, edit in edits.items()
        }
        for number, system in enumerate(SYSTEMS)
    }
    sheet, second, key = make_rating_sheet.make_sheets(list(edits), edits, outputs, 5, 6)
    paths = {
        "sheet": tmp_path / "human_eval" / "rating_sheet.csv",
        "second": tmp_path / "human_eval" / "rating_sheet_second_rater.csv",
        "key": tmp_path / "human_eval" / "key.csv",
    }
    write_csv(sheet, paths["sheet"], (*SHEET_COLUMNS, *RATING_COLUMNS))
    write_csv(second, paths["second"], (*SHEET_COLUMNS, *RATING_COLUMNS))
    write_csv(key, paths["key"], KEY_COLUMNS)
    return paths


def rate_all(task: RatingTask, score: str = "yes") -> None:
    for index in range(task.start_index(), len(task)):
        task.save(index, score, score, "partly", score, "no", "")


@pytest.fixture
def key_is_never_opened(monkeypatch):
    """Fail the test if anything opens a file called key.csv."""

    def guard(opener):
        def checked_open(file, *args, **kwargs):
            if Path(str(file)).name == "key.csv":
                raise AssertionError("the key file was opened")
            return opener(file, *args, **kwargs)

        return checked_open

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))


def test_ratings_round_trip_through_the_human_eval_readers(sheets, tmp_path):
    task = RatingTask(sheets["sheet"], tmp_path / "annotations", "rater1")
    rate_all(task)

    path = tmp_path / "annotations" / "ratings_rater1.csv"
    assert read_csv(path)[0].keys() == {"item_id", *RATING_COLUMNS}
    ratings = read_ratings(path)
    assert len(ratings) == 20
    expected = {"rule_correct": 2, "error_located": 2, "helps_fix": 1, "fits_level": 2}
    assert all(scores == expected | {"invented_rule": 0} for scores in ratings.values())
    rows = join_key(ratings, read_csv(sheets["key"]))
    by_system = summarise(rows, ("system",))
    assert by_system == [[system, 5, 2, 2, 1, 2, 0] for system in SYSTEMS]


def test_second_rater_file_is_separate_and_agreement_reads_both(sheets, tmp_path):
    out_dir = tmp_path / "annotations"
    rate_all(RatingTask(sheets["sheet"], out_dir, "rater1"))
    second = RatingTask(sheets["second"], out_dir, "Anna B.")
    rate_all(second, score="partly")

    second_path = out_dir / "ratings_anna_b.csv"
    assert second.store.path == second_path
    first_ratings = read_ratings(out_dir / "ratings_rater1.csv")
    second_ratings = read_ratings(second_path)
    assert len(second_ratings) == 6 and set(second_ratings) <= set(first_ratings)
    rows = {row[0]: row for row in agreement.agreement_rows(first_ratings, second_ratings)}
    assert rows["rule_correct"][:3] == ["rule_correct", 6, 0.0]
    assert rows["helps_fix"][:3] == ["helps_fix", 6, 1.0]
    info = json.loads((out_dir / "ratings_anna_b.json").read_text())
    assert (info["rater"], info["items"], info["saved"]) == ("Anna B.", 6, 6)


def test_views_and_saved_files_never_show_system_names(sheets, tmp_path, key_is_never_opened):
    out_dir = tmp_path / "annotations"
    for sheet, rater in ((sheets["sheet"], "rater1"), (sheets["second"], "rater2")):
        task = RatingTask(sheet, out_dir, rater)
        shown = []
        for index in range(len(task)):
            shown += [task.status(index), *task.view(index), *map(str, task.values(index))]
            task.save(index, "yes", "yes", "yes", "yes", "no", "fine")
            shown += [task.status(index), *map(str, task.values(index))]
        shown += [path.read_text() for path in out_dir.iterdir()]
        text = "\n".join(shown)
        assert "He go" in text.replace("\\", "")  # the items are shown ...
        assert not any(system in text for system in SYSTEMS)  # ... but not their systems


def test_second_rater_needs_only_the_sheet(sheets, tmp_path, monkeypatch):
    laptop = tmp_path / "laptop"
    laptop.mkdir()
    (laptop / "sheet.csv").write_bytes(sheets["second"].read_bytes())
    monkeypatch.chdir(laptop)

    task = RatingTask("sheet.csv", "outputs/annotations", "rater2")
    rate_all(task)

    assert sorted(p.name for p in (laptop / "outputs" / "annotations").iterdir()) == [
        "ratings_rater2.csv",
        "ratings_rater2.json",
    ]


def test_the_key_is_refused_as_a_sheet(sheets, tmp_path):
    with pytest.raises(ValueError, match="never the key"):
        RatingTask(sheets["key"], tmp_path, "rater1")


def test_resume_goes_back_and_keeps_saved_values(sheets, tmp_path):
    task = RatingTask(sheets["sheet"], tmp_path, "rater1")
    task.save(0, "no", "partly", "yes", "yes", "yes", "made-up rule")
    task.save(1, "yes", "yes", "yes", "yes", "no", "")

    resumed = RatingTask(sheets["sheet"], tmp_path, "rater1")
    assert resumed.start_index() == 2
    assert resumed.values(0) == ["no", "partly", "yes", "yes", "yes", "made-up rule"]
    assert resumed.values(2) == [None, None, None, None, None, ""]
    resumed.save(0, "partly", "partly", "yes", "yes", "yes", "made-up rule")
    assert read_ratings(tmp_path / "ratings_rater1.csv")["item-01"]["rule_correct"] == 1


def test_open_ratings_need_a_comment_and_numbers_are_read(sheets, tmp_path):
    task = RatingTask(sheets["sheet"], tmp_path, "rater1")

    with pytest.raises(ValueError, match="helps the learner fix it, invented rule"):
        task.save(0, "yes", "yes", None, "yes", None, "")
    task.save(0, "yes", "yes", None, "yes", "no", "Cannot tell whether this helps.")
    assert read_ratings(tmp_path / "ratings_rater1.csv")["item-01"]["helps_fix"] is None

    (tmp_path / "ratings_rater1.csv").write_text(
        "item_id,rule_correct,error_located,helps_fix,fits_level,invented_rule,comment\n"
        "item-01,2,1,0,2,1,\n"
    )
    assert RatingTask(sheets["sheet"], tmp_path, "rater1").values(0)[:5] == [
        "yes",
        "partly",
        "no",
        "yes",
        "yes",
    ]


def test_ratings_of_a_different_sheet_are_refused(sheets, tmp_path):
    rate_all(RatingTask(sheets["sheet"], tmp_path, "rater1"))
    rows = read_csv(sheets["sheet"])
    rows[0]["feedback"] = "A different text."
    write_csv(rows, sheets["sheet"], (*SHEET_COLUMNS, *RATING_COLUMNS))

    with pytest.raises(ValueError, match="different rating sheet"):
        RatingTask(sheets["sheet"], tmp_path, "rater1")
    assert len(read_sheet(sheets["sheet"])) == 20
    with pytest.raises(ValueError, match="different rating sheet"):
        RatingTask(sheets["second"], tmp_path, "rater1")


def test_rater_name_is_required(sheets, tmp_path):
    with pytest.raises(ValueError, match="rater name"):
        RatingTask(sheets["second"], tmp_path, "  ")
    assert not list(tmp_path.glob("ratings_*"))
