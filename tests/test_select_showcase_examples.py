"""scripts/select_showcase_examples.py on invented ratings, key, sheet and edits."""

import importlib.util
from pathlib import Path

from explain_to_learn.edits import Edit, write_jsonl
from explain_to_learn.human_eval import KEY_COLUMNS, RATING_COLUMNS, SHEET_COLUMNS
from explain_to_learn.io import write_csv

ROOT = Path(__file__).parents[1]


def load_script():
    path = ROOT / "scripts" / "select_showcase_examples.py"
    spec = importlib.util.spec_from_file_location("select_showcase_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


showcase = load_script()


def make_edit(edit_id, corpus, sentence, start, end, original, corrected, category):
    operation = "R" if original and corrected else ("M" if corrected else "U")
    return Edit(
        edit_id=edit_id,
        corpus=corpus,
        essay_id=edit_id.split("-", 1)[1],
        split="dev",
        cefr_level="B1",
        sent_idx=0,
        original=sentence,
        corrected=sentence,
        start=start,
        end=end,
        original_text=original,
        corrected_text=corrected,
        error_type=f"{operation}:{category}",
        operation=operation,
        error_category=category,
    )


EDITS = [
    make_edit(
        "wi-1-1-0-0", "wi", "My sister do n't like dogs .", 2, 4, "do n't", "does n't", "VERB:SVA"
    ),
    make_edit("wi-1-2-0-0", "wi", "I saw a elephant at the zoo .", 2, 3, "a", "an", "DET"),
    make_edit("wi-1-3-0-0", "wi", "He go to school every day .", 1, 2, "go", "goes", "VERB:SVA"),
    make_edit("locness-1-4-0-0", "locness", "They was late .", 1, 2, "was", "were", "VERB:SVA"),
]

YES = {"rule_correct": "yes", "error_located": "yes", "helps_fix": "yes", "fits_level": "yes"}
# item: (system, edit, mode, ratings, feedback)
ITEMS = {
    "item-1": (
        "D",
        "wi-1-1-0-0",
        "explain",
        YES | {"invented_rule": "no"},
        "Use 'doesn't' after 'she'.",
    ),
    "item-2": (
        "A",
        "wi-1-1-0-0",
        "explain",
        YES | {"helps_fix": "partly", "invented_rule": "no"},
        "Check the verb.",
    ),
    "item-3": (
        "D",
        "wi-1-2-0-0",
        "explain",
        YES | {"rule_correct": "no", "invented_rule": "yes"},
        "Use 'an' before nouns.",
    ),
    "item-4": (
        "B",
        "wi-1-3-0-0",
        "hint",
        YES | {"rule_correct": "no", "invented_rule": "no"},
        "Look at 'school'.",
    ),
    "item-5": (
        "D",
        "locness-1-4-0-0",
        "explain",
        YES | {"invented_rule": "no"},
        "Use 'were' with 'they'.",
    ),
    "item-6": ("C", "wi-1-3-0-0", "explain", {"rule_correct": "yes"}, "Incomplete rating."),
}


def write_inputs(root: Path) -> list[str]:
    write_jsonl(EDITS, root / "edits.jsonl")
    edits = {e.edit_id: e for e in EDITS}
    key, sheet, ratings = [], [], []
    for item_id, (system, edit_id, mode, scores, feedback) in ITEMS.items():
        edit = edits[edit_id]
        key.append(
            {
                "item_id": item_id,
                "system": system,
                "edit_id": edit_id,
                "error_category": edit.error_category,
                "level": "B1",
                "mode": mode,
            }
        )
        sheet.append(
            {
                "item_id": item_id,
                "level": "B1",
                "mode": mode,
                "sentence": edit.original,
                "corrected": edit.corrected,
                "edit": "x -> y",
                "feedback": feedback,
            }
        )
        ratings.append({"item_id": item_id, **scores, "comment": ""})
    write_csv(key, root / "key.csv", KEY_COLUMNS)
    write_csv(sheet, root / "sheet.csv", SHEET_COLUMNS)
    write_csv(ratings, root / "ratings.csv", ("item_id", *RATING_COLUMNS))
    return [
        "--ratings",
        str(root / "ratings.csv"),
        "--key",
        str(root / "key.csv"),
        "--sheet",
        str(root / "sheet.csv"),
        "--edits",
        str(root / "edits.jsonl"),
        "--out",
        str(root / "candidates.md"),
    ]


def test_candidates_rank_failures_compare_with_a_and_leave_out_locness(tmp_path):
    args = write_inputs(tmp_path)
    assert showcase.main(args) == 0
    text = (tmp_path / "candidates.md").read_text(encoding="utf-8")
    good, bad = text.split("## Bad candidates")

    assert "## Good candidates (1)" in good
    assert "system D (`wi-1-1-0-0`)" in good
    assert "**System A on the same edit:** Check the verb." in good
    assert "My sister [don't → doesn't] like dogs." in good
    assert "**Teacher comment:**" in good

    # The invented rule ranks before the wrong rule; the incomplete rating is left out.
    assert bad.index("wi-1-2-0-0") < bad.index("wi-1-3-0-0")
    assert "Incomplete rating." not in text
    # LOCNESS allows no redistribution, so its edits never appear.
    assert "locness" not in text and "They was late" not in text


def test_window_keeps_at_most_the_given_number_of_context_words():
    long = make_edit(
        "wi-1-9-0-0",
        "wi",
        " ".join(f"w{i}" for i in range(30)) + " She go home .",
        31,
        32,
        "go",
        "goes",
        "VERB:SVA",
    )
    text = showcase.window(long, 6)
    assert "[go → goes]" in text
    assert text.startswith("… ") and len(text.replace("[go → goes]", "").split()) <= 6 + 2


def test_no_ratings_yet_exits_cleanly_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "candidates.md"
    assert showcase.main(["--ratings", str(tmp_path / "missing.csv"), "--out", str(out)]) == 0
    assert "No ratings yet" in capsys.readouterr().out
    assert not out.exists()
