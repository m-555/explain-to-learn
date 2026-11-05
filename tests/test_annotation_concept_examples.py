"""Example edits per concept: counts from every split, examples from training essays only.

The sentences are invented; each one names its split so the tests can see where an
example came from.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("yaml")

from explain_to_learn.edits import Edit, write_jsonl  # noqa: E402

ROOT = Path(__file__).parents[1]
GRAPH_DIR = Path(__file__).parent / "fixtures" / "graph"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "concept_examples", ROOT / "scripts" / "concept_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


concept_examples = load_script()


def make_edit(essay, idx, split, error_type, level="B1"):
    operation, category = error_type.split(":", 1)
    if category == "DET":
        original, start, end, old, new = f"I like the {split} music {idx} .", 2, 3, "the", ""
    else:
        original, start, end, old, new = f"He go to {split} school {idx} .", 1, 2, "go", "goes"
    return Edit(
        edit_id=f"wi-{essay}-0-{idx}",
        corpus="wi",
        essay_id=essay,
        split=split,
        cefr_level=level,
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=old,
        corrected_text=new,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


EDITS = [
    *[make_edit("1", i, "train", "R:VERB:SVA", ["A2", "B1"][i % 2]) for i in range(6)],
    *[make_edit("2", i, "dev", "R:VERB:SVA", "C1") for i in range(4)],
    *[make_edit("3", i, "test", "R:VERB:SVA", "native") for i in range(3)],
    *[make_edit("4", i, "train", "U:DET") for i in range(2)],
    make_edit("5", 0, "dev", "R:DET"),
]


@pytest.fixture
def written(tmp_path):
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    (tmp_path / "train_essays.txt").write_text("wi-1\nwi-4\n")
    out = tmp_path / "concept_examples"
    concept_examples.main(
        [
            "--edits",
            str(tmp_path / "edits.jsonl"),
            "--train-essays",
            str(tmp_path / "train_essays.txt"),
        ]
        + ["--graph", str(GRAPH_DIR), "--out", str(out), "--n", "4", "--seed", "1"]
    )
    return out


def test_one_file_per_concept_starting_with_the_note(written):
    names = sorted(path.name for path in written.iterdir())
    assert names == [
        "articles_a_an_the.md",
        "index.md",
        "noun_plurals.md",
        "other_error.md",
        "subject_verb_agreement.md",
        "zero_article.md",
    ]
    for path in written.iterdir():
        text = path.read_text()
        assert "Do not copy these sentences into `grammar_graph/concepts.yaml`" in text
        assert text.index("Do not copy") < 200


def test_examples_come_from_training_essays_only(written):
    text = (written / "subject_verb_agreement.md").read_text()

    examples = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(examples) == 4
    assert all("train school" in line and "[go -> goes]" in line for line in examples)
    assert "dev school" not in text and "test school" not in text
    articles = (written / "articles_a_an_the.md").read_text()
    assert "No training edits." in articles and "dev music" not in articles


def test_counts_cover_every_split_per_level(written):
    text = (written / "subject_verb_agreement.md").read_text()

    assert "| `VERB:SVA` | 0 | 3 | 3 | 0 | 4 | 0 | 3 | 13 |" in text
    zero_article = (written / "zero_article.md").read_text()
    assert "### `U:DET`" in zero_article  # the override, not the DET category
    assert "[- the]" in zero_article


def test_concepts_without_edits_and_the_index(written):
    plurals = (written / "noun_plurals.md").read_text()
    assert "No edits in the edit table have the error types NOUN:NUM." in plurals
    assert "It is a prerequisite of: articles_a_an_the, subject_verb_agreement." in plurals

    index = (written / "index.md").read_text()
    rows = [line for line in index.splitlines() if line.startswith("| [")]
    assert rows[0].startswith("| [subject_verb_agreement](subject_verb_agreement.md) | `VERB:SVA`")
    assert "Edits whose error type has no concept: 0." in index


def test_examples_are_seeded():
    pool = [edit for edit in EDITS if edit.split == "train" and edit.error_type == "R:VERB:SVA"]
    first = concept_examples.sample_train(pool, {"wi-1"}, 3, seed=5)

    assert first == concept_examples.sample_train(list(reversed(pool)), {"wi-1"}, 3, seed=5)
    assert concept_examples.sample_train(pool, set(), 3) == []
