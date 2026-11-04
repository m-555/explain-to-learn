"""Tests for the grammar graph: loading, validation and queries."""

from pathlib import Path

import pytest

pytest.importorskip("networkx")
yaml = pytest.importorskip("yaml")

from explain_to_learn.graph import (  # noqa: E402
    ERRANT_CATEGORIES,
    OPERATIONS,
    GrammarGraph,
    load_graph,
    main,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graph"
REPO_GRAPH_DIR = Path(__file__).parents[1] / "grammar_graph"


def load_fixture() -> GrammarGraph:
    return load_graph(FIXTURE_DIR / "concepts.yaml", FIXTURE_DIR / "errant_map.yaml")


def fixture_data() -> tuple[list[dict], dict]:
    """Return the fixture files as plain data, to change for a test."""
    concepts = yaml.safe_load((FIXTURE_DIR / "concepts.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((FIXTURE_DIR / "errant_map.yaml").read_text(encoding="utf-8"))
    return concepts, mapping


def write_graph(tmp_path: Path, concepts: list[dict], mapping: dict) -> GrammarGraph:
    """Write changed fixture data to files and load them."""
    concepts_path = tmp_path / "concepts.yaml"
    map_path = tmp_path / "errant_map.yaml"
    concepts_path.write_text(yaml.safe_dump(concepts), encoding="utf-8")
    map_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return load_graph(concepts_path, map_path)


def messages(graph: GrammarGraph, severity: str = "error", strict: bool = False) -> list[str]:
    return [p.message for p in graph.validate(strict=strict) if p.severity == severity]


def by_id(concepts: list[dict], concept_id: str) -> dict:
    return next(c for c in concepts if c["id"] == concept_id)


# Loading


def test_load_fixture():
    graph = load_fixture()
    assert list(graph.concepts) == [
        "noun_plurals",
        "articles_a_an_the",
        "zero_article",
        "subject_verb_agreement",
        "other_error",
    ]
    assert graph.graph.has_edge("noun_plurals", "articles_a_an_the")
    assert graph.graph.has_edge("articles_a_an_the", "zero_article")
    assert graph.graph.nodes["noun_plurals"]["status"] == "reviewed"
    assert graph.graph.nodes["zero_article"]["name"] == "No article for general meaning"


def test_load_rejects_concept_without_id(tmp_path):
    concepts, mapping = fixture_data()
    del concepts[0]["id"]
    with pytest.raises(ValueError, match="concept #1"):
        write_graph(tmp_path, concepts, mapping)


def test_load_rejects_unknown_level_key(tmp_path):
    concepts, mapping = fixture_data()
    concepts[0]["rules"]["C1"] = "A C1 rule."
    with pytest.raises(ValueError, match="unknown level 'C1'"):
        write_graph(tmp_path, concepts, mapping)


# Validation


def test_fixture_has_no_errors_and_warns_about_drafts():
    graph = load_fixture()
    assert messages(graph, "error") == []
    assert messages(graph, "warning") == [
        "zero_article: missing rule A2, rule B1, rule B2, example, terms"
    ]


def test_strict_turns_incomplete_drafts_into_errors():
    graph = load_fixture()
    assert messages(graph, "error", strict=True) == [
        "zero_article: missing rule A2, rule B1, rule B2, example, terms"
    ]


def test_incomplete_reviewed_concept_is_an_error(tmp_path):
    concepts, mapping = fixture_data()
    by_id(concepts, "noun_plurals")["rules"]["B1"] = ""
    graph = write_graph(tmp_path, concepts, mapping)
    assert "noun_plurals: missing rule B1" in messages(graph, "error")


def test_missing_example_and_terms(tmp_path):
    concepts, mapping = fixture_data()
    concept = by_id(concepts, "other_error")
    concept["example"]["right"] = ""
    concept["terms"] = {}
    graph = write_graph(tmp_path, concepts, mapping)
    assert messages(graph, "warning") == [
        "zero_article: missing rule A2, rule B1, rule B2, example, terms",
        "other_error: missing example, terms",
    ]


def test_duplicate_id(tmp_path):
    concepts, mapping = fixture_data()
    concepts.append(dict(concepts[0]))
    graph = write_graph(tmp_path, concepts, mapping)
    assert "duplicate concept id 'noun_plurals' (2 times)" in messages(graph)


def test_bad_id_name_and_status(tmp_path):
    concepts, mapping = fixture_data()
    concepts.append({"id": "Bad-Id", "name": "", "status": "finished"})
    graph = write_graph(tmp_path, concepts, mapping)
    errors = messages(graph)
    assert "concept id 'Bad-Id' is not snake_case" in errors
    assert "Bad-Id: missing name" in errors
    assert "Bad-Id: status 'finished' must be draft or reviewed" in errors


def test_unknown_prerequisite(tmp_path):
    concepts, mapping = fixture_data()
    by_id(concepts, "zero_article")["prerequisites"].append("countable_nouns")
    graph = write_graph(tmp_path, concepts, mapping)
    assert "zero_article: unknown prerequisite 'countable_nouns'" in messages(graph)
    assert graph.snippet("zero_article", "B1")["prerequisites"] == ["Articles a, an and the"]


def test_prerequisite_cycle(tmp_path):
    concepts, mapping = fixture_data()
    by_id(concepts, "noun_plurals")["prerequisites"] = ["zero_article"]
    graph = write_graph(tmp_path, concepts, mapping)
    cycles = [m for m in messages(graph) if m.startswith("prerequisite cycle")]
    assert len(cycles) == 1
    for concept_id in ("noun_plurals", "articles_a_an_the", "zero_article"):
        assert concept_id in cycles[0]


def test_concept_that_is_its_own_prerequisite(tmp_path):
    concepts, mapping = fixture_data()
    by_id(concepts, "other_error")["prerequisites"] = ["other_error"]
    graph = write_graph(tmp_path, concepts, mapping)
    assert "prerequisite cycle: other_error -> other_error" in messages(graph)


def test_mapping_to_unknown_concept(tmp_path):
    concepts, mapping = fixture_data()
    mapping["categories"]["SPELL"] = "spelling"
    mapping["overrides"]["M:DET"] = "missing_article"
    graph = write_graph(tmp_path, concepts, mapping)
    errors = messages(graph)
    assert "errant_map: SPELL maps to unknown concept 'spelling'" in errors
    assert "errant_map: M:DET maps to unknown concept 'missing_article'" in errors
    assert graph.concept_for("R:SPELL") is None


def test_unmapped_category(tmp_path):
    concepts, mapping = fixture_data()
    del mapping["categories"]["WO"]
    mapping["categories"]["PUNCT"] = None
    graph = write_graph(tmp_path, concepts, mapping)
    errors = messages(graph)
    assert "errant_map: ERRANT category WO has no concept" in errors
    assert "errant_map: ERRANT category PUNCT has no concept" in errors


def test_unknown_category_and_bad_override(tmp_path):
    concepts, mapping = fixture_data()
    mapping["categories"]["VERB:MODAL"] = "other_error"
    mapping["overrides"]["X:DET"] = "zero_article"
    graph = write_graph(tmp_path, concepts, mapping)
    errors = messages(graph)
    assert "errant_map: 'VERB:MODAL' is not an ERRANT category" in errors
    assert "errant_map: override 'X:DET' must be M, R or U plus a category" in errors


# Queries


@pytest.mark.parametrize(
    ("error_type", "concept_id"),
    [
        ("R:VERB:SVA", "subject_verb_agreement"),
        ("U:DET", "zero_article"),  # operation override
        ("M:DET", "articles_a_an_the"),  # no override: falls back to the category
        ("R:DET", "articles_a_an_the"),
        ("DET", "articles_a_an_the"),  # bare category
        ("R:NOUN:NUM", "noun_plurals"),
        ("UNK", "other_error"),  # UNK has no operation prefix
        (" R:SPELL ", "other_error"),
        ("noop", None),
        ("R:VERB:MODAL", None),
    ],
)
def test_concept_for(error_type, concept_id):
    assert load_fixture().concept_for(error_type) == concept_id


def test_error_types_for():
    graph = load_fixture()
    assert graph.error_types_for("articles_a_an_the") == ["DET"]
    assert graph.error_types_for("zero_article") == ["U:DET"]


@pytest.mark.parametrize(
    ("level", "band"),
    [
        ("A1", "A2"),
        ("A2", "A2"),
        ("B1", "B1"),
        ("b1", "B1"),
        ("B2", "B2"),
        ("C1", "B2"),
        ("C2", "B2"),
        ("native", "B2"),
    ],
)
def test_snippet_uses_rule_for_level(level, band):
    graph = load_fixture()
    snippet = graph.snippet("subject_verb_agreement", level)
    assert snippet["rule"] == graph.concepts["subject_verb_agreement"].rules[band]


def test_snippet_fields():
    snippet = load_fixture().snippet("articles_a_an_the", "B1")
    assert snippet == {
        "concept_id": "articles_a_an_the",
        "name": "Articles a, an and the",
        "rule": "Use a or an with a singular countable noun the reader does not know yet.",
        "example": {"wrong": "I bought book.", "right": "I bought a book."},
        "prerequisites": ["Singular and plural nouns"],
    }


def test_snippet_rejects_unknown_level_and_concept():
    graph = load_fixture()
    with pytest.raises(ValueError, match="unknown level"):
        graph.snippet("noun_plurals", "B3")
    with pytest.raises(KeyError):
        graph.snippet("modal_verbs", "B1")


def test_level_terms_are_cumulative():
    graph = load_fixture()
    a2 = {"noun", "plural", "article", "verb", "word"}
    b1 = a2 | {"countable", "singular", "subject"}
    b2 = b1 | {"quantifier", "determiner", "agreement"}
    assert graph.level_terms("A1") == a2
    assert graph.level_terms("A2") == a2
    assert graph.level_terms("B1") == b1
    assert graph.level_terms("C2") == b2
    assert graph.level_terms("native") == b2


# Command line


def test_cli_validate(capsys):
    args = ["validate", "--concepts", str(FIXTURE_DIR / "concepts.yaml")]
    args += ["--map", str(FIXTURE_DIR / "errant_map.yaml")]
    assert main(args) == 0
    report = capsys.readouterr().out
    assert "Concepts: 5 (1 reviewed, 4 draft)" in report
    assert "ERRANT categories mapped: 25/25" in report
    assert "Errors: 0, warnings: 1" in report
    assert main([*args, "--strict"]) == 1


# The real grammar graph in grammar_graph/


def test_repository_graph_validates_and_maps_every_type():
    graph = load_graph(REPO_GRAPH_DIR / "concepts.yaml", REPO_GRAPH_DIR / "errant_map.yaml")
    assert [str(p) for p in graph.validate() if p.severity == "error"] == []
    for category in ERRANT_CATEGORIES:
        assert graph.concept_for(category) is not None, category
        for operation in OPERATIONS:
            assert graph.concept_for(f"{operation}:{category}") is not None
