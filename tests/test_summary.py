"""Teacher summary: grouping by concept, teaching order from prerequisites, Markdown.

Uses the fixture graph, where noun_plurals is a prerequisite of articles_a_an_the and
subject_verb_agreement, and articles_a_an_the of zero_article. All sentences are
invented.
"""

from pathlib import Path

import pytest

from explain_to_learn.pipeline import Span, make_edits
from explain_to_learn.summary import (
    ancestors,
    describe_position,
    essay_edits,
    markdown,
    summarize,
    teaching_order,
)

pytest.importorskip("networkx")
pytest.importorskip("yaml")

from explain_to_learn.graph import Concept, GrammarGraph, load_graph  # noqa: E402

FIXTURE_GRAPH = Path(__file__).parent / "fixtures" / "graph"


@pytest.fixture(scope="module")
def graph():
    return load_graph(FIXTURE_GRAPH / "concepts.yaml", FIXTURE_GRAPH / "errant_map.yaml")


def edit(sent_idx, start, end, old, new, error_type):
    """One edit in an invented sentence of five words."""
    tokens = ["w0", "w1", "w2", "w3", "w4"]
    tokens[start:end] = old.split()
    return make_edits(tokens, tokens, [Span(start, end, new, error_type)], "B1", sent_idx)[0]


SVA_1 = edit(0, 1, 2, "like", "likes", "R:VERB:SVA")
SVA_2 = edit(2, 0, 1, "go", "goes", "R:VERB:SVA")
ARTICLE = edit(1, 2, 2, "", "a", "M:DET")
PLURAL = edit(3, 3, 4, "cat", "cats", "R:NOUN:NUM")
ZERO = edit(4, 0, 1, "The", "", "U:DET")
UNKNOWN = edit(1, 4, 5, "x", "y", "R:NOT:MAPPED")


def test_a_concept_comes_after_its_prerequisites_with_errors(graph):
    summary = summarize([SVA_1, SVA_2, ARTICLE, PLURAL], graph, "B1")
    names = [c["concept_id"] for c in summary["concepts"]]

    # Agreement has the most edits, but builds on plurals, which also has an error.
    assert names == ["noun_plurals", "subject_verb_agreement", "articles_a_an_the"]
    assert [c["step"] for c in summary["concepts"]] == [1, 2, 3]
    assert summary["concepts"][1]["count"] == 2
    assert summary["concepts"][1]["builds_on"] == ["Singular and plural nouns"]


def test_prerequisites_count_through_a_chain(graph):
    # zero_article -> articles_a_an_the -> noun_plurals; articles has no error here.
    assert ancestors(graph, "zero_article") == {"articles_a_an_the", "noun_plurals"}
    summary = summarize([ZERO, ZERO, PLURAL], graph)
    assert [c["concept_id"] for c in summary["concepts"]] == ["noun_plurals", "zero_article"]
    assert summary["concepts"][1]["builds_on"] == ["Singular and plural nouns"]


def test_ties_go_to_the_concept_that_comes_first_in_the_text(graph):
    other = edit(0, 0, 1, "very", "really", "R:ADV")  # other_error, no prerequisites
    order = [c["concept_id"] for c in summarize([PLURAL, other], graph)["concepts"]]
    assert order == ["other_error", "noun_plurals"]
    shuffled = summarize([other, PLURAL], graph)
    assert [c["concept_id"] for c in shuffled["concepts"]] == order


def test_edits_without_a_concept_are_listed(graph):
    summary = summarize([SVA_1, UNKNOWN], graph)
    assert [c["concept_id"] for c in summary["concepts"]] == ["subject_verb_agreement"]
    assert summary["without_concept"] == [
        {
            "edit_id": "text-input-1-0",
            "sentence": 2,
            "start": 4,
            "end": 5,
            "original": "x",
            "correction": "y",
            "error_type": "R:NOT:MAPPED",
        }
    ]
    assert summary["sentences_with_edits"] == 2


def test_rule_is_worded_for_the_level(graph):
    a2 = summarize([SVA_1], graph, "A2")["concepts"][0]["rule"]
    b2 = summarize([SVA_1], graph, "B2")["concepts"][0]["rule"]
    assert a2 == "With he, she or it, add -s to the verb."
    assert b2 == "The verb agrees with its subject in number."
    assert summarize([SVA_1], graph)["concepts"][0]["rule"] == ""


def test_a_cycle_in_the_graph_does_not_stop_the_order():
    concepts = [
        Concept("first", "First", prerequisites=["second"]),
        Concept("second", "Second", prerequisites=["first"]),
    ]
    graph = GrammarGraph(concepts, {}, {})
    groups = {"first": [SVA_1], "second": [SVA_2]}
    assert sorted(teaching_order(groups, graph)) == ["first", "second"]


def test_markdown_lists_steps_positions_and_unmapped_edits(graph):
    edits = [SVA_1, SVA_2, ARTICLE, PLURAL, UNKNOWN]
    text = markdown(summarize(edits, graph, "B1", max_examples=1))

    assert text.startswith("# Teacher summary\n\nLevel B1 · 5 edits · 3 concepts\n")
    assert "1. **Singular and plural nouns** · 1 edit" in text
    assert (
        "2. **Subject-verb agreement** · 2 edits\n   - Builds on: Singular and plural nouns" in text
    )
    assert "   - sentence 1, word 2: *like* → *likes*\n   - and 1 more edit" in text
    assert "sentence 2, before word 3: add *a*" in text
    assert "## Edits without a grammar concept" in text
    assert "No corrections" in markdown(summarize([], graph, "B1"))


def test_describe_position_for_each_operation():
    base = {"sentence": 3, "original": "very much", "correction": "", "error_type": "U:ADV"}
    assert describe_position({**base, "start": 4, "end": 6}) == (
        "sentence 3, words 5-6: remove *very much*"
    )
    assert describe_position({**base, "start": 0, "end": 0, "original": "", "correction": "A"}) == (
        "sentence 3, before word 1: add *A*"
    )


def test_essay_edits_selects_one_essay(graph):
    other = make_edits(["a"], ["b"], [Span(0, 1, "b", "R:SPELL")], "B1", text_id="other")[0]
    assert essay_edits([SVA_1, other, SVA_2], "text-input") == [SVA_1, SVA_2]
