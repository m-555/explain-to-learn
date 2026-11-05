"""The end-to-end pipeline with a fake corrector, annotator and model.

The prompts, parsing and graph lookups are the real ones. Tests of the ERRANT
annotator skip when ERRANT or spaCy's English model is not installed. All sentences
are invented.
"""

import dataclasses
import difflib
import json
import re
from pathlib import Path

import pytest

from explain_to_learn.edits import Edit
from explain_to_learn.generate import Decoding, OpenAIBackend, edit_item, run_batch
from explain_to_learn.pipeline import (
    CORRECTION_PROMPT,
    EndpointCorrector,
    Explainer,
    Span,
    TextResult,
    clean_correction,
    explain_gold_edits,
    explain_text,
    feedback_markdown,
    load_correctors,
    make_corrector,
    make_edits,
    pair_sentences,
    text_edits,
)

ROOT = Path(__file__).parents[1]
FIXTURE_GRAPH = Path(__file__).parent / "fixtures" / "graph"
TYPES = {("like", "likes"): "R:VERB:SVA", ("", "a"): "M:DET", ("cat", "cats"): "R:NOUN:NUM"}


@pytest.fixture(scope="module")
def graph():
    pytest.importorskip("networkx")
    pytest.importorskip("yaml")
    from explain_to_learn.graph import load_graph

    return load_graph(FIXTURE_GRAPH / "concepts.yaml", FIXTURE_GRAPH / "errant_map.yaml")


class TableCorrector:
    """Corrects the sentences it knows and counts how often it is called."""

    def __init__(self, corrections: dict[str, str]):
        self.corrections, self.calls = corrections, 0

    def correct(self, sentences):
        self.calls += 1
        return [self.corrections.get(s, s) for s in sentences]


class WordAnnotator:
    """Sentences end with `.`; tokens are words and full stops; types from `TYPES`."""

    def sentences(self, text):
        return [" ".join(s.split()) for s in re.findall(r"[^.]+\.?", text) if s.strip()]

    def annotate(self, original, corrected):
        tokens = re.findall(r"\w+|\.", original)
        corrected_tokens = re.findall(r"\w+|\.", corrected)
        matcher = difflib.SequenceMatcher(a=tokens, b=corrected_tokens, autojunk=False)
        spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                old, new = " ".join(tokens[i1:i2]), " ".join(corrected_tokens[j1:j2])
                spans.append(Span(i1, i2, new, TYPES.get((old, new), "R:OTHER")))
        return tokens, corrected_tokens, spans


class RecordingModel:
    """Remembers the prompts and answers with the concept named in the prompt."""

    def __init__(self, reply: str | None = None):
        self.prompts, self.reply = [], reply

    def complete(self, messages, decoding):
        self.prompts.append(messages)
        if self.reply is not None:
            return self.reply
        user = messages[-1]["content"]
        concept = re.search(r"Concept: (.+)", user)
        hint = "Mode: hint" in user
        return json.dumps(
            {
                "concept": concept.group(1) if concept else "Unknown",
                "explanation": "" if hint else "Use the corrected words.",
                "hint": "Look at the verb." if hint else "",
            }
        )


TEXT = "She like tea. I bought book.  It is fine."
CORRECTIONS = {"She like tea.": "She likes tea.", "I bought book.": "I bought a book."}


def run(graph, text=TEXT, level="B1", mode="explain", grounded=True, **kwargs):
    model = RecordingModel()
    result = explain_text(
        text,
        level,
        mode,
        kwargs.pop("corrector", TableCorrector(CORRECTIONS)),
        Explainer(model, grounded=grounded, system="D"),
        graph,
        annotator=WordAnnotator(),
        **kwargs,
    )
    return result, model


def test_explain_text_gives_one_result_per_edit(graph):
    result, model = run(graph)

    assert result.pairs == [
        ("She like tea.", "She likes tea."),
        ("I bought book.", "I bought a book."),
        ("It is fine.", "It is fine."),
    ]
    assert result.corrected_text == "She likes tea. I bought a book. It is fine."
    edits = result.edits
    assert [(e.edit_id, e.error_type, e.original_text, e.corrected_text) for e in edits] == [
        ("text-input-0-0", "R:VERB:SVA", "like", "likes"),
        ("text-input-1-0", "M:DET", "", "a"),
    ]
    assert edits[1].original == "I bought book ." and edits[1].corrected == "I bought a book ."
    assert (edits[1].sent_idx, edits[1].start, edits[1].end, edits[1].operation) == (1, 2, 2, "M")
    assert [item.concept for item in result.explained] == [
        "Subject-verb agreement",
        "Articles a, an and the",
    ]
    assert [item.text for item in result.explained] == ["Use the corrected words."] * 2
    assert len(model.prompts) == 2


def test_free_text_edits_get_the_inference_prompt_of_the_evaluation(graph):
    result, model = run(graph, level="a2", mode="hint")

    for item, prompt in zip(result.explained, model.prompts, strict=True):
        assert item.edit.cefr_level == "A2"
        assert prompt == edit_item(item.edit, "hint", graph, grounded=True)[1]
        assert "Learner level: A2" in prompt[1]["content"]
        assert "Grammar rule (from the grammar graph):" in prompt[1]["content"]
    assert result.explained[0].text == "Look at the verb."
    assert result.explained[0].record["system"] == "D"


def test_ungrounded_explainer_gets_no_snippet_but_the_concept_is_named(graph):
    result, model = run(graph, grounded=False)

    assert all("grammar graph" not in prompt[1]["content"] for prompt in model.prompts)
    assert not any(item.record["grounded"] for item in result.explained)
    assert result.explained[0].concept == "Subject-verb agreement"


def test_a_given_correction_needs_no_corrector(graph):
    corrector = TableCorrector({})
    result, _ = run(graph, text="She like tea.", corrector=corrector, corrected="She likes tea.")

    assert corrector.calls == 0
    assert [item.edit.error_type for item in result.explained] == ["R:VERB:SVA"]


def test_pair_sentences_falls_back_to_one_unit():
    annotator = WordAnnotator()
    assert pair_sentences("A b. C d.", "A b. C e.", annotator) == [
        ("A b.", "A b."),
        ("C d.", "C e."),
    ]
    assert pair_sentences("A b. C d.", "A b and c d.", annotator) == [("A b. C d.", "A b and c d.")]
    assert pair_sentences("   ", "x", annotator) == []


def test_text_edits_checks_its_inputs():
    annotator = WordAnnotator()
    with pytest.raises(ValueError, match="level"):
        text_edits("She like tea.", "B3", TableCorrector({}), annotator)
    with pytest.raises(ValueError, match="corrector"):
        text_edits("She like tea.", "B1", None, annotator)
    with pytest.raises(ValueError, match="empty"):
        text_edits("She like tea.", "B1", None, annotator, corrected=" ")
    assert text_edits("", "C2", TableCorrector({}), annotator) == ([], [])


def test_gold_edits_give_the_rows_that_run_system_writes(graph, tmp_path):
    edits = [
        make_edits(
            ["She", "like", "tea", "."],
            ["She", "likes", "tea", "."],
            [Span(1, 2, "likes", "R:VERB:SVA")],
            "native",
        )[0],
        make_edits(
            ["I", "bought", "book", "."],
            ["I", "bought", "a", "book", "."],
            [Span(2, 2, "a", "M:DET")],
            "B2",
            sent_idx=3,
        )[0],
    ]
    explained = explain_gold_edits(
        edits, "explain", Explainer(RecordingModel(), system="B"), graph, run="gold"
    )

    items = [edit_item(e, "explain", graph, True, system="B", run="gold") for e in edits]
    run_batch(RecordingModel(), items, tmp_path / "rows.jsonl", Decoding())
    expected = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    for item, row in zip(explained, expected, strict=True):
        got = item.to_row()
        assert got.pop("latency_s") >= 0 and row.pop("latency_s") >= 0
        assert got == row
    assert explained[0].record["level"] == "C2"  # native writers are treated as C2


def test_unreadable_answers_and_markdown(graph):
    model = RecordingModel(reply="not json")
    result = explain_text(
        "She like tea.",
        "B2",
        "explain",
        TableCorrector(CORRECTIONS),
        Explainer(model, retries=0),
        graph,
        annotator=WordAnnotator(),
    )
    assert result.explained[0].text == "" and result.explained[0].generation.error
    assert "could not be read" in feedback_markdown(result)
    assert result.to_dict()["results"][0]["error_type"] == "R:VERB:SVA"

    unchanged, _ = run(graph, text="It is fine.")
    assert feedback_markdown(unchanged) == "No corrections needed."
    assert feedback_markdown(TextResult([], [])) == ""

    text = feedback_markdown(run(graph)[0])
    assert text.startswith("1. ~~like~~ → **likes** · *Subject-verb agreement*")
    assert "2. add **a** · *Articles a, an and the*" in text


def test_make_edits_fills_the_edit_table_fields():
    tokens = ["She", "do", "n't", "like", "it", "."]
    corrected = ["She", "does", "n't", "like", "it", "."]
    (edit,) = make_edits(tokens, corrected, [Span(1, 2, "does", "R:VERB:SVA")], "C1", 2, "e7")

    assert set(dataclasses.asdict(edit)) == {f.name for f in dataclasses.fields(Edit)}
    assert (edit.edit_id, edit.essay_key, edit.sent_idx) == ("text-e7-2-0", "text-e7", 2)
    assert (edit.original_text, edit.corrected_text, edit.cefr_level) == ("do", "does", "C1")
    assert (edit.operation, edit.error_category) == ("R", "VERB:SVA")
    assert edit.corrected == "She does n't like it ."


class ScriptedBackend:
    def __init__(self, replies):
        self.replies, self.messages = list(replies), []

    def complete(self, messages, decoding):
        self.messages.append(messages)
        return self.replies.pop(0)


def test_endpoint_corrector_prompts_and_cleans_replies():
    backend = ScriptedBackend(['"She likes tea."', "", "We were there.\nI fixed the verb."])
    corrector = EndpointCorrector(backend)

    assert corrector.correct(["She like tea.", "Fine.", "We was there."]) == [
        "She likes tea.",
        "Fine.",
        "We were there.",
    ]
    assert backend.messages[0][0] == {"role": "system", "content": CORRECTION_PROMPT}
    assert backend.messages[0][1] == {"role": "user", "content": "She like tea."}


def test_clean_correction_keeps_quotes_the_learner_wrote():
    assert clean_correction('"Hi," she said.', '"Hi," she say.') == '"Hi," she said.'
    assert clean_correction("'Fine.'", "Fine.") == "Fine."
    assert clean_correction("  \n", "Fine.") == "Fine."


def test_correctors_come_from_the_systems_config(monkeypatch):
    pytest.importorskip("yaml")
    from explain_to_learn.generate import load_systems

    correctors = load_correctors(ROOT / "configs" / "systems.yaml")
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    assert correctors["gec"]["kind"] == "gec"
    assert correctors["gec"]["model"] == "Unbabel/gec-t5_small"
    # The prompted corrector is the explanation model's base model on the A/B server.
    assert correctors["llm"]["model"] == systems["A"]["model"]
    assert correctors["llm"]["base_url"] == systems["A"]["base_url"]

    monkeypatch.setenv("API_KEY", "secret")
    corrector = make_corrector(correctors["llm"])
    assert isinstance(corrector.backend, OpenAIBackend)
    assert (corrector.backend.api_key, corrector.workers) == ("secret", 8)
    assert corrector.decoding.max_new_tokens == 256
    with pytest.raises(ValueError, match="Unknown corrector"):
        make_corrector({"kind": "magic"})


def test_errant_annotator_splits_tokenises_and_types(errant_annotator):
    sentences = errant_annotator.sentences("She don't like  tea.\n\nWe was happy!")
    assert sentences == ["She don't like tea.", "We was happy!"]

    tokens, corrected, spans = errant_annotator.annotate("I has a apple.", "I have an apple.")
    assert tokens == ["I", "has", "a", "apple", "."]
    assert corrected == ["I", "have", "an", "apple", "."]
    assert spans == [Span(1, 2, "have", "R:VERB:SVA"), Span(2, 3, "an", "R:DET")]
    assert errant_annotator.annotate("Fine.", "Fine.")[2] == []


def test_explain_text_with_errant(errant_annotator, graph):
    result = explain_text(
        "She don't like to going.",
        "B1",
        "explain",
        None,
        Explainer(RecordingModel()),
        graph,
        annotator=errant_annotator,
        corrected="She doesn't like to go.",
    )
    assert [(e.original_text, e.corrected_text, e.error_type) for e in result.edits] == [
        ("do", "does", "R:VERB:SVA"),
        ("going", "go", "R:VERB:FORM"),
    ]
    assert result.edits[0].original == "She do n't like to going ."
