import json
from pathlib import Path

import pytest

from explain_to_learn.edits import Edit
from explain_to_learn.metrics import (
    breakdown,
    concept_matches,
    count_sentences,
    faithfulness,
    find_terms,
    load_terms,
    markdown_report,
    mentions,
    paired_bootstrap,
    parse_errant_compare,
    score_item,
    summarize,
    write_report,
)

TERMS = ["verb", "subject", "past participle", "participle", "article", "auxiliary verb"]


def make_edit(original, start, end, corrected_text, error_type, level="B1", edit_id="e1"):
    """A synthetic edit; the sentences are invented for the tests."""
    tokens = original.split()
    operation, category = error_type.split(":", 1)
    return Edit(
        edit_id=edit_id,
        corpus="wi",
        essay_id="test",
        split="dev",
        cefr_level=level,
        sent_idx=0,
        original=original,
        corrected=original,
        start=start,
        end=end,
        original_text=" ".join(tokens[start:end]),
        corrected_text=corrected_text,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


SVA = make_edit("She do n't like tea .", 1, 2, "does", "R:VERB:SVA")
CONTRACTION = make_edit("He do n't like it .", 1, 3, "does n't", "R:VERB:SVA")
MISSING_DET = make_edit("I have cat .", 2, 2, "a", "M:DET")
EXTRA_ADV = make_edit("We were very very happy .", 3, 4, "", "U:ADV")
CASE = make_edit("I live in london .", 3, 4, "London", "R:ORTH")
PUNCT = make_edit("Hello , how are you ?", 1, 2, "!", "R:PUNCT")
MISSING_PUNCT = make_edit("However we left .", 1, 1, ",", "M:PUNCT")
FORMAL = make_edit("I do n't know .", 1, 3, "do not", "R:OTHER")


# --- mentions and faithfulness --------------------------------------------------


def test_mentions_ignores_case_quotes_and_punctuation():
    assert mentions("Use “Does,” here.", "does")
    assert mentions("Write 'doesn't' after she.", "does n't")
    assert mentions("Write does not after she.", "does n't")
    assert not mentions("Write it after she.", "does")


def test_short_words_count_only_when_quoted():
    assert mentions('Add "a" before "cat".', "a")
    assert not mentions("Add a word before cat.", "a")
    assert not mentions("", "a")


def test_punctuation_is_named_by_name_or_quoted_mark():
    assert mentions("Use an exclamation mark here.", "!")
    assert mentions('Put "!" at the end.', "!")
    assert not mentions("Change the punctuation, please.", ",")
    assert mentions("Add a comma after However.", ",")


def test_explain_replacement_needs_both_sides():
    good = faithfulness(SVA, "explain", 'With "she", use "does", not "do".')
    only_new = faithfulness(SVA, "explain", 'With "she", the verb is "does".')

    assert good.faithful and good.reason is None
    assert not only_new.faithful
    assert only_new.reason == "missing_original"


def test_explain_missing_and_unnecessary_words_need_one_side():
    assert faithfulness(MISSING_DET, "explain", 'Add "a" before a singular noun.').faithful
    assert not faithfulness(MISSING_DET, "explain", "Add an article before cat.").faithful
    assert faithfulness(EXTRA_ADV, "explain", "Remove the second very.").faithful
    assert faithfulness(EXTRA_ADV, "explain", "Remove one adverb.").reason == "missing_original"
    extra_det = make_edit("I like the music .", 2, 3, "", "U:DET")
    assert faithfulness(extra_det, "explain", 'Remove "the" before music.').faithful
    assert not faithfulness(extra_det, "explain", "Remove the article.").faithful


def test_explain_tokenised_contractions_match_written_forms():
    text = "Use “doesn’t” with he, not “don’t”."
    assert faithfulness(CONTRACTION, "explain", text).faithful


def test_contraction_style_edit_compares_exact_forms():
    # "don't" and "do not" are the same words, so the check must be exact.
    assert faithfulness(FORMAL, "explain", 'In formal writing use "do not", not "don\'t".').faithful
    assert not faithfulness(FORMAL, "explain", 'In formal writing use "do not".').faithful


def test_case_only_edit_needs_the_capitalised_form():
    assert faithfulness(CASE, "explain", 'Write "London" with a capital letter.').faithful
    assert faithfulness(CASE, "explain", 'Write "London", not "london".').faithful
    assert not faithfulness(CASE, "explain", 'Write "london" with a capital letter.').faithful


def test_punctuation_edits():
    assert faithfulness(PUNCT, "explain", "Use an exclamation mark, not a comma.").faithful
    assert not faithfulness(PUNCT, "explain", "Use an exclamation mark here.").faithful
    assert faithfulness(MISSING_PUNCT, "explain", 'Put a comma after "However".').faithful


def test_hint_locates_without_giving_the_answer():
    assert faithfulness(SVA, "hint", 'Look at "do" after "she".').faithful
    leak = faithfulness(SVA, "hint", 'Change "do" to "does".')
    assert not leak.faithful and leak.reason == "reveals_correction"
    assert faithfulness(SVA, "hint", "Look at the verb.").reason == "missing_original"


def test_hint_for_missing_word_names_a_neighbour():
    assert faithfulness(MISSING_DET, "hint", 'Is a word missing before "cat"?').faithful
    assert not faithfulness(MISSING_DET, "hint", "Is a word missing?").faithful
    assert not faithfulness(MISSING_DET, "hint", 'Add "a" before "cat".').faithful


def test_hint_correction_already_in_sentence_is_not_a_leak():
    edit = make_edit("The cat and the the dog .", 3, 5, "the", "R:OTHER")
    assert faithfulness(edit, "hint", 'Read "the the" again.').reveals_correction is False


# --- concepts and level fit -----------------------------------------------------


def test_concept_matches_id_or_name():
    labels = {"subject_verb_agreement", "Subject-verb agreement"}
    assert concept_matches("Subject verb agreement", labels)
    assert concept_matches("SUBJECT_VERB_AGREEMENT", labels)
    assert not concept_matches("Verb tense", labels)
    assert not concept_matches("", labels)


def test_find_terms_prefers_longer_terms_and_plurals():
    text = "Use past participles after an auxiliary verb; check the subject and articles."
    assert find_terms(text, TERMS) == ["past participle", "auxiliary verb", "subject", "article"]


def test_count_sentences_ignores_full_stops_in_quotes():
    assert count_sentences('Write "She likes tea." with -s.') == 1
    assert count_sentences("Use does. It agrees with she!") == 2
    assert count_sentences("No full stop") == 1
    assert count_sentences("") == 0


def test_bundled_term_list_loads():
    terms = load_terms(Path(__file__).parents[1] / "configs" / "grammar_terms.txt")
    assert "past participle" in terms and "#" not in "".join(terms)


# --- score_item and summaries ---------------------------------------------------

LABELS = {"subject_verb_agreement", "Subject-verb agreement"}
GOOD = {
    "concept": "Subject-verb agreement",
    "explanation": 'After "she", use "does", not "do": the verb takes -s.',
    "hint": "",
}


def test_score_item_good_reply():
    item = score_item(SVA, "explain", GOOD, LABELS, {"verb"}, TERMS)

    assert item["covered"] == 1.0
    assert item["concept_correct"] == 1.0
    assert item["faithful"] == 1.0
    assert item["within_length"] == 1.0
    assert (item["terms"], item["off_level_terms"]) == (1, 0)
    assert item["contradicts_rule"] is None


def test_score_item_failed_reply_counts_against_coverage():
    item = score_item(SVA, "explain", None, LABELS, {"verb"}, TERMS)

    assert (item["covered"], item["concept_correct"], item["faithful"]) == (0.0, 0.0, 0.0)
    assert item["within_length"] is None and item["terms"] is None


def test_score_item_unmapped_concept_and_off_level_terms():
    text = 'Use "does", not "do": the subject needs an auxiliary verb with -s. ' * 3
    reply = {**GOOD, "explanation": text}
    item = score_item(SVA, "explain", reply, None, {"verb"}, TERMS, level="A2")

    assert item["concept_correct"] is None
    assert item["within_length"] == 0.0
    assert item["off_level_terms"] == item["terms"] == 6


def test_score_item_hint_mode_uses_the_hint_field():
    reply = {"concept": "Subject-verb agreement", "explanation": "", "hint": 'Look at "do".'}
    assert score_item(SVA, "hint", reply, LABELS, set(), TERMS)["faithful"] == 1.0
    assert score_item(SVA, "explain", reply, LABELS, set(), TERMS)["covered"] == 0.0


def test_score_item_runs_the_contradiction_hook():
    snippet = {"rule": "Use -s with she.", "example": {}}
    item = score_item(
        SVA,
        "explain",
        GOOD,
        LABELS,
        set(),
        TERMS,
        snippet=snippet,
        contradiction_check=lambda text, snip: "never" in text,
    )
    assert item["contradicts_rule"] == 0.0


def test_summarize_and_breakdown():
    other = make_edit("I have cat .", 2, 2, "a", "M:DET", level="A2", edit_id="e2")
    items = [
        score_item(SVA, "explain", GOOD, LABELS, {"verb"}, TERMS),
        score_item(other, "explain", None, {"articles"}, set(), TERMS),
    ]
    summary = summarize(items)

    assert summary["edits"] == 2
    assert summary["covered"] == 0.5
    assert summary["within_length_n"] == 1
    assert summary["off_level_term_share"] == 0.0
    assert set(breakdown(items, "level")) == {"A2", "B1"}
    assert breakdown(items, "error_category")["DET"]["covered"] == 0.0


# --- paired bootstrap -----------------------------------------------------------


def test_bootstrap_identical_systems_give_zero_interval():
    scores = {f"e{i}": float(i % 3 == 0) for i in range(60)}
    result = paired_bootstrap(scores, dict(scores), n_resamples=500)

    assert result["diff"] == 0.0
    assert result["low"] == result["high"] == 0.0
    assert result["p_value"] == 1.0


def test_bootstrap_is_deterministic_and_covers_the_difference():
    a = {f"e{i}": float(i % 2 == 0) for i in range(100)}
    b = {f"e{i}": float(i % 4 == 0) for i in range(100)}
    first = paired_bootstrap(a, b, n_resamples=2000, seed=7)

    assert first == paired_bootstrap(a, b, n_resamples=2000, seed=7)
    assert first["diff"] == pytest.approx(0.25)
    assert first["low"] < 0.25 < first["high"]
    assert first["low"] > 0 and first["p_value"] < 0.05


def test_bootstrap_uses_only_shared_scored_edits():
    a = {"e1": 1.0, "e2": None, "e3": 1.0}
    b = {"e1": 0.0, "e2": 1.0}
    result = paired_bootstrap(a, b, n_resamples=100)

    assert result["n"] == 1 and result["diff"] == 1.0
    assert paired_bootstrap({}, {}, n_resamples=10)["diff"] is None


# --- ERRANT and report ----------------------------------------------------------

ERRANT_OUTPUT = """
=========== Span-Based Correction ============
TP	FP	FN	Prec	Rec	F0.5
2	1	3	0.6667	0.4	0.5882
==============================================
"""


def test_parse_errant_compare():
    assert parse_errant_compare(ERRANT_OUTPUT) == {
        "tp": 2,
        "fp": 1,
        "fn": 3,
        "precision": 0.6667,
        "recall": 0.4,
        "f0.5": 0.5882,
    }
    with pytest.raises(ValueError):
        parse_errant_compare("nothing here")


def test_report_has_tables_and_no_learner_text(tmp_path):
    items = [score_item(SVA, "explain", GOOD, LABELS, {"verb"}, TERMS)]
    system = {
        "overall": summarize(items),
        "by_level": breakdown(items, "level"),
        "by_error_category": breakdown(items, "error_category"),
    }
    comparison = paired_bootstrap({"e1": 1.0}, {"e1": 0.0}, n_resamples=10)
    results = {
        "run": "gold-explain",
        "systems": {"A": system, "B": system},
        "comparisons": {"B - A": {"faithful": comparison}},
    }
    md_path, json_path = write_report(results, tmp_path)

    text = md_path.read_text()
    assert md_path.name == "metrics_gold-explain.md"
    assert "| A | 1 | 100.0 | 100.0 | 100.0 | - | 100.0 | 0.0 |" in text
    assert "| B - A | faithful | 1 | 100.0 |" in text
    assert "like tea" not in text and "does" not in markdown_report(results)
    assert json.loads(json_path.read_text())["run"] == "gold-explain"
