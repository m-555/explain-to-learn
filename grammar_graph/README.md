# Grammar graph

The grammar graph is the teaching knowledge that grounds every explanation. It has two files:

- `concepts.yaml`: the grammar concepts, each with a rule worded for three learner levels, a short example, its prerequisite concepts and the technical terms a learner can be expected to know.
- `errant_map.yaml`: which concept explains each [ERRANT](https://github.com/chrisjbryant/errant) error type.

`src/explain_to_learn/graph.py` loads both files into a [NetworkX](https://networkx.org/) directed graph, looks up concepts and checks the files for problems.

## Status of the content

The concept list, the prerequisite links and the mapping are a first draft. Three concepts have complete content as examples of the format: `articles_a_an_the`, `subject_verb_agreement` and `verb_form_after_to`. For all other concepts the rules, example and terms are still empty. All concepts are marked `status: draft` until a teacher has written or checked them.

## concepts.yaml

The file is a list of concepts:

```yaml
- id: subject_verb_agreement
  name: Subject-verb agreement
  status: draft
  rules:
    A2: >-
      After he, she, it or one person or thing, add -s to the verb in the present
      simple: she likes, my dad works. After I, you, we and they, do not add -s.
    B1: >-
      In the present simple, a singular subject ...
    B2: >-
      The verb must agree with its subject in number ...
  example:
    wrong: My brother play football every Sunday.
    right: My brother plays football every Sunday.
  prerequisites: [present_simple, personal_pronouns, noun_plurals]
  terms:
    A2: [verb, present simple, singular, plural]
    B1: [subject, base form]
    B2: [agreement, auxiliary]
```

| Field | Meaning |
| --- | --- |
| `id` | Unique snake_case identifier, used by the mapping and by prerequisites. It does not change once data refers to it. |
| `name` | Short human-readable name, shown to the model and in figures. |
| `status` | `draft` until a teacher has written or checked the content, then `reviewed`. |
| `rules` | The rule worded for three bands: `A2`, `B1` and `B2`. One or two short sentences each. |
| `example` | One invented sentence with the error (`wrong`) and its correction (`right`). Never use learner text from a corpus. |
| `prerequisites` | IDs of the concepts a learner needs first. Links must not form a cycle. |
| `terms` | Technical words a learner at each band can be expected to know, such as *verb* and *plural* at A2 or *auxiliary* at B2. |

### Learner levels and rule bands

Learner levels are mapped to the three rule bands:

| Learner level | Rule wording used |
| --- | --- |
| A1, A2 | `A2` |
| B1 | `B1` |
| B2, C1, C2, native | `B2` |

Terms are cumulative: a B1 learner is expected to know the A2 and B1 terms of every concept, and a B2 learner all of them. The terms of all concepts together form the vocabulary a rule at that level may use, so a rule should only use technical words listed at its band or below. Terms are compared in lower case.

## errant_map.yaml

An ERRANT error type is an operation (`M` missing, `R` replacement, `U` unnecessary) followed by a category, for example `R:VERB:SVA` or `M:DET`. `UNK` has no operation.

```yaml
categories:
  DET: articles_a_an_the
  VERB:SVA: subject_verb_agreement
  # ... one line for every ERRANT category

overrides:
  U:DET: zero_article      # The life is hard -> Life is hard
```

- `categories` must map every ERRANT category: ADJ, ADJ:FORM, ADV, CONJ, CONTR, DET, MORPH, NOUN, NOUN:INFL, NOUN:NUM, NOUN:POSS, ORTH, OTHER, PART, PREP, PRON, PUNCT, SPELL, UNK, VERB, VERB:FORM, VERB:INFL, VERB:SVA, VERB:TENSE and WO.
- `overrides` is optional. It maps a full error type to a different concept when the operation changes what the learner needs to know. For example, an unnecessary article (`U:DET`) is about general meaning without an article, while a wrong article (`R:DET`) is about choosing *a* or *the*.

A lookup checks `overrides` for the full error type first, then `categories` for its category.

Several concepts are not the target of any error type. ERRANT categories are broad (all tense errors are `VERB:TENSE`), so finer concepts such as `present_perfect` are reached as prerequisites of the concept an error maps to.

## Adding or editing a concept

1. Add an entry to `concepts.yaml` with a new snake_case `id`, a `name`, `status: draft` and its `prerequisites`.
2. Write the three `rules`, the `example` and the `terms`. Keep each rule to one or two plain sentences and use only technical words that are listed in `terms` at that band or a lower one.
3. Point an error category or override in `errant_map.yaml` at the new ID, if an error type should be explained by it.
4. Check the files:

   ```bash
   python -m explain_to_learn.graph validate
   ```

   The command prints a coverage report and lists problems: duplicate IDs, unknown prerequisites, prerequisite cycles, mapped concepts that do not exist, ERRANT categories with no concept, and concepts that still miss rules, an example or terms. Missing content is a warning for draft concepts and an error for reviewed ones. With `--strict` it is always an error; the command exits with status 1 if there are errors.
5. When the content has been checked, change `status` to `reviewed`.

## Using the graph in code

```python
from explain_to_learn.graph import load_graph

graph = load_graph()  # reads grammar_graph/*.yaml
concept_id = graph.concept_for("R:VERB:SVA")  # "subject_verb_agreement"
snippet = graph.snippet(concept_id, "B1")  # concept_id, name, rule, example, prerequisites
terms = graph.level_terms("B1")  # A2 and B1 terms of all concepts
graph.graph  # networkx.DiGraph, prerequisite -> concept
```

## Figures and export

Install the `graph` extra (`pip install -e ".[graph]"`), then:

```bash
python scripts/draw_graph.py                 # report/figures/grammar_graph.png
python scripts/export_rdf.py                 # report/grammar_graph.ttl (SKOS in Turtle)
```
