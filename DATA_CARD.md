# Data card: Explain-to-Learn

The data used to train and evaluate the Explain-to-Learn systems: learner corrections
from public corpora, silver explanations written by a teacher model, a teacher-built
grammar graph, and the gold evaluation set with its human ratings.

> **Status:** the edit table, the essay splits and the gold edit IDs are built. The
> silver data, the ratings and their statistics are not available yet and will be
> filled in from the result files named below.

Learner text is never stored in the repository. The repository holds the scripts, the
edit IDs and aggregate tables, so that anyone with the corpora can rebuild the data.

## Sources

| Dataset | Use | Licence |
| --- | --- | --- |
| W&I+LOCNESS v2.1 (BEA-2019) | Edits and learner levels; training edits from the train split; gold set from the dev split | Non-commercial research and education |
| FCE v2.1 (BEA-2019) | Extra training edits | Non-commercial research and education |
| Grammar graph (`grammar_graph/`) | Concept, rule and example for each error type | MIT, with the code |
| Silver explanations | Training targets for systems C and D | Inherits the corpus terms |
| Gold set and ratings | Evaluation | Inherits the corpus terms |
| GEE data (Song et al., 2024) | Comparison point for explanations (English examples) | No licence file: used locally, never redistributed |

The corpora must be downloaded from the
[BEA-2019 shared-task page](https://www.cl.cam.ac.uk/research/nl/bea2019st/) under their
own licences (`scripts/download_data.sh`).

## Edit table

One row per gold ERRANT edit, built by aligning the official M2 files to the essays
(`src/explain_to_learn/edits.py`). Each row has the essay ID, the learner's CEFR level,
the sentence, the edit and its ERRANT error type. FCE essays have no individual level
and are labelled B2, the level of the FCE exam; LOCNESS essays are `native`.

Counts per corpus, split, level, operation and error category:
[eval/results/data_stats.md](eval/results/data_stats.md).

## Splits

- **Train and development:** the official BEA-2019 splits, which are by essay
  (`data/splits/*_essays.txt`).
- **Gold set:** 300 W&I+LOCNESS development edits, sampled with a fixed seed and
  stratified by error category and CEFR level (`data/splits/gold_edit_ids.txt`).
- **Fine-tuning validation:** 5% of the essays that have silver examples, chosen with a
  fixed seed, so that all edits of an essay are on the same side.

## Silver explanations

How they are made (`scripts/make_silver.py`, `src/explain_to_learn/silver.py`):

1. A seeded random sample of training edits, excluding the gold set and error types
   without a graph concept.
2. A share of the edits (25% by default) is put in hint mode, the rest in explain mode.
3. The teacher model (gpt-oss-20b, prompted with the grammar graph rule) writes the
   explanation or hint.
4. An output is kept only if it parses, the text for its mode is not empty, it names the
   graph concept, and it names the edited words as the mode requires (a hint must not
   reveal the correction).
5. A random sample of kept items is checked by a teacher, who marks each item correct
   or not.

| | Value |
| --- | --- |
| Edits sent to the teacher | Not available yet |
| Outputs kept, and rejections by reason | Not available yet (`eval/results/silver_filter.md`) |
| Hand-check: items checked | Not available yet |
| Hand-check: share marked correct (95% Wilson interval) | Not available yet |

## Fine-tuning examples

`python -m explain_to_learn.train_data` turns each kept silver item into one chat
example for system C (no grammar rule in the prompt) and one for system D (with the
rule). The two sets have the same edits, modes, levels and targets, so they differ only
in grounding.

| | System C | System D |
| --- | --- | --- |
| Training examples | Not available yet | Not available yet |
| Validation examples | Not available yet | Not available yet |
| Examples with a grammar rule in the prompt | 0 | Not available yet |

## Human ratings

- 100 gold edits x 4 systems = 400 items, shuffled with system names hidden, rated by
  one teacher on the rubric in [eval/rubric.md](eval/rubric.md).
- A random subset of 50 items rated by a second teacher, for agreement.
- The same 400 items rated by an LLM judge with the same rubric.

The rating sheets and the key stay in the git-ignored `outputs/` folder because they
quote learner text. Only aggregate tables are published in `eval/results/`.

| | Value |
| --- | --- |
| Raters and their background | Not available yet |
| Agreement between raters (Cohen's kappa per dimension) | Not available yet |
| Agreement between the LLM judge and the first rater | Not available yet |

## Personal and sensitive information

The essays were written by learners and may contain personal details, opinions or
names. They are not redistributed: the repository holds only IDs and aggregate counts.
Examples in the report and the cards are invented, or short excerpts within what the
corpus licences allow.

## Known biases and limitations

- Exam-style writing by teens and adults; other genres and younger learners are not
  covered.
- The learner levels are uneven: FCE essays all count as B2, and some levels have few
  edits (see the data statistics).
- Silver explanations come from one teacher model and inherit its errors and style; the
  filter checks form, not correctness, and the hand-check covers a sample.
- The grammar graph is written by one teacher and covers about 50 concepts.
- One main rater; agreement is measured on a subset only.

## Licences

W&I+LOCNESS and FCE may be used for non-commercial research and education only, and
publications must cite the papers named in their licences. Everything derived from
them (silver explanations, fine-tuning examples, ratings) carries the same restriction.
The code and the grammar graph are MIT-licensed.
