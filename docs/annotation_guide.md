# Annotation guide

The teacher's annotations are the core of this project's evaluation. This guide describes
the local annotation app that collects them, the files it writes and the scripts that
read those files. It is written for anyone who repeats the work, and for the second
rater, who only needs the [Second rater](#c-second-rater) section and the
[Advice](#advice-for-raters).

The app has one tab per task:

| Task | Items | Output in `outputs/annotations/` | Read by |
| --- | --- | --- | --- |
| [Gold explanations](#a-gold-explanations) | 300 gold edits | `gold_explanations.jsonl` | the gold reference set |
| [Blind ratings](#b-blind-ratings) | 400 system outputs | `ratings_<rater>.csv` | `explain_to_learn.human_eval`, `scripts/agreement.py` |
| [Second rater](#c-second-rater) | 50 of those outputs | `ratings_<rater>.csv` | `scripts/agreement.py` |
| [Silver hand-check](#d-silver-hand-check) | 200 silver training items | `silver_handcheck.csv` | `scripts/make_silver.py score` |
| [Preference pairs](#e-preference-pairs) | pairs of outputs | `preferences.jsonl` | the preference (DPO) training data |

Every file quotes learner text, so everything stays in the git-ignored `outputs/`
folder. Only aggregate tables, made by the scripts above, go into `eval/results/`.

## Starting the app

The app needs Python 3.11 or newer and Gradio (the `demo` extra). It runs on a CPU.
The gold and preference tasks also read the grammar graph (the `graph` extra).

```bash
pip install -e ".[demo,graph]"
python -m explain_to_learn.annotation
```

Then open <http://127.0.0.1:7861> in a browser. The app listens on this computer only,
so the learner text it shows is not reachable from other computers.

`--task` shows only the named tabs (`gold`, `ratings`, `second-rater`, `silver`,
`preferences`; repeat it for more). `--out-dir` changes the output folder, `--port`
the port. `python -m explain_to_learn.annotation --help` lists the input files of each
task with their defaults.

## How every task works

1. Open the tab and press **Start** (type your name first in the rating tabs). The task
   opens at the first item that has not been saved, so after a break, a closed browser
   or a restart you continue where you stopped.
2. The line above the item shows its number and how many items are saved.
3. **Save and next** saves the item and shows the next one. The whole output file is
   written again after every item, through a temporary file that then replaces the old
   one, so a crash can lose at most the item on screen.
4. **Back** shows the previous item with its saved answers. Change them and press
   **Save and next** to save the change. You cannot skip an item: the next item opens
   only after this one is saved.
5. If something is missing (for example a rating left open without a comment), the app
   says what and does not save.

Keyboard shortcuts act on the open tab:

| Key | Action |
| --- | --- |
| A number | Chooses the option that starts with that number, in the first question of the item that has no answer yet |
| Enter | Save and next |
| b | Back |
| Ctrl+Enter (Cmd+Enter on a Mac) | Save and next, also while typing in a text box |

Numbers, Enter and `b` do nothing while the cursor is in a text box, so they never
replace text. The numbers are the rubric's: in the rating tabs, `0` no, `1` partly and
`2` yes for the four dimensions in order, then `0` no and `1` yes for the invented-rule
flag. In the hand-check `1` is correct and `0` not correct; in the preference tab `1`
and `2` choose an explanation, `3` is a tie and `4` both bad. To change an answer that
is already given, click it.

## a. Gold explanations

The teacher writes a reference explanation and a hint for each of the 300 gold edits
(`data/splits/gold_edit_ids.txt`). The task needs the edit table
(`data/processed/edits.jsonl`, see the README's quick start) and the grammar graph.

```bash
python -m explain_to_learn.annotation --task gold \
    --drafts outputs/R/gold-explain.jsonl --drafts outputs/R/gold-hint.jsonl
```

For each edit the tab shows the sentence with the edit marked in brackets, the corrected
sentence, the ERRANT error type, the learner's level, and the grammar graph concept with
its rule for that level, its example and the concepts it builds on. The concept is
prefilled from the ERRANT mapping (`grammar_graph/errant_map.yaml`); choosing another
one shows its rule. Then:

- write the reference explanation (one or two sentences that name the original and
  corrected words and say why) and the hint (one sentence that points to the error
  without giving the answer), for the learner's level;
- add notes if useful;
- tick **Not explainable by a rule** when no grammar rule explains the correction (for
  example a change of style or content), and say why in the notes. The texts may then
  stay empty.

`--drafts` is optional. It takes run files of `scripts/run_system.py`; each field is
taken from the first file that has it, so an explain run and a hint run together give
both texts. The draft is shown next to the form, and **Fill empty boxes from the model
draft** copies it into the boxes that are still empty. Use drafts from system R, the
reference model, which is not in the blind rating sheet: drafts from systems A to D
would show you outputs that you later rate blind.

**Output:** `gold_explanations.jsonl`, one JSON object per edit in gold-ID order. The
field names follow the silver training rows.

| Field | Content |
| --- | --- |
| `edit_id` | The gold edit |
| `level` | The level the texts are written for; LOCNESS writers (`native`) count as C2, as in the prompts |
| `error_type` | The ERRANT error type, e.g. `R:VERB:SVA` |
| `concept_id`, `concept` | The chosen graph concept and its name (empty if none) |
| `explanation`, `hint` | The reference texts |
| `not_rule_explainable` | `true` if no grammar rule explains the edit |
| `notes` | Free text |
| `from_draft` | `true` if the texts were started from a model draft |

## b. Blind ratings

The first rater rates every item of the blind rating sheet on the rubric in
[eval/rubric.md](../eval/rubric.md). Make the sheet first (it also writes the key and
the second rater's sheet):

```bash
python scripts/make_rating_sheet.py \
    --system A=outputs/A/gold-explain.jsonl --system B=outputs/B/gold-explain.jsonl \
    --system C=outputs/C/gold-explain.jsonl --system D=outputs/D/gold-explain.jsonl
python -m explain_to_learn.annotation --task ratings
```

The rater name defaults to `rater1` (`--rater`). The tab shows the level, the mode, the
sentence, the corrected sentence, the edit and the feedback, with the full rubric in a
panel that opens on click. Rate the four dimensions and the invented-rule flag. Leave a
rating open only if you cannot decide, and say why in the comment; the app asks for the
comment.

**Output:** `ratings_rater1.csv` with the columns `item_id`, `rule_correct`,
`error_located`, `helps_fix`, `fits_level`, `invented_rule` and `comment`, one row per
saved item in sheet order. Ratings are the rubric's words (`no`, `partly`, `yes`). These
are the columns that `scripts/llm_judge.py` writes, so the human and the judge ratings
are read the same way:

```bash
python -m explain_to_learn.human_eval --ratings outputs/annotations/ratings_rater1.csv \
    --key outputs/human_eval/key.csv --out-dir eval/results --name human_eval_rater1
python scripts/agreement.py --first outputs/annotations/ratings_rater1.csv \
    --second outputs/human_eval/ratings_llm_judge.csv --title "Rater 1 vs LLM judge" \
    --out eval/results/agreement_judge.md
```

`ratings_rater1.json`, next to the ratings, records the rater's name, the sheet and a
fingerprint of its items.

### How the rating stays blind

- The rating tabs read only the sheet's own columns: item ID, level, mode, sentence,
  corrected sentence, edit and feedback. A file with any other column, such as the
  key's `system`, is refused, and the app never opens the key.
- `scripts/make_rating_sheet.py` shuffles the items with a seed and numbers them in the
  shuffled order, so neither the order nor the item ID says which system wrote an item.
- Nothing the app shows or saves contains a system name. The tests check this on every
  item of a sheet.
- The key is needed only after rating, to join the ratings with the systems.

If the sheet is made again after rating has started (`--force`), its item IDs may point
to different items. The app compares the sheet with the fingerprint in the `.json` file
and refuses to continue, rather than attach old ratings to new items. Rate a new sheet
under another rater name, or move the old files away.

## c. Second rater

The second rater rates the 50-item subset sheet on their own computer. They need the
repository, Python 3.11 or newer, and the file `rating_sheet_second_rater.csv` sent by
the first rater. The corpus, the grammar graph and a GPU are not needed.

```bash
pip install -e ".[demo]"
python -m explain_to_learn.annotation --task second-rater \
    --second-sheet rating_sheet_second_rater.csv
```

Open <http://127.0.0.1:7861>, type your name in **Rater name** and press **Start**.
Before rating, rate the worked examples in the rubric without looking at their answers,
then compare (rubric, "Procedure"). Rate as in [Blind ratings](#b-blind-ratings), on
your own, without discussing items with the first rater.

When you have finished, send back `outputs/annotations/ratings_<your name>.csv` and the
`.json` file next to it. Then delete the sheet and the ratings from your computer: they
quote learner essays, which may only be used for research under the corpus licence.

The first rater compares the two sets of ratings on the items both rated:

```bash
python scripts/agreement.py --first outputs/annotations/ratings_rater1.csv \
    --second outputs/annotations/ratings_rater2.csv --title "Rater 1 vs rater 2" \
    --out eval/results/agreement_raters.md
```

## d. Silver hand-check

The teacher marks a random sample of silver training items correct or not. The filter
step of `scripts/make_silver.py` writes the sample to `outputs/silver_handcheck.csv`.

```bash
python scripts/make_silver.py filter
python -m explain_to_learn.annotation --task silver
```

The tab shows the level, the mode, the error type, the sentence, the edit, the concept
and the explanation or hint. Mark an item **correct** if you would give this feedback
to a learner at this level as it is: the rule is true and is the reason for this
correction, the concept fits, and the text does what its mode asks (an explanation
names the words and says why; a hint points to the error without giving the answer).
Otherwise mark it **not correct** and, if useful, say why in the note. The progress
line shows the accuracy so far with its 95% Wilson interval.

**Output:** `silver_handcheck.csv`: the columns of the hand-check file plus `note`, with
`correct` set to `1` or `0`, one row per checked item. `silver.handcheck_accuracy`
reads it:

```bash
python scripts/make_silver.py score --handcheck outputs/annotations/silver_handcheck.csv
```

The marks are kept in their own file because the filter step rewrites
`outputs/silver_handcheck.csv` each time it runs. If an item that was already marked
has a different text in the new file, or is no longer in it, the app stops instead of
keeping a mark for a text you never saw.

## e. Preference pairs

The teacher chooses the better of two outputs for the same edit, for preference
tuning. The two outputs come from two run files of `scripts/run_system.py`: two
systems, or two sampled runs of one system. Give each a name:

```bash
python -m explain_to_learn.annotation --task preferences \
    --pair D-1=outputs/D/train-sample-1.jsonl --pair D-2=outputs/D/train-sample-2.jsonl
```

An edit is shown when both files have readable output for it in the same mode and the
two texts differ. The tab shows the sentence with the edit marked and the two outputs
side by side as **Explanation 1** and **Explanation 2**, without their names. Which one
is on the left is decided per edit by a seeded coin flip (`--seed`), so position and
source are not linked. Choose **1** or **2**, **Tie** (both equally good) or **Both
bad**.

**Output:** `preferences.jsonl`, one JSON object per choice:

| Field | Content |
| --- | --- |
| `edit_id`, `level`, `mode` | The edit, and the level and mode the outputs were written for |
| `prompt_messages` | The chat messages the model got, rebuilt with `generate.edit_item` (with the graph snippet if either source was grounded for this edit) |
| `chosen`, `rejected` | The better and the other output as the JSON string the model is trained to write (`prompts.build_target`, with the field that the mode does not use left empty, as in the fine-tuning targets); both `null` for a tie or when both are bad |
| `source_systems` | The two source names, sorted |
| `judgement` | `first` or `second` (the better source, by its position in `source_systems`), `tie` or `both_bad` |

Each record with a `chosen` and a `rejected` output is one preference pair, with
`prompt_messages` as its prompt. Ties and pairs where both are bad are kept for the
record and left out of training.

## Writing the grammar graph

`scripts/concept_examples.py` helps the teacher write `grammar_graph/concepts.yaml`.
For each concept it writes `outputs/concept_examples/<concept>.md` with the ERRANT error
types mapped to the concept, their edit counts per CEFR level, and random example edits
from the training split, with the edit marked. `index.md` lists the concepts by number
of edits.

```bash
python scripts/concept_examples.py --n 10
```

The examples come from training essays only, so the graph is not written from the
development edits that the gold set is drawn from. They show which errors a concept has
to explain; they are not material for the graph. The `example` of each concept must be
a sentence the teacher writes, never one copied or adapted from the corpus.

## Advice for raters

- Rate the items in the order given. The order is random, and the app does not let you
  skip ahead.
- Take a break every 50 items or so; rating quality drops when you are tired. You can
  close the browser at any time: everything up to the last saved item is kept.
- Do not open the key (`outputs/human_eval/key.csv`), other raters' files or the model
  output files while rating. If you see the key by accident, note it in a comment.
- Rate each dimension on its own, against the rubric, and not against your own idea of
  the best explanation, even for edits you wrote a reference explanation for.
- The rubric is in the panel below each item. Re-read the decision rules when an item
  is hard to rate.
- Do not edit the output files by hand or open them in a spreadsheet program while the
  app is running; the app rewrites them after every item. To keep a copy, copy the
  whole `outputs/annotations/` folder.
