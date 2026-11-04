# Rating rubric for grammar feedback

This rubric is used for the blind human ratings and, unchanged, as the instructions for
the LLM judge (`scripts/llm_judge.py`).

## What you rate

Each item is the feedback that one system wrote for one edit in a learner's sentence.
The rating sheet shows:

| Column | Meaning |
| --- | --- |
| `level` | The learner's CEFR level (A1 to C2) that the feedback was written for |
| `mode` | `explain` or `hint` (see below) |
| `sentence` | The learner's original sentence |
| `corrected` | The corrected sentence |
| `edit` | The words that changed, as `original -> correction` (`-` means nothing) |
| `feedback` | What the learner sees: an explanation or a hint |

- In **explain** mode the feedback is an explanation: it should name the original and
  corrected words and say why the correction is right.
- In **hint** mode the feedback is a hint: it should point the learner to the error and
  the rule without giving the corrected words, so that the learner can fix it alone.

The system names are hidden and the items are in random order.

## Scale

Four dimensions use a three-point scale. Write the word or its number:

| Word | Number | Meaning |
| --- | --- | --- |
| `no` | 0 | The feedback fails on this dimension. |
| `partly` | 1 | The feedback does part of the job, or does it with a clear flaw. |
| `yes` | 2 | The feedback does this well. |

The flag `invented_rule` is `yes` or `no`. The `comment` column is optional.

## Origin

The dimensions adapt the scheme of the BEA 2025 shared task on the pedagogical ability
of AI tutors ([Kochmar et al., 2025](https://arxiv.org/abs/2507.10579)). That task rates
tutor responses in mathematics dialogues on a three-point scale ("No", "To some extent",
"Yes") for mistake identification, mistake location, providing guidance and
actionability. Here the feedback is written, not part of a dialogue, and the error is
already known from the correction, so the dimensions change as follows:

| This rubric | BEA 2025 dimension |
| --- | --- |
| Rule correct | Mistake identification (here: is the error explained correctly) |
| Error located | Mistake location |
| Helps the learner fix it | Providing guidance and actionability, merged |
| Fits the level | New: suitability for the learner's CEFR level |
| Invented rule (flag) | New: a rule that does not exist in English |

## General decision rules

1. Rate what is written, not what the system probably meant.
2. Rate each dimension on its own. A wrong rule can still point to the right words.
3. The correction is given and is not rated. If you think it is wrong, say so in
   `comment`, and rate the feedback as feedback on the correction shown.
4. If the feedback is empty, unreadable or about a different edit, rate all four
   dimensions `no` and the flag `no`.
5. Judge rules against standard descriptions of English grammar, such as a learner's
   grammar reference. Where usage varies (both forms are accepted), a rule that
   describes one common usage is correct.
6. Do not look at other ratings or the key while rating. Take a break every 50 items or
   so; rating quality drops when tired.

## Dimensions

The examples are invented. All use this edit unless stated otherwise:

> **Level** A2 · **Sentence** *She don't like coffee.* · **Corrected** *She doesn't like
> coffee.* · **Edit** `do -> does`

### 1. Rule correct (`rule_correct`)

*Is the grammar the feedback states true, and does it fit this edit?*

- **yes**: Every grammar statement is correct, and the rule given (or, in a hint, the
  rule pointed to) is the reason the correction is needed.
- **partly**: The rule points in the right direction but is incomplete, over-general or
  imprecise in a way that could mislead in other sentences; or it is correct but the
  feedback also contains a smaller wrong statement; or the feedback says nothing about
  the grammar at all (only the correct form, or only where to look).
- **no**: The main rule is wrong, or it is about a different kind of error than the one
  corrected.

| Mode | Rating | Feedback |
| --- | --- | --- |
| explain | yes | "After *she*, use *does*, not *do*: *she doesn't like*." |
| explain | partly | "With one person, always add *-s* to the verb." (Over-general: *she can*, *she must*.) |
| explain | partly | "Write *doesn't* here." (Correct form, nothing about the grammar.) |
| explain | no | "Use *doesn't* because the sentence is about the past." |
| hint | yes | "Which form of *do* goes with *he*, *she* and *it*?" |

### 2. Error located (`error_located`)

*Does the feedback show the learner where the error is?*

- **yes**: It quotes or names the original words or their correction, or points to them
  so that there is only one place they can be ("the verb after *she*").
- **partly**: It points to the right area but not clearly enough to find the words
  ("check your verbs" in a sentence with more than one verb), or names only part of a
  longer edit.
- **no**: It does not point to any place, or it points to the wrong words.

| Mode | Rating | Feedback |
| --- | --- | --- |
| explain | yes | "*don't* should be *doesn't* after *she*." |
| hint | partly | "Look at the verbs in this sentence." (Two verbs: *do*, *like*.) |
| hint | no | "Check the word *coffee*." |

### 3. Helps the learner fix it (`helps_fix`)

*Would a learner at this level know what to do after reading the feedback?*

- **yes**: The learner can make the correction, and could avoid the error next time.
  In hint mode: the learner can find the correction from the hint alone.
- **partly**: The feedback gives a useful direction but leaves an important step open
  (it says what is wrong but not what to use instead, or the hint is too vague to act
  on). In hint mode, also `partly` when the hint gives away the corrected words: the
  learner can copy the answer but does not have to work it out.
- **no**: There is nothing the learner can act on, or following the feedback would lead
  to another wrong form.

In explain mode, giving the correction is expected and does not lower this rating.

| Mode | Rating | Feedback |
| --- | --- | --- |
| explain | yes | "After *she*, use *does*, not *do*: *she doesn't like*." |
| hint | yes | "Which form of *do* goes with *she*?" |
| hint | partly | "Use *doesn't* here." (Gives the answer away.) |
| explain | partly | "There is a problem with the verb." (No direction about what to change.) |
| explain | no | "Use *didn't* after *she*." (Leads to a wrong form.) |

### 4. Fits the level (`fits_level`)

*Is the feedback written so that a learner at this level can understand it?*

- **yes**: Words and sentences suit the level, and grammar terms are ones learners meet
  at this level or are explained. Explanations take one or two short sentences, hints
  one.
- **partly**: Mostly suitable, but one term or structure is above the level, or the
  feedback is much longer than needed, or it is so simplified that a C1 or C2 learner
  would find it unhelpful.
- **no**: Clearly too hard for the level (dense terminology, long complex sentences), or
  unsuited in another clear way.

Guide by level, as given to the systems: A1 and A2, short sentences, everyday words
and very few grammar terms; B1 and B2, common grammar terms (*present perfect*,
*countable noun*, *preposition*); C1 and C2, precise grammar terms are fine.

| Mode | Rating | Feedback |
| --- | --- | --- |
| explain | yes | "After *she*, use *does*, not *do*: *she doesn't like*." |
| explain | partly | "*She* is third person singular, so use *doesn't*." (One term above A2, otherwise clear.) |
| explain | no | "The auxiliary must agree in person and number with the subject in the present simple, so the third-person singular form is required." |

### Flag: invented rule (`invented_rule`)

*Does the feedback state a grammar rule that does not exist in English?*

- **yes**: The feedback states a rule that is false in general, not just for this
  sentence. For example: "*doesn't* is used only in questions", or "adjectives always
  come after the noun in English".
- **no**: All stated rules exist in English, even if one is applied to the wrong case,
  stated too broadly, or not stated at all.

A real rule applied to the wrong edit lowers **rule correct** but is not an invented
rule. An invented rule is always a reason to rate **rule correct** `no` (if it is the
main rule) or `partly` (if it is a side remark).

## Procedure

1. Before rating the sheet, rate the examples above without looking at the answers, then
   compare. Discuss any disagreement with the other rater before either starts.
2. For each item, read the level, the mode, the sentence, the correction and the edit
   first, then the feedback. Rate the four dimensions in order, then the flag.
3. Leave a cell empty only if you cannot decide, and say why in `comment`. Empty cells
   are left out of the averages and the agreement statistics.
4. The second rater rates the 50-item subset sheet independently, without discussing
   items with the first rater.
