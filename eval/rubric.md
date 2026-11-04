# Rating rubric for grammar feedback

This rubric is used for the blind human ratings and, unchanged, as the instructions for
the LLM judge (`scripts/llm_judge.py`).

## What you rate

Each item is the feedback that one system wrote for one edit in a learner's sentence.
The rating sheet shows:

| Column | Meaning |
| --- | --- |
| `level` | The learner's CEFR level (A2, B1 or B2) that the feedback was written for |
| `sentence` | The learner's original sentence |
| `corrected` | The corrected sentence |
| `edit` | The words that changed, as `original -> correction` (`-` means nothing) |
| `explanation` | The system's explanation of the edit |
| `hint` | The system's hint, meant to help the learner find the correction alone |

You rate the explanation and the hint together, as one piece of feedback. The system
names are hidden and the items are in random order.

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
   `comment`, and rate the feedback as an explanation of the correction shown.
4. If the feedback is empty, unreadable or about a different edit, rate all four
   dimensions `no` and the flag `no`.
5. Judge rules against standard descriptions of English grammar, such as a learner's
   grammar reference. Where usage varies (both forms are accepted), a rule that
   describes one common usage is correct.
6. Do not look at other ratings or the key while rating. Take a break every 50 items or
   so; rating quality drops when tired.

## Dimensions

The examples are invented. All use this item unless stated otherwise:

> **Level** A2 · **Sentence** *She don't like coffee .* · **Corrected** *She doesn't like
> coffee .* · **Edit** `don't -> doesn't`

### 1. Rule correct (`rule_correct`)

*Is the grammar the feedback states true, and does it explain this edit?*

- **yes**: Every grammar statement is correct and the main rule explains why the
  correction is needed.
- **partly**: The main rule points in the right direction but is incomplete,
  over-general or imprecise in a way that could mislead in other sentences, or it is
  correct but the feedback also contains a smaller wrong statement. Also `partly` when
  the feedback gives only the correct form and states no rule at all.
- **no**: The main rule is wrong, or it explains a different kind of error than the one
  corrected.

| Rating | Feedback |
| --- | --- |
| yes | "With *he*, *she* or *it*, use *doesn't*, not *don't*." |
| partly | "With one person, always add *-s* to the verb." (Over-general: *she can*, *she must*.) |
| partly | "Write *doesn't* here." (Correct form, no rule.) |
| no | "Use *doesn't* because the sentence is about the past." |

### 2. Error located (`error_located`)

*Does the feedback show the learner where the error is?*

- **yes**: It quotes or names the original words or their correction, or points to them
  so that there is only one place they can be ("the verb after *she*").
- **partly**: It points to the right area but not clearly enough to find the words
  ("check your verbs" in a sentence with more than one verb), or names only part of a
  longer edit.
- **no**: It does not point to any place, or it points to the wrong words.

| Rating | Feedback |
| --- | --- |
| yes | "*don't* should be *doesn't* after *she*." |
| partly | "Look at the verbs in this sentence." (Two verbs: *don't*, *like*.) |
| no | "Check the word *coffee*." |

### 3. Helps the learner fix it (`helps_fix`)

*Would a learner at this level know what to do after reading the feedback?*

- **yes**: The learner can make the correction, and could avoid the error next time,
  using the explanation and the hint.
- **partly**: The feedback gives a useful direction but leaves an important step open
  (it says what is wrong but not what to use instead, or the hint is too vague to act
  on without the explanation).
- **no**: There is nothing the learner can act on, or following the feedback would lead
  to another wrong form.

Giving the answer is allowed; it does not lower this rating.

| Rating | Feedback |
| --- | --- |
| yes | Explanation: "With *she*, use *doesn't*." Hint: "Which form of *do* goes with *she*?" |
| partly | "There is a problem with the verb." (No direction about what to change.) |
| no | "Use *didn't* after *she*." (Leads to a wrong correction.) |

### 4. Fits the level (`fits_level`)

*Is the feedback written so that a learner at this level can understand it?*

- **yes**: Words and sentences suit the level. Grammar terms are ones learners meet at
  this level, or are explained. One or two short sentences.
- **partly**: Mostly suitable, but one term or structure is above the level, or the
  feedback is much longer than needed, or it is so simplified that a B2 learner would
  find it unhelpful.
- **no**: Clearly too hard for the level (dense terminology, long complex sentences), or
  unsuited in another clear way.

Rough guide to terms by level: A2: *verb, noun, past, plural, question*; B1: *present
perfect, countable, comparative, preposition*; B2: *relative clause, passive, modal
verb, conditional*.

| Rating | Feedback |
| --- | --- |
| yes | "With *she*, use *doesn't*. *She doesn't like*, *he doesn't know*." |
| partly | "*She* is third person singular, so use *doesn't*." (One term above A2, otherwise clear.) |
| no | "The auxiliary must agree in person and number with the subject in the present simple, so the third-person singular inflected form is required." |

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
2. For each item, read the sentence, the correction and the edit first, then the
   feedback. Rate the four dimensions in order, then the flag.
3. Leave a cell empty only if you cannot decide, and say why in `comment`. Empty cells
   are left out of the averages and the agreement statistics.
4. The second rater rates the 50-item subset sheet independently, without discussing
   items with the first rater.
