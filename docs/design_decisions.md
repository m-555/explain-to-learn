# Design decisions

This document explains the main choices built into the code, and why they were made.
Each section says what the code does, why, and where to find it. It describes the method
only; results are reported elsewhere.

## Contents

1. [Gold corrections, so that explanation quality is measured on its own](#1-gold-corrections-so-that-explanation-quality-is-measured-on-its-own)
2. [Aligning the official M2 files to the essays instead of re-running ERRANT](#2-aligning-the-official-m2-files-to-the-essays-instead-of-re-running-errant)
3. [Splitting by essay](#3-splitting-by-essay)
4. [FCE essays are labelled B2](#4-fce-essays-are-labelled-b2)
5. [A stratified gold sample](#5-a-stratified-gold-sample)
6. [F0.5: precision counts more than recall](#6-f05-precision-counts-more-than-recall)
7. [Cohen's kappa, and weighted kappa for the three-point scales](#7-cohens-kappa-and-weighted-kappa-for-the-three-point-scales)
8. [Paired bootstrap intervals](#8-paired-bootstrap-intervals)
9. [LoRA rank and alpha](#9-lora-rank-and-alpha)
10. [Filtering the silver data](#10-filtering-the-silver-data)
11. [Checking the LLM judge against human ratings](#11-checking-the-llm-judge-against-human-ratings)
12. [Comparing with GEE](#12-comparing-with-gee)
13. [Preference tuning with DPO](#13-preference-tuning-with-dpo)

## 1. Gold corrections, so that explanation quality is measured on its own

**What the code does.** Training data and the main evaluation start from the corrections
written by the corpus annotators, not from a correction model. Every system gets the
same sentence and the same edit, and is scored only on what it says about that edit
(`src/explain_to_learn/edits.py`, `scripts/run_system.py`). Only the end-to-end demo adds
a correction model in front (`demo/pipeline.py`), and its correction quality is measured
separately with ERRANT's F0.5 (section 6).

**Why.** If a system had to correct the sentence first, a weak explanation and a wrong
correction would look the same in the scores. With gold corrections every system
explains the same edits, so differences between systems come from the explanations. The
same choice is made in GEE, which explains the edits between a given sentence and its
correction (Song et al., 2024). The cost is that the evaluation does not show how
correction errors carry through to the explanations in real use; the README lists this
as a limitation.

## 2. Aligning the official M2 files to the essays instead of re-running ERRANT

**What the code does.** The BEA-2019 releases come in two formats: JSON with one essay
per line (essay ID, CEFR level, text) and M2 with typed ERRANT edits per sentence but no
essay IDs. `align_essays` in `src/explain_to_learn/edits.py` walks through both files in
order and gives each essay its consecutive M2 sentences, checking that their characters
match the essay's text exactly (after the punctuation normalisation of the official
`json_to_m2.py` script). Any mismatch stops the build with an error; nothing is guessed.

**Why.** The project splits and samples by essay and needs each edit's essay and level.
Re-running ERRANT on the JSON text would provide that, but with a current spaCy model
the sentence splits, tokens and some error types would differ from the official gold
annotation that other work reports on (Bryant et al., 2017; Bryant et al., 2019). The
alignment keeps the official edits and types unchanged, needs no spaCy model, and gives
every edit a stable ID (`{corpus}-{essay}-{sentence}-{edit}`) that depends only on the
official files. That is why the repository can publish edit IDs instead of learner text.

## 3. Splitting by essay

**What the code does.** Train and development data follow the official BEA-2019 splits,
which are by essay, and `essay_splits` in `src/explain_to_learn/splits.py` checks that no
essay is in two splits. The fine-tuning validation set (`train_data.py`) and the
preference validation set (`preference.py`) also hold out whole essays, chosen with a
fixed seed.

**Why.** Edits of one essay share a writer, a topic, a level and often whole sentences.
If some of them were in training and others in evaluation, a model could score well by
remembering the writer's sentences and habits rather than by explaining grammar, and the
evaluation would overstate how well it generalises to new learners.

## 4. FCE essays are labelled B2

**What the code does.** W&I essays carry an individual CEFR rating, which is kept without
its sub-level (`A2.ii` becomes `A2`). FCE essays have no individual rating, so
`edits.py` labels every FCE essay `B2`. LOCNESS essays, written by native speakers, are
labelled `native`, and the prompts treat them as C2 (`prompts.prompt_level`).

**Why.** FCE essays are answers to the First Certificate in English, an exam set at
CEFR B2, so B2 is the level the writers were preparing for. It is the exam's target
level, not a rating of each writer, and the code and data card say so. To keep this
approximation out of the evaluation, the gold set is drawn only from W&I+LOCNESS, where
levels are individual (section 5); FCE is used for training edits.

## 5. A stratified gold sample

**What the code does.** `sample_gold` in `src/explain_to_learn/splits.py` draws 300 edits
from the W&I+LOCNESS development set, stratified by error category and CEFR level. The
300 draws are shared between the category-level strata in proportion to their sizes; each
stratum gets the whole part of its share, and the draws left over go to the strata with
the largest remainders (`allocate`). The seed is fixed (2019), so the same edit table
always gives the same 300 IDs (`data/splits/gold_edit_ids.txt`).

**Why.** A simple random sample of 300 could, by chance, miss a level or an error type, or
over-represent one, and the results are broken down by both. Proportional stratified
sampling keeps the sample's mix of categories and levels close to the population's, so
the overall scores stay representative while each breakdown has its share of edits
(Cochran, 1977). Rare combinations can still get no edit, and a breakdown with few edits
has wide intervals; the report's figures leave out groups below a minimum size.

## 6. F0.5: precision counts more than recall

**What the code does.** For the end-to-end demo pipeline, `metrics.correction_f05` scores
the correction model with ERRANT's span-based F0.5 (`errant_compare`), from true
positives, false positives and false negatives against the reference edits.

**Why.** F-beta combines precision P and recall R as
(1 + beta²)·P·R / (beta²·P + R); with beta = 0.5, precision weighs twice as much as
recall. For learners, a false correction (changing something that was right) does more
harm than a missed one: it teaches a wrong rule, and in this project it would also be
followed by a confident explanation of an error that does not exist. F0.5 has been the
standard measure for grammatical error correction since the CoNLL-2014 shared task
(Ng et al., 2014) and was used in the BEA-2019 shared task (Bryant et al., 2019), so the
numbers are also comparable with published systems.

## 7. Cohen's kappa, and weighted kappa for the three-point scales

**What the code does.** `scripts/agreement.py` compares two sets of ratings of the same
items (the second rater with the first, and the LLM judge with the first) and reports
per dimension: the number of items both rated, the share of exact agreement, Cohen's
kappa and, for the four three-point dimensions (no / partly / yes), quadratic-weighted
kappa. The invented-rule flag is yes/no, so it gets plain kappa only.

**Why.** Raw agreement looks high when most items get the same label anyway; kappa
subtracts the agreement two raters would reach by chance with the same label
frequencies (Cohen, 1960). On an ordered scale, "partly" against "yes" is a smaller
disagreement than "no" against "yes". Weighted kappa gives partial credit for near
misses (Cohen, 1968); with quadratic weights, ((i - j) / (k - 1))², a no-vs-yes
disagreement counts four times as much as an off-by-one one, and the statistic is close
to the intraclass correlation (Fleiss and Cohen, 1973). Exact agreement is reported next
to kappa because kappa can be low despite high agreement when one label dominates
(Feinstein and Cicchetti, 1990), which is likely for a flag that is rarely "yes".

## 8. Paired bootstrap intervals

**What the code does.** Every system is scored on the same gold edits.
`metrics.paired_bootstrap` resamples those edits with replacement (10,000 times, fixed
seed) and, in each resample, takes the mean score of both systems on the same edits; the
2.5th and 97.5th percentiles of the differences give the 95% interval, and a two-sided
p-value is reported too. `scripts/compute_metrics.py` uses it for the comparisons B-A,
C-A, D-B and D-C, and `scripts/make_report_assets.py` uses it for the same comparisons
on the human ratings, pairing the four systems' ratings of each edit.

The results table also gives each system's own 95% interval. For the human ratings it is
a percentile bootstrap over the rated items. The automatic metrics are 0/1 per edit, and
resampling n edits of which a share p score 1 gives a count that follows Binomial(n, p),
so the same bootstrap interval is computed from the mean and the number of edits in the
metrics file, without the per-edit scores.

**Why.** A few hundred edits leave real sampling uncertainty, and some edits are harder
than others for every system. Pairing removes the variation between edits from the
comparison: the question is whether system D does better than system B on the same
edits, not on edits of different difficulty. The bootstrap needs no assumption about the
distribution of scores, which are 0/1 or on a three-point scale here (Efron and
Tibshirani, 1993). Paired bootstrap resampling is a common significance test in NLP
(Koehn, 2004; Berg-Kirkpatrick et al., 2012; Dror et al., 2018). Intervals are reported
rather than only p-values, so the size of a difference stays visible.

## 9. LoRA rank and alpha

**What the code does.** `train/lora_config.yaml` trains a LoRA adapter (Hu et al., 2021)
of rank 16 with alpha 32 and dropout 0.05 on every attention projection (q, k, v, o) and
every MLP projection (gate, up, down) of the frozen bf16 base model. Variants train rank
8 and rank 32 with alpha always twice the rank, and rank 16 on a 4-bit base model
(QLoRA). Systems C and D use the same settings and data and differ only in whether the
prompt contains the grammar graph rule.

**Why.** LoRA adds a low-rank update B·A to each frozen weight matrix and scales it by
alpha / r. The task is narrow (one short JSON answer in a fixed format), and the original
paper found that small ranks, even 1 to 8, are often enough to adapt a large model to one
task. Rank 16 leaves some margin, and the rank 8 and rank 32 variants test the choice in
both directions. Keeping
alpha at twice the rank keeps the scale alpha / r = 2 fixed, so the variants differ in
capacity and not in the size of the update. Adapting all linear layers rather than only
the attention projections follows the QLoRA finding that this is needed to match full
fine-tuning (Dettmers et al., 2023). The base model stays in bf16 because a 4B model fits
easily on the 32 GB GPU; the QLoRA variant measures what 4-bit quantisation would cost.

## 10. Filtering the silver data

**What the code does.** The training targets for systems C and D are silver
explanations: the teacher model (system R, prompted with the grammar graph rule) writes
an explanation or a hint for a seeded sample of training edits
(`scripts/make_silver.py`; gold edits and error types without a graph concept are
excluded, and each edit is put in hint mode with probability 0.25).
`silver.filter_silver` keeps an output only if it parses as JSON, the text for its mode
is not empty, it names the edit's graph concept, and it names the edited words as the
mode requires: an explanation
names the original and corrected words, a hint locates the error without giving the
correction away (`metrics.faithfulness`). The first failed check is recorded as the
rejection reason. A random sample of 200 kept items is then marked correct or not by a
teacher, and the share marked correct is reported with a Wilson interval (Wilson, 1927).

**Why.** Student models learn whatever is in their targets, including format errors and
explanations of the wrong edit. The automatic checks are cheap and remove outputs that
are certainly unusable, and they use the same definitions as the evaluation metrics, so
training and evaluation agree on what a well-formed answer is. They check form, not
whether the grammar is right; the teacher's hand-check estimates how often the kept data
is actually correct. The rejection counts show what the filter removes, so a reader can
see whether it changes the mix of the data.

## 11. Checking the LLM judge against human ratings

**What the code does.** `scripts/llm_judge.py` rates the same blind items as the teachers,
with the rubric (`eval/rubric.md`) as its instructions, at temperature 0, and writes the
same rating columns as a human sheet. `scripts/agreement.py` then compares the judge
with the first rater on all items, exactly as the second rater is compared with the
first. `scripts/make_report_assets.py` draws both agreements side by side and also
compares each system's mean rating from the judge and from the first rater.

**Why.** An LLM judge is cheap and consistent, but it can be biased in ways a teacher is
not, for example towards longer answers or towards its own answers (Zheng et al., 2023).
That matters here: systems C and D learn from a model's explanations, so a judge from the
same model family could favour them. The second rater's agreement with the first rater
is the reference point for reading the judge's agreement: the judge can stand in for a
teacher only on dimensions where it agrees with the first rater about as well as a second
teacher does. The comparison of system means checks the use that matters most, whether
the judge ranks the systems as a teacher does.

## 12. Comparing with GEE

**What the code does.** `src/explain_to_learn/gee.py` reads the English explanations of
GEE (Song et al., 2024) from the GEE repository, which `scripts/download_data.sh` clones
into `data/raw/gee/`. Each explanation is paired with its atomic edit, and each atomic
edit with the official W&I+LOCNESS edit of the same sentence that has the same original
and corrected words. Atomic edits that GEE and ERRANT cut into different spans have no
counterpart and are dropped and counted. `scripts/evaluate_on_gee.py` runs the systems on
the matched edits at each learner's level and scores them and GEE's explanations with
coverage and faithfulness, the two automatic metrics that apply to both. Concept accuracy
does not apply because GEE labels edits with its own free-text error types, and level
fit does not apply because GEE's explanations were not written for a learner level. A
sheet with GEE's and every system's explanation side by side is written for a teacher to
compare by hand.

**Why.** The English GEE sentences come from the W&I+LOCNESS development set, so the
official edits, error types and learner levels are already in the edit table; matching
to them keeps the comparison on the same annotation as the rest of the project. Only
edits on which both annotations agree are compared, so neither side is scored on an edit
it did not explain. GEE's explanations were written by its GPT-4 pipeline and judged by
one annotator, so they are a comparison point, not a gold standard; the annotator's flags
are carried into the side-by-side sheet.

**Licence.** The GEE repository has no licence file. Its data is therefore never
committed to this repository: it stays in the git-ignored `data/raw/`, everything derived
from it that quotes text goes to the git-ignored `outputs/`, and only counts and shares
are published. The test fixture in GEE's format is invented.

## 13. Preference tuning with DPO

**What the code does.** As a stretch goal, `src/explain_to_learn/preference.py` turns the
teachers' pairwise judgements (which of two outputs for the same prompt is better) into
preference pairs; ties and "both bad" judgements say nothing about which output is better
and are skipped. `train/train_dpo.py` continues training the system D adapter with direct
preference optimisation (Rafailov et al., 2023), with a frozen copy of the D adapter as
the reference model (`train/dpo_config.yaml`: beta 0.1, learning rate 5e-6).

**Why.** Supervised fine-tuning imitates the silver explanations; preference pairs say
directly which of two explanations a teacher finds better, including differences the
automatic filter cannot see. DPO learns from such pairs without training a separate
reward model. Using system D as the reference keeps the tuned model close to D, so any
change can be compared with D on the same evaluation. The learning rate is far below the
SFT rate because DPO only adjusts an adapter that is already tuned.

## References

- Berg-Kirkpatrick, T., Burkett, D. and Klein, D. (2012). An empirical investigation of
  statistical significance in NLP. EMNLP-CoNLL.
- Bryant, C., Felice, M. and Briscoe, T. (2017). Automatic annotation and evaluation of
  error types for grammatical error correction. ACL.
- Bryant, C., Felice, M., Andersen, Ø. E. and Briscoe, T. (2019). The BEA-2019 shared
  task on grammatical error correction. BEA Workshop.
- Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
  Psychological Measurement*, 20(1), 37-46.
- Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled
  disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.
- Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023). QLoRA: Efficient
  finetuning of quantized LLMs. NeurIPS.
- Dror, R., Baumer, G., Shlomov, S. and Reichart, R. (2018). The hitchhiker's guide to
  testing statistical significance in natural language processing. ACL.
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman &
  Hall.
- Feinstein, A. R. and Cicchetti, D. V. (1990). High agreement but low kappa: I. The
  problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543-549.
- Fleiss, J. L. and Cohen, J. (1973). The equivalence of weighted kappa and the
  intraclass correlation coefficient as measures of reliability. *Educational and
  Psychological Measurement*, 33(3), 613-619.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W.
  (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685
  (ICLR 2022).
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation.
  EMNLP.
- Ng, H. T., Wu, S. M., Briscoe, T., Hadiwinoto, C., Susanto, R. H. and Bryant, C.
  (2014). The CoNLL-2014 shared task on grammatical error correction. CoNLL.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. and Finn, C. (2023).
  Direct preference optimization: Your language model is secretly a reward model.
  NeurIPS.
- Song, Y., Krishna, K., Bhatt, R., Gimpel, K. and Iyyer, M. (2024). GEE! Grammar error
  explanation with large language models. Findings of NAACL.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
  inference. *Journal of the American Statistical Association*, 22(158), 209-212.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z.,
  Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E. and Stoica, I. (2023). Judging
  LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS Datasets and Benchmarks.
