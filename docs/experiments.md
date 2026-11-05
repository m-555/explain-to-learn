# Experiment runbook

The complete sequence of runs, from a fresh machine to the v1.0 release, on one NVIDIA
RTX 5090 (32 GB) in a Windows PC, with the code in WSL2 Ubuntu and model servers in
Docker. Every command runs from the repository root in WSL2 with the virtual
environment active, unless a step says otherwise.

The times are rough planning estimates for this GPU, not measurements. Steps marked
*teacher time* are human work.

## What is committed

Learner text never enters the repository. Every step writes learner text (sentences,
model outputs, rating sheets, keys) to the git-ignored `data/processed/` or `outputs/`
folders, and commits only IDs and aggregate tables, mostly in `eval/results/`. Before
each commit, check that nothing from those folders is staged:

```bash
git status --short && git ls-files data/raw data/processed outputs
```

The second command must print nothing.

## Overview

| Step | What | Time | Committed |
| --- | --- | --- | --- |
| [1](#1-environment) | Environment | 1-2 h the first time | nothing |
| [2](#2-data-and-the-edit-table) | Data and the edit table | 5 min | nothing new (splits and statistics are already committed) |
| [3](#3-grammar-graph) | Grammar graph content and validation | days of teacher time; checks take seconds | `grammar_graph/`, the graph figure |
| [4](#4-baselines-a-and-b-on-a-dev-sample) | Baselines A and B on a dev sample | 30-60 min | `eval/results/metrics_dev-*.md`, `.json` |
| [5](#5-silver-data-from-the-teacher) | Silver data from the teacher (system R) | 30-90 min | nothing yet |
| [6](#6-silver-filter-and-the-200-item-hand-check) | Silver filter and hand-check | seconds; 3 h teacher time | `eval/results/silver_filter.*`, `silver_handcheck.json` |
| [7](#7-lora-adapters-for-c-and-d) | LoRA adapters for C and D, four variants | 15-40 min per adapter | the chosen adapter paths in `configs/systems.yaml` |
| [8](#8-gold-set-generation-for-a-b-c-d-and-r) | Gold-set generation for A, B, C, D and R | 1-2 h | nothing yet |
| [9](#9-automatic-metrics) | Automatic metrics | 2 min | `eval/results/metrics_gold-*.md`, `.json` |
| [10](#10-rating-sheets-and-ratings) | Rating sheets and ratings | seconds; 8-10 h teacher time | `eval/results/human_eval_*` |
| [11](#11-agreement-and-the-llm-judge) | Agreement and the LLM judge | 30 min | `eval/results/agreement_*.md`, `human_eval_llm_judge*` |
| [12](#12-correction-f05) | Correction F0.5 of the end-to-end pipeline | 15-30 min | `eval/results/correction_*.md`, `.json` |
| [13](#13-analysis-assets) | Analysis assets | minutes | report tables and figures |
| [14](#14-demo-and-hugging-face-space) | Demo and Hugging Face Space | 1 h | nothing (the Space is its own repository) |
| [15](#15-release-checklist) | Release checklist and tag v1.0 | 2-4 h | cards, README, report, tag |

## 1. Environment

Follow [gpu_setup.md](gpu_setup.md) for the Windows driver, WSL2 Ubuntu 24.04 and
Docker with GPU access (needed for the vLLM servers). Then create the environment. The
PyTorch build must be for CUDA 12.8 or newer to have `sm_120` kernels; this project uses
2.14.1 for CUDA 13.0:

```bash
git clone https://github.com/m-555/explain-to-learn.git ~/explain-to-learn
cd ~/explain-to-learn
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[train,graph,generate,data,demo,dev]"
python -m spacy download en_core_web_sm
```

Check the GPU build, then the code:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), 'sm_120' in torch.cuda.get_arch_list())"
ruff check . && pytest -q
```

The first command should end with `(12, 0) True`. With every extra and the spaCy model
installed, no test is skipped.

Set a Hugging Face token for model downloads and adapter uploads:

```bash
export HF_TOKEN=<token with write access>
```

*Committed:* nothing.

## 2. Data and the edit table

Read the licence notice, then download W&I+LOCNESS, FCE and the GEE data into
`data/raw/`:

```bash
bash scripts/download_data.sh
bash scripts/download_data.sh --accept-licence
```

Build the edit table, the essay splits, the gold edit IDs and the data statistics:

```bash
python -m explain_to_learn.edits build --raw data/raw --out data/processed/edits.jsonl
python -m explain_to_learn.splits --edits data/processed/edits.jsonl --out data/splits
python scripts/data_stats.py --edits data/processed/edits.jsonl --out eval/results/data_stats.md
```

The splits, gold IDs and statistics are already committed. Check that the rebuild
matches them exactly:

```bash
git diff --exit-code data/splits eval/results/data_stats.md
```

*Committed:* nothing new. *Local:* `data/raw/`, `data/processed/edits.jsonl`.

## 3. Grammar graph

The graph grounds systems B, D and R, and the teacher's silver data, so its content
must be complete before step 5. [grammar_graph/README.md](../grammar_graph/README.md)
describes the files and how to add or edit a concept. Write the rules, examples and
terms of every concept, have them checked, and set `status: reviewed`. Then:

```bash
python -m explain_to_learn.graph validate --strict
python scripts/draw_graph.py
python scripts/export_rdf.py
```

`validate --strict` must report no errors. The reasons behind the main design choices
are recorded in [design_decisions.md](design_decisions.md).

*Committed:* `grammar_graph/`, `report/figures/grammar_graph.png`,
`report/grammar_graph.ttl`.

## 4. Baselines A and B on a dev sample

A first run of the prompted systems checks the prompts, the JSON parsing and the
metrics before any expensive step. It uses 500 W&I+LOCNESS dev edits outside the gold
set, drawn with a fixed seed:

```bash
python scripts/sample_dev_ids.py --n 500
```

Run A and B with the model loaded in the process (transformers backend, about 8 GB of
GPU memory), then score them:

```bash
python scripts/run_system.py --system A --run dev-explain --ids outputs/dev_ids.txt --limit 20
python scripts/run_system.py --system A --run dev-explain --ids outputs/dev_ids.txt
python scripts/run_system.py --system B --run dev-explain --ids outputs/dev_ids.txt
python scripts/compute_metrics.py --run dev-explain --ids outputs/dev_ids.txt --systems A B --compare B:A
```

The `--limit 20` run is a smoke test; the full run resumes after it. Look at a few
replies in `outputs/A/dev-explain.jsonl`: `parsed` should be filled for nearly every
edit. Repeat with `--mode hint` and `--run dev-hint` for hint mode.

A server is faster for long runs. Start the base model in Docker on port 8000 and add
`--workers 16` after switching A and B to `backend: openai` in `configs/systems.yaml`:

```bash
bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507
```

*Committed:* `eval/results/metrics_dev-explain.md` and `.json` (and `dev-hint`).
*Local:* `outputs/dev_ids.txt`, `outputs/A/`, `outputs/B/`.

## 5. Silver data from the teacher

Serve the teacher model (system R, gpt-oss-20b) on port 8001, as `configs/systems.yaml`
expects. vLLM takes 90% of the GPU memory by default, so stop any process that holds a
model on the GPU first. In a second terminal:

```bash
PORT=8001 bash scripts/serve_model.sh openai/gpt-oss-20b
```

If vLLM's MXFP4 kernels do not run on `sm_120`, use the llama.cpp server on the same
port instead (see the comment on system R in `configs/systems.yaml`).

Generate explanations and hints for 8,000 training edits (25% in hint mode), excluding
the gold set and error types without a graph concept:

```bash
python scripts/make_silver.py generate --n 8000 --workers 16
```

The run appends to `outputs/silver/teacher.jsonl` and resumes if it stops.

*Committed:* nothing yet. *Local:* `outputs/silver/teacher.jsonl`.

## 6. Silver filter and the 200-item hand-check

```bash
python scripts/make_silver.py filter
```

This keeps outputs that parse, name the graph concept and name the edited words, and
writes `outputs/silver/kept.jsonl`, the counts in `eval/results/silver_filter.md` and
`.json`, and 200 random kept items to `outputs/silver_handcheck.csv`.

*Teacher time:* mark every item correct or not in the annotation app, judging the rule,
the located words and the level ([annotation guide](annotation_guide.md#d-silver-hand-check)):

```bash
python -m explain_to_learn.annotation --task silver
```

The app saves the marks to `outputs/annotations/silver_handcheck.csv`, not to the
sample file, which the filter rewrites each time it runs. Then:

```bash
python scripts/make_silver.py score --handcheck outputs/annotations/silver_handcheck.csv
```

This prints the share of correct items with a 95% Wilson interval and writes
`eval/results/silver_handcheck.json`. If the share is low, fix the prompts or the graph
and repeat steps 5 and 6 before training.

*Committed:* `eval/results/silver_filter.md`, `silver_filter.json`,
`silver_handcheck.json`. *Local:* `outputs/silver/`, `outputs/silver_handcheck.csv`,
`outputs/annotations/`.

## 7. LoRA adapters for C and D

Stop the teacher server (Ctrl+C in its terminal), so that training has the whole GPU.
Build the SFT examples: one set per system, the same edits and targets, with the graph
snippet in D's prompts only:

```bash
python -m explain_to_learn.train_data --silver outputs/silver/kept.jsonl --edits data/processed/edits.jsonl --out outputs/sft
```

Check memory and speed with a short run, then train the main variant for both systems:

```bash
python train/train_lora.py --variant r16 --system d --max-steps 20 --run-name d-r16-check
python train/train_lora.py --variant r16 --system c
python train/train_lora.py --variant r16 --system d
```

Then the other variants of `train/lora_config.yaml` (rank 8, rank 32, 4-bit QLoRA):

```bash
for variant in r8 r32 qlora_r16; do for system in c d; do python train/train_lora.py --variant "$variant" --system "$system"; done; done
```

Each adapter is saved to `outputs/adapters/<system>-<variant>/` with `run_info.json`
(settings, losses, peak memory, time, versions). To train in Docker instead, use the
`docker run` command in [gpu_setup.md](gpu_setup.md#build-and-run) with the same
arguments.

Compare the variants on the dev sample, never on the gold set:

```bash
for variant in r16 r8 r32 qlora_r16; do for system in C D; do python scripts/run_system.py --system "$system" --adapter "outputs/adapters/${system,,}-$variant" --run "dev-explain-$variant" --ids outputs/dev_ids.txt; done; python scripts/compute_metrics.py --run "dev-explain-$variant" --ids outputs/dev_ids.txt --systems C D --compare D:C; done
```

C and D load `outputs/adapters/c-r16` and `d-r16` (`adapter` in
`configs/systems.yaml`). If another variant is better on the dev sample, point both
systems at it and record the choice in the model card.

*Committed:* `eval/results/metrics_dev-explain-*.md` and `.json`; a changed `adapter`
path in `configs/systems.yaml` if another variant is chosen. *Local:* `outputs/sft/`,
`outputs/adapters/`, `outputs/checkpoints/`.

## 8. Gold-set generation for A, B, C, D and R

Run every system on the 300 gold edits, in both modes. A to D load their model in the
process, so run them while no server holds the GPU:

```bash
for system in A B C D; do for mode in explain hint; do python scripts/run_system.py --system "$system" --run "gold-$mode" --mode "$mode"; done; done
```

Then start the teacher server again (step 5) and run R:

```bash
for mode in explain hint; do python scripts/run_system.py --system R --run "gold-$mode" --mode "$mode" --workers 16; done
```

Every run reads `data/splits/gold_edit_ids.txt` and writes
`outputs/<system>/gold-<mode>.jsonl`.

*Committed:* nothing yet. *Local:* `outputs/{A,B,C,D,R}/gold-*.jsonl`.

## 9. Automatic metrics

```bash
python scripts/compute_metrics.py --run gold-explain --systems A B C D R
python scripts/compute_metrics.py --run gold-hint --mode hint --systems A B C D R
```

Each writes coverage, concept accuracy, faithfulness, length fit and off-level terms per
system, by CEFR level and by error category, and paired bootstrap intervals for B-A,
C-A, D-B and D-C (10,000 resamples).

*Committed:* `eval/results/metrics_gold-explain.md`, `.json`, `metrics_gold-hint.md`,
`.json`.

## 10. Rating sheets and ratings

Make the blind sheets: 100 gold edits x systems A to D = 400 items, shuffled with the
system names hidden, a 50-item subset for the second rater, and the key:

```bash
python scripts/make_rating_sheet.py --system A=outputs/A/gold-explain.jsonl --system B=outputs/B/gold-explain.jsonl --system C=outputs/C/gold-explain.jsonl --system D=outputs/D/gold-explain.jsonl
```

The files are `outputs/human_eval/rating_sheet.csv`,
`rating_sheet_second_rater.csv` and `key.csv`. Keep the key closed until all ratings
are done.

*Teacher time:* rate with the annotation app, following
[annotation_guide.md](annotation_guide.md), and the rubric in
[eval/rubric.md](../eval/rubric.md). Both raters first rate the rubric's examples and
discuss disagreements. The first rater's ratings are saved to
`outputs/annotations/ratings_rater1.csv` (400 items):

```bash
python -m explain_to_learn.annotation --task ratings
```

The second rater rates the 50-item sheet on their own computer
([annotation guide](annotation_guide.md#c-second-rater)) and sends back
`ratings_<their name>.csv`; save it in `outputs/annotations/` as `ratings_rater2.csv`.

Summarise the main rater's ratings per system, error category and level:

```bash
python -m explain_to_learn.human_eval --ratings outputs/annotations/ratings_rater1.csv --key outputs/human_eval/key.csv --out-dir eval/results --name human_eval_rater1
```

*Committed:* `eval/results/human_eval_rater1.md` and its `_by_*.csv` tables.
*Local:* `outputs/human_eval/`, `outputs/annotations/`.

## 11. Agreement and the LLM judge

Agreement between the two raters on their 50 shared items (exact agreement, Cohen's
kappa, quadratic-weighted kappa):

```bash
python scripts/agreement.py --first outputs/annotations/ratings_rater1.csv --second outputs/annotations/ratings_rater2.csv --title "Rater 1 vs rater 2" --out eval/results/agreement_raters.md
```

The LLM judge rates the same sheet with the rubric as its instructions. With the
teacher server on port 8001 as the judge, allow room for its reasoning and check 20
items first:

```bash
python scripts/llm_judge.py --model openai/gpt-oss-20b --base-url http://localhost:8001/v1 --max-tokens 2048 --limit 20 --out outputs/human_eval/ratings_llm_judge_check.csv
```

`outputs/human_eval/ratings_llm_judge_check.json` must show `"failed": 0`. Then rate
all items and compare the judge with the main rater:

```bash
python scripts/llm_judge.py --model openai/gpt-oss-20b --base-url http://localhost:8001/v1 --max-tokens 2048
python scripts/agreement.py --first outputs/annotations/ratings_rater1.csv --second outputs/human_eval/ratings_llm_judge.csv --title "LLM judge vs rater 1" --out eval/results/agreement_judge.md
python -m explain_to_learn.human_eval --ratings outputs/human_eval/ratings_llm_judge.csv --key outputs/human_eval/key.csv --out-dir eval/results --name human_eval_llm_judge
```

`scripts/make_report_assets.py` reads both agreement tables from these paths.

*Committed:* `eval/results/agreement_raters.md`, `agreement_judge.md`,
`human_eval_llm_judge.md` and its tables. *Local:* the judge's ratings and settings in
`outputs/human_eval/`.

## 12. Correction F0.5

The main evaluation uses gold corrections. The end-to-end pipeline adds a correction
step, scored here on all W&I+LOCNESS dev sentences against the official dev M2 files
(span-based correction F0.5 with ERRANT, overall and per CEFR level). The GEC model runs
in the process:

```bash
python scripts/evaluate_correction.py --run gec-t5-small --corrector gec --limit 50
python scripts/evaluate_correction.py --run gec-t5-small --corrector gec
```

The prompted alternative uses the base model on the server of port 8000:

```bash
bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507
```

```bash
python scripts/evaluate_correction.py --run qwen3-4b-prompted --corrector llm
```

The `--limit 50` run is a smoke test; the full run continues from its 50 corrected
sentences and replaces its result files. Hypotheses and M2 files go to
`outputs/correction/<run>/`, and a stopped run resumes.

*Committed:* `eval/results/correction_gec-t5-small.md`, `.json`,
`correction_qwen3-4b-prompted.md`, `.json`. *Local:* `outputs/correction/`.

## 13. Analysis assets

`scripts/make_report_assets.py` and the notebooks in `notebooks/` turn the result files
of steps 4 to 12 into the report's tables and figures (see the script's `--help` for its
options). The per-level and per-category breakdowns and the paired bootstrap intervals
are already in the `metrics_gold-*.json` files.

*Committed:* the generated tables and figures under `report/`.

## 14. Demo and Hugging Face Space

Try the pipeline from the command line, then in the demo:

```bash
explain-to-learn explain "She don't like to going to school." --level A2 --mode hint
python demo/app.py
```

`python demo/app.py --fake` starts the interface without models. Push the chosen D
adapter to the Hub, private until release:

```bash
hf upload <user>/explain-to-learn-d-r16 outputs/adapters/d-r16 . --private
```

Then follow [demo/README.md](../demo/README.md#deploy-to-a-hugging-face-space) to create
the Space.

*Committed:* nothing. The Space and the adapters are their own repositories on the Hub.

## 15. Release checklist

- [ ] All result files of steps 4 to 12 are committed, and `git ls-files data/raw
      data/processed outputs` prints nothing.
- [ ] [MODEL_CARD.md](../MODEL_CARD.md): training data sizes, hand-check accuracy,
      training time and peak memory (from `run_info.json`), results, and the chosen
      variant. Upload it as the adapter repositories' `README.md`.
- [ ] [DATA_CARD.md](../DATA_CARD.md): silver counts and rejections, hand-check, SFT
      example counts, raters and agreement.
- [ ] Report: fill `report/main.tex` and build it with `latexmk -pdf main.tex` in
      `report/`.
- [ ] README: the demo link, the results table and example outputs (invented sentences
      only), including negative results; tick the roadmap items that are done.
- [ ] Make the adapter repositories and the Space public, and pin
      `demo/requirements.txt` to the release tag.
- [ ] Tag the release:

      ```bash
      git tag -a v1.0 -m "Explain-to-Learn v1.0"
      git push origin v1.0
      ```
