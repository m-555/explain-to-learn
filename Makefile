# The experiment runbook (docs/experiments.md) as make targets, one per step.
#
# Run inside WSL2 Ubuntu, from the repository root, with the virtual environment active.
# Each target prints the commands it runs; `make -n TARGET` prints them without running.
# Variables are set on the command line, for example:
#
#   make baselines MODE=hint
#   make train SYSTEM=C VARIANT=r8
#   make gold-runs SYSTEMS=R WORKERS=16
#
# `make help` lists the targets and the current values of the variables.

PYTHON ?= python

# System for `train` (C or D).
SYSTEM ?= D
# Systems for `gold-runs` and `gee`. A to D load their model in the process; R needs the
# teacher server, so run it on its own: make gold-runs SYSTEMS=R WORKERS=16
SYSTEMS ?= A B C D
# LoRA variant of train/lora_config.yaml for `train` and `dev-variant`.
VARIANT ?= r16
# Feedback mode, explain or hint, and the run name it gives (`metrics`, `report-assets`).
MODE ?= explain
RUN ?= gold-$(MODE)
# Teacher model (system R, the silver data and the LLM judge) and its server port.
MODEL ?= openai/gpt-oss-20b
PORT ?= 8001
# Sizes: training edits for the silver data, dev edits for the first runs.
N_SILVER ?= 8000
N_DEV ?= 500
# Parallel requests: 1 for a model in the process, about 16 for a server.
WORKERS ?= 1
SILVER_WORKERS ?= 16
# File name part of the second rater's ratings (outputs/annotations/ratings_<name>.csv).
RATER2 ?= rater2
# Corrector of configs/systems.yaml (gec or llm) and the name of its run.
CORRECTOR ?= gec
CORRECTION_RUN ?= gec-t5-small
# Extra arguments, e.g. RUN_ARGS="--limit 20", TRAIN_ARGS="--max-steps 20", DEMO_ARGS=--fake.
RUN_ARGS ?=
TRAIN_ARGS ?=
DEMO_ARGS ?=
# `make graph-check STRICT=` checks a draft graph without --strict.
STRICT ?= --strict
DPO_VARIANT ?= beta01

EDITS := data/processed/edits.jsonl
DEV_IDS := outputs/dev_ids.txt
SHEETS := outputs/human_eval
MARKS := outputs/annotations
TRAIN_SYSTEM = $(shell echo $(SYSTEM) | tr '[:upper:]' '[:lower:]')

.DEFAULT_GOAL := help
.PHONY: help env-check data graph-check baselines serve-teacher silver silver-score \
	annotate sft-data train dev-variant gold-runs metrics gee rating-sheets human-eval \
	judge-check judge agreement correction-eval report-assets runs dpo-data dpo demo \
	smoke test lint

help: ## List the targets and the variables
	@echo "Targets (docs/experiments.md describes each step):"
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'
	@echo "Variables: SYSTEM=$(SYSTEM) SYSTEMS='$(SYSTEMS)' VARIANT=$(VARIANT) MODE=$(MODE)"
	@echo "  RUN=$(RUN) MODEL=$(MODEL) PORT=$(PORT) N_SILVER=$(N_SILVER) N_DEV=$(N_DEV)"
	@echo "  WORKERS=$(WORKERS) CORRECTOR=$(CORRECTOR) CORRECTION_RUN=$(CORRECTION_RUN)"

env-check: ## Step 1: PyTorch build for the RTX 5090, ERRANT with spaCy, the CLI
	$(PYTHON) -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), 'sm_120' in torch.cuda.get_arch_list())"
	$(PYTHON) -c "import errant; errant.load('en'); print('ERRANT and en_core_web_sm: ok')"
	explain-to-learn --help > /dev/null

data: ## Step 2: edit table, splits, gold IDs and statistics, compared with git
	test -d data/raw/wi+locness || { echo "Run bash scripts/download_data.sh first."; exit 1; }
	$(PYTHON) -m explain_to_learn.edits build --raw data/raw --out $(EDITS)
	$(PYTHON) -m explain_to_learn.splits --edits $(EDITS) --out data/splits
	$(PYTHON) scripts/data_stats.py --edits $(EDITS) --out eval/results/data_stats.md
	git diff --exit-code data/splits eval/results/data_stats.md

graph-check: ## Step 3: validate the grammar graph, draw it and export it as RDF
	$(PYTHON) -m explain_to_learn.graph validate $(STRICT)
	$(PYTHON) scripts/draw_graph.py
	$(PYTHON) scripts/export_rdf.py

baselines: ## Step 4: systems A and B on N_DEV dev edits in MODE, and their metrics
	$(PYTHON) scripts/sample_dev_ids.py --n $(N_DEV)
	$(PYTHON) scripts/run_system.py --system A --run dev-$(MODE) --mode $(MODE) --ids $(DEV_IDS) --workers $(WORKERS) $(RUN_ARGS)
	$(PYTHON) scripts/run_system.py --system B --run dev-$(MODE) --mode $(MODE) --ids $(DEV_IDS) --workers $(WORKERS) $(RUN_ARGS)
	$(PYTHON) scripts/compute_metrics.py --run dev-$(MODE) --mode $(MODE) --ids $(DEV_IDS) --systems A B --compare B:A

serve-teacher: ## Steps 5, 8 and 11: serve MODEL on PORT with vLLM (Ctrl+C stops)
	PORT=$(PORT) bash scripts/serve_model.sh $(MODEL)

silver: ## Steps 5 and 6: teacher outputs for N_SILVER edits, filter, hand-check
	$(PYTHON) scripts/make_silver.py generate --n $(N_SILVER) --workers $(SILVER_WORKERS)
	$(PYTHON) scripts/make_silver.py filter

silver-score: ## Step 6: accuracy of the hand-check marked in the annotation app
	$(PYTHON) scripts/make_silver.py score --handcheck $(MARKS)/silver_handcheck.csv

annotate: ## Steps 6, 8 and 10: start the annotation app on http://127.0.0.1:7861
	$(PYTHON) -m explain_to_learn.annotation

sft-data: ## Step 7: SFT examples for systems C and D from the kept silver items
	$(PYTHON) -m explain_to_learn.train_data --silver outputs/silver/kept.jsonl --edits $(EDITS) --out outputs/sft

train: ## Step 7: train the LoRA adapter of SYSTEM for VARIANT
	$(PYTHON) train/train_lora.py --variant $(VARIANT) --system $(TRAIN_SYSTEM) $(TRAIN_ARGS)

dev-variant: ## Step 7: C and D with the VARIANT adapters on the dev edits, metrics
	$(PYTHON) scripts/run_system.py --system C --adapter outputs/adapters/c-$(VARIANT) --run dev-explain-$(VARIANT) --ids $(DEV_IDS) $(RUN_ARGS)
	$(PYTHON) scripts/run_system.py --system D --adapter outputs/adapters/d-$(VARIANT) --run dev-explain-$(VARIANT) --ids $(DEV_IDS) $(RUN_ARGS)
	$(PYTHON) scripts/compute_metrics.py --run dev-explain-$(VARIANT) --ids $(DEV_IDS) --systems C D --compare D:C

gold-runs: ## Step 8: every system of SYSTEMS on the gold edits in MODE
	for system in $(SYSTEMS); do $(PYTHON) scripts/run_system.py --system $$system --run gold-$(MODE) --mode $(MODE) --workers $(WORKERS) $(RUN_ARGS) || exit 1; done

metrics: ## Step 9: automatic metrics of RUN for systems A to D and R
	$(PYTHON) scripts/compute_metrics.py --run $(RUN) --mode $(MODE) --systems A B C D R

gee: ## Step 9: systems of SYSTEMS on GEE's edits, then the comparison with GEE
	$(PYTHON) scripts/evaluate_on_gee.py prepare
	for system in $(SYSTEMS); do $(PYTHON) scripts/evaluate_on_gee.py run --system $$system --workers $(WORKERS) || exit 1; done
	$(PYTHON) scripts/evaluate_on_gee.py compare

rating-sheets: ## Step 10: blind rating sheets of A to D and the key
	$(PYTHON) scripts/make_rating_sheet.py --system A=outputs/A/gold-explain.jsonl --system B=outputs/B/gold-explain.jsonl --system C=outputs/C/gold-explain.jsonl --system D=outputs/D/gold-explain.jsonl

human-eval: ## Step 10: the first rater's ratings per system, error category and level
	$(PYTHON) -m explain_to_learn.human_eval --ratings $(MARKS)/ratings_rater1.csv --key $(SHEETS)/key.csv --out-dir eval/results --name human_eval_rater1

judge-check: ## Step 11: LLM judge (MODEL on PORT) on 20 items, as a check
	$(PYTHON) scripts/llm_judge.py --model $(MODEL) --base-url http://localhost:$(PORT)/v1 --max-tokens 2048 --limit 20 --out $(SHEETS)/ratings_llm_judge_check.csv

judge: ## Step 11: LLM judge on every item, and the summary of its ratings
	$(PYTHON) scripts/llm_judge.py --model $(MODEL) --base-url http://localhost:$(PORT)/v1 --max-tokens 2048
	$(PYTHON) -m explain_to_learn.human_eval --ratings $(SHEETS)/ratings_llm_judge.csv --key $(SHEETS)/key.csv --out-dir eval/results --name human_eval_llm_judge

agreement: ## Step 11: agreement of rater 2 and of the LLM judge with rater 1
	$(PYTHON) scripts/agreement.py --first $(MARKS)/ratings_rater1.csv --second $(MARKS)/ratings_$(RATER2).csv --title "Rater 1 vs rater 2" --out eval/results/agreement_raters.md
	$(PYTHON) scripts/agreement.py --first $(MARKS)/ratings_rater1.csv --second $(SHEETS)/ratings_llm_judge.csv --title "LLM judge vs rater 1" --out eval/results/agreement_judge.md

correction-eval: ## Step 12: correction F0.5 of CORRECTOR on W&I+LOCNESS dev
	$(PYTHON) scripts/evaluate_correction.py --run $(CORRECTION_RUN) --corrector $(CORRECTOR) $(RUN_ARGS)

report-assets: ## Step 13: report figures and tables from the edit table and results
	$(PYTHON) scripts/make_report_assets.py data
	$(PYTHON) scripts/make_report_assets.py results --run $(RUN)

runs: ## After each run: update the run registry eval/results/runs.md
	$(PYTHON) scripts/collect_runs.py

dpo-data: ## Optional, step 10: DPO pairs from the teacher's preference judgements
	$(PYTHON) -m explain_to_learn.preference

dpo: ## Optional, step 10: DPO on top of the system D adapter, variant DPO_VARIANT
	$(PYTHON) train/train_dpo.py --variant $(DPO_VARIANT) $(TRAIN_ARGS)

demo: ## Step 14: the demo on http://localhost:7860 (DEMO_ARGS=--fake)
	$(PYTHON) demo/app.py $(DEMO_ARGS)

smoke: ## Every runbook step on an invented corpus with a fake model server
	$(PYTHON) -m pytest -m smoke -q -rs tests/smoke

test: ## The test suite (without the smoke test)
	$(PYTHON) -m pytest -q -rs

lint: ## Style and formatting checks with ruff
	ruff check .
	ruff format --check .
