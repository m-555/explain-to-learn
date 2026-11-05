# Run registry

Every training and evaluation run, collected by `scripts/collect_runs.py` from the files the runs wrote: `outputs/adapters/*/run_info.json` and `eval/results/metrics_*.json` and `correction_*.json`. Scores are percentages; a dash means the run did not record the value. Do not edit by hand: run `python scripts/collect_runs.py` (or `make runs`) after each run.

## Training runs

Peak GPU memory is the memory PyTorch reserved. The dev columns are the adapter's scores in run `dev-explain-<variant>` (runbook step 7).

| Date | Commit | Run | System | Variant | Settings | Examples (train/val) | Train loss | Val loss | Peak GPU memory (GB) | Training time (min) | Dev concept acc. | Dev faithful |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Evaluation runs

| Run | System | Edits | Coverage | Concept acc. | Faithful | Length fit | Off-level terms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Correction runs

Span-based ERRANT scores on the W&I+LOCNESS dev sentences.

| Run | Corrector | Model | Sentences | Precision | Recall | F0.5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
