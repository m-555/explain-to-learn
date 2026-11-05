# Experiment log

A dated record of every experiment run for this project, written as it happens,
including runs that failed or gave negative results. The numbers live in
[eval/results/runs.md](../eval/results/runs.md), which `make runs`
(`scripts/collect_runs.py`) builds from each run's `run_info.json` and result files.
This log records what the numbers cannot: why a run was tried, what went wrong, and
what was decided because of it.

## How to write an entry

Add one entry per run or per small group of related runs, newest first, as soon as the
run finishes. Keep each entry short and factual.

- **Date** and **commit**: the date, and the git commit from `run_info.json`
  (a `-dirty` suffix means the run used uncommitted changes; say what they were).
- **What**: the command or `make` target and the settings that differ from the
  defaults (system, variant, run name, data size).
- **Why**: the question the run answers.
- **Outcome**: the headline numbers, with confidence intervals where the metrics give
  them, or the error if the run failed.
- **Learned**: what the outcome means and what changes because of it. A negative
  result is a result: say what it rules out.

Never paste learner sentences or model outputs that quote them; describe them
instead, or give edit IDs.

## Template

```markdown
### YYYY-MM-DD: <short title>

- **Commit:** <hash from run_info.json>
- **What:** <command or make target, and settings that differ from the defaults>
- **Why:** <the question>
- **Outcome:** <numbers with intervals, or the failure>
- **Learned:** <what it means; what changes next>
```

## Entries

No runs yet.
