# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Release steps are in
[docs/release.md](docs/release.md).

## [Unreleased]

Code only: no experiments have been run yet, so there are no results.

### Added

- Data pipeline: download script for the BEA-2019 corpora, an edit table that aligns
  the official M2 edits to their essays, the official essay-level splits, a stratified
  300-edit gold set, and data statistics.
- Grammar graph: YAML format, loader, validation and coverage report, a mapping of
  every ERRANT category, a prerequisite figure, and a SKOS export. The concepts are
  drafts.
- Explanation systems A-D and reference R: prompts for explain and hint modes,
  generation through an OpenAI-compatible server or Transformers, and system
  definitions.
- Silver training data: generation with the reference model, filtering, and a
  200-item hand-check.
- Automatic metrics: coverage, concept accuracy, faithfulness and level fit, with
  paired bootstrap confidence intervals by error category and level, and correction
  F0.5 with ERRANT.
- LoRA and DPO training for an RTX 5090, with configuration variants, run records
  (`run_info.json`, `environment.txt`) and a run registry.
- Human evaluation: rubric, blind rating sheets, Cohen's kappa, an LLM judge, and a
  local annotation app for gold explanations, ratings, hand-checks and preferences.
- End-to-end pipeline with a correction step, a teacher-facing summary, the
  `explain-to-learn` command, and a Gradio demo.
- Analysis: report tables and figures, notebooks, a comparison with GEE, and a
  showcase-example selector.
- Documentation: experiment runbook, GPU setup, architecture, design decisions,
  annotation guide, licences, release process and experiment log; model and data
  cards; an ACL-format report skeleton.
- Tooling: CI, an end-to-end smoke test of the runbook, report compilation, release
  and Space-sync workflows, pre-commit hooks, a Makefile and requirements files.

[Unreleased]: https://github.com/m-555/explain-to-learn/commits/main
