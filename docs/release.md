# Release process

How to publish a version of Explain-to-Learn: the GitHub release with the report, the
adapters on the Hugging Face Hub, and the demo Space. v1.0 is the first release with
results.

## 1. Before the release

Everything below must be true. Each item names the command or file that shows it.

- [ ] The grammar graph is complete: `python -m explain_to_learn.graph validate --strict`
      passes, and every concept has `status: reviewed`.
- [ ] Tests, lint and the end-to-end smoke test pass: `make test`, `make lint`,
      `make smoke` (and CI is green on `main`).
- [ ] Results are committed: `eval/results/` holds the metrics, the human-evaluation
      summaries, both agreement tables, the correction scores and `runs.md`
      (`make runs`).
- [ ] The README results table is pasted from `eval/results/results_table.md`, and the
      README shows the chosen five good and five bad outputs with a comment on each
      (`scripts/select_showcase_examples.py` proposes candidates).
- [ ] [MODEL_CARD.md](../MODEL_CARD.md) and [DATA_CARD.md](../DATA_CARD.md) have no
      "Not available yet" left where a result now exists, including the silver
      hand-check accuracy.
- [ ] The report compiles (the **Report** workflow's artifact) and has its results,
      limitations and ethics sections.
- [ ] [docs/experiment_log.md](experiment_log.md) records every run, including the
      ones that failed.
- [ ] [docs/licences.md](licences.md) still matches the models in
      `configs/systems.yaml`.

## 2. Final checks for learner text and secrets

Run these on a clean clone of `main`:

```bash
# Nothing from the corpora, outputs or the edit table is tracked, now or in history.
git ls-files | grep -E '^(data/raw|data/processed|outputs)/' && echo "STOP: tracked data"
git log --all --name-only --format= | grep -E '^(data/raw|data/processed|outputs)/' | sort -u
# No tokens or keys in tracked files.
git grep -n -I -E '(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})'
```

All three must print nothing. The pre-commit hook `no-learner-text` already blocks
commits from those folders; this check covers the history as well.

## 3. Version, changelog and tag

1. Set the version in `pyproject.toml` and `CITATION.cff` (`version`, `date-released`).
2. In [CHANGELOG.md](../CHANGELOG.md), move the "Unreleased" entries under a new
   heading `## [1.0.0] - YYYY-MM-DD`, add the results summary, and update the links at
   the bottom.
3. Commit ("Release 1.0.0"), push `main`, then tag and push the tag:

   ```bash
   git tag -a v1.0.0 -m "Explain-to-Learn 1.0.0"
   git push origin v1.0.0
   ```

The **Release** workflow (`.github/workflows/release.yml`) then runs the tests,
compiles the report, and creates the GitHub release with `explain-to-learn-report.pdf`
attached and the changelog section for that version as its notes.

## 4. Adapters on the Hugging Face Hub

Train with `--push-to-hub --hub-repo <user>/explain-to-learn-<system>-<variant>`, or
push an existing adapter folder later. For each adapter repository:

- Use the filled [MODEL_CARD.md](../MODEL_CARD.md) as its `README.md`, with `license:
  other` and the licence section explaining the non-commercial restriction
  ([docs/licences.md](licences.md)).
- Keep `run_info.json` and `environment.txt` in the repository: they record the commit,
  the settings and the exact package versions behind the reported results.
- Make the repositories public only when the release is ready (`--public`, or the
  repository settings).

## 5. The demo Space

One-time setup:

1. Create a Gradio Space on Hugging Face (CPU basic for the preview mode, ZeroGPU for
   the full pipeline), with the README header from [demo/README.md](../demo/README.md).
2. Create a Hugging Face access token with write access to that Space.
3. In the GitHub repository settings: add the token as the secret `HF_TOKEN`
   (Settings → Secrets and variables → Actions → Secrets), and the Space ID, for example
   `m-555/explain-to-learn`, as the variable `HF_SPACE_ID` (same page → Variables).
4. In the Space settings, set `DEMO_ADAPTER` to the system-D adapter's Hub ID, and add
   `HF_TOKEN` as a Space secret if the adapter is private.

After that, the **Sync demo Space** workflow (`.github/workflows/space-sync.yml`)
uploads the demo whenever it is run by hand (Actions → Sync demo Space → Run workflow)
and after every published release. It uploads `demo/`, `src/explain_to_learn/`,
`grammar_graph/` and `configs/systems.yaml`; nothing from `data/` or `outputs/`.

## 6. After the release

- Add the demo link and a short GIF of it at the top of the README.
- Check the Space starts and answers in both modes at every level.
- Add the release's DOI or URL to `CITATION.cff` if one is minted (for example through
  Zenodo's GitHub integration).
