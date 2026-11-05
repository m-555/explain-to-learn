"""Smoke test of the experiment runbook (docs/experiments.md), before any GPU time is spent.

The runbook's commands run in order as subprocesses, unchanged, in a temporary working
directory that looks like the repository after `scripts/download_data.sh`: the invented
corpus of `tests/smoke/data/raw/` (see `corpus.py`), the grammar graph, the rubric and a
copy of `configs/systems.yaml` whose systems and corrector point at the fake model server
of `fake_server.py`. Smaller sizes are passed where the runbook's defaults need the real
corpus (300 gold edits, 500 dev edits, 8,000 silver edits, 100 rated edits).

The teacher's work is simulated through the annotation app's own task classes
(`explain_to_learn.annotation`), so its files are written by the app's storage code: the
silver hand-check, two raters, the gold explanations and the preference pairs.

What this test cannot cover: real training (a stub adapter folder with a `run_info.json`
stands in for `train/train_lora.py`; DPO training is skipped), real model outputs and
their quality, the vLLM servers, Docker and the GPU, the Gradio interfaces, and the
teacher's judgement. Correction F0.5 and the `explain` command need ERRANT and spaCy's
`en_core_web_sm`; without them that test is skipped.

Run with `pytest -m smoke tests/smoke` or `make smoke`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("networkx")
pytest.importorskip("matplotlib")

import yaml  # noqa: E402

from explain_to_learn.annotation.gold import GoldTask  # noqa: E402
from explain_to_learn.annotation.handcheck import HandcheckTask  # noqa: E402
from explain_to_learn.annotation.preferences import PreferenceTask  # noqa: E402
from explain_to_learn.annotation.ratings import RatingTask  # noqa: E402
from explain_to_learn.edits import read_jsonl  # noqa: E402
from explain_to_learn.human_eval import read_csv  # noqa: E402
from explain_to_learn.prompts import detokenize  # noqa: E402

from . import corpus, fake_server  # noqa: E402

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).parents[2]
# Sizes for the small corpus; the runbook uses 300, 500, 8,000, 100 and 50.
N_GOLD, N_DEV, N_SILVER, N_RATED_EDITS, N_SECOND = 16, 12, 30, 8, 12
SYSTEMS = ("A", "B", "C", "D")
EDITS = "data/processed/edits.jsonl"
GOLD_IDS = "data/splits/gold_edit_ids.txt"
SHEETS = "outputs/human_eval"
MARKS = "outputs/annotations"  # the annotation app's output folder
# Folders where learner text may be written; everything else must stay free of it.
PRIVATE = ("data/raw", "data/processed", "outputs")
RATINGS = ("no", "partly", "yes")
MAKE_TARGETS = [
    "env-check",
    "data",
    "graph-check",
    "baselines",
    "serve-teacher",
    "silver",
    "sft-data",
    "train",
    "gold-runs",
    "metrics",
    "rating-sheets",
    "judge",
    "agreement",
    "report-assets",
    "correction-eval",
    "demo",
    "smoke",
    "test",
    "lint",
]


def errant_ready() -> bool:
    """True if ERRANT and spaCy's English model are installed."""
    try:
        import errant  # noqa: F401
        import spacy
    except ImportError:
        return False
    return spacy.util.is_package("en_core_web_sm")


def git_status() -> str:
    """`git status --porcelain` of the repository, or "" outside a git checkout."""
    if shutil.which("git") is None:
        return ""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""


@dataclass
class Runbook:
    """The working directory of the runbook and the commands run in it."""

    root: Path
    server: fake_server.FakeServer
    repo_status: str

    def run(self, command: str, creates: tuple[str, ...] = ()) -> str:
        """Run one runbook command; check that it succeeds and writes the files `creates`.

        The command is written as in the runbook. `python` is this test's interpreter,
        the scripts are the repository's, and `explain-to-learn` is the package's CLI.
        """
        print("$", command)
        args = shlex.split(command)
        if args[0] == "explain-to-learn":
            args[:1] = ["python", "-m", "explain_to_learn.cli"]
        if args[0] == "python":
            args[0] = sys.executable
        if args[1].startswith("scripts/"):
            args[1] = str(ROOT / args[1])
        env = os.environ | {"MPLBACKEND": "Agg"}
        result = subprocess.run(
            args, cwd=self.root, capture_output=True, text=True, env=env, timeout=600
        )
        assert result.returncode == 0, f"$ {command}\n{result.stdout}\n{result.stderr}"
        for name in creates:
            path = self.root / name
            assert path.is_file() and path.stat().st_size, f"$ {command}\ndid not write {name}"
        return result.stdout

    def path(self, name: str) -> Path:
        return self.root / name

    def lines(self, name: str) -> list[str]:
        return self.path(name).read_text(encoding="utf-8").splitlines()

    def json(self, name: str) -> dict:
        return json.loads(self.path(name).read_text(encoding="utf-8"))


def smoke_config(base_url: str) -> dict:
    """configs/systems.yaml with every system and the LLM corrector on the fake server."""
    config = yaml.safe_load((ROOT / "configs" / "systems.yaml").read_text(encoding="utf-8"))
    for system in config["systems"].values():
        system["backend"], system["base_url"] = "openai", base_url
    config["correctors"]["llm"]["base_url"] = base_url
    return config


def load_train_lora():
    """train/train_lora.py as a module (its helpers need no GPU)."""
    spec = importlib.util.spec_from_file_location("train_lora", ROOT / "train" / "train_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- The teacher, simulated through the annotation app ---------------------------------


def choose(*parts: str) -> int:
    """A stable number for a simulated choice."""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).digest()[0]


def simulate_handcheck(rb: Runbook) -> int:
    """Mark every silver sample item, one in five not correct; returns the item count."""
    task = HandcheckTask(rb.path("outputs/silver_handcheck.csv"), rb.path(MARKS))
    for index in range(len(task)):
        task.save(index, "0" if index % 5 == 4 else "1", "")
    return len(task)


def simulate_ratings(rb: Runbook, sheet: str, rater: str, disagree: bool) -> int:
    """Rate every item of a sheet; the second rater disagrees on some first ratings."""
    task = RatingTask(rb.path(sheet), rb.path(MARKS), rater)
    for index, item in enumerate(task.items):
        if not item["feedback"]:  # rubric rule 4
            task.save(index, "no", "no", "no", "no", "no", "")
            continue
        scores = [RATINGS[choose(item["item_id"], str(d)) % 3] for d in range(4)]
        if disagree and choose(item["item_id"]) % 3 == 0:
            scores[0] = RATINGS[(RATINGS.index(scores[0]) + 1) % 3]
        task.save(index, *scores, "no", "")
    return len(task)


def simulate_gold_explanations(rb: Runbook) -> int:
    """Write a reference for every gold edit, starting from system R's drafts."""
    drafts = [rb.path("outputs/R/gold-explain.jsonl"), rb.path("outputs/R/gold-hint.jsonl")]
    graph = rb.path("grammar_graph")
    task = GoldTask(rb.path(GOLD_IDS), rb.path(EDITS), graph, rb.path(MARKS), drafts)
    for index in range(len(task)):
        concept_id = task.values(index)[0]  # the concept the ERRANT mapping suggests
        explanation, hint, from_draft = task.fill_from_draft(index, "", "", False)
        if index % 7 == 6:
            task.save(index, concept_id, "", "", "A change of style.", True, False)
            continue
        explanation = explanation or "The teacher's explanation."
        hint = hint or "The teacher's hint."
        task.save(index, concept_id, explanation, hint, "", False, from_draft)
    return len(task)


def simulate_preferences(rb: Runbook) -> int:
    """Choose between systems C and D on every comparable gold edit."""
    sources = [(s, rb.path(f"outputs/{s}/gold-explain.jsonl")) for s in ("C", "D")]
    graph = rb.path("grammar_graph")
    task = PreferenceTask(sources, rb.path(EDITS), graph, rb.path(MARKS))
    choices = ("1", "2", "tie", "1", "both_bad", "2")
    for index in range(len(task)):
        task.save(index, choices[index % len(choices)])
    return len(task)


def write_stub_adapters(rb: Runbook) -> None:
    """Stub adapter folders for C and D in place of training, which needs a GPU.

    The run_info.json has the keys train_lora.py writes, with its real config, commit and
    data sizes; the numbers only a real run measures (losses, memory, time) stay empty.
    """
    train_lora = load_train_lora()
    for system in ("c", "d"):
        folder = rb.path(f"outputs/adapters/{system}-r16")
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 16}\n')
        examples = {
            split: len(rb.lines(f"outputs/sft/system_{system}/{split}.jsonl"))
            for split in ("train", "validation")
        }
        info = {
            "run": f"{system}-r16",
            "date": "2026-11-05T12:00:00+00:00",
            "git_commit": train_lora.git_commit(),
            "system": system,
            "variant": "r16",
            "backend": "stub",
            "config": train_lora.load_config(train_lora.CONFIG_PATH, "r16"),
            "examples": examples | {"longer_than_max_seq_length": 0},
            "train_loss": None,
            "eval_loss": None,
            "train_minutes": None,
            "peak_gpu_memory_gb": {"allocated": None, "reserved": None},
            "gpu": "none (smoke-test stub)",
            "versions": train_lora.library_versions(),
        }
        (folder / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# --- The runbook, step by step ----------------------------------------------------------


def data_and_graph(rb: Runbook) -> None:
    """Runbook steps 2 and 3: edit table, splits, statistics and the graph checks."""
    rb.run(f"python -m explain_to_learn.edits build --raw data/raw --out {EDITS}", (EDITS,))
    splits = ("train_essays.txt", "dev_essays.txt", "test_essays.txt", "gold_edit_ids.txt")
    rb.run(
        f"python -m explain_to_learn.splits --edits {EDITS} --out data/splits --n-gold {N_GOLD}",
        tuple(f"data/splits/{name}" for name in splits),
    )
    assert len(rb.lines(GOLD_IDS)) == N_GOLD
    stats = "eval/results/data_stats.md"
    rb.run(f"python scripts/data_stats.py --edits {EDITS} --out {stats}", (stats,))
    assert "Errors: 0" in rb.run("python -m explain_to_learn.graph validate")
    rb.run("python scripts/draw_graph.py", ("report/figures/grammar_graph.png",))
    rb.run("python scripts/export_rdf.py", ("report/grammar_graph.ttl",))
    rb.run("python scripts/concept_examples.py --n 3", ("outputs/concept_examples/index.md",))


def baselines(rb: Runbook) -> None:
    """Runbook step 4: systems A and B on dev edits outside the gold set."""
    rb.run(f"python scripts/sample_dev_ids.py --n {N_DEV}", ("outputs/dev_ids.txt",))
    dev_ids = rb.lines("outputs/dev_ids.txt")
    assert len(dev_ids) == N_DEV and not set(dev_ids) & set(rb.lines(GOLD_IDS))
    run = "python scripts/run_system.py --ids outputs/dev_ids.txt"
    rb.run(f"{run} --system A --run dev-explain --limit 5")
    assert "5 already done" in rb.run(f"{run} --system A --run dev-explain")
    rb.run(f"{run} --system B --run dev-explain")
    metrics = "python scripts/compute_metrics.py --ids outputs/dev_ids.txt --systems A B"
    rb.run(f"{metrics} --run dev-explain --compare B:A", ("eval/results/metrics_dev-explain.json",))
    for system in ("A", "B"):
        rb.run(f"{run} --system {system} --run dev-hint --mode hint")
    rb.run(
        f"{metrics} --run dev-hint --mode hint --compare B:A",
        ("eval/results/metrics_dev-hint.json",),
    )


def silver(rb: Runbook) -> None:
    """Runbook steps 5 and 6: teacher outputs, filter, hand-check and its score."""
    teacher = "outputs/silver/teacher.jsonl"
    rb.run(f"python scripts/make_silver.py generate --n {N_SILVER} --workers 4", (teacher,))
    assert len(rb.lines(teacher)) == N_SILVER
    rb.run(
        "python scripts/make_silver.py filter",
        (
            "outputs/silver/kept.jsonl",
            "outputs/silver_handcheck.csv",
            "eval/results/silver_filter.md",
            "eval/results/silver_filter.json",
        ),
    )
    marked = simulate_handcheck(rb)
    rb.run(
        f"python scripts/make_silver.py score --handcheck {MARKS}/silver_handcheck.csv",
        ("eval/results/silver_handcheck.json",),
    )
    score = rb.json("eval/results/silver_handcheck.json")
    assert (score["rated"], score["unrated"]) == (marked, 0)


def adapters(rb: Runbook) -> None:
    """Runbook step 7: SFT data, stub adapters instead of training, dev runs of C and D."""
    sft = tuple(f"outputs/sft/system_{s}/{x}.jsonl" for s in "cd" for x in ("train", "validation"))
    rb.run(
        f"python -m explain_to_learn.train_data --silver outputs/silver/kept.jsonl "
        f"--edits {EDITS} --out outputs/sft",
        sft,
    )
    write_stub_adapters(rb)
    for system in ("C", "D"):
        rb.run(
            f"python scripts/run_system.py --system {system} "
            f"--adapter outputs/adapters/{system.lower()}-r16 --run dev-explain-r16 "
            "--ids outputs/dev_ids.txt"
        )
    rb.run(
        "python scripts/compute_metrics.py --run dev-explain-r16 --ids outputs/dev_ids.txt "
        "--systems C D --compare D:C",
        ("eval/results/metrics_dev-explain-r16.json",),
    )


def gold_runs_and_metrics(rb: Runbook) -> None:
    """Runbook steps 8 and 9: every system on the gold edits, then the metrics."""
    for system in SYSTEMS:
        for mode in ("explain", "hint"):
            rb.run(
                f"python scripts/run_system.py --system {system} --run gold-{mode} --mode {mode}",
                (f"outputs/{system}/gold-{mode}.jsonl",),
            )
    for mode in ("explain", "hint"):
        rb.run(
            f"python scripts/run_system.py --system R --run gold-{mode} --mode {mode} --workers 16",
            (f"outputs/R/gold-{mode}.jsonl",),
        )
    for mode in ("explain", "hint"):
        rb.run(
            f"python scripts/compute_metrics.py --run gold-{mode} --mode {mode} "
            "--systems A B C D R",
            (f"eval/results/metrics_gold-{mode}.md", f"eval/results/metrics_gold-{mode}.json"),
        )
        results = rb.json(f"eval/results/metrics_gold-{mode}.json")
        assert set(results["systems"]) == {*SYSTEMS, "R"} and results["edits"] == N_GOLD
        assert set(results["comparisons"]) == {"B - A", "C - A", "D - B", "D - C"}


def human_evaluation(rb: Runbook) -> None:
    """Runbook steps 10 and 11: rating sheets, simulated teachers, agreement, LLM judge."""
    systems = " ".join(f"--system {s}=outputs/{s}/gold-explain.jsonl" for s in SYSTEMS)
    rb.run(
        f"python scripts/make_rating_sheet.py {systems} "
        f"--n-edits {N_RATED_EDITS} --n-second {N_SECOND}",
        (
            f"{SHEETS}/rating_sheet.csv",
            f"{SHEETS}/rating_sheet_second_rater.csv",
            f"{SHEETS}/key.csv",
        ),
    )
    items = len(SYSTEMS) * N_RATED_EDITS
    assert simulate_ratings(rb, f"{SHEETS}/rating_sheet.csv", "rater1", False) == items
    second = f"{SHEETS}/rating_sheet_second_rater.csv"
    assert simulate_ratings(rb, second, "rater2", True) == N_SECOND
    assert simulate_gold_explanations(rb) == N_GOLD
    assert len(rb.lines(f"{MARKS}/gold_explanations.jsonl")) == N_GOLD
    assert simulate_preferences(rb) > 0

    rater1, rater2 = f"{MARKS}/ratings_rater1.csv", f"{MARKS}/ratings_rater2.csv"
    judge = f"{SHEETS}/ratings_llm_judge.csv"
    summary = f"python -m explain_to_learn.human_eval --key {SHEETS}/key.csv --out-dir eval/results"
    rb.run(
        f"{summary} --ratings {rater1} --name human_eval_rater1",
        ("eval/results/human_eval_rater1.md", "eval/results/human_eval_rater1_by_system.csv"),
    )
    rb.run(
        f"python scripts/agreement.py --first {rater1} --second {rater2} "
        "--title 'Rater 1 vs rater 2' --out eval/results/agreement_raters.md",
        ("eval/results/agreement_raters.md",),
    )
    llm_judge = (
        f"python scripts/llm_judge.py --model openai/gpt-oss-20b "
        f"--base-url {rb.server.base_url} --max-tokens 2048"
    )
    rb.run(f"{llm_judge} --limit 5 --out {SHEETS}/ratings_llm_judge_check.csv")
    assert rb.json(f"{SHEETS}/ratings_llm_judge_check.json")["failed"] == 0
    rb.run(llm_judge, (judge,))
    assert len(read_csv(rb.path(judge))) == items
    rb.run(
        f"python scripts/agreement.py --first {rater1} --second {judge} "
        "--title 'LLM judge vs rater 1' --out eval/results/agreement_judge.md",
        ("eval/results/agreement_judge.md",),
    )
    rb.run(
        f"{summary} --ratings {judge} --name human_eval_llm_judge",
        ("eval/results/human_eval_llm_judge.md",),
    )


def preference_data(rb: Runbook) -> None:
    """The DPO pairs from the teacher's preferences (DPO training needs a GPU)."""
    rb.run(
        "python -m explain_to_learn.preference",
        ("outputs/dpo/train.jsonl", "outputs/dpo/validation.jsonl"),
    )
    pair = json.loads(rb.lines("outputs/dpo/train.jsonl")[0])
    assert {"prompt", "chosen", "rejected"} <= set(pair)


def gee_comparison(rb: Runbook) -> None:
    """The comparison with GEE's explanations, on the invented GEE file."""
    rb.run(
        "python scripts/evaluate_on_gee.py prepare",
        ("outputs/gee/edit_ids.txt", "outputs/gee/references.jsonl"),
    )
    for system in ("A", "D", "R"):
        rb.run(
            f"python scripts/evaluate_on_gee.py run --system {system}",
            (f"outputs/{system}/gee.jsonl",),
        )
    rb.run(
        "python scripts/evaluate_on_gee.py compare --systems A D R",
        ("eval/results/gee_comparison.md", "outputs/gee/side_by_side.csv"),
    )
    results = rb.json("eval/results/gee_comparison.json")
    assert set(results["sources"]) == {"GEE", "A", "D", "R"} and results["edits"] > 0


def report_and_summary(rb: Runbook) -> None:
    """Runbook steps 13 and 14: report figures and tables, and the teacher summary."""
    figures = ("categories_by_level", "edits_per_essay", "operations_by_level")
    rb.run(
        "python scripts/make_report_assets.py data",
        tuple(f"report/figures/data_{name}.png" for name in figures),
    )
    rb.run(
        "python scripts/make_report_assets.py results --run gold-explain "
        "--min-group 2 --resamples 500",
        (
            "eval/results/results_table.md",
            "report/tables/results_main.tex",
            "report/tables/paired_differences.tex",
            "report/tables/agreement.tex",
            "report/figures/judge_vs_rater_means.png",
        ),
    )
    table = rb.path("eval/results/results_table.md").read_text(encoding="utf-8")
    assert "| Rule correct" in table  # the human ratings were found
    agreement = rb.path("report/tables/agreement.tex").read_text(encoding="utf-8")
    assert "Rater 2 vs rater 1" in agreement and "LLM judge" in agreement

    essay = read_jsonl(rb.path(EDITS))[0].essay_key
    summary = rb.run(f"explain-to-learn summary --essay {essay} --edits {EDITS}")
    assert "# Teacher summary" in summary and "Suggested teaching order" in summary


@pytest.fixture(scope="module")
def runbook(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Runbook]:
    """A working directory after every runbook step that needs no ERRANT."""
    root = tmp_path_factory.mktemp("runbook")
    shutil.copytree(Path(__file__).parent / "data" / "raw", root / "data" / "raw")
    shutil.copytree(ROOT / "grammar_graph", root / "grammar_graph")
    (root / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "grammar_terms.txt", root / "configs")
    (root / "eval").mkdir()
    shutil.copy(ROOT / "eval" / "rubric.md", root / "eval")

    server = fake_server.start()
    config = smoke_config(server.base_url)
    (root / "configs" / "systems.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    rb = Runbook(root, server, git_status())
    try:
        data_and_graph(rb)
        baselines(rb)
        silver(rb)
        adapters(rb)
        gold_runs_and_metrics(rb)
        human_evaluation(rb)
        preference_data(rb)
        gee_comparison(rb)
        report_and_summary(rb)
        yield rb
    finally:
        server.shutdown()


def test_every_model_was_asked_with_its_settings(runbook: Runbook):
    seen = runbook.server.requests_seen
    models = {r["model"] for r in seen if r["kind"] == "explanation"}
    assert models == {"Qwen/Qwen3-4B-Instruct-2507", "system-c", "system-d", "openai/gpt-oss-20b"}
    teacher = [r for r in seen if r["model"] == "openai/gpt-oss-20b" and r["kind"] == "explanation"]
    assert all("reasoning_effort" in r["settings"] for r in teacher)
    judge = [r for r in seen if r["kind"] == "judge"]
    assert judge and all("response_format" in r["settings"] for r in judge)


@pytest.mark.skipif(not errant_ready(), reason="needs ERRANT and spaCy's en_core_web_sm")
def test_correction_and_feedback_commands(runbook: Runbook):
    """Runbook steps 12 and 14: correction F0.5 and the `explain` command."""
    rb = runbook
    correction = "python scripts/evaluate_correction.py --run qwen3-4b-prompted --corrector llm"
    rb.run(f"{correction} --limit 10")
    rb.run(
        correction,
        (
            "eval/results/correction_qwen3-4b-prompted.md",
            "outputs/correction/qwen3-4b-prompted/hyp.m2",
        ),
    )
    result = rb.json("eval/results/correction_qwen3-4b-prompted.json")
    dev_m2 = rb.path("data/raw/wi+locness/m2").glob("*.dev.gold.bea19.m2")
    lines = [line for path in dev_m2 for line in path.read_text(encoding="utf-8").splitlines()]
    sentences = sum(line.startswith("S ") for line in lines)
    assert result["overall"]["sentences"] == sentences and result["overall"]["tp"] > 0

    feedback = rb.run(
        'explain-to-learn explain "She don\'t like to going to school." --level A2 '
        "--mode hint --corrector llm"
    )
    assert "**Corrected:** She doesn't like to go to school." in feedback


def test_run_registry_collects_the_runs(runbook: Runbook):
    rb = runbook
    rb.run("python scripts/collect_runs.py", ("eval/results/runs.md",))
    sections = rb.path("eval/results/runs.md").read_text(encoding="utf-8").split("\n## ")
    training = [line for line in sections[1].splitlines() if line.startswith("| ")][2:]
    assert [line.split(" | ")[2] for line in training] == ["c-r16", "d-r16"]
    for run in ("dev-explain", "dev-hint", "dev-explain-r16", "gold-explain", "gold-hint"):
        assert f"| {run} |" in sections[2]
    if rb.path("eval/results/correction_qwen3-4b-prompted.json").exists():
        assert "| qwen3-4b-prompted | llm |" in sections[3]


def make_targets() -> list[str]:
    """The targets `make help` lists."""
    result = subprocess.run(["make", "help"], cwd=ROOT, capture_output=True, text=True, check=True)
    words = [line.split()[0] for line in result.stdout.splitlines() if line.startswith("  ")]
    return [word for word in words if "=" not in word]  # not the lines of variables


@pytest.mark.skipif(shutil.which("make") is None, reason="needs GNU make")
def test_make_targets_name_existing_scripts_and_modules():
    """`make -n` of every target: each script it runs exists, each module imports."""
    targets = make_targets()
    assert set(MAKE_TARGETS) <= set(targets)
    for target in targets:
        result = subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"make -n {target}:\n{result.stderr}"
        words = shlex.split(result.stdout.replace("$system", "A"))
        for word in words:
            if word.startswith(("scripts/", "train/", "demo/")):
                assert (ROOT / word).is_file(), f"make {target} runs {word}, which does not exist"
        for word, module in zip(words, words[1:], strict=False):
            if word == "-m" and module.startswith("explain_to_learn"):
                assert importlib.util.find_spec(module), f"make {target}: no module {module}"


@pytest.mark.skipif(shutil.which("make") is None, reason="needs GNU make")
def test_make_targets_rerun_the_steps(runbook: Runbook):
    """The make targets run in the runbook's working directory; finished steps resume."""
    rb = runbook
    links = [rb.path(name) for name in ("Makefile", "scripts", "train")]
    for link in links:
        link.symlink_to(ROOT / link.name)
    port = rb.server.server_address[1]
    make = f"make PYTHON={shlex.quote(sys.executable)} PORT={port}"
    try:
        rb.run(f"{make} graph-check STRICT=")
        assert "12 already done" in rb.run(f"{make} baselines N_DEV={N_DEV}")
        rb.run(f"{make} silver N_SILVER={N_SILVER} SILVER_WORKERS=4")
        rb.run(f"{make} silver-score")
        rb.run(f"{make} sft-data")
        rb.run(f"{make} dev-variant")
        assert "0 edits to run" in rb.run(f"{make} gold-runs SYSTEMS='A B C D R' MODE=hint")
        rb.run(f"{make} metrics MODE=hint")
        rb.run(f"{make} gee SYSTEMS='A D R'")
        rb.run(f"{make} human-eval")
        rb.run(f"{make} judge-check MODEL=openai/gpt-oss-20b")
        rb.run(f"{make} agreement")
        rb.run(f"{make} dpo-data")
        rb.run(f"{make} report-assets", ("report/figures/judge_vs_rater_means.png",))
        rb.run(f"{make} runs", ("eval/results/runs.md",))
    finally:
        for link in links:
            link.unlink()


def test_learner_text_stays_in_private_folders(runbook: Runbook):
    """No corpus sentence outside data/raw, data/processed and outputs/, and no change in
    the repository itself."""
    rb = runbook
    sentences = set()
    for path in rb.path("data/raw").rglob("*.m2"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("S "):
                sentences |= {line[2:], detokenize(line[2:])}
    for edit in read_jsonl(rb.path(EDITS)):
        sentences |= {edit.corrected, detokenize(edit.corrected)}
    for record in corpus.GEE_RECORDS:
        sentences |= {record["source"], record["target"]}
        sentences |= {explanation for *_, explanation, _ in record["edits"] if explanation}

    leaks = []
    for path in rb.root.rglob("*"):
        relative = path.relative_to(rb.root).as_posix()
        if not path.is_file() or relative.startswith(PRIVATE):
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        leaks += [f"{relative}: {s}" for s in sentences if s in text]
    assert not leaks, "learner text outside the private folders:\n" + "\n".join(leaks)
    assert git_status() == rb.repo_status, "the smoke test changed files in the repository"


def test_committed_corpus_matches_its_source(tmp_path: Path):
    for path in corpus.write_corpus(tmp_path):
        committed = corpus.RAW_DIR / path.relative_to(tmp_path)
        assert committed.read_bytes() == path.read_bytes(), "run python tests/smoke/corpus.py"
