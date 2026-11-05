"""The parts of the project working together, with a fake model and the fixture graph.

Each test runs one path through several modules with their real code, and checks that
what one step writes is what the next step reads:

- a gold edit -> graph snippet -> prompt -> model reply -> parsed JSON -> metrics;
- teacher outputs -> silver filter -> SFT examples for systems C and D, whose prompts
  must equal the prompts the systems get at inference time;
- the configuration files, which name the same models, adapters and endpoints.

All sentences are invented for the tests.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("yaml")

from explain_to_learn.edits import Edit, write_jsonl  # noqa: E402
from explain_to_learn.generate import (  # noqa: E402
    Decoding,
    edit_item,
    generate,
    load_systems,
    run_batch,
)
from explain_to_learn.graph import load_graph  # noqa: E402
from explain_to_learn.metrics import load_terms, score_item, summarize  # noqa: E402
from explain_to_learn.prompts import build_messages, format_snippet, prompt_level  # noqa: E402
from explain_to_learn.silver import filter_silver  # noqa: E402
from explain_to_learn.train_data import build_examples  # noqa: E402

ROOT = Path(__file__).parents[1]
FIXTURE_GRAPH = Path(__file__).parent / "fixtures" / "graph"
TERMS = ROOT / "configs" / "grammar_terms.txt"


def load_script(path: Path):
    """Import a script by its path; registered in sys.modules for its dataclasses."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def graph():
    return load_graph(FIXTURE_GRAPH / "concepts.yaml", FIXTURE_GRAPH / "errant_map.yaml")


def make_edit(edit_id, tokens, start, end, correction, error_type, level, split="dev"):
    """An edit-table row for an invented sentence."""
    words = tokens.split()
    corrected = [*words[:start], *correction.split(), *words[end:]]
    operation, category = error_type.split(":", 1)
    essay = edit_id.split("-")[1]
    return Edit(
        edit_id=edit_id,
        corpus="wi",
        essay_id=essay,
        split=split,
        cefr_level=level,
        sent_idx=int(edit_id.split("-")[2]),
        original=tokens,
        corrected=" ".join(corrected),
        start=start,
        end=end,
        original_text=" ".join(words[start:end]),
        corrected_text=correction,
        error_type=error_type,
        operation=operation,
        error_category=category,
    )


EDITS = [
    make_edit("wi-1-0-0", "She like tea .", 1, 2, "likes", "R:VERB:SVA", "A2"),
    make_edit("wi-1-1-0", "I bought book .", 2, 2, "a", "M:DET", "B1"),
    make_edit("wi-2-0-0", "I have two cat .", 3, 4, "cats", "R:NOUN:NUM", "native"),
    make_edit("wi-2-1-0", "I very like it .", 1, 2, "really", "R:ADV", "C1"),
]


class TeacherModel:
    """Answers like a good model: the graph concept name, quoting the edited words.

    It reads only the prompt, so it shows what a model can learn from each input.
    """

    def complete(self, messages, decoding):
        user = messages[-1]["content"]
        fields = dict(line.split(": ", 1) for line in user.splitlines() if ": " in line)
        concept = fields.get("  Concept", "Grammar")
        action = fields["Correction"].split(" (tokens")[0]  # e.g. replace "a" with "b"
        if fields["Mode"] == "explain":
            return json.dumps(
                {"concept": concept, "explanation": f"You need to {action}.", "hint": ""}
            )
        if action.startswith("add"):
            before = fields["In context"].split("[+")[0].split()[-1]
            hint = f'Something is missing after "{before}".'
        else:
            hint = f"Look again at {action.split(' with ')[0].split(' ', 1)[1]}."
        return json.dumps({"concept": concept, "explanation": "", "hint": hint})


def test_gold_edit_through_prompt_model_and_metrics(graph):
    terms = load_terms(TERMS)
    items = []
    for edit in EDITS:
        level = prompt_level(edit.cefr_level)
        concept_id = graph.concept_for(edit.error_type)
        snippet = graph.snippet(concept_id, level)
        record, messages = edit_item(edit, "explain", graph, grounded=True, system="B")

        # The prompt carries the edit, the level and the snippet worded for that level.
        assert messages == build_messages(edit, level, "explain", snippet)
        assert format_snippet(snippet) in messages[1]["content"]
        assert record["concept_id"] == concept_id and record["level"] == level

        parsed = generate(TeacherModel(), messages, Decoding()).parsed
        assert parsed["concept"] == snippet["name"]
        items.append(
            score_item(
                edit,
                record["mode"],
                parsed,
                {concept_id, snippet["name"]},
                graph.level_terms(level),
                terms,
                level=level,
            )
        )

    summary = summarize(items)
    assert summary["edits"] == 4
    assert summary["covered"] == summary["concept_correct"] == 1.0
    # "very" (three letters or more) is named, and the M:DET edit is quoted.
    assert summary["faithful"] == 1.0
    assert [item["level"] for item in items] == ["A2", "B1", "native", "C1"]


def test_run_batch_rows_feed_the_metrics_script(graph, tmp_path):
    compute_metrics = load_script(ROOT / "scripts" / "compute_metrics.py")
    items = [edit_item(edit, "hint", graph, grounded=True, system="D") for edit in EDITS]
    out = tmp_path / "D" / "gold.jsonl"
    run_batch(TeacherModel(), items, out)

    rows = compute_metrics.read_outputs(out)
    scored = compute_metrics.score_system(rows, EDITS, graph, load_terms(TERMS), "explain")
    assert [item["mode"] for item in scored] == ["hint"] * 4
    assert summarize(scored)["concept_correct"] == 1.0


def test_silver_rows_become_the_inference_prompts_of_c_and_d(graph):
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    train_edits = [
        make_edit(f"wi-{n}-0-0", "She like tea .", 1, 2, "likes", "R:VERB:SVA", "B1", "train")
        for n in range(10, 30)
    ] + [
        make_edit(f"wi-{n}-1-0", "I bought book .", 2, 2, "a", "M:DET", "A2", "train")
        for n in range(10, 30)
    ]
    by_id = {edit.edit_id: edit for edit in train_edits}
    teacher = [
        edit_item(edit, "hint" if i % 4 == 0 else "explain", graph, grounded=True, system="R")
        for i, edit in enumerate(train_edits)
    ]
    rows = [{**record, "parsed": generate(TeacherModel(), msgs).parsed} for record, msgs in teacher]

    kept, counts = filter_silver(rows, by_id)
    assert counts["kept"] == len(train_edits)

    for system in ("c", "d"):
        grounded = systems[system.upper()]["grounded"]
        examples = build_examples(kept, by_id, system, graph, share=0.1)
        for example in examples["train"] + examples["validation"]:
            edit = by_id[example["edit_id"]]
            # Training input == what run_system.py sends to this system at inference.
            _, inference = edit_item(edit, example["mode"], graph, grounded)
            assert example["prompt"] == inference
            assert example["grounded"] == grounded
            target = json.loads(example["completion"][0]["content"])
            assert (
                target["concept"] == graph.snippet(graph.concept_for(edit.error_type), "B1")["name"]
            )


def test_configs_name_the_same_models_adapters_and_servers(tmp_path, monkeypatch):
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    train_lora = load_script(ROOT / "train" / "train_lora.py")
    lora = train_lora.load_config(train_lora.CONFIG_PATH, "r16")
    monkeypatch.chdir(tmp_path)  # so that a trained adapter in outputs/ does not stop parse_args
    args = train_lora.parse_args(["--system", "d"])

    # The adapters are trained on the base model that A and B prompt.
    assert {systems[name]["model"] for name in "ABCD"} == {lora["model"]["name"]}
    # C and D load the adapters that train_lora.py writes by default.
    for name in "CD":
        assert systems[name]["adapter"] == f"outputs/adapters/{name.lower()}-r16"
    assert args.output_dir / args.run_name == Path("outputs/adapters/d-r16")
    # On a server, C and D are the adapter names that scripts/serve_model.sh serves.
    serve = (ROOT / "scripts" / "serve_model.sh").read_text(encoding="utf-8")
    for name in "CD":
        served = systems[name]["served_model"]
        assert f"{served}=outputs/adapters/{name.lower()}-r16" in serve
    assert systems["A"]["base_url"] == "http://localhost:8000/v1"
    assert systems["R"]["base_url"] == "http://localhost:8001/v1"


def test_edit_table_round_trip_keeps_every_field(tmp_path):
    from explain_to_learn.edits import read_jsonl

    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    assert read_jsonl(tmp_path / "edits.jsonl") == EDITS


def test_errant_on_free_text_gives_the_rows_of_the_edit_table(errant_annotator, tmp_path):
    from conftest import WI_M2

    from explain_to_learn.edits import Essay, essay_edits, parse_m2
    from explain_to_learn.pipeline import text_edits
    from explain_to_learn.prompts import detokenize

    blocks = WI_M2.split("\n\n")
    m2 = tmp_path / "wi.m2"
    m2.write_text("\n\n".join([blocks[0], blocks[3]]) + "\n", encoding="utf-8")
    sentences = parse_m2(m2)
    table = essay_edits("wi", "dev", Essay("1", "B1", ""), sentences)

    text = " ".join(detokenize(s.tokens) for s in sentences)
    corrected = " ".join(detokenize(e.corrected) for e in {e.sent_idx: e for e in table}.values())
    _, edits = text_edits(text, "B1", annotator=errant_annotator, corrected=corrected)

    fields = ("sent_idx", "original", "corrected", "start", "end", "original_text")
    fields += ("corrected_text", "error_type", "operation", "error_category", "cefr_level")
    assert [[getattr(e, f) for f in fields] for e in edits] == [
        [getattr(e, f) for f in fields] for e in table
    ]


def test_pipeline_output_scores_like_a_run_system_file(graph, tmp_path):
    from explain_to_learn.pipeline import Explainer, explain_gold_edits

    compute_metrics = load_script(ROOT / "scripts" / "compute_metrics.py")
    explained = explain_gold_edits(EDITS, "explain", Explainer(TeacherModel(), system="D"), graph)
    out = tmp_path / "D" / "gold.jsonl"
    out.parent.mkdir()
    out.write_text("".join(json.dumps(item.to_row()) + "\n" for item in explained))

    scored = compute_metrics.score_system(
        compute_metrics.read_outputs(out), EDITS, graph, load_terms(TERMS), "explain"
    )
    assert summarize(scored)["faithful"] == 1.0


class SentenceCorrector:
    """Corrects the sentences it knows; fails if `calls_allowed` is used up."""

    def __init__(self, corrections, calls_allowed=100):
        self.corrections, self.calls_allowed = corrections, calls_allowed

    def correct(self, sentences):
        assert self.calls_allowed > 0, "the corrector should not be called"
        self.calls_allowed -= 1
        return [self.corrections.get(s, s) for s in sentences]


def correction_args(raw_dir, tmp_path, *extra):
    return [
        "--run", "test", "--raw", str(raw_dir), "--config", str(ROOT / "configs" / "systems.yaml"),
        "--out-dir", str(tmp_path / "outputs"), "--results", str(tmp_path / "results"), *extra,
    ]  # fmt: skip


def test_dev_sentences_for_correction_come_with_levels_and_references(raw_dir):
    script = load_script(ROOT / "scripts" / "evaluate_correction.py")
    sentences = script.dev_sentences(raw_dir)

    # Three W&I dev files (A, B, C) with the two fixture essays, then LOCNESS.
    assert len(sentences) == 13
    assert [s.level for s in sentences[:4]] == ["A2", "A2", "A2", "C2"]
    assert sentences[-1].level == "native"
    assert sentences[0].tokens == "She do n't like to going ."
    assert sentences[0].reference.startswith("S She do n't like to going .\nA 1 2|||R:VERB:SVA")
    assert sentences[1].reference.endswith("|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")


def test_correction_run_scores_each_level_and_resumes(raw_dir, tmp_path, monkeypatch):
    script = load_script(ROOT / "scripts" / "evaluate_correction.py")

    def fake_parallel(orig, hyp, out_m2):
        lines = Path(hyp).read_text().splitlines()
        Path(out_m2).write_text("\n\n".join(f"S {line}" for line in lines) + "\n")

    def fake_compare(hyp_m2, ref_m2):
        n = len(script.m2_blocks(hyp_m2))
        assert n == len(script.m2_blocks(ref_m2))
        return {"tp": n, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f0.5": 1.0}

    monkeypatch.setattr(script, "errant_parallel", fake_parallel)
    monkeypatch.setattr(script, "errant_compare", fake_compare)
    monkeypatch.setattr(script, "spacy_tokeniser", lambda: lambda text: text)
    # One call for the full run and one for the short run; the resumed run makes none.
    corrector = SentenceCorrector({"I has a cat.": "I have a cat."}, calls_allowed=2)
    monkeypatch.setattr(script, "make_corrector", lambda settings: corrector)

    script.main(correction_args(raw_dir, tmp_path))
    results = json.loads((tmp_path / "results" / "correction_test.json").read_text())
    assert results["overall"]["sentences"] == 13
    assert {level: r["sentences"] for level, r in results["by_level"].items()} == {
        "A2": 9,
        "C2": 3,
        "native": 1,
    }
    assert results["corrector"]["model"] == "Unbabel/gec-t5_small"
    hyp = (tmp_path / "outputs" / "test" / "hyp.txt").read_text().splitlines()
    assert hyp[3] == "I have a cat." and len(hyp) == 13

    script.main(correction_args(raw_dir, tmp_path))  # resumes: no corrector call
    report = (tmp_path / "results" / "correction_test.md").read_text()
    assert "| C2 | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |" in report
    assert "cat" not in report and "She" not in report

    script.main(correction_args(raw_dir, tmp_path, "--run", "short", "--limit", "4"))
    short = json.loads((tmp_path / "results" / "correction_short.json").read_text())
    assert short["overall"]["sentences"] == 4 and short["limit"] == 4


def test_correction_f05_with_errant(raw_dir, tmp_path, monkeypatch, errant_annotator):
    script = load_script(ROOT / "scripts" / "evaluate_correction.py")
    corrector = SentenceCorrector({"I has a cat.": "I have a cat."})
    monkeypatch.setattr(script, "make_corrector", lambda settings: corrector)

    script.main(correction_args(raw_dir, tmp_path))
    results = json.loads((tmp_path / "results" / "correction_test.json").read_text())
    # Each W&I file has 5 gold edits and LOCNESS 1; only "has -> have" is corrected.
    overall = results["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (3, 0, 13)
    assert results["by_level"]["C2"]["f0.5"] == 1.0
    assert results["by_level"]["A2"]["recall"] == 0.0


def test_run_system_can_run_another_adapter(graph, tmp_path, monkeypatch):
    run_system = load_script(ROOT / "scripts" / "run_system.py")
    loaded = {}

    def backend(system):
        loaded.update(system)
        return TeacherModel()

    monkeypatch.setattr(run_system, "make_backend", backend)
    monkeypatch.setattr(run_system, "open_graph", lambda directory: graph)
    write_jsonl(EDITS, tmp_path / "edits.jsonl")
    (tmp_path / "ids.txt").write_text("wi-1-0-0\n")

    run_system.main(
        ["--system", "D", "--run", "dev-r8", "--adapter", "outputs/adapters/d-r8"]
        + ["--ids", str(tmp_path / "ids.txt"), "--edits", str(tmp_path / "edits.jsonl")]
        + ["--config", str(ROOT / "configs" / "systems.yaml"), "--out-dir", str(tmp_path)]
    )
    assert loaded["adapter"] == "outputs/adapters/d-r8" and loaded["grounded"]
    assert (tmp_path / "D" / "dev-r8.jsonl").is_file()
