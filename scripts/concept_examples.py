"""Write example edits for each grammar concept, to help the teacher write the graph.

For each concept in `grammar_graph/concepts.yaml`, writes `<out>/<concept_id>.md` with:

- the ERRANT error types that map to the concept (a category, or an operation-specific
  override such as `U:DET`) and their edit counts per CEFR level, from the whole edit
  table;
- for each of those error types, `--n` random example edits from training essays only,
  with the edit marked in the sentence.

Examples come from the training split only, so that the graph is not written from the
development edits the gold set is drawn from. Each file starts with a note that the
`example` in concepts.yaml must be written by the teacher, never copied from the corpus.
`<out>/index.md` lists the concepts by number of edits, to help decide which to write
first. The files quote learner text, so they go to the git-ignored `outputs/`.

Usage:
    python scripts/concept_examples.py --n 10
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.graph import OPERATIONS, GrammarGraph
from explain_to_learn.io import open_graph, read_ids
from explain_to_learn.prompts import mark_edit

LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2", "native")
NOTE = (
    "> **Do not copy these sentences into `grammar_graph/concepts.yaml`.** They are learner\n"
    "> sentences from the training split, shown only so you can see which errors this\n"
    "> concept has to explain. The `example` in concepts.yaml must be a sentence you write\n"
    "> yourself: the corpus licences do not allow publishing learner text, and the graph\n"
    "> must not repeat the data the models are trained and tested on."
)

# Concept ID -> mapping key (the category or override that decides it) -> edits.
Groups = dict[str, dict[str, list[Edit]]]


def mapping_key(graph: GrammarGraph, error_type: str) -> str:
    """The `errant_map.yaml` key that gives an error type its concept.

    The override for the full type if there is one, else the category (as in
    `GrammarGraph.concept_for`).
    """
    if graph.overrides.get(error_type):
        return error_type
    operation, _, category = error_type.partition(":")
    return category if operation in OPERATIONS else error_type


def group_edits(edits: Iterable[Edit], graph: GrammarGraph) -> tuple[Groups, int]:
    """Edits by concept and mapping key, and the number of edits with no concept."""
    groups: Groups = defaultdict(lambda: defaultdict(list))
    unmapped = 0
    for edit in edits:
        concept_id = graph.concept_for(edit.error_type)
        if concept_id is None:
            unmapped += 1
            continue
        groups[concept_id][mapping_key(graph, edit.error_type)].append(edit)
    return groups, unmapped


def sample_train(edits: list[Edit], train_essays: set[str], n: int, seed: int = 0) -> list[Edit]:
    """A seeded random sample of up to `n` edits from training essays, by edit ID."""
    pool = sorted((e for e in edits if e.essay_key in train_essays), key=lambda e: e.edit_id)
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    return sorted(sample, key=lambda e: e.edit_id)


def counts_table(groups: dict[str, list[Edit]]) -> str:
    """Edits per CEFR level for each mapping key, with a total row."""
    header = ["Error type", *LEVELS, "total"]
    lines = ["| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
    totals: Counter = Counter()
    for key in sorted(groups):
        counts = Counter(edit.cefr_level for edit in groups[key])
        totals.update(counts)
        cells = [counts[level] for level in LEVELS]
        lines.append(f"| `{key}` | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    cells = [totals[level] for level in LEVELS]
    lines.append("| total | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    return "\n".join(lines)


def example_line(edit: Edit) -> str:
    """One example: level, edit ID and the sentence with the edit marked."""
    sentence = mark_edit(edit).replace("`", "'")
    return f"- {edit.cefr_level} · `{edit.edit_id}` · `{sentence}`"


def concept_markdown(
    graph: GrammarGraph,
    concept_id: str,
    groups: dict[str, list[Edit]],
    train_essays: set[str],
    n: int,
    seed: int,
) -> str:
    """The Markdown file for one concept."""
    concept = graph.concepts[concept_id]
    lines = [f"# {concept.name} (`{concept_id}`)", "", NOTE, ""]
    lines.append(f"Status in concepts.yaml: {concept.status}.")
    missing = concept.missing_content()
    if missing:
        lines[-1] += f" Still to write: {', '.join(missing)}."
    if not groups:
        mapped = graph.error_types_for(concept_id)
        if mapped:
            lines += ["", f"No edits in the edit table have the error types {', '.join(mapped)}."]
        else:
            lines += ["", "No error type maps to this concept."]
        needed_by = sorted(graph.graph.successors(concept_id))
        if needed_by:
            lines.append(f"It is a prerequisite of: {', '.join(needed_by)}.")
        return "\n".join(lines) + "\n"

    lines += [
        "",
        "## Edits per level (all splits)",
        "",
        counts_table(groups),
        "",
        f"## Examples from the training split ({n} per error type, seed {seed})",
        "",
        "The edit is in brackets: `[old -> new]` replaces, `[+ new]` adds, `[- old]` removes.",
    ]
    for key in sorted(groups):
        examples = sample_train(groups[key], train_essays, n, seed)
        lines += ["", f"### `{key}`", ""]
        lines += [example_line(edit) for edit in examples] or ["No training edits."]
    return "\n".join(lines) + "\n"


def index_markdown(graph: GrammarGraph, groups: Groups, unmapped: int) -> str:
    """The list of concepts by number of edits, with their mapping keys and status."""
    totals = {
        cid: sum(len(edits) for edits in groups.get(cid, {}).values()) for cid in graph.concepts
    }
    lines = [
        "# Concepts by number of edits",
        "",
        NOTE,
        "",
        "| Concept | Error types | Edits | Status |",
        "| --- | --- | ---: | --- |",
    ]
    for concept_id in sorted(totals, key=lambda cid: (-totals[cid], cid)):
        keys = ", ".join(f"`{key}`" for key in sorted(groups.get(concept_id, {}))) or "–"
        status = graph.concepts[concept_id].status
        lines.append(
            f"| [{concept_id}]({concept_id}.md) | {keys} | {totals[concept_id]} | {status} |"
        )
    lines += ["", f"Edits whose error type has no concept: {unmapped}."]
    return "\n".join(lines) + "\n"


def write_concept_files(
    graph: GrammarGraph,
    edits: list[Edit],
    train_essays: set[str],
    out_dir: str | Path,
    n: int = 10,
    seed: int = 0,
) -> list[Path]:
    """Write one file per concept and the index; return the concept files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, unmapped = group_edits(edits, graph)
    paths = []
    for concept_id in graph.concepts:
        path = out_dir / f"{concept_id}.md"
        text = concept_markdown(
            graph, concept_id, groups.get(concept_id, {}), train_essays, n, seed
        )
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    (out_dir / "index.md").write_text(index_markdown(graph, groups, unmapped), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Example edits per grammar concept.")
    parser.add_argument("--edits", type=Path, default=Path("data/processed/edits.jsonl"))
    parser.add_argument("--train-essays", type=Path, default=Path("data/splits/train_essays.txt"))
    parser.add_argument("--graph", type=Path, default=Path("grammar_graph"))
    parser.add_argument("--out", type=Path, default=Path("outputs/concept_examples"))
    parser.add_argument("--n", type=int, default=10, help="examples per error type")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    graph = open_graph(args.graph)
    train_essays = set(read_ids(args.train_essays))
    paths = write_concept_files(
        graph, read_jsonl(args.edits), train_essays, args.out, args.n, args.seed
    )
    print(f"Wrote {len(paths)} concept files and index.md to {args.out}")


if __name__ == "__main__":
    main()
