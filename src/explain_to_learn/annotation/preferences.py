"""Preference pairs: which of two explanations of the same edit is better.

Two run files from `scripts/run_system.py` are compared: from two systems, or from two
samples of one system. Each is given a name, e.g. `D-1=outputs/D/train-sample-1.jsonl`.
An edit becomes a pair when both files have readable output for it in the same mode and
the texts the learner would see differ. The two outputs are shown as "1" and "2"
without their names, in an order fixed per edit by a seeded coin flip, so that screen
position and source are not linked.

Each choice is one JSON object in `<out-dir>/preferences.jsonl`:

    edit_id, level, mode  the edit, and the level and mode the outputs were written for
    prompt_messages       the chat messages the model got, rebuilt with
                          `generate.edit_item` (with the graph snippet if either source
                          was grounded for this edit)
    chosen, rejected      the better and the other output, as the JSON string the model
                          is trained to write (`prompts.build_target`, with the field the
                          mode does not use left empty, as in the SFT targets); both null
                          for a tie or when both are bad
    source_systems        the two source names, sorted
    judgement             "first" or "second" (the better source, by its position in
                          `source_systems`), "tie" or "both_bad"
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from explain_to_learn.annotation.display import escape, marked_sentence, quote
from explain_to_learn.annotation.store import Store
from explain_to_learn.annotation.task import Task
from explain_to_learn.edits import read_jsonl
from explain_to_learn.generate import edit_item
from explain_to_learn.metrics import shown_text
from explain_to_learn.prompts import build_target, detokenize

OUTPUT_FILE = "preferences.jsonl"
JUDGEMENTS = ("first", "second", "tie", "both_bad")
# Each option label starts with its keyboard shortcut. "1" and "2" are screen positions.
CHOICES = [
    ("1 Explanation 1 is better", "1"),
    ("2 Explanation 2 is better", "2"),
    ("3 Tie: both equally good", "tie"),
    ("4 Both bad", "both_bad"),
]


def read_run(path: str | Path) -> dict[str, dict[str, Any]]:
    """Generation records of one run file by edit ID; the last one wins."""
    with Path(path).open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return {row["edit_id"]: row for row in rows}


def build_pairs(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter]:
    """Pairs of outputs for the edits of `first` that can be compared, and skip counts."""
    pairs, skipped = [], Counter()
    for edit_id, a in first.items():
        b = second.get(edit_id)
        if b is None:
            skipped["only_in_one_file"] += 1
            continue
        mode = a.get("mode", "explain")
        if b.get("mode", "explain") != mode:
            skipped["different_mode"] += 1
            continue
        texts = (shown_text(a["parsed"], mode), shown_text(b["parsed"], mode))
        if not all(texts):
            skipped["no_readable_output"] += 1
        elif texts[0] == texts[1]:
            skipped["same_text"] += 1
        else:
            pairs.append(
                {
                    "edit_id": edit_id,
                    "level": a["level"],
                    "mode": mode,
                    "grounded": bool(a.get("grounded") or b.get("grounded")),
                    "outputs": (a["parsed"], b["parsed"]),
                }
            )
    return pairs, skipped


def training_target(parsed: dict[str, str], mode: str) -> str:
    """An output as the JSON string the model is trained to write, unused field empty."""
    return build_target(
        parsed["concept"],
        parsed["explanation"] if mode == "explain" else "",
        parsed["hint"] if mode == "hint" else "",
    )


class PreferenceTask(Task):
    """Choose the better of two outputs for the same edit."""

    def __init__(
        self,
        sources: Sequence[tuple[str, str | Path]],
        edits_path: str | Path,
        graph_dir: str | Path,
        out_dir: str | Path,
        seed: int = 0,
    ) -> None:
        names = sorted(name for name, _ in sources)
        if len(names) != 2 or names[0] == names[1]:
            raise ValueError("Give exactly two sources with different names, as NAME=PATH.")
        paths = dict(sources)
        self.names = (names[0], names[1])
        self.pairs, self.skipped = build_pairs(*(read_run(paths[name]) for name in names))
        if not self.pairs:
            raise ValueError(f"The two runs have no outputs to compare ({dict(self.skipped)}).")
        self.seed = seed
        wanted = {pair["edit_id"] for pair in self.pairs}
        self.edits = {e.edit_id: e for e in read_jsonl(edits_path) if e.edit_id in wanted}
        missing = sorted(wanted - set(self.edits))
        if missing:
            raise ValueError(f"{len(missing)} edits are not in {edits_path}, e.g. {missing[0]}")
        self.graph = None
        if any(pair["grounded"] for pair in self.pairs):
            from explain_to_learn.graph import load_graph

            graph_dir = Path(graph_dir)
            self.graph = load_graph(graph_dir / "concepts.yaml", graph_dir / "errant_map.yaml")
        order = [(pair["edit_id"], *self.names) for pair in self.pairs]
        self.store = Store(
            Path(out_dir) / OUTPUT_FILE,
            lambda record: (record["edit_id"], *record["source_systems"]),
            order,
        )

    def swapped(self, index: int) -> bool:
        """True if the second source is shown first for this pair (a seeded coin flip)."""
        return random.Random(f"{self.seed}:{self.pairs[index]['edit_id']}").random() < 0.5

    def source_at(self, index: int, position: int) -> int:
        """Which source (0 or 1) is shown at screen position 0 or 1."""
        return 1 - position if self.swapped(index) else position

    def view(self, index: int) -> list[str]:
        pair = self.pairs[index]
        edit = self.edits[pair["edit_id"]]
        item = "\n".join(
            [
                f"**Level** {escape(pair['level'])} · **Mode** {escape(pair['mode'])}",
                "",
                f"**Sentence:** {marked_sentence(edit)}",
                "",
                f"**Corrected:** {escape(detokenize(edit.corrected))}",
            ]
        )
        panels = [item]
        for position in (0, 1):
            parsed = pair["outputs"][self.source_at(index, position)]
            panels.append(
                f"**Explanation {position + 1}** · concept: {escape(parsed['concept'])}\n\n"
                + quote(shown_text(parsed, pair["mode"]))
            )
        return panels

    def values(self, index: int) -> list[str | None]:
        saved = self.saved(index)
        if saved is None:
            return [None]
        judgement = saved["judgement"]
        if judgement in ("first", "second"):
            source = JUDGEMENTS.index(judgement)
            position = 0 if self.source_at(index, 0) == source else 1
            return [str(position + 1)]
        return [judgement]

    def save(self, index: int, choice: str | None) -> None:
        if choice not in {value for _, value in CHOICES}:
            raise ValueError("Choose the better explanation, a tie, or both bad.")
        pair = self.pairs[index]
        edit = self.edits[pair["edit_id"]]
        chosen = rejected = None
        judgement = choice
        if choice in ("1", "2"):
            source = self.source_at(index, int(choice) - 1)
            judgement = JUDGEMENTS[source]
            chosen = training_target(pair["outputs"][source], pair["mode"])
            rejected = training_target(pair["outputs"][1 - source], pair["mode"])
        _, messages = edit_item(edit, pair["mode"], self.graph, grounded=pair["grounded"])
        self.store.save(
            {
                "edit_id": pair["edit_id"],
                "level": pair["level"],
                "mode": pair["mode"],
                "prompt_messages": messages,
                "chosen": chosen,
                "rejected": rejected,
                "source_systems": list(self.names),
                "judgement": judgement,
            }
        )
