"""Gold reference explanations, written by the teacher for the gold edits.

For each edit ID in `data/splits/gold_edit_ids.txt` the view shows the sentence with the
edit marked, the corrected sentence, the ERRANT error type, the learner's level, and the
grammar graph concept with its rule for that level and its example. The teacher picks
the concept (prefilled from the ERRANT mapping), writes a reference explanation and a
hint, can add notes, and can flag an edit that no grammar rule explains. Run files from
`scripts/run_system.py` can be given as model drafts to start from; the record says
whether a draft was used.

Records go to `<out-dir>/gold_explanations.jsonl`, one JSON object per edit, in the
order of the gold IDs:

    edit_id               the gold edit
    level                 the level the text is written for (as in the prompts,
                          LOCNESS `native` writers count as C2)
    error_type            the ERRANT error type, e.g. R:VERB:SVA
    concept_id, concept   the chosen graph concept and its name
    explanation, hint     the teacher's reference texts
    not_rule_explainable  true if no grammar rule explains the edit
    notes                 free text
    from_draft            true if the texts were started from a model draft

The field names follow the silver training rows (`silver.filter_silver`).
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any

from explain_to_learn.annotation.display import edit_change, escape, marked_sentence, quote
from explain_to_learn.annotation.store import Store
from explain_to_learn.annotation.task import Task
from explain_to_learn.edits import Edit, read_jsonl
from explain_to_learn.generate import OUTPUT_FIELDS
from explain_to_learn.io import open_graph, read_ids, read_rows
from explain_to_learn.prompts import detokenize, prompt_level

OUTPUT_FILE = "gold_explanations.jsonl"


def read_drafts(paths: Sequence[str | Path]) -> dict[str, dict[str, str]]:
    """Model drafts by edit ID, from run files of `scripts/run_system.py`.

    Each field (concept, explanation, hint) is taken from the first file that has it
    for the edit, so an explain run and a hint run together give both texts.
    """
    drafts: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_rows(path):
            parsed = row.get("parsed")
            if not parsed:  # the reply was not valid JSON
                continue
            draft = drafts.setdefault(row["edit_id"], dict.fromkeys(OUTPUT_FIELDS, ""))
            for field in OUTPUT_FIELDS:
                if not draft[field] and parsed.get(field):
                    draft[field] = parsed[field]
    return drafts


def level_text(edit: Edit) -> str:
    """The learner's level, saying which level the text is written for if they differ."""
    level = prompt_level(edit.cefr_level)
    if level == edit.cefr_level:
        return level
    return f"{edit.cefr_level} (write for {level})"


class GoldTask(Task):
    """Write a reference explanation and hint for each gold edit."""

    def __init__(
        self,
        gold_ids_path: str | Path,
        edits_path: str | Path,
        graph_dir: str | Path,
        out_dir: str | Path,
        drafts_paths: Sequence[str | Path] = (),
    ) -> None:
        try:
            self.graph = open_graph(graph_dir)
        except ImportError as exc:
            raise ValueError(
                'The gold task needs the graph extra: pip install -e ".[graph]"'
            ) from exc
        ids = read_ids(gold_ids_path)
        wanted = set(ids)
        edits = {edit.edit_id: edit for edit in read_jsonl(edits_path) if edit.edit_id in wanted}
        missing = [edit_id for edit_id in ids if edit_id not in edits]
        if missing:
            raise ValueError(
                f"{len(missing)} gold edits are not in {edits_path}, e.g. {missing[0]}"
            )
        self.edits = [edits[edit_id] for edit_id in ids]
        self.has_drafts = bool(drafts_paths)
        self.drafts = read_drafts(drafts_paths)
        self.store = Store(Path(out_dir) / OUTPUT_FILE, itemgetter("edit_id"), ids)

    def concept_choices(self) -> list[tuple[str, str]]:
        """Dropdown choices: `name (id)` labels for every graph concept, by name."""
        concepts = sorted(self.graph.concepts.values(), key=lambda c: c.name.lower())
        return [(f"{concept.name} ({concept.id})", concept.id) for concept in concepts]

    def suggested_concept(self, index: int) -> str | None:
        """The concept the ERRANT mapping gives for the edit's error type."""
        return self.graph.concept_for(self.edits[index].error_type)

    def concept_view(self, index: int, concept_id: str | None) -> str:
        """The chosen concept with its rule for the edit's level, example and prerequisites."""
        suggested = self.suggested_concept(index)
        lines = []
        if not concept_id or concept_id not in self.graph.concepts:
            lines.append("*No concept chosen.*")
        else:
            level = prompt_level(self.edits[index].cefr_level)
            snippet = self.graph.snippet(concept_id, level)
            status = self.graph.concepts[concept_id].status
            wrong, right = (escape(snippet["example"][key]) for key in ("wrong", "right"))
            example = f"wrong *{wrong}*, right *{right}*" if wrong else "*not written yet*"
            lines += [
                f"**Graph concept:** {escape(snippet['name'])} (`{concept_id}`, {status})",
                "",
                f"**Rule for {level}:** {escape(snippet['rule']) or '*not written yet*'}",
                "",
                f"**Example:** {example}",
            ]
            if snippet["prerequisites"]:
                lines += ["", f"**Builds on:** {escape(', '.join(snippet['prerequisites']))}"]
        if suggested != concept_id:
            mapped = f"`{suggested}`" if suggested else "no concept"
            lines += ["", f"The ERRANT mapping gives {mapped} for this error type."]
        return "\n".join(lines)

    def draft_view(self, index: int) -> str:
        """The model draft for the edit, if drafts were given."""
        if not self.has_drafts:
            return ""
        draft = self.drafts.get(self.edits[index].edit_id)
        if not draft:
            return "*No model draft for this edit.*"
        return "\n".join(
            [
                f"**Model draft** (concept: {escape(draft['concept']) or '–'})",
                "",
                "Explanation:",
                "",
                quote(draft["explanation"]),
                "",
                "Hint:",
                "",
                quote(draft["hint"]),
            ]
        )

    def fill_from_draft(
        self, index: int, explanation: str, hint: str, from_draft: bool
    ) -> tuple[str, str, bool]:
        """Copy the draft into the empty text boxes; the boxes already written are kept."""
        draft = self.drafts.get(self.edits[index].edit_id)
        if not draft:
            return explanation, hint, from_draft
        new_explanation = explanation if explanation.strip() else draft["explanation"]
        new_hint = hint if hint.strip() else draft["hint"]
        changed = (new_explanation, new_hint) != (explanation, hint)
        return new_explanation, new_hint, from_draft or changed

    def view(self, index: int) -> list[str]:
        edit = self.edits[index]
        item = "\n".join(
            [
                f"**Sentence:** {marked_sentence(edit)}",
                "",
                f"**Corrected:** {escape(detokenize(edit.corrected))}",
                "",
                f"**Correction:** {edit_change(edit)} · **Error type:** `{edit.error_type}` · "
                f"**Level:** {level_text(edit)}",
                "",
                f"Edit ID: `{edit.edit_id}`",
            ]
        )
        concept_id = self.values(index)[0]
        return [item, self.concept_view(index, concept_id), self.draft_view(index)]

    def values(self, index: int) -> list[Any]:
        saved = self.saved(index)
        if saved is None:
            return [self.suggested_concept(index), "", "", "", False, False]
        return [
            saved["concept_id"] or None,
            saved["explanation"],
            saved["hint"],
            saved["notes"],
            saved["not_rule_explainable"],
            saved["from_draft"],
        ]

    def save(
        self,
        index: int,
        concept_id: str | None,
        explanation: str,
        hint: str,
        notes: str,
        not_rule_explainable: bool,
        from_draft: bool,
    ) -> None:
        edit = self.edits[index]
        explanation, hint, notes = explanation.strip(), hint.strip(), notes.strip()
        if concept_id and concept_id not in self.graph.concepts:
            raise ValueError(f"Unknown concept {concept_id!r}.")
        if not_rule_explainable:
            if not notes:
                raise ValueError("Say in the notes why no grammar rule explains this edit.")
        elif not (concept_id and explanation and hint):
            raise ValueError(
                "Choose a concept and write both the explanation and the hint, or flag the "
                "edit as not explainable by a rule."
            )
        concept = self.graph.concepts[concept_id].name if concept_id else ""
        self.store.save(
            {
                "edit_id": edit.edit_id,
                "level": prompt_level(edit.cefr_level),
                "error_type": edit.error_type,
                "concept_id": concept_id or "",
                "concept": concept,
                "explanation": explanation,
                "hint": hint,
                "not_rule_explainable": bool(not_rule_explainable),
                "notes": notes,
                "from_draft": bool(from_draft),
            }
        )
