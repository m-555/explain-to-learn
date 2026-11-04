"""Grammar graph: load the teacher-written concepts and the ERRANT error type mapping.

Concepts (rule wording per CEFR level, example, prerequisites, technical terms) are
stored as YAML in `grammar_graph/` and loaded into a NetworkX graph. This module looks
up the concept for an ERRANT error type, returns the snippet used to ground an
explanation, and checks the files for problems.

The file formats are described in `grammar_graph/README.md`.

Usage:
    python -m explain_to_learn.graph validate [--strict]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

# Every error category ERRANT (v3) can output, from its English classifier.
ERRANT_CATEGORIES = (
    "ADJ",
    "ADJ:FORM",
    "ADV",
    "CONJ",
    "CONTR",
    "DET",
    "MORPH",
    "NOUN",
    "NOUN:INFL",
    "NOUN:NUM",
    "NOUN:POSS",
    "ORTH",
    "OTHER",
    "PART",
    "PREP",
    "PRON",
    "PUNCT",
    "SPELL",
    "UNK",
    "VERB",
    "VERB:FORM",
    "VERB:INFL",
    "VERB:SVA",
    "VERB:TENSE",
    "WO",
)

# ERRANT operations: Missing, Replacement, Unnecessary.
OPERATIONS = ("M", "R", "U")

# Rules are worded for three bands. Each learner level uses one of them.
LEVEL_BANDS = ("A2", "B1", "B2")
LEVEL_TO_BAND = {
    "A1": "A2",
    "A2": "A2",
    "B1": "B1",
    "B2": "B2",
    "C1": "B2",
    "C2": "B2",
    "native": "B2",
}

STATUSES = ("draft", "reviewed")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

DEFAULT_CONCEPTS_PATH = "grammar_graph/concepts.yaml"
DEFAULT_MAP_PATH = "grammar_graph/errant_map.yaml"


@dataclass
class Concept:
    """One grammar concept, as written in concepts.yaml."""

    id: str
    name: str = ""
    status: str = "draft"
    rules: dict[str, str] = field(default_factory=dict)
    example: dict[str, str] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    terms: dict[str, list[str]] = field(default_factory=dict)

    def missing_content(self) -> list[str]:
        """Return the teaching fields that are still empty."""
        missing = [f"rule {band}" for band in LEVEL_BANDS if not self.rules.get(band)]
        if not (self.example.get("wrong") and self.example.get("right")):
            missing.append("example")
        if not any(self.terms.get(band) for band in LEVEL_BANDS):
            missing.append("terms")
        return missing


@dataclass(frozen=True)
class Problem:
    """A validation problem. `severity` is "error" or "warning"."""

    severity: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def level_band(level: str) -> str:
    """Return the rule band (A2, B1 or B2) used for a learner level such as "C1"."""
    key = level.strip()
    key = "native" if key.lower() == "native" else key.upper()
    if key not in LEVEL_TO_BAND:
        raise ValueError(f"unknown level {level!r}; use A1-C2 or native")
    return LEVEL_TO_BAND[key]


class GrammarGraph:
    """Concepts, their prerequisite graph and the ERRANT error type mapping.

    `graph` has one node per concept (with `name` and `status` attributes) and an
    edge from each prerequisite to the concept that needs it, so a topological order
    of the graph is a possible teaching order.
    """

    def __init__(
        self,
        concepts: list[Concept],
        categories: dict[str, str],
        overrides: dict[str, str],
    ) -> None:
        self.concept_list = concepts  # as written, duplicates included
        self.concepts: dict[str, Concept] = {}
        for concept in concepts:
            self.concepts.setdefault(concept.id, concept)
        self.categories = categories
        self.overrides = overrides

        self.graph = nx.DiGraph()
        for concept in self.concepts.values():
            self.graph.add_node(concept.id, name=concept.name, status=concept.status)
        for concept in self.concepts.values():
            for prerequisite in concept.prerequisites:
                if prerequisite in self.concepts:
                    self.graph.add_edge(prerequisite, concept.id)

    def concept_for(self, error_type: str) -> str | None:
        """Return the concept ID for a full ERRANT error type such as "R:VERB:SVA".

        An operation-specific override ("M:DET") is checked first, then the category
        ("DET"). A bare category or "UNK" also works. Returns None if the type has no
        mapping or maps to a concept that does not exist.
        """
        error_type = error_type.strip()
        operation, _, category = error_type.partition(":")
        if operation not in OPERATIONS:
            category = error_type
        concept_id = self.overrides.get(error_type) or self.categories.get(category)
        return concept_id if concept_id in self.concepts else None

    def error_types_for(self, concept_id: str) -> list[str]:
        """Return the mapping keys (categories and overrides) that point at a concept."""
        mapping = {**self.categories, **self.overrides}
        return sorted(key for key, value in mapping.items() if value == concept_id)

    def snippet(self, concept_id: str, level: str) -> dict:
        """Return the grounding snippet for a concept, with the rule worded for `level`."""
        if concept_id not in self.concepts:
            raise KeyError(f"unknown concept {concept_id!r}")
        concept = self.concepts[concept_id]
        return {
            "concept_id": concept.id,
            "name": concept.name,
            "rule": concept.rules.get(level_band(level), ""),
            "example": {"wrong": concept.example["wrong"], "right": concept.example["right"]},
            "prerequisites": [
                self.concepts[p].name or p for p in concept.prerequisites if p in self.concepts
            ],
        }

    def level_terms(self, level: str) -> set[str]:
        """Return the technical terms a learner at `level` can be expected to know.

        Terms are cumulative: B1 includes the A2 terms, B2 includes A2 and B1.
        """
        last = LEVEL_BANDS.index(level_band(level))
        bands = LEVEL_BANDS[: last + 1]
        return {term for c in self.concepts.values() for band in bands for term in c.terms[band]}

    def validate(self, strict: bool = False) -> list[Problem]:
        """Check the graph and the mapping, and return the problems found.

        Missing rule wording, examples or terms are warnings for draft concepts and
        errors for reviewed ones. With `strict=True` they are always errors.
        """
        problems: list[Problem] = []

        def error(message: str) -> None:
            problems.append(Problem("error", message))

        for concept_id, count in Counter(c.id for c in self.concept_list).items():
            if count > 1:
                error(f"duplicate concept id '{concept_id}' ({count} times)")

        for concept in self.concepts.values():
            if not ID_PATTERN.match(concept.id):
                error(f"concept id '{concept.id}' is not snake_case")
            if not concept.name:
                error(f"{concept.id}: missing name")
            if concept.status not in STATUSES:
                error(f"{concept.id}: status '{concept.status}' must be draft or reviewed")
            for prerequisite in concept.prerequisites:
                if prerequisite not in self.concepts:
                    error(f"{concept.id}: unknown prerequisite '{prerequisite}'")
            missing = concept.missing_content()
            if missing:
                severity = "error" if strict or concept.status == "reviewed" else "warning"
                problems.append(Problem(severity, f"{concept.id}: missing {', '.join(missing)}"))

        for cycle in nx.simple_cycles(self.graph):
            error("prerequisite cycle: " + " -> ".join([*cycle, cycle[0]]))

        for key, concept_id in [*self.categories.items(), *self.overrides.items()]:
            if concept_id and concept_id not in self.concepts:
                error(f"errant_map: {key} maps to unknown concept '{concept_id}'")
        for key in self.categories:
            if key not in ERRANT_CATEGORIES:
                error(f"errant_map: '{key}' is not an ERRANT category")
        for key, concept_id in self.overrides.items():
            operation, _, category = key.partition(":")
            if operation not in OPERATIONS or category not in ERRANT_CATEGORIES:
                error(f"errant_map: override '{key}' must be M, R or U plus a category")
            if not concept_id:
                error(f"errant_map: override '{key}' has no concept")
        for category in ERRANT_CATEGORIES:
            if not self.categories.get(category):
                error(f"errant_map: ERRANT category {category} has no concept")

        return problems


def _text(value: object) -> str:
    """Return a YAML value as a stripped string, with empty values as ""."""
    return "" if value is None else str(value).strip()


def _parse_concept(raw: object, position: int) -> Concept:
    """Turn one YAML entry into a Concept. Raises ValueError if its shape is wrong."""
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError(f"concept #{position} must be a mapping with an 'id'")
    concept_id = _text(raw["id"])
    rules = raw.get("rules") or {}
    terms = raw.get("terms") or {}
    example = raw.get("example") or {}
    for key in [*rules, *terms]:
        if key not in LEVEL_BANDS:
            raise ValueError(f"{concept_id}: unknown level '{key}' (use A2, B1 or B2)")
    return Concept(
        id=concept_id,
        name=_text(raw.get("name")),
        status=_text(raw.get("status")) or "draft",
        rules={band: _text(rules.get(band)) for band in LEVEL_BANDS},
        example={key: _text(example.get(key)) for key in ("wrong", "right")},
        prerequisites=[_text(p) for p in raw.get("prerequisites") or []],
        terms={band: [_text(t).lower() for t in terms.get(band) or []] for band in LEVEL_BANDS},
    )


def _read_yaml(path: str | Path) -> object:
    """Read a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_graph(
    concepts_path: str | Path = DEFAULT_CONCEPTS_PATH,
    map_path: str | Path = DEFAULT_MAP_PATH,
) -> GrammarGraph:
    """Load concepts.yaml and errant_map.yaml into a GrammarGraph.

    Only the shape of the files is checked here; call `validate()` for the rest.
    """
    raw_concepts = _read_yaml(concepts_path) or []
    if not isinstance(raw_concepts, list):
        raise ValueError(f"{concepts_path}: expected a list of concepts")
    concepts = [_parse_concept(raw, i + 1) for i, raw in enumerate(raw_concepts)]

    raw_map = _read_yaml(map_path) or {}
    if not isinstance(raw_map, dict):
        raise ValueError(f"{map_path}: expected 'categories' and 'overrides' sections")
    categories = {_text(k): _text(v) for k, v in (raw_map.get("categories") or {}).items()}
    overrides = {_text(k): _text(v) for k, v in (raw_map.get("overrides") or {}).items()}
    return GrammarGraph(concepts, categories, overrides)


def coverage_report(graph: GrammarGraph, problems: list[Problem]) -> str:
    """Return a short plain-text summary of the graph and its problems."""
    concepts = list(graph.concepts.values())
    reviewed = sum(c.status == "reviewed" for c in concepts)
    complete = sum(not c.missing_content() for c in concepts)
    mapped_categories = sum(bool(graph.categories.get(c)) for c in ERRANT_CATEGORIES)
    unmapped = sorted(c.id for c in concepts if not graph.error_types_for(c.id))
    errors = [p for p in problems if p.severity == "error"]
    warnings = [p for p in problems if p.severity == "warning"]

    lines = [
        f"Concepts: {len(concepts)} ({reviewed} reviewed, {len(concepts) - reviewed} draft)",
        f"Concepts with complete content: {complete}/{len(concepts)}",
        f"Prerequisite links: {graph.graph.number_of_edges()}",
        f"ERRANT categories mapped: {mapped_categories}/{len(ERRANT_CATEGORIES)}",
        f"Operation overrides: {len(graph.overrides)}",
        f"Concepts reached only as prerequisites: {len(unmapped)}",
        f"Errors: {len(errors)}, warnings: {len(warnings)}",
    ]
    lines += [f"  {p}" for p in errors + warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns 1 if validation finds errors."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="check the graph files and print a coverage report")
    check.add_argument("--concepts", default=DEFAULT_CONCEPTS_PATH)
    check.add_argument("--map", default=DEFAULT_MAP_PATH)
    check.add_argument(
        "--strict", action="store_true", help="treat incomplete draft concepts as errors"
    )
    args = parser.parse_args(argv)

    graph = load_graph(args.concepts, args.map)
    problems = graph.validate(strict=args.strict)
    print(coverage_report(graph, problems))
    return 1 if any(p.severity == "error" for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
