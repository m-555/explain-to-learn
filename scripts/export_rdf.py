"""Export the grammar graph as RDF (Turtle), using SKOS terms.

Each concept becomes a `skos:Concept` in one `skos:ConceptScheme`, with its name as
`skos:prefLabel`, its ERRANT error types as `skos:notation` and its rule wording as
`skos:definition` (B2 band) and `skos:scopeNote` (A2 and B1 bands). Prerequisites
use the property `etl:hasPrerequisite`, declared as a sub-property of `skos:related`
so that SKOS tools still see the link.

Usage:
    python scripts/export_rdf.py [--out report/grammar_graph.ttl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from explain_to_learn.graph import DEFAULT_CONCEPTS_PATH, DEFAULT_MAP_PATH, GrammarGraph, load_graph

ETL = Namespace("https://github.com/m-555/explain-to-learn/grammar-graph#")


def to_rdf(grammar: GrammarGraph) -> Graph:
    """Return the concepts and prerequisite links as an rdflib Graph."""
    rdf = Graph()
    rdf.bind("skos", SKOS)
    rdf.bind("etl", ETL)

    scheme = ETL["GrammarGraph"]
    rdf.add((scheme, RDF.type, SKOS.ConceptScheme))
    rdf.add((scheme, SKOS.prefLabel, Literal("Explain-to-Learn grammar graph", lang="en")))

    rdf.add((ETL.hasPrerequisite, RDF.type, OWL.ObjectProperty))
    rdf.add((ETL.hasPrerequisite, RDFS.subPropertyOf, SKOS.related))
    rdf.add((ETL.hasPrerequisite, RDFS.label, Literal("has prerequisite", lang="en")))
    rdf.add((ETL.status, RDF.type, OWL.DatatypeProperty))
    rdf.add((ETL.status, RDFS.label, Literal("content status", lang="en")))

    for concept in grammar.concepts.values():
        node = ETL[concept.id]
        rdf.add((node, RDF.type, SKOS.Concept))
        rdf.add((node, SKOS.inScheme, scheme))
        rdf.add((node, SKOS.prefLabel, Literal(concept.name, lang="en")))
        rdf.add((node, ETL.status, Literal(concept.status)))
        for error_type in grammar.error_types_for(concept.id):
            rdf.add((node, SKOS.notation, Literal(error_type)))
        if concept.rules.get("B2"):
            rdf.add((node, SKOS.definition, Literal(concept.rules["B2"], lang="en")))
        for band in ("A2", "B1"):
            if concept.rules.get(band):
                note = f"{band}: {concept.rules[band]}"
                rdf.add((node, SKOS.scopeNote, Literal(note, lang="en")))
        for prerequisite in concept.prerequisites:
            if prerequisite in grammar.concepts:
                rdf.add((node, ETL.hasPrerequisite, ETL[prerequisite]))
    return rdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the grammar graph as RDF Turtle.")
    parser.add_argument("--concepts", default=DEFAULT_CONCEPTS_PATH)
    parser.add_argument("--map", default=DEFAULT_MAP_PATH)
    parser.add_argument("--out", default="report/grammar_graph.ttl")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_rdf(load_graph(args.concepts, args.map)).serialize(out_path, format="turtle")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
