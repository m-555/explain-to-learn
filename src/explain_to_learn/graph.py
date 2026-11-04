"""Grammar graph: load the teacher-written concepts and the ERRANT error type mapping.

Concepts (rule wording per CEFR level, example, prerequisites) are stored as YAML in
`grammar_graph/` and loaded into a NetworkX graph. This module looks up the concept
for an ERRANT error type and returns the snippet used to ground an explanation.
"""
