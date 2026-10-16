"""The scenario review skill: the order one review of a scenario reads in (D373).

A skill is text the harness puts in the prompt; this one says nothing a tool does not enforce and names
no domain, tool or value.
"""

from __future__ import annotations

SCENARIO_SKILL_NAME = "synth_review"

SCENARIO_SKILL = """Scenario review skill. For each scenario read, in this order: the statement; the policy lines
it cites, and the part of the policy in your prompt that governs it; the search evidence, searching for how
often this happens in this kind of business where none is held; the rows of the world, looking up the kind of
row and the step the statement needs; then rule. Ask whether this happens to customers of this kind of business,
whether this world holds the tools and the rows to serve it, and whether it serves a customer's own ask."""

__all__ = ["SCENARIO_SKILL", "SCENARIO_SKILL_NAME"]
