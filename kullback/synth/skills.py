"""The scenario review skill: the order one review of a scenario reads in (D373, D382).

A skill is text the harness puts in the prompt; this one says nothing a tool does not enforce and names
no domain, tool or value.
"""

from __future__ import annotations

SCENARIO_SKILL_NAME = "synth_review"

SCENARIO_SKILL = """Scenario review skill. For each scenario read, in this order: the statement, its binding and
what it requires; the policy lines it cites, and the part of the policy in your prompt that governs it; the web,
searching for how often this happens in this kind of business and reading the page you will cite; the bound rows
of the world and the tools the statement needs, looked up and found; then rule. Ask whether this happens to
customers of this kind of business and whether this world holds the tools and the rows to serve it."""

__all__ = ["SCENARIO_SKILL", "SCENARIO_SKILL_NAME"]
