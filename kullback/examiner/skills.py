"""The review skill: what one review of a Spec reads and rules (D320, D331).

A skill is text the harness puts in the prompt; this one says nothing a gate does not enforce and
names no domain, tool or value.
"""

from __future__ import annotations

REVIEW_SKILL_NAME = "review"

REVIEW_SKILL = """Review skill. For each Task read, in order: the Intent facts; each item with the facts or
policy line it cites; the facts no item covers; the judge items with their question, anchor and evidence.
Ask of each item whether the Intent or the policy asks for it, of each fact whether an item checks it, of
each fixed value whether the policy allows only that one, of each anchor whether the world holds it, and of
each prohibition in the instruction whether an item checks it. File each problem as one ruling with the
smallest fix the writer can make."""

__all__ = ["REVIEW_SKILL", "REVIEW_SKILL_NAME"]
