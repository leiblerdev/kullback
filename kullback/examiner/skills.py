"""The review skill: what one review per Task reads and files (D320).

A skill is text the harness puts in the prompt; this one says nothing a gate does not enforce and
names no domain, tool or value.
"""

from __future__ import annotations

REVIEW_SKILL_NAME = "review"

REVIEW_SKILL = """Review skill. For each Task read, in order: the Intent facts and their user turns; the
Verifier's expected end states, each cell with its source (a user turn, a policy clause, a tool result,
or unchanged); the forbidden list; the conduct rules; the fact and question checks with their because;
the verdict rows of the Runs. Ask of each cell whether its source says that value, of each check whether
a fact asks for it, and of each failing Run whether the Run or the Verifier is wrong. File what you find
as one review: the rows that show it and the smallest edit that fixes it."""

__all__ = ["REVIEW_SKILL", "REVIEW_SKILL_NAME"]
