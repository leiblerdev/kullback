"""Trust gate 1: every check rests on the Intent or the policy, every Intent fact has a check or a gap.

`valid_because` is the one implementation of the because rule; it replaces the copy the intentv
experiment carried.
"""

from __future__ import annotations

import re
from typing import Iterable

# The shortest verbatim Intent quote that counts as grounding a check.
MIN_QUOTE = 12
_SECTION = "section:"


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _quotes(because: str, intent: str) -> bool:
    flat = _norm(intent)
    for span in re.split(r"[.!?\n]+", because):
        span = span.strip("\"' ").strip()
        if len(span) >= MIN_QUOTE and span in flat:
            return True
    return False


def _names_section(because: str, sections: Iterable[str]) -> bool:
    for section in sections:
        name = _norm(section)
        name = name[len(_SECTION):] if name.startswith(_SECTION) else name
        if name and name in because:
            return True
    return False


def valid_because(because: object, intent: str, sections: Iterable[str] = ()) -> bool:
    """A because quotes at least MIN_QUOTE characters of the Intent, or names a given policy section."""
    text = _norm(because)
    return bool(text) and (_quotes(text, intent) or _names_section(text, sections))


def coverage(spec) -> list[str]:
    """The ids of Intent facts no check names and the Spec does not list as a gap."""
    named = {fact_id for check in spec.checks for fact_id in check.fact_ids}
    listed = set(spec.gaps)
    return [fact.id for fact in spec.intent.facts if fact.id not in named and fact.id not in listed]


def ground_spec(spec, policy_sections: Iterable[str] = ()) -> list[str]:
    """Every reason the Spec is not grounded; empty means grounded. Never raises."""
    try:
        sections = list(policy_sections or ())
        intent = spec.intent.text
        refusals = [f"check {check.id}: because quotes neither the Intent nor a policy section"
                    for check in spec.checks if not valid_because(check.because, intent, sections)]
        refusals += [f"fact {fact_id}: no check names it and it is not listed as a gap"
                     for fact_id in coverage(spec)]
        return refusals
    except Exception as exc:  # a malformed Spec is refused, never raised
        return [f"spec unreadable: {type(exc).__name__}"]
