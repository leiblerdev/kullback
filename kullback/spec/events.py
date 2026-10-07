"""The Spec's and the rulings' events on the workdir bus (DESIGN.md rule 4), published with Bus.publish_custom.

Payloads carry ids, counts and codes, never free text: a reason is one of REASON_CODES, a quote is
named by where it came from ([because cN], [turn N]), and an excerpt is kept only as its length and a
hash, so the bus can be read by any report without carrying what a user or the Examiner wrote.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional

from kullback.spec.rulings import RULING_KINDS

SPEC_WRITTEN = "spec.written"
SPEC_REPAIRED = "spec.repaired"
SPEC_DEFENDED = "spec.defended"
SPEC_REPAIR_DEFERRED = "spec.repair_deferred"
RULING_FILED = "ruling.filed"
RULING_ANSWERED = "ruling.answered"
RULING_CLOSED = "ruling.closed"
REVIEW_ROUND = "review.round"
TASK_SET_ASIDE = "task.set_aside"
EVENT_NAMES = (SPEC_WRITTEN, SPEC_REPAIRED, SPEC_DEFENDED, SPEC_REPAIR_DEFERRED, RULING_FILED, RULING_ANSWERED,
               RULING_CLOSED, REVIEW_ROUND, TASK_SET_ASIDE)

# Every reason a payload may carry: a ruling's kind and code (spec/rulings.py), never its words, which
# stay in the ruling's file.
RULING_CODES = (*RULING_KINDS, *(code for codes in RULING_KINDS.values() for code in codes), "other")
QUOTE_MISSING = "quote_missing"
NO_AGREEMENT = "no_agreement"
# The writer kept no check twice running (spec/stage.py).
EMPTY_SPEC = "empty_spec"
# No Run could pass the Spec, still so after one send-back (spec/compile.py UNSATISFIABLE).
UNSATISFIABLE_SPEC = "unsatisfiable_spec"
REASON_CODES = (QUOTE_MISSING, *RULING_CODES, NO_AGREEMENT, EMPTY_SPEC, UNSATISFIABLE_SPEC)


def code_of(value: Any) -> str:
    """A reason as its code: a listed code stays, anything else is `other`."""
    return value if value in REASON_CODES else "other"


def excerpt_ref(excerpt: Optional[str]) -> Optional[dict]:
    """An excerpt as its length and a hash: which text, never the text."""
    if not excerpt:
        return None
    return {"chars": len(excerpt), "sha": hashlib.sha256(excerpt.encode("utf-8")).hexdigest()[:12]}


def _ids(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in values or ()]


def spec_written(bus: Any, task_id: str, version: int):
    return bus.publish_custom(SPEC_WRITTEN, {"task_id": task_id, "version": int(version)})


def spec_repaired(bus: Any, task_id: str, version: int, check_ids: Iterable[str]):
    return bus.publish_custom(SPEC_REPAIRED, {"task_id": task_id, "version": int(version),
                                              "check_ids": _ids(check_ids)})


def spec_defended(bus: Any, task_id: str, check_ids: Iterable[str], reason: str):
    return bus.publish_custom(SPEC_DEFENDED, {"task_id": task_id, "check_ids": _ids(check_ids),
                                              "reason": code_of(reason)})


def spec_repair_deferred(bus: Any, task_id: str, number: Any):
    return bus.publish_custom(SPEC_REPAIR_DEFERRED, {"task_id": task_id, "number": number})


def ruling_payload(ruling: Any) -> dict:
    """A ruling as the bus carries it: ids, kind, code, blocking and round; its sentences stay in its file."""
    return {"task_id": ruling.task_id, "number": int(ruling.number), "round": int(ruling.round),
            "item": ruling.item, "kind": ruling.kind, "code": code_of(ruling.code), "blocking": bool(ruling.blocking)}


def ruling_filed(bus: Any, payload: dict):
    return bus.publish_custom(RULING_FILED, dict(payload))


def ruling_answered(bus: Any, task_id: str, number: int, action: str, changed: Iterable[str] = ()):
    return bus.publish_custom(RULING_ANSWERED, {"task_id": task_id, "number": int(number), "action": action,
                                                "changed": _ids(changed)})


def ruling_closed(bus: Any, task_id: str, number: int, kept_open: bool):
    return bus.publish_custom(RULING_CLOSED, {"task_id": task_id, "number": int(number), "kept_open": bool(kept_open)})


def review_round(bus: Any, round_number: int, side: str, counts: dict):
    """One round of the review loop: whose round (examiner or writer) and its counts, numbers only."""
    return bus.publish_custom(REVIEW_ROUND, {"round": int(round_number), "side": side, "counts": dict(counts)})


def task_set_aside(bus: Any, task_id: str, reason: str, rounds: int):
    return bus.publish_custom(TASK_SET_ASIDE, {"task_id": task_id, "reason": code_of(reason),
                                               "rounds": int(rounds)})


__all__ = ["EMPTY_SPEC", "EVENT_NAMES", "NO_AGREEMENT", "QUOTE_MISSING", "REASON_CODES", "REVIEW_ROUND",
           "RULING_ANSWERED", "RULING_CLOSED", "RULING_CODES", "RULING_FILED", "SPEC_DEFENDED", "SPEC_REPAIRED",
           "SPEC_REPAIR_DEFERRED", "SPEC_WRITTEN", "TASK_SET_ASIDE", "UNSATISFIABLE_SPEC", "code_of", "excerpt_ref",
           "review_round", "ruling_answered", "ruling_closed", "ruling_filed", "ruling_payload", "spec_defended",
           "spec_repair_deferred", "spec_repaired", "spec_written", "task_set_aside"]
