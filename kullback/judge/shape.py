"""The shape of a judge item (D328), with no model in reach: the Verdict reads it, so no provider here."""

from __future__ import annotations

from kullback.runner.records import Atom

FINAL_ANSWER = "final_answer"


def is_item(atom: Atom) -> bool:
    """A judge atom in the item shape: a judge flag and a question on its target."""
    return bool(atom.judge and isinstance(atom.target, dict) and atom.target.get("question"))


def anchor_of(target: dict) -> dict:
    """The anchor as {answer, accepted, reject}; a plain string is the answer alone."""
    anchor = target.get("anchor")
    if isinstance(anchor, dict):
        return {"answer": str(anchor.get("answer") or ""),
                "accepted": [str(a) for a in anchor.get("accepted") or []],
                "reject": [str(r) for r in anchor.get("reject") or []]}
    return {"answer": "" if anchor is None else str(anchor), "accepted": [], "reject": []}


def evidence_refs(target: dict) -> list[str]:
    refs = target.get("evidence") or [FINAL_ANSWER]
    return [str(r) for r in (refs if isinstance(refs, list) else [refs])]
