"""The shape of a judge item (D328), with no model in reach: the Verdict reads it, so no provider here."""

from __future__ import annotations

import re

from kullback.runner.atom_context import _text_of
from kullback.runner.records import Atom, Run

FINAL_ANSWER = "final_answer"
ALL_ASSISTANT = "assistant_turns"
NO_EVIDENCE = "no evidence"


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


def assistant_turns(run: Run) -> list[tuple[int, str]]:
    """Every assistant message of a Run with text, as (event index, text), in order."""
    out = []
    for event in run.events:
        if event.type == "model_call":
            text = _text_of(event.payload or {}).strip()
            if text:
                out.append((event.idx, text))
    return out


def _turns_after_call(run: Run, tool: str) -> list[tuple[int, str]]:
    """The assistant messages after the first result of `tool`, up to the next user turn."""
    seen, out = False, []
    for event in run.events:
        payload = event.payload or {}
        if event.type == "tool_result" and payload.get("name") == tool:
            seen = True
        elif seen and event.type == "user_turn" and out:
            break
        elif seen and event.type == "model_call":
            text = _text_of(payload).strip()
            if text:
                out.append((event.idx, text))
    return out


def resolve(run: Run, ref: str) -> list[tuple[str, int, str]]:
    """The turns one evidence ref names, as (label, event index, text).

    Refs are written before any Run exists, so none names an event index: `final_answer` is the
    last assistant message; `assistant_turns` is every one; `assistant:<k>` is the k-th (1-based,
    negative from the end); `after_call:<tool>` is what the Candidate said after that tool answered.
    """
    turns = assistant_turns(run)
    if ref == FINAL_ANSWER:
        return [("final answer", *turns[-1])] if turns else []
    if ref == ALL_ASSISTANT:
        return [(f"assistant turn {i + 1}", idx, text) for i, (idx, text) in enumerate(turns)]
    kind, _, arg = ref.partition(":")
    if kind == "assistant" and re.fullmatch(r"-?\d+", arg or ""):
        k = int(arg)
        pos = k - 1 if k > 0 else len(turns) + k
        return [(f"assistant turn {pos + 1}", *turns[pos])] if 0 <= pos < len(turns) else []
    if kind == "after_call" and arg:
        return [(f"after {arg}", idx, text) for idx, text in _turns_after_call(run, arg)]
    return []


def has_evidence(run: Run, target: dict) -> bool:
    """Whether the Run has any turn the item's evidence names; with none the item scores 0 by code."""
    return any(resolve(run, ref) for ref in evidence_refs(target))
