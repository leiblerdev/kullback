"""The shape of a judge item (D328), with no model in reach: the Verdict reads it, so no provider here."""

from __future__ import annotations

import re

from kullback.runner.atom_context import _text_of, is_transfer
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
    refs = target.get("evidence") or [ALL_ASSISTANT]
    return [str(r) for r in (refs if isinstance(refs, list) else [refs])]


def _window(run: Run) -> list:
    """The Run's events up to its first transfer or hand-off call: what the agent said while it
    still served the user. The message that makes the call is inside; anything after it is not."""
    out = []
    for event in run.events:
        if event.type == "tool_call" and is_transfer((event.payload or {}).get("name")):
            break
        out.append(event)
    return out


def assistant_turns(run: Run) -> list[tuple[int, str]]:
    """Every assistant message with text inside the window, as (event index, text), in order."""
    out = []
    for event in _window(run):
        if event.type == "model_call":
            text = _text_of(event.payload or {}).strip()
            if text:
                out.append((event.idx, text))
    return out


def _turns_after_call(run: Run, tool: str) -> list[tuple[int, str]]:
    """The assistant messages after the first result of `tool`, to the end of the window."""
    seen, out = False, []
    for event in _window(run):
        payload = event.payload or {}
        if event.type == "tool_result" and payload.get("name") == tool:
            seen = True
        elif seen and event.type == "model_call":
            text = _text_of(payload).strip()
            if text:
                out.append((event.idx, text))
    return out


def resolve(run: Run, ref: str) -> list[tuple[str, int, str]]:
    """The turns one evidence ref names, as (label, event index, text).

    Refs are written before any Run exists, so none names an event index. Every ref reads inside the
    window (up to the first transfer call): `assistant_turns` (the default) is every assistant
    message; `final_answer` is the last; `assistant:<k>` is the k-th (1-based, negative from the
    end); `after_call:<tool>` is what the Candidate said after that tool first answered.
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
