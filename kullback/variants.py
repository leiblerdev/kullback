"""Second paths synthesised from a Run's own calls, code only (D199).

Check 5 of the D79 suite asks whether the Verifier accepts a different way of reaching the same End
state. D189 buys that second way as fresh Runs, and on every live corpus a large share of the Tasks
that stood alone spent the whole re-roll cap and still stood alone: many Tasks have one path to
their End state and no Run anybody buys will find a second. What the check needs is not another
sample of the same model, it is a call path that differs and lands in the same place, and the
Reference Run already carries enough structure to write one down.

Three rewrites, each of which leaves the world where it was:

  reorder     two adjacent calls neither of which reads a row the other wrote, swapped;
  insert_read a read the Run already made, made again just before a write;
  drop_read   a read whose answer no later call's arguments use, left out.

Independence is judged on the rows the calls touched, read off the recorded arguments and the
recorded results: a value under a key named like an identifier is a row id, at any depth, and a
call's rows are the ids in its arguments and in what it was answered with. A call whose result
carried an error wrote nothing (D67), so it touches rows and writes none. Nothing here knows a
schema, a table or a tool: an id-shaped key is the whole of the vocabulary, so the same three
rewrites apply to any corpus.

A rewrite is a proposal and never a fact. The caller replays each one through the Environment from
the Task's Starting state and keeps only the ones that reach the Reference's End state under the
D111 rule, so a rewrite that was wrong about independence is thrown away by the replay rather than
believed. A Run with no independent adjacent pair, no read to insert and no read to drop yields
nothing, and that is a Task whose path is single by structure, which is an answer and not a failure.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from kullback.runner.records import plain

REORDER = "reorder"
INSERT_READ = "insert_read"
DROP_READ = "drop_read"
KINDS = (REORDER, INSERT_READ, DROP_READ)
# How many rewrites one Task offers the replay. Each one is a replay of the Run through the built
# tools, code only and no model call, so the cap is about the round's wall clock and not its bill.
MAX_VARIANTS = 6
# Where a value stops being a row id and starts being prose. An id is short; a sentence that happens
# to sit under a key ending in _id is not a row this rule should reason about.
MAX_ID_CHARS = 120


def _is_id_key(key: str) -> bool:
    """Is this argument or result key the name of a row identifier?

    The same shape `verifier_suite._entity` keys a write on, applied at any depth: `id`, `ids`, or a
    name ending in `_id` or `_ids`. It names no column of any customer's schema.
    """
    name = str(key).lower()
    return name in ("id", "ids") or name.endswith(("_id", "_ids"))


def _scalars(value: Any) -> list[str]:
    """The scalar values inside a value, as text, however deeply they are nested."""
    value = plain(value)
    if isinstance(value, dict):
        return [text for item in value.values() for text in _scalars(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _scalars(item)]
    if value is None or isinstance(value, bool):
        return []
    return [str(value)]


def row_ids(value: Any) -> frozenset[str]:
    """Every row id inside a value: the scalars under a key named like an identifier, at any depth."""
    out: set[str] = set()
    _collect(plain(value), out)
    return frozenset(out)


def _collect(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_id_key(key):
                out.update(text for text in _scalars(item) if len(text) <= MAX_ID_CHARS)
            _collect(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, out)


def call_rows(call: dict) -> frozenset[str]:
    """The rows one call touched: the ids in its arguments and in the answer it was given."""
    return row_ids(call.get("args")) | row_ids(call.get("result"))


def call_writes(call: dict, write_tools: Iterable[str]) -> frozenset[str]:
    """The rows one call wrote: none unless it is a write tool that was not refused (D67)."""
    if call.get("name") not in set(write_tools) or call.get("error"):
        return frozenset()
    return call_rows(call)


def independent(first: dict, second: dict, write_tools: Iterable[str]) -> bool:
    """Neither call reads a row the other wrote, so the world does not care which went first.

    Two reads are always independent, since neither wrote anything. Two writes on the same row are
    not, and a read of a row a write touched is not. The judgement is made on what the recording
    shows the calls touched, which is why the rewrite is replayed before it is believed.
    """
    tools = set(write_tools)
    writes_first, writes_second = call_writes(first, tools), call_writes(second, tools)
    return not (writes_first & call_rows(second)) and not (writes_second & call_rows(first))


def _values_of(call: dict) -> frozenset[str]:
    """The values a call's answer carries, as text: what a later call could have taken from it."""
    return frozenset(text for text in _scalars(call.get("result")) if text)


def _argument_values(call: dict) -> frozenset[str]:
    return frozenset(text for text in _scalars(call.get("args")) if text)


def _is_read(call: dict, write_tools: Iterable[str]) -> bool:
    return call.get("name") not in set(write_tools)


def _same_requestor(first: dict, second: dict) -> bool:
    """Only two calls the same speaker made may be swapped: the other order is another conversation."""
    return (first.get("requestor") or "assistant") == (second.get("requestor") or "assistant")


def _call(call: dict, *, call_id: Optional[str] = None) -> dict:
    """One call as the replay wants it: who asked, what was asked, under which id, in which turn.

    `turn` is where the call sat in the Run's own transcript, which the rewrite carries with it: a
    variant varies the call path inside the conversation the Run had, so what the agent asked and
    what it told the user are the Run's own and only the calls move.
    """
    return {"id": call_id if call_id is not None else call.get("id"),
            "name": str(call.get("name") or ""),
            "args": dict(call.get("args") or {}),
            "requestor": call.get("requestor") or "assistant",
            "turn": int(call.get("turn") or 0)}


def reorderings(calls: list[dict], write_tools: Iterable[str]) -> list[dict]:
    """Every adjacent independent pair of one speaker's calls, swapped, one variant per pair."""
    out = []
    for index in range(len(calls) - 1):
        first, second = calls[index], calls[index + 1]
        if not _same_requestor(first, second) or not independent(first, second, write_tools):
            continue
        rewritten = [_call(c) for c in calls]
        rewritten[index], rewritten[index + 1] = rewritten[index + 1], rewritten[index]
        # The turns swap with the calls: a call that moved past a spoken turn moved in the
        # conversation too, and a rewrite whose calls sit in their old turns is the Run again.
        rewritten[index]["turn"], rewritten[index + 1]["turn"] = (
            rewritten[index + 1]["turn"], rewritten[index]["turn"])
        out.append({"kind": REORDER, "at": index,
                    "calls": rewritten,
                    "why": f"{first.get('name')} and {second.get('name')} touch no row the other wrote"})
    return out


def inserted_reads(calls: list[dict], write_tools: Iterable[str]) -> list[dict]:
    """A read the Run already made, made once more just before a write.

    The read is one the Run itself made, so the row it reads is a row the Run touched, and a read
    changes nothing, so the writes that follow it are the writes that followed before.
    """
    tools = set(write_tools)
    out = []
    for index, call in enumerate(calls):
        if _is_read(call, tools):
            continue
        earlier = [c for c in calls[:index] if _is_read(c, tools) and _same_requestor(c, call)]
        source = earlier[-1] if earlier else next(
            (c for c in calls if _is_read(c, tools) and _same_requestor(c, call)), None)
        if source is None:
            continue
        again = dict(_call(source, call_id=f"{source.get('id') or 'call'}-again-{index}"),
                     turn=int(call.get("turn") or 0))
        rewritten = [_call(c) for c in calls]
        rewritten.insert(index, again)
        out.append({"kind": INSERT_READ, "at": index, "calls": rewritten,
                    "why": f"{source.get('name')} is read again before {call.get('name')}"})
    return out


def dropped_reads(calls: list[dict], write_tools: Iterable[str]) -> list[dict]:
    """A read left out, when no later call's arguments hold a value that read answered with.

    A later call whose arguments carry one of the values is a call that could have taken it from
    here, which is what a dependency looks like from outside the agent's head. A read nothing later
    used is a read the Run did not need, and leaving it out is another way to the same End state.
    """
    tools = set(write_tools)
    out = []
    for index, call in enumerate(calls):
        if not _is_read(call, tools) or call.get("error"):
            continue
        answered = _values_of(call)
        used = any(answered & _argument_values(later) for later in calls[index + 1:])
        if used or not answered:
            continue
        rewritten = [_call(c) for pos, c in enumerate(calls) if pos != index]
        if not rewritten:
            continue
        out.append({"kind": DROP_READ, "at": index, "calls": rewritten,
                    "why": f"no later call's arguments hold what {call.get('name')} answered with"})
    return out


def variants(calls: list[dict], write_tools: Iterable[str], limit: int = MAX_VARIANTS) -> list[dict]:
    """The rewrites of one Run's calls, at most `limit`, taken a kind at a time.

    Round robin over the three kinds rather than the first `limit` of one of them: a Run with twenty
    reorderings and one droppable read would otherwise offer the replay twenty near-identical paths
    and never the one that differs most.
    """
    calls = [c for c in calls if c.get("name")]
    if len(calls) < 2:
        return []
    by_kind = {REORDER: reorderings(calls, write_tools),
               INSERT_READ: inserted_reads(calls, write_tools),
               DROP_READ: dropped_reads(calls, write_tools)}
    out: list[dict] = []
    depth = 0
    while len(out) < limit and any(depth < len(rows) for rows in by_kind.values()):
        for kind in KINDS:
            rows = by_kind[kind]
            if depth < len(rows) and len(out) < limit:
                out.append(rows[depth])
        depth += 1
    return out


def _spoken(event: Any) -> Optional[dict]:
    """One event as a spoken turn, or None where the event is not one."""
    payload = dict(getattr(event, "payload", None) or {})
    if event.type == "model_call":
        reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else payload
        return {"role": "assistant", "content": str(reply.get("content") or "")}
    if event.type == "user_turn":
        return {"role": "user", "content": str(payload.get("content") or payload.get("text") or "")}
    return None


def transcript(run: Any) -> list[dict]:
    """The Run's spoken turns in order, which every rewrite of it carries unchanged.

    The End state has two halves (D43) and this rule is about the half made of calls: what the agent
    asked and what it told the user belong to the Run and are not the rewrite's to invent. Carrying
    them is also what makes a rewrite scoreable, since the atoms over questions and over stated facts
    read the transcript and a Run without one fails them however well its writes landed.
    """
    return [turn for turn in (_spoken(event) for event in getattr(run, "events", ()) or ())
            if turn is not None]


def call_results(run: Any) -> list[dict]:
    """The Run's calls with the answer each was given and the turn each sat in.

    `verifier_suite.run_calls` carries the arguments and the error and not the result, and the rows
    a read touched are mostly in what it answered with, so the pairing is done once more here over
    the same events: a tool_result matches the last tool_call whose id it names, or the last one.
    """
    calls: list[dict] = []
    turns = 0
    for event in getattr(run, "events", ()) or ():
        payload = dict(getattr(event, "payload", None) or {})
        if _spoken(event) is not None:
            turns += 1
        elif event.type == "tool_call":
            calls.append({"i": len(calls), "id": payload.get("id"),
                          "name": str(payload.get("name") or ""),
                          "args": dict(payload.get("args") or payload.get("arguments") or {}),
                          "requestor": payload.get("requestor") or "assistant",
                          "turn": max(turns - 1, 0), "result": None, "error": None})
        elif event.type == "tool_result":
            for call in reversed(calls):
                if payload.get("id") in (None, call["id"]):
                    call["result"] = payload.get("result")
                    call["error"] = payload.get("error")
                    break
    return calls


def _dumps(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, default=str)


def signature(calls: Iterable[dict]) -> str:
    """One rewrite's identity: the calls in order with their arguments, so a repeat is not replayed twice."""
    return _dumps([[c.get("name"), c.get("requestor") or "assistant", c.get("args") or {}] for c in calls])


__all__ = ["DROP_READ", "INSERT_READ", "KINDS", "MAX_VARIANTS", "REORDER", "call_results", "call_rows",
           "call_writes", "dropped_reads", "independent", "inserted_reads", "reorderings", "row_ids",
           "signature", "transcript", "variants"]
