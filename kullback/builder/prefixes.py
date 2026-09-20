from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from kullback.runner.records import as_dict, content_hash

SELECTOR_VERSION = "prefix-v2"
EVIDENCE_ONLY = "evidence_only"


@dataclass(frozen=True)
class PrefixSelection:
    cohort_id: str
    parent_trace_id: str
    parent_trace_hash: str
    parent_raw_hash: str
    sim_index: Optional[int]
    source: str
    ingest_version: str
    turn_end: int
    retained_turns: tuple
    retained_calls: tuple
    last_result_ptr: Any
    standing: str


@dataclass(frozen=True)
class Withheld:
    parent_trace_id: str
    parent_trace_hash: str
    reason: str


def _withheld(trace, reason):
    known = trace.hash if trace.hash else ""
    return Withheld(parent_trace_id=trace.trace_id, parent_trace_hash=known, reason=reason)


def _heldout(trace, heldout_ids):
    if trace.trace_id in heldout_ids:
        return _withheld(trace, "heldout_parent")
    return None


def _empty_reason(turns, calls):
    if not turns:
        return "empty_trace"
    if not calls:
        return "no_resolved_call"
    return None


def _turn_map(turns, raw_hash, sim_index):
    mapping = {}
    for position, item in enumerate(turns):
        rp = item.raw_ptr
        if rp is None or not isinstance(rp.msg_index, int):
            return (None, "ambiguous_provenance")
        if rp.file_hash != raw_hash or rp.sim_index != sim_index:
            return (None, "foreign_pointer")
        if rp.msg_index in mapping:
            return (None, "ambiguous_provenance")
        mapping[rp.msg_index] = position
    return (mapping, None)


def _site_reason(calls, turn_map, turns):
    for call in calls:
        site = turn_map.get(call.raw_ptr.msg_index)
        if site is None:
            return "ambiguous_provenance"
        if turns[site].role != "assistant":
            return "malformed_pointer"
    return None


def _provenance_reason(calls, raw_hash, sim_index, trace_id):
    seen = set()
    for call in calls:
        raw = call.raw_ptr
        if raw is None or not isinstance(raw.msg_index, int):
            return "ambiguous_provenance"
        if raw.file_hash != raw_hash or raw.sim_index != sim_index:
            return "foreign_pointer"
        if raw.msg_index in seen:
            return "ambiguous_provenance"
        seen.add(raw.msg_index)
        flawed = _call_owner_reason(call, trace_id)
        if flawed is not None:
            return flawed
    return None


def _flag_reason(call):
    if call.truncated:
        return "truncated_result"
    if call.has_result is not True or call.resolved is not True:
        return "malformed_pointer"
    if call.result_ptr is None:
        return "malformed_pointer"
    return None


def _parent_reason(call, raw_hash, sim_index):
    rp = call.result_ptr
    if rp.file_hash != raw_hash or rp.sim_index != sim_index:
        return "foreign_pointer"
    if not isinstance(rp.msg_index, int):
        return "ambiguous_provenance"
    return None


def _answer_site(call, turn_map, turns):
    rp = call.result_ptr
    site = turn_map.get(rp.msg_index)
    if site is None:
        return "unordered_pointer"
    if turns[site].role != "user":
        return "malformed_pointer"
    if rp.msg_index <= call.raw_ptr.msg_index:
        return "unordered_pointer"
    return None


def _id_seen(call, seen):
    ident = call.id
    if ident is None:
        return None
    if ident in seen:
        return "ambiguous_provenance"
    seen.add(ident)
    return None


def _scope_reason(turns, calls, raw_hash, sim_index, trace_id):
    turn_map, vague = _turn_map(turns, raw_hash, sim_index)
    if vague is not None:
        return (None, vague)
    flawed = _provenance_reason(calls, raw_hash, sim_index, trace_id)
    if flawed is not None:
        return (None, flawed)
    sited = _site_reason(calls, turn_map, turns)
    if sited is not None:
        return (None, sited)
    return (turn_map, None)


def _pointer_reason(call, raw_hash, sim_index, turn_map, turns):
    reason = _flag_reason(call)
    if reason is not None:
        return reason
    reason = _parent_reason(call, raw_hash, sim_index)
    if reason is not None:
        return reason
    return _answer_site(call, turn_map, turns)


def _collect_valid(calls, raw_hash, sim_index, turn_map, turns):
    valid = []
    seen_answers = set()
    seen_ids = set()
    for position, call in enumerate(calls):
        if call.has_result is True and call.resolved is True:
            reason = _pointer_reason(call, raw_hash, sim_index, turn_map, turns)
            if reason is not None:
                return ([], reason)
            key = call.result_ptr.msg_index
            if key in seen_answers:
                return ([], "ambiguous_provenance")
            seen_answers.add(key)
            reason = _id_seen(call, seen_ids)
            if reason is not None:
                return ([], reason)
            valid.append(position)
    return (valid, None)


def _boundary_position(calls, valid):
    best = valid[0]
    for position in valid[1:]:
        now = calls[position].result_ptr.msg_index
        top = calls[best].result_ptr.msg_index
        if now > top:
            best = position
        elif now == top and calls[position].raw_ptr.msg_index > calls[best].raw_ptr.msg_index:
            best = position
    return best


def _retained_reason(calls, valid, turn_end):
    inside = set(valid)
    for position, call in enumerate(calls):
        if call.raw_ptr.msg_index <= turn_end and position not in inside:
            if isinstance(call.args, dict) and call.args.get("commands") == []:
                return "orphan_marker"
            return "interior_unresolved"
    return None


def _assemble(trace, turns, calls, valid, raw_hash, sim_index, turn_map, content):
    best = _boundary_position(calls, valid)
    top = calls[best]
    turn_end = top.result_ptr.msg_index
    if turn_end not in turn_map:
        return _withheld(trace, "unordered_pointer")
    gap = _retained_reason(calls, valid, turn_end)
    if gap is not None:
        return _withheld(trace, gap)
    kept_turns = tuple(position for position, item in enumerate(turns) if item.raw_ptr.msg_index <= turn_end)
    kept_calls = tuple(position for position, item in enumerate(calls) if item.raw_ptr.msg_index <= turn_end)
    return PrefixSelection(
        cohort_id=_cohort_id(content, turn_end),
        parent_trace_id=trace.trace_id,
        parent_trace_hash=content,
        parent_raw_hash=raw_hash,
        sim_index=sim_index,
        source=trace.source,
        ingest_version=trace.ingest_version,
        turn_end=turn_end,
        retained_turns=kept_turns,
        retained_calls=kept_calls,
        last_result_ptr=top.result_ptr.model_copy(deep=True),
        standing=EVIDENCE_ONLY,
    )


def _trace_lists(trace):
    turns = trace.turns if trace.turns else []
    calls = trace.tool_calls if trace.tool_calls else []
    return (turns, calls)


def _trace_coords(trace):
    root = trace.raw_ptr
    if root is None:
        return (None, None, "ambiguous_provenance")
    if root.file_hash != trace.raw_hash:
        return (None, None, "foreign_pointer")
    return (trace.raw_hash, root.sim_index, None)


def _call_owner_reason(call, trace_id):
    owner = call.trace_id
    if owner is not None and owner != trace_id:
        return "foreign_pointer"
    return None


def _verified_content(trace):
    body = as_dict(trace)
    body["hash"] = ""
    computed = content_hash(body)
    if trace.hash and trace.hash != computed:
        return (None, "stale_hash")
    return (computed, None)


def _cohort_id(parent_content, turn_end):
    payload = SELECTOR_VERSION + "|" + parent_content + "|" + str(turn_end)
    return "prefix-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def select_prefix(trace, heldout_ids=frozenset()):
    stop = _heldout(trace, heldout_ids)
    if stop is not None:
        return stop
    turns, calls = _trace_lists(trace)
    empty = _empty_reason(turns, calls)
    if empty is not None:
        return _withheld(trace, empty)
    content, changed = _verified_content(trace)
    if changed is not None:
        return _withheld(trace, changed)
    raw_hash, sim_index, rooted = _trace_coords(trace)
    if rooted is not None:
        return _withheld(trace, rooted)
    turn_map, scoped = _scope_reason(turns, calls, raw_hash, sim_index, trace.trace_id)
    if scoped is not None:
        return _withheld(trace, scoped)
    valid, broken = _collect_valid(calls, raw_hash, sim_index, turn_map, turns)
    if broken is not None:
        return _withheld(trace, broken)
    if not valid:
        return _withheld(trace, "no_resolved_call")
    return _assemble(trace, turns, calls, valid, raw_hash, sim_index, turn_map, content)
