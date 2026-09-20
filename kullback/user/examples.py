from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal, Optional

from pydantic import ConfigDict, StrictInt, StrictStr, ValidationError

from kullback.runner.records import RawPtr, Record, Trace, Turn

Situation = Literal["unknown_fact", "company_only", "agent_refuses", "confirmation", "giving_up"]

MASKED = "<MASKED>"


class ExampleSpan(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    recording_id: StrictStr
    situation: Situation
    turn_idx: StrictInt
    start: StrictInt
    end: StrictInt


class ExampleCandidate(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    recording_id: str
    situation: Situation
    masked_excerpt: str
    turn_idx: int
    start: int
    end: int
    source: RawPtr


class ExampleWithheld(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    recording_id: str
    reason: str


def _raw_id(raw: Any) -> str:
    if isinstance(raw, ExampleSpan):
        return raw.recording_id
    if isinstance(raw, Mapping):
        rid = raw.get("recording_id")
        if isinstance(rid, str):
            return rid
    return "unknown"


def _offset_reason(exc: ValidationError, raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return "malformed_span"
    locs: list[tuple[str, ...]] = []
    for detail in exc.errors(include_url=False):
        if detail.get("type") != "int_type":
            return "malformed_span"
        loc = tuple(detail.get("loc", ()))
        if loc != ("turn_idx",) and loc != ("start",) and loc != ("end",):
            return "malformed_span"
        locs.append(loc)
    if not locs:
        return "malformed_span"
    if ("turn_idx",) in locs:
        return "unknown_turn"
    return "bad_bounds"


def _coerce(raw: Any) -> tuple[Optional[ExampleSpan], Optional[str]]:
    if isinstance(raw, ExampleSpan):
        return (raw, None)
    if not isinstance(raw, Mapping):
        return (None, "malformed_span")
    try:
        return (ExampleSpan.model_validate(raw), None)
    except ValidationError as exc:
        return (None, _offset_reason(exc, raw))


def _inventory_values(entry: Any) -> Optional[list[str]]:
    if entry is None:
        return None
    if isinstance(entry, str):
        return None
    if not isinstance(entry, (list, tuple, set, frozenset)):
        return None
    values = list(entry)
    for value in values:
        if not isinstance(value, str):
            return None
        if value == "":
            return None
    return values


def _lookup_trace(traces_by_id: Mapping[str, Trace], rid: str) -> tuple[Optional[Trace], Optional[str]]:
    trace = traces_by_id.get(rid)
    if trace is None:
        return (None, "missing_trace")
    if trace.trace_id != rid:
        return (None, "trace_mismatch")
    return (trace, None)


def _lookup_values(
    known_values_by_id: Mapping[str, Any], rid: str
) -> tuple[Optional[list[str]], Optional[str]]:
    if rid not in known_values_by_id:
        return (None, "unknown_mask_inventory")
    values = _inventory_values(known_values_by_id.get(rid))
    if values is None:
        return (None, "bad_inventory")
    for value in values:
        if value in MASKED:
            return (None, "token_collision")
    return (values, None)


def _find_turn(turns: list[Turn], idx: int) -> tuple[Optional[Turn], Optional[str]]:
    matches = [turn for turn in turns if turn.idx == idx]
    if not matches:
        return (None, "unknown_turn")
    if len(matches) > 1:
        return (None, "ambiguous_turn")
    return (matches[0], None)


def _partial(lo: int, size: int, start: int, end: int) -> bool:
    hi = lo + size
    if lo >= start and hi <= end:
        return False
    return lo < end and hi > start


def _straddles(content: str, start: int, end: int, values: list[str]) -> bool:
    for value in values:
        pos = content.find(value)
        while pos != -1:
            if _partial(pos, len(value), start, end):
                return True
            pos = content.find(value, pos + 1)
    return False


def _occurrences(piece: str, value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = piece.find(value)
    while pos != -1:
        out.append((pos, pos + len(value)))
        pos = piece.find(value, pos + 1)
    return out


def _extend(merged: list[list[int]], lo: int, hi: int) -> None:
    if not merged or lo >= merged[-1][1]:
        merged.append([lo, hi])
    elif hi > merged[-1][1]:
        merged[-1][1] = hi


def _merge(spans: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for lo, hi in spans:
        _extend(merged, lo, hi)
    return merged


def _render(piece: str, merged: list[list[int]]) -> str:
    out: list[str] = []
    pos = 0
    for lo, hi in merged:
        out.append(piece[pos:lo])
        out.append(MASKED)
        pos = hi
    out.append(piece[pos:])
    return "".join(out)


def _mask_all(piece: str, values: list[str]) -> str:
    spans: list[tuple[int, int]] = []
    for value in sorted(set(values)):
        spans.extend(_occurrences(piece, value))
    spans.sort()
    return _render(piece, _merge(spans))


def _mask_span(
    turn: Turn, span: ExampleSpan, values: list[str]
) -> tuple[Optional[ExampleCandidate], Optional[str]]:
    content = turn.content
    if not isinstance(content, str):
        return (None, "bad_bounds")
    if not 0 <= span.start < span.end <= len(content):
        return (None, "bad_bounds")
    piece = content[span.start : span.end]
    if piece == "":
        return (None, "bad_bounds")
    if _straddles(content, span.start, span.end, values):
        return (None, "partial_boundary")
    masked = _mask_all(piece, values)
    for value in values:
        if value in masked:
            return (None, "residual_known_value")
    return (
        ExampleCandidate(
            recording_id=span.recording_id,
            situation=span.situation,
            masked_excerpt=masked,
            turn_idx=span.turn_idx,
            start=span.start,
            end=span.end,
            source=turn.raw_ptr.model_copy(deep=True),
        ),
        None,
    )


def _decide_one(
    traces_by_id: Mapping[str, Trace], known_values_by_id: Mapping[str, Any], span: ExampleSpan
) -> tuple[Optional[ExampleCandidate], Optional[ExampleWithheld]]:
    rid = span.recording_id
    trace, reason = _lookup_trace(traces_by_id, rid)
    if reason is not None:
        return (None, ExampleWithheld(recording_id=rid, reason=reason))
    values, reason = _lookup_values(known_values_by_id, rid)
    if reason is not None:
        return (None, ExampleWithheld(recording_id=rid, reason=reason))
    turn, reason = _find_turn(trace.turns, span.turn_idx)
    if reason is not None:
        return (None, ExampleWithheld(recording_id=rid, reason=reason))
    candidate, reason = _mask_span(turn, span, values)
    if reason is not None:
        return (None, ExampleWithheld(recording_id=rid, reason=reason))
    return (candidate, None)


def _note(seen: set[ExampleWithheld], dropped: list[ExampleWithheld], item: ExampleWithheld) -> None:
    if item not in seen:
        seen.add(item)
        dropped.append(item)


def assemble_candidates(
    traces_by_id: Mapping[str, Trace],
    spans: Collection[Any],
    *,
    held_out_ids: Collection[str],
    known_values_by_id: Mapping[str, Any],
) -> tuple[tuple[ExampleCandidate, ...], tuple[ExampleWithheld, ...]]:
    held = set(held_out_ids)
    seen_spans: set[ExampleSpan] = set()
    seen_dropped: set[ExampleWithheld] = set()
    cands: list[ExampleCandidate] = []
    dropped: list[ExampleWithheld] = []
    for raw in spans:
        rid = _raw_id(raw)
        if rid in held:
            _note(seen_dropped, dropped, ExampleWithheld(recording_id=rid, reason="held_out"))
            continue
        span, reason = _coerce(raw)
        if reason is not None:
            _note(seen_dropped, dropped, ExampleWithheld(recording_id=rid, reason=reason))
            continue
        if span is None:
            continue
        if span.recording_id in held:
            _note(seen_dropped, dropped, ExampleWithheld(recording_id=span.recording_id, reason="held_out"))
            continue
        if span in seen_spans:
            continue
        seen_spans.add(span)
        candidate, why = _decide_one(traces_by_id, known_values_by_id, span)
        if why is not None:
            _note(seen_dropped, dropped, why)
        elif candidate is not None:
            cands.append(candidate)
    ordered_cands = sorted(
        cands, key=lambda cand: (cand.situation, cand.recording_id, cand.turn_idx, cand.start, cand.end)
    )
    ordered_dropped = sorted(dropped, key=lambda item: (item.recording_id, item.reason))
    return (tuple(ordered_cands), tuple(ordered_dropped))
