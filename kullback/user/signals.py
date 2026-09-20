from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal, Optional

from pydantic import ConfigDict, StrictInt, ValidationError

from kullback.runner.records import RawPtr, Record, Role, ToolCall, Trace, Turn

EndingActor = Literal["user", "assistant", "environment", "unknown"]
LabelKind = Literal["why_left", "stated_frustration"]


class EndEvidence(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    end_kind: str = "unknown"
    ending_actor: EndingActor = "unknown"


class SpanRef(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    turn_idx: StrictInt
    start: StrictInt
    end: StrictInt
    quote: str


class ModelLabel(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    kind: LabelKind
    span: SpanRef


class AcceptedLabel(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    kind: LabelKind
    quote: str
    turn_idx: int
    start: int
    end: int
    source: RawPtr


class DroppedLabel(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    kind: LabelKind
    reason: str


class Signal(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    trace_id: str
    user_turns: int
    assistant_turns: int
    tool_turns: int
    system_turns: int
    last_speaker: Optional[Role] = None
    repeated_user_utterances: int
    transfer_call: bool
    unanswered_calls: int
    truncated_calls: int
    end_kind: str = "unknown"
    ending_actor: EndingActor = "unknown"
    why_left: Optional[AcceptedLabel] = None
    stated_frustration: Optional[AcceptedLabel] = None
    dropped: tuple[DroppedLabel, ...] = ()
    contradictions: int = 0


def _role_counts(turns: list[Turn]) -> dict[str, int]:
    counts = {"system": 0, "assistant": 0, "user": 0, "tool": 0}
    for turn in turns:
        if turn.role in counts:
            counts[turn.role] += 1
    return counts


def _last_speaker(turns: list[Turn]) -> Optional[Role]:
    if not turns:
        return None
    pos = max(range(len(turns)), key=lambda i: (turns[i].idx, i))
    return turns[pos].role


def _repeated_user_utterances(turns: list[Turn]) -> int:
    seen: dict[str, bool] = {}
    repeats = 0
    for turn in turns:
        if turn.role != "user":
            continue
        if turn.content is None:
            continue
        if turn.content in seen:
            repeats += 1
        else:
            seen[turn.content] = True
    return repeats


def _call_counts(calls: list[ToolCall]) -> tuple[int, int]:
    unanswered = 0
    truncated = 0
    for call in calls:
        if call.truncated:
            truncated += 1
        if call.has_result is not True or call.resolved is not True:
            unanswered += 1
    return (unanswered, truncated)


def _evidence_turn(turns: list[Turn], turn_idx: Any) -> tuple[Optional[Turn], Optional[str]]:
    if type(turn_idx) is not int:
        return (None, "unknown_turn")
    matches = [turn for turn in turns if turn.idx == turn_idx]
    if not matches:
        return (None, "unknown_turn")
    if len(matches) > 1:
        return (None, "ambiguous_turn")
    return (matches[0], None)


def _span_reason(turn: Turn, span: SpanRef) -> Optional[str]:
    if turn.role != "user":
        return "non_user_turn"
    if type(span.start) is not int or type(span.end) is not int:
        return "bad_bounds"
    content = turn.content
    if content is None:
        return "bad_bounds"
    if not 0 <= span.start < span.end <= len(content):
        return "bad_bounds"
    if not isinstance(span.quote, str) or not span.quote.strip():
        return "empty_quote"
    if content[span.start:span.end] != span.quote:
        return "nonverbatim_quote"
    return None


def _offset_failure_reason(exc: ValidationError, raw: Any) -> Optional[tuple[str, str]]:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind != "why_left" and kind != "stated_frustration":
        return None
    offsets = (("span", "turn_idx"), ("span", "start"), ("span", "end"))
    locs = []
    for detail in exc.errors(include_url=False):
        if detail.get("type") != "int_type":
            return None
        if tuple(detail.get("loc", ())) not in offsets:
            return None
        locs.append(tuple(detail.get("loc", ())))
    if ("span", "turn_idx") in locs:
        return (kind, "unknown_turn")
    return (kind, "bad_bounds")


def _has_transfer(calls: list[ToolCall], tools: set[str]) -> bool:
    for call in calls:
        if call.name in tools:
            return True
    return False


def _adjudicate(
    turns: list[Turn], actor: EndingActor, label: ModelLabel
) -> tuple[Optional[AcceptedLabel], Optional[str], int]:
    turn, reason = _evidence_turn(turns, label.span.turn_idx)
    if reason is None:
        reason = _span_reason(turn, label.span)
    if reason is None and label.kind == "why_left" and actor in ("assistant", "environment"):
        return (None, "contradiction", 1)
    if reason is not None:
        return (None, reason, 0)
    return (
        AcceptedLabel(
            kind=label.kind,
            quote=label.span.quote,
            turn_idx=label.span.turn_idx,
            start=label.span.start,
            end=label.span.end,
            source=turn.raw_ptr.model_copy(deep=True),
        ),
        None,
        0,
    )


def _ordered_labels(labels: Any) -> Sequence[Any]:
    if isinstance(labels, (set, frozenset, Mapping, str, bytes, bytearray, memoryview)):
        raise TypeError(f"labels must be an ordered Sequence (list or tuple), got {type(labels).__name__}")
    if not isinstance(labels, Sequence):
        raise TypeError(f"labels must be an ordered Sequence (list or tuple), got {type(labels).__name__}")
    return labels


def mine_signal(
    trace: Trace,
    *,
    end: Any = None,
    labels: Sequence[Any] = (),
    transfer_tools: Collection[str] = (),
) -> Signal:
    labels = _ordered_labels(labels)
    evidence = EndEvidence.model_validate(end) if end is not None else EndEvidence()
    counts = _role_counts(trace.turns)
    unanswered, truncated = _call_counts(trace.tool_calls)
    why: Optional[AcceptedLabel] = None
    frustration: Optional[AcceptedLabel] = None
    dropped: list[DroppedLabel] = []
    contradictions = 0
    for raw in labels:
        try:
            label = ModelLabel.model_validate(raw)
        except ValidationError as exc:
            early = _offset_failure_reason(exc, raw)
            if early is None:
                raise
            dropped.append(DroppedLabel(kind=early[0], reason=early[1]))
            continue
        accepted, reason, delta = _adjudicate(trace.turns, evidence.ending_actor, label)
        contradictions += delta
        if reason is not None:
            dropped.append(DroppedLabel(kind=label.kind, reason=reason))
            continue
        if label.kind == "why_left":
            if why is None:
                why = accepted
            else:
                dropped.append(DroppedLabel(kind=label.kind, reason="duplicate"))
        elif frustration is None:
            frustration = accepted
        else:
            dropped.append(DroppedLabel(kind=label.kind, reason="duplicate"))
    return Signal(
        trace_id=trace.trace_id,
        user_turns=counts["user"],
        assistant_turns=counts["assistant"],
        tool_turns=counts["tool"],
        system_turns=counts["system"],
        last_speaker=_last_speaker(trace.turns),
        repeated_user_utterances=_repeated_user_utterances(trace.turns),
        transfer_call=_has_transfer(trace.tool_calls, set(transfer_tools)),
        unanswered_calls=unanswered,
        truncated_calls=truncated,
        end_kind=evidence.end_kind,
        ending_actor=evidence.ending_actor,
        why_left=why,
        stated_frustration=frustration,
        dropped=tuple(dropped),
        contradictions=contradictions,
    )
