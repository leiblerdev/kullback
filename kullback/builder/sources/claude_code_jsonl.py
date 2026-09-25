"""The adapter behind the intake seam for line-delimited assistant transcripts.

Each row carries a type (user or assistant) and a message with content blocks:
plain text, tool_use requests with an id, a name and input arguments, and
tool_result answers naming the id they answer. Rows group into one recording
per conversation id, turns follow row order, and every tool_use pairs with its
tool_result by id, so the intake gate reads an unpaired call or an orphan
result exactly as it reads one from the first mapped export.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace

# The row fields that name the conversation a row belongs to, preferred first.
_SESSION_FIELDS = ("sessionId", "session_id")

# Row types that read as dialogue even when the message names no role. Any
# other row type without a message role is transcript metadata, not a turn.
_DIALOGUE_TYPES = {"user": "user", "assistant": "assistant", "system": "system"}


class ClaudeCodeJsonlDetector:
    """Votes on line-delimited assistant transcript rows and maps them to traces."""

    name = "claude_code_jsonl"
    maps = True
    display = "Claude Code JSONL"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            if jsonl and any(_looks_claude_code(item) for item in document[:20]
                             if isinstance(item, dict)):
                return (0.8, ["line-delimited rows carry transcript fields"])
            return (0.0, ["no transcript row marker, or the file is one JSON document, not lines"])
        return (0.0, ["not a record list, so not line-delimited rows"])

    def recordings(self, document: Any) -> Iterator[Any]:
        """One recording per conversation id, rows in file order."""
        rows = document if isinstance(document, list) else [document]
        groups: dict[Any, list] = {}
        order: list = []
        for row in rows:
            key = _session_key(row)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)
        return iter([{"id": key, "rows": groups[key]} for key in order])

    def to_trace(self, recording: Any, ctx: Any) -> Trace:
        from kullback.builder.ingest import classify_error, detect_truncation

        rows = recording.get("rows") if isinstance(recording, dict) else []
        rows = rows if isinstance(rows, list) else []
        ident = recording.get("id") if isinstance(recording, dict) else None
        trace_id = str(ident or f"{ctx.raw_hash[:12]}-{ctx.index}")
        ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index)
        turns, calls, pending = [], [], {}
        for row_index, row in enumerate(rows):
            here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=row_index)
            turn, fresh = _map_row(row, here, trace_id, pending,
                                   classify_error, detect_truncation)
            if turn is None:
                continue
            turns.append(turn)
            calls.extend(fresh)
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=turns,
            tool_calls=calls,
            tools_declared=None,
            tools_declared_ptr=None,
            system_prompt=None,
            system_prompt_ptr=None,
            info_ptr=None,
            raw_ptr=ptr,
        )

    def environment(self, document: Any) -> dict:
        """No export-level block rides along in this format, so nothing, unless
        a row itself carries a domain."""
        rows = document if isinstance(document, list) else []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("domain"), str) and row["domain"]:
                return {"domain": row["domain"]}
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        """No grader fields ride along in this format, so nothing to strip."""
        return {}


def _session_key(row: Any) -> Optional[str]:
    """The conversation id of one row, or None when the row names none."""
    if isinstance(row, dict):
        for field in _SESSION_FIELDS:
            value = row.get(field)
            if value is not None:
                return str(value)
    return None


def _map_row(row: dict, here: RawPtr, trace_id: str, pending: dict,
             classify_error: Any, detect_truncation: Any) -> tuple[Any, list[ToolCall]]:
    """Map one row to its turn and any tool calls it requests, answering the
    calls a tool_result block resolves. `pending` carries the calls still
    waiting for their answer. A row with no dialogue role is transcript
    metadata and maps to no turn."""
    message = row.get("message") if isinstance(row.get("message"), dict) else {}
    role = message.get("role") or _DIALOGUE_TYPES.get(row.get("type"))
    if role is None:
        return (None, [])
    texts, uses, results = _split_content(message.get("content"))
    calls = [_requested_call(block, role, trace_id, here) for block in uses]
    _remember_requests(calls, pending, row)
    _answer_ready(results, pending, row, here, classify_error, detect_truncation)
    return (_row_turn(role, texts, uses, results, here), calls)


def _remember_requests(calls: list[ToolCall], pending: dict, row: dict) -> None:
    """Keep every requested call by id, so the row carrying its answer finds it."""
    for call in calls:
        if call.id:
            pending[call.id] = (call, row.get("timestamp"))


def _answer_ready(results: list[dict], pending: dict, row: dict, here: RawPtr,
                  classify_error: Any, detect_truncation: Any) -> None:
    """Attach each answer block to the call still waiting for it, if any."""
    for block in results:
        waiting = pending.pop(block.get("tool_use_id"), None)
        if waiting is not None:
            _attach_result(waiting[0], block, waiting[1], row.get("timestamp"), here,
                           classify_error, detect_truncation)


def _row_turn(role: str, texts: list[str], uses: list[dict], results: list[dict],
              here: RawPtr) -> Any:
    """One row's turn: the tool role when the row carries answers, else its own role."""
    from kullback.runner.records import Turn

    answered = [block.get("tool_use_id") for block in results if block.get("tool_use_id")]
    # A row carrying answers reads as the tool's turn, the way the tool
    # message of the first mapped export does, so orphan results surface.
    turn_role = "tool" if results else role
    call_ids = ([block.get("id") for block in uses if block.get("id")] +
                [answer for answer in answered])
    return Turn(idx=here.msg_index if here.msg_index is not None else 0, role=turn_role,
                content="\n".join(texts) or None, tool_call_ids=call_ids, raw_ptr=here)


def _split_content(content: Any) -> tuple[list[str], list[dict], list[dict]]:
    """One row's content as (text lines, tool_use blocks, tool_result blocks)."""
    if content is None:
        return ([], [], [])
    if isinstance(content, str):
        return ([content], [], [])
    blocks = [content] if isinstance(content, dict) else list(content or [])
    texts, uses, results = [], [], []
    for block in blocks:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind == "tool_use":
                uses.append(block)
            elif kind == "tool_result":
                results.append(block)
            elif isinstance(block.get("text"), str):
                texts.append(block["text"])
    return (texts, uses, results)


def _requested_call(block: dict, role: str, trace_id: str, here: RawPtr) -> ToolCall:
    return ToolCall(
        id=block.get("id"),
        name=block.get("name") or "",
        args=block.get("input") or {},
        requestor=role,
        raw_ptr=here,
        trace_id=trace_id,
    )


def _attach_result(call: ToolCall, block: dict, asked_at: Any, answered_at: Any,
                   ptr: Optional[RawPtr], classify_error: Any, detect_truncation: Any) -> None:
    """Put the tool_result block that answered this call on the call, and say
    where it came from.

    `has_result` is set for every answered call, a recorded JSON null included,
    because `result is None` alone cannot tell a null answer apart from a call
    whose answer was never captured (the ingest gate reads the flag).
    `resolved` says the answer landed on this call.
    """
    content = _result_content(block.get("content"))
    call.truncated, call.visible_len, call.cut_marker = detect_truncation(content)
    if block.get("is_error"):
        structured = content if isinstance(content, dict) else _structured(content)
        call.error = classify_error(content, structured, ptr=ptr)
    else:
        call.result = _parsed(content)
    call.has_result = True
    call.resolved = True
    call.result_ptr = ptr
    call.latency_ms = _latency_ms(asked_at, answered_at)


def _result_content(content: Any) -> Any:
    """One answer's payload as a scalar: text blocks joined, anything else as is."""
    if isinstance(content, list):
        texts = [block["text"] for block in content
                 if isinstance(block, dict) and isinstance(block.get("text"), str)]
        if texts or not content:
            return "\n".join(texts) or None
        return content
    return content


def _structured(content: Any) -> Optional[dict]:
    """A typed error body when the source sends one, so the classifier can use the code."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip()[:1] == "{":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _parsed(content: Any) -> Any:
    """Tool results arrive as JSON strings in this export; parse them so the miner sees fields."""
    if isinstance(content, str) and content.strip()[:1] in ("{", "["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content  # kept verbatim; the gate reports it as not parseable
    return content


def _latency_ms(asked: Any, answered: Any) -> Optional[float]:
    try:
        start = datetime.fromisoformat(str(asked))
        end = datetime.fromisoformat(str(answered))
    except (TypeError, ValueError):
        return None
    return (end - start).total_seconds() * 1000.0


def _looks_claude_code(item: dict) -> bool:
    if item.get("type") not in ("user", "assistant", "system", "summary", "result"):
        return False
    return any(key in item for key in ("message", "content", "uuid", "sessionId"))
