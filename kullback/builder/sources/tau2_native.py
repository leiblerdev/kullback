"""The first adapter behind the intake seam: the export this harness reads today.

It votes with positive evidence (a simulations list, or one recording with
a messages list and an id), yields one recording per simulation, and maps
each recording to a Trace with a raw pointer on every field, grader fields
stripped into a sidecar, tool errors classed and truncated results marked.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, as_dict, content_hash

# Answer keys. Stripped before anything else reads the trace (D66, D89).
GRADER_FIELDS = (
    "reward_info", "evaluation_criteria", "action_checks", "nl_assertions",
    "env_assertions", "reward", "trial", "task_id",
)


class Tau2NativeAdapter:
    """The export with a simulations list, one messages list per recording."""

    name = "tau2_native"
    maps = True
    display = "tau2_native"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            return (0.0, ["a list holds no export block of this shape"])
        if not isinstance(document, dict):
            return (0.0, ["not a parsed object, so no shape to read"])
        if isinstance(document.get("simulations"), list):
            count = len(document["simulations"])
            return (1.0, [f"has a simulations list with {count} entries"])
        if isinstance(document.get("messages"), list) and "id" in document:
            return (0.9, ["has a messages list and an id"])
        return (0.0, ["neither a simulations list nor a messages list with an id"])

    def recordings(self, document: Any) -> Iterator[Any]:
        if isinstance(document, dict) and isinstance(document.get("simulations"), list):
            return iter(document["simulations"])
        return iter([document])

    def environment(self, document: Any) -> dict:
        if not isinstance(document, dict):
            return {}
        return (document.get("info") or {}).get("environment_info") or {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        """The answer key beside one recording: its grader fields, plus the
        matching task's evaluation criteria when the export carries tasks."""
        tasks = {}
        if isinstance(document, dict):
            tasks = {str(task.get("id")): task for task in document.get("tasks") or []
                     if isinstance(task, dict)}
        fields = {key: recording[key] for key in GRADER_FIELDS if key in recording}
        task = tasks.get(str(recording.get("task_id")))
        if task and "evaluation_criteria" in task:
            fields["evaluation_criteria"] = task["evaluation_criteria"]
        return fields

    def to_trace(self, recording: dict, ctx: Any) -> Trace:
        from kullback.builder.ingest import classify_error, detect_truncation

        messages = recording.get("messages") or []
        ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index)
        # The info block is not a message, so its pointer names the section instead of a message
        # index; it is where `tools_declared` and `system_prompt` below were read from (D66).
        info = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, section="info.environment_info")
        trace_id = str(recording.get("id") or f"{ctx.raw_hash[:12]}-{ctx.index}")
        turns, calls, pending = [], [], {}
        environment = ctx.environment
        for msg_index, message in enumerate(messages):
            here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=msg_index)
            role = message.get("role") or "assistant"
            requested = message.get("tool_calls") or []
            if role == "tool":
                waiting = pending.pop(message.get("id"), None)
                if waiting is not None:
                    _attach_result(waiting[0], message, waiting[1], here,
                                   classify_error, detect_truncation)
                turns.append(Turn(idx=msg_index, role="tool", content=_text(message.get("content")),
                                  tool_call_ids=[message["id"]] if message.get("id") else [],
                                  raw_ptr=here))
                continue
            for request in requested:
                call = ToolCall(
                    id=request.get("id"),
                    name=request.get("name") or "",
                    args=request.get("arguments") or {},
                    requestor=request.get("requestor") or role,
                    raw_ptr=here,
                    trace_id=trace_id,
                )
                calls.append(call)
                if call.id:
                    pending[call.id] = (call, message.get("timestamp"))
            turns.append(Turn(idx=msg_index, role=role, content=_text(message.get("content")),
                              tool_call_ids=[r.get("id") for r in requested if r.get("id")],
                              raw_ptr=here))
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=turns,
            tool_calls=calls,
            tools_declared=environment.get("tool_defs")
            if isinstance(environment.get("tool_defs"), list) else None,
            system_prompt=environment.get("policy"),
            tools_declared_ptr=info if environment.get("tool_defs") else None,
            system_prompt_ptr=info if environment.get("policy") else None,
            info_ptr=info,
            raw_ptr=ptr,
        )


def _attach_result(call: ToolCall, message: dict, asked_at: Any, ptr: Optional[RawPtr],
                   classify_error: Any, detect_truncation: Any) -> None:
    """Put the tool message that answered this call on the call, and say where it came from.

    `has_result` is set for every answered call, a recorded JSON null included, because `result is
    None` alone cannot tell a null answer apart from a call whose tool message was never captured
    (the ingest gate reads the flag). `resolved` says the answer landed on this call.
    """
    content = message.get("content")
    call.truncated, call.visible_len, call.cut_marker = detect_truncation(content)
    flag = message.get("error")
    if flag:
        structured = flag if isinstance(flag, dict) else _structured(content)
        call.error = classify_error(content, structured, ptr=ptr)
    else:
        call.result = _parsed(content)
    call.has_result = True
    call.resolved = True
    call.result_ptr = ptr
    call.latency_ms = _latency_ms(asked_at, message.get("timestamp"))


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


def _text(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


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


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, stable across runs."""
    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)
