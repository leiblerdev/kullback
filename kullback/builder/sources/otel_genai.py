"""The OpenTelemetry GenAI adapter: model-call and tool-execution spans mapped to a Trace.

Read 2026-10-06 from https://opentelemetry.io/docs/specs/semconv/gen-ai/ (the spans, events and
attribute registry pages; the site now points to the GenAI conventions repository, which keeps the
same names). The names read here:
- spans: `gen_ai.operation.name` (chat, text_completion, generate_content, invoke_agent,
  execute_tool), span name `execute_tool <gen_ai.tool.name>` for a tool execution;
- messages as attributes: `gen_ai.input.messages`, `gen_ai.output.messages` (JSON strings of
  {role, parts}; a part of type text carries content, tool_call carries id, name, arguments,
  tool_call_response carries id and response, read here as response or result) and
  `gen_ai.system_instructions`;
- messages as span events: `gen_ai.system.message`, `gen_ai.user.message`,
  `gen_ai.assistant.message` (content, tool_calls of {id, function {name, arguments}}),
  `gen_ai.tool.message` (content, id) and `gen_ai.choice` (message, finish_reason), the payload
  in the event body or its attributes;
- tool spans: `gen_ai.tool.call.id`, `gen_ai.tool.name`, `gen_ai.tool.call.arguments`,
  `gen_ai.tool.call.result`, `error.type` and the span status;
- declared tools: `gen_ai.tool.definitions`, or `gen_ai.request.tools` on older exports.

Two payload shapes enter: an OTLP JSON export (resourceSpans, scopeSpans, spans, attributes as
{key, value} lists) and a flat list of span dicts with an attributes dict. One recording per
traceId, its spans ordered by start time. A chat span's input repeats the history the earlier
spans already showed, so only the messages past that history become new turns. A pointer's
sim_index is the span's index in the flattened span list and its section names the attribute or
event read, so every field reads back against the stored bytes. The format carries no benchmark
fields, so the sidecar is empty.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, as_dict, content_hash

INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = ("gen_ai.tool.definitions", "gen_ai.request.tools")
OPERATION = "gen_ai.operation.name"
TOOL_OPERATION = "execute_tool"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_NAME = "gen_ai.tool.name"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
ERROR_TYPE = "error.type"

# Event name to the role its message carries; a choice is the model's output.
EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
    "gen_ai.choice": "assistant",
}
CHOICE_EVENT = "gen_ai.choice"


class NothingToMap(TypeError):
    """A recording whose GenAI spans carry no message, event or tool call; the seam rejects it."""


class OtelGenaiDetector:
    """Votes on the GenAI span shape and maps one trace's spans to one Trace."""

    name = "otel_genai"
    maps = True
    display = "OpenTelemetry GenAI"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            heads = [item for item in document[:20] if isinstance(item, dict)]
            if any(_looks_otel(item) for item in heads):
                return (0.9, ["list records carry GenAI span names or attributes"])
            return (0.0, ["no list record carries a GenAI span name or attribute"])
        if isinstance(document, dict):
            if "resourceSpans" in document or "resource_spans" in document:
                return (0.9, ["has a resource spans block"])
            if _looks_otel(document):
                return (0.9, ["carries GenAI span names or attributes"])
            return (0.0, ["no resource spans block and no GenAI span marker"])
        return (0.0, ["not a parsed object, so no shape to read"])

    def recordings(self, document: Any) -> Iterator[Any]:
        """One recording per traceId, in first-seen order, spans ordered by start time."""
        groups: dict[str, list] = {}
        for index, span in enumerate(flat_spans(document)):
            groups.setdefault(_trace_id(span), []).append((index, span))
        return iter({"id": trace_id or None,
                     "spans": sorted(spans, key=lambda pair: (_nanos(pair[1], "start"), pair[0]))}
                    for trace_id, spans in groups.items())

    def environment(self, document: Any) -> dict:
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        return {}

    def to_trace(self, recording: dict, ctx: Any) -> Trace:
        from kullback.builder.ingest import classify_error, detect_truncation

        trace_id = str(recording.get("id") or f"{ctx.raw_hash[:12]}-{ctx.index}")
        state = _State(ctx.raw_hash, trace_id)
        tool_spans = []
        for index, span in recording.get("spans") or []:
            attributes = attribute_dict(span.get("attributes"))
            if _is_tool_span(span, attributes):
                tool_spans.append((index, span, attributes))
            else:
                state.read_model_span(index, span, attributes)
        for index, span, attributes in tool_spans:
            state.attach_tool_span(index, span, attributes, classify_error, detect_truncation)
        state.attach_responses(classify_error, detect_truncation)
        if not state.turns and not state.calls:
            raise NothingToMap(_nothing_reason(recording))
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=state.turns,
            tool_calls=state.calls,
            tools_declared=state.tools,
            system_prompt=state.system,
            tools_declared_ptr=state.tools_ptr,
            system_prompt_ptr=state.system_ptr,
            info_ptr=None,
            # The ruling finds a Trace's recording by this index, so it is the recording's own.
            raw_ptr=RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index),
        )


class _State:
    """What one trace's spans have mapped so far: turns, calls, the history length and the
    responses still waiting for a call."""

    def __init__(self, raw_hash: str, trace_id: str) -> None:
        self.raw_hash, self.trace_id = raw_hash, trace_id
        self.turns: list[Turn] = []
        self.calls: list[ToolCall] = []
        self.by_id: dict[str, tuple[ToolCall, int]] = {}
        self.responses: list[tuple[str, Any, RawPtr]] = []
        self.history = 0
        self.system: Optional[str] = None
        self.system_ptr: Optional[RawPtr] = None
        self.tools: Optional[list[dict]] = None
        self.tools_ptr: Optional[RawPtr] = None

    def ptr(self, index: int, section: str) -> RawPtr:
        return RawPtr(file_hash=self.raw_hash, sim_index=index, section=section)

    def read_model_span(self, index: int, span: dict, attributes: dict) -> None:
        self._read_tools(index, attributes)
        if isinstance(attributes.get(SYSTEM_INSTRUCTIONS), (str, list)) and self.system is None:
            self.system = _parts_text(_json(attributes[SYSTEM_INSTRUCTIONS]))
            self.system_ptr = self.ptr(index, f"attributes.{SYSTEM_INSTRUCTIONS}")
        inputs = _attribute_messages(attributes, INPUT_MESSAGES)
        outputs = _attribute_messages(attributes, OUTPUT_MESSAGES)
        if inputs or outputs:
            self._read_messages(index, inputs, outputs, span)
            return
        events = [(position, event) for position, event in enumerate(span.get("events") or [])
                  if isinstance(event, dict) and event.get("name") in EVENT_ROLES]
        read = [(f"events[{p}]", e.get("name"), _event_message(e)) for p, e in events]
        inputs = [(at + m["base"], m) for at, name, m in read if name != CHOICE_EVENT]
        outputs = [(at + m["base"], m) for at, name, m in read if name == CHOICE_EVENT]
        if inputs or outputs:
            self._read_messages(index, inputs, outputs, span)

    def _read_tools(self, index: int, attributes: dict) -> None:
        for key in TOOL_DEFINITIONS:
            declared = _json(attributes.get(key))
            if self.tools is None and isinstance(declared, list):
                self.tools = [tool for tool in declared if isinstance(tool, dict)]
                self.tools_ptr = self.ptr(index, f"attributes.{key}")

    def _read_messages(self, index: int, inputs: list, outputs: list, span: dict) -> None:
        """New input past the history the earlier spans showed, then this span's output."""
        fresh = inputs[self.history:] if len(inputs) >= self.history else inputs
        for section, message in fresh + outputs:
            self._add_message(index, section, message, span)
        self.history = len(inputs) + len(outputs)

    def _add_message(self, index: int, section: str, message: dict, span: dict) -> None:
        role = message.get("role")
        if role == "system":
            if self.system is None:
                self.system = message.get("text")
                self.system_ptr = self.ptr(index, section)
            return
        here = self.ptr(index, section)
        call_ids = []
        for sub, call_id, name, arguments in message.get("calls", []):
            call = ToolCall(id=call_id, name=name or "", args=_args(arguments),
                            requestor="assistant", raw_ptr=self.ptr(index, f"{section}{sub}"),
                            trace_id=self.trace_id)
            self.calls.append(call)
            if call_id:
                self.by_id[call_id] = (call, _nanos(span, "end"))
                call_ids.append(call_id)
        for sub, call_id, result in message.get("responses", []):
            self.responses.append((call_id, result, self.ptr(index, f"{section}{sub}")))
            call_ids.append(call_id)
        kind = role if role in ("user", "assistant", "tool") else "assistant"
        if message.get("responses") and kind == "user":
            kind = "tool"
        self.turns.append(Turn(idx=len(self.turns), role=kind, content=message.get("text"),
                               tool_call_ids=[i for i in call_ids if i], raw_ptr=here))

    def attach_tool_span(self, index: int, span: dict, attributes: dict,
                         classify_error: Any, detect_truncation: Any) -> None:
        """A tool execution answers the call with its id; one no model span asked for is a call
        of its own, requested and answered by the same span."""
        call_id = _string(attributes.get(TOOL_CALL_ID))
        found = self.by_id.get(call_id) if call_id else None
        if found is None:
            name = _string(attributes.get(TOOL_NAME)) or _span_tool_name(span)
            section = f"attributes.{TOOL_ARGUMENTS}" if TOOL_ARGUMENTS in attributes else "name"
            call = ToolCall(id=call_id, name=name or "", args=_args(attributes.get(TOOL_ARGUMENTS)),
                            requestor="assistant", raw_ptr=self.ptr(index, section),
                            trace_id=self.trace_id)
            self.calls.append(call)
            found = (call, _nanos(span, "start"))
            if call_id:
                self.by_id[call_id] = found
        call, asked = found
        if call.resolved:
            return
        result = attributes.get(TOOL_RESULT)
        where = self.ptr(index, f"attributes.{TOOL_RESULT}" if TOOL_RESULT in attributes
                         else "status")
        failed = _span_failed(span, attributes)
        _answer(call, result if not failed or result is not None else _status_message(span),
                where, failed, attributes.get(ERROR_TYPE), classify_error, detect_truncation)
        ended = _nanos(span, "end")
        call.latency_ms = (ended - asked) / 1e6 if ended and asked and ended >= asked else None

    def attach_responses(self, classify_error: Any, detect_truncation: Any) -> None:
        """A tool_call_response part answers its call when no tool span did."""
        for call_id, result, where in self.responses:
            found = self.by_id.get(call_id) if call_id else None
            if found is None or found[0].resolved:
                continue
            _answer(found[0], result, where, False, None, classify_error, detect_truncation)


def _answer(call: ToolCall, result: Any, where: RawPtr, failed: bool, error_type: Any,
            classify_error: Any, detect_truncation: Any) -> None:
    call.truncated, call.visible_len, call.cut_marker = detect_truncation(result)
    if failed:
        structured = {"type": error_type} if isinstance(error_type, str) else None
        call.error = classify_error(result, structured, ptr=where)
    else:
        call.result = _json(result)
    call.has_result = True
    call.resolved = True
    call.result_ptr = where


# --- span layout -----------------------------------------------------------------------------


def flat_spans(document: Any) -> list[dict]:
    """Every span in file order: an OTLP export's nested spans, or a flat list of span dicts."""
    if isinstance(document, dict):
        resource = document.get("resourceSpans", document.get("resource_spans"))
        if isinstance(resource, list):
            spans = []
            for block in resource:
                scopes = block.get("scopeSpans", block.get("scope_spans")) \
                    if isinstance(block, dict) else None
                for scope in scopes or []:
                    if isinstance(scope, dict):
                        spans.extend(span for span in scope.get("spans") or []
                                     if isinstance(span, dict))
            return spans
        return [document]
    if isinstance(document, list):
        return [span for span in document if isinstance(span, dict)]
    return []


def attribute_dict(attributes: Any) -> dict:
    """Attributes as a plain dict, whether the export wrote a dict or a list of {key, value}."""
    if isinstance(attributes, dict):
        return attributes
    if not isinstance(attributes, list):
        return {}
    return {item["key"]: any_value(item.get("value")) for item in attributes
            if isinstance(item, dict) and "key" in item}


def any_value(value: Any) -> Any:
    """One OTLP AnyValue as a plain value; anything else is returned as given."""
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "arrayValue" in value:
        return [any_value(item) for item in (value["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in value:
        return attribute_dict((value["kvlistValue"] or {}).get("values"))
    return value


def _trace_id(span: dict) -> str:
    context = span.get("context") if isinstance(span.get("context"), dict) else {}
    return str(span.get("traceId") or span.get("trace_id") or context.get("trace_id") or "")


def _nanos(span: dict, edge: str) -> int:
    value = span.get(f"{edge}TimeUnixNano", span.get(f"{edge}_time_unix_nano"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_tool_span(span: dict, attributes: dict) -> bool:
    return (attributes.get(OPERATION) == TOOL_OPERATION
            or str(span.get("name", "")).startswith(TOOL_OPERATION + " "))


def _span_tool_name(span: dict) -> str:
    name = str(span.get("name", ""))
    return name[len(TOOL_OPERATION) + 1:] if name.startswith(TOOL_OPERATION + " ") else ""


def _span_failed(span: dict, attributes: dict) -> bool:
    status = span.get("status") if isinstance(span.get("status"), dict) else {}
    return ERROR_TYPE in attributes or status.get("code") in (2, "STATUS_CODE_ERROR", "ERROR")


def _status_message(span: dict) -> Any:
    status = span.get("status") if isinstance(span.get("status"), dict) else {}
    return status.get("message")


def _nothing_reason(recording: dict) -> str:
    names = [str(span.get("name") or "unnamed") for _, span in recording.get("spans") or []
             if _looks_otel(span) or any(str(k).startswith("gen_ai.")
                                         for k in attribute_dict(span.get("attributes")))]
    if not names:
        return "the trace holds no GenAI span, so there is nothing to map"
    return (f"{len(names)} GenAI span(s) ({', '.join(sorted(set(names)))}) carry no input or "
            "output messages, no message events and no tool call, so there is nothing to map")


# --- messages --------------------------------------------------------------------------------


def _attribute_messages(attributes: dict, key: str) -> list[tuple[str, dict]]:
    """The messages one attribute holds, each with its pointer section."""
    messages = _json(attributes.get(key))
    if not isinstance(messages, list):
        return []
    return [(f"attributes.{key}[{i}]", _parts_message(message))
            for i, message in enumerate(messages) if isinstance(message, dict)]


def _parts_message(message: dict) -> dict:
    """A {role, parts} message as its text, the calls it requests and the responses it carries."""
    texts, calls, responses = [], [], []
    for j, part in enumerate(message.get("parts") or []):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "tool_call":
            calls.append((f".parts[{j}]", _string(part.get("id")), _string(part.get("name")),
                          part.get("arguments")))
        elif kind == "tool_call_response":
            result = part.get("response", part.get("result"))
            responses.append((f".parts[{j}]", _string(part.get("id")), result))
        elif isinstance(part.get("content"), str):
            texts.append(part["content"])
    return {"role": message.get("role"), "text": "\n".join(texts) if texts else None,
            "calls": calls, "responses": responses}


def _event_message(event: dict) -> dict:
    """One message event as the same shape; the payload sits in the body or the attributes, and
    base names where, so pointers cite the payload itself."""
    if isinstance(event.get("body"), dict):
        payload, base = event["body"], ".body"
    else:
        payload, base = attribute_dict(event.get("attributes")), ".attributes"
    role = EVENT_ROLES[event["name"]]
    if event["name"] == CHOICE_EVENT and isinstance(_json(payload.get("message")), dict):
        payload, base = _json(payload["message"]), base + ".message"
    calls = []
    for j, request in enumerate(_json(payload.get("tool_calls")) or []):
        if not isinstance(request, dict):
            continue
        function = request.get("function") if isinstance(request.get("function"), dict) else request
        calls.append((f".tool_calls[{j}]", _string(request.get("id")),
                      _string(function.get("name")), function.get("arguments")))
    responses = []
    if role == "tool":
        responses.append(("", _string(payload.get("id")), payload.get("content")))
    content = payload.get("content")
    text = content if isinstance(content, str) or content is None else json.dumps(
        content, ensure_ascii=False, default=str)
    return {"role": role, "text": text, "base": base, "calls": calls, "responses": responses}


def _parts_text(value: Any) -> Optional[str]:
    """System instructions as text: a string, or the content of their text parts."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, list):
        texts = [part.get("content") for part in value
                 if isinstance(part, dict) and isinstance(part.get("content"), str)]
        return "\n".join(texts) if texts else None
    return _text(value)


def _text(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _json(value: Any) -> Any:
    """A JSON string parsed, so fields read as fields; anything else is returned as given."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _args(value: Any) -> dict:
    parsed = _json(value)
    return parsed if isinstance(parsed, dict) else ({} if parsed in (None, "") else {"input": parsed})


def _looks_otel(item: dict) -> bool:
    if str(item.get("name", "")).startswith("gen_ai."):
        return True
    attributes = item.get("attributes")
    return isinstance(attributes, dict) and any(str(k).startswith("gen_ai.") for k in attributes)


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, stable across runs."""
    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)
