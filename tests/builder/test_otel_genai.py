"""OpenTelemetry GenAI spans enter through the adapter seam and map to Traces with pointers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kullback.builder import ingest

SYSTEM = {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]}
USER = {"role": "user", "parts": [{"type": "text", "content": "Where is order A1?"}]}
ASK = {"role": "assistant", "parts": [
    {"type": "text", "content": "Looking it up."},
    {"type": "tool_call", "id": "c1", "name": "find_order", "arguments": {"order": "A1"}}]}
ANSWER = {"role": "tool", "parts": [
    {"type": "tool_call_response", "id": "c1", "response": {"status": "shipped"}}]}
DONE = {"role": "assistant", "parts": [{"type": "text", "content": "It shipped."}]}
TOOLS = [{"type": "function", "name": "find_order", "parameters": {"type": "object"}}]


def otlp_value(value):
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": value if isinstance(value, str) else json.dumps(value)}


def otlp_span(trace_id, name, start, end, attributes, events=None, status=None):
    span = {"traceId": trace_id, "spanId": name.replace(" ", "-"), "name": name,
            "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
            "attributes": [{"key": k, "value": otlp_value(v)} for k, v in attributes.items()]}
    if events is not None:
        span["events"] = events
    if status is not None:
        span["status"] = status
    return span


def otlp_export(spans):
    return {"resourceSpans": [{"resource": {"attributes": []},
                               "scopeSpans": [{"scope": {"name": "toy"}, "spans": spans}]}]}


def chat_attributes(inputs, outputs):
    return {"gen_ai.operation.name": "chat", "gen_ai.tool.definitions": json.dumps(TOOLS),
            "gen_ai.input.messages": json.dumps(inputs),
            "gen_ai.output.messages": json.dumps(outputs)}


def ingest_document(document, workdir, tmp_path, name="spans.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    raw = ingest.store_raw(path, workdir)
    assert raw.format_detected == "otel_genai"
    return raw, ingest.derive_traces(raw.raw_hash, workdir)


# --- one helper reads every pointer back against the stored file ------------------------------


def _attributes(value):
    if isinstance(value, list) and all(isinstance(item, dict) and "key" in item for item in value):
        return {item["key"]: next(iter(item["value"].values())) for item in value}
    return value


def _parsed(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        return json.loads(value)
    return value


def _spans(document):
    if isinstance(document, dict) and "resourceSpans" in document:
        return [span for block in document["resourceSpans"] for scope in block["scopeSpans"]
                for span in scope["spans"]]
    return document


def resolve(spans, ptr):
    """The value a pointer names: the span at sim_index, then the section's path into it."""
    node, rest = spans[ptr.sim_index], ptr.section
    while rest:
        if rest.startswith("attributes.") and isinstance(node, dict):
            attributes = _attributes(node["attributes"])
            tail = rest[len("attributes."):]
            key = max((k for k in attributes if tail.startswith(k)), key=len, default=None)
            if key is not None:
                node, rest = _parsed(attributes[key]), tail[len(key):]
                continue
        step = re.match(r"\.?([A-Za-z_]+)|\[(\d+)\]", rest)
        name, position = step.groups()
        node = node[name] if name else node[int(position)]
        node = _attributes(node) if name == "attributes" else node
        node, rest = _parsed(node), rest[step.end():]
    return node


def _leaves(node):
    if isinstance(node, dict):
        return [leaf for value in node.values() for leaf in _leaves(value)]
    if isinstance(node, list):
        return [leaf for value in node for leaf in _leaves(value)]
    return [node]


def assert_every_pointer_resolves(raw, traces, workdir):
    spans = _spans(json.loads(Path(ingest.raw_path(raw.raw_hash, workdir)).read_text()))
    for trace in traces:
        for turn in trace.turns:
            found = resolve(spans, turn.raw_ptr)
            for line in (turn.content or "").splitlines():
                assert line in _leaves(found), (turn.raw_ptr, line)
        for call in trace.tool_calls:
            assert call.name in _leaves(resolve(spans, call.raw_ptr)) + [spans[call.raw_ptr.sim_index]
                                                                       ["name"].split(" ")[-1]]
            if call.result_ptr is not None:
                answer = resolve(spans, call.result_ptr)
                expected = call.error.payload if call.error else call.result
                assert answer == expected or expected in [answer] + _leaves(answer) \
                    or (isinstance(answer, dict) and expected in answer.values())
        for ptr in (trace.system_prompt_ptr, trace.tools_declared_ptr):
            if ptr is not None:
                assert resolve(spans, ptr) is not None


# --- behaviour ---------------------------------------------------------------------------------


def test_an_otlp_export_with_a_chat_span_and_a_tool_span_is_one_trace_with_result_and_latency(
        workdir, tmp_path):
    spans = [
        otlp_span("t1", "chat toy", 1_000_000_000, 2_000_000_000,
                  chat_attributes([SYSTEM, USER], [ASK])),
        otlp_span("t1", "execute_tool find_order", 2_050_000_000, 2_250_000_000,
                  {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "find_order",
                   "gen_ai.tool.call.id": "c1", "gen_ai.tool.call.arguments": '{"order": "A1"}',
                   "gen_ai.tool.call.result": '{"status": "shipped"}'}),
    ]
    raw, traces = ingest_document(otlp_export(spans), workdir, tmp_path)
    assert len(traces) == 1
    trace = traces[0]
    assert (trace.trace_id, trace.source) == ("t1", "otel_genai")
    assert [turn.role for turn in trace.turns] == ["user", "assistant"]
    assert trace.system_prompt == "Be brief."
    assert trace.tools_declared == TOOLS
    [call] = trace.tool_calls
    assert (call.id, call.name, call.args) == ("c1", "find_order", {"order": "A1"})
    assert call.result == {"status": "shipped"} and call.resolved and call.has_result
    assert call.latency_ms == 250.0
    assert call.result_ptr.sim_index == 1
    assert call.result_ptr.section == "attributes.gen_ai.tool.call.result"
    assert ingest.gate_ingest(traces, workdir, raw_hash=raw.raw_hash).passed
    assert_every_pointer_resolves(raw, traces, workdir)


def test_input_and_output_message_parts_map_to_turns_and_calls_with_pointers_into_the_attribute(
        workdir, tmp_path):
    first = {"traceId": "t1", "name": "chat toy", "startTimeUnixNano": 1, "endTimeUnixNano": 2,
             "attributes": chat_attributes([SYSTEM, USER], [ASK])}
    second = {"traceId": "t1", "name": "chat toy", "startTimeUnixNano": 3, "endTimeUnixNano": 4,
              "attributes": chat_attributes([SYSTEM, USER, ASK, ANSWER], [DONE])}
    raw, traces = ingest_document([second, first], workdir, tmp_path)
    [trace] = traces
    assert [turn.role for turn in trace.turns] == ["user", "assistant", "tool", "assistant"]
    assert [turn.content for turn in trace.turns] == [
        "Where is order A1?", "Looking it up.", None, "It shipped."]
    assert trace.turns[1].raw_ptr.section == "attributes.gen_ai.output.messages[0]"
    assert trace.turns[2].tool_call_ids == ["c1"]
    [call] = trace.tool_calls
    assert call.raw_ptr.sim_index == 1
    assert call.raw_ptr.section == "attributes.gen_ai.output.messages[0].parts[1]"
    assert call.result == {"status": "shipped"}
    assert (call.result_ptr.sim_index, call.result_ptr.section) == (
        0, "attributes.gen_ai.input.messages[3].parts[0]")
    assert_every_pointer_resolves(raw, traces, workdir)


def test_message_events_map_like_message_attributes_and_a_failed_tool_span_is_classed(
        workdir, tmp_path):
    events = [
        {"name": "gen_ai.system.message", "body": {"content": "Be brief."}},
        {"name": "gen_ai.user.message", "body": {"content": "Cancel order A1."}},
        {"name": "gen_ai.choice", "body": {"index": 0, "finish_reason": "tool_calls", "message": {
            "content": "Cancelling.", "tool_calls": [{"id": "c9", "type": "function", "function": {
                "name": "cancel_order", "arguments": '{"order": "A1"}'}}]}}},
    ]
    spans = [
        {"traceId": "t5", "name": "chat toy", "startTimeUnixNano": 10, "endTimeUnixNano": 20,
         "attributes": {"gen_ai.operation.name": "chat"}, "events": events},
        {"traceId": "t5", "name": "execute_tool cancel_order", "startTimeUnixNano": 30,
         "endTimeUnixNano": 1_000_020, "status": {"code": 2, "message": "order not found"},
         "attributes": {"gen_ai.tool.call.id": "c9", "error.type": "not_found_entity",
                        "gen_ai.tool.call.result": "order not found"}},
    ]
    raw, traces = ingest_document(spans, workdir, tmp_path)
    [trace] = traces
    assert trace.system_prompt == "Be brief."
    assert trace.system_prompt_ptr.section == "events[0].body"
    assert [turn.content for turn in trace.turns] == ["Cancel order A1.", "Cancelling."]
    [call] = trace.tool_calls
    assert call.raw_ptr.section == "events[2].body.message.tool_calls[0]"
    assert call.args == {"order": "A1"}
    assert call.error.class_ == "not_found_entity" and call.result is None
    assert call.latency_ms == 1.0
    assert_every_pointer_resolves(raw, traces, workdir)


def test_two_trace_ids_are_two_traces_and_a_genai_span_with_nothing_to_map_is_a_reject(
        workdir, tmp_path):
    spans = [
        otlp_span("ta", "chat toy", 1, 2, chat_attributes([USER], [DONE])),
        otlp_span("tb", "chat toy", 3, 4, chat_attributes([USER], [ASK])),
        otlp_span("tc", "gen_ai.client.inference.operation.details", 5, 6,
                  {"gen_ai.system": "toy", "gen_ai.request.model": "toy-1"}),
    ]
    raw, traces = ingest_document(otlp_export(spans), workdir, tmp_path)
    assert [trace.trace_id for trace in traces] == ["ta", "tb"]
    assert [trace.raw_ptr.sim_index for trace in traces] == [0, 1]
    [reject] = ingest.read_rejects(workdir, raw.raw_hash)
    assert reject["trace_id"] == "tc" and reject["sim_index"] == 2
    assert "carry no input or output messages" in reject["reason"]
    assert "gen_ai.client.inference.operation.details" in reject["reason"]
    assert_every_pointer_resolves(raw, traces, workdir)
