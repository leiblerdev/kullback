"""The provider-neutral stream over Model.query: assembled events, all three offline models, errors."""

from __future__ import annotations

import asyncio
import json

from kullback.ai import provider as pv
from kullback.ai.messages import AssistantMessage, ToolResultMessage, UserMessage, to_wire
from kullback.ai.provider import MemoModel, ModelReply, RecordedModel, TestModel, ToolCallRequest
from kullback.ai.stream import StreamDone, StreamError, assemble, normalize_stop_reason, stream


def events_of(model, messages, tools=None, system=None):
    async def collect():
        return [e async for e in stream(model, messages, tools=tools, system=system)]

    return asyncio.run(collect())


def test_plain_reply_streams_start_text_done():
    model = TestModel(["hello there"])
    events = events_of(model, [UserMessage(content="hi")], system="be brief")
    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "done"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reason == "stop"
    assert done.message.content == "hello there"
    assert done.message.tool_calls == []
    # the partial grows as the events go by, and the start carries an empty message
    assert events[0].partial.content is None
    assert events[2].delta == "hello there"
    assert events[3].partial.content == "hello there"
    # the system prompt went first on the wire, then the user message
    sent = model.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": "be brief"}
    assert sent[1] == {"role": "user", "content": "hi"}


def test_tool_calls_stream_one_block_each_and_get_ids_by_position():
    reply = ModelReply(
        content="looking",
        tool_calls=[
            ToolCallRequest(name="read", arguments={"path": "a"}),
            ToolCallRequest(id="given", name="read", arguments={"path": "b"}),
        ],
    )
    events = events_of(TestModel([reply]), [UserMessage(content="go")])
    assert [e.type for e in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    done = events[-1]
    assert done.reason == "tool_use"
    assert [c.id for c in done.message.tool_calls] == ["call_1_0", "given"]
    assert json.loads(events[5].delta) == {"path": "a"}
    assert events[6].tool_call.name == "read"
    assert len(events[6].partial.tool_calls) == 1
    assert len(events[9].partial.tool_calls) == 2


def test_provider_error_is_an_error_event_not_an_exception():
    class Broken(pv.Model):
        name = "broken"

        def query(self, messages, tools=None, config=None):
            raise pv.ProviderError("HTTP 500: no", status=500)

    events = events_of(Broken(), [UserMessage(content="hi")])
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "error"
    assert error.error.stop_reason == "error"
    assert "HTTP 500" in (error.error.error_message or "")
    assert error.error.model == "broken"


def test_stop_reasons_fold_onto_ours():
    assert normalize_stop_reason("end_turn", False) == "stop"
    assert normalize_stop_reason("stop_sequence", False) == "stop"
    assert normalize_stop_reason("tool_use", True) == "tool_use"
    assert normalize_stop_reason("tool_calls", True) == "tool_use"
    assert normalize_stop_reason("max_tokens", False) == "length"
    assert normalize_stop_reason("length", False) == "length"
    # unknown or missing: the shape decides
    assert normalize_stop_reason(None, True) == "tool_use"
    assert normalize_stop_reason("whatever", False) == "stop"


def test_assemble_keeps_usage_and_model():
    reply = ModelReply(content="x", usage=pv.Usage(input=3, output=4), model="m", stop_reason="end_turn")
    message = assemble(reply)
    assert message.usage.input == 3 and message.usage.output == 4
    assert message.model == "m"
    assert message.stop_reason == "stop"


def test_a_recorded_model_of_block_messages_streams_text_then_a_tool_call_with_its_recorded_id(tmp_path):
    run = tmp_path / "run.jsonl"
    lines = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "one"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "look", "input": {"k": 1}}],
            "stop_reason": "tool_use",
        },
    ]
    run.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    model = RecordedModel(run)
    first = events_of(model, [UserMessage(content="hi")])[-1]
    assert first.type == "done" and first.message.content == "one"
    second = events_of(model, [UserMessage(content="hi")])[-1]
    assert second.type == "done" and second.reason == "tool_use"
    assert second.message.tool_calls[0].id == "t1"
    assert second.message.tool_calls[0].arguments == {"k": 1}


def test_memo_model_streams_and_second_call_is_a_hit(tmp_path):
    inner = TestModel(["memoed"])
    model = MemoModel(inner, tmp_path)
    messages = [UserMessage(content="same")]
    a = events_of(model, messages)[-1]
    b = events_of(model, messages)[-1]
    assert a.message.content == b.message.content == "memoed"
    assert model.calls == 2 and model.hits == 1
    assert len(inner.calls) == 1


def test_to_wire_drops_details_and_empty_error_turns():
    messages = [
        UserMessage(content="q", details={"secret": 1}),
        AssistantMessage(content=None, stop_reason="error", error_message="boom"),
        AssistantMessage(content="", tool_calls=[{"id": "c1", "name": "t", "arguments": {"a": 1}}]),
        ToolResultMessage(tool_call_id="c1", tool_name="t", content="ok", details={"full": [1, 2]}),
    ]
    wire = to_wire(messages)
    assert wire == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": {"a": 1}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "ok"},
    ]
    assert "secret" not in json.dumps(wire) and "full" not in json.dumps(wire)
