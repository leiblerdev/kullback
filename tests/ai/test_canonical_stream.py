"""The canonicaliser: what an adapter reports becomes the events the loop reads."""

from __future__ import annotations

import asyncio

from kullback.ai._provider_events import (
    ProviderErrorEvent,
    ProviderResponseEnd,
    ProviderResponseStart,
    ProviderRetry,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderToolCall,
)
from kullback.ai.messages import AssistantMessage, ToolCall
from kullback.ai.stream import canonicalize_provider_stream
from kullback.ai.usage import Usage


def canonical(reported, **kwargs):
    async def source():
        for event in reported:
            yield event

    async def run():
        return [event async for event in canonicalize_provider_stream(source(), **kwargs)]

    return asyncio.run(run())


def end(**kwargs):
    return ProviderResponseEnd(message=AssistantMessage(**kwargs), finish_reason=kwargs.get("stop_reason"))


def test_a_retry_is_the_adapters_business_and_never_reaches_the_loop():
    events = canonical([
        ProviderRetry(attempt=2, max_attempts=3, delay_seconds=0.25, message="again"),
        ProviderTextDelta(delta="hi"),
        end(content="hi"),
    ])
    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "done"]


def test_a_channel_switch_closes_the_block_it_left_when_the_transport_keeps_order():
    events = canonical([
        ProviderThinkingDelta(delta="weigh"),
        ProviderTextDelta(delta="say"),
        end(content="say", thinking="weigh"),
    ])
    assert [event.type for event in events] == [
        "start", "thinking_start", "thinking_delta", "thinking_end",
        "text_start", "text_delta", "text_end", "done",
    ]


def test_channels_that_interleave_keep_their_own_blocks_open():
    events = canonical([
        ProviderTextDelta(delta="a"),
        ProviderThinkingDelta(delta="w"),
        ProviderTextDelta(delta="b"),
        end(content="ab", thinking="w"),
    ], independent_channels=True)
    assert [event.type for event in events] == [
        "start", "text_start", "text_delta", "thinking_start", "thinking_delta", "text_delta",
        "text_end", "thinking_end", "done",
    ]
    assert events[-1].message.content == "ab"


def test_a_tool_call_closes_the_open_blocks_and_arrives_whole():
    call = ToolCall(id="c1", name="t", arguments={"a": 1})
    events = canonical([
        ProviderTextDelta(delta="thinking about it"),
        ProviderToolCall(tool_call=call),
        end(tool_calls=[call], stop_reason="tool_use"),
    ])
    assert [event.type for event in events] == [
        "start", "text_start", "text_delta", "text_end",
        "toolcall_start", "toolcall_delta", "toolcall_end", "done",
    ]
    assert events[5].delta == '{"a": 1}'
    assert events[-1].reason == "tool_use"


def test_the_final_message_takes_its_usage_from_the_end_and_its_text_from_the_deltas():
    events = canonical([
        ProviderResponseStart(model="p/m"),
        ProviderTextDelta(delta="one "),
        ProviderTextDelta(delta="two"),
        ProviderResponseEnd(message=AssistantMessage(content="ignored", usage=Usage(input=5, output=2)),
                            finish_reason="stop"),
    ])
    done = events[-1]
    assert done.message.content == "one two"
    assert (done.message.usage.input, done.message.usage.output) == (5, 2)
    assert done.message.model == "p/m"


def test_a_stream_that_stops_without_saying_why_ends_as_an_error():
    events = canonical([ProviderTextDelta(delta="half an answer")])
    assert events[-1].type == "error"
    assert "without a done" in events[-1].error.error_message
    assert events[-1].error.content == "half an answer"


def test_a_reported_error_becomes_an_error_event_carrying_what_was_said_so_far():
    events = canonical([ProviderTextDelta(delta="half"), ProviderErrorEvent(message="the host went away")])
    assert [event.type for event in events] == ["start", "text_start", "text_delta", "error"]
    assert events[-1].error.error_message == "the host went away"
    assert events[-1].error.stop_reason == "error"
