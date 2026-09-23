"""The streaming providers: deltas reach the caller as they arrive, and a replay says the same."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from kullback.ai import provider as pv
from kullback.ai.anthropic import AnthropicProvider
from kullback.ai.messages import UserMessage
from kullback.ai.openai_compatible import OpenAICompatibleProvider
from kullback.ai.provider import AnthropicModel, MemoModel, ModelReply, OpenAIModel, TestModel
from kullback.ai.replay import ReplayProvider

CHAT_CHUNKS = [
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "Hel"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "lo"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}],'
    ' "usage": {"prompt_tokens": 12, "completion_tokens": 3}}',
    "[DONE]",
]


def sse(chunks):
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode("utf-8")


class Gate:
    """A byte stream that hands out one SSE chunk at a time, when a test lets it."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.released = [asyncio.Event() for _ in self.chunks]

    def release(self, index):
        self.released[index].set()

    async def __aiter__(self):
        for index, chunk in enumerate(self.chunks):
            await self.released[index].wait()
            yield f"data: {chunk}\n\n".encode("utf-8")


def streaming_client(body, *, status=200, seen=None, headers=None):
    """An httpx client whose transport answers one streaming response."""

    def handle(request):
        if seen is not None:
            seen.append(json.loads(request.content or b"{}"))
        if isinstance(body, (bytes, bytearray)):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, content=body, headers=headers)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def collect(provider, **kwargs):
    async def run():
        return [event async for event in provider.stream_response(**kwargs)]

    return asyncio.run(run())


@pytest.fixture
def live(monkeypatch):
    """The transports below never leave the machine, so the live-call gate is opened for them."""
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


def chat_provider(client, **kwargs):
    handle = OpenAIModel("openai/m", api_key="k", env={}, **kwargs)
    return OpenAICompatibleProvider(handle, client=client)


def test_chat_deltas_reach_the_caller_before_the_stream_ends(live):
    """The first text delta is readable while the endpoint still owes the rest of the answer,
    which is the whole difference from waiting for a finished reply."""
    gate = Gate(CHAT_CHUNKS)
    client = streaming_client(gate.__aiter__())

    async def run():
        provider = chat_provider(client)
        events = provider.stream_response(messages=[UserMessage(content="hi")])
        iterator = events.__aiter__()
        gate.release(0)
        seen = [await iterator.__anext__() for _ in range(3)]
        assert [event.type for event in seen] == ["start", "text_start", "text_delta"]
        assert seen[2].delta == "Hel"
        # The endpoint has sent nothing but the first chunk; the rest is still owed.
        assert all(not released.is_set() for released in gate.released[1:])
        for index in range(1, len(CHAT_CHUNKS)):
            gate.release(index)
        rest = [event async for event in iterator]
        return seen + rest

    events = asyncio.run(run())
    assert [event.type for event in events] == [
        "start", "text_start", "text_delta", "text_delta", "text_end", "done",
    ]
    done = events[-1]
    assert done.message.content == "Hello"
    assert (done.message.usage.input, done.message.usage.output) == (12, 3)
    assert done.reason == "stop"


def test_chat_streams_tool_calls_assembled_from_their_argument_fragments(live):
    chunks = [
        '{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",'
        ' "function": {"name": "t", "arguments": "{\\"a\\":"}}]}}]}',
        '{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}}]}',
        '{"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
        "[DONE]",
    ]
    provider = chat_provider(streaming_client(sse(chunks)))
    events = collect(provider, messages=[UserMessage(content="hi")], tools=[{"name": "t"}])
    assert [event.type for event in events] == [
        "start", "toolcall_start", "toolcall_delta", "toolcall_end", "done",
    ]
    call = events[-1].message.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("c1", "t", {"a": 1})
    assert events[-1].reason == "tool_use"


def test_chat_reports_a_reasoning_channel_as_its_own_block(live):
    chunks = [
        '{"choices": [{"delta": {"reasoning_content": "weigh"}}]}',
        '{"choices": [{"delta": {"content": "answer"}}]}',
        '{"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        "[DONE]",
    ]
    provider = chat_provider(streaming_client(sse(chunks)))
    events = collect(provider, messages=[UserMessage(content="hi")])
    assert [event.type for event in events] == [
        "start", "thinking_start", "thinking_delta", "text_start", "text_delta",
        "thinking_end", "text_end", "done",
    ]
    assert events[-1].message.thinking == "weigh"
    assert events[-1].message.content == "answer"


def test_the_request_asks_for_a_stream_and_for_the_usage_that_comes_with_it(live):
    seen = []
    provider = chat_provider(streaming_client(sse(CHAT_CHUNKS), seen=seen))
    collect(provider, messages=[UserMessage(content="hi")], system="be brief")
    body = seen[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0] == {"role": "system", "content": "be brief"}


def test_a_refused_request_becomes_an_error_event_rather_than_an_exception(live):
    provider = chat_provider(streaming_client(b'{"error": {"message": "no key"}}', status=401))
    events = collect(provider, messages=[UserMessage(content="hi")])
    assert [event.type for event in events] == ["start", "error"]
    assert "no key" in events[-1].error.error_message
    assert events[-1].error.stop_reason == "error"


def test_anthropic_blocks_are_read_as_they_are_opened_filled_and_stopped(live):
    chunks = [
        '{"type": "message_start", "message": {"usage": {"input_tokens": 9}}}',
        '{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}',
        '{"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta",'
        ' "thinking": "weigh"}}',
        '{"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}',
        '{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}}',
        '{"type": "message_delta", "delta": {"stop_reason": "end_turn"},'
        ' "usage": {"output_tokens": 4}}',
        '{"type": "message_stop"}',
    ]
    handle = AnthropicModel("anthropic/claude", api_key="k", env={})
    provider = AnthropicProvider(handle, client=streaming_client(sse(chunks)))
    events = collect(provider, messages=[UserMessage(content="hi")])
    assert [event.type for event in events] == [
        "start", "thinking_start", "thinking_delta", "thinking_end",
        "text_start", "text_delta", "text_end", "done",
    ]
    done = events[-1]
    assert (done.message.thinking, done.message.content) == ("weigh", "hi")
    assert (done.message.usage.input, done.message.usage.output) == (9, 4)
    assert done.reason == "stop"


def test_a_memo_round_trip_yields_the_events_the_live_stream_yielded(live, tmp_path):
    """The seam that keeps replay deterministic: a reply taken off a live stream, stored and read
    back, is reported through the same canonicaliser and produces the same events.

    The endpoint answers in one chunk here, because a replayed reply is one block and a live one
    that arrived in three fragments would be three deltas of the same text. Joined, they are the
    same stream; the test states the equality it can state exactly."""
    whole = [
        '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "Hello"},'
        ' "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}',
        "[DONE]",
    ]
    provider = chat_provider(streaming_client(sse(whole)))
    live_events = collect(provider, messages=[UserMessage(content="hi")])
    answered = live_events[-1].message

    stored = ModelReply(content=answered.content, usage=answered.usage, model=answered.model,
                        stop_reason=answered.stop_reason)
    inner = TestModel([stored], name=answered.model, loop=True)
    memo = MemoModel(inner, tmp_path)

    miss = collect(ReplayProvider(memo), messages=[UserMessage(content="hi")])
    hit = collect(ReplayProvider(memo), messages=[UserMessage(content="hi")])
    assert (memo.calls, memo.hits) == (2, 1)
    assert len(inner.calls) == 1

    assert miss == live_events
    assert [event.type for event in hit] == [event.type for event in live_events]
    # A hit is priced at nothing, which is the one thing a replayed reply does not carry over.
    assert hit[-1].message.content == answered.content
    assert hit[-1].message.usage.output == 0
