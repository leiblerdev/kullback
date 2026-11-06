"""A streamed call is logged and checked as a blocking one is: one row, with tokens and cost, and no other model."""

import asyncio
import json
import random
import threading
from types import SimpleNamespace

import httpx
import pytest

import kullback.ai.calllog as calls
import kullback.ai.provider as pv
import kullback.ai.streamlog as streamlog
from kullback.ai._provider_events import ProviderErrorEvent
from kullback.ai._sse import StreamFacts
from kullback.ai.events import StreamDone, StreamError
from kullback.ai.messages import UserMessage
from kullback.ai.openai_compatible import OpenAICompatibleProvider
from kullback.ai.provider import ModelConfig

ROUTED = {pv.ROUTE_VAR: "omniroute", pv.OMNIROUTE_KEY_VAR: "test-key"}
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400},
         "completion_tokens_details": {"reasoning_tokens": 30}}
PRICES = {"route-a": {"model-a": {"input": 2.0, "cached": 0.1, "cache_creation": 2.5, "output": 10.0}}}


@pytest.fixture(autouse=True)
def fresh_log_and_prices():
    pv._PRICING.clear()
    yield
    pv._PRICING.clear()
    calls.install(None)


def _sse(*chunks) -> bytes:
    return "".join(f"data: {json.dumps(c) if isinstance(c, dict) else c}\n\n" for c in chunks).encode()


def _answer(model="model-a", usage_on_choice=False) -> bytes:
    last = {"model": model, "choices": [{"delta": {}, "finish_reason": "stop"}]}
    if usage_on_choice:
        last["choices"][0]["usage"] = USAGE
        return _sse({"model": model, "choices": [{"delta": {"content": "ok"}}]}, last, "[DONE]")
    return _sse({"model": model, "choices": [{"delta": {"content": "ok"}}]}, last,
                {"model": model, "choices": [], "usage": USAGE}, {"model": model, "choices": [], "usage": None}, "[DONE]")


def _provider(stream, prices=PRICES, wire="route-a/model-a", on_price_read=lambda: None):
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})

    def blocking(request):
        on_price_read()
        return httpx.Response(200, json=prices)

    handle = pv.model_for(wire, env=ROUTED, client=httpx.Client(transport=httpx.MockTransport(blocking)),
                          sleep=lambda _s: None, rng=random.Random(0))
    return OpenAICompatibleProvider(handle, client=httpx.AsyncClient(transport=httpx.MockTransport(stream)))


def _turn(provider) -> list:
    async def run():
        return [event async for event in provider.stream_response(
            messages=[UserMessage(content="hi")], config=ModelConfig(temperature=1.0))]
    return asyncio.run(run())


def test_a_streamed_turn_writes_one_row_with_its_tokens_and_a_cost_from_the_pricing_table(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    events = _turn(_provider(lambda r: httpx.Response(200, content=_answer(), headers={
        "x-correlation-id": "c-1", "X-OmniRoute-Response-Cost": "0.0000000000"})))
    assert isinstance(events[-1], StreamDone)
    (row,) = calls.read(tmp_path)
    assert (row["input"], row["cache_read"], row["output"], row["reasoning"]) == (600, 400, 100, 30)
    assert row["usd_actual"] == pytest.approx((600 * 2.0 + 400 * 0.1 + 100 * 10.0) / 1e6)
    assert row["price_source"] == "pricing" and row["request_id"] == "c-1" and row["wall_ms"] >= 0
    assert row["http_attempts"] == 1 and "error" not in row and row["temperature_sent"] == 1.0


def test_the_usage_of_a_stream_is_read_from_a_choice_when_no_chunk_has_it_at_the_top(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    _turn(_provider(lambda r: httpx.Response(200, content=_answer(usage_on_choice=True))))
    (row,) = calls.read(tmp_path)
    assert row["input"] == 600 and row["usd_actual"] > 0


def test_a_stream_answered_by_another_model_named_in_the_headers_fails_and_is_logged(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    events = _turn(_provider(lambda r: httpx.Response(200, content=_answer(), headers={
        "X-OmniRoute-Provider": "route-b", "X-OmniRoute-Model": "model-b"})))
    assert isinstance(events[-1], StreamError) and "model-b" in str(events[-1].error.error_message)
    (row,) = calls.read(tmp_path)
    assert "ModelMismatch" in row["error"]


def test_a_stream_answered_by_another_model_named_only_in_its_chunks_fails_before_any_text_passes(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    events = _turn(_provider(lambda r: httpx.Response(200, content=_answer(model="model-b"))))
    assert isinstance(events[-1], StreamError)
    assert not [e for e in events if "ok" in str(getattr(e, "delta", ""))]
    assert "ModelMismatch" in calls.read(tmp_path)[0]["error"]


def test_a_stream_that_never_names_a_model_is_not_refused(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [], "usage": USAGE}, "[DONE]")
    events = _turn(_provider(lambda r: httpx.Response(200, content=body)))
    assert isinstance(events[-1], StreamDone) and "error" not in calls.read(tmp_path)[0]


def test_a_stream_that_reports_no_usage_has_an_unknown_cost_never_a_free_one(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    body = _sse({"model": "model-a", "choices": [{"delta": {"content": "ok"}}]}, "[DONE]")
    _turn(_provider(lambda r: httpx.Response(200, content=body)))
    (row,) = calls.read(tmp_path)
    assert row["input"] == 0 and row["usd_actual"] is None


def test_a_refused_stream_writes_one_row_with_the_error_and_the_failed_status(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    provider = _provider(lambda r: httpx.Response(403, json={"error": {"message": "not allowed"}}))
    events = _turn(provider)
    assert isinstance(events[-1], StreamError)
    (row,) = calls.read(tmp_path)
    assert "not allowed" in row["error"] and row["failed_statuses"] == [403] and row["input"] == 0


def test_a_streamed_shape_retry_is_one_row_counting_both_requests(tmp_path):
    bodies = []

    def stream(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {
                "message": "Function tools with reasoning_effort are not supported; set reasoning_effort to 'none'."}})
        return httpx.Response(200, content=_answer())

    calls.begin(tmp_path, "s", run_id="r1")
    provider = _provider(stream)

    async def run():
        return [event async for event in provider.stream_response(
            messages=[UserMessage(content="hi")], tools=[{"name": "echo", "parameters": {"type": "object"}}],
            config=ModelConfig(temperature=1.0))]

    asyncio.run(run())
    (row,) = calls.read(tmp_path)
    assert row["http_attempts"] == 2 and row["failed_statuses"] == [400] and "error" not in row


def test_a_stream_the_consumer_closes_early_still_writes_its_row(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    provider = _provider(lambda r: httpx.Response(200, content=_answer()))

    async def run():
        stream = provider.stream_provider_events(messages=[UserMessage(content="hi")],
                                                 config=ModelConfig(temperature=1.0))
        async for _event in stream:
            break
        await stream.aclose()

    asyncio.run(run())
    (row,) = calls.read(tmp_path)
    assert "closed" in row["error"]


def test_the_error_event_type_of_a_refusal_is_the_one_every_stream_failure_uses(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    provider = _provider(lambda r: httpx.Response(200, content=_answer(model="model-b")))

    async def run():
        return [e async for e in provider.stream_provider_events(messages=[UserMessage(content="hi")],
                                                                 config=ModelConfig(temperature=1.0))]

    assert isinstance(asyncio.run(run())[-1], ProviderErrorEvent)


def _until(event):
    """Other work on the loop, which runs only while the loop is free: it waits for `event`, and says so."""
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.001)
    return wait()


def test_other_work_on_the_loop_runs_while_a_stream_row_reads_the_price_table(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    asked, loop_ran, read_saw_the_loop = threading.Event(), threading.Event(), []

    def slow_read():
        asked.set()
        read_saw_the_loop.append(loop_ran.wait(2.0))

    provider = _provider(lambda r: httpx.Response(200, content=_answer()), on_price_read=slow_read)

    async def run():
        async def other_work():
            await _until(asked)
            loop_ran.set()

        other = asyncio.create_task(other_work())
        events = [event async for event in provider.stream_response(
            messages=[UserMessage(content="hi")], config=ModelConfig(temperature=1.0))]
        await other
        return events

    assert isinstance(asyncio.run(run())[-1], StreamDone)
    assert read_saw_the_loop == [True]
    (row,) = calls.read(tmp_path)
    assert row["usd_actual"] == pytest.approx((600 * 2.0 + 400 * 0.1 + 100 * 10.0) / 1e6)
    assert row["price_source"] == "pricing"


def test_a_stream_row_has_a_null_cost_when_the_price_table_cannot_be_read(tmp_path):
    def unreadable():
        raise httpx.ConnectError("the gateway is away")

    calls.begin(tmp_path, "s", run_id="r1")
    _turn(_provider(lambda r: httpx.Response(200, content=_answer()), on_price_read=unreadable))
    (row,) = calls.read(tmp_path)
    assert row["input"] == 600 and (row["usd_actual"], row["price_source"]) == (None, None)


def test_a_stream_with_logging_off_never_reads_the_price_table(monkeypatch):
    monkeypatch.delenv(calls.CALLS_DIR_VAR, raising=False)
    reads = []
    events = _turn(_provider(lambda r: httpx.Response(200, content=_answer()), on_price_read=lambda: reads.append(1)))
    assert isinstance(events[-1], StreamDone) and reads == []


def test_a_blocking_call_with_logging_off_never_reads_the_price_table(monkeypatch):
    monkeypatch.delenv(calls.CALLS_DIR_VAR, raising=False)
    handle = _provider(lambda r: httpx.Response(200), on_price_read=lambda: pytest.fail("read the price table")).handle
    assert "usd_priced" not in handle._usage_meta(USAGE, priced=True)


def test_a_slow_price_read_is_not_counted_in_the_wall_time_of_the_stream(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(streamlog, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def slow_read():
        clock[0] += 60.0

    calls.begin(tmp_path, "s", run_id="r1")
    _turn(_provider(lambda r: httpx.Response(200, content=_answer()), on_price_read=slow_read))
    assert calls.read(tmp_path)[0]["wall_ms"] == 0


def test_a_consumer_cancelled_during_the_price_read_still_closes_the_stream_and_writes_the_row(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    asked, let_go, closed = threading.Event(), threading.Event(), []

    def held_read():
        asked.set()
        let_go.wait(2.0)

    handle = _provider(lambda r: httpx.Response(200), on_price_read=held_read).handle
    facts = StreamFacts()
    facts.parser = SimpleNamespace(raw_usage=USAGE, body_model="model-a", _usage=None)

    async def events():
        try:
            yield SimpleNamespace()
            yield SimpleNamespace()
        finally:
            closed.append(True)

    async def consume():
        stream = streamlog.logged_stream(handle, events(), [{"role": "user", "content": "hi"}], None,
                                         ModelConfig(temperature=1.0), facts)
        await stream.__anext__()
        await stream.aclose()

    async def run():
        task = asyncio.create_task(consume())
        await _until(asked)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled() and closed == [True]
        let_go.set()

    asyncio.run(run())
    (row,) = calls.read(tmp_path)
    assert "closed" in row["error"]


def test_a_stream_row_written_off_the_loop_carries_the_context_of_the_call(tmp_path):
    calls.begin(tmp_path, "s", run_id="r1")
    provider = _provider(lambda r: httpx.Response(200, content=_answer()))

    async def run():
        with calls.labelled("the-label", attempt=3):
            return [event async for event in provider.stream_response(
                messages=[UserMessage(content="hi")], config=ModelConfig(temperature=1.0))]

    asyncio.run(run())
    (row,) = calls.read(tmp_path)
    assert (row["label"], row["attempt"], row["session_tag"]) == ("the-label", 3, "r1.s")
