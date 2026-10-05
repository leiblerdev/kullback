"""One rate limiter per provider key, shared across callers. Fake transports only."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time

import httpx
import pytest

from kullback.ai import limits, pricing
from kullback.ai import provider as pv
from kullback.ai._provider_events import ProviderErrorEvent, ProviderRetry
from kullback.ai._sse import stream_sse_events
from kullback.ai.limits import Bucket, RateSpec
from kullback.ai.openai_compatible import ChatStreamParser
from kullback.ai.retry import RetryPolicy, rate_limit_delay

OK = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

CHAT_CHUNKS = [
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "Hel"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "lo"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}],'
    ' "usage": {"prompt_tokens": 12, "completion_tokens": 3}}',
    "[DONE]",
]


def sse(chunks):
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode("utf-8")


def transport_of(handler):
    """An httpx client whose every request is answered by handler; nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def streaming_client(responses):
    """An async client answering each request from the (status, headers, body) queue in order."""
    calls = []

    def handle(request):
        calls.append(request)
        status, headers, body = responses[min(len(calls) - 1, len(responses) - 1)]
        return httpx.Response(status, headers=headers, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


class FakeClock:
    """A monotonic clock a test moves by hand, for the waits no test should sit through."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def live():
    """Adapters refuse to run while ALLOW_MODEL_REQUESTS is False; the mock transport keeps it offline."""
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})


@pytest.fixture(autouse=True)
def clean_buckets():
    limits.reset()
    yield
    limits.reset()


def probe_model(model_id, handler, sleeps=None, **kwargs):
    kwargs.setdefault("api_key", "k")
    return pv.OpenAICompatibleModel(
        model_id=model_id,
        base_url="https://probe.invalid/v1",
        env={},
        client=transport_of(handler),
        sleep=sleeps.append if sleeps is not None else None,
        rng=random.Random(0),
        **kwargs,
    )


@pytest.fixture
def sleeps():
    return []


def write_rates(tmp_path, entry):
    """A local provider registry beside the snapshot conftest already points at tmp_path."""
    (tmp_path / pricing.LOCAL_PROVIDERS_NAME).write_text(json.dumps(entry), encoding="utf-8")


def test_without_a_rate_row_a_call_waits_for_nothing_and_counts_nothing(live, sleeps, tmp_path):
    write_rates(tmp_path, {})
    model = probe_model("probe/plain", lambda request: httpx.Response(200, json=OK), sleeps)

    assert model.rate_bucket is None
    assert model.query([{"role": "user", "content": "hi"}]).content == "ok"
    assert sleeps == []
    assert limits.snapshot() == {}


def test_two_in_flight_slots_shared_by_five_callers():
    bucket = Bucket("two", in_flight=2)
    guard = threading.Lock()
    active = {"n": 0, "max": 0}

    def call():
        with bucket.slot():
            with guard:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.02)
            with guard:
                active["n"] -= 1

    threads = [threading.Thread(target=call) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert active["max"] <= 2
    assert bucket.counts()["calls"] == 5
    assert bucket.counts()["waits"] >= 1


def test_a_minute_cap_lets_n_starts_through_then_holds_the_next_until_the_window_passes():
    clock = FakeClock()
    bucket = Bucket("capped", requests_per_minute=2, clock=clock)

    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
    clock.advance(59.9)
    assert not bucket.try_acquire()
    clock.advance(0.2)
    assert bucket.try_acquire()


def test_a_429_cooldown_parks_that_bucket_for_the_header_and_leaves_others_alone():
    clock = FakeClock()
    first = Bucket("first", clock=clock)
    second = Bucket("second", clock=clock)

    first.note_rate_limited(3.0)

    assert not first.try_acquire()
    assert second.try_acquire()
    clock.advance(2.9)
    assert not first.try_acquire()
    clock.advance(0.2)
    assert first.try_acquire()
    assert first.counts()["rate_limited"] == 1


def test_a_row_rate_is_that_models_own_bucket_and_a_provider_rate_is_shared():
    catalog = {
        "probe": {
            "rate": {"requests_per_minute": 60, "in_flight": 5},
            "models": {"m-a": {"rate": {"requests_per_minute": 10}}},
        }
    }
    row = pricing.rate_from_catalog(catalog, "probe/m-a")
    assert row == RateSpec(requests_per_minute=10, in_flight=None, model_specific=True)
    shared = pricing.rate_from_catalog(catalog, "probe/m-b")
    assert shared == RateSpec(requests_per_minute=60, in_flight=5, model_specific=False)

    own = limits.bucket_for(provider="probe", host="h", key="fp", key_name="K", model="probe/m-a", spec=row)
    one = limits.bucket_for(provider="probe", host="h", key="fp", key_name="K", model="probe/m-b", spec=shared)
    two = limits.bucket_for(provider="probe", host="h", key="fp", key_name="K", model="probe/m-c", spec=shared)
    assert one is two
    assert own is not one
    assert "K" in one.name


def test_a_rate_with_no_usable_cap_means_no_limiter():
    catalog = {"probe": {"rate": {"requests_per_minute": -3, "in_flight": "many"}, "models": {}}}
    assert pricing.rate_from_catalog(catalog, "probe/m") is None
    assert pricing.rate_from_catalog({"other": {}}, "probe/m") is None
    assert pricing.rate_from_catalog(None, "probe/m") is None


def test_a_429_delay_is_the_header_when_sent_and_the_backoff_otherwise():
    policy = RetryPolicy()
    assert rate_limit_delay({"Retry-After": "7"}, 1, policy, random.Random(0)) == 7.0
    capped = rate_limit_delay({"Retry-After": "900"}, 1, policy, random.Random(0))
    assert capped == policy.max_retry_after_s


def test_streaming_429_waits_the_header_capped_by_the_handle_policy(live):
    bucket = Bucket("stream")
    client = streaming_client([
        (429, {"Retry-After": "30"}, b'{"error": {"message": "slow"}}'),
        (200, {}, sse(CHAT_CHUNKS)),
    ])

    async def run():
        return [event async for event in stream_sse_events(
            client=lambda: client,
            url="https://probe.invalid/v1/chat/completions",
            headers={},
            payload={"model": "m"},
            parser_factory=ChatStreamParser,
            parser_name="probe",
            max_retries=1,
            retry_policy=RetryPolicy(max_retry_after_s=0.2),
            rng=random.Random(0),
            rate_bucket=bucket,
        )]

    events = asyncio.run(run())
    retries = [event for event in events if isinstance(event, ProviderRetry)]
    assert len(retries) == 1
    assert retries[0].delay_seconds == pytest.approx(0.2)
    assert bucket.counts()["rate_limited"] == 1
    assert bucket.try_acquire()
    assert not any(isinstance(event, ProviderErrorEvent) for event in events)


def test_a_slot_is_released_when_the_blocking_call_raises(live, sleeps, tmp_path):
    write_rates(tmp_path, {"probe": {
        "id": "probe", "api": "https://probe.invalid/v1", "env": ["PROBE_API_KEY"],
        "rate": {"in_flight": 1},
    }})
    model = probe_model("probe/m", lambda request: httpx.Response(400, json={"error": "bad"}), sleeps)

    with pytest.raises(pv.ProviderError):
        model.query([{"role": "user", "content": "hi"}])
    assert model.rate_bucket.counts()["calls"] == 1
    assert model.rate_bucket.try_acquire()


def test_a_slot_is_released_when_the_stream_fails(live):
    bucket = Bucket("failing")
    client = streaming_client([(500, {}, b"down"), (500, {}, b"down")])

    async def run():
        return [event async for event in stream_sse_events(
            client=lambda: client,
            url="https://probe.invalid/v1/chat/completions",
            headers={},
            payload={"model": "m"},
            parser_factory=ChatStreamParser,
            parser_name="probe",
            max_retries=1,
            rate_bucket=bucket,
        )]

    events = asyncio.run(run())
    assert any(isinstance(event, ProviderErrorEvent) for event in events)
    assert bucket.try_acquire()


def test_a_429_on_one_caller_pauses_the_next_caller_on_the_same_key(live, tmp_path):
    write_rates(tmp_path, {"probe": {
        "id": "probe", "api": "https://probe.invalid/v1", "env": ["PROBE_API_KEY"],
        "rate": {"requests_per_minute": 20, "in_flight": 3},
    }})
    first_seen = threading.Event()
    second_arrived = {}

    def first_handler(request):
        if first_seen.is_set():
            return httpx.Response(200, json=OK)
        first_seen.set()
        return httpx.Response(429, headers={"Retry-After": "0.3"}, json={"error": {"message": "slow"}})

    def second_handler(request):
        second_arrived["t"] = time.monotonic()
        return httpx.Response(200, json=OK)

    first = probe_model("probe/m-a", first_handler, [])
    second_sleeps = []
    second = probe_model("probe/m-b", second_handler, second_sleeps)
    assert first.rate_bucket is second.rate_bucket

    first_thread = threading.Thread(
        target=lambda: first.query([{"role": "user", "content": "hi"}]))
    first_thread.start()
    assert first_seen.wait(timeout=5)
    started = time.monotonic()
    second.query([{"role": "user", "content": "hi"}])
    waited = time.monotonic() - started
    first_thread.join(timeout=5)

    assert waited >= 0.25
    assert second_sleeps == []
    counts = first.rate_bucket.counts()
    assert (counts["calls"], counts["rate_limited"]) == (3, 1)
    assert counts["waits"] >= 1


def test_a_retry_releases_its_slot_so_the_next_caller_waits_only_the_shared_cooldown(live, tmp_path):
    """Two callers, one in_flight slot: A draws two 429s then succeeds, B's request is posted
    before A's final attempt, and every attempt is counted."""
    write_rates(tmp_path, {"probe": {
        "id": "probe", "api": "https://probe.invalid/v1", "env": ["PROBE_API_KEY"],
        "rate": {"in_flight": 1},
    }})
    a_seen = threading.Event()
    order = []
    a_count = [0]

    def a_handler(request):
        a_count[0] += 1
        order.append(f"a{a_count[0]}")
        if a_count[0] == 1:
            a_seen.set()
        if a_count[0] <= 2:
            return httpx.Response(429, headers={"Retry-After": "0.3"}, json={"error": {"message": "slow"}})
        return httpx.Response(200, json=OK)

    def b_handler(request):
        order.append("b")
        return httpx.Response(200, json=OK)

    first = probe_model("probe/m-a", a_handler, None)
    second = probe_model("probe/m-b", b_handler, None)
    assert first.rate_bucket is second.rate_bucket

    first_thread = threading.Thread(target=lambda: first.query([{"role": "user", "content": "hi"}]))
    first_thread.start()
    assert a_seen.wait(timeout=5)
    deadline = time.monotonic() + 5
    while first.rate_bucket.counts()["rate_limited"] == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert second.query([{"role": "user", "content": "hi"}]).content == "ok"
    first_thread.join(timeout=10)
    assert not first_thread.is_alive()

    assert order[0] == "a1" and "b" in order
    counts = first.rate_bucket.counts()
    assert (counts["calls"], counts["rate_limited"]) == (4, 2)
    assert counts["waits"] >= 1


def test_two_handles_with_different_key_values_share_no_bucket(live, tmp_path):
    """The scope is the key value's fingerprint: one variable, two values, two buckets."""
    write_rates(tmp_path, {"probe": {
        "id": "probe", "api": "https://probe.invalid/v1", "env": ["PROBE_API_KEY"],
        "rate": {"in_flight": 1},
    }})
    handler = lambda request: httpx.Response(200, json=OK)  # noqa: E731
    one = probe_model("probe/m", handler, [], api_key="first-value")
    two = probe_model("probe/m", handler, [], api_key="second-value")
    same = probe_model("probe/m", handler, [], api_key="first-value")

    assert one.rate_bucket is not two.rate_bucket
    assert one.rate_bucket is same.rate_bucket
    assert "PROBE_API_KEY" in one.rate_bucket.name
    assert "first-value" not in one.rate_bucket.name


def test_a_429_on_the_final_streaming_attempt_still_parks_the_bucket(live):
    bucket = Bucket("last-429")
    client = streaming_client([
        (429, {"Retry-After": "5"}, b'{"error": {"message": "slow"}}'),
        (429, {"Retry-After": "5"}, b'{"error": {"message": "slow"}}'),
    ])

    async def run():
        return [event async for event in stream_sse_events(
            client=lambda: client,
            url="https://probe.invalid/v1/chat/completions",
            headers={},
            payload={"model": "m"},
            parser_factory=ChatStreamParser,
            parser_name="probe",
            max_retries=1,
            retry_policy=RetryPolicy(max_retry_after_s=0.2),
            rng=random.Random(0),
            rate_bucket=bucket,
        )]

    events = asyncio.run(run())
    assert any(isinstance(event, ProviderErrorEvent) for event in events)
    assert bucket.counts()["rate_limited"] == 2
    assert not bucket.try_acquire()


def test_signing_happens_after_the_slot_is_taken(live, tmp_path):
    """headers() runs inside the slot: while another caller holds the only slot, no signing happens."""
    write_rates(tmp_path, {"probe": {
        "id": "probe", "api": "https://probe.invalid/v1", "env": ["PROBE_API_KEY"],
        "rate": {"in_flight": 1},
    }})
    model = probe_model("probe/m", lambda request: httpx.Response(200, json=OK), None)
    calls = []
    real_headers = model.headers
    model.headers = lambda content=None: calls.append("headers") or real_headers(content)

    model.rate_bucket.acquire()
    thread = threading.Thread(target=lambda: model.query([{"role": "user", "content": "hi"}]))
    thread.start()
    time.sleep(0.3)
    assert calls == []
    model.rate_bucket.release()
    thread.join(timeout=5)

    assert calls == ["headers"]


def test_a_newer_catalogue_row_replaces_the_buckets_caps():
    first = limits.bucket_for(provider="probe", host="h", key="fp", key_name="K", model="probe/m",
                              spec=RateSpec(in_flight=1))
    second = limits.bucket_for(provider="probe", host="h", key="fp", key_name="K", model="probe/m",
                               spec=RateSpec(requests_per_minute=10, in_flight=3))

    assert second is first
    assert (first.requests_per_minute, first.in_flight) == (10, 3)


def test_two_key_values_keep_their_own_counters_in_the_snapshot():
    """Same readable name, different keys: the second bucket reads " #2", the third " #3"."""
    first = limits.bucket_for(provider="probe", host="h", key="fp1", key_name="K", model="probe/m",
                              spec=RateSpec(in_flight=1))
    second = limits.bucket_for(provider="probe", host="h", key="fp2", key_name="K", model="probe/m",
                               spec=RateSpec(in_flight=1))
    third = limits.bucket_for(provider="probe", host="h", key="fp3", key_name="K", model="probe/m",
                              spec=RateSpec(in_flight=1))

    assert (first.name, second.name, third.name) == (
        "probe h K probe/m", "probe h K probe/m #2", "probe h K probe/m #3")
    assert first.try_acquire()
    snap = limits.snapshot()
    assert (snap[first.name]["calls"], snap[second.name]["calls"]) == (1, 0)
