"""The OmniRoute route: with KULLBACK_ROUTE=omniroute every model id goes through the local gateway."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from kullback.ai import calllog as calls
from kullback.ai import provider as pv
from kullback.ai.provider import ModelConfig
from kullback.ai.usage import Usage

ROUTED = {pv.ROUTE_VAR: "omniroute", pv.OMNIROUTE_KEY_VAR: "test-key"}

# A chat completions reply in the shape the gateway returns for a cached Gemini call: cachedContentTokenCount
# mapped to prompt_tokens_details.cached_tokens, thoughtsTokenCount to completion_tokens_details.reasoning_tokens.
REPLY = {"id": "chatcmpl-1", "object": "chat.completion", "model": "gemini-4-argon",
         "choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "{\"applies\": true, \"holds\": true}"}}],
         "usage": {"prompt_tokens": 5000, "completion_tokens": 140, "total_tokens": 5140,
                   "prompt_tokens_details": {"cached_tokens": 2951},
                   "completion_tokens_details": {"reasoning_tokens": 120}}}


@pytest.fixture(autouse=True)
def no_call_log():
    """No test leaves a call sink or its context behind for the next."""
    yield
    calls.install(None)
    calls.set_default()


ANSWERED = {"X-OmniRoute-Provider": "gemini", "X-OmniRoute-Model": "gemini-4-argon", "x-correlation-id": "req-7",
            "X-OmniRoute-Response-Cost": "0.0012"}


def _ok(request):
    return httpx.Response(200, json=REPLY, headers=ANSWERED)


def _routed_model(handler, env=None):
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})
    return pv.model_for("google/gemini-4-argon", env=dict(ROUTED, **(env or {})),
                        client=httpx.Client(transport=httpx.MockTransport(handler)),
                        sleep=lambda _s: None, rng=random.Random(0))


def test_with_the_route_set_every_model_goes_to_the_gateway_and_without_it_nothing_changes():
    assert isinstance(pv.model_for("google/gemini-4-argon", env=ROUTED), pv.OmniRouteModel)
    assert isinstance(pv.model_for("anthropic/claude-opus-5-5", env=ROUTED), pv.OmniRouteModel)
    assert not isinstance(pv.model_for("google/gemini-4-argon", env={}), pv.OmniRouteModel)


def test_a_routed_call_carries_the_bearer_key_the_session_tag_and_the_headers_that_keep_prompt_and_answer_ours(
        tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return _ok(request)

    calls.begin(tmp_path, "audit-review", run_id="exp-9")
    _routed_model(handler).query([{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
    (request,) = seen
    assert str(request.url) == pv.OMNIROUTE_BASE_URL + "/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-omniroute-compression"] == "off"
    assert request.headers["x-omniroute-no-cache"] == "true"
    assert request.headers["x-omniroute-no-memory"] == "true"
    assert request.headers["x-omniroute-session-id"] == "exp-9.audit-review"


def test_a_routed_body_keeps_temperature_one_the_effort_and_the_gateways_model_id():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _ok(request)

    model = _routed_model(handler)
    model.query([{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0, effort="medium"))
    (body,) = bodies
    assert (body["model"], body["temperature"], body["reasoning_effort"]) == ("gemini/gemini-4-argon", 1.0, "medium")
    assert model.name == "google/gemini-4-argon"  # priced and logged under our own id


def test_a_routed_reply_reads_cached_input_and_thinking_and_the_call_log_records_them_with_the_request_id(tmp_path):
    calls.begin(tmp_path, "checks", cost=lambda *_: 0.0)
    reply = _routed_model(_ok).query([{"role": "user", "content": "hi"}])
    assert reply.usage == Usage(input=2049, cache_read=2951, output=140, reasoning=120)
    assert reply.model == "google/gemini-4-argon"
    (row,) = calls.read(tmp_path)
    assert (row["model"], row["input"], row["cache_read"], row["reasoning"]) == ("google/gemini-4-argon", 2049, 2951,
                                                                                120)
    assert row["request_id"] == "req-7"


@pytest.mark.parametrize("headers", [
    dict(ANSWERED, **{"X-OmniRoute-Provider": "nvidia", "X-OmniRoute-Model": "some-other-model"}),
    dict(ANSWERED, **{"X-OmniRoute-Emergency-Fallback": "true"}),
    {"X-OmniRoute-Model": "some-other-model", "x-correlation-id": "req-8"},
])
def test_an_answer_from_another_model_or_a_fallback_is_an_error_logged_with_its_usage_never_a_silent_answer(
        tmp_path, headers):
    calls.begin(tmp_path, "checks", cost=lambda *_: 0.0)
    model = _routed_model(lambda request: httpx.Response(200, json=REPLY, headers=headers))
    with pytest.raises(pv.ModelMismatch, match="not the model asked"):
        model.query([{"role": "user", "content": "hi"}])
    (row,) = calls.read(tmp_path)
    assert row["error"].startswith("ModelMismatch") and row["cache_read"] == 2951 and row["request_id"]


def test_a_routed_call_with_no_gateway_key_is_refused_by_name_before_anything_is_sent():
    sent = []
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})
    model = pv.model_for("google/gemini-4-argon", env={pv.ROUTE_VAR: "omniroute"},
                         client=httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r))))
    with pytest.raises(Exception, match=pv.OMNIROUTE_KEY_VAR):
        model.query([{"role": "user", "content": "hi"}])
    assert sent == []


def _body(model_id, messages, effort=None):
    model = pv.model_for(model_id, env=ROUTED)
    return model.build_body(messages, None, ModelConfig(temperature=1.0, reasoning_effort=effort))


def _marks(body):
    return [i for i, m in enumerate(body["messages"]) if isinstance(m.get("content"), list)
            and any("cache_control" in block for block in m["content"])]


LOOP = [{"role": "system", "content": "S"}] + [
    {"role": role, "content": f"{role} {i}"} for i in range(4) for role in ("user", "assistant")] + [
    {"role": "user", "content": "last"}]


def test_a_route_that_caches_only_where_marked_gets_at_most_four_marks_and_the_other_routes_get_none():
    for model_id in ("cc/model-a", "bedrock/us.vendor.model-a"):
        marks = _marks(_body(model_id, LOOP))
        assert 0 in marks and len(LOOP) - 1 in marks and len(marks) <= 4
    for model_id in ("cu/model-a", "cx/model-a", "gemini/model-a"):
        assert _marks(_body(model_id, LOOP)) == []


def test_a_claude_route_that_returns_thinking_text_asks_for_it_summarized_and_other_routes_never_do():
    assert _body("bedrock/us.anthropic.model-a", LOOP, "medium").get("_omnirouteCopilotReasoningSummary") == "summarized"
    assert "_omnirouteCopilotReasoningSummary" not in _body("bedrock/us.anthropic.model-a", LOOP)
    assert "_omnirouteCopilotReasoningSummary" not in _body("cc/model-a", LOOP, "medium")
    assert "_omnirouteCopilotReasoningSummary" not in _body("cx/model-a", LOOP, "medium")


def test_a_reply_with_no_thinking_count_logs_reasoning_as_unknown_with_the_cost_and_what_was_sent(tmp_path):
    reply = dict(REPLY, usage={"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.001})
    headers = dict(ANSWERED, **{"X-OmniRoute-Response-Cost": "0.002"})
    calls.begin(tmp_path, "s")
    _routed_model(lambda r: httpx.Response(200, json=reply, headers=headers)).query(
        [{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0, reasoning_effort="low"))
    (row,) = calls.read(tmp_path)
    assert row["reasoning"] is None
    assert (row["usd_gateway"], row["usd_billed"], row["usd_actual"]) == (0.002, 0.001, 0.001)
    assert (row["temperature_sent"], row["effort_sent"]) == (1.0, "low")


def test_an_id_the_gateway_key_does_not_allow_is_an_error_that_names_the_id():
    refused = lambda r: httpx.Response(403, json={"error": {"message": "model not allowed"}})  # noqa: E731
    with pytest.raises(pv.ProviderError, match="gemini-4-argon is not allowed by the gateway key"):
        _routed_model(refused).query([{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))


PRICING = {"gemini": {"gemini-4-argon": {"input": 2, "cached": 0.1, "cache_creation": 2, "output": 10}}}


def test_a_reply_with_no_cost_header_is_priced_from_the_gateways_pricing_table_read_once(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/api/pricing"):
            return httpx.Response(200, json=PRICING)
        return httpx.Response(200, json=REPLY, headers={k: v for k, v in ANSWERED.items() if "Cost" not in k})

    pv._PRICING.clear()
    calls.begin(tmp_path, "s")
    model = _routed_model(handler)
    for _ in range(2):
        model.query([{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
    assert seen.count("/api/pricing") == 1
    expected = round((2049 * 2 + 2951 * 0.1 + 140 * 10) / 1e6, 8)
    assert [(r["usd_gateway"], r["price_source"]) for r in calls.read(tmp_path)] == [(expected, "pricing")] * 2
    pv._PRICING.clear()


def test_a_cost_header_of_zero_means_no_price_and_reads_no_pricing_table(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=REPLY, headers=dict(ANSWERED, **{"X-OmniRoute-Response-Cost": "0"}))

    calls.begin(tmp_path, "s")
    _routed_model(handler).query([{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
    (row,) = calls.read(tmp_path)
    assert all(not path.endswith("/api/pricing") for path in seen)
    assert (row["usd_gateway"], row["usd_actual"], row["price_source"]) == (0.0, 0.0, None)


def test_the_row_keeps_the_gateways_latency_and_the_session_tag_we_sent(tmp_path):
    headers = dict(ANSWERED, **{"x-omniroute-latency-ms": "812"})
    calls.begin(tmp_path, "review", run_id="exp-3")
    _routed_model(lambda r: httpx.Response(200, json=REPLY, headers=headers)).query(
        [{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
    (row,) = calls.read(tmp_path)
    assert (row["gateway_ms"], row["session_tag"]) == (812.0, "exp-3.review")


def test_a_reply_with_no_model_header_is_checked_by_the_model_its_body_names(tmp_path):
    bare = {k: v for k, v in ANSWERED.items() if k not in ("X-OmniRoute-Provider", "X-OmniRoute-Model")}
    _routed_model(lambda r: httpx.Response(200, json=REPLY, headers=bare)).query(
        [{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
    other = dict(REPLY, model="some-other-model")
    with pytest.raises(pv.ModelMismatch):
        _routed_model(lambda r: httpx.Response(200, json=other, headers=bare)).query(
            [{"role": "user", "content": "hi"}], config=ModelConfig(temperature=1.0))
