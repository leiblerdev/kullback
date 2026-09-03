"""Tests for the real provider adapters: normalizations, cache points, retry, parsing. No network."""

from __future__ import annotations

import json
import random
import re

import httpx
import pytest

from kullback.ai import provider as pv


def transport_of(handler):
    """An httpx client whose every request is answered by handler; nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def live():
    """Adapters refuse to run while ALLOW_MODEL_REQUESTS is False; the mock transport keeps it offline.

    The flag is turned on through the module's own switch, the one path a person's environment
    takes; conftest's autouse no_live_models restores it to False when the test ends.
    """
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})


@pytest.fixture
def sleeps():
    return []


def anthropic_model(handler, sleeps, **kwargs):
    return pv.AnthropicModel(
        model_id=kwargs.pop("model_id", "anthropic/claude-opus-5"),
        api_key="k",
        client=transport_of(handler),
        sleep=sleeps.append,
        rng=random.Random(0),
        **kwargs,
    )


def ok_anthropic(body=None):
    payload = body or {
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 2,
        },
    }

    def handler(request):
        return httpx.Response(200, json=payload)

    return handler


# --- model ids and base urls ---


def test_split_model_id_gives_provider_and_wire_id():
    assert pv.split_model_id("anthropic/claude-opus-5") == ("anthropic", "claude-opus-5")
    assert pv.split_model_id("openai/gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_split_model_id_keeps_slashes_in_the_wire_id():
    assert pv.split_model_id("local/meta/llama-3.1") == ("local", "meta/llama-3.1")


def test_split_model_id_needs_a_provider():
    with pytest.raises(ValueError):
        pv.split_model_id("claude-opus-5")


def test_wire_id_can_differ_from_the_model_id(sleeps):
    model = anthropic_model(ok_anthropic(), sleeps, model_id="anthropic/opus", wire_id="claude-opus-5")
    assert model.name == "anthropic/opus"
    body = model.build_body([{"role": "user", "content": "hi"}], None, pv.ModelConfig())
    assert body["model"] == "claude-opus-5"


def test_substitute_env_fills_dollar_brace_vars():
    assert pv.substitute_env("${HOST}/v1", {"HOST": "http://localhost:11434"}) == "http://localhost:11434/v1"


def test_substitute_env_raises_on_a_missing_var():
    with pytest.raises(KeyError) as excinfo:
        pv.substitute_env("${NOPE}/v1", {})
    assert "NOPE" in str(excinfo.value)


# --- the three defensive normalizations ---


def test_strip_unpaired_surrogates_keeps_normal_text():
    assert pv.strip_unpaired_surrogates("café \U0001f600") == "café \U0001f600"


def test_strip_unpaired_surrogates_removes_a_lone_surrogate():
    assert pv.strip_unpaired_surrogates("a\ud800b") == "ab"


def test_strip_surrogates_reaches_nested_values():
    cleaned = pv.strip_surrogates_deep({"a": ["x\udc00y", {"b": "\ud83d"}]})
    assert cleaned == {"a": ["xy", {"b": ""}]}


def test_clean_tool_call_id_keeps_allowed_characters():
    assert pv.clean_tool_call_id("call_abc-123") == "call_abc-123"


def test_clean_tool_call_id_replaces_the_rest():
    cleaned = pv.clean_tool_call_id("call:1 2/3")
    assert cleaned.startswith("call_1_2_3_")
    assert not pv.ID_ALLOWED.search(cleaned)


def test_cleaning_keeps_distinct_ids_distinct():
    """Two ids that clean to the same characters must not become the same id: a tool_result
    would then be paired with the wrong tool_use inside one message."""
    assert pv.clean_tool_call_id("a:b") != pv.clean_tool_call_id("a_b")
    assert pv.clean_tool_call_id("a:b") != pv.clean_tool_call_id("a b")
    # And cleaning is a function, not a counter: the same id cleans the same way every time.
    assert pv.clean_tool_call_id("a:b") == pv.clean_tool_call_id("a:b")


def test_clean_tool_call_id_never_returns_empty():
    assert pv.clean_tool_call_id("") == "id"


def test_normalize_messages_drops_empty_messages():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]
    assert pv.normalize_messages(messages) == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_normalize_messages_drops_empty_reasoning_parts_and_keeps_full_ones():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "reasoning", "text": "  "},
                {"type": "thinking", "thinking": "kept"},
                {"type": "text", "text": "answer"},
            ],
        }
    ]
    blocks = pv.normalize_messages(messages)[0]["content"]
    assert [b.get("type") for b in blocks] == ["thinking", "text"]


def test_normalize_messages_keeps_a_message_that_only_calls_a_tool():
    messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a:b", "name": "t", "input": {}}]}]
    out = pv.normalize_messages(messages)
    assert out[0]["content"][0]["id"] == pv.clean_tool_call_id("a:b")
    assert out[0]["content"][0]["id"].startswith("a_b_")


def test_normalize_messages_cleans_tool_result_ids():
    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a b", "content": "r"}]}]
    assert pv.normalize_messages(messages)[0]["content"][0]["tool_use_id"] == pv.clean_tool_call_id("a b")


# --- cache points ---


def test_cache_points_mark_the_system_prompt():
    system = pv.cache_system([{"type": "text", "text": "policy"}])
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_points_mark_the_last_two_non_system_messages():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "one"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "two"}]},
        {"role": "user", "content": [{"type": "text", "text": "three"}]},
    ]
    marked = pv.cache_last_two(messages)
    assert "cache_control" not in marked[0]["content"][-1]
    assert marked[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert marked[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


# --- Anthropic adapter ---


def test_the_anthropic_body_carries_system_tools_stops_and_the_same_cleaned_id_on_call_and_result(live, sleeps):
    seen = {}

    def handler(request):
        seen["body"] = pv.json_body(request)
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a b", "name": "get", "arguments": {"x": 1}}]},
            {"role": "tool", "tool_call_id": "a b", "content": "result"},
        ],
        tools=[{"name": "get", "description": "d", "input_schema": {"type": "object"}}],
        config=pv.ModelConfig(max_tokens=64, temperature=0.0, stop=["END"]),
    )
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["headers"]["anthropic-version"]
    assert body["system"][0]["text"] == "be brief"
    assert body["max_tokens"] == 64
    assert body["stop_sequences"] == ["END"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    # The call and its result keep the same cleaned id, so the pair still matches.
    assert body["messages"][1]["content"][0]["id"] == pv.clean_tool_call_id("a b")
    assert body["messages"][2]["content"][0]["tool_use_id"] == pv.clean_tool_call_id("a b")
    assert body["tools"][0]["name"] == "get"


def test_anthropic_reply_carries_content_tool_calls_and_usage(live, sleeps):
    payload = {
        "content": [
            {"type": "text", "text": "sure"},
            {"type": "tool_use", "id": "t1", "name": "get_order", "input": {"id": "7"}},
        ],
        "model": "claude-opus-5",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    }
    model = anthropic_model(lambda request: httpx.Response(200, json=payload), sleeps)
    reply = model.query([{"role": "user", "content": "hi"}])
    assert reply.content == "sure"
    assert reply.tool_calls[0].name == "get_order"
    assert reply.tool_calls[0].arguments == {"id": "7"}
    assert reply.stop_reason == "tool_use"
    assert reply.usage.input == 11
    assert reply.usage.output == 5
    assert reply.usage.cache_read == 3
    assert reply.usage.cache_write == 2


def test_usage_is_present_even_when_the_provider_omits_it(live, sleeps):
    model = anthropic_model(ok_anthropic({"content": [], "usage": {}}), sleeps)
    reply = model.query([{"role": "user", "content": "hi"}])
    assert reply.usage.input == 0 and reply.usage.output == 0


def test_adapters_refuse_while_live_calls_are_off(sleeps):
    model = anthropic_model(ok_anthropic(), sleeps)
    with pytest.raises(RuntimeError) as excinfo:
        model.query([{"role": "user", "content": "hi"}])
    assert "ALLOW_MODEL_REQUESTS" in str(excinfo.value)


# --- retry ---


def test_retry_on_500_then_success(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    model = anthropic_model(handler, sleeps)
    assert model.query([{"role": "user", "content": "hi"}]).content == "ok"
    assert len(calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("seconds", [7, 90])
def test_retry_on_429_waits_exactly_the_retry_after_the_provider_asked_for(live, sleeps, seconds):
    """Both values are under max_retry_after_s, so the wait is honoured rather than given up on."""
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": str(seconds)},
                                  json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query([{"role": "user", "content": "hi"}])
    assert sleeps == [float(seconds)]


def test_retry_after_accepts_an_http_date():
    from email.utils import parsedate_to_datetime

    stamp = "Wed, 21 Oct 2099 07:28:00 GMT"
    seconds = pv.retry_after_seconds({"Retry-After": stamp}, now=0.0)
    assert seconds == parsedate_to_datetime(stamp).timestamp()


def test_network_errors_are_retried(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query([{"role": "user", "content": "hi"}])
    assert len(calls) == 2


def test_five_attempts_then_give_up(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "down"}})

    model = anthropic_model(handler, sleeps)
    with pytest.raises(pv.RetryExhausted):
        model.query([{"role": "user", "content": "hi"}])
    assert len(calls) == 5
    assert len(sleeps) == 4


def test_no_retry_on_a_plain_400(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad tool schema"}})

    model = anthropic_model(handler, sleeps)
    with pytest.raises(pv.ProviderError):
        model.query([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_never_retry_on_context_overflow(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "prompt is too long: 300000 tokens > 200000 maximum"}})

    model = anthropic_model(handler, sleeps)
    with pytest.raises(pv.ContextOverflowError):
        model.query([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert sleeps == []


def test_backoff_grows_and_stays_under_the_cap():
    policy = pv.RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=8.0, jitter=0.25)
    rng = random.Random(0)
    delays = [pv.backoff_delay(attempt, policy, rng) for attempt in range(1, 6)]
    assert delays[0] < delays[1] < delays[2]
    assert all(d <= policy.max_delay_s * (1 + policy.jitter) for d in delays)
    assert all(d > 0 for d in delays)


# --- OpenAI and OpenAI-compatible adapters ---


def openai_model(handler, sleeps, cls=None, **kwargs):
    cls = cls or pv.OpenAIModel
    return cls(
        model_id=kwargs.pop("model_id", "openai/gpt-4o-mini"),
        api_key="k",
        client=transport_of(handler),
        sleep=sleeps.append,
        rng=random.Random(0),
        **kwargs,
    )


def test_the_openai_adapter_sends_bearer_and_function_tools_and_subtracts_cached_tokens_from_input(live, sleeps):
    seen = {}
    payload = {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "on it",
                    "tool_calls": [
                        {"id": "call 1", "function": {"name": "get_order", "arguments": '{"id": "7"}'}}
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 6},
        },
    }

    def handler(request):
        seen["body"] = pv.json_body(request)
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, json=payload)

    model = openai_model(handler, sleeps)
    reply = model.query(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        tools=[{"name": "get_order", "description": "d", "input_schema": {"type": "object"}}],
        config=pv.ModelConfig(max_tokens=32, temperature=0.2),
    )
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["authorization"] == "Bearer k"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["tools"][0]["function"]["name"] == "get_order"
    assert reply.content == "on it"
    assert reply.tool_calls[0].id == pv.clean_tool_call_id("call 1")
    assert reply.tool_calls[0].arguments == {"id": "7"}
    # Usage.input is uncached input everywhere in the Harness; OpenAI's prompt_tokens counts
    # the cached ones too, so the adapter subtracts them here and budget.py bills each rate once.
    assert reply.usage.input == 6 and reply.usage.output == 4 and reply.usage.cache_read == 6


def test_openai_compatible_substitutes_the_base_url(live, sleeps):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})

    model = pv.OpenAICompatibleModel(
        model_id="local/llama",
        wire_id="llama-3.1-8b-instruct",
        base_url="${OLLAMA_HOST}/v1",
        env={"OLLAMA_HOST": "http://127.0.0.1:11434"},
        client=transport_of(handler),
        sleep=sleeps.append,
        rng=random.Random(0),
    )
    model.query([{"role": "user", "content": "hi"}])
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"


def test_openai_compatible_needs_no_key(live, sleeps):
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})

    model = pv.OpenAICompatibleModel(
        model_id="local/llama",
        base_url="http://127.0.0.1:11434/v1",
        client=transport_of(handler),
        sleep=sleeps.append,
        env={},
    )
    assert model.query([{"role": "user", "content": "hi"}]).content == "hi"


def test_a_missing_key_is_reported_before_the_request(live, sleeps):
    model = pv.AnthropicModel(
        model_id="anthropic/claude-opus-5",
        client=transport_of(ok_anthropic()),
        sleep=sleeps.append,
        env={},
    )
    with pytest.raises(pv.ProviderError) as excinfo:
        model.query([{"role": "user", "content": "hi"}])
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


# --- the live-call gate sits on the network path ---


def test_post_is_gated_too_not_only_query(sleeps):
    """A caller that builds a body and posts it must not reach the transport either."""
    hits = []

    def handler(request):
        hits.append(1)
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    with pytest.raises(RuntimeError) as excinfo:
        model.post({"model": "x"})
    assert "ALLOW_MODEL_REQUESTS" in str(excinfo.value)
    assert hits == []


def test_building_a_real_http_client_is_gated():
    """Nothing opens a socket while live calls are off, not even lazily."""
    model = pv.AnthropicModel(model_id="anthropic/claude-opus-5", api_key="k", env={})
    with pytest.raises(RuntimeError):
        model.client()


def test_live_calls_turn_on_only_from_the_environment():
    assert pv.enable_live_calls_from_env({}) is False
    assert pv.ALLOW_MODEL_REQUESTS is False
    assert pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "0"}) is False
    assert pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"}) is True
    assert pv.ALLOW_MODEL_REQUESTS is True
    require_live_calls_enabled_ok = pv.require_live_calls_enabled()
    assert require_live_calls_enabled_ok is None
    assert pv.enable_live_calls_from_env({}) is False


def registry_snapshot(tmp_path, monkeypatch, catalog):
    """A models.dev snapshot on disk, and the provider module pointed at it."""
    path = tmp_path / "models.dev.json"
    path.write_text(json.dumps({"fetched_at": "2026-09-01T00:00:00+00:00", "catalog": catalog}),
                    encoding="utf-8")
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(path))
    return path


REGISTRY = {
    "opencode-go": {"id": "opencode-go", "name": "OpenCode Go", "npm": "@ai-sdk/openai-compatible",
                    "api": "https://opencode.ai/zen/go/v1", "env": ["OPENCODE_API_KEY"],
                    "models": {"kimi-k3": {"limit": {"context": 1048576, "output": 131072},
                                           "cost": {"input": 3, "output": 15}}}},
    "google": {"id": "google", "name": "Google", "npm": "@ai-sdk/google", "env": ["GEMINI_API_KEY"],
               "models": {"gemini-3-pro": {}}},
    "vertex": {"id": "vertex", "name": "Vertex", "npm": "@ai-sdk/google-vertex",
               "api": "https://aiplatform.googleapis.com/v1", "env": ["GOOGLE_VERTEX_PROJECT"],
               "models": {"gemini-3-pro": {}}},
}


def test_a_provider_the_registry_names_needs_no_adapter_and_no_base_url(tmp_path, monkeypatch):
    """The founder's ask: choose any model, the way OpenCode does, without code per provider."""
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    model = pv.model_for("opencode-go/kimi-k3", env={"OPENCODE_API_KEY": "sk-zen"})
    assert isinstance(model, pv.RegistryModel)
    assert model.base_url == "https://opencode.ai/zen/go/v1"
    assert model.wire_id == "kimi-k3"
    assert model.api_key == "sk-zen"
    assert model.headers()["authorization"] == "Bearer sk-zen"


def test_the_registry_model_asks_for_the_key_variable_the_registry_names(tmp_path, monkeypatch, live):
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    model = pv.model_for("opencode-go/kimi-k3", env={})
    with pytest.raises(pv.ProviderError) as raised:
        model.query([{"role": "user", "content": "hi"}])
    assert "OPENCODE_API_KEY" in str(raised.value)


def test_a_base_url_the_caller_passes_wins_over_the_registry(tmp_path, monkeypatch):
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    model = pv.model_for("opencode-go/kimi-k3", base_url="http://127.0.0.1:8080/v1", env={})
    assert isinstance(model, pv.OpenAICompatibleModel)
    assert model.base_url == "http://127.0.0.1:8080/v1"


def test_a_provider_the_registry_serves_in_another_request_shape_is_refused_by_name(tmp_path, monkeypatch):
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    with pytest.raises(ValueError) as raised:
        pv.model_for("vertex/gemini-3-pro", env={})
    assert "@ai-sdk/google-vertex" in str(raised.value)
    assert "base_url" in str(raised.value)


def test_a_provider_with_no_host_in_the_registry_is_refused_like_an_unknown_one(tmp_path, monkeypatch):
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    for model_id in ("google/gemini-3-pro", "nowhere/model-1"):
        with pytest.raises(ValueError) as raised:
            pv.model_for(model_id, env={})
        assert "no host" in str(raised.value)


def test_the_registry_is_read_from_disk_and_never_from_the_network_with_live_calls_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(tmp_path / "absent.json"))

    def refuse(*args, **kwargs):
        raise AssertionError("the registry reached the network with live calls off")

    monkeypatch.setattr(httpx, "Client", refuse)
    with pytest.raises(ValueError):
        pv.model_for("opencode-go/kimi-k3", env={})


def test_the_registry_model_sends_no_reasoning_field_a_gateway_may_not_know(tmp_path, monkeypatch):
    registry_snapshot(tmp_path, monkeypatch, REGISTRY)
    model = pv.model_for("opencode-go/kimi-k3", env={"OPENCODE_API_KEY": "sk-zen"})
    body = model.build_body([{"role": "user", "content": "hi"}], None,
                            pv.ModelConfig(reasoning_effort="high"))
    assert "reasoning_effort" not in body
    assert body["model"] == "kimi-k3"


def test_model_for_builds_one_adapter_per_provider():
    anthropic = pv.model_for("anthropic/claude-opus-5", api_key="k", env={})
    assert isinstance(anthropic, pv.AnthropicModel) and anthropic.wire_id == "claude-opus-5"
    assert isinstance(pv.model_for("openai/o4-mini", api_key="k", env={}), pv.OpenAIModel)
    local = pv.model_for("local/llama", base_url="http://127.0.0.1:11434/v1", env={})
    assert isinstance(local, pv.OpenAICompatibleModel)
    with pytest.raises(ValueError):
        pv.model_for("local/llama", env={})


# --- the three reasoning branches ---


HI = [{"role": "user", "content": "hi"}]


def test_anthropic_sends_thinking_and_effort(sleeps):
    body = anthropic_model(ok_anthropic(), sleeps).build_body(
        HI, None, pv.ModelConfig(thinking={"type": "adaptive"}, effort="low"))
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert "budget_tokens" not in json.dumps(body), "the current models reject budget_tokens"


def test_openai_sends_reasoning_effort(sleeps):
    model = openai_model(ok_anthropic(), sleeps, model_id="openai/o4-mini")
    body = model.build_body(HI, None, pv.ModelConfig(reasoning_effort="low"))
    assert body["reasoning_effort"] == "low"


def test_a_local_endpoint_is_sent_no_reasoning_fields(sleeps):
    """Branch three: a server that does not know the field rejects the whole request."""
    model = pv.OpenAICompatibleModel(
        model_id="local/llama",
        base_url="http://127.0.0.1:11434/v1",
        client=transport_of(ok_anthropic()),
        env={},
        sleep=sleeps.append,
    )
    body = model.build_body(HI, None, pv.ModelConfig(reasoning_effort="high"))
    assert "reasoning_effort" not in body
    assert "thinking" not in body


def test_a_config_without_reasoning_sends_none_of_it(sleeps):
    body = anthropic_model(ok_anthropic(), sleeps).build_body(HI, None, pv.ModelConfig())
    assert "thinking" not in body and "output_config" not in body


# --- what the model is actually shown ---


def test_a_structured_tool_result_goes_as_json_not_a_python_repr():
    """D65: the Candidate sees what production gave it, and production never sent repr()."""
    result = {"order_id": "#W1", "status": "pending", "expedited": True, "note": None}
    _, out = pv._to_anthropic(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "name": "get", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "c1", "content": result},
        ]
    )
    text = out[-1]["content"][0]["content"]
    assert json.loads(text) == result
    assert "'" not in text and "True" not in text and "None" not in text


def test_two_tool_results_from_one_turn_group_into_one_user_message():
    """Two tool calls in one assistant turn must produce user, assistant, user, not two separate
    user messages back to back, which the Anthropic Messages API rejects."""
    _, out = pv._to_anthropic(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "get_order", "arguments": {}},
                    {"id": "c2", "name": "get_user", "arguments": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "order result"},
            {"role": "tool", "tool_call_id": "c2", "content": "user result"},
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    tool_results = out[-1]["content"]
    assert [b["tool_use_id"] for b in tool_results] == ["c1", "c2"]
    assert [b["content"] for b in tool_results] == ["order result", "user result"]


def test_an_openai_message_carrying_a_dict_result_goes_as_json():
    message = pv._openai_message({"role": "tool", "tool_call_id": "c1", "content": {"a": 1, "b": None}})
    assert json.loads(message["content"]) == {"a": 1, "b": None}


def test_an_assistant_message_without_tool_calls_goes_without_the_key():
    """The loop writes tool_calls on every assistant message; the API rejects an empty list."""
    plain = pv._openai_message({"role": "assistant", "content": "Which order?", "tool_calls": []})
    assert "tool_calls" not in plain and plain["content"] == "Which order?"
    calling = pv._openai_message({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"id": "1"}}]})
    assert calling["tool_calls"][0]["function"]["name"] == "get_order"


def test_a_recorded_assistant_message_of_blocks_replays(make_recorded_model):
    """Anthropic and Claude Code JSONL record assistant content as a list of blocks."""
    model = make_recorded_model(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me look"},
                    {"type": "text", "text": "one moment"},
                    {"type": "tool_use", "id": "c1", "name": "get_order", "input": {"id": "#W1"}},
                ],
            }
        ]
    )
    reply = model.query([])
    assert "one moment" in (reply.content or "")
    assert reply.tool_calls[0].name == "get_order"
    assert reply.tool_calls[0].arguments == {"id": "#W1"}


def test_an_empty_user_turn_keeps_the_request_ending_on_the_user(sleeps):
    """Dropping it would end the request on an assistant message, which the API reads as a
    prefill and rejects on the current models."""
    body = anthropic_model(ok_anthropic(), sleeps).build_body(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "what is your order id?"},
            {"role": "user", "content": "   "},
        ],
        None,
        pv.ModelConfig(),
    )
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][-1]["content"][0]["text"] == pv.EMPTY_USER_PLACEHOLDER


def test_an_empty_assistant_turn_is_still_dropped():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "  "}]
    assert [m["role"] for m in pv.normalize_messages(messages)] == ["user"]


# --- retry rules ---


def test_a_two_hundred_that_is_not_json_is_a_provider_error_and_is_retried(live, sleeps):
    """A proxy's HTML page must not escape query() as a raw JSONDecodeError."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text="<html>gateway timeout</html>")

    model = anthropic_model(handler, sleeps)
    with pytest.raises(pv.RetryExhausted):
        model.query([{"role": "user", "content": "hi"}])
    assert len(calls) == 5


def test_a_non_json_two_hundred_that_recovers_is_not_an_error(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, text="<html>gateway</html>")
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    model = anthropic_model(handler, sleeps)
    assert model.query([{"role": "user", "content": "hi"}]).content == "ok"
    assert len(calls) == 2


def test_a_retry_after_longer_than_the_build_will_wait_gives_up(live, sleeps):
    """One 429 asking for a day must not block the build for a day."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "86400"}, json={"error": {"message": "slow"}})

    model = anthropic_model(handler, sleeps)
    with pytest.raises(pv.RetryExhausted) as excinfo:
        model.query([{"role": "user", "content": "hi"}])
    assert "86400" in str(excinfo.value)
    assert sleeps == []
    assert len(calls) == 1


def test_load_dotenv_adds_keys_without_overriding(tmp_path):
    from kullback.ai.provider import load_dotenv

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# keys\nOPENAI_API_KEY='sk-test'\nexport HARNESS_ALLOW_MODEL_REQUESTS=1\n\nALREADY=new\nbroken line\n",
        encoding="utf-8",
    )
    env = {"ALREADY": "old"}
    added = load_dotenv(dotenv, env)
    assert added == {"OPENAI_API_KEY": "sk-test", "HARNESS_ALLOW_MODEL_REQUESTS": "1"}
    assert env == {"ALREADY": "old", "OPENAI_API_KEY": "sk-test", "HARNESS_ALLOW_MODEL_REQUESTS": "1"}
    assert load_dotenv(tmp_path / "missing.env", env) == {}


# --- prompt_cache_key (docs/prompt-caching.md item 4) ---


@pytest.mark.parametrize("config_kwargs,expected", [
    ({"prompt_cache_key": "kullback-abc-compile_tools"}, "kullback-abc-compile_tools"),
    ({}, None),
])
def test_the_openai_body_carries_the_prompt_cache_key_only_when_one_was_set(config_kwargs, expected):
    body = _body("openai/gpt-4o-mini", **config_kwargs)
    assert body.get("prompt_cache_key") == expected


def test_anthropic_body_ignores_the_prompt_cache_key(sleeps):
    body = anthropic_model(ok_anthropic(), sleeps).build_body(
        HI, None, pv.ModelConfig(prompt_cache_key="kullback-abc-compile_tools"))
    assert "prompt_cache_key" not in body


# --- MemoModel (docs/prompt-caching.md item 3) ---


def test_a_second_identical_query_is_a_hit_and_does_not_reach_the_inner_model(tmp_path):
    inner = pv.TestModel(["first", "second"])
    memo = pv.MemoModel(inner, tmp_path)
    messages = [{"role": "user", "content": "hi"}]

    first = memo.query(messages)
    assert first.content == "first"
    assert memo.calls == 1 and memo.hits == 0 and memo.last_hit is False

    second = memo.query(messages)
    assert second.content == "first"  # served from the memo, not TestModel's second scripted reply
    assert memo.calls == 2 and memo.hits == 1 and memo.last_hit is True
    assert len(inner.calls) == 1, "a hit must never reach the inner model"


def test_a_memo_hit_carries_zeroed_usage(tmp_path):
    inner = pv.TestModel([pv.ModelReply(content="x", usage=pv.Usage(input=10, output=5))])
    memo = pv.MemoModel(inner, tmp_path)
    messages = [{"role": "user", "content": "hi"}]

    first = memo.query(messages)
    assert first.usage.input == 10 and first.usage.output == 5

    second = memo.query(messages)
    assert second.usage.input == 0 and second.usage.output == 0


@pytest.mark.parametrize("first_kwargs,second_kwargs", [
    ({"config": pv.ModelConfig(temperature=0.1)}, {"config": pv.ModelConfig(temperature=0.9)}),
    ({"tools": [{"name": "get_order"}]}, {"tools": [{"name": "get_refund"}]}),
], ids=["config", "tool_list"])
def test_a_different_config_or_tool_list_is_a_miss(tmp_path, first_kwargs, second_kwargs):
    inner = pv.TestModel(["a", "b"])
    memo = pv.MemoModel(inner, tmp_path)
    messages = [{"role": "user", "content": "hi"}]

    memo.query(messages, **first_kwargs)
    memo.query(messages, **second_kwargs)
    assert memo.hits == 0 and len(inner.calls) == 2


def test_the_cache_directory_is_content_addressed(tmp_path):
    inner = pv.TestModel(["a"])
    memo = pv.MemoModel(inner, tmp_path)
    memo.query([{"role": "user", "content": "hi"}])
    files = list((tmp_path / pv.MemoModel.CACHE_DIR).glob("*.json"))
    assert len(files) == 1
    key = memo._key([{"role": "user", "content": "hi"}], None, None)
    assert files[0].name == f"{key}.json"


def test_the_memo_survives_a_second_memomodel_over_the_same_workdir(tmp_path):
    inner_one = pv.TestModel(["a"])
    pv.MemoModel(inner_one, tmp_path).query([{"role": "user", "content": "hi"}])

    inner_two = pv.TestModel(["should not be reached"])
    memo_two = pv.MemoModel(inner_two, tmp_path)
    reply = memo_two.query([{"role": "user", "content": "hi"}])
    assert reply.content == "a"
    assert inner_two.calls == []


# --- the request shape the gpt-5 family insists on (found on the first live build) ---

def _body(model_id: str, tools=None, **config):
    """The body one call would send, without sending it."""
    model = pv.OpenAIModel(model_id=model_id, api_key="k", env={})
    return model.build_body([{"role": "user", "content": "hi"}], tools, pv.ModelConfig(**config))


def test_the_gpt5_family_is_sent_max_completion_tokens_not_max_tokens():
    body = _body("openai/gpt-5.6-luna", max_tokens=2000)
    assert body["max_completion_tokens"] == 2000
    assert "max_tokens" not in body


def test_an_older_openai_model_keeps_the_field_it_has_always_taken():
    body = _body("openai/gpt-4.1-mini", max_tokens=2000)
    assert body["max_tokens"] == 2000
    assert "max_completion_tokens" not in body


def test_the_o_series_is_read_as_the_same_family():
    for model_id in ("openai/o1", "openai/o3-mini", "openai/o4-mini"):
        assert "max_completion_tokens" in _body(model_id, max_tokens=8)


def test_the_gpt5_family_is_not_sent_a_temperature_it_will_refuse():
    assert "temperature" not in _body("openai/gpt-5.6-luna", temperature=0)
    assert _body("openai/gpt-4.1-mini", temperature=0)["temperature"] == 0


def test_a_tool_call_to_the_gpt5_family_turns_reasoning_off_by_name():
    tools = [{"name": "get_order", "input_schema": {"type": "object"}}]
    assert _body("openai/gpt-5.6-luna", tools=tools)["reasoning_effort"] == "none"
    # Not sending the field is not the same as sending 'none': the endpoint applies its own
    # default, and that default is what returns HTTP 400 alongside tools.
    assert "reasoning_effort" not in _body("openai/gpt-5.6-luna")


def test_an_effort_the_caller_asked_for_survives_a_tool_call():
    tools = [{"name": "get_order", "input_schema": {"type": "object"}}]
    assert _body("openai/gpt-5.6-luna", tools=tools, reasoning_effort="high")["reasoning_effort"] == "high"


def test_an_older_model_is_left_alone_when_it_is_given_tools():
    tools = [{"name": "get_order", "input_schema": {"type": "object"}}]
    assert "reasoning_effort" not in _body("openai/gpt-4.1-mini", tools=tools)


def test_opencode_hosts_get_session_and_identity_headers(live, sleeps):
    """Go asks clients to identify (no broad user agents) and send x-opencode-session; without
    both, gateway traffic looks abusive and keys get blocked. Stable per process for caching."""
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})

    model = pv.OpenAICompatibleModel(
        model_id="opencode-go/kimi-k3", base_url="https://opencode.ai/zen/go/v1",
        client=transport_of(handler), sleep=sleeps.append, env={},
    )
    assert model.query([{"role": "user", "content": "hi"}]).content == "hi"
    assert seen["headers"]["user-agent"] == "kullback"
    assert re.fullmatch(r"[0-9a-f]{32}", seen["headers"]["x-opencode-session"])
    assert model.headers()["x-opencode-session"] == seen["headers"]["x-opencode-session"]


def test_other_hosts_see_no_opencode_headers(live, sleeps):
    def handler(request):
        assert "x-opencode-session" not in request.headers
        assert request.headers.get("user-agent", "").startswith("python-httpx")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})

    model = pv.OpenAICompatibleModel(
        model_id="local/llama", base_url="http://127.0.0.1:11434/v1",
        client=transport_of(handler), sleep=sleeps.append, env={},
    )
    assert model.query([{"role": "user", "content": "hi"}]).content == "hi"
