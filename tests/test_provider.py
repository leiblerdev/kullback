"""Tests for the real provider adapters: normalizations, cache points, retry, parsing. No network."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from harness.shared import provider as pv


def transport_of(handler):
    """An httpx client whose every request is answered by handler; nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def live(monkeypatch):
    """Adapters refuse to run while ALLOW_MODEL_REQUESTS is False; the mock transport keeps it offline."""
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


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


def test_wire_id_can_differ_from_the_model_id(live, sleeps):
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps, model_id="anthropic/opus", wire_id="claude-opus-5")
    model.query([{"role": "user", "content": "hi"}])
    assert model.name == "anthropic/opus"
    assert seen["model"] == "claude-opus-5"


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
    # An id that needed no cleaning is left exactly as it was.
    assert pv.clean_tool_call_id("call_abc-123") == "call_abc-123"
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


def test_anthropic_request_shape(live, sleeps):
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


def test_retry_on_429_honors_retry_after(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query([{"role": "user", "content": "hi"}])
    assert sleeps == [7.0]


def test_retry_after_accepts_an_http_date():
    seconds = pv.retry_after_seconds({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}, now=0.0)
    assert seconds is not None and seconds > 0


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


def test_openai_request_and_reply(live, sleeps):
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


def test_the_scaffold_interface_is_unchanged():
    assert issubclass(pv.AnthropicModel, pv.Model)
    assert issubclass(pv.OpenAIModel, pv.Model)
    assert issubclass(pv.OpenAICompatibleModel, pv.Model)
    assert pv.ALLOW_MODEL_REQUESTS is False
    assert isinstance(pv.TestModel(["ok"]).query([]), pv.ModelReply)


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


def test_the_flag_is_off_in_the_module_source_not_only_in_this_process():
    """conftest forces the flag False, so read the module's own assignment to see the default."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(pv.__file__).read_text(encoding="utf-8"))
    defaults = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "ALLOW_MODEL_REQUESTS" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert defaults == [False], defaults


def test_live_calls_turn_on_only_from_the_environment(monkeypatch):
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", False)
    assert pv.enable_live_calls_from_env({}) is False
    assert pv.ALLOW_MODEL_REQUESTS is False
    assert pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "0"}) is False
    assert pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"}) is True
    assert pv.ALLOW_MODEL_REQUESTS is True
    require_live_calls_enabled_ok = pv.require_live_calls_enabled()
    assert require_live_calls_enabled_ok is None
    assert pv.enable_live_calls_from_env({}) is False


def test_model_for_builds_one_adapter_per_provider():
    anthropic = pv.model_for("anthropic/claude-opus-5", api_key="k", env={})
    assert isinstance(anthropic, pv.AnthropicModel) and anthropic.wire_id == "claude-opus-5"
    assert isinstance(pv.model_for("openai/o4-mini", api_key="k", env={}), pv.OpenAIModel)
    local = pv.model_for("local/llama", base_url="http://127.0.0.1:11434/v1", env={})
    assert isinstance(local, pv.OpenAICompatibleModel)
    with pytest.raises(ValueError):
        pv.model_for("local/llama", env={})


# --- the three reasoning branches ---


def test_anthropic_sends_thinking_and_effort(live, sleeps):
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query(
        [{"role": "user", "content": "hi"}],
        config=pv.ModelConfig(thinking={"type": "adaptive"}, effort="low"),
    )
    assert seen["thinking"] == {"type": "adaptive"}
    assert seen["output_config"] == {"effort": "low"}
    assert "budget_tokens" not in json.dumps(seen), "the current models reject budget_tokens"


def test_openai_sends_reasoning_effort(live, sleeps):
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    model = openai_model(handler, sleeps, model_id="openai/o4-mini")
    model.query([{"role": "user", "content": "hi"}], config=pv.ModelConfig(reasoning_effort="low"))
    assert seen["reasoning_effort"] == "low"


def test_a_local_endpoint_is_sent_no_reasoning_fields(live, sleeps):
    """Branch three: a server that does not know the field rejects the whole request."""
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    model = pv.OpenAICompatibleModel(
        model_id="local/llama",
        base_url="http://127.0.0.1:11434/v1",
        client=transport_of(handler),
        env={},
        sleep=sleeps.append,
    )
    model.query([{"role": "user", "content": "hi"}], config=pv.ModelConfig(reasoning_effort="high"))
    assert "reasoning_effort" not in seen
    assert "thinking" not in seen


def test_a_config_without_reasoning_sends_none_of_it(live, sleeps):
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"content": [], "usage": {}})

    anthropic_model(handler, sleeps).query([{"role": "user", "content": "hi"}])
    assert "thinking" not in seen and "output_config" not in seen


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


def test_an_openai_message_carrying_a_dict_result_goes_as_json():
    message = pv._openai_message({"role": "tool", "tool_call_id": "c1", "content": {"a": 1, "b": None}})
    assert json.loads(message["content"]) == {"a": 1, "b": None}


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


def test_an_empty_user_turn_keeps_the_request_ending_on_the_user(live, sleeps):
    """Dropping it would end the request on an assistant message, which the API reads as a
    prefill and rejects on the current models."""
    seen = {}

    def handler(request):
        seen.update(pv.json_body(request))
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "what is your order id?"},
            {"role": "user", "content": "   "},
        ]
    )
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]
    assert seen["messages"][-1]["content"][0]["text"] == pv.EMPTY_USER_PLACEHOLDER


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


def test_a_retry_after_inside_the_cap_is_still_honoured(live, sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "90"}, json={"error": {"message": "slow"}})
        return httpx.Response(200, json={"content": [], "usage": {}})

    model = anthropic_model(handler, sleeps)
    model.query([{"role": "user", "content": "hi"}])
    assert sleeps == [90.0]


def test_load_dotenv_adds_keys_without_overriding(tmp_path):
    from harness.shared.provider import load_dotenv

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
