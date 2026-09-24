"""Claude through Amazon Bedrock's Messages endpoint: the Anthropic body, AWS authentication.

Every request here is answered by an httpx.MockTransport; nothing leaves the machine and no key
is real. A signature is checked by signing the bytes the transport received again, at the time
the request says it was signed, so a test fails if the handle signed one body and sent another.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random

import httpx
import pytest

from kullback.ai import provider as pv
from kullback.ai import sigv4
from kullback.ai.anthropic import AnthropicProvider
from kullback.ai.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, to_wire
from kullback.ai.model_limits import request_rules_for
from kullback.ai.usage import Usage
from kullback.runner import budget

MODEL = "bedrock/global.anthropic.claude-opus-5-5"
KEYS = {"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "not-a-real-secret"}
TOOL = {"name": "look_up", "description": "d", "input_schema": {"type": "object"}}
SIGNED_THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


@pytest.fixture
def live():
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})


def ok_reply(content=None, usage=None):
    return httpx.Response(200, json={
        "content": content or [{"type": "text", "text": "done"}],
        "model": "claude-opus-5-5", "stop_reason": "end_turn",
        "usage": usage or {"input_tokens": 10, "output_tokens": 2},
    })


def bedrock(handler, env=None, model_id=MODEL, **kwargs):
    return pv.BedrockAnthropicModel(model_id, env=KEYS if env is None else env,
                                    client=httpx.Client(transport=httpx.MockTransport(handler)),
                                    sleep=lambda _: None, rng=random.Random(0), **kwargs)


def capture(seen):
    def handler(request):
        seen.append(request)
        return ok_reply()
    return handler


def signature_holds(request, region, secret=KEYS["AWS_SECRET_ACCESS_KEY"]):
    """Sign the received bytes again at the request's own time and compare the whole header."""
    when = dt.datetime.strptime(request.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    given = {name: request.headers[name] for name in ("anthropic-version", "content-type")}
    again = sigv4.sign("POST", str(request.url), given, request.content, access_key="AKIDEXAMPLE",
                       secret_key=secret, session_token=request.headers.get("x-amz-security-token"),
                       region=region, service="bedrock", now=when)
    return again["authorization"] == request.headers["authorization"]


def test_the_endpoint_is_the_messages_path_on_the_region_from_the_environment_else_us_east_2(live):
    seen = []
    bedrock(capture(seen)).query([{"role": "user", "content": "hi"}])
    bedrock(capture(seen), env={**KEYS, "AWS_REGION": "eu-west-1"}).query([{"role": "user", "content": "hi"}])
    bedrock(capture(seen), env={**KEYS, "AWS_DEFAULT_REGION": "ap-northeast-1"}).query(
        [{"role": "user", "content": "hi"}])
    assert [str(r.url) for r in seen] == [
        "https://bedrock-runtime.us-east-2.amazonaws.com/anthropic/v1/messages",
        "https://bedrock-runtime.eu-west-1.amazonaws.com/anthropic/v1/messages",
        "https://bedrock-runtime.ap-northeast-1.amazonaws.com/anthropic/v1/messages",
    ]


def test_an_explicit_base_url_still_wins_over_the_region():
    model = pv.model_for(MODEL, base_url="https://a-proxy.invalid/anthropic", env=KEYS)
    assert model.base_url + model.path == "https://a-proxy.invalid/anthropic/v1/messages"


def test_access_keys_sign_the_exact_body_sent_with_no_api_key_header(live):
    seen = []
    bedrock(capture(seen)).query([{"role": "user", "content": "hi"}])
    request = seen[0]
    assert "x-api-key" not in request.headers
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["host"] == "bedrock-runtime.us-east-2.amazonaws.com"
    assert request.headers["authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/" + request.headers["x-amz-date"][:8]
        + "/us-east-2/bedrock/aws4_request, SignedHeaders=anthropic-version;content-type;host;x-amz-date,")
    assert "x-amz-security-token" not in request.headers
    assert signature_holds(request, "us-east-2")


def test_temporary_keys_send_their_session_token_inside_the_signature(live):
    seen = []
    bedrock(capture(seen), env={**KEYS, "AWS_SESSION_TOKEN": "a-session-token"}).query(
        [{"role": "user", "content": "hi"}])
    request = seen[0]
    assert request.headers["x-amz-security-token"] == "a-session-token"
    assert ";x-amz-security-token," in request.headers["authorization"]
    assert signature_holds(request, "us-east-2")


def test_a_bedrock_api_key_goes_as_a_bearer_token_and_wins_over_access_keys(live):
    seen = []
    bedrock(capture(seen), env={**KEYS, "AWS_BEARER_TOKEN_BEDROCK": "a-bearer"}).query(
        [{"role": "user", "content": "hi"}])
    request = seen[0]
    assert request.headers["authorization"] == "Bearer a-bearer"
    assert "x-amz-date" not in request.headers and "x-api-key" not in request.headers


def test_the_wire_id_goes_unchanged_and_cache_points_sit_on_blocks_never_top_level(live):
    seen = []
    bedrock(capture(seen)).query(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], tools=[TOOL])
    body = json.loads(seen[0].content)
    assert body["model"] == "global.anthropic.claude-opus-5-5"
    assert "cache_control" not in body
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_a_profile_id_or_an_arn_goes_as_given_and_a_bare_id_goes_on_the_global_profile_in_every_region():
    arn = "arn:aws:bedrock:us-east-2:111122223333:application-inference-profile/abc123"
    for env in (KEYS, {**KEYS, "AWS_REGION": "eu-west-1"}, {**KEYS, "AWS_REGION": "ap-northeast-1"}):
        def wire(model_id, env=env):
            return pv.BedrockAnthropicModel(model_id, env=env).wire_id
        assert wire("bedrock/us.anthropic.claude-opus-5-5") == "us.anthropic.claude-opus-5-5"
        assert wire("bedrock/global.anthropic.claude-opus-5-5") == "global.anthropic.claude-opus-5-5"
        assert wire("bedrock/anthropic.claude-opus-5-5") == "global.anthropic.claude-opus-5-5"
        assert wire("bedrock/anthropic.claude-haiku-4-5-20251001-v1:0") == (
            "global.anthropic.claude-haiku-4-5-20251001-v1:0")
        assert wire("bedrock/" + arn) == arn


def test_no_keys_at_all_is_refused_by_naming_the_aws_variables(live):
    assert pv.DEFAULT_MODEL.startswith("bedrock/")
    with pytest.raises(pv.ProviderError) as refused:
        bedrock(capture([]), env={}, model_id=pv.DEFAULT_MODEL).query([{"role": "user", "content": "hi"}])
    text = str(refused.value)
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                 "AWS_REGION"):
        assert name in text


def test_the_bedrock_prefix_resolves_to_the_bedrock_handle_and_the_messages_stream():
    assert isinstance(pv.model_for(MODEL, env=KEYS), pv.BedrockAnthropicModel)
    streamer = pv.provider_for(MODEL, env=KEYS)
    assert isinstance(streamer, AnthropicProvider) and isinstance(streamer.handle, pv.BedrockAnthropicModel)
    assert pv.key_var_for_provider("bedrock") == "AWS_BEARER_TOKEN_BEDROCK"
    assert pv.BedrockAnthropicModel.credential_vars() == (
        ("AWS_BEARER_TOKEN_BEDROCK",), ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"))


# --- the Opus 5.5 request rules, which hold on every host serving the model ---


def body_for(config, model_id=MODEL, messages=None, tools=(TOOL,)):
    handle = pv.BedrockAnthropicModel(model_id, env=KEYS)
    return handle.build_body(messages or [{"role": "user", "content": "hi"}], list(tools) or None, config)


def test_thinking_is_never_sent_disabled_or_budgeted_to_a_model_where_it_is_always_on():
    assert "thinking" not in body_for(pv.ModelConfig(thinking={"type": "disabled"}))
    assert "thinking" not in body_for(pv.ModelConfig(thinking={"type": "enabled", "budget_tokens": 2048}))
    assert body_for(pv.ModelConfig(thinking={"type": "adaptive"}))["thinking"] == {"type": "adaptive"}
    assert body_for(pv.ModelConfig(effort="low"))["output_config"] == {"effort": "low"}


def test_a_required_tool_call_goes_as_auto_with_the_demand_in_the_prompt_after_the_cache_point():
    body = body_for(pv.ModelConfig(tool_choice="required"),
                    messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    assert body["tool_choice"] == {"type": "auto"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][-1] == {"type": "text", "text": pv.FORCED_TOOL_LINE}
    older = body_for(pv.ModelConfig(tool_choice="required"), model_id="bedrock/anthropic.claude-haiku-4-5")
    assert older["tool_choice"] == {"type": "any"}


def test_sampling_parameters_are_left_out_for_a_model_that_refuses_them():
    assert "temperature" not in body_for(pv.ModelConfig(temperature=0.0))
    assert body_for(pv.ModelConfig(temperature=0.0), model_id="bedrock/anthropic.claude-haiku-4-5")["temperature"] == 0.0


def test_the_rules_are_read_by_model_whatever_prefix_the_host_puts_on_the_id():
    rules = request_rules_for("claude-opus-5-5")
    assert request_rules_for("anthropic.claude-opus-5-5") == rules
    assert request_rules_for("us.anthropic.claude-opus-5-5") == rules
    assert request_rules_for("global.anthropic.claude-opus-5-5") == rules
    assert request_rules_for("bedrock/global.anthropic.claude-opus-5-5") == rules
    assert rules.thinking_always_on and not rules.forced_tool_choice and not rules.sampling
    assert request_rules_for("anthropic.claude-haiku-4-5") == request_rules_for(None)


def test_signed_thinking_blocks_come_back_on_the_next_request_unchanged_and_first(live):
    """Preserved thinking: the reply's blocks, empty text and all, lead the assistant turn they
    belong to, and the cache point skips them because a thinking block cannot carry one."""
    replies = [ok_reply(content=[SIGNED_THINKING, {"type": "tool_use", "id": "t1", "name": "look_up", "input": {}}]),
               ok_reply()]
    seen = []

    def handler(request):
        seen.append(request)
        return replies[len(seen) - 1]

    model = bedrock(handler)
    first = model.query([{"role": "user", "content": "hi"}], tools=[TOOL])
    assert first.thinking_blocks == [SIGNED_THINKING]
    turn = AssistantMessage(content=None, tool_calls=[ToolCall(id="t1", name="look_up", arguments={})],
                            thinking_blocks=first.thinking_blocks, stop_reason="tool_use")
    history = [UserMessage(content="hi"), turn,
               ToolResultMessage(tool_call_id="t1", tool_name="look_up", content="found")]
    model.query(to_wire(history), tools=[TOOL])
    assistant = json.loads(seen[1].content)["messages"][1]
    assert assistant["content"][0] == SIGNED_THINKING
    assert assistant["content"][1]["type"] == "tool_use" and "cache_control" in assistant["content"][1]


def test_a_transcript_without_thinking_blocks_dumps_as_it_did_before():
    assert "thinking_blocks" not in AssistantMessage(content="x").model_dump()
    assert "thinking_blocks" not in pv.ModelReply(content="x").model_dump()
    assert "thinking_blocks" not in to_wire([AssistantMessage(content="x")])[0]


# --- streaming ---


def sse(chunks):
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode("utf-8")


def test_the_stream_is_signed_over_the_bytes_it_posts_and_keeps_the_signed_thinking(live):
    chunks = [
        '{"type": "message_start", "message": {"usage": {"input_tokens": 9, "cache_read_input_tokens": 700,'
        ' "cache_creation_input_tokens": 30}}}',
        '{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "",'
        ' "signature": ""}}',
        '{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}}',
        '{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1",'
        ' "name": "look_up"}}',
        '{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}}',
        '{"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}',
        '{"type": "message_stop"}',
    ]
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, content=sse(chunks))

    handle_model = pv.BedrockAnthropicModel(MODEL, env={**KEYS, "AWS_REGION": "eu-west-1"})
    streamer = AnthropicProvider(handle_model, client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))

    async def run():
        return [event async for event in streamer.stream_response(messages=[UserMessage(content="hi")],
                                                                  tools=[TOOL])]

    events = asyncio.run(run())
    request = seen[0]
    assert str(request.url) == "https://bedrock-runtime.eu-west-1.amazonaws.com/anthropic/v1/messages"
    assert json.loads(request.content)["stream"] is True
    assert signature_holds(request, "eu-west-1")
    done = events[-1].message
    assert done.thinking_blocks == [SIGNED_THINKING]
    assert [call.name for call in done.tool_calls] == ["look_up"]
    assert (done.usage.cache_read, done.usage.cache_write, done.usage.output) == (700, 30, 4)


# --- prices ---


def test_a_bedrock_call_is_priced_and_its_window_is_known():
    usage = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    assert budget.call_cost(usage, MODEL) == pytest.approx(4.0 + 20.0 + 0.2 + 5.0)
    assert budget.call_cost(usage, "anthropic/claude-opus-5-5") == pytest.approx(29.2)
    assert budget.window_for(MODEL) == 1_000_000
    assert budget.window_for("bedrock/anthropic.claude-haiku-4-5") == 200_000
    for model in ("bedrock/anthropic.claude-opus-5", "bedrock/anthropic.claude-sonnet-5",
                  "bedrock/anthropic.claude-haiku-4-5"):
        assert budget.is_priced(model)


def test_every_inference_profile_of_a_model_is_priced_and_windowed_by_the_one_row():
    usage = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    for profile in ("global.", "us.", "eu.", "apac.", ""):
        model = f"bedrock/{profile}anthropic.claude-opus-5-5"
        assert budget.call_cost(usage, model) == pytest.approx(29.2)
        assert budget.window_for(model) == 1_000_000
    assert budget.window_for("bedrock/us.anthropic.claude-haiku-4-5") == 200_000
    assert budget.price_source("bedrock/global.anthropic.claude-sonnet-5") is not None
