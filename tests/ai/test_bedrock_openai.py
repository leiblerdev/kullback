"""OpenAI's models through Amazon Bedrock: the Chat Completions shape on Bedrock's host and keys.

Every request is answered by an httpx.MockTransport; nothing leaves the machine and no key is real.
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
from kullback.ai.messages import UserMessage
from kullback.ai.openai_compatible import OpenAICompatibleProvider

MODEL = "bedrock/global.openai.gpt-6-sol"
KEYS = {"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "not-a-real-secret"}
TOOL = {"name": "look_up", "description": "d", "input_schema": {"type": "object"}}
USAGE = {"prompt_tokens": 900, "completion_tokens": 7,
         "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 250}}


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


def sse(chunks):
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode("utf-8")


def handle_for(seen, response):
    def handle(request):
        seen.append(request)
        return response
    return handle


def bedrock_openai(handler, env=None, model_id=MODEL):
    return pv.model_for(model_id, env=KEYS if env is None else env,
                        client=httpx.Client(transport=httpx.MockTransport(handler)),
                        sleep=lambda _: None, rng=random.Random(0))


def test_an_openai_id_on_bedrock_gets_the_chat_completions_adapter_and_an_anthropic_one_keeps_its_own():
    handle = pv.model_for(MODEL, env=KEYS)
    assert isinstance(handle, pv.BedrockOpenAIModel)
    assert isinstance(pv.provider_for(MODEL, env=KEYS), OpenAICompatibleProvider)
    assert isinstance(pv.model_for("bedrock/us.anthropic.claude-opus-5-5", env=KEYS), pv.BedrockAnthropicModel)
    assert handle.base_url + handle.path == "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions"
    eu = pv.model_for(MODEL, env={**KEYS, "AWS_REGION": "eu-west-1"})
    assert eu.base_url.startswith("https://bedrock-runtime.eu-west-1.amazonaws.com/")
    assert pv.BedrockOpenAIModel.credential_vars() == pv.BedrockAnthropicModel.credential_vars()


def test_a_call_sends_the_wire_id_unchanged_with_openai_tools_and_reads_the_cache_counts(live):
    seen = []
    reply = httpx.Response(200, json={
        "model": "gpt-6-sol",
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "look_up", "arguments": "{\"key\": \"a\"}"}}]}}],
        "usage": USAGE})
    got = bedrock_openai(handle_for(seen, reply)).query(
        [{"role": "user", "content": "hi"}], tools=[TOOL], config=pv.ModelConfig(max_tokens=64))
    body = json.loads(seen[0].content)
    assert body["model"] == "global.openai.gpt-6-sol"
    assert body["tools"][0]["type"] == "function" and body["tools"][0]["function"]["name"] == "look_up"
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
    assert [(call.name, call.arguments) for call in got.tool_calls] == [("look_up", {"key": "a"})]
    assert (got.usage.input, got.usage.cache_read, got.usage.cache_write, got.usage.output) == (50, 600, 250, 7)


def test_a_bearer_key_goes_as_bearer_and_access_keys_sign_the_exact_body(live):
    seen = []
    ok = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": USAGE})
    bedrock_openai(handle_for(seen, ok), env={**KEYS, "AWS_BEARER_TOKEN_BEDROCK": "a-bearer"}).query(
        [{"role": "user", "content": "hi"}])
    bedrock_openai(handle_for(seen, ok)).query([{"role": "user", "content": "hi"}])
    bearer, signed = seen
    assert bearer.headers["authorization"] == "Bearer a-bearer" and "x-amz-date" not in bearer.headers
    when = dt.datetime.strptime(signed.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    again = sigv4.sign("POST", str(signed.url), {"content-type": "application/json"}, signed.content,
                       access_key="AKIDEXAMPLE", secret_key=KEYS["AWS_SECRET_ACCESS_KEY"], session_token=None,
                       region="us-east-2", service="bedrock", now=when)
    assert again["authorization"] == signed.headers["authorization"]


def test_the_stream_is_real_sse_on_the_bedrock_path_and_ends_with_the_tool_call_and_its_usage(live):
    chunks = [
        '{"model": "gpt-6-sol", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",'
        ' "function": {"name": "look_up", "arguments": ""}}]}}]}',
        '{"model": "gpt-6-sol", "choices": [{"delta": {"tool_calls": [{"index": 0,'
        ' "function": {"arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}',
        json.dumps({"model": "gpt-6-sol", "choices": [], "usage": USAGE}),
        "[DONE]",
    ]
    seen = []
    handle = pv.model_for(MODEL, env={**KEYS, "AWS_BEARER_TOKEN_BEDROCK": "a-bearer"})
    streamer = OpenAICompatibleProvider(handle, client=httpx.AsyncClient(
        transport=httpx.MockTransport(handle_for(seen, httpx.Response(200, content=sse(chunks))))))

    async def run():
        return [event async for event in streamer.stream_response(messages=[UserMessage(content="hi")],
                                                                  tools=[TOOL])]

    done = asyncio.run(run())[-1].message
    request = seen[0]
    assert str(request.url) == "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(request.content)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert [call.name for call in done.tool_calls] == ["look_up"]
    assert (done.usage.cache_read, done.usage.cache_write, done.usage.output) == (600, 250, 7)


def test_gpt_6_sol_on_bedrock_is_priced_offline_and_its_cap_stays_in_the_priced_tier():
    from kullback.runner import budget

    assert budget.price_source(MODEL) == "table"
    assert (budget.price_for(MODEL)["input"], budget.price_for(MODEL)["output"]) == (2.0, 10.0)
    assert budget.context_cap_tokens(budget.window_for(MODEL)) <= 272_000
    assert budget.window_for(MODEL) == 680_000
