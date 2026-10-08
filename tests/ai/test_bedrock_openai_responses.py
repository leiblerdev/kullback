"""OpenAI's vendor on Amazon Bedrock: calls with tools or a reasoning effort take the Responses route.

Every request is answered by an httpx.MockTransport; nothing leaves the machine and no key is real.
No model is named beyond the vendor segment.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import httpx
import pytest

from kullback.ai import provider as pv
from kullback.ai import sigv4

MODEL = "bedrock/global.openai.some-model"
KEYS = {"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "not-a-real-secret"}
TOOL = {"name": "look_up", "description": "d", "input_schema": {"type": "object"}}
HOST = "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1"
RESPONSE = {
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
        {"type": "function_call", "call_id": "c1", "name": "look_up", "arguments": "{\"key\": \"a\"}"},
    ],
    "usage": {"input_tokens": 900, "output_tokens": 7, "input_tokens_details": {"cached_tokens": 600}},
}
CHAT = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


def bedrock(seen, env=None):
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=RESPONSE if request.url.path.endswith("/responses") else CHAT)

    return pv.model_for(MODEL, env=KEYS if env is None else env,
                        client=httpx.Client(transport=httpx.MockTransport(handle)),
                        sleep=lambda _: None, rng=random.Random(0))


def test_the_openai_vendor_on_bedrock_posts_to_responses_when_a_call_carries_tools_or_an_effort(live):
    seen = []
    model = bedrock(seen)
    assert pv.BEDROCK_VENDOR_ADAPTERS["openai"] is type(model)
    got = model.query([{"role": "user", "content": "hi"}], tools=[TOOL])
    model.query([{"role": "user", "content": "hi"}], config=pv.ModelConfig(reasoning_effort="medium"))
    model.query([{"role": "user", "content": "hi"}])
    assert [str(r.url) for r in seen] == [f"{HOST}/responses", f"{HOST}/responses", f"{HOST}/chat/completions"]
    with_tools, with_effort = json.loads(seen[0].content), json.loads(seen[1].content)
    assert with_tools["model"] == "global.openai.some-model"
    assert with_tools["input"] == [{"role": "user", "content": "hi"}]
    assert with_tools["tools"] == [{"type": "function", "name": "look_up", "description": "d",
                                    "parameters": {"type": "object"}}]
    assert with_effort["reasoning"] == {"summary": "auto", "effort": "medium"} and "tools" not in with_effort
    assert [(call.name, call.arguments) for call in got.tool_calls] == [("look_up", {"key": "a"})]
    assert (got.thinking, got.usage.cache_read, got.usage.output) == ("thought", 600, 7)


def test_the_responses_route_on_bedrock_signs_the_exact_body_or_sends_the_bearer_key(live):
    seen = []
    bedrock(seen).query([{"role": "user", "content": "hi"}], tools=[TOOL])
    bedrock(seen, env={**KEYS, "AWS_BEARER_TOKEN_BEDROCK": "a-bearer"}).query(
        [{"role": "user", "content": "hi"}], tools=[TOOL])
    signed, bearer = seen
    assert bearer.headers["authorization"] == "Bearer a-bearer" and "x-amz-date" not in bearer.headers
    when = dt.datetime.strptime(signed.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    again = sigv4.sign("POST", str(signed.url), {"content-type": "application/json"}, signed.content,
                       access_key="AKIDEXAMPLE", secret_key=KEYS["AWS_SECRET_ACCESS_KEY"], session_token=None,
                       region="us-east-2", service="bedrock", now=when)
    assert again["authorization"] == signed.headers["authorization"]


def test_the_responses_route_on_bedrock_refuses_a_call_with_no_aws_credentials(live):
    with pytest.raises(pv.ProviderError, match="no AWS credentials"):
        bedrock([], env={}).query([{"role": "user", "content": "hi"}], tools=[TOOL])
