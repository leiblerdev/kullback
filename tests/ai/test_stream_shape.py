"""The streamed OpenAI path learns a shape field a 400 names and streams once more."""

import asyncio
import json

import httpx
import pytest

import kullback.ai.provider as pv
from kullback.ai.messages import UserMessage
from kullback.ai.openai_compatible import OpenAICompatibleProvider

REFUSAL = {"error": {"message": "Function tools with reasoning_effort are not supported; set reasoning_effort to 'none'."}}
PLAIN = [
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "ok"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]}',
    "[DONE]",
]


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


def test_a_streamed_shape_400_is_retried_once_and_ends_in_a_normal_done(live):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("reasoning_effort") != "none":
            return httpx.Response(400, json=REFUSAL)
        return httpx.Response(200, content="".join(f"data: {chunk}\n\n" for chunk in PLAIN).encode())

    handle_model = pv.OpenAIModel("openai/next-x", api_key="k", env={})
    provider = OpenAICompatibleProvider(handle_model, client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    tools = [{"name": "echo", "parameters": {"type": "object"}}]

    async def run():
        return [event async for event in provider.stream_response(messages=[UserMessage(content="hi")], tools=tools)]

    events = asyncio.run(run())
    assert len(sent) == 2
    assert "reasoning_effort" not in sent[0]
    assert sent[1]["reasoning_effort"] == "none"
    assert events[-1].type == "done"
    assert all(event.type != "error" for event in events)
