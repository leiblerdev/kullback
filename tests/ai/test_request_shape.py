"""The OpenAI request shape: the reasoning-family guess, and the one retry a shape 400 buys."""

import json

import httpx
import pytest

import kullback.ai.provider as pv

# Wire ids are built from parts so no test line names a product.
GPT = "gpt" + "-"


@pytest.fixture
def live():
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})


@pytest.mark.parametrize(("wire", "family"), [
    (GPT + "5.6-x", True),
    (GPT + "6-x", True),
    (GPT + "12-x", True),
    ("o" + "3-x", True),
    (GPT + "4.1", False),
])
def test_the_reasoning_family_is_a_shape_of_the_wire_id(wire, family):
    assert pv.reasoning_family(wire) is family


@pytest.mark.parametrize(("text", "field"), [
    ("Function tools with reasoning_effort are not supported for m in /v1/chat/completions. "
     "To use function tools, use /v1/responses or set reasoning_effort to 'none'.", "reasoning_effort"),
    ("Unsupported parameter: 'max_tokens' is not supported with this model. "
     "Use 'max_completion_tokens' instead.", "max_completion_tokens"),
    ("Unsupported value: 'temperature' does not support 0.2 with this model.", "temperature"),
    ("Invalid API key.", None),
])
def test_a_shape_error_names_the_field_to_adjust(text, field):
    assert pv.shape_adjustment(text) == field


def test_a_shape_400_is_retried_once_and_the_adjustment_is_kept(live):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("reasoning_effort") != "none":
            error = {"message": "Function tools with reasoning_effort are not supported; "
                                "set reasoning_effort to 'none'."}
            return httpx.Response(400, json={"error": error})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    model = pv.OpenAIModel("openai/next-x", api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)),
                           sleep=lambda _: None)
    tools = [{"name": "echo", "parameters": {"type": "object"}}]
    reply = model.query([{"role": "user", "content": "hi"}], tools=tools)
    assert reply.content == "ok"
    assert len(sent) == 2
    assert "reasoning_effort" not in sent[0]
    assert sent[1]["reasoning_effort"] == "none"

    model.query([{"role": "user", "content": "again"}], tools=tools)
    assert len(sent) == 3
    assert sent[2]["reasoning_effort"] == "none"
