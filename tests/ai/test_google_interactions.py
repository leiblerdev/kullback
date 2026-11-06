"""The Interactions API adapter: the body it sends, the reply it reads, the stream it parses, the bill.

The payloads below were pinned live on 2026-10-28 and are kept verbatim: the tool-call reply and its
second turn from the brief's probe, and two SSE streams recorded with this adapter's request shape.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from kullback.ai import provider as pv
from kullback.ai.events import StreamDone, ThinkingDelta, ToolCallEnd
from kullback.ai.google_interactions import GoogleInteractionsProvider
from kullback.ai.messages import AssistantMessage, ToolResultMessage, UserMessage, to_wire
from kullback.ai.provider import GoogleInteractionsModel, ModelConfig, RecordedModel, _reply_from_dict
from kullback.ai.usage import Usage, usage_from_interactions
from kullback.runner.budget import call_cost

WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
SENT_TOOL = {"type": "function", "name": "get_weather", "description": "Weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
QUESTION = "What is the weather in Munich? Use the tool."

FIRST_BODY = {"model": "gemini-4-argon", "store": False, "system_instruction": "You are terse.",
              "tools": [SENT_TOOL],
              "input": [{"type": "user_input", "content": [{"type": "text", "text": QUESTION}]}],
              "generation_config": {"max_output_tokens": 200}}
FIRST_REPLY = {"status": "requires_action", "object": "interaction", "model": "gemini-4-argon",
               "usage": {"total_tokens": 118, "total_input_tokens": 96,
                         "input_tokens_by_modality": [{"modality": "text", "tokens": 96}],
                         "total_cached_tokens": 0, "total_output_tokens": 22,
                         "output_tokens_by_modality": [{"modality": "text", "tokens": 22}],
                         "total_tool_use_tokens": 0, "total_thought_tokens": 0, "initial_input_tokens": 96},
               "steps": [{"type": "thought", "signature": "ElkKVwFpFH0T...=="},
                         {"type": "function_call", "id": "call_2701184", "name": "get_weather",
                          "arguments": {"city": "Munich"}}]}
FUNCTION_RESULT = {"type": "function_result", "name": "get_weather", "call_id": "call_2701184",
                   "result": [{"type": "text", "text": "18C, cloudy"}]}
SECOND_REPLY = {"status": "completed", "object": "interaction", "model": "gemini-4-argon",
                "usage": {"total_input_tokens": 142, "total_output_tokens": 10, "total_thought_tokens": 0},
                "steps": [{"type": "thought", "signature": "Eq0BCqoB...=="},
                          {"type": "model_output", "content": [{"type": "text", "text": "18C, cloudy."}]}]}

# The tool-call turn, streamed (recorded live 2026-10-28).
TOOL_CALL_STREAM = [
    '{"interaction":{"object":"interaction","model":"gemini-4-argon"},"event_type":"interaction.created"}',
    '{"status":"in_progress","event_type":"interaction.status_update"}',
    '{"index":0,"step":{"type":"thought"},"event_type":"step.start"}',
    '{"index":0,"delta":{"signature":"ElkKVwFpFH0TbtfwW7Z9Fk3g9wOPwA4FPsjfkHRolL1XlBEm4z5b5wmnNdY8e6jzEy30yZu4WiLIzhR7nlQE8b5wuE7BvfXeDriPYqnzUqn05w0WmMI5ruQhaQ==","type":"thought_signature"},"event_type":"step.delta"}',
    '{"index":0,"event_type":"step.stop"}',
    '{"index":1,"step":{"id":"call_3316996","type":"function_call","name":"get_weather","arguments":{}},"event_type":"step.start"}',
    '{"index":1,"delta":{"arguments":"{\\"city\\":\\"Munich\\"}","type":"arguments_delta"},"event_type":"step.delta"}',
    '{"index":1,"event_type":"step.stop"}',
    '{"interaction":{"status":"requires_action","usage":{"total_tokens":122,"total_input_tokens":100,"input_tokens_by_modality":[{"modality":"text","tokens":100}],"total_cached_tokens":0,"total_output_tokens":22,"output_tokens_by_modality":[{"modality":"text","tokens":22}],"total_tool_use_tokens":0,"total_thought_tokens":0,"initial_input_tokens":100},"created":"2026-10-29T02:05:08Z","updated":"2026-10-29T02:05:08Z","service_tier":"standard","object":"interaction","model":"gemini-4-argon"},"event_type":"interaction.completed"}',
    '[DONE]',
]

# The answer after the function result, streamed with thought summaries on (recorded live 2026-10-28).
ANSWER_STREAM = [
    '{"interaction":{"object":"interaction","model":"gemini-4-argon"},"event_type":"interaction.created"}',
    '{"status":"in_progress","event_type":"interaction.status_update"}',
    '{"index":0,"step":{"type":"thought"},"event_type":"step.start"}',
    '{"index":0,"delta":{"content":{"text":"**Composing a Couplet**\\n\\nI\'ve got the weather data for Munich: 18 degrees Celsius and cloudy. Now, I\'m working on crafting a two-line poem, a couplet, that captures this information. This is where I\'m at! I\'m trying to decide if I should incorporate the temperature directly, or use it more figuratively.\\n\\n\\n","type":"text"},"type":"thought_summary"},"event_type":"step.delta"}',
    '{"index":0,"delta":{"signature":"EsUFCsIFAWkUfRN39zTMG/unqJYyiXI4u+9m6LQKiWgZ4I1miGfoEtRa+MQeRMdNv2HRlbt9vqfgj46SDPDuvoYkoOe79Qk0nk/qsGzdwG51XLzFlzdGglHNSLQb5vky1pKs1sa8FEx/6vrl5n1u0mfQBnSnrvkyBdpqNoSZRCxsPCICKrDIOoO19xwiu/EDNBuaem2IWgAOtuCf1S95npx5XhmcKgenT0CjivmR5QLgZk27DExx9ohbSz8OCFt229flP3gHG5sDnf5diDWcsfOtGbZH0HBf1RKi6+LlBUzW9ZQhFAo3/2VT7UEl7Hn7gfNX4FcYMqMZwpxvwXsAWKo9QwiPaAPj16gciEJsYmsR/I24v5HCCj8q4y925XAmMyHBhH1la7AHk/aV0Uly1ukCKoXD6bSNHeno9kvH8axoUFQ6z2zcXDtapL0pli7dUlUhYsoVd6wNqxMWOYXw+l8rXyhsbrMw9Itfc1w2DMELf1Xx8DguMQA9NSbP+LZvwx+X7/4rtLPvIKgRiVz/hY9+aIIsH/qn39VkXCYK0AVb+UmnvDbWiGf6yiNugvBRqUVbYAINmiEygsB7z1VL3G3w7phPC1cD4JS64dom8BaMwNk94HHW1qw8z31kuFMgYyUjGKd04ogn1fat50YwPnACR++5fPDtAJVMidPZ3K/IQi8bzPc/+YJmsY6wTbilzuFYZKkBCFD8Jnrvp/e4kh2QxX+DPn3epV2RdltFX5QIAOV5vCFom+9oObfMv4yJfLsXV7VyaC7P+wvmXdiVPZoagUUBRjkGoIm6V6q8LrEpYT3oMCgGE5NO3qcQdoZtosHZjyE1VpnoHxAxga1Xp3I5SKDIpj5FQJaI3lS+IhnC9VLtli6z2v+/9Ktcr1DqmGtCQLjFM5f9XIgMIj6YJrtatKjLetMmqaFQzD7fsj5bxfExEiCqBg==","type":"thought_signature"},"event_type":"step.delta"}',
    '{"index":0,"event_type":"step.stop"}',
    '{"index":1,"step":{"type":"model_output"},"event_type":"step.start"}',
    '{"index":1,"delta":{"text":"Beneath Munich\'s skies wrapped in a cloudy embrace,  \\nEighteen mild degrees set a comfortable pace.","type":"text"},"event_type":"step.delta"}',
    '{"index":1,"event_type":"step.stop"}',
    '{"interaction":{"status":"completed","usage":{"total_tokens":441,"total_input_tokens":270,"input_tokens_by_modality":[{"modality":"text","tokens":270}],"total_cached_tokens":0,"total_output_tokens":26,"output_tokens_by_modality":[{"modality":"text","tokens":26}],"total_tool_use_tokens":0,"total_thought_tokens":145,"initial_input_tokens":270},"created":"2026-10-29T02:05:37Z","updated":"2026-10-29T02:05:37Z","service_tier":"standard","object":"interaction","model":"gemini-4-argon"},"event_type":"interaction.completed"}',
    '[DONE]',
]


def handle(**kwargs):
    return GoogleInteractionsModel("google/gemini-4-argon", api_key="k", env={}, **kwargs)


def first_turn_wire():
    return to_wire([UserMessage(content=QUESTION)], "You are terse.")


def test_the_first_request_is_the_body_pinned_live():
    body = handle().build_body(first_turn_wire(), [WEATHER_TOOL], ModelConfig(max_tokens=200))
    assert body == FIRST_BODY


def test_the_tool_call_reply_is_read_as_a_call_with_its_thought_step_kept_whole():
    reply = handle().parse_reply(FIRST_REPLY)
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [("call_2701184", "get_weather", {"city": "Munich"})]
    assert reply.thinking_blocks == [FIRST_REPLY["steps"][0]]
    assert reply.stop_reason == "tool_calls" and reply.content is None
    assert reply.usage == Usage(input=96, output=22)


def test_the_second_request_echoes_the_reply_steps_verbatim_before_the_function_result():
    model = handle()
    first = model.parse_reply(FIRST_REPLY)
    assistant = AssistantMessage(content=first.content, thinking_blocks=first.thinking_blocks,
                                 tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments}
                                             for c in first.tool_calls])
    result = ToolResultMessage(tool_call_id="call_2701184", tool_name="get_weather", content="18C, cloudy")
    wire = to_wire([UserMessage(content=QUESTION), assistant, result], "You are terse.")
    body = model.build_body(wire, [WEATHER_TOOL], ModelConfig(max_tokens=200))
    assert body["input"] == FIRST_BODY["input"] + FIRST_REPLY["steps"] + [FUNCTION_RESULT]


def test_the_completed_reply_is_read_as_text_that_stops():
    reply = handle().parse_reply(SECOND_REPLY)
    assert reply.content == "18C, cloudy." and reply.tool_calls == []
    assert reply.stop_reason == "stop" and reply.usage == Usage(input=142, output=10)


def test_an_incomplete_reply_is_read_as_cut_at_the_length_cap():
    reply = handle().parse_reply({"status": "incomplete", "steps": [],
                                  "usage": {"total_input_tokens": 37, "total_output_tokens": 3,
                                            "total_thought_tokens": 47}})
    assert reply.stop_reason == "length" and reply.usage == Usage(input=37, output=50, reasoning=47)


def test_the_generation_config_carries_the_caller_settings_in_google_words():
    config = ModelConfig(temperature=0.2, stop=["zzz"], seed=3, tool_choice="required", effort="low")
    generation = handle().build_body(first_turn_wire(), [WEATHER_TOOL], config)["generation_config"]
    assert generation == {"temperature": 0.2, "stop_sequences": ["zzz"], "seed": 3, "tool_choice": "any",
                          "thinking_level": "low"}


def test_no_generation_config_and_no_tool_choice_are_sent_when_nothing_asks_for_them():
    body = handle().build_body(first_turn_wire(), None, ModelConfig(tool_choice="required"))
    assert "generation_config" not in body and "tools" not in body


def test_a_thought_block_another_provider_wrote_is_not_sent_as_a_step():
    wire = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "tool_calls": [],
             "thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": "s"}]}]
    steps = handle().build_body(wire, None, ModelConfig())["input"]
    assert [step["type"] for step in steps] == ["user_input", "model_output"]


def test_the_exchange_record_keeps_the_system_instruction_out_of_the_sampling():
    model = handle()
    body = model.build_body(first_turn_wire(), [WEATHER_TOOL], ModelConfig(max_tokens=200))
    exchange = model.exchange_of(body, pv.Posted({}, 200, 1, {}), 1.0)
    assert exchange.sampling == {"model": "gemini-4-argon", "store": False,
                                 "generation_config": {"max_output_tokens": 200}}
    assert exchange.n_messages == 1 and exchange.n_tools == 1


def test_interactions_usage_takes_the_cached_part_off_input_and_bills_thought_as_output():
    usage = usage_from_interactions({"total_input_tokens": 9600, "total_cached_tokens": 9000,
                                     "total_output_tokens": 30, "total_thought_tokens": 120,
                                     "total_tool_use_tokens": 0})
    assert usage == Usage(input=600, output=150, cache_read=9000, reasoning=120)


def test_interactions_usage_counts_tool_use_prompt_tokens_as_input_and_reads_nothing_as_zero():
    assert usage_from_interactions({"total_input_tokens": 10, "total_tool_use_tokens": 5}).input == 15
    assert usage_from_interactions(None) == Usage()


def test_a_stored_interactions_payload_replays_and_is_priced_as_the_live_reply(tmp_path):
    payload = {**FIRST_REPLY, "usage": {**FIRST_REPLY["usage"], "total_cached_tokens": 40, "total_thought_tokens": 7}}
    live = handle().parse_reply(payload)
    stored = _reply_from_dict(payload)
    assert stored.usage == live.usage == Usage(input=56, output=29, cache_read=40, reasoning=7)
    assert stored.thinking_blocks == live.thinking_blocks and stored.tool_calls == live.tool_calls
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({"type": "model_call", "payload": {"reply": payload}}) + "\n", encoding="utf-8")
    assert RecordedModel(run).query([]).usage == live.usage


def test_bare_interactions_usage_keys_on_a_stored_reply_are_read_by_the_interactions_rule():
    stored = _reply_from_dict({"content": "hi", "usage": {"total_input_tokens": 100, "total_cached_tokens": 60,
                                                          "total_output_tokens": 5, "total_thought_tokens": 20}})
    assert stored.usage == Usage(input=40, output=25, cache_read=60, reasoning=20)


def write_price(tmp_path):
    """A models.dev snapshot with Google's list price for the model (ai.google.dev pricing, 2026-10-28)."""
    row = {"cost": {"input": 2.0, "output": 10.0, "cache_read": 0.1}}
    catalog = {"google": {"id": "google", "models": {"gemini-4-argon": row}}}
    (tmp_path / "models.dev.json").write_text(
        json.dumps({"fetched_at": "2026-10-28T00:00:00+00:00", "catalog": catalog}), encoding="utf-8")


def test_the_call_cost_prices_cached_and_thought_tokens_at_their_own_rates(tmp_path):
    write_price(tmp_path)
    usage = usage_from_interactions({"total_input_tokens": 10_000, "total_cached_tokens": 8_000,
                                     "total_output_tokens": 300, "total_thought_tokens": 700})
    expected = (2_000 * 2.0 + 8_000 * 0.1 + (300 + 700) * 10.0) / 1_000_000
    assert call_cost(usage, "google/gemini-4-argon") == pytest.approx(expected)


@pytest.fixture
def live(monkeypatch):
    """The transports below never leave the machine, so the live-call gate is opened for them."""
    monkeypatch.setattr(pv, "ALLOW_MODEL_REQUESTS", True)


def streamed(chunks, seen=None, **kwargs):
    """Every canonical event of one streamed response, served by a local transport."""
    def answer(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, content="".join(f"data: {c}\n\n" for c in chunks).encode("utf-8"))

    provider = GoogleInteractionsProvider(handle(), client=httpx.AsyncClient(transport=httpx.MockTransport(answer)))

    async def run():
        return [event async for event in provider.stream_response(**kwargs)]

    return asyncio.run(run())


def test_the_streamed_tool_call_turn_ends_with_the_call_and_the_signed_thought_step(live):
    seen = []
    events = streamed(TOOL_CALL_STREAM, seen, system="You are terse.", messages=[UserMessage(content=QUESTION)],
                      tools=[WEATHER_TOOL])
    sent = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/interactions") and sent["stream"] is True
    assert seen[0].headers["x-goog-api-key"] == "k"
    calls = [e.tool_call for e in events if isinstance(e, ToolCallEnd)]
    assert [(c.id, c.name, c.arguments) for c in calls] == [("call_3316996", "get_weather", {"city": "Munich"})]
    done = events[-1]
    assert isinstance(done, StreamDone) and done.reason == "tool_use"
    assert done.message.thinking_blocks == [{"type": "thought", "signature": json.loads(TOOL_CALL_STREAM[3])["delta"]["signature"]}]
    assert done.message.usage == Usage(input=100, output=22)


def test_the_streamed_answer_reports_its_summary_as_thinking_and_keeps_it_on_the_thought_step(live):
    events = streamed(ANSWER_STREAM, messages=[UserMessage(content="hi")])
    summary = json.loads(ANSWER_STREAM[3])["delta"]["content"]["text"]
    assert "".join(e.delta for e in events if isinstance(e, ThinkingDelta)) == summary
    done = events[-1]
    assert isinstance(done, StreamDone) and done.reason == "stop"
    assert done.message.content.startswith("Beneath Munich")
    step = done.message.thinking_blocks[0]
    assert step["summary"] == [{"type": "text", "text": summary}] and step["signature"].startswith("EsUF")
    assert done.message.usage == Usage(input=270, output=171, reasoning=145)


def test_a_streamed_thought_step_cut_before_its_signature_is_not_kept(live):
    chunks = [ANSWER_STREAM[2], ANSWER_STREAM[3], ANSWER_STREAM[-2]]
    done = streamed(chunks, messages=[UserMessage(content="hi")])[-1]
    assert isinstance(done, StreamDone) and done.message.thinking_blocks is None


def test_a_failed_interaction_ends_the_stream_in_an_error(live):
    chunks = ['{"interaction":{"status":"failed","error":{"message":"quota"}},"event_type":"interaction.completed"}']
    last = streamed(chunks, messages=[UserMessage(content="hi")])[-1]
    assert last.type == "error" and "quota" in (last.error.error_message or "")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_a_blocking_interaction_that_ended_badly_raises_the_providers_message(status):
    with pytest.raises(pv.ProviderError, match="the quota ran out"):
        pv.interactions_reply({"status": status, "error": {"message": "the quota ran out"}, "steps": []}, "m")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_a_blocking_interaction_that_ended_badly_with_no_message_still_raises(status):
    with pytest.raises(pv.ProviderError, match=status):
        pv.interactions_reply({"status": status}, "m")
