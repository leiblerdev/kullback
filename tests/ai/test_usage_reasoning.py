"""Tests for the reasoning share of output: parsing, bounds, cost, stored records."""

from __future__ import annotations

import pytest

from kullback.ai import provider as pv
from kullback.ai.usage import Usage
from kullback.runner import budget
from kullback.runner.records import Cost

MODEL = "openai/gpt-4.1-mini"


def chat_model():
    return pv.OpenAIModel(model_id=MODEL, api_key="k", env={})


def chat_payload(reasoning=None, output=286):
    details = {"cached_tokens": 0}
    if reasoning is not None:
        details["reasoning_tokens"] = reasoning
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1342, "completion_tokens": output,
                  "prompt_tokens_details": details,
                  "completion_tokens_details": {"reasoning_tokens": reasoning} if reasoning is not None else {}},
    }


def responses_payload(reasoning=None):
    usage = {"input_tokens": 10, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 6}}
    if reasoning is not None:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return {
        "status": "completed",
        "model": "muse-spark-1.3-contributor",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        "usage": usage,
    }


def test_the_chat_shape_reports_its_reasoning_share():
    reply = chat_model().parse_reply(chat_payload(reasoning=200))
    assert (reply.usage.output, reply.usage.reasoning) == (286, 200)


def test_the_chat_shape_without_a_count_leaves_zero():
    reply = chat_model().parse_reply(chat_payload())
    assert reply.usage.reasoning == 0


def test_the_responses_shape_reports_its_reasoning_share():
    model = pv.OpenAIResponsesModel(
        model_id="opencode-go/muse-spark-1.3-contributor", base_url="http://127.0.0.1:8080/v1", env={})
    reply = model.parse_reply(responses_payload(reasoning=3))
    assert (reply.usage.output, reply.usage.reasoning) == (4, 3)


def test_the_responses_shape_without_a_count_leaves_zero():
    model = pv.OpenAIResponsesModel(
        model_id="opencode-go/muse-spark-1.3-contributor", base_url="http://127.0.0.1:8080/v1", env={})
    assert model.parse_reply(responses_payload()).usage.reasoning == 0


def test_the_messages_shape_carries_no_count_so_it_stays_zero():
    model = pv.AnthropicModel(model_id="anthropic/claude-opus-5", api_key="k", env={})
    reply = model.parse_reply({
        "content": [{"type": "text", "text": "hi"}],
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    })
    assert reply.usage.reasoning == 0


def test_a_stored_reply_shape_keeps_its_reasoning_count():
    reply = pv._reply_from_dict({
        "content": "hi",
        "usage": {"input": 10, "output": 9, "reasoning": 4},
    })
    assert (reply.usage.output, reply.usage.reasoning) == (9, 4)


def test_a_stored_reply_shape_without_a_count_defaults_to_zero():
    reply = pv._reply_from_dict({"content": "hi", "usage": {"input": 10, "output": 9}})
    assert reply.usage.reasoning == 0


def test_reasoning_is_never_negative_and_never_above_output():
    with pytest.raises(ValueError):
        Usage(reasoning=-1, output=10)
    with pytest.raises(ValueError):
        Usage(output=5, reasoning=6)
    assert Usage(output=5, reasoning=5).reasoning == 5
    assert Usage(output=5).reasoning == 0


def test_a_provider_count_above_output_is_refused_not_clamped():
    with pytest.raises(ValueError):
        chat_model().parse_reply(chat_payload(reasoning=300, output=286))


def test_a_record_with_the_count_costs_what_it_cost_without_it():
    model = "anthropic/claude-opus-5"
    plain = Usage(input=100, output=50)
    shared = Usage(input=100, output=50, reasoning=30)
    assert budget.call_cost(shared, model) == budget.call_cost(plain, model)
    assert budget.call_cost(shared, model) > 0


def test_a_record_stored_before_the_count_still_loads():
    loaded = Usage.model_validate({"input": 10, "output": 9, "cache_read": 1, "cache_write": 0})
    assert loaded.reasoning == 0
    cost = Cost.model_validate({
        "provider": "openai", "model": MODEL,
        "usage": {"input": 10, "output": 9}, "usd": 0.0, "wall_ms": 1.0,
    })
    assert cost.usage.reasoning == 0


def test_unknown_keys_are_still_refused():
    with pytest.raises(ValueError):
        Usage.model_validate({"input": 1, "output": 1, "reasoning_typo": 1})


def test_an_unreported_count_is_omitted_from_the_stored_form():
    """A record with no count is byte-identical to one written before the field existed."""
    assert Usage(input=10, output=9).model_dump() == {
        "input": 10, "output": 9, "cache_read": 0, "cache_write": 0,
    }
    assert "reasoning" not in Usage(input=10, output=9).model_dump_json()
    assert Usage(input=10, output=9).reasoning == 0


def test_a_reported_count_is_stored_and_round_trips():
    dumped = Usage(input=10, output=9, reasoning=4).model_dump()
    assert dumped["reasoning"] == 4
    assert Usage.model_validate(dumped) == Usage(input=10, output=9, reasoning=4)
    assert Usage.model_validate_json(Usage(input=10, output=9, reasoning=4).model_dump_json()) == \
        Usage(input=10, output=9, reasoning=4)
