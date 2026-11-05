"""Tests for thinking a chat or Responses endpoint bills inside total_tokens and reports nowhere else."""

from __future__ import annotations

from kullback.ai import provider as pv
from kullback.ai.usage import usage_from_openai_chat, usage_from_openai_responses

OBSERVED = {"prompt_tokens": 63, "completion_tokens": 7, "total_tokens": 153}


def test_the_observed_chat_payload_bills_the_eighty_three_hidden_tokens_as_output_and_reasoning():
    usage = usage_from_openai_chat(OBSERVED)
    assert (usage.input, usage.output, usage.reasoning) == (63, 90, 83)


def test_a_chat_payload_without_a_total_is_read_as_before():
    usage = usage_from_openai_chat({"prompt_tokens": 63, "completion_tokens": 7})
    assert (usage.input, usage.output, usage.reasoning) == (63, 7, 0)


def test_a_chat_total_equal_to_the_sum_or_zero_changes_nothing():
    for total in (70, 0, 50):
        usage = usage_from_openai_chat({"prompt_tokens": 63, "completion_tokens": 7, "total_tokens": total})
        assert (usage.output, usage.reasoning) == (7, 0)


def test_a_chat_reasoning_count_inside_completion_keeps_todays_reading():
    usage = usage_from_openai_chat({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                                    "completion_tokens_details": {"reasoning_tokens": 30}})
    assert (usage.output, usage.reasoning) == (50, 30)


def test_cached_and_written_tokens_still_come_off_input_when_thinking_is_hidden():
    usage = usage_from_openai_chat({**OBSERVED, "prompt_tokens_details": {"cached_tokens": 20,
                                                                          "cache_write_tokens": 3}})
    assert (usage.input, usage.cache_read, usage.cache_write, usage.output) == (40, 20, 3, 90)


def test_the_responses_shape_bills_hidden_thinking_and_keeps_a_reported_count_inside_output():
    hidden = usage_from_openai_responses({"input_tokens": 63, "output_tokens": 7, "total_tokens": 153})
    assert (hidden.output, hidden.reasoning) == (90, 83)
    inside = usage_from_openai_responses({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                                          "output_tokens_details": {"reasoning_tokens": 30}})
    assert (inside.output, inside.reasoning) == (50, 30)
    assert usage_from_openai_responses({"input_tokens": 63, "output_tokens": 7}).output == 7


def test_a_stored_chat_payload_with_a_hidden_total_bills_it_like_the_live_reader():
    reply = pv._reply_from_dict({"choices": [], "content": "266", "usage": OBSERVED})
    assert (reply.usage.output, reply.usage.reasoning) == (90, 83)
    stored = pv._reply_from_dict({"content": "266", "usage": {"input": 63, "output": 7, "total_tokens": 153}})
    assert stored.usage.output == 7


def test_written_tokens_a_gateway_keeps_outside_the_prompt_count_are_added_back_before_input_is_split():
    from kullback.ai.usage import usage_from_openai_chat

    outside = {"prompt_tokens": 29, "completion_tokens": 5, "prompt_tokens_details": {
        "cached_tokens": 0, "cache_creation_tokens": 10890, "cache_creation_in_prompt": False}}
    assert (usage_from_openai_chat(outside).input, usage_from_openai_chat(outside).cache_write) == (29, 10890)
    top_level = {"prompt_tokens": 11000, "completion_tokens": 5, "cache_read_input_tokens": 10000,
                 "cache_creation_input_tokens": 900}
    assert (usage_from_openai_chat(top_level).input, usage_from_openai_chat(top_level).cache_read,
            usage_from_openai_chat(top_level).cache_write) == (100, 10000, 900)
