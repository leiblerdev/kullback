"""Prompt caching: the request fingerprint, the cache points on the wire, the TTL and its price."""

from __future__ import annotations

import asyncio

import pytest

from kullback.agent.loop import LoopState, run_agent_loop
from kullback.agent.messages import UserMessage
from kullback.agent.tools import AgentTool, ToolRegistry
from kullback.ai import provider as pv
from kullback.ai.cache import MAX_CACHE_POINTS, cache_control, count_cache_points, fingerprint, ttl_for
from kullback.ai.provider import ModelConfig, TestModel
from kullback.ai.usage import Usage, usage_from_anthropic
from kullback.runner import budget
from tests.agent.conftest import AddArgs, AddResult, EchoArgs, EchoResult, _add, _echo, call, reply

TOOLS = [{"name": "lookup", "description": "Look a thing up.", "input_schema": {"type": "object", "properties": {}}},
         {"name": "write", "description": "Write a thing.", "input_schema": {"type": "object", "properties": {}}}]


def conversation(system="You answer questions.", last="what next?"):
    return [{"role": "system", "content": system},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": last}]


def add_tool():
    return AgentTool("add", "Add two integers.", AddArgs, AddResult, _add)


def echo_tool():
    return AgentTool("echo", "Echo text back.", EchoArgs, EchoResult, _echo)


# --- the fingerprint ---


def test_two_requests_that_differ_only_in_the_last_message_share_their_prefix_and_parent():
    one = fingerprint(conversation(last="what next?"), TOOLS)
    two = fingerprint(conversation(last="and after that?"), TOOLS)
    assert one["prefix_hash"] == two["prefix_hash"]
    assert one["parent_hash"] == two["parent_hash"]
    assert one["history_hash"] != two["history_hash"]


def test_a_changed_system_text_changes_the_prefix_hash():
    assert (fingerprint(conversation(system="You answer questions."), TOOLS)["prefix_hash"]
            != fingerprint(conversation(system="You answer questions briefly."), TOOLS)["prefix_hash"])


def test_a_reordered_tool_list_changes_the_prefix_hash_because_the_wire_prefix_changes():
    assert fingerprint(conversation(), TOOLS)["prefix_hash"] != fingerprint(conversation(), TOOLS[::-1])["prefix_hash"]


def test_a_conversation_that_grew_by_appending_names_the_earlier_request_as_its_parent():
    earlier = conversation()[:2]
    later = earlier + [{"role": "assistant", "content": "an answer"}, {"role": "user", "content": "more"}]
    assert fingerprint(earlier, TOOLS)["parent_hash"] is None
    assert fingerprint(later, TOOLS)["parent_hash"] == fingerprint(earlier, TOOLS)["history_hash"]
    assert fingerprint(later, TOOLS)["n_messages"] == 4


def test_every_assistant_message_of_an_agent_run_carries_the_fingerprint_of_the_request_behind_it():
    model = TestModel([reply("adding", call("add", {"a": 2, "b": 3})), reply("five")])
    events = []
    asyncio.run(run_agent_loop(LoopState(system="sys"), model, ToolRegistry([add_tool()]), None, events.append,
                               prompts=[UserMessage(content="go")]))
    requests = [e.request for e in events if e.type == "message_end" and e.message.role == "assistant"]
    assert len(requests) == 2 and all(requests)
    assert requests[0]["prefix_hash"] == requests[1]["prefix_hash"]
    assert requests[1]["parent_hash"] == requests[0]["history_hash"]


def test_a_message_end_without_a_request_serialises_as_it_did_before_the_field_existed():
    from kullback.agent.events import MessageEndEvent
    from kullback.ai.messages import AssistantMessage

    assert "request" not in MessageEndEvent(message=AssistantMessage(content="ok")).model_dump(mode="json")


# --- the cache points on an Anthropic request ---


def anthropic_body(messages, tools=TOOLS, **config):
    model = pv.AnthropicModel(model_id="anthropic/claude-opus-5", api_key="k", env={})
    return model.build_body(messages, tools, ModelConfig(**config))


def test_an_anthropic_request_marks_tools_system_and_the_last_two_messages_and_nothing_more():
    body = anthropic_body(conversation())
    assert count_cache_points(body) == MAX_CACHE_POINTS
    assert "cache_control" in body["tools"][-1] and "cache_control" not in body["tools"][0]
    assert "cache_control" in body["system"][-1]
    assert [("cache_control" in m["content"][-1]) for m in body["messages"]] == [False, True, True]


def test_no_anthropic_request_carries_more_than_four_marks_however_long_the_conversation():
    messages = conversation()
    for turn in range(30):
        messages += [{"role": "assistant", "content": f"answer {turn}"}, {"role": "user", "content": f"ask {turn}"}]
        assert count_cache_points(anthropic_body(messages)) <= MAX_CACHE_POINTS


def test_an_anthropic_request_never_carries_a_top_level_cache_control_field():
    for ttl in (None, "5m", "1h"):
        assert "cache_control" not in anthropic_body(conversation(), cache_ttl=ttl)


def test_a_one_hour_ttl_puts_the_ttl_on_every_mark_and_the_default_sends_the_bare_mark():
    hour = anthropic_body(conversation(), cache_ttl="1h")
    assert hour["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert hour["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert hour["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert anthropic_body(conversation())["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_a_thinking_block_is_skipped_and_the_mark_goes_on_the_block_before_it():
    messages = conversation() + [{"role": "assistant", "content": [
        {"type": "text", "text": "visible"}, {"type": "thinking", "thinking": "hidden", "signature": "s"}]}]
    marked = pv.cache_last_two(messages)
    assert "cache_control" in marked[-1]["content"][0] and "cache_control" not in marked[-1]["content"][1]


def test_an_empty_text_and_a_signed_empty_thinking_block_are_both_passed_over_for_the_mark():
    blocks = [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
              {"type": "redacted_thinking", "data": "opaque"},
              {"type": "thinking", "thinking": "", "signature": "s"},
              {"type": "text", "text": "  "}]
    marked = pv.cache_last_two(conversation() + [{"role": "assistant", "content": blocks}], ttl="1h")
    assert [("cache_control" in block) for block in marked[-1]["content"]] == [True, False, False, False]
    assert marked[-1]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_an_unknown_ttl_is_refused_where_it_is_set():
    with pytest.raises(ValueError):
        cache_control("2h")
    with pytest.raises(ValueError):
        ModelConfig(cache_ttl="2h")


def test_the_builder_session_writes_one_hour_marks_and_every_other_stage_the_default():
    assert ttl_for("builder_session") == "1h"
    assert ttl_for("runner") == ttl_for("examiner") == ttl_for(None) == "5m"


# --- tool order ---


def test_the_tool_list_a_model_sees_is_the_same_whatever_order_the_tools_were_registered_in():
    assert ToolRegistry([add_tool(), echo_tool()]).schemas() == ToolRegistry([echo_tool(), add_tool()]).schemas()


def test_mined_tool_definitions_come_out_by_name_whatever_order_they_were_mined_in():
    from types import SimpleNamespace

    from kullback.runner.world.loading import _tool_definitions

    sigs = [SimpleNamespace(name=name, args_schema={}, description=None, kind="read") for name in ("b", "c", "a")]
    assert [tool["name"] for tool in _tool_definitions(sigs)] == ["a", "b", "c"]
    assert _tool_definitions(sigs) == _tool_definitions(sigs[::-1])


# --- the price of a one-hour write ---


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setitem(budget.PRICES, "vendor/cached", {"input": 4.0, "output": 20.0, "cache_read": 0.4,
                                                         "cache_write": 5.0})
    return "vendor/cached"


def test_a_one_hour_write_costs_twice_the_input_price_and_a_five_minute_write_its_own_rate(priced):
    five = budget.call_cost(Usage(cache_write=1_000_000), priced)
    hour = budget.call_cost(Usage(cache_write=1_000_000, cache_write_1h=1_000_000), priced)
    assert (five, hour) == (5.0, 8.0)


def test_the_cache_effect_of_a_one_hour_write_is_its_premium_over_plain_input(priced):
    assert budget.cache_effect(Usage(cache_write=1_000_000, cache_write_1h=1_000_000), priced) == -4.0
    assert budget.cache_effect(Usage(cache_write=1_000_000), priced) == -1.0
    assert budget.cache_effect(Usage(cache_read=1_000_000), priced) == pytest.approx(3.6)


def test_the_anthropic_usage_splits_out_the_one_hour_part_of_the_write():
    usage = usage_from_anthropic({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                                  "cache_creation_input_tokens": 300,
                                  "cache_creation": {"ephemeral_5m_input_tokens": 100,
                                                     "ephemeral_1h_input_tokens": 200}})
    assert (usage.input, usage.cache_read, usage.cache_write, usage.cache_write_1h) == (10, 100, 300, 200)


def test_a_one_hour_part_larger_than_the_whole_write_is_refused():
    with pytest.raises(ValueError):
        Usage(cache_write=10, cache_write_1h=11)
