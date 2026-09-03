"""The harness: transcript across runs, subscribers, the queues, cancel, and the one-run rule."""

from __future__ import annotations

import asyncio

import pytest

from kullback.agent.harness import AgentHarness
from kullback.agent.messages import ToolResultMessage, UserMessage
from kullback.ai.provider import TestModel
from tests.agent.conftest import call, collect, reply, types_of


def test_prompt_yields_events_and_keeps_the_transcript():
    harness = AgentHarness(TestModel(["one", "two"]), system="sys")
    first = collect(harness.prompt("a"))
    assert types_of(first)[0] == "agent_start" and types_of(first)[-1] == "agent_end"
    second = collect(harness.prompt("b"))
    assert [m.role for m in harness.messages] == ["user", "assistant", "user", "assistant"]
    assert harness.messages[-1].content == "two"
    assert types_of(second).count("turn_start") == 1
    assert harness.is_running is False


def test_subscribers_see_every_event_in_order_and_can_unsubscribe():
    harness = AgentHarness(TestModel(["one", "two"]))
    seen = []
    unsubscribe = harness.subscribe(seen.append)
    yielded = collect(harness.prompt("a"))
    assert types_of(seen) == types_of(yielded)
    unsubscribe()
    collect(harness.prompt("b"))
    assert len(seen) == len(yielded)


def test_async_subscriber_is_awaited():
    harness = AgentHarness(TestModel(["one"]))
    seen = []

    async def listen(event):
        await asyncio.sleep(0)
        seen.append(event.type)

    harness.subscribe(listen)
    collect(harness.prompt("a"))
    assert seen[0] == "agent_start" and seen[-1] == "agent_end"


def test_steer_from_a_subscriber_lands_after_the_tool_batch(add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("ok")])
    harness = AgentHarness(model, tools=[add_tool])

    def on_event(event):
        if event.type == "tool_execution_end":
            harness.steer("careful")

    harness.subscribe(on_event)
    collect(harness.prompt("go"))
    assert [m.role for m in harness.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert harness.messages[3].content == "careful"


def test_follow_up_queued_before_the_run_is_delivered_at_the_end():
    harness = AgentHarness(TestModel(["one", "two"]))
    harness.follow_up("and then")
    events = collect(harness.prompt("go"))
    assert types_of(events).count("turn_start") == 2
    assert [m.content for m in harness.messages] == ["go", "one", "and then", "two"]


def test_continue_runs_on_the_transcript_as_it_stands():
    harness = AgentHarness(TestModel(["one", "two"]))
    collect(harness.prompt("go"))
    harness.steer("more")
    collect(harness.continue_())
    assert [m.content for m in harness.messages] == ["go", "one", "more", "two"]


def test_send_message_emits_custom_message_and_delivers_content_only():
    harness = AgentHarness(TestModel(["one", "two"]))
    seen = []
    harness.subscribe(seen.append)
    harness.send_message("finding: tool x is wrong", details={"gates": {"x": "fail"}}, deliver_as="follow_up")
    custom = [e for e in seen if e.type == "custom_message"]
    assert len(custom) == 1 and custom[0].deliver_as == "follow_up" and custom[0].details == {"gates": {"x": "fail"}}
    model = harness.model
    collect(harness.prompt("go"))
    delivered = harness.messages[2]
    assert isinstance(delivered, UserMessage) and delivered.details == {"gates": {"x": "fail"}}
    wire = model.calls[1]["messages"]
    assert wire[-1] == {"role": "user", "content": "finding: tool x is wrong"}


def test_a_second_run_while_running_is_refused():
    harness = AgentHarness(TestModel(["one"]))
    refused = []

    async def go():
        async for event in harness.prompt("go"):
            if event.type == "turn_start":
                assert harness.is_running
                # prompt() returns a generator; the check runs when it is first driven
                with pytest.raises(RuntimeError, match="already running"):
                    await harness.prompt("again").__anext__()
                refused.append(True)

    asyncio.run(go())
    assert refused == [True]
    assert harness.is_running is False


def test_cancel_marks_the_rest_of_the_batch_and_repairs_next_run(add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 1}), call("add", {"a": 2, "b": 2})), reply("after")])
    harness = AgentHarness(model, tools=[add_tool])

    def on_event(event):
        if event.type == "tool_execution_end":
            harness.cancel()

    harness.subscribe(on_event)
    collect(harness.prompt("go"))
    results = [m for m in harness.messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 2 and results[1].is_error
    assert harness.messages[-1].stop_reason == "error"
    # the next run finds nothing dangling, and the transcript is valid on the wire
    events = collect(harness.continue_())
    assert types_of(events)[-1] == "agent_end"
    assert harness.messages[-1].content == "after"


def test_system_prompt_is_the_sections_in_order():
    harness = AgentHarness(TestModel(["ok"]), system="base")
    harness.add_prompt_section("tools", "use them")
    harness.add_prompt_section("first", "read this first", position=0)
    harness.add_prompt_section("tools", "use them well")
    assert harness.system == "read this first\n\nbase\n\nuse them well"
    collect(harness.prompt("go"))
    assert harness.model.calls[0]["messages"][0] == {"role": "system", "content": harness.system}


def test_subscriber_exception_stops_the_run():
    harness = AgentHarness(TestModel(["ok"]))

    def bad(event):
        if event.type == "turn_start":
            raise OSError("disk full")

    harness.subscribe(bad)
    with pytest.raises(OSError):
        collect(harness.prompt("go"))
    assert harness.is_running is False


def test_beat_start_beat_end_and_the_exit_on_round_end_are_events_of_the_union_and_serialize_by_type():
    """The round driver's beats (D128) and the exit on round_end (D126) travel the one typed stream
    every subscriber reads, so a dict off the wire comes back as the typed event by its `type`."""
    from pydantic import TypeAdapter

    from kullback.agent.events import AgentEvent, BeatEnd, BeatStart, RoundEnd

    events = TypeAdapter(AgentEvent)
    start = events.validate_python({"type": "beat_start", "agent": "builder", "round": 1})
    end = events.validate_python({"type": "beat_end", "agent": "examiner", "round": 1, "spend": 0.25})
    finished = events.validate_python({"type": "round_end", "round": 1, "counts": {"trusted": 1}, "exit": "done"})
    assert isinstance(start, BeatStart) and isinstance(end, BeatEnd) and isinstance(finished, RoundEnd)
    assert (start.agent, start.round, end.spend, finished.exit) == ("builder", 1, 0.25, "done")
    assert BeatEnd(agent="builder", round=2).model_dump()["spend"] == 0.0
    assert RoundEnd(round=2).exit is None
    with pytest.raises(ValueError):
        events.validate_python({"type": "beat_start", "agent": "referee", "round": 1})
    with pytest.raises(ValueError):
        events.validate_python({"type": "round_end", "round": 1, "exit": "gave up"})
