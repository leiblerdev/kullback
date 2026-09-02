"""Extensions: setup(api) registers tools, sections, handlers, hooks, and sends messages."""

from __future__ import annotations

from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import ToolResult
from kullback.ai.provider import TestModel
from tests.agent.conftest import call, collect, reply


def test_sections_added_through_the_api_are_the_harness_prompt():
    # The assembly itself (order, position, same-name replacement) is
    # tests/agent/test_harness.py::test_system_prompt_is_the_sections_in_order; here the claim is
    # that each setup's section, wherever the api put it, is the harness's prompt.
    harness = AgentHarness(TestModel(["ok"]), system="You build Environments.")

    def skills(api: ExtensionAPI):
        api.add_prompt_section("skills", "Skills: grow, compile.")

    def identity(api: ExtensionAPI):
        api.add_prompt_section("identity", "You are the Builder.", position=0)

    def tools(api: ExtensionAPI):
        api.add_prompt_section("tools", "Tools: build(target).")

    api = load_extensions(harness, [skills, identity, tools])
    assert api.system_prompt == harness.system
    assert all(
        text in api.system_prompt
        for text in ("You are the Builder.", "You build Environments.", "Skills: grow, compile.", "Tools: build(target).")
    )


def test_register_tool_and_hooks_reach_the_loop(add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2})), reply("ok")])
    harness = AgentHarness(model)
    seen = []

    def setup(api: ExtensionAPI):
        api.register_tool(add_tool)
        api.tool_call(lambda c: {"a": c.arguments["a"], "b": 10})
        api.tool_result(lambda c, r: ToolResult(content=r.content + " ruled", details=r.details))
        api.on("tool_execution_end", lambda e: seen.append(e.result.content))

    load_extensions(harness, [setup])
    collect(harness.prompt("go"))
    assert seen == ['{"total": 11} ruled']
    assert harness.messages[2].content == '{"total": 11} ruled'


def test_on_filters_by_type_and_returns_unsubscribe():
    harness = AgentHarness(TestModel(["one", "two"]))
    api = ExtensionAPI(harness)
    turns = []
    off = api.on("turn_start", lambda e: turns.append(e.turn))
    collect(harness.prompt("a"))
    off()
    collect(harness.prompt("b"))
    assert turns == [1]


def test_send_message_as_follow_up_from_a_handler():
    harness = AgentHarness(TestModel(["one", "two"]))
    customs = []

    def setup(api: ExtensionAPI):
        api.on("custom_message", lambda e: customs.append((e.deliver_as, e.details)))

        def on_end(event):
            if event.type == "turn_end" and event.turn == 1:
                api.send_message("the examiner found a hole", details={"task": "t1"}, deliver_as="follow_up")

        api.on("turn_end", on_end)

    load_extensions(harness, [setup])
    collect(harness.prompt("go"))
    assert customs == [("follow_up", {"task": "t1"})]
    assert [m.content for m in harness.messages] == ["go", "one", "the examiner found a hole", "two"]
    assert harness.messages[2].details == {"task": "t1"}
