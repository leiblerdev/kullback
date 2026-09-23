"""AgentTool: the schema comes from the args model; both sides are validated; failures are results.
And the one format a gate's ruling takes when it rides a tool result."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import pytest

from kullback.agent.events import ToolExecutionEnd
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import AgentTool, NoArgs, TextResult, ToolRegistry, ToolResult, attach_ruling
from kullback.ai.provider import TestModel
from tests.agent.conftest import AddArgs, AddResult, call, collect, reply


def test_schema_is_derived_from_the_args_model(add_tool):
    schema = add_tool.schema()
    assert schema["name"] == "add"
    assert schema["description"] == "Add two integers."
    assert set(schema["input_schema"]["properties"]) == {"a", "b"}
    assert schema["input_schema"]["required"] == ["a", "b"]
    assert schema["input_schema"]["additionalProperties"] is False


def test_valid_arguments_run_and_the_result_is_json_with_details(add_tool):
    result = asyncio.run(add_tool.run({"a": 2, "b": 3}))
    assert result.is_error is False
    assert json.loads(result.content) == {"total": 5}
    assert result.details == {"total": 5}

    # a dict of the right shape is validated into the result model
    async def as_dict(args: AddArgs):
        return {"total": args.a + args.b}

    tool = AgentTool("add", "adds", AddArgs, AddResult, as_dict)
    result = asyncio.run(tool.run({"a": 1, "b": 1}))
    assert result.is_error is False and result.details == {"total": 2}


def test_invalid_arguments_are_an_error_result_not_an_exception(add_tool):
    result = asyncio.run(add_tool.run({"a": "x"}))
    assert result.is_error is True
    assert "invalid arguments for add" in result.content
    assert "a:" in result.content and "b:" in result.content
    assert result.details is None
    # in a run the error result reaches the transcript and the model answers it
    harness = AgentHarness(TestModel([reply(None, call("add", {"a": "two", "b": 3})), reply("sorry")]), tools=[add_tool])
    events = collect(harness.prompt("go"))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error is True
    assert "invalid arguments for add" in end.result.content
    tool_message = harness.messages[2]
    assert tool_message.is_error is True and tool_message.content == end.result.content
    assert harness.messages[-1].content == "sorry"


def test_render_decides_what_the_model_reads_and_details_keeps_the_rest(echo_tool):
    result = asyncio.run(echo_tool.run({"text": "hello"}))
    assert result.content == "hello"
    assert result.details == {"text": "hello", "length": 5}


def test_an_executor_that_raises_or_returns_the_wrong_shape_is_an_error_result():
    async def boom(args: NoArgs) -> TextResult:
        raise RuntimeError("no disk")

    tool = AgentTool("boom", "fails", NoArgs, TextResult, boom)
    result = asyncio.run(tool.run({}))
    assert result.is_error and "boom failed: RuntimeError: no disk" == result.content

    async def wrong(args: NoArgs):
        return {"not": "a text result"}

    tool = AgentTool("wrong", "returns junk", NoArgs, TextResult, wrong)
    result = asyncio.run(tool.run({}))
    assert result.is_error and "wrong shape" in result.content


def test_cancellation_escapes_the_tool():
    async def slow(args: NoArgs) -> TextResult:
        raise asyncio.CancelledError()

    tool = AgentTool("slow", "cancels", NoArgs, TextResult, slow)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tool.run({}))


def test_the_registry_holds_tools_by_name_and_renders_their_schemas_in_registration_order(add_tool, echo_tool):
    registry = ToolRegistry([add_tool])
    registry.register(echo_tool)
    assert registry.names() == ["add", "echo"]
    assert registry.get("add") is add_tool
    assert registry.get("nope") is None
    assert "echo" in registry and len(registry) == 2
    assert [s["name"] for s in registry.schemas()] == ["add", "echo"]


def test_the_registry_refuses_a_duplicate_name_and_removes_a_tool_by_name(add_tool, echo_tool):
    registry = ToolRegistry([add_tool, echo_tool])
    with pytest.raises(ValueError):
        registry.register(add_tool)
    registry.remove("add")
    assert registry.names() == ["echo"] and len(registry) == 1


# --- the ruling a gate attaches to a tool result ---


@dataclass
class Ruling:
    name: str
    accepted: bool
    line: str = ""
    rows: list = dataclass_field(default_factory=list)


def test_a_ruling_is_appended_in_one_format_with_the_rows_that_still_differ():
    rows = [{"call": "c1", "recorded": 3, "ours": 4}, {"call": "c2", "recorded": 1, "ours": 0}]
    result = attach_ruling(
        ToolResult(content="wrote a new file t.py, 40 characters", details={"path": "t.py"}),
        Ruling(name="body", accepted=False, line="2 of 5 calls differ", rows=rows),
    )
    assert result.content.splitlines() == [
        "wrote a new file t.py, 40 characters",
        '<ruling name="body" accepted="no">',
        "2 of 5 calls differ",
        '{"call": "c1", "ours": 4, "recorded": 3}',
        '{"call": "c2", "ours": 0, "recorded": 1}',
        "2 row(s) still differ; each line above is one of them.",
        "</ruling>",
    ]
    assert result.details["path"] == "t.py"
    assert result.details["rulings"] == [
        {"name": "body", "accepted": False, "line": "2 of 5 calls differ", "rows": rows}
    ]
    # an accepted ruling with no rows is the head, the line and the close
    result = attach_ruling(ToolResult(content="ok"), Ruling(name="verifier", accepted=True, line="every check held"))
    assert result.content == 'ok\n<ruling name="verifier" accepted="yes">\nevery check held\n</ruling>'
    assert result.details == {"rulings": [{"name": "verifier", "accepted": True, "line": "every check held", "rows": []}]}


def test_a_second_ruling_is_appended_beside_the_first_and_both_stand_in_details():
    first = attach_ruling(ToolResult(content="ok"), Ruling(name="body", accepted=True))
    second = attach_ruling(first, Ruling(name="shape", accepted=False, line="one column is missing"))
    assert second.content.count("<ruling") == 2
    assert [r["name"] for r in second.details["rulings"]] == ["body", "shape"]
    assert [r["accepted"] for r in second.details["rulings"]] == [True, False]


def test_a_ruling_never_changes_whether_the_call_itself_failed():
    failed = attach_ruling(ToolResult(content="write failed", is_error=True), Ruling(name="body", accepted=True))
    assert failed.is_error is True
    passed = attach_ruling(ToolResult(content="ok"), Ruling(name="body", accepted=False))
    assert passed.is_error is False
