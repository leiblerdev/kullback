"""AgentTool: the schema comes from the args model; both sides are validated; failures are results."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.agent.tools import AgentTool, NoArgs, TextResult, ToolRegistry
from tests.agent.conftest import AddArgs, AddResult


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


def test_invalid_arguments_are_an_error_result_not_an_exception(add_tool):
    result = asyncio.run(add_tool.run({"a": "x"}))
    assert result.is_error is True
    assert "invalid arguments for add" in result.content
    assert "a:" in result.content and "b:" in result.content
    assert result.details is None


def test_render_decides_what_the_model_reads_and_details_keeps_the_rest(echo_tool):
    result = asyncio.run(echo_tool.run({"text": "hello"}))
    assert result.content == "hello"
    assert result.details == {"text": "hello", "length": 5}


def test_executor_exception_is_an_error_result():
    async def boom(args: NoArgs) -> TextResult:
        raise RuntimeError("no disk")

    tool = AgentTool("boom", "fails", NoArgs, TextResult, boom)
    result = asyncio.run(tool.run({}))
    assert result.is_error and "boom failed: RuntimeError: no disk" == result.content


def test_wrong_result_shape_is_an_error_result():
    async def wrong(args: NoArgs):
        return {"not": "a text result"}

    tool = AgentTool("wrong", "returns junk", NoArgs, TextResult, wrong)
    result = asyncio.run(tool.run({}))
    assert result.is_error and "wrong shape" in result.content


def test_a_dict_of_the_right_shape_is_validated_into_the_result_model():
    async def as_dict(args: AddArgs):
        return {"total": args.a + args.b}

    tool = AgentTool("add", "adds", AddArgs, AddResult, as_dict)
    result = asyncio.run(tool.run({"a": 1, "b": 1}))
    assert result.is_error is False and result.details == {"total": 2}


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
