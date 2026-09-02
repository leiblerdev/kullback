"""Tools the loop can call: pydantic arguments in, pydantic result out, validated on both sides.

A tool is a name, a description, an args model, a result model and an async executor from one to
the other. The JSON schema the provider sees is derived from the args model, so the schema and the
validation can never disagree. Arguments are validated before the executor runs and a failure is a
tool result with is_error set, which the model reads and can correct; it is never an exception out
of the loop. Results are validated before they enter the transcript for the same reason in the
other direction: a tool that returns the wrong shape is a bug the transcript records as an error
result, not a crash that loses the run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Args = TypeVar("Args", bound=BaseModel)
Result = TypeVar("Result", bound=BaseModel)


class ToolResult(BaseModel):
    """What one tool execution produced, in the shape the transcript takes.

    `content` is what the model reads. `details` is the validated result as data, and whatever a
    `tool_result` hook appends to it (a gate ruling, D123); it never enters the model context.
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    details: Optional[dict[str, Any]] = None
    is_error: bool = False


class AgentTool(Generic[Args, Result]):
    """One tool: args model, result model, executor. `run` is the only way the loop calls it."""

    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[Args],
        result_model: type[Result],
        execute: Callable[[Args], Awaitable[Result]],
        render: Optional[Callable[[Result], str]] = None,
    ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.result_model = result_model
        self.execute = execute
        # The text the model reads; JSON of the whole result unless the tool says otherwise. A
        # tool whose result is large (a Run) renders a short account and leaves the rest in
        # details, which is how a result stays out of the context without being lost.
        self.render = render
        self._input_schema: Optional[dict] = None

    def schema(self) -> dict:
        """The tool in the provider-neutral shape Model.query takes: name, description, input_schema.

        The args model's JSON schema is derived once, on the first call: pydantic does not cache it
        and every context estimate renders every loaded tool's schema. Derived on first use rather
        than at construction, so a model with an unresolved forward reference fails where it is used.
        """
        if self._input_schema is None:
            self._input_schema = self.args_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._input_schema,
        }

    def parse_arguments(self, arguments: dict[str, Any]) -> Args:
        return self.args_model.model_validate(arguments)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate, execute, validate. Every failure is an is_error result; only a cancellation escapes."""
        try:
            args = self.parse_arguments(arguments)
        except ValidationError as exc:
            return ToolResult(content=f"invalid arguments for {self.name}: {_validation_text(exc)}", is_error=True)
        try:
            raw = await self.execute(args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool is an isolation boundary
            return ToolResult(content=f"{self.name} failed: {type(exc).__name__}: {exc}", is_error=True)
        try:
            result = raw if isinstance(raw, self.result_model) else self.result_model.model_validate(raw)
        except ValidationError as exc:
            return ToolResult(
                content=f"{self.name} returned a result of the wrong shape: {_validation_text(exc)}",
                is_error=True,
            )
        details = result.model_dump(mode="json")
        content = self.render(result) if self.render is not None else json.dumps(details, ensure_ascii=False)
        return ToolResult(content=content, details=details, is_error=False)


def _validation_text(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        where = ".".join(str(p) for p in error.get("loc", ())) or "(root)"
        parts.append(f"{where}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


class ToolRegistry:
    """The tools a run may call, by name, and their schemas for the provider."""

    def __init__(self, tools: Optional[list[AgentTool]] = None):
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


class NoArgs(BaseModel):
    """The args model of a tool that takes nothing."""

    model_config = ConfigDict(extra="forbid")


class TextResult(BaseModel):
    """The result model of a tool that returns one string."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="")
