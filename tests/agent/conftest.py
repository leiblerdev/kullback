"""Shared helpers: run an async iterator to a list, and the two scripted tools every loop test uses."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from kullback.agent.tools import AgentTool
from kullback.ai.provider import ModelReply, ToolCallRequest


def collect(aiter):
    """Every event of an async iterator, in order, driven by a fresh event loop."""

    async def go():
        return [event async for event in aiter]

    return asyncio.run(go())


def types_of(events) -> list[str]:
    return [event.type for event in events]


def call(name: str, arguments: dict, call_id: str | None = None) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def reply(content: str | None = None, *calls: ToolCallRequest) -> ModelReply:
    return ModelReply(content=content, tool_calls=list(calls))


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int
    b: int


class AddResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int


async def _add(args: AddArgs) -> AddResult:
    return AddResult(total=args.a + args.b)


@pytest.fixture
def add_tool() -> AgentTool:
    return AgentTool("add", "Add two integers.", AddArgs, AddResult, _add)


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class EchoResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    length: int


async def _echo(args: EchoArgs) -> EchoResult:
    return EchoResult(text=args.text, length=len(args.text))


@pytest.fixture
def echo_tool() -> AgentTool:
    return AgentTool("echo", "Echo text back.", EchoArgs, EchoResult, _echo, render=lambda r: r.text)
