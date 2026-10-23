"""Scripted model turns: what a TestModel answers with when a test plays the model."""

from __future__ import annotations

from kullback.ai.provider import ModelReply, ToolCallRequest


def reply(content, *calls) -> ModelReply:
    """One scripted assistant turn: some text, and the (name, arguments) tool calls it asks for in order."""
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                  for i, (n, a) in enumerate(calls)])


def tool_reply(name: str, **arguments) -> ModelReply:
    """One turn that calls the tool `name` with `arguments`, under the call id `<name>-1`."""
    return ModelReply(tool_calls=[ToolCallRequest(id=f"{name}-1", name=name, arguments=arguments)])


def tool_reply_as(call_id: str, name: str, arguments: dict) -> ModelReply:
    """One turn with no text that calls the tool `name` under the given call id."""
    return ModelReply(content=None, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])
