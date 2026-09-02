"""The shared agent core: messages, tools, typed events, the stateless loop, the harness, the session
tree and the extension api. Knows nothing about the application; the Builder and the Examiner are
extensions on it (D121, D123). Imports only `kullback.ai`."""

from __future__ import annotations

from kullback.agent.context import (
    ContextConfig,
    ContextEstimate,
    ContextManager,
    ContextStats,
    Refused,
    estimate_context,
)
from kullback.agent.events import AgentEvent
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.loop import CancelToken, Hooks, LoopState, run_agent_loop
from kullback.agent.messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from kullback.agent.tools import AgentTool, ToolRegistry, ToolResult

__all__ = [
    "AgentEvent",
    "AgentHarness",
    "AgentTool",
    "AssistantMessage",
    "CancelToken",
    "ContextConfig",
    "ContextEstimate",
    "ContextManager",
    "ContextStats",
    "ExtensionAPI",
    "Hooks",
    "LoopState",
    "Message",
    "Refused",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolResultMessage",
    "UserMessage",
    "estimate_context",
    "load_extensions",
    "run_agent_loop",
]
