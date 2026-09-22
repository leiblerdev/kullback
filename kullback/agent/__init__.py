"""The shared agent core, in the shape of tau_agent: messages, tools, typed events, the provider
seam, the stateless loop, the harness, the session tree, compaction, the extension api. On top of
tau it has the workdir's bus and the base tools bound to a root. It knows nothing about the
application; the Builder, the Examiner and the user are extensions on it (D121, D123). Imports only
`kullback.ai`."""

from __future__ import annotations

from kullback.agent.base_tools import Allowlist, base_tools, register_base_tools
from kullback.agent.bus import Bus, BusRecord
from kullback.agent.compaction import Compactor
from kullback.agent.context import (
    ContextConfig,
    ContextEstimate,
    ContextManager,
    ContextStats,
    estimate_context,
)
from kullback.agent.events import AgentEvent
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness, AgentHarnessConfig
from kullback.agent.loop import CancelToken, Hooks, LoopState, run_agent_loop
from kullback.agent.messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from kullback.agent.provider import ModelProvider, provider_for
from kullback.agent.skills import Skills, skills_from_dir
from kullback.agent.tool_history import interrupted_tool_results, repair_tool_history
from kullback.agent.tools import AgentTool, AgentToolResult, ToolRegistry, ToolResult, attach_ruling

__all__ = [
    "AgentEvent",
    "AgentHarness",
    "AgentHarnessConfig",
    "AgentTool",
    "AgentToolResult",
    "Allowlist",
    "AssistantMessage",
    "Bus",
    "BusRecord",
    "CancelToken",
    "Compactor",
    "ContextConfig",
    "ContextEstimate",
    "ContextManager",
    "ContextStats",
    "ExtensionAPI",
    "Hooks",
    "LoopState",
    "Message",
    "ModelProvider",
    "Skills",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolResultMessage",
    "UserMessage",
    "attach_ruling",
    "base_tools",
    "estimate_context",
    "interrupted_tool_results",
    "load_extensions",
    "provider_for",
    "register_base_tools",
    "repair_tool_history",
    "run_agent_loop",
    "skills_from_dir",
]
