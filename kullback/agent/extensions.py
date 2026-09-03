"""Extensions: how an application sits on the core without the core knowing it.

An extension is a `setup(api)` callable. Through the api it registers tools, adds sections to the
system prompt, subscribes to event types, installs `tool_call` and `tool_result` hooks, and sends
messages into the queues. The Builder and the Examiner are extensions (D123, ADR-0007); a customer
workdir may carry skills (text) and never an extension (code). The system prompt is the sections in
the order they were added, with an optional position for a section that must come earlier.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from kullback.agent.context import ContextManager
from kullback.agent.events import AgentEvent
from kullback.agent.harness import AgentHarness
from kullback.agent.loop import ToolCallHook, ToolResultHook
from kullback.agent.messages import ToolCall
from kullback.agent.tools import AgentTool

Setup = Callable[["ExtensionAPI"], Any]
Handler = Callable[[Any], Any]


def refuse_paths(finder: Callable[[Any], Optional[str]], reason: str, hook_name: str) -> ToolCallHook:
    """A `tool_call` hook that blocks any call whose arguments hold a string `finder` objects to.

    `finder` walks the arguments and returns the first offending string, or None. The raise is what
    the loop turns into an is_error result naming the hook, so the refusal is code the model reads
    and never a line in the prompt: the D122 write block on the gates and the Runner and the D123
    read block on the Builder's compiled side are two instances of this one hook.
    """

    def refuse(call: ToolCall) -> None:
        found = finder(call.arguments)
        if found is not None:
            raise PermissionError(f"the arguments name {found!r}, {reason}")

    refuse.hook_name = hook_name  # type: ignore[attr-defined]
    return refuse


class ExtensionAPI:
    def __init__(self, harness: AgentHarness):
        self.harness = harness

    def register_tool(self, tool: AgentTool) -> None:
        self.harness.register_tool(tool)

    def add_prompt_section(self, name: str, text: str, position: Optional[int] = None) -> None:
        self.harness.add_prompt_section(name, text, position)

    def on(self, event_type: str, handler: Handler) -> Callable[[], None]:
        """Call `handler` with every event of that type; returns the unsubscribe."""

        async def listen(event: AgentEvent) -> None:
            if getattr(event, "type", None) != event_type:
                return
            result = handler(event)
            if inspect.isawaitable(result):
                await result

        return self.harness.subscribe(listen)

    def tool_call(self, hook: ToolCallHook) -> None:
        """Before a tool runs: return rewritten arguments, None to leave them, or raise to block."""
        self.harness.hooks.tool_call.append(hook)

    def tool_result(self, hook: ToolResultHook) -> None:
        """After a tool ran: return a rewritten ToolResult, or None to leave it."""
        self.harness.hooks.tool_result.append(hook)

    def send_message(
        self, content: str, details: Optional[dict] = None, deliver_as: Literal["steer", "follow_up"] = "steer"
    ) -> None:
        self.harness.send_message(content, details=details, deliver_as=deliver_as)

    @property
    def system_prompt(self) -> str:
        return self.harness.system

    # --- context (phase 7, D124, D131) ---

    @property
    def context(self) -> ContextManager:
        return self.harness.context

    def protect(self, entry_ids: Sequence[str], reason: str) -> None:
        """Keep entries out of every forget and out of the floor: an unacted gate ruling, an open
        finding, an unfinished repair. The reason is what the model reads when it is refused."""
        self.harness.context.protect(entry_ids, reason)

    def unprotect(self, entry_ids: Sequence[str]) -> None:
        self.harness.context.unprotect(entry_ids)

    def entry_id_for(self, tool_call_id: str) -> Optional[str]:
        """The session entry a tool result has (or will have), so a hook can protect it by id."""
        return self.harness.context.entry_id_for(tool_call_id)

    def catalog_tool(self, tool: AgentTool, loaded: bool = False) -> None:
        """A tool the model may `load`; with `loaded`, it starts in the registry."""
        self.harness.context.catalog_tool(tool, loaded=loaded)

    def catalog_skill(self, name: str, text: str, loaded: bool = False) -> None:
        """A skill's text the model may `load`; with `loaded`, it starts in the prompt."""
        self.harness.context.catalog_skill(name, text, loaded=loaded)

    def remove_prompt_section(self, name: str) -> bool:
        return self.harness.remove_prompt_section(name)


def load_extensions(harness: AgentHarness, setups: Iterable[Setup]) -> ExtensionAPI:
    """Run each extension's setup against one api over the harness, in order."""
    api = ExtensionAPI(harness)
    for setup in setups:
        setup(api)
    return api
