"""The harness: the one stateful object, owning the transcript, the tools, the queues and the subscribers.

The loop is a function (loop.py); this is what calls it. It holds the transcript between runs, the
tool registry an extension registers into, the two queues (`steer` and `follow_up`), the hooks, the
system prompt's sections, and the subscribers every event is pushed to. One run at a time: a second
`prompt` while one is running is refused, because two runs would write one transcript (tau's rule,
and the discipline turn-taking gives two harnesses on one workdir, D128).

Context (phase 7, D124, D131). A harness given a `session` records every message the loop appends
into that session tree and puts the tree's active path in front of the model, so a compaction
(the model's forget, or the code floor) changes what the model sees on its next turn without
rewriting the transcript's history. `context` configures the window, the line, whether every tool
result carries the context note (off by default, so a build without it is byte-identical), whether
the floor runs at the end of each turn, and which arm the session is under. Without a session the
harness behaves as it did in phase 2, and `context_stats` still counts the fill at each turn end.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from contextlib import suppress
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Literal, Optional, Union

from kullback.agent.context import ContextConfig, ContextManager, ContextStats
from kullback.agent.events import AgentEvent, CustomMessage, MessageEnd, MessageStart
from kullback.agent.loop import (
    CancelToken,
    Hooks,
    LoopState,
    execute_tool_call,
    interrupted_tool_results,
    run_agent_loop,
)
from kullback.agent.messages import Message, ToolCall, UserMessage
from kullback.agent.session import SessionStore
from kullback.agent.tools import AgentTool, ToolRegistry, ToolResult
from kullback.ai.provider import Model, ModelConfig

Subscriber = Callable[[AgentEvent], Union[None, Awaitable[None]]]


class PromptSection:
    """One named piece of the system prompt; the prompt is the sections joined in order."""

    def __init__(self, name: str, text: str):
        self.name = name
        self.text = text


class AgentHarness:
    def __init__(
        self,
        model: Model,
        system: str = "",
        tools: Iterable[AgentTool] = (),
        hooks: Optional[Hooks] = None,
        config: Optional[ModelConfig] = None,
        max_turns: Optional[int] = None,
        messages: Iterable[Message] = (),
        session: Optional[SessionStore] = None,
        context: Optional[ContextConfig] = None,
    ):
        self.model = model
        self.registry = ToolRegistry(list(tools))
        self.hooks = hooks or Hooks()
        self.config = config
        self.max_turns = max_turns
        self.sections: list[PromptSection] = [PromptSection("base", system)] if system else []
        self._subscribers: list[Subscriber] = []
        self._state: Optional[LoopState] = None
        self._emit: Optional[Callable[[AgentEvent], Awaitable[None]]] = None
        self._steering: list[Message] = []
        self._follow_ups: list[Message] = []
        # The context manager records into the session (when there is one) and starts the
        # transcript from its active path; without a session it holds the messages given.
        self.context = ContextManager(self, session, context or ContextConfig())
        self._messages: list[Message] = self.context.bootstrap(list(messages))
        # asyncio keeps only weak references to tasks, so a subscriber scheduled from a
        # synchronous queue call is held here until it finishes.
        self._pending: set[asyncio.Task] = set()

    # --- what the harness owns ---

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def is_running(self) -> bool:
        return self._state is not None

    @property
    def context_stats(self) -> ContextStats:
        """The context counters (D131): tool calls, refusals, fallback compactions, fill per turn."""
        return self.context.stats

    @property
    def session(self) -> Optional[SessionStore]:
        return self.context.session

    @property
    def system(self) -> str:
        """The system prompt, assembled from the sections in order, blank-line separated."""
        return "\n\n".join(section.text for section in self.sections if section.text)

    def add_prompt_section(self, name: str, text: str, position: Optional[int] = None) -> None:
        """Add a section; the same name replaces in place, a position inserts, no position appends."""
        for section in self.sections:
            if section.name == name:
                section.text = text
                return
        section = PromptSection(name, text)
        if position is None:
            self.sections.append(section)
        else:
            self.sections.insert(position, section)

    def remove_prompt_section(self, name: str) -> bool:
        """Drop a section by name (a skill's text on unload); False when there was none."""
        before = len(self.sections)
        self.sections = [section for section in self.sections if section.name != name]
        return len(self.sections) < before

    def register_tool(self, tool: AgentTool) -> None:
        self.registry.register(tool)

    def sync_context(self) -> None:
        """Put the session's active path in front of the model, and re-read the system prompt
        into the running state, so a compaction or a skill change reaches the next model call."""
        if self.context.session is not None:
            self._messages[:] = self.context.session.active_messages()
        if self._state is not None:
            self._state.system = self.system

    async def emit(self, event: AgentEvent) -> None:
        """Push an event onto the run's stream (or straight to the subscribers between runs).
        The context tools use it for `compaction`; a scheduler will use it for stage events."""
        if self._emit is not None:
            await self._emit(event)
        else:
            await self._notify(event)

    # --- the queues ---

    def steer(self, content: str, details: Optional[dict] = None) -> None:
        """Deliver after the current tool batch, before the next assistant turn."""
        self._queue(UserMessage(content=content, details=details), "steer")

    def follow_up(self, content: str, details: Optional[dict] = None) -> None:
        """Deliver when the run would otherwise stop; each follow-up starts another turn."""
        self._queue(UserMessage(content=content, details=details), "follow_up")

    def send_message(
        self, content: str, details: Optional[dict] = None, deliver_as: Literal["steer", "follow_up"] = "steer"
    ) -> None:
        """An extension's message: content enters the context, details rides beside it and does not."""
        self._queue(UserMessage(content=content, details=details), deliver_as)
        self._notify_sync(CustomMessage(content=content, details=details, deliver_as=deliver_as))

    def _queue(self, message: Message, deliver_as: str) -> None:
        if self._state is not None:
            target = self._state.steering if deliver_as == "steer" else self._state.follow_ups
            target.append(message)
        elif deliver_as == "steer":
            self._steering.append(message)
        else:
            self._follow_ups.append(message)

    def cancel(self) -> None:
        if self._state is not None:
            self._state.cancel.cancel()

    # --- subscribers ---

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._subscribers.remove(fn)

        return unsubscribe

    async def _notify(self, event: AgentEvent) -> None:
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                await result

    def _notify_sync(self, event: AgentEvent) -> None:
        # Queueing is synchronous, so an async subscriber is scheduled on the running loop, or
        # run to completion when no loop is running (a message queued before the first run).
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                try:
                    task = asyncio.get_running_loop().create_task(_await(result))
                except RuntimeError:
                    asyncio.run(_await(result))
                else:
                    self._pending.add(task)
                    task.add_done_callback(self._pending.discard)

    # --- runs ---

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Start a run with one user message. Events are yielded and pushed to subscribers."""
        return self._run([UserMessage(content=content)])

    def continue_(self) -> AsyncIterator[AgentEvent]:
        """Start a run on the transcript as it stands, with whatever the queues hold."""
        return self._run([])

    async def _run(self, prompts: list[Message]) -> AsyncIterator[AgentEvent]:
        if self._state is not None:
            raise RuntimeError("the harness is already running; use steer() or follow_up() to queue a message")
        state = LoopState(
            messages=self._messages,
            system=self.system,
            steering=deque(self._steering),
            follow_ups=deque(self._follow_ups),
            cancel=CancelToken(),
            config=self.config,
            max_turns=self.max_turns,
        )
        self._steering = []
        self._follow_ups = []
        self._state = state
        # Every event goes through one queue so the generator's consumer and the subscribers
        # see the same order; a subscriber that raises stops the run, which is the right default
        # for a session store that could not write.
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        async def emit(event: AgentEvent) -> None:
            # The context manager records the message before anyone else sees the event, so a
            # handler that protects a result finds its entry already in the session.
            self.context.observe(event)
            await self._notify(event)
            await queue.put(event)
            if event.type == "turn_end":
                # The floor (D124): if the model left the context over the line, code compacts
                # and the compaction event follows the turn_end on the same stream.
                await self.context.after_turn(event.turn)

        note_hook = self._context_note_hook() if self.context.config.note else None
        if note_hook is not None:
            self.hooks.tool_result.append(note_hook)
        self._emit = emit

        async def drive() -> None:
            try:
                repairs = interrupted_tool_results(self._messages)
                for repair in repairs:
                    self._messages.append(repair)
                    await emit(MessageStart(message=repair))
                    await emit(MessageEnd(message=repair))
                await run_agent_loop(state, self.model, self.registry, self.hooks, emit, prompts=prompts)
            finally:
                await queue.put(done)

        task = asyncio.create_task(drive())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if note_hook is not None:
                with suppress(ValueError):
                    self.hooks.tool_result.remove(note_hook)
            self._emit = None
            self._steering.extend(state.steering)
            self._follow_ups.extend(state.follow_ups)
            self._state = None

    def _context_note_hook(self):
        """The last tool_result hook of a run: one line with the estimate, the window and the line
        (D124's live meter, which D131 notes has no precedent), after every gate has had its say."""

        def context_note(call: ToolCall, result: ToolResult) -> ToolResult:
            line = self.context.note_for(call.id, result.content)
            return ToolResult(content=f"{result.content}\n{line}", details=result.details, is_error=result.is_error)

        context_note.hook_name = "context_note"  # type: ignore[attr-defined]
        return context_note


async def _await(value: Awaitable[None]) -> None:
    await value


class DriverModel(Model):
    """The model of a harness no model drives: any call on it is a bug, not a request.

    `label` names the application in the error (the Builder's driver, the Examiner's) and in
    `name`, so a transcript says which driver held the harness.
    """

    def __init__(self, label: str = "code"):
        self.label = label
        self.name = f"{label.lower()}-driver"

    def query(self, messages: list[dict], tools: Optional[list[dict]] = None, config: Any = None) -> Any:
        raise RuntimeError(f"the {self.label} driver issues tool calls itself; no model turn was asked for")


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = "driver") -> ToolResult:
    """One tool call through the harness's hooks and registry, with no model turn.

    `loop.execute_tool_call` is the call: a code driver issues a tool the same way the loop would,
    so the hook order, what a raising hook does and the two execution events are the core's, stated
    in one place rather than copied into each application.
    """
    call = ToolCall(id=call_id, name=name, arguments=dict(arguments))
    return asyncio.run(execute_tool_call(call, harness.registry, harness.hooks, harness.emit))
