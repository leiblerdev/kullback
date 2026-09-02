"""The context tools as an extension: forget, recall, load, unload, the index, and the files arm's
note (D124, D131). `setup(api)` reads the harness's arm and registers what that arm carries.

Under the `tools` arm the model gets `forget(entry_ids, note)`, `recall(entry_id)`, `load(name,
kind)`, `unload(name, kind)` and `context_entries()`, the last because the model can only forget
what it can name: every tool result carries its entry id in the context note, and the index lists
the rest (user messages, summaries, the entries a recall can bring back) with sizes and guards.
Under `code_only` nothing is registered and the harness's floor does all the compacting, tau's
shape, the control arm. Under `files` the same need is met file-style (Letta's finding that
generic file operations beat a bespoke memory api): one `note(text)` tool that appends to a notes
section of the prompt, which survives every compaction and is always shown; the floor still runs,
and there is no forget or recall.

The operations themselves live in `context.py` on the manager; this module is their pydantic
shape, the prompt section, and the refusal path: a refused call raises `Refused`, which the loop
turns into an `is_error` result naming the rule.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.context import (
    ContextManager,
    EntriesResult,
    ForgetResult,
    NoteResult,
    RecallResult,
    ToolSetResult,
)
from kullback.agent.extensions import ExtensionAPI
from kullback.agent.tools import AgentTool, NoArgs

SECTION = "context"
CONTEXT_TOOL_NAMES = ("forget", "recall", "load", "unload", "context_entries")


class ForgetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_ids: list[str] = Field(description="Entry ids to drop from the context, as the notes and context_entries name them.")
    note: str = Field(description="The summary that stands in for them: what you still need from them, in your words.")


class RecallArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(description="A forgotten entry's id.")


class ToolSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["tool", "skill"]


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One thing to remember; it is added to the notes shown in every turn.")


def _render_forget(result: ForgetResult) -> str:
    text = f"forgot {len(result.replaced_entry_ids)} entries ({', '.join(result.replaced_entry_ids)}) into {result.compaction_id}"
    if result.widened_entry_ids:
        text += f"; {', '.join(result.widened_entry_ids)} included to keep tool calls paired with their results"
    return f"{text}; about {result.tokens_freed} tokens freed"


def _render_recall(result: RecallResult) -> str:
    return f"[recalled entry {result.entry_id}, {result.kind}; it now stands here, at the end of the context]\n{result.content}"


def _render_tool_set(result: ToolSetResult) -> str:
    text = f"{result.action}ed {result.kind} {result.name}; {result.loaded_tools} tools loaded, soft cap {result.cap}"
    if result.over_cap:
        text += " (over the cap: tool use degrades past it, unload what you are not using)"
    return text


def _render_entries(result: EntriesResult) -> str:
    lines = [result.estimate, "in context:"]
    for line in result.active:
        guard = f" [{line.guard}]" if line.guard else ""
        lines.append(f"- {line.entry_id} {line.kind} ~{line.tokens} tokens{guard}: {line.first_line}")
    if result.forgotten:
        lines.append("forgotten (recall brings one back):")
        for line in result.forgotten:
            lines.append(f"- {line.entry_id} {line.kind} ~{line.tokens} tokens: {line.first_line}")
    return "\n".join(lines)


def _render_note(result: NoteResult) -> str:
    return f"noted; {result.notes} notes are shown in the prompt"


def tools_section(manager: ContextManager) -> str:
    config = manager.config
    loaded = len(manager.harness.registry)
    return (
        "Context management. Your context is measured against a line at "
        f"{config.line:.0%} of a {config.window}-token window, and every tool result ends with the "
        "current estimate and that result's entry id. Keep the context under the line yourself: "
        "forget(entry_ids, note) replaces entries with your note; recall(entry_id) brings a forgotten "
        "entry back at the end of the context; unload(name, kind) drops a tool's schema or a skill's "
        "text and load(name, kind) brings one back; context_entries() lists what is in context with "
        "ids, sizes and guards. Forgetting one side of a tool call drops the whole exchange. You cannot "
        "forget the session root, anything from the current turn, the tool results of the last "
        f"{config.recent_tool_turns} turns, or an entry protected by an open finding or an unacted "
        "ruling. When the context is still over the line at the end of a turn, code compacts the oldest "
        f"entries into a summary and records that you did not. Loaded tools: {loaded} of a soft cap of "
        f"{config.tool_cap}; tool use degrades past the cap, so unload what you are not using."
    )


def files_section(manager: ContextManager) -> str:
    config = manager.config
    return (
        "Context management. Your context is measured against a line at "
        f"{config.line:.0%} of a {config.window}-token window. Your one memory is note(text): it appends "
        "to the notes section of this prompt, which is always shown and survives every compaction. Write "
        "down what you will need later (decisions, identifiers, open questions) before it is compacted "
        "away. When the context is over the line at the end of a turn, code compacts the oldest entries "
        "into a summary."
    )


def setup(api: ExtensionAPI) -> None:
    """Register what the harness's arm carries. Load it after the application's own extension, so
    the section reads the tool count with the application's tools in place."""
    manager = api.context
    if manager.arm == "tools":
        _setup_tools(api, manager)
    elif manager.arm == "files":
        _setup_files(api, manager)
    # code_only: nothing to register; the floor is the harness's and is on by configuration.


def _setup_tools(api: ExtensionAPI, manager: ContextManager) -> None:
    def refresh_section() -> None:
        api.add_prompt_section(SECTION, tools_section(manager))
        manager.refresh()

    async def forget(args: ForgetArgs) -> ForgetResult:
        return await manager.forget(args.entry_ids, args.note)

    async def recall(args: RecallArgs) -> RecallResult:
        return manager.recall(args.entry_id)

    async def load(args: ToolSetArgs) -> ToolSetResult:
        result = manager.load(args.name, args.kind)
        refresh_section()
        return result

    async def unload(args: ToolSetArgs) -> ToolSetResult:
        result = manager.unload(args.name, args.kind)
        refresh_section()
        return result

    async def entries(args: NoArgs) -> EntriesResult:
        return manager.entries()

    api.register_tool(
        AgentTool(
            "forget",
            "Drop entries from your context, replaced by your note. Refused for the session root, the "
            "current turn, recent tool output, and protected entries.",
            ForgetArgs,
            ForgetResult,
            forget,
            render=_render_forget,
        )
    )
    api.register_tool(
        AgentTool(
            "recall",
            "Bring a forgotten entry back; it lands at the end of the context, marked with its id.",
            RecallArgs,
            RecallResult,
            recall,
            render=_render_recall,
        )
    )
    api.register_tool(
        AgentTool(
            "load",
            "Load an available tool (its schema) or skill (its text) into context.",
            ToolSetArgs,
            ToolSetResult,
            load,
            render=_render_tool_set,
        )
    )
    api.register_tool(
        AgentTool(
            "unload",
            "Take a loaded tool's schema or a skill's text out of context; load brings it back.",
            ToolSetArgs,
            ToolSetResult,
            unload,
            render=_render_tool_set,
        )
    )
    api.register_tool(
        AgentTool(
            "context_entries",
            "List the entries in your context with ids, sizes and guards, and the forgotten ones you can recall.",
            NoArgs,
            EntriesResult,
            entries,
            render=_render_entries,
        )
    )
    manager.pinned_tools.update(CONTEXT_TOOL_NAMES)
    refresh_section()


def _setup_files(api: ExtensionAPI, manager: ContextManager) -> None:
    async def note(args: NoteArgs) -> NoteResult:
        return manager.add_note(args.text)

    api.register_tool(
        AgentTool(
            "note",
            "Remember one thing: it is added to the notes shown in your prompt every turn.",
            NoteArgs,
            NoteResult,
            note,
            render=_render_note,
        )
    )
    manager.pinned_tools.add("note")
    api.add_prompt_section(SECTION, files_section(manager))
    api.add_prompt_section("notes", manager.notes_text())
    manager.refresh()
