"""The world tools every scenario session holds: rows by key, rows by a field's value, tools by words (D381).

The generator binds each scenario to rows it looked up with these, and the review checks the same rows
and tools with them, so both roles read the world through one definition. They read the Starting state
and the tool list only; none changes anything.
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.examiner.domain_tools import find_tools
from kullback.spec.writer_tools import read_tools
from kullback.synth.rule import World

NAMES = ("lookup_rows", "search_rows", "find_tools")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupArgs(Strict):
    table: str
    key: str = ""


class RowSearchArgs(Strict):
    table: str
    field: str
    value: str


class FindArgs(Strict):
    query: str


class Rows(Strict):
    summary: str
    rows: list[dict] = Field(default_factory=list)


def render(result: Rows) -> str:
    return "\n".join([result.summary, *(json.dumps(row, sort_keys=True) for row in result.rows)])


def world_tools(world: World) -> list[AgentTool]:
    """The three lookups over one world's Starting state and tool list."""
    lookups = read_tools(world.state)

    async def lookup_rows(args: LookupArgs) -> Rows:
        return Rows(summary=lookups["lookup_rows"](args.table, args.key))

    async def search_rows(args: RowSearchArgs) -> Rows:
        return Rows(summary=lookups["search_rows"](args.table, args.field, args.value))

    async def find(args: FindArgs) -> Rows:
        found = find_tools(args.query, world.tools)
        return Rows(summary=found.summary, rows=found.tools)

    def tool(name: str, text: str, args: type, run: Callable) -> AgentTool:
        return AgentTool(name, text, args, Rows, run, render=render)

    return [tool("lookup_rows", "Read rows of the world by table and key; with no key, the count and sample keys.",
                 LookupArgs, lookup_rows),
            tool("search_rows", "Find row keys whose dotted field equals a value.", RowSearchArgs, search_rows),
            tool("find_tools", "Find the tools whose name or description match a few words.", FindArgs, find)]


def binding_refusal(binding: dict, world: World) -> str:
    """The first binding entry the world does not hold, named as table and key, or nothing."""
    for table, key in (binding or {}).items():
        rows = world.state.get(str(table))
        if not isinstance(rows, dict) or str(key) not in rows:
            return f"table {table!r} key {key!r}"
    return ""


__all__ = ["FindArgs", "LookupArgs", "NAMES", "RowSearchArgs", "binding_refusal", "world_tools"]
