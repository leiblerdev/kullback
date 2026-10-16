"""Web search and page reading as agent tools, the same two for every role (D375).

Any role on the agent harness (the Examiner, the Builder, the synthetic generator) registers
`search` and `read_page` with `register_web_tools`. The agent core sits below the Builder and the
domain module, so it imports neither: the role builds the search where its layer allows (the D115
chain, `builder.search.search_for`) and passes it in with the destination check (`refuse`, the
domain module's `destination_refusal`). No role wraps the chain itself.

Every result is a summary and rows, as the Examiner's domain tools are. A URL passes `refuse` before
it is fetched, and a refusal is a summary with no rows, never an exception to the model. With no
search both tools still answer, saying the search is off.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool

TOOL_NAMES = ("search", "read_page")
MAX_HITS = 10
DEFAULT_HITS = 8
DEFAULT_PAGE_CHARS = 6000
MAX_PAGE_CHARS = 20_000
OFF = "web search is off for this session: the role registered these tools without a search"

OnCall = Callable[[str, dict, BaseModel], Any]
Refuse = Callable[[str], str]


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="What to search the web for, in a few words.")
    limit: int = Field(default=DEFAULT_HITS, ge=1, le=MAX_HITS, description=f"How many hits, at most {MAX_HITS}.")


class ReadPageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="The http or https URL of the page, usually one a search hit gave.")
    max_chars: int = Field(default=DEFAULT_PAGE_CHARS, ge=1, le=MAX_PAGE_CHARS,
                           description="How many characters of the page's text to show.")
    from_line: int = Field(default=1, ge=1, description="The page line to start at; the summary names the next one.")


class WebResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    rows: list[dict] = Field(default_factory=list)


def _render(result: WebResult) -> str:
    """The summary, then each row: the model reads what came back, not a count."""
    return "\n".join([result.summary, *(json.dumps(row, ensure_ascii=False) for row in result.rows)])


def _searched(search: Any, args: WebSearchArgs) -> WebResult:
    hits = search.search(args.query, args.limit)
    rows = [{"title": hit.title, "url": hit.url, "snippet": hit.snippet, "site": hit.site} for hit in hits]
    return WebResult(summary=f"{len(rows)} hits for {args.query!r}", rows=rows)


def numbered_lines(text: str, max_chars: int, from_line: int = 1) -> list[dict]:
    """The page's lines from `from_line` on, numbered as in the whole page, as many as fit in max_chars.

    A line longer than what is left of the budget is cut to what fits and ends the window, so a long
    line never hides the lines after it: the next call starts at the line after it.
    """
    rows: list[dict] = []
    left = max_chars
    for number, line in enumerate(text.splitlines()[from_line - 1:], start=from_line):
        if left <= 0:
            break
        rows.append({"line": number, "text": line[:left]})
        if len(line) > left:
            break
        left -= len(line) + 1
    return rows


def _read(search: Any, args: ReadPageArgs, refuse: Refuse) -> WebResult:
    why = refuse(args.url)
    if why:
        return WebResult(summary=f"read_page refused {args.url!r}: {why}")
    page = search.fetch([args.url])[0]
    lines = page.text.splitlines()
    rows = numbered_lines(page.text, args.max_chars, args.from_line)
    head = f"{page.title or 'untitled'} ({len(page.text)} characters, {len(lines)} lines)"
    if rows:
        first, last = rows[0]["line"], rows[-1]["line"]
        parts = [f"{head}; showing lines {first} to {last} of {len(lines)}"]
        if len(rows[-1]["text"]) < len(lines[last - 1]):
            parts.append(f"line {last} cut to {len(rows[-1]['text'])} of {len(lines[last - 1])} characters")
        if last < len(lines):
            parts.append(f"call again with from_line={last + 1}")
    else:
        parts = [f"{head}; from_line={args.from_line} is past the end, no lines shown"]
    if page.error:
        parts.append(f"error: {page.error}")
    return WebResult(summary="; ".join(parts), rows=rows)


def web_tools(*, search: Any, refuse: Refuse, on_call: Optional[OnCall] = None) -> list[AgentTool]:
    """The two tools over `search`, anything with `search(query, limit)` and `fetch(urls)`; None is off."""

    def tool(name: str, description: str, args_model: type[BaseModel], answer: Callable[[Any], WebResult]) -> AgentTool:
        async def run(args: Any) -> WebResult:
            result = WebResult(summary=OFF) if search is None else await asyncio.to_thread(answer, args)
            if on_call is not None:
                on_call(name, args.model_dump(mode="json"), result.model_copy(deep=True))
            return result

        return AgentTool(name, description, args_model, WebResult, run, render=_render)

    return [
        tool("search", "Search the web: title, url, snippet and site of each hit. "
             "Example: search(query=\"how long does a refund take\", limit=5).",
             WebSearchArgs, lambda args: _searched(search, args)),
        tool("read_page", "Read one web page as numbered lines of its text, with its title and length. "
             "Example: read_page(url=\"https://example.com/help/refunds\", max_chars=4000, from_line=1).",
             ReadPageArgs, lambda args: _read(search, args, refuse)),
    ]


def register_web_tools(api: Any, *, search: Any, refuse: Refuse, on_call: Optional[OnCall] = None) -> list[str]:
    """Register search and read_page on anything with `register_tool`; returns their names.

    `refuse(url)` says why a destination may not be fetched, or "". `on_call(name, args, result)` is
    called after every call so a role can keep evidence; it gets a copy of the result and never
    changes what the model reads.
    """
    tools = web_tools(search=search, refuse=refuse, on_call=on_call)
    for tool in tools:
        api.register_tool(tool)
    return [tool.name for tool in tools]


__all__ = ["DEFAULT_HITS", "DEFAULT_PAGE_CHARS", "MAX_HITS", "MAX_PAGE_CHARS", "OFF", "TOOL_NAMES", "ReadPageArgs",
           "WebResult", "WebSearchArgs", "numbered_lines", "register_web_tools", "web_tools"]
