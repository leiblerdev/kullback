"""The synthetic generator's mining session: a role on the shared agent harness and event stream (D372).

Set up as the Examiner's sessions and the scenario review are: base tools over the scenario store, the
role's own tools with pydantic arguments and results carrying rows, prompt sections in GEPA order, the
policy whole beside them, the workdir bus under the agent name synth and the budget subscriber under a
synth stage. One session per batch of one source.

The rule lives in the tools. write_scenario and rewrite_scenario refuse a statement that names a tool
of the Environment or carries an id shape, a policy line the policy does not hold, a span the page does
not hold, a page the session never read, a typed line copied as typed, and say why as the tool result,
so the agent writes again. A second refusal of the same scenario drops it, and a scenario refused once
and never written again is dropped at the end of the session; both are counted per reason.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback import domain
from kullback.agent.base_tools import register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig, prompt_block
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import AgentTool
from kullback.examiner import session as session_mod
from kullback.runner import budget
from kullback.synth import mine_prompt
from kullback.synth import prompt as prompt_mod
from kullback.synth import records as R
from kullback.synth.rule import EMPTY, World, byte_range, statement_refusal

AGENT = "synth"
BASE_ONLY = ("read", "grep", "find", "ls")
# The turn budget of one session: per item, at least and at most.
TURNS_PER_ITEM = 3
MIN_TURNS = 12
MAX_TURNS = 80

NO_OPENING = "no_opening_grouped"
NO_SPAN = "no_verified_span"
NOT_READ = "page_not_read"
COPIES_PAGE = "copies_page"
CONTAMINATED = "contaminated"
POLICY_LINE = "policy_line_not_in_policy"
AS_TYPED = "line_copied_as_typed"
WRONG_SOURCE = "wrong_source"
UNKNOWN = "unknown_scenario"

SESSION_TOOLS = {"recordings": ("list_openings", "list_scenarios", "write_scenario"),
                 "ask": ("list_openings", "list_scenarios", "write_scenario"),
                 "search": ("list_scenarios", "search", "read_page", "write_scenario"),
                 "fix": ("list_scenarios", "rewrite_scenario")}


@dataclass
class MineRoot:
    """One mining session: the world, its source, its items, the web callables, what it wrote."""

    world: World
    kind: str
    items: list[dict] = field(default_factory=list)
    search: Optional[Callable[[str], list[str]]] = None
    fetch: Optional[Callable[[str], str]] = None
    corpus_url: Optional[str] = None
    exclude: tuple = ()
    found: list = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    bus: Any = None
    pages: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    written: set = field(default_factory=set)
    fixing: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)

    @property
    def workdir(self):
        return self.world.workdir


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Strict):
    pass


class QueryArgs(Strict):
    query: str


class UrlArgs(Strict):
    url: str


class WriteArgs(Strict):
    statement: str
    source: str = Field(description="The source of this session: recordings, search or ask.")
    policy_lines: list[str] = Field(default_factory=list, description="Policy lines it touches, word for word.")
    openings: list[int] = Field(default_factory=list, description="The numbered items it groups or rewrites.")
    url: str = Field(default="", description="For a page: the URL you read.")
    span: str = Field(default="", description="For a page: the span it rests on, word for word.")


class RewriteArgs(Strict):
    id: str
    statement: str


class Rows(Strict):
    summary: str
    rows: list[dict] = Field(default_factory=list)


def render(result: Rows) -> str:
    return "\n".join([result.summary, *(json.dumps(row, sort_keys=True) for row in result.rows)])


# --- refusals ------------------------------------------------------------------------------


def _refuse(root: MineRoot, anchor: Any, why: str) -> None:
    """Count one refusal of a scenario; the second drops it. Raised so the agent reads why."""
    count, _ = root.refused.get(anchor, (0, why))
    root.refused[anchor] = (count + 1, why)
    if count + 1 >= 2:
        root.dropped[why] += 1
        raise ValueError(f"refused again ({why.replace('_', ' ')}): this scenario is dropped; go on to the next")
    raise ValueError(f"refused ({why.replace('_', ' ')}): write it again once, following the rule")


def _closed(root: MineRoot, anchor: Any) -> None:
    if root.refused.get(anchor, (0, ""))[0] >= 2:
        raise ValueError("this scenario was refused twice and is dropped; go on to the next")


def settle_refusals(root: MineRoot) -> None:
    """A scenario refused once and never written again is dropped, counted under its reason."""
    for anchor, (count, why) in root.refused.items():
        if count == 1 and anchor not in root.written:
            root.dropped[why] += 1


def _policy_refusal(lines: list[str], policy: str) -> str:
    _, missing = R.policy_lines_in(lines, policy)
    return POLICY_LINE if missing else ""


# --- the write checks per source -------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _recordings(root: MineRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
    picked = sorted({at for at in args.openings if 0 <= at < len(root.items)})
    anchor = ("recordings", tuple(picked))
    if not picked:
        return anchor, NO_OPENING, {}
    source = R.Source(kind="recordings", task_ids=sorted({root.items[at]["task_id"] for at in picked}))
    return anchor, "", {"source": source, "frequency": len(picked)}


def _asked(root: MineRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
    picked = sorted({at for at in args.openings if 0 <= at < len(root.items)})
    anchor = ("ask", tuple(picked))
    if len(picked) != 1:
        return anchor, NO_OPENING, {}
    line = root.items[picked[0]]["text"]
    if _norm(statement) == _norm(line):
        return anchor, AS_TYPED, {}
    return anchor, "", {"source": R.Source(kind="ask", line=line), "frequency": 0}


def _searched(root: MineRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
    anchor = ("search", args.url, args.span)
    key = root.pages.get(args.url)
    if key is None:
        return anchor, NOT_READ, {}
    text = domain.cached_text(root.workdir, key)
    where = byte_range(text, args.span)
    if where is None:
        return anchor, NO_SPAN, {}
    if domain.copies_page({"goal": statement}, text):
        return anchor, COPIES_PAGE, {}
    if domain.contaminated({"goal": statement}, domain.corpus_strings(root.workdir)):
        return anchor, CONTAMINATED, {}
    source = R.Source(kind="search", url=args.url, key=key, start=where[0], end=where[1])
    return anchor, "", {"source": source, "frequency": 0}


CHECKS = {"recordings": _recordings, "ask": _asked, "search": _searched}


def written(root: MineRoot, args: WriteArgs) -> Rows:
    """One scenario written into this mining call's findings, or refused with why."""
    if args.source != root.kind:
        raise ValueError(f"this session mines the source {root.kind}; write with source {root.kind}")
    statement = " ".join(args.statement.split())
    anchor, why, made = CHECKS[root.kind](root, args, statement)
    _closed(root, anchor)
    why = why or (statement_refusal(statement, root.world) if statement else EMPTY)
    why = why or _policy_refusal(args.policy_lines, root.world.policy)
    if why:
        _refuse(root, anchor, why)
    record = R.born(statement, made["source"], R.policy_lines_in(args.policy_lines, root.world.policy)[0],
                    made["frequency"])
    root.found.append(record)
    root.written.add(anchor)
    row = {"id": record.id, "statement": record.statement, "source": root.kind}
    if root.bus is not None:
        root.bus.publish_custom("scenario_written", row)
    return Rows(summary=f"written {record.id}", rows=[row])


def rewritten(root: MineRoot, args: RewriteArgs) -> Rows:
    """One scenario sent back, rewritten under the rule, or refused with why."""
    record = root.fixing.get(args.id)
    if record is None:
        raise ValueError(f"{args.id!r} is not a scenario sent back in this session ({UNKNOWN})")
    anchor = ("fix", args.id)
    _closed(root, anchor)
    statement = " ".join(args.statement.split())
    why = statement_refusal(statement, root.world) if statement else EMPTY
    if why:
        _refuse(root, anchor, why)
    root.fixed[args.id] = statement
    root.written.add(anchor)
    if root.bus is not None:
        root.bus.publish_custom("scenario_rewritten", {"id": args.id, "statement": statement})
    return Rows(summary=f"rewritten {args.id}", rows=[{"id": args.id, "statement": statement}])


# --- the tools -----------------------------------------------------------------------------


def _scenario_rows(root: MineRoot) -> list[dict]:
    rows = [{"id": r.id, "statement": r.statement, "status": r.status} for r in R.load_scenarios(root.workdir)]
    return rows + [{"id": r.id, "statement": r.statement, "status": "written now"} for r in root.found]


def _read_page(root: MineRoot, url: str) -> Rows:
    crawled = domain.crawl(root.workdir, [{"url": url, "depth": 0, "via": domain.SEARCHED}], fetch=root.fetch,
                           depth=0, corpus_url=root.corpus_url, exclude=root.exclude, limit=1)
    if not crawled["pages"]:
        reason = (crawled["refused"] or [{"reason": domain.UNREACHABLE}])[0]["reason"]
        raise ValueError(f"page not read: {reason}")
    page = crawled["pages"][0]
    root.pages[url] = page["key"]
    return Rows(summary=domain.cached_text(root.workdir, page["key"])[:domain.PAGE_CHARS], rows=[{"url": url}])


def mine_tools(root: MineRoot) -> list[AgentTool]:
    """The tools of one session, the subset its source uses."""

    async def list_openings(args: NoArgs) -> Rows:
        rows = [{"number": at, "text": item["text"]} for at, item in enumerate(root.items)]
        return Rows(summary=f"{len(rows)} items", rows=rows)

    async def list_scenarios(args: NoArgs) -> Rows:
        rows = _scenario_rows(root)
        return Rows(summary=f"{len(rows)} scenarios", rows=rows)

    async def search(args: QueryArgs) -> Rows:
        urls = root.search(args.query) if root.search is not None else []
        return Rows(summary=f"{len(urls)} results", rows=[{"url": url} for url in urls])

    async def read_page(args: UrlArgs) -> Rows:
        return _read_page(root, args.url)

    async def write_scenario(args: WriteArgs) -> Rows:
        return written(root, args)

    async def rewrite_scenario(args: RewriteArgs) -> Rows:
        return rewritten(root, args)

    every = {"list_openings": ("The numbered items of this session.", NoArgs, list_openings),
             "list_scenarios": ("The scenarios already written.", NoArgs, list_scenarios),
             "search": ("The result URLs for a web search query.", QueryArgs, search),
             "read_page": ("The text of one page a search returned.", UrlArgs, read_page),
             "write_scenario": ("Write one scenario under the rule.", WriteArgs, write_scenario),
             "rewrite_scenario": ("Rewrite one scenario sent back.", RewriteArgs, rewrite_scenario)}
    return [AgentTool(name, every[name][0], every[name][1], Rows, every[name][2], render=render)
            for name in SESSION_TOOLS[root.kind]]


def mine_extension(root: MineRoot) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over the store, the session's tools, the prompt."""

    def setup(api: ExtensionAPI) -> None:
        folder = R.store_dir(root.workdir)
        folder.mkdir(parents=True, exist_ok=True)
        register_base_tools(api, folder, only=BASE_ONLY)
        for tool in mine_tools(root):
            api.register_tool(tool)
        world = root.world
        for name, text in mine_prompt.sections(SESSION_TOOLS[root.kind]):
            api.add_prompt_section(f"synth_{name}", prompt_block(name, text))
            if name == "receives":
                api.add_prompt_section("synth_environment", prompt_block(
                    "environment", prompt_mod.environment_text(world.policy, world.tools, world.description)))

    return setup


def bus_for(workdir: Any) -> Bus:
    """The workdir's one event stream, under the generator's agent name."""
    return Bus(R.store_dir(workdir).parent / "bus.jsonl", agent=AGENT)


def run(root: MineRoot, model: Any, opening: str, stage: str, ceiling: Optional[budget.Ceiling] = None,
        subscribers: tuple = ()) -> bool:
    """One session on its opening message; whether it stopped on its turn cap. Refusals settle after."""
    turns = min(MAX_TURNS, max(MIN_TURNS, TURNS_PER_ITEM * max(len(root.items), len(root.fixing), 1)))
    name = getattr(model, "name", None)
    harness = AgentHarness(model=model, max_turns=turns, context=ContextConfig(window=budget.window_for(name)),
                           bus=root.bus)
    detach = [harness.subscribe(subscriber) for subscriber in subscribers]
    detach.append(harness.subscribe(budget.subscriber(root.workdir, stage, name, ceiling)))
    try:
        load_extensions(harness, [mine_extension(root)])
        capped = session_mod._run_session(harness, opening, turns, [])
    finally:
        for unsubscribe in detach:
            unsubscribe()
    settle_refusals(root)
    return bool(capped)


__all__ = ["AGENT", "MineRoot", "SESSION_TOOLS", "bus_for", "mine_extension", "mine_tools", "rewritten", "run",
           "settle_refusals", "written"]
