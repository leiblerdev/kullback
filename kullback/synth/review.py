"""The Examiner's scenario review: a role on the shared agent harness and event stream (D373).

Set up exactly as `examiner_extension` is: base tools over a root, the role's own tools with pydantic
arguments and results carrying rows, prompt sections in GEPA order, the policy whole beside them, one
bus the Examiner's events already land on and the budget subscriber. One session per batch of open
scenarios.

The review is blind. Its root is <workdir>/scenarios/review, which holds a view per scenario and
nothing else of the store: the view carries the statement, the policy lines, the evidence the review
searched and the review so far, never the source or the frequency. That is what lets a live run mix
real asks in and measure the Examiner rather than trust it.

Outcomes are code, read off the verdict after the session: occurs no strikes the scenario with its
cite; any unsure puts it on the human queue with the open question; feasible no files an Environment
finding naming the missing tool or row class and ends it infeasible, never a drop; occurs yes and
feasible yes passes. `fix_scenario` sends a scenario back to the miner, which rewrites it once from
the same source with the reason; a second request on the same scenario ends it unsure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback import domain
from kullback.agent.base_tools import register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig, prompt_block
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import AgentTool
from kullback.examiner import findings as findings_mod
from kullback.examiner import session as session_mod
from kullback.examiner.domain_tools import find_tools
from kullback.examiner.plan import ExaminerPlan
from kullback.runner import budget
from kullback.runner.records import read_json, write_json
from kullback.spec.intent_tools import content_tokens
from kullback.spec.writer_tools import read_tools
from kullback.synth import miner
from kullback.synth import prompt as prompt_mod
from kullback.synth import records as R
from kullback.synth import search as search_mod

ROOT = "review"
VIEWS = "views"
POLICY_FILE = "policy.md"
TOOLS_FILE = "tools.json"
BASE_ONLY = ("read", "grep", "find", "ls")
STAGE = "synth_review"
FIX_STAGE = "synth_fix"
# How many scenarios one session reviews, and its turn budget per scenario and at most.
BATCH = 10
TURNS_PER_SCENARIO = 8
MAX_TURNS = 80
# How many search pages one call reads, and how many lines of each it shows.
PAGES_SHOWN = 3
LINES_SHOWN = 3
# How many times the review and fix loop goes round in one call.
PASSES = 3
# The record keys a view never carries: the review is blind to where a scenario came from.
HIDDEN = ("source", "also_from", "frequency", "fixes")

Answer = Literal["yes", "no", "unsure"]
TOOL_NAMES = prompt_mod.TOOLS


@dataclass
class ReviewRoot:
    """What one review works over: the world, the bus, the harness's web reader (or None) and the findings."""

    world: miner.World
    bus: Any = None
    engine: Any = None
    filed: list[dict] = field(default_factory=list)

    @property
    def workdir(self) -> Path:
        return self.world.workdir

    @property
    def dir(self) -> Path:
        return R.store_dir(self.workdir) / ROOT


def view(record: R.Scenario) -> dict:
    """The scenario as the review reads it: every field but where it came from and how often."""
    return {key: value for key, value in record.model_dump(mode="json").items() if key not in HIDDEN}


def write_views(root: ReviewRoot, ids: Iterable[str]) -> dict[str, str]:
    """Each scenario's view, the policy and the tool list under the review root; the root-relative paths."""
    out = {}
    for scenario_id in ids:
        path = root.dir / VIEWS / f"{scenario_id}.json"
        write_json(path, view(R.load_scenario(root.workdir, scenario_id)))
        out[scenario_id] = path.relative_to(root.dir).as_posix()
    (root.dir / POLICY_FILE).write_text(root.world.policy or "no policy recorded", encoding="utf-8")
    write_json(root.dir / TOOLS_FILE, root.world.tools)
    return out


# --- tool arguments and results ------------------------------------------------------------


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdArgs(Strict):
    id: str


class StatusArgs(Strict):
    status: str = Field(default="born", description="One of " + ", ".join(R.STATUSES) + ".")


class SearchArgs(Strict):
    id: str = Field(description="The scenario the search is evidence for.")
    query: str


class LookupArgs(Strict):
    table: str
    key: str = ""


class RowSearchArgs(Strict):
    table: str
    field: str
    value: str


class FindArgs(Strict):
    query: str


class RuleArgs(Strict):
    id: str
    occurs: Answer = Field(description="Does this happen to customers of this kind of business?")
    occurs_cite: str = Field(default="", description="A URL the search returned, Task ids, or a policy line.")
    feasible: Answer = Field(description="Can this world serve it with its tools and rows?")
    missing: str = Field(default="", description="When feasible is no: the tool or the kind of row that is missing.")
    serves_ask: Literal["yes", "no", "none"] = Field(default="none", description="Does it serve a customer's ask?")
    text: str = Field(description="Why, in one or two sentences.")


class FixArgs(Strict):
    id: str
    why: str = Field(description="Rewrite, rebind or narrow, and what to change.")


class Rows(Strict):
    summary: str
    rows: list[dict] = Field(default_factory=list)


def render(result: Rows) -> str:
    return "\n".join([result.summary, *(json.dumps(row, sort_keys=True) for row in result.rows)])


# --- the tools ----------------------------------------------------------------------------


def _record(root: ReviewRoot, scenario_id: str) -> R.Scenario:
    return R.load_scenario(root.workdir, scenario_id)


def _excerpt(text: str, query: str) -> list[str]:
    """The page lines sharing the most words with the query, best first."""
    words = content_tokens(query)
    scored = [(len(words & content_tokens(line)), line) for line in text.splitlines() if line.strip()]
    return [line[:300] for score, line in sorted(scored, key=lambda row: -row[0]) if score][:LINES_SHOWN]


def searched(root: ReviewRoot, scenario_id: str, query: str) -> Rows:
    """One search kept on the scenario as evidence: the URLs and the matching lines of the first pages."""
    record = _record(root, scenario_id)
    notes: list[str] = []
    urls: list[str] = []
    pages: list[dict] = []
    if root.engine is None:
        notes.append(search_mod.OFF_NOTE)
    else:
        search, fetch = search_mod.callables(root.engine, notes)
        urls = search(query)
        pages = domain.crawl(root.workdir, [{"url": url, "depth": 0, "via": domain.SEARCHED} for url in urls],
                             fetch=fetch, depth=0, limit=PAGES_SHOWN)["pages"]
    rows = [{"url": page["url"], "lines": _excerpt(domain.cached_text(root.workdir, page["key"]), query)}
            for page in pages]
    note = "; ".join(notes)
    evidence = {"query": query, "urls": urls, "read": [page["url"] for page in pages], "note": note}
    R.write_scenario(root.workdir, record.model_copy(update={"evidence": [*record.evidence, evidence]}))
    return Rows(summary=note or f"{len(urls)} results, {len(rows)} pages read", rows=rows)


def human_row(workdir: Any, record: R.Scenario, question: str) -> None:
    """One row per scenario on the human queue, the open question in words; a later row replaces it."""
    path = R.store_dir(workdir) / R.HUMAN
    rows = [row for row in read_json(path, []) or [] if row.get("id") != record.id]
    write_json(path, [*rows, {"id": record.id, "statement": record.statement, "question": question}])


def ruled(root: ReviewRoot, args: RuleArgs) -> Rows:
    """The verdict on the record; occurs yes without a cite and feasible no without `missing` are refused."""
    record = _record(root, args.id)
    if args.occurs == "yes" and not args.occurs_cite.strip():
        raise ValueError("occurs yes needs occurs_cite: a URL your search returned, Task ids, or a policy line")
    if args.feasible == "no" and not args.missing.strip():
        raise ValueError("feasible no needs missing: the tool or the kind of row this world lacks")
    row = {"kind": "verdict", **args.model_dump(exclude={"id"})}
    R.write_scenario(root.workdir, record.model_copy(update={"review": [*record.review, row]}))
    if root.bus is not None:
        root.bus.publish_custom("scenario_verdict", {"id": args.id, **row})
    return Rows(summary=f"verdict filed on {args.id}", rows=[row])


def sent_back(root: ReviewRoot, args: FixArgs) -> Rows:
    """The scenario sent to the miner with the reason; a second request ends it unsure, for a person."""
    record = _record(root, args.id)
    if record.fixes >= 1:
        row = {"kind": "fix_refused", "why": args.why}
        record = record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
        human_row(root.workdir, record, f"the reviewer asked for a second fix: {args.why}")
        summary = f"{args.id} was fixed once already; it goes to a person as unsure"
    else:
        row = {"kind": "fix_requested", "why": args.why}
        record = record.model_copy(update={"status": "fixing", "review": [*record.review, row]})
        summary = f"{args.id} goes back to the miner"
    R.write_scenario(root.workdir, record)
    if root.bus is not None:
        root.bus.publish_custom("synth_fix", {"id": args.id, **row})
    return Rows(summary=summary, rows=[row])


def scenario_tools(root: ReviewRoot) -> list[AgentTool]:
    """The role's own tools over one root; none edits a statement."""
    lookups = read_tools(root.world.state)

    async def read_scenario(args: IdArgs) -> Rows:
        return Rows(summary=args.id, rows=[view(_record(root, args.id))])

    async def list_scenarios(args: StatusArgs) -> Rows:
        if args.status not in R.STATUSES:
            raise ValueError(f"{args.status!r} is not a status; the statuses are {', '.join(R.STATUSES)}")
        rows = [{"id": r.id, "statement": r.statement, "status": r.status}
                for r in R.load_scenarios(root.workdir) if r.status == args.status]
        return Rows(summary=f"{len(rows)} scenarios {args.status}", rows=rows)

    async def search(args: SearchArgs) -> Rows:
        return searched(root, args.id, args.query)

    async def lookup_rows(args: LookupArgs) -> Rows:
        return Rows(summary=lookups["lookup_rows"](args.table, args.key))

    async def search_rows(args: RowSearchArgs) -> Rows:
        return Rows(summary=lookups["search_rows"](args.table, args.field, args.value))

    async def find(args: FindArgs) -> Rows:
        found = find_tools(args.query, root.world.tools)
        return Rows(summary=found.summary, rows=found.tools)

    async def rule_scenario(args: RuleArgs) -> Rows:
        return ruled(root, args)

    async def fix_scenario(args: FixArgs) -> Rows:
        return sent_back(root, args)

    def tool(name: str, text: str, args: type, run: Callable) -> AgentTool:
        return AgentTool(name, text, args, Rows, run, render=render)

    return [tool("read_scenario", "The view of one scenario.", IdArgs, read_scenario),
            tool("list_scenarios", "The scenarios with one status.", StatusArgs, list_scenarios),
            tool("search", "Search the web; the results are kept on the scenario as evidence.", SearchArgs, search),
            tool("lookup_rows", "Read rows of the world by table and key; with no key, the count and sample keys.",
                 LookupArgs, lookup_rows),
            tool("search_rows", "Find row keys whose dotted field equals a value.", RowSearchArgs, search_rows),
            tool("find_tools", "Find the tools whose name or description match a few words.", FindArgs, find),
            tool("rule_scenario", "File the verdict on one scenario.", RuleArgs, rule_scenario),
            tool("fix_scenario", "Send one scenario back to the miner with the reason.", FixArgs, fix_scenario)]


def scenario_extension(root: ReviewRoot) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over the blind root, the role's tools, the prompt."""

    def setup(api: ExtensionAPI) -> None:
        root.dir.mkdir(parents=True, exist_ok=True)
        register_base_tools(api, root.dir, only=BASE_ONLY)
        for tool in scenario_tools(root):
            api.register_tool(tool)
        for name, text in prompt_mod.sections():
            api.add_prompt_section(f"scenario_{name}", prompt_block(name, text))
            if name == "receives":
                world = root.world
                api.add_prompt_section("scenario_environment", prompt_block(
                    "environment", prompt_mod.environment_text(world.policy, world.tools, world.description)))

    return setup


# --- outcomes by code ----------------------------------------------------------------------


def last_verdict(record: R.Scenario) -> Optional[dict]:
    """The verdict filed since the scenario's statement last changed, or nothing."""
    for row in reversed(record.review):
        if row.get("kind") == "fixed":
            return None
        if row.get("kind") == "verdict":
            return row
    return None


def _question(verdict: dict) -> str:
    asked = [text for key, text in (("occurs", "does it occur here?"), ("feasible", "can this world serve it?"))
             if verdict.get(key) == "unsure"]
    return " ".join(asked) + f" The reviewer: {verdict.get('text') or ''}".rstrip()


def _environment_finding(root: ReviewRoot, record: R.Scenario, verdict: dict) -> dict:
    """The Environment finding for a scenario this world cannot serve, filed once per scenario."""
    plan = ExaminerPlan(workdir=root.workdir, inputs={})
    key = f"environment:scenario:{record.id}"
    held = [row for row in plan.store.get("findings", []) if isinstance(row, dict) and row.get("key") == key]
    if held:
        return held[0]
    missing = str(verdict.get("missing") or "").strip()
    finding = findings_mod.file_finding(
        plan, kind="environment", key=key, change=f"add {missing}",
        text=f"a scenario this world cannot serve needs {missing}: {record.statement}")
    body = finding.model_dump(mode="json")
    if root.bus is not None:
        root.bus.publish_custom("finding", body)
    root.filed.append(body)
    return body


def settle(root: ReviewRoot, record: R.Scenario) -> R.Scenario:
    """The scenario's status from its verdict, by code. Occurs no is read first: a situation that does not
    happen is no finding about the Environment."""
    verdict = last_verdict(record) if record.status == "born" else None
    if verdict is None:
        return record
    if verdict["occurs"] == "no":
        status = "struck"
    elif "unsure" in (verdict["occurs"], verdict["feasible"]):
        status = "unsure"
        human_row(root.workdir, record, _question(verdict))
    elif verdict["feasible"] == "no":
        status = "infeasible"
        _environment_finding(root, record, verdict)
    else:
        status = "passed"
    settled = record.model_copy(update={"status": status})
    R.write_scenario(root.workdir, settled)
    return settled


# --- the loop ------------------------------------------------------------------------------


def _session(root: ReviewRoot, ids: list[str], model: Any, subscribers: list[Callable],
             ceiling: Optional[budget.Ceiling]) -> bool:
    """One session over a batch; whether it stopped on its turn cap."""
    turns = min(MAX_TURNS, TURNS_PER_SCENARIO * len(ids))
    name = getattr(model, "name", None)
    harness = AgentHarness(model=model, max_turns=turns, context=ContextConfig(window=budget.window_for(name)),
                           bus=root.bus)
    detach = [harness.subscribe(subscriber) for subscriber in subscribers]
    detach.append(harness.subscribe(budget.subscriber(root.workdir, STAGE, name, ceiling)))
    try:
        paths = write_views(root, ids)
        load_extensions(harness, [scenario_extension(root)])
        lines = "\n".join(f"{scenario_id}: view {paths[scenario_id]}" for scenario_id in ids)
        entries = session_mod.root_listing(root.dir)
        capped = session_mod._run_session(harness, prompt_mod.opening(lines, entries), turns, ids)
    finally:
        for unsubscribe in detach:
            unsubscribe()
    return bool(capped)


def _fix_all(root: ReviewRoot, fixer: Any) -> list[str]:
    """Every scenario sent back, rewritten once by the miner; the ids that came back born."""
    out = []
    for record in R.load_scenarios(root.workdir):
        if record.status != "fixing":
            continue
        why = next((row["why"] for row in reversed(record.review) if row.get("kind") == "fix_requested"), "")
        fixed = miner.fix(root.world, record, why, fixer)
        if fixed.status == "unsure":
            human_row(root.workdir, fixed, f"the miner could not rewrite it: {why}")
        R.write_scenario(root.workdir, fixed)
        out += [fixed.id] if fixed.status == "born" else []
    return out


def _over(ceiling: Optional[budget.Ceiling]) -> bool:
    return ceiling is not None and ceiling.remaining() <= 0


def review(workdir: Any, *, model: Any, fixer: Any = None, batch: int = BATCH, engine: Any = None,
           env: Optional[dict] = None, ceiling: Optional[budget.Ceiling] = None,
           subscribers: Iterable[Callable] = (), description: str = "") -> dict:
    """Review every born scenario in batches, fix what was sent back and review it again, then settle.

    The first pass reviews every born scenario; each later pass reviews only what the miner rewrote in
    this call, at most PASSES passes. A scenario no session ruled stays born for the next call.
    """
    world = miner.world_of(workdir, description)
    root = ReviewRoot(world=world, bus=Bus(Path(workdir) / "bus.jsonl", agent="examiner"),
                      engine=engine if engine is not None else search_mod.reader(workdir, env))
    watchers = list(subscribers)
    todo = [record.id for record in R.load_scenarios(workdir) if record.status == "born"]
    capped: list[str] = []
    stopped = False
    for _ in range(PASSES):
        for start in range(0, len(todo), max(int(batch), 1)):
            if _over(ceiling):
                stopped = True
                break
            ids = todo[start:start + max(int(batch), 1)]
            capped += ids if _session(root, ids, model, watchers, ceiling) else []
            for scenario_id in ids:
                settle(root, R.load_scenario(workdir, scenario_id))
        todo = [] if stopped else _fix_all(root, fixer if fixer is not None else model)
        if not todo:
            break
    return {"findings": root.filed, "capped": sorted(set(capped)), "stopped_on_ceiling": stopped}


__all__ = ["BATCH", "HIDDEN", "ReviewRoot", "TOOL_NAMES", "last_verdict", "review", "scenario_extension",
           "scenario_tools", "settle", "view", "write_views"]
