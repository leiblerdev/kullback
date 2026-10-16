"""The Examiner's scenario review: a role on the shared agent harness and event stream (D373).

Set up exactly as `examiner_extension` is: base tools over a root, the role's own tools with pydantic
arguments and results carrying rows, prompt sections in GEPA order, the policy whole beside them, one
bus the Examiner's events already land on and the budget subscriber. One session per batch of open
scenarios.

The review is blind. Its root is <workdir>/scenarios/review, which holds a view per scenario and
nothing else of the store: the view carries the statement, the policy lines, the evidence the review
searched and the review so far, never the source or the frequency. That is what lets a live run mix
real asks in and measure the Examiner rather than trust it.

Outcomes are code, read off the verdict after the session: a repeat of another scenario strikes it,
citing the other's id; occurs no strikes the scenario with its cite. The web tools are the shared ones
(D377): every search and read_page call of a session is kept as evidence on each scenario that session
reviews. A cite is a URL a search kept on the scenario returned, or a line of the policy; code refuses
anything else, so no trace id ever reaches a verdict. Then any unsure puts it on the human queue with the open question; feasible no
files an Environment finding naming the missing tool or row class and ends it infeasible, never a drop;
occurs yes and feasible yes passes. `fix_scenario` sends a scenario back to the generator, which
rewrites it once from the same source with the reason; a second request on the same scenario ends it
unsure.
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
from kullback.agent.web_tools import OFF as WEB_OFF
from kullback.agent.web_tools import register_web_tools
from kullback.examiner import findings as findings_mod
from kullback.examiner import session as session_mod
from kullback.examiner.domain_tools import find_tools
from kullback.examiner.plan import ExaminerPlan
from kullback.runner import budget
from kullback.runner.records import read_json, write_json
from kullback.spec.writer_tools import read_tools
from kullback.synth import miner
from kullback.synth import prompt as prompt_mod
from kullback.synth import records as R
from kullback.synth import search as search_mod
from kullback.synth.rule import World, world_of

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
# How many times the review and fix loop goes round in one call.
PASSES = 3
# The record keys a view never carries: the review is blind to where a scenario came from.
HIDDEN = ("source", "also_from", "frequency", "fixes")

Answer = Literal["yes", "no", "unsure"]
TOOL_NAMES = prompt_mod.TOOLS


@dataclass
class ReviewRoot:
    """What one review works over: the world, the bus, the web reader (search.Crawled, or None), the
    findings, and the scenarios of the session running now, which its web calls are kept on."""

    world: World
    bus: Any = None
    reader: Any = None
    refuse: Callable[[str], str] = domain.destination_refusal
    filed: list[dict] = field(default_factory=list)
    current: list[str] = field(default_factory=list)

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
    occurs_cite: str = Field(default="", description="A URL your search returned for this scenario, or a policy line.")
    feasible: Answer = Field(description="Can this world serve it with its tools and rows?")
    missing: str = Field(default="", description="When feasible is no: the tool or the kind of row that is missing.")
    serves_ask: Literal["yes", "no", "none"] = Field(default="none", description="Does it serve a customer's ask?")
    repeat_of: str = Field(default="", description="The id of another scenario this one says again; it ends struck.")
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


def evidence_row(name: str, args: dict, result: Any) -> dict:
    """One shared web call as evidence: the URLs a search returned, or the URL a page read."""
    off = result.summary == WEB_OFF
    note = search_mod.OFF_NOTE if off else ""
    if name == "search":
        return {"tool": name, "query": args.get("query", ""), "urls": [row["url"] for row in result.rows],
                "read": [], "note": note}
    read = [args.get("url", "")] if result.rows else []
    return {"tool": name, "query": "", "urls": [], "read": read, "note": note or ("" if read else result.summary)}


def kept(root: ReviewRoot, name: str, args: dict, result: Any) -> None:
    """The shared web tools' on_call: the call kept on every scenario the session reviews."""
    row = evidence_row(name, args, result)
    for scenario_id in root.current:
        record = _record(root, scenario_id)
        R.write_scenario(root.workdir, record.model_copy(update={"evidence": [*record.evidence, row]}))


def human_row(workdir: Any, record: R.Scenario, question: str) -> None:
    """One row per scenario on the human queue, the open question in words; a later row replaces it."""
    path = R.store_dir(workdir) / R.HUMAN
    rows = [row for row in read_json(path, []) or [] if row.get("id") != record.id]
    write_json(path, [*rows, {"id": record.id, "statement": record.statement, "question": question}])


CITE = "a URL your own search returned for this scenario, or a line of the policy word for word"


def cited(root: ReviewRoot, record: R.Scenario, cite: str) -> bool:
    """Whether a cite is a URL a search kept on this scenario returned, or a substring of the policy."""
    text = cite.strip()
    returned = {url for row in record.evidence for url in row.get("urls") or ()}
    return text in returned or (bool(text) and text in (root.world.policy or ""))


def _filed(root: ReviewRoot, record: R.Scenario, row: dict) -> Rows:
    R.write_scenario(root.workdir, record.model_copy(update={"review": [*record.review, row]}))
    if root.bus is not None:
        root.bus.publish_custom("scenario_verdict", {"id": record.id, **row})
    return Rows(summary=f"verdict filed on {record.id}", rows=[row])


def _repeat(root: ReviewRoot, record: R.Scenario, args: RuleArgs) -> Rows:
    """A verdict that this scenario says again what another says; the other must be a different scenario."""
    if args.repeat_of == record.id:
        raise ValueError("repeat_of names this scenario itself; name the other scenario it repeats")
    R.load_scenario(root.workdir, args.repeat_of)
    row = {"kind": "verdict", **args.model_dump(exclude={"id"}), "occurs_cite": args.repeat_of}
    return _filed(root, record, row)


def ruled(root: ReviewRoot, args: RuleArgs) -> Rows:
    """The verdict on the record; occurs yes without a cite and feasible no without `missing` are refused."""
    record = _record(root, args.id)
    if args.repeat_of:
        return _repeat(root, record, args)
    if args.occurs == "yes" and not args.occurs_cite.strip():
        raise ValueError(f"occurs yes needs occurs_cite: {CITE}")
    if args.occurs_cite.strip() and not cited(root, record, args.occurs_cite):
        raise ValueError(f"occurs_cite is neither of what a cite may be: {CITE}")
    if args.feasible == "no" and not args.missing.strip():
        raise ValueError("feasible no needs missing: the tool or the kind of row this world lacks")
    return _filed(root, record, {"kind": "verdict", **args.model_dump(exclude={"id"})})


def sent_back(root: ReviewRoot, args: FixArgs) -> Rows:
    """The scenario sent to the generator with the reason; a second request ends it unsure, for a person."""
    record = _record(root, args.id)
    if record.fixes >= 1:
        row = {"kind": "fix_refused", "why": args.why}
        record = record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
        human_row(root.workdir, record, f"the reviewer asked for a second fix: {args.why}")
        summary = f"{args.id} was fixed once already; it goes to a person as unsure"
    else:
        row = {"kind": "fix_requested", "why": args.why}
        record = record.model_copy(update={"status": "fixing", "review": [*record.review, row]})
        summary = f"{args.id} goes back to the generator"
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
            tool("lookup_rows", "Read rows of the world by table and key; with no key, the count and sample keys.",
                 LookupArgs, lookup_rows),
            tool("search_rows", "Find row keys whose dotted field equals a value.", RowSearchArgs, search_rows),
            tool("find_tools", "Find the tools whose name or description match a few words.", FindArgs, find),
            tool("rule_scenario", "File the verdict on one scenario.", RuleArgs, rule_scenario),
            tool("fix_scenario", "Send one scenario back to the generator with the reason.", FixArgs, fix_scenario)]


def scenario_extension(root: ReviewRoot) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over the blind root, the role's tools, the shared web tools
    with their calls kept as evidence, the prompt."""

    def setup(api: ExtensionAPI) -> None:
        root.dir.mkdir(parents=True, exist_ok=True)
        register_base_tools(api, root.dir, only=BASE_ONLY)
        for tool in scenario_tools(root):
            api.register_tool(tool)
        register_web_tools(api, search=root.reader, refuse=root.refuse,
                           on_call=lambda name, args, result: kept(root, name, args, result))
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
    """The scenario's status from its verdict, by code. A repeat and occurs no are read first: a situation
    said twice or that does not happen is no finding about the Environment."""
    verdict = last_verdict(record) if record.status == "born" else None
    if verdict is None:
        return record
    if verdict.get("repeat_of") or verdict["occurs"] == "no":
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
    root.current = list(ids)
    try:
        paths = write_views(root, ids)
        load_extensions(harness, [scenario_extension(root)])
        lines = "\n".join(f"{scenario_id}: view {paths[scenario_id]}" for scenario_id in ids)
        entries = session_mod.root_listing(root.dir)
        capped = session_mod._run_session(harness, prompt_mod.opening(lines, entries), turns, ids)
    finally:
        for unsubscribe in detach:
            unsubscribe()
        root.current = []
    return bool(capped)


def _fix_all(root: ReviewRoot, fixer: Any, ceiling: Optional[budget.Ceiling] = None,
             subscribers: Iterable[Callable] = ()) -> list[str]:
    """Every scenario sent back, rewritten once by the generator in one session; the ids that came back born."""
    sent = []
    for record in R.load_scenarios(root.workdir):
        if record.status == "fixing":
            why = next((row["why"] for row in reversed(record.review) if row.get("kind") == "fix_requested"), "")
            sent.append((record, why))
    out = []
    for fixed, (_, why) in zip(miner.fix(root.world, sent, fixer, ceiling=ceiling, subscribers=subscribers), sent, strict=True):
        if fixed.status == "unsure":
            human_row(root.workdir, fixed, f"the generator could not rewrite it: {why}")
        R.write_scenario(root.workdir, fixed)
        out += [fixed.id] if fixed.status == "born" else []
    return out


def _over(ceiling: Optional[budget.Ceiling]) -> bool:
    return ceiling is not None and ceiling.remaining() <= 0


def review(workdir: Any, *, model: Any, fixer: Any = None, batch: int = BATCH, engine: Any = None,
           env: Optional[dict] = None, ceiling: Optional[budget.Ceiling] = None,
           subscribers: Iterable[Callable] = (), description: str = "",
           refuse: Optional[Callable[[str], str]] = None) -> dict:
    """Review every born scenario in batches, fix what was sent back and review it again, then settle.

    The first pass reviews every born scenario; each later pass reviews only what the generator rewrote in
    this call, at most PASSES passes. A scenario no session ruled stays born for the next call.
    """
    world = world_of(workdir, description)
    engine = engine if engine is not None else search_mod.reader(workdir, env)
    root = ReviewRoot(world=world, bus=Bus(Path(workdir) / "bus.jsonl", agent="examiner"),
                      reader=search_mod.Crawled(engine, workdir) if engine is not None else None,
                      refuse=refuse or domain.destination_refusal)
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
        todo = [] if stopped else _fix_all(root, fixer if fixer is not None else model, ceiling, watchers)
        if not todo:
            break
    return {"findings": root.filed, "capped": sorted(set(capped)), "stopped_on_ceiling": stopped}


__all__ = ["BATCH", "HIDDEN", "ReviewRoot", "TOOL_NAMES", "last_verdict", "review", "scenario_extension",
           "scenario_tools", "settle", "view", "write_views"]
