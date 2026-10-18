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
naming the other's id; occurs no strikes it. The web tools are the shared ones (D377). A cite is a page
this session read with read_page and a span found in that page's cached text; a search result alone is
no cite and a policy line is no cite, and code refuses anything else, so no trace id ever reaches a
verdict (D382). The verdict writes the cite's evidence row on its scenario alone. feasible yes names
the tools and the rows it checked, each found in the world by code (D401), and is refused while the scenario's
requires is open. Then any unsure puts it on the human queue with the open question; feasible no files
an Environment finding carrying the scenario's requires and binding, the order to build, and ends it
infeasible, never a drop; occurs yes and feasible yes passes. `fix_scenario` sends a scenario back to
the generator with the words it quotes, why and the change; the generator rewrites it once from the
same source, and a second request on the same scenario ends it unsure.
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
from kullback.agent.web_tools import register_web_tools
from kullback.examiner import findings as findings_mod
from kullback.examiner import session as session_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.runner import budget
from kullback.runner.records import read_json, write_json
from kullback.synth import generate
from kullback.synth import prompt as prompt_mod
from kullback.synth import records as R
from kullback.synth import search as search_mod
from kullback.synth.rule import World, byte_range, nearest_line, world_of
from kullback.synth.world_tools import world_tools

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
HIDDEN = ("source", "also_from", "frequency", "fixes", "situation")

Answer = Literal["yes", "no", "unsure"]
TOOL_NAMES = prompt_mod.TOOLS


@dataclass
class ReviewRoot:
    """What one review works over: the world, the bus, the web reader (search.Crawled, or None), the
    findings, and the pages the session running now read with read_page (URL to cache key), which are
    the only pages a verdict may cite."""

    world: World
    bus: Any = None
    reader: Any = None
    refuse: Callable[[str], str] = domain.destination_refusal
    filed: list[dict] = field(default_factory=list)
    read: dict[str, str] = field(default_factory=dict)

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


class RuleArgs(Strict):
    id: str
    occurs: Answer = Field(description="Does this happen to customers of this kind of business?")
    occurs_url: str = Field(default="", description="A URL you read with read_page in this session.")
    occurs_span: str = Field(default="", description="The words of that page that show it, word for word.")
    feasible: Answer = Field(description="Can this world serve it with its tools and rows?")
    checked: list[str] = Field(default_factory=list, description=(
        'For feasible yes: "tool:<name>" for each tool it needs, "row:<table>:<key>" for each row you looked up.'))
    missing: str = Field(default="", description="When feasible is no: the tool or the kind of row that is missing.")
    repeat_of: str = Field(default="", description="The id of another scenario this one says again; it ends struck.")
    text: str = Field(description="Why, in one or two sentences.")


class FixArgs(Strict):
    id: str
    quote: str = Field(description="The words of the scenario that are wrong, copied exactly from its statement, ask, "
                                   "end or requires.")
    why: str = Field(description="Why those words are wrong.")
    change: str = Field(description="What they should say instead, or what to look up.")


class Rows(Strict):
    summary: str
    rows: list[dict] = Field(default_factory=list)


def render(result: Rows) -> str:
    return "\n".join([result.summary, *(json.dumps(row, sort_keys=True) for row in result.rows)])


# --- the tools ----------------------------------------------------------------------------


def _record(root: ReviewRoot, scenario_id: str) -> R.Scenario:
    return R.load_scenario(root.workdir, scenario_id)


def kept(root: ReviewRoot, name: str, args: dict, result: Any) -> None:
    """The shared web tools' on_call: a page read with read_page becomes citable in this session; a search
    is not kept."""
    url = str(args.get("url") or "")
    keys = getattr(root.reader, "keys", {}) or {}
    if name == "read_page" and result.rows and url in keys:
        root.read[url] = keys[url]


def human_row(workdir: Any, record: R.Scenario, question: str) -> None:
    """One row per scenario on the human queue, the open question in words; a later row replaces it."""
    path = R.store_dir(workdir) / R.HUMAN
    rows = [row for row in read_json(path, []) or [] if row.get("id") != record.id]
    write_json(path, [*rows, {"id": record.id, "statement": record.statement, "question": question}])


CITE = "occurs_url, a page you read with read_page in this session, and occurs_span, its words copied exactly"
NO_POLICY_CITE = ("a policy line is no cite: it shows the business anticipated the situation, not that customers "
                  "bring it")


def cited(root: ReviewRoot, args: RuleArgs) -> Optional[dict]:
    """The evidence row of a verdict's cite, checked by code, or nothing when it cites nothing.

    The URL must be a page this session read with read_page, and the span must be in that page's cached
    text under the key the reader kept; a search result alone is no cite, and a policy line is no cite.
    """
    url, span = args.occurs_url.strip(), args.occurs_span
    if not url and not span.strip():
        return None
    if span.strip() and span.strip() in (root.world.policy or "") and url not in root.read:
        raise ValueError(f"{NO_POLICY_CITE}; cite {CITE}")
    if url not in root.read:
        raise ValueError(f"{url or 'no URL'} was not read with read_page in this session; a search result alone "
                         f"is no cite: read the page, then cite {CITE}")
    text = domain.cached_text(root.workdir, root.read[url])
    where = byte_range(text, span) if span.strip() else None
    if where is None:
        near = nearest_line(span, text.splitlines())
        raise ValueError(f"the span is not on the page you read at {url}; copy its words exactly"
                         + (f"; the nearest line of the page is \"{near}\"" if near else ""))
    return {"tool": "read_page", "url": url, "span": span, "start": where[0], "end": where[1]}


def _checked(root: ReviewRoot, item: str) -> bool:
    """Whether one checked entry is a tool of this world, or a row the world holds, bound or not (D401)."""
    kind, _, rest = str(item).partition(":")
    if kind == "tool":
        return rest in {str(tool.get("name")) for tool in root.world.tools}
    table, _, key = rest.partition(":")
    rows = root.world.state.get(table)
    return kind == "row" and bool(table) and isinstance(rows, dict) and key in rows


def _feasible(root: ReviewRoot, record: R.Scenario, args: RuleArgs) -> None:
    """feasible no needs missing; feasible yes needs checked entries code finds, and no requires left open."""
    if args.feasible == "no" and not args.missing.strip():
        raise ValueError("feasible no needs missing: the tool or the kind of row this world lacks")
    if args.feasible != "yes":
        return
    if record.requires:
        raise ValueError(f"this scenario requires what the world lacks ({'; '.join(record.requires)}): rule "
                         "feasible no with missing, or send it back if the world holds what requires names")
    if not args.checked:
        raise ValueError('feasible yes needs checked: "tool:<name>" for each tool it needs and '
                         '"row:<table>:<key>" for each row you looked up')
    bad = [item for item in args.checked if not _checked(root, item)]
    if bad:
        raise ValueError(f"checked names what this world does not hold: {', '.join(bad)}; look it up with "
                         "lookup_rows or find_tools and check only what they return")


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
    return _filed(root, record, {"kind": "verdict", **args.model_dump(exclude={"id"})})


def ruled(root: ReviewRoot, args: RuleArgs) -> Rows:
    """The verdict on the record, its cite's evidence row on this scenario alone; refused with why where the
    cite, the checked entries or `missing` do not hold."""
    record = _record(root, args.id)
    if args.repeat_of:
        return _repeat(root, record, args)
    evidence = cited(root, args)
    if args.occurs == "yes" and evidence is None:
        raise ValueError(f"occurs yes needs {CITE}")
    _feasible(root, record, args)
    if evidence is not None:
        record = record.model_copy(update={"evidence": [*record.evidence, evidence]})
    return _filed(root, record, {"kind": "verdict", **args.model_dump(exclude={"id"})})


def reason(row: dict) -> str:
    """A send back as the fix session reads it: the quoted words, why, and what to change."""
    return f"quote: \"{row.get('quote', '')}\"\nwhy: {row.get('why', '')}\nchange: {row.get('change', '')}"


def sent_back(root: ReviewRoot, args: FixArgs) -> Rows:
    """The scenario sent to the generator with the quoted words, why and the change; a second request ends it
    unsure, for a person. A quote no written part of it holds (statement, ask, end, requires) is refused (D400)."""
    record = _record(root, args.id)
    written = [record.statement, record.ask, record.end, *record.requires]
    if not args.quote.strip() or not any(args.quote in part for part in written):
        raise ValueError("quote is not in the scenario: copy the wrong words of its statement, ask, end or "
                         "requires exactly")
    asked = {"quote": args.quote, "why": args.why, "change": args.change}
    if record.fixes >= 1:
        row = {"kind": "fix_refused", **asked}
        record = record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
        human_row(root.workdir, record, f"the reviewer asked for a second fix: {args.why}")
        summary = f"{args.id} was fixed once already; it goes to a person as unsure"
    else:
        row = {"kind": "fix_requested", **asked}
        record = record.model_copy(update={"status": "fixing", "review": [*record.review, row]})
        summary = f"{args.id} goes back to the generator"
    R.write_scenario(root.workdir, record)
    if root.bus is not None:
        root.bus.publish_custom("synth_fix", {"id": args.id, **row})
    return Rows(summary=summary, rows=[row])


def scenario_tools(root: ReviewRoot) -> list[AgentTool]:
    """The role's own tools over one root and the shared world tools; none edits a statement."""

    async def read_scenario(args: IdArgs) -> Rows:
        return Rows(summary=args.id, rows=[view(_record(root, args.id))])

    async def list_scenarios(args: StatusArgs) -> Rows:
        if args.status not in R.STATUSES:
            raise ValueError(f"{args.status!r} is not a status; the statuses are {', '.join(R.STATUSES)}")
        rows = [{"id": r.id, "statement": r.statement, "status": r.status}
                for r in R.load_scenarios(root.workdir) if r.status == args.status]
        return Rows(summary=f"{len(rows)} scenarios {args.status}", rows=rows)

    async def rule_scenario(args: RuleArgs) -> Rows:
        return ruled(root, args)

    async def fix_scenario(args: FixArgs) -> Rows:
        return sent_back(root, args)

    def tool(name: str, text: str, args: type, run: Callable) -> AgentTool:
        return AgentTool(name, text, args, Rows, run, render=render)

    return [tool("read_scenario", "The view of one scenario.", IdArgs, read_scenario),
            tool("list_scenarios", "The scenarios with one status.", StatusArgs, list_scenarios),
            *world_tools(root.world),
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
    order = {"kind": "world", "requires": list(record.requires), "binding": dict(record.binding),
             "why": f"a scenario this world cannot serve needs {missing}"}
    finding = findings_mod.file_finding(
        plan, kind="environment", key=key, change=f"add {missing}", edits=[order],
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
    root.read = {}
    try:
        paths = write_views(root, ids)
        load_extensions(harness, [scenario_extension(root)])
        lines = "\n".join(f"{scenario_id}: view {paths[scenario_id]}" for scenario_id in ids)
        entries = session_mod.root_listing(root.dir)
        capped = session_mod._run_session(harness, prompt_mod.opening(lines, entries), turns, ids)
    finally:
        for unsubscribe in detach:
            unsubscribe()
        root.read = {}
    return bool(capped)


def _fix_all(root: ReviewRoot, fixer: Any, ceiling: Optional[budget.Ceiling] = None,
             subscribers: Iterable[Callable] = ()) -> list[str]:
    """Every scenario sent back, rewritten once by the generator in one session; the ids that came back born."""
    sent = []
    for record in R.load_scenarios(root.workdir):
        if record.status == "fixing":
            why = next((reason(row) for row in reversed(record.review) if row.get("kind") == "fix_requested"), "")
            sent.append((record, why))
    out = []
    for fixed, (_, why) in zip(generate.fix(root.world, sent, fixer, ceiling=ceiling, subscribers=subscribers), sent, strict=True):
        if fixed.status == "unsure":
            human_row(root.workdir, fixed, f"the generator could not rewrite it: {why}")
        R.write_scenario(root.workdir, fixed)
        out += [fixed.id] if fixed.status == "born" else []
    return out


def _over(ceiling: Optional[budget.Ceiling]) -> bool:
    return ceiling is not None and ceiling.remaining <= 0


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
