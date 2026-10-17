"""The scenario generator session: a role on the shared agent harness and event stream (D372).

Set up as the Examiner's sessions and the scenario review are: base tools over the scenario store, the
role's own tools with pydantic arguments and results carrying rows, prompt sections in GEPA order, the
policy whole beside them, the workdir bus under the agent name synth and the budget subscriber under a
synth stage. One session per batch of one source.

The rule lives in the tools. write_scenario and rewrite_scenario refuse a statement that names a tool
of the Environment or carries an id shape, a policy line the policy does not hold, a span the page does
not hold, a page the session never read, a typed line copied as typed, a statement that nearly repeats
one already written (D383), and say why as the tool result,
so the agent writes again. A second refusal of the same scenario drops it, and a scenario refused once
and never written again is dropped at the end of the session; both are counted per reason. An opening
ask or a typed line belongs to one scenario per session: a write picking one already grouped is refused
naming the scenario that holds it, and is no refusal of a new scenario.

A scenario also carries the customer's opening words (ask) and what a correct agent leaves behind (end),
apart (D393). The write and the rewrite refuse, by code, an ask and an end that share a sentence, a tool
name in either, and one of the pair given without the other; the reason names which.
"""

from __future__ import annotations

import json
import re
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
from kullback.agent.web_tools import TOOL_NAMES as WEB_TOOLS
from kullback.agent.web_tools import register_web_tools
from kullback.examiner import session as session_mod
from kullback.runner import budget
from kullback.synth import generate_prompt
from kullback.synth import plan as plan_mod
from kullback.synth import prompt as prompt_mod
from kullback.synth import records as R
from kullback.synth.rule import EMPTY, World, byte_range, rouge_l, statement_refusal
from kullback.synth.world_tools import NAMES as WORLD_TOOLS
from kullback.synth.world_tools import binding_refusal, world_tools

AGENT = "synth"
BASE_ONLY = ("read", "grep", "find", "ls")
# The turn budget of one session: per item, at least and at most; six per item since a bound scenario
# looks its rows up before it is written (D381). The search session has no items, so it
# has its own budget: a list, some searches, some reads and the writes, for the whole web.
TURNS_PER_ITEM = 6
MIN_TURNS = 12
MAX_TURNS = 160
SEARCH_TURNS = 60

NO_OPENING = "no_opening_grouped"
NO_SPAN = "no_verified_span"
NOT_READ = "page_not_read"
COPIES_PAGE = "copies_page"
POLICY_LINE = "policy_line_not_in_policy"
AS_TYPED = "line_copied_as_typed"
WRONG_SOURCE = "wrong_source"
OPENING_GROUPED = "opening_already_grouped"
UNKNOWN = "unknown_scenario"
BINDING = "binding_not_in_world"
UNBOUND = "unbound"
SITUATION_WRITTEN = "situation_already_written"
ASK_SHARES_END = "ask_shares_end"
TOOL_IN_ASK = "tool_name_in_ask"
TOOL_IN_END = "tool_name_in_end"
ASK_WITHOUT_END = "ask_without_end"
END_WITHOUT_ASK = "end_without_ask"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
NEAR = "near_repeat"
# A statement at or above this ROUGE-L against any written one is a near repeat, fixed before any run (D383).
NEAR_REPEAT = 0.7
# The sources whose scenarios must be bound to rows or say what the world lacks.
BOUND_KINDS = ("recordings", "search")
# How many opening texts a situation item shows; the rest are counted.
OPENINGS_SHOWN = 5

SESSION_TOOLS = {"plan": ("list_openings", "list_plan", "write_situation", "attach_openings", "skip_opening"),
                 "recordings": ("list_openings", "list_scenarios", *WORLD_TOOLS, "write_scenario"),
                 "ask": ("list_openings", "list_scenarios", *WORLD_TOOLS, "write_scenario"),
                 "search": ("list_scenarios", "search", "read_page", *WORLD_TOOLS, "write_scenario"),
                 "fix": ("list_scenarios", *WORLD_TOOLS, "rewrite_scenario")}


@dataclass
class GenerateRoot:
    """One generation session: the world, its source, its items, the web reader (or None), what it wrote.

    `reader` is search.Crawled over the harness's reader, registered as the shared web tools' search; the
    cache key of every page it reads lands in `pages`, which the search write checks a span against.
    """

    world: World
    kind: str
    items: list[dict] = field(default_factory=list)
    reader: Any = None
    refuse: Callable[[str], str] = domain.destination_refusal
    found: list = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    bus: Any = None
    pages: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    written: set = field(default_factory=set)
    fixing: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    grouped: dict[int, str] = field(default_factory=dict)
    plan: plan_mod.Plan = field(default_factory=plan_mod.Plan)

    @property
    def workdir(self):
        return self.world.workdir


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Strict):
    pass


class WriteArgs(Strict):
    statement: str
    source: str = Field(description="The source of this session: recordings, search or ask.")
    policy_lines: list[str] = Field(default_factory=list, description="Policy lines it touches, word for word.")
    openings: list[int] = Field(default_factory=list, description="For typed lines: the one number it rewrites.")
    url: str = Field(default="", description="For a page: the URL you read.")
    span: str = Field(default="", description="For a page: the span it rests on, word for word.")
    situation: str = Field(default="", description="For a situation of the plan: its si_ id.")
    variant: str = Field(default="", description="The variant line it writes, word for word, or empty for the situation.")
    binding: dict[str, str] = Field(default_factory=dict,
                                    description="The rows it is bound to: table to row key, as lookup_rows names them.")
    requires: list[str] = Field(default_factory=list, description="What the world lacks for it, in words.")
    ask: str = Field(default="", description="What the customer says to open, first person, never the outcome.")
    end: str = Field(default="", description="What a correct agent leaves the world and the customer with.")


class SituationArgs(Strict):
    statement: str = Field(description="One sentence in general words: one thing that happens.")
    kind: str = Field(description="One word: change, cancel, return, exchange, question, complaint, refusal or your own.")
    openings: list[int] = Field(default_factory=list, description="The numbered items that show it.")
    variants: list[str] = Field(default_factory=list, description="One line each: a variant that changes what the agent must do.")


class AttachArgs(Strict):
    id: str = Field(description="The si_ id of a situation of the plan.")
    openings: list[int]


class SkipArgs(Strict):
    openings: list[int]
    why: str = Field(description="Why these items are no opening ask.")


class RewriteArgs(Strict):
    id: str
    statement: str
    binding: Optional[dict[str, str]] = Field(default=None, description="The rows it is bound to; leave out to keep them.")
    requires: Optional[list[str]] = Field(default=None, description="What the world lacks; leave out to keep it.")
    ask: Optional[str] = Field(default=None, description="The customer's opening words; leave out to keep them.")
    end: Optional[str] = Field(default=None, description="What a correct agent leaves behind; leave out to keep it.")


class Rows(Strict):
    summary: str
    rows: list[dict] = Field(default_factory=list)


def render(result: Rows) -> str:
    return "\n".join([result.summary, *(json.dumps(row, sort_keys=True) for row in result.rows)])


# --- refusals ------------------------------------------------------------------------------


def _refuse(root: GenerateRoot, anchor: Any, why: str, detail: str = "") -> None:
    """Count one refusal of a scenario; the second drops it. Raised so the agent reads why."""
    count, _ = root.refused.get(anchor, (0, why))
    root.refused[anchor] = (count + 1, why)
    named = why.replace("_", " ") + (f": {detail}" if detail else "")
    if count + 1 >= 2:
        root.dropped[why] += 1
        raise ValueError(f"refused again ({named}): this scenario is dropped; go on to the next")
    raise ValueError(f"refused ({named}): write it again once, following the rule")


def _closed(root: GenerateRoot, anchor: Any) -> None:
    if root.refused.get(anchor, (0, ""))[0] >= 2:
        raise ValueError("this scenario was refused twice and is dropped; go on to the next")


def settle_refusals(root: GenerateRoot) -> None:
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


def _one_line(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


def _requires(lines: Optional[list[str]]) -> list[str]:
    return [" ".join(str(line).split()) for line in lines or () if str(line).strip()]


def _holder(root: GenerateRoot, situation: str, variant: str) -> str:
    """The scenario already written for this situation and variant, in this session or the store, or nothing."""
    held = [*root.found, *R.load_scenarios(root.workdir)]
    return next((r.id for r in held if r.situation == situation and r.variant == variant), "")


def _situation_item(root: GenerateRoot, args: WriteArgs) -> tuple[int, dict, str]:
    """The session's item for the situation named, and the variant line, or refused with why."""
    at = next((at for at, item in enumerate(root.items) if item["situation"] == args.situation), None)
    if at is None:
        raise ValueError(f"{args.situation!r} is not a situation of this session ({UNKNOWN}); list_openings names them")
    variant = " ".join(args.variant.split())
    if variant and variant not in root.items[at]["variants"]:
        raise ValueError(f"{variant!r} is not a variant of {args.situation}; copy it word for word, or leave it empty")
    holder = _holder(root, args.situation, variant)
    if holder:
        raise ValueError(f"refused ({SITUATION_WRITTEN.replace('_', ' ')}): {holder} already holds this situation"
                         f"{' and variant' if variant else ''}; write the next one")
    return at, root.items[at], variant


def _recordings(root: GenerateRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
    """A scenario of one situation of the plan; its Tasks and frequency come from the plan, never the model."""
    at, item, variant = _situation_item(root, args)
    source = R.Source(kind="recordings", task_ids=sorted({row["task_id"] for row in item["openings"]}))
    return ("recordings", item["situation"], variant), "", {
        "source": source, "frequency": len(item["openings"]), "picked": [at], "situation": item["situation"],
        "variant": variant}


def _asked(root: GenerateRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
    picked = sorted({at for at in args.openings if 0 <= at < len(root.items)})
    anchor = ("ask", tuple(picked))
    if len(picked) != 1:
        return anchor, NO_OPENING, {}
    line = root.items[picked[0]]["text"]
    if _norm(statement) == _norm(line):
        return anchor, AS_TYPED, {}
    return anchor, "", {"source": R.Source(kind="ask", line=line), "frequency": 0, "picked": picked}


def _searched(root: GenerateRoot, args: WriteArgs, statement: str) -> tuple[Any, str, dict]:
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
    source = R.Source(kind="search", url=args.url, key=key, start=where[0], end=where[1])
    return anchor, "", {"source": source, "frequency": 0}


CHECKS = {"recordings": _recordings, "ask": _asked, "search": _searched}


def _already_grouped(root: GenerateRoot, args: WriteArgs) -> None:
    """Refuse a write picking an item a scenario of this session already holds; no anchor is counted."""
    held = sorted({at for at in args.openings if at in root.grouped}) if root.kind == "ask" else []
    if held:
        named = ", ".join(f"{at} (in {root.grouped[at]})" for at in held)
        raise ValueError(f"refused ({OPENING_GROUPED.replace('_', ' ')}): item {named} already belongs to a "
                         "scenario; write the rest of your picks in a new call")


def _bound_why(world: World, kind: str, binding: dict, requires: list[str]) -> tuple[str, str]:
    """Why a binding is refused, with the table and key it names, or nothing (D381)."""
    missing = binding_refusal(binding, world)
    if missing:
        return BINDING, missing
    if kind in BOUND_KINDS and not binding and not requires:
        return UNBOUND, "bind it to rows you looked up, or say in requires what the world lacks"
    return "", ""


def _near(statement: str, record: R.Scenario, skip: str, situation: str) -> bool:
    """Whether a statement nearly repeats a written one. An exact twin is left to the fold, and a variant of
    the same situation to the one per situation and variant rule."""
    if record.id == skip or (situation and record.situation == situation):
        return False
    return NEAR_REPEAT <= rouge_l(statement, record.statement) < 1.0


def _near_why(root: GenerateRoot, statement: str, skip: str = "", situation: str = "") -> tuple[str, str]:
    """The first written statement, in the store or this session, the new one nearly repeats, or nothing (D383)."""
    held = [*R.load_scenarios(root.workdir), *root.found]
    twin = next((r.id for r in held if _near(statement, r, skip, situation)), "")
    return (NEAR, f"{twin} already says this") if twin else ("", "")


def _sentences(text: str) -> set[str]:
    from kullback.synth.generate import canonical

    return {said for said in (canonical(part) for part in _SENTENCE.split(text)) if said}


def pair_refusal(ask: str, end: str, world: World) -> str:
    """Why an ask and an end leak into each other or name a tool, or one is missing, or nothing (D393)."""
    if bool(ask) != bool(end):
        return ASK_WITHOUT_END if ask else END_WITHOUT_ASK
    if statement_refusal(ask, world, tools_only=True):
        return TOOL_IN_ASK
    if statement_refusal(end, world, tools_only=True):
        return TOOL_IN_END
    return ASK_SHARES_END if _sentences(ask) & _sentences(end) else ""


def _why(root: GenerateRoot, args: WriteArgs, statement: str, why: str) -> tuple[str, str]:
    why = why or (statement_refusal(statement, root.world) if statement else EMPTY)
    why = why or pair_refusal(_one_line(args.ask), _one_line(args.end), root.world)
    why = why or _policy_refusal(args.policy_lines, root.world.policy)
    if why:
        return why, ""
    near = _near_why(root, statement, situation=args.situation)
    return near if near[0] else _bound_why(root.world, root.kind, args.binding, _requires(args.requires))


def written(root: GenerateRoot, args: WriteArgs) -> Rows:
    """One scenario written into this generation call's findings, or refused with why."""
    if args.source != root.kind:
        raise ValueError(f"this session generates from the source {root.kind}; write with source {root.kind}")
    _already_grouped(root, args)
    statement = " ".join(args.statement.split())
    anchor, why, made = CHECKS[root.kind](root, args, statement)
    _closed(root, anchor)
    why, detail = _why(root, args, statement, why)
    if why:
        _refuse(root, anchor, why, detail)
    record = R.born(statement, made["source"], R.policy_lines_in(args.policy_lines, root.world.policy)[0],
                    made["frequency"]).model_copy(update={
                        "situation": made.get("situation", ""), "variant": made.get("variant", ""),
                        "binding": dict(args.binding), "requires": _requires(args.requires),
                        "ask": _one_line(args.ask), "end": _one_line(args.end)})
    root.found.append(record)
    root.written.add(anchor)
    root.grouped.update({at: record.id for at in made.get("picked", ())})
    row = {"id": record.id, "statement": record.statement, "source": root.kind, "ask": record.ask, "end": record.end}
    if root.bus is not None:
        root.bus.publish_custom("scenario_written", row)
    return Rows(summary=f"written {record.id}", rows=[row])


def rewritten(root: GenerateRoot, args: RewriteArgs) -> Rows:
    """One scenario sent back, rewritten under the rule, or refused with why."""
    record = root.fixing.get(args.id)
    if record is None:
        raise ValueError(f"{args.id!r} is not a scenario sent back in this session ({UNKNOWN})")
    anchor = ("fix", args.id)
    _closed(root, anchor)
    statement = " ".join(args.statement.split())
    binding = dict(record.binding if args.binding is None else args.binding)
    requires = list(record.requires) if args.requires is None else _requires(args.requires)
    ask = record.ask if args.ask is None else _one_line(args.ask)
    end = record.end if args.end is None else _one_line(args.end)
    why, detail = (statement_refusal(statement, root.world) if statement else EMPTY), ""
    why = why or pair_refusal(ask, end, root.world)
    why, detail = (why, detail) if why else _near_why(root, statement, args.id, record.situation)
    why, detail = (why, detail) if why else _bound_why(root.world, record.source.kind, binding, requires)
    if why:
        _refuse(root, anchor, why, detail)
    root.fixed[args.id] = {"statement": statement, "binding": binding, "requires": requires, "ask": ask, "end": end}
    root.written.add(anchor)
    row = {"id": args.id, **root.fixed[args.id]}
    if root.bus is not None:
        root.bus.publish_custom("scenario_rewritten", row)
    return Rows(summary=f"rewritten {args.id}", rows=[row])


# --- the tools -----------------------------------------------------------------------------


def _scenario_row(record: R.Scenario, status: str) -> dict:
    row = {"id": record.id, "statement": record.statement, "status": status}
    row = {**row, "ask": record.ask, "end": record.end} if record.ask or record.end else row
    return {**row, "situation": record.situation, "variant": record.variant} if record.situation else row


def _scenario_rows(root: GenerateRoot) -> list[dict]:
    rows = [_scenario_row(r, r.status) for r in R.load_scenarios(root.workdir)]
    return rows + [_scenario_row(r, "written now") for r in root.found]


def item_rows(root: GenerateRoot) -> list[dict]:
    """The numbered items as list_openings shows them; a situation shows at most OPENINGS_SHOWN opening texts."""
    if root.kind != "recordings":
        return [{"number": at, "text": item["text"]} for at, item in enumerate(root.items)]
    return [{"number": at, "situation": item["situation"], "statement": item["statement"], "kind": item["kind"],
             "variants": item["variants"], "openings": [row["text"] for row in item["openings"][:OPENINGS_SHOWN]],
             "more_openings": max(len(item["openings"]) - OPENINGS_SHOWN, 0)} for at, item in enumerate(root.items)]


def _plan_tools(root: GenerateRoot) -> dict[str, tuple]:
    """The plan session's own tools over the plan this generation call holds (D380)."""
    plan, world, items = root.plan, root.world, root.items

    async def list_plan(args: NoArgs) -> Rows:
        rows = plan.rows()
        return Rows(summary=f"{len(rows)} situations, {len(plan.skipped)} openings skipped", rows=rows)

    async def write_situation(args: SituationArgs) -> Rows:
        made = plan_mod.write_situation(plan, world, items, args.statement, args.kind, args.openings, args.variants)
        return Rows(summary=f"written {made['id']}", rows=[{"id": made["id"], "openings": len(made["openings"])}])

    async def attach_openings(args: AttachArgs) -> Rows:
        made = plan_mod.attach_openings(plan, items, args.id, args.openings)
        return Rows(summary=f"attached to {made['id']}", rows=[{"id": made["id"], "openings": len(made["openings"])}])

    async def skip_opening(args: SkipArgs) -> Rows:
        rows = plan_mod.skip_opening(plan, items, args.openings, args.why)
        return Rows(summary=f"skipped {len(rows)}")

    return {"list_plan": ("The situations of the plan so far.", NoArgs, list_plan),
            "write_situation": ("Write one situation of the plan.", SituationArgs, write_situation),
            "attach_openings": ("Add openings to a situation of the plan.", AttachArgs, attach_openings),
            "skip_opening": ("Mark items that are no opening ask, with why.", SkipArgs, skip_opening)}


def generate_tools(root: GenerateRoot) -> list[AgentTool]:
    """The role's own tools of one session, the subset its source uses; the web tools are D377's."""

    async def list_openings(args: NoArgs) -> Rows:
        rows = item_rows(root)
        return Rows(summary=f"{len(rows)} items", rows=rows)

    async def list_scenarios(args: NoArgs) -> Rows:
        rows = _scenario_rows(root)
        return Rows(summary=f"{len(rows)} scenarios", rows=rows)

    async def write_scenario(args: WriteArgs) -> Rows:
        return written(root, args)

    async def rewrite_scenario(args: RewriteArgs) -> Rows:
        return rewritten(root, args)

    every = {"list_openings": ("The numbered items of this session.", NoArgs, list_openings),
             "list_scenarios": ("The scenarios already written.", NoArgs, list_scenarios),
             "write_scenario": ("Write one scenario under the rule.", WriteArgs, write_scenario),
             "rewrite_scenario": ("Rewrite one scenario sent back.", RewriteArgs, rewrite_scenario),
             **_plan_tools(root)}
    return [AgentTool(name, every[name][0], every[name][1], Rows, every[name][2], render=render)
            for name in SESSION_TOOLS[root.kind] if name in every]


def generate_extension(root: GenerateRoot) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over the store, the session's tools, the shared web tools
    where its source searches, the prompt."""

    def setup(api: ExtensionAPI) -> None:
        folder = R.store_dir(root.workdir)
        folder.mkdir(parents=True, exist_ok=True)
        register_base_tools(api, folder, only=BASE_ONLY)
        for tool in generate_tools(root):
            api.register_tool(tool)
        if set(WORLD_TOOLS) & set(SESSION_TOOLS[root.kind]):
            for tool in world_tools(root.world):
                api.register_tool(tool)
        if set(WEB_TOOLS) & set(SESSION_TOOLS[root.kind]):
            register_web_tools(api, search=root.reader, refuse=root.refuse)
        world = root.world
        for name, text in generate_prompt.sections(SESSION_TOOLS[root.kind], root.kind):
            api.add_prompt_section(f"synth_{name}", prompt_block(name, text))
            if name == "receives":
                api.add_prompt_section("synth_environment", prompt_block(
                    "environment", prompt_mod.environment_text(world.policy, world.tools, world.description)))

    return setup


def bus_for(workdir: Any) -> Bus:
    """The workdir's one event stream, under the generator's agent name."""
    return Bus(R.store_dir(workdir).parent / "bus.jsonl", agent=AGENT)


def turns_for(root: GenerateRoot) -> int:
    """The turn budget of one session, by its kind: its own for the search, else sized per item."""
    if root.kind == "search":
        return min(MAX_TURNS, SEARCH_TURNS)
    return min(MAX_TURNS, max(MIN_TURNS, TURNS_PER_ITEM * max(len(root.items), len(root.fixing), 1)))


def run(root: GenerateRoot, model: Any, opening: str, stage: str, ceiling: Optional[budget.Ceiling] = None,
        subscribers: tuple = ()) -> bool:
    """One session on its opening message; whether it stopped on its turn cap. Refusals settle after."""
    turns = turns_for(root)
    name = getattr(model, "name", None)
    harness = AgentHarness(model=model, max_turns=turns, context=ContextConfig(window=budget.window_for(name)),
                           bus=root.bus)
    detach = [harness.subscribe(subscriber) for subscriber in subscribers]
    detach.append(harness.subscribe(budget.subscriber(root.workdir, stage, name, ceiling)))
    try:
        load_extensions(harness, [generate_extension(root)])
        capped = session_mod._run_session(harness, opening, turns, [])
    finally:
        for unsubscribe in detach:
            unsubscribe()
    settle_refusals(root)
    return bool(capped)


__all__ = ["AGENT", "ASK_SHARES_END", "ASK_WITHOUT_END", "END_WITHOUT_ASK", "TOOL_IN_ASK", "TOOL_IN_END",
           "GenerateRoot", "pair_refusal", "NEAR", "NEAR_REPEAT", "OPENING_GROUPED", "SEARCH_TURNS", "SESSION_TOOLS", "bus_for", "generate_extension", "generate_tools", "rewritten", "run",
           "settle_refusals", "turns_for", "written"]
