"""The scenario generator: scenarios from the built Environment, its recordings, a web search and typed lines (D372).

The generator is a role on the shared agent harness and event stream, as the Examiner and the scenario
review are (generate_tools.py): one session per batch of one source, on the workdir bus under the agent
name synth, its calls booked under the stage synth_generation by the budget subscriber. The recordings
source hands a session a batch of opening asks, one per recording, and the agent groups them into
statements; frequency is the count of openings grouped. The search source hands one session the
harness's web reader as the shared web tools (D377) over search.Crawled: the agent searches, reads pages
through the D225 crawl, and writes a statement only with the span of a page it read, whose byte range
becomes the citation. The ask source
hands a session a batch of lines a customer typed and the agent rewrites each as a scenario in the third
person; the line as typed stays in the source only, since the customer's own shape would tell the blind
review where it came from. A line naming a tool is dropped before any session.

The rule lives in the write tools, which refuse with the reason so the agent writes again; a second
refusal drops the scenario, counted per reason. Repeats fold by code only, statements equal up to case,
punctuation and spacing, recordings first, so a search or ask scenario the recordings already show keeps
the recordings record and adds its own source under `also_from`; no model judges a repeat. The agent
lists what is already written before it writes, and the review may rule a scenario a repeat of another. A record keeps the id it was born with,
also after a fix rewrites its statement. The fix step is a session too, under the stage synth_fix.

No session leaves items unwritten (D394): after the batched pass of one source, the items still unwritten
run once more in batches of half the size, at least RETRY_MIN; a session that ended empty or on its cap
left its items unwritten, so they are in that set. What the second pass wrote is counted per source. A retry
session gets at least the turns per item the first pass spent (D413): its items are the hard ones, and the
floor alone capped all eight small retries of one live run.

The gaps pass runs last (D398): the recordings mirror the recorded mix, so it asks, by code, for what the
store lacks: one item per cell of (situation kind, bound table) no kept scenario hits and one per policy
line no kept scenario quotes, one scenario each, through the same sessions, write tools and refusals.

Situations come from the web too (D410): plan_search sessions read the web as the search source does and write
situations, each on a page span, into the plan on disk, the thinnest kinds named first. The recordings pass
then writes their scenarios, the source carrying the span. With a target, generate runs in rounds (plan_search,
the recordings pass over the situations with no scenario yet, the gaps pass) until the store keeps the target
or a round adds under STOP_SHARE of it; the diversity of what is kept is measured at the end.

Rows first (D420): before any scenario session, code hands each item a slice of the world (slice.py), spread over
owners the store has not used, a kind and a policy line chosen by distance from the recorded share (plan.py
by_distance: kinds by the recorded Tasks behind them, merged by stem; policy lines evenly, since no recorded
share of a line is held), and the session writes onto them, a few slices per session (SLICE_BATCH). The search
session gets SEARCH_SLICES slices to write its pages' stories onto.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Optional

from kullback import domain
from kullback.runner import budget
from kullback.runner.records import write_json
from kullback.spec import intent_tools
from kullback.synth import generate_prompt
from kullback.synth import generate_tools as T
from kullback.synth import plan as plan_mod
from kullback.synth import records as R
from kullback.synth import search as search_mod
from kullback.synth import slice as slice_mod
from kullback.synth.rule import EMPTY, ID_SHAPE, TOOL_NAME, World, rouge_l, statement_refusal, world_of
from kullback.synth.tasks import recorded_fields

# How many opening asks or typed lines one session generates from.
BATCH = 25
# The smallest batch of the second pass over the items a first pass left unwritten (D394).
RETRY_MIN = 4
STAGE = "synth_generation"
FIX_STAGE = "synth_fix"
CEILING_NOTE = "stopped at the spend ceiling; generate again to continue"
# How many scenarios one gap item asks for (D398).
GAP_PER = 1
# A policy line under this many words is a heading or a field name, never a rule a situation turns on.
LINE_WORDS = 5
_PUNCT = re.compile(r"[^\w\s]")
# How many situations one plan_search session is asked for, at most (D410).
PLAN_SEARCH_BATCH = 25
# How many scenarios a plan_search situation is counted on when a round sizes its ask (D410).
PER_SITUATION = 3
# A round adding under this share of the target ends the rounds (D410).
STOP_SHARE = 0.05
# How many of the thinnest kinds the plan_search opening names (D410).
THIN = 3
# How many slices one scenario session writes onto at most, and how many the search session gets (D420).
SLICE_BATCH = 5
SEARCH_SLICES = 5


@dataclass
class Generation:
    """One generation call: the world, the model, what every session found and dropped, the stream."""

    world: World
    model: Any
    found: list = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    notes: list = field(default_factory=list)
    bus: Any = None
    ceiling: Optional[budget.Ceiling] = None
    subscribers: tuple = ()
    capped: Counter = field(default_factory=Counter)
    spent: Counter = field(default_factory=Counter)
    ungrouped: list = field(default_factory=list)
    plan: plan_mod.Plan = field(default_factory=plan_mod.Plan)
    unplanned: list = field(default_factory=list)
    retried: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)
    used: set = field(default_factory=set)
    fields: set = field(default_factory=set)

    def over(self) -> bool:
        if self.ceiling is not None and self.ceiling.remaining <= 0:
            if CEILING_NOTE not in self.notes:
                self.notes.append(CEILING_NOTE)
            return True
        return False

    def search_reader(self, engine: Any, env: Optional[dict], corpus_url: Optional[str], exclude: Iterable[str],
                      pages: dict) -> Any:
        """The harness's reader as a crawl, or None with the note saying why no search runs."""
        if not self.world.description:
            self.notes.append("no one-line description of the business, so no search was run")
            return None
        engine = engine if engine is not None else search_mod.reader(self.world.workdir, env)
        if engine is None:
            self.notes.append(search_mod.OFF_NOTE)
            return None
        return search_mod.Crawled(engine, self.world.workdir, keys=pages, corpus_url=corpus_url, exclude=exclude,
                                  notes=self.notes)

    def session(self, kind: str, opening: str, stage: str = STAGE, per_item: float = 0, **root: Any) -> T.GenerateRoot:
        made = T.GenerateRoot(world=self.world, kind=kind, found=self.found, dropped=self.dropped, bus=self.bus,
                              plan=self.plan, **root)
        capped, turns = T.run(made, self.model, opening, stage, self.ceiling, self.subscribers, per_item)
        if capped:
            self.capped[kind] += 1
        self.spent[kind] += turns
        return made


def _entries(workdir: Path) -> list[str]:
    folder = R.store_dir(workdir)
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


# --- the recordings ---------------------------------------------------------------------


def _opening_index(workdir: Path, task_id: str) -> dict[str, int]:
    """Per recording, the opening ask the Intent marked; none where the Task has no Spec yet."""
    from kullback.spec.schema import load_spec

    try:
        spec = load_spec(workdir, task_id)
    except (OSError, ValueError):
        return {}
    return intent_tools.opening_turns(spec.intent.facts) if spec is not None else {}


def openings(workdir: Any) -> list[dict]:
    """Every recording's opening ask, one row each: the Task, the recording and the user's words.

    The opening is the turn the Intent marked as the first ask where a Spec exists, else the
    recording's first user turn.
    """
    root = Path(workdir)
    traces = intent_tools.trace_index(root)
    out: list[dict] = []
    for path in sorted((root / "tasks").glob("*.json")):
        try:
            found = intent_tools.recordings_of(root, path.stem, traces)
        except ValueError:
            continue
        marked = _opening_index(root, path.stem)
        for recording, turns in sorted(found.recordings.items()):
            turn = next((t for t in turns if t.index == marked.get(recording)), turns[0])
            out.append({"task_id": path.stem, "recording": recording, "text": turn.text})
    return out


# --- the plan (D380) ---------------------------------------------------------------------


def _plan_session(generation: Generation, chunk: list[dict]) -> None:
    listing = generate_prompt.plan_listing(generation.plan.rows())
    opening = generate_prompt.opening("plan", _entries(generation.world.workdir), count=len(chunk), items=listing)
    generation.session("plan", opening, items=chunk)
    plan_mod.write_plan(generation.world.workdir, generation.plan)


def plan_openings(generation: Generation, *, batch: int = plan_mod.PLAN_BATCH) -> None:
    """The plan sessions over every opening not yet planned, then one more over what they left.

    The plan already on disk is read first, so planning again adds to it; what remains in no situation
    and not skipped is kept on the generation as unplanned.
    """
    workdir = generation.world.workdir
    generation.plan = plan_mod.load_plan(workdir)
    rows = openings(workdir)
    todo = plan_mod.unplanned(generation.plan, rows) if generation.model is not None else []
    size = max(int(batch), 1)
    for start in range(0, len(todo), size):
        if generation.over():
            break
        _plan_session(generation, todo[start:start + size])
    left = plan_mod.unplanned(generation.plan, rows)
    if left and todo and not generation.over():
        _plan_session(generation, left)
    generation.unplanned = plan_mod.unplanned(generation.plan, rows)


def items_text(generation: Generation, chunk: list[dict]) -> str:
    """A slice session's items as its opening lists them, the plan's kinds once (D420)."""
    rows = [T.item_row(at, item) for at, item in enumerate(chunk)]
    return generate_prompt.slice_items(rows, generation.plan.kinds())


def _pass(generation: Generation, kind: str, items: list[dict], size: int, retry: bool = False,
          per_item: float = 0) -> list[dict]:
    """One session per batch of items of one source, stopping at the ceiling; the items no session grouped."""
    left: list[dict] = []
    for start in range(0, len(items), size):
        if generation.over():
            return left
        chunk = items[start:start + size]
        opening = generate_prompt.opening(kind, count=len(chunk), items=items_text(generation, chunk))
        made = generation.session(kind, opening, per_item=per_item, items=chunk)
        row = {"source": kind, "batch": start // size, "items": len(chunk), "ungrouped": len(chunk) - len(made.grouped)}
        generation.ungrouped.append({**row, "retry": True} if retry else row)
        left += [item for at, item in enumerate(chunk) if at not in made.grouped]
    return left


def _batched(generation: Generation, kind: str, items: list[dict], batch: int,
             unwritten: Optional[Callable[[], list[dict]]] = None) -> None:
    """The batched pass of one source, then once more over what it left unwritten in halves (D394).

    `unwritten` names the items still unwritten after the first pass; by default, those no session grouped.
    """
    size = max(min(int(batch), SLICE_BATCH), 1)
    spent = generation.spent[kind]
    left = _pass(generation, kind, items, size)
    per_item = (generation.spent[kind] - spent) / max(len(items), 1)
    left = unwritten() if unwritten is not None else left
    if not left or generation.over():
        return
    before = len(generation.found)
    _pass(generation, kind, left, max(size // 2, RETRY_MIN), retry=True, per_item=per_item)
    generation.retried[kind] = {"items": len(left), "written": len(generation.found) - before}


def situation_items(generation: Generation) -> list[dict]:
    """The plan's situations as items, one per situation and one per variant not yet written, in the store or by
    this generation (D420)."""
    held = {(r.situation, r.variant) for r in [*R.load_scenarios(generation.world.workdir), *generation.found]
            if r.situation}
    out = []
    for situation in generation.plan.situations:
        for variant in ["", *situation["variants"]]:
            if (situation["id"], variant) in held:
                continue
            item = {"situation": situation["id"], "statement": situation["statement"], "kind": situation["kind"],
                    "variant": variant, "openings": list(situation["openings"])}
            out.append({**item, "source": situation["source"], "span": situation["span"]} if situation.get("source")
                       else item)
    return out


def _owners_used(generation: Generation) -> set[str]:
    """The owner keys the kept scenarios are bound to, and those handed out by this generation."""
    owner = slice_mod.slicer(generation.world, generation.fields).owner_table
    held = {key for record in _kept(generation) for key in R.listed(record.binding).get(owner, ())} if owner else set()
    return held | generation.used


def sliced(generation: Generation, items: list[dict], label: str) -> list[dict]:
    """Each item handed a kind, a policy line and a slice (D420). The kind and the line an item names stay; the
    rest are chosen by distance from the recorded share (plan.by_distance), counting the kept scenarios and the
    items handed out before it; the slice is picked by seed over owners not yet used, holding a row of the table a
    cell gap names where one does."""
    world, plan = generation.world, generation.plan
    kept = _kept(generation)
    kind_of = _kinds(kept, plan)
    have_kinds = Counter(kind_of.values())
    have_lines = Counter(line for record in kept for line in record.policy_lines)
    want_kinds = dict(plan_mod.recorded_kinds(plan))
    want_lines = {line: 1 for line in policy_lines(world.policy)}
    used = _owners_used(generation)
    out = []
    for at, item in enumerate(items):
        gap = item.get("gap") or {}
        kind = item.get("kind") or gap.get("kind") or plan_mod.by_distance(want_kinds, have_kinds)
        line = item.get("line") or gap.get("line") or plan_mod.by_distance(want_lines, have_lines)
        have_kinds[plan_mod.canonical_kind(plan, kind)] += 1
        have_lines[line] += 1
        seed = f"{label}:{len(kept)}:{at}:{item.get('situation') or item.get('text', '')}:{item.get('variant', '')}"
        cut = slice_mod.pick_slice(world, seed, used, str(gap.get("table") or ""), generation.fields)
        if cut is not None:
            used.add(cut.owner_key)
            generation.used.add(cut.owner_key)
        out.append({**item, "kind": kind, "line": line, "slice": cut})
    return out


def from_recordings(generation: Generation, *, batch: int = BATCH) -> None:
    """One scenario per situation of the plan and one per variant, each onto its slice, one session per batch
    (D381, D420)."""
    if generation.model is None:
        return
    _batched(generation, "recordings", sliced(generation, situation_items(generation), "recordings"), batch,
             lambda: sliced(generation, situation_items(generation), "recordings"))


# --- the search -------------------------------------------------------------------------


def from_search(generation: Generation, *, engine: Any = None, env: Optional[dict] = None,
                corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
                refuse: Optional[Callable[[str], str]] = None) -> None:
    """Scenarios read off pages the harness's reader names, each citing the byte range of its span.

    With nothing live and no memo there is no reader: no session runs, no page is read, and the note
    names the switch.
    """
    if generation.model is None:
        return
    pages: dict = {}
    reader = generation.search_reader(engine, env, corpus_url, exclude, pages)
    if reader is None or generation.over():
        return
    items = sliced(generation, [{"text": "a story a page you read shows, written onto this slice"}] * SEARCH_SLICES,
                   "search")
    opening = generate_prompt.opening("search", count=len(items), items=items_text(generation, items))
    generation.session("search", opening, reader=reader, pages=pages, refuse=refuse or domain.destination_refusal,
                       items=items)


def thin_line(plan: plan_mod.Plan) -> str:
    """The plan's kinds with their situation counts, fewest first, and the THIN thinnest named."""
    counts = plan.per_kind()
    listed = ", ".join(f"{kind} {n}" for kind, n in counts.items()) or "none yet"
    return f"{listed}; thinnest: {', '.join(list(counts)[:THIN]) or 'none'}"


def plan_search(generation: Generation, count: int, *, engine: Any = None, env: Optional[dict] = None,
                corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
                refuse: Optional[Callable[[str], str]] = None) -> int:
    """Situations from the web into the plan on disk, at most PLAN_SEARCH_BATCH per session, until `count` are
    written or a session writes none (D410); how many were written."""
    if generation.model is None or count <= 0:
        return 0
    pages: dict = {}
    reader = generation.search_reader(engine, env, corpus_url, exclude, pages)
    if reader is None:
        return 0
    start = len(generation.plan.situations)
    while not generation.over():
        left = count - (len(generation.plan.situations) - start)
        if left <= 0:
            break
        wanted = min(left, PLAN_SEARCH_BATCH)
        opening = generate_prompt.opening("plan_search", _entries(generation.world.workdir), count=wanted,
                                          items=thin_line(generation.plan))
        before = len(generation.plan.situations)
        generation.session("plan_search", opening, reader=reader, pages=pages, wanted=wanted,
                           refuse=refuse or domain.destination_refusal)
        plan_mod.write_plan(generation.world.workdir, generation.plan)
        if len(generation.plan.situations) == before:
            break
    return len(generation.plan.situations) - start


# --- the customer's lines ---------------------------------------------------------------


def ask_lines(path: Any) -> list[str]:
    target = Path(path)
    if not target.is_file():
        return []
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def from_asks(generation: Generation, lines: Iterable[str], *, batch: int = BATCH) -> None:
    """Each line a customer typed, rewritten as a scenario under the rule onto its slice, one session per batch.

    A line naming a tool of the Environment is dropped and counted before any session.
    """
    kept = []
    for line in lines:
        why = statement_refusal(line, generation.world, tools_only=True)
        if why:
            generation.dropped[why] += 1
        else:
            kept.append({"text": line})
    if generation.model is None:
        if kept:
            generation.notes.append("no model, so no ask line was rewritten as a scenario")
        return
    _batched(generation, "ask", sliced(generation, kept, "ask"), batch)


# --- the gaps (D398) --------------------------------------------------------------------


def policy_lines(policy: str) -> list[str]:
    """The lines of the policy a situation can turn on: each line with its list marks stripped, leaving out
    headings, labels ending in a colon and lines under LINE_WORDS words."""
    out: list[str] = []
    for raw in str(policy or "").splitlines():
        line = raw.strip().lstrip("-*").strip()
        if raw.lstrip().startswith("#") or line.endswith(":") or len(line.split()) < LINE_WORDS or line in out:
            continue
        out.append(line)
    return out


def cell_tables(state: dict) -> list[str]:
    """The tables of the world that hold rows; a table with none can never be bound, so it gives no cell."""
    return [table for table, rows in state.items() if rows]


def _covered(line: str, quoted: set[str]) -> bool:
    return any(quote in line or line in quote for quote in quoted)


def _kept(generation: Generation) -> list[R.Scenario]:
    """The scenarios the store holds and this generation wrote, one per statement as the fold compares them."""
    first: dict[str, R.Scenario] = {}
    for record in [*R.load_scenarios(generation.world.workdir), *generation.found]:
        first.setdefault(canonical(record.statement), record)
    return list(first.values())


def gap_text(gap: dict) -> str:
    """A gap as its item says it."""
    if gap.get("what") == "cell":
        return f"a {gap['kind']} situation bound to a {gap['table']} row"
    return f"a situation where this policy line decides: {gap['line']}"


def gap_items(generation: Generation) -> list[dict]:
    """The items of the gaps pass: every cell of (situation kind of the plan, table of the world with rows) no kept
    scenario hits, then every policy line no kept scenario quotes, each carrying the gap it is."""
    kept = _kept(generation)
    kind_of = _kinds(kept, generation.plan)
    hit = {(kind_of[r.id], table) for r in kept for table in r.binding}
    quoted = {line for r in kept for line in r.policy_lines}
    kinds = sorted({situation["kind"] for situation in generation.plan.situations})
    gaps = [{"what": "cell", "kind": kind, "table": table} for kind in kinds
            for table in cell_tables(generation.world.state)
            if (kind, table) not in hit]
    gaps += [{"what": "policy_line", "line": line} for line in policy_lines(generation.world.policy)
             if not _covered(line, quoted)]
    return [{"text": gap_text(gap), "gap": gap} for gap in gaps for _ in range(GAP_PER)]


def gap_counts(generation: Generation) -> dict:
    """The cells hit and the policy lines quoted over the kept scenarios, out of how many."""
    kept = _kept(generation)
    cells = spread(kept, generation.plan, cell_tables(generation.world.state))[1]
    lines = policy_lines(generation.world.policy)
    quoted = {line for r in kept for line in r.policy_lines}
    return {**cells, "lines_covered": sum(1 for line in lines if _covered(line, quoted)), "lines": len(lines)}


def from_gaps(generation: Generation, *, batch: int = BATCH) -> None:
    """One scenario per gap item, one session per batch, the counts before and after the pass kept."""
    if generation.model is None:
        return
    before = gap_counts(generation)
    items = sliced(generation, gap_items(generation), "gaps")
    _batched(generation, "gaps", items, batch)
    generation.gaps = {"items": len(items), "before": before, "after": gap_counts(generation)}


# --- folding and coverage ---------------------------------------------------------------


def canonical(text: str) -> str:
    """A statement as the fold compares it: lowercase, punctuation dropped, spacing collapsed."""
    return " ".join(_PUNCT.sub(" ", str(text or "").lower()).split())


def fold(records: list[R.Scenario]) -> tuple[list[R.Scenario], list[dict]]:
    """The records with repeats folded in by code, in the order given, and the folds.

    Two statements fold when they are equal up to case, punctuation and spacing; no model judges it.
    The first record keeps the others' sources under also_from, and two recordings records add up.
    """
    first: dict[str, R.Scenario] = {}
    folded: list[dict] = []
    for record in records:
        held = first.setdefault(canonical(record.statement), record)
        if held is record:
            continue
        held.also_from += [record.source, *record.also_from]
        if held.source.kind == record.source.kind == "recordings":
            held.frequency += record.frequency
            held.source.task_ids = sorted({*held.source.task_ids, *record.source.task_ids})
        folded.append({"id": record.id, "into": held.id, "kind": record.source.kind})
    return list(first.values()), folded


def _kinds(kept: list[R.Scenario], plan: plan_mod.Plan) -> dict[str, str]:
    """Each kept scenario's situation kind: its situation's in the plan, else the kind it was written with (a
    search scenario, D410), else the gap it fills, else none; merged by stem with the plan's first such kind (D420)."""
    kinds = {situation["id"]: situation["kind"] for situation in plan.situations}
    return {r.id: plan_mod.canonical_kind(plan, kinds.get(r.situation) or r.kind or r.source.gap.get("kind") or "none")
            for r in kept}


def spread(kept: list[R.Scenario], plan: plan_mod.Plan, tables: Iterable[str]) -> tuple[dict, dict]:
    """How the kept scenarios spread over situation kinds, bound tables and policy lines, and the cells of
    (situation kind, bound table) they hit out of the kinds seen times the tables of the world."""
    kind_of = _kinds(kept, plan)
    per_kind = Counter(kind_of.values())
    per_table = Counter(table for record in kept for table in set(record.binding))
    per_line = Counter(line for record in kept for line in set(record.policy_lines))
    hit = {(kind_of[record.id], table) for record in kept for table in record.binding}
    counters = {"situation_kind": dict(per_kind.most_common()), "bound_table": dict(per_table.most_common()),
                "policy_line": dict(per_line.most_common())}
    return counters, {"hit": len(hit), "possible": len(per_kind) * len(set(tables))}


def coverage(kept: list[R.Scenario], folded: list[dict], dropped: Counter, ungrouped: Iterable[dict] = (),
             unplanned: Iterable[dict] = (), *, plan: Optional[plan_mod.Plan] = None,
             tables: Iterable[str] = ()) -> dict:
    """Scenarios per source kind, what the recordings never show, the asks nothing else covers, drops, per
    batch the items no scenario grouped, the openings the plan left in no situation and not skipped, and
    the spread over situation kinds, bound tables and policy lines with the cells hit."""
    left = list(unplanned)
    counters, cells = spread(kept, plan or plan_mod.Plan(), tables)
    per_kind = Counter(record.source.kind for record in kept)
    return {"per_kind": {kind: per_kind.get(kind, 0) for kind in R.SOURCE_KINDS},
            "search_not_in_recordings": per_kind.get("search", 0),
            "asks_not_covered": per_kind.get("ask", 0),
            "folded": Counter(row["kind"] for row in folded),
            "dropped": dict(sorted(dropped.items())),
            "openings_ungrouped": list(ungrouped),
            "openings_unplanned": {"count": len(left), "ids": [R.mask(row.get("task_id")) for row in left]},
            "spread": counters, "cells": cells}


def diversity(workdir: Any, plan: plan_mod.Plan, tables: Iterable[str]) -> dict:
    """How diverse the store is (D410): the median pairwise ROUGE-L of its statements, scenarios per situation
    kind, the cells hit of possible, and situations per kind in the plan."""
    kept = R.load_scenarios(workdir)
    texts = [record.statement for record in kept]
    scores = [rouge_l(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    counters, cells = spread(kept, plan, tables)
    return {"kept": len(kept), "rouge_l_median": round(median(scores), 4) if scores else None,
            "situation_kind": counters["situation_kind"], "cells": cells, "situations_per_kind": plan.per_kind()}


def _settle(generation: Generation, workdir: Any, folds: list[dict]) -> list[R.Scenario]:
    """This generation's findings folded, the new ones written to the store and returned; the findings emptied."""
    kept, folded = fold(list(generation.found))
    folds += folded
    held = {canonical(record.statement) for record in R.load_scenarios(workdir)}
    new = [record for record in kept if canonical(record.statement) not in held]
    R.write_all(workdir, new)
    generation.found.clear()
    return new


def _round_rows(new: list[R.Scenario]) -> dict:
    per = Counter(record.source.kind for record in new)
    return {kind: per.get(kind, 0) for kind in R.SOURCE_KINDS}


def _one_pass(generation: Generation, wanted: set, searching: bool, asks: Any, batch: int, web: dict) -> None:
    """The sources named, each once, the gaps last; without the recordings the plan on disk is read."""
    workdir = generation.world.workdir
    if "recordings" in wanted:
        plan_openings(generation)
        from_recordings(generation, batch=batch)
    else:
        generation.plan = plan_mod.load_plan(workdir)
    if "search" in wanted and searching:
        from_search(generation, **web)
    elif "search" in wanted:
        generation.notes.append("search turned off")
    if "ask" in wanted:
        path = Path(asks) if asks else R.store_dir(workdir) / R.ASKS
        from_asks(generation, ask_lines(path), batch=batch)
    if "gaps" in wanted:
        from_gaps(generation, batch=batch)


def _stop(generation: Generation, target: int, last: dict) -> str:
    """Why the rounds end after this one (D410), or nothing."""
    if last["kept"] >= target:
        return "target"
    if generation.over():
        return "ceiling"
    return "slow" if sum(last["born"].values()) < STOP_SHARE * target else ""


def _rounds(generation: Generation, target: int, rounds: list[dict], born: list[R.Scenario], folds: list[dict],
            wanted: set, searching: bool, batch: int, web: dict) -> str:
    """Rounds of plan_search, the recordings pass and the gaps pass until the stop rule holds; why it stopped."""
    workdir = generation.world.workdir
    while not (stop := _stop(generation, target, rounds[-1])):
        searching = searching and "search" in wanted
        missing = math.ceil((target - rounds[-1]["kept"]) / PER_SITUATION)
        searched = plan_search(generation, missing, **web) if searching else 0
        if "recordings" in wanted or searched:
            from_recordings(generation, batch=batch)
        if "gaps" in wanted:
            from_gaps(generation, batch=batch)
        new = _settle(generation, workdir, folds)
        born += new
        rounds.append({"round": len(rounds) + 1, "situations": searched, "born": _round_rows(new),
                       "kept": len(R.load_scenarios(workdir))})
    return stop


def generate(workdir: Any, *, model: Any, asks: Any = None, searching: bool = True,
         batch: int = BATCH, description: str = "", engine: Any = None, env: Optional[dict] = None,
         corpus_url: Optional[str] = None, exclude: Iterable[str] = (), ceiling: Optional[budget.Ceiling] = None,
         subscribers: Iterable[Any] = (), refuse: Optional[Callable[[str], str]] = None,
         sources: Iterable[str] = R.SOURCE_KINDS, target: Optional[int] = None) -> dict:
    """Run the sources named as sessions, the gaps last, fold the repeats, write the new records and the counts.

    A record already in the store keeps its status and its review: generating again adds, never resets.
    Without the recordings no plan session runs; the plan on disk is read as it is. With a target (D410), the
    one pass is round 1, then each round runs plan_search for a third of what is missing, the recordings pass
    and the gaps pass, until the store keeps the target, a round adds under STOP_SHARE of it, or the ceiling.
    """
    wanted = set(sources)
    world = world_of(workdir, description)
    generation = Generation(world=world, model=model, bus=T.bus_for(workdir), ceiling=ceiling,
                    subscribers=tuple(subscribers), fields=recorded_fields(workdir))
    web = {"engine": engine, "env": env, "corpus_url": corpus_url, "exclude": exclude, "refuse": refuse}
    _one_pass(generation, wanted, searching, asks, batch, web)
    folds: list[dict] = []
    born = _settle(generation, workdir, folds)
    rounds = [{"round": 1, "situations": 0, "born": _round_rows(born), "kept": len(R.load_scenarios(workdir))}]
    stop = _rounds(generation, target, rounds, born, folds, wanted, searching, batch, web) if target is not None else ""
    counts = coverage(born, folds, generation.dropped, generation.ungrouped, generation.unplanned,
                      plan=generation.plan, tables=cell_tables(world.state))
    body = {"format": R.FORMAT, "notes": generation.notes, "coverage": counts,
            "born": len(born), "folded": folds, "capped": dict(sorted(generation.capped.items())),
            "retried": dict(sorted(generation.retried.items())), "gaps": generation.gaps}
    if target is not None:
        body.update(target=int(target), rounds=rounds, stop=stop,
                    diversity=diversity(workdir, generation.plan, cell_tables(world.state)))
    write_json(R.store_dir(workdir) / R.GENERATED, body)
    return body


# --- the fix step -----------------------------------------------------------------------


def source_text(world: World, record: R.Scenario) -> str:
    """What a record was written from: its grouped openings, its cited span, or the line as typed."""
    source = record.source
    if source.kind == "recordings":
        wanted = set(source.task_ids)
        return "\n".join(row["text"] for row in openings(world.workdir) if row["task_id"] in wanted)
    if source.kind == "search":
        raw = domain.cached_text(world.workdir, source.key).encode("utf-8")
        return raw[int(source.start or 0):int(source.end or 0)].decode("utf-8", errors="replace")
    return gap_text(source.gap) if source.kind == "gaps" else source.line


def _fix_items(world: World, sent: list[tuple[R.Scenario, str]]) -> str:
    return "\n\n".join(f"{record.id}: {record.statement}\n{why}\nbound to: {json.dumps(record.binding, sort_keys=True)}"
                       f"\nrequires: {json.dumps(record.requires)}\nask: {record.ask}\nend: {record.end}"
                       f"\nwritten from: {source_text(world, record)}"
                       for record, why in sent)


def _fixed(record: R.Scenario, why: str, rewrite: Optional[dict]) -> R.Scenario:
    if not rewrite:
        row = {"kind": "fix_failed", "why": why}
        return record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
    row = {"kind": "fixed", "why": why, "was": record.statement}
    return record.model_copy(update={**rewrite, "status": "born", "fixes": record.fixes + 1,
                                     "review": [*record.review, row]})


def fix(world: World, sent: list[tuple[R.Scenario, str]], model: Any, *, bus: Any = None,
        ceiling: Optional[budget.Ceiling] = None, subscribers: Iterable[Any] = ()) -> list[R.Scenario]:
    """Each scenario sent back, rewritten once from the same source with the reason, in one session.

    A scenario the session does not rewrite under the rule ends unsure, with why.
    """
    if not sent:
        return []
    generation = Generation(world=world, model=model, bus=bus if bus is not None else T.bus_for(world.workdir),
                    ceiling=ceiling, subscribers=tuple(subscribers))
    if model is None or generation.over():
        return [_fixed(record, why, None) for record, why in sent]
    opening = generate_prompt.opening("fix", _entries(world.workdir), items=_fix_items(world, sent))
    root = generation.session("fix", opening, stage=FIX_STAGE, fixing={record.id: record for record, _ in sent})
    return [_fixed(record, why, root.fixed.get(record.id)) for record, why in sent]


__all__ = ["SEARCH_SLICES", "SLICE_BATCH", "items_text", "sliced", "BATCH", "GAP_PER", "PER_SITUATION", "PLAN_SEARCH_BATCH", "STOP_SHARE", "THIN", "diversity",
           "plan_search", "thin_line", "LINE_WORDS", "cell_tables", "from_gaps", "gap_counts", "gap_items", "gap_text", "policy_lines",
           "RETRY_MIN", "plan_openings", "EMPTY", "FIX_STAGE", "ID_SHAPE", "Generation", "STAGE", "TOOL_NAME", "World", "canonical", "coverage", "spread",
           "fix", "fold", "from_asks", "from_recordings", "from_search", "generate", "openings", "source_text",
           "statement_refusal", "world_of"]
