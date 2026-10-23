"""The Spec stage of a build: each named Task's Intent mined, its Spec and Verifier written, the bus told.

A Task whose Spec and Spec Verifier are both on disk is skipped, so a resumed build writes only
what is missing; a Spec of the older demand grammar (no item) is rewritten as items, Intent and all,
so every Task gets the sanity item (automatic, no flag). An empty Spec is written once more; empty again, the Task is set aside with no
Verifier. A Spec no Run can pass after its one send-back is set aside the same way, as
unsatisfiable_spec. A Spec of a born Task leaving an end fact of its scenario unchecked (writer.unchecked_end,
D414) is set aside so as unchecked_end. A born Task's Spec goes back to the writer once where a write its end
names has no gate differing from the start, or a value the end names of the changed row is in no gate
(spec/ungated.py, D419); still found, it is marked (writes_ungated, end_values_ungated) and stands. A Verifier
whose code side holds on the Run that does nothing while its end names a write is marked held_by_idle (the trust
gate refuses it, spec/trust.py). The stage counts each mark. The miner also says whether the recorded user held to their opening asks, checked by
code (spec/coherence.py); a Task whose user strayed is set aside as intent_strayed before the writer, no
Verifier written. A set-aside Spec counts as done. Every call is priced under the `spec` stage of the workdir ledger (budget.json):
the miner through the harness subscriber, the writer through BudgetedModel, the two seams every
other stage uses. A Task that fails is counted with its error class and message, its Spec file put back as it was
before the attempt (so a rerun writes it again), and the loop goes on; the first failure message stands once
in the counts as first_failure.

After compile and the Examiner, `end_unpassable` ends each Task no right Run can pass with no Verifier (D439): a
gated write whose policy precondition names a value the user must give that no rendered fact carries
(precondition_unrendered), or References none of which holds every gate (no_reference_holds). The Reference
vetoes a Verifier and never sources one; a Task with no Reference keeps its Verifier.

A Verifier every code gate of which holds on the Run that makes no call and says nothing must carry a gated judge
item or a gated said item (D441). At write time such a Spec goes back to the writer once (`idle_send_back`); at
the stage's end, still so, the Task ends with no Verifier as idle_run_passes. An idle Run failing a code gate
changes nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.agent.bus import Bus
from kullback.gates.probes import write_tools_of
from kullback.gates.trust import schema_of
from kullback.runner.budget import BudgetedModel
from kullback.runner.canon import load_rules
from kullback.runner.records import read_json
from kullback.runner.verdict import verdict
from kullback.spec import events
from kullback.spec.actions import workdir_action_tools
from kullback.spec.canfail import Outcome, outcome_of
from kullback.spec.events import UNCHECKED_END
from kullback.spec.intent import STAGE, mine_intent
from kullback.spec.items import DEFAULT_GATE, World, _contains, _flat, contradictions, names_tool, norm, rendered
from kullback.spec.judged import judge_results_of
from kullback.spec.schema import SpecIntent, is_empty, load_spec, save_spec, spec_path
from kullback.spec.text import STOPWORDS, TOKEN_RE, normalise
from kullback.spec.trust import (
    HELD_BY_IDLE,
    faithful_references,
    held_by_idle,
    idle_run,
    load_spec_verifier,
    runs_on_disk,
)
from kullback.spec.ungated import END_VALUES_UNGATED, WRITES_UNGATED, send_back
from kullback.spec.writer import (
    EFFECTS_COUNT,
    NO_EFFECTS,
    Ruling,
    Written,
    is_item_check,
    item_checks,
    load_inputs,
    repair,
    runner_verifier_path,
    spec_verifier_path,
    unchecked_end,
    verifier_of,
    write_spec,
    write_verifier,
)

# What one Task's mining and writing may spend together when the caller names no ceiling.
SPEC_TASK_CEILING_USD = 1.0
# How much of a failed Task's error message its failure entry keeps.
FAILURE_CHARS = 200


def older_grammar(spec: Any) -> bool:
    """A Spec of the older demand grammar: checks, none of them an item, so its Verifier has no sanity item."""
    return spec is not None and bool(spec.checks) and not any(is_item_check(check) for check in spec.checks)


def written_already(workdir: Path, task_id: str) -> bool:
    """A Task whose Spec and its compiled Verifier are both on disk, or whose Spec is set aside; a Spec of
    the older grammar is not written yet: the stage rewrites it as items."""
    if not spec_path(workdir, task_id).is_file():
        return False
    spec = load_spec(workdir, task_id)
    if spec.set_aside is not None:
        return True
    return spec_verifier_path(workdir, task_id).is_file() and not older_grammar(spec)


def _version(workdir: Path, task_id: str) -> int:
    spec = load_spec(workdir, task_id)
    return spec.version if spec is not None else 0


def write_from_intent(workdir: Path, task_id: str, intent: SpecIntent, model: Any, ceiling_usd: float,
                      bus: Bus, coherence: Optional[dict] = None) -> float:
    """Write and compile one Task's Spec from a given Intent under `intent`; return the spend.

    An empty first Spec is written once more in a fresh session. Empty again, the Spec is saved set aside
    as empty_spec, task.set_aside is published and no Verifier is written; a Spec no Run can pass is set
    aside so as unsatisfiable_spec; else spec.written is published. `coherence` is the miner stage's
    record (spec/coherence.py), kept on the Spec.
    """
    model_id = getattr(model, "name", None)
    inputs = load_inputs(workdir, task_id, intent)
    priced = BudgetedModel(model, stage=STAGE, workdir=workdir, model_id=model_id)
    tries = [write_spec(task_id, inputs, priced, model_id=model_id, ceiling_usd=ceiling_usd)]
    empty_first = is_empty(tries[0].spec)
    if empty_first:
        left = max(ceiling_usd - float(tries[0].session.get("usd") or 0.0), 0.0)
        tries.append(write_spec(task_id, inputs, priced, model_id=model_id, ceiling_usd=left))
    written, empty = tries[-1], is_empty(tries[-1].spec)
    spent = sum(float(t.session.get("usd") or 0.0) for t in tries)
    if not empty:
        written, again = send_back(written, inputs, priced, model_id=model_id,
                                   ceiling_usd=max(ceiling_usd - spent, 0.0))
        spent += again
        written, again = idle_send_back(written, inputs, priced, model_id=model_id,
                                        ceiling_usd=max(ceiling_usd - spent, 0.0))
        spent += again
    rounds = sum(int(t.session.get("tool_rounds") or 0) for t in tries)
    writer = dict(written.counts, usd=spent, read_rounds=rounds, attempts=len(tries), empty_first=empty_first,
                  empty=empty)
    if len(tries) > 1:
        writer["first_attempt"] = dict(tries[0].counts, usd=float(tries[0].session.get("usd") or 0.0))
    spec = written.spec.model_copy(update={"version": _version(workdir, task_id) + 1, "writer": writer,
                                           "coherence": coherence or {}})
    aside = events.EMPTY_SPEC if empty else events.UNSATISFIABLE_SPEC if written.counts.get("unsatisfiable") else \
        UNCHECKED_END if written.counts.get(UNCHECKED_END) else None
    if aside:
        save_spec(workdir, spec.model_copy(update={"set_aside": aside}))
        events.task_set_aside(bus, task_id, aside, 0)
        return spent
    write_verifier(_idle_marked(spec, inputs), workdir, inputs)
    events.spec_written(bus, task_id, spec.version)
    return spent


def _idle_marked(spec: Any, inputs: Any) -> Any:
    """The Spec marked held_by_idle when its Verifier's code side holds on the Run that does nothing (D419)."""
    verifier = verifier_of(spec, inputs).verifier
    if not held_by_idle(spec, verifier, idle_run(spec.task_id, inputs.state), inputs.canon, inputs.write_tools):
        return spec
    return spec.model_copy(update={"writer": dict(spec.writer, **{HELD_BY_IDLE: True})})


def write_one(workdir: Path, task_id: str, model: Any, ceiling_usd: float, bus: Bus) -> float:
    """Mine one Task's Intent with its coherence record, then write its Spec from it (`write_from_intent`);
    return the spend. A Task whose user strayed from an opening ask (the miner's outcomes, checked by code)
    is saved set aside as intent_strayed with the record, task.set_aside published, no writer."""
    mined = mine_intent(workdir, task_id, model, ceiling_usd=ceiling_usd)
    coherence, spent = mined.coherence, mined.spent_usd
    if coherence.get("strayed"):
        spec = load_spec(workdir, task_id)
        save_spec(workdir, spec.model_copy(update={"coherence": coherence, "set_aside": events.INTENT_STRAYED}))
        events.task_set_aside(bus, task_id, events.INTENT_STRAYED, 0)
        return spent
    rest = max(ceiling_usd - spent, 0.0)
    return spent + write_from_intent(workdir, task_id, mined.intent, model, rest, bus, coherence=coherence)


def failure_line(error: Exception) -> str:
    """A failed Task's entry: the error class and the front of its message."""
    return f"{type(error).__name__}: {error}"[:FAILURE_CHARS]


def _restore(path: Path, before: Optional[bytes]) -> None:
    """Put the Spec file back as it was before a failed attempt, so a rerun writes the Task again (D421)."""
    if before is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(before)


def write_specs(workdir: Any, task_ids: Iterable[str], model: Any, *,
                ceiling_usd: Optional[float] = None, bus: Optional[Bus] = None) -> dict:
    """Write the Spec of every named Task that has none; counts of written, skipped, failed, set aside, and the spend."""
    workdir = Path(workdir)
    bus = bus or Bus(workdir / "bus.jsonl", agent="spec")
    ceiling = SPEC_TASK_CEILING_USD if ceiling_usd is None else ceiling_usd
    counts: dict = {"written": 0, "skipped_existing": 0, "failed": 0, "spent_usd": 0.0, "failures": {},
                    "empty_seen": 0, "retried": 0, "set_aside_empty": 0, "set_aside_unsatisfiable": 0,
                    "set_aside_strayed": 0, "set_aside_unchecked_end": 0, "rewritten_older": 0,
                    "effects_none_learned": 0, WRITES_UNGATED: 0, END_VALUES_UNGATED: 0, HELD_BY_IDLE: 0,
                    IDLE_SENT_BACK: 0}
    for task_id in task_ids:
        if written_already(workdir, task_id):
            counts["skipped_existing"] += 1
            continue
        counts["rewritten_older"] += int(older_grammar(load_spec(workdir, task_id)))
        path = spec_path(workdir, task_id)
        before = path.read_bytes() if path.is_file() else None
        try:
            counts["spent_usd"] += write_one(workdir, task_id, model, ceiling, bus)
        except Exception as error:  # one Task's failure is recorded; the others are still written
            _restore(path, before)
            counts["failed"] += 1
            counts["failures"][task_id] = failure_line(error)
            counts.setdefault("first_failure", counts["failures"][task_id])
            continue
        spec = load_spec(workdir, task_id)
        counts["empty_seen"] += int(bool(spec.writer.get("empty_first")))
        counts["retried"] += int(spec.writer.get("attempts", 1) > 1)
        counts["effects_none_learned"] += int(spec.writer.get(EFFECTS_COUNT) == NO_EFFECTS)
        counts[IDLE_SENT_BACK] += int(bool(spec.writer.get(IDLE_SENT_BACK)))
        for mark in (WRITES_UNGATED, END_VALUES_UNGATED, HELD_BY_IDLE):
            counts[mark] += int(bool(spec.writer.get(mark)))
        aside = {events.EMPTY_SPEC: "set_aside_empty", events.UNSATISFIABLE_SPEC: "set_aside_unsatisfiable",
                 events.INTENT_STRAYED: "set_aside_strayed", UNCHECKED_END: "set_aside_unchecked_end"}
        counts[aside.get(spec.set_aside, "written")] += 1
    return counts


# --- a Verifier an idle Run passes ships only with a gated judge or said item (D441) ---------------

# The end reason, and the writer's count naming the code gates the idle Run held when the Spec went back.
IDLE_RUN_PASSES = events.IDLE_RUN_PASSES
IDLE_SENT_BACK = "idle_sent_back"
IDLE_FIX = ("Where the right end writes, gate the change: a row_is whose expected value differs from the start, "
            "or a row_new. Where the right end writes nothing (an ask the policy refuses, a lookup), add a gated "
            "judge item asking what the right Run must look up and tell the user, from the policy and the user's "
            "words, or a gated said item with a value the Candidate must state. Where none can be named, change "
            "nothing and reply with one line starting \"rebut: cannot verify:\" and why.")


def _told_gate(atom: Any) -> bool:
    """A gated judge item, or a gated said item naming a value the Candidate must state."""
    target = atom.target or {}
    return bool(atom.gate) and (bool(atom.judge) or (target.get("kind") == "said" and bool(target.get("values"))))


def idle_held(verifier: Any, idle: Any, canon: Any = None, write_tools: Any = None,
              schema: Any = None) -> Optional[list[str]]:
    """The ids of the code gates the idle Run holds, when no code gate fails on it (judge items left unsettled)
    and the Verifier carries no gated judge or said item; else None. A Run the scorer cannot read shows nothing."""
    if any(_told_gate(atom) for atom in verifier.atoms):
        return None
    try:
        result = verdict(idle, verifier, canon, None, write_tools=write_tools, schema=schema)
    except Exception:  # a Run the scorer cannot read shows nothing either way
        return None
    gates = [item for item in result.items if item.gate and item.kind != "judge"]
    return None if any(item.holds is False for item in gates) else [item.id for item in gates]


def idle_ruling(spec: Any, held: list[str]) -> Ruling:
    """One ruling naming what the idle Run satisfies (every code gate, by its verdict id), ruling on the Spec's gated
    code items so the repair may replace them."""
    gated = [item["id"] for item in item_checks(spec) if item.get("gate") is True and item["kind"] != "judge"]
    named = ", ".join(held) or "none"
    return Ruling(check_ids=gated, kind=IDLE_RUN_PASSES, fix=IDLE_FIX,
                  reason=f"a Run that makes no call and says nothing holds every code gate ({named}) and no gated "
                         f"judge or said item stands, so a do-nothing agent passes the Task")


def idle_send_back(written: Written, inputs: Any, model: Any, *, model_id: Optional[str] = None,
                   ceiling_usd: Optional[float] = None) -> tuple[Written, float]:
    """(the Written after one send back for an idle Run holding every code gate, its spend); unchanged and 0
    when the idle Run fails a code gate or a gated judge or said item stands. The counts keep IDLE_SENT_BACK,
    the gates held, and the unchecked end and clashes as the repair left them; the stage's end judges again. A
    Spec already set aside for its clashes or an unchecked end is not sent back."""
    if written.counts.get("unsatisfiable") or written.counts.get(UNCHECKED_END):
        return written, 0.0
    idle = idle_run(written.spec.task_id, inputs.state)
    held = idle_held(verifier_of(written.spec, inputs).verifier, idle, inputs.canon, inputs.write_tools)
    if held is None:
        return written, 0.0
    fixed = repair(written.spec, idle_ruling(written.spec, held), inputs, model, model_id=model_id,
                   ceiling_usd=ceiling_usd)
    spec = fixed.spec.model_copy(update={"version": written.spec.version})
    counts = dict(written.counts, **{IDLE_SENT_BACK: held})
    counts.pop(UNCHECKED_END, None)
    if left := unchecked_end(spec):
        counts[UNCHECKED_END] = left
    counts["unsatisfiable"] = sorted({item_id for item_id, _ in contradictions(item_checks(spec))})
    return Written(spec, counts, fixed.session), float(fixed.session.get("usd") or 0.0)


# --- the stage's end: a Task no right Run can pass ships no Verifier (D439) -------------------------

# A gated called write whose policy precondition names a value the user must give, and no rendered fact gives it.
PRECONDITION_UNRENDERED = events.PRECONDITION_UNRENDERED
# The Task has References and none holds every gate of its Verifier: the Reference vetoes, never sources.
NO_REFERENCE_HOLDS = events.NO_REFERENCE_HOLDS
ENDED_COUNTS = {PRECONDITION_UNRENDERED: "set_aside_precondition_unrendered",
                NO_REFERENCE_HOLDS: "set_aside_no_reference_holds", IDLE_RUN_PASSES: "set_aside_idle_run_passes"}
# A policy line that binds before a write: it says must (or needs, has to, is required) ...
MUST_RE = re.compile(r"\b(?:must|needs?|has to|have to|required)\b")
# ... and has the user give a value: the user provides, gives or supplies it, or it is obtained from the user.
USER_GIVES_RE = re.compile(r"\b(?:user|customer)\b[^.]*\b(?:provides?|gives?|supply|supplies)\b"
                           r"|\bfrom the (?:user|customer)\b")
# Where a policy line splits into clauses: a comma, a semicolon, a colon or a full stop.
CLAUSE_RE = re.compile(r"[,;:.]\s")
# The words naming who gives the value, left out before a field is looked for in the line.
GIVER_RE = re.compile(r"\b(?:from )?the (?:user|customer)\b")


def _words(text: str) -> set[str]:
    """The content words of a name or a heading, each in one spelling (spec/text.py)."""
    return {normalise(word) for word in TOKEN_RE.findall(str(text).casefold()) if word not in STOPWORDS}


def tool_lines(policy: str, tool: str) -> list[str]:
    """The policy lines that speak of the tool: a line naming it (items.names_tool, as the writer reads a
    line for a tool), or any line under a heading sharing a word with the tool's name."""
    out, heading = [], set()
    for line in (policy or "").splitlines():
        if line.startswith("#"):
            heading = _words(line.strip("# "))
            continue
        if line.strip() and (names_tool(line, tool) or heading & _words(tool.replace("_", " "))):
            out.append(line.strip())
    return out


def given_fields(line: str, fields: Iterable[str]) -> list[str]:
    """The fields a policy line says the user must give before the write: a clause of the line binds
    (MUST_RE) and has the user give a value (USER_GIVES_RE), and that clause names the field, underscores read
    as spaces. A field named in another clause ("if the field changes, ...") is not the one given."""
    found: set[str] = set()
    for clause in CLAUSE_RE.split(" ".join(line.split()).casefold()):
        if MUST_RE.search(clause) and USER_GIVES_RE.search(clause):
            said = GIVER_RE.sub(" ", clause)
            found |= {field for field in fields
                      if re.search(rf"\b{re.escape(field.replace('_', ' ').casefold())}\b", said)}
    return sorted(found)


def tool_fields(inputs: Any, tool: str) -> set[str]:
    """What the tool's schema names: its arguments, and the columns of the tables it writes: those its learned
    effects write, and those holding a column named as one of its arguments (the row an argument names)."""
    args = set(next((row.get("args") or () for row in inputs.tools if row.get("name") == tool), ()))
    world = World(inputs.state, columns=inputs.columns)
    tables = {write.get("table") for write in (inputs.effects.get(tool) or {}).get("writes") or ()
              if isinstance(write, dict)}
    tables |= {table for table in world.state if args & set(world.columns(table))}
    return args | {field for table in tables if table for field in world.columns(table)}


def _field_values(state: dict, field: str) -> set[str]:
    """Every value the world holds in a field of that name, on any table, as comparable text."""
    out: set[str] = set()
    for rows in state.values():
        for row in (rows.values() if isinstance(rows, dict) else ()):
            value = row.get(field) if isinstance(row, dict) else None
            out.update(norm(v) for v in (value if isinstance(value, list) else [value])
                       if isinstance(v, (str, int)) and norm(v))
    return out


def unrendered_field(spec: Any, inputs: Any) -> Optional[dict]:
    """The first gated called write whose policy precondition names a field the user must give while no
    rendered fact carries a value the world holds in it: {"tool", "field", "line"}; else None. A field
    the world holds no value of is never asked of the facts: nothing tells whether a fact gives it."""
    said = [_flat(fact.text) for fact in spec.intent.facts if rendered(fact)]
    for item in item_checks(spec):
        tool = str(item.get("tool") or "")
        if item.get("kind") != "called" or tool not in inputs.write_tools \
                or item.get("gate", DEFAULT_GATE["called"]) is not True:
            continue
        for line in tool_lines(inputs.policy_text, tool):
            for field in given_fields(line, tool_fields(inputs, tool)):
                values = _field_values(inputs.state, field)
                if values and not any(_contains(text, value) for text in said for value in values):
                    return {"tool": tool, "field": field, "line": line}
    return None


def reference_holds(verifier: Any, references: Iterable[Any], *, canon: Any = None, write_tools: Any = None,
                    schema: Any = None, judged: Optional[dict] = None) -> Optional[bool]:
    """Whether any Reference Run holds every gate of the Verifier (a gate left open is not failed); None
    with no Reference Run."""
    runs = list(references)
    if not runs:
        return None
    judged = judged or {}
    return any(_reference_outcome(verifier, run, canon, write_tools, schema, judged.get(run.run_id)).failed
               is None for run in runs)


def _reference_outcome(verifier: Any, run: Any, canon: Any, write_tools: Any, schema: Any,
                       judged: Optional[dict]) -> Outcome:
    try:
        return outcome_of(verifier, run, canon, write_tools, schema=schema, judge_results=judged)
    except Exception:  # a Run the scorer cannot read neither holds nor fails: it vetoes nothing
        return Outcome(False, None, False)


def end_task(workdir: Path, task_id: str, reason: str, detail: dict, bus: Bus) -> None:
    """Set the Spec aside for `reason` with what decided it, remove its Verifier files, tell the bus."""
    spec = load_spec(workdir, task_id)
    save_spec(workdir, spec.model_copy(update={"set_aside": reason,
                                               "writer": dict(spec.writer, ended=dict(detail, reason=reason))}))
    for path in (spec_verifier_path(workdir, task_id), runner_verifier_path(workdir, task_id)):
        path.unlink(missing_ok=True)
    events.task_set_aside(bus, task_id, reason, spec.round)


class _Shared(NamedTuple):
    """What every Task's Reference check reads off the workdir once."""
    references: dict
    runs: dict
    canon: Any
    write_tools: set
    schema: Any


def _shared(workdir: Path) -> _Shared:
    sigs = read_json(workdir / "tool_sigs.json", None) or []
    tools = write_tools_of(sigs.get("sigs", []) if isinstance(sigs, dict) else sigs) - workdir_action_tools(workdir)
    return _Shared(faithful_references(workdir), runs_on_disk(workdir), load_rules(workdir / "canon-rules.json"),
                   tools, schema_of(workdir))


def end_of(workdir: Path, task_id: str, shared: _Shared) -> Optional[tuple[str, dict]]:
    """(why the Task ends with no Verifier, what decided it), or None for a Task that keeps its Verifier:
    a policy precondition unrendered first, then an idle Run holding every code gate with no told gate (D441),
    then no Reference holding."""
    spec, verifier = load_spec(workdir, task_id), load_spec_verifier(workdir, task_id)
    if spec is None or verifier is None or spec.set_aside is not None:
        return None
    inputs = load_inputs(workdir, task_id, spec.intent)
    missing = unrendered_field(spec, inputs)
    if missing:
        return PRECONDITION_UNRENDERED, missing
    held = idle_held(verifier, idle_run(task_id, inputs.state), shared.canon, shared.write_tools, shared.schema)
    if held is not None:
        return IDLE_RUN_PASSES, {"held": held}
    kept = set(shared.references.get(task_id) or ())
    runs = [run for run in shared.runs.get(task_id, ()) if run.run_id in kept]
    judged = judge_results_of(workdir, task_id, verifier, runs) if runs else {}
    holds = reference_holds(verifier, runs, canon=shared.canon, write_tools=shared.write_tools,
                            schema=shared.schema, judged=judged)
    return (NO_REFERENCE_HOLDS, {"references": sorted(kept)}) if holds is False else None


def end_unpassable(workdir: Any, task_ids: Iterable[str], *, bus: Optional[Bus] = None) -> dict:
    """After compile and the Examiner, end each named Task no right Run can pass with no Verifier (D439):
    a gated write's policy precondition the instruction never renders (precondition_unrendered), an idle
    Run holding every code gate with no gated judge or said item (idle_run_passes, D441), or
    References none of which holds every gate (no_reference_holds). A Task with no Reference keeps today's
    path. Returns the Tasks checked and the counts per reason, beside write_specs' set_aside counts."""
    workdir = Path(workdir)
    bus = bus or Bus(workdir / "bus.jsonl", agent="spec")
    shared = _shared(workdir)
    counts: dict = {"checked": 0, **{name: 0 for name in ENDED_COUNTS.values()}, "ended": {}}
    for task_id in task_ids:
        counts["checked"] += 1
        ended = end_of(workdir, task_id, shared)
        if ended is None:
            continue
        reason, detail = ended
        end_task(workdir, task_id, reason, detail, bus)
        counts[ENDED_COUNTS[reason]] += 1
        counts["ended"][task_id] = reason
    return counts


__all__ = ["ENDED_COUNTS", "FAILURE_CHARS", "IDLE_RUN_PASSES", "IDLE_SENT_BACK", "NO_REFERENCE_HOLDS", "PRECONDITION_UNRENDERED", "SPEC_TASK_CEILING_USD",
           "end_of", "end_task", "end_unpassable", "failure_line", "given_fields", "idle_held", "idle_ruling",
           "idle_send_back", "older_grammar", "reference_holds",
           "rendered", "tool_fields", "tool_lines", "unrendered_field", "write_from_intent", "write_one",
           "write_specs", "written_already"]
