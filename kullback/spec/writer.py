"""The Verifier writer: items from the Intent, the policy and the world, never from a Reference.

One model session reads the Intent facts, the policy, the tool list and, through the world tools
(`writer_tools.WORLD_TOOLS`), the Starting state and the policy sections; it writes typed items
(spec/items.py: state, event, text and judge items), each with gate, weight and provenance (the Intent
facts it answers or a verbatim policy line). It never sees a Run, a Reference, a Verifier or an End
state: the inputs come from the Environment's own files and `writer_tools.refuse_run_inputs` refuses
anything else. Nothing here reads the Reference index (spec/witness.py does, for the witness flag).

Code decides what stands, in the tool: an item naming a value no world row, policy line or user turn
holds is refused with the reason, as is an item with no provenance or resting on unwitnessed facts
alone, so the writer fixes it in the same session. Items no Run can pass together go back once in a
follow-up turn, as do asks no gate item cites; for those, and in the repair of an unchecked_ask ruling, the
writer may call rebut_fact, which code keeps only when its why quotes the fact's own words or a policy
line (D376). The Spec records each kept rebuttal by fact id; nothing else excuses an ask. Code adds the
sanity item. Facts no item names become gaps. `repair` answers an
Examiner ruling (D331): it applies the ruling's fix (drop or replace a ruled item, add items for an
uncovered fact) or rebuts it with why, under the same rules; when code refused every answer it gets
one retry in the same round, shown each refusal.

A Task born of a scenario carries the scenario's end (D414): born/<task>.json in the workdir, written by
synth/tasks.py, holds the end, what a correct agent leaves the world as and tells. `load_inputs` adds it
to the Intent as end facts, one per clause, marked end and never a user turn; each is an ask, so one no
gated item cites goes back like any ask, and a Spec still leaving one uncited and unrebutted is refused
(`unchecked_end`): the Spec stage sets it aside with no Verifier. A recorded Task has no end.

Every session (a first write, each repair) is saved whole under exam/writer/<task>.jsonl, one line per
session: the messages, the tool calls and answers, the replies (D351). Nothing the writer is shown is read
from there.

A Spec of the older demand grammar still compiles to atoms (`compile_spec`, `must_not`) and gates on
nothing else; the Spec stage rewrites it as items.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional

from pydantic import Field

from kullback.runner.records import Record, Verifier, read_json
from kullback.spec import effects as tool_effects
from kullback.spec import events, writer_tools
from kullback.spec import items as I
from kullback.spec.actions import action_tables_of, action_tools_of
from kullback.spec.compile import Compiled, compile_spec, merged_atoms, merged_cells
from kullback.spec.ground import coverage
from kullback.spec.intent_tools import mark_echo, mark_side, recordings_of, task_of, trace_index
from kullback.spec.must_not import must_not
from kullback.spec.schema import (
    Check,
    FactSource,
    IntentFact,
    Spec,
    SpecIntent,
    intent_text,
    is_reaction,
    load_spec,
    save_spec,
    spec_verifier_path,
    unchecked_asks,
)

# The ruling code for an ask no gate item cites (spec/mutant_rulings.py); its repair may call rebut_fact.
UNCHECKED_ASK = "unchecked_ask"

WHAT_YOU_RECEIVE = """\
# What you receive
One customer request as Intent facts (each with an id; a fact marked unwitnessed was only accepted by \
the user, never volunteered; a fact marked echo carries the agent's proposal, so a gate value may not \
rest on it alone; a fact marked unfounded was ruled never asked for, so no gate may rest on it alone; a \
fact's condition says what prompted it, with the agent line it answers quoted), the policy the agent works \
under, the tools the agent may call, and \
read access to the world as it stands before the agent starts. You have never seen any agent run or \
reference solution for this request, and you must not ask for one. Your job is the Verifier: the \
items a correct run shows, each resting on an Intent fact or a policy line. A Task born of a scenario \
also carries the scenario's end as facts marked end: the truth the Task was written with, what a correct \
agent leaves the world as and what it tells. The user never said them."""

TOOLS_TEXT = """\
# Tools
- read_schema: tables with their fields and row counts, tools with their argument names.
  Example: read_schema({})
- find_rows: rows holding a value, in one field or any field, one table or all.
  Example: find_rows({"value": "the id the user gave", "table": "records"})
- read_row: one row by table and key. Example: read_row({"table": "records", "key": "R-1"})
- read_policy: the policy headings, or one section's text. Example: read_policy({"section": "Changes"})
- add_items: add items; each comes back kept (with the row it resolved) or refused with the reason.
  Example: add_items({"items": [{"id": "s1", "kind": "row_is", "table": "records", \
"find": {"record_id": "R-1"}, "expect": {"status": "closed"}, "gate": true, "fact_ids": ["f1"]}]})
- drop_items: remove items you added. Example: drop_items({"ids": ["s1"]})
- rebut_fact: only for a fact code sent back as missing its gate: why it needs none, copying the user's \
words of that fact or a policy line word for word. Example: rebut_fact({"fact_id": "f2", "why": \
"the policy says: <the policy line, copied>"})

# Item kinds
- row_is {table, find: {field: value}, expect: {field: value | [values] | {"one_of": [...]} | {"not": [...]}}, \
free: {field: why}}: the row find names (exactly one) holds these values after the run. A list value names \
every member, in any order. Use one_of where the policy or the user allows several values, not where a \
value is forbidden. Put in free, each with one line on why, the fields of that row a correct run changes to a value you \
cannot find (records a tool fills in, a balance it computes); code checks nothing in them but lets them \
move. A value the user names goes in expect, never in free; expect names more than the find key.
- row_new {table, where: {field: value}, count}: the run makes count new rows holding these values.
- row_keeps {table, find, fields: [...]}: these fields of that row must not change (one per "do not change").
- called {tool, args?} / not_called {tool, args?}: a call that must or must not happen.
- before {first: "confirm_turn" | tool, then: tool}: the user confirms (or a tool is called) before then.
- said {values: [...], not_values: [...]}: short values the agent must (not) state to the user.
- judge {question, anchor: {answer, accepted: [...], reject: [...], row: {table, find, fields: [...]}}, \
evidence: [...]}: one yes or no question about what the agent told the user. Find the anchor's row with \
find_rows first, cite it in row, and state its fields in answer; only a refusal anchor that quotes the \
item's policy_line goes without a row. Leave evidence out: every agent turn up to the first transfer is \
read. Narrow it ("after_call:<tool>", "final_answer", "assistant:<k>") only with evidence_why, one line on \
why the answer can come nowhere else.
Every item: id, gate (true when a run failing it is wrong whatever else it does), weight (default 1), \
and fact_ids or policy_line (a verbatim line of the policy, at least 12 characters). A hand-off item \
gates only when its policy_line demands the hand-off. An item may carry confirms {table, find}: the one \
row whose read confirms a condition the agent claimed about the world. When the Intent or the policy names a condition \
(if this, do that; else the other), write each branch as its own end state: its state items carry alt \
(0, 1, ...); an item without alt holds in all."""

EXAMPLES = """\
# Examples
Placeholders in angle brackets stand for the Task's own names; one example per item kind and per condition.
- row_is: the user asks to set <field> of <record> to <value> (do, condition none): row_is on <record> with \
<field> <value>, gated, citing that fact.
- row_keeps: the user asks that <other record> stay as it is (keep): row_keeps on <other record>, gated, \
citing that fact.
- row_new: the user wants a new <entry> with a minted id: row_new with the fields the user gave, never the id.
- before: the policy says <write> needs the user's yes first: before {first: "confirm_turn", then: \
"<write tool>"} (gate, policy_line quoting that rule).
- called: the policy says <step> must be done through <tool>: called <tool> (gate, policy_line).
- not_called: the policy forbids <action> and the user asks for it: not_called <tool> (gate, policy_line), \
no state item (code already fails any change), and one gated judge item asking whether the agent \
declined, the anchor quoting the policy line.
- judge: the user asks to be told <value> of <record> (told): a gated judge item whose anchor is <value> \
read from the row that holds it (row cites that row). Never compute a figure the world does not hold.
- said: the same told <value> stored as is: said with <value>, gated, citing the told fact.
- one_of: the user asks for one of several <options> the policy allows and does not say which: the item \
accepts every allowed one (one_of), never only the one an agent might pick.
- condition world_or_policy: the user asks for <other change> if, as the agent said, <change> is not \
allowed: look the claim up (read_row, or read_policy) and gate the branch the world selects, citing the \
row in confirms or the rule in policy_line.
- condition agent: the user asks for <other change> only after the agent could not do <change>: any item \
on it is scored, never gated."""

CHOICE_RULE = """\
# Choosing items
Write an item only for what the Intent asks for or the policy requires, never for a sensible step, and \
one item per ask. Every value comes from a world row you read, a policy line or the user's words: find it \
first, then add it. Code adds the sanity item by itself (no row outside your state items may change), so \
list every row a correct run changes, and only those. Gate what the user asked for and the policy requires. \
Every gate rests on an ask fact or on a policy_line that states the rule. A rule the policy does not state is \
never gated, however sensible: weight the item, and where the Task turns on it, leave the fact a gap. \
Where \
the user left a choice open and the policy allows several, accept any allowed one or weight it. A forbid \
(not_called, row_keeps) gates only on a keep fact the user said or the policy line that forbids it; else weight it. A value the user \
asked to be told is a gated judge item anchored on the world row holding it. One of several options the \
world holds that the user named is a gated check on that field. A refusal judge item always gates. Any other told item (judge, \
said) gates only citing a told fact or quoting the policy line that requires the telling: code weighs \
any other. A fact marked side asks only to be \
told something beside the Task's ask, so any item on it is scored, never gated: code weighs such a gate. \
A fact's condition says what prompted it. A fact of condition agent is a reaction to the agent's own \
conduct (it could not, declined citing no rule, stalled, or proposed another way): any item on it is \
scored, never gated. A fact of condition world_or_policy is an ask under a condition the agent claimed \
about the world or the rules: the claim is the agent's until you confirm it. Look it up yourself, the \
row with find_rows and read_row (cite it in confirms) or the rule with read_policy (cite it in \
policy_line), and gate the branch the world selects, not the branch the recording took; a claim you \
cannot confirm leaves the item scored. A called gate stands only where its policy_line names that tool as \
the only way; code weighs any other. Gate who the user is by the outcome (the writes on the rows of the \
user the agent found), never by which lookup tool found them.
One judge question asks one thing, and never names the answer it wants.
Gate only a value the user asked for or the policy demands; weigh any other."""

END_RULE = """\
# The scenario's end
Facts marked end are the truth this Task was written with. Every change an end fact names is a gated \
state item (row_is or row_new) on the row and field it names, with the value it names, citing that fact. \
Every action or change an end fact rules out is gated absent: not_called or row_keeps, citing it. Every \
value an end fact says the user is told is a gated judge or said item citing it. A gate no end fact names \
and no policy line demands is weighed. Where an end fact says the change happens once the user confirms, \
the user will say yes: gate the change (and a before item on confirm_turn where the policy asks for the \
yes), never a not_called on it. Where the end has the user pick one of several writes, write each as its \
own end state (alt 0, 1, ...), never both gated in one; where alt cannot hold them, gate neither and add \
one gated judge item on the outcome. A write's gate expects a value other than the start row's and carries \
every number, id or quoted value the end names of the changed row; a before or called item never gates a write. \
An end fact that needs no item of its own (a step on the way, what \
the user does) goes to rebut_fact, copying its words. Code refuses a Spec with an end fact no gated item \
cites and no rebuttal excuses."""

FEEDBACK = """\
# Feedback
add_items answers each item: kept, or refused with the reason (a value not found, a find matching no \
row or several, a field the table lacks, no provenance). Fix a refused item and add it again under its \
id, or leave it out. A gated value must be one a tool can write or the user fixed: when code names \
reachable values for a field, gate one of them or weigh the item."""

STOP_RULE = """\
# Stop
When every Intent fact has its items (or none fits), reply with one line saying you are done, no tool \
call. Spend at most 14 tool rounds."""

SYSTEM_PROMPT = "\n\n".join((WHAT_YOU_RECEIVE, TOOLS_TEXT, EXAMPLES, CHOICE_RULE, END_RULE, FEEDBACK, STOP_RULE)) + "\n"

FOLLOWUP_PROMPT = """Code found items no run can pass together, each with the reason. Drop or re-add \
them under their ids so a correct run can pass, then reply that you are done.
"""

# Facts the user raised (not reactions) no gated item covers go back in one follow-up turn of the same conversation.
ASKED_FOLLOWUP_PROMPT = """Code lists facts the user raised that no gated item covers, each with its id and why \
(uncovered, or weighted only). Add one gated item for each, or call rebut_fact with its id and why it needs \
none, copying the user's words of that fact or a policy line word for word; nothing else excuses it. Then \
reply that you are done.
"""


def _gates_alone(fact: IntentFact) -> bool:
    """A fact a gate may rest on alone: raised by the user (volunteered, or a conditional ask), not a
    reaction, not unfounded, not side."""
    return fact.stance != "accepted" and not (is_reaction(fact) or fact.unfounded or fact.side)


def asked_missing(items: Iterable[dict], intent: SpecIntent) -> list[dict]:
    """Facts the user raised (not a reaction, not unfounded, not side) no gated item covers.

    Each row names the fact id and why it is missing: uncovered (no item names it) or
    weighted only (named only by scored items). A gate counts as set where the item says
    so, else by the kind default, as code keeps it. Accepted facts and reactions are out of
    scope here by design (D366).
    """
    rows = list(items or [])
    named = {fact_id for item in rows for fact_id in item.get("fact_ids") or []}
    gated = {fact_id for item in rows
             if item.get("gate", I.DEFAULT_GATE.get(item.get("kind"), False)) is True
             for fact_id in item.get("fact_ids") or []}
    out = []
    for fact in intent.facts:
        if not _gates_alone(fact):
            continue
        if fact.id in gated:
            continue
        out.append({"id": fact.id, "reason": "uncovered" if fact.id not in named else "weighted only"})
    return out

REPAIR_PROMPT = """The Examiner ruled on items you wrote for this customer request. A ruling names \
one item (an item id, an Intent fact id, or the task), what is wrong and a fix. You still have never \
seen any agent run, and you must not ask for one. Every rule of the writer still holds.
Apply: fix what the ruling names in the ruled items (a value, a gate, an anchor), or add the items it says \
are missing; you may not drop or replace any other item. Drop a ruled item only when it asks for \
something the user never asked for, and say so. Cover a missing fact in the fact_ids of a ruled item \
that checks it, or with one new item for what no item checks; never add a copy of an item.
Rebut: when the user's words or a policy line require the items as written, change nothing and reply \
with one line starting "rebut:" that quotes those words or that line. With neither, apply the ruling. \
A reply that changes nothing always starts with "rebut:", also when the ruling asks for no item change.
When the only fix would loosen a check so the recorded run passes, apply nothing and reply with one line \
"rebut: cannot verify: <why>", quoting the user's words or the policy line the check rests on.
An unchecked_ask ruling names a fact no gated item cites: add a gated item citing it, or call rebut_fact \
with that fact id and why it needs none, copying the user's words of that fact or a policy line word for \
word. A "rebut:" reply alone leaves the fact unchecked.
Reply that you are done when finished.

"""

# Why a demand carrying code is refused (D320): what a Run must write is the end state's cells.
WRITE_PREDICATE = "write demands are end-state cells"

# The longest message body a session is sent, about 6000 tokens: it bounds one call's input cost
# whatever the policy or tool list holds (held-out bodies ran 12000 to 18000 characters).
BODY_CHARS = 24000
# The longest ruling excerpt a repair session is shown: a quote, never a transcript.
MAX_EXCERPT = 1000
RETRY_PROMPT = """Code refused every item you sent, so nothing changed. Each refusal and its reason:
{refusals}
Send the items again with the refusals fixed, or reply with one line starting "rebut:" and why.
"""


class WriterInputs(NamedTuple):
    """Everything one writer session may see; none of it comes from a Run or a Reference."""
    intent: SpecIntent
    policy_text: str
    sections: list[str]
    tools: list[dict]
    write_tools: set[str]
    state: dict
    constraints: list[dict]
    canon: Callable[[Any], Any]
    # The tools the world records as actions (spec/actions.py): never in write_tools.
    action_tools: frozenset = frozenset()
    # The tables those actions are recorded in: a call's record, never a change the sanity item reads.
    action_tables: frozenset = frozenset()
    # The schema's column names per table, for tables the Starting state holds no row of.
    columns: dict = {}
    # What each write tool always moves and what each tool's recorded calls passed (spec/effects.py).
    effects: dict = {}
    signatures: dict = {}
    # What tools can write per field (D387): the learned writes and the Task's recorded write results.
    reach: Optional[I.Reach] = None
    # Where the sessions are saved (exam/writer); None saves nothing. Never shown to the writer.
    workdir: Optional[Path] = None
    # The workdir has write tools but no learned writes (no effects file, or one where no tool carries writes),
    # so no learned tool write can reach a value (D417).
    effects_missing: bool = False


class Ruling(Record):
    """What a repair answers: the ruled check ids, the item ruled on, the reason and the fix (D331)."""
    check_ids: list[str]
    reason: str
    excerpt: str = ""
    item: str = ""
    fix: str = ""
    kind: str = ""
    # The fact ids an unchecked_ask ruling filed by code names (its verified rows, D376).
    facts: list[str] = Field(default_factory=list)


# The writer count and the Spec's reason when the Spec was written without learned effects (D417).
EFFECTS_COUNT = "effects"
NO_EFFECTS = "none learned"


class Written(NamedTuple):
    spec: Spec
    counts: dict
    session: dict


def load_inputs(workdir: Path, task_id: str, intent: Optional[SpecIntent] = None) -> WriterInputs:
    """The Environment's own files for one Task: policy, tool schemas, Starting state, constraints.

    The policy is the recorded agent's own system prompt, else the Environment's policy.md. The
    Starting state is the shared world with the Task's overlay laid over it, never a Run's record.
    An action tool (spec/actions.py) is kept out of the write tools. No Reference is read.
    """
    from kullback.runner.state import StateView
    from kullback.runner.target import canon_fn
    from kullback.runner.world.environment import BuiltEnvironment

    env = BuiltEnvironment(workdir)
    traces = trace_index(workdir)
    if intent is None:
        spec = load_spec(workdir, task_id)
        if spec is None:
            raise FileNotFoundError(f"no Spec for Task {task_id}: mine its Intent first")
        intent = spec.intent
    facts = mark_side(mark_echo(intent.facts, recordings_of(workdir, task_id, traces)))
    end = scenario_end(workdir, task_id)
    intent = with_end(with_records(intent.model_copy(update={"facts": facts}), record_facts(workdir, task_id) if end
                                   else []), end)
    policy = env.system_prompt(env.task(task_id)) or ""
    overlay, rows = env.overlay(task_id)
    tools = writer_tools.tool_rows(env.sigs)
    actions = frozenset(action_tools_of(env.schema))
    learned = tool_effects.load(workdir)
    effects = learned.get("effects") or {}
    write_tools = {row["name"] for row in tools if row["kind"] == "write"} - actions
    tools = [dict(row, writes=_writes_view(effects[row["name"]])) if _writes_view(effects.get(row["name"]))
             else row for row in tools]
    return WriterInputs(intent=intent, policy_text=policy, sections=writer_tools.policy_sections_of(workdir),
                        tools=tools, write_tools=write_tools,
                        state=StateView(env.db, overlay, rows).shared, constraints=_constraints(workdir),
                        canon=canon_fn(env.canon_rules), action_tools=actions,
                        action_tables=frozenset(action_tables_of(env.schema)), columns=_columns(env.schema),
                        effects=effects, signatures=learned.get("signatures") or {},
                        reach=I.reach_of_inputs(effects, _write_results(workdir, task_id, traces, write_tools)),
                        workdir=Path(workdir), effects_missing=bool(write_tools) and not (
                            learned.get("found") and any(row.get("writes") for row in effects.values())))


# Where a born Task's scenario end lies in the workdir, one file per Task (synth/tasks.py writes it, D414).
BORN_DIR = "born"
# Why the Spec stage sets a Spec aside: an end fact no gated item cites and no rebuttal excuses (events.py).
UNCHECKED_END = events.UNCHECKED_END


def born_path(workdir: Any, task_id: str) -> Path:
    return Path(workdir) / BORN_DIR / f"{task_id}.json"


def scenario_end(workdir: Any, task_id: str) -> str:
    """The end of the scenario a Task was born of, or empty for a recorded Task."""
    body = read_json(born_path(workdir, task_id), None) if "/" not in str(task_id) else None
    return str(body.get("end") or "").strip() if isinstance(body, dict) else ""


def end_facts(end: str) -> list[IntentFact]:
    """The end as facts, one per clause (split at a full stop or a semicolon), ids end1, end2, ...: asks of
    kind do, said by no user turn, their source the end itself (items.END_SOURCE)."""
    clauses = [clause.strip() for clause in re.split(r"(?<=[.;])\s+", end or "") if clause.strip()]
    return [IntentFact(id=f"end{n}", text=clause, source=FactSource(recording=I.END_SOURCE, turn=0),
                       stance="volunteered", kind="do") for n, clause in enumerate(clauses, 1)]


def with_end(intent: SpecIntent, end: str) -> SpecIntent:
    """The Intent with the end facts it lacks appended; unchanged with no end or with them already there."""
    held = {fact.id for fact in intent.facts}
    added = [fact for fact in end_facts(end) if fact.id not in held]
    if not added:
        return intent
    facts = list(intent.facts) + added
    return intent.model_copy(update={"facts": facts, "text": intent_text(facts)})


def record_facts(workdir: Any, task_id: str) -> list[IntentFact]:
    """The bound records a born Task's user gives when asked (synth/tasks.py, D416), as facts the user says: the
    facts of its recordings' user rules that no user turn holds (no span) and that are given on request, ids
    rec1, rec2, ..., their source items.RECORDS_SOURCE. They are details, not asks, and never end facts."""
    root = Path(workdir)
    task = read_json(root / "tasks" / f"{task_id}.json", None) if "/" not in str(task_id) else None
    said: list[str] = []
    for run_id in (task.get("run_ids") or ()) if isinstance(task, dict) else ():
        rules = read_json(root / "user_rules" / f"{run_id}.json", None) if "/" not in str(run_id) else None
        if not isinstance(rules, dict):
            continue
        asked = {rule.get("field") for rule in rules.get("disclosure") or () if rule.get("on_request")}
        said += [f"{fact['field']}: {fact.get('value')}" for fact in rules.get("facts") or ()
                 if fact.get("span") is None and fact.get("field") in asked]
    return [IntentFact(id=f"rec{n}", text=text, source=FactSource(recording=I.RECORDS_SOURCE, turn=0),
                       stance="volunteered", kind="detail", condition="asked") for n, text in enumerate(dict.fromkeys(said), 1)]


def with_records(intent: SpecIntent, records: list[IntentFact]) -> SpecIntent:
    """The Intent with the record facts it lacks appended (D416); unchanged with none to add."""
    held = {fact.id for fact in intent.facts}
    added = [fact for fact in records if fact.id not in held]
    if not added:
        return intent
    facts = list(intent.facts) + added
    return intent.model_copy(update={"facts": facts, "text": intent_text(facts)})


def unchecked_end(spec: Spec) -> list[str]:
    """The end facts no gate item cites and no kept rebuttal excuses (`schema.unchecked_asks` named for the
    end): the Spec of a born Task leaving one is refused."""
    unchecked = set(unchecked_asks(spec))
    return [fact.id for fact in spec.intent.facts if I.is_end_fact(fact) and fact.id in unchecked]


def _writes_view(learned: Optional[dict]) -> dict:
    """What a write tool writes, as the writer's tool list shows it: {table.field: its literal values, or
    "from <argument>" where it copies what the call passed}."""
    return {f"{row['table']}.{row['field']}": row["values"] if row.get("values") else
            f"from {', '.join(row['from_args'])}" for row in (learned or {}).get("writes") or []
            if row.get("values") or row.get("from_args")}


def _write_results(workdir: Path, task_id: str, traces: dict, write_tools: set) -> list[tuple[str, str, Any]]:
    """(recording, tool, result) of every successful write call in the Task's recordings (D387): what a
    tool wrote, read for the values it shows, never for the path the recording took."""
    out = []
    for run_id in task_of(workdir, task_id).run_ids:
        trace = traces.get(run_id)
        for call in trace.tool_calls if trace is not None else ():
            if call.name in write_tools and call.error is None and call.result is not None:
                out.append((run_id, call.name, call.result))
    return out


def _columns(schema: Any) -> dict:
    out: dict = {}
    for column in (schema.get("columns") if isinstance(schema, dict) else getattr(schema, "columns", None)) or ():
        table, name = (column.get("table"), column.get("name")) if isinstance(column, dict) else (
            getattr(column, "table", None), getattr(column, "name", None))
        if table and name:
            out.setdefault(str(table), []).append(str(name).split(".")[0])
    return {table: sorted(set(names)) for table, names in out.items()}


def _constraints(workdir: Path) -> list[dict]:
    """The compiled policy constraints the Examiner kept, or none."""
    path = Path(workdir) / "constraints_check.json"
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("constraints") or [])
    except (OSError, ValueError, AttributeError):
        return []


def unwitnessed(fact: IntentFact) -> bool:
    """An accepted fact no other recording volunteers (DESIGN.md rule 5)."""
    return fact.stance == "accepted" and not fact.witnesses


def _condition_view(fact: IntentFact) -> dict:
    """A fact's condition and the agent line it answers, when it has either."""
    out = {"condition": fact.condition} if fact.condition != "none" else {}
    return dict(out, answers=fact.in_reply_to.quote) if fact.in_reply_to else out


def _facts_view(intent: SpecIntent) -> list[dict]:
    return [dict({"id": fact.id, "text": fact.text, "stance": fact.stance}, **_condition_view(fact),
                 **({"unwitnessed": True} if unwitnessed(fact) else {}), **({"echo": True} if fact.echo else {}),
                 **({"after_pushback": True} if fact.after_pushback else {}),
                 **({"unfounded": True} if fact.unfounded else {}), **({"side": True} if fact.side else {}),
                 **({"end": True} if I.is_end_fact(fact) else {}))
            for fact in intent.facts]


def _body(inputs: WriterInputs) -> dict:
    return {"intent": _facts_view(inputs.intent), "tools": inputs.tools,
            "policy_sections": json.loads(world_of(inputs).policy_section()) or "no policy sections",
            "policy_text": inputs.policy_text or "no recorded agent policy found"}


def _message(prompt: str, body: dict) -> tuple[list[dict], int]:
    """The one user message, its body cut at BODY_CHARS; the count of characters cut, 0 when whole."""
    text = json.dumps(body)
    cut = max(len(text) - BODY_CHARS, 0)
    tail = "\n(inputs truncated; read_policy shows any section whole)" if cut else ""
    return [{"role": "user", "content": prompt + "\n" + text[:BODY_CHARS] + tail}], cut


def writer_messages(inputs: WriterInputs) -> tuple[list[dict], int]:
    return _message(SYSTEM_PROMPT, _body(inputs))


def world_of(inputs: WriterInputs) -> I.World:
    """The world the writer's tools read and its values are sourced from."""
    return I.World(inputs.state, inputs.policy_text, inputs.intent.facts, inputs.tools, inputs.columns,
                   inputs.signatures, inputs.reach)


def store_of(inputs: WriterInputs, items: Iterable[dict] = (), ruled: Optional[set[str]] = None,
             rebuttals: Optional[dict] = None) -> writer_tools.ItemStore:
    return writer_tools.ItemStore(world_of(inputs), {fact.id: fact for fact in inputs.intent.facts}, unwitnessed,
                                  items, ruled, rebuttals)


def is_item_check(check: Check) -> bool:
    """A check written as an item (spec/items.py), not a demand of the older grammar."""
    return check.demand.get("kind") in I.ITEM_KINDS


def check_of(item: dict, inputs: WriterInputs, world: Optional[I.World] = None) -> Check:
    """One kept item as a Check: its gate decides the kind and tier, its provenance the because."""
    facts = {fact.id: fact for fact in inputs.intent.facts}
    return Check(id=item["id"], kind="required" if item["gate"] else "allowed", demand=dict(item),
                 because=I.because_of(item, facts, world or world_of(inputs)),
                 tier="critical" if item["gate"] else "important", fact_ids=list(item.get("fact_ids") or []),
                 gate=bool(item["gate"]), weight=float(item["weight"]), policy_line=item.get("policy_line"))


def _spec_of(task_id: str, items: Iterable[dict], counts: dict, inputs: WriterInputs,
             rebuttals: Optional[dict] = None) -> Spec:
    world = world_of(inputs)
    checks = [check_of(item, inputs, world) for item in items]
    spec = with_gaps(Spec(task_id=task_id, intent=inputs.intent, checks=checks, rebuttals=dict(rebuttals or {})),
                     inputs)
    counts["gaps"] = len(spec.gaps)
    if spec.rebuttals:
        counts["rebutted"] = len(spec.rebuttals)
    return spec


def with_gaps(spec: Spec, inputs: WriterInputs) -> Spec:
    """The Spec with its gaps: facts no check covers, then, for older demands, what compile drops."""
    spec = spec.model_copy(update={"gaps": []})
    legacy = spec.model_copy(update={"checks": [check for check in spec.checks if not is_item_check(check)]})
    reasons = compile_spec(legacy, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools).reasons
    return spec.model_copy(update={"gaps": coverage(spec) + list(reasons)})


def _price(model_id: Optional[str]) -> Callable[[dict], float]:
    from kullback.ai.usage import Usage
    from kullback.runner.budget import call_cost

    return lambda usage: call_cost(Usage(**usage), model_id)


def _config(config: Any) -> Any:
    from kullback.ai.provider import ModelConfig

    return config or ModelConfig(thinking={"type": "adaptive", "display": "summarized"}, max_tokens=4000)


# Each refusal counted under the family of code wording it carries (spec/items.py), never its text:
# a cut of the text kept values (a field's value, a quoted sentence) and left some keys empty.
REASON_FAMILIES = (
    (I.UNREACHABLE, "unreachable value"), ("rests only on echo facts", "echo value at a gate"), ("unfounded facts", "unfounded facts at a gate"),
    ("is not in the world", "value not found"),
    ("unwitnessed facts", "unwitnessed facts alone"), ("evidence_why", "narrow evidence without why"),
    ("which is no value of the cited row", "computed anchor"), ("anchor.answer does not state", "anchor misses row"),
    ("anchor", "anchor shape"), ("evidence", "evidence ref"), ("a list value", "list value"),
    ("value set", "value set shape"), ("scalar field", "nested value"), ("find matches", "find not one row"),
    ("find names", "find shape"), ("has no field", "unknown field"), ("no table", "unknown table"),
    ("no tool", "unknown tool"), (" takes ", "unknown argument"), ("free ", "free field"),
    ("expect", "expect shape"), ("said", "said shape"), ("fact_ids", "no provenance"),
    ("policy_line", "policy line"), ("not ruled", "not ruled"), ("question", "question shape"),
    ("count", "count"), ("alt ", "alt"), ("gate", "gate or weight"), ("kind", "kind"), ("id", "id"))
NO_REASON = "no reason"


def reason_family(reason: str) -> str:
    text = (reason or "").strip()
    if not text:
        return NO_REASON
    return next((key for phrase, key in REASON_FAMILIES if phrase in text), "other")


def _store_counts(store: writer_tools.ItemStore) -> dict:
    reasons: dict[str, int] = {}
    for row in store.refused:
        key = reason_family(row["reason"])
        reasons[key] = reasons.get(key, 0) + 1
    return {"offered": store.offered, "kept": len(store.items), "refused": len(store.refused),
            "refused_why": dict(sorted(reasons.items())), "dropped": len(store.dropped),
            "reachability": {"refused": reasons.get("unreachable value", 0), "warned": len(store.warned)}}


# One file per Task, one line per writer session: the first write and every repair (D351).
SESSIONS_DIR = Path("exam") / "writer"


def sessions_path(workdir: Path, task_id: str) -> Path:
    return Path(workdir) / SESSIONS_DIR / f"{task_id}.jsonl"


class _Recorded:
    """The model with the conversation of its last call kept: the messages sent and the reply."""

    def __init__(self, model: Any):
        self.model, self.last = model, None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def query(self, messages: list[dict], **kwargs: Any) -> Any:
        reply = self.model.query(messages, **kwargs)
        self.last = (list(messages), reply)
        return reply

    def transcript(self) -> list[dict]:
        if self.last is None:
            return []
        messages, reply = self.last
        turn = writer_tools._assistant_tool_turn(reply) if reply.tool_calls else \
            {"role": "assistant", "content": reply.content}
        return messages + [dict(turn, thinking=writer_tools._thinking_of(reply))]


def _session(model: Any, messages: list[dict], store: writer_tools.ItemStore, inputs: WriterInputs, config: Any,
             price: Callable[[dict], float], ceiling_usd: Optional[float]) -> dict:
    recorded = _Recorded(model)
    session = writer_tools.run_session(recorded, messages, inputs.state, _config(config),
                                       writer_tools.MAX_ITEM_ROUNDS, price, force_answer=True,
                                       ceiling_usd=ceiling_usd, tools=writer_tools.WORLD_TOOLS,
                                       impl=writer_tools.world_tools(store))
    return dict(session, transcripts=[recorded.transcript()])


def _joined(first: dict, again: dict) -> dict:
    return dict(again, usd=float(first.get("usd") or 0.0) + float(again.get("usd") or 0.0),
                tool_rounds=first.get("tool_rounds", 0) + again.get("tool_rounds", 0),
                tool_uses=first.get("tool_uses", []) + again.get("tool_uses", []),
                transcripts=first.get("transcripts", []) + again.get("transcripts", []))


def save_session(inputs: WriterInputs, task_id: str, kind: str, round_number: int, session: dict,
                 counts: dict, ruling: Optional[Ruling] = None) -> Optional[Path]:
    """Append one writer session to the Task's file: what it was (a write or a repair, the round, the
    ruling answered), each conversation it held (a follow-up turn is a second one), its spend and counts."""
    if inputs.workdir is None:
        return None
    path = sessions_path(inputs.workdir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"task_id": task_id, "kind": kind, "round": int(round_number),
           "ruling": None if ruling is None else {"item": ruling.item, "kind": ruling.kind,
                                                  "check_ids": list(ruling.check_ids)},
           "usd": float(session.get("usd") or 0.0), "unfinished": bool(session.get("unfinished")),
           "text": session.get("text") or "", "counts": counts, "transcripts": session.get("transcripts", [])}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    return path


def _side_counts(intent: SpecIntent, items: list[dict]) -> dict:
    """Side facts, reactions and conditional asks, and the gates weighed for resting on one, and the told
    gates weighed for resting on no told fact and no policy line (D369); nothing for a Task with none."""
    out = {}
    for name, has, weighed in (("side", lambda f: f.side, I.SIDE_WEIGHED),
                               ("reaction", is_reaction, I.AGENT_WEIGHED),
                               ("conditional", lambda f: f.condition == "world_or_policy", I.UNCONFIRMED_WEIGHED)):
        found = sum(1 for fact in intent.facts if has(fact))
        if found:
            out.update({f"{name}_facts": found,
                        f"{name}_weighed": sum(1 for item in items if item.get("weighed") == weighed)})
    untold = sum(1 for item in items if item.get("weighed") == I.TELL_WEIGHED)
    if untold:
        out["tell_weighed"] = untold
    return out


def _clash_rows(store: writer_tools.ItemStore) -> list[dict]:
    """Items no run can pass together, each with why, as one send-back turn lists them."""
    return [{"id": item_id, "reason": why} for item_id, why in I.contradictions(list(store.items.values()))]


def write_spec(task_id: str, inputs: WriterInputs, model: Any, *, model_id: Optional[str] = None,
               config: Any = None, ceiling_usd: Optional[float] = None) -> Written:
    """One writer session for one Task, and the Spec code keeps from it; past `ceiling_usd` it answers at once.

    Items that contradict each other (`items.contradictions`) go back once in a follow-up turn of the same
    conversation; any still there are named in counts["unsatisfiable"]. Volunteered facts no gated item
    covers (`asked_missing`) go back once the same way; the writer gates each or answers why not. Items
    that answer clashing with others go back once more, so an added gate is never left unpassable unseen.
    """
    writer_tools.refuse_run_inputs(inputs.intent, inputs.state)
    messages, cut = writer_messages(inputs)
    store = store_of(inputs)
    price = _price(model_id)
    session = _session(model, messages, store, inputs, config, price, ceiling_usd)

    def again(turn: list, done: dict, prompt: str, rows: list) -> tuple[list, dict]:
        turn = turn + [{"role": "assistant", "content": done["text"] or "done"},
                       {"role": "user", "content": prompt + json.dumps(rows)}]
        left = None if ceiling_usd is None else max(ceiling_usd - float(done.get("usd") or 0.0), 0.0)
        return turn, _joined(done, _session(model, turn, store, inputs, config, price, left))

    turn = list(messages)
    clashes = _clash_rows(store)
    if clashes:
        turn, session = again(turn, session, FOLLOWUP_PROMPT, clashes)
    missing = asked_missing(list(store.items.values()), inputs.intent)
    if missing:
        store.rebuttable = {row["id"] for row in missing}
        turn, session = again(turn, session, ASKED_FOLLOWUP_PROMPT, missing)
    asked_clashes = _clash_rows(store) if missing else []
    if asked_clashes:
        turn, session = again(turn, session, FOLLOWUP_PROMPT, asked_clashes)
    items = list(store.items.values())
    counts = _store_counts(store)
    counts.update(sent_back=len(clashes), unsatisfiable=sorted(
        {item_id for item_id, _ in I.contradictions(items)}),
        asked_sent_back=len(missing), truncated_chars=cut, unfinished=bool(session.get("unfinished")))
    spec = _spec_of(task_id, items, counts, inputs, store.rebuttals)
    left = unchecked_end(spec)
    if left:
        counts[UNCHECKED_END] = left
    counts.update(I.counts_of(items + [I.sanity_item([])]))
    counts["echo_facts"] = sum(1 for fact in inputs.intent.facts if fact.echo)
    counts.update(_side_counts(inputs.intent, items))
    if asked_clashes:
        counts["asked_clash_sent_back"] = len(asked_clashes)
    if inputs.effects_missing:
        counts[EFFECTS_COUNT] = NO_EFFECTS
    save_session(inputs, task_id, "write", 1, session, counts)
    return Written(spec, counts, session)


def item_checks(spec: Spec) -> list[dict]:
    return [check.demand | {"id": check.id} for check in spec.checks if is_item_check(check)]


def verifier_of(spec: Spec, inputs: WriterInputs) -> Compiled:
    """The Spec's Verifier: items to atoms (the verdict decides each), end states with the sanity item.

    Checks of the older demand grammar compile to atoms beside the items, with the must-not atoms when
    the Spec has no item; nothing else gates on them (the Spec stage rewrites such a Spec as items). A
    repeated item atom or end state cell is folded into one, each fold a line of the reasons (D428).
    """
    items = item_checks(spec)
    legacy = spec.model_copy(update={"checks": [c for c in spec.checks if not is_item_check(c)]})
    compiled = compile_spec(legacy, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools)
    atoms = list(compiled.verifier.atoms)
    if not items and legacy.checks:
        must = [atom for atom in atoms if atom.kind in ("required", "allowed")]
        atoms += must_not(must, spec.intent.text, inputs.constraints, sorted(inputs.write_tools))
    expected, merged = [], []
    if items or not legacy.checks:
        expected, merged = merged_cells(I.end_states_of(items, world_of(inputs), inputs.action_tables, inputs.effects))
        own, folded = merged_atoms([I.atom_of(item) for item in items])
        atoms, merged = own + [I.sanity_atom(expected)] + atoms, merged + folded
    verifier = Verifier(task_id=spec.task_id, atoms=atoms,
                        verifier_version=f"spec{spec.version}", seed_run_ids=[], expected=expected)
    return Compiled(verifier, compiled.dropped, compiled.reasons + tuple(merged),
                    compiled.unsatisfiable + tuple(I.contradictions(items)))


def runner_verifier_path(workdir: Path, task_id: str) -> Path:
    """Where the Runner reads a Task's Verifier from (kullback score, BuiltEnvironment.verifier)."""
    return Path(workdir) / "verifiers" / f"{task_id}.json"


def end_state_counts(verifier: Verifier, spec: Optional[Spec] = None) -> dict:
    """What the Spec record keeps of its Verifier: items by kind and gate, end states, cells, gates."""
    cells = verifier.expected[0].cells if verifier.expected else []
    items = item_checks(spec) if spec is not None else []
    out = {"end_states": len(verifier.expected), "cells": len(cells),
           "new_rows": sum(len(state.new_rows) for state in verifier.expected),
           "forbidden": len(verifier.forbidden), "conduct": len(verifier.conduct)}
    if items:
        out = {**I.counts_of(items + [I.sanity_item(verifier.expected)]), **out}
    if spec is not None:
        out["echo_facts"] = sum(1 for fact in spec.intent.facts if fact.echo)
    return out


def reachability(spec: Spec, inputs: WriterInputs) -> dict:
    """The reachability rule's count on the Spec (D387): gated values code refused in the first write, and
    the kept items naming a value no source reaches, weighed (warned) or, on an older Spec, gated."""
    world = world_of(inputs)
    found = [item for item in item_checks(spec) if world.unreachable(item)]
    return {"refused": int(((spec.writer or {}).get("reachability") or {}).get("refused", 0)),
            "warned": sum(1 for item in found if not item.get("gate")),
            "gated": sum(1 for item in found if item.get("gate"))}


def write_verifier(spec: Spec, workdir: Path, inputs: Optional[WriterInputs] = None) -> list[Path]:
    """Saves the Spec and its whole Verifier, written from its items by code; no Reference is read.

    The Spec is the one writer of a Task's Verifier (D320): the Runner's file and the Spec's copy are
    the same bytes. The counts ride on the saved Spec as `end_state`.
    """
    inputs = inputs or load_inputs(workdir, spec.task_id, spec.intent)
    compiled = verifier_of(spec, inputs)
    verifier: Verifier = compiled.verifier
    text = json.dumps(verifier.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    counts = end_state_counts(verifier, spec)
    merged = [line for line in compiled.reasons if ": merged" in line]
    if merged:
        counts["merged"] = merged
    if inputs.reach is not None:
        counts["reachability"] = reachability(spec, inputs)
    spec = spec.model_copy(update={"end_state": counts})
    paths = [save_spec(workdir, spec), spec_verifier_path(workdir, spec.task_id),
             runner_verifier_path(workdir, spec.task_id)]
    for path in paths[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return paths


def _ruled_view(spec: Spec, ruling: Ruling) -> dict:
    ruled = [check.model_dump(mode="json") for check in spec.checks if check.id in set(ruling.check_ids)]
    return {"items": item_checks(spec), "ruled_items": ruled,
            "ruling": {"item": ruling.item, "kind": ruling.kind, "reason": ruling.reason, "fix": ruling.fix,
                       "excerpt": str(ruling.excerpt)[:MAX_EXCERPT]}}


def repair_messages(spec: Spec, ruling: Ruling, inputs: WriterInputs) -> tuple[list[dict], int]:
    return _message(SYSTEM_PROMPT + "\n" + REPAIR_PROMPT, dict(_body(inputs), **_ruled_view(spec, ruling)))


def repair(spec: Spec, ruling: Ruling, inputs: WriterInputs, model: Any, *,
           model_id: Optional[str] = None, config: Any = None,
           ceiling_usd: Optional[float] = None) -> Written:
    """One repair session answering a ruling: apply the fix or rebut it; no Run is shown, only the excerpt.

    The session starts from the Spec's items; it may drop or replace only ruled items and add new ones,
    under the same find-do-not-invent rule, and code answers each add with its refusal reason inside
    the session. When code refused everything and the writer neither changed anything nor rebutted,
    it gets one retry in the same conversation, shown each refusal and its reason. A Spec of the older
    grammar is rewritten as items. The counts say what changed, dropped and was added, what code
    refused, whether it retried, and the writer's rebuttal when it changed nothing.
    """
    ruling = Ruling.model_validate(ruling.model_dump() if isinstance(ruling, Record) else ruling)
    writer_tools.refuse_run_inputs(inputs.intent, inputs.state, ruling.excerpt)
    messages, cut = repair_messages(spec, ruling, inputs)
    store = _repair_store(spec, ruling, inputs)
    before = {item["id"]: item for item in item_checks(spec)}
    price = _price(model_id)
    session = _session(model, messages, store, inputs, config, price, ceiling_usd)
    retried = 0
    if store.refused and store.items == before and not rebuttal_of(session.get("text") or ""):
        refusals = "\n".join(f"- {row['id']}: {row['reason']}" for row in store.refused)
        turn = messages + [{"role": "assistant", "content": session["text"] or "done"},
                           {"role": "user", "content": RETRY_PROMPT.format(refusals=refusals)}]
        left = None if ceiling_usd is None else max(ceiling_usd - float(session.get("usd") or 0.0), 0.0)
        session, retried = _joined(session, _session(model, turn, store, inputs, config, price, left)), 1
    items = list(store.items.values())
    counts = _store_counts(store)
    counts.update(changed=sum(1 for item in items if item["id"] in before and before[item["id"]] != item),
                  dropped=sum(1 for check in spec.checks if check.id not in store.items),
                  added=sum(1 for item in items if item["id"] not in before),
                  not_ruled=store.not_ruled, truncated_chars=cut, retried=retried,
                  refusals=[{"id": row["id"], "kind": row["kind"]} for row in store.refused],
                  rebut=rebuttal_of(session.get("text") or ""))
    repaired = _spec_of(spec.task_id, items, counts, inputs, store.rebuttals).model_copy(update={
        "version": spec.version + 1, "round": spec.round, "rulings_open": spec.rulings_open,
        "writer": spec.writer})
    save_session(inputs, spec.task_id, "repair", spec.round + 1, session, counts, ruling)
    return Written(repaired, counts, session)


def _repair_store(spec: Spec, ruling: Ruling, inputs: WriterInputs) -> writer_tools.ItemStore:
    """The repair's store: the Spec's items, drops limited to the ruled ones (any, on the older grammar), and
    the rebuttals but those of the facts an unchecked_ask ruling names, which the writer may rebut anew."""
    legacy = not all(is_item_check(check) for check in spec.checks)
    asked = asks_ruled(spec, ruling)
    store = store_of(inputs, item_checks(spec), None if legacy else set(ruling.check_ids),
                     {fact_id: why for fact_id, why in spec.rebuttals.items() if fact_id not in asked})
    store.rebuttable = asked
    return store


def asks_ruled(spec: Spec, ruling: Ruling) -> set[str]:
    """The facts an unchecked_ask ruling names (D376): its fact rows when code filed it, its item when the
    Examiner did. A rebuttal of such a fact was found wrong, so the repair drops it; the writer may rebut anew."""
    if not ruling.kind.endswith(UNCHECKED_ASK):
        return set()
    named = set(ruling.facts) | {ruling.item}
    return named & {fact.id for fact in spec.intent.facts}


def rebuttal_of(text: str) -> str:
    """The writer's why when it rebuts: the text after "rebut:" in its last reply, else empty."""
    found = re.search(r"rebut:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return " ".join(found.group(1).split()) if found else ""


def ruling_of(body: dict) -> Ruling:
    """A ruling as the router hands it over: its checks, its reason, and its quoted evidence.

    The evidence is the ruling's excerpt, else its quotes joined; a quote that is not text is left out.
    """
    quotes = body.get("quotes") or body.get("quoted") or []
    excerpt = body.get("excerpt") or "\n".join(str(q) for q in quotes if isinstance(q, str))
    check_ids = body.get("check_ids") or ([body["item"]] if body.get("item") else [])
    return Ruling(check_ids=[str(c) for c in check_ids], reason=str(body.get("reason") or ""),
                  excerpt=str(excerpt), item=str(body.get("item") or ""), fix=str(body.get("fix") or ""),
                  kind="/".join(str(body[key]) for key in ("kind", "code") if body.get(key)),
                  facts=_fact_rows(body.get("verified")))


def _fact_rows(verified: Any) -> list[str]:
    """The fact ids a ruling's verified rows name (an unchecked_ask ruling filed by code, D376)."""
    return [str(row["fact_id"]) for row in verified or () if isinstance(row, dict) and row.get("fact_id")]


def repair_for_ruling(workdir: Path, task_id: str, ruling: dict, model: Any,
                      ceiling_usd: Optional[float] = None, *, model_id: Optional[str] = None) -> Spec:
    """The router's entry point: repair the Task's Spec from one ruling file and tell the bus.

    Loads the Spec and the writer's inputs, repairs the ruled checks, saves the Spec (version + 1),
    rewrites both Verifier files and publishes spec.repaired when any check changed,
    else spec.defended.
    """
    from kullback.agent.bus import Bus

    spec = load_spec(workdir, task_id)
    if spec is None:
        raise FileNotFoundError(f"no Spec for Task {task_id}: nothing to repair")
    inputs = load_inputs(workdir, task_id, spec.intent)
    ruled = ruling_of(ruling)
    written = repair(spec, ruled, inputs, model, model_id=model_id or getattr(model, "name", None),
                     ceiling_usd=ceiling_usd)
    write_verifier(written.spec, workdir, inputs)
    bus = Bus(Path(workdir) / "bus.jsonl", agent="spec")
    before = {check.id: check for check in spec.checks}
    after = {check.id for check in written.spec.checks}
    changed = [check.id for check in written.spec.checks if before.get(check.id) != check]
    changed += [check_id for check_id in before if check_id not in after]
    if changed:
        events.spec_repaired(bus, task_id, written.spec.version, changed)
    else:
        events.spec_defended(bus, task_id, ruled.check_ids, str(ruling.get("code") or "other"))
    return load_spec(workdir, task_id)


# The one fixed question code asks when a Spec writes and no item waits for the user's yes (item 22).
CONFIRM_QUESTION = ("This Spec has a write item and no before item with first confirm_turn. Does the policy "
                    "require the user's explicit yes before this write?")
CONFIRM_FIX = ("If it does, add a before item {first: \"confirm_turn\", then: <the write tool>}, gated, with the "
               "policy line that requires the yes as its policy_line. If it does not, reply with one line "
               "starting \"rebut:\" saying none.")


def lacks_confirm(spec: Spec, write_tools: Iterable[str]) -> bool:
    """A Spec with a write item (a state change, or a call of a write tool) and no confirm-turn before item."""
    items, writes = item_checks(spec), set(write_tools)
    wrote = any(item["kind"] in ("row_is", "row_new") or (item["kind"] == "called" and item.get("tool") in writes)
                for item in items)
    return wrote and not any(item["kind"] == "before" and item.get("first") == I.CONFIRM_TURN for item in items)


def ask_confirm(workdir: Path, task_id: str, model: Any, ceiling_usd: Optional[float] = None) -> dict:
    """Ask the writer the fixed confirm question when the Spec needs it, before the Examiner's round.

    The question goes as a ruling on the Task (no item ruled), so the writer may only add items or rebut.
    No model, no Spec, or a Spec that does not need the question: nothing is asked.
    """
    spec = load_spec(workdir, task_id) if model is not None else None
    if spec is None or not lacks_confirm(spec, load_inputs(workdir, task_id, spec.intent).write_tools):
        return {"asked": False}
    held = {check.id for check in spec.checks}
    after = repair_for_ruling(workdir, task_id, {"item": "", "reason": CONFIRM_QUESTION, "fix": CONFIRM_FIX,
                                                 "kind": "confirm_coverage"}, model, ceiling_usd)
    added = [check for check in after.checks if check.id not in held and check.demand.get("kind") == "before"
             and check.demand.get("first") == I.CONFIRM_TURN]
    return {"asked": True, "added": len(added)}


def spoken_intent(task_id: str, text: str) -> SpecIntent:
    """Spoken user text as one volunteered fact per sentence: the stand-in until a mined Intent exists."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    facts = [IntentFact(id=f"f{n}", text=s, stance="volunteered", source=FactSource(recording="spoken", turn=0))
             for n, s in enumerate(sentences)]
    return SpecIntent(task_id=task_id, facts=facts, text=intent_text(facts))

