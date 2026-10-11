"""Synthetic Tasks: a walk over the mined graph, run in the rebuilt world, derived and reported apart (D224).

`graph.py` proposes a walk. This module is what makes one a Task. It binds each step's arguments,
taking a value from the earlier step's own answer where the edge says the value came from there and
from a row of the rebuilt world otherwise; it runs the walk through the Builder's own tool bodies,
so a walk the world refuses is thrown away rather than repaired; it replays the bound calls once
more to leave a Run on disk the way every other Run of the harness is on disk; and it hands that Run
to the same derive path a Reference goes through, so a synthetic Task's Verifier is derived by the
code that derives every Verifier and by nothing written for this stage.

Three rules keep a synthetic Task honest.

  Nothing is invented. An argument no earlier step feeds is bound to a value the recordings showed
  for that argument, or to the id of a row the rebuilt world holds, and never to a value made up
  here. Real rows are drawn before synthetic ones, and synthetic rows are grown (`synth.py`) only
  when a request asks for more Tasks than the real rows can carry.

  Nothing is repaired. A walk whose write a body refuses is discarded and counted, because a refusal
  is the world saying this path does not exist, and a generator that argues with the bodies would
  manufacture Tasks the Environment does not hold. The refusals are the measurement: a corpus whose
  walks are mostly refused has a graph missing a precondition edge, and the count is what says so.

  Nothing synthetic is trusted. The Tasks live in their own store, under `synthetic/`, and never in
  tasks.json; the counts are reported under their own heading. Trusted still means a recorded trace
  replayed at fidelity with a Reference confirmed, and no walk can earn that.

The simulated user's facts follow D210's two classes over the corpus rather than over one Run: a
field some recorded user of this Environment stated is askable, and a field no recorded user ever
stated is a record fact this user will not say. The Intent is written in symptom form (D196) from
the walk's askable values alone, and a draft that carries a record value is refused here rather than
edited, so the model cannot leak what the world alone knows.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback import derive as derive_mod
from kullback import difficulty
from kullback import graph as graph_mod
from kullback import variants as variants_mod
from kullback.builder import compile_env, mine, synth
from kullback.examiner import reference as reference_mod
from kullback.gates import tool_runs, verifier_suite
from kullback.gates.tool_runs import CRASH_ERRORS
from kullback.runner import canon, route
from kullback.runner import replay as replay_mod
from kullback.runner.records import (
    EntitySchema,
    Task,
    ToolSig,
    UserFact,
    UserRules,
    Verifier,
    as_dict,
    read_json,
    write_json,
)
from kullback.sampling import build_salt, sample_key

# Where a synthesised Task and everything derived from it lives, and the version of that shape. It
# is a directory of its own and not a row in tasks.json on purpose: a count that must never enter
# trusted is kept where the trusted count cannot read it.
DIR = "synthetic"
INDEX = "index.json"
FORMAT = 1

# How many walks are proposed per Task asked for before the request gives up on a bucket. A refused
# walk costs one run of the bodies and no model call, so the ceiling is about wall clock.
ATTEMPTS_PER_TASK = 6
# How many rewrites of a kept walk are run to look for a second path to the same End state (D199).
SECOND_PATH_LIMIT = 4

REFUSED = "refused"
CRASHED = "crashed"
UNBOUND = "unbound"


# --- the rebuilt world, read off a workdir ------------------------------------------

class World:
    """Everything one walk needs to run, read off a finished workdir and never from a model."""

    def __init__(self, workdir: Any) -> None:
        self.workdir = Path(workdir)
        self.schema = EntitySchema.model_validate(read_json(self.workdir / "schema.json", {}) or {})
        self.sigs = [ToolSig.model_validate(row) for row in read_json(self.workdir / "tool_sigs.json", []) or []]
        self.bodies = read_json(self.workdir / "bodies.json", {}) or {}
        db = read_json(self.workdir / "db.json", None)
        self.db = db if db is not None else (read_json(self.workdir / "env" / "db.json", {}) or {})
        self.environment = read_json(self.workdir / "environment.json", {}) or {}
        self.readers = read_json(self.workdir / "readers.json", []) or []
        self.canon_rules = canon.load_rules(self.workdir / "canon-rules.json")
        self.write_tools = {sig.name for sig in self.sigs if sig.kind == "write"}
        self.source = compile_env.module_source(self.schema, self.sigs, self.bodies)
        self.comparer = tool_runs.ReplayComparer(self.schema, tool_runs.load_readers(self.readers),
                                                 self.canon_rules)

    def fresh_db(self) -> dict:
        """A world of this Environment nothing has written to yet, one per walk."""
        return json.loads(json.dumps(self.db))

    def toolkit(self, db: dict):
        return compile_env.load_toolkit(self.source, db)

    def rows_of(self, table: str) -> list[str]:
        """The ids of one table, the rows the recordings showed before the rows nobody recorded."""
        rows = synth.table_rows(self.db, self.schema, table)
        synthetic = set(self.schema.synthetic_rows or ())
        real = [key for key in sorted(rows) if key not in synthetic]
        return real + [key for key in sorted(rows) if key in synthetic]


def askable_fields(workdir: Any) -> set[str]:
    """The argument names some recorded user of this Environment stated (D210's askable class).

    Over the corpus and not over one Run: a synthetic Task has no recorded user of its own, and the
    class a field belongs to is a property of the world's users, not of the Run in hand. A field no
    user ever said is a record fact, and this user will not say it however the Intent is written.
    """
    body = read_json(Path(workdir) / "user_facts.json", {}) or {}
    return {str(row.get("field")) for row in body.get("facts") or [] if row.get("field")}


# --- binding one walk's arguments ---------------------------------------------------

def _set_path(args: dict, path: str, value: Any) -> None:
    """Put a value at an argument path, a list collapsed to its first element (`items[].id`).

    Two edges can name one argument at two depths, one as a scalar and one as the field of an
    object, because two recorded calls filled it two ways. The second one gives way rather than
    writing into the first: an argument is bound once, and a path that has to walk through a value
    already bound is a path this walk cannot take.
    """
    parts = [part for part in str(path).split(".") if part]
    node: Any = args
    for index, part in enumerate(parts):
        if not isinstance(node, dict):
            return
        listed = part.endswith("[]")
        name = part[:-2] if listed else part
        if index == len(parts) - 1:
            node[name] = [value] if listed else value
            return
        if listed:
            if not isinstance(node.get(name), list) or not node[name]:
                node[name] = [{}]
            node = node[name][0]
            continue
        if not isinstance(node.get(name), dict):
            node[name] = {}
        node = node[name]


def declared(sig: Any) -> tuple[dict, list[str]]:
    """One tool's arguments as its mined signature declares them: the properties and the required.

    A walk fills what a tool takes and nothing else. A recording that once passed an argument the
    signature does not declare is a recording of a call the world refused, and passing it again
    would fail the body on the harness's mistake rather than on the world's rule.
    """
    schema = getattr(sig, "args_schema", None) or {}
    properties = dict(schema.get("properties") or {}) if isinstance(schema, dict) else {}
    required = [str(name) for name in (schema.get("required") or properties)] if properties else []
    return properties, required


def _typed(spec: Any, value: Any) -> Any:
    """One value in the type the signature declares for its argument: a list wrapped, a number cast."""
    kinds = [str(kind) for kind in ((spec or {}).get("type") or [])]
    if "list" in kinds:
        return value if isinstance(value, list) else [value]
    for kind, cast in (("int", int), ("float", float)):
        if kind in kinds and not isinstance(value, bool):
            try:
                return cast(value)
            except (TypeError, ValueError):
                return value
    return value


def _aliases(path: str) -> list[str]:
    """The names one leaf answers under: its own, its singular and plural, and its parent's id.

    An argument called `item_ids` and a column called `item_id` are the same thing said twice, and a
    leaf called `id` under a collection called `payment_methods` is a payment method id: both are the
    constructions `mine.py` already reads an id column by, so nothing new about naming is invented
    here. A name is the only vocabulary this module has, and these are the forms of one name.
    """
    parts = [part for part in str(path).replace("[]", "").replace("{}", "").split(".") if part]
    if not parts:
        return []
    name = parts[-1]
    out = [name, mine._singular(name), mine._plural(name)]
    if name in ("id", "ids") and len(parts) > 1:
        entity = mine._singular(parts[-2])
        out += [f"{entity}_id", f"{entity}_ids"]
    seen: list[str] = []
    for alias in out:
        if alias and alias not in seen:
            seen.append(alias)
    return seen


def named_leaves(value: Any) -> dict[str, list[str]]:
    """Every scalar inside a value, by every name it answers under (`address.zip` under `zip`).

    An argument and a column of the row it came from are named the same thing far more often than
    they are nested the same way, and the name is the only vocabulary this module has.
    """
    out: dict[str, list[str]] = {}
    for path, text in graph_mod._leaves(value):  # noqa: SLF001 - one flattening, shared with the graph
        for name in _aliases(path):
            if text not in out.setdefault(name, []):
                out[name].append(text)
    return out


def _row_source(world: World, owed: Iterable[str], ident: str) -> dict[str, list[str]]:
    """One row of the rebuilt world, chosen because it carries the names this call still owes.

    One row and not one value per argument: a first name from one row and a postcode from another
    name nobody, and the body refuses the call for a reason the walk invented. Real rows are drawn
    before synthetic ones, so a walk uses what the recordings paid for until there is not enough of
    it (`rows_of`).
    """
    owed = [str(name) for name in owed]
    if not owed:
        return {}
    best: tuple[int, str, list[str]] = (0, "", [])
    for table in sorted(world.schema.tables or ()):
        keys = world.rows_of(table)
        if not keys:
            continue
        rows = synth.table_rows(world.db, world.schema, table)
        sample = named_leaves(rows.get(keys[0]))
        covered = sum(1 for name in owed if name in sample)
        if covered > best[0]:
            best = (covered, table, keys)
    if not best[0]:
        return {}
    _covered, table, keys = best
    rows = synth.table_rows(world.db, world.schema, table)
    key = keys[sample_key("synth-row", f"{ident}:{table}", "") % len(keys)]
    leaves = named_leaves(rows.get(key))
    leaves.setdefault(_id_name(world, table), [key])
    return leaves


def _id_name(world: World, table: str) -> str:
    """What a row of this table is named by, so the row drawn can be passed as its own id."""
    for column in world.schema.columns or ():
        if getattr(column, "table", None) == table and str(getattr(column, "name", "")).endswith("id"):
            return str(column.name)
    return f"{table.rstrip('s')}_id"


# How many reads one write may pull in when its row-naming arguments have nothing to bind from,
# and how many fresh draws a refused write is retried with before the walk is discarded.
MAX_INSERTED_READS = 3
WRITE_RETRIES = 8


def _observed_pool(observed: dict, tool: str, name: str) -> list:
    """The values the recordings showed for one argument, under either spelling of a list."""
    rec = observed.get(tool) or {}
    for key in (name, name + "[]"):
        values = (rec.get(key) or {}).get("values") or []
        if values:
            return list(values)
    return []


def _read_leaves(answers: list[Any]) -> dict[str, list[str]]:
    """Every name the walk's earlier answers carry, the latest answer's values first."""
    seen: dict[str, list[str]] = {}
    for answer in reversed(answers):
        for name, values in named_leaves(answer).items():
            seen.setdefault(name, values)
    return seen


def _missing_row_args(step: dict, answers: list[Any], sigs: dict) -> list[str]:
    """Required row-naming arguments of one write no edge binding and no earlier answer can fill."""
    _properties, required = declared(sigs.get(str(step.get("tool"))))
    bound = set()
    for binding in step.get("bindings") or ():
        if graph_mod.values_at(answers[int(binding["from"])], str(binding["column"])):
            bound.add(graph_mod.head(binding["arg"]))
    read = _read_leaves(answers)
    return [name for name in required
            if name not in bound and name not in read
            and (name.endswith("_id") or name.endswith("_ids"))]


def _supplier(graph: dict, write: str, arg: str, writes: set[str], used: set[str]) -> Optional[dict]:
    """The read the recordings most often fed this argument from: the heaviest edge into the write."""
    best = None
    for edge in graph.get("edges") or ():
        if (str(edge.get("to")) == write and graph_mod.head(edge.get("arg") or "") == arg
                and edge.get("kind") == graph_mod.VALUE_EDGE
                and str(edge.get("from")) not in writes and str(edge.get("from")) != write
                and str(edge.get("from")) not in used):
            if best is None or int(edge.get("weight") or 0) > int(best.get("weight") or 0):
                best = edge
    return best


def bind(world: World, step: dict, answers: list[Any], observed: dict, sigs: dict,
         ident: str, *, is_write: bool = False) -> Optional[dict]:
    """One step's arguments, in the order the evidence for them gets weaker.

    An edge first, because the recordings showed the value travelling that way; then the nearest
    earlier answer, because a walk that names what it has just been told is a walk that hangs
    together; then one row of the rebuilt world, all of whose values belong to each other; then the
    values the recordings showed for this argument. An argument the signature requires and none of
    the four can fill leaves the walk unbound, which is counted and never guessed at.

    A list argument is looked up under both spellings: the graph stores it as `name[]` and the
    signature names it `name`, and the two are one argument's evidence. A row-naming argument of
    a write (a name ending in `_id` or `_ids`) binds only from what the walk has read, an edge or
    an earlier answer, never from a row drawn for the occasion or the recorded pool: a write that
    names a row the walk never touched is a walk that does not hang together.
    """
    tool = str(step.get("tool"))
    properties, required = declared(sigs.get(tool))
    if not properties:
        return None
    args = _edge_args(step, answers, properties, ident)
    owed = [name for name in required if name not in args]
    seen = _read_leaves(answers) if is_write else (named_leaves(answers[-1]) if answers else {})
    row = _row_source(world, [name for name in owed if name not in seen], ident)
    for name in owed:
        pool = _arg_pool(is_write, name, seen, row, observed, tool)
        if not pool:
            return None
        spec = properties.get(name) or {}
        args[name] = _typed(spec, pool[sample_key("synth-arg", f"{ident}:{tool}:{name}", "") % len(pool)])
    return {name: value for name, value in args.items() if name in properties}


def _edge_args(step: dict, answers: list[Any], properties: dict, ident: str) -> dict:
    """What the step's edge bindings fill: the values the recordings showed travelling that way.

    An edge whose answer does not carry its column falls back to the sources below rather than
    failing the walk on one binding.
    """
    args: dict = {}
    for binding in step.get("bindings") or ():
        if graph_mod.head(binding["arg"]) not in properties:
            continue
        found = graph_mod.values_at(answers[int(binding["from"])], str(binding["column"]))
        if not found:
            continue
        _set_path(args, str(binding["arg"]),
                  found[sample_key("synth-edge-value", f"{ident}:{binding['arg']}", "") % len(found)])
    return args


def _arg_pool(is_write: bool, name: str, seen: dict, row: dict, observed: dict,
              tool: str) -> list:
    """The values one owed argument binds from, weakest evidence last.

    A row-naming argument of a write reads only what the walk has read; every other argument then
    reads one row of the rebuilt world, then the values the recordings showed for it.
    """
    if is_write and (name.endswith("_id") or name.endswith("_ids")):
        return seen.get(name) or []
    return seen.get(name) or row.get(name) or _observed_pool(observed, tool, name)


# --- running one walk ---------------------------------------------------------------

def run_walk(world: World, steps: list[dict], observed: dict, ident: str, *,
             graph: Optional[dict] = None, write_tools: Iterable[str] = (),
             note: Optional[dict] = None) -> tuple[list[dict], str]:
    """Bind and run the walk through the Builder's own tool bodies; the calls it made, and why not.

    One toolkit per walk, so what a step wrote the next step sees, which is the whole point of a
    walk. An exception whose class the harness already reads as a body fault (`CRASH_ERRORS`) is a
    crash; every other exception is the body refusing the call, which is the world saying this path
    does not exist.

    A write whose row-naming arguments nothing the walk read can fill pulls in the read the graph
    says feeds each one, heaviest edge first, up to MAX_INSERTED_READS reads: a write that still
    names nothing is unbound and the walk is discarded. A refused write is retried with a new draw,
    up to WRITE_RETRIES tries, each on a toolkit holding the accepted prefix and nothing refused.
    `note`, when given, is filled with the refused argument values, so a description can carry them.
    """
    body = dict(graph) if graph is not None else load_graph(world.workdir)
    writes = {str(tool) for tool in write_tools} or set(world.write_tools)
    toolkit = world.toolkit(world.fresh_db())
    sigs = {sig.name: sig for sig in world.sigs}
    answers: list[Any] = []
    calls: list[dict] = []
    refused: list[str] = []
    for index, step in enumerate(steps):
        name = str(step.get("tool"))
        if name in writes:
            why, toolkit = _run_write(world, body, writes, observed, sigs, steps, index,
                                      answers, calls, refused, ident)
            if why:
                _take_note(note, refused)
                return calls, why
            continue
        args = bind(world, step, answers, observed, sigs, f"{ident}:{index}")
        if args is None:
            _take_note(note, refused)
            return calls, UNBOUND
        runner = getattr(toolkit, name, None)
        if runner is None:
            _take_note(note, refused)
            return calls, CRASHED
        try:
            answer = runner(**args)
        except Exception as error:  # noqa: BLE001 - the class is the ruling, see CRASH_ERRORS
            _take_note(note, refused)
            return calls, CRASHED if type(error).__name__ in CRASH_ERRORS else REFUSED
        answers.append(answer)
        calls.append({"id": f"{ident}-{index}", "name": name, "args": args,
                      "requestor": "assistant", "turn": index, "result": answer, "error": None})
    _take_note(note, refused)
    return calls, ""


def _take_note(note: Optional[dict], refused: list[str]) -> None:
    """The refused argument values into the caller's note, where one was given."""
    if note is not None:
        note["refused"] = list(refused)


def _run_write(world: World, graph: dict, writes: set[str], observed: dict, sigs: dict,
               steps: list[dict], index: int, answers: list[Any], calls: list[dict],
               refused: list[str], ident: str) -> tuple[str, Any]:
    """One write step with its missing reads inserted; "" and the toolkit holding it on success."""
    name = str(steps[index].get("tool"))
    for attempt in range(WRITE_RETRIES):
        sub = f"{ident}:{index}" if attempt == 0 else f"{ident}:{index}:r{attempt}"
        toolkit = world.toolkit(world.fresh_db())
        for call in calls:
            getattr(toolkit, call["name"])(**call["args"])
        local_answers = list(answers)
        local_calls: list[dict] = []
        step = dict(steps[index])
        step["bindings"] = list(step.get("bindings") or ())
        if not _insert_reads(world, graph, writes, observed, sigs, toolkit, name, step,
                             local_answers, local_calls, calls, sub):
            return UNBOUND, toolkit
        args = bind(world, step, local_answers, observed, sigs, sub, is_write=True)
        if args is None:
            return UNBOUND, toolkit
        runner = getattr(toolkit, name, None)
        if runner is None:
            return CRASHED, toolkit
        try:
            answer = runner(**args)
        except Exception as error:  # noqa: BLE001 - the class is the ruling, see CRASH_ERRORS
            if type(error).__name__ in CRASH_ERRORS:
                return CRASHED, toolkit
            refused.extend(text for _path, text in graph_mod._leaves(args))  # noqa: SLF001
            continue
        for call in local_calls:
            calls.append({"id": f"{ident}-{len(calls)}", "name": call["name"], "args": call["args"],
                          "requestor": "assistant", "turn": len(calls),
                          "result": call["result"], "error": None})
        answers.extend(local_answers[len(answers):])
        answers.append(answer)
        calls.append({"id": f"{ident}-{len(calls)}", "name": name, "args": args,
                      "requestor": "assistant", "turn": len(calls), "result": answer,
                      "error": None})
        return "", toolkit
    return REFUSED, world.toolkit(world.fresh_db())


def _insert_reads(world: World, graph: dict, writes: set[str], observed: dict, sigs: dict,
                  toolkit: Any, name: str, step: dict, answers: list[Any], local_calls: list[dict],
                  calls: list[dict], sub: str) -> bool:
    """Pull in the reads feeding one write's row-naming arguments; False when one stays unfed."""
    used: set[str] = set()
    for _ in range(MAX_INSERTED_READS):
        missing = _missing_row_args(step, answers, sigs)
        if not missing:
            return True
        edge = _supplier(graph, name, missing[0], writes, used)
        if edge is None:
            return False
        used.add(str(edge["from"]))
        reader = _reader_step(graph, edge, calls, local_calls)
        rargs = bind(world, reader, answers, observed, sigs, f"{sub}:ins{len(used)}")
        if rargs is None:
            return False
        try:
            found = getattr(toolkit, str(edge["from"]))(**rargs)
        except Exception:  # noqa: BLE001 - a read that will not run feeds nothing
            return False
        answers.append(found)
        step["bindings"] = list(step.get("bindings") or ()) + [
            {"arg": str(edge.get("arg")), "column": str(edge.get("column")),
             "from": len(answers) - 1}]
        local_calls.append({"name": str(edge["from"]), "args": rargs, "result": found})
    return not _missing_row_args(step, answers, sigs)


def _reader_step(graph: dict, edge: dict, calls: list[dict], local_calls: list[dict]) -> dict:
    """One inserted read as a step: the value edges into it from the tools the walk holds."""
    names = [call["name"] for call in calls] + [call["name"] for call in local_calls]
    source = str(edge["from"])
    return {"tool": source, "bindings": [
        {"arg": str(item.get("arg")), "column": str(item.get("column")), "from": pos}
        for item in graph.get("edges") or ()
        if str(item.get("to")) == source and item.get("kind") == graph_mod.VALUE_EDGE
        and item.get("arg")
        for pos, seen in enumerate(names) if seen == str(item.get("from"))]}


DONE = "Done."
CLOSED = "Thank you."


def spoken_for(intent: str, calls: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """The conversation the walk is spoken in, and the calls placed in the turn that makes them.

    A Run with no transcript fails every atom over questions and stated facts however well its
    writes landed, so the walk is spoken even though nothing said it: the user asks, the agent makes
    every call in one turn and says it is done, and the user closes. The closing turn is not
    decoration: the replay's user is finished when it has no turn left, and a conversation whose
    user is finished before the agent has acted stops before the walk runs at all. The calls are
    answered with a copy that names the turn they belong to, because the replay places a call by the
    turn it carries.
    """
    turns = [{"role": "user", "content": intent}, {"role": "assistant", "content": DONE},
             {"role": "user", "content": CLOSED}]
    return turns, [dict(call, turn=1) for call in calls or ()]


def record_run(world: World, task_id: str, calls: list[dict], run_id: str,
               transcript: list[dict], *, runs_at: Optional[Path] = None) -> Optional[dict]:
    """The walk replayed once more from a fresh world, so it lands on disk as a Run like any other.

    The same replay every recorded Trace goes through, over the same Router and the same scoring;
    what is different is only where the Trace came from, which is code and not a recording.
    `runs_at` is where the Run lands; by default under the workdir's own synthetic runs.
    """
    from kullback.runner.tool import call_trace  # imported late: this module's peers import the runner

    db = world.fresh_db()
    toolkit = world.toolkit(db)
    router = route.Router(env_tools_module=toolkit, starting_state=world.fresh_db(), tool_sigs=world.sigs,
                          canon_rules=world.canon_rules, synthetic_rows=world.schema.synthetic_rows)
    route.refuse_stand_in(router)  # G28: a scored synthetic walk is never answered by a model
    trace = call_trace(task_id, calls, transcript, run_id)
    target = Path(runs_at) if runs_at is not None else world.workdir / DIR / "runs" / task_id
    result = replay_mod.replay_trace(trace, router, workdir=target,
                                     task_id=task_id, env_id=world.environment.get("env_id"),
                                     write_tools=world.write_tools, canon_rules=world.canon_rules,
                                     comparer=world.comparer, run_id=run_id)
    if not result.path:
        return None
    return {"run_id": result.run_id, "path": result.path,
            "termination_reason": result.termination_reason, "crashed": result.crashed}


def second_paths(world: World, task_id: str, calls: list[dict], transcript: list[dict],
                 reference: Any, ident: str, limit: int = SECOND_PATH_LIMIT) -> list[dict]:
    """Rewrites of the walk's own call path that land in the same End state (D199), as Runs on disk.

    A synthetic Task's held-out pool is its own re-rolls, and this is the first of them: it costs no
    model call and it is the same rule a recorded Task's second path is found by.
    """
    fn = verifier_suite.canon_fn(world.canon_rules)
    kept: list[dict] = []
    for index, plan in enumerate(variants_mod.variants(calls, world.write_tools, limit=limit), 1):
        row = record_run(world, task_id, plan["calls"], f"{ident}-alt{index}", transcript)
        if not row or not row.get("path"):
            continue
        try:
            record = reference_mod.load(row["path"], reference_mod.REROLL, run_id=row["run_id"],
                                        write_tools=world.write_tools, fn=fn, atoms=())
        except (OSError, ValueError, TypeError):
            continue
        if record.violated or record.end_state != reference.end_state:
            continue
        kept.append(dict(row, kind=plan["kind"]))
    return kept


# --- the Intent and the user's facts ------------------------------------------------

def facts_of(calls: Iterable[dict], askable: set[str]) -> tuple[list[dict], list[dict]]:
    """The walk's argument values split by D210's classes: what a user could say, and what it could not.

    An argument path's last name is the field: `payment.method_id` is asked about as a method id
    wherever it sits, which is the same reading `user_sim` gives a mined fact.
    """
    said: list[dict] = []
    held: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for call in calls or ():
        for path, value in graph_mod.argument_leaves(call):
            field = str(path).split(".")[-1].replace("[]", "")
            if (field, value) in seen:
                continue
            seen.add((field, value))
            (said if field in askable else held).append({"field": field, "value": value})
    return said, held


def user_rules_for(said: Iterable[dict]) -> UserRules:
    """The simulated user of a synthetic Task: exactly the askable values, and nothing else."""
    return UserRules(facts=[UserFact(field=str(row["field"]), value=row["value"]) for row in said])


INTENT_PROMPT = (
    "A person is writing to a support agent about a problem they want solved. Write their first "
    "message.\n\n"
    "Write the symptom and the outcome they want, in their own words, in one or two sentences. Do "
    "not name any tool, any system, any table or any step, and do not say how the agent should do "
    "it. Do not include any value that is not in the list below.\n\n"
    "The only values this person knows and may state:\n{facts}\n\n"
    "Write the message and nothing else.")


def intent_text(model: Any, said: list[dict], held: list[dict], fallback: str) -> tuple[str, str]:
    """The Intent in symptom form, and why the model's draft was not used where it was not.

    A draft that carries a value only the world knows is refused and not edited: a generator that
    trims a leak out of a sentence keeps the sentence that was written around it. Without a model,
    and after a refusal, the Intent is the code-written one, which names no value at all.
    """
    if model is None:
        return fallback, "no Intent writer was given"
    lines = "\n".join(f"- {row['field']}: {row['value']}" for row in said) or "- nothing"
    try:
        reply = model.query([{"role": "user", "content": INTENT_PROMPT.format(facts=lines)}])
    except Exception as error:  # noqa: BLE001 - a provider failure leaves the code-written Intent
        return fallback, f"the Intent writer failed: {type(error).__name__}"
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return fallback, "the Intent writer answered nothing"
    leaked = [row["field"] for row in held if str(row["value"]) and str(row["value"]) in text]
    if leaked:
        return fallback, "the draft stated a value only the world knows"
    return text, ""


def written_intent(calls: Iterable[dict], write_tools: Iterable[str]) -> str:
    """The Intent when no model wrote one: the outcome in the harness's own words, no value in it.

    It names the writes the walk made and nothing else, so it can be handed to a Candidate and can
    never leak a row: the symptom is thin, and a thin Intent is honest about being generated.
    """
    writes = [str(call.get("name")) for call in calls or () if str(call.get("name")) in set(write_tools)]
    if not writes:
        return "I need to check on something in my account."
    return "I need this sorted out: " + ", then ".join(writes).replace("_", " ") + "."


# --- one request --------------------------------------------------------------------

def _grow_for(world: World, wanted: int) -> None:
    """Grow the tables a walk binds from when the request asks for more Tasks than rows (D107).

    The rows a request needs are the rows its start bindings draw from. A table with fewer rows than
    Tasks asked for would hand two Tasks the same row and the same End state; `synth.grow` composes
    the rest out of the observed ones, which is the same growth the Starting state already uses.
    """
    targets = {}
    for table in sorted(world.schema.tables or ()):
        held = len(synth.table_rows(world.db, world.schema, table))
        if 0 < held < wanted:
            targets[table] = wanted
    if not targets:
        return
    grown = synth.grow(world.db, world.schema, targets, seed=0)
    world.schema.synthetic_rows = sorted(set(world.schema.synthetic_rows or ()) | set(grown.ids))


def load_graph(workdir: Any) -> dict:
    """The graph this workdir stored, or an empty one where nothing has mined it yet."""
    body = read_json(Path(workdir) / graph_mod.FILE_NAME, None)
    return body if isinstance(body, dict) else {"format": graph_mod.FORMAT, "runs": 0,
                                                "nodes": [], "edges": [], "args": {}}


def mine_graph(workdir: Any, world: Optional[World] = None) -> dict:
    """Mine the graph off every Run this workdir holds and write it down.

    The Runs and not the raw traces: a Run is the recording as the harness reads it, calls paired
    with the answers they were given, which is what an edge is mined from.
    """
    workdir = Path(workdir)
    world = world if world is not None else World(workdir)
    runs = []
    for path in sorted(difficulty.run_paths(workdir).values()):
        try:
            runs.append(variants_mod.call_results(verifier_suite.as_run(path)))
        except (OSError, ValueError, TypeError):
            continue
    body = graph_mod.mine(runs, world.write_tools)
    write_json(workdir / graph_mod.FILE_NAME, body)
    return body


def _task_id(ident: str) -> str:
    return "synth_" + str(abs(hash(ident)) % (16 ** 12)).rjust(12, "0")[:12]


def store_dir(workdir: Any) -> Path:
    return Path(workdir) / DIR


def read_index(workdir: Any) -> dict:
    body = read_json(store_dir(workdir) / INDEX, None)
    return body if isinstance(body, dict) else {"format": FORMAT, "tasks": [], "counts": {}}


def counts_of(workdir: Any) -> dict:
    """What the synthetic store holds, as a round line and a report read it.

    Every number here is its own; none of them is added to fidelity, to trusted or to a Reference
    count anywhere, because the store they are read from is not the Task store.
    """
    body = read_index(workdir)
    rows = body.get("tasks") or []
    per_bucket: dict[str, dict] = {}
    for row in rows:
        bucket = str(row.get("bucket") or "")
        held = per_bucket.setdefault(bucket, {"bucket": bucket, "tasks": 0, "verified": 0})
        held["tasks"] += 1
        held["verified"] += 1 if row.get("suite_passed") else 0
    counts = dict(body.get("counts") or {})
    counts.update({"synthetic_tasks": len(rows),
                   "synthetic_verified": sum(1 for row in rows if row.get("suite_passed")),
                   "synthetic_buckets": [per_bucket[key] for key in sorted(per_bucket)]})
    return counts


def synthesise(workdir: Any, buckets: dict[str, int], *, seed: Optional[str] = None,
               model: Any = None, on_task: Optional[Callable] = None) -> dict:
    """Generate the Tasks a request asks for, run them, derive their Verifiers and store them apart.

    `buckets` is `{bucket name: how many}`; the walk targets the band's writes and tools by
    construction and its paths by looking for a rewrite of its own call path afterwards. What the
    walk actually reached is the bucket the Task is filed under, which may not be the one asked for:
    a band the graph cannot reach is answered with the walk it did reach and a count that says so.
    """
    workdir = Path(workdir)
    world = World(workdir)
    salt = seed if seed is not None else build_salt(workdir)
    body = load_graph(workdir)
    if not body.get("edges"):
        body = mine_graph(workdir, world)
    observed = body.get("args") or {}
    fn = verifier_suite.canon_fn(world.canon_rules)
    tried = refused = crashed = unbound = 0
    rows: list[dict] = []
    _grow_for(world, max(buckets.values() or [0]))
    refused_bands: dict[str, str] = {}
    for bucket in sorted(buckets):
        # D224's own finding, taken as a rule: a walk that writes nothing leaves the Starting state,
        # so nothing was derived from it and nothing could be. The band is refused, with the reason.
        if graph_mod.bands(bucket)[0] == 0:
            refused_bands[bucket] = READ_ONLY_REFUSED
            continue
        wanted = int(buckets[bucket])
        made = 0
        for attempt in range(wanted * ATTEMPTS_PER_TASK):
            if made >= wanted:
                break
            ident = f"{salt}:{bucket}:{attempt}"
            steps = graph_mod.walk(body, bucket, ident)
            if not steps:
                break
            tried += 1
            calls, why = run_walk(world, steps, observed, ident, graph=body,
                                  write_tools=world.write_tools)
            if why == REFUSED:
                refused += 1
                continue
            if why == CRASHED:
                crashed += 1
                continue
            if why == UNBOUND or not calls:
                unbound += 1
                continue
            row = _make_task(world, body, calls, bucket, ident, salt, model=model, fn=fn)
            if row is None:
                crashed += 1
                continue
            rows.append(row)
            made += 1
            if on_task is not None:
                on_task(row)
    index = {"format": FORMAT, "requested": {str(k): int(v) for k, v in buckets.items()},
             "salt": str(salt), "bands_refused": refused_bands,
             "counts": {"walks_tried": tried, "walks_refused": refused, "walks_crashed": crashed,
                        "walks_unbound": unbound, "bands_refused": len(refused_bands),
                        "graph": graph_mod.summary(body)},
             "tasks": rows}
    write_json(store_dir(workdir) / INDEX, index)
    return index


def _make_task(world: World, body: dict, calls: list[dict], bucket: str, ident: str, salt: str,
               *, model: Any = None, fn: Any = None) -> Optional[dict]:
    """One kept walk as a stored Task: the Run, the Verifier, the suite and the row that reports it."""
    task_id = _task_id(ident)
    askable = askable_fields(world.workdir)
    said, held = facts_of(calls, askable)
    fallback = written_intent(calls, world.write_tools)
    text, refusal = intent_text(model, said, held, fallback)
    transcript, spoken = spoken_for(text, calls)
    run = record_run(world, task_id, spoken, f"synth-{task_id}", transcript)
    if run is None:
        return None
    try:
        reference = reference_mod.load(run["path"], reference_mod.REROLL, run_id=run["run_id"],
                                       write_tools=world.write_tools, fn=fn, atoms=())
    except (OSError, ValueError, TypeError):
        return None
    alternates = second_paths(world, task_id, spoken, transcript, reference, f"synth-{task_id}")
    task = Task(id=task_id, run_ids=[run["run_id"]], intent=text, name=None)
    verifier = derive_mod.derive_verifier(task, run["path"],
                                          rerun_paths=[row["path"] for row in alternates],
                                          canon=world.canon_rules, write_tools=world.write_tools,
                                          intent=text)
    passed, failures = _suite(world, verifier, run, alternates, text)
    writes, tools = graph_mod.reached([{"tool": call["name"]} for call in calls], world.write_tools)
    reached = difficulty.bucket_key(difficulty.write_count(verifier), tools, 1 + len(alternates))
    store = store_dir(world.workdir)
    write_json(store / "tasks" / f"{task_id}.json",
               {"format": FORMAT, "synthetic": True, "task": as_dict(task), "bucket_requested": bucket,
                "bucket": reached, "seed": str(salt), "walk": [{"tool": c["name"], "args": c["args"]}
                                                               for c in calls],
                "user_rules": as_dict(user_rules_for(said)),
                "record_facts": [row["field"] for row in held],
                "intent_refused": refusal})
    write_json(store / "verifiers" / f"{task_id}.json", as_dict(verifier))
    return {"task_id": task_id, "bucket_requested": bucket, "bucket": reached,
            "writes": writes, "tools": tools, "pool": len(alternates),
            "suite_passed": passed, "suite_failures": failures,
            "intent_written": bool(model is not None and not refusal),
            "run_id": run["run_id"]}


def _suite(world: World, verifier: Verifier, run: dict, alternates: list[dict],
           intent: str) -> tuple[bool, list[str]]:
    """The D79 suite over a synthetic Task, run the way it runs over any other Task."""
    try:
        wrong = verifier_suite.wrong_run(verifier, run["path"], canon=world.canon_rules)
        alt = alternates[0]["path"] if alternates else None
        results = verifier_suite.validate_verifier(
            verifier, run["path"], wrong_run=wrong, alt_path_run=alt, intent_text=intent,
            canon=world.canon_rules, write_tools=world.write_tools,
            seed_runs=[row["path"] for row in alternates])
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, [f"the suite could not run: {type(error).__name__}"]
    failed = [result.stage for result in results if not result.passed
              and not (result.metrics or {}).get("skipped")]
    return (not failed), failed


# --- shaped walks: a Task that is an instance of something the domain attests (D225) ---

# The rungs of the realism bar, in the order they are climbed. A Task falls at the first it misses
# and the fall is counted there, so a reader sees which rung is the bottleneck rather than a total.
RUNGS = ("attested", "executable", "graded", "uncontaminated", "plausible")

# The band a shaped walk is filed under when its archetype maps onto n writes. A read only band is
# refused outright (D224's own finding), so this never returns a w0 name.
SHAPED_PATHS = 1

READ_ONLY_REFUSED = ("a walk that writes nothing leaves the Starting state unchanged, so the derive "
                     "path has no claim to make and no Verifier to check: the read only band is "
                     "refused rather than filled (D224)")


def shaped_bucket(write_tools: Iterable[str], tools: Iterable[str]) -> str:
    """The band a shaped walk asks for: the writes its archetype maps onto and the tools it names."""
    writes = len({str(name) for name in write_tools or ()})
    return difficulty.bucket_key(writes, max(len({str(name) for name in tools or ()}), writes),
                                 SHAPED_PATHS)


ARCHETYPE_INTENT = (
    "A person is writing to a support agent about a problem they want solved. Write their first "
    "message.\n\n"
    "This is what they want, in their own words:\n  {goal}\n\n"
    "Write the symptom and the outcome they want, in their own words, in one or two sentences. Do "
    "not name any tool, any system, any table or any step, and do not say how the agent should do "
    "it. Do not include any value that is not in the list below.\n\n"
    "The only values this person knows and may state:\n{facts}\n\n"
    "Write the message and nothing else.")


def archetype_intent(model: Any, record: dict, said: list[dict], held: list[dict],
                     fallback: str) -> tuple[str, str]:
    """The Intent of a shaped Task: symptom form (D196) written from the archetype's own goal line.

    The goal is the customer's voice, so the Intent is written from it rather than from the walk;
    the askable values are still D210's, and a draft carrying a value only the world knows is
    refused rather than edited, exactly as the unshaped writer refuses one.
    """
    if model is None:
        return fallback, "no Intent writer was given"
    lines = "\n".join(f"- {row['field']}: {row['value']}" for row in said) or "- nothing"
    try:
        reply = model.query([{"role": "user", "content": ARCHETYPE_INTENT.format(
            goal=str(record.get("goal") or ""), facts=lines)}])
    except Exception as error:  # noqa: BLE001 - a provider failure leaves the code-written Intent
        return fallback, f"the Intent writer failed: {type(error).__name__}"
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return fallback, "the Intent writer answered nothing"
    if [row for row in held if str(row["value"]) and str(row["value"]) in text]:
        return fallback, "the draft stated a value only the world knows"
    return text, ""


def shape(workdir: Any, *, archetype_ids: Iterable[str] = (), per_archetype: int = 1,
          seed: Optional[str] = None, model: Any = None, judges: Iterable[Any] = (),
          bucket: Optional[str] = None) -> dict:
    """Tasks shaped to what the domain's own material attests, each climbing the realism bar (D225).

    An archetype says what a person asks for; the graph says what this Environment can do; a shaped
    walk is the intersection, and it is still a walk of recorded edges bound to reachable rows. The
    D224 bands stay as filters over it: the band is read off the archetype's mapped writes, and the
    read only band is refused with a reason rather than filled.

    Every Task climbs five rungs in order, and the count of what fell at each is the measurement:
    attested, so the Task names an archetype and the archetype a source; executable, so the walk ran
    in the rebuilt world and left an End state; graded, so the Verifier passes the D79 suite;
    uncontaminated, so no string of the corpus's own Task list is in the Intent; and plausible, so
    neither judge could cite a line of the archetype the Task contradicts. A judge can never pass a
    Task, only remove one.
    """
    from kullback import domain as domain_mod

    workdir = Path(workdir)
    if bucket is not None and graph_mod.bands(bucket)[0] == 0:
        return {"format": FORMAT, "refused": READ_ONLY_REFUSED, "tasks": [],
                "counts": {rung: 0 for rung in RUNGS}}
    world = World(workdir)
    salt = seed if seed is not None else build_salt(workdir)
    body = load_graph(workdir)
    if not body.get("edges"):
        body = mine_graph(workdir, world)
    observed = body.get("args") or {}
    fn = verifier_suite.canon_fn(world.canon_rules)
    read = domain_mod.read_domain(workdir)
    wanted = {str(name) for name in archetype_ids or ()}
    records = [row for row in read.get("archetypes") or ()
               if row.get("write_tools") and (not wanted or str(row.get("id")) in wanted)]
    strings = domain_mod.corpus_strings(workdir)
    fell = {rung: 0 for rung in RUNGS}
    rows: list[dict] = []
    per_source_rows: dict[str, int] = {}
    _grow_for(world, max(int(per_archetype), 1) * max(len(records), 1))
    for record in records:
        asked = shaped_bucket(record.get("write_tools"), record.get("tools"))
        if graph_mod.bands(asked)[0] == 0:
            fell["attested"] += 1
            continue
        made = 0
        for attempt in range(max(int(per_archetype), 1) * ATTEMPTS_PER_TASK):
            if made >= max(int(per_archetype), 1):
                break
            ident = f"{salt}:{record.get('id')}:{attempt}"
            steps = graph_mod.walk(body, asked, ident, must_visit=record.get("write_tools") or ())
            if not steps:
                fell["attested"] += 1
                continue
            calls, why = run_walk(world, steps, observed, ident, graph=body,
                                  write_tools=world.write_tools)
            if why or not calls:
                fell["executable"] += 1
                continue
            row = _shaped_task(world, calls, record, asked, ident, salt, strings, judges,
                               model=model, fn=fn, fell=fell)
            if row is None:
                continue
            rows.append(row)
            made += 1
            for url in record.get("sources") or [record.get("source")]:
                per_source_rows[str(url)] = per_source_rows.get(str(url), 0) + 1
    index = {"format": FORMAT, "salt": str(salt), "shaped": True,
             "counts": {"archetypes_read": len(records), "tasks_shaped": len(rows),
                        "fell": fell,
                        "per_source": [{"source": key, "tasks": per_source_rows[key]}
                                       for key in sorted(per_source_rows)]},
             "tasks": rows}
    write_json(store_dir(workdir) / "shaped.json", index)
    return index


def _shaped_task(world: World, calls: list[dict], record: dict, bucket: str, ident: str, salt: str,
                 strings: list[str], judges: Iterable[Any], *, model: Any = None, fn: Any = None,
                 fell: dict) -> Optional[dict]:
    """One shaped walk climbing the rungs above the executable one, or nothing and a count."""
    from kullback import domain as domain_mod

    task_id = _task_id(ident)
    said, held = facts_of(calls, askable_fields(world.workdir))
    fallback = written_intent(calls, world.write_tools)
    text, refusal = archetype_intent(model, record, said, held, fallback)
    transcript, spoken = spoken_for(text, calls)
    run = record_run(world, task_id, spoken, f"synth-{task_id}", transcript)
    if run is None:
        fell["executable"] += 1
        return None
    try:
        reference = reference_mod.load(run["path"], reference_mod.REROLL, run_id=run["run_id"],
                                       write_tools=world.write_tools, fn=fn, atoms=())
    except (OSError, ValueError, TypeError):
        fell["executable"] += 1
        return None
    alternates = second_paths(world, task_id, spoken, transcript, reference, f"synth-{task_id}")
    task = Task(id=task_id, run_ids=[run["run_id"]], intent=text, name=None)
    verifier = derive_mod.derive_verifier(task, run["path"],
                                          rerun_paths=[row["path"] for row in alternates],
                                          canon=world.canon_rules, write_tools=world.write_tools,
                                          intent=text)
    passed, failures = _suite(world, verifier, run, alternates, text)
    if not passed:
        fell["graded"] += 1
        return None
    if domain_mod.contaminated({"goal": text}, strings):
        fell["uncontaminated"] += 1
        return None
    rejections = domain_mod.judged(judges, record, text, [call["name"] for call in calls])
    if rejections:
        fell["plausible"] += 1
        return None
    writes, tools = graph_mod.reached([{"tool": call["name"]} for call in calls], world.write_tools)
    reached = difficulty.bucket_key(difficulty.write_count(verifier), tools, 1 + len(alternates))
    store = store_dir(world.workdir)
    write_json(store / "tasks" / f"{task_id}.json",
               {"format": FORMAT, "synthetic": True, "task": as_dict(task),
                "archetype": record.get("id"), "source": record.get("source"),
                "bucket_requested": bucket, "bucket": reached, "seed": str(salt),
                "walk": [{"tool": c["name"], "args": c["args"]} for c in calls],
                "user_rules": as_dict(user_rules_for(said)),
                "record_facts": [row["field"] for row in held],
                "intent_refused": refusal})
    write_json(store / "verifiers" / f"{task_id}.json", as_dict(verifier))
    return {"task_id": task_id, "archetype": record.get("id"), "source": record.get("source"),
            "bucket_requested": bucket, "bucket": reached, "writes": writes, "tools": tools,
            "pool": len(alternates), "suite_passed": passed, "suite_failures": failures,
            "intent_written": bool(model is not None and not refusal), "run_id": run["run_id"]}


def shaped_counts(workdir: Any) -> dict:
    """What the last shaping left, as a round line and a report read it (D225)."""
    body = read_json(store_dir(workdir) / "shaped.json", None)
    if not isinstance(body, dict):
        return {}
    counts = dict(body.get("counts") or {})
    return {"tasks_shaped": int(counts.get("tasks_shaped") or 0),
            "shaped_fell": dict(counts.get("fell") or {}),
            "shaped_per_source": list(counts.get("per_source") or [])}


# --- the synthetic store: a task description over a kept walk (D360) ----------------------

# The store is a workdir-shaped directory with its own tasks, runs, references and replays, linked
# against the parent's Environment files. The Spec writer, the Examiner and the trust chain run on
# it unchanged: a Task with no recording reads its instruction as one source (intent_tools seam).
# The walk is the Task's Reference. This path never derives a Verifier.
STORE = "store"
STORE_INDEX = "index.json"

# The parent's Environment files the store links to: symlinks, never copies of the database. The
# store holds its own tasks, runs, references, replays, overlays and index beside these links.
STORE_LINKS = ("env", "schema.json", "tool_sigs.json", "bodies.json", "canon-rules.json",
               "db.json", "environment.json", "readers.json", "user_facts.json", "anchor.json")

# Words in the shipped policy that ask the customer to confirm before a write. Plain words of the
# harness's own language, never a corpus name: where the policy asks, the description carries the
# customer's go-ahead, so a confirm-before-write gate has something to rest on.
CONFIRM_WORDS = ("confirm", "approval", "consent", "permission", "go-ahead", "authorize",
                 "authorise")

# Where the policy asks, the go-ahead is the one user turn a confirm-before-write gate can read: a
# sentence opening on a yes, and no sentence opening on a no (the Verdict's own confirm reading).
CONSENT_LINE = ("They give their go-ahead as a sentence of its own that opens with \"Go ahead\", and "
                "no sentence of the message opens with a negative word such as no, do not or don't.")
CONSENT_RETRY = ("Your message did not give the go-ahead as asked; write it again with a sentence "
                 "opening with \"Go ahead\" and no sentence opening with a negative word.")
LEAK_RETRY = "Your message repeated this text; write it again without it: "

# How many recorded Tasks are read looking for a style example of the walk's own write tools.
STYLE_SEARCH_LIMIT = 200

DESCRIPTION_PROMPT = (
    "A person is writing to a support agent about a problem they want solved. Write their first "
    "message, in the first person.\n\n"
    "What they want done:\n{ask}\n\n"
    "Things only they know, to state in their own words:\n{facts}\n\n"
    "{extra}"
    "State every value listed above exactly as written. Do not name any tool, system, table or "
    "step, and add no value that is not listed. Write the message and nothing else.")


def store_path(workdir: Any) -> Path:
    """Where the synthetic store lives: a workdir-shaped directory under the parent's synthetic one."""
    return store_dir(workdir) / STORE


def _link_store(parent: Path) -> Path:
    """Lay out the store on first use: the marker, the links to the parent's Environment, and the
    store's own directories. Later runs find the marker and only repair missing links."""
    from kullback.spec.intent_tools import SYNTHETIC_MARKER, SYNTHETIC_ORIGIN

    store = store_path(parent)
    store.mkdir(parents=True, exist_ok=True)
    if not (store / SYNTHETIC_MARKER).is_file():
        write_json(store / SYNTHETIC_MARKER, {"format": FORMAT, "parent": "..",
                                              "origin": SYNTHETIC_ORIGIN})
    for name in STORE_LINKS:
        source, link = parent / name, store / name
        if source.exists() and not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(source, store),
                            target_is_directory=source.is_dir())
    for folder in ("tasks", "runs", "overlays"):
        (store / folder).mkdir(parents=True, exist_ok=True)
    return store


def _store_policy(store: Path) -> str:
    """The shipped policy text the store links to, or nothing where the parent ships none."""
    try:
        return (store / "env" / "policy.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


def _store_consent(store: Path) -> bool:
    """Whether the shipped policy asks the customer to confirm before a write."""
    policy = _store_policy(store)
    return bool(policy) and any(word in policy.casefold() for word in CONFIRM_WORDS)


def _write_names_of(parent: Path) -> set[str]:
    """The parent's write tools, read off its signatures, so a style example matches the walk."""
    try:
        sigs = read_json(parent / "tool_sigs.json", []) or []
    except (OSError, ValueError):
        return set()
    return {str(row.get("name")) for row in sigs if isinstance(row, dict)
            and row.get("kind") == "write"}


def _style_example(parent: Path, writes: set[str]) -> str:
    """The opening user turn of one recorded Task with the walk's own write tools, or nothing.

    Style only: the prompt tells the model never to carry its details over. A workdir with no
    recordings, or none with the same writes, writes its descriptions without an example."""
    if not writes:
        return ""
    targets = set(writes)
    folder = parent / "tasks"
    paths = sorted(folder.glob("*.json"))[:STYLE_SEARCH_LIMIT] if folder.is_dir() else []
    kinds = _write_names_of(parent)
    for path in paths:
        opening = _trace_opening(parent, path, targets, kinds)
        if opening:
            return opening
    return ""


def _trace_opening(parent: Path, path: Path, targets: set[str], kinds: set[str]) -> str:
    """One recorded Task's opening user turn where its writes match the walk's, or nothing."""
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(task, dict):
        return ""
    for run_id in task.get("run_ids") or ():
        trace = _trace_of(parent, run_id)
        if trace is None:
            continue
        made = {str(call.get("name")) for call in trace.get("tool_calls") or ()} & kinds
        if made != targets:
            continue
        for turn in trace.get("turns") or ():
            if isinstance(turn, dict) and turn.get("role") == "user" \
                    and str(turn.get("content") or "").strip():
                return str(turn["content"])
    return ""


def _trace_of(parent: Path, run_id: str) -> Optional[dict]:
    """One recorded trace, or nothing where it is missing or torn."""
    try:
        trace = json.loads((parent / "traces" / f"{run_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return trace if isinstance(trace, dict) else None


# Words of a grader and never of a person asking for help: a description carrying one leaks the check.
GRADER_WORDS = ("rubric", "grader", "graded", "grading", "score", "scored", "scoring", "verifier",
                "checklist", "checker", "test case", "evaluation", "criteria")
# A line that opens as a heading or a list item, or bold markers anywhere: a form, not a message.
FORMATTED = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s)|\*\*", re.MULTILINE)


def _family(tool: str) -> str:
    """A tool's family: the first word of its name, read through underscores and camel case."""
    parts = [part for part in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", str(tool)) if part]
    return parts[0].casefold() if parts else ""


def unreal_by_code(text: str, tools: Iterable[str], walk_writes: Iterable[str],
           write_tools: Iterable[str]) -> str:
    """Why a description does not read as a person's request, as a drop reason; "" when it does.

    Grader words leak the check; a tool name is the harness speaking; a heading or a list is a form;
    a write family the walk never made is a second, unrelated ask bolted on (named by its first word).
    """
    lowered = str(text or "").casefold()
    words = set(re.findall(r"[a-z0-9][a-z0-9'-]*", lowered))
    if any((phrase in lowered) if " " in phrase else (phrase in words) for phrase in GRADER_WORDS):
        return "grader_words"
    if any(str(name).casefold() in lowered for name in tools if name):
        return "tool_named"
    if FORMATTED.search(str(text or "")):
        return "formatted"
    made = {_family(name) for name in walk_writes}
    others = {_family(name) for name in write_tools if name not in set(walk_writes)} - made
    if words & {family for family in others if family}:
        return "second_ask"
    return ""


def _description_inputs(calls: list[dict], refused: list[str], writes: Iterable[str] = ()
                        ) -> tuple[list[dict], list[dict], list[str]]:
    """The description's matter: customer-known values, the ask per write, and refused values.

    A bound value the walk already read from an earlier answer is not the customer's to state;
    every other value is, ids included: an id no answer carried is the customer's own. The ask is
    each write's own values no earlier answer carried: the change the customer wants. A refused
    value the walk never read is the customer's own constraint."""
    facts, read_values = _customer_facts(calls)
    ask = _ask_for(calls, set(writes))
    seen: set[str] = set()
    refusals = [value for value in refused or ()
                if value and value not in read_values and value not in seen
                and not seen.add(value)]
    return facts, ask, refusals


def _leaf_field(path: str) -> str:
    """One argument path's field: the last name, a list marker collapsed."""
    return str(path).split(".")[-1].replace("[]", "")


def _customer_facts(calls: list[dict]) -> tuple[list[dict], set[str]]:
    """The bound values the customer states, and every value the walk read from the world.

    A value an earlier answer already carried is not the customer's to state; every other value
    is, ids included.
    """
    facts: list[dict] = []
    pairs: set[tuple[str, str]] = set()
    read_values: set[str] = set()
    for call in calls or ():
        for path, value in graph_mod.argument_leaves(call):
            field = _leaf_field(path)
            if value and (field, value) not in pairs:
                pairs.add((field, value))
                if value not in read_values:
                    facts.append({"field": field, "value": value})
        for _path, text in graph_mod._leaves(call.get("result")):  # noqa: SLF001
            read_values.add(text)
    return facts, read_values


def _ask_for(calls: list[dict], writes: set[str]) -> list[dict]:
    """Each write's values no earlier answer carried: the change the customer wants, by field.

    A value the walk read first (a row the agent finds) is the agent's to look up, never asked."""
    ask: list[dict] = []
    asked: set[tuple[str, str]] = set()
    read_values: set[str] = set()
    for call in calls or ():
        if str(call.get("name")) in writes:
            for path, value in graph_mod.argument_leaves(call):
                field = _leaf_field(path)
                if value and value not in read_values and (field, value) not in asked:
                    asked.add((field, value))
                    ask.append({"field": field, "value": value})
        read_values.update(text for _path, text in graph_mod._leaves(call.get("result")))  # noqa: SLF001
    return ask


def description_text(model: Any, facts: list[dict], ask: list[dict], refusals: list[str], *,
                     consent: bool = False, example: str = "",
                     retry: str = "") -> tuple[str, str]:
    """The customer's first-person message from the walk's matter, or nothing and why not.

    `retry` is the one line code sends back with a draft it refused: what to write again and why."""
    if model is None:
        return "", "no description writer was given"
    lines = "\n".join(f"- {row['field']}: {row['value']}" for row in facts) or "- nothing"
    wanted = "\n".join(f"- {row['field']}: {row['value']}" for row in ask) \
        or "what the values above point to"
    extra = ""
    if refusals:
        extra += "It must refuse this, as their own constraint:\n" + "\n".join(
            f"- {value}" for value in refusals) + "\n\n"
    if consent:
        extra += CONSENT_LINE + "\n\n"
    if example:
        extra += ("Match the tone of this unrelated request, never its details:\n" + example
                  + "\n\n")
    if retry:
        extra += retry + "\n\n"
    try:
        reply = model.query([{"role": "user", "content": DESCRIPTION_PROMPT.format(
            ask=wanted, facts=lines, extra=extra)}])
    except Exception as error:  # noqa: BLE001 - a provider failure drops the walk, never guesses
        return "", f"the description writer failed: {type(error).__name__}"
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return "", "the description writer answered nothing"
    return text, ""


def consents(text: str) -> bool:
    """Whether a message gives the go-ahead the way the Verdict reads one before a write: a sentence
    opening on a yes and none opening on a no."""
    from kullback.runner.atom_context import AFFIRM_OPEN, NEGATIVE_OPEN, _opening

    sentences = [_opening(part) for part in re.split(r"[.!?\n]+", text or "") if _opening(part)]
    return any(AFFIRM_OPEN.match(s) for s in sentences) and \
        not any(NEGATIVE_OPEN.match(s) for s in sentences)


def _refusal_of(text: str, strings: list[str], consent: bool) -> tuple[str, str]:
    """Why code refuses a draft, as the line sent back and the count it falls under; empty if kept."""
    from kullback import domain as domain_mod

    leaked = domain_mod.contaminated({"goal": text}, strings)
    if leaked:
        return LEAK_RETRY + leaked, "contaminated"
    if consent and not consents(text):
        return CONSENT_RETRY, "unconsented"
    return "", ""


def _described(model: Any, facts: list[dict], ask: list[dict], refusals: list[str],
               consent: bool, example: str, strings: list[str]) -> tuple[str, str]:
    """The description, sent back once with the reason when code refuses the first draft; else
    nothing and the count the walk is dropped under (undescribed, contaminated, unconsented)."""
    text, _why = description_text(model, facts, ask, refusals, consent=consent, example=example)
    if not text:
        return "", "undescribed"
    retry, _drop = _refusal_of(text, strings, consent)
    if not retry:
        return text, ""
    text, _why = description_text(model, facts, ask, refusals, consent=consent, example=example,
                                  retry=retry)
    if not text:
        return "", "undescribed"
    _retry, drop = _refusal_of(text, strings, consent)
    return ("", drop) if drop else (text, "")


def build_store(workdir: Any, buckets: dict[str, int], *, seed: Optional[str] = None,
                model: Any = None, realism: Optional[Callable[[str], dict]] = None) -> dict:
    """Keep walks as scenario sources of Tasks carrying a description, in a store linked to this one.

    For each kept walk: bind and run it through the bodies as today; write its description from one
    model call over the walk's matter; scan the description against the corpus strings, rewriting it
    once and dropping the walk where it stays contaminated; drop a description that reads as a form
    or a grader by code; hold every walk argument against the description or an earlier result by
    code, dropping unfair walks; ask the realism step, where one is given, whether the request
    happens in the domain. The walk is kept as the Task's scenario source and never as a Reference:
    the Verifier comes from the description, the policy, the tools and the Starting state."""
    from kullback import domain as domain_mod

    parent = Path(workdir)
    store = _link_store(parent)
    world = World(store)
    salt = seed if seed is not None else build_salt(parent)
    body = load_graph(parent)
    if not body.get("edges"):
        body = mine_graph(parent, World(parent))
    observed = body.get("args") or {}
    strings = domain_mod.corpus_strings(parent)
    consent = _store_consent(store)
    _grow_for(world, max(int(count) for count in buckets.values()) if buckets else 0)
    counts = {"walks_tried": 0, "walks_refused": 0, "walks_unbound": 0, "walks_crashed": 0,
              **{f"walks_{reason}": 0 for reason in DROP_REASONS},
              "realism_not_run": 0, "tasks_kept": 0}
    unreal_rows: list[dict] = []
    per_bucket: dict[str, int] = {}
    rows: list[dict] = []
    status: dict[str, dict] = {}
    for bucket in sorted(buckets):
        if graph_mod.bands(bucket)[0] == 0:
            continue
        wanted = int(buckets[bucket])
        made = 0
        for attempt in range(wanted * ATTEMPTS_PER_TASK):
            if made >= wanted:
                break
            calls, note, ident = _attempt_walk(world, body, observed, bucket, salt, attempt,
                                              counts)
            if ident is None:
                break
            if not calls:
                continue
            row, drop = _store_task(world, parent, store, calls, note, bucket, ident,
                                    model, strings, consent, status, realism, unreal_rows)
            if row is None:
                counts[f"walks_{drop}"] += 1
                continue
            counts["realism_not_run"] += 1 if row["realism"] == REALISM_NOT_RUN else 0
            rows.append(row)
            made += 1
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
    return _write_store_index(store, buckets, salt, counts, per_bucket, rows, status, unreal_rows)


def _attempt_walk(world: World, body: dict, observed: dict, bucket: str, salt: str,
                  attempt: int, counts: dict) -> tuple[list[dict], dict, Optional[str]]:
    """One walk proposed, bound and run through the bodies, counted.

    No ident where the graph offers no walk for the bucket; no calls where the walk fell."""
    ident = f"{salt}:{bucket}:{attempt}"
    steps = graph_mod.walk(body, bucket, ident)
    note: dict = {}
    if not steps:
        return [], note, None
    counts["walks_tried"] += 1
    calls, why = run_walk(world, steps, observed, ident, graph=body,
                          write_tools=world.write_tools, note=note)
    if why == REFUSED:
        counts["walks_refused"] += 1
        return [], note, ident
    if why == CRASHED:
        counts["walks_crashed"] += 1
        return [], note, ident
    if why == UNBOUND or not calls:
        counts["walks_unbound"] += 1
        return [], note, ident
    return calls, note, ident


def _write_store_index(store: Path, buckets: dict, salt: str, counts: dict,
                       per_bucket: dict, rows: list[dict], status: dict,
                       unreal_rows: list[dict]) -> dict:
    """The store's status and index, written once every walk is kept; the index names the source
    the realism step leaned on for every walk it dropped as unreal."""
    write_json(store / "task_status.json", status)
    counts["tasks_kept"] = len(rows)
    index = {"format": FORMAT, "requested": {str(k): int(v) for k, v in buckets.items()},
             "salt": str(salt),
             "counts": dict(counts, per_bucket={str(k): int(v) for k, v in per_bucket.items()}),
             "unreal": unreal_rows, "tasks": rows}
    write_json(store / STORE_INDEX, index)
    return index


# Why a walk is dropped after it ran, each a count of its own in the store index.
DROP_REASONS = ("contaminated", "unfair", "undescribed", "unconsented", "grader_words", "tool_named",
                "formatted", "second_ask", "unreal", "unjudged", "crashed")
# The realism field of a kept row where no realism step was given.
REALISM_NOT_RUN = "not_run"
# Where a store Task's scenario source sits: the walk's steps and the rows it touched, per Task.
SCENARIOS = "scenarios"


def scenario_source(calls: list[dict]) -> dict:
    """The walk as a scenario source: its steps and the row ids it read or wrote, never a Run."""
    rows: set[str] = set()
    for call in calls:
        rows |= set(variants_mod.call_rows(call))
    return {"steps": [{"tool": str(call.get("name")), "args": call.get("args") or {}} for call in calls],
            "rows": sorted(rows)}


REALISM_PROMPT = (
    "Someone wrote this message to the support desk of a business:\n\n{text}\n\n"
    "Below are public pages about that kind of business. Using only these pages, say whether a request "
    "like this one is something that actually happens there: customers really ask for it and the "
    "business really handles it. Lean on one page.\n\n{pages}\n\n"
    'Answer with JSON only, as {{"happens": "yes" or "no", "source": "the url of the one page you '
    'leaned on", "why": "one short line"}}.')
# How many pages a realism question reads, and how much of each.
REALISM_PAGES = 3
REALISM_CHARS = 4000


def domain_realism(model: Any, *, search: Optional[Callable[[str, str], list[str]]] = None,
                   fetch: Optional[Callable[[str], str]] = None, env: Optional[dict] = None,
                   corpus_url: Optional[str] = None, exclude: Iterable[str] = ()
                   ) -> Callable[[str], dict]:
    """The realism step: does a request like this description happen in the domain, and on what page.

    It reads the description only, never the walk or the tools. The description is the search query
    (D225's search, run only where a search key is set); up to REALISM_PAGES public pages are fetched
    through D225's fetch, which refuses non-public destinations, and never under the corpus's own
    pages. One model call answers yes or no and names the one page it leaned on; a source that is not
    one of the pages read, or no pages at all, judges nothing (`happens` None)."""
    from kullback import domain as domain_mod

    reader = fetch if fetch is not None else domain_mod.http_fetch

    def judge(text: str) -> dict:
        urls, note = domain_mod.search_sources(text, search=search, env=env)
        pages: list[tuple[str, str]] = []
        for url in urls:
            if len(pages) >= REALISM_PAGES or domain_mod.refusal(url, corpus_url, exclude):
                continue
            try:
                body = domain_mod.page_text(reader(url))
            except Exception:  # noqa: BLE001 - a page that will not read is one page fewer
                continue
            if body:
                pages.append((url, body[:REALISM_CHARS]))
        if not pages:
            return {"happens": None, "source": "", "why": note or "no page could be read"}
        listing = "\n\n".join(f"Page {url}:\n{body}" for url, body in pages)
        answer = domain_mod._json_reply(model, REALISM_PROMPT.format(text=text, pages=listing))  # noqa: SLF001
        answer = answer if isinstance(answer, dict) else {}
        source = str(answer.get("source") or "")
        said = str(answer.get("happens") or "").strip().casefold()
        if source not in {url for url, _body in pages} or said not in ("yes", "no"):
            return {"happens": None, "source": source, "why": "the realism answer named no page read"}
        return {"happens": said == "yes", "source": source, "why": str(answer.get("why") or "")}

    return judge


def _realism_of(realism: Optional[Callable[[str], dict]], text: str) -> tuple[str, dict]:
    """The realism step's answer as (drop reason or "", the answer); no step leaves it not run."""
    if realism is None:
        return "", {"happens": None, "source": "", "why": REALISM_NOT_RUN}
    try:
        answer = dict(realism(text) or {})
    except Exception as error:  # noqa: BLE001 - a step that fails judges nothing, and is counted
        answer = {"happens": None, "source": "", "why": f"the realism step failed: {type(error).__name__}"}
    if answer.get("happens") is True:
        return "", answer
    return ("unreal" if answer.get("happens") is False else "unjudged"), answer


def _store_task(world: World, parent: Path, store: Path, calls: list[dict], note: dict,
                bucket: str, ident: str, model: Any, strings: list[str], consent: bool,
                status: dict, realism: Optional[Callable[[str], dict]] = None,
                unreal_rows: Optional[list] = None) -> tuple[Optional[dict], str]:
    """One kept walk as a store Task, or nothing and the count to raise (DROP_REASONS).

    A description that reads as a form or a grader is dropped by its code reason (`unreal_by_code`);
    one the realism step says does not happen in the domain is dropped as unreal, with the source it
    leaned on kept in `unreal_rows`. The Task keeps no Run: the walk is its scenario source."""
    from kullback.runner.world import birth as birth_mod
    from kullback.spec.intent_tools import SYNTHETIC_ORIGIN

    task_id = _task_id(ident)
    writes = {str(call.get("name")) for call in calls} & set(world.write_tools)
    facts, ask, refusals = _description_inputs(calls, note.get("refused") or [], writes)
    text, drop = _described(model, facts, ask, refusals, consent, _style_example(parent, writes),
                            strings)
    if not text:
        return None, drop
    drop = unreal_by_code(text, [sig.name for sig in world.sigs], writes, world.write_tools)
    if drop:
        return None, drop
    trace = birth_mod.check_arguments_trace(
        walk=[{"args": call.get("args")} for call in calls],
        results=[{"id": f"call-{index + 1}", "result": call.get("result")}
                 for index, call in enumerate(calls)],
        intent=text, user_facts=[])
    if trace.outcome != birth_mod.HELD:
        return None, "unfair"
    drop, answer = _realism_of(realism, text)
    if drop:
        if unreal_rows is not None:
            unreal_rows.append({"task_id": task_id, "reason": drop, "source": answer.get("source") or "",
                                "why": answer.get("why") or ""})
        return None, drop
    write_json(store / "tasks" / f"{task_id}.json",
               as_dict(Task(id=task_id, run_ids=[], intent=text, facts_in_instruction=False)))
    write_json(store / "overlays" / f"{task_id}.json",
               {"overlay": {"task_id": task_id, "rows": []}, "values": {}})
    write_json(store / SCENARIOS / f"{task_id}.json", scenario_source(calls))
    status[task_id] = {}
    tools = {str(call.get("name")) for call in calls}
    return {"task_id": task_id, "origin": SYNTHETIC_ORIGIN, "bucket_requested": bucket,
            "writes": len(writes), "tools": len(tools),
            "realism": REALISM_NOT_RUN if realism is None else "happens",
            "realism_source": answer.get("source") or ""}, ""


__all__ = ["ARCHETYPE_INTENT", "ATTEMPTS_PER_TASK", "CONFIRM_WORDS", "CONSENT_LINE", "CRASHED", "DIR", "DROP_REASONS", "FORMAT",
           "INDEX", "MAX_INSERTED_READS", "READ_ONLY_REFUSED", "REFUSED", "RUNGS",
           "REALISM_NOT_RUN", "SCENARIOS", "SECOND_PATH_LIMIT", "SHAPED_PATHS", "STORE", "STORE_INDEX", "STORE_LINKS", "UNBOUND",
           "WRITE_RETRIES", "World", "archetype_intent", "askable_fields", "bind", "build_store",
           "consents", "counts_of", "description_text", "domain_realism", "facts_of", "unreal_by_code",
           "intent_text", "load_graph", "mine_graph", "read_index", "record_run", "run_walk",
           "scenario_source", "second_paths", "shape", "shaped_bucket", "shaped_counts", "spoken_for", "store_dir",
           "store_path", "synthesise", "user_rules_for", "written_intent"]
