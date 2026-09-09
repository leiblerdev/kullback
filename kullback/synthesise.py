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
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback import difficulty
from kullback import graph as graph_mod
from kullback.builder import compile_env, mine, synth
from kullback.examiner import derive as derive_mod
from kullback.examiner import reference as reference_mod
from kullback.examiner import variants as variants_mod
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
# A walk is bound and run before it is replayed; this is how deep an argument value is looked for in
# an earlier answer before the walk gives up on the binding.
MISSING_BINDING = "an earlier step's answer did not carry the value the edge names"

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


def bind(world: World, step: dict, answers: list[Any], observed: dict, sigs: dict,
         ident: str) -> Optional[dict]:
    """One step's arguments, in the order the evidence for them gets weaker.

    An edge first, because the recordings showed the value travelling that way; then the nearest
    earlier answer, because a walk that names what it has just been told is a walk that hangs
    together; then one row of the rebuilt world, all of whose values belong to each other; then the
    values the recordings showed for this argument. An argument the signature requires and none of
    the four can fill leaves the walk unbound, which is counted and never guessed at.
    """
    tool = str(step.get("tool"))
    properties, required = declared(sigs.get(tool))
    if not properties:
        return None
    args: dict = {}
    for binding in step.get("bindings") or ():
        head = str(binding["arg"]).split(".")[0].replace("[]", "")
        if head not in properties:
            continue
        source = answers[int(binding["from"])]
        found = graph_mod.values_at(source, str(binding["column"]))
        if not found:
            # The edge is what the recordings showed; this answer did not carry it, so the argument
            # falls back to the sources below rather than the walk falling over on one binding.
            continue
        _set_path(args, str(binding["arg"]),
                  found[sample_key("synth-edge-value", f"{ident}:{binding['arg']}", "") % len(found)])
    owed = [name for name in required if name not in args]
    seen = named_leaves(answers[-1]) if answers else {}
    row = _row_source(world, [name for name in owed if name not in seen], ident)
    for name in owed:
        spec = properties.get(name) or {}
        pool = seen.get(name) or row.get(name) or list(
            ((observed.get(tool) or {}).get(name) or {}).get("values") or [])
        if not pool:
            return None
        args[name] = _typed(spec, pool[sample_key("synth-arg", f"{ident}:{tool}:{name}", "") % len(pool)])
    return {name: value for name, value in args.items() if name in properties}


# --- running one walk ---------------------------------------------------------------

def run_walk(world: World, steps: list[dict], observed: dict, ident: str) -> tuple[list[dict], str]:
    """Bind and run the walk through the Builder's own tool bodies; the calls it made, and why not.

    One toolkit per walk, so what a step wrote the next step sees, which is the whole point of a
    walk. An exception whose class the harness already reads as a body fault (`CRASH_ERRORS`) is a
    crash; every other exception is the body refusing the call, which is the world saying this path
    does not exist. Either way the walk is discarded, never repaired.
    """
    toolkit = world.toolkit(world.fresh_db())
    sigs = {sig.name: sig for sig in world.sigs}
    answers: list[Any] = []
    calls: list[dict] = []
    for index, step in enumerate(steps):
        name = str(step.get("tool"))
        args = bind(world, step, answers, observed, sigs, f"{ident}:{index}")
        if args is None:
            return calls, UNBOUND
        body = getattr(toolkit, name, None)
        if body is None:
            return calls, CRASHED
        try:
            answer = body(**args)
        except Exception as error:  # noqa: BLE001 - the class is the ruling, see CRASH_ERRORS
            return calls, CRASHED if type(error).__name__ in CRASH_ERRORS else REFUSED
        answers.append(answer)
        calls.append({"id": f"{ident}-{index}", "name": name, "args": args,
                      "requestor": "assistant", "turn": index, "result": answer, "error": None})
    return calls, ""


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
               transcript: list[dict]) -> Optional[dict]:
    """The walk replayed once more from a fresh world, so it lands on disk as a Run like any other.

    The same replay every recorded Trace goes through, over the same Router and the same scoring;
    what is different is only where the Trace came from, which is code and not a recording.
    """
    from kullback.builder.build import call_trace  # imported late: build.py imports this module's peers

    db = world.fresh_db()
    toolkit = world.toolkit(db)
    router = route.Router(env_tools_module=toolkit, starting_state=world.fresh_db(), tool_sigs=world.sigs,
                          canon_rules=world.canon_rules, synthetic_rows=world.schema.synthetic_rows)
    trace = call_trace(task_id, calls, transcript, run_id)
    result = replay_mod.replay_trace(trace, router, workdir=world.workdir / DIR / "runs" / task_id,
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
            calls, why = run_walk(world, steps, observed, ident)
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
            calls, why = run_walk(world, steps, observed, ident)
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


__all__ = ["ARCHETYPE_INTENT", "ATTEMPTS_PER_TASK", "CRASHED", "DIR", "FORMAT", "INDEX",
           "READ_ONLY_REFUSED", "REFUSED", "RUNGS", "SECOND_PATH_LIMIT", "SHAPED_PATHS", "UNBOUND",
           "World", "archetype_intent", "askable_fields", "bind", "counts_of", "facts_of",
           "intent_text", "load_graph", "mine_graph", "read_index", "record_run", "run_walk",
           "second_paths", "shape", "shaped_bucket", "shaped_counts", "spoken_for", "store_dir",
           "synthesise", "user_rules_for", "written_intent"]
