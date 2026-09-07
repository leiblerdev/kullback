"""Builds the customer's world: one shared db.json by inverse replay, a per-Task overlay, the tau2 file
shape, and each tool body with its five gates and bounded repair.

The gates run each generated tool body in a subprocess sandbox before it is trusted; see
kullback.builder.sandbox for what that sandbox does and does not close off. Nothing here calls a
model on its own: write_tool_body takes a Model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.builder import synth
from kullback.builder.body_skill import BODY_SKILL
from kullback.builder.mine import is_assistant_call, is_scalar_result
from kullback.builder.sandbox import (
    DB_CLASS,
    HELPERS,
    Sandbox,
    SandboxError,
    args_text,
    gate_parses,
    id_field,
    id_pattern_for,
    key_fields,
    key_separator,
    match_table,
    parse_result,
    partial_key,
    run_gates,
)

# Re-exported for tests and callers that reach the gates through this module rather than through
# kullback.builder.sandbox directly; compile_env.py's own code calls only run_gates and gate_parses.
from kullback.builder.sandbox import gate_deterministic as gate_deterministic
from kullback.builder.sandbox import gate_executes_on_s0 as gate_executes_on_s0
from kullback.builder.sandbox import gate_non_trivial as gate_non_trivial
from kullback.builder.sandbox import gate_replay_fidelity as gate_replay_fidelity
from kullback.gates.confinement import ALLOWED_IMPORTS, DENIED_BUILTINS, TOOLS_CLASS, source_confinement
from kullback.gates.tool_runs import body_replay_fidelity_gate
from kullback.runner.budget import (
    CHARS_PER_TOKEN,
    CONTEXT_CAP_FRACTION,
    context_cap_tokens,
)
from kullback.runner.budget import (
    DEFAULT_CONTEXT_WINDOW as CONTEXT_WINDOW,
)
from kullback.runner.canon import (
    canonicalize as canon,  # D39: one canonicalizer, no local variant
)
from kullback.runner.records import (
    Atom,
    Column,
    EntitySchema,
    Environment,
    GateResult,
    OverlayRow,
    Task,
    TaskOverlay,
    ToolCall,
    ToolSig,
    Trace,
    Verifier,
    as_dict,
    content_hash,
)

DB_FILE = "db.json"
OVERLAY_DIR = "overlays"
NODE_DIR = "tool_nodes"
OUTCOME_DETAIL_CHARS = 300  # how much of one differing call's sentence a per-call row carries (D171)
MAX_REPAIR_ATTEMPTS = 3
# D117: at most this many model calls inside one attempt's tool-use loop, so a model that keeps
# reaching for lookup_rows or test_body instead of ever submitting a body cannot spend an attempt
# for free; the last reply's content is taken as the body once the rounds run out.
MAX_TOOL_ROUNDS = 6
# CONTEXT_WINDOW and CHARS_PER_TOKEN are budget.py's own (imported above): one context-sizing
# constant per Harness, not a second copy that could drift from what budget.py actually bills.
MAX_EVIDENCE_CHARS = context_cap_tokens(CONTEXT_WINDOW, CONTEXT_CAP_FRACTION) * CHARS_PER_TOKEN
EVIDENCE_LABELS = ("initial", "failing_call", "all_failing_calls", "full_call_table")
HELD_OUT_LABEL = "held_out_failed, shown calls only"  # no shown call failed, so none can be pointed at


class OverlayConflict(ValueError):
    """Two Tasks pin the same row in different versions: a Gate failure for the tau2 export (D74)."""


@dataclass
class StartingState:
    """The shared world plus one overlay per Task, as written under the workdir."""
    db: dict
    overlays: list[TaskOverlay] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    path: Optional[Path] = None
    synthetic_rows: list[str] = field(default_factory=list)


@dataclass
class EnvBundle:
    """Everything the tau2 export needs; the Environment record holds only identity and versions."""
    environment: Environment
    schema: EntitySchema
    tools: list[ToolSig] = field(default_factory=list)
    bodies: dict = field(default_factory=dict)
    db: dict = field(default_factory=dict)
    overlays: list[TaskOverlay] = field(default_factory=list)
    overlay_values: dict = field(default_factory=dict)
    policy_text: str = ""
    tasks: list[Task] = field(default_factory=list)
    verifiers: list[Verifier] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    domain: str = "domain"
    conflicts: list[str] = field(default_factory=list)  # overlay rows the tau2 export could not keep both of


@dataclass
class ToolBuild:
    """One compiled tool: the accepted body, every attempt as a node, and whether it went assisted."""
    name: str
    body: str
    nodes: list[dict] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    assisted: bool = False
    kept_attempt: Optional[int] = None  # which attempt's body this is, when no attempt passed
    # D171: one row per recorded call of this tool, so the miss can be attributed to the Task whose
    # Trace made the call. The corpus gate keeps one fidelity number; this keeps which call it was.
    call_outcomes: list[dict] = field(default_factory=list)

# --- reading rows out of recorded tool results ---

def extract_rows(schema: EntitySchema, result: Any, args: Optional[dict] = None) -> list[tuple[str, str, dict]]:
    """Rows a result states directly: itself, or the elements of a returned list.

    A value nested inside a row (an item inside an order) is not read as a row of its own. The
    call's arguments are passed through to the key, because a table with a composite key can be
    told which row it is answering by the call rather than by the row (`tool_runs.row_key`).
    """
    values = result if isinstance(result, list) else [result]
    return [(t, i, v) for v in values for t, i in [match_table(schema, v, args) or (None, None)] if t]

# --- inverse replay over the whole corpus (D33, D74) ---

@dataclass
class _Obs:
    """One sighting of one row in one trace, and whether a write had already touched it."""
    table: str
    row_id: str
    row: dict
    trace_id: str
    order: tuple
    after_write: bool


def _observations(traces: list[Trace], schema: EntitySchema, write_tools: set[str]) -> list[_Obs]:
    """Every row sighting in corpus order; a write marks the rows it returned or named in its args."""
    out: list[_Obs] = []
    for trace_index, trace in enumerate(traces):
        written: set[str] = set()
        for call_index, call in enumerate(trace.tool_calls):
            # A call the simulated user made through its own tools (telecom's phone tools) is not a
            # sighting of the customer's system; only the assistant's calls describe it (R33).
            if call.error is not None or not is_assistant_call(call):
                continue
            is_write = call.name in write_tools
            rows = extract_rows(schema, parse_result(call.result), call.args)
            for table, row_id, row in rows:
                out.append(_Obs(table, row_id, row, trace.trace_id, (trace_index, call_index),
                                is_write or row_id in written))
            if is_write:
                written |= {row_id for _, row_id, _ in rows}
                written |= {v for v in call.args.values() if isinstance(v, str)}
    return out


def trace_worlds(traces: Iterable[Trace], schema: EntitySchema, write_tools: set[str]) -> dict[str, dict]:
    """Per trace, the version of every row it saw before any write touched it: the world it started in.

    Two traces that saw one row in two such versions started in different worlds. cluster.py keeps
    them in different Tasks, because a Task's overlay can pin one version only (D74), and a trace
    replayed on the other version differs on every read of that row; telecom's one customer seen
    across 456 traces in as many states is where this was found.
    """
    worlds: dict[str, dict] = {}
    for obs in _observations(list(traces), schema, write_tools):
        if not obs.after_write:
            worlds.setdefault(obs.trace_id, {}).setdefault((obs.table, obs.row_id), content_hash(obs.row))
    return worlds


def build_starting_state(
    traces: Iterable[Trace],
    schema: EntitySchema,
    workdir: Path | str,
    tasks: Optional[Iterable[Task]] = None,
    tool_sigs: Optional[Iterable[ToolSig]] = None,
    synthetic: bool = True,
    grow: Optional[dict[str, int]] = None,
    grow_seed: int = 0,
) -> StartingState:
    """One shared db.json for the customer, plus one TaskOverlay per Task (D33, D74).

    Inverse replay: a row's shared value is the latest sighting that no write had touched yet, so an
    observed write is undone. Where a trace shows only the post-state, that state is kept and the
    assumption is recorded. Order is the order the traces are passed in, then call order; nothing is
    keyed by wall-clock time (design section 8). Ids the traces asked for but never showed are then
    filled with tagged synthetic rows (D40), unless `synthetic` is off. `grow` names a row count per
    table to reach with rows composed from the observed ones (D107, `synth.grow`); what was added,
    the rules it followed and the checks it passed are written to synthetic.json.
    """
    traces, workdir = list(traces), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_tools = {s.name for s in (tool_sigs or []) if s.kind == "write"}
    observations = _observations(traces, schema, write_tools)
    by_row: dict[tuple[str, str], list[_Obs]] = {}
    for obs in observations:
        by_row.setdefault((obs.table, obs.row_id), []).append(obs)

    db: dict[str, dict] = {table: {} for table in sorted(schema.tables)}
    assumptions: list[str] = []
    for (table, row_id), seen in sorted(by_row.items()):
        clean = [o for o in seen if not o.after_write]
        chosen = max(clean or seen, key=lambda o: o.order)
        if not clean:
            assumptions.append(f"{table} row {row_id} was only ever seen after a write; "
                               "its post-state is kept as the starting value")
        db.setdefault(table, {})[row_id] = chosen.row

    assumptions += [f"{table} row {row_id} was seen without every part of its key; it was folded "
                    f"into the {count} rows whose known key columns match and is not a row of its own"
                    for table, row_id, count in fold_partial_rows(db, schema)]
    added = add_synthetic_rows(db, schema, traces) if synthetic else []
    assumptions += [f"{table_of} row {row_id} was never shown by a trace; it is a synthetic row "
                    "shaped from the observed rows and a Run that reads it is assisted"
                    for table_of, row_id in added]
    overlays = _build_overlays(observations, tasks or [], workdir, assumptions)
    assumptions += [f"{table_of} row {row_id} is stored under {home}; the standalone copy was folded "
                    "into it and a Task overlay that pins it re-adds the standalone copy"
                    for table_of, row_id, home in fold_into_homes(db, schema)]
    grown_ids: list[str] = []
    if grow:
        grown = synth.grow(db, schema, dict(grow), seed=grow_seed)
        grown_ids = grown.ids
        schema.synthetic_rows = sorted(set(schema.synthetic_rows) | set(grown_ids))
        assumptions += [f"{table_of} holds {len(ids)} synthetic rows composed from the observed ones "
                        "(D107); a Run that reads one is assisted"
                        for table_of, ids in sorted(grown.added.items())]
        if not grown.checks.get("ok", False):
            assumptions.append("the synthetic rows failed a check; see synthetic.json")
        (workdir / "synthetic.json").write_text(
            json.dumps(synth.report(grown), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    path = workdir / DB_FILE
    path.write_text(json.dumps(db, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (workdir / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n", encoding="utf-8")
    return StartingState(db=db, overlays=overlays, assumptions=assumptions, path=path,
                         synthetic_rows=[row_id for _, row_id in added] + grown_ids)


def fold_partial_rows(db: dict, schema: EntitySchema) -> list[tuple[str, str, int]]:
    """A row seen without every part of its key is not a new row and not a contradiction.

    The rule. Such a sighting is folded into every row whose known key columns match it, filling
    only the columns those rows were never shown holding, so it can add what the keyed sightings
    did not show and can never overwrite what they did. Where no keyed row matches, it is kept as
    the partial sighting it is, under its own partial key, because dropping it would lose an
    observation and inventing the missing part would be a guess (D41). Returns (table, key, rows it
    was folded into) per folded row.

    Nothing to do on a table keyed by one column, which is every table until the miner names a
    composite key: a key of one part is never partial.
    """
    folded: list[tuple[str, str, int]] = []
    separator = key_separator(schema)
    for table in sorted(db):
        fields = key_fields(schema, table)
        if len(fields) < 2:
            continue
        rows = db[table]
        for key in sorted(rows):
            if not partial_key(schema, table, key):
                continue
            known = key.split(separator)
            matches = [other for other in sorted(rows) if other != key
                       and _key_agrees(other.split(separator), known)]
            for other in matches:
                for name, value in rows[key].items():
                    rows[other].setdefault(name, value)
            if matches:
                del rows[key]
                folded.append((table, key, len(matches)))
    return folded


def _key_agrees(parts: list[str], known: list[str]) -> bool:
    """Whether a full key matches every part a partial key does know."""
    return len(parts) == len(known) and all(a == b for a, b in zip(parts, known, strict=False) if b)


def fold_into_homes(db: dict, schema: EntitySchema) -> list[tuple[str, str, str]]:
    """Move a row the corpus stores inside another row (schema.homes) out of its top-level table.

    One row, one place. A standalone sighting (get_item_details) and a nested one (the same item
    under products.variants) are the same row; keeping both would let a write land in one and a
    read come from the other. The nested copy wins the position, the standalone copy contributes
    any field the nested one lacks, and a row whose parent the traces never showed stays where it
    is, because there is nowhere to put it. Returns (table, id, home) per folded row.
    """
    folded: list[tuple[str, str, str]] = []
    for table, home in sorted((schema.homes or {}).items()):
        parent, column = home.split(".", 1)
        rows = db.get(table) or {}
        parents = [r for r in (db.get(parent) or {}).values() if isinstance(r, dict)]
        for row_id in sorted(rows):
            for parent_row in parents:
                nest = parent_row.get(column)
                if isinstance(nest, dict) and isinstance(nest.get(row_id), dict):
                    for name, value in rows[row_id].items():
                        nest[row_id].setdefault(name, value)
                    del rows[row_id]
                    folded.append((table, row_id, home))
                    break
    return folded


def referenced_ids(traces: Iterable[Trace], schema: EntitySchema) -> list[tuple[str, str]]:
    """(table, id) pairs a call that succeeded named in its arguments, by the table's own id column.

    Public: build.py's build_environment gate wires these ids into validate.environment_gate, to
    check db.json actually holds every id a trace referenced.
    """
    keys = {table: key_fields(schema, table) for table in schema.tables}
    out: set[tuple[str, str]] = set()
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None:  # an id the customer's tool refused is not a row we owe
                continue
            args = call.args or {}
            for table, fields in keys.items():
                if not fields or not isinstance(args.get(fields[0]), str):
                    continue
                pattern = id_pattern_for(schema, table, fields[0])
                if pattern and not re.match(pattern, args[fields[0]]):
                    continue
                # A composite key the call does not complete names no row: a partial id would be a
                # row of its own, which is exactly what the composite key exists to prevent.
                if any(args.get(name) is None for name in fields[1:]):
                    continue
                out.add((table, key_separator(schema).join(
                    [args[fields[0]]] + [str(args[name]) for name in fields[1:]])))
    return sorted(out)


def add_synthetic_rows(db: dict, schema: EntitySchema, traces: Iterable[Trace]) -> list[tuple[str, str]]:
    """Fill ids the traces referenced but never showed, shaped from the rows they did show (D40, D41).

    Shape and values are the observed rows' own: per column the value seen most often, with the id
    column set to the referenced id. Nothing is invented beyond that recombination, so a table with
    no observed row is left empty rather than made up. The ids are tagged in
    `EntitySchema.synthetic_rows`, which is what marks a Run that reads one as assisted (D49).
    """
    added: list[tuple[str, str]] = []
    for table, row_id in referenced_ids(traces, schema):
        rows = db.get(table) or {}
        if row_id in rows or not rows:
            continue
        fields = key_fields(schema, table)
        parts = row_id.split(key_separator(schema)) if len(fields) > 1 else [row_id]
        named = dict(zip(fields, parts, strict=False)) if fields else {}
        db.setdefault(table, {})[row_id] = dict(_modal_row(rows.values()), **named)
        added.append((table, row_id))
    schema.synthetic_rows = sorted(set(schema.synthetic_rows) | {row_id for _, row_id in added})
    return added


def _modal_row(rows: Iterable[dict]) -> dict:
    """The value seen most often per column across the observed rows, ties broken canonically."""
    columns: dict[str, dict[str, int]] = {}
    firsts: dict[str, dict[str, Any]] = {}
    for row in rows:
        for name, value in row.items():
            key = canon(value)
            counts = columns.setdefault(name, {})
            counts[key] = counts.get(key, 0) + 1
            firsts.setdefault(name, {}).setdefault(key, value)
    return {name: firsts[name][sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]]
            for name, counts in columns.items()}


def _build_overlays(observations: list[_Obs], tasks: Iterable[Task], workdir: Path,
                    assumptions: list[str]) -> list[TaskOverlay]:
    """A Task's rows in the version its own Runs saw: the first sighting inside each of its traces.

    One Task can hold Runs that saw one row in two versions. The overlay can pin only one of them,
    so the disagreement is recorded as an assumption rather than passing silently: the Runs on the
    other version cannot replay on this overlay, and the report and the setup review need to see it.
    """
    overlays = []
    for task in tasks:
        members, rows = set(task.run_ids), {}
        per_trace: dict[tuple[str, str], dict[str, str]] = {}
        for obs in observations:
            key = (obs.table, obs.row_id)
            if obs.trace_id not in members:
                continue
            if key not in rows or obs.order < rows[key].order:
                rows[key] = obs
            per_trace.setdefault(key, {}).setdefault(obs.trace_id, content_hash(obs.row))
        assumptions += [f"task {task.id} runs disagree on {table} row {row_id}: the overlay pins the "
                        "earliest sighting, so the runs that saw the other version cannot replay on it"
                        for (table, row_id), hashes in sorted(per_trace.items()) if len(set(hashes.values())) > 1]
        overlay = TaskOverlay(task_id=task.id, rows=[
            OverlayRow(table=t, id=i, version_hash=content_hash(rows[(t, i)].row),
                       trace_id=rows[(t, i)].trace_id, after_write=rows[(t, i)].after_write)
            for t, i in sorted(rows)
        ])
        assumptions += [f"task {task.id} pins {t} row {i} from a post-write sighting"
                        for (t, i), obs in sorted(rows.items()) if obs.after_write]
        _write_overlay(workdir, overlay, {content_hash(o.row): o.row for o in rows.values()})
        overlays.append(overlay)
    return overlays


def _write_overlay(workdir: Path, overlay: TaskOverlay, values: dict) -> Path:
    """One file per Task: the overlay record and the row values behind its version hashes."""
    directory = Path(workdir) / OVERLAY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{overlay.task_id}.json"
    payload = {"overlay": as_dict(overlay), "values": values}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def load_overlay(workdir: Path | str, task_id: str) -> tuple[TaskOverlay, dict]:
    """The Task's overlay and its row values; route.py reads this before the shared db.json (D74)."""
    payload = json.loads((Path(workdir) / OVERLAY_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
    return TaskOverlay.model_validate(payload["overlay"]), payload["values"]


def overlay_values(workdir: Path | str) -> dict:
    """Every overlay row value written under the workdir, keyed by version hash."""
    directory = Path(workdir) / OVERLAY_DIR
    values: dict = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        values.update(json.loads(path.read_text(encoding="utf-8"))["values"])
    return values


def merge_overlays(db: dict, overlays: Iterable[TaskOverlay], values: dict,
                   conflicts: Optional[list[str]] = None) -> dict:
    """Merge every Task overlay into the one db tau2's harness loads (D74).

    Two Tasks pinning one row in different versions is a disagreement about the world, which the
    per-Task overlays in the Runner never have to settle: each Task reads its own. The single db of
    the tau2 export has to pick one, so with `conflicts` given it keeps the version a Task saw before
    any write (over one seen after a write) and, between two of the same standing, the first Task's,
    and appends one line per conflict for the export gate. Without `conflicts` a disagreement raises,
    which is the contract the single-overlay callers rely on.
    """
    merged = copy.deepcopy(db)
    pinned: dict[tuple[str, str], tuple[str, str, bool]] = {}
    for overlay in overlays:
        for row in overlay.rows:
            seen = pinned.get((row.table, row.id))
            if seen and seen[0] != row.version_hash:
                if conflicts is None:
                    raise OverlayConflict(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row "
                                          f"{row.id} in different versions")
                conflicts.append(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row {row.id} in "
                                 f"different versions; the tau2 export keeps "
                                 f"{overlay.task_id if seen[2] and not row.after_write else seen[1]}'s")
                if not (seen[2] and not row.after_write):
                    continue  # the pinned version stands: it was seen before a write, or both were
            pinned[(row.table, row.id)] = (row.version_hash, overlay.task_id, row.after_write)
            if row.version_hash in values:
                merged.setdefault(row.table, {})[row.id] = values[row.version_hash]
    return merged

# --- rendering the tau2 file shape: code writes the signature, docstring and schema (D56) ---

_TYPES = {"str": "str", "int": "int", "float": "float", "bool": "bool", "list": "list", "dict": "dict"}

_DATA_MODEL_HEAD = '''"""Generated by the Harness from the customer's traces. Do not edit by hand."""
# Annotations stay eager: the gates exec this module without registering it in sys.modules, so a
# postponed annotation would have no module globals to resolve against.
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:  # tau2's own DB base when the export is loaded inside tau2-bench
    from tau2.environment.db import DB as _DBBase
except ImportError:
    _DBBase = BaseModel
'''

_DB_LOAD = '''
    @classmethod
    def load(cls, path):
        """Load the emitted db.json."""
        import json

        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))
'''

_TOOLKIT_SHIM = '''try:  # tau2's own toolkit base when the export is loaded inside tau2-bench
    from tau2.environment.toolkit import ToolKitBase as _ToolKitBase, ToolType, is_tool
except ImportError:
    class _ToolKitBase:
        def __init__(self, db):
            self.db = db

    class ToolType:
        READ = "read"
        WRITE = "write"
        GENERIC = "generic"

    def is_tool(tool_type):
        def decorate(func):
            func.__tool_type__ = tool_type
            return func

        return decorate
'''


def _class_name(table: str) -> str:
    """orders -> Order, payment_methods -> PaymentMethod."""
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", singular) if p) or "Row"


def _annotation(types: Iterable[str]) -> str:
    """One observed type gives that type; a union of several gives Any (D72)."""
    kinds = {_TYPES.get(t, "Any") for t in types if t != "NoneType"}
    return kinds.pop() if len(kinds) == 1 else "Any"


def render_data_model(schema: EntitySchema) -> str:
    """One class per table plus the DB class, every column Optional[Any] so no real row is rejected.

    A mined column carries a handful of display samples, not the customer's type (D72): samples that
    are all int say nothing about the row holding 10.5, and samples that are all str say nothing
    about the row holding 94016. A narrow annotation there makes pydantic reject a real row, the
    module never loads, and every body fails gate 2 for a reason no body can fix. So the emitted
    model is as wide as the union it stands for, and the column classes in `EntitySchema` (D73), not
    the annotation, are what a Verdict compares by.
    """
    parts = [_DATA_MODEL_HEAD]
    for table in sorted(schema.tables):
        fields = []
        for name in sorted({c.name for c in schema.columns if c.table == table}):
            fields.append(f"    {name}: Optional[Any] = Field(default=None)")
        body = "\n".join(fields) or "    pass"
        parts.append(f'\n\nclass {_class_name(table)}(BaseModel):\n    """One row of {table}."""\n{body}\n')
    tables = "\n".join(f"    {t}: Dict[str, {_class_name(t)}] = Field(default_factory=dict)"
                       for t in sorted(schema.tables)) or "    pass"
    parts.append(f'\n\nclass {DB_CLASS}(_DBBase):\n    """The customer\'s world as the traces show '
                 f'it."""\n{tables}\n{_DB_LOAD}')
    return "".join(parts)


def _signature(sig: ToolSig) -> str:
    """The call signature, required arguments first, from the mined args_fields."""
    required = [f"{f.name}: {_annotation(f.types)}" for f in sig.args_fields if not f.optional]
    optional = [f"{f.name}: Optional[{_annotation(f.types)}] = None" for f in sig.args_fields if f.optional]
    return f"    def {sig.name}({', '.join(['self'] + required + optional)}) -> Any:"


def _in_docstring(text: str) -> str:
    """The customer's own words, safe to paste inside a triple-quoted string.

    A description holding a triple quote or a trailing backslash would otherwise end the docstring
    early and break gate 1 on every attempt, for a reason no model could fix (design section 7: code
    owns the docstring, so a code-owned break is code's to prevent).
    """
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _docstring(sig: ToolSig) -> str:
    """The customer's description, the argument list and the observed result fields."""
    lines = ['        """' + _in_docstring(
        sig.description or f"The customer's {sig.name} tool, as the traces show it.")]
    if sig.args_fields:
        lines += ["", "        Args:"] + [f"            {f.name}: {_annotation(f.types)}" for f in sig.args_fields]
    if is_scalar_result(sig):
        lines += ["", "        Returns:",
                  "            a bare " + _annotation(sig.result_schema[0].types) + ", not wrapped in an object"]
    elif sig.result_schema:
        lines += ["", "        Returns:", "            " + ", ".join(f.name for f in sig.result_schema)]
    return "\n".join(lines + ['        """'])


def render_tools(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict,
                 class_name: str = TOOLS_CLASS, with_imports: bool = True) -> str:
    """The toolkit class: code owns everything but the body, which the model wrote."""
    head = ""
    if with_imports:
        names = ", ".join([DB_CLASS] + [_class_name(t) for t in sorted(schema.tables)])
        head = ('"""Generated by the Harness from the customer\'s traces. Do not edit by hand."""\n'
                f"from typing import Any, Optional\n\nfrom data_model import {names}\n\n")
    parts = [head, _TOOLKIT_SHIM,
             f'\n\nclass {class_name}(_ToolKitBase):\n    """Every tool mined from the customer\'s '
             f'traces."""\n\n    def __init__(self, db) -> None:\n        super().__init__(db)\n'
             "        self.db = db\n"]
    for sig in sigs:
        body = textwrap.dedent(bodies.get(sig.name, "raise NotImplementedError")).strip("\n") or "pass"
        parts.append(f"\n    @is_tool(ToolType.{sig.kind.upper()})\n{_signature(sig)}\n{_docstring(sig)}\n"
                     + textwrap.indent(body, "        ").rstrip() + "\n")
    return "".join(parts)


def module_source(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict) -> str:
    """One self-contained module, data model plus toolkit, for the gates to execute."""
    return render_data_model(schema) + "\n\n" + render_tools(schema, sigs, bodies, with_imports=False)


def load_toolkit(source: str, db: dict, class_name: str = TOOLS_CLASS, db_class: str = DB_CLASS,
                 overlay: Optional[TaskOverlay] = None, overlay_values: Optional[dict] = None):
    """The generated module loaded in this process, with the Task's Starting state inside it.

    Given a Task's overlay, the world handed to the toolkit is the shared db with that overlay
    merged, so the March Task's code-routed call sees the March row and not the June one (D74). A
    generated body reads `self.db` and cannot do the overlay lookup itself, so a toolkit built on the
    shared db alone leaves the overlay dead for every code route; `route.py`'s StateView stays the
    lookup for the recording and stand-in routes.

    dont_inherit keeps a caller's `from __future__ import annotations` out of the generated module:
    a postponed annotation has no module globals for pydantic to resolve against. This is the loader
    the Runner's router is given; the gates use the subprocess Sandbox instead.

    `evaluate_arithmetic` is put in the namespace, not imported by the body: `ast`, `eval`, `exec`
    and `compile` stay refused, and a recorded tool that evaluates an expression string is served by
    the one code-owned evaluator instead of a parser the model writes again on every build (see
    runner/arith.py). `PROVIDED_HELPERS` is what the confinement gate reads it under, so the two
    cannot drift: a name bound here that the gate does not know is a body refused for a NameError it
    would never have raised.
    """
    if overlay is not None:
        db = merge_overlays(db, [overlay], overlay_values or {})
    refused = source_confinement(source, class_name)
    if refused:
        raise SandboxError("the generated module is not confined and would run in this process: "
                           + "; ".join(refused))
    namespace: dict = {"__name__": "generated_tools", **HELPERS}
    exec(compile(source, "<generated>", "exec", dont_inherit=True), namespace)  # noqa: S102
    return namespace[class_name](namespace[db_class].model_validate(db))

# --- the model writes the body (D56, D75) ---

# tau2 wraps every raised exception as f"Error: {e}" before it reaches the agent
# (vendor/tau2-bench, Environment.get_response), and the first live build copied that wrapper into
# seven bodies, faithfully, because nothing said it was not part of the message. The wrapper is
# the transport talking, not the customer's tool, but which wrapper a customer's transport adds
# is not ours to know in advance (D51): it is read off the corpus. A prefix that every recorded
# error shares, ending at a ": " boundary and leaving a message behind on every one of them, is
# transport; the copy shown to the model has it peeled off. `ToolCallError.payload` keeps the
# wrapper (D67 keeps the payload verbatim, with `raw_ptr` back to D66's untouched byte).


def shared_error_prefix(calls: Iterable[ToolCall]) -> str:
    """The prefix every string error payload in `calls` shares, cut at the last ": " they all have.

    Two payloads at least, or one message alone would be its own prefix. The remainder has to be
    non-empty on every payload, so a corpus whose errors are all one identical message yields the
    part before its last ": " only if that leaves something after it.
    """
    payloads = [call.error.payload for call in calls
                if call.error is not None and isinstance(call.error.payload, str)]
    if len(payloads) < 2:
        return ""
    common = payloads[0]
    for payload in payloads[1:]:
        limit = min(len(common), len(payload))
        cut = next((i for i in range(limit) if common[i] != payload[i]), limit)
        common = common[:cut]
    boundary = common.rfind(": ")
    while boundary >= 0:
        prefix = common[:boundary + 2]
        if all(len(payload) > len(prefix) for payload in payloads):
            return prefix
        boundary = common.rfind(": ", 0, boundary)
    return ""


def _display_error_payload(payload: Any, prefix: str = "") -> Any:
    """The error payload as the model should read it, with the corpus's shared prefix peeled off.

    Only a leading, exact `prefix` is peeled, and only from a str payload; a JSON payload (D67's
    `code` class) and a payload that never carried the prefix pass through untouched.
    """
    if prefix and isinstance(payload, str) and payload.startswith(prefix) and len(payload) > len(prefix):
        return payload[len(prefix):]
    return payload


_SYSTEM = ("You write the body of one Python method of a tool class rebuilt from a customer's traces. "
           "Return only the body: no signature, no fences, no explanation. The body may read and write "
           "self.db, a pydantic model with one dict per table. Each dict's values are pydantic model rows, "
           "not plain dicts: read or write a row's field by attribute, as in order.status or "
           "order.status = \"cancelled\", never with .get(...) or any other dict method. That holds for the "
           "row and stops there: what a column of a row holds is not a model. A column whose value is a "
           "list or a dict is a plain Python list or dict, so index it, read its parts by key, as in "
           "row.entries[0][\"amount\"], and write them by plain assignment, as in "
           "row.entries[0][\"amount\"] = 0; row.entries[0].amount raises AttributeError. Where a column "
           "holds one, the tables below say so and show the form that reads it. The body reads the "
           "world's tables for every id and every value it works with: an id or a value copied out "
           "of a recorded call and written into the body as a literal is refused by the "
           "memorised_values gate, however well it replays the calls you were shown. Raise ValueError "
           "with the customer's own message where the traces show an error. Where every recorded error "
           "in this corpus begins with the same transport prefix, it is shown with that prefix removed, "
           "so write the message exactly as shown and do not put a prefix of your own in front of it.")


def _example_block(calls: Iterable[ToolCall], error_prefix: Optional[str] = None) -> str:
    """The recorded calls as the model sees them: arguments, then result or error class, in full.

    Nothing is cut here. D75's third attempt is the full call table, and a node that says
    `evidence_calls: 30` has to mean the model saw thirty complete rows; the only limit is the D65
    cap, which refuses the call rather than shortening it. `error_prefix` is the corpus-wide shared
    error prefix (`shared_error_prefix`); given None it is read off these calls alone.
    """
    calls = list(calls)
    prefix = shared_error_prefix(calls) if error_prefix is None else error_prefix
    lines = []
    for call in calls:
        outcome = (f"error {call.error.class_}: {_display_error_payload(call.error.payload, prefix)!r}"
                   if call.error is not None
                   else "result " + json.dumps(parse_result(call.result), default=str))
        lines.append(f"- args {json.dumps(call.args, sort_keys=True, default=str)} -> {outcome}")
    return "\n".join(lines)


def _sample_value(column: Column) -> Any:
    """One mined sample of a column as a Python value, or None when there is none to read.

    Mining stores a nested sample as JSON text and cuts it at a length (`mine._short`), so a long
    one does not parse; a sample that does not parse is no sample, and the caller falls back to the
    form that names no field.
    """
    sample = column.samples[0] if column.samples else None
    if isinstance(sample, (list, dict)):
        return sample
    if isinstance(sample, str):
        try:
            return json.loads(sample)
        except ValueError:
            return None
    return None


def _column_access(table: str, column: Column) -> str:
    """How a column that holds a list or a dict is read, in one line, off its own mined sample.

    The data model types a nested column as `Optional[Any]`, so the value inside it is a plain
    Python list or dict however the row around it is typed. A body that took `_SYSTEM` at its word
    and read a row's field by attribute all the way down crashed on every attempt of one live
    build, at the same call, with the same AttributeError, and the retry could not help because the
    instruction that caused it was still true of the row and false of what the column held. The
    sample already beside the column says which it is and what one part is called, so the form that
    reads it is derived here rather than left to the model to infer.

    Everything is read off the schema: no name here comes from a domain.
    """
    types = (column.evidence or {}).get("types") or []
    sample = _sample_value(column)
    name = f"{table}.{column.name}"
    if "list" in types:
        element = next((v for v in sample if isinstance(v, dict)), None) if isinstance(sample, list) else None
        if element:
            key = next(iter(element))
            return (f"{name} is a list of plain dicts, not model rows; one element looks like: "
                    f"{json.dumps(element, default=str)}; read element[\"{key}\"], "
                    f"never element.{key}")
        return (f"{name} is a plain list, not a model row: index it, and read anything it holds by "
                "key, never by attribute")
    if "dict" in types:
        rows = sample if isinstance(sample, dict) else {}
        key, inner = next(((k, v) for k, v in rows.items() if isinstance(v, dict)), ("", None))
        if inner:
            keyed_by = next((f for f, v in inner.items() if v == key), "its own key")
            field = next(iter(inner))
            return (f"{name} is a dict of plain dicts keyed by {keyed_by}; read "
                    f"{column.name}[key][\"{field}\"], never {column.name}[key].{field}")
        return (f"{name} is a plain dict, not a model row: read it by key, as in "
                f"{column.name}[\"key\"], never by attribute")
    return ""


def _schema_block(schema: EntitySchema) -> str:
    """Tables and columns of the customer's world: the same for every tool in this build (D65's
    stable prefix, docs/prompt-caching.md item 1).

    A column mined as a dict gets one sample beside it, and a column that holds a list or a dict
    gets the form that reads it (`_column_access`). Mining cannot tell a customer's real table from
    a stand-in built out of a value nested inside another table's column: a tool that answers with
    one nested row makes mining propose a table for it, while the table that stores those rows
    keeps them in a column of its own. Both are real observations and nothing here decides which is
    the customer's real storage; the sample is what lets the model notice the second on its own,
    instead of trusting the proposed table alone and raising "not found" on a database that only
    fills it through the nesting. That is what happened on the first live build, nine calls out of
    nine.
    """
    lines = ["Tables on self.db:"]
    for table in sorted(schema.tables):
        columns = sorted((c for c in schema.columns if c.table == table), key=lambda c: c.name)
        names = ", ".join(c.name for c in columns) if columns else "(no columns observed)"
        lines.append(f"- {table}: {names}")
        fields = key_fields(schema, table)
        if len(fields) > 1:
            separator = key_separator(schema)
            form = separator.join("{" + name + "}" for name in fields)
            lines.append(f"    self.db.{table} is keyed by {' and '.join(fields)} together, joined "
                         f"with {separator!r}: the key of a row is f\"{form}\". One "
                         f"{fields[0]} stands for several rows here, one per "
                         f"{' and '.join(fields[1:])}, so build the whole key to reach a row and "
                         f"never look one up by {fields[0]} alone. A row's own "
                         f"{' or '.join(fields[1:])} column can be null where the customer's tool "
                         f"took the value from the call's arguments; the key is what says which row "
                         f"this is. Split a key on {separator!r} to read its parts back, and answer "
                         f"rows whose key parts match the arguments you were given.")
        home = (schema.homes or {}).get(table)
        if home:
            parent, column = home.split(".", 1)
            key = id_field(schema, table) or "its id"
            lines.append(f"    {table} rows are stored inside {home}, keyed by {key}: walk "
                         f"self.db.{parent}.values() and read .{column} to find one. self.db.{table} "
                         f"holds only rows the traces showed on their own and may be empty on the "
                         f"customer's real database, so look in {home} first and in self.db.{table} "
                         f"second.")
        for column in columns:
            if "dict" in (column.evidence or {}).get("types", []) and column.samples:
                lines.append(f"    {table}.{column.name} looks like: {column.samples[0]}")
            access = _column_access(table, column)
            if access:
                lines.append(f"    {access}")
    return "\n".join(lines)


def _confinement_block(denied: Iterable[str] = DENIED_BUILTINS, allowed: Iterable[str] = ALLOWED_IMPORTS) -> str:
    """The confinement gate's own rules, in words, generated from the gate's own constants.

    The first live build spent four of sixteen tools discovering these by being refused: the model
    reached for `getattr` and `__dict__` because nothing had told it not to, and each refusal cost
    an attempt to learn one rule. Written out here it costs nothing, because this text sits in the
    stable system prefix that every call of the stage reuses from the provider's cache.

    Generated rather than written so the two can never drift: if sandbox.py starts denying a name,
    the prompt says so on the next build without anyone remembering to edit it. The two lists are
    parameters defaulting to the gate's own constants, so a caller can show a reader (or a test)
    what a different gate would render without reaching into this module.

    The helper sentence sits here rather than in a per-tool message because it is the same for every
    tool of every build, so it is read out of the provider's cache and costs nothing. Build 12 spent
    four attempts on one tool watching the model write an arithmetic parser by hand, each one worse
    than the last, because nothing had said there was one already.
    """
    return ("The body is checked before it runs and is refused if it names anything outside the "
            "customer's world. It may not use: " + ", ".join(sorted(denied)) + ". It may "
            "not touch a dunder attribute (`__dict__`, `__class__`, `__globals__` and the rest). "
            "It may import only: " + ", ".join(sorted(allowed)) + ". Read fields by name "
            "(`order.status`) or by key (`self.db.orders[order_id]`), never through getattr. "
            "evaluate_arithmetic(expression) is provided in the body's namespace: it evaluates "
            "+ - * / // % ** with parentheses over a decimal and returns a Decimal, so never write "
            "a parser and never reach for eval; for example `float(evaluate_arithmetic(expression))` "
            "or round it to the places the recording shows.")


# D117: said once, only when compile_tool was asked to offer the two builder tools, so a caller
# that turns them off (or an old caller that never knew about them) gets exactly the old prompt.
_BUILDER_TOOLS_PARAGRAPH = (
    "Two tools are available while you write this body: lookup_rows(table, key=None) reads a row "
    "of the Starting state (or, with no key, that table's row count and a few sample keys), and "
    "test_body(body) runs a draft through the same gates this attempt will face, on the calls you "
    "were shown. Look a row up before guessing its shape; test the draft before you submit it. "
    "A tool call is never a submission: when a draft passes, send the body itself as your reply. "
    "The rounds are few; if they run out, the last body you tested is taken as your reply.")


def _stable_system(schema: Optional[EntitySchema] = None, tool_names: Iterable[str] = (),
                   builder_tools: bool = False) -> str:
    """`_SYSTEM` plus what every tool in this build shares: one prefix, sent unchanged on every
    call of the stage, long enough on a real customer to clear a provider's cache minimum."""
    # The body skill (D168) sits in the prefix too: it is the same bytes for every tool of a build,
    # and it is read before the tables, which is where a body's mistakes are made.
    parts = [_SYSTEM, BODY_SKILL, _confinement_block()]
    if schema is not None:
        parts.append(_schema_block(schema))
    names = sorted(set(tool_names))
    if names:
        parts.append("Tools in this build: " + ", ".join(names))
    if builder_tools:
        parts.append(_BUILDER_TOOLS_PARAGRAPH)
    return "\n\n".join(parts)


def _tool_block(toolsig: ToolSig, examples: Iterable[ToolCall], error_prefix: Optional[str] = None) -> str:
    """What differs call to call: this one tool, its signature and its recorded calls."""
    if is_scalar_result(toolsig):
        result_line = ("Result: a bare " + _annotation(toolsig.result_schema[0].types)
                       + ", returned directly, not as {\"value\": ...} or any other wrapper object")
    else:
        result_line = "Result fields: " + ", ".join(f.name for f in toolsig.result_schema)
    parts = [f"Tool: {toolsig.name}",
             f"Description: {toolsig.description or 'not declared by the customer'}",
             "Arguments: " + ", ".join(f"{f.name} ({_annotation(f.types)})" for f in toolsig.args_fields),
             result_line,
             "Recorded calls:", _example_block(examples, error_prefix)]
    return "\n".join(parts)


def body_messages(toolsig: ToolSig, examples: Iterable[ToolCall], schema: Optional[EntitySchema] = None,
                  failure: str = "", tool_names: Iterable[str] = (),
                  error_prefix: Optional[str] = None, builder_tools: bool = False,
                  lesson: str = "") -> list[dict]:
    """The whole message list one body request sends, so its size can be checked before it goes.

    The system message carries the fixed instructions plus what is the same for every tool in
    this build (the schema, the tool list): one prefix, unchanged call to call, for a provider's
    cache to reuse. The user message carries only this one tool, its recorded calls, and (for a
    one-shot request outside the repair loop) the failure of a previous attempt. `builder_tools`
    (D117) adds the one paragraph naming lookup_rows and test_body to the stable prefix; it is the
    same value on every call of one compile_tool, so the cached bytes never move mid-build.

    `lesson` is what earlier attempts at this one tool already failed on (memory.lesson_for): it
    belongs to this tool and not to the build, so it goes in the user turn and leaves the stable
    prefix alone. Empty, the messages are the same bytes they were before there was a lesson.
    """
    user = _tool_block(toolsig, examples, error_prefix)
    if lesson:
        user += "\n\n" + lesson
    if failure:
        user += "\n\nThe previous body failed these gates:\n" + failure
    return [{"role": "system", "content": _stable_system(schema, tool_names, builder_tools)},
            {"role": "user", "content": user}]


def _append_retry(messages: list[dict], reply_content: str, evidence: Iterable[ToolCall],
                  failure: str, error_prefix: Optional[str] = None) -> list[dict]:
    """A gate-failure retry (D75, docs/prompt-caching.md item 2): the messages so far are kept
    exactly as they were sent, so the system and first user turn stay the cached prefix; the
    model's previous reply arrives as an assistant turn and the new evidence and failure as a
    new user turn, never folded back into the first one."""
    turn = ("Recorded calls:\n" + _example_block(evidence, error_prefix)
            + "\n\nThe previous body failed these gates:\n" + failure)
    return messages + [{"role": "assistant", "content": reply_content},
                       {"role": "user", "content": turn}]


def _context_cap_error() -> tuple:
    """budget.py's D65 refusal, imported late so this module does not need budget.py to exist."""
    try:
        from kullback.runner.budget import ContextCapExceeded
        return (ContextCapExceeded,)
    except Exception:  # pragma: no cover - budget.py is always importable in this package
        return ()


def prompt_chars(messages: Iterable[dict]) -> int:
    """The size the D65 cap is checked against: the whole prompt, not the evidence block alone."""
    return sum(len(message.get("content") or "") for message in messages)


def write_tool_body(model, toolsig: ToolSig, examples: Iterable[ToolCall],
                    schema: Optional[EntitySchema] = None, failure: str = "") -> str:
    """Ask the Model for one tool body. Code owns the signature, the docstring and the schema."""
    return _body_from(model, body_messages(toolsig, examples, schema, failure))


def _body_from(model, messages: list[dict]) -> str:
    reply = model.query(messages)
    return _strip_fence(reply.content or "")


def _strip_fence(text: str) -> str:
    """Take the code out of a fenced reply and dedent it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        text = "\n".join(lines[:-1] if lines and lines[-1].strip().startswith("```") else lines)
    return textwrap.dedent(text).strip("\n")

# --- the two tools the body-writing model may call (D117) ---
#
# A body-writing model has to guess a row's shape from the schema block and the recorded calls
# alone, and has to guess whether its own draft clears the gates. lookup_rows answers the first
# guess and test_body answers the second, both without spending a repair attempt to find out.
# Neither is allowed to touch the held-out split: lookup_rows only reads db and the shown calls'
# own worlds, and test_body only ever gates the shown calls, with an empty held-out list, so
# nothing the held-out replay would have caught can leak back through either tool.

LOOKUP_ROWS_TOOL = {
    "name": "lookup_rows",
    "description": ("Read one row of the Starting state by table and key, or (with no key) a "
                    "table's row count and a few sample keys. Read-only: nothing here changes "
                    "the world. Never returns a held-out row."),
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "description": "the table to look in"},
            "key": {"type": "string", "description": "the row's id; omit to see the table's shape"},
        },
        "required": ["table"],
    },
}

TEST_BODY_TOOL = {
    "name": "test_body",
    "description": ("Run a draft body through the same gates this attempt will face, on the "
                    "calls you were shown (never the held-out ones). Returns which gate failed "
                    "and why, or that every gate passed."),
    "parameters": {
        "type": "object",
        "properties": {"body": {"type": "string", "description": "the full body to test, as Python source"}},
        "required": ["body"],
    },
}

BUILDER_TOOLS = [LOOKUP_ROWS_TOOL, TEST_BODY_TOOL]


def _find_row(schema: EntitySchema, world: dict, table: str, key: str) -> tuple[Optional[dict], Optional[str]]:
    """A row and where it sits: the table itself, or (schema.homes) the parent it is nested in."""
    row = (world.get(table) or {}).get(key)
    if isinstance(row, dict):
        return row, table
    home = (schema.homes or {}).get(table)
    if home:
        parent, column = home.split(".", 1)
        for parent_row in (world.get(parent) or {}).values():
            nest = parent_row.get(column) if isinstance(parent_row, dict) else None
            if isinstance(nest, dict) and isinstance(nest.get(key), dict):
                return nest[key], home
    return None, None


def _shown_worlds(db: dict, shown: list[ToolCall], call_states: Optional[dict]) -> list[dict]:
    """db, then every distinct world a shown call ran on (D74's per-Task overlay), never a held-out
    one: the same call_states lookup Sandbox.state_for makes, read here off the dict directly
    since no Sandbox exists yet when the model is still drafting the body."""
    seen = {id(db)}
    worlds = [db]
    for call in shown if call_states else []:
        state = call_states.get(call.id, db) if call.id else db
        if id(state) not in seen:
            seen.add(id(state))
            worlds.append(state)
    return worlds


def _lookup_rows_text(schema: EntitySchema, db: dict, shown: list[ToolCall], call_states: Optional[dict],
                      table: Optional[str] = None, key: Optional[str] = None) -> str:
    """What lookup_rows answers: a row, a table's shape, or the table list for an unknown name."""
    tables = sorted(schema.tables)
    if table not in tables:
        return f"unknown table {table!r}; tables on self.db: {', '.join(tables)}"
    if not key:
        rows = db.get(table) or {}
        return f"{table}: {len(rows)} rows; sample keys: {sorted(rows)[:3]}"
    row, location = None, None
    for world in _shown_worlds(db, shown, call_states):
        row, location = _find_row(schema, world, table, key)
        if row is not None:
            break
    if row is None:
        return f"{table} row {key!r} was not found in the Starting state or a shown call's world"
    text = json.dumps(row, sort_keys=True, default=str)
    note = ""
    if len(text) > 2000:
        text, note = text[:2000], " (truncated to 2000 characters)"
    where = f"table {table}" if location == table else f"nested inside {location} (schema.homes)"
    return f"{table} row {key} is stored in {where}{note}: {text}"


def _build_tools_impl(schema: EntitySchema, toolsig: ToolSig, shown: list[ToolCall], db: dict,
                      call_states: Optional[dict], workdir: Path, attempt: int, timeout: float,
                      rules: Any) -> dict[str, Callable[..., str]]:
    """lookup_rows and test_body, closed over one attempt's own evidence and probe directory.

    test_body gates on `shown` alone, with an empty held-out list: the split the repair loop keeps
    hidden from the model stays hidden from the model's own probing too, not just from the failure
    text a rejected attempt is shown.
    """
    probes = {"n": 0}

    def lookup_rows(table: Optional[str] = None, key: Optional[str] = None) -> str:
        return _lookup_rows_text(schema, db, shown, call_states, table, key)

    def test_body(body: Optional[str] = None) -> str:
        probes["n"] += 1
        source = module_source(schema, [toolsig], {toolsig.name: body or ""})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}_probe_{probes['n']}", timeout=timeout,
                          call_states=call_states)
        gates = run_gates(source, sandbox, shown, [], schema, rules,
                          probe_refusals=toolsig.kind == "write", sig=toolsig)
        if all(g.passed for g in gates):
            return "passed every gate: " + ", ".join(g.stage for g in gates)
        return _failure_text(gates)

    return {"lookup_rows": lookup_rows, "test_body": test_body}


def _tool_use_record(call: Any, result_text: str) -> dict:
    """What a tool use is remembered as on the node: never the row or the body, only what was asked."""
    if call.name == "test_body":
        body = (call.arguments or {}).get("body") or ""
        arguments: dict = {"body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    else:
        arguments = {"table": (call.arguments or {}).get("table"), "key": (call.arguments or {}).get("key")}
    return {"name": call.name, "arguments": arguments, "result_chars": len(result_text)}


def _run_builder_tool(call: Any, tools_impl: dict[str, Callable[..., str]]) -> str:
    impl = tools_impl.get(call.name)
    if impl is None:
        return f"unknown tool {call.name!r}; available tools: {', '.join(sorted(tools_impl))}"
    try:
        return impl(**(call.arguments or {}))
    except TypeError as exc:
        return f"bad arguments for {call.name}: {exc}"


def _assistant_tool_turn(reply: Any) -> dict:
    """The canonical assistant turn a tool-calling reply becomes, the same shape the Runner's own
    loop and provider.py's adapters read (runner.loop._assistant_message)."""
    return {"role": "assistant", "content": reply.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls]}


def _reply_with_tools(model, messages: list[dict], tools_impl: dict[str, Callable[..., str]],
                      max_rounds: int = MAX_TOOL_ROUNDS) -> tuple[str, list[dict], str]:
    """Query the model with lookup_rows and test_body on, executing whatever it calls (D117).

    `messages` (the attempt's own system and user turns, exactly as compile_tool built them) is
    never mutated: the tool exchange runs over a local copy, so the next attempt's retry still
    appends its assistant reply and new evidence onto the plain chain `_append_retry` expects,
    never onto a transcript of tool calls. At most `max_rounds` model calls; a reply that asks for
    no tool is the body. The third value is the last body the model handed to test_body, so a
    model that tests a draft and keeps probing until the rounds run out (the sixth retail build's
    cancel_pending_order: three drafts tested, then a row lookup, and no reply) still hands the
    caller the draft it was working on rather than nothing.
    """
    working = list(messages)
    tool_uses: list[dict] = []
    reply = None
    draft = ""
    for _ in range(max_rounds):
        reply = model.query(working, tools=BUILDER_TOOLS)
        if not reply.tool_calls:
            return reply.content or "", tool_uses, draft
        working = working + [_assistant_tool_turn(reply)]
        for call in reply.tool_calls:
            result_text = _run_builder_tool(call, tools_impl)
            tool_uses.append(_tool_use_record(call, result_text))
            if call.name == "test_body" and (call.arguments or {}).get("body"):
                draft = str(call.arguments["body"])
            working = working + [{"role": "tool", "tool_call_id": call.id, "content": result_text}]
    return (reply.content or "") if reply is not None else "", tool_uses, draft

# --- the bounded repair loop (D75) ---

def split_calls(calls: Iterable[ToolCall], every: int = 3) -> tuple[list[ToolCall], list[ToolCall]]:
    """The held-out split the LLM is never shown (D51, D75): every third call, deterministically."""
    calls = list(calls)
    if len(calls) < 2:
        return calls, []
    held_out = [c for i, c in enumerate(calls) if i % every == every - 1]
    shown = [c for i, c in enumerate(calls) if i % every != every - 1]
    return (shown, held_out) if held_out else (shown[:-1], shown[-1:])


def _failing_calls(gates: list[GateResult], shown: list[ToolCall]) -> list[ToolCall]:
    """The shown calls a gate named in its failures; empty when no shown call was named."""
    named = [f for gate in gates if not gate.passed for f in gate.failures]
    return [c for c in shown if any(args_text(c) in f for f in named)]


def _held_out_only(gates: list[GateResult]) -> bool:
    """True when the held-out replay is the only gate that failed, so no shown call can be shown."""
    failed = [g for g in gates if not g.passed]
    return bool(failed) and all(g.metrics.get("split") == "held_out" for g in failed)


def _constant_evidence_note(evidence: Iterable[ToolCall]) -> str:
    """Said once, when the calls just shown already share one answer among themselves.

    Gate 4 fails a body that answers every call the same way, and the failure line reads like an
    instruction to make the answers differ. For a tool whose own recorded calls differ in nothing
    but their arguments, that is not a defect the next attempt can write its way out of; it is
    what the evidence says the tool does. Left unsaid, the model takes gate 4 at its word and
    invents a categorization the traces never showed. transfer_to_human_agents did exactly that on
    the first live build: every recorded call answered "Transfer successful" whatever the summary,
    three repair attempts each added logic to make the answers differ, and the last used a name it
    never imported and crashed on every call.
    """
    results = [canon(parse_result(call.result)) for call in evidence if call.error is None]
    if len(results) < 2 or len(set(results)) != 1:
        return ""
    return ("\nThe recorded calls above already answer every one of them the same way. If your body "
            "does too, that may be the correct behaviour; do not invent a detail the calls never "
            "showed just to make the answers differ.")


def _failure_text(gates: list[GateResult], held_out: Iterable[ToolCall] = ()) -> str:
    """What the next attempt is told, with the held-out split kept out of it (D51, D75).

    The held-out calls are the ones the model is never shown, so neither their arguments nor the
    outcome expected of them may travel back in a repair prompt. A failure on them is reported as a
    count, which tells the model its body is wrong somewhere without handing it the answers. Gates 2
    to 4 run over every call, so their failure lines are filtered by the same rule.
    """
    hidden = [args_text(call) for call in held_out]
    lines = []
    for gate in gates:
        if gate.passed:
            continue
        split = gate.metrics.get("split", "")
        kept = [] if split == "held_out" else [f for f in gate.failures if not any(h in f for h in hidden)]
        withheld = len(gate.failures) - len(kept)
        text = "; ".join(kept)
        if withheld:
            text += ("; " if text else "") + f"{withheld} more on calls you were not shown"
        lines.append(f"- gate {gate.stage} ({split}): {text}")
    lines += _import_hints(gates)
    return "\n".join(lines)


_RAISED = re.compile(r"\braised ([A-Za-z_][A-Za-z0-9_.]*): (.+)")


def exception_in(failure: str) -> str:
    """The exception a gate's failure line quotes, as `ExceptionClass: message`; "" when it quotes none.

    The executes gate writes `<tool>(<args>) raised <class>: <message>` (gates/tool_runs.py), so the
    class and the message are what is left when the call in front of them is dropped. Two attempts
    that crashed the same way then compare equal whatever calls their failures named.
    """
    match = _RAISED.search(failure)
    return f"{match.group(1)}: {match.group(2).strip()}" if match else ""


def exception_line(gates: Iterable[GateResult]) -> str:
    """The first exception the failing gates report; "" when none of them reports one."""
    for gate in gates:
        if gate.passed:
            continue
        for failure in gate.failures:
            found = exception_in(failure)
            if found:
                return found
    return ""


# Said only when the attempt that just ran raised what the attempt before it raised. One live build
# spent all four attempts at one write tool on the same AttributeError at the same call: the retry
# showed the crash each time and the model rewrote around it each time, because nothing said the
# error had not moved. This goes in the new user turn, never in the cached prefix (D65).
_SAME_ERROR = ("\nThe same error as the attempt before, at the same call. The body must change "
               "where it raised: {error}")

_UNDEFINED_NAME = re.compile(r"NameError: name '(\w+)' is not defined")


def _import_hints(gates: list[GateResult]) -> list[str]:
    """A NameError on a module the sandbox allows names the fix; say it instead of the traceback.

    transfer_to_human_agents called re.findall on the first live build and never imported re, and
    every one of its 25 replays died on the same NameError. The gate already knew which name was
    missing and that the module is on the allowed list; the retry was handing back the raw error.
    """
    names = sorted({m.group(1) for gate in gates if not gate.passed
                    for failure in gate.failures for m in _UNDEFINED_NAME.finditer(failure)})
    return [f"- `{name}` is on the allowed import list but the body never imported it; put "
            f"`import {name}` at the top of the body" for name in names if name in ALLOWED_IMPORTS]


def _evidence_for(attempt: int, shown: list[ToolCall], gates: list[GateResult]) -> list[ToolCall]:
    """Evidence grows per attempt: the failing call, then all failing calls, then the full table.

    When no shown call is named in the failures (the held-out split failed on its own, or the module
    did not parse at all), there is no failing call to point at, so the attempt gets every shown call
    rather than the first one, which passed.
    """
    if attempt == 0:
        return shown[:3]
    failing = _failing_calls(gates, shown)
    if not failing:
        return shown
    return failing[:1] if attempt == 1 else (failing if attempt == 2 else shown)


def attempt_score(gates: Iterable[GateResult]) -> tuple[int, int]:
    """How far one attempt's body got, as the key the kept body is chosen by.

    `run_gates` runs the gates in one order (parses, confined, executes_on_s0, deterministic,
    non_trivial, replay_fidelity, refuses_unknown) and stops at the first failure, so the count of
    gates a body passed is how far down that order it reached. The tie-break is how many recorded
    calls its replay actually matched, which separates two bodies that fell at the same gate.
    """
    gates = list(gates)
    passed = sum(1 for gate in gates if gate.passed)
    matched = sum(int(gate.metrics.get("success_matches") or 0)
                  for gate in gates if gate.stage == "replay_fidelity")
    return passed, matched


def _evidence_label(attempt: int, gates: list[GateResult]) -> str:
    """The node's own name for its evidence, honest about the held-out case."""
    if attempt and _held_out_only(gates):
        return HELD_OUT_LABEL
    return EVIDENCE_LABELS[min(attempt, len(EVIDENCE_LABELS) - 1)]


def call_starting_states(db: dict, overlays: Iterable[TaskOverlay], values: dict,
                         call_tasks: dict) -> dict:
    """Call id to the Starting state of the Task whose trace recorded it (D74).

    `call_tasks` maps a recorded call's id to its Task id; it is what carries the trace-to-Task map
    into the gates, since a `ToolCall` does not name its trace. The state is the shared world with
    that Task's overlay merged, which is the world that call actually ran on, so a corpus holding one
    row in two versions does not make a correct body look wrong.
    """
    states = {overlay.task_id: merge_overlays(db, [overlay], values) for overlay in overlays}
    return {call_id: states[task_id] for call_id, task_id in call_tasks.items() if task_id in states}


def replay_outcomes(toolsig: ToolSig, body: str, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
                    workdir: Path | str, call_states: Optional[dict] = None, rules: Any = None,
                    timeout: float = 30.0) -> list[dict]:
    """Which of these recorded calls the kept body answers the way the recording did, one row per call (D171).

    The corpus gate (`gate_replay_fidelity`) rules over all of a tool's recorded calls at once and
    leaves one fidelity number and a list of failure sentences. That is the right grain for the
    Builder, which repairs a body, and the wrong grain for a Task, which only ever makes some of
    those calls: on one live build a body replayed 239 of its 240 calls and the one miss was read as
    a fault of the tool everywhere the tool was called. So the sandbox is run once over every call
    and the same ruling, `body_replay_fidelity_gate`, is then asked about each call on its own. A
    call counted replayed here is a call the corpus gate counted matched, because it is the same
    ruling over the same result; what is added is the call's id, which is what ties a miss to the
    Trace and so to the Task that made it.

    A body that does not exist, or a module that will not run at all, misses every call: there is
    nothing to attribute a pass to. `rules` is the customer's CanonRules, the same ones the corpus
    gate compares under (D39).
    """
    calls = list(calls)
    if not calls:
        return []

    def missed(detail: str) -> list[dict]:
        return [{"tool": toolsig.name, "call_id": call.id, "replayed": False, "detail": detail}
                for call in calls]

    if not (body or "").strip():
        return missed("no body was compiled for this tool")
    source = module_source(schema, [toolsig], {toolsig.name: body})
    sandbox = Sandbox(source, db, Path(workdir) / "attribution", timeout=timeout, call_states=call_states)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return missed(f"the module did not run: {exc}")
    rows: list[dict] = []
    for index, call in enumerate(calls):
        if index >= len(results):
            rows.append({"tool": toolsig.name, "call_id": call.id, "replayed": False,
                         "detail": "the sandbox returned no result for this call"})
            continue
        ruling = body_replay_fidelity_gate([call], [results[index]], schema, label="per_call", rules=rules)
        detail = "" if ruling.passed else (ruling.failures[0] if ruling.failures else "differs")
        rows.append({"tool": toolsig.name, "call_id": call.id, "replayed": bool(ruling.passed),
                     "detail": _clamped_detail(detail)})
    return rows


def _clamped_detail(text: str) -> str:
    """One difference, short enough to sit in a Task's status row.

    A tool that returns a whole catalogue writes a failure sentence thousands of characters long,
    and the row is read by a person and by the repair verb, not by the comparer; the gate's own
    failures keep the sentence whole.
    """
    text = " ".join(text.split())
    if len(text) <= OUTCOME_DETAIL_CHARS:
        return text
    return f"{text[:OUTCOME_DETAIL_CHARS].rstrip()} [+{len(text) - OUTCOME_DETAIL_CHARS} characters]"


def compile_tool(model, toolsig: ToolSig, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
                 workdir: Path | str, max_attempts: int = MAX_REPAIR_ATTEMPTS,
                 max_evidence_chars: Optional[int] = MAX_EVIDENCE_CHARS, timeout: float = 30.0,
                 call_states: Optional[dict] = None, rules: Any = None,
                 tool_names: Iterable[str] = (), error_prefix: Optional[str] = None,
                 builder_tools: bool = True, lesson: str = "") -> ToolBuild:
    """Write one tool body, gate it, and repair it at most three times with growing evidence (D75).

    Attempt 1 sees the failing call, attempt 2 every failing call, attempt 3 the full call table, and
    every rewrite is re-checked on the shown calls and on the held-out calls separately, with the
    held-out calls never named back to the model. An attempt whose whole prompt would exceed the D65
    cap is refused, not truncated, and the cap is on by default. `call_states` (from
    `call_starting_states`) is the per-Task world each recorded call ran on; without it every call
    replays on the shared db, which is right only where the corpus shows one version of each row.
    `rules` is the customer's CanonRules (D39): given none, the gates compare under the module
    defaults and can fail a body over a difference the customer's own rules fold away. `tool_names`
    is every tool name in this build, stable across every call of this stage, so it lives in the
    system message beside the schema (docs/prompt-caching.md item 1).

    Attempt 0 sends the system and the first user turn; a retry (docs/prompt-caching.md item 2)
    never rewrites either: it appends the previous reply as an assistant turn and the new evidence
    and failure as a new user turn, so the first two messages of every call in the loop are the
    same bytes a cache can reuse. An attempt that raised what the attempt before it raised is told
    so in that new turn (`_SAME_ERROR`), with the exception quoted, since a rewrite that lands on
    the same crash is not a rewrite of the line that crashed.
    Each attempt is a node dict; after the last miss the tool is marked assisted (D49) and the nodes
    are written under the workdir. When no attempt passes, the body kept is the best attempt's, not
    the last one's (`attempt_score`, `ToolBuild.kept_attempt`).

    `builder_tools` (D117, on by default) lets the model call `lookup_rows` and `test_body` while
    it drafts this attempt's reply, inside `_reply_with_tools`'s own bounded loop
    (`MAX_TOOL_ROUNDS`), before the reply it settles on is gated the same way an old, tool-less
    reply always was. Turn it off for a caller that wants the old one-call-per-attempt behaviour.

    The build carries `call_outcomes` (D171): one row per recorded call saying whether the kept body
    answered it the way the recording did, so a miss can be attributed to the Task whose Trace made
    the call and not to every Task that calls the tool.

    `lesson` is what this tool already failed on in an earlier build or an earlier recompile
    request (`memory.lesson_for`, written by `repair.record_tool_lesson`). It is carried into the
    first user turn, so it is in the prefix every retry of this attempt chain keeps; without it the
    recompile asks the same question again and the model has no way to know it was asked before.
    """
    workdir, calls = Path(workdir), list(calls)
    if error_prefix is None:  # build.py passes the corpus-wide prefix; alone, this tool's own calls
        error_prefix = shared_error_prefix(calls)
    shown, held_out = split_calls(calls)
    build, failure = ToolBuild(name=toolsig.name, body=""), ""
    skeleton = gate_parses(module_source(schema, [toolsig], {toolsig.name: "pass"}))
    messages: list[dict] = []
    reply_content = ""
    # The loop grows its evidence off the attempt that just ran, not off the best one so far, so
    # the two are kept apart: `latest` drives the next prompt, `build` holds what is kept.
    latest: list[GateResult] = []
    best: Optional[tuple[int, int]] = None
    last_error = ""  # the exception the attempt before raised, so a repeat can be named as one
    for attempt in range(max_attempts + 1):
        if not skeleton.passed:  # code owns the skeleton, so no model call can repair it
            build.gates = [skeleton]
            build.nodes.append({"attempt": attempt, "tool": toolsig.name, "evidence": "none",
                                "evidence_calls": 0, "passed": False, "refused": True,
                                "failures": [f"the code-owned skeleton does not parse: {f}"
                                             for f in skeleton.failures]})
            break
        evidence = _evidence_for(attempt, shown, latest)
        node = {"attempt": attempt, "tool": toolsig.name, "evidence": _evidence_label(attempt, latest),
                "evidence_calls": len(evidence), "passed": False, "refused": False}
        if attempt == 0:
            messages = body_messages(toolsig, evidence, schema=schema, tool_names=tool_names,
                                     error_prefix=error_prefix, builder_tools=builder_tools,
                                     lesson=lesson)
        else:
            messages = _append_retry(messages, reply_content, evidence, failure, error_prefix)
        # Fewer whole calls, never a shortened one. `_example_block` refuses to cut a call in
        # half and that stays true: what is dropped here is the last recorded call, entire. The
        # first live build refused `get_order_details` outright at 815,972 characters, because the
        # last-resort evidence is every shown call and this corpus has hundreds of them; a body
        # written from thirty complete calls is worth more than an attempt not taken.
        while (max_evidence_chars is not None and len(evidence) > 1
               and prompt_chars(messages) > max_evidence_chars):
            evidence = evidence[:-1]
            node["evidence_calls"] = len(evidence)
            messages = (body_messages(toolsig, evidence, schema=schema, tool_names=tool_names,
                                      error_prefix=error_prefix, builder_tools=builder_tools,
                                      lesson=lesson)
                        if attempt == 0
                        else _append_retry(messages[:-2], reply_content, evidence, failure, error_prefix))
        size = prompt_chars(messages)
        if max_evidence_chars is not None and size > max_evidence_chars:
            node["failures"] = [f"a prompt of {size} characters is over the cap "
                                f"of {max_evidence_chars}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        tools_impl = (_build_tools_impl(schema, toolsig, shown, db, call_states, workdir, attempt,
                                        timeout, rules)
                     if builder_tools else None)
        try:
            if builder_tools:
                reply_content, tool_uses, draft = _reply_with_tools(model, messages, tools_impl)
            else:
                reply_content, tool_uses, draft = model.query(messages).content or "", [], ""
        except _context_cap_error() as refusal:
            # `max_evidence_chars` above is not enough on its own, and the first live build is
            # what showed it. It counts the characters of the message contents; budget.py counts
            # the tokens of the JSON the request will actually carry, which is larger by the
            # envelope and by every escaped quote and newline in a tool result. A prompt of
            # 358,580 content characters was under the 320,000 cap on nothing and over the 80,000
            # token cap at 89,645. No constant reconciles two different measures, so the one that
            # is authoritative is the one the wrapper raises, and it is caught here.
            #
            # Before this it left compile_tool and killed the build: one tool with a large corpus
            # took the other thirteen with it. The stage already has a word for a tool it could not
            # write, so the refusal becomes that word: assisted (D49), and the build carries on.
            node["failures"] = [f"{refusal}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        if tool_uses:
            node["tool_uses"] = tool_uses
        body = _strip_fence(reply_content)
        if not body and draft:
            # The rounds ran out on a model that had tested a draft and was still probing: that
            # draft is what it was submitting, so it is gated like any reply (D117).
            body, reply_content = _strip_fence(draft), draft
            node["body_from_draft"] = True
        if not body:
            # No reply and no draft: nothing to gate, and gating "pass" would blame the code-owned
            # skeleton for a reply the model never gave. The attempt fails and the next one runs
            # with that said; the sixth retail build failed whole on a refusal here, because the
            # stage gate reads an empty body as a tool the Builder could not write.
            failure = "\nno body was submitted: every round asked for a tool; send the body as your reply"
            reply_content = "(no body was submitted)"
            last_error = ""  # nothing ran, so the next attempt's error follows no error of ours
            node["failures"] = ["no body was submitted"]
            build.nodes.append(node)
            continue
        source = module_source(schema, [toolsig], {toolsig.name: body})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}", timeout=timeout,
                          call_states=call_states)
        gates = run_gates(source, sandbox, shown, held_out, schema, rules,
                          probe_refusals=toolsig.kind == "write", sig=toolsig)
        node.update(body_hash=content_hash(body), gates=[as_dict(g) for g in gates],
                    passed=all(g.passed for g in gates))
        build.nodes.append(node)
        latest = gates
        # The last attempt is not the best attempt. A live build's fourth attempt at one write tool
        # crashed on all 67 of its calls where the third had replayed 44 of 44 and failed only the
        # refusal probe; keeping the last one cost 94 replay misses and 38 Tasks. So the body kept
        # is the one that got furthest through the gates (`attempt_score`), and a later attempt
        # replaces it only by beating it. A passing attempt ends the loop, so it is always the best.
        score = attempt_score(gates)
        if best is None or score > best:
            best, build.body, build.gates, build.kept_attempt = score, body, gates, attempt
        if node["passed"]:
            break
        failure = "\n" + _failure_text(gates, held_out)
        if any(g.stage == "non_trivial" and not g.passed for g in gates):
            failure += _constant_evidence_note(evidence)
        error = exception_line(gates)
        if error and error == last_error:
            failure += _SAME_ERROR.format(error=error)
        last_error = error
    build.assisted = not (build.gates and all(g.passed for g in build.gates))
    for record in build.nodes:
        if record["attempt"] == build.kept_attempt:
            record["kept"] = True
    # D171: a tool that cleared every gate replayed every recorded call, since `run_gates` holds the
    # shown and the held-out split to 100% each and only calls a build unassisted when both passed;
    # so its per-call rows are known without a sandbox and only an assisted tool pays for the run.
    build.call_outcomes = (
        replay_outcomes(toolsig, build.body, calls, schema, db, workdir, call_states=call_states,
                        rules=rules, timeout=timeout)
        if build.assisted else
        [{"tool": toolsig.name, "call_id": call.id, "replayed": True, "detail": ""} for call in calls])
    directory = workdir / NODE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{toolsig.name}.json").write_text(
        json.dumps({"tool": toolsig.name, "assisted": build.assisted,
                    "kept_attempt": build.kept_attempt, "nodes": build.nodes,
                    "call_outcomes": build.call_outcomes},
                   indent=2, default=str) + "\n", encoding="utf-8")
    return build

# --- emitting tau2's five files plus the sidecar (D56) ---

def _action(atom: Atom) -> dict:
    """A required atom becomes a tau2 action: the tool it names, with the value it pins as arguments.

    The Verifier's structured target is the source (`Atom.target`); an atom that spelled the call out
    as JSON in `predicate_src` still works, and one that names neither falls back to its description.
    """
    call = dict(atom.target)
    if not call and atom.predicate_src:
        try:
            parsed = json.loads(atom.predicate_src)
            call = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            call = {}
    arguments = call.get("arguments") or (
        {call["field"]: call.get("raw")} if call.get("kind") == "write_value" else {})
    return {"action_id": atom.id, "name": call.get("name") or call.get("tool") or atom.description,
            "arguments": arguments, "info": None}


def _tau2_task(task: Task, verifier: Optional[Verifier], domain: str) -> dict:
    """One entry of tasks.json; everything tau2's shape has no room for goes to the sidecar."""
    atoms = verifier.atoms if verifier else []
    return {
        "id": task.id,
        "description": {"purpose": task.intent, "relevant_policies": None, "notes": None},
        "user_scenario": {"persona": None, "instructions": {
            "task_instructions": None, "domain": domain, "reason_for_call": task.intent,
            "known_info": None, "unknown_info": None}},
        "initial_state": None,
        "evaluation_criteria": {"actions": [_action(a) for a in atoms if a.kind == "required"]},
    }


def build_environment(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict, policy_text: str,
                      builds: Optional[dict] = None, parent_env_id: Optional[str] = None,
                      version: int = 1, files: Optional[dict] = None,
                      assisted_tools: Iterable[str] = ()) -> Environment:
    """The Environment record: identity, the D97 sub-versions, assisted tools and the flags.

    `env_id` is the hash of the five emitted files plus the three sub-versions (design section 5).
    `files` is the file name to file text map `tau2_files` returns; hand it in whenever the emitted
    world is known, because without it two worlds holding different rows, or different Tasks, share
    one env_id and a regrade cannot tell them apart. The hashes are kept on `Environment.files`.

    Flags are what the setup review has to close before the Environment is trusted: a tool whose
    errors are mostly `unknown` (D67) and a tool whose read or write class nobody confirmed (D70).

    `assisted_tools` are the names the compile_tools stage marked assisted (D49), by name because the
    stage hands its builds on as JSON; the second retail build wrote `assisted_tools: []` on an
    Environment with six assisted tools because nothing passed them in.
    """
    from kullback.builder.mine import (
        unknown_error_flags,  # builder to builder; the Runner imports neither
    )

    sigs = list(sigs)
    schema_version = content_hash(as_dict(schema))[:12]
    tools_version = content_hash({"sigs": [as_dict(s) for s in sigs], "bodies": bodies})[:12]
    policy_version = content_hash(policy_text)[:12]
    flags = unknown_error_flags(sigs)
    flags += [f"{sig.name}: read or write not confirmed, defaulted to read (D70)"
              for sig in sorted(sigs, key=lambda s: s.name) if sig.unclassified]
    file_hashes = {name: content_hash(text) for name, text in sorted((files or {}).items())}
    return Environment(
        env_id=content_hash({"schema": schema_version, "tools": tools_version,
                             "policy": policy_version, "files": file_hashes}),
        schema_version=schema_version, tools_version=tools_version, policy_version=policy_version,
        version=version, parent_env_id=parent_env_id, files=file_hashes,
        assisted_tools=sorted({name for name, build in (builds or {}).items() if build.assisted}
                              | set(assisted_tools)),
        flags=flags,
    )


def tau2_files(env: EnvBundle) -> dict:
    """The five tau2 files as text, keyed by name: what is written, and what env_id hashes.

    Overlays merge into the one db.json tau2's harness loads; a conflict between two Tasks is recorded
    on `env.conflicts` for the export gate, and the Task stays gradeable in the Runner (D74).
    """
    db = merge_overlays(env.db, env.overlays, env.overlay_values, env.conflicts) if env.overlays else env.db
    verifiers = {v.task_id: v for v in env.verifiers}
    return {
        "data_model.py": render_data_model(env.schema),
        "tools.py": render_tools(env.schema, env.tools, env.bodies),
        "db.json": json.dumps(db, sort_keys=True, indent=2) + "\n",
        "policy.md": env.policy_text,
        "tasks.json": json.dumps([_tau2_task(t, verifiers.get(t.id), env.domain) for t in env.tasks],
                                 indent=2) + "\n",
    }


def emit_tau2_shape(env: EnvBundle, workdir: Path | str, files: Optional[dict] = None) -> dict:
    """Write data_model.py, tools.py, db.json, policy.md, tasks.json and sidecar.json (D56).

    A caller that already computed `tau2_files(env)` for its own purposes (build.py's environment
    stage needs the files to size the build_environment gate) passes it in, so the render does not
    run a second time on the same inputs.
    """
    workdir = Path(workdir)
    files = tau2_files(env) if files is None else files
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in files.items():
        (workdir / name).write_text(text, encoding="utf-8")
        paths[name] = workdir / name
    sidecar = {
        "env_id": env.environment.env_id,
        "schema_version": env.environment.schema_version,
        "tools_version": env.environment.tools_version,
        "policy_version": env.environment.policy_version,
        "files": {name: content_hash(text) for name, text in files.items()},
        "assisted_tools": list(env.environment.assisted_tools),
        "assumptions": list(env.assumptions),
        "synthetic_rows": list(env.schema.synthetic_rows),
        "overlays": [as_dict(o) for o in env.overlays],
        "atoms": {v.task_id: [as_dict(a) for a in v.atoms] for v in env.verifiers},
    }
    paths["sidecar.json"] = workdir / "sidecar.json"
    paths["sidecar.json"].write_text(json.dumps(sidecar, indent=2, default=str) + "\n", encoding="utf-8")
    return paths
