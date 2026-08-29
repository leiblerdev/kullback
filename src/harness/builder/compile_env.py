"""Builds the customer's world: one shared db.json by inverse replay, a per-Task overlay, the tau2 file
shape, and each tool body with its five gates and bounded repair.

The gates run each generated tool body in a subprocess sandbox before it is trusted; see
harness.builder.sandbox for what that sandbox does and does not close off. Nothing here calls a
model on its own: write_tool_body takes a Model.
"""

from __future__ import annotations

import copy
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.builder.mine import is_assistant_call, is_scalar_result
from harness.builder.sandbox import (
    ALLOWED_IMPORTS,
    DB_CLASS,
    DENIED_BUILTINS,
    TOOLS_CLASS,
    Sandbox,
    SandboxError,
    args_text,
    gate_parses,
    id_field,
    id_pattern_for,
    match_table,
    parse_result,
    run_gates,
    source_confinement,
)

# Re-exported for tests and callers that reach the gates through this module rather than through
# harness.builder.sandbox directly; compile_env.py's own code calls only run_gates and gate_parses.
from harness.builder.sandbox import gate_deterministic as gate_deterministic
from harness.builder.sandbox import gate_executes_on_s0 as gate_executes_on_s0
from harness.builder.sandbox import gate_non_trivial as gate_non_trivial
from harness.builder.sandbox import gate_replay_fidelity as gate_replay_fidelity
from harness.shared.budget import (
    CHARS_PER_TOKEN,
    CONTEXT_CAP_FRACTION,
    context_cap_tokens,
)
from harness.shared.budget import (
    DEFAULT_CONTEXT_WINDOW as CONTEXT_WINDOW,
)
from harness.shared.canon import (
    canonicalize as canon,  # D39: one canonicalizer, no local variant
)
from harness.shared.records import (
    Atom,
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
MAX_REPAIR_ATTEMPTS = 3
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


@dataclass
class ToolBuild:
    """One compiled tool: the accepted body, every attempt as a node, and whether it went assisted."""
    name: str
    body: str
    nodes: list[dict] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    assisted: bool = False

# --- reading rows out of recorded tool results ---

def extract_rows(schema: EntitySchema, result: Any) -> list[tuple[str, str, dict]]:
    """Rows a result states directly: itself, or the elements of a returned list.

    A value nested inside a row (an item inside an order) is not read as a row of its own.
    """
    values = result if isinstance(result, list) else [result]
    return [(t, i, v) for v in values for t, i in [match_table(schema, v) or (None, None)] if t]

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
            rows = extract_rows(schema, parse_result(call.result))
            for table, row_id, row in rows:
                out.append(_Obs(table, row_id, row, trace.trace_id, (trace_index, call_index),
                                is_write or row_id in written))
            if is_write:
                written |= {row_id for _, row_id, _ in rows}
                written |= {v for v in call.args.values() if isinstance(v, str)}
    return out


def build_starting_state(
    traces: Iterable[Trace],
    schema: EntitySchema,
    workdir: Path | str,
    tasks: Optional[Iterable[Task]] = None,
    tool_sigs: Optional[Iterable[ToolSig]] = None,
    synthetic: bool = True,
) -> StartingState:
    """One shared db.json for the customer, plus one TaskOverlay per Task (D33, D74).

    Inverse replay: a row's shared value is the latest sighting that no write had touched yet, so an
    observed write is undone. Where a trace shows only the post-state, that state is kept and the
    assumption is recorded. Order is the order the traces are passed in, then call order; nothing is
    keyed by wall-clock time (design section 8). Ids the traces asked for but never showed are then
    filled with tagged synthetic rows (D40), unless `synthetic` is off.
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

    added = add_synthetic_rows(db, schema, traces) if synthetic else []
    assumptions += [f"{table_of} row {row_id} was never shown by a trace; it is a synthetic row "
                    "shaped from the observed rows and a Run that reads it is assisted"
                    for table_of, row_id in added]
    overlays = _build_overlays(observations, tasks or [], workdir, assumptions)
    path = workdir / DB_FILE
    path.write_text(json.dumps(db, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (workdir / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n", encoding="utf-8")
    return StartingState(db=db, overlays=overlays, assumptions=assumptions, path=path,
                         synthetic_rows=[row_id for _, row_id in added])


def referenced_ids(traces: Iterable[Trace], schema: EntitySchema) -> list[tuple[str, str]]:
    """(table, id) pairs a call that succeeded named in its arguments, by the table's own id column.

    Public: build.py's build_environment gate wires these ids into validate.environment_gate, to
    check db.json actually holds every id a trace referenced.
    """
    fields = {table: id_field(schema, table) for table in schema.tables}
    out: set[tuple[str, str]] = set()
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None:  # an id the customer's tool refused is not a row we owe
                continue
            for name, value in call.args.items():
                if not isinstance(value, str):
                    continue
                for table, field_name in fields.items():
                    pattern = id_pattern_for(schema, table, field_name)
                    if field_name == name and (not pattern or re.match(pattern, value)):
                        out.add((table, value))
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
        name = id_field(schema, table)
        db.setdefault(table, {})[row_id] = dict(_modal_row(rows.values()), **({name: row_id} if name else {}))
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
                       trace_id=rows[(t, i)].trace_id)
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


def merge_overlays(db: dict, overlays: Iterable[TaskOverlay], values: dict) -> dict:
    """Merge every Task overlay into the one db tau2's harness loads; disagreement is a failure (D74)."""
    merged = copy.deepcopy(db)
    pinned: dict[tuple[str, str], tuple[str, str]] = {}
    for overlay in overlays:
        for row in overlay.rows:
            seen = pinned.get((row.table, row.id))
            if seen and seen[0] != row.version_hash:
                raise OverlayConflict(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row "
                                      f"{row.id} in different versions")
            pinned[(row.table, row.id)] = (row.version_hash, overlay.task_id)
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
    """
    if overlay is not None:
        db = merge_overlays(db, [overlay], overlay_values or {})
    refused = source_confinement(source, class_name)
    if refused:
        raise SandboxError("the generated module is not confined and would run in this process: "
                           + "; ".join(refused))
    namespace: dict = {"__name__": "generated_tools"}
    exec(compile(source, "<generated>", "exec", dont_inherit=True), namespace)  # noqa: S102
    return namespace[class_name](namespace[db_class].model_validate(db))

# --- the model writes the body (D56, D75) ---

# tau2 wraps every raised exception as f"Error: {e}" before it reaches the agent
# (vendor/tau2-bench, Environment.get_response), on every domain we have: retail, airline and
# telecom raw traces all carry it. That prefix is the transport talking, not the customer's tool,
# so a stored payload of "Error: User not found" is not the message a ValueError should carry.
# `ToolCallError.payload` keeps the wrapper (D67 keeps the payload verbatim, with `raw_ptr` back to
# D66's untouched byte); only the copy shown to the model has it peeled off. The first live build
# copied it into seven bodies, faithfully, because nothing said it was not part of the message.
_ERROR_TRANSPORT_PREFIX = "Error: "


def _display_error_payload(payload: Any) -> Any:
    """The error payload as the model should read it, with the known transport wrapper peeled off.

    Only a leading, exact "Error: " is peeled, and only from a str payload; a JSON payload (D67's
    `code` class) and a payload that never carried the prefix pass through untouched.
    """
    if isinstance(payload, str) and payload.startswith(_ERROR_TRANSPORT_PREFIX):
        return payload[len(_ERROR_TRANSPORT_PREFIX):]
    return payload


_SYSTEM = ("You write the body of one Python method of a tool class rebuilt from a customer's traces. "
           "Return only the body: no signature, no fences, no explanation. The body may read and write "
           "self.db, a pydantic model with one dict per table. Each dict's values are pydantic model rows, "
           "not plain dicts: read or write a row's field by attribute, as in order.status or "
           "order.status = \"cancelled\", never with .get(...) or any other dict method. Raise ValueError "
           "with the customer's own message where the traces show an error. A recorded error is shown "
           "without the transport's leading 'Error: ', so write the message exactly as shown and do not "
           "put an 'Error: ' of your own in front of it.")


def _example_block(calls: Iterable[ToolCall]) -> str:
    """The recorded calls as the model sees them: arguments, then result or error class, in full.

    Nothing is cut here. D75's third attempt is the full call table, and a node that says
    `evidence_calls: 30` has to mean the model saw thirty complete rows; the only limit is the D65
    cap, which refuses the call rather than shortening it.
    """
    lines = []
    for call in calls:
        outcome = (f"error {call.error.class_}: {_display_error_payload(call.error.payload)!r}"
                   if call.error is not None
                   else "result " + json.dumps(parse_result(call.result), default=str))
        lines.append(f"- args {json.dumps(call.args, sort_keys=True, default=str)} -> {outcome}")
    return "\n".join(lines)


def _schema_block(schema: EntitySchema) -> str:
    """Tables and columns of the customer's world: the same for every tool in this build (D65's
    stable prefix, docs/prompt-caching.md item 1).

    A column mined as a dict gets one sample beside it. Mining cannot tell a customer's real table
    from a stand-in built out of a value nested inside another table's column: retail's
    get_item_details answers with an item-shaped dict, so mining proposes an "items" table for it,
    and get_product_details answers with a product whose "variants" column holds dicts of exactly
    that shape. Both are real observations and nothing here decides which is the customer's real
    storage; the sample is what lets the model notice the second on its own, instead of trusting
    the "items" table alone and raising "not found" on a database that only fills it through the
    nesting. That is what happened on the first live build, nine calls out of nine.
    """
    lines = ["Tables on self.db:"]
    for table in sorted(schema.tables):
        columns = sorted((c for c in schema.columns if c.table == table), key=lambda c: c.name)
        names = ", ".join(c.name for c in columns) if columns else "(no columns observed)"
        lines.append(f"- {table}: {names}")
        for column in columns:
            if "dict" in (column.evidence or {}).get("types", []) and column.samples:
                lines.append(f"    {table}.{column.name} looks like: {column.samples[0]}")
    return "\n".join(lines)


def _confinement_block() -> str:
    """The confinement gate's own rules, in words, generated from the gate's own constants.

    The first live build spent four of sixteen tools discovering these by being refused: the model
    reached for `getattr` and `__dict__` because nothing had told it not to, and each refusal cost
    an attempt to learn one rule. Written out here it costs nothing, because this text sits in the
    stable system prefix that every call of the stage reuses from the provider's cache.

    Generated rather than written so the two can never drift: if sandbox.py starts denying a name,
    the prompt says so on the next build without anyone remembering to edit it.
    """
    return ("The body is checked before it runs and is refused if it names anything outside the "
            "customer's world. It may not use: " + ", ".join(sorted(DENIED_BUILTINS)) + ". It may "
            "not touch a dunder attribute (`__dict__`, `__class__`, `__globals__` and the rest). "
            "It may import only: " + ", ".join(sorted(ALLOWED_IMPORTS)) + ". Read fields by name "
            "(`order.status`) or by key (`self.db.orders[order_id]`), never through getattr.")


def _stable_system(schema: Optional[EntitySchema] = None, tool_names: Iterable[str] = ()) -> str:
    """`_SYSTEM` plus what every tool in this build shares: one prefix, sent unchanged on every
    call of the stage, long enough on a real customer to clear a provider's cache minimum."""
    parts = [_SYSTEM, _confinement_block()]
    if schema is not None:
        parts.append(_schema_block(schema))
    names = sorted(set(tool_names))
    if names:
        parts.append("Tools in this build: " + ", ".join(names))
    return "\n\n".join(parts)


def _tool_block(toolsig: ToolSig, examples: Iterable[ToolCall]) -> str:
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
             "Recorded calls:", _example_block(examples)]
    return "\n".join(parts)


def body_messages(toolsig: ToolSig, examples: Iterable[ToolCall], schema: Optional[EntitySchema] = None,
                  failure: str = "", tool_names: Iterable[str] = ()) -> list[dict]:
    """The whole message list one body request sends, so its size can be checked before it goes.

    The system message carries the fixed instructions plus what is the same for every tool in
    this build (the schema, the tool list): one prefix, unchanged call to call, for a provider's
    cache to reuse. The user message carries only this one tool, its recorded calls, and (for a
    one-shot request outside the repair loop) the failure of a previous attempt.
    """
    user = _tool_block(toolsig, examples)
    if failure:
        user += "\n\nThe previous body failed these gates:\n" + failure
    return [{"role": "system", "content": _stable_system(schema, tool_names)},
            {"role": "user", "content": user}]


def _append_retry(messages: list[dict], reply_content: str, evidence: Iterable[ToolCall],
                  failure: str) -> list[dict]:
    """A gate-failure retry (D75, docs/prompt-caching.md item 2): the messages so far are kept
    exactly as they were sent, so the system and first user turn stay the cached prefix; the
    model's previous reply arrives as an assistant turn and the new evidence and failure as a
    new user turn, never folded back into the first one."""
    turn = "Recorded calls:\n" + _example_block(evidence) + "\n\nThe previous body failed these gates:\n" + failure
    return messages + [{"role": "assistant", "content": reply_content},
                       {"role": "user", "content": turn}]


def _context_cap_error() -> tuple:
    """budget.py's D65 refusal, imported late so this module does not need budget.py to exist."""
    try:
        from harness.shared.budget import ContextCapExceeded
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
    return "\n".join(lines)


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


def compile_tool(model, toolsig: ToolSig, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
                 workdir: Path | str, max_attempts: int = MAX_REPAIR_ATTEMPTS,
                 max_evidence_chars: Optional[int] = MAX_EVIDENCE_CHARS, timeout: float = 30.0,
                 call_states: Optional[dict] = None, rules: Any = None,
                 tool_names: Iterable[str] = ()) -> ToolBuild:
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
    same bytes a cache can reuse.
    Each attempt is a node dict; after the last miss the tool is marked assisted (D49) and the nodes
    are written under the workdir.
    """
    workdir, calls = Path(workdir), list(calls)
    shown, held_out = split_calls(calls)
    build, failure = ToolBuild(name=toolsig.name, body=""), ""
    skeleton = gate_parses(module_source(schema, [toolsig], {toolsig.name: "pass"}))
    messages: list[dict] = []
    reply_content = ""
    for attempt in range(max_attempts + 1):
        if not skeleton.passed:  # code owns the skeleton, so no model call can repair it
            build.gates = [skeleton]
            build.nodes.append({"attempt": attempt, "tool": toolsig.name, "evidence": "none",
                                "evidence_calls": 0, "passed": False, "refused": True,
                                "failures": [f"the code-owned skeleton does not parse: {f}"
                                             for f in skeleton.failures]})
            break
        evidence = _evidence_for(attempt, shown, build.gates)
        node = {"attempt": attempt, "tool": toolsig.name, "evidence": _evidence_label(attempt, build.gates),
                "evidence_calls": len(evidence), "passed": False, "refused": False}
        if attempt == 0:
            messages = body_messages(toolsig, evidence, schema=schema, tool_names=tool_names)
        else:
            messages = _append_retry(messages, reply_content, evidence, failure)
        # Fewer whole calls, never a shortened one. `_example_block` refuses to cut a call in
        # half and that stays true: what is dropped here is the last recorded call, entire. The
        # first live build refused `get_order_details` outright at 815,972 characters, because the
        # last-resort evidence is every shown call and this corpus has hundreds of them; a body
        # written from thirty complete calls is worth more than an attempt not taken.
        while (max_evidence_chars is not None and len(evidence) > 1
               and prompt_chars(messages) > max_evidence_chars):
            evidence = evidence[:-1]
            node["evidence_calls"] = len(evidence)
            messages = (body_messages(toolsig, evidence, schema=schema, tool_names=tool_names)
                        if attempt == 0 else _append_retry(messages[:-2], reply_content, evidence, failure))
        size = prompt_chars(messages)
        if max_evidence_chars is not None and size > max_evidence_chars:
            node["failures"] = [f"a prompt of {size} characters is over the cap "
                                f"of {max_evidence_chars}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        try:
            reply = model.query(messages)
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
        reply_content = reply.content or ""
        body = _strip_fence(reply_content)
        source = module_source(schema, [toolsig], {toolsig.name: body})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}", timeout=timeout,
                          call_states=call_states)
        gates = run_gates(source, sandbox, shown, held_out, schema, rules)
        node.update(body_hash=content_hash(body), gates=[as_dict(g) for g in gates],
                    passed=all(g.passed for g in gates))
        build.nodes.append(node)
        build.body, build.gates = body, gates
        if node["passed"]:
            break
        failure = "\n" + _failure_text(gates, held_out)
        if any(g.stage == "non_trivial" and not g.passed for g in gates):
            failure += _constant_evidence_note(evidence)
    build.assisted = not build.nodes[-1]["passed"]
    directory = workdir / NODE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{toolsig.name}.json").write_text(
        json.dumps({"tool": toolsig.name, "assisted": build.assisted, "nodes": build.nodes},
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
                      version: int = 1, files: Optional[dict] = None) -> Environment:
    """The Environment record: identity, the D97 sub-versions, assisted tools and the flags.

    `env_id` is the hash of the five emitted files plus the three sub-versions (design section 5).
    `files` is the file name to file text map `tau2_files` returns; hand it in whenever the emitted
    world is known, because without it two worlds holding different rows, or different Tasks, share
    one env_id and a regrade cannot tell them apart. The hashes are kept on `Environment.files`.

    Flags are what the setup review has to close before the Environment is trusted: a tool whose
    errors are mostly `unknown` (D67) and a tool whose read or write class nobody confirmed (D70).
    """
    from harness.builder.mine import (
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
        assisted_tools=sorted(name for name, build in (builds or {}).items() if build.assisted),
        flags=flags,
    )


def tau2_files(env: EnvBundle) -> dict:
    """The five tau2 files as text, keyed by name: what is written, and what env_id hashes.

    Overlays merge into the one db.json tau2's harness loads; a conflict between two Tasks is a Gate
    failure and raises here, before anything is written (D74).
    """
    db = merge_overlays(env.db, env.overlays, env.overlay_values) if env.overlays else env.db
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
