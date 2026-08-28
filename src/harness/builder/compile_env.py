"""Builds the customer's world: one shared db.json by inverse replay, a per-Task overlay, the tau2 file shape, and each tool body with its five gates and bounded repair.

Minimal sandbox: a generated tool body runs only in a subprocess (`python -I`, environment cleared,
a wall-clock timeout, blocked imports of the process and client modules, and socket connect cut),
never in this process. That is a blast-radius reducer, not a security boundary; a real sandbox is on
todo.md. Said plainly: the body runs with the rights of the runner that calls it, so a body that
wants to can pop the import block off `sys.meta_path`, reach `os.system`, or touch the filesystem;
in-process blocks cannot stop that and this module does not claim to. What is closed is the cheap
forgery: the parent hands the child a nonce on stdin, the child reads and closes stdin before it
executes any generated code, and a result file without that nonce is refused, so a body cannot write
its own answers and exit. Nothing here calls a model on its own: write_tool_body takes a Model.
"""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.shared.budget import CONTEXT_CAP_FRACTION, context_cap_tokens
from harness.shared.canon import canonicalize as canon  # D39: one canonicalizer, no local variant
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
DB_CLASS = "DomainDB"
TOOLS_CLASS = "DomainTools"
MAX_REPAIR_ATTEMPTS = 3
CONTEXT_WINDOW = 200_000  # tokens; the window a Builder call is sized against until config names one
CHARS_PER_TOKEN = 4  # a plain estimate: the D65 cap is a refusal threshold, not token accounting
MAX_EVIDENCE_CHARS = context_cap_tokens(CONTEXT_WINDOW, CONTEXT_CAP_FRACTION) * CHARS_PER_TOKEN
EVIDENCE_LABELS = ("initial", "failing_call", "all_failing_calls", "full_call_table")
HELD_OUT_LABEL = "held_out_failed, shown calls only"  # no shown call failed, so none can be pointed at
CRASH_ERRORS = frozenset({"NameError", "AttributeError", "TypeError", "ImportError",
                          "ModuleNotFoundError", "IndentationError", "SyntaxError", "RecursionError"})


class OverlayConflict(ValueError):
    """Two Tasks pin the same row in different versions: a Gate failure for the tau2 export (D74)."""


class SandboxError(RuntimeError):
    """The generated module did not load, or the subprocess crashed or ran out of time."""


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

def _parse_result(result: Any) -> Any:
    """A recorded result is a value, or the JSON text of one."""
    if isinstance(result, str) and (result.strip()[:1] in "[{"):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def columns_of(schema: EntitySchema, table: str, kind: Optional[str] = None) -> list[str]:
    """Column names of one table, all of them or only those of one class (D73)."""
    return sorted(c.name for c in schema.columns if c.table == table and (kind is None or c.class_ == kind))


def id_field(schema: EntitySchema, table: str) -> Optional[str]:
    """The column holding a row's id, by the customer's own naming."""
    names = set(columns_of(schema, table))
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    for candidate in (f"{singular}_id", "id", f"{table}_id"):
        if candidate in names:
            return candidate
    return next((n for n in sorted(names) if n.endswith("_id")), None)


def match_table(schema: EntitySchema, value: Any) -> Optional[tuple[str, str]]:
    """Which table a returned row belongs to, and its id; None when the value is not a row."""
    if not isinstance(value, dict):
        return None
    best, best_score = None, 1
    for table in sorted(schema.tables):
        name = id_field(schema, table)
        if not name or not isinstance(value.get(name), str):
            continue
        pattern = schema.id_patterns.get(table)
        if pattern and not re.match(pattern, value[name]):
            continue
        score = len(set(columns_of(schema, table)) & set(value))
        if score > best_score:
            best, best_score = (table, value[name]), score
    return best


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
            if call.error is not None:
                continue
            is_write = call.name in write_tools
            rows = extract_rows(schema, _parse_result(call.result))
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


def _referenced_ids(traces: Iterable[Trace], schema: EntitySchema) -> list[tuple[str, str]]:
    """(table, id) pairs a call that succeeded named in its arguments, by the table's own id column."""
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
                    pattern = schema.id_patterns.get(table)
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
    for table, row_id in _referenced_ids(traces, schema):
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
    merged = json.loads(json.dumps(db))
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
    if sig.result_schema:
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
    namespace: dict = {"__name__": "generated_tools"}
    exec(compile(source, "<generated>", "exec", dont_inherit=True), namespace)  # noqa: S102
    return namespace[class_name](namespace[db_class].model_validate(db))

# --- the model writes the body (D56, D75) ---

_SYSTEM = ("You write the body of one Python method of a tool class rebuilt from a customer's traces. "
           "Return only the body: no signature, no fences, no explanation. The body may read and write "
           "self.db, a pydantic model with one dict per table. Raise ValueError with the customer's own "
           "message where the traces show an error.")


def _example_block(calls: Iterable[ToolCall]) -> str:
    """The recorded calls as the model sees them: arguments, then result or error class, in full.

    Nothing is cut here. D75's third attempt is the full call table, and a node that says
    `evidence_calls: 30` has to mean the model saw thirty complete rows; the only limit is the D65
    cap, which refuses the call rather than shortening it.
    """
    lines = []
    for call in calls:
        outcome = (f"error {call.error.class_}: {call.error.payload!r}" if call.error is not None
                   else "result " + json.dumps(_parse_result(call.result), default=str))
        lines.append(f"- args {json.dumps(call.args, sort_keys=True, default=str)} -> {outcome}")
    return "\n".join(lines)


def body_messages(toolsig: ToolSig, examples: Iterable[ToolCall],
                  schema: Optional[EntitySchema] = None, failure: str = "") -> list[dict]:
    """The whole message list one body request sends, so its size can be checked before it goes."""
    parts = [f"Tool: {toolsig.name}",
             f"Description: {toolsig.description or 'not declared by the customer'}",
             "Arguments: " + ", ".join(f"{f.name} ({_annotation(f.types)})" for f in toolsig.args_fields),
             "Result fields: " + ", ".join(f.name for f in toolsig.result_schema)]
    if schema is not None:
        parts.append("Tables on self.db: " + ", ".join(sorted(schema.tables)))
    parts += ["Recorded calls:", _example_block(examples)]
    if failure:
        parts += ["The previous body failed these gates:", failure]
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


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

# --- the minimal sandbox (see the module docstring) ---

_RUNNER = '''
import collections, datetime, importlib.abc, inspect, json, math, re, socket, sys, typing
import pydantic

BLOCKED = {"urllib", "urllib3", "requests", "httpx", "ftplib", "smtplib",
           "telnetlib", "subprocess", "multiprocessing", "webbrowser"}


class _NoNetwork(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked inside the tool sandbox: " + name)
        return None


def _cut(*args, **kwargs):
    raise OSError("the network is blocked in the tool sandbox")


def _plain(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def main():
    # The nonce arrives on stdin, which is read out and closed before any generated code runs, so a
    # body cannot read it back and forge the result file. See the module docstring for what this
    # sandbox does not do.
    nonce = sys.stdin.read().strip()
    sys.stdin.close()
    with open(sys.argv[1], encoding="utf-8") as handle:
        job = json.load(handle)
    sys.meta_path.insert(0, _NoNetwork())
    # socket itself stays importable: pydantic and the standard library reach for it. Only the
    # calls that leave the machine are cut.
    socket.socket.connect = socket.socket.connect_ex = socket.socket.bind = _cut
    socket.create_connection = _cut
    namespace = {"__name__": "generated_tools"}
    exec(compile(job["source"], "<generated>", "exec"), namespace)
    toolkit, db_class = namespace[job["class_name"]], namespace[job["db_class"]]
    results = []
    for call in job["calls"]:
        # Every call starts on the Starting state its own trace ran on: a fresh toolkit over a
        # freshly validated world, so a write cannot leave the next call standing on its output.
        function = getattr(toolkit(db_class.model_validate(job["dbs"][call["db"]])), call["name"])
        try:
            inspect.signature(function).bind(**call["args"])
        except TypeError as exc:
            # The arguments do not fit the signature: an answer for gate 5 to match against the
            # recorded invalid_arguments class, not a crash of the module.
            results.append({"ok": False, "error": "TypeError", "message": str(exc), "binding": True})
            continue
        try:
            results.append({"ok": True, "value": _plain(function(**call["args"]))})
        except Exception as exc:
            results.append({"ok": False, "error": type(exc).__name__, "message": str(exc)})
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump({"nonce": nonce, "results": results}, handle, default=str)


main()
'''


class Sandbox:
    """Runs generated tool code in a subprocess only, with a timeout and no network.

    Every recorded call executes on its own Starting state (design section 6, gate 2): the shared
    world by default, and the Task's own world where `call_states` names one for that call id, which
    is how a corpus showing one row in two versions still replays (D74).
    """

    def __init__(self, source: str, db: dict, workdir: Path | str, class_name: str = TOOLS_CLASS,
                 db_class: str = DB_CLASS, timeout: float = 30.0,
                 call_states: Optional[dict] = None):
        self.source, self.db, self.timeout = source, db, timeout
        self.class_name, self.db_class = class_name, db_class
        self.call_states = dict(call_states or {})  # call id -> the Starting state that call ran on
        self.dir = Path(workdir) / "sandbox"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        self._state_hashes: dict[int, str] = {}
        self.runner = self.dir / "run_tool.py"
        self.runner.write_text(_RUNNER, encoding="utf-8")

    def state_for(self, call: ToolCall) -> dict:
        """The world this call runs on: its own Task's, or the shared one."""
        return self.call_states.get(call.id, self.db) if call.id else self.db

    def _state_hash(self, state: dict) -> str:
        key = self._state_hashes.get(id(state))
        if key is None:
            key = self._state_hashes[id(state)] = content_hash(state)
        return key

    def run(self, calls: Iterable[ToolCall], use_cache: bool = True) -> list[dict]:
        """One result dict per call, in order; the memo is keyed by call, arguments and pre-state.

        Because every call starts on its own pre-state, two identical calls have one answer, and the
        memo returns the first result rather than whatever the world looked like after the last one.
        """
        calls = list(calls)
        keys = [content_hash({"name": c.name, "args": c.args, "state": self._state_hash(self.state_for(c))})
                for c in calls]
        todo, seen = [], set()
        for key, call in zip(keys, calls):
            if (use_cache and key in self.cache) or key in seen:
                continue
            todo.append((key, call))
            seen.add(key)
        for (key, _), result in zip(todo, self._execute([c for _, c in todo]) if todo else []):
            self.cache[key] = result
        return [dict(self.cache[k]) for k in keys]

    def _execute(self, calls: list[ToolCall]) -> list[dict]:
        """One subprocess for one batch of calls; a crash or a timeout is a SandboxError."""
        job, out = self.dir / "job.json", self.dir / "out.json"
        out.unlink(missing_ok=True)
        states: list[dict] = []
        indexes, at = [], {}
        for call in calls:
            key = self._state_hash(self.state_for(call))
            if key not in at:
                at[key] = len(states)
                states.append(self.state_for(call))
            indexes.append(at[key])
        nonce = secrets.token_hex(16)
        job.write_text(json.dumps({"source": self.source, "dbs": states, "db_class": self.db_class,
                                   "class_name": self.class_name,
                                   "calls": [{"name": c.name, "args": c.args, "db": i}
                                             for c, i in zip(calls, indexes)]},
                                  default=str), encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(self.runner), str(job), str(out)],
                                  input=nonce, env={}, cwd=str(self.dir), capture_output=True,
                                  text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise SandboxError(f"timeout after {self.timeout} seconds")
        if done.returncode != 0 or not out.exists():
            raise SandboxError(f"module did not load: {done.stderr.strip()[-400:]}")
        payload = json.loads(out.read_text(encoding="utf-8"))
        if payload.get("nonce") != nonce:
            raise SandboxError("the result file does not carry this run's nonce, so the runner did "
                               "not write it; the results are refused")
        return payload["results"]

# --- the five gates, in the order that localizes a failure (design section 6) ---

def _gate(stage: str, passed: bool, metrics: dict, failures: Iterable[str] = ()) -> GateResult:
    return GateResult(stage=stage, **{"pass": passed}, metrics=metrics, failures=list(failures)[:5])


def _args_text(call: ToolCall) -> str:
    return json.dumps(call.args, sort_keys=True, default=str)


def gate_parses(source: str) -> GateResult:
    """1. The generated module is Python."""
    try:
        compile(source, "<generated>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return _gate("parses", False, {}, [f"line {exc.lineno}: {exc.msg}"])
    return _gate("parses", True, {"chars": len(source)})


def _argument_answer(call: ToolCall, result: dict) -> bool:
    """A refusal about the arguments is an answer gate 5 matches, not a crash of the module (D67).

    The customer's own logs hold calls their tool rejected for a wrong or missing argument. Replaying
    one raises TypeError where the arguments meet the signature, before any body runs, and TypeError
    is otherwise a crash. Such a call is handed on to gate 5, which matches it against the recorded
    `invalid_arguments` class the same way route.py maps TypeError to it.
    """
    return bool(result.get("binding")) or (call.error is not None and call.error.class_ == "invalid_arguments")


def gate_executes_on_s0(sandbox: Sandbox, calls: Iterable[ToolCall]) -> GateResult:
    """2. Every recorded call runs against its own Starting state without crashing the module."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("executes_on_s0", False, {"calls": len(calls)}, [str(exc)])
    crashes = [f"{c.name}({_args_text(c)}) raised {r['error']}: {r['message']}"
               for c, r in zip(calls, results)
               if not r["ok"] and r["error"] in CRASH_ERRORS and not _argument_answer(c, r)]
    return _gate("executes_on_s0", not crashes, {"calls": len(calls), "crashes": len(crashes)}, crashes)


def gate_deterministic(sandbox: Sandbox, calls: Iterable[ToolCall]) -> GateResult:
    """3. Two fresh runs of the same calls give the same answers."""
    calls = list(calls)
    try:
        first, second = sandbox.run(calls, use_cache=False), sandbox.run(calls, use_cache=False)
    except SandboxError as exc:
        return _gate("deterministic", False, {"calls": len(calls)}, [str(exc)])
    differing = [c.name for c, a, b in zip(calls, first, second) if canon(a) != canon(b)]
    return _gate("deterministic", not differing, {"calls": len(calls), "differing": len(differing)},
                 [f"{name} answered differently on a second run" for name in differing])


def gate_non_trivial(sandbox: Sandbox, calls: Iterable[ToolCall]) -> GateResult:
    """4. Different arguments do not all give one constant answer."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("non_trivial", False, {}, [str(exc)])
    metrics = {"arg_sets": len({content_hash(c.args) for c in calls}),
               "distinct_answers": len({content_hash(canon(r)) for r in results})}
    if metrics["arg_sets"] < 2:
        return _gate("non_trivial", True, dict(metrics, insufficient_evidence=True))
    trivial = metrics["distinct_answers"] < 2
    return _gate("non_trivial", not trivial, metrics,
                 ["the body answers every call the same way"] if trivial else [])


def _classify_exception(result: dict) -> str:
    """The raised exception in the D67 classes, so errors are matched by shape and not by text."""
    message, name = (result.get("message") or "").lower(), result.get("error") or ""
    if "not found" in message or "unknown" in message or name == "KeyError":
        return "not_found_entity"
    if name in ("TypeError", "ValidationError") or "invalid" in message or "must be" in message:
        return "invalid_arguments"
    if "permission" in message or "not allowed" in message or "forbidden" in message:
        return "permission_denied"
    return "business_error" if name == "ValueError" else "unknown"


def _row_pairs(schema: EntitySchema, expected: list, got: list) -> list[tuple[Any, Any]]:
    """Pair two lists of rows by id where every row on both sides carries one, else by position."""
    left = [match_table(schema, value) for value in expected]
    right = {found: value for value in got for found in [match_table(schema, value)] if found}
    if all(left) and len(right) == len(got) and all(found in right for found in left):
        return [(value, right[found]) for value, found in zip(expected, left)]
    return list(zip(expected, got))


def _compare(schema: EntitySchema, expected: Any, got: Any) -> tuple[bool, list[str]]:
    """Hard columns must match after canon; semantic ones are reported, not failed (D73, D84).

    A list of rows and a dict wrapping rows are walked into, so the column classes decide there too.
    Comparing a wrapped result as one canonical string would let an exempt column fail a replay that
    the same row returned on its own passes, which is the opposite of what D73 and D84 ask for.
    """
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return False, [f"list of {len(expected)} against {len(got)}"]
        ok, notes = True, []
        for one, other in _row_pairs(schema, expected, got):
            one_ok, one_notes = _compare(schema, one, other)
            ok, notes = ok and one_ok, notes + one_notes
        return ok, notes
    found = match_table(schema, expected)
    if found is None and isinstance(expected, dict) and isinstance(got, dict):
        if set(expected) != set(got):
            return False, [f"keys differ: {sorted(set(expected) ^ set(got))}"]
        ok, notes = True, []
        for key in sorted(expected):
            key_ok, key_notes = _compare(schema, expected[key], got[key])
            ok, notes = ok and key_ok, notes + key_notes
        return ok, notes
    if not found or not isinstance(got, dict):
        return canon(expected) == canon(got), []
    differs = [n for n in columns_of(schema, found[0], "hard") if canon(expected.get(n)) != canon(got.get(n))]
    semantic = [f"semantic:{n}" for n in columns_of(schema, found[0], "semantic")
                if canon(expected.get(n)) != canon(got.get(n))]
    return not differs, differs + semantic


def gate_replay_fidelity(sandbox: Sandbox, calls: Iterable[ToolCall], schema: EntitySchema,
                         label: str = "held_out", threshold: float = 1.0) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("replay_fidelity", False, {"split": label}, [str(exc)])
    hits = {"success_calls": 0, "success_matches": 0, "error_calls": 0, "error_matches": 0}
    semantic, failures = 0, []
    for call, result in zip(calls, results):
        if call.error is not None:
            hits["error_calls"] += 1
            got = _classify_exception(result) if not result["ok"] else None
            if got == call.error.class_:
                hits["error_matches"] += 1
            else:
                failures.append(f"{call.name}({_args_text(call)}): expected error {call.error.class_}, got {got}")
            continue
        hits["success_calls"] += 1
        if not result["ok"]:
            failures.append(f"{call.name}({_args_text(call)}): expected a result, got "
                            f"{result['error']}: {result['message']}")
            continue
        ok, differing = _compare(schema, _parse_result(call.result), result["value"])
        semantic += sum(1 for n in differing if n.startswith("semantic:"))
        if ok:
            hits["success_matches"] += 1
        else:
            failures.append(f"{call.name}({_args_text(call)}): hard columns differ: "
                            f"{', '.join(differing) or 'value'}")
    success = hits["success_matches"] / hits["success_calls"] if hits["success_calls"] else 1.0
    errors = hits["error_matches"] / hits["error_calls"] if hits["error_calls"] else 1.0
    metrics = dict(hits, split=label, success_fidelity=success, error_fidelity=errors,
                   semantic_differences=semantic)
    return _gate("replay_fidelity", success >= threshold and errors >= threshold, metrics, failures)


def run_gates(source: str, sandbox: Sandbox, shown: Iterable[ToolCall], held_out: Iterable[ToolCall],
              schema: EntitySchema) -> list[GateResult]:
    """The five gates in order, stopping at the first failure so the failure localizes (EvoEnv).

    Gate 3 runs over every recorded call, not a first pair: a body that is steady on the first two
    calls and rolls a die on the third is nondeterministic, and one more subprocess is the whole cost
    of seeing it.
    """
    shown, held_out = list(shown), list(held_out)
    every = shown + held_out
    gates = [gate_parses(source)]
    for gate, calls in ((gate_executes_on_s0, every), (gate_deterministic, every),
                        (gate_non_trivial, every)):
        if not gates[-1].passed:
            return gates
        gates.append(gate(sandbox, calls))
    if not gates[-1].passed:
        return gates
    gates.append(gate_replay_fidelity(sandbox, shown, schema, label="shown"))
    if held_out and gates[-1].passed:
        gates.append(gate_replay_fidelity(sandbox, held_out, schema, label="held_out"))
    return gates

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
    return [c for c in shown if any(_args_text(c) in f for f in named)]


def _held_out_only(gates: list[GateResult]) -> bool:
    """True when the held-out replay is the only gate that failed, so no shown call can be shown."""
    failed = [g for g in gates if not g.passed]
    return bool(failed) and all(g.metrics.get("split") == "held_out" for g in failed)


def _failure_text(gates: list[GateResult], held_out: Iterable[ToolCall] = ()) -> str:
    """What the next attempt is told, with the held-out split kept out of it (D51, D75).

    The held-out calls are the ones the model is never shown, so neither their arguments nor the
    outcome expected of them may travel back in a repair prompt. A failure on them is reported as a
    count, which tells the model its body is wrong somewhere without handing it the answers. Gates 2
    to 4 run over every call, so their failure lines are filtered by the same rule.
    """
    hidden = [_args_text(call) for call in held_out]
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
                 call_states: Optional[dict] = None) -> ToolBuild:
    """Write one tool body, gate it, and repair it at most three times with growing evidence (D75).

    Attempt 1 sees the failing call, attempt 2 every failing call, attempt 3 the full call table, and
    every rewrite is re-checked on the shown calls and on the held-out calls separately, with the
    held-out calls never named back to the model. An attempt whose whole prompt would exceed the D65
    cap is refused, not truncated, and the cap is on by default. `call_states` (from
    `call_starting_states`) is the per-Task world each recorded call ran on; without it every call
    replays on the shared db, which is right only where the corpus shows one version of each row.
    Each attempt is a node dict; after the last miss the tool is marked assisted (D49) and the nodes
    are written under the workdir.
    """
    workdir, calls = Path(workdir), list(calls)
    shown, held_out = split_calls(calls)
    build, failure = ToolBuild(name=toolsig.name, body=""), ""
    skeleton = gate_parses(module_source(schema, [toolsig], {toolsig.name: "pass"}))
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
        messages = body_messages(toolsig, evidence, schema=schema, failure=failure)
        size = prompt_chars(messages)
        if max_evidence_chars is not None and size > max_evidence_chars:
            node["failures"] = [f"a prompt of {size} characters is over the cap "
                                f"of {max_evidence_chars}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        body = _body_from(model, messages)
        source = module_source(schema, [toolsig], {toolsig.name: body})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}", timeout=timeout,
                          call_states=call_states)
        gates = run_gates(source, sandbox, shown, held_out, schema)
        node.update(body_hash=content_hash(body), gates=[as_dict(g) for g in gates],
                    passed=all(g.passed for g in gates))
        build.nodes.append(node)
        build.body, build.gates = body, gates
        if node["passed"]:
            break
        failure = "\n" + _failure_text(gates, held_out)
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
    from harness.builder.mine import unknown_error_flags  # builder to builder; the Runner imports neither

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


def emit_tau2_shape(env: EnvBundle, workdir: Path | str) -> dict:
    """Write data_model.py, tools.py, db.json, policy.md, tasks.json and sidecar.json (D56)."""
    workdir = Path(workdir)
    files = tau2_files(env)
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
