"""Readers: the columns a requestor's prose results describe, proposed as code and gated on the recording.

A customer's recording may carry a requestor beside the assistant, running a toolkit of its own
(D164). Where that toolkit answers with prose rather than with rows, `compile_env.extract_rows`
reads nothing from it and R33 keeps it out of the Starting state, so the world holds no row for the
thing those tools read and write, and every body written for them reads state the world does not
have.

This module builds that row. Per requestor with prose results, the model is shown the distinct
result shapes of each of its tools, with the digits masked, and proposes one table, its columns,
and per tool a reader: `def read(result)` mapping one result string to the column values it
asserts, or None for a result that asserts nothing. A gate runs every reader over every recorded
result of its tool and hands what failed back as the next attempt's evidence, one line per masked
shape. The readers and the derivations run in the same subprocess sandbox as a tool body
(`builder/sandbox.py`), never in this process.

Two gates, chosen by one flag, because which one is right is the open question:

- `declared`: the proposal also says per column whether it is stored or derived (with the
  derivation as code) and per tool which shapes are acknowledgements, a result asserting no column
  value, plus for a tool that only acknowledges the columns its write changes. The gate holds all
  three declarations to the recording.
- `replay`: the proposal is readers only, and the gate asks only that each one parses, runs on
  every recorded result without raising, and answers with a dict or None. Everything else is left
  to replay fidelity.

Nothing here names a domain, a tool or a value: what is domain is read off the corpus, and what is
code is the shape of a row and the shape of a function over a string.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.builder.mine import _reply_json, propose_column_class
from kullback.builder.sandbox import Sandbox, SandboxError, gate_parses, parse_result
from kullback.gates.confinement import ALLOWED_IMPORTS, DENIED_BUILTINS, gate_confined
from kullback.runner.canon import canonicalize as canon
from kullback.runner.records import (
    Column,
    EntitySchema,
    RawPtr,
    ToolCall,
    Trace,
    content_hash,
)

READERS_FILE = "readers.json"
GATE_DECLARED = "declared"
GATE_REPLAY = "replay"
GATES = (GATE_DECLARED, GATE_REPLAY)
NO_PROSE = "no prose results"
ASSISTANT = "assistant"

MAX_ATTEMPTS = 4
MAX_EXAMPLES = 3  # verbatim results shown per masked shape
MAX_SEQUENCES = 3  # recordings whose call order is shown, so an effect between tools can be seen
MAX_SEQUENCE_CALLS = 24
MAX_FEEDBACK_LINES = 40  # failures carried into the next attempt, one line per masked shape
MAX_VALUE_CHARS = 120  # a value quoted back inside a failure line
SANDBOX_TIMEOUT = 300.0
STORED, DERIVED = "stored", "derived"

_DIGITS = re.compile(r"\d+")
_PTR = RawPtr(file_hash="readers")  # every ToolCall cites a raw location (D66); these are not recorded calls


def mask(text: str) -> str:
    """One result with its digits masked, which is the grain the model and the gate both name.

    Two results that differ only in a battery percentage or a signal strength are one shape: the
    model proposes one reader for them and the gate reports one failure line for them.
    """
    return _DIGITS.sub("N", text)


def is_prose(call: Any) -> bool:
    """Whether this call answered with a string the row extractor cannot read a row out of."""
    if call.error is not None:
        return False
    parsed = parse_result(call.result)
    return isinstance(parsed, str) and bool(parsed.strip())


def prose_calls(traces: Iterable[Trace]) -> dict[str, dict[str, list[ToolCall]]]:
    """Per requestor other than the assistant, per tool, the calls that answered with prose.

    The assistant's own calls are left out on purpose: only they describe the customer's system
    (R33), and this module is about the rows another requestor's tools reveal.
    """
    out: dict[str, dict[str, list[ToolCall]]] = {}
    for trace in traces:
        for call in trace.tool_calls:
            requestor = call.requestor or ASSISTANT
            if requestor == ASSISTANT or not is_prose(call):
                continue
            out.setdefault(requestor, {}).setdefault(call.name, []).append(call)
    return out


@dataclass
class ResultShape:
    """One masked shape of one tool's results, with how many calls carried it and a few verbatim."""
    shape: str
    count: int
    examples: list[str] = field(default_factory=list)


def shapes_of(calls: Iterable[ToolCall]) -> list[ResultShape]:
    """The distinct masked shapes of these calls' results, commonest first.

    An example is carried only where masking actually changed the text: where it did not, the shape
    is the result and repeating it verbatim says nothing the shape has not already said.
    """
    seen: dict[str, ResultShape] = {}
    for call in calls:
        text = str(parse_result(call.result))
        shape = mask(text)
        entry = seen.setdefault(shape, ResultShape(shape=shape, count=0))
        entry.count += 1
        if text != shape and text not in entry.examples and len(entry.examples) < MAX_EXAMPLES:
            entry.examples.append(text)
    return sorted(seen.values(), key=lambda s: (-s.count, s.shape))


@dataclass
class ColumnSpec:
    """One proposed column: stored on the row, or derived from the stored ones by `derive`."""
    name: str
    origin: str = STORED
    derive: str = ""

    @property
    def derived(self) -> bool:
        return self.origin == DERIVED


@dataclass
class ToolReader:
    """One tool's reader: the source of a function over a result string, and its declarations.

    `acknowledgements` are the masked shapes the proposal says assert no column value, and
    `changes` the columns a tool that only acknowledges says its write changes. Both are empty
    under the `replay` gate, which asks for neither.
    """
    tool: str
    source: str
    acknowledgements: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)


@dataclass
class Proposal:
    """One requestor's world as the model proposed it, and how the gate ruled on it."""
    requestor: str
    table: str
    columns: list[ColumnSpec] = field(default_factory=list)
    readers: list[ToolReader] = field(default_factory=list)
    gate: str = GATE_DECLARED
    attempts: int = 0
    failures: list[str] = field(default_factory=list)
    assisted: bool = False

    def reader_for(self, tool: str) -> Optional[ToolReader]:
        return next((r for r in self.readers if r.tool == tool), None)

    def column(self, name: str) -> Optional[ColumnSpec]:
        return next((c for c in self.columns if c.name == name), None)

    def stored_names(self) -> list[str]:
        return [c.name for c in self.columns if not c.derived]

    def derived_names(self) -> list[str]:
        return [c.name for c in self.columns if c.derived]

    def changes_of(self, tool: str) -> list[str]:
        reader = self.reader_for(tool)
        return list(reader.changes) if reader is not None else []

    def to_dict(self) -> dict:
        return {
            "requestor": self.requestor, "table": self.table, "gate": self.gate,
            "attempts": self.attempts, "failures": list(self.failures), "assisted": self.assisted,
            "columns": [{"name": c.name, "origin": c.origin, "derive": c.derive} for c in self.columns],
            "readers": [{"tool": r.tool, "source": r.source, "acknowledgements": list(r.acknowledgements),
                         "changes": list(r.changes)} for r in self.readers],
        }

    @classmethod
    def from_dict(cls, body: dict) -> "Proposal":
        return cls(
            requestor=str(body.get("requestor") or ""), table=str(body.get("table") or ""),
            gate=str(body.get("gate") or GATE_DECLARED), attempts=int(body.get("attempts") or 0),
            failures=[str(f) for f in body.get("failures") or []], assisted=bool(body.get("assisted")),
            columns=[ColumnSpec(name=str(c.get("name") or ""), origin=str(c.get("origin") or STORED),
                                derive=str(c.get("derive") or "")) for c in body.get("columns") or []],
            readers=[ToolReader(tool=str(r.get("tool") or ""), source=str(r.get("source") or ""),
                                acknowledgements=[str(s) for s in r.get("acknowledgements") or []],
                                changes=[str(s) for s in r.get("changes") or []])
                     for r in body.get("readers") or []],
        )


def proposals_from(artifact: Any) -> list[Proposal]:
    """The proposals held in the `readers` artifact, in requestor order; none when there were none."""
    body = artifact if isinstance(artifact, dict) else {}
    return [Proposal.from_dict(row) for _, row in sorted((body.get("proposals") or {}).items())]


def rows_from(artifact: Any) -> dict:
    """The per-trace starting rows the readers parsed, as `{requestor: {trace_id: {column: value}}}`."""
    body = artifact if isinstance(artifact, dict) else {}
    return dict(body.get("rows") or {})


# --- the model's turn -------------------------------------------------------

_SYSTEM = (
    "You are given the recorded results of the tools one requestor of a recording ran itself, "
    "beside the assistant's own. Those results are prose strings, not objects, so nothing reads a "
    "row out of them and the rebuilt world holds no row for the thing they describe. Your job is to "
    "propose that row: one table, the columns it holds, and per tool a reader, a Python function "
    "over one result string.\n\n"
    "What you receive, as one JSON object: per tool, the arguments it was called with, and every "
    "distinct shape its results took, with the digits masked as N, how many calls carried that "
    "shape, and up to three verbatim results behind it. You also receive the order of the calls "
    "inside a few recordings, so an effect one tool has on what a later tool answers can be seen.\n\n"
    "A reader is `def read(result):` returning a dict of the column values that result asserts, or "
    "None for a result that asserts none. It reads the string and nothing else: it takes no other "
    "argument, keeps no state between calls, and holds no value that is not in the string it was "
    "given. Every key it returns is a column you propose. A column value is a plain scalar, the "
    "same value however the sentence around it is worded, so two results that say the same thing "
    "in two wordings parse to the same value.\n\n"
    "Name the table and the columns after what they hold, in the recording's own words, lowercase "
    "with underscores. Do not propose a column that is an id: the table holds one row per "
    "requestor and code owns its key, so a column named id, or the table's name with _id after "
    "it, is refused.\n"
)

_REPLY_SHAPE = (
    "Answer with one JSON object and nothing else:\n"
    '{"table": "<table name>",\n'
    ' "columns": [{"name": "<column>"}, ...],\n'
    ' "readers": [{"tool": "<tool name>", "source": "def read(result):\\n    ..."}, ...]}\n'
    "One reader per tool you were shown, and no reader for a tool you were not.\n"
)

_REPLY_SHAPE_DECLARED = (
    "Answer with one JSON object and nothing else:\n"
    '{"table": "<table name>",\n'
    ' "columns": [{"name": "<column>", "origin": "stored"},\n'
    '             {"name": "<column>", "origin": "derived",\n'
    '              "derive": "def derive_<column>(row):\\n    ..."}, ...],\n'
    ' "readers": [{"tool": "<tool name>", "source": "def read(result):\\n    ...",\n'
    '              "acknowledgements": ["<masked shape>", ...],\n'
    '              "changes": ["<column>", ...]}, ...]}\n'
    "One reader per tool you were shown, and no reader for a tool you were not.\n\n"
    "A column is stored when the row has to remember it, and derived when its value follows from "
    "the stored columns of the same row: then `derive` is a function over the row that computes it, "
    "and the gate checks that computed value against every result the reader parsed it out of. A "
    "derivation that cannot tell from the row it was given returns None and is not checked there.\n"
    "`acknowledgements` are the masked shapes of this tool whose results assert no column value at "
    "all, the ones its reader answers None on; every other shape must parse to a non-empty dict of "
    "the columns you propose. A tool whose every shape is an acknowledgement is a write nobody can "
    "read the world out of, so it names in `changes` the columns its call changes, and the gate "
    "looks for that change between a read before the call and a read after it.\n"
)


def _code_rules() -> str:
    """The sandbox's own rules, generated from the gate's constants so the two cannot drift."""
    return ("The reader and the derivations are checked before they run and are refused if they "
            "name anything outside their own argument. They may not use: "
            + ", ".join(sorted(DENIED_BUILTINS)) + ". They may not touch a dunder attribute. They "
            "may import only: " + ", ".join(sorted(ALLOWED_IMPORTS)) + ". Write the import inside "
            "the function, and do not read or write any file, clock or network.")


def _feedback_shape() -> str:
    return ("If a proposal does not hold, you are given it back with one line per shape that "
            "failed, naming the tool, the masked shape and what went wrong. Answer with the whole "
            "object again, corrected. Stop when every line is answered: send the object, not a "
            "commentary on it.")


def system_prompt(gate: str) -> str:
    """The stable prefix of every call of this stage: the same bytes for every attempt and requestor."""
    shape = _REPLY_SHAPE_DECLARED if gate == GATE_DECLARED else _REPLY_SHAPE
    return "\n".join([_SYSTEM, shape, _code_rules(), _feedback_shape()])


def evidence_payload(requestor: str, calls_by_tool: dict[str, list[ToolCall]],
                     traces: Iterable[Trace]) -> dict:
    """What one proposal is asked from: the shapes per tool, and a few recordings' call order."""
    tools = []
    for name in sorted(calls_by_tool):
        calls = calls_by_tool[name]
        arguments = sorted({key for call in calls for key in (call.args or {})})
        tools.append({
            "tool": name,
            "arguments": arguments,
            "calls": len(calls),
            "shapes": [{"shape": s.shape, "calls": s.count, **({"examples": s.examples} if s.examples else {})}
                       for s in shapes_of(calls)],
        })
    return {"requestor": requestor, "tools": tools,
            "call_order_in_recordings": _sequences(requestor, traces)}


def _sequences(requestor: str, traces: Iterable[Trace]) -> list[list[dict]]:
    """The order this requestor's calls came in, inside the few recordings that show the most of them."""
    walks: list[tuple[int, str, list[dict]]] = []
    for trace in traces:
        steps = [{"tool": call.name, "shape": mask(str(parse_result(call.result)))}
                 for call in trace.tool_calls
                 if (call.requestor or ASSISTANT) == requestor and is_prose(call)]
        if len(steps) > 1:
            walks.append((-len(steps), trace.trace_id, steps[:MAX_SEQUENCE_CALLS]))
    return [steps for _, _, steps in sorted(walks)[:MAX_SEQUENCES]]


def _parse_reply(reply: Any, requestor: str, gate: str) -> Optional[Proposal]:
    """The model's JSON as a Proposal; None when the reply carries no usable object."""
    body = _reply_json(reply)
    table = str(body.get("table") or "").strip()
    columns = body.get("columns")
    readers = body.get("readers")
    if not table or not isinstance(columns, list) or not isinstance(readers, list):
        return None
    declared = gate == GATE_DECLARED
    specs = [ColumnSpec(name=str(c.get("name") or "").strip(),
                        origin=(str(c.get("origin") or STORED).strip() if declared else STORED),
                        derive=(str(c.get("derive") or "") if declared else ""))
             for c in columns if isinstance(c, dict) and str(c.get("name") or "").strip()]
    tools = [ToolReader(tool=str(r.get("tool") or "").strip(), source=str(r.get("source") or ""),
                        acknowledgements=([str(s) for s in r.get("acknowledgements") or []] if declared else []),
                        changes=([str(s) for s in r.get("changes") or []] if declared else []))
             for r in readers if isinstance(r, dict) and str(r.get("tool") or "").strip()]
    if not specs or not tools:
        return None
    return Proposal(requestor=requestor, table=table, columns=specs, readers=tools, gate=gate)


# --- the sandbox module the readers and derivations run in -------------------

def _last_function(source: str) -> Optional[str]:
    """The name of the last function the source defines at its top level, or None when it defines none."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return names[-1] if names else None


def _method_body(source: str, argument: str) -> Optional[str]:
    """One model-written function as the body of a generated method: its source, then a call to it.

    The function is nested inside the method, so it is checked by the same confinement gate a tool
    body is and it runs in the same subprocess sandbox. The name is read off the source rather than
    fixed, so a derivation named after its column and a reader named `read` both work.
    """
    name = _last_function(source)
    return None if name is None else f"{source.rstrip()}\n\nreturn {name}({argument})"


def _method_names(proposal: Proposal) -> tuple[dict[str, str], dict[str, str]]:
    """Stable method names for the readers and the derivations: position, never a customer's word."""
    readers = {reader.tool: f"reader_{index}" for index, reader in enumerate(sorted(
        proposal.readers, key=lambda r: r.tool))}
    derived = {name: f"derivation_{index}" for index, name in enumerate(sorted(proposal.derived_names()))}
    return readers, derived


def _module(proposal: Proposal) -> tuple[str, list[str]]:
    """The one module the sandbox runs: a method per reader and per derivation. Refusals come back."""
    from kullback.builder.compile_env import module_source  # builder to builder, at call time
    from kullback.runner.records import FieldStat, ToolSig

    reader_names, derived_names = _method_names(proposal)
    sigs, bodies, refused = [], {}, []
    for reader in sorted(proposal.readers, key=lambda r: r.tool):
        body = _method_body(reader.source, "result")
        if body is None:
            refused.append(f"{reader.tool}: the reader source defines no function, so there is nothing to run")
            continue
        sigs.append(ToolSig(name=reader_names[reader.tool], kind="read",
                            args_fields=[FieldStat(name="result", types=["str"], optional=False)]))
        bodies[reader_names[reader.tool]] = body
    for name in sorted(proposal.derived_names()):
        spec = proposal.column(name)
        body = _method_body(spec.derive if spec is not None else "", "row")
        if body is None:
            refused.append(f"column {name}: it is declared derived and its derive source defines no function")
            continue
        sigs.append(ToolSig(name=derived_names[name], kind="read",
                            args_fields=[FieldStat(name="row", types=["dict"], optional=False)]))
        bodies[derived_names[name]] = body
    return module_source(EntitySchema(), sigs, bodies), refused


def _run(source: str, calls: list[ToolCall], workdir: Path) -> tuple[list[dict], str]:
    """One subprocess over these calls; a sandbox refusal comes back as a sentence, not an exception."""
    if not calls:
        return [], ""
    box = Sandbox(source, {}, Path(workdir), timeout=SANDBOX_TIMEOUT)
    try:
        return box.run(calls), ""
    except SandboxError as exc:
        return [], f"the readers module did not run: {exc}"


# --- the gate ---------------------------------------------------------------

@dataclass
class Ruling:
    """What the gate made of one proposal: the failures, and the values it parsed on the way."""
    failures: list[str] = field(default_factory=list)
    failing_shapes: int = 0
    parsed: dict = field(default_factory=dict)  # (tool, result text) -> dict or None
    ran: bool = False


def _clip(value: Any) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."


def _texts_by_tool(calls_by_tool: dict[str, list[ToolCall]]) -> dict[str, list[str]]:
    """The distinct result texts per tool: a reader is a function of the string, so one run answers all."""
    out: dict[str, list[str]] = {}
    for tool, calls in calls_by_tool.items():
        seen: dict[str, None] = {}
        for call in calls:
            seen.setdefault(str(parse_result(call.result)), None)
        out[tool] = list(seen)
    return out


def gate_proposal(proposal: Proposal, calls_by_tool: dict[str, list[ToolCall]],
                  traces: Iterable[Trace], workdir: Path | str) -> Ruling:
    """Run every reader over every recorded result of its tool and rule on what came back.

    Under both gates a reader has to parse, run and answer with a dict or None. Under `declared` the
    three declarations are held to the recording as well: every result parses to a non-empty dict of
    declared columns or is a declared acknowledgement, a derived column's parsed value equals its
    derivation over the stored columns as walked to that point in the same trace, and a tool that
    only acknowledges is seen to change each column it declares.

    Failures are named per masked shape, never per result: a shape carried by four hundred calls is
    one line for the model to answer, and the count of failing shapes is how one attempt is ranked
    against another.
    """
    workdir = Path(workdir)
    traces = list(traces)
    ruling = Ruling()
    source, refused = _module(proposal)
    ruling.failures += refused
    for check in (gate_parses(source), gate_confined(source)):
        if not check.passed:
            ruling.failures += [f"the readers module {check.stage}: {line}" for line in check.failures]
    ruling.failures += _naming_failures(proposal)
    missing = sorted(set(calls_by_tool) - {r.tool for r in proposal.readers})
    ruling.failures += [f"{tool}: no reader was proposed for a tool that has recorded results" for tool in missing]
    if ruling.failures:
        ruling.failing_shapes = _shape_count(calls_by_tool)
        return ruling

    reader_names, derived_names = _method_names(proposal)
    texts = _texts_by_tool(calls_by_tool)
    jobs = [(tool, text) for tool in sorted(texts) for text in texts[tool]]
    calls = [ToolCall(name=reader_names[tool], args={"result": text}, raw_ptr=_PTR) for tool, text in jobs]
    results, refusal = _run(source, calls, workdir / "readers")
    if refusal:
        ruling.failures.append(refusal)
        ruling.failing_shapes = _shape_count(calls_by_tool)
        return ruling
    ruling.ran = True
    failed: dict[tuple[str, str], str] = {}
    for (tool, text), result in zip(jobs, results, strict=False):
        line = _read_failure(proposal, tool, text, result)
        if line is not None:
            failed.setdefault((tool, mask(text)), line)
        else:
            ruling.parsed[(tool, text)] = result.get("value")
    if proposal.gate == GATE_DECLARED:
        _derivation_failures(proposal, traces, ruling.parsed, derived_names, source, workdir, failed)
        ruling.failures += _effect_failures(proposal, traces, ruling.parsed, calls_by_tool)
    ruling.failing_shapes = len(failed) + len(ruling.failures)
    ruling.failures = sorted(failed.values()) + ruling.failures
    return ruling


def _shape_count(calls_by_tool: dict[str, list[ToolCall]]) -> int:
    """Every masked shape in the corpus: what a proposal that never ran is ranked as having failed."""
    return sum(len(shapes_of(calls)) for calls in calls_by_tool.values())


def _naming_failures(proposal: Proposal) -> list[str]:
    """The names code owns: the table's key is the requestor, so no column of it may be an id."""
    table = proposal.table
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    reserved = {"id", f"{table}_id", f"{singular}_id"}
    return [f"column {name}: the table holds one row per requestor and code owns its key, so it may "
            "not hold a column that reads as an id; name the value it holds instead"
            for name in sorted({c.name for c in proposal.columns} & reserved)]


def _read_failure(proposal: Proposal, tool: str, text: str, result: dict) -> Optional[str]:
    """What is wrong with one reader's answer to one recorded result, or None when nothing is."""
    shape = mask(text)
    label = f"{tool} shape {shape!r}"
    if not result.get("ok", False):
        return f"{label}: the reader raised {result.get('error')}: {result.get('message')}"
    value = result.get("value")
    if value is not None and not isinstance(value, dict):
        return (f"{label}: the reader answered a {type(value).__name__}; a reader answers a dict of "
                "column values or None")
    if proposal.gate != GATE_DECLARED:
        # The `replay` gate asks nothing beyond this: a key nobody proposed becomes a column of the
        # table (`absorb_columns`), and whether the column was worth having is replay's to say.
        return None
    names = {c.name for c in proposal.columns}
    if isinstance(value, dict):
        unknown = sorted(set(value) - names)
        if unknown:
            return (f"{label}: the reader answered the key(s) {', '.join(unknown)}, which are not "
                    "columns of the table you proposed; propose the column or drop the key")
    reader = proposal.reader_for(tool)
    acknowledged = shape in set(reader.acknowledgements if reader is not None else [])
    if value is None or not value:
        if acknowledged:
            return None
        return (f"{label}: the reader asserted nothing and this shape is not declared an "
                "acknowledgement; either read a column value out of it or declare the shape")
    if acknowledged:
        return (f"{label}: the shape is declared an acknowledgement and the reader answered "
                f"{_clip(value)}; a result either asserts column values or it does not")
    return None


def _walk(trace: Trace, proposal: Proposal, parsed: dict) -> list[tuple[str, str, Optional[dict]]]:
    """This requestor's prose calls in the order the recording made them, with what the reader read."""
    out = []
    for call in trace.tool_calls:
        if (call.requestor or ASSISTANT) != proposal.requestor or not is_prose(call):
            continue
        text = str(parse_result(call.result))
        key = (call.name, text)
        if key not in parsed:
            continue
        out.append((call.name, text, parsed[key]))
    return out


def starting_row(trace: Trace, proposal: Proposal, parsed: dict) -> dict:
    """The row this recording started in: per stored column, the first value read before it was written.

    A column a tool declares it changes is closed at that tool's call, so a value read only after
    the write is not the starting value; nothing else closes a column, so the first reading of it
    stands. Under the `replay` gate no tool declares anything, and the first reading always stands.
    """
    row: dict = {}
    closed: set[str] = set()
    stored = set(proposal.stored_names())
    for tool, _text, values in _walk(trace, proposal, parsed):
        closed |= set(proposal.changes_of(tool))
        for name, value in (values or {}).items():
            if name in stored and name not in row and name not in closed:
                row[name] = value
    return row


def _derivation_failures(proposal: Proposal, traces: list[Trace], parsed: dict,
                         derived_names: dict[str, str], source: str, workdir: Path,
                         failed: dict[tuple[str, str], str]) -> None:
    """A derived column's parsed value against its derivation over the stored columns walked so far.

    The row handed to the derivation is the world as this recording had shown it up to and including
    the call being checked, so a status a tool reports about its own write is checked against the
    write it just made. A derivation that answers None could not tell from that row and is not
    checked there, which is what keeps a partially observed row from failing a correct derivation.
    """
    derived = set(proposal.derived_names())
    if not derived:
        return
    checks: list[tuple[str, str, str, dict, Any]] = []  # column, tool, text, row, parsed value
    stored = set(proposal.stored_names())
    for trace in traces:
        row: dict = {}
        for tool, text, values in _walk(trace, proposal, parsed):
            for name, value in (values or {}).items():
                if name in stored:
                    row[name] = value
            for name, value in (values or {}).items():
                if name in derived:
                    checks.append((name, tool, text, dict(row), value))
    if not checks:
        return
    wanted: dict[tuple[str, str], dict] = {}
    for name, _tool, _text, row, _value in checks:
        wanted.setdefault((name, content_hash(row)), row)
    order = sorted(wanted)
    calls = [ToolCall(name=derived_names[name], args={"row": wanted[(name, key)]}, raw_ptr=_PTR)
             for name, key in order]
    results, refusal = _run(source, calls, workdir / "derivations")
    if refusal:
        failed[("derivations", "")] = refusal
        return
    answers = {key: result for key, result in zip(order, results, strict=False)}
    for name, tool, text, row, value in checks:
        result = answers.get((name, content_hash(row)))
        if result is None:
            continue
        label = (tool, mask(text))
        if not result.get("ok", False):
            failed.setdefault(label, f"{tool} shape {mask(text)!r}: derive_{name} raised "
                                     f"{result.get('error')}: {result.get('message')}")
            continue
        computed = result.get("value")
        if computed is None or canon(computed) == canon(value):
            continue
        failed.setdefault(label, (
            f"{tool} shape {mask(text)!r}: the reader read {name} as {_clip(value)} and derive_{name} "
            f"computed {_clip(computed)} from the stored columns as this recording had shown them, "
            f"{_clip(row)}; either the column is stored, or the derivation or the reader is wrong"))


def _effect_failures(proposal: Proposal, traces: list[Trace], parsed: dict,
                     calls_by_tool: dict[str, list[ToolCall]]) -> list[str]:
    """A tool whose every shape acknowledges has to name a column it changes, and be seen changing it."""
    out: list[str] = []
    for reader in sorted(proposal.readers, key=lambda r: r.tool):
        calls = calls_by_tool.get(reader.tool) or []
        if not calls:
            continue
        shapes = {s.shape for s in shapes_of(calls)}
        if not shapes or not shapes <= set(reader.acknowledgements):
            continue
        if not reader.changes:
            out.append(f"{reader.tool}: every shape of it is declared an acknowledgement, so nothing "
                       "reads the world out of it; name in changes the columns its call changes")
            continue
        for name in sorted(set(reader.changes)):
            if not _change_seen(proposal, traces, parsed, reader.tool, name):
                out.append(f"{reader.tool}: it declares that it changes {name}, and in no recording "
                           f"does a reading of {name} before one of its calls differ from the first "
                           "reading after it; declare the column it does change, or drop the claim")
    return out


def _change_seen(proposal: Proposal, traces: list[Trace], parsed: dict, tool: str, column: str) -> bool:
    """Whether any recording reads this column differently before and after a call of this tool."""
    for trace in traces:
        steps = _walk(trace, proposal, parsed)
        for index, (name, _text, _values) in enumerate(steps):
            if name != tool:
                continue
            before = next((v[column] for _n, _t, v in reversed(steps[:index]) if v and column in v), None)
            after = next((v[column] for _n, _t, v in steps[index + 1:] if v and column in v), None)
            if before is not None and after is not None and canon(before) != canon(after):
                return True
    return False


def absorb_columns(proposal: Proposal, parsed: dict) -> Proposal:
    """Make every key the readers actually answered a column of the table.

    Under the `replay` gate a key nobody proposed is not a failure, so the world would otherwise
    hold a value under no column and the export would drop it. Under `declared` the gate already
    refused such a key, so this adds nothing there.
    """
    known = {c.name for c in proposal.columns}
    extra = sorted({name for value in parsed.values() if isinstance(value, dict) for name in value} - known)
    proposal.columns = proposal.columns + [ColumnSpec(name=name) for name in extra]
    return proposal


def column_values(proposal: Proposal, parsed: dict) -> dict[str, list]:
    """Per column, the values the readers actually read, for the column class rule (D73)."""
    out: dict[str, list] = {}
    for value in parsed.values():
        for name, item in (value or {}).items() if isinstance(value, dict) else ():
            out.setdefault(name, []).append(item)
    return out


# --- the stage's own loop ---------------------------------------------------

def propose(model: Any, requestor: str, calls_by_tool: dict[str, list[ToolCall]], traces: Iterable[Trace],
            workdir: Path | str, *, gate: str = GATE_DECLARED,
            max_attempts: int = MAX_ATTEMPTS) -> tuple[Proposal, list[dict], dict]:
    """Ask for one requestor's proposal, gate it, and retry with the failures at most `max_attempts` times.

    The prompt is two messages: a system prefix that is the same bytes for every attempt and every
    requestor of a build, and a user turn carrying this requestor's evidence. A retry keeps both
    exactly as they were sent and appends the previous reply and the failures as new turns, so the
    prefix a provider caches never moves, the way compile_tool's repair loop does it (D75).

    Nothing is cached here: the model handed in is already memoized on the bytes of its request
    (`build._wrap`), and those bytes carry every recorded result this proposal saw, so a rebuild
    over the same results never reaches the network.

    After the last attempt the proposal with the fewest failing shapes is kept, marked assisted, and
    its reasons are recorded, the way a kept assisted body is.
    """
    if gate not in GATES:
        raise ValueError(f"unknown readers gate {gate!r}; it is one of {', '.join(GATES)}")
    traces, workdir = list(traces), Path(workdir)
    messages = [{"role": "system", "content": system_prompt(gate)},
                {"role": "user", "content": json.dumps(evidence_payload(requestor, calls_by_tool, traces),
                                                       default=str, sort_keys=True)}]
    nodes: list[dict] = []
    best: Optional[Proposal] = None
    best_failing = None
    best_parsed: dict = {}
    for attempt in range(max_attempts):
        reply = model.query(messages)
        content = getattr(reply, "content", None) or ""
        proposal = _parse_reply(reply, requestor, gate)
        if proposal is None:
            nodes.append({"attempt": attempt, "requestor": requestor, "passed": False,
                          "failures": ["the reply carried no usable proposal object"]})
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "That reply carried no usable JSON object. Answer with the "
                                            "object described above and nothing else."}]
            continue
        proposal.attempts = attempt + 1
        ruling = gate_proposal(proposal, calls_by_tool, traces, workdir)
        nodes.append({"attempt": attempt, "requestor": requestor, "passed": not ruling.failures,
                      "failing_shapes": ruling.failing_shapes, "columns": len(proposal.columns),
                      "readers": len(proposal.readers), "failures": ruling.failures[:MAX_FEEDBACK_LINES]})
        if best_failing is None or ruling.failing_shapes < best_failing:
            best, best_failing, best_parsed = proposal, ruling.failing_shapes, ruling.parsed
            best.failures = ruling.failures[:MAX_FEEDBACK_LINES]
        if not ruling.failures:
            proposal.assisted = False
            return absorb_columns(proposal, ruling.parsed), nodes, ruling.parsed
        messages = messages + [{"role": "assistant", "content": content},
                               {"role": "user", "content": _retry_turn(ruling)}]
    kept = best if best is not None else Proposal(requestor=requestor, table="", gate=gate,
                                                  attempts=max_attempts,
                                                  failures=["no attempt produced a usable proposal"])
    kept.assisted = True
    return absorb_columns(kept, best_parsed), nodes, best_parsed


def _retry_turn(ruling: Ruling) -> str:
    lines = ruling.failures[:MAX_FEEDBACK_LINES]
    more = len(ruling.failures) - len(lines)
    text = "The proposal failed on these shapes:\n" + "\n".join(f"- {line}" for line in lines)
    if more > 0:
        text += f"\n- and {more} more of the same kinds"
    return text + "\n\nAnswer with the whole object again, corrected."


# --- what the proposal leaves for the rest of the build ---------------------

def apply_to_schema(schema: EntitySchema, proposals: Iterable[Proposal],
                    values: Optional[dict] = None) -> EntitySchema:
    """Add each proposal's table and its stored columns to the mined schema, marked by who revealed them.

    The mark rides on `Column.evidence` under `revealed_by`, which is the least the schema record
    has to change to carry it: a column of a table another requestor revealed is not a column of the
    customer's system, and the export says so rather than passing it off as one. A derived column is
    not a column of the world at all: it is computed, so nothing stores it, and the body that has to
    answer with it is given the derivation instead (`body_note`).
    """
    values = values or {}
    for proposal in proposals:
        if not proposal.table:
            continue
        seen = {(c.table, c.name) for c in schema.columns}
        if proposal.table not in schema.tables:
            schema.tables = sorted([*schema.tables, proposal.table])
        for name in proposal.stored_names():
            if (proposal.table, name) in seen:
                continue
            observed = list((values.get(proposal.requestor) or {}).get(name) or [])
            ruling = propose_column_class(proposal.table, name, observed)
            schema.columns.append(Column(
                table=proposal.table, name=name, class_=ruling.column_class,
                class_rule=ruling.column_class, class_confidence=ruling.confidence,
                class_reason=ruling.reason, classified_by="rule",
                evidence={**ruling.evidence, "revealed_by": proposal.requestor},
                samples=observed[:MAX_EXAMPLES]))
    schema.columns = sorted(schema.columns, key=lambda c: (c.table, c.name))
    return schema


def revealed_tables(schema: EntitySchema) -> dict[str, str]:
    """Per table, the requestor that revealed it; a table the assistant's calls describe is not here."""
    out: dict[str, str] = {}
    for column in schema.columns:
        who = (column.evidence or {}).get("revealed_by")
        if who:
            out[column.table] = str(who)
    return out


def environment_flags(schema: EntitySchema) -> list[str]:
    """One flag per revealed table, so the setup review sees what is not the customer's own system."""
    return [f"{table}: revealed by the {who}'s own tools and not by the assistant's, so it is marked "
            "in the export and is not part of the customer's system (R33)"
            for table, who in sorted(revealed_tables(schema).items())]


def body_note(proposals: Iterable[Proposal]) -> str:
    """What a body writer has to know about a revealed table: how to reach its row, and the derivations.

    It sits in the stable system prefix beside the tables block, so it is the same bytes for every
    tool of a build. The row is reached without naming its key, because the key is the requestor's
    name and a literal id in a body is refused by the memorised_values gate (D162).
    """
    parts: list[str] = []
    for proposal in proposals:
        if not proposal.table:
            continue
        lines = [f"Table {proposal.table} holds what the {proposal.requestor}'s own tools read and "
                 f"write, not what the assistant's do. It holds exactly one row: read it with "
                 f"next(iter(self.db.{proposal.table}.values())), never by key.",
                 "Its stored columns are: " + (", ".join(proposal.stored_names()) or "(none)") + "."]
        for name in proposal.derived_names():
            spec = proposal.column(name)
            lines.append(f"{name} is not stored: it follows from the stored columns of that row, by "
                         f"this function, which a body that has to answer with it computes for itself:\n"
                         f"{(spec.derive if spec is not None else '').strip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def starting_rows(traces: Iterable[Trace], proposal: Proposal, parsed: dict) -> dict[str, dict]:
    """Per trace, the row this proposal's requestor started that recording in."""
    return {trace.trace_id: starting_row(trace, proposal, parsed) for trace in traces}


def merge_worlds(worlds: dict[str, dict], artifact: Any) -> dict[str, dict]:
    """Add the revealed row's pre-write columns to `compile_env.trace_worlds`, one key per column.

    A row assembled column by column out of many calls is not seen whole the way a returned row is,
    so two recordings contradict each other only on a column they both read and read differently
    (D74). Keyed per column, that is what `cluster.split_by_world` compares.
    """
    for proposal in proposals_from(artifact):
        rows = (rows_from(artifact).get(proposal.requestor) or {})
        for trace_id, row in rows.items():
            for name, value in (row or {}).items():
                worlds.setdefault(trace_id, {})[(proposal.table, proposal.requestor, name)] = content_hash(value)
    return worlds


def reader_rows(artifact: Any) -> dict[tuple[str, str], dict[str, dict]]:
    """The revealed rows as `build_starting_state` takes them: (table, row id) to trace to row."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    rows = rows_from(artifact)
    for proposal in proposals_from(artifact):
        by_trace = {trace_id: row for trace_id, row in (rows.get(proposal.requestor) or {}).items() if row}
        if by_trace:
            out[(proposal.table, proposal.requestor)] = by_trace
    return out
