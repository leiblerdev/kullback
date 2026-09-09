"""Readers: the columns a requestor's prose results describe, proposed as code and gated on the recording.

A customer's recording may carry a requestor beside the assistant, running a toolkit of its own
(D164). Where that toolkit answers with prose rather than with rows, `compile_env.extract_rows`
reads nothing from it and R33 keeps it out of the Starting state, so the world holds no row for the
thing those tools read and write, and every body written for them reads state the world does not
have.

This module builds that row. Per requestor with prose results, the model is shown the distinct
result shapes of each of its tools, with the digits masked, and proposes one table, its columns,
and per tool two functions: a reader, `def read(result)` mapping one result string to the column
values it asserts, or None for a result that asserts nothing; and a render, `def render(row)`
writing the result string that tool answers from a row, the inverse of the reader. A gate runs
every reader over every recorded result of its tool, and then asks every render for each of those
results again out of the row the recording had when it was answered. What failed comes back as the
next attempt's evidence, one line per masked shape. Both run in the same subprocess sandbox as a
tool body (`builder/sandbox.py`), never in this process.

The gate asks of a reader only that it parses, runs on every recorded result without raising, and
answers with a dict or None; a column named as an id is refused, because code owns the row's key.
A shape whose reader reads nothing out of it is counted per tool and reported in the stage ruling,
never refused: the experiment of 2026-09-07 measured the refusing version of that rule and it
bought one extra attempt and no fidelity.

The render is where the recording rules on the proposal, and it is the part a free reader gate
cannot ask for. A result whose last line is composed out of several other columns was read into
columns and then written back by the compiled bodies with a word the row never held, on most of
the recorded calls of a dozen tools. The round trip is the check: `render` of the row a result was
answered from has to be that result again, character for character. A value the sentence states
that no column holds follows from the columns that do, and working it out is the render's own
business, so no column is invented for it. A tool whose result asserts nothing renders the constant
it answers. What the render writes is then what the body writes: its source is in the body writer's
prompt for that tool.

What the requestor writes is mined from the recording rather than declared, the way
`mine.observed_effects` credits the assistant's writes: a column a write tool of that requestor is
seen to change closes at its calls, so a value read only after a write is not that recording's
starting value. A column no recording reads before its first write is filled with the commonest
pre-write value the corpus shows for it, and each fill is recorded as an assumption of the Starting
state; a column no recording reads before any write is left unset and named in the ruling.

Nothing here names a domain, a tool or a value: what is domain is read off the corpus, and what is
code is the shape of a row and the shape of a function over a string.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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
NO_PROSE = "no prose results"
ASSISTANT = "assistant"

MAX_ATTEMPTS = 4
MAX_EXAMPLES = 3  # verbatim results shown per masked shape
MAX_SEQUENCES = 3  # recordings whose call order is shown, so an effect between tools can be seen
MAX_SEQUENCE_CALLS = 24
MAX_FEEDBACK_LINES = 40  # failures carried into the next attempt, one line per masked shape
MAX_VALUE_CHARS = 120  # a value quoted back inside a failure line
SANDBOX_TIMEOUT = 300.0
CREDIT_ODDS = 3.0  # how much likelier a column's change is when a tool was called, for a credit
CREDIT_CHANGES = 3  # changing intervals the tool was called in, at least
CREDIT_ABSENT = 3  # intervals it was not called in, at least, so there is a contrast to read
CREDIT_SHARE = 0.5  # the column's changes it has to be in, so a credit explains them

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
class ToolReader:
    """One tool's reader: the source of a function over one result string."""
    tool: str
    source: str


@dataclass
class ToolRender:
    """One tool's render: the source of a function over the row, the inverse of the reader."""
    tool: str
    source: str


@dataclass
class Proposal:
    """One requestor's world as the model proposed it, and how the gate ruled on it.

    `effects` is not the model's: it is mined from the recording after the gate has run, per tool of
    this requestor, as the columns the corpus credits its calls with changing (`write_effects`), and
    those credits are what say which of these tools is a write (`kinds_for`).
    """
    requestor: str
    table: str
    columns: list[str] = field(default_factory=list)
    readers: list[ToolReader] = field(default_factory=list)
    renders: list[ToolRender] = field(default_factory=list)
    effects: dict[str, list[str]] = field(default_factory=dict)
    attempts: int = 0
    failures: list[str] = field(default_factory=list)
    silent: dict[str, int] = field(default_factory=dict)  # per tool, shapes the reader reads nothing from
    round_trips: tuple[int, int] = (0, 0)  # recorded results the renders wrote back, of those asked
    assisted: bool = False

    def reader_for(self, tool: str) -> Optional[ToolReader]:
        return next((r for r in self.readers if r.tool == tool), None)

    def render_for(self, tool: str) -> Optional[ToolRender]:
        return next((r for r in self.renders if r.tool == tool), None)

    def changes_of(self, tool: str) -> list[str]:
        return list(self.effects.get(tool) or [])

    def to_dict(self) -> dict:
        return {
            "requestor": self.requestor, "table": self.table, "attempts": self.attempts,
            "failures": list(self.failures), "assisted": self.assisted, "columns": list(self.columns),
            "silent": dict(self.silent), "round_trips": list(self.round_trips),
            "effects": {k: list(v) for k, v in sorted(self.effects.items())},
            "readers": [{"tool": r.tool, "source": r.source} for r in self.readers],
            "renders": [{"tool": r.tool, "source": r.source} for r in self.renders],
        }

    @classmethod
    def from_dict(cls, body: dict) -> "Proposal":
        return cls(
            requestor=str(body.get("requestor") or ""), table=str(body.get("table") or ""),
            attempts=int(body.get("attempts") or 0), assisted=bool(body.get("assisted")),
            failures=[str(f) for f in body.get("failures") or []],
            silent={str(k): int(v) for k, v in (body.get("silent") or {}).items()},
            round_trips=tuple(int(n) for n in (body.get("round_trips") or [0, 0]))[:2] or (0, 0),
            effects={str(k): [str(c) for c in v] for k, v in (body.get("effects") or {}).items()},
            columns=[str(c) for c in body.get("columns") or []],
            readers=[ToolReader(tool=str(r.get("tool") or ""), source=str(r.get("source") or ""))
                     for r in body.get("readers") or []],
            renders=[ToolRender(tool=str(r.get("tool") or ""), source=str(r.get("source") or ""))
                     for r in body.get("renders") or []],
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
    "Per tool you also propose a render, `def render(row):` returning the result string that tool "
    "answers when the row stands as it is given. It is the inverse of the reader: for every "
    "recorded result of the tool, render of the row that result was answered from has to be that "
    "result again, character for character, and that round trip is what the proposal is held to. "
    "The row it is given holds every column you proposed, the ones this tool reads and the ones it "
    "does not. A result that asserts nothing, an acknowledgement the tool answers whatever the row "
    "holds, is rendered as that constant string. A value the sentence states that no column of the "
    "row holds, because it follows from two or more columns that do, is the render's own work: "
    "compute it there from the columns it follows from, and do not add a column for it.\n\n"
    "Name the table and the columns after what they hold, in the recording's own words, lowercase "
    "with underscores. Do not propose a column that is an id: the table holds one row per "
    "requestor and code owns its key, so a column named id, or the table's name with _id after "
    "it, is refused.\n"
)

_REPLY_SHAPE = (
    "Answer with one JSON object and nothing else:\n"
    '{"table": "<table name>",\n'
    ' "columns": [{"name": "<column>"}, ...],\n'
    ' "readers": [{"tool": "<tool name>", "source": "def read(result):\\n    ..."}, ...],\n'
    ' "renders": [{"tool": "<tool name>", "source": "def render(row):\\n    ..."}, ...]}\n'
    "One reader and one render per tool you were shown, and neither for a tool you were not.\n"
)

def _code_rules() -> str:
    """The sandbox's own rules, generated from the gate's constants so the two cannot drift."""
    return ("The reader and the render are checked before they run and are refused if they "
            "name anything outside their own argument. They may not use: "
            + ", ".join(sorted(DENIED_BUILTINS)) + ". They may not touch a dunder attribute. They "
            "may import only: " + ", ".join(sorted(ALLOWED_IMPORTS)) + ". Write the import inside "
            "the function, and do not read or write any file, clock or network.")


def _feedback_shape() -> str:
    return ("If a proposal does not hold, you are given it back with one line per shape that "
            "failed, naming the tool, the masked shape and what went wrong. Answer with the whole "
            "object again, corrected. Stop when every line is answered: send the object, not a "
            "commentary on it.")


def system_prompt() -> str:
    """The stable prefix of every call of this stage: the same bytes for every attempt and requestor."""
    return "\n".join([_SYSTEM, _REPLY_SHAPE, _code_rules(), _feedback_shape()])


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


def _parse_reply(reply: Any, requestor: str) -> Optional[Proposal]:
    """The model's JSON as a Proposal; None when the reply carries no usable object."""
    body = _reply_json(reply)
    table = str(body.get("table") or "").strip()
    columns = body.get("columns")
    readers = body.get("readers")
    if not table or not isinstance(columns, list) or not isinstance(readers, list):
        return None
    names = [str(c.get("name") or "").strip() if isinstance(c, dict) else str(c).strip() for c in columns]
    names = [name for name in names if name]
    tools = [ToolReader(tool=str(r.get("tool") or "").strip(), source=str(r.get("source") or ""))
             for r in readers if isinstance(r, dict) and str(r.get("tool") or "").strip()]
    shown = [ToolRender(tool=str(r.get("tool") or "").strip(), source=str(r.get("source") or ""))
             for r in (body.get("renders") or []) if isinstance(r, dict) and str(r.get("tool") or "").strip()]
    if not names or not tools:
        return None
    return Proposal(requestor=requestor, table=table, columns=names, readers=tools, renders=shown)


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


def _method_names(proposal: Proposal) -> dict[str, str]:
    """Stable method names for the readers: position, never a customer's word."""
    return {reader.tool: f"reader_{index}" for index, reader in enumerate(sorted(
        proposal.readers, key=lambda r: r.tool))}


def _render_names(proposal: Proposal) -> dict[str, str]:
    """Stable method names for the renders, on the same rule as the readers'."""
    return {render.tool: f"render_{index}" for index, render in enumerate(sorted(
        proposal.renders, key=lambda r: r.tool))}


def _module(proposal: Proposal) -> tuple[str, list[str]]:
    """The one module the sandbox runs: a method per reader and per render.

    Both go in one module so both are checked by the one confinement gate and run in the one
    subprocess; nothing here is ever executed in this process. Refusals come back as sentences.
    """
    from kullback.builder.compile_env import module_source  # builder to builder, at call time
    from kullback.runner.records import FieldStat, ToolSig

    reader_names, render_names = _method_names(proposal), _render_names(proposal)
    sigs, bodies, refused = [], {}, []
    for reader in sorted(proposal.readers, key=lambda r: r.tool):
        body = _method_body(reader.source, "result")
        if body is None:
            refused.append(f"{reader.tool}: the reader source defines no function, so there is nothing to run")
            continue
        sigs.append(ToolSig(name=reader_names[reader.tool], kind="read",
                            args_fields=[FieldStat(name="result", types=["str"], optional=False)]))
        bodies[reader_names[reader.tool]] = body
    for render in sorted(proposal.renders, key=lambda r: r.tool):
        body = _method_body(render.source, "row")
        if body is None:
            refused.append(f"{render.tool}: the render source defines no function, so there is nothing to run")
            continue
        sigs.append(ToolSig(name=render_names[render.tool], kind="read",
                            args_fields=[FieldStat(name="row", types=["dict"], optional=False)]))
        bodies[render_names[render.tool]] = body
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
    """What the gate made of one proposal: the failures, what it read, and what it read nothing from."""
    failures: list[str] = field(default_factory=list)
    failing_shapes: int = 0
    parsed: dict = field(default_factory=dict)  # (tool, result text) -> dict or None
    silent: dict[str, int] = field(default_factory=dict)  # per tool, shapes the reader read nothing from
    ran: bool = False
    round_trips: int = 0  # recorded results the render was asked to reproduce
    round_trips_matched: int = 0
    render_shapes_failing: int = 0


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
                  workdir: Path | str, traces: Optional[Iterable[Trace]] = None) -> Ruling:
    """Run every reader over every recorded result of its tool, then hold the renders to the round trip.

    A reader has to parse, run and answer with a dict or None, and no column may read as an id.
    Nothing else is asked of a reader: a shape a reader reads nothing out of is counted per tool and
    reported (`Ruling.silent`), because the arm that refused it, measured on a customer corpus on
    2026-09-07, bought one extra attempt and no fidelity.

    The render is where the recording rules on the proposal. For every recorded result of a tool,
    `render` of the row that result was answered from has to be that result again, character for
    character (`round_trip_failures`). That is what a free reader gate cannot ask and what the
    bodies kept getting wrong: a result whose last line is composed out of several other columns was
    parsed into columns and then written back with a word the row never held. A row is not the
    reader's own dict but the row as the recording had it at that call, so a value the sentence
    states and no column holds has to be computed by the render out of the columns that do.

    Failures are named per masked shape, never per result: a shape carried by four hundred calls is
    one line for the model to answer, and the count of failing shapes is how one attempt is ranked
    against another.
    """
    workdir = Path(workdir)
    traces = list(traces or [])
    ruling = Ruling()
    source, refused = _module(proposal)
    ruling.failures += refused
    for check in (gate_parses(source), gate_confined(source)):
        if not check.passed:
            ruling.failures += [f"the readers module {check.stage}: {line}" for line in check.failures]
    ruling.failures += _naming_failures(proposal)
    ruling.failures += [f"{tool}: no reader was proposed for a tool that has recorded results"
                        for tool in sorted(set(calls_by_tool) - {r.tool for r in proposal.readers})]
    ruling.failures += [f"{tool}: no render was proposed for a tool that has recorded results, so "
                        "nothing can write its results back out of the row"
                        for tool in sorted(set(calls_by_tool) - {r.tool for r in proposal.renders})]
    if ruling.failures:
        ruling.failing_shapes = _shape_count(calls_by_tool)
        return ruling

    reader_names = _method_names(proposal)
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
    silent: dict[str, set] = {}
    for (tool, text), result in zip(jobs, results, strict=False):
        line = _read_failure(tool, text, result)
        if line is not None:
            failed.setdefault((tool, mask(text)), line)
            continue
        value = result.get("value")
        ruling.parsed[(tool, text)] = value
        if not value:
            silent.setdefault(tool, set()).add(mask(text))
    ruling.silent = {tool: len(shapes) for tool, shapes in sorted(silent.items())}
    if failed:  # a row read wrong is a row nothing can be rendered from, so the round trip waits
        ruling.failing_shapes = len(failed)
        ruling.failures = sorted(failed.values())
        return ruling
    round_trip_failures(proposal, traces, ruling, source, workdir)
    ruling.failing_shapes = len(failed) + ruling.render_shapes_failing
    return ruling


def round_trip_failures(proposal: Proposal, traces: list[Trace], ruling: Ruling,
                        source: str, workdir: Path) -> None:
    """Ask every render for the result the recording holds, from the row that result came out of.

    The rows come from `render_rows`, which walks each recording the way `starting_row` does and
    then lays each call's own readings over the row in the order the recording made them, so the
    row handed to a render is the row the Environment will hold at that point. Identical (row,
    result) pairs are asked once; two results the recording answered from one row are both asked,
    and one of them failing is the honest answer that no function of the row alone can write both.

    Nothing here refuses a proposal on its own: the failures are lines for the next attempt, and
    after the last attempt the proposal with the fewest failing shapes is kept and marked assisted,
    which is how a reader that could not be satisfied has always been recorded.
    """
    rows = render_rows(traces, proposal, ruling.parsed)
    if not rows:
        return
    render_names = _render_names(proposal)
    jobs: dict[tuple[str, str, str], tuple[str, dict, str]] = {}
    for tool, text, row in rows:
        if tool in render_names:
            jobs.setdefault((tool, canon_text(row), text), (tool, row, text))
    ordered = [jobs[key] for key in sorted(jobs)]
    calls = [ToolCall(name=render_names[tool], args={"row": row}, raw_ptr=_PTR)
             for tool, row, _text in ordered]
    results, refusal = _run(source, calls, workdir / "readers")
    if refusal:
        ruling.failures.append(refusal)
        ruling.render_shapes_failing = _shape_count_of(rows)
        return
    failed: dict[tuple[str, str], str] = {}
    for (tool, row, text), result in zip(ordered, results, strict=False):
        ruling.round_trips += 1
        line = _render_failure(tool, text, row, result)
        if line is None:
            ruling.round_trips_matched += 1
            continue
        failed.setdefault((tool, mask(text)), line)
    ruling.render_shapes_failing = len(failed)
    ruling.failures = sorted(failed.values())


def canon_text(value: Any) -> str:
    """One value as the text two of them are compared by; the row's own key in the round trip."""
    return json.dumps(canon(value), default=str, sort_keys=True)


def _shape_count_of(rows: Iterable[tuple[str, str, dict]]) -> int:
    """Every masked shape among these recorded results: what a render that never ran has failed."""
    return len({(tool, mask(text)) for tool, text, _row in rows})


def _render_failure(tool: str, text: str, row: dict, result: dict) -> Optional[str]:
    """What is wrong with one render's answer, or None when it wrote the recorded result back."""
    label = f"{tool} shape {mask(text)!r}"
    if not result.get("ok", False):
        return (f"{label}: the render raised {result.get('error')}: {result.get('message')} on the "
                f"row {_clip(row)}")
    value = result.get("value")
    if not isinstance(value, str):
        return (f"{label}: the render answered a {type(value).__name__}; a render answers the "
                "result string the tool would have answered from that row")
    if value != text:
        return (f"{label}: the round trip does not hold. From the row {_clip(row)} the render "
                f"answered {_clip(value)} where the recording answered {_clip(text)}")
    return None


def render_rows(traces: Iterable[Trace], proposal: Proposal,
                parsed: dict) -> list[tuple[str, str, dict]]:
    """Per recorded prose call: the tool, the result it answered, and the row it answered it from.

    The row is the recording's own starting row (`starting_row`) over the corpus fills the Starting
    state will use (`fills_for`), with each call's readings laid over it in the order the recording
    made them. So the row a render is asked about holds every column the proposal named, whether or
    not this tool reads it, which is what lets a render write a line composed out of columns its own
    tool never mentions.
    """
    traces = list(traces)
    read_rows = starting_rows(traces, proposal, parsed)
    fills, _sentences, _unset = fills_for(read_rows, proposal)
    out: list[tuple[str, str, dict]] = []
    for trace in traces:
        row = {**fills, **(read_rows.get(trace.trace_id) or {})}
        for tool, text, values in _walk(trace, proposal, parsed):
            row.update({name: value for name, value in (values or {}).items()})
            out.append((tool, text, dict(row)))
    return out


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
            for name in sorted(set(proposal.columns) & reserved)]


def _read_failure(tool: str, text: str, result: dict) -> Optional[str]:
    """What is wrong with one reader's answer to one recorded result, or None when nothing is."""
    label = f"{tool} shape {mask(text)!r}"
    if not result.get("ok", False):
        return f"{label}: the reader raised {result.get('error')}: {result.get('message')}"
    value = result.get("value")
    if value is not None and not isinstance(value, dict):
        return (f"{label}: the reader answered a {type(value).__name__}; a reader answers a dict of "
                "column values or None")
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


def _intervals(steps: list[tuple[str, str, Optional[dict]]], column: str):
    """Per reading interval of one column: whether it changed, and which tools were called in it.

    An interval runs from the last reading of the column before a stretch of calls to the first
    reading after it, the reading's own call included in the stretch, because a tool that answers
    the new value is a candidate for having set it.
    """
    reads = [(index, values[column]) for index, (_t, _x, values) in enumerate(steps)
             if values and column in values]
    for (start, before), (end, after) in zip(reads, reads[1:], strict=False):
        yield canon(before) != canon(after), {steps[k][0] for k in range(start + 1, end + 1)}


def write_effects(traces: Iterable[Trace], proposal: Proposal, parsed: dict) -> dict[str, list[str]]:
    """Per tool of this requestor, the columns the corpus credits its calls with changing.

    Credit is by association, not by which call happens to sit in the interval. A tool that
    restates the whole world is inside changing and unchanging intervals alike, so its presence
    says nothing; a tool that sets a column is inside changing ones and almost nowhere else. So per
    column and tool the four counts are taken over every reading interval of the corpus (tool
    called and the column changed, called and unchanged, absent and changed, absent and unchanged),
    and the tool is credited when calling it makes a change likelier by a clear margin and the
    column's changes are mostly its:

        changed share with the tool >= CREDIT_ODDS * changed share without it, and
        the changing intervals it was called in >= CREDIT_SHARE of all of them,

    on at least `CREDIT_CHANGES` changing intervals it was called in and `CREDIT_ABSENT` intervals
    it was not, because a tool the corpus never runs without offers no contrast to read. The share
    is what separates a cause from a passenger: a tool that rides along with a repair is in some of
    a column's changes, the tool that sets it is in most of them.

    Nothing here consults the miner's `kind`: this is what decides it for these tools (`kinds_for`).
    """
    walks = [_walk(trace, proposal, parsed) for trace in traces]
    counts: dict[tuple[str, str], list[int]] = {}
    totals: dict[str, list[int]] = {}
    for column in proposal.columns:
        for steps in walks:
            for changed, present in _intervals(steps, column):
                totals.setdefault(column, [0, 0])[0 if changed else 1] += 1
                for tool in present:
                    counts.setdefault((tool, column), [0, 0])[0 if changed else 1] += 1
    found: dict[str, set] = {}
    for (tool, column), (with_changed, with_same) in counts.items():
        changed, same = totals[column]
        without_changed, without_same = changed - with_changed, same - with_same
        without = without_changed + without_same
        if with_changed < CREDIT_CHANGES or without < CREDIT_ABSENT:
            continue
        if with_changed < CREDIT_SHARE * changed:
            continue
        share_with = with_changed / (with_changed + with_same)
        share_without = without_changed / without
        if share_with >= CREDIT_ODDS * share_without:
            found.setdefault(tool, set()).add(column)
    return {tool: sorted(columns) for tool, columns in sorted(found.items())}


def kinds_for(proposal: Proposal) -> dict[str, str]:
    """The kind of every tool this proposal reads: a write when the corpus credits it with a column.

    The miner rules on a tool by what its results are seen to do to the rebuilt world's rows
    (`mine.observed_effects`), and a tool whose result is prose touches no row, so its kind there is
    read off the shape of the string and the words in it. That is what put four device checks of one
    customer corpus in the write column on 2026-09-07: a check that restates the whole state looks
    like a tool that sets it. Here the row exists, so the kind follows the credits.
    """
    return {reader.tool: ("write" if proposal.changes_of(reader.tool) else "read")
            for reader in proposal.readers}


def apply_to_sigs(sigs: Iterable[Any], proposals: Iterable[Proposal]) -> list[Any]:
    """The mined sigs with the kind of every prose-result tool of a requestor set by its credits.

    Only those tools move, and only their `kind`, `kind_reason` and how it was classified; every
    other sig is handed on as it came.
    """
    kinds: dict[str, str] = {}
    for proposal in proposals:
        kinds.update(kinds_for(proposal))
    out = []
    for sig in sigs:
        want = kinds.get(getattr(sig, "name", ""))
        if want is None or getattr(sig, "kind", "") == want:
            out.append(sig)
            continue
        out.append(sig.model_copy(update={
            "kind": want, "kind_confidence": "medium", "classified_by": "observed",
            "kind_reason": ("the results of this tool are prose, and over the recorded readings of "
                            "the row it reveals its calls are " +
                            ("credited with changing " if want == "write" else "not credited with changing ") +
                            "a column of that row")}))
    return out


def starting_row(trace: Trace, proposal: Proposal, parsed: dict) -> dict:
    """The row this recording started in: per column, the first value read before it was written.

    A column the proposal's mined effects credit to a tool is closed at that tool's calls, so a
    value read only after the write, the write's own result included, is not the starting value.
    Nothing else closes a column, so the first reading of it stands.
    """
    row: dict = {}
    closed: set = set()
    for tool, _text, values in _walk(trace, proposal, parsed):
        closed |= set(proposal.changes_of(tool))
        for name, value in (values or {}).items():
            if name in proposal.columns and name not in row and name not in closed:
                row[name] = value
    return row


def absorb_columns(proposal: Proposal, parsed: dict) -> Proposal:
    """Make every key the readers actually answered a column of the table.

    A key nobody proposed is not a failure, so without this the world would hold a value under no
    column and the export would drop it.
    """
    known = set(proposal.columns)
    extra = sorted({name for value in parsed.values() if isinstance(value, dict) for name in value} - known)
    proposal.columns = list(proposal.columns) + extra
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
            workdir: Path | str, *, max_attempts: int = MAX_ATTEMPTS) -> tuple[Proposal, list[dict], dict]:
    """Ask for one requestor's proposal, gate it, and retry with the failures at most `max_attempts` times.

    The prompt is two messages: a system prefix that is the same bytes for every attempt and every
    requestor of a build, and a user turn carrying this requestor's evidence. A retry keeps both
    exactly as they were sent and appends the previous reply and the failures as new turns, so the
    prefix a provider caches never moves, the way compile_tool's repair loop does it (D75).

    Nothing is cached here: the model handed in is already memoized on the bytes of its request
    (`build._wrap`), and those bytes carry every recorded result this proposal saw, so a rebuild
    over the same results never reaches the network.

    After the last attempt the proposal with the fewest failing shapes is kept, marked assisted, and
    its reasons are recorded, the way a kept assisted body is. Whichever proposal is kept, the
    columns the corpus credits each of its tools with changing are mined before it is handed back.
    """
    traces, workdir = list(traces), Path(workdir)
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": json.dumps(evidence_payload(requestor, calls_by_tool, traces),
                                                       default=str, sort_keys=True)}]
    nodes: list[dict] = []
    best: Optional[Proposal] = None
    best_failing = None
    best_parsed: dict = {}
    best_silent: dict = {}
    best_trips: tuple[int, int] = (0, 0)

    def finish(proposal: Proposal, parsed: dict, silent: dict) -> tuple[Proposal, list[dict], dict]:
        proposal.silent = dict(silent)
        proposal = absorb_columns(proposal, parsed)
        proposal.effects = write_effects(traces, proposal, parsed)
        return proposal, nodes, parsed

    for attempt in range(max_attempts):
        reply = model.query(messages)
        content = getattr(reply, "content", None) or ""
        proposal = _parse_reply(reply, requestor)
        if proposal is None:
            nodes.append({"attempt": attempt, "requestor": requestor, "passed": False,
                          "failures": ["the reply carried no usable proposal object"]})
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "That reply carried no usable JSON object. Answer with the "
                                            "object described above and nothing else."}]
            continue
        proposal.attempts = attempt + 1
        ruling = gate_proposal(proposal, calls_by_tool, workdir, traces)
        nodes.append({"attempt": attempt, "requestor": requestor, "passed": not ruling.failures,
                      "failing_shapes": ruling.failing_shapes, "columns": len(proposal.columns),
                      "readers": len(proposal.readers), "renders": len(proposal.renders),
                      "silent_shapes": sum(ruling.silent.values()),
                      "round_trips": ruling.round_trips,
                      "round_trips_matched": ruling.round_trips_matched,
                      "render_shapes_failing": ruling.render_shapes_failing,
                      "failures": ruling.failures[:MAX_FEEDBACK_LINES]})
        if best_failing is None or ruling.failing_shapes < best_failing:
            best, best_failing = proposal, ruling.failing_shapes
            best_parsed, best_silent = ruling.parsed, ruling.silent
            best_trips = (ruling.round_trips_matched, ruling.round_trips)
            best.failures = ruling.failures[:MAX_FEEDBACK_LINES]
        if not ruling.failures:
            proposal.assisted = False
            proposal.round_trips = (ruling.round_trips_matched, ruling.round_trips)
            return finish(proposal, ruling.parsed, ruling.silent)
        messages = messages + [{"role": "assistant", "content": content},
                               {"role": "user", "content": _retry_turn(ruling)}]
    kept = best if best is not None else Proposal(requestor=requestor, table="", attempts=max_attempts,
                                                  failures=["no attempt produced a usable proposal"])
    kept.assisted = True
    kept.round_trips = best_trips
    return finish(kept, best_parsed, best_silent)


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
    customer's system, and the export says so rather than passing it off as one.
    """
    values = values or {}
    for proposal in proposals:
        if not proposal.table:
            continue
        seen = {(c.table, c.name) for c in schema.columns}
        if proposal.table not in schema.tables:
            schema.tables = sorted([*schema.tables, proposal.table])
        for name in proposal.columns:
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
    """What a body writer has to know about a revealed table: how to reach its row, what it holds,
    and the render that writes each of these tools' results back out of it.

    It sits in the stable system prefix beside the tables block, so it is the same bytes for every
    tool of a build. The row is reached without naming its key, because the key is the requestor's
    name and a literal id in a body is refused by the memorised_values gate (D162).

    The render's own source is here because it is the answer, checked against every recorded result
    of its tool by round trip. A body that writes the sentence again from the columns gets a word
    wrong on the values the recording never showed it; a body that renders the row the way the
    render does cannot.
    """
    parts: list[str] = []
    for proposal in proposals:
        if not proposal.table:
            continue
        lines = [f"Table {proposal.table} holds what the {proposal.requestor}'s own tools read and "
                 f"write, not what the assistant's do. It holds exactly one row: read it with "
                 f"next(iter(self.db.{proposal.table}.values())), never by key.",
                 "Its columns are: " + (", ".join(proposal.columns) or "(none)") + "."]
        changed = {tool: proposal.changes_of(tool) for tool in sorted(proposal.effects)}
        for tool, columns in changed.items():
            lines.append(f"The recording shows {tool} changing: {', '.join(columns)}.")
        for render in sorted(proposal.renders, key=lambda r: r.tool):
            lines.append(f"\nThe result of {render.tool} is this function of the row, which writes "
                         f"every recorded result of it back out of the row it came from. Read the "
                         f"row into a plain dict of its columns, apply this tool's own effect to "
                         f"that dict and to the row, and answer what this function answers:\n"
                         f"{render.source.rstrip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def starting_rows(traces: Iterable[Trace], proposal: Proposal, parsed: dict) -> dict[str, dict]:
    """Per trace, the row this proposal's requestor started that recording in."""
    return {trace.trace_id: starting_row(trace, proposal, parsed) for trace in traces}


def fills_for(rows: dict[str, dict], proposal: Proposal) -> tuple[dict, list[str], list[str]]:
    """Per column no recording read before its first write, the value the corpus shows commonest.

    A recording that never read a column before writing it starts, without this, on whatever the
    Environment's own default is, which is a value no recording ever showed. The corpus does hold
    one: the pre-write value of that column in the recordings that did read it. The commonest of
    those is taken, every recording that lacked the column is filled with it, and each fill is
    recorded as an assumption of the Starting state the way the state's own guesses are. A column no
    recording reads before any write has no such value, so it stays unset and is named.

    Returns the fills, the assumption sentences, and the columns left unset.
    """
    fills: dict = {}
    assumptions: list[str] = []
    unset: list[str] = []
    for name in proposal.columns:
        seen: list = [row[name] for row in rows.values() if name in row]
        missing = sum(1 for row in rows.values() if name not in row)
        if not missing:
            continue
        if not seen:
            unset.append(name)
            continue
        counted: dict = {}
        for value in seen:
            key = content_hash(value)
            entry = counted.setdefault(key, [0, value])
            entry[0] += 1
        best = sorted(counted.values(), key=lambda e: (-e[0], _clip(e[1])))[0]
        fills[name] = best[1]
        assumptions.append(
            f"{proposal.table} row {proposal.requestor} column {name} was not read before a write in "
            f"{missing} of {len(rows)} recordings; those start on {_clip(best[1])}, which {best[0]} "
            f"of the {len(seen)} recordings that did read it before writing showed")
    return fills, assumptions, unset


def fills_from(artifact: Any) -> dict:
    """Per requestor, the columns filled from the corpus and the value each was filled with."""
    body = artifact if isinstance(artifact, dict) else {}
    return dict(body.get("fills") or {})


def filled_columns(artifact: Any) -> set[tuple[str, str]]:
    """As (table, column): every column this stage filled from the corpus rather than read.

    A filled column is one no recording read before its first write, so the value on it is the
    corpus's commonest and not this Task's. The Starting-state pinner has to be able to tell such a
    column from one a recording actually read, because a read is evidence and a fill is a guess, and
    a guess is what D202 inverts a write against.
    """
    fills = fills_from(artifact)
    return {(proposal.table, str(name))
            for proposal in proposals_from(artifact)
            for name in (fills.get(proposal.requestor) or {})}


def reader_assumptions(artifact: Any) -> list[str]:
    """The sentences the fills leave for `assumptions.json`, in the order the stage wrote them."""
    body = artifact if isinstance(artifact, dict) else {}
    return [str(line) for line in body.get("assumptions") or []]


def reader_rows(artifact: Any) -> dict[tuple[str, str], dict[str, dict]]:
    """The revealed rows as `build_starting_state` takes them: (table, row id) to trace to row.

    The corpus fills are folded in here, under the columns the recording did not read for itself,
    so the world holds a value the corpus showed rather than the Environment's own default.
    """
    out: dict[tuple[str, str], dict[str, dict]] = {}
    rows = rows_from(artifact)
    fills = fills_from(artifact)
    for proposal in proposals_from(artifact):
        filled = dict(fills.get(proposal.requestor) or {})
        by_trace = {trace_id: {**filled, **row}
                    for trace_id, row in (rows.get(proposal.requestor) or {}).items() if row or filled}
        if by_trace:
            out[(proposal.table, proposal.requestor)] = by_trace
    return out


def result_reader(artifact: Any, traces: Iterable[Trace], workdir: Path | str
                  ) -> Optional[Callable[[str, Any], Optional[dict]]]:
    """What each tool's reader reads out of a result, as one function of (tool, result).

    The Starting-state pinner has a use for a reader beyond the requestor whose prose it was
    proposed for: a read that answers a bare value and leaves which row it is about to the call
    states its columns inside that value, and only this tool's reader can say what they are
    (`compile_env.home_partial_result`). Rather than hand the pinner a sandbox, every reader is run
    once here over the distinct results its tool answered anywhere in the corpus, and the answers
    are served from that table.

    Beside the proposed readers, the artifact may hold readers this build derived for itself from a
    tool's own recorded results, for a tool nobody proposed one for or for the shapes a proposed
    reader read nothing out of (`builder/templates.py`). A proposed reader answers first and a
    derived one fills what it left: the model's reading of a shape it was shown is the one to keep,
    and the derivation exists for the shapes no model was ever asked about.

    None when neither kind of reader exists, so a corpus without readers starts no subprocess. The
    readers run in the same sandbox as everything else in this module, never in this process.
    """
    from kullback.builder import templates as templates_mod  # builder to builder, at call time

    proposals = [p for p in proposals_from(artifact) if p.readers]
    traces = list(traces)
    derived = templates_mod.derived_reader(artifact, traces, workdir)
    if not proposals and not derived:
        return None
    values: dict[tuple[str, str], Optional[dict]] = {}
    for proposal in proposals:
        names = _method_names(proposal)
        source, refused = _module(proposal)
        if refused or not gate_parses(source).passed or not gate_confined(source).passed:
            continue
        jobs: list[tuple[str, str]] = []
        for trace in traces:
            for call in trace.tool_calls:
                if call.error is not None or call.name not in names:
                    continue
                key = (call.name, str(parse_result(call.result)))
                if key not in values and key not in jobs:
                    jobs.append(key)
        calls = [ToolCall(name=names[tool], args={"result": text}, raw_ptr=_PTR) for tool, text in jobs]
        results, refusal = _run(source, calls, Path(workdir) / "readers")
        if refusal:
            continue
        for key, result in zip(jobs, results, strict=False):
            value = result.get("value") if result.get("ok") else None
            values[key] = value if isinstance(value, dict) and value else None

    def read(tool: str, result: Any) -> Optional[dict]:
        key = (tool, str(result))
        return values.get(key) or derived.get(key)

    return read
