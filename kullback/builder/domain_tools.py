"""The Builder's domain tools: ingest, derive_world, grow, replay, run, status, examine.

Each tool wraps a pure, measured module the way the Examiner's tools do (pydantic args in,
a short rendered result out): `ingest_files` and `derive_world` for the first pass, the
runner tool for replay and reroll, `trusted_gate` for the per-Task status, and the function
the session passes in for examine (stream 6 delivers it as `kullback.examiner.session.examine`;
the session resolves it lazily, so this module never imports the examiner). Every result
carries rows, never a count alone.

Carried: D155 (the recording is the standard a replay row measures against), D171
(per-call rows in every answer), D224 (growing the world never lands in a trusted Task).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.builder import env_files, world_tools
from kullback.builder.replay_evidence import (
    HOLDOUT_ANSWERS_FILE,
    REPLAYS_FILE,
    HeldOutRuns,
    holdout_answers,
)
from kullback.gates.fidelity import reference_replay_gate
from kullback.gates.hook import ruling_line
from kullback.gates.ledger import GateLedger
from kullback.gates.trust import trusted_gate
from kullback.runner import tool as runner_tool
from kullback.runner.records import read_json, write_json
from kullback.runner.replay import AGREES

#: How many per-call rows the rendered text carries before the count of the rest.
ROWS_SHOWN = 20


def _row_line(row: dict[str, Any]) -> str:
    column = row.get("first_differing_column") or "agrees"
    return (f"{row.get('call_id')} {row.get('tool')}: recorded {row.get('recorded')} "
            f"against ours {row.get('ours')} [{column}]")


class IngestArgs(BaseModel):
    """The customer files to store as traces, as paths under the workdir."""

    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(description="Customer files to ingest, as paths under the workdir.")


class IngestResult(BaseModel):
    """What storing the files published: the counts and the ruling."""

    summary: str = ""
    files: list[str] = Field(default_factory=list)
    runs: int = 0
    tool_calls: int = 0
    gate: dict = Field(default_factory=dict)
    paths: list[str] = Field(default_factory=list)


def _render_ingest(result: IngestResult) -> str:
    return (f"{result.summary}\nfiles: {len(result.files)}, runs: {result.runs}, "
            f"calls: {result.tool_calls}")


class DeriveArgs(BaseModel):
    """How to grow the world, if at all."""

    model_config = ConfigDict(extra="forbid")

    grow: Optional[dict] = Field(default=None, description="Synthetic growth, never in a trusted Task.")


class DeriveResult(BaseModel):
    """What the first pass derived: row counts, call counts, the files it wrote."""

    summary: str = ""
    tables: dict[str, int] = Field(default_factory=dict)
    tools: dict[str, int] = Field(default_factory=dict)
    tasks: dict[str, int] = Field(default_factory=dict)
    files_written: list[str] = Field(default_factory=list)
    calls_exposed: dict[str, int] = Field(default_factory=dict)


def _render_derive(result: DeriveResult) -> str:
    lines = [result.summary]
    for table, count in sorted(result.tables.items()):
        lines.append(f"table {table}: {count} rows")
    for name, count in sorted(result.tools.items()):
        lines.append(f"tool {name}: {count} recorded calls")
    for task_id, count in sorted(result.tasks.items()):
        lines.append(f"task {task_id}: {count} runs")
    return "\n".join(lines)


class ReplayArgs(BaseModel):
    """The Task whose recorded calls to replay."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task whose recorded calls to replay.")


class ReplayRow(BaseModel):
    """One recorded call replayed: both answers and the first column that parted them."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = ""
    tool: str = ""
    verdict: str = ""
    first_differing_column: Optional[str] = None
    recorded: str = ""
    ours: str = ""


class ReplayTrace(BaseModel):
    """One Trace of the Task replayed: whose it is and how far it played back."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = ""
    reference: bool = False
    held_out: bool = False
    confirmed: bool = False
    fidelity: float = 0.0
    differing: int = 0
    calls: int = 0


class ReplayResult(BaseModel):
    """One Task replayed over every Trace: a line per Trace, the reference's per-call rows."""

    summary: str = ""
    task_id: str = ""
    fidelity: float = 0.0
    traces: list[ReplayTrace] = Field(default_factory=list)
    rows: list[ReplayRow] = Field(default_factory=list)
    total: int = 0
    rulings: list[dict] = Field(default_factory=list, description="One record per ruling in the "
                                "shape attach_ruling uses: name, accepted, line, rows.")


def _render_replay(result: ReplayResult) -> str:
    lines = [result.summary]
    for trace in result.traces:
        grain = "reference" if trace.reference else "held out" if trace.held_out else "seed"
        lines.append(f"trace {trace.trace_id} ({grain}): fidelity {trace.fidelity:.4f}, "
                     f"{trace.differing} differing of {trace.calls} calls")
    lines += [f"{row.call_id} {row.tool}: recorded {row.recorded} against ours {row.ours} "
              f"[{row.first_differing_column or 'agrees'}]" for row in result.rows[:ROWS_SHOWN]]
    rest = result.total - len(result.rows[:ROWS_SHOWN])
    if rest > 0:
        lines.append(f"and {rest} more of {result.total}")
    lines.append(f"fidelity: {result.fidelity:.4f} over {result.total} recorded calls")
    lines += [ruling["line"] for ruling in result.rulings]
    return "\n".join(lines)


def _ruling_record(ruling: Any) -> dict:
    """One ruling in the shape attach_ruling uses: name, accepted, line, rows."""
    return {"name": ruling.stage, "accepted": bool(ruling.passed), "line": ruling_line(ruling),
            "rows": [dict(row) for row in (getattr(ruling, "rows", None) or [])]}


def fidelity_index(replays: Any) -> dict[str, Any]:
    """The per-tool and per-Task replay fidelity off the replays dict, for tool_fidelity.json.

    Carried shape from build.py `attribute_fidelity` (D171), keyed on the runner's ReplayCall
    rows instead of the deleted call outcomes: `tools` counts every recorded call once,
    `tasks` keeps the first three differing calls per Task per tool, already worded.
    A call counts as replayed when its verdict is in the runner's AGREES, else as
    differing; a tool with a differing call is assisted.
    """
    tools: dict[str, Any] = {}
    tasks: dict[str, Any] = {}
    for task_id, traces in (replays or {}).items():
        if not isinstance(traces, dict):
            continue
        for _, row in traces.items():
            calls = row.get("calls") if isinstance(row, dict) else None
            for call in calls or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("tool") or "")
                if not name:
                    continue
                entry = tools.setdefault(
                    name, {"calls": 0, "replayed": 0, "differing": 0, "assisted": False})
                task_row = tasks.setdefault(str(task_id), {}).setdefault(
                    name, {"replayed": 0, "differing": 0, "reasons": []})
                entry["calls"] += 1
                if str(call.get("verdict") or "") in AGREES:
                    entry["replayed"] += 1
                    task_row["replayed"] += 1
                else:
                    entry["differing"] += 1
                    entry["assisted"] = True
                    task_row["differing"] += 1
                    if len(task_row["reasons"]) < 3:
                        task_row["reasons"].append(
                            f"call {call.get('call_id')}: {call.get('first_differing_column')} "
                            f"recorded {call.get('recorded')} ours {call.get('ours')}")
    return {"tools": tools, "tasks": tasks}


class RunArgs(BaseModel):
    """The Task to play fresh Runs of, and how many."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task to play fresh Runs of.")
    count: int = Field(default=1, ge=1, le=10, description="How many fresh Runs to play.")


class RunRow(BaseModel):
    """One fresh Run: its verdict, the routes taken, what moved, what it cost."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    verdict: Any = None
    routes: list[dict] = Field(default_factory=list)
    world_diff: dict = Field(default_factory=dict)
    spend: dict = Field(default_factory=dict)
    termination_reason: str = ""


class RunResult(BaseModel):
    """Fresh Runs played: one row per Run, never a count alone."""

    summary: str = ""
    task_id: str = ""
    runs: list[RunRow] = Field(default_factory=list)


def _render_run(result: RunResult) -> str:
    lines = [result.summary]
    for row in result.runs:
        routes = ", ".join(f"{item.get('name')}:{item.get('route')}" for item in row.routes)
        diff_tables = ", ".join(sorted(row.world_diff)) or "no table moved"
        spend = row.spend.get("usd") if isinstance(row.spend, dict) else None
        lines.append(f"run {row.run_id}: verdict {row.verdict} [{routes or 'no calls'}]; "
                     f"world: {diff_tables}; spend: {spend} USD; ended: {row.termination_reason}")
    return "\n".join(lines)


class ExamineArgs(BaseModel):
    """The Tasks to examine, or all of them."""

    model_config = ConfigDict(extra="forbid")

    task_ids: Optional[list[str]] = Field(default=None, description="The Tasks to examine, or all.")


class ExamineResult(BaseModel):
    """What the examination said: the summary and one record per finding."""

    summary: str = ""
    findings: list[dict] = Field(default_factory=list)


def _render_examine(result: ExamineResult) -> str:
    lines = [result.summary]
    for finding in result.findings:
        rows = finding.get("rows") or []
        lines.append(f"{finding.get('kind')} on {finding.get('task_id')}: {finding.get('path')}: "
                     f"{finding.get('change')} [{len(rows)} rows]")
    return "\n".join(lines)


def default_examine_fn(workdir: Any, task_ids: Any, *, model: Any = None, judge_model: Any = None,
                       probe_model: Any = None, reroll_model: Any = None) -> Any:
    """The examination stream 6 delivers, resolved lazily so this module never imports it.

    `model` runs the Examiner session; with None the call is derivation only. The judges, the
    loophole probe and the re-rolls run on their own models when named, else on `model`.
    """
    from kullback.examiner.session import examine

    return examine(Path(workdir), task_ids=task_ids, model=model, judge_model=judge_model or model,
                   probe_model=probe_model or model, reroll_model=reroll_model or model)


def _finding_row(item: Any) -> dict:
    """One finding as a dict: `as_dict()` where the examiner's Finding offers it."""
    if isinstance(item, dict):
        return dict(item)
    as_dict = getattr(item, "as_dict", None)
    if callable(as_dict):
        try:
            return dict(as_dict())
        except Exception:  # noqa: BLE001 - a finding that cannot render still counts
            pass
    return {"text": str(item)}


def _coerce_examine(raw: Any) -> ExamineResult:
    if isinstance(raw, ExamineResult):
        return raw
    if isinstance(raw, list):
        rows = [_finding_row(item) for item in raw]
        return ExamineResult(summary=f"{len(rows)} findings open", findings=rows)
    if isinstance(raw, BaseModel):
        body = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        body = dict(raw)
    else:
        return ExamineResult(summary=str(raw))
    findings = body.get("findings") or []
    rows = [_finding_row(item) for item in findings]
    return ExamineResult(summary=str(body.get("summary") or ""), findings=rows)


def status_of(workdir: Any) -> dict[str, Any]:
    """One row per Task (trusted, refused or open, with its reason) and per tool (last replay fidelity).

    Read off the workdir files through `trusted_gate`, the way the Examiner's summary does;
    the reading is carried here, not the import, since that module is going away. Files not
    there yet read as empty, so a Task with no Verifier is open with its reason.
    """
    root = Path(workdir)
    task_ids = [path.stem for path in sorted((root / "tasks").glob("*.json")) if path.name != "tasks.json"]
    task_status = read_json(root / "task_status.json", None) or {}
    verifiers: list[dict] = []
    if (root / "verifiers").is_dir():
        for path in sorted((root / "verifiers").glob("*.json")):
            try:
                verifiers.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    refusals: dict[str, Any] = {}
    for folder in (root / "refusals", root / "env" / "refusals"):
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                try:
                    refusals.setdefault(path.stem, json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
    # Probes and version history are not workdir files in this flow (the Examiner's open
    # item), and task runs are the Examiner's derivation, so the status rules on the suite,
    # the refusals and the version reason only.
    ruling = trusted_gate(task_status if isinstance(task_status, dict) else {}, verifiers, {},
                          {}, refusals, {},
                          read_json(root / "replays.json", None) or {},
                          read_json(root / "rerolls.json", None) or {},
                          read_json(root / "canon-rules.json", None),
                          read_json(root / "tool_sigs.json", None) or [])
    trusted = set(ruling.metrics.get("trusted") or [])
    refused = ruling.metrics.get("refused") or {}
    untrusted = ruling.metrics.get("untrusted") or {}
    tasks = []
    for task_id in task_ids:
        if task_id in trusted:
            tasks.append({"task_id": task_id, "state": "trusted", "reason": ""})
        elif task_id in refused:
            tasks.append({"task_id": task_id, "state": "refused", "reason": str(refused[task_id])})
        else:
            tasks.append({"task_id": task_id, "state": "open",
                          "reason": str(untrusted.get(task_id) or "no Verifier derived yet")})
    per_tool = (read_json(root / "tool_fidelity.json", None) or {}).get("tools") or {}
    tools = []
    for sig in read_json(root / "tool_sigs.json", None) or []:
        name = sig.get("name") if isinstance(sig, dict) else None
        if not name:
            continue
        grain = per_tool.get(name) if isinstance(per_tool, dict) else None
        calls = int((grain or {}).get("calls") or 0)
        if grain is not None and calls:
            replayed = int(grain.get("replayed") or 0)
            tools.append({"tool": name, "fidelity": replayed / calls, "calls": calls,
                          "assisted": bool(grain.get("assisted")), "reason": ""})
        else:
            tools.append({"tool": name, "fidelity": None, "calls": 0,
                          "assisted": False, "reason": "no replay rows yet"})
    return {"tasks": tasks, "tools": tools}


class GrowArgs(BaseModel):
    """The table to grow with synthetic rows, and the row count to reach."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="The table to grow with synthetic rows.")
    count: int = Field(ge=1, description="The row count for the table to reach.")


class GrowResult(BaseModel):
    """What growing the world added: the rows per table and the ruling behind them."""

    summary: str = ""
    tables: dict[str, int] = Field(default_factory=dict)
    paths: list[str] = Field(default_factory=list)


def _render_grow(result: GrowResult) -> str:
    lines = [result.summary]
    for table, count in sorted(result.tables.items()):
        lines.append(f"table {table}: {count} rows")
    return "\n".join(lines)


class StatusArgs(BaseModel):
    """No arguments: the status is read off the workdir files."""

    model_config = ConfigDict(extra="forbid")


class StatusResult(BaseModel):
    """One row per Task and per tool, in text and in details."""

    summary: str = ""
    tasks: list[dict] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list)


def _render_status(result: StatusResult) -> str:
    lines = [result.summary]
    for row in result.tasks:
        lines.append(f"task {row.get('task_id')}: {row.get('state')}"
                     + (f" ({row.get('reason')})" if row.get("reason") else ""))
    for row in result.tools:
        fidelity = row.get("fidelity")
        lines.append(f"tool {row.get('tool')}: "
                     + (f"fidelity {fidelity:.4f} over {row.get('calls')} calls"
                        if fidelity is not None else str(row.get("reason"))))
    return "\n".join(lines)
def domain_tools(*, workdir: Any, model: Any = None,
                 examine_fn: Optional[Callable[[Any, Any], Any]] = None, judge_model: Any = None,
                 probe_model: Any = None, reroll_model: Any = None) -> list[AgentTool]:
    """The seven domain tools bound to one workdir.

    `model` drives the fresh Runs `run` buys and the Examiner session the default
    `examine` runs; `examine_fn(workdir, task_ids)` answers `examine` and defaults to
    the lazily resolved examination stream 6 delivers, which hands the judges, the probe
    and the re-rolls their own models (each `model` when not named).
    """
    root = Path(workdir)
    if examine_fn is not None:
        examine_call = examine_fn
    else:
        def examine_call(workdir: Any, task_ids: Any) -> Any:
            return default_examine_fn(workdir, task_ids, model=model, judge_model=judge_model,
                                      probe_model=probe_model, reroll_model=reroll_model)

    async def ingest(args: IngestArgs) -> IngestResult:
        files = [(root / name if not Path(name).is_absolute() else Path(name)) for name in args.files]
        report = world_tools.ingest_files(files, root)
        return IngestResult(summary=f"ingested {len(report.files)} files",
                            files=list(report.files), runs=report.runs,
                            tool_calls=report.tool_calls, gate=dict(report.gate or {}),
                            paths=list(report.paths))

    async def derive(args: DeriveArgs) -> DeriveResult:
        report = world_tools.derive_world(root, grow=dict(args.grow or {}) if args.grow else None)
        written = env_files.explode(root)
        exposed = env_files.expose(root)
        files_written = [f"env/tools/{name}.py" for name in written]
        return DeriveResult(
            summary=(f"derived the world over {len(report.tools)} tools and {len(report.tasks)} tasks"),
            tables=dict(report.tables), tools={k: int(v) for k, v in report.tools.items()},
            tasks={k: int(v) for k, v in report.tasks.items()}, files_written=files_written,
            calls_exposed={k: int(v) for k, v in (exposed.get("calls") or {}).items()})

    async def grow(args: GrowArgs) -> GrowResult:
        report = world_tools.derive_world(root, grow={args.table: args.count})
        return GrowResult(summary=(f"grew table {args.table} to {args.count} rows with synthetic rows"),
                          tables={key: int(value) for key, value in report.tables.items()},
                          paths=list(report.paths))

    async def status(args: StatusArgs) -> StatusResult:
        _ = args
        body = status_of(root)
        states = [row["state"] for row in body["tasks"]]
        return StatusResult(
            summary=(f"{states.count('trusted')} trusted, {states.count('refused')} refused, "
                     f"{states.count('open')} open over {len(states)} tasks"),
            tasks=body["tasks"], tools=body["tools"])


    async def replay(args: ReplayArgs) -> ReplayResult:
        reports = runner_tool.replay_all(root, args.task_id, workdir=root)
        if not reports:
            raise ValueError(f"task {args.task_id} holds no recorded Trace to replay")
        replays = read_json(root / REPLAYS_FILE, None) or {}
        if not isinstance(replays, dict):
            replays = {}
        # Every Trace of the Task was replayed just now, so the Task's rows are replaced whole.
        replays[args.task_id] = {report.trace_id: report.as_dict() for report in reports}
        write_json(root / REPLAYS_FILE, replays)
        write_json(root / "tool_fidelity.json", fidelity_index(replays))
        write_json(root / HOLDOUT_ANSWERS_FILE, holdout_answers(replays, HeldOutRuns.of(root)))
        # D108 section 6: a Task none of whose Traces replay to their End state is named.
        ruling = GateLedger(root).record("replay_reference", reference_replay_gate(replays))
        reference = reports[0]
        rows = [ReplayRow(call_id=str(call.call_id or ""), tool=str(call.tool),
                          verdict=str(call.verdict),
                          first_differing_column=call.first_differing_column,
                          recorded=str(call.recorded), ours=str(call.ours))
                for call in reference.calls]
        traces = [ReplayTrace(trace_id=report.trace_id, reference=report.reference,
                              held_out=report.held_out,
                              confirmed=report.confirmed, fidelity=report.fidelity,
                              differing=sum(1 for call in report.calls if call.verdict not in AGREES),
                              calls=len(report.calls))
                  for report in reports]
        return ReplayResult(
            summary=(f"replay of task {args.task_id} over {len(reports)} traces: reference "
                     f"{reference.trace_id} at {reference.fidelity:.4f} fidelity"),
            task_id=args.task_id, fidelity=reference.fidelity, traces=traces, rows=rows,
            total=len(rows), rulings=[_ruling_record(ruling)])

    async def run(args: RunArgs) -> RunResult:
        if model is None:
            raise ValueError("run needs the session model to play fresh Runs with")
        reports = runner_tool.reroll(root, args.task_id, model, count=args.count, workdir=root)
        rows = [RunRow(run_id=item.run_id, verdict=item.verdict,
                       routes=[dict(route) for route in item.routes],
                       world_diff=dict(item.world_diff or {}),
                       spend=dict(item.spend or {}) if isinstance(item.spend, dict) else {},
                       termination_reason=str(item.termination_reason or ""))
                for item in reports]
        finished = sum(1 for item in reports if (item.termination_reason or "") == "finished")
        return RunResult(summary=(f"reroll of task {args.task_id}: {len(rows)} Runs, "
                                  f"{finished} finished"),
                         task_id=args.task_id, runs=rows)

    async def examine(args: ExamineArgs) -> ExamineResult:
        return _coerce_examine(examine_call(root, args.task_ids))

    return [
        AgentTool("ingest", "Store customer files as traces under the workdir.",
                  IngestArgs, IngestResult, ingest, render=_render_ingest),
        AgentTool("derive_world", "Derive the world from the stored traces: rows, signatures, "
                  "Tasks, the Starting state. Writes the tool files and the read surface.",
                  DeriveArgs, DeriveResult, derive, render=_render_derive),
        AgentTool("grow", "Grow one table with synthetic rows, never in a trusted Task.",
                  GrowArgs, GrowResult, grow, render=_render_grow),
        AgentTool("replay", "Replay every recorded Trace of one Task through the built tools, "
                  "call by call.",
                  ReplayArgs, ReplayResult, replay, render=_render_replay),
        AgentTool("run", "Play fresh Runs of one Task through the runner and report each one.",
                  RunArgs, RunResult, run, render=_render_run),
        AgentTool("status", "Read one row per Task and per tool off the workdir files.",
                  StatusArgs, StatusResult, status, render=_render_status),
        AgentTool("examine", "Examine Tasks when the Environment looks ready and file the findings.",
                  ExamineArgs, ExamineResult, examine, render=_render_examine),
    ]


__all__ = ["ROWS_SHOWN", "default_examine_fn", "domain_tools", "fidelity_index", "status_of"]
