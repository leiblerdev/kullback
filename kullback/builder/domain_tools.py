"""The Builder's domain tools: ingest, derive_world, replay, run, examine.

Each tool wraps a pure, measured module the way the Examiner's tools do (pydantic args in,
a short rendered result out): `ingest_files` and `derive_world` for the first pass, the
runner tool for replay and reroll, and the function the session passes in for examine
(stream 6 delivers it as `kullback.examiner.session.examine`; the session resolves it
lazily, so this module never imports the examiner). Every result carries per-call rows,
never a count alone.

Carried: D155 (the recording is the standard a replay row measures against), D171
(per-call rows in every answer), D224 (growing the world never lands in a trusted Task).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.builder import env_files, world_tools
from kullback.runner import tool as runner_tool

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


class ReplayResult(BaseModel):
    """One Task replayed: the per-call rows and the fidelity they compose."""

    summary: str = ""
    task_id: str = ""
    fidelity: float = 0.0
    rows: list[ReplayRow] = Field(default_factory=list)
    total: int = 0


def _render_replay(result: ReplayResult) -> str:
    shown = [f"{row.call_id} {row.tool}: recorded {row.recorded} against ours {row.ours} "
             f"[{row.first_differing_column or 'agrees'}]" for row in result.rows[:ROWS_SHOWN]]
    rest = result.total - len(result.rows[:ROWS_SHOWN])
    if rest > 0:
        shown.append(f"and {rest} more of {result.total}")
    shown.append(f"fidelity: {result.fidelity:.4f} over {result.total} recorded calls")
    return "\n".join([result.summary, *shown])


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
        lines.append(str(finding.get("text") or finding.get("summary") or finding))
    return "\n".join(lines)


def default_examine_fn(workdir: Any, task_ids: Any) -> Any:
    """The examination stream 6 delivers, resolved lazily so this module never imports it."""
    from kullback.examiner.session import examine

    return examine(Path(workdir), task_ids)


def _coerce_examine(raw: Any) -> ExamineResult:
    if isinstance(raw, ExamineResult):
        return raw
    if isinstance(raw, BaseModel):
        body = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        body = dict(raw)
    else:
        return ExamineResult(summary=str(raw))
    findings = body.get("findings") or []
    rows = [item if isinstance(item, dict) else {"text": str(item)} for item in findings]
    return ExamineResult(summary=str(body.get("summary") or ""), findings=rows)


def domain_tools(*, workdir: Any, model: Any = None,
                 examine_fn: Optional[Callable[[Any, Any], Any]] = None) -> list[AgentTool]:
    """The five domain tools bound to one workdir: ingest, derive_world, replay, run, examine.

    `model` drives the fresh Runs `run` buys; `examine_fn(workdir, task_ids)` answers
    `examine` and defaults to the lazily resolved examination stream 6 delivers.
    """
    root = Path(workdir)
    examine_call = examine_fn if examine_fn is not None else default_examine_fn

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

    async def replay(args: ReplayArgs) -> ReplayResult:
        report = runner_tool.replay(root, args.task_id, workdir=root)
        rows = [ReplayRow(call_id=str(call.call_id or ""), tool=str(call.tool),
                          verdict=str(call.verdict),
                          first_differing_column=call.first_differing_column,
                          recorded=str(call.recorded), ours=str(call.ours))
                for call in report.calls]
        return ReplayResult(
            summary=f"replay of task {args.task_id}: {report.fidelity:.4f} fidelity",
            task_id=args.task_id, fidelity=report.fidelity, rows=rows, total=len(rows))

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
        AgentTool("replay", "Replay one Task's recorded calls through the built tools, call by call.",
                  ReplayArgs, ReplayResult, replay, render=_render_replay),
        AgentTool("run", "Play fresh Runs of one Task through the runner and report each one.",
                  RunArgs, RunResult, run, render=_render_run),
        AgentTool("examine", "Examine Tasks when the Environment looks ready and file the findings.",
                  ExamineArgs, ExamineResult, examine, render=_render_examine),
    ]


__all__ = ["ROWS_SHOWN", "default_examine_fn", "domain_tools"]
