"""The Builder's domain tools: ingest, derive_world, grow, replay, rulings, run, status, examine, note_task.

Each tool wraps a pure, measured module the way the Examiner's tools do (pydantic args in,
a short rendered result out): `ingest_files` and `derive_world` for the first pass, the
runner tool for replay and reroll, `trusted_gate` for the per-Task status, and the function
the session passes in for examine (stream 6 delivers it as `kullback.examiner.session.examine`;
the session resolves it lazily, so this module imports the examiner only inside the calls that
need it: the note channel the two agents share). Every result carries rows, never a count alone.

Carried: D155 (the recording is the standard a replay row measures against), D171
(per-call rows in every answer), D224 (growing the world never lands in a trusted Task).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.ai.provider import MemoModel
from kullback.builder import env_files, world_tools
from kullback.builder.replay_evidence import (
    HOLDOUT_ANSWERS_FILE,
    REPLAYS_FILE,
    HeldOutRuns,
    holdout_answers,
)
from kullback.gates.fidelity import reference_replay_gate
from kullback.gates.hook import compact, ruled, ruling_line
from kullback.gates.ledger import GateLedger
from kullback.gates.loosening import legitimate_runs
from kullback.gates.trust import trusted_gate
from kullback.runner import budget
from kullback.runner import tool as runner_tool
from kullback.runner.records import EXAM_DIR, read_json, run_path, write_json
from kullback.runner.replay import AGREES, OURS_REFUSED, THEIRS_REFUSED
from kullback.runner.target import as_run
from kullback.runner.world.environment import BuiltEnvironment
from kullback.user import fidelity as user_fidelity
from kullback.user.rules import GOAL_SATISFIED, goal_write_set
from kullback.user.simulated import SimulatedUser
from kullback.user.value_strip import value_strip

#: How many per-call rows the rendered text carries before the count of the rest.
ROWS_SHOWN = 20


def _row_line(row: "ReplayRow") -> str:
    """One replayed call as the Builder reads it: the Task, the call, the arguments, then the error
    or both answers under the leaf column that parted them (F42)."""
    head = (f"task {row.task_id} call {row.call_id} {row.tool}"
            + (f" args {row.arguments}" if row.arguments else ""))
    if row.error:
        return f"{head}: {row.verdict} {row.error}"
    return f"{head}: recorded {row.recorded} against ours {row.ours} [{row.first_differing_column or 'agrees'}]"


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
    """The Task whose recorded calls to replay, or none for every Task."""

    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = Field(default=None, description="The Task whose recorded calls to "
                                   "replay; leave it out to replay every Task.")


class ReplayRow(BaseModel):
    """One recorded call replayed: both answers and the first column that parted them."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = ""
    task_id: str = ""
    tool: str = ""
    arguments: str = Field(default="", description="The call's arguments, compact JSON cut at 160 characters.")
    verdict: str = ""
    first_differing_column: Optional[str] = None
    recorded: str = ""
    ours: str = ""
    error: Optional[str] = Field(default=None, description="Where one side refused: the error class "
                                 "and message that side answered.")


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


class ReplayTask(BaseModel):
    """One Task of a replay over every Task: its reference and where it first parts."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = ""
    fidelity: float = 0.0
    confirmed: bool = False
    traces: int = 0
    first_differing_tool: Optional[str] = None
    first_differing_column: Optional[str] = None
    error: Optional[str] = None


class ReplayResult(BaseModel):
    """One Task replayed over every Trace: a line per Trace, the reference's per-call rows.

    Over every Task (no task_id), `tasks` holds one row per Task and `tools` the per-tool
    fidelity the status tool reads; the rows of every reference ride in details.
    """

    summary: str = ""
    task_id: str = ""
    fidelity: float = 0.0
    traces: list[ReplayTrace] = Field(default_factory=list)
    rows: list[ReplayRow] = Field(default_factory=list)
    total: int = 0
    rulings: list[dict] = Field(default_factory=list, description="One record per ruling in the "
                                "shape attach_ruling uses: name, accepted, line, rows.")
    tasks: list[ReplayTask] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list)


def _render_replay_all(result: ReplayResult) -> str:
    lines = [result.summary]
    for row in result.tools:
        fidelity = row.get("fidelity")
        lines.append(f"tool {row.get('tool')}: "
                     + (f"fidelity {fidelity:.4f} over {row.get('calls')} calls"
                        if fidelity is not None else str(row.get("reason"))))
    worst = sorted(result.tasks, key=lambda task: (task.error is None, task.fidelity, task.task_id))
    for task in worst[:ROWS_SHOWN]:
        if task.error is not None:
            lines.append(f"task {task.task_id}: not replayed ({task.error})")
            continue
        where = (f"first differs at {task.first_differing_tool} [{task.first_differing_column}]"
                 if task.first_differing_tool else "every call agrees")
        lines.append(f"task {task.task_id}: fidelity {task.fidelity:.4f}, "
                     f"{'confirmed' if task.confirmed else 'not confirmed'}, {where}")
    rest = len(result.tasks) - len(worst[:ROWS_SHOWN])
    if rest > 0:
        lines.append(f"and {rest} more Tasks, all at or above the fidelity shown")
    lines += [ruling["line"] for ruling in result.rulings]
    return "\n".join(lines)


def _render_replay(result: ReplayResult) -> str:
    if result.tasks:
        return _render_replay_all(result)
    lines = [result.summary]
    for trace in result.traces:
        grain = "reference" if trace.reference else "held out" if trace.held_out else "seed"
        lines.append(f"trace {trace.trace_id} ({grain}): fidelity {trace.fidelity:.4f}, "
                     f"{trace.differing} differing of {trace.calls} calls")
    lines += [_row_line(row) for row in result.rows[:ROWS_SHOWN]]
    rest = result.total - len(result.rows[:ROWS_SHOWN])
    if rest > 0:
        lines.append(f"and {rest} more of {result.total}")
    lines.append(f"fidelity: {result.fidelity:.4f} over {result.total} recorded calls")
    lines += [ruling["line"] for ruling in result.rulings]
    return "\n".join(lines)


class RulingsArgs(BaseModel):
    """The tool file to re-run the write rulings on, as it is on disk."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="A tool file under your root, e.g. tools/<name>.py.")


class RulingsResult(BaseModel):
    """The rulings a write of the file would draw, drawn without a write."""

    summary: str = ""
    path: str = ""
    rulings: list[dict] = Field(default_factory=list, description="One record per ruling in the "
                                "shape attach_ruling uses: name, accepted, line, rows.")


def _render_rulings(result: RulingsResult) -> str:
    return "\n".join([result.summary, *(ruling["line"] for ruling in result.rulings)])


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


#: The most fresh Runs one run call plays; the Tasks past it are named as not run.
RUNS_PER_CALL = 20


class RunArgs(BaseModel):
    """The Tasks to play fresh Runs of, and how many per Task."""

    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = Field(default=None, description="One Task to play fresh Runs of.")
    task_ids: Optional[list[str]] = Field(
        default=None, description="The Tasks to play fresh Runs of; leave both out for every "
        "open Task with a confirmed Reference.")
    count: int = Field(default=1, ge=1, le=10, description="How many fresh Runs to play per Task.")


class RunRow(BaseModel):
    """One fresh Run: its verdict, the routes taken, what moved, what it cost."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    verdict: Any = None
    routes: list[dict] = Field(default_factory=list)
    world_diff: dict = Field(default_factory=dict)
    spend: dict = Field(default_factory=dict)
    termination_reason: str = ""
    user_end: Optional[str] = None
    task_id: str = ""


class RunResult(BaseModel):
    """Fresh Runs played: one row per Run, never a count alone, and the Tasks left unplayed."""

    summary: str = ""
    task_id: str = ""
    runs: list[RunRow] = Field(default_factory=list)
    not_run: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list, description="Why each not_run Task was not played, "
                               "in the same order.")
    no_verifier: int = Field(default=0, description="Open Tasks with a confirmed Reference left out "
                             "because no Verifier is derived for them yet.")


# Why a named Task was not played: past the per-call cap, or no Simulated user to answer it (F34).
CAP_REASON = "past the cap of {cap} Runs per call"
NO_USER_REASON = "no Simulated user: the Reference is not confirmed, replay first"


def _run_spend(root: Path, report: Any) -> dict:
    """What one Run cost, summed off its own model_call events: the runner never totals it (F29)."""
    spend = dict(report.spend or {}) if isinstance(report.spend, dict) else {}
    try:
        events = as_run(run_path(root, report.path)).events if report.path else []
    except (OSError, ValueError, TypeError):
        return spend
    costs = [event.cost for event in events if event.type == "model_call" and event.cost is not None]
    return {**spend, "usd": round(sum(cost.usd for cost in costs), 6), "calls": len(costs)}


def _render_run(result: RunResult) -> str:
    lines = [result.summary]
    for row in result.runs:
        routes = ", ".join(f"{item.get('name')}:{item.get('route')}" for item in row.routes)
        diff_tables = ", ".join(sorted(row.world_diff)) or "no table moved"
        spend = row.spend.get("usd") if isinstance(row.spend, dict) else None
        lines.append(f"run {row.run_id}: verdict {row.verdict} [{routes or 'no calls'}]; "
                     f"world: {diff_tables}; spend: {spend} USD; ended: {row.termination_reason}"
                     + (f", user {row.user_end}" if row.user_end else "")
                     + ("; no Verifier yet: examine derives it from the confirmed Reference, "
                        "then run scores against it" if row.verdict is None else ""))
    by_reason: dict[str, list[str]] = {}
    for index, task_id in enumerate(result.not_run):
        reason = result.reasons[index] if index < len(result.reasons) else CAP_REASON
        by_reason.setdefault(reason, []).append(task_id)
    for reason, task_ids in by_reason.items():
        shown = ", ".join(task_ids[:ROWS_SHOWN])
        more = len(task_ids) - len(task_ids[:ROWS_SHOWN])
        reason = reason.format(cap=RUNS_PER_CALL)
        lines.append(f"not run, {reason}: {shown}" + (f" and {more} more" if more > 0 else ""))
    if result.no_verifier:
        lines.append(f"{result.no_verifier} open Tasks skipped for no Verifier: examine derives them first")
    return "\n".join(lines)


class ExamineArgs(BaseModel):
    """The Tasks to examine, or all of them."""

    model_config = ConfigDict(extra="forbid")

    task_ids: Optional[list[str]] = Field(default=None, description="The Tasks to examine, or all.")


class ExamineResult(BaseModel):
    """What the examination said: the summary, one record per finding, and the Tasks held."""

    summary: str = ""
    findings: list[dict] = Field(default_factory=list)
    held: list[str] = Field(default_factory=list, description="One line per Task the exam status "
                            "holds: the Task id and the check that holds it.")


#: The most held Tasks the examine result names before counting the rest.
HELD_SHOWN = 40


def _one_line(text: Any) -> str:
    """A finding's change on one line, cut at the row field cap."""
    return compact(" ".join(str(text or "").split()))


def _render_examine(result: ExamineResult) -> str:
    lines = [result.summary]
    for finding in result.findings:
        rows = finding.get("rows") or []
        lines.append(f"{finding.get('task_id') or 'no task'}: {finding.get('kind')}: "
                     f"{_one_line(finding.get('change') or finding.get('text'))}"
                     + (f" ({finding.get('path')})" if finding.get("path") else "") + f" [{len(rows)} rows]")
    lines += result.held[:HELD_SHOWN]
    if len(result.held) > HELD_SHOWN:
        lines.append(f"and {len(result.held) - HELD_SHOWN} more held Tasks")
    return "\n".join(lines)


def held_tasks(workdir: Any) -> list[str]:
    """One line per Task exam/task_status.json holds, naming the check that holds it (F59).

    A Task is held where its Verifier did not pass (a waived check on a passed Verifier holds
    nothing). A check the Examiner could not run reads "not run", one it ran and failed "failed";
    a Verifier not passed with no check named reads "verifier not passed". Read only.
    """
    status = read_json(Path(workdir) / EXAM_DIR / "task_status.json", None) or {}
    lines = []
    for task_id, row in sorted(status.items()) if isinstance(status, dict) else ():
        if not isinstance(row, dict):
            continue
        not_run = row.get("not_run_reasons") or {}
        failed = [check for check, ok in (row.get("checks") or {}).items() if not ok]
        passed = row.get("verifier_passed")
        if passed is True or (passed is None and not failed):
            continue
        reasons = [f"{check} {'not run' if check in not_run else 'failed'}" for check in failed]
        lines.append(f"{task_id}: {', '.join(reasons) or 'verifier not passed'}")
    return lines


def default_examine_fn(workdir: Any, task_ids: Any, *, model: Any = None, judge_model: Any = None,
                       probe_model: Any = None, reroll_model: Any = None) -> Any:
    """The examination stream 6 delivers, resolved lazily so this module never imports it.

    `model` runs the Examiner session; with None the call is derivation only. The judges, the
    loophole probe and the re-rolls run on their own models when named, else on `model`.
    With no finished Run among the named Tasks the examination opens no Examiner session: its
    selection hands the session only Tasks with a finished Run and a confirmed Reference (F24).
    """
    from kullback.examiner.session import examine

    return examine(Path(workdir), task_ids=task_ids, model=model, judge_model=judge_model or model,
                   probe_model=probe_model or model, reroll_model=reroll_model or model)


def priced_model(model: Any, workdir: Any, stage: str, *, cap_context: bool = True) -> Any:
    """The model priced into budget.json under `stage`, so every call it makes lands in the ledger.

    A model already priced is passed through as it is: wrapping it twice would charge each call
    twice. The candidate of a Run is tested under the production setting and never capped (D65),
    so the runner stage passes `cap_context=False`.
    """
    if model is None or isinstance(model, budget.BudgetedModel):
        return model
    return budget.BudgetedModel(model, stage=stage, workdir=Path(workdir),
                                model_id=getattr(model, "name", None), cap_context=cap_context)


def reader_model(model: Any, workdir: Any) -> Any:
    """The session model as the readers step takes it, priced under its own stage (F12, F23).

    Memoised on disk (provider.MemoModel): a reader proposal is a deterministic re-ask of the same
    recording, so a rebuild or a resumed build in the same workdir asks nothing it asked before and
    the ledger counts the hit under memo_hits. A model already priced is taken as it is.
    """
    if model is None or isinstance(model, budget.BudgetedModel):
        return model
    return priced_model(MemoModel(model, workdir), workdir, "readers")


def runner_model(model: Any, workdir: Any) -> Any:
    """The model a fresh Run plays, priced under the runner stage (F29)."""
    return priced_model(model, workdir, "runner", cap_context=False)


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


def stage_of(workdir: Any) -> dict[str, Any]:
    """Where the build stands, off the workdir files: traces, world, toolkit, stubs, finished Runs.

    `finished` counts the Tasks with a finished Run the way `trusted_gate` reads them (confirmed
    replays plus finished re-rolls), so the reading and the gate never disagree on it.
    """
    root = Path(workdir)
    traces = root / "traces"
    tools_dir = root / "env" / env_files.TOOLS_DIR
    tool_files = sorted(tools_dir.glob("*.py")) if tools_dir.is_dir() else []
    tasks = [path for path in (root / "tasks").glob("*.json") if path.name != "tasks.json"]
    runs = legitimate_runs(read_json(root / "replays.json", None) or {},
                           read_json(root / "rerolls.json", None) or {})
    return {
        "traces": sum(1 for path in traces.rglob("*") if path.is_file()) if traces.is_dir() else 0,
        "world_derived": (root / env_files.SCHEMA_FILE).is_file() and (root / env_files.SIGS_FILE).is_file(),
        "toolkit_rendered": (root / "env" / "tools.py").is_file() and (root / "env" / "data_model.py").is_file(),
        "stubs": sum(1 for path in tool_files if env_files.is_stub(path)),
        "tools": len(tool_files),
        "finished": sum(1 for run_ids in runs.values() if run_ids),
        "tasks": len(tasks),
    }


def stage_line(stage: dict[str, Any]) -> str:
    """The stage as one line, the first the status tool renders."""
    return (f"stage: traces {stage['traces']}, "
            f"{'world derived' if stage['world_derived'] else 'world not derived'}, "
            f"{'toolkit rendered' if stage['toolkit_rendered'] else 'toolkit not rendered'}, "
            f"stubs {stage['stubs']} of {stage['tools']}, "
            f"tasks with a finished run {stage['finished']} of {stage['tasks']}")


def root_line(workdir: Any) -> str:
    """What env/ holds directly, names only, sorted, directories with a trailing slash (F19)."""
    env = Path(workdir) / "env"
    names = sorted(path.name + ("/" if path.is_dir() else "") for path in env.iterdir()) if env.is_dir() else []
    return f"Your root holds: {', '.join(names)}." if names else "Your root is empty."


def opening_for(workdir: Any, default: str) -> str:
    """The Builder's first message: the state and the next step, what the root holds, then `default`."""
    stage = stage_of(workdir)
    lines = []
    if stage["traces"] and not stage["world_derived"]:
        lines.append(f"{stage['traces']} traces are stored and the world is not derived: "
                     f"call derive_world first.")
    elif stage["world_derived"] and stage["tools"] and stage["stubs"] == stage["tools"]:
        lines.append(f"The world is derived and every body is a stub ({stage['stubs']} of {stage['tools']}): "
                     f"write each body from its calls, then replay.")
    return "\n".join(lines + notes_lines(workdir) + [root_line(workdir), default])


def notes_lines(workdir: Any) -> list[str]:
    """Your notes on Tasks with the Examiner's ruling on each, or open while it has not ruled."""
    from kullback.examiner.domain_tools import note_line, notes_of

    notes = notes_of(workdir)
    if not notes:
        return []
    return ["Your notes and the Examiner's rulings on them:"] + [
        note_line(task_id, note, ruling) for task_id, (note, ruling) in sorted(notes.items())]


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="One of: outcome_not_in_state, intent_contradicts_reference, "
                                    "fact_unavailable_to_user, needs_action_record.")
    sentence: str = Field(description="One sentence saying why the Task is not verifiable as written; "
                                      "never an atom or Verifier text.")


class NoteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    note: dict = Field(default_factory=dict)


def _render_note(result: NoteResult) -> str:
    return result.summary


def status_of(workdir: Any) -> dict[str, Any]:
    """One row per Task (trusted, refused or open, with its reason) and per tool (last replay fidelity).

    Read off the workdir files through `trusted_gate`, the way the Examiner's summary does;
    the reading is carried here, not the import, since that module is going away. The Examiner's
    artefacts (its proposed Verifiers, the version histories, the task runs, the probe pools) are
    read where the Examiner wrote them, through the path functions in records (F22). Files not
    there yet read as empty, so a Task with no Verifier is open with its reason.
    """
    root = Path(workdir)
    task_ids = [path.stem for path in sorted((root / "tasks").glob("*.json")) if path.name != "tasks.json"]
    task_status = read_json(root / "task_status.json", None) or {}
    # The Examiner's artefacts, named where the Examiner writes them (F22); imported here so the
    # module's shared import block stays untouched by this region.
    from kullback.runner.records import exam_verifier_path, load_exam_history, load_exam_task_runs, load_probe_pools

    verifiers: list[dict] = []
    for path in sorted((root / "verifiers").glob("*.json")) if (root / "verifiers").is_dir() else []:
        # Each Task's live Verifier: the Examiner's proposal where it wrote one, else the derived file.
        proposal = exam_verifier_path(root, path.stem)
        for source in (proposal, path) if proposal.is_file() else (path,):
            try:
                verifiers.append(json.loads(source.read_text(encoding="utf-8")))
                break
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
    ruling = trusted_gate(task_status if isinstance(task_status, dict) else {}, verifiers,
                          load_probe_pools(root), load_exam_history(root), refusals, load_exam_task_runs(root),
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
    stage: dict = Field(default_factory=dict)
    tasks: list[dict] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list)


def _render_status(result: StatusResult) -> str:
    lines = ([stage_line(result.stage)] if result.stage else []) + [result.summary]
    for row in result.tasks:
        lines.append(f"task {row.get('task_id')}: {row.get('state')}"
                     + (f" ({row.get('reason')})" if row.get("reason") else ""))
    for row in result.tools:
        fidelity = row.get("fidelity")
        lines.append(f"tool {row.get('tool')}: "
                     + (f"fidelity {fidelity:.4f} over {row.get('calls')} calls"
                        if fidelity is not None else str(row.get("reason"))))
    return "\n".join(lines)
def rule_user(workdir: Any, task_id: str, router: Any) -> Optional[SimulatedUser]:
    """The rule-driven Simulated user of a fresh Run, or nothing where the Task has no rules.

    It opens with the goal the recorded user stated and answers from the rules, then the Run's
    own Starting state (D44, D77). The runner cannot import the user package, so the Run gets it
    through `make_user`; without rules the Run falls back to the Task's intent as before.
    """
    env = BuiltEnvironment(workdir)
    task = env.task(task_id)
    rules = env.rules(task)
    if rules is None:
        return None
    writes = env.write_tools()
    reference = env.reference(task)
    members = env.members(task)
    return SimulatedUser(
        rules, starting_state_reader=router.state, vocab=user_fidelity.vocabulary_of(workdir),
        write_tools=writes,
        goal_writes=goal_write_set(reference, writes) if reference is not None else None,
        answer_strip=value_strip(members) if members else None)


def call_arguments(env: Any) -> dict[str, Any]:
    """The arguments of every shown recorded call, by call id, off the calls files under env/."""
    folder = Path(env) / env_files.CALLS_DIR
    out: dict[str, Any] = {}
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else ():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("id"):
                out[str(row["id"])] = row.get("args")
    return out


def domain_tools(*, workdir: Any, model: Any = None,
                 examine_fn: Optional[Callable[[Any, Any], Any]] = None, judge_model: Any = None,
                 probe_model: Any = None, reroll_model: Any = None, env: Any = None) -> list[AgentTool]:
    """The nine domain tools bound to one workdir and the Builder's root `env` (workdir/env by default).

    `model` drives the fresh Runs `run` buys and the Examiner session the default
    `examine` runs; `examine_fn(workdir, task_ids)` answers `examine` and defaults to
    the lazily resolved examination stream 6 delivers, which hands the judges, the probe
    and the re-rolls their own models (each `model` when not named).
    """
    root = Path(workdir)
    env_root = Path(env) if env is not None else root / "env"
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
        # A corpus with prose results needs a model to propose its readers: the session model,
        # priced under its own stage so its calls land in budget.json.
        report = world_tools.derive_world(root, reader_model=reader_model(model, root),
                                          grow=dict(args.grow or {}) if args.grow else None)
        written = env_files.explode(root)
        env_files.regenerate(root)
        exposed = env_files.expose(root)
        files_written = [f"env/tools/{name}.py" for name in written]
        return DeriveResult(
            summary=(f"derived the world over {len(report.tools)} tools and {len(report.tasks)} tasks"),
            tables=dict(report.tables), tools={k: int(v) for k, v in report.tools.items()},
            tasks={k: int(v) for k, v in report.tasks.items()}, files_written=files_written,
            calls_exposed={k: int(v) for k, v in (exposed.get("calls") or {}).items()})

    async def grow(args: GrowArgs) -> GrowResult:
        report = world_tools.derive_world(root, reader_model=reader_model(model, root),
                                          grow={args.table: args.count})
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
            stage=stage_of(root), tasks=body["tasks"], tools=body["tools"])


    def replay_rows(reference: Any, arguments: dict[str, Any]) -> list[ReplayRow]:
        """The reference's calls as rows: the arguments off the record, else the calls files (F42)."""
        rows = []
        for call in reference.calls:
            args = getattr(call, "args", None)
            if args is None:
                args = arguments.get(str(call.call_id or ""))
            error = (str(call.ours) if call.verdict == OURS_REFUSED
                     else str(call.recorded) if call.verdict == THEIRS_REFUSED else None)
            rows.append(ReplayRow(call_id=str(call.call_id or ""), task_id=str(reference.task_id),
                                  tool=str(call.tool), arguments=compact(args) if args is not None else "",
                                  verdict=str(call.verdict),
                                  first_differing_column=call.first_differing_column,
                                  recorded=str(call.recorded), ours=str(call.ours), error=error))
        return rows

    def store_replays(replayed: dict[str, list]) -> Any:
        replays = read_json(root / REPLAYS_FILE, None) or {}
        if not isinstance(replays, dict):
            replays = {}
        # Every Trace of each Task was replayed just now, so each Task's rows are replaced whole.
        for task_id, reports in replayed.items():
            replays[task_id] = {report.trace_id: report.as_dict() for report in reports}
        write_json(root / REPLAYS_FILE, replays)
        write_json(root / "tool_fidelity.json", fidelity_index(replays))
        write_json(root / HOLDOUT_ANSWERS_FILE, holdout_answers(replays, HeldOutRuns.of(root)))
        # D108 section 6: a Task none of whose Traces replay to their End state is named.
        return GateLedger(root).record("replay_reference", reference_replay_gate(replays))

    def replay_every_task() -> ReplayResult:
        task_ids = [path.stem for path in sorted((root / "tasks").glob("*.json"))
                    if path.name != "tasks.json"]
        replayed: dict[str, list] = {}
        tasks: list[ReplayTask] = []
        rows: list[ReplayRow] = []
        arguments = call_arguments(env_root)
        for task_id in task_ids:
            try:
                reports = runner_tool.replay_all(root, task_id, workdir=root)
            except Exception as exc:  # noqa: BLE001 - one Task that cannot replay never stops the rest
                tasks.append(ReplayTask(task_id=task_id, error=str(exc) or type(exc).__name__))
                continue
            if not reports:
                tasks.append(ReplayTask(task_id=task_id, error="no recorded Trace to replay"))
                continue
            replayed[task_id] = reports
            reference = reports[0]
            first = next((call for call in reference.calls if call.verdict not in AGREES), None)
            rows += replay_rows(reference, arguments)
            tasks.append(ReplayTask(task_id=task_id, fidelity=reference.fidelity,
                                    confirmed=reference.confirmed, traces=len(reports),
                                    first_differing_tool=str(first.tool) if first else None,
                                    first_differing_column=first.first_differing_column if first else None))
        ruling = store_replays(replayed)
        confirmed = sum(1 for task in tasks if task.confirmed)
        mean = sum(task.fidelity for task in tasks) / len(tasks) if tasks else 0.0
        return ReplayResult(
            summary=(f"replay of every Task: {len(replayed)} of {len(task_ids)} Tasks replayed, "
                     f"{confirmed} confirmed References, mean reference fidelity {mean:.4f}"),
            fidelity=mean, rows=rows, total=len(rows), rulings=[_ruling_record(ruling)],
            tasks=tasks, tools=status_of(root)["tools"])

    async def replay(args: ReplayArgs) -> ReplayResult:
        if args.task_id is None:
            return replay_every_task()
        reports = runner_tool.replay_all(root, args.task_id, workdir=root)
        if not reports:
            raise ValueError(f"task {args.task_id} holds no recorded Trace to replay")
        ruling = store_replays({args.task_id: reports})
        reference = reports[0]
        rows = replay_rows(reference, call_arguments(env_root))
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

    async def rulings(args: RulingsArgs) -> RulingsResult:
        # The same rulings a write draws, on the file as it is on disk; nothing is written (F43).
        path = args.path.replace("\\", "/").lstrip("/")
        if not (env_root / path).is_file():
            raise ValueError(f"{args.path} is not a file under your root")
        drawn = ruled(env_root, path, root, execute=env_files.executor(root, env_root))
        if not drawn:
            raise ValueError(f"{args.path} is bound to no ruling: name a tool file, e.g. tools/<name>.py")
        failed = [ruling.stage for ruling in drawn if not ruling.passed]
        return RulingsResult(summary=(f"rulings on {path}: " + (f"fails {', '.join(failed)}" if failed
                                                              else f"{len(drawn)} passed")),
                             path=path, rulings=[_ruling_record(ruling) for ruling in drawn])

    def runnable_tasks() -> tuple[list[str], int]:
        """Every open Task whose reference Trace replayed confirmed and that has a Verifier file, in
        Task order, and how many confirmed open Tasks were left out for no Verifier (F54)."""
        replays = read_json(root / REPLAYS_FILE, None) or {}
        open_ids = [row["task_id"] for row in status_of(root)["tasks"] if row["state"] == "open"]
        confirmed = [task_id for task_id in open_ids
                     if any(isinstance(trace, dict) and trace.get("reference") and trace.get("confirmed")
                            for trace in ((replays.get(task_id) or {}) if isinstance(replays, dict)
                                          else {}).values())]
        verified = [task_id for task_id in confirmed if (root / "verifiers" / f"{task_id}.json").is_file()]
        return verified, len(confirmed) - len(verified)

    async def run(args: RunArgs) -> RunResult:
        if model is None:
            raise ValueError("run needs the session model to play fresh Runs with")
        named = args.task_ids if args.task_ids is not None else [args.task_id] if args.task_id else None
        runnable, no_verifier = runnable_tasks() if named is None else ([], 0)
        task_ids = list(named) if named is not None else runnable
        # A Task without user rules would be played against no Simulated user: the candidate
        # gets its system prompt alone and invents a customer. It is named, never played (F34).
        env = BuiltEnvironment(root)
        userless = [task_id for task_id in task_ids if env.rules(env.task(task_id)) is None]
        playable = [task_id for task_id in task_ids if task_id not in userless]
        fits = max(1, RUNS_PER_CALL // args.count)
        played, capped = playable[:fits], playable[fits:]
        not_run = userless + capped
        reasons = [NO_USER_REASON] * len(userless) + [CAP_REASON] * len(capped)
        priced = runner_model(model, root)
        rows: list[RunRow] = []
        finished = 0
        for task_id in played:
            reports = runner_tool.reroll(root, task_id, priced, count=args.count, workdir=root,
                                         make_user=lambda router, task_id=task_id: rule_user(root, task_id, router))
            rows += [RunRow(run_id=item.run_id, verdict=item.verdict,
                            routes=[dict(route) for route in item.routes],
                            world_diff=dict(item.world_diff or {}),
                            spend=_run_spend(root, item),
                            termination_reason=str(item.termination_reason or ""),
                            user_end=item.user_end, task_id=task_id)
                     for item in reports]
            # A Run is finished when its Simulated user ended it with the goal met (D210); the loop's
            # own end reason says only who stopped, never whether the goal was reached.
            finished += sum(1 for item in reports if item.user_end == GOAL_SATISFIED)
        if len(task_ids) == 1 and named is not None and played:
            summary = f"reroll of task {task_ids[0]}: {len(rows)} Runs, {finished} finished"
        elif len(task_ids) == 1 and named is not None:
            summary = f"task {task_ids[0]} not run: {reasons[0].format(cap=RUNS_PER_CALL)}"
        else:
            summary = (f"reroll of {len(played)} Tasks: {len(rows)} Runs, {finished} finished, "
                       f"{len(not_run)} Tasks left not run"
                       + ("" if task_ids or named is not None or no_verifier
                          else "; no open Task has a confirmed Reference, replay first"))
        return RunResult(summary=summary, task_id=task_ids[0] if len(task_ids) == 1 else "",
                         runs=rows, not_run=not_run, reasons=reasons, no_verifier=no_verifier)

    async def note_task(args: NoteArgs) -> NoteResult:
        from kullback.examiner.domain_tools import write_note

        if not (root / "tasks" / f"{args.task_id}.json").is_file():
            raise ValueError(f"there is no Task {args.task_id!r} under tasks/")
        note = write_note(root, args.task_id, args.reason, args.sentence)
        return NoteResult(summary=f"note on task {args.task_id} ({args.reason}) written; the Examiner "
                                  "rules on it before the Task can be refused",
                          task_id=args.task_id, note=note)

    async def examine(args: ExamineArgs) -> ExamineResult:
        # The Examiner session owns its own event loop, so it runs off the Builder's, on a thread.
        result = _coerce_examine(await asyncio.to_thread(examine_call, root, args.task_ids))
        # The Tasks the Examiner session left out, and why, lead the summary (F24).
        left_out = [f.get("change") for f in result.findings
                    if any(isinstance(r, dict) and "left_out" in r for r in f.get("rows") or [])]
        # The confirmed Tasks past this call's derivation cap come first, so the Builder calls again (F40).
        not_derived = [f.get("text") for f in result.findings
                       if any(isinstance(r, dict) and "not_derived" in r for r in f.get("rows") or [])]
        if left_out or not_derived:
            result.summary = "; ".join(filter(None, [*not_derived, result.summary, *left_out]))
        stage = stage_of(root)
        if not result.findings or not stage["finished"]:
            why = f"{stage['finished']} of {stage['tasks']} tasks have a finished run"
            if not stage["finished"]:
                why += ", so there is nothing to examine yet: replay or run the Tasks first"
            result.summary = f"{result.summary}: {why}" if result.summary else why
        result.held = held_tasks(root)
        return result

    return [
        AgentTool("ingest", "Store customer files as traces under the workdir.",
                  IngestArgs, IngestResult, ingest, render=_render_ingest),
        AgentTool("derive_world", "Derive the world from the stored traces: rows, signatures, "
                  "Tasks, the Starting state. Writes the tool files and the read surface.",
                  DeriveArgs, DeriveResult, derive, render=_render_derive),
        AgentTool("grow", "Grow one table with synthetic rows, never in a trusted Task.",
                  GrowArgs, GrowResult, grow, render=_render_grow),
        AgentTool("replay", "Replay every recorded Trace of one Task through the built tools, "
                  "call by call, or of every Task when no task_id is given.",
                  ReplayArgs, ReplayResult, replay, render=_render_replay),
        AgentTool("rulings", "Re-run the rulings on a tool file without editing it: every failing row, "
                  "grouped, the way a write of the file would draw them.",
                  RulingsArgs, RulingsResult, rulings, render=_render_rulings),
        AgentTool("run", "Play fresh Runs of the Tasks named, or of every open Task with a confirmed "
                  "Reference, through the runner and report each one.",
                  RunArgs, RunResult, run, render=_render_run),
        AgentTool("status", "Read one row per Task and per tool off the workdir files.",
                  StatusArgs, StatusResult, status, render=_render_status),
        AgentTool("examine", "Examine Tasks when the Environment looks ready and file the findings.",
                  ExamineArgs, ExamineResult, examine, render=_render_examine),
        AgentTool("note_task", "Say a Task is not verifiable as written: one reason off the fixed list "
                  "and one sentence, never an atom. The Examiner rules on it next.",
                  NoteArgs, NoteResult, note_task, render=_render_note),
    ]


__all__ = ["HELD_SHOWN", "ROWS_SHOWN", "RUNS_PER_CALL", "call_arguments", "default_examine_fn", "domain_tools",
           "fidelity_index", "held_tasks", "notes_lines", "opening_for", "root_line", "stage_line", "stage_of", "status_of"]
