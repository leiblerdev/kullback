"""The face of the Harness: one Markdown report that opens with whether the Environment was built, then the numbers per Task, then a suggestion the person decides on (D85); it reads records and never computes a Verdict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from harness.shared.records import (
    Constraint,
    Environment,
    GateResult,
    Record,
    Run,
    Task,
    TaskOverlay,
    ToolSig,
    Verdict,
    Verifier,
)

SECTIONS = ("## Environment", "## Tasks", "## Disagreement queue", "## Lessons set aside")


class ScorecardItem(Record):
    """One scorecard number with its raw and explained value side by side (D62, D80)."""
    name: str
    raw: str = ""
    explained: str = ""
    note: str = ""


class StageStatus(Record):
    """One pipeline stage as the run recorded it, for the DAG."""
    name: str
    status: str = "pending"
    gate: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 0
    usd: float = 0.0


class TaskCoverage(Record):
    """Whether one Task is covered, and the first reason it is not (D96)."""
    task_id: str
    covered: bool = False
    reason: Optional[str] = None
    run_count: int = 0


class SetAsideLesson(Record):
    """A cross-customer lesson the Builder judged irrelevant here, listed so a wrong call is visible (D87)."""
    id: str = ""
    pattern: str = ""
    reason: str = ""


class ReportData(BaseModel):
    """Everything the report reads, already on disk as records; nothing here is computed by a model."""
    title: str = "Harness build report"
    kind: Literal["build", "batch"] = "build"
    built: bool = False
    stopped_reason: Optional[str] = None
    stopped: dict = Field(default_factory=dict)
    records_not_read: list[str] = Field(default_factory=list)
    environment: Optional[Environment] = None
    gates: list[GateResult] = Field(default_factory=list)
    scorecard: list[ScorecardItem] = Field(default_factory=list)
    stages: list[StageStatus] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    verifiers: list[Verifier] = Field(default_factory=list)
    tool_sigs: list[ToolSig] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    overlays: list[TaskOverlay] = Field(default_factory=list)
    policy_items: list[Constraint] = Field(default_factory=list)
    policy_exercised: list[str] = Field(default_factory=list)
    task_coverage: list[TaskCoverage] = Field(default_factory=list)
    frontier_models: list[str] = Field(default_factory=list)
    assisted_share: dict[str, float] = Field(default_factory=dict)
    judge_disagreement: dict = Field(default_factory=dict)
    audit_rate: Optional[float] = None
    disagreement_queue: list[dict] = Field(default_factory=list)
    tasks_aside: list[dict] = Field(default_factory=list)
    lessons_set_aside: list[SetAsideLesson] = Field(default_factory=list)


# --- numbers ---------------------------------------------------------------

def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _rate(passed: int, total: int) -> Optional[float]:
    return (passed / total) if total else None


def _tools_called(run: Optional[Run]) -> set[str]:
    """Every tool one Run called, read off its events."""
    if run is None:
        return set()
    return {str((e.payload or {}).get("name") or "") for e in run.events if e.type == "tool_call"} - {""}


def flagged_tool_verdicts(data: ReportData) -> dict[str, int]:
    """Per flagged tool (D70 `unclassified`), how many Verdicts rest on a Run that called it.

    D70 asks for this by name: a customer should see how much of a pass rate rests on tools nobody
    has confirmed as read or write.
    """
    flagged = {sig.name for sig in data.tool_sigs if sig.unclassified}
    if not flagged:
        return {}
    runs = {run.run_id: run for run in data.runs}
    counts: dict[str, int] = {name: 0 for name in sorted(flagged)}
    for record in data.verdicts:
        for name in _tools_called(runs.get(record.run_id)) & flagged:
            counts[name] += 1
    return counts


def overlay_rows(data: ReportData, task_id: str) -> int:
    """How many of this Task's rows are overlay rows (D74), stated per Task as the decision asks."""
    return sum(len(o.rows) for o in data.overlays if o.task_id == task_id)


def version_match(data: ReportData, task: Task, record: Verdict) -> int:
    """How well one Verdict's versions match the Environment and Verifier on disk now (design section 8).

    A Verdict is content-addressed per version, so one Run can hold several Verdict files. The
    highest score is the one this build's records were graded under; a version the Verdict left
    empty neither helps nor hurts.
    """
    verifier = next((v for v in data.verifiers if v.task_id == task.id), None)
    wanted = {
        "verifier_version": getattr(verifier, "verifier_version", None),
        "env_id": getattr(data.environment, "env_id", None),
        "schema_version": getattr(data.environment, "schema_version", None),
        "tools_version": getattr(data.environment, "tools_version", None),
        "policy_version": getattr(data.environment, "policy_version", None),
    }
    score = 0
    for name, value in wanted.items():
        mine = getattr(record, name, None)
        if value is None or mine is None:
            continue
        score += 1 if mine == value else -1
    return score


def current_verdicts(data: ReportData, task: Task, verdicts: list[Verdict]) -> tuple[list[Verdict], list[Verdict]]:
    """One Verdict per Run: the one graded under the current versions, and the ones it supersedes.

    Counting both a Verdict and its regraded twin doubles Runs graded and skews every pass rate, so
    the older file is dropped and named instead. Ties go to the last file read, which is the last in
    sorted order under `verdicts/`.
    """
    best: dict[str, Verdict] = {}
    superseded: list[Verdict] = []
    for record in verdicts:
        seen = best.get(record.run_id)
        if seen is None or version_match(data, task, record) >= version_match(data, task, seen):
            if seen is not None:
                superseded.append(seen)
            best[record.run_id] = record
        else:
            superseded.append(record)
    return [best[run_id] for run_id in best], superseded


def assisted_tools_of(run: Optional[Run]) -> list[str]:
    """The tools that stood in on one Run (D49), read off the events the loop marked assisted."""
    if run is None:
        return []
    names = {str((e.payload or {}).get("name") or "") for e in run.events if e.assisted}
    return sorted(names - {""})


def assisted_share_from_runs(runs: list[Run]) -> dict[str, float]:
    """Per tool, the share of its calls a stand-in answered (D49), counted off the Runs themselves."""
    total: dict[str, int] = {}
    assisted: dict[str, int] = {}
    for run in runs:
        for event in run.events:
            if event.type != "tool_result":
                continue
            name = str((event.payload or {}).get("name") or "")
            if not name:
                continue
            total[name] = total.get(name, 0) + 1
            if event.assisted:
                assisted[name] = assisted.get(name, 0) + 1
    return {name: assisted.get(name, 0) / count for name, count in sorted(total.items()) if count}


def task_numbers(data: ReportData, task: Task) -> dict:
    """Count one Task's stored Verdicts. Assisted Runs and environment-suspected Runs are not counted (D49, D88)."""
    runs = {run.run_id: run for run in data.runs}
    members = set(task.run_ids) | {r.run_id for r in data.runs if r.task_id == task.id}
    verdicts, superseded = current_verdicts(data, task, [v for v in data.verdicts if v.run_id in members])
    graded, uncounted = [], []
    for record in verdicts:
        run = runs.get(record.run_id)
        if record.environment_suspected or (run is not None and run.assisted):
            uncounted.append(record)
        else:
            graded.append(record)
    not_counted = len(uncounted)

    def split(frontier: bool) -> list[Verdict]:
        out = []
        for record in graded:
            run = runs.get(record.run_id)
            model = run.model if run is not None else None
            if (model in data.frontier_models) is frontier:
                out.append(record)
        return out

    frontier, candidate = split(True), split(False)
    frontier_rate = _rate(sum(1 for v in frontier if v.passed), len(frontier))
    candidate_rate = _rate(sum(1 for v in candidate if v.passed), len(candidate))
    kinds = {a.id: a.kind for v in data.verifiers if v.task_id == task.id for a in v.atoms}
    failing: dict[str, int] = {}
    causes: dict[str, int] = {}
    for record in graded:
        if record.failing_atom:
            kind = kinds.get(record.failing_atom, "unknown")
            failing[kind] = failing.get(kind, 0) + 1
        if not record.passed and record.cause:
            causes[record.cause] = causes.get(record.cause, 0) + 1
    return {
        "runs_graded": len(graded),
        "assisted_not_counted": not_counted,
        "overlay_rows": overlay_rows(data, task.id),
        "judge_atoms": sum(1 for v in data.verifiers if v.task_id == task.id for a in v.atoms if a.judge),
        "judge_disagreement_rate": data.judge_disagreement.get("rate"),
        "frontier_runs": len(frontier),
        "frontier_pass_rate": frontier_rate,
        "candidate_runs": len(candidate),
        "candidate_pass_rate": candidate_rate,
        "margin": None if (frontier_rate is None or candidate_rate is None) else candidate_rate - frontier_rate,
        "failing_atoms": failing,
        "causes": causes,
        "superseded": len(superseded),
        "counted": graded,
        "uncounted": uncounted,
        "atom_kinds": kinds,
    }


def suggestion(numbers: dict, built: bool, aside: Optional[str] = None) -> str:
    """A suggestion, never a decision (D85). The person routing the Task is the one who decides."""
    tail = " The decision is yours."
    if aside:
        return f"Suggestion: this Task is not gradeable, Reference disputed ({aside}); it comes back once a person resolves it." + tail
    if not built:
        return "Suggestion: the Environment was not built, so the numbers cannot support a routing decision yet." + tail
    if not numbers.get("runs_graded"):
        return "Suggestion: no Runs were graded for this Task, so the numbers say nothing either way." + tail
    margin = numbers.get("margin")
    if margin is None:
        return "Suggestion: there is no frontier Run to compare against, so the numbers say nothing either way." + tail
    verb = "support" if margin >= 0 else "do not support"
    return f"Suggestion: the numbers {verb} routing this Task to the Candidate (margin {margin:+.2f})." + tail


# --- the pipeline DAG ------------------------------------------------------

def node_id(name: str) -> str:
    """A mermaid node id from a stage name: a space or a hyphen would end the id and break the graph."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "stage"


def pipeline_dag(stages: list[StageStatus]) -> str:
    """The pipeline as mermaid, in the order the run recorded it, with a rollback edge per failed gate."""
    if not stages:
        return 'flowchart TD\n    empty["no stages recorded"]'
    lines = ["flowchart TD"]
    for stage in stages:
        lines.append(f'    {node_id(stage.name)}["{stage.name} ({stage.status})"]')
    for before, after in zip(stages, stages[1:]):
        lines.append(f"    {node_id(before.name)} --> {node_id(after.name)}")
    for stage in stages:
        if stage.status == "failed" and stage.gate:
            attempts = max(stage.attempts, 1)
            allowed = max(stage.max_attempts, attempts)
            node = node_id(stage.name)
            lines.append(
                f'    {node} -. "gate {stage.gate} failed, attempt {attempts} of {allowed}" .-> {node}'
            )
    return "\n".join(lines)


# --- the sections ----------------------------------------------------------

ENVIRONMENT_GATE_NAMES = ("build_environment", "environment", "build_env")


def _gate_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (name or "").strip().lower())


def environment_gate(data: ReportData) -> Optional[GateResult]:
    """The build Environment gate (design section 6), the one that decides whether it was built."""
    for gate in data.gates:
        if _gate_key(gate.stage) in ENVIRONMENT_GATE_NAMES:
            return gate
    return None


def _usd(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.2f}"


def _stop_lines(data: ReportData) -> list[str]:
    """Where the build stopped, what it spent and what finishing costs (D86)."""
    stop = data.stopped
    if not stop:
        return []
    done = [s.name for s in data.stages if s.status in ("ran", "cached", "rolled_back")]
    left = [s.name for s in data.stages if s.status in ("pending", "stopped", "failed")]
    lines = [
        f"Stopped in stage {stop.get('stage') or 'unknown'} on {stop.get('item') or 'no item named'}.",
        f"Completed stages: {', '.join(done) or 'none'}. Still to do: {', '.join(left) or 'none'}.",
        f"Spent {_usd(stop.get('spent'))} against a ceiling of {_usd(stop.get('ceiling_usd'))}; "
        f"about {_usd(stop.get('estimate_to_finish'))} more to finish.",
    ]
    if stop.get("reason"):
        lines.append(f"Reason: {stop['reason']}.")
    lines.append("Continuing needs a person's permission, given as a new ceiling (D86).")
    return lines


def _environment(data: ReportData) -> list[str]:
    lines = [SECTIONS[0], ""]
    lines.append(f"Environment built: {'yes' if data.built else 'no'}.")
    gate = environment_gate(data)
    if gate is not None:
        lines.append(
            f"The build Environment gate {'passed' if gate.passed else 'failed'}, which is what decides that"
            + (f": {'; '.join(gate.failures)}" if gate.failures else "") + "."
        )
    else:
        lines.append("No build Environment gate was recorded, so this reads the Environment file and "
                     "the pipeline status instead.")
    env = data.environment
    if env is not None:
        lines.append(
            f"Environment {env.env_id}, version {env.version} "
            f"(schema {env.schema_version}, tools {env.tools_version}, policy {env.policy_version})."
        )
    if data.stopped_reason:
        lines.append(f"Stopped: {data.stopped_reason}.")
    lines += _stop_lines(data)
    if data.records_not_read:
        lines += ["", "### Records not read", "",
                  "These files are on disk and did not load, so every number below is counted without them."]
        lines += [f"- {name}" for name in data.records_not_read]
    lines += ["", "### Gates", "", "| stage | result | failures |", "| --- | --- | --- |"]
    for gate in data.gates or []:
        lines.append(f"| {gate.stage} | {'pass' if gate.passed else 'fail'} | {'; '.join(gate.failures)} |")
    if not data.gates:
        lines.append("| none recorded |  |  |")

    lines += ["", "### Scorecard, raw and explained side by side", "",
              "| number | raw | explained | note |", "| --- | --- | --- | --- |"]
    for item in data.scorecard or []:
        lines.append(f"| {item.name} | {item.raw} | {item.explained} | {item.note} |")
    if not data.scorecard:
        lines.append("| none recorded |  |  |  |")

    assisted = list(env.assisted_tools) if env is not None else []
    lines += ["", "### Assisted tools", ""]
    lines += [
        f"- {name}" + (f": {_percent(data.assisted_share[name])} of its calls stood in"
                       if name in data.assisted_share else "")
        for name in assisted
    ] or ["No assisted tools: every tool here is real code."]

    unguarded = [t for t in data.tasks if t.unguarded]
    lines += ["", "### Unguarded Tasks", ""]
    lines += [f"- {t.id}: {t.name or t.intent or 'no name yet'}" for t in unguarded] or [
        "No unguarded Tasks: every Task held Runs back for the anchor."]

    flagged = flagged_tool_verdicts(data)
    lines += ["", "### Flagged tools, read or write not confirmed (D70)", ""]
    lines += [f"- {name}: {count} Verdicts rest on a Run that called it" for name, count in flagged.items()] or [
        "No flagged tools: every tool here is confirmed read or write."]

    lines += ["", "### Open flags on the Environment", ""]
    lines += [f"- {flag}" for flag in (env.flags if env is not None else [])] or [
        "No open flags: nothing on the Environment is waiting on the setup review."]

    rows = sum(len(o.rows) for o in data.overlays)
    lines += ["", "### Overlays", "",
              f"- Tasks with an overlay: {len(data.overlays)}", f"- Overlay rows: {rows}"]
    lines += [f"  - {o.task_id}: {len(o.rows)} overlay rows" for o in data.overlays]

    lines += ["", "### Coverage", ""] + _coverage(data)
    lines += ["", "### Pipeline", "", "```mermaid", pipeline_dag(data.stages), "```"]
    cost = sum(stage.usd for stage in data.stages)
    if cost:
        lines += ["", "Cost per stage: " + ", ".join(f"{s.name} ${s.usd:.2f}" for s in data.stages if s.usd),
                  f"Cost so far: ${cost:.2f}."]
    else:
        lines += ["", "Cost per stage: nothing recorded, so this build's spend is not known."]
    return lines


def _coverage(data: ReportData) -> list[str]:
    covered = [c for c in data.task_coverage if c.covered]
    total_runs = sum(c.run_count for c in data.task_coverage)
    covered_runs = sum(c.run_count for c in covered)
    share = _percent(_rate(covered_runs, total_runs))
    lines = [f"- Task coverage: {len(covered)} of {len(data.task_coverage)} Tasks covered, {share} of Runs."]
    for item in data.task_coverage:
        if not item.covered:
            lines.append(f"  - {item.task_id}: not covered, {item.reason or 'no reason recorded'}")
    exercised = [c for c in data.policy_items if c.id in set(data.policy_exercised)]
    untested = [c for c in data.policy_items if c.id not in set(data.policy_exercised)]
    lines.append(
        f"- Policy coverage: your traces exercise {len(exercised)} of {len(data.policy_items)} "
        "policy items; the rest are not tested."
    )
    for item in untested:
        lines.append(f"  - {item.id}: {item.text}")
    return lines


def _tasks(data: ReportData) -> list[str]:
    aside = {str(row.get("task_id")): str(row.get("reason", "disputed")) for row in data.tasks_aside}
    lines = [SECTIONS[1], ""]
    if data.kind == "batch":
        lines += ["These are the numbers for one Run batch against the Environment above.", ""]
    if not data.tasks:
        lines += ["No Tasks in this build.", ""]
    for task in data.tasks:
        numbers = task_numbers(data, task)
        lines.append(f"### Task {task.id}: {task.name or task.intent or 'no name yet'}")
        lines.append("")
        if task.intent:
            lines.append(f"Intent: {task.intent}")
            lines.append("")
        if task.id in aside:
            lines.append(f"Not gradeable, Reference disputed ({aside[task.id]}).")
            lines.append("")
        lines += _task_numbers_lines(data, numbers)
        lines += ["", suggestion(numbers, data.built, aside.get(task.id)), ""]
    return lines


def _task_numbers_lines(data: ReportData, numbers: dict) -> list[str]:
    judge_note = f"judge disagreement rate {_percent(numbers['judge_disagreement_rate'])}"
    judge_note += (f", audit rate {_percent(data.audit_rate)}" if data.audit_rate is not None
                   else ", no human labels yet, so no error bound")
    lines = [
        f"- Runs graded: {numbers['runs_graded']}",
        f"- Runs not counted (assisted or environment suspected): {numbers['assisted_not_counted']}",
        f"- Overlay rows in this Task's Starting state: {numbers['overlay_rows']}",
        f"- Judge atoms: {numbers['judge_atoms']} ({judge_note})",
        f"- Frontier pass rate: {_percent(numbers['frontier_pass_rate'])} ({numbers['frontier_runs']} Runs)",
        f"- Candidate pass rate: {_percent(numbers['candidate_pass_rate'])} ({numbers['candidate_runs']} Runs)",
        f"- Margin: {'n/a' if numbers['margin'] is None else format(numbers['margin'], '+.2f')}",
        "- Failing atoms by class: " + (_counts(numbers["failing_atoms"]) or "none"),
        "- Causes: " + (_counts(numbers["causes"]) or "none"),
    ]
    return lines


def _counts(counts: dict) -> str:
    return ", ".join(f"{name} {number}" for name, number in sorted(counts.items()))


def _queue(data: ReportData) -> list[str]:
    lines = [SECTIONS[2], ""]
    pairs = data.judge_disagreement.get("pairs", 0)
    disagreements = data.judge_disagreement.get("disagreements", len(data.disagreement_queue))
    lines.append(
        f"Judge disagreement: {disagreements} of {pairs} pairs "
        f"({_percent(data.judge_disagreement.get('rate'))}). No human labels yet, so this number has no error bound."
    )
    lines.append("")
    if not data.disagreement_queue and not data.tasks_aside:
        lines.append("The judges agreed everywhere they were asked: nothing in the queue, no Tasks set aside.")
        return lines
    if data.disagreement_queue:
        lines += ["### Items a person may resolve", ""]
        for row in data.disagreement_queue:
            lines.append(
                f"- {row.get('use', 'judge')} on {row.get('item_id', 'unknown')}: "
                f"one judge said {row.get('verdict_a', 'unknown')}, the other said {row.get('verdict_b', 'unknown')}"
            )
        lines.append("")
    if data.tasks_aside:
        lines += ["### Tasks set aside, not gradeable", ""]
        for row in data.tasks_aside:
            lines.append(
                f"- {row.get('task_id', 'unknown')}: {row.get('reason', 'disputed')}, "
                "not gradeable until a person resolves it"
            )
        lines.append("")
    return lines


def _lessons(data: ReportData) -> list[str]:
    lines = [SECTIONS[3], ""]
    if not data.lessons_set_aside:
        lines.append("The Builder applied every lesson it carried: no lessons were set aside for this build.")
        return lines
    lines.append("Lessons the Builder judged not relevant here. A wrongly discarded lesson is visible below.")
    lines.append("")
    for lesson in data.lessons_set_aside:
        lines.append(f"- {lesson.id or 'lesson'}: {lesson.pattern} (set aside: {lesson.reason})")
    return lines


def render(data: ReportData) -> str:
    """The whole report as Markdown, in the one order D85 fixes."""
    lines = [f"# {data.title}", ""]
    lines += _environment(data) + [""]
    lines += _tasks(data) + [""]
    lines += _queue(data) + [""]
    lines += _lessons(data) + [""]
    return "\n".join(lines)


def write_report(data: ReportData, workdir: Any, name: str = "report.md") -> Path:
    """Write the report beside the records it read."""
    path = Path(workdir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data), encoding="utf-8")
    return path


# --- reading the records off disk ------------------------------------------

def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in (path.read_text(encoding="utf-8").splitlines() if path.is_file() else []):
        value = json.loads(line) if line.strip() else None
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _records(folder: Path, model: type) -> list:
    """Every record of one kind under a folder.

    A writer may keep the record beside other data in the same file (compile_env.py writes an
    overlay as {"overlay": ..., "values": ...}), so a file that is not the record itself is searched
    one level down for it. A file that holds neither is skipped, never raised on.
    """
    out = []
    for path in sorted(folder.rglob("*.json")) if folder.is_dir() else []:
        body = _json(path)
        if not isinstance(body, dict):
            continue
        for candidate in [body] + [v for v in body.values() if isinstance(v, dict)]:
            try:
                out.append(model.model_validate(candidate))
                break
            except ValidationError:
                continue
    return out


def _list_of(path: Path, model: type) -> list:
    body = _json(path)
    return [model.model_validate(item) for item in body] if isinstance(body, list) else []


def coverage_rows(tasks: list[Task], uncovered: dict[str, str]) -> list[TaskCoverage]:
    """The D96 rows from a Task list and the first failing reason per uncovered Task.

    The rule itself is validate.py's `task_coverage`, so the scorecard and the report cannot drift
    apart; cli.py applies it and hands the reasons here, which keeps the Runner out of this module.
    """
    return [TaskCoverage(task_id=task.id, covered=task.id not in uncovered,
                         reason=uncovered.get(task.id), run_count=len(task.run_ids)) for task in tasks]


def load_tool_sigs(workdir: Any) -> list[ToolSig]:
    """The mined ToolSigs of a build: tool_sigs.json when a stage wrote one list, else one file each.

    cli.py reads them too, so a Verdict knows which tools write (extra-write and entity-count checks)
    and which are still flagged (D70), and the report and the Verdict cannot disagree about it.
    """
    root = Path(workdir)
    return _list_of(root / "tool_sigs.json", ToolSig) or _records(root / "tool_sigs", ToolSig)


def load(workdir: Any) -> ReportData:
    """Read every record the report shows from one workdir. Missing files mean a shorter report, not an error."""
    root = Path(workdir)
    env_body = _json(root / "environment.json")
    environment = Environment.model_validate(env_body) if isinstance(env_body, dict) else None
    state = _json(root / "pipeline" / "state.json") or {}
    statuses = state.get("statuses", {}) if isinstance(state, dict) else {}
    stages = [StageStatus(name=name, status=str(status)) for name, status in statuses.items()]
    for row in state.get("log", []) if isinstance(state, dict) else []:
        for stage in stages:
            if row.get("stage") == stage.name:
                stage.gate = row.get("gate") or stage.gate
                stage.attempts = max(stage.attempts, int(row.get("attempt") or 0))
    config = _json(root / "report_config.json") or {}
    pairs = _jsonl(root / "judge_pairs.jsonl")
    disagreements = sum(1 for row in pairs if row.get("disagreement"))
    tasks = _records(root / "tasks", Task)
    runs = _records(root / "runs", Run)
    return ReportData(
        title=config.get("title") or ("Run batch report" if config.get("kind") == "batch" else "Harness build report"),
        kind=config.get("kind", "build"),
        built=environment is not None and state.get("status", "complete") not in ("failed", "stopped"),
        stopped_reason=state.get("stopped_reason") or config.get("stopped_reason"),
        environment=environment,
        gates=_list_of(root / "gates.json", GateResult) + _records(root / "gates", GateResult),
        scorecard=_list_of(root / "scorecard.json", ScorecardItem),
        stages=stages,
        tasks=tasks,
        verifiers=_records(root / "verifiers", Verifier),
        tool_sigs=load_tool_sigs(root),
        runs=runs,
        verdicts=_records(root / "verdicts", Verdict),
        overlays=_records(root / "overlays", TaskOverlay),
        policy_items=_list_of(root / "constraints.json", Constraint) + _records(root / "constraints", Constraint),
        policy_exercised=(_json(root / "policy_coverage.json") or {}).get("exercised", []),
        task_coverage=_list_of(root / "coverage.json", TaskCoverage),
        frontier_models=config.get("frontier_models", []),
        assisted_share=config.get("assisted_share", {}),
        judge_disagreement={"pairs": len(pairs), "disagreements": disagreements,
                            "rate": (disagreements / len(pairs)) if pairs else None},
        audit_rate=config.get("audit_rate"),
        disagreement_queue=_jsonl(root / "disagreement_queue.jsonl"),
        tasks_aside=_jsonl(root / "tasks_aside.jsonl"),
        lessons_set_aside=_list_of(root / "lessons_set_aside.json", SetAsideLesson),
    )
