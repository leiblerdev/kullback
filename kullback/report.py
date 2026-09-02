"""The face of the Harness: one Markdown report that opens with whether the Environment was built, then the
numbers per Task, then a suggestion the person decides on (D85); it reads records and never computes a Verdict."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from kullback.runner.records import (
    Constraint,
    Environment,
    GateResult,
    Record,
    Run,
    SetAsideLesson,
    Task,
    TaskOverlay,
    ToolSig,
    Verdict,
    Verifier,
    disagreement_stats,
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
    cache_share: float = 0.0  # share of input tokens the provider served from its prompt cache
    memo_hits: int = 0  # calls answered from the on-disk memo, never sent to a provider


class TaskCoverage(Record):
    """Whether one Task is covered, and the first reason it is not (D96)."""
    task_id: str
    covered: bool = False
    reason: Optional[str] = None
    run_count: int = 0


# SetAsideLesson is imported from records.py above and re-exported here: every record lives in
# records.py (build brief rule 4), and this is the name the report's own field is typed with.


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


def _wanted_versions(data: ReportData, task: Task) -> dict[str, Optional[str]]:
    """The versions on disk now that a Verdict of this Task is scored against. Constant per Task."""
    verifier = next((v for v in data.verifiers if v.task_id == task.id), None)
    return {
        "verifier_version": getattr(verifier, "verifier_version", None),
        "env_id": getattr(data.environment, "env_id", None),
        "schema_version": getattr(data.environment, "schema_version", None),
        "tools_version": getattr(data.environment, "tools_version", None),
        "policy_version": getattr(data.environment, "policy_version", None),
    }


def _version_score(record: Verdict, wanted: dict[str, Optional[str]]) -> int:
    score = 0
    for name, value in wanted.items():
        mine = getattr(record, name, None)
        if value is None or mine is None:
            continue
        score += 1 if mine == value else -1
    return score


def version_match(data: ReportData, task: Task, record: Verdict) -> int:
    """How well one Verdict's versions match the Environment and Verifier on disk now (design section 8).

    A Verdict is content-addressed per version, so one Run can hold several Verdict files. The
    highest score is the one this build's records were graded under; a version the Verdict left
    empty neither helps nor hurts.
    """
    return _version_score(record, _wanted_versions(data, task))


def current_verdicts(data: ReportData, task: Task, verdicts: list[Verdict]) -> tuple[list[Verdict], list[Verdict]]:
    """One Verdict per Run: the one graded under the current versions, and the ones it supersedes.

    Counting both a Verdict and its regraded twin doubles Runs graded and skews every pass rate, so
    the older file is dropped and named instead. Ties go to the last file read, which is the last in
    sorted order under `verdicts/`.
    """
    wanted = _wanted_versions(data, task)
    best: dict[str, Verdict] = {}
    superseded: list[Verdict] = []
    for record in verdicts:
        seen = best.get(record.run_id)
        if seen is None or _version_score(record, wanted) >= _version_score(seen, wanted):
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
        "runs_by_id": runs,
        "atom_kinds": kinds,
    }


def suggestion(numbers: dict, built: bool, aside: Optional[str] = None) -> str:
    """A suggestion, never a decision (D85). The person routing the Task is the one who decides."""
    tail = " The decision is yours."
    if aside:
        return (f"Suggestion: this Task is not gradeable, Reference disputed ({aside}); it comes back once a "
                "person resolves it.") + tail
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

def _cell(text: str) -> str:
    """One Markdown table cell: a bar inside a gate failure or a scorecard note would end the cell
    and shift every column after it, and those strings come from the customer's own artifacts."""
    return str(text).replace("|", "\\|")


def node_id(name: str) -> str:
    """A mermaid node id from a stage name: a space or a hyphen would end the id and break the graph."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "stage"


def pipeline_dag(stages: list[StageStatus]) -> str:
    """The pipeline as mermaid, in the order the run recorded it, with a rollback edge per failed gate."""
    if not stages:
        return 'flowchart TD\n    empty["no stages recorded"]'
    lines = ["flowchart TD"]
    for stage in stages:
        label = f"{stage.name} ({stage.status})".replace('"', "'")
        lines.append(f'    {node_id(stage.name)}["{label}"]')
    for before, after in zip(stages, stages[1:], strict=False):
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
        f"the estimated cost to finish is {_usd(stop.get('estimate_to_finish'))}.",
    ]
    if stop.get("reason"):
        lines.append(f"Reason: {stop['reason']}.")
    lines.append("Continuing needs a person's permission, given as a new ceiling (D86).")
    return lines


def _bullets(items: list[str], empty: str) -> list[str]:
    """A bulleted subsection, or the one sentence that says there is nothing in it."""
    return items or [empty]


def _headline(data: ReportData) -> list[str]:
    """Built or not built, what decided that, and what stopped the build if anything did."""
    lines = [f"Environment built: {'yes' if data.built else 'no'}."]
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
    return lines


def _gates_table(data: ReportData) -> list[str]:
    lines = ["", "### Gates", "", "| stage | result | failures |", "| --- | --- | --- |"]
    for gate in data.gates or []:
        lines.append(f"| {_cell(gate.stage)} | {'pass' if gate.passed else 'fail'} | "
                     f"{_cell('; '.join(gate.failures))} |")
    return lines if data.gates else lines + ["| none recorded |  |  |"]


def _scorecard_table(data: ReportData) -> list[str]:
    lines = ["", "### Scorecard, raw and explained side by side", "",
             "| number | raw | explained | note |", "| --- | --- | --- | --- |"]
    for item in data.scorecard or []:
        lines.append(f"| {_cell(item.name)} | {_cell(item.raw)} | {_cell(item.explained)} | "
                     f"{_cell(item.note)} |")
    return lines if data.scorecard else lines + ["| none recorded |  |  |  |"]


def _tool_notes(data: ReportData) -> list[str]:
    """What a person has to look at before trusting the numbers: tools that stood in, Tasks with
    no anchor, tools nobody classed read or write (D70), and the Environment's open flags."""
    env = data.environment
    assisted = list(env.assisted_tools) if env is not None else []
    lines = ["", "### Assisted tools", ""]
    lines += _bullets(
        [f"- {name}" + (f": {_percent(data.assisted_share[name])} of its calls stood in"
                        if name in data.assisted_share else "")
         for name in assisted],
        "No assisted tools: every tool here is real code.")

    lines += ["", "### Unguarded Tasks", ""]
    lines += _bullets([f"- {t.id}: {t.name or t.intent or 'no name yet'}" for t in data.tasks if t.unguarded],
                      "No unguarded Tasks: every Task held Runs back for the anchor.")

    lines += ["", "### Flagged tools, read or write not confirmed (D70)", ""]
    lines += _bullets([f"- {name}: {count} Verdicts rest on a Run that called it"
                       for name, count in flagged_tool_verdicts(data).items()],
                      "No flagged tools: every tool here is confirmed read or write.")

    lines += ["", "### Open flags on the Environment", ""]
    lines += _bullets([f"- {flag}" for flag in (env.flags if env is not None else [])],
                      "No open flags: nothing on the Environment is waiting on the setup review.")
    return lines


def _overlay_lines(data: ReportData) -> list[str]:
    rows = sum(len(o.rows) for o in data.overlays)
    return ["", "### Overlays", "",
            f"- Tasks with an overlay: {len(data.overlays)}", f"- Overlay rows: {rows}"] + [
        f"  - {o.task_id}: {len(o.rows)} overlay rows" for o in data.overlays]


def _pipeline_lines(data: ReportData) -> list[str]:
    lines = ["", "### Pipeline", "", "```mermaid", pipeline_dag(data.stages), "```"]
    cost = sum(stage.usd for stage in data.stages)
    if not cost:
        return lines + ["", "Cost per stage: nothing recorded, so this build's spend is not known."]
    lines += ["", "Cost per stage: " + ", ".join(_cost_cell(s) for s in data.stages if s.usd),
              f"Cost so far: ${cost:.2f}."]
    memo = sum(s.memo_hits for s in data.stages)
    if memo:
        lines += [f"Calls answered from the memo, at no cost: {memo}."]
    return lines


def _environment(data: ReportData) -> list[str]:
    """The Environment section, in the order a person reads it: what was built, what the gates and
    the scorecard said, what still needs a look, and what the pipeline did and cost."""
    return ([SECTIONS[0], ""] + _headline(data) + _gates_table(data) + _scorecard_table(data)
            + _tool_notes(data) + _overlay_lines(data)
            + ["", "### Coverage", ""] + _coverage(data) + _pipeline_lines(data))


def _cost_cell(stage: StageStatus) -> str:
    cell = f"{stage.name} ${stage.usd:.2f}"
    if stage.cache_share:
        cell += f" ({_percent(stage.cache_share)} of input from cache)"
    return cell


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
    lines += _pending_review(data)
    return lines


def _pending_review(data: ReportData) -> list[str]:
    """Rewritten rules waiting for the setup review (D76, D48).

    A rule the Builder rewrote but nobody accepted is in no Verifier and in no residual list, so
    without this block nothing tells the reviewer it exists. The filter is policy.pending_review's,
    repeated over the records because report.py reads records off disk rather than importing builder/.
    """
    pending = [c for c in data.policy_items
               if c.rewritten_text and not c.compiled and not c.judge_atom and not c.residual_reason]
    if not pending:
        return []
    lines = [f"- Awaiting setup review, not checked: {len(pending)} rewritten policy "
             f"{'rule' if len(pending) == 1 else 'rules'}. Until a person accepts the rewrite, each "
             "is in no Verifier and in no residual list, so nothing checks it."]
    for item in pending:
        lines.append(f"  - {item.id}: {item.text}")
        lines.append(f"    rewritten as: {item.rewritten_text}")
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


def _path_words(record: Verdict) -> str:
    """D46: a Verdict says whether the Run reached the End state on the Reference's path or another one."""
    if record.same_path is None:
        return "path not recorded"
    return "same path as the Reference" if record.same_path else "different path from the Reference"


def _verdict_line(record: Verdict, kinds: dict) -> str:
    """One counted Verdict as the design words it: pass or fail, the failing atom, the path, the cause."""
    atom = (f"failing atom {record.failing_atom} ({kinds.get(record.failing_atom, 'unknown')})"
            if record.failing_atom else "no failing atom")
    cause = f"cause {record.cause}" if record.cause else "no cause recorded"
    return (f"  - {record.run_id}: {'pass' if record.passed else 'fail'}, {atom}, "
            f"{_path_words(record)}, {cause}")


def _uncounted_line(run: Optional[Run], record: Verdict) -> str:
    """One Verdict left out of the numbers, with the mark or the tool that excluded it (D49, D88)."""
    reasons = []
    if record.environment_suspected:
        reasons.append("environment suspected" + (f", cause {record.cause}" if record.cause else ""))
    if run is not None and run.assisted:
        stood_in = assisted_tools_of(run)
        reasons.append("assisted Run" + (f", {', '.join(stood_in)} stood in" if stood_in
                                         else ", a tool stood in"))
    return f"  - {record.run_id}: {'; '.join(reasons) or 'not counted'}"


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
    if numbers.get("superseded"):
        lines.append(f"- Superseded Verdicts not counted (an older Verifier or Environment version): "
                     f"{numbers['superseded']}; each Run is counted once, under the versions on disk now")
    kinds = numbers.get("atom_kinds") or {}
    counted = numbers.get("counted") or []
    lines.append("- Verdicts counted, one line each:" if counted else "- Verdicts counted, one line each: none")
    lines += [_verdict_line(record, kinds) for record in counted]
    uncounted = numbers.get("uncounted") or []
    if uncounted:
        lines.append("- Verdicts not counted, and what excluded them:")
        by_id = numbers.get("runs_by_id") or {}
        lines += [_uncounted_line(by_id.get(record.run_id), record) for record in uncounted]
    return lines


def _counts(counts: dict) -> str:
    return ", ".join(f"{name} {number}" for name, number in sorted(counts.items()))


def _queue(data: ReportData) -> list[str]:
    lines = [SECTIONS[2], ""]
    pairs = data.judge_disagreement.get("pairs", 0)
    disagreements = data.judge_disagreement.get("disagreements", len(data.disagreement_queue))
    bound = (f" Audit rate {_percent(data.audit_rate)} from the queue items a person has resolved, "
             "which is the labelled set this number is bounded by (D92)."
             if data.audit_rate is not None
             else " No human labels yet, so this number has no error bound.")
    lines.append(
        f"Judge disagreement: {disagreements} of {pairs} pairs "
        f"({_percent(data.judge_disagreement.get('rate'))})." + bound
    )
    abstains = data.judge_disagreement.get("abstains")
    if abstains is not None:
        lines.append(
            f"Judge abstention: {abstains} of {pairs} pairs "
            f"({_percent(data.judge_disagreement.get('abstain_rate'))}). An abstain decides nothing, "
            "so D92 sends it to a person on the same terms as a split."
        )
    lines.append("")
    if not data.disagreement_queue and not data.tasks_aside:
        lines.append("The judges agreed everywhere they were asked: nothing in the queue, no Tasks set aside.")
        return lines
    split, undecided = _queue_split(data.disagreement_queue)
    for heading, rows in (("### Items a person may resolve", split),
                          ("### Items the judges did not decide", undecided)):
        if not rows:
            continue
        lines += [heading, ""]
        for row in rows:
            third = f", a third sample said {row['verdict_c']}" if row.get("verdict_c") else ""
            lines.append(
                f"- {row.get('use', 'judge')} on {row.get('item_id', 'unknown')}: "
                f"one judge said {row.get('verdict_a', 'unknown')}, "
                f"the other said {row.get('verdict_b', 'unknown')}" + third
                + _queue_reason_words(row)
            )
            lines += _cited_spans(row)
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


_QUEUE_REASONS = {
    "refused": ", and both refused for want of a tool check",
    "agreed_abstain": ", and both abstained",
    "abstain_majority": ", and the majority of the three abstained",
}


def _queue_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The queue in its two halves: judges that split, and judges that decided nothing (D92, D88).

    A row with no reason predates the reason field and is read as a split, which is what the two
    verdicts on it say.
    """
    split = [r for r in rows if (r.get("reason") or "split") == "split"]
    return split, [r for r in rows if (r.get("reason") or "split") != "split"]


def _queue_reason_words(row: dict) -> str:
    return _QUEUE_REASONS.get(str(row.get("reason") or ""), "")


def _cited_spans(row: dict) -> list[str]:
    """The spans each judge cited, which is what D92 puts in the queue beside the two verdicts."""
    lines = []
    for key, who in (("judge_a", "first judge"), ("judge_b", "second judge"), ("judge_c", "third sample")):
        side = row.get(key)
        spans = list((side or {}).get("cited_spans") or []) if isinstance(side, dict) else []
        for span in spans:
            lines.append(f"  - {who} cited: {span}")
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

EVENT_TYPES = ("model_call", "tool_call", "tool_result", "user_turn", "error", "stop")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path, unread: Optional[list] = None) -> list[dict]:
    """One JSONL file as rows. A line that does not parse is named, never raised on."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines() if path.is_file() else [], 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            _note(unread, f"{path.name} line {number}: not JSON")
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _relative(folder: Path, path: Path) -> str:
    """The file as the workdir names it, so a reader can go and open it."""
    try:
        return str(path.relative_to(folder.parent))
    except ValueError:
        return path.name


def _note(unread: Optional[list], text: str) -> None:
    if unread is not None and text not in unread:
        unread.append(text)


def _records(folder: Path, model: type, unread: Optional[list] = None) -> list:
    """Every record of one kind under a folder.

    A writer may keep the record beside other data in the same file (compile_env.py writes an
    overlay as {"overlay": ..., "values": ...}), so a file that is not the record itself is searched
    one level down for it. A file that holds neither is skipped and, where every file in the folder
    is meant to be this record, named in `unread` so the counts are never quietly short.
    """
    out = []
    for path in sorted(folder.rglob("*.json")) if folder.is_dir() else []:
        body = _json(path)
        if not isinstance(body, dict):
            _note(unread, f"{_relative(folder, path)}: not a JSON object")
            continue
        for candidate in [body] + [v for v in body.values() if isinstance(v, dict)]:
            try:
                out.append(model.model_validate(candidate))
                break
            except ValidationError:
                continue
        else:
            _note(unread, f"{_relative(folder, path)}: not a {model.__name__} this report can read")
    return out


def run_from_jsonl(path: Path) -> Optional[Run]:
    """One stored Run as loop.py writes it: header lines, one line per event, a footer (D90).

    report.py may not import the Runner (design section 4 item 18), so the shape is read again here;
    tests/test_report.py asserts a Run the loop actually wrote still loads.
    """
    head: dict = {}
    events: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            continue
        if obj.get("type") in EVENT_TYPES:
            events.append(obj)
        else:
            events.extend(obj.pop("events", []) or [])
            head.update(obj)
    head = {k: v for k, v in head.items() if k in Run.model_fields}
    head.setdefault("run_id", path.stem)
    head["events"] = [dict(event, idx=event.get("idx", index)) for index, event in enumerate(events)]
    # loop.py marks the Run assisted the moment one event is; the footer does not repeat the flag.
    head["assisted"] = bool(head.get("assisted") or any(event.get("assisted") for event in events))
    try:
        return Run.model_validate(head)
    except ValidationError:
        return None


def load_runs(folder: Path, unread: Optional[list] = None) -> list[Run]:
    """Every stored Run under `runs/`: the JSONL the loop writes, and a Run stored as one JSON file."""
    out: dict[str, Run] = {}
    for path in sorted(folder.rglob("*.jsonl")) if folder.is_dir() else []:
        run = run_from_jsonl(path)
        if run is None:
            _note(unread, f"{_relative(folder, path)}: not a Run JSONL this report can read")
            continue
        out[run.run_id] = run
    for run in _records(folder, Run):
        out.setdefault(run.run_id, run)
    return [out[run_id] for run_id in sorted(out)]


def _rate_row(name: str, part: dict) -> ScorecardItem:
    matched, total = part.get("matched", 0), part.get("total", 0)
    note = f"{matched} of {total} matched"
    if part.get("explained_misses"):
        note += f", {part['explained_misses']} misses explained"
    if part.get("unexplained"):
        note += f", {part['unexplained']} unexplained"
    return ScorecardItem(name=name, raw=_percent(part.get("raw")), explained=_percent(part.get("explained")),
                         note=note)


def scorecard_rows(body: Any) -> list[ScorecardItem]:
    """The scorecard as rows, from a list of ScorecardItem or from gates/scorecard.py's own dict (D62, D80).

    gates/scorecard.py returns one nested dict per build, which is the only scorecard any module produces,
    so the report flattens that shape rather than asking for a second writer.
    """
    if isinstance(body, list):
        out = []
        for item in body:
            try:
                out.append(ScorecardItem.model_validate(item))
            except ValidationError:
                continue
        return out
    if not isinstance(body, dict):
        return []
    rows: list[ScorecardItem] = []
    fidelity = body.get("tool_fidelity") or {}
    for kind in ("success", "error"):
        part = fidelity.get(kind)
        if isinstance(part, dict):
            rows.append(_rate_row(f"replay fidelity, {kind} calls", part))
    for key, name in (("verdict_agreement", "verdict agreement"),
                      ("user_fact_consistency", "user fact consistency")):
        part = body.get(key)
        if isinstance(part, dict):
            rows.append(_rate_row(name, part))
    coverage = body.get("task_coverage")
    if isinstance(coverage, dict):
        rows.append(ScorecardItem(
            name="task coverage",
            raw=f"{coverage.get('tasks_covered', 0)} of {coverage.get('tasks_total', 0)} Tasks",
            explained=_percent(coverage.get("run_weighted")),
            note="the explained column is the Run-weighted share (D96)"))
    return rows


def stage_statuses(state: dict, budget: Any = None, stopped: Any = None) -> list[StageStatus]:
    """The pipeline's own state.json as the DAG rows: status, attempts, the gate that failed, spend.

    pipeline.py writes `statuses`, `attempts`, `gates`, `failed_stage` and a log of plain sentences,
    so all of them are read here rather than a log of dict rows nothing writes.
    """
    statuses = state.get("statuses") or {}
    attempts = state.get("attempts") or {}
    failed = state.get("failed_stage")
    per_stage = dict((stopped or {}).get("stages") or {})
    buckets = {name: bucket for name, bucket in ((budget or {}).get("stages") or {}).items()
               if isinstance(bucket, dict)}
    for name, bucket in buckets.items():
        if bucket.get("usd"):
            per_stage[name] = bucket["usd"]
    stages = []
    for name, status in statuses.items():
        stage = StageStatus(name=name, status=str(status), usd=float(per_stage.get(name) or 0.0),
                            cache_share=_cache_share(buckets.get(name, {})),
                            memo_hits=int(buckets.get(name, {}).get("memo_hits") or 0))
        try:
            stage.attempts = int(attempts.get(name) or 0)
        except (TypeError, ValueError):
            stage.attempts = 0
        stages.append(stage)
    by_name = {stage.name: stage for stage in stages}
    for body in state.get("gates") or []:
        if not isinstance(body, dict) or body.get("pass", True):
            continue
        gate_stage = str(body.get("stage") or "")
        owner = by_name.get(gate_stage) or by_name.get(gate_stage.split(".")[0]) or by_name.get(str(failed))
        if owner is not None and owner.status == "failed":
            owner.gate = gate_stage.split(".")[-1]
    for row in state.get("log") or []:
        _read_log_row(row, by_name)
    return stages


def _cache_share(bucket: dict) -> float:
    """cache_read over all input tokens: the number `docs/prompt-caching.md` says to watch per stage."""
    try:
        read = float(bucket.get("cache_read") or 0)
        total = read + float(bucket.get("input") or 0)
    except (TypeError, ValueError):
        return 0.0
    return read / total if total else 0.0


def _read_log_row(row: Any, by_name: dict) -> None:
    """One rollback log entry, the sentence pipeline.py writes to `result.log` (its only writer).

    pipeline.py logs "compile_tools: attempt 2 of 3, gate replay_fidelity failed" and, on the last
    attempt, "compile_tools: gate replay_fidelity failed 3 times, stage failed".
    """
    if not isinstance(row, str) or ":" not in row:
        return
    name, said = row.split(":", 1)
    stage = by_name.get(name.strip())
    if stage is None:
        return
    gate = re.search(r"gate (\S+) failed", said)
    if gate:
        stage.gate = gate.group(1)
    attempt = re.search(r"attempt (\d+) of (\d+)", said)
    if attempt:
        stage.attempts = max(stage.attempts, int(attempt.group(1)))
        stage.max_attempts = max(stage.max_attempts, int(attempt.group(2)))
    times = re.search(r"failed (\d+) times", said)
    if times:
        stage.attempts = max(stage.attempts, int(times.group(1)))
        stage.max_attempts = max(stage.max_attempts, int(times.group(1)))


def _list_of(path: Path, model: type) -> list:
    return _list_of_bodies(_json(path), model)


def _list_of_bodies(body: Any, model: type) -> list:
    """A JSON list as records, skipping anything in it that is not one."""
    out = []
    for item in body if isinstance(body, list) else []:
        try:
            out.append(model.model_validate(item))
        except ValidationError:
            continue
    return out


def coverage_rows(tasks: list[Task], uncovered: dict[str, str]) -> list[TaskCoverage]:
    """The D96 rows from a Task list and the first failing reason per uncovered Task.

    The rule itself is gates/scorecard.py's `task_coverage`, so the scorecard and the report cannot drift
    apart; cli.py applies it and hands the reasons here, which keeps the Runner out of this module.
    """
    return [TaskCoverage(task_id=task.id, covered=task.id not in uncovered,
                         reason=uncovered.get(task.id), run_count=len(task.run_ids)) for task in tasks]


def load_tool_sigs(workdir: Any) -> list[ToolSig]:
    """The mined ToolSigs of a build, from the tool_sigs.json one stage writes.

    cli.py reads them too, so a Verdict knows which tools write (extra-write and entity-count checks)
    and which are still flagged (D70), and the report and the Verdict cannot disagree about it.
    """
    root = Path(workdir)
    return _list_of(root / "tool_sigs.json", ToolSig)


def load(workdir: Any) -> ReportData:
    """Read every record the report shows from one workdir. Missing files mean a shorter report, not an error."""
    root = Path(workdir)
    unread: list[str] = []
    env_body = _json(root / "environment.json")
    environment = Environment.model_validate(env_body) if isinstance(env_body, dict) else None
    state = _json(root / "pipeline" / "state.json")
    state = state if isinstance(state, dict) else {}
    stopped = state.get("stopped") if isinstance(state.get("stopped"), dict) else {}
    budget = _json(root / "budget.json")
    stages = stage_statuses(state, budget if isinstance(budget, dict) else {}, stopped)
    config = _json(root / "report_config.json") or {}
    pairs = _jsonl(root / "judge_pairs.jsonl", unread)
    tasks = _records(root / "tasks", Task, unread)
    runs = load_runs(root / "runs", unread)
    gates = (_list_of(root / "gates.json", GateResult) + _list_of_bodies(state.get("gates"), GateResult))
    status = str(state.get("status", "complete"))
    data = ReportData(
        title=config.get("title") or ("Run batch report" if config.get("kind") == "batch" else "Harness build report"),
        kind=config.get("kind", "build"),
        built=environment is not None and status not in ("failed", "stopped"),
        stopped_reason=state.get("stopped_reason") or stopped.get("reason") or config.get("stopped_reason"),
        stopped=stopped,
        records_not_read=unread,
        environment=environment,
        gates=gates,
        scorecard=scorecard_rows(_json(root / "scorecard.json")),
        stages=stages,
        tasks=tasks,
        verifiers=_records(root / "verifiers", Verifier, unread),
        tool_sigs=load_tool_sigs(root),
        runs=runs,
        verdicts=_records(root / "verdicts", Verdict, unread),
        overlays=_records(root / "overlays", TaskOverlay),
        policy_items=_list_of(root / "constraints.json", Constraint),
        policy_exercised=(_json(root / "policy_coverage.json") or {}).get("exercised", []),
        task_coverage=_list_of(root / "coverage.json", TaskCoverage),
        frontier_models=config.get("frontier_models", []),
        assisted_share=config.get("assisted_share") or assisted_share_from_runs(runs),
        judge_disagreement=disagreement_stats(pairs),
        audit_rate=config.get("audit_rate"),
        disagreement_queue=_jsonl(root / "disagreement_queue.jsonl", unread),
        tasks_aside=_jsonl(root / "tasks_aside.jsonl", unread),
        lessons_set_aside=_list_of(root / "lessons_set_aside.json", SetAsideLesson),
    )
    gate = environment_gate(data)
    if gate is not None and not gate.passed:
        # design section 6: the build Environment gate is what says the Environment was built.
        data.built = False
    return data
