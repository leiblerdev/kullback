"""The numbers beside each Task, counted off the stored Verdicts and never decided."""

from __future__ import annotations

from typing import Optional

from kullback.report.data import ReportData
from kullback.runner.records import Run, Task, Verdict, disagreement_stats


def false_rejection_by_task(data: ReportData) -> dict[str, Optional[float]]:
    """Per Task, the share of held-out frontier Runs the required atoms reject (D133), off the trusted
    ruling's metrics; None where a Task held nothing out."""
    if data.trusted is None:
        return {}
    return {str(task_id): value for task_id, value in (data.trusted.metrics.get("false_rejection") or {}).items()}


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


def _runs_by_id(data: ReportData) -> dict[str, Run]:
    """Every stored Run by id, so a Verdict finds the Run it grades."""
    return {run.run_id: run for run in data.runs}


def _member_verdicts(data: ReportData, task: Task) -> tuple[list[Verdict], list[Verdict]]:
    """This Task's Verdicts, one per Run under the versions on disk now."""
    members = set(task.run_ids) | {r.run_id for r in data.runs if r.task_id == task.id}
    return current_verdicts(data, task, [v for v in data.verdicts if v.run_id in members])


def _split_graded(verdicts: list[Verdict], runs: dict[str, Run]) -> tuple[list[Verdict], list[Verdict]]:
    """The counted Verdicts and the ones an assisted Run or a suspect Environment excludes."""
    graded, uncounted = [], []
    for record in verdicts:
        run = runs.get(record.run_id)
        if record.environment_suspected or (run is not None and run.assisted):
            uncounted.append(record)
        else:
            graded.append(record)
    return graded, uncounted


def _split_frontier(
    graded: list[Verdict], runs: dict[str, Run], frontier_models: list[str]
) -> tuple[list[Verdict], list[Verdict]]:
    """The graded Verdicts on frontier models and the ones on the Candidate."""
    frontier, candidate = [], []
    for record in graded:
        run = runs.get(record.run_id)
        model = run.model if run is not None else None
        if (model in frontier_models) is True:
            frontier.append(record)
        else:
            candidate.append(record)
    return frontier, candidate


def _atom_kinds(data: ReportData, task: Task) -> dict[str, str]:
    """The class of every atom this Task's Verifier holds."""
    return {a.id: a.kind for v in data.verifiers if v.task_id == task.id for a in v.atoms}


def _failing_counts(graded: list[Verdict], kinds: dict[str, str]) -> dict[str, int]:
    """The counted failures by atom class, unknown where the atom is gone."""
    failing: dict[str, int] = {}
    for record in graded:
        if record.failing_atom:
            kind = kinds.get(record.failing_atom, "unknown")
            failing[kind] = failing.get(kind, 0) + 1
    return failing


def _cause_counts(graded: list[Verdict]) -> dict[str, int]:
    """The counted failures by cause, counted only where a cause is named."""
    causes: dict[str, int] = {}
    for record in graded:
        if not record.passed and record.cause:
            causes[record.cause] = causes.get(record.cause, 0) + 1
    return causes


def _margin_of(frontier_rate: Optional[float], candidate_rate: Optional[float]) -> Optional[float]:
    """How far the Candidate stands above frontier, or nothing where either side is missing."""
    if frontier_rate is None or candidate_rate is None:
        return None
    return candidate_rate - frontier_rate


def task_numbers(data: ReportData, task: Task) -> dict:
    """Count one Task's stored Verdicts. Assisted Runs and environment-suspected Runs are not counted (D49, D88)."""
    runs = _runs_by_id(data)
    verdicts, superseded = _member_verdicts(data, task)
    graded, uncounted = _split_graded(verdicts, runs)
    frontier, candidate = _split_frontier(graded, runs, data.frontier_models)
    frontier_rate = _rate(sum(1 for v in frontier if v.passed), len(frontier))
    candidate_rate = _rate(sum(1 for v in candidate if v.passed), len(candidate))
    kinds = _atom_kinds(data, task)
    return {
        "runs_graded": len(graded),
        "assisted_not_counted": len(uncounted),
        "overlay_rows": overlay_rows(data, task.id),
        "judge_atoms": sum(1 for v in data.verifiers if v.task_id == task.id for a in v.atoms if a.judge),
        "judge_disagreement_rate": data.judge_disagreement.get("rate"),
        "frontier_runs": len(frontier),
        "frontier_pass_rate": frontier_rate,
        "candidate_runs": len(candidate),
        "candidate_pass_rate": candidate_rate,
        "margin": _margin_of(frontier_rate, candidate_rate),
        "failing_atoms": _failing_counts(graded, kinds),
        "causes": _cause_counts(graded),
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
        return (
            f"Suggestion: this Task is not gradeable, Reference disputed ({aside}); it comes back once a "
            "person resolves it."
        ) + tail
    if not built:
        return "Suggestion: the Environment was not built, so the numbers cannot support a routing decision yet." + tail
    if not numbers.get("runs_graded"):
        return "Suggestion: no Runs were graded for this Task, so the numbers say nothing either way." + tail
    margin = numbers.get("margin")
    if margin is None:
        return "Suggestion: there is no frontier Run to compare against, so the numbers say nothing either way." + tail
    verb = "support" if margin >= 0 else "do not support"
    return f"Suggestion: the numbers {verb} routing this Task to the Candidate (margin {margin:+.2f})." + tail


def claims_for_task(data: ReportData, task_id: str) -> dict:
    """One Task's row of the claim record, or an empty row where the build wrote none."""
    row = ((data.claims or {}).get("tasks") or {}).get(task_id)
    return dict(row) if isinstance(row, dict) else {}


def _by_pair(rows: list[dict]) -> dict:
    """Judge disagreement per pair of judges, off the pair name each row carries (D160).

    The counting is `records.disagreement_stats`, the same rule as the rate over every row, and the
    pair's name is the one judge.py wrote into the row, so neither number nor name is invented here.
    The grouping is repeated rather than imported: the report reads records and reaches into no other
    Runner module (design section 4 item 18). A row from before D160 names no pair and joins none.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get("judges") or "")
        if name:
            grouped.setdefault(name, []).append(row)
    return {
        name: {key: disagreement_stats(group)[key] for key in ("pairs", "disagreements", "rate")}
        for name, group in sorted(grouped.items())
    }
