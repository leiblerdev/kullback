"""The D62 scorecard, as D80 and D96 leave it: tool fidelity, Task coverage, user fact
consistency and Verdict agreement, raw and explained side by side.

It sits in `gates/` rather than `runner/` because its one verdict, the `scorecard` gate that is
never green over nothing, is built on the per-tool fidelity bar (`gates/fidelity.py`), and the
Runner cannot import this package. It reads the build directory it is given and nothing else."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from kullback.gates.fidelity import replay_fidelity_gate
from kullback.runner.gate_support import MISS_REASONS, _get, _passed, _rate, _same, _share, gate
from kullback.runner.records import as_dict, content_hash

FROZEN_TASKS_NAME = "tasks_frozen.json"
COVERAGE_TAGS = ("fact_unavailable", "overlay_miss", "reconstructed", "truncated")
# The D96 reasons a Run record can actually carry today: user_sim.py tags a user_turn
# `fact_unavailable` and loop.py tags a tool_result `overlay_miss`. Nothing writes `reconstructed`
# (mine.py hands its tag to the Builder, never onto an Event) or `truncated` onto an Event, so the
# scorecard says those two are not measured instead of letting their absence read as coverage.
MEASURED_COVERAGE_TAGS = ("fact_unavailable", "overlay_miss")


def freeze_tasks(build_dir: Union[str, Path], tasks) -> list[str]:
    """Write the Task list this build's coverage is measured against, once, at the cluster stage (D96).

    A frozen list already on disk is returned unchanged: a later split or re-cluster must not move the
    denominator, which is exactly what D96 forbids.
    """
    path = Path(build_dir) / FROZEN_TASKS_NAME
    if path.is_file():
        return _frozen_ids(_load(Path(build_dir), FROZEN_TASKS_NAME, None)) or []
    ids = [_get(task, "id", "?") for task in tasks or ()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_ids": ids, "hash": content_hash(ids)}, indent=2, sort_keys=True),
                    encoding="utf-8")
    return ids


def _frozen_ids(data: Any) -> Optional[list[str]]:
    """The frozen Task ids, or None when this build never froze a list."""
    if data is None:
        return None
    return list(data.get("task_ids", []) if isinstance(data, dict) else data)


def scorecard(build_dir: Union[str, Path], reference_verdicts=None) -> dict:
    """Tool fidelity, Task coverage, user fact consistency and Verdict agreement, raw and explained side by side."""
    root = Path(build_dir)
    tasks = _listed(_load(root, "tasks.json", []), "tasks")
    runs = _listed(_load(root, "runs.json", []), "runs")
    verdicts = _listed(_load(root, "verdicts.json", []), "verdicts")
    reference = reference_verdicts if reference_verdicts is not None else \
        _listed(_load(root, "reference_verdicts.json", []), "verdicts")
    calls = [c for c in _listed(_load(root, "held_out_calls.json", []), "calls") if _get(c, "held_out", True)]
    fidelity = replay_fidelity_gate(calls)
    card = {
        "tool_fidelity": fidelity.metrics,
        "task_coverage": task_coverage(tasks, runs, _load(root, "task_status.json", {}) or {},
                                       frozen_ids=_frozen_ids(_load(root, FROZEN_TASKS_NAME, None))),
        "user_fact_consistency": _fact_consistency(_listed(_load(root, "user_facts.json", []), "facts")),
        "verdict_agreement": _agreement(verdicts, reference, _not_gradeable(root)),
        "policy_coverage": _load(root, "policy.json", {}) or {},
    }
    failures = list(fidelity.failures)
    failures += [f"Run {m['run_id']}: fact {m['field']} differs on a re-run with no reason"
                 for m in card["user_fact_consistency"]["misses"] if m.get("reason") not in MISS_REASONS]
    failures += [_agreement_failure(m) for m in card["verdict_agreement"]["misses"]
                 if m.get("reason") not in MISS_REASONS]
    if tasks and not card["task_coverage"]["tasks_covered"]:
        # A green scorecard that grades nothing is the one output that should never be green: the
        # second retail build passed here with tasks_covered 0.0 over 205 Tasks.
        failures.append(f"no Task is gradeable: {len(tasks)} Tasks and not one has a confirmed Reference "
                        "and a passing Verifier")
    card["gate"] = as_dict(gate(
        "scorecard", failures,
        tasks_covered=card["task_coverage"]["tasks"],
        run_weighted=card["task_coverage"]["run_weighted"],
        success_fidelity=card["tool_fidelity"]["success"]["explained"],
        error_fidelity=card["tool_fidelity"]["error"]["explained"],
        verdict_agreement=card["verdict_agreement"]["explained"],
    ))
    return card


def _agreement_failure(miss: dict) -> str:
    if miss.get("missing"):
        return f"Run {miss['run_id']}: the reference has a Verdict for this Run and we produced none"
    return f"Run {miss['run_id']}: our Verdict disagrees with the reference with no reason"


def _not_gradeable(root: Path) -> set:
    """Task or Run ids set aside as not gradeable (D49, D93); they leave the agreement denominator."""
    data = _load(root, "not_gradeable.json", [])
    if isinstance(data, dict):
        return set(data.get("task_ids", []) or ()) | set(data.get("run_ids", []) or ())
    return set(data or ())


def _load(root: Path, name: str, default: Any) -> Any:
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _listed(obj: Any, key: str) -> list:
    return list(obj.get(key, []) if isinstance(obj, dict) else obj or ())


def task_coverage(tasks: list, runs: list, status: Optional[dict] = None,
                  frozen_ids: Optional[Sequence[str]] = None) -> dict:
    """Covered Tasks over the frozen Task list, plain and Run-weighted, first failing reason attached (D96).

    Public because the report shows the two D96 headline numbers and computes them the same way the
    scorecard does; `status` carries `reference_confirmed` and `verifier_passed` per Task id, and a
    Task with no entry is uncovered, because nothing confirmed its Reference or passed its Verifier.
    `frozen_ids` is the Task list fixed at the start of the build: Tasks that appeared after it are
    listed apart instead of joining the denominator.
    """
    status = status or {}
    by_id = {_get(run, "run_id"): run for run in runs}
    by_task = {_get(task, "id", "?"): task for task in tasks}
    ids = list(frozen_ids) if frozen_ids is not None else list(by_task)
    added_later = sorted(set(by_task) - set(ids)) if frozen_ids is not None else []
    covered, uncovered, runs_total, runs_covered = 0, [], 0, 0
    for task_id in ids:
        task = by_task.get(task_id)
        if task is None:
            uncovered.append({"task_id": task_id, "runs": 0,
                              "reason": "the Task is on the frozen list and not in this build's Task list"})
            continue
        run_ids = list(_get(task, "run_ids", []) or ())
        runs_total += len(run_ids)
        reason = _uncovered_reason(run_ids, by_id, status.get(task_id) or {})
        if reason is None:
            covered += 1
            runs_covered += len(run_ids)
        else:
            uncovered.append({"task_id": task_id, "reason": reason, "runs": len(run_ids)})
    return {"tasks_total": len(ids), "tasks_covered": covered, "tasks": _share(covered, len(ids)),
            "runs_total": runs_total, "runs_covered": runs_covered,
            "run_weighted": _share(runs_covered, runs_total), "uncovered": uncovered,
            "added_later": added_later,
            "reasons_not_measured": [tag for tag in COVERAGE_TAGS if tag not in MEASURED_COVERAGE_TAGS]}


def _uncovered_reason(run_ids: list, by_id: dict, status: dict) -> Optional[str]:
    if not run_ids:
        return "Task has no Runs"
    for run_id in run_ids:
        run = by_id.get(run_id)
        if run is None:
            return f"Run {run_id} was not replayed"
        counts = _get(run, "route_counts", {}) or {}
        if _get(run, "assisted", False) or int(counts.get("llm") or 0) > 0:
            return f"Run {run_id} is assisted (D49)"
        for event in _get(run, "events", []) or ():
            payload = _get(event, "payload", {}) or {}
            # The route on the event is read beside the flag: a Run whose flag was lost still shows
            # the LLM stand-in in its own events (D49).
            if _get(event, "assisted", False) or _get(event, "route") == "llm":
                return f"Run {run_id} has an assisted event (D49)"
            tags = list(payload.get("tags") or ())
            for tag in COVERAGE_TAGS:
                if payload.get(tag) or tag in tags:
                    return f"Run {run_id} hit {tag}"
    if not status.get("reference_confirmed", False):
        return "the Reference is not confirmed (D57, D93)" if "reference_confirmed" in status \
            else "no Reference confirmation is recorded for this Task (D57, D93)"
    if not status.get("verifier_passed", False):
        return "the Verifier did not pass the D79 suite" if "verifier_passed" in status \
            else "no D79 result is recorded for this Task's Verifier"
    return None


def _fact_consistency(facts: list) -> dict:
    """The Simulated user gave the same facts on the re-runs it gave in the trace (D44)."""
    total, matched, misses = 0, 0, []
    for fact in facts:
        total += 1
        if _same(_get(fact, "expected"), _get(fact, "observed")):
            matched += 1
        else:
            misses.append({"run_id": _get(fact, "run_id"), "field": _get(fact, "field"),
                           "reason": _get(fact, "reason")})
    return _rate(total, matched, misses)


def _agreement(verdicts: list, reference: list, not_gradeable=()) -> dict:
    """Our Verdict against the supplied reference set, every miss carrying a D80 reason.

    A reference Run we produced no Verdict for is a miss, not a Run that quietly leaves the
    denominator; only a Task or Run set aside as not gradeable (D49, D93) is skipped.
    """
    ours = {_get(v, "run_id"): v for v in verdicts}
    aside = set(not_gradeable or ())
    total, matched, misses = 0, 0, []
    for entry in reference:
        run_id = _get(entry, "run_id")
        if run_id in aside or _get(entry, "task_id") in aside:
            continue
        mine = ours.get(run_id)
        total += 1
        if mine is None:
            misses.append({"run_id": run_id, "ours": None, "reference": _passed(entry),
                           "reason": _get(entry, "reason"), "missing": True})
            continue
        if _passed(mine) == _passed(entry):
            matched += 1
        else:
            misses.append({"run_id": run_id, "ours": _passed(mine), "reference": _passed(entry),
                           "reason": _get(entry, "reason") or _note_reason(mine)})
    return _rate(total, matched, misses)


def _note_reason(verdict: Any) -> Optional[str]:
    for note in _get(verdict, "notes", []) or ():
        if str(note).startswith("miss_reason:"):
            return str(note).split(":", 1)[1]
    return None
