"""Tasks born of passed scenarios, through the mechanism every recorded Task already goes through (D389).

A passed scenario of source recordings with a binding and nothing in requires becomes one Task and one
recording in the Environment's workdir. The Task record is plain; the recording names the scenario
(source "scenario", raw_hash the scenario id) and holds one user turn whose content is the scenario's
ask, or its statement where it has no ask, and nothing else, so `scenario_of` reads where a Task came from off its recordings. The intent
miner, the Spec writer, the review rounds and the Examiner then run over it unchanged. The binding and the policy lines are not
copied anywhere: the writer reads the world itself. No Reference, no Run and no fidelity number come of
it, and trust tiers such a Task synthetic, never trusted (spec/trust.py). Where the workdir's build.json holds
a Task sample, the born ids join it, so the tiers see them. Its Starting state is the shared world as it stands
(D390): an empty overlay, since the binding names rows that exist and nothing is pinned. Its recording
carries the agent prompt most recordings of the workdir carry (D395), since the writer and the Examiner read
the policy off a Task's recording and a work copy ships no policy file.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import RawPtr, Task, TaskOverlay, Trace, Turn, as_dict, read_json, write_json
from kullback.runner.world.loading import OVERLAY_DIR
from kullback.spec import intent_tools
from kullback.synth.records import FORMAT, Scenario, load_scenarios

SECTION = "scenario"
SOURCE = "scenario"
TASKS_INDEX = "tasks.json"
BUILD_MANIFEST = "build.json"


def bears_task(scenario: Scenario) -> bool:
    """Whether a scenario may become a Task: passed, from the recordings, bound to rows, needing nothing."""
    return (scenario.status == "passed" and scenario.source.kind == "recordings"
            and bool(scenario.binding) and not scenario.requires)


def _digest(scenario_id: str) -> str:
    return hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:12]


def task_id_of(scenario_id: str) -> str:
    return "task_" + _digest(scenario_id)


def trace_id_of(scenario_id: str) -> str:
    return "sc-" + _digest(scenario_id)


def shared_prompt(workdir: Any) -> Optional[str]:
    """The agent prompt the most recordings of the workdir carry; None when no recording carries one."""
    counts = Counter(trace.system_prompt for trace in intent_tools.trace_index(Path(workdir)).values()
                     if trace.system_prompt)
    return counts.most_common(1)[0][0] if counts else None


def recording_of(scenario: Scenario, system_prompt: Optional[str] = None) -> Trace:
    """The one turn recording: the user says the scenario's ask (its statement where it has none), no agent
    turn, no tool call, the agent prompt given."""
    ptr = RawPtr(file_hash=scenario.id, section=SECTION)
    turn = Turn(idx=0, role="user", content=scenario.ask or scenario.statement, raw_ptr=ptr)
    return Trace(trace_id=trace_id_of(scenario.id), raw_hash=scenario.id, ingest_version=str(FORMAT),
                 source=SOURCE, turns=[turn], raw_ptr=ptr, system_prompt=system_prompt)


def task_of_scenario(scenario: Scenario) -> Task:
    return Task(id=task_id_of(scenario.id), run_ids=[trace_id_of(scenario.id)], unguarded=True, anchor_run_ids=[])


def scenario_of(workdir: Any, task_id: str) -> Optional[str]:
    """The scenario a Task was born of: the id its recordings name when every one of them has source
    scenario, else None. A recording is read at traces/<run id>.json, where born_tasks writes it."""
    root = Path(workdir)
    task = read_json(root / "tasks" / f"{task_id}.json", None) if "/" not in str(task_id) else None
    run_ids = (task or {}).get("run_ids") if isinstance(task, dict) else None
    if not run_ids:
        return None
    named = set()
    for run_id in run_ids:
        trace = read_json(root / "traces" / f"{run_id}.json", None) if "/" not in str(run_id) else None
        if not isinstance(trace, dict) or trace.get("source") != SOURCE:
            return None
        named.add(str(trace.get("raw_hash") or ""))
    return named.pop() if len(named) == 1 and "" not in named else None


def _write_empty_overlay(workdir: Path, task_id: str) -> bool:
    """The empty overlay of a born Task in the shape every recorded Task's overlay has (D390): no row
    pinned, no step, no Run layer, so the writer reads the shared world. False where one exists."""
    path = workdir / OVERLAY_DIR / f"{task_id}.json"
    if path.is_file():
        return False
    overlay = TaskOverlay(task_id=task_id, rows=[], steps=[])
    write_json(path, {"overlay": as_dict(overlay), "values": {}, "runs": {}, "reference_run_id": ""})
    return True


def _write_prompt(workdir: Path, trace_id: str, system_prompt: Optional[str]) -> bool:
    """The agent prompt written into an existing born recording that lacks one (D395). False otherwise."""
    path = workdir / "traces" / f"{trace_id}.json"
    trace = read_json(path, None) if system_prompt else None
    if not isinstance(trace, dict) or trace.get("system_prompt"):
        return False
    write_json(path, as_dict(Trace.model_validate({**trace, "system_prompt": system_prompt})))
    return True


def _append_index(workdir: Path, tasks: list[Task]) -> None:
    """The born Tasks appended to tasks.json in its own shape, {"tasks": [...]}."""
    body = read_json(workdir / TASKS_INDEX, {}) or {}
    rows = list(body.get("tasks") or [])
    rows.extend(as_dict(task) for task in tasks)
    write_json(workdir / TASKS_INDEX, {**body, "tasks": rows})


def _join_sample(workdir: Path, task_ids: list[str]) -> None:
    """The born ids appended to the build manifest's Task sample where it holds one; nothing else touched."""
    body = read_json(workdir / BUILD_MANIFEST, None)
    if not isinstance(body, dict) or not isinstance(body.get("tasks"), list):
        return
    write_json(workdir / BUILD_MANIFEST, {**body, "tasks": body["tasks"] + task_ids})


def born_tasks(workdir: Any, store_dir: Any, *, limit: Optional[int] = None) -> dict:
    """One Task and one recording per scenario that bears a Task, in id order, up to `limit`; a scenario
    whose Task already exists is skipped, though its overlay is written where it lacks one (D390) and its
    recording gets the shared agent prompt where it lacks one (D395). Returns the counts and the ids born;
    prompts_written counts the recordings written with a prompt, new or repaired."""
    root = Path(workdir)
    candidates = [scenario for scenario in load_scenarios(store_dir) if bears_task(scenario)]
    if limit is not None:
        candidates = candidates[:limit]
    prompt = shared_prompt(root)
    born: list[Task] = []
    skipped = overlays = prompts = 0
    for scenario in candidates:
        task = task_of_scenario(scenario)
        overlays += _write_empty_overlay(root, task.id)
        if (root / "tasks" / f"{task.id}.json").is_file():
            skipped += 1
            prompts += _write_prompt(root, trace_id_of(scenario.id), prompt)
            continue
        trace = recording_of(scenario, prompt)
        prompts += prompt is not None
        write_json(root / "traces" / f"{trace.trace_id}.json", as_dict(trace))
        write_json(root / "tasks" / f"{task.id}.json", as_dict(task))
        born.append(task)
    if born:
        _append_index(root, born)
        _join_sample(root, [task.id for task in born])
    return {"candidates": len(candidates), "born": len(born), "skipped_existing": skipped,
            "overlays_written": overlays, "prompts_written": prompts, "ids": [task.id for task in born]}


__all__ = ["bears_task", "born_tasks", "recording_of", "scenario_of", "shared_prompt", "task_id_of", "task_of_scenario",
           "trace_id_of"]
