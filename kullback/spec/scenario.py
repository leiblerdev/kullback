"""A synthetic store Task's expected Run: the Starting state and the Verifier's compiled expected end state.

A store Task has no Reference (D360, synth-add3-fix): its walk is only the scenario source that grounds the
description in real rows. Where a recorded Task lays its mutants and its do-nothing Run on a Reference, a
store Task lays them on this Run instead: no events but the stop, its start state the Task's Starting state
and its end state that start with the Verifier's first expected end state written over it. Nothing here
reads the walk or calls a model.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import Event, Run, Verifier, read_json

# The run id suffix of the expected Run, so a report says which base a mutant was laid on.
EXPECTED = "expected"
# The reason a store Task is held when a Run passes its Verifier by a route the scenario cannot explain.
SHORTCUT = "shortcut"
# Where a store keeps each Task's scenario source (synthesise.SCENARIOS, named here to keep spec below it).
SCENARIOS = "scenarios"


def starting_state(workdir: Any, task_id: str) -> dict:
    """The Task's Starting state: the shared world with its overlay laid over it, as the writer reads it."""
    from kullback.runner.state import StateView
    from kullback.runner.world.environment import BuiltEnvironment

    env = BuiltEnvironment(Path(workdir))
    overlay, rows = env.overlay(task_id)
    return StateView(env.db, overlay, rows).shared


def _cell_value(value: Any) -> tuple[bool, Any]:
    """(whether a value can be written, the value): the first of a `one_of`, never a `not` set."""
    if isinstance(value, dict) and "one_of" in value:
        options = list(value.get("one_of") or [])
        return bool(options), options[0] if options else None
    if isinstance(value, dict) and "not" in value:
        return False, None
    return True, value


def _set_path(row: dict, path: str, value: Any) -> None:
    parts = str(path).split(".")
    node = row
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


def expected_end(verifier: Verifier, start: dict) -> dict:
    """The start state with the first expected end state's cells and new rows written over it."""
    end = copy.deepcopy(start)
    if not verifier.expected:
        return end
    state = verifier.expected[0]
    for cell in state.cells:
        writable, value = _cell_value(cell.value)
        if not writable:
            continue
        table = end.setdefault(cell.table, {})
        if cell.field is None:
            if value is None:
                table.pop(cell.row_id, None)
            else:
                table[cell.row_id] = copy.deepcopy(value)
            continue
        _set_path(table.setdefault(cell.row_id, {}), cell.field, value)
    for number, spec in enumerate(state.new_rows):
        table = end.setdefault(str(spec.get("table")), {})
        for copy_no in range(max(int(spec.get("count") or 1), 1)):
            table[f"{EXPECTED}-{number}-{copy_no}"] = copy.deepcopy(dict(spec.get("where") or {}))
    return end


def expected_run(verifier: Verifier, start: dict, task_id: str) -> Run:
    """The Run that does exactly what the Verifier's expected end state says, and nothing else."""
    return Run(run_id=f"{task_id}.{EXPECTED}", task_id=task_id,
               events=[Event(idx=0, type="stop", payload={"start_state": copy.deepcopy(start),
                                                          "end_state": expected_end(verifier, start)})],
               termination_reason="max_turns")


def state_part(verifier: Verifier) -> Verifier:
    """The Verifier's end state checks alone: what a mutant of the expected end state is scored against."""
    return Verifier(task_id=verifier.task_id, expected=list(verifier.expected),
                    forbidden=list(verifier.forbidden))


def store_expected_run(workdir: Any, task_id: str, verifier: Optional[Verifier]) -> Optional[Run]:
    """The expected Run of one store Task, or None where it has no Verifier or no Starting state."""
    if verifier is None:
        return None
    try:
        return expected_run(verifier, starting_state(workdir, task_id), task_id)
    except (OSError, ValueError, KeyError):
        return None


def walk_writes(workdir: Any, task_id: str, write_tools: Any) -> set[str]:
    """The write tools the Task's scenario source called, read off the store; none where it has none."""
    body = read_json(Path(workdir) / SCENARIOS / f"{task_id}.json", None) or {}
    return {str(step.get("tool")) for step in body.get("steps") or ()} & set(write_tools or ())


def shortcut(run: Run, description: str, writes: set[str]) -> str:
    """Why a Run that passed is a shortcut, or "" where it is not.

    It is one where it called none of the scenario's write tools, so the End state came another way,
    or where a call argument carries a value no read of that Run returned and the description does not
    state: the Run knew what it could not have been told. The assistant's own text is not read.
    """
    from kullback import variants as variants_mod
    from kullback.runner.world import birth as birth_mod

    calls = variants_mod.call_results(run)
    if writes and not {call["name"] for call in calls} & writes:
        return "called none of the scenario's write tools"
    trace = birth_mod.check_arguments_trace(
        walk=[{"args": call["args"]} for call in calls],
        results=[{"id": f"call-{index + 1}", "result": call["result"]} for index, call in enumerate(calls)],
        intent=description, user_facts=[])
    if trace.outcome != birth_mod.HELD:
        return f"{len(trace.untraced)} argument value(s) no read returned and the description does not state"
    return ""


def shortcut_audit(workdir: Any, task_id: str, runs: Any, passed: Any, write_tools: Any) -> list[dict]:
    """One row per passing Run of a store Task that is a shortcut: its run id and why."""
    task = read_json(Path(workdir) / "tasks" / f"{task_id}.json", None) or {}
    writes = walk_writes(workdir, task_id, write_tools)
    wanted = set(passed or ())
    out = []
    for run in runs or ():
        if run.run_id in wanted:
            why = shortcut(run, str(task.get("intent") or ""), writes)
            if why:
                out.append({"run_id": run.run_id, "why": why})
    return out


__all__ = ["EXPECTED", "SCENARIOS", "SHORTCUT", "shortcut", "shortcut_audit", "walk_writes", "expected_end", "expected_run", "starting_state", "state_part", "store_expected_run"]
