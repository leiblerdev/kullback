"""Tasks born of passed scenarios, through the mechanism every recorded Task already goes through (D389).

A passed scenario with a binding, an ask, an end and nothing in requires becomes one Task and one
recording in the Environment's workdir, whatever its source (D415): a search born scenario carries its
page and span as provenance, as a recordings one carries its Task ids. The Task record is plain; the recording names the scenario
(source "scenario", raw_hash the scenario id) and holds one user turn whose content is the scenario's
ask, or its statement where it has no ask, and nothing else, so `scenario_of` reads where a Task came from off its recordings. The intent
miner, the Spec writer, the review rounds and the Examiner then run over it unchanged. The binding and the policy lines are not
copied anywhere: the writer reads the world itself. No Reference, no Run and no fidelity number come of
it, and trust tiers such a Task synthetic, never trusted (spec/trust.py). Where the workdir's build.json holds
a Task sample, the born ids join it, so the tiers see them. Its Starting state is the shared world as it stands
(D390): an empty overlay, since the binding names rows that exist and nothing is pinned. Its recording
carries the agent prompt most recordings of the workdir carry (D395), since the writer and the Examiner read
the policy off a Task's recording and a work copy ships no policy file. The scenario's end goes beside it
(D414): born/<task>.json holds it for the Spec writer, which gates what the end says, and the recording's
user rules (user_rules/<run id>.json) carry the ask as the goal and the end as a fact, so a user asked to
confirm the end's change can answer. Both are written in the workdir, so a work copy carries them. The user
rules also carry the scalar cells of the rows the scenario binds, nested leaves included, on request (D416):
the person's own records, kept where the column is a field the recorded users of the workdir give, else only
the bound keys.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import (
    DisclosureRule,
    RawPtr,
    Task,
    TaskOverlay,
    Trace,
    Turn,
    UserFact,
    as_dict,
    read_json,
    write_json,
)
from kullback.runner.world.loading import OVERLAY_DIR
from kullback.spec import intent_tools
from kullback.spec.writer import born_path
from kullback.synth.records import FORMAT, Scenario, load_scenarios
from kullback.synth.rule import world_of
from kullback.user.rules import derive_user_rules

SECTION = "scenario"
SOURCE = "scenario"
TASKS_INDEX = "tasks.json"
BUILD_MANIFEST = "build.json"
RULES_DIR = "user_rules"
# The user fact that carries the scenario's end (D414).
END_FIELD = "end"


def bears_task(scenario: Scenario) -> bool:
    """Whether a scenario may become a Task (D415): passed, bound to rows, with an ask and an end, needing
    nothing, whatever its source."""
    return (scenario.status == "passed" and bool(scenario.binding) and bool(scenario.ask.strip())
            and bool(scenario.end.strip()) and not scenario.requires)


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


def _write_end(workdir: Path, task_id: str, scenario: Scenario) -> bool:
    """The scenario's end beside a born Task (D414), for the Spec writer. False where it is there already."""
    path = born_path(workdir, task_id)
    body = {"scenario": scenario.id, "end": scenario.end}
    if read_json(path, None) == body:
        return False
    write_json(path, body)
    return True


def recorded_fields(workdir: Any) -> set[str]:
    """The fact fields the recorded users of the workdir give: over the user rules of every Run whose recording is
    not born of a scenario (a recorded Run's rules are named by its Run id, a born one's by its recording)."""
    root = Path(workdir)
    fields: set[str] = set()
    for path in sorted((root / RULES_DIR).glob("*.json")):
        trace = read_json(root / "traces" / path.name, None)
        rules = read_json(path, None)
        born = isinstance(trace, dict) and trace.get("source") == SOURCE
        if not born and isinstance(rules, dict):
            fields.update(str(fact.get("field")) for fact in rules.get("facts") or () if isinstance(fact, dict))
    return fields


def _schema_order(workdir: Path, table: str, row: dict) -> list[str]:
    """The row's columns in the order the schema names them, those it does not name after in row order."""
    schema = read_json(workdir / "schema.json", None)
    named = [str(column.get("name")) for column in (schema.get("columns") or () if isinstance(schema, dict) else ())
             if isinstance(column, dict) and column.get("table") == table]
    return [name for name in named if name in row] + [name for name in row if name not in named]


def _leaves(column: str, value: Any) -> list[tuple[str, Any]]:
    """The column and its value, then, where the value is an object, every key and value under it at any depth;
    a list is not walked."""
    out = [(column, value)]
    if isinstance(value, dict):
        for key, inner in value.items():
            out += _leaves(str(key), inner)
    return out


def _bound_leaves(root: Path, binding: dict) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Each bound row the world holds, as its key and its columns and nested leaves in schema order."""
    state = world_of(root).state
    out = []
    for table, key in (binding or {}).items():
        rows = state.get(str(table))
        row = rows.get(str(key)) if isinstance(rows, dict) else None
        if isinstance(row, dict):
            out.append((str(key), [leaf for name in _schema_order(root, str(table), row) for leaf in _leaves(name, row[name])]))
    return out


def _owner(rows: list[tuple[str, list[tuple[str, Any]]]]) -> Optional[int]:
    """The one bound row the others refer to: its key is a scalar cell or leaf of another bound row. None where no
    row or more than one is referred to."""
    referred = [at for at, (key, _) in enumerate(rows)
                if any(str(value) == key for other, (_, leaves) in enumerate(rows) if other != at
                       for _, value in leaves if isinstance(value, (str, int, float, bool)))]
    return referred[0] if len(referred) == 1 else None


def _values(leaves: list[tuple[str, Any]]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for column, value in leaves:
        seen.setdefault(column, set()).add(json.dumps(value, sort_keys=True))
    return seen


def bound_cells(workdir: Any, binding: dict, fields: set[str]) -> list[tuple[str, Any]]:
    """The scalar cells of the rows a scenario binds, column and value, in schema order and once per column:
    those whose column is one of `fields`, or with no fields the cells holding a bound key (D416). A nested
    object is walked to its scalar leaves, each a cell named by its leaf key; a list is no cell. A column the
    bound rows hold with two different values (one row's name, another row's nested name) keeps the value of
    the owner, the one bound row the others refer to by its key, where that row holds one value of it; else it
    names no one value of the person and is left out."""
    rows = _bound_leaves(Path(workdir), binding)
    owner = _owner(rows)
    seen = _values([leaf for _, leaves in rows for leaf in leaves])
    own = _values(rows[owner][1]) if owner is not None else {}
    cells: dict[str, Any] = {}
    for at, (key, leaves) in enumerate(rows):
        for column, value in leaves:
            scalar = isinstance(value, (str, int, float, bool))
            kept = column in fields if fields else str(value) == key
            if len(seen[column]) > 1:
                kept = kept and at == owner and len(own[column]) == 1
            if scalar and kept and column not in cells:
                cells[column] = value
    return list(cells.items())


def user_rules_of(trace: Trace, end: str, cells: list[tuple[str, Any]] = ()) -> dict:
    """The user rules of a born recording: what code reads off its one user turn, the bound cells it does not
    state, each given on request (D416), and the end as a fact, last."""
    rules = derive_user_rules(trace)
    stated = {fact.field for fact in rules.facts} | {str(fact.value) for fact in rules.facts}
    for field, value in cells:
        if field not in stated and str(value) not in stated:
            rules.facts.append(UserFact(field=field, value=value, context=end))
            rules.disclosure.append(DisclosureRule(field=field, on_request=True))
    rules.facts.append(UserFact(field=END_FIELD, value=end, context=end))
    return as_dict(rules)


def _write_rules(workdir: Path, scenario: Scenario, fields: set[str]) -> bool:
    """The user rules of a born recording where it has none. False where they exist or no recording does."""
    trace_id = trace_id_of(scenario.id)
    path, body = workdir / RULES_DIR / f"{trace_id}.json", read_json(workdir / "traces" / f"{trace_id}.json", None)
    if path.is_file() or not isinstance(body, dict):
        return False
    cells = bound_cells(workdir, scenario.binding, fields)
    write_json(path, user_rules_of(Trace.model_validate(body), scenario.end, cells))
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
    recording gets the shared agent prompt where it lacks one (D395), and each gets the scenario's end and
    its recording's user rules where it lacks them (D414). Returns the counts and the ids born;
    prompts_written counts the recordings written with a prompt, new or repaired, ends_written and
    rules_written the ends and user rules written, new or repaired."""
    root = Path(workdir)
    candidates = [scenario for scenario in load_scenarios(store_dir) if bears_task(scenario)]
    if limit is not None:
        candidates = candidates[:limit]
    prompt = shared_prompt(root)
    fields = recorded_fields(root)
    born: list[Task] = []
    skipped = overlays = prompts = ends = rules = 0
    for scenario in candidates:
        task = task_of_scenario(scenario)
        overlays += _write_empty_overlay(root, task.id)
        if (root / "tasks" / f"{task.id}.json").is_file():
            skipped += 1
            prompts += _write_prompt(root, trace_id_of(scenario.id), prompt)
        else:
            trace = recording_of(scenario, prompt)
            prompts += prompt is not None
            write_json(root / "traces" / f"{trace.trace_id}.json", as_dict(trace))
            write_json(root / "tasks" / f"{task.id}.json", as_dict(task))
            born.append(task)
        ends += _write_end(root, task.id, scenario)
        rules += _write_rules(root, scenario, fields)
    if born:
        _append_index(root, born)
        _join_sample(root, [task.id for task in born])
    return {"candidates": len(candidates), "born": len(born), "skipped_existing": skipped,
            "overlays_written": overlays, "prompts_written": prompts, "ends_written": ends, "rules_written": rules,
            "ids": [task.id for task in born]}


__all__ = ["END_FIELD", "bears_task", "born_tasks", "bound_cells", "recorded_fields", "recording_of", "scenario_of",
           "shared_prompt", "task_id_of", "task_of_scenario", "trace_id_of", "user_rules_of"]
