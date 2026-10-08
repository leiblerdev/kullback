"""Trust on the Spec's Verifier: constructed Runs fail and no ruling is open, by code (D333).

A Task is trusted when both constructed Runs score 0 under its Verifier and no blocking ruling is open:

1. the do-nothing Run: the Task's start state left as it was, no call, no message (`empty_run`); a
   refusal or no-write Verifier (`forbids_only`) is passed by it rightly, so it is not asked to fail;
2. the stray-write Run: the same start state with one write to a row outside every row the Verifier
   declares (its expected cells and `allowed`), which must fail the sanity item (nothing outside the
   declared rows changed).

Otherwise the Task is untrusted with one reason, in this order: no_intent (no Spec, an empty Spec, no
Verifier, or a Task refused), open_ruling (a ruling open, or the Task set aside by the router's rule),
constructed_run_passed (a constructed Run passed, or none could be built: not shown to fail). A stray
write that fails at a cell or an atom, or that sits on the do-nothing Run, never tested the sanity check:
it counts as passed and the row's `detail` says stray_untested.

Two flags ride on the row and never gate. reference_passes: every faithful replay of a kept Reference
passes the Verifier (None without one); a fail is handed to the Examiner as a code ruling
(`examiner_rulings`), who rules which side is wrong. solvable: any fresh Run passed (None without one).
Sourced is enforced in the writer's tool, not here. Every row names the Runs it scored and carries
`code_hash` (runner/code_hash.py): two rows are comparable when their hashes match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates.probes import write_tools_of
from kullback.gates.trust import schema_of, workdir_trusted_ruling
from kullback.gates.verifier_suite import empty_run, forbids_only
from kullback.runner.atom_context import AtomContext
from kullback.runner.canon import load_rules
from kullback.runner.code_hash import CODE_HASH
from kullback.runner.records import Run, Verifier, load_run_jsonl, read_json, read_jsonl
from kullback.runner.replay import RECORDED
from kullback.spec.actions import action_tables_of, workdir_action_tools
from kullback.spec.canfail import judge_run
from kullback.spec.events import TASK_SET_ASIDE
from kullback.spec.router import (  # noqa: F401  (ROUNDS_CAP re-exported for readers of the tiers)
    ROUNDS_CAP,
    is_set_aside,
)
from kullback.spec.schema import Spec, is_empty, load_spec, spec_ids
from kullback.spec.writer import spec_verifier_path

TRUSTED, UNTRUSTED = "trusted", "untrusted"
TIERS = (TRUSTED, UNTRUSTED)
NO_INTENT, OPEN_RULING, CONSTRUCTED_RUN_PASSED = "no_intent", "open_ruling", "constructed_run_passed"
# The reasons a Task is not trusted, in the order they are read.
REASONS = (NO_INTENT, OPEN_RULING, CONSTRUCTED_RUN_PASSED)
# The two constructed Runs, each of which must fail.
CONSTRUCTED = ("do_nothing", "stray_write")
# How the Verdict names a failure at the sanity check: a row outside the declared ones moved.
SANITY_MARKERS = ("collateral:", "sanity")
STRAY_UNTESTED = "stray_untested"
FLAGS = ("reference_passes", "solvable")
# The build manifest in the workdir: the build's Task sample, kept for a resume.
BUILD_MANIFEST = "build.json"
# The code ruling a failing Reference becomes, for the Examiner to rule which side is wrong.
REFERENCE_FAILS = "reference_fails"


class TaskTier(NamedTuple):
    tier: str
    row: dict


# --- which Run is a replay -------------------------------------------------------------------------

def run_models(run: Run) -> set[str]:
    """Every model the Run names: its own model field and the model each of its model replies carries."""
    named = {str(run.model)} if run.model else set()
    for event in run.events:
        reply = event.payload.get("reply") if event.type == "model_call" else None
        if isinstance(reply, dict) and reply.get("model"):
            named.add(str(reply["model"]))
    return named


def is_replay(run: Run) -> bool:
    """A Run the recorded agent played, whatever its run id: the recorded model is among its models."""
    return RECORDED in run_models(run)


def is_fresh(run: Run) -> bool:
    """A live Candidate Run: it names a model and the recorded one is not among them."""
    models = run_models(run)
    return bool(models) and RECORDED not in models


# --- the constructed Runs --------------------------------------------------------------------------

def _stop(run: Run) -> Optional[int]:
    """The index of the Run's last stop event carrying a start and an end state, or None."""
    for i in range(len(run.events) - 1, -1, -1):
        payload = run.events[i].payload or {}
        if run.events[i].type == "stop" and isinstance(payload.get("start_state"), dict) \
                and isinstance(payload.get("end_state"), dict):
            return i
    return None


def declared_rows(verifier: Verifier) -> tuple[set[tuple[str, str]], set[str]]:
    """The (table, row) pairs the Verifier names and the tables it allows whole, over every end state."""
    rows: set[tuple[str, str]] = set()
    tables: set[str] = set()
    for state in verifier.expected:
        rows |= {(cell.table, str(cell.row_id)) for cell in state.cells}
        for rule in state.allowed:
            if rule.get("table") is None:
                continue
            if rule.get("row_id") is None:
                tables.add(str(rule["table"]))
            else:
                rows.add((str(rule["table"]), str(rule["row_id"])))
    for rule in verifier.forbidden:
        if rule.table and rule.row_id is not None:
            rows.add((rule.table, str(rule.row_id)))
    return rows, tables


def _stray_value(value: Any) -> Any:
    """A value of the same kind that differs from `value`; None where no scalar stands."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " (stray)"
    return None


def stray_write_run(verifier: Verifier, base: Optional[Run], *, schema: Any = None,
                    canon: Any = None) -> Optional[Run]:
    """`base` with one more write, to a row outside the declared rows; None when none stands.

    Tables that record calls (`action_tables_of`) are not state and are skipped. The first row, in sorted
    order, with a scalar field whose change the diff shows (an exempt column shows nothing) is written,
    so the Run differs from `base` by that one row and fails, if at all, at the sanity item.
    """
    at = _stop(base) if base is not None else None
    if at is None:
        return None
    end = base.events[at].payload["end_state"]
    rows, whole = declared_rows(verifier)
    skip = action_tables_of(schema) | whole
    before = set(AtomContext(base, canon).diff())
    for table in sorted(end, key=str):
        if table in skip or not isinstance(end[table], dict):
            continue
        for row_id in sorted(end[table], key=str):
            row = end[table][row_id]
            if (table, str(row_id)) in rows or not isinstance(row, dict):
                continue
            for field in sorted(row, key=str):
                value = _stray_value(row[field])
                if value is None:
                    continue
                run = base.model_copy(deep=True)
                run.run_id = f"{base.run_id}.stray_write"
                run.events[at].payload["end_state"][table][row_id][field] = value
                if set(AtomContext(run, canon).diff()) - before:
                    return run
    return None


def _do_nothing(runs: list[Run]) -> Optional[Run]:
    """The empty Run on the Task's start state, off the first Run that carries one; None without one."""
    base = next((run for run in runs if _stop(run) is not None), None)
    return empty_run(base) if base is not None else None


def _judge(verifier: Verifier, run: Run, canon: Any, write_tools: Any) -> tuple[bool, Any]:
    try:
        return judge_run(verifier, run, canon, write_tools)
    except Exception:  # a Run the scorer cannot read is not a pass, and never stops the ruling
        return False, None


def _stray_base(runs: list[Run], passing: Iterable[str], empty: Optional[Run]) -> tuple[Optional[Run], bool]:
    """(the Run the stray write is laid on, whether it passes): the first Run in `passing` (a faithful
    Reference first, then a fresh Run), else the do-nothing Run, on which the sanity check goes untested."""
    by_id = {run.run_id: run for run in runs}
    base = next((by_id[run_id] for run_id in passing if run_id in by_id and _stop(by_id[run_id]) is not None), None)
    return (base, True) if base is not None else (empty, False)


def constructed_runs(verifier: Verifier, runs: list[Run], passing: Iterable[str], *, schema: Any = None,
                     canon: Any = None) -> dict[str, Optional[Run]]:
    """name -> the constructed Run, None where it cannot be built; do_nothing is left out where it is right.

    The stray write is laid on `_stray_base`: on a passing base, the stray row is the one thing that can fail it.
    """
    empty = _do_nothing(runs)
    base, _passing = _stray_base(runs, passing, empty)
    out: dict[str, Optional[Run]] = {}
    if not forbids_only(verifier):
        out["do_nothing"] = empty
    out["stray_write"] = stray_write_run(verifier, base, schema=schema, canon=canon)
    return out


def at_sanity(atom: Any) -> bool:
    """Whether a failing atom is the sanity check: nothing outside the declared rows moved."""
    return any(marker in str(atom or "") for marker in SANITY_MARKERS)


def _constructed(verifier: Verifier, runs: list[Run], passing: list[str], canon: Any, tools: Any,
                 schema: Any) -> dict[str, dict]:
    """name -> {built, failed, atom, run_id}, and for the stray write `untested`.

    The stray write counts as failed only when it fails at the sanity check on a base the Verifier passes.
    On a failing base (the do-nothing Run, unless the Verifier rightly passes it), or failing at a cell or an
    atom, the sanity check was never tested (`untested`).
    """
    base, on_passing = _stray_base(runs, passing, _do_nothing(runs))
    if not on_passing and base is not None:  # a refusal Verifier is passed by the do-nothing Run rightly
        on_passing = _judge(verifier, base, canon, tools)[0]
    out = {}
    for name, run in constructed_runs(verifier, runs, passing, schema=schema, canon=canon).items():
        if run is None:
            out[name] = {"built": False, "failed": False, "atom": None, "run_id": None}
            continue
        passed, atom = _judge(verifier, run, canon, tools)
        out[name] = {"built": True, "failed": not passed, "atom": None if passed else atom, "run_id": run.run_id}
        if name == "stray_write" and not passed:
            untested = not on_passing or not at_sanity(atom)
            out[name].update(failed=not untested, untested=untested)
    return out


# --- the tier --------------------------------------------------------------------------------------

def _event_name(event: Any) -> str:
    event = event if isinstance(event, dict) else {}
    inner = event.get("event") if isinstance(event.get("event"), dict) else event
    return str(inner.get("name") or inner.get("type") or "")


def _event_task(event: Any) -> Optional[str]:
    event = event if isinstance(event, dict) else {}
    inner = event.get("event") if isinstance(event.get("event"), dict) else event
    payload = inner.get("payload") if isinstance(inner.get("payload"), dict) else inner
    return payload.get("task_id")


def set_aside_of(spec: Spec, rulings: Iterable[Any] = ()) -> bool:
    """The router's rule on the Spec first (`router.is_set_aside`), then a set-aside event about the Task on
    the bus, for a Spec file written before the router marked it."""
    return is_set_aside(spec) or any(_event_name(event) == TASK_SET_ASIDE
                                     and _event_task(event) in (None, spec.task_id) for event in rulings or ())


def _flag(verdicts: dict[str, tuple[bool, Any]], run_ids: Iterable[str], every: bool) -> Optional[bool]:
    scored = [verdicts[run_id][0] for run_id in run_ids if run_id in verdicts]
    if not scored:
        return None
    return all(scored) if every else any(scored)


def tier_of_task(spec: Spec, verifier: Verifier, runs: Iterable[Run], rulings: Iterable[Any] = (), *,
                 canon: Any, write_tools: Optional[Iterable[str]] = None, references: Iterable[str] = (),
                 schema: Any = None) -> TaskTier:
    """The tier of one Task under the Spec's Verifier, with the row of evidence behind it.

    `runs` are every Run of the Task, the replays included; `rulings` are the bus events about it;
    `references` are the run ids of the faithful replays of its kept References (`faithful_references`).
    """
    runs = list(runs or ())
    tools = set(write_tools) if write_tools is not None else None
    verdicts = {run.run_id: _judge(verifier, run, canon, tools) for run in runs}
    fresh = [run.run_id for run in runs if is_fresh(run)]
    kept = sorted(set(references) & set(verdicts))
    passing = [run_id for run_id in [*kept, *fresh] if verdicts[run_id][0]]
    constructed = _constructed(verifier, runs, passing, canon, tools, schema)
    set_aside = set_aside_of(spec, rulings)
    gates = {f"{name}_fails": constructed[name]["failed"] for name in constructed}
    gates["no_open_ruling"] = spec.rulings_open == 0 and not set_aside
    if is_empty(spec) or not spec.intent.facts or not verifier_items(verifier):
        reason = NO_INTENT
    elif not gates["no_open_ruling"]:
        reason = OPEN_RULING
    elif not all(gates[f"{name}_fails"] for name in constructed):
        reason = CONSTRUCTED_RUN_PASSED
    else:
        reason = None
    failed_reference = next((run_id for run_id in kept if not verdicts[run_id][0]), None)
    row = {"task_id": spec.task_id, "tier": UNTRUSTED if reason else TRUSTED, "reason": reason, "gates": gates,
           "constructed": constructed,
           "detail": STRAY_UNTESTED if (constructed.get("stray_write") or {}).get("untested") else None,
           "reference_passes": _flag(verdicts, kept, every=True), "solvable": _flag(verdicts, fresh, every=False),
           "reference": {"runs": kept, "failed": failed_reference,
                         "atom": verdicts[failed_reference][1] if failed_reference else None},
           "runs_scored": sorted(verdicts), "fresh_runs": len(fresh),
           "fresh_passed": sorted(r for r in fresh if verdicts[r][0]),
           "rulings_open": spec.rulings_open, "round": spec.round, "set_aside": set_aside,
           "code_hash": CODE_HASH}
    return TaskTier(row["tier"], row)


def verifier_items(verifier: Verifier) -> bool:
    """Whether the Verifier checks anything at all: an atom, an end state, a forbidden write or conduct."""
    return bool(verifier.atoms or verifier.expected or verifier.forbidden or verifier.conduct)


def unruled_row(task_id: str, reason: str, why: str) -> TaskTier:
    """A Task nothing can be scored on: untrusted for `reason`, with the words and no Run scored."""
    return TaskTier(UNTRUSTED, {"task_id": task_id, "tier": UNTRUSTED, "reason": reason, "why": why,
                                "gates": {}, "constructed": {}, "reference_passes": None, "solvable": None,
                                "runs_scored": [], "fresh_runs": 0, "fresh_passed": [],
                                "set_aside": reason == OPEN_RULING, "code_hash": CODE_HASH})


def missing_spec_row(task_id: str) -> TaskTier:
    """A Task with no Spec file: no Intent to verify against."""
    return unruled_row(task_id, NO_INTENT, "no spec")


def examiner_rulings(tiers: dict[str, TaskTier]) -> list[dict]:
    """The code rulings the Examiner reads: one per Task whose kept Reference fails its Verifier.

    A Reference failing is not a verdict on either side: the Examiner rules whether the Verifier or the
    Reference is wrong. Ids and the failing atom only; nothing from the recording.
    """
    out = []
    for task_id, task_tier in sorted(tiers.items()):
        reference = task_tier.row.get("reference") or {}
        if task_tier.row.get("reference_passes") is False:
            out.append({"task_id": task_id, "kind": REFERENCE_FAILS, "run_id": reference.get("failed"),
                        "atom": reference.get("atom"), "blocking": False})
    return out


# --- the build manifest ----------------------------------------------------------------------------

def _store(workdir: Any, key: str, value: Any) -> Path:
    """One key of the build manifest written, keeping whatever else the manifest holds."""
    path = Path(workdir) / BUILD_MANIFEST
    body = read_json(path, None)
    body = dict(body) if isinstance(body, dict) else {}
    body[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def store_tasks(workdir: Any, task_ids: Iterable[str]) -> Path:
    """Records the build's Task sample in the manifest: the only Tasks it plays or examines."""
    return _store(workdir, "tasks", [str(task_id) for task_id in task_ids])


def stored_tasks(workdir: Any) -> Optional[list[str]]:
    """The build's Task sample, or None when the build covers every Task."""
    body = read_json(Path(workdir) / BUILD_MANIFEST, None)
    value = body.get("tasks") if isinstance(body, dict) else None
    return [str(task_id) for task_id in value] if isinstance(value, list) else None


def in_sample(workdir: Any, task_ids: Iterable[str]) -> list[str]:
    """The named Tasks inside the build's sample, in their order; all of them when there is none."""
    sample = stored_tasks(workdir)
    if sample is None:
        return list(task_ids)
    kept = set(sample)
    return [task_id for task_id in task_ids if task_id in kept]


# --- one workdir -----------------------------------------------------------------------------------

def set_aside_events(workdir: Any) -> list[dict]:
    """The router's set-aside events off the workdir bus; a missing or torn bus reads as none."""
    try:
        records = read_jsonl(Path(workdir) / "bus.jsonl")
    except (OSError, ValueError):
        return []
    return [record for record in records if _event_name(record) == TASK_SET_ASIDE]


def _task_ids(root: Path) -> list[str]:
    status = read_json(root / "task_status.json", None)
    ids = set(status) if isinstance(status, dict) else set()
    return sorted(ids | set(spec_ids(root)))


def load_spec_verifier(workdir: Any, task_id: str) -> Optional[Verifier]:
    """The written Verifier as a record, or None when the file is not there or does not read as one."""
    path = spec_verifier_path(workdir, task_id)
    try:
        return Verifier.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def faithful_references(workdir: Any) -> dict[str, list[str]]:
    """task id -> run ids of the faithful replays (fidelity 1, confirmed) of the Task's kept References.

    The kept References are the harness's own (references.json); the replay index (replays.json) says how
    faithfully each recording was played back. A missing or torn file reads as none.
    """
    root = Path(workdir)
    references, replays = read_json(root / "references.json", None), read_json(root / "replays.json", None)
    kept = {(str(task_id), str(ref.get("trace_id")))
            for task_id, row in (references.items() if isinstance(references, dict) else ())
            for ref in ((row or {}).get("references") or []) if isinstance(ref, dict)}
    out: dict[str, list[str]] = {}
    for rows in (replays.values() if isinstance(replays, dict) else ()):
        for row in (rows.values() if isinstance(rows, dict) else ()):
            row = row if isinstance(row, dict) else {}
            if ((str(row.get("task_id")), str(row.get("trace_id"))) in kept and row.get("run_id")
                    and row.get("confirmed") is True and row.get("fidelity") == 1):
                out.setdefault(str(row["task_id"]), []).append(str(row["run_id"]))
    return out


def runs_on_disk(workdir: Any) -> dict[str, list[Run]]:
    """Every Run file the build wrote under runs/, per Task; a file that does not load is skipped."""
    out: dict[str, list[Run]] = {}
    for path in sorted((Path(workdir) / "runs").rglob("*.jsonl")):
        try:
            run = load_run_jsonl(path)
        except (OSError, ValueError, TypeError):
            continue
        if run.task_id:
            out.setdefault(str(run.task_id), []).append(run)
    return out


def intent_tiers(workdir: Any, specs: Optional[dict[str, Spec]] = None) -> dict[str, TaskTier]:
    """Every Task of the workdir under the intent foundation, the build's sample only when the manifest
    holds one; `specs` stands in for the spec folder."""
    root = Path(workdir)
    canon = load_rules(root / "canon-rules.json")
    sigs = read_json(root / "tool_sigs.json", None) or []
    tools = write_tools_of(sigs.get("sigs", []) if isinstance(sigs, dict) else sigs) - workdir_action_tools(root)
    runs = runs_on_disk(root)
    references = faithful_references(root)
    events = set_aside_events(root)
    schema = schema_of(root)
    out: dict[str, TaskTier] = {}
    for task_id in in_sample(root, sorted(set(_task_ids(root)) | set(specs or ()))):
        spec = (specs or {}).get(task_id) or load_spec(root, task_id)
        if spec is None:
            out[task_id] = missing_spec_row(task_id)
            continue
        verifier = load_spec_verifier(root, task_id)
        if verifier is None:  # an empty Spec is set aside before any Verifier is written
            reason = OPEN_RULING if set_aside_of(spec, events) and not is_empty(spec) else NO_INTENT
            out[task_id] = unruled_row(task_id, reason, spec.set_aside or "no verifier file")
            continue
        out[task_id] = tier_of_task(spec, verifier, runs.get(task_id, []), events, canon=canon,
                                    write_tools=tools, references=references.get(task_id, ()), schema=schema)
    return out


def workdir_tiers(workdir: Any) -> dict[str, TaskTier]:
    """Every Task's tier: the Spec writes every Verifier, so this is `intent_tiers` (D320)."""
    return intent_tiers(workdir)


def spec_tiers_of(workdir: Any) -> Optional[dict[str, TaskTier]]:
    """The Spec's tiers where the workdir has Specs, else None: what the gates take as `spec_tiers` (D322)."""
    root = Path(workdir)
    return workdir_tiers(root) if spec_ids(root) else None


def workdir_ruling(workdir: Any) -> Any:
    """The one trusted ruling over a workdir (D322): the Spec's tiers where it has Specs, read through
    `gates.trust.workdir_trusted_ruling` so status, the round snapshot and the report print one tier per
    Task from one place; a workdir with no Spec keeps the gates' own ruling."""
    root = Path(workdir)
    return workdir_trusted_ruling(root, spec_tiers=spec_tiers_of(root))
