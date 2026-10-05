"""The trust bar under `--verifier-from intent`: four gates, in order, all code (DESIGN.md, the trust bar).

1. grounded: `ground_spec` has nothing to say (every check quotes the Intent or names a policy section,
   every fact has a check or is listed as a gap);
2. can fail: the D79 mutations of a passing Run fail the Spec's Verifier (`can_fail`);
3. independent pass: a Run that is not a replay of a recording passes under `check_run`;
4. no open ruling: `spec.rulings_open == 0` and the Task is not set aside.

The replay's own verdict is kept on the row as evidence and is never the condition. The tiers are
trusted (all four), replay_only (1 and 2, the replay passes, no fresh Run passes yet), set_aside
(the router's rule, spec/router.py: marked on the Spec, or ROUNDS_CAP rounds with a ruling open;
a set-aside event on the bus counts too) and untrusted
(everything else). Every row names the first failing gate and the ids of the Runs it scored.

Under `--verifier-from runs` nothing here decides: `runs_tiers` is the D126 trusted ruling of
kullback/gates/trust.py mapped onto the tier words, so the switch cannot move that foundation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates.probes import write_tools_of
from kullback.gates.trust import trusted_gate, workdir_trusted_ruling
from kullback.runner.canon import load_rules
from kullback.runner.records import GateResult, Run, Verifier, load_exam_task_runs, read_json, read_jsonl
from kullback.runner.replay import RECORDED
from kullback.runner.target import check_run
from kullback.spec import VERIFIER_FROM
from kullback.spec.canfail import can_fail
from kullback.spec.events import TASK_SET_ASIDE
from kullback.spec.ground import ground_spec
from kullback.spec.router import (  # noqa: F401  (ROUNDS_CAP re-exported for readers of the tiers)
    ROUNDS_CAP,
    is_set_aside,
)
from kullback.spec.schema import Spec, load_spec, spec_ids
from kullback.spec.writer import spec_verifier_path
from kullback.spec.writer_tools import policy_sections_of

TIERS = ("trusted", "replay_only", "untrusted", "set_aside")
GATES = ("grounded", "can_fail", "independent_pass", "no_open_ruling")
# The build manifest in the workdir: the Verifier foundation the build chose, kept for a resume.
BUILD_MANIFEST = "build.json"
DEFAULT_VERIFIER_FROM = "runs"
# How many grounding refusals a row repeats; the count is always whole.
_REASONS_KEPT = 5


class TaskTier(NamedTuple):
    tier: str
    row: dict


# --- which Run is a replay -------------------------------------------------------------------------

def replay_prefix(replays: Any) -> Optional[str]:
    """The run id prefix the Runner gives a replay, read off its own replay index, or None.

    Each replay row names its run id and the recording it replays; the run id less the recording id
    is the prefix. The commonest one wins, so a row written by hand does not move it.
    """
    found: Counter = Counter()
    for rows in (replays or {}).values() if isinstance(replays, dict) else ():
        for row in (rows or {}).values() if isinstance(rows, dict) else ():
            run_id, trace_id = str((row or {}).get("run_id") or ""), str((row or {}).get("trace_id") or "")
            if trace_id and run_id.endswith(trace_id) and len(run_id) > len(trace_id):
                found[run_id[:-len(trace_id)]] += 1
    return found.most_common(1)[0][0] if found else None


def is_replay(run: Run, prefix: Optional[str]) -> bool:
    """A Run the recorded agent played: its id carries the replay prefix, or, with none known, its model is
    the recorded one."""
    if prefix:
        return run.run_id.startswith(prefix)
    return run.model == RECORDED


# --- the four gates --------------------------------------------------------------------------------

def _score(verifier: Verifier, runs: list[Run], canon: Any, write_tools: Any) -> dict[str, bool]:
    scores = {}
    for run in runs:
        try:
            scores[run.run_id] = bool(check_run(verifier, run, canon, write_tools=write_tools)[0])
        except Exception:  # a Run the scorer cannot read is not a pass, and never stops the ruling
            scores[run.run_id] = False
    return scores


def _mutation_base(runs: list[Run], scores: dict[str, bool], prefix: Optional[str]) -> Optional[Run]:
    """The Run the D79 mutations start from: a passing replay, else any passing Run, else the first replay.

    Mutating a Run the Verifier passes is what asks whether it can fail; a Verifier nothing passes is
    still asked, from the recording, so an empty Verifier is refused whatever the Runs did.
    """
    passing = [run for run in runs if scores.get(run.run_id)]
    replays = [run for run in runs if is_replay(run, prefix)]
    for pool in ([r for r in passing if is_replay(r, prefix)], passing, replays, runs):
        if pool:
            return pool[0]
    return None


def _can_fail(verifier: Verifier, runs: list[Run], scores: dict, prefix: Optional[str], canon: Any,
              write_tools: Any) -> tuple[bool, list[dict], Optional[str]]:
    base = _mutation_base(runs, scores, prefix)
    if base is None:
        return False, [], None
    try:
        result = can_fail(verifier, base, [run for run in runs if run is not base], canon=canon,
                          write_tools=write_tools)
    except Exception as exc:  # a suite that cannot run the mutations proves nothing can fail
        return False, [{"stage": "error", "passed": False, "failures": [type(exc).__name__]}], base.run_id
    return result.passed, result.rows, base.run_id


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


def _tier(gates: dict[str, bool], replay_passed: bool, set_aside: bool) -> str:
    if set_aside:
        return "set_aside"
    if all(gates.values()):
        return "trusted"
    if gates["grounded"] and gates["can_fail"] and replay_passed and not gates["independent_pass"]:
        return "replay_only"
    return "untrusted"


def tier_of_task(spec: Spec, verifier: Verifier, runs: Iterable[Run], rulings: Iterable[Any] = (), *,
                 canon: Any, write_tools: Optional[Iterable[str]] = None, policy_sections: Iterable[str] = (),
                 prefix: Optional[str] = None) -> TaskTier:
    """The tier of one Task under the Spec's Verifier, with the row of evidence behind it.

    `runs` are every Run of the Task, the replays included; `rulings` are the bus events about it.
    `prefix` is the Runner's replay prefix (`replay_prefix`); without one a replay is known by its
    recorded model.
    """
    runs = list(runs or ())
    tools = set(write_tools) if write_tools is not None else None
    refusals = ground_spec(spec, policy_sections)
    scores = _score(verifier, runs, canon, tools)
    fresh = [run.run_id for run in runs if not is_replay(run, prefix)]
    replay = {run.run_id: scores[run.run_id] for run in runs if is_replay(run, prefix)}
    failed_ok, mutation_rows, mutated_from = _can_fail(verifier, runs, scores, prefix, canon, tools)
    set_aside = set_aside_of(spec, rulings)
    gates = {"grounded": not refusals, "can_fail": failed_ok,
             "independent_pass": any(scores[run_id] for run_id in fresh),
             "no_open_ruling": spec.rulings_open == 0 and not set_aside}
    tier = _tier(gates, any(replay.values()), set_aside)
    row = {"task_id": spec.task_id, "tier": tier,
           "failing": next((name for name in GATES if not gates[name]), None), "gates": gates,
           "grounding": {"refusals": len(refusals), "first": refusals[:_REASONS_KEPT]},
           "can_fail": {"from": mutated_from, "rows": mutation_rows},
           "runs_scored": sorted(scores), "fresh_passed": sorted(r for r in fresh if scores[r]),
           "replay": replay, "rulings_open": spec.rulings_open, "round": spec.round, "set_aside": set_aside}
    return TaskTier(tier, row)


def unruled_row(task_id: str, failing: str, reason: str) -> TaskTier:
    """A Task the gates cannot rule on: untrusted at `failing`, with the reason and no Run scored."""
    gates = {name: False for name in GATES}
    return TaskTier("untrusted", {"task_id": task_id, "tier": "untrusted", "failing": failing, "gates": gates,
                                  "reason": reason, "runs_scored": [], "fresh_passed": [], "replay": {},
                                  "set_aside": False})


def set_aside_row(task_id: str, reason: str) -> TaskTier:
    """A Task set aside before any Verifier was written: tier set_aside, with the reason and no Run scored."""
    row = dict(unruled_row(task_id, "", reason).row, tier="set_aside", failing=None, set_aside=True)
    return TaskTier("set_aside", row)


def missing_spec_row(task_id: str) -> TaskTier:
    """A Task with no Spec file: nothing grounds it, so it is untrusted at the first gate."""
    return unruled_row(task_id, "grounded", "no spec")


# --- the runs foundation, unchanged ----------------------------------------------------------------

def tiers_of_ruling(ruling: GateResult) -> dict[str, TaskTier]:
    """The D126 trusted ruling in tier words: trusted stays trusted, every other ruled Task is untrusted.

    The runs foundation has no replay-only and no set-aside tier; its reasons ride on the row as they are.
    """
    metrics = ruling.metrics or {}
    out = {task_id: TaskTier("trusted", {"task_id": task_id, "tier": "trusted", "failing": None})
           for task_id in metrics.get("trusted") or []}
    for task_id, reason in (metrics.get("untrusted") or {}).items():
        out[task_id] = TaskTier("untrusted", {"task_id": task_id, "tier": "untrusted", "failing": "trusted_gate",
                                              "reason": reason})
    return out


def runs_tiers(**world: Any) -> dict[str, TaskTier]:
    """`gates.trust.trusted_gate` over the same inputs, in tier words; never a ruling of its own."""
    return tiers_of_ruling(trusted_gate(**world))


# --- the build manifest ----------------------------------------------------------------------------

def stored_verifier_from(workdir: Any) -> Optional[str]:
    """The foundation the workdir's build chose, or None when none was stored."""
    body = read_json(Path(workdir) / BUILD_MANIFEST, None)
    value = body.get("verifier_from") if isinstance(body, dict) else None
    return value if value in VERIFIER_FROM else None


def store_verifier_from(workdir: Any, value: str) -> Path:
    """Records the foundation in the build manifest, keeping whatever else the manifest holds."""
    if value not in VERIFIER_FROM:
        raise ValueError(f"verifier_from must be one of {', '.join(VERIFIER_FROM)}, not {value!r}")
    return _store(workdir, "verifier_from", value)


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


def verifier_from_of(workdir: Any) -> str:
    return stored_verifier_from(workdir) or DEFAULT_VERIFIER_FROM


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


def intent_tiers(workdir: Any, specs: Optional[dict[str, Spec]] = None) -> dict[str, TaskTier]:
    """Every Task of the workdir under the intent foundation, the build's sample only when the manifest
    holds one; `specs` stands in for the spec folder."""
    root = Path(workdir)
    canon = load_rules(root / "canon-rules.json")
    sigs = read_json(root / "tool_sigs.json", None) or []
    tools = write_tools_of(sigs.get("sigs", []) if isinstance(sigs, dict) else sigs)
    sections = policy_sections_of(root)
    runs = load_exam_task_runs(root)
    prefix = replay_prefix(read_json(root / "replays.json", None))
    events = set_aside_events(root)
    out: dict[str, TaskTier] = {}
    for task_id in in_sample(root, sorted(set(_task_ids(root)) | set(specs or ()))):
        spec = (specs or {}).get(task_id) or load_spec(root, task_id)
        if spec is None:
            out[task_id] = missing_spec_row(task_id)
            continue
        if set_aside_of(spec, events):
            out[task_id] = set_aside_row(task_id, spec.set_aside or "set aside")
            continue
        verifier = load_spec_verifier(root, task_id)
        if verifier is None:
            out[task_id] = unruled_row(task_id, "can_fail", "no verifier file")
            continue
        out[task_id] = tier_of_task(spec, verifier, runs.get(task_id, []), events, canon=canon,
                                    write_tools=tools, policy_sections=sections, prefix=prefix)
    return out


def workdir_tiers(workdir: Any, verifier_from: Optional[str] = None) -> dict[str, TaskTier]:
    """Every Task's tier under the foundation named, or the one the build manifest stored."""
    chosen = verifier_from or verifier_from_of(workdir)
    if chosen == "intent":
        return intent_tiers(workdir)
    return tiers_of_ruling(workdir_trusted_ruling(workdir))
