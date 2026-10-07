"""The trust bar on the Spec's Verifier: five gates, in order, all code (DESIGN.md, the trust bar).

1. grounded: `ground_spec` has nothing to say (every check quotes the Intent or names a policy section,
   every fact has a check or is listed as a gap);
2. can fail: the D79 mutations of a passing Run fail the Spec's Verifier (`can_fail`);
3. reference replay: the Verifier passes every faithful replay (fidelity 1, confirmed) of a recording the
   harness keeps as the Task's Reference (references.json and replays.json in the workdir);
4. independent pass: a live Candidate Run (not the recorded model) passes under the Verdict;
5. no open ruling: `spec.rulings_open == 0` and the Task is not set aside.

A replay's verdict is a condition only for a faithful kept Reference (gate 3); every other replay
verdict rides on the row as evidence. The tiers are
trusted (all five), replay_only (1 to 3, a replay passes, no fresh Run passes yet; its reason says whether
no fresh Run is on disk, "not run", or every fresh Run failed), set_aside
(the router's rule, spec/router.py: marked on the Spec, or ROUNDS_CAP rounds with a ruling open;
a set-aside event on the bus counts too) and untrusted
(everything else). Every row names the first failing gate and the ids of the Runs it scored, and carries
`code_hash`, the content hash of the Runner, gates and Spec code that scored it (runner/code_hash.py): two
rows are comparable when their hashes match.

The row also carries `writer_disagrees` (D327): where the Spec writer, who never sees a Run, and the kept
Reference disagree on what was written. Trust by agreement cannot see a Reference the fresh Runs copy; the
writer's blind reading is the one signal that can, so it rides on the row as a flag and never gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates.probes import write_tools_of
from kullback.gates.trust import code_ruling, schema_of, workdir_trusted_ruling
from kullback.runner.atom_context import AtomContext
from kullback.runner.canon import load_rules
from kullback.runner.code_hash import CODE_HASH
from kullback.runner.expected import action_tables, attr_of
from kullback.runner.records import Run, Verifier, load_run_jsonl, read_json, read_jsonl
from kullback.runner.replay import RECORDED
from kullback.spec.actions import action_tools_of, workdir_action_tools
from kullback.spec.canfail import can_fail, judge_run
from kullback.spec.events import TASK_SET_ASIDE
from kullback.spec.ground import ground_spec
from kullback.spec.router import (  # noqa: F401  (ROUNDS_CAP re-exported for readers of the tiers)
    ROUNDS_CAP,
    is_set_aside,
)
from kullback.spec.schema import Spec, load_spec, spec_ids
from kullback.spec.writer import spec_verifier_path
from kullback.spec.writer_tools import policy_sections_of

TIERS = ("trusted", "replay_only", "untrusted", "set_aside", "unconfirmed", "pending", "refused")
GATES = ("grounded", "sourced", "can_fail", "reference_replay", "independent_pass", "no_open_ruling")
# The code ruling's tiers (gates/trust.py, D319) that stand in for the Spec's when its gate fails.
_CODE_TIERS = ("unconfirmed", "refused", "pending")
# The build manifest in the workdir: the build's Task sample, kept for a resume.
BUILD_MANIFEST = "build.json"
# How many grounding refusals a row repeats; the count is always whole.
_REASONS_KEPT = 5
# Why a Task is replay_only: nothing was played, or what was played failed (D327).
NOT_RUN = "not run: no fresh Run on disk"
NO_FRESH_PASS = "no fresh Run passes"


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


# --- the five gates ---------------------------------------------------------------------------------

def _score(verifier: Verifier, runs: list[Run], canon: Any, write_tools: Any) -> dict[str, tuple[bool, Any]]:
    """run id -> (passed, the first failing atom id or None)."""
    verdicts = {}
    for run in runs:
        try:
            verdicts[run.run_id] = judge_run(verifier, run, canon, write_tools)
        except Exception:  # a Run the scorer cannot read is not a pass, and never stops the ruling
            verdicts[run.run_id] = (False, None)
    return verdicts


def _reference_replay(verdicts: dict, references: Iterable[str]) -> dict:
    """The faithful Reference replays scored, and the first the Verifier fails with its failing atom."""
    scored = sorted(run_id for run_id in set(references) if run_id in verdicts)
    failed = next((run_id for run_id in scored if not verdicts[run_id][0]), None)
    return {"runs": scored, "failed": failed, "atom": verdicts[failed][1] if failed else None}


def _mutation_base(runs: list[Run], scores: dict[str, bool]) -> Optional[Run]:
    """The Run the D79 mutations start from: a passing replay, else any passing Run, else the first replay.

    Mutating a Run the Verifier passes is what asks whether it can fail; a Verifier nothing passes is
    still asked, from the recording, so an empty Verifier is refused whatever the Runs did.
    """
    passing = [run for run in runs if scores.get(run.run_id)]
    replays = [run for run in runs if is_replay(run)]
    for pool in ([r for r in passing if is_replay(r)], passing, replays, runs):
        if pool:
            return pool[0]
    return None


def _can_fail(verifier: Verifier, runs: list[Run], scores: dict, canon: Any, write_tools: Any) -> tuple[bool, list[dict], Optional[str]]:
    base = _mutation_base(runs, scores)
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


def _row_parts(row_id: str, separator: str) -> set[str]:
    """A diffed row's id and, for a composite key, each of its parts, compared without case or spacing."""
    parts = {row_id, *row_id.split(separator)} if separator else {row_id}
    return {" ".join(part.split()).lower() for part in parts}


def writer_disagreement(spec: Spec, reference: Optional[Run], schema: Any = None) -> list[str]:
    """Where the Spec's write demands and the Reference's diff disagree, by code (D327); empty without one.

    Each required write demand whose tool the Reference never called, or whose named row is in no diffed
    row, is named by its check id. An action tool's demand is read by its call alone: its rows are the
    harness's record of the call, never a change, so they never count. A diffed row no demand names is
    not flagged: it fired on right References only (docs/validation/trust-1007.md). A flag, never a gate.
    """
    if reference is None:
        return []
    context = AtomContext(reference)
    separator = str(attr_of(schema, "key_separator") or "")
    actions, action_tools = action_tables(schema), action_tools_of(schema or {})
    rows = {key: _row_parts(key.partition(".")[2], separator) for key in context.diff()
            if key.partition(".")[0] not in actions}
    out = []
    for check in spec.checks:
        if check.demand.get("demand") != "write" or check.kind != "required":
            continue
        entity = " ".join(str(check.demand.get("entity") or "").split()).lower()
        if not context.called(str(check.demand.get("tool") or "")):
            out.append(f"{check.id}: no call to {check.demand.get('tool')}")
        elif check.demand.get("tool") not in action_tools and not any(entity in parts for parts in rows.values()):
            out.append(f"{check.id}: its row is in no diffed row")
    return out


def set_aside_of(spec: Spec, rulings: Iterable[Any] = ()) -> bool:
    """The router's rule on the Spec first (`router.is_set_aside`), then a set-aside event about the Task on
    the bus, for a Spec file written before the router marked it."""
    return is_set_aside(spec) or any(_event_name(event) == TASK_SET_ASIDE
                                     and _event_task(event) in (None, spec.task_id) for event in rulings or ())


def _tier(gates: dict[str, bool], replay_passed: bool, set_aside: bool, code_tier: str = "trusted") -> str:
    if set_aside:
        return "set_aside"
    if code_tier in _CODE_TIERS:
        return code_tier
    if all(gates.values()):
        return "trusted"
    if (gates["grounded"] and gates["can_fail"] and gates["reference_replay"] and replay_passed
            and not gates["independent_pass"]):
        return "replay_only"
    return "untrusted"


def tier_of_task(spec: Spec, verifier: Verifier, runs: Iterable[Run], rulings: Iterable[Any] = (), *,
                 canon: Any, write_tools: Optional[Iterable[str]] = None, policy_sections: Iterable[str] = (),
                 references: Iterable[str] = (), status: Any = None, schema: Any = None) -> TaskTier:
    """The tier of one Task under the Spec's Verifier, with the row of evidence behind it.

    `runs` are every Run of the Task, the replays included; `rulings` are the bus events about it.
    A replay is known by its recorded model (`is_replay`), never by its run id. `references` are the
    run ids of the faithful replays of the Task's kept References (`faithful_references`). The
    `sourced` gate is `gates.trust.code_ruling` on those replays and the Task's `status` row (D322):
    unconfirmed, refused and pending become the tier with its reason; the consistency checks (D321)
    ride on the row as `contradicts` and `unasked` and never gate.
    """
    runs = list(runs or ())
    tools = set(write_tools) if write_tools is not None else None
    refusals = ground_spec(spec, policy_sections)
    verdicts = _score(verifier, runs, canon, tools)
    scores = {run_id: passed for run_id, (passed, _atom) in verdicts.items()}
    reference = _reference_replay(verdicts, references)
    fresh = [run.run_id for run in runs if is_fresh(run)]
    replay = {run.run_id: scores[run.run_id] for run in runs if is_replay(run)}
    failed_ok, mutation_rows, mutated_from = _can_fail(verifier, runs, scores, canon, tools)
    set_aside = set_aside_of(spec, rulings)
    kept = set(references)
    code = code_ruling(verifier, [run for run in runs if run.run_id in kept], status or {}, canon, tools, schema)
    disagrees = writer_disagreement(spec, next((run for run in runs if run.run_id in kept), None), schema)
    gates = {"grounded": not refusals, "sourced": code.tier == "trusted", "can_fail": failed_ok, "reference_replay": reference["failed"] is None,
             "independent_pass": any(scores[run_id] for run_id in fresh),
             "no_open_ruling": spec.rulings_open == 0 and not set_aside}
    tier = _tier(gates, any(replay.values()), set_aside, code.tier)
    failing = code.reason if tier in _CODE_TIERS else next((name for name in GATES if not gates[name]), None)
    if tier == "replay_only":  # not run is not failed: the reason says which (D327)
        failing = NO_FRESH_PASS if fresh else NOT_RUN
    row = {"task_id": spec.task_id, "tier": tier, "failing": failing, "gates": gates,
           "sourced": {"tier": code.tier, "reason": code.reason, "unsupported": code.unsupported},
           "contradicts": list(code.contradicts), "unasked": list(code.unasked), "writer_disagrees": disagrees,
           "grounding": {"refusals": len(refusals), "first": refusals[:_REASONS_KEPT]},
           "can_fail": {"from": mutated_from, "rows": mutation_rows}, "reference_replay": reference,
           "runs_scored": sorted(scores), "fresh_runs": len(fresh),
           "fresh_passed": sorted(r for r in fresh if scores[r]),
           "replay": replay, "rulings_open": spec.rulings_open, "round": spec.round, "set_aside": set_aside,
           "code_hash": CODE_HASH}
    return TaskTier(tier, row)


def unruled_row(task_id: str, failing: str, reason: str) -> TaskTier:
    """A Task the gates cannot rule on: untrusted at `failing`, with the reason and no Run scored."""
    gates = {name: False for name in GATES}
    return TaskTier("untrusted", {"task_id": task_id, "tier": "untrusted", "failing": failing, "gates": gates,
                                  "reason": reason, "runs_scored": [], "fresh_runs": 0, "fresh_passed": [],
                                  "replay": {}, "writer_disagrees": [],
                                  "set_aside": False, "code_hash": CODE_HASH})


def set_aside_row(task_id: str, reason: str) -> TaskTier:
    """A Task set aside before any Verifier was written: tier set_aside, with the reason and no Run scored."""
    row = dict(unruled_row(task_id, "", reason).row, tier="set_aside", failing=None, set_aside=True)
    return TaskTier("set_aside", row)


def missing_spec_row(task_id: str) -> TaskTier:
    """A Task with no Spec file: nothing grounds it, so it is untrusted at the first gate."""
    return unruled_row(task_id, "grounded", "no spec")


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
    sections = policy_sections_of(root)
    runs = runs_on_disk(root)
    references = faithful_references(root)
    events = set_aside_events(root)
    statuses = read_json(root / "task_status.json", None)
    statuses = statuses if isinstance(statuses, dict) else {}
    schema = schema_of(root)
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
                                    write_tools=tools, policy_sections=sections,
                                    references=references.get(task_id, ()), status=statuses.get(task_id),
                                    schema=schema)
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
