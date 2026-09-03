"""The Examiner's seven tools: pydantic arguments in, a short result out, every ruling from a gate (D123, D127, D128).

`read` is the Examiner's whole read surface: a Task, a Trace, an Intent, a Run, a Verifier, a
probe pool, the task status, the rulings, the re-roll and replay rows, the References. `derive`
runs the derivation over every Task (or one) and is the Builder's old derive_verifier stage, byte
for byte. `probe` scores one hand-written Run against the current Verifier and keeps it in the
Task's pool forever. `repair` proposes a new Verifier version and the gates decide whether it is
accepted: the D79 suite, the pool, one-directional loosening. `refuse` asks to give a Task up and
the refuse gate admits it only when no frontier Run finished. `reroll` buys more frontier Runs
through the Runner callable the Builder handed over. `finding` files what is wrong on the
Builder's side. Nothing here writes a tool body, a table or the Environment, and no tool decides a
ruling: each one calls a registered gate and reports what it said.

A result is what the model reads plus what it does not: the rendered text is the summary and the
rulings on one line each, the rest is in `details` for the transcript.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.events import StageEnd, StageStart
from kullback.agent.tools import AgentTool
from kullback.examiner import stage as stage_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.gates import Ruling, artifacts, ruling_line, ruling_of, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates.loosening import loosening_gate
from kullback.gates.probes import (
    consecutive_failed,
    probe_admission_gate,
    probe_pool_gate,
    version_hash,
    write_tools_of,
)
from kullback.gates.trust import finished_runs, refuse_gate
from kullback.gates.verifier_suite import check_run, make_atom
from kullback.runner import budget
from kullback.runner.records import (
    Event,
    Finding,
    GateResult,
    Intent,
    Probe,
    ProbePool,
    Refusal,
    Run,
    Verifier,
    VerifierHistory,
    VerifierVersion,
    apply_intent,
    as_dict,
    read_json,
    write_json,
)

Sink = Callable[[Any], Awaitable[None]]
STAGE = stage_mod.STAGE
ReadKind = Literal["task", "trace", "intent", "run", "verifier", "probes", "task_status", "gates", "rerolls",
                   "replays", "references"]
# The eight classes of skills.BUG_CLASSES, spelled out so the tool schema names them; the test pins the two.
BugClass = Literal["loose answer extraction", "missing final-answer markers", "numeric-tolerance abuse",
                   "schema-only validation", "extra-field acceptance", "visible-test overfitting",
                   "stdout spoofing", "missing timeouts", "other"]


def render(result: BaseModel) -> str:
    """The summary line, then the rulings on one line; everything else stays in details."""
    lines = [getattr(result, "summary", None) or getattr(result, "text", "") or ""]
    rulings = getattr(result, "rulings", None)
    if rulings:
        lines.append(ruling_line("rulings", rulings))
    return "\n".join(line for line in lines if line)


# --- argument and result models ---------------------------------------------------

class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ReadKind = Field(description="What to read: task, trace, intent, run, verifier, probes, task_status, "
                                       "gates, rerolls, replays or references.")
    id: Optional[str] = Field(default=None, description="The Task, Trace, Run or ruling name; every row when "
                                                        "omitted where the kind allows it.")


class ReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    id: Optional[str] = None
    text: str


class DeriveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(default="all", description="`all` for every Task, or one Task id.")


class DeriveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    target: str
    status: str
    verifiers: list[str] = Field(default_factory=list)
    passed: int = 0
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["verifiers", "task_status", "history"])


class ProbeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task whose Verifier the probe attacks.")
    bug_class: BugClass = Field(description="One of the eight classes of the probe skill, or other.")
    note: str = Field(default="", description="What the probe changes and why the Task is not done.")
    events: list[dict] = Field(description="The probe's events, in the Runner's shape: type and payload each.")
    termination_reason: str = Field(default="success", description="How the Run ended, in the Runner's words.")
    base_run_id: Optional[str] = Field(default=None, description="The Run the probe was edited from.")


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    probe_id: str
    task_id: str
    scored_pass: bool
    failing_atom: Optional[str] = None
    pool_size: int
    consecutive_failed: int
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["probes"])


class RepairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="Why the current version is wrong and what the new one changes.")
    drop: list[str] = Field(default_factory=list, description="Atom ids to remove.")
    add: list[dict] = Field(default_factory=list,
                            description="Atoms to add: id, kind (required, allowed, question, communicate, hard), "
                                        "payload (the atom's target) and an optional description.")


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    content_hash: str
    verifier_version: str
    accepted: bool
    rejected_by: list[str] = Field(default_factory=list)
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["verifiers", "history", "task_status"])


class RefuseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str


class RefuseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    admitted: bool
    finished_runs: list[str] = Field(default_factory=list)
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["refusals", "task_status"])


class RerollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    count: int = Field(default=1, ge=1, le=10, description="How many frontier Runs to buy (D112, D133).")


class RerollResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    runs: list[str] = Field(default_factory=list)
    finished: int = 0
    spent_usd: float = 0.0
    produced: list[str] = Field(default_factory=lambda: ["rerolls", "task_runs"])


class FindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = None
    kind: Literal["assisted_tool", "fidelity", "reference_disagreement", "environment", "other"]
    text: str
    run_id: Optional[str] = None
    tool: Optional[str] = None
    suggested: Literal["compile_tool", "replay", "reroll", "none"] = "none"
    about_call_id: Optional[str] = Field(default=None, description="The tool call whose result the finding is about.")


class FindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    finding_id: str
    finding: dict = Field(default_factory=dict)
    produced: list[str] = Field(default_factory=lambda: ["findings"])


# --- helpers ---------------------------------------------------------------------------

def _task(plan: ExaminerPlan, task_id: str):
    task = next((t for t in plan.inputs.get("tasks") or [] if t.id == task_id), None)
    if task is None:
        raise KeyError(f"no Task is named {task_id}")
    return task


def _current(plan: ExaminerPlan, task_id: str) -> Verifier:
    verifier = plan.current(task_id)
    if verifier is None:
        raise LookupError(f"task {task_id} has no current Verifier; derive first")
    return verifier


def _text(body: Any) -> str:
    return json.dumps(body, indent=2, sort_keys=True, default=str, ensure_ascii=False)


def _find_run_path(plan: ExaminerPlan, run_id: str) -> Optional[str]:
    for rows in (plan.store.get("replays") or {}).values():
        for row in (rows or {}).values():
            if row.get("run_id") == run_id and row.get("path"):
                return row["path"]
    for rows in (plan.store.get("rerolls") or {}).values():
        for row in rows or []:
            if row.get("run_id") == run_id and row.get("path"):
                return row["path"]
    return None


def _find_run(plan: ExaminerPlan, run_id: str) -> Run:
    path = _find_run_path(plan, run_id)
    if path is not None:
        return verifier_suite.load_run(path)
    for pool in (plan.store.get("probes") or {}).values():
        for probe in pool.probes:
            if probe.probe_id == run_id or probe.run.run_id == run_id:
                return probe.run
    raise KeyError(f"no Run is named {run_id} among the replays, the re-rolls and the probes")


def _reference_paths(plan: ExaminerPlan, task_id: str) -> list[str]:
    """The References' paths, references.json run_ids joined back to the replay and re-roll rows."""
    references = read_json(plan.workdir / "references.json", {}) or {}
    rows = (references.get(task_id) or {}).get("references") or []
    paths = [path for path in (_find_run_path(plan, row["run_id"]) for row in rows) if path]
    if not paths:
        raise LookupError(f"task {task_id} has no Reference on disk; derive first")
    return paths


def _rules_trace(plan: ExaminerPlan, task_id: str) -> Optional[str]:
    references = read_json(plan.workdir / "references.json", {}) or {}
    rows = (references.get(task_id) or {}).get("references") or []
    return next((row.get("trace_id") for row in rows if row.get("trace_id")), None)


def _events(rows: list[dict]) -> list[Event]:
    out = []
    for number, row in enumerate(rows):
        body = dict(row)
        body.setdefault("idx", number)
        out.append(Event.model_validate(body))
    return out


def _may_probe(plan: ExaminerPlan) -> bool:
    return plan.probe_model is not None and plan.run_probe is not None and \
        (plan.probe_limit is None or plan.probe_limit > 0)


def _atom_of(row: dict):
    body = dict(row)
    atom_id, kind = body.pop("id"), body.pop("kind")
    payload = body.pop("payload", None) or body.pop("target", None) or {}
    return make_atom(atom_id, kind, payload, helpers=verifier_suite.HELPERS_SRC, **body)


async def _emit(plan: ExaminerPlan, sink: Optional[Sink], event: Any) -> None:
    if plan.on_event is not None:
        plan.on_event(event)
    if sink is not None:
        await sink(event)


# --- the executors -----------------------------------------------------------------------

def _read(plan: ExaminerPlan):
    async def read(args: ReadArgs) -> ReadResult:
        kind, key = args.kind, args.id
        store = plan.store
        if kind == "task":
            body = as_dict(_task(plan, key or ""))
        elif kind == "trace":
            trace = next((t for t in plan.inputs.get("traces") or [] if t.trace_id == key), None)
            if trace is None:
                raise KeyError(f"no Trace is named {key}")
            body = as_dict(trace)
        elif kind == "intent":
            intents = plan.inputs.get("intents") or {}
            if key not in intents:
                raise KeyError(f"no Intent for task {key}")
            body = as_dict(Intent.model_validate(intents[key]))
        elif kind == "run":
            body = as_dict(_find_run(plan, key or ""))
        elif kind == "verifier":
            body = as_dict(_current(plan, key or ""))
        elif kind == "probes":
            pool = (store.get("probes") or {}).get(key)
            body = as_dict(pool) if pool is not None else as_dict(ProbePool(task_id=key or ""))
        elif kind == "task_status":
            status = store.get("task_status") or {}
            body = status if key is None else {key: status.get(key)}
        elif kind == "gates":
            rows = read_json(plan.workdir / "gates.json", []) or []
            body = rows if key is None else [row for row in rows if row.get("stage") == key]
        elif kind == "rerolls":
            rows = store.get("rerolls") or {}
            body = rows if key is None else {key: rows.get(key, [])}
        elif kind == "replays":
            rows = store.get("replays") or {}
            body = rows if key is None else {key: rows.get(key, {})}
        else:
            rows = read_json(plan.workdir / "references.json", {}) or {}
            body = rows if key is None else {key: rows.get(key)}
        return ReadResult(kind=kind, id=key, text=_text(body))

    return read


def _derive(plan: ExaminerPlan, sink: Optional[Sink]):
    async def derive(args: DeriveArgs) -> DeriveResult:
        only = None if args.target == "all" else args.target
        ctx = stage_mod.ExamContext(plan.workdir, plan.ledger, anchor=plan.anchor)
        # derive_all rewrites task_status.json and references.json before the loosening gate rules:
        # the pre-derivation rows, so a rejected version restores a version-consistent artifact set.
        prior = {name: read_json(plan.workdir / name, {}) or {}
                 for name in ("task_status.json", "references.json")}
        started = time.monotonic()
        await _emit(plan, sink, StageStart(name=STAGE))
        try:
            out = await asyncio.to_thread(
                stage_mod.derive_all, ctx, plan.inputs, probe_model=plan.probe_model,
                probe_limit=plan.probe_limit, judge_model=plan.judge_model, run_probe=plan.run_probe, only=only)
        except Exception as exc:
            await _emit(plan, sink, StageEnd(name=STAGE, counts={
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": int((time.monotonic() - started) * 1000)}))
            raise
        status = out["task_status"]
        passed = sum(1 for row in status.values() if row.get("verifier_passed"))
        counts = {"status": "ran", "tasks": len(status), "verifiers": len(out["verifiers"]), "passed": passed,
                  "elapsed_ms": int((time.monotonic() - started) * 1000)}
        await _emit(plan, sink, StageEnd(name=STAGE, counts=counts))
        plan.load_state()
        loosening = _record_versions(plan, out["verifiers"], prior)
        rulings = [ruling_of(r) for r in ctx.recorded] + [ruling_of(r) for r in loosening]
        plan.last_rulings = rulings
        failed = [r.stage for r in rulings if not r.passed]
        summary = (f"derive {args.target}: {len(out['verifiers'])} Verifiers over {len(status)} Tasks, "
                   f"{passed} passed the D79 suite" + (f"; failed rulings: {', '.join(failed)}" if failed else ""))
        return DeriveResult(summary=summary, target=args.target, status="ran",
                            verifiers=[v.task_id for v in out["verifiers"]], passed=passed, rulings=rulings)

    return derive


def _restore_prior_row(plan: ExaminerPlan, filename: str, store_key: Optional[str], task_id: str,
                       prior: dict) -> None:
    """One Task's pre-derivation row back, in the store and on disk: a rejected version must leave
    the artifact set version-consistent, the restored Verifier against its own metadata."""
    rows = read_json(plan.workdir / filename, {}) or {}
    before = (prior.get(filename) or {})
    if task_id in before:
        rows[task_id] = before[task_id]
    else:
        rows.pop(task_id, None)
    write_json(plan.workdir / filename, rows)
    if store_key is not None:
        store_rows = dict(plan.store.get(store_key) or {})
        if task_id in before:
            store_rows[task_id] = before[task_id]
        else:
            store_rows.pop(task_id, None)
        plan.store[store_key] = store_rows


def _record_versions(plan: ExaminerPlan, verifiers: list, prior: dict) -> list[GateResult]:
    """One `derive` row per Task whose derived hash is not the history's last row (D127).

    A Task's first version is accepted as derived. A Task that already has an accepted version gets
    the derived one as a new version like any other: the loosening gate rules on it against the last
    accepted version, and when it fails (a derivation that would undo a repair's tightening) the row
    is recorded rejected and the accepted version is put back on disk together with its own
    task_status.json and references.json rows, from `prior`. The loosening rulings are recorded in
    gates.json and returned.
    """
    history = plan.store.setdefault("history", {})
    rulings: list[GateResult] = []
    changed = False
    rejected = False
    for record in verifiers:
        digest = version_hash(record)
        hist = history.get(record.task_id) or VerifierHistory(task_id=record.task_id)
        if hist.versions and hist.versions[-1].content_hash == digest:
            continue
        accepted = [v for v in hist.versions if v.accepted]
        row = VerifierVersion(
            task_id=record.task_id, content_hash=digest, verifier_version=str(len(hist.versions) + 1),
            parent_hash=accepted[-1].content_hash if accepted else None, round=plan.round, by="derive",
            reason="derived from the References", accepted=not accepted, verifier=record)
        if accepted:
            trial = hist.model_copy(deep=True)
            trial.versions.append(row)
            ruling = plan.ledger.record(STAGE, loosening_gate(
                {record.task_id: trial}, plan.store.get("task_runs") or {}, plan.store.get("replays") or {},
                plan.store.get("rerolls") or {}, plan.store.get("canon_rules"), plan.store.get("sigs") or []))
            rulings.append(ruling)
            if ruling.passed:
                row = row.model_copy(update={"accepted": True})
            else:
                row = row.model_copy(update={"rejected_by": ["loosening"]})
                plan.set_current(accepted[-1].verifier)
                _restore_prior_row(plan, "task_status.json", "task_status", record.task_id, prior)
                _restore_prior_row(plan, "references.json", None, record.task_id, prior)
                rejected = True
        hist.versions.append(row)
        history[record.task_id] = hist
        changed = True
    if changed:
        plan.write_state()
    if rejected:
        # derive_all rewrote scorecard.json from the rejected rows before the gate ruled; recompute
        # it from the restored files so the customer-facing report reads the version-consistent world.
        # Recomputed, not snapshotted: other Tasks' accepted derivations stay counted.
        write_json(plan.workdir / "scorecard.json", scorecard_mod.scorecard(plan.workdir))
    return rulings


def _probe(plan: ExaminerPlan):
    async def probe(args: ProbeArgs) -> ProbeResult:
        _task(plan, args.task_id)
        verifier = _current(plan, args.task_id)
        pools = plan.store.setdefault("probes", {})
        admission = probe_admission_gate(pools, plan.store["verifiers"])
        if args.task_id in admission.metrics.get("closed", []):
            reason = next((f for f in admission.failures if f.startswith(f"task {args.task_id}:")),
                          admission.failures[0] if admission.failures else "closed")
            raise PermissionError(f"probe refused: {reason}")
        pool = pools.get(args.task_id) or ProbePool(task_id=args.task_id)
        probe_id = f"probe-{args.task_id}-{len(pool.probes) + 1}"
        run = Run(run_id=probe_id, env_id=plan.env_id, task_id=args.task_id, model="probe:examiner",
                  events=_events(args.events), termination_reason=args.termination_reason,
                  parent_run_id=args.base_run_id)
        write_tools = write_tools_of(plan.store.get("sigs") or [])
        passed, failing = check_run(verifier, run, plan.store.get("canon_rules"), write_tools=write_tools)
        digest = version_hash(verifier)
        pool.probes.append(Probe(probe_id=probe_id, task_id=args.task_id, bug_class=args.bug_class, note=args.note,
                                 base_run_id=args.base_run_id, verifier_hash=digest, round=plan.round,
                                 scored_pass=bool(passed), run=run))
        pools[args.task_id] = pool
        plan.write_state()
        pool_gate = plan.ledger.record("probe", probe_pool_gate(plan.store["verifiers"], pools,
                                                                 plan.store.get("canon_rules"),
                                                                 plan.store.get("sigs") or []))
        admission = plan.ledger.record("probe", probe_admission_gate(pools, plan.store["verifiers"]))
        rulings = [ruling_of(pool_gate), ruling_of(admission)]
        plan.last_rulings = rulings
        tail = consecutive_failed(pool, digest)
        verdict = "scores a pass (the Verifier is loose)" if passed else f"rejected at {failing}"
        summary = (f"probe {probe_id} ({args.bug_class}) on task {args.task_id}: {verdict}; "
                   f"pool holds {len(pool.probes)}, {tail} rejected in a row against version {digest[:12]}")
        return ProbeResult(summary=summary, probe_id=probe_id, task_id=args.task_id, scored_pass=bool(passed),
                           failing_atom=failing if not passed else None, pool_size=len(pool.probes),
                           consecutive_failed=tail, rulings=rulings)

    return probe


def _repair(plan: ExaminerPlan):
    async def repair(args: RepairArgs) -> RepairResult:
        task = _task(plan, args.task_id)
        current = _current(plan, args.task_id)
        history = plan.store.setdefault("history", {})
        hist = history.get(args.task_id) or VerifierHistory(task_id=args.task_id)
        if not hist.versions:
            hist.versions.append(VerifierVersion(
                task_id=args.task_id, content_hash=version_hash(current), verifier_version=str(1),
                round=plan.round, by="derive", reason="the Verifier on disk before any repair",
                accepted=True, verifier=current))
        drop = set(args.drop)
        atoms = [a for a in current.atoms if a.id not in drop] + [_atom_of(row) for row in args.add]
        if not atoms:
            raise ValueError("a repair cannot leave the Verifier without atoms")
        candidate = current.model_copy(deep=True, update={
            "atoms": atoms, "verifier_version": str(len(hist.versions) + 1)})
        canon_rules = plan.store.get("canon_rules")
        sigs = plan.store.get("sigs") or []
        write_tools = write_tools_of(sigs)
        paths = _reference_paths(plan, args.task_id)
        intents = plan.inputs.get("intents") or {}
        task_for = (apply_intent(task, Intent.model_validate(intents[task.id])) if task.id in intents else task)
        gates = stage_mod.suite_for(task_for, candidate, paths, canon_rules=canon_rules, write_tools=write_tools,
                                    user_rules=plan.inputs.get("user_rules") or {},
                                    rules_trace=_rules_trace(plan, args.task_id), probe_model=plan.probe_model,
                                    run_probe=plan.run_probe, may_probe=_may_probe(plan))
        results = verifier_suite.d79_results(gates)
        d79 = artifacts.verifier_gate(results)
        pools = plan.store.get("probes") or {}
        pool_gate = probe_pool_gate([candidate], {k: v for k, v in pools.items() if k == args.task_id},
                                    canon_rules, sigs)
        trial = hist.model_copy(deep=True)
        digest = version_hash(candidate)
        row = VerifierVersion(task_id=args.task_id, content_hash=digest, verifier_version=candidate.verifier_version,
                              parent_hash=version_hash(current), round=plan.round, by="repair", reason=args.reason,
                              accepted=False, verifier=candidate)
        trial.versions.append(row)
        loosening = loosening_gate({args.task_id: trial}, plan.store.get("task_runs") or {},
                                   plan.store.get("replays") or {}, plan.store.get("rerolls") or {},
                                   canon_rules, sigs)
        rejected_by = [g.stage for g in gates if not g.passed]
        if not pool_gate.passed:
            rejected_by.append(pool_gate.stage)
        if not loosening.passed:
            rejected_by.append(loosening.stage)
        accepted = d79.passed and pool_gate.passed and loosening.passed
        row = row.model_copy(update={"accepted": accepted, "rejected_by": rejected_by})
        hist.versions.append(row)
        history[args.task_id] = hist
        if accepted:
            plan.set_current(candidate)
            status = plan.store.setdefault("task_status", {})
            entry = dict(status.get(args.task_id) or {})
            entry.update({"verifier_passed": True, "checks": results,
                          "not_run": [g.stage for g in gates if g.metrics.get("skipped")]})
            status[args.task_id] = entry
            write_json(plan.workdir / "task_status.json", status)
        plan.write_state()
        # The build-wide rulings land in gates.json; the result carries the candidate's own.
        plan.ledger.record("repair", probe_pool_gate(plan.store["verifiers"], pools, canon_rules, sigs))
        plan.ledger.record("repair", loosening_gate(history, plan.store.get("task_runs") or {},
                                                    plan.store.get("replays") or {},
                                                    plan.store.get("rerolls") or {}, canon_rules, sigs))
        rulings = [ruling_of(d79), ruling_of(pool_gate), ruling_of(loosening)]
        plan.last_rulings = rulings
        outcome = "accepted" if accepted else f"rejected by {', '.join(rejected_by)}"
        summary = (f"repair of task {args.task_id}: version {candidate.verifier_version} ({digest[:12]}) {outcome}; "
                   f"{len(atoms)} atoms, {len(args.drop)} dropped, {len(args.add)} added")
        return RepairResult(summary=summary, task_id=args.task_id, content_hash=digest,
                            verifier_version=candidate.verifier_version, accepted=accepted,
                            rejected_by=rejected_by, rulings=rulings)

    return repair


def _refuse(plan: ExaminerPlan):
    async def refuse(args: RefuseArgs) -> RefuseResult:
        _task(plan, args.task_id)
        replays, rerolls = plan.store.get("replays") or {}, plan.store.get("rerolls") or {}
        finished = finished_runs(args.task_id, replays, rerolls)
        trial = {**(plan.store.get("refusals") or {}),
                 args.task_id: {"task_id": args.task_id, "reason": args.reason, "round": plan.round}}
        ruling = refuse_gate(trial, replays, rerolls)
        if args.task_id not in ruling.metrics.get("refused", []):
            raise PermissionError(f"refusal of task {args.task_id} rejected: the frontier finished it: "
                                  f"{', '.join(finished)}")
        refusal = Refusal(task_id=args.task_id, reason=args.reason, round=plan.round, admitted=True,
                          finished_runs=finished)
        plan.store.setdefault("refusals", {})[args.task_id] = as_dict(refusal)
        status = plan.store.setdefault("task_status", {})
        entry = dict(status.get(args.task_id) or {})
        entry["refused"] = {"reason": args.reason, "round": plan.round}
        status[args.task_id] = entry
        write_json(plan.workdir / "task_status.json", status)
        plan.write_state()
        recorded = plan.ledger.record("refuse", refuse_gate(plan.store["refusals"], replays, rerolls))
        rulings = [ruling_of(recorded)]
        plan.last_rulings = rulings
        return RefuseResult(summary=f"task {args.task_id} refused: {args.reason}", task_id=args.task_id,
                            admitted=True, finished_runs=finished, rulings=rulings)

    return refuse


def _reroll(plan: ExaminerPlan):
    async def reroll(args: RerollArgs) -> RerollResult:
        _task(plan, args.task_id)
        if plan.run_rerolls is None:
            raise RuntimeError("no Runner for re-rolls in this session: the Builder handed over no reroll callable")
        if plan.allowance_remaining is not None and plan.allowance_remaining <= 0:
            raise RuntimeError(f"the round's allowance is spent ({plan.allowance_remaining:.2f} USD left)")
        prefix = f"reroll-r{plan.round}"
        existing = {row.get("run_id") for rows in (plan.store.get("rerolls") or {}).values() for row in rows}
        number = 0
        # `_candidate_runs` numbers Runs from zero under the prefix, so a second call in the same round
        # takes a longer prefix rather than writing over the first call's files.
        while any(str(run_id).startswith(f"{prefix}-{args.task_id}-") for run_id in existing):
            number += 1
            prefix = f"reroll-r{plan.round}-{number}"
        before = plan.spend()
        try:
            rows = await asyncio.to_thread(plan.run_rerolls, args.task_id, args.count, prefix)
        except budget.BudgetExceeded:
            plan.ceiling_reached = True
            raise
        rows = [dict(row) for row in rows or []]
        spent = max(0.0, plan.spend() - before)
        if plan.allowance_remaining is not None:
            plan.allowance_remaining -= spent
        plan.extra_rerolls.setdefault(args.task_id, []).extend(rows)
        plan.write_state()
        plan.load_state()
        finished = sum(1 for row in rows if (row.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS)
        summary = (f"reroll of task {args.task_id}: {len(rows)} Runs, {finished} finished, {spent:.4f} USD; "
                   f"the Task now has {len(plan.store['rerolls'].get(args.task_id, []))} re-rolls")
        return RerollResult(summary=summary, task_id=args.task_id, runs=[row.get("run_id") for row in rows],
                            finished=finished, spent_usd=spent)

    return reroll


def _finding(plan: ExaminerPlan):
    async def finding(args: FindingArgs) -> FindingResult:
        rows = plan.store.setdefault("findings", [])
        finding_id = f"finding-{len(rows) + 1}"
        about = plan.entry_id_for(args.about_call_id) if args.about_call_id else None
        record = Finding(finding_id=finding_id, task_id=args.task_id, kind=args.kind, text=args.text,
                         run_id=args.run_id, tool=args.tool, suggested=args.suggested, about_entry_id=about,
                         round=plan.round, status="open")
        rows.append(as_dict(record))
        plan.write_state()
        summary = f"finding {finding_id} ({args.kind}) filed for the Builder" + \
                  (f" on task {args.task_id}" if args.task_id else "") + \
                  (f", suggested {args.suggested}" if args.suggested != "none" else "")
        return FindingResult(summary=summary, finding_id=finding_id, finding=as_dict(record))

    return finding


def examiner_tools(plan: ExaminerPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The seven tools over one plan; `sink` is where the derive stage's events go (the harness's `emit`)."""
    return [
        AgentTool("read", "Read a Task, a Trace, an Intent, a Run, a Verifier, a probe pool, the task status, "
                  "the rulings, the re-roll or replay rows, or the References, as JSON.",
                  ReadArgs, ReadResult, _read(plan), render=render),
        AgentTool("derive", "Derive one Verifier per Task from its References through the D79 suite "
                  "(`all`, or one Task id).", DeriveArgs, DeriveResult, _derive(plan, sink), render=render),
        AgentTool("probe", "Score a hand-written Run against a Task's current Verifier and keep it in the "
                  "Task's pool forever (D127).", ProbeArgs, ProbeResult, _probe(plan), render=render),
        AgentTool("repair", "Propose a new Verifier version by dropping and adding atoms; the D79 suite, the "
                  "pool and the loosening gate decide whether it is accepted.",
                  RepairArgs, RepairResult, _repair(plan), render=render),
        AgentTool("refuse", "Give a Task up; admitted only when no frontier Run of it finished (D128).",
                  RefuseArgs, RefuseResult, _refuse(plan), render=render),
        AgentTool("reroll", "Buy more frontier Runs of a Task through the Runner (D112, D133).",
                  RerollArgs, RerollResult, _reroll(plan), render=render),
        AgentTool("finding", "File what is wrong on the Builder's side: an assisted tool, a fidelity gap, a "
                  "Reference disagreement, the Environment.", FindingArgs, FindingResult, _finding(plan),
                  render=render),
    ]
