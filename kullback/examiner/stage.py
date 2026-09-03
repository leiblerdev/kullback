"""The derive_verifier stage body, lifted out of build.py with its helpers, run against a small context in
place of the pipeline's StageContext, so the same bytes land in verifiers/, task_status.json,
references.json and constraints_check.json (D130).

The stage used three things of its context: `workdir`, `seed_runs` (the anchor, D81) and
`record_gate` (the ledger); `ExamContext` is those three. What the stage reads is the Builder's
store filtered by `DERIVE_INPUTS`: the Tasks, the mined signatures, the Constraints, the canon
rules, the replays, the re-rolls, the Intents, the Simulated user's rules, the Traces and the
assisted tools. The tool bodies, the Starting state, the schema and the Environment are not on the
list and `inputs_from` refuses a store that names them: the Examiner never reads what the Builder
compiled (D123). The one Run the old stage executed, check 6's loophole probe, reads bodies and so
stays in the Builder as `build.probe_runner(plan)`; it arrives here as the `run_probe` callable, the
Runner as a tool of both agents (D120).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.examiner import derive as verifier_mod
from kullback.examiner import reference as reference_mod
from kullback.gates import artifacts, fidelity, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates import stages as stage_gates
from kullback.gates.ledger import GateLedger
from kullback.runner.canon import rules_of
from kullback.runner.records import (
    GateResult,
    Intent,
    Task,
    Verifier,
    apply_intent,
    as_dict,
    read_json,
    write_json,
)

DERIVE_INPUTS = ("tasks", "sigs", "constraints", "canon_rules", "replays", "rerolls", "intents", "user_rules",
                 "traces", "assisted_tools")
FORBIDDEN_INPUTS = ("bodies", "db", "schema", "environment", "overlays", "synthetic_rows", "policy_text",
                    "lessons_applied", "lessons_set_aside")
STAGE = "derive_verifier"


class ExamContext:
    """What the derivation is given beside its inputs: where to write, the anchor to seed from, the ledger."""

    def __init__(self, workdir: Path, ledger: GateLedger, anchor: Any = None, stage: str = STAGE):
        self.workdir = Path(workdir)
        self.ledger = ledger
        self.anchor = anchor
        self.stage = stage
        self.recorded: list[GateResult] = []

    def seed_runs(self, task_id: str, run_ids: Iterable[str]) -> list[str]:
        """The Task's Runs minus its anchor when one was chosen, every Run otherwise (D81)."""
        if self.anchor is None:
            return list(run_ids)
        return list(self.anchor.seed_runs(task_id, run_ids))

    def record_gate(self, result: GateResult) -> GateResult:
        """One ruling into gates.json under this stage's name, remembered for the tool result."""
        self.recorded.append(result)
        return self.ledger.record(self.stage, result)


def inputs_from(store: dict) -> dict:
    """The derivation's inputs out of a store; a store naming a forbidden artifact is refused."""
    forbidden = [name for name in FORBIDDEN_INPUTS if name in store]
    if forbidden:
        raise ValueError(f"the Examiner never reads {', '.join(forbidden)}; hand it the derivation inputs only (D123)")
    return {name: store[name] for name in DERIVE_INPUTS if name in store}


def seed_ids(ctx: Any, task: Task) -> set[str]:
    """The Task's Runs the derivation may use: minus the anchor when one was chosen (D81)."""
    return set(ctx.seed_runs(task.id, task.run_ids))


def request_text(task: Task, intents: dict, traces: dict) -> str:
    """What the user asked, for the judge: the grounded Intent, else the Task's name, else the first user turn."""
    record = intents.get(task.id)
    if record is not None and record.grounded and record.text:
        return record.text
    if task.intent or task.name:
        return task.intent or task.name or ""
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        for turn in (trace.turns if trace else []):
            if turn.role == "user" and turn.content:
                return turn.content
    return ""


# --- the moved helpers ---------------------------------------------------------

def final_constraints(ctx, inputs: dict, seed_replays: dict, write_tools: set, read_tools: set,
                      fn: Any) -> tuple[list, list]:
    """The constraints a Verifier may check, and the ones the recordings demoted (D76).

    Every compiled constraint is run over the confirmed recordings corpus-wide first: the recordings
    are the frontier under the customer's real policy, so a rule they mostly break is a miscompiled
    rule, and it becomes a residual, reported in the setup review and checked in no Verdict. The
    compile_policy gate is recorded again over the final list, after the recordings have had their
    say, which is the order the reference check has to run in: on the second retail build 15 of 39
    compiled rules fired on confirmed recordings and poisoned every Verifier.
    """
    compiled = [c for c in inputs["constraints"] if c.compiled or c.judge_atom]
    rates = reference_mod.constraint_rates(compiled, [r["path"] for rows in seed_replays.values() for r in rows],
                                           write_tools, fn, read_tools)
    constraints, demoted = reference_mod.demote(compiled, rates)
    by_id = {c.id: c for c in compiled}
    residual = [by_id[row["id"]].model_copy(update={"compiled": False, "judge_atom": False,
                                                    "residual_reason": row["reason"]})
                for row in demoted]
    untouched = [c for c in inputs["constraints"] if not (c.compiled or c.judge_atom)]
    final = constraints + residual + untouched
    ctx.record_gate(artifacts.policy_gate(final))
    write_json(ctx.workdir / "constraints_check.json",
               {"rates": rates, "demoted": demoted, "constraints": [as_dict(c) for c in final]})
    return constraints, demoted


def no_reference_status(ctx, task: Task, confirmation: Any, *, seed_replays: list, replays: dict,
                        rerolls: dict, traces: dict, assisted_tools: set) -> dict:
    """Why this Task has no Reference, in the words the setup review needs (D49): a Task with none is not verdicted."""
    reason = confirmation.reason or "no Run to confirm"
    if not seed_replays:
        reason = ("no seed Trace was replayed" if not (replays.get(task.id) or {}) else
                  fidelity.unconfirmed_reason({t: r for t, r in replays[task.id].items()
                                               if t in seed_ids(ctx, task)}))
    # D49: the status names the blocking tool. A seed Trace that calls an assisted tool replays
    # through a body that failed the fidelity gates, so its divergence is the tool's, and the setup
    # review needs the tool's name, not the diff.
    assisted_used = sorted({c.name for tid in seed_ids(ctx, task) if tid in traces
                            for c in traces[tid].tool_calls if c.name in assisted_tools})
    if assisted_used:
        reason = (f"the seed Trace calls {', '.join(assisted_used)}, an assisted tool whose body "
                  f"failed the fidelity gates (D49); {reason}")
    return {"reference_confirmed": False, "verifier_passed": False, "reason": reason,
            "recordings": len(seed_replays), "rerolls": len(rerolls.get(task.id, [])),
            "judged": confirmation.judged, "assisted_tools": assisted_used}


def derive_for(task_for: Task, confirmation: Any, *, canon_rules: Any, write_tools: set, constraints: list,
               verifier_version: str = "1") -> Verifier:
    """The derivation over the References: the first is the Reference, the rest its re-runs."""
    paths = [r.path for r in confirmation.references]
    return verifier_mod.derive_verifier(task_for, paths[0], paths[1:], canon_rules,
                                        write_tools=write_tools, constraints=constraints,
                                        successful_run_ids=[r.run_id for r in confirmation.references],
                                        verifier_version=verifier_version)


def suite_for(task_for: Task, verifier: Verifier, paths: list, *, canon_rules: Any, write_tools: set,
              user_rules: dict, rules_trace: Optional[str], probe_model: Any, run_probe: Any,
              may_probe: bool) -> list[GateResult]:
    """The whole D79 suite over one Verifier: the wrong Run, the second path, the leak, the probe (D79, D119)."""
    return verifier_suite.validate_verifier(
        verifier, paths[0], canon=canon_rules, write_tools=write_tools, seed_runs=paths[1:],
        wrong_run=verifier_suite.wrong_run(verifier, paths[0], canon_rules),
        alt_path_run=paths[1] if len(paths) > 1 else None,
        intent_text=task_for.intent, user_rules=user_rules.get(rules_trace),
        model=probe_model if may_probe else None, run_probe=run_probe)


def verifier_for(ctx, task: Task, confirmation: Any, *, canon_rules: Any, write_tools: set, constraints: list,
                 intents: dict, user_rules: dict, recordings: int, rerolls: int, probe: Any,
                 probe_model: Any, may_probe: bool, verifier_version: str = "1") -> tuple[Verifier, dict]:
    """One Task's Verifier from its References, through the whole D79 suite, with its status row."""
    paths = [r.path for r in confirmation.references]
    first = confirmation.references[0]
    task_for = apply_intent(task, intents[task.id]) if task.id in intents else task
    record = derive_for(task_for, confirmation, canon_rules=canon_rules, write_tools=write_tools,
                        constraints=constraints, verifier_version=verifier_version)
    rules_trace = first.trace_id or next((r.trace_id for r in confirmation.references if r.trace_id), None)
    gates = suite_for(task_for, record, paths, canon_rules=canon_rules, write_tools=write_tools,
                      user_rules=user_rules, rules_trace=rules_trace, probe_model=probe_model,
                      run_probe=probe, may_probe=may_probe)
    results = verifier_suite.d79_results(gates)
    passed = artifacts.verifier_gate(results).passed
    write_json(ctx.workdir / "verifiers" / f"{task.id}.json", as_dict(record))
    status = {"reference_confirmed": True, "verifier_passed": bool(passed),
              "references": len(confirmation.references), "reference_kind": first.kind,
              "recordings": recordings, "rerolls": rerolls,
              "failed_recordings": dict(confirmation.failed), "judged": confirmation.judged,
              "checks": results,
              "not_run": [g.stage for g in gates if g.metrics.get("skipped")]}
    return record, status


# --- the stage body --------------------------------------------------------------

def derive_all(ctx: ExamContext, inputs: dict, *, probe_model: Any = None, probe_limit: Optional[int] = None,
               judge_model: Any = None, run_probe: Any = None, only: Optional[str] = None) -> dict:
    """One Verifier per Task from its References by the D111 rule, through the whole D79 suite.

    The References are the confirmed seed replays plus the finished re-rolls that agree on one End
    state after the recordings that broke a Hard constraint are out; the judge is the residue when
    two End states remain and fails at most one side (D110, D111). The Reference proper is the first
    recording of that group, the rest are the re-runs whose agreement sets required against allowed
    (D43) and the second path of check 5, and the anchor is never among them (D81). Before any of
    that, every compiled constraint is checked against the confirmed recordings corpus-wide and the
    ones they mostly break are demoted (D76). Check 4's wrong Run is built from the Reference by
    code; check 6's loophole probe is the one Run per Task `run_probe` executes, and `probe_limit`
    caps how many Tasks get one. A Task with no Reference is not verdicted.

    With `only` one Task is derived and its rows are merged into the task_status.json and
    references.json already on disk; without it the whole build is derived and the files are the
    stage's, byte for byte. Either way scorecard.json is rewritten last, so it reads as it did when
    the stage ran inside the pipeline.
    """
    canon_rules = rules_of(inputs)
    fn = verifier_suite.canon_fn(canon_rules)
    write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}
    read_tools = {s.name for s in inputs["sigs"] if s.kind != "write"}
    replays = inputs.get("replays") or {}
    rerolls = inputs.get("rerolls") or {}
    intents = {t: Intent.model_validate(d) for t, d in (inputs.get("intents") or {}).items()}
    user_rules = inputs.get("user_rules") or {}
    traces = {t.trace_id: t for t in inputs.get("traces") or []}
    policy_lines = [c.text for c in inputs["constraints"]]
    seed_replays = {task.id: [r for tid, r in sorted((replays.get(task.id) or {}).items())
                              if tid in seed_ids(ctx, task) and r.get("confirmed") and r.get("path")]
                    for task in inputs["tasks"]}
    # D76, D111: a compiled rule the confirmed recordings mostly break is demoted before any
    # Verifier is derived from them.
    constraints, demoted = final_constraints(ctx, inputs, seed_replays, write_tools, read_tools, fn)
    assisted_tools = set(inputs.get("assisted_tools") or ())
    atoms = reference_mod.hard_atoms(constraints, write_tools, read_tools)
    probe = run_probe if probe_model is not None else None
    probed = 0
    tasks = list(inputs["tasks"])
    if only is not None:
        tasks = [task for task in tasks if task.id == only]
        if not tasks:
            raise ValueError(f"no Task is named {only}")
    verifiers, status, references = [], {}, {}
    for task in tasks:
        recordings = [reference_mod.load(r["path"], reference_mod.RECORDING, run_id=r["run_id"],
                                         trace_id=r["trace_id"], write_tools=write_tools, fn=fn, atoms=atoms)
                      for r in seed_replays[task.id]]
        recordings += [reference_mod.load(r["path"], reference_mod.REROLL, run_id=r["run_id"],
                                          write_tools=write_tools, fn=fn, atoms=atoms)
                       for r in rerolls.get(task.id, [])
                       if (r.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS]
        confirmation = reference_mod.confirm(recordings, request=request_text(task, intents, traces),
                                             policy_lines=policy_lines, judge=judge_model)
        references[task.id] = confirmation.as_dict()
        if not confirmation.references:
            status[task.id] = no_reference_status(ctx, task, confirmation,
                                                  seed_replays=seed_replays[task.id], replays=replays,
                                                  rerolls=rerolls, traces=traces,
                                                  assisted_tools=assisted_tools)
            continue
        may_probe = probe is not None and (probe_limit is None or probed < probe_limit)
        probed += int(may_probe)
        record, status[task.id] = verifier_for(
            ctx, task, confirmation, canon_rules=canon_rules, write_tools=write_tools,
            constraints=constraints, intents=intents, user_rules=user_rules,
            recordings=len(seed_replays[task.id]), rerolls=len(rerolls.get(task.id, [])),
            probe=probe, probe_model=probe_model, may_probe=may_probe)
        verifiers.append(record)
    if only is not None:
        status = {**(read_json(ctx.workdir / "task_status.json", {}) or {}), **status}
        references = {**(read_json(ctx.workdir / "references.json", {}) or {}), **references}
    write_json(ctx.workdir / "task_status.json", status)
    write_json(ctx.workdir / "references.json", references)
    # Section 6: a Task whose Verifier does not clear D79 is "not verdicted, Verifier
    # immature", which is a Task the report leaves uncounted, not a failed build.
    ctx.record_gate(stage_gates.task_verifiers_gate(
        status,
        verifiers=len(verifiers), references=sum(1 for r in status.values() if r["reference_confirmed"]),
        passed=sum(1 for r in status.values() if r["verifier_passed"]), tasks=len(status),
        probed=probed, constraints_demoted=len(demoted),
        failed_recordings=sum(len(r.get("failed") or {}) for r in references.values()),
        judged=sum(1 for r in references.values() if r.get("judged")),
        disagreeing=sum(1 for r in references.values()
                        if not r["references"] and (r.get("reason") or "").startswith("recordings disagree"))))
    write_json(ctx.workdir / "scorecard.json", scorecard_mod.scorecard(ctx.workdir))
    return {"verifiers": verifiers, "task_status": status}
