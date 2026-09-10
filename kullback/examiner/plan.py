"""The Examiner's plan: what one session over one workdir is given, and the store its gates rule over.

The Builder's plan holds the pipeline's artifacts; this one holds the derivation inputs (never a tool
body, a table or the Environment, D123) plus what the Examiner itself wrote to disk: the Verifiers,
the task status, the probe pools, the Verifier histories, the refusals, the findings and its own
re-roll rows and the automatic loosening proposals. `store` is what `gates_over` binds a
registered gate's arguments to, so every name a
gate spec lists (`verifiers`, `probes`, `history`, `task_runs`, `refusals`, `replays`, `rerolls`,
`canon_rules`, `sigs`, `task_status`) is a key here. `load_state` reads all of it back off disk, so a
second session, or the next round, finds what the last one left (D127: a probe stays in the pool).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.examiner import lifecycle
from kullback.examiner.stage import inputs_from
from kullback.gates.ledger import GateLedger
from kullback.gates.verifier_suite import load_run
from kullback.runner.canon import rules_of
from kullback.runner.records import (
    Finding,
    ProbePool,
    Refusal,
    Verifier,
    VerifierHistory,
    as_dict,
    read_json,
    write_json,
)

STATE_DIR = "examiner"


def _no_unprotect(entry_ids: Iterable[str]) -> None:
    """The plan's unprotect before an extension set one: a driver without a session protects nothing."""
    return None


def _no_entry(tool_call_id: str) -> Optional[str]:
    return None


@dataclass
class ExaminerPlan:
    """Everything one Examiner session is given, and the state it reads back and writes.

    `inputs` is the derivation's store, filtered through `inputs_from` (a store naming a body, the
    db, the schema or the Environment is refused). `run_probe` and `run_rerolls` are the Runner as
    a callable the Builder built over its own store (`build.probe_runner`, `build.reroll_runner`):
    the Examiner never touches what they read. `run_variant` is the third of them
    (`build.variant_runner`, D199): a call path written by code, replayed from a Task's Starting
    state, which costs no model call. `probe_model` is the model the loophole probe runs
    with; `judge_model` the residue judge of D111 and `judge_agent` whether that judge is the agent
    with a bounded look rather than the one-shot judge (D185, off by default); the re-roll model is
    the callable's own, never named here. `workers` is how many Tasks the derivation derives at once (D163), the Builder's
    number for the same thing. `allowance_remaining` is the round driver's number, in
    dollars: the reroll tool refuses at or below zero. `round` names the round the records it
    writes belong to. `unprotect` and `entry_id_for` are set by the extension when a harness with
    a session loads it; until then they are no-ops.
    """
    workdir: Path
    inputs: dict
    env_id: Optional[str] = None
    probe_model: Any = None
    judge_model: Any = None
    judge_agent: bool = False
    run_probe: Any = None
    run_rerolls: Any = None
    run_variant: Any = None
    probe_limit: Optional[int] = None
    workers: int = 1
    anchor: Any = None
    on_event: Optional[Any] = None
    round: int = 0
    allowance_remaining: Optional[float] = None
    store: dict = field(init=False, default_factory=dict)
    ledger: GateLedger = field(init=False, repr=False)
    unprotect: Callable[[Iterable[str]], None] = field(init=False, repr=False, default=_no_unprotect)
    entry_id_for: Callable[[str], Optional[str]] = field(init=False, repr=False, default=_no_entry)
    ceiling_reached: bool = field(init=False, default=False)
    last_rulings: list = field(init=False, default_factory=list)
    extra_rerolls: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.inputs = inputs_from(self.inputs)
        self.ledger = GateLedger(self.workdir)
        self.load_state()

    @property
    def state_dir(self) -> Path:
        return self.workdir / STATE_DIR

    def refresh(self, store: dict) -> None:
        """A new Builder store (the next round's): the inputs are refiltered and the state read back."""
        self.inputs = inputs_from(store)
        self.load_state()

    def load_state(self) -> None:
        """The store: the derivation inputs plus everything the Examiner and the derivation wrote to disk."""
        workdir = self.workdir
        task_status = read_json(workdir / "task_status.json", {}) or {}
        # D208: the one accessor. Every gate and every rule reads `verifiers` off this store, so
        # filtering here is what stops a file whose Reference has been withdrawn from being scored:
        # no consumer globs the directory itself, and a workdir an earlier build left with orphans
        # is read the same way the next derivation will leave it.
        verifiers, retired = lifecycle.partition(
            (Verifier.model_validate(read_json(path))
             for path in sorted((workdir / "verifiers").glob("*.json"))), task_status)
        probes = {path.parent.name: ProbePool.model_validate(read_json(path))
                  for path in sorted((workdir / "probes").glob("*/pool.json"))}
        history = {path.stem: VerifierHistory.model_validate(read_json(path))
                   for path in sorted((self.state_dir / "history").glob("*.json"))}
        refusals = read_json(self.state_dir / "refusals.json", {}) or {}
        findings = read_json(self.state_dir / "findings.json", []) or []
        # D205: one row per automatic loosening proposal, which is what the per Task and per round
        # caps are counted off and what the round reads its four auto_loosen counts from.
        loosened = read_json(self.state_dir / "auto_loosen.json", []) or []
        self.extra_rerolls = read_json(self.state_dir / "rerolls.json", {}) or {}
        replays = self.inputs.get("replays") or {}
        rerolls = merged_rerolls(self.inputs.get("rerolls") or {}, self.extra_rerolls)
        self.store = dict(self.inputs)
        self.store.update({
            "verifiers": verifiers,
            "retired_verifiers": retired,
            "task_status": task_status,
            "probes": probes,
            "history": history,
            "refusals": refusals,
            "findings": findings,
            "replays": replays,
            "rerolls": rerolls,
            "canon_rules": rules_of(self.inputs),
            "sigs": list(self.inputs.get("sigs") or []),
            "task_runs": task_runs_of(replays, rerolls),
            "auto_loosen": loosened,
        })

    def write_state(self) -> None:
        """Everything the Examiner owns on disk: pools, histories, refusals, findings, its re-roll rows,
        the automatic loosening proposals."""
        for task_id, pool in sorted(self.store.get("probes", {}).items()):
            write_json(self.workdir / "probes" / task_id / "pool.json", as_dict(pool))
        for task_id, hist in sorted(self.store.get("history", {}).items()):
            write_json(self.state_dir / "history" / f"{task_id}.json", as_dict(hist))
        write_json(self.state_dir / "refusals.json", self.store.get("refusals", {}))
        write_json(self.state_dir / "findings.json", self.store.get("findings", []))
        write_json(self.state_dir / "auto_loosen.json", self.store.get("auto_loosen", []))
        write_json(self.state_dir / "rerolls.json", self.extra_rerolls)

    def current(self, task_id: str) -> Optional[Verifier]:
        """The Task's live Verifier, or None when its Reference was withdrawn and it was retired (D208)."""
        return next((v for v in self.store.get("verifiers", []) if v.task_id == task_id), None)

    def set_current(self, verifier: Verifier) -> None:
        """Replace the Task's Verifier in the store and on disk (an accepted repair)."""
        others = [v for v in self.store.get("verifiers", []) if v.task_id != verifier.task_id]
        self.store["verifiers"] = others + [verifier]
        write_json(self.workdir / "verifiers" / f"{verifier.task_id}.json", as_dict(verifier))

    def close_findings(self, finding_ids: Iterable[str]) -> list[str]:
        """Mark findings closed (the Builder acted on them) and release the entries they protected."""
        wanted = set(finding_ids)
        closed: list[str] = []
        released: list[str] = []
        for row in self.store.get("findings", []):
            if row.get("finding_id") in wanted and row.get("status") != "closed":
                row["status"] = "closed"
                closed.append(row["finding_id"])
                if row.get("about_entry_id"):
                    released.append(row["about_entry_id"])
        if released:
            self.unprotect(released)
        if closed:
            self.write_state()
        return closed

    def open_findings(self) -> list[Finding]:
        return [Finding.model_validate(row) for row in self.store.get("findings", []) if row.get("status") == "open"]

    def refusal(self, task_id: str) -> Optional[Refusal]:
        row = self.store.get("refusals", {}).get(task_id)
        return Refusal.model_validate(row) if row else None

    def spend(self) -> float:
        """What this workdir has spent so far, off budget.json; zero before any model call."""
        from kullback.runner import budget
        try:
            return float(budget.load_totals(self.workdir)["total"]["usd"])
        except (TypeError, ValueError):
            return 0.0


def merged_rerolls(builder_rows: dict, examiner_rows: dict) -> dict:
    """The Builder's re-roll rows plus the Examiner's, per Task, one row per run_id; the frontier pool grows (D133)."""
    out: dict[str, list] = {task_id: list(rows) for task_id, rows in (builder_rows or {}).items()}
    for task_id, rows in (examiner_rows or {}).items():
        seen = {row.get("run_id") for row in out.get(task_id, [])}
        for row in rows:
            if row.get("run_id") not in seen:
                out.setdefault(task_id, []).append(dict(row))
                seen.add(row.get("run_id"))
    return out


def task_runs_of(replays: dict, rerolls: dict) -> dict:
    """Per Task, every Run on disk the loosening gate may compare versions over: the replays and the re-rolls.

    A row whose file is missing or unreadable is skipped rather than failing the whole load: the
    gate then rules over what it can read, and a Run that is not there cannot be newly passed.
    """
    out: dict[str, list] = {}
    for task_id, rows in sorted((replays or {}).items()):
        for _, row in sorted((rows or {}).items()):
            _append_run(out, task_id, row)
    for task_id, rows in sorted((rerolls or {}).items()):
        for row in rows or []:
            _append_run(out, task_id, row)
    return out


def _append_run(out: dict, task_id: str, row: dict) -> None:
    path = row.get("path") if isinstance(row, dict) else None
    if not path or not Path(path).is_file():
        return
    try:
        run = load_run(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if any(r.run_id == run.run_id for r in out.get(task_id, [])):
        return
    out.setdefault(task_id, []).append(run)
