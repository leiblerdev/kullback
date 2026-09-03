"""The Examiner's plan: what one session over one workdir is given, and the store its gates rule over.

The Builder's plan holds the pipeline's artifacts; this one holds the derivation inputs (never a tool
body, a table or the Environment, D123) plus what the Examiner itself wrote to disk: the Verifiers,
the task status, the probe pools, the Verifier histories, the refusals, the findings and its own
re-roll rows. `store` is what `gates_over` binds a registered gate's arguments to, so every name a
gate spec lists (`verifiers`, `probes`, `history`, `task_runs`, `refusals`, `replays`, `rerolls`,
`canon_rules`, `sigs`, `task_status`) is a key here. `load_state` reads all of it back off disk, so a
second session, or the next round, finds what the last one left (D127: a probe stays in the pool).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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
    the Examiner never touches what they read. `probe_model` is the model the loophole probe runs
    with; `judge_model` the residue judge of D111; the re-roll model is the callable's own, never
    named here. `allowance_remaining` is the round driver's number, in
    dollars: the reroll tool refuses at or below zero. `round` names the round the records it
    writes belong to. `unprotect` and `entry_id_for` are set by the extension when a harness with
    a session loads it; until then they are no-ops.
    """
    workdir: Path
    inputs: dict
    env_id: Optional[str] = None
    probe_model: Any = None
    judge_model: Any = None
    run_probe: Any = None
    run_rerolls: Any = None
    probe_limit: Optional[int] = None
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
        verifiers = [Verifier.model_validate(read_json(path))
                     for path in sorted((workdir / "verifiers").glob("*.json"))]
        probes = {path.parent.name: ProbePool.model_validate(read_json(path))
                  for path in sorted((workdir / "probes").glob("*/pool.json"))}
        history = {path.stem: VerifierHistory.model_validate(read_json(path))
                   for path in sorted((self.state_dir / "history").glob("*.json"))}
        refusals = read_json(self.state_dir / "refusals.json", {}) or {}
        findings = read_json(self.state_dir / "findings.json", []) or []
        self.extra_rerolls = read_json(self.state_dir / "rerolls.json", {}) or {}
        replays = self.inputs.get("replays") or {}
        rerolls = merged_rerolls(self.inputs.get("rerolls") or {}, self.extra_rerolls)
        self.store = dict(self.inputs)
        self.store.update({
            "verifiers": verifiers,
            "task_status": read_json(workdir / "task_status.json", {}) or {},
            "probes": probes,
            "history": history,
            "refusals": refusals,
            "findings": findings,
            "replays": replays,
            "rerolls": rerolls,
            "canon_rules": rules_of(self.inputs),
            "sigs": list(self.inputs.get("sigs") or []),
            "task_runs": task_runs_of(replays, rerolls),
        })

    def write_state(self) -> None:
        """Everything the Examiner owns on disk: pools, histories, refusals, findings, its re-roll rows."""
        for task_id, pool in sorted(self.store.get("probes", {}).items()):
            write_json(self.workdir / "probes" / task_id / "pool.json", as_dict(pool))
        for task_id, hist in sorted(self.store.get("history", {}).items()):
            write_json(self.state_dir / "history" / f"{task_id}.json", as_dict(hist))
        write_json(self.state_dir / "refusals.json", self.store.get("refusals", {}))
        write_json(self.state_dir / "findings.json", self.store.get("findings", []))
        write_json(self.state_dir / "rerolls.json", self.extra_rerolls)

    def current(self, task_id: str) -> Optional[Verifier]:
        """The Task's current Verifier, the file under verifiers/ as the store holds it."""
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
        totals = read_json(self.workdir / "budget.json", {}) or {}
        try:
            return float((totals.get("total") or {}).get("usd") or 0.0)
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
