"""The world the Examiner's gate tests share: the cancel-order Verifier of verifier_fixtures in a strict,
a plain and a loosened version, probes written against one of them, history rows, and the replay and
re-roll rows the Runner would have written for the Task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gates.verifier_fixtures import WRITE_TOOLS, derive
from kullback.gates.probes import version_hash
from kullback.gates.verifier_suite import check_run
from kullback.runner.records import (
    Probe,
    ProbePool,
    Run,
    ToolSig,
    Verifier,
    VerifierHistory,
    VerifierVersion,
)

TASK = "t1"
SIGS = [ToolSig(name="cancel_pending_order", kind="write", kind_confidence="high", unclassified=False),
        ToolSig(name="get_order_details", kind="read", kind_confidence="high", unclassified=False)]


def base(tmp_path: Path) -> Verifier:
    """The derived Verifier: the write and its order id required, the reason allowed, one write at most."""
    return derive(tmp_path)


def tighten(verifier: Verifier) -> Verifier:
    """The same Verifier with the cancel reason required as well, so a Run giving another reason fails."""
    atoms = [a.model_copy(update={"kind": "required"}) if a.id == "w0.reason" else a for a in verifier.atoms]
    return verifier.model_copy(deep=True, update={"atoms": atoms})


def loosen(verifier: Verifier) -> Verifier:
    """The same Verifier without the write cap, so a Run that writes one entity more passes."""
    return verifier.model_copy(deep=True, update={"atoms": [a for a in verifier.atoms if a.id != "entity_count"]})


def scores(verifier: Verifier, run: Run) -> bool:
    return check_run(verifier, run, None, write_tools=WRITE_TOOLS)[0]


def probe(probe_id: str, run: Run, against: Verifier, bug_class: str = "other",
          scored_pass: Optional[bool] = None, task_id: str = TASK) -> Probe:
    """A probe written against a version: its hash and what that version said about the Run."""
    return Probe(probe_id=probe_id, task_id=task_id, bug_class=bug_class, verifier_hash=version_hash(against),
                 scored_pass=scores(against, run) if scored_pass is None else scored_pass,
                 run=run.model_copy(update={"run_id": probe_id, "task_id": task_id}))


def pool(*probes: Probe, task_id: str = TASK) -> ProbePool:
    return ProbePool(task_id=task_id, probes=list(probes))


def version(verifier: Verifier, number: int = 1, *, accepted: bool = True, by: str = "derive",
            rejected_by: Optional[list[str]] = None, parent: Optional[Verifier] = None) -> VerifierVersion:
    return VerifierVersion(task_id=verifier.task_id, content_hash=version_hash(verifier), verifier_version=str(number),
                           parent_hash=version_hash(parent) if parent is not None else None, by=by,
                           accepted=accepted, rejected_by=list(rejected_by or []), verifier=verifier)


def history(*versions: VerifierVersion, task_id: str = TASK) -> dict[str, VerifierHistory]:
    return {task_id: VerifierHistory(task_id=task_id, versions=list(versions))}


def replay_row(trace_id: str, confirmed: bool, run_id: Optional[str] = None, path: str = "") -> dict:
    return {"trace_id": trace_id, "run_id": run_id or f"replay-{trace_id}", "confirmed": confirmed,
            "path": path, "reasons": [] if confirmed else ["writes differ"]}


def reroll_row(run_id: str, termination_reason: str = "success", path: str = "") -> dict:
    return {"run_id": run_id, "path": path, "termination_reason": termination_reason}


def status(verifier_passed: bool = True, reference_confirmed: bool = True, **more: Any) -> dict:
    return dict({"reference_confirmed": reference_confirmed, "verifier_passed": verifier_passed,
                 "assisted_tools": [], "recordings": 1, "rerolls": 2, "judged": False}, **more)
