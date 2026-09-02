"""The rulings the Builder's stages used to make inline in build.py, each over the artifact the stage made.

Design section 6 gives every stage its own answer to a failed gate, and only "build Environment"
and "compile tools" stop a build; the rest are "flag, do not synthesize", "Task not verdicted",
"export conflict", and are recorded in gates.json without rolling the stage back. Until phase 4 the
decision behind each of these was a `gate_support.gate(...)` call in the middle of the stage, which
put the ruling outside the package that is hashed per release (D122) and outside the registry the
tool_result hook runs. They are here now, as functions over the artifact the stage returns, so the
hook can run them again over what a tool produced and get the same words. The stage still gathers
the evidence; only the ruling moved.

`rerolls_gate` keeps its history: green only when some re-roll finished, because the third retail
build's 597 re-rolls all stopped on a provider error and a gate that counted them passed over
nothing, the same green-over-nothing the scorecard gate once had.
"""

from __future__ import annotations

from typing import Any, Iterable

from kullback.gates.verifier_suite import SUCCESS_TERMINATIONS
from kullback.runner.gate_support import _get, gate
from kullback.runner.records import GateResult


def cluster_gate(tasks: Iterable[Any], categories: Iterable[Any] = ()) -> GateResult:
    """Every Task holds at least one Run; a Task with none is named, the build goes on (section 6)."""
    tasks = list(tasks)
    empty = [_get(t, "id") for t in tasks if not _get(t, "run_ids")]
    return gate("cluster", [f"task {i} holds no Run" for i in empty],
                tasks=len(tasks), categories=len(list(categories)))


def compile_tools_gate(bodies: dict, assisted_tools: Iterable[str] = ()) -> GateResult:
    """Every tool has a body; a tool with none fails the stage, which is the rollback edge (section 6)."""
    missing = sorted(name for name, body in bodies.items() if not (body or "").strip())
    return gate("compile_tools", [f"{name} has no body" for name in missing],
                tools=len(bodies), assisted=len(list(assisted_tools)))


def intent_gate(intents: dict) -> GateResult:
    """An ungrounded Intent is a Task with no Verdict, never a failed build (D47, section 6)."""
    failures = [f"task {t}: {_get(r, 'reason')}" for t, r in sorted(intents.items()) if not _get(r, "grounded")]
    return gate("intent", failures, tasks=len(intents), grounded=sum(1 for r in intents.values() if _get(r, "grounded")))


def rerolls_gate(rerolls: dict, per_task: int) -> GateResult:
    """Green only when some re-roll finished: a Run the frontier cannot complete says nothing about the Task (D112)."""
    total = sum(len(rows) for rows in rerolls.values())
    finished = sum(1 for rows in rerolls.values() for r in rows
                   if (r.get("termination_reason") or "") in SUCCESS_TERMINATIONS)
    failures = ([f"no re-roll finished: {total} Runs and every one stopped on an error; a Run the frontier "
                 f"cannot complete says nothing about the Task"] if total and not finished else [])
    return gate("rerolls", failures, tasks=len(rerolls), per_task=per_task, runs=total, finished=finished)


def tau2_export_gate(conflicts: Iterable[str]) -> GateResult:
    """Two Tasks pinning one row in two versions is a failure of the tau2 export, which has one db.json,
    and not of the Environment, whose Runner reads each Task's own overlay (D74)."""
    conflicts = list(conflicts)
    return gate("tau2_export", conflicts, overlay_conflicts=len(conflicts))


def vocabulary_gate(vocab: Any) -> GateResult:
    """The derived Vocabulary counted (D115): fields, the ones derived from this corpus, what the web added."""
    fields = list(_get(vocab, "fields", []) or [])
    searched = list(_get(vocab, "searched", []) or [])
    derived = [f for f in fields if "generic" not in (_get(f, "sources", []) or [])]
    return gate("vocabulary", [], fields=len(fields), derived=len(derived), searched=len(searched),
                web_aliases=sum(len(r.get("aliases") or []) for r in searched),
                notes=len(_get(vocab, "notes", []) or []))


def task_verifiers_gate(task_status: dict, **metrics: Any) -> GateResult:
    """A Task whose Verifier does not clear D79 is "not verdicted, Verifier immature": a Task the report
    leaves uncounted, not a failed build (section 6). `metrics` are the stage's counts, recorded as given."""
    broken = [t for t, row in task_status.items()
              if row["reference_confirmed"] and not row["verifier_passed"]]
    return gate("derive_verifier", [f"task {t}: the D79 suite did not pass" for t in broken], **metrics)

