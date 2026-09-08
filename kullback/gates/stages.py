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


def readers_gate(proposals: Iterable[Any], requestors: int = 0, assumptions: Iterable[Any] = (),
                 unset: Any = None, kinds: Any = None, derived: Iterable[Any] = (),
                 totals: Any = None) -> GateResult:
    """A proposal the readers gate could not satisfy is flagged and kept, never a failed build.

    Section 6 again: a requestor whose readers stayed assisted still leaves a world, and what that
    world is worth is replay fidelity's to say, not this gate's. Each proposal is the plain dict the
    stage wrote to readers.json, so nothing in the gates package has to know the Builder's records.

    Three things are reported and none of them fails a build: the shapes a reader read nothing out
    of, per tool, which is the strictness one arm of the 2026-09-07 experiment refused on and this
    one only counts; the columns filled from the corpus because no recording read them before a
    write, one assumption each; and the columns left unset because no recording read them before any
    write, which are named so a reader of the build can see what the world is guessing at. The kind
    these credits give each prose-result tool is reported beside them, because it overrides the
    miner's own.

    A fourth thing is reported the same way and fails nothing either (D203): the readers this build
    derived from a tool's own recorded results for a homed prose result no proposal covered, the
    ones a forced call had to settle, the slots the corpus could not bind to a column, and the
    results still unread per tool. A corpus reading zero on all four is one whose prose results were
    already read, which is what says the mechanism is off rather than that it did nothing.
    """
    proposals = list(proposals)
    derived = list(derived)
    totals = dict(totals or {})
    assisted = [p for p in proposals if _get(p, "assisted")]
    unset = dict(unset or {})
    kinds = dict(kinds or {})
    failures = [f"{_get(p, 'requestor')}: kept after {_get(p, 'attempts')} attempts with "
                f"{len(_get(p, 'failures') or [])} shape(s) still failing: "
                f"{(_get(p, 'failures') or ['no reason recorded'])[0]}"
                for p in assisted]
    silent = {}
    effects = 0
    for p in proposals:
        for tool, count in sorted((_get(p, "silent") or {}).items()):
            silent[str(tool)] = int(count)
        effects += sum(len(columns or []) for columns in (_get(p, "effects") or {}).values())
    return gate("readers", failures, requestors=requestors, proposals=len(proposals),
                assisted=len(assisted),
                columns=sum(len(_get(p, "columns") or []) for p in proposals),
                readers=sum(len(_get(p, "readers") or []) for p in proposals),
                silent_shapes=sum(silent.values()), silent_by_tool=silent,
                changed_columns=effects, filled_columns=len(list(assumptions)),
                unset_columns={str(k): sorted(v or []) for k, v in unset.items()},
                write_tools=sorted(name for per in kinds.values()
                                   for name, kind in (per or {}).items() if kind == "write"),
                read_tools=sorted(name for per in kinds.values()
                                  for name, kind in (per or {}).items() if kind != "write"),
                readers_derived=int(totals.get("readers_derived") or 0),
                readers_forced=int(totals.get("readers_forced") or 0),
                columns_revealed=int(totals.get("columns_revealed") or 0),
                forced_calls=int(totals.get("forced_calls") or 0),
                slots_unbound=int(totals.get("slots_unbound") or 0),
                results_unread=int(totals.get("results_unread") or 0),
                results_unread_by_tool={str(k): int(v) for k, v in
                                        sorted((totals.get("results_unread_by_tool") or {}).items())},
                derived_tools=sorted(str(_get(d, "tool")) for d in derived))


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

