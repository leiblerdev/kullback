"""Derive one Task's Verifier from its Reference and k re-runs read off disk (D42, D43, D91).

Nothing here executes a Run (D91): re-runs arrive as `Run` JSONL paths the Runner already wrote.
The vocabulary the atoms are written in (reading Runs, write effects, provenance, questions and
communicate facts, the predicate templates) and the nine D79 checks that rule on the result live
in `kullback.gates.verifier_suite`; this module is the derivation, moved here from
`builder/verifier.py` in phase 5 because the Examiner is the one writer of Verifiers (D123). What
it adds to the gates' `make_atom` is the transcript helpers a compiled Hard rule may call, which
the suite holds as `HELPERS_SRC` so the policy compiler and this module read one text.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.gates.verifier_suite import (
    HELPERS_SRC,
    SUCCESS_TERMINATIONS,
    as_run,
    atom_payload,
    canon_fn,
    classify_provenance,
    communicate_values,
    make_atom,
    ptr,
    question_keys,
    resolve_write_tools,
    run_calls,
    text_of,
    write_effects,
)
from kullback.runner.records import Atom, Constraint, Run, Task, Verifier

# A write value is required only when it is agreed across the good Runs *and* the customer or their
# world is where it came from; a value the Candidate invented is never required of it.
REQUIRED_PROVENANCE = ("user_stated", "system_derived")


def load_runs(paths: Iterable[Any]) -> list[Run]:
    return [as_run(p) for p in paths]


def successful(run: Run, successful_run_ids: Optional[Iterable[str]]) -> bool:
    """Did this re-run finish well enough to derive from? The caller's list wins, else the reason.

    Derivation is the only reader, so this lives here rather than in the suite; `SUCCESS_TERMINATIONS`
    stays there because build.py and gates/stages.py rule on it too.
    """
    if successful_run_ids is not None:
        return run.run_id in set(successful_run_ids)
    return (run.termination_reason or "") in SUCCESS_TERMINATIONS


def _helpers_src() -> str:
    """The transcript helpers a compiled rule calls, so it finds them at Verdict time."""
    return HELPERS_SRC


def _atom(atom_id: str, kind: str, payload: dict, **fields: Any) -> Atom:
    """One atom, its Hard predicate wrapped with the transcript helpers (the gates' `make_atom` otherwise)."""
    return make_atom(atom_id, kind, payload, helpers=_helpers_src(), **fields)


# --- derivation ------------------------------------------------------------

def derive_verifier(task: Any, reference_run: Any, rerun_paths: Optional[list[str]] = None, canon: Any = None, *,
                    write_tools: Optional[Iterable[str]] = None,
                    constraints: Optional[Iterable[Constraint]] = None,
                    successful_run_ids: Optional[Iterable[str]] = None,
                    verifier_version: str = "1") -> Verifier:
    """The atoms for one Task: write-set diff over the Reference and its successful re-runs (D42, D43)."""
    fn = canon_fn(canon)
    reference = as_run(reference_run)
    reruns = load_runs(rerun_paths or [])
    tools = resolve_write_tools([reference] + reruns, write_tools)
    good = [reference] + [r for r in reruns if successful(r, successful_run_ids)]
    good_effects = [write_effects(r, tools, fn) for r in good]
    everywhere = set.intersection(*[set(e) for e in good_effects]) if good_effects else set()
    somewhere = set().union(*[set(e) for e in good_effects]) if good_effects else set()
    reads = {c["name"] for r in [reference] + reruns for c in run_calls(r)} - tools
    atoms: list[Atom] = []

    # D43: present in every successful re-run is required, in some is allowed, in none is not an
    # atom. A write only a failed re-run made is therefore not a forbidden atom; the write cap below
    # and verdict.py's extra-write check are what keep a Candidate from writing more than the good
    # Runs did.
    for number, key in enumerate(sorted(somewhere)):
        run, effect = _first_with(good, good_effects, key)
        atom_id = f"w{number}"
        kind = "required" if key in everywhere else "allowed"
        base = {"tool": effect["tool"], "entity": effect["entity"], "entity_raw": effect["entity_raw"],
                "id_field": effect["id_field"], "at": effect["idx"]}
        if effect["requestor"]:
            base["requestor"] = effect["requestor"]  # D71: a user-side write says so on the atom
        atoms.append(_atom(atom_id, kind, dict(base, kind="write"),
                           spans=[ptr(run, effect["idx"])],
                           description=f"{effect['tool']} writes {effect['entity'] or 'an entity'}"))
        for field in sorted(effect["values"]):
            value, canon_key = effect["args"][field], effect["values"][field]
            provenance, span = classify_provenance(run, effect["pos"], value, fn)
            agreed = all(key in e and e[key]["values"].get(field) == canon_key for e in good_effects)
            atoms.append(_atom(f"{atom_id}.{field}",
                               "required" if agreed and provenance in REQUIRED_PROVENANCE else "allowed",
                               dict(base, kind="write_value", field=field, value=canon_key, raw=value),
                               provenance=provenance, spans=[span] if span else [],
                               description=f"{effect['tool']} {field} is {text_of(value)}"))

    if good_effects:
        cap = max(len(e) for e in good_effects)
        atoms.append(_atom("entity_count", "required", {"kind": "entity_count", "count": cap},
                           description=f"the Run makes at most {cap} write calls"))

    asked = [question_keys(r, e, fn) for r, e in zip(good, good_effects, strict=False)]
    for key in sorted(set.intersection(*[set(a) for a in asked]) if asked else set()):
        seen = asked[0][key]
        atoms.append(_atom(f"q.{key}", "question",
                           {"kind": "question", "key": key, "tool": seen.get("tool"),
                            "field": seen.get("field")},
                           spans=[seen["span"]] if seen.get("span") else [],
                           description=f"the agent asks the user about {key.split(':', 1)[-1]}"))

    said = [communicate_values(r, fn) for r in good]
    for number, key in enumerate(sorted(set.intersection(*[set(s) for s in said]) if said else set())):
        fact = said[0][key]
        atoms.append(_atom(f"c{number}", "communicate",
                           {"kind": "communicate", "value": key, "text": fact["text"]},
                           provenance="system_derived", spans=[fact["span"]],
                           description=f"the final answer states {fact['text']}"))

    for rule in constraints or []:
        if not (rule.compiled or rule.judge_atom):
            continue  # a residual constraint is reported, never verdicted (D76)
        atoms.append(_atom(f"hard.{rule.id}", "hard",
                           {"kind": "hard", "constraint_id": rule.id, "judge": rule.judge_atom,
                            "predicate_src": rule.predicate_src, "write_tools": sorted(tools),
                            "read_tools": sorted(reads)},
                           judge=bool(rule.judge_atom), description=rule.text,
                           spans=[rule.span] if rule.span else []))

    task_id = task if isinstance(task, str) else (task.id if isinstance(task, Task) else str(task))
    return Verifier(task_id=task_id, atoms=atoms, verifier_version=verifier_version,
                    seed_run_ids=[r.run_id for r in good])


def _first_with(runs: list[Run], effects: list[dict], key: str):
    """The first Run that has this write effect; the Reference comes first, so its spans win."""
    for run, effect in zip(runs, effects, strict=False):
        if key in effect:
            return run, effect[key]
    raise KeyError(key)



# --- export ----------------------------------------------------------------

def export_tau2_actions(verifier: Verifier, *, include_allowed: bool = True) -> list[dict]:
    """The Verifier's write atoms in tau2's `evaluation_criteria.actions` shape.

    A write the trace recorded as the user's own carries `requestor: user` (D71), which is the field
    tau2's Action uses to say who performs it; dropping it asked the Candidate agent to do the user's work.
    """
    wanted = ("required", "allowed") if include_allowed else ("required",)
    order, arguments = [], {}
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        if payload.get("kind") == "write" and atom.kind in wanted:
            order.append((atom.id, payload["tool"], payload.get("requestor") or "assistant"))
            arguments.setdefault(atom.id, {})
        elif payload.get("kind") == "write_value":
            arguments.setdefault(atom.id.rsplit(".", 1)[0], {})[payload["field"]] = payload.get("raw")
    return [{"action_id": f"{verifier.task_id}_{number}", "requestor": requestor, "name": tool,
             "arguments": arguments.get(atom_id, {}), "info": None}
            for number, (atom_id, tool, requestor) in enumerate(order)]
