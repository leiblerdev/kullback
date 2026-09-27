"""A Spec's checks to a Verifier with the existing atom builders, no new kinds (ported from intentv).

Each demand compiles exactly as the intentv experiment compiled it; the Spec adds only its because,
appended to every atom's description, and drops a check whose because is not grounded.
"""

from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Optional

from kullback.examiner import derive
from kullback.gates import verifier_suite
from kullback.runner.records import Atom, Verifier
from kullback.runner.target import _key as canon_key
from kullback.spec.ground import valid_because

ATOM_KINDS = ("required", "allowed", "forbidden", "question", "communicate", "hard")


class Compiled(NamedTuple):
    verifier: Verifier
    dropped: int


def _write(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if demand.get("tool") not in write_tools or not demand.get("id_field"):
        return None
    tool, entity = demand["tool"], str(demand.get("entity", ""))
    base = {"tool": tool, "entity": entity, "entity_raw": entity, "id_field": demand["id_field"], "at": 0}
    atoms = [verifier_suite.make_atom(atom_id, kind, dict(base, kind="write"),
                                      description=f"{tool} writes {entity or 'an entity'}")]
    values = demand.get("values") or {}
    if isinstance(values, dict):
        for field in sorted(values):
            atoms.append(verifier_suite.make_atom(
                f"{atom_id}.{field}", kind,
                dict(base, kind="write_value", field=field, value=canon_key(fn, values[field]), raw=values[field]),
                description=f"{tool} {field} is {values[field]}"))
    return atoms


def _say(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if not demand.get("text"):
        return None
    text = str(demand["text"])
    return [verifier_suite.make_atom(
        atom_id, "communicate",
        {"kind": "communicate", "value": canon_key(fn, text), "text": text, "field": None, "source_tool": None},
        description=f"the final answer states {text}")]


def _ask(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if demand.get("field"):
        key, tool, field = f"field:{demand['field']}", demand.get("tool"), demand["field"]
    elif demand.get("confirm_tool"):
        key, tool, field = f"confirm:{demand['confirm_tool']}", demand["confirm_tool"], None
    else:
        return None
    return [verifier_suite.make_atom(
        atom_id, "question", {"kind": "question", "key": key, "tool": tool, "field": field},
        description=f"the agent asks the user about {key.split(':', 1)[-1]}")]


def _no_write(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    return [derive.no_write_atom(write_tools).model_copy(update={"id": atom_id})]


def _cap(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if not isinstance(demand.get("count"), int):
        return None
    return [verifier_suite.make_atom(
        atom_id, "required", {"kind": "entity_count", "count": demand["count"]},
        description=f"the Run makes at most {demand['count']} write calls")]


def _shape(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if demand.get("tool") not in write_tools or not demand.get("field") or not demand.get("id_field"):
        return None
    return [derive.shape_atom(atom_id, demand["tool"], demand["field"], demand["id_field"], write_tools)]


BUILDERS = {"write": _write, "say": _say, "ask": _ask, "no_write": _no_write, "cap": _cap, "shape": _shape}


def _because(atom: Atom, because: str) -> Atom:
    return atom.model_copy(update={"description": f"{atom.description or ''} because: {because}"})


def compile_demands(demands: list, intent: str, sections: Iterable[str], write_tools: Iterable[str],
                    fn: Callable, task_id: str = "", version: str = "spec") -> Compiled:
    """Demand dicts (each with kind, demand and because) to a Verifier, and how many were dropped."""
    tools, sections = set(write_tools), list(sections)
    atoms: list[Atom] = []
    dropped = 0
    for number, demand in enumerate(demands):
        if (not isinstance(demand, dict) or demand.get("kind") not in ATOM_KINDS
                or not valid_because(demand.get("because"), intent, sections)):
            dropped += 1
            continue
        build = BUILDERS.get(demand.get("demand"))
        built = build(f"i{number}", demand["kind"], demand, tools, fn) if build else None
        if not built:
            dropped += 1
            continue
        atoms += [_because(atom, str(demand["because"])) for atom in built]
    return Compiled(Verifier(task_id=task_id, atoms=atoms, verifier_version=version, seed_run_ids=[]), dropped)


def compile_spec(spec, write_tools: Iterable[str], canon_fn: Callable,
                 policy_sections: Iterable[str] = ()) -> Compiled:
    """The Spec's checks as a Verifier; a check whose because is not grounded is dropped and counted."""
    demands = [dict(check.demand, kind=check.kind, because=check.because) for check in spec.checks]
    return compile_demands(demands, spec.intent.text, policy_sections, write_tools, canon_fn,
                           task_id=spec.task_id, version=f"spec{spec.version}")
