"""A Spec's checks to a Verifier with the existing atom builders, no new kinds (ported from intentv).

Each demand compiles exactly as the intentv experiment compiled it; the Spec adds only its because,
appended to every atom's description, and drops a check whose because is not grounded. A write or
no-write demand on an action tool (spec/actions.py, D308) compiles to an action atom instead.
"""

from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Optional

from kullback.gates import verifier_suite
from kullback.runner.records import Atom, Verifier
from kullback.runner.target import _key as canon_key
from kullback.spec.atoms import action_atoms, no_write_atom, shape_atom, tool_cap_atom
from kullback.spec.ground import valid_because

ATOM_KINDS = ("required", "allowed", "forbidden", "question", "communicate", "hard")
WRITE_WITHOUT_ENTITY = "write without entity"
# Why no Run can pass a Spec (D310): a no-write over a tool a required write needs, a demand both
# forbidden and required, a required write that names no one row.
NO_WRITE_BESIDE_REQUIRED = "no_write_beside_required_write"
FORBIDDEN_EQUALS_REQUIRED = "forbidden_equals_required"
WRITE_WITHOUT_ROW = "required_write_without_row"
UNSATISFIABLE = (NO_WRITE_BESIDE_REQUIRED, FORBIDDEN_EQUALS_REQUIRED, WRITE_WITHOUT_ROW)
# The keys of a demand dict that say why and how strongly, not what is demanded.
_NOT_DEMANDED = ("kind", "because", "check_id")


class Compiled(NamedTuple):
    verifier: Verifier
    dropped: int
    # One line per drop this module can name, "<check id>: <reason>"; an ungrounded because is counted only.
    reasons: tuple[str, ...] = ()
    # (check id, one of UNSATISFIABLE) per check that makes the Spec unpassable; empty when a Run can pass.
    unsatisfiable: tuple[tuple[str, str], ...] = ()


def _write(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if demand.get("tool") not in write_tools or not demand.get("id_field"):
        return None
    tool, entity = demand["tool"], str(demand.get("entity", ""))
    # The atom holds the id as the write effect keys it (text_of(fn(raw))), as derive writes it, since check_run
    # compares that form; canon_key would add JSON quotes and never match.
    base = {"tool": tool, "entity": verifier_suite.text_of(fn(entity)), "entity_raw": entity,
            "id_field": demand["id_field"], "at": 0}
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
    """No write at all, or, when the demand names a tool, no call of that tool alone."""
    tools = {demand["tool"]} if demand.get("tool") else write_tools
    return [no_write_atom(tools).model_copy(update={"id": atom_id})]


def _cap(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    """At most `count` writes, or, when the demand names a tool, at most `count` calls of that tool."""
    if not isinstance(demand.get("count"), int):
        return None
    if demand.get("tool"):
        return [tool_cap_atom(atom_id, demand["tool"], demand["count"])]
    return [verifier_suite.make_atom(
        atom_id, "required", {"kind": "entity_count", "count": demand["count"]},
        description=f"the Run makes at most {demand['count']} write calls")]


def _shape(atom_id: str, kind: str, demand: dict, write_tools: set, fn: Callable) -> Optional[list[Atom]]:
    if demand.get("tool") not in write_tools or not demand.get("field") or not demand.get("id_field"):
        return None
    return [shape_atom(atom_id, demand["tool"], demand["field"], demand["id_field"], write_tools)]


BUILDERS = {"write": _write, "say": _say, "ask": _ask, "no_write": _no_write, "cap": _cap, "shape": _shape}


def _action(atom_id: str, demand: dict) -> list[Atom]:
    """A write or no-write demand on an action tool: required, allowed or forbidden as the call happening."""
    if demand.get("demand") == "no_write" or demand["kind"] == "forbidden":
        return action_atoms(atom_id, "forbidden", demand["tool"])
    return action_atoms(atom_id, "required" if demand["kind"] in ("required", "hard") else "allowed",
                        demand["tool"])


def _what(demand: dict) -> dict:
    return {key: value for key, value in demand.items() if key not in _NOT_DEMANDED}


def unsatisfiable(demands: list[tuple[str, dict]], write_tools: set, action_tools: set) -> list[tuple[str, str]]:
    """The (check id, reason) pairs that make these grounded demands unpassable, in check order."""
    out: list[tuple[str, str]] = []
    required = [(cid, d) for cid, d in demands if d["kind"] == "required" and d.get("demand") == "write"]
    for cid, demand in required:
        if demand.get("tool") in write_tools and (not demand.get("id_field") or isinstance(demand.get("entity"), dict)):
            out.append((cid, WRITE_WITHOUT_ROW))
    for cid, demand in demands:
        if demand.get("demand") == "no_write":
            covered = {demand["tool"]} if demand.get("tool") else write_tools
            clash = [other for other, d in required if d.get("tool") in covered]
            out += [(other, NO_WRITE_BESIDE_REQUIRED) for other in [cid] * bool(clash) + clash]
        if demand["kind"] == "forbidden":
            same = [other for other, d in demands if d["kind"] == "required" and _what(d) == _what(demand)]
            out += [(other, FORBIDDEN_EQUALS_REQUIRED) for other in [cid] * bool(same) + same]
    order = {cid: n for n, (cid, _) in enumerate(demands)}
    return sorted(dict.fromkeys(out), key=lambda pair: (order[pair[0]], pair[1]))


def _because(atom: Atom, because: str) -> Atom:
    return atom.model_copy(update={"description": f"{atom.description or ''} because: {because}"})


def compile_demands(demands: list, intent: str, sections: Iterable[str], write_tools: Iterable[str],
                    fn: Callable, task_id: str = "", version: str = "spec",
                    action_tools: Iterable[str] = ()) -> Compiled:
    """Demand dicts (each with kind, demand and because) to a Verifier, and how many were dropped."""
    actions = set(action_tools)
    tools, sections = set(write_tools) - actions, list(sections)
    atoms: list[Atom] = []
    dropped = 0
    reasons: list[str] = []
    grounded: list[tuple[str, dict]] = []
    for number, demand in enumerate(demands):
        if (not isinstance(demand, dict) or demand.get("kind") not in ATOM_KINDS
                or not valid_because(demand.get("because"), intent, sections)):
            dropped += 1
            continue
        grounded.append((str(demand.get("check_id") or f"i{number}"), demand))
        if demand.get("tool") in actions and demand.get("demand") in ("write", "no_write"):
            atoms += [_because(atom, str(demand["because"])) for atom in _action(f"i{number}", demand)]
            continue
        if demand.get("demand") == "write" and not demand.get("entity"):
            # An empty entity never matches a row, so the atom would fail every Run, the Reference included.
            dropped += 1
            reasons.append(f"{demand.get('check_id') or f'i{number}'}: {WRITE_WITHOUT_ENTITY}")
            continue
        build = BUILDERS.get(demand.get("demand"))
        built = build(f"i{number}", demand["kind"], demand, tools, fn) if build else None
        if not built:
            dropped += 1
            continue
        atoms += [_because(atom, str(demand["because"])) for atom in built]
    return Compiled(Verifier(task_id=task_id, atoms=atoms, verifier_version=version, seed_run_ids=[]), dropped,
                    tuple(reasons), tuple(unsatisfiable(grounded, tools, actions)))


def compile_spec(spec, write_tools: Iterable[str], canon_fn: Callable,
                 policy_sections: Iterable[str] = (), action_tools: Iterable[str] = ()) -> Compiled:
    """The Spec's checks as a Verifier; a check whose because is not grounded is dropped and counted."""
    demands = [dict(check.demand, kind=check.kind, because=check.because, check_id=check.id) for check in spec.checks]
    return compile_demands(demands, spec.intent.text, policy_sections, write_tools, canon_fn,
                           task_id=spec.task_id, version=f"spec{spec.version}", action_tools=action_tools)
