"""Three repairs the Harbor export makes to a written Verifier, by code, before it is laid (D431, D432, D437).

1. A cell the scorer cannot read (D423) whose path meets a list at a non-index part, read on the start row (the
   row the writer saw). A value that is a list with one entry per member of that list, members a required tool
   appends included (its learned effects write into that list), is one indexed cell per member: the same gate,
   the weight split evenly, in the start row's member order (D437). A list of exactly one member and one value
   names that member by index 0 (D431, the one-member case). Any other count is left as it is, and the D423
   gate sets the Task aside: no member is guessed. The Reference is never read.
2. The world's own effects are not the agent's collateral. A tool the Verifier requires moves columns every one
   of its calls moves (spec/effects.py). Required are: a tool a called item names; the one tool whose learned
   effects alone answer a declared row, as the Spec stage reads it (spec/items.py `expected_tools`); and a tool
   the Task's kept References call on a table the end state declares a row in. Each end state
   gains an allowed rule per such column, on the declared rows of its table or on every row where the
   end state declares none, unless a cell already checks it or a rule already frees it. A tool the Verifier
   does not require gains nothing, so a stray write still fails sanity.
3. A called rule over a set of values (spec/atoms.py `called_set_atom`) matches a set against one argument value,
   so a list argument never holds. Each such rule, not a prohibition, tests every member of a list argument
   against the set (D437).
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.expected import path_value
from kullback.runner.records import Verifier, as_dict, write_json
from kullback.spec import effects as learned_effects
from kullback.spec.atoms import CALLED_SET_ATOM
from kullback.spec.items import expected_tools, unreadable_path

EFFECT_WHY = "free: learned effect: {tool}"
FROM_FILE = "effects file"
LEARNED_ABSENT = "learned from the recordings: no effects file"
LEARNED_STALE = "learned from the recordings: every writes list in the effects file is empty"


# --- 1. a cell on a list leaf (D431, D437) ---------------------------------------------


def _list_at(row: Any, field: str) -> Optional[tuple[str, list, str]]:
    """(the path to the first list `field` meets at a non-index part, that list on `row`, the rest of the path),
    or None where it meets none."""
    node, walked, parts = row, [], str(field).split(".")
    for n, part in enumerate(parts):
        if isinstance(node, list) and not part.isdigit():
            return ".".join(walked), node, ".".join(parts[n:])
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list):
            node = node[int(part)] if int(part) < len(node) else None
        else:
            return None
        walked.append(part)
    return None


def _appends(effects: dict, tools: Iterable[str], table: str, at: str) -> bool:
    """Whether a required tool's learned effects write into the list at `at` (a list pattern `<at>.N`), so
    the end may hold members the start row lacks."""
    want = ".".join("N" if part.isdigit() else part for part in at.split(".")) + ".N"
    return any(str(column["table"]) == table and any(str(p) == want or str(p).startswith(want + ".")
                                                     for p in column.get("patterns") or ())
               for tool in tools for column in (effects.get(tool) or {}).get("columns") or [])


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _aligned(values: list, members: list, rest: str) -> list:
    """`values` in the start row's member order: where the members' own leaves are all among the values, each
    member takes its own and the rest follow in the writer's order; else the writer's order as it is."""
    leaves = [_key(path_value(member, rest)) for member in members]
    pool = [_key(value) for value in values]
    if any(pool.count(leaf) < leaves.count(leaf) for leaf in set(leaves)):
        return list(values)
    left = list(values)
    head = []
    for leaf in leaves:
        at = next(n for n, value in enumerate(left) if _key(value) == leaf)
        head.append(left.pop(at))
    return head + left


def _members(cell: Any, at: str, values: list, members: list, rest: str) -> list:
    """One cell per member of the list at `at`: same gate, the weight split evenly, value i on member i."""
    weight = cell.weight / len(values)
    return [cell.model_copy(update={"field": f"{at}.{n}.{rest}", "value": value, "weight": weight,
                                    "leaf_sources": {k[len(f"{n}."):]: v for k, v in cell.leaf_sources.items()
                                                     if k.startswith(f"{n}.")}})
            for n, value in enumerate(_aligned(values, members, rest))]


def resolve_cell(cell: Any, row: Any, appends: Any = lambda at: False) -> list:
    """The cells an unreadable cell on a list leaf stands for, on its start `row`, or [cell] where none can.

    A value that is a list with one entry per member (members a required tool appends included, `appends(at)`
    true where one does) is one indexed cell per member (D437). Else a list of exactly one member names it by
    index 0 and keeps the value (D431), where the value is one value or the member's leaf is itself a list (the
    value is then the whole leaf). A value list whose length does not match the member count stays as it is,
    as does one value under a list of several: no member is guessed.
    """
    found = _list_at(row, cell.field) if cell.field else None
    if found is None:
        return [cell]
    at, members, rest = found
    values = cell.value
    whole_leaf = any(isinstance(path_value(member, rest), list) for member in members)
    if isinstance(values, list) and values and not whole_leaf and (
            len(values) == len(members) or (len(values) > len(members) and appends(at))):
        cells = _members(cell, at, values, members, rest)
    elif len(members) == 1 and (whole_leaf or not isinstance(values, list)):
        cells = [cell.model_copy(update={"field": f"{at}.0.{rest}"})]
    else:
        return [cell]
    return [one for made in cells for one in resolve_cell(made, row, appends)]


def resolve_cells(verifier: Verifier, start: dict, effects: Optional[dict] = None) -> tuple[Verifier, list[str]]:
    """The Verifier with every unreadable cell on a start list resolved (`resolve_cell`), and what moved,
    "<table>: <old path> -> <new path>, ..." once per change. Gates are kept; a projected cell's weight is split.
    `effects` (per tool, its learned effects) decides which tools the Verifier requires and so may append."""
    effects = effects or {}
    tools = required_tools(verifier, effects) if effects else []
    resolved: list[str] = []
    states = []
    for state in verifier.expected:
        cells = []
        for cell in state.cells:
            row = (start.get(cell.table) or {}).get(cell.row_id)
            made = resolve_cell(cell, row, lambda at, table=cell.table: _appends(effects, tools, table, at)) \
                if cell.field and unreadable_path(row, cell.field) else [cell]
            cells += made
            if made != [cell]:
                moved = f"{cell.table}: {cell.field} -> {', '.join(str(one.field) for one in made)}"
                resolved += [moved] if moved not in resolved else []
        states.append(state.model_copy(update={"cells": cells}))
    return verifier.model_copy(update={"expected": states}), resolved


def resolve_job(job: dict, verifier: Verifier, start: dict) -> Verifier:
    """`resolve_cells` and `list_sets` on one export job (hub/harbor.py `_job`), with the build's learned effects:
    the paths moved go to its metadata as `cells_resolved`, the rules widened as `list_sets`, and a Verifier that
    changed is written under the job's scratch folder and laid from there."""
    effects, _ = tool_effects(Path(job["workdir"]))
    verifier, resolved = resolve_cells(verifier, start, effects)
    verifier, widened = list_sets(verifier)
    job["metadata"]["cells_resolved"] = resolved
    job["metadata"]["list_sets"] = widened
    if resolved or widened:
        job["verifier_path"] = write_json(Path(job["scratch"]) / f"{job['task_id']}.json", as_dict(verifier))
    return verifier


def counts(listed: Iterable[dict]) -> dict:
    """The manifest's totals over the exported Tasks' rows: cells resolved (D431, D437), called sets widened over
    a list (D437), allowed rules added (D432)."""
    rows = list(listed)
    return {"cells_resolved": sum(len(row.get("cells_resolved") or ()) for row in rows),
            "list_sets": sum(len(row.get("list_sets") or ()) for row in rows),
            "effects_allowed": sum(int(row.get("effects_allowed") or 0) for row in rows)}


# --- 2. the world's own effects (D432) -----------------------------------------------


def _stale(effects: dict) -> bool:
    return bool(effects) and all(not (row or {}).get("writes") for row in effects.values())


@lru_cache(maxsize=8)
def tool_effects(workdir: Path) -> tuple[dict, str]:
    """(per tool, what every one of its calls moves, where that came from): the build's effects file, or
    learned again from the recordings, offline, where the file is absent or stale (`builder/tool_effects.py`)."""
    stored = learned_effects.load(workdir)
    if stored["found"] and not _stale(stored["effects"]):
        return stored["effects"], FROM_FILE
    from kullback.builder.tool_effects import learn

    with tempfile.TemporaryDirectory() as scratch:
        effects = learn(workdir, scratch)["effects"]
    return effects, LEARNED_STALE if stored["found"] else LEARNED_ABSENT


def _declared_tables(verifier: Verifier) -> set[str]:
    return {cell.table for state in verifier.expected for cell in state.cells} | {
        str(row.get("table")) for state in verifier.expected for row in state.new_rows}


def _touched(learned: dict) -> set[str]:
    """The tables a tool's learned effects move columns of or make rows in."""
    return {str(column["table"]) for column in learned.get("columns") or []} | set(learned.get("makes") or [])


def _items(verifier: Verifier) -> list[dict]:
    """The Verifier read back as the Spec stage's items: its called items, one row_is per declared row of an end
    state with the columns its cells expect, one row_new per declared new row."""
    called = [dict(atom.target or {}) for atom in verifier.atoms if (atom.target or {}).get("kind") == "called"]
    rows: dict[tuple, dict] = {}
    for state in verifier.expected:
        for cell in state.cells:
            row = rows.setdefault((cell.table, cell.row_id), {"kind": "row_is", "table": cell.table,
                                                             "row": cell.row_id, "expect": {}})
            row["expect"].update({cell.field: cell.value} if cell.field else {})
    made = [{"kind": "row_new", "table": str(row.get("table"))} for state in verifier.expected for row in state.new_rows]
    return called + list(rows.values()) + made


def required_tools(verifier: Verifier, effects: dict, oracle: Iterable[dict] = ()) -> list[str]:
    """The write tools the Verifier requires: those its items expect (`expected_tools`: a called item's tool, the
    one tool that alone answers a declared row), and those the oracle calls whose learned effects move or make
    rows in a table the end states declare a row in."""
    named = set(expected_tools(_items(verifier), effects))
    tables = _declared_tables(verifier)
    made = {name for name in (str(call.get("name")) for call in oracle)
            if name in effects and tables & _touched(effects[name])}
    return sorted((named | made) & set(effects))


def _allowed(rules: list[dict], table: str, row: Optional[str], field: str) -> bool:
    return any(all(rule.get(key) in (None, want) for key, want in (("table", table), ("row_id", row), ("field", field)))
               for rule in rules)


def _effect_rules(state: Any, tools: list[str], effects: dict) -> list[dict]:
    declared: dict[str, set[str]] = {}
    for cell in state.cells:
        declared.setdefault(cell.table, set()).add(cell.row_id)
    checked = {(cell.table, cell.row_id, str(cell.field).split(".")[0]) for cell in state.cells if cell.field}
    rules: list[dict] = []
    for tool in tools:
        for column in effects[tool].get("columns") or []:
            table, field = str(column["table"]), str(column["column"])
            for row in sorted(declared.get(table) or ()) or [None]:
                if (table, row, field) in checked or _allowed(list(state.allowed) + rules, table, row, field):
                    continue
                rule = {"table": table, "field": field, "why": EFFECT_WHY.format(tool=tool)}
                rules.append(dict(rule, row_id=row) if row is not None else rule)
    return rules


def allow_effects(verifier: Verifier, effects: dict, oracle: Iterable[dict] = ()) -> tuple[Verifier, int]:
    """The Verifier with each end state allowing what its required tools always move, and how many distinct
    (table, row, field) rules were added."""
    tools = required_tools(verifier, effects, list(oracle))
    added: set[tuple] = set()
    states = []
    for state in verifier.expected:
        rules = _effect_rules(state, tools, effects)
        added |= {(rule["table"], rule.get("row_id"), rule["field"]) for rule in rules}
        states.append(state.model_copy(update={"allowed": list(state.allowed) + rules}) if rules else state)
    return verifier.model_copy(update={"expected": states}), len(added)


# --- 3. a called set over a list argument (D437) ---------------------------------------

# The line of the called-set rule (spec/atoms.py `called_set_atom`) that matches a set against one value, and
# what it becomes: a list argument, where no member of the set is itself a list, holds when every member does.
SET_LINE = '        return any(_same(one, got) for one in want["one_of"])\n'
SET_OVER_LIST = (
    '        if any(_same(one, got) for one in want["one_of"]):\n'
    '            return True\n'
    '        return (isinstance(got, (list, tuple)) and len(got) > 0\n'
    '                and not any(isinstance(one, (list, tuple)) for one in want["one_of"])\n'
    '                and all(any(_same(one, member) for one in want["one_of"]) for member in got))\n')


def list_sets(verifier: Verifier) -> tuple[Verifier, list[str]]:
    """The Verifier with each called-set rule (not a prohibition) testing every member of a list argument against
    its set, and the ids of the rules changed. A rule whose source lacks the set line is left as it is."""
    widened: list[str] = []
    atoms = []
    for atom in verifier.atoms:
        target = atom.target or {}
        if target.get("derived_as") != CALLED_SET_ATOM or target.get("negate") or SET_LINE not in (
                atom.predicate_src or ""):
            atoms.append(atom)
            continue
        widened.append(atom.id)
        source = {"predicate_src": str(target.get("predicate_src") or "").replace(SET_LINE, SET_OVER_LIST)}
        atoms.append(atom.model_copy(update={"predicate_src": atom.predicate_src.replace(SET_LINE, SET_OVER_LIST),
                                             "target": dict(target, **source)}))
    return verifier.model_copy(update={"atoms": atoms}), widened


__all__ = ["EFFECT_WHY", "counts", "FROM_FILE", "LEARNED_ABSENT", "LEARNED_STALE", "allow_effects", "list_sets",
           "required_tools", "resolve_cell", "resolve_cells", "resolve_job", "tool_effects"]
