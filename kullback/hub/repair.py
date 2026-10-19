"""Two repairs the Harbor export makes to a written Verifier, by code, before it is laid (D431, D432).

1. A cell the scorer cannot read (D423) whose path meets a list at a non-index part, where that list holds
   exactly one member on the start row (the row the writer saw): the writer can only have meant that
   member, so the path names it by index 0. The start row alone decides; the Reference is never read and
   no value changes. A list of several members is left as it is, and the D423 gate sets the Task aside.
2. The world's own effects are not the agent's collateral. A tool the Verifier requires moves columns every one
   of its calls moves (spec/effects.py). Required are: a tool a called item names; the one tool whose learned
   effects alone answer a declared row, as the Spec stage reads it (spec/items.py `expected_tools`); and a tool
   the Task's kept References call on a table the end state declares a row in. Each end state
   gains an allowed rule per such column, on the declared rows of its table or on every row where the
   end state declares none, unless a cell already checks it or a rule already frees it. A tool the Verifier
   does not require gains nothing, so a stray write still fails sanity.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import Verifier, as_dict, write_json
from kullback.spec import effects as learned_effects
from kullback.spec.items import expected_tools, unreadable_path

EFFECT_WHY = "free: learned effect: {tool}"
FROM_FILE = "effects file"
LEARNED_ABSENT = "learned from the recordings: no effects file"
LEARNED_STALE = "learned from the recordings: every writes list in the effects file is empty"


# --- 1. a cell on a one-member list (D431) -------------------------------------------


def indexed_path(row: Any, field: str) -> Optional[str]:
    """`field` with each one-member list it meets at a non-index part named by index 0, on `row`; None where
    a list it meets that way holds no member or several, or nothing needed naming."""
    node, parts, changed = row, [], False
    for part in str(field).split("."):
        if isinstance(node, list) and not part.isdigit():
            if len(node) != 1:
                return None
            parts.append("0")
            node, changed = node[0], True
        parts.append(part)
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list):
            node = node[int(part)] if part.isdigit() and int(part) < len(node) else None
        else:
            node = None
    return ".".join(parts) if changed else None


def resolve_cells(verifier: Verifier, start: dict) -> tuple[Verifier, list[str]]:
    """The Verifier with every unreadable cell on a one-member start list named by index, and what moved,
    "<table>: <old path> -> <new path>" once per path. Values, gates and weights are kept."""
    resolved: list[str] = []
    states = []
    for state in verifier.expected:
        cells = []
        for cell in state.cells:
            row = (start.get(cell.table) or {}).get(cell.row_id)
            new = indexed_path(row, cell.field) if cell.field and unreadable_path(row, cell.field) else None
            if new is None:
                cells.append(cell)
                continue
            moved = f"{cell.table}: {cell.field} -> {new}"
            resolved += [moved] if moved not in resolved else []
            cells.append(cell.model_copy(update={"field": new}))
        states.append(state.model_copy(update={"cells": cells}))
    return verifier.model_copy(update={"expected": states}), resolved


def resolve_job(job: dict, verifier: Verifier, start: dict) -> Verifier:
    """`resolve_cells` on one export job (hub/harbor.py `_job`): the paths moved go to its metadata as
    `cells_resolved`, and a Verifier that changed is written under the job's scratch folder and laid from there."""
    verifier, resolved = resolve_cells(verifier, start)
    job["metadata"]["cells_resolved"] = resolved
    if resolved:
        job["verifier_path"] = write_json(Path(job["scratch"]) / f"{job['task_id']}.json", as_dict(verifier))
    return verifier


def counts(listed: Iterable[dict]) -> dict:
    """The manifest's totals over the exported Tasks' rows: cells resolved (D431), allowed rules added (D432)."""
    rows = list(listed)
    return {"cells_resolved": sum(len(row.get("cells_resolved") or ()) for row in rows),
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


__all__ = ["EFFECT_WHY", "counts", "FROM_FILE", "LEARNED_ABSENT", "LEARNED_STALE", "allow_effects", "indexed_path",
           "required_tools", "resolve_cells", "resolve_job", "tool_effects"]
