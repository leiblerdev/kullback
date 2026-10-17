"""Code mutants of the Task's base end state, keyed to the ask: each is that state with one change, each should fail.

The base is built by code from the Verifier, never from a Run (D375, `constructed_base`): the Task's start
state with every gated state cell set to its expected value and every gated new row made. On it the state
gates pass by construction (checked: every gated state item and the sanity item hold, else no base is
built). A gated cell whose expectation is not one literal value (a set of several members, a `not` set, a
row that is not there), or whose value no source reaches (its source None, D380), cannot be built: then the base is the newest fresh Run the Verifier passes
(`passing_base`, D368), and with neither the report says no_base. The Reference is never a base.

Called and judge gates have no base to mutate: no Run's calls or answers are laid on a constructed base.
The do-nothing Run tests a called gate (no call made, it must fail; spec/trust.py); judge gates stay
untested by mutants. A mutant is scored against its base: it is killed when a gate item that held on the
base fails on it, so a called gate failing on the constructed base kills nothing.

Only the changes the ask's gates hold are mutated (`ask_keys`): the cells a gated row_is or row_keeps item
citing an ask names, the columns a gated called item's tool always moves (spec/effects.py), and the tables a
gated row_new item or that tool makes rows in. A keyed cell the base left as it was (a row kept) is flipped
too, so a keep the ask names must hold.

The survivors reach the writer only as the Examiner's rulings (spec/mutant_rulings.py, D362), and trust
reads the untouched sample for the stray-write gate (`untouched_rows`). Four operators, all by code, no model:

1. value: each keyed field the base changed, and each keyed cell it kept, set to another value, the same
   field of another row of the start state where one differs, else the stray value (`trust._stray_value`);
2. undone: each keyed field the base changed on a kept row put back to its start value, each keyed row it
   inserted removed;
3. untouched: one write per table on a row the Verifier does not declare (`trust.stray_writes`), and one
   per field the base left alone on a kept row it changed under a key, valued as in `value`;
4. extra_row: per keyed table the base inserted into, a copy of its first inserted row under a new key.

A change the diff does not show (an exempt column, a canon-equal value) is not built. At most `CAP`
mutants per Task, shuffled per operator with a fixed seed and taken round robin. A mutant copies the
base's turns, so it is scored with the base's judge answers. A kill is for the right reason when a
failing gate item reads the mutated table, row and field (`reads_cell`); for an untouched or extra row,
the sanity item or a new-row count on that table also is.

What mutants cannot do (H4): they test that the gates hold what they declare, keyed to the ask; a gate
that holds the wrong value kills its mutants as readily as a right one, so no count here says a gate is right.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates.verifier_suite import empty_run
from kullback.runner.atom_context import AtomContext
from kullback.runner.records import EXAM_DIR, EndState, ItemResult, Run, Verifier, write_json
from kullback.runner.target import check_run
from kullback.runner.verdict import verdict
from kullback.spec.actions import action_tables_of
from kullback.spec.canfail import has_gates, outcome_of
from kullback.spec.items import value_set_of
from kullback.spec.schema import is_ask
from kullback.spec.trust import _stop, _stray_value, is_fresh, stray_writes

OPERATORS = ("value", "undone", "untouched", "extra_row")
CAP = 40
SEED = "mutants-1010"
MUTANTS_DIR = "mutants"
KILLED, SURVIVED, OPEN = "killed", "survived", "open"
# The report's status when no base can be built from the Verifier and no fresh Run passes it (D375).
NO_BASE = "no_base"
# Which base the mutants and the stray write were laid on.
CONSTRUCTED, FRESH = "constructed", "fresh"
# Where the mutated row is new or gone, a check on the row's count or on what else moved reads it.
ROW_LEVEL = ("untouched", "extra_row")


class Mutant(NamedTuple):
    id: str
    operator: str
    table: str
    row_id: str
    field: Optional[str]
    run: Run


class AskKeys(NamedTuple):
    """What the ask's gates hold: (table, row id, column) cells, (table, column) a called tool moves on any
    row, and tables a row_new item or a called tool makes rows in."""
    cells: frozenset
    columns: frozenset
    tables: frozenset


def _demand_keys(demand: dict, effects: dict) -> tuple[set, set, set]:
    """One gated item's (cells, columns, tables): row_is and row_keeps their cells, row_new its table, called the
    columns and tables its tool writes."""
    kind = demand.get("kind")
    if kind in ("row_is", "row_keeps") and demand.get("row") is not None:
        return {(demand["table"], str(demand["row"]), str(f).split(".")[0]) for f in demand.get("expect") or {}}, set(), set()
    if kind == "row_new":
        return set(), set(), {demand.get("table")}
    if kind == "called":
        effect = effects.get(demand.get("tool")) or {}
        return set(), {(c["table"], c["column"]) for c in effect.get("columns") or ()}, set(effect.get("makes") or ())
    return set(), set(), set()


def ask_keys(spec: Any, effects: Optional[dict] = None) -> AskKeys:
    """The keys of the gated items that cite an ask (schema.is_ask); `effects` is spec/effects.py's
    {tool: {columns, makes}}. A policy-only item or a reaction keys nothing."""
    asks = {fact.id for fact in spec.intent.facts if is_ask(fact)}
    cells, columns, tables = set(), set(), set()
    for check in spec.checks:
        demand = check.demand or {}
        if demand.get("gate") is True and asks & set(demand.get("fact_ids") or ()):
            more = _demand_keys(demand, effects or {})
            cells, columns, tables = cells | more[0], columns | more[1], tables | more[2]
    return AskKeys(frozenset(cells), frozenset(columns), frozenset(tables))


def keyed(change: tuple, keys: Optional[AskKeys]) -> bool:
    """Whether a (table, row id, field, how) change is one the ask's gates hold; every change when keys is None."""
    if keys is None:
        return True
    table, row_id, field, how = change
    if field is None or how == "inserted":
        return table in keys.tables
    column = field.split(".")[0]
    return (table, row_id, column) in keys.cells or (table, column) in keys.columns


def passing_base(verifier: Verifier, runs: Iterable[Run], judged: Optional[dict] = None, *, canon: Any = None,
                 tools: Any = None, schema: Any = None) -> Optional[Run]:
    """The newest fresh Run the Verifier passes, scored with its own judge answers (`judged` by run id), else None."""
    for run in reversed([run for run in runs if is_fresh(run) and _stop(run) is not None]):
        try:
            if outcome_of(verifier, run, canon, tools, schema=schema, judge_results=(judged or {}).get(run.run_id)).passed:
                return run
        except Exception:  # a Run the scorer cannot read is no base
            continue
    return None


class Base(NamedTuple):
    """The end state mutants and the stray write are laid on, and which kind it is (CONSTRUCTED or FRESH)."""
    run: Run
    kind: str


class _Unbuildable(Exception):
    """A gated cell whose expectation is not one literal value: the end state cannot be built by code."""


def _literal(value: Any) -> Any:
    """The one value a cell expects: a plain value or list as it is, a one_of set of one member as that member."""
    if not value_set_of(value):
        return value
    members = value.get("one_of")
    if isinstance(members, list) and len(members) == 1:
        return members[0]
    raise _Unbuildable


def _reached(cell: Any) -> Any:
    """A gated cell's one literal value; a value no source reaches is no end state a Run can make (D380)."""
    if cell.source is None:
        raise _Unbuildable
    return _literal(cell.value)


def _set_path(row: dict, field: str, value: Any) -> None:
    """`row` with `value` at the dotted path `field`, making the dicts on the way; a list step must exist."""
    parts = str(field).split(".")
    node: Any = row
    for part in parts[:-1]:
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node.setdefault(part, {})
        else:
            raise _Unbuildable
    last = parts[-1]
    if isinstance(node, list) and last.isdigit() and int(last) < len(node):
        node[int(last)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise _Unbuildable


def _build_state(state: EndState, end: dict) -> None:
    """`end` (a copy of the start state) with every gated cell of `state` set and every gated new row made."""
    for cell in state.cells:
        if not cell.gate:
            continue
        table = end.setdefault(cell.table, {})
        if cell.field is None:
            if cell.value is None:
                table.pop(cell.row_id, None)
            elif cell.row_id not in table:
                raise _Unbuildable
            continue
        if not isinstance(table.get(cell.row_id), dict):
            raise _Unbuildable
        _set_path(table[cell.row_id], cell.field, _reached(cell))
    for spec in state.new_rows:
        if not spec.get("gate", True):
            continue
        table = end.setdefault(spec["table"], {})
        for _ in range(int(spec.get("count") or 1)):
            key = f"constructed.{len(table)}"
            while key in table:
                key += "x"
            table[key] = {}
            for field, value in sorted((spec.get("where") or {}).items()):
                _set_path(table[key], field, _literal(value))


def _state_gates_hold(verifier: Verifier, run: Run, canon: Any, schema: Any) -> bool:
    """Whether every gated state item (cells, new-row counts) and the sanity item hold on the Run."""
    try:
        result = verdict(run, verifier, canon, None, schema=schema)
    except Exception:  # a state the scorer cannot read is no base
        return False
    state = [item for item in result.items if item.kind in ("state", "sanity") and item.gate]
    return bool(state) and all(item.holds is True for item in state)


def constructed_base(verifier: Verifier, runs: Iterable[Run], *, canon: Any = None,
                     schema: Any = None) -> Optional[Run]:
    """The Verifier's expected end state built by code on the Task's start state (D375), else None.

    The start state is read off the first Run that carries one (any Run: the start state is the Task's).
    Each expected end state is tried in order; the first whose gated cells are all literal values and on which
    every gated state item and the sanity item then hold is the base. The Run has no call and no message.
    """
    start = next((run for run in runs if _stop(run) is not None), None)
    if start is None or not has_gates(verifier):
        return None
    for state in verifier.expected:
        run = empty_run(start)
        run.run_id = f"{verifier.task_id}.constructed"
        try:
            _build_state(state, run.events[_stop(run)].payload["end_state"])
        except _Unbuildable:
            continue
        if _state_gates_hold(verifier, run, canon, schema):
            return run
    return None


def mutant_base(verifier: Verifier, runs: Iterable[Run], judged: Optional[dict] = None, *, canon: Any = None,
                tools: Any = None, schema: Any = None) -> Optional[Base]:
    """The base built from the Verifier (`constructed_base`), else the newest passing fresh Run, else None."""
    runs = list(runs)
    built = constructed_base(verifier, runs, canon=canon, schema=schema)
    if built is not None:
        return Base(built, CONSTRUCTED)
    fresh = passing_base(verifier, runs, judged, canon=canon, tools=tools, schema=schema)
    return Base(fresh, FRESH) if fresh is not None else None


def _states(run: Run) -> tuple[int, dict, dict]:
    at = _stop(run)
    payload = run.events[at].payload
    return at, payload["start_state"], payload["end_state"]


def run_changes(run: Run, canon: Any = None, schema: Any = None) -> list[tuple[str, str, Optional[str], str]]:
    """(table, row id, field, how) for each change the Run made: "kept" for a field of a kept row,
    "inserted" for a field of a new row, and (table, row id, None, "inserted"/"removed") per row made or gone.
    Tables that record calls are not state and are skipped."""
    skip = action_tables_of(schema)
    out = []
    for key, moved in sorted(AtomContext(run, canon, schema=schema).diff().items()):
        table, _, row_id = key.partition(".")
        if table in skip:
            continue
        if moved["present_before"] and moved["present_after"]:
            out += [(table, row_id, field, "kept") for field in sorted(moved["fields"])]
        elif moved["present_after"]:
            out += [(table, row_id, None, "inserted")]
            out += [(table, row_id, field, "inserted") for field in sorted(moved["fields"])]
        else:
            out.append((table, row_id, None, "removed"))
    return out


def _other_value(start: dict, table: str, row_id: str, field: str, now: Any, norm: Any) -> Any:
    """The same field of another start row whose value differs from `now` after `norm` (the canon), in
    sorted order, else the stray value; None where neither stands."""
    for other in sorted(start.get(table) or {}, key=str):
        row = (start.get(table) or {}).get(other)
        if str(other) != row_id and isinstance(row, dict) and field in row and row[field] is not None \
                and norm(row[field]) != norm(now):
            return row[field]
    return _stray_value(now)


def _with(base: Run, mutant_id: str, change: Any) -> Run:
    run = base.model_copy(deep=True)
    run.run_id = f"{base.run_id}.mutant.{mutant_id}"
    change(run.events[_stop(run)].payload["end_state"])
    return run


def _set(table: str, row_id: str, field: str, value: Any):
    def change(end: dict) -> None:
        end[table][row_id][field] = value
    return change


def _drop(table: str, row_id: str):
    def change(end: dict) -> None:
        end[table].pop(row_id, None)
    return change


def _put(table: str, row_id: str, row: dict):
    def change(end: dict) -> None:
        end[table][row_id] = dict(row)
    return change


def _value_mutants(base: Run, changes: list, norm: Any) -> Iterable[tuple]:
    _, start, end = _states(base)
    for table, row_id, field, _how in changes:
        if field is None:
            continue
        now = (end.get(table) or {}).get(row_id, {}).get(field)
        value = _other_value(start, table, row_id, field, now, norm)
        if value is not None:
            yield "value", table, row_id, field, _set(table, row_id, field, value)


def _undone_mutants(base: Run, changes: list) -> Iterable[tuple]:
    _, start, _ = _states(base)
    for table, row_id, field, how in changes:
        if how == "kept":
            yield "undone", table, row_id, field, _set(table, row_id, field, start[table][row_id].get(field))
        elif how == "inserted" and field is None:
            yield "undone", table, row_id, None, _drop(table, row_id)
        elif how == "removed":
            yield "undone", table, row_id, None, _put(table, row_id, start[table][row_id])


def _untouched_mutants(verifier: Verifier, base: Run, changes: list, canon: Any, schema: Any,
                       norm: Any) -> Iterable[tuple]:
    for table, row_id, field, run in stray_writes(verifier, base, schema=schema, canon=canon):
        value = run.events[_stop(run)].payload["end_state"][table][row_id][field]
        yield "untouched", table, row_id, field, _set(table, row_id, field, value)
    _, start, end = _states(base)
    written: dict[tuple[str, str], set] = {}
    for table, row_id, field, how in changes:
        if how == "kept":
            written.setdefault((table, row_id), set()).add(field)
    for (table, row_id), fields in sorted(written.items()):
        row = end[table][row_id]
        for field in sorted(set(row) - fields, key=str):
            value = _other_value(start, table, row_id, field, row[field], norm)
            if value is not None:
                yield "untouched", table, row_id, field, _set(table, row_id, field, value)


def _extra_row_mutants(base: Run, changes: list) -> Iterable[tuple]:
    _, _, end = _states(base)
    seen = set()
    for table, row_id, field, how in changes:
        if how != "inserted" or field is not None or table in seen:
            continue
        seen.add(table)
        key = f"{row_id}.extra"
        while key in end[table]:
            key += "x"
        yield "extra_row", table, key, None, _put(table, key, end[table][row_id])


def _kept_cells(base: Run, changes: list, keys: Optional[AskKeys]) -> list[tuple]:
    """(table, row id, column, "kept") per keyed cell the base left as it was, on a row it holds: a keep the
    ask names, flipped by `value` like a change."""
    _, _, end = _states(base)
    moved = {(table, row_id, str(field).split(".")[0]) for table, row_id, field, _ in changes if field is not None}
    return [(table, row_id, column, "kept") for table, row_id, column in sorted((keys.cells if keys else ()), key=str)
            if (table, row_id, column) not in moved and column in ((end.get(table) or {}).get(row_id) or {})]


def candidate_mutants(verifier: Verifier, base: Run, keys: Optional[AskKeys] = None, *, canon: Any = None,
                      schema: Any = None) -> tuple[list[Mutant], int]:
    """(every mutant of a keyed change whose change the diff shows, how many were not built because it shows
    nothing)."""
    changes = [change for change in run_changes(base, canon, schema) if keyed(change, keys)]
    before = AtomContext(base, canon, schema=schema)
    made = [*_value_mutants(base, changes + _kept_cells(base, changes, keys), before.c), *_undone_mutants(base, changes),
            *_untouched_mutants(verifier, base, changes, canon, schema, before.c), *_extra_row_mutants(base, changes)]
    before = before.diff()
    out, no_effect = [], 0
    for n, (operator, table, row_id, field, change) in enumerate(made):
        mutant_id = f"{operator}.{n}"
        run = _with(base, mutant_id, change)
        if AtomContext(run, canon, schema=schema).diff() == before:
            no_effect += 1
            continue
        out.append(Mutant(mutant_id, operator, table, row_id, field, run))
    return out, no_effect


def capped(mutants: list[Mutant], task_id: str, cap: int = CAP) -> list[Mutant]:
    """At most `cap` mutants: shuffled within each operator with the Task's seed, then round robin."""
    rng = random.Random(f"{SEED}:{task_id}")
    pools = []
    for operator in OPERATORS:
        pool = [m for m in mutants if m.operator == operator]
        rng.shuffle(pool)
        pools.append(pool)
    out: list[Mutant] = []
    while len(out) < cap and any(pools):
        for pool in pools:
            if pool and len(out) < cap:
                out.append(pool.pop(0))
    return out


# --- scoring ---------------------------------------------------------------------------------------

def _names_row(where: Optional[str], table: str, row_id: str, field: Optional[str]) -> bool:
    """Whether a `collateral:<table>.<row>[.<field>]` reason names the mutated row (and field, given one)."""
    prefix = f"collateral:{table}.{row_id}"
    text = str(where or "")
    if text == prefix:
        return True
    return text.startswith(prefix + ".") and (field is None or text == f"{prefix}.{field}")


def reads_cell(item: ItemResult, mutant: Mutant, verifier: Verifier) -> bool:
    """Whether a failing gate item reads the mutated table, row and field (any field for a row mutant)."""
    table, row_id, field = mutant.table, mutant.row_id, mutant.field
    if item.kind == "state" and item.id.startswith(f"state:{table}.{row_id}."):
        cell = item.id[len(f"state:{table}.{row_id}."):].split(".")[0]
        return field is None or cell == field
    if item.kind == "state" and item.id.startswith(f"new_rows:{table}:"):
        return _count_reads(verifier, mutant)
    if item.kind == "sanity":
        return _names_row(item.why, table, row_id, field)
    if item.id.startswith("forbidden:"):
        rule = verifier.forbidden[int(item.id.split(":")[1])]
        return all(want in (None, got) for want, got in ((rule.table, table), (rule.row_id, row_id),
                                                         (rule.field, field)))
    return _atom_reads(verifier, item.id, mutant)


def _count_reads(verifier: Verifier, mutant: Mutant) -> bool:
    """Whether a new-row count on the mutated table reads the mutant: a row mutant, or a field its `where` names."""
    wheres = {name for state in verifier.expected for spec in state.new_rows
              if spec.get("table") == mutant.table for name in (spec.get("where") or {})}
    return mutant.operator in ROW_LEVEL or mutant.field is None or mutant.field in wheres


def _atom_reads(verifier: Verifier, atom_id: Any, mutant: Mutant) -> bool:
    """Whether the atom's target names the mutated table and row."""
    target = next((atom.target or {} for atom in verifier.atoms if atom.id == atom_id), {})
    return target.get("table") == mutant.table and str(target.get("row") or target.get("entity") or "") == mutant.row_id


def held_gates(verifier: Verifier, base: Run, canon: Any, tools: Any, schema: Any,
               judge_results: Optional[dict]) -> Optional[frozenset]:
    """The ids of the gate items that hold on the base, which a mutant is scored against; None for a Verifier
    with no gates (its atoms decide)."""
    if not has_gates(verifier):
        return None
    result = verdict(base, verifier, canon, judge_results, write_tools=tools, schema=schema)
    return frozenset(item.id for item in result.items if item.gate and item.holds is True)


def score_mutant(verifier: Verifier, mutant: Mutant, canon: Any, tools: Any, schema: Any,
                 judge_results: Optional[dict], held: Optional[frozenset] = None) -> dict:
    """{id, operator, table, row, field, outcome, at, failing, right_reason}: killed when a gate item that held
    on the base (`held`, every gate item when None) holds False, survived when the Verdict passes or every
    one of them still holds, else open (not verdicted)."""
    row = {"id": mutant.id, "operator": mutant.operator, "table": mutant.table, "row": mutant.row_id,
           "field": mutant.field}
    if not has_gates(verifier):
        passed, atom = check_run(verifier, mutant.run, canon, write_tools=tools)
        name = None if passed else str(getattr(atom, "id", atom) or "atom")
        return dict(row, outcome=SURVIVED if passed else KILLED, at=name, failing=[name] if name else [],
                    right_reason=None if passed else _atom_reads(verifier, name, mutant))
    result = verdict(mutant.run, verifier, canon, judge_results, write_tools=tools, schema=schema)
    outcome, failing = _gate_outcome(result, held)
    return dict(row, outcome=outcome, at=failing[0].id if failing else result.failing_atom,
                failing=[item.id for item in failing],
                right_reason=any(reads_cell(item, mutant, verifier) for item in failing) if failing else None)


def _gate_outcome(result: Any, held: Optional[frozenset]) -> tuple[str, list]:
    """The outcome on the gate items that held on the base (every gate item when None), and the failing ones."""
    gates = [item for item in result.items if item.gate and (held is None or item.id in held)]
    failing = [item for item in gates if item.holds is False]
    if failing:
        return KILLED, failing
    still = held is not None and all(item.holds is True for item in gates)
    return (SURVIVED if result.passed or still else OPEN), failing


def counts(rows: list[dict]) -> dict:
    """built, killed, survived, open and wrong_reason, overall and per operator."""
    def tally(part: list[dict]) -> dict:
        return {"built": len(part), "killed": sum(r["outcome"] == KILLED for r in part),
                "survived": sum(r["outcome"] == SURVIVED for r in part),
                "open": sum(r["outcome"] == OPEN for r in part),
                "wrong_reason": sum(r["outcome"] == KILLED and r["right_reason"] is False for r in part)}
    return dict(tally(rows), by_operator={op: tally([r for r in rows if r["operator"] == op]) for op in OPERATORS})


def mutant_report(verifier: Verifier, runs: Iterable[Run], keys: Optional[AskKeys] = None, *, canon: Any = None,
                  tools: Any = None, schema: Any = None, judged: Optional[dict] = None, cap: int = CAP) -> dict:
    """One Task's mutant report on its base (`mutant_base`: built from the Verifier, else the newest passing
    fresh Run); no_base without one. `judged` maps a run id to its judge answers; the mutants are scored with
    the base's, a constructed base has none."""
    base = mutant_base(verifier, runs, judged, canon=canon, tools=tools, schema=schema)
    if base is None:
        return {"task_id": verifier.task_id, "status": NO_BASE}
    answers = (judged or {}).get(base.run.run_id)
    made, no_effect = candidate_mutants(verifier, base.run, keys, canon=canon, schema=schema)
    kept = capped(made, verifier.task_id, cap)
    held = held_gates(verifier, base.run, canon, tools, schema, answers)
    rows = [score_mutant(verifier, m, canon, tools, schema, answers, held) for m in kept]
    return dict({"task_id": verifier.task_id, "base": base.run.run_id, "base_kind": base.kind}, status="scored",
                candidates=len(made), dropped=len(made) - len(kept), no_effect=no_effect, **counts(rows),
                mutants=rows)


def mutants_path(workdir: Any, task_id: str) -> Path:
    return Path(workdir) / EXAM_DIR / MUTANTS_DIR / f"{task_id}.json"


def mutant_context(workdir: Any) -> tuple[Any, Any, Any]:
    """(canon, write tools, schema) of a workdir, as scoring a mutant reads them."""
    from kullback.gates.probes import write_tools_of
    from kullback.gates.trust import schema_of
    from kullback.runner.canon import load_rules
    from kullback.runner.records import read_json
    from kullback.spec.actions import workdir_action_tools

    root = Path(workdir)
    sigs = read_json(root / "tool_sigs.json", None) or []
    tools = write_tools_of(sigs.get("sigs", []) if isinstance(sigs, dict) else sigs) - workdir_action_tools(root)
    return load_rules(root / "canon-rules.json"), tools, schema_of(root)


def untouched_rows(verifier: Verifier, base: Run, keys: Optional[AskKeys] = None, *, canon: Any = None,
                   tools: Any = None, schema: Any = None, judge_results: Optional[dict] = None,
                   cap: int = CAP) -> list[dict]:
    """The untouched mutants of the Task's capped sample (the ones `mutant_report` builds), scored on `base`."""
    made, _ = candidate_mutants(verifier, base, keys, canon=canon, schema=schema)
    held = held_gates(verifier, base, canon, tools, schema, judge_results)
    return [score_mutant(verifier, m, canon, tools, schema, judge_results, held)
            for m in capped(made, verifier.task_id, cap) if m.operator == "untouched"]


def task_mutants(workdir: Any, verifier: Verifier, runs: Optional[list[Run]] = None, cap: int = CAP) -> dict:
    """One Task's mutant report from the workdir: its Spec's ask keys, its Runs and their kept judge answers."""
    from kullback.spec import effects as effects_mod
    from kullback.spec.judged import judge_results_of
    from kullback.spec.schema import load_spec
    from kullback.spec.trust import runs_on_disk

    root = Path(workdir)
    canon, tools, schema = mutant_context(root)
    task_runs = runs if runs is not None else runs_on_disk(root).get(verifier.task_id, [])
    spec = load_spec(root, verifier.task_id)
    keys = ask_keys(spec, effects_mod.load(root)["effects"]) if spec is not None else AskKeys(*(frozenset(),) * 3)
    return mutant_report(verifier, task_runs, keys, canon=canon, tools=tools, schema=schema,
                         judged=judge_results_of(root, verifier.task_id, verifier, task_runs), cap=cap)


def workdir_mutants(workdir: Any, task_ids: Optional[Iterable[str]] = None, cap: int = CAP) -> dict[str, dict]:
    """Every Task with a Verifier on disk (or the named ones): its mutant report, written to
    exam/mutants/<task>.json."""
    from kullback.spec.schema import spec_verifier_path
    from kullback.spec.trust import load_spec_verifier, runs_on_disk

    root = Path(workdir)
    runs = runs_on_disk(root)
    folder = spec_verifier_path(root, "_").parent
    wanted = sorted(task_ids) if task_ids is not None else sorted(p.stem for p in folder.glob("*.json"))
    out = {}
    for task_id in wanted:
        verifier = load_spec_verifier(root, task_id)
        if verifier is None:
            continue
        report = task_mutants(root, verifier, runs.get(task_id, []), cap)
        write_json(mutants_path(root, task_id), report)
        out[task_id] = report
    return out
