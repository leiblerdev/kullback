"""Code mutants of a passing fresh Run, keyed to the ask: each is its end state with one change, each should fail.

The base is the newest fresh Run the Verifier passes (`passing_base`), never the Reference (D368): with
none the report says no_passing_run. Only the base's changes the ask's gates hold are mutated
(`ask_keys`): the cells a gated row_is or row_keeps item citing an ask names, the columns a gated called
item's tool always moves (spec/effects.py), and the tables a gated row_new item or that tool makes rows in.

The survivors reach the writer only as the Examiner's rulings (spec/mutant_rulings.py, D362), and trust
reads the untouched sample for the stray-write gate (`untouched_rows`). Four operators, all by code, no model:

1. value: each keyed field the base changed set to another value, the same field of another row of the
   start state where one differs, else the stray value (`trust._stray_value`);
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

from kullback.runner.atom_context import AtomContext
from kullback.runner.records import EXAM_DIR, ItemResult, Run, Verifier, write_json
from kullback.runner.target import check_run
from kullback.runner.verdict import verdict
from kullback.spec.actions import action_tables_of
from kullback.spec.canfail import has_gates, outcome_of
from kullback.spec.schema import is_ask
from kullback.spec.trust import _stop, _stray_value, is_fresh, stray_writes

OPERATORS = ("value", "undone", "untouched", "extra_row")
CAP = 40
SEED = "mutants-1010"
MUTANTS_DIR = "mutants"
KILLED, SURVIVED, OPEN = "killed", "survived", "open"
# The report's status when no fresh Run passes the Verifier: nothing to mutate (H5).
NO_PASSING_RUN = "no_passing_run"
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


def candidate_mutants(verifier: Verifier, base: Run, keys: Optional[AskKeys] = None, *, canon: Any = None,
                      schema: Any = None) -> tuple[list[Mutant], int]:
    """(every mutant of a keyed change whose change the diff shows, how many were not built because it shows
    nothing)."""
    changes = [change for change in run_changes(base, canon, schema) if keyed(change, keys)]
    before = AtomContext(base, canon, schema=schema)
    made = [*_value_mutants(base, changes, before.c), *_undone_mutants(base, changes),
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


def score_mutant(verifier: Verifier, mutant: Mutant, canon: Any, tools: Any, schema: Any,
                 judge_results: Optional[dict]) -> dict:
    """{id, operator, table, row, field, outcome, at, failing, right_reason}: killed when a gate item holds
    False, survived when the Verdict passes, else open (not verdicted)."""
    row = {"id": mutant.id, "operator": mutant.operator, "table": mutant.table, "row": mutant.row_id,
           "field": mutant.field}
    if not has_gates(verifier):
        passed, atom = check_run(verifier, mutant.run, canon, write_tools=tools)
        name = None if passed else str(getattr(atom, "id", atom) or "atom")
        return dict(row, outcome=SURVIVED if passed else KILLED, at=name, failing=[name] if name else [],
                    right_reason=None if passed else _atom_reads(verifier, name, mutant))
    result = verdict(mutant.run, verifier, canon, judge_results, write_tools=tools, schema=schema)
    failing = [item for item in result.items if item.gate and item.holds is False]
    outcome = KILLED if failing else SURVIVED if result.passed else OPEN
    return dict(row, outcome=outcome, at=result.failing_atom, failing=[item.id for item in failing],
                right_reason=any(reads_cell(item, mutant, verifier) for item in failing) if failing else None)


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
    """One Task's mutant report on its newest passing fresh Run (`passing_base`); no_passing_run without one.
    `judged` maps a run id to its judge answers; the mutants are scored with the base's."""
    base = passing_base(verifier, runs, judged, canon=canon, tools=tools, schema=schema)
    if base is None:
        return {"task_id": verifier.task_id, "status": NO_PASSING_RUN}
    answers = (judged or {}).get(base.run_id)
    made, no_effect = candidate_mutants(verifier, base, keys, canon=canon, schema=schema)
    kept = capped(made, verifier.task_id, cap)
    rows = [score_mutant(verifier, m, canon, tools, schema, answers) for m in kept]
    return dict({"task_id": verifier.task_id, "base": base.run_id}, status="scored", candidates=len(made),
                dropped=len(made) - len(kept), no_effect=no_effect, **counts(rows), mutants=rows)


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
    return [score_mutant(verifier, m, canon, tools, schema, judge_results)
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
