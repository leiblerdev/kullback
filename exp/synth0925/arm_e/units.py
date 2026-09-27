"""Arm E step 1: one-write units mined from trusted reference runs.

A unit is one recorded write call plus the goal it served. Binding a unit
onto one anchor (one customer) is by role: an argument that names a row
(its name ends in _id or _ids) is rebound to a row of the same table that
holds the anchor; every other argument keeps its recorded value and becomes
a user fact. No table, tool or field name is fixed here; all of it is read
off the workdir. The body is the precondition oracle: a bound write that
the body refuses is tried on the next row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kullback import graph as graph_mod
from kullback import synthesise as S
from kullback.builder import synth as synth_mod
from kullback.builder.compile_env import ACTIONS_TABLE
from kullback.builder.mine import CONSTANTS_TABLE
from kullback.examiner import variants as variants_mod
from kullback.gates import verifier_suite
from kullback.gates.tool_runs import CRASH_ERRORS
from kullback.sampling import sample_key

SKIP_TABLES = (ACTIONS_TABLE, CONSTANTS_TABLE)

TRIES = 8


def _is_id_arg(name: str) -> bool:
    n = str(name)
    return n.endswith("_id") or n.endswith("_ids")


def trusted_ids(workdir: Any) -> list[str]:
    body = json.loads((Path(workdir) / "task_status.json").read_text())
    out = [
        tid
        for tid, row in body.items()
        if isinstance(row, dict) and row.get("verifier_passed") and row.get("reference_confirmed")
    ]
    return sorted(out)


def askable_fields(workdir: Any) -> set[str]:
    """D210's askable class: fields some recorded user stated.

    synthesise.askable_fields reads user_facts.json, which this workdir does
    not carry; the same class is the union of fact fields over user_rules.
    """
    got = S.askable_fields(workdir)
    if got:
        return set(got)
    out: set[str] = set()
    for path in sorted((Path(workdir) / "user_rules").glob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        facts = body.get("facts") if isinstance(body, dict) else None
        for row in facts or ():
            if isinstance(row, dict) and row.get("field"):
                out.add(str(row["field"]))
    return out


def reference_calls(workdir: Any, task_id: str) -> list[dict]:
    """The write and read calls of one trusted task's first reference run."""
    body = json.loads((Path(workdir) / "task_status.json").read_text())
    row = body.get(task_id) or {}
    run_ids = row.get("reference_run_ids") or []
    if not run_ids:
        return []
    path = Path(workdir) / "runs" / task_id / f"{run_ids[0]}.jsonl"
    if not path.exists():
        return []
    try:
        run = verifier_suite.as_run(str(path))
    except (OSError, ValueError, TypeError):
        return []
    return variants_mod.call_results(run)


def mine_units(world: Any, workdir: Any, trusted: list[str]) -> tuple[list[dict], dict]:
    """One unit per recorded write call that did not error."""
    units: list[dict] = []
    stats = {"trusted": len(trusted), "no_calls": 0, "writes": 0, "errored": 0}
    for tid in trusted:
        calls = reference_calls(workdir, tid)
        if not calls:
            stats["no_calls"] += 1
            continue
        for call in calls:
            name = str(call.get("name") or "")
            if name not in world.write_tools:
                continue
            if call.get("error"):
                stats["errored"] += 1
                continue
            stats["writes"] += 1
            units.append({"src": tid, "tool": name, "args": dict(call.get("args") or {})})
    return units, stats


def coverage_votes(world: Any, mined: list[dict]) -> dict[str, int]:
    """Units naming one row of each table, read off the mined units' arguments."""
    tables = [t for t in (world.schema.tables or ()) if t not in SKIP_TABLES]
    lower: dict[str, set[str]] = {}
    for table in tables:
        lower[table] = {str(k).lower() for k in synth_mod.table_rows(world.db, world.schema, table)}
    votes: dict[str, int] = {}
    for unit in mined:
        vals = set()
        for value in unit["args"].values():
            if isinstance(value, list):
                vals.update(str(v).lower() for v in value)
            else:
                vals.add(str(value).lower())
        for table in tables:
            if vals & lower[table]:
                votes[table] = votes.get(table, 0) + 1
    return votes


def profile_anchor_table(world: Any, workdir: Any, trusted: list[str]) -> str:
    """The customer table: rows the runs' own first reads return.

    Every recorded run opens by looking up who is calling; the table that
    lookup returns is who a long task about one customer is about. No table
    or tool name is fixed: the first read of each trusted reference run is
    replayed and its answer mapped to the table holding it.
    """
    askable = askable_fields(workdir)
    tables = [t for t in (world.schema.tables or ()) if t not in SKIP_TABLES]
    lower: dict[str, set[str]] = {
        t: {str(k).lower() for k in synth_mod.table_rows(world.db, world.schema, t)} for t in tables
    }
    sigs = {s.name: s for s in world.sigs}
    votes: dict[str, int] = {}
    tried = 0
    for tid in trusted:
        if tried >= 12 and votes:
            break
        calls = reference_calls(workdir, tid)
        reads = [c for c in calls if str(c.get("name") or "") not in world.write_tools]
        if not reads:
            continue
        call = reads[0]
        name = str(call.get("name") or "")
        _props, req = S.declared(sigs.get(name))
        if not req or any(r not in askable for r in req):
            continue
        toolkit = world.toolkit(world.fresh_db())
        body = getattr(toolkit, name, None)
        if body is None:
            continue
        try:
            answer = body(**{k: v for k, v in call["args"].items() if k in req})
        except Exception:  # noqa: BLE001 - any refusal just skips this read
            continue
        tried += 1
        vals = {str(v).lower() for vs in S.named_leaves(answer).values() for v in vs}
        if isinstance(answer, (str, int, float)):
            vals.add(str(answer).lower())
        for table in tables:
            if vals & lower[table]:
                votes[table] = votes.get(table, 0) + 1
    if votes:
        return sorted(votes, key=lambda t: (-votes[t], t))[0]
    return anchor_table(world, {})


def anchor_table(world: Any, votes: dict[str, int] | None = None) -> str:
    """The hub table: pointed at widely, pointing back narrowly, named by users.

    Generic one-customer pick: fewest outgoing references to other tables,
    then most mined units naming one of its rows, then fewest rows.
    Votes maps table to units naming it and is read off the mined units.
    """
    tables = [t for t in (world.schema.tables or ()) if t not in SKIP_TABLES]
    hits: dict[str, set[str]] = {}
    for table in tables:
        rows = synth_mod.table_rows(world.db, world.schema, table)
        for _key, content in rows.items():
            for _path, text in graph_mod._leaves(content):
                hits.setdefault(text, set()).add(table)
                hits.setdefault(text.lower(), set()).add(table)
    ids: dict[str, set[str]] = {t: set(synth_mod.table_rows(world.db, world.schema, t)) for t in tables}
    lower: dict[str, set[str]] = {t: {k.lower() for k in v} for t, v in ids.items()}
    votes = votes or {}
    best = ("", 10**9, -1, 10**9)
    for table in tables:
        rows = synth_mod.table_rows(world.db, world.schema, table)
        back: set[str] = set()
        for _key, content in rows.items():
            for _path, text in graph_mod._leaves(content):
                for other in tables:
                    if other != table and (text in ids[other] or text.lower() in lower[other]):
                        back.add(other)
        key = (len(back), -votes.get(table, 0), len(ids[table]))
        if key < (best[1], best[2], best[3]):
            best = (table, key[0], key[1], key[2])
    if best[0]:
        return best[0]
    return sorted(tables)[0]


def anchor_values(world: Any, table: str) -> list[str]:
    rows = synth_mod.table_rows(world.db, world.schema, table)
    synthetic = set(world.schema.synthetic_rows or ())
    real = sorted(k for k in rows if k not in synthetic)
    return real or sorted(rows)


def table_of_id(world: Any, value: str) -> Optional[str]:
    for table in sorted(world.schema.tables or ()):
        rows = synth_mod.table_rows(world.db, world.schema, table)
        if str(value) in rows:
            return table
    return None


def rows_holding(world: Any, table: str, anchor: str) -> list[str]:
    """Rows of one table that belong to the anchor customer, real rows first."""
    synthetic = set(world.schema.synthetic_rows or ())
    want = str(anchor).lower()
    hit = [
        key
        for (tab, key, vals) in _index(world).values()
        if tab == table and (str(key).lower() == want or want in vals)
    ]
    real = sorted(k for k in hit if k not in synthetic)
    return real + sorted(k for k in hit if k in synthetic)


def _row_id_of(world: Any, table: str, key: str) -> str:
    rows = synth_mod.table_rows(world.db, world.schema, table)
    leaves = S.named_leaves(rows.get(key))
    name = S._id_name(world, table)
    vals = leaves.get(name) or []
    return vals[0] if vals else key


_INDEX: dict[str, Any] = {}


def _index(world: Any) -> dict:
    """One row lookup per world: lower key to (table, key, leaf set)."""
    tag = id(world)
    hit = _INDEX.get(tag)
    if hit is not None:
        return hit
    built: dict[str, tuple[str, str, set[str]]] = {}
    for table in world.schema.tables or ():
        rows = synth_mod.table_rows(world.db, world.schema, table)
        for key, content in rows.items():
            vals = {str(v).lower() for vs in S.named_leaves(content).values() for v in vs}
            built.setdefault(str(key).lower(), (table, key, vals))
    _INDEX[tag] = built
    return built


def _linked(world: Any, table: str, chosen: list[str]) -> list[str]:
    """Rows of one table tied to already chosen rows, either direction."""
    want = {str(c).lower() for c in chosen}
    index = _index(world)
    other_vals: set[str] = set()
    for other in chosen:
        found = index.get(str(other).lower())
        if found is not None:
            other_vals |= found[2]
    hit = [
        key
        for (tab, key, vals) in index.values()
        if tab == table and (str(key).lower() in want or vals & want or str(key).lower() in other_vals)
    ]
    synthetic = set(world.schema.synthetic_rows or ())
    real_keys = sorted(k for k in hit if k not in synthetic)
    return real_keys + sorted(k for k in hit if k in synthetic)


def bind_unit(world: Any, sigs: dict, unit: dict, anchor: str, ident: str) -> tuple[Optional[dict], str]:
    """Bind one unit's write onto the anchor's rows; solo run proves it.

    Row-naming arguments come from rows tied to the anchor or to rows
    already chosen (the coherent rule: never from the recorded pool, never
    invented); anything no table holds stays a user fact. The body is the
    precondition oracle across rotating draws. Returns bound args or None
    plus the failure class for counting.
    """
    sig = sigs.get(str(unit["tool"]))
    properties, required = S.declared(sig)
    if not properties:
        return None, "no_sig"
    base = {n: v for n, v in unit["args"].items() if n in properties}
    for name in required:
        if name not in base or base[name] is None:
            return None, "unbound"
    id_args = [n for n in required if _is_id_arg(n)]
    tables: dict[str, Optional[str]] = {}
    for name in id_args:
        first = base[name][0] if isinstance(base[name], list) and base[name] else base[name]
        tables[name] = table_of_id(world, str(first))
    index = _index(world)
    rec: dict[str, set[str]] = {}
    for name in id_args:
        first = base[name][0] if isinstance(base[name], list) and base[name] else base[name]
        found = index.get(str(first).lower())
        rec[name] = found[2] if found is not None else set()
    allvals: set[str] = set()
    for name in id_args:
        first = base[name][0] if isinstance(base[name], list) and base[name] else base[name]
        allvals.add(str(first).lower())
    owned: set[str] = set()
    for name in id_args:
        first = base[name][0] if isinstance(base[name], list) and base[name] else base[name]
        rest = allvals - {str(first).lower()}
        if tables[name] is None or (rec[name] & rest):
            owned.add(name)
            continue
        for other in id_args:
            if other != name and str(first).lower() in rec[other]:
                owned.add(name)
                break
    id_args = [n for n in id_args if n in owned]
    rooted = {n for n in id_args if tables[n] and rows_holding(world, tables[n], anchor)}
    id_args = sorted(rooted) + [n for n in id_args if n not in rooted]
    toolkit = world.toolkit(world.fresh_db())
    body = getattr(toolkit, str(unit["tool"]), None)
    if body is None:
        return None, "crashed"
    for attempt in range(TRIES):
        args = dict(base)
        chosen: list[str] = [anchor]
        skip = False
        for name in id_args:
            table = tables[name]
            if table is None:
                continue
            cands = rows_holding(world, table, anchor) or _linked(world, table, chosen)
            if not cands:
                skip = True
                break
            shift = sample_key("arm-e-bind", f"{ident}:{attempt}:{name}", "") % len(cands)
            key = cands[shift]
            value = _row_id_of(world, table, key)
            spec = properties.get(name) or {}
            if isinstance(base[name], list):
                args[name] = [S._typed({"type": ["str"]}, value)]
            else:
                args[name] = S._typed(spec, value)
            chosen.append(key)
            chosen.append(str(value))
        if skip:
            continue
        try:
            body(**{n: v for n, v in args.items() if n in properties})
        except Exception as error:  # noqa: BLE001 - the class is the ruling
            if type(error).__name__ in CRASH_ERRORS:
                return None, "crashed"
            continue
        return args, ""
    return None, "refused"
