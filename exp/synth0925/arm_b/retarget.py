"""Arm B retarget engine: rebind a Reference's write calls to other rows.

Roles, not names: the primary row comes from row_homes.json, contained rows
from the primary row's own nested structure, replacements from the parent
group of the mapped element, methods from the actor row's map at the same
path, state-like columns from closed vocabularies in the db. Execution
through the bodies is the precondition oracle. Table, column and tool names
are read from the workdir's schema, row homes and graph; none appear here.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import units
from units import Binding, actor_of, key_index, leaves_of, nested_maps, sibling_ids

from kullback import synthesise as S
from kullback.sampling import sample_key

SALT = "synth0925-b"


def state_like_columns(db: dict, table: str) -> list[str]:
    """String columns with a small closed vocabulary: the row's state."""
    rows = list((db.get(table) or {}).values())
    if not rows:
        return []
    out = []
    first = rows[0] if isinstance(rows[0], dict) else {}
    for k in first:
        if not all(isinstance(r.get(k), str) for r in rows if isinstance(r, dict)):
            continue
        vals = {r.get(k) for r in rows if isinstance(r, dict)}
        if 1 < len(vals) <= 8 and all(len(str(v)) <= 32 for v in vals):
            out.append(k)
    return out


def at_path(body: Any, path: tuple) -> Any:
    node = body
    for step in path:
        if isinstance(node, dict) and step in node:
            node = node[step]
        elif isinstance(node, list) and isinstance(step, int) and step < len(node):
            node = node[step]
        else:
            return None
    return node


def locate(body: Any, value: str, path: tuple = ()) -> list[tuple[tuple, Any, Any]]:
    """Where a value lives in a row: (path, container, element-or-key)."""
    found: list[tuple[tuple, Any, Any]] = []
    if isinstance(body, dict):
        for k, v in body.items():
            if isinstance(v, str) and v == value and path:
                found.append((path, body, k))
            else:
                found.extend(locate(v, value, path + (str(k),)))
    elif isinstance(body, list):
        for i, v in enumerate(body):
            if isinstance(v, str) and v == value:
                found.append((path, body, i))
            else:
                found.extend(locate(v, value, path + (i,)))
    return found



def shared_scalars(a: Any, b: Any) -> int:
    """Scalar values two elements hold in common: the similarity vote."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0
    sa = {str(v) for v in a.values() if isinstance(v, (str, int, float))}
    sb = {str(v) for v in b.values() if isinstance(v, (str, int, float))}
    return len(sa & sb)


def parent_groups(bind: Binding, db: dict, keys: dict[str, str], value: str,
                  exclude: tuple[str, str] | None = None) -> list[tuple]:
    """Parent groups holding a value as a key: top-level rows and nested
    maps, outside the excluded row's own subtree. A group is (table, row)
    or (table, row, path)."""
    groups: list[tuple] = []
    if value in keys and (exclude is None or keys[value] != exclude[0] or value != exclude[1]):
        groups.append((keys[value], value))
    for table in bind.tables:
        for row, body in (db.get(table) or {}).items():
            if exclude is not None and table == exclude[0] and str(row) == exclude[1]:
                continue
            if not isinstance(body, dict):
                continue
            for path, names in nested_maps(body):
                if value in names:
                    groups.append((table, str(row), path))
    return groups


def element_key(elem: Any, value: str | None = None) -> tuple[str, str] | None:
    """The id-like field of a collection element: the field holding the
    sought value where one is given, else the first field shaped like a row
    id. Field names come from the data, never from literals."""
    if isinstance(elem, dict):
        if value is not None:
            for k in sorted(elem):
                if str(elem[k]) == value:
                    return str(k), value
        for k in sorted(elem):
            v = elem[k]
            if isinstance(v, str) and re.match(r"^#?[A-Za-z0-9 _-]{3,64}$", v):
                return str(k), v
    return None


def run_call(toolkit, name: str, args: dict) -> tuple[bool, str, object]:
    body = getattr(toolkit, name, None)
    if body is None:
        return False, "missing_tool", None
    try:
        return True, "", body(**copy.deepcopy(args))
    except Exception as error:  # noqa: BLE001 - the class is the ruling
        from kullback.gates.tool_runs import CRASH_ERRORS
        if type(error).__name__ in CRASH_ERRORS:
            return False, "crash:" + type(error).__name__, None
        return False, type(error).__name__ + ":" + str(error)[:120], None


def _map_element(old_elems: list, new_elems: list, value: str,
                 taken: set[str]) -> str | None:
    """A value held in a field of an old collection element, re-homed to
    the matching new element: most shared scalars first, same position
    after, first untaken last. Field names come from the data."""
    old_elem: Any = {}
    pos, field = 0, None
    for i, el in enumerate(old_elems):
        hit = element_key(el, value)
        if hit is not None:
            old_elem, pos, field = el, i, hit[0]
            break
    if field is None:
        return None
    best, best_score = None, -1
    for el in new_elems:
        v = el.get(field) if isinstance(el, dict) else None
        if not isinstance(v, str) or v in taken:
            continue
        score = shared_scalars(old_elem, el)
        if score > best_score:
            best, best_score = v, score
    if best is not None and best_score > 0:
        return best
    for i, el in enumerate(new_elems):
        v = el.get(field) if isinstance(el, dict) else None
        if isinstance(v, str) and v not in taken and i == pos:
            return v
    for el in new_elems:
        v = el.get(field) if isinstance(el, dict) else None
        if isinstance(v, str) and v not in taken:
            return v
    return None

def rehome_value(bind: Binding, db: dict, keys: dict[str, str],
                 old_body: dict, new_body: dict,
                 old_actor: tuple[str, str], new_actor: tuple[str, str],
                 old_row: tuple[str, str], new_row: str,
                 value: str, partner: str | None,
                 taken: set[str], ident: str) -> str | None:
    """One row-ref value re-homed to the new row. Contained values follow
    their collection (similarity first, position after); actor refs follow
    the actor; outside values follow their parent group (catalog) or the
    actor's map at the same path (methods)."""
    old_actor_body = (db.get(old_actor[0]) or {}).get(old_actor[1]) or {}
    actor_map_paths = [p for p, names in nested_maps(old_actor_body)
                       if value in names]
    if value == old_actor[1]:
        return new_actor[1]
    if actor_map_paths:
        # a method of the old actor: the new actor's map at the same path,
        # preferring one the new primary row already names
        actor_body = (db.get(new_actor[0]) or {}).get(new_actor[1]) or {}
        for path in actor_map_paths:
            node = at_path(actor_body, path)
            pool = sorted(node.keys()) if isinstance(node, dict) else []
            prim_leaves = leaves_of(new_body)
            same = [k for k in pool if str(k) in prim_leaves and str(k) not in taken]
            rest = [k for k in pool if str(k) not in taken and k not in same]
            if same or rest:
                ordered = same + rest
                return str(ordered[sample_key("arm-b", ident, SALT) % len(ordered)])
        return None
    if value in leaves_of(old_body):
        locs = locate(old_body, value)
        if not locs:
            return None
        locs.sort(key=lambda loc: 0 if (isinstance(loc[1], list) and loc[1]
                                        and isinstance(loc[1][0], dict)) else 1)
        path, container, _elem = locs[0]
        if isinstance(container, list):
            cands = at_path(new_body, path)
            if not isinstance(cands, list) or not cands:
                return None
            if all(not isinstance(el, dict) for el in list(container) + list(cands)):
                # a plain id list: same position first, then first untaken
                try:
                    spos = list(container).index(value)
                except ValueError:
                    spos = 0
                if spos < len(cands) and isinstance(cands[spos], str) \
                        and cands[spos] not in taken:
                    return cands[spos]
                for cand in cands:
                    if isinstance(cand, str) and cand not in taken:
                        return cand
                return None
            return _map_element(list(container), list(cands), value, taken)
        if isinstance(container, dict):
            node = at_path(new_body, path)
            if isinstance(node, dict) and value in node:
                return value
            # the value sits in a field of an old collection element: map
            # through the parent collection the way bare elements map
            if path:
                par_old = at_path(old_body, path[:-1])
                par_new = at_path(new_body, path[:-1])
                if isinstance(par_old, list) and isinstance(par_new, list) \
                        and par_new and any(isinstance(el, dict)
                                            for el in par_old):
                    hit = _map_element(par_old, par_new, value, taken)
                    if hit is not None:
                        return hit
            return None
        return None
    groups = parent_groups(bind, db, keys, value, exclude=old_row)
    nested = [g for g in groups if len(g) > 2]
    if not nested:
        return None
    home = nested[0]
    want = (home[0], home[2])
    # the nested maps of rows the mapped partner element names (its parent
    # references), analogous to the old value's home
    target_groups: list[tuple] = []
    if partner:
        for _path, container, _e in locate(new_body, partner):
            elem = None
            if isinstance(container, list):
                for el in container:
                    hit = element_key(el, partner)
                    if hit is not None:
                        elem = el
                        break
            elif isinstance(container, dict):
                elem = container
            for s in (leaves_of(elem) if isinstance(elem, dict) else set()):
                parent_table = keys.get(s)
                if parent_table is not None:
                    parent_body = (db.get(parent_table) or {}).get(s) or {}
                    for p, _names in nested_maps(parent_body):
                        target_groups.append((parent_table, s, p))
        target_groups.extend(
            g for g in parent_groups(bind, db, keys, partner,
                                     exclude=(old_row[0], new_row)))
    for g in target_groups:
        if (g[0], g[2] if len(g) > 2 else ()) == want and len(g) > 2:
            node = at_path((db.get(g[0]) or {}).get(g[1]) or {}, g[2])
            cands = [str(k) for k in (node.keys() if isinstance(node, dict) else [])
                     if str(k) != partner and str(k) not in taken]
            if cands:
                return cands[sample_key("arm-b", ident, SALT) % len(cands)]
    for g in target_groups:
        if len(g) > 2:
            node = at_path((db.get(g[0]) or {}).get(g[1]) or {}, g[2])
            cands = [str(k) for k in (node.keys() if isinstance(node, dict) else [])
                     if str(k) != partner and str(k) not in taken]
            if cands:
                return cands[sample_key("arm-b", ident, SALT) % len(cands)]
    if partner:
        sibs = sibling_rows(bind, db, keys, new_body, partner, value,
                            taken, exclude=(old_row[0], new_row))
        if sibs:
            return sibs[sample_key("arm-b", ident, SALT) % len(sibs)]
    return None


def _has_false_flag(body: Any) -> bool:
    """An explicit negative flag anywhere in the row: shape, not name."""
    stack = [body]
    while stack:
        v = stack.pop()
        if v is False:
            return True
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return False


def sibling_rows(bind: Binding, db: dict, keys: dict[str, str],
                 new_body: dict, partner: str, value: str,
                 taken: set[str],
                 exclude: tuple[str, str] | None = None) -> list[str]:
    """Rows sharing a parent reference with the mapped partner element:
    id-shaped leaves the partner names, held by rows of the tables the
    sought value's shape allows. Rows carrying an explicit negative flag
    sort last; the bodies rule on the pick."""
    pleaves: set[str] = set()
    if partner in keys:
        pleaves |= leaves_of((db.get(keys[partner]) or {}).get(partner) or {})
    for _path, container, _e in locate(new_body, partner):
        if isinstance(container, list):
            for el in container:
                if element_key(el, partner) is not None and isinstance(el, dict):
                    pleaves |= leaves_of(el)
                    break
        elif isinstance(container, dict):
            pleaves |= leaves_of(container)
    refs = {s for s in pleaves
            if s != partner and units.classify(s, keys, bind.patterns)}
    if not refs:
        return []
    want_tables = units.classify(value, keys, bind.patterns)
    cands: list[str] = []
    for table in bind.tables:
        if table not in want_tables:
            continue
        for row, body in (db.get(table) or {}).items():
            if str(row) in taken or str(row) == partner:
                continue
            if exclude is not None and table == exclude[0] \
                    and str(row) == exclude[1]:
                continue
            if not isinstance(body, dict):
                continue
            if refs & leaves_of(body):
                cands.append(str(row))
    cands.sort()
    live = [c for c in cands
            if not _has_false_flag((db.get(keys[c]) or {}).get(c))]
    return live or cands


def rebind_write(bind: Binding, db: dict, keys: dict[str, str], write: dict,
                 old_actor: tuple[str, str], new_primary: str,
                 ident: str) -> dict | None:
    """The reference write with its primary row swapped and dependents
    re-homed by role. Anything with no home stays as is; the oracle rules."""
    old = write["primary"]
    old_row = (old["table"], str(old["row"]))
    old_body = (db.get(old["table"]) or {}).get(str(old["row"])) or {}
    new_body = (db.get(old["table"]) or {}).get(new_primary) or {}
    new_actor = actor_of(bind, db, keys, old["table"], new_primary)
    args = copy.deepcopy(write["args"])
    args[old["arg"]] = new_primary
    taken = {new_primary}
    old_leaves = leaves_of(old_body)
    # pass one: contained values, recording each mapped element by (arg, pos)
    mapped: dict[tuple[str, int], str] = {}
    done: set[str] = set()
    for arg, val in list(args.items()):
        if arg == old["arg"]:
            continue
        vals = [str(x) for x in (val if isinstance(val, list) else [val])]
        if not any(v in old_leaves and _is_ref(bind, keys, db, v) for v in vals):
            continue
        out = []
        for pos, v in enumerate(vals):
            if v not in old_leaves or not _is_ref(bind, keys, db, v):
                out.append(args[arg][pos] if isinstance(val, list) else val)
                continue
            nv = rehome_value(bind, db, keys, old_body, new_body, old_actor,
                              new_actor, old_row, new_primary, v, None,
                              taken, f"{ident}:{arg}:{pos}")
            if nv is None:
                return None
            taken.add(nv)
            out.append(nv)
            mapped[(arg, pos)] = nv
        args[arg] = out if isinstance(val, list) else out[0]
        done.add(arg)
    list_args = [a for a, v in args.items()
                 if isinstance(v, list) and a != old["arg"]]
    peers: dict[str, str] = {}
    for i in range(len(list_args) - 1):
        a, b = list_args[i], list_args[i + 1]
        if len(args[a]) == len(args[b]):
            peers.setdefault(a, b)
            peers.setdefault(b, a)
    # pass two: outside values (catalog, methods) with their mapped peer
    for arg, val in list(args.items()):
        if arg == old["arg"] or arg in done:
            continue
        vals = [str(x) for x in (val if isinstance(val, list) else [val])]
        if any(v in old_leaves and _is_ref(bind, keys, db, v) for v in vals):
            continue  # done in pass one
        if not any(_is_ref(bind, keys, db, v) for v in vals):
            continue  # free values stay as the recording showed them
        out = []
        for pos, v in enumerate(vals):
            if not _is_ref(bind, keys, db, v):
                out.append(args[arg][pos] if isinstance(val, list) else val)
                continue
            peer = peers.get(arg)
            partner = mapped.get((peer, pos)) if peer else None
            if partner is None and mapped:
                partner = next(iter(mapped.values()))
            nv = rehome_value(bind, db, keys, old_body, new_body, old_actor,
                              new_actor, old_row, new_primary, v, partner,
                              taken, f"{ident}:{arg}:{pos}")
            if nv is None:
                return None
            taken.add(nv)
            out.append(nv)
        args[arg] = out if isinstance(val, list) else out[0]
    return args


def _is_ref(bind: Binding, keys: dict[str, str], db: dict, value: str) -> bool:
    if value in keys:
        return True
    for _t, _c, rx in bind.patterns:
        if rx.match(value):
            return True
    for table in bind.tables:
        for body in (db.get(table) or {}).values():
            if not isinstance(body, dict):
                continue
            for _path, names in nested_maps(body):
                if value in names:
                    return True
    return False


def retarget_do(bind: Binding, world: S.World, template: dict, ident: str,
                db: dict | None = None) -> dict | None:
    """Rebind one template's writes to another row the reads reached.

    Same-actor siblings first, other actors after; rows matching the old
    row's closed-vocabulary state go first; the bodies rule. Returns the
    concrete unit or None.
    """
    db = db if db is not None else world.fresh_db()
    keys = key_index(db, bind.tables)
    writes = template["writes"]
    first = writes[0]
    table = first["primary"]["table"]
    old_row = str(first["primary"]["row"])
    actor = (template["actor"]["table"], template["actor"]["id"])
    old_body = (db.get(table) or {}).get(old_row) or {}
    states = state_like_columns(db, table)
    held = set((db.get(table) or {}).keys())
    scopes: list[tuple[str, list[str]]] = []
    if table == actor[0]:
        pool = [r for r in sorted(held) if r != old_row]
        scopes.append(("cross-actor", pool))
    else:
        same = [c for c in sibling_ids(db, keys, actor, table, old_row) if c in held]
        scopes.append(("same-actor", same))
        other = [r for r in sorted(held)
                 if r != old_row and actor_of(bind, db, keys, table, r) != actor]
        scopes.append(("cross-actor", other))
    reached = _reached(template)

    def state_key(c: str) -> int:
        return sum(1 for s in states
                   if ((db.get(table) or {}).get(c) or {}).get(s) == old_body.get(s))
    for scope_name, pool in scopes:
        if table != actor[0] and scope_name == "same-actor" and reached:
            pool = [c for c in pool if c in reached] or pool
        pool = sorted(pool, key=state_key, reverse=True)
        for n, cand in enumerate(pool[:12]):
            args_list, good = [], True
            for w in writes:
                if str(w["primary"]["row"]) != old_row:
                    good = False
                    break
                a = rebind_write(bind, db, keys, w, actor, cand, f"{ident}:{n}")
                if a is None:
                    good = False
                    break
                args_list.append(a)
            if not good:
                continue
            probe_db = copy.deepcopy(db)
            probe_kit = world.toolkit(probe_db)
            ok_all = True
            for w, a in zip(writes, args_list, strict=True):
                ok, _err, _ans = run_call(probe_kit, w["name"], a)
                if not ok:
                    ok_all = False
                    break
            if ok_all:
                new_actor = actor_of(bind, db, keys, table, cand)
                return {"kind": "do", "source_task_id": template["source_task_id"],
                        "scope": ("same-actor" if new_actor == actor else "cross-actor"),
                        "actor": {"table": new_actor[0], "id": new_actor[1]},
                        "calls": [{"name": w["name"], "args": a}
                                  for w, a in zip(writes, args_list, strict=True)],
                        "rows": [cand],
                        "edge_covered": _edge_covered(bind, template)}
        # next scope
    return None


def _reached(template: dict) -> set[str]:
    seen: set[str] = set()
    for v in (template.get("read_results") or []):
        seen |= units.leaves_of(v)
    return seen


def _edge_covered(bind: Binding, template: dict) -> bool:
    """Some reference read is one the graph says supplies a unit write."""
    read_names = {r.get("name") for r in template.get("reads") or []}
    for w in template["writes"]:
        sup = set(bind.row_edges.get(str(w.get("name")), []) and
                  [n for n, _w in bind.row_edges[str(w.get("name"))]])
        if sup & read_names:
            return True
    return False


def try_call_fresh(world: S.World, name: str, args: dict) -> tuple[bool, str]:
    db = world.fresh_db()
    ok, err, _ = run_call(world.toolkit(db), name, args)
    return ok, err


def decline_candidates(bind: Binding, world: S.World, template: dict,
                       ident: str) -> list[dict]:
    """Rows where the template's first write must not happen: same-actor
    rows in another state first, then dangling dependents from other rows.
    Only calls the bodies refuse are kept, with the error class."""
    db = world.fresh_db()
    keys = key_index(db, bind.tables)
    first = template["writes"][0]
    table = first["primary"]["table"]
    old_row = str(first["primary"]["row"])
    actor = (template["actor"]["table"], template["actor"]["id"])
    states = state_like_columns(db, table)
    old_body = (db.get(table) or {}).get(old_row) or {}
    out: list[dict] = []
    if table == actor[0]:
        sibs = [r for r in sorted((db.get(table) or {}).keys()) if r != old_row][:6]
    else:
        held = set((db.get(table) or {}).keys())
        sibs = [c for c in sibling_ids(db, keys, actor, table, old_row) if c in held]
        sibs = sorted(sibs, key=lambda c: sum(
            1 for s in states
            if ((db.get(table) or {}).get(c) or {}).get(s) == old_body.get(s)))
    for cand in sibs[:6]:
        a = rebind_write(bind, db, keys, first, actor, cand, f"{ident}:ws:{cand}")
        if a is None:
            continue
        ok, err = try_call_fresh(world, first["name"], a)
        if not ok and not err.startswith("crash:") and not err.startswith("missing"):
            out.append({"sort": "body-refused", "call": {"name": first["name"], "args": a},
                        "rows": [cand], "error": err,
                        "source_task_id": template["source_task_id"],
                        "actor": {"table": actor[0], "id": actor[1]}})
    others = [r for r in sorted((db.get(table) or {}).keys()) if r != old_row][:4]
    for other in others:
        obody = (db.get(table) or {}).get(other) or {}
        vals = [v for v in leaves_of(obody) if v in keys or _is_ref(bind, keys, db, v)]
        if not vals:
            continue
        a = copy.deepcopy(first["args"])
        for arg, val in list(a.items()):
            if arg == first["primary"]["arg"]:
                continue
            if isinstance(val, list) and val:
                a[arg] = [vals[0]] * len(val)
            elif isinstance(val, str) and _is_ref(bind, keys, db, val):
                a[arg] = vals[0]
        ok, err = try_call_fresh(world, first["name"], a)
        if not ok and not err.startswith("crash:") and not err.startswith("missing"):
            out.append({"sort": "body-refused", "call": {"name": first["name"], "args": a},
                        "rows": [old_row], "error": err,
                        "source_task_id": template["source_task_id"],
                        "actor": {"table": actor[0], "id": actor[1]}})
    return out


def policy_predicates(workdir: Path) -> tuple[list[dict], str]:
    """The compiled policy the Environment holds, if it holds any."""
    import json
    for name in ("constraints.json", "policy.json"):
        p = Path(workdir) / name
        if not p.exists():
            continue
        try:
            body = json.loads(p.read_text())
        except ValueError:
            continue
        # the compiled entries live in the list of constraint records,
        # found by shape (records carrying compiled source), not by key
        cands = body if isinstance(body, list) else [
            v for v in (body.values() if isinstance(body, dict) else [])
            if isinstance(v, list)]
        records = next(
            (c for c in cands
             if any(isinstance(r, dict) and r.get("predicate_src") for r in c)),
            [])
        preds = [c for c in records
                 if isinstance(c, dict) and c.get("compiled") and c.get("predicate_src")]
        if preds:
            return preds, f"{name}: {len(preds)} compiled"
    return [], ("no compiled predicate in the workdir: constraints.json and "
                "policy.json are absent and constraints_check.json holds "
                "empty constraints")
