"""Coherent write binding for arm C: row ids come from the world, never the model.

The synthesiser proposes blueprints with literal arguments. A literal row id
is a guess: the model transcribes it from truncated read answers, and any
miss burns a revision. Here every argument that names a row (`*_id`,
`*_ids`) is rebound from the world before the write runs:

1. seen first: a value the earlier answers of this same run carry under that
   name (latest answer wins);
2. inserted reads: the read the mined graph says supplies the argument, run
   here so the id genuinely answers in this run (up to MAX_INSERT per write);
3. world draw: one row of the rebuilt world carrying the owed names, else the
   recorded value pool.

A refused write is resampled with fresh draws on a replayed prefix up to
RETRIES times. When distinct host rows keep refusing with one class, the
verdict is no_host_row (the scenario asks for a state the world cannot
host), not another revision. No names of any corpus, tool, task or model
appear here; every name is read off the workdir at run time.
"""
from __future__ import annotations

from typing import Any

RETRIES = 8
MAX_INSERT = 2


ANCHOR_SALT = "arm-c-anchor"


def _askable_leaves(row: Any) -> list:
    """Scalar leaves a customer states: shallow fields, never row ids or lists.

    A row id, a list of references, or a deeply nested record value is
    something the world knows, not something the user says. Name parts and
    address parts live one level down; anything deeper is a record fact.
    """
    facts: list = []

    def walk(node: Any, parts: list) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                walk(val, parts + [str(key)])
        elif isinstance(node, list):
            return
        elif isinstance(node, (str, int, float)) and not isinstance(node, bool):
            field = parts[-1] if parts else ""
            if 1 <= len(parts) <= 2 and field not in ("id", "ids") \
                    and not _is_row_arg(field) and str(node):
                if (field, node) not in [(f["field"], f["value"]) for f in facts]:
                    facts.append({"field": field, "value": node})

    walk(row, [])
    return facts


def draw_anchor(world: Any, seed_tag: str) -> dict | None:
    """A real row the scenario user is, drawn with a keyed seed, and its facts.

    The anchor table is discovered, not named: among the read tools keyed by
    plain facts (no row id required), the table whose rows cover the most key
    names wins. Returns {facts, anchor_id, table} or None.
    """
    from kullback import synthesise as S
    from kullback.sampling import sample_key

    read_names = [s.name for s in world.sigs or ()
                  if getattr(s, "kind", "") == "read"]
    key_names: set = set()
    for sig in world.sigs or ():
        if getattr(sig, "name", "") not in read_names:
            continue
        schema = getattr(sig, "args_schema", None) or {}
        req = [str(r) for r in (schema.get("required") or [])]
        if req and not any(_is_row_arg(r) for r in req):
            key_names.update(req)
    if not key_names:
        return None
    tables = sorted(getattr(world.schema, "tables", None) or ())
    best = (0, "")
    for table in tables:
        keys = world.rows_of(table)
        if not keys:
            continue
        try:
            rows = S.synth.table_rows(world.db, world.schema, table)
        except Exception:
            continue
        sample = S.named_leaves(rows.get(keys[0]) or {})
        covered = sum(1 for name in key_names if name in sample)
        if covered > best[0]:
            best = (covered, table)
    if not best[0]:
        return None
    keys = world.rows_of(best[1])
    anchor_id = keys[sample_key(ANCHOR_SALT, str(seed_tag), "") % len(keys)]
    try:
        rows = S.synth.table_rows(world.db, world.schema, best[1])
    except Exception:
        return None
    facts = _askable_leaves(rows.get(anchor_id))
    if not facts:
        return None
    return {"facts": facts, "anchor_id": anchor_id, "table": best[1]}


def _props(world: Any, name: str) -> dict:
    for sig in world.sigs or ():
        if getattr(sig, "name", "") == name:
            schema = getattr(sig, "args_schema", None) or {}
            return dict(schema.get("properties") or {})
    return {}


def _required(world: Any, name: str) -> list:
    for sig in world.sigs or ():
        if getattr(sig, "name", "") == name:
            schema = getattr(sig, "args_schema", None) or {}
            req = schema.get("required") or []
            return [str(r) for r in req]
    return []


def _is_row_arg(arg: str) -> bool:
    return str(arg).endswith("_id") or str(arg).endswith("_ids")


def _seen_by_name(answers: list) -> dict:
    """Named leaves of earlier answers, latest answer first (the fix2 rule)."""
    from kullback import synthesise as S

    seen: dict = {}
    for answer in reversed(answers):
        for name, vals in S.named_leaves(answer).items():
            seen.setdefault(name, [v for v in vals if v not in seen.get(name, [])])
    return seen


def _pool_values(observed: dict, tool: str, arg: str) -> list:
    """Recorded values for one argument, with the list-name fix (a mined
    `name[]` pool also serves `name`)."""
    rec = (observed.get(tool) or {})
    vals = (rec.get(arg) or {}).get("values") or []
    if not vals:
        vals = (rec.get(arg + "[]") or {}).get("values") or []
    return [str(v) for v in vals]


def _supplier(graph: dict, write: str, arg: str, writes: set, used: set) -> dict | None:
    """The read the recordings most often fed this argument from."""
    best = None
    for edge in graph.get("edges") or ():
        if str(edge.get("to")) == write \
                and str(edge.get("arg")).split(".")[0].replace("[]", "") == arg \
                and edge.get("from") not in writes \
                and edge.get("from") != write and edge.get("from") not in used:
            if best is None or float(edge.get("weight") or 0) > float(best.get("weight") or 0):
                best = edge
    return best


def _draw(seen: dict, row: dict, pool: list, salt: str, arg: str) -> str | None:
    """One value for a row argument: seen, then row, then pool, salted choice."""
    from kullback.sampling import sample_key

    cands: list = []
    for source in (seen.get(arg) or [], row.get(arg) or [], pool):
        for val in source:
            text = str(val)
            if text and text not in cands:
                cands.append(text)
    if not cands:
        return None
    return cands[sample_key("arm-c-row", f"{salt}:{arg}", "") % len(cands)]


def _bind_inserted_read(world: Any, toolkit: Any, read: str, bindings: list,
                        answers: list, observed: dict, ident: str) -> tuple | None:
    """Run one inserted supplier read, bound from earlier answers then the world."""
    from kullback import graph as G
    from kullback import synthesise as S

    args: dict = {}
    for binding in bindings:
        found = G.values_at(answers[int(binding["from"])], str(binding["column"]))
        if found:
            from kullback.sampling import sample_key
            S._set_path(args, str(binding["arg"]),
                        found[sample_key("arm-c-edge", f"{ident}:{binding['arg']}", "")
                              % len(found)])
    props = _props(world, read)
    seen = _seen_by_name(answers)
    owed = [n for n in _required(world, read) if n not in args]
    row = S._row_source(world, [n for n in owed if n not in seen], ident) \
        if owed else {}
    for name in owed:
        pool = seen.get(name) or row.get(name) or _pool_values(observed, read, name)
        if not pool:
            return None
        from kullback.sampling import sample_key
        args[name] = S._typed(props.get(name) or {},
                              pool[sample_key("arm-c-arg", f"{ident}:{read}:{name}", "")
                                   % len(pool)])
    body = getattr(toolkit, read, None)
    if body is None:
        return None
    final = {n: v for n, v in args.items() if n in props} if props else args
    try:
        return body(**final), final
    except Exception:
        return None


def bind_write(world: Any, toolkit: Any, graph: dict, observed: dict, writes: set,
               name: str, model_args: dict, answers: list, read_names: list,
               ident: str, stats: dict) -> tuple:
    """Write arguments with every row id rebound from the world.

    Non-row arguments stay exactly as the synthesiser wrote them. Returns
    (args, new_answers, new_calls): the full argument map or None when a row
    id cannot be served at all, plus the inserted supplier reads that served
    it, so the caller replays a whole run. Supplied lists are never mutated.
    """
    from kullback import synthesise as S

    props = _props(world, name)
    args = dict(model_args or {})
    seen = _seen_by_name(answers)
    owed = [a for a in props if _is_row_arg(a)]
    if not owed:
        return args, [], []
    work_answers = list(answers)
    work_names = list(read_names)
    new_answers: list = []
    new_calls: list = []
    row = S._row_source(world, [a for a in owed if a not in seen], ident)
    used: set = set()
    for _ in range(MAX_INSERT + 1):
        missing = [a for a in owed
                   if not (seen.get(a) or row.get(a) or _pool_values(observed, name, a))]
        if not missing:
            break
        edge = _supplier(graph, name, missing[0], writes, used)
        if edge is None:
            break
        used.add(edge["from"])
        step_bindings = [{"arg": e2["arg"], "column": e2["column"], "from": i}
                         for e2 in graph.get("edges") or ()
                         if e2.get("to") == edge["from"] and e2.get("kind") == "value"
                         for i, nm in enumerate(work_names) if nm == e2["from"]]
        found = _bind_inserted_read(world, toolkit, str(edge["from"]), step_bindings,
                                    work_answers, observed, f"{ident}:ins{len(used)}")
        if found is None:
            break
        answer, rargs = found
        work_answers.append(answer)
        new_answers.append(answer)
        new_calls.append({"id": f"{ident}-ins{len(used)}", "name": str(edge["from"]),
                          "args": rargs, "requestor": "assistant", "turn": -1,
                          "result": answer, "error": None})
        work_names.append(str(edge["from"]))
        for leaf, vals in S.named_leaves(answer).items():
            seen.setdefault(leaf, [v for v in vals if v not in seen.get(leaf, [])])
        stats["inserted"] = stats.get("inserted", 0) + 1
        if owed[0] not in row:
            row = S._row_source(world, [a for a in owed if a not in seen], ident)
    final: dict = {}
    for arg in owed:
        if arg in args and str(args[arg]) in (seen.get(arg) or []):
            final[arg] = args[arg]
        val = _draw(seen, row, _pool_values(observed, name, arg), ident, arg)
        if val is None:
            stats["unbound"] = stats.get("unbound", 0) + 1
            return None, [], []
        final[arg] = S._typed(props.get(arg) or {}, val)
        stats["rebound"] = stats.get("rebound", 0) + 1
    final.update({k: v for k, v in args.items() if k not in final})
    return ({k: v for k, v in final.items() if (not props or k in props)},
            new_answers, new_calls)


def run_blueprint(world: Any, reads: list, writes: list, ident: str,
                  observed: dict | None = None, graph: dict | None = None) -> tuple:
    """Run blueprint reads then writes on a fresh world. Returns (calls, why, info).

    why is "" on success, else read_failed:<Class>, refused:<Class>,
    no_host_row:<Class>, crashed:<Class>, unknown_tool or unbound:<arg>.
    info carries the failed tool, its index, and binding counts. Row ids
    always come from the world; a refusal resamples them up to RETRIES times.
    """
    from kullback import synthesise as S
    from kullback.gates.tool_runs import CRASH_ERRORS

    if observed is None:
        try:
            observed = (S.load_graph(world.workdir).get("args") or {})
        except Exception:
            observed = {}
    if graph is None:
        try:
            graph = S.load_graph(world.workdir)
        except Exception:
            graph = {}
    writes_set = {str(w.get("name") or "") for w in (writes or [])}
    try:
        writes_set |= set(world.write_tools or ())
    except Exception:
        pass
    db = world.fresh_db()
    toolkit = world.toolkit(db)
    answers: list = []
    calls: list = []
    stats: dict = {}
    seq = [{"kind": "read", **r} for r in (reads or [])]
    seq += [{"kind": "write", **w} for w in (writes or [])]
    ran_any_write = False
    index = 0
    for item in seq:
        name = str(item.get("name") or "")
        body = getattr(toolkit, name, None)
        if body is None:
            return calls, "unknown_tool", {"tool": name, "index": index, "stats": stats}
        props = _props(world, name)
        if item["kind"] == "read":
            args = dict(item.get("args") or {})
            for req in _required(world, name):
                if req not in args:
                    seen = _seen_by_name(answers)
                    row = S._row_source(world, [req], ident)
                    pool = seen.get(req) or row.get(req) or _pool_values(
                        observed, name, req)
                    if pool:
                        args[req] = pool[0]
            try:
                answer = body(**args)
            except Exception as error:
                cls = type(error).__name__
                if cls in CRASH_ERRORS:
                    return calls, f"crashed:{cls}", {"tool": name, "index": index,
                                                     "stats": stats}
                return calls, f"read_failed:{cls}", {"tool": name, "index": index,
                                                     "stats": stats}
            answers.append(answer)
            calls.append({"id": f"{ident}-{index}", "name": name, "args": args,
                          "requestor": "assistant", "turn": index,
                          "result": answer, "error": None})
            index += 1
            continue
        has_row = any(_is_row_arg(a) for a in props)
        tries = RETRIES if has_row else 1
        classes: list = []
        tried: set = set()
        ran = False
        for attempt in range(tries):
            sub = ident if attempt == 0 else f"{ident}:t{attempt}"
            if attempt > 0:
                toolkit = world.toolkit(world.fresh_db())
                for done in calls:
                    getattr(toolkit, done["name"])(**done["args"])
            bound, extra_answers, extra_calls = bind_write(
                world, toolkit, graph, observed, writes_set, name,
                dict(item.get("args") or {}), answers,
                [c["name"] for c in calls], sub, stats)
            if bound is None:
                missing = [a for a in props if _is_row_arg(a)]
                return calls, f"unbound:{missing[0] if missing else 'id'}", {
                    "tool": name, "index": index, "stats": stats}
            tried.add(tuple(sorted((k, str(v)) for k, v in bound.items()
                                   if _is_row_arg(k))))
            try:
                answer = getattr(toolkit, name)(**bound)
            except Exception as error:
                cls = type(error).__name__
                if cls in CRASH_ERRORS:
                    return calls, f"crashed:{cls}", {"tool": name, "index": index,
                                                     "stats": stats}
                classes.append(cls)
                continue
            answers.extend(extra_answers)
            calls.extend(extra_calls)
            answers.append(answer)
            calls.append({"id": f"{ident}-{index}", "name": name, "args": bound,
                          "requestor": "assistant", "turn": index,
                          "result": answer, "error": None})
            ran = True
            break
        if ran:
            ran_any_write = True
            index += 1
            continue
        top = max(set(classes), key=classes.count) if classes else "refused"
        if len(tried) >= 3 and classes and all(c == top for c in classes):
            return calls, f"no_host_row:{top}", {"tool": name, "index": index,
                                                 "stats": stats, "tried": len(tried)}
        return calls, f"refused:{top}", {"tool": name, "index": index,
                                         "stats": stats, "tried": len(tried)}
    if not ran_any_write:
        return calls, "no_writes", {"tool": None, "index": index, "stats": stats}
    return calls, "", {"tool": None, "index": index, "stats": stats}
