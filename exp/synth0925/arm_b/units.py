"""Arm B mining: read-only pass over trusted recorded Tasks.

Every table, id column and relation is read from the workdir: schema.json
(tables, id patterns), row_homes.json (which table each write's argument
names, with its basis), tool_sigs.json (write kind) and graph.json (which
reads feed which write). No table, column, tool or corpus names appear below
as literals. Outputs go to the out dir, never the repo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

R = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.claude/herdr/research-0925/rs-synth/exp/arm-b/retail")
OUT = R.parent


def load(p: Path):
    return json.loads(p.read_text())


def out_path(name: str, outdir: Path | None = None) -> Path:
    return (outdir or OUT) / name


class Binding:
    """The workdir's names, read once: tables, id patterns, row homes, edges."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        schema = load(self.workdir / "schema.json")
        self.tables: list[str] = list(schema.get("tables") or [])
        self.patterns: list[tuple[str, str, Any]] = [
            (str(k).split(".")[0], str(k).split(".")[-1], re.compile(str(v)))
            for k, v in (schema.get("id_patterns") or {}).items()]
        self.row_homes: dict = load(self.workdir / "row_homes.json")
        try:
            graph = load(self.workdir / "graph.json")
        except OSError:
            graph = {}
        self.row_edges: dict[str, list[tuple[str, int]]] = {}
        self.value_edges: dict[str, dict[str, list[tuple[str, str, int]]]] = {}
        for e in graph.get("edges") or []:
            to, frm = str(e.get("to")), str(e.get("from"))
            if e.get("kind") == "row":
                self.row_edges.setdefault(to, []).append((frm, int(e.get("weight") or 0)))
            elif e.get("kind") == "value":
                self.value_edges.setdefault(to, {}).setdefault(
                    str(e.get("arg")), []).append(
                    (frm, str(e.get("column")), int(e.get("weight") or 0)))
        for to in self.row_edges:
            self.row_edges[to].sort(key=lambda r: -r[1])
        sigs = load(self.workdir / "tool_sigs.json")
        self.write_tools = {s["name"] for s in sigs
                            if s.get("kind") == "write" or s.get("effects_observed")}
        self.read_tools = {s["name"] for s in sigs if s.get("kind") == "read"}


def trusted_ids(workdir: Path) -> tuple[list[str], dict]:
    """Tasks whose Reference is confirmed and whose Verifier passed."""
    status = load(Path(workdir) / "task_status.json")
    ids = [t for t, v in status.items()
           if v.get("reference_confirmed") and v.get("verifier_passed")]
    return sorted(ids), status


def trace_index(workdir: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    for f in (Path(workdir) / "traces").iterdir():
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("trace_id"):
            idx[d["trace_id"]] = f.name
    return idx


def task_trace_id(workdir: Path, task_id: str) -> str | None:
    try:
        d = load(Path(workdir) / "tasks" / f"{task_id}.json")
    except OSError:
        return None
    runs = d.get("run_ids") or []
    return runs[0] if runs else None


def trace_calls(workdir: Path, idx: dict[str, str], task_id: str) -> list[dict] | None:
    tid = task_trace_id(workdir, task_id)
    if tid is None or tid not in idx:
        return None
    d = load(Path(workdir) / "traces" / idx[tid])
    return d.get("tool_calls") or []


def ok_calls(calls: list[dict]) -> list[dict]:
    """Recorded calls the world accepted, in order."""
    return [c for c in calls if c.get("error") is None and c.get("has_result")]


def key_index(db: dict, tables: list[str]) -> dict[str, str]:
    """Every top-level row id the world holds, by table."""
    out: dict[str, str] = {}
    for table in tables:
        for key in (db.get(table) or {}):
            out.setdefault(str(key), table)
    return out


def nested_maps(body: Any, path: tuple = ()) -> list[tuple[tuple, list[str]]]:
    """Dict-of-dict maps inside a row, with their paths and keys.

    A map whose values are all dicts names rows of its own; plain attribute
    dicts do not qualify. Found by shape, never by column name.
    """
    out: list[tuple[tuple, list[str]]] = []
    if isinstance(body, dict):
        if path and body and all(isinstance(v, dict) for v in body.values()):
            out.append((path, [str(k) for k in body]))
        for k, v in body.items():
            out.extend(nested_maps(v, path + (str(k),)))
    elif isinstance(body, list):
        for i, v in enumerate(body):
            out.extend(nested_maps(v, path + (i,)))
    return out


def leaves_of(body: Any) -> set[str]:
    """Every scalar leaf of a value, as strings."""
    out: set[str] = set()
    stack = [body]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, str):
            out.add(v)
    return out


def top_scalars(body: Any) -> set[str]:
    """Scalar leaves of a row's top-level fields only (not nested rows)."""
    out: set[str] = set()
    if isinstance(body, dict):
        for v in body.values():
            if isinstance(v, str):
                out.add(v)
    return out


def classify(value: str, keys: dict[str, str], patterns) -> set[str]:
    """Tables a value could name: top-level keys first, id patterns after."""
    if value in keys:
        return {keys[value]}
    return {t for t, _c, rx in patterns if rx.match(value)}


def homed_primary(bind: Binding, keys: dict[str, str], tool: str,
                  args: dict) -> tuple[str, str, str] | None:
    """The row a write acts on, from row_homes.json: which table the call's
    argument names, with the observed basis. The rule names the argument."""
    homed = (bind.row_homes.get(tool) or {}).get("homed") or {}
    for table, info in homed.items():
        m = re.search(r"value of ([\w\[\]]+)", str(info.get("rule") or ""))
        arg = m.group(1) if m else None
        if arg and args.get(arg) is not None and keys.get(str(args[arg])) == table:
            return table, arg, str(args[arg])
    for table, info in homed.items():
        m = re.search(r"value of ([\w\[\]]+)", str(info.get("rule") or ""))
        arg = m.group(1) if m else None
        if arg and args.get(arg) is not None:
            return table, arg, str(args[arg])
    return None


def actor_of(bind: Binding, db: dict, keys: dict[str, str],
             table: str, row: str) -> tuple[str, str]:
    """The conversational actor owning a row: the other table named by the
    row's own top-level fields, followed as a reference. A row naming no
    other table is its own actor."""
    body = (db.get(table) or {}).get(row) or {}
    for val in sorted(top_scalars(body)):
        other = keys.get(val)
        if other is not None and other != table:
            return other, val
    return table, row


def sibling_ids(db: dict, keys: dict[str, str], actor: tuple[str, str],
                table: str, row: str) -> list[str]:
    """Other rows of the same table the actor's own record reaches."""
    body = (db.get(actor[0]) or {}).get(actor[1]) or {}
    return sorted({v for v in leaves_of(body)
                   if keys.get(v) == table and v != row})


def suppliers(bind: Binding, tool: str) -> list[str]:
    """Reads the mined graph says supply this write's rows, heaviest first."""
    return [name for name, _w in bind.row_edges.get(tool, [])]


def mine(workdir: Path = R, outdir: Path | None = None) -> dict:
    bind = Binding(workdir)
    db = load(Path(workdir) / "db.json")
    keys = key_index(db, bind.tables)
    tids, _ = trusted_ids(workdir)
    idx = trace_index(workdir)
    templates, stats = [], {"trusted": len(tids), "no_trace": 0,
                            "no_row_write": 0, "kept": 0}
    for tid in tids:
        calls = trace_calls(workdir, idx, tid)
        if not calls:
            stats["no_trace"] += 1
            continue
        ok = ok_calls(calls)
        reads = [c for c in ok if c.get("name") not in bind.write_tools]
        outs = [c for c in ok if c.get("name") in bind.write_tools]
        kept = []
        for c in outs:
            args = c.get("args") or {}
            prim = homed_primary(bind, keys, str(c.get("name")), args)
            if prim is None:
                continue  # escalation markers and pure prose name no row
            table, _arg, _row = prim
            actor = actor_of(bind, db, keys, table, _row)
            kept.append({"name": c.get("name"), "args": args,
                         "primary": {"table": table, "arg": _arg, "row": _row},
                         "actor": {"table": actor[0], "id": actor[1]},
                         "suppliers": suppliers(bind, str(c.get("name")))})
        if not kept:
            stats["no_row_write"] += 1
            continue
        first = (kept[0]["actor"]["table"], kept[0]["actor"]["id"])
        templates.append({"source_task_id": tid, "actor": kept[0]["actor"],
                          "actor_key": f"{first[0]}:{first[1]}",
                          "writes": kept,
                          "reads": [{"name": c.get("name"), "args": c.get("args") or {},
                                     "result": c.get("result")} for c in reads]})
        stats["kept"] += 1
    out = {"templates": templates, "stats": stats,
           "write_tools": sorted(bind.write_tools)}
    out_path("units.json", outdir).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else R
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out = mine(wd, od)
    s = out["stats"]
    print(f"trusted={s['trusted']} no_trace={s['no_trace']} "
          f"no_row_write={s['no_row_write']} kept={s['kept']}")
