"""Run the offline slice on one tau2 domain and compare it against that domain's real tools and database.

The cross-domain check of 2026-08-28 (docs/cross-domain-check.md) was run from scratch scripts that
lived outside the repository, and two bugs in them had to be found before the numbers were honest.
This is the same check as a committed script, so a rerun after a fix is one command and the numbers
move for a reason that is in git.

It reads only what is already vendored or already downloaded: the raw traces under `data/raw/`, the
domain's real `tools.py` and `db.json` under `vendor/tau2-bench/`. It calls no model, so every stage
it runs is an offline one (ingest, mine, cluster, canon rules, Starting state).

    uv run python scripts/xdomain_check.py retail airline telecom
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harness.builder import cluster, compile_env, ingest, mine  # noqa: E402
from harness.shared import canon  # noqa: E402
from harness.shared.records import Trace  # noqa: E402

VENDOR = REPO / "vendor" / "tau2-bench"
RAW = REPO / "data" / "raw"
DEFAULT_WORK = REPO / "data" / "work"

# One claude-3-7-sonnet file per domain, the same choice the retail slice made.
RAW_FILE = "claude-3-7-sonnet-20250219_{domain}_default_gpt-4.1-2025-04-14_4trials.json"

TOOL_KINDS = {"READ": "read", "WRITE": "write", "GENERIC": "generic", "THINK": "generic"}


# --- the domain's own truth, read without importing tau2 ---------------------

def real_tools(domain: str) -> dict[str, dict]:
    """Every `@is_tool`-decorated method of the domain's agent toolkit, by name.

    Read from the source with `ast` rather than by importing tau2, which would pull in its whole
    dependency tree. `user_tools.py` is deliberately not read: those are the simulated user's own
    actions and no part of the customer's Environment (docs/cross-domain-check.md, Judgement).
    """
    path = VENDOR / "src" / "tau2" / "domains" / domain / "tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kind = _decorated_kind(node)
        if kind is None:
            continue
        args = [a.arg for a in node.args.args if a.arg != "self"]
        out[node.name] = {"kind": kind, "args": args}
    return out


def _decorated_kind(node: ast.AST) -> Optional[str]:
    for decorator in getattr(node, "decorator_list", []):
        if not (isinstance(decorator, ast.Call) and getattr(decorator.func, "id", "") == "is_tool"):
            continue
        for arg in decorator.args:
            if isinstance(arg, ast.Attribute) and arg.attr in TOOL_KINDS:
                return TOOL_KINDS[arg.attr]
    return None


def real_db(domain: str) -> dict[str, dict[str, dict]]:
    """The domain's seed database, with every table flattened to a list of rows.

    Two shapes to absorb, both real and neither a harness concern. The file is `db.json` for retail
    and airline and `db.toml` for telecom; and a table is a dict keyed by row id for retail and
    airline and a bare list for telecom. The first of each pair was all the scratch scripts knew.
    `user_db.toml` is not read: it is the simulated user's own phone, not the customer's world.
    """
    folder = VENDOR / "data" / "tau2" / "domains" / domain
    if (folder / "db.json").exists():
        blob = json.loads((folder / "db.json").read_text(encoding="utf-8"))
    else:
        blob = tomllib.loads((folder / "db.toml").read_text(encoding="utf-8"))
    return {table: _by_id(body) for table, body in blob.items()}


def _by_id(body: Any) -> dict[str, dict]:
    """One table as row id to row, whichever of tau2's two table shapes it arrived in.

    A dict-shaped table (retail, airline) is already keyed by row id. A list-shaped table (telecom)
    is keyed by the row's own id column. The key is never written into the row: a row's fields are
    what the domain put there, and adding one would make our rows and theirs differ by our own hand.
    """
    if isinstance(body, dict):
        return {str(key): row for key, row in body.items() if isinstance(row, dict)}
    if isinstance(body, list):
        return {_row_id(row): row for row in body if isinstance(row, dict)}
    return {}


def raw_task_ids(path: Path) -> dict[int, str]:
    """Simulation index to its tau2 `task_id`: the ground truth cluster labels.

    Read from the raw file, never from a Trace: `task_id` is a grader field and ingest strips it
    (D66). The index is what `Trace.raw_ptr.sim_index` carries.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {index: str(sim.get("task_id"))
            for index, sim in enumerate(blob.get("simulations") or [])}


# --- the offline slice -------------------------------------------------------

def offline_slice(domain: str, workdir: Path) -> dict:
    """Ingest, mine, cluster, canon rules and Starting state, with no model anywhere."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = RAW / RAW_FILE.format(domain=domain)
    if not path.exists():
        raise SystemExit(f"no raw file at {path}; run scripts/fetch_tau2_traces.sh first")
    ingest.ingest_file(path, workdir)
    traces = [Trace.model_validate_json(p.read_text(encoding="utf-8"))
              for p in sorted((workdir / "traces").glob("*.json"))]
    sigs = mine.mine_tools(traces)
    schema = mine.mine_schema(traces)
    categories, tasks = cluster.cluster_runs(traces, sigs)
    rows = [row for trace in traces for call in trace.tool_calls for row in _result_rows(call.result)]
    rules = canon.learn_rules(schema, rows)
    state = compile_env.build_starting_state(traces, schema, workdir, tasks, sigs)
    return {"path": path, "traces": traces, "sigs": sigs, "schema": schema,
            "categories": categories, "tasks": tasks, "rules": rules, "state": state}


def _result_rows(result: Any) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


# --- the comparisons ---------------------------------------------------------

def compare_tools(sigs: list, truth: dict[str, dict]) -> dict:
    """Mined signatures against the real toolkit: names, argument names and kind."""
    mined = {s.name: s for s in sigs}
    both = sorted(set(mined) & set(truth))
    arg_match = [n for n in both if set(mined[n].args_schema or {}) == set(truth[n]["args"])
                 or _arg_names(mined[n]) == set(truth[n]["args"])]
    kind_match = [n for n in both if mined[n].kind == truth[n]["kind"]]
    return {
        "real": len(truth),
        "mined": len(mined),
        "never_called": sorted(set(truth) - set(mined)),
        "not_real": sorted(set(mined) - set(truth)),
        "args_exact": f"{len(arg_match)}/{len(both)}",
        "kind_exact": f"{len(kind_match)}/{len(both)}",
        "kind_wrong": sorted((n, mined[n].kind, truth[n]["kind"]) for n in both if n not in kind_match),
    }


def _arg_names(sig: Any) -> set[str]:
    return {f.name for f in (sig.args_fields or [])}


def cluster_f1(tasks: list, traces: list[Trace], labels: dict[int, str], sigs: list) -> dict:
    """Pair F1 of our Task grouping against tau2's own `task_id` grouping.

    Pair counting, not cluster counting: two Runs are a true positive when we put them in one Task
    and tau2 gave them one task_id. The ceiling is what any tool-set clustering can reach at all,
    which is the share of task_ids whose Runs all write through the same tools.
    """
    by_id = {t.trace_id: t for t in traces}
    truth = {t.trace_id: labels.get(t.raw_ptr.sim_index) for t in traces}
    ours: dict[str, str] = {run_id: task.id for task in tasks for run_id in task.run_ids}
    run_ids = [r for r in ours if r in by_id and truth.get(r)]
    tp = fp = fn = 0
    for i, a in enumerate(run_ids):
        for b in run_ids[i + 1:]:
            same_ours, same_truth = ours[a] == ours[b], truth[a] == truth[b]
            tp += same_ours and same_truth
            fp += same_ours and not same_truth
            fn += same_truth and not same_ours
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": round(f1, 3), "precision": round(precision, 3), "recall": round(recall, 3),
            "pairs": len(run_ids), "ceiling": round(_ceiling(traces, truth, sigs), 3)}


def _ceiling(traces: list[Trace], truth: dict[str, Optional[str]], sigs: list) -> float:
    """The share of task_ids whose Runs all share one write signature, which no tool-set rule beats."""
    groups: dict[str, list[Trace]] = {}
    for trace in traces:
        label = truth.get(trace.trace_id)
        if label:
            groups.setdefault(label, []).append(trace)
    if not groups:
        return 0.0
    writes = cluster.write_tool_names(sigs)
    same = sum(1 for runs in groups.values()
               if len({cluster.category_signature(t, writes) for t in runs}) == 1)
    return same / len(groups)


def compare_state(db: dict, truth: dict[str, dict[str, dict]], rules: Any,
                  synthetic: Iterable[str] = ()) -> dict:
    """Every row our Starting state built, against the row the real database holds.

    Compared under the customer's own canonicalization rules, so a formatting difference the rules
    already know about is not counted as a mismatch. `norm` on top of that settles the one thing
    the retail-tuned rules never had to see: `"2025-01-15 10:30:00"` and `"2025-01-15T10:30:00"`
    are the same datetime, and telecom writes the first where the other domains write the second.
    """
    made_up = set(synthetic)
    tables_found = sorted(set(db) & set(truth))
    exact = mismatch = missing = synthesized = 0
    ours_fields = real_fields = fields_exact = fields_common = 0
    examples: list[str] = []
    for table in tables_found:
        for row_id, row in _by_id(db[table]).items():
            real = truth[table].get(row_id)
            if real is None:
                missing += 1
                continue
            if row_id in made_up:
                # A row the traces named but never showed, filled from the observed rows and tagged
                # so a Run that reads it is Assisted (D40, D41). It was never a claim about the real
                # database, so scoring it against one would measure the wrong thing.
                synthesized += 1
                continue
            ours_fields += len(row)
            real_fields += len(real)
            shared = [k for k in row if k in real]
            fields_common += len(shared)
            fields_exact += sum(1 for k in shared if _same({k: row[k]}, {k: real[k]}, rules))
            if _same(*_comparable(row, real), rules):
                exact += 1
            else:
                mismatch += 1
                if len(examples) < 5:
                    examples.append(f"{table}.{row_id}")
    return {"tables": f"{len(tables_found)}/{len(truth)}",
            "tables_missing": sorted(set(truth) - set(db)),
            "exact": exact, "mismatch": mismatch, "not_in_real_db": missing,
            "synthetic_not_scored": synthesized,
            # A thin row matches easily, so the match rate is only worth as much as this ratio:
            # how many of the real row's fields our row carries at all.
            "field_coverage": round(ours_fields / real_fields, 3) if real_fields else 0.0,
            # A whole-row mismatch says nothing about how wrong the row is. A tool that flattens a
            # nested table gives us every flat field right and the nesting missing, which is a
            # different thing from a wrong value, and this is the number that tells them apart.
            "fields_exact": f"{fields_exact}/{fields_common}",
            "examples": examples}


def _same(ours: dict, real: dict, rules: Any) -> bool:
    """Two rows equal under the customer's canonicalization rules, datetime separators settled."""
    return canon.canonicalize(norm(ours), rules) == canon.canonicalize(norm(real), rules)


def _row_id(row: dict) -> str:
    for key in ("id", *[k for k in row if isinstance(k, str) and k.endswith("_id")]):
        if key in row and isinstance(row[key], (str, int)):
            return str(row[key])
    return json.dumps(row, sort_keys=True, default=str)[:60]


def _comparable(ours: dict, real: dict) -> tuple[dict, dict]:
    """The two rows reduced to what they can honestly be compared on.

    Two conventions, both about absence and neither a defect in either row. A row we built from
    partial reads is not wrong for being partial, so only the fields our row claims are compared.
    And a field the customer's model declares optional is written as `null` by a tool result and
    left out of the seed file altogether, so an absent field and a null field are the same field.
    """
    keys = {k for k in ours if k in real or ours.get(k) is not None}
    return ({k: v for k, v in ours.items() if k in keys and v is not None},
            {k: v for k, v in real.items() if k in keys and v is not None})


DATETIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")


def norm(value: Any) -> Any:
    """Datetime separators settled, so a space and a T are the same instant."""
    if isinstance(value, str):
        found = DATETIME.match(value)
        return f"{found.group(1)}T{found.group(2)}{value[found.end():]}" if found else value
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [norm(v) for v in value]
    return value


# --- reporting ---------------------------------------------------------------

def check(domain: str, workdir: Path) -> dict:
    built = offline_slice(domain, workdir)
    truth_tools = real_tools(domain)
    labels = raw_task_ids(built["path"])
    calls = [c for t in built["traces"] for c in t.tool_calls]
    return {
        "domain": domain,
        "ingest": {"traces": len(built["traces"]), "calls": len(calls),
                   "by_assistant": sum(1 for c in calls if (c.requestor or "assistant") == "assistant"),
                   "errors": sum(1 for c in calls if c.error is not None)},
        "tools": compare_tools(built["sigs"], truth_tools),
        "cluster": cluster_f1(built["tasks"], built["traces"], labels, built["sigs"]),
        "state": compare_state(built["state"].db, real_db(domain), built["rules"],
                               built["state"].synthetic_rows),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", nargs="*", default=["retail", "airline", "telecom"])
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--json", type=Path, default=None, help="Write the results here as well.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = [check(domain, args.workdir / f"xdomain-{domain}") for domain in args.domains]
    for result in results:
        print(f"\n## {result['domain']}")
        for section in ("ingest", "tools", "cluster", "state"):
            print(f"\n{section}:")
            for key, value in result[section].items():
                print(f"  {key}: {value}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
