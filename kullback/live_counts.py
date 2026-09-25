"""One live count of a workdir, and one row per Task, for every frontend.

The Watch screen shows fidelity, trusted, refused, open and drifted while a build runs, and the
Tasks screen lists every Task with why it stands where it stands. The code-only count lived in
cli._counts_line as a string the screen could not import cleanly; it lives here as data both
frontends import, cheap enough to call every few seconds, with one row per Task beside it.

Nothing here calls a model. Rows carry kinds, ids, counts and field paths only, never recorded
values, so the screen can print them without leaking customer strings.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from kullback import round_snapshot
from kullback.gates import counts as counts_mod
from kullback.gates.probes import as_pool, as_verifier, probe_scores, write_tools_of
from kullback.gates.trust import workdir_trusted_ruling
from kullback.runner.canon import CanonRules

# Files workdir_counts reads: the fidelity inputs, the round snapshot, and every file the trusted
# ruling reads (the Examiner's history, task runs and re-rolls, both probe pools, the Examiner's
# Verifier proposals, and the Runs the seed provenance step loads).
_FIXED_FILES = ("task_status.json", "replays.json", "rerolls.json", "canon-rules.json", "tool_sigs.json",
                "exam/history.json", "exam/task_runs.json", "examiner/rerolls.json")
_DIR_GLOBS = ("verifiers/*.json", "refusals/*.json", "env/refusals/*.json", "rounds/*/tasks.json",
              "probes/*/pool.json", "exam/verifiers/*.json", "exam/probes/*/*.json", "runs/**/*.jsonl")

# A file no reader could parse reads as this, so a caller can tell "no file" from "an empty record".
_MISSING: Any = object()

# The last count per workdir: the key, when it was read, and the dict itself.
_CACHE: dict[str, dict] = {}

# Statuses a row can stand in. Drift is a separate flag, not a status: a drifted Task keeps its word.
TRUSTED, REFUSED, OPEN = "trusted", "refused", "open"


def _read_json(path: Any, default: Any = None) -> Any:
    """One JSON file, or the default where it is missing or half-written."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def json_at(root: Path, name: str, reader: Optional[Callable] = None) -> dict:
    """One JSON record of a workdir, or nothing where the file is missing or half-written."""
    read = reader or _read_json
    body = read(Path(root) / name, _MISSING)
    return body if isinstance(body, dict) else {}


def verifier_dicts(root: Path, reader: Optional[Callable] = None) -> list:
    """The stored Verifiers as dicts; a workdir with none yet reads as empty."""
    read = reader or _read_json
    folder = Path(root) / "verifiers"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        body = read(path, _MISSING)
        if body is not _MISSING:
            out.append(body)
    return out


def refusal_dicts(root: Path, reader: Optional[Callable] = None) -> dict:
    """The stored refusals keyed by Task; a workdir with none yet reads as empty."""
    read = reader or _read_json
    out: dict = {}
    for folder in (Path(root) / "refusals", Path(root) / "env" / "refusals"):
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                body = read(path, _MISSING)
                if body is not _MISSING:
                    out.setdefault(path.stem, body)
    return out


def _sig(path: Path) -> tuple:
    """What the cache keys one file on: its mtime and size, or its absence."""
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _cache_key(root: Path) -> tuple:
    """What the cached count is keyed on: every file the count reads, so an unchanged key means
    unchanged inputs and the count can be returned without re-reading or re-counting."""
    parts = [_sig(root / name) for name in _FIXED_FILES]
    for pattern in _DIR_GLOBS:
        parts += [_sig(path) for path in sorted(root.glob(pattern))]
    return tuple(parts)


def _canon_rules(body: Any) -> CanonRules:
    """The customer's canonicalizer rules, or the defaults where the file is missing."""
    return CanonRules.model_validate(body) if isinstance(body, dict) else CanonRules()


def _inputs(root: Path, read: Callable) -> dict:
    """Everything the counts and the rows read off the live files, in one place."""
    task_status = json_at(root, "task_status.json", read)
    replays = json_at(root, "replays.json", read)
    rerolls = json_at(root, "rerolls.json", read)
    sigs_body = read(root / "tool_sigs.json", _MISSING)
    sigs = sigs_body if isinstance(sigs_body, list) else (sigs_body.get("sigs", [])
                                                          if isinstance(sigs_body, dict) else [])
    canon = _canon_rules(read(root / "canon-rules.json", _MISSING))
    return {"task_status": task_status, "verifiers": verifier_dicts(root, read),
            "refusals": refusal_dicts(root, read), "replays": replays, "rerolls": rerolls,
            "canon": canon, "sigs": sigs if isinstance(sigs, list) else []}


def _snapshot(root: Path, read: Callable) -> Optional[dict]:
    """The last closed round's table, or None where no round has closed yet."""
    rounds = sorted(path for path in root.glob("rounds/*/tasks.json") if path.parent.name.isdigit())
    if not rounds:
        return None
    body = read(rounds[-1], _MISSING)
    return body if isinstance(body, dict) else None


def _trust_words(root: Path) -> tuple[set, dict]:
    """The ruling's trusted ids and refusals, the same words the Builder's status reads."""
    ruling = workdir_trusted_ruling(root)
    return set(ruling.metrics.get("trusted") or ()), dict(ruling.metrics.get("refused") or {})


def workdir_counts(workdir: Any, max_age: float = 5.0, reader: Optional[Callable] = None) -> dict:
    """The live count of a workdir as data: tasks, fidelity, trusted, refused, open and drifted.

    Fidelity comes from the gate rulings over the replay rows; trusted, refused and open come from
    the one trusted ruling the Builder's status and the round snapshot read. A second call with no
    file changed returns the cached dict without re-reading or re-counting, and a call younger than
    max_age seconds returns it even if files moved, so the screen can call this every few seconds.
    `reader` is the module's JSON reader by default; a caller may pass its own to count the opens.
    """
    root = Path(workdir)
    now = time.time()
    cached = _CACHE.get(str(root))
    if cached is not None and now - cached["read_at"] < max_age:
        return cached["result"]
    key = _cache_key(root)
    if cached is not None and cached["key"] == key:
        return cached["result"]
    read = reader or _read_json
    data = _inputs(root, read)
    result = counts_mod.round_counts(
        data["task_status"], data["verifiers"], {}, {}, data["refusals"], {}, data["replays"],
        data["rerolls"], data["canon"], data["sigs"], workdir=root)
    trusted, refused = _trust_words(root)
    snapshot = _snapshot(root, read)
    drifted = round_snapshot.drift(
        snapshot, task_status=data["task_status"], replays=data["replays"], named=0)["status_drift"]
    tasks = result["tasks"]
    opened = len([task_id for task_id in data["task_status"] if task_id not in trusted and task_id not in refused])
    out = {"tasks": tasks, "fidelity": result["fidelity"], "trusted": len(trusted & set(data["task_status"])),
           "refused": len(refused.keys() & set(data["task_status"])), "open": opened,
           "drifted": drifted, "read_at": now}
    _CACHE[str(root)] = {"key": key, "read_at": now, "result": out}
    return out


def _findings(root: Path, read: Callable) -> list:
    """The Examiner's findings as dicts; a workdir with none yet reads as empty."""
    body = read(root / "findings.json", _MISSING)
    return [row for row in body if isinstance(row, dict)] if isinstance(body, list) else []


def _gates(root: Path, read: Callable) -> list:
    """The recorded gate rulings as dicts; a workdir with none yet reads as empty."""
    body = read(root / "gates.json", _MISSING)
    return [row for row in body if isinstance(row, dict)] if isinstance(body, list) else []


def _task_ids(root: Path, data: dict) -> list[str]:
    """Every Task the workdir knows: live rows, replays, Task files, Verifiers and synthetic Tasks."""
    ids = set(data["task_status"]) | set(data["replays"]) | set(data["refusals"])
    for folder in (root / "tasks", root / "verifiers", root / "synthetic" / "tasks"):
        if folder.is_dir():
            ids |= {path.stem for path in folder.glob("*.json")}
    return sorted(ids)


def _synthetic_ids(root: Path) -> set[str]:
    """The Tasks the synthesis store holds apart from the build's own Tasks."""
    folder = root / "synthetic" / "tasks"
    return {path.stem for path in folder.glob("*.json")} if folder.is_dir() else set()


def _replay_word(per_task: Any) -> Optional[str]:
    """Whether some Trace of the Task replayed to its End state, or None with no replay at all."""
    if not isinstance(per_task, dict) or not per_task:
        return None
    return "confirmed" if any(isinstance(row, dict) and row.get("confirmed")
                              for row in per_task.values()) else "unconfirmed"


def _suite_word(row: Any) -> Optional[str]:
    """Whether the Task's Verifier cleared its suite, or None where the Task has no status row."""
    if not isinstance(row, dict):
        return None
    return "pass" if row.get("verifier_passed") else "fail"


def _probes_word(pool_body: Any, verifier_body: Any, canon: CanonRules, sigs: list) -> Optional[str]:
    """The probes the Task's current Verifier passes, over the pool size, or None with no pool.

    Scored with the same function the trusted gate scores with, so the row and the ruling agree.
    """
    if pool_body is None or verifier_body is None:
        return None
    try:
        pool = as_pool(pool_body)
        if not pool.probes:
            return None
        scores = probe_scores(as_verifier(verifier_body), pool, canon, write_tools_of(sigs))
    except Exception:
        return None
    return f"{sum(1 for ok in scores.values() if ok)}/{len(pool.probes)}"


def _last_finding(findings: list, task_id: str) -> tuple:
    """The last open finding over the Task: its kind and its path, or nothing where there is none."""
    kind, path = None, None
    for row in findings:
        if row.get("task_id") != task_id and task_id not in (row.get("task_ids") or ()):
            continue
        if row.get("status", "open") != "open":
            continue
        kind, path = row.get("kind"), row.get("path") or None
    return kind, path


def _status_of(task_id: str, trusted: set, refused: set) -> str:
    """The word the Task stands in: trusted, refused, or the loop's remaining work."""
    if task_id in trusted:
        return TRUSTED
    if task_id in refused:
        return REFUSED
    return OPEN


def task_rows(workdir: Any) -> list[dict]:
    """One row per Task: its status word, replay, suite, probes, drift and last finding.

    Every value is filled from the workdir files that hold it; a value that is not there is None.
    Kinds, ids, counts and field paths only: recorded values never enter a row.
    """
    root = Path(workdir)
    read = _read_json
    data = _inputs(root, read)
    trusted, refused_map = _trust_words(root)
    refused = set(refused_map)
    ids = _task_ids(root, data)
    snapshot = _snapshot(root, read)
    drifted = {row["task_id"]: row["stage"]
               for row in round_snapshot.drift(
                   snapshot, task_status=data["task_status"], replays=data["replays"],
                   named=len(ids))["first"]}
    findings = _findings(root, read)
    pools = {path.parent.name: read(path, None) for path in sorted((root / "probes").glob("*/pool.json"))} \
        if (root / "probes").is_dir() else {}
    verifiers = {path.stem: read(path, None) for path in sorted((root / "verifiers").glob("*.json"))} \
        if (root / "verifiers").is_dir() else {}
    synthetic = _synthetic_ids(root)
    rows = []
    for task_id in ids:
        kind, path = _last_finding(findings, task_id)
        rows.append({"task_id": task_id, "status": _status_of(task_id, trusted, refused),
                     "replay": _replay_word(data["replays"].get(task_id)),
                     "suite": _suite_word(data["task_status"].get(task_id)),
                     "probes": _probes_word(pools.get(task_id), verifiers.get(task_id),
                                            data["canon"], data["sigs"]),
                     "drift": drifted.get(task_id),
                     "last_finding_kind": kind, "last_finding_path": path,
                     "synthetic": task_id in synthetic})
    return rows


def _failing_gate(status_row: Any, gates_rows: list) -> Optional[str]:
    """The gate that holds the Task: its first failed suite check, else the first failed gate row."""
    checks = (status_row or {}).get("checks") if isinstance(status_row, dict) else None
    if isinstance(checks, dict):
        for name in list(round_snapshot.CHECK_ORDER) + sorted(set(checks) - set(round_snapshot.CHECK_ORDER)):
            if name in checks and not checks[name]:
                return str(name)
    for row in gates_rows or ():
        if isinstance(row, dict) and not row.get("passed", row.get("pass", True)):
            return str(row.get("stage") or "unknown")
    return None


def _replay_numbers(per_task: Any) -> tuple:
    """The confirmed replay's matched and total calls, or nothing where no replay recorded counts."""
    matched, total, seen = 0, 0, False
    for row in (per_task or {}).values() if isinstance(per_task, dict) else ():
        counts = row.get("counts") if isinstance(row, dict) else None
        if not isinstance(counts, dict):
            continue
        seen = True
        total += int(counts.get("calls") or 0)
        matched += int(counts.get("writes_matched") or 0) + int(counts.get("reads_same") or 0) \
            + int(counts.get("reads_cosmetic") or 0) + int(counts.get("reads_both_refused") or 0)
    return (matched, total) if seen else (None, None)


def task_detail(workdir: Any, task_id: str) -> dict:
    """One Task's row plus the gate that failed, its open findings, its Verifier's atoms and its replay.

    Findings carry kind and path only, atoms are counted by kind, and the replay carries matched and
    total calls. A Task the workdir does not know raises KeyError.
    """
    root = Path(workdir)
    read = _read_json
    data = _inputs(root, read)
    rows = {row["task_id"]: row for row in task_rows(workdir)}
    if task_id not in rows:
        raise KeyError(task_id)
    detail = dict(rows[task_id])
    findings = _findings(root, read)
    detail["failing_gate"] = _failing_gate(data["task_status"].get(task_id), _gates(root, read))
    detail["open_findings"] = [{"kind": row.get("kind"), "path": row.get("path") or None} for row in findings
                               if (row.get("task_id") == task_id or task_id in (row.get("task_ids") or ()))
                               and row.get("status", "open") == "open"]
    verifier = read(root / "verifiers" / f"{task_id}.json", None)
    atoms = verifier.get("atoms") if isinstance(verifier, dict) else None
    detail["atom_counts"] = dict(Counter(str(atom.get("kind")) for atom in atoms
                                         if isinstance(atom, dict))) if isinstance(atoms, list) else None
    matched, total = _replay_numbers(data["replays"].get(task_id))
    detail["replay_matched"], detail["replay_total"] = matched, total
    return detail


__all__ = ["json_at", "refusal_dicts", "task_detail", "task_rows", "verifier_dicts", "workdir_counts"]
