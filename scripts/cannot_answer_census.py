"""Census of Router dead ends over stored Runs (G28 step 1).

Reads workdirs in place and writes nothing into them. For every stored Run it
counts tool calls that ended in the Router's own dead end: route code with error
class tool_not_found, which is neither code answering nor the recording answering
nor a body refusing (a body never raises that class). Each dead end is split into
caller-filter refusals (the tool is mined for other callers only), plain listed
(the mined tool list holds the tool for this caller and the Environment has no
body for it: the Environment's fault) and plain unlisted (a name the Environment
never had: the Candidate's mistake). Runs carrying a dead end are scored by code
against their stored Verifier, and the Verdict classes are reported.

Usage: uv run python scripts/cannot_answer_census.py <workdir> [<workdir> ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter


def file_kind(filename: str) -> str:
    if filename.startswith("replay-"):
        return "replay"
    if filename.startswith("reroll-"):
        return "reroll"
    if filename.startswith("probe-"):
        return "probe"
    if filename.startswith("second-path"):
        return "second-path"
    if filename.startswith("synth"):
        return "synth"
    return "other"


def dead_ends(path: str) -> list[tuple[str, bool]]:
    """(tool name, caller-filtered) pairs for every Router dead end in one Run file."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if '"tool_not_found"' not in line:
                continue
            event = json.loads(line)
            if event.get("type") != "tool_result" or event.get("route") != "code":
                continue
            error = (event.get("payload") or {}).get("error") or {}
            if error.get("class") != "tool_not_found":
                continue
            name = (event.get("payload") or {}).get("name")
            out.append((name, " for the " in str(error.get("payload"))))
    return out


def task_of(path: str) -> str | None:
    parent = os.path.basename(os.path.dirname(path))
    if parent.startswith("task_"):
        return parent
    base = os.path.basename(path)
    if base.startswith("probe-"):
        return base[len("probe-"): -len(".jsonl")]
    return None


def load_listed(workdir: str) -> set[str]:
    with open(os.path.join(workdir, "tool_sigs.json"), encoding="utf-8") as handle:
        return {entry["name"] for entry in json.load(handle)}


def load_verifier(workdir: str, cache: dict, task_id: str | None):
    if task_id is None or task_id in cache:
        return cache.get(task_id)
    path = os.path.join(workdir, "verifiers", f"{task_id}.json")
    if not os.path.exists(path):
        cache[task_id] = None
        return None
    from kullback.runner.records import Verifier

    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    try:
        cache[task_id] = Verifier.model_validate(stored.get("verifier", stored))
    except Exception:
        cache[task_id] = None
    return cache[task_id]


def score_affected(path: str, kind: str, task_id: str | None, workdir: str,
                   cache: dict, classes: Counter) -> None:
    from kullback.runner.verdict import verdict

    verifier = load_verifier(workdir, cache, task_id)
    if verifier is None:
        classes[(kind, "no-verifier")] += 1
        return
    try:
        classes[(kind, verdict(path, verifier).class_)] += 1
    except Exception as error:
        classes[(kind, f"score-error:{type(error).__name__}")] += 1


def census(workdir: str) -> dict:
    listed = load_listed(workdir)
    cache: dict[str, object] = {}
    files = sorted(glob.glob(os.path.join(workdir, "runs", "*", "*.jsonl"))
                   + glob.glob(os.path.join(workdir, "probes", "*.jsonl")))
    calls: Counter = Counter()
    runs: Counter = Counter()
    tools: Counter = Counter()
    classes: Counter = Counter()
    for path in files:
        kind = file_kind(os.path.basename(path))
        found = dead_ends(path)
        if not found:
            continue
        buckets = set()
        for name, filtered in found:
            bucket = "caller-filter" if filtered else ("listed" if name in listed else "unlisted")
            calls[(kind, bucket)] += 1
            tools[(kind, bucket, name)] += 1
            buckets.add(bucket)
        for bucket in buckets:
            runs[(kind, bucket)] += 1
        score_affected(path, kind, task_of(path), workdir, cache, classes)
    return {"files": len(files), "calls": calls, "runs": runs, "tools": tools, "classes": classes}


def main(workdirs: list[str]) -> None:
    for workdir in workdirs:
        name = os.path.basename(os.path.normpath(workdir))
        result = census(workdir)
        print(f"== {name}: {result['files']} stored Run files ==")
        kinds = sorted({key[0] for key in list(result["calls"]) + list(result["runs"])})
        for kind in kinds:
            total_calls = sum(count for (key, count) in result["calls"].items() if key[0] == kind)
            total_runs = sum(count for (key, count) in result["runs"].items() if key[0] == kind)
            print(f"  {kind}: {total_calls} dead-end calls in {total_runs} Runs")
            for bucket in ("caller-filter", "listed", "unlisted"):
                bucket_calls = result["calls"].get((kind, bucket), 0)
                bucket_runs = result["runs"].get((kind, bucket), 0)
                names = {tool: count for (k, b, tool), count in result["tools"].items()
                         if (k, b) == (kind, bucket)}
                print(f"    {bucket}: {bucket_calls} calls in {bucket_runs} Runs, tools {dict(names)}")
        print(f"  Verdict class of affected Runs: "
              f"{ {f'{k}/{c}': n for (k, c), n in sorted(result['classes'].items())} }")


if __name__ == "__main__":
    main(sys.argv[1:])
