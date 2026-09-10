"""The pipeline as the run recorded it: one row per stage, with the gate that failed."""

from __future__ import annotations

import re
from typing import Any, Optional

from kullback.report.data import ReportData, StageStatus
from kullback.runner.records import GateResult


def node_id(name: str) -> str:
    """A mermaid node id from a stage name: a space or a hyphen would end the id and break the graph."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "stage"


def pipeline_dag(stages: list[StageStatus]) -> str:
    """The pipeline as mermaid, in the order the run recorded it, with a rollback edge per failed gate."""
    if not stages:
        return 'flowchart TD\n    empty["no stages recorded"]'
    lines = ["flowchart TD"]
    for stage in stages:
        label = f"{stage.name} ({stage.status})".replace('"', "'")
        lines.append(f'    {node_id(stage.name)}["{label}"]')
    for before, after in zip(stages, stages[1:], strict=False):
        lines.append(f"    {node_id(before.name)} --> {node_id(after.name)}")
    for stage in stages:
        if stage.status == "failed" and stage.gate:
            attempts = max(stage.attempts, 1)
            allowed = max(stage.max_attempts, attempts)
            node = node_id(stage.name)
            lines.append(f'    {node} -. "gate {stage.gate} failed, attempt {attempts} of {allowed}" .-> {node}')
    return "\n".join(lines)


ENVIRONMENT_GATE_NAMES = ("build_environment", "environment", "build_env")


def _gate_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (name or "").strip().lower())


def environment_gate(data: ReportData) -> Optional[GateResult]:
    """The build Environment gate (design section 6), the one that decides whether it was built."""
    for gate in data.gates:
        if _gate_key(gate.stage) in ENVIRONMENT_GATE_NAMES:
            return gate
    return None


def stage_statuses(state: dict, budget: Any = None, stopped: Any = None) -> list[StageStatus]:
    """The pipeline's own state.json as the DAG rows: status, attempts, the gate that failed, spend.

    pipeline.py writes `statuses`, `attempts`, `gates`, `failed_stage` and a log of plain sentences,
    so all of them are read here rather than a log of dict rows nothing writes.
    """
    statuses = state.get("statuses") or {}
    attempts = state.get("attempts") or {}
    failed = state.get("failed_stage")
    per_stage = dict((stopped or {}).get("stages") or {})
    buckets = {
        name: bucket for name, bucket in ((budget or {}).get("stages") or {}).items() if isinstance(bucket, dict)
    }
    for name, bucket in buckets.items():
        if bucket.get("usd"):
            per_stage[name] = bucket["usd"]
    stages = []
    for name, status in statuses.items():
        stage = StageStatus(
            name=name,
            status=str(status),
            usd=float(per_stage.get(name) or 0.0),
            cache_share=_cache_share(buckets.get(name, {})),
            memo_hits=int(buckets.get(name, {}).get("memo_hits") or 0),
        )
        try:
            stage.attempts = int(attempts.get(name) or 0)
        except (TypeError, ValueError):
            stage.attempts = 0
        stages.append(stage)
    by_name = {stage.name: stage for stage in stages}
    for body in state.get("gates") or []:
        if not isinstance(body, dict) or body.get("pass", True):
            continue
        gate_stage = str(body.get("stage") or "")
        owner = by_name.get(gate_stage) or by_name.get(gate_stage.split(".")[0]) or by_name.get(str(failed))
        if owner is not None and owner.status == "failed":
            owner.gate = gate_stage.split(".")[-1]
    for row in state.get("log") or []:
        _read_log_row(row, by_name)
    return stages


def _cache_share(bucket: dict) -> float:
    """cache_read over all input tokens: the number `docs/prompt-caching.md` says to watch per stage."""
    try:
        read = float(bucket.get("cache_read") or 0)
        total = read + float(bucket.get("input") or 0)
    except (TypeError, ValueError):
        return 0.0
    return read / total if total else 0.0


def _read_log_row(row: Any, by_name: dict) -> None:
    """One rollback log entry, the sentence pipeline.py writes to `result.log` (its only writer).

    pipeline.py logs "compile_tools: attempt 2 of 3, gate replay_fidelity failed" and, on the last
    attempt, "compile_tools: gate replay_fidelity failed 3 times, stage failed".
    """
    if not isinstance(row, str) or ":" not in row:
        return
    name, said = row.split(":", 1)
    stage = by_name.get(name.strip())
    if stage is None:
        return
    gate = re.search(r"gate (\S+) failed", said)
    if gate:
        stage.gate = gate.group(1)
    attempt = re.search(r"attempt (\d+) of (\d+)", said)
    if attempt:
        stage.attempts = max(stage.attempts, int(attempt.group(1)))
        stage.max_attempts = max(stage.max_attempts, int(attempt.group(2)))
    times = re.search(r"failed (\d+) times", said)
    if times:
        stage.attempts = max(stage.attempts, int(times.group(1)))
        stage.max_attempts = max(stage.max_attempts, int(times.group(1)))
