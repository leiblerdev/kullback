"""Re-score stored Runs against a new Verifier or Environment version without re-executing them (D97, tau3 --fresh-tasks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.runner.verdict import VERDICT_VERSION, load_run, verdict
from harness.shared.canon import QUEUE_FILE, clear_regrade_queue, queued_regrades
from harness.shared.records import Verdict, Verifier, as_dict, content_hash


def _fingerprint(value: Any) -> Any:
    """One verdict input as something content_hash can order: records as data, sets sorted."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, dict):
        return {str(k): _fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint(item) for item in value]
    if hasattr(value, "model_dump"):
        return as_dict(value)
    rules = getattr(value, "rules", None)
    if rules is not None and hasattr(rules, "model_dump"):
        return {"canon_rules": as_dict(rules)}
    return getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or repr(value)


def cache_key(run_id: str, verifier: Verifier, environment: Any = None,
              runner_version: Optional[str] = None, judge_version: Optional[str] = None,
              verdict_version: str = VERDICT_VERSION, **verdict_inputs: Any) -> str:
    """Every input a Verdict depends on, hashed; a change in any of them retires the stored Verdict.

    Versions alone are not enough (design section 8 keys a cached artifact by every input): the
    write-tool set, the flagged tools, the schema, the Reference path, the canonicalizer's rules and
    the judge's own answers all move a Verdict without moving a version, and a person who overturns
    a judge atom or an equivalence entry moves none of them at all (D39, D76, D84).
    """
    return content_hash({
        "run_id": run_id,
        "verifier_version": verifier.verifier_version,
        "verifier": as_dict(verifier),
        "env_id": getattr(environment, "env_id", None),
        "schema_version": getattr(environment, "schema_version", None),
        "tools_version": getattr(environment, "tools_version", None),
        "policy_version": getattr(environment, "policy_version", None),
        "runner_version": runner_version,
        "judge_version": judge_version,
        "verdict_version": verdict_version,
        # An input left None is the same as an input not passed, so it is dropped rather
        # than hashed: otherwise the same Verdict would sit under two keys.
        "verdict_inputs": _fingerprint({k: v for k, v in verdict_inputs.items() if v is not None}),
    })


def verdict_path(out_dir: Any, run_id: str, key: str) -> Path:
    """Where one regraded Verdict is stored: content-addressed on its versions (design section 8)."""
    return Path(out_dir) / f"{run_id}.{key[:16]}.json"


def regrade_run(run_jsonl: Any, verifier: Verifier, canon: Any = None, *, out_dir: Any = None,
                judge_results: Optional[dict] = None, judge_version: Optional[str] = None,
                environment: Any = None, runner_version: Optional[str] = None,
                verdict_version: str = VERDICT_VERSION, refresh: bool = False,
                **verdict_kwargs: Any) -> Verdict:
    """Re-run verdict() over one stored Run; a Verdict already written under the same versions is reused.

    `refresh` re-scores anyway, which is what an overturned equivalence entry needs (D84): the
    versions have not moved, but the answer the Verdict rested on has.
    """
    run_id = load_run(run_jsonl).run_id
    key = cache_key(run_id, verifier, environment, runner_version, judge_version, verdict_version,
                    canon=canon, judge_results=judge_results, **verdict_kwargs)
    path = verdict_path(out_dir, run_id, key) if out_dir is not None else None
    if path is not None and path.is_file() and not refresh:
        return Verdict.model_validate(json.loads(path.read_text(encoding="utf-8")))

    result = verdict(
        run_jsonl, verifier, canon, judge_results,
        environment=environment, runner_version=runner_version,
        verdict_version=verdict_version, **verdict_kwargs,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(as_dict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def regrade(run_jsonls: Iterable[Any], verifier: Verifier, canon: Any = None, *, out_dir: Any = None,
            judge_results: Optional[dict] = None, judge_version: Optional[str] = None,
            environment: Any = None, runner_version: Optional[str] = None,
            verdict_version: str = VERDICT_VERSION, queue_dir: Any = None, refresh: bool = False,
            **verdict_kwargs: Any) -> list[Verdict]:
    """Re-score a batch of stored Runs; judge_results is keyed by run id, then by atom id (D76).

    `queue_dir` is the workdir holding canon.py's regrade queue (D84): a Run in it is re-scored even
    though its stored Verdict is still under the current versions, because a person overturned an
    equivalence entry it rested on. Only the Runs this batch re-scored leave the queue; a queued Run
    that was not in the batch stays there, or it would keep its stale Verdict forever.

    `refresh` re-scores every Run in the batch whatever the cache holds.
    """
    forced = set(queued_regrades(queue_dir)) if queue_dir is not None else set()
    done: set = set()
    out: list[Verdict] = []
    for run_jsonl in run_jsonls:
        run_id = load_run(run_jsonl).run_id
        per_run = (judge_results or {}).get(run_id)
        out.append(regrade_run(
            run_jsonl, verifier, canon, out_dir=out_dir, judge_results=per_run,
            judge_version=judge_version, environment=environment, runner_version=runner_version,
            verdict_version=verdict_version, refresh=refresh or run_id in forced, **verdict_kwargs,
        ))
        done.add(run_id)
    if forced & done:
        _drop_from_queue(queue_dir, forced & done)
    return out


def _drop_from_queue(queue_dir: Any, done: set) -> None:
    """Take the re-scored Runs out of canon.py's queue and leave the rest of it standing."""
    path = Path(queue_dir) / QUEUE_FILE
    if not path.is_file():
        return
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("run_id") not in done]
    if kept:
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    else:
        clear_regrade_queue(queue_dir)
