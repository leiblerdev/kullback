"""Local backends for acting on training: asks, smoke runs and env checks."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from kullback import curriculum, sampling
from kullback.trainer import rollout as rollout_mod
from kullback.trainer.observe import open_asks

SMOKE_KIND = "smoke"


def ask_tasks(bus: Any, *, train_run_id: str, model_id: str, checkpoint: str,
              step: int, level: str, count: int, kind: Optional[str] = None,
              evidence: Optional[dict] = None, exemplars: Any = ()) -> dict:
    """Publish a TaskAsk unless this run already has an open one."""
    open_ids = open_asks(bus.replay(), train_run_id)[2]
    if open_ids:
        return {"published": False,
                "reason": f"run {train_run_id} already has open ask {open_ids[0]}"}
    ask = curriculum.TaskAsk(
        ask_id=curriculum.TaskAsk.ask_id_for(train_run_id, step, level),
        train_run_id=train_run_id,
        model_id=model_id,
        checkpoint=checkpoint,
        step=step,
        level=level,
        kind=kind,
        count=count,
        evidence=dict(evidence or {}),
        exemplars=list(exemplars),
    )
    bus.publish_custom("task_ask", ask.model_dump())
    return {"published": True, "ask": ask.model_dump()}


def smoke_sample(env: Any, n: int, key: Any) -> list[str]:
    """n task ids by keyed draw over verifiable rule-driven tasks."""
    pool = [tid for tid in env.task_ids()
            if env.verifier(tid) is not None and not env.agent_driven(tid)]
    ordered = sampling.keyed_order(f"{SMOKE_KIND}:{key}", pool, env.salt)
    if n >= len(ordered):
        return ordered
    return ordered[:max(0, n)]


def smoke_test(env: Any, policy: Any, *, n: int, k: int, outdir: Any, key: Any,
               model_id: str, checkpoint: str, train_run_id: str,
               max_turns: int = 30) -> dict:
    """Play a smoke sample and summarise its groups like a training window."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ids = smoke_sample(env, n, key)
    records: list[curriculum.SolveRate] = []
    skipped = 0
    played = 0
    start = time.time()
    for task_id in ids:
        try:
            _, record = rollout_mod.run_group(
                env, task_id, k, policy, outdir=out, train_run_id=train_run_id,
                model_id=model_id, checkpoint=checkpoint, source="smoke",
                max_turns=max_turns)
        except rollout_mod.TaskSkipped:
            skipped += 1
            continue
        records.append(curriculum.SolveRate(**record))
        played += k
    elapsed = time.time() - start
    views = curriculum.fold(records, max(1, len(records)))
    summary = curriculum.window_summary(views)
    stop: dict[str, int] = {}
    for record in records:
        for name, count in record.stop_reasons.items():
            stop[name] = stop.get(name, 0) + count
    summary["skipped"] = skipped
    summary["seconds_per_rollout"] = (elapsed / played) if played else None
    summary["stop_reasons"] = stop
    return summary


def check_env(env: Any, policy_do_nothing: Any = None) -> dict:
    """Load counts plus a do-nothing run that must not pass."""
    ids = env.task_ids()
    verifiable = [tid for tid in ids if env.verifier(tid) is not None]
    playable = [tid for tid in verifiable if not env.agent_driven(tid)]
    report: dict[str, Any] = {
        "task_count": len(ids),
        "with_verifier": len(verifiable),
        "agent_driven": sum(1 for tid in ids if env.agent_driven(tid)),
        "do_nothing_task": playable[0] if playable else None,
        "do_nothing_passes": False,
        "do_nothing_class": None,
    }
    if not playable:
        return report
    policy = policy_do_nothing or (lambda messages, tools: {"content": "done",
                                                            "tool_calls": []})
    with tempfile.TemporaryDirectory() as tmp:
        rollout = rollout_mod.play(
            env, playable[0],
            rollout_mod.rollout_seed(env, playable[0], 0),
            policy, outdir=tmp)
    report["do_nothing_passes"] = rollout.reward == 1
    report["do_nothing_class"] = rollout.reward_class
    return report


def _plain_call(call: Any) -> dict:
    """One tool call as a plain dict: the Episode reads id, name and arguments off dicts."""
    if isinstance(call, dict):
        return dict(call)
    return call.model_dump()


def policy_from_model(model: Any) -> Any:
    """Wrap query(messages, tools) into the Policy callable shape."""

    def policy(messages: list[dict], tools: list[dict]) -> dict:
        reply = model.query(messages, tools)
        if isinstance(reply, dict):
            content = reply.get("content")
            calls = reply.get("tool_calls", ()) or ()
        else:
            content = getattr(reply, "content", None)
            calls = getattr(reply, "tool_calls", ()) or ()
        return {"content": content, "tool_calls": [_plain_call(call) for call in calls]}

    return policy
