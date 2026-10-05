"""One Task played through the Episode interface under the rule user, grouped in k.

A training agent plays each Task k times and learns from the group. The rule user
drives every Run, so agent driven Tasks are refused before the first attempt and
each group is written as one record line beside the Runs it scores.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from kullback import sampling as sampling_mod
from kullback.runner import tool as runner_tool
from kullback.runner.world.environment import BuiltEnvironment, episode_message
from kullback.runner.world.episode import Episode, EpisodeError


def rule_user(env: BuiltEnvironment, task: Any, router: Any, rules: Any, reference: Any) -> Any:
    """The Simulated user over the Task's own evidence, for trainer Episodes."""
    from kullback.user.rules import goal_write_set
    from kullback.user.simulated import SimulatedUser
    from kullback.user.value_strip import value_strip
    from kullback.user.vocabulary import GENERIC
    goal_writes = goal_write_set(reference, env.write_tools()) if reference is not None else None
    members = env.members(task)
    answer_strip = value_strip(members, schema=env.schema,
                               rules=env.canon_rules) if members else None
    vocab = env.vocab if len(env.vocab.fields) else GENERIC
    return SimulatedUser(
        rules, starting_state_reader=router.state, vocab=vocab,
        write_tools=env.write_tools(), goal_writes=goal_writes, answer_strip=answer_strip)


class Policy(Protocol):
    """What plays the Task: transcript and specs in, one assistant message out."""

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        """Answer with {"content": Optional[str], "tool_calls": list[dict]}."""
        ...


@dataclass
class Rollout:
    """One played Task: its closing transcript and what the Verifier said by code."""

    task_id: str
    seed: int
    messages: list[dict]
    reward: Optional[int]
    reward_class: str
    reason: str
    stop_reason: Optional[str]
    run_file: str
    interrupted: bool


class TaskSkipped(Exception):
    """The group refused its Task before the first attempt."""

    def __init__(self, task_id: str, reason: str):
        super().__init__(f"Task {task_id} skipped: {reason}")
        self.task_id = task_id
        self.reason = reason


SEED_KIND = "trainer_seed"


def rollout_seed(env: BuiltEnvironment, task_id: str, attempt: int) -> int:
    """The seed of one attempt, drawn from its Task and number under the build salt."""
    return sampling_mod.sample_seed(SEED_KIND, f"{task_id}-{attempt}", env.salt)


def _termination(episode: Episode) -> Optional[str]:
    """The Run's stop reason where the record can still be read, else nothing."""
    try:
        return episode.run_record().get("termination_reason")
    except Exception:
        return None


def _run_file(episode: Episode) -> str:
    path = episode.run_file()
    if path is None:
        raise EpisodeError("reset wrote no Run file")
    return str(path)


def play(env: BuiltEnvironment, task_id: str, seed: int, policy: Policy, *,
         outdir: Any, max_turns: int = 30) -> Rollout:
    """Play one Task with the policy through the Episode interface."""
    episode = Episode(env, outdir=outdir, max_turns=max_turns, user_factory=rule_user)
    info = episode.reset(task_id, seed)
    if episode.run_file() is None:
        raise EpisodeError("reset wrote no Run file")
    try:
        while True:
            try:
                reply = policy(episode.transcript(), info.tools)
                content = reply.get("content") if isinstance(reply, dict) \
                    else getattr(reply, "content", None)
                calls = reply.get("tool_calls", ()) if isinstance(reply, dict) \
                    else getattr(reply, "tool_calls", ()) or ()
            except Exception as exc:
                try:
                    episode.abort("policy_error")
                except EpisodeError:
                    pass
                return Rollout(task_id, seed, episode.transcript(), None,
                               "interrupted", str(exc), "policy_error",
                               _run_file(episode), True)
            try:
                out = episode.step(episode_message(content, calls))
            except EpisodeError:
                if _termination(episode) != "env_error":
                    try:
                        episode.abort("policy_error")
                    except EpisodeError:
                        pass
                    raise
                return Rollout(task_id, seed, episode.transcript(), None,
                               "env_error", "env_error", "env_error",
                               _run_file(episode), False)
            if out.done:
                break
        scored = episode.reward()
        return Rollout(task_id, seed, episode.transcript(), scored.score,
                        scored.class_, scored.reason, out.stop_reason,
                        _run_file(episode), False)
    except KeyboardInterrupt:
        try:
            episode.abort("cancelled")
        except EpisodeError:
            pass
        raise


def _tally(rollouts: list[Rollout]) -> tuple[dict[str, int], dict[str, int]]:
    """No signal counts by class and stop reason counts for one group."""
    no_signal: dict[str, int] = {}
    for rollout in rollouts:
        if rollout.reward is None and not rollout.interrupted:
            no_signal[rollout.reward_class] = no_signal.get(rollout.reward_class, 0) + 1
    stop_reasons: dict[str, int] = {}
    for rollout in rollouts:
        key = rollout.stop_reason if rollout.stop_reason is not None else "unknown"
        stop_reasons[key] = stop_reasons.get(key, 0) + 1
    return no_signal, stop_reasons


def _group_record(task_id: str, k: int, rollouts: list[Rollout], root: Path, *,
                  train_run_id: str, model_id: str, checkpoint: str, step: Any,
                  source: str, sampling: Any, started_at: float, ended_at: float,
                  task_origin: str = "recorded", birth_level: Any = None) -> dict:
    """The solve rate record of one group, ready to append as one JSON line."""
    no_signal, stop_reasons = _tally(rollouts)
    return {
        "format": 1,
        "train_run_id": train_run_id,
        "task_id": task_id,
        "task_origin": task_origin,
        "birth_level": birth_level,
        "model_id": model_id,
        "checkpoint": checkpoint,
        "step": step,
        "source": source,
        "k": k,
        "passes": sum(1 for rollout in rollouts if rollout.reward == 1),
        "fails": sum(1 for rollout in rollouts if rollout.reward == 0),
        "interrupted": sum(1 for rollout in rollouts if rollout.interrupted),
        "no_signal": no_signal,
        "stop_reasons": stop_reasons,
        "user_driver": "rule",
        "sampling": dict(sampling) if sampling is not None else {},
        "seeds": [rollout.seed for rollout in rollouts],
        "env_hash": {"runner_live": runner_tool.version()},
        "run_files": [os.path.relpath(rollout.run_file, root) for rollout in rollouts],
        "started_at": started_at,
        "ended_at": ended_at,
    }


def run_group(env: BuiltEnvironment, task_id: str, k: int, policy: Policy, *,
              outdir: Any, train_run_id: str, model_id: str, checkpoint: str,
              step: Any = None, source: str = "training", sampling: Any = None,
              max_turns: int = 30, task_origin: str = "recorded",
              birth_level: Any = None) -> tuple[list[Rollout], dict]:
    """Play one Task k times and append the group's record as one JSON line."""
    if env.agent_driven(task_id):
        raise TaskSkipped(task_id, "the Task is driven by the agent user, "
                                   "which needs a model the interface does not have")
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    rollouts = [play(env, task_id, rollout_seed(env, task_id, attempt), policy,
                     outdir=root, max_turns=max_turns) for attempt in range(k)]
    ended_at = time.time()
    record = _group_record(task_id, k, rollouts, root, train_run_id=train_run_id,
                           model_id=model_id, checkpoint=checkpoint, step=step,
                           source=source, sampling=sampling,
                           started_at=started_at, ended_at=ended_at,
                           task_origin=task_origin, birth_level=birth_level)
    with open(root / "solve_rates.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return rollouts, record
