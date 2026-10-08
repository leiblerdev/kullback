"""One Task played k times under the rule user, scored by code, written as one line (G1).

Needs the step-split patch (docs/frozen-patches/step-split.patch): the rollout plays the
Task through the Episode interface, so this module skips unless the patch is applied.
"""

from __future__ import annotations

import json

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)

from kullback import curriculum  # noqa: E402
from kullback.runner.world import BuiltEnvironment  # noqa: E402
from kullback.trainer.rollout import TaskSkipped, play, rollout_seed, run_group  # noqa: E402
from tests.episode.invented import SCRIPTED_RENAME, write_env  # noqa: E402


def rename_policy(messages, tools):
    """Replay the scripted rename: the turn count picks the scripted line."""
    n = sum(1 for message in messages if message.get("role") == "assistant")
    step = SCRIPTED_RENAME[min(n, len(SCRIPTED_RENAME) - 1)]
    return {"content": step.get("content"),
            "tool_calls": [dict(call) for call in step.get("tool_calls", [])]}


def done_policy(messages, tools):
    return {"content": "done", "tool_calls": []}


def boom_policy(messages, tools):
    raise RuntimeError("boom")


def test_scripted_rename_policy_passes_with_reward_one(tmp_path):
    root = write_env(tmp_path / "env")
    rollout = play(BuiltEnvironment(root), "widget_task", 7, rename_policy,
                   outdir=tmp_path / "out")
    assert rollout.reward == 1
    assert rollout.reward_class == "pass"
    assert not rollout.interrupted


def test_done_only_policy_fails_with_zero(tmp_path):
    root = write_env(tmp_path / "env")
    rollout = play(BuiltEnvironment(root), "widget_task", 7, done_policy,
                   outdir=tmp_path / "out", max_turns=5)
    assert rollout.reward == 0
    assert not rollout.interrupted


def test_raising_policy_counts_interrupted_and_aborts_the_run(tmp_path):
    from pathlib import Path

    root = write_env(tmp_path / "env")
    rollout = play(BuiltEnvironment(root), "widget_task", 7, boom_policy,
                   outdir=tmp_path / "out")
    assert rollout.reward is None
    assert rollout.reward_class == "interrupted"
    assert rollout.interrupted
    assert rollout.stop_reason == "policy_error"
    assert "policy_error" in Path(rollout.run_file).read_text(encoding="utf-8")


def test_group_of_three_writes_one_line_with_counts_adding_to_k(tmp_path):
    root = write_env(tmp_path / "env")
    outdir = tmp_path / "out"
    rollouts, record = run_group(BuiltEnvironment(root), "widget_task", 3, rename_policy,
                                 outdir=outdir, train_run_id="tr1",
                                 model_id="m1", checkpoint="c1")
    assert len(rollouts) == 3
    assert record["passes"] + record["fails"] + sum(record["no_signal"].values()) \
        + record["interrupted"] == 3
    assert len(set(record["seeds"])) == 3
    assert record["user_driver"] == "rule"
    assert len(record["run_files"]) == 3
    for entry in record["run_files"]:
        assert isinstance(entry, str)
        assert (outdir / entry).is_file()
    lines = (outdir / "solve_rates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["passes"] == record["passes"]


def test_same_task_and_attempt_draws_the_same_seed(tmp_path):
    root = write_env(tmp_path / "env")
    env = BuiltEnvironment(root)
    assert rollout_seed(env, "widget_task", 0) == rollout_seed(env, "widget_task", 0)
    assert rollout_seed(env, "widget_task", 0) != rollout_seed(env, "widget_task", 1)


def test_agent_driven_task_raises_task_skipped(tmp_path):
    root = write_env(tmp_path / "env")
    (root / "user_fidelity.json").write_text(json.dumps({
        "format": 1,
        "tasks": [{"task_id": "widget_task", "drives": True}],
    }), encoding="utf-8")
    with pytest.raises(TaskSkipped):
        run_group(BuiltEnvironment(root), "widget_task", 1, rename_policy,
                  outdir=tmp_path / "out", train_run_id="tr1",
                  model_id="m1", checkpoint="c1")


def test_group_record_carries_origin_and_birth_level(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "env"))
    _, record = run_group(env, "widget_task", 1, rename_policy,
                          outdir=tmp_path / "out", train_run_id="tr1",
                          model_id="m1", checkpoint="c1",
                          task_origin="synthetic", birth_level="hard")
    assert record["task_origin"] == "synthetic" and record["birth_level"] == "hard"
    assert curriculum.SolveRate(**record).birth_level == "hard"
    _, default = run_group(env, "widget_task", 1, rename_policy,
                           outdir=tmp_path / "out2", train_run_id="tr1",
                           model_id="m1", checkpoint="c1")
    assert default["task_origin"] == "recorded" and default["birth_level"] is None
