"""Behaviour tests for kullback.trainer.act."""

from __future__ import annotations

import json

import pytest

from kullback.agent.bus import Bus  # noqa: E402
from kullback.runner import loop  # noqa: E402
from kullback.runner.world import BuiltEnvironment  # noqa: E402
from kullback.trainer import act  # noqa: E402
from tests.episode.invented import SCRIPTED_RENAME, write_env  # noqa: E402

NEEDS_SPLIT = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)


def ask_kwargs(**over):
    base = {"train_run_id": "r1", "model_id": "m", "checkpoint": "c",
            "step": 3, "level": "easy", "count": 2}
    base.update(over)
    return base


def rename_policy(messages, tools):
    n = sum(1 for message in messages if message.get("role") == "assistant")
    step = SCRIPTED_RENAME[min(n, len(SCRIPTED_RENAME) - 1)]
    return {"content": step.get("content"),
            "tool_calls": [dict(call) for call in step.get("tool_calls", [])]}


def test_ask_tasks_publishes_the_ask_on_first_request(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl", agent="t")
    got = act.ask_tasks(bus, **ask_kwargs())
    assert got["published"] is True
    assert got["ask"]["train_run_id"] == "r1" and got["ask"]["level"] == "easy"
    assert len(bus.replay()) == 1
    event = bus.replay()[0].event
    assert event.name == "task_ask" and event.payload["ask_id"] == got["ask"]["ask_id"]


def test_ask_tasks_refuses_while_an_ask_stays_open(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl", agent="t")
    first = act.ask_tasks(bus, **ask_kwargs())
    assert first["published"] is True
    second = act.ask_tasks(bus, **ask_kwargs(step=4))
    assert second["published"] is False and "reason" in second
    assert len(bus.replay()) == 1


def test_ask_tasks_refuses_on_a_hand_written_open_ask(tmp_path):
    ask_id = "handwritten1"
    (tmp_path / "bus.jsonl").write_text(json.dumps(
        {"seq": 1, "recorded_at": 0.0, "agent": "t",
         "event": {"type": "custom", "name": "task_ask",
                   "payload": {"train_run_id": "r1", "ask_id": ask_id}}}),
        encoding="utf-8")
    bus = Bus(tmp_path / "bus.jsonl", agent="t")
    got = act.ask_tasks(bus, **ask_kwargs())
    assert got["published"] is False and ask_id in got["reason"]
    assert len(bus.replay()) == 1


def test_ask_tasks_publishes_where_the_bus_is_empty(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl", agent="t")
    got = act.ask_tasks(bus, **ask_kwargs())
    assert got["published"] is True


def test_smoke_sample_is_stable_per_key_and_capped_at_the_pool(tmp_path):
    root = write_env(tmp_path / "env")
    write_env(root, task_id="second_task")
    write_env(root, task_id="bare_task", with_verifier=False)
    env = BuiltEnvironment(root)
    assert act.smoke_sample(env, 2, "k1") == act.smoke_sample(env, 2, "k1")
    assert sorted(act.smoke_sample(env, 9, "k1")) == ["second_task", "widget_task"]
    assert set(act.smoke_sample(env, 1, "k1")) <= {"second_task", "widget_task"}


@NEEDS_SPLIT
def test_smoke_test_plays_the_sample_and_summarises(tmp_path):
    root = write_env(tmp_path / "env")
    env = BuiltEnvironment(root)
    got = act.smoke_test(env, rename_policy, n=1, k=2, outdir=tmp_path / "out",
                         key="k1", model_id="m", checkpoint="c", train_run_id="r1")
    assert got["tasks"] == 1 and got["all_pass"] == 1
    assert got["skipped"] == 0
    assert got["seconds_per_rollout"] is not None and got["seconds_per_rollout"] >= 0
    assert got["stop_reasons"] and (tmp_path / "out" / "solve_rates.jsonl").is_file()


@NEEDS_SPLIT
def test_check_env_counts_and_rejects_the_do_nothing_run(tmp_path):
    root = write_env(tmp_path / "env")
    got = act.check_env(BuiltEnvironment(root))
    assert got == {"task_count": 1, "with_verifier": 1, "agent_driven": 0,
                   "do_nothing_task": "widget_task", "do_nothing_passes": False,
                   "do_nothing_class": "fail"}

@NEEDS_SPLIT
def test_check_env_plays_the_first_rule_driven_task(tmp_path):
    root = write_env(tmp_path / "env")
    write_env(root, task_id="second_task")
    (root / "user_fidelity.json").write_text(json.dumps({
        "format": 1,
        "tasks": [{"task_id": "second_task", "drives": True}],
    }), encoding="utf-8")
    got = act.check_env(BuiltEnvironment(root))
    assert got["do_nothing_task"] == "widget_task"
    assert got["do_nothing_passes"] is False


def test_check_env_names_no_do_nothing_task_where_all_are_agent_driven(tmp_path):
    root = write_env(tmp_path / "env")
    (root / "user_fidelity.json").write_text(json.dumps({
        "format": 1,
        "tasks": [{"task_id": "widget_task", "drives": True}],
    }), encoding="utf-8")
    assert act.check_env(BuiltEnvironment(root))["do_nothing_task"] is None


class FixedReply:
    def __init__(self, content, calls):
        self.content = content
        self.tool_calls = calls


class FixedModel:
    def __init__(self, reply):
        self.reply = reply

    def query(self, messages, tools):
        assert isinstance(messages, list) and isinstance(tools, list)
        return self.reply


def test_policy_from_model_passes_through_object_replies():
    policy = act.policy_from_model(FixedModel(FixedReply("hi", [{"id": "c1"}])))
    assert policy([], []) == {"content": "hi", "tool_calls": [{"id": "c1"}]}


def test_policy_from_model_passes_through_dict_replies():
    policy = act.policy_from_model(
        FixedModel({"content": "done", "tool_calls": []}))
    assert policy([], []) == {"content": "done", "tool_calls": []}
