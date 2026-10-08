"""Behaviour tests for kullback.trainer.tools and the push backend."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from kullback.agent.bus import Bus  # noqa: E402
from kullback.ai.provider import TestModel  # noqa: E402
from kullback.runner import loop  # noqa: E402
from kullback.runner.world import BuiltEnvironment  # noqa: E402
from kullback.trainer import act  # noqa: E402
from kullback.trainer import remote as remote_mod  # noqa: E402
from kullback.trainer.remote import RemoteConfig  # noqa: E402
from kullback.trainer.tools import TrainerContext, trainer_tools  # noqa: E402
from tests.episode.invented import SCRIPTED_RENAME, write_env  # noqa: E402

NEEDS_SPLIT = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)


def call(tool, args):
    result = asyncio.run(tool.run(args))
    assert not result.is_error, result.content
    return result.details


def bare_ctx(workdir, **over):
    base = {"workdir": workdir, "train_run_id": "r1"}
    base.update(over)
    return TrainerContext(**base)


def by_name(ctx):
    return {tool.name: tool for tool in trainer_tools(ctx)}


def make_cfg(root):
    root.mkdir(parents=True, exist_ok=True)
    return RemoteConfig(host="local", root=str(root), python=sys.executable)


def rename_policy(messages, tools):
    n = sum(1 for message in messages if message.get("role") == "assistant")
    step = SCRIPTED_RENAME[min(n, len(SCRIPTED_RENAME) - 1)]
    return {"content": step.get("content"),
            "tool_calls": [dict(call) for call in step.get("tool_calls", [])]}

def test_failed_sync_answers_sync_error(tmp_path):
    cfg = make_cfg(tmp_path / "remote")
    ctx = bare_ctx(tmp_path / "work", cfg=cfg, train_run_id="ghost")
    tools = by_name(ctx)
    assert call(tools["training_status"], {})["sync_error"].strip() != ""
    assert call(tools["metrics"], {})["sync_error"].strip() != ""


def rate_line(task_id, step, k, passes, fails, started, ended):
    return json.dumps({
        "format": 1, "train_run_id": "r1", "task_id": task_id,
        "task_origin": "recorded", "model_id": "m", "checkpoint": "c",
        "step": step, "source": "training", "k": k, "passes": passes,
        "fails": fails, "interrupted": 0, "no_signal": {},
        "stop_reasons": {"done": k}, "user_driver": "rule", "sampling": {},
        "seeds": list(range(k)), "env_hash": {}, "run_files": [],
        "started_at": started, "ended_at": ended,
    })


def test_tools_without_config_name_remote_json(tmp_path):
    tools = by_name(bare_ctx(tmp_path))
    assert "remote.json" in call(tools["gpu_status"], {})["error"]
    assert "remote.json" in call(tools["push"], {})["error"]
    assert "remote.json" in call(tools["start_run"], {"script": "scripts/x.py"})["error"]
    assert "remote.json" in call(tools["stop_run"], {})["error"]
    assert "remote.json" in call(tools["sync"], {})["error"]
    assert "remote.json" in call(tools["shell"], {"command": "echo hi"})["error"]


def test_training_status_and_metrics_read_synced_files(tmp_path):
    root = tmp_path / "remote"
    folder = root / "runs" / "r1"
    folder.mkdir(parents=True)
    (folder / "state.json").write_text(
        json.dumps({"status": "running", "step": 2, "checkpoint": "c", "updated_at": 1.0}),
        encoding="utf-8")
    (folder / "run.log").write_text("line1\nline2\n", encoding="utf-8")
    (folder / "solve_rates.jsonl").write_text(
        rate_line("t1", 1, 4, 2, 2, 0.0, 60.0) + "\n"
        + rate_line("t1", 2, 4, 3, 1, 60.0, 120.0) + "\n", encoding="utf-8")
    (folder / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "kl": 0.5}) + "\n" + json.dumps({"step": 2, "kl": 0.4}) + "\n",
        encoding="utf-8")
    ctx = bare_ctx(tmp_path / "work", cfg=make_cfg(root))
    tools = by_name(ctx)
    status = call(tools["training_status"], {})
    assert status["step"] == 2 and status["log_tail"] == ["line1", "line2"]
    assert (ctx.workdir / "training" / "r1" / "state.json").is_file()
    got = call(tools["metrics"], {})
    assert got["bands"]["tasks"] == 1 and got["trend"]["kl"]["last"] == 0.4


def test_ask_tasks_publishes_once_then_refuses_while_open(tmp_path):
    ctx = bare_ctx(tmp_path, bus=Bus(tmp_path / "bus.jsonl", agent="trainer"),
                   policy_model_id="m")
    tools = by_name(ctx)
    args = {"level": "easy", "count": 2, "step": 3, "checkpoint": "c"}
    assert call(tools["ask_tasks"], args)["published"] is True
    assert call(tools["ask_tasks"], args)["published"] is False
    assert len(ctx.bus.replay()) == 1


def test_finding_appends_one_line(tmp_path):
    ctx = bare_ctx(tmp_path)
    got = call(by_name(ctx)["finding"], {"kind": "note", "text": "hello"})
    assert got["written"] is True
    lines = (tmp_path / "training" / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    body = json.loads(lines[0])
    assert (body["kind"], body["text"], body["run_id"]) == ("note", "hello", "r1")


def test_push_copies_scripts_to_a_local_root(tmp_path):
    scripts = tmp_path / "work" / "training" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "go.py").write_text("print('go')\n", encoding="utf-8")
    cfg = make_cfg(tmp_path / "remote")
    got = remote_mod.push(cfg, tmp_path / "work")
    assert got["copied"] == 1 and (tmp_path / "remote" / "scripts" / "go.py").is_file()


def test_push_of_a_missing_subdir_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path / "remote")
    got = remote_mod.push(cfg, tmp_path / "work")
    assert got["copied"] == 0 and got["error"].strip() != ""


def test_start_run_then_stop_run_through_the_tools(tmp_path):
    scripts = tmp_path / "work" / "training" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "hello.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    cfg = make_cfg(tmp_path / "remote")
    tools = by_name(bare_ctx(tmp_path / "work", cfg=cfg))
    try:
        started = call(tools["start_run"], {"run_id": "abc123", "script": "scripts/hello.py"})
        assert started["started"] is True and started["push"]["copied"] == 1
        stopped = call(tools["stop_run"], {"run_id": "abc123"})
        assert stopped["stopped"] is True
    finally:
        remote_mod.stop_run(cfg, "abc123", workdir=tmp_path / "work")


def test_shell_runs_on_a_local_root(tmp_path):
    ctx = bare_ctx(tmp_path, cfg=make_cfg(tmp_path / "remote"))
    got = call(by_name(ctx)["shell"], {"command": "echo hi"})
    assert got["returncode"] == 0 and got["stdout"].strip() == "hi"


def test_read_run_reads_the_last_200_lines(tmp_path):
    folder = tmp_path / "training" / "r1"
    folder.mkdir(parents=True)
    (folder / "run.log").write_text("\n".join(f"line {n}" for n in range(250)) + "\n",
                                    encoding="utf-8")
    tools = by_name(bare_ctx(tmp_path))
    got = call(tools["read_run"], {"name": "run.log"})
    assert (got["total_lines"], len(got["lines"])) == (250, 200)
    assert got["lines"][-1] == "line 249"
    assert "one of" in call(tools["read_run"], {"name": "nope.json"})["error"]


def test_runs_task_pool_and_synth_status_read_local_shapes(tmp_path):
    for name in ("b", "a"):
        folder = tmp_path / "training" / name
        folder.mkdir(parents=True)
        (folder / "state.json").write_text("{}", encoding="utf-8")
    write_env(tmp_path)
    env = BuiltEnvironment(tmp_path)
    tools = by_name(bare_ctx(tmp_path, env=env))
    assert call(tools["runs"], {}) == {"runs": ["a", "b"]}
    assert call(tools["task_pool"], {})["total"] == 1
    assert call(tools["synth_status"], {}) == {"asked": 0, "answered": 0, "open_ask_ids": []}

def test_observe_tools_without_an_environment_name_it(tmp_path):
    tools = by_name(bare_ctx(tmp_path))
    assert "no built Environment" in call(tools["task_pool"], {})["error"]
    assert "no built Environment" in call(tools["check_env"], {})["error"]


@NEEDS_SPLIT
def test_check_env_through_the_tool_counts_and_rejects(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "env"))
    got = call(by_name(bare_ctx(tmp_path, env=env))["check_env"], {})
    assert got["task_count"] == 1 and got["do_nothing_passes"] is False


@NEEDS_SPLIT
def test_smoke_test_through_the_tool_with_a_scripted_policy(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "env"))
    ctx = bare_ctx(tmp_path, env=env, policy=rename_policy, policy_model_id="m")
    got = call(by_name(ctx)["smoke_test"], {"n": 1, "k": 1, "key": "k1", "checkpoint": "c1"})
    assert got["tasks"] == 1
    assert (tmp_path / "training" / "smoke" / "solve_rates.jsonl").is_file()


def test_smoke_test_without_a_policy_names_it(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "env"))
    got = call(by_name(bare_ctx(tmp_path, env=env))["smoke_test"], {})
    assert "policy" in got["error"]


def test_policy_from_model_hands_the_episode_plain_dicts():
    model = TestModel([{"content": None, "tool_calls": [
        {"id": "c1", "name": "rename_widget",
         "arguments": {"widget_id": "w1", "label": "striped"}}]}])
    reply = act.policy_from_model(model)([], [])
    assert reply["content"] is None
    assert reply["tool_calls"] == [{"id": "c1", "name": "rename_widget",
                                   "arguments": {"widget_id": "w1", "label": "striped"}}]
    assert all(type(call) is dict for call in reply["tool_calls"])


def test_run_id_tools_refuse_an_id_that_escapes_training(tmp_path):
    tools = by_name(bare_ctx(tmp_path))
    assert "bad run_id" in call(tools["read_run"], {"run_id": "../..", "name": "run.log"})["error"]
    assert "bad run_id" in call(tools["training_status"], {"run_id": "../.."})["error"]
    assert "bad run_id" in call(tools["metrics"], {"run_id": "../.."})["error"]
    assert not (tmp_path / "training").exists()
    scripts = tmp_path / "work" / "training" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "go.py").write_text("print('go')\n", encoding="utf-8")
    gated = by_name(bare_ctx(tmp_path / "work", cfg=make_cfg(tmp_path / "remote")))
    assert "bad run_id" in call(
        gated["start_run"], {"run_id": "../..", "script": "scripts/go.py"})["error"]
    assert "bad run_id" in call(gated["stop_run"], {"run_id": "../.."})["error"]
    assert "bad run_id" in call(gated["sync"], {"run_id": "../.."})["error"]
    assert not (tmp_path / "remote" / "scripts").exists()
    assert not (tmp_path / "remote" / "runs").exists()


def test_start_run_refuses_when_push_fails(tmp_path):
    cfg = make_cfg(tmp_path / "remote")
    got = call(by_name(bare_ctx(tmp_path / "work", cfg=cfg))["start_run"],
               {"run_id": "r9", "script": "scripts/go.py"})
    assert got["started"] is False
    assert got["reason"].startswith("push failed:")
    assert got["push"]["error"].strip() != ""
    assert not (tmp_path / "remote" / "runs").exists()
