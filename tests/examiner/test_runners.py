"""The Examiner's runners over the runner tool: probe, re-rolls, variants, and the no-touch rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.ai.usage import Usage
from kullback.examiner import runners as R
from kullback.runner import budget, tool
from kullback.runner.records import Verifier, write_json
from kullback.runner.target import check_run
from tests.runner.test_tool import _env_with_trace


def _rename_script():
    """A model that reads the widget and writes the striped label, then stops."""
    return TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)


def _exam_env(root: Path) -> Path:
    """The invented environment with one confirmed replay row to seed from."""
    env = _env_with_trace(root)
    seed = tool.run(env, "widget_task", _rename_script(), workdir=root)
    write_json(root / "replays.json", {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": seed.run_id,
                 "confirmed": True, "path": seed.path}}})
    return env


def _verifier(root: Path) -> Verifier:
    return Verifier.model_validate(json.loads(
        (root / "verifiers" / "widget_task.json").read_text(encoding="utf-8")))


def test_run_probe_returns_the_scored_run_and_files_it_under_probes(tmp_path):
    """Check 6 through the runner: the probe Run passes the Verifier and never lands in runs/."""
    root = _exam_env(tmp_path / "env")
    runners = R.runners_for(root)
    run = runners["run_probe"](_rename_script(), _verifier(root))
    assert run.run_id == "probe-widget_task"
    assert (root / "probes" / "probe-widget_task.jsonl").is_file()
    assert list((root / "runs").glob("probe-*.jsonl")) == []
    passed, _ = check_run(_verifier(root), run)
    assert passed is True


def test_run_rerolls_buys_prefixed_rows_that_carry_user_end(tmp_path):
    """Bought re-rolls file under the buyer's prefix with the rows the derivation keeps."""
    root = _exam_env(tmp_path / "env")
    runners = R.runners_for(root, reroll_model=_rename_script())
    rows = runners["run_rerolls"]("widget_task", 2, "second-path-r0-b0")
    assert [row["run_id"] for row in rows] == ["second-path-r0-b0-0", "second-path-r0-b0-1"]
    assert all(Path(row["path"]).is_file() for row in rows)
    assert all(set(row) >= {"run_id", "path", "termination_reason", "user_end"} for row in rows)
    assert all(row["user_end"] is not None for row in rows)


def _priced_script():
    """The rename script under a priced model id, each turn carrying a provider's usage."""
    usage = Usage(input=1890, output=17)
    return TestModel([reply.model_copy(update={"model": None, "usage": usage})
                      for reply in _rename_script().replies], loop=True, name=next(iter(budget.PRICES)))


def test_examiner_rerolls_and_probes_are_priced_under_the_runner_stage(tmp_path):
    """Every model call of an Examiner Run lands in budget.json under runner, with its usd (F29)."""
    root = _exam_env(tmp_path / "env")
    reroll = _priced_script()
    runners = R.runners_for(root, reroll_model=reroll)
    runners["run_rerolls"]("widget_task", 1, "second-path-r0-b0")
    stage = budget.load_totals(root)["stages"]["runner"]
    assert stage["calls"] == len(reroll.calls) > 0 and stage["usd"] > 0
    probe = _priced_script()
    runners["run_probe"](probe, _verifier(root))
    assert budget.load_totals(root)["stages"]["runner"]["calls"] == len(reroll.calls) + len(probe.calls)


def test_run_rerolls_without_a_model_is_refused(tmp_path):
    """With no re-roll model there is nothing to buy with: a loud refusal, not silent Runs."""
    root = _exam_env(tmp_path / "env")
    runners = R.runners_for(root)
    with pytest.raises(ValueError, match="no model to re-roll with"):
        runners["run_rerolls"]("widget_task", 1, "second-path-r0-b0")


def test_run_variant_replays_a_rewritten_path_and_indexes_it(tmp_path):
    """A code-written call path replays cost-free and the caller indexes it afterwards."""
    root = _exam_env(tmp_path / "env")
    runners = R.runners_for(root)
    calls = [
        {"id": "c1", "name": "describe_widget", "args": {"widget_id": "w1"},
         "requestor": "assistant", "turn": 0},
        {"id": "c2", "name": "rename_widget", "args": {"widget_id": "w1", "label": "striped"},
         "requestor": "assistant", "turn": 0},
    ]
    row = runners["run_variant"]("widget_task", calls, "synth-r0-widget_task-1", ())
    assert row is not None and set(row) >= {"run_id", "path", "termination_reason", "crashed"}
    assert row["run_id"] == "synth-r0-widget_task-1" and Path(row["path"]).is_file()
    runners["run_variant"].write_runs_index()
    indexed = {run["run_id"] for run in json.loads(
        (root / "runs.json").read_text(encoding="utf-8"))["runs"]}
    assert "synth-r0-widget_task-1" in indexed


def test_held_out_runs_never_seed_the_reference(tmp_path):
    """D81 through the anchor: the Reference is the earliest confirmed run not held out."""
    root = _exam_env(tmp_path / "env")
    (root / "user_rules" / "rec2.json").write_text(
        (root / "user_rules" / "rec1.json").read_text(encoding="utf-8"), encoding="utf-8")
    rows = {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": "rec1", "confirmed": True, "path": "runs/rec1.json"},
        "rec2": {"trace_id": "rec2", "run_id": "rec2", "confirmed": True, "path": "runs/rec2.json"}}}
    write_json(root / "replays.json", rows)
    assert R._reference_id(root, "widget_task", ["rec1", "rec2"], None) == "rec1"
    assert R._reference_id(root, "widget_task", ["rec1", "rec2"],
                           {"held_out": {"widget_task": ["rec1"]}}) == "rec2"
    assert R._seed_ids({"held_out": {"widget_task": ["rec1"]}}, "widget_task",
                       ["rec1", "rec2"]) == ["rec2"]


def test_runners_module_never_names_the_compiled_side():
    """D123 as a source scan: the runner reads the compiled side, the Examiner does not."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    for name in ("bodies.json", "db.json", "schema.json", "env/"):
        assert name not in source, name


def test_probe_prompt_names_the_end_state_and_refuses_the_policy_step(tmp_path):
    """The carried probe prompt: the End state verbatim, reached without asking or verifying."""
    root = _exam_env(tmp_path / "env")
    prompt = tool._probe_prompt
    verifier = _verifier(root)
    from kullback.runner.world.environment import BuiltEnvironment
    task = BuiltEnvironment(root).task("widget_task")
    text = prompt(task, verifier, BuiltEnvironment(root).sigs)
    assert "striped" in text and "###STOP###" in text
    assert "Do not ask the user" in text
