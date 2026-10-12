"""The re-roll user and Reference choice the Builder's run tool uses (moved from the Examiner, D320)."""

from __future__ import annotations

import json
from pathlib import Path

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.ai.usage import Usage
from kullback.builder import run_user as R
from kullback.runner import budget, tool
from kullback.runner.records import Verifier, write_json
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



def _priced_script():
    """The rename script under a priced model id, each turn carrying a provider's usage."""
    usage = Usage(input=1890, output=17)
    return TestModel([reply.model_copy(update={"model": None, "usage": usage})
                      for reply in _rename_script().replies], loop=True, name=next(iter(budget.PRICES)))



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


