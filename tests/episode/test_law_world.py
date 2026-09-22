"""One Task's built Environment as a world the laws are checked against.

An EnvironmentLawWorld adapts one Task, so check_laws runs over the real
toolkit: reset rebuilds the Starting state, tools names the tools with argument
shapes from the Task's own recorded calls, call routes through the Run's
Router, and snapshot returns the state the laws compare.
"""

from __future__ import annotations

import json

import pytest

from kullback.episode.environment import BuiltEnvironment
from kullback.episode.law_world import EnvironmentLawWorld
from kullback.laws import ArgSpec, Outcome, ToolInfo, check_laws
from kullback.runner import loop
from kullback.runner.records import RawPtr, ToolCall, Trace
from tests.episode.invented import write_env


def _trace(root, trace_id, calls):
    """One Trace file of recorded calls, the shape the Task's members read."""
    ptr = RawPtr(file_hash="invented")
    trace = Trace(trace_id=trace_id, raw_hash="invented", ingest_version="1", source="invented",
                  raw_ptr=ptr,
                  tool_calls=[ToolCall(name=name, args=dict(args), raw_ptr=ptr)
                              for name, args in calls])
    folder = root / "traces"
    folder.mkdir(exist_ok=True)
    (folder / f"{trace_id}.json").write_text(trace.model_dump_json(), encoding="utf-8")


def test_check_laws_runs_over_the_built_fixture_and_reports_a_number(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "work"))
    report = check_laws(EnvironmentLawWorld(env, "widget_task"), seed=7, sequences=6, max_length=3)
    assert report.consistency == 1.0
    assert sum(report.checked.values()) > 0
    assert sum(report.broken.values()) == 0


@pytest.mark.skipif(not hasattr(loop, "advance"), reason="needs the step-split patch")
def test_reset_matches_the_episode_starting_state(tmp_path):
    from kullback.episode import Episode

    env = BuiltEnvironment(write_env(tmp_path / "work"))
    episode = Episode(env, outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    world = EnvironmentLawWorld(env, "widget_task", seed=7)
    world.reset()
    assert world.snapshot() == episode._router.world()


def test_call_answers_a_recorded_read_from_the_starting_state(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "work"))
    world = EnvironmentLawWorld(env, "widget_task")
    world.reset()
    outcome = world.call("describe_widget", {"widget_id": "w1"})
    assert outcome.ok is True
    assert outcome.value == {"widget_id": "w1", "label": "plain"}


def test_call_refuses_an_unknown_tool_and_a_missing_row(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "work"))
    world = EnvironmentLawWorld(env, "widget_task")
    world.reset()
    missing = world.call("describe_widget", {"widget_id": "nowhere"})
    assert missing.ok is False and missing.error
    unknown = world.call("no_such_tool", {})
    assert unknown.ok is False and unknown.error


def test_tools_draw_argument_shapes_from_the_task_traces(tmp_path):
    root = write_env(tmp_path / "work")
    _trace(root, "rec1", [
        ("describe_widget", {"widget_id": "w1"}),
        ("rename_widget", {"widget_id": "w1", "label": "striped", "n": 3,
                           "score": 1.5, "chosen": True}),
        ("rename_widget", {"widget_id": "w1", "n": 4, "score": 2.5, "chosen": False}),
    ])
    world = EnvironmentLawWorld(BuiltEnvironment(root), "widget_task")
    shapes = {info.name: (info.kind, [(spec.name, spec.type, spec.optional)
                                     for spec in info.args])
              for info in world.tools()}
    assert shapes["describe_widget"] == ("read", [("widget_id", "str", False)])
    assert shapes["rename_widget"] == ("write", [("chosen", "bool", False),
                                                ("label", "str", True),
                                                ("n", "int", False),
                                                ("score", "float", False),
                                                ("widget_id", "str", False)])


def test_tools_without_recorded_calls_list_names_with_no_arguments(tmp_path):
    root = write_env(tmp_path / "work")
    (root / "tasks" / "ghost.json").write_text(json.dumps({
        "id": "ghost", "run_ids": ["ghost"], "intent": "nothing recorded",
    }), encoding="utf-8")
    world = EnvironmentLawWorld(BuiltEnvironment(root), "ghost")
    assert [(info.name, info.kind, list(info.args)) for info in world.tools()] == [
        ("describe_widget", "read", []), ("rename_widget", "write", [])]


class _CountingReadWorld:
    """A read that counts its own calls inside the snapshot it returns."""

    def __init__(self):
        self._reads = 0

    def reset(self):
        self._reads = 0

    def tools(self):
        return [ToolInfo("scan", "read", [ArgSpec("slot", "int")])]

    def snapshot(self):
        return {"reads": self._reads}

    def call(self, name, args):
        self._reads += 1
        return Outcome(True, None)


def test_flawed_tool_world_reports_consistency_below_one():
    report = check_laws(_CountingReadWorld(), seed=11, sequences=8, max_length=3)
    assert report.consistency is not None
    assert report.consistency < 1.0
    assert sum(report.broken.values()) > 0


def test_built_world_with_noop_write_reports_consistency_below_one(tmp_path):
    from kullback.laws import SUCCESS_CHANGED_SOMETHING
    from tests.episode.invented import TOOLS

    root = write_env(tmp_path / "work")
    (root / "env" / "tools.py").write_text(
        TOOLS.replace("        row.label = label\n", ""), encoding="utf-8")
    _trace(root, "rec1", [("rename_widget", {"widget_id": "w1", "label": "striped"})])
    world = EnvironmentLawWorld(BuiltEnvironment(root), "widget_task")
    report = check_laws(world, seed=7, sequences=10, max_length=3)
    assert report.consistency is not None
    assert report.consistency < 1.0
    assert [(v.law, v.tool) for v in report.violations] == [
        (SUCCESS_CHANGED_SOMETHING, "rename_widget")]
