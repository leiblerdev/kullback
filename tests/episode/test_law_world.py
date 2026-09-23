"""One Task's built Environment as a world the laws are checked against.

An EnvironmentLawWorld adapts one Task, so check_laws runs over the real
toolkit: reset rebuilds the Starting state, tools names the tools with argument
shapes from the Task's own recorded calls, call routes through the Run's
Router, and snapshot returns the state the laws compare.
"""

from __future__ import annotations

import json

import pytest

from kullback.laws import (
    READ_CHANGES_NOTHING,
    SUCCESS_CHANGED_SOMETHING,
    check_laws,
)
from kullback.runner import loop
from kullback.runner.records import RawPtr, ToolCall, Trace
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.law_world import EnvironmentLawWorld
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


def _write_sigs(root, tools):
    """tool_sigs.json from (name, kind, [(arg, types, optional)]) triples, the miner's shape."""
    sigs = []
    for name, kind, fields in tools:
        sigs.append({
            "name": name, "kind": kind, "unclassified": False,
            "args_fields": [{"name": arg, "types": list(types), "optional": optional}
                            for arg, types, optional in fields],
            "args_schema": {
                "type": "object",
                "properties": {arg: ({"type": list(types)} if types else {})
                               for arg, types, _ in fields},
                "required": [arg for arg, _, optional in fields if not optional],
            },
        })
    (root / "tool_sigs.json").write_text(json.dumps(sigs), encoding="utf-8")


def test_check_laws_runs_over_the_built_fixture_and_reports_a_number(tmp_path):
    env = BuiltEnvironment(write_env(tmp_path / "work"))
    report = check_laws(EnvironmentLawWorld(env, "widget_task"), seed=7, sequences=6, max_length=3)
    assert report.consistency == 1.0
    assert sum(report.checked.values()) > 0
    assert sum(report.broken.values()) == 0


@pytest.mark.skipif(not hasattr(loop, "advance"), reason="needs the step-split patch")
def test_reset_matches_the_episode_starting_state(tmp_path):
    from kullback.runner.world import Episode

    env = BuiltEnvironment(write_env(tmp_path / "work"))
    episode = Episode(env, outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    world = EnvironmentLawWorld(env, "widget_task", seed=7)
    world.reset()
    assert world.snapshot() == episode._router.world()


@pytest.mark.parametrize("name,args,value", [
    ("describe_widget", {"widget_id": "w1"}, {"widget_id": "w1", "label": "plain"}),
    ("describe_widget", {"widget_id": "nowhere"}, None),
    ("no_such_tool", {}, None),
])
def test_call_answers_a_recorded_read_and_refuses_an_unknown_tool_and_a_missing_row(tmp_path, name, args, value):
    env = BuiltEnvironment(write_env(tmp_path / "work"))
    world = EnvironmentLawWorld(env, "widget_task")
    world.reset()
    outcome = world.call(name, args)
    if value is None:
        assert outcome.ok is False and outcome.error
    else:
        assert outcome.ok is True
        assert outcome.value == value


def _recorded_extra_args(root):
    _trace(root, "rec1", [
        ("describe_widget", {"widget_id": "w1"}),
        ("rename_widget", {"widget_id": "w1", "label": "striped", "n": 3}),
    ])
    return "widget_task"


def _nothing_recorded(root):
    (root / "tasks" / "ghost.json").write_text(json.dumps({
        "id": "ghost", "run_ids": ["ghost"], "intent": "nothing recorded",
    }), encoding="utf-8")
    return "ghost"


def _required_but_not_recorded(root):
    _write_sigs(root, [
        ("describe_widget", "read", [("widget_id", [], False)]),
        ("rename_widget", "write", [("widget_id", ["str"], False), ("label", ["str"], False)]),
    ])
    _trace(root, "rec1", [("describe_widget", {"widget_id": "w1"})])
    return "widget_task"


OPTIONAL_STRINGS = {"describe_widget": ("read", [("widget_id", "string", True)]),
                    "rename_widget": ("write", [("label", "string", True), ("widget_id", "string", True)])}


@pytest.mark.parametrize("arrange,shapes", [
    (_recorded_extra_args, OPTIONAL_STRINGS),
    (_nothing_recorded, OPTIONAL_STRINGS),
    (_required_but_not_recorded, {
        "describe_widget": ("read", [("widget_id", "str", False)]),
        "rename_widget": ("write", [("label", "str", False), ("widget_id", "str", False)])}),
])
def test_tools_take_names_types_and_required_flags_from_the_signature_not_the_recordings(
        tmp_path, arrange, shapes):
    root = write_env(tmp_path / "work")
    task_id = arrange(root)
    world = EnvironmentLawWorld(BuiltEnvironment(root), task_id)
    assert {info.name: (info.kind, [(spec.name, spec.type, spec.optional) for spec in info.args])
            for info in world.tools()} == shapes
    if arrange is _required_but_not_recorded:
        report = check_laws(world, seed=7, sequences=10, max_length=3)
        assert report.checked.get((SUCCESS_CHANGED_SOMETHING, "rename_widget"), 0) > 0


def _set_callers(root, callers_by_tool):
    """Rewrite tool_sigs.json with a callers list per tool, the miner's requestor shape."""
    sigs = json.loads((root / "tool_sigs.json").read_text(encoding="utf-8"))
    for sig in sigs:
        if sig["name"] in callers_by_tool:
            sig["callers"] = callers_by_tool[sig["name"]]
    (root / "tool_sigs.json").write_text(json.dumps(sigs), encoding="utf-8")


def _untyped_required(root):
    _write_sigs(root, [
        ("describe_widget", "read", [("widget_id", [], False)]),
        ("rename_widget", "write", [("widget_id", ["str"], False), ("label", ["str"], False)]),
    ])


@pytest.mark.parametrize("arrange,reason", [
    (_untyped_required, "required argument widget_id has no type"),
    (lambda root: _set_callers(root, {"describe_widget": []}), "tool is listed for neither side"),
])
def test_untyped_required_argument_or_no_caller_side_makes_the_tool_unfit(tmp_path, arrange, reason):
    root = write_env(tmp_path / "work")
    arrange(root)
    world = EnvironmentLawWorld(BuiltEnvironment(root), "widget_task")
    assert world.unfit() == {"describe_widget": reason}
    assert [info.name for info in world.tools()] == ["rename_widget"]


class _CallLog:
    """A LawWorld wrapper that logs whether each call answered or refused."""

    def __init__(self, inner):
        self._inner = inner
        self.answered = []

    def reset(self):
        self._inner.reset()

    def tools(self):
        return self._inner.tools()

    def snapshot(self):
        return self._inner.snapshot()

    def unfit(self):
        return self._inner.unfit()

    def call(self, name, args):
        outcome = self._inner.call(name, args)
        self.answered.append((name, outcome.ok))
        return outcome


def test_user_side_tool_is_called_under_its_own_requestor_and_checked_by_the_laws(tmp_path):
    root = write_env(tmp_path / "work")
    _set_callers(root, {"describe_widget": ["user"]})
    world = _CallLog(EnvironmentLawWorld(BuiltEnvironment(root), "widget_task"))
    report = check_laws(world, seed=7, sequences=6, max_length=3)
    assert report.checked.get((READ_CHANGES_NOTHING, "describe_widget"), 0) > 0
    assert any(name == "describe_widget" and ok for name, ok in world.answered)
    direct = EnvironmentLawWorld(BuiltEnvironment(root), "widget_task")
    direct.reset()
    outcome = direct.call("describe_widget", {"widget_id": "w1"})
    assert outcome.ok is True
    assert outcome.value == {"widget_id": "w1", "label": "plain"}


def test_write_needing_an_existing_id_reaches_a_successful_write(tmp_path):
    world = EnvironmentLawWorld(BuiltEnvironment(write_env(tmp_path / "work")), "widget_task")
    report = check_laws(world, seed=2, sequences=6, max_length=3)
    assert report.checked.get((SUCCESS_CHANGED_SOMETHING, "rename_widget"), 0) > 0


def test_built_world_with_noop_write_reports_consistency_below_one(tmp_path):
    from tests.episode.invented import TOOLS

    root = write_env(tmp_path / "work")
    (root / "env" / "tools.py").write_text(
        TOOLS.replace("        row.label = label\n", ""), encoding="utf-8")
    _write_sigs(root, [
        ("describe_widget", "read", [("widget_id", ["str"], False)]),
        ("rename_widget", "write", [("widget_id", ["str"], False), ("label", ["str"], False)]),
    ])
    world = EnvironmentLawWorld(BuiltEnvironment(root), "widget_task")
    report = check_laws(world, seed=7, sequences=10, max_length=3)
    assert report.consistency is not None
    assert report.consistency < 1.0
    assert (SUCCESS_CHANGED_SOMETHING, "rename_widget") in [
        (v.law, v.tool) for v in report.violations]
