"""Every JSON file the Environment reads obeys one loader rule (G1).

Missing where optional falls back; present but torn or not the expected shape
raises EnvironmentError naming the path and what was expected.
"""

import json

import pytest

from kullback.episode.environment import BuiltEnvironment, EnvironmentError
from tests.episode.invented import write_env

OBJECT_BODIES = ["null", "[]", '"label"', "4"]
ARRAY_BODIES = ["null", "{}", '"label"', "4"]

INIT_FILES = [
    ("environment.json", OBJECT_BODIES, "not an object"),
    ("schema.json", OBJECT_BODIES, "not an object"),
    ("tool_sigs.json", ARRAY_BODIES, "not an array"),
    ("bodies.json", OBJECT_BODIES, "not an object"),
    ("anchor.json", OBJECT_BODIES, "not an object"),
    ("replays.json", OBJECT_BODIES, "not an object"),
]


@pytest.mark.parametrize("filename,body,match", [
    (filename, body, match) for filename, bodies, match in INIT_FILES for body in bodies
])
def test_present_but_misshapen_init_file_is_refused(tmp_path, filename, body, match):
    root = write_env(tmp_path / "root")
    (root / filename).write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError, match=match) as excinfo:
        BuiltEnvironment(root)
    assert filename in str(excinfo.value)


def test_anchor_with_misshapen_held_out_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "anchor.json").write_text(json.dumps({"held_out": ["rec1"]}), encoding="utf-8")
    with pytest.raises(EnvironmentError) as excinfo:
        BuiltEnvironment(root)
    assert "anchor.json" in str(excinfo.value)


def test_unknown_overlay_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError, match="no overlay") as excinfo:
        env.overlay("no_such_task")
    assert "no_such_task" in str(excinfo.value)


@pytest.mark.parametrize("body", ["{", "[]", '{"values": {}}', '{"overlay": [], "values": {}}'])
def test_broken_overlay_file_is_refused(tmp_path, body):
    root = write_env(tmp_path / "root")
    env = BuiltEnvironment(root)
    (root / "overlays" / "widget_task.json").write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError) as excinfo:
        env.overlay("widget_task")
    assert "widget_task.json" in str(excinfo.value)


@pytest.mark.parametrize("body", ["{", "[]", '{"id": 1}'])
def test_broken_task_file_is_refused(tmp_path, body):
    root = write_env(tmp_path / "root")
    (root / "tasks" / "widget_task.json").write_text(body, encoding="utf-8")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError) as excinfo:
        env.task("widget_task")
    assert "widget_task.json" in str(excinfo.value)


def test_broken_trace_file_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "traces").mkdir()
    (root / "traces" / "rec1.json").write_text("{", encoding="utf-8")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError) as excinfo:
        env.traces()
    assert "rec1.json" in str(excinfo.value)


def test_broken_rules_file_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "user_rules" / "rec1.json").write_text("[]", encoding="utf-8")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError) as excinfo:
        env.rules(env.task("widget_task"))
    assert "rec1.json" in str(excinfo.value)


def test_broken_verifier_file_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "verifiers" / "widget_task.json").write_text("[]", encoding="utf-8")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError) as excinfo:
        env.verifier("widget_task")
    assert "widget_task.json" in str(excinfo.value)
