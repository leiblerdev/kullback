"""Every JSON file the Environment reads obeys one loader rule (G1).

Missing where optional falls back; present but torn or not the expected shape
raises EnvironmentError naming the path and what was expected.
"""

import json

import pytest

from kullback.runner.world.environment import BuiltEnvironment, EnvironmentError
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


TORN_FILES = ["environment.json", "schema.json", "tool_sigs.json", "bodies.json", "db.json"]


@pytest.mark.parametrize("filename,body,match", [
    *[(filename, body, match) for filename, bodies, match in INIT_FILES for body in bodies],
    *[(filename, "{", "unreadable or torn") for filename in TORN_FILES],
    ("anchor.json", json.dumps({"held_out": ["rec1"]}), None),
])
def test_present_but_torn_or_misshapen_init_file_is_refused(tmp_path, filename, body, match):
    root = write_env(tmp_path / "root")
    (root / "env" / "db.json").write_text(json.dumps({"widgets": {}}), encoding="utf-8")
    (root / filename).write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError, match=match) as excinfo:
        BuiltEnvironment(root)
    assert filename in str(excinfo.value)


def test_unknown_overlay_is_refused(tmp_path):
    root = write_env(tmp_path / "root")
    env = BuiltEnvironment(root)
    with pytest.raises(EnvironmentError, match="no overlay") as excinfo:
        env.overlay("no_such_task")
    assert "no_such_task" in str(excinfo.value)


LAZY_FILES = [
    *[("overlays/widget_task.json", body, "overlay") for body in
      ["{", "[]", '{"values": {}}', '{"overlay": [], "values": {}}']],
    *[("tasks/widget_task.json", body, "task") for body in ["{", "[]", '{"id": 1}']],
    ("traces/rec1.json", "{", "traces"),
    ("user_rules/rec1.json", "[]", "rules"),
    ("verifiers/widget_task.json", "[]", "verifier"),
]


@pytest.mark.parametrize("relpath,body,reader", LAZY_FILES)
def test_broken_per_task_file_is_refused_when_read(tmp_path, relpath, body, reader):
    root = write_env(tmp_path / "root")
    path = root / relpath
    path.parent.mkdir(exist_ok=True)
    if reader == "overlay":
        env = BuiltEnvironment(root)
        path.write_text(body, encoding="utf-8")
    else:
        path.write_text(body, encoding="utf-8")
        env = BuiltEnvironment(root)
    read = {"overlay": lambda: env.overlay("widget_task"), "task": lambda: env.task("widget_task"),
            "traces": env.traces, "rules": lambda: env.rules(env.task("widget_task")),
            "verifier": lambda: env.verifier("widget_task")}[reader]
    with pytest.raises(EnvironmentError) as excinfo:
        read()
    assert path.name in str(excinfo.value)


@pytest.mark.parametrize("replays", [
    {"widget_task": []},
    {"widget_task": {"rec1": ["rec1"]}},
    {"widget_task": {"rec1": {"confirmed": True}}},
    {"widget_task": {"rec1": {"trace_id": 7, "confirmed": True}}},
    {"widget_task": {"rec1": {"trace_id": "rec1", "confirmed": "yes"}}},
])
def test_misshapen_replay_rows_are_refused_at_construction(tmp_path, replays):
    root = write_env(tmp_path / "root")
    (root / "replays.json").write_text(json.dumps(replays), encoding="utf-8")
    with pytest.raises(EnvironmentError) as excinfo:
        BuiltEnvironment(root)
    assert "replays.json" in str(excinfo.value)

