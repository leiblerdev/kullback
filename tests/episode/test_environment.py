import json
import sys

import pytest

from kullback.runner.world import loading
from kullback.runner.world.environment import BuiltEnvironment, EnvironmentError, _json, _text
from tests.episode.invented import write_env


@pytest.mark.parametrize("read,break_file", [
    (lambda path: _json(path, None), "deny"),
    (_text, "deny"),
    (_text, "bad_bytes"),
])
def test_present_but_unreadable_or_undecodable_input_is_refused_not_emptied(tmp_path, read, break_file):
    path = tmp_path / "input"
    if break_file == "deny":
        path.write_text("{}", encoding="utf-8")
        path.chmod(0)
    else:
        path.write_bytes(b"\xff")
    try:
        with pytest.raises(EnvironmentError, match="unreadable or torn"):
            read(path)
    finally:
        path.chmod(0o600)


def test_missing_policy_text_is_optional(tmp_path):
    assert _text(tmp_path / "absent.md") is None


_MISSING = object()


def _write_reference_fixture(root, *, replays=_MISSING):
    import json

    from kullback.runner.records import RawPtr, Trace

    traces_dir = root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for rid in ("rec1", "held", "rec2"):
        trace = Trace(
            trace_id=rid,
            raw_hash="invented",
            ingest_version="1",
            source="invented",
            raw_ptr=RawPtr(file_hash="invented"),
            system_prompt=rid,
        )
        (traces_dir / f"{rid}.json").write_text(
            json.dumps(trace.model_dump(mode="json")), encoding="utf-8"
        )
        (root / "user_rules" / f"{rid}.json").write_text(
            json.dumps({"facts": [{"field": "speaker", "value": rid}]}),
            encoding="utf-8",
        )
    (root / "tasks" / "widget_task.json").write_text(
        json.dumps(
            {
                "id": "widget_task",
                "run_ids": ["rec1", "held", "rec2"],
                "intent": "give widget w1 the label striped",
            }
        ),
        encoding="utf-8",
    )
    (root / "anchor.json").write_text(
        json.dumps({"held_out": {"widget_task": ["held"]}}), encoding="utf-8"
    )
    replay_path = root / "replays.json"
    if replays is _MISSING:
        replays = {
            "widget_task": {
                "rec1": {"trace_id": "rec1", "confirmed": False},
                "held": {"trace_id": "held", "confirmed": True},
                "rec2": {"trace_id": "rec2", "confirmed": True},
            }
        }
    if replays is None:
        if replay_path.is_file():
            replay_path.unlink()
    else:
        replay_path.write_text(json.dumps(replays), encoding="utf-8")


def test_confirmed_reference_skips_unconfirmed_and_held_out(tmp_path):
    root = write_env(tmp_path / "env")
    _write_reference_fixture(root)
    env = BuiltEnvironment(root)
    task = env.task("widget_task")
    rules = env.rules(task)
    assert [f.value for f in rules.facts if f.field == "speaker"] == ["rec2"]
    assert [f.value for f in loading._user_rules(root, task).facts if f.field == "speaker"] == ["rec2"]
    assert env.reference(task).trace_id == "rec2"
    assert [t.trace_id for t in env.members(task)] == ["rec1", "rec2"]
    assert "held" not in [t.trace_id for t in env.members(task)]
    assert env.reference(task).trace_id != "held"


def test_confirmed_reference_prefers_sorted_confirmed_id(tmp_path):
    root = write_env(tmp_path / "env")
    _write_reference_fixture(
        root,
        replays={
            "widget_task": {
                "rec1": {"trace_id": "rec1", "confirmed": True},
                "held": {"trace_id": "held", "confirmed": True},
                "rec2": {"trace_id": "rec2", "confirmed": True},
            }
        },
    )
    env = BuiltEnvironment(root)
    task = env.task("widget_task")
    assert env.reference(task).trace_id == "rec1"
    assert [f.value for f in env.rules(task).facts if f.field == "speaker"] == ["rec1"]


def test_no_replays_file_selects_first_non_held(tmp_path):
    root = write_env(tmp_path / "env")
    _write_reference_fixture(root, replays=None)
    env = BuiltEnvironment(root)
    task = env.task("widget_task")
    assert [f.value for f in env.rules(task).facts if f.field == "speaker"] == ["rec1"]
    assert [f.value for f in loading._user_rules(root, task).facts if f.field == "speaker"] == ["rec1"]
    assert env.reference(task).trace_id == "rec1"
    assert [t.trace_id for t in env.members(task)] == ["rec1", "rec2"]


def test_no_confirmed_rows_selects_nothing(tmp_path):
    root = write_env(tmp_path / "env")
    _write_reference_fixture(
        root,
        replays={
            "widget_task": {
                "rec1": {"trace_id": "rec1", "confirmed": False},
                "held": {"trace_id": "held", "confirmed": False},
                "rec2": {"trace_id": "rec2", "confirmed": False},
            }
        },
    )
    env = BuiltEnvironment(root)
    task = env.task("widget_task")
    assert env.rules(task) is None
    assert loading._user_rules(root, task) is None
    assert env.reference(task) is None


def test_rules_path_rejects_unsafe_recording_ids(tmp_path):
    root = write_env(tmp_path / "env")
    env = BuiltEnvironment(root)
    for bad in ("", ".", "..", "a/b", "a\\b"):
        with pytest.raises(EnvironmentError):
            env._rules_path(bad)


def _hide_fidelity(monkeypatch, error):
    monkeypatch.delitem(sys.modules, "kullback.user.fidelity", raising=False)
    parent = sys.modules.get("kullback.user")
    if parent is not None and hasattr(parent, "fidelity"):
        monkeypatch.delattr(parent, "fidelity")

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "kullback.user.fidelity":
                raise error
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)


def test_agent_driven_answers_false_without_scores_and_reads_them_without_the_user_package(tmp_path, monkeypatch):
    root = write_env(tmp_path / "env")
    assert BuiltEnvironment(root).agent_driven("widget_task") is False
    (root / "user_fidelity.json").write_text(json.dumps(
        {"format": 1, "tasks": [{"task_id": "widget_task", "drives": True}]}), encoding="utf-8")
    env = BuiltEnvironment(root)
    _hide_fidelity(monkeypatch, ModuleNotFoundError("No module named 'kullback.user.fidelity'",
                                                    name="kullback.user"))
    assert env.agent_driven("widget_task") is True
