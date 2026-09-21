import json
import sys
from pathlib import Path

import pytest

from kullback.episode.environment import BuiltEnvironment, EnvironmentError, _json
from tests.episode.invented import write_env


@pytest.mark.parametrize("filename", ["environment.json", "schema.json", "tool_sigs.json", "bodies.json", "db.json"])
def test_environment_rejects_corrupt_present_inputs(tmp_path, filename):
    root = write_env(tmp_path / "env")
    (root / filename).write_text("{", encoding="utf-8")
    (root / "env" / "db.json").write_text(json.dumps({"widgets": {}}), encoding="utf-8")
    with pytest.raises(EnvironmentError, match="unreadable or torn"):
        BuiltEnvironment(root)


def test_missing_optional_input_retains_its_default(tmp_path):
    default = {}
    assert _json(tmp_path / "missing.json", default) is default


def test_permission_failure_does_not_become_an_empty_input(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(EnvironmentError, match="unreadable or torn"):
        _json(tmp_path / "db.json", None)


def test_missing_root_db_still_uses_the_package_db(tmp_path, monkeypatch):
    root = write_env(tmp_path / "env")
    expected = {"widgets": {"w9": {"widget_id": "w9", "label": "packaged"}}}
    (root / "env" / "db.json").write_text(json.dumps(expected), encoding="utf-8")
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == root / "db.json":
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    assert BuiltEnvironment(root).db == expected


def test_missing_policy_text_is_optional(tmp_path):
    from kullback.episode.environment import _text
    assert _text(tmp_path / "absent.md") is None


def test_present_unreadable_policy_text_is_not_silently_dropped(tmp_path, monkeypatch):
    from pathlib import Path

    from kullback.episode.environment import EnvironmentError, _text
    def unreadable(self, *args, **kwargs):
        raise PermissionError("unreadable")
    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(EnvironmentError, match="unreadable or torn"):
        _text(tmp_path / "policy.md")


def test_invalid_utf8_policy_text_is_reported(tmp_path):
    from kullback.episode.environment import EnvironmentError, _text
    path = tmp_path / "policy.md"
    path.write_bytes(b"\xff")
    with pytest.raises(EnvironmentError, match="unreadable or torn"):
        _text(path)


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


def test_agent_driven_missing_module_answers_false(tmp_path, monkeypatch):
    root = write_env(tmp_path / "env")
    env = BuiltEnvironment(root)
    _hide_fidelity(monkeypatch, ModuleNotFoundError("No module named 'kullback.user.fidelity'",
                                                    name="kullback.user"))
    assert env.agent_driven("widget_task") is False


def test_agent_driven_broken_transitive_import_propagates(tmp_path, monkeypatch):
    root = write_env(tmp_path / "env")
    env = BuiltEnvironment(root)
    _hide_fidelity(monkeypatch, ImportError("broken transitive dependency", name="inner_dep"))
    with pytest.raises(ImportError, match="broken transitive"):
        env.agent_driven("widget_task")
