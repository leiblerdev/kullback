"""A present db.json must hold a JSON object, never fall back silently (G1).

A db.json that exists and contains `null` used to take the same packaged fallback as a
genuinely missing file, and a list or scalar was kept as is. Only a missing file falls back;
anything present that is not an object is refused, including the packaged copy when used.
"""

import json

import pytest

from kullback.episode.environment import BuiltEnvironment, EnvironmentError
from tests.episode.invented import write_env

PACKAGED = {"widgets": {"w1": {"widget_id": "w1", "label": "packaged"}}}


def _write_packaged(root):
    (root / "env" / "db.json").write_text(json.dumps(PACKAGED), encoding="utf-8")


def test_missing_root_db_uses_the_packaged_db(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "db.json").unlink()
    _write_packaged(root)
    assert BuiltEnvironment(root).db == PACKAGED


@pytest.mark.parametrize("body", ["null", "[]", '"just a label"', "4"])
def test_present_but_invalid_root_db_is_refused(tmp_path, body):
    root = write_env(tmp_path / "root")
    _write_packaged(root)
    (root / "db.json").write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError, match="not an object") as excinfo:
        BuiltEnvironment(root)
    assert "db.json" in str(excinfo.value)


def test_torn_root_db_still_raises(tmp_path):
    root = write_env(tmp_path / "root")
    _write_packaged(root)
    (root / "db.json").write_text("{", encoding="utf-8")
    with pytest.raises(EnvironmentError, match="unreadable or torn"):
        BuiltEnvironment(root)


@pytest.mark.parametrize("body", ["null", "[]"])
def test_packaged_db_is_held_to_the_same_rule(tmp_path, body):
    root = write_env(tmp_path / "root")
    (root / "db.json").unlink()
    (root / "env" / "db.json").write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError, match="not an object") as excinfo:
        BuiltEnvironment(root)
    assert "env" in str(excinfo.value)
