"""A present db.json must hold a JSON object, never fall back silently (G1).

A db.json that exists and contains `null` used to take the same packaged fallback as a
genuinely missing file, and a list or scalar was kept as is. Only a missing file falls back;
anything present that is not an object is refused, including the packaged copy when used.
"""

import json

import pytest

from kullback.runner.world.environment import BuiltEnvironment, EnvironmentError
from tests.episode.invented import write_env

PACKAGED = {"widgets": {"w1": {"widget_id": "w1", "label": "packaged"}}}


def _write_packaged(root):
    (root / "env" / "db.json").write_text(json.dumps(PACKAGED), encoding="utf-8")


def test_missing_root_db_uses_the_packaged_db(tmp_path):
    root = write_env(tmp_path / "root")
    (root / "db.json").unlink()
    _write_packaged(root)
    assert BuiltEnvironment(root).db == PACKAGED


@pytest.mark.parametrize("where,body,match,named", [
    *[("root", body, "not an object", "db.json") for body in ["null", "[]", '"just a label"', "4"]],
    ("root", "{", "unreadable or torn", None),
    *[("packaged", body, "not an object", "env") for body in ["null", "[]"]],
])
def test_present_but_invalid_root_or_packaged_db_is_refused(tmp_path, where, body, match, named):
    root = write_env(tmp_path / "root")
    if where == "root":
        _write_packaged(root)
        (root / "db.json").write_text(body, encoding="utf-8")
    else:
        (root / "db.json").unlink()
        (root / "env" / "db.json").write_text(body, encoding="utf-8")
    with pytest.raises(EnvironmentError, match=match) as excinfo:
        BuiltEnvironment(root)
    if named:
        assert named in str(excinfo.value)
