"""Tests for kullback/gates/hook.py: the tool_result hook over bound writes."""

from __future__ import annotations

from types import SimpleNamespace

from kullback.gates.hook import attach_ruling, gate_writes
from kullback.runner.records import Column, EntitySchema


def _call(name: str, path: str | None = None) -> SimpleNamespace:
    arguments = {} if path is None else {"path": path}
    return SimpleNamespace(name=name, arguments=arguments, id="call-1")


def _result(content: str = "wrote it") -> SimpleNamespace:
    return SimpleNamespace(content=content, details={}, is_error=False)


def _world() -> dict:
    schema = EntitySchema(tables=["widgets"],
                          columns=[Column(table="widgets", name="widget_id", **{"class": "hard"})])
    return {"schema": schema, "db": {"widgets": {"W-1": {"widget_id": "W-1"}}}, "calls": []}


def test_attach_ruling_appends_a_line_and_a_record():
    result = _result()
    ruling = SimpleNamespace(stage="replay_fidelity", passed=False,
                             failures=["c2 differs"], rows=[{"call": "c2"}])
    attach_ruling(result, ruling)
    assert "replay_fidelity fail" in result.content
    [record] = result.details["rulings"]
    assert record == {"name": "replay_fidelity", "accepted": False,
                      "line": result.content.splitlines()[-1], "rows": [{"call": "c2"}]}


def test_the_hook_rules_a_bound_write_and_leaves_the_passing_gates_on_it(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "m.py").write_text("hold = \"W-1\"\nx = 1\n", encoding="utf-8")
    hook = gate_writes(root, work, evidence=_world())
    result = _result()
    assert hook(_call("write", "tools/m.py"), result) is result
    assert result.details["rulings"]
    assert any(not record["accepted"] for record in result.details["rulings"])


def test_the_hook_refuses_nothing_on_a_read(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "m.py").write_text("hold = \"W-1\"\nx = 1\n", encoding="utf-8")
    hook = gate_writes(root, work)
    result = _result()
    assert hook(_call("read", "tools/m.py"), result) is None
    assert result.content == "wrote it" and result.details == {}


def test_an_unbound_write_and_an_error_result_pass_through(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "s.py").write_text("x = 1\n", encoding="utf-8")
    hook = gate_writes(root, work)
    assert hook(_call("write", "notes/s.py"), _result()) is None
    failing = SimpleNamespace(content="boom", details={}, is_error=True)
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert hook(_call("write", "tools/m.py"), failing) is None
    assert failing.details == {}
