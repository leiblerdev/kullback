"""Tests for kullback/gates/hook.py: the tool_result hook over bound writes."""

from __future__ import annotations

from types import SimpleNamespace

from kullback.gates.hook import attach_ruling, gate_writes, ruled, ruling_line
from kullback.runner.records import Column, EntitySchema, RawPtr, ToolCall


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
                      "line": result.content.split("\n", 1)[1], "rows": [{"call": "c2"}]}


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


def test_a_failing_ruling_lists_every_row_grouped_by_what_parted_it_and_counts_the_rest():
    differ = [{"call": f"d{index}", "tool": "t", "arguments": {"id": index}, "trace": "tr1",
               "recorded": {"label": "plain"}, "ours": {"label": "odd"}, "column": "label"}
              for index in range(50)]
    raised = [{"call": f"e{index}", "tool": "t", "arguments": {"id": index}, "trace": "tr2",
               "exception": "KeyError: 'missing'", "line": "return table['missing']", "column": "error"}
              for index in range(30)]
    ruling = SimpleNamespace(stage="replay_fidelity", passed=False, failures=["d0 differs"],
                             rows=raised + differ, note="")
    lines = ruling_line(ruling).splitlines()
    assert lines[0] == "replay_fidelity fail (d0 differs) [80 rows]"
    assert lines[1] == "50 rows differ at label:"
    assert lines[2] == ('  trace tr1 call d0 args {"id":0}: ours {"label":"odd"} recorded {"label":"plain"}')
    assert "30 rows raise KeyError:" in lines
    assert "  trace tr2 call e0 args {\"id\":0}: KeyError: 'missing' at `return table['missing']`" in lines
    assert lines[-1] == "and 20 more rows"
    assert len("\n".join(lines)) < 6200


def test_ruled_names_every_failing_call_past_the_bindings_cap_with_its_trace_and_arguments(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "m.py").write_text("def m(self, widget_id):\n    return widget_id\n", encoding="utf-8")
    calls = [ToolCall(id=f"c{index}", name="m", args={"widget_id": f"W-{index}"}, result={"label": "x"},
                      raw_ptr=RawPtr(file_hash="f" * 64, sim_index=0, msg_index=index), trace_id=f"tr{index}")
             for index in range(30)]

    def execute(source, calls, db):
        return [{"ok": False, "error": "KeyError", "message": f"'{call.args['widget_id']}'",
                 "line": "return self.db[widget_id]"} for call in calls]

    rulings = ruled(root, "tools/m.py", work, execute=execute, evidence={**_world(), "calls": calls})
    failing = next(ruling for ruling in rulings if not ruling.passed)
    assert len(failing.rows) == 30
    assert failing.rows[29]["trace"] == "tr29" and failing.rows[29]["arguments"] == {"widget_id": "W-29"}
    text = ruling_line(failing)
    assert "30 rows raise KeyError:" in text
    assert "trace tr29 call c29 args {\"widget_id\":\"W-29\"}: KeyError: 'W-29' at `return self.db[widget_id]`" in text


def test_ruled_hands_the_executor_every_keyword_the_rulings_pass_it(tmp_path, monkeypatch):
    import inspect

    from kullback.gates import hook

    seen = []

    def recording(source, calls, db, **kwargs):
        seen.append(kwargs)
        return []

    def rulings_stub(root, path, workdir, *, execute=None, evidence=None):
        takes = any(param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in inspect.signature(execute).parameters.values())
        execute("source", [], {}, **({"trace_calls": ["prefix"]} if takes else {}))
        return []

    monkeypatch.setattr(hook, "rulings_for", rulings_stub)
    assert ruled(tmp_path, "tools/m.py", tmp_path, execute=recording) == []
    assert seen == [{"trace_calls": ["prefix"]}]
