"""The Builder's domain tools: replay rows call by call, reroll rows run by run, examine delegates."""

from __future__ import annotations

import asyncio
import json

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, write_json
from tests.builder.session_fixtures import reply
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def _workdir(tmp_path):
    root = write_env(tmp_path / "work")
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Give widget w1 the label striped.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"widget_id": "w1"}', tool_call_ids=["c1"],
             raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=["c2"], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"widget_id": "w1"}', tool_call_ids=["c2"],
             raw_ptr=PTR),
        Turn(idx=6, role="assistant", content="Done.", raw_ptr=PTR),
        Turn(idx=7, role="user", content="Thanks, that is the right label.", raw_ptr=PTR),
    ]
    calls = [
        ToolCall(id="c1", name="describe_widget", args={"widget_id": "w1"},
                 result={"widget_id": "w1", "label": "plain"}, raw_ptr=PTR, trace_id="rec1"),
        ToolCall(id="c2", name="rename_widget", args={"widget_id": "w1", "label": "striped"},
                 result={"widget_id": "w1", "label": "striped"}, raw_ptr=PTR, trace_id="rec1"),
    ]
    trace = Trace(trace_id="rec1", raw_hash="r" * 64, ingest_version="1", source="test",
                  turns=turns, tool_calls=calls, raw_ptr=PTR)
    (traces / "rec1.json").write_text(json.dumps(trace.model_dump(mode="json", by_alias=True)),
                                      encoding="utf-8")
    return root


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_replay_answers_each_recorded_call_with_both_values(tmp_path):
    root = _workdir(tmp_path)
    (replay,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "replay"]
    result = _run(replay, {"task_id": "widget_task"})
    assert not result.is_error, result.content
    body = json.loads(result.content) if result.content.startswith("{") else None
    assert "c1" in result.content and "c2" in result.content
    assert "describe_widget" in result.content and "rename_widget" in result.content
    assert "fidelity" in result.content
    assert body is None  # the rendered text is rows, the full rows ride in details
    assert result.details["fidelity"] == 1.0
    assert result.details["total"] == 2
    first = result.details["rows"][0]
    assert (first["call_id"], first["tool"]) == ("c1", "describe_widget")
    assert first["recorded"] and first["ours"]


def test_run_reports_each_fresh_run_with_verdict_routes_diff_and_spend(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel([
        reply(None, ("describe_widget", {"widget_id": "w1"})),
        reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
        reply("Done."),
    ])
    (run,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model) if tool.name == "run"]
    result = _run(run, {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    assert len(result.details["runs"]) == 1
    row = result.details["runs"][0]
    assert row["run_id"] == "widget_task-0"
    assert row["routes"]
    assert isinstance(row["world_diff"], dict)
    assert isinstance(row["spend"], dict)
    assert "widget_task-0" in result.content


def test_run_without_a_model_is_an_error_result(tmp_path):
    root = _workdir(tmp_path)
    (run,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "run"]
    result = _run(run, {"task_id": "widget_task", "count": 1})
    assert result.is_error


def test_examine_calls_the_function_the_session_passed_in(tmp_path):
    root = _workdir(tmp_path)
    seen = {}

    class FakeFinding:
        def __init__(self, kind, task_id):
            self._body = {"finding_id": f"f-{kind}", "task_id": task_id, "kind": kind,
                          "text": f"{kind} is off", "rows": [{"call_id": "c1"}], "path": "tools/a.py",
                          "change": "answer the recorded call", "source": "examiner"}

        def as_dict(self):
            return dict(self._body)

    def examine_fn(workdir, task_ids):
        seen["workdir"] = workdir
        seen["task_ids"] = task_ids
        return [FakeFinding("fidelity", "widget_task"), FakeFinding("suite", "widget_task")]

    (examine,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, examine_fn=examine_fn)
                  if tool.name == "examine"]
    result = _run(examine, {"task_ids": ["widget_task"]})
    assert not result.is_error, result.content
    assert seen["task_ids"] == ["widget_task"]
    assert len(result.details["findings"]) == 2
    assert result.details["findings"][0]["finding_id"] == "f-fidelity"
    assert "fidelity on widget_task: tools/a.py: answer the recorded call [1 rows]" in result.content


RENAME_BODY = (
    "row = self.db.widgets.get(widget_id)\n"
    "if row is None:\n"
    "    raise KeyError(widget_id)\n"
    "row.label = label\n"
    'return {"widget_id": widget_id, "label": label}'
)


def test_derive_writes_tool_files_and_the_read_surface(tmp_path):
    root = _workdir(tmp_path)
    from kullback.runner.records import write_json as _write_json
    _write_json(root / "bodies.json", {"rename_widget": RENAME_BODY})
    (derive,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "derive_world"]
    result = _run(derive, {})
    assert not result.is_error, result.content
    assert (root / "env" / "tools" / "rename_widget.py").is_file()
    assert (root / "env" / "calls" / "rename_widget.jsonl").is_file()
    assert "rename_widget" in result.content


def test_grow_adds_rows_to_the_table_named_and_says_how_many(tmp_path):
    root = _workdir(tmp_path)
    before = len((json.loads((root / "db.json").read_text(encoding="utf-8")) or {}).get("widgets", {}))
    assert before == 1
    (grow,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "grow"]
    result = _run(grow, {"table": "widgets", "count": 3})
    assert not result.is_error, result.content
    after = len((json.loads((root / "db.json").read_text(encoding="utf-8")) or {}).get("widgets", {}))
    assert after == 3
    assert result.details["tables"]["widgets"] == after
    assert "widgets" in result.content and "3" in result.content


def test_replay_writes_replays_and_fidelity_and_replaces_on_repeat(tmp_path):
    root = _workdir(tmp_path)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    first = _run(tools["replay"], {"task_id": "widget_task"})
    assert not first.is_error, first.content
    stored = json.loads((root / "replays.json").read_text(encoding="utf-8"))
    row = stored["widget_task"]["rec1"]
    assert isinstance(row["confirmed"], bool) and isinstance(row["trace_id"], str)
    fidelity = json.loads((root / "tool_fidelity.json").read_text(encoding="utf-8"))
    assert fidelity["tools"]["rename_widget"]["calls"] == 1
    assert fidelity["tools"]["describe_widget"]["replayed"] == 1
    second = _run(tools["replay"], {"task_id": "widget_task"})
    assert not second.is_error, second.content
    again = json.loads((root / "replays.json").read_text(encoding="utf-8"))
    assert list(again["widget_task"]) == ["rec1"]


def _with_held_out_run(root):
    """A second Run of the widget Task that only looks w1 up, held out by the anchor."""
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="What label does widget w1 carry?", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=["h1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"widget_id": "w1", "label": "plain"}',
             tool_call_ids=["h1"], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content="It carries plain.", raw_ptr=PTR),
        Turn(idx=5, role="user", content="Thanks.", raw_ptr=PTR),
    ]
    calls = [ToolCall(id="h1", name="describe_widget", args={"widget_id": "w1"},
                      result={"widget_id": "w1", "label": "plain"}, raw_ptr=PTR, trace_id="rec2")]
    trace = Trace(trace_id="rec2", raw_hash="h" * 64, ingest_version="1", source="test",
                  turns=turns, tool_calls=calls, raw_ptr=PTR)
    write_json(root / "traces" / "rec2.json", trace.model_dump(mode="json", by_alias=True))
    write_json(root / "tasks" / "widget_task.json",
               {"id": "widget_task", "run_ids": ["rec1", "rec2"],
                "intent": "give widget w1 the label striped"})
    write_json(root / "anchor.json", {"held_out": {"widget_task": ["rec2"]}})
    return root


def test_replay_writes_every_traces_row_and_the_holdout_answers(tmp_path):
    root = _with_held_out_run(_workdir(tmp_path))
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    result = _run(tools["replay"], {"task_id": "widget_task"})
    assert not result.is_error, result.content
    stored = json.loads((root / "replays.json").read_text(encoding="utf-8"))
    assert sorted(stored["widget_task"]) == ["rec1", "rec2"]
    assert stored["widget_task"]["rec2"]["held_out"] is True
    fidelity = json.loads((root / "tool_fidelity.json").read_text(encoding="utf-8"))
    assert fidelity["tools"]["describe_widget"]["calls"] == 2
    answers = json.loads((root / "holdout_answers.json").read_text(encoding="utf-8"))
    assert set(answers) == {"totals", "columns", "tasks", "tools"}
    assert "trace rec1 (reference): fidelity 1.0000, 0 differing of 2 calls" in result.content
    assert "trace rec2 (held out): fidelity 1.0000, 0 differing of 1 calls" in result.content
    assert [row["call_id"] for row in result.details["rows"]] == ["c1", "c2"]


def test_replay_attaches_the_reference_replay_ruling(tmp_path):
    root = _workdir(tmp_path)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    result = _run(tools["replay"], {"task_id": "widget_task"})
    (ruling,) = result.details["rulings"]
    assert (ruling["name"], ruling["accepted"]) == ("replay_reference", True)
    assert ruling["line"] in result.content
    ledger = json.loads((root / "gates.json").read_text(encoding="utf-8"))
    assert [row["stage"] for row in ledger] == ["replay_reference"]


def test_a_task_whose_traces_all_fail_is_named_by_the_ruling(tmp_path):
    root = _workdir(tmp_path)
    path = root / "env" / "tools.py"
    path.write_text(path.read_text(encoding="utf-8").replace("row.label = label", 'row.label = "smudged"')
                    .replace('return {"widget_id": widget_id, "label": label}',
                             'return {"widget_id": widget_id, "label": "smudged"}'), encoding="utf-8")
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    result = _run(tools["replay"], {"task_id": "widget_task"})
    (ruling,) = result.details["rulings"]
    assert not ruling["accepted"]
    assert "replay_reference fail (task widget_task:" in ruling["line"]


def test_fidelity_index_counts_both_grains_with_worded_reasons():
    replays = {"task-1": {"trace-1": {"task_id": "task-1", "trace_id": "trace-1",
                                      "confirmed": False, "calls": [
                {"call_id": "c1", "tool": "a-tool", "verdict": "same",
                 "first_differing_column": None, "recorded": "1", "ours": "1"},
                {"call_id": "c2", "tool": "a-tool", "verdict": "differs",
                 "first_differing_column": "label", "recorded": "plain", "ours": "smudged"},
            ]}}}
    index = domain_tools_mod.fidelity_index(replays)
    assert index["tools"]["a-tool"] == {"calls": 2, "replayed": 1, "differing": 1, "assisted": True}
    task_row = index["tasks"]["task-1"]["a-tool"]
    assert (task_row["replayed"], task_row["differing"]) == (1, 1)
    assert task_row["reasons"] == ["call c2: label recorded plain ours smudged"]


def test_status_answers_one_row_per_task_and_per_tool(tmp_path):
    from kullback.runner.records import read_json as _read_json
    from kullback.runner.records import write_json as _write_json

    root = _workdir(tmp_path)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    fresh = _run(tools["status"], {})
    assert not fresh.is_error, fresh.content
    assert [(row["task_id"], row["state"]) for row in fresh.details["tasks"]] == [("widget_task", "open")]
    assert fresh.details["tasks"][0]["reason"]
    assert [row["reason"] for row in fresh.details["tools"]] == ["no replay rows yet"] * 2
    replayed = _run(tools["replay"], {"task_id": "widget_task"})
    assert not replayed.is_error, replayed.content
    result = _run(tools["status"], {})
    assert not result.is_error, result.content
    assert [(row["task_id"], row["state"]) for row in result.details["tasks"]] == [("widget_task", "open")]
    by_tool = {row["tool"]: row for row in result.details["tools"]}
    assert by_tool["rename_widget"]["fidelity"] == 1.0
    assert by_tool["rename_widget"]["calls"] == 1
    assert "widget_task" in result.content and "rename_widget" in result.content
    fidelity = _read_json(root / "tool_fidelity.json", {})
    fidelity["tools"].pop("describe_widget")
    _write_json(root / "tool_fidelity.json", fidelity)
    partial = _run(tools["status"], {})
    by_tool = {row["tool"]: row for row in partial.details["tools"]}
    assert by_tool["rename_widget"]["fidelity"] == 1.0
    assert by_tool["describe_widget"]["reason"] == "no replay rows yet"

def test_the_examine_tool_runs_the_examiner_session_from_inside_the_builders_loop(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel([reply("Nothing to file.")])
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model)}
    result = _run(tools["examine"], {})
    assert not result.is_error, result.content
    assert (root / "findings.json").exists()


def _record_examine(monkeypatch):
    from kullback.examiner import session as examiner_session

    seen = {}

    def examine(workdir, *, task_ids=None, model=None, judge_model=None, probe_model=None,
                reroll_model=None, **kwargs):
        seen.update(model=model, judge_model=judge_model, probe_model=probe_model,
                    reroll_model=reroll_model)
        return []

    monkeypatch.setattr(examiner_session, "examine", examine)
    return seen


def test_default_examine_runs_judges_probe_and_rerolls_on_the_builder_model(tmp_path, monkeypatch):
    seen = _record_examine(monkeypatch)
    builder = object()
    domain_tools_mod.default_examine_fn(tmp_path, None, model=builder)
    assert seen == {"model": builder, "judge_model": builder, "probe_model": builder,
                    "reroll_model": builder}


def test_default_examine_hands_a_named_judge_probe_or_reroll_model_its_own_role(tmp_path, monkeypatch):
    seen = _record_examine(monkeypatch)
    builder, judge, probe, reroll = object(), object(), object(), object()
    domain_tools_mod.default_examine_fn(tmp_path, None, model=builder, judge_model=judge,
                                        probe_model=probe, reroll_model=reroll)
    assert seen == {"model": builder, "judge_model": judge, "probe_model": probe,
                    "reroll_model": reroll}


def test_default_examine_with_no_model_leaves_every_model_none(tmp_path, monkeypatch):
    seen = _record_examine(monkeypatch)
    domain_tools_mod.default_examine_fn(tmp_path, None)
    assert seen == {"model": None, "judge_model": None, "probe_model": None, "reroll_model": None}
