"""The Builder's domain tools: replay rows call by call, reroll rows run by run, examine delegates."""

from __future__ import annotations

import asyncio
import json

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.builder import env_files, world_tools
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, write_json
from tests.builder.session_fixtures import reply
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def _workdir(tmp_path, **env):
    root = write_env(tmp_path / "work", **env)
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
    assert "widget_task: fidelity: answer the recorded call (tools/a.py) [1 rows]" in result.content


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


def _unrendered_workdir(tmp_path):
    """A fresh workdir: traces and a derivable world, no bodies, no rendered toolkit under env/."""
    root = _workdir(tmp_path)
    (root / "env" / "tools.py").unlink()
    (root / "env" / "data_model.py").unlink()
    return root


def test_derive_renders_the_toolkit_so_replay_no_longer_lacks_it(tmp_path):
    root = _unrendered_workdir(tmp_path)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    derived = _run(tools["derive_world"], {})
    assert not derived.is_error, derived.content
    assert (root / "env" / "tools.py").is_file() and (root / "env" / "data_model.py").is_file()
    assert sorted(derived.details["files_written"]) == ["env/tools/describe_widget.py",
                                                        "env/tools/rename_widget.py"]
    replayed = _run(tools["replay"], {"task_id": "widget_task"})
    assert "no rendered toolkit" not in replayed.content


def test_status_renders_the_stage_line_first(tmp_path):
    root = _unrendered_workdir(tmp_path)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    _run(tools["derive_world"], {})
    result = _run(tools["status"], {})
    assert not result.is_error, result.content
    tasks = len(result.details["tasks"])
    assert result.content.splitlines()[0] == (
        f"stage: traces 1, world derived, toolkit rendered, stubs 2 of 2, tasks with a finished run 0 of {tasks}")
    assert result.details["stage"]["stubs"] == 2


def test_examine_with_no_finished_run_says_replay_or_run_comes_first(tmp_path):
    root = _workdir(tmp_path)
    (examine,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, examine_fn=lambda w, t: [])
                  if tool.name == "examine"]
    result = _run(examine, {"task_ids": None})
    assert not result.is_error, result.content
    assert "0 of 1 tasks have a finished run" in result.content
    assert "replay or run the Tasks first" in result.content


def _with_prose_result(root):
    """The recording also carries one call another requestor made, answered with a sentence."""
    path = root / "traces" / "rec1.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["tool_calls"].append(ToolCall(
        id="u1", name="check_panel", args={}, result="Panel: DIM", requestor="user",
        raw_ptr=PTR, trace_id="rec1", has_result=True, resolved=True,
    ).model_dump(mode="json", by_alias=True))
    path.write_text(json.dumps(body), encoding="utf-8")
    return root


def test_derive_hands_the_session_model_to_the_readers_step_and_prices_it(tmp_path):
    """Prose results from another requestor are read with the session model, under stage readers."""
    root = _with_prose_result(_workdir(tmp_path))
    model = TestModel([reply("no reader")], loop=True)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model)}
    derived = _run(tools["derive_world"], {})
    assert "no model to propose" not in derived.content
    assert not derived.is_error, derived.content
    assert model.calls, "the readers step never asked the session model"
    assert "check_panel" in json.dumps(model.calls[0]["messages"])
    ledger = json.loads((root / "budget.json").read_text(encoding="utf-8"))
    assert "readers" in json.dumps(ledger)


def test_derive_without_a_model_still_refuses_prose_results_by_name(tmp_path):
    root = _with_prose_result(_workdir(tmp_path))
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root)}
    derived = _run(tools["derive_world"], {})
    assert derived.is_error
    assert "no model to propose" in derived.content


def _greeting_first_candidate():
    """A candidate that greets before it acts, the way the smoke build's candidate opened."""
    return TestModel([
        reply("What would you like help with?"),
        reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
        reply("Done, widget w1 is labelled striped."),
    ], loop=True)


def _fresh_workdir(tmp_path):
    """A workdir whose Task carries no intent and whose rules only derive_world can write."""
    root = _workdir(tmp_path, with_rules=False)
    task = root / "tasks" / "widget_task.json"
    task.write_text(json.dumps({**json.loads(task.read_text(encoding="utf-8")), "intent": None}),
                    encoding="utf-8")
    return root


def test_a_fresh_run_without_user_rules_is_not_played_and_says_why(tmp_path):
    """Before derive_world writes rules there is no Simulated user, so the Task is named, not played (F34)."""
    root = _fresh_workdir(tmp_path)
    candidate = _greeting_first_candidate()
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=candidate)}
    result = _run(tools["run"], {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    assert result.details["runs"] == [] and candidate.calls == []
    assert result.details["not_run"] == ["widget_task"]
    assert "no Simulated user" in result.content


def test_derived_user_rules_open_a_fresh_run_that_finishes_with_a_verdict(tmp_path):
    """derive_world writes the rules, the rule user opens with the goal, the goal is met, a Verdict."""
    root = _fresh_workdir(tmp_path)
    candidate = _greeting_first_candidate()
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=candidate)}
    # The world alone, so the invented bodies stand in for the ones the Builder writes after it.
    assert "user_rules/rec1.json" in world_tools.derive_world(root).paths
    result = _run(tools["run"], {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    (row,) = result.details["runs"]
    opening = candidate.calls[0]["messages"]
    assert any(m.get("role") == "user" and "striped" in str(m.get("content")) for m in opening)
    assert [route["name"] for route in row["routes"]] == ["rename_widget"]
    assert row["user_end"] == "goal_satisfied"
    assert row["verdict"] is not None and row["verdict"]["passed"]
    assert "1 finished" in result.content


def test_the_examine_summary_leads_with_the_tasks_not_derived_yet(tmp_path):
    """F40: the cap note comes first in the summary, so the Builder knows to call examine again."""
    from kullback.examiner.session import not_derived_finding

    root = _workdir(tmp_path)
    note = not_derived_finding(["t3", "t4"])
    (examine,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, examine_fn=lambda w, t: [note])
                  if tool.name == "examine"]
    result = _run(examine, {"task_ids": None})
    assert not result.is_error, result.content
    assert result.content.startswith("2 confirmed Tasks not derived yet: t3, t4; call examine again")


def _exposed(tmp_path):
    root = _workdir(tmp_path)
    sigs = json.loads((root / "tool_sigs.json").read_text(encoding="utf-8"))
    for sig in sigs:
        sig["args_fields"] = [{"name": "widget_id", "types": ["str"], "optional": False}]
    (root / "tool_sigs.json").write_text(json.dumps(sigs), encoding="utf-8")
    env_files.explode(root)
    env_files.expose(root)
    return root


def _tool(root, name, **kwargs):
    return {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, **kwargs)}[name]


def test_the_rulings_tool_rules_a_tool_file_on_disk_and_names_the_raising_call_and_body_line(tmp_path):
    root = _exposed(tmp_path)
    body = root / "env" / "tools" / "describe_widget.py"
    body.write_text(body.read_text(encoding="utf-8").replace("raise NotImplementedError",
                                                             "table = {}\n    return table[widget_id]"),
                    encoding="utf-8")
    before = {path: path.read_bytes() for path in (root / "env").rglob("*") if path.is_file()}
    result = _run(_tool(root, "rulings"), {"path": "tools/describe_widget.py"})
    assert not result.is_error, result.content
    assert result.content.startswith("rulings on tools/describe_widget.py: fails ")
    assert "1 rows raise KeyError:" in result.content
    assert "trace rec1 call c1 args {\"widget_id\":\"w1\"}: KeyError: 'w1' at `return table[widget_id]`" in result.content
    assert {path: path.read_bytes() for path in (root / "env").rglob("*") if path.is_file()} == before


def test_replay_rows_carry_the_task_and_the_arguments_off_the_calls_files(tmp_path):
    root = _exposed(tmp_path)
    result = _run(_tool(root, "replay"), {"task_id": "widget_task"})
    first = result.details["rows"][0]
    assert (first["task_id"], first["arguments"], first["error"]) == ("widget_task", '{"widget_id":"w1"}', None)
    assert "task widget_task call c1 describe_widget args {\"widget_id\":\"w1\"}: recorded" in result.content


def test_run_plays_only_tasks_with_a_verifier_and_counts_the_ones_skipped(tmp_path):
    root = _workdir(tmp_path)
    _run(_tool(root, "replay"), {"task_id": "widget_task"})
    (root / "verifiers" / "widget_task.json").unlink()
    model = TestModel([reply("Done.")], loop=True)
    result = _run(_tool(root, "run", model=model), {})
    assert not result.is_error, result.content
    assert result.details["runs"] == [] and model.calls == []
    assert result.content.splitlines()[-1] == "1 open Tasks skipped for no Verifier: examine derives them first"


def test_examine_names_each_finding_by_task_and_each_held_task_by_the_check_holding_it(tmp_path):
    root = _workdir(tmp_path)
    write_json(root / "exam" / "task_status.json", {
        "task_a": {"verifier_passed": False, "checks": {"loophole_probe_fails": False, "empty_fails": True},
                   "not_run_reasons": {"loophole_probe_fails": "no model"}},
        "task_b": {"verifier_passed": True, "checks": {"second_path_passes": False}},
        "task_c": {"verifier_passed": False, "checks": {"empty_fails": False}}})
    finding = {"task_id": "task_c", "kind": "suite", "change": "tighten the\n   atom", "rows": []}
    result = _run(_tool(root, "examine", examine_fn=lambda workdir, task_ids: [finding]), {})
    lines = result.content.splitlines()
    assert "task_c: suite: tighten the atom [0 rows]" in lines
    assert lines[-2:] == ["task_a: loophole_probe_fails not run", "task_c: empty_fails failed"]
