"""The Examiner called by the Builder: derive by code, then one model session (D79, D128, D230)."""

from __future__ import annotations

from pathlib import Path

from examiner.worlds import make_world
from gates.examiner_fixtures import SIGS
from kullback.agent.events import ToolExecutionEnd
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness, drive_tool
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import session as S
from kullback.examiner.exam_files import ExamRoot
from kullback.gates.probes import version_hash
from kullback.runner import budget, tool
from kullback.runner.records import Verifier, as_dict, read_json, write_json
from tests.runner.test_tool import _env_with_trace


def reply(content, *calls) -> ModelReply:
    """One scripted assistant turn, the shape tests/builder/session_fixtures.py:reply gives."""
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                  for i, (n, a) in enumerate(calls)])


def materialize(world) -> None:
    """The derivation's store as workdir files: tasks, sigs, replays, re-rolls, rules."""
    workdir = world.workdir
    (workdir / "tasks").mkdir(exist_ok=True)
    for task in world.inputs["tasks"]:
        write_json(workdir / "tasks" / f"{task.id}.json", as_dict(task))
    write_json(workdir / "tool_sigs.json", [as_dict(sig) for sig in SIGS])
    write_json(workdir / "constraints.json", [])
    write_json(workdir / "canon-rules.json", {})
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])


def test_examine_without_a_model_returns_derive_filed_findings_with_rows(tmp_path):
    world = make_world(tmp_path, confirmed=False)
    materialize(world)
    findings = S.examine(world.workdir, model=None)
    assert findings, "an unconfirmed world owes a fidelity finding"
    assert all(f.rows for f in findings)
    assert all(f.source == "derive" for f in findings)
    assert {f.kind for f in findings} <= {"assisted_tool", "fidelity", "reference_disagreement",
                                          "suite", "false_rejection", "environment", "other"}
    written = read_json(world.workdir / "findings.json", [])
    assert len(written) == len(findings) and all(w["rows"] for w in written)


def _scripted_events(world, replies):
    events: list = []
    S.examine(world.workdir, model=TestModel(replies), subscribers=[events.append])
    return events


def _ends(events, tool):
    return [e for e in events if isinstance(e, ToolExecutionEnd) and e.tool_name == tool]


def test_fresh_session_seeds_version_1_accepted_for_each_derived_verifier(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    _scripted_events(world, [reply("done")])
    derived = Verifier.model_validate(read_json(world.workdir / "verifiers" / "t1.json"))
    written = read_json(world.workdir / "exam" / "history.json")
    [row] = written["t1"]["versions"]
    assert (row["verifier_version"], row["accepted"], row["by"]) == ("1", True, "derive")
    assert row["reason"] == "the Verifier on disk before any repair"
    assert row["content_hash"] == version_hash(derived)


# A proposal that changes the atoms and none of the rulings: a write cap no Run reaches, which
# no check can mutate. A proposal of no change is refused before the suite (F37).
_WIDE_CAP = {"id": "wide_cap", "kind": "allowed", "payload": {"kind": "entity_count", "count": 1000}}


def test_model_propose_verifier_lands_while_the_loophole_probe_cannot_run(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    events = _scripted_events(world, [
        reply("proposing", ("propose_verifier", {"task_id": "t1", "reason": "tighten",
                                                 "add": [_WIDE_CAP]})),
        reply("done"),
    ])
    [proposed] = _ends(events, "propose_verifier")
    assert proposed.is_error is False
    assert "verifier_loophole not run (no model" in proposed.result.content
    assert (world.workdir / "exam" / "verifiers" / "t1.json").is_file()
    written = read_json(world.workdir / "exam" / "history.json")
    assert [v["accepted"] for v in written["t1"]["versions"]] == [True, True]
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is False and "verifier_loophole" in row["not_run"]


def test_a_model_write_or_edit_under_verifiers_is_refused_and_names_the_tools_that_write(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    before = read_json(world.workdir / "exam" / "verifiers" / "t1.json")
    events = _scripted_events(world, [
        reply("proposing", ("write", {"path": "verifiers/t1.json", "content": "{}"})),
        reply("editing", ("edit", {"path": "verifiers/t1.json", "old": "t1", "new": "t2"})),
        reply("done"),
    ])
    refused = _ends(events, "write") + _ends(events, "edit")
    assert [end.is_error for end in refused] == [True, True]
    for end in refused:
        assert ("the Examiner writes Verifiers through propose_verifier and probes through probe, "
                "not with write or edit") in end.result.content
    assert read_json(world.workdir / "exam" / "verifiers" / "t1.json") == before


def test_model_finding_publishes_and_duplicates_are_refused_with_the_id(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    args = {"task_id": "t1", "kind": "fidelity", "text": "replays differ",
            "rows": [{"run_id": "ref"}], "path": "env/tools/x.py", "change": "answer the call"}
    events = _scripted_events(world, [
        reply("filing", ("finding", args)),
        reply("filing again", ("finding", args)),
        reply("done"),
    ])
    filed = _ends(events, "finding")
    assert filed[0].is_error is False
    finding_id = filed[0].result.details["finding"]["finding_id"]
    assert filed[1].is_error is True
    assert finding_id in filed[1].result.content
    bus_lines = (world.workdir / "bus.jsonl").read_text().splitlines()
    assert any(finding_id in line for line in bus_lines)
    written = read_json(world.workdir / "findings.json", [])
    assert any(w.get("finding_id") == finding_id for w in written)


def test_bash_write_and_edit_are_not_registered_and_base_tools_are_scoped_to_exam(tmp_path):
    world = make_world(tmp_path)
    root = ExamRoot(workdir=world.workdir)
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.examiner_extension(root)])
    names = harness.registry.names()
    assert not {"bash", "write", "edit"} & set(names)
    assert {"read", "grep", "find", "ls", "web_search",
            "propose_verifier", "probe", "finding", "reroll"} <= set(names)
    S.expose(world.workdir)
    refused = drive_tool(harness, "write", {"path": "runs/x.jsonl", "content": "{}"})
    assert refused.is_error is True


def test_the_examiner_gets_inspect_among_its_base_tools_and_still_no_write_edit_or_bash(tmp_path):
    world = make_world(tmp_path)
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.examiner_extension(ExamRoot(workdir=world.workdir))])
    names = set(harness.registry.names())
    assert "inspect" in names
    assert not {"write", "edit", "bash"} & names


def _rename_loop() -> TestModel:
    """A model that reads the widget and writes the striped label, then stops, on every Run."""
    return TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)


def test_examine_with_a_reroll_model_buys_second_path_rerolls(tmp_path):
    """A lone Reference buys re-rolls through the runner callable: Runs on disk, rows kept."""
    root = _env_with_trace(tmp_path / "env")
    seed = tool.run(root, "widget_task", _rename_loop(), workdir=root)
    write_json(root / "replays.json", {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": seed.run_id,
                 "confirmed": True, "path": seed.path}}})
    write_json(root / "rerolls.json", {"widget_task": []})
    write_json(root / "constraints.json", [])
    S.examine(root, model=None, reroll_model=_rename_loop())
    bought = sorted((root / "runs").glob("second-path-*.jsonl"))
    assert bought, "a Reference standing alone buys re-rolls through the callable"
    rows = (read_json(root / "examiner" / "rerolls.json", {}) or {}).get("widget_task") or []
    assert rows and all(row.get("reason") == "second_path" for row in rows)


def _proposed_over_a_runner_world(tmp_path, probe_model):
    """A built Environment with two Runs of one Task on disk, examined with a probe model, and
    a scripted Examiner that proposes the derived Verifier with a wide write cap through propose_verifier."""
    root = _env_with_trace(tmp_path / "work")
    seed = tool.run(root, "widget_task", _rename_loop(), workdir=root)
    other = tool.run(root, "widget_task", _rename_loop(), workdir=root / "second")
    write_json(root / "replays.json", {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": seed.run_id, "confirmed": True, "path": seed.path}}})
    write_json(root / "rerolls.json", {"widget_task": [
        {"run_id": other.run_id, "path": str(root / "second" / other.path),
         "termination_reason": "user_stop"}]})
    write_json(root / "constraints.json", [])
    events: list = []
    S.examine(root, model=TestModel([
        reply("proposing", ("propose_verifier", {"task_id": "widget_task", "reason": "again",
                                                 "add": [_WIDE_CAP]})),
        reply("done")]), probe_model=probe_model, subscribers=[events.append])
    [proposed] = _ends(events, "propose_verifier")
    return root, proposed


def test_session_proposal_runs_the_loophole_probe_through_the_runner_and_is_trusted(tmp_path):
    root, proposed = _proposed_over_a_runner_world(tmp_path, TestModel(["I will not."], loop=True))
    assert proposed.is_error is False
    assert "verifier_loophole pass" in proposed.result.content
    assert "trusted pass" in proposed.result.content
    status = read_json(root / "exam" / "task_status.json")
    assert status["widget_task"]["verifier_passed"] is True
    assert list((root / "probes").glob("*.jsonl")), "the probe is a Run the runner played"


def test_session_proposal_whose_probe_reaches_the_end_state_is_refused_at_the_loophole(tmp_path):
    root, proposed = _proposed_over_a_runner_world(tmp_path, _rename_loop())
    assert proposed.is_error is True
    assert "verifier_loophole fail" in proposed.result.content
    written = read_json(root / "exam" / "history.json")
    assert written["widget_task"]["versions"][-1]["rejected_by"] == ["verifier_loophole"]



def test_an_examiner_session_prices_its_model_calls_into_the_ledger_under_its_stage(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    usage = {"input": 1890, "output": 17, "cache_write": 1887}
    model = TestModel([reply("done").model_copy(update={"usage": usage})],
                      name=next(iter(budget.PRICES)))
    S.examine(world.workdir, model=model)
    stage = budget.load_totals(world.workdir)["stages"]["examiner"]
    assert stage["calls"] == 1 and stage["usd"] > 0


def _replayed_root(tmp_path):
    """The invented environment with one seed Run played into its own runs/ and a replay row naming it."""
    root = _env_with_trace(tmp_path / "env")
    seed = tool.run(root, "widget_task", _rename_loop(), workdir=root)
    write_json(root / "replays.json", {"widget_task": {
        "rec1": {"trace_id": "rec1", "run_id": seed.run_id, "confirmed": True, "path": seed.path}}})
    write_json(root / "rerolls.json", {"widget_task": []})
    write_json(root / "constraints.json", [])
    return root


def test_every_run_path_in_the_exam_replays_opens_under_the_exam_root(tmp_path):
    """The Examiner reads exam/ only, so each path its replays.json names is a file under exam/."""
    root = _replayed_root(tmp_path)
    S.examine(root, model=None)
    replays = read_json(root / "exam" / "replays.json")
    paths = [row["path"] for rows in replays.values() for row in rows.values()]
    assert paths and all(not Path(path).is_absolute() for path in paths)
    assert all((root / "exam" / path).is_file() for path in paths)


def test_examine_from_another_cwd_opens_the_runs_and_stores_bought_rows_relative(tmp_path, monkeypatch):
    """Run rows resolve against the workdir, not the cwd: second paths are bought and stored as runs/..."""
    root = _replayed_root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    S.examine(root, model=None, reroll_model=_rename_loop())
    rows = (read_json(root / "examiner" / "rerolls.json", {}) or {}).get("widget_task") or []
    assert rows and all(row["path"].startswith("runs/") for row in rows)
    assert all((root / row["path"]).is_file() for row in rows)
    status = read_json(root / "task_status.json")
    assert status["widget_task"]["reference_confirmed"] is True


def test_an_examiner_stopped_on_its_turn_cap_says_its_findings_are_partial(tmp_path):
    """A session that is still reading at the cap files one note naming the cap, in findings.json too."""
    world = make_world(tmp_path)
    materialize(world)
    reading = TestModel([reply("reading", ("ls", {"path": "."}))], loop=True)
    findings = S.examine(world.workdir, model=reading, max_turns=2)
    note = findings[-1]
    assert note.change.startswith("the Examiner ran out of turns at 2")
    assert note.rows == [{"max_turns": 2}]
    written = read_json(world.workdir / "findings.json")
    assert written[-1]["change"] == note.change


def test_an_examiner_that_finishes_under_its_cap_files_no_turn_note(tmp_path):
    """The default cap leaves room for the reading, and a session that ends on its own adds nothing."""
    world = make_world(tmp_path)
    materialize(world)
    reading = TestModel([reply("reading", ("ls", {"path": "."}))] * 9 + [reply("done")])
    findings = S.examine(world.workdir, model=reading)
    assert not any("ran out of turns" in f.change for f in findings)


def test_session_prompt_names_dot_as_root_lists_entries_and_the_paths_of_each_task(tmp_path):
    world = make_world(tmp_path)
    materialize(world)
    model = TestModel([reply("done")])
    S.examine(world.workdir, model=model)
    [call] = model.calls
    system = " ".join(str(m.get("content")) for m in call["messages"])
    assert 'Your root is "."' in system
    assert str(world.workdir) not in system
    assert ("The root holds: derived/, history.json, references.json, replays.json, rerolls.json, "
            "runs/, spoken/, task_status.json, tasks/.") in system
    assert "on your first write under them: verifiers/, probes/" in system
    assert "runs: runs/t1/ref.jsonl, runs/t1/alt.jsonl" in system
    assert "proposal: verifiers/t1.json once you first propose one" in system


def test_examine_with_no_finished_run_opens_no_session_and_names_the_task_left_out(tmp_path):
    world = make_world(tmp_path, confirmed=False, rerolls=())
    materialize(world)
    model = TestModel([reply("done")])
    findings = S.examine(world.workdir, model=model)
    assert model.calls == []
    [note] = [f for f in findings if any("left_out" in r for r in f.rows)]
    assert (note.kind, note.source) == ("other", "derive")
    assert note.rows == [{"task_id": "t1", "left_out": S.NO_FINISHED_RUN}]
    assert "no Examiner session opened" in note.change
    assert not (world.workdir / "budget.json").is_file() or "examiner" not in (
        world.workdir / "budget.json").read_text()


def test_a_task_without_a_confirmed_reference_is_left_out_with_its_reason():
    replays = {"t1": {"a": {"run_id": "a", "confirmed": True}},
               "t2": {"b": {"run_id": "b", "confirmed": True}},
               "t3": {"c": {"run_id": "c", "confirmed": False}}}
    status = {"t1": {"reference_confirmed": True}, "t2": {"reference_confirmed": False}}
    selected, left_out = S.select_for_session(["t1", "t2", "t3"], replays, {}, status)
    assert selected == ["t1"]
    assert left_out == [("t2", S.NO_CONFIRMED_REFERENCE), ("t3", S.NO_FINISHED_RUN)]
    note = S.left_out_finding(left_out, examined=1)
    assert "t2 (no confirmed Reference), t3 (no finished Run)" in note.text
    assert note.change.startswith("the Examiner session read 1 Tasks")


def test_the_rulings_line_covers_only_the_tasks_the_session_examines(tmp_path):
    world = make_world(tmp_path, tasks=2)
    root = ExamRoot(workdir=world.workdir, replays=world.inputs["replays"])
    lines = S.rulings_line(root, ["t2"]).splitlines()
    assert [line.split(":")[0] for line in lines] == ["t2"]


def test_the_rulings_line_names_the_derived_verifier_and_the_user_rules_of_a_confirmed_replay(tmp_path):
    world = make_world(tmp_path, tasks=2)
    write_json(world.workdir / "verifiers" / "t1.json", {"task_id": "t1"})
    S.expose(world.workdir)
    replays = dict(world.inputs["replays"])
    replays["t2"] = {"ref": {**replays["t2"]["ref"], "confirmed": False}}
    root = ExamRoot(workdir=world.workdir, replays=replays)
    lines = dict(line.split(": ", 1) for line in S.rulings_line(root).splitlines())
    assert "derived verifier: derived/t1.json" in lines["t1"]
    assert "proposal: verifiers/t1.json once you first propose one" in lines["t1"]
    assert "user rules: user_rules/ref.json" in lines["t1"]
    assert "no user rules until the Reference is confirmed" in lines["t2"]


def test_the_rulings_line_names_the_spoken_file_expose_wrote_with_its_extension(tmp_path):
    """The spoken part is the real file under exam/spoken, so no read guesses its extension (F35)."""
    world = make_world(tmp_path)
    materialize(world)
    write_json(world.workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}]}})
    S.examine(world.workdir, model=None)
    root = S._exam_root(world.workdir, S.load_store(world.workdir), [])
    [spoken] = sorted((root.exam_dir / "spoken").glob("t1.*"))
    assert f"spoken: spoken/{spoken.name}" in S.rulings_line(root, ["t1"])
    spoken.rename(spoken.with_suffix(".md"))
    assert "spoken: spoken/t1.md" in S.rulings_line(root, ["t1"])
    spoken.with_suffix(".md").unlink()
    assert "spoken: none" in S.rulings_line(root, ["t1"])


def test_a_session_capped_after_a_refused_proposal_files_its_note_on_that_task_with_the_refusal(tmp_path):
    """The turn-cap finding names the Task the session was on and the last refusal it read (F36)."""
    world = make_world(tmp_path)
    materialize(world)
    proposing = TestModel([reply("proposing", ("propose_verifier", {"task_id": "t1", "reason": "tighten",
                                                                    "drop": [], "add": []}))], loop=True)
    findings = S.examine(world.workdir, model=proposing, max_turns=2)
    note = findings[-1]
    assert note.change.startswith("the Examiner ran out of turns at 2")
    assert note.task_id == "t1"
    assert "; last refusal: " in note.change
    assert read_json(world.workdir / "findings.json")[-1]["task_id"] == "t1"


def _derived(world) -> list[str]:
    return sorted(path.stem for path in (world.workdir / "verifiers").glob("*.json"))


def _not_derived(findings) -> list:
    return [f for f in findings if any("not_derived" in row for row in f.rows)]


def test_one_examine_call_derives_all_twelve_confirmed_tasks_and_files_no_not_derived_note(tmp_path):
    """F40, F55: examine derives every Task derive_pick returns, with no cap per call."""
    world = make_world(tmp_path, tasks=12)
    materialize(world)
    findings = S.examine(world.workdir, model=None)
    assert _derived(world) == sorted(f"t{n}" for n in range(1, 13))
    assert not _not_derived(findings)
    assert S.derive_pick(world.workdir, S.load_store(world.workdir), None) == []


def test_the_exam_view_after_examine_holds_the_verifiers_and_status_this_call_derived(tmp_path):
    """F52: expose runs after the derivation, so the first call's session reads current files."""
    world = make_world(tmp_path, tasks=2)
    materialize(world)
    S.examine(world.workdir, model=None)
    exam = world.workdir / "exam"
    assert sorted(read_json(exam / "task_status.json")) == ["t1", "t2"]
    assert read_json(exam / "task_status.json") == read_json(world.workdir / "task_status.json")
    assert sorted(p.stem for p in (exam / "derived").glob("*.json")) == ["t1", "t2"]


def test_examine_with_a_limit_derives_that_many_and_names_the_rest_until_a_second_call(tmp_path):
    """F40: an explicit limit derives the first Tasks in id order; one finding names the rest."""
    world = make_world(tmp_path, tasks=4)
    materialize(world)
    first = S.examine(world.workdir, model=None, limit=2)
    assert _derived(world) == ["t1", "t2"]
    [note] = _not_derived(first)
    assert note.kind == "other" and note.source == "derive"
    assert note.text == "2 confirmed Tasks not derived yet: t3, t4; call examine again"
    assert note.change.startswith("this examine call was limited to 2 Tasks; ")
    assert note.rows == [{"task_id": t, "not_derived": "limit"} for t in ("t3", "t4")]
    second = S.examine(world.workdir, model=None)
    assert _derived(world) == ["t1", "t2", "t3", "t4"]
    assert not _not_derived(second)
    status = read_json(world.workdir / "task_status.json")
    assert sorted(status) == ["t1", "t2", "t3", "t4"], "each call merges its rows into the status"


def test_examine_with_task_ids_derives_exactly_those(tmp_path):
    world = make_world(tmp_path, tasks=4)
    materialize(world)
    findings = S.examine(world.workdir, task_ids=["t3", "t1"], model=None)
    assert _derived(world) == ["t1", "t3"]
    assert not _not_derived(findings)


def test_examine_with_every_confirmed_task_derived_derives_nothing_and_files_no_cap_note(tmp_path):
    world = make_world(tmp_path, tasks=2)
    materialize(world)
    S.examine(world.workdir, model=None)
    before = (world.workdir / "task_status.json").read_bytes()
    findings = S.examine(world.workdir, model=None)
    assert (world.workdir / "task_status.json").read_bytes() == before, "no derivation ran"
    assert not _not_derived(findings)


def test_the_cap_note_names_ten_tasks_and_counts_the_rest():
    note = S.not_derived_finding([f"t{n:02d}" for n in range(1, 13)])
    assert note.text.startswith("12 confirmed Tasks not derived yet: t01, t02,")
    assert "t10 and 2 more; call examine again" in note.text
    assert len(note.rows) == 12


def test_a_task_left_unconfirmed_is_picked_again_once_its_replay_is_confirmed(tmp_path):
    world = make_world(tmp_path, confirmed=False, rerolls=())
    materialize(world)
    S.examine(world.workdir, model=None)
    status = read_json(world.workdir / "task_status.json")
    assert status["t1"]["reference_confirmed"] is False
    assert not (world.workdir / "verifiers" / "t1.json").exists()
    assert S.derive_pick(world.workdir, S.load_store(world.workdir), None) == []
    replays = read_json(world.workdir / "replays.json")
    replays["t1"]["ref"].update(confirmed=True, reasons=[])
    write_json(world.workdir / "replays.json", replays)
    assert S.derive_pick(world.workdir, S.load_store(world.workdir), None) == ["t1"]
