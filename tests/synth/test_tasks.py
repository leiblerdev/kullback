"""Tasks born of scenarios: passed, bound scenarios with an ask and an end, of any source, become one Task and one
recording (D389, D411), the end beside it for the writer and in its user rules (D410)."""

from __future__ import annotations

import json

from kullback.runner.records import RawPtr, Task, Trace, as_dict, write_json
from kullback.runner.world.loading import load_overlay
from kullback.spec.intent_tools import recordings_of, task_of, trace_index
from kullback.spec.writer import scenario_end
from kullback.synth.records import Source, born, write_all
from kullback.synth.tasks import END_FIELD, bears_task, born_tasks, recording_of, scenario_of, task_id_of, trace_id_of

RECORDINGS = Source(kind="recordings", task_ids=["task-a1"])
BOUND = {"widget_rows": "W1001"}
END = "The widget W1001 sits in slot seven; the holder is told the new slot."


def _scenario(statement, status="passed", source=RECORDINGS, binding=None, requires=(), ask=None, end=END):
    record = born(statement, source)
    return record.model_copy(update={"status": status, "binding": dict(BOUND if binding is None else binding),
                                     "requires": list(requires), "ask": statement if ask is None else ask,
                                     "end": end})


def _store(root):
    """Three passed bound scenarios with an ask and an end, one of them from a search page, and one each struck,
    unbound, needing a row, with no ask and with no end."""
    records = [_scenario("A holder wants a widget moved to slot seven before it is sealed."),
               _scenario("A holder asks to move a sealed widget and is offered a replacement."),
               _scenario("A holder read online that widgets can be moved.",
                         source=Source(kind="search", url="u", start=0, end=40)),
               _scenario("A holder wants a widget moved twice in one day.", status="struck"),
               _scenario("A holder wants a widget moved to a slot that does not exist.", binding={}),
               _scenario("A holder wants a widget moved to a slot no row holds.", requires=["a slot row"]),
               _scenario("A holder wants a widget moved and says nothing.", ask=""),
               _scenario("A holder wants a widget moved and no end is written.", end="")]
    write_all(root, records)
    return sorted(records[:3], key=lambda record: record.id)


def _workdir(root):
    existing = Task(id="task_existing", run_ids=["rec-a"])
    write_json(root / "tasks" / "task_existing.json", as_dict(existing))
    write_json(root / "tasks.json", {"tasks": [as_dict(existing)]})
    return root


def test_only_passed_bound_scenarios_with_an_ask_and_an_end_and_nothing_required_bear_tasks_of_any_source(tmp_path):
    passed = _store(tmp_path / "store")
    counts = born_tasks(_workdir(tmp_path / "env"), tmp_path / "store")
    assert counts == {"candidates": 3, "born": 3, "skipped_existing": 0, "overlays_written": 3,
                      "prompts_written": 0, "ends_written": 3, "rules_written": 3,
                      "ids": [task_id_of(record.id) for record in passed]}
    assert sorted(record.source.kind for record in passed) == ["recordings", "recordings", "search"]
    index = json.loads((tmp_path / "env" / "tasks.json").read_text())["tasks"]
    assert [row["id"] for row in index] == ["task_existing", *counts["ids"]]
    assert all("scenario" not in row for row in index)


def test_a_born_task_reads_back_with_one_user_turn_holding_the_statement(tmp_path):
    passed = _store(tmp_path / "store")
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    record = passed[0]
    task = task_of(workdir, task_id_of(record.id))
    assert task.unguarded and task.anchor_run_ids == []
    assert task.run_ids == [trace_id_of(record.id)] and task.id.startswith("task_") and len(task.id) == 17
    trace = trace_index(workdir)[trace_id_of(record.id)]
    assert trace.source == "scenario" and trace.raw_hash == record.id and trace.tool_calls == []
    assert trace.raw_ptr.file_hash == record.id and trace.raw_ptr.section == "scenario"
    turns = recordings_of(workdir, task.id).recordings[trace.trace_id]
    assert [(turn.index, turn.text) for turn in turns] == [(0, record.statement)]


def test_a_second_call_skips_the_tasks_that_exist_and_appends_nothing(tmp_path):
    _store(tmp_path / "store")
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    again = born_tasks(workdir, tmp_path / "store")
    assert again == {"candidates": 3, "born": 0, "skipped_existing": 3, "overlays_written": 0,
                     "prompts_written": 0, "ends_written": 0, "rules_written": 0, "ids": []}
    assert len(json.loads((workdir / "tasks.json").read_text())["tasks"]) == 4


def test_a_born_task_loads_an_empty_overlay_so_its_starting_state_is_the_shared_world(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    counts = born_tasks(workdir, tmp_path / "store")
    assert counts["overlays_written"] == 3
    overlay, values = load_overlay(workdir, task_id_of(record.id))
    assert overlay.task_id == task_id_of(record.id) and overlay.rows == [] and overlay.steps == [] and values == {}


def test_a_second_call_writes_the_overlay_a_born_task_lacks_and_skips_the_task(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    (workdir / "overlays" / f"{task_id_of(record.id)}.json").unlink()
    again = born_tasks(workdir, tmp_path / "store")
    assert (again["born"], again["skipped_existing"], again["overlays_written"]) == (0, 3, 1)
    assert load_overlay(workdir, task_id_of(record.id))[0].rows == []


def test_the_limit_takes_the_first_scenarios_in_id_order(tmp_path):
    passed = _store(tmp_path / "store")
    counts = born_tasks(_workdir(tmp_path / "env"), tmp_path / "store", limit=1)
    assert counts["candidates"] == 1 and counts["ids"] == [task_id_of(passed[0].id)]


def test_born_ids_join_the_build_sample_and_the_tiers_call_the_born_task_synthetic(tmp_path):
    from kullback.spec.trust import intent_tiers

    passed = _store(tmp_path / "store")
    workdir = _workdir(tmp_path / "env")
    write_json(workdir / "build.json", {"tasks": ["task_existing"]})
    counts = born_tasks(workdir, tmp_path / "store")
    assert json.loads((workdir / "build.json").read_text()) == {"tasks": ["task_existing", *counts["ids"]]}
    born_id = task_id_of(passed[0].id)
    (workdir / "task_status.json").write_text(json.dumps({born_id: {}, "task_existing": {}}))
    tiers = intent_tiers(workdir)
    assert tiers[born_id].tier == "synthetic" and tiers[born_id].row["scenario"] == passed[0].id
    assert tiers["task_existing"].tier == "untrusted"


def test_a_workdir_without_a_build_sample_gets_no_build_file(tmp_path):
    _store(tmp_path / "store")
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    assert not (workdir / "build.json").exists()


def test_the_task_record_is_plain_and_scenario_of_reads_the_scenario_off_its_recording(tmp_path):
    passed = _store(tmp_path / "store")
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    task_id = task_id_of(passed[0].id)
    assert "scenario" not in json.loads((workdir / "tasks" / f"{task_id}.json").read_text())
    assert scenario_of(workdir, task_id) == passed[0].id and scenario_of(workdir, "task_existing") is None


def _recordings_with_prompts(workdir, prompts):
    for number, prompt in enumerate(prompts):
        trace = Trace(trace_id=f"rec-{number}", raw_hash="h", ingest_version="1", source="recorded",
                      raw_ptr=RawPtr(file_hash="h", section="s"), system_prompt=prompt)
        write_json(workdir / "traces" / f"rec-{number}.json", as_dict(trace))


def test_a_born_recording_carries_the_agent_prompt_most_recordings_share(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    _recordings_with_prompts(workdir, ["Rare policy.", "Shared policy.", "Shared policy.", None])
    counts = born_tasks(workdir, tmp_path / "store")
    assert counts["prompts_written"] == 3
    assert trace_index(workdir)[trace_id_of(record.id)].system_prompt == "Shared policy."


def test_a_born_recording_has_no_prompt_when_no_recording_carries_one(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    _recordings_with_prompts(workdir, [None])
    assert born_tasks(workdir, tmp_path / "store")["prompts_written"] == 0
    assert trace_index(workdir)[trace_id_of(record.id)].system_prompt is None


def test_the_user_turn_is_the_ask_when_the_scenario_has_one_else_the_statement(tmp_path):
    asked = _store(tmp_path / "store")[0]
    write_all(tmp_path / "store", [asked.model_copy(update={"ask": "I need my widget moved to slot seven."})])
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    assert trace_index(workdir)[trace_id_of(asked.id)].turns[0].content == "I need my widget moved to slot seven."
    plain = asked.model_copy(update={"ask": ""})
    assert recording_of(plain).turns[0].content == plain.statement


def test_a_rerun_writes_the_shared_prompt_into_a_born_recording_that_lacks_it_and_counts_it(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    _recordings_with_prompts(workdir, ["Shared policy."])
    again = born_tasks(workdir, tmp_path / "store")
    assert (again["born"], again["skipped_existing"], again["prompts_written"]) == (0, 3, 3)
    trace = trace_index(workdir)[trace_id_of(record.id)]
    assert trace.system_prompt == "Shared policy." and trace.turns[0].content == record.statement
    assert born_tasks(workdir, tmp_path / "store")["prompts_written"] == 0


def test_a_scenario_without_an_ask_or_an_end_bears_no_task_whatever_its_source():
    search = _scenario("A holder read online that widgets can be moved.", source=Source(kind="search", url="u"))
    assert bears_task(search)
    assert not bears_task(search.model_copy(update={"ask": " "})) and not bears_task(search.model_copy(update={"end": ""}))


def test_the_scenario_end_lies_beside_a_born_task_for_the_writer_and_a_recorded_task_has_none(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    assert scenario_end(workdir, task_id_of(record.id)) == END
    assert scenario_end(workdir, "task_existing") == ""


def test_a_rerun_writes_the_end_and_the_user_rules_a_born_task_lacks(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    (workdir / "born" / f"{task_id_of(record.id)}.json").unlink()
    (workdir / "user_rules" / f"{trace_id_of(record.id)}.json").unlink()
    again = born_tasks(workdir, tmp_path / "store")
    assert (again["ends_written"], again["rules_written"]) == (1, 1)
    assert scenario_end(workdir, task_id_of(record.id)) == END


def test_the_born_recordings_user_rules_carry_the_ask_as_goal_and_the_end_as_a_fact(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    rules = json.loads((workdir / "user_rules" / f"{trace_id_of(record.id)}.json").read_text())
    facts = {fact["field"]: fact["value"] for fact in rules["facts"]}
    assert facts[END_FIELD] == END and facts["goal"] == record.ask
