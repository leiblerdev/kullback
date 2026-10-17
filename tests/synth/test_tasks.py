"""Tasks born of scenarios: passed, bound scenarios of the recordings become one Task and one recording (D389)."""

from __future__ import annotations

import json

from kullback.runner.records import Task, as_dict, write_json
from kullback.runner.world.loading import load_overlay
from kullback.spec.intent_tools import recordings_of, task_of, trace_index
from kullback.synth.records import Source, born, write_all
from kullback.synth.tasks import born_tasks, scenario_of, task_id_of, trace_id_of

RECORDINGS = Source(kind="recordings", task_ids=["task-a1"])
BOUND = {"widget_rows": "W1001"}


def _scenario(statement, status="passed", source=RECORDINGS, binding=None, requires=()):
    record = born(statement, source)
    return record.model_copy(update={"status": status, "binding": dict(BOUND if binding is None else binding),
                                     "requires": list(requires)})


def _store(root):
    """Two passed bound scenarios, and one each struck, unbound, needing a row, and of another source."""
    records = [_scenario("A holder wants a widget moved to slot seven before it is sealed."),
               _scenario("A holder asks to move a sealed widget and is offered a replacement."),
               _scenario("A holder wants a widget moved twice in one day.", status="struck"),
               _scenario("A holder wants a widget moved to a slot that does not exist.", binding={}),
               _scenario("A holder wants a widget moved to a slot no row holds.", requires=["a slot row"]),
               _scenario("A holder read online that widgets can be moved.", source=Source(kind="search", url="u"))]
    write_all(root, records)
    return sorted(records[:2], key=lambda record: record.id)


def _workdir(root):
    existing = Task(id="task_existing", run_ids=["rec-a"])
    write_json(root / "tasks" / "task_existing.json", as_dict(existing))
    write_json(root / "tasks.json", {"tasks": [as_dict(existing)]})
    return root


def test_only_passed_bound_scenarios_of_the_recordings_with_nothing_required_bear_tasks(tmp_path):
    passed = _store(tmp_path / "store")
    counts = born_tasks(_workdir(tmp_path / "env"), tmp_path / "store")
    assert counts == {"candidates": 2, "born": 2, "skipped_existing": 0, "overlays_written": 2,
                      "ids": [task_id_of(record.id) for record in passed]}
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
    assert again == {"candidates": 2, "born": 0, "skipped_existing": 2, "overlays_written": 0, "ids": []}
    assert len(json.loads((workdir / "tasks.json").read_text())["tasks"]) == 3


def test_a_born_task_loads_an_empty_overlay_so_its_starting_state_is_the_shared_world(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    counts = born_tasks(workdir, tmp_path / "store")
    assert counts["overlays_written"] == 2
    overlay, values = load_overlay(workdir, task_id_of(record.id))
    assert overlay.task_id == task_id_of(record.id) and overlay.rows == [] and overlay.steps == [] and values == {}


def test_a_second_call_writes_the_overlay_a_born_task_lacks_and_skips_the_task(tmp_path):
    record = _store(tmp_path / "store")[0]
    workdir = _workdir(tmp_path / "env")
    born_tasks(workdir, tmp_path / "store")
    (workdir / "overlays" / f"{task_id_of(record.id)}.json").unlink()
    again = born_tasks(workdir, tmp_path / "store")
    assert (again["born"], again["skipped_existing"], again["overlays_written"]) == (0, 2, 1)
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
