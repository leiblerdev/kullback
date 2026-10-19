"""The scenario record and its store (D372)."""

from __future__ import annotations

import json

import pytest

from kullback.synth import records as R

POLICY = "Changes are allowed before dispatch.\nRefunds go to the original payment method."


def test_a_record_id_is_a_hash_of_its_statement_and_source_kind():
    first = R.born("a customer wants to change something", R.Source(kind="ask", line="x"))
    again = R.born("a customer  wants to change something", R.Source(kind="ask", line="y"))
    other = R.born("a customer wants to change something", R.Source(kind="search", url="https://a.example"))
    assert first.id == again.id and first.id != other.id and first.status == "born"


def test_policy_lines_are_kept_only_where_the_policy_holds_them_verbatim():
    kept, dropped = R.policy_lines_in(["Changes are allowed before dispatch.", "Refunds are instant."], POLICY)
    assert kept == ["Changes are allowed before dispatch."] and dropped == 1


def test_the_store_writes_one_file_per_record_an_index_and_a_marker_and_no_task_file(tmp_path):
    record = R.born("a customer wants a refund", R.Source(kind="recordings", task_ids=["task-0001"]), frequency=2)
    R.write_scenario(tmp_path, record)
    folder = tmp_path / R.DIR
    assert (folder / f"{record.id}.json").is_file() and (folder / R.MARKER).is_file()
    index = json.loads((folder / R.INDEX).read_text())
    assert index["scenarios"] == [{"id": record.id, "kind": "recordings", "status": "born"}]
    assert not (tmp_path / "tasks.json").exists() and not (tmp_path / "tasks").exists()
    assert R.load_scenario(tmp_path, record.id) == record


def test_the_loader_refuses_a_record_carrying_an_undeclared_key(tmp_path):
    record = R.born("a customer wants a refund", R.Source(kind="ask", line="refund"))
    R.write_scenario(tmp_path, record)
    path = tmp_path / R.DIR / f"{record.id}.json"
    body = json.loads(path.read_text())
    body["trusted"] = True
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        R.load_scenarios(tmp_path)


def test_an_id_that_is_not_a_scenario_name_is_refused_before_any_path_is_built(tmp_path):
    with pytest.raises(ValueError):
        R.load_scenario(tmp_path, "../tasks")


def test_a_task_id_prints_masked_to_its_last_four_characters():
    assert R.mask("task-123456") == "...3456"
