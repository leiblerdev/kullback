"""The Reference audit pass: trusted minus checked, in chunks, flaggable."""

from __future__ import annotations

from kullback.examiner import audit as A
from kullback.runner.records import write_json


def _rounds(workdir, rows, number=1):
    folder = workdir / "rounds" / str(number)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "tasks.json", {"rows": rows})


def test_audit_list_is_trusted_minus_checked(tmp_path):
    _rounds(tmp_path, [{"task_id": "t1", "trusted": True}, {"task_id": "t2", "trusted": True},
                       {"task_id": "t3", "trusted": False}])
    A.record_check(tmp_path, "t1")
    assert A.audit_list(tmp_path) == ["t2"]


def test_audit_list_is_empty_without_a_round_file(tmp_path):
    A.record_check(tmp_path, "t1")
    assert A.audit_list(tmp_path) == []


def test_trusted_ids_come_from_the_latest_round(tmp_path):
    _rounds(tmp_path, [{"task_id": "t1", "trusted": True}], number=2)
    _rounds(tmp_path, [{"task_id": "t2", "trusted": True}], number=5)
    assert A.trusted_ids(tmp_path) == ["t2"]


def test_chunks_hold_thirty_tasks_each():
    ids = [f"t{n:03d}" for n in range(65)]
    parts = A.chunks(ids)
    assert [len(p) for p in parts] == [30, 30, 5]
    assert [t for p in parts for t in p] == sorted(ids)
    assert A.chunks([]) == []


def test_record_check_marks_and_checked_ids_reads_back(tmp_path):
    assert A.checked_ids(tmp_path) == set()
    A.record_check(tmp_path, "t9")
    assert A.checked_ids(tmp_path) == {"t9"}


