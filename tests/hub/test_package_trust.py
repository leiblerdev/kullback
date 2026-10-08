"""The package carries each Task's trust reason and the two flags off the last round's table (D333)."""

from kullback.hub import package as P
from kullback.round_snapshot import snapshot_path
from kullback.runner.records import write_json


def test_task_rows_carry_the_reason_and_flags_of_the_last_closed_round_and_the_manifest_counts_them(tmp_path):
    for number, solvable in ((1, False), (2, True)):
        write_json(snapshot_path(tmp_path, number), {"round": number, "rows": [
            {"task_id": "a", "trust_reason": None, "reference_passes": True, "solvable": solvable},
            {"task_id": "b", "trust_reason": "open_ruling", "reference_passes": False, "solvable": None}]})
    rows = P.task_index(tmp_path, ["a", "b", "c"])
    assert [(r["trust_reason"], r["reference_passes"], r["solvable"]) for r in rows] == \
        [(None, True, True), ("open_ruling", False, None), (None, None, None)]
    counts = P._row_counts(rows)
    assert counts["trust_reasons"] == {"open_ruling": 1}
    assert counts["reference_passes"] == {"true": 1, "false": 1} and counts["solvable"] == {"true": 1, "false": 0}
