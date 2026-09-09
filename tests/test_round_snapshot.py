"""D218: a round's Task rows are written once at round close and every reader reads that table.

The domain here is invented (a library that lends and returns copies), and nothing in it is a
customer's tool, column or corpus. What is under test is the bookkeeping: one table per round, a
compile snapshot that no longer overwrites the round's rulings, a live file that says when it moved,
a search that always writes a reason, and a reader that names the round it read and the drift since.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kullback import cli, round_snapshot
from kullback.builder.build import compile_snapshot_rows
from kullback.examiner import loosen, stage
from kullback.gates.ledger import COMPILE_NAME, GateLedger
from kullback.gates.verifier_suite import D79_STAGES
from kullback.runner.records import GateResult

runner = CliRunner()

EVERY_CHECK = {name: True for name in D79_STAGES.values()}


def _trusted(trusted=(), untrusted=None, refused=None, fractions=None, pools=None):
    """A trusted ruling in the shape the gate hands the driver."""
    return GateResult(stage="trusted", passed=not untrusted,
                      failures=[f"task {task_id}: {why}" for task_id, why in (untrusted or {}).items()],
                      metrics={"trusted": list(trusted), "untrusted": dict(untrusted or {}),
                               "refused": dict(refused or {}), "probes_passing": 0,
                               "false_rejection": dict(fractions or {}),
                               "false_rejection_pool": dict(pools or {}),
                               "false_rejection_ruling": {}, "checks_not_run": {}})


def _status(**rows):
    return {task_id: dict(row) for task_id, row in rows.items()}


def _replays(**per_task):
    return {task_id: {f"trace_{task_id}": {"confirmed": ok, "reasons": [] if ok else ["a copy was not returned"],
                                           "counts": {}}}
            for task_id, ok in per_task.items()}


# --- rule 1: one table per round, written once -------------------------------------------

def test_a_snapshot_row_names_every_d79_check_in_the_suites_own_order():
    rows = round_snapshot.task_rows(
        3, task_status=_status(lend_a_copy={"reference_confirmed": True, "verifier_passed": False,
                                            "checks": {**EVERY_CHECK, "second_path_passes": False},
                                            "not_run": ["verifier_loophole"],
                                            "not_run_reasons": {"loophole_probe_fails": "no probe was bought"}}),
        replays=_replays(lend_a_copy=True), trusted=_trusted(untrusted={"lend_a_copy": "the D79 suite did not pass"}))
    checks = rows[0]["checks"]
    assert [check["check"] for check in checks] == list(D79_STAGES.values())
    by_name = {check["check"]: check for check in checks}
    assert by_name["second_path_passes"]["ruling"] == round_snapshot.FAILED
    assert by_name["loophole_probe_fails"] == {"check": "loophole_probe_fails",
                                               "ruling": round_snapshot.NOT_RUN,
                                               "reason": "no probe was bought"}
    assert by_name["empty_fails"]["ruling"] == round_snapshot.PASSED


def test_a_closed_rounds_snapshot_is_byte_identical_after_a_later_loosening(tmp_path):
    status = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": False,
                                  "checks": {**EVERY_CHECK, "mutation_flips": False}})
    rows = round_snapshot.task_rows(2, task_status=status, replays=_replays(lend_a_copy=True),
                                    trusted=_trusted(untrusted={"lend_a_copy": "mutation_flips failed"}))
    round_snapshot.write_snapshot(tmp_path, 2, rows)
    before = round_snapshot.snapshot_path(tmp_path, 2).read_bytes()
    # The loosening relaxes an atom, the Verifier passes and the Task is trusted; the live file moves.
    loosened = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                    "checks": dict(EVERY_CHECK)})
    round_snapshot.write_snapshot(tmp_path, 2, round_snapshot.task_rows(
        2, task_status=loosened, replays=_replays(lend_a_copy=True), trusted=_trusted(trusted=["lend_a_copy"])))
    assert round_snapshot.snapshot_path(tmp_path, 2).read_bytes() == before


def test_the_rounds_task_counts_are_derived_from_the_rows_and_ride_on_the_history_row(tmp_path):
    rows = round_snapshot.task_rows(
        1, task_status=_status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                            "checks": dict(EVERY_CHECK)},
                               return_a_copy={"reference_confirmed": False, "verifier_passed": False,
                                              "reason": "the recordings disagree about the shelf"}),
        replays=_replays(lend_a_copy=True, return_a_copy=False), trusted=_trusted(trusted=["lend_a_copy"]))
    body = round_snapshot.write_snapshot(tmp_path, 1, rows)
    assert body["counts"] == {"tasks": 2, "fidelity": 1, "reference": 1, "verifier_passed": 1,
                              "trusted": 1, "refused": 0}
    ledger = GateLedger(tmp_path)
    ledger.record("derive_verifier", GateResult(stage="derive_verifier", passed=True))
    ledger.snapshot(1, tasks=body["counts"])
    history = json.loads((tmp_path / "gates_by_round.json").read_text(encoding="utf-8"))
    assert history[-1]["tasks"] == body["counts"]


def test_a_row_carries_the_kept_body_hash_of_every_tool_its_task_calls():
    rows = round_snapshot.task_rows(
        1, task_status=_status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                            "checks": dict(EVERY_CHECK),
                                            "tool_calls_replayed": {"find_copy": 2, "mark_lent": 1}}),
        replays=_replays(lend_a_copy=True), trusted=_trusted(trusted=["lend_a_copy"]),
        bodies={"find_copy": "def find_copy(): pass", "mark_lent": "def mark_lent(): pass"})
    bodies = rows[0]["bodies"]
    assert sorted(bodies) == ["find_copy", "mark_lent"]
    assert all(len(digest) == 16 for digest in bodies.values())
    moved = round_snapshot.task_rows(
        1, task_status=_status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                            "checks": dict(EVERY_CHECK),
                                            "tool_calls_replayed": {"find_copy": 2, "mark_lent": 1}}),
        replays=_replays(lend_a_copy=True), trusted=_trusted(trusted=["lend_a_copy"]),
        bodies={"find_copy": "def find_copy(): return 1", "mark_lent": "def mark_lent(): pass"})
    assert moved[0]["bodies"]["find_copy"] != bodies["find_copy"]
    assert moved[0]["bodies"]["mark_lent"] == bodies["mark_lent"]


# --- rule 2: gates.json is the round's, the compile snapshot is its own ------------------

def test_gates_json_holds_the_rounds_rulings_after_a_mid_round_compile(tmp_path):
    ledger = GateLedger(tmp_path)
    ledger.record("replay_reference", GateResult(stage="replay_reference", passed=True))
    ledger.record("trusted", GateResult(stage="trusted", passed=True))
    closed = json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
    ledger.write_compile_snapshot("compile_tools", [{"tool": "find_copy", "stage": "parses", "pass": True}], 4)
    assert json.loads((tmp_path / "gates.json").read_text(encoding="utf-8")) == closed
    body = json.loads((tmp_path / COMPILE_NAME).read_text(encoding="utf-8"))
    assert body["round"] == 4 and body["rows"][0]["tool"] == "find_copy"


def test_a_narrowed_compile_keeps_the_rows_of_the_tools_it_did_not_measure():
    kept = [{"tool": "find_copy", "stage": "parses", "pass": True},
            {"tool": "mark_lent", "stage": "parses", "pass": False}]
    fresh = {"mark_lent": [{"tool": "mark_lent", "stage": "parses", "pass": True}]}
    assert compile_snapshot_rows(kept, fresh, ["mark_lent"]) == [
        {"tool": "find_copy", "stage": "parses", "pass": True},
        {"tool": "mark_lent", "stage": "parses", "pass": True}]
    # A full run measured every tool, so it replaces the file rather than merging into it.
    assert compile_snapshot_rows(kept, fresh, None) == [{"tool": "mark_lent", "stage": "parses", "pass": True}]


def test_a_status_row_keeps_its_stamp_until_the_row_itself_moves():
    first = stage.stamped(_status(lend_a_copy={"verifier_passed": False}), None, 1, now=100.0)
    assert first["lend_a_copy"]["round"] == 1 and first["lend_a_copy"]["updated_at"] == 100.0
    same = stage.stamped(_status(lend_a_copy={"verifier_passed": False}), first, 2, now=200.0)
    assert same["lend_a_copy"]["round"] == 1 and same["lend_a_copy"]["updated_at"] == 100.0
    moved = stage.stamped(_status(lend_a_copy={"verifier_passed": True}), first, 2, now=200.0)
    assert moved["lend_a_copy"]["round"] == 2 and moved["lend_a_copy"]["updated_at"] == 200.0


# --- rule 3: a search that ends without a ruling writes its reason -----------------------

def test_a_second_path_search_that_finds_nothing_writes_its_reason():
    row = stage.second_path_row(3, 9, found=False)
    assert row["reason"] == "no second path in 3 batches" and row["exhausted"]
    assert stage.with_reason(row, what="the second path search", task_id="lend_a_copy") is row


def test_a_second_path_record_with_neither_a_find_nor_a_reason_is_refused():
    with pytest.raises(stage.MissingReason):
        stage.with_reason({"batches": 0, "found": False, "reason": ""},
                          what="the second path search", task_id="lend_a_copy")
    with pytest.raises(stage.MissingReason):
        stage.with_reason(None, what="the second path search", task_id="lend_a_copy")


def test_a_loosening_that_can_propose_nothing_says_which_of_the_three_ways_it_ended():
    assert loosen.nothing_proposed(None, object()) == loosen.NO_VERIFIER
    assert loosen.nothing_proposed(object(), None) == loosen.NO_REJECTED_RUN
    assert loosen.nothing_proposed(object(), object()) is None


def test_a_row_written_because_nothing_could_be_proposed_is_not_an_attempt():
    rows = [{"task_id": "lend_a_copy", "round": 1, "proposed": False, "reason": loosen.NO_RELAXATION},
            {"task_id": "lend_a_copy", "round": 2, "proposed": True, "kinds": ["read"], "accepted": True}]
    assert loosen.attempts(rows, "lend_a_copy") == [rows[1]]
    assert loosen.may_propose(rows, "lend_a_copy", 3) is True
    counts = loosen.round_counts(rows, 1)
    assert counts["auto_loosen_proposed"] == 0 and counts["auto_loosen_unproposed"] == 1


# --- rule 4: the reader names the round and the drift ------------------------------------

def test_the_drift_names_the_tasks_whose_live_status_has_moved_and_the_stage_each_moved_at():
    status = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": False,
                                  "checks": {**EVERY_CHECK, "mutation_flips": False}},
                     return_a_copy={"reference_confirmed": False, "verifier_passed": False,
                                    "reason": "the recordings disagree about the shelf"})
    replays = _replays(lend_a_copy=True, return_a_copy=False)
    snapshot = {"round": 4, "rows": round_snapshot.task_rows(4, task_status=status, replays=replays,
                                                             trusted=_trusted())}
    assert round_snapshot.drift(snapshot, task_status=status, replays=replays)["status_drift"] == 0
    loosened = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                    "checks": dict(EVERY_CHECK)},
                       return_a_copy={"reference_confirmed": True, "verifier_passed": False,
                                      "checks": dict(EVERY_CHECK)})
    report = round_snapshot.drift(snapshot, task_status=loosened, replays=replays)
    assert report["status_drift"] == 2 and report["round"] == 4
    assert report["by_stage"] == {"reference": 1, "verifier": 1}
    assert {row["task_id"] for row in report["first"]} == {"lend_a_copy", "return_a_copy"}


def test_the_status_command_names_the_round_it_read_and_the_first_drifted_ids(tmp_path):
    status = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": False,
                                  "checks": {**EVERY_CHECK, "mutation_flips": False}})
    replays = _replays(lend_a_copy=True)
    round_snapshot.write_snapshot(tmp_path, 5, round_snapshot.task_rows(
        5, task_status=status, replays=replays, trusted=_trusted()))
    moved = _status(lend_a_copy={"reference_confirmed": True, "verifier_passed": True,
                                 "checks": dict(EVERY_CHECK)})
    (tmp_path / "task_status.json").write_text(json.dumps(moved), encoding="utf-8")
    (tmp_path / "replays.json").write_text(json.dumps(replays), encoding="utf-8")
    out = runner.invoke(cli.app, ["status", "--workdir", str(tmp_path)])
    assert out.exit_code == 0
    assert "round 5 snapshot of 1 Tasks" in out.stdout
    assert "status_drift 1" in out.stdout and "verifier 1" in out.stdout
    assert "moved: lend_a_copy at verifier" in out.stdout


def test_a_workdir_with_no_snapshot_says_so_rather_than_reporting_a_drift_of_zero(tmp_path: Path):
    out = runner.invoke(cli.app, ["status", "--workdir", str(tmp_path)])
    assert out.exit_code == 0 and "No round snapshot is on disk" in out.stdout
