"""One ledger for gates.json, written through by both agents in turn (D122, D128)."""

from __future__ import annotations

import json

from kullback.builder import pipeline
from kullback.gates.ledger import GateLedger
from kullback.runner.records import GateResult


def _ruling(stage: str, passed: bool = True, **metrics) -> GateResult:
    return GateResult(stage=stage, passed=passed, metrics=metrics)


def _stages(workdir) -> list[tuple[str, dict]]:
    return [(row["stage"], row["metrics"]) for row in json.loads((workdir / "gates.json").read_text(encoding="utf-8"))]


def test_recording_a_ruling_replaces_the_row_of_the_same_stage_and_appends_it_last(tmp_path):
    ledger = GateLedger(tmp_path)
    ledger.begin()
    ledger.record("derive", _ruling("derive_verifier", n=1))
    ledger.record("probe", _ruling("probe_pool", n=1))
    ledger.record("derive", _ruling("derive_verifier", n=2))
    assert _stages(tmp_path) == [("probe_pool", {"n": 1}), ("derive_verifier", {"n": 2})]
    assert ledger.rulings("derive") == ["derive_verifier"] and ledger.rulings("probe") == ["probe_pool"]
    ledger.write("compile", [_ruling("parses"), _ruling("confined")])
    assert _stages(tmp_path) == [("parses", {}), ("confined", {})]


def test_two_writers_taking_turns_leave_the_file_as_one_writer_would(tmp_path):
    builder, examiner = GateLedger(tmp_path / "turns"), GateLedger(tmp_path / "turns")
    alone = GateLedger(tmp_path / "alone")
    script = [(builder, "build_environment"), (builder, "rerolls"), (examiner, "derive_verifier"),
              (examiner, "probe_pool"), (builder, "rerolls"), (examiner, "trusted"), (examiner, "derive_verifier")]
    for turn, (writer, stage) in enumerate(script):
        writer.begin()
        writer.record(stage, _ruling(stage, turn=turn))
    for turn, (_, stage) in enumerate(script):
        alone.record(stage, _ruling(stage, turn=turn))
    assert (tmp_path / "turns" / "gates.json").read_bytes() == (tmp_path / "alone" / "gates.json").read_bytes()
    assert [stage for stage, _ in _stages(tmp_path / "turns")] == ["build_environment", "probe_pool", "rerolls",
                                                                    "trusted", "derive_verifier"]


def test_replay_lands_the_writes_in_stage_order_from_what_the_file_held_at_begin(tmp_path):
    ledger = GateLedger(tmp_path)
    ledger.begin()
    ledger.record("earlier", _ruling("earlier"))
    ledger.begin()
    # Recorded out of order, as two threads would; replayed in the order the schedule names.
    ledger.record("right", _ruling("right"))
    ledger.record("left", _ruling("left"))
    ledger.replay(["left", "right"])
    assert [stage for stage, _ in _stages(tmp_path)] == ["earlier", "left", "right"]
    # A replay with nothing recorded since begin leaves the file alone.
    ledger.begin()
    ledger.replay(["left", "right"])
    assert [stage for stage, _ in _stages(tmp_path)] == ["earlier", "left", "right"]


def _history(workdir) -> list[dict]:
    return json.loads((workdir / "gates_by_round.json").read_text(encoding="utf-8"))


def test_a_snapshot_keeps_the_rulings_one_round_left_without_touching_gates_json(tmp_path):
    ledger = GateLedger(tmp_path)
    ledger.record("compile", _ruling("replay_fidelity", passed=False, n=1))
    before = (tmp_path / "gates.json").read_bytes()
    ledger.snapshot(1)
    assert (tmp_path / "gates.json").read_bytes() == before, "the snapshot writes its own file only"
    assert [(row["round"], [r["stage"] for r in row["rulings"]]) for row in _history(tmp_path)] == [
        (1, ["replay_fidelity"])]


def test_a_ruling_that_went_red_to_green_is_still_readable_round_by_round(tmp_path):
    """gates.json holds the last ruling per stage, so round 1's red is gone the moment round 2 rules
    again; the per-round rows are what says the gate was ever red."""
    ledger = GateLedger(tmp_path)
    ledger.record("compile", _ruling("replay_fidelity", passed=False))
    ledger.record("intent", _ruling("intent", passed=False))
    ledger.snapshot(1)
    ledger.record("compile", _ruling("replay_fidelity", passed=True))
    ledger.snapshot(2)
    assert [row["stage"] for row in json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
            if not row["pass"]] == ["intent"]
    rulings = {row["round"]: {r["stage"]: r["pass"] for r in row["rulings"]} for row in _history(tmp_path)}
    assert rulings[1] == {"replay_fidelity": False, "intent": False}
    assert rulings[2] == {"intent": False, "replay_fidelity": True}


def test_a_round_snapshotted_twice_keeps_one_row_and_the_rows_stay_in_round_order(tmp_path):
    ledger = GateLedger(tmp_path)
    ledger.record("compile", _ruling("parses"))
    ledger.snapshot(2)
    ledger.snapshot(1)
    ledger.record("compile", _ruling("parses", passed=False))
    ledger.snapshot(2)
    assert [row["round"] for row in _history(tmp_path)] == [1, 2]
    assert _history(tmp_path)[1]["rulings"][0]["pass"] is False


def test_the_pipeline_still_reaches_the_ledger_under_its_old_name(tmp_path):
    """builder/pipeline.py names GateLedger as before and it is the one class in kullback/gates, so both
    agents record their rulings through the same code and the same lock; a ledger built under either
    name writes the same file the same way."""
    assert pipeline.GateLedger is GateLedger
    old, new = pipeline.GateLedger(tmp_path / "old"), GateLedger(tmp_path / "new")
    for ledger in (old, new):
        ledger.begin()
        ledger.record("a", _ruling("stage_a", n=1))
        ledger.record("b", _ruling("stage_b", passed=False, n=2))
        ledger.record("a", _ruling("stage_a", n=3))
    assert (tmp_path / "old" / "gates.json").read_bytes() == (tmp_path / "new" / "gates.json").read_bytes()
    assert old.rulings("a") == new.rulings("a") == ["stage_a"]
