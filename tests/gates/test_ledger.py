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
