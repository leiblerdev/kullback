"""One ledger for gates.json, written through by both agents in turn (D122, D128)."""

from __future__ import annotations

import json

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
    # D218 rule 2: a per tool set goes to its own file, and the ledger offers no way at all to
    # replace gates.json, so the rulings the other stages recorded cannot be lost to a recompile.
    ledger.write_compile_snapshot("compile", [{"stage": "parses", "pass": True, "tool": "lookup"}], 4)
    assert _stages(tmp_path) == [("probe_pool", {"n": 1}), ("derive_verifier", {"n": 2})]
    assert not hasattr(ledger, "write")
    assert ledger.rulings("compile") == ["parses"], "the stage still names what it ruled on"


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
