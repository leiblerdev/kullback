"""Tests for admission per recording: standings, the floor, the ruling, the evidence folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.builder import ingest


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def wrap(simulations, info=None) -> dict:
    return {
        "timestamp": "2025-06-05T14:00:00",
        "info": info if info is not None else {"environment_info": {}},
        "tasks": [],
        "simulations": simulations,
    }


def plain_msg(content="hello") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None, "turn_idx": 0}


def good_sim(sim_id, content="hello") -> dict:
    return {"id": sim_id, "termination_reason": "user_stop", "messages": [plain_msg(content)]}


def dangling_sim(sim_id) -> dict:
    calls = [{"id": "c9", "name": "look_up", "arguments": {}, "requestor": "assistant"}]
    return {"id": sim_id, "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": None, "tool_calls": calls, "turn_idx": 0}]}


def ingest_payload(payload, workdir, tmp_path, name="file.json") -> dict:
    return ingest.ingest_file(write_json(tmp_path / name, payload), workdir)


def ingest_raises(payload, workdir, tmp_path, name="file.json") -> ingest.IntakeGateError:
    """Ingest a file under the floor: the ruling and the evidence are written, then it raises."""
    with pytest.raises(ingest.IntakeGateError) as caught:
        ingest_payload(payload, workdir, tmp_path, name)
    return caught.value


# --- one bad call sets that recording aside and the rest build ----------------


def test_one_bad_call_sets_that_recording_aside_and_the_rest_build(workdir, tmp_path):
    sims = [good_sim("g1"), good_sim("g2"), good_sim("g3"), dangling_sim("b1")]
    summary = ingest_payload(wrap(sims), workdir, tmp_path)
    assert summary["gate"]["pass"] is True
    assert summary["task_eligible"] == 3
    assert summary["evidence_only"] == 1
    assert summary["runs"] == 3
    written = {p.stem for p in (workdir / "traces").glob("*.json")}
    assert set(summary["trace_hashes"]) == written
    assert len(written) == 3
    evidence = list((workdir / "evidence_traces").glob("*.json"))
    assert len(evidence) == 1
    kept = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert kept["trace_id"] == "b1"
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["counts"] == {"task_eligible": 3, "evidence_only": 1, "rejected": 0}
    assert ruling["passed"] is True


def test_nothing_downstream_reads_the_evidence_folder(workdir, tmp_path):
    from kullback.builder.build import load_traces

    sims = [good_sim("g1"), good_sim("g2"), good_sim("g3"), dangling_sim("b1")]
    ingest_payload(wrap(sims), workdir, tmp_path)
    assert {t.trace_id for t in load_traces(workdir)} == {"g1", "g2", "g3"}


def test_under_the_floor_ingest_stops_instead_of_returning(workdir, tmp_path):
    error = ingest_raises(wrap([good_sim("g1"), dangling_sim("b1")]), workdir, tmp_path)
    assert "1 of 2 recordings task-eligible" in str(error)
    assert error.gate.metrics["task_eligible"] == 1
    assert error.gate.metrics["evidence_only"] == 1


# --- under the floor the stage fails with the counts in the message -----------


def test_under_the_floor_the_stage_fails_with_the_counts_in_the_message(workdir, tmp_path):
    error = ingest_raises(wrap([good_sim("g1"), dangling_sim("b1")]), workdir, tmp_path)
    gate = error.gate
    assert gate.passed is False
    floor_lines = [line for line in gate.failures if "under the floor" in line]
    assert len(floor_lines) == 1
    assert "1 of 2 recordings task-eligible" in floor_lines[0]
    assert "1 task-eligible, 1 evidence-only, 0 rejected" in floor_lines[0]
    assert any("c9" in line for line in gate.failures)


def test_at_the_floor_the_stage_passes(workdir, tmp_path):
    sims = [good_sim("g1"), good_sim("g2"), good_sim("g3"), dangling_sim("b1")]
    summary = ingest_payload(wrap(sims), workdir, tmp_path)
    assert summary["gate"]["metrics"]["eligible_share"] == 0.75
    assert summary["gate"]["pass"] is True


# --- the ruling counts per standing and per reason, with the set-aside mix ----


def test_the_ruling_carries_reasons_and_the_set_aside_mix(workdir, tmp_path):
    sims = [good_sim("g1"), dangling_sim("b1"),
            {"id": "u1", "termination_reason": "max_steps", "messages": [plain_msg()]}]
    ingest_raises(wrap(sims), workdir, tmp_path)
    ruling = json.loads(list((workdir / "intake").glob("*.json"))[0].read_text(encoding="utf-8"))
    assert ruling["reasons"] == {"complete_record": 1, "unresolved_call": 1, "unfinished": 1}
    assert ruling["set_aside"]["outcomes"] == {"user_stop": 1, "max_steps": 1}
    assert ruling["set_aside"]["turns"] == {"1-5": 2}
    assert ruling["set_aside"]["tool_calls"] == {"0": 1, "1-3": 1}
    assert ruling["floor"] == ingest.MIN_TASK_ELIGIBLE_SHARE


# --- each standing ------------------------------------------------------------


def test_a_duplicate_recording_is_evidence_only(workdir, tmp_path):
    ingest_raises(wrap([good_sim("same"), good_sim("same")]), workdir, tmp_path)
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"] == {"complete_record": 1, "duplicate": 1}


def test_a_failed_intake_publishes_no_traces(workdir, tmp_path):
    with pytest.raises(ingest.IntakeGateError):
        ingest_payload(wrap([good_sim("g1"), dangling_sim("b1")]), workdir, tmp_path)
    assert list((workdir / "traces").glob("*.json")) == []
    assert not (workdir / "evidence_traces").exists()
    # The ruling is still written, so the failure stays readable.
    assert len(list((workdir / "intake").glob("*.json"))) == 1


def test_customer_keys_ending_in_ptr_are_not_blanked_for_duplicates(workdir, tmp_path):
    def answered(ptr_value) -> dict:
        calls = [{"id": "c1", "name": "look_up", "arguments": {"cursor_ptr": ptr_value},
                  "requestor": "assistant"}]
        return {"termination_reason": "user_stop", "messages": [
            {"role": "assistant", "content": None, "tool_calls": calls, "turn_idx": 0},
            {"role": "tool", "id": "c1", "content": '{"ok": true}', "turn_idx": 1}]}

    summary = ingest_payload(wrap([answered("A"), answered("B")]), workdir, tmp_path)
    assert summary["runs"] == 2
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"] == {"complete_record": 2}


def test_recordings_with_no_id_match_by_content_not_by_fallback_id(workdir, tmp_path):
    sims = [{"termination_reason": "user_stop", "messages": [plain_msg("same")]},
            {"termination_reason": "user_stop", "messages": [plain_msg("same")]}]
    ingest_raises(wrap(sims), workdir, tmp_path)
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"] == {"complete_record": 1, "duplicate": 1}


def test_a_malformed_recording_is_rejected_and_the_file_builds_on(workdir, tmp_path):
    sims = [good_sim("g1"), good_sim("g2"), good_sim("g3"),
            {"id": "broken", "messages": ["oops"]}]
    summary = ingest_payload(wrap(sims), workdir, tmp_path)
    assert summary["runs"] == 3
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"] == {"complete_record": 3, "not_a_recording": 1}


def test_an_unfinished_recording_is_evidence_only(workdir, tmp_path):
    sims = [good_sim("g1"), {"id": "u1", "termination_reason": "max_steps",
                             "messages": [plain_msg()]}]
    error = ingest_raises(wrap(sims), workdir, tmp_path)
    assert error.gate.metrics["evidence_only"] == 1
    assert error.gate.passed is False  # 1 of 2 is under the floor


def test_an_unparseable_recording_is_rejected(workdir, tmp_path):
    calls = [{"id": "c1", "name": "look_up", "arguments": {}, "requestor": "assistant"}]
    cut = '{"rows": [{"id": "r1"}, {"id": "r2'
    sims = [good_sim("g1"), {"id": "cut", "messages": [
        {"role": "assistant", "content": None, "tool_calls": calls, "turn_idx": 0},
        {"role": "tool", "id": "c1", "content": cut, "turn_idx": 1}]}]
    error = ingest_raises(wrap(sims), workdir, tmp_path)
    assert error.gate.metrics["rejected"] == 1
    assert error.gate.passed is False
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"]["unparseable_result"] == 1


def test_a_recording_with_no_turns_is_evidence_only(workdir, tmp_path):
    error = ingest_raises(wrap([good_sim("g1"), {"id": "empty", "messages": []}]), workdir,
                          tmp_path)
    ruling = json.loads((workdir / "intake_ruling.json").read_text(encoding="utf-8"))
    assert ruling["reasons"]["missing_start"] == 1
    assert error.gate.passed is False


def test_a_recording_with_no_end_marker_reads_as_complete_when_every_call_resolved(workdir, tmp_path):
    summary = ingest_payload(wrap([{"id": "plain", "messages": [plain_msg()]}]), workdir, tmp_path)
    assert summary["gate"]["pass"] is True
    assert summary["task_eligible"] == 1


# --- the seam hashes after the move -------------------------------------------


def test_fixture_hashes_are_stable_within_this_seam(tau2_small_path, workdir):
    summary = ingest.ingest_file(tau2_small_path, workdir)
    assert summary["trace_hashes"] == [
        "452ccae335ba1c9999ac120978ac32927f8265b45a2cc0e225cd541b4172fb81",
        "82818ebaba242db1498492ecb33943741acb23daf983e62e89b3a82949ebccc3",
        "b201add5d7eb96a93d8f079b63ea3067519757f7f122fe569dfde84f1193cdb2",
    ]
