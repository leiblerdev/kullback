"""Tests for the terminal adapter: detection both ways, the shell mapping, batch results.

Every recording here is invented; the shapes mirror the format's wire layout and no content
comes from any real corpus.
"""

from __future__ import annotations

import json
from typing import Any

from kullback.builder import ingest, sources
from kullback.builder.sources.terminus_2 import Terminus2Adapter


def invented_turns() -> list[dict]:
    """An invented recording: instruction, one batch of two commands, its output, prose."""
    batch = json.dumps({
        "analysis": "invented reasoning about listing files",
        "plan": "run two listings",
        "task_complete": False,
        "commands": [
            {"keystrokes": "invented-list-alpha", "duration": 0.2},
            {"keystrokes": "invented-list-beta", "duration": 0.3},
        ],
    })
    return [
        {"role": "user",
         "content": "Invented instruction template.\n\nCurrent terminal state:\n\ninvented-shell-ready"},
        {"role": "assistant", "content": batch},
        {"role": "user",
         "content": "New Terminal Output:\n\ninvented output of both listings"},
        {"role": "assistant", "content": "Invented prose with no commands in it."},
    ]


def invented_recording(**overrides: Any) -> dict:
    """One invented recording with trial markers; overrides replace columns."""
    recording = {
        "conversations": invented_turns(),
        "trial_name": "invented-trial-1",
        "run_id": "invented-run-1",
        "episode": 3,
        "original_source": "invented-source",
        "task": "invented task text",
        "result": "",
    }
    recording.update(overrides)
    return recording


def invented_envelope(count: int = 2) -> dict:
    """An invented rows envelope holding count recordings."""
    rows = []
    for index in range(count):
        recording = invented_recording(trial_name=f"invented-trial-{index}")
        rows.append({"row_idx": index, "row": recording, "truncated_cells": []})
    return {"rows": rows}


def ctx(index: int = 0):
    return sources.MapContext(raw_hash="0" * 64, index=index, environment={},
                              ingest_version="test")


def test_detect_votes_on_rows_envelope():
    adapter = Terminus2Adapter()
    confidence, reasons = adapter.detect(invented_envelope())
    assert confidence > 0
    assert any("rows list" in reason for reason in reasons)


def test_detect_votes_on_bare_recording():
    confidence, _ = Terminus2Adapter().detect(invented_recording())
    assert confidence > 0


def test_detect_stays_quiet_on_first_adapter_payload():
    payload = {"simulations": [{"id": "s1", "messages": []}]}
    assert Terminus2Adapter().detect(payload)[0] == 0.0


def test_first_adapter_stays_quiet_on_these_payloads():
    from kullback.builder.sources.tau2_native import Tau2NativeAdapter

    assert Tau2NativeAdapter().detect(invented_envelope())[0] == 0.0
    assert Tau2NativeAdapter().detect(invented_recording())[0] == 0.0


def test_seam_routes_envelope_to_new_adapter():
    decision = sources.detect_format(invented_envelope())
    assert decision.winner == "terminus_2"


def test_seam_still_routes_first_adapter_payload():
    payload = {"simulations": [{"id": "s1", "messages": []}]}
    assert sources.detect_format(payload).winner == "tau2_native"


def test_recordings_unwrap_rows():
    envelope = invented_envelope(count=3)
    recordings = list(Terminus2Adapter().recordings(envelope))
    assert [recording["trial_name"] for recording in recordings] == [
        "invented-trial-0", "invented-trial-1", "invented-trial-2"]


def test_to_trace_maps_batch_to_shell_calls():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    assert trace.trace_id == "invented-trial-1"
    assert trace.source == "terminus_2"
    assert [turn.role for turn in trace.turns] == ["user", "assistant", "user", "assistant"]
    assert len(trace.tool_calls) == 2
    assert {call.name for call in trace.tool_calls} == {"shell"}
    assert [call.args for call in trace.tool_calls] == [
        {"keystrokes": "invented-list-alpha"}, {"keystrokes": "invented-list-beta"}]
    assert [call.latency_ms for call in trace.tool_calls] == [200.0, 300.0]


def test_batch_output_lands_on_last_call():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    first, last = trace.tool_calls
    assert (first.has_result, first.resolved, first.result) == (False, False, None)
    assert (last.has_result, last.resolved) == (True, True)
    assert "invented output of both listings" in str(last.result)


def test_calls_cite_request_and_answer():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    first, last = trace.tool_calls
    assert first.raw_ptr.msg_index == 1
    assert last.result_ptr is not None and last.result_ptr.msg_index == 2


def test_assistant_prose_and_opening_turn_are_kept():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    assert "Invented instruction template." in str(trace.turns[0].content)
    assert "Invented prose with no commands" in str(trace.turns[3].content)


def test_sidecar_carries_outcome_and_grouping_columns():
    adapter = Terminus2Adapter()
    sidecar = adapter.sidecar(invented_recording(), invented_envelope())
    assert sidecar["task"] == "invented task text"
    assert sidecar["episode"] == 3
    assert "conversations" not in sidecar


def test_batch_before_complaint_leaves_calls_unobserved():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-one"}, {"keystrokes": "invented-two"}]})},
        {"role": "user", "content": "Invented scaffold complaint: no valid JSON found."},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 2
    assert all(not call.resolved and not call.has_result for call in trace.tool_calls)
    assert trace.turns[2].role == "user"


def test_last_turn_commands_stay_unobserved():
    turns = invented_turns()[:2]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 2
    assert all(not call.resolved for call in trace.tool_calls)


def test_prose_broken_and_empty_turns_yield_no_calls():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": "Invented prose, no JSON at all."},
        {"role": "user", "content": "New Terminal Output:\n\ninvented"},
        {"role": "assistant", "content": '{"analysis": "invented", "commands": ['},
        {"role": "user", "content": "New Terminal Output:\n\ninvented"},
        {"role": "assistant", "content": json.dumps({"analysis": "done", "task_complete": True,
                                                     "commands": []})},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert trace.tool_calls == []
    assert len(trace.turns) == 6


def test_unusable_durations_stay_unrecorded():
    from kullback.builder.sources import terminus_2 as adapter_mod

    for duration in (True, -1.0, "0.2", float("nan"), float("inf")):
        turns = [
            {"role": "user", "content": "Invented instruction."},
            {"role": "assistant", "content": json.dumps(
                {"commands": [{"keystrokes": "invented", "duration": duration}]})},
            {"role": "user", "content": "New Terminal Output:\n\ninvented"},
        ]
        trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
        assert trace.tool_calls[0].latency_ms is None
    assert adapter_mod._latency_ms(0.2) == 200.0


def test_trace_id_falls_back_without_trial_marker():
    recording = invented_recording()
    del recording["trial_name"]
    trace = Terminus2Adapter().to_trace(recording, ctx(index=4))
    assert trace.trace_id == f"{'0' * 12}-4"


def test_ingest_file_publishes_invented_envelope(tmp_path):
    single = json.dumps({"analysis": "invented", "task_complete": False,
                           "commands": [{"keystrokes": "invented-solo", "duration": 0.1}]})
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": single},
        {"role": "user", "content": "New Terminal Output:\n\ninvented solo output"},
    ]
    envelope = {"rows": [
        {"row_idx": index,
         "row": invented_recording(conversations=turns, trial_name=f"invented-trial-{index}"),
         "truncated_cells": []} for index in range(4)]}
    target = tmp_path / "invented.json"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    workdir = tmp_path / "work"
    summary = ingest.ingest_file(target, workdir)
    assert summary["format"] == "terminus_2"
    assert summary["runs"] == 4
    assert summary["gate"]["pass"] is True
    assert len(list((workdir / "traces").glob("*.json"))) == 4
    assert len(list((workdir / "grader").glob("*.json"))) == 4


def test_new_adapter_needs_no_seam_edit():
    assert sources.by_name("terminus_2") is not None
    assert "terminus_2" in [adapter.name for adapter in sources.registered()]
