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


def test_detect_votes_on_a_rows_envelope_and_a_bare_recording():
    adapter = Terminus2Adapter()
    confidence, reasons = adapter.detect(invented_envelope())
    assert confidence > 0
    assert any("rows list" in reason for reason in reasons)
    confidence, _ = Terminus2Adapter().detect(invented_recording())
    assert confidence > 0


def test_each_adapter_stays_quiet_on_the_others_payload():
    payload = {"simulations": [{"id": "s1", "messages": []}]}
    assert Terminus2Adapter().detect(payload)[0] == 0.0
    from kullback.builder.sources.tau2_native import Tau2NativeAdapter

    assert Tau2NativeAdapter().detect(invented_envelope())[0] == 0.0
    assert Tau2NativeAdapter().detect(invented_recording())[0] == 0.0


def test_to_trace_maps_batch_to_shell_calls():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    assert trace.trace_id == "invented-trial-1"
    assert trace.source == "terminus_2"
    assert [turn.role for turn in trace.turns] == ["user", "assistant", "user", "assistant"]
    assert len(trace.tool_calls) == 1
    (call,) = trace.tool_calls
    assert call.name == "shell"
    assert call.args == {"commands": [
        {"keystrokes": "invented-list-alpha", "duration": 0.2},
        {"keystrokes": "invented-list-beta", "duration": 0.3}]}
    assert call.latency_ms is None


def test_batch_output_lands_on_the_one_call():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    (call,) = trace.tool_calls
    assert (call.has_result, call.resolved) == (True, True)
    assert "invented output of both listings" in str(call.result)


def test_calls_cite_request_and_answer():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    (call,) = trace.tool_calls
    assert call.raw_ptr.msg_index == 1
    assert call.result_ptr is not None and call.result_ptr.msg_index == 2


def test_commands_after_prose_become_one_call():
    batch = json.dumps({"analysis": "invented", "task_complete": False,
                        "commands": [{"keystrokes": "invented-a"},
                                     {"keystrokes": "invented-b", "duration": 0.5}]})
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant",
         "content": "Invented plan in prose. ```json\n" + batch + "\n```\nInvented tail."},
        {"role": "user", "content": "New Terminal Output:\n\ninvented batch output"},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    (call,) = trace.tool_calls
    assert call.args == {"commands": [
        {"keystrokes": "invented-a"}, {"keystrokes": "invented-b", "duration": 0.5}]}
    assert (call.has_result, call.resolved) == (True, True)
    assert "invented batch output" in str(call.result)
    assert "Invented plan in prose." in str(trace.turns[1].content)
    assert "Invented tail." in str(trace.turns[1].content)


def test_assistant_prose_and_opening_turn_are_kept():
    trace = Terminus2Adapter().to_trace(invented_recording(), ctx())
    assert "Invented instruction template." in str(trace.turns[0].content)
    assert "Invented prose with no commands" in str(trace.turns[3].content)


def test_the_sidecar_carries_outcome_grouping_task_ref_and_instruction_columns():
    adapter = Terminus2Adapter()
    sidecar = adapter.sidecar(invented_recording(), invented_envelope())
    assert sidecar["task"] == "invented task text"
    assert sidecar["episode"] == 3
    assert "conversations" not in sidecar
    recording = invented_recording(conversations=invented_task_turn("# invented-seven\nSolve invented."),
                                   trial_name="invented-task-7__invented-run")
    sidecar = Terminus2Adapter().sidecar(recording, invented_envelope())
    assert sidecar["task_ref"] == {"id": "invented-task-7", "source": "invented-source"}
    assert sidecar["instruction"] == "# invented-seven\nSolve invented."
    assert sidecar["task"] == "invented task text"
    assert "conversations" not in sidecar
    recording = invented_recording(conversations=invented_task_turn("# invented\n"),
                                   trial_name="invented-task-8")
    del recording["original_source"]
    sidecar = Terminus2Adapter().sidecar(recording, invented_envelope())
    assert sidecar["task_ref"] == {"id": "invented-task-8", "source": None}
    sidecar = Terminus2Adapter().sidecar(invented_recording(), invented_envelope())
    assert sidecar["instruction"] is None
    assert sidecar["task_ref"] == {"id": "invented-trial-1", "source": "invented-source"}


def test_a_scaffold_complaint_leaves_the_call_before_it_unobserved_and_is_counted():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-one"}, {"keystrokes": "invented-two"}]})},
        {"role": "user", "content": "Invented scaffold complaint: no valid JSON found."},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    (call,) = trace.tool_calls
    assert (call.has_result, call.resolved, call.result) == (False, False, None)
    assert trace.turns[2].role == "user"
    from kullback.builder.sources import terminus_2 as adapter_mod

    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-one"}]})},
        {"role": "user", "content": "Previous response had parsing errors: invented."},
    ]
    counts = adapter_mod.contradictions(turns)
    assert counts["complaint_after_parsed"] == 1
    assert counts["output_without_commands"] == 0
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert not trace.tool_calls[0].resolved


def test_commands_on_the_last_turn_stay_unobserved():
    turns = invented_turns()[:2]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    assert not trace.tool_calls[0].resolved
    turns = ending_on_empty_commands()[:2]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    assert not trace.tool_calls[0].resolved


def test_output_after_no_command_marks_recording_damaged(tmp_path):
    from kullback.builder.sources import terminus_2 as adapter_mod

    clean = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-solo"}]})},
        {"role": "user", "content": "New Terminal Output:\n\ninvented solo output"},
    ]
    damaged = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": "Invented prose with no commands in it."},
        {"role": "user", "content": "New Terminal Output:\n\ninvented stray output"},
    ]
    assert adapter_mod.contradictions(damaged)["output_without_commands"] == 1
    envelope = {"rows": [
        {"row_idx": 0, "row": invented_recording(conversations=clean,
                                                   trial_name="invented-clean"),
         "truncated_cells": []},
        {"row_idx": 1, "row": invented_recording(conversations=clean,
                                                   trial_name="invented-clean-2"),
         "truncated_cells": []},
        {"row_idx": 2, "row": invented_recording(conversations=clean,
                                                   trial_name="invented-clean-3"),
         "truncated_cells": []},
        {"row_idx": 3, "row": invented_recording(conversations=damaged,
                                                   trial_name="invented-damaged"),
         "truncated_cells": []}]}
    target = tmp_path / "invented.json"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    workdir = tmp_path / "work"
    summary = ingest.ingest_file(target, workdir)
    assert summary["runs"] == 3
    ruling = ingest.read_intake_ruling(workdir, summary["raw_hash"])
    assert ruling["reasons"] == {"complete_record": 3, "unresolved_call": 1}
    damaged_row = [row for row in ruling["recordings"]
                   if row["trace_id"] == "invented-damaged"][0]
    assert damaged_row["standing"] == "evidence_only"


def test_unusable_durations_stay_out_of_arguments():
    for duration in (True, -1.0, "0.2", float("nan"), float("inf")):
        turns = [
            {"role": "user", "content": "Invented instruction."},
            {"role": "assistant", "content": json.dumps(
                {"commands": [{"keystrokes": "invented", "duration": duration}]})},
            {"role": "user", "content": "New Terminal Output:\n\ninvented"},
        ]
        trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
        assert trace.tool_calls[0].args == {"commands": [{"keystrokes": "invented"}]}
        assert trace.tool_calls[0].latency_ms is None


def test_empty_commands_answered_by_output_is_one_call_with_that_result():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"analysis": "done", "task_complete": True,
                                                     "commands": []})},
        {"role": "user", "content": "New Terminal Output:\n\ninvented screen"},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    (call,) = trace.tool_calls
    assert call.args == {"commands": []}
    assert (call.has_result, call.resolved) == (True, True)
    assert "invented screen" in str(call.result)


def test_empty_commands_answered_by_a_scaffold_note_is_no_call():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"analysis": "done", "task_complete": True,
                                                     "commands": []})},
        {"role": "user", "content": "Invented scaffold note with no terminal header."},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert trace.tool_calls == []
    assert [turn.role for turn in trace.turns] == ["user", "assistant", "user"]


def ending_on_empty_commands() -> list[dict]:
    """An invented recording whose last turn submits an empty command list and nothing answers."""
    return [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-solo", "duration": 0.1}]})},
        {"role": "user", "content": "New Terminal Output:\n\ninvented solo output"},
        {"role": "assistant", "content": json.dumps({"analysis": "invented", "task_complete": True,
                                                     "commands": []})},
    ]


def test_a_recording_ending_on_empty_commands_is_complete_with_no_unresolved_call(tmp_path):
    turns = ending_on_empty_commands()
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    assert len(trace.tool_calls) == 1
    (call,) = trace.tool_calls
    assert call.raw_ptr.msg_index == 1
    assert call.resolved
    turns = ending_on_empty_commands()
    envelope = {"rows": [
        {"row_idx": index,
         "row": invented_recording(conversations=turns, trial_name=f"invented-end-{index}"),
         "truncated_cells": []} for index in range(4)]}
    target = tmp_path / "invented.json"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    workdir = tmp_path / "work"
    summary = ingest.ingest_file(target, workdir)
    ruling = ingest.read_intake_ruling(workdir, summary["raw_hash"])
    assert ruling["reasons"] == {"complete_record": 4}


def test_warning_plus_output_gives_output_as_result():
    turns = [
        {"role": "user", "content": "Invented instruction."},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-one"}]})},
        {"role": "user", "content": "Invented warnings block.\n\nNew Terminal Output:\n\n"
                                    "invented output"},
    ]
    trace = Terminus2Adapter().to_trace(invented_recording(conversations=turns), ctx())
    (call,) = trace.tool_calls
    assert (call.has_result, call.resolved) == (True, True)
    assert "New Terminal Output:" in str(call.result)
    assert "Invented warnings block." in str(call.result)


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


def test_the_seam_routes_both_formats_with_no_edit_for_the_new_adapter():
    assert sources.by_name("terminus_2") is not None
    assert "terminus_2" in [adapter.name for adapter in sources.registered()]
    decision = sources.detect_format(invented_envelope())
    assert decision.winner == "terminus_2"
    payload = {"simulations": [{"id": "s1", "messages": []}]}
    assert sources.detect_format(payload).winner == "tau2_native"


def invented_task_turn(instruction: str) -> list[dict]:
    """An invented opening turn carrying the instruction between the boundary headers."""
    return [
        {"role": "user",
         "content": ("Invented scaffold preamble.\n\nTask Description:\n" + instruction
                     + "\n\nCurrent terminal state:\n\ninvented-shell-ready")},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-solo"}]})},
        {"role": "user", "content": "New Terminal Output:\n\ninvented solo output"},
    ]
