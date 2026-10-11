"""The synthetic seam: a store Task with no recording reads its own instruction (D360)."""

from __future__ import annotations

import asyncio
import json

from kullback.spec import intent_tools as T

INTENT = "Please retie my teal kite at length 99."


def _store(root, task):
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "tasks" / f"{task['id']}.json").write_text(json.dumps(task))
    (root / T.SYNTHETIC_MARKER).write_text(json.dumps({"format": 1}))
    return root


def test_a_store_task_with_no_recording_reads_its_instruction_as_one_turn(tmp_path):
    _store(tmp_path, {"id": "t1", "run_ids": [], "intent": INTENT})
    found = T.recordings_of(tmp_path, "t1")
    assert found.recordings == {"description": [T.UserTurn(index=0, text=INTENT, agent_before="")]}


def test_a_store_task_with_no_instruction_reads_as_no_recording(tmp_path):
    _store(tmp_path, {"id": "t1", "run_ids": [], "intent": None})
    assert T.recordings_of(tmp_path, "t1").recordings == {}


def test_a_store_task_with_user_turns_keeps_its_recordings(tmp_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": ["rec1"]}))
    (tmp_path / "traces" / "rec1.json").write_text(json.dumps({
        "trace_id": "rec1", "raw_hash": "h", "ingest_version": "v", "source": "s", "turns": [
            {"idx": 0, "role": "user", "content": "Spoken, not written.",
             "tool_call_ids": [], "raw_ptr": {"file_hash": "h"}}],
        "tool_calls": [], "raw_ptr": {"file_hash": "h"}}))
    (tmp_path / T.SYNTHETIC_MARKER).write_text(json.dumps({"format": 1}))
    found = T.recordings_of(tmp_path, "t1")
    assert list(found.recordings) == ["rec1"]


def test_a_recorded_workdir_without_the_marker_is_unchanged(tmp_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": ["rec1", "quiet"]}))
    spoken = {
        "trace_id": "rec1", "raw_hash": "h", "ingest_version": "v", "source": "s", "turns": [
            {"idx": 0, "role": "user", "content": "Spoken, not written.",
             "tool_call_ids": [], "raw_ptr": {"file_hash": "h"}}],
        "tool_calls": [], "raw_ptr": {"file_hash": "h"}}
    (tmp_path / "traces" / "rec1.json").write_text(json.dumps(spoken))
    (tmp_path / "traces" / "quiet.json").write_text(json.dumps(dict(spoken, trace_id="quiet", turns=[])))
    found = T.recordings_of(tmp_path, "t1")
    assert [turn.text for turn in found.recordings["rec1"]] == ["Spoken, not written."]
    assert "quiet" not in found.recordings and "description" not in found.recordings


def test_a_fact_kept_off_the_description_is_volunteered(tmp_path):
    _store(tmp_path, {"id": "t1", "run_ids": [], "intent": INTENT})
    state = T.MinerState(found=T.recordings_of(tmp_path, "t1"))
    tools = {tool.name: tool for tool in T.miner_tools(state)}
    result = asyncio.run(tools["add_facts"].run(
        {"facts": [{"text": "my teal kite at length 99", "recording": "description", "turn": 0,
                    "stance": "volunteered"}]}))
    assert not result.is_error
    assert len(state.facts) == 1
    fact = state.facts[0]
    assert (fact.stance, fact.echo, fact.source.recording, fact.source.turn) == \
        ("volunteered", False, "description", 0)
    assert T.counts(fact)
