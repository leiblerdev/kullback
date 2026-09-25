"""Tests for the Claude Code JSONL adapter: transcript rows map to paired traces."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kullback.builder import ingest
from kullback.builder.sources import by_name


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(session: str, kind: str, role: str, content: list[dict]) -> dict:
    return {"type": kind, "sessionId": session,
            "message": {"role": role, "content": content}}


def _paired_rows(session: str = "sess-1") -> list[dict]:
    return [
        _row(session, "user", "user", [{"type": "text", "text": "hi"}]),
        _row(session, "assistant", "assistant", [
            {"type": "text", "text": "looking it up"},
            {"type": "tool_use", "id": "call-1", "name": "tool_a",
             "input": {"field_x": 1}},
        ]),
        _row(session, "assistant", "assistant", [
            {"type": "tool_result", "tool_use_id": "call-1",
             "content": "{\"field_y\": 2}"},
        ]),
    ]


def test_the_transcript_file_passes_the_intake_gate_with_three_traces(
        world_dir, tmp_path):
    """The shipped transcript dry-runs clean with every call paired."""
    result = ingest.dry_run(world_dir("world_b") / "traces_small.jsonl", tmp_path)
    assert result["format"] == "claude_code_jsonl"
    assert result["passed"] is True, result["failures"]
    assert result["traces"] == 3
    assert result["tools"] > 0


def test_the_transcript_and_its_twin_give_the_same_traces_turns_and_tool_calls(
        world_dir, workdir, tmp_path):
    """The transcript and the mapped twin agree on trace count, turns per role,
    and tool call names with arguments."""
    twin = world_dir("world_b") / "traces_small_tau2.json"
    transcript = world_dir("world_b") / "traces_small.jsonl"
    raw_transcript = ingest.store_raw(transcript, workdir)
    raw_twin = ingest.store_raw(twin, workdir)
    from_transcript = ingest.derive_traces(raw_transcript.raw_hash, workdir)
    from_twin = ingest.derive_traces(raw_twin.raw_hash, workdir)
    assert len(from_transcript) == len(from_twin) == 3
    assert sorted(len(trace.turns) for trace in from_transcript) == sorted(
        len(trace.turns) for trace in from_twin)
    assert Counter(turn.role for trace in from_transcript for turn in trace.turns) == Counter(
        turn.role for trace in from_twin for turn in trace.turns)

    def calls(traces):
        return sorted((call.name, json.dumps(call.args, sort_keys=True))
                      for trace in traces for call in trace.tool_calls)

    assert calls(from_transcript) == calls(from_twin)


def test_a_tool_use_with_no_answer_fails_the_gate_naming_the_call(workdir, tmp_path):
    """An unpaired call is evidence-only, and the gate names it as having no result."""
    rows = _paired_rows()[:2]
    raw = ingest.store_raw(_write_jsonl(tmp_path / "unpaired.jsonl", rows), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert len(traces) == 1
    assert traces[0].tool_calls[0].resolved is False
    gate = ingest.gate_ingest(traces, workdir, raw_hash=raw.raw_hash)
    assert gate.passed is False
    assert any("call-1" in failure and "has no result" in failure
               for failure in gate.failures)


def test_rows_from_two_conversations_give_two_recordings(tmp_path):
    """The session id splits rows, so two conversations map to two recordings."""
    rows = _paired_rows("sess-1") + _paired_rows("sess-2")
    path = _write_jsonl(tmp_path / "two.jsonl", rows)
    document = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    recordings = list(by_name("claude_code_jsonl").recordings(document))
    assert [recording["id"] for recording in recordings] == ["sess-1", "sess-2"]
    assert all(len(recording["rows"]) == 3 for recording in recordings)
