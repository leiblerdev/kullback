"""Claude Code session transcripts enter through the adapter seam and map to Traces with pointers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kullback.builder import ingest


def line(kind, uuid, parent, content, session="s1", at="2026-01-01T00:00:00.000Z"):
    role = "assistant" if kind == "assistant" else "user"
    return {"type": kind, "uuid": uuid, "parentUuid": parent, "sessionId": session,
            "timestamp": at, "message": {"role": role, "content": content}}


def session_lines(session="s1"):
    """A hand-written session: a request, two tool calls (one fails), and a closing answer."""
    return [
        {"type": "summary", "summary": "toy session", "leafUuid": "u6"},
        line("user", "u1", None, "Rename the notes file.", session),
        line("assistant", "u2", "u1", [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "Listing first."},
            {"type": "tool_use", "id": "tu1", "name": "list_files", "input": {"path": "."}}],
            session, "2026-01-01T00:00:01.000Z"),
        line("user", "u3", "u2", [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "notes.txt"}],
            session, "2026-01-01T00:00:01.500Z"),
        line("assistant", "u4", "u3", [
            {"type": "tool_use", "id": "tu2", "name": "move_file",
             "input": {"from": "notes.txt", "to": "todo.txt"}}],
            session, "2026-01-01T00:00:02.000Z"),
        line("user", "u5", "u4", [
            {"type": "tool_result", "tool_use_id": "tu2", "is_error": True,
             "content": "permission denied: read-only folder"}],
            session, "2026-01-01T00:00:02.250Z"),
        line("assistant", "u6", "u5", [{"type": "text", "text": "The folder is read-only."}],
             session),
        {"type": "result", "sessionId": session, "subtype": "success"},
    ]


def ingest_lines(lines, workdir, tmp_path, name="session.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(item) + "\n" for item in lines), encoding="utf-8")
    raw = ingest.store_raw(path, workdir)
    assert raw.format_detected == "claude_code_jsonl"
    return raw, ingest.derive_traces(raw.raw_hash, workdir)


def resolve(lines, ptr):
    """The value a pointer names: the line at sim_index, then the section's path into it."""
    node, rest = lines[ptr.sim_index], ptr.section or ""
    while rest:
        step = re.match(r"\.?([A-Za-z_]+)|\[(\d+)\]", rest)
        name, position = step.groups()
        node, rest = (node[name] if name else node[int(position)]), rest[step.end():]
    return node


def assert_every_pointer_resolves(raw, traces, workdir):
    text = Path(ingest.raw_path(raw.raw_hash, workdir)).read_text(encoding="utf-8")
    lines = [json.loads(row) for row in text.splitlines() if row.strip()]
    for trace in traces:
        for turn in trace.turns:
            message = resolve(lines, turn.raw_ptr)
            assert message["role"] in ("user", "assistant")
            for piece in (turn.content or "").splitlines():
                content = message["content"]
                assert piece == content or piece in [b.get("text") for b in content]
        for call in trace.tool_calls:
            block = resolve(lines, call.raw_ptr)
            assert (block["type"], block["id"], block["name"]) == ("tool_use", call.id, call.name)
            if call.result_ptr is not None:
                answer = resolve(lines, call.result_ptr)
                assert (answer["type"], answer["tool_use_id"]) == ("tool_result", call.id)
                content = answer["content"]
                as_blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
                assert (call.error.payload == content) if call.error else (call.result == as_blocks)


def test_a_session_is_one_trace_whose_tool_use_and_tool_result_pair_by_id_with_the_error_classed(
        workdir, tmp_path):
    raw, traces = ingest_lines(session_lines(), workdir, tmp_path)
    [trace] = traces
    assert (trace.trace_id, trace.source) == ("s1", "claude_code_jsonl")
    assert (trace.system_prompt, trace.tools_declared) == (None, None)
    assert [turn.role for turn in trace.turns] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert trace.turns[1].content == "Listing first."
    listed, moved = trace.tool_calls
    assert (listed.name, listed.args) == ("list_files", {"path": "."})
    assert listed.result == [{"type": "text", "text": "notes.txt"}]
    assert listed.latency_ms == 500.0
    assert (listed.raw_ptr.sim_index, listed.raw_ptr.section) == (2, "message.content[2]")
    assert (listed.result_ptr.sim_index, listed.result_ptr.section) == (3, "message.content[0]")
    assert moved.result is None and moved.error.class_ == "permission_denied"
    assert moved.resolved and moved.has_result
    assert_every_pointer_resolves(raw, traces, workdir)


def test_summary_and_result_lines_are_not_turns_and_are_counted_in_the_sidecar(workdir, tmp_path):
    raw, traces = ingest_lines(session_lines(), workdir, tmp_path)
    [trace] = traces
    assert len(trace.turns) == 6
    sidecar = json.loads(ingest.grader_file(trace, workdir).read_text(encoding="utf-8"))
    assert "non_turn_lines" in json.dumps(sidecar)
    assert '"summary": 1' in json.dumps(sidecar) and '"result": 1' in json.dumps(sidecar)
    assert "parentUuid order" in json.dumps(sidecar)
    assert_every_pointer_resolves(raw, traces, workdir)


def test_turns_follow_the_parent_chain_and_a_broken_chain_falls_back_to_file_order(
        workdir, tmp_path):
    lines = session_lines()
    lines[3], lines[4] = lines[4], lines[3]  # file order now disagrees with the chain
    raw, traces = ingest_lines(lines, workdir, tmp_path, "swapped.jsonl")
    assert [turn.raw_ptr.sim_index for turn in traces[0].turns] == [1, 2, 4, 3, 5, 6]
    assert_every_pointer_resolves(raw, traces, workdir)
    broken = session_lines("s2")
    broken[4]["parentUuid"] = "missing"
    raw, traces = ingest_lines(broken, workdir, tmp_path, "broken.jsonl")
    assert [turn.raw_ptr.sim_index for turn in traces[0].turns] == [1, 2, 3, 4, 5, 6]
    sidecar = json.loads(ingest.grader_file(traces[0], workdir).read_text(encoding="utf-8"))
    assert "broken, file order used" in json.dumps(sidecar)
    assert_every_pointer_resolves(raw, traces, workdir)


def test_a_text_result_that_opens_with_a_bracket_is_text_and_passes_the_gate(workdir, tmp_path):
    lines = session_lines()
    lines[3]["message"]["content"][0]["content"] = "[main 1a2b] rename notes"
    raw, traces = ingest_lines(lines, workdir, tmp_path, "bracket.jsonl")
    listed = traces[0].tool_calls[0]
    assert listed.result == [{"type": "text", "text": "[main 1a2b] rename notes"}]
    assert listed.truncated is False
    assert ingest.gate_ingest(traces, workdir, raw_hash=raw.raw_hash).passed
    assert_every_pointer_resolves(raw, traces, workdir)


def test_two_sessions_in_one_file_are_two_traces(workdir, tmp_path):
    raw, traces = ingest_lines(session_lines("s1") + session_lines("s2"), workdir, tmp_path)
    assert [trace.trace_id for trace in traces] == ["s1", "s2"]
    assert [trace.raw_ptr.sim_index for trace in traces] == [0, 1]
    assert_every_pointer_resolves(raw, traces, workdir)
