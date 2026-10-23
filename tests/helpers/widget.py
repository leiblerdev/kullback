"""The invented widget world's one recording, and the scripted candidate that redoes it.

The world itself is tests/episode/invented.py: one widget, a read that describes it and a write that
renames it. The recording gives w1 the label striped; the candidate reads, writes and stops.
"""

from __future__ import annotations

from pathlib import Path

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, write_json

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def widget_trace(trace_id: str = "rec1", prefix: str = "c") -> Trace:
    """The recording: the user asks for the striped label, the agent reads w1 and renames it."""
    first, second = f"{prefix}1", f"{prefix}2"
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Give widget w1 the label striped.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=[first], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"widget_id": "w1"}', tool_call_ids=[first], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=[second], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"widget_id": "w1"}', tool_call_ids=[second], raw_ptr=PTR),
        Turn(idx=6, role="assistant", content="Done.", raw_ptr=PTR),
        Turn(idx=7, role="user", content="Thanks, that is the right label.", raw_ptr=PTR),
    ]
    calls = [
        ToolCall(id=first, name="describe_widget", args={"widget_id": "w1"},
                 result={"widget_id": "w1", "label": "plain"}, raw_ptr=PTR, trace_id=trace_id),
        ToolCall(id=second, name="rename_widget", args={"widget_id": "w1", "label": "striped"},
                 result={"widget_id": "w1", "label": "striped"}, raw_ptr=PTR, trace_id=trace_id),
    ]
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def write_widget_trace(root: Path) -> Path:
    """The recording stored under root/traces, where the workdir keeps its Traces."""
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    write_json(traces / "rec1.json", widget_trace().model_dump(mode="json", by_alias=True))
    return root


def rename_script(*opening: ModelReply) -> TestModel:
    """A model that says the `opening` turns, reads the widget, writes the striped label and stops, on every Run."""
    return TestModel([
        *opening,
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)
