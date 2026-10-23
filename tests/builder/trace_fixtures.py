"""Recorded calls, traces and pinned rows the Builder's tests build by hand.

Each helper was copied word for word across several test files; it lives here once instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PTR
from kullback.ai.provider import TestModel
from kullback.builder import compile_env as ce
from kullback.runner.records import ToolCall, Trace, Turn
from tests.helpers.scripted import reply


def _call(name, args, result=None, idx=0):
    return ToolCall(id=f"c{idx}", name=name, args=args, result=result, raw_ptr=PTR)


def _trace(trace_id, calls):
    return Trace(trace_id=trace_id, raw_hash="rawhash", ingest_version="1", source="test",
                 tool_calls=calls, raw_ptr=PTR)


def one_trace(trace_id: str, calls: list[dict], tools_declared=None) -> Trace:
    return Trace(
        trace_id=trace_id,
        raw_hash="h",
        ingest_version="test",
        source="synthetic",
        tools_declared=tools_declared,
        tool_calls=[ToolCall(**{"raw_ptr": PTR, **c}) for c in calls],
        raw_ptr=PTR,
    )


def call_turns(ask: str, calls: list[ToolCall]) -> list[Turn]:
    """A greeting, the user's ask, one assistant turn and one tool turn per call, then the close."""
    turns = [Turn(idx=0, role="assistant", content="How can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content=ask, raw_ptr=PTR)]
    for one in calls:
        turns.append(Turn(idx=len(turns), role="assistant", content=None,
                          tool_call_ids=[one.id], raw_ptr=PTR))
        turns.append(Turn(idx=len(turns), role="tool", content=json.dumps(one.result),
                          tool_call_ids=[one.id], raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="assistant", content="Done.", raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="user", content="Thanks. ###STOP###", raw_ptr=PTR))
    return turns


def sig_by_name(sigs, name):
    return next(s for s in sigs if s.name == name)


def col(schema, table, name):
    return next(c for c in schema.columns if c.table == table and c.name == name)


def _pinned(workdir, task_id):
    """The one row version each pinned row carries, as {(table, id): row}."""
    overlay, values = ce.load_overlay(workdir, task_id)
    return {(row.table, row.id): values[row.version_hash] for row in overlay.rows}


def _pins(workdir):
    return json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _candidate() -> TestModel:
    return TestModel([reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
                      reply("Done, widget w1 is labelled striped.")], loop=True)
