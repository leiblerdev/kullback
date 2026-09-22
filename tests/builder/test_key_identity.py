"""A row homed under a composed id carries its key parts, and no write is charged for them.

The world here is an invented harbour. A dock is let out per shift, so one dock id
stands for two rows and the row is named by dock id and shift together. A listing
tool answers the rows for the shift it was asked about and leaves the shift column
null on the row itself.
"""

from __future__ import annotations

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import effects
from kullback.runner.records import Column, EntitySchema, ToolCall, Trace


def dock_schema() -> EntitySchema:
    names = {"dock_id": "hard", "shift": "hard", "seats": "hard", "zone": "hard"}
    return EntitySchema(
        tables=["docks"],
        columns=[Column(table="docks", name=name, **{"class": kind}, classified_by="rule")
                 for name, kind in sorted(names.items())],
        id_patterns={"docks.dock_id": r"^dock_\d$"},
        composite_keys={"docks": ["dock_id", "shift"]},
    )


def _call(name, args, result=None, idx=0):
    return ToolCall(id=f"c{idx}", name=name, args=args, result=result, raw_ptr=PTR)


def _trace(trace_id, calls):
    return Trace(trace_id=trace_id, raw_hash="rawhash", ingest_version="1",
                 source="test", tool_calls=calls, raw_ptr=PTR)


def test_a_row_homed_under_a_composed_id_carries_its_key_parts_as_columns(workdir):
    schema = dock_schema()
    trace = _trace("A", [
        _call("list_docks", {"shift": "early"},
              result=[{"dock_id": "dock_1", "shift": None, "seats": 4, "zone": "north"}],
              idx=0),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[], synthetic=False)
    row = state.db["docks"]["dock_1|early"]
    assert row["dock_id"] == "dock_1"
    assert row["shift"] == "early"


def test_a_write_is_never_charged_for_a_column_that_is_part_of_the_row_key():
    """A key part is a key sighting, not an effect, so it is dropped from the evidence."""
    schema = dock_schema()
    trace = _trace("A", [
        _call("get_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": None, "seats": 4, "zone": "north"},
              idx=0),
        _call("adjust_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": "early", "seats": 9, "zone": "north"},
              idx=1),
        _call("get_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": "early", "seats": 9, "zone": "north"},
              idx=2),
    ])
    seen = effects.observe_effects([trace], schema, {"adjust_dock"})
    paths = [(column.table, column.path)
             for effect in seen.get("adjust_dock", []) for column in effect.columns]
    assert ("docks", "shift") not in paths
    assert ("docks", "shift") not in [
        (row["table"], row["path"]) for rows in effects.replay_evidence(seen).values()
        for row in rows]


def test_a_non_key_column_on_the_same_row_is_still_charged():
    schema = dock_schema()
    trace = _trace("A", [
        _call("get_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": None, "seats": 4, "zone": "north"},
              idx=0),
        _call("adjust_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": "early", "seats": 9, "zone": "north"},
              idx=1),
        _call("get_dock", {"dock_id": "dock_1", "shift": "early"},
              result={"dock_id": "dock_1", "shift": "early", "seats": 9, "zone": "north"},
              idx=2),
    ])
    seen = effects.observe_effects([trace], schema, {"adjust_dock"})
    seats = [column for effect in seen.get("adjust_dock", []) for column in effect.columns
             if (column.table, column.path) == ("docks", "seats")]
    assert seats and (seats[0].before, seats[0].after) == (4, 9)
