"""The Starting-state pinner (D188): rows at any depth, a keyless read homed by the call, per column.

One invented domain throughout: a delivery network of routes, each route holding the legs it is made
of, and meters that a reading tool answers for without ever naming the meter in its answer.
"""

from __future__ import annotations

import json

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.runner.records import Column, EntitySchema, Task, ToolCall, ToolSig, Trace


def _schema(hard=("status", "reading", "carrier", "surface"), tables=("routes", "legs", "meters")):
    columns = [Column(table=table, name=name, **{"class": "hard"}, classified_by="rule")
               for table in tables for name in hard]
    columns += [Column(table="routes", name="route_id", **{"class": "hard"}, classified_by="rule"),
                Column(table="legs", name="leg_id", **{"class": "hard"}, classified_by="rule"),
                Column(table="meters", name="meter_id", **{"class": "hard"}, classified_by="rule")]
    return EntitySchema(tables=sorted(tables), columns=columns,
                        id_patterns={"routes": r"^R\d+$", "legs": r"^L\d+$", "meters": r"^M\d+$"})


def _call(name, args, result=None, idx=0):
    return ToolCall(id=f"c{idx}", name=name, args=args, result=result, raw_ptr=PTR)


def _trace(trace_id, calls):
    return Trace(trace_id=trace_id, raw_hash="rawhash", ingest_version="1", source="tau2",
                 tool_calls=calls, raw_ptr=PTR)


def _sigs():
    return [ToolSig(name="search_routes", kind="read", unclassified=False),
            ToolSig(name="get_route_details", kind="read", unclassified=False),
            ToolSig(name="get_meter_reading", kind="read", unclassified=False)]


def _pinned(workdir, task_id):
    """The one row version each pinned row carries, as {(table, id): row}."""
    overlay, values = ce.load_overlay(workdir, task_id)
    return {(row.table, row.id): values[row.version_hash] for row in overlay.rows}


def test_a_row_two_levels_inside_a_result_is_pinned(workdir):
    """A search answers a list of routes, each holding its legs; the legs are rows of the world."""
    result = [{"route_id": "R1", "carrier": "north", "legs": [
        {"leg_id": "L1", "surface": "gravel"}, {"leg_id": "L2", "surface": "asphalt"}]}]
    trace = _trace("A", [_call("search_routes", {"carrier": "north"}, result=result)])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False)
    assert set(state.db["legs"]) == {"L1", "L2"}
    assert state.db["legs"]["L2"]["surface"] == "asphalt"


def test_a_parent_row_and_the_child_rows_it_embeds_are_both_pinned(workdir):
    result = {"route_id": "R1", "carrier": "north", "legs": [{"leg_id": "L1", "surface": "gravel"}]}
    trace = _trace("A", [_call("get_route_details", {"route_id": "R1"}, result=result)])
    ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])], _sigs(),
                            synthetic=False)
    pinned = _pinned(workdir, "t1")
    assert set(pinned) == {("routes", "R1"), ("legs", "L1")}
    assert pinned[("legs", "L1")]["surface"] == "gravel"


def test_a_leg_mentioned_inside_a_route_does_not_rewrite_the_leg_the_corpus_states_on_its_own(workdir):
    """What a leg carries inside a route is about that route; the leg's own reading is the leg."""
    inside = _call("get_route_details", {"route_id": "R1"},
                   result={"route_id": "R1", "legs": [{"leg_id": "L1", "surface": "gravel"}]}, idx=0)
    itself = _call("search_routes", {"carrier": "north"},
                   result=[{"leg_id": "L1", "surface": "asphalt", "status": "open"}], idx=1)
    state = ce.build_starting_state([_trace("A", [inside, itself])], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    assert state.db["legs"]["L1"] == {"leg_id": "L1", "surface": "asphalt", "status": "open"}
    pins = json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))
    assert pins["nested_sightings_of_a_row_already_stated"] == 1


def test_a_leg_only_ever_mentioned_inside_a_route_is_completed_from_the_legs_the_corpus_stated(workdir):
    """The mention is kept, the columns it never carried are shaped from the stated rows (D40)."""
    inside = _call("get_route_details", {"route_id": "R1"},
                   result={"route_id": "R1", "legs": [{"leg_id": "L9", "surface": "gravel"}]}, idx=0)
    stated = _call("search_routes", {"carrier": "north"},
                   result=[{"leg_id": "L1", "surface": "asphalt", "status": "open"}], idx=1)
    schema = _schema()
    state = ce.build_starting_state([_trace("A", [inside, stated])], schema, workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs())
    assert state.db["legs"]["L9"] == {"leg_id": "L9", "surface": "gravel", "status": "open"}
    assert "L9" in schema.synthetic_rows
    assert any("L9 was only ever mentioned in part" in line for line in state.assumptions)


def test_the_walk_stops_at_the_depth_cap_and_says_how_often(workdir):
    """A row nested past the cap is counted rather than quietly going missing."""
    deep: dict = {"leg_id": "L9", "surface": "gravel"}
    for _ in range(ce.ROW_WALK_DEPTH + 2):
        deep = {"inside": deep}
    top = {"route_id": "R1", "carrier": "north", "buried": deep}
    trace = _trace("A", [_call("search_routes", {"carrier": "north"}, result=[top])])
    ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])], _sigs(),
                            synthetic=False)
    pins = json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))
    assert set(_pinned(workdir, "t1")) == {("routes", "R1")}
    assert pins["depth_cap"] == ce.ROW_WALK_DEPTH
    assert pins["results_deeper_than_the_cap"] >= 1


def test_the_walk_ends_on_a_result_that_points_back_at_itself():
    """No recorded result can be cyclic; the walk still ends on one, and keeps what it did reach."""
    loop: dict = {"route_id": "R1", "carrier": "north"}
    loop["itself"] = loop
    rows, capped = ce.walk_result_rows(_schema(), loop, {"carrier": "north"})
    assert [(table, row_id) for table, row_id, _row, _depth in rows] == [("routes", "R1")]
    assert capped == 0


def test_a_result_without_a_key_pins_its_columns_on_the_row_the_call_named(workdir):
    """The reading names no meter; the call did, so the columns are the named meter's own."""
    trace = _trace("A", [_call("get_meter_reading", {"meter_id": "M1"}, result={"reading": "41"})])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False)
    assert state.db["meters"]["M1"]["reading"] == "41"
    assert _pinned(workdir, "t1")[("meters", "M1")]["reading"] == "41"


def test_a_partial_reading_never_replaces_the_columns_a_whole_row_showed(workdir):
    whole = _call("get_route_details", {"meter_id": "M1"},
                  result={"meter_id": "M1", "reading": "12", "status": "live"}, idx=0)
    later = _call("get_meter_reading", {"meter_id": "M1"}, result={"reading": "41"}, idx=1)
    state = ce.build_starting_state([_trace("A", [whole, later])], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    assert state.db["meters"]["M1"] == {"meter_id": "M1", "reading": "41", "status": "live"}


def test_a_task_pins_the_value_its_own_run_read_first_where_the_corpus_disagrees(workdir):
    """Two Tasks read one meter differently; the shared world holds one value and each Task its own."""
    first = _trace("A", [_call("get_meter_reading", {"meter_id": "M1"}, result={"reading": "41"})])
    second = _trace("B", [_call("get_meter_reading", {"meter_id": "M1"}, result={"reading": "77"})])
    tasks = [Task(id="t_low", run_ids=["A"]), Task(id="t_high", run_ids=["B"])]
    ce.build_starting_state([first, second], _schema(), workdir, tasks, _sigs(), synthetic=False)
    assert _pinned(workdir, "t_low")[("meters", "M1")]["reading"] == "41"
    assert _pinned(workdir, "t_high")[("meters", "M1")]["reading"] == "77"
    pins = json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))
    assert pins["tasks"]["t_low"]["columns_the_corpus_disagrees_on"] == 1
    assert pins["totals"]["columns_homed"] == 4


def test_a_column_only_a_later_run_read_is_pinned_beside_the_earliest_value(workdir):
    """One Task, two Runs: the earliest value of a column it shares, and the column only one read."""
    first = _trace("A", [_call("get_route_details", {"route_id": "R1"},
                               result={"route_id": "R1", "carrier": "north"})])
    second = _trace("B", [_call("get_route_details", {"route_id": "R1"},
                                result={"route_id": "R1", "carrier": "south", "status": "open"})])
    ce.build_starting_state([first, second], _schema(), workdir, [Task(id="t1", run_ids=["A", "B"])],
                            _sigs(), synthetic=False)
    row = _pinned(workdir, "t1")[("routes", "R1")]
    assert row == {"route_id": "R1", "carrier": "north", "status": "open"}


def test_two_runs_that_read_one_row_differently_are_recorded_as_a_disagreement(workdir):
    first = _trace("A", [_call("get_route_details", {"route_id": "R1"},
                               result={"route_id": "R1", "carrier": "north"})])
    second = _trace("B", [_call("get_route_details", {"route_id": "R1"},
                                result={"route_id": "R1", "carrier": "south"})])
    state = ce.build_starting_state([first, second], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A", "B"])], _sigs(), synthetic=False)
    assert any("runs disagree on routes row R1" in line for line in state.assumptions)


def test_two_runs_that_read_different_parts_of_one_row_do_not_disagree(workdir):
    first = _trace("A", [_call("get_route_details", {"route_id": "R1"},
                               result={"route_id": "R1", "carrier": "north"})])
    second = _trace("B", [_call("get_route_details", {"route_id": "R1"},
                                result={"route_id": "R1", "status": "open"})])
    state = ce.build_starting_state([first, second], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A", "B"])], _sigs(), synthetic=False)
    assert not any("runs disagree" in line for line in state.assumptions)


def test_a_scalar_reading_is_pinned_through_the_tools_own_reader(workdir):
    """The reading is a sentence; only the tool's reader (D176) says which columns it asserts."""
    def read(tool, result):
        return {"reading": str(result).split()[-1]} if tool == "get_meter_reading" else None

    trace = _trace("A", [_call("get_meter_reading", {"meter_id": "M1"}, result="the meter reads 41")])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False, read_result=read)
    assert state.db["meters"]["M1"]["reading"] == "41"
    assert _pinned(workdir, "t1")[("meters", "M1")]["reading"] == "41"


def test_a_scalar_reading_with_no_reader_pins_nothing_and_is_counted(workdir):
    trace = _trace("A", [_call("get_meter_reading", {"meter_id": "M1"}, result="the meter reads 41")])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False)
    pins = json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))
    assert state.db["meters"] == {}
    assert pins["unread_partial_results"] == 1
    assert pins["tasks"]["t1"]["rows"] == 0


@pytest.mark.parametrize("tool_name", ["get_legs_for_route", "list_legs_by_route"])
def test_a_keyless_answer_is_not_filed_under_the_parent_the_call_addressed_it_by(workdir, tool_name):
    """The name says legs, addressed by a route, so the route's id homes nothing (D180)."""
    sigs = [ToolSig(name=tool_name, kind="read", unclassified=False)]
    trace = _trace("A", [_call(tool_name, {"route_id": "R1"}, result={"surface": "gravel"})])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    sigs, synthetic=False)
    assert state.db["routes"] == {}
