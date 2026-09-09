"""The Starting-state pinner (D188): rows at any depth, a keyless read homed by the call, per column.

One invented domain throughout: a delivery network of routes, each route holding the legs it is made
of, and meters that a reading tool answers for without ever naming the meter in its answer.
"""

from __future__ import annotations

import json

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.runner import route
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


# --- D197: a row the Task reads twice, with the column moving and no write between the reads ---


def _reader_router(workdir, task_id, db):
    """A Router over the Task's overlay with no tool code and no recordings behind it.

    Every call it is given is answered with an error, which is beside the point: what is under test
    is the world the call leaves behind, which is what a tool body would have read.
    """
    overlay, values = ce.load_overlay(workdir, task_id)
    return route.Router(starting_state=json.loads(json.dumps(db)), overlay=overlay,
                        overlay_rows=values, tool_sigs=_sigs())


def _read_twice(first, second, third=None):
    """One trace reading one meter two or three times, each read answering its own value."""
    results = [first, second] + ([third] if third is not None else [])
    return _trace("A", [_call("get_meter_reading", {"meter_id": "M1"},
                              result={"meter_id": "M1", "reading": value}, idx=index)
                        for index, value in enumerate(results)])


def test_two_reads_of_one_row_with_a_column_that_moved_are_served_in_call_order(workdir):
    """The meter read 41 and then 77 with nothing written between: both values, in that order."""
    state = ce.build_starting_state([_read_twice("41", "77")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    router = _reader_router(workdir, "t1", state.db)
    assert router.state.row("meters", "M1")["reading"] == "41"
    router.route("get_meter_reading", {"meter_id": "M1"})
    assert router.state.row("meters", "M1")["reading"] == "41"
    router.route("get_meter_reading", {"meter_id": "M1"})
    assert router.state.row("meters", "M1")["reading"] == "77"


def test_a_read_past_the_end_of_the_sequence_holds_the_last_recorded_value(workdir):
    state = ce.build_starting_state([_read_twice("41", "77")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    router = _reader_router(workdir, "t1", state.db)
    for _ in range(3):
        router.route("get_meter_reading", {"meter_id": "M1"})
    assert router.state.row("meters", "M1")["reading"] == "77"


def test_a_column_that_did_not_move_keeps_one_pin_and_no_sequence(workdir):
    state = ce.build_starting_state([_read_twice("41", "41")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    overlay, _ = ce.load_overlay(workdir, "t1")
    router = _reader_router(workdir, "t1", state.db)
    router.route("get_meter_reading", {"meter_id": "M1"})
    router.route("get_meter_reading", {"meter_id": "M1"})
    assert overlay.steps == []
    assert router.state.row("meters", "M1")["reading"] == "41"


def test_a_read_the_recording_never_made_is_answered_from_the_pin(workdir):
    """The sequence is served to the reads the Task recorded, not to every call of that tool."""
    state = ce.build_starting_state([_read_twice("41", "77")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    router = _reader_router(workdir, "t1", state.db)
    router.route("get_meter_reading", {"meter_id": "M2"})
    router.route("get_meter_reading", {"meter_id": "M2"})
    assert router.state.row("meters", "M1")["reading"] == "41"


def test_a_write_between_two_reads_ends_the_sequence_and_the_write_stands(workdir):
    """A row a write touched is not time-varying: what a later read sees is the write's own value."""
    sigs = _sigs() + [ToolSig(name="set_meter_reading", kind="write", unclassified=False)]
    trace = _trace("A", [
        _call("get_meter_reading", {"meter_id": "M1"}, result={"meter_id": "M1", "reading": "41"}, idx=0),
        _call("set_meter_reading", {"meter_id": "M1", "reading": "99"},
              result={"meter_id": "M1", "reading": "99"}, idx=1),
        _call("get_meter_reading", {"meter_id": "M1"}, result={"meter_id": "M1", "reading": "99"}, idx=2)])
    state = ce.build_starting_state([trace], _schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    sigs, synthetic=False)
    overlay, _ = ce.load_overlay(workdir, "t1")
    router = _reader_router(workdir, "t1", state.db)
    router.route("get_meter_reading", {"meter_id": "M1"})
    router.route("get_meter_reading", {"meter_id": "M1"})
    assert overlay.steps == []
    assert router.state.row("meters", "M1")["reading"] == "41"


def test_the_sequence_a_task_pinned_is_counted_per_task_and_per_table(workdir):
    state = ce.build_starting_state([_read_twice("41", "77", "77")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    pins = json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))
    assert state.db["meters"]["M1"]["reading"] == "77"
    assert pins["tasks"]["t1"]["columns_time_varying"] == 1
    assert pins["totals"]["sequences_served"] == 3
    assert pins["columns_time_varying_by_table"] == {"meters": 1}


def test_the_world_a_body_is_scored_on_carries_the_value_that_calls_own_read_saw(workdir):
    """The scoring path serves the sequence the replay serves, or a body is punished for a state
    the replay would have given it."""
    state = ce.build_starting_state([_read_twice("41", "77")], _schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    states = ce.call_starting_states(state.db, state.overlays, ce.overlay_values(workdir),
                                     {"c0": "t1", "c1": "t1"})
    assert states["c0"]["meters"]["M1"]["reading"] == "41"
    assert states["c1"]["meters"]["M1"]["reading"] == "77"


# --- D207: a nested row is named by its own scope ------------------------------------------------

def _run_schema() -> EntitySchema:
    """The same network, with a table let out per day so its key is two columns.

    A `runs` row is one leg on one day: two runs of one leg are two rows, and a result that answers
    several of them together is where a key part taken from the wrong scope names the wrong row.
    """
    names = {"routes": ["route_id", "carrier"], "runs": ["run_id", "day", "seats", "surface"]}
    return EntitySchema(
        tables=sorted(names),
        columns=[Column(table=table, name=name, **{"class": "hard"}, classified_by="rule")
                 for table, columns in sorted(names.items()) for name in columns],
        id_patterns={"routes.route_id": r"^R\d+$", "runs.run_id": r"^N\d+$"},
        composite_keys={"runs": ["run_id", "day"]},
    )


def _pins(workdir):
    return json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))


def test_a_nested_element_is_named_by_its_own_key_and_not_by_where_it_sits_in_the_list(workdir):
    """Two elements of one list carry two keys, so they are two rows and neither takes the other's."""
    result = [{"route_id": "R1", "carrier": "north", "runs": [
        {"run_id": "N1", "day": "mon", "seats": 4}, {"run_id": "N1", "day": "tue", "seats": 9}]}]
    trace = _trace("A", [_call("search_routes", {"carrier": "north"}, result=result)])
    state = ce.build_starting_state([trace], _run_schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False)
    assert state.db["runs"]["N1|mon"]["seats"] == 4
    assert state.db["runs"]["N1|tue"]["seats"] == 9
    assert _pins(workdir)["homed_by"] == {"key": 3, "position": 0}


def test_a_nested_element_takes_the_missing_key_part_from_the_dict_it_sits_in(workdir):
    """The day is stated once per group; a run inside a group is that group's day, not the call's."""
    result = [{"day": "mon", "runs": [{"run_id": "N1", "seats": 4}]},
              {"day": "tue", "runs": [{"run_id": "N1", "seats": 9}]}]
    trace = _trace("A", [_call("search_routes", {"day": "wed", "carrier": "north"}, result=result)])
    state = ce.build_starting_state([trace], _run_schema(), workdir, [Task(id="t1", run_ids=["A"])],
                                    _sigs(), synthetic=False)
    assert sorted(state.db["runs"]) == ["N1|mon", "N1|tue"]
    assert state.db["runs"]["N1|mon"]["seats"] == 4
    assert _pins(workdir)["nested_key_sources"] == {"own": 0, "parent": 2, "args": 0, "missing": 0}


def test_a_key_part_the_call_states_twice_names_no_row_and_the_sighting_keeps_a_partial_key(workdir):
    """One candidate in the arguments can name a row; two candidates name none of them."""
    one = _call("search_routes", {"day": "mon"}, result=[{"run_id": "N1", "seats": 4}], idx=0)
    both = _call("search_routes", {"days": [{"day": "mon"}, {"day": "tue"}]},
                 result=[{"run_id": "N2", "seats": 9}], idx=1)
    state = ce.build_starting_state([_trace("A", [one, both])], _run_schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False)
    assert "N1|mon" in state.db["runs"]
    assert "N2|" in state.db["runs"], "the part the call could not name is left empty, not guessed"
    assert _pins(workdir)["nested_key_sources"]["args"] == 1


def test_a_nested_element_a_write_named_is_marked_written_so_the_world_keeps_its_earlier_value(workdir):
    """A write names its run one level down and answers no row; the read after it is still post-write.

    The write's own key is composed from the element and the argument beside it, so the row it named
    is the row the later sighting is about, and the inverse replay undoes the write rather than
    keeping its value as the one the world started in.
    """
    sigs = _sigs() + [ToolSig(name="set_seats", kind="write", unclassified=False)]
    seen = _call("search_routes", {"carrier": "north"},
                 result=[{"run_id": "N1", "day": "mon", "seats": 4}], idx=0)
    wrote = _call("set_seats", {"runs": [{"run_id": "N1"}], "day": "mon", "seats": 9},
                  result={"changed": 1}, idx=1)
    again = _call("search_routes", {"carrier": "north"},
                  result=[{"run_id": "N1", "day": "mon", "seats": 9}], idx=2)
    state = ce.build_starting_state([_trace("A", [seen, wrote, again])], _run_schema(), workdir,
                                    [Task(id="t1", run_ids=["A"])], sigs, synthetic=False)
    assert state.db["runs"]["N1|mon"]["seats"] == 4


def test_one_key_naming_two_different_rows_in_one_result_is_a_finding(workdir):
    """Two sightings that disagree under one key mean the key does not name a row."""
    result = [{"run_id": "N1", "day": "mon", "seats": 4}, {"run_id": "N1", "day": "mon", "seats": 9}]
    trace = _trace("A", [_call("search_routes", {"carrier": "north"}, result=result)])
    ce.build_starting_state([trace], _run_schema(), workdir, [Task(id="t1", run_ids=["A"])],
                            _sigs(), synthetic=False)
    pins = _pins(workdir)
    assert pins["nested_key_collision"] == 1
    assert pins["nested_key_collisions"] == [{"table": "runs", "depth": 0, "key_class": "composite"}]


def test_a_list_element_that_carries_no_key_beside_ones_that_do_is_counted(workdir):
    """Nothing is homed by its position; an element with no key of its own is counted, not guessed."""
    result = [{"route_id": "R1", "carrier": "north", "runs": [
        {"run_id": "N1", "day": "mon", "seats": 4}, {"seats": 9}]}]
    trace = _trace("A", [_call("search_routes", {"carrier": "north"}, result=result)])
    ce.build_starting_state([trace], _run_schema(), workdir, [Task(id="t1", run_ids=["A"])],
                            _sigs(), synthetic=False)
    assert _pins(workdir)["list_elements_carrying_no_key"] == 1
