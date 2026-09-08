"""Readers derived from a tool's own recorded results, for a homed prose result nobody read (D203).

The domain here is invented, the way every Builder test's is: a courier depot whose tracking tool
answers in a sentence and whose lookup answers a row, so the corpus knows what the sentence is
talking about. Nothing in this file comes from a customer's corpus.
"""

from __future__ import annotations

import pytest

from kullback.ai.provider import ModelReply, TestModel
from kullback.builder import templates
from kullback.builder.readers import result_reader
from kullback.runner.records import Column, EntitySchema, RawPtr, ToolCall, Trace, Turn

PTR = RawPtr(file_hash="testfile", sim_index=0)


def schema() -> EntitySchema:
    """One table of parcels, keyed by its id, with a courier and a state the corpus states plainly."""
    return EntitySchema(
        tables=["parcels"],
        columns=[Column(table="parcels", name="parcel_id", class_="hard"),
                 Column(table="parcels", name="courier", class_="hard"),
                 Column(table="parcels", name="state", class_="hard")],
        id_patterns={"parcels": r"^P-\d+$"})


def call(name, result, args=None):
    return ToolCall(name=name, args=dict(args or {}), result=result, raw_ptr=PTR,
                    has_result=True, resolved=True)


def trace(trace_id, calls):
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 turns=[Turn(idx=0, role="user", content="where is my parcel", raw_ptr=PTR)],
                 tool_calls=calls)


def parcel(parcel_id, courier, state):
    return {"parcel_id": parcel_id, "courier": courier, "state": state}


def depot(rows, sentence="Parcel {parcel_id} is with {courier} and now {state}."):
    """One recording per parcel: the lookup states the row, the tracker answers a sentence about it."""
    out = []
    for index, row in enumerate(rows):
        out.append(trace(f"run_{index}", [
            call("fetch_parcel", row, {"parcel_id": row["parcel_id"]}),
            call("track_parcel", sentence.format(**row), {"parcel_id": row["parcel_id"]}),
        ]))
    return out


THREE = [parcel("P-1", "dana", "transit"),
         parcel("P-2", "eli", "delivered"),
         parcel("P-3", "dana", "held")]


# --- alignment -------------------------------------------------------------


def test_results_of_one_width_align_into_a_template_whose_varying_positions_are_slots():
    made, left = templates.templates_for([
        "Parcel P-1 is with dana.", "Parcel P-2 is with eli.", "Parcel P-3 is with dana."])
    assert left == []
    assert [t.segments for t in made] == [["Parcel ", " is with ", ""]]
    assert made[0].values == [["P-1", "P-2", "P-3"], ["dana.", "eli.", "dana."]]


def test_a_slot_is_cut_down_to_its_middle_only_where_the_whole_token_matches_no_column(tmp_path):
    # The sentence writes the state in brackets and with a full stop; the column holds the word.
    traces = depot(THREE, "Parcel {parcel_id} is ({state}).")
    made, found = templates.derive(traces, schema(), tmp_path)
    assert found["tools"] == {}
    assert made[0].templates[0]["segments"] == ["Parcel ", " is (", ")."]
    assert made[0].columns == ["parcel_id", "state"]


def test_a_shape_with_two_results_is_not_derived_and_comes_back_as_a_result_no_template_covers():
    made, left = templates.templates_for(["Parcel P-1 is with dana.", "Parcel P-2 is with eli."])
    assert made == []
    assert left == ["Parcel P-1 is with dana.", "Parcel P-2 is with eli."]


def test_results_that_share_too_little_literal_text_align_into_no_template():
    made, left = templates.templates_for(["a b c", "d e f", "g h i"])
    assert made == []
    assert len(left) == 3


# --- binding and the derived pair ------------------------------------------


def test_three_results_of_one_shape_bind_a_slot_to_a_column_and_the_reader_reads_a_fourth(tmp_path):
    traces = depot(THREE)
    made, found = templates.derive(traces, schema(), tmp_path)
    assert [d.tool for d in made] == ["track_parcel"]
    assert made[0].columns == ["courier", "parcel_id", "state"]
    assert made[0].round_trips[0] == 3
    assert found["tools"] == {}
    fourth = trace("run_new", [call("track_parcel", "Parcel P-9 is with fay and now lost.",
                                    {"parcel_id": "P-9"})])
    read = result_reader({"derived": [d.to_dict() for d in made]}, [fourth], tmp_path)
    assert read("track_parcel", "Parcel P-9 is with fay and now lost.") == {
        "parcel_id": "P-9", "courier": "fay", "state": "lost"}


def test_a_derived_reader_is_what_the_starting_state_pinner_reads_a_homed_result_with(tmp_path):
    from kullback.builder.compile_env import home_partial_result

    traces = depot(THREE)
    made, _report = templates.derive(traces, schema(), tmp_path)
    read = result_reader({"derived": [d.to_dict() for d in made]}, traces, tmp_path)
    homed = home_partial_result(schema(), "track_parcel", "Parcel P-1 is with dana and now transit.",
                                {"parcel_id": "P-1"}, read)
    assert homed is not None and homed.table == "parcels" and homed.row_id == "P-1"
    assert homed.row == {"parcel_id": "P-1", "courier": "dana", "state": "transit"}


def test_a_slot_matching_two_columns_is_left_unbound_and_the_tool_is_asked_about_once(tmp_path):
    # Both couriers carry a parcel whose state reads the same word, so one slot matches two columns
    # on every result and no column of the row is the one the sentence states.
    rows = [parcel("P-1", "dana", "dana"), parcel("P-2", "eli", "eli"),
            parcel("P-3", "fay", "fay")]
    traces = depot(rows, "Parcel {parcel_id} is now {state}.")
    made, found = templates.derive(traces, schema(), tmp_path)
    assert made == []
    assert found["tools"]["track_parcel"]["slots_unbound"] == 1
    model = TestModel(replies=[ModelReply(content="{}")])
    pairs, gaps = templates.close_gaps(model, traces, schema(), tmp_path)
    assert gaps["totals"]["forced_calls"] == 1
    assert pairs == []
    assert len(model.calls) == 1


def test_a_forced_proposal_the_round_trip_holds_is_kept_and_counted_as_forced(tmp_path):
    rows = [parcel("P-1", "dana", "dana"), parcel("P-2", "eli", "eli"), parcel("P-3", "fay", "fay")]
    traces = depot(rows, "Parcel {parcel_id} is now {state}.")
    reply = ('{"reader": "def read(result):\\n    parts = result.split(\\" \\")\\n'
             '    return {\\"parcel_id\\": parts[1], \\"state\\": parts[3].rstrip(\\".\\")}",\n'
             ' "render": "def render(row):\\n    return \\"Parcel %s is now %s.\\" % '
             '(row[\\"parcel_id\\"], row[\\"state\\"])"}')
    pairs, gaps = templates.close_gaps(TestModel(replies=[ModelReply(content=reply)]),
                                       traces, schema(), tmp_path)
    assert [(p.tool, p.forced) for p in pairs] == [("track_parcel", True)]
    assert gaps["totals"] == {"readers_derived": 0, "readers_forced": 1, "forced_calls": 1,
                              "columns_revealed": 0, "slots_unbound": 1,
                              "results_unread_by_tool": {}, "results_unread": 0,
                              "results_constant_by_tool": {}}


def test_the_round_trip_rejects_a_derived_pair_whose_render_does_not_reproduce_the_result(tmp_path):
    template = templates.Template(frame=["Parcel ", " is with dana."], heads=[""], tails=[""],
                                  raw=[["P-1"]], trimmed=[False], columns=["parcel_id"],
                                  results=["Parcel P-1 is with dana."])
    reader = templates.reader_source([template])
    wrong = "def render(row):\n    return 'Parcel ' + str(row['parcel_id']) + ' is with eli.'\n"
    homings = [templates.Homing("track_parcel", "Parcel P-1 is with dana.", "parcels", "P-1")]
    values = {("parcels", "P-1"): {"parcel_id": ["P-1"], "courier": ["dana"]}}
    matched, asked, failures, _revealed = templates.round_trip("track_parcel", reader, wrong,
                                                              homings, values, tmp_path)
    assert (matched, asked) == (0, 1)
    assert failures and "the round trip does not hold" in failures[0]


def test_a_result_no_template_reads_is_counted_unread_for_its_own_tool_and_named_by_its_shape(tmp_path):
    traces = depot(THREE)
    traces.append(trace("run_odd", [
        call("fetch_parcel", parcel("P-4", "gil", "returned"), {"parcel_id": "P-4"}),
        call("track_parcel", "Nobody has parcel P-4 today.", {"parcel_id": "P-4"})]))
    made, gaps = templates.close_gaps(None, traces, schema(), tmp_path)
    assert [d.tool for d in made] == ["track_parcel"]
    assert gaps["totals"]["results_unread_by_tool"] == {"track_parcel": 1}
    assert gaps["tools"]["track_parcel"]["unread_shapes"] == ["Nobody has parcel P-N today."]


def test_a_corpus_whose_prose_results_are_all_read_derives_nothing_and_counts_nothing(tmp_path):
    traces = depot(THREE)
    read = result_reader({"derived": []}, traces, tmp_path)
    assert read is None
    made, gaps = templates.close_gaps(
        None, traces, schema(), tmp_path,
        lambda tool, result: {"state": "known"})
    assert made == []
    assert gaps["totals"]["results_unread"] == 0


def test_a_tool_whose_result_is_the_bare_value_binds_it_to_the_one_column_that_holds_it(tmp_path):
    traces = depot(THREE, "{state}")
    made, found = templates.derive(traces, schema(), tmp_path)
    assert [d.tool for d in made] == ["track_parcel"]
    assert made[0].columns == ["state"]
    assert made[0].templates[0]["segments"] == ["", ""]
    read = result_reader({"derived": [d.to_dict() for d in made]}, traces, tmp_path)
    assert read("track_parcel", "held") == {"state": "held"}
    assert found["tools"] == {}


def test_a_tool_that_answers_one_string_to_every_call_asserts_nothing_and_is_not_counted_unread(tmp_path):
    traces = depot(THREE, "Noted.")
    made, gaps = templates.close_gaps(None, traces, schema(), tmp_path)
    assert made == []
    assert gaps["totals"]["results_unread_by_tool"] == {}
    assert gaps["totals"]["results_constant_by_tool"] == {"track_parcel": 3}


def test_a_prose_result_the_call_names_no_row_for_is_not_a_gap_this_module_can_close(tmp_path):
    traces = [trace("run_a", [call("depot_notice", "The depot is shut on Sunday.")]),
              trace("run_b", [call("depot_notice", "The depot is shut on Monday.")]),
              trace("run_c", [call("depot_notice", "The depot is shut on Tuesday.")])]
    made, gaps = templates.close_gaps(None, traces, schema(), tmp_path)
    assert made == []
    assert gaps["totals"]["results_unread_by_tool"] == {}


# --- what the counts reach ---------------------------------------------------


def test_the_readers_gate_reports_what_was_derived_forced_and_left_unread():
    from kullback.gates.stages import readers_gate

    ruling = readers_gate([], 0, derived=[templates.Derived(tool="track_parcel", table="parcels",
                                                            reader="", render="")],
                          totals={"readers_derived": 1, "readers_forced": 0, "forced_calls": 2,
                                  "slots_unbound": 3, "results_unread": 4,
                                  "results_unread_by_tool": {"track_parcel": 4}})
    assert ruling.passed
    assert ruling.metrics["readers_derived"] == 1
    assert ruling.metrics["slots_unbound"] == 3
    assert ruling.metrics["results_unread_by_tool"] == {"track_parcel": 4}
    assert ruling.metrics["derived_tools"] == ["track_parcel"]


def test_a_tool_whose_results_stay_unread_is_filed_as_a_finding_naming_its_shapes(tmp_path):
    from kullback.examiner.findings import unread_result_rows

    artifact = {"gaps": {"totals": {"results_unread_by_tool": {"track_parcel": 2}},
                         "tools": {"track_parcel": {"unread_shapes": ["Nobody has parcel P-N today."]}}}}
    status = {"task_1": {"reference_confirmed": False, "blocking_tools": ["track_parcel"]}}
    rows = unread_result_rows(artifact, status)
    assert [row["kind"] for row in rows] == ["environment"]
    assert rows[0]["tool"] == "track_parcel"
    assert rows[0]["task_ids"] == ["task_1"]
    assert "Nobody has parcel P-N today." in rows[0]["text"]
    assert rows[0]["key"] == "environment:track_parcel:"


@pytest.mark.parametrize("results", [[], ["one result only"]])
def test_a_tool_with_too_few_results_aligns_into_no_template(results):
    made, left = templates.templates_for(results)
    assert made == []
    assert left == sorted(set(results))


# --- a column the read reveals ---------------------------------------------

REVEAL = ('{"reader": "def read(result):\\n    return {\\"timeliness\\": result}",\n'
          ' "render": "def render(row):\\n    return str(row[\\"timeliness\\"])"}')


def tracked(words):
    """A recording per parcel whose tracker answers a bare word the parcel row does not hold."""
    out = []
    for index, (row, word) in enumerate(zip(THREE, words, strict=False)):
        out.append(trace(f"run_{index}", [
            call("fetch_parcel", row, {"parcel_id": row["parcel_id"]}),
            call("track_parcel", word, {"parcel_id": row["parcel_id"]})]))
    return out


def test_a_forced_reader_may_name_a_column_the_row_does_not_hold_and_the_schema_carries_it(tmp_path):
    made, gaps = templates.close_gaps(TestModel(replies=[ModelReply(content=REVEAL)]),
                                      tracked(["delayed", "early", "on time"]), schema(), tmp_path)
    assert [(pair.tool, pair.forced) for pair in made] == [("track_parcel", True)]
    assert sorted(made[0].revealed) == ["timeliness"]
    assert gaps["totals"]["columns_revealed"] == 1
    assert gaps["totals"]["results_unread"] == 0
    carried = templates.apply_revealed(schema(), made)
    column = next(c for c in carried.columns if c.name == "timeliness")
    assert column.table == "parcels"
    assert column.evidence["revealed_by"] == "track_parcel"


def test_a_column_one_row_answers_two_values_for_in_one_run_is_not_revealed(tmp_path):
    traces = tracked(["delayed", "early", "on time"])
    traces[0].tool_calls.append(call("track_parcel", "early", {"parcel_id": "P-1"}))
    made, gaps = templates.close_gaps(TestModel(replies=[ModelReply(content=REVEAL)]),
                                      traces, schema(), tmp_path)
    assert made == []
    assert gaps["totals"]["columns_revealed"] == 0
    assert gaps["totals"]["results_unread_by_tool"] == {"track_parcel": 3}


def test_a_revealed_column_holding_one_value_over_the_whole_corpus_is_refused(tmp_path):
    # The sentence names the parcel, which the row holds, and one word that never changes, which no
    # column holds: the round trip is happy either way, and the word asserts nothing about the row.
    reader = ("def read(result):\n"
              "    parts = result.split(' ')\n"
              "    return {'parcel_id': parts[1], 'timeliness': 'delayed'}\n")
    render = "def render(row):\n    return 'Parcel ' + str(row['parcel_id']) + ' is delayed.'\n"
    homings = [templates.Homing("track_parcel", f"Parcel {row['parcel_id']} is delayed.",
                                "parcels", row["parcel_id"], f"run_{index}")
               for index, row in enumerate(THREE)]
    values = {("parcels", row["parcel_id"]): {"parcel_id": [row["parcel_id"]],
                                              "courier": [row["courier"]]} for row in THREE}
    matched, asked, failures, revealed = templates.round_trip("track_parcel", reader, render,
                                                              homings, values, tmp_path)
    assert (matched, asked, revealed) == (0, 3, {})
    assert failures and "says nothing about the row" in failures[0]
