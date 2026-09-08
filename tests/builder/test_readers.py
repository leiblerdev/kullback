"""The readers stage: the row a requestor's own prose results reveal.

The domain here is invented, the way every Builder test's is: a greenhouse whose caretaker runs a
few tools of its own against a vent and a heater, and answers in sentences rather than in rows.
Nothing in this file comes from a customer's corpus.
"""

from __future__ import annotations

import json

import pytest

from kullback.ai.provider import ModelReply, TestModel
from kullback.builder import build as build_module
from kullback.builder import readers
from kullback.builder.build import BuildPlan, execute
from kullback.gates.stages import readers_gate
from kullback.runner.records import EntitySchema, RawPtr, ToolCall, Trace, Turn, as_dict

PTR = RawPtr(file_hash="testfile", sim_index=0)
CARETAKER = "caretaker"

VENT_READER = """
def read(result):
    return {"vent": result.split(":", 1)[1].strip().lower()}
"""

CLIMATE_READER = """
def read(result):
    out = {}
    for line in result.splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "Vent":
            out["vent"] = value.strip().lower()
        if name.strip() == "Warmth":
            out["warmth"] = int(value.strip().split()[0])
    return out
"""

OPEN_READER = """
def read(result):
    return None
"""

OPEN_REPORTING_READER = """
def read(result):
    return {"vent": "open"}
"""


def call(name, result, requestor=CARETAKER, call_id=None, args=None):
    return ToolCall(id=call_id, name=name, args=dict(args or {}), result=result,
                    requestor=requestor, raw_ptr=PTR, has_result=True, resolved=True)


def trace(trace_id, calls, said="the vent is stuck and the plants are cold"):
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 turns=[Turn(idx=0, role="user", content=said, raw_ptr=PTR)], tool_calls=calls)


def two_traces():
    """Two recordings of the caretaker's tools: one starts closed and warm, one open and cold."""
    return [
        trace("run_a", [call("check_vent", "Vent: CLOSED"),
                        call("open_vent", "Vent opened."),
                        call("check_climate", "Vent: OPEN\nWarmth: 21 units")]),
        trace("run_b", [call("check_vent", "Vent: OPEN"),
                        call("check_climate", "Vent: OPEN\nWarmth: 12 units")]),
    ]


def credit_traces():
    """Twelve recordings: the vent moves when the caretaker opens it, and not otherwise.

    The climate check restates the whole greenhouse, so it sits inside intervals where the vent
    moved and intervals where it did not; the warmth moves in every interval, so nothing is credited
    with it. That is the shape the association rule has to tell apart.
    """
    out = []
    for index in range(4):
        out.append(trace(f"open_{index}", [call("check_climate", "Vent: CLOSED\nWarmth: 20 units"),
                                           call("open_vent", "Vent opened."),
                                           call("check_climate", "Vent: OPEN\nWarmth: 22 units")]))
        out.append(trace(f"quiet_{index}", [call("check_climate", "Vent: OPEN\nWarmth: 20 units"),
                                            call("check_climate", "Vent: OPEN\nWarmth: 21 units")]))
        out.append(trace(f"vent_{index}", [call("check_vent", "Vent: CLOSED"),
                                           call("open_vent", "Vent opened."),
                                           call("check_vent", "Vent: OPEN")]))
    return out


def proposal_json(*, vent=VENT_READER, climate=CLIMATE_READER, open_reader=OPEN_READER,
                  columns=("vent", "warmth")):
    return json.dumps({
        "table": "greenhouse",
        "columns": [{"name": name} for name in columns],
        "readers": [{"tool": "check_vent", "source": vent},
                    {"tool": "check_climate", "source": climate},
                    {"tool": "open_vent", "source": open_reader}],
    })


def gated(tmp_path, reply_json, traces=None, effects=None):
    """One proposal parsed out of a reply, run through the gate over these recordings.

    `effects` stands in for the credits where a test is about what closing a column does rather than
    about which tool the corpus credits with it.
    """
    traces = two_traces() if traces is None else traces
    by_requestor = readers.prose_calls(traces)
    proposal = readers._parse_reply(ModelReply(content=reply_json), CARETAKER)
    assert proposal is not None
    ruling = readers.gate_proposal(proposal, by_requestor[CARETAKER], tmp_path)
    readers.absorb_columns(proposal, ruling.parsed)
    proposal.effects = (readers.write_effects(traces, proposal, ruling.parsed)
                        if effects is None else dict(effects))
    return proposal, ruling


# --- what the stage is for ---------------------------------------------------

def test_a_corpus_whose_every_call_is_the_assistants_has_no_prose_results_to_read():
    traces = [trace("run_a", [call("look_up_plot", {"plot_id": "p1"}, requestor="assistant")])]
    assert readers.prose_calls(traces) == {}


def test_the_prose_results_of_another_requestor_are_grouped_by_the_tool_that_gave_them():
    found = readers.prose_calls(two_traces())
    assert sorted(found) == [CARETAKER]
    assert sorted(found[CARETAKER]) == ["check_climate", "check_vent", "open_vent"]
    assert len(found[CARETAKER]["check_vent"]) == 2


def test_results_that_differ_only_in_their_digits_are_one_shape_with_the_count_on_it():
    calls = [call("check_climate", "Warmth: 21 units"), call("check_climate", "Warmth: 12 units")]
    shapes = readers.shapes_of(calls)
    assert [s.shape for s in shapes] == ["Warmth: N units"]
    assert shapes[0].count == 2 and len(shapes[0].examples) == 2


# --- the gate ----------------------------------------------------------------

def test_a_proposal_whose_readers_run_over_every_recorded_result_passes(tmp_path):
    _, ruling = gated(tmp_path, proposal_json())
    assert ruling.failures == []
    assert ruling.parsed[("check_vent", "Vent: CLOSED")] == {"vent": "closed"}


def test_a_reader_that_raises_is_named_by_its_tool_and_its_masked_shape(tmp_path):
    broken = """
def read(result):
    return {"vent": result.split("=", 1)[1]}
"""
    _, ruling = gated(tmp_path, proposal_json(vent=broken))
    assert len(ruling.failures) == 2, ruling.failures
    assert all(line.startswith("check_vent shape ") for line in ruling.failures)
    assert "IndexError" in ruling.failures[0] and "'Vent: CLOSED'" in ruling.failures[0]


def test_the_gate_names_one_line_per_shape_and_not_one_per_recorded_result(tmp_path):
    """Two recordings answer 'Vent: OPEN', so a reader that fails on it fails once, not twice."""
    traces = [trace("run_a", [call("check_vent", "Vent: OPEN")]),
              trace("run_b", [call("check_vent", "Vent: OPEN")])]
    body = json.dumps({"table": "greenhouse", "columns": [{"name": "vent"}],
                       "readers": [{"tool": "check_vent",
                                    "source": "def read(result):\n    return result"}]})
    _, ruling = gated(tmp_path, body, traces=traces)
    assert len(ruling.failures) == 1 and "answered a str" in ruling.failures[0]


def test_a_reader_that_reads_nothing_out_of_a_shape_is_counted_and_not_refused(tmp_path):
    """The refusing version of this rule was measured on a customer corpus and bought nothing."""
    _, ruling = gated(tmp_path, proposal_json())
    assert ruling.failures == []
    assert ruling.silent == {"open_vent": 1}, "the write's own result asserts no column value"


def test_a_key_no_column_was_proposed_for_becomes_a_column_of_the_table(tmp_path):
    proposal, ruling = gated(tmp_path, proposal_json(columns=("vent",)))
    assert ruling.failures == []
    assert proposal.columns == ["vent", "warmth"]


def test_a_column_that_reads_as_an_id_is_refused_because_code_owns_the_rows_key(tmp_path):
    _, ruling = gated(tmp_path, proposal_json(columns=("greenhouse_id", "vent", "warmth")))
    assert any("greenhouse_id" in line and "code owns its key" in line for line in ruling.failures)


# --- which tool the corpus credits with a column -----------------------------

def test_a_tool_the_column_mostly_moves_around_is_credited_with_changing_it(tmp_path):
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=credit_traces())
    assert proposal.effects == {"open_vent": ["vent"]}


def test_a_check_that_restates_the_whole_state_is_credited_with_nothing_it_only_reports(tmp_path):
    """It sits inside changing and unchanging intervals alike, so its presence says nothing."""
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=credit_traces())
    assert "check_climate" not in proposal.effects and "check_vent" not in proposal.effects


def test_a_column_that_moves_in_every_interval_is_credited_to_nobody(tmp_path):
    """The warmth moves whatever was called, so no call explains it."""
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=credit_traces())
    assert all("warmth" not in columns for columns in proposal.effects.values())


def test_a_tool_the_corpus_never_runs_without_is_not_credited_for_want_of_a_contrast(tmp_path):
    """Four recordings, all of them opening the vent: nothing says the vent would have stayed shut."""
    traces = [trace(f"run_{i}", [call("check_vent", "Vent: CLOSED"),
                                 call("open_vent", "Vent opened."),
                                 call("check_vent", "Vent: OPEN")]) for i in range(4)]
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=traces)
    assert proposal.effects == {}


def test_a_change_seen_two_or_three_times_is_too_little_to_credit_a_tool_with(tmp_path):
    traces = ([trace("open_0", [call("check_vent", "Vent: CLOSED"),
                                call("open_vent", "Vent opened."),
                                call("check_vent", "Vent: OPEN")])]
              + [trace(f"quiet_{i}", [call("check_vent", "Vent: OPEN"),
                                      call("check_vent", "Vent: OPEN")]) for i in range(4)])
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=traces)
    assert proposal.effects == {}, "one changing interval is not a corpus saying anything"


def test_a_write_that_reports_its_own_new_value_is_read_as_evidence_of_the_change(tmp_path):
    """A write whose result states the new state is the commonest evidence a corpus has."""
    traces = ([trace(f"open_{i}", [call("check_vent", "Vent: CLOSED"),
                                   call("open_vent", "Vent opened.")]) for i in range(4)]
              + [trace(f"quiet_{i}", [call("check_vent", "Vent: OPEN"),
                                      call("check_vent", "Vent: OPEN")]) for i in range(4)])
    proposal, _ = gated(tmp_path, proposal_json(open_reader=OPEN_REPORTING_READER), traces=traces)
    assert proposal.effects == {"open_vent": ["vent"]}, "its own result is the reading after the call"


# --- what those credits settle ------------------------------------------------

def test_a_prose_tool_is_a_write_when_it_is_credited_with_a_column_and_a_read_when_it_is_not(tmp_path):
    proposal, _ruling = gated(tmp_path, proposal_json(), traces=credit_traces())
    assert readers.kinds_for(proposal) == {"check_climate": "read", "check_vent": "read",
                                           "open_vent": "write"}


def test_only_the_kind_of_a_tool_the_proposal_reads_is_overridden_and_the_reason_is_recorded(tmp_path):
    from kullback.runner.records import ToolSig

    proposal, _ruling = gated(tmp_path, proposal_json(), traces=credit_traces())
    mined = [ToolSig(name="check_climate", kind="write"), ToolSig(name="open_vent", kind="read"),
             ToolSig(name="look_up_plot", kind="read")]
    out = {sig.name: sig for sig in readers.apply_to_sigs(mined, [proposal])}
    assert out["check_climate"].kind == "read" and out["open_vent"].kind == "write"
    assert out["look_up_plot"] is mined[2], "a tool with no prose results is handed on untouched"
    assert "credited with changing" in (out["open_vent"].kind_reason or "")
    assert out["open_vent"].classified_by == "observed"


# --- what closing a column does to the starting row --------------------------

def test_the_starting_row_is_what_was_read_before_the_write_and_not_after_it(tmp_path):
    proposal, ruling = gated(tmp_path, proposal_json(), effects={"open_vent": ["vent"]})
    rows = readers.starting_rows(two_traces(), proposal, ruling.parsed)
    assert rows["run_a"] == {"vent": "closed", "warmth": 21}, "the vent was open only after the write"
    assert rows["run_b"] == {"vent": "open", "warmth": 12}


def test_a_column_read_only_after_the_write_of_another_column_still_stands(tmp_path):
    """Only the columns that tool is seen to change close at it; the rest keep the first reading."""
    proposal, ruling = gated(tmp_path, proposal_json(), effects={"open_vent": ["vent"]})
    only_after = trace("run_c", [call("open_vent", "Vent opened."),
                                 call("check_climate", "Vent: OPEN\nWarmth: 21 units")])
    assert readers.starting_row(only_after, proposal, ruling.parsed) == {"warmth": 21}


def test_with_no_write_seen_to_change_it_the_first_reading_of_a_column_stands(tmp_path):
    proposal, ruling = gated(tmp_path, proposal_json(), effects={})
    rows = readers.starting_rows(two_traces(), proposal, ruling.parsed)
    assert rows["run_a"] == {"vent": "closed", "warmth": 21}


# --- the columns no recording read before writing them -----------------------

def three_traces():
    """Two recordings that read the vent before the write, and one that only writes it."""
    return two_traces() + [trace("run_c", [call("open_vent", "Vent opened."),
                                           call("check_climate", "Vent: OPEN\nWarmth: 19 units")])]


def test_a_column_a_recording_never_read_before_writing_is_filled_from_the_corpus(tmp_path):
    traces = three_traces()
    proposal, ruling = gated(tmp_path, proposal_json(), traces=traces, effects={"open_vent": ["vent"]})
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    assert "vent" not in rows["run_c"], "that recording opened the vent before it ever read it"
    fills, assumptions, unset = readers.fills_for(rows, proposal)
    assert set(fills) == {"vent"} and fills["vent"] in {"closed", "open"}
    assert unset == []
    assert len(assumptions) == 1
    assert "column vent was not read before a write in 1 of 3 recordings" in assumptions[0]


def test_the_fill_is_the_value_the_most_recordings_showed_before_their_first_write(tmp_path):
    traces = three_traces() + [trace("run_d", [call("check_vent", "Vent: OPEN")])]
    proposal, ruling = gated(tmp_path, proposal_json(), traces=traces, effects={"open_vent": ["vent"]})
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    fills, _assumptions, _unset = readers.fills_for(rows, proposal)
    assert fills["vent"] == "open", "two recordings started open against one closed"


def test_a_column_the_corpus_never_shows_before_a_write_is_left_unset_and_named(tmp_path):
    """Nothing in the corpus says what the lamp was, so the world is not made to invent one."""
    traces = two_traces()
    proposal, ruling = gated(tmp_path, proposal_json(columns=("vent", "warmth", "lamp")),
                             traces=traces)
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    fills, assumptions, unset = readers.fills_for(rows, proposal)
    assert unset == ["lamp"] and fills == {} and assumptions == []


def test_the_fills_reach_the_world_and_the_split_is_left_to_what_was_read(tmp_path):
    traces = three_traces()
    proposal, ruling = gated(tmp_path, proposal_json(), traces=traces, effects={"open_vent": ["vent"]})
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    fills, assumptions, _unset = readers.fills_for(rows, proposal)
    artifact = {"proposals": {CARETAKER: proposal.to_dict()}, "rows": {CARETAKER: rows},
                "fills": {CARETAKER: fills}, "assumptions": assumptions}
    reached = readers.reader_rows(artifact)[("greenhouse", CARETAKER)]
    assert reached["run_c"]["vent"] == fills["vent"], "the world holds a value the corpus showed"
    worlds: dict = {}
    readers.merge_worlds(worlds, artifact)
    key = ("greenhouse", CARETAKER, "vent")
    assert key not in worlds["run_c"], "a filled column is the same everywhere and splits nothing"
    assert worlds["run_a"][key] != worlds["run_b"][key]
    assert readers.reader_assumptions(artifact) == assumptions


# --- which revealed columns may split a Task ---------------------------------

PROBE_READER = """
def read(result):
    out = {}
    for line in result.splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "Vent":
            out["vent_open"] = value.strip() == "OPEN"
        if name.strip() == "Warmth":
            out["warmth"] = float(value.strip().split()[0])
    return out
"""


def probe_traces():
    """Six recordings: the vent is a switch in one of two states, the warmth a probe's own reading."""
    return [trace(f"run_{index}",
                  [call("check_climate",
                        f"Vent: {'OPEN' if index % 2 else 'CLOSED'}\nWarmth: {19.1 + index} units")])
            for index in range(6)]


def probe_proposal(tmp_path):
    body = json.dumps({"table": "greenhouse",
                       "columns": [{"name": "vent_open"}, {"name": "warmth"}],
                       "readers": [{"tool": "check_climate", "source": PROBE_READER}]})
    traces = probe_traces()
    proposal, ruling = gated(tmp_path, body, traces=traces)
    schema = EntitySchema(tables=[], columns=[])
    readers.apply_to_schema(schema, [proposal], {CARETAKER: readers.column_values(proposal, ruling.parsed)})
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    artifact = {"proposals": {CARETAKER: proposal.to_dict()}, "rows": {CARETAKER: rows},
                "fills": {}, "assumptions": []}
    return traces, schema, artifact


def test_a_revealed_reading_is_exempt_and_a_revealed_switch_is_hard(tmp_path):
    _traces, schema, _artifact = probe_proposal(tmp_path)
    classes = {column.name: column.class_ for column in schema.columns}
    assert classes == {"vent_open": "hard", "warmth": "exempt"}
    assert all(column.classified_by == "rule" for column in schema.columns)


def test_a_revealed_reading_splits_no_task_and_a_revealed_switch_still_does(tmp_path):
    from kullback.builder.cluster import split_by_world

    traces, schema, artifact = probe_proposal(tmp_path)
    worlds = readers.merge_worlds({}, artifact, schema)
    keys = {key for world in worlds.values() for key in world}
    assert ("greenhouse", CARETAKER, "vent_open") in keys
    assert ("greenhouse", CARETAKER, "warmth") not in keys, "no two recordings read the same warmth"
    parts = split_by_world(traces, worlds)
    assert len(parts) == 2, "one Task per state of the switch, not one per reading"
    assert sorted(len(part) for part in parts) == [3, 3]


def test_with_no_classes_to_hand_every_revealed_column_still_counts(tmp_path):
    """A caller with no schema says every column, which is what merge_worlds did before the classes."""
    traces, _schema, artifact = probe_proposal(tmp_path)
    worlds = readers.merge_worlds({}, artifact)
    assert len(split_by_world_of(traces, worlds)) == 6


def split_by_world_of(traces, worlds):
    from kullback.builder.cluster import split_by_world

    return split_by_world(traces, worlds)


# --- what the proposal leaves for the rest of the build ----------------------

def test_the_columns_carry_the_requestor_that_revealed_them_and_the_export_flags_the_table(tmp_path):
    proposal, ruling = gated(tmp_path, proposal_json())
    schema = EntitySchema(tables=["plots"], columns=[])
    readers.apply_to_schema(schema, [proposal], {CARETAKER: readers.column_values(proposal, ruling.parsed)})
    assert schema.tables == ["greenhouse", "plots"]
    revealed = [c for c in schema.columns if c.table == "greenhouse"]
    assert {c.name for c in revealed} == {"vent", "warmth"}
    assert all(c.evidence["revealed_by"] == CARETAKER for c in revealed)
    assert readers.revealed_tables(schema) == {"greenhouse": CARETAKER}
    assert any("greenhouse" in flag and CARETAKER in flag for flag in readers.environment_flags(schema))


def test_a_body_is_told_to_read_the_one_row_without_naming_its_key_and_what_the_writes_change(tmp_path):
    """The row's key is the requestor's name, and a literal id in a body is refused by D162."""
    proposal, _ = gated(tmp_path, proposal_json(), effects={"open_vent": ["vent"]})
    note = readers.body_note([proposal])
    assert "next(iter(self.db.greenhouse.values()))" in note and "never by key" in note
    assert "The recording shows open_vent changing: vent." in note
    assert CARETAKER not in note.split("Its columns are")[1]


# --- the attempts ------------------------------------------------------------

def test_a_failing_proposal_is_handed_back_the_shapes_that_failed_and_the_next_attempt_is_gated(tmp_path):
    model = TestModel([proposal_json(vent="def read(result):\n    return {'vent': result[99]}"),
                       proposal_json()])
    traces = credit_traces()
    proposal, nodes, parsed = readers.propose(model, CARETAKER, readers.prose_calls(traces)[CARETAKER],
                                              traces, tmp_path)
    assert proposal.assisted is False and proposal.attempts == 2
    assert [n["passed"] for n in nodes] == [False, True]
    retry = model.calls[1]["messages"][-1]["content"]
    assert "check_vent shape 'Vent: CLOSED'" in retry and "IndexError" in retry
    assert parsed[("check_vent", "Vent: OPEN")] == {"vent": "open"}
    assert proposal.effects == {"open_vent": ["vent"]}, "the effects are mined off the kept proposal"


def test_after_the_last_attempt_the_proposal_with_the_fewest_failing_shapes_is_kept_and_assisted(tmp_path):
    worse = proposal_json(vent="def read(result):\n    return {'vent': result[99]}",
                          climate="def read(result):\n    return {'warmth': result[99]}")
    better = proposal_json(vent="def read(result):\n    return {'vent': result[99]}")
    model = TestModel([worse, better, worse, worse])
    traces = two_traces()
    proposal, nodes, _ = readers.propose(model, CARETAKER, readers.prose_calls(traces)[CARETAKER],
                                         traces, tmp_path)
    assert len(nodes) == readers.MAX_ATTEMPTS and proposal.assisted is True
    assert [n["failing_shapes"] for n in nodes] == [3, 2, 3, 3]
    assert proposal.reader_for("check_climate").source.strip() == CLIMATE_READER.strip()
    assert len(proposal.failures) == 2 and all("check_vent" in line for line in proposal.failures)


def test_the_stage_ruling_reports_the_silent_shapes_and_the_fills_and_fails_no_build():
    kept = readers.Proposal(requestor=CARETAKER, table="greenhouse", attempts=4, assisted=True,
                            silent={"open_vent": 2}, effects={"open_vent": ["vent"]},
                            failures=["check_vent shape 'Vent: N': the reader raised KeyError"])
    ruling = readers_gate([kept.to_dict()], 1, assumptions=["one column was filled"],
                          unset={CARETAKER: ["warmth"]},
                          kinds={CARETAKER: {"open_vent": "write", "check_vent": "read"}})
    assert ruling.passed is False and ruling.metrics["assisted"] == 1
    assert ruling.metrics["silent_shapes"] == 2 and ruling.metrics["silent_by_tool"] == {"open_vent": 2}
    assert ruling.metrics["changed_columns"] == 1 and ruling.metrics["filled_columns"] == 1
    assert ruling.metrics["unset_columns"] == {CARETAKER: ["warmth"]}
    assert ruling.metrics["write_tools"] == ["open_vent"] and ruling.metrics["read_tools"] == ["check_vent"]
    assert CARETAKER in ruling.failures[0]
    assert readers_gate([], 0).passed is True


# --- the stage inside the build ---------------------------------------------

def _write_traces(workdir, traces):
    for record in traces:
        path = workdir / "traces" / f"{record.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(as_dict(record)), encoding="utf-8")


def test_a_corpus_with_no_prose_results_records_the_note_and_asks_no_model(tmp_path):
    """The other corpora have no requestor of their own, so the stage is a no-op there."""
    _write_traces(tmp_path, [trace("run_a", [call("look_up_plot", {"plot_id": "p1"},
                                                  requestor="assistant", call_id="c1")])])
    plan = BuildPlan(workdir=tmp_path, model=TestModel([]))  # a call would raise
    result = execute(plan, "readers")
    assert result.artifacts["readers"]["note"] == readers.NO_PROSE
    assert result.artifacts["schema"] is result.artifacts["mined_schema"]
    assert result.artifacts["sigs"] is result.artifacts["mined_sigs"]
    assert json.loads((tmp_path / readers.READERS_FILE).read_text(encoding="utf-8")) == {
        "note": readers.NO_PROSE}


def test_the_stage_puts_the_revealed_row_in_the_world_and_splits_the_tasks_that_disagree(tmp_path):
    _write_traces(tmp_path, two_traces())
    plan = BuildPlan(workdir=tmp_path, model=TestModel([proposal_json()], loop=True))
    result = execute(plan, "db")
    db = result.artifacts["db"]
    assert list(db["greenhouse"]) == [CARETAKER]
    overlays = {o.task_id: o for o in result.artifacts["overlays"]}
    pinned = {t: [(r.table, r.id) for r in o.rows] for t, o in overlays.items()}
    assert all(("greenhouse", CARETAKER) in rows for rows in pinned.values())
    assert len(result.artifacts["tasks"]) == 2, "the two recordings started with the vent in two states"
    assert readers.revealed_tables(result.artifacts["schema"]) == {"greenhouse": CARETAKER}


def test_the_stage_records_every_column_it_filled_from_the_corpus_as_an_assumption(tmp_path):
    """A recording that never read a column starts on what the corpus showed, and says so."""
    traces = [trace("run_a", [call("check_vent", "Vent: OPEN"),
                              call("check_climate", "Vent: OPEN\nWarmth: 21 units")]),
              trace("run_c", [call("check_climate", "Vent: OPEN\nWarmth: 19 units")])]
    _write_traces(tmp_path, traces)
    plan = BuildPlan(workdir=tmp_path, model=TestModel([json.dumps({
        "table": "greenhouse",
        "columns": [{"name": "vent"}, {"name": "warmth"}, {"name": "lamp"}],
        "readers": [{"tool": "check_vent", "source": VENT_READER},
                    {"tool": "check_climate", "source": CLIMATE_READER}]})], loop=True))
    result = execute(plan, "db")
    artifact = result.artifacts["readers"]
    assert artifact["fills"][CARETAKER] == {}, "both recordings read every column they hold"
    assert artifact["unset"][CARETAKER] == ["lamp"], "nothing in the corpus ever read the lamp"
    assert "lamp" not in result.artifacts["db"]["greenhouse"][CARETAKER]
    assumptions = json.loads((tmp_path / "assumptions.json").read_text(encoding="utf-8"))
    assert result.artifacts["assumptions"] == assumptions


def test_a_fill_reaches_the_starting_state_and_its_assumption_file(tmp_path):
    traces = [trace("run_a", [call("check_vent", "Vent: OPEN"),
                              call("check_climate", "Vent: OPEN\nWarmth: 21 units")]),
              trace("run_c", [call("check_climate", "Warmth: 19 units")])]
    _write_traces(tmp_path, traces)
    plan = BuildPlan(workdir=tmp_path, model=TestModel([proposal_json()], loop=True))
    result = execute(plan, "db")
    assert result.artifacts["readers"]["fills"][CARETAKER] == {"vent": "open"}
    assumptions = json.loads((tmp_path / "assumptions.json").read_text(encoding="utf-8"))
    assert any("column vent was not read before a write in 1 of 2 recordings" in line
               for line in assumptions), assumptions


def test_the_stage_needs_a_model_where_a_corpus_has_prose_results(tmp_path):
    _write_traces(tmp_path, two_traces())
    with pytest.raises(build_module.BuildError, match="no model"):
        execute(BuildPlan(workdir=tmp_path), "readers")
