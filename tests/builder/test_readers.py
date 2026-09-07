"""The readers stage: the row a requestor's own prose results reveal, under either gate.

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

BANNER_READER = """
def read(result):
    line = result.split("Banner:", 1)[1].strip()
    return {"banner": line}
"""

BANNER_DERIVE = """
def derive_banner(row):
    if row.get("vent") is None or row.get("warmth") is None:
        return None
    return row["vent"] + " | " + str(row["warmth"])
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


def proposal_json(*, gate=readers.GATE_DECLARED, vent=VENT_READER, climate=CLIMATE_READER,
                  open_shapes=("Vent opened.",), changes=("vent",), columns=None):
    declared = gate == readers.GATE_DECLARED
    body = {
        "table": "greenhouse",
        "columns": columns if columns is not None else [{"name": "vent", "origin": "stored"},
                                                        {"name": "warmth", "origin": "stored"}],
        "readers": [
            {"tool": "check_vent", "source": vent, "acknowledgements": [], "changes": []},
            {"tool": "check_climate", "source": climate, "acknowledgements": [], "changes": []},
            {"tool": "open_vent", "source": OPEN_READER,
             "acknowledgements": list(open_shapes) if declared else [],
             "changes": list(changes) if declared else []},
        ],
    }
    return json.dumps(body)


def gate_once(tmp_path, reply_json, *, gate=readers.GATE_DECLARED, traces=None):
    """One proposal parsed out of a reply and run through the gate over these recordings."""
    traces = two_traces() if traces is None else traces
    by_requestor = readers.prose_calls(traces)
    proposal = readers._parse_reply(ModelReply(content=reply_json), CARETAKER, gate)
    assert proposal is not None
    return proposal, readers.gate_proposal(proposal, by_requestor[CARETAKER], traces, tmp_path)


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

def test_a_proposal_whose_readers_answer_every_recorded_result_passes_the_declared_gate(tmp_path):
    _, ruling = gate_once(tmp_path, proposal_json())
    assert ruling.failures == []
    assert ruling.parsed[("check_vent", "Vent: CLOSED")] == {"vent": "closed"}


def test_a_reader_that_raises_is_named_by_its_tool_and_its_masked_shape(tmp_path):
    broken = """
def read(result):
    return {"vent": result.split("=", 1)[1]}
"""
    _, ruling = gate_once(tmp_path, proposal_json(vent=broken))
    named = [line for line in ruling.failures if line.startswith("check_vent shape ")]
    assert len(named) == 2, ruling.failures
    assert "IndexError" in named[0] and "'Vent: CLOSED'" in named[0]


def test_the_gate_names_one_line_per_shape_and_not_one_per_recorded_result(tmp_path):
    """Two recordings answer 'Vent: OPEN', so a reader that fails on it fails once, not twice."""
    traces = [trace("run_a", [call("check_vent", "Vent: OPEN")]),
              trace("run_b", [call("check_vent", "Vent: OPEN")])]
    body = json.dumps({"table": "greenhouse", "columns": [{"name": "vent", "origin": "stored"}],
                       "readers": [{"tool": "check_vent", "source": "def read(result):\n    return None",
                                    "acknowledgements": [], "changes": []}]})
    _, ruling = gate_once(tmp_path, body, traces=traces)
    assert len(ruling.failures) == 1 and "asserted nothing" in ruling.failures[0]


def test_a_result_that_asserts_nothing_has_to_be_declared_an_acknowledgement(tmp_path):
    _, ruling = gate_once(tmp_path, proposal_json(open_shapes=()))
    assert len(ruling.failures) == 1
    assert "open_vent shape 'Vent opened.'" in ruling.failures[0]
    assert "not declared an acknowledgement" in ruling.failures[0]


def test_the_replay_gate_asks_only_that_a_reader_runs_and_answers_a_dict_or_none(tmp_path):
    """The same proposal, minus every declaration: nothing is left to fail on."""
    _, ruling = gate_once(tmp_path, proposal_json(gate=readers.GATE_REPLAY),
                          gate=readers.GATE_REPLAY)
    assert ruling.failures == []


def test_the_replay_gate_still_refuses_a_reader_that_answers_something_that_is_not_a_row(tmp_path):
    _, ruling = gate_once(tmp_path, proposal_json(gate=readers.GATE_REPLAY,
                                                  vent="def read(result):\n    return result"),
                          gate=readers.GATE_REPLAY)
    assert len(ruling.failures) == 2
    assert "answered a str" in ruling.failures[0]


def test_a_key_no_column_was_proposed_for_is_a_failure_under_declared_and_a_column_under_replay(tmp_path):
    one_column = [{"name": "vent", "origin": "stored"}]
    declared_proposal, declared = gate_once(tmp_path, proposal_json(columns=one_column))
    assert any("warmth" in line for line in declared.failures)
    free_proposal, free = gate_once(tmp_path, proposal_json(gate=readers.GATE_REPLAY, columns=one_column),
                                    gate=readers.GATE_REPLAY)
    assert free.failures == []
    assert readers.absorb_columns(free_proposal, free.parsed).stored_names() == ["vent", "warmth"]
    assert declared_proposal.stored_names() == ["vent"]


def test_a_derived_column_is_checked_against_its_derivation_over_the_stored_columns(tmp_path):
    wrong = """
def derive_banner(row):
    if row.get("vent") is None or row.get("warmth") is None:
        return None
    return "always the same"
"""
    traces = [trace("run_a", [call("check_climate", "Vent: OPEN\nWarmth: 21 units"),
                              call("check_banner", "Banner: open | 21")])]
    body = {"table": "greenhouse",
            "columns": [{"name": "vent", "origin": "stored"}, {"name": "warmth", "origin": "stored"},
                        {"name": "banner", "origin": "derived", "derive": wrong}],
            "readers": [{"tool": "check_climate", "source": CLIMATE_READER},
                        {"tool": "check_banner", "source": BANNER_READER}]}
    _, ruling = gate_once(tmp_path, json.dumps(body), traces=traces)
    assert len(ruling.failures) == 1
    assert "derive_banner computed" in ruling.failures[0] and "check_banner shape" in ruling.failures[0]

    body["columns"][2]["derive"] = BANNER_DERIVE
    _, agreeing = gate_once(tmp_path, json.dumps(body), traces=traces)
    assert agreeing.failures == []


def test_a_tool_that_only_acknowledges_has_to_name_a_column_it_is_seen_to_change(tmp_path):
    _, unnamed = gate_once(tmp_path, proposal_json(changes=()))
    assert len(unnamed.failures) == 1 and "name in changes" in unnamed.failures[0]
    _, unseen = gate_once(tmp_path, proposal_json(changes=("warmth",)))
    assert len(unseen.failures) == 1 and "it changes warmth" in unseen.failures[0]


def test_a_column_that_reads_as_an_id_is_refused_because_code_owns_the_rows_key(tmp_path):
    columns = [{"name": "greenhouse_id", "origin": "stored"}, {"name": "vent", "origin": "stored"},
               {"name": "warmth", "origin": "stored"}]
    _, ruling = gate_once(tmp_path, proposal_json(columns=columns))
    assert any("greenhouse_id" in line and "code owns its key" in line for line in ruling.failures)


# --- the world the readers leave --------------------------------------------

def test_the_starting_row_is_what_was_read_before_the_declared_write_and_not_after_it(tmp_path):
    proposal, ruling = gate_once(tmp_path, proposal_json())
    traces = two_traces()
    rows = readers.starting_rows(traces, proposal, ruling.parsed)
    assert rows["run_a"] == {"vent": "closed", "warmth": 21}, "the vent was open only after open_vent"
    assert rows["run_b"] == {"vent": "open", "warmth": 12}


def test_without_a_declared_write_the_first_reading_of_a_column_stands(tmp_path):
    proposal, ruling = gate_once(tmp_path, proposal_json(gate=readers.GATE_REPLAY),
                                 gate=readers.GATE_REPLAY)
    rows = readers.starting_rows(two_traces(), proposal, ruling.parsed)
    assert rows["run_a"] == {"vent": "closed", "warmth": 21}, "the reader read nothing out of the write"


def test_the_columns_carry_the_requestor_that_revealed_them_and_the_export_flags_the_table(tmp_path):
    proposal, ruling = gate_once(tmp_path, proposal_json())
    schema = EntitySchema(tables=["plots"], columns=[])
    readers.apply_to_schema(schema, [proposal], {CARETAKER: readers.column_values(proposal, ruling.parsed)})
    assert schema.tables == ["greenhouse", "plots"]
    revealed = [c for c in schema.columns if c.table == "greenhouse"]
    assert {c.name for c in revealed} == {"vent", "warmth"}
    assert all(c.evidence["revealed_by"] == CARETAKER for c in revealed)
    assert readers.revealed_tables(schema) == {"greenhouse": CARETAKER}
    assert any("greenhouse" in flag and CARETAKER in flag for flag in readers.environment_flags(schema))


def test_a_body_is_told_to_read_the_one_row_without_naming_its_key(tmp_path):
    """The row's key is the requestor's name, and a literal id in a body is refused by D162."""
    proposal, _ = gate_once(tmp_path, proposal_json())
    note = readers.body_note([proposal])
    assert "next(iter(self.db.greenhouse.values()))" in note and "never by key" in note
    assert CARETAKER not in note.split("Its stored columns")[1]


def test_a_derived_column_reaches_the_body_writer_as_the_function_and_not_as_a_column(tmp_path):
    body = {"table": "greenhouse",
            "columns": [{"name": "vent", "origin": "stored"}, {"name": "warmth", "origin": "stored"},
                        {"name": "banner", "origin": "derived", "derive": BANNER_DERIVE}],
            "readers": [{"tool": "check_vent", "source": VENT_READER},
                        {"tool": "check_climate", "source": CLIMATE_READER},
                        {"tool": "open_vent", "source": OPEN_READER,
                         "acknowledgements": ["Vent opened."], "changes": ["vent"]}]}
    proposal, ruling = gate_once(tmp_path, json.dumps(body))
    assert ruling.failures == []
    schema = EntitySchema()
    readers.apply_to_schema(schema, [proposal], {})
    assert {c.name for c in schema.columns} == {"vent", "warmth"}, "nothing stores a derived column"
    assert "def derive_banner(row):" in readers.body_note([proposal])


def test_two_recordings_that_read_one_column_differently_start_in_different_worlds(tmp_path):
    proposal, ruling = gate_once(tmp_path, proposal_json())
    artifact = {"proposals": {CARETAKER: proposal.to_dict()},
                "rows": {CARETAKER: readers.starting_rows(two_traces(), proposal, ruling.parsed)}}
    worlds: dict = {}
    readers.merge_worlds(worlds, artifact)
    key = ("greenhouse", CARETAKER, "vent")
    assert worlds["run_a"][key] != worlds["run_b"][key]
    assert worlds["run_a"][("greenhouse", CARETAKER, "warmth")] != worlds["run_b"][
        ("greenhouse", CARETAKER, "warmth")]


# --- the attempts ------------------------------------------------------------

def test_a_failing_proposal_is_handed_back_the_shapes_that_failed_and_the_next_attempt_is_gated(tmp_path):
    model = TestModel([proposal_json(vent="def read(result):\n    return {'vent': result[99]}"),
                       proposal_json()])
    traces = two_traces()
    proposal, nodes, parsed = readers.propose(model, CARETAKER, readers.prose_calls(traces)[CARETAKER],
                                              traces, tmp_path, gate=readers.GATE_DECLARED)
    assert proposal.assisted is False and proposal.attempts == 2
    assert [n["passed"] for n in nodes] == [False, True]
    retry = model.calls[1]["messages"][-1]["content"]
    assert "check_vent shape 'Vent: CLOSED'" in retry and "IndexError" in retry
    assert parsed[("check_vent", "Vent: OPEN")] == {"vent": "open"}


def test_after_the_last_attempt_the_proposal_with_the_fewest_failing_shapes_is_kept_and_assisted(tmp_path):
    worse = proposal_json(vent="def read(result):\n    return {'vent': result[99]}",
                          climate="def read(result):\n    return {'warmth': result[99]}")
    better = proposal_json(vent="def read(result):\n    return {'vent': result[99]}")
    model = TestModel([worse, better, worse, worse])
    traces = two_traces()
    proposal, nodes, _ = readers.propose(model, CARETAKER, readers.prose_calls(traces)[CARETAKER],
                                         traces, tmp_path, gate=readers.GATE_DECLARED)
    assert len(nodes) == readers.MAX_ATTEMPTS and proposal.assisted is True
    assert [n["failing_shapes"] for n in nodes] == [4, 3, 4, 4]
    assert proposal.reader_for("check_climate").source.strip() == CLIMATE_READER.strip(), "the second attempt"
    assert sum("check_vent" in line for line in proposal.failures) == 2


def test_an_unknown_gate_is_refused_rather_than_run_under_a_default(tmp_path):
    with pytest.raises(ValueError, match="unknown readers gate"):
        readers.propose(TestModel([]), CARETAKER, {}, [], tmp_path, gate="whatever")


def test_the_stage_ruling_flags_an_assisted_proposal_and_never_fails_the_build():
    kept = readers.Proposal(requestor=CARETAKER, table="greenhouse", attempts=4, assisted=True,
                            failures=["check_vent shape 'Vent: N': the reader raised KeyError"])
    ruling = readers_gate([kept.to_dict()], 1)
    assert ruling.passed is False and ruling.metrics["assisted"] == 1
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
    assert json.loads((tmp_path / readers.READERS_FILE).read_text(encoding="utf-8")) == {
        "note": readers.NO_PROSE}


def test_the_stage_puts_the_revealed_row_in_the_world_and_splits_the_tasks_that_disagree(tmp_path):
    _write_traces(tmp_path, two_traces())
    plan = BuildPlan(workdir=tmp_path, model=TestModel([proposal_json()], loop=True),
                     readers_gate=readers.GATE_DECLARED)
    result = execute(plan, "db")
    db = result.artifacts["db"]
    assert list(db["greenhouse"]) == [CARETAKER]
    overlays = {o.task_id: o for o in result.artifacts["overlays"]}
    pinned = {t: [(r.table, r.id) for r in o.rows] for t, o in overlays.items()}
    assert all(("greenhouse", CARETAKER) in rows for rows in pinned.values())
    assert len(result.artifacts["tasks"]) == 2, "the two recordings started with the vent in two states"
    schema = result.artifacts["schema"]
    assert "greenhouse" in schema.tables
    assert readers.revealed_tables(schema) == {"greenhouse": CARETAKER}


def test_the_flag_chooses_the_gate_the_stage_holds_the_proposal_to(tmp_path):
    """The same reply, with no declaration in it, is refused under declared and kept under replay."""
    bare = json.dumps({"table": "greenhouse", "columns": [{"name": "vent"}],
                       "readers": [{"tool": "check_vent", "source": VENT_READER},
                                   {"tool": "check_climate", "source": CLIMATE_READER},
                                   {"tool": "open_vent", "source": OPEN_READER}]})
    _write_traces(tmp_path, two_traces())
    strict = execute(BuildPlan(workdir=tmp_path, model=TestModel([bare], loop=True),
                               readers_gate=readers.GATE_DECLARED), "readers")
    assert strict.artifacts["readers"]["proposals"][CARETAKER]["assisted"] is True

    free_dir = tmp_path / "free"
    _write_traces(free_dir, two_traces())
    free = execute(BuildPlan(workdir=free_dir, model=TestModel([bare], loop=True),
                             readers_gate=readers.GATE_REPLAY), "readers")
    assert free.artifacts["readers"]["proposals"][CARETAKER]["assisted"] is False


def test_a_gate_nobody_registered_is_refused_when_the_plan_is_made(tmp_path):
    with pytest.raises(ValueError, match="unknown readers gate"):
        BuildPlan(workdir=tmp_path, readers_gate="whatever")


def test_the_stage_needs_a_model_where_a_corpus_has_prose_results(tmp_path):
    _write_traces(tmp_path, two_traces())
    with pytest.raises(build_module.BuildError, match="no model"):
        execute(BuildPlan(workdir=tmp_path), "readers")
