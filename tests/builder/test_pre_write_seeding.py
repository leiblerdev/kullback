"""The pre-write inversion (D202): a column a Task first touches with a write is pinned from that write.

One invented domain throughout: a delivery network of meters, each carrying a seal that a write
flips, a mode a write cycles, and a reading nothing here changes. The write bodies are hand written,
standing in for the bodies a build would have compiled by the time this stage runs again.
"""

from __future__ import annotations

import json

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.runner.records import Column, EntitySchema, FieldStat, Task, ToolCall, ToolSig, Trace

FLIP_SEAL_BODY = """
meter = self.db.meters[meter_id]
meter.sealed = not bool(meter.sealed)
return {"meter_id": meter_id, "sealed": meter.sealed}
"""

CYCLE_MODE_BODY = """
order = ["idle", "metered", "audited"]
meter = self.db.meters[meter_id]
nxt = order[(order.index(meter.mode) + 1) % len(order)] if meter.mode in order else "idle"
meter.mode = nxt
return {"meter_id": meter_id, "mode": nxt}
"""

REFUSING_BODY = """
raise ValueError("the meter is not reachable")
"""


def _schema():
    names = ("meter_id", "sealed", "mode", "reading")
    columns = [Column(table="meters", name=name, **{"class": "hard"}, classified_by="rule")
               for name in names]
    return EntitySchema(tables=["meters"], columns=columns, id_patterns={"meters": r"^M\d+$"})


def _sigs():
    meter = [FieldStat(name="meter_id", types=["str"], optional=False)]
    return [ToolSig(name="get_meter", kind="read", unclassified=False, args_fields=meter),
            ToolSig(name="flip_seal", kind="write", unclassified=False, args_fields=meter),
            ToolSig(name="cycle_mode", kind="write", unclassified=False, args_fields=meter)]


def _call(name, args, result=None, idx=0):
    return ToolCall(id=f"c{idx}", name=name, args=args, result=result, raw_ptr=PTR)


def _trace(trace_id, calls):
    return Trace(trace_id=trace_id, raw_hash="rawhash", ingest_version="1", source="tau2",
                 tool_calls=calls, raw_ptr=PTR)


def _meter(meter_id, sealed, mode, reading=41):
    return {"meter_id": meter_id, "sealed": sealed, "mode": mode, "reading": reading}


def _pinned(workdir, task_id):
    overlay, values = ce.load_overlay(workdir, task_id)
    return {(row.table, row.id): values[row.version_hash] for row in overlay.rows}


def _pins(workdir):
    return json.loads((workdir / ce.PINS_FILE).read_text(encoding="utf-8"))


@pytest.fixture
def elsewhere():
    """Two traces of other meters, so the corpus knows the domain of every column."""
    return [_trace("B", [_call("get_meter", {"meter_id": "M2"}, result=_meter("M2", True, "idle"))]),
            _trace("C", [_call("get_meter", {"meter_id": "M3"},
                               result=_meter("M3", False, "metered", 12))]),
            _trace("D", [_call("get_meter", {"meter_id": "M4"},
                               result=_meter("M4", True, "audited", 7))])]


def _build(traces, workdir, bodies, tasks=None, schema=None):
    return ce.build_starting_state(traces, schema or _schema(), workdir,
                                   tasks if tasks is not None else [Task(id="t1", run_ids=["A"])],
                                   _sigs(), synthetic=False, bodies=bodies)


def test_a_flag_a_task_only_ever_wrote_is_pinned_to_the_value_the_write_flipped_it_from(
        workdir, elsewhere):
    """The write's own recorded result is the only evidence of what the flag held before it."""
    task = _trace("A", [_call("flip_seal", {"meter_id": "M2"},
                              result={"meter_id": "M2", "sealed": True})])
    state = _build([task, *elsewhere], workdir, {"flip_seal": FLIP_SEAL_BODY})

    assert _pinned(workdir, "t1")[("meters", "M2")]["sealed"] is False
    assert state.db["meters"]["M2"]["sealed"] is True, "the shared world keeps what the corpus agrees on"
    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 1
    inversion = pins["tasks"]["t1"]["inversions"][0]
    assert inversion["column"] == "sealed" and inversion["reason"] == "inverted_from_write"
    assert inversion["call_id"] == "c0" and inversion["tool"] == "flip_seal"
    assert inversion["body"], "the reason names the body the inversion trusted"


def test_an_enum_a_task_only_ever_wrote_is_pinned_to_the_one_value_that_reproduces_the_write(
        workdir, elsewhere):
    """Three values are on offer and one of them is the pre-state the recording implies."""
    task = _trace("A", [_call("cycle_mode", {"meter_id": "M2"},
                              result={"meter_id": "M2", "mode": "audited"})])
    _build([task, *elsewhere], workdir, {"cycle_mode": CYCLE_MODE_BODY})

    assert _pinned(workdir, "t1")[("meters", "M2")]["mode"] == "metered"
    assert _pins(workdir)["totals"]["columns_inverted"] == 1


def test_a_write_no_candidate_starting_value_reproduces_leaves_the_shared_value_and_is_counted(
        workdir, elsewhere):
    """A body that refuses whatever it is given cannot be inverted, and the pinner says so."""
    task = _trace("A", [_call("flip_seal", {"meter_id": "M2"},
                              result={"meter_id": "M2", "sealed": True})])
    _build([task, *elsewhere], workdir, {"flip_seal": REFUSING_BODY})

    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 0
    assert pins["totals"]["columns_no_inverse"] > 0
    assert pins["totals"]["inversion_candidates_tried"] > 0
    assert pins["inversion_by_tool"]["flip_seal"]["columns_no_inverse"] > 0


def test_a_column_the_task_read_before_the_write_is_pinned_by_the_read_and_never_inverted(
        workdir, elsewhere):
    """A read is better evidence of a starting value than an inversion, so it stands even when the
    write it precedes cannot be reproduced from it."""
    task = _trace("A", [
        _call("get_meter", {"meter_id": "M2"}, result=_meter("M2", False, "idle"), idx=0),
        _call("cycle_mode", {"meter_id": "M2"}, result={"meter_id": "M2", "mode": "audited"}, idx=1),
    ])
    _build([task, *elsewhere], workdir, {"cycle_mode": CYCLE_MODE_BODY})

    assert _pinned(workdir, "t1")[("meters", "M2")]["mode"] == "idle"
    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 0
    assert not pins["tasks"]["t1"]["inversions"]


def test_a_write_the_pinned_pre_state_already_reproduces_is_left_alone(workdir, elsewhere):
    """Nothing is inverted where the world in front of the write already answers the way it did."""
    task = _trace("A", [
        _call("get_meter", {"meter_id": "M2"}, result=_meter("M2", False, "idle"), idx=0),
        _call("flip_seal", {"meter_id": "M2"}, result={"meter_id": "M2", "sealed": True}, idx=1),
    ])
    _build([task, *elsewhere], workdir, {"flip_seal": FLIP_SEAL_BODY})

    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 0
    assert pins["totals"]["inversion_candidates_tried"] == 0


def test_a_column_two_writes_touch_before_any_read_is_inverted_for_the_first_write_only(
        workdir, elsewhere):
    """The second write reads the state the first one left, which is the replay's business."""
    task = _trace("A", [
        _call("flip_seal", {"meter_id": "M2"}, result={"meter_id": "M2", "sealed": True}, idx=0),
        _call("flip_seal", {"meter_id": "M2"}, result={"meter_id": "M2", "sealed": False}, idx=1),
    ])
    _build([task, *elsewhere], workdir, {"flip_seal": FLIP_SEAL_BODY})

    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 1
    assert [row["call_id"] for row in pins["tasks"]["t1"]["inversions"]] == ["c0"]


def test_the_body_scorer_is_given_the_inverted_pin(workdir, elsewhere):
    """The world `call_starting_states` hands the gates is the inverted one, or a body is scored wrong."""
    task = _trace("A", [_call("flip_seal", {"meter_id": "M2"},
                              result={"meter_id": "M2", "sealed": True})])
    state = _build([task, *elsewhere], workdir, {"flip_seal": FLIP_SEAL_BODY})

    states = ce.call_starting_states(state.db, state.overlays, ce.overlay_values(workdir),
                                     {"c0": "t1"})
    assert states["c0"]["meters"]["M2"]["sealed"] is False


def test_nothing_is_inverted_before_the_build_has_a_body_for_the_tool(workdir, elsewhere):
    """A first build has compiled no body, so there is nothing to run a candidate world against."""
    task = _trace("A", [_call("flip_seal", {"meter_id": "M2"},
                              result={"meter_id": "M2", "sealed": True})])
    _build([task, *elsewhere], workdir, {})

    pins = _pins(workdir)
    assert pins["totals"]["columns_inverted"] == 0
    assert pins["totals"]["inversion_candidates_tried"] == 0


def test_a_write_another_requestor_made_is_inverted_too(workdir, elsewhere):
    """The Router serves every requestor out of one Task world, so any write constrains that world."""
    flip = ToolCall(id="c0", name="flip_seal", args={"meter_id": "M2"}, requestor="user",
                    result={"meter_id": "M2", "sealed": True}, raw_ptr=PTR)
    _build([_trace("A", [flip]), *elsewhere], workdir, {"flip_seal": FLIP_SEAL_BODY})

    assert _pinned(workdir, "t1")[("meters", "M2")]["sealed"] is False
    assert _pins(workdir)["totals"]["columns_inverted"] == 1


def test_a_write_that_names_no_row_is_inverted_against_the_tasks_own_rows(workdir, elsewhere):
    """A device tool run on the one thing in front of its requestor takes no argument to home on."""
    schema = _schema()
    sigs = _sigs()
    sigs[1] = ToolSig(name="flip_seal", kind="write", unclassified=False)
    task = _trace("A", [
        _call("get_meter", {"meter_id": "M2"}, result={"meter_id": "M2", "mode": "idle"}, idx=0),
        ToolCall(id="c1", name="flip_seal", args={}, result="the seal is now off", raw_ptr=PTR),
    ])
    body = """
meter = self.db.meters["M2"]
meter.sealed = not bool(meter.sealed)
return "the seal is now on" if meter.sealed else "the seal is now off"
"""
    ce.build_starting_state([task, *elsewhere], schema, workdir, [Task(id="t1", run_ids=["A"])],
                            sigs, synthetic=False, bodies={"flip_seal": body})

    assert _pinned(workdir, "t1")[("meters", "M2")]["sealed"] is True
    assert _pins(workdir)["totals"]["columns_inverted"] == 1


def test_a_column_another_stage_filled_from_the_corpus_is_not_a_read_and_is_inverted(
        workdir, elsewhere):
    """The revealed row carries a value nobody read, which is exactly what a write can settle."""
    task = _trace("A", [
        _call("get_meter", {"meter_id": "M2"}, result={"meter_id": "M2", "mode": "idle"}, idx=0),
        _call("flip_seal", {"meter_id": "M2"}, result={"meter_id": "M2", "sealed": True}, idx=1),
    ])
    revealed = {("meters", "M2"): {"A": {"meter_id": "M2", "sealed": True}}}
    kept = ce.build_starting_state([task, *elsewhere], _schema(), workdir,
                                   [Task(id="t1", run_ids=["A"])], _sigs(), synthetic=False,
                                   revealed_rows=revealed, bodies={"flip_seal": FLIP_SEAL_BODY})
    assert _pinned(workdir, "t1")[("meters", "M2")]["sealed"] is True, "a read holds against a write"
    assert _pins(workdir)["totals"]["columns_inverted"] == 0
    assert kept.db["meters"]["M2"]["sealed"] is True

    other = workdir.parent / "guessed"
    other.mkdir()
    ce.build_starting_state([task, *elsewhere], _schema(), other, [Task(id="t1", run_ids=["A"])],
                            _sigs(), synthetic=False, revealed_rows=revealed,
                            bodies={"flip_seal": FLIP_SEAL_BODY},
                            guessed_columns=[("meters", "sealed")])
    assert _pinned(other, "t1")[("meters", "M2")]["sealed"] is False
    assert _pins(other)["totals"]["columns_inverted"] == 1


def test_the_candidates_of_a_column_are_the_corpus_domain_most_frequent_first(elsewhere):
    """A boolean gives two candidates and an enum a few; the order is what the corpus showed most."""
    schema, sigs = _schema(), _sigs()
    observations = ce._observations(list(elsewhere), schema, {"flip_seal", "cycle_mode"})
    domains = ce.column_domains(observations)
    assert domains[("meters", "sealed")] == [True, False]
    assert set(domains[("meters", "mode")]) == {"idle", "metered", "audited"}
    assert [sig.name for sig in sigs if sig.kind == "write"] == ["flip_seal", "cycle_mode"]
