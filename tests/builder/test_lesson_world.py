"""The Triples the replay stage hands the lesson builder carry world (D211).

The domain is invented: a depot that files parcels into crates, each crate carrying a
status the guards read. Nothing here names a corpus, a customer tool or a customer column.
"""

from __future__ import annotations

from kullback.builder import compile_env as ce
from kullback.builder import lesson
from kullback.runner.records import Column, EntitySchema, RawPtr, ToolCall, ToolSig


def _schema():
    columns = [Column(table="crates", name=name, **{"class": "hard"}, classified_by="rule")
               for name in ("crate_id", "parcels", "status")]
    return EntitySchema(tables=["crates"], columns=columns, id_patterns={"crates": r"^k\d+$"})


def _call(call_id, args, result=None, trace="t1", position=0):
    return ToolCall(id=call_id, name="file_parcel", args=args, result=result, trace_id=trace,
                    raw_ptr=RawPtr(file_hash="testfile", sim_index=0, msg_index=position))


def _sig():
    return ToolSig(name="file_parcel", description="File one parcel into a crate.", kind="write",
                   unclassified=False)


ROW_K1 = {"crate_id": "k1", "parcels": 2, "status": "quarantined"}


def test_a_call_whose_trace_showed_the_named_row_two_calls_earlier_carries_its_columns():
    earlier = _call("c1", {"crate_id": "k1"}, dict(ROW_K1), position=0)
    between = _call("c2", {"crate_id": "k2"},
                    {"crate_id": "k2", "parcels": 5, "status": "cleared"}, position=1)
    failing = _call("c3", {"crate_id": "k1", "parcel": "p9"}, {"note": "filed"}, position=2)
    worlds = ce.witnessed_worlds([earlier, between, failing], _schema(), {})
    assert worlds["c3"] == ROW_K1


def test_the_latest_sighting_before_the_call_is_the_world_it_read():
    first = _call("c1", {"crate_id": "k1"}, dict(ROW_K1), position=0)
    second = _call("c2", {"crate_id": "k1"},
                   {"crate_id": "k1", "parcels": 2, "status": "cleared"}, position=1)
    failing = _call("c3", {"crate_id": "k1", "parcel": "p9"}, {"note": "filed"}, position=2)
    worlds = ce.witnessed_worlds([first, second, failing], _schema(), {})
    assert worlds["c3"]["status"] == "cleared"


def test_a_call_whose_row_was_never_sighted_carries_the_starting_state_row():
    failing = _call("c3", {"crate_id": "k1", "parcel": "p9"}, {"note": "filed"}, position=2)
    starting = {"c3": {"crates": {"k1": {"crate_id": "k1", "parcels": 5, "status": "cleared"}}}}
    worlds = ce.witnessed_worlds([failing], _schema(), starting)
    assert worlds["c3"] == {"crate_id": "k1", "parcels": 5, "status": "cleared"}


def test_a_call_naming_no_row_carries_an_empty_world():
    failing = _call("c3", {"note": "filed"}, {"note": "filed"}, position=0)
    starting = {"crates": {"k1": dict(ROW_K1)}}
    worlds = ce.witnessed_worlds([failing], _schema(), starting)
    assert worlds["c3"] == {}


def test_the_rendered_lesson_names_the_status_column_and_its_witnessed_value():
    earlier = _call("c1", {"crate_id": "k1"}, dict(ROW_K1), position=0)
    failing = _call("c3", {"crate_id": "k1", "parcel": "p9"}, {"note": "filed"}, position=1)
    worlds = ce.witnessed_worlds([earlier, failing], _schema(), {})
    triples = ce.call_triples(_sig(), [failing], [{"ok": False, "error": "NotPending"}],
                              [{"call_id": "c3", "replayed": False}], worlds=worlds)
    assert triples[0].world == ROW_K1
    invented = [r for r in lesson.relations_over(triples) if r.kind == "invented_refusal"]
    assert invented and all(r.on_all for r in invented)
    assert any("status" in r.sentence() and "quarantined" in r.sentence() for r in invented)
