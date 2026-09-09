"""The rewrites of a Run's call path (D199), on an invented greenhouse: benches hold pots, a pot is
watered, and a bench is read. Nothing here knows a real corpus; an id-shaped key is the whole of the
vocabulary the independence rule has."""

from __future__ import annotations

from kullback.examiner import variants
from kullback.runner.records import Event, Run

WATER = "water_pot"
WRITE_TOOLS = {WATER, "move_pot"}


def _call(name: str, args: dict, result=None, cid: str = "", requestor: str = "assistant",
          error=None) -> dict:
    return {"name": name, "args": args, "result": result, "id": cid or name,
            "requestor": requestor, "error": error}


def _read_bench(bench: str = "b1", pots=("p1",)) -> dict:
    return _call("read_bench", {"bench_id": bench}, {"bench_id": bench, "pots": [{"pot_id": p} for p in pots]},
                 cid=f"read-{bench}")


def _water(pot: str = "p1", litres: float = 0.5) -> dict:
    return _call(WATER, {"pot_id": pot, "litres": litres}, {"pot_id": pot, "watered": True}, cid=f"water-{pot}")


def _kinds(rows) -> list[str]:
    return [row["kind"] for row in rows]


def test_a_row_id_is_read_from_an_id_shaped_key_at_any_depth():
    call = _read_bench("b1", pots=("p1", "p2"))
    assert variants.call_rows(call) == {"b1", "p1", "p2"}
    assert variants.row_ids({"note": "b9 is dry", "count": 3}) == frozenset()


def test_two_calls_that_touch_no_row_the_other_wrote_are_reordered():
    calls = [_read_bench("b1"), _water("p9")]
    assert variants.independent(calls[0], calls[1], WRITE_TOOLS) is True
    rows = variants.reorderings(calls, WRITE_TOOLS)
    assert _kinds(rows) == [variants.REORDER]
    assert [c["name"] for c in rows[0]["calls"]] == [WATER, "read_bench"]


def test_a_read_of_a_row_the_next_write_touches_is_not_reordered():
    calls = [_read_bench("b1", pots=("p1",)), _water("p1")]
    assert variants.independent(calls[0], calls[1], WRITE_TOOLS) is False
    assert variants.reorderings(calls, WRITE_TOOLS) == []


def test_two_writes_on_one_row_are_not_reordered_and_a_refused_write_wrote_nothing():
    calls = [_water("p1", 0.5), _water("p1", 0.25)]
    assert variants.independent(calls[0], calls[1], WRITE_TOOLS) is False
    refused = dict(_water("p2"), error={"payload": "the bench is closed"})
    assert variants.call_writes(refused, WRITE_TOOLS) == frozenset(), "a refused call wrote nothing (D67)"
    assert variants.independent(refused, calls[1], WRITE_TOOLS) is True


def test_a_call_the_other_speaker_made_is_not_swapped_with_one_of_ours():
    calls = [_read_bench("b1"), dict(_water("p9"), requestor="user")]
    assert variants.independent(calls[0], calls[1], WRITE_TOOLS) is True
    assert variants.reorderings(calls, WRITE_TOOLS) == []


def test_a_read_the_run_already_made_is_made_again_before_a_write():
    calls = [_read_bench("b1", pots=("p1",)), _water("p1")]
    rows = variants.inserted_reads(calls, WRITE_TOOLS)
    assert _kinds(rows) == [variants.INSERT_READ]
    assert [c["name"] for c in rows[0]["calls"]] == ["read_bench", "read_bench", WATER]
    assert rows[0]["calls"][1]["args"] == {"bench_id": "b1"}


def test_a_run_whose_only_read_follows_every_write_still_has_a_read_to_insert():
    calls = [_water("p1"), _read_bench("b1")]
    rows = variants.inserted_reads(calls, WRITE_TOOLS)
    assert [c["name"] for c in rows[0]["calls"]] == ["read_bench", WATER, "read_bench"]


def test_a_read_whose_answer_a_later_call_carries_in_its_arguments_is_not_dropped():
    calls = [_read_bench("b1", pots=("p1",)), _water("p1")]
    assert variants.dropped_reads(calls, WRITE_TOOLS) == []


def test_a_read_nothing_later_used_is_dropped():
    calls = [_read_bench("b7", pots=("p7",)), _water("p1")]
    rows = variants.dropped_reads(calls, WRITE_TOOLS)
    assert _kinds(rows) == [variants.DROP_READ]
    assert [c["name"] for c in rows[0]["calls"]] == [WATER]


def test_a_run_that_offers_no_rewrite_of_any_kind_yields_nothing():
    """One read the write depends on and nothing else: no independent pair, no read to drop, and one
    call is too few for a path. This is a path that is single by structure."""
    assert variants.variants([_water("p1")], WRITE_TOOLS) == []
    assert variants.variants([], WRITE_TOOLS) == []


def test_the_rewrites_come_a_kind_at_a_time_and_stop_at_the_limit():
    calls = [_read_bench("b1"), _read_bench("b2"), _read_bench("b3"), _water("p9")]
    rows = variants.variants(calls, WRITE_TOOLS, limit=3)
    assert len(rows) == 3 and len(set(_kinds(rows))) == 3, "one of each before a second of any"


def test_the_calls_and_the_conversation_are_read_off_a_runs_own_events():
    run = Run(run_id="r1", task_id="t1", termination_reason="success", events=[
        Event(idx=0, type="user_turn", payload={"content": "is bench b1 dry?"}),
        Event(idx=1, type="model_call", payload={"reply": {"content": "let me look"}}),
        Event(idx=2, type="tool_call", payload={"id": "c0", "name": "read_bench", "args": {"bench_id": "b1"}}),
        Event(idx=3, type="tool_result", payload={"id": "c0", "result": {"bench_id": "b1"}}),
        Event(idx=4, type="model_call", payload={"reply": {"content": "bench b1 is dry"}}),
    ])
    calls = variants.call_results(run)
    assert [c["name"] for c in calls] == ["read_bench"] and calls[0]["result"] == {"bench_id": "b1"}
    assert calls[0]["turn"] == 1, "the call belongs to the turn that made it"
    assert variants.transcript(run) == [{"role": "user", "content": "is bench b1 dry?"},
                                        {"role": "assistant", "content": "let me look"},
                                        {"role": "assistant", "content": "bench b1 is dry"}]


def test_a_reordering_moves_the_calls_turn_with_it():
    calls = [dict(_read_bench("b1"), turn=1), dict(_water("p9"), turn=3)]
    rows = variants.reorderings(calls, WRITE_TOOLS)
    assert [(c["name"], c["turn"]) for c in rows[0]["calls"]] == [(WATER, 1), ("read_bench", 3)]


def test_an_inserted_read_joins_the_turn_of_the_write_it_goes_before():
    calls = [dict(_read_bench("b1", pots=("p1",)), turn=1), dict(_water("p1"), turn=3)]
    rows = variants.inserted_reads(calls, WRITE_TOOLS)
    assert [(c["name"], c["turn"]) for c in rows[0]["calls"]] == [
        ("read_bench", 1), ("read_bench", 3), (WATER, 3)]


def test_one_rewrite_is_told_from_another_by_its_calls_and_arguments():
    first = [_call("read_bench", {"bench_id": "b1"}), _call(WATER, {"pot_id": "p1"})]
    same = [_call("read_bench", {"bench_id": "b1"}, cid="other"), _call(WATER, {"pot_id": "p1"})]
    assert variants.signature(first) == variants.signature(same), "a call id is not part of the path"
    assert variants.signature(first) != variants.signature(list(reversed(first)))
