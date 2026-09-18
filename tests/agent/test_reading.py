"""The shared reading helpers: the map first, evidence in small pages (G21).

The records here are invented (a shelving room with numbered slots) and share nothing with any
corpus: what is tested is the shape of an outline, a page, a locate and a neighbourhood.
"""

from __future__ import annotations

import json

import pytest

from kullback.agent import reading
from kullback.agent.reading import PAGE_CHARS

SLOT = "slot-7"


def _turn(idx: int, role: str, content: str) -> dict:
    return {"idx": idx, "role": role, "content": content, "tool_call_ids": []}


def _call(name: str, args: dict, result: dict | None = None, error: dict | None = None) -> dict:
    return {"name": name, "args": args, "result": result, "error": error}


def _trace() -> dict:
    return {
        "trace_id": "trace-shelf-1",
        "turns": [_turn(0, "user", f"put the crate in {SLOT}"),
                  _turn(1, "assistant", "on my way"),
                  _turn(2, "user", "confirm when done")],
        "tool_calls": [_call("fetch_slot", {"slot": SLOT}, {"slot": SLOT, "free": True}),
                       _call("mark_slot", {"slot": SLOT}, None, {"code": "jammed"})],
    }


def _event(idx: int, kind: str, payload: dict) -> dict:
    return {"idx": idx, "type": kind, "payload": payload}


def _run() -> dict:
    return {
        "run_id": "run-shelf-1",
        "task_id": "task-shelf-1",
        "model": "probe:shelf",
        "termination_reason": "success",
        "events": [
            _event(0, "user_turn", {"content": f"put the crate in {SLOT}"}),
            _event(1, "tool_call", {"id": "c0", "name": "fetch_slot", "args": {"slot": SLOT}}),
            _event(2, "tool_result", {"id": "c0", "name": "fetch_slot",
                                      "result": {"slot": SLOT, "free": True}}),
            _event(3, "tool_call", {"id": "c1", "name": "mark_slot", "args": {"slot": SLOT}}),
            _event(4, "tool_result", {"id": "c1", "name": "mark_slot",
                                      "result": {"slot": SLOT}, "error": {"code": "jammed"}}),
        ],
    }


def _dump(body) -> str:
    return json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def test_kind_of_tells_a_run_from_a_trace_and_refuses_the_rest():
    assert reading.kind_of(_run()) == "run"
    assert reading.kind_of(_trace()) == "trace"
    with pytest.raises(ValueError):
        reading.kind_of({"run_id": "no-lists-here"})
    with pytest.raises(ValueError):
        reading.kind_of([])


def test_outline_of_a_trace_counts_turns_and_calls_by_name():
    found = reading.outline(_trace())
    assert found["kind"] == "trace" and found["outline"] is True
    assert found["trace_id"] == "trace-shelf-1"
    assert found["turn_count"] == 3 and found["turns_total"] == 3
    assert [row["locator"] for row in found["turns"]] == ["turn:0", "turn:1", "turn:2"]
    assert found["call_count"] == 2 and found["calls_total"] == 2
    assert found["by_tool"] == {"fetch_slot": 1, "mark_slot": 1}
    assert found["total_chars"] == len(_dump(_trace()))


def test_outline_of_a_trace_names_what_errored_and_how_to_continue():
    found = reading.outline(_trace())
    assert found["errored"] == ["call:1"]
    assert found["end"] is None
    assert all(set(row) == {"locator", "role", "size"} for row in found["turns"])
    assert all(set(row) == {"locator", "name", "size", "error"} for row in found["calls"])
    assert "turn:<n>" in found["note"] and "call:<n>" in found["note"]


def test_outline_of_a_run_echoes_what_it_outlines_and_counts_turns_and_calls():
    found = reading.outline(_run())
    assert found["kind"] == "run" and found["outline"] is True
    assert found["run_id"] == "run-shelf-1" and found["model"] == "probe:shelf"
    assert found["turn_count"] == 1 and found["call_count"] == 2
    assert found["by_tool"] == {"fetch_slot": 1, "mark_slot": 1}
    assert found["end"] == "success"


def test_outline_of_a_run_names_errored_events_and_how_to_continue():
    found = reading.outline(_run())
    assert found["errored"] == ["event:4"], "the result carrying the error is what errored"
    assert [row["locator"] for row in found["events"]] == [f"event:{n}" for n in range(5)]
    assert found["events_total"] == 5 and found["header"]["locator"] == "header"
    assert "event:<n>" in found["note"] and "header" in found["note"]


def test_outline_never_carries_a_payload_whatever_the_record_holds():
    text = _dump(reading.outline(_run())) + _dump(reading.outline(_trace()))
    assert "crate" not in text and SLOT not in text and "jammed" not in text


def test_outline_of_a_long_record_stays_small_while_every_part_stays_addressable():
    big = _run()
    big["events"] = [
        _event(n, "tool_call", {"id": f"c{n}", "name": "fetch_slot",
                                "args": {"slot": SLOT, "padding": "x" * 4000}})
        for n in range(60)
    ]
    assert len(_dump(big)) > 200000
    found = reading.outline(big)
    assert len(_dump(found)) < 8000, "the map stays a fixed size whatever the record's length"
    assert found["events_total"] == 60 and found["call_count"] == 60
    assert found["by_tool"] == {"fetch_slot": 60}
    for n in (0, 30, 59):
        assert reading.part(big, f"event:{n}")["payload"]["id"] == f"c{n}"


def test_pages_walk_a_text_end_to_end_with_no_character_lost_or_repeated():
    text = _dump(_run()) * 30
    assert len(text) > PAGE_CHARS
    seen = ""
    offset: int | None = 0
    while offset is not None:
        faced = reading.page(text, offset)
        assert faced["total"] == len(text)
        seen += faced["text"]
        offset = faced["next"]
    assert seen == text
    assert all(len(reading.page(text, at)["text"]) <= PAGE_CHARS for at in (0, 1, len(text) - 1))
    last = reading.page(text, len(text))
    assert last == {"text": "", "offset": len(text), "next": None, "total": len(text)}
    with pytest.raises(ValueError):
        reading.page(text, 0, 0)


def test_locate_answers_locators_with_ranges_into_the_part_and_a_snippet_each():
    found = reading.locate(_trace(), SLOT)
    assert found["total"] == 4 and found["truncated"] is False
    assert [hit["locator"] for hit in found["matches"]] == ["turn:0", "call:0", "call:0", "call:1"]
    for hit in found["matches"]:
        body = _dump(reading.part(_trace(), hit["locator"]))
        assert body[hit["start"]:hit["end"]].casefold() == SLOT.casefold()
        assert SLOT in hit["snippet"] or SLOT.casefold() in hit["snippet"].casefold()
    with pytest.raises(ValueError):
        reading.locate(_trace(), "   ")


def test_locate_stops_at_the_cap_and_says_so():
    big = {"trace_id": "trace-shelf-9",
           "turns": [_turn(n, "user", f"crate {n} in {SLOT}") for n in range(120)]}
    found = reading.locate(big, SLOT)
    assert found["total"] == 120 and len(found["matches"]) == reading.LOCATE_MAX
    assert found["truncated"] is True


def test_part_pulls_one_addressed_piece_and_refuses_what_is_not_there():
    assert reading.part(_trace(), "turn:0")["content"].endswith(SLOT)
    assert reading.part(_trace(), "call:1")["error"] == {"code": "jammed"}
    assert reading.part(_run(), "event:2")["payload"]["name"] == "fetch_slot"
    header = reading.part(_run(), "header")
    assert header["run_id"] == "run-shelf-1" and "events" not in header
    for bad in ("turn:9", "call:x", "event:1", "header", "nowhere"):
        with pytest.raises(ValueError):
            reading.part(_trace(), bad)
    with pytest.raises(ValueError):
        reading.part(_run(), "turn:0")


def test_around_shows_the_neighbours_as_skeletons_without_payloads():
    near = reading.around(_run(), "event:2", 1)
    assert near["locator"] == "event:2"
    assert [row["locator"] for row in near["parts"]] == ["event:1", "event:2", "event:3"]
    assert all("payload" not in row for row in near["parts"])
    turns = reading.around(_trace(), "turn:1", 5)
    assert [row["locator"] for row in turns["parts"]] == ["turn:0", "turn:1", "turn:2"]
    calls = reading.around(_trace(), "call:0", 1)
    assert [row["locator"] for row in calls["parts"]] == ["call:0", "call:1"], \
        "neighbours stay in their own list: turns never leak into calls"
    header = reading.around(_run(), "header")
    assert header["parts"] == [reading.around(_run(), "header")["parts"][0]]
    with pytest.raises(ValueError):
        reading.around(_run(), "event:9")
    with pytest.raises(ValueError):
        reading.around(_run(), "event:1", -1)


def test_errored_lists_are_capped_with_a_total_on_long_failing_records():
    calls = [_call("fetch_slot", {"slot": SLOT}, None, {"code": "jammed"}) for _ in range(60)]
    record = {"trace_id": "trace-shelf-9", "turns": [_turn(0, "user", "go")], "tool_calls": calls}
    found = reading.outline(record)
    assert len(found["errored"]) == reading.OUTLINE_PARTS and found["errored_total"] == 60
    assert len(_dump(found)) < 8000, "even an all-error record outlines small"
    run = _run()
    run["events"] = [_event(n, "error", {"note": f"bad {n}"}) for n in range(60)]
    outlined = reading.outline(run)
    assert len(outlined["errored"]) == reading.OUTLINE_PARTS and outlined["errored_total"] == 60
    assert outlined["errored"][0] == "event:0" and outlined["errored"][-1] == "event:49"


def test_locate_ranges_slice_the_rendered_text_even_when_folding_changes_lengths():
    record = {"trace_id": "trace-shelf-8",
              "turns": [_turn(0, "user", "die StraSSE ist breit")], "tool_calls": []}
    found = reading.locate(record, "straße")
    assert found["total"] == 1 and found["truncated"] is False
    hit = found["matches"][0]
    body = reading._dump(reading.part(record, hit["locator"]))
    assert body[hit["start"]:hit["end"]] == "StraSSE"
    assert "StraSSE" in hit["snippet"]


def test_select_answers_one_part_a_range_or_a_whole_view():
    assert reading.select(_trace(), "turn:1") == reading.part(_trace(), "turn:1")
    assert reading.select(_run(), "header") == reading.part(_run(), "header")
    ranged = reading.select(_trace(), "turn:0-2")
    assert ranged["locator"] == "turn:0-2"
    assert [turn["content"] for turn in ranged["parts"]] == [
        turn["content"] for turn in _trace()["turns"]]
    assert reading.select(_run(), "all") == _run()
    assert [call["name"] for call in reading.select(_trace(), "call:0-1")["parts"]] == [
        "fetch_slot", "mark_slot"]
    for bad in ("turn:2-0", "turn:0-9", "event:0-1", "everywhere"):
        with pytest.raises(ValueError):
            reading.select(_trace(), bad)


def test_a_reversed_or_out_of_range_range_is_refused_with_the_valid_range_named():
    with pytest.raises(ValueError, match="turn:0-2"):
        reading.select(_trace(), "turn:2-0")
    with pytest.raises(ValueError, match="call:0-1"):
        reading.select(_trace(), "call:0-9")


def test_turns_view_is_every_spoken_turn_in_order_without_tool_payloads():
    found = reading.select(_trace(), "turns")
    assert found["locator"] == "turns"
    assert [(row["speaker"], row["locator"]) for row in found["turns"]] == [
        ("user", "turn:0"), ("assistant", "turn:1"), ("user", "turn:2")]
    assert all(set(row) == {"locator", "speaker", "text"} for row in found["turns"])
    run = {"run_id": "run-shelf-2", "events": [
        _event(0, "user_turn", {"content": "hello"}),
        _event(1, "tool_call", {"id": "c0", "name": "fetch_slot", "args": {}}),
        _event(2, "model_call", {"reply": {"content": "on my way"}}),
        _event(3, "tool_result", {"id": "c0", "name": "fetch_slot", "result": {}})]}
    spoken = reading.select(run, "turns")["turns"]
    assert [(row["speaker"], row["text"]) for row in spoken] == [
        ("user", "hello"), ("assistant", "on my way")]
    assert [row["locator"] for row in spoken] == ["event:0", "event:2"]


def test_turns_view_reads_text_where_the_runner_writes_user_speech():
    run = {"run_id": "run-shelf-3", "events": [
        _event(0, "user_turn", {"text": "the Runner writes speech here"}),
        _event(1, "user_turn", {"content": "older rows say it here"})]}
    spoken = reading.select(run, "turns")["turns"]
    assert [(row["speaker"], row["text"]) for row in spoken] == [
        ("user", "the Runner writes speech here"), ("user", "older rows say it here")]


def test_a_long_record_reads_end_to_end_through_outlines_parts_and_pages():
    big = _run()
    big["events"] = [
        _event(n, "tool_call", {"id": f"c{n}", "name": "fetch_slot",
                                "args": {"slot": SLOT, "padding": "y" * 4000}})
        for n in range(50)
    ]
    assert len(_dump(big)) > 200000
    found = reading.outline(big)
    assert len(_dump(found)) < 8000
    rebuilt = []
    for n in range(found["events_total"]):
        target = _dump(reading.part(big, f"event:{n}"))
        faced = ""
        offset: int | None = 0
        while offset is not None:
            batch = reading.page(target, offset)
            faced += batch["text"]
            offset = batch["next"]
        rebuilt.append(json.loads(faced))
    assert rebuilt == big["events"], "every event back exactly once, no character lost or repeated"
