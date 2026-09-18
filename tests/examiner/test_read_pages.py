"""Outline first, paged reads with an offset, search that returns locations (G21).

The world here is invented (a shelving room with numbered slots) and shares nothing with any
corpus: what is tested is that a long record reads end to end with no character lost or repeated,
that the outline stays small whatever the record's length, and that a locator from `search`
round-trips into `read`.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PTR
from examiner.worlds import drive
from kullback.agent import reading as reading_mod
from kullback.examiner import agent as examiner_agent
from kullback.examiner import tools as tools_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.runner.records import (
    Intent,
    Task,
    ToolCall,
    Trace,
    Turn,
    as_dict,
    write_json,
)

SLOT = "slot-7"
FIRST, SECOND = "task_shelf_1", "task_shelf_2"


def _turn(idx: int, role: str, content: str) -> Turn:
    return Turn(idx=idx, role=role, content=content, raw_ptr=PTR)


def _trace(trace_id: str, padding: str = "") -> Trace:
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 turns=[_turn(0, "user", f"put the crate in {SLOT}{padding}"),
                        _turn(1, "assistant", "on my way")],
                 tool_calls=[ToolCall(name="fetch_slot", args={"slot": SLOT}, raw_ptr=PTR,
                                      result={"slot": SLOT, "free": True})])


def _run_file(folder: Path, run_id: str, calls: int, pad: int) -> str:
    """One Run as the Runner writes it: the Run's own fields first, then one event a line."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.jsonl"
    rows = [{"run_id": run_id, "termination_reason": "success"},
            {"idx": 0, "type": "user_turn", "payload": {"content": f"put the crate in {SLOT}"}}]
    rows += [{"idx": n + 1, "type": "tool_call",
              "payload": {"id": f"c{n}", "name": "fetch_slot",
                          "args": {"slot": SLOT, "padding": "x" * pad}}}
             for n in range(calls)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def _world(tmp_path: Path, calls: int = 2, pad: int = 0) -> ExaminerPlan:
    """Two small Tasks, plus one long Run of `calls` padded calls on the first Task."""
    workdir = tmp_path / "shelves"
    workdir.mkdir()
    inputs = {
        "tasks": [Task(id=FIRST, run_ids=["trace_a"]), Task(id=SECOND, run_ids=["trace_b"])],
        "traces": [_trace("trace_a"), _trace("trace_b")],
        "intents": {
            FIRST: as_dict(Intent(task_id=FIRST, text=f"put the crate in {SLOT}", grounded=True)),
            SECOND: as_dict(Intent(task_id=SECOND, text="stack the crates", grounded=True)),
        },
        "replays": {
            FIRST: {"trace_a": {"trace_id": "trace_a", "run_id": "replay_a", "confirmed": True,
                                "path": _run_file(workdir / "runs" / FIRST, "replay_a", 1, 0)},
                    "long": {"trace_id": "trace_a", "run_id": "long", "confirmed": True,
                             "path": _run_file(workdir / "runs" / FIRST, "long", calls, pad)}},
            SECOND: {"trace_b": {"trace_id": "trace_b", "run_id": "replay_b", "confirmed": True,
                                 "path": _run_file(workdir / "runs" / SECOND, "replay_b", 1, 0)}},
        },
        "rerolls": {},
    }
    write_json(workdir / "task_status.json",
               {FIRST: {"reference_confirmed": True, "verifier_passed": False},
                SECOND: {"reference_confirmed": True, "verifier_passed": False}})
    return ExaminerPlan(workdir=workdir, inputs=inputs)


def _read(harness, **arguments):
    return drive(harness, "read", arguments)


def _read_pages(harness, **arguments):
    """Every page of one addressed read, concatenated: the whole answer, nothing cut."""
    seen, calls = "", 0
    offset: int | None = 0
    while offset is not None:
        result = _read(harness, offset=offset, **arguments)
        assert result.is_error is False, result.content
        seen += result.details["text"]
        offset = result.details["next_offset"]
        calls += 1
    return seen, calls


def _wide_world(tmp_path: Path) -> ExaminerPlan:
    """One Task with a forty-turn Trace of about eighty thousand characters."""
    workdir = tmp_path / "aisles"
    workdir.mkdir()
    turns = [Turn(idx=n, role="user" if n % 2 == 0 else "assistant", raw_ptr=PTR,
                  content=f"crate {n} in {SLOT} " + "y" * 1900) for n in range(40)]
    trace = Trace(trace_id="trace_wide", raw_hash="h", ingest_version="1", source="test",
                  raw_ptr=PTR, turns=turns)
    inputs = {"tasks": [Task(id=FIRST, run_ids=["trace_wide"])], "traces": [trace],
              "intents": {}, "replays": {}, "rerolls": {}}
    write_json(workdir / "task_status.json", {FIRST: {"reference_confirmed": False}})
    return ExaminerPlan(workdir=workdir, inputs=inputs)


def test_a_record_past_one_page_answers_its_outline_first(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    result = _read(harness, kind="run", id="long")
    assert result.is_error is False
    assert result.details["next_offset"] is None, "the map is answered whole"
    found = json.loads(result.details["text"])
    assert found["outline"] is True and found["run_id"] == "long"
    assert found["events_total"] == 51 and found["call_count"] == 50
    assert found["by_tool"] == {"fetch_slot": 50}
    assert found["total_chars"] > 200000


def test_a_long_record_reads_end_to_end_through_pages_with_no_character_lost_or_repeated(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    whole = json.loads(_read_pages(harness, kind="run", id="long", locator="header")[0])
    assert whole["run_id"] == "long" and "events" not in whole
    events = []
    for number in range(51):
        target = f"event:{number}"
        text = _read_pages(harness, kind="run", id="long", locator=target)[0]
        assert len(text) > 0
        events.append(json.loads(text))
    assert [event["idx"] for event in events] == list(range(51))
    assert events[0]["payload"]["content"].endswith(SLOT)
    assert all(event["payload"]["args"]["padding"] == "x" * 4000 for event in events[1:])


def test_pages_of_one_part_concatenate_to_the_part_and_offsets_walk_forward_only(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    first = _read(harness, kind="run", id="long", locator="event:1")
    assert first.is_error is False
    assert first.details["next_offset"] == reading_mod.PAGE_CHARS
    assert len(first.details["text"]) == reading_mod.PAGE_CHARS
    rest = _read(harness, kind="run", id="long", locator="event:1",
                 offset=first.details["next_offset"])
    assert rest.details["next_offset"] is None
    assert json.loads(first.details["text"] + rest.details["text"])["payload"]["id"] == "c0"
    past = _read(harness, kind="run", id="long", locator="event:1", offset=10 ** 9)
    assert past.details["text"] == "" and past.details["next_offset"] is None


def test_the_outline_of_a_record_never_exceeds_a_fixed_size_whatever_the_record_holds(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    text = _read(harness, kind="run", id="long").details["text"]
    assert len(text) < 8000 < 200000
    assert len(json.loads(text)["events"]) <= reading_mod.OUTLINE_PARTS
    # A long Trace is bounded the same way, and every part past the cap stays addressable.
    big = as_dict(_trace("trace_wide", padding="y" * 4000))
    big["turns"] = [as_dict(_turn(n, "user", f"crate {n} in {SLOT}{'z' * 4000}")) for n in range(60)]
    assert len(json.dumps(big)) > 200000
    found = reading_mod.outline(big)
    assert len(json.dumps(found)) < 8000
    assert found["turns_total"] == 60 and [row["locator"] for row in found["turns"]][-1] == "turn:49"
    assert reading_mod.part(big, "turn:59")["content"].startswith("crate 59")


def test_a_locator_from_search_round_trips_into_a_run_read(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    details = drive(harness, "search", {"text": SLOT}).details
    assert details["matches"] > 0 and len(details["locations"]) == details["shown"]
    run_hit = next(hit for hit in details["locations"] if hit["kind"] == "run")
    assert run_hit["locator"] is not None
    read_back = _read(harness, kind="run", id=run_hit["record"], locator=run_hit["locator"])
    assert read_back.is_error is False and SLOT in read_back.details["text"]


def test_a_locator_from_search_round_trips_into_a_trace_read(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    details = drive(harness, "search", {"text": SLOT}).details
    trace_hit = next(hit for hit in details["locations"] if hit["kind"] == "trace")
    read_back = _read(harness, kind="trace", id=trace_hit["record"], locator=trace_hit["locator"])
    assert read_back.is_error is False and SLOT in read_back.details["text"]
    intent_hit = next(hit for hit in details["locations"] if hit["kind"] == "intent")
    assert intent_hit["locator"] is None, "outside Runs and Traces a search hit has no part locator"
    assert _read(harness, kind="intent", id=intent_hit["record"]).is_error is False


def test_a_search_hit_on_a_run_header_reads_back_as_the_header(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    details = drive(harness, "search", {"text": "replay_a", "kinds": ["run"]}).details
    hit = next(hit for hit in details["locations"] if hit["where"] == "run fields")
    assert hit["locator"] == "header"
    header = json.loads(_read(harness, kind="run", id="replay_a", locator="header").details["text"])
    assert header["run_id"] == "replay_a" and "events" not in header


def test_a_forty_turn_record_reads_whole_through_turns_in_wide_pages(tmp_path):
    import math

    plan = _wide_world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    trace = as_dict(next(t for t in plan.inputs["traces"] if t.trace_id == "trace_wide"))
    full = json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    assert len(full) > tools_mod.READ_CHARS
    text, calls = _read_pages(harness, kind="trace", id="trace_wide", locator="turns")
    assert calls <= math.ceil(len(text) / tools_mod.READ_CHARS)
    found = json.loads(text)
    assert found["locator"] == "turns" and len(found["turns"]) == 40
    assert found["turns"][0]["text"].startswith("crate 0")
    assert found["turns"][-1]["locator"] == "turn:39"
    assert all(set(row) == {"locator", "speaker", "text"} for row in found["turns"])


def test_a_forty_turn_record_reads_whole_through_all_and_reconstructs_exactly(tmp_path):
    import math

    plan = _wide_world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    trace = as_dict(next(t for t in plan.inputs["traces"] if t.trace_id == "trace_wide"))
    full = json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    text, calls = _read_pages(harness, kind="trace", id="trace_wide", locator="all")
    assert calls <= math.ceil(len(full) / tools_mod.READ_CHARS)
    assert json.loads(text) == trace


def test_a_range_crossing_a_page_boundary_reconstructs_exactly(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    body = tools_mod._read_run(plan, "long")
    text, calls = _read_pages(harness, kind="run", id="long", locator="event:1-30")
    assert calls > 1, "thirty padded parts cross the wide page more than once"
    found = json.loads(text)
    assert found["locator"] == "event:1-30"
    assert found["parts"] == body["events"][1:31]


def test_a_reversed_or_out_of_range_range_is_refused_with_the_valid_range_named(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    backward = _read(harness, kind="trace", id="trace_a", locator="turn:1-0")
    assert backward.is_error and "turn:0-1" in backward.content
    past = _read(harness, kind="trace", id="trace_a", locator="call:0-9")
    assert past.is_error and "call:0-0" in past.content
    wrong = _read(harness, kind="run", id="replay_a", locator="turn:0-1")
    assert wrong.is_error and "event:" in wrong.content


def test_a_read_with_offset_and_no_locator_continues_the_whole_record(tmp_path):
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    first = _read(harness, kind="run", id="long")
    assert json.loads(first.details["text"])["outline"] is True
    at = _read(harness, kind="run", id="long", offset=30000)
    whole = _read(harness, kind="run", id="long", locator="all", offset=30000)
    assert at.details["text"] == whole.details["text"]
    assert at.details["next_offset"] == whole.details["next_offset"]
    assert at.details["text"] != first.details["text"], "offset pages the record, never the outline again"
    assert json.loads(first.details["text"])["events_total"] == 51


def test_turns_views_carry_speakers_and_no_tool_payloads(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    trace = json.loads(_read(harness, kind="trace", id="trace_a", locator="turns").details["text"])
    assert [(row["speaker"], row["locator"]) for row in trace["turns"]] == [
        ("user", "turn:0"), ("assistant", "turn:1")]
    assert SLOT in trace["turns"][0]["text"]
    run = json.loads(_read(harness, kind="run", id="replay_a", locator="turns").details["text"])
    assert [(row["speaker"], row["locator"]) for row in run["turns"]] == [("user", "event:0")]
    assert "args" not in json.dumps(run) and "result" not in json.dumps(run)


def test_a_locator_on_anything_but_a_run_or_a_trace_is_refused(tmp_path):
    harness = examiner_agent.examiner_harness(_world(tmp_path))
    result = _read(harness, kind="task_status", id=FIRST, locator="event:1")
    assert result.is_error and "Run or a Trace" in result.content
    result = _read(harness, kind="run", id="long", locator="event:500")
    assert result.is_error and "names no part" in result.content


def test_paged_reads_replace_the_sixty_thousand_character_cut_with_no_loss(tmp_path):
    """Sibling of the D175 cut test: the old `read` cut a long record at 60,000 characters with a
    note to read something narrower, and whatever was cut stayed unreachable. The new `read`
    answers the outline first and every addressed part in pages to a null next offset, so the
    same record reconstructs exactly."""
    plan = _world(tmp_path, calls=50, pad=4000)
    harness = examiner_agent.examiner_harness(plan)
    body = tools_mod._read_run(plan, "long")
    old = tools_mod._clamped_text(body)
    assert "characters cut" in old.splitlines()[-1], "the old path still cuts: this pins what changed"
    assert len(tools_mod._text(body)) - tools_mod.READ_CHARS > 100000, "most of the record was cut"
    found = json.loads(_read(harness, kind="run", id="long").details["text"])
    rebuilt = [json.loads(_read_pages(harness, kind="run", id="long", locator=f"event:{n}")[0])
               for n in range(found["events_total"])]
    assert rebuilt == body["events"]
