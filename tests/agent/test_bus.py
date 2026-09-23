"""The workdir's bus: one ordered durable line per event, subscribers fed through it, a follower,
and two processes appending to one log without losing a record or repeating a number."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading

import pytest

from kullback.agent.bus import Bus, BusRecord
from kullback.agent.events import AgentEnd, CustomEvent, TurnStart
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import TestModel
from tests.agent.conftest import collect


def test_every_published_event_becomes_one_numbered_line_carrying_the_writers_name(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl", agent="builder")
    first = bus.publish_sync(TurnStart(turn=1))
    second = bus.publish_sync(AgentEnd())
    assert [r.seq for r in (first, second)] == [1, 2]
    lines = (tmp_path / "bus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    written = [json.loads(line) for line in lines]
    assert [row["seq"] for row in written] == [1, 2]
    assert {row["agent"] for row in written} == {"builder"}
    assert [row["event"]["type"] for row in written] == ["turn_start", "agent_end"]
    assert written[0]["recorded_at"] > 0


def test_replay_reads_the_log_back_as_typed_events_after_a_sequence_number(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl")
    bus.publish_sync(TurnStart(turn=1))
    bus.publish_sync(TurnStart(turn=2))
    reader = Bus(tmp_path / "bus.jsonl")
    records = reader.replay()
    assert [type(r).__name__ for r in records] == ["BusRecord", "BusRecord"]
    assert isinstance(records[0], BusRecord) and isinstance(records[0].event, TurnStart)
    assert [e.turn for e in reader.events()] == [1, 2]
    assert [e.turn for e in reader.events(since_seq=1)] == [2]
    # a custom event carries a name and a payload the core never reads
    bus.publish_custom("verdict", {"task": "t1", "trusted": True})
    event = Bus(tmp_path / "bus.jsonl").events(since_seq=2)[0]
    assert isinstance(event, CustomEvent)
    assert event.name == "verdict" and event.payload == {"task": "t1", "trusted": True}

def test_a_line_the_reader_cannot_parse_is_skipped_and_the_rest_still_reads(tmp_path):
    path = tmp_path / "bus.jsonl"
    bus = Bus(path)
    bus.publish_sync(TurnStart(turn=1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not a record\n")
    bus.publish_sync(TurnStart(turn=2))
    assert [e.turn for e in Bus(path).events()] == [1, 2]


def test_subscribers_are_called_in_registration_order_and_unsubscribe_stops_one(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl")
    seen = []
    bus.subscribe(lambda e: seen.append(("first", e.type)))
    unsubscribe = bus.subscribe(lambda e: seen.append(("second", e.type)))
    bus.publish_sync(TurnStart(turn=1))
    unsubscribe()
    bus.publish_sync(TurnStart(turn=2))
    assert seen == [("first", "turn_start"), ("second", "turn_start"), ("first", "turn_start")]
    # an async subscriber is awaited by publish and by the harness
    bus = Bus(tmp_path / "async.jsonl")
    heard = []

    async def watch(event):
        await asyncio.sleep(0)
        heard.append(event.type)

    bus.subscribe(watch)
    asyncio.run(bus.publish(TurnStart(turn=1)))
    assert heard == ["turn_start"]
    harness = AgentHarness(TestModel(["one"]))
    harness.subscribe(watch)
    collect(harness.prompt("a"))
    assert heard[1] == "agent_start" and heard[-1] == "agent_end"


def test_a_harness_feeds_subscribers_through_its_bus_log_or_in_process_without_one(tmp_path):
    bus = Bus(tmp_path / "bus.jsonl", agent="builder")
    harness = AgentHarness(TestModel(["done"]), bus=bus)
    seen = []
    unsubscribe = harness.subscribe(lambda e: seen.append(e.type))
    events = collect(harness.prompt("go"))
    assert [e.type for e in events] == seen
    written = Bus(tmp_path / "bus.jsonl").replay()
    assert [r.event.type for r in written] == seen
    assert [r.seq for r in written] == list(range(1, len(seen) + 1))
    assert {r.agent for r in written} == {"builder"}
    unsubscribe()
    collect(harness.prompt("again"))
    assert len(seen) < len(Bus(tmp_path / "bus.jsonl").replay())
    # without a bus the subscribers stay in this process and nothing is written
    local = tmp_path / "local"
    local.mkdir()
    harness = AgentHarness(TestModel(["one", "two"]))
    seen = []
    unsubscribe = harness.subscribe(seen.append)
    yielded = collect(harness.prompt("a"))
    assert [e.type for e in seen] == [e.type for e in yielded] and seen[-1].type == "agent_end"
    unsubscribe()
    collect(harness.prompt("b"))
    assert len(seen) == len(yielded) and not list(local.iterdir())

def test_a_follower_reads_what_is_there_and_then_what_appears(tmp_path):
    path = tmp_path / "bus.jsonl"
    writer = Bus(path)
    writer.publish_sync(TurnStart(turn=1))
    turns = []
    done = threading.Event()

    def follow():
        for record in Bus(path).tail(poll_seconds=0.01, stop=done.is_set):
            turns.append(record.event.turn)

    follower = threading.Thread(target=follow)
    follower.start()
    for turn in (2, 3):
        writer.publish_sync(TurnStart(turn=turn))
    while len(turns) < 3 and follower.is_alive():
        pass
    done.set()
    follower.join(timeout=5)
    assert turns == [1, 2, 3]


def test_two_processes_appending_to_one_log_number_every_record_once(tmp_path):
    path = tmp_path / "bus.jsonl"
    script = (
        "import sys;"
        "sys.path.insert(0, sys.argv[3]);"
        "from kullback.agent.bus import Bus;"
        "from kullback.agent.events import TurnStart;"
        "bus = Bus(sys.argv[1], agent=sys.argv[2]);"
        "[bus.publish_sync(TurnStart(turn=i)) for i in range(20)]"
    )
    root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    writers = [
        subprocess.Popen([sys.executable, "-c", script, str(path), name, root])
        for name in ("builder", "examiner")
    ]
    for writer in writers:
        assert writer.wait(timeout=60) == 0
    records = Bus(path).replay()
    assert [r.seq for r in records] == list(range(1, 41))
    assert sorted(r.agent for r in records) == ["builder"] * 20 + ["examiner"] * 20


def test_a_short_write_is_continued_until_the_whole_record_is_on_disk(tmp_path, monkeypatch):
    bus = Bus(tmp_path / "bus.jsonl", agent="builder")
    bus.publish_custom("first", {"n": 1})
    real_write = os.write
    calls = []

    def half_then_rest(handle, data):
        calls.append(len(data))
        return real_write(handle, bytes(data)[: max(len(data) // 2, 1)] if len(calls) == 1 else data)

    monkeypatch.setattr(os, "write", half_then_rest)
    bus.publish_custom("second", {"text": "x" * 200})
    monkeypatch.setattr(os, "write", real_write)
    assert len(calls) == 2 and calls[1] < calls[0]
    records = Bus(tmp_path / "bus.jsonl", agent="reader").replay()
    assert [(r.seq, r.event.name) for r in records] == [(1, "first"), (2, "second")]
    assert records[1].event.payload == {"text": "x" * 200}


def test_a_write_that_raises_leaves_the_log_at_its_old_length(tmp_path, monkeypatch):
    path = tmp_path / "bus.jsonl"
    bus = Bus(path, agent="builder")
    bus.publish_custom("first", {"n": 1})
    length = path.stat().st_size
    real_write = os.write
    calls = []

    def part_then_fail(handle, data):
        calls.append(len(data))
        if len(calls) == 1:
            return real_write(handle, bytes(data)[:10])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", part_then_fail)
    with pytest.raises(OSError, match="No space left"):
        bus.publish_custom("second", {"n": 2})
    monkeypatch.setattr(os, "write", real_write)
    assert path.stat().st_size == length
    bus.publish_custom("third", {"n": 3})
    assert [(r.seq, r.event.name) for r in Bus(path, agent="reader").replay()] == [(1, "first"), (2, "third")]
