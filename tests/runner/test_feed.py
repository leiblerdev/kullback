"""A build's live feed: what it writes as it happens, and what a watcher reads back."""

from __future__ import annotations

import json

from kullback.runner import feed


def test_events_are_read_once_in_order_from_where_the_reader_left_off(tmp_path):
    feed.start(tmp_path, model="openai/gpt-5.6-luna")
    feed.append(tmp_path, "model_call", stage="mine", usd=0.01)
    rows, offset = feed.read_since(tmp_path, 0)
    assert [row["kind"] for row in rows] == ["build_start", "model_call"]
    assert rows[0]["model"] == "openai/gpt-5.6-luna" and rows[0]["pid"]
    again, offset_again = feed.read_since(tmp_path, offset)
    assert again == [] and offset_again == offset
    feed.append(tmp_path, "model_call", stage="cluster", usd=0.02)
    rows, _ = feed.read_since(tmp_path, offset)
    assert [row["stage"] for row in rows] == ["cluster"]


def test_a_half_written_last_line_is_delivered_once_it_is_whole(tmp_path):
    """The writer is another process mid-append; a torn line is waited for, never half-read."""
    feed.append(tmp_path, "model_call", stage="mine")
    path = feed.path_for(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "model_call", "stage": "half')
    rows, offset = feed.read_since(tmp_path, 0)
    assert [row["stage"] for row in rows] == ["mine"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('way"}\n')
    rows, _ = feed.read_since(tmp_path, offset)
    assert [row["stage"] for row in rows] == ["halfway"]


def test_a_workdir_built_again_is_read_from_the_top(tmp_path):
    feed.append(tmp_path, "model_call", stage="one")
    feed.append(tmp_path, "model_call", stage="two")
    _, offset = feed.read_since(tmp_path, 0)
    feed.start(tmp_path)
    rows, _ = feed.read_since(tmp_path, offset)
    assert [row["kind"] for row in rows] == ["build_start"]


def test_writing_the_feed_never_raises_and_never_stops_a_build(tmp_path):
    """A feed is a courtesy to whoever is watching, so a directory that cannot be written to
    costs the build nothing."""
    blocked = tmp_path / "file-not-a-directory"
    blocked.write_text("", encoding="utf-8")
    feed.append(blocked / "work", "model_call", stage="mine")
    assert feed.read_since(blocked / "work", 0) == ([], 0)


def test_an_enormous_event_is_recorded_as_a_note_not_a_wall_of_text(tmp_path):
    feed.append(tmp_path, "model_call", stage="mine", body="x" * (feed.MAX_LINE * 2))
    rows, _ = feed.read_since(tmp_path, 0)
    assert rows[0]["note"] == "event too long to record in full"
    assert len(json.dumps(rows[0])) < feed.MAX_LINE


def test_a_model_call_reads_as_the_stage_the_model_the_tokens_and_the_cost(tmp_path):
    line = feed.describe({"kind": "model_call", "stage": "compile_tools", "model": "openai/gpt-5.6-luna",
                          "input": 12000, "output": 900, "cache_read": 4000, "usd": 0.0031,
                          "wall_ms": 2400})
    assert "compile_tools" in line and "openai/gpt-5.6-luna" in line
    assert "12,000 in / 900 out / 4,000 cached" in line and "$0.0031" in line and "2.4s" in line


def test_the_typed_round_and_beat_events_become_feed_rows_and_token_updates_do_not():
    """One row per token would be a feed nobody can read, so partial message updates are dropped."""
    from kullback.agent.events import BeatStart, MessageUpdate, RoundEnd

    assert feed.event_row(RoundEnd(round=2, counts={}, exit="done")) == {
        "kind": "round", "round": 2, "state": "end", "exit": "done"}
    assert feed.event_row(BeatStart(agent="examiner", round=2)) == {
        "kind": "beat", "agent": "examiner", "round": 2, "state": "start"}
    assert feed.event_row(MessageUpdate.model_construct()) is None


def _cached_reply(workdir, name, model="openai/gpt-5.6-luna", **usage):
    directory = workdir / feed.CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"model": model, "usage": {"input": 700, "output": 400, **usage}}))
    return path


def test_a_build_that_writes_no_feed_is_still_followed_through_its_reply_cache(tmp_path):
    """A build already running under code older than the feed never opens one, and watching it
    showed a still board; the cache it does write says what it has just called."""
    import os

    first = _cached_reply(tmp_path, "aaa")
    os.utime(first, (1000, 1000))
    rows, cursor = feed.derived_since(tmp_path, 0.0)
    assert [row["model"] for row in rows] == ["openai/gpt-5.6-luna"]
    assert rows[0]["input"] == 700 and rows[0]["output"] == 400 and cursor == 1000

    assert feed.derived_since(tmp_path, cursor) == ([], cursor)
    later = _cached_reply(tmp_path, "bbb", input=90)
    os.utime(later, (2000, 2000))
    rows, cursor = feed.derived_since(tmp_path, cursor)
    assert [row["input"] for row in rows] == [90] and cursor == 2000


def test_opening_a_watch_on_a_long_build_shows_the_last_calls_not_every_call(tmp_path):
    import os

    for index in range(6):
        os.utime(_cached_reply(tmp_path, f"c{index}"), (1000 + index, 1000 + index))
    rows, cursor = feed.derived_since(tmp_path, 0.0, limit=2)
    assert len(rows) == 2 and cursor == 1005
    assert feed.derived_since(tmp_path, cursor)[0] == []


def test_a_derived_call_says_the_tokens_it_knows_and_no_cost_it_does_not(tmp_path):
    line = feed.describe({"kind": "model_call", "model": "openai/gpt-5.6-luna",
                          "input": 700, "output": 400})
    assert "700 in / 400 out" in line and "$" not in line and "0.0s" not in line


def test_a_workdir_with_no_cache_at_all_is_silence_not_a_crash(tmp_path):
    assert feed.derived_since(tmp_path / "nowhere", 0.0) == ([], 0.0)
