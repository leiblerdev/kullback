"""The screen's session list: heartbeats are written by the CLI and read by the TUI."""

from __future__ import annotations

import json
import os

from kullback.runner import heartbeat


def test_beat_writes_a_heartbeat_and_read_all_lists_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    first = tmp_path / "a"
    second = tmp_path / "b"
    heartbeat.beat(first, "m/one", "running")
    heartbeat.beat(second, "m/two", "running")
    records = heartbeat.read_all()
    assert [r["workdir"] for r in records] == [str(second), str(first)]
    assert records[0]["pid"] == os.getpid() and records[0]["status"] == "running"
    assert "started_at" in records[0] and "updated_at" in records[0]


def test_beat_keeps_started_at_and_updates_status(tmp_path, monkeypatch):
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    path = heartbeat.beat(tmp_path, "m/one", "running")
    started = json.loads(path.read_text(encoding="utf-8"))["started_at"]
    heartbeat.beat(tmp_path, "m/one", "done", exit="converged")
    record = heartbeat.read_all()[0]
    assert record["started_at"] == started and record["status"] == "done"
    assert record["exit"] == "converged"


def test_corrupt_heartbeats_are_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    heartbeat.beat(tmp_path, "m/one", "running")
    (heartbeat.sessions_dir() / "half.json").write_text("{nope", encoding="utf-8")
    assert len(heartbeat.read_all()) == 1


def test_alive_reports_this_process_and_rejects_bad_pids():
    assert heartbeat.alive(os.getpid()) is True
    assert heartbeat.alive(999999999) is False
    assert heartbeat.alive("nope") is False
