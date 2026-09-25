"""The workdir checklist: six rows off files and names, never values or models.

Every test uses invented values in a tmp workdir. A key set to a marker string
must never appear in any row, only the variable name and where it came from.
"""

from __future__ import annotations

import json
import os

from kullback.tui.checklist import Row, next_step, where_it_stands

OPENAI_MODEL = "openai/gpt-6-luna"


def _clean(monkeypatch, tmp_path):
    """No key and no live switch from the shell, and no .env but the workdir's own."""
    for name in ("HARNESS_ALLOW_MODEL_REQUESTS", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_the_rows_come_in_the_order_a_newcomer_works_through_them(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    assert [row.name for row in where_it_stands(tmp_path, {}, OPENAI_MODEL)] == [
        "model", "live calls", "traces", "build", "runner", "publish"]


def test_an_empty_workdir_names_ingest_once_model_and_live_calls_are_done(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    env = {"OPENAI_API_KEY": "x", "HARNESS_ALLOW_MODEL_REQUESTS": "1"}
    rows = where_it_stands(tmp_path, env, OPENAI_MODEL)
    assert [row.name for row in rows[2:]] == ["traces", "build", "runner", "publish"]
    assert not any(row.done for row in rows[2:])
    assert "kullback ingest" in next_step(rows)


def test_a_set_key_names_the_variable_and_its_source_not_its_value(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "marker-secret-value-1")
    rows = where_it_stands(tmp_path, dict(os.environ), OPENAI_MODEL)
    assert rows[0] == Row("model", True, "OPENAI_API_KEY set (environment)", "/login")


def test_a_key_from_a_dotenv_file_is_named_as_from_dotenv(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=marker-secret-value-2\n", encoding="utf-8")
    rows = where_it_stands(tmp_path, {}, OPENAI_MODEL)
    assert rows[0].done and rows[0].detail == "OPENAI_API_KEY set (.env)"
    assert "marker-secret-value-2" not in "\n".join(row.detail for row in rows)


def test_live_calls_read_the_environment_or_a_dotenv_file(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    assert where_it_stands(tmp_path, {}, OPENAI_MODEL)[1].done is False
    assert where_it_stands(tmp_path, {"HARNESS_ALLOW_MODEL_REQUESTS": "1"}, OPENAI_MODEL)[1].done is True
    (tmp_path / ".env").write_text("HARNESS_ALLOW_MODEL_REQUESTS=1\n", encoding="utf-8")
    assert where_it_stands(tmp_path, {}, OPENAI_MODEL)[1].done is True


def test_traces_detail_gives_files_and_trace_counts(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    (tmp_path / "ingest_summary.json").write_text(json.dumps(
        [{"raw_hash": "a", "runs": 10}, {"raw_hash": "b", "runs": 5}]), encoding="utf-8")
    row = where_it_stands(tmp_path, {}, OPENAI_MODEL)[2]
    assert (row.done, row.detail) == (True, "2 files, 15 traces")


def test_build_detail_gives_the_last_round_trusted_and_fidelity(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"trusted": 2, "fidelity": 1, "tasks": 7}},
        {"round": 2, "counts": {"trusted": 5, "fidelity": 4, "tasks": 7}}]), encoding="utf-8")
    row = where_it_stands(tmp_path, {}, OPENAI_MODEL)[3]
    assert (row.done, row.detail) == (True, "round 2: trusted 5, fidelity 4/7")


def test_runner_is_done_when_the_version_file_exists(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    assert where_it_stands(tmp_path, {}, OPENAI_MODEL)[4].done is False
    (tmp_path / "runner_version.json").write_text(
        json.dumps({"runner_version": "deadbeefcafe0001"}), encoding="utf-8")
    row = where_it_stands(tmp_path, {}, OPENAI_MODEL)[4]
    assert row.done and "frozen" in row.detail


def test_publish_is_never_done_and_names_the_bar(tmp_path, monkeypatch):
    _clean(monkeypatch, tmp_path)
    row = where_it_stands(tmp_path, {}, OPENAI_MODEL)[5]
    assert (row.done, row.detail) == (False, "needs fidelity 0.90 over Tasks")
    assert "publish" in next_step([Row("model", True, "", ""), Row("live calls", True, "", ""),
                                   Row("traces", True, "", ""), Row("build", True, "", ""),
                                   Row("runner", True, "", ""), row])


def test_next_step_names_the_command_for_each_first_undone_row():
    names = ["model", "live calls", "traces", "build", "runner", "publish"]
    commands = ["/login", "HARNESS_ALLOW_MODEL_REQUESTS=1", "kullback ingest",
                "/build", "kullback freeze-runner", "kullback publish"]
    for at, command in enumerate(commands):
        rows = [Row(name, done=index < at, detail="", next_key="") for index, name in enumerate(names)]
        assert command in next_step(rows)
