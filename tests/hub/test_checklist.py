"""The publish checklist names every missing gate and never carries a secret."""

from __future__ import annotations

import json

from kullback.hub import checklist as checklist_mod


def test_the_checklist_never_contains_the_token_value(nursery, monkeypatch):
    token = "s3cr3t-checklist-token-value"
    monkeypatch.setenv("HF_TOKEN", token)
    rows = checklist_mod.publish_checklist(nursery, "leibler/nursery")
    token_row = next(row for row in rows if row.name == "HF token")
    assert token_row.done and "HF_TOKEN" in token_row.detail
    rendered = "\n".join(f"[{'x' if row.done else ' '}] {row.name}: {row.detail}" for row in rows)
    assert token not in rendered


def test_the_checklist_reports_preview_only_below_the_bar(nursery, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "a cached login")
    rows = checklist_mod.publish_checklist(nursery, "leibler/nursery")
    by_name = {row.name: row for row in rows}
    assert by_name["fidelity over Tasks"].done is False
    assert "preview only" in by_name["fidelity over Tasks"].detail
    assert checklist_mod.readiness(rows) == "preview only"


def test_the_checklist_reports_release_at_the_bar_with_a_frozen_runner(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "env").mkdir()
    (workdir / "tasks").mkdir()
    (workdir / "verifiers").mkdir()
    (workdir / "environment.json").write_text(json.dumps({"env_id": "widgets-1"}), encoding="utf-8")
    (workdir / "schema.json").write_text(json.dumps({"tables": [], "columns": []}), encoding="utf-8")
    (workdir / "tool_sigs.json").write_text(json.dumps([]), encoding="utf-8")
    (workdir / "bodies.json").write_text(json.dumps({}), encoding="utf-8")
    (workdir / "canon-rules.json").write_text(json.dumps({}), encoding="utf-8")
    (workdir / "env" / "policy.md").write_text("Be helpful.", encoding="utf-8")
    (workdir / "tasks" / "t1.json").write_text(
        json.dumps({"id": "t1", "run_ids": [], "intent": "nothing to do"}), encoding="utf-8")
    (workdir / "replays.json").write_text(json.dumps({"t1": {"trace-t1": {"confirmed": True}}}),
                                          encoding="utf-8")
    (workdir / "runner_version.json").write_text(json.dumps({"runner_version": "rv-release-1"}),
                                                 encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "a token for the release check")
    rows = checklist_mod.publish_checklist(workdir, "leibler/widgets")
    by_name = {row.name: row for row in rows}
    assert by_name["fidelity over Tasks"].done is True
    assert "release" in by_name["fidelity over Tasks"].detail
    assert by_name["runner frozen"].done is True
    assert "rv-release-1" in by_name["runner frozen"].detail
    assert checklist_mod.readiness(rows) == "ready to release"


def test_the_checklist_refuses_a_repo_that_is_not_an_organisation_and_a_name(nursery, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    rows = checklist_mod.publish_checklist(nursery, "not-a-repo")
    by_name = {row.name: row for row in rows}
    assert by_name["repo name"].done is False
    assert "organisation/name" in by_name["repo name"].detail
    assert checklist_mod.readiness(rows) == "not ready"
