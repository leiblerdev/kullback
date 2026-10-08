"""Behaviour tests for kullback.trainer.session."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kullback.agent.extensions import ExtensionAPI  # noqa: E402
from kullback.agent.harness import AgentHarness  # noqa: E402
from kullback.ai.provider import TestModel  # noqa: E402
from kullback.trainer import session as session_mod  # noqa: E402
from kullback.trainer.session import train, trainer_extension  # noqa: E402
from kullback.trainer.tools import TrainerContext  # noqa: E402


def tool_call(call_id, name, arguments):
    return {"content": None, "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def test_session_calls_tools_files_a_finding_and_reports_turns(tmp_path):
    model = TestModel([
        tool_call("c1", "check_env", {}),
        tool_call("c2", "gpu_status", {}),
        tool_call("c3", "finding", {"kind": "done", "text": "watched nothing"}),
        "all quiet",
    ])
    got = train(tmp_path, model, train_run_id="r1", max_turns=20)
    assert got["stopped"] == "done" and got["turns"] >= 3
    assert [(row["kind"], row["text"]) for row in got["findings"]] == [("done", "watched nothing")]
    lines = (tmp_path / "training" / "findings.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["run_id"] == "r1"


def test_session_with_no_tool_calls_files_nothing(tmp_path):
    got = train(tmp_path, TestModel(["nothing to do"]), train_run_id="r1", max_turns=5)
    assert got["stopped"] == "done" and got["findings"] == []


def test_skill_folder_with_skill_md_is_loaded_at_start(tmp_path):
    skills = tmp_path / "training" / "skills"
    extra = skills / "extra"
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("Extra way of working.\n", encoding="utf-8")
    (skills / "notes.md").write_text("Filed notes.\n", encoding="utf-8")
    ctx = TrainerContext(workdir=tmp_path, train_run_id="r1")
    harness = AgentHarness(model=TestModel([]))
    trainer_extension(ctx)(ExtensionAPI(harness))
    catalog = harness.context.skills.catalog
    assert catalog["extra"] == "Extra way of working.\n"
    assert catalog["notes"] == "Filed notes.\n"
    assert "train" in harness.context.skills.loaded
    assert "extra" in harness.context.skills.loaded
    sections = {section.name: section.text for section in harness.sections}
    assert "Extra way of working." in sections["skill:extra"]
    assert "Filed notes." in sections["skill:notes"]


def test_bundled_trl_training_skill_is_loaded_at_start(tmp_path):
    ctx = TrainerContext(workdir=tmp_path, train_run_id="r1")
    harness = AgentHarness(model=TestModel([]))
    trainer_extension(ctx)(ExtensionAPI(harness))
    catalog = harness.context.skills.catalog
    expected = (Path(session_mod.__file__).parent / "skills" / "trl-training" / "SKILL.md")
    assert catalog["trl-training"] == expected.read_text(encoding="utf-8")
    assert "trl-training" in harness.context.skills.loaded
    sections = {section.name: section.text for section in harness.sections}
    assert expected.read_text(encoding="utf-8") in sections["skill:trl-training"]


def test_workdir_skill_replaces_the_bundled_text(tmp_path):
    extra = tmp_path / "training" / "skills" / "trl-training"
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("Workdir way of working.\n", encoding="utf-8")
    ctx = TrainerContext(workdir=tmp_path, train_run_id="r1")
    harness = AgentHarness(model=TestModel([]))
    trainer_extension(ctx)(ExtensionAPI(harness))
    assert harness.context.skills.catalog["trl-training"] == "Workdir way of working.\n"
    assert "trl-training" in harness.context.skills.loaded
    sections = {section.name: section.text for section in harness.sections}
    assert "Workdir way of working." in sections["skill:trl-training"]


def test_remote_json_missing_root_makes_train_raise(tmp_path):
    training = tmp_path / "training"
    training.mkdir(parents=True)
    (training / "remote.json").write_text(json.dumps({"host": "local"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        train(tmp_path, TestModel(["hi"]), train_run_id="r1", max_turns=5)
