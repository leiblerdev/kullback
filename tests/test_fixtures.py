"""The checked-in fixtures keep the shape the module tests are written against."""

from __future__ import annotations

import json

GRADER_FIELDS = {"reward_info", "trial"}


def test_small_tau2_file_has_the_raw_shape(tau2_small: dict):
    assert set(tau2_small) == {"timestamp", "info", "tasks", "simulations"}
    assert len(tau2_small["simulations"]) == 3
    referenced = {s["task_id"] for s in tau2_small["simulations"]}
    assert {t["id"] for t in tau2_small["tasks"]} == referenced


def test_grader_fields_are_still_present_so_ingest_can_be_tested_stripping_them(tau2_small: dict):
    for sim in tau2_small["simulations"]:
        assert GRADER_FIELDS.issubset(sim)
        assert "action_checks" in sim["reward_info"]
    for task in tau2_small["tasks"]:
        assert "evaluation_criteria" in task


def test_simulations_carry_messages_with_tool_calls(tau2_small: dict):
    sim = tau2_small["simulations"][0]
    assert {"id", "task_id", "termination_reason", "seed", "messages"} <= set(sim)
    roles = {m["role"] for m in sim["messages"]}
    assert roles <= {"assistant", "user", "tool", "system"}
    assert any(m.get("tool_calls") for m in sim["messages"])
    tool_messages = [m for m in sim["messages"] if m["role"] == "tool"]
    assert tool_messages and {"id", "content", "requestor", "error"} <= set(tool_messages[0])


def test_tau2_retail_domain_files_are_there(tau2_retail_dir):
    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    assert {"users", "orders", "products"} <= set(db)
    tasks = json.loads((tau2_retail_dir / "tasks.json").read_text(encoding="utf-8"))
    assert isinstance(tasks, list) and tasks[0]["id"]
    assert (tau2_retail_dir / "policy.md").read_text(encoding="utf-8").strip()
