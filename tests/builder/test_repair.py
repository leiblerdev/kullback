"""Phase 6 repair verbs, ratchet, and tool lessons."""

from __future__ import annotations

import asyncio
import json

from kullback.agent.messages import ToolCall
from kullback.builder import repair


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_verbs_record_requests(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    assert sorted(tools) == ["repair_escalate", "repair_grow", "repair_recompile",
                              "repair_refuse_task", "repair_rewrite_skill"]
    out = _run(tools["repair_recompile"], {"name": "get_order"})
    assert not out.is_error and "repair_recompile get_order" in out.content
    out = _run(tools["repair_grow"], {"table": "orders", "count": 50})
    assert not out.is_error
    out = _run(tools["repair_rewrite_skill"], {"name": "compile", "content": "# compile skill"})
    assert not out.is_error and (tmp_path / "skills" / "compile" / "SKILL.md").is_file()
    out = _run(tools["repair_refuse_task"], {"task_id": "t1", "reason": "no frontier run finishes"})
    assert not out.is_error
    out = _run(tools["repair_escalate"], {"task_id": "t2", "queue": "review"})
    assert not out.is_error
    assert (tmp_path / "repairs" / "repair_recompile.jsonl").is_file()


def test_verbs_reject_bad_args(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    out = _run(tools["repair_grow"], {"table": "orders", "count": 0})
    assert out.is_error


def test_repair_verbs_do_not_collide_with_builder_tools(tmp_path):
    from kullback.builder.build import BuildPlan
    from kullback.builder.tools import builder_tools
    builder_names = {t.name for t in builder_tools(BuildPlan(workdir=tmp_path))}
    repair_names = {t.name for t in repair.repair_tools(tmp_path)}
    assert not (builder_names & repair_names)


def test_ratchet_never_replaces_pass_with_fail():
    prior = {"bodies": {"a": "old-good", "b": "old-b"}}
    new = {"bodies": {"a": "new-bad", "b": "new-b"}}
    out = repair.ratchet_bodies(prior, new, {"a": False, "b": True})
    assert out["bodies"]["a"] == "old-good"
    assert out["bodies"]["b"] == "new-b"


def test_ratchet_keeps_tools_only_the_prior_has():
    prior = {"bodies": {"kept": "old-good", "a": "old-a"}}
    new = {"bodies": {"a": "new-a"}}
    out = repair.ratchet_bodies(prior, new, {"a": True})
    assert out["bodies"]["kept"] == "old-good"
    assert out["bodies"]["a"] == "new-a"


def test_ratchet_hook_restores_prior(tmp_path):
    (tmp_path / "bodies.json").write_text(json.dumps({"bodies": {"calc": "old-good"}}))
    hook = repair.ratchet_hook(tmp_path)
    call = ToolCall(id="c1", name="compile_tool", arguments={"name": "calc"})
    from kullback.agent.tools import ToolResult
    failed = ToolResult(content="compile_tool calc: failed",
                        details={"produced": ["bodies"], "payload": {},
                                 "stage_gates": [{"stage": "executes", "passed": False}]})
    out = hook(call, failed)
    assert out is not None and out.details["ratchet_restored"] == "calc"
    assert out.details["ratchet_restored_body"] == "old-good"


def test_tool_lesson_round_trip(tmp_path):
    assert repair.lesson_for_tool(tmp_path, "calc") == ""
    repair.record_tool_lesson(tmp_path, "calc", ["executes: import decimal missing"])
    text = repair.lesson_for_tool(tmp_path, "calc")
    assert "decimal" in text and "do not repeat" in text
