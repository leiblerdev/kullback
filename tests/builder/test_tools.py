"""The Builder's tools: `status` over the records the build left, and the repair verbs that act (D135, D138)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from session_fixtures import fixture_path
from test_build import Bodies

from kullback.builder import agent as builder_agent
from kullback.builder import repair as repair_module
from kullback.builder import tools as builder_tools
from kullback.builder.build import BuildPlan

ASSISTED = "list_all_product_types"


@pytest.fixture(scope="module")
def built(tmp_path_factory, request) -> Path:
    """One code-driven build of the fixture, which leaves red lights of every kind to read."""
    workdir = tmp_path_factory.mktemp("tools")
    builder_agent.run_builder(workdir, model=Bodies(), files=[fixture_path(request)], max_attempts=0)
    return workdir


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def _tool(plan: BuildPlan, name: str):
    tools = {t.name: t for t in [*builder_tools.builder_tools(plan), *builder_tools.repair_verb_tools(plan)]}
    return tools[name]


# --- status -------------------------------------------------------------------

def test_status_names_every_failing_gate_with_its_tool_or_task_and_the_verb_that_owns_it(built):
    result = builder_tools.status_of(built)
    assert result.red_lights, "the fixture build leaves failing gates to report"
    by_stage = {light.stage: light for light in result.red_lights}
    assert by_stage["intent"].kind == "task" and by_stage["intent"].target.startswith("task_")
    assert by_stage["intent"].verb == "repair_refuse_task"
    assert by_stage["replay_reference"].kind == "task"
    assert by_stage["replay_reference"].verb == "repair_recompile"
    assisted = next(light for light in result.red_lights if light.target == ASSISTED and "assisted" in light.failure)
    assert assisted.stage == "compile_tools" and assisted.kind == "tool" and assisted.verb == "repair_recompile"
    assert "parses" in result.passing and "intent" in result.failing


def test_status_reads_the_records_and_asks_no_model(built):
    """Every red light comes off gates.json, replays.json and tool_builds.json; nothing else is read."""
    lights = builder_tools.red_lights(built)
    recorded = json.loads((built / "gates.json").read_text(encoding="utf-8"))
    failing = {row["stage"] for row in recorded if not row["pass"]}
    replays = json.loads((built / "replays.json").read_text(encoding="utf-8"))
    unconfirmed = {task for task, rows in replays.items() if not any(r["confirmed"] for r in rows.values())}
    assert failing <= {light.stage for light in lights}
    assert unconfirmed <= {light.target for light in lights if light.stage == "replay_reference"}
    plan = BuildPlan(workdir=built, iterate=True)
    assert plan.model is None, "status takes no model and the plan it reads from need not hold one"
    out = _run(_tool(plan, "status"), {})
    assert not out.is_error and out.content.startswith("status: ")
    assert f"-> {lights[0].verb}" in out.content


def test_status_over_a_workdir_with_nothing_built_says_so(tmp_path):
    result = builder_tools.status_of(tmp_path)
    assert result.red_lights == [] and result.passing == []
    assert "0 red lights" in result.summary
    assert "the gates are green" in builder_tools.render_status(result)


@pytest.mark.parametrize("failure,target,kind", [
    ("task task_7: no Trace of the Task was replayed", "task_7", "task"),
    ("get_order: a recorded success call replays differently", "get_order", "tool"),
    ('get_order({"id": 1}): hard columns differ', "get_order", "tool"),
    ("search_products has no body", "search_products", "tool"),
    ("no re-roll finished: 12 Runs and every one stopped on an error", "", "build"),
])
def test_a_failure_names_the_tool_or_the_task_the_way_the_gate_wrote_it(failure, target, kind):
    assert builder_tools._named_by(failure) == (target, kind)


def test_every_ruling_the_registry_names_has_a_repair_verb_or_goes_to_a_person():
    import kullback.gates as gates_pkg

    verbs = {"repair_recompile", "repair_grow", "repair_refuse_task", "repair_escalate"}
    for spec in gates_pkg.GATES:
        for ruling in spec.rulings:
            assert builder_tools.verb_for(ruling) in verbs, ruling
    assert builder_tools.verb_for("a_ruling_nobody_registered") == "repair_escalate"


# --- the repair verbs ----------------------------------------------------------

def test_the_four_repair_verbs_are_registered_and_the_skill_rewrite_is_not(tmp_path):
    names = [t.name for t in builder_tools.repair_verb_tools(BuildPlan(workdir=tmp_path))]
    assert names == ["repair_recompile", "repair_grow", "repair_refuse_task", "repair_escalate"]
    assert "repair_rewrite_skill" not in names, "a model rewriting its own prompt stays gated (GEPA caution)"


def test_repair_recompile_keeps_the_hint_as_a_lesson_and_compiles_the_tool_again(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    before = json.loads((built / "bodies.json").read_text(encoding="utf-8"))
    out = _run(_tool(plan, "repair_recompile"), {"name": ASSISTED, "hint": "import decimal before using it"})
    assert not out.is_error, out.content
    assert out.details["target"] == "compile_tools" and "compile_tools" in out.content
    assert repair_module.lesson_for_tool(built, ASSISTED).count("import decimal before using it") == 1
    request = (built / "repairs" / "repair_recompile.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(request[-1])["arguments"]["hint"] == "import decimal before using it"
    after = json.loads((built / "bodies.json").read_text(encoding="utf-8"))
    assert set(after) == set(before), "one tool repaired, every other body still there"


def test_repair_recompile_without_a_hint_records_no_lesson(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    before = repair_module.lesson_for_tool(built, "get_user_details")
    out = _run(_tool(plan, "repair_recompile"), {"name": "get_user_details"})
    assert not out.is_error, out.content
    assert repair_module.lesson_for_tool(built, "get_user_details") == before


def test_repair_grow_grows_the_table_and_records_the_request(built):
    plan = BuildPlan(workdir=built, iterate=True, model=Bodies(), max_attempts=0)
    out = _run(_tool(plan, "repair_grow"), {"table": "users", "count": 4})
    assert not out.is_error, out.content
    assert out.details["target"] == "starting_state"
    row = json.loads((built / "repairs" / "repair_grow.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["target"] == "users" and row["arguments"]["count"] == 4


def test_a_deciding_verb_records_its_decision_and_produces_no_artifact(tmp_path):
    plan = BuildPlan(workdir=tmp_path)
    out = _run(_tool(plan, "repair_refuse_task"), {"task_id": "task_1", "reason": "no frontier Run finishes it"})
    assert not out.is_error and out.details["verb"] == "repair_refuse_task"
    assert not out.details.get("produced"), "a decision changes no artifact, so no gate rules on it"
    escalated = _run(_tool(plan, "repair_escalate"), {"task_id": "task_1", "queue": "review"})
    assert not escalated.is_error and "repair_escalate task_1" in escalated.content


def test_a_repair_verb_with_bad_arguments_is_an_error_result_not_a_crash(tmp_path):
    plan = BuildPlan(workdir=tmp_path)
    out = _run(_tool(plan, "repair_grow"), {"table": "users", "count": 0})
    assert out.is_error and "count" in out.content
