"""The Builder's examine and grow tools: no Examiner session on nothing, grow reads prose (F23, F24)."""

from __future__ import annotations

import asyncio
import json

from examiner.worlds import make_world
from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from tests.builder.session_fixtures import reply
from tests.builder.test_domain_tools import _with_prose_result, _workdir
from tests.examiner.test_session import materialize


def _tools(root, model):
    return {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model)}


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_examine_with_no_finished_run_opens_no_session_and_prices_no_examiner(tmp_path):
    world = make_world(tmp_path, confirmed=False, rerolls=())
    materialize(world)
    model = TestModel([reply("done")], loop=True)
    result = _run(_tools(world.workdir, model)["examine"], {"task_ids": ["t1"]})
    assert not result.is_error, result.content
    assert model.calls == []
    budget_file = world.workdir / "budget.json"
    assert not budget_file.is_file() or "examiner" not in budget_file.read_text(encoding="utf-8")
    assert "nothing to examine yet: replay or run the Tasks first" in result.content


def test_examine_summary_names_the_tasks_the_session_left_out_and_why(tmp_path):
    world = make_world(tmp_path, tasks=2)
    materialize(world)
    status_file = world.workdir / "task_status.json"

    def examine_fn(workdir, task_ids):
        from kullback.examiner import session as S
        findings = S.examine(workdir, task_ids=task_ids, model=None)
        status = json.loads(status_file.read_text(encoding="utf-8"))
        status["t2"]["reference_confirmed"] = False
        _, left_out = S.select_for_session(task_ids, world.inputs["replays"], {}, status)
        return findings + [S.left_out_finding(left_out, examined=1)]

    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(
        workdir=world.workdir, examine_fn=examine_fn)}
    result = _run(tools["examine"], {"task_ids": ["t1", "t2"]})
    assert not result.is_error, result.content
    first = result.content.splitlines()[0]
    assert "1 Tasks left out: t2 (no confirmed Reference)" in first


def test_grow_hands_the_session_model_to_the_readers_step_and_prices_it(tmp_path):
    root = _with_prose_result(_workdir(tmp_path))
    model = TestModel([reply("no reader")], loop=True)
    grown = _run(_tools(root, model)["grow"], {"table": "widgets", "count": 3})
    assert "no model to propose" not in grown.content
    assert model.calls, "grow never asked the session model for its readers"
    assert "check_panel" in json.dumps(model.calls[0]["messages"])
    ledger = json.loads((root / "budget.json").read_text(encoding="utf-8"))
    assert "readers" in json.dumps(ledger)


def test_a_stopped_examine_says_so_in_its_first_line_and_its_details(tmp_path):
    world = make_world(tmp_path, tasks=2)
    materialize(world)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(
        workdir=world.workdir, should_stop=lambda: True)}
    result = _run(tools["examine"], {})
    assert not result.is_error, result.content
    first = result.content.splitlines()[0]
    assert first.startswith("examine stopped early: 0 of 2 Tasks derived, Examiner session not opened")
    assert result.details["stopped"] is True
    assert not (world.workdir / "verifiers").is_dir() or not list((world.workdir / "verifiers").glob("*.json"))
