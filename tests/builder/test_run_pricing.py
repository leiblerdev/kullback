"""Every model call of a fresh Run is priced under the runner stage, the Builder's and the Examiner's (F29)."""

from __future__ import annotations

import json

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.runner import budget
from tests.builder.session_fixtures import priced, reply
from tests.builder.test_domain_tools import _run, _workdir

PRICED_ID = next(iter(budget.PRICES))


def _priced_candidate() -> TestModel:
    """A candidate that renames the widget then says so, each turn carrying a provider's usage."""
    return TestModel([priced(reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"}))),
                      priced(reply("Done, widget w1 is labelled striped."))], loop=True, name=PRICED_ID)


def _runner_stage(root) -> dict:
    return json.loads((root / "budget.json").read_text(encoding="utf-8"))["stages"]["runner"]


def test_a_fresh_run_lands_in_the_runner_stage_and_its_row_carries_the_spend(tmp_path):
    root = _workdir(tmp_path)
    candidate = _priced_candidate()
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=candidate)}
    result = _run(tools["run"], {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    (row,) = result.details["runs"]
    stage = _runner_stage(root)
    assert stage["calls"] == len(candidate.calls) == row["spend"]["calls"] > 0
    assert stage["usd"] > 0
    assert row["spend"]["usd"] == round(stage["usd"], 6)
    assert "spend: 0.0 USD" not in result.content


def test_a_model_already_priced_is_passed_through_and_never_charged_twice(tmp_path):
    root = _workdir(tmp_path)
    wrapped = budget.BudgetedModel(_priced_candidate(), stage="runner", workdir=root, model_id=PRICED_ID)
    assert domain_tools_mod.runner_model(wrapped, root) is wrapped
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=wrapped)}
    result = _run(tools["run"], {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    assert _runner_stage(root)["calls"] == wrapped.calls


def test_the_readers_step_keeps_its_own_stage_on_the_shared_helper(tmp_path):
    model = domain_tools_mod.reader_model(_priced_candidate(), tmp_path)
    assert (model.stage, model.cap_context) == ("readers", True)
    assert domain_tools_mod.runner_model(_priced_candidate(), tmp_path).cap_context is False
    assert domain_tools_mod.reader_model(None, tmp_path) is None
