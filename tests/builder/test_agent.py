"""The Builder session: the model drives it, no turn cap, the spend ceiling is the hard stop (D135, D138).

The code driver is unchanged and still leaves what `build.build()` leaves, byte for byte; that is
the last test here and the reason the offline path stays deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from session_fixtures import collect, fixture_path, reply, tree
from test_build import Bodies

from kullback.agent.events import ToolExecutionEnd
from kullback.agent.messages import AssistantMessage
from kullback.ai.provider import TestModel
from kullback.ai.usage import Usage
from kullback.builder import agent as builder_agent
from kullback.builder import build as build_module
from kullback.builder.build import BuildPlan
from kullback.runner import budget

ASSISTED = "list_all_product_types"
# The tokens and the price row are each other's counterpart: one call costs a thousand dollars, so
# a one dollar ceiling is reached by the first model call any stage makes.
TOKENS = 1000
DEAR = {"input": 1_000_000.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}


class Priced(Bodies):
    """The fixture's Builder model with tokens on every reply, so a ceiling in dollars is reachable."""

    def query(self, messages, tools=None, config=None):
        return super().query(messages, tools=tools, config=config).model_copy(
            update={"usage": Usage(input=TOKENS)})


@pytest.fixture(scope="module")
def built(tmp_path_factory, request) -> Path:
    """One code-driven build of the fixture: the world a session then reads and repairs."""
    workdir = tmp_path_factory.mktemp("session")
    builder_agent.run_builder(workdir, model=Bodies(), files=[fixture_path(request)], max_attempts=0)
    return workdir


def _a_task(workdir: Path) -> str:
    return sorted(json.loads((workdir / "replays.json").read_text(encoding="utf-8")))[0]


# --- the model drives ----------------------------------------------------------

def test_a_session_reads_status_repairs_the_tool_it_names_replays_and_ends_when_it_stops_calling_tools(built):
    task = _a_task(built)
    model = TestModel([
        reply(None, ("status", {})),
        reply(None, ("repair_recompile", {"name": ASSISTED, "hint": "the rows must match the recording"})),
        reply(None, ("replay", {"task": task})),
        reply("every gate read; nothing left to repair."),
    ])
    events = []
    result = builder_agent.run_builder(built, model=Bodies(), iterate=True, max_attempts=0,
                                       agent_model=model, subscribers=[events.append])
    called = [e.tool_name for e in events if isinstance(e, ToolExecutionEnd)]
    assert called == ["status", "repair_recompile", "replay"]
    assert len(model.calls) == 4, "one call per turn, and the fourth answer ends the run"
    # The status the model read named the assisted tool and the verb that owns it, and that is the
    # verb it called next.
    status = json.dumps(model.calls[1]["messages"])
    assert ASSISTED in status and "-> repair_recompile" in status
    assert events[-1].type == "agent_end"
    last = [m for m in events if getattr(m, "type", None) == "message_end"][-1].message
    assert isinstance(last, AssistantMessage) and not last.tool_calls
    assert result["status"] == "complete" and result["tool_result"]["is_error"] is False


def test_a_session_runs_past_the_turn_cap_phase_four_had_because_there_is_no_cap(built):
    """tau's rule: the run ends when the model answers with no tool call, and nothing else ends it."""
    turns = 12  # phase 4 stopped at 8
    model = TestModel([*[reply(None, ("status", {})) for _ in range(turns)], reply("nothing to do.")])
    plan = BuildPlan(workdir=built, iterate=True)
    harness = builder_agent.build_harness(plan, agent_model=model)
    assert harness.max_turns is None, "the Builder's session has no turn cap (D135)"
    events = collect(harness.prompt(builder_agent.builder_message("environment")))
    assert len([e for e in events if isinstance(e, ToolExecutionEnd)]) == turns
    assert len(model.calls) == turns + 1
    text = " ".join(m.message.content or "" for m in events
                    if getattr(m, "type", None) == "message_end" and isinstance(m.message, AssistantMessage))
    assert "max_turns" not in text


def test_the_opening_message_asks_it_to_begin_with_status(built):
    message = builder_agent.builder_message("environment")
    assert message == ("Begin on the target 'environment'. Call status first and read the red lights, "
                       "then use the tools to make every gate pass. Answer with no tool call when "
                       "every gate is green, or when nothing is changing.")
    harness = builder_agent.build_harness(BuildPlan(workdir=built, iterate=True))
    assert "make every gate pass" in harness.system
    assert "Repair only what a model wrote" in harness.system
    assert "Never a gate, the Runner, the judge or the Simulated user" in harness.system
    assert "repair_recompile(name, hint)" in harness.system
    assert "—" not in harness.system and "–" not in harness.system


def test_a_model_that_calls_no_building_tool_at_all_is_a_build_error(tmp_path, request):
    model = TestModel([reply(None, ("status", {})), reply("I would rather not.")])
    with pytest.raises(build_module.BuildError, match="never called build"):
        builder_agent.run_builder(tmp_path, model=Bodies(), files=[fixture_path(request)],
                                  agent_model=model, max_attempts=0)


# --- the ceiling is the hard stop ----------------------------------------------

def test_the_spend_ceiling_stops_the_session_on_the_tool_call_that_reaches_it(tmp_path, request, monkeypatch):
    """D86 and D135: with no turn cap the ceiling is what bounds a run, so the tool that reaches it
    is the last one and the model is never asked again."""
    monkeypatch.setitem(budget.PRICES, "test", DEAR)
    model = TestModel([reply(None, ("build", {"target": "environment"})),
                       reply(None, ("status", {})),
                       reply("still going.")])
    result = builder_agent.run_builder(tmp_path, model=Priced(), files=[fixture_path(request)],
                                       max_attempts=0, agent_model=model, ceiling_usd=1.0)
    assert result["status"] == "stopped"
    assert result["stopped"]["ceiling_usd"] == pytest.approx(1.0)
    assert result["stopped"]["spent"] >= 1.0
    assert len(model.calls) == 1, "the ceiling cancelled the run, so the model was never asked again"


def test_without_a_ceiling_nothing_guards_the_stream(built):
    plan = BuildPlan(workdir=built, iterate=True)
    harness = builder_agent.build_harness(plan)
    release, stop = builder_agent.ceiling_guard(harness, plan)
    release()
    assert stop == {} and plan.ceiling is None


# --- the code driver is unchanged ----------------------------------------------

def test_the_code_driver_leaves_what_build_build_leaves_byte_for_byte(tmp_path, request):
    """The comparison phase 4 rests on, held against the function this phase did not touch: with no
    agent model the driver's workdir and `build.build()`'s are the same bytes, file by file."""
    driven, plain = tmp_path / "driven", tmp_path / "plain"
    builder_agent.run_builder(driven, model=Bodies(), files=[fixture_path(request)], max_attempts=0)
    build_module.build(plain, model=Bodies(), files=[fixture_path(request)], max_attempts=0)
    left, right = tree(driven), tree(plain)
    assert set(left) == set(right)
    assert [name for name in left if left[name] != right[name]] == []
