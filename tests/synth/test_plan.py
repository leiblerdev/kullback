"""The plan session before any scenario (D380), on scripted models only."""

from __future__ import annotations

import json

import pytest

from kullback.agent.loop import UNFINISHED_ASKS
from kullback.ai.provider import ModelReply, TestModel
from kullback.synth import generate as M
from kullback.synth import generate_prompt
from kullback.synth import generate_tools as T
from kullback.synth import plan as P
from kullback.synth import records as R
from tests.synth.fixture import OPENINGS, call, make_workdir, new_generation

MOVE = "a holder wants a widget moved to another slot"
BROKEN = "a holder wants a replacement for a widget that arrived broken"


def _situation(call_id: str, statement: str, openings: list[int], kind: str = "change", variants=()) -> ModelReply:
    return call("write_situation", {"statement": statement, "kind": kind, "openings": openings,
                                    "variants": list(variants)}, call_id)


def _done() -> ModelReply:
    return ModelReply(content="Planned them.")


def _results(model: TestModel, at: int = -1) -> list[str]:
    return [str(m.get("content")) for m in model.calls[at]["messages"] if m.get("role") == "tool"]


def test_a_plan_session_writes_skips_and_a_later_session_attaches_to_a_situation_written_earlier(tmp_path):
    generation, model = new_generation(
        tmp_path,
        call("list_openings", {}, "p0"),
        _situation("p1", MOVE, [0], variants=["the widget is already sealed, so a replacement is offered"]),
        call("skip_opening", {"openings": [1], "why": "identification only"}, "p2"), _done(),
        call("list_plan", {}, "q0"), call("attach_openings", {"id": P.situation_id(MOVE), "openings": [0]}, "q1"),
        _done())
    M.plan_openings(generation, batch=2)
    [situation] = generation.plan.situations
    assert [row["task_id"] for row in situation["openings"]] == ["task-a1", "task-c3"]
    assert situation["variants"] == ["the widget is already sealed, so a replacement is offered"]
    assert generation.plan.skipped == [{"task_id": "task-b2", "recording": "rec-task-b2", "why": "identification only"}]
    second_opening = json.dumps(model.calls[4]["messages"])
    assert P.situation_id(MOVE) in second_opening and MOVE in second_opening
    assert generation.unplanned == []


def test_an_opening_a_situation_holds_is_refused_naming_the_holder(tmp_path):
    generation, model = new_generation(
        tmp_path, _situation("p0", MOVE, [0, 1]), _situation("p1", BROKEN, [1, 2], kind="complaint"),
        _situation("p2", BROKEN, [2], kind="complaint"), _done())
    M.plan_openings(generation)
    refusal = _results(model)[1]
    assert P.HELD.replace("_", " ") in refusal and P.situation_id(MOVE) in refusal and "1 (in" in refusal
    assert [len(s["openings"]) for s in generation.plan.situations] == [2, 1]


@pytest.mark.parametrize("hedged", ["a holder sometimes wants a widget moved",
                                    "a holder wants a widget moved or replaced"])
def test_a_hedged_statement_is_refused_and_asks_for_a_variant(tmp_path, hedged):
    generation, model = new_generation(tmp_path, _situation("p0", hedged, [0, 1, 2]), _situation("p1", MOVE, [0, 1, 2]),
                                       _done())
    M.plan_openings(generation)
    assert "hedged" in _results(model)[0] and "variant" in _results(model)[0]
    assert [s["statement"] for s in generation.plan.situations] == [MOVE]


def test_the_retry_session_runs_only_over_the_openings_left_in_no_situation(tmp_path):
    generation, model = new_generation(
        tmp_path, _situation("p0", MOVE, [0, 1]), *[_done()] * (1 + UNFINISHED_ASKS),
        call("list_openings", {}, "r0"), _situation("r1", BROKEN, [0], kind="complaint"), _done())
    M.plan_openings(generation)
    listed = _results(model)[0]
    assert OPENINGS["task-c3"] in listed and OPENINGS["task-a1"] not in listed and "1 items" in listed
    assert [s["openings"][0]["task_id"] for s in generation.plan.situations] == ["task-a1", "task-c3"]
    assert generation.unplanned == [] and len(model.calls) == 5 + UNFINISHED_ASKS


def test_the_plan_is_written_with_its_shape_after_every_session(tmp_path):
    generation, _ = new_generation(tmp_path, _situation("p0", MOVE, [0, 1]),
                                   call("skip_opening", {"openings": [2], "why": "no ask"}, "p1"), _done())
    M.plan_openings(generation)
    body = json.loads(P.plan_path(generation.world.workdir).read_text())
    assert body == {"format": 1, "situations": [
        {"id": P.situation_id(MOVE), "statement": MOVE, "kind": "change", "variants": [],
         "openings": [{"task_id": "task-a1", "recording": "rec-task-a1", "text": OPENINGS["task-a1"]},
                      {"task_id": "task-b2", "recording": "rec-task-b2", "text": OPENINGS["task-b2"]}]}],
        "skipped": [{"task_id": "task-c3", "recording": "rec-task-c3", "why": "no ask"}]}


def test_openings_unplanned_counts_what_the_retry_left_and_masks_the_task_ids(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([_situation("p0", MOVE, [0]), _done(), _done(), _done()])
    body = M.generate(workdir, model=model, searching=False)
    assert body["coverage"]["openings_unplanned"] == {"count": 2, "ids": [R.mask("task-b2"), R.mask("task-c3")]}


def test_planning_again_reads_the_plan_on_disk_and_runs_no_session_when_every_opening_has_its_place(tmp_path):
    generation, _ = new_generation(tmp_path, _situation("p0", MOVE, [0, 1, 2]), _done())
    M.plan_openings(generation)
    again = M.Generation(world=generation.world, model=TestModel([]))
    M.plan_openings(again)
    assert [s["id"] for s in again.plan.situations] == [P.situation_id(MOVE)] and again.unplanned == []


def test_the_plan_prompt_runs_in_gepa_order_and_holds_its_choice_rule():
    sections = dict(generate_prompt.sections(T.SESSION_TOOLS["plan"], "plan"))
    assert list(sections) == ["receives", "tools", "examples", "choice", "feedback", "stop"]
    assert "never a second situation and never a hedge" in " ".join(sections["choice"].split())
    assert all(f"{name}:" in sections["tools"] for name in ("write_situation", "attach_openings", "skip_opening"))
