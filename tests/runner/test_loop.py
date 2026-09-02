"""One turn at a time: the JSONL shape, the route counts, and step-by-step equal to run() (D90)."""

from __future__ import annotations

import json

import pytest
from test_route import SHARED, sigs, tools_module

from kullback.ai.provider import TestModel
from kullback.runner.loop import finish, new_run_state, run, step
from kullback.runner.records import Event, Run
from kullback.runner.route import Router
from kullback.runner.state import StateView


def make_router(stand_in_model=None) -> Router:
    return Router(
        env_tools_module=tools_module(),
        recordings=None,
        starting_state=StateView(shared=json.loads(json.dumps(SHARED))),
        overlay=None,
        stand_in_model=stand_in_model,
        tool_sigs=sigs(),
    )


def scripted() -> TestModel:
    """Call get_order_details, then answer in words and stop."""
    return TestModel([
        {"content": None, "tool_calls": [{"id": "c1", "name": "get_order_details", "arguments": {"order_id": "123"}}]},
        {"content": "Your order 123 was delivered."},
    ])


def lines_of(path) -> list[dict]:
    """The event lines of a Run JSONL; finish() also writes a trailing line naming the Run."""
    objects = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [obj for obj in objects if "type" in obj]


def footer_of(path) -> dict:
    """The trailing line: the Run's identity, its termination reason and its Start and End state."""
    objects = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return next((obj for obj in reversed(objects) if "type" not in obj), {})


def test_run_writes_one_jsonl_line_per_event(workdir):
    state = new_run_state("r1", workdir=workdir, system_prompt="you are support", first_user="where is order 123")
    run(state, scripted(), tools=[{"name": "get_order_details"}], router=make_router())
    events = lines_of(state.path)
    assert [e["type"] for e in events] == ["model_call", "tool_call", "tool_result", "model_call", "stop"]
    assert [e["idx"] for e in events] == [0, 1, 2, 3, 4]
    assert len(events) == len(state.run.events)
    assert state.stopped is True
    assert state.run.termination_reason == "agent_stop"


def test_route_counts_and_the_route_on_the_event(workdir):
    state = new_run_state("r1", workdir=workdir)
    run(state, scripted(), router=make_router())
    assert state.run.route_counts == {"code": 1}
    result_event = [e for e in lines_of(state.path) if e["type"] == "tool_result"][0]
    assert result_event["route"] == "code"
    assert result_event["payload"]["result"]["status"] == "delivered"
    assert state.run.assisted is False


def test_the_tool_result_goes_back_into_the_transcript(workdir):
    state = new_run_state("r1", workdir=workdir)
    run(state, scripted(), router=make_router())
    roles = [m["role"] for m in state.messages]
    assert roles == ["assistant", "tool", "assistant"]
    assert "delivered" in state.messages[1]["content"]


def test_step_by_step_equals_run(workdir):
    """D90: the loop is a function that advances one turn, and stepping it gives the same JSONL."""
    whole = new_run_state("r1", workdir=workdir, first_user="where is order 123")
    run(whole, scripted(), router=make_router())

    # the same Run, stepped by hand one turn at a time
    stepped = new_run_state("r1", path=workdir / "stepped.jsonl", first_user="where is order 123")
    model, stepped_router = scripted(), make_router()
    for _ in range(10):
        if stepped.stopped:
            break
        step(stepped, model, router=stepped_router)
    finish(stepped, stepped_router)

    assert lines_of(stepped.path) == lines_of(whole.path)
    assert stepped.run.route_counts == whole.run.route_counts
    assert stepped.run.end_state_hash == whole.run.end_state_hash
    assert stepped.messages == whole.messages


def test_step_on_a_stopped_run_does_nothing(workdir):
    state = new_run_state("r1", workdir=workdir)
    model = scripted()
    run(state, model, router=make_router())
    before = lines_of(state.path)
    step(state, model, router=make_router())
    assert lines_of(state.path) == before


def test_max_turns_stops_the_run(workdir):
    calling = TestModel(
        [{"tool_calls": [{"id": "c1", "name": "get_order_details", "arguments": {"order_id": "123"}}]}],
        loop=True,
    )
    state = new_run_state("r1", workdir=workdir, max_turns=3)
    run(state, calling, router=make_router())
    assert state.run.termination_reason == "max_turns"
    assert state.turn == 3
    assert state.run.route_counts == {"code": 3}


def test_a_stand_in_marks_the_run_assisted(workdir):
    """D49: one stand-in event and the whole Run is Assisted."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "lookup_user", "arguments": {"email": "a@b.com"}}]},
        {"content": "done"},
    ])
    state = new_run_state("r1", workdir=workdir)
    run(state, model, router=make_router(stand_in_model=TestModel(['{"name": "Ada"}'])))
    assert state.run.assisted is True
    assert state.run.route_counts == {"llm": 1}
    assert [e for e in lines_of(state.path) if e["type"] == "tool_result"][0]["assisted"] is True


def test_a_bad_call_is_answered_and_the_run_carries_on(workdir):
    """D45: a hallucinated tool call has no effect and does not end the Run."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "teleport_order", "arguments": {}}]},
        {"content": "sorry, I cannot do that"},
    ])
    state = new_run_state("r1", workdir=workdir)
    run(state, model, router=make_router())
    result = [e for e in lines_of(state.path) if e["type"] == "tool_result"][0]
    assert result["payload"]["error"]["class"] == "tool_not_found"
    assert state.run.termination_reason == "agent_stop"
    assert state.run.assisted is False


def test_the_simulated_user_gets_the_turn_when_there_are_no_tool_calls(workdir):
    class Chatty:
        def __init__(self):
            self.done = False
            self.turns = 0

        def reply(self, transcript):
            self.turns += 1
            if self.turns >= 2:
                self.done = True
                return "###STOP###"
            return "yes please cancel it"

    model = TestModel([{"content": "anything else?"}], loop=True)
    state = new_run_state("r1", workdir=workdir, user=Chatty())
    run(state, model, router=make_router())
    types_ = [e["type"] for e in lines_of(state.path)]
    assert types_ == ["model_call", "user_turn", "model_call", "user_turn", "stop"]
    assert state.run.termination_reason == "user_stop"
    assert state.messages[-2]["role"] == "user"


class Answering:
    """A Simulated user in the shape user_sim.SimulatedUser has: a reply, and its own event per turn."""

    def __init__(self, answers, payload=None, assisted=False, events=None):
        self.answers = list(answers)
        self.payload = dict(payload or {})
        self.assisted = assisted
        self.events: list[Event] = list(events or [])
        self.done = False

    def reply(self, transcript):
        answer = self.answers.pop(0) if self.answers else "###STOP###"
        if self.payload:
            self.events.append(Event(idx=len(self.events), type="user_turn",
                                     payload=dict(self.payload, text=answer), assisted=self.assisted))
        return answer


def test_the_user_turn_carries_the_users_own_assisted_mark(workdir):
    """D49, D77: a Simulated user read that hit a synthetic row makes the whole Run Assisted."""
    user = Answering(["My email is ada@b.com."], payload={"sources": {"email": "world"}}, assisted=True)
    state = new_run_state("r1", workdir=workdir, user=user, max_turns=2)
    run(state, TestModel([{"content": "What is your email address?"}], loop=True), router=make_router())
    turn = [e for e in state.run.events if e.type == "user_turn"][0]
    assert turn.assisted is True
    assert turn.payload["sources"] == {"email": "world"}
    assert state.run.assisted is True
    assert [e for e in lines_of(state.path) if e["type"] == "user_turn"][0]["assisted"] is True


def test_the_user_turn_names_the_field_the_world_could_not_give(workdir):
    """D77: the fact_unavailable event carries the field name, which is what the report separates on."""
    user = Answering(["I do not have my email."],
                     payload={"tags": ["fact_unavailable"], "unavailable_fields": ["email"]})
    state = new_run_state("r1", workdir=workdir, user=user, max_turns=2)
    run(state, TestModel([{"content": "What is your email address?"}], loop=True), router=make_router())
    payload = [e for e in state.run.events if e.type == "user_turn"][0].payload
    assert payload["tags"] == ["fact_unavailable"]
    assert payload["fact_unavailable"] is True
    assert payload["unavailable_fields"] == ["email"]


def test_a_tag_from_an_earlier_turn_is_not_copied_onto_this_one(workdir):
    """The turn carries what the user recorded for this turn, not what it recorded for the last one."""
    earlier = Event(idx=0, type="user_turn", payload={"tags": ["fact_unavailable"]})
    user = Answering(["sure"], events=[earlier])  # answers without recording an event of its own
    state = new_run_state("r1", workdir=workdir, user=user, max_turns=1)
    run(state, TestModel([{"content": "anything else?"}], loop=True), router=make_router())
    payload = [e for e in state.run.events if e.type == "user_turn"][0].payload
    assert payload == {"text": "sure"}


def test_a_transfer_is_named_in_the_termination_reason(workdir):
    """D46: a Run handed to a human is a transfer, which is the class the Verdict reads."""
    state = new_run_state("r1", workdir=workdir, user=Answering(["ok"]))
    run(state, TestModel([{"content": "I will transfer you to a human. ###TRANSFER###"}]),
        router=make_router())
    assert state.run.termination_reason == "transfer"
    assert footer_of(state.path)["termination_reason"] == "transfer"


def test_a_stop_marker_is_still_a_user_stop(workdir):
    state = new_run_state("r1", workdir=workdir, user=Answering(["###STOP###"]))
    run(state, TestModel([{"content": "anything else?"}], loop=True), router=make_router())
    assert state.run.termination_reason == "user_stop"


def test_a_model_that_raises_ends_the_run_with_an_error_event_and_a_footer(workdir):
    """Section 5: a provider that fell over leaves a Run that says so, not a JSONL with no ending."""
    class Boom(TestModel):
        def query(self, messages, tools=None, config=None):
            raise RuntimeError("provider down")

    state = new_run_state("r1", workdir=workdir)
    with pytest.raises(RuntimeError):
        run(state, Boom([]), router=make_router())
    error = [e for e in lines_of(state.path) if e["type"] == "error"][0]
    assert error["payload"]["class"] == "env_error"
    assert "provider down" in error["payload"]["message"]
    assert state.run.termination_reason == "env_error"
    assert footer_of(state.path)["termination_reason"] == "env_error"
    assert len([line for line in state.path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 3


def test_a_user_that_raises_ends_the_run_the_same_way(workdir):
    class Angry:
        done = False

        def reply(self, transcript):
            raise RuntimeError("user sim down")

    state = new_run_state("r1", workdir=workdir, user=Angry())
    with pytest.raises(RuntimeError):
        run(state, TestModel([{"content": "anything else?"}], loop=True), router=make_router())
    assert state.run.termination_reason == "env_error"
    assert footer_of(state.path)["termination_reason"] == "env_error"


def test_leaving_by_max_steps_still_names_a_termination_reason(workdir):
    """D90: a Run stepped by a caller that stops early is still a Run with an ending."""
    calling = TestModel(
        [{"tool_calls": [{"id": "c1", "name": "get_order_details", "arguments": {"order_id": "123"}}]}],
        loop=True,
    )
    state = new_run_state("r1", workdir=workdir, max_turns=50)
    run(state, calling, router=make_router(), max_steps=2)
    assert state.run.termination_reason == "max_steps"
    assert state.stopped is True
    assert [e["type"] for e in lines_of(state.path)][-1] == "stop"
    assert footer_of(state.path)["termination_reason"] == "max_steps"


def test_events_carry_no_wall_clock(workdir):
    """Section 8: nothing is keyed by wall-clock time or run order beyond idx."""
    state = new_run_state("r1", workdir=workdir)
    run(state, scripted(), router=make_router())
    assert lines_of(state.path)
    for event in lines_of(state.path):
        assert event["ts"] is None
        assert (event.get("cost") or {}).get("wall_ms", 0) == 0
    # the same Run written twice is the same file, which is what content addressing rests on
    twice = new_run_state("r1", path=workdir / "again.jsonl")
    run(twice, scripted(), router=make_router())
    assert twice.path.read_text(encoding="utf-8") == state.path.read_text(encoding="utf-8")


def test_the_model_call_event_replays_through_recordedmodel(workdir):
    """The JSONL our loop writes is the JSONL RecordedModel reads."""
    from kullback.ai.provider import RecordedModel

    state = new_run_state("r1", workdir=workdir)
    run(state, scripted(), router=make_router())
    replayed = RecordedModel(state.path)
    first = replayed.query([])
    assert first.tool_calls[0].name == "get_order_details"
    assert replayed.query([]).content == "Your order 123 was delivered."


def test_run_record_fields_are_carried_through(workdir):
    state = new_run_state("r1", workdir=workdir, env_id="e1", task_id="t1", seed=7, model="m")
    assert isinstance(state.run, Run)
    router = make_router()
    before = router.state_hash()
    run(state, scripted(), router=router)
    assert (state.run.env_id, state.run.task_id, state.run.seed) == ("e1", "t1", 7)
    assert state.run.end_state_hash == router.state_hash()
    footer = footer_of(state.path)
    assert (footer["env_id"], footer["task_id"], footer["seed"]) == ("e1", "t1", 7)
    assert footer["end_state_hash"] == state.run.end_state_hash

    # a Run that wrote ends on a different state hash than it started on, and the footer shows both
    writing = TestModel([
        {"tool_calls": [{"id": "c1", "name": "cancel_order", "arguments": {"order_id": "123"}}]},
        {"content": "cancelled"},
    ])
    wrote = new_run_state("r2", workdir=workdir)
    run(wrote, writing, router=router)
    assert wrote.run.end_state_hash != before
    # The states are on the stop event, not on the footer: the footer is validated as a Run.
    stop = [e for e in wrote.run.events if e.type == "stop"][-1]
    assert stop.payload["start_state"]["orders"]["123"]["status"] == "delivered"
    assert stop.payload["end_state"]["orders"]["123"]["status"] == "cancelled"
    assert "start_state" not in footer_of(wrote.path)


def test_a_tool_call_without_a_router_is_refused(workdir):
    state = new_run_state("r1", workdir=workdir)
    with pytest.raises(ValueError):
        run(state, scripted(), router=None)
