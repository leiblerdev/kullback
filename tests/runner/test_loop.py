"""One turn at a time: the JSONL shape, the route counts, and step-by-step equal to run() (D90)."""

from __future__ import annotations

import json

import pytest
from test_route import SHARED, sigs, tools_module

from kullback.ai.provider import RetryExhausted, TestModel
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

    # a step on a stopped Run does nothing
    before = lines_of(stepped.path)
    step(stepped, model, router=stepped_router)
    assert lines_of(stepped.path) == before


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


@pytest.mark.parametrize(
    "user, question, payload, assisted",
    [
        # D49, D77: a Simulated user read that hit a synthetic row makes the whole Run Assisted.
        (lambda: Answering(["My email is ada@b.com."], payload={"sources": {"email": "world"}}, assisted=True),
         "What is your email address?",
         {"sources": {"email": "world"}, "text": "My email is ada@b.com."}, True),
        # D77: the fact_unavailable event carries the field name, which is what the report separates on.
        (lambda: Answering(["I do not have my email."],
                           payload={"tags": ["fact_unavailable"], "unavailable_fields": ["email"]}),
         "What is your email address?",
         {"tags": ["fact_unavailable"], "fact_unavailable": True, "unavailable_fields": ["email"],
          "text": "I do not have my email."}, False),
        # The refusal count reaches the Run's JSONL, a zero included: a build's report reads the refusal rate off it.
        (lambda: Answering(["My zip is 19122."], payload={"refused": 0, "refused_so_far": 2}),
         "What is your zip code?",
         {"refused": 0, "refused_so_far": 2, "text": "My zip is 19122."}, False),
        # The turn carries what the user recorded for this turn, not what it recorded for the last one.
        (lambda: Answering(["sure"], events=[Event(idx=0, type="user_turn", payload={"tags": ["fact_unavailable"]})]),
         "anything else?", {"text": "sure"}, False),
    ],
    ids=["assisted_mark", "unavailable_field", "refusal_count", "no_earlier_tag"],
)
def test_a_user_turn_carries_its_own_marks_and_no_earlier_ones(workdir, user, question, payload, assisted):
    state = new_run_state("r1", workdir=workdir, user=user(), max_turns=1 if question == "anything else?" else 2)
    run(state, TestModel([{"content": question}], loop=True), router=make_router())
    turn = [e for e in state.run.events if e.type == "user_turn"][0]
    assert turn.payload == payload
    assert turn.assisted is assisted
    assert state.run.assisted is assisted
    assert [e for e in lines_of(state.path) if e["type"] == "user_turn"][0]["assisted"] is assisted


def _calling() -> TestModel:
    return TestModel(
        [{"tool_calls": [{"id": "c1", "name": "get_order_details", "arguments": {"order_id": "123"}}]}],
        loop=True,
    )


@pytest.mark.parametrize(
    "user, model, extra, reason",
    [
        # D46: a Run handed to a human is a transfer, which is the class the Verdict reads.
        (["ok"], lambda: TestModel([{"content": "I will transfer you to a human. ###TRANSFER###"}]), {}, "transfer"),
        (["###STOP###"], lambda: TestModel([{"content": "anything else?"}], loop=True), {}, "user_stop"),
        # D90: a Run stepped by a caller that stops early is still a Run with an ending.
        (None, _calling, {"max_steps": 2}, "max_steps"),
    ],
    ids=["transfer", "stop_marker", "max_steps"],
)
def test_every_way_out_is_named_in_the_termination_reason(workdir, user, model, extra, reason):
    state = new_run_state("r1", workdir=workdir, user=Answering(user) if user else None, max_turns=50)
    run(state, model(), router=make_router(), **extra)
    assert state.run.termination_reason == reason
    assert state.stopped is True
    assert [e["type"] for e in lines_of(state.path)][-1] == "stop"
    assert footer_of(state.path)["termination_reason"] == reason


class _Boom(TestModel):
    def query(self, messages, tools=None, config=None):
        raise RuntimeError("provider down")


class _Down(TestModel):
    def query(self, messages, tools=None, config=None):
        raise RetryExhausted("anthropic/claude-opus-5: 5 attempts failed", status=503, attempts=5)


class _Angry:
    done = False

    def reply(self, transcript):
        raise RuntimeError("user sim down")


@pytest.mark.parametrize(
    "model, user, raised, error",
    [
        (lambda: _Boom([]), None, RuntimeError, {"message": "provider down"}),
        # The status and the attempts say whether the provider refused the body once or was down for five tries.
        (lambda: _Down([]), None, RetryExhausted, {"status": 503, "attempts": 5}),
        (lambda: TestModel([{"content": "anything else?"}], loop=True), _Angry, RuntimeError, None),
    ],
    ids=["model_raises", "provider_error", "user_raises"],
)
def test_a_raising_model_or_user_ends_the_run_with_an_error_event(workdir, model, user, raised, error):
    """Section 5: a provider that fell over leaves a Run that says so, not a JSONL with no ending.
    The class stays env_error, which is what the Verdict reads."""
    state = new_run_state("r1", workdir=workdir, user=user() if user else None)
    with pytest.raises(raised):
        run(state, model(), router=make_router())
    assert state.run.termination_reason == "env_error"
    assert footer_of(state.path)["termination_reason"] == "env_error"
    if error is not None:
        event = [e for e in lines_of(state.path) if e["type"] == "error"][0]
        assert event["payload"]["class"] == "env_error"
        if "message" in error:
            assert error["message"] in event["payload"]["message"]
            assert len([line for line in state.path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 4
        else:
            assert {key: event["payload"][key] for key in error} == error


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


# --- what the Run says about the call itself (D159) ---


class Wired(TestModel):
    """A model whose replies come back with an exchange, the way a live adapter's do."""

    def __init__(self, replies, exchange):
        super().__init__(replies)
        self.exchange = exchange

    def query(self, messages, tools=None, config=None):
        reply = super().query(messages, tools, config)
        reply.exchange = self.exchange
        return reply


def test_the_model_call_event_is_priced_and_timed_from_the_exchange(workdir):
    """A Run read on its own used to say every call cost nothing and took no time: only the
    budget ledger's own copy of the call was priced."""
    from kullback.ai.provider import Exchange, ModelReply
    from kullback.ai.usage import Usage

    model = Wired(
        [ModelReply(content="Your order 123 was delivered.", model="claude-opus-5",
                    usage=Usage(input=1_000_000))],
        Exchange(provider="anthropic", wall_ms=12.5, attempts=1, status=200),
    )
    state = new_run_state("r1", workdir=workdir)
    run(state, model, router=make_router())
    call = [e for e in lines_of(state.path) if e["type"] == "model_call"][0]
    assert call["cost"]["provider"] == "anthropic"
    assert call["cost"]["wall_ms"] == 12.5
    # anthropic/claude-opus-5 is 5.00 USD per million input tokens in the offline price table.
    assert call["cost"]["usd"] == pytest.approx(5.0)
    assert call["cost"]["price_source"] == "table"


def test_the_stop_event_carries_the_system_prompt_and_the_tool_specs_the_candidate_was_given(workdir):
    tools = [{"name": "get_order_details", "description": "look one order up"}]
    state = new_run_state("r1", workdir=workdir, system_prompt="you are support")
    run(state, scripted(), tools=tools, router=make_router())
    stop = [e for e in lines_of(state.path) if e["type"] == "stop"][-1]
    assert stop["payload"]["system_prompt"] == "you are support"
    assert stop["payload"]["tools"] == tools

    bare = new_run_state("r2", workdir=workdir)
    run(bare, scripted(), router=make_router())
    stop = [e for e in lines_of(bare.path) if e["type"] == "stop"][-1]
    assert stop["payload"]["system_prompt"] is None
    assert stop["payload"]["tools"] == []


def test_a_run_file_opens_with_its_run_and_task_and_a_second_write_to_its_path_raises(workdir):
    """D281: the first line names the Run and its Task, and a path that exists is never rewritten."""
    state = new_run_state("r1", workdir=workdir, task_id="t1")
    assert json.loads(state.path.read_text(encoding="utf-8").splitlines()[0]) == {"run_id": "r1", "task_id": "t1"}
    with pytest.raises(FileExistsError, match=str(state.path)):
        new_run_state("r1", workdir=workdir, task_id="t2")
    assert json.loads(state.path.read_text(encoding="utf-8").splitlines()[0])["task_id"] == "t1"


def test_a_superseding_run_replaces_only_its_own_file(workdir):
    """A replay of the same Run replaces its file; a file of another Task under that name is refused."""
    new_run_state("r1", workdir=workdir, task_id="t1")
    again = new_run_state("r1", workdir=workdir, task_id="t1", supersedes=True)
    assert again.path.read_text(encoding="utf-8").count("\n") == 1
    with pytest.raises(FileExistsError, match="Task t1"):
        new_run_state("r1", workdir=workdir, task_id="t2", supersedes=True)
