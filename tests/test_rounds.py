"""The round driver (D126, D128): turn-taking on one stream, the four beats, the counts off the gates,
the three exits, the allowance, and the findings drained from the Examiner into the Builder.

Nothing here calls a model. The Builder's stages run over the small tau2 fixture on the scripted
Bodies model, and the model-driven paths run on a TestModel scripted to call the tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from builder.test_build import Bodies
from kullback import rounds
from kullback.agent.events import BeatEnd, BeatStart, RoundEnd, RoundStart, ToolExecutionEnd, ToolExecutionStart
from kullback.agent.harness import AgentHarness
from kullback.ai.messages import UserMessage
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import agent as builder_agent
from kullback.builder import pipeline, repair
from kullback.builder.build import BuildError, BuildPlan
from kullback.builder.tools import repair_verb_tools
from kullback.examiner import agent as examiner_agent
from kullback.examiner.agent import examiner_message, examiner_round_message
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.stage import DERIVE_INPUTS, FORBIDDEN_INPUTS
from kullback.gates import round_end
from kullback.gates.ledger import GateLedger
from kullback.runner import budget
from kullback.runner.records import Finding, GateResult, RoundRecord, as_dict

TARGET = "environment"
# sha256 of task_status.json from the full offline build over the fixture made before this phase, through
# run_builder with derive_verifier still a Builder stage (D130): the rounds must leave the same bytes.
# Re-pinned once when `reference.describe` began saying, of a Run that wrote nothing, whether its answer
# stated facts read from the world: the same two groups and the same verdicts, one longer state sentence
# inside the reason. Re-pinned again for D171: every row gained the Task's own replay fidelity
# (`blocking_tools`, `tool_calls_replayed`, `tool_calls_differing`) and the reason names a tool only
# where one of the Task's own recorded calls differs. Every other byte of the three rows is the
# pre-phase build's.
TASK_STATUS_SHA256_BEFORE_THE_PHASE = "a6186b51b650951ec677399bf4e71f19a753423400b93593ba263c747199d8b2"


def _fixture(request) -> Path:
    return Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"


def _reply(content, *calls):
    return ModelReply(content=content, tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a)
                                                   for i, (n, a) in enumerate(calls)])


def _collect(aiter):
    async def go():
        return [event async for event in aiter]
    return asyncio.run(go())


def _finding(task_id: str, suggested: str = "replay", finding_id: str = "f1") -> Finding:
    return Finding(finding_id=finding_id, task_id=task_id, kind="fidelity", suggested=suggested, round=1,
                   text="the replay of this Task diverges from its Trace at the second call")


def _bare_loop(tmp_path: Path, **kwargs) -> rounds.Loop:
    """A Loop over an empty plan, for what the driver decides without a beat: the allowance, the exit."""
    plan = BuildPlan(workdir=tmp_path / "work")
    return rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan), **kwargs)


def _beats(events) -> list[tuple[str, str, int]]:
    return [(e.type, e.agent, e.round) for e in events if isinstance(e, (BeatStart, BeatEnd))]


# --- the Builder's side of a round, driven by code ----------------------------------------

@pytest.fixture(scope="module")
def code_loop(tmp_path_factory, request):
    """A code-driven Loop after its round-1 Builder beat, then a round-2 beat with one finding pending.

    The Examiner's beat is not run here: this fixture pins what the driver does with the Builder alone.
    """
    workdir = tmp_path_factory.mktemp("code")
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0)
    events, dicts = [], []
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, None, [events.append]),
                       target=TARGET, subscribers=[events.append], on_event=dicts.append)
    loop.allowance = {agent: loop.allowance_for(agent) for agent in rounds.AGENTS}
    loop.builder_beat(1)
    first = len(events)
    task_id = plan.last.artifacts["tasks"][0].id
    # The second finding's verb is the Examiner's own (D170): the Builder has no such tool, so the
    # driver must deliver it and not call it.
    loop.pending_findings = [_finding(task_id), _finding(task_id, suggested="repair", finding_id="f2")]
    loop.builder_beat(2)
    return {"loop": loop, "plan": plan, "events": events, "dicts": dicts, "first": first, "task_id": task_id}


def test_beat_events_name_the_agent_and_the_round_and_the_dict_stream_sees_them_too(code_loop):
    events, dicts = code_loop["events"], code_loop["dicts"]
    assert _beats(events) == [("beat_start", "builder", 1), ("beat_end", "builder", 1),
                              ("beat_start", "builder", 2), ("beat_end", "builder", 2)]
    kinds = [e.type for e in events[:code_loop["first"]]]
    assert kinds[0] == "beat_start" and kinds[-1] == "beat_end"
    assert "stage_start" in kinds and "tool_execution_end" in kinds, "the build's own events sit inside the beat"
    assert [(d["kind"], d["state"], d["agent"], d["round"]) for d in dicts if d.get("kind") == "beat"] == [
        ("beat", "start", "builder", 1), ("beat", "end", "builder", 1),
        ("beat", "start", "builder", 2), ("beat", "end", "builder", 2)]
    assert all(isinstance(e.spend, float) for e in events if isinstance(e, BeatEnd))


def test_the_code_driver_calls_the_builder_tool_a_finding_names_and_skips_a_verb_the_builder_has_not(code_loop):
    """A finding that suggests `replay` for a Task is a replay(task) call through the Builder's hooks,
    before the build of the target; the code driver never asks a model what to do with it. A finding
    whose answer is the Examiner's own `repair` names an artifact the Builder cannot touch (D123), so
    it is delivered and not called: driving it would only ever be an error result (D170)."""
    second = code_loop["events"][code_loop["first"]:]
    starts = [e for e in second if isinstance(e, ToolExecutionStart)]
    assert [(e.tool_name, e.arguments) for e in starts] == [
        ("replay", {"task": code_loop["task_id"]}), ("build", {"target": TARGET})]
    ends = [e for e in second if isinstance(e, ToolExecutionEnd)]
    assert [e.is_error for e in ends] == [False, False]
    assert code_loop["loop"].pending_findings == [], "a delivered finding is not pending any more"
    assert code_loop["loop"].build_result is ends[-1].result


def test_the_examiner_beat_starts_only_after_the_builder_harness_is_idle_and_the_driver_refuses_otherwise(tmp_path):
    """D128: one agent at a time. Either beat asked for while the other harness is mid-run is a
    RuntimeError from the driver, before any tool call."""
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=AgentHarness(TestModel(["built"])))
    loop.examiner = AgentHarness(TestModel(["examined"]))
    refused = []

    def while_builder_runs(event):
        if event.type == "turn_start":
            assert loop.builder.is_running
            with pytest.raises(RuntimeError, match="the Builder is still running; one agent at a time"):
                loop.examiner_beat(1)
            refused.append("examiner")

    def while_examiner_runs(event):
        if event.type == "turn_start":
            with pytest.raises(RuntimeError, match="the Examiner is still running; one agent at a time"):
                loop.builder_beat(1)
            refused.append("builder")

    loop.builder.subscribe(while_builder_runs)
    _collect(loop.builder.prompt("build"))
    loop.examiner.subscribe(while_examiner_runs)
    _collect(loop.examiner.prompt("examine"))
    assert refused == ["examiner", "builder"]
    assert not loop.builder.is_running and not loop.examiner.is_running


# --- the Builder's side of a round, driven by a model ----------------------------------------

@pytest.fixture(scope="module")
def model_loop(tmp_path_factory, request):
    """A model-driven Loop under an allowance of zero: the round-1 beat builds twice, the round-2 beat
    is one finding delivered as a follow-up. The model is scripted turn for turn."""
    workdir = tmp_path_factory.mktemp("agent")
    agent_model = TestModel([
        _reply(None, ("build", {"target": TARGET})),
        _reply(None, ("build", {"target": "cluster"})),
        _reply("every ruling passed."),
        _reply("read the round message."),
        _reply("acted on the finding."),
    ])
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0)
    events = []
    harness = builder_agent.build_harness(plan, agent_model, [events.append])
    loop = rounds.Loop(plan=plan, builder=harness, target=TARGET, agent_model=agent_model, allowance_usd=0.0,
                       subscribers=[events.append])
    loop.allowance = {agent: loop.allowance_for(agent) for agent in rounds.AGENTS}
    loop.builder_beat(1)
    after_first = len(harness.messages)
    task_id = plan.last.artifacts["tasks"][0].id
    finding = _finding(task_id, suggested="none", finding_id="f7")
    loop.pending_findings = [finding]
    loop.builder_beat(2)
    return {"loop": loop, "harness": harness, "events": events, "after_first": after_first,
            "finding": finding, "model": agent_model}


def test_an_exhausted_allowance_steers_a_model_driven_agent_once(model_loop):
    """An allowance of zero is spent at the first tool end; the driver steers the Builder once, after
    the tool batch, and not again at the second tool end (D123: the ceiling is a steer, not a stop)."""
    harness, loop = model_loop["harness"], model_loop["loop"]
    first = harness.messages[:model_loop["after_first"]]
    steers = [m for m in first if isinstance(m, UserMessage) and m.content == rounds.ALLOWANCE_STEER]
    assert len(steers) == 1
    assert [m.role for m in first] == ["user", "assistant", "tool", "user", "assistant", "tool", "assistant"]
    assert loop.spent_allowance["builder"] is True
    ends = [e for e in model_loop["events"] if isinstance(e, ToolExecutionEnd)]
    assert len(ends) == 4, "two tool ends of the model's own; the driver's build at the first beat, where " \
                           "the model's last build was another target and the store held only its artifacts " \
                           "(D161); and the driver's build at the second beat, where the model acted on the " \
                           "finding and never called build (D153)"
    assert loop.driver_built == [1, 2]


def test_a_finding_from_the_examiner_is_a_follow_up_on_the_builder_at_the_next_beat_with_the_record_in_details(model_loop):
    """D123: the two agents talk by events. The finding reaches the Builder as a user message whose
    details carry the Finding record as data, which never enters the model's context."""
    harness, finding = model_loop["harness"], model_loop["finding"]
    second = harness.messages[model_loop["after_first"]:]
    users = [m for m in second if isinstance(m, UserMessage)]
    assert [m.role for m in second] == ["user", "assistant", "user", "assistant"]
    assert users[0].content.startswith("round 2:") and users[0].details is None
    assert users[1].details == {"finding": as_dict(finding)}
    assert finding.finding_id in users[1].content and finding.text in users[1].content
    assert model_loop["loop"].pending_findings == []


def test_the_builder_is_handed_the_finding_that_costs_the_most_tasks_and_a_verb_it_has(model_loop):
    """D170: three builds opened every round on one Task's Intent while an assisted tool blocked
    fifty. The lead is the costliest finding naming a verb the Builder can call; a finding answered
    by the Examiner's own `repair` is delivered as a report, since rendering `repair(...)` would send
    the Builder after the one artifact D123 keeps out of its hands."""
    tool = Finding(finding_id="f1", kind="assisted_tool", tool="renew_loan", suggested="repair_recompile",
                   hint="the due_date column differs", text="renew_loan is assisted",
                   task_ids=["t1", "t2", "t3"])
    verifier = Finding(finding_id="f2", kind="suite", suggested="repair", text="mutation_flips failed",
                       task_ids=["t1", "t2", "t3", "t4"])
    intent = Finding(finding_id="f3", kind="fidelity", task_id="t9", suggested="repair_intent",
                     hint="the Runs say renewal", text="the Intent says extension")
    assert rounds.leading_finding([intent, verifier, tool]) is tool, "costliest of the ones it can call"
    assert rounds.leading_finding([verifier]) is None and rounds.leading_finding([]) is None
    assert [f.finding_id for f in sorted([intent, tool, verifier], key=lambda f: -f.cost)] == ["f2", "f1", "f3"]
    message = rounds.finding_message(tool)
    assert message.startswith("Finding f1 (assisted_tool, 3 Tasks):")
    assert "repair_recompile(name='renew_loan', hint='the due_date column differs')" in message
    owned = rounds.finding_message(verifier)
    assert "Answered by repair, and the Examiner owns it" in owned and "repair(" not in owned
    # The steer of a real beat says the order the messages after it come in.
    steer = model_loop["harness"].messages[model_loop["after_first"]].content
    assert steer.startswith("round 2: ") and "the costliest first" in steer


# --- the Examiner's side of a round, driven by a model ----------------------------------------

@pytest.fixture(scope="module")
def model_examiner_loop(tmp_path_factory, request):
    """A model-driven Examiner across two beats under an allowance of zero: round 1 is the prompt and
    one derive, round 2 the steer and a continuation of the same session. The Builder's harness has
    its own scripted model so the round-1 Builder beat runs the model path too."""
    workdir = tmp_path_factory.mktemp("examiner-agent")
    builder_model = TestModel([_reply(None, ("build", {"target": TARGET})), _reply("built.")])
    examiner_model = TestModel([_reply(None, ("derive", {"target": "all"})), _reply("read."),
                                _reply(None, ("derive", {"target": "all"})), _reply("re-derived on round 2.")])
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, builder_model), target=TARGET,
                       agent_model=examiner_model, allowance_usd=0.0)
    loop.allowance = {agent: loop.allowance_for(agent) for agent in rounds.AGENTS}
    loop.builder_beat(1)
    loop.examiner_beat(1)
    after_first = len(loop.examiner.messages)
    loop.examiner_beat(2)
    return {"loop": loop, "after_first": after_first, "model": examiner_model}


def test_the_examiner_is_one_session_across_rounds_prompted_once_then_steered_and_continued(model_examiner_loop):
    """D128 and D123: the Examiner has its own session and keeps it across rounds. Round 1 is the one
    examiner message and a derive; round 2 is a `round 2:` steer and a re-derive on the same
    transcript, never a second prompt, and the session file holds one session_info root."""
    loop = model_examiner_loop["loop"]
    messages = loop.examiner.messages
    first, second = messages[:model_examiner_loop["after_first"]], messages[model_examiner_loop["after_first"]:]
    assert [m.role for m in first] == ["user", "assistant", "tool", "user", "assistant"]
    assert first[0].content == examiner_message()
    assert first[3].content == rounds.ALLOWANCE_STEER, "an allowance of zero is spent at the first tool end"
    assert [m.role for m in second] == ["user", "assistant", "tool", "user", "assistant"]
    assert second[0].content.startswith("round 2:") and second[0].details is None
    assert loop.examiner_result is not None and not loop.examiner_result.is_error, \
        "the round-2 beat derived successfully"
    assert [m.content for m in messages if isinstance(m, UserMessage)].count(examiner_message()) == 1
    lines = (loop.plan.workdir / rounds.EXAMINER_SESSION).read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in lines if json.loads(line).get("type") == "session_info") == 1


def test_the_round_n_examiner_steer_asks_for_derive_again(model_examiner_loop):
    """Build 9: round 2 was steered to `read the rulings and act`, the model read them, filed a
    finding and answered in one line, and the round failed for the derive the beat requires. The
    steer now asks for the call the beat checks for, in the words examiner/agent.py holds."""
    loop = model_examiner_loop["loop"]
    steer = loop.examiner.messages[model_examiner_loop["after_first"]]
    assert steer.content == examiner_round_message(2, rounds.EXAMINER_TARGET)
    assert f"call the derive tool with target={rounds.EXAMINER_TARGET!r}" in steer.content


def test_an_examiner_that_derives_in_round_two_does_not_fail_the_round(tmp_path, request):
    """The other half of the same bug: a round-2 beat that does what the steer asks (derive again,
    then file what the rulings call for and answer with one line) closes the beat. What failed the
    live round was the missing derive, not the finding or the one-line answer."""
    loop = _model_driven_beat(tmp_path, request, [
        _reply(None, ("derive", {"target": "all"})),
        _reply("derived; the rulings are read."),
        _reply(None, ("derive", {"target": "all"})),
        _reply(None, ("finding", {"task_id": "1", "kind": "fidelity",
                                  "text": "the replay diverges at the second call", "suggested": "replay"})),
        _reply("rulings remain blocked; filed the finding."),
    ], "derives-in-round-two")
    loop.examiner_beat(1)
    loop.examiner_beat(2)
    assert loop.examiner_result is not None and not loop.examiner_result.is_error
    queued = [f.finding_id for f in loop.pending_findings]
    assert [f.task_id for f in loop.pending_findings][-1] == "1", "the round-2 finding is queued, not lost"
    assert len(queued) == len(set(queued)), "and the rule findings each derive filed are queued once (D170)"
    assert loop.close_round(2, loop.counts()).failed is False


def test_the_examiner_plan_sees_its_allowance_shrink_at_every_tool_end(model_examiner_loop):
    """rounds.py hands the Examiner's plan what is left of the round's allowance after each tool, so the
    reroll tool refuses at the right moment; under an allowance of zero nothing is left after derive."""
    loop = model_examiner_loop["loop"]
    assert loop.eplan.allowance_remaining is not None and loop.eplan.allowance_remaining <= 0
    assert loop.spent_allowance["examiner"] is True


# --- which models judged the build (D160) -------------------------------------------------

def test_a_build_records_the_judge_models_only_when_one_was_named(tmp_path):
    """The report names the judge models beside the build model; a build that judges with its own
    model records nothing, so its files are what they were before this."""
    own = BuildPlan(workdir=tmp_path / "own", model=TestModel(["hi"], name="vendor/large"))
    rounds._record_judge_models(own)
    assert not (own.workdir / "report_config.json").exists()

    named = BuildPlan(workdir=tmp_path / "named", model=TestModel(["hi"], name="vendor/large"),
                      judge_model=TestModel(["hi"], name="other/small"),
                      second_judge_model=TestModel(["hi"], name="third/tiny"))
    (named.workdir / "report_config.json").write_text(json.dumps({"audit_rate": 0.5}), encoding="utf-8")
    rounds._record_judge_models(named)
    body = json.loads((named.workdir / "report_config.json").read_text(encoding="utf-8"))
    assert body["judge_models"] == {"build": "vendor/large", "judge": "other/small",
                                    "second_judge": "third/tiny"}
    assert body["audit_rate"] == 0.5, "what the file already said is kept"


# --- the allowance and the exits, decided by the driver -----------------------------------

def test_the_allowance_defaults_to_round_ones_spend_per_agent(tmp_path):
    loop = _bare_loop(tmp_path)
    assert loop.allowance_for("builder") is None and loop.allowance_for("examiner") is None
    loop.rounds = [RoundRecord(round=1, counts={"spend": {"builder": 0.25, "examiner": 0.0, "total": 0.25}})]
    assert loop.allowance_for("builder") == 0.25
    assert loop.allowance_for("examiner") is None, "a round-1 spend of zero means no allowance"
    given = _bare_loop(tmp_path, allowance_usd=0.5)
    given.rounds = list(loop.rounds)
    assert (given.allowance_for("builder"), given.allowance_for("examiner")) == (0.5, 0.5)


def _record(n: int, **counts) -> RoundRecord:
    base = {"fidelity": 1, "tasks": 2, "trusted": 0, "refused_count": 0, "assisted_runs": 0,
            "probes_passing": 0, "unfinished": ["t2"], "spend": {"builder": 0.0, "examiner": 0.0, "total": 0.0}}
    base.update(counts)
    return RoundRecord(round=n, counts=base)


def _exit_after(loop: rounds.Loop, records: list[RoundRecord], spent: tuple[bool, ...] = ()) -> list:
    """The exit the driver puts on each record through its own `close_round`, rounds closed in order;
    `spent` says per round whether an agent spent its allowance, what the beats would have marked."""
    exits = []
    for number, record in enumerate(records):
        loop.spent_allowance = {"builder": bool(spent[number]) if number < len(spent) else False}
        exits.append(loop.close_round(record.round, record.counts).exit)
    return exits


def test_the_loop_exits_stalled_after_stall_rounds_rounds_that_moved_nothing(tmp_path):
    loop = _bare_loop(tmp_path)
    assert _exit_after(loop, [_record(1), _record(2)]) == [None, "stalled"]
    assert [r.exit for r in rounds.load_rounds(loop.plan.workdir)] == [None, "stalled"]
    moved = _bare_loop(tmp_path)
    assert _exit_after(moved, [_record(1), _record(2, trusted=1), _record(3, trusted=1)]) == [None, None, "stalled"]


def test_stall_rounds_two_waits_one_more_round(tmp_path):
    loop = _bare_loop(tmp_path, stall_rounds=2)
    assert _exit_after(loop, [_record(1), _record(2), _record(3)]) == [None, None, "stalled"]


def test_the_loop_exits_ceiling_when_the_builder_stopped_at_the_spend_ceiling(tmp_path):
    """A Builder result carrying `stopped` (the pipeline's BudgetStop report) is the ceiling exit,
    ahead of done and stalled, and so is the Examiner's plan saying it reached the ceiling."""
    loop = _bare_loop(tmp_path)
    loop.plan.last = pipeline.PipelineResult(status="stopped", stopped={"stage": "rerolls", "reason": "spend ceiling"})
    assert _exit_after(loop, [_record(1, unfinished=[])]) == ["ceiling"]
    other = _bare_loop(tmp_path)
    other.eplan = ExaminerPlan(workdir=tmp_path / "examiner", inputs={})
    other.eplan.ceiling_reached = True
    assert _exit_after(other, [_record(1, unfinished=[])]) == ["ceiling"]


# --- the soft stop and the hard stop of a session with no turn cap (D135) -----------------

def _stalling_loop(tmp_path: Path, replies: list) -> rounds.Loop:
    """A model-driven Loop whose Builder has already had one run, so the round can be told it stalled."""
    plan = BuildPlan(workdir=tmp_path / "work")
    model = TestModel(replies)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, agent_model=model),
                       agent_model=model, target=TARGET)
    _collect(loop.builder.prompt(builder_agent.builder_message(TARGET)))
    loop.rounds = [_record(1)]
    return loop


def test_a_builder_that_repairs_and_answers_without_building_has_the_target_built_by_the_driver(tmp_path, request):
    """Build 13, round 1: the model zoomed, repaired and answered with no build call, so the store held
    only what the repairs ran and the Examiner's derive failed on a table that was never loaded. The
    driver now builds the target itself in that case, as the code path does, and the round says so."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    model = TestModel([_reply("nothing to repair; finishing.")])
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, agent_model=model),
                       agent_model=model, target=TARGET)
    loop.plan.round = 1
    loop.builder_beat(1)
    assert loop.build_result is not None and not loop.build_result.is_error
    assert TARGET in plan.store or plan.last is not None, "the target was built, by the driver"
    assert loop.driver_built == [1] and loop.driver_counts()["built_by_driver"] is True


def test_a_builder_that_repairs_after_building_has_the_target_built_again_by_the_driver(tmp_path, request):
    """Build 13, round 2: the model built the target, then acted on a follow-up finding with a narrowed
    recompile and answered. The store then held only that one stage's artifacts and the Examiner's
    derive failed on the Constraints again. The driver notices the last run was narrowed and builds
    the target once more, so the handover holds everything the derivation reads."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    model = TestModel([_reply("", ("build", {"target": TARGET})),
                       _reply("", ("repair_recompile", {"name": "get_order_details", "hint": "return the row"})),
                       _reply("repaired; finishing.")])
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, agent_model=model),
                       agent_model=model, target=TARGET)
    loop.plan.round = 1
    loop.builder_beat(1)
    assert loop.build_result is not None and not loop.build_result.is_error
    assert plan.last_narrowing == {} and plan.last_target == TARGET, "the last run was the target in full"
    assert "constraints" in plan.store, "the artifact the Examiner's derive reads is in the handover"
    assert loop.driver_built == [1] and loop.driver_counts()["built_by_driver"] is True


def test_a_round_that_moved_no_gate_count_tells_the_builder_once_and_ends_if_it_stops_again(tmp_path):
    """D126 as the soft stop of a Builder with no turn cap: the round that changed nothing sends one
    follow-up asking it to finish with what it has, and a model that stops again ends the round."""
    loop = _stalling_loop(tmp_path, [_reply("nothing to do."), _reply("finishing with what I have.")])
    assert loop.moved_nothing(_record(2).counts) is True
    loop.tell_the_builder_nothing_changed(2)
    said = [m["content"] for m in loop.agent_model.calls[-1]["messages"] if m["role"] == "user"]
    assert sum(1 for line in said if line.startswith(rounds.STALL_FOLLOW_UP)) == 1
    assert len(loop.agent_model.calls) == 2, "the model was asked once more, and it stopped again"
    loop.tell_the_builder_nothing_changed(2)
    assert len(loop.agent_model.calls) == 2, "one follow-up a round, however often the driver asks"


def test_a_builder_round_with_every_stage_cached_is_told_that_nothing_changed_and_which_verbs_change_something(
        tmp_path, request):
    """Build 9's round 2: the findings came in, the model answered each with replay and build, every
    stage was served from the cache and the round moved no count. A cached build reads like any
    other, so the tool result now says which it was in its first line, and the follow-up says the
    same in words and names the verbs of this session that can change an artifact."""
    plan = BuildPlan(workdir=tmp_path / "cached", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    harness = builder_agent.build_harness(plan)
    for _ in range(2):  # the second build re-ingests the file; the third has nothing left to run
        builder_agent.drive_tool(harness, "build", {"target": TARGET})
    third = builder_agent.drive_tool(harness, "build", {"target": TARGET})
    assert not third.is_error and "nothing changed: all" in third.content.splitlines()[0]
    assert all(stage["cached"] for stage in third.details["stages"] if stage["status"] != "pending")

    loop = _stalling_loop(tmp_path, [_reply("nothing to do."), _reply("finishing with what I have.")])
    loop.tell_the_builder_nothing_changed(2)
    told = [m["content"] for m in loop.agent_model.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert told.startswith(rounds.STALL_FOLLOW_UP)
    assert "without a repair returns that same cached result" in told
    verbs = [tool.name for tool in repair_verb_tools(loop.plan)]
    assert "repair_recompile" in verbs and "hint" in told, "the recompile is offered with its hint"
    assert all(f"{name} (" in told for name in verbs)
    assert "build (" not in told, "build is what was served from the cache, not a way out of it"


# --- what a round changed: the artifacts and the repairs, beside the gate counts ---------

def _write(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_a_round_that_rewrote_a_tool_body_is_not_stalled_even_when_no_count_moved(tmp_path):
    """The founder's point: the mechanic repaired an artifact, so the Examiner has not derived over
    the new body yet and no gate count could have moved for it. That round moved and is not counted
    toward the stall; the round after it, which rewrites nothing, is."""
    loop = _bare_loop(tmp_path)
    workdir = loop.plan.workdir
    _write(workdir / "bodies.json", {"get_order": "def get_order(db, order_id): return None"})
    assert loop.close_round(1, _record(1).counts).exit is None
    _write(workdir / "bodies.json", {"get_order": "def get_order(db, order_id): return db['orders']"})
    second = loop.close_round(2, _record(2).counts)
    assert second.counts["artifacts_changed"] == ["bodies"]
    assert second.counts["moved"] is True
    assert second.exit is None, "the counts are the round before's, but a tool body was rewritten"
    third = loop.close_round(3, _record(3).counts)
    assert third.counts["artifacts_changed"] == [] and third.counts["moved"] is False
    assert third.exit == "stalled", "nothing moved in this one, and stall_rounds is one"


def test_a_round_with_no_count_and_no_artifact_change_is_stalled(tmp_path):
    """The plain stall, unchanged by any of this: the counts stand still, every artifact hashes the
    same as the round before, and the round is one the stall counts."""
    loop = _bare_loop(tmp_path)
    _write(loop.plan.workdir / "bodies.json", {"get_order": "def get_order(): return None"})
    assert _exit_after(loop, [_record(1), _record(2)]) == [None, "stalled"]
    stored = rounds.load_rounds(loop.plan.workdir)
    assert [record.counts["moved"] for record in stored] == [True, False]
    assert stored[1].counts["artifacts_changed"] == []
    assert stored[1].counts["artifacts"] == stored[0].counts["artifacts"] != ""
    assert set(stored[1].counts["artifact_hashes"]) == set(rounds.MODEL_ARTIFACTS)


def test_a_round_whose_repair_moved_a_gate_ruling_is_not_stalled(tmp_path):
    """The third way a round moves: a repair whose round left the gates ruling differently from the
    round before, which holds even where the artifact it rewrote came out byte for byte the same."""
    loop = _bare_loop(tmp_path)
    workdir = loop.plan.workdir
    ledger = GateLedger(workdir)
    ledger.record("compile_tools", GateResult(stage="compile_tools", **{"pass": False}))
    assert loop.close_round(1, _record(1).counts).exit is None
    repair.record_request(workdir, "repair_recompile", "get_order", {}, round_no=2)
    ledger.record("compile_tools", GateResult(stage="compile_tools", **{"pass": True}))
    second = loop.close_round(2, _record(2).counts)
    assert second.counts["moved"] is True and second.exit is None
    assert second.counts["repairs"] == [{"verb": "repair_recompile", "target": "get_order",
                                         "artifact": "bodies", "changed": False}]
    assert loop.close_round(3, _record(3).counts).exit == "stalled", "no repair, no ruling, no count"


def test_a_round_whose_only_repair_decided_something_is_stalled(tmp_path):
    """`repair_refuse_task` and `repair_escalate` write a row for the report and change no artifact,
    which their own descriptions say. A round that called nothing else moved nothing."""
    loop = _bare_loop(tmp_path)
    assert loop.close_round(1, _record(1).counts).exit is None
    repair.record_request(loop.plan.workdir, "repair_refuse_task", "t2", {}, round_no=2)
    second = loop.close_round(2, _record(2).counts)
    assert second.counts["repairs"] == [{"verb": "repair_refuse_task", "target": "t2",
                                         "artifact": None, "changed": False}]
    assert second.counts["moved"] is False and second.exit == "stalled"


def test_round_one_compares_the_artifacts_against_the_fingerprint_the_driver_started_with(tmp_path):
    """A first round had no round before it to compare with, so it reported that nothing changed
    however much it wrote, and every repair of round 1 read as having changed nothing."""
    loop = _bare_loop(tmp_path)
    assert loop.started_hashes == rounds.artifact_hashes(loop.plan.workdir)
    _write(loop.plan.workdir / "bodies.json", {"renew_loan": "def renew_loan(db): return db"})
    first = loop.close_round(1, _record(1).counts)
    assert first.counts["artifacts_changed"] == ["bodies"]
    assert loop.close_round(2, _record(2).counts).counts["artifacts_changed"] == []


def test_a_first_round_over_a_workdir_that_already_held_its_artifacts_changed_none_of_them(tmp_path):
    """The other half of the same rule: `--iterate` starts on a workdir that already holds bodies
    and Intents, and a round that rewrote none of them says so."""
    plan = BuildPlan(workdir=tmp_path / "work")
    _write(plan.workdir / "bodies.json", {"renew_loan": "def renew_loan(db): return db"})
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    assert loop.close_round(1, _record(1).counts).counts["artifacts_changed"] == []


def test_a_round_records_the_change_each_repair_measured_on_its_own_target(tmp_path):
    """Six Intents written again and a seventh still refused change `intents` once, so the round's
    own fingerprint cannot tell the seven repairs apart; each verb hashed the one file it wrote."""
    loop = _bare_loop(tmp_path)
    workdir = loop.plan.workdir
    repair.record_request(workdir, "repair_intent", "task_a",
                          {"changed": True, "hash_before": "aaa", "hash_after": "bbb"}, round_no=1)
    repair.record_request(workdir, "repair_intent", "task_b",
                          {"changed": False, "hash_before": "ccc", "hash_after": "ccc"}, round_no=1)
    assert loop.close_round(1, _record(1).counts).counts["repairs"] == [
        {"verb": "repair_intent", "target": "task_a", "artifact": "intents", "changed": True},
        {"verb": "repair_intent", "target": "task_b", "artifact": "intents", "changed": False}]


def test_the_stall_follow_up_names_the_pending_findings_and_the_repairs_made(tmp_path):
    """What is pending is what the model cannot read out of its own transcript: a finding no beat
    acted on, and whether each repair it called actually changed the artifact that verb owns. Each
    repair answers for its own target, which is what its request recorded when the verb ran."""
    loop = _stalling_loop(tmp_path, [_reply("nothing to do."), _reply("finishing with what I have.")])
    workdir = loop.plan.workdir
    loop.pending_findings = [_finding("t2", suggested="compile_tool", finding_id="f7")]
    repair.record_request(workdir, "repair_recompile", "get_order", {"changed": True}, round_no=2)
    repair.record_request(workdir, "repair_intent", "t2", {"changed": False}, round_no=2)
    repair.record_request(workdir, "repair_refuse_task", "t3", {}, round_no=2)
    loop.tell_the_builder_nothing_changed(2)

    told = [m["content"] for m in loop.agent_model.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert told.startswith(rounds.STALL_FOLLOW_UP)
    assert "1 finding(s) not acted on (f7 suggests compile_tool)" in told
    assert "repair_recompile on get_order changed bodies" in told
    assert "repair_intent on t2 left intents as it was" in told
    assert "repair_refuse_task on t3 changed no artifact" in told
    assert "What changes an artifact is a repair verb" in told, "the verbs are still named"


def test_the_stall_follow_up_says_so_plainly_when_nothing_is_pending(tmp_path):
    loop = _stalling_loop(tmp_path, [_reply("nothing to do."), _reply("finishing with what I have.")])
    loop.tell_the_builder_nothing_changed(2)
    told = [m["content"] for m in loop.agent_model.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "Pending: no finding is open and this round called no repair verb." in told


def test_a_round_that_moved_a_gate_count_tells_the_builder_nothing(tmp_path):
    loop = _stalling_loop(tmp_path, [_reply("nothing to do.")])
    assert loop.moved_nothing(_record(2, trusted=1).counts) is False
    fresh = _bare_loop(tmp_path / "fresh")
    assert fresh.moved_nothing(_record(1).counts) is False, "round one has nothing to compare with"


def test_the_code_driven_builder_is_never_told_anything_because_it_has_no_model(tmp_path):
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan), target=TARGET)
    loop.rounds = [_record(1)]
    loop.tell_the_builder_nothing_changed(2)
    assert loop.stall_told == 0 and loop.builder.messages == ()


def test_the_ceiling_the_guard_caught_is_the_rounds_ceiling_exit(tmp_path):
    """The session's guard cancels the run and hands the stop to the round, which is a ceiling exit
    even though the pipeline result on the plan carries none."""
    loop = _bare_loop(tmp_path)
    assert loop.ceiling_reached() is False
    loop.builder_stop = {"stage": "builder_session", "reason": "spend ceiling", "ceiling_usd": 1.0}
    assert loop.ceiling_reached() is True
    assert _exit_after(loop, [_record(1, unfinished=[])]) == ["ceiling"]


class Priced(Bodies):
    """The Builder's scripted model with tokens on every reply, so a ceiling in dollars is reachable."""

    def query(self, messages, tools=None, config=None):
        from kullback.ai.usage import Usage

        return super().query(messages, tools=tools, config=config).model_copy(
            update={"usage": Usage(input=1000)})


def test_a_run_whose_builder_reached_the_ceiling_exits_ceiling_and_is_not_a_failed_round(tmp_path, request,
                                                                                         monkeypatch):
    """D86 with no turn cap: one call costs more than the ceiling, so the build stops, the Examiner
    has nothing whole to derive from, and the run ends on the ceiling rather than on a broken agent."""
    monkeypatch.setitem(budget.PRICES, "test", {"input": 1_000_000.0, "output": 0.0,
                                                "cache_read": 0.0, "cache_write": 0.0})
    workdir = tmp_path / "ceiling"
    agent_model = TestModel([_reply(None, ("build", {"target": TARGET})), _reply("built."),
                             _reply("I have read everything and all is well.")])
    result = rounds.run_rounds(workdir, model=Priced(), agent_model=agent_model, files=[_fixture(request)],
                               max_attempts=0, ceiling_usd=1.0)
    assert result["exit"] == "ceiling"
    stored = rounds.load_rounds(workdir)
    assert len(stored) == 1 and stored[-1].exit == "ceiling"
    assert stored[-1].failed is False, "no money left is a stop, not a failure"


def test_the_loop_exits_ceiling_when_the_allowance_was_exhausted_two_rounds_in_a_row(tmp_path):
    loop = _bare_loop(tmp_path)
    assert _exit_after(loop, [_record(1), _record(2, trusted=1)], spent=(True, True)) == [None, "ceiling"]
    assert loop.exhausted == [True, True]
    once = _bare_loop(tmp_path)
    assert _exit_after(once, [_record(1), _record(2, trusted=1), _record(3, trusted=2)],
                       spent=(True, False, True)) == [None, None, None]
    assert once.exhausted == [True, False, True]


def test_rounds_json_holds_one_record_per_round_with_the_exit_on_the_last(tmp_path):
    records = [_record(1), _record(2, trusted=1), _record(3, trusted=1, unfinished=[])]
    records[-1].exit = "done"
    path = rounds.write_rounds(tmp_path, records)
    assert path == tmp_path / rounds.ROUNDS_NAME
    body = json.loads(path.read_text(encoding="utf-8"))
    assert [row["round"] for row in body] == [1, 2, 3]
    assert [row["exit"] for row in body] == [None, None, "done"]
    assert rounds.load_rounds(tmp_path) == records
    assert rounds.load_rounds(tmp_path / "nowhere") == []


# --- whole rounds, Builder then Examiner --------------------------------------------------

@pytest.fixture(scope="module")
def driven(tmp_path_factory, request):
    """run_rounds over the fixture with both agents driven by code, every event on one list."""
    workdir = tmp_path_factory.mktemp("rounds")
    events, dicts, builder_rows = [], [], []

    def after_the_builder_beat(event):
        if isinstance(event, BeatEnd) and event.agent == "builder":
            builder_rows[:] = json.loads((workdir / "gates.json").read_text(encoding="utf-8"))

    # One round: the fixture's Tasks never get a Reference, so under D172 the loop would run on to a
    # stall; the round cap (D169) ends it after the one round these tests read.
    result = rounds.run_rounds(workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0, max_rounds=1,
                               subscribers=[events.append, after_the_builder_beat], on_event=dicts.append)
    return {"workdir": workdir, "result": result, "events": events, "dicts": dicts, "builder_rows": builder_rows}


def test_a_round_is_a_builder_beat_then_an_examiner_beat_on_one_stream_in_that_order(driven):
    events = driven["events"]
    kinds = [e.type for e in events]
    assert kinds[0] == "round_start" and kinds[-1] == "round_end"
    assert _beats(events)[:4] == [("beat_start", "builder", 1), ("beat_end", "builder", 1),
                                  ("beat_start", "examiner", 1), ("beat_end", "examiner", 1)]
    marks = [i for i, e in enumerate(events) if isinstance(e, (BeatStart, BeatEnd, RoundStart, RoundEnd))]
    builder = events[marks[1] + 1:marks[2]]
    examiner = events[marks[3] + 1:marks[4]]
    assert [e.tool_name for e in builder if isinstance(e, ToolExecutionEnd)] == ["build"]
    assert [e.tool_name for e in examiner if isinstance(e, ToolExecutionEnd)] == ["derive"]
    assert not [e for e in examiner if getattr(e, "tool_name", None) == "build"]


def test_round_end_carries_every_count_d126_lists_and_none_comes_from_a_model(driven):
    """Every count on round_end is one gates.round_end computed off the rulings and the records, plus
    the three the driver alone knows; the same counts are the ones rounds.json holds."""
    end = [e for e in driven["events"] if isinstance(e, RoundEnd)][-1]
    counts = end.counts
    assert set(counts) >= {"fidelity", "tasks", "tasks_with_reference", "trusted", "trusted_ids", "refused",
                           "refused_count", "assisted_runs", "probes_passing", "false_rejection", "unfinished",
                           "fallback_compactions", "spend", "findings"}
    assert set(round_end.GATE_COUNTS) <= set(counts)
    assert counts["tasks"] == len(driven["result"]["tasks"]) == 3
    assert counts["fallback_compactions"] == {"builder": 0, "examiner": 0}
    assert set(counts["spend"]) == {"builder", "examiner", "total", "cache_saved"}
    # No model filed any of these: they are the losses the round's own records show (D170), and the
    # count is the list of ids because that is what the next round's Builder beat is handed.
    assert counts["findings"] and all(f.startswith("finding-") for f in counts["findings"])
    assert rounds.load_rounds(driven["workdir"])[-1].counts == counts
    assert [d for d in driven["dicts"] if d.get("kind") == "round"][-1]["counts"] == counts


def test_a_rounds_counts_carry_its_clock_its_spend_its_turns_and_its_context_fill(driven):
    """A build's duration is read from these and from nothing else: pipeline/state.json records the
    stage statuses and no clock, and a repair's timestamp has no round to sit against without them."""
    counts = rounds.load_rounds(driven["workdir"])[-1].counts
    assert counts["started_at"] > 0 and counts["ended_at"] >= counts["started_at"]
    assert counts["ended_at"] - counts["started_at"] < 3600
    assert counts["turns"] == {"builder": 0, "examiner": 0, "total": 0}, "the code driver takes no turn"
    assert counts["context_fill"] == {"builder": 0.0, "examiner": 0.0}
    assert set(counts["spend"]) == {"builder", "examiner", "total", "cache_saved"}


def test_every_rounds_gate_rulings_are_kept_beside_gates_json_round_by_round(driven):
    """gates.json holds the last ruling per stage, so the next round overwrites it; the per-round
    rows are what lets a repair be read against the rulings before and after its round."""
    workdir = driven["workdir"]
    history = json.loads((workdir / "gates_by_round.json").read_text(encoding="utf-8"))
    assert [row["round"] for row in history] == [r["round"] for r in driven["result"]["rounds"]]
    assert history[-1]["rulings"] == json.loads((workdir / "gates.json").read_text(encoding="utf-8"))
    assert any(not ruling["pass"] for ruling in history[-1]["rulings"]), "the fixture leaves red gates"


def test_the_driver_counts_say_the_target_was_not_built_when_the_last_run_failed_a_stage(tmp_path):
    """D166: the counts off the gates cannot see a build that never finished, so the driver says it.
    A round with no run at all, and a round whose run failed a stage, both report built False."""
    loop = _bare_loop(tmp_path)
    assert loop.plan.last is None and loop.driver_counts()["built"] is False
    loop.plan.last = pipeline.PipelineResult(status="failed", failed_stage="compile_tools")
    assert loop.driver_counts()["built"] is False
    loop.plan.last = pipeline.PipelineResult(status="ok")
    assert loop.driver_counts()["built"] is True


def test_the_driver_counts_the_turns_of_the_beat_alone_and_how_full_the_context_got(tmp_path):
    """The turns and the fill are the harness's own counters (D131), read as a delta over the beat,
    so round 2 reports the turns round 2 took and not the turns of the session so far."""
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(
        plan, TestModel([_reply("built."), _reply("built again.")])))
    loop.turns_seen = {agent: len(loop.fills(agent)) for agent in rounds.AGENTS}
    _collect(loop.builder.prompt("build"))
    first = loop.driver_counts()
    assert first["turns"] == {"builder": 1, "examiner": 0, "total": 1}
    assert 0.0 < first["context_fill"]["builder"] <= 1.0
    assert first["context_fill"]["examiner"] == 0.0, "no Examiner harness took a turn"
    loop.turns_seen = {agent: len(loop.fills(agent)) for agent in rounds.AGENTS}
    _collect(loop.builder.prompt("build again"))
    assert loop.driver_counts()["turns"] == {"builder": 1, "examiner": 0, "total": 1}
    assert len(loop.fills("builder")) == 2, "the session kept both turns; the beat counted one"


def test_a_round_that_failed_still_records_its_clock_its_spend_and_its_turns(tmp_path, request):
    """The round that broke is the one a reader most wants the numbers of, and it closes with no
    gate counts at all, so the driver's own numbers are put on the record there too."""
    workdir = tmp_path / "broken-examiner"
    agent_model = TestModel([
        _reply(None, ("build", {"target": TARGET})),
        _reply("built."),
        _reply("I have read everything and all is well."),
    ])
    result = rounds.run_rounds(workdir, model=Bodies(), agent_model=agent_model, files=[_fixture(request)],
                               max_attempts=0, allowance_usd=0.0)
    assert result["failed"] is True
    counts = rounds.load_rounds(workdir)[-1].counts
    assert counts["started_at"] > 0 and counts["ended_at"] >= counts["started_at"]
    assert counts["turns"]["builder"] == 2 and counts["turns"]["total"] == counts["turns"]["builder"] + \
        counts["turns"]["examiner"]
    assert counts["context_fill"]["builder"] > 0.0
    assert set(counts["spend"]) == {"builder", "examiner", "total", "cache_saved"}
    assert "fidelity" not in counts, "a round that failed computed no gate count"
    assert json.loads((workdir / "gates_by_round.json").read_text(encoding="utf-8"))[-1]["round"] == 1


def test_the_round_the_driver_is_in_is_on_the_plan_before_the_first_beat(tmp_path):
    """A repair verb is registered once and the round moves under it, so the round a request records
    is the one the plan carries when the verb is called (repairs/*.jsonl round). The round driver is
    what moves it, and it moves it before the Builder beat that may call a verb."""
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    verbs = {tool.name: tool for tool in repair_verb_tools(plan)}
    assert plan.round == 1, "a build with no round driver is one pass"
    for n in (1, 2, 3):
        with pytest.raises(BuildError):  # nothing to build in this workdir; the round is set first
            loop.round(n)
        assert plan.round == n
        asyncio.run(verbs["repair_escalate"].run({"task_id": f"t{n}"}))
    rows = [json.loads(line) for line in
            (plan.workdir / "repairs" / "repair_escalate.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["target"], row["round"]) for row in rows] == [("t1", 1), ("t2", 2), ("t3", 3)]


def test_the_loop_over_the_fixture_is_not_done_after_a_round_and_stops_on_its_round_cap(driven):
    """No Task on the fixture has a confirmed Reference: before D172 that read as D126's state and
    the driver exited done at fidelity 0; now those Tasks are the loop's unfinished work, and the
    fixture's one-round cap is what ends it."""
    assert driven["result"]["exit"] == "max_rounds"
    assert [r["round"] for r in driven["result"]["rounds"]] == [1]
    assert driven["events"][-1].exit == "max_rounds"


def test_the_result_carries_the_build_result_the_rounds_the_trusted_tasks_and_the_refusals(driven):
    result = driven["result"]
    assert set(result) >= {"status", "workdir", "env_id", "failed_stage", "stopped", "gates", "tasks", "target",
                           "rulings", "tool_result", "rounds", "exit", "trusted", "refused", "examiner"}
    assert result["status"] == "complete" and result["target"] == TARGET and result["env_id"]
    assert result["tool_result"] == {"content": result["tool_result"]["content"], "is_error": False}
    assert result["trusted"] == [] and result["refused"] == {}
    assert set(result["examiner"]) == {"rulings", "tool_result"}
    assert result["examiner"]["tool_result"]["is_error"] is False
    assert "derive_verifier" in result["examiner"]["rulings"] or result["examiner"]["rulings"]


def test_the_examiner_receives_the_builder_artifacts_without_bodies_db_schema_or_environment(tmp_path, request):
    """D123: the Examiner never reads tool bodies or the Environment; what it is handed is DERIVE_INPUTS."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0,
                     workers=2)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.builder_beat(1)
    assert {"bodies", "db", "schema", "environment"} <= set(plan.store)
    loop.examiner_beat(1)
    assert loop.eplan.workers == plan.workers, "the derivation derives on the build's workers (D163)"
    handed = set(loop.eplan.store) & set(plan.store)
    assert handed <= set(DERIVE_INPUTS)
    assert not handed & set(FORBIDDEN_INPUTS)
    assert loop.eplan.env_id == plan.store["environment"].env_id


def _closed_over(fn, value) -> bool:
    return any(cell.cell_contents is value for cell in (fn.__closure__ or ()))


def test_the_examiner_runs_in_the_environment_the_builder_left_at_its_latest_beat(tmp_path, request):
    """D120: the Runner callables the Examiner probes and re-rolls through are built over the Builder's
    store at every Examiner beat, not once at round 1, so a rebuilt Starting state or body is what
    the next probe runs in."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.builder_beat(1)
    loop.examiner_beat(1)
    first_db, first_rerolls = plan.store["db"], loop.eplan.run_rerolls
    assert _closed_over(first_rerolls, first_db)
    loop.builder_beat(2)
    loop.examiner_beat(2)
    assert loop.eplan.env_id == plan.store["environment"].env_id
    assert loop.eplan.run_rerolls is not first_rerolls
    assert _closed_over(loop.eplan.run_rerolls, plan.store["db"])
    assert _closed_over(loop.eplan.run_probe, plan.store["db"])


def test_the_code_driven_rounds_over_the_fixture_leave_the_task_status_the_single_pipeline_wrote(driven):
    """D130: the derivation moved to the Examiner without changing a byte of what it writes. The three
    rows of the fixture and their reasons are pinned here so CI holds the claim without the snapshot."""
    path = driven["workdir"] / "task_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    assert len(status) == 3
    assert all(row.get("reference_confirmed") is False for row in status.values())
    assert all(row.get("verifier_passed") is False for row in status.values())
    assert all(row.get("reason") for row in status.values())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TASK_STATUS_SHA256_BEFORE_THE_PHASE
    assert not list((driven["workdir"] / "verifiers").glob("*.json"))
    assert driven["result"]["rounds"][-1]["counts"]["tasks_with_reference"] == 0


def test_the_examiner_appends_its_round_end_rulings_after_the_builders_rows_and_moves_none(driven):
    """D130 and D128: one ledger written in turn. gates.json after the round is the Builder's rows as its
    beat left them, in their order, then the Examiner's derive rulings, then the one `trusted` row
    round_end adds. The one row the derivation re-records is compile_policy, with the same content, the
    way the derive_verifier stage did when it was the Builder's (the ledger replaces a ruling of the
    same stage name), so the file reads as the single pipeline wrote it; the fidelity ruling round_end
    recomputes equals the Builder's replay_reference row and is not written a second time."""
    rows = json.loads((driven["workdir"] / "gates.json").read_text(encoding="utf-8"))
    builder_rows = driven["builder_rows"]
    kept = [r for r in builder_rows if r["stage"] != "compile_policy"]
    assert kept and rows[:len(kept)] == kept
    assert [r["stage"] for r in kept][-2:] == ["replay_reference", "rerolls"]
    assert [r["stage"] for r in rows[len(kept):]] == ["compile_policy", "derive_verifier", "trusted"]
    assert rows[len(kept)] == next(r for r in builder_rows if r["stage"] == "compile_policy")
    assert [r["stage"] for r in rows].count("replay_reference") == 1


def test_a_delivered_finding_is_closed_and_its_entry_unprotected_after_the_builder_beat(tmp_path, request):
    """A finding the Examiner filed is closed once the Builder's next beat delivered it, and the
    Examiner's session entry it protected is released (D124, D131)."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.builder_beat(1)
    loop.examiner_beat(1)
    task_id = plan.last.artifacts["tasks"][0].id
    filed = examiner_agent.drive_tool(loop.examiner, "finding", {
        "task_id": task_id, "kind": "fidelity", "text": "the replay diverges at the second call",
        "suggested": "replay"})
    assert filed.is_error is False, filed.content
    # The beat's own derive filed the losses its records show first (D170); the model's is the last.
    assert [f.finding_id for f in loop.pending_findings][-1] == filed.details["finding"]["finding_id"]
    loop.builder_beat(2)
    findings = json.loads((plan.workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))
    assert {f["status"] for f in findings} == {"closed"}, "the rule findings close with the model's"
    assert loop.pending_findings == []


# --- terminal rounds keep their findings, failed derives fail the round ------------------

def test_close_round_with_findings_pending_clears_a_done_exit_and_continues(tmp_path):
    """Greptile P1 (round 2): a round that reaches done with findings pending does not exit. The
    findings owe the Builder a beat, so the exit is cleared, the findings ride on the record in
    rounds.json, and the loop runs another round instead of reporting done with work open."""
    loop = _bare_loop(tmp_path)
    loop.plan.last = pipeline.PipelineResult(status="ok")  # the target was built, so done is reachable (D166)
    loop.pending_findings = [_finding("t1")]
    record = loop.close_round(1, _record(1, unfinished=[]).counts)
    assert record.exit is None
    assert "owe the Builder a beat" in (record.exit_note or "")
    assert [f.finding_id for f in record.pending_findings] == ["f1"]
    body = json.loads((tmp_path / "work" / rounds.ROUNDS_NAME).read_text(encoding="utf-8"))
    assert body[0]["exit"] is None
    assert body[0]["pending_findings"][0]["finding_id"] == "f1"
    assert rounds.load_rounds(tmp_path / "work")[0].pending_findings[0].finding_id == "f1"


def test_close_round_with_findings_pending_clears_a_stalled_exit(tmp_path):
    """Two rounds that move nothing would stall, but with a finding pending the second round
    continues: the finding may be the way out of the stall, and the Builder has not acted yet."""
    loop = _bare_loop(tmp_path)
    loop.pending_findings = [_finding("t1")]
    assert loop.close_round(1, _record(1).counts).exit is None
    second = loop.close_round(2, _record(2).counts)
    assert second.exit is None
    assert "owe the Builder a beat" in (second.exit_note or "")


def test_close_round_keeps_a_ceiling_exit_with_findings_pending(tmp_path):
    """Only the ceiling ends a round with findings still open: no money left means no next beat
    to owe, and the note says the findings never got one."""
    loop = _bare_loop(tmp_path)
    loop.ceiling_reached = lambda: True
    loop.pending_findings = [_finding("t1")]
    record = loop.close_round(1, _record(1, unfinished=[]).counts)
    assert record.exit == "ceiling"
    assert "ceiling ended the run first" in (record.exit_note or "")
    assert [f.finding_id for f in record.pending_findings] == ["f1"]


def test_close_round_persists_pending_findings_without_an_exit(tmp_path):
    """Findings pending in a round that does not exit are still on the record, for the next round."""
    loop = _bare_loop(tmp_path)
    loop.pending_findings = [_finding("t1")]
    record = loop.close_round(1, _record(1).counts)
    assert record.exit is None
    assert [f.finding_id for f in record.pending_findings] == ["f1"]


def _model_driven_beat(tmp_path, request, examiner_replies, name):
    """One Builder beat by code over the fixture, then the Examiner's beat on scripted replies."""
    workdir = tmp_path / name
    builder_model = TestModel([_reply(None, ("build", {"target": TARGET})), _reply("built.")])
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan, builder_model), target=TARGET,
                       agent_model=TestModel(examiner_replies))
    loop.allowance = {agent: loop.allowance_for(agent) for agent in rounds.AGENTS}
    loop.builder_beat(1)
    return loop


def test_an_examiner_that_never_derives_fails_the_round(tmp_path, request):
    """Greptile P1: a model that chats without calling derive must fail the round, never close it."""
    from kullback.examiner.agent import ExaminerError

    loop = _model_driven_beat(tmp_path, request, [_reply("I have read everything and all is well.")], "no-derive")
    with pytest.raises(ExaminerError, match="never called derive"):
        loop.examiner_beat(1)
    assert loop.examiner_result is None
    assert loop.rounds == [], "a failed derivation closes no round"


def test_an_examiner_whose_derive_errors_fails_the_round(tmp_path, request):
    """Greptile P1: a derive that errors is the round failing on stale state, never done or stalled."""
    from kullback.examiner.agent import ExaminerError

    loop = _model_driven_beat(tmp_path, request, [_reply(None, ("derive", {"target": "no-such-task"}))],
                              "bad-derive")
    with pytest.raises(ExaminerError, match="no Task is named"):
        loop.examiner_beat(1)
    assert loop.examiner_result is not None and loop.examiner_result.is_error
    assert loop.rounds == [], "a failed derivation closes no round"


def test_a_broken_examiner_stalls_the_run_with_the_reason_on_the_record(tmp_path, request):
    """Greptile P1, run level: a model that never derives must stall the run with a record, never
    crash it without one. rounds.json tells the whole story for the next session."""
    workdir = tmp_path / "broken-examiner"
    agent_model = TestModel([
        _reply(None, ("build", {"target": TARGET})),
        _reply("built."),
        _reply("I have read everything and all is well."),
    ])
    result = rounds.run_rounds(workdir, model=Bodies(), agent_model=agent_model, files=[_fixture(request)],
                               max_attempts=0, allowance_usd=0.0)
    assert result["exit"] == "stalled" and result["failed"] is True
    stored = rounds.load_rounds(workdir)
    assert len(stored) == 1
    assert stored[0].exit == "stalled" and "never called derive" in (stored[0].exit_note or "")
    assert stored[0].failed is True


def test_a_finding_filed_in_round_one_is_performed_in_round_two_then_the_run_exits(tmp_path, request):
    """Greptile P1 (round 2), run level: the finding in round 1 clears the would-be exit, round 2's
    Builder beat performs it, and the run exits only with nothing pending. The required follow-up
    is performed, not just recorded."""
    workdir = tmp_path / "finding-then-done"
    agent_model = TestModel([
        _reply(None, ("build", {"target": TARGET})),
        _reply("built."),
        _reply(None, ("finding", {"task_id": "1", "kind": "fidelity",
                                  "text": "the replay diverges at the second call", "suggested": "replay"})),
        _reply(None, ("derive", {"target": "all"})),
        _reply("filed and derived."),
        # Round 1's derive files three findings of its own off the records before the model files
        # its one (D170), and round 2's Builder beat reads each as its own follow-up message.
    ] + [_reply("read the follow-up.")] * 4 + [
        _reply("acted on the findings."),
        _reply(None, ("derive", {"target": "all"})),
        _reply("re-derived clean."),
    ])
    result = rounds.run_rounds(workdir, model=Bodies(), agent_model=agent_model, files=[_fixture(request)],
                               max_attempts=0)
    stored = rounds.load_rounds(workdir)
    assert len(stored) == 2
    assert stored[0].exit is None and [f.finding_id for f in stored[0].pending_findings] != []
    # The fixture's Tasks never get a Reference, so the run cannot be done (D172): with nothing
    # pending and no gate count moved since round 1, it exits stalled.
    assert stored[1].exit == "stalled" and stored[1].pending_findings == []
    assert result["exit"] == "stalled" and result["failed"] is False
    findings = json.loads((workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))
    assert [f["status"] for f in findings] == ["closed"] * 4, "the rule findings close with the model's"


def test_a_builder_error_keeps_its_findings_queued(tmp_path, monkeypatch):
    """Greptile P1 (round 3): the beat clears findings only on success. A Builder error leaves the
    queue exactly as it found it, so the findings stay open and queued instead of orphaned."""
    from kullback.agent.tools import ToolResult
    from kullback.builder.build import BuildError

    loop = _bare_loop(tmp_path)
    loop.pending_findings = [_finding("t1", suggested="none")]
    monkeypatch.setattr(builder_agent, "drive_tool",
                        lambda *args, **kwargs: ToolResult(content="the build broke", is_error=True))
    with pytest.raises(BuildError):
        loop.builder_beat(1)
    assert [f.finding_id for f in loop.pending_findings] == ["f1"]


def test_a_builder_that_quits_in_round_two_stalls_with_the_finding_still_queued(tmp_path, request):
    """Greptile P1 (round 3), run level: round 1 files a finding, round 2's Builder never builds.
    The run stalls with the reason on the record and the unperformed finding still queued on it,
    instead of crashing with the finding lost."""
    workdir = tmp_path / "builder-quits"
    agent_model = TestModel([
        _reply(None, ("build", {"target": TARGET})),
        _reply("built."),
        _reply(None, ("finding", {"task_id": "1", "kind": "fidelity",
                                  "text": "the replay diverges at the second call", "suggested": "replay"})),
        _reply(None, ("derive", {"target": "all"})),
        _reply("filed and derived."),
        _reply(None, ("build", {"target": "no-such-target"})),
    ])
    result = rounds.run_rounds(workdir, model=Bodies(), agent_model=agent_model, files=[_fixture(request)],
                               max_attempts=0)
    stored = rounds.load_rounds(workdir)
    assert len(stored) == 2
    assert stored[0].exit is None and [f.finding_id for f in stored[0].pending_findings] != []
    assert stored[1].exit == "stalled" and "builder failed" in (stored[1].exit_note or "")
    assert [f.finding_id for f in stored[1].pending_findings] != []
    assert result["exit"] == "stalled"


def test_the_builder_is_handed_the_finding_with_the_verb_and_hint_as_a_callable_line(tmp_path):
    """The suggestion reaches the Builder as the call it can make, hint and all, and the arguments
    the code driver passes are the ones that verb's tool takes: the Examiner's line is not a
    paraphrase the Builder has to reconstruct."""
    harness = builder_agent.build_harness(BuildPlan(workdir=tmp_path / "work"))
    intent = Finding(finding_id="finding-1", task_id="task_x", kind="fidelity", suggested="repair_intent",
                     hint="the Runs only ever cancel one order", round=2,
                     text="the Intent says gift card and no Run says it")
    message = rounds.finding_message(intent)
    assert message == ("Finding finding-1 (fidelity, 1 Task): the Intent says gift card and no Run says it "
                       "Task task_x. Suggested: repair_intent(task_id='task_x', "
                       "hint='the Runs only ever cancel one order')")
    recompile = Finding(finding_id="finding-2", kind="fidelity", suggested="repair_recompile",
                        tool="cancel_pending_order", hint="the status column differs on replay",
                        text="the body writes a different status")
    assert rounds.finding_message(recompile).endswith(
        "Suggested: repair_recompile(name='cancel_pending_order', hint='the status column differs on replay')")
    # The rebuild verbs are unchanged and take no hint.
    assert rounds.finding_message(_finding("t1")).endswith("Suggested: replay(task='t1')")
    for finding in (intent, recompile):
        arguments = rounds.finding_arguments(finding)
        tool = harness.registry.get(finding.suggested)
        assert tool is not None, f"the Builder has no {finding.suggested} tool"
        assert tool.args_model.model_validate(arguments).hint == finding.hint
        assert rounds.suggested_call(finding) in rounds.finding_message(finding)


def test_a_failed_suggested_action_keeps_its_finding_queued(tmp_path, monkeypatch):
    """Greptile P1 (round 4): on the code path a suggested action that errors keeps its finding
    queued for the next beat, while the finding with nothing to do is closed by the success."""
    from kullback.agent.tools import ToolResult

    loop = _bare_loop(tmp_path)
    loop.plan.last = object()  # a previous build exists, so the beat may succeed
    loop.pending_findings = [_finding("t1", suggested="replay"), _finding("t2", suggested="none",
                                                                           finding_id="f2")]

    def fake_drive(*args, **kwargs):
        if args[1] == "build":
            return ToolResult(content="built", is_error=False)
        return ToolResult(content="the action broke", is_error=True)

    monkeypatch.setattr(builder_agent, "drive_tool", fake_drive)
    loop.builder_beat(1)
    assert [f.finding_id for f in loop.pending_findings] == ["f1"]


def test_a_new_loop_resumes_findings_an_earlier_run_left_open(tmp_path):
    """Greptile P1 (round 4): a ceiling that ends a run with findings pending strands nothing. The
    next Loop over the workdir re-queues what findings.json still holds open (and only that)."""
    workdir = tmp_path / "work"
    (workdir / "examiner").mkdir(parents=True)
    rows = [{**as_dict(_finding("t1")), "status": "open"},
            {**as_dict(_finding("t2", finding_id="f2")), "status": "closed"}]
    (workdir / "examiner" / "findings.json").write_text(json.dumps(rows), encoding="utf-8")
    assert [f.finding_id for f in _bare_loop(tmp_path).pending_findings] == ["f1"]


def test_a_new_loop_survives_a_half_written_findings_file(tmp_path):
    """A findings file cut mid-write resumes nothing and stops nothing: the loop that would drain
    it must not die on it."""
    workdir = tmp_path / "work"
    (workdir / "examiner").mkdir(parents=True)
    (workdir / "examiner" / "findings.json").write_text('[{"finding_id": "f1", ', encoding="utf-8")
    assert _bare_loop(tmp_path).pending_findings == []


def test_a_resumed_finding_is_closed_once_the_examiner_opens(tmp_path, request):
    """Greptile P1 (round 5): round 1's Builder beat delivers the resumed finding before eplan
    exists, so it cannot close then — but it must close the moment the Examiner opens, exactly
    once. Otherwise every later invocation would repeat the remediation with the entry still
    protected."""
    workdir = tmp_path / "work"
    (workdir / "examiner").mkdir(parents=True)
    rows = [{**as_dict(_finding("t1", suggested="none")), "status": "open"}]
    (workdir / "examiner" / "findings.json").write_text(json.dumps(rows), encoding="utf-8")
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    assert [f.finding_id for f in loop.pending_findings] == ["f1"]
    loop.builder_beat(1)
    assert loop.pending_findings == [] and loop._unclosed == ["f1"]
    loop.examiner_beat(1)
    assert loop._unclosed == []
    stored = json.loads((workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))
    assert stored[0]["status"] == "closed" and stored[0]["finding_id"] == "f1"
    assert all(r["status"] == "open" for r in stored[1:]), "the rule findings this beat filed are open"
    # And a second Loop over the workdir finds f1 answered: no repeated remediation. What it does
    # resume is the findings this beat's own derive filed, which no Builder beat has been handed.
    resumed = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan)).pending_findings
    assert "f1" not in [f.finding_id for f in resumed]


def test_a_resumed_finding_reaches_the_model_on_round_one(tmp_path, request):
    """Greptile P1 (round 6): round 1 has no steer, only the prompt — but a resumed finding must
    still reach the model as a follow-up in the same watched stream, not be dequeued unheard."""
    from kullback.ai.messages import UserMessage

    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    finding = _finding("t1", suggested="none")
    agent_model = TestModel([_reply("starting the build."), _reply("noted the finding.")])
    harness = builder_agent.build_harness(plan, agent_model)
    loop = rounds.Loop(plan=plan, builder=harness, agent_model=agent_model)
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.pending_findings = [finding]
    loop.builder_beat(1)
    delivered = [m for m in harness.messages if isinstance(m, UserMessage) and m.details is not None]
    assert {f["finding_id"] for m in delivered for f in [m.details["finding"]]} == {"f1"}
    # The model heard the finding and never called build; the driver built the target itself, so
    # the beat succeeded and the finding it delivered is closed rather than queued again.
    assert loop.pending_findings == [] and loop.driver_built == [1]


def test_result_with_no_build_reports_the_stalled_round_without_crashing(tmp_path, request):
    """Build 9 post-mortem: a first beat that fails leaves plan.last None, and result() must
    report the recorded stalled round (failed, with the reason) instead of AttributeError."""
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.pending_findings = [_finding("t1", suggested="none")]
    record = loop.close_round(1, {})
    record.exit = "stalled"
    record.failed = True
    record.exit_note = "builder failed: boom"
    out = loop.result()
    assert out["exit"] == "stalled" and out["failed"] is True
    assert out["trusted"] == [] and out["refused"] == {}
    assert out["rounds"][0]["exit_note"] == "builder failed: boom"


def test_the_round_cap_ends_the_loop_even_with_findings_pending_and_says_so(tmp_path):
    """D169: the cap is a hard stop like the ceiling; a finding still open rides on the record with
    the note that the cap ended the run first."""
    loop = _bare_loop(tmp_path, max_rounds=2)
    loop.plan.last = pipeline.PipelineResult(status="ok")
    loop.close_round(1, _record(1, fidelity=10, trusted=1).counts)
    loop.pending_findings = [_finding("t1")]
    record = loop.close_round(2, _record(2, fidelity=11, trusted=2).counts)
    assert record.exit == "max_rounds"
    assert "round cap" in (record.exit_note or "")
    assert [f.finding_id for f in record.pending_findings] == ["f1"]


def test_fidelity_flat_for_the_window_exits_stalled_with_the_reason_on_the_record(tmp_path):
    loop = _bare_loop(tmp_path, fidelity_stall=2)
    loop.plan.last = pipeline.PipelineResult(status="ok")
    loop.close_round(1, _record(1, fidelity=10, trusted=1).counts)
    loop.close_round(2, _record(2, fidelity=10, trusted=2).counts)
    record = loop.close_round(3, _record(3, fidelity=10, trusted=1).counts)
    assert record.exit == "stalled"
    assert "fidelity did not rise in 2 rounds" in (record.exit_note or "")
