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

import pytest

from builder.test_build import Bodies
from kullback import rounds
from kullback.agent.events import BeatEnd, BeatStart, RoundEnd, RoundStart, ToolExecutionEnd, ToolExecutionStart
from kullback.agent.harness import AgentHarness
from kullback.ai.messages import UserMessage
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import agent as builder_agent
from kullback.builder import pipeline
from kullback.builder.build import BuildPlan
from kullback.examiner import agent as examiner_agent
from kullback.examiner.agent import examiner_message
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.stage import DERIVE_INPUTS, FORBIDDEN_INPUTS
from kullback.gates import round_end
from kullback.runner.records import Finding, RoundRecord, as_dict

TARGET = "environment"
# sha256 of task_status.json from the full offline build over the fixture made before this phase, through
# run_builder with derive_verifier still a Builder stage (D130): the rounds must leave the same bytes.
TASK_STATUS_SHA256_BEFORE_THE_PHASE = "e6ec7ca50f375163805c0651745c411d39d45bfa8dfe9a2e59ba5bcea335cd73"


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
    loop.pending_findings = [_finding(task_id)]
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


def test_the_code_driver_acts_on_a_finding_by_calling_the_builder_tool_it_names(code_loop):
    """A finding that suggests `replay` for a Task is a replay(task) call through the Builder's hooks,
    before the build of the target; the code driver never asks a model what to do with it."""
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
    assert sum(1 for e in model_loop["events"] if isinstance(e, ToolExecutionEnd)) == 2


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


def test_the_examiner_plan_sees_its_allowance_shrink_at_every_tool_end(model_examiner_loop):
    """rounds.py hands the Examiner's plan what is left of the round's allowance after each tool, so the
    reroll tool refuses at the right moment; under an allowance of zero nothing is left after derive."""
    loop = model_examiner_loop["loop"]
    assert loop.eplan.allowance_remaining is not None and loop.eplan.allowance_remaining <= 0
    assert loop.spent_allowance["examiner"] is True


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

    result = rounds.run_rounds(workdir, model=Bodies(), files=[_fixture(request)], max_attempts=0,
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
    assert set(counts["spend"]) == {"builder", "examiner", "total"} and counts["findings"] == []
    assert rounds.load_rounds(driven["workdir"])[-1].counts == counts
    assert [d for d in driven["dicts"] if d.get("kind") == "round"][-1]["counts"] == counts


def test_the_loop_exits_done_when_the_state_holds_after_a_round_over_the_fixture(driven):
    """No Task on the fixture has a confirmed Reference, so D126's state holds after round 1 (the
    gates' own claim in tests/gates/test_round_end.py), and the driver stops there."""
    assert driven["result"]["exit"] == "done"
    assert [r["round"] for r in driven["result"]["rounds"]] == [1]
    assert driven["events"][-1].exit == "done"


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
    plan = BuildPlan(workdir=tmp_path / "work", model=Bodies(), files=[_fixture(request)], max_attempts=0)
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    loop.allowance = {agent: None for agent in rounds.AGENTS}
    loop.builder_beat(1)
    assert {"bodies", "db", "schema", "environment"} <= set(plan.store)
    loop.examiner_beat(1)
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
    assert [f.finding_id for f in loop.pending_findings] == [filed.details["finding"]["finding_id"]]
    loop.builder_beat(2)
    findings = json.loads((plan.workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))
    assert [f["status"] for f in findings] == ["closed"]
    assert loop.pending_findings == []


# --- terminal rounds keep their findings, failed derives fail the round ------------------

def test_close_round_never_reports_done_with_findings_pending(tmp_path):
    """Greptile P1: a round that also reaches done must not drop the Examiner's findings. The
    pending findings ride on the terminal record in rounds.json, and the exit is stalled, not done."""
    loop = _bare_loop(tmp_path)
    loop.pending_findings = [_finding("t1")]
    record = loop.close_round(1, _record(1, unfinished=[]).counts)
    assert record.exit == "stalled"
    assert "finding" in (record.exit_note or "")
    assert [f.finding_id for f in record.pending_findings] == ["f1"]
    body = json.loads((tmp_path / "work" / rounds.ROUNDS_NAME).read_text(encoding="utf-8"))
    assert body[0]["exit"] == "stalled"
    assert body[0]["pending_findings"][0]["finding_id"] == "f1"
    assert rounds.load_rounds(tmp_path / "work")[0].pending_findings[0].finding_id == "f1"


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
