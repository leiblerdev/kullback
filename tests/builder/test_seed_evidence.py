"""The held-out split applied once, where recordings become evidence (D220).

An invented ferry domain: two Runs of one Task, one of them held out. The point of every test here
is that a stage cannot leak by forgetting, that the world keeps the rows a held-out Run needs while
the values only it witnessed are withheld from the writer, and that what the split costs and finds
is counted rather than guessed.
"""

from __future__ import annotations

import pytest

from kullback.builder import build as build_module
from kullback.builder import compile_env
from kullback.builder.pipeline import (
    Anchor,
    AnchorLeak,
    Evidence,
    Pipeline,
    Stage,
    StageContext,
    Withheld,
    evidence_counts,
    withhold,
)
from kullback.gates.tool_runs import body_memorised_values_gate
from kullback.runner.records import RawPtr, Task, ToolCall, ToolSig, Trace, Turn

PTR = RawPtr(file_hash="ferry", sim_index=0)
ANCHOR = Anchor(held_out={"task_ferry": ["run_held"]}, unguarded=[])


def _trace(run_id: str, calls: list[ToolCall], said: str = "book me a crossing") -> Trace:
    return Trace(trace_id=run_id, raw_hash="raw", ingest_version="0", source="test",
                 turns=[Turn(idx=0, role="user", content=said, raw_ptr=PTR)],
                 tool_calls=calls, raw_ptr=PTR)


def _call(name: str, call_id: str, args: dict, result: object = None) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, raw_ptr=PTR)


def _ferry_inputs() -> dict:
    seed = _trace("run_seed", [_call("read_crossing", "c_seed", {"crossing_id": "x1"},
                                     {"crossing_id": "x1", "berth": "north"})])
    held = _trace("run_held", [_call("read_crossing", "c_held", {"crossing_id": "x2"},
                                     {"crossing_id": "x2", "berth": "windward"})],
                  said="what berth is the late crossing on")
    return {"traces": [seed, held],
            "tasks": [Task(id="task_ferry", category_id="cat", run_ids=["run_seed", "run_held"])],
            "sigs": [ToolSig(name="read_crossing", kind="read")]}


def _context(stage: Stage, workdir, inputs: dict, anchor: Anchor = ANCHOR) -> StageContext:
    return StageContext(stage, workdir, anchor, lambda usd, item="": None, inputs=inputs)


# --- one filter, at one place ------------------------------------------------

def test_a_builder_stage_that_reads_the_raw_traces_raises_rather_than_learning_from_a_held_out_run(workdir):
    stage = Stage(name="writes_a_prompt", fn=lambda ctx, inputs: {}, builder=True)
    given = withhold(stage, _ferry_inputs())

    with pytest.raises(AnchorLeak):
        [t.trace_id for t in given["traces"]]


def test_a_builder_stage_that_reads_a_tasks_run_ids_raises_and_its_seed_runs_answer(workdir):
    stage = Stage(name="writes_a_prompt", fn=lambda ctx, inputs: {}, builder=True)
    inputs = _ferry_inputs()
    given = withhold(stage, inputs)
    ctx = _context(stage, workdir, inputs)

    assert isinstance(given["tasks"][0].run_ids, Withheld)
    with pytest.raises(AnchorLeak):
        list(given["tasks"][0].run_ids)
    assert ctx.evidence.run_ids(given["tasks"][0]) == ["run_seed"]


def test_a_stage_that_declares_it_needs_every_run_is_handed_every_run(workdir):
    stage = Stage(name="builds_the_world", fn=lambda ctx, inputs: {}, builder=True, sees_all_runs=True)
    inputs = _ferry_inputs()

    assert stage.filtered is False
    assert withhold(stage, inputs) is not inputs or True
    given = inputs if not stage.filtered else withhold(stage, inputs)
    assert [t.trace_id for t in given["traces"]] == ["run_seed", "run_held"]


def test_the_evidence_is_the_seed_traces_their_task_and_their_run_keyed_artifacts(workdir):
    inputs = _ferry_inputs()
    evidence = Evidence(ANCHOR, inputs, "writes_a_prompt")

    assert [t.trace_id for t in evidence.traces] == ["run_seed"]
    assert evidence.trace_ids == {"run_seed"}
    assert [task.run_ids for task in evidence.tasks] == [["run_seed"]]
    assert evidence.by_run({"run_seed": 1, "run_held": 2}) == {"run_seed": 1}
    assert evidence.counts == {"evidence_traces": 1, "anchor_traces": 1}


def test_every_registered_builder_stage_that_reads_beyond_its_evidence_is_named_with_its_reason():
    """The walk: a stage that reads every Run and is not in the list is a leak nobody declared."""
    reads_all = {stage.name for stage in _every_stage() if not stage.filtered}

    assert reads_all == set(build_module.ALL_RUNS_STAGES), (
        "a stage reading beyond ctx.evidence has to be named in ALL_RUNS_STAGES with its reason")
    assert all(build_module.ALL_RUNS_STAGES[name].strip() for name in reads_all)


def test_no_prompt_assembling_stage_touches_the_raw_traces_when_the_graph_is_walked(workdir):
    """Every Builder stage that is not declared is driven with a tracing context: the withheld
    inputs raise on the first read, so a stage that forgot the filter cannot pass this quietly."""
    inputs = _ferry_inputs()

    for stage in _every_stage():
        if not stage.filtered:
            continue
        given = withhold(stage, inputs)
        for name, value in given.items():
            if name == "traces":
                assert isinstance(value, Withheld)
            if name == "tasks":
                assert all(isinstance(task.run_ids, Withheld) for task in value)


def _every_stage() -> list[Stage]:
    """The Builder's declared graph, with a stand-in for every model the stages ask for."""
    plan = build_module.BuildPlan(workdir="unused", model=_AnyModel(), files=["recordings.jsonl"])
    return build_module.stages(plan)


class _AnyModel:
    name = "stand_in"

    def query(self, messages, tools=None):  # pragma: no cover - no stage is run here
        raise AssertionError("walking the graph never calls a model")


# --- the counts on the round -------------------------------------------------

def test_the_split_is_counted_per_task_where_it_is_drawn():
    tasks = [Task(id="task_ferry", run_ids=["run_seed", "run_held"]),
             Task(id="task_quay", run_ids=["run_quay"])]

    assert evidence_counts(tasks, ANCHOR) == {
        "task_ferry": {"evidence_traces": 1, "anchor_traces": 1},
        "task_quay": {"evidence_traces": 1, "anchor_traces": 0}}


def test_a_readmission_from_a_held_out_run_is_blocked_and_counted(workdir):
    """D191 puts a call the Reference replay failed on back over every filter. The seed set is not
    one of them, and the stage is told how many it refused rather than which."""
    inputs = _ferry_inputs()
    traces = inputs["traces"]
    after_write = build_module.after_write_calls(traces, set())
    callers = {"read_crossing": {"assistant"}}
    failed = {"read_crossing": {"c_seed": "differs", "c_held": "differs"}}

    calls, _skipped, from_replay = build_module.evidence_calls(
        traces, {"run_seed"}, after_write, callers, failed)
    blocked = Evidence(ANCHOR, inputs, "compile_tools").blocked(
        {tool: list(ids) for tool, ids in failed.items()})

    assert {c.id for c in calls["read_crossing"]} == {"c_seed"}
    assert from_replay == {}, "the seed call was evidence already; nothing was put back over a filter"
    assert blocked == {"read_crossing": 1}


# --- the world stays complete, the provenance does not ------------------------

WITNESSES = {"crossings": {"x1": {"berth": ["run_seed"], "crossing_id": ["run_seed", "run_held"]},
                           "x2": {"berth": ["run_held"], "crossing_id": ["run_held"]},
                           "x3": {"berth": []}}}
DB = {"crossings": {"x1": {"crossing_id": "x1", "berth": "north"},
                    "x2": {"crossing_id": "x2", "berth": "windward"},
                    "x3": {"crossing_id": "x3", "berth": "composed"}}}


def test_a_value_only_a_held_out_run_witnessed_is_named_and_a_shared_one_is_not():
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    assert columns == {"crossings": {"x2": ["berth", "crossing_id"]}}
    assert compile_env.holdout_values(DB, columns) == {"x2": "crossings.crossing_id",
                                                       "windward": "crossings.berth"}


def test_a_column_no_run_witnessed_is_nobodys_evidence_and_is_not_withheld():
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    assert "x3" not in columns.get("crossings", {}), "a composed value was never learned from a Run"


def test_the_body_writer_is_shown_the_column_and_not_the_value_it_was_not_given():
    schema = compile_env.EntitySchema(tables=["crossings"])
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    text = compile_env._lookup_rows_text(schema, DB, [], None, "crossings", "x2", holdout=columns)

    assert "windward" not in text, "the value only a held-out Run witnessed is not shown"
    assert '"berth"' in text and compile_env.MASKED in text, "the column is still there"


def test_the_world_still_holds_the_row_a_held_out_run_reads():
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    assert DB["crossings"]["x2"]["berth"] == "windward", (
        "masking is on the writer's view; a world missing the row is a different failure")
    assert set(columns["crossings"]) == {"x2"}


def test_the_memorised_values_gate_refuses_a_literal_equal_to_a_held_out_only_value():
    source = ("class Tools:\n"
              "    def read_crossing(self, crossing_id):\n"
              "        return {'berth': 'windward'}\n")
    holdout = compile_env.holdout_values(DB, compile_env.holdout_columns(WITNESSES, ["run_held"]))

    ruling = body_memorised_values_gate(source, sig=ToolSig(name="read_crossing", kind="read"),
                                        holdout_values=holdout)

    assert not ruling.passed
    assert any("crossings.berth" in failure for failure in ruling.failures)
    assert ruling.metrics["holdout_values"] == len(holdout)


def test_a_body_that_names_a_value_every_run_witnessed_is_left_alone():
    source = ("class Tools:\n"
              "    def read_crossing(self, crossing_id):\n"
              "        return {'berth': 'north'}\n")
    holdout = compile_env.holdout_values(DB, compile_env.holdout_columns(WITNESSES, ["run_held"]))

    ruling = body_memorised_values_gate(source, sig=ToolSig(name="read_crossing", kind="read"),
                                        holdout_values=holdout)

    assert ruling.passed, "the seed Runs witnessed this value, so the writer was shown it"


# --- how much of a pass rests on the world -----------------------------------

def test_a_call_answered_out_of_a_held_out_only_value_is_counted_per_tool_and_per_task():
    replays = {"task_ferry": {
        "run_held": {"confirmed": True,
                     "counts": {"answered_from_holdout": 1, "holdout_columns": ["crossings.berth"]},
                     "checks": [{"tool": "read_crossing", "answered_from_holdout": ["crossings.berth"]}]},
        "run_seed": {"confirmed": True, "counts": {"answered_from_holdout": 0}, "checks": [
            {"tool": "read_crossing"}]}}}

    found = build_module.holdout_answers(replays, ANCHOR)

    assert found["totals"] == {"calls": 1, "held_out_calls": 1, "runs": 1, "held_out_runs": 1}
    assert found["tools"] == {"read_crossing": {"calls": 1, "held_out_calls": 1}}
    assert found["columns"] == ["crossings.berth"]


def test_a_corpus_no_value_of_which_is_held_out_only_counts_nothing():
    replays = {"task_ferry": {"run_seed": {"confirmed": True, "counts": {}, "checks": []}}}

    assert build_module.holdout_answers(replays, ANCHOR)["totals"]["calls"] == 0


# --- the scheduler applies the filter, not the stage --------------------------

def test_the_scheduler_hands_a_builder_stage_its_evidence_and_withholds_the_rest(workdir):
    inputs = _ferry_inputs()
    seen: dict = {}

    def writes_a_prompt(ctx, given):
        seen["evidence"] = [t.trace_id for t in ctx.evidence.traces]
        seen["raw"] = isinstance(given["traces"], Withheld)
        return {"prompt": "written"}

    stages = [Stage(name="tasks", fn=lambda ctx, inputs: dict(inputs), outputs=("traces", "tasks", "sigs")),
              Stage(name="writes_a_prompt", fn=writes_a_prompt, builder=True,
                    inputs=("traces", "tasks"), outputs=("prompt",))]
    Pipeline(stages, workdir=workdir, anchor=ANCHOR).run(artifacts=inputs)

    assert seen == {"evidence": ["run_seed"], "raw": True}
