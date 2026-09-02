"""The pipeline as a scheduler over the declared DAG (phase 4): order from reads and writes, stages side
by side, targets that resolve upstream first, and the typed stage events."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from kullback.agent.events import StageEnd, StageStart
from kullback.builder.pipeline import Anchor, BudgetStop, GraphError, Pipeline, Stage
from kullback.runner.records import GateResult, Task


def _stage(name, inputs=(), outputs=(), body=None, **kw):
    def fn(ctx, inp):
        return body(ctx, inp) if body else {out: name for out in outputs}
    return Stage(name, fn, list(inputs), list(outputs), code_version=kw.pop("code_version", f"{name}:v1"), **kw)


# --- order from the declaration -------------------------------------------

def test_stages_run_in_the_order_their_reads_and_writes_imply_whatever_the_declared_order(tmp_path):
    ran = []

    def body(ctx, inputs):
        ran.append(ctx.name)
        # Each output is built from what the stage read, so an artifact that came out right could
        # only have been made after the stage that produced its input.
        made = f"{'+'.join(sorted(inputs))}>{ctx.name}" if inputs else ctx.name
        return {out: made for out in ctx.stage.outputs}

    stages = [_stage("report", ["env", "tasks"], ["report"], body), _stage("tasks", ["traces"], ["tasks"], body),
              _stage("env", ["traces"], ["env"], body), _stage("ingest", [], ["traces"], body)]
    result = Pipeline(stages, tmp_path).run()
    assert result.status == "complete"
    assert ran.index("ingest") < ran.index("tasks") < ran.index("report")
    assert ran.index("ingest") < ran.index("env") < ran.index("report")
    assert result.statuses == {"ingest": "ran", "tasks": "ran", "env": "ran", "report": "ran"}
    assert result.artifacts["tasks"] == "traces>tasks"
    assert result.artifacts["report"] == "env+tasks>report"


def test_independent_stages_run_side_by_side_when_workers_allow_it(tmp_path):
    """Two stages that read the same artifact and nothing of each other overlap in time at two workers,
    and at one worker they do not (D118 at the stage level)."""
    windows = {}
    gate = threading.Barrier(2, timeout=5)

    def overlapping(ctx, inputs):
        windows[ctx.name] = [time.perf_counter(), None]
        if ctx.name in ("left", "right"):
            gate.wait()  # both are in flight at once, or the barrier times out and the test fails
        windows[ctx.name][1] = time.perf_counter()
        return {out: ctx.name for out in ctx.stage.outputs}

    stages = [_stage("root", [], ["a"], overlapping), _stage("left", ["a"], ["b"], overlapping),
              _stage("right", ["a"], ["c"], overlapping), _stage("join", ["b", "c"], ["d"], overlapping)]
    result = Pipeline(stages, tmp_path / "two", workers=2).run()
    assert result.status == "complete" and result.artifacts["d"] == "join"
    assert windows["left"][0] < windows["right"][1] and windows["right"][0] < windows["left"][1]
    assert windows["root"][1] <= min(windows["left"][0], windows["right"][0])
    assert windows["join"][0] >= max(windows["left"][1], windows["right"][1])

    # At one worker, no clock: each stage says when it entered and when it left, and the sequence
    # has to nest, so no stage's entry falls between another's entry and its exit.
    order = []

    def in_and_out(ctx, inputs):
        order.append((ctx.name, "in"))
        time.sleep(0.02)  # a window a second stage would enter, if the scheduler let one start
        order.append((ctx.name, "out"))
        return {out: ctx.name for out in ctx.stage.outputs}

    one = [_stage("root", [], ["a"], in_and_out), _stage("left", ["a"], ["b"], in_and_out),
           _stage("right", ["a"], ["c"], in_and_out), _stage("join", ["b", "c"], ["d"], in_and_out)]
    assert Pipeline(one, tmp_path / "one", workers=1).run().status == "complete"
    assert order == [("root", "in"), ("root", "out"), ("left", "in"), ("left", "out"),
                     ("right", "in"), ("right", "out"), ("join", "in"), ("join", "out")]


def test_a_worker_count_of_two_writes_the_same_artifacts_and_state_as_one(tmp_path):
    def body(ctx, inputs):
        return {out: sorted(inputs) + [ctx.name] for out in ctx.stage.outputs}

    def make(root, workers):
        stages = [_stage("root", [], ["a"], body), _stage("left", ["a"], ["b"], body), _stage("right", ["a"], ["c"], body),
                  _stage("join", ["b", "c"], ["d"], body)]
        return Pipeline(stages, root, workers=workers).run()

    one, two = make(tmp_path / "one", 1), make(tmp_path / "two", 2)
    assert one.artifacts == two.artifacts and one.statuses == two.statuses
    assert sorted(p.name for p in (tmp_path / "one" / "cache").glob("*.json")) == \
        sorted(p.name for p in (tmp_path / "two" / "cache").glob("*.json"))


# --- targets resolve upstream first ----------------------------------------

def test_a_target_runs_only_itself_and_what_is_upstream_of_it(tmp_path):
    ran = []

    def body(ctx, inputs):
        ran.append(ctx.name)
        return {out: ctx.name for out in ctx.stage.outputs}

    stages = [_stage("ingest", [], ["traces"], body), _stage("mine", ["traces"], ["sigs"], body),
              _stage("cluster", ["traces", "sigs"], ["tasks"], body), _stage("policy", ["traces"], ["rules"], body)]
    result = Pipeline(stages, tmp_path).run(targets=["tasks"])
    assert ran == ["ingest", "mine", "cluster"]
    assert result.statuses == {"ingest": "ran", "mine": "ran", "cluster": "ran", "policy": "pending"}
    assert set(result.reports) == {"ingest", "mine", "cluster"}
    assert Pipeline(stages, tmp_path).run(targets=["policy"]).statuses["policy"] == "ran"
    with pytest.raises(GraphError, match="neither a stage nor an artifact"):
        Pipeline(stages, tmp_path).run(targets=["nothing"])


def test_a_stale_input_is_rebuilt_before_the_target_and_a_current_one_is_served_from_the_cache(tmp_path):
    ran = []

    def body(ctx, inputs):
        ran.append(ctx.name)
        return {out: f"{ctx.name}:{ctx.stage.code_version}" for out in ctx.stage.outputs}

    def graph(mine_version):
        return [_stage("ingest", [], ["traces"], body), _stage("mine", ["traces"], ["sigs"], body, code_version=mine_version),
                _stage("cluster", ["sigs"], ["tasks"], body)]

    Pipeline(graph("mine:v1"), tmp_path).run(targets=["tasks"])
    ran.clear()
    second = Pipeline(graph("mine:v1"), tmp_path).run(targets=["tasks"])
    assert ran == [] and all(r.cached for r in second.reports.values())
    ran.clear()
    third = Pipeline(graph("mine:v2"), tmp_path).run(targets=["cluster"])
    assert ran == ["mine", "cluster"], "the changed upstream stage ran first, then the target over its fresh output"
    assert third.reports["ingest"].cached and not third.reports["cluster"].cached
    assert third.artifacts["tasks"] == "cluster:cluster:v1" and third.artifacts["sigs"] == "mine:mine:v2"


# --- the typed stage events -------------------------------------------------

def _stage_states(events) -> list:
    """The (stage, state) pairs both channels carry: the dict events name them, the typed events are them."""
    out = []
    for event in events:
        if isinstance(event, StageStart):
            out.append((event.name, "start"))
        elif isinstance(event, StageEnd):
            out.append((event.name, event.counts["status"]))
        elif isinstance(event, dict) and event.get("kind") == "stage":
            out.append((event["stage"], event["state"]))
    return out


@pytest.mark.parametrize("channel", ["emit", "on_event"])
def test_stage_start_and_stage_end_are_emitted_in_order_with_what_the_stage_came_to(tmp_path, channel):
    """Both subscriber channels: `emit` carries the typed events, `on_event` the dicts a screen reads."""
    seen = []

    def recorder(ctx, inputs):
        ctx.record_gate(GateResult(stage="mine_ok", passed=True))
        return {"sigs": [1]}

    def gate(ctx, outputs):
        return GateResult(stage="mine", passed=True)

    stages = [_stage("ingest", [], ["traces"]), Stage("mine", recorder, ["traces"], ["sigs"], gate=gate, code_version="m")]
    Pipeline(stages, tmp_path, **{channel: seen.append}).run()
    assert _stage_states(seen) == [("ingest", "start"), ("ingest", "ran"), ("mine", "start"), ("mine", "ran")]
    if channel == "emit":
        assert all(isinstance(e, (StageStart, StageEnd)) for e in seen)
        end = seen[-1]
        assert end.counts["cached"] is False and end.counts["attempts"] == 1
        assert end.counts["rulings"] == ["mine_ok", "mine"] and end.counts["elapsed_ms"] >= 0
        assert end.counts["produced"] == ["sigs"]
    else:
        assert seen[-1] == {"kind": "pipeline", "state": "complete", "failed_stage": None}

    again = []
    Pipeline(stages, tmp_path, **{channel: again.append}).run()
    assert _stage_states(again) == [("ingest", "start"), ("ingest", "cached"),
                                    ("mine", "start"), ("mine", "cached")]
    if channel == "emit":
        assert again[-1].counts["cached"] is True
        assert again[-1].counts["rulings"] == ["mine"], "a cached stage records nothing; its own gate still rules"


@pytest.mark.parametrize("channel", ["emit", "on_event"])
def test_a_callback_that_raises_does_not_fail_the_build(tmp_path, channel):
    """Each channel has its own guard, so each has to swallow the exception a subscriber threw."""
    def angry(event):
        raise RuntimeError("the screen died")

    result = Pipeline([_stage("s", [], ["a"])], tmp_path, **{channel: angry}).run()
    assert result.status == "complete"
    assert result.artifacts["a"] == "s", "a build that quietly lost its output is not one that went fine"


# --- gates.json through the ledger ----------------------------------------

def test_rulings_recorded_from_stages_on_two_threads_land_in_stage_order(tmp_path):
    def recorder(ctx, inputs):
        time.sleep(0.02 if ctx.name == "left" else 0.0)
        ctx.record_gate(GateResult(stage=ctx.name, passed=True, metrics={"n": 1}))
        return {out: ctx.name for out in ctx.stage.outputs}

    stages = [_stage("root", [], ["a"], recorder), _stage("left", ["a"], ["b"], recorder),
              _stage("right", ["a"], ["c"], recorder), _stage("join", ["b", "c"], ["d"], recorder)]
    Pipeline(stages, tmp_path / "one", workers=1).run()
    Pipeline(stages, tmp_path / "two", workers=2).run()
    one = (tmp_path / "one" / "gates.json").read_bytes()
    assert one == (tmp_path / "two" / "gates.json").read_bytes()
    assert [g["stage"] for g in json.loads(one)] == ["root", "left", "right", "join"]


def test_write_gates_replaces_the_file_and_a_later_record_appends_to_it(tmp_path):
    def writer(ctx, inputs):
        ctx.write_gates([GateResult(stage="parses", passed=True), GateResult(stage="confined", passed=True)])
        return {"bodies": {}}

    def recorder(ctx, inputs):
        ctx.record_gate(GateResult(stage="intent", passed=False, failures=["no"]))
        return {"intents": {}}

    (tmp_path / "gates.json").write_text(json.dumps([{"stage": "mine", "pass": True}]), encoding="utf-8")
    result = Pipeline([_stage("tools", [], ["bodies"], writer), _stage("intent", ["bodies"], ["intents"], recorder)],
                      tmp_path).run()
    assert [g["stage"] for g in json.loads((tmp_path / "gates.json").read_text())] == ["parses", "confined", "intent"]
    assert result.reports["tools"].rulings == ["parses", "confined"] and result.rulings == ["parses", "confined", "intent"]


# --- the anchor and the ceiling under the scheduler --------------------------

def test_the_anchor_is_drawn_when_the_tasks_land_and_every_builder_stage_waits_for_it(tmp_path):
    ran = []
    task = Task(id="t1", name="one", run_ids=["r1", "r2", "r3", "r4", "r5"])

    def body(ctx, inputs):
        ran.append(ctx.name)
        return {out: ctx.name for out in ctx.stage.outputs}

    def cluster(ctx, inputs):
        ran.append("cluster")
        return {"tasks": [task]}

    def builder(ctx, inputs):
        ran.append("tools")
        return {"bodies": ctx.seed_runs("t1", task.run_ids)}

    stages = [_stage("ingest", [], ["traces"], body), Stage("cluster", cluster, ["traces"], ["tasks"], code_version="c"),
              _stage("canon", ["traces"], ["rules"], body), Stage("tools", builder, ["traces"], ["bodies"], builder=True,
                                                                  code_version="t")]
    pipe = Pipeline(stages, tmp_path, anchor_from="tasks")
    assert pipe.anchor is None
    result = pipe.run()
    assert result.status == "complete"
    assert ran.index("cluster") < ran.index("canon") and ran.index("cluster") < ran.index("tools")
    assert isinstance(pipe.anchor, Anchor) and pipe.anchor.held_out["t1"]
    assert set(result.artifacts["bodies"]) == set(task.run_ids) - set(pipe.anchor.held_out["t1"])
    assert Path(tmp_path / "anchor.json").is_file()


def test_a_budget_stop_inside_a_stage_still_stops_the_build_at_two_workers(tmp_path):
    def spender(ctx, inputs):
        raise BudgetStop({"stage": ctx.name, "item": "tool_1", "spent": 2.0})

    stages = [_stage("root", [], ["a"]), _stage("left", ["a"], ["b"], spender), _stage("right", ["a"], ["c"]),
              _stage("join", ["b", "c"], ["d"])]
    result = Pipeline(stages, tmp_path, workers=2).run()
    assert result.status == "stopped" and result.stopped["item"] == "tool_1"
    assert result.statuses["left"] == "stopped" and result.statuses["join"] == "pending"
    assert "d" not in result.artifacts
