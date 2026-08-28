"""Toy stages over the DAG runner: order, caching, the rollback edge, the anchor, and the budget stop."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from harness.runner.pipeline import (
    ANCHOR_MIN_RUNS,
    ANCHOR_SHARE,
    Anchor,
    AnchorLeak,
    BudgetStop,
    CycleError,
    GraphError,
    Pipeline,
    PipelineError,
    Stage,
    choose_anchor,
    code_hash,
    load_anchor,
)
from harness.shared.records import Category, GateResult, Task

# --- toy stages -------------------------------------------------------------

def make_counter():
    calls = []

    def fn(ctx, inputs):
        calls.append({"attempt": ctx.attempt, "failure": ctx.failure, "inputs": dict(inputs)})
        return {}

    fn.calls = calls
    return fn


def stage_double(ctx, inputs):
    return {"doubled": inputs["number"] * 2}


def stage_plus_one(ctx, inputs):
    return {"final": inputs["doubled"] + 1}


def passing_gate(ctx, outputs):
    return GateResult(stage="always_ok", **{"pass": True})


def failing_gate(ctx, outputs):
    return GateResult(stage="replay_fidelity", **{"pass": False}, failures=["writes differ"])


# --- graph ------------------------------------------------------------------

def test_runs_stages_in_topological_order(workdir):
    seen = []

    def first(ctx, inputs):
        seen.append("first")
        return {"a": 1}

    def second(ctx, inputs):
        seen.append("second")
        return {"b": inputs["a"] + 1}

    def third(ctx, inputs):
        seen.append("third")
        return {"c": inputs["b"] + 1}

    pipe = Pipeline(
        [
            Stage("third", third, inputs=["b"], outputs=["c"]),
            Stage("first", first, inputs=[], outputs=["a"]),
            Stage("second", second, inputs=["a"], outputs=["b"]),
        ],
        workdir=workdir,
    )
    result = pipe.run()
    assert seen == ["first", "second", "third"]
    assert result.status == "complete"
    assert result.artifacts["c"] == 3
    assert result.statuses == {"first": "ran", "second": "ran", "third": "ran"}


def test_cycle_is_refused(workdir):
    stages = [
        Stage("a", stage_double, inputs=["final"], outputs=["doubled"]),
        Stage("b", stage_plus_one, inputs=["doubled"], outputs=["final"]),
    ]
    with pytest.raises(CycleError):
        Pipeline(stages, workdir=workdir)


def test_duplicate_names_and_outputs_are_refused(workdir):
    with pytest.raises(GraphError):
        Pipeline(
            [Stage("a", stage_double, outputs=["x"]), Stage("a", stage_plus_one, outputs=["y"])],
            workdir=workdir,
        )
    with pytest.raises(GraphError):
        Pipeline(
            [Stage("a", stage_double, outputs=["x"]), Stage("b", stage_plus_one, outputs=["x"])],
            workdir=workdir,
        )


def test_missing_input_is_refused_at_run(workdir):
    pipe = Pipeline([Stage("a", stage_double, inputs=["number"], outputs=["doubled"])], workdir=workdir)
    with pytest.raises(GraphError):
        pipe.run()
    result = pipe.run({"number": 4})
    assert result.artifacts["doubled"] == 8


# --- caching (design section 8) ---------------------------------------------

def test_second_run_is_a_cache_hit(workdir):
    fn = make_counter()

    def counted(ctx, inputs):
        fn(ctx, inputs)
        return {"out": inputs["number"] * 2}

    stages = [Stage("double", counted, inputs=["number"], outputs=["out"], code_version="v1")]
    first = Pipeline(stages, workdir=workdir).run({"number": 3})
    second = Pipeline(stages, workdir=workdir).run({"number": 3})
    assert first.statuses["double"] == "ran"
    assert second.statuses["double"] == "cached"
    assert second.artifacts["out"] == 6
    assert len(fn.calls) == 1


def test_changed_input_and_changed_code_both_miss_the_cache(workdir):
    fn = make_counter()

    def counted(ctx, inputs):
        fn(ctx, inputs)
        return {"out": inputs["number"] * 2}

    Pipeline([Stage("d", counted, ["number"], ["out"], code_version="v1")], workdir=workdir).run({"number": 3})
    Pipeline([Stage("d", counted, ["number"], ["out"], code_version="v1")], workdir=workdir).run({"number": 4})
    assert len(fn.calls) == 2
    Pipeline([Stage("d", counted, ["number"], ["out"], code_version="v2")], workdir=workdir).run({"number": 4})
    assert len(fn.calls) == 3


def test_records_survive_the_cache(workdir):
    def make_task(ctx, inputs):
        return {"task": Task(id="t1", run_ids=["r1", "r2"], intent="cancel an order")}

    stages = [Stage("cluster", make_task, [], ["task"], code_version="v1")]
    Pipeline(stages, workdir=workdir).run()
    second = Pipeline(stages, workdir=workdir).run()
    assert second.statuses["cluster"] == "cached"
    task = second.artifacts["task"]
    assert isinstance(task, Task)
    assert task.run_ids == ["r1", "r2"]


def test_records_nested_in_a_dict_and_a_mixed_list_survive_the_cache(workdir):
    """A cache hit hands back what the stage returned, not the string of its repr."""

    def make(ctx, inputs):
        return {"by_id": {"t1": Task(id="t1", run_ids=["r1"], intent="cancel an order")},
                "mixed": [Task(id="t1", run_ids=["r1"]), Category(id="c1", task_ids=["t1"])]}

    stages = [Stage("cluster", make, [], ["by_id", "mixed"], code_version="v1")]
    Pipeline(stages, workdir=workdir).run()
    second = Pipeline(stages, workdir=workdir).run()
    assert second.statuses["cluster"] == "cached"
    task = second.artifacts["by_id"]["t1"]
    assert isinstance(task, Task) and task.run_ids == ["r1"]
    assert [type(item).__name__ for item in second.artifacts["mixed"]] == ["Task", "Category"]
    assert second.artifacts["mixed"][1].task_ids == ["t1"]


@pytest.mark.parametrize("value", [Path("/tmp/x"), {1, 2}, (1, 2), {1: "a"}])
def test_an_artifact_the_cache_cannot_round_trip_is_refused(workdir, value):
    """Better a refusal at write time than a second build that reads a different world."""

    def make(ctx, inputs):
        return {"out": value}

    with pytest.raises(PipelineError) as caught:
        Pipeline([Stage("s", make, [], ["out"], code_version="v1")], workdir=workdir).run()
    assert "s" in str(caught.value) and "out" in str(caught.value)


def test_a_model_that_is_not_a_record_is_refused_at_write_time(workdir):
    from pydantic import BaseModel

    class Foreign(BaseModel):
        x: int = 1

    def make(ctx, inputs):
        return {"out": Foreign()}

    with pytest.raises(PipelineError) as caught:
        Pipeline([Stage("s", make, [], ["out"], code_version="v1")], workdir=workdir).run()
    assert "Foreign" in str(caught.value)


def test_two_functions_with_one_stage_name_do_not_share_a_cache_entry(workdir):
    def fn_a(ctx, inputs):
        return {"out": "A"}

    def fn_b(ctx, inputs):
        return {"out": "B"}

    Pipeline([Stage("s", fn_a, [], ["out"])], workdir=workdir).run()
    second = Pipeline([Stage("s", fn_b, [], ["out"])], workdir=workdir).run()
    assert second.artifacts["out"] == "B"
    assert second.statuses["s"] == "ran"


def test_two_partials_of_one_function_do_not_share_a_cache_entry(workdir):
    def scale(factor, ctx, inputs):
        return {"out": 3 * factor}

    Pipeline([Stage("s", functools.partial(scale, 2), [], ["out"])], workdir=workdir).run()
    second = Pipeline([Stage("s", functools.partial(scale, 10), [], ["out"])], workdir=workdir).run()
    assert second.artifacts["out"] == 30
    assert second.statuses["s"] == "ran"


def test_a_function_with_no_findable_source_is_refused_rather_than_hashed_as_a_constant(workdir):
    class Callable_:
        def __call__(self, ctx, inputs):
            return {"out": 1}

    with pytest.raises(PipelineError) as caught:
        Pipeline([Stage("s", Callable_(), [], ["out"])], workdir=workdir).run()
    assert "code_version" in str(caught.value)
    assert Pipeline([Stage("s", Callable_(), [], ["out"], code_version="v1")],
                    workdir=workdir).run().status == "complete"


def test_the_code_hash_of_one_function_is_not_moved_by_an_edit_elsewhere_in_its_file(tmp_path):
    """A stage with no code_version must be busted by its own edits, not by every other edit in
    the module that defines it: build.py holds 11 stages in one 500-line file."""
    import importlib.util
    import sys

    def load(name, path, other_body):
        path.write_text(
            "def stage(ctx, inputs):\n    return {}\n\n\n"
            f"def other(ctx, inputs):\n    {other_body}\n",
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Same module name both times, so __module__ agrees and only the file's own bytes differ;
    # that is exactly the "one stage edited elsewhere in the same build.py" scenario.
    before = load("stage_module_under_test", tmp_path / "a.py", "return {'x': 1}")
    after = load("stage_module_under_test", tmp_path / "b.py", "return {'x': 2}")

    assert code_hash(Stage("s", before.stage, [], ["out"])) == code_hash(Stage("s", after.stage, [], ["out"]))


def test_a_declared_input_path_moves_the_cache_key(workdir):
    raw = Path(workdir) / "raw.txt"
    raw.write_text("one", encoding="utf-8")

    def ingest(ctx, inputs):
        return {"text": (Path(ctx.workdir) / "raw.txt").read_text(encoding="utf-8")}

    stages = [Stage("ingest", ingest, [], ["text"], code_version="v1", input_paths=["raw.txt"])]
    assert Pipeline(stages, workdir=workdir).run().artifacts["text"] == "one"
    raw.write_text("two", encoding="utf-8")
    second = Pipeline(stages, workdir=workdir).run()
    assert second.artifacts["text"] == "two"
    assert second.statuses["ingest"] == "ran"


def test_the_anchor_is_part_of_the_cache_key(workdir):
    def report_stage(ctx, inputs):
        return {"held": ctx.anchor.anchor_runs("t")}

    stages = [Stage("report", report_stage, [], ["held"], code_version="v1")]
    first = Anchor(held_out={"t": ["r1"]}, unguarded=[])
    second = Anchor(held_out={"t": ["r9"]}, unguarded=[])
    assert Pipeline(stages, workdir=workdir, anchor=first).run().artifacts["held"] == ["r1"]
    later = Pipeline(stages, workdir=workdir, anchor=second).run()
    assert later.artifacts["held"] == ["r9"]
    assert later.statuses["report"] == "ran"


def test_cache_keys_do_not_depend_on_run_order(workdir):
    """Same inputs, same code, different position in the graph: still one cache entry."""

    def head(ctx, inputs):
        return {"n": 2}

    def body(ctx, inputs):
        return {"out": inputs["n"] * 2}

    Pipeline([Stage("body", body, ["n"], ["out"], code_version="v1")], workdir=workdir).run({"n": 2})
    result = Pipeline(
        [Stage("body", body, ["n"], ["out"], code_version="v1"), Stage("head", head, [], ["n"])],
        workdir=workdir,
    ).run()
    assert result.statuses["body"] == "cached"


# --- the one rollback edge --------------------------------------------------

def test_failed_gate_rolls_back_to_the_generating_stage(workdir):
    fn = make_counter()
    attempts = []

    def flaky(ctx, inputs):
        fn(ctx, inputs)
        attempts.append(ctx.attempt)
        return {"out": ctx.attempt}

    def gate(ctx, outputs):
        ok = outputs["out"] >= 2
        return GateResult(stage="replay_fidelity", **{"pass": ok}, failures=[] if ok else ["writes differ"])

    result = Pipeline(
        [Stage("compile_tools", flaky, [], ["out"], gate=gate, code_version="v1")], workdir=workdir
    ).run()
    assert result.status == "complete"
    assert attempts == [1, 2]
    assert result.statuses["compile_tools"] == "rolled_back"
    assert "compile_tools: attempt 2 of 3, gate replay_fidelity failed" in result.log
    assert fn.calls[1]["failure"] == "writes differ"


def test_three_failed_attempts_fail_the_pipeline(workdir):
    ran = []

    def never_good(ctx, inputs):
        ran.append(ctx.attempt)
        return {"out": 0}

    def after(ctx, inputs):
        ran.append("after")
        return {"done": True}

    result = Pipeline(
        [
            Stage("compile_tools", never_good, [], ["out"], gate=failing_gate, code_version="v1"),
            Stage("after", after, ["out"], ["done"], code_version="v1"),
        ],
        workdir=workdir,
    ).run()
    assert result.status == "failed"
    assert result.failed_stage == "compile_tools"
    assert ran == [1, 2, 3]
    assert result.statuses["after"] == "pending"
    assert "compile_tools: attempt 2 of 3, gate replay_fidelity failed" in result.log
    assert "compile_tools: attempt 3 of 3, gate replay_fidelity failed" in result.log
    assert [g.stage for g in result.gates] == ["replay_fidelity"] * 3


def test_rollback_ignores_the_cache(workdir):
    calls = []

    def counted(ctx, inputs):
        calls.append(ctx.attempt)
        return {"out": 1}

    Pipeline([Stage("s", counted, [], ["out"], gate=passing_gate, code_version="v1")], workdir=workdir).run()
    assert calls == [1]
    calls.clear()
    result = Pipeline([Stage("s", counted, [], ["out"], gate=failing_gate, code_version="v1")], workdir=workdir).run()
    assert calls == [2, 3], "a cached artifact whose gate fails is regenerated, not reused"
    assert result.status == "failed"


def test_passing_gate_is_recorded(workdir):
    result = Pipeline(
        [Stage("s", stage_double, ["number"], ["doubled"], gate=passing_gate, code_version="v1")],
        workdir=workdir,
    ).run({"number": 2})
    assert result.status == "complete"
    assert len(result.gates) == 1 and result.gates[0].passed is True


# --- the anchor (D81) -------------------------------------------------------

def test_anchor_share_floor_and_unguarded(workdir):
    tasks = [
        Task(id="big", run_ids=[f"r{i}" for i in range(10)]),
        Task(id="small", run_ids=["a", "b", "c"]),
        Task(id="tiny", run_ids=["x", "y"]),
    ]
    anchor = choose_anchor(tasks, workdir)
    assert len(anchor.anchor_runs("big")) == 2
    assert len(anchor.anchor_runs("small")) == 1
    assert anchor.anchor_runs("tiny") == []
    assert anchor.unguarded == ["tiny"]
    assert anchor.share == ANCHOR_SHARE and anchor.min_runs == ANCHOR_MIN_RUNS


def test_anchor_is_chosen_once_and_stored(workdir):
    tasks = [Task(id="t", run_ids=[f"r{i}" for i in range(10)])]
    first = choose_anchor(tasks, workdir, seed=1)
    again = choose_anchor(tasks, workdir, seed=999)
    assert again.anchor_runs("t") == first.anchor_runs("t")
    assert load_anchor(workdir).anchor_runs("t") == first.anchor_runs("t")
    stored = json.loads((Path(workdir) / "anchor.json").read_text(encoding="utf-8"))
    assert stored["held_out"]["t"] == first.anchor_runs("t")


def test_anchor_is_deterministic_and_task_order_free(workdir, tmp_path):
    tasks = [Task(id="t", run_ids=[f"r{i}" for i in range(10)]), Task(id="u", run_ids=list("abcdefgh"))]
    other = tmp_path / "other"
    other.mkdir()
    a = choose_anchor(tasks, workdir, seed=7)
    b = choose_anchor(list(reversed(tasks)), other, seed=7)
    assert a.held_out == b.held_out


def test_task_records_are_marked_with_their_anchor(workdir):
    tasks = [Task(id="t", run_ids=[f"r{i}" for i in range(5)]), Task(id="u", run_ids=["a", "b"])]
    anchor = choose_anchor(tasks, workdir)
    marked = anchor.mark(tasks)
    assert marked[0].anchor_run_ids == anchor.anchor_runs("t")
    assert marked[0].unguarded is False
    assert marked[1].unguarded is True and marked[1].anchor_run_ids == []
    assert tasks[0].anchor_run_ids == [], "mark returns copies, it does not edit in place"


def test_builder_stage_cannot_read_the_anchor_but_later_stages_can(workdir):
    tasks = [Task(id="t", run_ids=[f"r{i}" for i in range(10)])]
    anchor = choose_anchor(tasks, workdir)
    held = anchor.anchor_runs("t")
    seen = {}

    def builder_stage(ctx, inputs):
        with pytest.raises(AnchorLeak):
            _ = ctx.anchor
        seen["seeds"] = ctx.seed_runs("t", [f"r{i}" for i in range(10)])
        return {"env": "built"}

    def report_stage(ctx, inputs):
        seen["anchor"] = ctx.anchor.anchor_runs("t")
        return {"report": "written"}

    Pipeline(
        [
            Stage("compile_env", builder_stage, [], ["env"], builder=True),
            Stage("report", report_stage, ["env"], ["report"]),
        ],
        workdir=workdir,
        anchor=anchor,
    ).run()
    assert seen["anchor"] == held
    assert set(seen["seeds"]).isdisjoint(held)
    assert len(seen["seeds"]) == 8


def test_a_task_that_appears_after_the_anchor_gets_its_own_held_out_share(workdir):
    """An iterate build or a split must not hand the Builder a Task with nothing held out (D81)."""
    first = choose_anchor([Task(id="t", run_ids=[f"r{i}" for i in range(10)])], workdir)
    later = choose_anchor(
        [Task(id="t", run_ids=[f"r{i}" for i in range(10)]),
         Task(id="new", run_ids=[f"n{i}" for i in range(10)])], workdir)
    assert later.anchor_runs("t") == first.anchor_runs("t"), "an existing Task keeps its exact anchor"
    assert len(later.anchor_runs("new")) == 2
    assert load_anchor(workdir).anchor_runs("new") == later.anchor_runs("new")
    marked = later.mark([Task(id="new", run_ids=[f"n{i}" for i in range(10)])])
    assert marked[0].unguarded is False
    assert set(marked[0].anchor_run_ids) == set(later.anchor_runs("new"))


def test_a_task_the_anchor_never_saw_is_marked_unguarded(workdir):
    anchor = choose_anchor([Task(id="t", run_ids=[f"r{i}" for i in range(10)])], workdir)
    marked = anchor.mark([Task(id="stranger", run_ids=[f"s{i}" for i in range(10)])])
    assert marked[0].unguarded is True
    assert marked[0].anchor_run_ids == []


def test_mark_takes_tasks_as_dicts_the_way_the_rest_of_the_module_does(workdir):
    anchor = choose_anchor([{"id": "t", "run_ids": [f"r{i}" for i in range(10)]}], workdir)
    marked = anchor.mark([{"id": "t", "run_ids": [f"r{i}" for i in range(10)]}])
    assert isinstance(marked[0], Task)
    assert marked[0].anchor_run_ids == anchor.anchor_runs("t")
    assert marked[0].unguarded is False


def test_a_builder_stage_with_no_anchor_is_refused(workdir):
    seen = {}

    def builder_stage(ctx, inputs):
        seen["seeds"] = ctx.seed_runs("t", ["r1", "r2"])
        return {"env": "built"}

    with pytest.raises(PipelineError) as caught:
        Pipeline([Stage("compile_env", builder_stage, [], ["env"], builder=True)], workdir=workdir).run()
    assert "anchor" in str(caught.value)
    assert seen == {}, "the stage never ran, so nothing was built from every Run"


def test_seed_runs_without_an_anchor_is_refused(workdir):
    def stage_fn(ctx, inputs):
        return {"seeds": ctx.seed_runs("t", ["r1", "r2"])}

    with pytest.raises(PipelineError):
        Pipeline([Stage("s", stage_fn, [], ["seeds"])], workdir=workdir).run()


def test_seed_runs_refuses_a_run_list_that_is_all_anchor(workdir):
    tasks = [Task(id="t", run_ids=[f"r{i}" for i in range(10)])]
    anchor = choose_anchor(tasks, workdir)
    assert anchor.seed_runs("t", anchor.anchor_runs("t")) == []
    assert anchor.is_held_out(anchor.anchor_runs("t")[0]) is True
    assert anchor.is_held_out("not-a-run") is False


# --- budget ceiling (D86) ---------------------------------------------------

class FakeCeiling:
    """The duck type the Pipeline needs from budget.Ceiling: remaining, add, report."""

    def __init__(self, usd):
        self.usd = usd
        self.spent = 0.0

    @property
    def remaining(self):
        return max(0.0, self.usd - self.spent)

    def add(self, usd, stage, item, items_left=0):
        self.spent += usd
        if self.spent >= self.usd:
            raise BudgetStop(self.report(stage, item, items_left))
        return self.remaining

    def report(self, stage, item, items_left=0):
        return {
            "stage": stage,
            "item": item,
            "spent": self.spent,
            "ceiling_usd": self.usd,
            "remaining": self.remaining,
            "items_left": items_left,
            "estimate_to_finish": 1.0,
        }


def test_ceiling_inside_a_stage_stops_and_reports(workdir):
    reached = []

    def spender(ctx, inputs):
        for i in range(10):
            ctx.charge(0.4, item=f"tool_{i}")
            reached.append(i)
        return {"env": "built"}

    def later(ctx, inputs):
        reached.append("later")
        return {"report": True}

    result = Pipeline(
        [Stage("compile_env", spender, [], ["env"]), Stage("report", later, ["env"], ["report"])],
        workdir=workdir,
        ceiling=FakeCeiling(1.0),
    ).run()
    assert result.status == "stopped"
    assert reached == [0, 1]
    assert result.stopped["stage"] == "compile_env"
    assert result.stopped["item"] == "tool_2"
    assert result.statuses["compile_env"] == "stopped"
    assert result.statuses["report"] == "pending"
    assert "env" not in result.artifacts


def test_ceiling_is_checked_between_stages(workdir):
    ran = []

    def first(ctx, inputs):
        ran.append("first")
        return {"a": 1}

    def second(ctx, inputs):
        ran.append("second")
        return {"b": 2}

    ceiling = FakeCeiling(1.0)
    ceiling.spent = 1.0
    result = Pipeline(
        [Stage("first", first, [], ["a"]), Stage("second", second, ["a"], ["b"])],
        workdir=workdir,
        ceiling=ceiling,
    ).run()
    assert ran == []
    assert result.status == "stopped"
    assert result.stopped["stage"] == "first"


def test_real_budget_ceiling_stops_the_pipeline(workdir):
    budget = pytest.importorskip("harness.shared.budget")

    def spender(ctx, inputs):
        for i in range(10):
            ctx.charge(0.6, item=f"tool_{i}")
        return {"env": "built"}

    result = Pipeline([Stage("compile_env", spender, [], ["env"])], workdir=workdir,
                      ceiling=budget.Ceiling(usd=1.0)).run()
    assert result.status == "stopped"
    assert result.stopped["stage"] == "compile_env"
    assert result.stopped["spent"] == pytest.approx(1.2)


def test_charge_without_a_ceiling_is_a_no_op(workdir):
    def spender(ctx, inputs):
        ctx.charge(100.0, item="whatever")
        return {"a": 1}

    assert Pipeline([Stage("s", spender, [], ["a"])], workdir=workdir).run().status == "complete"


# --- mermaid ----------------------------------------------------------------

def test_mermaid_before_and_after_a_run(workdir):
    stages = [
        Stage("ingest", lambda ctx, i: {"traces": [1]}, [], ["traces"], code_version="v1"),
        Stage("mine", lambda ctx, i: {"sigs": []}, ["traces"], ["sigs"], gate=passing_gate, code_version="v1"),
    ]
    pipe = Pipeline(stages, workdir=workdir)
    before = pipe.to_mermaid()
    assert before.startswith("flowchart TD")
    assert "ingest" in before and "mine" in before
    assert "pending" in before
    assert "ingest --> mine" in before

    pipe.run()
    after = pipe.to_mermaid()
    assert "ran" in after
    assert "pending" not in after

    cached = Pipeline(stages, workdir=workdir)
    cached.run()
    assert "cached" in cached.to_mermaid()


def test_mermaid_shows_the_rollback_edge_and_failure(workdir):
    pipe = Pipeline(
        [Stage("compile_tools", lambda ctx, i: {"out": 0}, [], ["out"], gate=failing_gate)],
        workdir=workdir,
    )
    pipe.run()
    text = pipe.to_mermaid()
    assert "failed" in text
    assert "-." in text and "replay_fidelity" in text
    assert "attempt 3 of 3" in text


def test_mermaid_marks_a_stopped_stage(workdir):
    def spender(ctx, inputs):
        ctx.charge(2.0, item="tool_0")
        return {"a": 1}

    pipe = Pipeline([Stage("s", spender, [], ["a"])], workdir=workdir, ceiling=FakeCeiling(1.0))
    pipe.run()
    assert "stopped" in pipe.to_mermaid()


# --- state on disk and module hygiene ---------------------------------------

def test_run_state_is_written_to_the_workdir(workdir):
    Pipeline([Stage("s", stage_double, ["number"], ["doubled"])], workdir=workdir).run({"number": 1})
    state = json.loads((Path(workdir) / "pipeline" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["statuses"]["s"] == "ran"
    assert state["log"] == []


def test_a_crashing_stage_still_leaves_state_on_disk(workdir):
    """A build that died has to say where, so the report can show it (D86, section 4 item 18)."""

    def boom(ctx, inputs):
        raise RuntimeError("the tool table is empty")

    pipe = Pipeline([Stage("first", lambda ctx, i: {"a": 1}, [], ["a"], code_version="v1"),
                     Stage("mine", boom, ["a"], ["sigs"], code_version="v1")], workdir=workdir)
    with pytest.raises(RuntimeError):
        pipe.run()
    state = json.loads((Path(workdir) / "pipeline" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "crashed"
    assert state["failed_stage"] == "mine"
    assert state["statuses"]["first"] == "ran"
    assert any("the tool table is empty" in line for line in state["log"])
    assert "crashed" in state["mermaid"]


def test_mermaid_escapes_a_quote_in_a_stage_or_gate_name(workdir):
    def quoted_gate(ctx, outputs):
        return GateResult(stage='replay "fidelity"', **{"pass": False}, failures=["writes differ"])

    pipe = Pipeline([Stage('compile "tools"', lambda ctx, i: {"out": 0}, [], ["out"], gate=quoted_gate)],
                    workdir=workdir)
    pipe.run()
    text = pipe.to_mermaid()
    assert '"' not in text.replace('["', "").replace('"]', "").replace('-. "', "").replace('" .->', "")
    assert "#quot;tools#quot;" in text
    assert "#quot;fidelity#quot;" in text


def test_pipeline_does_not_import_the_builder():
    """Build brief rule 7 and D89, checked with the gate rather than a substring scan."""
    from harness.runner import validate

    src = Path(__file__).resolve().parents[1] / "src"
    gate = validate.import_boundary_check(src)
    assert gate.passed, gate.failures
    assert not any("pipeline.py" in failure for failure in gate.failures)


def test_the_boundary_gate_would_catch_a_dynamic_builder_import_from_pipeline(tmp_path: Path):
    """The guard above is only worth something if the gate catches what it is guarding against."""
    from harness.runner import validate

    root = tmp_path / "harness"
    (root / "runner").mkdir(parents=True)
    source = (Path(__file__).resolve().parents[1] / "src" / "harness" / "runner" / "pipeline.py").read_text(
        encoding="utf-8")
    extra = "\n\ndef sneak():\n    import importlib\n    return importlib.import_module('harness' + '.builder')\n"
    (root / "runner" / "pipeline.py").write_text(source + extra, encoding="utf-8")
    assert validate.import_boundary_check(root).passed is False
