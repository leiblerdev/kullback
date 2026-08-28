"""report.py renders the one Markdown report: Environment first, then Tasks, then the queues (D85, D92, D96)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.shared.records import (
    Atom,
    Constraint,
    Environment,
    GateResult,
    OverlayRow,
    Run,
    Task,
    TaskOverlay,
    Verdict,
    Verifier,
    as_dict,
)
from harness.shared.report import (
    SECTIONS,
    ReportData,
    ScorecardItem,
    SetAsideLesson,
    StageStatus,
    TaskCoverage,
    load,
    pipeline_dag,
    render,
    suggestion,
    task_numbers,
    write_report,
)

# --- hand-built records -----------------------------------------------------

def a_verdict(run_id: str, passed: bool, **kwargs) -> Verdict:
    body = {
        "run_id": run_id,
        "env_id": "env-1",
        "pass": passed,
        "class": "pass" if passed else "fail",
    }
    body.update(kwargs)
    return Verdict(**body)


def a_run(run_id: str, model: str, **kwargs) -> Run:
    return Run(run_id=run_id, task_id="t1", env_id="env-1", model=model, **kwargs)


@pytest.fixture
def data() -> ReportData:
    """One built Environment, one Task with four graded Runs and one assisted Run."""
    verifier = Verifier(
        task_id="t1",
        verifier_version="v2",
        atoms=[
            Atom(id="a_cancel", kind="required", predicate_src="True"),
            Atom(id="a_tone", kind="communicate", judge=True),
            Atom(id="a_refund", kind="forbidden", predicate_src="False"),
        ],
    )
    return ReportData(
        title="Harness build report",
        kind="build",
        built=True,
        environment=Environment(
            env_id="env-1",
            version=3,
            schema_version="s1",
            tools_version="t2",
            policy_version="p3",
            assisted_tools=["search_products"],
        ),
        gates=[
            GateResult(stage="ingest", **{"pass": True}, metrics={"traces": 3}),
            GateResult(stage="replay_fidelity", **{"pass": False}, failures=["writes differ on orders"]),
        ],
        scorecard=[
            ScorecardItem(name="replay fidelity", raw="96%", explained="100%", note="4 misses explained"),
            ScorecardItem(name="verdict agreement", raw="90%", explained="95%"),
        ],
        stages=[
            StageStatus(name="ingest", status="ran"),
            StageStatus(name="mine", status="ran"),
            StageStatus(name="compile_tools", status="failed", gate="replay_fidelity", attempts=3),
        ],
        tasks=[
            Task(
                id="t1", name="cancel an order", intent="cancel the pending order",
                run_ids=["r1", "r2", "r3", "r4", "r5"],
            )
        ],
        verifiers=[verifier],
        runs=[
            a_run("r1", "frontier-model"),
            a_run("r2", "frontier-model"),
            a_run("r3", "candidate-model"),
            a_run("r4", "candidate-model"),
            a_run("r5", "candidate-model", assisted=True),
        ],
        verdicts=[
            a_verdict("r1", True),
            a_verdict("r2", False, failing_atom="a_cancel", cause="candidate"),
            a_verdict("r3", True),
            a_verdict("r4", True, judge_used=True),
            a_verdict("r5", False, failing_atom="a_refund", cause="environment", environment_suspected=True),
        ],
        frontier_models=["frontier-model"],
        overlays=[TaskOverlay(task_id="t1", rows=[
            OverlayRow(table="orders", id="W1", version_hash="h1"),
            OverlayRow(table="users", id="U1", version_hash="h2"),
        ])],
        policy_items=[
            Constraint(id="p1", text="never cancel a delivered order"),
            Constraint(id="p2", text="always ask before a refund"),
            Constraint(id="p3", text="never disclose another user's address"),
        ],
        policy_exercised=["p1"],
        task_coverage=[
            TaskCoverage(task_id="t1", covered=False, reason="1 Run assisted on search_products", run_count=5)
        ],
        judge_disagreement={"pairs": 8, "disagreements": 2, "rate": 0.25},
        disagreement_queue=[
            {"use": "cause", "item_id": "r5", "verdict_a": "environment", "verdict_b": "candidate"},
        ],
        tasks_aside=[{"task_id": "t9", "reason": "reference_disputed"}],
        lessons_set_aside=[
            SetAsideLesson(id="l1", pattern="a tool returning a list needs a paging field",
                           reason="no tool here returns a list"),
        ],
    )


# --- section order ----------------------------------------------------------

def test_section_order_is_environment_tasks_queue_lessons(data):
    text = render(data)
    headings = [line.strip() for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)


def test_the_environment_section_comes_before_any_task(data):
    text = render(data)
    assert text.index("## Environment") < text.index("## Tasks") < text.index("### Task t1")


def test_the_title_is_the_first_line(data):
    assert render(data).splitlines()[0] == "# Harness build report"


# --- environment section ----------------------------------------------------

def test_built_or_not_built_is_the_first_thing_said(data):
    assert "Environment built: yes" in render(data)
    data.built = False
    data.stopped_reason = "spend ceiling reached in compile_tools on search_products"
    text = render(data)
    assert "Environment built: no" in text
    assert "spend ceiling reached" in text


def test_gates_are_listed_with_their_failures(data):
    text = render(data)
    assert "replay_fidelity" in text
    assert "writes differ on orders" in text


def test_scorecard_shows_raw_and_explained_side_by_side(data):
    row = [line for line in render(data).splitlines() if line.startswith("| replay fidelity")][0]
    assert "96%" in row and "100%" in row
    assert row.index("96%") < row.index("100%")


def test_assisted_tools_unguarded_tasks_and_overlay_counts(data):
    data.tasks[0].unguarded = True
    text = render(data)
    assert "search_products" in text
    assert "Unguarded Tasks" in text and "t1" in text
    assert "Overlay rows: 2" in text
    assert "Tasks with an overlay: 1" in text


def test_the_assisted_share_per_tool_is_shown_when_it_is_known(data):
    data.assisted_share = {"search_products": 0.12}
    assert "- search_products: 12% of its calls stood in" in render(data)


def test_task_coverage_gives_both_numbers(data):
    text = render(data)
    assert "0 of 1 Tasks" in text
    assert "1 Run assisted on search_products" in text


def test_policy_coverage_line_lists_the_untested_items(data):
    text = render(data)
    assert "your traces exercise 1 of 3 policy items; the rest are not tested" in text
    assert "always ask before a refund" in text
    assert "never disclose another user's address" in text


def test_the_pipeline_dag_is_mermaid_in_the_environment_section(data):
    text = render(data)
    assert "```mermaid" in text
    assert "flowchart TD" in text
    assert text.index("flowchart TD") < text.index("## Tasks")


def test_pipeline_dag_marks_a_failed_gate_with_a_rollback_edge():
    text = pipeline_dag([
        StageStatus(name="mine", status="ran"),
        StageStatus(name="compile_tools", status="failed", gate="replay_fidelity", attempts=3),
    ])
    assert "flowchart TD" in text
    assert "mine --> compile_tools" in text
    assert "-." in text and "replay_fidelity" in text
    assert "attempt 3 of 3" in text


def test_pipeline_dag_without_stages_says_so(data):
    data.stages = []
    assert "no stages recorded" in render(data)


# --- per task numbers -------------------------------------------------------

def test_numbers_come_before_the_suggestion(data):
    block = render(data).split("### Task t1", 1)[1]
    assert block.index("Runs graded") < block.index("Suggestion")


def test_assisted_runs_are_not_counted(data):
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["runs_graded"] == 4
    assert numbers["assisted_not_counted"] == 1
    assert "Runs not counted (assisted or environment suspected): 1" in render(data)


def test_pass_rates_and_margin(data):
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["frontier_pass_rate"] == 0.5
    assert numbers["candidate_pass_rate"] == 1.0
    assert numbers["margin"] == 0.5
    assert "Margin: +0.50" in render(data)


def test_judge_atoms_carry_the_disagreement_rate(data):
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["judge_atoms"] == 1
    assert "Judge atoms: 1 (judge disagreement rate 25%" in render(data)


def test_failing_atoms_by_class_and_cause_counts(data):
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["failing_atoms"] == {"required": 1}
    assert numbers["causes"] == {"candidate": 1}
    assert "Failing atoms by class: required 1" in render(data)
    assert "Causes: candidate 1" in render(data)


def test_a_task_with_no_verdicts_is_reported_as_not_graded(data):
    data.verdicts = []
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["runs_graded"] == 0
    assert numbers["margin"] is None
    assert "Runs graded: 0" in render(data)


# --- suggestion, never a decision (D85) -------------------------------------

def test_the_suggestion_is_worded_as_a_suggestion(data):
    text = render(data)
    assert "Suggestion: the numbers support routing this Task to the Candidate" in text
    assert "The decision is yours." in text


def test_a_negative_margin_does_not_support_routing(data):
    data.verdicts = [
        a_verdict("r1", True), a_verdict("r2", True),
        a_verdict("r3", False, failing_atom="a_cancel"), a_verdict("r4", False, failing_atom="a_cancel"),
    ]
    text = render(data)
    assert "Suggestion: the numbers do not support routing this Task to the Candidate" in text


def test_no_graded_runs_gives_no_suggestion_either_way(data):
    data.verdicts = []
    assert "say nothing either way" in render(data)


def test_an_unbuilt_environment_suggests_nothing(data):
    data.built = False
    assert "the Environment was not built" in render(data).split("### Task t1", 1)[1]


def test_a_task_with_a_disputed_reference_is_not_gradeable(data):
    data.tasks_aside = [{"task_id": "t1", "reason": "reference_disputed"}]
    block = render(data).split("### Task t1", 1)[1]
    assert "not gradeable, Reference disputed" in block
    assert "a person resolves" in block


@pytest.mark.parametrize("numbers,built,aside", [
    ({"runs_graded": 4, "margin": 0.5}, True, None),
    ({"runs_graded": 4, "margin": -0.5}, True, None),
    ({"runs_graded": 4, "margin": None}, True, None),
    ({"runs_graded": 0, "margin": None}, True, None),
    ({"runs_graded": 4, "margin": 0.5}, False, None),
    ({"runs_graded": 4, "margin": 0.5}, True, "reference_disputed"),
])
def test_suggestion_never_decides(numbers, built, aside):
    """D85: every branch is worded as a suggestion and hands the decision back to the person."""
    said = suggestion(numbers, built=built, aside=aside)
    assert said.startswith("Suggestion: ")
    assert said.endswith(" The decision is yours.")
    assert "route this Task" not in said.replace("routing this Task", "")


# --- queues -----------------------------------------------------------------

def test_the_disagreement_queue_lists_both_verdicts_and_the_tasks_set_aside(data):
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "environment" in block and "candidate" in block
    assert "r5" in block
    assert "t9" in block and "reference_disputed" in block
    assert "8 pairs" in block and "2" in block


def test_an_empty_queue_says_so(data):
    data.disagreement_queue = []
    data.tasks_aside = []
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "nothing in the queue" in block


def test_lessons_set_aside_carry_their_reason(data):
    block = render(data).split("## Lessons set aside", 1)[1]
    assert "no tool here returns a list" in block
    assert "a tool returning a list needs a paging field" in block


def test_no_lessons_set_aside_says_so(data):
    data.lessons_set_aside = []
    assert "no lessons were set aside" in render(data).split("## Lessons set aside", 1)[1]


# --- run batch report -------------------------------------------------------

def test_a_run_batch_report_still_opens_with_the_environment(data):
    data.kind = "batch"
    data.title = "Run batch report"
    text = render(data)
    assert text.splitlines()[0] == "# Run batch report"
    headings = [line.strip() for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)
    assert "Run batch" in text


# --- reading records off disk -----------------------------------------------

def test_load_reads_records_from_a_workdir(workdir: Path, data: ReportData):
    (workdir / "environment.json").write_text(json.dumps(as_dict(data.environment)), encoding="utf-8")
    (workdir / "gates.json").write_text(json.dumps([as_dict(g) for g in data.gates]), encoding="utf-8")
    for folder, records in (("tasks", data.tasks), ("verifiers", data.verifiers),
                            ("verdicts", data.verdicts), ("overlays", data.overlays)):
        (workdir / folder).mkdir()
        for index, record in enumerate(records):
            (workdir / folder / f"{index}.json").write_text(json.dumps(as_dict(record)), encoding="utf-8")
    (workdir / "runs").mkdir()
    for record in data.runs:
        (workdir / "runs" / f"{record.run_id}.json").write_text(json.dumps(as_dict(record)), encoding="utf-8")
    (workdir / "pipeline").mkdir()
    (workdir / "pipeline" / "state.json").write_text(
        json.dumps({"status": "complete", "statuses": {"ingest": "ran", "mine": "ran"}, "log": []}),
        encoding="utf-8",
    )
    (workdir / "disagreement_queue.jsonl").write_text(
        json.dumps({"use": "cause", "item_id": "r5", "verdict_a": "a", "verdict_b": "b"}) + "\n", encoding="utf-8")
    (workdir / "tasks_aside.jsonl").write_text(
        json.dumps({"task_id": "t9", "reason": "reference_disputed"}) + "\n", encoding="utf-8")

    loaded = load(workdir)
    assert loaded.environment.env_id == "env-1"
    assert [t.id for t in loaded.tasks] == ["t1"]
    assert len(loaded.verdicts) == 5
    assert len(loaded.runs) == 5
    assert [s.name for s in loaded.stages] == ["ingest", "mine"]
    assert loaded.disagreement_queue and loaded.tasks_aside
    assert loaded.built is True
    text = render(loaded)
    assert text.startswith("# ")


def test_load_counts_judge_pairs_the_same_way_judge_py_does(workdir: Path):
    """Row 20: load() shares judge.py's disagreement_stats, so abstains reach the report (D92).

    Before this fix load() recomputed only pairs/disagreements/rate by hand and never set abstains,
    so the abstention line in _queue() never fired even though a judge_pairs.jsonl carried abstains.
    """
    from harness.runner.judge import disagreement_rate

    (workdir / "judge_pairs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [
            {"disagreement": True}, {"disagreement": False}, {"abstain": True}, {"abstain": True},
        ]) + "\n",
        encoding="utf-8",
    )
    loaded = load(workdir)
    assert loaded.judge_disagreement == disagreement_rate(workdir)
    assert loaded.judge_disagreement["abstains"] == 2
    assert loaded.judge_disagreement["abstain_rate"] == 0.5
    assert "Judge abstention: 2 of 4 pairs (50%)" in render(loaded)


def test_load_on_an_empty_workdir_still_renders(workdir: Path):
    text = render(load(workdir))
    assert "Environment built: no" in text
    headings = [line.strip() for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)


def test_write_report_writes_markdown(workdir: Path, data: ReportData):
    path = write_report(data, workdir)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("# Harness build report")


# --- D70 flagged tools, D74 per-Task overlay rows, D96 coverage rows ---------

def test_the_report_counts_verdicts_that_rest_on_a_flagged_tool(data):
    """D70: a customer sees how much of a pass rate rests on tools nobody has confirmed."""
    from harness.shared.records import Event, ToolSig
    from harness.shared.report import flagged_tool_verdicts

    call = Event(idx=0, type="tool_call", payload={"name": "search_products"})
    other = Event(idx=0, type="tool_call", payload={"name": "get_order_details"})
    data.runs[0].events = [call]
    data.runs[1].events = [call]
    data.runs[2].events = [other]
    data.tool_sigs = [ToolSig(name="search_products", unclassified=True),
                      ToolSig(name="get_order_details", unclassified=False, kind="read")]
    assert flagged_tool_verdicts(data) == {"search_products": 2}
    text = render(data)
    assert "search_products: 2 Verdicts rest on a Run that called it" in text


def test_no_flagged_tools_says_so(data):
    """The sentence only holds when every mined ToolSig is confirmed read or write (D70)."""
    from harness.shared.records import ToolSig
    from harness.shared.report import flagged_tool_verdicts

    data.tool_sigs = [ToolSig(name="cancel_order", kind="write", unclassified=False),
                      ToolSig(name="get_order_details", kind="read", unclassified=False)]
    assert flagged_tool_verdicts(data) == {}
    assert "No flagged tools" in render(data)
    data.tool_sigs.append(ToolSig(name="mystery_tool", unclassified=True))
    assert "No flagged tools" not in render(data)


def test_each_task_states_how_many_of_its_rows_are_overlay_rows(data):
    """D74: the report states this per Task, not only build wide."""
    numbers = task_numbers(data, data.tasks[0])
    assert numbers["overlay_rows"] == 2
    assert "- Overlay rows in this Task's Starting state: 2" in render(data)


def test_the_environment_lists_its_open_flags(data):
    data.environment.flags = ["charge_card: 7 of 10 observed errors are unknown (70%)"]
    text = render(data)
    assert "Open flags on the Environment" in text
    assert "charge_card" in text


def test_coverage_rows_carry_the_first_failing_reason():
    from harness.shared.report import coverage_rows

    tasks = [Task(id="t1", run_ids=["r1", "r2"]), Task(id="t2", run_ids=["r3"])]
    rows = coverage_rows(tasks, {"t2": "Run r3 is assisted (D49)"})
    assert [(r.task_id, r.covered, r.run_count) for r in rows] == [("t1", True, 2), ("t2", False, 1)]
    assert rows[1].reason == "Run r3 is assisted (D49)"


def test_tool_sigs_are_read_off_disk(tmp_path):
    from harness.shared.records import ToolSig
    from harness.shared.report import load_tool_sigs

    assert load_tool_sigs(tmp_path) == []
    (tmp_path / "tool_sigs.json").write_text(
        json.dumps([as_dict(ToolSig(name="cancel_order", kind="write", unclassified=False))]),
        encoding="utf-8")
    sigs = load_tool_sigs(tmp_path)
    assert [s.name for s in sigs] == ["cancel_order"]
    assert sigs[0].kind == "write"


# --- house rules ------------------------------------------------------------

SOURCE = Path(__file__).resolve().parents[1] / "src" / "harness" / "shared" / "report.py"


def test_report_never_computes_a_verdict():
    """Design section 4 item 18: report.py reads records, it never computes a Verdict."""
    source = SOURCE.read_text(encoding="utf-8")
    assert "runner.verdict" not in source
    assert "harness.runner" not in source
    assert "harness.builder" not in source


def test_no_em_dashes_in_the_source_or_the_output(data):
    source = SOURCE.read_text(encoding="utf-8")
    assert "—" not in source and "–" not in source
    text = render(data)
    assert "—" not in text and "–" not in text


# --- what the pipeline and the loop actually write (D85, D86, D90) ----------

def a_state(**changes) -> dict:
    """A pipeline state.json in the shape pipeline.py writes it: log rows are sentences, not dicts."""
    body = {
        "status": "failed",
        "statuses": {"mine": "ran", "compile_tools": "failed"},
        "attempts": {"mine": 1, "compile_tools": 3},
        "log": ["compile_tools: attempt 2 of 3, gate replay_fidelity failed",
                "compile_tools: gate replay_fidelity failed 3 times, stage failed"],
        "gates": [as_dict(GateResult(stage="mine", **{"pass": True})),
                  as_dict(GateResult(stage="compile_tools.replay_fidelity", **{"pass": False},
                                     failures=["writes differ on orders"]))],
        "stopped": None,
        "failed_stage": "compile_tools",
    }
    body.update(changes)
    return body


def write_state(workdir: Path, body: dict) -> None:
    (workdir / "pipeline").mkdir(exist_ok=True)
    (workdir / "pipeline" / "state.json").write_text(json.dumps(body), encoding="utf-8")


def test_load_reads_the_state_json_the_pipeline_writes(workdir: Path):
    """The log rows pipeline.py writes are sentences; reading them as dicts crashed the whole report."""
    write_state(workdir, a_state())
    loaded = load(workdir)
    stages = {s.name: s for s in loaded.stages}
    assert stages["compile_tools"].status == "failed"
    assert stages["compile_tools"].attempts == 3
    assert stages["compile_tools"].max_attempts == 3
    assert stages["compile_tools"].gate == "replay_fidelity"
    assert [g.stage for g in loaded.gates] == ["mine", "compile_tools.replay_fidelity"]
    assert "writes differ on orders" in render(loaded)
    assert "attempt 3 of 3" in render(loaded)


def test_a_real_pipeline_run_is_readable_by_the_report(workdir: Path):
    """The one guard against the two modules drifting: a Pipeline writes, the report reads."""
    from harness.runner.pipeline import Pipeline, Stage

    failing = GateResult(stage="mine_gate", **{"pass": False}, failures=["tool_a: 1 call"])
    stages = [Stage("mine", lambda ctx, inputs: {"sigs": [1]}, outputs=["sigs"],
                    gate=lambda ctx, out: failing, max_attempts=2)]
    Pipeline(stages, workdir).run()
    loaded = load(workdir)
    assert [s.name for s in loaded.stages] == ["mine"]
    assert loaded.stages[0].status == "failed"
    assert loaded.stages[0].gate == "mine_gate"
    assert [g.passed for g in loaded.gates] == [False, False]
    assert "mine_gate" in render(loaded)


def test_a_spend_ceiling_stop_names_the_stage_the_cost_and_what_is_left(workdir: Path):
    """D86: report as is, with where it stopped, what it spent and what finishing costs."""
    stopped = {"stage": "compile_tools", "item": "search_products", "spent": 1.3, "ceiling_usd": 1.0,
               "estimate_to_finish": 0.8, "items_left": 1, "stages": {"mine": 0.5, "compile_tools": 0.8},
               "reason": "spend ceiling reached in stage compile_tools on search_products"}
    write_state(workdir, a_state(status="stopped", statuses={"mine": "ran", "compile_tools": "stopped"},
                                 stopped=stopped, log=[], failed_stage=None))
    (workdir / "environment.json").write_text(
        json.dumps(as_dict(Environment(env_id="env-1"))), encoding="utf-8")
    loaded = load(workdir)
    assert loaded.built is False
    text = render(loaded).split("## Tasks")[0]
    assert "Stopped in stage compile_tools on search_products" in text
    assert "Completed stages: mine" in text
    assert "$1.30" in text and "$1.00" in text and "$0.80" in text
    assert "Cost per stage: mine $0.50, compile_tools $0.80" in text
    assert "permission" in text


def test_a_failed_build_environment_gate_means_not_built(workdir: Path):
    """Design section 6: the gate decides, not the presence of the file."""
    (workdir / "environment.json").write_text(
        json.dumps(as_dict(Environment(env_id="env-1"))), encoding="utf-8")
    (workdir / "gates.json").write_text(json.dumps([
        as_dict(GateResult(stage="build_environment", **{"pass": False}, failures=["db.json is missing"]))]),
        encoding="utf-8")
    loaded = load(workdir)
    assert loaded.built is False
    text = render(loaded)
    assert "Environment built: no" in text
    assert "The build Environment gate failed" in text and "db.json is missing" in text


def test_runs_are_read_from_the_jsonl_the_loop_writes(workdir: Path, make_test_model):
    """loop.py writes runs/<task>/<run>.jsonl; reading only *.json left every Run count at zero."""
    from harness.runner import loop

    class Router:
        def route(self, name, args):
            from types import SimpleNamespace
            return SimpleNamespace(result={"ok": True}, error=None, route="llm", assisted=True)

    model = make_test_model([{"tool_calls": [{"id": "c1", "name": "mystery_tool", "arguments": {}}]},
                             {"content": "done"}], loop=True)
    state = loop.new_run_state("r1", workdir=workdir / "runs" / "t1", env_id="env-1", task_id="t1",
                               model="cand/y", max_turns=2)
    loop.run(state, model, router=Router())

    loaded = load(workdir)
    assert [r.run_id for r in loaded.runs] == ["r1"]
    assert loaded.runs[0].task_id == "t1" and loaded.runs[0].model == "cand/y"
    assert loaded.runs[0].assisted is True, "an assisted event makes the Run assisted (D49)"
    assert [e.type for e in loaded.runs[0].events].count("tool_call") == 1
    assert loaded.assisted_share["mystery_tool"] == 1.0


def test_one_run_is_counted_once_even_with_two_verdict_versions(workdir: Path, data: ReportData):
    """A regrade writes <run>.<key>.json beside the old one; counting both doubles every number."""
    (workdir / "verdicts" / "t1").mkdir(parents=True)
    old = a_verdict("r1", False, verifier_version="v1", env_id="env-1", failing_atom="a_cancel")
    new = a_verdict("r1", True, verifier_version="v2", env_id="env-1")
    for name, record in (("r1.aaaa.json", old), ("r1.bbbb.json", new)):
        (workdir / "verdicts" / "t1" / name).write_text(json.dumps(as_dict(record)), encoding="utf-8")
    (workdir / "verifiers").mkdir()
    (workdir / "verifiers" / "t1.json").write_text(json.dumps(as_dict(data.verifiers[0])), encoding="utf-8")
    (workdir / "environment.json").write_text(json.dumps(as_dict(data.environment)), encoding="utf-8")
    (workdir / "tasks").mkdir()
    (workdir / "tasks" / "t1.json").write_text(
        json.dumps(as_dict(Task(id="t1", run_ids=["r1"]))), encoding="utf-8")

    loaded = load(workdir)
    assert len(loaded.verdicts) == 2, "both files are on disk"
    numbers = task_numbers(loaded, loaded.tasks[0])
    assert numbers["runs_graded"] == 1
    assert numbers["superseded"] == 1
    assert numbers["counted"][0].verifier_version == "v2", "the Verdict under the Verifier on disk now"
    assert numbers["candidate_pass_rate"] == 1.0
    assert "Superseded Verdicts not counted" in render(loaded)


def test_the_scorecard_reads_the_nested_dict_validate_returns(workdir: Path):
    """validate.scorecard is the only writer, and it returns one nested dict, not a list of rows."""
    from harness.shared.report import scorecard_rows

    card = {
        "tool_fidelity": {"success": {"total": 25, "matched": 24, "raw": 0.96, "explained": 1.0,
                                      "explained_misses": 1, "unexplained": 0},
                          "error": {"total": 4, "matched": 4, "raw": 1.0, "explained": 1.0}},
        "task_coverage": {"tasks_total": 2, "tasks_covered": 1, "run_weighted": 0.5},
        "verdict_agreement": {"total": 10, "matched": 9, "raw": 0.9, "explained": 0.95},
        "user_fact_consistency": {"total": 3, "matched": 3, "raw": 1.0, "explained": 1.0},
    }
    (workdir / "scorecard.json").write_text(json.dumps(card), encoding="utf-8")
    rows = {item.name: item for item in load(workdir).scorecard}
    assert set(rows) == {"replay fidelity, success calls", "replay fidelity, error calls",
                         "verdict agreement", "user fact consistency", "task coverage"}
    assert (rows["replay fidelity, success calls"].raw, rows["replay fidelity, success calls"].explained) \
        == ("96%", "100%")
    assert rows["task coverage"].raw == "1 of 2 Tasks"
    assert scorecard_rows(None) == []
    text = render(load(workdir))
    assert "| replay fidelity, success calls | 96% | 100% |" in text


def test_each_counted_verdict_is_listed_with_its_atom_and_its_path(data):
    """D46: the failing atom and same-path or different-path, per Verdict, not only as a count."""
    data.verdicts[1] = a_verdict("r2", False, failing_atom="a_cancel", cause="candidate", same_path=False)
    data.verdicts[0] = a_verdict("r1", True, same_path=True)
    block = render(data).split("### Task t1", 1)[1]
    assert "  - r2: fail, failing atom a_cancel (required), different path from the Reference, cause candidate" in block
    assert "  - r1: pass, no failing atom, same path as the Reference" in block


def test_the_verdicts_left_out_are_listed_with_what_excluded_them(data):
    """D49: an assisted or environment-suspected Verdict is shown underneath, not silently dropped."""
    from harness.shared.records import Event

    data.runs[4].events = [Event(idx=0, type="tool_result", payload={"name": "search_products"}, assisted=True)]
    block = render(data).split("### Task t1", 1)[1]
    assert "- Verdicts not counted, and what excluded them:" in block
    assert "r5: environment suspected, cause environment; assisted Run, search_products stood in" in block


def test_the_queue_shows_the_spans_each_judge_cited(data):
    """D92: the queue lists the two verdicts and the cited spans, which is what a person resolves on."""
    data.disagreement_queue = [{
        "use": "cause", "item_id": "r5", "verdict_a": "environment", "verdict_b": "candidate",
        "verdict_c": "candidate",
        "judge_a": {"judge": "A", "cited_spans": ["turn 4: I refuse"]},
        "judge_b": {"judge": "B", "cited_spans": ["tool_result 2: status pending"]},
    }]
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "first judge cited: turn 4: I refuse" in block
    assert "second judge cited: tool_result 2: status pending" in block
    assert "a third sample said candidate" in block


def test_the_audit_rate_appears_beside_the_disagreement_rate(data):
    """D92: once a person resolves queue items those resolutions are the labelled set."""
    assert "no human labels yet" in render(data).lower()
    data.audit_rate = 0.9
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "Audit rate 90%" in block
    assert "no human labels yet" not in block.lower()


def test_a_stage_name_with_a_space_is_still_valid_mermaid():
    """A raw name as a node id ends the id at the space, which breaks the whole diagram."""
    text = pipeline_dag([
        StageStatus(name="build Environment", status="ran"),
        StageStatus(name="compile-tools", status="failed", gate="replay_fidelity", attempts=1, max_attempts=3),
    ])
    assert 'build_Environment["build Environment (ran)"]' in text
    assert "build_Environment --> compile_tools" in text
    assert 'compile_tools -. "gate replay_fidelity failed, attempt 1 of 3" .-> compile_tools' in text
    assert "build Environment[" not in text


def test_a_record_that_does_not_load_is_named_rather_than_silently_dropped(workdir: Path):
    """D85: the person decides on these numbers, so a file that did not load has to be visible."""
    (workdir / "verdicts" / "t1").mkdir(parents=True)
    good = as_dict(a_verdict("r1", True))
    (workdir / "verdicts" / "t1" / "r1.json").write_text(json.dumps(good), encoding="utf-8")
    broken = dict(good, run_id="r2")
    broken["class"] = "passed"
    (workdir / "verdicts" / "t1" / "r2.json").write_text(json.dumps(broken), encoding="utf-8")
    (workdir / "verdicts" / "t1" / "r3.json").write_text("{not json", encoding="utf-8")
    (workdir / "disagreement_queue.jsonl").write_text('{"use": "cause"}\n{broken\n', encoding="utf-8")

    loaded = load(workdir)
    assert [v.run_id for v in loaded.verdicts] == ["r1"]
    assert loaded.disagreement_queue == [{"use": "cause"}]
    assert loaded.records_not_read == [
        "verdicts/t1/r2.json: not a Verdict this report can read",
        "verdicts/t1/r3.json: not a JSON object",
        "disagreement_queue.jsonl line 2: not JSON",
    ]
    text = render(loaded)
    assert "### Records not read" in text
    assert "verdicts/t1/r2.json" in text


# --- rewritten policy rules nobody has accepted, and judge abstentions (D76, D48, D92) ---

def test_a_rewritten_rule_awaiting_review_is_named_as_checked_by_nobody(data):
    """D76: until a person accepts the rewrite the rule is in no Verifier and in no residual list."""
    data.policy_items = data.policy_items + [
        Constraint(id="p4", text="be reasonable about refunds",
                   rewritten_text="refund only within 30 days of delivery"),
    ]
    text = render(data)
    assert "Awaiting setup review, not checked: 1 rewritten policy rule" in text
    assert "p4: be reasonable about refunds" in text
    assert "rewritten as: refund only within 30 days of delivery" in text


def test_a_compiled_rewrite_is_not_awaiting_review(data):
    data.policy_items = data.policy_items + [
        Constraint(id="p4", text="be reasonable", rewritten_text="refund within 30 days", compiled=True),
    ]
    assert "Awaiting setup review" not in render(data)


def test_the_abstained_items_are_listed_apart_from_the_splits(data):
    """D92 sends both to a person, and they are two different things to look at."""
    data.judge_disagreement = {"pairs": 8, "disagreements": 2, "rate": 0.25,
                               "abstains": 3, "abstain_rate": 0.375}
    data.disagreement_queue = [
        {"use": "cause", "item_id": "r5", "verdict_a": "environment", "verdict_b": "candidate",
         "reason": "split"},
        {"use": "policy_atom", "item_id": "r6:a1", "verdict_a": "abstain", "verdict_b": "abstain",
         "reason": "agreed_abstain"},
    ]
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "Judge abstention: 3 of 8 pairs (38%)" in block
    splits, undecided = block.split("### Items the judges did not decide", 1)
    assert "r5" in splits and "r5" not in undecided
    assert "r6:a1" in undecided and "and both abstained" in undecided


def test_a_queue_row_with_no_reason_is_still_read_as_a_split(data):
    """The reason field is newer than the queue; a row without one has two verdicts that disagree."""
    block = render(data).split("## Disagreement queue", 1)[1]
    assert "### Items a person may resolve" in block
    assert "### Items the judges did not decide" not in block
