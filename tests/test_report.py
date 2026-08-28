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
        tasks=[Task(id="t1", name="cancel an order", intent="cancel the pending order", run_ids=["r1", "r2", "r3", "r4", "r5"])],
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
        task_coverage=[TaskCoverage(task_id="t1", covered=False, reason="1 Run assisted on search_products", run_count=5)],
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


def test_suggestion_never_decides():
    assert "The decision is yours." in suggestion({"runs_graded": 4, "margin": 0.5}, built=True)


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
    from harness.shared.report import flagged_tool_verdicts

    assert flagged_tool_verdicts(data) == {}
    assert "No flagged tools" in render(data)


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
