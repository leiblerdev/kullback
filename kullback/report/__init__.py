"""The face of the Harness: one Markdown report that opens with whether the Environment was built, then the
numbers per Task, then a suggestion the person decides on (D85); it reads records and never computes a Verdict."""

from __future__ import annotations

from kullback.report.data import (
    ENVIRONMENT,
    LESSONS,
    QUEUE,
    ROUNDS,
    SECTIONS,
    SYNTHETIC,
    TASKS,
    USER_FIDELITY,
    ReportData,
    ScorecardItem,
    SetAsideLesson,
    StageStatus,
    TaskCoverage,
)
from kullback.report.load import (
    DOMAIN_ARCHETYPES,
    DOMAIN_GAPS,
    EVENT_TYPES,
    SHAPED_INDEX,
    SYNTHETIC_INDEX,
    coverage_rows,
    load,
    load_runs,
    load_tool_sigs,
    run_from_jsonl,
    scorecard_rows,
)
from kullback.report.numbers import (
    assisted_share_from_runs,
    assisted_tools_of,
    claims_for_task,
    current_verdicts,
    false_rejection_by_task,
    flagged_tool_verdicts,
    overlay_rows,
    suggestion,
    task_numbers,
    version_match,
)
from kullback.report.pipeline import (
    ENVIRONMENT_GATE_NAMES,
    environment_gate,
    node_id,
    pipeline_dag,
    stage_statuses,
)
from kullback.report.render import _claims_table as _claims_table
from kullback.report.render import _difficulty_table as _difficulty_table
from kullback.report.render import (
    assisted_tool_note,
    render,
    tool_fidelity_counts,
    write_report,
)

# The two section helpers the tests read off the package root stay reachable here.

__all__ = [
    "DOMAIN_ARCHETYPES",
    "DOMAIN_GAPS",
    "ENVIRONMENT",
    "ENVIRONMENT_GATE_NAMES",
    "EVENT_TYPES",
    "LESSONS",
    "QUEUE",
    "ROUNDS",
    "SECTIONS",
    "SHAPED_INDEX",
    "SYNTHETIC",
    "SYNTHETIC_INDEX",
    "TASKS",
    "USER_FIDELITY",
    "ReportData",
    "ScorecardItem",
    "SetAsideLesson",
    "StageStatus",
    "TaskCoverage",
    "assisted_share_from_runs",
    "assisted_tool_note",
    "assisted_tools_of",
    "claims_for_task",
    "coverage_rows",
    "current_verdicts",
    "environment_gate",
    "false_rejection_by_task",
    "flagged_tool_verdicts",
    "load",
    "load_runs",
    "load_tool_sigs",
    "node_id",
    "overlay_rows",
    "pipeline_dag",
    "render",
    "run_from_jsonl",
    "scorecard_rows",
    "stage_statuses",
    "suggestion",
    "task_numbers",
    "tool_fidelity_counts",
    "version_match",
    "write_report",
]

# --- numbers ---------------------------------------------------------------


# --- the pipeline DAG ------------------------------------------------------


# --- the sections ----------------------------------------------------------


# --- reading the records off disk ------------------------------------------
