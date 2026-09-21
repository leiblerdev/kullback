"""Every single-file workdir artifact the package writes today, declared once (D267, registry step only).

One ArtifactSpec per file the code writes to a fixed path under a workdir: the name,
the path, format 1, the owning stage as the code calls it, and a validator that
checks the top level shape and nothing deeper, unless a record model already
accepts that shape, in which case the validator runs the model and answers the
plain value back. No caller is migrated here: the specs are proven to work with
the existing store, and the caller table in the round report says which write
sites each declaration will replace. Per Task, per Run and per round families
cannot be one fixed path, so they stay out; the report carries the proposal.
"""

from __future__ import annotations

from typing import Any

from kullback.runner.canon import CanonRules, EquivalenceTable
from kullback.runner.records import (
    Constraint,
    EntitySchema,
    Environment,
    Finding,
    GateResult,
    RoundRecord,
    SetAsideLesson,
    Task,
    ToolSig,
)
from kullback.store import ArtifactSpec


def _want_dict(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("want object")
    return value


def _want_list(value: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError("want array")
    return value


def _want_keys(*required: str):
    def check(value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("want object")
        for key in required:
            if key not in value:
                raise ValueError("missing %s" % key)
        return value

    return check


def _want_model(model):
    def check(value: Any) -> Any:
        model.model_validate(value)
        return value

    return check


def _want_model_list(model):
    def check(value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("want array")
        for item in value:
            model.model_validate(item)
        return value

    return check


def _want_tasks_index(value: Any) -> Any:
    if not isinstance(value, dict) or "tasks" not in value:
        raise ValueError("want object with tasks")
    rows = value["tasks"]
    if not isinstance(rows, list):
        raise ValueError("want array of tasks")
    for row in rows:
        Task.model_validate(row)
    return value


def _want_history(value: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError("want array")
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("want object row")
        for key in ("round", "rulings", "tasks"):
            if key not in row:
                raise ValueError("missing %s" % key)
    return value


STORE_SPECS: tuple = (
    ArtifactSpec(name="rounds", path="rounds.json", format=1, owner="rounds",
                 validate=_want_model_list(RoundRecord)),
    ArtifactSpec(name="report_config", path="report_config.json", format=1, owner="rounds",
                 validate=_want_dict),
    ArtifactSpec(name="gates", path="gates.json", format=1, owner="gates",
                 validate=_want_model_list(GateResult)),
    ArtifactSpec(name="gates_by_round", path="gates_by_round.json", format=1, owner="rounds",
                 validate=_want_history),
    ArtifactSpec(name="compile_snapshot", path="compile_snapshot.json", format=1, owner="compile_tools",
                 validate=_want_keys("format", "round", "stage", "rows")),
    ArtifactSpec(name="tasks_frozen", path="tasks_frozen.json", format=1, owner="cluster",
                 validate=_want_keys("format", "task_ids", "tasks", "hash")),
    ArtifactSpec(name="scorecard", path="scorecard.json", format=1, owner="gates",
                 validate=_want_dict),
    ArtifactSpec(name="tool_sigs", path="tool_sigs.json", format=1, owner="mine",
                 validate=_want_model_list(ToolSig)),
    ArtifactSpec(name="unknown_tools", path="unknown_tools.json", format=1, owner="mine",
                 validate=_want_list),
    ArtifactSpec(name="row_homes", path="row_homes.json", format=1, owner="mine",
                 validate=_want_dict),
    ArtifactSpec(name="world_constants", path="world_constants.json", format=1, owner="mine",
                 validate=_want_keys("table", "row", "columns")),
    ArtifactSpec(name="schema", path="schema.json", format=1, owner="mine",
                 validate=_want_model(EntitySchema)),
    ArtifactSpec(name="readers", path="readers.json", format=1, owner="readers",
                 validate=_want_dict),
    ArtifactSpec(name="canon_rules", path="canon-rules.json", format=1, owner="canon_rules",
                 validate=_want_model(CanonRules)),
    ArtifactSpec(name="equivalence", path="equivalence.json", format=1, owner="replay_reference",
                 validate=_want_model(EquivalenceTable)),
    ArtifactSpec(name="tasks_index", path="tasks.json", format=1, owner="cluster",
                 validate=_want_tasks_index),
    ArtifactSpec(name="task_split", path="task_split.json", format=1, owner="cluster",
                 validate=_want_dict),
    ArtifactSpec(name="grouping", path="grouping.json", format=1, owner="cluster",
                 validate=_want_keys("format", "fingerprint")),
    ArtifactSpec(name="db", path="db.json", format=1, owner="starting_state",
                 validate=_want_dict),
    ArtifactSpec(name="world_provenance", path="world_provenance.json", format=1, owner="starting_state",
                 validate=_want_keys("witnesses", "holdout_columns", "holdout_rows",
                                     "holdout_columns_total")),
    ArtifactSpec(name="synthetic", path="synthetic.json", format=1, owner="starting_state",
                 validate=_want_dict),
    ArtifactSpec(name="assumptions", path="assumptions.json", format=1, owner="starting_state",
                 validate=_want_list),
    ArtifactSpec(name="overlay_pins", path="overlay_pins.json", format=1, owner="starting_state",
                 validate=_want_keys("totals", "tasks")),
    ArtifactSpec(name="bodies", path="bodies.json", format=1, owner="compile_tools",
                 validate=_want_dict),
    ArtifactSpec(name="tool_builds", path="tool_builds.json", format=1, owner="compile_tools",
                 validate=_want_dict),
    ArtifactSpec(name="tool_call_outcomes", path="tool_call_outcomes.json", format=1,
                 owner="compile_tools", validate=_want_dict),
    ArtifactSpec(name="kept_bodies", path="kept_bodies.json", format=1, owner="compile_tools",
                 validate=_want_dict),
    ArtifactSpec(name="tool_lesson_counts", path="tool_lesson_counts.json", format=1,
                 owner="compile_tools", validate=_want_dict),
    ArtifactSpec(name="tool_fidelity", path="tool_fidelity.json", format=1, owner="compile_tools",
                 validate=_want_dict),
    ArtifactSpec(name="replays", path="replays.json", format=1, owner="replay_reference",
                 validate=_want_dict),
    ArtifactSpec(name="runs_index", path="runs.json", format=1, owner="replay_reference",
                 validate=_want_keys("runs")),
    ArtifactSpec(name="holdout_answers", path="holdout_answers.json", format=1,
                 owner="replay_reference",
                 validate=_want_keys("totals", "columns", "tasks", "tools")),
    ArtifactSpec(name="replay_evidence", path="replay_evidence.json", format=1,
                 owner="replay_reference", validate=_want_dict),
    ArtifactSpec(name="write_effects", path="write_effects.json", format=1, owner="replay_reference",
                 validate=_want_keys("totals", "per_tool")),
    ArtifactSpec(name="semantic_counts", path="semantic_counts.json", format=1,
                 owner="replay_reference", validate=_want_dict),
    ArtifactSpec(name="readmission_blocked", path="readmission_blocked.json", format=1,
                 owner="replay_reference", validate=_want_dict),
    ArtifactSpec(name="constraints", path="constraints.json", format=1, owner="compile_policy",
                 validate=_want_model_list(Constraint)),
    ArtifactSpec(name="policy_coverage", path="policy_coverage.json", format=1, owner="compile_policy",
                 validate=_want_keys("exercised")),
    ArtifactSpec(name="policy", path="policy.json", format=1, owner="compile_policy",
                 validate=_want_keys("items", "compiled")),
    ArtifactSpec(name="lessons_set_aside", path="lessons_set_aside.json", format=1,
                 owner="judge_lessons", validate=_want_model_list(SetAsideLesson)),
    ArtifactSpec(name="vocabulary", path="vocabulary.json", format=1, owner="vocabulary",
                 validate=_want_dict),
    ArtifactSpec(name="user_facts", path="user_facts.json", format=1, owner="user_rules",
                 validate=_want_keys("facts")),
    ArtifactSpec(name="environment", path="environment.json", format=1, owner="build_environment",
                 validate=_want_model(Environment)),
    ArtifactSpec(name="batch_gates", path="batch_gates.json", format=1, owner="candidate",
                 validate=_want_model_list(GateResult)),
    ArtifactSpec(name="pipeline_state", path="pipeline/state.json", format=1, owner="pipeline",
                 validate=_want_keys("status", "statuses", "attempts", "log", "gates")),
    ArtifactSpec(name="anchor", path="anchor.json", format=1, owner="pipeline",
                 validate=_want_dict),
    ArtifactSpec(name="evidence_counts", path="evidence_counts.json", format=1, owner="pipeline",
                 validate=_want_dict),
    ArtifactSpec(name="intake_ruling", path="intake_ruling.json", format=1, owner="ingest",
                 validate=_want_keys("files", "counts", "reasons", "floor", "eligible_share",
                                     "passed")),
    ArtifactSpec(name="claims", path="claims.json", format=1, owner="claims",
                 validate=_want_keys("format", "runs", "tasks", "totals", "flagged",
                                     "write_tools")),
    ArtifactSpec(name="difficulty", path="difficulty.json", format=1, owner="difficulty",
                 validate=_want_keys("format", "tasks", "no_record", "buckets")),
    ArtifactSpec(name="user_lessons", path="user_lessons.json", format=1, owner="user",
                 validate=_want_keys("format", "lessons")),
    ArtifactSpec(name="user_fidelity", path="user_fidelity.json", format=1, owner="user",
                 validate=_want_keys("format", "round", "tasks", "summary")),
    ArtifactSpec(name="domain_gaps", path="domain/gaps.json", format=1, owner="domain",
                 validate=_want_keys("format", "gaps")),
    ArtifactSpec(name="domain_archetypes", path="domain/archetypes.json", format=1, owner="domain",
                 validate=_want_keys("format", "archetypes")),
    ArtifactSpec(name="synth_index", path="synthetic/index.json", format=1, owner="synthesise",
                 validate=_want_keys("format", "requested", "salt", "counts", "tasks")),
    ArtifactSpec(name="synth_shaped", path="synthetic/shaped.json", format=1, owner="synthesise",
                 validate=_want_keys("format", "salt", "shaped", "counts", "tasks")),
    ArtifactSpec(name="graph", path="graph.json", format=1, owner="synthesise",
                 validate=_want_keys("format", "runs", "nodes", "edges")),
    ArtifactSpec(name="budget", path="budget.json", format=1, owner="budget",
                 validate=_want_keys("stages", "total")),
    ArtifactSpec(name="sample_salt", path="sample_salt.json", format=1, owner="sampling",
                 validate=_want_keys("salt")),
    ArtifactSpec(name="examiner_findings", path="examiner/findings.json", format=1, owner="examiner",
                 validate=_want_model_list(Finding)),
    ArtifactSpec(name="examiner_refusals", path="examiner/refusals.json", format=1, owner="examiner",
                 validate=_want_dict),
    ArtifactSpec(name="examiner_auto_loosen", path="examiner/auto_loosen.json", format=1,
                 owner="examiner", validate=_want_list),
    ArtifactSpec(name="examiner_rerolls", path="examiner/rerolls.json", format=1, owner="examiner",
                 validate=_want_dict),
    ArtifactSpec(name="task_status", path="task_status.json", format=1, owner="examiner",
                 validate=_want_dict),
    ArtifactSpec(name="references", path="references.json", format=1, owner="examiner",
                 validate=_want_dict),
    ArtifactSpec(name="constraints_check", path="constraints_check.json", format=1, owner="examiner",
                 validate=_want_keys("rates", "demoted", "constraints")),
)

BY_NAME: dict = {spec.name: spec for spec in STORE_SPECS}
