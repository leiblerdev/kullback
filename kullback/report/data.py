"""The shapes the report reads: one object holding every record a workdir yields."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from kullback.runner.records import (
    Constraint,
    Environment,
    GateResult,
    Record,
    RoundRecord,
    Run,
    SetAsideLesson,
    Task,
    TaskOverlay,
    ToolSig,
    Verdict,
    Verifier,
)

SECTIONS = (
    "## Environment",
    "## Rounds",
    "## Tasks",
    "## Synthetic Tasks",
    "## Disagreement queue",
    "## The Simulated user",
    "## Lessons set aside",
)

ENVIRONMENT, ROUNDS, TASKS, SYNTHETIC, QUEUE, USER_FIDELITY, LESSONS = SECTIONS


class ScorecardItem(Record):
    """One scorecard number with its raw and explained value side by side (D62, D80)."""

    name: str
    raw: str = ""
    explained: str = ""
    note: str = ""


class StageStatus(Record):
    """One pipeline stage as the run recorded it, for the DAG."""

    name: str
    status: str = "pending"
    gate: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 0
    usd: float = 0.0
    cache_share: float = 0.0  # share of input tokens the provider served from its prompt cache
    memo_hits: int = 0  # calls answered from the on-disk memo, never sent to a provider


class TaskCoverage(Record):
    """Whether one Task is covered, and the first reason it is not (D96)."""

    task_id: str
    covered: bool = False
    reason: Optional[str] = None
    run_count: int = 0


# SetAsideLesson is imported from records.py above and re-exported here: every record lives in
# records.py (build brief rule 4), and this is the name the report's own field is typed with.


class ReportData(BaseModel):
    """Everything the report reads, already on disk as records; nothing here is computed by a model."""

    title: str = "Harness build report"
    kind: Literal["build", "batch"] = "build"
    built: bool = False
    stopped_reason: Optional[str] = None
    stopped: dict = Field(default_factory=dict)
    records_not_read: list[str] = Field(default_factory=list)
    environment: Optional[Environment] = None
    gates: list[GateResult] = Field(default_factory=list)
    scorecard: list[ScorecardItem] = Field(default_factory=list)
    stages: list[StageStatus] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    verifiers: list[Verifier] = Field(default_factory=list)
    tool_sigs: list[ToolSig] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    overlays: list[TaskOverlay] = Field(default_factory=list)
    policy_items: list[Constraint] = Field(default_factory=list)
    policy_exercised: list[str] = Field(default_factory=list)
    task_coverage: list[TaskCoverage] = Field(default_factory=list)
    frontier_models: list[str] = Field(default_factory=list)
    assisted_share: dict[str, float] = Field(default_factory=dict)
    judge_disagreement: dict = Field(default_factory=dict)
    # Which model each side of the judging ran on, by role (D160): "build", "judge", "second_judge".
    judge_models: dict = Field(default_factory=dict)
    audit_rate: Optional[float] = None
    disagreement_queue: list[dict] = Field(default_factory=list)
    tasks_aside: list[dict] = Field(default_factory=list)
    lessons_set_aside: list[SetAsideLesson] = Field(default_factory=list)
    # D171: tool_fidelity.json, the compile_tools artifact: `tools` per tool over the corpus,
    # `tasks` per Task per tool over that Task's own recorded calls.
    tool_fidelity: dict = Field(default_factory=dict)
    # D214: user_fidelity.json, the round driver's artifact: how close each driver's turns are to
    # the recorded ones, per Task and per corpus. Empty on a build written before D214.
    user_fidelity: dict = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)
    trusted: Optional[GateResult] = None
    # D209: difficulty.json, the record and the bucket per Task with the Tasks that carry neither.
    difficulty: dict = Field(default_factory=dict)
    # D224: synthetic/index.json, the walks that became Tasks. Its own field and its own section,
    # never added into `tasks`: nothing generated may be counted where the recorded Tasks are.
    synthetic: dict = Field(default_factory=dict)
    # D223: claims.json, what the transcripts claimed against what the state received, per Run, per
    # Task and over the corpus, with the Tasks flagged for the Simulated user's end protocol.
    claims: dict = Field(default_factory=dict)
    # D225: domain/archetypes.json, domain/gaps.json and synthetic/shaped.json. What the domain's own
    # public material attests, what this Environment cannot execute of it, and the Tasks shaped to
    # the rest. Its own fields for the same reason the synthetic one is its own: nothing read off a
    # public page is evidence about a recording.
    domain: dict = Field(default_factory=dict)
    domain_gaps: list[dict] = Field(default_factory=list)
    shaped: dict = Field(default_factory=dict)
    findings: list[dict] = Field(default_factory=list)  # repairs/repair_record_finding.jsonl (D155)
    # D218: the last closed round's Task table and how far the live files have moved from it. The
    # report reads the table, so its Task level numbers are one round's answer rather than a join
    # over three files that move at different times, and it says how much has moved since.
    snapshot: dict = Field(default_factory=dict)
    drift: dict = Field(default_factory=dict)
