"""The Examiner's eight tools: pydantic arguments in, a short result out, every ruling from a gate (D123, D127, D128).

`read` is the Examiner's whole read surface: a Task, a Trace, an Intent, a Run, a Verifier, a
probe pool, the task status, the rulings, the re-roll and replay rows, the References. `search`
looks for one phrase across those records instead of reading them: the Traces, the Runs, the
Intents, the task status and the Verifiers, answering a count per kind and one line per match with
where it matched, so a claim about many Tasks is grounded without holding any of them in context.
`derive` runs the derivation over every Task (or one) and is the Builder's old derive_verifier
stage, byte for byte. `probe` scores one hand-written Run against the current Verifier and keeps it in the
Task's pool forever. `repair` proposes a new Verifier version and the gates decide whether it is
accepted: the D79 suite, the pool, one-directional loosening. `refuse` asks to give a Task up and
the refuse gate admits it only when no frontier Run finished. `reroll` buys more frontier Runs
through the Runner callable the Builder handed over. `finding` files what is wrong on the
Builder's side, with the Builder verb that answers it and the one-line hint that verb needs.
Nothing here writes a tool body, a table or the Environment, and no tool decides a
ruling: each one calls a registered gate and reports what it said.

A result is what the model reads plus what it does not: the rendered text is the summary and the
rulings on one line each, the rest is in `details` for the transcript.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.events import StageEnd, StageStart
from kullback.agent.tools import AgentTool, RetryableToolError, counted_ruling_line
from kullback.examiner import findings as findings_mod
from kullback.examiner import lifecycle
from kullback.examiner import loosen as loosen_mod
from kullback.examiner import stage as stage_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.gates import Ruling, artifacts, ruling_of, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates.loosening import loosening_gate
from kullback.gates.probes import (
    consecutive_failed,
    probe_admission_gate,
    probe_pool_gate,
    version_hash,
    write_tools_of,
)
from kullback.gates.trust import finished_runs, refuse_gate
from kullback.gates.verifier_suite import check_run, make_atom
from kullback.runner import budget
from kullback.runner.records import (
    Atom,
    Event,
    FindingKind,
    FindingVerb,
    GateResult,
    Intent,
    Probe,
    ProbePool,
    Refusal,
    Run,
    Verifier,
    VerifierHistory,
    VerifierVersion,
    apply_intent,
    as_dict,
    read_json,
    write_json,
)

Sink = Callable[[Any], Awaitable[None]]
STAGE = stage_mod.STAGE
ReadKind = Literal["task", "trace", "intent", "run", "verifier", "probes", "task_status", "gates", "rerolls",
                   "replays", "references"]
SearchKind = Literal["trace", "run", "intent", "task_status", "verifier"]
SEARCH_KINDS: tuple[str, ...] = ("trace", "run", "intent", "task_status", "verifier")
SEARCH_LINE = 200  # about one line of the record around a match, the match inside it
SEARCH_LIMIT = 50
# --- what a finding is about, and the names a model reaches for instead ---
#
# One live build's Examiner made 28 `finding` calls and 17 of them were refused for passing a ruling
# name (`replay_reference`) where the kind enum was wanted; 13 of the 17 were about real replay
# differences on four tools and none was retried, so the round lost them. The kinds are what the
# record has (`FindingKind`); the names below are what the Examiner is looking at when it files one,
# and each maps to the kind it is about. Nothing here knows a domain: the left column is the gates'
# own ruling and stage vocabulary, and a name that is neither a kind nor a ruling is refused with
# both lists rather than with pydantic's enum line.
FINDING_KINDS: tuple[str, ...] = get_args(FindingKind)
KIND_OF_RULING: dict[str, str] = {
    # the fidelity rulings: a body's answer to a recorded call parts from the recording, at any grain
    "replay_reference": "fidelity",
    "replay_fidelity": "fidelity",
    "compile_tools.replay_fidelity": "fidelity",
    "gate_a_oracle_replay": "fidelity",
    # the tool-body stages: the tool has no body that cleared its gates
    "compile_tools": "assisted_tool",
    "compile_tools.parses": "assisted_tool",
    "compile_tools.executes": "assisted_tool",
    "compile_tools.deterministic": "assisted_tool",
    "compile_tools.non_trivial": "assisted_tool",
    "compile_tools.memorised_values": "assisted_tool",
    "sensitivity": "assisted_tool",
    "compile_tools.bodies": "assisted_tool",
    "assisted": "assisted_tool",
    # the reference rulings: the recordings of a Task do not settle on one End state
    "reference": "reference_disagreement",
    "references": "reference_disagreement",
    "reference_agreement": "reference_disagreement",
    "disagreement": "reference_disagreement",
    # D193: every End state of a Task was failed on one key the failures agree the Intent or a policy
    # line required. Nothing about the recordings disagreeing; the world or the Intent is what to look
    # at, so it maps to the world and the Examiner may still file it as `other` for an Intent.
    "no_correct_recording": "environment",
    # the Verifier's own rulings
    "derive_verifier": "suite",
    "verifier_suite": "suite",
    "loosening": "false_rejection",
    # the world itself
    "build_environment": "environment",
    "starting_state": "environment",
    "intent": "other",
}
# The D79 checks and the stages they report under are all one kind: a check of the suite.
KIND_OF_RULING.update({name: "suite" for name in verifier_suite.D79_STAGES})
KIND_OF_RULING.update({check: "suite" for check in verifier_suite.D79_STAGES.values()})


def finding_kind(name: str, tools: Iterable[str] = ()) -> str:
    """The finding kind a model's `kind` argument means: itself, a ruling's, or a tool's.

    A kind is taken as it is; a ruling or stage name is mapped to the kind that ruling is about; a
    name of a tool of this build is `assisted_tool`, since a finding filed under a tool's own name
    is a finding about that tool's body. Anything else raises with both lists, so the answer the
    model reads says what to send instead rather than only what was wrong.
    """
    text = (name or "").strip()
    if text in FINDING_KINDS:
        return text
    mapped = KIND_OF_RULING.get(text)
    if mapped is not None:
        return mapped
    if text in set(tools):
        return "assisted_tool"
    raise ValueError(f"{text!r} is not a finding kind. The kinds are: {', '.join(FINDING_KINDS)}. "
                     f"A ruling name is taken as the kind it is about ({ruling_mapping()}), and the "
                     f"name of a tool of this build is taken as assisted_tool.")


def ruling_mapping() -> str:
    """The ruling-to-kind map as one line per kind, which is how the refusal states it."""
    grouped: dict[str, list[str]] = {}
    for ruling, kind in sorted(KIND_OF_RULING.items()):
        grouped.setdefault(kind, []).append(ruling)
    return "; ".join(f"{kind} <- {', '.join(rulings)}" for kind, rulings in sorted(grouped.items()))


def _tool_names(plan: ExaminerPlan) -> list[str]:
    """Every tool of this build, off the mined signatures the plan carries."""
    return [str(getattr(sig, "name", "")) for sig in (plan.store.get("sigs") or [])]


# The eight classes of skills.BUG_CLASSES, spelled out so the tool schema names them; the test pins the two.
BugClass = Literal["loose answer extraction", "missing final-answer markers", "numeric-tolerance abuse",
                   "schema-only validation", "extra-field acceptance", "visible-test overfitting",
                   "stdout spoofing", "missing timeouts", "other"]


def render(result: BaseModel) -> str:
    """The summary line, then the rulings on one line; everything else stays in details.

    A gate rules over every target at once, so a gate failing on more than one says how many and
    names the first as an example (`counted_ruling_line`), the same way the Builder's results read.
    """
    lines = [getattr(result, "summary", None) or getattr(result, "text", "") or ""]
    rulings = getattr(result, "rulings", None)
    if rulings:
        lines.append(counted_ruling_line("rulings", rulings))
    return "\n".join(line for line in lines if line)


# --- argument and result models ---------------------------------------------------

class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ReadKind = Field(description="What to read: task, trace, intent, run, verifier, probes, task_status, "
                                       "gates, rerolls, replays or references. `intent` is the whole Intent "
                                       "record of one Task, including `ungrounded_phrases` (the noun phrases "
                                       "the intent gate refused) and `run_coverage` (phrase to the Runs that "
                                       "evidence it): read it before suggesting repair_intent.")
    id: Optional[str] = Field(default=None, description="The Task, Trace, Run or ruling name. Omitted, the "
                                                        "whole-file kinds answer an index (one line per id), "
                                                        "not the file; no read is longer than READ_CHARS.")


class ReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    id: Optional[str] = None
    text: str


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="What to look for: a case-insensitive substring, or a regular "
                                  "expression when `regex` is true.")
    regex: bool = Field(default=False, description="Read `text` as a regular expression instead of a "
                                                   "substring. Off by default, so a phrase with a "
                                                   "bracket or a dot in it searches for itself.")
    kinds: list[SearchKind] = Field(default_factory=list,
                                    description="Which records to search: trace, run, intent, task_status, "
                                                "verifier. Empty is all five.")
    tool: Optional[str] = Field(default=None, description="Only the calls of this tool. A tool call in a "
                                                          "Trace and an event of a Run belong to a tool; "
                                                          "nothing else does, so the other kinds answer "
                                                          "nothing when this is set.")
    task_id: Optional[str] = Field(default=None, description="Only the records of this Task.")
    limit: int = Field(default=SEARCH_LIMIT, ge=1, le=500,
                       description="The most matching lines to answer; the counts are of every match.")


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: int
    shown: int
    counts: dict[str, int] = Field(default_factory=dict)
    text: str


class DeriveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(default="all", description="`all` for every Task, or one Task id.")


class DeriveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    target: str
    status: str
    verifiers: list[str] = Field(default_factory=list)
    passed: int = 0
    rulings: list[Ruling] = Field(default_factory=list)
    # The findings the round's records filed by themselves, ranked by the Tasks they cost (D170);
    # the round driver reads them off here, the way it reads a model's finding off its tool result.
    findings: list[dict] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["verifiers", "task_status", "history"])


class ProbeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task whose Verifier the probe attacks.")
    bug_class: BugClass = Field(description="One of the eight classes of the probe skill, or other.")
    note: str = Field(default="", description="What the probe changes and why the Task is not done.")
    events: list[dict] = Field(description="The probe's events, in the Runner's shape: type and payload each.")
    termination_reason: str = Field(default="success", description="How the Run ended, in the Runner's words.")
    base_run_id: Optional[str] = Field(default=None, description="The Run the probe was edited from.")


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    probe_id: str
    task_id: str
    scored_pass: bool
    failing_atom: Optional[str] = None
    pool_size: int
    consecutive_failed: int
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["probes"])


class RepairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="Why the current version is wrong and what the new one changes.")
    drop: list[str] = Field(default_factory=list, description="Atom ids to remove.")
    add: list[dict] = Field(default_factory=list,
                            description="Atoms to add: id, kind (required, allowed, question, communicate, hard), "
                                        "payload (the atom's target) and an optional description.")


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    content_hash: str
    verifier_version: str
    accepted: bool
    rejected_by: list[str] = Field(default_factory=list)
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["verifiers", "history", "task_status"])


class RefuseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str


class RefuseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    admitted: bool
    finished_runs: list[str] = Field(default_factory=list)
    rulings: list[Ruling] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=lambda: ["refusals", "task_status"])


class RerollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    count: int = Field(default=1, ge=1, le=10, description="How many frontier Runs to buy (D112, D133).")


class RerollResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    runs: list[str] = Field(default_factory=list)
    finished: int = 0
    spent_usd: float = 0.0
    produced: list[str] = Field(default_factory=lambda: ["rerolls", "task_runs"])


class FindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = None
    kind: str = Field(description="What the finding is about, one of: assisted_tool (a tool no body "
                                  "cleared the gates for), fidelity (a body that replays "
                                  "differently), reference_disagreement (a Task whose recordings do "
                                  "not settle on an End state), suite (a D79 check), "
                                  "false_rejection (a Verifier that rejects every held-out Run), "
                                  "environment (the world itself), other. The name of the ruling you "
                                  "are answering is taken as the kind it is about, and so is the name "
                                  "of a tool of this build; anything else is refused with the list.")
    text: str
    run_id: Optional[str] = None
    tool: Optional[str] = None
    suggested: FindingVerb = Field(default="none",
                                   description="The Builder verb that answers this finding. `repair_intent` for "
                                               "an Intent the Runs do not evidence, `repair_recompile` for a tool "
                                               "body that is wrong, `compile_tool`, `replay` or `reroll` when a "
                                               "rebuild with nothing new to say is enough, `none` when there is "
                                               "no verb for it.")
    hint: str = Field(default="", description="The one line the suggested verb is given: for repair_intent, what "
                                              "the Task's Runs actually evidence (name the refused phrases and "
                                              "what the Runs say instead); for repair_recompile, the tool and the "
                                              "columns whose values differ on replay. A repair verb suggested "
                                              "with no hint asks for the same repair again with nothing new.")
    about_call_id: Optional[str] = Field(default=None, description="The tool call whose result the finding is about.")


class FindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    finding_id: str
    finding: dict = Field(default_factory=dict)
    produced: list[str] = Field(default_factory=lambda: ["findings"])


# --- helpers ---------------------------------------------------------------------------

def _task(plan: ExaminerPlan, task_id: str):
    task = next((t for t in plan.inputs.get("tasks") or [] if t.id == task_id), None)
    if task is None:
        raise KeyError(f"no Task is named {task_id}")
    return task


def _current(plan: ExaminerPlan, task_id: str) -> Verifier:
    verifier = plan.current(task_id)
    if verifier is None:
        # D208: a Task whose Reference was withdrawn had its Verifier retired, and repairing it
        # would be repairing a check on an End state the Task no longer says it reached.
        gone = lifecycle.retired_row((plan.store.get("task_status") or {}).get(task_id))
        why = f"; its Verifier was retired: {gone['text']}" if gone else "; derive first"
        raise LookupError(f"task {task_id} has no live Verifier{why}")
    return verifier


def _text(body: Any) -> str:
    return json.dumps(body, indent=2, sort_keys=True, default=str, ensure_ascii=False)


READ_CHARS = 60000  # the most one `read` hands the model (D175): one row of any kind fits, no whole file does
WHOLE_FILE_KINDS = ("task_status", "gates", "rerolls", "replays", "references")


NARROWER = "read one id at a time, or a narrower kind"


def _clamp(text: str, advice: str = NARROWER) -> str:
    """Any tool's text cut at READ_CHARS with the cut and what to do about it named (D175)."""
    if len(text) <= READ_CHARS:
        return text
    return f"{text[:READ_CHARS].rstrip()}\n[+{len(text) - READ_CHARS} characters cut: {advice}]"


def _clamped_text(body: Any) -> str:
    """One read's text, cut at READ_CHARS with the cut named (D175).

    A read with no id of task_status, gates or references handed the Examiner the whole file: on
    one live build its context fill reached 899 percent of the window in the first round, with
    four compactions in the second, and the beat that followed read nothing it had read before.
    A read that does not fit says how much was cut and what to read instead.
    """
    return _clamp(_text(body))


def _index(kind: str, rows: Any) -> dict:
    """What a read with no id answers for a whole-file kind: one line per id, not the file (D175)."""
    if kind == "task_status":
        out = {}
        for task_id, row in sorted((rows or {}).items()):
            row = row or {}
            parts = ["reference confirmed" if row.get("reference_confirmed") else "no reference",
                     "verifier passed" if row.get("verifier_passed") else "verifier not passed"]
            checks = row.get("checks") or {}
            failed = sorted(name for name, ok in checks.items() if ok is False) if isinstance(checks, dict) else []
            if failed:
                parts.append("failed " + ", ".join(failed))
            if row.get("blocking_tools"):
                parts.append("blocked by " + ", ".join(row["blocking_tools"]))
            # D208: a Task whose Verifier was retired reads as retired and why, so the absence of a
            # Verifier is not read as a Verifier that is merely failing its checks.
            gone = lifecycle.retired_row(row)
            if gone:
                parts.append(f"verifier retired ({gone.get('reason')})")
            out[task_id] = "; ".join(parts)
        return {"tasks": len(out), "rows": out, "note": "read task_status with an id for the whole row"}
    if kind == "gates":
        stages: dict[str, dict] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            entry = stages.setdefault(str(row.get("stage")), {"rulings": 0, "failing": 0})
            entry["rulings"] += 1
            entry["failing"] += 0 if row.get("pass", row.get("passed")) else 1
        return {"stages": stages, "note": "read gates with a stage name for its rulings"}
    counts = {key: (len(value) if isinstance(value, (dict, list)) else 1) for key, value in sorted((rows or {}).items())}
    return {"ids": counts, "note": f"read {kind} with an id for its rows"}


def _find_run_path(plan: ExaminerPlan, run_id: str) -> Optional[str]:
    for rows in (plan.store.get("replays") or {}).values():
        for row in (rows or {}).values():
            if row.get("run_id") == run_id and row.get("path"):
                return row["path"]
    for rows in (plan.store.get("rerolls") or {}).values():
        for row in rows or []:
            if row.get("run_id") == run_id and row.get("path"):
                return row["path"]
    return None


READ_IDS = 20  # how many ids an unknown id is answered with; a build has hundreds of Runs


def _task_id_of(plan: ExaminerPlan, key: Optional[str]) -> Optional[str]:
    """The Task this id names, when it names one: a model reaching for a Run often passes a Task id."""
    return key if any(task.id == key for task in plan.inputs.get("tasks") or []) else None


def _run_ids(plan: ExaminerPlan) -> list[str]:
    """Every Run id the Examiner can reach, as `task: run`, so an unknown id is answered with what exists."""
    out: list[str] = []
    for task, rows in sorted((plan.store.get("replays") or {}).items()):
        out += [f"{task}: {row.get('run_id')}" for row in (rows or {}).values() if row.get("run_id")]
    for task, rows in sorted((plan.store.get("rerolls") or {}).items()):
        out += [f"{task}: {row.get('run_id')}" for row in rows or [] if row.get("run_id")]
    for task, pool in sorted((plan.store.get("probes") or {}).items()):
        out += [f"{task}: {probe.probe_id}" for probe in pool.probes]
    return out


def _trace_ids(plan: ExaminerPlan) -> list[str]:
    """Every Trace id, as `task: trace` where a Task owns it."""
    task_of = _task_of_trace(plan)
    return [f"{task_of.get(trace.trace_id, 'no task')}: {trace.trace_id}"
            for trace in plan.inputs.get("traces") or []]


def _unknown_id(kind: str, key: Optional[str], ids: list[str], task_id: Optional[str] = None) -> dict:
    """What a read of an id no record carries answers: the ids that do exist, not a KeyError.

    One live build's Examiner read the same missing Run three times and the same missing Trace once,
    and the tool answered each with the exception and nothing to try instead. The ids of the Task
    when the id names one, else the first `READ_IDS` of the build, with the count, so the next read
    is a read of something that is there.
    """
    narrowed = [row for row in ids if task_id and row.startswith(f"{task_id}: ")] if task_id else []
    shown = (narrowed or ids)[:READ_IDS]
    where = f"of task {task_id}" if narrowed else "of this build"
    return {"kind": kind, "id": key, "found": False,
            "note": f"no {kind} is named {key!r}; the {kind} ids {where} ({len(shown)} of "
                    f"{len(narrowed or ids)})",
            "ids": shown}


def _find_run(plan: ExaminerPlan, run_id: str) -> Run:
    path = _find_run_path(plan, run_id)
    if path is not None:
        return verifier_suite.load_run(path)
    for pool in (plan.store.get("probes") or {}).values():
        for probe in pool.probes:
            if probe.probe_id == run_id or probe.run.run_id == run_id:
                return probe.run
    raise KeyError(f"no Run is named {run_id} among the replays, the re-rolls and the probes")


# What a Task with no readable Reference is answered with. Nothing can be derived from such a Task,
# so nothing can be proposed for it and nothing can score a candidate against it; that is a state the
# Task is in and not a fault in the call, so it is an answer with the state on it and never an
# exception (D230). The verbs that do buy something for such a Task are named, the way an unknown id
# is answered with the ids that exist.
NO_REFERENCE = "no_reference"
NO_REFERENCE_ALTERNATIVES = ("read the Runs it did record and why each was turned down (`read` the "
                             "`references` row of the Task, then the Runs it names), `refuse` the Task "
                             "when no frontier Run of it finished, or file a finding saying what its "
                             "Runs evidence")
# How many distinct reasons the refusal names. The reasons repeat across the Runs of one Task, so a
# handful of them with a count each says what a list of every Run would, in a fraction of the context.
NO_REFERENCE_REASONS = 5


def reference_state(plan: ExaminerPlan, task_id: str) -> dict:
    """Whether this Task's Reference can be read, and what stands in its place when it cannot.

    `reference` True carries the Reference Runs' `paths`. False carries the refusal: how many Runs
    the Task recorded and were turned down, the distinct reasons they were turned down under with a
    count each, whether a Reference is named that this session cannot read (which is a different
    thing from a Task that never had one), and the verbs that buy something instead.
    """
    references = read_json(plan.workdir / "references.json", {}) or {}
    row = references.get(task_id) or {}
    named = [str(r.get("run_id")) for r in (row.get("references") or []) if r.get("run_id")]
    paths = [path for path in (_find_run_path(plan, run_id) for run_id in named) if path]
    if paths:
        return {"task_id": task_id, "reference": True, "paths": paths}
    failed = row.get("failed") or {}
    counts: dict[str, int] = {}
    for reason in failed.values():
        text = str(reason)
        counts[text] = counts.get(text, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    why = ("a Reference is named on the Task and none of its Runs is on disk in this session"
           if named else str(row.get("reason") or "no Run of it was confirmed"))
    return {"task_id": task_id, "reference": False, "found": False,
            "note": f"task {task_id} has no Reference this session can read, so there is nothing to "
                    f"derive a Verifier from and nothing to score a version against; {why}",
            "runs_turned_down": len(failed),
            "reasons": dict(ranked[:NO_REFERENCE_REASONS]),
            "reasons_not_shown": max(0, len(ranked) - NO_REFERENCE_REASONS),
            "references_named_not_on_disk": len(named),
            "do_instead": NO_REFERENCE_ALTERNATIVES}


class NoReference(LookupError):
    """A Task asked about whose Reference cannot be read: the state travels with it (D230).

    Every tool asks `reference_state` before it does any work, and answers the model with the state
    rather than starting something it cannot finish. This is the guard under that: the one place
    that joins a Task to its Reference files raises it, so a caller that has not asked stops there
    with the state on the exception instead of failing three frames down on an empty list.
    """

    def __init__(self, state: dict) -> None:
        super().__init__(state["note"])
        self.state = state


def _reference_paths(plan: ExaminerPlan, task_id: str) -> list[str]:
    """The References' paths, references.json run_ids joined back to the replay and re-roll rows."""
    state = reference_state(plan, task_id)
    if not state["reference"]:
        raise NoReference(state)
    return state["paths"]


def _rules_trace(plan: ExaminerPlan, task_id: str) -> Optional[str]:
    references = read_json(plan.workdir / "references.json", {}) or {}
    rows = (references.get(task_id) or {}).get("references") or []
    return next((row.get("trace_id") for row in rows if row.get("trace_id")), None)


def _events(rows: list[dict]) -> list[Event]:
    out = []
    for number, row in enumerate(rows):
        body = dict(row)
        body.setdefault("idx", number)
        out.append(Event.model_validate(body))
    return out


def _may_probe(plan: ExaminerPlan) -> bool:
    return plan.probe_model is not None and plan.run_probe is not None and \
        (plan.probe_limit is None or plan.probe_limit > 0)


ATOM_SHAPE = ("an atom is an object with `id`, `kind` (required, allowed, question, communicate, hard) "
              "and `payload`, an object naming the atom's target")
# The one line the loop puts in front of the model after a shape refusal (agent/loop.py). It says
# the corrected call and asks for it once; it names no artifact of any corpus, since the shape it
# carries is built from the atoms of the Verifier the repair was called on.
_SHAPE_ASK = ("Your repair was refused for the shape of one atom, not for what it says. Call repair "
              "again with the same reason and that atom written as: {shape}")


def payload_shape(atoms: Iterable[Any], kind: Any) -> dict:
    """The payload an atom of this kind carries in the Verifier being repaired, or any atom's.

    The shape is read off the Verifier the repair starts from rather than written down here, so the
    correction is the one this Task's own atoms use and nothing about any domain is hardcoded. An
    empty Verifier leaves an empty payload, and the refusal falls back to the words of ATOM_SHAPE.
    """
    rows = [atom for atom in atoms or () if getattr(atom, "target", None)]
    for atom in rows:
        if getattr(atom, "kind", None) == kind:
            return dict(atom.target)
    return dict(rows[0].target) if rows else {}


def corrected_atom(row: dict, atoms: Iterable[Any] = ()) -> str:
    """The row the model sent, written out in the shape the tool takes: its own id, its own kind,
    its own extra fields, and a payload that is an object.

    One example is worth the enum: the refusal that names only the rule ("the payload is a str, not
    an object") was answered in one live build by not calling the tool again. The example is built
    from the row's own fields and from a sibling atom of the same kind, so it is this Task's shape
    and not a shape from anywhere else.
    """
    body = dict(row)
    body.pop("payload", None)
    body.pop("target", None)
    example = {"id": body.pop("id", None) or "atom-1", "kind": body.pop("kind", None) or "required",
               "payload": payload_shape(atoms, row.get("kind"))}
    example.update({key: value for key, value in body.items() if key != "predicate_src"})
    return json.dumps(example, sort_keys=True, default=str, ensure_ascii=False)


def _atom_of(row: dict, atoms: Iterable[Any] = ()):
    """One `add` row as an Atom, with a validation error rather than a crash on a row of the wrong shape.

    A live build's Examiner sent a payload as a string; nothing here checked, and the predicate
    writer asked it for a key three frames down, so the whole tool answered `AttributeError: 'str'
    object has no attribute 'get'` and the model spent thirteen of that session's twenty-two turns
    on the Task without landing anything. What the row must be is said once, here, in the words the
    tool's own schema uses.

    The refusal carries the corrected call and not only the rule (`corrected_atom`), and it is a
    `RetryableToolError`, so the loop puts that corrected call in front of the model once and the
    repair is tried again instead of abandoned. A second refusal with the same ask stands.
    """
    body = dict(row)
    missing = [name for name in ("id", "kind") if not body.get(name)]
    if missing:
        raise RetryableToolError(
            f"the atom {row!r} names no {', '.join(missing)}: {ATOM_SHAPE}. The same atom with "
            f"every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    atom_id, kind = body.pop("id"), body.pop("kind")
    payload = body.pop("payload", None) or body.pop("target", None) or {}
    if not isinstance(payload, dict):
        raise RetryableToolError(
            f"the payload of atom {atom_id} is {type(payload).__name__}, not an object: {ATOM_SHAPE}. "
            f"The same atom with the payload as an object: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    return make_atom(atom_id, kind, payload, helpers=verifier_suite.HELPERS_SRC, **body)


async def _emit(plan: ExaminerPlan, sink: Optional[Sink], event: Any) -> None:
    if plan.on_event is not None:
        plan.on_event(event)
    if sink is not None:
        await sink(event)


# --- the executors -----------------------------------------------------------------------

# One kind of read: what the Examiner asked for by id, or the whole file as an index when it asked
# for no id. One function per kind, and the table below is the whole surface `read` answers.
ReadHandler = Callable[[ExaminerPlan, Optional[str]], Any]


def _read_task(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The Task record this id names."""
    return as_dict(_task(plan, key or ""))


def _read_trace(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The Trace this id names, or the Trace ids that do exist when none carries it."""
    trace = next((t for t in plan.inputs.get("traces") or [] if t.trace_id == key), None)
    if trace is None:
        return _unknown_id("Trace", key, _trace_ids(plan), _task_id_of(plan, key))
    return as_dict(trace)


def _read_intent(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The whole Intent record of the Task this id names."""
    intents = plan.inputs.get("intents") or {}
    if key not in intents:
        raise KeyError(f"no Intent for task {key}")
    return as_dict(Intent.model_validate(intents[key]))


def _read_run(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The Run this id names, or the Run ids that do exist when none carries it."""
    try:
        return as_dict(_find_run(plan, key or ""))
    except KeyError:
        return _unknown_id("Run", key, _run_ids(plan), _task_id_of(plan, key))


def _read_verifier(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The live Verifier of the Task this id names."""
    return as_dict(_current(plan, key or ""))


def _read_probes(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The probe pool of the Task this id names, empty where the Task has none yet."""
    pool = (plan.store.get("probes") or {}).get(key)
    return as_dict(pool) if pool is not None else as_dict(ProbePool(task_id=key or ""))


def _rows_or_index(kind: str, rows: Any, key: Optional[str], missing: Any = None) -> Any:
    """A whole-file kind that is a mapping of id to rows: the index with no id, else the one row.

    `missing` is what an id no row carries answers, which is the empty shape of that kind's rows so
    a reader is not handed a null where it reads a list or an object.
    """
    return _index(kind, rows) if key is None else {key: rows.get(key, missing)}


def _read_task_status(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The status index of every Task, or the whole status row of the Task this id names."""
    return _rows_or_index("task_status", plan.store.get("task_status") or {}, key)


def _read_gates(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The ruling index by stage, or every ruling of the stage this id names."""
    rows = read_json(plan.workdir / "gates.json", []) or []
    if key is None:
        return _index("gates", rows)
    return [row for row in rows if row.get("stage") == key]


def _read_rerolls(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The re-roll index by Task, or the re-roll rows of the Task this id names."""
    return _rows_or_index("rerolls", plan.store.get("rerolls") or {}, key, missing=[])


def _read_replays(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The replay index by Task, or the replay rows of the Task this id names."""
    return _rows_or_index("replays", plan.store.get("replays") or {}, key, missing={})


def _read_off_file(plan: ExaminerPlan, kind: str, key: Optional[str]) -> Any:
    """The References index, or the References of the id: what a kind outside the table also reads."""
    return _rows_or_index(kind, read_json(plan.workdir / "references.json", {}) or {}, key)


def _read_references(plan: ExaminerPlan, key: Optional[str]) -> Any:
    """The References index by id, or the References the id names."""
    return _read_off_file(plan, "references", key)


READ_HANDLERS: dict[str, ReadHandler] = {
    "task": _read_task,
    "trace": _read_trace,
    "intent": _read_intent,
    "run": _read_run,
    "verifier": _read_verifier,
    "probes": _read_probes,
    "task_status": _read_task_status,
    "gates": _read_gates,
    "rerolls": _read_rerolls,
    "replays": _read_replays,
    "references": _read_references,
}


def _read(plan: ExaminerPlan):
    async def read(args: ReadArgs) -> ReadResult:
        handler = READ_HANDLERS.get(args.kind)
        # The schema's own kinds are the table's keys, so a kind outside it arrives only from a
        # caller that went round the schema; it is read off the References file, which is where the
        # if chain this table replaced sent it, with the kind kept so the index names what was asked.
        body = handler(plan, args.id) if handler is not None else _read_off_file(plan, args.kind, args.id)
        # D175: a whole-file read is an index, and no read is longer than READ_CHARS.
        return ReadResult(kind=args.kind, id=args.id, text=_clamped_text(body))

    return read


# --- search over the records ------------------------------------------------------

# One candidate line of a record: the record's id, the Task it belongs to, the tool it belongs to
# when it belongs to one, where in the record it is, and the text a match is looked for in.
Row = tuple[str, Optional[str], Optional[str], str, str]


def _pattern(text: str, regex: bool) -> "re.Pattern[str]":
    """The needle: a case-insensitive substring, or the text as a regular expression when asked.

    A flag rather than a guess at metacharacters, so a phrase holding a bracket, a dot or a dollar
    searches for itself and a call that wants a pattern says so.
    """
    if not text.strip():
        raise ValueError("search needs text to look for")
    try:
        return re.compile(text if regex else re.escape(text), re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"{text!r} is not a regular expression: {exc}") from None


def _excerpt(text: str, match: "re.Match[str]", width: int = SEARCH_LINE) -> str:
    """The line the match is on, cut to `width` characters with the match inside what is left."""
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    line = text[start:len(text) if end < 0 else end].strip()
    if len(line) <= width:
        return line
    at = max(0, match.start() - start - width // 3)
    left = min(at, len(line) - width)
    return ("..." if left > 0 else "") + line[left:left + width] + ("..." if left + width < len(line) else "")


def _match_line(kind: str, row: Row, match: "re.Match[str]") -> str:
    """One match as the model reads it: the kind and the id, the Task, where it matched, the line."""
    record, task_id, _, where, text = row
    task = f" (task {task_id})" if task_id and task_id != record else ""
    return f"{kind} {record}{task}: {where}: {_excerpt(text, match)}"


def _task_of_trace(plan: ExaminerPlan) -> dict[str, str]:
    """Which Task each Trace belongs to, off the Tasks' own run ids."""
    return {run_id: task.id for task in (plan.inputs.get("tasks") or []) for run_id in task.run_ids}


def _trace_rows(plan: ExaminerPlan, task_id: Optional[str]) -> Iterator[Row]:
    """Every Trace's turns and recorded tool calls: what the customer's agent said, called and got."""
    task_of = _task_of_trace(plan)
    for trace in plan.inputs.get("traces") or []:
        task = task_of.get(trace.trace_id)
        if task_id is not None and task != task_id:
            continue
        for turn in trace.turns:
            if turn.content:
                yield trace.trace_id, task, None, f"turn {turn.idx} {turn.role}", turn.content
        for tool_call in trace.tool_calls:
            where = f"tool call {tool_call.name}"
            yield trace.trace_id, task, tool_call.name, f"{where} args", _text(tool_call.args)
            if tool_call.result is not None:
                yield trace.trace_id, task, tool_call.name, f"{where} result", _text(tool_call.result)
            if tool_call.error is not None:
                yield trace.trace_id, task, tool_call.name, f"{where} error", _text(as_dict(tool_call.error))


def _run_paths(plan: ExaminerPlan, task_id: Optional[str]) -> Iterator[tuple[str, str, str]]:
    """Every Run the Examiner can reach as (Task, run id, path): the replay rows and the re-roll
    rows, which is where `read` with kind `run` finds a Run too."""
    for task, rows in sorted((plan.store.get("replays") or {}).items()):
        if task_id is None or task == task_id:
            for row in (rows or {}).values():
                if row.get("path"):
                    yield task, str(row.get("run_id") or Path(row["path"]).stem), row["path"]
    for task, rows in sorted((plan.store.get("rerolls") or {}).items()):
        if task_id is None or task == task_id:
            for row in rows or []:
                if row.get("path"):
                    yield task, str(row.get("run_id") or Path(row["path"]).stem), row["path"]


def _run_file(plan: ExaminerPlan, task_id: str, path: str) -> Optional[Path]:
    """The Run's file: the path the row holds, else the same name under this workdir's runs/<task>/,
    since a row written by another build's working directory names a path relative to that one."""
    file = Path(path)
    if file.is_file():
        return file
    inside = plan.workdir / "runs" / task_id / file.name
    return inside if inside.is_file() else None


def _run_rows(plan: ExaminerPlan, task_id: Optional[str], pattern: "re.Pattern[str]") -> Iterator[Row]:
    """Every Run's events. A Run file is one JSON line per event, with the Run's own fields on a line
    of their own, and the files run to megabytes; so a line is scanned as text and only a line the
    pattern already hit is parsed, to say which event it is."""
    for task, run_id, path in _run_paths(plan, task_id):
        file = _run_file(plan, task, path)
        if file is None:
            continue
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if pattern.search(line) is None:
                    continue
                body = json.loads(line) if line.lstrip().startswith("{") else {}
                payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
                name = payload.get("name") if isinstance(payload.get("name"), str) else None
                where = ("run fields" if "type" not in body else
                         f"event {body.get('idx')} {body['type']}" + (f" {name}" if name else ""))
                yield run_id, task, name, where, line.rstrip("\n")


def _intent_rows(plan: ExaminerPlan, task_id: Optional[str]) -> Iterator[Row]:
    """Every Intent: the line itself, the phrases the intent gate refused, what evidences each."""
    for task, body in sorted((plan.inputs.get("intents") or {}).items()):
        if task_id is not None and task != task_id:
            continue
        record = Intent.model_validate(body)
        yield task, task, None, "text", record.text
        for phrase in record.ungrounded_phrases:
            yield task, task, None, "ungrounded_phrase", phrase
        for phrase, run_ids in sorted(record.run_coverage.items()):
            yield task, task, None, f"run_coverage {phrase}", ", ".join(run_ids)
        for span in record.spans:
            yield task, task, None, f"span {span.source} in {span.trace_id}", span.text


def _status_rows(plan: ExaminerPlan, task_id: Optional[str]) -> Iterator[Row]:
    """Every task_status row, field by field: the reason, the checks, the tools that blocked it."""
    for task, row in sorted((plan.store.get("task_status") or {}).items()):
        if task_id is not None and task != task_id:
            continue
        for field, value in sorted((row or {}).items()):
            yield task, task, None, field, value if isinstance(value, str) else _text(value)


def _verifier_rows(plan: ExaminerPlan, task_id: Optional[str]) -> Iterator[Row]:
    """Every current Verifier's atoms, one row each."""
    for verifier in plan.store.get("verifiers") or []:
        if task_id is not None and verifier.task_id != task_id:
            continue
        for atom in verifier.atoms:
            yield verifier.task_id, verifier.task_id, None, f"atom {atom.id} {atom.kind}", _text(as_dict(atom))


def _rows(plan: ExaminerPlan, kind: str, args: SearchArgs, pattern: "re.Pattern[str]") -> Iterator[Row]:
    if kind == "trace":
        return _trace_rows(plan, args.task_id)
    if kind == "run":
        return _run_rows(plan, args.task_id, pattern)
    if kind == "intent":
        return _intent_rows(plan, args.task_id)
    if kind == "task_status":
        return _status_rows(plan, args.task_id)
    return _verifier_rows(plan, args.task_id)


def _search(plan: ExaminerPlan):
    async def search(args: SearchArgs) -> SearchResult:
        pattern = _pattern(args.text, args.regex)
        kinds = [kind for kind in SEARCH_KINDS if not args.kinds or kind in args.kinds]
        counts = {kind: 0 for kind in kinds}
        lines: list[str] = []
        for kind in kinds:
            for row in _rows(plan, kind, args, pattern):
                if args.tool is not None and row[2] != args.tool:
                    continue
                match = pattern.search(row[4])
                if match is None:
                    continue
                counts[kind] += 1
                if len(lines) < args.limit:
                    lines.append(_match_line(kind, row, match))
        total = sum(counts.values())
        head = (f"search {args.text!r} over {', '.join(kinds)}: {total} matches ("
                + ", ".join(f"{kind} {count}" for kind, count in counts.items())
                + f"), showing {len(lines)}")
        text = _clamp("\n".join([head, *lines]), "search a narrower phrase, one kind, or one Task")
        return SearchResult(matches=total, shown=len(lines), counts=counts, text=text)

    return search


def _derive(plan: ExaminerPlan, sink: Optional[Sink]):
    async def derive(args: DeriveArgs) -> DeriveResult:
        only = None if args.target == "all" else args.target
        ctx = stage_mod.ExamContext(plan.workdir, plan.ledger, anchor=plan.anchor)
        # derive_all rewrites task_status.json and references.json before the loosening gate rules:
        # the pre-derivation rows, so a rejected version restores a version-consistent artifact set.
        prior = {name: read_json(plan.workdir / name, {}) or {}
                 for name in ("task_status.json", "references.json")}
        started = time.monotonic()
        await _emit(plan, sink, StageStart(name=STAGE))
        try:
            out = await asyncio.to_thread(
                stage_mod.derive_all, ctx, plan.inputs, probe_model=plan.probe_model,
                probe_limit=plan.probe_limit, judge_model=plan.judge_model,
                judge_agent=plan.judge_agent, run_probe=plan.run_probe,
                # D189: a Task whose Reference stands alone is re-rolled for a second path, bounded
                # by the stage's cap; a round whose allowance is spent buys no batch, the same line
                # the `reroll` tool draws.
                run_rerolls=(plan.run_rerolls if plan.allowance_remaining is None
                             or plan.allowance_remaining > 0 else None),
                # D199: the replay callable is not under the allowance, since a synthesised path is
                # written by code and buys nothing.
                run_variant=plan.run_variant,
                round_number=plan.round,
                only=only, workers=plan.workers)
        except Exception as exc:
            await _emit(plan, sink, StageEnd(name=STAGE, counts={
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": int((time.monotonic() - started) * 1000)}))
            raise
        if out.get("ceiling_reached"):
            plan.ceiling_reached = True
        status = out["task_status"]
        passed = sum(1 for row in status.values() if row.get("verifier_passed"))
        counts = {"status": "ran", "tasks": len(status), "verifiers": len(out["verifiers"]), "passed": passed,
                  # D163: how many Tasks came off the per-Task cache and how many were derived again.
                  "cached": out.get("cached", 0), "ran": out.get("ran", len(status)),
                  # D189: the extra frontier Runs the search for a second path bought this call.
                  "second_path_runs": out.get("second_path_runs", 0),
                  # D208: the Verifiers this call retired, because the Reference they were derived
                  # from is no longer the one their Task holds.
                  **lifecycle.counts(out.get("retired") or []),
                  "elapsed_ms": int((time.monotonic() - started) * 1000)}
        await _emit(plan, sink, StageEnd(name=STAGE, counts=counts))
        plan.load_state()
        # D208: what the derivation retired goes into version history before anything else runs, so
        # a Verifier that leaves verifiers/ is still readable as the version it was and the reason
        # it stopped being live is on the record beside it.
        retired = _record_retirements(plan, out.get("retired") or [])
        loosening = _record_versions(plan, out["verifiers"], prior)
        rulings = [ruling_of(r) for r in ctx.recorded] + [ruling_of(r) for r in loosening]
        plan.last_rulings = rulings
        failed = [r.stage for r in rulings if not r.passed]
        # D205: before the losses are filed, the ones the harness can answer itself are answered. An
        # over-strict Verifier is loosened here by code and ruled on by the same gates a repair goes
        # through; what the gates turn down, and what no rule can relax, stays a finding below.
        loosened = auto_loosen(plan)
        # D170: the losses the records show are filed here, before the model has chosen anything, so
        # the list exists on a code-driven beat too and the ranking is what the Builder is handed.
        filed = findings_mod.file_rule_findings(plan)
        ranked = "; ".join(f"{f.kind} {f.tool or f.task_id or ''} costs {f.cost} Tasks" for f in filed[:3])
        summary = (f"derive {args.target}: {len(out['verifiers'])} Verifiers over {len(status)} Tasks, "
                   f"{passed} passed the D79 suite" + (f"; failed rulings: {', '.join(failed)}" if failed else "")
                   + (f"; {len(retired)} Verifiers retired, their Reference withdrawn" if retired else "")
                   + (f"; {loosened['auto_loosen_proposed']} over-strict Verifiers loosened by rule, "
                      f"{loosened['auto_loosen_accepted']} accepted" if loosened["auto_loosen_proposed"] else "")
                   + (f"; {len(filed)} findings filed from the records, most costly first: {ranked}"
                      if filed else ""))
        return DeriveResult(summary=summary, target=args.target, status="ran",
                            verifiers=[v.task_id for v in out["verifiers"]], passed=passed, rulings=rulings,
                            findings=[as_dict(f) for f in filed])

    return derive


def auto_loosen(plan: ExaminerPlan) -> dict:
    """The loosening step the derivation runs by itself over the Tasks the gate rules over-strict (D205).

    For each such Task the first held-out Run that reached the Reference End state and the Verifier
    rejects is taken, the atoms that reject it are relaxed by kind (`loosen.proposal_for`), and the
    result goes through `propose_version`, which is the repair tool's own path: the D79 suite, the
    probe pool and one-directional loosening rule on it before it becomes the Verifier on disk. A
    proposal the gates turn down leaves the Task exactly as it was and is recorded with
    `auto_loosen_rejected` and the check that turned it down, so the next round reads why.

    At most one proposal per Task per round and at most `MAX_ROUNDS` rounds; a Task still over-strict
    after that keeps its finding, which the Builder can no longer close (D192's gap, closed in
    rounds.py), so the Examiner is steered at it with the atoms named. An accepted proposal closes
    that finding, because the loss it reports is gone.

    Nothing here calls a model. The counts land on the round through `plan.store["auto_loosen"]`.
    """
    rows = plan.store.setdefault("auto_loosen", [])
    write_tools = write_tools_of(plan.store.get("sigs") or [])
    canon_rules = plan.store.get("canon_rules")
    task_runs = plan.store.get("task_runs") or {}
    for task_id, seen in sorted(loosen_mod.over_strict_rows(plan.store).items()):
        if not loosen_mod.may_propose(rows, task_id, plan.round):
            continue
        verifier = plan.current(task_id)
        run = next((r for r in task_runs.get(task_id, []) if r.run_id in set(seen["rejected_ids"])), None)
        # D218 rule 3: a step that ends without a proposal writes why it ended. The ways it can are a
        # Task with no Verifier to loosen, a Task whose rejected Runs are not on disk to read, a
        # Verifier no rule here can relax, and, since D230, a Task whose Reference cannot be read,
        # which has nothing to score a loosened version against and used to raise out of the step and
        # take the whole derivation with it. Each left the round with nothing recorded, so a Task the
        # loosening never touched read the same as a Task it touched and could not move.
        stopped = (loosen_mod.NO_REFERENCE if not reference_state(plan, task_id)["reference"]
                   else loosen_mod.nothing_proposed(verifier, run))
        if stopped is None and (proposal := loosen_mod.proposal_for(verifier, run, canon_rules,
                                                                   write_tools)) is None:
            stopped = loosen_mod.NO_RELAXATION
        if stopped is not None:
            rows.append({"task_id": task_id, "round": plan.round, "run_id": getattr(run, "run_id", ""),
                         "kinds": [], "atoms": [], "accepted": False, "rejected_by": [],
                         "proposed": False, "reason": stopped})
            continue
        out = propose_version(plan, task_id, drop=proposal.drop, add=proposal.add,
                              reason=proposal.reason, by="auto_loosen")
        rows.append({"task_id": task_id, "round": plan.round, "run_id": proposal.run_id,
                     "kinds": proposal.kinds, "atoms": proposal.drop, "accepted": out.accepted,
                     "rejected_by": list(out.rejected_by), "proposed": True,
                     "reason": proposal.reason if out.accepted
                               else f"{loosen_mod.REJECTED}: {', '.join(out.rejected_by)}"})
        if out.accepted:
            plan.close_findings([row["finding_id"] for row in plan.store.get("findings") or []
                                 if isinstance(row, dict) and row.get("status") == "open"
                                 and row.get("kind") == "false_rejection" and row.get("task_id") == task_id])
    plan.store["auto_loosen"] = rows
    plan.write_state()
    return loosen_mod.round_counts(rows, plan.round)


def _restore_prior_row(plan: ExaminerPlan, filename: str, store_key: Optional[str], task_id: str,
                       prior: dict) -> None:
    """One Task's pre-derivation row back, in the store and on disk: a rejected version must leave
    the artifact set version-consistent, the restored Verifier against its own metadata."""
    rows = read_json(plan.workdir / filename, {}) or {}
    before = (prior.get(filename) or {})
    if task_id in before:
        rows[task_id] = before[task_id]
    else:
        rows.pop(task_id, None)
    write_json(plan.workdir / filename, rows)
    if store_key is not None:
        store_rows = dict(plan.store.get(store_key) or {})
        if task_id in before:
            store_rows[task_id] = before[task_id]
        else:
            store_rows.pop(task_id, None)
        plan.store[store_key] = store_rows


def _record_retirements(plan: ExaminerPlan, retired: list[dict]) -> list[dict]:
    """One history row per Verifier the derivation retired, and the finding it closes (D208).

    The row is not accepted and never becomes the Task's current version: a retirement is the end of
    a version, not the proposal of one, and the reason names which of the two ways the source went.
    Keeping it in history is what lets the next round read that this Task had a Verifier and why it
    stopped being one, without the file staying somewhere a gate can score it.

    Any finding filed about that Verifier is closed in the same step. The loss it reported was
    measured against a Verifier the harness no longer stands behind, so leaving it open would ask
    the Builder to repair an artefact that is gone; the Task's own loss is that it has no Reference,
    which its status row already says and the reference rules already file.
    """
    if not retired:
        return []
    history = plan.store.setdefault("history", {})
    for row in retired:
        record = row["verifier"]
        hist = history.get(row["task_id"]) or VerifierHistory(task_id=row["task_id"])
        hist.versions.append(VerifierVersion(
            task_id=row["task_id"], content_hash=version_hash(record),
            verifier_version=str(len(hist.versions) + 1),
            parent_hash=next((v.content_hash for v in reversed(hist.versions) if v.accepted), None),
            round=int(row.get("round") or 0), by="derive", reason=f"{row['reason']}: {row['text']}",
            accepted=False, rejected_by=[row["reason"]], verifier=record))
        history[row["task_id"]] = hist
    plan.write_state()
    plan.close_findings([f["finding_id"] for f in plan.store.get("findings") or []
                         if isinstance(f, dict) and f.get("status") == "open"
                         and f.get("kind") == "false_rejection"
                         and f.get("task_id") in {row["task_id"] for row in retired}])
    return retired


def _record_versions(plan: ExaminerPlan, verifiers: list, prior: dict) -> list[GateResult]:
    """One `derive` row per Task whose derived hash is not the history's last row (D127).

    A Task's first version is accepted as derived. A Task that already has an accepted version gets
    the derived one as a new version like any other: the loosening gate rules on it against the last
    accepted version, and when it fails (a derivation that would undo a repair's tightening) the row
    is recorded rejected and the accepted version is put back on disk together with its own
    task_status.json and references.json rows, from `prior`. The loosening rulings are recorded in
    gates.json and returned.
    """
    history = plan.store.setdefault("history", {})
    rulings: list[GateResult] = []
    changed = False
    rejected = False
    for record in verifiers:
        digest = version_hash(record)
        hist = history.get(record.task_id) or VerifierHistory(task_id=record.task_id)
        if hist.versions and hist.versions[-1].content_hash == digest:
            continue
        accepted = [v for v in hist.versions if v.accepted]
        row = VerifierVersion(
            task_id=record.task_id, content_hash=digest, verifier_version=str(len(hist.versions) + 1),
            parent_hash=accepted[-1].content_hash if accepted else None, round=plan.round, by="derive",
            reason="derived from the References", accepted=not accepted, verifier=record)
        if accepted:
            trial = hist.model_copy(deep=True)
            trial.versions.append(row)
            ruling = plan.ledger.record(STAGE, loosening_gate(
                {record.task_id: trial}, plan.store.get("task_runs") or {}, plan.store.get("replays") or {},
                plan.store.get("rerolls") or {}, plan.store.get("canon_rules"), plan.store.get("sigs") or []))
            rulings.append(ruling)
            if ruling.passed:
                row = row.model_copy(update={"accepted": True})
            else:
                row = row.model_copy(update={"rejected_by": ["loosening"]})
                plan.set_current(accepted[-1].verifier)
                _restore_prior_row(plan, "task_status.json", "task_status", record.task_id, prior)
                _restore_prior_row(plan, "references.json", None, record.task_id, prior)
                rejected = True
        hist.versions.append(row)
        history[record.task_id] = hist
        changed = True
    if changed:
        plan.write_state()
    if rejected:
        # derive_all rewrote scorecard.json from the rejected rows before the gate ruled; recompute
        # it from the restored files so the customer-facing report reads the version-consistent world.
        # Recomputed, not snapshotted: other Tasks' accepted derivations stay counted.
        write_json(plan.workdir / "scorecard.json", scorecard_mod.scorecard(plan.workdir))
    return rulings


def _probe(plan: ExaminerPlan):
    async def probe(args: ProbeArgs) -> ProbeResult:
        _task(plan, args.task_id)
        verifier = _current(plan, args.task_id)
        pools = plan.store.setdefault("probes", {})
        admission = probe_admission_gate(pools, plan.store["verifiers"])
        if args.task_id in admission.metrics.get("closed", []):
            reason = next((f for f in admission.failures if f.startswith(f"task {args.task_id}:")),
                          admission.failures[0] if admission.failures else "closed")
            raise PermissionError(f"probe refused: {reason}")
        pool = pools.get(args.task_id) or ProbePool(task_id=args.task_id)
        probe_id = f"probe-{args.task_id}-{len(pool.probes) + 1}"
        run = Run(run_id=probe_id, env_id=plan.env_id, task_id=args.task_id, model="probe:examiner",
                  events=_events(args.events), termination_reason=args.termination_reason,
                  parent_run_id=args.base_run_id)
        write_tools = write_tools_of(plan.store.get("sigs") or [])
        passed, failing = check_run(verifier, run, plan.store.get("canon_rules"), write_tools=write_tools)
        digest = version_hash(verifier)
        pool.probes.append(Probe(probe_id=probe_id, task_id=args.task_id, bug_class=args.bug_class, note=args.note,
                                 base_run_id=args.base_run_id, verifier_hash=digest, round=plan.round,
                                 scored_pass=bool(passed), run=run))
        pools[args.task_id] = pool
        plan.write_state()
        pool_gate = plan.ledger.record("probe", probe_pool_gate(plan.store["verifiers"], pools,
                                                                 plan.store.get("canon_rules"),
                                                                 plan.store.get("sigs") or []))
        admission = plan.ledger.record("probe", probe_admission_gate(pools, plan.store["verifiers"]))
        rulings = [ruling_of(pool_gate), ruling_of(admission)]
        plan.last_rulings = rulings
        tail = consecutive_failed(pool, digest)
        verdict = "scores a pass (the Verifier is loose)" if passed else f"rejected at {failing}"
        summary = (f"probe {probe_id} ({args.bug_class}) on task {args.task_id}: {verdict}; "
                   f"pool holds {len(pool.probes)}, {tail} rejected in a row against version {digest[:12]}")
        return ProbeResult(summary=summary, probe_id=probe_id, task_id=args.task_id, scored_pass=bool(passed),
                           failing_atom=failing if not passed else None, pool_size=len(pool.probes),
                           consecutive_failed=tail, rulings=rulings)

    return probe


# How many times one check may reject one Task's repairs in a session before the tool stops taking
# them. The probe skill already stops after three consecutive probes against one version (D133,
# `PROBE_STOP`); a repair had no such stop, and one live build's Examiner spent a whole session on
# 13 repair calls of which none was accepted, two Tasks repaired four times each with the same gate
# failing every time. Two rejections by one check say the check is not what the rewrite can move.
REPAIR_STOP = 2
# The two verbs that buy what a third repair cannot: a Task no frontier Run finished is refused
# (D128), and a check that never had a second finished Run to score is bought with a re-roll and a
# derivation (D170's `reroll_then_derive`), not with another version of the atoms.
REPAIR_ALTERNATIVES = "refuse, reroll_then_derive"


def repair_lock(rejections: dict[tuple[str, str], list[str]], task_id: str,
                limit: int = REPAIR_STOP) -> Optional[str]:
    """Why a further repair of this Task is refused, or None when it is still open.

    The count is per Task and per check, so a Task rejected once by one check and once by another
    is still open: two different checks are two different things to answer, and the next version can
    answer both. A check that has rejected the same Task `limit` times is the one that says the
    rewrite is not what moves it, and the message names it, both rejections and the two verbs that
    do buy something.
    """
    for (task, check), rejected in sorted(rejections.items()):
        if task == task_id and len(rejected) >= limit:
            return (f"task {task_id} has already been rejected {len(rejected)} times by {check} in this "
                    f"session ({'; '.join(rejected)}); a further repair against the same check is "
                    f"refused. What buys something instead: {REPAIR_ALTERNATIVES}.")
    return None


@dataclass
class Proposed:
    """What one proposed Verifier version came to: the candidate, the gates' answer and their rulings."""
    candidate: Verifier
    digest: str
    accepted: bool
    rejected_by: list[str]
    rulings: list[Ruling]


def propose_version(plan: ExaminerPlan, task_id: str, *, drop: Iterable[str], add: Iterable[Any],
                    reason: str, by: str) -> Proposed:
    """One new Verifier version through the gates that rule on a repair, whoever proposed it.

    The path is the `repair` tool's, lifted out of it so that nothing can propose a version by
    another route: the D79 suite over the candidate, the Task's probe pool and one-directional
    loosening all rule before anything is written, the version is kept in the history either way with
    the checks that rejected it, and only an accepted one becomes the file on disk. `by` is what the
    history row records as the proposer, which is how a reader tells the model's repair from the
    harness's own loosening step (D205), and it changes nothing about what the gates ask.

    `add` rows are atoms or the objects the tool's schema takes; a row of the wrong shape is refused
    by `_atom_of` with the corrected call, as it always was.
    """
    task = _task(plan, task_id)
    current = _current(plan, task_id)
    history = plan.store.setdefault("history", {})
    hist = history.get(task_id) or VerifierHistory(task_id=task_id)
    if not hist.versions:
        hist.versions.append(VerifierVersion(
            task_id=task_id, content_hash=version_hash(current), verifier_version=str(1),
            round=plan.round, by="derive", reason="the Verifier on disk before any repair",
            accepted=True, verifier=current))
    dropped = set(drop)
    added = [row if isinstance(row, Atom) else _atom_of(row, current.atoms) for row in add]
    atoms = [a for a in current.atoms if a.id not in dropped] + added
    if not atoms:
        raise ValueError("a repair cannot leave the Verifier without atoms")
    candidate = current.model_copy(deep=True, update={
        "atoms": atoms, "verifier_version": str(len(hist.versions) + 1)})
    canon_rules = plan.store.get("canon_rules")
    sigs = plan.store.get("sigs") or []
    write_tools = write_tools_of(sigs)
    paths = _reference_paths(plan, task_id)
    intents = plan.inputs.get("intents") or {}
    task_for = (apply_intent(task, Intent.model_validate(intents[task.id])) if task.id in intents else task)
    gates = stage_mod.suite_for(task_for, candidate, paths, canon_rules=canon_rules, write_tools=write_tools,
                                user_rules=plan.inputs.get("user_rules") or {},
                                rules_trace=_rules_trace(plan, task_id), probe_model=plan.probe_model,
                                run_probe=plan.run_probe, may_probe=_may_probe(plan))
    results = verifier_suite.d79_results(gates)
    d79 = artifacts.verifier_gate(results)
    pools = plan.store.get("probes") or {}
    pool_gate = probe_pool_gate([candidate], {k: v for k, v in pools.items() if k == task_id},
                                canon_rules, sigs)
    trial = hist.model_copy(deep=True)
    digest = version_hash(candidate)
    row = VerifierVersion(task_id=task_id, content_hash=digest, verifier_version=candidate.verifier_version,
                          parent_hash=version_hash(current), round=plan.round, by=by, reason=reason,
                          accepted=False, verifier=candidate)
    trial.versions.append(row)
    loosening = loosening_gate({task_id: trial}, plan.store.get("task_runs") or {},
                               plan.store.get("replays") or {}, plan.store.get("rerolls") or {},
                               canon_rules, sigs)
    rejected_by = [g.stage for g in gates if not g.passed]
    if not pool_gate.passed:
        rejected_by.append(pool_gate.stage)
    if not loosening.passed:
        rejected_by.append(loosening.stage)
    accepted = d79.passed and pool_gate.passed and loosening.passed
    row = row.model_copy(update={"accepted": accepted, "rejected_by": rejected_by})
    hist.versions.append(row)
    history[task_id] = hist
    if accepted:
        plan.set_current(candidate)
        status = plan.store.setdefault("task_status", {})
        entry = dict(status.get(task_id) or {})
        entry.update({"verifier_passed": True, "checks": results,
                      "not_run": [g.stage for g in gates if g.metrics.get("skipped")]})
        status[task_id] = entry
        write_json(plan.workdir / "task_status.json", status)
    plan.write_state()
    # The build-wide rulings land in gates.json; the result carries the candidate's own.
    plan.ledger.record(by, probe_pool_gate(plan.store["verifiers"], pools, canon_rules, sigs))
    plan.ledger.record(by, loosening_gate(history, plan.store.get("task_runs") or {},
                                          plan.store.get("replays") or {},
                                          plan.store.get("rerolls") or {}, canon_rules, sigs))
    rulings = [ruling_of(d79), ruling_of(pool_gate), ruling_of(loosening)]
    plan.last_rulings = rulings
    return Proposed(candidate=candidate, digest=digest, accepted=accepted, rejected_by=rejected_by,
                    rulings=rulings)


def _repair(plan: ExaminerPlan):
    # One session's rejections, per Task and per check, in the order they happened (REPAIR_STOP).
    rejections: dict[tuple[str, str], list[str]] = {}

    async def repair(args: RepairArgs) -> RepairResult:
        locked = repair_lock(rejections, args.task_id)
        if locked is not None:
            raise PermissionError(f"repair refused: {locked}")
        # D230: a Task with no readable Reference cannot be repaired, and that is an answer with the
        # Task's state on it rather than an error: the model is told what the Runs it does have came
        # to and which verbs buy something, and the session goes on.
        state = reference_state(plan, args.task_id)
        if not state["reference"]:
            return RepairResult(summary=f"repair of task {args.task_id} refused: {_text(state)}",
                                task_id=args.task_id, content_hash="", verifier_version="",
                                accepted=False, rejected_by=[NO_REFERENCE], rulings=[], produced=[])
        out = propose_version(plan, args.task_id, drop=args.drop, add=args.add, reason=args.reason, by="repair")
        for check in out.rejected_by:
            rejections.setdefault((args.task_id, check), []).append(
                f"version {out.candidate.verifier_version}: {args.reason}")
        outcome = "accepted" if out.accepted else f"rejected by {', '.join(out.rejected_by)}"
        summary = (f"repair of task {args.task_id}: version {out.candidate.verifier_version} "
                   f"({out.digest[:12]}) {outcome}; {len(out.candidate.atoms)} atoms, "
                   f"{len(args.drop)} dropped, {len(args.add)} added")
        return RepairResult(summary=summary, task_id=args.task_id, content_hash=out.digest,
                            verifier_version=out.candidate.verifier_version, accepted=out.accepted,
                            rejected_by=out.rejected_by, rulings=out.rulings)

    return repair


def _refuse(plan: ExaminerPlan):
    async def refuse(args: RefuseArgs) -> RefuseResult:
        _task(plan, args.task_id)
        replays, rerolls = plan.store.get("replays") or {}, plan.store.get("rerolls") or {}
        finished = finished_runs(args.task_id, replays, rerolls)
        trial = {**(plan.store.get("refusals") or {}),
                 args.task_id: {"task_id": args.task_id, "reason": args.reason, "round": plan.round}}
        ruling = refuse_gate(trial, replays, rerolls)
        if args.task_id not in ruling.metrics.get("refused", []):
            raise PermissionError(f"refusal of task {args.task_id} rejected: the frontier finished it: "
                                  f"{', '.join(finished)}")
        refusal = Refusal(task_id=args.task_id, reason=args.reason, round=plan.round, admitted=True,
                          finished_runs=finished)
        plan.store.setdefault("refusals", {})[args.task_id] = as_dict(refusal)
        status = plan.store.setdefault("task_status", {})
        entry = dict(status.get(args.task_id) or {})
        entry["refused"] = {"reason": args.reason, "round": plan.round}
        status[args.task_id] = entry
        write_json(plan.workdir / "task_status.json", status)
        plan.write_state()
        recorded = plan.ledger.record("refuse", refuse_gate(plan.store["refusals"], replays, rerolls))
        rulings = [ruling_of(recorded)]
        plan.last_rulings = rulings
        return RefuseResult(summary=f"task {args.task_id} refused: {args.reason}", task_id=args.task_id,
                            admitted=True, finished_runs=finished, rulings=rulings)

    return refuse


def _reroll(plan: ExaminerPlan):
    async def reroll(args: RerollArgs) -> RerollResult:
        _task(plan, args.task_id)
        if plan.run_rerolls is None:
            raise RuntimeError("no Runner for re-rolls in this session: the Builder handed over no reroll callable")
        if plan.allowance_remaining is not None and plan.allowance_remaining <= 0:
            raise RuntimeError(f"the round's allowance is spent ({plan.allowance_remaining:.2f} USD left)")
        prefix = f"reroll-r{plan.round}"
        # `_candidate_runs` numbers Runs from zero under the prefix, so a second call in the same round
        # takes a longer prefix rather than writing over the first call's files. D212: the attempt
        # index is read off this Task's own rows and grows upward, so no other Task's re-rolls are an
        # input to it and the indexes already taken are never reshuffled.
        existing = {str(row.get("run_id")) for row in (plan.store.get("rerolls") or {}).get(args.task_id) or []}
        number = 0
        while any(run_id.startswith(f"{prefix}-{args.task_id}-") for run_id in existing):
            number += 1
            prefix = f"reroll-r{plan.round}-{number}"
        before = plan.spend()
        try:
            rows = await asyncio.to_thread(plan.run_rerolls, args.task_id, args.count, prefix)
        except budget.BudgetExceeded:
            plan.ceiling_reached = True
            raise
        rows = [dict(row) for row in rows or []]
        spent = max(0.0, plan.spend() - before)
        if plan.allowance_remaining is not None:
            plan.allowance_remaining -= spent
        plan.extra_rerolls.setdefault(args.task_id, []).extend(rows)
        plan.write_state()
        plan.load_state()
        finished = sum(1 for row in rows if (row.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS)
        summary = (f"reroll of task {args.task_id}: {len(rows)} Runs, {finished} finished, {spent:.4f} USD; "
                   f"the Task now has {len(plan.store['rerolls'].get(args.task_id, []))} re-rolls")
        return RerollResult(summary=summary, task_id=args.task_id, runs=[row.get("run_id") for row in rows],
                            finished=finished, spent_usd=spent)

    return reroll


def _finding(plan: ExaminerPlan):
    async def finding(args: FindingArgs) -> FindingResult:
        # The kind a ruling name means, so a finding filed under the ruling it answers lands instead
        # of being refused; an unknown name raises with the kinds and the mapping.
        kind = finding_kind(args.kind, _tool_names(plan))
        # D170: one loss is one finding. A key already open is refused with the id of the finding
        # that holds it, so a model that files the same thing twenty-four times to get seven on the
        # list is told which one it already has instead of being counted again.
        key = findings_mod.finding_key(kind, args.tool or "", args.task_id or "")
        existing = findings_mod.open_by_key(plan).get(key)
        if existing is not None:
            raise ValueError(f"{existing} already says this ({kind}"
                             + (f" on {args.tool}" if args.tool else "")
                             + (f" on task {args.task_id}" if args.task_id else "")
                             + "); it is open and the Builder has not answered it yet. Act on it, or file "
                               "a finding that says something else.")
        about = plan.entry_id_for(args.about_call_id) if args.about_call_id else None
        record = findings_mod.file_finding(
            plan, kind=kind, text=args.text, key=key, suggested=args.suggested, hint=args.hint,
            task_id=args.task_id, task_ids=[args.task_id] if args.task_id else (), tool=args.tool,
            run_id=args.run_id, about_entry_id=about)
        summary = f"finding {record.finding_id} ({kind}) filed for the Builder" + \
                  (f" on task {args.task_id}" if args.task_id else "") + \
                  (f", suggested {args.suggested}" if args.suggested != "none" else "") + \
                  (" with a hint" if record.hint else "")
        return FindingResult(summary=summary, finding_id=record.finding_id, finding=as_dict(record))

    return finding


def examiner_tools(plan: ExaminerPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The eight tools over one plan; `sink` is where the derive stage's events go (the harness's `emit`)."""
    return [
        AgentTool("read", "Read a Task, a Trace, an Intent, a Run, a Verifier, a probe pool, the task status, "
                  "the rulings, the re-roll or replay rows, or the References, as JSON.",
                  ReadArgs, ReadResult, _read(plan), render=render),
        AgentTool("search", "Find one phrase across the records without reading them: the Traces, the Runs, "
                  "the Intents, the task status and the Verifiers. Answers a count per kind and one line "
                  "per match, saying which record, which Task and where in it the phrase is.",
                  SearchArgs, SearchResult, _search(plan), render=render),
        AgentTool("derive", "Derive one Verifier per Task from its References through the D79 suite "
                  "(`all`, or one Task id).", DeriveArgs, DeriveResult, _derive(plan, sink), render=render),
        AgentTool("probe", "Score a hand-written Run against a Task's current Verifier and keep it in the "
                  "Task's pool forever (D127).", ProbeArgs, ProbeResult, _probe(plan), render=render),
        AgentTool("repair", "Propose a new Verifier version by dropping and adding atoms; the D79 suite, the "
                  "pool and the loosening gate decide whether it is accepted.",
                  RepairArgs, RepairResult, _repair(plan), render=render),
        AgentTool("refuse", "Give a Task up; admitted only when no frontier Run of it finished (D128).",
                  RefuseArgs, RefuseResult, _refuse(plan), render=render),
        AgentTool("reroll", "Buy more frontier Runs of a Task through the Runner (D112, D133).",
                  RerollArgs, RerollResult, _reroll(plan), render=render),
        AgentTool("finding", "File what is wrong on the Builder's side: an assisted tool, a fidelity gap, a "
                  "Reference disagreement, the Environment. Name the Builder verb that answers it in "
                  "`suggested` and what that verb needs to know in `hint`; the Builder is handed the two "
                  "as a line it can call.", FindingArgs, FindingResult, _finding(plan),
                  render=render),
    ]
