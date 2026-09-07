"""The round driver (D126, D128): one round is a Builder beat then an Examiner beat on one stream.

The two agents never run at once. The driver holds both harnesses, one shared subscriber list and
the Builder's plan; a round starts with `RoundStart`, hands the stream to the Builder (`BeatStart`
then `BeatEnd`), hands it to the Examiner the same way, and ends with `RoundEnd` carrying the counts
`gates.round_end` computed off the rulings, never off a model. The Examiner's findings are collected
off its `ToolExecutionEnd` events and delivered to the Builder at its next beat as follow-ups with
the record in `details` (D123); the code driver acts on them itself by calling the Builder tool the
finding names. Three exits: `done` when D126's state holds, `stalled` after `stall_rounds` rounds
that moved nothing, `ceiling` when the Builder stopped at the spend ceiling, the Examiner
reported the ceiling, or an agent spent its allowance two rounds in a row. The allowance is per
agent and per round: `allowance_usd` when given, otherwise the agent's own round-1 spend from
round 2 on. A model-driven agent that crosses it is steered once to finish with what it has.

The Builder's session has no turn cap (D135): it runs until the model answers with no tool call.
The ceiling is the hard stop, watched here and in `builder/agent.py` on every tool end. Stalled is
the soft stop: a round that moved nothing sends the model one follow-up saying so, naming what is
still pending and which verbs change an artifact, and the round ends if it still stops. The
Examiner's later rounds are steered with `examiner_round_message`, which asks for the derive this
beat requires.

A round moved when a gate count moved, when a model-written artifact changed (`MODEL_ARTIFACTS`,
fingerprinted into the round's counts as `artifacts` and `artifacts_changed`), or when a repair
this round made left the gates ruling differently from the round before (`round_moved`). The
artifact and the repair are there because a round in which the mechanic rewrote a tool body or
grew a table did something the Examiner has not derived over yet: no count could have moved for it
by the time the round ends, and calling that a stall stops a build that was working. Round 1 is
compared against the fingerprint the driver takes as it starts, so `artifacts_changed` says the
same thing in a first round as in a fifth. Whether one repair changed anything is the repair's own
answer, hashed on its own target when it ran (`builder.repair.change_of`) and read back here.

This module is the top of the layering: it imports both applications, and the Builder's artifacts
reach the Examiner through `examiner.stage.DERIVE_INPUTS`, never bodies, the db, the schema or the
Environment (D123).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Optional

from kullback.agent.events import BeatEnd, BeatStart, RoundEnd, RoundStart, StageEnd, StageStart, ToolExecutionEnd
from kullback.agent.harness import AgentHarness
from kullback.agent.session import SessionStore
from kullback.agent.tools import ToolResult
from kullback.ai.provider import Model
from kullback.builder import agent as builder_agent
from kullback.builder import build as build_module
from kullback.builder import pipeline
from kullback.builder import repair as repair_module
from kullback.builder.agent import builder_message
from kullback.builder.build import DEFAULT_REROLLS, TARGET_ALL, BuildError, BuildPlan
from kullback.builder.tools import BUILD_TOOLS
from kullback.examiner import agent as examiner_agent
from kullback.examiner.agent import ExaminerError, examiner_message, examiner_round_message
from kullback.examiner.plan import STATE_DIR, ExaminerPlan
from kullback.examiner.stage import DERIVE_INPUTS
from kullback.gates import round_end
from kullback.gates.ledger import HISTORY_NAME, GateLedger
from kullback.runner import budget, feed
from kullback.runner.records import (
    Finding,
    GateResult,
    RoundRecord,
    as_dict,
    content_hash,
    read_json,
    write_json,
)

ROUNDS_NAME = "rounds.json"
GATES_NAME = "gates.json"
AGENTS = ("builder", "examiner")
MAX_TURNS = 8  # the Examiner's cap; the Builder runs with none (D135)
ALLOWANCE_STEER = "Your allowance for this round is spent: finish with what you have (D123)"
# The first sentence of the stall follow-up; the rest names the verbs that can change an artifact,
# which only the plan's own registry knows (builder.agent.nothing_changed_message).
STALL_FOLLOW_UP = builder_agent.NOTHING_CHANGED
EXAMINER_TARGET = "all"
BUILDER_SESSION = Path("builder") / "session.jsonl"
EXAMINER_SESSION = Path("examiner") / "session.jsonl"

# The artifacts a model writes and a repair verb rewrites, by the name a round's counts call them.
# Each is a stage output of `builder.build`: `compile_tools` writes bodies.json, `starting_state`
# writes db.json, the `policy` stage writes constraints.json (the compiled predicates; policy.json
# and policy_coverage.json beside it are counts of that file), and the `intent` stage writes one
# file per Task under intents/. The Simulated user rules are deliberately not here: `user_sim`
# derives them from the trace's own user turns in code, no repair verb rewrites them, and a hash
# over them would only ever move when the traces did.
MODEL_ARTIFACTS: dict[str, str] = {
    "bodies": "bodies.json",
    "intents": "intents",
    "policy": "constraints.json",
    "starting_state": "db.json",
}
# The artifact each acting repair verb rewrites (`builder.tools.repair_verb_tools`). The map lives
# in builder/repair.py, where the verbs hash that artifact either side of their own call, so the
# name a round's counts give an artifact and the bytes a repair is measured on cannot drift apart.
REPAIR_ARTIFACT: dict[str, str] = repair_module.REPAIR_ARTIFACT


def _read_json(path: Path, fallback: Any) -> Any:
    """One JSON file the driver reads for a decision; a missing or half-written file is the fallback."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _artifact_body(path: Path) -> Any:
    """What one artifact holds: a directory is its *.json children by name, a file its text, a
    workdir that has not written it yet is None."""
    if path.is_dir():
        return {child.name: _text(child) for child in sorted(path.glob("*.json"))}
    return _text(path)


def _text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def artifact_hashes(workdir: Any) -> dict[str, str]:
    """A short hex per model-written artifact, over the bytes this workdir holds (`MODEL_ARTIFACTS`).

    An artifact the build never wrote hashes as None rather than being left out, so the map has the
    same keys every round and a first write reads as a change like any other.
    """
    return {name: content_hash(_artifact_body(Path(workdir) / relative))[:12]
            for name, relative in MODEL_ARTIFACTS.items()}


def artifact_fingerprint(workdir: Any) -> tuple[str, dict[str, str]]:
    """The one short hex that stands for every model-written artifact, and the hash of each."""
    per = artifact_hashes(workdir)
    return content_hash(per)[:12], per


def _stage_rulings(rows: Any) -> dict[str, bool]:
    """A gates.json body as `{stage: passed}`, which is what two rounds are compared on."""
    return {str(row.get("stage")): bool(row.get("pass"))
            for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)}


def _since_last_move(records: list[RoundRecord]) -> list[RoundRecord]:
    """The rounds from the last one that moved something onward, which is what `exit_for` is given.

    `gates.round_end` is frozen and its `stalled` reads a window of D126's gate counts: it cannot
    see an artifact that changed or a ruling a repair moved. Handing it the tail that begins at the
    last round that moved says the same thing in the terms it has. The tail is one round that moved
    followed by the rounds that moved nothing, so `len(tail) > stall_rounds` is exactly
    `stall_rounds` rounds that moved nothing; every round in the tail carries the same gate counts,
    since a round whose counts moved moved; and `done` and `ceiling`, which read the last record,
    see the record they would have seen anyway.
    """
    moved = [i for i, record in enumerate(records) if (record.counts or {}).get("moved")]
    return records[moved[-1]:] if moved else records


def finding_message(finding: Finding) -> str:
    """A finding as the Builder reads it; the record itself rides in the message's details.

    The suggestion is rendered as the call the Builder can make, arguments and all, so a hint the
    Examiner wrote reaches the verb verbatim instead of being paraphrased back out of the prose.
    """
    text = f"Finding {finding.finding_id} ({finding.kind}): {finding.text}"
    if finding.task_id:
        text += f" Task {finding.task_id}."
    if finding.tool:
        text += f" Tool {finding.tool}."
    if finding.suggested != "none":
        text += f" Suggested: {suggested_call(finding)}"
    return text


def suggested_call(finding: Finding) -> str:
    """The Builder tool a finding suggests, written as the call: `repair_intent(task_id='t1', hint='...')`."""
    arguments = ", ".join(f"{name}={value!r}" for name, value in finding_arguments(finding).items())
    return f"{finding.suggested}({arguments})"


def finding_arguments(finding: Finding) -> dict:
    """The arguments of the Builder tool a finding suggests.

    The repair verbs take the hint the Examiner wrote (`repair_intent(task_id, hint)`,
    `repair_recompile(name, hint)`, builder/repair.py); the rebuild verbs take the tool by name, or
    the Task, and have nothing to be told.
    """
    if finding.suggested == "compile_tool":
        return {"name": finding.tool or ""}
    if finding.suggested == "repair_recompile":
        return {"name": finding.tool or "", "hint": finding.hint}
    if finding.suggested == "repair_intent":
        return {"task_id": finding.task_id or "", "hint": finding.hint}
    return {"task": finding.task_id or ""}


def as_dict_event(event: Any) -> Optional[dict]:
    """A round or beat event as the dict a screen reads; anything else is not the driver's to map."""
    kind = getattr(event, "type", None)
    if kind in ("round_start", "round_end"):
        return {"kind": "round", "state": "start" if kind == "round_start" else "end", "round": event.round,
                "counts": dict(getattr(event, "counts", None) or {}), "exit": getattr(event, "exit", None)}
    if kind in ("beat_start", "beat_end"):
        return {"kind": "beat", "state": "start" if kind == "beat_start" else "end",
                "agent": event.agent, "round": event.round, "spend": float(getattr(event, "spend", 0.0) or 0.0)}
    return None


def _dict_sink(on_event: Optional[Callable[[dict], Any]]) -> Optional[Callable[[Any], None]]:
    """The Examiner's stage events for the dict stream the Builder's pipeline writes: a typed
    StageStart or StageEnd becomes the `{"kind": "stage", ...}` dict a screen already reads, a dict
    passes through, anything else is not the screen's. A screen that raises never fails a beat."""
    if on_event is None:
        return None

    def sink(event: Any) -> None:
        if isinstance(event, dict):
            row: Optional[dict] = event
        elif isinstance(event, StageStart):
            row = {"kind": "stage", "stage": event.name, "state": "start", "attempt": 1}
        elif isinstance(event, StageEnd):
            counts = dict(event.counts or {})
            row = {"kind": "stage", "stage": event.name, "state": str(counts.get("status") or "ran"),
                   "attempt": int(counts.get("attempts") or 1)}
        else:
            row = as_dict_event(event)
        if row is None:
            return
        try:
            on_event(row)
        except Exception:
            pass

    return sink


def _open_findings(workdir: Any) -> list[Finding]:
    """The findings an earlier invocation left open, oldest first; none when there are none.

    Rows that do not validate are skipped, not fatal: a half-written findings file must not stop
    the loop that would drain it."""
    try:
        rows = json.loads(Path(workdir, STATE_DIR, "findings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("status") == "open":
            try:
                out.append(Finding.model_validate(row))
            except ValueError:
                continue
    return out


def write_rounds(workdir: Any, rounds: Iterable[RoundRecord]) -> Path:
    """rounds.json: one RoundRecord per round, the exit on the last.

    A record's `counts` carry D126's gate counts and, beside them, what only the driver saw:
    `started_at` and `ended_at` (the round's clock, which is where a build's duration is read from),
    `spend`, `turns` and `context_fill` per agent, the compactions and the findings
    (`driver_counts`), the artifact fingerprint with the artifacts this round changed, the repairs
    it made, and `moved`, which is the whole of why it did or did not count toward a stall.
    """
    path = Path(workdir) / ROUNDS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([as_dict(r) for r in rounds], indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path


def load_rounds(workdir: Any) -> list[RoundRecord]:
    """The rounds a workdir recorded, in order; none when the file is missing."""
    path = Path(workdir) / ROUNDS_NAME
    if not path.is_file():
        return []
    body = json.loads(path.read_text(encoding="utf-8"))
    return [RoundRecord.model_validate(row) for row in (body if isinstance(body, list) else [])]


def _session(workdir: Path, name: Path) -> SessionStore:
    """A fresh session file for this run: one run of the loop is one session per agent."""
    path = Path(workdir) / name
    if path.exists():
        path.unlink()
    return SessionStore(path)


def _builder_harness(plan: BuildPlan, agent_model: Optional[Model], subscribers: list) -> AgentHarness:
    """The Builder's harness with its own session file (D128) and no turn cap (D135)."""
    return builder_agent.build_harness(plan, agent_model, subscribers,
                                       session=_session(plan.workdir, BUILDER_SESSION))


def _handover(store: dict) -> dict:
    """The Builder's artifacts the Examiner may read: DERIVE_INPUTS and nothing else (D123)."""
    return {name: store[name] for name in DERIVE_INPUTS if name in store}


def _runners(plan: BuildPlan) -> tuple[Optional[Callable], Optional[Callable]]:
    """The Builder's two Runner callables, the loophole probe and the re-roll (D120), or none of either
    when the Environment is not in the store (a narrowed target built nothing to run in)."""
    if "environment" not in plan.store:
        return None, None
    return build_module.probe_runner(plan), build_module.reroll_runner(plan)


async def _drain(events: AsyncIterator[Any], tool: str, tools: tuple[str, ...] = ()) -> Optional[ToolResult]:
    """Run one harness run to its end; the last result of `tool` (or of any of `tools`), else None."""
    wanted = set(tools) | {tool}
    last: Optional[ToolResult] = None
    async for event in events:
        if isinstance(event, ToolExecutionEnd) and event.tool_name in wanted:
            last = event.result
    return last


async def _await(value: Any) -> None:
    await value


@dataclass
class Loop:
    """The driver's state: both agents, their plans, the rounds so far and what the round in hand collected."""

    plan: BuildPlan
    builder: AgentHarness
    target: str = TARGET_ALL
    agent_model: Optional[Model] = None
    allowance_usd: Optional[float] = None
    stall_rounds: int = 1
    subscribers: list = field(default_factory=list)
    on_event: Optional[Callable[[dict], Any]] = None
    max_turns: int = MAX_TURNS  # the Examiner's; the Builder has none (D135)
    eplan: Optional[ExaminerPlan] = None
    examiner: Optional[AgentHarness] = None
    rounds: list[RoundRecord] = field(default_factory=list)
    pending_findings: list[Finding] = field(default_factory=list)
    exhausted: list[bool] = field(default_factory=list)
    allowance: dict[str, Optional[float]] = field(default_factory=dict)
    build_result: Optional[ToolResult] = None
    examiner_result: Optional[ToolResult] = None
    sent: list[str] = field(default_factory=list)
    beat_spend: dict[str, float] = field(default_factory=dict)
    spent_allowance: dict[str, bool] = field(default_factory=dict)
    compactions_seen: dict[str, int] = field(default_factory=dict)
    turns_seen: dict[str, int] = field(default_factory=dict)
    round_started: float = 0.0
    round_saved_start: float = 0.0
    driver_built: list[int] = field(default_factory=list)  # rounds whose target the driver built itself
    builder_stop: dict = field(default_factory=dict)
    stall_told: int = 0
    started_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """A new Loop resumes the workdir's unfinished business: findings an earlier invocation
        left open join the pending queue, so a ceiling (or a crash) that ended that run strands no
        finding without a Builder beat to answer it. A fresh workdir has no findings file: no-op.

        The artifacts are fingerprinted here too, before any beat runs, and that is what round 1 is
        compared against. Without it a first round said no artifact changed however much it wrote,
        because there was no round before it to hold the hashes: `--iterate` over a workdir that
        already holds bodies and Intents is exactly the run where round 1 rewrites the most.
        """
        self.started_hashes = artifact_hashes(self.plan.workdir)
        seen = {finding.finding_id for finding in self.pending_findings}
        self.pending_findings = list(self.pending_findings) + [
            finding for finding in _open_findings(self.plan.workdir) if finding.finding_id not in seen]
        # Findings the Builder performed before the Examiner opened: no plan to close them on yet.
        # They flush into the store the moment _open_examiner runs, never dropped, never double-closed.
        self._unclosed: list[str] = []

    # --- the stream ------------------------------------------------------------

    def emit(self, event: Any) -> None:
        """A round or beat event to every subscriber, typed, and to `on_event` as a dict."""
        for subscriber in list(self.subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                asyncio.run(_await(result))
        if self.on_event is not None:
            body = as_dict_event(event)
            if body is not None:
                self.on_event(body)

    def spend(self) -> float:
        """What the workdir has spent so far, off the file budget.py writes."""
        return float(budget.load_totals(self.plan.workdir)["total"].get("usd") or 0.0)

    def cache_saved(self) -> float:
        """What the provider's cache has taken off the workdir's bill so far, off the same file."""
        return float(budget.load_totals(self.plan.workdir)["total"].get("cache_saved_usd") or 0.0)

    def compactions(self, agent: str) -> int:
        harness = self.builder if agent == "builder" else self.examiner
        return int(harness.context_stats.fallback_compactions) if harness is not None else 0

    def fills(self, agent: str) -> list[float]:
        """How full this agent's context was at the end of each turn it has taken (D131, D124).

        One entry per model turn, appended by the context manager as the turn ends, so the length
        is the turns the agent took and the largest entry is how close it came to the line."""
        harness = self.builder if agent == "builder" else self.examiner
        return list(harness.context_stats.fill_at_turn_end) if harness is not None else []

    # --- the allowance ------------------------------------------------------------

    def allowance_for(self, agent: str) -> Optional[float]:
        """`allowance_usd` when given; otherwise none in round 1, and from round 2 on the agent's own
        round-1 spend, where a spend of zero means no allowance at all."""
        if self.allowance_usd is not None:
            return float(self.allowance_usd)
        if not self.rounds:
            return None
        first = float((self.rounds[0].counts.get("spend") or {}).get(agent) or 0.0)
        return first if first > 0 else None

    def _watched(self, harness: AgentHarness, agent: str, events: AsyncIterator[Any],
                 tool: str, tools: tuple[str, ...] = ()) -> Optional[ToolResult]:
        """One model-driven run, with the allowance watched on every tool end: crossing it steers the
        agent once and marks the beat exhausted; the Examiner's plan sees what is left as it goes.

        `tools`, when given, widens what counts as the beat's result to every tool in it, which is
        how a Builder beat that repaired without calling `build` still has one (D135).
        """
        allowance = self.allowance.get(agent)
        before = self.spend()
        steered = False

        def watch(event: Any) -> None:
            nonlocal steered
            if not isinstance(event, ToolExecutionEnd) or allowance is None:
                return
            spent = self.spend() - before
            if agent == "examiner" and self.eplan is not None:
                self.eplan.allowance_remaining = allowance - spent
            if spent >= allowance and not steered:
                steered = True
                self.spent_allowance[agent] = True
                harness.steer(ALLOWANCE_STEER)

        unsubscribe = harness.subscribe(watch)
        # The spend ceiling is the hard stop of a session with no turn cap (D86, D135); the guard
        # cancels the run on the tool call that reaches it, and the stop rides on the round's exit.
        release, stop = ((lambda: None), {}) if agent != "builder" else builder_agent.ceiling_guard(harness, self.plan)
        try:
            return asyncio.run(_drain(events, tool, tools))
        finally:
            unsubscribe()
            release()
            if stop:
                self.builder_stop = dict(stop)

    def _beat_done(self, agent: str, n: int, before: float) -> None:
        spent = self.spend() - before
        self.beat_spend[agent] = spent
        allowance = self.allowance.get(agent)
        if allowance is not None and spent >= allowance:
            self.spent_allowance[agent] = True
        self.emit(BeatEnd(agent=agent, round=n, spend=spent))

    # --- the Builder's beat ------------------------------------------------------------

    def builder_beat(self, n: int) -> None:
        """The Builder holds the stream: the pending findings are acted on, then the target is built.

        Delivered, not dropped: the queue clears only with the beat's success below, so a Builder
        error leaves every finding queued instead of orphaning it open and unqueued. On the code
        path a suggested action that errors keeps its finding queued for the next beat; on the model
        path the action is unobservable, so a successful beat closes what it delivered.

        The code path calls the suggested verb through the same registry the model has, so
        `repair_intent` runs the narrowed intent stage and `repair_recompile` the narrowed
        compile_tools stage (builder/tools.py, `repair_verb_tools`) with the Examiner's hint."""
        if self.examiner is not None and self.examiner.is_running:
            raise RuntimeError("the Examiner is still running; one agent at a time (D128)")
        self.emit(BeatStart(agent="builder", round=n))
        before = self.spend()
        delivered = list(self.pending_findings)
        failed: set[str] = set()
        if self.agent_model is None:
            for finding in delivered:
                if finding.suggested != "none":
                    action = builder_agent.drive_tool(self.builder, finding.suggested,
                                                      finding_arguments(finding))
                    if action.is_error:
                        failed.add(finding.finding_id)
            self.build_result = builder_agent.drive_tool(self.builder, "build", {"target": self.target})
        else:
            if n == 1:
                events = self.builder.prompt(builder_message(self.target))
            else:
                self.builder.steer(f"round {n}: the Examiner's findings follow, one per message; act on each, "
                                   f"then build {self.target!r} again and read the rulings.")
                events = self.builder.continue_()
            # Every beat, including round 1: a resumed finding that never reaches the model would be
            # dequeued as delivered and later closed without ever being acted on. Queued follow-ups
            # fire when the run would otherwise stop, inside the same watched stream.
            for finding in delivered:
                self.builder.follow_up(finding_message(finding), {"finding": as_dict(finding)})
            self.build_result = self._watched(self.builder, "builder", events, "build", BUILD_TOOLS)
            if self.build_result is None or self._store_is_partial():
                # The model repaired and answered without building the target (build 13, round 1),
                # or built it and then repaired again on a follow-up finding (build 13, round 2): the
                # store then holds only what the last repair's stage ran, and the Examiner's derive
                # read an artifact that was not there (KeyError on the Constraints, both times). The
                # driver builds the target, as the code path does; every stage the repairs left
                # current comes from the cache (D153, D161).
                self.build_result = builder_agent.drive_tool(self.builder, "build", {"target": self.target})
                self.driver_built.append(n)
        result = self.build_result
        if self.plan.last is None or (result is not None and result.is_error):
            raise BuildError(result.content if result is not None
                             else f"the model never called build({self.target!r})")
        closed = [finding.finding_id for finding in delivered if finding.finding_id not in failed]
        if self.eplan is not None:
            if closed:
                self.eplan.close_findings(closed)
        else:
            self._unclosed.extend(closed)
        closed_ids = set(closed)
        self.pending_findings = [finding for finding in self.pending_findings
                                 if finding.finding_id not in closed_ids]
        self._beat_done("builder", n, before)

    def _store_is_partial(self) -> bool:
        """Whether the last `execute` was anything but the round's target in full: a narrowed stage
        (a repair verb) or another target leaves `plan.store` short of what the Examiner reads (D161)."""
        plan = self.plan
        return plan.last is None or bool(plan.last_narrowing) or plan.last_target != self.target

    # --- the Examiner's beat ------------------------------------------------------------

    def _open_examiner(self, n: int) -> None:
        """Round 1: the Examiner's plan over the Builder's artifacts; the Runner callables and the env_id
        are bound at every beat (`examiner_beat`), since a later Builder beat may leave a new Environment."""
        models = self.plan.models or {}
        self.eplan = ExaminerPlan(
            workdir=self.plan.workdir, inputs=_handover(self.plan.store),
            probe_model=models.get("loophole_probe"), judge_model=models.get("reference_judge"),
            probe_limit=self.plan.probe_limit, workers=self.plan.workers,
            anchor=pipeline.load_anchor(self.plan.workdir),
            on_event=_dict_sink(self.plan.on_event), round=n)
        self.examiner = examiner_agent.examiner_harness(
            self.eplan, self.agent_model, [*self.subscribers, self._collect_finding], max_turns=self.max_turns,
            session=_session(self.plan.workdir, EXAMINER_SESSION))
        self._flush_closed_findings()

    def _flush_closed_findings(self) -> None:
        """Findings performed before the Examiner opened, closed now that a plan holds the store.

        Round 1's Builder beat delivers (and dequeues) resumed findings while `eplan` is still None;
        dropping the ids there would orphan them open on disk, so they wait here and close exactly
        once, against the freshly loaded store."""
        if self._unclosed and self.eplan is not None:
            self.eplan.close_findings(self._unclosed)
            self._unclosed = []

    def _collect_finding(self, event: Any) -> None:
        if not isinstance(event, ToolExecutionEnd) or event.tool_name != "finding" or event.is_error:
            return
        body = (event.result.details or {}).get("finding")
        if body:
            finding = Finding.model_validate(body)
            self.pending_findings.append(finding)
            self.sent.append(finding.finding_id)

    def examiner_beat(self, n: int) -> None:
        """The Examiner holds the stream: it reads the Builder's artifacts as they stand and derives."""
        if self.builder.is_running:
            raise RuntimeError("the Builder is still running; one agent at a time (D128)")
        self.emit(BeatStart(agent="examiner", round=n))
        before = self.spend()
        if self.eplan is None:
            self._open_examiner(n)
        self.eplan.round = n
        self.eplan.refresh(_handover(self.plan.store))
        # The Runner runs in the Environment the Builder just left: a rebuilt db, schema or body would
        # otherwise stay behind a callable closed over the round-1 store (D120).
        self.eplan.run_probe, self.eplan.run_rerolls = _runners(self.plan)
        self.eplan.env_id = getattr(self.plan.store.get("environment"), "env_id", None)
        self.eplan.allowance_remaining = self.allowance.get("examiner")
        if self.agent_model is None:
            self.examiner_result = examiner_agent.drive_tool(self.examiner, "derive", {"target": EXAMINER_TARGET})
            if self.examiner_result.is_error:
                raise ExaminerError(self.examiner_result.content)
        else:
            if n == 1:
                events = self.examiner.prompt(examiner_message())
            else:
                self.examiner.steer(examiner_round_message(n, EXAMINER_TARGET))
                events = self.examiner.continue_()
            self.examiner_result = self._watched(self.examiner, "examiner", events, "derive")
            if self.examiner_result is None:
                raise ExaminerError(f"the model never called derive({EXAMINER_TARGET!r})")
            if self.examiner_result.is_error:
                raise ExaminerError(self.examiner_result.content)
        self._beat_done("examiner", n, before)

    # --- the round ------------------------------------------------------------

    def _land(self, ruling: GateResult) -> None:
        """A round_end ruling into gates.json unless an equal row is already there: the Builder's own
        fidelity ruling stays where its stage wrote it, so the file reads as the single pipeline wrote it."""
        path = self.eplan.ledger.path
        try:
            rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        except ValueError:
            rows = []
        if as_dict(ruling) in rows:
            return
        self.eplan.ledger.record("round_end", ruling)

    def driver_counts(self) -> dict:
        """What only the driver knows about the round in hand: when it ran, what each beat spent,
        how many turns each agent took and how full its context got, the compactions code had to
        make for it, and the findings it filed.

        These ride on the round's counts rather than on a record of their own, so a round that
        failed carries them too (`close_round` is given empty counts there) and rounds.json is the
        one file a report reads a round's clock and turns from.

        `artifacts` is the fingerprint of everything a model wrote, `artifact_hashes` the hash per
        artifact (which is how the next round has something to compare against) and
        `artifacts_changed` the artifacts this round rewrote. They sit beside the gate counts
        because a round that rewrote a tool body did something, whether or not a count has caught
        up with it yet.
        """
        spend = {agent: round(self.beat_spend.get(agent, 0.0), 6) for agent in AGENTS}
        spend["total"] = round(sum(spend.values()), 6)
        # What the cache took off this round's bill: the ledger's figure now less the figure at the
        # round's start, so a round's spend reads with the cache's effect beside it.
        spend["cache_saved"] = round(self.cache_saved() - self.round_saved_start, 6)
        turns: dict[str, int] = {}
        fill: dict[str, float] = {}
        for agent in AGENTS:
            beat = self.fills(agent)[self.turns_seen.get(agent, 0):]
            turns[agent] = len(beat)
            fill[agent] = max(beat) if beat else 0.0
        turns["total"] = sum(turns[agent] for agent in AGENTS)
        fingerprint, per, changed = self.artifacts_now()
        return {
            "started_at": self.round_started, "ended_at": time.time(), "spend": spend,
            "turns": turns, "context_fill": fill,
            "built_by_driver": self.plan.round in self.driver_built,
            "fallback_compactions": {
                agent: self.compactions(agent) - self.compactions_seen.get(agent, 0) for agent in AGENTS},
            "findings": list(self.sent),
            "artifacts": fingerprint, "artifact_hashes": per, "artifacts_changed": changed,
        }

    def counts(self) -> dict:
        """D126's counts off the gates, plus what only the driver knows (`driver_counts`)."""
        store = self.eplan.store if self.eplan is not None else {}
        counts = round_end.round_counts(
            store.get("task_status") or {}, store.get("verifiers") or [], store.get("probes") or {},
            store.get("history") or {}, store.get("refusals") or {}, store.get("task_runs") or {},
            store.get("replays") or {}, store.get("rerolls") or {}, store.get("canon_rules"),
            store.get("sigs") or [], record=self._land)
        counts.update(self.driver_counts())
        return counts

    def keep_gate_history(self, n: int) -> None:
        """gates.json as this round leaves it, kept per round in gates_by_round.json.

        gates.json holds the last ruling per stage, so the next round overwrites it; a repair can
        only be said to have turned a red gate green if the round before it is still readable."""
        ledger = self.eplan.ledger if self.eplan is not None else GateLedger(self.plan.workdir)
        ledger.snapshot(n)

    def ceiling_reached(self) -> bool:
        last = self.plan.last
        return (bool(last is not None and last.stopped) or bool(self.builder_stop)
                or bool(self.eplan is not None and self.eplan.ceiling_reached))

    # --- what the round in hand changed ------------------------------------------------------

    def artifacts_now(self) -> tuple[str, dict[str, str], list[str]]:
        """This workdir's artifact fingerprint, the hash of each artifact, and which of them changed
        since the round before, or since the driver started when this is the first round.

        A first round used to have nothing to compare with and so reported that nothing changed in
        it, whatever it rewrote. The driver takes the fingerprint once as it starts (`__post_init__`)
        and that is round 1's before, so `artifacts_changed` means the same thing in every round.
        """
        fingerprint, per = artifact_fingerprint(self.plan.workdir)
        before = (dict((self.rounds[-1].counts or {}).get("artifact_hashes") or {}) if self.rounds
                  else dict(self.started_hashes))
        changed = sorted(name for name, digest in per.items() if before and before.get(name) != digest)
        return fingerprint, per, changed

    def repairs_in(self, n: int) -> list[dict]:
        """The repair requests recorded under round `n`, oldest first.

        Every verb appends its request to `repairs/<verb>.jsonl` with the round it was made in
        (`builder.repair.record_request`), so this is what the mechanic asked for this round. A row
        that does not parse is skipped: a half-written line must not decide whether a round stalled.
        """
        folder = Path(self.plan.workdir) / "repairs"
        if not folder.is_dir():
            return []
        rows = []
        for path in sorted(folder.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and int(row.get("round") or 0) == n:
                    rows.append(row)
        return rows

    def repairs_made(self, n: int) -> list[dict]:
        """This round's repair requests, each with the artifact its verb owns and whether this call
        changed it, as the verb itself measured on its own target (`repair.change_of`).

        The round's fingerprint cannot answer that. A round that wrote six Intents again and left a
        seventh refused changed `intents` once, and every one of the seven requests read as having
        changed it; the verb hashed the one file it wrote, so each request answers for itself. A
        deciding verb owns no artifact and never changed one. A request an older build recorded
        carries no `changed`, and reads as False, since nothing measured it at the time.
        """
        out = []
        for row in self.repairs_in(n):
            verb = str(row.get("verb") or "?")
            out.append({"verb": verb, "target": str(row.get("target") or "?"),
                        "artifact": REPAIR_ARTIFACT.get(verb), "changed": bool(row.get("changed"))})
        return out

    def rulings_moved(self, n: int) -> bool:
        """True when the rulings this round leaves differ from the ones the round before it left.

        gates.json holds the last ruling per stage, which is round `n`'s: `keep_gate_history` has
        not snapshotted it yet when this is asked. The round before is read back out of
        gates_by_round.json, the file that snapshot writes. A first round has no round before it,
        so nothing can be said to have moved.
        """
        history = _read_json(Path(self.plan.workdir) / HISTORY_NAME, [])
        rows = [row for row in (history if isinstance(history, list) else [])
                if isinstance(row, dict) and int(row.get("round") or 0) == n - 1]
        if not rows:
            return False
        before = _stage_rulings(rows[-1].get("rulings"))
        return _stage_rulings(_read_json(Path(self.plan.workdir) / GATES_NAME, [])) != before

    # --- the soft stop (D126) ------------------------------------------------------------

    def moved_nothing(self, counts: dict) -> bool:
        """True when this round's counts match the last round's on every one of D126's gate counts."""
        if not self.rounds:
            return False
        last = self.rounds[-1].counts or {}
        return all(counts.get(key) == last.get(key) for key in round_end.GATE_COUNTS)

    def round_moved(self, n: int, counts: dict) -> bool:
        """Whether this round moved anything, which is what the stalled exit counts (D126).

        A gate count moving is one way. So is an artifact: a round in which the mechanic rewrote a
        tool body, wrote an Intent again or grew a table did something, and the Examiner has not
        derived over the new artifact yet, so no count could have moved for it. So is a repair
        request whose round left the gates ruling differently from the round before: the repair
        acted, even where the artifact it rewrote came out byte for byte the same. A first round
        has no round to be compared with and always moved.
        """
        if not self.rounds:
            return True
        if not self.moved_nothing(counts):
            return True
        if counts.get("artifacts_changed"):
            return True
        return bool(self.repairs_in(n)) and self.rulings_moved(n)

    def tell_the_builder_nothing_changed(self, n: int) -> None:
        """One follow-up, once per round, when the round moved nothing (D126, D135).

        Stalled is the soft stop: a model-driven Builder is told in one message that the round
        changed nothing, which verbs can change an artifact and that a build or a replay without a
        repair comes back from the cache, and that finishing is the way out when none of them
        answers what is left (`builder.agent.nothing_changed_message`). The message rides the
        steering queue, which drains before the next assistant turn, so the follow-up costs exactly
        one more turn rather than a wasted one first; if the model stops again the round ends, and
        the exit is the stalled exit `exit_for` was already going to give. The code driver gets none
        of this: it has no model to tell.

        The message also says what is still pending: the findings no beat has acted on with the
        verb each suggests, and the repair requests this round did make with whether each one's
        artifact changed. A round that stalled with a finding open stalled for a reason, and the
        model cannot see either list from inside its own transcript.
        """
        if self.agent_model is None or self.stall_told == n:
            return
        self.stall_told = n
        before = self.spend()
        self.builder.steer(builder_agent.nothing_changed_message(
            self.plan, findings=self.pending_findings, repairs=self.repairs_made(n)))
        result = self._watched(self.builder, "builder", self.builder.continue_(), "build", BUILD_TOOLS)
        if result is not None:
            self.build_result = result
        self.beat_spend["builder"] = self.beat_spend.get("builder", 0.0) + (self.spend() - before)

    def round(self, n: int) -> RoundRecord:
        """One round: the Builder's beat, the Examiner's beat, the counts, the exit, rounds.json."""
        self.round_started = time.time()
        self.round_saved_start = self.cache_saved()
        self.plan.round = n  # the round a repair request records itself under (D126)
        self.emit(RoundStart(round=n))
        self.sent, self.beat_spend, self.spent_allowance = [], {}, {}
        self.compactions_seen = {agent: self.compactions(agent) for agent in AGENTS}
        self.turns_seen = {agent: len(self.fills(agent)) for agent in AGENTS}
        self.allowance = {agent: self.allowance_for(agent) for agent in AGENTS}
        self.builder_beat(n)
        self.examiner_beat(n)
        counts = self.counts()
        if not self.round_moved(n, counts) and not self.ceiling_reached():
            self.tell_the_builder_nothing_changed(n)
            counts = self.counts()
        record = self.close_round(n, counts)
        self.emit(RoundEnd(round=n, counts=record.counts, exit=record.exit))
        return record

    def close_round(self, n: int, counts: dict) -> RoundRecord:
        """The round's record: whether an allowance was spent goes on the exhausted list, the exit is
        `exit_for` over every round so far, and rounds.json is rewritten with the record appended.
        Findings filed but never delivered ride on the record, and a round that would exit `done`
        or `stalled` while any are pending does not exit at all: the findings owe the Builder a beat,
        so the exit is cleared and the loop runs another round. Only the ceiling (no money left)
        ends a round with findings still open, and then the note says so.

        Whether the round moved (`round_moved`: a gate count, an artifact, or a ruling under one of
        its own repairs) rides on the counts, and the tail from the last round that moved is what
        `exit_for` is given, so `stall_rounds` counts only the rounds that moved nothing
        (`_since_last_move`)."""
        self.exhausted.append(any(self.spent_allowance.values()))
        # The driver's own numbers last and freshest: a round that failed comes here with no counts
        # at all, and its clock, spend and turns are as true as a round that finished.
        record = RoundRecord(round=n, counts={**counts, **self.driver_counts()})
        record.counts["moved"] = self.round_moved(n, record.counts)
        record.counts["repairs"] = self.repairs_made(n)
        record.exit = round_end.exit_for(_since_last_move(self.rounds + [record]), self.stall_rounds,
                                         ceiling_reached=self.ceiling_reached(), exhausted=self.exhausted)
        if self.pending_findings:
            record.pending_findings = list(self.pending_findings)
            if record.exit == "ceiling":
                record.exit_note = (f"{len(self.pending_findings)} finding(s) still need the Builder; "
                                    "the ceiling ended the run first")
            elif record.exit is not None:
                record.exit = None
                record.exit_note = (f"{len(self.pending_findings)} finding(s) owe the Builder a beat; "
                                    "the round continues")
        self.rounds.append(record)
        write_rounds(self.plan.workdir, self.rounds)
        self.keep_gate_history(n)
        return record

    def result(self) -> dict:
        """run_builder's dict plus the rounds, the exit, the trusted Tasks, the refusals and the Examiner's say."""
        last = self.plan.last
        if last is None:
            # No stage ever built: the rounds carry the whole story (a first beat that failed
            # records its stalled round). Same keys, empty build fields, never a traceback.
            out: dict = {"status": "failed", "workdir": str(self.plan.workdir), "env_id": None,
                           "failed_stage": None, "stopped": False, "gates": [], "tasks": [],
                           "target": self.target, "rulings": [],
                           "tool_result": _tool_result(self.build_result)}
        else:
            out = build_module.result_of(self.plan.workdir, last, last.artifacts.get("environment"))
            out["target"] = self.target
            out["rulings"] = list(last.rulings)
            out["tool_result"] = _tool_result(self.build_result)
        out["rounds"] = [as_dict(r) for r in self.rounds]
        out["exit"] = self.rounds[-1].exit if self.rounds else None
        out["failed"] = bool(self.rounds and self.rounds[-1].failed)
        counts = self.rounds[-1].counts if self.rounds else {}
        out["trusted"] = list(counts.get("trusted_ids") or [])
        out["refused"] = dict(counts.get("refused") or {})
        out["examiner"] = {"rulings": list(self.eplan.last_rulings) if self.eplan is not None else [],
                           "tool_result": _tool_result(self.examiner_result)}
        return out


def _tool_result(result: Optional[ToolResult]) -> Optional[dict]:
    return {"content": result.content, "is_error": result.is_error} if result is not None else None


def _record_judge_models(plan: BuildPlan) -> None:
    """Write which model each side of the judging ran on into report_config.json (D160).

    Only when a judge model was named: a build that judges with its own model writes nothing here,
    so its records are what they were before this, byte for byte.
    """
    if plan.judge_model is None and plan.second_judge_model is None:
        return
    path = plan.workdir / "report_config.json"
    config = dict(_read_config(path))
    config["judge_models"] = plan.judge_model_ids()
    write_json(path, config)


def _read_config(path: Path) -> dict:
    body = read_json(path, {}) or {}
    return body if isinstance(body, dict) else {}


def run_rounds(workdir: Any, model: Any = None, *, agent_model: Optional[Model] = None, files: Optional[list] = None,
               judge_model: Any = None, second_judge_model: Any = None,
               iterate: bool = False, ceiling_usd: Optional[float] = None, allowance_usd: Optional[float] = None,
               stall_rounds: int = 1, target: str = TARGET_ALL, domain: str = "domain", max_attempts: int = 3,
               memory_dir: Any = None, grow: Optional[dict] = None, grow_seed: int = 0,
               probe_limit: Optional[int] = None, rerolls: int = DEFAULT_REROLLS, search: Any = None,
               workers: int = 1, on_event: Optional[Any] = None, subscribers: Iterable[Callable[[Any], Any]] = (),
               max_turns: int = MAX_TURNS) -> dict:
    """Rounds over one workdir until an exit: what `kullback build` runs and the screen's /build calls.

    `model` is the Builder's model for the stages that call one, and through the plan's wrapped models
    the Examiner's too; `agent_model` drives both sessions when given, and with None code issues the
    tool calls (build for the Builder, derive for the Examiner). `judge_model` is the model the
    build's judge runs on and `model` when it is None; `second_judge_model` is the other side of a
    two-judge question (D160). The dict is run_builder's plus the rounds, the exit, the trusted
    Tasks, the refusals and the Examiner's rulings.
    """
    plan = BuildPlan(workdir=Path(workdir), iterate=iterate, model=model, judge_model=judge_model,
                     second_judge_model=second_judge_model, files=list(files or []),
                     ceiling_usd=ceiling_usd, domain=domain, max_attempts=max_attempts, memory_dir=memory_dir,
                     on_event=on_event, grow=grow, grow_seed=grow_seed, probe_limit=probe_limit, rerolls=rerolls,
                     search=search, workers=workers)
    _record_judge_models(plan)
    # The feed subscribes like anything else. Attaching here rather than inside the two harnesses
    # means both agents' streams reach it through the one seam the harness already offers: the
    # stages in either arm, and the messages and tool calls in the arm where a model drives.
    shared = [*subscribers, lambda event: feed.from_event(plan.workdir, event)]
    loop = Loop(plan=plan, builder=_builder_harness(plan, agent_model, shared), target=target,
                agent_model=agent_model, allowance_usd=allowance_usd, stall_rounds=stall_rounds,
                subscribers=shared, on_event=on_event, max_turns=max_turns)
    n = 0
    while True:
        n += 1
        try:
            record = loop.round(n)
        except (ExaminerError, BuildError) as exc:
            # A broken agent contract fails the round; it never closes one on stale state and
            # never dies without a record. The round is stalled with the reason on it, so rounds.json
            # tells the whole story, and findings still queued ride on the record (close_round
            # persists them) instead of dying with the process.
            record = loop.close_round(n, {})
            # A ceiling that ended the build left the other agent nothing to work on, and D86 says
            # to stop and report as is; that is the ceiling exit, not a broken agent.
            ceiling = loop.ceiling_reached()
            record.exit = "ceiling" if ceiling else "stalled"  # a broken agent is not fixed by another beat
            record.failed = not ceiling
            record.exit_note = (f"examiner failed: {exc}" if isinstance(exc, ExaminerError)
                                else f"builder failed: {exc}")
            if record.pending_findings:
                record.exit_note += (f"; {len(record.pending_findings)} finding(s) "
                                     "never got a Builder beat")
            write_rounds(plan.workdir, loop.rounds)
            loop.emit(RoundEnd(round=n, counts=record.counts, exit=record.exit))
            break
        if record.exit is not None:
            break
    return loop.result()
