"""The round driver (D126, D128): one round is a Builder beat then an Examiner beat on one stream.

The two agents never run at once. The driver holds both harnesses, one shared subscriber list and
the Builder's plan; a round starts with `RoundStart`, hands the stream to the Builder (`BeatStart`
then `BeatEnd`), hands it to the Examiner the same way, and ends with `RoundEnd` carrying the counts
`gates.round_end` computed off the rulings, never off a model. The Examiner's findings are collected
off its `ToolExecutionEnd` events and delivered to the Builder at its next beat as follow-ups with
the record in `details` (D123); the code driver acts on them itself by calling the Builder tool the
finding names. Three exits: `done` when D126's state holds, `stalled` after `stall_rounds` rounds
that moved no gate count, `ceiling` when the Builder stopped at the spend ceiling, the Examiner
reported the ceiling, or an agent spent its allowance two rounds in a row. The allowance is per
agent and per round: `allowance_usd` when given, otherwise the agent's own round-1 spend from
round 2 on. A model-driven agent that crosses it is steered once to finish with what it has.

This module is the top of the layering: it imports both applications, and the Builder's artifacts
reach the Examiner through `examiner.stage.DERIVE_INPUTS`, never bodies, the db, the schema or the
Environment (D123).
"""

from __future__ import annotations

import asyncio
import inspect
import json
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
from kullback.builder.agent import builder_message
from kullback.builder.build import DEFAULT_REROLLS, TARGET_ALL, BuildError, BuildPlan
from kullback.examiner import agent as examiner_agent
from kullback.examiner.agent import ExaminerError, examiner_message
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.stage import DERIVE_INPUTS
from kullback.gates import round_end
from kullback.runner import budget
from kullback.runner.records import Finding, GateResult, RoundRecord, as_dict

ROUNDS_NAME = "rounds.json"
AGENTS = ("builder", "examiner")
MAX_TURNS = 8
ALLOWANCE_STEER = "Your allowance for this round is spent: finish with what you have (D123)"
EXAMINER_TARGET = "all"
BUILDER_SESSION = Path("builder") / "session.jsonl"
EXAMINER_SESSION = Path("examiner") / "session.jsonl"


def finding_message(finding: Finding) -> str:
    """A finding as the Builder reads it; the record itself rides in the message's details."""
    text = f"Finding {finding.finding_id} ({finding.kind}): {finding.text}"
    if finding.task_id:
        text += f" Task {finding.task_id}."
    if finding.tool:
        text += f" Tool {finding.tool}."
    if finding.suggested != "none":
        text += f" Suggested: {finding.suggested}."
    return text


def finding_arguments(finding: Finding) -> dict:
    """The arguments of the Builder tool a finding suggests: the tool by name, or the Task."""
    if finding.suggested == "compile_tool":
        return {"name": finding.tool or ""}
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


def write_rounds(workdir: Any, rounds: Iterable[RoundRecord]) -> Path:
    """rounds.json: one RoundRecord per round, the exit on the last."""
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


def _builder_harness(plan: BuildPlan, agent_model: Optional[Model], subscribers: list, max_turns: int) -> AgentHarness:
    """The Builder's harness with its own session file (D128)."""
    return builder_agent.build_harness(plan, agent_model, subscribers, max_turns=max_turns,
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


async def _drain(events: AsyncIterator[Any], tool: str) -> Optional[ToolResult]:
    """Run one harness run to its end; the last result of `tool`, or None when it was never called."""
    last: Optional[ToolResult] = None
    async for event in events:
        if isinstance(event, ToolExecutionEnd) and event.tool_name == tool:
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
    max_turns: int = MAX_TURNS
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

    def compactions(self, agent: str) -> int:
        harness = self.builder if agent == "builder" else self.examiner
        return int(harness.context_stats.fallback_compactions) if harness is not None else 0

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

    def _watched(self, harness: AgentHarness, agent: str, events: AsyncIterator[Any], tool: str) -> Optional[ToolResult]:
        """One model-driven run, with the allowance watched on every tool end: crossing it steers the
        agent once and marks the beat exhausted; the Examiner's plan sees what is left as it goes."""
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
        try:
            return asyncio.run(_drain(events, tool))
        finally:
            unsubscribe()

    def _beat_done(self, agent: str, n: int, before: float) -> None:
        spent = self.spend() - before
        self.beat_spend[agent] = spent
        allowance = self.allowance.get(agent)
        if allowance is not None and spent >= allowance:
            self.spent_allowance[agent] = True
        self.emit(BeatEnd(agent=agent, round=n, spend=spent))

    # --- the Builder's beat ------------------------------------------------------------

    def builder_beat(self, n: int) -> None:
        """The Builder holds the stream: the pending findings are acted on, then the target is built."""
        if self.examiner is not None and self.examiner.is_running:
            raise RuntimeError("the Examiner is still running; one agent at a time (D128)")
        self.emit(BeatStart(agent="builder", round=n))
        before = self.spend()
        delivered, self.pending_findings = list(self.pending_findings), []
        if self.agent_model is None:
            for finding in delivered:
                if finding.suggested != "none":
                    builder_agent.drive_tool(self.builder, finding.suggested, finding_arguments(finding))
            self.build_result = builder_agent.drive_tool(self.builder, "build", {"target": self.target})
        else:
            if n == 1:
                events = self.builder.prompt(builder_message(self.target))
            else:
                self.builder.steer(f"round {n}: the Examiner's findings follow, one per message; act on each, "
                                   f"then build {self.target!r} again and read the rulings.")
                for finding in delivered:
                    self.builder.follow_up(finding_message(finding), {"finding": as_dict(finding)})
                events = self.builder.continue_()
            self.build_result = self._watched(self.builder, "builder", events, "build")
        result = self.build_result
        if self.plan.last is None or (result is not None and result.is_error):
            raise BuildError(result.content if result is not None
                             else f"the model never called build({self.target!r})")
        if delivered and self.eplan is not None:
            self.eplan.close_findings([finding.finding_id for finding in delivered])
        self._beat_done("builder", n, before)

    # --- the Examiner's beat ------------------------------------------------------------

    def _open_examiner(self, n: int) -> None:
        """Round 1: the Examiner's plan over the Builder's artifacts; the Runner callables and the env_id
        are bound at every beat (`examiner_beat`), since a later Builder beat may leave a new Environment."""
        models = self.plan.models or {}
        self.eplan = ExaminerPlan(
            workdir=self.plan.workdir, inputs=_handover(self.plan.store),
            probe_model=models.get("loophole_probe"), judge_model=models.get("reference_judge"),
            probe_limit=self.plan.probe_limit, anchor=pipeline.load_anchor(self.plan.workdir),
            on_event=_dict_sink(self.plan.on_event), round=n)
        self.examiner = examiner_agent.examiner_harness(
            self.eplan, self.agent_model, [*self.subscribers, self._collect_finding], max_turns=self.max_turns,
            session=_session(self.plan.workdir, EXAMINER_SESSION))

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
                self.examiner.steer(f"round {n}: read the rulings and act")
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

    def counts(self) -> dict:
        """D126's counts off the gates, plus what only the driver knows: compactions, spend, findings."""
        store = self.eplan.store if self.eplan is not None else {}
        counts = round_end.round_counts(
            store.get("task_status") or {}, store.get("verifiers") or [], store.get("probes") or {},
            store.get("history") or {}, store.get("refusals") or {}, store.get("task_runs") or {},
            store.get("replays") or {}, store.get("rerolls") or {}, store.get("canon_rules"),
            store.get("sigs") or [], record=self._land)
        counts["fallback_compactions"] = {
            agent: self.compactions(agent) - self.compactions_seen.get(agent, 0) for agent in AGENTS}
        spend = {agent: round(self.beat_spend.get(agent, 0.0), 6) for agent in AGENTS}
        spend["total"] = round(sum(spend.values()), 6)
        counts["spend"] = spend
        counts["findings"] = list(self.sent)
        return counts

    def ceiling_reached(self) -> bool:
        last = self.plan.last
        return bool(last is not None and last.stopped) or bool(self.eplan is not None and self.eplan.ceiling_reached)

    def round(self, n: int) -> RoundRecord:
        """One round: the Builder's beat, the Examiner's beat, the counts, the exit, rounds.json."""
        self.emit(RoundStart(round=n))
        self.sent, self.beat_spend, self.spent_allowance = [], {}, {}
        self.compactions_seen = {agent: self.compactions(agent) for agent in AGENTS}
        self.allowance = {agent: self.allowance_for(agent) for agent in AGENTS}
        self.builder_beat(n)
        self.examiner_beat(n)
        record = self.close_round(n, self.counts())
        self.emit(RoundEnd(round=n, counts=record.counts, exit=record.exit))
        return record

    def close_round(self, n: int, counts: dict) -> RoundRecord:
        """The round's record: whether an allowance was spent goes on the exhausted list, the exit is
        `exit_for` over every round so far, and rounds.json is rewritten with the record appended.
        Findings filed but never delivered ride on the record, and `done` is downgraded to `stalled`
        while any are pending: a terminal round never reports the work finished with required Builder
        follow-ups still open."""
        self.exhausted.append(any(self.spent_allowance.values()))
        record = RoundRecord(round=n, counts=counts)
        record.exit = round_end.exit_for(self.rounds + [record], self.stall_rounds,
                                         ceiling_reached=self.ceiling_reached(), exhausted=self.exhausted)
        if self.pending_findings:
            record.pending_findings = list(self.pending_findings)
            if record.exit == "done":
                record.exit = "stalled"
                record.exit_note = (f"{len(self.pending_findings)} finding(s) still need the Builder; "
                                    "done waits for the next round")
        self.rounds.append(record)
        write_rounds(self.plan.workdir, self.rounds)
        return record

    def result(self) -> dict:
        """run_builder's dict plus the rounds, the exit, the trusted Tasks, the refusals and the Examiner's say."""
        last = self.plan.last
        out = build_module.result_of(self.plan.workdir, last, last.artifacts.get("environment"))
        out["target"] = self.target
        out["rulings"] = list(last.rulings)
        out["tool_result"] = _tool_result(self.build_result)
        out["rounds"] = [as_dict(r) for r in self.rounds]
        out["exit"] = self.rounds[-1].exit if self.rounds else None
        counts = self.rounds[-1].counts if self.rounds else {}
        out["trusted"] = list(counts.get("trusted_ids") or [])
        out["refused"] = dict(counts.get("refused") or {})
        out["examiner"] = {"rulings": list(self.eplan.last_rulings) if self.eplan is not None else [],
                           "tool_result": _tool_result(self.examiner_result)}
        return out


def _tool_result(result: Optional[ToolResult]) -> Optional[dict]:
    return {"content": result.content, "is_error": result.is_error} if result is not None else None


def run_rounds(workdir: Any, model: Any = None, *, agent_model: Optional[Model] = None, files: Optional[list] = None,
               iterate: bool = False, ceiling_usd: Optional[float] = None, allowance_usd: Optional[float] = None,
               stall_rounds: int = 1, target: str = TARGET_ALL, domain: str = "domain", max_attempts: int = 3,
               memory_dir: Any = None, grow: Optional[dict] = None, grow_seed: int = 0,
               probe_limit: Optional[int] = None, rerolls: int = DEFAULT_REROLLS, search: Any = None,
               workers: int = 1, on_event: Optional[Any] = None, subscribers: Iterable[Callable[[Any], Any]] = (),
               max_turns: int = MAX_TURNS) -> dict:
    """Rounds over one workdir until an exit: what `kullback build` runs and the screen's /build calls.

    `model` is the Builder's model for the stages that call one, and through the plan's wrapped models
    the Examiner's too; `agent_model` drives both sessions when given, and with None code issues the
    tool calls (build for the Builder, derive for the Examiner). The dict is run_builder's plus the
    rounds, the exit, the trusted Tasks, the refusals and the Examiner's rulings.
    """
    plan = BuildPlan(workdir=Path(workdir), iterate=iterate, model=model, files=list(files or []),
                     ceiling_usd=ceiling_usd, domain=domain, max_attempts=max_attempts, memory_dir=memory_dir,
                     on_event=on_event, grow=grow, grow_seed=grow_seed, probe_limit=probe_limit, rerolls=rerolls,
                     search=search, workers=workers)
    shared = list(subscribers)
    loop = Loop(plan=plan, builder=_builder_harness(plan, agent_model, shared, max_turns), target=target,
                agent_model=agent_model, allowance_usd=allowance_usd, stall_rounds=stall_rounds,
                subscribers=shared, on_event=on_event, max_turns=max_turns)
    n = 0
    while True:
        n += 1
        try:
            record = loop.round(n)
        except ExaminerError as exc:
            # A broken examiner contract fails the round; it never closes one on stale state and
            # never dies without a record. The round is stalled with the reason on it, so rounds.json
            # tells the whole story and the next session knows the Examiner needs fixing, not the Tasks.
            record = loop.close_round(n, {})
            record.exit = "stalled"
            record.exit_note = f"examiner failed: {exc}"
            write_rounds(plan.workdir, loop.rounds)
            loop.emit(RoundEnd(round=n, counts=record.counts, exit=record.exit))
            break
        if record.exit is not None:
            break
    return loop.result()
