"""Replay a recorded Trace through the rebuilt Environment: the Reference Run, scored call by call.

Design section 6 ("Gate A, oracle replay"): replaying the Reference's own calls has to reach its
End state, every write matched after canonicalization and no semantic read mismatch. The Verifier
reads Runs off disk (D91) and nothing turned a Trace into one, so no Task ever had a confirmed
Reference and the first live build derived zero Verifiers. Here the Trace's assistant turns are the
model, its user turns are the user, and every tool call it made goes through the same Router a
Candidate gets. The loop writes the Run the way it writes any other; this module only scores each
routed result against the recorded one and says whether the replay confirms the Reference (D108).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.runner import loop
from harness.shared.canon import canonicalize
from harness.shared.provider import Model, ModelConfig, ModelReply, ToolCallRequest
from harness.shared.records import ToolCall, Trace, Turn, as_dict, plain

RECORDED = "recorded"
SPOKEN_ROLES = ("assistant", "user")
# How one routed call compares with the recording. The first three agree on effect; the rest do not.
SAME, COSMETIC, BOTH_REFUSED = "same", "cosmetic", "both_refused"
DIFFERS, OURS_REFUSED, THEIRS_REFUSED, UNRECORDED = "differs", "ours_refused", "theirs_refused", "unrecorded"
AGREES = frozenset({SAME, COSMETIC, BOTH_REFUSED})


class _Script:
    """One cursor over the Trace's turns, shared by the scripted model and the scripted user."""

    def __init__(self, trace: Trace):
        self.turns = list(trace.turns)
        self.calls = {call.id: call for call in trace.tool_calls if call.id}
        self.pos = 0

    def take(self, role: str) -> Optional[Turn]:
        """The next turn of this role, if it is the next spoken turn; tool turns are skipped."""
        for index in range(self.pos, len(self.turns)):
            turn = self.turns[index]
            if turn.role not in SPOKEN_ROLES:
                continue
            if turn.role == role:
                self.pos = index + 1
                return turn
            return None
        return None

    def remaining(self, role: str) -> bool:
        return any(turn.role == role for turn in self.turns[self.pos:])

    def requests(self, turn: Turn) -> list[ToolCall]:
        return [self.calls[i] for i in turn.tool_call_ids if i in self.calls]


class TraceModel(Model):
    """The recorded assistant, one turn per query: its text and the tool calls it made."""

    def __init__(self, script: _Script, name: str = RECORDED):
        self.script = script
        self.name = name
        self.expected: deque[ToolCall] = deque()
        self.gaps = 0  # a user turn stood where an assistant turn was due, or the reverse

    def query(self, messages: list[dict], tools: Optional[list[dict]] = None,
              config: Optional[ModelConfig] = None) -> ModelReply:
        turn = self.script.take("assistant")
        if turn is None:
            if self.script.remaining("assistant"):
                self.gaps += 1
            return ModelReply(content="", model=self.name)
        requests = self.script.requests(turn)
        self.expected.extend(requests)
        return ModelReply(content=turn.content or "", model=self.name, tool_calls=[
            ToolCallRequest(id=call.id, name=call.name, arguments=dict(call.args or {})) for call in requests])


class TraceUser:
    """The recorded user, one turn per reply; a tool it called itself is routed and written under its name (D71)."""

    def __init__(self, script: _Script):
        self.script = script
        self.events: list = []
        self.state: Any = None
        self.router: Any = None
        self.gaps = 0

    @property
    def done(self) -> bool:
        return not self.script.remaining("user")

    def reply(self, transcript: list) -> str:
        turn = self.script.take("user")
        if turn is None:
            self.gaps += 1
            return ""
        for call in self.script.requests(turn):
            self._own_call(call)
        return turn.content or ""

    def _own_call(self, call: ToolCall) -> None:
        if self.state is None or self.router is None:
            return
        args = dict(call.args or {})
        loop.emit(self.state, "tool_call", {"id": call.id, "name": call.name, "args": args, "requestor": "user"})
        outcome = self.router.route(call.name, args, recorded=call)
        payload: dict = {"id": call.id, "name": call.name, "result": outcome.result, "requestor": "user"}
        if outcome.error is not None:
            payload["error"] = as_dict(outcome.error)
        loop.emit(self.state, "tool_result", payload, route=outcome.route, assisted=outcome.assisted)
        counts = self.state.run.route_counts
        counts[outcome.route] = counts.get(outcome.route, 0) + 1


class ScoredRouter:
    """The Router with each answer compared against the call the Trace recorded for it."""

    def __init__(self, router: Any, expected: deque, write_tools: Iterable[str] = (), canon_rules: Any = None):
        self.inner = router
        self.expected = expected
        self.write_tools = set(write_tools)
        self.canon_rules = canon_rules
        self.checks: list[dict] = []

    def __getattr__(self, name: str) -> Any:  # state_hash, world, start_world, state: the loop's reads
        return getattr(self.inner, name)

    def route(self, name: str, args: Optional[dict] = None, recorded: Optional[ToolCall] = None) -> Any:
        outcome = self.inner.route(name, args)
        recorded = recorded if recorded is not None else self._take(name)
        verdict = UNRECORDED if recorded is None else compare_call(
            recorded, outcome.result, outcome.error, self.canon_rules)
        self.checks.append({
            "tool": name, "kind": "write" if name in self.write_tools else "read", "verdict": verdict,
            "route": outcome.route, "call_id": recorded.id if recorded is not None else None,
            "ours": _preview(outcome.error if outcome.error is not None else outcome.result),
            "recorded": _preview(recorded.error if recorded is not None and recorded.error is not None
                                 else (recorded.result if recorded is not None else None))})
        return outcome

    def _take(self, name: str) -> Optional[ToolCall]:
        for index, call in enumerate(self.expected):
            if call.name == name:
                del self.expected[index]
                return call
        return None


def compare_call(recorded: ToolCall, result: Any, error: Any, rules: Any = None) -> str:
    """One routed answer against the recorded one: same, cosmetic, or one of the ways they part."""
    ours_failed, theirs_failed = error is not None, recorded.error is not None
    if ours_failed and theirs_failed:
        return BOTH_REFUSED
    if ours_failed:
        return OURS_REFUSED
    if theirs_failed:
        return THEIRS_REFUSED
    ours, theirs = _norm(result), _norm(recorded.result)
    if _dumps(ours) == _dumps(theirs):
        return SAME
    if canonicalize(ours, rules) == canonicalize(theirs, rules):
        return COSMETIC
    return DIFFERS


def _norm(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value.strip()
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _preview(value: Any, limit: int = 160) -> str:
    text = _dumps(plain(value))
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class Replay:
    """One Trace replayed: where its Run went, how each call compared, and whether it confirms the Reference."""
    run_id: str
    trace_id: str
    task_id: str
    path: str
    confirmed: bool
    termination_reason: Optional[str]
    counts: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    crashed: Optional[str] = None

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "trace_id": self.trace_id, "task_id": self.task_id, "path": self.path,
                "confirmed": self.confirmed, "termination_reason": self.termination_reason,
                "counts": dict(self.counts), "reasons": list(self.reasons), "checks": list(self.checks),
                "crashed": self.crashed}


def replay_trace(trace: Trace, router: Any, *, workdir: Any, task_id: str, env_id: Optional[str] = None,
                 write_tools: Iterable[str] = (), canon_rules: Any = None, run_id: Optional[str] = None) -> Replay:
    """Drive the loop with the Trace's own turns over `router`; the Run lands under `workdir`."""
    script = _Script(trace)
    model, user = TraceModel(script), TraceUser(script)
    scored = ScoredRouter(router, model.expected, write_tools, canon_rules)
    run_id = run_id or f"replay-{trace.trace_id}"
    state = loop.new_run_state(run_id, workdir=workdir, env_id=env_id, task_id=task_id,
                               trace_id=trace.trace_id, model=RECORDED, user=user,
                               max_turns=len(trace.turns) + 2, system_prompt=trace.system_prompt)
    user.state, user.router = state, scored
    crashed: Optional[str] = None
    try:
        loop.run(state, model, router=scored)
    except Exception as exc:  # the loop wrote the error and the stop before raising
        crashed = f"{type(exc).__name__}: {exc}"
    return _score(trace, state, scored, model, user, crashed, write_tools)


def _score(trace: Trace, state: Any, scored: ScoredRouter, model: TraceModel, user: TraceUser,
           crashed: Optional[str], write_tools: Iterable[str]) -> Replay:
    writes = [c for c in scored.checks if c["kind"] == "write"]
    reads = [c for c in scored.checks if c["kind"] == "read"]
    writes_off = [c for c in writes if c["verdict"] not in AGREES]
    reads_off = [c for c in reads if c["verdict"] not in AGREES]
    unmade = [call.name for call in model.expected]
    counts = {
        "calls": len(scored.checks), "writes": len(writes), "writes_matched": len(writes) - len(writes_off),
        "reads": len(reads), "reads_same": sum(c["verdict"] == SAME for c in reads),
        "reads_cosmetic": sum(c["verdict"] == COSMETIC for c in reads),
        "reads_both_refused": sum(c["verdict"] == BOTH_REFUSED for c in reads),
        "reads_semantic": len(reads_off), "unmade": len(unmade), "gaps": model.gaps + user.gaps,
        "routes": dict(state.run.route_counts),
    }
    reasons = [f"{c['tool']} write: {c['verdict']}" for c in writes_off]
    reasons += [f"{c['tool']} read: {c['verdict']}" for c in reads_off]
    reasons += [f"{name} was recorded and never called" for name in unmade]
    if counts["gaps"]:
        reasons.append(f"{counts['gaps']} turn(s) out of order")
    if crashed:
        reasons.append(f"replay crashed: {crashed}")
    if int(state.run.route_counts.get("llm") or 0):
        reasons.append("a call was answered by the stand-in")
    return Replay(run_id=state.run.run_id, trace_id=trace.trace_id, task_id=state.run.task_id or "",
                  path=str(state.path) if state.path else "", confirmed=not reasons,
                  termination_reason=state.run.termination_reason, counts=counts, reasons=reasons,
                  checks=scored.checks, crashed=crashed)


def summarize(replays: dict[str, dict[str, dict]]) -> dict:
    """The stage's numbers over every replay: Traces, confirmed, per Task, writes and reads."""
    rows = [r for per_task in replays.values() for r in per_task.values()]
    tasks_confirmed = sum(any(r["confirmed"] for r in per_task.values()) for per_task in replays.values())
    total = lambda key: sum(int((r.get("counts") or {}).get(key) or 0) for r in rows)  # noqa: E731
    return {"traces": len(rows), "confirmed": sum(bool(r["confirmed"]) for r in rows),
            "tasks": len(replays), "tasks_confirmed": tasks_confirmed,
            "writes": total("writes"), "writes_matched": total("writes_matched"),
            "reads": total("reads"), "reads_semantic": total("reads_semantic"),
            "reads_cosmetic": total("reads_cosmetic"), "unmade": total("unmade")}


def unconfirmed_reason(per_task: dict[str, dict]) -> str:
    """Why a Task has no Reference: the most common first reason across its replays."""
    firsts = [r["reasons"][0] for r in per_task.values() if r.get("reasons")]
    if not firsts:
        return "no Trace of the Task was replayed"
    return max(sorted(set(firsts)), key=firsts.count)


__all__ = ["Replay", "ScoredRouter", "TraceModel", "TraceUser", "compare_call", "replay_trace",
           "summarize", "unconfirmed_reason"]
_ = Path  # Path is part of the public annotations via loop.new_run_state callers
