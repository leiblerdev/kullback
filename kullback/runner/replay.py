"""Replay a recorded Trace through the rebuilt Environment: the Reference Run, scored call by call.

Design section 6 ("Gate A, oracle replay"): replaying the Reference's own calls has to reach its
End state, every write matched after canonicalization and no semantic read mismatch. The Verifier
reads Runs off disk (D91) and nothing turned a Trace into one, so no Task ever had a confirmed
Reference and the first live build derived zero Verifiers. Here the Trace's assistant turns are the
model, its user turns are the user, and every tool call it made goes through the same Router a
Candidate gets. The loop writes the Run the way it writes any other; this module only scores each
routed result against the recorded one and says whether the replay confirms the Reference (D108).

A write is scored on two things, not one (D215). Its own answer, as it always was; and the rows the
recording shows it moving that its answer never mentions, read back out of the world once the write
has replayed. A body that answers correctly and leaves another table's row, a history list or a
recomputed total where it found them fails the write check with the reason kind `effect`, and the
later read that saw the stale value is marked `downstream_of` that write rather than blamed on its
own account.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from kullback.ai.provider import Model, ModelConfig, ModelReply, ToolCallRequest
from kullback.runner import loop
from kullback.runner.canon import canonicalize, first_difference, value_at
from kullback.runner.records import ToolCall, Trace, Turn, as_dict, plain

RECORDED = "recorded"
SPOKEN_ROLES = ("assistant", "user")
# Which side the driving loop comes back to on its own after a turn that called a tool (D204).
# `loop.step` asks the model again for as long as its reply carries tool calls, and asks the user
# exactly once for a reply that carries none. So a model turn that called a tool closes its run:
# the loop itself will ask for whatever follows it. A user turn that called a tool closes nothing,
# because nothing will ask the user again before the model speaks.
REASKED_AFTER_CALLS = {"assistant": True, "user": False}
# How one routed call compares with the recording. The first three agree on effect; the rest do not.
SAME, COSMETIC, BOTH_REFUSED = "same", "cosmetic", "both_refused"
DIFFERS, OURS_REFUSED, THEIRS_REFUSED, UNRECORDED = "differs", "ours_refused", "theirs_refused", "unrecorded"
AGREES = frozenset({SAME, COSMETIC, BOTH_REFUSED})
# D215. A write is judged on every row it changed, not only on the answer it gave. `EFFECT` is the
# reason kind a write fails under when the recording shows it moving a column and the replayed body
# left that column where it was; `DOWNSTREAM` marks the later read that saw the stale value, so a
# reader groups the two under one cause instead of repairing the read.
EFFECT, DOWNSTREAM = "effect", "downstream_of"
EFFECTS_NAMED = 20  # columns one failing write names, against a body that moved none of forty


class _Script:
    """One cursor over the Trace's turns, shared by the scripted model and the scripted user.

    A recording does not always alternate one turn of one side with one turn of the other, and the
    loop that drives the replay does (`loop.step`). Wherever the user side can act, a run of
    consecutive turns of one role is a legal recorded shape: a user application that calls a tool of
    its own and then speaks again, a person who sends two messages before the assistant answers, a
    note dropped between two turns. Handing such a run out one turn at a time stalls the cursor,
    because the loop asks the other side next and the role it asks for is not the role that stands
    there; every later turn then counts as a gap and never replays at all.

    So the cursor takes a run as one logical turn, the way it already reads past tool turns: the
    side that is asked is handed everything it said before the other side speaks, in order, and
    every tool call those turns made. A gap is counted only where the turn standing at the cursor is
    of neither the asked role nor a run the cursor can absorb (D204).
    """

    def __init__(self, trace: Trace):
        self.turns = list(trace.turns)
        self.calls = {call.id: call for call in trace.tool_calls if call.id}
        self.pos = 0
        # Per role: the runs of more than one turn taken as one, and the turns they folded in.
        self.absorbed = {role: 0 for role in SPOKEN_ROLES}
        self.absorbed_turns = {role: 0 for role in SPOKEN_ROLES}

    def take(self, role: str) -> list[Turn]:
        """This role's run at the cursor: its turns up to the next turn of the other side.

        Tool turns are read past, here as before, so a run holds turns the recording separated by
        the results of their own calls. The run closes at a turn that called a tool where the loop
        comes back to that role by itself (`REASKED_AFTER_CALLS`), which is what keeps the model
        side reading exactly one turn per query the way it always has.
        """
        run: list[Turn] = []
        for index in range(self.pos, len(self.turns)):
            turn = self.turns[index]
            if turn.role not in SPOKEN_ROLES:
                continue
            if turn.role != role:
                break
            run.append(turn)
            self.pos = index + 1
            if REASKED_AFTER_CALLS.get(role) and turn.tool_call_ids:
                break
        if len(run) > 1:
            self.absorbed[role] += 1
            self.absorbed_turns[role] += len(run) - 1
        return run

    def remaining(self, role: str) -> bool:
        return any(turn.role == role for turn in self.turns[self.pos:])

    def said(self, run: Iterable[Turn]) -> str:
        """What the run says as one turn: its turns in order, one newline between them."""
        return "\n".join(turn.content for turn in run if turn.content)

    def requests(self, run: Iterable[Turn]) -> list[ToolCall]:
        return [self.calls[i] for turn in run for i in turn.tool_call_ids if i in self.calls]


class TraceModel(Model):
    """The recorded assistant, one turn per query: its text and the tool calls it made."""

    def __init__(self, script: _Script, name: str = RECORDED):
        self.script = script
        self.name = name
        self.expected: deque[ToolCall] = deque()
        self.gaps = 0  # a user turn stood where an assistant turn was due, or the reverse

    def query(self, messages: list[dict], tools: Optional[list[dict]] = None,
              config: Optional[ModelConfig] = None) -> ModelReply:
        run = self.script.take("assistant")
        if not run:
            if self.script.remaining("assistant"):
                self.gaps += 1
            return ModelReply(content="", model=self.name)
        requests = self.script.requests(run)
        self.expected.extend(requests)
        return ModelReply(content=self.script.said(run), model=self.name, tool_calls=[
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
        run = self.script.take("user")
        if not run:
            self.gaps += 1
            return ""
        # In recorded order, and before the assistant is asked again: a call the user's own turn
        # made is part of what the assistant answers next, so the Environment has to have seen it.
        for call in self.script.requests(run):
            self._own_call(call)
        return self.script.said(run)

    def _own_call(self, call: ToolCall) -> None:
        if self.state is None or self.router is None:
            return
        args = dict(call.args or {})
        loop.emit(self.state, "tool_call", {"id": call.id, "name": call.name, "args": args, "requestor": "user"})
        outcome = self.router.route(call.name, args, recorded=call, requestor="user")
        payload: dict = {"id": call.id, "name": call.name, "result": outcome.result, "requestor": "user"}
        if outcome.error is not None:
            payload["error"] = as_dict(outcome.error)
        loop.emit(self.state, "tool_result", payload, route=outcome.route, assisted=outcome.assisted)
        counts = self.state.run.route_counts
        counts[outcome.route] = counts.get(outcome.route, 0) + 1


class ScoredRouter:
    """The Router with each answer compared against the call the Trace recorded for it."""

    def __init__(self, router: Any, expected: deque, write_tools: Iterable[str] = (), canon_rules: Any = None,
                 comparer: Any = None, effects: Optional[dict] = None):
        self.inner = router
        self.expected = expected
        self.write_tools = set(write_tools)
        self.canon_rules = canon_rules
        # What knows the schema's column classes and the readers' columns (D187). The Runner cannot
        # import the gates, so it is handed an object with one `agrees` method, the way it is handed
        # the canonicalizer's rules; given none, the comparison is canonical equality alone.
        self.comparer = comparer
        # D215: per recorded call id, the rows and columns the recording shows that call moving, as
        # plain dicts the Builder wrote (`builder/effects.replay_evidence`). Given none, this scores
        # exactly what it scored before. The evidence is code, not a hint, so a held-out Run is
        # checked the same way: nothing about it reaches anyone who writes a body.
        self.effects = dict(effects or {})
        self.checks: list[dict] = []
        # The rows a write was supposed to move and did not, kept so a later read that answered the
        # stale value is marked as the write's consequence rather than as a fault of its own.
        self.stale: list[dict] = []

    def __getattr__(self, name: str) -> Any:  # state_hash, world, start_world, state: the loop's reads
        return getattr(self.inner, name)

    def route(self, name: str, args: Optional[dict] = None, recorded: Optional[ToolCall] = None,
              requestor: str = "assistant") -> Any:
        # D164: the caller goes through, so the inner Router can refuse a tool this caller never had.
        outcome = self.inner.route(name, args, requestor=requestor)
        recorded = recorded if recorded is not None else self._take(name)
        verdict, notes = (UNRECORDED, []) if recorded is None else compare_call_notes(
            recorded, outcome.result, outcome.error, self.canon_rules, self.comparer)
        check = {
            "tool": name, "kind": "write" if name in self.write_tools else "read", "verdict": verdict,
            # Who the call was made by, so a reason can say a user turn's own call parted (D204).
            "requestor": requestor,
            "route": outcome.route, "call_id": recorded.id if recorded is not None else None,
            "ours": _preview(outcome.error if outcome.error is not None else outcome.result),
            "recorded": _preview(recorded.error if recorded is not None and recorded.error is not None
                                 else (recorded.result if recorded is not None else None))}
        if verdict not in AGREES and verdict != UNRECORDED:
            # A call that agreed needs nothing beyond the preview; a call that parted has to say
            # what parted, and 160 characters is not enough to say it (D66).
            check["difference"] = difference(outcome.result, outcome.error, recorded, self.canon_rules)
            if notes:
                # Which columns parted and under which class, which the leaf path cannot say (D187).
                check["difference"]["columns"] = notes[:COLUMN_NOTES]
        self._check_effects(check, recorded, outcome)
        self._mark_downstream(check, recorded)
        self.checks.append(check)
        return outcome

    def _check_effects(self, check: dict, recorded: Optional[ToolCall], outcome: Any) -> None:
        """D215 rule 3: the rows the recording shows this write moving, read back out of the world.

        The write's own answer is already scored; this asks the other half of the question, which
        nothing asked before: did the body leave the world the way the recording left it. Every
        column the evidence names is read at its own path out of the world as it now stands, and one
        that did not reach the value the recording shows is a failure of this write, whatever its
        answer looked like. A row the world no longer holds is the same failure, said as a missing
        row, since a write that deleted what it should have amended moved that column too.

        A column the recording could only credit ambiguously (more than one write between the two
        sightings) still arrives, and says so, but only against the last write credited with it: the
        recording says where such a column ended once every write between the sightings has run, and
        holding an earlier one to it fails a write for a column the next was always going to move.
        Dropping them instead would check nothing wherever a Run wrote twice, which is most of them.
        """
        expected = self.effects.get(str(recorded.id or "")) if recorded is not None else None
        if not expected or check["kind"] != "write" or outcome.error is not None:
            return
        world = self.inner.world() if hasattr(self.inner, "world") else {}
        misses = []
        for row in expected:
            held = ((world.get(str(row.get("table"))) or {}) if isinstance(world, dict) else {})
            record = held.get(str(row.get("row"))) if isinstance(held, dict) else None
            path = str(row.get("path") or "")
            if record is None:
                misses.append({**_effect_note(row), "ours": None, "reason": "the row is not there"})
                continue
            found, value = value_at(plain(record), path)
            if found and canonicalize(value, self.canon_rules) == canonicalize(row.get("after"),
                                                                              self.canon_rules):
                continue
            misses.append({**_effect_note(row), "ours": _preview(value if found else None),
                           "reason": "the column is not there" if not found else "the column did not move"})
        check["effect_checks"] = len(expected)
        if misses:
            check["effect_failures"] = misses[:EFFECTS_NAMED]
            check["effect_failures_total"] = len(misses)
            self.stale.extend({"call_id": check["call_id"], "tool": check["tool"],
                               "table": miss["table"], "row": miss["row"]} for miss in misses)

    def _mark_downstream(self, check: dict, recorded: Optional[ToolCall]) -> None:
        """A later call that parted over a row an earlier write left stale is that write's doing.

        The mark carries the write's call id, so the two lines a reader sees group under one cause.
        It is never a pass: the read still parted and still counts. What it stops is the next round
        spending itself repairing the read, which answered the world it was given correctly.
        """
        if not self.stale or check["verdict"] in AGREES or check["verdict"] == UNRECORDED:
            return
        text = _dumps(plain(recorded.args if recorded is not None else None)) + " " + str(
            check.get("recorded") or "")
        blamed = [row for row in self.stale if row["row"] and str(row["row"]) in text]
        if blamed:
            check[DOWNSTREAM] = blamed[-1]["call_id"]
            check["downstream_tool"] = blamed[-1]["tool"]

    def _take(self, name: str) -> Optional[ToolCall]:
        for index, call in enumerate(self.expected):
            if call.name == name:
                del self.expected[index]
                return call
        return None


def _effect_note(row: dict) -> dict:
    """One expected effect as the fields a reason and a lesson are written from (D215 rules 3, 4)."""
    return {"table": str(row.get("table") or ""), "row": str(row.get("row") or ""),
            "path": str(row.get("path") or ""), "recorded": _preview(row.get("after")),
            "before": _preview(row.get("before")), "ambiguous": bool(row.get("ambiguous")),
            "formulas": list(row.get("formulas") or [])}


def effect_reasons(check: dict) -> list[str]:
    """The reason lines one write's missed effects leave: the kind, then the table and column path."""
    out = []
    for miss in (check.get("effect_failures") or ()):
        out.append(f"{check.get('tool')} write: {EFFECT} {miss.get('table')}.{miss.get('path')} "
                   f"{miss.get('reason')}")
    return out


def compare_call(recorded: ToolCall, result: Any, error: Any, rules: Any = None,
                 comparer: Any = None) -> str:
    """One routed answer against the recorded one: same, cosmetic, or one of the ways they part."""
    return compare_call_notes(recorded, result, error, rules, comparer)[0]


def compare_call_notes(recorded: ToolCall, result: Any, error: Any, rules: Any = None,
                       comparer: Any = None) -> tuple[str, list[str]]:
    """The verdict and, where the two answers were compared column by column, what parted (D187).

    The first two routes are whole-answer equality: the same bytes, then the same canonical string.
    Both hold every value to a hard column's bar, because `canonicalize` takes no column and no
    table. So a third route asks the `comparer`, which knows the schema's classes and the readers'
    columns: a column the schema marks exempt is equal whatever the two hold, a semantic one is
    reported and left to the judge, a prose result is read into the columns it asserts, and a list
    of keyed rows is paired by key rather than by position. Two answers that agree there agree
    cosmetically: the effect is the same and the bytes are not.

    Given no comparer the two old routes stand alone, which is what every caller had before.
    """
    ours_failed, theirs_failed = error is not None, recorded.error is not None
    if ours_failed and theirs_failed:
        return BOTH_REFUSED, []
    if ours_failed:
        return OURS_REFUSED, []
    if theirs_failed:
        return THEIRS_REFUSED, []
    ours, theirs = _norm(result), _norm(recorded.result)
    if _dumps(ours) == _dumps(theirs):
        return SAME, []
    if canonicalize(ours, rules) == canonicalize(theirs, rules):
        return COSMETIC, []
    if comparer is None:
        return DIFFERS, []
    agreed, notes = comparer.agrees(recorded.name, theirs, ours)
    return (COSMETIC if agreed else DIFFERS), list(notes)


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


# What a check that did not agree keeps of each side, against the 160 characters of the preview.
DIFFERENCE_LIMIT = 4000
ERROR_LIMIT = 1000
COLUMN_NOTES = 20  # columns named on a check that parted, each with the class it was held to


def _error_text(error: Any) -> str:
    """The message inside a tool error, which is the part that says why that side refused."""
    if error is None:
        return ""
    payload = error.get("payload") if isinstance(error, dict) else getattr(error, "payload", None)
    return str(payload if payload is not None else _dumps(plain(error)))[:ERROR_LIMIT]


def _kept(value: Any, errored: bool) -> tuple[str, bool, str]:
    """One side's answer as text, whether that was the whole of it, and the type it had."""
    text = _dumps(plain(value))
    return text[:DIFFERENCE_LIMIT], len(text) > DIFFERENCE_LIMIT, "error" if errored else type(plain(value)).__name__


def difference(result: Any, error: Any, recorded: Optional[ToolCall], rules: Any = None) -> dict:
    """Why one routed call parted from its recording, in a form the cause can be read off.

    The two preview fields are cut at 160 characters, which is enough to see an answer and not
    enough to say what about it differs, so a report reading them alone has to call the cause
    unreadable. This is written for the calls that did not agree and for those only: each side's
    error message, the type each answer had, the keys only one side has and the keys both have
    with different values, and each answer up to `DIFFERENCE_LIMIT` characters with a flag saying
    whether that was all of it. The preview fields keep the bytes they always kept. `leaf` names
    the first place the two answers part under the canon rules (D154): the key list says `items`
    changed, the leaf says which item, which field, and both values.
    """
    ours_errored = error is not None
    theirs_errored = recorded is not None and recorded.error is not None
    ours = plain(error if ours_errored else result)
    theirs = plain(recorded.error if theirs_errored else recorded.result) if recorded is not None else None
    ours_text, ours_cut, ours_type = _kept(ours, ours_errored)
    theirs_text, theirs_cut, theirs_type = _kept(theirs, theirs_errored)
    out = {"ours": ours_text, "ours_truncated": ours_cut, "ours_type": ours_type,
           "ours_errored": ours_errored, "ours_error": _error_text(error),
           "theirs": theirs_text, "theirs_truncated": theirs_cut, "theirs_type": theirs_type,
           "theirs_errored": theirs_errored,
           "theirs_error": _error_text(recorded.error if recorded is not None else None),
           "type_mismatch": ours_type != theirs_type}
    if isinstance(ours, dict) and isinstance(theirs, dict):
        out["keys_only_ours"] = sorted(set(ours) - set(theirs))
        out["keys_only_theirs"] = sorted(set(theirs) - set(ours))
        out["keys_changed"] = sorted(key for key in set(ours) & set(theirs)
                                     if _dumps(ours[key]) != _dumps(theirs[key]))
    if isinstance(ours, list) and isinstance(theirs, list):
        out["lengths"] = [len(ours), len(theirs)]
    if not ours_errored and not theirs_errored:
        out["leaf"] = first_difference(_norm(result), _norm(recorded.result) if recorded else None, rules)
    return out


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
                 write_tools: Iterable[str] = (), canon_rules: Any = None, run_id: Optional[str] = None,
                 comparer: Any = None, effects: Optional[dict] = None) -> Replay:
    """Drive the loop with the Trace's own turns over `router`; the Run lands under `workdir`.

    `effects` is D215's write evidence, per recorded call id: the rows and columns the recording
    shows that call moving. Given none, every call is scored on its own answer, which is what this
    did before.
    """
    script = _Script(trace)
    model, user = TraceModel(script), TraceUser(script)
    scored = ScoredRouter(router, model.expected, write_tools, canon_rules, comparer, effects)
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
    return _score(trace, state, scored, script, model, user, crashed)


def _label(check: dict) -> str:
    """How a check is named in a reason: a call a user turn made itself says so, not read or write."""
    return "user_call" if check.get("requestor") == "user" else check["kind"]


def _score(trace: Trace, state: Any, scored: ScoredRouter, script: _Script, model: TraceModel,
           user: TraceUser, crashed: Optional[str]) -> Replay:
    writes = [c for c in scored.checks if c["kind"] == "write"]
    reads = [c for c in scored.checks if c["kind"] == "read"]
    # D215: a write that answered the way the recording did and left a row the recording moved where
    # it was has not replayed the recording, so it counts among the writes that are off. The verdict
    # on its own answer is left as it was: the two say different things and both are worth reading.
    effect_off = [c for c in scored.checks if c.get("effect_failures")]
    writes_off = [c for c in writes if c["verdict"] not in AGREES or c.get("effect_failures")]
    reads_off = [c for c in reads if c["verdict"] not in AGREES]
    unmade = [call.name for call in model.expected]
    counts = {
        "calls": len(scored.checks), "writes": len(writes), "writes_matched": len(writes) - len(writes_off),
        "reads": len(reads), "reads_same": sum(c["verdict"] == SAME for c in reads),
        "reads_cosmetic": sum(c["verdict"] == COSMETIC for c in reads),
        "reads_both_refused": sum(c["verdict"] == BOTH_REFUSED for c in reads),
        "reads_semantic": len(reads_off), "unmade": len(unmade), "gaps": model.gaps + user.gaps,
        # What the cursor read as one logical turn rather than as a gap it could never resync from.
        "absorbed_user_runs": script.absorbed["user"], "absorbed_model_runs": script.absorbed["assistant"],
        "absorbed_turns": sum(script.absorbed_turns.values()),
        # D215's own counters: columns checked after a write replayed, columns that did not reach
        # the value the recording shows, and the later calls those columns are blamed for.
        "effect_checks": sum(int(c.get("effect_checks") or 0) for c in scored.checks),
        "effect_failures": sum(int(c.get("effect_failures_total") or 0) for c in scored.checks),
        "effects_downstream": sum(1 for c in scored.checks if c.get(DOWNSTREAM)),
        "routes": dict(state.run.route_counts),
    }
    reasons = [f"{c['tool']} {_label(c)}: {c['verdict']}"
               for c in writes_off if c["verdict"] not in AGREES]
    reasons += [line for c in effect_off for line in effect_reasons(c)]
    reasons += [f"{c['tool']} {_label(c)}: {c['verdict']}"
                + (f", {DOWNSTREAM} {c['downstream_tool']}" if c.get(DOWNSTREAM) else "")
                for c in reads_off]
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


# The per-Task ruling over these records (some Trace confirmed, or why none did) is
# `kullback.gates.fidelity.reference_replay_gate`; this module scores one replay and stops.
__all__ = ["Replay", "ScoredRouter", "TraceModel", "TraceUser", "compare_call", "compare_call_notes",
           "difference", "effect_reasons", "replay_trace"]
