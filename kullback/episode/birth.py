"""Checks at birth for a Task, by code alone: a scripted walk passes, do-nothing fails.

The walk played through the episode interface must score 1 while a run that calls
nothing scores 0; every argument in the walk must trace to the intent, to a fact the
simulated user holds, or to an earlier result; and a forbidden step must trip the
caller's policy predicate. This layer owns no policy: the caller supplies
`violations_of`, and this module never imports the builder that compiles it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from kullback.episode.episode import EpisodeError

HELD = "held"
FAILED = "failed"
NO_SIGNAL = "no_signal"
NOT_RUN = "not_run"


@dataclass
class CheckResult:
    """One check's verdict: its name, held/failed/no_signal/not_run, and a plain reason."""

    name: str
    outcome: str
    reason: str


@dataclass
class UntracedArg:
    """One argument value no allowed source holds: call index, dotted path, string value."""

    call_index: int
    path: str
    value: str


@dataclass
class WalkResult(CheckResult):
    """The walk-passes check plus every step result, for the trace check to reuse."""

    results: list = field(default_factory=list)


@dataclass
class TraceResult(CheckResult):
    """The arguments-trace check plus each untraced argument."""

    untraced: list = field(default_factory=list)


@dataclass
class BirthReport:
    """Every birth check that applied, plus whether the Task is born."""

    walk_passes: CheckResult
    do_nothing_fails: CheckResult
    arguments_trace: CheckResult
    forbidden_step: CheckResult
    born: bool


def _field(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _leaf_paths(value: Any, segments: list) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, segments + [str(key)])
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, segments + [str(index)])
    elif value is None or value == "" or isinstance(value, bool):
        return
    else:
        yield (".".join(segments), str(value))


def _leaf_set(value: Any) -> set:
    return {text for _, text in _leaf_paths(value, [])}


def _intent_holds(intent_text: str, text: str) -> bool:
    """Whether the intent holds the value as a whole token run, never inside a longer one."""
    if not intent_text:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(text) + r"(?![A-Za-z0-9])",
                     intent_text) is not None


def _fact_texts(user_facts: Any) -> list:
    if user_facts is None:
        return []
    if isinstance(user_facts, dict):
        items = list(user_facts.values())
    else:
        items = list(user_facts)
    texts = []
    for item in items:
        if isinstance(item, dict) and "value" in item:
            texts.extend(_leaf_set(item["value"]))
        else:
            texts.extend(_leaf_set(item))
    return texts


def _call_id(index: int) -> str:
    """The id `_drive` assigns the walk's call at the 0-based position: call-1, call-2, ..."""
    return f"call-{index + 1}"


def _call_args(call: Any) -> Any:
    if isinstance(call, dict):
        args = call.get("arguments", None)
        if args is None:
            args = call.get("args", {})
        return args or {}
    args = getattr(call, "arguments", None)
    if args is None:
        args = getattr(call, "args", {})
    return args or {}


def _drive(episode: Any, task_id: str, calls: list, *, seed: int, closing: str,
           max_steps: int) -> list:
    """Reset, send every call in one message, then empty closings until the run ends."""
    episode.reset(task_id, seed)
    planned = []
    for index, call in enumerate(calls):
        planned.append({"id": _call_id(index), "name": _field(call, "name"),
                        "arguments": _call_args(call)})
    out = episode.step({"content": closing, "tool_calls": planned})
    results = list(_field(out, "results") or [])
    steps = 1
    while not _field(out, "done") and steps < max_steps:
        out = episode.step({"content": closing, "tool_calls": []})
        results.extend(_field(out, "results") or [])
        steps += 1
    if not _field(out, "done"):
        raise EpisodeError(f"Run {task_id} did not finish within {max_steps} steps")
    return results


def check_walk_passes(episode: Any, task_id: str, walk: Any, *, seed: int = 0,
                      closing: str = "Done.", max_steps: int = 8) -> WalkResult:
    """Drive the walk's calls in one message; holds only when the reward scores 1."""
    try:
        results = _drive(episode, task_id, list(walk or []), seed=seed, closing=closing,
                         max_steps=max_steps)
        reward = episode.reward()
    except EpisodeError as exc:
        return WalkResult(name="walk_passes", outcome=FAILED, reason=str(exc), results=[])
    score = _field(reward, "score")
    words = _field(reward, "reason") or _field(reward, "class_") or ""
    if score == 1:
        return WalkResult(name="walk_passes", outcome=HELD, reason="walk scored 1",
                          results=results)
    if score is None:
        return WalkResult(name="walk_passes", outcome=NO_SIGNAL,
                          reason=f"walk gave no signal ({words})", results=results)
    return WalkResult(name="walk_passes", outcome=FAILED,
                      reason=f"walk scored {score} ({words})", results=results)


def check_do_nothing_fails(episode: Any, task_id: str, *, seed: int = 0,
                           closing: str = "Done.", max_steps: int = 8) -> CheckResult:
    """Drive a run with no tool calls; holds only when the reward scores 0."""
    try:
        _drive(episode, task_id, [], seed=seed, closing=closing, max_steps=max_steps)
        reward = episode.reward()
    except EpisodeError as exc:
        return CheckResult(name="do_nothing_fails", outcome=FAILED, reason=str(exc))
    score = _field(reward, "score")
    words = _field(reward, "reason") or _field(reward, "class_") or ""
    if score == 0:
        return CheckResult(name="do_nothing_fails", outcome=HELD, reason="do-nothing scored 0")
    if score is None:
        return CheckResult(name="do_nothing_fails", outcome=NO_SIGNAL,
                           reason=f"do-nothing gave no signal ({words}): "
                           "a verifier that cannot say no gives no training signal")
    return CheckResult(name="do_nothing_fails", outcome=FAILED,
                       reason=f"do-nothing scored {score} ({words}): "
                       "the verifier passes a run that did nothing")


def check_arguments_trace(walk: Any, results: Any, *, intent: Any = None,
                          user_facts: Any = None) -> TraceResult:
    """Hold every scalar arg leaf against the intent, user facts, and earlier results.

    Booleans, None and the empty string carry no trace burden.
    """
    facts = set(_fact_texts(user_facts))
    intent_text = intent if isinstance(intent, str) else ""
    prior = set()
    untraced = []
    calls = list(walk or [])
    seen_by_id = {}
    for entry in list(results or []):
        seen_by_id[_field(entry, "id")] = _leaf_set(_field(entry, "result"))
    for index, call in enumerate(calls):
        for path, text in _leaf_paths(_call_args(call), []):
            if text in facts or text in prior or _intent_holds(intent_text, text):
                continue
            untraced.append(UntracedArg(call_index=index, path=path, value=text))
        prior.update(seen_by_id.get(_call_id(index), set()))
    if not untraced:
        return TraceResult(name="arguments_trace", outcome=HELD,
                           reason="every argument traces to the intent, "
                           "a user fact, or an earlier result", untraced=[])
    detail = "; ".join(f"call {item.call_index} {item.path or 'args'}={item.value!r}"
                       for item in untraced)
    return TraceResult(name="arguments_trace", outcome=FAILED,
                       reason=f"{len(untraced)} untraced argument(s): {detail}",
                       untraced=untraced)


def check_forbidden_step_trips(episode: Any, task_id: str, forbidden_walk: Any,
                               violations_of: Optional[Callable[..., Any]], *, seed: int = 0,
                               closing: str = "Done.", max_steps: int = 8,
                               control_walk: Any = None) -> CheckResult:
    """Drive the forbidden walk; holds when the caller's predicate reports a violation.

    A given control walk is driven first and must leave the predicate quiet: a predicate
    that trips on the passing walk too fails the check before the forbidden walk is driven.
    """
    if violations_of is None:
        return CheckResult(name="forbidden_step", outcome=NOT_RUN,
                           reason="no policy predicate supplied")
    if forbidden_walk is None:
        return CheckResult(name="forbidden_step", outcome=NOT_RUN,
                           reason="no forbidden walk supplied")
    try:
        if control_walk is not None:
            _drive(episode, task_id, list(control_walk), seed=seed, closing=closing,
                   max_steps=max_steps)
            control_hits = violations_of(episode.run_path())
            if control_hits:
                return CheckResult(name="forbidden_step", outcome=FAILED,
                                   reason="predicate trips on the passing walk too: "
                                   "forbidden walk not driven")
        _drive(episode, task_id, list(forbidden_walk), seed=seed, closing=closing,
               max_steps=max_steps)
        found = violations_of(episode.run_path())
    except EpisodeError as exc:
        return CheckResult(name="forbidden_step", outcome=FAILED, reason=str(exc))
    items = list(found or [])
    if items:
        return CheckResult(name="forbidden_step", outcome=HELD,
                           reason=f"forbidden walk tripped {len(items)} violation(s)")
    return CheckResult(name="forbidden_step", outcome=FAILED,
                       reason="forbidden walk tripped no violation")


def birth_checks(episode: Any, task_id: str, walk: Any, *, seed: int = 0,
                 closing: str = "Done.", max_steps: int = 8, intent: Any = None,
                 user_facts: Any = None, forbidden_walk: Any = None,
                 violations_of: Optional[Callable[..., Any]] = None) -> BirthReport:
    """Run the checks that apply; born only when each one that ran held."""
    walked = check_walk_passes(episode, task_id, walk, seed=seed, closing=closing,
                               max_steps=max_steps)
    idle = check_do_nothing_fails(episode, task_id, seed=seed, closing=closing,
                                  max_steps=max_steps)
    if intent is None and user_facts is None:
        traced: CheckResult = TraceResult(name="arguments_trace", outcome=NOT_RUN,
                                          reason="no intent or user facts supplied",
                                          untraced=[])
    else:
        traced = check_arguments_trace(walk, walked.results, intent=intent,
                                       user_facts=user_facts)
    forbidden = check_forbidden_step_trips(episode, task_id, forbidden_walk, violations_of,
                                           seed=seed, closing=closing, max_steps=max_steps,
                                           control_walk=walk)
    ran = [check for check in (walked, idle, traced, forbidden)
           if check.outcome != NOT_RUN]
    born = bool(ran) and all(check.outcome == HELD for check in ran)
    return BirthReport(walk_passes=walked, do_nothing_fails=idle, arguments_trace=traced,
                       forbidden_step=forbidden, born=born)
