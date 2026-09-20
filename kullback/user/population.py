from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any, Optional

from pydantic import ConfigDict, StrictBool

from kullback.runner.records import Record, Task, Trace, UserRules, as_dict, canonical_json, content_hash
from kullback.sampling import sample_key


class RecordingStanding(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    task_eligible: StrictBool
    complete: StrictBool
    replay_confirmed: Optional[StrictBool] = None


class Withheld(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    recording_id: str
    reasons: tuple[str, ...]


class Population(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    task_id: str
    reference_id: str
    eligible_ids: tuple[str, ...]
    withheld: tuple[Withheld, ...]
    support_count: int
    fingerprint: str


class Selection(Record):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)
    task_id: str
    chosen_id: Optional[str] = None
    seed: int
    population_fingerprint: str


def _write_signature(trace: Trace, write_tools: Collection[str]) -> tuple[tuple[str, str], ...]:
    tools = set(write_tools)
    pairs: list[tuple[str, str]] = []
    for call in trace.tool_calls:
        if call.name in tools:
            pairs.append((call.name, canonical_json(call.args)))
    return tuple(pairs)


def _trace_reasons(trace: Trace) -> list[str]:
    truncated = False
    unanswered = False
    for call in trace.tool_calls:
        if call.truncated:
            truncated = True
        if call.has_result is not True:
            unanswered = True
        elif call.resolved is not True:
            unanswered = True
    out: list[str] = []
    if truncated:
        out.append("truncated")
    if unanswered:
        out.append("unanswered")
    return out


def _standing_reasons(standing: Optional[RecordingStanding]) -> list[str]:
    if standing is None:
        return ["unknown_standing"]
    out: list[str] = []
    if not standing.task_eligible:
        out.append("ineligible")
    if not standing.complete:
        out.append("incomplete")
    if standing.replay_confirmed is not None and standing.replay_confirmed is False:
        out.append("unconfirmed")
    return out


def _reference_signature(
    task: Task,
    traces: Mapping[str, Trace],
    standings: Mapping[str, RecordingStanding],
    rules: Mapping[str, UserRules],
    held_set: set[str],
    tools_set: set[str],
    reference_id: str,
) -> tuple[tuple[str, str], ...]:
    if reference_id not in list(task.run_ids):
        raise ValueError("reference not in task")
    if reference_id in held_set:
        raise ValueError("reference held out")
    standing_problems = _standing_reasons(standings.get(reference_id))
    if standing_problems:
        raise ValueError("reference standing " + standing_problems[0])
    trace = traces.get(reference_id)
    if trace is None:
        raise ValueError("reference trace missing")
    if rules.get(reference_id) is None:
        raise ValueError("reference rules missing")
    call_problems = _trace_reasons(trace)
    if call_problems:
        raise ValueError("reference calls " + call_problems[0])
    return _write_signature(trace, tools_set)


def _candidate_reasons(
    rid: str,
    held_set: set[str],
    standings: Mapping[str, RecordingStanding],
    traces: Mapping[str, Trace],
    rules: Mapping[str, UserRules],
    tools_set: set[str],
    ref_sig: tuple[tuple[str, str], ...],
) -> list[str]:
    out: list[str] = []
    if rid in held_set:
        out.append("held_out")
    out.extend(_standing_reasons(standings.get(rid)))
    trace = traces.get(rid)
    if trace is None:
        out.append("missing_trace")
    if rules.get(rid) is None:
        out.append("missing_rules")
    if trace is not None:
        out.extend(_trace_reasons(trace))
        if _write_signature(trace, tools_set) != ref_sig:
            out.append("write_mismatch")
    return out


def _fingerprint(
    task_id: str,
    reference_id: str,
    tools_sorted: list[str],
    ordered_ids: list[str],
    held_set: set[str],
    standings: Mapping[str, RecordingStanding],
    traces: Mapping[str, Trace],
    rules: Mapping[str, UserRules],
    tools_set: set[str],
) -> str:
    runs: list[dict[str, Any]] = []
    for rid in ordered_ids:
        standing = standings.get(rid)
        if standing is None:
            standing_part: Any = None
        else:
            standing_part = [standing.task_eligible, standing.complete, standing.replay_confirmed]
        rule = rules.get(rid)
        if rule is None:
            rules_part: Any = None
        else:
            rules_part = as_dict(rule)
        trace = traces.get(rid)
        if trace is None:
            writes_part: Any = None
            calls_part: Any = None
        else:
            writes: list[Any] = []
            for call in trace.tool_calls:
                if call.name in tools_set:
                    writes.append([call.name, call.args])
            writes_part = writes
            calls: list[Any] = []
            for call in trace.tool_calls:
                calls.append([call.truncated, call.has_result, call.resolved])
            calls_part = calls
        runs.append(
            {
                "calls": calls_part,
                "has_trace": trace is not None,
                "held_out": rid in held_set,
                "id": rid,
                "rules": rules_part,
                "standing": standing_part,
                "writes": writes_part,
            }
        )
    payload: dict[str, Any] = {
        "reference_id": reference_id,
        "runs": runs,
        "task_id": task_id,
        "write_tools": tools_sorted,
    }
    return content_hash(payload)


def build_population(
    task: Task,
    traces: Mapping[str, Trace],
    *,
    reference_id: str,
    standings: Mapping[str, RecordingStanding],
    rules: Mapping[str, UserRules],
    held_out_ids: Collection[str],
    write_tools: Collection[str],
) -> Population:
    held_set = set(held_out_ids)
    tools_set = set(write_tools)
    tools_sorted = sorted(tools_set)
    ref_sig = _reference_signature(task, traces, standings, rules, held_set, tools_set, reference_id)
    ordered = sorted(set(task.run_ids))
    eligible: list[str] = []
    withheld_list: list[Withheld] = []
    for rid in ordered:
        if rid == reference_id:
            eligible.append(rid)
            continue
        reasons = _candidate_reasons(rid, held_set, standings, traces, rules, tools_set, ref_sig)
        if reasons:
            withheld_list.append(Withheld(recording_id=rid, reasons=tuple(reasons)))
        else:
            eligible.append(rid)
    eligible_sorted = sorted(eligible)
    fingerprint = _fingerprint(
        task.id, reference_id, tools_sorted, ordered, held_set, standings, traces, rules, tools_set
    )
    return Population(
        task_id=task.id,
        reference_id=reference_id,
        eligible_ids=tuple(eligible_sorted),
        withheld=tuple(withheld_list),
        support_count=len(eligible_sorted),
        fingerprint=fingerprint,
    )


def pick_user(population: Population, *, seed: int, salt: str) -> Selection:
    if type(seed) is not int:
        raise TypeError("seed must be int")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(salt, str):
        raise TypeError("salt must be str")
    eligible = list(population.eligible_ids)
    if not eligible:
        return Selection(
            task_id=population.task_id,
            chosen_id=None,
            seed=seed,
            population_fingerprint=population.fingerprint,
        )
    ident = json.dumps(
        [population.task_id, seed, population.fingerprint],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    key = sample_key("user_population", ident, salt)
    chosen = sorted(eligible)[key % len(eligible)]
    return Selection(
        task_id=population.task_id,
        chosen_id=chosen,
        seed=seed,
        population_fingerprint=population.fingerprint,
    )
