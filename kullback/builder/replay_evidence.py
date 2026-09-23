"""Replay evidence: which recorded calls a replay failed on, and the sentences they become.

Carried verbatim from kullback/builder/build.py (the stage graph is gone; these functions are
generic replay diffing and evidence counting over a schema and traces, not stage specific).
`failure_shapes` and `failure_shape` came along from kullback/builder/repair.py, which
`replay_lesson` reads them from.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.builder import compile_env
from kullback.runner.records import Trace
from kullback.runner.records import read_json as _read_json

# What the replay of the References wrote, and the two verdicts it leaves on a check that agreed.
# Everything else is a recorded call the replay failed to reproduce (D191).
REPLAYS_FILE = "replays.json"
HOLDOUT_ANSWERS_FILE = "holdout_answers.json"
REPLAY_AGREED = frozenset({"same", "both_refused"})
# The head of the lesson those failures become, said once above the shapes.
REPLAY_LESSON_HEAD = (
    "The replay of the References failed on recorded calls of this tool. Every one of them is a call "
    "a Task's own recording made, and a Run of that Task cannot confirm until the body answers it the "
    "way the recording did. Repair these first.")

SHAPES_SHOWN = 3
_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+\b")


def failure_shape(text: str) -> str:
    """One failure sentence with its values generalized, so two calls that failed the same way group."""
    return _LITERAL.sub("*", " ".join(str(text).split()))


def failure_shapes(failures: Iterable[str]) -> list[tuple[str, int]]:
    """The distinct kinds of failure among a gate's sentences: the first of each kind, and how many share it.

    A gate that ruled over sixty recorded calls leaves sixty sentences, and one of them was all the
    mechanic ever saw, so the hint it wrote answered one example and the next recompile met the
    fifty-nine it had not been told about. Grouped by shape, the same sixty sentences say how many
    kinds of failure there are, which is what a hint has to answer.
    """
    seen: dict[str, list[Any]] = {}
    for failure in failures:
        row = seen.setdefault(failure_shape(failure), [str(failure), 0])
        row[1] += 1
    return [(text, count) for text, count in seen.values()]


def replay_difference(check: Any) -> str:
    """One replay-failing check as a sentence: the verdict and the leaf where the two answers part.

    `runner/replay.compare_call` writes a check per recorded call with `verdict`, and for a call it
    failed a `difference` holding either a `leaf` (the column and the two values) or the key sets
    that differ. The leaf is what a body is repaired by: a column name and the recorded value
    against the replayed one. A check with neither is named by its verdict alone, which still says
    the recording refused where the body answered or the other way round.
    """
    if not isinstance(check, dict):
        return ""
    verdict = str(check.get("verdict") or "differs")
    # D215 rule 4: a write that answered correctly and left a row the recording moved has its
    # differing leaf on its own line, named by the column and by the formula the recording implies,
    # so the next rewrite repairs the write rather than the read that later saw the stale value.
    effect = _effect_sentence(check)
    if effect:
        return effect
    difference = check.get("difference")
    if isinstance(difference, dict):
        leaf = difference.get("leaf")
        if leaf:
            return f"{verdict}: {leaf}"
        changed = [str(k) for k in (difference.get("keys_changed") or [])]
        missing = [str(k) for k in (difference.get("keys_only_theirs") or [])]
        extra = [str(k) for k in (difference.get("keys_only_ours") or [])]
        parts = [part for part in (
            f"columns that differ: {', '.join(sorted(changed))}" if changed else "",
            f"columns the recording has and the body does not: {', '.join(sorted(missing))}" if missing else "",
            f"columns the body has and the recording does not: {', '.join(sorted(extra))}" if extra else "",
        ) if part]
        if parts:
            return f"{verdict}: {'; '.join(parts)}"
    return verdict


def replay_failures(workdir: Any) -> dict[str, dict[str, str]]:
    """Per tool, the recorded calls the last replay of the References failed on, and how (D191).

    The compile gate scores a body against the calls that survive the evidence filters (the
    after-write skip, the anchor hold-out, the caller filter). Those filters each answer a question
    about what a body may be written from, and none of them answers the question the Reference
    replay asks, which is whether a Task's own recording plays back. So a call the replay failed on
    is evidence whatever a filter says about it: the recording made that call, and the Task cannot
    confirm while the body answers it differently.

    Read off `replays.json`, which the replay stage rewrites whole every round, so this is the
    latest round's reasons and nothing older. A build that has not replayed yet has no file and
    this is empty, which is the first pass: there is nothing yet to have failed.
    """
    return replay_failures_of(_read_json(Path(workdir) / REPLAYS_FILE, {}) or {})


def replay_failures_of(replays: Any) -> dict[str, dict[str, str]]:
    """`replay_failures` over the replay record in hand, so the stage that writes it can index it."""
    rows: dict[str, dict[str, str]] = {}
    if not isinstance(replays, dict):
        return rows
    for per_task in replays.values():
        for record in (per_task or {}).values() if isinstance(per_task, dict) else ():
            if not isinstance(record, dict) or record.get("confirmed"):
                continue
            for check in record.get("checks") or []:
                if not isinstance(check, dict):
                    continue
                # D215: a write whose own answer agreed and whose effects did not is a call this
                # replay could not reproduce, so it is evidence for the body like any other.
                if check.get("verdict") in REPLAY_AGREED and not check.get("effect_failures"):
                    continue
                tool, call_id = check.get("tool"), check.get("call_id")
                if not tool or not call_id:
                    continue
                rows.setdefault(str(tool), {}).setdefault(str(call_id), replay_difference(check))
    return rows


class HeldOutRuns:
    """anchor.json's held-out Runs, answering the one question `holdout_answers` asks of an anchor."""

    def __init__(self, held_out: Any):
        self.held_out = held_out if isinstance(held_out, dict) else {}

    @classmethod
    def of(cls, workdir: Any) -> "HeldOutRuns":
        """The workdir's anchor, or one holding nothing out where no anchor was chosen."""
        body = _read_json(Path(workdir) / "anchor.json", None)
        return cls((body or {}).get("held_out") if isinstance(body, dict) else None)

    def is_held_out(self, run_id: str) -> bool:
        return any(run_id in (runs or ()) for runs in self.held_out.values())


def holdout_answers(replays: dict, anchor: Any = None) -> dict:
    """How many replayed calls were answered out of a value only a held-out Run witnessed (D220 2c).

    Counted per Task and over the corpus, and split by whether the Run being replayed is itself one
    of the held-out ones, because those are the Runs the number is asked about: a held-out Run that
    passes on a value the world holds only on its own word passed on the world, not on the body.
    This is a finding on the record; no gate reads it.
    """
    per_task: dict[str, dict] = {}
    per_tool: dict[str, dict] = {}
    totals = {"calls": 0, "held_out_calls": 0, "runs": 0, "held_out_runs": 0}
    columns: set[str] = set()
    for task_id, rows in sorted((replays or {}).items()):
        for trace_id, row in sorted((rows or {}).items()):
            counts = (row or {}).get("counts") or {}
            calls = int(counts.get("answered_from_holdout") or 0)
            held = bool(anchor is not None and anchor.is_held_out(trace_id))
            # The runner's ReplayReport keeps its per-call rows under "calls"; the deleted stage
            # wrote the replay's own "checks". Either is read, the first one present.
            for check in (row or {}).get("checks") or (row or {}).get("calls") or ():
                if not isinstance(check, dict) or not check.get("answered_from_holdout"):
                    continue
                tool = per_tool.setdefault(str(check.get("tool") or ""),
                                           {"calls": 0, "held_out_calls": 0})
                tool["calls"] += 1
                if held:
                    tool["held_out_calls"] += 1
            if not calls:
                continue
            columns.update(counts.get("holdout_columns") or ())
            slot = per_task.setdefault(task_id, {"calls": 0, "held_out_calls": 0, "runs": 0,
                                                 "held_out_runs": 0})
            slot["calls"] += calls
            slot["runs"] += 1
            totals["calls"] += calls
            totals["runs"] += 1
            if held:
                slot["held_out_calls"] += calls
                slot["held_out_runs"] += 1
                totals["held_out_calls"] += calls
                totals["held_out_runs"] += 1
    return {"totals": totals, "columns": sorted(columns), "tasks": per_task,
            "tools": dict(sorted(per_tool.items()))}


def _effect_sentence(check: dict) -> str:
    """What a write left where the recording moved it, as the line the next body answers.

    The column and the value it should have reached, then the formula where the recording's own
    numbers held one. The formula is a rule over columns and never a value, so a writer following
    it computes the value rather than writing it down, which is what D215 rule 5 refuses.
    """
    misses = [m for m in (check.get("effect_failures") or ()) if isinstance(m, dict)]
    if not misses:
        return ""
    parts = []
    for miss in misses[:SHAPES_SHOWN]:
        where = f"{miss.get('table')}.{miss.get('path')}"
        formulas = [str(f) for f in (miss.get("formulas") or ())]
        rule = f", and the recording implies {formulas[0]}" if formulas else ""
        parts.append(f"{where} should have reached {miss.get('recorded')} and the body left it at "
                     f"{miss.get('ours')} ({miss.get('reason')}){rule}")
    left = len(misses) - len(parts)
    if left:
        parts.append(f"{left} more column(s) the recording moved and the body did not")
    return ("this call changed rows its own answer never mentions, and the body changed none of "
            "them: " + "; ".join(parts))


def replay_lesson(failures: dict[str, str], shown_ids: Iterable[str]) -> str:
    """What the Reference replay failed on for this tool, as the sentence the next body answers.

    Grouped by shape with counts, the way every other hint is (D181 rule 6): a tool whose replay
    failed on a hundred calls leaves a hundred sentences, and a writer shown one of them answers one
    of them. `shown_ids` is the calls this writer is allowed to see; a call it is not shown (the
    held-out split, a Run held out as the anchor) is counted and never quoted, which is the same
    masking `kept_body_hint` applies.
    """
    shown_ids = set(shown_ids)
    quoted = [text for call_id, text in sorted(failures.items()) if call_id in shown_ids and text]
    withheld = len(failures) - len(quoted)
    shapes = failure_shapes(quoted)
    text = "; ".join(f"{shape} ({count} call{'' if count == 1 else 's'})"
                     for shape, count in shapes[:SHAPES_SHOWN])
    if len(shapes) > SHAPES_SHOWN:
        text += f"; {len(shapes) - SHAPES_SHOWN} more shapes"
    if withheld:
        text += ("; " if text else "") + f"{withheld} more on calls you were not shown"
    return f"{REPLAY_LESSON_HEAD}\n- {text}" if text else ""


def after_write_calls(traces: Iterable[Trace], write_tools: Iterable[str]) -> set[tuple[str, int]]:
    """(trace id, position) of every call that names a value an earlier write in the same trace named.

    The value is any string argument, on its own or in a list: an id the write changed the row of,
    which a later read or write in that trace then saw in its changed version. Only a write that
    succeeded counts, since a refused write changed nothing.
    """
    writes = set(write_tools)
    out: set[tuple[str, int]] = set()
    for trace in traces:
        touched: set[str] = set()
        for at, call in enumerate(trace.tool_calls):
            names = {v for v in call.args.values() if isinstance(v, str)}
            names |= {v for vs in call.args.values() if isinstance(vs, list) for v in vs if isinstance(v, str)}
            if names & touched:
                out.add((trace.trace_id, at))
            if call.name in writes and call.error is None:
                touched |= names
    return out


def evidence_calls(traces: Iterable[Trace], seeds: set[str], after_write: set[tuple[str, int]],
                   callers: dict[str, set[str]],
                   replay_failed: Optional[dict[str, dict[str, str]]] = None
                   ) -> tuple[dict[str, list], dict[str, int], dict[str, int]]:
    """Per tool, the recorded calls its body is written against and graded on, with the two counts.

    Three filters say a recorded call is not evidence about a body: a call that followed a write on
    the same row saw a world no Starting state can put back (D74), a call in a Run held out as the
    anchor is not the Builder's to learn from (D81), and a call from a requestor the tool never
    answered was refused for a reason of its own (D164). Each of those is about what a body may be
    written from.

    `replay_failed` is the other question: which recorded calls the last replay of the References
    could not reproduce. A call there is evidence whatever a filter says, because a Task confirms
    only when its own recording plays back, and a body scored on a set that leaves those calls out
    can clear every gate while the corpus stays where it was (D191). The counts returned are the
    after-write skips, which the tool's row has always carried, and how many calls a replay failure
    put back over a filter, which is what says how much of the evidence is there for that reason.

    D220: the seed filter is not one of the filters a replay failure overrides. D191's premise holds
    for a Run the Builder may learn from and for no other: a held-out Run's failure belongs in the
    corpus fidelity number, which the replay stage already reports, and putting its arguments and its
    recorded result in front of the body writer is the leak the readmission was never meant to be.
    """
    replay_failed = replay_failed or {}
    calls_by_tool: dict[str, list] = {}
    skipped: dict[str, int] = {}
    from_replay: dict[str, int] = {}
    for trace in traces:
        for at, call in enumerate(trace.tool_calls):
            readmitted = (bool(call.id) and call.id in replay_failed.get(call.name, {})
                          and trace.trace_id in seeds)
            if not (readmitted or is_evidence_call(call, callers)):
                continue
            dropped = (trace.trace_id, at) in after_write or trace.trace_id not in seeds
            if dropped and not readmitted:
                if (trace.trace_id, at) in after_write:
                    skipped[call.name] = skipped.get(call.name, 0) + 1
                continue
            calls_by_tool.setdefault(call.name, []).append(call)
            if dropped:  # a filter would have dropped it and the replay failure overrode the filter
                from_replay[call.name] = from_replay.get(call.name, 0) + 1
    return calls_by_tool, skipped, from_replay


def callers_by_tool(sigs: Iterable[Any]) -> dict[str, set[str]]:
    """Each mined tool's callers as a set, for the evidence filter below (D164)."""
    return {sig.name: set(sig.callers or ["assistant"]) for sig in sigs}


def is_evidence_call(call: Any, callers: dict[str, set[str]]) -> bool:
    """Whether this recorded call is evidence for the body of the tool it named (D164).

    A tool the recording answers for one caller and refuses for another was refused for a reason,
    and the refusal says nothing about what the body does. Only a call from a caller the tool
    answers is written against; a call from anyone else is dropped, the same way the Router will
    refuse it in a Run.
    """
    return (call.requestor or "assistant") in callers.get(call.name, {"assistant"})


def trace_worlds(db: dict, overlays: Iterable[Any], values: dict, tasks: Iterable[Any]) -> dict[str, dict]:
    """Per Trace id, the world its Task starts in: the shared db with that Task's overlay merged.

    The Starting-state pin, read the way `call_starting_states` reads it and by the same merge, so a
    row a Run wrote before anything read it has a before value that is the Task's own and not the
    shared world's (D215 rule 1). A Task with no overlay contributes nothing and its Runs fall back
    to the shared db, which is what they would have been pinned to anyway.
    """
    by_task = {overlay.task_id: compile_env.merge_overlays(db, [overlay], values)
               for overlay in overlays}
    return {run_id: by_task[task.id] for task in tasks if task.id in by_task
            for run_id in task.run_ids}


__all__ = ["HOLDOUT_ANSWERS_FILE", "HeldOutRuns", "REPLAYS_FILE", "REPLAY_AGREED", "REPLAY_LESSON_HEAD", "SHAPES_SHOWN", "after_write_calls",
           "callers_by_tool", "evidence_calls", "failure_shape", "failure_shapes", "holdout_answers",
           "is_evidence_call",
           "replay_difference", "replay_failures", "replay_failures_of", "replay_lesson", "trace_worlds"]
