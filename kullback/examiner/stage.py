"""The derive_verifier stage body, lifted out of build.py with its helpers, run against a small context in
place of the pipeline's StageContext, so the same bytes land in verifiers/, task_status.json,
references.json and constraints_check.json (D130).

The stage used three things of its context: `workdir`, `seed_runs` (the anchor, D81) and
`record_gate` (the ledger); `ExamContext` is those three. What the stage reads is the Builder's
store filtered by `DERIVE_INPUTS`: the Tasks, the mined signatures, the Constraints, the canon
rules, the replays, the re-rolls, the Intents, the Simulated user's rules, the Traces, the assisted
tools and the replay fidelity attributed per tool and per Task (D171, counts and tool names, the
same kind of summary `assisted_tools` already was and never a body). The tool bodies, the Starting
state, the schema and the Environment are not on the
list and `inputs_from` refuses a store that names them: the Examiner never reads what the Builder
compiled (D123). The one Run the old stage executed, check 6's loophole probe, reads bodies and so
stays in the Builder as `build.probe_runner(plan)`; it arrives here as the `run_probe` callable, the
Runner as a tool of both agents (D120).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback import sampling
from kullback.examiner import derive as verifier_mod
from kullback.examiner import judge as judge_mod
from kullback.examiner import lifecycle
from kullback.examiner import reference as reference_mod
from kullback.examiner import variants as variants_mod
from kullback.gates import artifacts, fidelity, loosening, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates import stages as stage_gates
from kullback.gates.ledger import GateLedger
from kullback.runner import budget, parallel
from kullback.runner.canon import rules_of
from kullback.runner.records import (
    GateResult,
    Intent,
    Task,
    Verifier,
    apply_intent,
    as_dict,
    content_hash,
    read_json,
    write_json,
)

DERIVE_INPUTS = ("tasks", "sigs", "constraints", "canon_rules", "replays", "rerolls", "intents", "user_rules",
                 "traces", "assisted_tools", "tool_fidelity")
# The inputs the derivation reads without a default: a store short of any of them is a build that
# stopped before the stage that releases it, and the Examiner has nothing to derive from rather
# than a KeyError halfway through (`missing_inputs`).
REQUIRED_INPUTS = ("tasks", "sigs", "constraints")
FORBIDDEN_INPUTS = ("bodies", "db", "schema", "environment", "overlays", "synthetic_rows", "policy_text",
                    "lessons_applied", "lessons_set_aside")
STAGE = "derive_verifier"
# The per-Task cache under the workdir (D163). Bumped when the entry's shape changes, so an old entry
# is a miss rather than a row read with the wrong meaning.
CACHE_DIR = ("examiner", "cache")
CACHE_FORMAT = 9  # D218: every status row carries its second-path record with a reason
# and the shapes dropped for rejecting their own Reference, D189's second-path batches and D198's
# reason per check with no input, the reference record carries D193's second pass over a residue and
# what deriving a Verifier per survivor settled (D198), the second-path row carries D199's
# synthesised paths, what they were rewritten from and whether the Task's path is single by
# structure, and the status row names the Reference's own Run ids, which is what says whether a
# Verifier on disk was derived from the Reference the Task holds (D208)
# The modules a Task's derivation runs through, hashed into every key: an edit to any of them is a
# different derivation and must not be served a stale entry (the Builder's stages hash the same way,
# build.py's `_version`).
CODE_MODULES = (verifier_mod, reference_mod, judge_mod, verifier_suite, variants_mod)


class ExamContext:
    """What the derivation is given beside its inputs: where to write, the anchor to seed from, the ledger."""

    def __init__(self, workdir: Path, ledger: GateLedger, anchor: Any = None, stage: str = STAGE):
        self.workdir = Path(workdir)
        self.ledger = ledger
        self.anchor = anchor
        self.stage = stage
        self.recorded: list[GateResult] = []

    def seed_runs(self, task_id: str, run_ids: Iterable[str]) -> list[str]:
        """The Task's Runs minus its anchor when one was chosen, every Run otherwise (D81)."""
        if self.anchor is None:
            return list(run_ids)
        return list(self.anchor.seed_runs(task_id, run_ids))

    def record_gate(self, result: GateResult) -> GateResult:
        """One ruling into gates.json under this stage's name, remembered for the tool result."""
        self.recorded.append(result)
        return self.ledger.record(self.stage, result)


def missing_inputs(store: dict) -> list[str]:
    """The derivation inputs a store does not hold, in `REQUIRED_INPUTS` order; empty when it can derive.

    A `--target` naming a stage before compile_policy leaves the store without the artifacts the
    derivation reads with no default, and the Examiner used to open on it and fail on the first one
    it reached. The driver asks this before the beat and ends the round on the build instead.
    """
    return [name for name in REQUIRED_INPUTS if name not in store]


def inputs_from(store: dict) -> dict:
    """The derivation's inputs out of a store; a store naming a forbidden artifact is refused."""
    forbidden = [name for name in FORBIDDEN_INPUTS if name in store]
    if forbidden:
        raise ValueError(f"the Examiner never reads {', '.join(forbidden)}; hand it the derivation inputs only (D123)")
    return {name: store[name] for name in DERIVE_INPUTS if name in store}


def seed_ids(ctx: Any, task: Task) -> set[str]:
    """The Task's Runs the derivation may use: minus the anchor when one was chosen (D81)."""
    return set(ctx.seed_runs(task.id, task.run_ids))


def request_text(task: Task, intents: dict, traces: dict) -> str:
    """What the user asked, for the judge: the grounded Intent, else the Task's name, else the first user turn."""
    record = intents.get(task.id)
    if record is not None and record.grounded and record.text:
        return record.text
    if task.intent or task.name:
        return task.intent or task.name or ""
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        for turn in (trace.turns if trace else []):
            if turn.role == "user" and turn.content:
                return turn.content
    return ""


def grounded_phrases(intent: Optional[Intent]) -> list[str]:
    """The noun phrases of a grounded Intent, in order, without repeats, for the judge's `intent` tool.

    An ungrounded Intent has no phrase the Runs evidence (D47), so it offers none: the judge would
    otherwise read a phrase the intent gate has already refused as if the Runs stood behind it.
    """
    if intent is None or not intent.grounded:
        return []
    out: list[str] = []
    for span in intent.spans:
        if span.phrase and span.phrase not in out:
            out.append(span.phrase)
    return out


# --- the moved helpers ---------------------------------------------------------

def final_constraints(ctx, inputs: dict, seed_replays: dict, write_tools: set, read_tools: set,
                      fn: Any) -> tuple[list, list]:
    """The constraints a Verifier may check, and the ones the recordings demoted (D76).

    Every compiled constraint is run over the confirmed recordings corpus-wide first: the recordings
    are the frontier under the customer's real policy, so a rule they mostly break is a miscompiled
    rule, and it becomes a residual, reported in the setup review and checked in no Verdict. The
    compile_policy gate is recorded again over the final list, after the recordings have had their
    say, which is the order the reference check has to run in: on one early build 15 of 39
    compiled rules fired on confirmed recordings and poisoned every Verifier.
    """
    compiled = [c for c in inputs["constraints"] if c.compiled or c.judge_atom]
    paths = [r["path"] for rows in seed_replays.values() for r in rows]
    # Every rule with code is asked, so a rule that gates nothing still says how it fares against the
    # frontier and a rule that judged nothing is not read as a rule that held.
    rates = reference_mod.constraint_rates(list(inputs["constraints"]), paths, write_tools, fn, read_tools)
    constraints, demoted = reference_mod.demote(compiled, rates)
    by_id = {c.id: c for c in compiled}
    residual = [by_id[row["id"]].model_copy(update={"compiled": False, "judge_atom": False,
                                                    "residual_reason": row["reason"]})
                for row in demoted]
    untouched = [c for c in inputs["constraints"] if not (c.compiled or c.judge_atom)]
    final = constraints + residual + untouched
    ctx.record_gate(artifacts.policy_gate(final))
    write_json(ctx.workdir / "constraints_check.json",
               {"rates": rates, "demoted": demoted, "constraints": [as_dict(c) for c in final]})
    return constraints, demoted


# The false-rejection pool (D133) asks how strict the required atoms are over Runs that did the Task,
# and a Run that did not do it is not evidence about that. These are the two ways to miss.
WROTE_NOTHING = "did not reach the Reference's End state: it wrote nothing"
WROTE_OTHERWISE = "did not reach the Reference's End state: it wrote something else"


def not_at_reference(confirmation: Any, pool_runs: Iterable[tuple[str, str]], write_tools: set,
                     fn: Callable) -> dict[str, str]:
    """The pool Runs whose settled End state is not the Reference's, and why each is left out (D133).

    The D133 pool counts a held-out Run as legitimate on its success termination alone, so a Run that
    answered and stopped is in it whatever the Task asked for. On two corpora every write rejection
    was a Run like that: of 124 on the first, 85 wrote nothing, 8 used another tool and 31 made some
    of the writes; on the second, 43 of 43 wrote nothing. Counted as false rejections they say the
    Verifier is over-strict, when what the Verifier did was catch a Run that did not do the Task, and
    the number gates a Verifier for catching it.

    So a Run enters the pool when its settled End state (D182: the writes in order, with the values
    and what the tool answered) is the Reference's. One comparison covers both halves of the rule: a
    Reference that wrote nothing has the empty settled state, and so has every Run that wrote nothing.
    What false rejection then measures is the atoms that are not writes, the stated facts, the
    questions and the caps, against Runs that did the Task, which is what the number is for.
    """
    if not confirmation.references:
        return {}
    target = confirmation.references[0].settled
    known = {rec.run_id: rec.settled for rec in confirmation.recordings}
    out: dict[str, str] = {}
    for run_id, path in pool_runs:
        settled = known.get(run_id)
        if settled is None:
            settled = reference_mod.settled_state(path, write_tools, fn)
        if settled != target:
            out[run_id] = WROTE_NOTHING if not settled else WROTE_OTHERWISE
    return out


def pool_user_ends(task_id: str, rerolls: dict) -> dict[str, str]:
    """How the Simulated user ended each held-out Run of this Task, by the four kinds (D210, D133).

    A Run that ended `scenario_exhausted` or `gave_up` did not stop because the Task was done: its
    user ran out of what it could say, or the Run ran out of turns. Counting such a Run as a false
    rejection says the Verifier is over-strict when what happened was that the Run never finished,
    which is the same reading D133 already fixed for Runs that reached another End state. The kinds
    are named per Run here so a reading over the pool can leave them out; nothing is filtered on
    them yet, because the pool's own rule is the settled End state and this is a second opinion.
    """
    return {str(row["run_id"]): str(row["user_end"])
            for row in rerolls.get(task_id) or []
            if row.get("run_id") and row.get("user_end")}


def pool_runs_of(task_id: str, replays: dict, rerolls: dict) -> list[tuple[str, str]]:
    """The Runs D133's pool holds for one Task, as (run id, path): the confirmed replays, the anchor
    among them (D81), and the re-rolls of any round that reached a success termination.

    The same two sources `loosening.finished_run_ids` reads, off the same rows, so what is filtered
    here is what the gate would otherwise count.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _trace_id, row in sorted((replays.get(task_id) or {}).items()):
        if row.get("confirmed") and row.get("run_id") and row.get("path") and row["run_id"] not in seen:
            seen.add(row["run_id"])
            out.append((str(row["run_id"]), str(row["path"])))
    for row in rerolls.get(task_id) or []:
        if (row.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS \
                and row.get("run_id") and row.get("path") and row["run_id"] not in seen:
            seen.add(row["run_id"])
            out.append((str(row["run_id"]), str(row["path"])))
    return out


# --- the second path (D189) ------------------------------------------------------

# A Task whose Reference stands alone leaves check 5 with nothing to score: `suite_for` has no
# second Run to pass and the suite reports it not run, never passed (D173). The stage buys more
# frontier Runs of that Task, a batch at a time, and stops as soon as one of them reaches the
# Reference's End state, which is the second path the check wants. The cap is what bounds the cost:
# at most `SECOND_PATH_BATCHES` batches of `SECOND_PATH_RUNS` Runs per Task per call, and a Task
# that exhausts it keeps the check not run and says so on its row.
SECOND_PATH_REASON = "second_path"
SECOND_PATH_BATCHES = 3
SECOND_PATH_RUNS = 3  # the Runs one batch buys, the count the re-roll stage samples per Task (D112)
SECOND_PATH_PREFIX = "second-path"
PROBE_KIND = "probe_slot"  # D212: the kind the probe budget's order is keyed under, per Task id
NO_REROLL_RUNNER = "no re-roll Runner in this session"
CEILING_REACHED = "the budget ceiling was reached"
# Where the Examiner's own re-roll rows live, the file `ExaminerPlan` reads back and merges into the
# frontier pool (D133). The rows this stage writes carry their reason, so a re-run of the derivation
# reads its own batches back and buys nothing, and the rows another verb wrote are left to it.
EXTRA_REROLLS = ("examiner", "rerolls.json")
_EXTRA_REROLLS_LOCK = threading.Lock()


def extra_rerolls_path(workdir: Path) -> Path:
    return Path(workdir).joinpath(*EXTRA_REROLLS)


def second_path_rows(workdir: Path, task_id: str) -> list[dict]:
    """The extra batches this stage already bought for one Task, off the Examiner's re-roll file.

    Keyed by the reason the way D177 keys a Task's re-rolls by what they were sampled under: a
    second derivation over the same Task reads these rows back rather than buying the batch again.
    """
    try:
        rows = read_json(extra_rerolls_path(workdir), {}) or {}
    except (OSError, ValueError):
        return []
    return [dict(row) for row in (rows.get(task_id) or [])
            if isinstance(row, dict) and row.get("reason") == SECOND_PATH_REASON
            and row.get("path") and Path(row["path"]).is_file()]


def next_batch(workdir: Path, task_id: str) -> int:
    """The attempt index the next batch for this Task takes: one above the highest already recorded.

    D212: a batch count that must grow grows by adding higher attempt indexes and never by
    reshuffling the ones already taken. The counter used to restart at one on every call, so a
    second search over the same Task in the same round wrote its Runs over the first search's files
    under the same names and the batch the row counted was not the batch on disk.
    """
    rows = second_path_rows(workdir, task_id)
    return max((int(row.get("batch") or 0) for row in rows), default=0) + 1


def record_second_path(workdir: Path, task_id: str, rows: Iterable[dict], batch: int) -> list[dict]:
    """One batch's Runs into the Examiner's own re-roll file, tagged with the batch and the reason.

    They join the Task's Runs like any other re-roll: the plan merges this file into `rerolls`, so
    the frontier pool, the loosening gate and the next derivation all see them.
    """
    tagged = [{**dict(row), "reason": SECOND_PATH_REASON, "batch": int(batch)} for row in rows]
    with _EXTRA_REROLLS_LOCK:
        path = extra_rerolls_path(workdir)
        try:
            every = read_json(path, {}) or {}
        except (OSError, ValueError):
            every = {}
        seen = {row.get("run_id") for row in every.get(task_id) or []}
        every.setdefault(task_id, []).extend(row for row in tagged if row.get("run_id") not in seen)
        write_json(path, every)
    return tagged


def finished_recordings(rows: Iterable[dict], *, write_tools: set, fn: Callable, atoms: Any) -> list:
    """The Runs of some re-roll rows that reached a success termination, as Recordings."""
    return [reference_mod.load(row["path"], reference_mod.REROLL, run_id=row["run_id"],
                               write_tools=write_tools, fn=fn, atoms=atoms)
            for row in rows
            if row.get("path")
            and (row.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS]


def reaches_reference(reference: Any, candidate: Any) -> bool:
    """Does this Run reach the Reference's End state? The D111 grouping rule, asked of two Runs.

    `reference.group` is the definition and it is not restated here: two Runs the rule would have
    put in one group are one End state, which is the same writes with the same answers whatever
    order a list argument came in (D182), and a Run that reaches it by other reads or other wording
    is the second path check 5 scores (D46).
    """
    return len(reference_mod.group([reference, candidate])) == 1


def merge_second_path(confirmation: Any, candidates: Iterable[Any]) -> int:
    """The extra batches' Runs onto a settled Confirmation; how many of them reach the Reference.

    The Reference is what the D111 rule settled over the Task's evidence. These Runs were bought
    after it and for one purpose, to find a second way to the End state it settled on, so they
    corroborate it and never re-open it. One that landed somewhere else did not do the Task and is
    recorded in the same words the false-rejection pool uses for such a Run (D133), so it is counted
    and left out rather than dropped.
    """
    if not confirmation.references:
        return 0
    reference = confirmation.references[0]
    joined = 0
    for rec in candidates:
        if any(seen.run_id == rec.run_id for seen in confirmation.recordings):
            continue
        confirmation.recordings.append(rec)
        if reaches_reference(reference, rec):
            confirmation.references.append(rec)
            joined += 1
        else:
            confirmation.failed[rec.run_id] = WROTE_NOTHING if not rec.settled else WROTE_OTHERWISE
    return joined


def second_path_row(batches: int, runs: int, *, found: bool, cap: int = SECOND_PATH_BATCHES,
                    reason: str = "", synthesised: int = 0, kinds: Iterable[str] = (),
                    tried: int = 0, kept: int = 0, structural: bool = False) -> dict:
    """What the search for a second path cost this Task and what it bought, for the row and the counts."""
    if found:
        why = ""
    elif structural:
        why = SINGLE_PATH_BY_STRUCTURE
    else:
        why = reason or (f"no second path in {batches} batches" if batches else NO_REROLL_RUNNER)
    return {"batches": int(batches), "runs": int(runs), "found": bool(found),
            "exhausted": bool(not found and batches >= cap), "reason": why,
            "synthesised": int(synthesised), "synth_kinds": sorted(set(kinds)),
            "synth_tried": int(tried), "synth_kept": int(kept), "structural": bool(structural)}


# D218 rule 2: the two fields every status row carries so a reader can see it moved. task_status.json
# is the live file and goes on being written after a round has closed its own table; without a round
# and a time on the row, a row that moved reads exactly like a row the round wrote, and 64 rows of one
# live build disagreed with the frozen ruling with nothing on either to say which had moved.
STAMPS = ("round", "updated_at")


def stamped(rows: dict, prior: Optional[dict], round_number: int, *, now: Optional[float] = None) -> dict:
    """Every status row with the round it was last written in and when it was written.

    A row whose content is what the round before left is not a row this round wrote: it keeps the
    stamp it had, so the round on a row is the round that last changed it and not the round that last
    read it. That is what makes the stamp worth reading; stamping every row every round would say
    only that a derivation ran.
    """
    at = float(now if now is not None else time.time())
    out: dict = {}
    for task_id, row in (rows or {}).items():
        body = {key: value for key, value in (row or {}).items() if key not in STAMPS}
        was = (prior or {}).get(task_id)
        before = {key: value for key, value in (was or {}).items() if key not in STAMPS} if was else None
        if before == body and was is not None and all(key in was for key in STAMPS):
            out[task_id] = {**body, "round": was["round"], "updated_at": was["updated_at"]}
        else:
            out[task_id] = {**body, "round": int(round_number), "updated_at": at}
    return out


class MissingReason(ValueError):
    """A search or a check ended without a ruling and recorded no reason for it (D218 rule 3)."""


def with_reason(record: Any, *, what: str, task_id: str) -> dict:
    """One search's record, refused where it ended without a ruling and without a reason.

    Every search the harness runs can come back with nothing: no second path was bought, no rewrite
    reached the End state, no survivor could be derived from, no atom could be relaxed. Each of those
    is an answer and each has to say which one it is, because the round's table reports the reason
    and a blank one cannot be told from a search that never ran. Two Tasks of one live build carried
    an empty record where a reason belonged, and no reading of the artifacts on disk could say what
    had happened to them. So the writer raises rather than writing the blank.
    """
    if not isinstance(record, dict):
        raise MissingReason(f"task {task_id}: {what} wrote {type(record).__name__} where a record with "
                            "a reason belongs (D218 rule 3)")
    if not record.get("found") and not str(record.get("reason") or "").strip():
        raise MissingReason(f"task {task_id}: {what} found nothing and recorded no reason for it "
                            "(D218 rule 3)")
    return record


def _second_path(row: Any) -> dict:
    """One status row's second-path record, with zeroes for a row that carries none (an older entry)."""
    found = (row or {}).get("second_path") if isinstance(row, dict) else None
    if not isinstance(found, dict):
        return second_path_row(0, 0, found=False)
    return {"batches": int(found.get("batches") or 0), "runs": int(found.get("runs") or 0),
            "found": bool(found.get("found")), "exhausted": bool(found.get("exhausted")),
            "reason": str(found.get("reason") or ""),
            "synthesised": int(found.get("synthesised") or 0),
            "synth_kinds": list(found.get("synth_kinds") or ()),
            "synth_tried": int(found.get("synth_tried") or 0),
            "synth_kept": int(found.get("synth_kept") or 0),
            "structural": bool(found.get("structural"))}


def second_path_search(task_id: str, confirmation: Any, *, workdir: Path, run_rerolls: Any,
                       round_number: int, write_tools: set, fn: Callable, atoms: Any,
                       cap: int = SECOND_PATH_BATCHES,
                       count: int = SECOND_PATH_RUNS) -> tuple[dict, list, bool]:
    """Batches of fresh Runs until one reaches the Reference's End state, or the cap (D189).

    Bounded per Task by the cap and per round by the Tasks that have one Reference and no more: at
    most `cap` batches of `count` Runs each, about ten model calls a Run. A batch that finds the
    second path is the last one bought; a Task that exhausts the cap keeps check 5 not run and its
    row says how many batches went into saying so. The Runs of every batch are recorded whatever
    they reached, so the round can be read for what the search cost.
    """
    bought, runs, ceiling = 0, 0, False
    attempt = next_batch(workdir, task_id)
    recordings: list = []
    while run_rerolls is not None and bought < cap and len(confirmation.references) < 2:
        prefix = f"{SECOND_PATH_PREFIX}-r{round_number}-b{attempt}"
        try:
            rows = [dict(row) for row in run_rerolls(task_id, count, prefix) or []]
        except budget.BudgetExceeded:
            ceiling = True
            break
        bought += 1
        runs += len(rows)
        record_second_path(workdir, task_id, rows, attempt)
        attempt += 1
        fresh = finished_recordings(rows, write_tools=write_tools, fn=fn, atoms=atoms)
        recordings.extend(fresh)
        merge_second_path(confirmation, fresh)
    found = len(confirmation.references) > 1
    row = second_path_row(bought, runs, found=found, cap=cap,
                          reason=CEILING_REACHED if ceiling and not found else "")
    return row, recordings, ceiling


# --- the synthesised second path (D199) ------------------------------------------

# When the search above is spent, the Reference Run is asked for a second path instead of the model.
# variants.py rewrites its call path three ways, each of which leaves the world where it was, and
# each rewrite is replayed through the Environment from the Task's Starting state and kept only if
# it reaches the Reference's End state under the same D111 rule a bought Run is held to. A Run with
# no independent adjacent pair, no read to insert and no read to drop offers nothing, and that Task's
# path is single by structure, which is an honest end state and is counted as one.
SYNTH_PREFIX = "synth-path"
SINGLE_PATH_BY_STRUCTURE = "single_path_by_structure"
SYNTH_KEPT_NONE = "no rewrite of the Reference's call path reaches its End state"


def synth_second_path(task_id: str, confirmation: Any, *, run_variant: Any, round_number: int,
                      write_tools: set, fn: Callable, atoms: Any,
                      limit: int = variants_mod.MAX_VARIANTS) -> tuple[dict, list]:
    """Rewrites of the Reference's own call path, replayed and kept where they land in the same place.

    Answers with what the synthesis found, for the row and the counts, and the Recordings of the
    rewrites that were kept. They join the Confirmation the way a bought second path does, so check 5
    scores the Verifier on them and on nothing else that is new.
    """
    reference = confirmation.references[0]
    try:
        run = verifier_suite.as_run(reference.path)
    except (OSError, ValueError, TypeError):
        return {"tried": 0, "kinds": [], "structural": False, "reason": SYNTH_KEPT_NONE}, []
    plans = variants_mod.variants(variants_mod.call_results(run), write_tools, limit=limit)
    if not plans:
        return {"tried": 0, "kinds": [], "structural": True,
                "reason": SINGLE_PATH_BY_STRUCTURE}, []
    spoken = variants_mod.transcript(run)
    seen = {variants_mod.signature(variants_mod.call_results(run))}
    tried, kept, kinds = 0, [], []
    for index, plan in enumerate(plans, 1):
        signature = variants_mod.signature(plan["calls"])
        if signature in seen:
            continue
        seen.add(signature)
        tried += 1
        try:
            row = run_variant(task_id, plan["calls"],
                              f"{SYNTH_PREFIX}-r{round_number}-{task_id}-{index}", spoken)
        except Exception:
            # A rewrite the Environment cannot run is a rewrite that is not kept. It buys nothing
            # and it is not the round's business to fall over on one, the way a crashed replay is
            # scored as a replay that did not confirm rather than as a build that failed.
            continue
        if not row or not row.get("path"):
            continue
        try:
            rec = reference_mod.load(row["path"], reference_mod.REROLL, run_id=row["run_id"],
                                     write_tools=write_tools, fn=fn, atoms=atoms)
        except (OSError, ValueError, TypeError):
            continue
        if rec.violated or not reaches_reference(reference, rec):
            continue
        kept.append(rec)
        kinds.append(plan["kind"])
    return {"tried": tried, "kinds": kinds, "structural": False,
            "reason": "" if kept else SYNTH_KEPT_NONE}, kept


def pool_at_reference(pool_runs: Iterable[tuple[str, str]], confirmation: Any,
                      left_out: dict) -> list[str]:
    """The held-out Runs that reach the Reference and are not References themselves (D133).

    A Task whose path is single by structure has no second path of its own to show, and this is the
    second path the evidence already holds: a Run nobody derived from that landed on the Reference's
    settled End state. `left_out` is the same pool with the Runs that landed elsewhere named, so
    what is left here is the pool minus those and minus the References.
    """
    references = {r.run_id for r in confirmation.references}
    return sorted({run_id for run_id, _path in pool_runs
                   if run_id not in left_out and run_id not in references})


def task_fidelity(tool_fidelity: Any, task_id: str) -> dict:
    """One Task's replay fidelity over its own recorded calls, per tool (D171).

    `tool_fidelity` is the compile_tools artifact: `tools` is the corpus ruling per tool, `tasks` is
    per Task per tool how many of that Task's own recorded calls the kept body replayed and how many
    differed. A Task the artifact does not name made no evidence call of any tool.
    """
    return dict(((tool_fidelity or {}).get("tasks") or {}).get(task_id) or {})


def fidelity_fields(row: dict) -> dict:
    """The two count maps every status row carries (D171).

    `tool_calls_replayed` names every tool this Task made a recorded call of, whatever the outcome;
    `tool_calls_differing` names only the tools some own call parted from, so a row with an empty
    map is a Task whose every own call replayed exactly.
    """
    return {"tool_calls_replayed": {tool: int(counts.get("replayed") or 0)
                                    for tool, counts in sorted(row.items())},
            "tool_calls_differing": {tool: int(counts.get("differing") or 0)
                                     for tool, counts in sorted(row.items()) if counts.get("differing")}}


def blocking_tools(row: dict) -> list[str]:
    """The tools one of this Task's own recorded calls parts from: the only tools that block it (D171)."""
    return sorted(tool for tool, counts in row.items() if int(counts.get("differing") or 0))


def _differing_call_text(row: dict, tools: Iterable[str]) -> str:
    """The differing calls themselves, in the corpus gate's own words, so the repair has a target."""
    named = [f"{tool}: {reason}" for tool in tools
             for reason in (row.get(tool, {}).get("reasons") or ["a recorded call replays differently"])[:1]]
    return "; ".join(named)


def no_reference_status(ctx, task: Task, confirmation: Any, *, seed_replays: list, replays: dict,
                        rerolls: dict, traces: dict, assisted_tools: set, fidelity_row: dict) -> dict:
    """Why this Task has no Reference, in the words the setup review needs (D49): a Task with none is not verdicted.

    D171 narrows which tool the row names. A tool is assisted when it misses one of the corpus's
    recorded calls, and the wording here used to hand that verdict to every Task whose seed Traces
    call the tool: on one live build a body replayed 239 of its 240 recorded calls and stayed
    assisted, and of the 64 rows that named an assisted tool as their blocker, 51 make no call any
    assisted body answers differently. That sent the repair at bodies that answer those Tasks
    correctly and hid the reason they actually failed. Only a tool one of this Task's own recorded
    calls parts from is named now, with that call quoted; the corpus-level fact stays on the row as
    `assisted_tools`, so nothing is lost, and the tools that block are `blocking_tools`.

    This is sound because the body is exercised on exactly the calls this Task's Run makes: the
    per-call rows come from replaying the kept body against those very calls and comparing under the
    same ruling the corpus gate uses. What confirms the Reference is still the replay reaching the
    End state, and above it the D79 suite and the held-out check still judge the Verifier that is
    derived. "Confirmed" therefore means right on this Task's calls, not right on every call in the
    corpus, and the suite above it is what keeps that honest.
    """
    reason = confirmation.reason or "no Run to confirm"
    if not seed_replays:
        reason = ("no seed Trace was replayed" if not (replays.get(task.id) or {}) else
                  fidelity.unconfirmed_reason({t: r for t, r in replays[task.id].items()
                                               if t in seed_ids(ctx, task)}))
    assisted_used = sorted({c.name for tid in seed_ids(ctx, task) if tid in traces
                            for c in traces[tid].tool_calls if c.name in assisted_tools})
    blocking = blocking_tools(fidelity_row)
    if blocking:
        reason = (f"this Task's own recorded calls of {', '.join(blocking)} do not replay "
                  f"({_differing_call_text(fidelity_row, blocking)}) (D49, D171); {reason}")
    return {"reference_confirmed": False, "verifier_passed": False, "reason": reason,
            "recordings": len(seed_replays), "rerolls": len(rerolls.get(task.id, [])),
            "judged": confirmation.judged, "assisted_tools": assisted_used,
            "blocking_tools": blocking, **fidelity_fields(fidelity_row)}


def derive_for(task_for: Task, confirmation: Any, *, canon_rules: Any, write_tools: set, constraints: list,
               intent: Any = None, verifier_version: str = "1") -> Verifier:
    """The derivation over the References: the first is the Reference, the rest its re-runs.

    The Task's Intent record goes in beside the Task itself: which facts the answer must state is
    settled by what the request asked about, and the record carries the grounded span behind every
    phrase of it, which the Intent line alone does not.
    """
    paths = [r.path for r in confirmation.references]
    return verifier_mod.derive_verifier(task_for, paths[0], paths[1:], canon_rules,
                                        write_tools=write_tools, constraints=constraints,
                                        successful_run_ids=[r.run_id for r in confirmation.references],
                                        intent=intent, writes_elsewhere=wrote_outside(confirmation),
                                        verifier_version=verifier_version)


def wrote_outside(confirmation: Any) -> bool:
    """Did a Run of this Task that is not one of its References write something?

    The D111 rule saw every one of them, and a Reference group that wrote nothing beside a Run that
    did is the Task saying it has more than one path. What the derivation does with that is its own
    rule; this only reports the fact, off the End states the rule already grouped on.
    """
    references = {r.run_id for r in confirmation.references}
    return any(r.end_state and r.end_state != reference_mod.ANSWERED
               for r in confirmation.recordings if r.run_id not in references)


def suite_for(task_for: Task, verifier: Verifier, paths: list, *, canon_rules: Any, write_tools: set,
              user_rules: dict, rules_trace: Optional[str], probe_model: Any, run_probe: Any,
              may_probe: bool, intent: Any = None) -> list[GateResult]:
    """The whole D79 suite over one Verifier: the wrong Run, the second path, the leak, the probe (D79, D119).

    The Intent record goes in beside its line so the leak check reads it as an audit of the D196
    strip and names the column of anything the strip missed.
    """
    return verifier_suite.validate_verifier(
        verifier, paths[0], canon=canon_rules, write_tools=write_tools, seed_runs=paths[1:],
        wrong_run=verifier_suite.wrong_run(verifier, paths[0], canon_rules),
        alt_path_run=paths[1] if len(paths) > 1 else None,
        intent_text=task_for.intent, user_rules=user_rules.get(rules_trace), intent=intent,
        model=probe_model if may_probe else None, run_probe=run_probe)


# --- the residue, settled by deriving a Verifier per survivor (D198) --------------

# How many surviving End states a Task may derive a Verifier from. Each one is a derivation and a
# code-only suite, so the cost is bounded by this and by nothing else; a Task with more states than
# this takes the first four in recording order and says so on its row.
SURVIVOR_CAP = 4


def survivor_pool(trial: Any, pool_runs: Iterable[tuple[str, str]], write_tools: set,
                  fn: Callable) -> tuple[list, set[str]]:
    """The held-out Runs D133 measures against one survivor, and their ids.

    The pool is the Task's frontier Runs less the ones that did not reach this survivor's End state,
    which is the same subtraction the released Verifier's ruling makes (D173, D194); a survivor is
    therefore measured against the Runs that did what it did, exactly as the Reference would be.
    """
    left_out = not_at_reference(trial, pool_runs, write_tools, fn)
    legitimate = {run_id for run_id, _ in pool_runs} - set(left_out) - set(trial.failed)
    runs = [verifier_suite.as_run(path) for run_id, path in pool_runs if run_id in legitimate]
    return runs, legitimate


def survivor_row(label: str, group: dict, results: dict, skipped: list[str], held: dict) -> dict:
    """How one survivor's Verifier fared, as the row the choice is made on and the record keeps.

    `passed` is no check having failed on its merits. A check with no input is not a verdict on the
    Verifier (D173), and the loophole probe is bought for the Task and not for each candidate, so a
    check nobody could run neither passes a survivor nor fails it; it does cost a point of
    `checks_passed`, because a state more Runs reached has a second path to score and that is better
    evidence about it, not worse.
    """
    failed = sorted(name for name, ok in results.items() if not ok and name not in skipped)
    return {"label": str(label), "runs": len(group.get("runs") or []),
            "checks_passed": sum(1 for ok in results.values() if ok),
            "checks_failed": failed, "checks_not_run": sorted(skipped),
            "passed": not failed,
            "false_rejection": held.get("fraction"), "held_out": int(held.get("held_out") or 0)}


def survivor_order(row: dict) -> tuple:
    """The order the choice reads: most checks passed first, then the lowest false rejection.

    A survivor with nothing held out has no false-rejection measurement, and a measurement that was
    not made is not evidence against it, so it sorts where a Verifier that rejects nothing sorts
    (D194 leaves such a Task to its other checks for the same reason).
    """
    fraction = row.get("false_rejection")
    return (-int(row["checks_passed"]), float(fraction) if fraction is not None else 0.0)


def choose_survivor(rows: list[dict], order: Iterable[str]) -> tuple[Optional[dict], str, str]:
    """Which survivor's Verifier the harness stands behind, in one reading of the rows (D198).

    Most checks passed decides, then the lowest false rejection. Survivors whose Verifiers are equal
    on both are equal under every check the harness has: nothing it can ask tells those End states
    apart, so the Task takes the first by recording order and says that is why, rather than reading
    an order the rows were sorted in as a finding. When no survivor's Verifier stands up the Task
    keeps no Reference, and the reason names the best of them, which is where a person starts.
    """
    ranked = sorted(rows, key=survivor_order)
    standing = [row for row in ranked if row["passed"]]
    if not standing:
        best = ranked[0]
        return None, reference_mod.SURVIVORS_ALL_FAIL, (
            f"{reference_mod.SURVIVORS_ALL_FAIL}: a Verifier was derived from each of {len(rows)} "
            f"surviving End states and none of them passed the suite; the best is {best['label']}, "
            f"which passed {best['checks_passed']} checks and failed {', '.join(best['checks_failed'])}")
    best = standing[0]
    tied = [row for row in standing if survivor_order(row) == survivor_order(best)]
    if len(tied) > 1:
        places = list(order)
        first = min(tied, key=lambda row: places.index(row["label"]))
        return first, reference_mod.SURVIVORS_EQUIVALENT, (
            f"{reference_mod.SURVIVORS_EQUIVALENT}: {', '.join(row['label'] for row in tied)} each "
            f"passed {best['checks_passed']} checks with the same false rejection, so they are "
            f"equivalent under every check the harness has; the Task takes {first['label']}, the "
            "first by recording order")
    return best, reference_mod.SURVIVOR_CHOSEN, (
        f"{reference_mod.SURVIVOR_CHOSEN}: {best['label']} of {len(rows)} surviving End states, "
        f"its Verifier passing {best['checks_passed']} checks with false rejection "
        f"{_fraction_text(best)}")


def settle_residue(task: Task, confirmation: Any, *, canon_rules: Any, write_tools: set,
                   constraints: list, intents: dict, user_rules: dict, fn: Callable,
                   pool_runs: Iterable[tuple[str, str]] = (), verifier_version: str = "1",
                   cap: int = SURVIVOR_CAP) -> None:
    """Derive one Verifier per surviving End state and let the suite and the held-out pool choose (D198).

    The judge is fail-only and a residue is the question it has already declined: three decisions
    measured it over one and it settled none of them, so asking a third time is asking the same
    question again. What the harness has instead is the check it applies to every Reference it does
    keep. So each survivor is derived from as if it were the Reference, by the same path and the same
    atoms, and put through the D79 suite and D133's false rejection over the Runs that reached it.
    The survivor whose Verifier stands up best is the Reference; survivors whose Verifiers are equal
    on every check the harness has are equivalent under everything it can ask, so the Task takes the
    first by recording order rather than pretending a difference was found (D184's keep-by-beating
    idea, applied to References). When no survivor's Verifier stands up the Task keeps no Reference
    and the row names the best of them, which is where a person looks first.

    Nothing here buys a Run: the derivation is code, the suite's Runs are the Reference's own and the
    two it synthesizes, and the loophole probe is left to the Task's released Verifier. So a survivor
    is compared on the checks that can be run for all of them alike, and the winner still faces the
    whole suite, probe included, on the ordinary path afterwards.
    """
    survivors = list(confirmation.survivors)
    if len(survivors) < 2:
        return
    if len(survivors) > cap:
        confirmation.survivors_capped = True
        survivors = survivors[:cap]
    task_for = apply_intent(task, intents[task.id]) if task.id in intents else task
    pool_runs = list(pool_runs)
    rows: list[dict] = []
    by_label: dict[str, dict] = {}
    for group in survivors:
        members = list(group.get("members") or ())
        if not members:
            continue
        others = {rec.run_id for g in survivors if g is not group for rec in (g.get("members") or ())}
        trial = reference_mod.Confirmation(
            references=members, recordings=list(confirmation.recordings),
            failed={run_id: reason for run_id, reason in confirmation.failed.items()
                    if run_id not in others})
        verifier = derive_for(task_for, trial, canon_rules=canon_rules, write_tools=write_tools,
                              constraints=constraints, intent=intents.get(task.id),
                              verifier_version=verifier_version)
        paths = [r.path for r in members]
        rules_trace = next((r.trace_id for r in members if r.trace_id), None)
        gates = suite_for(task_for, verifier, paths, canon_rules=canon_rules, write_tools=write_tools,
                          user_rules=user_rules, rules_trace=rules_trace, probe_model=None,
                          run_probe=None, may_probe=False)
        results = verifier_suite.d79_results(gates)
        skipped = [verifier_suite.D79_STAGES[g.stage] for g in gates if g.metrics.get("skipped")]
        runs, legitimate = survivor_pool(trial, pool_runs, write_tools, fn)
        held = loosening.false_rejection(verifier, runs, legitimate, canon_rules, write_tools)
        row = survivor_row(group["label"], group, results, skipped, held)
        rows.append(row)
        by_label[row["label"]] = group
    if not rows:
        # D218 rule 3: a search that ends without a ruling writes its reason. Every surviving group
        # held no Run to derive from, so there was nothing to choose between; without this the Task
        # kept no Reference and no record said why, which reads on the round's table as a Task the
        # harness silently gave up on.
        confirmation.survivor_reason = reference_mod.SURVIVORS_EMPTY
        confirmation.reason = (f"{reference_mod.SURVIVORS_EMPTY}: {len(survivors)} End states survived "
                               "the judgement and none of them holds a Run to derive a Verifier from")
        return
    confirmation.residue_derived = True
    confirmation.survivor_scores = sorted(rows, key=survivor_order)
    best, why, said = choose_survivor(rows, [row["label"] for row in rows])
    confirmation.survivor_reason, confirmation.reason = why, said
    if best is None:
        return
    confirmation.survivor_chosen = best["label"]
    won = by_label[best["label"]]
    confirmation.references = list(won.get("members") or ())
    for group in survivors:
        if group is won:
            continue
        for rec in group.get("members") or ():
            confirmation.failed.setdefault(
                rec.run_id, f"survivor {group['label']} was not chosen: {confirmation.reason}")


def _fraction_text(row: dict) -> str:
    """A survivor's false rejection with its denominator, or the words for a pool that held nothing."""
    fraction = row.get("false_rejection")
    if fraction is None:
        return "not measured, nothing was held out"
    return f"{float(fraction):.2f} of {row['held_out']} held-out Runs"


def verifier_for(ctx, task: Task, confirmation: Any, *, canon_rules: Any, write_tools: set, constraints: list,
                 intents: dict, user_rules: dict, recordings: int, rerolls: int, probe: Any,
                 probe_model: Any, may_probe: bool, fidelity_row: Optional[dict] = None,
                 pool_runs: Iterable[tuple[str, str]] = (), fn: Optional[Callable] = None,
                 second_path: Optional[dict] = None, user_ends: Optional[dict] = None,
                 verifier_version: str = "1") -> tuple[Verifier, dict]:
    """One Task's Verifier from its References, through the whole D79 suite, with its status row.

    The row carries this Task's own replay fidelity too (D171), so a confirmed Reference says on
    which of the Task's calls the bodies were exercised and not only that the Reference stands.

    `failed_recordings` is what the false-rejection pool subtracts (D173), and it now carries the
    held-out Runs that did not reach the Reference's End state beside the recordings the D111 rule
    discarded; `did_not_reach_reference` names that half on its own, so a reader can tell the two
    apart and the ruling can count them.

    `second_path` is what the search for a second path to the Reference cost this Task and whether
    it found one (D189); a Task that never had to search carries a row of zero batches.
    """
    paths = [r.path for r in confirmation.references]
    first = confirmation.references[0]
    task_for = apply_intent(task, intents[task.id]) if task.id in intents else task
    record = derive_for(task_for, confirmation, canon_rules=canon_rules, write_tools=write_tools,
                        constraints=constraints, intent=intents.get(task.id),
                        verifier_version=verifier_version)
    rules_trace = first.trace_id or next((r.trace_id for r in confirmation.references if r.trace_id), None)
    gates = suite_for(task_for, record, paths, canon_rules=canon_rules, write_tools=write_tools,
                      user_rules=user_rules, rules_trace=rules_trace, probe_model=probe_model,
                      run_probe=probe, may_probe=may_probe, intent=intents.get(task.id))
    results = verifier_suite.d79_results(gates)
    passed = artifacts.verifier_gate(results).passed
    write_json(ctx.workdir / "verifiers" / f"{task.id}.json", as_dict(record))
    left_out = not_at_reference(confirmation, pool_runs, write_tools,
                                fn or verifier_suite.canon_fn(canon_rules))
    # D199: a Task whose path is single by structure has no second path to score and never will, so
    # check 5 is not run and the suite reports it as not passed (D173). That not run stops blocking
    # this Task alone, and only where every other check passes and the held-out pool holds a Run
    # that reached the Reference: the pool Run is the second path, arrived at by an agent nobody
    # derived from. Where the pool holds none the Task stays uncounted, as it was.
    at_reference = pool_at_reference(pool_runs, confirmation, left_out)
    waived = bool(not passed and (second_path or {}).get("reason") == SINGLE_PATH_BY_STRUCTURE
                  and at_reference
                  and all(value for name, value in results.items() if name != "second_path_passes"))
    status = {"reference_confirmed": True, "verifier_passed": bool(passed or waived),
              "second_path_waived": waived, "pool_at_reference": at_reference,
              "references": len(confirmation.references), "reference_kind": first.kind,
              # D208: which Reference this is, by the Run ids it is made of. The Verifier written
              # beside it names the same ids as its seeds, so a later round can ask whether the file
              # on disk was derived from the Reference the Task holds or from one since withdrawn.
              "reference_run_ids": [r.run_id for r in confirmation.references],
              "recordings": recordings, "rerolls": rerolls,
              "failed_recordings": {**dict(confirmation.failed), **left_out},
              "did_not_reach_reference": sorted(left_out), "judged": confirmation.judged,
              # D210: how the Simulated user of each held-out Run ended it, so a false-rejection
              # reading can tell a Run that finished from one whose scenario ran out or that spent
              # its turns, rather than reading every success termination as a Run that did the Task.
              "user_ends": dict(sorted((user_ends or {}).items())),
              "checks": results,
              "not_run": [g.stage for g in gates if g.metrics.get("skipped")],
              # D196: the columns the leak check found the strip had missed, so a finding can be
              # keyed on the Task and the column and the next build's strip can cover it.
              "leak_columns": sorted(next((g.metrics.get("columns") or []
                                           for g in gates if g.stage == "verifier_leak"), [])),
              # D198 rule 2: why each check had no input, in the check's own words, so the ruling
              # that reports a suite failure names the reason rather than a table of its own.
              "not_run_reasons": {verifier_suite.D79_STAGES[g.stage]: str(g.metrics["not_run_reason"])
                                  for g in gates if g.metrics.get("not_run_reason")
                                  and g.stage in verifier_suite.D79_STAGES},
              **verifier_mod.derivation_counts(record),
              # D218 rule 3: never a blank record. A Task that never had to search carries a row of
              # zero batches with the words for that, and a record with neither a find nor a reason
              # is refused here rather than written for a later reader to guess at.
              "second_path": with_reason(
                  second_path if second_path is not None else second_path_row(
                      0, 0, found=len(confirmation.references) > 1),
                  what="the second path search", task_id=task.id),
              **fidelity_fields(fidelity_row or {})}
    return record, status


# --- the per-Task cache (D163) ---------------------------------------------------

@lru_cache(maxsize=1)
def module_code_hash() -> str:
    """The code side of a Task's cache key: the bytes of this module and of the ones it derives through.

    A Verifier, its status row and its D79 suite are what this file, derive.py, reference.py and
    verifier_suite.py compute; an edit to any of them is a different answer over the same inputs, so
    the key moves with their bytes and yesterday's entry is a miss. The whole file is hashed rather
    than one function's source: the derivation runs through most of each of them.
    """
    paths = [Path(__file__)] + [Path(getattr(module, "__file__", "") or "") for module in CODE_MODULES]
    return content_hash({f"{path.parent.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths if path.is_file()})[:16]


def model_name(model: Any) -> Optional[str]:
    """How a model enters a cache key: its name, or that there is one at all."""
    if model is None:
        return None
    return str(getattr(model, "name", None) or type(model).__name__)


def cache_key(task: Task, recordings: list, common: dict, *, intents: dict, user_rules: dict,
              traces: dict, fidelity_row: Optional[dict] = None) -> str:
    """The content hash of everything this Task's derivation reads.

    The Runs are in by id and by the hash of what they wrote, which is what the D111 rule groups on
    and what every atom is derived from; a Run whose file changed under the same id lands on a
    different End state and so on a different key. The Intent is in twice, as the grounded record the
    Verifier is derived under and as the request text the residue judge is handed. The Task's user
    rules are in for every Trace its recordings name, since which of them the leak check reads is
    settled by the References, inside the derivation. `common` is what every Task of the call shares:
    the constraints after D76's demotion, the policy lines, the canon rules, the write tools, the
    probe's model and limit, the judge's name, which judge it is (D185's `judge_agent`, so the
    one-shot judge's answer is never read back for the agent's) and the code hash. `fidelity_row` is
    this Task's own replay fidelity (D171): it decides which tools the row names as blocking, so a
    recompile that moves it has to move the key with it.
    """
    return content_hash({
        "task": {"id": task.id, "name": task.name, "intent": task.intent, "run_ids": list(task.run_ids)},
        "intent_record": as_dict(intents[task.id]) if task.id in intents else None,
        "request": request_text(task, intents, traces),
        "fidelity": fidelity_row or {},
        "runs": [{"run_id": r.run_id, "trace_id": r.trace_id, "kind": r.kind,
                  "end_state": content_hash(r.end_state), "violated": sorted(r.violated)}
                 for r in recordings],
        "user_rules": {trace_id: user_rules.get(trace_id)
                       for trace_id in sorted({r.trace_id for r in recordings if r.trace_id})},
        **common,
    })


def cache_path(workdir: Path, task_id: str, key: str) -> Path:
    return Path(workdir).joinpath(*CACHE_DIR) / task_id / f"{key[:16]}.json"


def read_entry(workdir: Path, task_id: str, key: str) -> Optional[dict]:
    """The Task's cached outputs under this key, or None.

    An entry of another shape is a miss, and so is a half-written one: a build killed mid-derivation
    leaves whatever bytes it had reached, and the Task derives again rather than the round failing on
    a file the cache itself wrote.
    """
    try:
        entry = read_json(cache_path(workdir, task_id, key))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("format") != CACHE_FORMAT or entry.get("key") != key:
        return None
    if entry.get("task_id") != task_id or not isinstance(entry.get("status"), dict):
        return None
    return entry


@dataclass
class _Job:
    """One Task through the two pools: what the first found, what the second has to run."""
    task: Task
    key: str
    entry: Optional[dict] = None  # the cache hit, with the Task's outputs in it
    confirmation: Any = None      # the D111 answer, on a miss
    may_probe: bool = False
    # Every Run the key was taken over: the Task's own recordings and the extra batches an earlier
    # derivation bought for the second path (D189), which are merged after the rule has settled.
    recordings: list = field(default_factory=list)
    extra: list = field(default_factory=list)

    @property
    def cached(self) -> bool:
        return self.entry is not None


@dataclass
class _DeriveState:
    """Everything one `derive_all` call shares: what it loaded, what it demoted, what it judges with."""
    ctx: Any
    sample_salt: str
    canon_rules: Any
    fn: Any
    write_tools: set
    read_tools: set
    replays: dict
    rerolls: dict
    intents: dict
    user_rules: dict
    traces: dict
    policy_lines: list
    seed_replays: dict
    constraints: list
    demoted: list
    assisted_tools: set
    tool_fidelity: dict
    atoms: Any
    judge: Any
    probe: Any
    probe_model: Any
    probe_limit: Any
    run_rerolls: Any
    run_variant: Any
    round_number: int
    common: dict


def _tool_names(sigs: Iterable[Any]) -> tuple[set, set]:
    """The write tools and the rest, off the mined signatures."""
    write_tools = {s.name for s in sigs if s.kind == "write"}
    read_tools = {s.name for s in sigs if s.kind != "write"}
    return write_tools, read_tools


def _store_map(inputs: dict, name: str) -> dict:
    """One mapping input, empty when the store does not hold it."""
    return inputs.get(name) or {}


def _store_list(inputs: dict, name: str) -> list:
    """One list input, empty when the store does not hold it."""
    return inputs.get(name) or []


def _validated_intents(inputs: dict) -> dict:
    """The Intent records, validated, keyed by Task id."""
    return {t: Intent.model_validate(d) for t, d in _store_map(inputs, "intents").items()}


def _trace_index(inputs: dict) -> dict:
    """The Traces, keyed by trace id."""
    return {t.trace_id: t for t in _store_list(inputs, "traces")}


def _policy_lines(inputs: dict) -> list:
    """The policy text lines the residue judge reads."""
    return [c.text for c in inputs["constraints"]]


def _seed_replays_of(ctx: ExamContext, replays: dict, tasks: list) -> dict:
    """The confirmed seed replays per Task, minus the anchor's held-out Runs (D81)."""
    out = {}
    for task in tasks:
        rows = _store_map(replays, task.id)
        out[task.id] = [r for tid, r in sorted(rows.items())
                         if tid in seed_ids(ctx, task) and _seed_row_confirmed(r)]
    return out


def _seed_row_confirmed(row: dict) -> bool:
    """Whether one seed replay row is confirmed and has a path to load."""
    return bool(row.get("confirmed")) and bool(row.get("path"))


def _build_judge(judge_model: Any, judge_agent: bool, constraints: list, write_tools: set,
                 read_tools: set, fn: Any) -> Any:
    """The residue judge: nothing, the one-shot judge, or the agent with a bounded look (D185)."""
    if judge_model is None:
        return None
    if judge_agent:
        return judge_mod.AgentJudge(judge_model, constraints=constraints,
                                    write_tools=write_tools, read_tools=read_tools, fn=fn)
    return judge_model


def _probe_of(run_probe: Any, probe_model: Any) -> Any:
    """The loophole probe runner, or nothing when no probe model was given."""
    if probe_model is None:
        return None
    return run_probe


def _select_tasks(inputs: dict, only: Optional[str]) -> list:
    """The Tasks to derive, or the one named Task; an unknown name is refused."""
    tasks = list(inputs["tasks"])
    if only is None:
        return tasks
    picked = [task for task in tasks if task.id == only]
    if not picked:
        raise ValueError(f"no Task is named {only}")
    return picked


def _key_base(code_hash: Optional[str], canon_rules: Any, constraints: list, policy_lines: list,
              write_tools: set, probe_model: Any, probe_limit: Any, probe: Any,
              judge_model: Any, judge_agent: bool) -> dict:
    """What every Task's cache key shares: format, code, rules, constraints, models."""
    if code_hash is None:
        code_hash = module_code_hash()
    return {"format": CACHE_FORMAT, "code": code_hash, "canon_rules": canon_rules,
            "constraints": [as_dict(c) for c in constraints],
            "policy_lines": list(policy_lines), "write_tools": sorted(write_tools),
            "probe": {"model": model_name(probe_model), "limit": probe_limit,
                      "runner": probe is not None},
            "judge": model_name(judge_model), "judge_agent": bool(judge_agent)}


def _init_state(ctx: ExamContext, inputs: dict, *, probe_model: Any = None,
                probe_limit: Optional[int] = None, judge_model: Any = None,
                judge_agent: bool = False, run_probe: Any = None, run_rerolls: Any = None,
                run_variant: Any = None, round_number: int = 0,
                code_hash: Optional[str] = None) -> _DeriveState:
    """Load the call's shared inputs, demote the broken rules, build the judge and the cache key base.

    The demotion writes constraints_check.json and records its gate before anything else reads the
    constraints, which is the order the reference check runs in; selecting the Tasks comes after
    it in the orchestrator, so a bad name still leaves the demotion behind, as it did.
    """
    sample_salt = sampling.build_salt(ctx.workdir)
    canon_rules = rules_of(inputs)
    fn = verifier_suite.canon_fn(canon_rules)
    write_tools, read_tools = _tool_names(inputs["sigs"])
    replays = _store_map(inputs, "replays")
    rerolls = _store_map(inputs, "rerolls")
    intents = _validated_intents(inputs)
    user_rules = _store_map(inputs, "user_rules")
    traces = _trace_index(inputs)
    policy_lines = _policy_lines(inputs)
    seed_replays = _seed_replays_of(ctx, replays, list(inputs["tasks"]))
    constraints, demoted = final_constraints(ctx, inputs, seed_replays, write_tools, read_tools, fn)
    assisted_tools = set(_store_list(inputs, "assisted_tools"))
    tool_fidelity = _store_map(inputs, "tool_fidelity")
    atoms = reference_mod.hard_atoms(constraints, write_tools, read_tools)
    judge = _build_judge(judge_model, judge_agent, constraints, write_tools, read_tools, fn)
    probe = _probe_of(run_probe, probe_model)
    common = _key_base(code_hash, canon_rules, constraints, policy_lines, write_tools,
                       probe_model, probe_limit, probe, judge_model, judge_agent)
    return _DeriveState(ctx=ctx, sample_salt=sample_salt, canon_rules=canon_rules, fn=fn,
                         write_tools=write_tools, read_tools=read_tools, replays=replays,
                         rerolls=rerolls, intents=intents, user_rules=user_rules, traces=traces,
                         policy_lines=policy_lines, seed_replays=seed_replays,
                         constraints=constraints, demoted=demoted, assisted_tools=assisted_tools,
                         tool_fidelity=tool_fidelity, atoms=atoms, judge=judge, probe=probe,
                         probe_model=probe_model, probe_limit=probe_limit,
                         run_rerolls=run_rerolls, run_variant=run_variant,
                         round_number=round_number, common=common)


def _prepare_one(task: Task, state: _DeriveState) -> _Job:
    """One Task's Runs, its cache key, and the D111 answer when the key is not on disk."""
    recordings = [reference_mod.load(r["path"], reference_mod.RECORDING, run_id=r["run_id"],
                                     trace_id=r["trace_id"], write_tools=state.write_tools,
                                     fn=state.fn, atoms=state.atoms)
                  for r in state.seed_replays[task.id]]
    recordings += [reference_mod.load(r["path"], reference_mod.REROLL, run_id=r["run_id"],
                                      write_tools=state.write_tools, fn=state.fn, atoms=state.atoms)
                   for r in state.rerolls.get(task.id, [])
                   if (r.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS]
    # D189: the extra batches an earlier derivation bought sit outside the D111 rule's evidence,
    # so they are in the key (a batch bought is a different derivation) and are merged onto the
    # Confirmation afterwards rather than being handed to `confirm`.
    extra = finished_recordings(second_path_rows(state.ctx.workdir, task.id),
                                write_tools=state.write_tools, fn=state.fn, atoms=state.atoms)
    key = cache_key(task, recordings + extra, state.common, intents=state.intents,
                    user_rules=state.user_rules, traces=state.traces,
                    fidelity_row=task_fidelity(state.tool_fidelity, task.id))
    entry = read_entry(state.ctx.workdir, task.id, key)
    if entry is not None:
        return _Job(task=task, key=key, entry=entry)
    confirmation = reference_mod.confirm(recordings, intent=request_text(task, state.intents,
                                                                         state.traces),
                                         policy_lines=state.policy_lines, judge=state.judge,
                                         phrases=grounded_phrases(state.intents.get(task.id)))
    return _Job(task=task, key=key, confirmation=confirmation,
                recordings=recordings + extra, extra=extra)


def _prepare_all(tasks: list, state: _DeriveState, workers: int) -> list[_Job]:
    """Every Task through its key lookup and, on a miss, the D111 rule, in Task order."""
    return parallel.each(tasks, lambda task: _prepare_one(task, state), workers)


def _assign_probe_slots(jobs: list[_Job], state: _DeriveState) -> int:
    """Hand the loophole probe budget out by the Tasks' own keys (D212), and count slots spent.

    A Task served from the cache spent its slot when it ran, so the slots left are the limit minus
    those; which Tasks hold a slot never moves when a Task is added or dropped.
    """
    probed = sum(1 for job in jobs if job.cached and job.entry.get("probed"))
    eligible = [job for job in jobs if not job.cached and job.confirmation.references]
    if state.probe is None:
        chosen: set[str] = set()
    elif state.probe_limit is None:
        chosen = {job.task.id for job in eligible}
    else:
        order = sampling.keyed_order(PROBE_KIND, [job.task.id for job in eligible],
                                     state.sample_salt)
        chosen = set(order[:max(0, state.probe_limit - probed)])
    for job in eligible:
        job.may_probe = job.task.id in chosen
        probed += int(job.may_probe)
    return probed


def _finish_cached(state: _DeriveState, job: _Job) -> dict:
    """A cache hit's entry, with its Verifier file rewritten beside the live rows."""
    if job.entry.get("verifier") is not None:
        write_json(state.ctx.workdir / "verifiers" / f"{job.task.id}.json", job.entry["verifier"])
    return job.entry


def _settle_job(state: _DeriveState, job: _Job) -> None:
    """Settle a reference-less residue by deriving one Verifier per surviving End state (D198)."""
    confirmation = job.confirmation
    if not confirmation.references and confirmation.survivors:
        settle_residue(job.task, confirmation, canon_rules=state.canon_rules,
                       write_tools=state.write_tools, constraints=state.constraints,
                       intents=state.intents, user_rules=state.user_rules, fn=state.fn,
                       pool_runs=pool_runs_of(job.task.id, state.replays, state.rerolls))


def _second_path_outcome(state: _DeriveState, job: _Job, ceiling: threading.Event) -> tuple[dict, list]:
    """The Task's second path: bought batches first (D189), then a synthesised one (D199).

    Merges the earlier batches, buys fresh ones until check 5 has a second path, writes one from
    the Reference's own call path when the buying found nothing, rekeys when batches were bought,
    and raises the ceiling flag when a batch hit the run ceiling.
    """
    task, confirmation = job.task, job.confirmation
    merge_second_path(confirmation, job.extra)
    second, bought, hit = second_path_search(
        task.id, confirmation, workdir=state.ctx.workdir, run_rerolls=state.run_rerolls,
        round_number=state.round_number, write_tools=state.write_tools, fn=state.fn,
        atoms=state.atoms)
    if not second["found"] and state.run_variant is not None:
        synth, made = synth_second_path(
            task.id, confirmation, run_variant=state.run_variant,
            round_number=state.round_number, write_tools=state.write_tools, fn=state.fn,
            atoms=state.atoms)
        merge_second_path(confirmation, made)
        # They are not in the cache key and not in the false-rejection pool: a synthesised
        # Run is written by code from a Run already in the key, so a round that reads the
        # entry back gets the same answer, and it is nobody's held-out Run to reject.
        second = second_path_row(
            second["batches"], second["runs"], found=len(confirmation.references) > 1,
            reason=synth["reason"] or second["reason"], synthesised=len(made),
            kinds=synth["kinds"], tried=synth["tried"], kept=len(made),
            structural=synth["structural"] and not made)
    if bought:
        job.recordings = job.recordings + bought
        fidelity_row = task_fidelity(state.tool_fidelity, task.id)
        job.key = cache_key(task, job.recordings, state.common, intents=state.intents,
                            user_rules=state.user_rules, traces=state.traces,
                            fidelity_row=fidelity_row)
    if hit:
        ceiling.set()
    return second, bought


def _write_entry(state: _DeriveState, task_id: str, key: str, row: dict, reference: dict,
                 verifier: Optional[dict], may_probe: bool) -> dict:
    """One Task's cache entry, built and written in one place."""
    entry = {"format": CACHE_FORMAT, "task_id": task_id, "key": key, "status": row,
             "references": reference, "verifier": verifier, "probed": may_probe}
    write_json(cache_path(state.ctx.workdir, task_id, key), entry)
    return entry


def _derive_and_store(state: _DeriveState, job: _Job, second: dict, bought: list,
                      fidelity_row: dict) -> dict:
    """One Task's Verifier through the D79 suite, and its cache entry on disk."""
    task, confirmation = job.task, job.confirmation
    extra_pool = [(r.run_id, r.path) for r in job.extra + bought]
    record, row = verifier_for(
        state.ctx, task, confirmation, canon_rules=state.canon_rules,
        write_tools=state.write_tools, constraints=state.constraints, intents=state.intents,
        user_rules=state.user_rules, recordings=len(state.seed_replays[task.id]),
        rerolls=len(state.rerolls.get(task.id, [])), probe=state.probe,
        probe_model=state.probe_model, may_probe=job.may_probe, fidelity_row=fidelity_row,
        second_path=second,
        pool_runs=pool_runs_of(task.id, state.replays, state.rerolls) + extra_pool,
        fn=state.fn, user_ends=pool_user_ends(task.id, state.rerolls))
    return _write_entry(state, task.id, job.key, row, confirmation.as_dict(), as_dict(record),
                        job.may_probe)


def _finish_one(state: _DeriveState, job: _Job, ceiling: threading.Event) -> dict:
    """One Task's outputs: the cache entry as it stands, or the derivation and a new entry."""
    if job.cached:
        return _finish_cached(state, job)
    _settle_job(state, job)
    confirmation = job.confirmation
    fidelity_row = task_fidelity(state.tool_fidelity, job.task.id)
    if not confirmation.references:
        row = no_reference_status(state.ctx, job.task, confirmation,
                                  seed_replays=state.seed_replays[job.task.id],
                                  replays=state.replays, rerolls=state.rerolls,
                                  traces=state.traces, assisted_tools=state.assisted_tools,
                                  fidelity_row=fidelity_row)
        return _write_entry(state, job.task.id, job.key, row, confirmation.as_dict(), None,
                              job.may_probe)
    second, bought = _second_path_outcome(state, job, ceiling)
    return _derive_and_store(state, job, second, bought, fidelity_row)


def _finish_all(state: _DeriveState, jobs: list[_Job], workers: int,
                ceiling: threading.Event) -> list[dict]:
    """Every Task to its outputs, in Task order."""
    return parallel.each(jobs, lambda job: _finish_one(state, job, ceiling), workers)


def _collect_outputs(jobs: list[_Job], entries: list[dict]) -> tuple[list, dict, dict, int]:
    """The pools' answers back in Task order: verifiers, status rows, reference rows, cache hits."""
    verifiers, status, references = [], {}, {}
    for job, entry in zip(jobs, entries, strict=True):
        status[job.task.id] = entry["status"]
        references[job.task.id] = entry["references"]
        if entry["verifier"] is not None:
            verifiers.append(Verifier.model_validate(entry["verifier"]))
    return verifiers, status, references, sum(1 for job in jobs if job.cached)


def _read_prior(workdir: Path, only: Optional[str]) -> tuple[dict, dict]:
    """The live rows already on disk, for the `only` merge and the stamps.

    The references file is read only when one Task is merged into it, as it was.
    """
    prior_status = read_json(workdir / "task_status.json", {})
    if not prior_status:
        prior_status = {}
    if only is None:
        return prior_status, {}
    prior_references = read_json(workdir / "references.json", {})
    if not prior_references:
        prior_references = {}
    return prior_status, prior_references


def _persist_results(ctx: ExamContext, status: dict, references: dict, only: Optional[str],
                     round_number: int) -> tuple[dict, dict, dict]:
    """Merge a single-Task derivation, retire withdrawn artefacts, write the live files.

    A Verifier lives only while its Task holds the Reference it was derived from (D208), so the
    artefacts the withdrawn ones left are retired in the same step, before anything reads them;
    the stamps say on every row which round last moved it (D218 rule 2).
    """
    prior_status, prior_references = _read_prior(ctx.workdir, only)
    if only is not None:
        status = {**prior_status, **status}
        references = {**prior_references, **references}
    retired = lifecycle.retire(ctx.workdir, status, round_number=round_number)
    lifecycle.carry_forward(ctx.workdir, status, prior_status)
    write_json(ctx.workdir / "task_status.json", stamped(status, prior_status, round_number))
    write_json(ctx.workdir / "references.json", references)
    return status, references, retired


def _confirmed_count(status: dict) -> int:
    """How many rows hold a Reference."""
    return sum(1 for r in status.values() if r["reference_confirmed"])


def _passed_count(status: dict) -> int:
    """How many rows hold a Verifier that cleared the suite."""
    return sum(1 for r in status.values() if r["verifier_passed"])


def _blocked_count(status: dict) -> int:
    """How many Tasks a tool actually blocks with a differing own call (D171)."""
    return sum(1 for r in status.values() if r.get("blocking_tools"))


def _failed_recordings_count(references: dict) -> int:
    """The failed recordings every reference row carries, counted for the round."""
    total = 0
    for r in references.values():
        failed = r.get("failed")
        if failed is None:
            continue
        total += len(failed)
    return total


def _judged_count(references: dict) -> int:
    """How many reference rows went through the judge."""
    return sum(1 for r in references.values() if r.get("judged"))


def _base_counts(status: dict, verifiers: list, references: dict, probed: int, demoted: list,
                 retired: dict) -> dict:
    """The round's counts over the rows: verdicts, probes, retirements, blocks, failures."""
    return {"verifiers": len(verifiers), "references": _confirmed_count(status),
            "passed": _passed_count(status), "tasks": len(status), "probed": probed,
            "constraints_demoted": len(demoted), **lifecycle.counts(retired),
            "blocked_by_own_calls": _blocked_count(status),
            "failed_recordings": _failed_recordings_count(references),
            "judged": _judged_count(references)}


def _judge_counts(references: dict) -> dict:
    """What the residue judging cost and settled: uncited drops (D186), second passes (D193)."""
    return {"judge_uncited": sum(1 for r in references.values() if r.get("judge_uncited")),
            "judge_second_pass": sum(1 for r in references.values()
                                      if _passes_of(r) > 1),
            "judge_residue_resolved": sum(1 for r in references.values()
                                           if r.get("judge_residue_resolved")),
            "judge_residue_abstained": sum(1 for r in references.values()
                                            if r.get("judge_residue_abstained")),
            "no_correct_recording": sum(1 for r in references.values()
                                         if r.get("no_correct_recording"))}


def _passes_of(row: dict) -> int:
    """How many judging passes one reference row went through."""
    passes = row.get("judge_passes")
    if not passes:
        return 0
    return passes


def _residue_counts(references: dict) -> dict:
    """What deriving a Verifier per surviving End state settled (D198)."""
    return {"residue_derived": sum(1 for r in references.values() if r.get("residue_derived")),
            "survivor_chosen": sum(1 for r in references.values()
                                    if r.get("survivor_reason") == reference_mod.SURVIVOR_CHOSEN),
            "survivors_equivalent": sum(1 for r in references.values()
                                         if r.get("survivor_reason")
                                         == reference_mod.SURVIVORS_EQUIVALENT),
            "survivors_all_fail": sum(1 for r in references.values()
                                       if r.get("survivor_reason")
                                       == reference_mod.SURVIVORS_ALL_FAIL),
            "survivors_capped": sum(1 for r in references.values()
                                     if r.get("survivors_capped"))}


def _pool_counts(status: dict) -> dict:
    """The held-out Runs left out short of the Reference (D133), and how their users ended (D210)."""
    return {"did_not_reach_reference": sum(len(_left_out_of(r)) for r in status.values()),
            "user_ends_by_kind": dict(sorted(Counter(
                kind for r in status.values() for kind in _user_ends_of(r).values()).items()))}


def _left_out_of(row: dict) -> list:
    """The held-out Runs one status row names as short of the Reference."""
    left_out = row.get("did_not_reach_reference")
    if not left_out:
        return []
    return left_out


def _user_ends_of(row: dict) -> dict:
    """How one row's held-out Runs ended, keyed by Run."""
    ends = row.get("user_ends")
    if not ends:
        return {}
    return ends


def _found_batches_of(row: dict) -> bool:
    """Whether one row's second-path search both bought batches and found a path."""
    second = _second_path(row)
    return bool(second["found"]) and bool(second["batches"])


def _path_counts(status: dict) -> dict:
    """What the search for a second path cost and bought this round (D189)."""
    return {"second_path_found": sum(1 for r in status.values() if _found_batches_of(r)),
            "second_path_exhausted": sum(1 for r in status.values()
                                          if _second_path(r)["exhausted"]),
            "second_path_batches": sum(_second_path(r)["batches"] for r in status.values()),
            "second_path_runs": sum(_second_path(r)["runs"] for r in status.values())}


def _synth_counts(status: dict) -> dict:
    """What the synthesis wrote when the buying was spent (D199), and the waived not runs."""
    return {"second_path_synthesised": sum(1 for r in status.values()
                                            if _second_path(r)["synthesised"]),
            "single_path_by_structure": sum(1 for r in status.values()
                                             if _second_path(r)["structural"]),
            "synth_variants_tried": sum(_second_path(r)["synth_tried"] for r in status.values()),
            "synth_variants_kept": sum(_second_path(r)["synth_kept"] for r in status.values()),
            "second_path_waived": sum(1 for r in status.values()
                                       if r.get("second_path_waived"))}


def _disagreeing_count(references: dict) -> int:
    """The Tasks with no Reference whose recordings still disagree."""
    return sum(1 for r in references.values() if _still_disagrees(r))


def _still_disagrees(row: dict) -> bool:
    """Whether one reference row holds no Reference over disagreeing recordings."""
    if row["references"]:
        return False
    reason = row.get("reason")
    if not reason:
        reason = ""
    return reason.startswith("recordings disagree")


def _round_metrics(status: dict, verifiers: list, references: dict, probed: int, demoted: list,
                   retired: dict) -> dict:
    """Every count the derive_verifier gate records, in one mapping."""
    return {**_base_counts(status, verifiers, references, probed, demoted, retired),
            **_judge_counts(references), **_residue_counts(references),
            **_pool_counts(status), **_path_counts(status), **_synth_counts(status),
            "disagreeing": _disagreeing_count(references)}


def _record_round(ctx: ExamContext, status: dict, metrics: dict) -> None:
    """Record the round's gate and rewrite the scorecard last, as the pipeline stage did."""
    ctx.record_gate(stage_gates.task_verifiers_gate(status, **metrics))
    write_json(ctx.workdir / "scorecard.json", scorecard_mod.scorecard(ctx.workdir))


# --- the stage body --------------------------------------------------------------

def derive_all(ctx: ExamContext, inputs: dict, *, probe_model: Any = None, probe_limit: Optional[int] = None,
               judge_model: Any = None, judge_agent: bool = False, run_probe: Any = None,
               run_rerolls: Any = None, run_variant: Any = None, round_number: int = 0,
               only: Optional[str] = None,
               workers: int = 1, code_hash: Optional[str] = None) -> dict:
    """One Verifier per Task from its References by the D111 rule, through the whole D79 suite.

    The References are the confirmed seed replays plus the finished re-rolls that agree on one End
    state after the recordings that broke a Hard constraint are out; the judge is the residue when
    two End states remain and fails at most one side (D110, D111). A judgement that leaves two or
    more states in is asked once more over those states alone, under a stop rule saying one of them
    may remain, and anything but one state left is an abstention (D193). What that abstention leaves,
    and a judgement that failed every state on grounds that do not agree, is settled without the judge:
    a Verifier is derived from each surviving End state and the suite and the held-out pool choose
    between them (D198). The Reference proper is the first
    recording of that group, the rest are the re-runs whose agreement sets required against allowed
    (D43) and the second path of check 5, and the anchor is never among them (D81). Before any of
    that, every compiled constraint is checked against the confirmed recordings corpus-wide and the
    ones they mostly break are demoted (D76). Check 4's wrong Run is built from the Reference by
    code; check 6's loophole probe is the one Run per Task `run_probe` executes, and `probe_limit`
    caps how many Tasks get one. A Task with no Reference is not verdicted.

    `judge_model` is the model that residue judging runs on and `judge_agent` is which judge it runs:
    off, the default, is the one-shot judge of D110, one call over the Intent, the policy and the End
    states; on, it is the agent with a bounded look of D185, which asks its own read tools before it
    rules and falls back to the one-shot judge over its cap. The flag is in the cache key beside the
    judge's name, so turning it on or off derives every Reference again rather than reading the other
    judge's answer off disk.

    `run_rerolls` is the Builder's re-roll callable (D120), and a Task whose Reference stands alone
    is re-rolled through it until a Run reaches that Reference's End state by another path or the
    cap of `SECOND_PATH_BATCHES` batches is spent (D189). Without the callable nothing is bought and
    check 5 stays not run, as it was. The Runs of every batch are recorded in the Examiner's own
    re-roll file under the reason `second_path`, so a second derivation over the same Task reads
    them back rather than buying them again.

    `run_variant` is the Builder's replay callable (D199), and a Task whose Reference still stands
    alone after that search has a second path written from the Reference's own call path instead:
    rewrites that leave the world where it was, each replayed from the Task's Starting state and
    kept only where it reaches the Reference's End state under the same rule. No model is called and
    nothing is bought. A Task the rewrites cannot touch is single by structure, which its row says
    and the round counts.

    An assisted tool is a corpus-level ruling and it blocks no Task on its own (D171): a Task is
    blocked by a tool only when one of the Task's own recorded calls of it differs, which is what
    `blocking_tools` on the row names. Every row carries the Task's own replay fidelity per tool,
    confirmed or not.

    With `only` one Task is derived and its rows are merged into the task_status.json and
    references.json already on disk; without it the whole build is derived and the files are the
    stage's, byte for byte. Either way scorecard.json is rewritten last, so it reads as it did when
    the stage ran inside the pipeline.

    The Tasks are independent, so their bodies run on `workers` threads and a Task whose inputs have
    not moved since the last call is served from `<workdir>/examiner/cache/<task_id>/` without
    running anything: build 13 spent 13.7 of its 16 hours in two of these calls over the same
    artifacts, most of it in the probe Run and the re-rolls of the D79 suite (D163). The results are
    gathered in Task order and the two files are written once, after the pool, so both give the
    bytes the serial run gave; the probe budget is handed out between the two pools by the Tasks'
    own keys (D212), so which Tasks get check 6 does not move when a Task is added or dropped.
    `cached` and `ran` in the result count which Tasks came from where.
    """
    state = _init_state(ctx, inputs, probe_model=probe_model, probe_limit=probe_limit,
                        judge_model=judge_model, judge_agent=judge_agent, run_probe=run_probe,
                        run_rerolls=run_rerolls, run_variant=run_variant,
                        round_number=round_number, code_hash=code_hash)
    tasks = _select_tasks(inputs, only)
    jobs = _prepare_all(tasks, state, workers)
    probed = _assign_probe_slots(jobs, state)

    # Set when a second-path batch hit the run ceiling, so the caller can stop the round rather than
    # read a derivation that quietly bought nothing (the re-roll tool raises for the same reason).
    ceiling_reached = threading.Event()

    entries = _finish_all(state, jobs, workers, ceiling_reached)
    verifiers, status, references, cached = _collect_outputs(jobs, entries)
    status, references, retired = _persist_results(ctx, status, references, only, round_number)
    # Section 6: a Task whose Verifier does not clear D79 is "not verdicted, Verifier
    # immature", which is a Task the report leaves uncounted, not a failed build.
    _record_round(ctx, status, _round_metrics(status, verifiers, references, probed,
                                              state.demoted, retired))
    return {"verifiers": verifiers, "task_status": status, "cached": cached, "ran": len(jobs) - cached,
            "second_path_runs": sum(_second_path(r)["runs"] for r in status.values()),
            "retired": retired, "ceiling_reached": ceiling_reached.is_set()}
