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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.examiner import derive as verifier_mod
from kullback.examiner import judge as judge_mod
from kullback.examiner import reference as reference_mod
from kullback.gates import artifacts, fidelity, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates import stages as stage_gates
from kullback.gates.ledger import GateLedger
from kullback.runner import parallel
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
CACHE_FORMAT = 3  # the reference record carries what the judge looked at, and the status row the pool
# The modules a Task's derivation runs through, hashed into every key: an edit to any of them is a
# different derivation and must not be served a stale entry (the Builder's stages hash the same way,
# build.py's `_version`).
CODE_MODULES = (verifier_mod, reference_mod, judge_mod, verifier_suite)


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
    say, which is the order the reference check has to run in: on the second retail build 15 of 39
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
              may_probe: bool) -> list[GateResult]:
    """The whole D79 suite over one Verifier: the wrong Run, the second path, the leak, the probe (D79, D119)."""
    return verifier_suite.validate_verifier(
        verifier, paths[0], canon=canon_rules, write_tools=write_tools, seed_runs=paths[1:],
        wrong_run=verifier_suite.wrong_run(verifier, paths[0], canon_rules),
        alt_path_run=paths[1] if len(paths) > 1 else None,
        intent_text=task_for.intent, user_rules=user_rules.get(rules_trace),
        model=probe_model if may_probe else None, run_probe=run_probe)


def verifier_for(ctx, task: Task, confirmation: Any, *, canon_rules: Any, write_tools: set, constraints: list,
                 intents: dict, user_rules: dict, recordings: int, rerolls: int, probe: Any,
                 probe_model: Any, may_probe: bool, fidelity_row: Optional[dict] = None,
                 pool_runs: Iterable[tuple[str, str]] = (), fn: Optional[Callable] = None,
                 verifier_version: str = "1") -> tuple[Verifier, dict]:
    """One Task's Verifier from its References, through the whole D79 suite, with its status row.

    The row carries this Task's own replay fidelity too (D171), so a confirmed Reference says on
    which of the Task's calls the bodies were exercised and not only that the Reference stands.

    `failed_recordings` is what the false-rejection pool subtracts (D173), and it now carries the
    held-out Runs that did not reach the Reference's End state beside the recordings the D111 rule
    discarded; `did_not_reach_reference` names that half on its own, so a reader can tell the two
    apart and the ruling can count them.
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
                      run_probe=probe, may_probe=may_probe)
    results = verifier_suite.d79_results(gates)
    passed = artifacts.verifier_gate(results).passed
    write_json(ctx.workdir / "verifiers" / f"{task.id}.json", as_dict(record))
    left_out = not_at_reference(confirmation, pool_runs, write_tools,
                                fn or verifier_suite.canon_fn(canon_rules))
    status = {"reference_confirmed": True, "verifier_passed": bool(passed),
              "references": len(confirmation.references), "reference_kind": first.kind,
              "recordings": recordings, "rerolls": rerolls,
              "failed_recordings": {**dict(confirmation.failed), **left_out},
              "did_not_reach_reference": sorted(left_out), "judged": confirmation.judged,
              "checks": results,
              "not_run": [g.stage for g in gates if g.metrics.get("skipped")],
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

    @property
    def cached(self) -> bool:
        return self.entry is not None


# --- the stage body --------------------------------------------------------------

def derive_all(ctx: ExamContext, inputs: dict, *, probe_model: Any = None, probe_limit: Optional[int] = None,
               judge_model: Any = None, judge_agent: bool = False, run_probe: Any = None,
               only: Optional[str] = None,
               workers: int = 1, code_hash: Optional[str] = None) -> dict:
    """One Verifier per Task from its References by the D111 rule, through the whole D79 suite.

    The References are the confirmed seed replays plus the finished re-rolls that agree on one End
    state after the recordings that broke a Hard constraint are out; the judge is the residue when
    two End states remain and fails at most one side (D110, D111). The Reference proper is the first
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
    bytes the serial run gave; the probe budget is handed out between the two pools, in Task order,
    for the same reason. `cached` and `ran` in the result count which Tasks came from where.
    """
    canon_rules = rules_of(inputs)
    fn = verifier_suite.canon_fn(canon_rules)
    write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}
    read_tools = {s.name for s in inputs["sigs"] if s.kind != "write"}
    replays = inputs.get("replays") or {}
    rerolls = inputs.get("rerolls") or {}
    intents = {t: Intent.model_validate(d) for t, d in (inputs.get("intents") or {}).items()}
    user_rules = inputs.get("user_rules") or {}
    traces = {t.trace_id: t for t in inputs.get("traces") or []}
    policy_lines = [c.text for c in inputs["constraints"]]
    seed_replays = {task.id: [r for tid, r in sorted((replays.get(task.id) or {}).items())
                              if tid in seed_ids(ctx, task) and r.get("confirmed") and r.get("path")]
                    for task in inputs["tasks"]}
    # D76, D111: a compiled rule the confirmed recordings mostly break is demoted before any
    # Verifier is derived from them.
    constraints, demoted = final_constraints(ctx, inputs, seed_replays, write_tools, read_tools, fn)
    assisted_tools = set(inputs.get("assisted_tools") or ())
    tool_fidelity = inputs.get("tool_fidelity") or {}
    atoms = reference_mod.hard_atoms(constraints, write_tools, read_tools)
    # D185: the residue judge is the one-shot judge unless the build asked for the agent with a
    # bounded look, which wraps the one-shot judge as its own fallback. Either way it is built once
    # for the build and asked once per disagreeing Task; `judge_groups` dispatches on the object.
    if judge_model is None:
        judge = None
    elif judge_agent:
        judge = judge_mod.AgentJudge(judge_model, constraints=constraints, write_tools=write_tools,
                                     read_tools=read_tools, fn=fn)
    else:
        judge = judge_model
    probe = run_probe if probe_model is not None else None
    tasks = list(inputs["tasks"])
    if only is not None:
        tasks = [task for task in tasks if task.id == only]
        if not tasks:
            raise ValueError(f"no Task is named {only}")
    common = {"format": CACHE_FORMAT,
              "code": code_hash if code_hash is not None else module_code_hash(),
              "canon_rules": canon_rules,
              "constraints": [as_dict(c) for c in constraints],
              "policy_lines": list(policy_lines),
              "write_tools": sorted(write_tools),
              "probe": {"model": model_name(probe_model), "limit": probe_limit, "runner": probe is not None},
              "judge": model_name(judge_model), "judge_agent": bool(judge_agent)}

    def prepare(task: Task) -> _Job:
        """The Task's Runs, its key, and the D111 answer when the key is not on disk."""
        recordings = [reference_mod.load(r["path"], reference_mod.RECORDING, run_id=r["run_id"],
                                         trace_id=r["trace_id"], write_tools=write_tools, fn=fn, atoms=atoms)
                      for r in seed_replays[task.id]]
        recordings += [reference_mod.load(r["path"], reference_mod.REROLL, run_id=r["run_id"],
                                          write_tools=write_tools, fn=fn, atoms=atoms)
                       for r in rerolls.get(task.id, [])
                       if (r.get("termination_reason") or "") in verifier_suite.SUCCESS_TERMINATIONS]
        key = cache_key(task, recordings, common, intents=intents, user_rules=user_rules, traces=traces,
                        fidelity_row=task_fidelity(tool_fidelity, task.id))
        entry = read_entry(ctx.workdir, task.id, key)
        if entry is not None:
            return _Job(task=task, key=key, entry=entry)
        confirmation = reference_mod.confirm(recordings, intent=request_text(task, intents, traces),
                                             policy_lines=policy_lines, judge=judge,
                                             phrases=grounded_phrases(intents.get(task.id)))
        return _Job(task=task, key=key, confirmation=confirmation)

    jobs = parallel.each(tasks, prepare, workers)
    # The probe budget is spent in Task order whatever order the threads ran in, so which Tasks get
    # check 6 is the serial run's answer; a Task served from the cache spent its slot when it ran.
    probed = 0
    for job in jobs:
        if job.cached:
            probed += int(bool(job.entry.get("probed")))
        elif job.confirmation.references:
            job.may_probe = probe is not None and (probe_limit is None or probed < probe_limit)
            probed += int(job.may_probe)

    def finish(job: _Job) -> dict:
        """The Task's outputs: the cache entry as it stands, or the derivation and a new entry."""
        task = job.task
        if job.cached:
            if job.entry.get("verifier") is not None:
                write_json(ctx.workdir / "verifiers" / f"{task.id}.json", job.entry["verifier"])
            return job.entry
        confirmation = job.confirmation
        fidelity_row = task_fidelity(tool_fidelity, task.id)
        if not confirmation.references:
            row = no_reference_status(ctx, task, confirmation, seed_replays=seed_replays[task.id],
                                      replays=replays, rerolls=rerolls, traces=traces,
                                      assisted_tools=assisted_tools, fidelity_row=fidelity_row)
            verifier = None
        else:
            record, row = verifier_for(
                ctx, task, confirmation, canon_rules=canon_rules, write_tools=write_tools,
                constraints=constraints, intents=intents, user_rules=user_rules,
                recordings=len(seed_replays[task.id]), rerolls=len(rerolls.get(task.id, [])),
                probe=probe, probe_model=probe_model, may_probe=job.may_probe,
                fidelity_row=fidelity_row, pool_runs=pool_runs_of(task.id, replays, rerolls), fn=fn)
            verifier = as_dict(record)
        entry = {"format": CACHE_FORMAT, "task_id": task.id, "key": job.key, "status": row,
                 "references": confirmation.as_dict(), "verifier": verifier, "probed": job.may_probe}
        write_json(cache_path(ctx.workdir, task.id, job.key), entry)
        return entry

    entries = parallel.each(jobs, finish, workers)
    verifiers, status, references = [], {}, {}
    for job, entry in zip(jobs, entries, strict=True):
        status[job.task.id] = entry["status"]
        references[job.task.id] = entry["references"]
        if entry["verifier"] is not None:
            verifiers.append(Verifier.model_validate(entry["verifier"]))
    cached = sum(1 for job in jobs if job.cached)
    if only is not None:
        status = {**(read_json(ctx.workdir / "task_status.json", {}) or {}), **status}
        references = {**(read_json(ctx.workdir / "references.json", {}) or {}), **references}
    write_json(ctx.workdir / "task_status.json", status)
    write_json(ctx.workdir / "references.json", references)
    # Section 6: a Task whose Verifier does not clear D79 is "not verdicted, Verifier
    # immature", which is a Task the report leaves uncounted, not a failed build.
    ctx.record_gate(stage_gates.task_verifiers_gate(
        status,
        verifiers=len(verifiers), references=sum(1 for r in status.values() if r["reference_confirmed"]),
        passed=sum(1 for r in status.values() if r["verifier_passed"]), tasks=len(status),
        probed=probed, constraints_demoted=len(demoted),
        # D171: how many Tasks a tool actually blocks, which is how many have a differing own call.
        blocked_by_own_calls=sum(1 for r in status.values() if r.get("blocking_tools")),
        failed_recordings=sum(len(r.get("failed") or {}) for r in references.values()),
        judged=sum(1 for r in references.values() if r.get("judged")),
        # D133: the held-out Runs the pool leaves out because they did not reach the Reference's
        # End state, named per Task on the status row and counted here for the Examiner.
        did_not_reach_reference=sum(len(r.get("did_not_reach_reference") or ()) for r in status.values()),
        disagreeing=sum(1 for r in references.values()
                        if not r["references"] and (r.get("reason") or "").startswith("recordings disagree"))))
    write_json(ctx.workdir / "scorecard.json", scorecard_mod.scorecard(ctx.workdir))
    return {"verifiers": verifiers, "task_status": status, "cached": cached, "ran": len(jobs) - cached}
