"""The gates of design section 6, each a function returning a GateResult, plus the D80 scorecard,
the D89 import boundary and the RunnerVersion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from harness.runner.boundary import import_boundary_check, runner_version  # noqa: F401
from harness.runner.gate_support import MISS_REASONS, _checks_gate, _get, _n, _passed, _rate, _same, gate  # noqa: F401
from harness.runner.scorecard import FROZEN_TASKS_NAME, freeze_tasks, scorecard, task_coverage  # noqa: F401
from harness.shared.confinement import SAFE_BUILTIN_NAMES as SAFE_PREDICATE_BUILTIN_NAMES  # noqa: F401
from harness.shared.confinement import SAFE_BUILTINS as SAFE_PREDICATE_BUILTINS
from harness.shared.confinement import confine
from harness.shared.records import GateResult, canonical_json

# SAFE_PREDICATE_BUILTIN_NAMES, SAFE_PREDICATE_BUILTINS and confine() live in shared/confinement.py
# and are shared with runner/verdict.py's atom gate, so the two copies of this check cannot drift
# the way the decision log's Verification pass found them drifting once already.
# gate, _get, _passed, _same, _rate, _checks_gate, _n and MISS_REASONS now live in gate_support.py;
# import_boundary_check and runner_version live in runner/boundary.py; freeze_tasks, scorecard,
# task_coverage and FROZEN_TASKS_NAME live in runner/scorecard.py. All are re-exported here so this
# module stays the one place every gate and the scorecard are found, without three copies of the
# small helpers they are built from.
GRADER_FIELDS = ("reward_info", "evaluation_criteria", "action_checks", "nl_assertions", "trial")
TAU2_FILES = ("data_model.py", "tools.py", "db.json", "policy.md", "tasks.json")
LEAK_MIN_LENGTH = 2
D79_CHECKS = (
    "provenance_spans", "oracle_passes", "empty_fails", "plausible_wrong_fails",
    "second_path_passes", "loophole_probe_fails", "leak_check_clean", "mutation_flips",
)
VERDICT_GOLDEN_CHECKS = ("oracle_passes", "empty_fails", "plausible_wrong_fails", "two_orders_pass")
VERDICT_VERSIONS = (
    "env_id", "schema_version", "tools_version", "policy_version",
    "verifier_version", "verdict_version", "runner_version",
)


# --- ingest and mine ---

def ingest_gate(traces, second_pass: Optional[dict] = None) -> GateResult:
    """Every tool call has a result or an error, hashes hold across two passes, grader fields are gone (D66)."""
    traces = list(traces or ())
    failures, calls = [], 0
    for trace in traces:
        tid = _get(trace, "trace_id", "?")
        for call in _get(trace, "tool_calls", []) or ():
            calls += 1
            if _nothing_recorded(call):
                failures.append(f"{tid}: tool call {_get(call, 'name')} has neither a result nor an error")
        body = canonical_json(trace)
        failures += [f"{tid}: grader field {f} survived ingest" for f in GRADER_FIELDS if f'"{f}"' in body]
        stored = _get(trace, "hash") or ""
        if not stored:
            failures.append(f"{tid}: no content hash on the Trace")
        elif second_pass and second_pass.get(tid, stored) != stored:
            failures.append(f"{tid}: the content hash moved between two ingest passes")
    return gate("ingest", failures, traces=len(traces), tool_calls=calls)


def _nothing_recorded(call: Any) -> bool:
    """Nothing came back for this call.

    A tool that answers with a JSON null answered: `result is None` alone cannot tell that apart from
    a call whose tool message was never recorded, so a call that carries `has_result` is believed and
    only a call without it falls back to the older, blunter test.
    """
    if _get(call, "error") is not None:
        return False
    recorded = _get(call, "has_result")
    return not recorded if recorded is not None else _get(call, "result") is None


def mine_gate(sigs, calls=None, min_calls: int = 3) -> GateResult:
    """Each ToolSig rests on enough calls or is flagged llm, and recorded args fit the mined schema."""
    failures, by_name = [], {}
    for sig in sigs or ():
        name = _get(sig, "name")
        by_name[name] = sig
        strength = _get(sig, "evidence_strength")
        # A ToolSig that counted its calls is believed, zero included: falling back to the length of
        # the evidence list would let a signature that saw no call pass on the traces it names.
        observed = _get(strength, "call_count", 0) or 0 if strength is not None \
            else len(_get(sig, "evidence", []) or ())
        if observed < min_calls and _get(sig, "source") != "llm":
            failures.append(f"{name}: {observed} observed calls, under {min_calls}, and not flagged llm")
    for call in calls or ():
        sig = by_name.get(_get(call, "name"))
        if sig is not None:
            failures += _arg_failures(sig, _get(call, "args", {}) or {})
    return gate("mine", failures, tools=len(by_name), calls=len(list(calls or ())))


def _arg_failures(sig: Any, args: dict) -> list[str]:
    name = _get(sig, "name")
    schema = _get(sig, "args_schema", {}) or {}
    known = set(schema.get("properties") or {}) | {_get(f, "name") for f in _get(sig, "args_fields", []) or ()}
    out = [f"{name}: recorded argument {key} is not in the mined schema" for key in args if known and key not in known]
    out += [f"{name}: recorded call is missing required argument {key}"
            for key in (schema.get("required") or ()) if key not in args]
    return out


# --- the five compile-tool gates, in the order that localizes a failure ---

def parses_gate(sources: Optional[dict]) -> GateResult:
    """Gate 1: every generated tool body parses."""
    failures = []
    for name, src in (sources or {}).items():
        try:
            compile(src, f"<tool {name}>", "exec")
        except SyntaxError as exc:
            failures.append(f"{name}: does not parse: {exc.msg} on line {exc.lineno}")
    return gate("compile_tools.parses", failures, tools=len(sources or {}))


def executes_gate(outcomes: Optional[dict]) -> GateResult:
    """Gate 2: every tool executes on the Starting state."""
    failures = [f"{name}: does not execute on the Starting state: {_get(out, 'error', 'no reason given')}"
                for name, out in (outcomes or {}).items() if not _get(out, "ok", False)]
    return gate("compile_tools.executes", failures, tools=len(outcomes or {}))


def deterministic_gate(pairs: Optional[dict], canon_rules: Any = None) -> GateResult:
    """Gate 3: two runs on the same input agree after canonicalization, under the customer's rules."""
    failures = []
    for name, results in (pairs or {}).items():
        seen = list(results)
        if len(seen) < 2:
            failures.append(f"{name}: needs two runs to check determinism")
        elif not all(_same(seen[0], other, canon_rules=canon_rules) for other in seen[1:]):
            failures.append(f"{name}: two runs on the same input differ")
    return gate("compile_tools.deterministic", failures, tools=len(pairs or {}))


def non_trivial_gate(outputs: Optional[dict], canon_rules: Any = None) -> GateResult:
    """Gate 4: a tool that returns the same thing for differing inputs is a constant, not a tool."""
    failures = []
    for name, results in (outputs or {}).items():
        seen = list(results)
        if len(seen) < 2:
            failures.append(f"{name}: needs at least 2 sample outputs to tell a constant from a tool")
        elif all(_same(seen[0], other, canon_rules=canon_rules) for other in seen[1:]):
            failures.append(f"{name}: returns a constant over differing inputs")
    return gate("compile_tools.non_trivial", failures, tools=len(outputs or {}))


def replay_match(call: Any, canon_rules: Any = None) -> bool:
    """Does the rebuilt tool answer a recorded call the way the recording did (errors by shape, D51)."""
    expected_error, actual_error = _get(call, "expected_error"), _get(call, "actual_error")
    if expected_error is not None:
        return actual_error is not None and _get(expected_error, "class_") == _get(actual_error, "class_")
    return actual_error is None and _same(_get(call, "expected"), _get(call, "actual"),
                                          canon_rules=canon_rules)


def replay_fidelity_gate(calls) -> GateResult:
    """Gate 5: replay of recorded calls, success and error fidelity reported separately (R22 item 8, D80)."""
    failures, misses, per_tool = [], [], {}
    totals = {"success": [0, 0], "error": [0, 0]}
    for call in calls or ():
        kind = "error" if _get(call, "expected_error") is not None else "success"
        tool = _get(call, "tool", "?")
        slot = per_tool.setdefault(tool, {"success": {"total": 0, "matched": 0}, "error": {"total": 0, "matched": 0}})
        totals[kind][0] += 1
        slot[kind]["total"] += 1
        if replay_match(call):
            totals[kind][1] += 1
            slot[kind]["matched"] += 1
            continue
        held_out = bool(_get(call, "held_out", False))
        reason = _get(call, "reason")
        misses.append({"tool": tool, "kind": kind, "reason": reason, "held_out": held_out})
        if not held_out:
            failures.append(f"{tool}: a recorded {kind} call replays differently; D80 wants 100% on recorded calls")
        elif reason not in MISS_REASONS:
            failures.append(f"{tool}: a held-out {kind} call misses with no reason ({reason!r})")
    metrics = {kind: _rate(totals[kind][0], totals[kind][1], [m for m in misses if m["kind"] == kind])
               for kind in ("success", "error")}
    metrics["per_tool"] = per_tool
    return gate("compile_tools.replay_fidelity", failures, **metrics)


def compile_tools_gates(evidence: dict) -> list[GateResult]:
    """The five gates in order, stopping at the first failure so the failure localizes (EvoEnv)."""
    steps = ((parses_gate, "sources"), (executes_gate, "outcomes"), (deterministic_gate, "runs"),
             (non_trivial_gate, "outputs"), (replay_fidelity_gate, "calls"))
    out: list[GateResult] = []
    for func, key in steps:
        out.append(func((evidence or {}).get(key)))
        if not out[-1].passed:
            break
    return out


# --- policy, environment, simulated user ---

def policy_gate(constraints, evaluate: Optional[Callable] = None, reference_violations=()) -> GateResult:
    """Positive and negative tests pass, and no Reference's own path violates a compiled constraint (R22 1.5)."""
    check = evaluate or _run_predicate
    failures, compiled, residual = [], 0, 0
    for constraint in constraints or ():
        cid = _get(constraint, "id", "?")
        if not _get(constraint, "compiled", False):
            residual += 1
            continue
        compiled += 1
        tests = _get(constraint, "tests")
        for label, want in (("pos", True), ("neg", False)):
            for case in _get(tests, label, []) or ():
                try:
                    got = bool(check(constraint, case))
                except Exception as exc:
                    failures.append(f"{cid}: {label} case raised {type(exc).__name__}: {exc}")
                    continue
                if got is not want:
                    failures.append(f"{cid}: {label} case {canonical_json(case)} returned {got}")
    failures += [f"{_get(v, 'run_id', '?')}: the Reference's own path violates constraint "
                 f"{_get(v, 'constraint_id', '?')}" for v in reference_violations or ()]
    return gate("compile_policy", failures, compiled=compiled, residual=residual)


def predicate_confinement(source: str) -> list[str]:
    """Everything a model-written constraint predicate names that reaches outside its own case.

    Restricting `__builtins__` is not enough on its own: `check.__globals__` hands the predicate this
    module's globals and `().__class__.__base__.__subclasses__()` walks every loaded class, so a
    predicate could read or write anything the process can. policy.py certifies a predicate the same
    way when it compiles one, but a Constraint can reach this gate from disk, so it is certified
    again here. It is a name check, not a proof; it is stated as one. shared/confinement.py holds
    the actual check, the same one runner/verdict.py's atom gate uses, so the two cannot drift.
    """
    return confine(source)


def _run_predicate(constraint: Any, case: dict) -> bool:
    """Run a compiled constraint's predicate, once a static check has certified it.

    A predicate that names an import, a dunder attribute or a denied builtin is refused rather than
    run, and what does run sees a restricted `__builtins__`. Neither is a sandbox: a real one for
    model-written code is still on todo.md, and this is the static check that stands in for it.
    """
    source = _get(constraint, "predicate_src") or ""
    cid = _get(constraint, "id", "?")
    refused = predicate_confinement(source)
    if refused:
        raise ValueError(f"predicate is not confined and would run in this process: {'; '.join(refused)}")
    namespace: dict = {"__builtins__": SAFE_PREDICATE_BUILTINS}
    exec(compile(source, f"<constraint {cid}>", "exec"), namespace)  # noqa: S102
    func = namespace.get("check") or next(
        (v for k, v in reversed(list(namespace.items())) if callable(v) and not k.startswith("__")), None)
    if func is None:
        raise ValueError("predicate_src defines no function")
    return bool(func(case))


def environment_gate(environment, files_dir=None, referenced_ids=(), db_ids=(), synthetic_rows=()) -> GateResult:
    """The tau2 shape is complete and parses, db.json holds every id the traces reference, synthetic rows are tagged."""
    files = set(_get(environment, "files", {}) or {})
    failures, ids = [], set(db_ids or ())
    if files_dir is not None:
        root = Path(files_dir)
        files |= {path.name for path in root.iterdir()}
        for name in ("db.json", "tasks.json"):
            if not (root / name).is_file():
                continue
            try:
                loaded = json.loads((root / name).read_text(encoding="utf-8"))
            except ValueError as exc:
                failures.append(f"{name} does not parse, so it cannot load in tau2's harness: {exc}")
                continue
            if name == "db.json":
                ids |= _ids_in(loaded)
    failures = [f"{name} is missing from the Environment" for name in TAU2_FILES if name not in files] + failures
    failures += [f"id {i} is referenced by the traces and is not in db.json" for i in referenced_ids or ()
                 if i not in ids]
    for row in synthetic_rows or ():
        tagged = isinstance(row, dict) and (row.get("synthetic") or row.get("tag"))
        if not tagged:
            failures.append(f"synthetic row {row.get('id', row) if isinstance(row, dict) else row} is not tagged")
    return gate("build_environment", failures, files=sorted(files & set(TAU2_FILES)), db_ids=len(ids),
                synthetic_rows=len(list(synthetic_rows or ())))


def _ids_in(obj: Any) -> set:
    """Every value under a key named id or ending in _id; pass db_ids yourself when the db keys rows differently."""
    found: set = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (str, int)) and (key == "id" or str(key).endswith("_id")):
                found.add(value)
            else:
                found |= _ids_in(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= _ids_in(item)
    return found


def user_rules_gate(rules, asked_fields=(), trace_refusals=(), rerun_facts=(), canon_rules=None) -> GateResult:
    """A disclosure rule per asked fact, a refusal branch where the trace shows one, facts stable on re-run (D44)."""
    disclosed = {_get(d, "field") for d in _get(rules, "disclosure", []) or ()}
    refusals = list(_get(rules, "refusals", []) or ())
    facts = {_get(f, "field"): _get(f, "value") for f in _get(rules, "facts", []) or ()}
    failures = [f"the agent asked for {field} and there is no disclosure rule for it"
                for field in asked_fields or () if field not in disclosed]
    failures += [f"the trace shows a refusal ({r}) and the rules have no refusal branch for it"
                 for r in trace_refusals or () if r not in refusals]
    for observed in rerun_facts or ():
        for field, value in (observed or {}).items():
            if field in facts and not _same(facts[field], value, canon_rules=canon_rules):
                failures.append(f"fact {field} came back different on a re-run: {facts[field]!r} then {value!r}")
    return gate("build_user_rules", failures, facts=len(facts), disclosure=len(disclosed),
                refusals=len(refusals), incomplete_reasons=list(failures))


# --- Gate A, the verifier suite, and the gates after it ---

def oracle_replay_gate(replays, canon_rules: Any = None, equivalence: Any = None) -> GateResult:
    """Replaying a Reference's own calls reaches its End state, seed and held-out counted apart (D39, D51).

    Not wired into any pipeline stage yet; belongs in build.py's Reference-derivation stage, next to
    where the replay records themselves are produced.
    """
    splits = {name: {"runs": 0, "writes": 0, "matched": 0, "semantic_mismatches": 0} for name in ("seed", "held_out")}
    failures = []
    for replay in replays or ():
        split = splits["held_out" if _get(replay, "held_out", False) else "seed"]
        run_id = _get(replay, "run_id", "?")
        split["runs"] += 1
        for write in _get(replay, "writes", []) or ():
            split["writes"] += 1
            if _same(_get(write, "expected"), _get(write, "actual"), canon_rules=canon_rules):
                split["matched"] += 1
            else:
                failures.append(f"{run_id}: a write does not match the Reference after canonicalization")
        for read in _get(replay, "semantic_reads", []) or ():
            if not _same(_get(read, "expected"), _get(read, "actual"), "semantic",
                         canon_rules=canon_rules, equivalence=equivalence,
                         column=_get(read, "column", "")):
                split["semantic_mismatches"] += 1
                failures.append(f"{run_id}: a semantic read does not match the Reference")
    return gate("gate_a_oracle_replay", failures, **splits)


def leak_gate(texts, constants, min_length: int = LEAK_MIN_LENGTH) -> GateResult:
    """D79 check 7: nothing only the Verifier should know appears in what reaches the Candidate (D89).

    The match is on whole tokens, so the constant 25 does not fire on "250", and a constant too short
    to identify anything is skipped and named in the metrics rather than flagging every text.
    """
    joined = " ".join(str(t) for t in texts or ()).lower()
    failures, skipped = [], []
    for constant in constants or ():
        text = str(constant).strip().lower()
        if len(text) < min_length:
            skipped.append(str(constant))
            continue
        if re.search(rf"(?<![0-9a-z_]){re.escape(text)}(?![0-9a-z_])", joined):
            failures.append(f"leak: the Verifier constant {constant} appears in text the Candidate sees")
    return gate("leak_check", failures, constants=len(list(constants or ())), texts=len(list(texts or ())),
                skipped=skipped, min_length=min_length)


def verifier_gate(checks: Optional[dict]) -> GateResult:
    """The D79 suite: every check runs and every check passes before a Verifier enters the pool."""
    return _checks_gate("derive_verifier", D79_CHECKS, checks)


def verdict_golden_gate(checks: Optional[dict]) -> GateResult:
    """The golden files of section 6: oracle passes, empty fails, plausible-wrong fails, two valid orders pass.

    Not wired into any pipeline stage yet; belongs in the verdict.py release check that runs the
    golden fixtures before a new VERDICT_VERSION ships.
    """
    return _checks_gate("verdict", VERDICT_GOLDEN_CHECKS, checks)


def setup_review_gate(prominent_task_ids, reviewed_task_ids) -> GateResult:
    """Every Task prominent by D36 priority has been through the setup review (D48 check 1).

    Not wired into any pipeline stage yet; belongs in build.py's Task-authoring stage, once setup
    review sign-off has somewhere to be recorded.
    """
    reviewed = set(reviewed_task_ids or ())
    failures = [f"Task {t} is prominent by D36 priority and the setup review has not seen it"
                for t in prominent_task_ids or () if t not in reviewed]
    return gate("setup_review", failures, prominent=len(list(prominent_task_ids or ())), reviewed=len(reviewed))


def candidate_runs_gate(runs, k: int = 1, seeds=None) -> GateResult:
    """k Runs per Task, fixed seeds, and a complete JSONL each.

    Not wired into any pipeline stage yet; belongs in cli.py's `_score` (src/harness/cli.py), right
    before the Runs it loads are handed to `score`.
    """
    failures, groups = [], {}
    for run in runs or ():
        run_id = _get(run, "run_id", "?")
        groups.setdefault(_get(run, "task_id") or _get(run, "trace_id"), []).append(run)
        if not any(_get(event, "type") == "stop" for event in _get(run, "events", []) or ()):
            failures.append(f"{run_id}: the JSONL has no stop event, so the Run is incomplete")
        if _get(run, "seed") is None:
            failures.append(f"{run_id}: no seed, so the Run cannot be repeated")
    for task_id, group in groups.items():
        if len(group) < k:
            failures.append(f"{task_id}: {len(group)} Runs, under the {k} the config asks for")
        if seeds is not None and {_get(r, "seed") for r in group} != set(seeds):
            failures.append(f"{task_id}: the seeds are not the fixed set the config names")
    return gate("candidate_runs", failures, runs=len(list(runs or ())), tasks=len(groups))


def budget_gate(totals: Any, stage: str = "candidate") -> GateResult:
    """No unpriced call in a Candidate batch: an unpriced call has no cost, so the batch has no cost.

    budget.py counts a call it could not price rather than dropping it, and the report used to carry
    that count as a number beside the spend. A number nobody has to act on is not a check: a batch
    whose calls were not priced cannot be compared against the frontier on cost (D85), so it fails
    here instead (D65).
    """
    buckets = _get(totals, "stages", {}) or {}
    bucket = buckets.get(stage) if isinstance(buckets, dict) else None
    unpriced = int(_get(bucket or {}, "unpriced_calls", 0) or 0)
    calls = int(_get(bucket or {}, "calls", 0) or 0)
    failures = []
    if unpriced:
        failures.append(f"{stage}: {unpriced} of {calls} model calls were not priced, so this batch "
                        "has no cost and no cost margin against the frontier")
    return gate("budget", failures, bucket=stage, calls=calls, unpriced_calls=unpriced)


def audit_gate(samples_by_task, task_ids, min_sample: int = 1, agreement=None) -> GateResult:
    """A blind audit sample per Task, with the agreement published (D48 check 2).

    Not wired into any pipeline stage yet; belongs in cli.py's report stage, once the blind audit
    sample and its published agreement have a home to be read from.
    """
    samples = samples_by_task or {}
    failures = [f"Task {t}: {_n(samples.get(t))} Runs audited, under the {min_sample} the config asks for"
                for t in task_ids or () if _n(samples.get(t)) < min_sample]
    if agreement is None:
        failures.append("the blind audit agreement is not published (D48 check 2)")
    return gate("audit", failures, tasks=len(list(task_ids or ())), agreement=agreement)


def regrade_gate(verdicts) -> GateResult:
    """Every Verdict carries the env, verifier, verdict and runner versions it was made under (D97)."""
    failures = []
    for verdict in verdicts or ():
        run_id = _get(verdict, "run_id", "?")
        failures += [f"{run_id}: {field} is not on the Verdict" for field in VERDICT_VERSIONS
                     if not _get(verdict, field)]
    return gate("regrade", failures, verdicts=len(list(verdicts or ())))

