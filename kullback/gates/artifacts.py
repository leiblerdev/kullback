"""The gates of design section 6 over each build artifact, each a function returning a GateResult.

Traces after ingest, ToolSigs after mining, the evidence a compiled tool body produced, Constraints,
the Environment's tau2 shape, the Simulated user's rules, the D79 results of a Verifier, the Verdict
golden files, Candidate Runs, the budget and a regrade's Verdicts: one function per artifact,
reading only what it is given, calling no model (D110, D122). The fidelity bar is in
`gates/fidelity.py`, the two confinement gates in `gates/confinement.py`, the D79 suite in
`gates/verifier_suite.py` and the scorecard in `gates/scorecard.py`; this module holds the rest.
The small helpers every gate is built from (`gate`, `_get`, `_same`, `_checks_gate`) stay in
`runner/gate_support.py`, because `runner/boundary.py` builds its own ruling from them and the
Runner cannot import this package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from kullback.gates.confinement import predicate_confinement
from kullback.gates.fidelity import replay_fidelity_gate
from kullback.gates.verifier_suite import D79_STAGES
from kullback.runner.confinement import SAFE_BUILTINS as SAFE_PREDICATE_BUILTINS
from kullback.runner.gate_support import _checks_gate, _get, _n, _same, gate
from kullback.runner.records import GateResult, canonical_json

# SAFE_PREDICATE_BUILTINS lives in runner/confinement.py and is shared with runner/verdict.py's atom
# gate, so the two copies of this check cannot drift the way the decision log's Verification pass
# found them drifting once already.
GRADER_FIELDS = ("reward_info", "evaluation_criteria", "action_checks", "nl_assertions", "trial")
TAU2_FILES = ("data_model.py", "tools.py", "db.json", "policy.md", "tasks.json")
LEAK_MIN_LENGTH = 2
# The nine D79 checks by the names the suite reports them under, in the suite's own order: one list,
# so the gate cannot ask for a name the suite does not produce.
D79_CHECKS = tuple(D79_STAGES.values())
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
        carried = set(_grader_fields(trace))
        for turn in _get(trace, "turns", []) or ():
            carried |= set(_grader_fields(turn))
        for call in _get(trace, "tool_calls", []) or ():
            calls += 1
            carried |= set(_grader_fields(call)) | set(_grader_fields(_get(call, "args", {}) or {}))
            if _nothing_recorded(call):
                failures.append(f"{tid}: tool call {_get(call, 'name')} has neither a result nor an error")
        failures += [f"{tid}: grader field {f} survived ingest" for f in GRADER_FIELDS if f in carried]
        stored = _get(trace, "hash") or ""
        if not stored:
            failures.append(f"{tid}: no content hash on the Trace")
        elif second_pass and second_pass.get(tid, stored) != stored:
            failures.append(f"{tid}: the content hash moved between two ingest passes")
    return gate("ingest", failures, traces=len(traces), tool_calls=calls)


def _grader_fields(obj: Any) -> list[str]:
    """The benchmark's own sidecar keys carried on this record, by name.

    The keys are what D66 is about, and they sit on the Trace, on a recorded call or in its
    arguments where the ingest mapper can leave them. A customer's own tool answering `trial: true`
    deep inside a result is the customer's data, not a grader field, so the check reads keys at
    those three levels rather than grepping the serialized Trace.

    A `Trace`, `Turn` or `ToolCall` record refuses an unknown key at load (`Record`, extra="forbid"),
    so a grader field can only ride in on the dict form or in a call's arguments.
    """
    if isinstance(obj, dict):
        return [f for f in GRADER_FIELDS if f in obj]
    return [f for f in GRADER_FIELDS if getattr(obj, f, None) is not None]


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
    sigs, calls = list(sigs or ()), list(calls or ())
    failures, by_name = [], {}
    for sig in sigs:
        name = _get(sig, "name")
        by_name[name] = sig
        strength = _get(sig, "evidence_strength")
        # A ToolSig that counted its calls is believed, zero included: falling back to the length of
        # the evidence list would let a signature that saw no call pass on the traces it names.
        observed = _get(strength, "call_count", 0) or 0 if strength is not None \
            else len(_get(sig, "evidence", []) or ())
        if observed < min_calls and _get(sig, "source") != "llm":
            failures.append(f"{name}: {observed} observed calls, under {min_calls}, and not flagged llm")
    for call in calls:
        sig = by_name.get(_get(call, "name"))
        if sig is not None:
            failures += _arg_failures(sig, _get(call, "args", {}) or {})
    return gate("mine", failures, tools=len(by_name), calls=len(calls))


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


def compile_tools_gates(evidence: dict, canon_rules: Any = None) -> list[GateResult]:
    """The five gates in order, stopping at the first failure so the failure localizes (EvoEnv).

    Gate 5, `replay_fidelity_gate`, is the per-tool fidelity bar and lives in `gates/fidelity.py`.
    The customer's canonicalization reaches the three gates that compare values, so the sequence
    rules the way the Verdict does rather than under the module defaults.
    """
    steps = ((parses_gate, "sources", False), (executes_gate, "outcomes", False),
             (deterministic_gate, "runs", True), (non_trivial_gate, "outputs", True),
             (replay_fidelity_gate, "calls", True))
    out: list[GateResult] = []
    for func, key, compares in steps:
        found = (evidence or {}).get(key)
        out.append(func(found, canon_rules) if compares else func(found))
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


def _run_predicate(constraint: Any, case: dict) -> bool:
    """Run a compiled constraint's predicate, once a static check has certified it.

    A predicate that names an import, a dunder attribute or a denied builtin is refused rather than
    run (`gates/confinement.py`'s `predicate_confinement`), and what does run sees a restricted
    `__builtins__`. Neither is a sandbox: a real one for model-written code is still on todo.md, and
    this is the static check that stands in for it.

    Each predicate is handed a fresh copy of the allowlist. The static check does not refuse
    `__builtins__.clear()` (the name is not denied and `clear` is not a dunder), and handing the one
    module-level mapping to model-written code lets a predicate take the allowlist away from, or add
    a name to, every predicate and atom scored after it in this process.
    """
    source = _get(constraint, "predicate_src") or ""
    cid = _get(constraint, "id", "?")
    refused = predicate_confinement(source)
    if refused:
        raise ValueError(f"predicate is not confined and would run in this process: {'; '.join(refused)}")
    namespace: dict = {"__builtins__": dict(SAFE_PREDICATE_BUILTINS)}
    exec(compile(source, f"<constraint {cid}>", "exec"), namespace)  # noqa: S102
    func = namespace.get("check") or next(
        (v for k, v in reversed(list(namespace.items())) if callable(v) and not k.startswith("__")), None)
    if func is None:
        raise ValueError("predicate_src defines no function")
    return bool(func(case))


def environment_gate(environment, files_dir=None, referenced_ids=(), db_ids=(), synthetic_rows=()) -> GateResult:
    """The tau2 shape is complete and parses, db.json holds every id the traces reference, synthetic rows are tagged."""
    files = set(_get(environment, "files", {}) or {})
    referenced_ids, synthetic_rows = list(referenced_ids or ()), list(synthetic_rows or ())
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
    failures += [f"id {i} is referenced by the traces and is not in db.json" for i in referenced_ids
                 if i not in ids]
    for row in synthetic_rows:
        tagged = isinstance(row, dict) and (row.get("synthetic") or row.get("tag"))
        if not tagged:
            failures.append(f"synthetic row {row.get('id', row) if isinstance(row, dict) else row} is not tagged")
    return gate("build_environment", failures, files=sorted(files & set(TAU2_FILES)), db_ids=len(ids),
                synthetic_rows=len(synthetic_rows))


def _ids_in(obj: Any) -> set:
    """Every value under a key named id or ending in _id, and every key a collection of rows is keyed by.

    The row key is how a document db names a row whatever its id column is called: airline's
    flights are keyed by `flight_number`, and the first airline build failed this gate over three
    flights that were in db.json under keys the `_id` rule never looked at. Pass `db_ids` yourself
    when the db names rows some third way.
    """
    found: set = set()
    if isinstance(obj, dict):
        if obj and all(isinstance(v, dict) for v in obj.values()):
            found |= {key for key in obj if isinstance(key, (str, int))}
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


# --- the verifier suite's verdict, the leak check, and the gates after it ---

def leak_gate(texts, constants, min_length: int = LEAK_MIN_LENGTH) -> GateResult:
    """D79 check 7: nothing only the Verifier should know appears in what reaches the Candidate (D89).

    The match is on whole tokens, so the constant 25 does not fire on "250", and a constant too short
    to identify anything is skipped and named in the metrics rather than flagging every text. An
    underscore is part of a word here, so the constant 25 does not fire on "order_25" either: an id
    a customer's world writes that way is one token, not the number inside it.

    Not wired into any pipeline stage yet: the check the build runs is `verifier_suite._leak_gate`,
    which reads the Verifier's own constants and draws the token boundary at `isalnum`, so it does
    fire on "order_25". A caller that wires this one in has to say which boundary rules.
    """
    texts, constants = list(texts or ()), list(constants or ())
    joined = " ".join(str(t) for t in texts).lower()
    failures, skipped = [], []
    for constant in constants:
        text = str(constant).strip().lower()
        if len(text) < min_length:
            skipped.append(str(constant))
            continue
        if re.search(rf"(?<![0-9a-z_]){re.escape(text)}(?![0-9a-z_])", joined):
            failures.append(f"leak: the Verifier constant {constant} appears in text the Candidate sees")
    return gate("leak_check", failures, constants=len(constants), texts=len(texts),
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
    prominent_task_ids = list(prominent_task_ids or ())
    reviewed = set(reviewed_task_ids or ())
    failures = [f"Task {t} is prominent by D36 priority and the setup review has not seen it"
                for t in prominent_task_ids if t not in reviewed]
    return gate("setup_review", failures, prominent=len(prominent_task_ids), reviewed=len(reviewed))


def candidate_runs_gate(runs, k: int = 1, seeds=None) -> GateResult:
    """k Runs per Task, fixed seeds, and a complete JSONL each.

    Not wired into any pipeline stage yet; belongs in cli.py's `_score` (kullback/cli.py), right
    before the Runs it loads are handed to `score`.
    """
    runs = list(runs or ())
    failures, groups = [], {}
    for run in runs:
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
    return gate("candidate_runs", failures, runs=len(runs), tasks=len(groups))


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
    task_ids = list(task_ids or ())
    failures = [f"Task {t}: {_n(samples.get(t))} Runs audited, under the {min_sample} the config asks for"
                for t in task_ids if _n(samples.get(t)) < min_sample]
    if agreement is None:
        failures.append("the blind audit agreement is not published (D48 check 2)")
    return gate("audit", failures, tasks=len(task_ids), agreement=agreement)


def regrade_gate(verdicts) -> GateResult:
    """Every Verdict carries the env, verifier, verdict and runner versions it was made under (D97)."""
    verdicts = list(verdicts or ())
    failures = []
    for verdict in verdicts:
        run_id = _get(verdict, "run_id", "?")
        failures += [f"{run_id}: {field} is not on the Verdict" for field in VERDICT_VERSIONS
                     if not _get(verdict, field)]
    return gate("regrade", failures, verdicts=len(verdicts))

