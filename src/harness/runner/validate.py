"""The gates of design section 6, each a function returning a GateResult, plus the D80 scorecard, the D89 import boundary and the RunnerVersion."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from harness.shared.records import GateResult, RunnerVersion, as_dict, canonical_json, content_hash

GRADER_FIELDS = ("reward_info", "evaluation_criteria", "action_checks", "nl_assertions", "trial")
MISS_REASONS = ("our_bug", "reference_bug", "ambiguous")
TAU2_FILES = ("data_model.py", "tools.py", "db.json", "policy.md", "tasks.json")
RUNNER_FILES = ("loop.py", "route.py", "verdict.py")
FROZEN_TASKS_NAME = "tasks_frozen.json"
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
COVERAGE_TAGS = ("fact_unavailable", "overlay_miss", "reconstructed", "truncated")
# The D96 reasons a Run record can actually carry today. `overlay_miss` and `reconstructed` are read
# here but nothing writes them onto an Event yet (route.py falls through to the shared world silently,
# mine.py hands its reconstruction tag to the Builder), so the scorecard says they are not measured
# instead of letting their absence read as coverage.
MEASURED_COVERAGE_TAGS = ("fact_unavailable", "truncated")
# Nothing under runner/ or shared/ may reach the module system at runtime: a dynamic import is a way
# around the D89 boundary that no static scan can follow.
DYNAMIC_IMPORT_MODULES = ("importlib", "runpy", "pkgutil")
DYNAMIC_IMPORT_CALLS = (
    "import_module", "__import__", "spec_from_file_location", "module_from_spec",
    "run_module", "run_path", "resolve_name", "get_loader", "find_loader",
)


# --- small shared helpers ---

def gate(stage: str, failures, **metrics) -> GateResult:
    """One gate result: it passes when nothing failed."""
    failures = list(failures)
    return GateResult(stage=stage, passed=not failures, metrics=metrics, failures=failures)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off a record or off the plain dict the same record becomes in JSON."""
    if isinstance(obj, dict):
        return obj[name] if name in obj else obj.get(name.rstrip("_"), default)
    return getattr(obj, name, default)


def _passed(obj: Any) -> bool:
    """The `pass` field of a Verdict or a reference entry, under either spelling."""
    if isinstance(obj, dict):
        return bool(obj.get("pass", obj.get("passed")))
    return bool(getattr(obj, "passed", False))


def _same(a: Any, b: Any, column_class: str = "hard") -> bool:
    """Equality after canonicalization (D39), so a gate and a Verdict agree by construction."""
    try:
        from harness.shared import canon
    except ImportError:
        return content_hash(a) == content_hash(b)
    return canon.equal(a, b, column_class)


def _share(part: int, total: int) -> Optional[float]:
    """A rate, or None when there is nothing to rate; an empty sample is not 100%."""
    return None if not total else round(part / total, 4)


def _rate(total: int, matched: int, misses: list) -> dict:
    """Raw and explained side by side, with every miss and its reason (D80)."""
    explained = [m for m in misses if m.get("reason") in MISS_REASONS]
    return {
        "total": total, "matched": matched,
        "raw": _share(matched, total),
        "explained": _share(matched + len(explained), total),
        "explained_misses": len(explained),
        "unexplained": len(misses) - len(explained),
        "by_reason": dict(Counter(m["reason"] for m in explained)),
        "misses": misses,
    }


def _checks_gate(stage: str, required: tuple, results: Optional[dict]) -> GateResult:
    """A gate whose evidence is a set of named checks run elsewhere; a check not run is a failure."""
    results = results or {}
    failures = [f"{name}: {'failed' if name in results else 'not run'}" for name in required if not results.get(name)]
    return gate(stage, failures, checks={name: bool(results.get(name)) for name in required})


def _n(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, set, dict)) else int(value or 0)


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


def deterministic_gate(pairs: Optional[dict]) -> GateResult:
    """Gate 3: two runs on the same input agree after canonicalization."""
    failures = []
    for name, results in (pairs or {}).items():
        seen = list(results)
        if len(seen) < 2:
            failures.append(f"{name}: needs two runs to check determinism")
        elif not all(_same(seen[0], other) for other in seen[1:]):
            failures.append(f"{name}: two runs on the same input differ")
    return gate("compile_tools.deterministic", failures, tools=len(pairs or {}))


def non_trivial_gate(outputs: Optional[dict]) -> GateResult:
    """Gate 4: a tool that returns the same thing for differing inputs is a constant, not a tool."""
    failures = []
    for name, results in (outputs or {}).items():
        seen = list(results)
        if len(seen) < 2:
            failures.append(f"{name}: needs at least 2 sample outputs to tell a constant from a tool")
        elif all(_same(seen[0], other) for other in seen[1:]):
            failures.append(f"{name}: returns a constant over differing inputs")
    return gate("compile_tools.non_trivial", failures, tools=len(outputs or {}))


def replay_match(call: Any) -> bool:
    """Does the rebuilt tool answer a recorded call the way the recording did (errors by shape, D51)."""
    expected_error, actual_error = _get(call, "expected_error"), _get(call, "actual_error")
    if expected_error is not None:
        return actual_error is not None and _get(expected_error, "class_") == _get(actual_error, "class_")
    return actual_error is None and _same(_get(call, "expected"), _get(call, "actual"))


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


def _run_predicate(constraint: Any, case: dict) -> bool:
    """Run a compiled constraint's predicate. There is no sandbox for model-written code yet; it is on todo.md."""
    namespace: dict = {}
    exec(compile(_get(constraint, "predicate_src") or "", f"<constraint {_get(constraint, 'id', '?')}>", "exec"),
         namespace)
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


def user_rules_gate(rules, asked_fields=(), trace_refusals=(), rerun_facts=()) -> GateResult:
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
            if field in facts and not _same(facts[field], value):
                failures.append(f"fact {field} came back different on a re-run: {facts[field]!r} then {value!r}")
    return gate("build_user_rules", failures, facts=len(facts), disclosure=len(disclosed),
                refusals=len(refusals), incomplete_reasons=list(failures))


# --- Gate A, the verifier suite, and the gates after it ---

def oracle_replay_gate(replays) -> GateResult:
    """Replaying a Reference's own calls reaches its End state, seed and held-out counted apart (D39, D51)."""
    splits = {name: {"runs": 0, "writes": 0, "matched": 0, "semantic_mismatches": 0} for name in ("seed", "held_out")}
    failures = []
    for replay in replays or ():
        split = splits["held_out" if _get(replay, "held_out", False) else "seed"]
        run_id = _get(replay, "run_id", "?")
        split["runs"] += 1
        for write in _get(replay, "writes", []) or ():
            split["writes"] += 1
            if _same(_get(write, "expected"), _get(write, "actual")):
                split["matched"] += 1
            else:
                failures.append(f"{run_id}: a write does not match the Reference after canonicalization")
        for read in _get(replay, "semantic_reads", []) or ():
            if not _same(_get(read, "expected"), _get(read, "actual"), "semantic"):
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
    """The golden files of section 6: oracle passes, empty fails, plausible-wrong fails, two valid orders pass."""
    return _checks_gate("verdict", VERDICT_GOLDEN_CHECKS, checks)


def setup_review_gate(prominent_task_ids, reviewed_task_ids) -> GateResult:
    """Every Task prominent by D36 priority has been through the setup review (D48 check 1)."""
    reviewed = set(reviewed_task_ids or ())
    failures = [f"Task {t} is prominent by D36 priority and the setup review has not seen it"
                for t in prominent_task_ids or () if t not in reviewed]
    return gate("setup_review", failures, prominent=len(list(prominent_task_ids or ())), reviewed=len(reviewed))


def candidate_runs_gate(runs, k: int = 1, seeds=None) -> GateResult:
    """k Runs per Task, fixed seeds, and a complete JSONL each."""
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


def audit_gate(samples_by_task, task_ids, min_sample: int = 1, agreement=None) -> GateResult:
    """A blind audit sample per Task, with the agreement published (D48 check 2)."""
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


# --- D89 import boundary and the RunnerVersion ---

def import_boundary_check(src_root: Union[str, Path]) -> GateResult:
    """Both directions of the D89 and D91 boundary, over runner/, shared/ and builder/verifier.py.

    A dynamic import is a failure on its own: an aliased `import_module`, a module name built by
    concatenation and an `exec` string all read the same to a static scan, so the primitives are
    refused rather than their arguments inspected. Sites that run code from a value this scan cannot
    read (the Verifier atoms, the policy predicates) are listed in the metrics, not failed.
    """
    root = Path(src_root)
    if (root / "harness").is_dir():
        root = root / "harness"
    failures, files, sites = [], 0, []
    for part in ("runner", "shared"):
        directory = root / part
        for path in sorted(directory.rglob("*.py")) if directory.is_dir() else ():
            files += 1
            found, seen = _import_failures(path, part)
            failures += found
            sites += seen
    verifier = root / "builder" / "verifier.py"
    if verifier.is_file():
        files += 1
        failures += _verifier_failures(verifier)
    return gate("import_boundary", failures, files=files, dynamic_code_sites=sites)


def _import_failures(path: Path, part: str) -> tuple[list[str], list[str]]:
    where = f"{part}/{path.name}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, ValueError) as exc:
        return [f"{where} does not parse, so the D89 boundary cannot be checked on it: {exc}"], []
    except OSError as exc:
        return [f"{where} cannot be read, so the D89 boundary cannot be checked on it: {exc}"], []
    return _boundary_failures(tree, where)


def _boundary_failures(tree: ast.AST, where: str, inside: str = "") -> tuple[list[str], list[str]]:
    out: list[str] = []
    sites: list[str] = []
    for node in ast.walk(tree):
        for name, how in _imported_names(node):
            out += _boundary_line(where + inside, name, how)
        if isinstance(node, ast.Name) and node.id == "__import__":
            out.append(f"{where}{inside} uses __import__; nothing here reaches the module system at "
                       "runtime, because the D89 boundary cannot be read off such a call")
        if isinstance(node, ast.Attribute) and node.attr == "modules" and \
                isinstance(node.value, ast.Name) and node.value.id == "sys":
            out.append(f"{where}{inside} reaches sys.modules; nothing here reaches the module system "
                       "at runtime (D89)")
        if not isinstance(node, ast.Call):
            continue
        callee = _callee(node)
        if callee in DYNAMIC_IMPORT_CALLS:
            out.append(f"{where}{inside} calls {callee}; nothing here imports by name at runtime, "
                       "whatever module the call names (D89)")
            out += [line for value in _string_args(node) for line in _boundary_line(where + inside, value, "call")]
        elif callee in ("exec", "eval"):
            source = node.args[0] if node.args else None
            if isinstance(source, ast.Constant) and isinstance(source.value, str):
                out += _exec_failures(source.value, where, inside)
            else:
                sites.append(f"{where}: {callee} on a value this scan cannot read, line {node.lineno}")
    return sorted(set(out)), sites


def _exec_failures(source: str, where: str, inside: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _boundary_failures(tree, where, inside + " (inside an exec string)")[0]


def _boundary_line(where: str, name: str, how: str) -> list[str]:
    verb = "imports" if how == "import" else "names"
    if _is_builder(name):
        return [f"{where} {verb} {name}; the Runner never imports the Builder (D89)"]
    if _is_verifier(name):
        return [f"{where} {verb} {name}; nothing here reads a Verifier file (D89)"]
    if how == "import" and name.lstrip(".").split(".")[0] in DYNAMIC_IMPORT_MODULES:
        return [f"{where} imports {name}; nothing here reaches the module system at runtime, which "
                "is how an import of the Builder would step around this check (D89)"]
    return []


def _verifier_failures(path: Path) -> list[str]:
    """D91's other direction: verifier.py talks to the Runner through records and cli, never its internals."""
    where = "builder/verifier.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, ValueError, OSError) as exc:
        return [f"{where} cannot be parsed, so the D91 boundary cannot be checked on it: {exc}"]
    out = []
    for node in ast.walk(tree):
        for name, _how in _imported_names(node):
            if _is_runner_internal(name):
                out.append(f"{where} imports {name}; verifier.py asks the Runner for Runs through cli "
                           "and reads records back, it never imports Runner internals (D91)")
    return sorted(set(out))


def _callee(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _string_args(node: ast.Call) -> list[str]:
    values = list(node.args) + [keyword.value for keyword in node.keywords]
    return [v.value for v in values if isinstance(v, ast.Constant) and isinstance(v.value, str)]


def _imported_names(node: ast.AST) -> list[tuple]:
    if isinstance(node, ast.Import):
        return [(alias.name, "import") for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = "." * (node.level or 0) + (node.module or "")
        return [(base, "import")] + [(f"{base}.{alias.name}", "import") for alias in node.names]
    return []


def _is_builder(name: str) -> bool:
    parts = name.lstrip(".").split(".")
    return parts[0] == "builder" or parts[:2] == ["harness", "builder"]


def _is_verifier(name: str) -> bool:
    return name.lstrip(".").split(".")[-1] == "verifier"


def _is_runner_internal(name: str) -> bool:
    parts = name.lstrip(".").split(".")
    return parts[0] == "runner" or parts[:2] == ["harness", "runner"]


def runner_version(src_root: Union[str, Path], routing_config: Any = None,
                   created_at: Optional[str] = None, confirmed_by: Optional[str] = None) -> RunnerVersion:
    """The content hash of loop.py, route.py, verdict.py and the routing config, written by freeze-runner."""
    root = Path(src_root)
    if (root / "harness").is_dir():
        root = root / "harness"
    hashes = {}
    for name in RUNNER_FILES:
        path = root / "runner" / name
        hashes[name] = content_hash(path.read_text(encoding="utf-8")) if path.is_file() else "missing"
    config_hash = content_hash(routing_config) if routing_config is not None else None
    return RunnerVersion(
        runner_version=content_hash({"files": hashes, "routing_config": config_hash}),
        file_hashes=hashes, routing_config_hash=config_hash,
        created_at=created_at, confirmed_by=confirmed_by,
    )


# --- the D62 scorecard, as D80 and D96 leave it ---

def freeze_tasks(build_dir: Union[str, Path], tasks) -> list[str]:
    """Write the Task list this build's coverage is measured against, once, at the cluster stage (D96).

    A frozen list already on disk is returned unchanged: a later split or re-cluster must not move the
    denominator, which is exactly what D96 forbids.
    """
    path = Path(build_dir) / FROZEN_TASKS_NAME
    if path.is_file():
        return _frozen_ids(_load(Path(build_dir), FROZEN_TASKS_NAME, None)) or []
    ids = [_get(task, "id", "?") for task in tasks or ()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_ids": ids, "hash": content_hash(ids)}, indent=2, sort_keys=True),
                    encoding="utf-8")
    return ids


def _frozen_ids(data: Any) -> Optional[list[str]]:
    """The frozen Task ids, or None when this build never froze a list."""
    if data is None:
        return None
    return list(data.get("task_ids", []) if isinstance(data, dict) else data)


def scorecard(build_dir: Union[str, Path], reference_verdicts=None) -> dict:
    """Tool fidelity, Task coverage, user fact consistency and Verdict agreement, raw and explained side by side."""
    root = Path(build_dir)
    tasks = _listed(_load(root, "tasks.json", []), "tasks")
    runs = _listed(_load(root, "runs.json", []), "runs")
    verdicts = _listed(_load(root, "verdicts.json", []), "verdicts")
    reference = reference_verdicts if reference_verdicts is not None else \
        _listed(_load(root, "reference_verdicts.json", []), "verdicts")
    calls = [c for c in _listed(_load(root, "held_out_calls.json", []), "calls") if _get(c, "held_out", True)]
    fidelity = replay_fidelity_gate(calls)
    card = {
        "tool_fidelity": fidelity.metrics,
        "task_coverage": task_coverage(tasks, runs, _load(root, "task_status.json", {}) or {},
                                       frozen_ids=_frozen_ids(_load(root, FROZEN_TASKS_NAME, None))),
        "user_fact_consistency": _fact_consistency(_listed(_load(root, "user_facts.json", []), "facts")),
        "verdict_agreement": _agreement(verdicts, reference, _not_gradeable(root)),
        "policy_coverage": _load(root, "policy.json", {}) or {},
    }
    failures = list(fidelity.failures)
    failures += [f"Run {m['run_id']}: fact {m['field']} differs on a re-run with no reason"
                 for m in card["user_fact_consistency"]["misses"] if m.get("reason") not in MISS_REASONS]
    failures += [_agreement_failure(m) for m in card["verdict_agreement"]["misses"]
                 if m.get("reason") not in MISS_REASONS]
    card["gate"] = as_dict(gate(
        "scorecard", failures,
        tasks_covered=card["task_coverage"]["tasks"],
        run_weighted=card["task_coverage"]["run_weighted"],
        success_fidelity=card["tool_fidelity"]["success"]["explained"],
        error_fidelity=card["tool_fidelity"]["error"]["explained"],
        verdict_agreement=card["verdict_agreement"]["explained"],
    ))
    return card


def _agreement_failure(miss: dict) -> str:
    if miss.get("missing"):
        return f"Run {miss['run_id']}: the reference has a Verdict for this Run and we produced none"
    return f"Run {miss['run_id']}: our Verdict disagrees with the reference with no reason"


def _not_gradeable(root: Path) -> set:
    """Task or Run ids set aside as not gradeable (D49, D93); they leave the agreement denominator."""
    data = _load(root, "not_gradeable.json", [])
    if isinstance(data, dict):
        return set(data.get("task_ids", []) or ()) | set(data.get("run_ids", []) or ())
    return set(data or ())


def _load(root: Path, name: str, default: Any) -> Any:
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _listed(obj: Any, key: str) -> list:
    return list(obj.get(key, []) if isinstance(obj, dict) else obj or ())


def task_coverage(tasks: list, runs: list, status: Optional[dict] = None,
                  frozen_ids: Optional[Sequence[str]] = None,
                  measured_tags: Optional[Sequence[str]] = None) -> dict:
    """Covered Tasks over the frozen Task list, plain and Run-weighted, first failing reason attached (D96).

    Public because the report shows the two D96 headline numbers and computes them the same way the
    scorecard does; `status` carries `reference_confirmed` and `verifier_passed` per Task id, and a
    Task with no entry is uncovered, because nothing confirmed its Reference or passed its Verifier.
    `frozen_ids` is the Task list fixed at the start of the build: Tasks that appeared after it are
    listed apart instead of joining the denominator.
    """
    status = status or {}
    by_id = {_get(run, "run_id"): run for run in runs}
    by_task = {_get(task, "id", "?"): task for task in tasks}
    ids = list(frozen_ids) if frozen_ids is not None else list(by_task)
    added_later = sorted(set(by_task) - set(ids)) if frozen_ids is not None else []
    measured = tuple(measured_tags) if measured_tags is not None else MEASURED_COVERAGE_TAGS
    covered, uncovered, runs_total, runs_covered = 0, [], 0, 0
    for task_id in ids:
        task = by_task.get(task_id)
        if task is None:
            uncovered.append({"task_id": task_id, "runs": 0,
                              "reason": "the Task is on the frozen list and not in this build's Task list"})
            continue
        run_ids = list(_get(task, "run_ids", []) or ())
        runs_total += len(run_ids)
        reason = _uncovered_reason(run_ids, by_id, status.get(task_id) or {})
        if reason is None:
            covered += 1
            runs_covered += len(run_ids)
        else:
            uncovered.append({"task_id": task_id, "reason": reason, "runs": len(run_ids)})
    return {"tasks_total": len(ids), "tasks_covered": covered, "tasks": _share(covered, len(ids)),
            "runs_total": runs_total, "runs_covered": runs_covered,
            "run_weighted": _share(runs_covered, runs_total), "uncovered": uncovered,
            "added_later": added_later,
            "reasons_not_measured": [tag for tag in COVERAGE_TAGS if tag not in measured]}


def _uncovered_reason(run_ids: list, by_id: dict, status: dict) -> Optional[str]:
    if not run_ids:
        return "Task has no Runs"
    for run_id in run_ids:
        run = by_id.get(run_id)
        if run is None:
            return f"Run {run_id} was not replayed"
        counts = _get(run, "route_counts", {}) or {}
        if _get(run, "assisted", False) or int(counts.get("llm") or 0) > 0:
            return f"Run {run_id} is assisted (D49)"
        for event in _get(run, "events", []) or ():
            payload = _get(event, "payload", {}) or {}
            # The route on the event is read beside the flag: a Run whose flag was lost still shows
            # the LLM stand-in in its own events (D49).
            if _get(event, "assisted", False) or _get(event, "route") == "llm":
                return f"Run {run_id} has an assisted event (D49)"
            tags = list(payload.get("tags") or ())
            for tag in COVERAGE_TAGS:
                if payload.get(tag) or tag in tags:
                    return f"Run {run_id} hit {tag}"
    if not status.get("reference_confirmed", False):
        return "the Reference is not confirmed (D57, D93)" if "reference_confirmed" in status \
            else "no Reference confirmation is recorded for this Task (D57, D93)"
    if not status.get("verifier_passed", False):
        return "the Verifier did not pass the D79 suite" if "verifier_passed" in status \
            else "no D79 result is recorded for this Task's Verifier"
    return None


def _fact_consistency(facts: list) -> dict:
    """The Simulated user gave the same facts on the re-runs it gave in the trace (D44)."""
    total, matched, misses = 0, 0, []
    for fact in facts:
        total += 1
        if _same(_get(fact, "expected"), _get(fact, "observed")):
            matched += 1
        else:
            misses.append({"run_id": _get(fact, "run_id"), "field": _get(fact, "field"),
                           "reason": _get(fact, "reason")})
    return _rate(total, matched, misses)


def _agreement(verdicts: list, reference: list, not_gradeable=()) -> dict:
    """Our Verdict against the supplied reference set, every miss carrying a D80 reason.

    A reference Run we produced no Verdict for is a miss, not a Run that quietly leaves the
    denominator; only a Task or Run set aside as not gradeable (D49, D93) is skipped.
    """
    ours = {_get(v, "run_id"): v for v in verdicts}
    aside = set(not_gradeable or ())
    total, matched, misses = 0, 0, []
    for entry in reference:
        run_id = _get(entry, "run_id")
        if run_id in aside or _get(entry, "task_id") in aside:
            continue
        mine = ours.get(run_id)
        total += 1
        if mine is None:
            misses.append({"run_id": run_id, "ours": None, "reference": _passed(entry),
                           "reason": _get(entry, "reason"), "missing": True})
            continue
        if _passed(mine) == _passed(entry):
            matched += 1
        else:
            misses.append({"run_id": run_id, "ours": _passed(mine), "reference": _passed(entry),
                           "reason": _get(entry, "reason") or _note_reason(mine)})
    return _rate(total, matched, misses)


def _note_reason(verdict: Any) -> Optional[str]:
    for note in _get(verdict, "notes", []) or ():
        if str(note).startswith("miss_reason:"):
            return str(note).split(":", 1)[1]
    return None
