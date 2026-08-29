"""Runs a generated tool body only in a subprocess, and the five gates that check it before it is trusted.

Minimal sandbox: a generated tool body runs only in a subprocess (`python -I`, environment cleared,
a wall-clock timeout, blocked imports of the process and client modules, and socket connect cut),
never in this process. That is a blast-radius reducer, not a security boundary; a real sandbox is on
todo.md. Said plainly: the body runs with the rights of the runner that calls it, so a body that
wants to can pop the import block off `sys.meta_path`, reach `os.system`, or touch the filesystem;
in-process blocks cannot stop that and this module does not claim to. What is closed is the cheap
forgery: the parent hands the child a nonce on stdin, the child reads and closes stdin before it
executes any generated code, and a result file without that nonce is refused, so a body cannot write
its own answers and exit.
"""

from __future__ import annotations

import ast
import json
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.shared.canon import (
    canonicalize as canon,  # D39: one canonicalizer, no local variant
)
from harness.shared.records import EntitySchema, GateResult, ToolCall, content_hash

DB_CLASS = "DomainDB"
TOOLS_CLASS = "DomainTools"
CRASH_ERRORS = frozenset({"NameError", "AttributeError", "TypeError", "ImportError",
                          "ModuleNotFoundError", "IndentationError", "SyntaxError", "RecursionError"})
# What a generated module may name. The skeleton is code-owned and imports the first four; the rest
# are what a tool body plausibly needs to compute a value. `os`, `sys`, `subprocess`, `socket`,
# `pathlib`, `importlib` and everything else are not on it.
ALLOWED_IMPORTS = frozenset({"typing", "pydantic", "tau2", "data_model", "datetime", "decimal",
                             "math", "json", "re", "copy", "collections", "itertools", "functools",
                             "string", "statistics", "random", "time", "uuid"})
DENIED_BUILTINS = frozenset({"__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
                             "globals", "locals", "vars", "exit", "quit", "memoryview",
                             "getattr", "setattr", "delattr"})
# The dunders the code-owned skeleton itself writes; every other one is an object walk.
ALLOWED_DUNDERS = frozenset({"__init__", "__tool_type__", "__doc__", "__name__"})


class SandboxError(RuntimeError):
    """The generated module did not load, or the subprocess crashed or ran out of time."""


# --- reading rows out of recorded tool results: shared with compile_env.py's inverse replay ---

def parse_result(result: Any) -> Any:
    """A recorded result is a value, or the JSON text of one."""
    if isinstance(result, str) and (result.strip()[:1] in "[{"):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def columns_of(schema: EntitySchema, table: str, kind: Optional[str] = None) -> list[str]:
    """Column names of one table, all of them or only those of one class (D73)."""
    return sorted(c.name for c in schema.columns if c.table == table and (kind is None or c.class_ == kind))


def id_pattern_for(schema: EntitySchema, table: str, name: Optional[str] = None) -> Optional[str]:
    """The shape a table's ids take, under either key the schema may hold it by.

    `mine_schema` records a pattern per column, keyed `table.column`; a schema written by hand
    keys it by the table alone. Reading only the second key is what made this check dead code on
    every mined schema: the lookup missed, the pattern came back None, and every candidate row
    passed the guard whatever its id looked like.
    """
    patterns = schema.id_patterns or {}
    if name and f"{table}.{name}" in patterns:
        return patterns[f"{table}.{name}"]
    return patterns.get(table)


def id_field(schema: EntitySchema, table: str) -> Optional[str]:
    """The column holding a row's id, by the customer's own naming.

    The three name candidates first, because they are the customer's own convention where they
    apply. Then the columns the miner recorded a pattern for, which is where an id the name rule
    cannot see arrives: airline's `flights` are addressed by `flight_number`, and reading only the
    `_id` names left the table proposed and empty.
    """
    names = set(columns_of(schema, table))
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    for candidate in (f"{singular}_id", "id", f"{table}_id"):
        if candidate in names:
            return candidate
    mined = [key.split(".", 1)[1] for key in sorted(schema.id_patterns or {})
             if key.startswith(f"{table}.") and key.split(".", 1)[1] in names]
    preferred = [n for n in mined if n.startswith(singular)]
    return next(iter(preferred or mined), None) or next((n for n in sorted(names) if n.endswith("_id")), None)


def match_table(schema: EntitySchema, value: Any) -> Optional[tuple[str, str]]:
    """Which table a returned row belongs to, and its id; None when the value is not a row."""
    if not isinstance(value, dict):
        return None
    best, best_score = None, 1
    for table in sorted(schema.tables):
        name = id_field(schema, table)
        if not name or not isinstance(value.get(name), str):
            continue
        pattern = id_pattern_for(schema, table, name)
        if pattern and not re.match(pattern, value[name]):
            continue
        score = len(set(columns_of(schema, table)) & set(value))
        if score > best_score:
            best, best_score = (table, value[name]), score
    return best

# --- the minimal sandbox (see the module docstring) ---

_RUNNER = '''
import collections, datetime, importlib.abc, inspect, json, math, re, socket, sys, typing
import pydantic

BLOCKED = {"urllib", "urllib3", "requests", "httpx", "ftplib", "smtplib",
           "telnetlib", "subprocess", "multiprocessing", "webbrowser"}


class _NoNetwork(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked inside the tool sandbox: " + name)
        return None


def _cut(*args, **kwargs):
    raise OSError("the network is blocked in the tool sandbox")


def _plain(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def main():
    # The nonce arrives on stdin, which is read out and closed before any generated code runs, so a
    # body cannot read it back and forge the result file. See the module docstring for what this
    # sandbox does not do.
    nonce = sys.stdin.read().strip()
    sys.stdin.close()
    with open(sys.argv[1], encoding="utf-8") as handle:
        job = json.load(handle)
    sys.meta_path.insert(0, _NoNetwork())
    # socket itself stays importable: pydantic and the standard library reach for it. Only the
    # calls that leave the machine are cut.
    socket.socket.connect = socket.socket.connect_ex = socket.socket.bind = _cut
    socket.create_connection = _cut
    namespace = {"__name__": "generated_tools"}
    exec(compile(job["source"], "<generated>", "exec"), namespace)
    toolkit, db_class = namespace[job["class_name"]], namespace[job["db_class"]]
    results = []
    for call in job["calls"]:
        # Every call starts on the Starting state its own trace ran on: a fresh toolkit over a
        # freshly validated world, so a write cannot leave the next call standing on its output.
        function = getattr(toolkit(db_class.model_validate(job["dbs"][call["db"]])), call["name"])
        try:
            inspect.signature(function).bind(**call["args"])
        except TypeError as exc:
            # The arguments do not fit the signature: an answer for gate 5 to match against the
            # recorded invalid_arguments class, not a crash of the module.
            results.append({"ok": False, "error": "TypeError", "message": str(exc), "binding": True})
            continue
        try:
            results.append({"ok": True, "value": _plain(function(**call["args"]))})
        except Exception as exc:
            results.append({"ok": False, "error": type(exc).__name__, "message": str(exc)})
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump({"nonce": nonce, "results": results}, handle, default=str)


main()
'''


class Sandbox:
    """Runs generated tool code in a subprocess only, with a timeout and no network.

    Every recorded call executes on its own Starting state (design section 6, gate 2): the shared
    world by default, and the Task's own world where `call_states` names one for that call id, which
    is how a corpus showing one row in two versions still replays (D74).
    """

    def __init__(self, source: str, db: dict, workdir: Path | str, class_name: str = TOOLS_CLASS,
                 db_class: str = DB_CLASS, timeout: float = 30.0,
                 call_states: Optional[dict] = None):
        self.source, self.db, self.timeout = source, db, timeout
        self.class_name, self.db_class = class_name, db_class
        self.call_states = dict(call_states or {})  # call id -> the Starting state that call ran on
        # Absolute, because the subprocess is started with cwd inside this directory: a relative
        # workdir would be resolved against it a second time and every path would double. Found on
        # the first live build, where `--workdir .work-retail` made all sixteen tools fail the
        # executes gate with a run_tool.py that was not there. Every test passes tmp_path, which is
        # already absolute, so nothing caught it.
        self.dir = (Path(workdir) / "sandbox").resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        self._state_hashes: dict[int, str] = {}
        self.runner = self.dir / "run_tool.py"
        self.runner.write_text(_RUNNER, encoding="utf-8")

    def state_for(self, call: ToolCall) -> dict:
        """The world this call runs on: its own Task's, or the shared one."""
        return self.call_states.get(call.id, self.db) if call.id else self.db

    def _state_hash(self, state: dict) -> str:
        key = self._state_hashes.get(id(state))
        if key is None:
            key = self._state_hashes[id(state)] = content_hash(state)
        return key

    def run(self, calls: Iterable[ToolCall], use_cache: bool = True) -> list[dict]:
        """One result dict per call, in order; the memo is keyed by call, arguments and pre-state.

        Because every call starts on its own pre-state, two identical calls have one answer, and the
        memo returns the first result rather than whatever the world looked like after the last one.
        """
        calls = list(calls)
        keys = [content_hash({"name": c.name, "args": c.args, "state": self._state_hash(self.state_for(c))})
                for c in calls]
        todo, seen = [], set()
        for key, call in zip(keys, calls, strict=False):
            if (use_cache and key in self.cache) or key in seen:
                continue
            todo.append((key, call))
            seen.add(key)
        for (key, _), result in zip(todo, self._execute([c for _, c in todo]) if todo else [], strict=False):
            self.cache[key] = result
        return [dict(self.cache[k]) for k in keys]

    def _execute(self, calls: list[ToolCall]) -> list[dict]:
        """One subprocess for one batch of calls; a crash or a timeout is a SandboxError."""
        job, out = self.dir / "job.json", self.dir / "out.json"
        out.unlink(missing_ok=True)
        states: list[dict] = []
        indexes, at = [], {}
        for call in calls:
            key = self._state_hash(self.state_for(call))
            if key not in at:
                at[key] = len(states)
                states.append(self.state_for(call))
            indexes.append(at[key])
        nonce = secrets.token_hex(16)
        job.write_text(json.dumps({"source": self.source, "dbs": states, "db_class": self.db_class,
                                   "class_name": self.class_name,
                                   "calls": [{"name": c.name, "args": c.args, "db": i}
                                             for c, i in zip(calls, indexes, strict=False)]},
                                  default=str), encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(self.runner), str(job), str(out)],
                                  input=nonce, env={}, cwd=str(self.dir), capture_output=True,
                                  text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"timeout after {self.timeout} seconds") from exc
        if done.returncode != 0 or not out.exists():
            raise SandboxError(f"module did not load: {done.stderr.strip()[-400:]}")
        payload = json.loads(out.read_text(encoding="utf-8"))
        if payload.get("nonce") != nonce:
            raise SandboxError("the result file does not carry this run's nonce, so the runner did "
                               "not write it; the results are refused")
        return payload["results"]

# --- the five gates, in the order that localizes a failure (design section 6) ---

def _gate(stage: str, passed: bool, metrics: dict, failures: Iterable[str] = ()) -> GateResult:
    return GateResult(stage=stage, **{"pass": passed}, metrics=metrics, failures=list(failures)[:5])


def args_text(call: ToolCall) -> str:
    return json.dumps(call.args, sort_keys=True, default=str)


def gate_parses(source: str) -> GateResult:
    """1. The generated module is Python."""
    try:
        compile(source, "<generated>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return _gate("parses", False, {}, [f"line {exc.lineno}: {exc.msg}"])
    return _gate("parses", True, {"chars": len(source)})


def source_confinement(source: str, class_name: str = TOOLS_CLASS) -> list[str]:
    """Everything a model-written tool body names that reaches outside the customer's world.

    The Builder's gates run a body in the subprocess sandbox, but `load_toolkit` executes the same
    module in the Runner's own process, where a body that opens a file, imports `os` or walks
    `().__class__.__base__.__subclasses__()` runs with the Runner's rights. A real sandbox for
    model-written tool code is deferred (design section 4, "Deliberately absent"), so this is the
    static check that stands in for it: an import outside the allowlist, a denied builtin and a
    dunder attribute are refused before the module is executed anywhere. It is a name check, not a
    proof; it is stated as one.

    Only the tool methods are checked. The data model, the toolkit shim and `DomainDB.load` are
    code-owned: they are the same bytes for every customer and no model wrote them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"does not parse: {exc.msg}"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name == "__init__":
                continue
            bad += [f"{member.name} {line}" for line in _body_confinement(member)]
    return sorted(set(bad))


def _body_confinement(function: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(function):
        for name in _imported(node):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"imports {name}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") \
                and node.attr not in ALLOWED_DUNDERS:
            out.append(f"touches {node.attr}")
        elif isinstance(node, ast.Name) and node.id in DENIED_BUILTINS:
            out.append(f"uses {node.id}")
    return out


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


def gate_confined(source: str) -> GateResult:
    """0. Nothing in the module reaches past the customer's world (see `source_confinement`)."""
    failures = source_confinement(source)
    return _gate("confined", not failures, {"chars": len(source)}, failures)


def _argument_answer(call: ToolCall, result: dict) -> bool:
    """A refusal about the arguments is an answer gate 5 matches, not a crash of the module (D67).

    The customer's own logs hold calls their tool rejected for a wrong or missing argument. Replaying
    one raises TypeError where the arguments meet the signature, before any body runs, and TypeError
    is otherwise a crash. Such a call is handed on to gate 5, which matches it against the recorded
    `invalid_arguments` class the same way route.py maps TypeError to it.
    """
    return bool(result.get("binding")) or (call.error is not None and call.error.class_ == "invalid_arguments")


def gate_executes_on_s0(sandbox: Sandbox, calls: Iterable[ToolCall]) -> GateResult:
    """2. Every recorded call runs against its own Starting state without crashing the module."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("executes_on_s0", False, {"calls": len(calls)}, [str(exc)])
    crashes = [f"{c.name}({args_text(c)}) raised {r['error']}: {r['message']}"
               for c, r in zip(calls, results, strict=False)
               if not r["ok"] and r["error"] in CRASH_ERRORS and not _argument_answer(c, r)]
    return _gate("executes_on_s0", not crashes, {"calls": len(calls), "crashes": len(crashes)}, crashes)


def gate_deterministic(sandbox: Sandbox, calls: Iterable[ToolCall], rules: Any = None) -> GateResult:
    """3. Two fresh runs of the same calls give the same answers, under the customer's rules (D39)."""
    calls = list(calls)
    try:
        first, second = sandbox.run(calls, use_cache=False), sandbox.run(calls, use_cache=False)
    except SandboxError as exc:
        return _gate("deterministic", False, {"calls": len(calls)}, [str(exc)])
    differing = [c.name for c, a, b in zip(calls, first, second, strict=False) if canon(a, rules) != canon(b, rules)]
    return _gate("deterministic", not differing, {"calls": len(calls), "differing": len(differing)},
                 [f"{name} answered differently on a second run" for name in differing])


def gate_non_trivial(sandbox: Sandbox, calls: Iterable[ToolCall], rules: Any = None) -> GateResult:
    """4. Different arguments do not all give one constant answer."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("non_trivial", False, {}, [str(exc)])
    metrics = {"arg_sets": len({content_hash(c.args) for c in calls}),
               "distinct_answers": len({content_hash(canon(r, rules)) for r in results})}
    if metrics["arg_sets"] < 2:
        return _gate("non_trivial", True, dict(metrics, insufficient_evidence=True))
    trivial = metrics["distinct_answers"] < 2
    return _gate("non_trivial", not trivial, metrics,
                 ["the body answers every call the same way"] if trivial else [])


def _classify_exception(result: dict) -> str:
    """The raised exception in the D67 classes, so errors are matched by shape and not by text."""
    message, name = (result.get("message") or "").lower(), result.get("error") or ""
    if "not found" in message or "unknown" in message or name == "KeyError":
        return "not_found_entity"
    if name in ("TypeError", "ValidationError") or "invalid" in message or "must be" in message:
        return "invalid_arguments"
    if "permission" in message or "not allowed" in message or "forbidden" in message:
        return "permission_denied"
    return "business_error" if name == "ValueError" else "unknown"


def _row_pairs(schema: EntitySchema, expected: list, got: list) -> list[tuple[Any, Any]]:
    """Pair two lists of rows by id where every row on both sides carries one, else by position."""
    left = [match_table(schema, value) for value in expected]
    right = {found: value for value in got for found in [match_table(schema, value)] if found}
    if all(left) and len(right) == len(got) and all(found in right for found in left):
        return [(value, right[found]) for value, found in zip(expected, left, strict=False)]
    return list(zip(expected, got, strict=False))


def _compare(schema: EntitySchema, expected: Any, got: Any, rules: Any = None) -> tuple[bool, list[str]]:
    """Hard columns must match after canon; semantic ones are reported, not failed (D73, D84).

    A list of rows and a dict wrapping rows are walked into, so the column classes decide there too.
    Comparing a wrapped result as one canonical string would let an exempt column fail a replay that
    the same row returned on its own passes, which is the opposite of what D73 and D84 ask for.
    """
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return False, [f"list of {len(expected)} against {len(got)}"]
        ok, notes = True, []
        for one, other in _row_pairs(schema, expected, got):
            one_ok, one_notes = _compare(schema, one, other, rules)
            ok, notes = ok and one_ok, notes + one_notes
        return ok, notes
    found = match_table(schema, expected)
    if found is None and isinstance(expected, dict) and isinstance(got, dict):
        if set(expected) != set(got):
            return False, [f"keys differ: {sorted(set(expected) ^ set(got))}"]
        ok, notes = True, []
        for key in sorted(expected):
            key_ok, key_notes = _compare(schema, expected[key], got[key], rules)
            ok, notes = ok and key_ok, notes + key_notes
        return ok, notes
    if not found or not isinstance(got, dict):
        return canon(expected, rules) == canon(got, rules), []
    differs = [n for n in columns_of(schema, found[0], "hard")
               if canon(expected.get(n), rules) != canon(got.get(n), rules)]
    semantic = [f"semantic:{n}" for n in columns_of(schema, found[0], "semantic")
                if canon(expected.get(n), rules) != canon(got.get(n), rules)]
    return not differs, differs + semantic


def gate_replay_fidelity(sandbox: Sandbox, calls: Iterable[ToolCall], schema: EntitySchema,
                         label: str = "held_out", threshold: float = 1.0,
                         rules: Any = None) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return _gate("replay_fidelity", False, {"split": label}, [str(exc)])
    hits = {"success_calls": 0, "success_matches": 0, "error_calls": 0, "error_matches": 0}
    semantic, failures = 0, []
    for call, result in zip(calls, results, strict=False):
        if call.error is not None:
            hits["error_calls"] += 1
            got = _classify_exception(result) if not result["ok"] else None
            if got == call.error.class_:
                hits["error_matches"] += 1
            else:
                failures.append(f"{call.name}({args_text(call)}): expected error {call.error.class_}, got {got}")
            continue
        hits["success_calls"] += 1
        if not result["ok"]:
            failures.append(f"{call.name}({args_text(call)}): expected a result, got "
                            f"{result['error']}: {result['message']}")
            continue
        ok, differing = _compare(schema, parse_result(call.result), result["value"], rules)
        semantic += sum(1 for n in differing if n.startswith("semantic:"))
        if ok:
            hits["success_matches"] += 1
        else:
            failures.append(f"{call.name}({args_text(call)}): hard columns differ: "
                            f"{', '.join(differing) or 'value'}")
    success = hits["success_matches"] / hits["success_calls"] if hits["success_calls"] else 1.0
    errors = hits["error_matches"] / hits["error_calls"] if hits["error_calls"] else 1.0
    metrics = dict(hits, split=label, success_fidelity=success, error_fidelity=errors,
                   semantic_differences=semantic)
    return _gate("replay_fidelity", success >= threshold and errors >= threshold, metrics, failures)


def run_gates(source: str, sandbox: Sandbox, shown: Iterable[ToolCall], held_out: Iterable[ToolCall],
              schema: EntitySchema, rules: Any = None) -> list[GateResult]:
    """The five gates in order, stopping at the first failure so the failure localizes (EvoEnv).

    Gate 3 runs over every recorded call, not a first pair: a body that is steady on the first two
    calls and rolls a die on the third is nondeterministic, and one more subprocess is the whole cost
    of seeing it.
    """
    shown, held_out = list(shown), list(held_out)
    every = shown + held_out
    gates = [gate_parses(source)]
    if gates[-1].passed:
        gates.append(gate_confined(source))
    for gate, calls, extra in ((gate_executes_on_s0, every, {}), (gate_deterministic, every, {"rules": rules}),
                               (gate_non_trivial, every, {"rules": rules})):
        if not gates[-1].passed:
            return gates
        gates.append(gate(sandbox, calls, **extra))
    if not gates[-1].passed:
        return gates
    gates.append(gate_replay_fidelity(sandbox, shown, schema, label="shown", rules=rules))
    if held_out and gates[-1].passed:
        gates.append(gate_replay_fidelity(sandbox, held_out, schema, label="held_out", rules=rules))
    return gates
