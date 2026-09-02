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

import json
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# Gate 0 (confinement) is a static check over the body and runs no subprocess, so it lives with the
# other pure gates in kullback.gates (D122); the two names the five gates below need stay imported
# here. Everything else the callers read straight out of kullback.gates.
from kullback.gates import tool_runs
from kullback.gates.confinement import TOOLS_CLASS, gate_confined
from kullback.gates.tool_runs import (
    body_deterministic_gate,
    body_executes_gate,
    body_non_trivial_gate,
    body_parses_gate,
    body_refuses_unknown_gate,
    body_replay_fidelity_gate,
)
from kullback.runner.records import EntitySchema, GateResult, ToolCall, content_hash

DB_CLASS = "DomainDB"


class SandboxError(RuntimeError):
    """The generated module did not load, or the subprocess crashed or ran out of time."""


# The row helpers moved with the rulings (the replay ruling compares rows column by column); they
# stay importable from here for compile_env.py and synth.py.
parse_result, id_pattern_for, id_field, match_table = (
    tool_runs.parse_result, tool_runs.id_pattern_for, tool_runs.id_field, tool_runs.match_table)
args_text = tool_runs.args_text

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
#
# The rulings live in kullback.gates.tool_runs (D122: the accept-or-reject decision is in the gates
# package and hashed with it). What stays here is the work: each function runs the sandbox, then
# hands the calls and what came back (or the sandbox's own error) to the ruling of the same name.

def gate_parses(source: str) -> GateResult:
    """1. The generated module is Python."""
    return body_parses_gate(source)


def gate_executes_on_s0(sandbox: Sandbox, calls: Iterable[ToolCall]) -> GateResult:
    """2. Every recorded call runs against its own Starting state without crashing the module."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return body_executes_gate(calls, None, error=str(exc))
    return body_executes_gate(calls, results)


def gate_deterministic(sandbox: Sandbox, calls: Iterable[ToolCall], rules: Any = None) -> GateResult:
    """3. Two fresh runs of the same calls give the same answers, under the customer's rules (D39)."""
    calls = list(calls)
    try:
        first, second = sandbox.run(calls, use_cache=False), sandbox.run(calls, use_cache=False)
    except SandboxError as exc:
        return body_deterministic_gate(calls, None, None, rules, error=str(exc))
    return body_deterministic_gate(calls, first, second, rules)


def gate_non_trivial(sandbox: Sandbox, calls: Iterable[ToolCall], rules: Any = None) -> GateResult:
    """4. Different arguments do not all give one constant answer, unless the recorded tool answered them that way."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return body_non_trivial_gate(calls, None, rules, error=str(exc))
    return body_non_trivial_gate(calls, results, rules)


def gate_replay_fidelity(sandbox: Sandbox, calls: Iterable[ToolCall], schema: EntitySchema,
                         label: str = "held_out", threshold: float = 1.0,
                         rules: Any = None) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return body_replay_fidelity_gate(calls, None, schema, label, threshold, rules, error=str(exc))
    return body_replay_fidelity_gate(calls, results, schema, label, threshold, rules)


def _collection_keys(state: Any) -> set[str]:
    """Every key of a keyed collection in a world: a dict of rows, keyed by what the row is called.

    A collection is a dict whose values are all dicts and that holds more than one of them, or one
    whose row carries its own key as a field. Column names never qualify, since a row has scalar
    fields beside its nested ones.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            rows = node and all(isinstance(v, dict) for v in node.values())
            if rows and (len(node) > 1 or any(isinstance(inner, str) and inner == key
                                                for key, row in node.items() for inner in row.values())):
                out.update(k for k in node if isinstance(k, str))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(state)
    return out


def reference_args(sandbox: Sandbox, calls: Iterable[ToolCall]) -> dict[str, ToolCall]:
    """Arguments whose every recorded value names a row the call's own world holds, each with a call that carries it.

    A value that is a collection key of the Starting state refers to something the world holds. An
    argument whose recorded values all do is a reference argument: the customer's tool looked the row
    up by it, and a value it cannot find is one it refused. Values are strings or lists of strings;
    a call the tool refused is not evidence, since its value may be the very one nobody holds.
    """
    hits: dict[str, list[bool]] = {}
    carriers: dict[str, ToolCall] = {}
    keys_of: dict[int, set[str]] = {}
    for call in calls:
        if call.error is not None:
            continue
        state = sandbox.state_for(call)
        keys = keys_of.get(id(state))
        if keys is None:
            keys = keys_of[id(state)] = _collection_keys(state)
        for name, value in call.args.items():
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, str) and v for v in values):
                continue
            hits.setdefault(name, []).append(all(v in keys for v in values))
            carriers.setdefault(name, call)
    return {name: carriers[name] for name, seen in hits.items() if seen and all(seen)}


def _unknown_value(value: str, keys: set[str]) -> str:
    """A value shaped like this one that the world does not hold."""
    candidates = [value[:-1] + ("0" if value[-1] != "0" else "1"), value + "0", value[::-1], "x" + value]
    return next((c for c in candidates if c not in keys), f"{value}_unknown")


def gate_refuses_unknown(sandbox: Sandbox, calls: Iterable[ToolCall], rules: Any = None) -> GateResult:
    """6. A write given a reference the world does not hold refuses it, as the recorded tool would.

    Every reference argument (`reference_args`) is probed once with a value nobody holds, on the
    same world as a recorded call; the ruling over what the body answered is `body_refuses_unknown_gate`.
    """
    calls = list(calls)
    probes = []
    for name, call in sorted(reference_args(sandbox, calls).items()):
        keys = _collection_keys(sandbox.state_for(call))
        value = call.args[name]
        unknown = ([_unknown_value(value[0], keys)] + list(value[1:]) if isinstance(value, list)
                   else _unknown_value(value, keys))
        probes.append((name, unknown, call.model_copy(update={"args": dict(call.args, **{name: unknown})})))
    if not probes:
        return body_refuses_unknown_gate(probes, None)
    try:
        results = sandbox.run([probe for _, _, probe in probes])
    except SandboxError as exc:
        return body_refuses_unknown_gate(probes, None, error=str(exc))
    return body_refuses_unknown_gate(probes, results)


def run_gates(source: str, sandbox: Sandbox, shown: Iterable[ToolCall], held_out: Iterable[ToolCall],
              schema: EntitySchema, rules: Any = None, probe_refusals: bool = False) -> list[GateResult]:
    """The gates in order, stopping at the first failure so the failure localizes (EvoEnv).

    Gate 3 runs over every recorded call, not a first pair: a body that is steady on the first two
    calls and rolls a die on the third is nondeterministic, and one more subprocess is the whole cost
    of seeing it. Gate 6, the refusal probe, runs only where `probe_refusals` says so: on a write
    tool, since a read given an id nobody holds may answer with nothing and be right.
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
    if probe_refusals and gates[-1].passed:
        gates.append(gate_refuses_unknown(sandbox, every, rules))
    return gates
