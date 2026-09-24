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

The child's namespace for the generated module holds one thing beside `__name__`: the code-owned
helpers of `HELPERS`, the same ones `compile_env.load_toolkit` binds in the Runner's own process, so
a body that passes the gates here behaves the same way there.
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
from kullback.builder import effects as effects_mod
from kullback.builder import mine
from kullback.gates import tool_runs
from kullback.gates.confinement import TOOLS_CLASS, gate_confined
from kullback.gates.tool_runs import (
    MAX_SENSITIVITY_PAIRS,
    WHOLE_ANSWER,
    SensitivityPair,
    body_deterministic_gate,
    body_executes_gate,
    body_memorised_values_gate,
    body_non_trivial_gate,
    body_parses_gate,
    body_refuses_unknown_gate,
    body_replay_fidelity_gate,
    body_sensitivity_gate,
    column_differences,
)
from kullback.runner import arith, transaction
from kullback.runner.records import EntitySchema, GateResult, ToolCall, content_hash
from kullback.runner.world.loading import DB_CLASS, HELPERS, SandboxError

# The child runs under `python -I` with the environment cleared, so it cannot be relied on to import
# kullback at all. It gets the evaluator's own bytes instead, prepended to the runner script, and the
# job names which of them to bind, so the body in the subprocess calls exactly the function the body
# in the Runner's process calls.
_ARITH_SOURCE = Path(arith.__file__).read_text(encoding="utf-8")
# The same way, the Router's own snapshot and restore (runner/transaction.py), so a prefix call that
# raises is rolled back by the code that rolls back a body in a real Run, not by a second copy of it.
_TRANSACTION_SOURCE = Path(transaction.__file__).read_text(encoding="utf-8")


# The row helpers moved with the rulings (the replay ruling compares rows column by column); they
# stay importable from here for compile_env.py and synth.py.
parse_result, id_pattern_for, id_field, match_table = (
    tool_runs.parse_result, tool_runs.id_pattern_for, tool_runs.id_field, tool_runs.match_table)
key_fields, key_separator, row_key, partial_key = (
    tool_runs.key_fields, tool_runs.key_separator, tool_runs.row_key, tool_runs.partial_key)
args_text = tool_runs.args_text

# --- the minimal sandbox (see the module docstring) ---

_RUNNER_BODY = '''
import collections, copy, datetime, importlib.abc, inspect, json, math, re, socket, sys, typing
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


def _source_line(exc, source):
    # The last line of the generated module the traceback stood on. The parent cannot see the
    # child's traceback, and a bare "KeyError: 'x'" says nothing about what the body was reading;
    # the line does, and it is the body's own text, which the parent already has.
    lines = source.splitlines()
    out, tb = "", exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<generated>":
            index = tb.tb_lineno - 1
            if 0 <= index < len(lines):
                out = lines[index].strip()
        tb = tb.tb_next
    return out


def _traced(function, args, seen):
    # Which lines of the generated module this one call actually ran (D211). Only frames of the
    # generated module are traced, so the cost is the body's own lines and nothing of the world
    # model or pydantic behind it. A body that no call reaches a branch of is what the caller is
    # after, and line coverage per call is the only reading that can say so.
    def line_of(frame, event, arg):
        if event == "line":
            seen.add(frame.f_lineno)
        return line_of

    def entered(frame, event, arg):
        if frame.f_code.co_filename != "<generated>":
            return None
        seen.add(frame.f_lineno)
        return line_of

    sys.settrace(entered)
    try:
        return function(**args)
    finally:
        sys.settrace(None)


def _feed(instance, feed):
    ctx = getattr(instance, "ctx", None)
    if ctx is not None:
        ctx.feed_call(feed or {})


def _run_prefix(instance, entries):
    # The calls the same trace made before the gated one, each with its own recorded feed, on
    # the same toolkit and world. Their answers are thrown away: they are here only for the
    # state they leave. Each one is a transaction the way the Router makes every body one: the
    # toolkit's db is frozen before the call and put back if it raises, so a call that wrote and
    # then raised leaves nothing, as in a real Run. The instance holds its world as `db` only
    # (no StateView in the child), so that is the whole snapshot. A tool the module does not hold
    # is skipped, the way the recording's own refusal left no state either.
    for entry in entries:
        _feed(instance, entry.get("feed"))
        function = getattr(instance, entry["name"], None)
        if function is None:
            continue
        snapshot = snapshot_db(instance)
        try:
            function(**entry["args"])
        except Exception:
            restore_db(instance, snapshot)


def _plain(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# How many changed rows one call's diff carries before it stops listing them. Worlds are small
# and a recorded write touches one or two rows; hundreds of changed rows is a body rewriting the
# world, which the parent rules on as stated rather than by comparing each row.
DIFF_ROW_CAP = 500


def _wanted_rows(world, want):
    """The asked-about rows as the world holds them after the call, missing rows as None.

    `want` is the parent's list of [table, row] pairs, known before the run from the witnessed
    keys. Uncapped: the set is bounded by the evidence, a handful of rows per call, while the
    diff cap and its truncated rule stay exactly as they are.
    """
    world = world if isinstance(world, dict) else {}
    out = {}
    for pair in want or []:
        table, row = str(pair[0]), str(pair[1])
        rows = world.get(table)
        out.setdefault(table, {})[row] = rows.get(row) if isinstance(rows, dict) else None
    return out


def _diff_pair(before, after):
    """Rows the run left different, as table to row id to the row on either side.

    Both sides are the validated world dumped plain, so what the schema never declared is absent
    from both and can be neither agreement nor disagreement: the comparison is over the world
    the body actually saw against the one it left. A row the run added carries None before, one
    it deleted carries None after; untouched rows are absent. Past the cap the listing stops and
    the caller is told, so a body rewriting the world is ruled on as stated rather than row by
    row.
    """
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    changed, truncated = {}, False
    for table in sorted(set(before) | set(after)):
        old_rows = before.get(table) if isinstance(before.get(table), dict) else {}
        new_rows = after.get(table) if isinstance(after.get(table), dict) else {}
        for row_id in sorted(set(old_rows) | set(new_rows), key=str):
            old, new = old_rows.get(row_id), new_rows.get(row_id)
            if old == new:
                continue
            if sum(len(rows) for rows in changed.values()) >= DIFF_ROW_CAP:
                truncated = True
                break
            changed.setdefault(str(table), {})[str(row_id)] = {"before": old, "after": new}
        if truncated:
            break
    return changed, truncated


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
    # The code-owned helpers named by the job, defined by the source prepended to this file. A name
    # the job asks for and the prefix does not define is a KeyError here, which the parent reads back
    # as a module that did not load; a body may not shadow one, because it is bound before the module
    # runs and the confinement gate already counts it as bound.
    namespace = {"__name__": "generated_tools"}
    namespace.update({name: globals()[name] for name in job["helpers"]})
    # dont_inherit for the reason compile_env.load_toolkit gives: this file carries a
    # `from __future__ import annotations` of its own, and a postponed annotation leaves pydantic
    # with no module globals to resolve DomainDB's fields against.
    exec(compile(job["source"], "<generated>", "exec", dont_inherit=True), namespace)
    toolkit, db_class = namespace[job["class_name"]], namespace[job["db_class"]]
    results = []
    trace = bool(job.get("trace"))
    want_diff = bool(job.get("diff"))
    for call in job["calls"]:
        # Every call starts on the Starting state its own trace ran on: a fresh toolkit over a
        # freshly validated world, so a write cannot leave the next call standing on its output.
        # The instance is kept where a diff is asked for, so the world the call leaves can be
        # read back out of it; anywhere else nothing behind the answer is held. The copy is the
        # point: validation reuses the job world's nested structures by reference, so a body that
        # appends to a list in place would otherwise leave its output standing in the world the
        # next call sharing this entry validates against, and one call's diff would carry another
        # call's writes.
        # A call with a prefix copies too: its prefix writes into the world it runs on.
        prefix = [job["prefix_calls"][index] for index in call.get("prefix") or []]
        world = job["dbs"][call["db"]]
        if want_diff or prefix:
            world = copy.deepcopy(world)
        instance = toolkit(db_class.model_validate(world))
        _run_prefix(instance, prefix)
        # A fresh toolkit carries a fresh context, so the recorded feed of this call is laid
        # on it before the call runs: at replay and in the gates a body reads the values the
        # recording witnessed for this call, off them the seeded feed answers, counted. The
        # feed rides on the call's own entry; a call with none takes the seeded feed, and after
        # a prefix that means laying an empty feed over the last prefix call's.
        feed = call.get("feed")
        if feed or prefix:
            _feed(instance, feed)
        function = getattr(instance, call["name"])
        # The world the body actually sees, dumped before it runs: validation itself narrows the
        # world to what the schema declares, so the diff below compares what the body saw with
        # what it left rather than the raw Starting state with the narrowed one.
        before_plain = _plain(instance.db) if want_diff else None
        try:
            inspect.signature(function).bind(**call["args"])
        except TypeError as exc:
            # The arguments do not fit the signature: an answer for gate 5 to match against the
            # recorded invalid_arguments class, not a crash of the module.
            results.append({"ok": False, "error": "TypeError", "message": str(exc), "binding": True})
            continue
        # The traced run carries the lines it ran and the untraced one carries nothing extra: a
        # result is compared whole by the deterministic ruling and digested whole by the per-call
        # rows, so a field only some runs hold would be a difference the body never made.
        seen = set()
        try:
            value = _traced(function, call["args"], seen) if trace else function(**call["args"])
            result = {"ok": True, "value": _plain(value)}
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "message": str(exc),
                      "line": _source_line(exc, job["source"])}
        if trace:
            result["lines"] = sorted(seen)
        if want_diff and result.get("ok"):
            # What the call leaves behind, as the rows that part from the world it ran on, beside
            # the after state of the rows the parent asked about: a witnessed column on a row the
            # call never touched never reaches the diff, and the ruling still has to read what the
            # world holds there. A call that raised leaves nothing to compare: the executes gate
            # has already failed it and the ruling there is the one the repair loop reads.
            try:
                after_plain = _plain(instance.db)
                changed, truncated = _diff_pair(before_plain, after_plain)
                result["wanted"] = _wanted_rows(after_plain, job.get("want") or [])
            except Exception:
                changed, truncated = None, False
            result["changed"] = changed
            if truncated:
                result["changed_truncated"] = True
        results.append(result)
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump({"nonce": nonce, "results": results}, handle, default=str)


main()
'''

# What the child actually runs: the evaluator's own source, the transaction source, then the runner above.
# The evaluator comes first because its `from __future__` line must open the file.
_RUNNER = _ARITH_SOURCE + "\n" + _TRANSACTION_SOURCE + _RUNNER_BODY


# Calls without a recorded position order after positioned calls, by call id, so the result never depends on input order.
_MISSING_MSG_INDEX = 1 << 62


def calls_by_trace(calls: Iterable[ToolCall]) -> dict[str, list[ToolCall]]:
    """Recorded calls grouped by the trace that made them, for `Sandbox(trace_calls=...)`."""
    out: dict[str, list[ToolCall]] = {}
    for call in calls:
        if call.trace_id:
            out.setdefault(call.trace_id, []).append(call)
    return out


def recorded_call_parts(call: Any) -> tuple:
    """The five parts of a recorded call, for a ToolCall or a dict."""
    if isinstance(call, ToolCall):
        return (call.id, call.trace_id, call.raw_ptr, call.args or {}, call.result)
    node = call or {}
    return (node.get("id"), node.get("trace_id"), node.get("raw_ptr"), node.get("args") or {}, node.get("result"))


def context_feed_key(call: Any) -> tuple:
    """The identity of one recorded call for its tool context feed: trace, position, id.

    A call id alone does not name a call: ingestion permits an id issued again after the
    earlier call resolved, and several traces travel in one list. The triple tells those
    apart; a missing trace or id reads as "" and a missing position orders last. Two calls
    sharing the whole triple cannot be told apart and are never fed.
    """
    call_id, trace_id, raw, _, _ = recorded_call_parts(call)
    position = raw.get("msg_index") if isinstance(raw, dict) else getattr(raw, "msg_index", None)
    if not isinstance(position, int):
        position = _MISSING_MSG_INDEX
    return (trace_id or "", position, call_id or "")


class Sandbox:
    """Runs generated tool code in a subprocess only, with a timeout and no network.

    Every recorded call executes on its own Starting state (design section 6, gate 2): the shared
    world by default, and the Task's own world where `call_states` names one for that call id, which
    is how a corpus showing one row in two versions still replays (D74).

    Where `trace_calls` holds the call's trace, the call's prefix runs first on that same world:
    every earlier call of the trace (smaller msg_index), whatever tool it names, each under its
    own recorded feed, answers discarded. A recording made after an earlier write of its own trace
    only matches on the world that write left, and without the prefix the same call on two traces
    has two recorded answers and one input. A call the recording refused is left out of every
    prefix: its refusal left no state.
    """

    def __init__(self, source: str, db: dict, workdir: Path | str, class_name: str = TOOLS_CLASS,
                 db_class: str = DB_CLASS, timeout: float = 30.0,
                 call_states: Optional[dict] = None, call_tasks: Optional[dict] = None,
                 call_context: Optional[dict] = None, trace_calls: Optional[dict] = None):
        self.source, self.db, self.timeout = source, db, timeout
        self.class_name, self.db_class = class_name, db_class
        self.call_states = dict(call_states or {})  # call id -> the Starting state that call ran on
        self.call_tasks = dict(call_tasks or {})  # call id -> the Task whose trace made the call (D195)
        # Feed key -> the values that call witnessed for the tool context (its new ids, its
        # time). The child lays the feed on the fresh toolkit before the call runs, so a body
        # reads what the recording showed for this call; a call with no witnessed value takes
        # the seeded feed.
        self.call_context = dict(call_context or {})
        # Trace id -> every recorded call of that trace, any tool, in recorded order: the one
        # table each gated call's prefix is read out of.
        self.trace_calls = {trace_id: sorted(calls, key=context_feed_key)
                            for trace_id, calls in (trace_calls or {}).items() if trace_id}
        self._prefix_digests: dict[tuple, str] = {}
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

    def task_of(self, call: ToolCall) -> str:
        """The Task whose world this call runs on, as a failure line may name it.

        The map the compile stage already builds for `call_starting_states` when the caller passes
        it; otherwise the Trace the call came from, which a Task holds, so a ruling always has a
        name for the two sides of a pair and never has to quote a record to tell them apart.
        """
        return (self.call_tasks.get(call.id) if call.id else None) or call.trace_id or (call.id or "unknown")

    def prefix_of(self, call: ToolCall) -> list[ToolCall]:
        """The calls this call's trace made before it, in order; none without a trace or a position."""
        trace_id, position, _ = context_feed_key(call)
        if not trace_id or position == _MISSING_MSG_INDEX:
            return []
        return [earlier for earlier in self.trace_calls.get(trace_id, ())
                if context_feed_key(earlier)[1] < position and earlier.error is None]

    def state_key(self, call: ToolCall) -> str:
        """A comparable key for the world this call runs on, memoised per world and prefix.

        Two calls with the same arguments on two different worlds are two different inputs, and a
        body that answers them alike answered without looking at either. The same Starting state
        after a different prefix is a different world; a call with no prefix keys its world alone.
        """
        state = self._state_hash(self.state_for(call))
        prefix = self._prefix_digest(call)
        return state if prefix is None else content_hash({"state": state, "prefix": prefix})

    def _prefix_digest(self, call: ToolCall) -> Optional[str]:
        key = context_feed_key(call)
        if key not in self._prefix_digests:
            prefix = self.prefix_of(call)
            self._prefix_digests[key] = content_hash(
                [[earlier.name, earlier.args, self._feed_of(earlier)] for earlier in prefix]) if prefix else None
        return self._prefix_digests[key]

    def _feed_of(self, call: ToolCall) -> Optional[dict]:
        """The recorded feed that travels with this call, or None where it witnessed nothing."""
        feed = self.call_context.get(context_feed_key(call))
        if isinstance(feed, dict) and (feed.get("now") is not None or feed.get("new_ids")):
            return feed
        return None

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
        keys = [content_hash({"name": c.name, "args": c.args, "state": self.state_key(c)}) for c in calls]
        todo, seen = [], set()
        for key, call in zip(keys, calls, strict=False):
            if (use_cache and key in self.cache) or key in seen:
                continue
            todo.append((key, call))
            seen.add(key)
        for (key, _), result in zip(todo, self._execute([c for _, c in todo]) if todo else [], strict=False):
            self.cache[key] = result
        return [dict(self.cache[k]) for k in keys]

    def run_diff(self, calls: Iterable[ToolCall],
                 want: Iterable[tuple[str, str]] = ()) -> list[dict]:
        """Per call, the rows the body left different and whether the listing stopped early.

        One subprocess over the calls given, never memoised: the memo holds answers and this
        asks what the world looks like after each call. Each entry carries `changed` (table to
        row id to the row on either side, None where the call raised and there is no world to
        compare), `truncated` (the call changed more rows than the child lists) and `wanted`
        (the asked-about rows as the world holds them after the call, missing rows as None).
        `want` is (table, row) pairs known before the run from the witnessed keys. A call the
        body raised on reads back as no change to compare, which the transition gate counts and
        fails: a write that crashes mid-write is not one the recording shows leaving this state.
        """
        calls = list(calls)
        if not calls:
            return []
        out = []
        for result in self._execute(calls, want_diff=True, want=list(want)):
            if not result.get("ok"):
                out.append({"changed": None, "truncated": False, "wanted": {}})
                continue
            out.append({"changed": result.get("changed"),
                        "truncated": bool(result.get("changed_truncated")),
                        "wanted": result.get("wanted") or {}})
        return out

    def executed_lines(self, calls: Iterable[ToolCall]) -> set[int]:
        """Every line of the generated module the recorded calls between them actually ran (D211).

        One traced subprocess over the calls given, never memoised, since the memo holds answers
        and this asks a different question of the same run. What it is for is the other half of a
        lesson: the failures say which calls the body gets wrong, and this says which of the body's
        own branches, loops and assignments no recorded call has ever reached, which is the part a
        writer cannot see from a failure list at all. It rules on nothing; a sandbox that will not
        run answers with no lines, and the gates have already said so in their own words.
        """
        calls = list(calls)
        if not calls:
            return set()
        try:
            results = self._execute(calls, trace=True)
        except SandboxError:
            return set()
        return {int(line) for result in results for line in (result.get("lines") or [])}

    def _execute(self, calls: list[ToolCall], trace: bool = False,
                 want_diff: bool = False, want: Iterable[Any] = ()) -> list[dict]:
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
        # Only a feed that witnessed something travels: an empty one would only tell the child
        # what the seeded feed already says, and the job stays the bytes it always was for it.
        # The feed rides on its call's own entry, looked up here by feed key, so the child
        # needs no identity beyond the entry; a call absent from the mapping gets no feed.
        # A prefix call travels once in `prefix_calls`, and each entry names its prefix by index
        # into that one table, so a trace's early calls are not written once per later call.
        entries, prefix_calls, prefix_at = [], [], {}
        for call, i in zip(calls, indexes, strict=False):
            entry = {"id": call.id, "name": call.name, "args": call.args, "db": i}
            feed = self._feed_of(call)
            if feed is not None:
                entry["feed"] = feed
            prefix = []
            for earlier in self.prefix_of(call):
                key = context_feed_key(earlier)
                if key not in prefix_at:
                    prefix_at[key] = len(prefix_calls)
                    prefix_calls.append({"name": earlier.name, "args": earlier.args,
                                         "feed": self._feed_of(earlier)})
                prefix.append(prefix_at[key])
            if prefix:
                entry["prefix"] = prefix
            entries.append(entry)
        job.write_text(json.dumps({"source": self.source, "dbs": states, "db_class": self.db_class,
                                   "class_name": self.class_name, "helpers": sorted(HELPERS),
                                   "trace": bool(trace), "diff": bool(want_diff),
                                   "want": [[str(pair[0]), str(pair[1])] for pair in want],
                                   "prefix_calls": prefix_calls, "calls": entries},
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
                         rules: Any = None, readers: Any = None) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart."""
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return body_replay_fidelity_gate(calls, None, schema, label, threshold, rules, error=str(exc),
                                         readers=readers)
    return body_replay_fidelity_gate(calls, results, schema, label, threshold, rules, readers=readers)


#: The transition gate's own stage, in the `compile_tools.*` grain the per-tool rows carry.
TRANSITION_STAGE = "compile_tools.transition"


def _transition_reason(call: ToolCall, compared: dict) -> str:
    """One repair line for a call whose state change is not what the recording shows.

    The line carries the call's own arguments the way the replay gate's lines do, so the
    held-out filter reads it the same way: a failure over a call the writer was never shown is
    withheld from the prompt rather than quoted into it.
    """
    parts = []
    for miss in compared.get("missing", []):
        parts.append(f"did not move {miss['table']}.{miss['path']} on row {miss['row']}")
    for miss in compared.get("wrong", []):
        parts.append(f"left {miss['table']}.{miss['path']} on row {miss['row']} holding another "
                     "value than the recording shows")
    for miss in compared.get("extra_rows", []):
        parts.append(f"moved a row no later read shows this call moving: {miss['table']} row "
                     f"{miss['row']}")
    return f"{call.name}({args_text(call)}): the state change differs from the recording: " + "; ".join(parts)


def _wanted_keys(evidence: dict, calls: Iterable[ToolCall]) -> list[tuple[str, str]]:
    """Every witnessed (table, row) these calls may be ruled on, for the run to read back."""
    out = set()
    for call in calls:
        for row in evidence.get(call.id) or []:
            out.add((str(row.get("table")), str(row.get("row"))))
    return sorted(out)


def _rule_transition_call(call: ToolCall, entry: dict, rows: list,
                          schema: EntitySchema, rules: Any) -> tuple[str, Optional[str], int]:
    """One evidenced call's metric key, failure line or None, and accepted extra count."""
    changed = entry.get("changed")
    if changed is None:
        return ("transition_raised",
                f"{call.name}({args_text(call)}): the body raised, so no state "
                "change of it can be compared with the recording", 0)
    if entry.get("truncated"):
        return ("transition_disagrees",
                f"{call.name}({args_text(call)}): the call changed more rows than "
                "the gate reads, so the comparison is partial and the body is refused", 0)
    made = effects_mod.body_made_change(changed, schema, rules)
    compared = effects_mod.compare_transition(made, rows, rules, entry.get("wanted"))
    if compared["verdict"] == "agrees":
        return ("transition_agrees", None, len(compared["extra"]))
    return ("transition_disagrees", _transition_reason(call, compared), len(compared["extra"]))


def gate_transition(sandbox: Sandbox, calls: Iterable[ToolCall], schema: EntitySchema,
                    evidence: Optional[dict], rules: Any = None) -> GateResult:
    """6. For every recorded write, what the body changed is what the traces show it changing.

    Each recorded write call runs on its own Starting state and the rows it left different are
    compared with the change the traces witness through later reads of the same recording (the
    call's checked evidence rows, `builder/effects.replay_evidence`), under the workdir's canon
    rules the way scoring compares. A witnessed column counts as missing only where the world
    after the call does not hold the witnessed value, so a body that leaves a column alone
    because the world already held it agrees; a column left holding another value, a witnessed
    row the world has lost, and a row no later read shows the call touching each fail the body.
    Where no trace shows the after-state the gate rules unwitnessed, which never fails a body
    and is published per tool in the metrics. Error calls are not ruled here: the replay gate
    owns what a refusal answers.

    The ruling lives here rather than in kullback.gates (against D122's placement): the sandbox
    diff it reads can only be gathered beside the subprocess, and the frozen trees stay
    byte-identical this way. The result shape is the siblings': stage, pass, metrics, failures.
    """
    calls = [call for call in calls if call.id and call.error is None]
    evidence = dict(evidence or {})
    metrics = {"transition_calls": len(calls), "transition_agrees": 0, "transition_disagrees": 0,
               "transition_unwitnessed": 0, "transition_raised": 0, "transition_extras": 0}
    failures: list[str] = []
    evidenced = [call for call in calls if evidence.get(call.id)]
    metrics["transition_unwitnessed"] = len(calls) - len(evidenced)
    if not evidenced:
        return GateResult(stage=TRANSITION_STAGE, **{"pass": True}, metrics=metrics, failures=[])
    try:
        diffs = sandbox.run_diff(evidenced, _wanted_keys(evidence, evidenced))
    except SandboxError as exc:
        return GateResult(stage=TRANSITION_STAGE, **{"pass": False}, metrics=metrics,
                          failures=[f"the state runs did not come back: {exc}"])
    if len(diffs) != len(evidenced):
        # The run answered for fewer calls than it was given: ruling on the partial set would
        # pass calls nothing compared, so the gate fails closed the way it does on a truncated
        # diff.
        return GateResult(stage=TRANSITION_STAGE, **{"pass": False}, metrics=metrics,
                          failures=[f"the state runs came back for {len(diffs)} of {len(evidenced)} "
                                    "calls, so no call is ruled"])
    for call, entry in zip(evidenced, diffs, strict=False):
        key, failure, extras = _rule_transition_call(call, entry, evidence.get(call.id) or [],
                                                     schema, rules)
        metrics[key] += 1
        metrics["transition_extras"] += extras
        if failure is not None:
            failures.append(failure)
    return GateResult(stage=TRANSITION_STAGE, **{"pass": not failures}, metrics=metrics,
                      failures=failures[:5])


# --- 8. the sensitivity pairs the D195 ruling is made over ---
#
# The gate itself is `body_sensitivity_gate` in kullback.gates. What lives here is the search for
# its evidence, next to `reference_args`, which gathers the refusal probe's the same way: the calls
# are the Builder's, the worlds are the sandbox's, and only the decision belongs in the gates
# package (D122).

# At most this many worlds out of one bucket are paired with each other. Every extra world is a
# quadratic cost for a corpus that has already shown its columns at the third or fourth.
WORLDS_PER_BUCKET = 4
# What a row id becomes when two calls' arguments are compared with the row they name taken out.
ROW_ID_MASK = "\x00row"


def _argument_scalars(args: Any) -> list[tuple[str, Any]]:
    """Every scalar an argument set carries and the name it sits under, however deeply nested.

    A call can name its row at the top (`order_id="..."`) or inside an object a list holds, and the
    homing rules read a flat mapping of column name to value, so the nesting is flattened here and
    the leaf's own name is kept, which is the name those rules prefer when several ids match.
    """
    out: list[tuple[str, Any]] = []

    def walk(name: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(str(key), inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                walk(name, inner)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool) and value != "":
            out.append((name, value))

    for key, value in (args if isinstance(args, dict) else {}).items():
        walk(str(key), value)
    return out


def row_id_values(schema: EntitySchema, call: ToolCall, world_keys: set[str]) -> set:
    """Every argument value of this call that names a row, by the two rules the pinner already uses.

    A value the call's own world holds as the key of a keyed collection names a row of that
    collection; that is `reference_args`' rule, read per value rather than per argument. And the id
    column of a row the recording answered, whose value the call passed, names a row too: that is
    `mine.asked_for_id`, which is `_home_of`'s first rule and what `home_partial_result` places a
    keyless result by. The second is what finds a row id an argument carries inside a nested object,
    where the first sees nothing because the world files that row under a key of its own.
    """
    scalars = _argument_scalars(call.args)
    found = {value for _, value in scalars if isinstance(value, str) and value in world_keys}
    flat = dict(scalars)
    if not flat:
        return found
    id_names = sorted({name for table in schema.tables for name in tool_runs.key_fields(schema, table)})
    for row in _result_rows(tool_runs.parse_result(call.result)):
        column = mine.asked_for_id(call.name, row, id_names, flat)
        value = row.get(column) if column else None
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            found.add(value)
    return found


def _result_rows(value: Any) -> list[dict]:
    """The row objects a recorded result states, at the top or one list deep."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [row for row in value if isinstance(row, dict)]
    return []


def _masked_args(call: ToolCall, ids: set) -> str:
    """This call's arguments with every row id it names replaced, so two calls that differ only in
    which row they ask about share one key."""

    def mask(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: mask(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [mask(inner) for inner in value]
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and value in ids:
            return ROW_ID_MASK
        return value

    return content_hash(mask(call.args if isinstance(call.args, dict) else {}))


def _worlds_apart(sandbox: "Sandbox", calls: list[ToolCall]) -> list[tuple[ToolCall, ToolCall]]:
    """One call per world out of a bucket, each paired with every other, in a fixed order."""
    first: dict[str, ToolCall] = {}
    for call in calls:
        first.setdefault(sandbox.state_key(call), call)
    chosen = [first[key] for key in sorted(first)][:WORLDS_PER_BUCKET]
    return [(chosen[i], chosen[j]) for i in range(len(chosen)) for j in range(i + 1, len(chosen))]


def sensitivity_pairs(sandbox: "Sandbox", calls: Iterable[ToolCall], schema: EntitySchema,
                      rules: Any = None, readers: Any = None,
                      max_pairs: int = MAX_SENSITIVITY_PAIRS) -> list[SensitivityPair]:
    """Two recorded calls of one tool, made under two worlds, whose recorded results differ (D195).

    The calls are bucketed twice: by their arguments exactly, which is the stronger pair because
    nothing but the world can then account for a difference, and by their arguments with the row
    they name masked out (`row_id_values`), which is the pair a tool that takes an id per Task
    leaves. The exact buckets are drained first, so a tool that has both kinds is ruled on by the
    stronger. Within a bucket only calls that ran on different worlds are paired, since two calls on
    one world are one input and a body that answers them alike is right to.

    A call the recording answered with an error carries no columns and is not evidence here; the
    error classes are the replay ruling's to compare.
    """
    calls = [call for call in calls if call.error is None and call.result is not None]
    keys_of: dict[int, set[str]] = {}
    ids_of: dict[str, set] = {}
    world_columns: dict[int, dict] = {}
    exact: dict[str, list[ToolCall]] = {}
    masked: dict[str, list[ToolCall]] = {}
    for call in calls:
        state = sandbox.state_for(call)
        keys = keys_of.get(id(state))
        if keys is None:
            keys = keys_of[id(state)] = _collection_keys(state)
        ids = ids_of[call.id or ""] = row_id_values(schema, call, keys)
        exact.setdefault(content_hash(call.args), []).append(call)
        masked.setdefault(_masked_args(call, ids), []).append(call)
    out: list[SensitivityPair] = []
    seen: set[tuple[str, str]] = set()
    for buckets, same_args in ((exact, True), (masked, False)):
        for key in sorted(buckets):
            for one, other in _worlds_apart(sandbox, buckets[key]):
                mark = (min(one.id or "", other.id or ""), max(one.id or "", other.id or ""))
                if mark in seen:
                    continue
                seen.add(mark)
                recorded = (tool_runs.parse_result(one.result), tool_runs.parse_result(other.result))
                differences = column_differences(schema, recorded[0], recorded[1], rules, one.name, readers)
                columns = tuple(sorted(differences if same_args else _not_the_row_named(
                    differences, ids_of.get(one.id or "", set()), ids_of.get(other.id or "", set()))))
                columns = _held_by_the_world(sandbox.state_for(one), sandbox.state_for(other),
                                             columns, world_columns)
                if columns:
                    out.append(SensitivityPair(one, other, columns,
                                               (sandbox.task_of(one), sandbox.task_of(other)), same_args,
                                               _read_as_columns(one.name, recorded, readers)))
                if len(out) >= max_pairs:
                    return out
    return out


def _world_columns(state: Any) -> dict[str, frozenset]:
    """Every column name a world holds anywhere, and the set of values it holds under that name.

    Column names, not paths: a world files one row under several shapes and a recorded result names
    the column it stated, so the question this answers is the only one that can be asked of both.
    """
    out: dict[str, set] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
                else:
                    out.setdefault(str(key), set()).add(json.dumps(value, sort_keys=True, default=str))
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(state)
    return {name: frozenset(values) for name, values in out.items()}


def _held_by_the_world(one: dict, other: dict, columns: Iterable[str],
                       cache: Optional[dict] = None) -> tuple[str, ...]:
    """The columns of a difference that the two worlds themselves hold differently.

    The two-stage check has two stages: the recordings have to differ, and the state the body is
    given has to be able to tell the body which of the two it is standing in. Where the two worlds
    hold one value for the column the recordings part in, no body that reads the world can answer
    them differently, and failing one for it would be accusing the body of what the Starting state
    did: the Task is already blocked by the replay ruling, which is where an unpinned column
    belongs. Whole-answer differences carry no column name and cannot be asked, so they are kept.
    """
    cache = {} if cache is None else cache
    for state in (one, other):
        if id(state) not in cache:
            cache[id(state)] = _world_columns(state)
    left, right = cache[id(one)], cache[id(other)]
    return tuple(column for column in columns
                 if column == WHOLE_ANSWER
                 or left.get(column.rsplit(".", 1)[-1]) != right.get(column.rsplit(".", 1)[-1]))


def _read_as_columns(tool: str, recorded: tuple[Any, Any], readers: Any) -> bool:
    """Whether this tool's reader read both recorded results into columns (D176).

    It says which reading the pair was found under, so the body's own two answers are read the same
    way. Where the reader reads nothing out of the recordings the pair rests on the whole answer,
    and holding the body's answers to columns the reader can only find on their side would fail a
    body for a mismatch of readings rather than for anything it did.
    """
    if readers is None or not tool:
        return False
    if not all(isinstance(value, str) for value in recorded) or readers.table_of(tool) is None:
        return False
    return all(isinstance(readers.read(tool, value), dict) for value in recorded)


def _not_the_row_named(differences: dict, ids_one: set, ids_other: set) -> dict:
    """The differences left once the ones the two calls' own arguments account for are dropped.

    Two calls that name two rows have two recorded results whose id column differs, and a body that
    echoes its argument produces that difference without reading anything. So a column whose value
    on each side is a row id that side's own call named is no evidence; where the two calls carry
    the same arguments there is nothing to drop, and the caller does not ask.
    """
    def named(value: Any, ids: set) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool) and value in ids

    return {path: sides for path, sides in differences.items()
            if not (named(sides[0], ids_one) and named(sides[1], ids_other))}


def gate_sensitivity(sandbox: Sandbox, calls: Iterable[ToolCall], schema: EntitySchema,
                     rules: Any = None, readers: Any = None) -> GateResult:
    """8. Two calls the recordings answered differently are answered differently by the body (D195).

    The two calls of a pair have already run under their own Task's overlay for the gates before
    this one, so the sandbox answers both out of its memo and this gate starts no subprocess of its
    own on a body that got this far.
    """
    pairs = sensitivity_pairs(sandbox, calls, schema, rules, readers)
    if not pairs:
        return body_sensitivity_gate([], [], [], schema, rules, readers)
    try:
        first = sandbox.run([pair.first for pair in pairs])
        second = sandbox.run([pair.second for pair in pairs])
    except SandboxError as exc:
        return body_sensitivity_gate(pairs, None, None, schema, rules, readers, error=str(exc))
    return body_sensitivity_gate(pairs, first, second, schema, rules, readers)


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
              schema: EntitySchema, rules: Any = None, probe_refusals: bool = False,
              sig: Any = None, readers: Any = None,
              effect_values: Optional[dict] = None,
              holdout_values: Optional[dict] = None,
              transition_evidence: Optional[dict] = None) -> list[GateResult]:
    """The gates in order, stopping at the first failure so the failure localizes (EvoEnv).

    Gate 3 runs over every recorded call, not a first pair: a body that is steady on the first two
    calls and rolls a die on the third is nondeterministic, and one more subprocess is the whole cost
    of seeing it. Gate 6, the refusal probe, runs only where `probe_refusals` says so: on a write
    tool, since a read given an id nobody holds may answer with nothing and be right.

    Gate 7 (D162) sits beside the confinement gate rather than after the sandbox runs: both are
    static reads of the source, and a body that memorised the recordings is refused before a
    subprocess is started for it. `sig` is the tool's mined signature, which is what tells an enum
    member the description lists from an id a recorded call happened to carry. `effect_values`
    (D215) is what this tool's writes were seen to leave on rows they never named, so a body that
    writes down one of those values rather than computing it is refused there too.

    Gate 8 (D195) is the exception to the stopping rule, for the reason given where it is appended.

    `holdout_values` are the values the world holds only because a held-out Run witnessed them
    (`compile_env.holdout_values`). Gate 7 refuses a literal equal to one of them under its own
    rule (D220 rule 2b): the writer was shown that column masked, so a body spelling the value out
    took it from somewhere it was not entitled to.

    `transition_evidence` (G6) is the tool's per-call witnessed change (`effects.replay_evidence`),
    call id to the checked rows the traces show that call moving. The transition gate runs after
    the replay rulings passed, so a body that answers wrong is still repaired for its answer
    first; on an empty reading every call rules unwitnessed, which never fails. Where no trace
    shows the after-state the gate rules unwitnessed, and the share rides the metrics.
    """
    shown, held_out = list(shown), list(held_out)
    every = shown + held_out
    gates = [gate_parses(source)]
    if gates[-1].passed:
        gates.append(gate_confined(source))
    if gates[-1].passed:
        gates.append(body_memorised_values_gate(source, schema, sandbox.db, every, sig,
                                                readers=readers, effect_values=effect_values,
                                                holdout_values=holdout_values))
    for gate, calls, extra in ((gate_executes_on_s0, every, {}), (gate_deterministic, every, {"rules": rules}),
                               (gate_non_trivial, every, {"rules": rules})):
        if not gates[-1].passed:
            return gates
        gates.append(gate(sandbox, calls, **extra))
    if not gates[-1].passed:
        return gates
    # Gate 8 (D195) is the one gate that does not stop the chain. It has to run before the replay
    # rulings, because that is what puts a body which reads the world above one which memorises it
    # in `attempt_score`, and the chain has to go on past it, because a failed sensitivity gate is a
    # tie among every body that fails it and the replay count is what breaks that tie. Both rulings
    # are free once the body has reached here: the sandbox has already answered these calls.
    gates.append(gate_sensitivity(sandbox, every, schema, rules=rules, readers=readers))
    gates.append(gate_replay_fidelity(sandbox, shown, schema, label="shown", rules=rules, readers=readers))
    if held_out and gates[-1].passed:
        gates.append(gate_replay_fidelity(sandbox, held_out, schema, label="held_out", rules=rules,
                                          readers=readers))
    if gates[-1].passed:
        gates.append(gate_transition(sandbox, every, schema, transition_evidence, rules=rules))
    if probe_refusals and gates[-1].passed:
        gates.append(gate_refuses_unknown(sandbox, every, rules))
    return gates
