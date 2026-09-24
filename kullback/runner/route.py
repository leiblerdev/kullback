"""Answers one tool call: real first, then code, the recording table, an LLM stand-in (D262, D45, D49, D74)."""

from __future__ import annotations

import inspect
import json
import traceback
from typing import Any, Iterable, NamedTuple, Optional

from kullback.runner.canon import canonical_args
from kullback.runner.real_tools import RealTool, limit_kept_export
from kullback.runner.records import ToolCallError, ToolSig, content_hash
from kullback.runner.records import plain as _plain
from kullback.runner.state import StateView, _db_put
from kullback.runner.transaction import freeze, restore_db, snapshot_db, thaw

# The default cap on one real tool's kept workspace export: 64 MiB, overridable per
# Router through `real_export_limit` (D262).
REAL_EXPORT_LIMIT_BYTES = 64 * 1024 * 1024

STATE_PARAMS = ("state", "db", "world", "env")
# The Run stops here: a tool the Environment lists to the Candidate and cannot answer by code
# or by the recording (G28). The Candidate is never shown an answer for it, and the Verdict
# carries no reward either way.
CANNOT_ANSWER_REASON = "environment_cannot_answer"
# How much of a body fault's message the record keeps (F57).
FAULT_MESSAGE_LIMIT = 500
# A TypeError is not here: arguments the signature refuses are classed at binding, and one the
# body raises afterwards is a body fault (F57).
EXCEPTION_CLASSES = {
    KeyError: "not_found_entity",
    LookupError: "not_found_entity",
    PermissionError: "permission_denied",
    TimeoutError: "transient",
    ConnectionError: "transient",
}


class RouteResult(NamedTuple):
    """What answered the call, by which route, and whether that made the Run Assisted (D49)."""
    result: Any
    route: str
    assisted: bool = False
    error: Optional[ToolCallError] = None
    overlay_miss: Optional[list] = None  # D74 rows the Starting state could not pin (D88 env mark)


def recording_key(tool: str, args: dict, state_hash: Optional[str], rules: Any = None) -> str:
    """Section 8: a recorded result is keyed by tool, canonical args and pre-state hash.

    The args are canonicalized under the customer's own CanonRules (D39): with the module defaults
    a customer id the rules would fold two ways round keys two rows in this table, and a recorded
    call the Run should have hit is missed.
    """
    return content_hash({"tool": tool, "args": canonical_args(args or {}, rules), "state": state_hash})


def call_fingerprint(tool: str, args: dict) -> str:
    """Which recorded call a routed call is, with no world in it: the tool name and its canonical args.

    The pin sequence (D197) is served by this and not by the recording key, because the whole point
    of a time-varying row is that the world moved between two calls the recording keys apart. No
    CanonRules are read on either side, so the Builder that writes a sequence and the Router that
    serves it compute the same value without the rules having to travel with the overlay; a Run that
    reaches the row by arguments the recording never showed matches nothing and keeps the pin.
    """
    return content_hash({"tool": tool, "args": canonical_args(args or {})})


def recording(tool: str, args: dict, state_hash: str, result: Any, error: Optional[dict] = None,
              writes: Optional[dict] = None) -> dict:
    """One row of the recording table, in the shape Router indexes.

    `writes` is the row-level effect the recorded call had, as {table: {id: row}}. A write answered
    from the recording lands it in the world, so a later read of the same entity is not stale and an
    End-state atom on that write sees the change.
    """
    return {"tool": tool, "args": args, "state_hash": state_hash, "result": result, "error": error,
            "writes": writes or {}}


class Router:
    """Route order: real, then code, then recording, then the LLM stand-in (D262); a bad call gets an error, not a raise."""

    def __init__(self, env_tools_module: Any = None, recordings: Any = None, starting_state: Any = None,
                 overlay: Any = None, stand_in_model: Any = None, tool_sigs: Optional[list[ToolSig]] = None,
                 overlay_rows: Optional[dict] = None, canon_rules: Any = None,
                 synthetic_rows: Optional[Iterable[str]] = None,
                 overlay_steps: Optional[Iterable[Any]] = None,
                 real_tools: Optional[dict[str, RealTool]] = None):
        self.tools = env_tools_module
        self.real_tools = dict(real_tools or {})
        self._last_real_receipts: dict[str, list] = {}
        self.real_export_limit = REAL_EXPORT_LIMIT_BYTES
        self.real_end_states: dict[str, bytes] = {}
        # D40: a result that names a synthetic row was answered from a row no trace showed.
        self.synthetic_rows = frozenset(synthetic_rows or ())
        self.state = starting_state if isinstance(starting_state, StateView) else StateView(starting_state)
        self.state.add(overlay, overlay_rows)  # a view handed beside an overlay keeps both (D74)
        self.stand_in = stand_in_model
        self.sigs = {sig.name: sig for sig in (tool_sigs or [])}
        self.canon_rules = canon_rules  # the customer's CanonRules, which key the recording table (D39)
        self.recordings, self.unkeyed_recordings = _index_recordings(recordings, canon_rules)
        self.marked_tools = _has_tool_markers(self.tools)
        # D197: the rows whose columns the Task's own recording shows moving between two reads, by
        # fingerprint and then by how many calls of that fingerprint this Run has already made.
        self.steps = _index_steps(overlay_steps if overlay_steps is not None
                                  else getattr(overlay, "steps", None))
        self.steps_served = 0  # how many of them this Run laid in the world, for the Run's record
        self._calls_seen: dict[str, int] = {}
        self.calls_routed = 0  # every call this Run asked for, so a body can know its own turn (D282)
        self._lay_overlay_in_db()
        self.start_world = self.world()

    def _lay_overlay_in_db(self) -> None:
        """A compiled body reads self.db and cannot do the D74 lookup, so the overlay goes into that db."""
        db = getattr(self.tools, "db", None)
        for table, rows in self.state.overlay.items():
            for row_id, row in (rows or {}).items():
                _db_put(db, table, str(row_id), row)

    def world(self) -> dict:
        """The world as it stands, table by table: the generated toolkit's own db when it has one.

        A compiled toolkit keeps the world inside itself, so reading only the state view would show
        a Run that never changed anything. The Start and End state of a Run come from here (D46).
        One store and one layer: every write lands in it, so the latest write is what this shows.
        """
        db = getattr(self.tools, "db", None)
        if db is not None:
            return _plain(db)
        return {table: _plain(rows) for table, rows in self.state.shared.items() if isinstance(rows, dict)}

    def state_hash(self) -> str:
        db = getattr(self.tools, "db", None)
        return self.state.hash() if db is None else content_hash({"view": self.state.hash(), "db": _plain(db)})

    def _apply(self, entry: dict) -> None:
        """Land a recorded write's rows in the world the next call reads: the view and the toolkit db."""
        db = getattr(self.tools, "db", None)
        for table, rows in (entry.get("writes") or {}).items():
            for row_id, row in (rows or {}).items():
                self.state.put(table, row_id, row)
                _db_put(db, table, str(row_id), row)

    def route(self, name: str, args: Optional[dict] = None, requestor: str = "assistant") -> RouteResult:
        args = dict(args or {})
        self.calls_routed += 1
        if not self._may_call(name, requestor):
            # D164: the recording answered this tool for other callers and refused this one, so the
            # Run refuses it too, in the same class, before any code, recording or stand-in is asked
            # and without touching the world.
            return self._error(name, "tool_not_found", f"no tool named {name} for the {requestor}")
        if name in self.real_tools:
            # D262: a Real tool runs for real before any imitation, and never touches the
            # in-memory world, so the pin bookkeeping below does not see the call either.
            return self._real(name, args)
        self._advance(name, args)
        function = getattr(self.tools, name, None) if self.tools is not None else None
        if self._is_tool(name, function):
            return self._code(name, function, args)
        # state_hash walks the whole toolkit db, so it is only worth computing when a recording
        # could answer at all.
        entry = (self.recordings.get(recording_key(name, args, self.state_hash(), self.canon_rules))
                 if self.recordings else None)
        if entry is not None:
            error = _error_record(entry.get("error"))
            if error is None:
                self._apply(entry)  # a recorded write has to change the world, not only answer
            return RouteResult(entry.get("result"), "recording", False, error, self._misses())
        if self._listed_for(name, requestor):
            # G28: the Environment lists this tool to this caller and has neither code nor a
            # recording for it, so the Run ends here. A model stand-in never answers a listed
            # tool, and the Candidate is never shown a made up answer.
            return self._cannot_answer(name)
        if self.stand_in is not None:
            return self._stand_in(name, args)
        return self._error(name, "tool_not_found", f"no tool named {name}")

    def _advance(self, name: str, args: dict) -> None:
        """D197: serve the nth read of a time-varying row the nth value its Task's recording holds.

        A row the customer's live system moves between two reads with no write between them (a
        status that changes, a reading that is taken again, a queue that shortens) cannot be one
        pinned version: the first read matches and every later one does not. The pinner writes what
        each of the Task's own reads saw as a sequence, and this lays the next value in the world
        before the call is answered, so a body that reads the world answers what was recorded.

        Only the columns that moved are laid, so a column nothing moved keeps its single pin, and
        only a call whose fingerprint the recording holds moves anything, so a Run that reads the
        row some other way is answered from the pin. Once a fingerprint's sequence is spent nothing
        more is laid and the last value stands, which is what a system that stopped changing does.
        A write is untouched by this: `_apply` lands the write's own rows and a later read sees
        those, because the write's value is laid after any step of the same call.
        """
        if not self.steps:
            return
        fingerprint = call_fingerprint(name, args)
        index = self._calls_seen.get(fingerprint, 0)
        self._calls_seen[fingerprint] = index + 1
        db = getattr(self.tools, "db", None)
        for step in (self.steps.get(fingerprint) or {}).get(index) or ():
            row = dict(step.values)
            if not row:
                continue
            self.state.put(step.table, step.id, row)
            _db_put(db, step.table, str(step.id), row)
            self.steps_served += 1

    def _may_call(self, name: str, requestor: str) -> bool:
        """D164: whether this caller is one the mined tool answers. An unmined name is nobody's."""
        sig = self.sigs.get(name)
        if sig is None:
            return True
        return requestor in (getattr(sig, "callers", None) or ["assistant"])

    def _is_tool(self, name: str, function: Any) -> bool:
        """D45: only the customer's own tools run. A public helper on the toolkit is not one of them."""
        if not callable(function) or name.startswith("_"):
            return False
        if self.sigs:
            return name in self.sigs
        return not self.marked_tools or getattr(function, "__tool_type__", None) is not None

    def _misses(self) -> Optional[list]:
        """D88: a Starting state the Builder could not pin is an environment mark on every call of the Run."""
        return list(self.state.overlay_misses) or None

    def _real(self, name: str, args: dict) -> RouteResult:
        """D262: run the tool for real in its container, before any imitation.

        The container is the world for that tool: the in-memory db world is untouched and
        `state_hash` stays what it is. A command the world ran is an ordinary result,
        whatever its exit code. But a world that could not run the call at all (a start
        or exec failure, a timeout, an unavailable runtime, an export or output limit
        refused) is an Environment failure, never the Candidate's answer: the outcome
        carries `world_answered` False and the Run ends here as `cannot_answer` with
        reason `environment_cannot_answer`, exactly as for a tool no code can answer,
        with the tool name and the world's own failure class and message on the record.
        """
        outcome = self.real_tools[name].call(args)
        self._last_real_receipts[name] = list(outcome.receipts or [])
        if not outcome.world_answered:
            error = ToolCallError(class_="cannot_answer", payload={
                "tool": name, "reason": CANNOT_ANSWER_REASON,
                "world_error_class": outcome.error_class, "world_error": outcome.error_message},
                encoding="json", classified_by="code")
            return RouteResult(None, "cannot_answer", False, error, self._misses())
        if outcome.error_class is None:
            return RouteResult(outcome.result, "real", False, None, self._misses())
        return self._error(name, outcome.error_class, outcome.error_message or outcome.error_class,
                           route="real")

    def last_real_receipts(self, name: str) -> list:
        """The receipts of the latest real call of that tool, oldest first (D262).

        The result stays the terminal text; the exit codes live here for the record, so a
        later round can read them without re-running. Empty before any call of the tool.
        """
        return list(self._last_real_receipts.get(name) or [])

    def close_real(self) -> list[tuple[str, str]]:
        """Release every opened real world and keep each export, whatever fails (D262).

        Every tool is closed even when one close raises, and nothing is raised: failures
        come back as (tool name, message) pairs, so a failing close never masks the Run's
        own outcome. Each opened tool's workspace is exported once before its close under
        `real_export_limit` and kept for `real_end_state`; a failed export is listed too.
        """
        failures: list[tuple[str, str]] = []
        for name, tool in self.real_tools.items():
            if tool.opened:
                try:
                    self.real_end_states[name] = tool.end_state(self.real_export_limit)
                except Exception as exc:
                    failures.append((name, _message_of(exc)))
            try:
                tool.close()
            except Exception as exc:
                failures.append((name, _message_of(exc)))
        return failures

    def real_end_state(self, name: str, limit_bytes: int) -> bytes:
        """One real tool's workspace bytes: the live export while open, the kept bytes after close (D262).

        The kept bytes obey the caller's limit the same way the live export does: over the
        limit is the same refusal the open path gives, never a silent truncation and never a
        larger payload. A tool never called has nothing to export and answers empty bytes.
        """
        tool = self.real_tools[name]
        if tool.opened:
            return tool.end_state(limit_bytes)
        return limit_kept_export(self.real_end_states.get(name, b""), limit_bytes)

    def _code(self, name: str, function: Any, args: dict) -> RouteResult:
        begin_call = getattr(getattr(self.tools, "ctx", None), "begin_call", None)
        if begin_call is not None:
            begin_call(self.calls_routed)
        try:
            _bind(function, self.state, args)
        except TypeError as exc:
            # F57: arguments the signature cannot take are the caller's mistake, checked before
            # the body runs, so a TypeError the body raises later is never read as one.
            return self._error(name, "invalid_arguments", _message_of(exc))
        snapshot = _snapshot_world(self.tools, self.state)
        try:
            result = _call(function, self.state, args)
            return RouteResult(result, "code", self._reads_synthetic(result, args), None, self._misses())
        except Exception as exc:  # every call is a transaction: a body that raises leaves the world as it found it
            _restore_world(self.tools, self.state, snapshot)
            if type(exc) is ValueError or _is_pythons(exc):
                # A deliberate refusal is exactly ValueError, which is what the body-writing prompt
                # tells bodies to raise. A subclass (a validation failure, a decode error, a unicode
                # error) is the body failing at its own work, so it is a fault, not the customer's answer.
                error_class = _class_of(exc)
                sample = _corpus_error(self.sigs.get(name), error_class) if _is_pythons(exc) else None
                return self._error(name, error_class, _message_of(exc), sample=sample)
            return self._body_fault(name, exc, function)

    def _body_fault(self, name: str, exc: Exception, function: Any = None) -> RouteResult:
        """A body fault (G27): the body's own bug, never the customer's answer and never a business error.

        The Candidate gets a neutral unavailable answer in the tool's own error encoding. The Run
        records the tool name, the exception type, its message and the body line that raised
        (F57), so a replay row and a Run transcript say what broke, and carries the D88
        environment mark so the fault counts against the Environment and not the Candidate.
        """
        fault = type(exc).__name__
        encoding = _encoding_for(self.sigs.get(name), "body_fault")
        result: Any = "unavailable" if encoding == "text" else {"error": "unavailable", "class": "body_fault"}
        payload = {"tool": name, "fault": fault, "message": _message_of(exc)[:FAULT_MESSAGE_LIMIT],
                   **_raising_line(exc, function)}
        error = ToolCallError(class_="body_fault", payload=payload, encoding="json", classified_by="code")
        misses = self._misses() or []
        misses.append({"body_fault": name, "fault": fault})
        return RouteResult(result, "code", False, error, misses)

    def _reads_synthetic(self, result: Any, args: dict) -> bool:
        """The call named or returned a synthetic row (D40): the Run is assisted (D49)."""
        if not self.synthetic_rows:
            return False
        text = json.dumps([result, args], default=str, ensure_ascii=False)
        return any(row_id in text for row_id in self.synthetic_rows)

    def _listed_for(self, name: str, requestor: str) -> bool:
        """Whether the Environment lists this tool to this caller (G28).

        The mined tool list decides: a name it never held is a Candidate mistake and keeps
        today's refusal, while a tool it lists to this caller and cannot answer ends the Run.
        A tool listed only to other callers never reaches here, the D164 refusal above keeps it.
        """
        sig = self.sigs.get(name)
        if sig is None:
            return False
        return requestor in (getattr(sig, "callers", None) or ["assistant"])

    def _cannot_answer(self, name: str) -> RouteResult:
        """A listed tool with no code and no recording (G28): no answer, only the record.

        The result is never shown to the Candidate: loop.py stops the Run on this route before
        appending anything to the transcript. The event carries the tool name and the reason,
        and the Verdict reads the reason as an environment failure with no reward either way.
        """
        error = ToolCallError(class_="cannot_answer", payload={"tool": name, "reason": CANNOT_ANSWER_REASON},
                              encoding="json", classified_by="code")
        return RouteResult(None, "cannot_answer", False, error, self._misses())

    def _stand_in(self, name: str, args: dict) -> RouteResult:
        """D49: an LLM answers a tool with no code and no recording, and the Run is Assisted."""
        prompt = (
            f"Answer the tool call as the customer's system would. Tool: {name}. "
            f"Arguments: {json.dumps(canonical_args(args), sort_keys=True, default=str)}. "
            "Reply with the tool result only, as JSON when the tool returns JSON."
        )
        reply = self.stand_in.query([{"role": "user", "content": prompt}])
        return RouteResult(_parsed(reply.content), "llm", True, None, self._misses())

    def _error(self, name: str, error_class: str, message: str, sample: Any = None,
               route: str = "code") -> RouteResult:
        """D45: answered in the customer's own error encoding, taken from ToolSig.error_shapes.

        `sample` is the corpus's own payload for this class on this tool, used in place of the
        message when the body raised one of Python's exceptions: a KeyError's text is Python
        talking, and the recorded system said "Order not found" in its own words. Build 8 answered
        155 unknown-id lookups with `KeyError: '#W...'` beside recordings that said otherwise.
        """
        encoding = _encoding_for(self.sigs.get(name), error_class)
        if sample is not None:
            payload: Any = sample
        else:
            payload = message if encoding == "text" else {"error": message, "class": error_class}
        error = ToolCallError(class_=error_class, payload=payload, encoding=encoding, classified_by="code")
        return RouteResult(payload, route, False, error, self._misses())


def _snapshot_world(tools: Any, state: StateView) -> tuple[Any, Any, Any, Any]:
    """The world before one body runs, frozen: the toolkit db and the whole StateView.

    Two snapshots, not one: the toolkit db and the StateView shared world are different objects
    (model rows beside plain rows) and a body can move either one, so one snapshot cannot cover
    both. Frozen, not aliased: a snapshot that shared objects with the live world would move with
    it and restore nothing. No mined tool kind is trusted to skip the snapshot.
    """
    return (snapshot_db(tools), freeze(state.shared), freeze(state.overlay), freeze(state.overlay_misses))


def _restore_world(tools: Any, state: StateView, snapshot: tuple[Any, Any, Any, Any]) -> None:
    """Roll one body's writes back: the toolkit db and the StateView as the call found them.

    The whole StateView: the shared world, the overlay pins and the miss list. A body reaches
    all three through the state it is handed, so restoring only the shared world would let a
    failed body's pins and marks leak into later calls. Thawing parses the bytes, so it runs
    only here, on the rare rollback, and never on the per call path.
    """
    db_snapshot, shared_snapshot, overlay_snapshot, misses_snapshot = snapshot
    state.shared.clear()
    state.shared.update(thaw(shared_snapshot))
    state.overlay.clear()
    state.overlay.update(thaw(overlay_snapshot))
    state.overlay_misses[:] = thaw(misses_snapshot)
    restore_db(tools, db_snapshot)


def refuse_stand_in(router: Any) -> None:
    """A Run that will be scored can never be given a model stand-in (G28).

    The scoring callers call this on the Router they built: a Router carrying a stand-in is
    refused loudly instead of merely going unused, so no scored Run can ever be answered by
    a model. The stand-in class itself stays for unlisted tools off the scoring path (D49).
    """
    if getattr(router, "stand_in", None) is not None:
        raise ValueError("a scored Run cannot use a Router carrying a model stand-in (G28)")


def _has_tool_markers(tools: Any) -> bool:
    """A tau2-shaped toolkit marks its tools; where it does, an unmarked method is not a tool (D45)."""
    if tools is None:
        return False
    return any(getattr(getattr(tools, name, None), "__tool_type__", None) is not None
               for name in dir(tools) if not name.startswith("_"))


def _index_recordings(recordings: Any, rules: Any = None) -> tuple[dict, int]:
    """A list of rows becomes a keyed table; an already-keyed dict is used as it is."""
    if not recordings:
        return {}, 0
    if isinstance(recordings, dict):
        return dict(recordings), 0
    table, unkeyed = {}, 0
    for entry in recordings:
        tool = entry.get("tool") or entry.get("name")
        state_hash = entry.get("state_hash")
        if not tool or not state_hash:  # without a pre-state hash a recording cannot be trusted
            unkeyed += 1
            continue
        table[recording_key(tool, entry.get("args") or {}, state_hash, rules)] = entry
    return table, unkeyed


def _index_steps(steps: Any) -> dict[str, dict[int, list]]:
    """The Task's pin sequences by fingerprint, then by which call of that fingerprint they belong to."""
    out: dict[str, dict[int, list]] = {}
    for step in steps or ():
        out.setdefault(str(step.call), {}).setdefault(int(step.index or 0), []).append(step)
    return out


def _error_record(error: Any) -> Optional[ToolCallError]:
    if error is None:
        return None
    return error if isinstance(error, ToolCallError) else ToolCallError.model_validate(error)


def _encoding_for(sig: Optional[ToolSig], error_class: str) -> str:
    if sig is None or not sig.error_shapes:
        return "text"
    for shape in sig.error_shapes:
        if shape.class_ == error_class:
            return shape.encoding
    return sig.error_shapes[0].encoding


def _takes_state(function: Any) -> bool:
    parameters = list(inspect.signature(function).parameters)
    return bool(parameters) and parameters[0] in STATE_PARAMS


def _call(function: Any, state: StateView, args: dict) -> Any:
    """Generated tool bodies take the state view first when they name it; others take args only."""
    if _takes_state(function):
        return function(state, **args)
    return function(**args)


def _bind(function: Any, state: StateView, args: dict) -> None:
    """Raise TypeError when the call's arguments do not fit the body's signature, before it runs.

    A callable with no signature Python can read is not checked here; its call still runs.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    if _takes_state(function):
        signature.bind(state, **args)
    else:
        signature.bind(**args)


def _raising_line(exc: Exception, function: Any) -> dict:
    """The body line that raised: the innermost frame in the body's own file, else the innermost frame.

    Empty when the exception carries no traceback. `code` is the source text when the file is readable.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return {}
    own_file = getattr(getattr(inspect.unwrap(function), "__code__", None), "co_filename", None) if function else None
    own = [frame for frame in frames if frame.filename == own_file]
    frame = (own or frames)[-1]
    out: dict = {"line": frame.lineno}
    if frame.line:
        out["code"] = frame.line.strip()
    return out


def _is_pythons(exc: Exception) -> bool:
    """A builtin exception the classifier maps by type carries Python's message, not the customer's."""
    return any(isinstance(exc, exc_type) for exc_type in EXCEPTION_CLASSES)


def _corpus_error(sig: Optional[ToolSig], error_class: str) -> Any:
    """The payload the corpus shows for this class on this tool, when the miner recorded one."""
    for shape in (sig.error_shapes if sig is not None else []):
        if shape.class_ == error_class and shape.sample_payload not in (None, ""):
            return shape.sample_payload
    return None


def _class_of(exc: Exception) -> str:
    for exc_type, error_class in EXCEPTION_CLASSES.items():
        if isinstance(exc, exc_type):
            return error_class
    return "business_error"


def _message_of(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _parsed(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
