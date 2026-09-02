"""Answers one tool call: code first, then the recording table, then an LLM stand-in (D45, D49, D74)."""

from __future__ import annotations

import inspect
import json
from typing import Any, Iterable, NamedTuple, Optional

from kullback.runner.canon import canonical_args
from kullback.runner.records import ToolCallError, ToolSig, content_hash
from kullback.runner.records import plain as _plain
from kullback.runner.state import StateView, _db_put

STATE_PARAMS = ("state", "db", "world", "env")
EXCEPTION_CLASSES = {
    TypeError: "invalid_arguments",
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
    """Route order: code, then recording, then the LLM stand-in; a bad call gets an error, not a raise."""

    def __init__(self, env_tools_module: Any = None, recordings: Any = None, starting_state: Any = None,
                 overlay: Any = None, stand_in_model: Any = None, tool_sigs: Optional[list[ToolSig]] = None,
                 overlay_rows: Optional[dict] = None, canon_rules: Any = None,
                 synthetic_rows: Optional[Iterable[str]] = None):
        self.tools = env_tools_module
        # D40: a result that names a synthetic row was answered from a row no trace showed.
        self.synthetic_rows = frozenset(synthetic_rows or ())
        self.state = starting_state if isinstance(starting_state, StateView) else StateView(starting_state)
        self.state.add(overlay, overlay_rows)  # a view handed beside an overlay keeps both (D74)
        self.stand_in = stand_in_model
        self.sigs = {sig.name: sig for sig in (tool_sigs or [])}
        self.canon_rules = canon_rules  # the customer's CanonRules, which key the recording table (D39)
        self.recordings, self.unkeyed_recordings = _index_recordings(recordings, canon_rules)
        self.marked_tools = _has_tool_markers(self.tools)
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

    def route(self, name: str, args: Optional[dict] = None) -> RouteResult:
        args = dict(args or {})
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
        if self.stand_in is not None:
            return self._stand_in(name, args)
        return self._error(name, "tool_not_found", f"no tool named {name}")

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

    def _code(self, name: str, function: Any, args: dict) -> RouteResult:
        try:
            result = _call(function, self.state, args)
            return RouteResult(result, "code", self._reads_synthetic(result, args), None, self._misses())
        except Exception as exc:  # the customer's tools answer with an error, they do not crash the Run
            return self._error(name, _class_of(exc), _message_of(exc))

    def _reads_synthetic(self, result: Any, args: dict) -> bool:
        """The call named or returned a synthetic row (D40): the Run is assisted (D49)."""
        if not self.synthetic_rows:
            return False
        text = json.dumps([result, args], default=str, ensure_ascii=False)
        return any(row_id in text for row_id in self.synthetic_rows)

    def _stand_in(self, name: str, args: dict) -> RouteResult:
        """D49: an LLM answers a tool with no code and no recording, and the Run is Assisted."""
        prompt = (
            f"Answer the tool call as the customer's system would. Tool: {name}. "
            f"Arguments: {json.dumps(canonical_args(args), sort_keys=True, default=str)}. "
            "Reply with the tool result only, as JSON when the tool returns JSON."
        )
        reply = self.stand_in.query([{"role": "user", "content": prompt}])
        return RouteResult(_parsed(reply.content), "llm", True, None, self._misses())

    def _error(self, name: str, error_class: str, message: str) -> RouteResult:
        """D45: answered in the customer's own error encoding, taken from ToolSig.error_shapes."""
        encoding = _encoding_for(self.sigs.get(name), error_class)
        payload: Any = message if encoding == "text" else {"error": message, "class": error_class}
        error = ToolCallError(class_=error_class, payload=payload, encoding=encoding, classified_by="code")
        return RouteResult(payload, "code", False, error, self._misses())


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


def _call(function: Any, state: StateView, args: dict) -> Any:
    """Generated tool bodies take the state view first when they name it; others take args only."""
    parameters = list(inspect.signature(function).parameters)
    if parameters and parameters[0] in STATE_PARAMS:
        return function(state, **args)
    return function(**args)


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
