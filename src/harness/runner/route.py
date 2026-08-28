"""Answers one tool call: code first, then the recording table, then an LLM stand-in (D45, D49, D74)."""

from __future__ import annotations

import inspect
import json
from typing import Any, NamedTuple, Optional, get_args

from harness.shared.canon import canonical_args
from harness.shared.records import ToolCallError, ToolSig, content_hash

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


def recording_key(tool: str, args: dict, state_hash: Optional[str]) -> str:
    """Section 8: a recorded result is keyed by tool, canonical args and pre-state hash."""
    return content_hash({"tool": tool, "args": canonical_args(args or {}), "state": state_hash})


def recording(tool: str, args: dict, state_hash: str, result: Any, error: Optional[dict] = None,
              writes: Optional[dict] = None) -> dict:
    """One row of the recording table, in the shape Router indexes.

    `writes` is the row-level effect the recorded call had, as {table: {id: row}}. A write answered
    from the recording lands it in the world, so a later read of the same entity is not stale and an
    End-state atom on that write sees the change.
    """
    return {"tool": tool, "args": args, "state_hash": state_hash, "result": result, "error": error,
            "writes": writes or {}}


class StateView:
    """The Task's Starting state: the shared world with the Task's own overlay rows laid over it (D74)."""

    def __init__(self, shared: Optional[dict] = None, overlay: Any = None, overlay_rows: Optional[dict] = None):
        self.shared: dict = _plain(shared) if shared is not None else {}
        self.overlay: dict = {}  # which rows this Task pins, for the hash and the report
        self.overlay_misses: list[dict] = []
        self.add(overlay, overlay_rows)

    def add(self, overlay: Any, overlay_rows: Optional[dict] = None) -> None:
        """Lay a Task's overlay over the world, so the pinned rows are the world one layer deep.

        One layer, not two: a lookup that reads the overlay first and the world second would also
        shadow every later write to a pinned row, and the Run would end where it started (D46).
        """
        tables, misses = _overlay_tables(overlay, overlay_rows or {})
        self.overlay_misses.extend(misses)
        for table, rows in tables.items():
            for row_id, row in (rows or {}).items():
                self.overlay.setdefault(table, {})[str(row_id)] = row
                self.put(table, row_id, row)

    def row(self, table: str, row_id: Any) -> Any:
        """One row by id; the lookup lives here, never in a tool body."""
        return (self.shared.get(table) or {}).get(str(row_id))

    def put(self, table: str, row_id: Any, row: Any) -> None:
        """The write path: a tool body and a recorded write both land a row here, and the End state has it."""
        rows = self.shared.setdefault(table, {})
        current = rows.get(str(row_id))
        rows[str(row_id)] = dict(current, **row) if isinstance(current, dict) and isinstance(row, dict) else row

    def get(self, field: str) -> Any:
        """A flat field lookup, which is how a caller with no row of its own reads the world (D77).

        Scoped to this Task's rows: where the overlay pins rows in a table only those are read, and
        where it pins none the table answers only when its rows agree. The first row that happens to
        carry the field is another customer's fact, which is an invented fact for this Task (D41).
        """
        if field in self.shared and not isinstance(self.shared[field], dict):
            return self.shared[field]
        for table in sorted(self.shared, key=lambda name: name not in self.overlay):
            rows = self.shared.get(table)
            if not isinstance(rows, dict):
                continue
            scope = list(self.overlay.get(table) or {}) or list(rows)
            values = [found for found in (_nested(rows.get(row_id), field) for row_id in scope)
                      if found is not None]
            if values:
                return values[0] if all(value == values[0] for value in values) else None
        return None

    def hash(self) -> str:
        return content_hash({"shared": self.shared, "overlay": self.overlay})


def _overlay_tables(overlay: Any, overlay_rows: dict) -> tuple[dict, list]:
    """A TaskOverlay plus its row store, or an already-resolved {table: {id: row}} dict, and the misses."""
    if overlay is None:
        return {}, []
    if isinstance(overlay, dict):
        return overlay, []
    tables: dict = {}
    misses: list[dict] = []
    for row in getattr(overlay, "rows", []):
        pinned = overlay_rows.get(row.version_hash)
        if pinned is None:  # D74 overlay miss: the Task's own row is not the one this Run will read
            misses.append({"table": row.table, "id": str(row.id), "version_hash": row.version_hash})
            continue
        tables.setdefault(row.table, {})[str(row.id)] = pinned
    return tables, misses


def _nested(row: Any, field: str) -> Any:
    """One field of a row, however deep the customer nests it (tau2's address.zip, payment_methods)."""
    if not isinstance(row, dict):
        return None
    if field in row:
        return row[field]
    for value in row.values():
        found = _nested(value, field)
        if found is not None:
            return found
    return None


class Router:
    """Route order: code, then recording, then the LLM stand-in; a bad call gets an error, not a raise."""

    def __init__(self, env_tools_module: Any = None, recordings: Any = None, starting_state: Any = None,
                 overlay: Any = None, stand_in_model: Any = None, tool_sigs: Optional[list[ToolSig]] = None,
                 overlay_rows: Optional[dict] = None):
        self.tools = env_tools_module
        self.state = starting_state if isinstance(starting_state, StateView) else StateView(starting_state)
        self.state.add(overlay, overlay_rows)  # a view handed beside an overlay keeps both (D74)
        self.stand_in = stand_in_model
        self.sigs = {sig.name: sig for sig in (tool_sigs or [])}
        self.recordings, self.unkeyed_recordings = _index_recordings(recordings)
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
        entry = self.recordings.get(recording_key(name, args, self.state_hash()))
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
            return RouteResult(_call(function, self.state, args), "code", False, None, self._misses())
        except Exception as exc:  # the customer's tools answer with an error, they do not crash the Run
            return self._error(name, _class_of(exc), _message_of(exc))

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


def _plain(value: Any) -> Any:
    """A pydantic world as plain JSON data, so a state hash and an End state are comparable."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _db_put(db: Any, table: str, row_id: str, row: Any) -> None:
    """Write one row into a generated toolkit's own db, which is the world its bodies read (D74, D46)."""
    if db is None or not isinstance(row, dict):
        return
    rows = db.get(table) if isinstance(db, dict) else getattr(db, table, None)
    if not isinstance(rows, dict):
        return
    current = rows.get(row_id)
    merged = _plain(current) if isinstance(_plain(current), dict) else {}
    merged.update(row)
    model = type(current) if hasattr(current, "model_validate") else _row_model(db, table)
    rows[row_id] = model.model_validate(merged) if model is not None else merged


def _row_model(db: Any, table: str) -> Any:
    """The row class of one table of a pydantic db, so a row written into it stays that class."""
    field = getattr(type(db), "model_fields", {}).get(table)
    for arg in (get_args(field.annotation) if field is not None else ()):
        if isinstance(arg, type) and hasattr(arg, "model_validate"):
            return arg
    return None


def _has_tool_markers(tools: Any) -> bool:
    """A tau2-shaped toolkit marks its tools; where it does, an unmarked method is not a tool (D45)."""
    if tools is None:
        return False
    return any(getattr(getattr(tools, name, None), "__tool_type__", None) is not None
               for name in dir(tools) if not name.startswith("_"))


def _index_recordings(recordings: Any) -> tuple[dict, int]:
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
        table[recording_key(tool, entry.get("args") or {}, state_hash)] = entry
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
