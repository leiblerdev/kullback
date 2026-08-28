"""The Task's Starting state, its overlay over the shared world, and the write path into a
generated toolkit's own db (D74, D46)."""

from __future__ import annotations

from typing import Any, Optional, get_args

from harness.shared.records import content_hash
from harness.shared.records import plain as _plain


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
                pinned = _plain(row)  # the caller's row store is not this Run's to write through
                self.overlay.setdefault(table, {})[str(row_id)] = pinned
                self.put(table, row_id, pinned)

    def row(self, table: str, row_id: Any) -> Any:
        """One row by id; the lookup lives here, never in a tool body."""
        return (self.shared.get(table) or {}).get(str(row_id))

    def put(self, table: str, row_id: Any, row: Any) -> None:
        """The write path: a tool body and a recorded write both land a row here, and the End state has it."""
        rows = self.shared.setdefault(table, {})
        current = rows.get(str(row_id))
        rows[str(row_id)] = dict(current, **row) if isinstance(current, dict) and isinstance(row, dict) else row

    def any_value(self, field: str) -> Any:
        """A flat field lookup, which is how a caller with no row of its own reads the world (D77).

        Not called `get`: a StateView is not a mapping, and a caller that reached for `state.get`
        expecting `dict.get` would get this scoped, cross-row search of the whole world instead.

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
