"""One tool call as a transaction on a toolkit's db: snapshot before the body runs, restore if it raises.

The Router takes this snapshot around every generated body (route.py `_snapshot_world`), and the
builder's sandbox child takes the same one around every prefix call it replays (sandbox.py
`_run_prefix`), so a body that raises leaves the db as it found it in both places and a gated call
never stands on a state no real Run can reach.

Self-contained on purpose: the sandbox child runs under `python -I` with the environment cleared and
cannot import kullback, so it gets this file's own bytes prepended to its runner, the way it gets
arith.py's. That is why this module imports only the standard library and pydantic, carries no
`from __future__` line (a second one further down the child's file would not compile), and keeps
its private names distinct from the runner's own.
"""

import copy
import json
from typing import Any, get_args

from pydantic import BaseModel


def freeze(value: Any) -> tuple:
    """Snapshot one store as bytes when it round trips through JSON, else as plain objects.

    The success path pays only the serialise; the bytes are parsed back solely on rollback, which
    almost never happens. A pydantic db serialises through its own Rust encoder. Where serialising
    fails (a value JSON cannot carry, like a set), the snapshot falls back to a deep copy, which
    restores by replacement with no aliases back into the live world.
    """
    try:
        if hasattr(value, "model_dump_json"):
            return ("json", value.model_dump_json().encode("utf-8"))
        return ("json", json.dumps(value).encode("utf-8"))
    except (TypeError, ValueError):
        return ("deepcopy", copy.deepcopy(value))


def thaw(snapshot: Any) -> Any:
    """Parse a frozen snapshot back to plain data. Runs only on rollback, never per call."""
    if snapshot is None:
        return None
    kind, payload = snapshot
    return json.loads(payload) if kind == "json" else payload


def snapshot_db(tools: Any) -> Any:
    """The toolkit's db before one body runs, frozen; None where the toolkit holds no db."""
    db = getattr(tools, "db", None)
    return freeze(db) if db is not None else None


def restore_db(tools: Any, snapshot: Any) -> None:
    """Put the toolkit's db back to what `snapshot_db` froze: added rows go, removed rows return."""
    _restore_db_to(getattr(tools, "db", None), thaw(snapshot))


def row_model(db: Any, table: str) -> Any:
    """The row class of one table of a pydantic db, so a row written into it stays that class."""
    field = getattr(type(db), "model_fields", {}).get(table)
    for arg in (get_args(field.annotation) if field is not None else ()):
        if isinstance(arg, type) and hasattr(arg, "model_validate"):
            return arg
    return None


def _row_plain(value: Any) -> Any:
    """A pydantic value as plain JSON data (records.plain, kept inline for the sandbox child)."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict):
        return {k: _row_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_row_plain(v) for v in value]
    return value


def _prune_added(current: dict, snapshot: dict) -> None:
    """Drop the rows a body added: everything the snapshot never held."""
    for key in [key for key in current if key not in snapshot]:
        del current[key]


def _restore_plain_db(db: dict, snapshot: dict) -> None:
    """Put a plain dict db back: added tables removed, added rows dropped, kept rows reset."""
    for table in [key for key in db if key not in snapshot]:
        del db[table]
    for table, rows in snapshot.items():
        current = db.get(table)
        if isinstance(current, dict) and isinstance(rows, dict):
            _prune_added(current, rows)
            current.update(rows)
        else:
            db[table] = rows


def _restore_model_table(db: Any, table: str, rows: Any) -> None:
    """Put one table of a pydantic db back, its row classes rebuilt as state.py's `_db_put` builds them.

    A body that replaced the table with a plain value or deleted it gets a fresh dict, so the
    corruption does not survive the rollback and later bodies read rows by attribute again.
    """
    if not isinstance(rows, dict):
        return
    current = getattr(db, table, None)
    if not isinstance(current, dict):
        current = {}
        setattr(db, table, current)
    _prune_added(current, rows)
    model = row_model(db, table)
    for row_id, row in rows.items():
        current[row_id] = model.model_validate(_row_plain(row)) if model is not None else row


def _restore_db_to(db: Any, snapshot: Any) -> None:
    """Put a toolkit db back to its snapshot: added rows go, removed rows return, changed rows reset.

    A pydantic db gets its row classes back through the same model lookup `_db_put` uses, so a
    later body still reads rows by attribute. A plain dict db already holds plain rows, so its
    snapshot values land as they are and tables the snapshot never held are removed outright.
    """
    if isinstance(db, BaseModel) and isinstance(snapshot, type(db)):
        db.__setstate__(copy.deepcopy(snapshot.__getstate__()))
        return
    if db is None or snapshot is None or not isinstance(snapshot, dict):
        return
    if isinstance(db, dict):
        _restore_plain_db(db, snapshot)
        return
    for table, rows in snapshot.items():
        _restore_model_table(db, table, rows)
    _clear_added_tables(db, snapshot)


def _clear_added_tables(db: Any, snapshot: dict) -> None:
    """Drop what a body added past the snapshot: unknown tables emptied, unknown attrs removed."""
    for table in (getattr(type(db), "model_fields", {}) or {}):
        current = getattr(db, table, None)
        if table not in snapshot and isinstance(current, dict):
            current.clear()
    _drop_unknown_attrs(db, snapshot)


def _drop_unknown_attrs(db: Any, snapshot: dict) -> None:
    """Remove the non field attributes a body set past the snapshot."""
    fields = set(getattr(type(db), "model_fields", {}) or {})
    for name in [key for key in vars(db) if key not in snapshot and key not in fields
                 and not key.startswith("_")]:
        try:
            delattr(db, name)
        except AttributeError:
            pass
