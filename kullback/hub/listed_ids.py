"""The listed ids a shipped world drops: members of an id list that name no row of the table they key (D429).

A world is learned from recordings only, so a row can list an id no recording opened (D425 counts them); the
agent then opens an id the world itself handed it and the tool answers that it does not exist. The build world
stays as learned. The Harbor export drops those members from the world it ships, where the table a column keys
can be read off its values without doubt: every member that names a row names a row of the same one table, and
at least half the members name one. A column whose members name rows of two tables, or too few rows, is ambiguous:
nothing is dropped from it and the export says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from kullback.builder.world_tools import _listed

# The share of a column's members that must name a row before the column is read as a list of ids.
MIN_NAMED_SHARE = 0.5
# Why a Task is set aside: its Reference Run reads an id the shipped world drops.
READS_DROPPED_ID = "reads_dropped_id"


@dataclass
class Column:
    """One list column of text values: the table it keys (None where ambiguous) and the members naming no row."""

    table: str
    field: str
    keys: str | None
    listed: int
    missing: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.table}.{self.field}"


def _tables_named(member: str, keys: dict[str, set[str]]) -> set[str]:
    return {table for table, ids in keys.items() if member in ids}


def read_column(table: str, name: str, members: list[str], keys: dict[str, set[str]]) -> Column | None:
    """The column's reading, or None where no member names a row of any table (no list of ids)."""
    named = [_tables_named(member, keys) for member in members]
    hit = [tables for tables in named if tables]
    if not hit:
        return None
    targets = set().union(*hit)
    sure = len(targets) == 1 and len(hit) >= MIN_NAMED_SHARE * len(members)
    missing = sorted({member for member, tables in zip(members, named, strict=True) if not tables})
    return Column(table, name, next(iter(targets)) if sure else None, len(members), missing)


def columns(state: dict) -> list[Column]:
    """Every list column of the state that names a row of some table and lists a member that names none."""
    keys = {table: set(rows) for table, rows in (state or {}).items() if isinstance(rows, dict)}
    out = []
    for table in sorted(keys):
        for name, members in sorted(_listed(state[table]).items()):
            column = read_column(table, name, members, keys)
            if column is not None and column.missing:
                out.append(column)
    return out


def drops(found: Iterable[Column]) -> dict[str, dict[str, list[str]]]:
    """Per table and column, the members the shipped world drops: those of every column read without doubt."""
    out: dict[str, dict[str, list[str]]] = {}
    for column in found:
        if column.keys is not None:
            out.setdefault(column.table, {})[column.field] = list(column.missing)
    return out


def prune_rows(rows: Any, gone: dict[str, list[str]]) -> int:
    """Drop the named members from each row's list columns in place; answers how many members went."""
    count = 0
    for row in (rows.values() if isinstance(rows, dict) else ()):
        for name, members in gone.items():
            value = row.get(name) if isinstance(row, dict) else None
            if isinstance(value, list):
                kept = [member for member in value if member not in members]
                count += len(value) - len(kept)
                row[name] = kept
    return count


def prune_overlay(payload: dict, plan: dict[str, dict[str, list[str]]]) -> None:
    """Drop the planned members from the row values a Task's overlay pins, in place."""
    values = payload.get("values") or {}
    for row in (payload.get("overlay") or {}).get("rows") or ():
        gone = plan.get(str(row.get("table")))
        body = values.get(row.get("version_hash"))
        if gone and isinstance(body, dict):
            prune_rows({"row": body}, gone)


def reads(calls: Iterable[dict], ids: set[str]) -> bool:
    """Whether any call carries an argument equal to one of `ids`, at any depth."""

    def found(value: Any) -> bool:
        if isinstance(value, dict):
            return any(found(item) for item in value.values())
        if isinstance(value, list):
            return any(found(item) for item in value)
        return isinstance(value, str) and value in ids

    return any(found(call.get("arguments")) for call in calls)
