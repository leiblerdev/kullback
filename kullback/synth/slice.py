"""The slice of the world one scenario is written onto, picked by code before any session (D420).

A slice is one owner row, every row of any table holding a scalar cell equal to the owner's key, and the rows
those rows refer to by key one level down (their line items, their legs). The owner table is found from the
schema and the recordings, generally, as tasks.py `_owner` finds the owner among bound rows: the table whose
keys the scalar cells (top level or nested objects, never lists) of other tables' rows name most often; where
no table is named so, the table whose rows hold the most columns the recorded users give (recorded_fields).

The scenario's binding and the values of its end are checked against the slice by code at write time: a row is
bound where the statement, ask or end names its key or a value only it holds in the slice (the owner is always
bound), and every number, id shaped word and quoted value of the end must be in the slice, in a policy line
the scenario quotes, or be a sum or difference of two slice amounts.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Optional

from kullback.synth.rule import World, id_word

# How many characters the rendered slice holds at most; a row past it is named with its size only.
LIMIT = 6000
# A string cell names its row only at this many characters or more; a number only at this size or more.
NAME_CHARS = 4
NAME_NUMBER = 10
# Two amounts are equal within this much.
CENT = 0.005
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”")


@dataclass
class Slice:
    """One owner row and the rows around it: table to key to row, the owner first, in the order picked."""

    owner_table: str
    owner_key: str
    rows: dict[str, dict[str, dict]] = field(default_factory=dict)

    def items(self) -> list[tuple[str, str, dict]]:
        return [(table, key, row) for table, keyed in self.rows.items() for key, row in keyed.items()]

    def text(self, limit: int = LIMIT) -> str:
        """The slice rendered compactly, one row per line as table, key and cells; rows past `limit` named only."""
        lines, used, cut = [], 0, []
        for table, key, row in self.items():
            line = f"{table} {key}: {json.dumps(row, sort_keys=True, separators=(',', ':'))}"
            if used + len(line) > limit and lines:
                cut.append(f"{table} {key} ({len(line)} chars)")
                continue
            lines.append(line)
            used += len(line) + 1
        if cut:
            lines.append("not shown, past the size limit: " + ", ".join(cut))
        return "\n".join(lines)


def _scalars(node: Any, lists: bool) -> Iterable[Any]:
    """Every scalar under a row at any depth; inside lists only where `lists` is true."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _scalars(value, lists)
    elif isinstance(node, list):
        if lists:
            for value in node:
                yield from _scalars(value, lists)
    elif node is not None and not isinstance(node, bool):
        yield node


def _tables(world: World) -> dict[str, dict]:
    return {str(table): rows for table, rows in world.state.items() if isinstance(rows, dict) and rows}


def _columns(row: Any) -> set[str]:
    """The column names of a row and of the objects nested in it, never inside a list."""
    out: set[str] = set()
    if isinstance(row, dict):
        for name, value in row.items():
            out.add(str(name))
            out |= _columns(value) if isinstance(value, dict) else set()
    return out


def _referrals(tables: dict, key_tables: dict[str, list[str]]) -> tuple[Counter, dict]:
    """How often each table's keys are named in other tables' scalar cells, and by which rows, per key."""
    named: Counter = Counter()
    referring: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for table, rows in tables.items():
        for key, row in rows.items():
            for value in {str(v) for v in _scalars(row, lists=False)}:
                for other in (t for t in key_tables.get(value, ()) if t != table):
                    named[other] += 1
                    referring.setdefault(other, {}).setdefault(value, []).append((table, str(key)))
    return named, referring


def _by_fields(tables: dict, fields: Iterable[str]) -> str:
    """The table holding the most columns recorded users gave, or nothing."""
    wanted = set(fields)
    score = {t: len(wanted & set().union(*(_columns(r) for r in rows.values()))) for t, rows in tables.items()}
    return min((t for t in score if score[t]), key=lambda t: (-score[t], t), default="")


@dataclass
class Slicer:
    """The indexes one world's slices are cut from: the owner table, and per owner key the rows referring to it."""

    world: World
    owner_table: str = ""
    referring: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    key_tables: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def of(cls, world: World, fields: Iterable[str] = ()) -> "Slicer":
        tables = _tables(world)
        key_tables: dict[str, list[str]] = {}
        for table, rows in tables.items():
            for key in rows:
                key_tables.setdefault(str(key), []).append(table)
        named, referring = _referrals(tables, key_tables)
        owner = min(named, key=lambda t: (-named[t], t)) if named else _by_fields(tables, fields)
        return cls(world=world, owner_table=owner, referring=referring.get(owner, {}), key_tables=key_tables)

    def slice_of(self, key: str) -> Optional[Slice]:
        """The slice around one owner key, or None where the owner table does not hold it."""
        state = self.world.state
        owner_rows = state.get(self.owner_table)
        if not isinstance(owner_rows, dict) or key not in owner_rows:
            return None
        cut = Slice(owner_table=self.owner_table, owner_key=key)

        def add(table: str, row_key: str) -> bool:
            row = state.get(table, {}).get(row_key)
            if not isinstance(row, dict) or row_key in cut.rows.get(table, {}):
                return False
            cut.rows.setdefault(table, {})[row_key] = row
            return True

        add(self.owner_table, key)
        near = [(table, row_key) for table, row_key in self.referring.get(key, ()) if add(table, row_key)]
        for table, row_key in [(self.owner_table, key), *near]:
            for value in {str(v) for v in _scalars(state[table][row_key], lists=True)}:
                for other in self.key_tables.get(value, ()):
                    if other != self.owner_table:
                        add(other, value)
        return cut

    def owners(self) -> list[str]:
        rows = self.world.state.get(self.owner_table)
        return sorted(str(key) for key in rows) if isinstance(rows, dict) else []


_SLICERS: dict[int, Slicer] = {}


def slicer(world: World, fields: Iterable[str] = ()) -> Slicer:
    """The world's slicer, built once per world object."""
    held = _SLICERS.get(id(world))
    if held is None or held.world is not world:
        held = _SLICERS[id(world)] = Slicer.of(world, fields)
    return held


def pick_slice(world: World, seed: Any, used: Iterable[str] = (), table: str = "",
               fields: Iterable[str] = ()) -> Optional[Slice]:
    """A slice around one owner, deterministic by seed: an owner not in `used` where one is left, one with rows
    referring to it before one without, one whose slice holds a row of `table` where one is asked for and any
    does. None where the world has no owner table."""
    cutter = slicer(world, fields)
    keys = cutter.owners()
    if not keys:
        return None
    spent = {str(key) for key in used}
    pool = [key for key in keys if key not in spent] or keys
    pool = [key for key in pool if cutter.referring.get(key)] or pool
    if table:
        holding = [key for key in pool if table in (cutter.slice_of(key) or Slice("", "")).rows]
        pool = holding or pool
    return cutter.slice_of(random.Random(str(seed)).choice(pool))


def slice_for(world: World, binding: dict, fields: Iterable[str] = ()) -> Optional[Slice]:
    """The slice a record's binding was cut from: the one around its owner key, or None."""
    cutter = slicer(world, fields)
    keys = (binding or {}).get(cutter.owner_table) if cutter.owner_table else None
    keys = [keys] if isinstance(keys, str) else list(keys or ())
    return cutter.slice_of(str(keys[0])) if keys else None


# --- the binding and the values, by code ---------------------------------------------------------


def _named(lowered: str, needle: str) -> bool:
    return bool(needle) and re.search(r"(?<!\w)" + re.escape(needle.lower()) + r"(?!\w)", lowered) is not None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _names_row(lowered: str, key: str, row: dict, owner_values: set[str]) -> bool:
    if _named(lowered, key):
        return True
    for value in _scalars(row, lists=True):
        if str(value).lower() in owner_values:
            continue
        number = _number(value) if not isinstance(value, str) else None
        if number is not None and abs(number) >= NAME_NUMBER and _named(lowered, _plain(number)):
            return True
        if isinstance(value, str) and len(value) >= NAME_CHARS and _named(lowered, value):
            return True
    return False


def _plain(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


def binding_of(cut: Slice, *texts: str) -> dict[str, list[str]]:
    """Every slice row the texts name by key or by a value the owner row does not hold, the owner always first."""
    lowered = " ".join(str(text or "") for text in texts).lower()
    owner_row = cut.rows[cut.owner_table][cut.owner_key]
    owner_values = {str(value).lower() for value in _scalars(owner_row, lists=True)}
    out: dict[str, list[str]] = {cut.owner_table: [cut.owner_key]}
    for table, key, row in cut.items():
        if (table, key) != (cut.owner_table, cut.owner_key) and _names_row(lowered, key, row, owner_values):
            out.setdefault(table, []).append(key)
    return out


def _amounts(cut: Slice) -> list[float]:
    found = []
    for _, key, row in cut.items():
        for value in [key, *_scalars(row, lists=True)]:
            if isinstance(value, str):
                found += [n for n in (_number(token) for token in _NUMBER.findall(value)) if n is not None]
            elif _number(value) is not None:
                found.append(float(value))
    return found


def _known(number: float, amounts: list[float], pairs: set[float]) -> bool:
    return any(abs(number - amount) < CENT for amount in amounts) or any(abs(number - p) < CENT for p in pairs)


def _quoted_refusal(text: str, quoted_lines: str, lowered: list[str]) -> str:
    for match in _QUOTED.finditer(text):
        said = (match.group(1) or match.group(2) or "").strip()
        held = said.lower() in quoted_lines.lower() or any(said.lower() in value for value in lowered)
        if said and not held:
            return f"\"{said}\""
    return ""


def _number_refusal(text: str, cut: Slice, quoted_lines: str) -> str:
    amounts = sorted(set(_amounts(cut)))
    policy = [n for n in (_number(token) for token in _NUMBER.findall(quoted_lines)) if n is not None]
    pairs = {a + b for a, b in combinations(amounts, 2)} | {abs(a - b) for a, b in combinations(amounts, 2)}
    for token in _NUMBER.findall(text):
        number = _number(token)
        if number is not None and not _known(number, [*amounts, *policy], pairs):
            return token
    return ""


def value_refusal(cut: Slice, end: str, lines: Iterable[str] = ()) -> str:
    """The first number, id shaped word or quoted value of the end that the slice does not hold, the policy lines
    do not hold and no sum or difference of two slice amounts gives, or nothing."""
    text = str(end or "")
    quoted_lines = " ".join(str(line) for line in lines)
    strings = [str(value) for _, key, row in cut.items() for value in [key, *_scalars(row, lists=True)]]
    return (_quoted_refusal(text, quoted_lines, [value.lower() for value in strings])
            or id_word(text, spare=strings) or _number_refusal(text, cut, quoted_lines))


__all__ = ["LIMIT", "Slice", "Slicer", "binding_of", "pick_slice", "slice_for", "slicer", "value_refusal"]
