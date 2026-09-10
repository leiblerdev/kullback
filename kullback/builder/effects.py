"""What a write changed on rows its own arguments never named, read off the recording (D215).

A write is judged on its own result and on nothing else. The body writer is shown the call, its
arguments and the answer it came back with (`compile_env._tool_block`), and the replay scores that
answer against the recording (`runner/replay.ScoredRouter`). Neither says anything about the rows
the write moved on the way: another table's row reached through a foreign key the customer's tool
read inside itself, a history list appended on the same row, a total recomputed out of two
catalogue rows. A body that leaves those alone answers its own call perfectly and leaves the world
one column short, and the shortfall only ever surfaces later, on a read that happens to look, and
not at all when nothing looks again.

The evidence was in the recording the whole time. A Run reads a row, writes, and reads something
that carries the row again; the two sightings differ, and what sits between them is the write. That
is the same evidence `mine.observed_effects` reads, narrowed there to two identical calls of one
read tool around the write and consumed only to raise a tool's kind. Here it is read from any two
sightings of one row, whatever call carried them, and it is kept: which rows moved, which column
paths moved on them, what each held before and after, and whether one write or several sat between
the two sightings.

Nothing in this module is about a domain. A row is identified by the miner's own homing, a column
by the path it sits at, and a relation between numbers by the catalogue `lesson` already holds. The
names in what comes out are the customer's, read out of the customer's data; none of them is
written here.

Three readers. The body prompt gets the columns a call changed on rows its arguments do not name,
with the formula the recording implies where the numbers hold one (`effects_block`). The memorised
gate gets the after values, so a body that writes down a value it should have computed is refused
the way D187 rule (d) refuses one it read off a result (`effect_values`). The replay gets the rows
and columns to check once the write has replayed (`replay_evidence`), which is the only one of the
three that is allowed to see a held-out Run, because it is code and tells no one anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from kullback.builder import mine
from kullback.builder.lesson import arithmetic_relations
from kullback.runner.canon import canonicalize as canon
from kullback.runner.canon import leaves
from kullback.runner.records import EntitySchema, Trace

# How many numbers one effect formula is searched over. The search is quadratic in this and the
# same shape as `lesson.RELATION_OPERANDS`, which is why it is the same size: a recording states
# its relation in the first handful of numbers in front of it or does not hold one.
EFFECT_OPERANDS = 8
# How many changed columns one call's evidence keeps, and how many calls one tool's block shows.
# Both are caps on a prompt, not on the observation: what is left out is counted and said.
COLUMNS_PER_CALL = 12
CALLS_SHOWN = 6
# How long a value printed in the block may be. A column's before and after are what the writer
# needs; the row behind them is not.
VALUE_CHARS = 60


@dataclass
class EffectColumn:
    """One column one write moved on one row, with what it held on either side of the write."""
    table: str
    row_id: str
    path: str
    before: Any
    after: Any
    # The write's own arguments name this row, so the writer already has it in front of it and the
    # block leaves it out. The replay still checks it: a body that answers the right row and stores
    # the wrong one is exactly the failure nothing was watching for.
    named: bool = False
    # More than one write sat between the two sightings, so the column is filed under the writes
    # that owe it and none other (D234). Said in the block rather than dropped, because a write
    # that may have moved a column is still the only thing the recording offers about that column.
    ambiguous: bool = False
    # Whether the replay checks this column after this write. Only the last write that owes an
    # ambiguous column is checked, because that is the only point the recording actually constrains:
    # the world has to hold the after value once every write between the two sightings has run, and
    # asking it of an earlier one fails a write for a column a later write was going to move.
    checked: bool = True
    # No write between the two sightings owes this column: none named its row and none of their
    # tools was seen moving it alone elsewhere in the recordings this pass was given (on the
    # replay-evidence path, that Task's Runs). Filed once, under the last write, and never
    # checked against any call (D234).
    unattributed: bool = False
    # The rule the recording implies for a numeric column, as a formula over other columns; empty
    # where the numbers hold none. Never a value: the operands are named by table and column path.
    formulas: tuple[str, ...] = ()


@dataclass
class WriteEffect:
    """One recorded write call and everything it was seen to change beyond its own answer."""
    tool: str
    call_id: str
    trace_id: str
    columns: list[EffectColumn] = field(default_factory=list)

    @property
    def unnamed(self) -> list[EffectColumn]:
        return [column for column in self.columns if not column.named]


def _exempt(schema: EntitySchema, table: str, path: str) -> bool:
    """A column that moves on its own is not evidence that a write moved it.

    Two rules, and both already exist: the name rule the miner reads a timestamp and a counter by,
    and the class the schema mined for the column, since a column the corpus itself showed varying
    across successful re-runs was exempted for that reason (D73).
    """
    if mine._exempt_field(path):
        return True
    head = path.split(".")[0].split("[")[0]
    return any(column.table == table and column.name == head and column.class_ == "exempt"
               for column in schema.columns)


def _numbers_of(row: dict) -> list[tuple[str, float]]:
    """(path, number) for every numeric leaf of a row, booleans left out."""
    return [(path, float(value)) for path, value in leaves(row).items()
            if not isinstance(value, bool) and isinstance(value, (int, float))]


def _argument_numbers(args: Any, path: str = "", out: Optional[list] = None) -> list[tuple[str, float]]:
    """(the name it sat under, the number) for every number a call's arguments carry."""
    out = [] if out is None else out
    if isinstance(args, dict):
        for key in sorted(args):
            _argument_numbers(args[key], str(key), out)
    elif isinstance(args, (list, tuple)):
        out.append((f"the number of {path or 'arguments'}", float(len(args))))
        for item in args:
            _argument_numbers(item, path, out)
    elif isinstance(args, bool):
        return out
    elif isinstance(args, (int, float)):
        out.append((path or "the argument", float(args)))
    return out


class _RowView:
    """The last version of every row a Run had seen by a given point, by (table, row id).

    `order` is what the Run itself read, in the order it read it, and the world it started in is not
    in it. A Starting state is every row the customer has, and a formula searched over all of them
    would be searched over rows the call never touched; what a write's arithmetic is written over is
    what the Run had just put in front of it.
    """

    def __init__(self, world: Optional[dict] = None):
        self.rows: dict[tuple[str, str], dict] = {}
        self.order: list[tuple[str, str]] = []
        for table, rows in (world or {}).items():
            if not isinstance(rows, dict):
                continue
            for row_id, row in rows.items():
                if isinstance(row, dict):
                    self.rows[(str(table), str(row_id))] = dict(row)

    def put(self, table: str, row_id: str, row: dict) -> None:
        key = (table, row_id)
        if key not in self.order:
            self.order.append(key)
        held = self.rows.get(key)
        self.rows[key] = {**held, **row} if held else dict(row)

    def get(self, table: str, row_id: str) -> Optional[dict]:
        return self.rows.get((table, row_id))


def _ordinal(rank: int) -> str:
    return {1: "first", 2: "second", 3: "third"}.get(rank, f"{rank}th")


def _operands(view: _RowView, named: Iterable[tuple[str, str]], args: Any) -> list[tuple[str, float]]:
    """The numbers a formula may be written over: the call's arguments, then the rows it can see.

    The rows the call names come first, because a write's own row is where its arithmetic usually
    starts, and the rest follow in the order the Run read them. Every operand is named by its table
    and column path and never by its value, so the formula that comes out is a rule a body can
    follow on any world rather than a value it could memorise. Where one table supplies two rows,
    the operand names say which of them, counted in the order the Run read them, since a difference
    between two rows of one table is exactly the shape this is looking for.
    """
    wanted = {key for key in named if key in view.rows}
    keys = [key for key in view.order if key in wanted]
    keys += [key for key in view.order if key not in wanted]
    ranks: dict[str, int] = {}
    out: list[tuple[str, float]] = list(_argument_numbers(args))
    for table, row_id in keys:
        rank = ranks.get(table, 0) + 1
        ranks[table] = rank
        for path, number in _numbers_of(view.rows[(table, row_id)]):
            label = f"{table}.{path}" if rank == 1 \
                else f"{table}.{path} of the {_ordinal(rank)} row read"
            out.append((label, number))
    return out[:EFFECT_OPERANDS]


def _formulas(table: str, path: str, before: Any, after: Any, view: _RowView,
              named: Iterable[tuple[str, str]], args: Any) -> tuple[str, ...]:
    """How the after value follows from the columns the call could see, where the numbers say so.

    Two questions, and a numeric column answers one of them or neither. What the column now holds,
    as a number over the operands, which is what a recomputed total looks like. And what it moved
    by, which is what a debit looks like: a balance that falls by the difference between two
    catalogue rows holds no relation as a value at all and an exact one as a change.
    """
    if isinstance(after, bool) or not isinstance(after, (int, float)):
        return ()
    operands = _operands(view, named, args)
    if not operands:
        return ()
    where = f"{table}.{path}"
    found = [f"{where} is {text}" for text in sorted(arithmetic_relations(float(after), operands))]
    if not isinstance(before, bool) and isinstance(before, (int, float)):
        found += [f"{where} changes by {text}"
                  for text in sorted(arithmetic_relations(float(after) - float(before), operands))]
    return tuple(found)


def observe_effects(traces: Iterable[Trace], schema: EntitySchema, write_tools: Iterable[str],
                    db: Optional[dict] = None, worlds: Optional[dict] = None,
                    read_result: Optional[Callable[[str, Any], Any]] = None,
                    ) -> dict[str, list[WriteEffect]]:
    """Per write tool, what each of its recorded calls was seen to change (rule 1 of D215).

    One Run at a time. Every row the Run's calls state is homed the way the miner homes it, so two
    sightings of one row are recognised as one row however the two results were shaped, and the
    sightings of each row are walked in recorded order. Where two consecutive sightings of a row
    differ, the writes that sit between them owe the columns that moved, under D234's narrowing:
    where more than one sits between, a write owes a column only where its own arguments name the
    row the column sits on or where the same tool moved that column alone elsewhere in the
    recordings this pass was given (on the replay-evidence path, that Task's Runs), and a column
    no write in the span owes is filed once as unattributed rather than charged to the last call.
    A row whose first sighting comes after a write takes its before value from the world that Run
    started in (`worlds` for that Run, else the shared `db`), which is the Starting-state pin and
    the only evidence there is for a row nothing read first.

    Only the rows a Run's own calls returned are read as sightings, so the same R33 rule that keeps
    another requestor's results out of the world keeps them out of this. A write by any requestor is
    credited, because every requestor's call goes through the same world (D164) and a user's own
    write moves that world exactly as the assistant's does.
    """
    writes = set(write_tools)
    traces = list(traces)
    proven = _proven_moves(traces, schema, writes, db, worlds, read_result)
    out: dict[str, list[WriteEffect]] = {}
    for trace in traces:
        for effect in _effects_of_trace(trace, schema, writes, db, worlds, read_result, proven):
            if effect.columns:
                out.setdefault(effect.tool, []).append(effect)
    return out


def _write_at(calls: list, writes: set[str]) -> dict[int, Any]:
    """Index to call for every recorded write in the set, in recorded order."""
    return {index: call for index, call in enumerate(calls)
            if call.name in writes and call.error is None and call.id}


def _sightings(trace: Trace, schema: EntitySchema, writes: set[str],
               read_result: Optional[Callable[[str, Any], Any]]) -> dict[int, list]:
    """Every row sighting of one trace, by the call index that carried it."""
    from kullback.builder.compile_env import _observations

    sightings: dict[int, list] = {}
    for obs in _observations([trace], schema, writes, read_result=read_result):
        sightings.setdefault(int(obs.order[1]), []).append(obs)
    return sightings


def _trace_start(db: Optional[dict], worlds: Optional[dict], trace_id: str
                 ) -> tuple[_RowView, dict[tuple[str, str], tuple[int, dict]]]:
    """One trace's rows as last seen: the world it started in, before any of its calls."""
    start = (worlds or {}).get(trace_id) if worlds else None
    view = _RowView(start if start is not None else db)
    return view, {key: (-1, dict(row)) for key, row in view.rows.items()}


def _walk_spans(trace: Trace, schema: EntitySchema, writes: set[str], db: Optional[dict],
                 worlds: Optional[dict], read_result: Optional[Callable[[str, Any], Any]],
                 ) -> Iterable[tuple[list, Any, dict, _RowView]]:
    """Each write span of one trace: the writes between two sightings of one row.

    Yields (credited, sighting, before, view) in recorded order, where the view is the Run's rows
    as they stood before the sighting. The view keeps advancing as the walk goes, so each span
    has to be filed before the next is pulled.
    """
    write_at = _write_at(list(trace.tool_calls), writes)
    if not write_at:
        return
    view, seen = _trace_start(db, worlds, trace.trace_id)
    for index in sorted(found := _sightings(trace, schema, writes, read_result)):
        for obs in found[index]:
            key = (obs.table, obs.row_id)
            previous = seen.get(key)
            credited = [call for at, call in sorted(write_at.items())
                        if previous is not None and previous[0] < at <= index]
            if credited and previous is not None:
                yield (credited, obs, previous[1], view)
            merged = {**previous[1], **obs.row} if previous is not None and obs.partial else dict(obs.row)
            seen[key] = (index, merged)
            view.put(obs.table, obs.row_id, merged)


def _effects_of_trace(trace: Trace, schema: EntitySchema, writes: set[str], db: Optional[dict],
                      worlds: Optional[dict], read_result: Optional[Callable[[str, Any], Any]],
                      proven: set[tuple[str, str, str]]) -> list[WriteEffect]:
    from kullback.builder.compile_env import named_rows

    effects: dict[str, WriteEffect] = {}
    for credited, obs, before, view in _walk_spans(trace, schema, writes, db, worlds, read_result):
        _credit(effects, credited, obs, before, schema, view, trace, named_rows, proven)
    return list(effects.values())


def _changed(obs: Any, before: dict, schema: EntitySchema) -> tuple[dict, dict, list[str]]:
    """The two sightings as leaf maps and the non-exempt paths they part on."""
    old, new = leaves(before), leaves(obs.row)
    return old, new, [path for path in sorted(set(old) & set(new))
                      if canon(old[path]) != canon(new[path])
                      and not _exempt(schema, obs.table, path)]


def _proven_moves(traces: Iterable[Trace], schema: EntitySchema, writes: set[str],
                  db: Optional[dict], worlds: Optional[dict],
                  read_result: Optional[Callable[[str, Any], Any]]) -> set[tuple[str, str, str]]:
    """(tool, table, path) triples one write was seen to move with no other write between.

    Only a span with a single write in it proves anything: where several sat between, the
    recording does not say which of them moved the column, so such a span proves nothing about
    any of them. Read over the recordings this pass was given alone, before any credit is
    narrowed by it, which on the replay-evidence path is that Task's Runs.
    """
    proven: set[tuple[str, str, str]] = set()
    for trace in traces:
        for credited, obs, before, _view in _walk_spans(trace, schema, writes, db, worlds,
                                                       read_result):
            if len(credited) != 1:
                continue
            for path in _changed(obs, before, schema)[2]:
                proven.add((credited[0].name, obs.table, path))
    return proven


def _owners(credited: list, obs: Any, path: str, schema: EntitySchema, named_rows: Callable,
            proven: set[tuple[str, str, str]]) -> list:
    """The writes of one span that owe one column: the span itself where it holds one write,
    else the writes naming the row and the tools proven on that column elsewhere in the
    recordings this pass was given (on the replay-evidence path, that Task's Runs)."""
    if len(credited) == 1:
        return list(credited)
    return [call for call in credited
            if (obs.table, obs.row_id) in named_rows(schema, call.args or {})
            or (call.name, obs.table, path) in proven]


def _file(effects: dict[str, WriteEffect], call: Any, obs: Any, path: str, old: dict, new: dict,
          schema: EntitySchema, view: _RowView, trace: Trace, named_rows: Callable,
          checked: bool, unattributed: bool, ambiguous: bool) -> None:
    """One column on one write's evidence, left alone where an earlier span already filed it."""
    call_id = str(call.id or "")
    names = named_rows(schema, call.args or {})
    effect = effects.setdefault(call_id, WriteEffect(tool=call.name, call_id=call_id,
                                                     trace_id=trace.trace_id))
    held = {(column.table, column.row_id, column.path) for column in effect.columns}
    if (obs.table, obs.row_id, path) in held:
        return
    effect.columns.append(EffectColumn(
        table=obs.table, row_id=obs.row_id, path=path, before=old[path], after=new[path],
        named=(obs.table, obs.row_id) in names, ambiguous=ambiguous,
        checked=checked, unattributed=unattributed,
        formulas=_formulas(obs.table, path, old[path], new[path], view, names, call.args or {})))


def _credit(effects: dict[str, WriteEffect], credited: list, obs: Any, before: dict,
            schema: EntitySchema, view: _RowView, trace: Trace, named_rows: Callable,
            proven: set[tuple[str, str, str]]) -> None:
    """The columns two sightings of one row part on, filed under the writes that owe them.

    Where one write sat between the sightings it is filed under it, as before. Where several sat
    between, a write owes a column only where its own arguments name the row the column sits on
    or where the same tool moved that column alone elsewhere in the recordings this pass was
    given (on the replay-evidence path, that Task's Runs); both tests are over the recordings
    alone. A column no write in the span owes is filed
    once, under the last write, as unattributed and unchecked: the recording shows it moving and
    says nothing about which call moved it, so checking it against any of them fails whichever
    call happened to come last. The view is the Run's rows as they stood before this sighting,
    which is where the formula search looks.
    """
    old, new, changed = _changed(obs, before, schema)
    if not changed:
        return
    for path in changed:
        owners = _owners(credited, obs, path, schema, named_rows, proven)
        for call in owners or [credited[-1]]:
            owned = call in owners
            _file(effects, call, obs, path, old, new, schema, view, trace, named_rows,
                  checked=(call is owners[-1]) if owned else False, unattributed=not owned,
                  ambiguous=len(credited) > 1)


# --- what the three readers are handed ---------------------------------------------

def counts(effects: dict[str, list[WriteEffect]]) -> dict[str, int]:
    """The round's numbers over what was observed, per D215's counter list."""
    calls = [effect for rows in effects.values() for effect in rows]
    columns = [column for effect in calls for column in effect.columns]
    return {"effects_observed": len(calls), "effect_columns": len(columns),
            "effect_columns_checked": sum(1 for column in columns if column.checked),
            "effect_formulas": sum(len(column.formulas) for column in columns),
            "effects_ambiguous": sum(1 for column in columns if column.ambiguous),
            "effects_unattributed": sum(1 for column in columns if column.unattributed),
            "effect_tools": len(effects)}


def per_tool_counts(effects: dict[str, list[WriteEffect]]) -> dict[str, dict[str, int]]:
    """The same numbers per write tool, which is the grain a round reads a corpus at."""
    return {tool: counts({tool: rows}) for tool, rows in sorted(effects.items())}


def effect_values(effects: Iterable[WriteEffect]) -> dict[str, str]:
    """Every after value one tool's effects carry, as text, with where it was seen (rule 5).

    Handed to the memorised-values gate beside the values the recorded results carried: a body that
    writes down the number a column ended up holding, rather than working it out, has memorised the
    recording exactly as one that writes down a value a result answered.
    """
    out: dict[str, str] = {}
    for effect in effects:
        for column in effect.columns:
            value = column.after
            if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
                continue
            out.setdefault(str(value), f"{column.table}.{column.path}")
    return out


def _shown(effect: WriteEffect) -> list[EffectColumn]:
    """The columns the block may show: unnamed, and owned by this write.

    A column no write in its span owed belongs to no tool's prompt: telling the last call's
    author to move it is the mis-credit D234 removes, so the block leaves it out.
    """
    return [column for column in effect.unnamed if not column.unattributed]


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= VALUE_CHARS else text[:VALUE_CHARS] + "..."


def effects_block(tool: str, effects: Iterable[WriteEffect]) -> str:
    """The effects section of a write tool's block, or empty where nothing was observed (rule 2).

    Per recorded call, the columns it changed on rows its arguments do not name, with what each
    held on either side. Then the formulas, gathered over the calls rather than left on each: a
    formula every call that offered one agreed on is the rule, and a column whose calls disagree
    says so and shows both, because a writer told one rule follows it and a writer told two knows
    the recording is not settled.
    """
    rows = [effect for effect in effects if _shown(effect)]
    if not rows:
        return ""
    lines = [f"Rows {tool} changed that its arguments never named. The recording shows these "
             f"columns moving across the call, on rows the call did not address; the body has to "
             f"move them too, and has to work each value out rather than write it down."]
    for effect in rows[:CALLS_SHOWN]:
        columns = _shown(effect)[:COLUMNS_PER_CALL]
        lines.append(f"  call {effect.call_id}:")
        for column in columns:
            note = (", and another write of this call's own Run may be the one that moved it"
                    if column.ambiguous else "")
            lines.append(f"    {column.table}.{column.path}: {_short(column.before)} became "
                         f"{_short(column.after)}{note}")
        left = len(_shown(effect)) - len(columns)
        if left:
            lines.append(f"    {left} more column(s) moved on this call")
    if len(rows) > CALLS_SHOWN:
        lines.append(f"  {len(rows) - CALLS_SHOWN} more call(s) moved rows they did not name")
    formulas = _agreed_formulas(rows)
    if formulas:
        lines.append("  What the numbers say:")
        lines.extend(f"    {line}" for line in formulas)
    return "\n".join(lines)


def _agreed_formulas(effects: Iterable[WriteEffect]) -> list[str]:
    """Per column, the formula every call offering one agreed on, or both sides of a disagreement."""
    per_column: dict[str, list[set[str]]] = {}
    for effect in effects:
        for column in effect.unnamed:
            if column.formulas and not column.unattributed:
                per_column.setdefault(f"{column.table}.{column.path}", []).append(set(column.formulas))
    out: list[str] = []
    for where in sorted(per_column):
        offered = per_column[where]
        agreed = sorted(set.intersection(*offered))
        if agreed:
            out.append(agreed[0] + (", on every call of this tool that shows the column moving"
                                    if len(offered) > 1 else ""))
            continue
        both = sorted({line for one in offered for line in one})[:2]
        out.append(f"the calls do not agree on {where}: one shows {both[0]}"
                   + (f" and another shows {both[1]}" if len(both) > 1 else ""))
    return out


#: Key of `replay_evidence` carrying the columns no write in their span owed (D234). It is not
#: a call id: no recorded call is checked against it, so a runner that does not know it ignores
#: it, and no per-call evidence is emitted for such a column. The leading tag follows D39's rule
#: for the recorded-text key, and the writer bumps it past any recorded id that claims it, so the
#: strays land under a key no call looks up. The re-freeze teaches the check to count what sits
#: under it instead of failing the last call.
UNATTRIBUTED = "\x00unattributed"


def replay_evidence(effects: dict[str, list[WriteEffect]]) -> dict[str, list[dict]]:
    """Per recorded call id, the rows and columns the replay checks once that call has replayed.

    Plain dicts, because this crosses into the Runner, which reads records and not builder objects.
    The rows the arguments named are here too: a body that answers the right row and stores the
    wrong one is a write failure the result comparison cannot see. A column only the writes that
    owe it carry is here only under the last of those, because that is the only write after which
    the recording says the world holds the after value; asking it of an earlier one fails that
    write for a column the write after it was always going to move. A column no write owes rides
    under the `unattributed` key instead of under any call, with the trace that showed it moving,
    so the round can count what the recording left unowned rather than fail the call that came last.
    """
    out: dict[str, list[dict]] = {}
    for rows in effects.values():
        for effect in rows:
            out.setdefault(effect.call_id, []).extend(
                {"tool": effect.tool, "table": column.table, "row": column.row_id,
                 "path": column.path, "before": column.before, "after": column.after,
                 "named": column.named, "ambiguous": column.ambiguous,
                 "unattributed": column.unattributed,
                 "formulas": list(column.formulas)}
                for column in effect.columns if column.checked and not column.unattributed)
    stray = [{"tool": effect.tool, "trace_id": effect.trace_id, "table": column.table,
              "row": column.row_id, "path": column.path, "before": column.before,
              "after": column.after, "named": column.named, "ambiguous": column.ambiguous,
              "unattributed": True, "formulas": list(column.formulas)}
             for rows in effects.values() for effect in rows for column in effect.columns
             if column.unattributed]
    out = {call_id: rows for call_id, rows in out.items() if rows}
    if stray:
        # The tag is a convention, not an invariant, since a recorded id is an unconstrained
        # string: bump the key past any id the recordings claim, so the strays land where no
        # call looks them up. The frozen check finds flagged rows by scanning, never by key.
        claimed = {effect.call_id for rows in effects.values() for effect in rows}
        key = UNATTRIBUTED
        while key in claimed:
            key += "\x00"
        out.setdefault(key, []).extend(stray)
    return out


__all__ = ["EffectColumn", "UNATTRIBUTED", "WriteEffect", "counts", "effect_values",
           "effects_block", "observe_effects", "per_tool_counts", "replay_evidence"]
