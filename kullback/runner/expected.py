"""A Task's expected end states, their sources and its forbidden list, derived by code from a
Reference Run, and the check of one Run's end state against them (D315, D316). No model."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Iterable, Optional

from kullback.runner.atom_context import AFFIRM_OPEN, NEGATIVE_OPEN, AtomContext
from kullback.runner.canon import DIFFERENT, EQUAL, UNRESOLVED
from kullback.runner.records import (
    EndState,
    Event,
    ExpectedCell,
    Forbidden,
    Run,
    ValueSource,
    load_run_jsonl,
)

NEGATION = re.compile(r"\b(?:do not|don'?t|never)\b([^.!?\n]*)", re.I)
# "a or b", "either a or b", "a, b or c": the bare words around the joiner, nothing longer.
ALTERNATIVES = re.compile(r"(?:either\s+)?((?:[\w#@.$-]+\s*,\s*)*[\w#@.$-]+)\s+or\s+([\w#@.$-]+)", re.I)
MAX_END_STATES = 8


def _as_run(source: Any) -> Run:
    if isinstance(source, Run):
        return source
    if isinstance(source, dict):
        return Run.model_validate(source)
    if isinstance(source, list):
        return Run(run_id="expected", events=[e if isinstance(e, Event) else Event.model_validate(e)
                                              for e in source])
    return load_run_jsonl(source)


def _turns(user_turns: Iterable[Any]) -> list[str]:
    return [text if isinstance(text, str) else str(text[1]) for text in user_turns or ()]


def _leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _clauses(policy_text: Optional[str]) -> list[str]:
    return [line.strip() for line in re.split(r"\n+|(?<=[.;])\s+", policy_text or "") if line.strip()]


def _source(value: Any, context: AtomContext, turns: list[str], clauses: list[str],
            recording: Optional[str]) -> Optional[ValueSource]:
    """A user turn, then a policy clause, then the first tool result showing the value before any
    call passed it; else None.

    An empty value has no text to be found by, so it is unsupported rather than found everywhere.
    """
    needle = context.t(value)
    if needle in ("", "null", "[]", "{}"):
        return None
    for i, text in enumerate(turns):
        if needle in context.t(text):
            return ValueSource(kind="user_turn", ptr={"recording": recording, "turn": i})
    raw = value if isinstance(value, str) else json.dumps(value)
    for clause in clauses:
        if raw in clause:
            return ValueSource(kind="policy", ptr={"clause": clause})
    # A write's own result echoes what it wrote, so only a result shown before any call passed the
    # value counts as where the value came from.
    passed = [call["idx"] for call in context.calls
              if any(context.c(leaf) == context.c(value) for leaf in _leaves(call["args"]))]
    for idx, result in context.results:
        if passed and idx > min(passed):
            break
        if any(context.c(leaf) == context.c(value) for leaf in _leaves(result)):
            before = [call["name"] for call in context.calls if call["idx"] < idx]
            return ValueSource(kind="tool_result", ptr={"event_idx": idx, "tool": before[-1] if before else None})
    return None


def _passed(value: Any, context: AtomContext) -> bool:
    return any(context.c(leaf) == context.c(value) for call in context.calls for leaf in _leaves(call["args"]))


def _affirmed_source(value: Any, context: AtomContext, recording: Optional[str]) -> Optional[ValueSource]:
    """A value a call passed, an assistant turn stated, and the next user turn opened on a yes to (no no).

    The confirmation is D306's cue (AFFIRM_OPEN, NEGATIVE_OPEN). It does not read what was affirmed:
    a yes to a turn holding several values affirms each of them.
    """
    needle = context.t(value)
    if needle in ("", "null", "[]", "{}") or not _passed(value, context):
        return None
    for said_at, text in context.assistant:
        if needle not in context.t(text):
            continue
        reply = next(((i, turn) for i, (idx, turn) in enumerate(context.user) if idx > said_at), None)
        if reply is None:
            return None
        sentences = [part.strip() for part in re.split(r"[.!?\n]+", reply[1]) if part.strip()]
        if any(AFFIRM_OPEN.match(s) for s in sentences) and not any(NEGATIVE_OPEN.match(s) for s in sentences):
            return ValueSource(kind="user_turn", ptr={"recording": recording, "turn": reply[0],
                                                      "affirmed": said_at})
    return None


def _touching_write(row_id: str, made: Any, context: AtomContext, separator: str,
                    turns: list[str], clauses: list[str], recording: Optional[str]) -> Optional[dict]:
    """The last successful sourced call that names the row (every part of its key among its arguments),
    shows the row's id in its result, or, for a row the Run made (`made`, its end values), passed one of
    its values; sourced means at least one of its arguments has a source."""
    parts = {context.c(part) for part in str(row_id).split(separator)}
    values = {context.c(leaf) for leaf in _leaves(made or {}) if context.t(leaf) not in ("", "null", "none")}
    for call in reversed(context.calls):
        if call["error"]:
            continue
        args = {context.c(leaf) for leaf in _leaves(call["args"])}
        result = next((result for idx, result in context.results if idx > call["idx"]), None)
        shown = any(context.c(leaf) == context.c(row_id) for leaf in _leaves(result))
        if not (parts <= args or shown or values & args):
            continue
        if any(_source(leaf, context, turns, clauses, recording) for leaf in _leaves(call["args"])):
            return call
    return None


def _world_source(value: Any, write: Optional[dict], context: AtomContext) -> Optional[ValueSource]:
    """A value nobody said and nobody passed, on a row a sourced write made or touched, is the world's
    answer to that write (D323)."""
    if write is None or _passed(value, context):
        return None
    return ValueSource(kind="tool_result", ptr={"event_idx": write["idx"], "tool": write["name"], "world": True})


def _sourced(value: Any, context: AtomContext, turns: list[str], clauses: list[str], recording: Optional[str],
             write: Optional[dict]) -> Optional[ValueSource]:
    """`_source`, then a user affirmation of what the agent stated, then the world's answer to a write."""
    return (_source(value, context, turns, clauses, recording) or _affirmed_source(value, context, recording)
            or _world_source(value, write, context))


def _paths(value: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a nested value by its path ("items.0.price"); an empty container is a leaf."""
    if isinstance(value, dict) and value:
        items = value.items()
    elif isinstance(value, (list, tuple)) and value:
        items = enumerate(value)
    else:
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, item in items:
        out.update(_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _nested_source(before: Any, after: Any, context: AtomContext, turns: list[str], clauses: list[str],
                   recording: Optional[str], write: Optional[dict] = None) -> tuple[Optional[ValueSource], dict]:
    """A nested value sourced leaf by leaf: only a leaf new or changed at its path needs a source."""
    was = _paths(before) if isinstance(before, type(after)) else {}
    changed = {path: leaf for path, leaf in _paths(after).items()
               if path not in was or context.c(was[path]) != context.c(leaf)}
    if not changed:
        return ValueSource(kind="unchanged"), {}
    leaves = {path: _sourced(leaf, context, turns, clauses, recording, write) for path, leaf in changed.items()}
    return (None if None in leaves.values() else next(iter(leaves.values()))), leaves


def _split_key(key: str) -> tuple[str, str]:
    table, _, row_id = key.partition(".")
    return table, row_id


def attr_of(item: Any, name: str) -> Any:
    """A dict's key or a model's attribute, so a schema reads the same from JSON or from EntitySchema."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def action_tables(schema: Any) -> set[str]:
    """Tables whose columns record tool calls (evidence.actions_of), read as spec/actions.py reads them (D308)."""
    return {str(attr_of(column, "table")) for column in attr_of(schema, "columns") or ()
            if (attr_of(column, "evidence") or {}).get("actions_of")}


def expected_from_run(run_events: Any, user_turns: Iterable[Any], policy_text: Optional[str] = None,
                      canon: Any = None, *, recording: Optional[str] = None, schema: Any = None) -> EndState:
    """The Reference Run's diff, start to end, as cells, each value and row with its source (D315).

    A value is sourced by a user turn whose canonical text contains it, a policy clause holding it
    verbatim, or the call result that first showed it; a row id the same way. Neither found means
    unsupported (`source` or `row_source` None). A list or dict value is sourced by its leaves that
    are new or changed at their path, each the same way, kept in `leaf_sources`; the cell is sourced
    when every such leaf is, and is `unchanged` when none is new (reordered only). Short values (a digit, a yes) are found by plain
    containment and may be sourced by a turn that says them for another reason.

    Neither found, a value a call passed is sourced by a user turn that opened on a yes right after
    the agent stated it; and a value no call passed, on a row a sourced call named or whose result
    showed the row's id, is the world's answer to that call (`tool_result` with `world` in its ptr,
    D323). With a schema, a table whose columns record tool calls (an actions table, D308) gives no
    cells: its rows are conduct, not state, so the whole table is `allowed`.
    """
    context = AtomContext(_as_run(run_events), canon, schema=schema)
    turns, clauses = _turns(user_turns), _clauses(policy_text)
    actions = action_tables(schema) if schema is not None else set()
    separator = attr_of(schema, "key_separator") or "|"
    cells: list[ExpectedCell] = []
    allowed = [{"table": table, "why": "action table: calls are conduct, not state"} for table in sorted(actions)]
    for key, moved in context.diff().items():
        table, row_id = _split_key(key)
        if table in actions:
            continue
        row = (context.end_state.get(table) or {}).get(row_id) or {}
        created = moved["present_after"] and not moved["present_before"]
        write = _touching_write(row_id, row if created else None, context, separator, turns, clauses, recording)
        row_source = _sourced(row_id, context, turns, clauses, recording, write if created else None)
        if not moved["present_after"]:
            cells.append(ExpectedCell(table=table, row_id=row_id, row_source=row_source,
                                      source=row_source))
            continue
        start_row = (context.start_state.get(table) or {}).get(row_id) or {}
        for name in moved["fields"]:
            value, leaves = row.get(name), {}
            if isinstance(value, (list, dict)):
                source, leaves = _nested_source(start_row.get(name), value, context, turns, clauses, recording,
                                                write)
            else:
                source = _sourced(value, context, turns, clauses, recording, write)
            cells.append(ExpectedCell(table=table, row_id=row_id, field=name, value=value, source=source,
                                      row_source=row_source, leaf_sources=leaves))
    return EndState(cells=cells, allowed=allowed)


def alternatives_from_turns(end_state: EndState, user_turns: Iterable[Any], canon: Any = None) -> list[EndState]:
    """One EndState per value where one user turn names two or more values for the same cell.

    The rule is literal: in a turn holding the cell's value as one bare word of "a or b", "either a
    or b" or "a, b or c", each other word is an alternative value for that cell. It does not catch
    values of more than one word, alternatives spread over several turns, alternatives the agent
    offered and the user accepted, or an order of writes (the end state does not see order).
    """
    context = AtomContext(Run(run_id="alternatives"), canon)
    options: list[list[Any]] = []
    for cell in end_state.cells:
        found = [cell.value]
        needle = context.t(cell.value)
        for text in _turns(user_turns):
            for match in ALTERNATIVES.finditer(text):
                words = [w.strip(" ,.") for w in re.split(r"\s*,\s*", match.group(1))] + [match.group(2)]
                if needle and needle in [context.t(w) for w in words]:
                    found += [w for w in words if context.t(w) != needle and w not in found]
        options.append(found)
    if all(len(found) == 1 for found in options):
        return [end_state]
    states = []
    for values in itertools.islice(itertools.product(*options), MAX_END_STATES):
        cells = [cell.model_copy(update={"value": value}) for cell, value in zip(end_state.cells, values, strict=True)]
        states.append(EndState(cells=cells, allowed=list(end_state.allowed)))
    return states


def refusals_of(run_events: Any) -> list[dict]:
    """Every call the recording's tool refused with a message: tool, args, message, event index."""
    run = _as_run(run_events)
    names = {}
    out = []
    for event in run.events:
        payload = event.payload or {}
        if event.type == "tool_call":
            names[payload.get("id")] = (payload.get("name"), payload.get("args") or payload.get("arguments") or {})
        elif event.type == "tool_result" and payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            if message:
                tool, args = names.get(payload.get("id"), (payload.get("name"), {}))
                out.append({"tool": tool or payload.get("name"), "args": args, "message": message,
                            "event_idx": event.idx})
    return out


def forbidden_from(end_state: EndState, user_turns: Iterable[Any], refusals: Iterable[dict] = (),
                   *, start_state: Optional[dict] = None, recording: Optional[str] = None,
                   canon: Any = None) -> list[Forbidden]:
    """The forbidden list from a user negation and a recorded refusal only (D315).

    The collateral rule is not enumerated here: anything outside the expected cells and `allowed`
    moving already fails the end state check. A negation ("do not", "don't", "never") forbids any
    write to a row of `start_state` it names by id outside the expected rows, and a write to a field
    of an expected row it names (underscores read as spaces) that is not itself expected. A recorded
    refusal forbids that tool's write, on the expected row its arguments name when they name one.
    """
    context = AtomContext(Run(run_id="forbidden"), canon)
    start = start_state or {}
    cells = {(c.table, c.row_id, c.field) for c in end_state.cells}
    rows = {(c.table, c.row_id) for c in end_state.cells}
    out: list[Forbidden] = []
    for i, text in enumerate(_turns(user_turns)):
        source = ValueSource(kind="user_turn", ptr={"recording": recording, "turn": i})
        for negation in NEGATION.finditer(text):
            clause = f" {context.t(negation.group(1))} "
            for table, table_rows in sorted(start.items()):
                for row_id, row in sorted((table_rows or {}).items(), key=lambda pair: str(pair[0])):
                    if (table, str(row_id)) in rows:
                        names = [n for n in sorted(row or {}) if (table, str(row_id), n) not in cells
                                 and f" {n.replace('_', ' ').lower()} " in clause]
                        out += [Forbidden(kind="write", table=table, row_id=str(row_id), field=n, source=source)
                                for n in names]
                    elif f" {context.t(row_id)} " in clause:
                        out.append(Forbidden(kind="write", table=table, row_id=str(row_id), source=source))
    row_ids = {row_id for _, row_id in rows}
    for refusal in refusals:
        named = [str(v) for v in (refusal.get("args") or {}).values() if str(v) in row_ids]
        out.append(Forbidden(kind="write", tool=refusal.get("tool"), row_id=named[0] if named else None,
                             source=ValueSource(kind="tool_result", ptr={
                                 "event_idx": refusal.get("event_idx"), "tool": refusal.get("tool"),
                                 "message": refusal.get("message")})))
    return out


@dataclass
class Match:
    """One end state check: `ok` True, False or None (unsettled), and why, readable back."""
    ok: Optional[bool]
    why: list[str] = dataclass_field(default_factory=list)


def _allowed(allowed: list[dict], table: str, row_id: str, name: Optional[str]) -> bool:
    return any(all(rule.get(key) in (None, want) for key, want in
                   (("table", table), ("row_id", row_id), ("field", name))) for rule in allowed)


VALUE_SETS = ("one_of", "not")


def value_set(value: Any) -> Optional[tuple[str, list]]:
    """("one_of" | "not", values) when an expected value is a value set, else None."""
    if isinstance(value, dict) and len(value) == 1:
        (key, values), = value.items()
        if key in VALUE_SETS and isinstance(values, list):
            return key, values
    return None


def path_value(row: Any, field: str) -> Any:
    """The value at a dotted path of a row (a list step is its index), or None when absent."""
    node = row
    for part in str(field).split("."):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def cell_resolution(context: AtomContext, column: str, expected: Any, actual: Any) -> str:
    """How an actual value meets an expected one: a plain value, a list (same members, any order), or a value
    set ("one_of", "not")."""
    if isinstance(expected, list):
        def members(values: list) -> list[str]:
            return sorted(json.dumps(context.c(v), sort_keys=True, default=str) for v in values)
        same = isinstance(actual, list) and members(expected) == members(actual)
        return EQUAL if same else DIFFERENT
    found = value_set(expected)
    if found is None:
        return context.resolution(column, expected, actual)
    key, values = found
    answers = [context.resolution(column, value, actual) for value in values]
    if EQUAL in answers:
        return DIFFERENT if key == "not" else EQUAL
    if UNRESOLVED in answers:
        return UNRESOLVED
    return EQUAL if key == "not" else DIFFERENT


def _new_row_matches(spec: dict, row: Any, context: AtomContext) -> bool:
    where = spec.get("where") or {}
    return isinstance(row, dict) and all(context.c(path_value(row, field)) == context.c(value)
                                         for field, value in where.items())


def _match_new_row(end_state: EndState, context: AtomContext, table: str, row_id: str) -> Optional[int]:
    """The index of the declared new-row spec a made row answers, or None where it answers none."""
    row = (context.end_state.get(table) or {}).get(row_id)
    return next((n for n, s in enumerate(end_state.new_rows)
                 if s.get("table") == table and _new_row_matches(s, row, context)), None)


def new_row_counts(end_state: EndState, context: AtomContext) -> list[tuple[dict, int]]:
    """Each declared new-row spec with how many made rows answered it."""
    made = [0] * len(end_state.new_rows)
    for key, moved in context.diff().items():
        table, row_id = _split_key(key)
        if moved["present_before"] and moved["present_after"]:
            continue
        spec = _match_new_row(end_state, context, table, row_id)
        if spec is not None:
            made[spec] += 1
    return list(zip(end_state.new_rows, made, strict=True))

def cell_results(end_state: EndState, context: AtomContext) -> list[tuple[ExpectedCell, Optional[bool], str]]:
    """Each cell of one end state on the Run's end: held, failed or unsettled (None), and its name.

    A cell on an exempt column holds by definition: the exemption says its value never decides. A
    cell's field may be a dotted path, and its value a list or a value set (`cell_resolution`).
    """
    out: list[tuple[ExpectedCell, Optional[bool], str]] = []
    for cell in end_state.cells:
        row = (context.end_state.get(cell.table) or {}).get(cell.row_id)
        where = f"{cell.table}.{cell.row_id}.{cell.field}"
        if cell.field is None:
            out.append((cell, (row is None) == (cell.value is None), f"cell:{where}"))
            continue
        if f"{cell.table}.{cell.field}" in context.exempt:
            out.append((cell, True, f"cell:{where}"))
            continue
        column = f"{cell.table}.{cell.field}"
        found = cell_resolution(context, column, cell.value, path_value(row or {}, cell.field))
        if found == DIFFERENT or row is None:
            out.append((cell, False, f"cell:{where}"))
        elif found == UNRESOLVED:
            out.append((cell, None, f"unsettled:{cell.table}.{cell.field}"))
        else:
            out.append((cell, True, f"cell:{where}"))
    return out


def _kept_row_moves(end_state: EndState, cells: set, table: str, row_id: str,
                    fields: dict) -> tuple[list[str], list[str]]:
    """(failed, open) names for fields of a kept row that moved outside the declared cells."""
    failed, open_pairs = [], []
    for name, change in fields.items():
        if (table, row_id, name) in cells or _allowed(end_state.allowed, table, row_id, name):
            continue
        if change.get("unresolved"):
            open_pairs.append(f"unsettled:{table}.{name}")
        else:
            failed.append(f"collateral:{table}.{row_id}.{name}")
    return failed, open_pairs


def collateral(end_state: EndState, context: AtomContext) -> Match:
    """The sanity item (D329): nothing outside the end state's declared rows, cells and `allowed` moved.

    A row made or removed is declared by any cell on it or by a declared new row; a row kept is
    declared field by field (a dotted cell declares its top-level column). Whether a declared new
    row was made as many times as it says is its own item, not sanity.
    """
    failed: list[str] = []
    open_pairs: list[str] = []
    cells = {(c.table, c.row_id, str(c.field).split(".")[0] if c.field is not None else None)
             for c in end_state.cells}
    touched = {(table, row_id) for table, row_id, _ in cells}
    for key, moved in context.diff().items():
        table, row_id = _split_key(key)
        if not (moved["present_before"] and moved["present_after"]):
            # A row made or removed: any cell on it speaks for it, a declared new row answers it.
            if (table, row_id) in touched or _allowed(end_state.allowed, table, row_id, None):
                continue
            if _match_new_row(end_state, context, table, row_id) is None:
                failed.append(f"collateral:{table}.{row_id}")
            continue
        more, open_here = _kept_row_moves(end_state, cells, table, row_id, moved["fields"])
        failed += more
        open_pairs += open_here
    if failed:
        return Match(False, failed)
    if open_pairs:
        return Match(None, open_pairs)
    return Match(True, [])


def match_context(end_state: EndState, context: AtomContext) -> Match:
    """`match` on a context the caller already built, so the Verdict reads its own copy."""
    cells = cell_results(end_state, context)
    rest = collateral(end_state, context)
    failed = [why for _, ok, why in cells if ok is False] + (rest.why if rest.ok is False else [])
    open_pairs = [why for _, ok, why in cells if ok is None] + (rest.why if rest.ok is None else [])
    if failed:
        return Match(False, failed)
    if open_pairs:
        return Match(None, open_pairs)
    return Match(True, [])


def match(end_state: EndState, start: dict, end: dict, canon: Any = None, *, schema: Any = None,
          rules: Any = None, equivalence: Any = None) -> Match:
    """True when every cell holds in `end` and nothing outside the cells and `allowed` moved from
    `start`; False when a cell fails or collateral moved; None when a pair cannot be settled (D219).
    """
    run = Run(run_id="match", events=[Event(idx=0, type="stop",
                                            payload={"start_state": start, "end_state": end})])
    return match_context(end_state, AtomContext(run, canon, schema=schema, rules=rules, equivalence=equivalence))


def _named(needle: str, text: str) -> bool:
    """The canonical value said as a whole word or phrase of the canonical turn, not inside another."""
    return bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text) is not None


def _column_path(path: str) -> str:
    """A leaf path without its list positions ("items.0.id" reads "items.id"), so rows compare."""
    return ".".join(part for part in path.split(".") if not part.isdigit())


def _domain(start_state: dict, table: str, name: str, path: Optional[str]) -> list:
    """Every value the column (at this leaf path, for a nested value) holds anywhere in the start state."""
    out = set()
    for row in ((start_state or {}).get(table) or {}).values():
        value = (row or {}).get(name) if isinstance(row, dict) else None
        if path is None:
            if not isinstance(value, (dict, list)):
                out.add(json.dumps(value, sort_keys=True, default=str))
            continue
        out.update(json.dumps(leaf, sort_keys=True, default=str) for leaf_path, leaf in _paths(value).items()
                   if _column_path(leaf_path) == path)
    return [json.loads(item) for item in out]


def contradicts_user(end_state: EndState, user_turns: Iterable[Any], start_state: Optional[dict],
                     canon: Any = None) -> list[str]:
    """Each cell whose value is not one the user named, where the user named some value of its column.

    The column's domain is every value it holds anywhere in the start state (for a nested value, every
    leaf at the same path, list positions ignored), so a word of the user's is a value of the column
    only when the column holds it. A cell (or changed leaf) contradicts the user when their own turns
    name one or more domain values and its value is none of them. A column the user named no value of
    is not contradicted: that is availability, which the source already speaks to. Values compare by
    canonical text; a semantic pair is not settled here (it is the end state match's question).
    """
    context = AtomContext(Run(run_id="contradicts"), canon)
    turns = [context.t(text) for text in _turns(user_turns)]
    start = start_state or {}
    out: list[str] = []
    for cell in end_state.cells:
        if cell.field is None:
            continue
        start_row = ((start.get(cell.table) or {}).get(cell.row_id) or {})
        before = start_row.get(cell.field) if isinstance(start_row, dict) else None
        if isinstance(cell.value, (dict, list)):
            was = _paths(before) if isinstance(before, type(cell.value)) else {}
            leaves = {path: leaf for path, leaf in _paths(cell.value).items()
                      if path not in was or context.c(was[path]) != context.c(leaf)}
        else:
            leaves = {None: cell.value}
        for path, value in leaves.items():
            column = None if path is None else _column_path(path)
            domain = {context.t(v) for v in _domain(start, cell.table, cell.field, column)} - {"", "none", "null"}
            named = {v for v in domain if any(_named(v, text) for text in turns)}
            if named and context.t(value) not in named:
                where = f"{cell.table}.{cell.row_id}.{cell.field}" + (f".{path}" if path else "")
                out.append(where)
    return out


def unasked_writes(run_events: Any, user_turns: Iterable[Any] = (), write_tools: Optional[Iterable[str]] = None
                   ) -> list[str]:
    """Each successful write call no user turn before it asked for, as "tool@event index".

    Asked means a user turn before the call names one of its argument values (a row id or a scalar
    argument, canonical text, whole words) or opens a sentence on a yes with none on a no (D306's
    reading). Turn order comes from the Run's own user turns; `user_turns` stands in, all before every
    call, only when the Run carries none. No write tools named means no call is read as a write.
    """
    context = AtomContext(_as_run(run_events), None)
    turns = context.user or [(-1, text) for text in _turns(user_turns)]
    tools = set(write_tools or ())
    out: list[str] = []
    for call in context.calls:
        if call["name"] not in tools or call["error"]:
            continue
        before = [context.t(text) for idx, text in turns if idx < call["idx"]]
        values = {context.t(leaf) for leaf in _leaves(call["args"])} - {"", "none", "null", "true", "false"}
        if any(_named(value, text) for value in values for text in before):
            continue
        sentences = [part.strip() for idx, text in turns if idx < call["idx"]
                     for part in re.split(r"[.!?\n]+", text) if part.strip()]
        if any(AFFIRM_OPEN.match(s) for s in sentences) and not any(NEGATIVE_OPEN.match(s) for s in sentences):
            continue
        out.append(f"{call['name']}@{call['idx']}")
    return out
