"""What the agent user may look up about itself beyond its facts: its choices and its own account.

A recorded person offered two ways to do the thing picks one, and a person asked for a detail of
their own account can usually read it off a screen. The rule-driven user has neither: a choice the
regex cues miss reads as "nothing was asked", and it restates the goal and closes. So the agent user
gets two readings, both mined and both tagged with where the answer came from, so the turn record
tells a mined value from a recorded one.

Choices answer in a fixed order, nearest first: this Task's recording, then the same customer's
other recordings, then what the corpus's recorded users chose for the same kind of option. With none
of the three there is no answer, never an invented one. The population layer carries only word-shaped
values: an id another customer chose is that customer's, not a kind of option.

The account is read only and code-limited twice over. Rows: only this Task's customer's, the row the
rules' identity matches the way the rule-driven user matches it, and the rows that carry its id.
Columns: only the ones some recorded user of the corpus ever stated in a turn, mined once per workdir,
so a column no customer ever said stays hidden however it is asked for. The snapshot is the Starting
state as the Run opened, taken before the Candidate's first call.

Nothing here names a corpus, a table or a column.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from kullback.runner.records import Trace, UserRules
from kullback.user import rules as rules_mod
from kullback.user.context import MIN_RECORD_VALUE, _keyed_leaves, reference_choices

# Where an answer came from. Tagged on the tool's answer and on the turn record, so derivation can
# tell a mined value from one the recorded user said.
RECORDING = "recording"
OTHER_RECORDING = "other_recording"
POPULATION = "population"
ACCOUNT = "account"
CHOICE_SOURCES = (RECORDING, OTHER_RECORDING, POPULATION)

NO_CHOICE = "You have no record of choosing that."
NO_ACCOUNT = "Your account shows nothing for that."


def _key(text: Any) -> str:
    """One option kind or column as two spellings of it compare: lower case, words joined."""
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _value_shaped(value: str) -> bool:
    from kullback.user.guards import value_tokens
    return bool(value_tokens(value))


class ChoiceBook:
    """This user's choices by option kind, in the three layers, nearest first.

    Each layer keeps the kind's recorded spelling beside its value, in the order the
    recordings gave them, so a kind asked for in other words can be answered with the
    kinds held.
    """

    def __init__(self, own: Optional[Mapping[str, str]] = None,
                 others: Optional[Mapping[str, str]] = None,
                 population: Optional[Mapping[str, tuple[str, int, int]]] = None):
        self.layers: list[tuple[str, dict[str, tuple[str, Any]]]] = [
            (RECORDING, {_key(k): (str(k), v) for k, v in (own or {}).items()}),
            (OTHER_RECORDING, {_key(k): (str(k), v) for k, v in (others or {}).items()}),
            (POPULATION, {_key(k): (str(k), v) for k, v in (population or {}).items()}),
        ]

    def held(self) -> list[tuple[str, str]]:
        """Every kind held, as its recorded spelling and source, nearest layer first."""
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for source, choices in self.layers:
            for norm, (shown, _) in choices.items():
                if norm not in seen:
                    seen.add(norm)
                    out.append((shown, source))
        return out

    def answer(self, kind: str) -> tuple[Optional[str], Optional[str], str]:
        """The value, its source and the line the tool says, or no value and the held kinds."""
        wanted = _key(kind)
        for source, choices in self.layers:
            found = choices.get(wanted)
            if found is None:
                continue
            shown, payload = found
            if source == POPULATION:
                value, count, total = payload
                return value, source, (f"{kind}: {value} (source: {source}, "
                                       f"{count} of {total} recorded users chose it)")
            return str(payload), source, f"{kind}: {payload} (source: {source})"
        held = ", ".join(f"{shown} (source: {source})" for shown, source in self.held())
        if held:
            return None, None, f"{NO_CHOICE} What you hold: {held}."
        return None, None, NO_CHOICE


def _keyish(value: str) -> bool:
    """A value that keys one customer: anything but plain words, since two customers can share a
    name and cannot share an address of mail or an id."""
    return bool(value) and re.fullmatch(r"[A-Za-z' .-]+", value) is None


def customer_keys(user_rules: Optional[UserRules], identity_fields: Iterable[str]) -> list[str]:
    """The keying values the rules hold for this customer's identity, which key its recordings."""
    fields = set(identity_fields)
    out: list[str] = []
    for fact in (user_rules.facts if user_rules is not None else ()):
        value = str(fact.value or "").strip()
        if fact.field in fields and _keyish(value) and value not in out:
            out.append(value)
    return out


def id_keys(user_rules: Optional[UserRules], identity_fields: Iterable[str] = ()) -> list[str]:
    """The keying values the rules hold outside the person-shaped identity: an id, which two
    customers cannot share. A customer keyed by an id alone holds no person-shaped key, so
    the account lookup reads these where the identity reads nothing."""
    fields = set(identity_fields)
    out: list[str] = []
    for fact in (user_rules.facts if user_rules is not None else ()):
        value = str(fact.value or "").strip()
        if fact.field not in fields and _keyish(value) and value not in out:
            out.append(value)
    return out

def choice_book(reference: Optional[Trace], write_tools: Iterable[str], *,
                customer: Iterable[str] = (), traces: Optional[Mapping[str, Trace]] = None,
                rules: Optional[Mapping[str, UserRules]] = None,
                identity_fields: Iterable[str] = ()) -> ChoiceBook:
    """The three layers for one Task, mined from the workdir's recordings (D151's choices, widened).

    `customer` are this customer's keys (`customer_keys`); another recording is the same customer's
    when its rules hold one of them.
    """
    writes = list(write_tools)
    traces = traces or {}
    rules = rules or {}
    fields = list(identity_fields)
    mine = {str(value).lower() for value in customer}
    own_id = reference.trace_id if reference is not None else None
    others: dict[str, str] = {}
    counts: dict[str, Counter] = {}
    for trace_id in sorted(traces):
        trace = traces[trace_id]
        chosen = reference_choices(trace, writes)
        for kind, value in chosen.items():
            if not _value_shaped(value):
                counts.setdefault(kind, Counter())[value] += 1
        if trace_id == own_id or not mine:
            continue
        theirs = {value.lower() for value in customer_keys(rules.get(trace_id), fields)}
        if theirs & mine:
            for kind, value in chosen.items():
                others.setdefault(kind, value)
    population: dict[str, tuple[str, int, int]] = {}
    for kind, counter in counts.items():
        value, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
        population[kind] = (value, count, sum(counter.values()))
    return ChoiceBook(reference_choices(reference, writes), others, population)


def stated_columns(traces: Mapping[str, Trace]) -> frozenset[str]:
    """The columns some recorded user stated in a turn: a value a tool returned that a user said.

    Mined once per workdir. A column no customer ever said is a column a person on the phone does
    not read off their own screen, so the account never shows it.
    """
    out: set[str] = set()
    for trace in traces.values():
        said = [turn.content for turn in trace.turns if turn.role == "user" and turn.content]
        if not said:
            continue
        for call in trace.tool_calls:
            for column, value in _keyed_leaves(call.result):
                if column in out or isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    continue
                text = str(value).strip()
                if len(text) >= MIN_RECORD_VALUE and any(rules_mod._said_in(turn, text) for turn in said):
                    out.add(column)
    return frozenset(out)


class AccountView:
    """One customer's rows of the Starting state, cut to the stated columns, read only."""

    def __init__(self, rows: Iterable[tuple[str, dict]] = (), columns: Iterable[str] = ()):
        self.columns = frozenset(columns)
        self.lines: list[tuple[str, str, str]] = []   # table, column, value
        for table, row in rows:
            for column, value in _keyed_leaves(row):
                if column not in self.columns or isinstance(value, (dict, list)) or value is None:
                    continue
                text = str(value).strip()
                if text:
                    self.lines.append((table, column, text))

    def read(self, what: str = "") -> tuple[str, dict[str, str]]:
        """The lines that answer `what` (a column or table, any spelling; empty is everything) and
        the values given, by value, with their column, for the guard."""
        wanted = _key(what)
        hits = [(table, column, value) for table, column, value in self.lines
                if not wanted or wanted in _key(column) or wanted in _key(table)]
        if not hits:
            return NO_ACCOUNT, {}
        given = {value: column for _, column, value in hits}
        return "\n".join(f"{table} {column}: {value} (source: {ACCOUNT})"
                         for table, column, value in hits), given


def _named_rows(tables: Mapping[str, Any], keys: Iterable[str]) -> list[tuple[str, dict]]:
    """The rows the keys name by row id, in table order. Each key is tried on its own."""
    wanted = {str(key).strip().lower() for key in keys or () if str(key).strip()}
    named: list[tuple[str, dict]] = []
    for body in (tables or {}).values():
        for row_id, row in (body.items() if isinstance(body, dict) else []):
            if isinstance(row, dict) and str(row_id).strip().lower() in wanted:
                named.append((str(row_id), row))
    return named


def _owner_of(named: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """One named row, or the one another named row carries as a plain value, else nothing.

    A row carries its owner's id on it, while the owner carries what it points at inside
    a list. Anything else is a guess, so nothing.
    """
    if len(named) == 1:
        return named
    if not named:
        return []
    owners = [row_id for row_id, _ in named
              if any(str(value).strip().lower() == row_id.lower()
                     for other_id, other in named if other_id.lower() != row_id.lower()
                     for _, value in _plain_leaves(other))]
    if len(owners) == 1:
        return [(row_id, row) for row_id, row in named if row_id.lower() == owners[0].lower()]
    return []


def _keyed_owner(tables: Mapping[str, Any], keys: Iterable[str]) -> list[tuple[str, dict]]:
    """The one row a key names by row id, or nothing where the keys name none or two.

    Where two keys each name a row, the one another row carries as a plain value is the
    owner. Anything else is a guess, so nothing.
    """
    return _owner_of(_named_rows(tables, keys))


def _plain_leaves(row: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """The scalar leaves a row carries as plain values, leaving out what it points at in lists."""
    yield from _walk_plain(row, key="", in_list=False)


def _walk_plain(value: Any, key: str, in_list: bool) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for inner_key, inner in value.items():
            yield from _walk_plain(inner, str(inner_key), in_list)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk_plain(inner, key, True)
    elif key and value is not None and not in_list and not isinstance(value, (dict, list)):
        yield key, value


def _owner_match(tables: Mapping[str, Any], identity: Mapping[str, Any],
                 keys: Iterable[str] = ()) -> list:
    """This customer's row by identity, or by key where the identity names none.

    An identity matching more than one row keeps only the rows its own values name;
    an id names one row and other tables only carry it. Where that leaves no single
    row, each key is tried as a row id on its own.
    """
    from kullback.user.simulated import _matching_rows
    matches = _matching_rows(tables, dict(identity)) if identity else []
    if len(matches) > 1:  # an id names one row; other tables only carry it
        named = {str(value) for value in identity.values()}
        matches = [match for match in matches if match[0] in named]
    if len(matches) != 1:
        matches = _keyed_owner(tables, keys)
    return matches


def account_view(tables: Mapping[str, Any], identity: Mapping[str, Any],
                 columns: Iterable[str], keys: Iterable[str] = ()) -> AccountView:
    """This customer's rows of `tables`, snapshotted now: the row its identity matches, and the rows
    that carry that row's id. No identity match is no account, never a guess.

    `keys` are the ids the rules hold outside the person-shaped identity. Where the identity
    names no row, each key is tried as a row id on its own; one row so named is its owner,
    with the same carried rows. Two keys naming two rows is no account, unless the rows
    themselves say which carries which: the one another row carries as a plain value is the
    owner, the one carried only inside a list is what it points at.
    """
    if not identity and not keys:
        return AccountView((), columns)
    matches = _owner_match(tables, identity, keys)
    if len(matches) != 1:
        return AccountView((), columns)
    owner_id = matches[0][0].lower()
    rows: list[tuple[str, dict]] = []
    for table in sorted(tables or {}):
        body = tables[table]
        for row_id, row in (body.items() if isinstance(body, dict) else []):
            if not isinstance(row, dict):
                continue
            carries = any(str(value).strip().lower() == owner_id
                          for _, value in _keyed_leaves(row) if not isinstance(value, (dict, list)))
            if str(row_id).lower() == owner_id or carries:
                rows.append((str(table), copy.deepcopy(row)))
    return AccountView(rows, columns)
