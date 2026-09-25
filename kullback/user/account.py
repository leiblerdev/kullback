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
    """This user's choices by option kind, in the three layers, nearest first."""

    def __init__(self, own: Optional[Mapping[str, str]] = None,
                 others: Optional[Mapping[str, str]] = None,
                 population: Optional[Mapping[str, tuple[str, int, int]]] = None):
        self.layers: list[tuple[str, dict[str, Any]]] = [
            (RECORDING, {_key(k): v for k, v in (own or {}).items()}),
            (OTHER_RECORDING, {_key(k): v for k, v in (others or {}).items()}),
            (POPULATION, {_key(k): v for k, v in (population or {}).items()}),
        ]

    def answer(self, kind: str) -> tuple[Optional[str], Optional[str], str]:
        """The value, its source and the line the tool says, or no value and the no-record line."""
        wanted = _key(kind)
        for source, choices in self.layers:
            found = choices.get(wanted)
            if found is None:
                continue
            if source == POPULATION:
                value, count, total = found
                return value, source, (f"{kind}: {value} (source: {source}, "
                                       f"{count} of {total} recorded users chose it)")
            return str(found), source, f"{kind}: {found} (source: {source})"
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


def account_view(tables: Mapping[str, Any], identity: Mapping[str, Any],
                 columns: Iterable[str]) -> AccountView:
    """This customer's rows of `tables`, snapshotted now: the row its identity matches, and the rows
    that carry that row's id. No identity match is no account, never a guess."""
    from kullback.user.simulated import _matching_rows
    if not identity:
        return AccountView((), columns)
    matches = _matching_rows(tables, dict(identity))
    if len(matches) > 1:  # an id names one row; other tables only carry it
        named = {str(value) for value in identity.values()}
        matches = [match for match in matches if match[0] in named]
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
