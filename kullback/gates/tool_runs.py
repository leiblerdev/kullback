"""The eight rulings over what a generated tool body did when the sandbox ran it (design section 6).

`builder/sandbox.py` runs a body in a subprocess and hands back one result dict per recorded call;
nothing here starts a process. Each ruling takes the calls and those results (or the sandbox's
error, when the module did not load or timed out) and decides: does the module parse, did every
call run without crashing, do two fresh runs agree, do different arguments give different answers,
do the recorded calls replay to their recorded results, and does a write refuse a reference the
world does not hold. The sandbox's gate functions are thin wrappers that run and then call one of
these, so the accept-or-reject decision is in this package and hashed with it (D122) while the
subprocess stays where it was.

The seventh, `body_memorised_values_gate` (D162), runs no calls at all: it reads the body's own
source and refuses a literal that is data the recordings carried rather than code the body needs.

The eighth, `body_sensitivity_gate` (D195), rules over calls the sandbox has already run: two
recorded calls of one tool made under two Tasks, answered differently by the recording, have to be
answered differently by the body too, in the columns the recordings part in.

The row helpers (`parse_result`, `match_table`, `row_key`, `columns_of`, `id_field`, `key_fields`,
`id_pattern_for`) live here because the replay ruling compares rows column by column under the
schema's classes (D73, D84); `sandbox.py` and `compile_env.py` read them back from here.
"""

from __future__ import annotations

import ast
import json
import re
import weakref
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates.confinement import TOOLS_CLASS, function_confinement
from kullback.runner.canon import (
    DIFFERS_NOTE,
    EMPTY_CANON,
    EXEMPT_NOTE,
    PRESENCE_NOTE,
    SEMANTIC_NOTE,
    TOKEN_NOTE,
    UNRESOLVED,
    UNRESOLVED_NOTE,
    class_of,
    first_difference,
    resolution_of,
)
from kullback.runner.canon import canonicalize as canon
from kullback.runner.canon import compare as compare_column
from kullback.runner.gate_support import _share
from kullback.runner.records import EntitySchema, GateResult, ToolCall, content_hash

CRASH_ERRORS = frozenset({"NameError", "AttributeError", "TypeError", "ImportError",
                          "ModuleNotFoundError", "IndentationError", "SyntaxError", "RecursionError"})
MEMORISED_STAGE = "compile_tools.memorised_values"
SENSITIVITY_STAGE = "sensitivity"
TOOL_RUN_STAGES = ("parses", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity",
                   "refuses_unknown", MEMORISED_STAGE, SENSITIVITY_STAGE)
# A note that names a column class is a reading, not a failure: an exempt column is equal whatever
# it holds and a semantic one is reported and left to the judge (D73, D84). Every other note the
# comparison leaves is a hard column that parted, and those are what fail a ruling: a hard column's
# leaf, a value only one side holds, and two values made of different domain tokens (D217).
CLASS_NOTES = (EXEMPT_NOTE, SEMANTIC_NOTE)
# What a build counts about its semantic columns (D219), in the order a reader wants them: how many
# were compared, how many of those a judge was actually asked about, and how the three-valued
# answer came out. A round writes these and a rate of unresolved over compared is the number that
# says whether the judge is wired at all.
SEMANTIC_COUNTS = ("semantic_compared", "semantic_judged", "semantic_equal", "semantic_different",
                   "semantic_unresolved")
# --- the words a column's world uses, which a forgiven difference may not change (D217) ---
ENUM_DISTINCT = 12   # a column the corpus showed this many distinct values or fewer draws from a set
TOKEN_MIN = 2        # a one-character value stands inside half the words there are and names nothing
TOKEN_MAX = 60       # a longer value is a sentence, not one of the names a column draws from
TOKEN_LIMIT = 400    # names read for one comparison, so a wide table cannot make this quadratic
WORD_EDGE = re.compile(r"[0-9A-Za-z]")
# A prose result the reader read no columns out of falls back to comparing the whole string, and
# the failure line says which of the two routes it took (D187).
STRING_ROUTE = "string differs"
COLUMN_ROUTE = "read as columns"


# --- reading rows out of recorded tool results: shared with compile_env.py's inverse replay ---

def parse_result(result: Any) -> Any:
    """A recorded result is a value, or the JSON text of one."""
    if isinstance(result, str) and (result.strip()[:1] in "[{"):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def columns_of(schema: EntitySchema, table: str, kind: Optional[str] = None) -> list[str]:
    """Column names of one table, all of them or only those of one class (D73)."""
    return sorted(c.name for c in schema.columns if c.table == table and (kind is None or c.class_ == kind))


def id_pattern_for(schema: EntitySchema, table: str, name: Optional[str] = None) -> Optional[str]:
    """The shape a table's ids take, under either key the schema may hold it by.

    `mine_schema` records a pattern per column, keyed `table.column`; a schema written by hand
    keys it by the table alone. Reading only the second key is what made this check dead code on
    every mined schema: the lookup missed, the pattern came back None, and every candidate row
    passed the guard whatever its id looked like.
    """
    patterns = schema.id_patterns or {}
    if name and f"{table}.{name}" in patterns:
        return patterns[f"{table}.{name}"]
    return patterns.get(table)


def id_field(schema: EntitySchema, table: str) -> Optional[str]:
    """The column holding a row's id, by the customer's own naming.

    The three name candidates first, because they are the customer's own convention where they
    apply. Then the columns the miner recorded a pattern for, which is where an id the name rule
    cannot see arrives: airline's `flights` are addressed by `flight_number`, and reading only the
    `_id` names left the table proposed and empty.
    """
    names = set(columns_of(schema, table))
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    for candidate in (f"{singular}_id", "id", f"{table}_id"):
        if candidate in names:
            return candidate
    mined = [key.split(".", 1)[1] for key in sorted(schema.id_patterns or {})
             if key.startswith(f"{table}.") and key.split(".", 1)[1] in names]
    preferred = [n for n in mined if n.startswith(singular)]
    return next(iter(preferred or mined), None) or next((n for n in sorted(names) if n.endswith("_id")), None)


def key_fields(schema: EntitySchema, table: str) -> list[str]:
    """The columns a row's key is made of: the composite key the schema names, or the id column.

    Every table is keyed by its id column until the miner finds one id standing for several rows
    (`mine.composite_keys`), so this answers a one-element list on a schema that names no composite
    key, which is what every reader saw before composite keys existed.
    """
    composite = [str(name) for name in (getattr(schema, "composite_keys", None) or {}).get(table) or ()]
    if composite:
        return composite
    name = id_field(schema, table)
    return [name] if name else []


def key_separator(schema: EntitySchema) -> str:
    """The string that joins a composite key's values into the row id (D74's rows stay strings)."""
    return getattr(schema, "key_separator", None) or "|"


def row_key(schema: EntitySchema, table: str, row: dict, args: Optional[dict] = None) -> Optional[str]:
    """A row's key: its id, or the composite key's values joined; None when it carries no id.

    A key column the row leaves out or answers null is taken from the arguments of the call that
    returned the row, because a customer's tool can take the value that says which row this is from
    the call and not repeat it in the row (a search that answers rows for the date it was asked
    about, and leaves the date column null). What neither the row nor the call supplies is empty in
    the key, and `partial_key` is what tells a reader the sighting is missing a part of its key.
    """
    fields = key_fields(schema, table)
    if not fields or not isinstance(row.get(fields[0]), str):
        return None
    parts = [row[fields[0]]]
    for name in fields[1:]:
        value = row.get(name)
        if value is None and args:
            value = args.get(name)
        parts.append("" if value is None else str(value))
    return key_separator(schema).join(parts)


def partial_key(schema: EntitySchema, table: str, key: str) -> bool:
    """Whether a key is missing a part: a sighting of the row without what says which row it is."""
    fields = key_fields(schema, table)
    if len(fields) < 2:
        return False
    parts = key.split(key_separator(schema))
    return len(parts) != len(fields) or any(part == "" for part in parts[1:])


def match_table(schema: EntitySchema, value: Any, args: Optional[dict] = None) -> Optional[tuple[str, str]]:
    """Which table a returned row belongs to, and its key; None when the value is not a row."""
    if not isinstance(value, dict):
        return None
    best, best_score = None, 1
    for table in sorted(schema.tables):
        name = id_field(schema, table)
        if not name or not isinstance(value.get(name), str):
            continue
        pattern = id_pattern_for(schema, table, name)
        if pattern and not re.match(pattern, value[name]):
            continue
        score = len(set(columns_of(schema, table)) & set(value))
        if score > best_score:
            key = row_key(schema, table, value, args)
            if key is not None:
                best, best_score = (table, key), score
    return best


# --- the columns a prose result asserts, so two of them compare under the schema's classes (D187) ---

def _last_function(source: str) -> Optional[str]:
    """The name of the last function the source defines at its top level, or None when it defines none."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return names[-1] if names else None


class ResultReaders:
    """Per tool, the function that reads its prose result into the column values the string asserts.

    Some tools answer with one sentence that folds several columns together: a confirmation and a
    device's state in one line. The readers stage (D176) has a model write `read(result)` per such
    tool and holds it to a round trip against every recorded result, and the schema classes the
    columns it reads the way it classes any other (D73). Nothing consulted that when a replayed
    answer was scored: the whole sentence was compared as one hard value, so a drift in a semantic
    sub-field failed the call and the classes were dead code for prose.

    This runs those readers where the comparison runs. The source is checked by the same
    confinement rules a tool body is (`function_confinement`) before it is executed, and a reader
    that will not parse, will not confine, raises, or answers anything but a non-empty dict simply
    does not apply: the caller falls back to comparing the strings, which is what it did before.
    Readings are cached per (tool, text), since a reader is a function of the string alone.
    """

    def __init__(self, sources: Optional[dict] = None, tables: Optional[dict] = None):
        self._sources = {str(k): str(v) for k, v in (sources or {}).items()}
        self._tables = {str(k): str(v) for k, v in (tables or {}).items()}
        self._functions: dict[str, Any] = {}
        self._readings: dict[tuple[str, str], Optional[dict]] = {}

    def __bool__(self) -> bool:
        return bool(self._sources)

    @property
    def tools(self) -> list[str]:
        return sorted(self._sources)

    def table_of(self, tool: str) -> Optional[str]:
        """The table whose columns this tool's reader answers, or None when it has no reader."""
        return self._tables.get(tool) or None if tool in self._sources else None

    def _function(self, tool: str) -> Any:
        if tool in self._functions:
            return self._functions[tool]
        source = self._sources.get(tool) or ""
        name = _last_function(source)
        found = None
        if name is not None and not function_confinement(source):
            namespace: dict = {"__name__": "reader_source"}
            try:
                exec(compile(source, "<reader>", "exec", dont_inherit=True), namespace)  # noqa: S102
                candidate = namespace.get(name)
                found = candidate if callable(candidate) else None
            except Exception:  # a reader that will not load does not apply, and nothing else fails
                found = None
        self._functions[tool] = found
        return found

    def read(self, tool: str, text: Any) -> Optional[dict]:
        """The columns this tool's result asserts, or None when no reader applies or it reads nothing."""
        if not isinstance(text, str) or tool not in self._sources:
            return None
        key = (tool, text)
        if key in self._readings:
            return self._readings[key]
        function = self._function(tool)
        value: Any = None
        if function is not None:
            try:
                value = function(text)
            except Exception:  # a reader that raises on a result is a reader that does not apply
                value = None
        reading = value if isinstance(value, dict) and value else None
        self._readings[key] = reading
        return reading


def load_readers(artifact: Any) -> ResultReaders:
    """The readers held in the `readers` artifact, per tool, with the table their columns belong to."""
    body = artifact if isinstance(artifact, dict) else {}
    sources: dict[str, str] = {}
    tables: dict[str, str] = {}
    for _requestor, proposal in sorted((body.get("proposals") or {}).items()):
        if not isinstance(proposal, dict):
            continue
        table = str(proposal.get("table") or "")
        for reader in proposal.get("readers") or []:
            tool = str((reader or {}).get("tool") or "") if isinstance(reader, dict) else ""
            source = str((reader or {}).get("source") or "") if isinstance(reader, dict) else ""
            if tool and source:
                sources[tool] = source
                tables[tool] = table
    return ResultReaders(sources, tables)


def hard_notes(notes: Iterable[str]) -> list[str]:
    """The notes that fail a ruling: every one that does not name a class the ruling forgives."""
    return [note for note in notes if not note.startswith(CLASS_NOTES)]


# --- domain tokens: the names the world uses, which no forgiveness may change (D217) ---

def _token(value: Any, rules: Any = None) -> Optional[str]:
    """One recorded value as a domain token, or None when it is not one of the world's names.

    A name carries a letter and stands on its own: a bare number is a quantity that means nothing
    without the column it sits in, a single character stands inside half the words there are, and a
    long string is a sentence rather than a name.
    """
    if not isinstance(value, str):
        return None
    text = canon(value, rules)
    if not TOKEN_MIN <= len(text) <= TOKEN_MAX or not any(char.isalpha() for char in text):
        return None
    return text


# The suffixes a column that holds an id is named with, which is the same reading the miner takes
# before it records a vocabulary at all. An id is addressed rather than named, so it lends nothing.
ID_SUFFIXES = ("_id", "_number", "_no", "_code", "_key", "_ref", "_uuid")


def _holds_ids(column: Any, schema: Any = None) -> bool:
    """Whether this column holds ids rather than names, by its name, its table's key or its values.

    The name rule is the customer's convention and is free wherever it fires. A column the corpus
    showed keying its table is an id whatever it is called, which is where a `sku` or a `username`
    is caught. And a column whose every kept sighting has a mined id shape is holding ids however
    it is named, which is the reading that does not depend on a convention at all.
    """
    name = str(getattr(column, "name", "") or "")
    if name == "id" or name.endswith(ID_SUFFIXES):
        return True
    keys = (getattr(schema, "composite_keys", None) or {}).get(getattr(column, "table", None)) or ()
    if name in keys:
        return True
    present = [value for value in (getattr(column, "samples", None) or []) if value is not None]
    return bool(present) and all(isinstance(value, str) and _pattern_hit(schema, value) is not None
                                 for value in present)


def _enumerated(column: Any, rules: Any = None, schema: Any = None) -> bool:
    """Whether the corpus showed this column drawing from a set of names.

    The miner records the set where it found one (`Column.vocabulary`). A schema mined before it did
    still keeps a few sightings per column, and those stand in, held to the miner's own three tests
    rather than to one of them: every value a short name; an id column lending nothing, whatever its
    ids look like and however it is named; and a set of names repeating, so a column whose every
    kept sighting was a value of its own is holding data however short it is, where nothing counted
    what it held. Without the last two a build carried over a schema mined before D217 read an id
    column as a set of names, lent those ids to every free text column of the table, and failed
    sound replays on ids that differ incidentally. The stand-in is deliberately the stricter
    reading: where it says no, the comparison is what it was before D217, and a fresh mine puts the
    real vocabulary back.
    """
    if list(getattr(column, "vocabulary", None) or ()):
        return True
    present = [value for value in (getattr(column, "samples", None) or []) if value is not None]
    if not present or _holds_ids(column, schema):
        return False
    distinct = (getattr(column, "evidence", None) or {}).get("distinct")
    if isinstance(distinct, int):
        if distinct > ENUM_DISTINCT:
            return False
    elif len({canon(str(value), rules) for value in present}) >= len(present):
        # No count of what the column held, so the kept sightings are the only evidence that it
        # repeats at all, and a column whose every sighting was a value of its own does not.
        return False
    return all(_token(value, rules) is not None for value in present)


def _names_of(column: Any) -> list:
    """Every value of this column the schema kept: its own vocabulary, else its sightings."""
    return list(getattr(column, "vocabulary", None) or getattr(column, "samples", None) or ())


def domain_tokens(schema: Any, table: Optional[str], name: str, rules: Any = None) -> frozenset:
    """The names this column's world is known to use, for the token set test of D217.

    A column that draws from a set of names lends its own. A free text column lends its table's,
    because a sentence about a row names that row's states with the words the world stores them
    under: "the line is disabled" is written out of the same vocabulary the state column holds.
    A world that lent no names at all answers the empty set, and then two values always carry the
    same tokens and the comparison is what it was before this rule.
    """
    columns = [c for c in (getattr(schema, "columns", None) or ())
               if table is None or getattr(c, "table", None) == table]
    own = next((c for c in columns if getattr(c, "name", None) == name), None)
    lenders = [own] if own is not None and _enumerated(own, rules, schema) else [
        c for c in columns if _enumerated(c, rules, schema)]
    tokens: set = set()
    for column in lenders:
        for value in _names_of(column):
            token = _token(value, rules)
            if token is not None:
                tokens.add(token)
            if len(tokens) >= TOKEN_LIMIT:
                return frozenset(tokens)
    return frozenset(tokens)


_TOKEN_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def tokens_for(schema: Any, table: Optional[str], name: str, rules: Any = None) -> frozenset:
    """`domain_tokens` held per schema, since one comparison asks for the same column many times."""
    try:
        per_schema = _TOKEN_CACHE.setdefault(schema, {})
    except TypeError:  # a schema nothing can hold a weak reference to is simply not cached
        return domain_tokens(schema, table, name, rules)
    key = (table, name)
    if key not in per_schema:
        per_schema[key] = domain_tokens(schema, table, name, rules)
    return per_schema[key]


def value_tokens(value: Any, tokens: Iterable[str], rules: Any = None) -> frozenset:
    """The domain tokens one value carries, whether it is one name or a sentence made of them."""
    text = canon(value, rules)
    return frozenset(token for token in tokens if _carries(text, token))


def _carries(text: str, token: str) -> bool:
    """Whether the token stands in the text as a word of its own, not inside a longer one."""
    start = text.find(token)
    while start >= 0:
        before, after = text[start - 1:start], text[start + len(token):start + len(token) + 1]
        if not WORD_EDGE.match(before or " ") and not WORD_EDGE.match(after or " "):
            return True
        start = text.find(token, start + 1)
    return False


def _shown_tokens(tokens: Iterable[str]) -> str:
    return "[" + ", ".join(sorted(tokens)) + "]"


def _shown(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."


def _present(row: dict, name: str, rules: Any = None) -> bool:
    """Whether this side holds a value for the column at all, empty being no value (D217)."""
    return name in row and canon(row.get(name), rules) not in EMPTY_CANON


def _tally(tally: Optional[dict], *names: str) -> None:
    """Count one semantic outcome, where the caller asked to be told (D219)."""
    if tally is None:
        return
    for name in names:
        tally[name] = int(tally.get(name) or 0) + 1


def compare_columns(schema: EntitySchema, table: Optional[str], expected: dict, got: dict,
                    rules: Any = None, judge: Any = None, equivalence: Any = None,
                    route: str = "", tally: Optional[dict] = None) -> tuple[bool, list[str]]:
    """Two readings of one row compared column by column under the schema's classes (D73, D84, D187).

    Every key either side carries is asked for its class. An exempt column is equal whatever the two
    hold and is noted, never failed; a semantic one goes through `canon.compare` and comes back with
    one of three answers, equal, different or unresolved, of which only the first is agreement
    (D219); a hard one is compared by canonical string and names the leaf it parted at. A column the
    schema does not class for this table takes the rules' default, which is what `canon.compare`
    would have given it.

    A semantic column that is not equal fails, and so does one nobody settled. Reporting a semantic
    difference on a note the ruling forgives made "reported" mean "never failed": there was then no
    path anywhere in the harness by which a semantic column could fail a replay, judged or not, and
    a column class became a way of not being checked rather than a way of being checked differently.
    An unresolved pair fails under its own name, because it is not evidence that the two sides
    agree and it is not evidence that they part; it is the absence of an answer, and a reader has
    to be able to see how much of a build's agreement rests on one.

    `route` names how the two dicts were arrived at, so a failure line says whether the comparison
    read a prose result into columns or walked a returned row.

    Two differences are failures whatever the column's class, because forgiving them would let the
    two sides say different things about the world and still be called the same answer (D217). The
    first is presence: a value one side holds and the other does not is a difference, and only two
    sides the canonical form calls empty are not. The second is the domain tokens two values carry:
    a judge is asked whether two sentences mean the same thing only once both are written out of the
    same set of the world's own names, and where they are not the answer is that they differ, which
    also spares the judge the call.
    """
    prefix = f"{route}: " if route else ""
    hard: list[str] = []
    notes: list[str] = []
    for name in sorted(set(expected) | set(got)):
        holds, theirs_holds = _present(got, name, rules), _present(expected, name, rules)
        if holds != theirs_holds:
            hard.append(f"{prefix}{PRESENCE_NOTE}{name}: ours {_shown(got.get(name))}, "
                        f"recorded {_shown(expected.get(name))}")
            continue
        if not holds:
            # Neither side holds a value, which D217 already settled is not a difference in
            # presence. It is not a difference of any other kind either, so there is nothing here
            # to compare, to forgive or to put to a judge, whatever the column's class.
            continue
        column_class = class_of(schema, table, name, rules)
        if column_class == "exempt":
            if canon(expected.get(name), rules) != canon(got.get(name), rules):
                notes.append(f"{EXEMPT_NOTE}{name}")
            continue
        if column_class == "semantic":
            tokens = tokens_for(schema, table, name, rules)
            ours = value_tokens(got.get(name), tokens, rules)
            theirs = value_tokens(expected.get(name), tokens, rules)
            if ours != theirs:
                _tally(tally, "semantic_compared", "semantic_different")
                hard.append(f"{prefix}{TOKEN_NOTE}{name}: ours {_shown_tokens(ours)}, "
                            f"recorded {_shown_tokens(theirs)}")
                continue
            verdict = compare_column(expected.get(name), got.get(name), "semantic", rules=rules,
                                     judge=judge, table=equivalence, column=name)
            found = resolution_of(verdict)
            _tally(tally, "semantic_compared", f"semantic_{found}")
            if verdict.judge_called:
                _tally(tally, "semantic_judged")
            if found == UNRESOLVED:
                hard.append(f"{prefix}{UNRESOLVED_NOTE}{name}: nobody settled ours "
                            f"{_shown(got.get(name))} against recorded {_shown(expected.get(name))}"
                            + (f" ({verdict.note})" if verdict.note else ""))
            elif not verdict.equal:
                hard.append(f"{prefix}{DIFFERS_NOTE}{name}: ours {_shown(got.get(name))}, "
                            f"recorded {_shown(expected.get(name))}")
            elif verdict.route != "canon":
                # Equal, but only because the judge or the table said so: the note is what lets a
                # reader tell agreement that rests on a judgement from agreement on the values.
                notes.append(f"{SEMANTIC_NOTE}{name}")
            continue
        found_at = first_difference(got.get(name), expected.get(name), rules, name)
        if found_at:
            hard.append(f"{prefix}{found_at}")
    return not hard, hard + notes


def read_as_columns(schema: EntitySchema, tool: str, expected: Any, got: Any, rules: Any = None,
                    readers: Any = None, judge: Any = None, equivalence: Any = None,
                    tally: Optional[dict] = None) -> Optional[tuple[bool, list[str]]]:
    """Two prose results compared by the columns their reader asserts; None when no reader applies.

    Both sides have to be strings the tool's own reader reads to a dict. When either reads to
    nothing, or the tool has no reader at all, this answers None and the caller compares the strings
    as it always did.
    """
    if readers is None or not tool or not isinstance(expected, str) or not isinstance(got, str):
        return None
    table = readers.table_of(tool)
    if table is None:
        return None
    left, right = readers.read(tool, expected), readers.read(tool, got)
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    return compare_columns(schema, table, left, right, rules, judge, equivalence,
                           route=COLUMN_ROUTE, tally=tally)


class ReplayComparer:
    """What a replayed answer is compared by beyond canonical equality of the whole answer (D187).

    `runner/replay.py` scores a replayed call by canonicalizing both answers under one hardcoded
    class, so it consults neither the schema's column classes nor the readers' columns; the Runner
    cannot import this package to reach either. So it is handed one object with one method, the way
    it is handed `canon_rules`, and everything that knows the schema stays here.
    """

    def __init__(self, schema: Optional[EntitySchema] = None, readers: Any = None, rules: Any = None,
                 judge: Any = None, equivalence: Any = None):
        self.schema = schema if schema is not None else EntitySchema()
        self.readers = readers
        self.rules = rules
        self.judge = judge
        self.equivalence = equivalence
        # What the semantic comparisons of this build came to, so a round can say how many pairs
        # were compared, how many a judge was actually asked about, and how many nobody settled
        # (D219). Without these a build cannot tell an environment with no semantic column from one
        # whose every semantic column went unanswered.
        self.counts: dict = {name: 0 for name in SEMANTIC_COUNTS}

    def agrees(self, tool: str, expected: Any, got: Any) -> tuple[bool, list[str]]:
        """Whether the replayed answer agrees with the recording, and what parted, by class."""
        reading = read_as_columns(self.schema, tool, expected, got, self.rules, self.readers,
                                  self.judge, self.equivalence, self.counts)
        if reading is not None:
            return reading
        ok, notes = compare_results(self.schema, expected, got, self.rules, tool=tool,
                                    readers=self.readers, judge=self.judge,
                                    equivalence=self.equivalence, tally=self.counts)
        return ok, notes


# --- the rulings, in the order that localizes a failure ---

def _ruling(stage: str, passed: bool, metrics: dict, failures: Iterable[str] = ()) -> GateResult:
    return GateResult(stage=stage, **{"pass": passed}, metrics=metrics, failures=list(failures)[:5])


def args_text(call: ToolCall) -> str:
    return json.dumps(call.args, sort_keys=True, default=str)


def body_parses_gate(source: str) -> GateResult:
    """1. The generated module is Python."""
    try:
        compile(source, "<generated>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return _ruling("parses", False, {}, [f"line {exc.lineno}: {exc.msg}"])
    return _ruling("parses", True, {"chars": len(source)})


def _argument_answer(call: ToolCall, result: dict) -> bool:
    """A refusal about the arguments is an answer gate 5 matches, not a crash of the module (D67).

    The customer's own logs hold calls their tool rejected for a wrong or missing argument. Replaying
    one raises TypeError where the arguments meet the signature, before any body runs, and TypeError
    is otherwise a crash. Such a call is handed on to gate 5, which matches it against the recorded
    `invalid_arguments` class the same way route.py maps TypeError to it.
    """
    return bool(result.get("binding")) or (call.error is not None and call.error.class_ == "invalid_arguments")


def body_executes_gate(calls: Iterable[ToolCall], results: Optional[list[dict]],
                       error: Optional[str] = None) -> GateResult:
    """2. Every recorded call ran against its own Starting state without crashing the module.

    `error` is the sandbox's own failure (the module did not load, or the run timed out), which
    fails the ruling with that message; otherwise `results` is one dict per call.
    """
    calls = list(calls)
    if error is not None:
        return _ruling("executes_on_s0", False, {"calls": len(calls)}, [error])
    crashes = [f"{c.name}({args_text(c)}) raised {r['error']}: {r['message']}"
               for c, r in zip(calls, results or [], strict=False)
               if not r["ok"] and r["error"] in CRASH_ERRORS and not _argument_answer(c, r)]
    return _ruling("executes_on_s0", not crashes, {"calls": len(calls), "crashes": len(crashes)}, crashes)


def body_deterministic_gate(calls: Iterable[ToolCall], first: Optional[list[dict]], second: Optional[list[dict]],
                            rules: Any = None, error: Optional[str] = None) -> GateResult:
    """3. Two fresh runs of the same calls gave the same answers, under the customer's rules (D39)."""
    calls = list(calls)
    if error is not None:
        return _ruling("deterministic", False, {"calls": len(calls)}, [error])
    differing = [c.name for c, a, b in zip(calls, first or [], second or [], strict=False)
                 if canon(a, rules) != canon(b, rules)]
    return _ruling("deterministic", not differing, {"calls": len(calls), "differing": len(differing)},
                   [f"{name} answered differently on a second run" for name in differing])


def _arg_sets_varying(calls: list[ToolCall], rules: Any = None) -> int:
    """How many argument sets the recording answered in more than one way (D165).

    A tool that gave two answers to one argument set answers by the world and not by its arguments:
    a write earlier in the recording moved what a later read returns. This gate runs every call on
    one world, so a right body can only give one answer there, and counting distinct answers over
    all calls would fail it for being right.
    """
    answers: dict[str, set[str]] = {}
    for call in calls:
        if call.error is None:
            answers.setdefault(content_hash(call.args), set()).add(
                content_hash(canon(parse_result(call.result), rules)))
    return sum(1 for seen in answers.values() if len(seen) > 1)


def body_non_trivial_gate(calls: Iterable[ToolCall], results: Optional[list[dict]], rules: Any = None,
                          error: Optional[str] = None) -> GateResult:
    """4. Different arguments do not all give one constant answer, unless the recorded tool answered them that way.

    The recording is the standard: a hand-off tool that acknowledged 32 argument sets with the same
    line is faithfully constant, and a body that matches it is right. The second retail build failed
    `transfer_to_human_agents` here for doing what the real tool did, 25 of 25 replays agreeing.

    A state-driven tool is passed on to replay_fidelity, which runs the calls in sequence and is the
    only ruling that can judge one (D165): when the recording answered one argument set in more than
    one way, the answer came from the world and this gate, running on a single world, cannot ask for
    two answers. The ruling then carries `state_driven` and how many argument sets varied.
    """
    calls = list(calls)
    if error is not None:
        return _ruling("non_trivial", False, {}, [error])
    metrics = {"arg_sets": len({content_hash(c.args) for c in calls}),
               "distinct_answers": len({content_hash(canon(r, rules)) for r in results or []}),
               "recorded_answers": len({content_hash(canon(parse_result(c.result), rules))
                                        for c in calls if c.error is None})}
    varying = _arg_sets_varying(calls, rules)
    if varying:
        return _ruling("non_trivial", True, dict(metrics, state_driven=True, arg_sets_varying=varying))
    if metrics["arg_sets"] < 2:
        return _ruling("non_trivial", True, dict(metrics, insufficient_evidence=True))
    if metrics["recorded_answers"] < 2:
        return _ruling("non_trivial", True, dict(metrics, recorded_constant=True))
    trivial = metrics["distinct_answers"] < 2
    return _ruling("non_trivial", not trivial, metrics,
                   ["the body answers every call the same way"] if trivial else [])


def classify_exception(result: dict) -> str:
    """The raised exception in the D67 classes, so errors are matched by shape and not by text."""
    message, name = (result.get("message") or "").lower(), result.get("error") or ""
    if "not found" in message or "unknown" in message or name == "KeyError":
        return "not_found_entity"
    if name in ("TypeError", "ValidationError") or "invalid" in message or "must be" in message:
        return "invalid_arguments"
    if "permission" in message or "not allowed" in message or "forbidden" in message:
        return "permission_denied"
    return "business_error" if name == "ValueError" else "unknown"


def keys_of(schema: EntitySchema, values: Iterable[Any]) -> Optional[list[tuple[str, str]]]:
    """The row key of every value in this list, or None when any of them carries none.

    A key is the table and the row id `match_table` reads off a returned row (`row_key`), which is
    what says which row a sighting is of whatever position it came back in.
    """
    found = [match_table(schema, value) for value in values]
    return found if found and all(f is not None for f in found) else None  # type: ignore[return-value]


def _row_pairs(schema: EntitySchema, expected: list, got: list) -> list[tuple[Any, Any]]:
    """Pair two lists of rows by their key when both sides are keyed, else by position (D182, D187).

    A list result comes back in the body's own iteration order, and the recording's order is
    whatever the customer's tool answered; D182 already ruled that list order is not an End-state
    fact, and this is the read-result twin. Order is only ignored where there is something better to
    pair by: every row on both sides carrying a key, and no key standing for two rows, since a key
    twice leaves nothing to say which of the two a row on the other side is.
    """
    left, right = keys_of(schema, expected), keys_of(schema, got)
    if left is not None and right is not None and len(set(right)) == len(right) and set(left) == set(right):
        by_key = dict(zip(right, got, strict=False))
        return [(value, by_key[found]) for value, found in zip(expected, left, strict=False)]
    return list(zip(expected, got, strict=False))


def compare_results(schema: EntitySchema, expected: Any, got: Any, rules: Any = None, tool: str = "",
                    readers: Any = None, judge: Any = None, equivalence: Any = None,
                    tally: Optional[dict] = None) -> tuple[bool, list[str]]:
    """Hard columns must match after canon; exempt and semantic ones are reported, not failed (D73, D84).

    A list of rows and a dict wrapping rows are walked into, so the column classes decide there too.
    Comparing a wrapped result as one canonical string would let an exempt column fail a replay that
    the same row returned on its own passes, which is the opposite of what D73 and D84 ask for.

    A tool whose result is one prose sentence has its columns read out of it by that tool's reader
    (D176) and compared the same way (D187); when the reader reads nothing out of either side the
    strings are compared whole, and the failure line says which of the two routes it took.
    """
    reading = read_as_columns(schema, tool, expected, got, rules, readers, judge, equivalence, tally)
    if reading is not None:
        return reading
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return False, [f"list of {len(expected)} against {len(got)}"]
        ok, notes = True, []
        for one, other in _row_pairs(schema, expected, got):
            one_ok, one_notes = compare_results(schema, one, other, rules, judge=judge,
                                                equivalence=equivalence, tally=tally)
            ok, notes = ok and one_ok, notes + one_notes
        return ok, notes
    found = match_table(schema, expected)
    if found is None and isinstance(expected, dict) and isinstance(got, dict):
        if set(expected) != set(got):
            return False, [f"keys differ: {sorted(set(expected) ^ set(got))}"]
        ok, notes = True, []
        for key in sorted(expected):
            key_ok, key_notes = compare_results(schema, expected[key], got[key], rules, judge=judge,
                                                equivalence=equivalence, tally=tally)
            ok, notes = ok and key_ok, notes + key_notes
        return ok, notes
    if not found or not isinstance(got, dict):
        # The leaf is named, not only the column: `items` sends a reader into two dumps, and
        # `items[1].options.size: ours "large", recorded "small"` is the repair (D154).
        found_at = first_difference(got, expected, rules)
        if found_at and readers is not None and tool and readers.table_of(tool) is not None:
            found_at = f"{STRING_ROUTE}: {found_at}"  # a reader exists and read nothing out of it
        return found_at is None, [found_at] if found_at else []
    return compare_columns(schema, found[0], expected, got, rules, judge, equivalence, tally=tally)


def body_replay_fidelity_gate(calls: Iterable[ToolCall], results: Optional[list[dict]], schema: EntitySchema,
                              label: str = "held_out", threshold: float = 1.0, rules: Any = None,
                              error: Optional[str] = None, readers: Any = None) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart.

    `readers` are the readers of D176, given where a tool's results are prose: the two sentences are
    then compared by the columns they assert rather than as one hard value (D187).

    A half with no calls of its kind is unmeasured, not perfect (D219). A tool whose recordings hold
    no error call used to score 1.0 on error fidelity by definition and clear the bar on that alone,
    so the emptier a tool's evidence the easier its ruling was to pass. An empty pool is neither a
    pass nor a failure: its rate is None, the ruling names which half went unmeasured, and the bar
    is held only against the halves something was actually counted in. A ruling with nothing counted
    at all does not pass, because no evidence is not evidence of fidelity.
    """
    calls = list(calls)
    if error is not None:
        return _ruling("replay_fidelity", False, {"split": label}, [error])
    hits = {"success_calls": 0, "success_matches": 0, "error_calls": 0, "error_matches": 0}
    tally: dict = {name: 0 for name in SEMANTIC_COUNTS}
    failures = []
    for call, result in zip(calls, results or [], strict=False):
        if call.error is not None:
            hits["error_calls"] += 1
            got = classify_exception(result) if not result["ok"] else None
            if got == call.error.class_:
                hits["error_matches"] += 1
            else:
                failures.append(f"{call.name}({args_text(call)}): expected error {call.error.class_}, got {got}")
            continue
        hits["success_calls"] += 1
        if not result["ok"]:
            failures.append(f"{call.name}({args_text(call)}): expected a result, got "
                            f"{result['error']}: {result['message']}")
            continue
        ok, differing = compare_results(schema, parse_result(call.result), result["value"], rules,
                                        tool=call.name, readers=readers, tally=tally)
        if ok:
            hits["success_matches"] += 1
        else:
            failures.append(f"{call.name}({args_text(call)}): hard columns differ: "
                            f"{'; '.join(hard_notes(differing)) or 'value'}")
    success = _share(hits["success_matches"], hits["success_calls"])
    errors = _share(hits["error_matches"], hits["error_calls"])
    unmeasured = [half for half, rate in (("success", success), ("error", errors)) if rate is None]
    measured = [rate for rate in (success, errors) if rate is not None]
    passed = bool(measured) and all(rate >= threshold for rate in measured)
    metrics = dict(hits, split=label, success_fidelity=success, error_fidelity=errors,
                   not_measured=unmeasured,
                   semantic_differences=tally["semantic_different"] + tally["semantic_unresolved"],
                   **tally)
    # A ruling says which half it could not measure in `not_measured`, and not on the failure list:
    # a failure line is a repair instruction, read one at a time by the caller that shows it, and a
    # half with no calls is nothing anyone can repair. A ruling that measured neither half is a
    # different case: there the emptiness is the finding, and it fails.
    if not measured:
        failures.append("no recorded call of either kind: this ruling measured nothing")
    return _ruling("replay_fidelity", passed, metrics, failures)


def body_refuses_unknown_gate(probes: list[tuple[str, Any, ToolCall]], results: Optional[list[dict]],
                              error: Optional[str] = None) -> GateResult:
    """6. A write given a reference the world does not hold refused it, as the recorded tool would.

    `probes` is what the sandbox built: for each reference argument, its name, the unknown value it
    was probed with and the probing call (`sandbox.reference_args`); `results` is what the body
    answered. Gate 5 holds the body to the refusals the corpus recorded; this is the refusal the
    corpus never recorded, and permissive is the direction that flatters a Candidate, which is why
    it is a gate and not a note: the second retail build's `modify_pending_order_payment` accepted
    any payment_method_id and wrote, where the real tool raises "Payment method not found".
    """
    if not probes:
        return _ruling("refuses_unknown", True, {"reference_args": 0, "insufficient_evidence": True})
    if error is not None:
        return _ruling("refuses_unknown", False, {"reference_args": len(probes)}, [error])
    failures = [f"{probe.name}({args_text(probe)}): accepted {name}={unknown!r}, which the world does not "
                f"hold, and answered {json.dumps(result['value'], default=str)[:80]}; a reference the tool "
                "cannot find has to be refused, not written"
                for (name, unknown, probe), result in zip(probes, results or [], strict=False) if result["ok"]]
    return _ruling("refuses_unknown", not failures,
                   {"reference_args": len(probes), "accepted": len(failures)}, failures)


# --- 7. a body may not memorise the recordings (D162) ---
#
# Build 13: of 150 replay calls that differed from their recording, 126 were one KeyError raised by
# the body of the tool that replaces items on an order, and 22 more were the same failure seen in a
# later read of the same row. The body never looked the new item up in the world's products table;
# it held three item ids from the recorded calls in a dict, so every other id crashed it. The model
# was asked to recompile that tool 19 times in the round and never converged. A stronger model may
# avoid it; a code gate makes it impossible for any model, which is what the harness is for.
#
# The check is static and knows no domain: it reads the literals out of the model's own methods and
# asks three questions of each. Does it have the shape the schema mined for some table's ids; is it
# a row id the Starting state actually holds; is it a value some recorded call passed this tool as
# an argument. A body that answers yes to any of them copied data out of the recordings, and the
# repair is always the same sentence: look it up in the world's tables.

MEMORISED_LESSON = ("the body memorised recorded ids; write the lookup over the world's tables "
                    "instead of holding a value copied from a recorded call")
# Shorter than this is code, not data: "id", "", a one-letter key. Same for the small integers a
# body counts, indexes and compares with; a recorded id is never 0, 1 or 7.
MIN_LITERAL_CHARS = 3
MIN_LITERAL_NUMBER = 10
# An id pattern that accepts an ordinary word describes no shape at all: `mine.id_pattern` falls
# back to a bare character class when a column's values share nothing, and reading that as an id
# shape would refuse every alphanumeric literal a body writes. A pattern that accepts one of the
# probes is not asked rule (a).
#
# The probes are a plain lowercase word and a Capitalised word at every length a literal reaches
# (D167). Three fixed words of four and five letters missed `^.{6}$`, the pattern one build's miner
# gave a table whose id values share nothing: six characters of any kind is no shape, and reading it
# as one refused a six-letter dict key, a six-letter status word and two Capitalised place names as
# memorised ids, leaving five tools assisted for four attempts on that alone. A word of any length,
# either case, is what a shapeless pattern takes. No probe is digits only, because a pattern that
# takes ten digits and nothing else is a real id shape and has to stay one.
SHAPELESS_PROBE_LENGTHS = range(1, 25)
SHAPELESS_PROBES = tuple(
    probe for n in SHAPELESS_PROBE_LENGTHS for probe in ("a" * n, "A" + "a" * (n - 1)))


def _scopes(tree: ast.AST, class_name: str) -> list[ast.AST]:
    """The model's own code in a generated module: the toolkit's methods, `__init__` excepted.

    The data model, the toolkit shim and `DomainDB.load` are code-owned bytes no model wrote, and
    their literals are evidence of nothing (`gates/confinement.py` draws the same line). A source
    that defines no toolkit class is a bare body, which parses as a module on its own and is then
    read whole, so a caller can rule on what a model replied with before it is wrapped.
    """
    methods = [member
               for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name
               for member in node.body
               if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name != "__init__"]
    return list(methods) if methods else [tree]


def _docstring_node(scope: Any) -> Optional[ast.Constant]:
    """The docstring of one scope, which `compile_env` writes and the model does not."""
    body = getattr(scope, "body", None) or []
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        return first.value
    return None


def body_literals(source: str, class_name: str = TOOLS_CLASS) -> list[Any]:
    """Every string and number the model's own code spells out, each once, in a fixed order.

    The order is `ast.walk`'s, which is level by level and so reads a dict display's keys before its
    values; what matters is that it is the same order every time, since it is the order the failures
    come out in. `ast.walk` reaches a dict display's keys the same way it reaches its values, so
    `{"1008292230": ...}` is read as the literal it is; a memorising body writes its table that way
    more often than any other. Booleans and None are code by construction, docstrings are
    code-owned, and the two size floors keep loop counters and one-word keys out.
    """
    found: dict[tuple[str, Any], Any] = {}
    for scope in _scopes(ast.parse(source), class_name):
        doc = _docstring_node(scope)
        for node in ast.walk(scope):
            if not isinstance(node, ast.Constant) or node is doc:
                continue
            value = node.value
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, str) and len(value) >= MIN_LITERAL_CHARS:
                found.setdefault(("str", value), value)
            elif isinstance(value, (int, float)) and abs(value) >= MIN_LITERAL_NUMBER:
                found.setdefault(("num", value), value)
    return list(found.values())


def _rows_in(node: Any) -> list[dict]:
    """The rows of a table however the world stores it: keyed by id, or in a list."""
    if isinstance(node, dict):
        return [row for row in node.values() if isinstance(row, dict)]
    if isinstance(node, list):
        return [row for row in node if isinstance(row, dict)]
    return []


def _ids_in(schema: EntitySchema, table: str, node: Any) -> set[str]:
    """The ids of one table's rows: the keys the world files them under, and the id column itself."""
    ids: set[str] = set()
    if isinstance(node, dict) and node and all(isinstance(row, dict) for row in node.values()):
        ids |= {key for key in node if isinstance(key, str)}
    name = id_field(schema, table)
    for row in _rows_in(node):
        value = row.get(name) if name else None
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            ids.add(str(value))
    return ids


def starting_state_ids(schema: EntitySchema, db: Any) -> dict[str, set[str]]:
    """Every row id the Starting state holds, by the table that holds it.

    A table whose rows live inside another table's rows is read there too (`schema.homes`): the
    corpus that set this decision keeps its items under the products' variants and its own top-level
    table empty, so reading the top level alone would find none of the ids that were memorised.
    """
    out: dict[str, set[str]] = {}
    world = db if isinstance(db, dict) else {}
    for table in sorted(schema.tables or []):
        ids = _ids_in(schema, table, world.get(table))
        parent, _, column = ((schema.homes or {}).get(table) or "").partition(".")
        if column:
            for row in _rows_in(world.get(parent)):
                ids |= _ids_in(schema, table, row.get(column))
        if ids:
            out[table] = ids
    return out


def recorded_argument_values(calls: Iterable[ToolCall]) -> dict[str, str]:
    """Every scalar the recorded calls passed, as text, with the argument that carried it.

    Values inside a list or a dict argument count: an id is memorised the same way whether the
    recording passed it on its own or in a list of them.
    """
    out: dict[str, str] = {}

    def walk(name: str, value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (str, int, float)):
            out.setdefault(str(value), name)
        elif isinstance(value, list):
            for item in value:
                walk(name, item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(name, item)

    for call in calls:
        for name, value in sorted((call.args or {}).items()):
            walk(str(name), value)
    return out


def recorded_result_values(calls: Iterable[ToolCall], readers: Any = None) -> dict[str, str]:
    """Every scalar a recorded call's own result carried, as text, with the tool that answered it.

    The twin of `recorded_argument_values`, read the same way: every scalar leaf at any depth, so a
    fee inside a payment history and an amount inside a row of a list both count. Where a tool's
    results are prose, its reader (D176) is asked for the columns the sentence asserts and those
    count too, since a value the body memorised out of a sentence is memorised the same way as one
    it took out of a dict.

    A body holding one of these picked a value the recordings answered instead of deriving it: a
    fee chosen by leg count, a timestamp copied off an existing row, a row matched by an amount a
    result once carried. That is exactly the shape D162 exists to refuse, and the gate had only ever
    looked at what the recordings passed in, never at what they answered back (D187).
    """
    out: dict[str, str] = {}

    def walk(tool: str, value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (str, int, float)):
            out.setdefault(str(value), tool)
        elif isinstance(value, list):
            for item in value:
                walk(tool, item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(tool, item)

    for call in calls:
        if call.error is not None:
            continue
        value = parse_result(call.result)
        walk(call.name, value)
        reading = readers.read(call.name, value) if readers is not None else None
        if isinstance(reading, dict):
            walk(call.name, reading)
    return out


def _names_in(node: Any) -> set[str]:
    """Every property name a JSON schema declares, at any depth."""
    out: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            out |= {str(key) for key in properties}
        required = node.get("required")
        if isinstance(required, list):
            out |= {str(key) for key in required if isinstance(key, str)}
        for value in node.values():
            out |= _names_in(value)
    elif isinstance(node, list):
        for value in node:
            out |= _names_in(value)
    return out


def _declared_in(node: Any) -> set[str]:
    """Every value a JSON schema spells out itself: an enum's members, a default, a const."""
    out: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "enum" and isinstance(value, list):
                out |= {str(v) for v in value
                        if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            elif key in ("default", "const") and isinstance(value, (str, int, float)) \
                    and not isinstance(value, bool):
                out.add(str(value))
            else:
                out |= _declared_in(value)
    elif isinstance(node, list):
        for value in node:
            out |= _declared_in(value)
    return out


def structure_names(schema: Optional[EntitySchema], sig: Any = None) -> set[str]:
    """The names of the world and of the signature, which a body says and never memorises.

    A table, a column and an argument name are how a body addresses the world at all: `"item_id"`
    as a dict key is structure, not an id. They are exempt from every rule, unlike the values of
    rule (c), so a body can build the row shape the recording shows without tripping the gate.
    """
    out: set[str] = set()
    if schema is not None:
        out |= {str(table) for table in schema.tables or []}
        out |= {str(column.name) for column in schema.columns or []}
        out |= {key.split(".", 1)[-1] for key in schema.id_patterns or {}}
    if sig is not None:
        out |= _names_in(getattr(sig, "args_schema", {}) or {})
        out |= {str(getattr(field, "name", "")) for field in getattr(sig, "args_fields", []) or []}
        out |= {str(getattr(field, "name", "")) for field in getattr(sig, "result_schema", []) or []}
    return {name for name in out if name}


def signature_values(sig: Any = None) -> set[str]:
    """The values the tool's own signature spells out: an enum's members, a default, a const.

    These are the customer's vocabulary, not their data, and a body is meant to name them. What the
    description lists is checked separately, as text, because a mined signature often carries the
    enum only in the sentence that introduced it.
    """
    return _declared_in(getattr(sig, "args_schema", {}) or {}) if sig is not None else set()


_DIGIT = re.compile(r"\d")


def _is_datum(literal: Any) -> bool:
    """Whether a literal is a value that had to be worked out, rather than a word the tool says.

    Rule (d) asks about the values a tool's own recorded results carried, and almost every result
    carries words as well as data: the status a write sets, the sentence a hand-off acknowledges
    with, the state a device reports. A body has to spell those out, and the first measurement of
    the rule refused a correct write tool for writing the one status word its own recording answers.
    So the rule asks only about what a word never is: a number, or a string carrying a digit, which
    is what an amount, a timestamp, a count and a reference all carry and what a plain word does
    not. It is the same line rule (a) draws with `SHAPELESS_PROBES`, drawn over a value instead of
    over a pattern.
    """
    if isinstance(literal, bool):
        return False
    if isinstance(literal, (int, float)):
        return True
    return isinstance(literal, str) and bool(_DIGIT.search(literal))


def _shaped(pattern: str) -> bool:
    """An id pattern that accepts an ordinary word describes no id shape and is not asked about.

    Any length of word, and Capitalised as well as lowercase (D167): a pattern that takes a word of
    six characters takes a status word and a city name too, so it tells a memorised id from nothing.
    A pattern that only takes digits is a shape and stays one.
    """
    try:
        return not any(re.fullmatch(pattern, probe) for probe in SHAPELESS_PROBES)
    except re.error:
        return False


def _pattern_hit(schema: Optional[EntitySchema], text: str) -> Optional[tuple[str, str]]:
    """The first mined id shape this text has, as `table.column` and the pattern; None for no hit."""
    for key in sorted((schema.id_patterns if schema is not None else None) or {}):
        pattern = schema.id_patterns[key]
        if not _shaped(pattern):
            continue
        try:
            if re.fullmatch(pattern, text):
                return key, pattern
        except re.error:
            continue
    return None


def body_memorised_values_gate(source: str, schema: Optional[EntitySchema] = None, db: Any = None,
                               calls: Iterable[ToolCall] = (), sig: Any = None,
                               class_name: str = TOOLS_CLASS, readers: Any = None) -> GateResult:
    """7. A body may not memorise the recordings: every id and value comes out of the world (D162).

    Four rules over the literals of the model's own methods, in the order that says most about
    where a value came from. (a) The literal has the shape the schema mined for some table's ids.
    (b) The literal is a row id the Starting state holds. (c) The literal is a value a recorded call
    passed this tool, and neither the signature nor the description names it: an enum member the
    description lists is the tool's vocabulary and is allowed, an order id it happened to be called
    with is not. (d) The literal is a value one of this tool's own recorded results carried and is a
    datum rather than a word (`_is_datum`), under the same exclusions (D187): a fee the body picks by
    leg count, a timestamp it spells out, a row it matches by an amount it once saw answered, are
    all values it should have derived.

    The failure names the literal as the code holds it, the rule that caught it and the table, the
    argument or the tool it came from. It is the model's own source, so quoting it back leaks
    nothing, and the held-out split is never named: rules (c) and (d) say an argument's or a tool's
    name, never a call.
    """
    schema = schema if schema is not None else EntitySchema()
    label = f"{getattr(sig, 'name', '')}: " if getattr(sig, "name", "") else ""
    try:
        literals = body_literals(source, class_name)
    except SyntaxError as exc:
        return _ruling(MEMORISED_STAGE, False, {}, [f"{label}does not parse: {exc.msg}"])
    structure = structure_names(schema, sig)
    named = signature_values(sig) | structure
    description = getattr(sig, "description", "") or ""
    calls = list(calls)
    by_table = starting_state_ids(schema, db)
    by_argument = recorded_argument_values(calls)
    by_result = recorded_result_values(calls, readers)
    failures: list[str] = []
    for literal in literals:
        text = literal if isinstance(literal, str) else str(literal)
        if text in structure:
            continue
        found = _pattern_hit(schema, text) if isinstance(literal, str) else None
        if found is not None:
            failures.append(f"{label}the literal {literal!r} has the shape of {found[0]} ids "
                            f"({found[1]}); look the row up in the world's tables instead of "
                            "holding an id a recorded call carried")
            continue
        table = next((name for name in sorted(by_table) if text in by_table[name]), None)
        if table is not None:
            failures.append(f"{label}the literal {literal!r} is a row id of {table} in the Starting "
                            "state; look the row up in the world's tables instead of holding an id "
                            "a recorded call carried")
            continue
        argument = by_argument.get(text)
        if argument is not None and text not in named and text not in description:
            failures.append(f"{label}the literal {literal!r} is a value the recorded calls passed as "
                            f"{argument}, and neither the description nor the signature names it; "
                            "read it from the argument instead of holding a recorded value")
            continue
        answered = by_result.get(text) if _is_datum(literal) else None
        if answered is not None and text not in named and text not in description:
            failures.append(f"{label}the literal {literal!r} is a value the recorded results of "
                            f"{answered} carried, and neither the description nor the signature "
                            "names it; derive it from the world's rows instead of holding a value a "
                            "recorded result answered")
    return _ruling(MEMORISED_STAGE, not failures,
                   {"literals": len(literals), "memorised": len(failures),
                    "tables": len(by_table), "recorded_values": len(by_argument),
                    "recorded_results": len(by_result)}, failures)


# --- 8. a body that reads state answers differently when that state differs (D195) ---
#
# Repo2RLEnv's two-stage check: a test earns its place only when it fails on a stub and passes on
# the real thing. Kullback already asks that of a Verifier (D79) and never asked it of a tool body.
# A body passes the replay ruling when its answers match the recordings under each call's own Task
# overlay, and nothing there asks whether the body read the state it should have read. D187 rule (d)
# catches a literal that equals recorded result text, but a value can be memorised through a dict
# keyed on a row id, a chain of ifs on the argument, or a number computed from the argument alone,
# and none of those is a literal echo. One live round kept a body that wrote a status line as a
# constant over an attempt that derived it from four columns of the row, because the constant got
# further through the gates.
#
# The evidence is already on the table: two recorded calls of one tool, made under two Tasks, whose
# recorded results differ in a column while their arguments are the same or differ only in the row
# id. Whatever produced that difference was the world, so a body that reads the world answers those
# two calls differently in that column, and a body that answers them alike did not read it.

SENSITIVITY_LESSON_HEAD = "the body answered two Tasks alike where their recordings differ in "
SENSITIVITY_LESSON_TAIL = ("; derive those columns from the row the call names in the world the body "
                           "is given, never from the arguments alone and never from a value written "
                           "into the body")
# A pair costs nothing to rule on (both calls have already been run under their own worlds) but
# finding one is quadratic in a tool's recorded calls, so the search stops here. Enough pairs to
# name the columns a body has to read; not enough to make the gate the slow part of a compile.
MAX_SENSITIVITY_PAIRS = 25
# What the whole answer is called when it has no columns to name: a prose result the tool's own
# reader read nothing out of, or a scalar. Not "value", which a customer's row may hold as a column.
WHOLE_ANSWER = "the whole answer"
# How many keys of an object a failure spells out before it says how many more there are: a shape
# is meant to be read at a glance, and a wide row would otherwise fill the line.
SHAPE_KEYS = 8


class SensitivityPair(NamedTuple):
    """Two recorded calls of one tool that ran on two worlds and were answered differently.

    `columns` are the columns their recorded results part in, which is what the body's own two
    answers have to part in as well; `tasks` names the two Tasks, for the failure line; `same_args`
    is true where the two calls carry byte-identical arguments, which is the stronger pair, since
    then nothing but the world can account for the difference. `read_columns` says the tool's reader
    read both recorded results into columns (D176), so the body's own two answers must be read the
    same way: a pair whose recordings were compared as whole prose and answers compared column by
    column are two different questions, and the body would be failed for the mismatch and not for
    what it did.
    """
    first: ToolCall
    second: ToolCall
    columns: tuple[str, ...]
    tasks: tuple[str, str]
    same_args: bool = False
    read_columns: bool = False


def sensitivity_lesson(columns: Iterable[str]) -> str:
    """The one sentence a body refused for answering two worlds alike leaves behind (D195).

    It names the columns, because a lesson that only said "read the world" is what the body writer
    already believes it did; the repair is a column, and the column is the thing the gate knows.
    """
    named = sorted({str(column) for column in columns if column})
    return SENSITIVITY_LESSON_HEAD + ", ".join(named) + SENSITIVITY_LESSON_TAIL if named else ""


def _walk_columns(schema: EntitySchema, table: Optional[str], left: Any, right: Any, rules: Any,
                  path: str, out: dict[str, tuple[Any, Any]]) -> None:
    """Every leaf two values part at, by column path; exempt columns are never a difference."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = match_table(schema, left) or match_table(schema, right)
        here = found[0] if found else table
        for name in sorted(set(left) | set(right)):
            if class_of(schema, here, name, rules) == "exempt":
                continue
            _walk_columns(schema, here, left.get(name), right.get(name), rules,
                          f"{path}.{name}" if path else str(name), out)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            out.setdefault(path or WHOLE_ANSWER, (left, right))
            return
        for one, other in _row_pairs(schema, list(left), list(right)):
            _walk_columns(schema, table, one, other, rules, path, out)
        return
    if canon(left, rules) != canon(right, rules):
        out.setdefault(path or WHOLE_ANSWER, (left, right))


def column_differences(schema: EntitySchema, expected: Any, got: Any, rules: Any = None,
                       tool: str = "", readers: Any = None) -> dict[str, tuple[Any, Any]]:
    """Every column two answers of one tool part in, with both sides (D73, D84, D187).

    `compare_results` answers whether two results agree and names the first leaf that parted; this
    names all of them, which is what a ruling that has to ask "did the body move the same columns
    the recordings moved" needs. An exempt column is equal whatever it holds, so it is never a
    difference and never evidence that a body read anything. A prose result is read into its columns
    by that tool's own reader first (D176), so a tool that answers with one sentence is compared the
    same way here as it is at replay; with no reader the two sentences are one value and part or do
    not part as a whole. The two sides come back with the paths because a caller has to be able to
    ask whether a difference is one the arguments already account for; nothing that reads the values
    may put them in a failure line.
    """
    out: dict[str, tuple[Any, Any]] = {}
    left, right, table = expected, got, None
    if readers is not None and tool and isinstance(expected, str) and isinstance(got, str):
        found = readers.table_of(tool)
        one, other = ((readers.read(tool, expected), readers.read(tool, got))
                      if found is not None else (None, None))
        if isinstance(one, dict) and isinstance(other, dict):
            left, right, table = one, other, found
    _walk_columns(schema, table, left, right, rules, "", out)
    return out


def differing_columns(schema: EntitySchema, expected: Any, got: Any, rules: Any = None,
                      tool: str = "", readers: Any = None) -> list[str]:
    """The names alone of the columns two answers of one tool part in."""
    return sorted(column_differences(schema, expected, got, rules, tool, readers))


def answer_shape(result: Any) -> str:
    """What one sandbox answer looks like with no value of it in the line.

    A failure of this ruling is about two answers being the same, and the reader needs to see what
    kind of thing they are, not what they hold: a customer's row never travels into a gate's
    failures (D162 draws the same line for the literals it quotes, which are the model's own).
    """
    if isinstance(result, dict) and "ok" in result and not result.get("ok"):
        return f"a raised {result.get('error') or 'error'}"
    value = result.get("value") if isinstance(result, dict) and "ok" in result else result
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return f"text of {len(value)} characters"
    if isinstance(value, (list, tuple)):
        return f"a list of {len(value)}"
    if isinstance(value, dict):
        names = sorted(str(key) for key in value)
        more = f" and {len(names) - SHAPE_KEYS} more" if len(names) > SHAPE_KEYS else ""
        return "an object with " + ", ".join(names[:SHAPE_KEYS]) + more
    return "a value"


def _answer_difference(schema: EntitySchema, pair: SensitivityPair, one: Any, other: Any,
                       rules: Any = None, readers: Any = None) -> set[str]:
    """The columns the body's own two answers part in, read the way the pair's recordings were read.

    A raise is an answer here, as it is at replay: where one call raised and the other did not, or
    the two raised different classes, the body did answer them differently, and this ruling has
    nothing to add over what the replay ruling will say about which of them is right.
    """
    ok_one = bool(isinstance(one, dict) and one.get("ok"))
    ok_other = bool(isinstance(other, dict) and other.get("ok"))
    if ok_one != ok_other:
        return set(pair.columns)
    if not ok_one:
        return set(pair.columns) if classify_exception(one or {}) != classify_exception(other or {}) else set()
    return set(differing_columns(schema, one.get("value"), other.get("value"), rules,
                                 pair.first.name, readers if pair.read_columns else None))


def body_sensitivity_gate(pairs: Iterable[SensitivityPair], first: Optional[list[dict]],
                          second: Optional[list[dict]], schema: Optional[EntitySchema] = None,
                          rules: Any = None, readers: Any = None,
                          error: Optional[str] = None) -> GateResult:
    """8. A body answers two worlds differently wherever their recordings differ (D195).

    `pairs` are the sensitivity pairs the caller found over this tool's recorded calls; `first` and
    `second` are what the body answered the two calls of each pair, each under its own Task's
    overlay. A pair is met when the body's two answers part in at least one of the columns the two
    recordings part in. Answers that are equal, or that part only in columns the recordings agree
    on, mean the body did not read the state that moved: it is memorising, or it is reading a row
    the call did not name.

    A tool with no pair is a tool the corpus never showed twice under two worlds, and there is
    nothing to conclude from that, so the ruling records `no_pairs` and passes; the metric is what
    says how much of a corpus this gate could see at all.

    A pair that parts in the whole answer and no column is counted and not ruled on. It is a prose
    result the tool's own reader read nothing out of (D176), so the check has no column to name, no
    way to ask whether the two worlds hold that column differently, and nothing to put in a lesson;
    what it has is a reader to grow, and `unread_pairs` is where a build reads how much of this gate
    a missing reader costs.
    """
    pairs = list(pairs)
    schema = schema if schema is not None else EntitySchema()
    if error is not None:
        return _ruling(SENSITIVITY_STAGE, False, {"pairs": len(pairs), "failed": 0}, [error])
    answers = dict(zip((id(p) for p in pairs), zip(first or [], second or [], strict=False), strict=False))
    unread = [pair for pair in pairs if tuple(pair.columns) == (WHOLE_ANSWER,)]
    pairs = [pair for pair in pairs if tuple(pair.columns) != (WHOLE_ANSWER,)]
    if not pairs:
        return _ruling(SENSITIVITY_STAGE, True,
                       {"pairs": 0, "failed": 0, "no_pairs": True, "columns": [], "same_args_pairs": 0,
                        "unread_pairs": len(unread)})
    failures, columns = [], set()
    for pair in pairs:
        one, other = answers.get(id(pair), (None, None))
        if _answer_difference(schema, pair, one, other, rules, readers) & set(pair.columns):
            continue
        columns |= set(pair.columns)
        failures.append(
            f"{pair.first.name}({args_text(pair.first)}) and {pair.second.name}({args_text(pair.second)}): "
            f"the recordings of tasks {pair.tasks[0]} and {pair.tasks[1]} differ in "
            f"{', '.join(pair.columns)}, and the body answers the first with {answer_shape(one)} and "
            f"the second with {answer_shape(other)}, which do not differ there; read "
            f"{', '.join(pair.columns)} off the row the call names in the world the body is given")
    return _ruling(SENSITIVITY_STAGE, not failures,
                   {"pairs": len(pairs), "failed": len(failures), "no_pairs": False,
                    "columns": sorted(columns), "unread_pairs": len(unread),
                    "same_args_pairs": sum(1 for pair in pairs if pair.same_args)}, failures)
