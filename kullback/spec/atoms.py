"""The two generated Hard rules a Verifier is written in, whoever writes it (moved from kullback/derive.py).

The Examiner's derivation and the Spec's compiler both build a shape atom and the no-write atom; the
rules live here so the Spec never reads the Examiner, and derive.py re-exports every name it moved.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.gates.verifier_suite import HELPERS_SRC, make_atom
from kullback.runner.records import Atom

# The two ways D190 rewrites what the derivation would otherwise have written, marked on the atom so
# the status row can count them without re-deriving.
SHAPE_ATOM = "shape"
NO_WRITE_ATOM = "no_write"
# A demand on a tool the world records as an action (spec/actions.py, D308): the call happened or not.
ACTION_ATOM = "action"
# A cap on one named tool (D309): its successful calls, not every write of the Run.
TOOL_CAP_ATOM = "tool_cap"


def _helpers_src() -> str:
    """The transcript helpers a compiled rule calls, so it finds them at Verdict time."""
    return HELPERS_SRC


# The world at large is still not searched, which is what keeps the demand real: a value the Run
# never read is rejected however common it is elsewhere, so the mutation check still flips and a
# swap to something nobody produced still fails. Scoping it to the row instead, which is what D190
# wrote, rejects the ordinary copy across rows (a value read off one entity and written onto
# another), and it rejected the very Reference the atom was derived from on 17 Tasks of one build.
#
# `_SOURCES` names which of the four ways may satisfy the rule, and a stored atom carries no more of
# them than its own Reference needed. The derivation builds the same rule with one source at a time
# to find out which one answers, so the count on the status row is read off the rule rather than off
# a second copy of it in Python, and then stores the rung that source sits on (`SHAPE_LADDER`). A
# Task whose Reference the row rule already accepted therefore keeps D190's rule exactly, and the
# widening is spent only where the row rule rejected the Run it was read from.
_SHAPE_SRC = '''
_TOOL = {tool!r}
_FIELD = {field!r}
_ID_FIELD = {id_field!r}
_SOURCES = {sources!r}


def _names(field):
    return [field, field[:-1]] if field.endswith("s") else [field]


def _under(node, field):
    """Every scalar this part of the world holds under one of these column names, canonical.

    An empty column holds nothing: a row that names the column and has nothing in it is a row the
    shape has no value to demand, not a row demanding the word null. A row seeded before the write
    it is about carries every column empty, and reading those as values asked the Run to write the
    emptiness back.
    """
    wanted = _names(field)
    out = []
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for name, value in item.items():
                if name in wanted and not isinstance(value, (dict, list, tuple)):
                    if value is not None and value != "":
                        out.append(str(canon(value)))
                else:
                    stack.append(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                stack.append(value)
    return out


def _row(state, id_field, id_value):
    """The part of the Starting state the write's own id names: a row keyed by it, or one holding it."""
    want = str(canon(id_value))
    stack = [state]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if id_field in item and str(canon(item.get(id_field))) == want:
                return item
            for name, value in item.items():
                if str(canon(name)) == want and isinstance(value, (dict, list)):
                    return value
                stack.append(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                stack.append(value)
    return None


def _results(transcript):
    """Every result the Run received before this write, in order."""
    return [turn.get("result") for turn in transcript or [] if turn.get("role") == "tool"]


def _read_before(transcript, field):
    """Every value the Run was handed under this column, by any call, before this write."""
    out = []
    for result in _results(transcript):
        out.extend(_under(result, field))
    return out


def _stated_before(transcript, value):
    """Did a prose result state this value as a token? The reader's answer is text, not a column."""
    for result in _results(transcript):
        if isinstance(result, str) and _holds(result, value):
            return True
    return False


def check(pre_state, write_call, transcript):
    if write_call.get("name") != _TOOL:
        return True
    args = write_call.get("arguments") or {{}}
    if _FIELD not in args:
        return True
    value = args[_FIELD]
    if value is None or value == "" or value == [] or value == {{}}:
        return False
    parts = value if isinstance(value, (list, tuple)) else [value]
    wanted = [str(canon(part)) for part in parts]
    row = _row(pre_state, _ID_FIELD, args.get(_ID_FIELD)) if _ID_FIELD in args else None
    known = _under(row, _FIELD) if row is not None else []
    if "own_row" in _SOURCES and known and all(part in known for part in wanted):
        return True
    if "prior_result" in _SOURCES:
        read = _read_before(transcript, _FIELD)
        if read and all(part in read for part in wanted):
            return True
    if "result_text" in _SOURCES and _stated_before(transcript, value):
        return True
    # The row the write acts on names no such column, so the shape has nothing of its own to demand
    # and the Run only has to have written something there, which it did.
    return "no_column" in _SOURCES and not known
'''

SHAPE_SOURCES = ("own_row", "no_column", "prior_result", "result_text")

# The claim a Task whose Reference wrote nothing still makes about its End state: on the tables these
# tools touch, the End state is the Starting state. A Run that wrote fails it, which is what the
# wrong and unsolved checks need, and because the rule is asked about every call the Run made it
# answers on a read-only Reference too, so the mutation check can put a rule that never holds in its
# place and watch the Reference stop passing.
_NO_WRITE_SRC = '''
_WRITES = {tools!r}


def check(pre_state, write_call, transcript):
    return (write_call.get("name") or "") not in _WRITES
'''


# An action the Run must take: some successful call of the tool is in the Run. The rule reads the
# Run's own `calls`, so it answers the same on every call the Hard wrapper asks it about.
_ACTION_DONE_SRC = '''
_ACTION = {tool!r}


def check(pre_state, write_call, transcript):
    return any(call.get("name") == _ACTION and not call.get("error") for call in calls)
'''

# An action the Run must not take: no call of the tool, the no-write rule over that tool alone.
_ACTION_NOT_DONE_SRC = '''
_ACTION = {tool!r}


def check(pre_state, write_call, transcript):
    return (write_call.get("name") or "") != _ACTION
'''


# At most this many successful calls of one tool; the Run's other writes are not counted.
_TOOL_CAP_SRC = '''
_TOOL = {tool!r}
_COUNT = {count!r}


def check(pre_state, write_call, transcript):
    return sum(1 for call in calls if call.get("name") == _TOOL and not call.get("error")) <= _COUNT
'''


def _rule_atom(atom_id: str, source: str, tools: Iterable[str], derived_as: str, description: str) -> Atom:
    """One generated rule as a Hard atom, asked about every call the Run made.

    `read_tools` is empty on purpose: the compiled-policy wrapper skips a call to a tool the seed
    Runs only ever read with, and a rule that skipped every call of a read-only Run would answer
    nothing, which the D79 mutation check rightly counts as an atom that cannot be falsified.
    """
    return _atom(atom_id, "hard",
                 {"kind": "hard", "constraint_id": atom_id, "judge": False, "predicate_src": source,
                  "write_tools": sorted(tools), "read_tools": [], "derived_as": derived_as},
                 judge=False, description=description)


def shape_atom(atom_id: str, tool: str, field: str, id_field: str, tools: Iterable[str],
               sources: Iterable[str] = SHAPE_SOURCES, source: Optional[str] = None) -> Atom:
    """The shape a required write value keeps when no user said the value itself.

    `sources` says which of `SHAPE_SOURCES` may satisfy the rule, which is how the derivation asks
    the rule itself which ones its own Reference needs; `source` is the name of the one that
    answered, recorded on the atom as a name and never as the value it stands for. The rule calls
    `_holds` for a prose result, which the Hard wrapper defines beside it
    (`verifier_suite._SPANS_SRC`), the same way it reads `canon` out of the namespace it is run in.
    """
    payload_source = _SHAPE_SRC.format(tool=tool, field=field, id_field=id_field,
                                       sources=tuple(sources))
    atom = _rule_atom(atom_id, payload_source, tools, SHAPE_ATOM,
                      f"{tool} writes under {field} a value this Run read for itself")
    # The column the shape is about, so the swap probe (D286) knows which value of the Reference to
    # replace; the rule itself reads the same names out of its source.
    atom.target.update(tool=tool, field=field, id_field=id_field)
    if source:
        atom.target["shape_source"] = source
    return atom


def no_write_atom(tools: Iterable[str]) -> Atom:
    """The one claim a Task that wrote nothing and stated nothing still makes about its End state."""
    return _rule_atom(NO_WRITE_ATOM, _NO_WRITE_SRC.format(tools=sorted(tools)), tools, NO_WRITE_ATOM,
                      "the End state is the Starting state: the Run wrote nothing")


def tool_cap_atom(atom_id: str, tool: str, count: int) -> Atom:
    """A cap on one tool: the Run makes at most `count` successful calls of it."""
    return _rule_atom(atom_id, _TOOL_CAP_SRC.format(tool=tool, count=count), [tool], TOOL_CAP_ATOM,
                      f"the Run makes at most {count} {tool} calls")


def action_atoms(atom_id: str, kind: str, tool: str) -> list[Atom]:
    """A demand on an action tool: `required` and `forbidden` as a Hard rule over the Run's calls.

    A required or allowed action also carries an allowed whole-tool write atom, so a scorer that still
    counts the tool among its writes (the Runner's own Verdict) never names the call an extra write.
    """
    if kind == "forbidden":
        return [_rule_atom(atom_id, _ACTION_NOT_DONE_SRC.format(tool=tool), [tool], ACTION_ATOM,
                           f"the Run does not call {tool}")]
    allowed = make_atom(f"{atom_id}.allowed" if kind == "required" else atom_id, "allowed",
                        {"kind": "write", "tool": tool}, description=f"the Run may call {tool}")
    if kind != "required":
        return [allowed]
    return [_rule_atom(atom_id, _ACTION_DONE_SRC.format(tool=tool), [tool], ACTION_ATOM,
                       f"the Run calls {tool}"), allowed]


def _atom(atom_id: str, kind: str, payload: dict, **fields: Any) -> Atom:
    """One atom, its Hard predicate wrapped with the transcript helpers (the gates' `make_atom` otherwise)."""
    return make_atom(atom_id, kind, payload, helpers=_helpers_src(), **fields)
