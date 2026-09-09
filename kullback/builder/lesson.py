"""What a stalled tool body is told: every differing leaf, the relation across its failing calls,
the lines no recorded call reaches, and the change of strategy past the stall limit (D211).

A body that fails the replay ruling is recompiled with a hint, and until now the hint was written
per call and one leaf deep: the first leaf two answers part on (`runner/canon.first_difference`),
three masked shapes of a gate's failures, and a sentence saying the tool had stalled. Nothing
looked across a tool's failing calls for what they have in common, and nothing told the writer
which lines of the body it already has no recorded call ever reaches. So a build spent twenty and
thirty recompiles on one tool answering the same one example, and the incumbent stood because a
tie keeps it (D184).

Four things live here, and every one of them is keyed on structure alone: on how the two answers
are shaped, on what the arguments hold and on what the recording did, never on a column name, a
tool name or a corpus. Names read out of the data reach the lesson because the writer needs them;
none of them is written into this module.

1. The full diff (`differing_leaves`): every leaf a recorded answer and the body's answer part on,
   with the path and both sides, capped, with the count of leaves left out.
2. The relation step (`relations_over`, `refusal_predicates`): a small catalogue of relations
   tested against the (arguments, recorded, ours) triples of one tool's failing calls. A relation
   that holds on every failing call and no passing call is stated with its counts; one that holds
   on most is stated with its exceptions named.
3. The witnessed-branch read (`constructs_of`, `unwitnessed`, `missing_refusal`): the lines each
   recorded call actually executed, against the branches, loops and assignments the body holds. A
   construct no call reaches is unsupported by the evidence; a behaviour the recording shows and
   the body has no line for is missing.
4. The stall limit (`stalled`, `blocked_gate`): past a fixed number of recompiles that scored no
   higher, the strategy changes rather than the sentence. The ask becomes a rewrite from the
   recorded calls, the whole failing set is shown instead of three shapes, and a tie where both
   bodies fall at one gate before the fidelity ruling is named as blocked by that gate, so the
   round is spent on the gate rather than on the fidelity number behind it.

Nothing here calls a model, edits `kullback/gates` or `kullback/runner`, or rules on anything: the
witnessed-branch read is an input to a lesson, not a gate.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from kullback.gates.tool_runs import MEMORISED_STAGE, SENSITIVITY_STAGE
from kullback.runner.canon import canonicalize as canon

# --- the constants, in one place -------------------------------------------------

# How many differing leaves one failing call's line carries. Past it the line says how many were
# left out, so a wide answer says it is wide instead of silently showing the first of forty.
LEAF_CAP = 8
# How many recompiles in a row may score no higher before the ask itself changes. Two (D191's
# `STALLED_AFTER`) is when the stall is worth saying; this is when saying it has demonstrably
# bought nothing and the round has to ask a different question.
STALL_LIMIT = 6
# How many failing calls, and how many failure shapes, a stalled round is shown. The point of the
# larger set is that a writer answering three shapes leaves the rest as they were.
FAILING_SET_SHOWN = 12
# How many argument and result numbers one arithmetic relation is searched over, and how many
# elements of a list are compared for a broadcast or an ordering. Both are quadratic searches, and
# a corpus shows its relation in the first handful or does not hold one.
RELATION_OPERANDS = 8
RELATION_ELEMENTS = 24
# How much of one side of a leaf reaches the line. `first_difference` cuts at 60 for the same
# reason: the leaf is what the writer needs, not the whole row behind it.
LEAF_CHARS = 60
# A relation that holds on this fraction of the failing calls or more, but not on all of them, is
# stated with its exceptions. Below it the relation is not the shape of the failure.
RELATION_MOST = 0.6
# How many exception call ids one relation names.
EXCEPTIONS_NAMED = 3
# How many unwitnessed constructs one lesson lists.
CONSTRUCTS_SHOWN = 6
# How many leaves of one failing call the across-calls relation looks at. Wider than the cap the
# lesson prints, because a relation is a statement about the set and a leaf dropped from the line
# is still a leaf the set holds.
RELATION_LEAVES = 32

# The stages that run before the fidelity ruling, in the order `sandbox.run_gates` runs them. A tie
# at one of these is a tie the fidelity number cannot break, because the fidelity ruling never ran.
GATES_BEFORE_FIDELITY = ("parses", "confined", MEMORISED_STAGE, "executes_on_s0",
                         "deterministic", "non_trivial", SENSITIVITY_STAGE)
FIDELITY_STAGE = "replay_fidelity"

RELATION_KINDS = ("broadcast", "shared_value", "refusal_predicate", "arithmetic", "order")
# Which construct a line is named as when two of them start on it, most telling first: the branch
# an evidence set never enters is the thing to remove, and the assignment inside it follows from it.
CONSTRUCT_ORDER = ("branch not taken", "loop not taken", "branch", "loop", "raise", "assignment")


# --- the triples a relation is tested against ------------------------------------


@dataclass(frozen=True)
class Triple:
    """One recorded call as the relation step reads it: what it was given, what came back, what ours did.

    `theirs` and `ours` are the two answers as values; where a side raised instead, its value is
    None and the class it raised is on `their_error` or `our_error`. A relation is asked of the
    triples and never of a record, so a caller that has the three parts from anywhere can use it.
    """

    call_id: str
    args: dict = field(default_factory=dict)
    theirs: Any = None
    ours: Any = None
    their_error: str = ""
    our_error: str = ""
    matched: bool = False


# --- 1. the full diff -------------------------------------------------------------


def _shown(value: Any, limit: int = LEAF_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def differing_leaves(theirs: Any, ours: Any, rules: Any = None,
                     cap: int = LEAF_CAP) -> tuple[list[tuple[str, str, str]], int]:
    """Every leaf two answers part on, with the path and both sides; the second value is how many were dropped.

    `first_difference` walks the same two structures and stops at the first leaf, which is the
    right answer for a ruling that has to name where a body went wrong and the wrong one for a
    writer that has to repair it: a body that copies one element's value onto twelve has one
    defect and twelve differing leaves, and shown one of them it repairs one element. Dicts are
    walked by sorted key and lists by position, and two sides that canonicalize alike are equal
    however they are spelled, so the leaves listed are the ones a fidelity ruling turns on.
    """
    out: list[tuple[str, str, str]] = []
    dropped = 0

    def walk(left: Any, right: Any, path: str) -> None:
        where = path or "value"
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                _add(where, sorted(set(left) - set(right)), sorted(set(right) - set(left)))
                return
            for key in sorted(left):
                walk(left[key], right[key], _join(path, str(key)))
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                _add(where, f"{len(left)} elements", f"{len(right)} elements")
                return
            for index, (one, other) in enumerate(zip(left, right, strict=False)):
                walk(one, other, f"{path}[{index}]")
            return
        if canon(left, rules, path) != canon(right, rules, path):
            _add(where, left, right)

    def _add(where: str, left: Any, right: Any) -> None:
        nonlocal dropped
        if len(out) >= cap:
            dropped += 1
            return
        out.append((where, _shown(left), _shown(right)))

    walk(theirs, ours, "")
    return out, dropped


def full_diff_line(triple: Triple, rules: Any = None, cap: int = LEAF_CAP) -> str:
    """One failing call as a line naming every leaf it parts on, not only the first."""
    if triple.their_error or triple.our_error:
        theirs = f"raised {triple.their_error}" if triple.their_error else "answered"
        ours = f"raised {triple.our_error}" if triple.our_error else "answered"
        return f"- call {triple.call_id}: the recording {theirs}, the body {ours}"
    leaves, dropped = differing_leaves(triple.theirs, triple.ours, rules, cap)
    if not leaves:
        return ""
    parts = "; ".join(f"{where}: recorded {left}, ours {right}" for where, left, right in leaves)
    more = f"; {dropped} more leaves differ" if dropped else ""
    return f"- call {triple.call_id} parts at {len(leaves) + dropped} leaves: {parts}{more}"


# --- 2. the relation step ---------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    """One relation the triples hold, as the lesson states it: its kind, where, and on how many calls."""

    kind: str
    where: str
    detail: str
    holds_on: int = 0
    of: int = 0
    exceptions: tuple[str, ...] = ()

    @property
    def on_all(self) -> bool:
        return self.of > 0 and self.holds_on == self.of

    def sentence(self) -> str:
        if self.on_all:
            return (f"- {self.kind} at {self.where}: {self.detail}. It holds on all {self.of} "
                    f"failing calls and on no call the body already answers.")
        named = ", ".join(self.exceptions) or "some of them"
        left = self.of - self.holds_on - len(self.exceptions)
        more = f" and {left} more" if left > 0 else ""
        return (f"- {self.kind} at {self.where}: {self.detail}. It holds on {self.holds_on} of "
                f"{self.of} failing calls; it does not hold on {named}{more}.")


def _rows(value: Any) -> Optional[list[dict]]:
    """A list of row objects, or None for anything else; the shape every element relation needs."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not all(isinstance(item, dict) for item in value):
        return None
    return list(value)[:RELATION_ELEMENTS]


def _parallel_lists(theirs: Any, ours: Any, path: str = "") -> list[tuple[str, list[dict], list[dict]]]:
    """Every path where both answers hold a list of rows of the same length, at any depth."""
    out: list[tuple[str, list[dict], list[dict]]] = []
    left, right = _rows(theirs), _rows(ours)
    if left is not None and right is not None and len(left) == len(right):
        out.append((path or "value", left, right))
    if isinstance(theirs, dict) and isinstance(ours, dict):
        for key in sorted(set(theirs) & set(ours)):
            out += _parallel_lists(theirs[key], ours[key], _join(path, str(key)))
    return out


def _column(rows: list[dict], name: str, rules: Any, path: str) -> list[str]:
    return [canon(row.get(name), rules, path) for row in rows]


def _broadcast_signatures(triple: Triple, rules: Any = None) -> set[tuple[str, str, str]]:
    """The recording puts one element's value on every element where ours varies (the copying system).

    A recorded system that applies one changed element's value to all of them is a behaviour the
    body must copy, since the recording is the standard. It is told from an invented sharing by
    which side is constant and by the constant matching one end of the other side.
    """
    found: set[tuple[str, str, str]] = set()
    for where, left, right in _parallel_lists(triple.theirs, triple.ours):
        for name in sorted(set(left[0]) & set(right[0])):
            theirs = _column(left, name, rules, where)
            ours = _column(right, name, rules, where)
            if len(set(theirs)) != 1 or len(set(ours)) < 2:
                continue
            end = "first" if theirs[0] == ours[0] else "last" if theirs[0] == ours[-1] else ""
            if end:
                found.add(("broadcast", f"{where}.{name}",
                           f"the recording carries the {end} element's value on every element, "
                           f"while the body works one out per element"))
    return found


def _shared_value_signatures(triple: Triple, rules: Any = None) -> set[tuple[str, str, str]]:
    """One side holds a single value across positions the other distinguishes.

    The direction matters and is stated: ours flat against a recording that varies is a rule the
    body invented, and a recording flat against a body that varies is a rule the body has not
    copied. A pair that `broadcast` already explains is left to it, since naming which end the
    recording took is the more useful sentence.
    """
    explained = {where for _, where, _ in _broadcast_signatures(triple, rules)}
    found: set[tuple[str, str, str]] = set()
    for where, left, right in _parallel_lists(triple.theirs, triple.ours):
        for name in sorted(set(left[0]) & set(right[0])):
            theirs = _column(left, name, rules, where)
            ours = _column(right, name, rules, where)
            if f"{where}.{name}" in explained:
                continue
            if len(set(ours)) == 1 and len(set(theirs)) > 1:
                found.add(("shared_value", f"{where}.{name}",
                           "the body puts one value on every element where the recording gives "
                           "each element its own"))
            elif len(set(theirs)) == 1 and len(set(ours)) > 1:
                found.add(("shared_value", f"{where}.{name}",
                           "the recording puts one value on every element where the body gives "
                           "each element its own"))
    return found


def _order_signatures(triple: Triple, rules: Any = None) -> set[tuple[str, str, str]]:
    """The two answers hold the same elements in a different order, and the recorded order is a sort."""
    found: set[tuple[str, str, str]] = set()
    for where, left, right in _parallel_lists(triple.theirs, triple.ours):
        theirs = [canon(row, rules, where) for row in left]
        ours = [canon(row, rules, where) for row in right]
        if sorted(theirs) != sorted(ours) or theirs == ours:
            continue
        for name in sorted(set(left[0]) & set(right[0])):
            keys = _column(left, name, rules, where)
            if len(set(keys)) < 2:
                continue
            if keys == sorted(keys):
                found.add(("order", where, f"the recorded list is in ascending order of {name} "
                                           f"and the body's is not"))
            elif keys == sorted(keys, reverse=True):
                found.add(("order", where, f"the recorded list is in descending order of {name} "
                                           f"and the body's is not"))
    return found


def _numbers(value: Any, path: str = "", out: Optional[list] = None) -> list[tuple[str, str, float]]:
    """Every number a value carries, as (the path it sits at, what to call it, the number).

    The length of every list is a number too, and it sits at the list's own path: a recorded list
    of three where the body answers two is a difference at that path, and how many elements the
    recording carries is exactly the kind of thing an argument determines.
    """
    out = [] if out is None else out
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append((path or "value", path or "value", float(value)))
    elif isinstance(value, dict):
        for key in sorted(value):
            _numbers(value[key], _join(path, str(key)), out)
    elif isinstance(value, (list, tuple)):
        where = path or "value"
        out.append((where, f"the number of elements of {where}", float(len(value))))
        for index, item in enumerate(value):
            _numbers(item, f"{path}[{index}]", out)
    return out


def _arithmetic_signatures(triple: Triple, rules: Any = None) -> set[tuple[str, str, str]]:
    """A numeric leaf of the recording is a sum, difference, product or count over the call's own numbers.

    The operands are the numbers the arguments carry and the lengths of the lists in them, which is
    what a body has in front of it; the leaf tested is one the two answers part on, since a leaf
    they agree on needs no rule.
    """
    if triple.their_error or triple.our_error:
        return set()
    operands = [(name, value) for _, name, value in _numbers(triple.args)][:RELATION_OPERANDS]
    if not operands:
        return set()
    leaves, _ = differing_leaves(triple.theirs, triple.ours, rules, cap=RELATION_OPERANDS)
    wanted = {where for where, _, _ in leaves}
    found: set[tuple[str, str, str]] = set()
    for where, label, value in _numbers(triple.theirs):
        if where not in wanted:
            continue
        for name, number in operands:
            if value == number:
                found.add(("arithmetic", where, f"{label} is {name}"))
        for i, (one_name, one) in enumerate(operands):
            for other_name, other in operands[i + 1:]:
                if value == one + other:
                    found.add(("arithmetic", where, f"{label} is {one_name} plus {other_name}"))
                if value == one - other:
                    found.add(("arithmetic", where, f"{label} is {one_name} less {other_name}"))
                if value == other - one:
                    found.add(("arithmetic", where, f"{label} is {other_name} less {one_name}"))
                if value == one * other:
                    found.add(("arithmetic", where, f"{label} is {one_name} times {other_name}"))
    return found


_PER_CALL = (_broadcast_signatures, _shared_value_signatures, _order_signatures, _arithmetic_signatures)


def _across_calls(failing: list[Triple], rules: Any = None) -> list[Relation]:
    """One leaf where one side answers every failing call alike and the other tells them apart.

    The within-answer relations above ask about the elements of one answer; this asks the same
    question of the failing set itself, which is where a whole class of stalled body lives: the
    recording answers a run of calls with one thing and the body works out a different thing for
    each, or the body answers a run of calls with one thing the recording gives each its own. Both
    are a rule stated once, and both are invisible to a hint written per call, which is what every
    round of these tools had.
    """
    seen: dict[str, list[tuple[str, str, str]]] = {}
    for triple in failing:
        if triple.their_error or triple.our_error:
            continue
        leaves, _ = differing_leaves(triple.theirs, triple.ours, rules, cap=RELATION_LEAVES)
        for where, theirs, ours in leaves:
            seen.setdefault(where, []).append((triple.call_id, theirs, ours))
    out: list[Relation] = []
    for where in sorted(seen):
        rows = seen[where]
        if len(rows) < 2 or len(rows) < RELATION_MOST * len(failing):
            continue
        theirs = {value for _, value, _ in rows}
        ours = {value for _, _, value in rows}
        missing = [t.call_id for t in failing if t.call_id not in {call for call, _, _ in rows}]
        if len(theirs) == 1 and len(ours) > 1:
            detail = ("the recording answers every one of these calls with one value here while the "
                      "body works out a different one for each")
        elif len(ours) == 1 and len(theirs) > 1:
            detail = ("the body answers every one of these calls with one value here while the "
                      "recording gives each of them its own")
        else:
            continue
        out.append(Relation("shared_value", where, detail, len(rows), len(failing),
                            tuple(missing[:EXCEPTIONS_NAMED])))
    return out


def _argument_leaves(args: Any, path: str = "", out: Optional[dict] = None) -> dict[str, Any]:
    """Every scalar an argument set carries, under the leaf name it sits at."""
    out = {} if out is None else out
    if isinstance(args, dict):
        for key in sorted(args):
            _argument_leaves(args[key], str(key), out)
    elif isinstance(args, (list, tuple)):
        for item in args:
            _argument_leaves(item, path, out)
    elif path:
        out.setdefault(path, args)
    return out


def _value_class(value: Any) -> str:
    """What class of value this is, with none of the value in it: a shape a predicate can be stated over."""
    if value is None:
        return "absent"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "an empty string" if not value.strip() else "a string"
    if isinstance(value, (list, tuple)):
        return "an empty list" if not value else "a list"
    if isinstance(value, dict):
        return "an object"
    return "a value"


def refusal_predicates(triples: Iterable[Triple]) -> list[Relation]:
    """What the calls the recording refused share that the calls it accepted do not.

    Two predicates, both over the argument leaves alone: an argument the refused calls all carry
    and no accepted call does (or the reverse), and an argument whose class of value on every
    refused call is one no accepted call ever passes. A tool the recording never refused, or never
    accepted, has no predicate to state, and the step says nothing rather than guessing one.
    """
    triples = list(triples)
    refused = [t for t in triples if t.their_error]
    accepted = [t for t in triples if not t.their_error]
    if not refused or not accepted:
        return []
    refused_args = [_argument_leaves(t.args) for t in refused]
    accepted_args = [_argument_leaves(t.args) for t in accepted]
    names = sorted({name for row in refused_args + accepted_args for name in row})
    out: list[Relation] = []
    for name in names:
        in_refused = all(name in row for row in refused_args)
        in_accepted = any(name in row for row in accepted_args)
        if in_refused and not in_accepted:
            out.append(Relation("refusal_predicate", name,
                                f"the recording refused every call that carries {name} and no call "
                                f"it accepted carries it",
                                len(refused), len(refused)))
            continue
        if not any(name in row for row in refused_args) and all(name in row for row in accepted_args):
            out.append(Relation("refusal_predicate", name,
                                f"the recording accepted no call without {name}",
                                len(refused), len(refused)))
            continue
        refused_classes = {_value_class(row.get(name)) for row in refused_args}
        accepted_classes = {_value_class(row.get(name)) for row in accepted_args}
        if len(refused_classes) == 1 and not (refused_classes & accepted_classes):
            only = next(iter(refused_classes))
            out.append(Relation("refusal_predicate", name,
                                f"the recording refused every call whose {name} is {only}, and no "
                                f"call it accepted passes one",
                                len(refused), len(refused)))
    return out


def relations_over(failing: Iterable[Triple], passing: Iterable[Triple] = (),
                   rules: Any = None) -> list[Relation]:
    """The relations one tool's failing calls hold, tested against the calls it already answers.

    Each kind is asked of one call at a time and the answers are counted: a relation that every
    failing call holds and no passing call holds is the rule the body is missing, and is stated
    with its counts. One that most of them hold is stated with the calls it does not hold on, so a
    writer can see whether it is the rule with an exception or two rules. A relation a passing call
    also holds is dropped: it is not what tells the failing calls apart.
    """
    failing, passing = list(failing), list(passing)
    if not failing:
        return refusal_predicates(passing)
    holds: dict[tuple[str, str, str], list[str]] = {}
    for triple in failing:
        for detector in _PER_CALL:
            for signature in detector(triple, rules):
                holds.setdefault(signature, []).append(triple.call_id)
    ruled_out: set[tuple[str, str, str]] = set()
    for triple in passing:
        for detector in _PER_CALL:
            ruled_out |= detector(triple, rules)
    out: list[Relation] = []
    for signature, ids in sorted(holds.items()):
        if signature in ruled_out:
            continue
        kind, where, detail = signature
        share = len(ids) / len(failing)
        if share < RELATION_MOST:
            continue
        missing = [t.call_id for t in failing if t.call_id not in set(ids)]
        out.append(Relation(kind, where, detail, len(ids), len(failing),
                            tuple(missing[:EXCEPTIONS_NAMED])))
    named = {(r.kind, r.where) for r in out}
    across = [r for r in _across_calls(failing, rules) if (r.kind, r.where) not in named]
    return out + across + refusal_predicates(failing + passing)


# --- 3. the witnessed-branch read --------------------------------------------------


@dataclass(frozen=True)
class Construct:
    """One branch, loop, assignment or raise in a body, and the line a call has to reach to exercise it."""

    kind: str
    line: int
    text: str


def _text_at(source: str, line: int) -> str:
    lines = source.splitlines()
    return lines[line - 1].strip()[:LEAF_CHARS] if 0 < line <= len(lines) else ""


def constructs_of(source: str, function: str) -> list[Construct]:
    """Every branch, loop, assignment and raise inside one function of a module, with the line to reach.

    The line is the one a call has to run for the construct to have been exercised, which for a
    branch is the first statement of the branch and not the test above it: a test runs whenever the
    body runs, and an `if` whose block no call ever entered is exactly the construct nothing
    supports. A module that does not parse holds no constructs, which is the parses gate's ruling
    and not this step's.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    target = next((node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function),
                  None)
    if target is None:
        return []
    seen: set[tuple[str, int, str]] = set()
    for node in ast.walk(target):
        if isinstance(node, (ast.If, ast.For, ast.While)):
            kind = "branch" if isinstance(node, ast.If) else "loop"
            if node.body:
                seen.add((kind, node.body[0].lineno, _text_at(source, node.lineno)))
            if node.orelse:
                seen.add((f"{kind} not taken", node.orelse[0].lineno, _text_at(source, node.lineno)))
        elif isinstance(node, ast.ExceptHandler) and node.body:
            seen.add(("branch", node.body[0].lineno, _text_at(source, node.lineno)))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            seen.add(("assignment", node.lineno, _text_at(source, node.lineno)))
        elif isinstance(node, ast.Raise):
            seen.add(("raise", node.lineno, _text_at(source, node.lineno)))
    # One construct per line: the first statement of a branch is an assignment as often as not, and
    # a lesson that names the same line twice reads as two things to repair rather than one.
    best: dict[int, tuple[str, int, str]] = {}
    for row in sorted(seen, key=lambda row: (row[1], CONSTRUCT_ORDER.index(row[0]))):
        best.setdefault(row[1], row)
    return [Construct(*best[line]) for line in sorted(best)]


def unwitnessed(constructs: Iterable[Construct], executed: Iterable[int]) -> list[Construct]:
    """The constructs no recorded call ever reached, in the order they appear in the body."""
    reached = set(int(line) for line in executed)
    return [c for c in constructs if c.line not in reached]


def has_raise(source: str, function: str) -> bool:
    """Whether one function of a module holds a raise at all."""
    return any(c.kind == "raise" for c in constructs_of(source, function))


def missing_refusal(triples: Iterable[Triple], source: str = "", function: str = "") -> str:
    """The refusal the recording shows and the body has no line for; "" when there is none.

    A required behaviour with no supporting line is a different repair from a branch nothing
    reaches: the writer is not being asked to delete something, it is being asked to add the one
    thing the evidence has been showing it all along.
    """
    triples = list(triples)
    refused = [t for t in triples if t.their_error]
    if not refused:
        return ""
    answered = [t for t in refused if not t.our_error]
    if not answered:
        return ""
    classes = sorted({t.their_error for t in refused})
    note = (" and the body holds no raise at all" if source and function
            and not has_raise(source, function) else "")
    return (f"the recording refused {len(refused)} of these calls ({', '.join(classes)}) and the "
            f"body answered {len(answered)} of them instead{note}; a refusal the recording shows "
            f"on that many calls is behaviour the body has to reproduce")


# --- 4. the stall limit --------------------------------------------------------------


def stalled(unbeaten: int, limit: int = STALL_LIMIT) -> bool:
    """True once this many recompiles in a row have scored no higher than the body already there."""
    return int(unbeaten or 0) >= int(limit)


def _first_failing(gates: Iterable[Any]) -> str:
    for gate in gates or ():
        stage = gate.get("stage") if isinstance(gate, dict) else getattr(gate, "stage", "")
        passed = gate.get("pass", gate.get("passed")) if isinstance(gate, dict) else getattr(gate, "passed", True)
        if not passed:
            return str(stage or "")
    return ""


def blocked_gate(kept: Iterable[Any], attempt: Iterable[Any]) -> str:
    """The gate before the fidelity ruling that both bodies fall at; "" when they do not.

    A tie the fidelity number cannot break is not a fidelity problem. Both sides scoring the same
    with the fidelity ruling never reached reads as "nothing beat the incumbent", and the round
    that follows writes another hint about the recorded calls when what stands between the tool and
    its calls is one static gate.
    """
    one, other = _first_failing(kept), _first_failing(attempt)
    if not one or one != other or one not in GATES_BEFORE_FIDELITY:
        return ""
    return one


# --- what the lesson says --------------------------------------------------------------


REWRITE_HEAD = (
    "This tool has stalled: {unbeaten} recompiles in a row scored no higher than the body it "
    "already has, so patching that body is not the ask. Write a new body from the recorded calls "
    "below and the relation stated with them, as if the tool had no body at all. The whole failing "
    "set is shown rather than a sample of it.")
BLOCKED_HEAD = (
    "Both the body already there and the last attempt fail the {gate} gate, which runs before the "
    "replay ruling, so no recorded call was ever compared and the two tie at a number neither of "
    "them earned. Repair what that gate refuses first; the fidelity of this tool cannot move until "
    "it passes.")
RELATION_HEAD = "The failing calls of this tool share a relation the body does not hold:"
DIFF_HEAD = "Every leaf the failing calls part on, not only the first:"
UNWITNESSED_HEAD = ("These lines of the body no recorded call ever reached, so no evidence supports "
                    "them; remove them or say which recorded call needs them:")
MISSING_HEAD = "A behaviour the recording shows that the body has no line for:"


@dataclass(frozen=True)
class Diagnosis:
    """What the code-only steps found for one tool, as the lesson and as the round's counts."""

    tool: str = ""
    relations: tuple[Relation, ...] = ()
    diffs: tuple[str, ...] = ()
    unwitnessed: tuple[Construct, ...] = ()
    missing: str = ""
    unbeaten: int = 0
    blocked: str = ""

    @property
    def rewrite(self) -> bool:
        return stalled(self.unbeaten)

    def lesson(self) -> str:
        """The lesson these findings make, or "" when they found nothing worth a line."""
        lines: list[str] = []
        if self.rewrite:
            lines.append(REWRITE_HEAD.format(unbeaten=self.unbeaten))
        if self.blocked:
            lines.append(BLOCKED_HEAD.format(gate=self.blocked))
        if self.relations:
            lines.append(RELATION_HEAD)
            lines += [relation.sentence() for relation in self.relations]
        if self.diffs:
            lines.append(DIFF_HEAD)
            lines += list(self.diffs)
        if self.unwitnessed:
            lines.append(UNWITNESSED_HEAD)
            lines += [f"- line {c.line} ({c.kind}): {c.text}" for c in self.unwitnessed[:CONSTRUCTS_SHOWN]]
            if len(self.unwitnessed) > CONSTRUCTS_SHOWN:
                lines.append(f"- and {len(self.unwitnessed) - CONSTRUCTS_SHOWN} more such lines")
        if self.missing:
            lines += [MISSING_HEAD, f"- {self.missing}"]
        return "\n".join(lines)

    def counts(self) -> dict:
        """The round's counts for this tool: relations by kind, unwitnessed lines, rewrites, blocks."""
        found = {kind: sum(1 for relation in self.relations if relation.kind == kind)
                 for kind in RELATION_KINDS}
        return {"relations_found": found,
                "unwitnessed_lines": len(self.unwitnessed),
                "rewrites_forced": 1 if self.rewrite else 0,
                "blocked_by_gate": 1 if self.blocked else 0}


def diagnose(tool: str, triples: Iterable[Triple], source: str = "", function: str = "",
             executed: Iterable[int] = (), unbeaten: int = 0, blocked: str = "",
             rules: Any = None, shown: int = FAILING_SET_SHOWN) -> Diagnosis:
    """The whole code-only read of one stalled tool, from its triples and the lines its calls ran.

    `shown` is how many failing calls the full diff carries; a stalled tool is given the whole
    failing set, which is what the caller passes when the stall limit is reached.
    """
    triples = list(triples)
    failing = [t for t in triples if not t.matched]
    passing = [t for t in triples if t.matched]
    diffs = [line for line in (full_diff_line(t, rules) for t in failing[:shown]) if line]
    constructs = constructs_of(source, function) if source and function else []
    dead = unwitnessed(constructs, executed) if constructs else []
    return Diagnosis(tool=tool, relations=tuple(relations_over(failing, passing, rules)),
                     diffs=tuple(diffs), unwitnessed=tuple(dead),
                     missing=missing_refusal(triples, source, function),
                     unbeaten=int(unbeaten or 0), blocked=blocked)


def merge_counts(rows: Iterable[dict]) -> dict:
    """The per-tool counts added up, which is what one round records (D211)."""
    total = {"relations_found": {kind: 0 for kind in RELATION_KINDS},
             "unwitnessed_lines": 0, "rewrites_forced": 0, "blocked_by_gate": 0}
    for row in rows:
        for kind, count in (row.get("relations_found") or {}).items():
            total["relations_found"][kind] = total["relations_found"].get(kind, 0) + int(count)
        for name in ("unwitnessed_lines", "rewrites_forced", "blocked_by_gate"):
            total[name] += int(row.get(name) or 0)
    return total
