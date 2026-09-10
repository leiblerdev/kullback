"""Derive one Task's Verifier from its Reference and k re-runs read off disk (D42, D43, D91).

Nothing here executes a Run (D91): re-runs arrive as `Run` JSONL paths the Runner already wrote.
The vocabulary the atoms are written in (reading Runs, write effects, provenance, questions and
communicate facts, the predicate templates) and the nine D79 checks that rule on the result live
in `kullback.gates.verifier_suite`; this module is the derivation, moved here from
`builder/verifier.py` in phase 5 because the Examiner is the one writer of Verifiers (D123). What
it adds to the gates' `make_atom` is the transcript helpers a compiled Hard rule may call, which
the suite holds as `HELPERS_SRC` so the policy compiler and this module read one text.

D43 sets an atom's kind by agreement across the good Runs, and for a fact the answer states that is
not the whole rule: agreement says every good Run mentioned the number, not that the Task turned on
it. A stated fact is required only when the request asked the agent about it, and reported without
rejecting anything when it did not.

D190 adds the other half of that rule: what a demand may be written in. A value only the world knows
is not a value the Candidate can be asked to reproduce, so a required write value no user of this
Task ever said is written as a shape over its own column instead of as the literal, and a fact is
demanded on the strength of its own number only when a user said the number and not because the
mined Intent repeats it. A Task whose Reference wrote nothing and told the user nothing still
carries one claim about its End state, so that something in its Verifier can be made to say
something false.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.gates.verifier_suite import (
    HELPERS_SRC,
    SUCCESS_TERMINATIONS,
    _key,
    _texts,
    _token_in,
    _tokens,
    _user_text,
    as_run,
    atom_payload,
    canon_fn,
    classify_provenance,
    communicate_values,
    hard_holds,
    make_atom,
    ptr,
    question_keys,
    resolve_write_tools,
    run_calls,
    text_of,
    write_effects,
)
from kullback.runner.records import Atom, Constraint, Run, Task, Verifier

# A write value is required only when it is agreed across the good Runs *and* the customer or their
# world is where it came from; a value the Candidate invented is never required of it.
REQUIRED_PROVENANCE = ("user_stated", "system_derived")
# The payload kind of a communicate atom the request never asked for. It carries the same value and
# the same predicate as a required one, so the Verdict and the report can still say whether the Run
# stated the fact, and it is not the kind the scorers read as a demand, so it rejects nothing.
REPORTED_COMMUNICATE = "communicate_reported"
# The two ways D190 rewrites what the derivation would otherwise have written, marked on the atom so
# the status row can count them without re-deriving.
SHAPE_ATOM = "shape"
NO_WRITE_ATOM = "no_write"
# The values the leak check never calls a secret whoever said them: a single character says nothing,
# and a JSON literal is not a constant of the customer's world.
NEVER_SECRET = ("true", "false", "null")
_WORDS = re.compile(r"[A-Za-z0-9]+")


def load_runs(paths: Iterable[Any]) -> list[Run]:
    return [as_run(p) for p in paths]


def successful(run: Run, successful_run_ids: Optional[Iterable[str]]) -> bool:
    """Did this re-run finish well enough to derive from? The caller's list wins, else the reason.

    Derivation is the only reader, so this lives here rather than in the suite; `SUCCESS_TERMINATIONS`
    stays there because build.py and gates/stages.py rule on it too.
    """
    if successful_run_ids is not None:
        return run.run_id in set(successful_run_ids)
    return (run.termination_reason or "") in SUCCESS_TERMINATIONS


def _helpers_src() -> str:
    """The transcript helpers a compiled rule calls, so it finds them at Verdict time."""
    return HELPERS_SRC


# --- what a user said (D190) ------------------------------------------------

def spoken_text(runs: Iterable[Run]) -> str:
    """Every user turn of every Run the derivation saw, as one text.

    The leak check reads the users of all the Task's seed recordings and not the one that happens to
    be the Reference, because the Intent is mined from all of them; a derivation that read fewer
    would call a value unspoken that the check calls spoken, and the two must agree.
    """
    return " ".join(_user_text(event) for run in runs
                    for event in run.events if event.type == "user_turn")


def user_said(spoken: str, value: Any) -> bool:
    """Did a user of this Task say this value, by the leak check's own rule?

    `_leak_gate` calls a value a secret when any part of it is longer than one character, is not a
    JSON literal, and is not a whole token of the users' text. Said is the negation of that, part by
    part, so an atom this returns True for is one the check never reads as a constant of the
    Verifier's own, and an atom it returns False for is one the check would.
    """
    if value is None:
        return False
    parts = [text for text in _texts(value) if text]
    if not parts:
        return False
    return all(len(text) <= 1 or text in NEVER_SECRET or _token_in(spoken, text) for text in parts)


# --- shapes, for a value no user said (D190, scoped by D206) ----------------

# A required value the world produced is checked as a shape over its own column, and D206 says which
# world the shape is checked against: the Run's own, not the row's. A system_derived value is by
# definition one the Candidate read for itself, so the value written has to be one the Run received
# under that column in a result before the write (any call, any row), or one the write's own row of
# the Starting state holds under it, and where a prose result is what the Run got, one that result
# states as a token (D203 readers). The literal never enters the atom, so a Verifier holding this
# keeps no constant the Intent could give away, and the rule reads the Run and the Starting state at
# Verdict time instead. A plural column is looked up under its singular too, because a list argument
# (`..._ids`) is usually a list of what one row holds one of (`..._id`).
#
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

# The four ways a shape may be satisfied, in the order the derivation asks about them, and the rungs
# the stored rule is allowed to stand on. The first rung is D190's rule unchanged, the row and its
# own fallback, which are one demand and never separated: a rule that took the fallback alone would
# reject the Candidate whose row does hold the column, which is stricter than D190 and not what
# either decision asks. Widening past it is what D206 adds, and the derivation climbs no further
# than the Reference it read makes it: a Task the row rule already accepted keeps the row rule, so
# the wrong Run and the mutation still fail on it exactly as they did.
SHAPE_SOURCES = ("own_row", "no_column", "prior_result", "result_text")
_ROW_SOURCES = ("own_row", "no_column")
SHAPE_LADDER = (_ROW_SOURCES, _ROW_SOURCES + ("prior_result",),
                _ROW_SOURCES + ("prior_result", "result_text"))
# Which rung a source first appears on, so one probe of the rule answers both what to store and what
# to call it.
_RUNG_OF = {source: min(i for i, rung in enumerate(SHAPE_LADDER) if source in rung)
            for source in SHAPE_SOURCES}

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
    if source:
        atom.target["shape_source"] = source
    return atom


def shape_for(atom_id: str, tool: str, field: str, id_field: str, tools: Iterable[str],
              run: Run, fn: Any) -> tuple[Optional[Atom], Optional[str]]:
    """The shape atom for this write value, and which source satisfies it on the Run it came from.

    The rule is asked one source at a time, so what the status row counts is read off the rule
    rather than off a second copy of it written in Python, and the rung the answering source sits on
    is what gets stored: no wider than the Reference needs. A Task the row rule already accepted is
    stored with the row rule, which is why widening it here takes nothing off the wrong Run or the
    mutation on any Task D190 was already right about.

    None for the atom is the rule rejecting the very Reference it was derived from, which is a
    derivation defect and not a demand: it is counted as `shape_self_reject` and nothing is stored.
    "mixed" is a rung holding over the Run while no single source holds over all of its calls, which
    is one Run writing the same column twice from two different sources.
    """
    tools = sorted(tools)
    for source in SHAPE_SOURCES:
        one = shape_atom(atom_id, tool, field, id_field, tools, sources=(source,))
        if hard_holds(one, run, set(tools), fn) is not False:
            rung = SHAPE_LADDER[_RUNG_OF[source]]
            return shape_atom(atom_id, tool, field, id_field, tools, sources=rung, source=source), source
    for rung in SHAPE_LADDER:
        whole = shape_atom(atom_id, tool, field, id_field, tools, sources=rung, source="mixed")
        if hard_holds(whole, run, set(tools), fn) is not False:
            return whole, "mixed"
    return None, None


def no_write_atom(tools: Iterable[str]) -> Atom:
    """The one claim a Task that wrote nothing and stated nothing still makes about its End state."""
    return _rule_atom(NO_WRITE_ATOM, _NO_WRITE_SRC.format(tools=sorted(tools)), tools, NO_WRITE_ATOM,
                      "the End state is the Starting state: the Run wrote nothing")


def shape_sources(verifier: Verifier) -> dict[str, int]:
    """How many of this Task's shape atoms each source satisfied at derive time (D206), by name.

    A name, never a value: what the row says is that the Run read the value off an earlier result
    rather than off the row it wrote, not what the value was.
    """
    named = [atom_payload(atom).get("shape_source") for atom in verifier.atoms
             if atom_payload(atom).get("derived_as") == SHAPE_ATOM]
    return {source: named.count(source) for source in sorted(set(named)) if source}


def derivation_counts(verifier: Verifier) -> dict[str, Any]:
    """How many atoms D190 rewrote or added on this Task, and what satisfied each shape (D206)."""
    marks = [atom_payload(atom).get("derived_as") for atom in verifier.atoms]
    return {"shape_sources": shape_sources(verifier),
            "atoms_dropped_as_shape_self_reject": sum(
                len(atom_payload(atom).get("shape_self_reject") or ()) for atom in verifier.atoms),
            "atoms_relaxed_to_shape": marks.count(SHAPE_ATOM),
            "atoms_added_for_falsification": marks.count(NO_WRITE_ATOM)}


def _atom(atom_id: str, kind: str, payload: dict, **fields: Any) -> Atom:
    """One atom, its Hard predicate wrapped with the transcript helpers (the gates' `make_atom` otherwise)."""
    return make_atom(atom_id, kind, payload, helpers=_helpers_src(), **fields)


# --- what the request asked for (D43) ---------------------------------------

def request_phrases(intent: Any) -> list[str]:
    """The request in the customer's own words: an Intent record's line and every grounded span, a
    Task's Intent line and name, or a plain string.

    Both records are read by attribute rather than by type so the derivation keeps taking a Task, an
    Intent or a string, and so it never has to import the Builder to know what an Intent is (D123).
    """
    if intent is None:
        return []
    if isinstance(intent, str):
        return [intent]
    out = [str(getattr(intent, "text", "") or ""), str(getattr(intent, "intent", "") or ""),
           str(getattr(intent, "name", "") or "")]
    for span in getattr(intent, "spans", None) or []:
        out += [str(getattr(span, "phrase", "") or ""), str(getattr(span, "text", "") or "")]
    return [phrase for phrase in out if phrase.strip()]


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORDS.findall(text or "")}


def asked_facts(intents: Iterable[Any], runs: Iterable[Run], fn: Any) -> tuple[set[str], set[str]]:
    """What this Task's request asks the answer to state: the facts it names, and its words.

    The facts are keyed the way `communicate_values` keys them, so a value the user spelled one way
    and a tool result another are one key (D39). The words are everything the request and the Runs'
    own user turns are written in, which is how a fact asked for by the name of the field it lives
    under ("the total") is recognised without the user having said the number.

    Only a user names a fact by its own number (D190). The Intent is mined from the recordings (D83),
    so a number in its line may be one the recorded agent found and told the user, and demanding the
    answer state it because the Intent repeats it asks the Candidate for a constant the Intent has
    already handed it. The Intent's words still count for the field-name half: a request that says
    "the total" names the field without naming the number.
    """
    said = [_user_text(event) for run in runs for event in run.events if event.type == "user_turn"]
    texts = said + [phrase for intent in intents for phrase in request_phrases(intent)]
    keys = {_key(fn, token) for text in said for token in _tokens(text)}
    return keys, {word for text in texts for word in _words(text)}


def source_fields(run: Run, span: Any, token: str) -> set[str]:
    """The field names the tool result this fact was read from held it under.

    A fact the user never spelled out is still one the request asked for when the request names the
    field it came from, so the field is read off the very result `communicate_values` found the fact
    in, and never off the whole world.
    """
    idx = getattr(span, "msg_index", None)
    event = next((e for e in run.events if e.idx == idx and e.type == "tool_result"), None)
    if event is None:
        return set()
    found: set[str] = set()
    _walk_fields((event.payload or {}).get("result"), token, None, found)
    return found


def _walk_fields(value: Any, token: str, field: Optional[str], found: set[str]) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _walk_fields(item, token, str(name), found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_fields(item, token, field, found)
    elif field and _token_in(text_of(value), token):
        found.add(field)


def communicate_kind(fact: dict, key: str, run: Run, asked: tuple[set[str], set[str]]) -> bool:
    """Is this stated fact one the request asked for, so that the answer has to repeat it?

    D43 makes a fact stated by every good Run an atom, which is right for the fact the user asked
    about and wrong for the ones the recorded agent volunteered beside it: a price, a total, a
    duration nobody requested. On the first two corpora about seven in ten derived answer facts are
    of the second kind, and a held-out Run that solved the Task without repeating the recorded
    agent's number was rejected for it. Asked for means the request or a user turn of this Run names
    the fact itself, or names the field of the tool result it was read from.
    """
    keys, words = asked
    if key in keys:
        return True
    for field in source_fields(run, fact.get("span"), fact["text"]):
        named = _words(field)
        if named and named <= words:
            return True
    return False


# --- derivation ------------------------------------------------------------

def derive_verifier(task: Any, reference_run: Any, rerun_paths: Optional[list[str]] = None, canon: Any = None, *,
                    write_tools: Optional[Iterable[str]] = None,
                    constraints: Optional[Iterable[Constraint]] = None,
                    successful_run_ids: Optional[Iterable[str]] = None,
                    intent: Any = None,
                    writes_elsewhere: bool = False,
                    verifier_version: str = "1") -> Verifier:
    """The atoms for one Task: write-set diff over the Reference and its successful re-runs (D42, D43).

    `writes_elsewhere` is the caller saying that a Run of this Task the D111 rule saw, and that is
    not one of these References, wrote something; it decides whether a cap of 0 is written (below).
    """
    fn = canon_fn(canon)
    reference = as_run(reference_run)
    reruns = load_runs(rerun_paths or [])
    tools = resolve_write_tools([reference] + reruns, write_tools)
    good = [reference] + [r for r in reruns if successful(r, successful_run_ids)]
    good_effects = [write_effects(r, tools, fn) for r in good]
    spoken = spoken_text(good)

    atoms = _write_atoms(good, good_effects, tools, spoken, fn)
    atoms += _question_atoms(good, good_effects, fn)
    facts, request_asked = _communicate_atoms(reference, good, spoken,
                                              asked_facts([intent, task], good, fn), atoms, fn)
    atoms += facts
    # The cap comes after the facts so the rule on a cap of 0 can see what else the request asked for.
    atoms += _cap_atoms(good_effects, writes_elsewhere, request_asked)
    atoms += _constraint_atoms(constraints, tools, [reference] + reruns)

    if _nothing_to_fail(atoms):
        # D190: the Reference wrote nothing and its answer stated no fact read from the world, so
        # every atom here either cannot be failed (a cap of 0) or has no call of a read-only Run to
        # judge. What is still true of this End state is that it is the Starting state, and a Run
        # that wrote is a Run that did something else; the claim is written so that the Verifier
        # says something a Run can contradict.
        atoms.append(no_write_atom(tools))

    return Verifier(task_id=_task_id(task), atoms=atoms, verifier_version=verifier_version,
                    seed_run_ids=[r.run_id for r in good])


def _demand_kind(demanded: bool) -> str:
    """The two kinds the write-set diff writes: what the Candidate has to do, and what it may do."""
    return "required" if demanded else "allowed"


def _task_id(task: Any) -> str:
    """The id of the Task this Verifier is for, whichever of the three shapes the caller passed."""
    if isinstance(task, str):
        return task
    return task.id if isinstance(task, Task) else str(task)


# --- writes (D42, D43) ------------------------------------------------------

def _agreed_and_seen(effects: list[dict]) -> tuple[set[str], set[str]]:
    """The write keys every good Run made, and the keys any of them made."""
    keys = [set(effect) for effect in effects]
    if not keys:
        return set(), set()
    return set.intersection(*keys), set().union(*keys)


def _agreed_fields(key: str, effect: dict, effects: list[dict]) -> set[str]:
    """The columns of this write every good Run wrote the same value under (D39 canonical)."""
    return {field for field, value in effect["values"].items()
            if all(key in e and e[key]["values"].get(field) == value for e in effects)}


def _write_atoms(good: list[Run], good_effects: list[dict], tools: set[str], spoken: str,
                 fn: Any) -> list[Atom]:
    """One atom per write the good Runs made, each followed by the atoms for its columns.

    D43: present in every successful re-run is required, in some is allowed, in none is not an atom.
    A write only a failed re-run made is therefore not a forbidden atom; the write cap and verdict.py's
    extra-write check are what keep a Candidate from writing more than the good Runs did.
    """
    everywhere, somewhere = _agreed_and_seen(good_effects)
    atoms: list[Atom] = []
    for number, key in enumerate(sorted(somewhere)):
        run, effect = _first_with(good, good_effects, key)
        atom_id = f"w{number}"
        base = {"tool": effect["tool"], "entity": effect["entity"], "entity_raw": effect["entity_raw"],
                "id_field": effect["id_field"], "at": effect["idx"]}
        if effect["requestor"]:
            base["requestor"] = effect["requestor"]  # D71: a user-side write says so on the atom
        write_atom = _atom(atom_id, _demand_kind(key in everywhere), dict(base, kind="write"),
                           spans=[ptr(run, effect["idx"])],
                           description=f"{effect['tool']} writes {effect['entity'] or 'an entity'}")
        atoms.append(write_atom)
        values, self_rejected = _value_atoms(atom_id, base, run, effect, tools, spoken, fn,
                                             _agreed_fields(key, effect, good_effects))
        atoms += values
        if self_rejected:
            # The columns whose shape was not stored, on the write they belong to: the count is what
            # the status row reads, and the names are what a person re-derives from (D206).
            write_atom.target["shape_self_reject"] = self_rejected
    return atoms


def _kept_as_shape(demanded: bool, provenance: str, spoken: str, value: Any) -> bool:
    """Is this required value one no user said, so that the column is demanded as a shape (D190)?

    No user of this Task said the value, so it is one the world knows and the Candidate has to read
    for itself. Demanding the literal asks it to reproduce a constant it was never told, and it is
    the constant the leak check finds in the Intent. The column keeps its demand as a shape; the row
    the write acts on is still the row the write atom names, because which row the Task is about is
    the Task and not a value the derivation chose.

    Only a system_derived value is read this way, which is the only kind the leak check reads. D42
    has already said a user_stated value came from a user, and it compares canonically ("$150" and
    150.0 are one value, D39) where the check compares text; taking the check's stricter reading over
    an atom it never looks at would relax a value the customer themself gave.
    """
    return demanded and provenance == "system_derived" and not user_said(spoken, value)


def _value_atoms(atom_id: str, base: dict, run: Run, effect: dict, tools: set[str], spoken: str,
                 fn: Any, agreed: set[str]) -> tuple[list[Atom], list[str]]:
    """The atoms for one write's columns, and the columns whose shape rejected its own Reference.

    D206: a shape is checked against what the Run read, so the rule is asked here whether it accepts
    the Run it was derived from. An atom that rejects its own Reference is a derivation defect, and
    it is dropped and counted rather than stored to fail every Candidate and the Reference alike.
    """
    atoms: list[Atom] = []
    self_rejected: list[str] = []
    for field in sorted(effect["values"]):
        value, canon_key = effect["args"][field], effect["values"][field]
        provenance, span = classify_provenance(run, effect["pos"], value, fn)
        demanded = field in agreed and provenance in REQUIRED_PROVENANCE
        if _kept_as_shape(demanded, provenance, spoken, value):
            shape, _source = shape_for(f"{atom_id}.{field}", effect["tool"], field,
                                       effect["id_field"], tools, run, fn)
            if shape is None:
                self_rejected.append(field)
                continue
            atoms.append(shape)
            continue
        atoms.append(_atom(f"{atom_id}.{field}", _demand_kind(demanded),
                           dict(base, kind="write_value", field=field, value=canon_key, raw=value),
                           provenance=provenance, spans=[span] if span else [],
                           description=f"{effect['tool']} {field} is {text_of(value)}"))
    return atoms, sorted(self_rejected)


# --- questions and facts ----------------------------------------------------

def _question_atoms(good: list[Run], good_effects: list[dict], fn: Any) -> list[Atom]:
    """One atom per thing every good Run asked the user about before it wrote."""
    asked = [question_keys(r, e, fn) for r, e in zip(good, good_effects, strict=False)]
    common = set.intersection(*[set(a) for a in asked]) if asked else set()
    atoms: list[Atom] = []
    for key in sorted(common):
        seen = asked[0][key]
        atoms.append(_atom(f"q.{key}", "question",
                           {"kind": "question", "key": key, "tool": seen.get("tool"),
                            "field": seen.get("field")},
                           spans=[seen["span"]] if seen.get("span") else [],
                           description=f"the agent asks the user about {key.split(':', 1)[-1]}"))
    return atoms


def _stated_facts(said: list[dict], atoms: list[Atom]) -> set[str]:
    """The facts this Task's answer is derived from: the ones every good Run stated, else the Reference's.

    D190: agreement across the good Runs is the rule while there is anything else to fail, and on a
    Task with nothing else it leaves a Verifier no Run can fail at all. What the Reference itself told
    the user is then the evidence of the work, and the facts of it the request asked for are demanded;
    the ones it did not ask for stay reported, as D182 says.
    """
    common = set.intersection(*[set(s) for s in said]) if said else set()
    if common or _demands_something(atoms):
        return common
    return set(said[0]) if said else set()


def _redemand_reported(atoms: list[Atom], reported: list[int], spoken: str) -> None:
    """Demand the facts that were reported, in place.

    Nothing else in this Verifier can be falsified, so the facts stand: on a Task whose request names
    no fact and whose Reference wrote nothing, reporting them all leaves a Verifier an empty Run
    passes, which the D79 suite is right to distrust. What the Reference told the user is then the
    only evidence of the work, and it is demanded again.
    """
    for at in reported:
        payload = dict(atoms[at].target, kind="communicate")
        atoms[at] = _demanded_fact(atoms[at].id, payload,
                                   atoms[at].spans[0] if atoms[at].spans else None,
                                   asked=False, spoken=user_said(spoken, payload.get("text")))


def _communicate_atoms(reference: Run, good: list[Run], spoken: str, request: tuple[set[str], set[str]],
                       atoms: list[Atom], fn: Any) -> tuple[list[Atom], bool]:
    """The atoms for the facts the answer states, and whether this Verifier demands anything else.

    `atoms` is what the derivation has written so far, which is what says whether there is anything
    else a Run can fail; the flag is read before the fallback below, which the cap ignores.
    """
    said = [communicate_values(r, fn) for r in good]
    facts: list[Atom] = []
    reported: list[int] = []
    for number, key in enumerate(sorted(_stated_facts(said, atoms))):
        fact = said[0][key]
        payload = {"kind": "communicate", "value": key, "text": fact["text"]}
        if communicate_kind(fact, key, reference, request):
            facts.append(_demanded_fact(f"c{number}", payload, fact["span"],
                                        spoken=user_said(spoken, fact["text"])))
            continue
        # Same value, same predicate, reported and never a rejection: the request did not ask for
        # this fact, so a Run that solved the Task without repeating it has done the job.
        reported.append(len(facts))
        facts.append(_reported_fact(f"c{number}", payload, fact["span"]))
    request_asked = _demands_something(atoms + facts)
    if reported and not request_asked:
        _redemand_reported(facts, reported, spoken)
    return facts, request_asked


# --- the cap and the compiled constraints -----------------------------------

def _cap_atoms(good_effects: list[dict], writes_elsewhere: bool, request_asked: bool) -> list[Atom]:
    """How many entities the Run may write, which is what the good Runs wrote.

    A cap of 0 is what the Reference did, and on a Task whose other Runs wrote it is also a claim
    the Task's own evidence contradicts. Where the request asked for nothing else (nothing required,
    no question, no fact it named), it is not written: an atom no Run can fail but a writing one is
    not a bar, it is the shape D173 names, an empty Run and the wrong Run both satisfy it, and every
    held-out Run that took the writing path is rejected by it. The D133 route is taken rather than a
    cap widened to the writing group: those Runs are held out as evidence that the Task has more than
    one path, the false-rejection number counts them, and the D79 suite says a Verifier with nothing
    in it that can be falsified is not trusted, which is the honest reading and the one that sends
    the Examiner to re-roll and repair. Widening the cap instead would pass a path the D111 rule had
    just set aside, on the strength of a Run no rule ruled good, and it would still not admit the
    writing Run: with no write atom covering it the Verdict's extra-write check rejects it whatever
    the cap says.
    """
    if not good_effects:
        return []
    cap = max(len(effect) for effect in good_effects)
    if cap or not writes_elsewhere or request_asked:
        return [_atom("entity_count", "required", {"kind": "entity_count", "count": cap},
                      description=f"the Run makes at most {cap} write calls")]
    return []


def _constraint_atoms(constraints: Optional[Iterable[Constraint]], tools: set[str],
                      runs: list[Run]) -> list[Atom]:
    """One atom per constraint the Builder compiled or handed to a judge."""
    reads = {c["name"] for r in runs for c in run_calls(r)} - tools
    atoms: list[Atom] = []
    for rule in constraints or []:
        if not (rule.compiled or rule.judge_atom):
            continue  # a residual constraint is reported, never verdicted (D76)
        atoms.append(_atom(f"hard.{rule.id}", "hard",
                           {"kind": "hard", "constraint_id": rule.id, "judge": rule.judge_atom,
                            "predicate_src": rule.predicate_src, "write_tools": sorted(tools),
                            "read_tools": sorted(reads)},
                           judge=bool(rule.judge_atom), description=rule.text,
                           spans=[rule.span] if rule.span else []))
    return atoms


def _demanded_fact(atom_id: str, payload: dict, span: Any, asked: bool = True, spoken: bool = False) -> Atom:
    """A fact the answer has to state, why it has to, and where the value came from.

    The provenance used to be written as system_derived whatever the fact was, which said of a number
    the user themself gave that the Verifier had it off the world. It is classified now the way a
    written value is (D42): a fact a user of this Task said is user_stated.
    """
    why = ("which the request asked the agent about" if asked
           else "and the Verifier asks nothing else that a Run can fail")
    return _atom(atom_id, "communicate", payload,
                 provenance="user_stated" if spoken else "system_derived",
                 spans=[span] if span else [],
                 description=f"the final answer states {payload['text']}, {why}")


def _reported_fact(atom_id: str, payload: dict, span: Any) -> Atom:
    """A fact the answer may state: same value, same predicate, and no Run is rejected for it.

    It carries no provenance (D190). Provenance is what evidences a demand (D42), and this atom makes
    none: the Verdict evaluates it and the report says whether the Run stated the fact, and no Run is
    ever rejected for it. Writing one said of a value the Verifier does not keep that it keeps it,
    which is what the leak check reads an atom's provenance for.
    """
    atom = _atom(atom_id, "allowed", payload, spans=[span] if span else [],
                 description=f"the final answer may state {payload['text']}; the request did not ask "
                             "the agent about it, so it is reported and rejects no Run")
    return atom.model_copy(update={"target": dict(payload, kind=REPORTED_COMMUNICATE)})


def _demands_something(atoms: Iterable[Atom]) -> bool:
    """Does this Verifier already ask the Run for anything, cap aside? (the kinds a Verdict must hold)."""
    return any(atom.kind in ("required", "question", "communicate") for atom in atoms)


def _nothing_to_fail(atoms: Iterable[Atom]) -> bool:
    """Is every demand here one an empty Run also satisfies? A cap of 0 is the only one there is."""
    for atom in atoms:
        if atom.kind in ("question", "communicate"):
            return False
        if atom.kind == "required" and not (atom_payload(atom).get("kind") == "entity_count"
                                            and not atom_payload(atom).get("count")):
            return False
    return True


def _first_with(runs: list[Run], effects: list[dict], key: str):
    """The first Run that has this write effect; the Reference comes first, so its spans win."""
    for run, effect in zip(runs, effects, strict=False):
        if key in effect:
            return run, effect[key]
    raise KeyError(key)



# --- export ----------------------------------------------------------------

def export_tau2_actions(verifier: Verifier, *, include_allowed: bool = True) -> list[dict]:
    """The Verifier's write atoms in tau2's `evaluation_criteria.actions` shape.

    A write the trace recorded as the user's own carries `requestor: user` (D71), which is the field
    tau2's Action uses to say who performs it; dropping it asked the Candidate agent to do the user's work.
    """
    wanted = ("required", "allowed") if include_allowed else ("required",)
    order, arguments = [], {}
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        if payload.get("kind") == "write" and atom.kind in wanted:
            order.append((atom.id, payload["tool"], payload.get("requestor") or "assistant"))
            arguments.setdefault(atom.id, {})
        elif payload.get("kind") == "write_value":
            arguments.setdefault(atom.id.rsplit(".", 1)[0], {})[payload["field"]] = payload.get("raw")
    return [{"action_id": f"{verifier.task_id}_{number}", "requestor": requestor, "name": tool,
             "arguments": arguments.get(atom_id, {}), "info": None}
            for number, (atom_id, tool, requestor) in enumerate(order)]
