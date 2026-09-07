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
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.gates.verifier_suite import (
    HELPERS_SRC,
    SUCCESS_TERMINATIONS,
    _key,
    _token_in,
    _tokens,
    _user_text,
    as_run,
    atom_payload,
    canon_fn,
    classify_provenance,
    communicate_values,
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


def asked_facts(intents: Iterable[Any], run: Run, fn: Any) -> tuple[set[str], set[str]]:
    """What this Task's request asks the answer to state: the facts it names, and its words.

    The facts are keyed the way `communicate_values` keys them, so a value the user spelled one way
    and a tool result another are one key (D39). The words are everything the request and the Run's
    own user turns are written in, which is how a fact asked for by the name of the field it lives
    under ("the total") is recognised without the user having said the number.
    """
    texts = [phrase for intent in intents for phrase in request_phrases(intent)]
    texts += [_user_text(event) for event in run.events if event.type == "user_turn"]
    keys = {_key(fn, token) for text in texts for token in _tokens(text)}
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
    everywhere = set.intersection(*[set(e) for e in good_effects]) if good_effects else set()
    somewhere = set().union(*[set(e) for e in good_effects]) if good_effects else set()
    reads = {c["name"] for r in [reference] + reruns for c in run_calls(r)} - tools
    atoms: list[Atom] = []

    # D43: present in every successful re-run is required, in some is allowed, in none is not an
    # atom. A write only a failed re-run made is therefore not a forbidden atom; the write cap below
    # and verdict.py's extra-write check are what keep a Candidate from writing more than the good
    # Runs did.
    for number, key in enumerate(sorted(somewhere)):
        run, effect = _first_with(good, good_effects, key)
        atom_id = f"w{number}"
        kind = "required" if key in everywhere else "allowed"
        base = {"tool": effect["tool"], "entity": effect["entity"], "entity_raw": effect["entity_raw"],
                "id_field": effect["id_field"], "at": effect["idx"]}
        if effect["requestor"]:
            base["requestor"] = effect["requestor"]  # D71: a user-side write says so on the atom
        atoms.append(_atom(atom_id, kind, dict(base, kind="write"),
                           spans=[ptr(run, effect["idx"])],
                           description=f"{effect['tool']} writes {effect['entity'] or 'an entity'}"))
        for field in sorted(effect["values"]):
            value, canon_key = effect["args"][field], effect["values"][field]
            provenance, span = classify_provenance(run, effect["pos"], value, fn)
            agreed = all(key in e and e[key]["values"].get(field) == canon_key for e in good_effects)
            atoms.append(_atom(f"{atom_id}.{field}",
                               "required" if agreed and provenance in REQUIRED_PROVENANCE else "allowed",
                               dict(base, kind="write_value", field=field, value=canon_key, raw=value),
                               provenance=provenance, spans=[span] if span else [],
                               description=f"{effect['tool']} {field} is {text_of(value)}"))

    asked = [question_keys(r, e, fn) for r, e in zip(good, good_effects, strict=False)]
    for key in sorted(set.intersection(*[set(a) for a in asked]) if asked else set()):
        seen = asked[0][key]
        atoms.append(_atom(f"q.{key}", "question",
                           {"kind": "question", "key": key, "tool": seen.get("tool"),
                            "field": seen.get("field")},
                           spans=[seen["span"]] if seen.get("span") else [],
                           description=f"the agent asks the user about {key.split(':', 1)[-1]}"))

    said = [communicate_values(r, fn) for r in good]
    request = asked_facts([intent, task], reference, fn)
    reported: list[int] = []
    for number, key in enumerate(sorted(set.intersection(*[set(s) for s in said]) if said else set())):
        fact = said[0][key]
        payload = {"kind": "communicate", "value": key, "text": fact["text"]}
        if communicate_kind(fact, key, reference, request):
            atoms.append(_demanded_fact(f"c{number}", payload, fact["span"]))
            continue
        # Same value, same predicate, reported and never a rejection: the request did not ask for
        # this fact, so a Run that solved the Task without repeating it has done the job.
        reported.append(len(atoms))
        atoms.append(_reported_fact(f"c{number}", payload, fact["span"]))
    request_asked = _demands_something(atoms)  # read before the fallback below, which the cap ignores
    if reported and not request_asked:
        # Nothing else in this Verifier can be falsified, so the facts stand: on a Task whose request
        # names no fact and whose Reference wrote nothing, reporting them all leaves a Verifier an
        # empty Run passes, which the D79 suite is right to distrust. What the Reference told the
        # user is then the only evidence of the work, and it is demanded again.
        for at in reported:
            atoms[at] = _demanded_fact(atoms[at].id, dict(atoms[at].target, kind="communicate"),
                                       atoms[at].spans[0] if atoms[at].spans else None,
                                       asked=False)

    # The cap comes last so the rule can see whether the request asked for anything else.
    #
    # A cap of 0 is what the Reference did, and on a Task whose other Runs wrote it is also a claim
    # the Task's own evidence contradicts. Where the request asked for nothing else (nothing
    # required, no question, no fact it named), it is not written: an
    # atom no Run can fail but a writing one is not a bar, it is the shape D173 names, an empty Run
    # and the wrong Run both satisfy it, and every held-out Run that took the writing path is
    # rejected by it. The D133 route is taken rather than a cap widened to the writing group: those
    # Runs are held out as evidence that the Task has more than one path, the false-rejection number
    # counts them, and the D79 suite says a Verifier with nothing in it that can be falsified is not
    # trusted, which is the honest reading and the one that sends the Examiner to re-roll and repair.
    # Widening the cap instead would pass a path the D111 rule had just set aside, on the strength of
    # a Run no rule ruled good, and it would still not admit the writing Run: with no write atom
    # covering it the Verdict's extra-write check rejects it whatever the cap says.
    if good_effects:
        cap = max(len(e) for e in good_effects)
        if cap or not writes_elsewhere or request_asked:
            atoms.append(_atom("entity_count", "required", {"kind": "entity_count", "count": cap},
                               description=f"the Run makes at most {cap} write calls"))

    for rule in constraints or []:
        if not (rule.compiled or rule.judge_atom):
            continue  # a residual constraint is reported, never verdicted (D76)
        atoms.append(_atom(f"hard.{rule.id}", "hard",
                           {"kind": "hard", "constraint_id": rule.id, "judge": rule.judge_atom,
                            "predicate_src": rule.predicate_src, "write_tools": sorted(tools),
                            "read_tools": sorted(reads)},
                           judge=bool(rule.judge_atom), description=rule.text,
                           spans=[rule.span] if rule.span else []))

    task_id = task if isinstance(task, str) else (task.id if isinstance(task, Task) else str(task))
    return Verifier(task_id=task_id, atoms=atoms, verifier_version=verifier_version,
                    seed_run_ids=[r.run_id for r in good])


def _demanded_fact(atom_id: str, payload: dict, span: Any, asked: bool = True) -> Atom:
    """A fact the answer has to state, and why it has to."""
    why = ("which the request asked the agent about" if asked
           else "and the Verifier asks nothing else that a Run can fail")
    return _atom(atom_id, "communicate", payload, provenance="system_derived",
                 spans=[span] if span else [],
                 description=f"the final answer states {payload['text']}, {why}")


def _reported_fact(atom_id: str, payload: dict, span: Any) -> Atom:
    """A fact the answer may state: same value, same predicate, and no Run is rejected for it."""
    atom = _atom(atom_id, "allowed", payload, provenance="system_derived", spans=[span] if span else [],
                 description=f"the final answer may state {payload['text']}; the request did not ask "
                             "the agent about it, so it is reported and rejects no Run")
    return atom.model_copy(update={"target": dict(payload, kind=REPORTED_COMMUNICATE)})


def _demands_something(atoms: Iterable[Atom]) -> bool:
    """Does this Verifier already ask the Run for anything, cap aside? (the kinds a Verdict must hold)."""
    return any(atom.kind in ("required", "question", "communicate") for atom in atoms)


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
