"""The nine D79 checks over a Verifier, and the scorer they share with the Examiner (D79, D119, D122).

A gate is code with no model call in it that accepts or rejects something an agent made (D110,
D122). This module rules on a Verifier: it reads the Reference and its re-runs off disk as `Run`
records (D91), synthesizes the Runs no Runner has to write (the empty Run of check 3, the
unfinished Run of check 9), scores each against the atoms with `check_run`, and returns one
`GateResult` per check. The wrong Run, the second path and the loophole probe are Runs the caller
supplies; a check whose Run is missing is reported as not run and never as a pass.

Atom payloads: every atom carries a small structured object in `Atom.target`, read back with
`atom_payload`, and the same check as one line of the Runner's atom vocabulary in `predicate_src`, so
verdict.py evaluates it without importing anything from here (D89). The `kind` key of the target is:
  write          {tool, entity, entity_raw, id_field, at, requestor}  the write effect happened
  write_value    {..., field, value, raw}                one value of that write (D42 provenance on the Atom)
  question       {key, tool, field}                       the agent asked the user for this (D43)
  communicate    {value, text}                            the final answer states this fact
  entity_count   {count}                                  the Run makes at most this many write calls
  hard           {constraint_id, predicate_src, judge, write_tools, read_tools}  a Hard constraint or
                                                          a judge atom (D76)
`value` is the canonical comparison key; `raw` and `text` are the value as the trace had it.

The target and the predicate say the same thing on purpose: `check_run` here scores a Run off the
target and verdict.py scores it off `predicate_src`, so a check one of them makes and the other does
not is a hole no D79 gate can see. tests/gates/test_verifier_runner_agreement.py holds the two
together. The derivation that writes atoms from Runs is `kullback.examiner.derive.derive_verifier`;
it builds its atoms with `make_atom` from here, wrapping a Hard predicate with `HELPERS_SRC`.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Iterable, Optional

from kullback.runner import target as _target
from kullback.runner.records import (
    Atom,
    GateResult,
    Run,
    UserRules,
    Verifier,
    as_dict,
    canonical_json,
)

# loop.py's own stop reasons are in here: a re-run that ran to the end without a Simulated user
# stops with `agent_stop`, and reading that as a failure dropped a good re-run out of the agreement.
# Success is properly the caller's to say (`successful_run_ids`); this list is the fallback.
SUCCESS_TERMINATIONS = frozenset({"success", "stop", "user_stop", "agent_stop", "task_complete",
                                  "completed", "done"})
AFFIRMATIONS = _target.AFFIRMATIONS  # moved to kullback/runner/target.py (G3), imported here
_TOKEN = _target._TOKEN  # moved to kullback/runner/target.py (G3), imported here
_WORD = _target._WORD  # moved to kullback/runner/target.py (G3), imported here
# canon.py's default currency symbols (D39); a word starting with one is also the bare number.
CURRENCY = _target.CURRENCY  # moved to kullback/runner/target.py (G3), imported here
# What check 8 puts in an atom's place: a value no Run of the customer's world produced.
_MUTANT = "harness_mutation_no_such_value"
_NEVER_HOLDS = "def check(pre_state, write_call, transcript):\n    return False\n"
# Why check 5 had nothing to score, when the caller passed no second path: the Task has one
# Reference. It is the only reason the Examiner's stage leaves that Run out (`suite_for` passes the
# second Reference or None), and a check with no input is not evidence of a narrow Verifier (D173).
ALT_PATH_NOT_RUN = "one Reference, so there is no second path to score"
# How a check with no input words its failure line. The GateResult stays not passed, so the
# derivation keeps the Task untrusted, but a ruling on a proposal reads a line that starts with
# this as a check that did not run, never as a check the proposal failed (F45).
NOT_RUN = "not run: "


def not_run_reason(failures: Iterable[str]) -> Optional[str]:
    """Why a check did not run, when every failure line it carries says so; None for a check that ran."""
    lines = [str(line) for line in failures or ()]
    if not lines or not all(line.startswith(NOT_RUN) for line in lines):
        return None
    return "; ".join(line[len(NOT_RUN):] for line in lines)

# The transcript helpers a compiled Hard predicate may call, pasted into its source by `_predicate`
# at derivation time and by the policy compiler's sandbox at build time. One text, read by both
# sides: the Examiner's derivation and the Builder's policy.py (which re-exports it) sit above this
# package and neither imports the other. Moved byte-for-byte from builder/policy.py in phase 5.
HELPERS_SRC = '''
_YES = ("yes", "yeah", "yep", "confirm", "confirmed", "sure", "ok", "okay", "correct", "proceed")
_YES_PHRASES = ("go ahead", "please do", "that is right", "that's right")
_NO = ("no", "not", "never", "nope", "nah", "cannot", "cant", "dont", "stop", "wait", "wrong", "incorrect")

def _plain_words(text):
    return set("".join(c if (c.isalnum() or c.isspace()) else " " for c in text).split())

def user_confirmed(transcript):
    """True when the last user turn says yes to an action an assistant turn proposed (D43 case 3)."""
    messages = list(transcript or [])
    last = None
    for pos in range(len(messages) - 1, -1, -1):
        if messages[pos].get("role") == "user":
            last = pos
            break
    if last is None:
        return False
    proposal = ""
    for msg in reversed(messages[:last]):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            proposal = msg.get("content") or ""
            break
    if "?" not in proposal:
        return False  # nothing was proposed, so nothing was confirmed
    text = (messages[last].get("content") or "").lower()
    words = _plain_words(text)
    if words & set(_NO) or "n't" in text:
        return False  # a refusal that happens to carry a yes word is still a refusal
    return bool(words & set(_YES)) or any(phrase in text for phrase in _YES_PHRASES)

def called_before(transcript, *names):
    """True when the transcript already holds a tool call with one of these names."""
    wanted = set(names)
    for msg in list(transcript or []):
        for call in msg.get("tool_calls") or []:
            if call.get("name") in wanted:
                return True
    return False

def said_before(transcript, *needles):
    """True when an assistant turn already contains one of these strings, case-insensitive."""
    for msg in list(transcript or []):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").lower()
        if any(needle.lower() in text for needle in needles):
            return True
    return False
'''

# The names HELPERS_SRC binds. A caller that has to guess which function in a namespace is the
# predicate skips these, so a rule that defines no function of its own is still reported as such
# rather than answered by a helper the caller pasted in itself.
HELPER_NAMES = frozenset(
    node.name for node in ast.parse(HELPERS_SRC).body if isinstance(node, ast.FunctionDef)
)


def predicate_source(predicate_src: Optional[str]) -> str:
    """A compiled constraint as it is run anywhere: the transcript helpers, then the rule itself.

    The compiler's prompt tells the model that `user_confirmed`, `called_before` and `said_before`
    are already in scope, so a rule about a prior confirmation is one line. Whoever runs the rule
    owes it those definitions. Build 10's gate execed the rule alone and three constraints failed
    with `NameError: name 'called_before' is not defined`, a failure of ours reported against the
    model, so the assembly is stated once here and both runners call it.
    """
    return HELPERS_SRC + "\n" + (predicate_src or "")


def predicate_args(case: Any) -> tuple[dict, dict, list]:
    """The three arguments a compiled constraint takes, read off one of its cases.

    `def check(pre_state, write_call, transcript)` is the signature the compiler's contract states
    and its static check enforces, and it is what `builder/policy.py`'s sandbox, this module's Hard
    wrapper and `gates/artifacts.py`'s policy gate all hand a predicate. Build 8's gate handed the
    whole case as one argument instead and every compiled constraint failed with a TypeError, so
    the argument list lives here rather than at each call site.
    """
    row = dict(case or {})
    return row.get("pre_state") or {}, row.get("write_call") or {}, row.get("transcript") or []


# --- reading Runs off disk (D91) ------------------------------------------

load_run = _target.load_run  # moved to kullback/runner/target.py (G3), imported here


as_run = _target.as_run  # moved to kullback/runner/target.py (G3), imported here


# --- events ----------------------------------------------------------------

_payload = _target._payload  # moved to kullback/runner/target.py (G3), imported here


_reply = _target._reply  # moved to kullback/runner/target.py (G3), imported here


_assistant_text = _target._assistant_text  # moved to kullback/runner/target.py (G3), imported here


_user_text = _target._user_text  # moved to kullback/runner/target.py (G3), imported here


_entity = _target._entity  # moved to kullback/runner/target.py (G3), imported here


ptr = _target.ptr  # moved to kullback/runner/target.py (G3), imported here


resolve_write_tools = _target.resolve_write_tools  # moved to kullback/runner/target.py (G3), imported here


_args = _target._args  # moved to kullback/runner/target.py (G3), imported here


run_calls = _target.run_calls  # moved to kullback/runner/target.py (G3), imported here


write_effects = _target.write_effects  # moved to kullback/runner/target.py (G3), imported here


# --- canonicalization and text matching ------------------------------------

canon_fn = _target.canon_fn  # moved to kullback/runner/target.py (G3), imported here


_key = _target._key  # moved to kullback/runner/target.py (G3), imported here


text_of = _target.text_of  # moved to kullback/runner/target.py (G3), imported here


_texts = _target._texts  # moved to kullback/runner/target.py (G3), imported here


_token_in = _target._token_in  # moved to kullback/runner/target.py (G3), imported here


_words_of = _target._words_of  # moved to kullback/runner/target.py (G3), imported here


_matches = _target._matches  # moved to kullback/runner/target.py (G3), imported here


_tokens = _target._tokens  # moved to kullback/runner/target.py (G3), imported here


# --- provenance (D42) ------------------------------------------------------

classify_provenance = _target.classify_provenance  # moved to kullback/runner/target.py (G3), imported here


_preceded_by_question = _target._preceded_by_question  # moved to kullback/runner/target.py (G3), imported here


_next_user = _target._next_user  # moved to kullback/runner/target.py (G3), imported here


# --- questions and communicate facts (D43) --------------------------------

question_keys = _target.question_keys  # moved to kullback/runner/target.py (G3), imported here


communicate_values = _target.communicate_values  # moved to kullback/runner/target.py (G3), imported here


# --- atoms: the payload and the predicate vocabulary -----------------------

atom_payload = _target.atom_payload  # moved to kullback/runner/target.py (G3), imported here


# The pieces every generated predicate shares: value matching that agrees with the Builder's own
# (verbatim first, canonical second, D39) and the slice of the transcript before one call.
_SPANS_SRC = '''
def _has_token(text, needle):
    hay = str(text).lower()
    need = str(needle).lower()
    if not need or not hay:
        return False
    start = 0
    while True:
        at = hay.find(need, start)
        if at < 0:
            return False
        before = hay[at - 1] if at else " "
        after = hay[at + len(need)] if at + len(need) < len(hay) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        start = at + 1


def _said_word(out, word):
    word = word.strip("./-")
    if word:
        out.append(str(canon(word)))
        if word[:1] in "$€£¥" and word[1:]:
            out.append(str(canon(word[1:])))
    return out


def _said_words(text):
    out = []
    word = ""
    for char in str(text):
        if char.isalnum() or char in "#$._/-€£¥":
            word = word + char
        else:
            _said_word(out, word)
            word = ""
    return _said_word(out, word)


def _holds(text, value):
    parts = value if isinstance(value, (list, tuple)) else [value]
    if not parts:
        return False
    if all(_has_token(text, part) for part in parts):
        return True
    words = _said_words(text)
    return all(str(canon(part)) in words for part in parts)


def _before(turns, name, seen):
    out = []
    hits = 0
    for turn in turns:
        for made in turn.get("tool_calls") or []:
            if made.get("name") == name:
                if hits == seen:
                    return out
                hits = hits + 1
        out.append(turn)
    return out
'''

# A Hard constraint is a before-write predicate (policy.py's contract): it sees the transcript up to
# the call it is judging and nothing after it, and it is asked about every call the Run made except
# the ones the seed Runs only ever read with, so a rule about a tool the Reference never used still
# fires. `start_state` is the state at the top of the Run; a Run records no per-write snapshot, so a
# rule that reads state judges the second write of a Run against the first write's input.
#
# `judged()` says how many calls the rule was asked about, because "every call held" and "there was
# no call" are different answers and `check()` alone cannot tell them apart: it returns True for
# both, which is what the Verdict needs (a Run that never touched the tool broke no rule) and not
# what a gate needs. On a Run with no write call, replacing the rule with one that never holds still
# returns True, so the mutation check counted an atom that had nothing to say as an atom that could
# not be falsified: 27 Tasks of one build and 28 of another, every one of them read-only (D173).
# `hard_holds` reads `judged()` and answers None there, the way it already refuses to read a missing
# `check` as "the constraint held".
_HARD_WRAPPER = """{helpers}
{spans}
{rule}

_rule = check
_WRITE_TOOLS = {write_tools}
_READ_TOOLS = {read_tools}
_judged = []


def judged():
    return len(_judged)


def check():
    seen = {{}}
    del _judged[:]
    for _call in calls:
        _name = _call.get("name") or ""
        _count = seen.get(_name, 0)
        seen[_name] = _count + 1
        if _call.get("error"):
            continue
        if _name in _READ_TOOLS and _name not in _WRITE_TOOLS:
            continue
        _args = _call.get("args") or _call.get("arguments") or {{}}
        _turns = _before(transcript, _name, _count)
        _judged.append(_name)
        if not _rule(start_state, {{"name": _name, "arguments": _args}}, _turns):
            return False
    return True
"""

# D43: a question whose answer landed in a write binds that write's value to the user's own reply in
# the Candidate's Run. The field name is not the question, so this reads the Run rather than grepping
# the agent's wording for the field name.
_FIELD_QUESTION_WRAPPER = """{spans}

def _elicited(turns, value):
    asked = ""
    for turn in turns:
        if turn.get("role") == "user":
            if _holds(turn.get("content") or "", value):
                return "?" in asked
            asked = ""
        elif turn.get("content"):
            asked = turn.get("content")
    return False


def check():
    seen = 0
    for _call in calls:
        if _call.get("name") != {tool}:
            continue
        _args = _call.get("args") or _call.get("arguments") or {{}}
        if not _call.get("error") and {field} in _args:
            if _elicited(_before(transcript, {tool}, seen), _args[{field}]):
                return True
        seen = seen + 1
    return False
"""

# The user said yes to something the agent proposed, and only then did the agent write.
_CONFIRM_WRAPPER = """_AFFIRM = {words}


def check():
    asked = False
    confirmed = False
    for turn in transcript:
        for _call in turn.get("tool_calls") or []:
            if _call.get("name") == {tool} and confirmed:
                return True
        if turn.get("role") == "user":
            text = " ".join(str(turn.get("content") or "").split()).lower()
            if asked and any(text.startswith(word) for word in _AFFIRM):
                confirmed = True
            asked = False
        elif "?" in (turn.get("content") or ""):
            asked = True
    return False
"""


start_state = _target.start_state  # moved to kullback/runner/target.py (G3), imported here


_result_turn = _target._result_turn  # moved to kullback/runner/target.py (G3), imported here


_transcript = _target._transcript  # moved to kullback/runner/target.py (G3), imported here


hard_holds = _target.hard_holds  # moved to kullback/runner/target.py (G3), imported here



def _write_fields(payload: dict) -> dict:
    """The arguments that name the write this atom is about: its entity, and its own field."""
    fields: dict = {}
    if payload.get("id_field"):
        fields[payload["id_field"]] = payload.get("entity_raw")
    if payload.get("kind") == "write_value":
        fields[payload["field"]] = payload.get("raw")
    return fields


def _predicate(payload: dict, helpers: str = "") -> str:
    """One atom target as source in the Runner's atom vocabulary.

    Stored on every atom for now and read only for Hard rules, which are compiled policy
    source by nature. Every other atom is scored off its structured target by the one
    interpreter in kullback/runner/target.py, which the Verdict and the gates both call (G3).
    `helpers` is the transcript helper source a compiled Hard rule may call (policy.py's,
    passed by the derivation); a gate that only needs a rule that never holds passes none.
    """
    kind = payload.get("kind")
    if kind in ("write", "write_value"):
        fields = _write_fields(payload)
        return f"wrote({payload['tool']!r}, **{fields!r})" if fields else f"wrote({payload['tool']!r})"
    if kind == "entity_count":
        return f"writes_count() <= {int(payload['count'])}"
    if kind == "question":
        # question_keys() is the only producer of a "question" payload, and both of its shapes
        # ("confirm:{tool}" and "field:{field}", always with tool and field set) land in one of
        # these two branches, so there is no third shape left for a predicate to fall back to.
        head, _, rest = str(payload.get("key") or "").partition(":")
        if head == "confirm":
            return _CONFIRM_WRAPPER.format(words=repr(tuple(AFFIRMATIONS)), tool=repr(rest))
        if payload.get("tool") and payload.get("field"):
            return _FIELD_QUESTION_WRAPPER.format(spans=_SPANS_SRC, tool=repr(payload["tool"]),
                                                  field=repr(payload["field"]))
    if kind == "communicate":
        return f"communicated({str(payload.get('text') or payload.get('value'))!r})"
    if kind == "hard":
        rule = payload.get("predicate_src")
        if not rule or payload.get("judge"):
            return ""  # a judge atom is answered by judge.py, never by code (D76)
        return _HARD_WRAPPER.format(helpers=helpers, spans=_SPANS_SRC, rule=rule,
                                    write_tools=repr(sorted(payload.get("write_tools") or [])),
                                    read_tools=repr(sorted(payload.get("read_tools") or [])))
    return ""


def make_atom(atom_id: str, kind: str, payload: dict, *, helpers: str = "", **fields: Any) -> Atom:
    """One atom: the structured target for the checks here, the predicate source for the Runner."""
    return Atom(id=atom_id, kind=kind, target=payload, predicate_src=_predicate(payload, helpers), **fields)


# --- scoring a Run against the atoms (the gates' side; verdict.py has its own, D91) ---

verifier_write_tools = _target.verifier_write_tools  # moved to kullback/runner/target.py (G3), imported here


scored_write_tools = _target.scored_write_tools  # moved to kullback/runner/target.py (G3), imported here


names_no_row = _target.names_no_row  # moved to kullback/runner/target.py (G3), imported here


check_run = _target.check_run  # moved to kullback/runner/target.py (G3), imported here


_extra_write = _target._extra_write  # moved to kullback/runner/target.py (G3), imported here


# --- D79 validation --------------------------------------------------------

# The stage each D79 check reports under, and the name artifacts.py's `verifier_gate` wants back.
D79_STAGES = {
    "verifier_provenance_spans": "provenance_spans",
    "verifier_oracle": "oracle_passes",
    "verifier_empty_run": "empty_fails",
    "verifier_wrong_run": "plausible_wrong_fails",
    "verifier_unfinished_run": "unsolved_state_fails",
    "verifier_alt_path": "second_path_passes",
    "verifier_loophole": "loophole_probe_fails",
    "verifier_leak": "leak_check_clean",
    "verifier_mutation": "mutation_flips",
}


def d79_results(gates: Iterable[GateResult]) -> dict[str, bool]:
    """The suite's answer in the shape artifacts.py's `verifier_gate` reads: one bool per D79 check."""
    seen = {gate.stage: bool(gate.passed) for gate in gates}
    return {name: seen.get(stage, False) for stage, name in D79_STAGES.items()}


def validate_verifier(verifier: Verifier, reference_run: Any, empty_run: Any = None, wrong_run: Any = None,
                      alt_path_run: Any = None, intent_text: Optional[str] = None,
                      user_rules: Optional[UserRules] = None, *, canon: Any = None,
                      write_tools: Optional[Iterable[str]] = None, model: Any = None,
                      run_probe: Optional[Callable] = None,
                      seed_runs: Optional[Iterable[Any]] = None,
                      intent: Any = None) -> list[GateResult]:
    """The nine D79 checks as GateResults. A check whose input is missing fails as "not run".

    Nothing here executes a Run (D91), so the wrong Run, the second path and the loophole probe are
    the caller's to supply; a check the caller left out is reported as not run, the way artifacts.py
    counts it, and never as a pass. The empty Run and the unfinished Run need no input and are
    synthesized from the Reference.

    `intent` is the Task's Intent record where the caller has it, which turns check 7 into an audit
    of the D196 strip: the record says which columns the miner already took out, and the check names
    the column of anything it missed.
    """
    reference = as_run(reference_run)
    runs = {r.trace_id or r.run_id: r for r in [as_run(s) for s in (seed_runs or [])] + [reference]}

    def score(atoms_of: Verifier, run):
        return check_run(atoms_of, run, canon, write_tools=write_tools)

    def scored(run):
        return score(verifier, run)

    def oracle_scored(run):
        """The oracle plus admission: a Hard defect fails the Verifier, never the Run.

        check_run passes a Run over a defective Hard rule (it is not a Candidate failure),
        so the oracle asks the extra question the reward must not: did every Hard rule run.
        A defective rule fails admission here while probes, trust and loosening keep scoring
        the Run a pass through check_run.
        """
        passed, failing = score(verifier, run)
        if not passed:
            return passed, failing
        fn = canon_fn(canon)
        tools = scored_write_tools(verifier, as_run(run), write_tools)
        for atom in verifier.atoms:
            if atom.kind == "hard" and _target.hard_defect(atom, as_run(run), tools, fn):
                return False, atom.id
        return True, None

    return [
        _spans_gate(verifier, runs, canon_fn(canon)),
        _run_gate("verifier_oracle", oracle_scored, reference, expect_pass=True),
        _run_gate("verifier_empty_run", scored,
                  empty_run if empty_run is not None else _empty_run(reference), expect_pass=False,
                  atoms=verifier.atoms),
        _run_gate("verifier_wrong_run", scored, wrong_run, expect_pass=False, atoms=verifier.atoms),
        _run_gate("verifier_unfinished_run", scored, unfinished_run(verifier, reference, canon),
                  expect_pass=False, atoms=verifier.atoms),
        _run_gate("verifier_alt_path", scored, alt_path_run, expect_pass=True, missing=ALT_PATH_NOT_RUN),
        loophole_probe(verifier, model, run_probe=run_probe, canon=canon, write_tools=write_tools),
        _leak_gate(verifier, reference, intent_text, user_rules, runs.values(), intent),
        _mutation_gate(verifier, reference, score, canon_fn(canon), write_tools),
    ]


def wrong_run(verifier: Verifier, reference: Any, canon: Any = None) -> Optional[Run]:
    """Check 4's Run, built from the Reference with no Runner: every required write aimed at the wrong entity.

    D79 names the two shapes of a plausible wrong End state, the wrong entity and the missing
    question. Writes are swapped first: the tool is right and the row is not, which is the mistake a
    Candidate makes when it picks the wrong order, and the swap takes another id the Reference itself
    showed under the same field where one exists. A Verifier with no required write instead loses
    what the agent asked and told the user. A Verifier that requires nothing has no wrong Run, and
    the check stays not run rather than passing on a Run that is not wrong.
    """
    fn = canon_fn(canon)
    reference = as_run(reference)
    run = reference.model_copy(deep=True)
    run.run_id = f"{reference.run_id}.wrong"
    writes = [atom_payload(a) for a in verifier.atoms
              if a.kind == "required" and atom_payload(a).get("kind") == "write"]
    swapped = 0
    for payload in writes:
        event = next((e for e in run.events if e.type == "tool_call" and e.idx == payload.get("at")), None)
        field = payload.get("id_field")
        if event is None or not field:
            continue
        args = dict(_args(event))
        current = text_of(fn(args.get(field)))
        others = [v for v in _values_named(reference, field) if text_of(fn(v)) != current]
        args[field] = others[0] if others else f"{_MUTANT}_{current}"
        event.payload = dict(event.payload, args=args)
        swapped += 1
    if swapped:
        return run
    demanded = [a for a in verifier.atoms
                if a.kind == "required" or atom_payload(a).get("kind") in ("question", "communicate")]
    if not demanded:
        return None
    for event in run.events:
        if event.type == "model_call":
            event.payload = {"reply": {"content": "", "tool_calls": []}}
        elif event.type == "user_turn":
            event.payload = dict(event.payload, text="", content="")
    return run


def _values_named(run: Run, field: str) -> list[Any]:
    """Every scalar the Reference showed under this field name, in tool arguments and results, in order."""
    out: list[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == field and not isinstance(item, (dict, list, tuple)) and item not in out:
                    out.append(item)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for event in run.events:
        if event.type == "tool_call":
            walk(_args(event))
        elif event.type == "tool_result":
            walk(_payload(event).get("result"))
    return out


def unfinished_run(verifier: Verifier, reference: Any, canon: Any = None) -> Optional[Run]:
    """Check 9's Run: the Reference stopped one step short, which must not score a pass (D119).

    GLM 5.3's unsolved-state check, beside the oracle and the no-op: a Verifier that rewards a Run
    that got most of the way there rewards leaving the job unfinished. Where the cut falls depends
    on what the Task's outcome is (D156, D173):

      - the Verifier requires a write: just before the last required write, so the Run reads as one
        that did the reading and never acted;
      - the Verifier requires no write: the outcome is the answer, so the cut falls before the first
        assistant turn that states a fact a communicate atom asks for, else before the last
        assistant turn. The first, not the last: a recorded conversation states the facts and then
        repeats them in a closing turn, so cutting before the closing turn leaves the answer
        standing earlier in the Run and the cut Run passes. Cutting before the first statement is
        the state this check is about, an agent that read the world and never told the user;
      - nothing else to go on: before the last tool call, else before the final assistant turn.

    A Reference with nothing to cut has no unfinished Run and the check stays not run. Nothing after
    the cut survives, so the Run reads as one that ran out of turns rather than one that did the
    write and skipped the goodbye.
    """
    reference = as_run(reference)
    run = reference.model_copy(deep=True)
    run.run_id = f"{reference.run_id}.unfinished"
    writes = [atom_payload(a).get("at") for a in verifier.atoms
              if a.kind == "required" and atom_payload(a).get("kind") == "write"]
    present = {e.idx for e in run.events if e.type == "tool_call"}
    cut = max((at for at in writes if isinstance(at, int) and at in present), default=None)
    if cut is None and not writes:
        cut = _first_answer(verifier, run, canon_fn(canon))
    for kind in ("tool_call", "model_call"):
        if cut is not None:
            break
        cut = max((e.idx for e in run.events if e.type == kind), default=None)
    if cut is None:
        return None
    run.events = [e for e in run.events if e.idx < cut]
    run.termination_reason = "max_turns"
    return run


def _first_answer(verifier: Verifier, run: Run, fn: Callable) -> Optional[int]:
    """The idx of the first assistant turn stating a fact this Verifier's communicate atoms ask for,
    or of the last assistant turn when it asks for none. None when the Run has no assistant turn.

    The facts are keyed the way `communicate_values` keys them, so an atom and the turn that states
    it are compared as one canonical value and not as two spellings of it (D39).
    """
    wanted = {p.get("value") for p in map(atom_payload, verifier.atoms) if p.get("kind") == "communicate"}
    answers = [e for e in run.events if e.type == "model_call" and _assistant_text(e)]
    for event in answers:
        if wanted & {_key(fn, token) for token in _tokens(_assistant_text(event))}:
            return event.idx
    return answers[-1].idx if answers else None


def _empty_run(reference: Run) -> Run:
    """Check 3's Run that does nothing (CUA-Gym's `r(s_init) < r(s_gold)`), which needs no Runner."""
    return Run(run_id=f"{reference.run_id}.empty", task_id=reference.task_id, env_id=reference.env_id,
               events=[], termination_reason="max_turns")


def _spans_gate(verifier: Verifier, runs: dict[str, Run], fn: Callable) -> GateResult:
    """Check 1: a user-stated value sits in a user turn, a system-derived value in an earlier tool result.

    A span names the Run it came from, which for an atom the Reference does not hold is a re-run;
    resolving every span against the Reference reported the wrong turn for those atoms.
    """
    failures, checked = [], 0
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        if payload.get("kind") != "write_value" or atom.provenance in (None, "agent_chosen"):
            continue
        checked += 1
        span = atom.spans[0] if atom.spans else None
        run = runs.get(span.file_hash) if span else None
        event = {e.idx: e for e in run.events}.get(span.msg_index) if run else None
        if span is None:
            failures.append(f"{atom.id}: no span")
        elif run is None:
            failures.append(f"{atom.id}: the span names Run {span.file_hash}, which was not supplied")
        elif event is None:
            failures.append(f"{atom.id}: no event {span.msg_index} in Run {span.file_hash}")
        elif atom.provenance in ("user_stated", "user_elicited"):
            if event.type != "user_turn" or not _matches(_user_text(event), payload.get("raw"), fn):
                failures.append(f"{atom.id}: span is not a user turn holding the value")
        elif event.type != "tool_result" or not _matches(canonical_json(_payload(event).get("result")),
                                                         payload.get("raw"), fn):
            failures.append(f"{atom.id}: span is not a tool result holding the value")
        elif span.msg_index is not None and payload.get("at") is not None and span.msg_index >= payload["at"]:
            failures.append(f"{atom.id}: span is not before the write")
    return GateResult(stage="verifier_provenance_spans", passed=not failures,
                      metrics={"atoms_checked": checked}, failures=failures)


# An expected-fail Run passed a Verifier none of whose atoms check_run evaluates (F37).
NO_ATOM_CHECKED = "no atom of the Verifier is one a Run is checked on, so no Run can fail it"


def _run_gate(stage: str, scored: Callable, run: Any, *, expect_pass: bool,
              missing: str = "no Run was supplied for this check",
              atoms: Iterable[Atom] = ()) -> GateResult:
    """Checks 2 to 5: one Run, one expected outcome. No Run, no evidence, so no pass.

    A check with no Run is still a failure, and `missing` says which kind it is: the second path is
    missing because the Task has one Reference and not because the Verifier turned a second path
    away, and a reader that cannot tell those apart reaches for the wrong repair (D173).

    A Run expected to fail that passed passed every atom check_run evaluates, so the result names
    those by id and kind as the atoms to change (F37), and never an atom check_run skips.
    """
    if run is None:
        return GateResult(stage=stage, passed=False, metrics={"skipped": True, "not_run_reason": missing},
                          failures=[f"{NOT_RUN}{missing}"])
    passed, failing_atom = scored(run)
    want = "pass" if expect_pass else "fail"
    failures = [] if passed is expect_pass else [f"expected {want}, got {'pass' if passed else 'fail'}"]
    metrics = {"run_passed": passed, "failing_atom": failing_atom}
    if passed and not expect_pass:
        checked = _target.evaluated_atoms(atoms)
        metrics["passing_atoms"] = [{"id": atom.id, "kind": atom.kind} for atom in checked]
        if not checked:
            failures.append(NO_ATOM_CHECKED)
    return GateResult(stage=stage, passed=passed is expect_pass, failures=failures, metrics=metrics)


def _mutation_gate(verifier: Verifier, reference: Run, score: Callable, fn: Callable,
                   write_tools: Optional[Iterable[str]] = None) -> GateResult:
    """Check 8: change what an atom demands and the Reference must stop passing.

    An atom nothing can fail is an atom that is not being checked, which is how a Hard constraint
    that never gets evaluated, or a value the scorer never compares, hides in a passing suite. An
    atom with nothing to mutate is a different thing and is not counted as a failed flip (D173): a
    write cap of 0 has no smaller cap to ask for, and a Hard rule the Run gave no call to judge
    would answer the same whatever rule it carried. Both are skipped and neither is a failure of
    the atom. What is a failure is a Verifier in which no atom at all can be mutated, since nothing
    in it can be made to say something false about its own Reference.
    """
    tools = scored_write_tools(verifier, reference, write_tools)
    failures, mutated, inert = [], 0, 0
    for atom in verifier.atoms:
        mutant = _mutant(atom, fn)
        if mutant is None or _judged_no_call(atom, reference, tools, fn):
            inert += 1
            continue
        mutated += 1
        changed = Verifier(task_id=verifier.task_id, verifier_version=verifier.verifier_version,
                           seed_run_ids=verifier.seed_run_ids,
                           atoms=[mutant if a.id == atom.id else a for a in verifier.atoms])
        if score(changed, reference)[0]:
            failures.append(f"{atom.id}: the Reference still passes when this atom is changed")
    if not mutated:
        failures.append(f"no atom of this Verifier can be mutated ({inert} of them have nothing to "
                        "mutate or judged no call of the Reference), so nothing in it can be falsified")
    return GateResult(stage="verifier_mutation", passed=not failures,
                      metrics={"atoms_mutated": mutated, "atoms_not_mutable": inert}, failures=failures)


def _judged_no_call(atom: Atom, run: Run, write_tools: set[str], fn: Callable) -> bool:
    """Was this compiled Hard atom asked about no call of the Run, so that no rule in its place
    would answer differently? `hard_holds` says so by answering None over an atom that is code."""
    payload = atom_payload(atom)
    if payload.get("kind") != "hard" or atom.judge or not atom.predicate_src:
        return False
    return hard_holds(atom, run, write_tools, fn) is None


def _mutant(atom: Atom, fn: Callable) -> Optional[Atom]:
    """The same atom asking for something the Reference did not do, or None when it cannot be mutated."""
    payload = atom_payload(atom)
    kind = payload.get("kind")
    if atom.kind not in ("required", "question", "communicate", "hard"):
        return None
    if kind == "write":
        return make_atom(atom.id, atom.kind, dict(payload, entity=_key(fn, _MUTANT), entity_raw=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "write_value":
        return make_atom(atom.id, atom.kind, dict(payload, value=_key(fn, _MUTANT), raw=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "entity_count":
        # A cap of 0 has no smaller cap: max(count - 1, 0) is the atom itself, so the "mutant" asked
        # for exactly what the original asked for and the Reference rightly still passed. Nothing to
        # mutate is not an atom that cannot be failed, and the gate is told so rather than shown a
        # flip that was never made (D173).
        count = int(payload.get("count", 0))
        return None if count <= 0 else make_atom(atom.id, atom.kind, dict(payload, count=count - 1),
                                                 description=atom.description)
    if kind == "question":
        return make_atom(atom.id, atom.kind, dict(payload, key=f"{payload.get('key')}.{_MUTANT}",
                                              field=_MUTANT), description=atom.description)
    if kind == "communicate":
        return make_atom(atom.id, atom.kind, dict(payload, value=_key(fn, _MUTANT), text=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "hard" and not atom.judge and payload.get("predicate_src"):
        return make_atom(atom.id, atom.kind, dict(payload, predicate_src=_NEVER_HOLDS),
                     description=atom.description)
    return None


def atom_column(atom: Atom) -> str:
    """The column an atom's value sat under, as the Intent strip and a finding name it (D196).

    A write value names its tool and its field; a fact the answer states names the atom's own kind,
    because a communicated token is not a column of anything. Nothing here is a value.
    """
    payload = atom_payload(atom)
    field = payload.get("field") or payload.get("id_field") or payload.get("key")
    tool = payload.get("tool")
    if field and tool:
        return f"{tool}.{field}"
    return str(field or tool or atom.kind)


def _leak_gate(verifier: Verifier, reference: Run, intent_text: Optional[str],
               user_rules: Optional[UserRules], runs: Iterable[Run] = (),
               intent: Any = None) -> GateResult:
    """Check 7: constants only the Verifier should know, found in the Intent or the Simulated user rules.

    A value is only a secret when no user said it, and the users who count are the users of every
    seed recording of the Task, not the one recording that happens to be the Reference. The Intent is
    mined from all of them (D83), so reading `said_by_user` off the Reference alone made this check
    stricter than the miner it polices: on the last build, of 24 leaked values across 14 Tasks, 18
    were spoken by a user in another recording of the same Task, and those 14 Tasks were blocked for
    nothing. `runs` is the Task's other seed Runs; the Reference is read whether or not it is among
    them, so a caller that has only the Reference keeps the behaviour it always had.

    With `intent`, the Intent record itself, this is an audit of the D196 strip rather than the first
    time anyone looked: the miner has already taken every system-known value out of the line and said
    on the record which columns it took, so what reaches here is what the strip missed. The check
    still fails, exactly as it did before, and it names the column each missed value came from, which
    is what a finding can be keyed on and what the next build's strip has to cover. A caller with no
    record passes none and the check reads as it always did.
    """
    said_by_user = " ".join(_user_text(e) for run in [reference, *runs]
                            for e in run.events if e.type == "user_turn")
    secrets: dict[str, str] = {}
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        if atom.provenance not in ("system_derived", "agent_chosen"):
            continue
        # Numbers leak as readily as strings: an amount read off a tool result and repeated in the
        # Intent is the constant D79 check 7 is looking for, whether the trace stored it as 150.0 or "150.0".
        value = payload["text"] if payload.get("text") is not None else payload.get("raw")
        for text in _texts(value) if value is not None else []:
            if len(text) > 1 and text not in ("true", "false", "null") and not _token_in(said_by_user, text):
                secrets.setdefault(text, atom_column(atom))
    blobs = {"intent": intent_text or ""}
    if user_rules is not None:
        blobs["user_rules"] = canonical_json(as_dict(user_rules))
    missed = [(text, secrets[text], where) for text in sorted(secrets)
              for where, blob in blobs.items() if _token_in(blob, text)]
    failures = [f"{where} leaks {text} (column {column})" for text, column, where in missed]
    stripped = list(getattr(intent, "stripped", ()) or ())
    return GateResult(stage="verifier_leak", passed=not failures,
                      metrics={"constants": len(secrets), "columns": sorted({c for _, c, _ in missed}),
                               "stripped": len(stripped), "audited": intent is not None},
                      failures=failures)


def loophole_probe(verifier: Verifier, model: Any, *, run_probe: Optional[Callable] = None, canon: Any = None,
                   write_tools: Optional[Iterable[str]] = None) -> GateResult:
    """Check 6: an agent told to reach the End state while skipping the policy step must score fail.

    A gate runs nothing (D91, D122), so the caller passes `run_probe(model, verifier) -> Run`. With no model
    or no runner the probe is skipped, and a skipped probe is not evidence that the Verifier is tight.
    """
    if model is None or run_probe is None:
        missing = "no model" if model is None else "no run_probe"
        why = f"{missing}, so the Verifier is not known to be tight"
        return GateResult(stage="verifier_loophole", passed=False,
                          metrics={"skipped": True, "not_run_reason": why},
                          failures=[f"{NOT_RUN}{why}"])
    passed, failing_atom = check_run(verifier, run_probe(model, verifier), canon, write_tools=write_tools)
    return GateResult(stage="verifier_loophole", passed=not passed,
                      metrics={"probe_passed": passed, "failing_atom": failing_atom},
                      failures=["the loophole probe reached the End state and scored pass"] if passed else [])
