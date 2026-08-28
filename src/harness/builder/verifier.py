"""Derive one Task's Verifier from its Reference and k re-runs read off disk, and run the D79 checks over it.

Nothing here executes a Run (D91): re-runs arrive as `Run` JSONL paths the Runner already wrote.

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
not is a hole no D79 gate can see. tests/test_verifier_runner_agreement.py holds the two together.
"""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from harness.shared.canon import canon_value
from harness.shared.records import (
    Atom,
    Constraint,
    Event,
    GateResult,
    RawPtr,
    Run,
    Task,
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
AFFIRMATIONS = ("yes", "yeah", "yep", "sure", "please do", "go ahead", "confirm", "correct", "ok", "okay")
REQUIRED_PROVENANCE = ("user_stated", "system_derived")
EVENT_TYPES = frozenset({"model_call", "tool_call", "tool_result", "user_turn", "error", "stop"})
_TOKEN = re.compile(r"[#$]?[A-Za-z0-9][A-Za-z0-9_./#-]*")
_WORD = re.compile(r"[A-Za-z0-9#$€£¥._/-]+")
CURRENCY = "$€£¥"  # canon.py's default symbols; a word starting with one is also the bare number
# What check 8 puts in an atom's place: a value no Run of the customer's world produced.
_MUTANT = "harness_mutation_no_such_value"
_NEVER_HOLDS = "def check(pre_state, write_call, transcript):\n    return False\n"


# --- reading Runs off disk (D91) ------------------------------------------

def load_run(path: Any) -> Run:
    """Read one Run from a JSONL of events (header or footer lines included) or from a whole-Run JSON.

    loop.py writes the Starting and End state on a trailing footer line, which is not a `Run` field:
    those keys become a stop event, the way verdict.py reads the same file, so a Hard rule that reads
    the state sees it here too.
    """
    file = Path(path)
    header: dict = {}
    events: list[dict] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") in EVENT_TYPES:
            events.append(obj)
        else:
            events.extend(obj.pop("events", None) or [])
            header.update(obj)
    extra = {key: value for key, value in header.items() if key not in Run.model_fields}
    if extra:
        events.append({"type": "stop", "payload": extra})
    header = {key: value for key, value in header.items() if key in Run.model_fields}
    header.setdefault("run_id", file.stem)
    header["events"] = [dict(event, idx=event.get("idx", pos)) for pos, event in enumerate(events)]
    return Run.model_validate(header)


def load_runs(paths: Iterable[Any]) -> list[Run]:
    return [_as_run(p) for p in paths]


def _as_run(obj: Any) -> Run:
    return obj if isinstance(obj, Run) else load_run(obj)


def _successful(run: Run, successful_run_ids: Optional[Iterable[str]]) -> bool:
    if successful_run_ids is not None:
        return run.run_id in set(successful_run_ids)
    return (run.termination_reason or "") in SUCCESS_TERMINATIONS


# --- events ----------------------------------------------------------------

def _payload(event: Event) -> dict:
    return event.payload or {}


def _reply(event: Event) -> dict:
    """A model_call's assistant message, whether the payload nests it under `reply` or not."""
    payload = _payload(event)
    return payload.get("reply") if isinstance(payload.get("reply"), dict) else payload


def _assistant_text(event: Event) -> str:
    return str(_reply(event).get("content") or "") if event.type == "model_call" else ""


def _user_text(event: Event) -> str:
    return str(_payload(event).get("content") or _payload(event).get("text") or "")


def _entity(args: dict, fn: Callable) -> tuple[str, Any, str]:
    """The id a write acts on: its field, its raw value and its canonical form.

    A scalar id wins over a list of ids, so tau2's `exchange_delivered_order_items` keys on
    `order_id` and not on `item_ids`, whose order two equally good Runs may differ on.
    """
    named = [f for f in sorted(args) if f in ("id", "ids") or f.endswith(("_id", "_ids"))]
    scalar = [f for f in named if not isinstance(args[f], (list, tuple, dict))]
    field = next(iter(scalar + named), "")
    return (field, args[field], _text_of(fn(args[field]))) if field else ("", None, "")


def _ptr(run: Run, idx: Optional[int]) -> RawPtr:
    return RawPtr(file_hash=run.trace_id or run.run_id, msg_index=idx)


def resolve_write_tools(runs: Iterable[Run], write_tools: Optional[Iterable[str]] = None) -> set[str]:
    """The write tools the caller named, plus any tool a Run's own event marked as a write."""
    tools = set(write_tools or ())
    for run in runs:
        for event in run.events:
            marked = _payload(event).get("kind") == "write" or _payload(event).get("is_write") is True
            if event.type == "tool_call" and marked:
                tools.add(str(_payload(event).get("name") or ""))
    tools.discard("")
    return tools


def _args(event: Event) -> dict:
    return _payload(event).get("args") or _payload(event).get("arguments") or {}


def _calls(run: Run) -> list[dict]:
    """Every tool call of the Run with the error its result carried, paired by call id (D67)."""
    out: list[dict] = []
    for pos, event in enumerate(run.events):
        if event.type == "tool_call":
            out.append({"i": len(out), "pos": pos, "idx": event.idx, "id": _payload(event).get("id"),
                        "name": str(_payload(event).get("name") or ""), "error": None,
                        "requestor": _payload(event).get("requestor"), "args": _args(event)})
        elif event.type == "tool_result":
            for call in reversed(out):
                if _payload(event).get("id") in (None, call["id"]):
                    call["error"] = _payload(event).get("error")
                    break
    return out


def _effects(run: Run, write_tools: set[str], fn: Callable) -> dict[str, dict]:
    """The Run's write set: one entry per write call that succeeded, keyed by tool, entity and repeat.

    A call whose result carried an error changed nothing (D67), so it is not an effect and never
    becomes an atom the next Run has to reproduce.
    """
    out: dict[str, dict] = {}
    for call in _calls(run):
        if call["name"] not in write_tools or call["error"]:
            continue
        args = call["args"]
        id_field, entity_raw, entity = _entity(args, fn)
        key = base = f"{call['name']}|{entity}"
        repeat = 1
        while key in out:
            repeat += 1
            key = f"{base}|{repeat}"
        out[key] = {"tool": call["name"], "entity": entity, "entity_raw": entity_raw,
                    "id_field": id_field, "requestor": call["requestor"], "pos": call["pos"],
                    "idx": call["idx"], "args": args,
                    "values": {f: _key(fn, v) for f, v in args.items()}}
    return out


# --- canonicalization and text matching ------------------------------------

def _canon_fn(canon: Any) -> Callable[[Any], Any]:
    """canon.py's rules by default (D39); a caller may pass the module or its own callable.

    The default used to be the identity, so a Verifier derived without an explicit canon compared
    raw values while the Runner compared canonical ones. There is one canonicalizer; this is it.
    """
    if canon is None:
        return canon_value
    for attr in ("canon_value", "canonicalize", "canonical", "normalize"):
        if callable(getattr(canon, attr, None)):
            return getattr(canon, attr)
    return canon if callable(canon) else canon_value


def _key(fn: Callable, value: Any) -> str:
    return canonical_json(fn(value))


def _text_of(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value).strip('"')


def _texts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _texts(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _texts(item)]
    return [_text_of(value)]


def _token_in(haystack: str, needle: str) -> bool:
    """Substring match that will not fire inside a longer word or id."""
    if not needle or not haystack:
        return False
    hay, need = haystack.lower(), needle.lower()
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


def _all_tokens_in(haystack: str, value: Any) -> bool:
    parts = [t for t in _texts(value) if t]
    return bool(parts) and all(_token_in(haystack, t) for t in parts)


def _words_of(text: str) -> list[str]:
    """The words a value could hide in, split the way the generated predicates split them.

    "$150" is one word and also the number 150, and the canonicalizer keeps the currency (D39), so
    both spellings go to it and it decides whether either is the written value.
    """
    out = []
    for word in _WORD.findall(text or ""):
        word = word.strip("./-")
        out.append(word)
        if word[:1] in CURRENCY:
            out.append(word[1:])
    return [w for w in out if w]


def _matches(haystack: str, value: Any, fn: Optional[Callable] = None) -> bool:
    """Does this text hold the value, verbatim or after canonicalization?

    "$150" and 150.0 are the same value to canon.py (D39), so a user who said one and an agent that
    wrote the other is user_stated, not agent_chosen.
    """
    parts = [t for t in _texts(value) if t]
    if not parts:
        return False
    if all(_token_in(haystack, part) for part in parts):
        return True
    if fn is None:
        return False
    words = {str(fn(word)) for word in _words_of(haystack) if word}
    return all(str(fn(part)) in words for part in parts)


def _tokens(text: str) -> list[str]:
    """The id-shaped and numeric tokens of a message: the facts an answer can state."""
    out = []
    for token in _TOKEN.findall(text or ""):
        token = token.rstrip("./-") if token.startswith(("#", "$")) else token.strip("./-#")
        if len(token) > 1 and any(c.isdigit() for c in token):
            out.append(token)
    return out


# --- provenance (D42) ------------------------------------------------------

def classify_provenance(run: Run, write_pos: int, value: Any, fn: Optional[Callable] = None):
    """Where a written value came from, with the span that evidences it (D42).

    D42 makes this an audited LLM call; this is the code half of it, and the model half is on the
    todo list, so a value only a person would recognise as quoted is still agent_chosen here.
    """
    for pos in range(0, write_pos):
        event = run.events[pos]
        if event.type == "user_turn" and _matches(_user_text(event), value, fn):
            elicited = _preceded_by_question(run, pos)
            return ("user_elicited" if elicited else "user_stated"), _ptr(run, event.idx)
    for pos in range(0, write_pos):
        event = run.events[pos]
        if event.type == "tool_result" and _matches(canonical_json(_payload(event).get("result")), value, fn):
            return "system_derived", _ptr(run, event.idx)
    return "agent_chosen", _ptr(run, run.events[write_pos].idx if write_pos < len(run.events) else None)


def _preceded_by_question(run: Run, user_pos: int) -> bool:
    """Did the agent ask something just before this user turn? Then the answer was elicited, not stated."""
    for pos in range(user_pos - 1, -1, -1):
        event = run.events[pos]
        if event.type == "user_turn":
            return False
        if event.type == "model_call" and _assistant_text(event):
            return "?" in _assistant_text(event)
    return False


def _next_user(run: Run, pos: int) -> Optional[int]:
    return next((later for later in range(pos + 1, len(run.events)) if run.events[later].type == "user_turn"), None)


# --- questions and communicate facts (D43) --------------------------------

def question_keys(run: Run, effects: dict, fn: Callable) -> dict[str, dict]:
    """The questions this Run asked, keyed so the same question is recognisable in another Run.

    Each key carries the write it belongs to, so the atom's predicate can bind the written value to
    the user's own reply in the Candidate's Run (D43) instead of trusting the field name.
    """
    out: dict[str, dict] = {}
    for effect in effects.values():
        for field, value in effect["args"].items():
            kind, span = classify_provenance(run, effect["pos"], value, fn)
            if kind == "user_elicited":
                out.setdefault(f"field:{field}", {"span": span, "tool": effect["tool"], "field": field})
    for pos, event in enumerate(run.events):
        reply_pos = _next_user(run, pos) if "?" in _assistant_text(event) else None
        if reply_pos is None:
            continue
        answer = _user_text(run.events[reply_pos]).strip().lower()
        if not any(answer.startswith(word) for word in AFFIRMATIONS):
            continue
        for effect in effects.values():
            if effect["pos"] > reply_pos:
                out.setdefault(f"confirm:{effect['tool']}",
                               {"span": _ptr(run, run.events[reply_pos].idx), "tool": effect["tool"],
                                "field": None})
    return out


def communicate_values(run: Run, fn: Callable) -> dict[str, dict]:
    """Facts read from the world that the Run's final answer states back to the user."""
    out: dict[str, dict] = {}
    final_pos, text = None, ""
    for pos, event in enumerate(run.events):
        if _assistant_text(event) and not _reply(event).get("tool_calls"):
            final_pos, text = pos, _assistant_text(event)
    if final_pos is None:
        return out
    results = [(pos, canonical_json(_payload(e).get("result")))
               for pos, e in enumerate(run.events) if e.type == "tool_result" and pos < final_pos]
    for token in _tokens(text):
        for pos, blob in results:
            if _token_in(blob, token):
                out.setdefault(_key(fn, token), {"text": token, "span": _ptr(run, run.events[pos].idx)})
                break
    return out


# --- derivation ------------------------------------------------------------

def _pack(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def atom_payload(atom: Atom) -> dict:
    """The structured target an Atom carries; an atom stored before `target` existed is decoded."""
    if atom.target:
        return dict(atom.target)
    try:
        loaded = json.loads(atom.predicate_src or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
_HARD_WRAPPER = """{helpers}
{spans}
{rule}

_rule = check
_WRITE_TOOLS = {write_tools}
_READ_TOOLS = {read_tools}


def check():
    seen = {{}}
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


# The names policy.py's _RUNNER_SRC certifies at build time, so a rule that passed its positive and
# negative test evaluates here too instead of raising and being read as "the constraint held".
_HARD_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
    "max", "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "KeyError", "TypeError", "ValueError")}


def _start_state(run: Run) -> dict:
    """The Starting state the Run's stop event carries, which is what a Hard rule reads."""
    for event in run.events:
        if event.type == "stop" and isinstance(_payload(event).get("start_state"), dict):
            return _payload(event)["start_state"]
    return {}


def _transcript(run: Run) -> list[dict]:
    """The Run as a policy predicate reads it: role, content and tool calls, in event order."""
    out: list[dict] = []
    for event in run.events:
        if event.type == "model_call" and _assistant_text(event):
            out.append({"role": "assistant", "content": _assistant_text(event), "tool_calls": []})
        elif event.type == "user_turn":
            out.append({"role": "user", "content": _user_text(event), "tool_calls": []})
        elif event.type == "tool_call":
            out.append({"role": "assistant", "content": None, "tool_calls": [
                {"name": str(_payload(event).get("name") or ""),
                 "arguments": _payload(event).get("args") or _payload(event).get("arguments") or {}}]})
    return out


def _hard_holds(atom: Atom, run: Run, write_tools: Optional[set[str]] = None,
                fn: Optional[Callable] = None) -> Optional[bool]:
    """Does this compiled Hard constraint hold over the Run? None when the atom is not code (D76).

    A predicate that raises is a Verifier defect and returns False, so the D79 oracle check reports
    it; returning None used to read as "the constraint held" and hid a rule that never ran.
    """
    source = atom.predicate_src
    if not source or atom.judge:
        return None
    calls = [dict(call) for call in _calls(run)]
    tools = set(write_tools or ())
    namespace: dict = {"__builtins__": _HARD_BUILTINS, "start_state": _start_state(run),
                       "transcript": _transcript(run), "calls": calls, "canon": fn or canon_value,
                       "write_calls": lambda: [c for c in calls if c["name"] in tools and not c["error"]]}
    try:
        exec(compile(source, "<hard>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        return None if check is None else bool(check())
    except Exception:
        return False


def _helpers_src() -> str:
    """policy.py's transcript helpers, so a compiled rule finds them at Verdict time."""
    from harness.builder import policy  # builder to builder; the Runner imports neither (D89)

    return policy.HELPERS_SRC


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^A-Za-z0-9]+", text) if w]


def _write_fields(payload: dict) -> dict:
    """The arguments that name the write this atom is about: its entity, and its own field."""
    fields: dict = {}
    if payload.get("id_field"):
        fields[payload["id_field"]] = payload.get("entity_raw")
    if payload.get("kind") == "write_value":
        fields[payload["field"]] = payload.get("raw")
    return fields


def _predicate(payload: dict) -> str:
    """One atom target as source in the Runner's atom vocabulary (verdict.py evaluates this).

    The predicate has to express the same check as the target: the D79 suite scores Runs off the
    target and the Verdict scores them off this source, so anything the target says and the source
    leaves out is a hole no gate can see.
    """
    kind = payload.get("kind")
    if kind in ("write", "write_value"):
        fields = _write_fields(payload)
        return f"wrote({payload['tool']!r}, **{fields!r})" if fields else f"wrote({payload['tool']!r})"
    if kind == "entity_count":
        return f"writes_count() <= {int(payload['count'])}"
    if kind == "question":
        head, _, rest = str(payload.get("key") or "").partition(":")
        if head == "confirm":
            return _CONFIRM_WRAPPER.format(words=repr(tuple(AFFIRMATIONS)), tool=repr(rest))
        if payload.get("tool") and payload.get("field"):
            return _FIELD_QUESTION_WRAPPER.format(spans=_SPANS_SRC, tool=repr(payload["tool"]),
                                                  field=repr(payload["field"]))
        return "asked(" + ", ".join(repr(w) for w in _words(rest)) + ")"
    if kind == "communicate":
        return f"communicated({str(payload.get('text') or payload.get('value'))!r})"
    if kind == "hard":
        rule = payload.get("predicate_src")
        if not rule or payload.get("judge"):
            return ""  # a judge atom is answered by judge.py, never by code (D76)
        return _HARD_WRAPPER.format(helpers=_helpers_src(), spans=_SPANS_SRC, rule=rule,
                                    write_tools=repr(sorted(payload.get("write_tools") or [])),
                                    read_tools=repr(sorted(payload.get("read_tools") or [])))
    return ""


def _atom(atom_id: str, kind: str, payload: dict, **fields: Any) -> Atom:
    """One atom: the structured target for the Builder, the predicate source for the Runner."""
    return Atom(id=atom_id, kind=kind, target=payload, predicate_src=_predicate(payload), **fields)


def derive_verifier(task: Any, reference_run: Any, rerun_paths: Optional[list[str]] = None, canon: Any = None, *,
                    write_tools: Optional[Iterable[str]] = None,
                    constraints: Optional[Iterable[Constraint]] = None,
                    successful_run_ids: Optional[Iterable[str]] = None,
                    verifier_version: str = "1") -> Verifier:
    """The atoms for one Task: write-set diff over the Reference and its successful re-runs (D42, D43)."""
    fn = _canon_fn(canon)
    reference = _as_run(reference_run)
    reruns = load_runs(rerun_paths or [])
    tools = resolve_write_tools([reference] + reruns, write_tools)
    good = [reference] + [r for r in reruns if _successful(r, successful_run_ids)]
    good_effects = [_effects(r, tools, fn) for r in good]
    everywhere = set.intersection(*[set(e) for e in good_effects]) if good_effects else set()
    somewhere = set().union(*[set(e) for e in good_effects]) if good_effects else set()
    reads = {c["name"] for r in [reference] + reruns for c in _calls(r)} - tools
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
                           spans=[_ptr(run, effect["idx"])],
                           description=f"{effect['tool']} writes {effect['entity'] or 'an entity'}"))
        for field in sorted(effect["values"]):
            value, canon_key = effect["args"][field], effect["values"][field]
            provenance, span = classify_provenance(run, effect["pos"], value, fn)
            agreed = all(key in e and e[key]["values"].get(field) == canon_key for e in good_effects)
            atoms.append(_atom(f"{atom_id}.{field}",
                               "required" if agreed and provenance in REQUIRED_PROVENANCE else "allowed",
                               dict(base, kind="write_value", field=field, value=canon_key, raw=value),
                               provenance=provenance, spans=[span] if span else [],
                               description=f"{effect['tool']} {field} is {_text_of(value)}"))

    if good_effects:
        cap = max(len(e) for e in good_effects)
        atoms.append(_atom("entity_count", "required", {"kind": "entity_count", "count": cap},
                           description=f"the Run makes at most {cap} write calls"))

    asked = [question_keys(r, e, fn) for r, e in zip(good, good_effects)]
    for key in sorted(set.intersection(*[set(a) for a in asked]) if asked else set()):
        seen = asked[0][key]
        atoms.append(_atom(f"q.{key}", "question",
                           {"kind": "question", "key": key, "tool": seen.get("tool"),
                            "field": seen.get("field")},
                           spans=[seen["span"]] if seen.get("span") else [],
                           description=f"the agent asks the user about {key.split(':', 1)[-1]}"))

    said = [communicate_values(r, fn) for r in good]
    for number, key in enumerate(sorted(set.intersection(*[set(s) for s in said]) if said else set())):
        fact = said[0][key]
        atoms.append(_atom(f"c{number}", "communicate",
                           {"kind": "communicate", "value": key, "text": fact["text"]},
                           provenance="system_derived", spans=[fact["span"]],
                           description=f"the final answer states {fact['text']}"))

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


def _first_with(runs: list[Run], effects: list[dict], key: str):
    """The first Run that has this write effect; the Reference comes first, so its spans win."""
    for run, effect in zip(runs, effects):
        if key in effect:
            return run, effect[key]
    raise KeyError(key)


# --- scoring a Run against the atoms (Builder side; verdict.py has its own, D91) ---

def verifier_write_tools(verifier: Verifier) -> set[str]:
    return {p["tool"] for p in map(atom_payload, verifier.atoms) if p.get("tool")}


def scored_write_tools(verifier: Verifier, run: Run, write_tools: Optional[Iterable[str]] = None) -> set[str]:
    """The tools whose calls count as writes when scoring this Run.

    A caller with the mined write tools (mine.py's classification) is the authority and is taken as
    given, so this scorer and the Verdict count the same calls. With no caller set, the Verifier's
    own atoms are not enough on their own: a Run that called a write tool the Reference never used
    would be scored as if it had written nothing, so the Run's own event marking is read too.
    """
    if write_tools:
        return set(write_tools)
    return resolve_write_tools([run], verifier_write_tools(verifier))


def check_run(verifier: Verifier, run: Any, canon: Any = None, *,
              write_tools: Optional[Iterable[str]] = None) -> tuple[bool, Optional[str]]:
    """Does this Run satisfy the atoms? Used by the D79 checks below, never by the Runner."""
    fn = _canon_fn(canon)
    run = _as_run(run)
    tools = scored_write_tools(verifier, run, write_tools)
    effects = _effects(run, tools, fn)
    present = {(e["tool"], e["entity"]) for e in effects.values()}
    values = {(e["tool"], e["entity"], f, v) for e in effects.values() for f, v in e["values"].items()}
    asked, said = set(question_keys(run, effects, fn)), set(communicate_values(run, fn))
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        kind = payload.get("kind")
        target = (payload.get("tool"), payload.get("entity"))
        if atom.kind == "forbidden" and kind == "write" and target in present:
            return False, atom.id
        if atom.kind == "hard":
            # A Hard constraint is a gate; the D79 checks have to see it fail on a Run that breaks it.
            if _hard_holds(atom, run, tools, fn) is False:
                return False, atom.id
            continue
        if atom.kind != "required" and kind not in ("question", "communicate"):
            continue
        if kind == "write" and target not in present:
            return False, atom.id
        if kind == "write_value" and target + (payload.get("field"), payload.get("value")) not in values:
            return False, atom.id
        # The cap counts write calls, which is what the Runner's `writes_count()` counts; counting
        # entities here let a Run that wrote the same entity twice pass the Builder and fail the Verdict.
        if kind == "entity_count" and len(effects) > payload.get("count", 0):
            return False, atom.id
        if kind == "question" and payload.get("key") not in asked:
            return False, atom.id
        if kind == "communicate" and payload.get("value") not in said:
            return False, atom.id
    return True, None


# --- D79 validation --------------------------------------------------------

# The stage each D79 check reports under, and the name validate.py's `verifier_gate` wants back.
D79_STAGES = {
    "verifier_provenance_spans": "provenance_spans",
    "verifier_oracle": "oracle_passes",
    "verifier_empty_run": "empty_fails",
    "verifier_wrong_run": "plausible_wrong_fails",
    "verifier_alt_path": "second_path_passes",
    "verifier_loophole": "loophole_probe_fails",
    "verifier_leak": "leak_check_clean",
    "verifier_mutation": "mutation_flips",
}


def d79_results(gates: Iterable[GateResult]) -> dict[str, bool]:
    """The suite's answer in the shape validate.py's `verifier_gate` reads: one bool per D79 check."""
    seen = {gate.stage: bool(gate.passed) for gate in gates}
    return {name: seen.get(stage, False) for stage, name in D79_STAGES.items()}


def validate_verifier(verifier: Verifier, reference_run: Any, empty_run: Any = None, wrong_run: Any = None,
                      alt_path_run: Any = None, intent_text: Optional[str] = None,
                      user_rules: Optional[UserRules] = None, *, canon: Any = None,
                      write_tools: Optional[Iterable[str]] = None, model: Any = None,
                      run_probe: Optional[Callable] = None,
                      seed_runs: Optional[Iterable[Any]] = None) -> list[GateResult]:
    """The eight D79 checks as GateResults. A check whose input is missing fails as "not run".

    Nothing here executes a Run (D91), so the wrong Run, the second path and the loophole probe are
    the caller's to supply; a check the caller left out is reported as not run, the way validate.py
    counts it, and never as a pass. The empty Run needs no input and is synthesized.
    """
    reference = _as_run(reference_run)
    runs = {r.trace_id or r.run_id: r for r in [_as_run(s) for s in (seed_runs or [])] + [reference]}

    def score(atoms_of: Verifier, run):
        return check_run(atoms_of, run, canon, write_tools=write_tools)

    def scored(run):
        return score(verifier, run)

    return [
        _spans_gate(verifier, runs, _canon_fn(canon)),
        _run_gate("verifier_oracle", scored, reference, expect_pass=True),
        _run_gate("verifier_empty_run", scored,
                  empty_run if empty_run is not None else _empty_run(reference), expect_pass=False),
        _run_gate("verifier_wrong_run", scored, wrong_run, expect_pass=False),
        _run_gate("verifier_alt_path", scored, alt_path_run, expect_pass=True),
        loophole_probe(verifier, model, run_probe=run_probe, canon=canon, write_tools=write_tools),
        _leak_gate(verifier, reference, intent_text, user_rules),
        _mutation_gate(verifier, reference, score, _canon_fn(canon)),
    ]


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


def _run_gate(stage: str, scored: Callable, run: Any, *, expect_pass: bool) -> GateResult:
    """Checks 2 to 5: one Run, one expected outcome. No Run, no evidence, so no pass."""
    if run is None:
        return GateResult(stage=stage, passed=False, metrics={"skipped": True},
                          failures=["not run: no Run was supplied for this check"])
    passed, failing_atom = scored(run)
    want = "pass" if expect_pass else "fail"
    failures = [] if passed is expect_pass else [f"expected {want}, got {'pass' if passed else 'fail'}"]
    return GateResult(stage=stage, passed=passed is expect_pass, failures=failures,
                      metrics={"run_passed": passed, "failing_atom": failing_atom})


def _mutation_gate(verifier: Verifier, reference: Run, score: Callable, fn: Callable) -> GateResult:
    """Check 8: change what an atom demands and the Reference must stop passing.

    An atom nothing can fail is an atom that is not being checked, which is how a Hard constraint
    that never gets evaluated, or a value the scorer never compares, hides in a passing suite.
    """
    failures, mutated = [], 0
    for atom in verifier.atoms:
        mutant = _mutant(atom, fn)
        if mutant is None:
            continue
        mutated += 1
        changed = Verifier(task_id=verifier.task_id, verifier_version=verifier.verifier_version,
                           seed_run_ids=verifier.seed_run_ids,
                           atoms=[mutant if a.id == atom.id else a for a in verifier.atoms])
        if score(changed, reference)[0]:
            failures.append(f"{atom.id}: the Reference still passes when this atom is changed")
    return GateResult(stage="verifier_mutation", passed=not failures,
                      metrics={"atoms_mutated": mutated}, failures=failures)


def _mutant(atom: Atom, fn: Callable) -> Optional[Atom]:
    """The same atom asking for something the Reference did not do, or None when it cannot be mutated."""
    payload = atom_payload(atom)
    kind = payload.get("kind")
    if atom.kind not in ("required", "question", "communicate", "hard"):
        return None
    if kind == "write":
        return _atom(atom.id, atom.kind, dict(payload, entity=_key(fn, _MUTANT), entity_raw=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "write_value":
        return _atom(atom.id, atom.kind, dict(payload, value=_key(fn, _MUTANT), raw=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "entity_count":
        return _atom(atom.id, atom.kind, dict(payload, count=max(int(payload.get("count", 0)) - 1, 0)),
                     description=atom.description)
    if kind == "question":
        return _atom(atom.id, atom.kind, dict(payload, key=f"{payload.get('key')}.{_MUTANT}",
                                              field=_MUTANT), description=atom.description)
    if kind == "communicate":
        return _atom(atom.id, atom.kind, dict(payload, value=_key(fn, _MUTANT), text=_MUTANT),
                     provenance=atom.provenance, spans=atom.spans, description=atom.description)
    if kind == "hard" and not atom.judge and payload.get("predicate_src"):
        return _atom(atom.id, atom.kind, dict(payload, predicate_src=_NEVER_HOLDS),
                     description=atom.description)
    return None


def _leak_gate(verifier: Verifier, reference: Run, intent_text: Optional[str],
               user_rules: Optional[UserRules]) -> GateResult:
    """Check 7: constants only the Verifier should know, found in the Intent or the Simulated user rules."""
    said_by_user = " ".join(_user_text(e) for e in reference.events if e.type == "user_turn")
    secrets = set()
    for atom in verifier.atoms:
        payload = atom_payload(atom)
        if atom.provenance not in ("system_derived", "agent_chosen"):
            continue
        # Numbers leak as readily as strings: an amount read off a tool result and repeated in the
        # Intent is the constant D79 check 7 is looking for, whether the trace stored it as 150.0 or "150.0".
        value = payload["text"] if payload.get("text") is not None else payload.get("raw")
        for text in _texts(value) if value is not None else []:
            if len(text) > 1 and text not in ("true", "false", "null") and not _token_in(said_by_user, text):
                secrets.add(text)
    blobs = {"intent": intent_text or ""}
    if user_rules is not None:
        blobs["user_rules"] = canonical_json(as_dict(user_rules))
    failures = [f"{where} leaks {text}" for text in sorted(secrets)
                for where, blob in blobs.items() if _token_in(blob, text)]
    return GateResult(stage="verifier_leak", passed=not failures,
                      metrics={"constants": len(secrets)}, failures=failures)


def loophole_probe(verifier: Verifier, model: Any, *, run_probe: Optional[Callable] = None, canon: Any = None,
                   write_tools: Optional[Iterable[str]] = None) -> GateResult:
    """Check 6: an agent told to reach the End state while skipping the policy step must score fail.

    verifier.py runs nothing (D91), so the caller passes `run_probe(model, verifier) -> Run`. With no model
    or no runner the probe is skipped, and a skipped probe is not evidence that the Verifier is tight.
    """
    if model is None or run_probe is None:
        missing = "no model" if model is None else "no run_probe"
        return GateResult(stage="verifier_loophole", passed=False, metrics={"skipped": True},
                          failures=[f"not run: {missing}, so the Verifier is not known to be tight"])
    passed, failing_atom = check_run(verifier, run_probe(model, verifier), canon, write_tools=write_tools)
    return GateResult(stage="verifier_loophole", passed=not passed,
                      metrics={"probe_passed": passed, "failing_atom": failing_atom},
                      failures=["the loophole probe reached the End state and scored pass"] if passed else [])


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
