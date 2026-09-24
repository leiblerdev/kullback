"""One scorer for a Verifier atom: the structured target as the only meaning (G3).

Both the Verdict and the gates call this module. The gates keep their behaviour through it;
`gates/verifier_suite.py` imports the scoring core from here. The Runner never imports the gates.

A Hard rule that raises is a Verifier defect, never a Candidate failure: `hard_holds` answers
None and `check_run` does not fail the Run on it. The Verdict reads the same None as not
verdicted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.runner.canon import CanonRules, canon_value
from kullback.runner.confinement import SAFE_BUILTINS, confine
from kullback.runner.records import Atom, Event, RawPtr, Run, Verifier, canonical_json, load_run_jsonl

_TOKEN = re.compile(r"[#$]?[A-Za-z0-9][A-Za-z0-9_./#-]*")
_WORD = re.compile(r"[A-Za-z0-9#$€£¥._/-]+")
CURRENCY = "".join(CanonRules().currency_symbols)
AFFIRMATIONS = ("yes", "yeah", "yep", "sure", "please do", "go ahead", "confirm", "correct", "ok", "okay")
JUDGE_MUST_HOLD = frozenset({"required", "question", "communicate", "hard"})
# The payload kinds check_run scores a required atom on; a Hard atom is scored on its predicate_src (F51).
SCORED_KINDS = ("write", "write_value", "entity_count", "question", "communicate")


def is_judge_must_hold(atom: Atom) -> bool:
    return atom.judge and atom.kind in JUDGE_MUST_HOLD


def judge_must_hold_atoms(verifier: Verifier) -> list[Atom]:
    return [atom for atom in verifier.atoms if is_judge_must_hold(atom)]


def code_atoms(verifier: Verifier) -> list[Atom]:
    return [atom for atom in verifier.atoms if not atom.judge]


def first_refusal(verifier: Verifier, effects: dict,
                  write_tools: Optional[Iterable[str]],
                  resolved: Optional[Iterable[str]] = None) -> Optional[str]:
    candidate = resolved if (resolved is not None and judge_write_tools(verifier)) else write_tools
    extra = _extra_write(verifier, effects, candidate)
    if extra is not None:
        return extra
    refused = judge_must_hold_atoms(verifier)
    if refused:
        return refused[0].id
    return None


def load_run(path: Any) -> Run:
    """Read one Run from a JSONL of events (header or footer lines included) or from a whole-Run JSON."""
    return load_run_jsonl(path)


def as_run(obj: Any) -> Run:
    return obj if isinstance(obj, Run) else load_run(obj)


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


class CanonMissing(TypeError):
    """A scorer was asked to canonicalise without the Environment's rules (D284)."""


def _caller() -> str:
    """The first frame outside this module and the suite that re-exports it: the caller to name."""
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.endswith(("runner/target.py", "gates/verifier_suite.py")):
        frame = frame.f_back
    if frame is None:
        return "unknown caller"
    return f"{frame.f_code.co_name} ({Path(frame.f_code.co_filename).name}:{frame.f_lineno})"


def canon_fn(canon: Any) -> Callable[[Any], Any]:
    """The one canonicaliser of an Environment, bound from its CanonRules (D39, D284).

    The rules are built once per Environment (`canon.load_rules`, `canon.rules_of`) and passed to
    every scorer. A callable already bound from them passes through. Anything else, None or the
    raw dict of the rules file included, raises and names the caller: a silent fallback to the
    module defaults used to score the status with other rules than the suite that derived the atom.
    """
    if isinstance(canon, CanonRules):
        return lambda value: canon_value(value, rules=canon)
    if callable(canon):
        return canon
    raise CanonMissing(f"{_caller()} scored without the Environment's CanonRules "
                       f"(got {type(canon).__name__}); load them once with canon.load_rules or canon.rules_of")


def _key(fn: Callable, value: Any) -> str:
    return canonical_json(fn(value))


def text_of(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value).strip('"')


def _texts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _texts(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _texts(item)]
    return [text_of(value)]


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


def _words_of(text: str) -> list[str]:
    """The words a value could hide in, split the way the generated predicates split them."""
    out = []
    for word in _WORD.findall(text or ""):
        word = word.strip("./-")
        out.append(word)
        if word[:1] in CURRENCY:
            out.append(word[1:])
    return [w for w in out if w]


def _matches(haystack: str, value: Any, fn: Optional[Callable] = None) -> bool:
    """Does this text hold the value, verbatim or after canonicalization?"""
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


def _entity(args: dict, fn: Callable) -> tuple[str, Any, str]:
    """The id a write acts on: its field, its raw value and its canonical form.

    A scalar id wins over a list of ids, so a write carrying both one id and a list of ids
    keys on the scalar, whose order two equally good Runs cannot differ on.
    """
    named = [f for f in sorted(args) if f in ("id", "ids") or f.endswith(("_id", "_ids"))]
    scalar = [f for f in named if not isinstance(args[f], (list, tuple, dict))]
    field = next(iter(scalar + named), "")
    return (field, args[field], text_of(fn(args[field]))) if field else ("", None, "")


def ptr(run: Run, idx: Optional[int]) -> RawPtr:
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


def run_calls(run: Run) -> list[dict]:
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


def write_effects(run: Run, write_tools: set[str], fn: Callable) -> dict[str, dict]:
    """The Run's write set: one entry per write call that succeeded, keyed by tool, entity and repeat.

    A call whose result carried an error changed nothing (D67), so it is not an effect and never
    becomes an atom the next Run has to reproduce.
    """
    out: dict[str, dict] = {}
    for call in run_calls(run):
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


def classify_provenance(run: Run, write_pos: int, value: Any, fn: Optional[Callable] = None):
    """Where a written value came from, with the span that evidences it (D42)."""
    for pos in range(0, write_pos):
        event = run.events[pos]
        if event.type == "user_turn" and _matches(_user_text(event), value, fn):
            elicited = _preceded_by_question(run, pos)
            return ("user_elicited" if elicited else "user_stated"), ptr(run, event.idx)
    for pos in range(0, write_pos):
        event = run.events[pos]
        if event.type == "tool_result" and _matches(canonical_json(_payload(event).get("result")), value, fn):
            return "system_derived", ptr(run, event.idx)
    return "agent_chosen", ptr(run, run.events[write_pos].idx if write_pos < len(run.events) else None)


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


def question_keys(run: Run, effects: dict, fn: Callable) -> dict[str, dict]:
    """The questions this Run asked, keyed so the same question is recognisable in another Run."""
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
                               {"span": ptr(run, run.events[reply_pos].idx), "tool": effect["tool"],
                                "field": None})
    return out


def communicate_values(run: Run, fn: Callable) -> dict[str, dict]:
    """Facts read from the world that the Run's answers state back to the user.

    Every answer turn counts, not the closing one alone. Only a turn that carried no tool call
    counts as an answer: a fact stated in the same turn as a call is not a stated fact.
    """
    out: dict[str, dict] = {}
    answers = [(pos, _assistant_text(e)) for pos, e in enumerate(run.events)
               if _assistant_text(e) and not _reply(e).get("tool_calls")]
    results = [(pos, canonical_json(_payload(e).get("result")))
               for pos, e in enumerate(run.events) if e.type == "tool_result"]
    for answer_pos, text in answers:
        for token in _tokens(text):
            for pos, blob in results:
                if pos < answer_pos and _token_in(blob, token):
                    out.setdefault(_key(fn, token), {"text": token, "span": ptr(run, run.events[pos].idx)})
                    break
    return out


def leaves(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every scalar under a result or a row, with the dotted field path it sits under (list positions dropped)."""
    if isinstance(value, dict):
        return [leaf for name, item in value.items() for leaf in leaves(item, f"{path}.{name}" if path else str(name))]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in leaves(item, path)]
    return [(path, value)] if path and value is not None and value != "" else []


def whole_field(result: Any, key: str, fn: Callable) -> Optional[str]:
    """The field whose whole value is this canonical key, or None when the key is only part of a value.

    A fact is keyed by meaning (D285): the column it came from plus the value. A token that is a
    piece of a longer value (a year or a day out of a date, the digits of an id with a prefix) names
    no column and is a fragment, never a fact of its own.
    """
    return next((path for path, value in leaves(result) if _key(fn, value) == key), None)


def fact_source(run: Run, span: Any, key: str, fn: Callable) -> tuple[Optional[str], Optional[str]]:
    """The tool and the field a stated fact was read from: the result its span names, and its call."""
    idx = getattr(span, "msg_index", None)
    result = next((e for e in run.events if e.idx == idx and e.type == "tool_result"), None)
    if result is None:
        return None, None
    call = next((e for e in reversed(run.events) if e.type == "tool_call" and e.idx < result.idx
                 and _payload(e).get("id") in (None, _payload(result).get("id"))), None)
    tool = str(_payload(call).get("name") or "") or None if call is not None else None
    return tool, whole_field(_payload(result).get("result"), key, fn)


def field_of_value(run: Run, key: str, fn: Callable) -> Optional[str]:
    """The first field of any result of the Run whose whole value is this canonical key."""
    for event in run.events:
        if event.type == "tool_result":
            field = whole_field(_payload(event).get("result"), key, fn)
            if field:
                return field
    return None


def atom_payload(atom: Atom) -> dict:
    """The structured target an Atom carries; an atom stored before `target` existed is decoded."""
    if atom.target:
        return dict(atom.target)
    try:
        import json as _json

        loaded = _json.loads(atom.predicate_src or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def names_no_row(payload: dict) -> bool:
    """Is this write atom about the tool rather than about one row?"""
    return not (payload.get("entity") or payload.get("entity_raw") or payload.get("id_field"))


def verifier_write_tools(verifier: Any) -> set[str]:
    return {p["tool"] for p in map(atom_payload, code_atoms(verifier)) if p.get("tool")}


def judge_write_tools(verifier: Any) -> set[str]:
    return {p["tool"] for p in map(atom_payload, [a for a in verifier.atoms if a.judge])
            if p.get("kind") in ("write", "write_value") and isinstance(p.get("tool"), str)
            and p.get("tool")}


def scored_write_tools(verifier: Any, run: Run, write_tools: Optional[Iterable[str]] = None) -> set[str]:
    """The tools whose calls count as writes when scoring this Run."""
    if write_tools:
        return set(write_tools)
    return resolve_write_tools([run], verifier_write_tools(verifier) | judge_write_tools(verifier))


def start_state(run: Run) -> dict:
    """The Starting state the Run's stop event carries, which is what a Hard rule reads."""
    for event in run.events:
        if event.type == "stop" and isinstance(_payload(event).get("start_state"), dict):
            return _payload(event)["start_state"]
    return {}


def _result_turn(event: Event) -> dict:
    """One tool result as a turn: what the Run was handed, under `result`, with no content of its own."""
    return {"role": "tool", "content": None, "result": _payload(event).get("result"), "tool_calls": []}


def _transcript(run: Run) -> list[dict]:
    """The Run as a policy predicate reads it: role, content, results and tool calls, in event order."""
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
        elif event.type == "tool_result":
            out.append(_result_turn(event))
    return out


def hard_holds(atom: Atom, run: Run, write_tools: Optional[set[str]] = None,
               fn: Optional[Callable] = None) -> Optional[bool]:
    """Does this compiled Hard constraint hold over the Run? None when it judged nothing.

    None is "no answer": the atom is not code (a judge atom), the rule saw no call at all, the
    source is refused by the gate, or the predicate raised. A predicate that raises is a Verifier
    defect, never a Candidate failure: it answers None here and `check_run` does not fail the Run
    on it, the same way the Verdict leaves the Run not verdicted.
    """
    source = atom.predicate_src
    if not source or atom.judge:
        return None
    if confine(source):
        return None
    calls = [dict(call) for call in run_calls(run)]
    tools = set(write_tools or ())
    namespace: dict = {"__builtins__": dict(SAFE_BUILTINS), "start_state": start_state(run),
                       "transcript": _transcript(run), "calls": calls, "canon": fn or canon_value,
                       "write_calls": lambda: [c for c in calls if c["name"] in tools and not c["error"]]}
    try:
        exec(compile(source, "<hard>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        if check is None:
            return None
        held = bool(check())
        judged = namespace.get("judged")
        return None if held and callable(judged) and not judged() else held
    except Exception:
        return None


def hard_defect(atom: Atom, run: Run, write_tools: Optional[set[str]] = None,
                fn: Optional[Callable] = None) -> bool:
    """Is this Hard atom defective on this Run: refused by the gate or raising?

    A defective rule is a Verifier defect, never a Candidate failure. check_run passes the
    Run over it; the D79 oracle fails the Verifier on it, so a policy rule that cannot run
    cannot clear admission. A rule that judged no call at all is not defective, it simply
    saw nothing to judge.
    """
    source = atom.predicate_src
    if not source or atom.judge:
        return False
    if confine(source):
        return True
    calls = [dict(call) for call in run_calls(run)]
    tools = set(write_tools or ())
    namespace: dict = {"__builtins__": dict(SAFE_BUILTINS), "start_state": start_state(run),
                       "transcript": _transcript(run), "calls": calls, "canon": fn or canon_value,
                       "write_calls": lambda: [c for c in calls if c["name"] in tools and not c["error"]]}
    try:
        exec(compile(source, "<hard>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        if check is None:
            return False
        check()
    except Exception:
        return True
    return False


def _extra_write(verifier: Any, effects: dict, write_tools: Optional[Iterable[str]]) -> Optional[str]:
    """The first write no atom asked for or allowed, named the way the Verdict names it."""
    if not write_tools:
        return None
    covered = set()
    whole_tool = set()
    for atom in code_atoms(verifier):
        payload = atom_payload(atom)
        if atom.kind != "forbidden" and payload.get("kind") in ("write", "write_value"):
            covered.add((payload.get("tool"), payload.get("entity")))
            if names_no_row(payload):
                whole_tool.add(payload.get("tool"))
    for effect in effects.values():
        if effect["tool"] not in whole_tool and (effect["tool"], effect["entity"]) not in covered:
            return f"extra_write:{effect['tool']}"
    return None


def atom_holds(atom: Atom, run: Run, fn: Callable, write_tools: Optional[set[str]] = None, *,
               effects: Optional[dict] = None, asked: Optional[set] = None,
               said: Optional[set] = None) -> Optional[bool]:
    """One non-Hard atom's structured target over the Run: True holds, False fails, None skips.

    Hard and judge atoms are not read here; the caller keeps them on their own path. An `allowed`
    atom that is neither a question nor a communicate carries no check and answers None.
    """
    payload = atom_payload(atom)
    kind = payload.get("kind")
    if atom.kind == "hard" or atom.judge:
        return None
    if (atom.kind != "required" and kind not in ("question", "communicate")
            and not (atom.kind == "forbidden" and kind == "write")):
        return None
    if effects is None or asked is None or said is None:
        tools = scored_write_tools(_verifier_of(atom), run, write_tools)
        effects = write_effects(run, tools, fn)
        asked, said = set(question_keys(run, effects, fn)), set(communicate_values(run, fn))
        return _one_atom(atom, payload, kind, effects, asked, said)
    return _one_atom(atom, payload, kind, effects, asked, said)


def _verifier_of(atom: Atom) -> Any:
    """A one-atom holder so the fallback path scores under this atom's own write tools."""

    class _One:
        atoms = [atom]

    return _One()


def _effect_sets(effects: dict) -> tuple[set, set, set]:
    present = {(e["tool"], e["entity"]) for e in effects.values()}
    wrote_with = {e["tool"] for e in effects.values()}
    values = {(e["tool"], e["entity"], f, v) for e in effects.values() for f, v in e["values"].items()}
    return present, wrote_with, values


def _one_atom(atom: Atom, payload: dict, kind: Any, effects: dict, asked: set, said: set) -> Optional[bool]:
    present, wrote_with, values = _effect_sets(effects)
    target = (payload.get("tool"), payload.get("entity"))
    if atom.kind == "forbidden" and kind == "write":
        return payload.get("tool") in wrote_with if names_no_row(payload) else target in present
    if kind == "write" and (payload.get("tool") not in wrote_with if names_no_row(payload)
                            else target not in present):
        return False
    if kind == "write_value" and target + (payload.get("field"), payload.get("value")) not in values:
        return False
    if kind == "entity_count" and len(effects) > payload.get("count", 0):
        return False
    if kind == "question" and payload.get("key") not in asked:
        return False
    if kind == "communicate" and payload.get("value") not in said:
        return False
    return True


def evaluated_atoms(atoms: Iterable[Atom]) -> list[Atom]:
    """The atoms check_run evaluates on a Run, in order; every other atom is skipped there.

    A code atom is evaluated when it is Hard, required, a forbidden write, or a question or
    communicate atom of any kind. Judge atoms and the rest (an allowed count, a forbidden
    value) carry no check a Run can fail here, so nothing may name them as passed (F37).
    """
    kept = []
    for atom in atoms:
        kind = atom_payload(atom).get("kind")
        if atom.judge:
            continue
        if (atom.kind in ("hard", "required") or (atom.kind == "forbidden" and kind == "write")
                or kind in ("question", "communicate")):
            kept.append(atom)
    return kept


def unscored(atom: Atom) -> str:
    """The failure check_run names for a required atom whose payload kind it does not score."""
    return f"atom {atom.id}: kind {atom_payload(atom).get('kind')} is not scored"


def _write_missing(kind: Any, payload: dict, present: set, wrote_with: set, values: set) -> bool:
    """A write or write_value atom whose row, tool or field value no write of the Run shows."""
    target = (payload.get("tool"), payload.get("entity"))
    if kind == "write":
        return (payload.get("tool") not in wrote_with if names_no_row(payload)
                else target not in present)
    if kind == "write_value":
        return target + (payload.get("field"), payload.get("value")) not in values
    return False


def check_run(verifier: Any, run: Any, canon: Any = None, *,
              write_tools: Optional[Iterable[str]] = None) -> tuple[bool, Optional[str]]:
    """Does this Run satisfy the atoms? Called by the D79 checks and, through them, the gates."""
    fn = canon_fn(canon)
    run = as_run(run)
    tools = scored_write_tools(verifier, run, write_tools)
    effects = write_effects(run, tools, fn)
    present = {(e["tool"], e["entity"]) for e in effects.values()}
    wrote_with = {e["tool"] for e in effects.values()}
    values = {(e["tool"], e["entity"], f, v) for e in effects.values() for f, v in e["values"].items()}
    asked, said = set(question_keys(run, effects, fn)), set(communicate_values(run, fn))
    for atom in evaluated_atoms(verifier.atoms):
        payload = atom_payload(atom)
        kind = payload.get("kind")
        if atom.kind == "forbidden" and kind == "write":
            if atom_holds(atom, run, fn, tools, effects=effects, asked=asked, said=said):
                return False, atom.id
            continue
        if atom.kind == "hard":
            held = hard_holds(atom, run, tools, fn)
            if held is False:
                return False, atom.id
            continue
        if atom.kind == "required" and kind not in SCORED_KINDS:
            # A required atom nothing here can score is a failure, never a vacuous pass (F51).
            return False, unscored(atom)
        if _write_missing(kind, payload, present, wrote_with, values):
            return False, atom.id
        if kind == "entity_count" and len(effects) > payload.get("count", 0):
            return False, atom.id
        if kind == "question" and payload.get("key") not in asked:
            return False, atom.id
        if kind == "communicate" and payload.get("value") not in said:
            return False, atom.id
    extra = first_refusal(verifier, effects, write_tools, tools)
    return (True, None) if extra is None else (False, extra)
