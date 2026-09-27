"""Write one Task's Verifier from the Task only, must side (vf-write).

The writer is one model session whose only tools read the Starting state. It sees
the request in the user's own words, the policy rules, the tool schemas and reads
of the Starting state. No tool, prompt line or file shows another episode of the
Task. It writes must checks (required, allowed, question, communicate), each with
a because: a quote from the request or a policy rule named as the policy names it.

Grounding is code, no model: a check is refused whose quote is not in the request
(same tokeniser as builder/intent.py, vendored below so this module never imports
the Builder), whose rule is not in the policy, or whose value no read returned
and the request does not hold. Refusals go back to the session for a rewrite, up
to 3 attempts; the attempt with the most kept checks wins.

The Verifier is built with make_atom from the D79 suite, the same call the
derivation uses, so the existing scorer rules on it unchanged. Several acceptable
end states are allowed checks, never one required literal. The must-not side is
another module's work; this one writes required, allowed, question, communicate.

write_verifier(task, intent, policy, tools, starting_state, model) shapes:
- task: a Task record, a dict with an id, or the id itself.
- intent: an Intent record (.text), a Task (.intent), a dict with text, or the
  user's words as a string. On the smoke builds there are no written Intents, so
  the live check passes exam/spoken/<task>.txt, the user turns of the Task.
- policy: Constraints, dicts with id/text, objects with .id/.text, or a plain
- tools: entries naming the Environment's tools. Each entry is a ToolSig record,
  a dict, an object with attributes, or a bare callable. Read from each: name,
  kind (default read), description, args_schema, and the read body under
  body/fn/func/impl (or the entry itself when callable). The session may call
  only kind read entries that carry a body; every named tool is described in the
  prompt so the checks can name writes too.
- starting_state: the world as {table: {id: row}}, or an object with a .shared
  mapping (a StateView-like; its .overlay mapping is laid over .shared). It is
  copied before any body runs. Bodies run the way the Runner runs them: a first
  parameter named state, db, world or env takes a small view over the copy,
  otherwise the body takes the call arguments only; a body that raises leaves
  the copy as it found it, and the session reads the failure as text.
- model: anything with .query(messages, tools) giving .content and .tool_calls.

Extra keyword-only knobs: canon (the Environment's CanonRules, required, so value
keys score under the same rules as the suite), max_attempts (3), max_turns per
attempt (8), stats (a dict that is filled with attempts, written, refused,
kept_checks, atoms, calls, reads, input_tokens, output_tokens when given).
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from typing import Any, Optional

from kullback.gates.verifier_suite import (
    HELPERS_SRC,
    _key,
    _tokens as _score_tokens,
    canon_fn,
    make_atom,
    text_of,
)
from kullback.runner.records import Atom, Verifier, plain

# The request tokeniser, a copy of kullback/builder/intent.py's TOKEN_RE,
# normalise and _tokens. Copied rather than imported so the Examiner never reads
# what the Builder compiled; tests/examiner/test_intent_derive.py holds the two
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:(?<=[0-9])[.,:][0-9]+)*")
_APOSTROPHE_RE = re.compile("[" + chr(39) + chr(8217) + chr(700) + "]")


def _undouble(stem: str) -> str:
    doubled = len(stem) > 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou"
    return stem[:-1] if doubled else stem


def _strip_suffix(word: str) -> str:
    for suffix, stem in (("ies", word[:-3] + "y"), ("es", word[:-2]), ("s", word[:-1]),
                         ("ing", _undouble(word[:-3])), ("ed", _undouble(word[:-2])), ("e", word[:-1])):
        if not word.endswith(suffix) or len(word) - len(suffix) < 4:
            continue
        if suffix == "s" and word.endswith("ss"):
            continue
        return stem
    return word


def _normalise(word: str) -> str:
    if not word.isalpha():
        return word
    while True:
        stem = _strip_suffix(word)
        if stem == word:
            return word
        word = stem


def _intent_tokens(text: Any) -> list[str]:
    lowered = _APOSTROPHE_RE.sub("", str(text or "").lower())
    return [_normalise(token) for token in _TOKEN_RE.findall(lowered)]


MUST_KINDS = ("required", "allowed", "question", "communicate")
MAX_ATTEMPTS = 3
MAX_TURNS = 8
READ_ARG_NAMES = ("state", "db", "world", "env")
RESULT_CHARS = 4000


def tier_of(atom: Any) -> str:
    """The tier of one check from its kind only (DESIGN.md Tiers).

    Required end-state writes are critical; forbidden checks are sanity;
    questions, message facts and allowed checks are important. Hard atoms are
    generated rules, prohibitions and claims, the gate family, so sanity.
    """
    kind = atom.get("kind") if isinstance(atom, dict) else getattr(atom, "kind", None)
    if kind == "required":
        return "critical"
    if kind in ("forbidden", "hard"):
        return "sanity"
    return "important"


def _task_id(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return str(task.get("id", task))
    return str(getattr(task, "id", task))


def _intent_text(intent: Any) -> str:
    if intent is None:
        return ""
    if isinstance(intent, str):
        return intent
    if isinstance(intent, dict):
        return str(intent.get("text", intent.get("intent", "")) or "")
    return str(getattr(intent, "text", getattr(intent, "intent", "")) or "")


def _policy_rules(policy: Any) -> dict[str, str]:
    """Policy rule names to their text, from Constraints, dicts or plain maps."""
    if policy is None:
        return {}
    if isinstance(policy, dict):
        out: dict[str, str] = {}
        for name, rule in policy.items():
            if isinstance(rule, str):
                out[str(name)] = rule
            elif isinstance(rule, dict):
                out[str(rule.get("id", name))] = str(rule.get("rewritten_text") or rule.get("text") or "")
            else:
                out[str(getattr(rule, "id", name))] = str(
                    getattr(rule, "rewritten_text", None) or getattr(rule, "text", "") or "")
        return {name: text for name, text in out.items() if name}
    rules: dict[str, str] = {}
    for item in policy:
        if isinstance(item, dict):
            name = str(item.get("id", ""))
            text = str(item.get("rewritten_text") or item.get("text") or "")
        else:
            name = str(getattr(item, "id", ""))
            text = str(getattr(item, "rewritten_text", None) or getattr(item, "text", "") or "")
        if name:
            rules[name] = text
    return rules


class _Entry:
    """One tool the writer may name, and may call when it is a read with a body."""

    def __init__(self, name: str, kind: str = "read", description: str = "",
                 args_schema: Any = None, body: Any = None):
        self.name = name
        self.kind = kind
        self.description = description
        self.args_schema = args_schema
        self.body = body


def _tool_entries(tools: Any) -> list[_Entry]:
    """Every named tool as an _Entry, from ToolSigs, dicts, objects or callables."""
    if tools is None:
        return []
    if callable(tools) and not hasattr(tools, "name"):
        tools = [tools]
    out: list[_Entry] = []
    for item in tools:
        if callable(item) and getattr(item, "__name__", None) and not hasattr(item, "name"):
            out.append(_Entry(str(item.__name__), body=item))
            continue
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            body = next((item.get(key) for key in ("body", "fn", "func", "impl")
                         if callable(item.get(key))), None)
            out.append(_Entry(name, str(item.get("kind", "read") or "read"),
                              str(item.get("description", "") or ""),
                              item.get("args_schema"), body))
            continue
        name = str(getattr(item, "name", "") or "")
        if not name:
            continue
        body = next((getattr(item, key, None) for key in ("body", "fn", "func", "impl")),
                    None)
        out.append(_Entry(name, str(getattr(item, "kind", "read") or "read"),
                          str(getattr(item, "description", "") or ""),
                          getattr(item, "args_schema", None),
                          body if callable(body) else None))
    return [entry for entry in out if entry.name]


def _read_entries(entries: list[_Entry]) -> list[_Entry]:
    return [entry for entry in entries if entry.kind == "read" and callable(entry.body)]


# The provider-ready schema shape, a copy of runner/world/loading.py's _JSON_TYPES
# and _json_schema: mined schemas spell types the miners' way ("str"), providers
# read draft 2020-12. Copied so the Examiner never loads the toolkit loader;
# tests/examiner/test_intent_derive.py holds the two together.
_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
               "list": "array", "NoneType": "null"}


def _json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {key: _json_schema(value) for key, value in node.items() if key != "type"}
        if "type" in node:
            types = node["type"] if isinstance(node["type"], list) else [node["type"]]
            names = sorted({_JSON_TYPES.get(str(t), str(t)) for t in types})
            out["type"] = names[0] if len(names) == 1 else names
        return out
    if isinstance(node, list):
        return [_json_schema(item) for item in node]
    return node


def _input_schema(entry: _Entry) -> dict:
    schema = entry.args_schema
    if not (isinstance(schema, dict) and "properties" in schema):
        schema = {"type": "object",
                  "properties": {name: {"type": "string"} for name in (schema or {})}}
    return _json_schema(schema)


def _world_tables(starting_state: Any) -> dict:
    """The Starting state as plain {table: {id: row}}, copied."""
    if isinstance(starting_state, dict):
        return copy.deepcopy(plain(starting_state))
    shared = getattr(starting_state, "shared", None)
    if isinstance(shared, dict):
        world = copy.deepcopy(plain(shared))
        overlay = getattr(starting_state, "overlay", None)
        if isinstance(overlay, dict):
            for table, rows in plain(overlay).items():
                if isinstance(rows, dict):
                    world.setdefault(table, {}).update(rows)
        return world
    raise ValueError("starting_state is neither a world dict nor a shared/overlay view")


class _View:
    """The small state a state-first body reads: row lookup and the write path."""

    def __init__(self, tables: dict):
        self.tables = tables
        self.shared = tables

    def row(self, table: str, row_id: Any) -> Any:
        return (self.tables.get(table) or {}).get(str(row_id))

    def put(self, table: str, row_id: Any, row: Any) -> None:
        rows = self.tables.setdefault(table, {})
        current = rows.get(str(row_id))
        rows[str(row_id)] = dict(current, **row) if isinstance(current, dict) and isinstance(row, dict) else row

    def any_value(self, field: str) -> Any:
        for rows in self.tables.values():
            if isinstance(rows, dict):
                for row in rows.values():
                    if isinstance(row, dict) and field in row:
                        return row[field]
        return None


def _takes_state(function: Any) -> bool:
    """Mirror of the Runner's route: a first parameter naming the state takes the view."""
    try:
        parameters = list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return False
    return bool(parameters) and parameters[0] in READ_ARG_NAMES


def _execute(body: Any, tables: dict, args: dict) -> tuple[bool, Any]:
    """One read body against the world copy; a raise leaves the copy untouched."""
    try:
        signature = inspect.signature(body)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            if _takes_state(body):
                signature.bind(_View(tables), **args)
            else:
                signature.bind(**args)
        except TypeError as exc:
            return False, f"invalid_arguments: {exc}"
    snapshot = copy.deepcopy(tables)
    try:
        if _takes_state(body):
            return True, body(_View(tables), **args)
        return True, body(**args)
    except Exception as exc:
        tables.clear()
        tables.update(snapshot)
        return False, f"{type(exc).__name__}: {exc}"


def _head() -> str:
    """The invariant instruction lines, the same bytes for every Task and attempt."""
    return "\n".join([
        "You write end-state checks for one task from its request, its policy,",
        "its tool schemas and reads of its starting state.",
        "Every check states one thing the end state must hold or may hold, or one",
        "thing the final message must ask or state. Several acceptable end states",
        "are all welcome as checks; never pin one value where several are right.",
        "Each check carries a because: quote the request in because_quote using its",
        "own words, or name a policy rule in because_rule using the name the policy",
        "gives it. A check with neither is refused, and so is one whose quote the",
        "request does not hold, whose rule is not in the policy, or whose value no",
        "read returned and the request does not hold.",
        "Exact values come from what your reads return or from what the request",
        "says, never from habit. Read first where the request points at the world.",
        "Answer with one JSON object: {\"checks\": [{\"kind\", ...}]}.",
        "A required or allowed check over a write names tool, entity and id_field: entity is the",
        "id of the exact row the write acts on, as your reads show it, with id_field naming that",
        "id column, never the table. One over a column adds field, value and alternatives.",
        "A question names tool and, where one is asked for, field.",
        "A communicate check carries the text the final message must state.",
    ])


def _tool_lines(entries: list[_Entry], readable: set[str]) -> list[str]:
    lines = []
    for entry in entries:
        args = _input_schema(entry)
        props = ", ".join(sorted((args.get("properties") or {}).keys())) if isinstance(args, dict) else ""
        mark = "callable" if entry.name in readable else "named only"
        lines.append(f"- {entry.name} ({entry.kind}, {mark}): "
                     f"{entry.description or 'no description'}"
                     f"{f' args: {props}' if props else ''}")
    return lines


def _body(intent_text: str, rules: dict[str, str], entries: list[_Entry],
          readable: set[str]) -> str:
    lines = [f"The request, in the user's own words:\n{intent_text or '(no words recorded)'}"]
    if rules:
        lines.append("Policy rules:\n" + "\n".join(f"- {name}: {text}" for name, text in sorted(rules.items())))
    else:
        lines.append("Policy rules: none.")
    lines.append("Tools:\n" + "\n".join(_tool_lines(entries, readable)))
    lines.append("The starting state is behind the callable tools; read what the request points at, "
                 "then answer with the JSON object.")
    return "\n\n".join(lines)


def _arguments(call: Any) -> dict:
    args = call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return dict(args) if isinstance(args, dict) else {}


def _session(model: Any, messages: list[dict], schemas: list[dict],
             bodies: dict[str, Any], tables: dict, read_tokens: set[str], *,
             max_turns: int, tally: dict) -> str:
    """Run one attempt: reads until the session answers without a tool call."""
    for _ in range(max_turns):
        reply = model.query(messages, tools=schemas or None)
        tally["calls"] += 1
        usage = getattr(reply, "usage", None)
        tally["input_tokens"] += int(getattr(usage, "input", 0) or 0)
        tally["output_tokens"] += int(getattr(usage, "output", 0) or 0)
        calls = list(getattr(reply, "tool_calls", None) or [])
        messages.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [c if isinstance(c, dict) else c.model_dump() for c in calls]})
        if not calls:
            return reply.content or ""
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = _arguments(call)
            call_id = (call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) or name
            body = bodies.get(str(name))
            if body is None:
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": f"no such read tool: {name}"})
                continue
            tally["reads"] += 1
            ok, result = _execute(body, tables, args)
            text = json.dumps(result, default=str, ensure_ascii=False)
            if len(text) > RESULT_CHARS:
                text = text[:RESULT_CHARS] + "...[truncated]"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if ok:
                read_tokens.update(_intent_tokens(text))
    reply = model.query(messages)
    tally["calls"] += 1
    return reply.content or ""


def _unfence(text: str) -> str:
    """A JSON answer out of the fences models wrap it in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped[:4].lower() == "json":
            stripped = stripped[4:]
        end = stripped.rfind("```")
        if end != -1:
            stripped = stripped[:end]
    return stripped.strip()


def _parse_checks(text: str) -> tuple[list[dict], Optional[str]]:
    try:
        body = json.loads(_unfence(text))
    except ValueError:
        return [], "the answer is not a JSON object with a checks array"
    checks = body.get("checks") if isinstance(body, dict) else None
    if not isinstance(checks, list):
        return [], "the answer is not a JSON object with a checks array"
    return [check for check in checks if isinstance(check, dict)], None


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    pos = 0
    for needle in needles:
        try:
            pos = haystack.index(needle, pos) + 1
        except ValueError:
            return False
    return True


def _ground(checks: list[dict], intent_toks: list[str], intent_set: set[str],
            rule_names: set[str], read_set: set[str]) -> tuple[list[dict], list[dict]]:
    """Split candidate checks into kept and refused, by code only."""
    kept: list[dict] = []
    refused: list[dict] = []
    for index, check in enumerate(checks):
        reason = _refusal(check, intent_toks, intent_set, rule_names, read_set)
        if reason is None:
            kept.append(check)
        else:
            refused.append({"index": index, "kind": check.get("kind"), "reason": reason})
    return kept, refused


def _value_tokens(check: dict) -> list[str]:
    toks: list[str] = []
    for value in [check.get("value"), *((check.get("alternatives") or []))]:
        if value is None:
            continue
        toks.extend(_intent_tokens(value) or [str(value)])
    text = check.get("text")
    if isinstance(text, str) and text:
        toks.extend(_score_tokens(text))
    return toks


def _refusal(check: dict, intent_toks: list[str], intent_set: set[str],
             rule_names: set[str], read_set: set[str]) -> Optional[str]:
    kind = check.get("kind")
    if kind not in MUST_KINDS:
        return f"kind {kind!r} is not a must check"
    quote = check.get("because_quote")
    rule = check.get("because_rule")
    quote_ok = bool(isinstance(quote, str) and quote.strip()) and _is_subsequence(
        _intent_tokens(quote), intent_toks)
    rule_ok = bool(isinstance(rule, str) and rule in rule_names)
    if not (quote_ok or rule_ok):
        if quote and not rule:
            return "the quote is not in the request"
        if rule and not quote:
            return f"the policy names no rule {rule!r}"
        return "no because from the request or the policy"
    if kind in ("required", "allowed") and not check.get("tool"):
        return "a write check names its tool"
    if kind == "question" and not check.get("tool"):
        return "a question names its tool"
    if kind == "question" and check.get("field") is not None and not isinstance(check.get("field"), str):
        return "a question names one field"
    text = check.get("text")
    if kind == "communicate" and not (isinstance(text, str) and text.strip()):
        return "a communicate check carries its text"
    if kind == "communicate" and not _score_tokens(str(text)):
        return "the text states no token the scorer reads"
    if kind in ("required", "allowed") and check.get("field") is not None:
        if check.get("value") is None:
            return "a column check carries its value"
    held = intent_set | read_set
    need = [folded for token in _value_tokens(check) for folded in (_intent_tokens(token) or [token])]
    missing = sorted({tok for tok in need if tok not in held})
    if missing:
        return f"no read returned these and the request does not hold them: {', '.join(missing[:5])}"
    return None


def _because(check: dict) -> str:
    quote = check.get("because_quote")
    if isinstance(quote, str) and quote.strip():
        return f"because {quote.strip()!r}"
    return f"because rule {check.get('because_rule')}"


def _build_atoms(kept: list[dict], fn: Any, intent_set: set[str]) -> list[Atom]:
    atoms: list[Atom] = []
    write_no = 0
    comm_no = 0
    for check in kept:
        kind = check["kind"]
        because = _because(check)
        if kind == "question":
            tool = check["tool"]
            field = check.get("field")
            key = f"field:{field}" if isinstance(field, str) and field else f"confirm:{tool}"
            atoms.append(make_atom(f"q.{key}", "question",
                                   {"kind": "question", "key": key, "tool": tool, "field": field},
                                   helpers=HELPERS_SRC,
                                   description=f"the final message asks about {key.split(':', 1)[-1]}; {because}"))
        elif kind == "communicate":
            for token in _score_tokens(str(check["text"])):
                atoms.append(make_atom(f"c{comm_no}", "communicate",
                                       {"kind": "communicate", "value": _key(fn, token),
                                        "text": token, "field": None, "source_tool": None},
                                       helpers=HELPERS_SRC,
                                       provenance=("user_stated" if token in intent_set
                                                   else "system_derived"),
                                       description=f"the final message states {token}; {because}"))
                comm_no += 1
        else:
            base = {"tool": check["tool"], "entity": check.get("entity"),
                    "entity_raw": check.get("entity"), "id_field": check.get("id_field")}
            atoms.append(make_atom(f"w{write_no}", kind, dict(base, kind="write"),
                                   helpers=HELPERS_SRC,
                                   description=f"{check['tool']} writes {check.get('entity') or 'an entity'}; "
                                               f"{because}"))
            field = check.get("field")
            if field is not None:
                values = [check.get("value"), *((check.get("alternatives") or []))]
                for pos, value in enumerate(values):
                    if value is None:
                        continue
                    sub_kind = kind if pos == 0 else "allowed"
                    atoms.append(make_atom(
                        f"w{write_no}.{field}" + ("" if pos == 0 else f".alt{pos}"), sub_kind,
                        dict(base, kind="write_value", field=field, value=_key(fn, value), raw=value),
                        helpers=HELPERS_SRC,
                        provenance=("user_stated" if set(_intent_tokens(value)) <= intent_set
                                    else "system_derived"),
                        description=f"{check['tool']} {field} is {text_of(value)}; {because}"))
            write_no += 1
    return atoms


def _feedback(refused: list[dict], checks: list[dict]) -> str:
    lines = []
    for item in refused:
        check = checks[item["index"]] if 0 <= item["index"] < len(checks) else {}
        lines.append(f"- check {item['index']} ({item['kind']}): {item['reason']}; "
                     f"it said {json.dumps(check, default=str)[:200]}")
    return "\n".join(lines)


def write_verifier(task: Any, intent: Any, policy: Any, tools: Any, starting_state: Any,
                   model: Any, *, canon: Any, max_attempts: int = MAX_ATTEMPTS,
                   max_turns: int = MAX_TURNS, stats: Optional[dict] = None) -> Verifier:
    """Write one Task's Verifier from the Task only; the scorer rules on it unchanged."""
    task_id = _task_id(task)
    words = _intent_text(intent) or _intent_text(task)
    rules = _policy_rules(policy)
    entries = _tool_entries(tools)
    readable = _read_entries(entries)
    bodies = {entry.name: entry.body for entry in readable}
    readable_names = set(bodies)
    schemas = [{"name": entry.name, "description": entry.description or f"read with {entry.name}",
                "input_schema": _input_schema(entry)} for entry in readable]
    tables = _world_tables(starting_state)
    fn = canon_fn(canon)
    intent_toks = _intent_tokens(words)
    intent_set = set(intent_toks)
    read_tokens: set[str] = set()
    tally = {"calls": 0, "reads": 0, "input_tokens": 0, "output_tokens": 0}
    messages: list[dict] = [{"role": "system", "content": _head()},
                            {"role": "user", "content": _body(words, rules, entries, readable_names)}]
    best: tuple = ()
    best_atoms: list[Atom] = []
    best_counts = {"written": 0, "refused": 0, "kept_checks": 0}
    attempts = 0
    for _ in range(max(1, max_attempts)):
        attempts += 1
        text = _session(model, messages, schemas, bodies, tables, read_tokens,
                        max_turns=max_turns, tally=tally)
        checks, parse_error = _parse_checks(text)
        if parse_error is not None:
            refused = [{"index": -1, "kind": None, "reason": parse_error}]
            kept: list[dict] = []
        else:
            kept, refused = _ground(checks, intent_toks, intent_set,
                                    set(rules), set(read_tokens))
        score = (len(kept), -len(refused))
        if not best or score > best:
            best = score
            best_atoms = _build_atoms(kept, fn, intent_set)
            best_counts = {"written": len(checks), "refused": len(refused), "kept_checks": len(kept)}
        if not refused:
            break
        messages.append({"role": "user",
                         "content": f"Your last checks were refused as follows:\n"
                                    f"{_feedback(refused, checks)}\nWrite the corrected checks as JSON."})
    verifier = Verifier(task_id=task_id, atoms=best_atoms, verifier_version="intent-1",
                        seed_run_ids=[])
    if stats is not None:
        stats.update({"attempts": attempts, **best_counts, "atoms": len(best_atoms), **tally})
    return verifier


__all__ = ["MAX_ATTEMPTS", "MAX_TURNS", "MUST_KINDS", "tier_of", "write_verifier"]
