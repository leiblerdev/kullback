"""The seven rulings over what a generated tool body did when the sandbox ran it (design section 6).

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

The row helpers (`parse_result`, `match_table`, `columns_of`, `id_field`, `id_pattern_for`) live
here because the replay ruling compares rows column by column under the schema's classes (D73,
D84); `sandbox.py` and `compile_env.py` read them back from here.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable, Optional

from kullback.gates.confinement import TOOLS_CLASS
from kullback.runner.canon import canonicalize as canon
from kullback.runner.canon import first_difference
from kullback.runner.records import EntitySchema, GateResult, ToolCall, content_hash

CRASH_ERRORS = frozenset({"NameError", "AttributeError", "TypeError", "ImportError",
                          "ModuleNotFoundError", "IndentationError", "SyntaxError", "RecursionError"})
MEMORISED_STAGE = "compile_tools.memorised_values"
TOOL_RUN_STAGES = ("parses", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity",
                   "refuses_unknown", MEMORISED_STAGE)


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


def match_table(schema: EntitySchema, value: Any) -> Optional[tuple[str, str]]:
    """Which table a returned row belongs to, and its id; None when the value is not a row."""
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
            best, best_score = (table, value[name]), score
    return best


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


def body_non_trivial_gate(calls: Iterable[ToolCall], results: Optional[list[dict]], rules: Any = None,
                          error: Optional[str] = None) -> GateResult:
    """4. Different arguments do not all give one constant answer, unless the recorded tool answered them that way.

    The recording is the standard: a hand-off tool that acknowledged 32 argument sets with the same
    line is faithfully constant, and a body that matches it is right. The second retail build failed
    `transfer_to_human_agents` here for doing what the real tool did, 25 of 25 replays agreeing.
    """
    calls = list(calls)
    if error is not None:
        return _ruling("non_trivial", False, {}, [error])
    metrics = {"arg_sets": len({content_hash(c.args) for c in calls}),
               "distinct_answers": len({content_hash(canon(r, rules)) for r in results or []}),
               "recorded_answers": len({content_hash(canon(parse_result(c.result), rules))
                                        for c in calls if c.error is None})}
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


def _row_pairs(schema: EntitySchema, expected: list, got: list) -> list[tuple[Any, Any]]:
    """Pair two lists of rows by id where every row on both sides carries one, else by position."""
    left = [match_table(schema, value) for value in expected]
    right = {found: value for value in got for found in [match_table(schema, value)] if found}
    if all(left) and len(right) == len(got) and all(found in right for found in left):
        return [(value, right[found]) for value, found in zip(expected, left, strict=False)]
    return list(zip(expected, got, strict=False))


def compare_results(schema: EntitySchema, expected: Any, got: Any, rules: Any = None) -> tuple[bool, list[str]]:
    """Hard columns must match after canon; semantic ones are reported, not failed (D73, D84).

    A list of rows and a dict wrapping rows are walked into, so the column classes decide there too.
    Comparing a wrapped result as one canonical string would let an exempt column fail a replay that
    the same row returned on its own passes, which is the opposite of what D73 and D84 ask for.
    """
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return False, [f"list of {len(expected)} against {len(got)}"]
        ok, notes = True, []
        for one, other in _row_pairs(schema, expected, got):
            one_ok, one_notes = compare_results(schema, one, other, rules)
            ok, notes = ok and one_ok, notes + one_notes
        return ok, notes
    found = match_table(schema, expected)
    if found is None and isinstance(expected, dict) and isinstance(got, dict):
        if set(expected) != set(got):
            return False, [f"keys differ: {sorted(set(expected) ^ set(got))}"]
        ok, notes = True, []
        for key in sorted(expected):
            key_ok, key_notes = compare_results(schema, expected[key], got[key], rules)
            ok, notes = ok and key_ok, notes + key_notes
        return ok, notes
    if not found or not isinstance(got, dict):
        found_at = first_difference(got, expected, rules)
        return found_at is None, [found_at] if found_at else []
    # The leaf is named, not only the column: `items` sends a reader into two dumps, and
    # `items[1].options.size: ours "large", recorded "small"` is the repair (D154).
    differs = [found_at for n in columns_of(schema, found[0], "hard")
               for found_at in [first_difference(got.get(n), expected.get(n), rules, n)] if found_at]
    semantic = [f"semantic:{n}" for n in columns_of(schema, found[0], "semantic")
                if canon(expected.get(n), rules) != canon(got.get(n), rules)]
    return not differs, differs + semantic


def body_replay_fidelity_gate(calls: Iterable[ToolCall], results: Optional[list[dict]], schema: EntitySchema,
                              label: str = "held_out", threshold: float = 1.0, rules: Any = None,
                              error: Optional[str] = None) -> GateResult:
    """5. Recorded calls replay: hard columns match after canon, errors match by class, both apart."""
    calls = list(calls)
    if error is not None:
        return _ruling("replay_fidelity", False, {"split": label}, [error])
    hits = {"success_calls": 0, "success_matches": 0, "error_calls": 0, "error_matches": 0}
    semantic, failures = 0, []
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
        ok, differing = compare_results(schema, parse_result(call.result), result["value"], rules)
        semantic += sum(1 for n in differing if n.startswith("semantic:"))
        if ok:
            hits["success_matches"] += 1
        else:
            failures.append(f"{call.name}({args_text(call)}): hard columns differ: "
                            f"{'; '.join(n for n in differing if not n.startswith('semantic:')) or 'value'}")
    success = hits["success_matches"] / hits["success_calls"] if hits["success_calls"] else 1.0
    errors = hits["error_matches"] / hits["error_calls"] if hits["error_calls"] else 1.0
    metrics = dict(hits, split=label, success_fidelity=success, error_fidelity=errors,
                   semantic_differences=semantic)
    return _ruling("replay_fidelity", success >= threshold and errors >= threshold, metrics, failures)


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
# shape would refuse every alphanumeric literal a body writes. The probes are plain English words
# no customer owns; a pattern that accepts one of them is not asked rule (a).
SHAPELESS_PROBES = ("value", "name", "text")


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


def _shaped(pattern: str) -> bool:
    """An id pattern that accepts an ordinary word describes no id shape and is not asked about."""
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
                               class_name: str = TOOLS_CLASS) -> GateResult:
    """7. A body may not memorise the recordings: every id and value comes out of the world (D162).

    Three rules over the literals of the model's own methods, in the order that says most about
    where a value came from. (a) The literal has the shape the schema mined for some table's ids.
    (b) The literal is a row id the Starting state holds. (c) The literal is a value a recorded call
    passed this tool, and neither the signature nor the description names it: an enum member the
    description lists is the tool's vocabulary and is allowed, an order id it happened to be called
    with is not. Names are never data (`structure_names`), so a body writes `row["item_id"]` freely.

    The failure names the literal as the code holds it, the rule that caught it and the table or the
    argument it came from. It is the model's own source, so quoting it back leaks nothing, and the
    held-out split is never named: rule (c) says an argument's name, never a call.
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
    by_table = starting_state_ids(schema, db)
    by_argument = recorded_argument_values(calls)
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
    return _ruling(MEMORISED_STAGE, not failures,
                   {"literals": len(literals), "memorised": len(failures),
                    "tables": len(by_table), "recorded_values": len(by_argument)}, failures)
