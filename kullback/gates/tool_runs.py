"""The six rulings over what a generated tool body did when the sandbox ran it (design section 6).

`builder/sandbox.py` runs a body in a subprocess and hands back one result dict per recorded call;
nothing here starts a process. Each ruling takes the calls and those results (or the sandbox's
error, when the module did not load or timed out) and decides: does the module parse, did every
call run without crashing, do two fresh runs agree, do different arguments give different answers,
do the recorded calls replay to their recorded results, and does a write refuse a reference the
world does not hold. The sandbox's gate functions are thin wrappers that run and then call one of
these, so the accept-or-reject decision is in this package and hashed with it (D122) while the
subprocess stays where it was.

The row helpers (`parse_result`, `match_table`, `columns_of`, `id_field`, `id_pattern_for`) live
here because the replay ruling compares rows column by column under the schema's classes (D73,
D84); `sandbox.py` and `compile_env.py` read them back from here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from kullback.runner.canon import canonicalize as canon
from kullback.runner.records import EntitySchema, GateResult, ToolCall, content_hash

CRASH_ERRORS = frozenset({"NameError", "AttributeError", "TypeError", "ImportError",
                          "ModuleNotFoundError", "IndentationError", "SyntaxError", "RecursionError"})
TOOL_RUN_STAGES = ("parses", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity", "refuses_unknown")


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
        return canon(expected, rules) == canon(got, rules), []
    differs = [n for n in columns_of(schema, found[0], "hard")
               if canon(expected.get(n), rules) != canon(got.get(n), rules)]
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
                            f"{', '.join(differing) or 'value'}")
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
