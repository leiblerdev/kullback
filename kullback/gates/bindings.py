"""Which gates run when an agent writes a file, and the rows each refusal names.

A `Binding` maps a glob, relative to the extension root the write landed under, to the
ordered gate names that run on that write and the loaders that fetch what those gates
need from the workdir. `rulings_for` runs them in order and stops at the first refusal,
the way the stage gate did, with one exception: the static gates of a tool body (those that
read its text and run nothing) all rule before any execution gate, and a memorised-values
refusal does not stop the chain, so every static failure is reported together, ahead of the
replay ones. Every `Ruling` carries `rows`: the per-call records behind
a refusal (call id, tool, arguments digest, recorded value, our value, first differing
column), never a bare count.

The loaders read the workdir files the build already writes (`traces/`, `schema.json`,
`db.json`, `tool_sigs.json`, `canon-rules.json`, and the Examiner's history, task runs
and probe pools where records names them); anything a loader cannot find is missing evidence and the gates that need it
are skipped, never failed. Running a tool body needs an executor, which lives outside
this package (the gates never import an agent or a builder): `rulings_for` and
`gate_writes` take it as `execute(source, calls, db)`, returning one sandbox-shaped
result dict per call (`ok` with `value`, or `error` with `message`). With no executor
the execution gates are skipped the same way. An executor that takes a `trace_calls`
keyword is handed every recorded call of the workdir grouped by trace (all tools), so it
can replay each gated call's trace prefix first.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NamedTuple, Optional

from kullback.gates.confinement import gate_confined, predicate_confinement_gate, written_body
from kullback.gates.fidelity import oracle_replay_gate
from kullback.gates.loosening import false_rejection_gate, loosening_gate
from kullback.gates.stages import intent_gate
from kullback.gates.tool_runs import (
    CRASH_ERRORS,
    MEMORISED_STAGE,
    body_deterministic_gate,
    body_executes_gate,
    body_memorised_values_gate,
    body_non_trivial_gate,
    body_parses_gate,
    body_refuses_unknown_gate,
    body_replay_fidelity_gate,
    body_sensitivity_gate,
    classify_exception,
    compare_results,
    differing_columns,
    parse_result,
)
from kullback.gates.trust import finished_runs, refuse_gate, trusted_gate
from kullback.gates.verifier_suite import D79_STAGES, validate_verifier
from kullback.runner.canon import CanonRules
from kullback.runner.canon import load_rules as load_canon_rules
from kullback.runner.records import (
    EntitySchema,
    GateResult,
    ToolCall,
    Verifier,
    content_hash,
    load_exam_history,
    load_exam_task_runs,
    load_probe_pools,
)

#: How many rows one ruling carries before the count of the rest.
ROWS_CAP = 20

#: The D79 stage a single-path Task fails by structure, the one D199 may waive, and the key of the
#: row a waived ruling carries so a reader of the ruling alone can tell a waiver from a pass.
ALT_PATH_STAGE = "verifier_alt_path"
WAIVED_ROW = "second_path_waived"

#: Evidence loaders: (workdir, written relpath) -> the artifact, or None when absent.
Loader = Callable[[Path, str], Any]


class Binding(NamedTuple):
    """One written-file shape and what it is ruled by: glob `pattern`, ordered gate
    `gates`, and `reads`, the workdir loaders those gates need."""

    pattern: str
    gates: tuple[str, ...]
    reads: dict[str, Loader]


def _read_json(workdir: Path, name: str) -> Any:
    try:
        return json.loads((workdir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _as_calls(raw: Any) -> Optional[list[ToolCall]]:
    if raw is None:
        return None
    items = raw.get("tool_calls") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return None
    out = []
    for item in items:
        try:
            out.append(item if isinstance(item, ToolCall) else ToolCall.model_validate(item))
        except ValueError:
            continue
    return out


def load_calls(workdir: Path, path: str) -> Optional[list[ToolCall]]:
    """The recorded calls of the tool this file implements, out of `traces/`."""
    stem = PurePosixPath(path).stem
    traces = workdir / "traces"
    if not traces.is_dir():
        return None
    out: list[ToolCall] = []
    for file in sorted(traces.glob("*.json")):
        for call in _as_calls(_read_json(workdir, f"traces/{file.name}") or []) or []:
            if call.name == stem:
                out.append(call)
    return out or None


#: Where a workdir keeps its recorded calls one jsonl per tool, read where `traces/` holds none.
CALLS_DIRS = ("env/calls", "calls")


def _jsonl_calls(file: Path) -> list[ToolCall]:
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return _as_calls(rows) or []


def load_trace_calls(workdir: Path, path: str) -> Optional[dict[str, list[ToolCall]]]:
    """Every recorded call, any tool, grouped by the trace that made it.

    The table a gated call's prefix is read from: the calls its own trace made before it. The
    source is `traces/`, the same files `load_calls` reads the gated calls from, so every gated
    call finds its own trace here; a workdir with no calls there is read from its calls
    directory, one jsonl per tool.
    """
    _ = path
    calls: list[ToolCall] = []
    traces = workdir / "traces"
    if traces.is_dir():
        for file in sorted(traces.glob("*.json")):
            calls += _as_calls(_read_json(workdir, f"traces/{file.name}") or []) or []
    for name in CALLS_DIRS if not calls else ():
        folder = workdir / name
        if folder.is_dir():
            for file in sorted(folder.glob("*.jsonl")):
                calls += _jsonl_calls(file)
            break
    out: dict[str, list[ToolCall]] = {}
    for call in calls:
        if call.trace_id:
            out.setdefault(call.trace_id, []).append(call)
    for rows in out.values():
        rows.sort(key=_recorded_order)
    return out or None


def _recorded_order(call: ToolCall) -> tuple:
    position = getattr(call.raw_ptr, "msg_index", None)
    return (position is None, position if isinstance(position, int) else 0, call.id or "")


def load_schema(workdir: Path, path: str) -> Optional[EntitySchema]:
    _ = path
    raw = _read_json(workdir, "schema.json")
    if raw is None:
        return None
    try:
        return raw if isinstance(raw, EntitySchema) else EntitySchema.model_validate(raw)
    except ValueError:
        return None


def load_db(workdir: Path, path: str) -> Any:
    _ = path
    return _read_json(workdir, "db.json")


def load_rules(workdir: Path, path: str) -> Optional[CanonRules]:
    """The workdir's canon rules as the canonicalizer takes them, or None for the defaults.

    The raw JSON dict is not enough: every gate that canonicalizes a value reads the rules as
    `CanonRules`, and a dict made each of them raise, which the suite skips silently, so a write
    drew no deterministic, non-trivial or fidelity ruling at all.
    """
    _ = path
    try:
        return load_canon_rules(Path(workdir) / "canon-rules.json")
    except (OSError, ValueError):
        return None


def load_history(workdir: Path, path: str) -> Optional[dict]:
    """The Verifier histories the Examiner saved (records.exam_history_path), or None when there are none."""
    _ = path
    return load_exam_history(workdir) or None


def load_task_runs(workdir: Path, path: str) -> Optional[dict]:
    """The Runs per Task the Examiner saved (records.exam_task_runs_path), or None when there are none."""
    _ = path
    return load_exam_task_runs(workdir) or None


def load_probes(workdir: Path, path: str) -> Optional[dict]:
    """Every probe pool on disk (records.load_probe_pools), or None when there are none."""
    _ = path
    return load_probe_pools(workdir) or None


def _file_loader(name: str) -> Loader:
    def load(workdir: Path, path: str) -> Any:
        _ = path
        return _read_json(workdir, name)

    load.__name__ = f"load_{name.split('.')[0].replace('-', '_')}"
    return load


#: The tool-body gates that read the text and run nothing. They rule first, all of them.
STATIC_TOOL_GATES = ("confined", "parses", MEMORISED_STAGE)
#: A failed static gate among these stops the chain: a body that is not Python, or not confined,
#: is never run. A memorised-values refusal lets the execution gates rule after it.
STOPPING_STATIC_GATES = frozenset({"confined", "parses"})
TOOL_GATES = STATIC_TOOL_GATES + ("executes_on_s0", "deterministic", "non_trivial", "sensitivity",
                                  "replay_fidelity", "refuses_unknown")
VERIFIER_GATES = ("gate_a_oracle_replay", "verifier_suite", "loosening", "false_rejection", "trusted")

BINDINGS: tuple[Binding, ...] = (
    Binding("tools/*.py", TOOL_GATES,
            {"calls": load_calls, "db": load_db, "schema": load_schema, "rules": load_rules,
             "trace_calls": load_trace_calls}),
    Binding("verifiers/*.json", VERIFIER_GATES,
            {"replays": _file_loader("replays.json"), "rerolls": _file_loader("rerolls.json"),
             "history": load_history, "task_runs": load_task_runs,
             "reference": _file_loader("reference.json"), "verifiers": _file_loader("verifiers.json"),
             "probes": load_probes, "sigs": _file_loader("tool_sigs.json"),
             "rules": load_rules}),
    Binding("refusals/*.json", ("refuse",),
            {"replays": _file_loader("replays.json"), "rerolls": _file_loader("rerolls.json")}),
    Binding("intents/*.json", ("intent",), {}),
    Binding("policy/*.py", ("compile_policy.confined",), {}),
)


def binding_for(path: str) -> Optional[Binding]:
    """The first binding whose glob matches this extension-relative path, or None."""
    posix = path.replace("\\", "/").lstrip("/")
    for binding in BINDINGS:
        if PurePosixPath(posix).match(binding.pattern) or fnmatch.fnmatchcase(posix, binding.pattern):
            return binding
    return None


# --- evidence ---

def _schema_of(evidence: dict) -> EntitySchema:
    schema = evidence.get("schema")
    if isinstance(schema, EntitySchema):
        return schema
    if schema is not None:
        try:
            return EntitySchema.model_validate(schema)
        except ValueError:
            pass
    return EntitySchema()


def _calls_of(evidence: dict) -> Optional[list[ToolCall]]:
    calls = evidence.get("calls")
    if calls is None:
        return None
    if isinstance(calls, list) and all(isinstance(call, ToolCall) for call in calls):
        return calls
    return _as_calls(calls)


def _verifier_of(evidence: dict) -> Optional[Verifier]:
    verifier = evidence.get("verifier", evidence.get("document"))
    if verifier is None:
        return None
    if isinstance(verifier, Verifier):
        return verifier
    try:
        return Verifier.model_validate(verifier)
    except ValueError:
        return None


def _takes_trace_calls(execute: Any) -> bool:
    try:
        parameters = inspect.signature(execute).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "trace_calls" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _execute(evidence: dict, calls: list[ToolCall]) -> list[dict]:
    """The executor over these calls, handed the trace table where it takes one."""
    execute, trace_calls = evidence["execute"], evidence.get("trace_calls")
    if trace_calls and _takes_trace_calls(execute):
        return list(execute(evidence.get("source"), calls, evidence.get("db"), trace_calls=trace_calls))
    return list(execute(evidence.get("source"), calls, evidence.get("db")))


def _run_execute(evidence: dict, calls: list[ToolCall]) -> Optional[list[dict]]:
    """One shared execution of the body over these calls, cached on the evidence."""
    if "_results" not in evidence:
        if evidence.get("execute") is None:
            return None
        try:
            evidence["_results"] = _execute(evidence, calls)
        except Exception as exc:  # noqa: BLE001, the sandbox's own failure is a ruling, not a crash
            evidence["_results"] = None
            evidence["_exec_error"] = str(exc)
    return evidence.get("_results")


def _exec_error(evidence: dict, calls: list[ToolCall]) -> Optional[str]:
    return evidence.get("_exec_error") if _run_execute(evidence, calls) is None else None


# --- the bound gates, each a list of results so the suite can rule nine times ---

def _gate_confined(evidence: dict) -> Optional[list[GateResult]]:
    """Confinement over the module the Runner will load, when the caller can render it.

    The written file is one plain function, so a check over it alone finds no tool method and
    passes what `load_toolkit` then refuses. `render(source)` (the executor's, see
    builder/env_files.py) returns the whole rendered module with this body in place, and the
    gate reads that; a body that does not render is ruled on its own text. The ruling is about
    the written body: another body's failures ride it as a note (F27, see `gate_confined`).
    """
    source = evidence.get("source")
    if source is None:
        return None
    render = evidence.get("render")
    module = None
    if callable(render):
        try:
            module = render(source)
        except Exception:  # noqa: BLE001, a body that does not render is ruled on its own text
            module = None
    if module is None:
        return [gate_confined(source)]
    return [gate_confined(module, written=written_body(source))]


def _gate_parses(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("source") is None:
        return None
    return [body_parses_gate(evidence["source"])]


def _gate_executes(evidence: dict) -> Optional[list[GateResult]]:
    calls = _calls_of(evidence)
    if not calls or evidence.get("execute") is None:
        return None
    return [body_executes_gate(calls, _run_execute(evidence, calls), error=_exec_error(evidence, calls))]


def _gate_deterministic(evidence: dict) -> Optional[list[GateResult]]:
    calls = _calls_of(evidence)
    if not calls or evidence.get("execute") is None:
        return None
    try:
        first, second = _execute(evidence, calls), _execute(evidence, calls)
    except Exception as exc:  # noqa: BLE001, see _run_execute
        return [body_deterministic_gate(calls, None, None, evidence.get("rules"), error=str(exc))]
    evidence["_first"], evidence["_second"] = first, second
    return [body_deterministic_gate(calls, first, second, evidence.get("rules"))]


def _gate_non_trivial(evidence: dict) -> Optional[list[GateResult]]:
    calls = _calls_of(evidence)
    if not calls or evidence.get("execute") is None:
        return None
    return [body_non_trivial_gate(calls, _run_execute(evidence, calls), evidence.get("rules"),
                                 error=_exec_error(evidence, calls))]


def _gate_sensitivity(evidence: dict) -> Optional[list[GateResult]]:
    pairs, first, second = evidence.get("pairs"), evidence.get("first"), evidence.get("second")
    if pairs is None or first is None or second is None:
        return None
    return [body_sensitivity_gate(pairs, first, second, _schema_of(evidence), evidence.get("rules"),
                                 evidence.get("readers"))]


def _gate_replay(evidence: dict) -> Optional[list[GateResult]]:
    calls = _calls_of(evidence)
    if not calls or evidence.get("execute") is None:
        return None
    return [body_replay_fidelity_gate(calls, _run_execute(evidence, calls), _schema_of(evidence),
                                     rules=evidence.get("rules"), readers=evidence.get("readers"),
                                     error=_exec_error(evidence, calls))]


def _gate_refuses_unknown(evidence: dict) -> Optional[list[GateResult]]:
    probes = evidence.get("unknown_probes")
    if probes is None:
        return None
    if not probes:
        return [body_refuses_unknown_gate([], None)]
    if evidence.get("execute") is None:
        return None
    calls = [probe[2] for probe in probes]
    try:
        results = _execute(evidence, calls)
    except Exception as exc:  # noqa: BLE001, see _run_execute
        return [body_refuses_unknown_gate(probes, None, error=str(exc))]
    return [body_refuses_unknown_gate(probes, results)]


def _gate_memorised(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("source") is None:
        return None
    return [body_memorised_values_gate(evidence["source"], evidence.get("schema"), evidence.get("db"),
                                      _calls_of(evidence) or (), sig=evidence.get("sig"),
                                      readers=evidence.get("readers"))]


def _gate_oracle_replay(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("replays") is None:
        return None
    return [oracle_replay_gate(evidence["replays"], evidence.get("rules"))]


def _waive_alt_path(evidence: dict, verifier: Verifier, results: list[GateResult]) -> list[GateResult]:
    """D199 on the write, as the derivation rules it: a Task whose file row says its second path was
    waived (single by structure, a held-out Run at the Reference) has the not-run alt-path check pass
    as waived, where every other stage passed. The flag is read off the row, never re-derived here."""
    row = (evidence.get("task_status") or {}).get(verifier.task_id) or {}
    failing = [result for result in results if not result.passed]
    if not (row.get("second_path_waived") and len(failing) == 1 and failing[0].stage == ALT_PATH_STAGE
            and (failing[0].metrics or {}).get("skipped")):
        return results
    waived = GateResult(stage=ALT_PATH_STAGE, passed=True, failures=[],
                        metrics={**(failing[0].metrics or {}), WAIVED_ROW: True})
    return [waived if result is failing[0] else result for result in results]


def _gate_suite(evidence: dict) -> Optional[list[GateResult]]:
    verifier = _verifier_of(evidence)
    if verifier is None or evidence.get("reference") is None:
        return None
    return _waive_alt_path(evidence, verifier, list(validate_verifier(verifier, evidence["reference"], evidence.get("empty_run"),
                                 evidence.get("wrong_run"), evidence.get("alt_path_run"),
                                 canon=evidence.get("rules"), write_tools=evidence.get("write_tools"),
                                 seed_runs=evidence.get("seed_runs"), model=evidence.get("probe_model"),
                                 run_probe=evidence.get("run_probe"))))


def _gate_loosening(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("history") is None:
        return None
    return [loosening_gate(evidence["history"], evidence.get("task_runs") or {}, evidence.get("replays") or {},
                           evidence.get("rerolls") or {}, evidence.get("rules"), evidence.get("sigs") or [])]


def _gate_false_rejection(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("verifiers") is None:
        return None
    return [false_rejection_gate(evidence["verifiers"], evidence.get("task_runs") or {},
                                evidence.get("replays") or {}, evidence.get("rerolls") or {},
                                evidence.get("rules"), evidence.get("sigs") or [],
                                task_status=evidence.get("task_status"))]


def _alt_path_waived(rulings: Any) -> bool:
    """Whether the alt-path ruling among these passed by the D199 waiver: its row says so."""
    return any(getattr(ruling, "stage", None) == ALT_PATH_STAGE and ruling.passed
               and any(WAIVED_ROW in row for row in ruling.rows or ()) for ruling in rulings or ())


def _suite_status(evidence: dict) -> Optional[tuple[str, dict]]:
    """The ruled Verifier's Task and the status row the D79 suite that ran in this ruling gives it,
    in the shape the derivation writes: `verifier_passed`, `checks` by check name, `not_run` by stage.
    A waived alt-path check (D199) is written as the derivation writes it: the check not passed, the
    Verifier passed, `second_path_waived` true."""
    rulings = evidence.get("rulings") or ()
    ran = {ruling.stage: bool(ruling.passed) for ruling in rulings
           if getattr(ruling, "stage", None) in D79_STAGES}
    verifier = _verifier_of(evidence)
    if not ran or verifier is None:
        return None
    row = {"verifier_passed": all(ran.values()),
           "checks": {D79_STAGES[stage]: passed for stage, passed in ran.items()},
           "not_run": [stage for stage in D79_STAGES if stage not in ran]}
    if _alt_path_waived(rulings):
        row["checks"][D79_STAGES[ALT_PATH_STAGE]] = False
        row["second_path_waived"] = True
    return verifier.task_id, row


def _gate_trusted(evidence: dict) -> Optional[list[GateResult]]:
    """The trusted gate rules on the suite that just ran on the write: its row overlays the Task's
    status for this ruling only. With no suite in this ruling it falls back to the file row."""
    task_status = evidence.get("task_status")
    suite = _suite_status(evidence)
    if suite is not None:
        task_id, row = suite
        task_status = {**(task_status or {}), task_id: {**((task_status or {}).get(task_id) or {}), **row}}
    if evidence.get("verifiers") is None or task_status is None:
        return None
    return [trusted_gate(task_status, evidence["verifiers"], evidence.get("probes") or {},
                         evidence.get("history") or {}, evidence.get("refusals") or {},
                         evidence.get("task_runs") or {}, evidence.get("replays") or {},
                         evidence.get("rerolls") or {}, evidence.get("rules"), evidence.get("sigs") or [])]


def _gate_refuse(evidence: dict) -> Optional[list[GateResult]]:
    refusals = evidence.get("refusals", evidence.get("document"))
    if refusals is None:
        return None
    return [refuse_gate(refusals, evidence.get("replays") or {}, evidence.get("rerolls") or {})]


def _gate_intent(evidence: dict) -> Optional[list[GateResult]]:
    intents = evidence.get("intents", evidence.get("document"))
    if intents is None:
        return None
    return [intent_gate(intents)]


def _gate_policy_confined(evidence: dict) -> Optional[list[GateResult]]:
    if evidence.get("source") is None:
        return None
    return [predicate_confinement_gate(evidence["source"])]


ADAPTERS: dict[str, Callable[[dict], Optional[list[GateResult]]]] = {
    "confined": _gate_confined,
    "parses": _gate_parses,
    "executes_on_s0": _gate_executes,
    "deterministic": _gate_deterministic,
    "non_trivial": _gate_non_trivial,
    "sensitivity": _gate_sensitivity,
    "replay_fidelity": _gate_replay,
    "refuses_unknown": _gate_refuses_unknown,
    MEMORISED_STAGE: _gate_memorised,
    "gate_a_oracle_replay": _gate_oracle_replay,
    "verifier_suite": _gate_suite,
    "loosening": _gate_loosening,
    "false_rejection": _gate_false_rejection,
    "trusted": _gate_trusted,
    "refuse": _gate_refuse,
    "intent": _gate_intent,
    "compile_policy.confined": _gate_policy_confined,
}


# --- rows: what a refusal names ---

def _digest(call: ToolCall) -> str:
    try:
        return str(content_hash(call.args))
    except Exception:  # noqa: BLE001, a digest never fails a ruling
        return repr(sorted((call.args or {}).keys()))


def _replay_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = result
    calls, results = _calls_of(evidence), evidence.get("_results")
    if not calls or not results:
        return []
    schema, rules, readers = _schema_of(evidence), evidence.get("rules"), evidence.get("readers")
    rows = []
    for index, (call, answered) in enumerate(zip(calls, results, strict=False)):
        answered = answered or {}
        if call.error is not None:
            got = classify_exception(answered) if not answered.get("ok") else None
            if got != call.error.class_:
                rows.append({"call": call.id or index, "tool": call.name, "args": _digest(call),
                             "recorded": {"error": call.error.class_}, "ours": {"error": got},
                             "column": "error"})
            continue
        if not answered.get("ok"):
            rows.append({"call": call.id or index, "tool": call.name, "args": _digest(call),
                         "recorded": parse_result(call.result), "ours": None,
                         "column": "error",
                         "exception": f"{answered.get('error')}: {answered.get('message')}"})
            continue
        try:
            ok, _ = compare_results(schema, parse_result(call.result), answered.get("value"), rules,
                                   tool=call.name, readers=readers)
        except Exception:  # noqa: BLE001, an unreadable pair is still a differing call
            ok = False
        if ok:
            continue
        try:
            columns = differing_columns(schema, parse_result(call.result), answered.get("value"), rules,
                                       call.name, readers)
        except Exception:  # noqa: BLE001, see above
            columns = []
        rows.append({"call": call.id or index, "tool": call.name, "args": _digest(call),
                     "recorded": parse_result(call.result), "ours": answered.get("value"),
                     "column": columns[0] if columns else "value"})
    return rows


def _executes_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = result
    calls, results = _calls_of(evidence), evidence.get("_results")
    if not calls or not results:
        return []
    rows = []
    for index, (call, answered) in enumerate(zip(calls, results, strict=False)):
        answered = answered or {}
        if answered.get("ok") or answered.get("error") not in CRASH_ERRORS:
            continue
        rows.append({"call": call.id or index, "tool": call.name, "args": _digest(call),
                     "exception": f"{answered.get('error')}: {answered.get('message')}"})
    return rows


_MEMORISED_PATTERNS = (
    (r"the literal (.*?) has the shape of (.*?) ids", "shape of {} ids"),
    (r"the literal (.*?) is a row id of (.*?) in", "row id of {}"),
    (r"the literal (.*?) is a value the recorded calls passed as (.*?),", "argument {}"),
    (r"the literal (.*?) is a value the recorded results of (.*?) carried", "result of {}"),
    (r"the literal (.*?) is the value (.*?) was left holding", "left holding by {}"),
    (r"the literal (.*?) is the value of (.*?) in", "value of {}"),
)

_TASK_REASON = r"task (\S+?): (.*)"


def _memorised_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = evidence
    import re as _re

    rows = []
    for line in result.failures:
        for pattern, shape in _MEMORISED_PATTERNS:
            found = _re.search(pattern, line)
            if found:
                rows.append({"literal": found.group(1), "match": shape.format(found.group(2))})
                break
        else:
            rows.append({"failure": line})
    return rows


def _atom_text(evidence: dict, atom_id: Any) -> str:
    verifier = _verifier_of(evidence)
    atoms = getattr(verifier, "atoms", None) or []
    for atom in atoms:
        if getattr(atom, "id", None) == atom_id:
            payload = getattr(atom, "payload", None)
            if isinstance(payload, dict) and payload.get("raw") is not None:
                return str(payload["raw"])[:200]
            text = getattr(atom, "text", None)
            if text:
                return str(text)[:200]
            return str(atom)[:200]
    return ""


def _suite_rows(evidence: dict, result: GateResult) -> list[dict]:
    import re as _re

    if (result.metrics or {}).get(WAIVED_ROW):
        return [{"check": result.stage, WAIVED_ROW: (result.metrics or {}).get("not_run_reason", ""),
                 "why": "D199: single path by structure, a held-out Run reached the Reference"}]
    atom_id = (result.metrics or {}).get("failing_atom")
    if atom_id:
        return [{"check": result.stage, "atom": atom_id, "text": _atom_text(evidence, atom_id)}]
    rows = []
    for line in result.failures:
        found = _re.match(r"(\S+):", line)
        atom = found.group(1) if found else ""
        if atom and atom != "not":
            rows.append({"check": result.stage, "atom": atom, "text": _atom_text(evidence, atom)})
        else:
            rows.append({"check": result.stage, "failure": line})
    if not result.passed:
        # F37: an expected-fail Run that passed passed every atom; each is a row the refusal names.
        rows += [{"check": result.stage, "atom": atom["id"], "kind": atom["kind"]}
                 for atom in (result.metrics or {}).get("passing_atoms") or []]
    return rows


def _loosening_rows(evidence: dict, result: GateResult) -> list[dict]:
    import re as _re

    loosened = (result.metrics or {}).get("loosened") or {}
    rows = [{"task": task, "run": run_id} for task, run_ids in loosened.items() for run_id in run_ids]
    if rows:
        return rows
    for line in result.failures:
        found = _re.search(r"task (\S+): version \S+ newly passes (\S+?),", line)
        rows.append({"task": found.group(1), "run": found.group(2)} if found else {"failure": line})
    return rows


def _false_rejection_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = evidence
    import re as _re

    per_task = (result.metrics or {}).get("per_task") or {}
    rows = [{"task": task, "run": run_id}
            for task, row in per_task.items() for run_id in (row.get("rejected_ids") or [])]
    if rows:
        return rows
    for line in result.failures:
        found = _re.search(r"task (\S+):", line)
        rows.append({"task": found.group(1)} if found else {"failure": line})
    return rows


def _refuse_rows(evidence: dict, result: GateResult) -> list[dict]:
    import re as _re

    rejected = (result.metrics or {}).get("rejected") or []
    rows = []
    for task in rejected:
        try:
            finished = finished_runs(task, evidence.get("replays") or {}, evidence.get("rerolls") or {})
        except Exception:  # noqa: BLE001, the line still names the task
            finished = []
        rows.append({"task": task, "finished": finished})
    for line in result.failures:
        found = _re.match(_TASK_REASON, line)
        if found and found.group(1) not in rejected:
            rows.append({"task": found.group(1), "reason": found.group(2)})
    return rows


def _intent_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = evidence
    import re as _re

    rows = []
    for line in result.failures:
        found = _re.match(_TASK_REASON, line)
        rows.append({"task": found.group(1), "reason": found.group(2)} if found else {"failure": line})
    return rows


def _oracle_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = evidence
    import re as _re

    rows = []
    for line in result.failures:
        found = _re.match(r"(\S+?): (.*)", line)
        rows.append({"run": found.group(1), "failure": found.group(2)} if found else {"failure": line})
    return rows


def _trusted_rows(evidence: dict, result: GateResult) -> list[dict]:
    _ = evidence
    import re as _re

    rows = []
    for line in result.failures:
        found = _re.match(_TASK_REASON, line)
        rows.append({"task": found.group(1), "reason": found.group(2)} if found else {"failure": line})
    return rows


ROWS_FOR: dict[str, Callable[[dict, GateResult], list[dict]]] = {
    "replay_fidelity": _replay_rows,
    "executes_on_s0": _executes_rows,
    MEMORISED_STAGE: _memorised_rows,
    "loosening": _loosening_rows,
    "false_rejection": _false_rejection_rows,
    "refuse": _refuse_rows,
    "intent": _intent_rows,
    "gate_a_oracle_replay": _oracle_rows,
    "trusted": _trusted_rows,
}


def rows_for(result: GateResult, evidence: dict) -> list[dict]:
    """The rows behind this ruling: the extractor the stage names, the suite extractor
    for every D79 check, else one row per failure line, so a refusal never rides empty."""
    extractor = ROWS_FOR.get(result.stage)
    if extractor is None and result.stage in D79_STAGES:
        extractor = _suite_rows
    try:
        rows = list(extractor(evidence, result)) if extractor is not None else []
    except Exception:  # noqa: BLE001, row building never fails a ruling
        rows = []
    if not rows and not result.passed:
        rows = [{"failure": line} for line in result.failures]
    if len(rows) > ROWS_CAP:
        rows = rows[:ROWS_CAP] + [{"rest": len(rows) - ROWS_CAP}]
    return [dict(row) for row in rows]


def _to_ruling(result: GateResult, evidence: dict) -> Any:
    from kullback.gates import Ruling

    return Ruling(stage=result.stage, passed=bool(result.passed), failures=list(result.failures),
                  rows=rows_for(result, evidence), note=str(result.metrics.get("note") or ""))


def _rel(root: Any, written_path: str) -> Optional[str]:
    try:
        return str(Path(root, written_path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return None


def rulings_for(root: Any, written_path: str, workdir: Any, *, execute: Any = None,
                evidence: Optional[dict] = None) -> list[Any]:
    """Every ruling a write of this path draws, in binding order, stopping at the first
    refusal, except that a memorised-values refusal lets the execution gates rule after it
    (see the module docstring). A path no binding matches, or a file that is gone, draws nothing.

    `execute(source, calls, db)` runs a tool body over recorded calls; without it the
    execution gates are skipped for missing evidence; its `render(source)`, when it carries
    one, is the rendered module the confinement gate reads. `evidence` overrides or extends
    what the loaders read from the workdir, keyed by evidence name. Each gate sees the
    rulings drawn before it as evidence "rulings", so trusted reads the suite that just ran.
    """
    rel = _rel(root, written_path)
    binding = binding_for(rel) if rel is not None else None
    if rel is None or binding is None:
        return []
    try:
        text = Path(root, rel).read_text(encoding="utf-8")
    except OSError:
        return []
    workdir_path = Path(workdir)
    merged: dict[str, Any] = {"path": rel, "stem": PurePosixPath(rel).stem}
    if rel.endswith(".py"):
        merged["source"] = text
    if rel.endswith(".json"):
        try:
            merged["document"] = json.loads(text)
        except ValueError:
            merged["document"] = None
    for name, load in binding.reads.items():
        try:
            merged.setdefault(name, load(workdir_path, rel))
        except Exception:  # noqa: BLE001, a loader that trips is missing evidence
            merged.setdefault(name, None)
    if execute is not None:
        merged["execute"] = execute
        if callable(getattr(execute, "render", None)):
            merged["render"] = execute.render
    for name, value in (evidence or {}).items():
        if value is not None:
            merged[name] = value
    out: list[Any] = []
    halted = False  # a stopping static gate refused: no execution gate runs
    for gate in binding.gates:
        if halted and gate not in STATIC_TOOL_GATES:
            return out
        adapter = ADAPTERS.get(gate)
        if adapter is None:
            continue
        try:
            results = adapter(merged)
        except Exception:  # noqa: BLE001, a gate that trips is skipped, not failed
            continue
        if not results:
            continue
        for gate_result in results:
            out.append(_to_ruling(gate_result, merged))
            if out[-1].passed:
                continue
            if gate not in STATIC_TOOL_GATES:
                return out
            halted = halted or gate in STOPPING_STATIC_GATES
        merged["rulings"] = list(out)
    return out
