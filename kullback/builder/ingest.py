"""Stores the customer's files byte for byte and content-hashed (D66), then derives Trace records
from them with a raw pointer on every field, grader fields stripped into a sidecar, tool errors
classed (D67) and truncated results marked (D95)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from kullback.ai.provider import Model
from kullback.builder import sources
from kullback.builder.mine import _reply_json
from kullback.runner.records import (
    GateResult,
    RawFile,
    RawPtr,
    ToolCallError,
    Trace,
    as_dict,
    content_hash,
)


def _ingest_version() -> str:
    """D66 keys re-derivation on (raw hash, ingest code hash), so the version is this file's own hash."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except OSError:  # pragma: no cover - the source is on disk in every supported install
        return "unhashed"


INGEST_VERSION = _ingest_version()

# Longest first, so "... (truncated)" is not read as a bare "...".
CUT_MARKERS = ("... (truncated)", "[output truncated]", "[truncated]", "<truncated>", "…", "...")

# A JSON result that does not parse was cut off by the customer's log limit even when no marker survived.
UNPARSED_JSON_MARKER = "unterminated_json"

# D67 classes as regexes over the lowercased payload. Every rule is scored and the longest matched
# phrase wins, with the order below as the tie-break, so a loose fragment cannot beat a specific one.
ERROR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tool_not_found", (
        r"\b(?:unknown|unrecognized|undefined|no such|invalid)\s+tool\b",
        # The name token is anything but a quote or whitespace: a source that names its tools with
        # a placeholder ("$DEVICE_ACTION") or a call shape ("$AGENT_FUNCTION{check_x}") says tool
        # not found in the same sentence, and a word-character token let those fall to not_found_entity.
        r"\btool\b['\"\s:=]*[^'\"\s]*['\"]?\s*(?:was\s+|is\s+)?"
        r"(?:not found|does not exist|is not a valid|not a valid|is unknown)",
    )),
    ("permission_denied", ("permission", "not authorized", "unauthorized", "forbidden", "access denied")),
    ("invalid_arguments", ("invalid", "missing required", "required positional argument",
                           "unexpected keyword argument", "validation error", "must be a", "malformed")),
    ("transient", ("rate limit", "timed out", "timeout", "temporarily unavailable", "try again later",
                   "service unavailable", "overloaded", "connection reset",
                   r"\b(?:http|https|status(?:\s+code)?|code)\s*[:=]?\s*(?:429|5\d\d)\b",
                   r"^\s*(?:429|5\d\d)\b",
                   r"\b5\d\d\s+(?:internal server|bad gateway|gateway timeout|service unavailable)")),
    ("cancelled", ("cancelled by", "canceled by", "was cancelled", "was canceled", "aborted")),
    ("not_found_entity", ("not found", "not_found", "no such", "does not exist")),
    ("business_error", ("cannot be", "can not be", "should be", "should match", "insufficient",
                        "not allowed", "not eligible", "not permitted", "already been", "policy")),
)

ERROR_CLASSES = frozenset(rule[0] for rule in ERROR_RULES) | {"unknown"}

_COMPILED_RULES = tuple(
    (class_, tuple(re.compile(pattern) for pattern in patterns)) for class_, patterns in ERROR_RULES
)


# --- raw store -------------------------------------------------------------


def format_detect(obj: Any, jsonl: bool = False) -> str:
    """Name the export format of an already parsed file.

    The vote lives behind the intake seam (sources): every registered adapter votes with
    positive evidence, and the strongest vote names the format, or "unknown" when no adapter
    finds positive evidence."""
    return sources.detect_format(obj, jsonl).winner


def detect_reasons(obj: Any, jsonl: bool = False) -> list[str]:
    """Why the adapters voted as they did, in words; an unknown payload is refused with these."""
    return sources.detect_format(obj, jsonl).reasons


def _decode(payload: bytes) -> tuple[Any, bool]:
    """Parse the stored bytes as one JSON document, or line by line as JSONL; (None, False) when neither."""
    try:
        text = payload.decode("utf-8-sig")  # a BOM is the customer's editor, not a different format
    except UnicodeDecodeError:
        return (None, False)
    try:
        return (json.loads(text), False)
    except json.JSONDecodeError:
        pass
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return (None, False)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return (None, False)
    return (records, True)


def store_raw(path: str | Path, workdir: str | Path) -> RawFile:
    """Copy the customer's file byte for byte into workdir/raw/<sha256>.json and describe it."""
    source = Path(path)
    payload = source.read_bytes()
    raw_hash = hashlib.sha256(payload).hexdigest()
    store = Path(workdir) / "raw"
    store.mkdir(parents=True, exist_ok=True)
    target = store / (raw_hash + ".json")
    if not target.exists():
        shutil.copyfile(source, target)
    parsed, jsonl = _decode(payload)
    return RawFile(
        raw_hash=raw_hash,
        path=str(target),
        format_detected=format_detect(parsed, jsonl) if parsed is not None else "unknown",
        bytes=len(payload),
    )


def raw_path(raw_hash: str, workdir: str | Path) -> Path:
    return Path(workdir) / "raw" / (raw_hash + ".json")


# --- error and truncation marking -----------------------------------------


def classify_error(payload: Any, structured: Optional[dict] = None,
                   ptr: Optional[RawPtr] = None) -> ToolCallError:
    """Put a tool error in the D67 taxonomy, keeping the customer's verbatim payload and encoding.

    `ptr` is where the error message sits in the raw file, so a derived error class can be read back
    against the bytes it came from (D66); it is optional because the rules run on payloads alone.
    """
    encoding = "text"
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped[:1] in ("{", "["):
            try:
                json.loads(stripped)
                encoding = "json"
            except json.JSONDecodeError:
                pass
    elif payload is not None:
        encoding = "json"
    if structured:
        declared = str(structured.get("code") or structured.get("type") or "")
        if declared in ERROR_CLASSES:
            return ToolCallError(class_=declared, payload=payload, encoding=encoding,
                                 classified_by="code", raw_ptr=ptr)
    text = (payload if isinstance(payload, str) else json.dumps(payload, default=str)).lower()
    return ToolCallError(class_=_rule_class(text), payload=payload, encoding=encoding,
                         classified_by="rule", raw_ptr=ptr)


def _rule_class(text: str) -> str:
    """The D67 class whose longest matching phrase is longest; rule order breaks a tie."""
    best_class, best_len = "unknown", 0
    for class_, patterns in _COMPILED_RULES:
        matched = max((len(m.group(0)) for m in (p.search(text) for p in patterns) if m), default=0)
        if matched > best_len:
            best_class, best_len = class_, matched
    return best_class


ERROR_SYSTEM = (
    "You put one tool error payload in a fixed taxonomy. Answer with one JSON object: "
    '{"class": "...", "reason": "..."} where class is one of tool_not_found, invalid_arguments, '
    "permission_denied, business_error, not_found_entity, transient, cancelled, unknown. "
    "Use only the payload given; answer unknown when it does not settle it."
)


def classify_error_llm(model: Model, error: ToolCallError) -> ToolCallError:
    """The second pass of D67 for string-only sources: rules first, the model only on what they left unknown."""
    reply = model.query([
        {"role": "system", "content": ERROR_SYSTEM},
        {"role": "user", "content": json.dumps({"payload": error.payload}, default=str)},
    ])
    proposed = str(_reply_json(reply).get("class") or "")
    if proposed not in ERROR_CLASSES or proposed == "unknown":
        return error
    return error.model_copy(update={"class_": proposed, "classified_by": "llm"})


def detect_truncation(text: Any) -> tuple[bool, Optional[int], Optional[str]]:
    """Spot a cut-off tool result (D95): returns (truncated, visible length, the marker found)."""
    if not isinstance(text, str):
        return (False, None, None)
    trimmed = text.rstrip()
    for marker in CUT_MARKERS:
        if trimmed.endswith(marker):
            return (True, len(text), marker)
    if unparsed_json(text):  # cut mid-object, marker eaten by the log limit
        return (True, len(text), UNPARSED_JSON_MARKER)
    return (False, None, None)


def unparsed_json(value: Any) -> bool:
    """True when a value is still the JSON string it arrived as, because that string does not parse."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped[:1] not in ("{", "["):
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return True
    return False


# --- trace derivation behind the seam ------------------------------------------


def derive_traces(raw_hash: str, workdir: str | Path, model: Optional[Model] = None) -> list[Trace]:
    """Derive Trace records from a stored raw file, writing one grader sidecar per trace.

    The winning adapter behind the intake seam (sources) reads the payload; this function only
    orchestrates (derive, sidecar, rejects). A simulation the records refuse is left out with its
    reason in workdir/rejects, which the gate reads, so one broken message never costs the whole
    file (design section 6, on failure: reject trace with reason)."""
    document, jsonl = _decode(raw_path(raw_hash, workdir).read_bytes())
    decision = sources.detect_format(document, jsonl)
    adapter = sources.by_name(decision.winner)
    if adapter is None:
        expected = ", ".join(sources.display_names())
        raise ValueError(
            f"unknown export format for raw file {raw_hash}; expected {expected}. "
            + "; ".join(decision.reasons)
        )
    if not adapter.maps:
        raise NotImplementedError(adapter.unmapped_message())  # type: ignore[attr-defined]
    environment = adapter.environment(document)
    traces, rejects = [], []
    recordings = list(adapter.recordings(document))
    if not recordings:
        rejects.append({"trace_id": None, "sim_index": None,
                        "reason": "the file declares an empty simulations list, so it holds no run"})
    for sim_index, simulation in enumerate(recordings):
        if not isinstance(simulation, dict):
            rejects.append({"trace_id": None, "sim_index": sim_index,
                            "reason": f"simulation is a {type(simulation).__name__}, not an object"})
            continue
        ctx = sources.MapContext(raw_hash=raw_hash, index=sim_index, environment=environment,
                                 ingest_version=INGEST_VERSION)
        try:
            trace = adapter.to_trace(simulation, ctx)
        except ValidationError as exc:
            rejects.append({"trace_id": str(simulation.get("id") or f"{raw_hash[:12]}-{sim_index}"),
                            "sim_index": sim_index, "reason": _validation_reason(exc)})
            continue
        if model is not None:
            _llm_error_pass(model, trace)
        trace.hash = trace_hash(trace)
        _write_grader(trace, adapter.sidecar(simulation, document), workdir)
        traces.append(trace)
    _write_rejects(raw_hash, rejects, workdir)
    return traces


def _validation_reason(exc: ValidationError) -> str:
    first = (exc.errors() or [{}])[0]
    where = ".".join(str(part) for part in first.get("loc", ()))
    return f"the records refuse this simulation: {first.get('msg', exc)}" + (f" at {where}" if where else "")


def _llm_error_pass(model: Model, trace: Trace) -> None:
    """D67 second pass: only the calls the rules left unknown reach the model, and they get classified_by llm."""
    for call in trace.tool_calls:
        if call.error is not None and call.error.class_ == "unknown" and call.error.classified_by == "rule":
            call.error = classify_error_llm(model, call.error)


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, so it is stable across runs."""
    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)


# --- files on disk, named by content (design section 8) --------------------


def trace_file(trace: Trace, workdir: str | Path) -> Path:
    """Where this Trace is written: named by its content hash, so two files sharing an id cannot collide."""
    return Path(workdir) / "traces" / ((trace.hash or trace_hash(trace)) + ".json")


def grader_file(trace: Trace, workdir: str | Path) -> Path:
    """Where this Trace's grader sidecar is written; same name as the trace, different folder (D66)."""
    return Path(workdir) / "grader" / ((trace.hash or trace_hash(trace)) + ".json")


def rejects_file(raw_hash: str, workdir: str | Path) -> Path:
    return Path(workdir) / "rejects" / (raw_hash + ".json")


def read_rejects(workdir: str | Path, raw_hash: Optional[str] = None) -> list[dict]:
    """The simulations ingest refused, for one raw file or for the whole workdir."""
    folder = Path(workdir) / "rejects"
    files = [rejects_file(raw_hash, workdir)] if raw_hash else sorted(folder.glob("*.json"))
    out = []
    for path in files:
        if path.is_file():
            out.extend(json.loads(path.read_text(encoding="utf-8")))
    return out


def _write_rejects(raw_hash: str, rejects: list[dict], workdir: str | Path) -> None:
    target = rejects_file(raw_hash, workdir)
    if not rejects:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, rejects)


# --- grader sidecar (D66) --------------------------------------------------


def _write_grader(trace: Trace, fields: dict, workdir: str | Path) -> Path:
    """Move the export's answer key out of the trace into its own file beside the trace."""
    target = grader_file(trace, workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(target, {
        "trace_id": trace.trace_id, "trace_hash": trace.hash, "raw_hash": trace.raw_hash,
        "raw_ptr": as_dict(trace.raw_ptr) if trace.raw_ptr else None, "fields": fields,
    })


# --- gate and entry point --------------------------------------------------


def gate_ingest(traces: list[Trace], workdir: str | Path, raw_hash: Optional[str] = None) -> GateResult:
    """Section 6 ingest gate: every tool call has a parseable result or an error, and the grader fields are out."""
    failures: list[str] = []
    calls = errors = truncated = unresolved = unparseable = orphans = reused = 0
    for trace in traces:
        answered = {i for turn in trace.turns if turn.role == "tool" for i in turn.tool_call_ids}
        requested = {call.id for call in trace.tool_calls if call.id}
        by_id: dict = {}
        for call in trace.tool_calls:
            calls += 1
            errors += 1 if call.error else 0
            truncated += 1 if call.truncated else 0
            if call.id:
                by_id.setdefault(call.id, []).append(call)
            named = call.id or call.name
            # `resolved` is per call object, unlike `answered` which is only the set of id strings
            # that appear on some "tool" turn; a reused id can leave this exact call unresolved while
            # its id string does get answered, on a different call (D67, gate_ingest row 8).
            if not call.resolved:
                unresolved += 1
                failures.append(f"{trace.trace_id}: tool call {named} has no result and no error")
            elif unparsed_json(call.result):
                unparseable += 1
                failures.append(f"{trace.trace_id}: tool call {named} has a result that does not parse")
        for call_id, group in by_id.items():
            if len(group) > 1 and any(not earlier.resolved for earlier in group[:-1]):
                reused += 1
                failures.append(
                    f"{trace.trace_id}: tool call id {call_id} was issued again while the earlier "
                    "call with that id was still pending"
                )
        for orphan in sorted(answered - requested):
            orphans += 1
            failures.append(f"{trace.trace_id}: tool result {orphan} answers no recorded call")
        failures += _grader_failures(trace, workdir)
        if trace.hash != trace_hash(trace):
            failures.append(f"{trace.trace_id}: trace hash does not match its content")
    rejects = read_rejects(workdir, raw_hash)
    failures += [f"{r.get('trace_id') or 'file'}: rejected at ingest, {r.get('reason')}" for r in rejects]
    metrics = {"traces": len(traces), "tool_calls": calls, "errors": errors, "truncated": truncated,
               "unresolved": unresolved, "unparseable": unparseable, "orphan_results": orphans,
               "reused_pending_ids": reused, "rejected": len(rejects)}
    return GateResult(stage="ingest", passed=not failures, metrics=metrics, failures=failures)


def _grader_failures(trace: Trace, workdir: str | Path) -> list[str]:
    """The sidecar must exist and belong to this trace, not to another file that reused the id."""
    sidecar = grader_file(trace, workdir)
    if not sidecar.is_file():
        return [f"{trace.trace_id}: grader sidecar missing"]
    try:
        body = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"{trace.trace_id}: grader sidecar does not parse"]
    if body.get("raw_hash") != trace.raw_hash or body.get("trace_id") != trace.trace_id:
        return [f"{trace.trace_id}: grader sidecar belongs to another trace"]
    return []


def write_traces(traces: list[Trace], workdir: str | Path) -> list[Path]:
    """Write one JSON per Trace under workdir/traces, named by the trace's content hash (design section 8)."""
    (Path(workdir) / "traces").mkdir(parents=True, exist_ok=True)
    return [_write_json(trace_file(trace, workdir), as_dict(trace)) for trace in traces]


def _write_json(target: Path, body: Any) -> Path:
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def ingest_file(path: str | Path, workdir: str | Path, model: Optional[Model] = None) -> dict:
    """Store one customer file, derive its Traces, write them, run the gate, print the counts."""
    raw = store_raw(path, workdir)
    traces = derive_traces(raw.raw_hash, workdir, model=model)
    write_traces(traces, workdir)
    gate = gate_ingest(traces, workdir, raw_hash=raw.raw_hash)
    summary = {
        "raw_hash": raw.raw_hash,
        "format": raw.format_detected,
        "runs": len(traces),
        "tool_calls": gate.metrics["tool_calls"],
        "errors": gate.metrics["errors"],
        "truncated": gate.metrics["truncated"],
        "rejected": gate.metrics["rejected"],
        "trace_hashes": [trace.hash for trace in traces],
        "gate": as_dict(gate),
    }
    print(
        f"ingest {raw.format_detected}: {summary['runs']} runs, {summary['tool_calls']} tool calls, "
        f"{summary['errors']} errors, {summary['truncated']} truncated, {summary['rejected']} rejected, "
        f"gate {'pass' if gate.passed else 'fail'}"
    )
    return summary
