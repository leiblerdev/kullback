"""Stores the customer's files byte for byte and content-hashed (D66), then derives Trace records from them with a raw pointer on every field, grader fields stripped into a sidecar, tool errors classed (D67) and truncated results marked (D95)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from harness.shared.provider import Model
from harness.shared.records import (
    GateResult,
    RawFile,
    RawPtr,
    ToolCall,
    ToolCallError,
    Trace,
    Turn,
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

# Benchmark answer keys. Stripped before anything else reads the trace (D66, D89).
GRADER_FIELDS = (
    "reward_info", "evaluation_criteria", "action_checks", "nl_assertions",
    "env_assertions", "reward", "trial", "task_id",
)

# Longest first, so "... (truncated)" is not read as a bare "...".
CUT_MARKERS = ("... (truncated)", "[output truncated]", "[truncated]", "<truncated>", "…", "...")

# A JSON result that does not parse was cut off by the customer's log limit even when no marker survived.
UNPARSED_JSON_MARKER = "unterminated_json"

# Formats section 4 names that format_detect recognizes and no mapper reads yet (the slice is tau2 first, D55).
UNMAPPED_FORMATS = {"otel_genai": "OpenTelemetry GenAI", "claude_code_jsonl": "Claude Code JSONL"}

# D67 classes as regexes over the lowercased payload. Every rule is scored and the longest matched
# phrase wins, with the order below as the tie-break, so a loose fragment cannot beat a specific one.
ERROR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tool_not_found", (
        r"\b(?:unknown|unrecognized|undefined|no such|invalid)\s+tool\b",
        r"\btool\b['\"\s:=]*[\w.\-]*['\"]?\s*(?:was\s+|is\s+)?"
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
    """Name the export format of an already parsed file: tau2 native, OpenTelemetry GenAI, Claude Code JSONL, unknown."""
    if isinstance(obj, list):
        heads = [item for item in obj[:20] if isinstance(item, dict)]
        if any(_looks_otel(item) for item in heads):
            return "otel_genai"
        if jsonl and any(_looks_claude_code(item) for item in heads):
            return "claude_code_jsonl"
        return "unknown"
    if isinstance(obj, dict):
        if isinstance(obj.get("simulations"), list):
            return "tau2_native"
        if isinstance(obj.get("messages"), list) and "id" in obj:
            return "tau2_native"
        if "resourceSpans" in obj or "resource_spans" in obj or _looks_otel(obj):
            return "otel_genai"
    return "unknown"


def _looks_otel(item: dict) -> bool:
    if str(item.get("name", "")).startswith("gen_ai."):
        return True
    attributes = item.get("attributes")
    return isinstance(attributes, dict) and any(str(k).startswith("gen_ai.") for k in attributes)


def _looks_claude_code(item: dict) -> bool:
    if item.get("type") not in ("user", "assistant", "system", "summary", "result"):
        return False
    return any(key in item for key in ("message", "content", "uuid", "sessionId"))


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


def _reply_json(reply: Any) -> dict:
    """The first JSON object in a model reply, or an empty dict when there is none."""
    text = getattr(reply, "content", None) or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


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


# --- tau2 native derivation ------------------------------------------------


def derive_traces(raw_hash: str, workdir: str | Path, model: Optional[Model] = None) -> list[Trace]:
    """Derive Trace records from a stored raw file, writing one grader sidecar per trace.

    A simulation the records refuse is left out with its reason in workdir/rejects, which the gate reads,
    so one broken message never costs the whole file (design section 6, on failure: reject trace with reason).
    """
    document, jsonl = _decode(raw_path(raw_hash, workdir).read_bytes())
    format_detected = format_detect(document, jsonl)
    if format_detected in UNMAPPED_FORMATS:
        raise NotImplementedError(
            f"{UNMAPPED_FORMATS[format_detected]} ingest ({format_detected}) is not written yet; only the tau2 "
            "native export is mapped so far (D55). format_detect names the format so the mapper has a home here."
        )
    if format_detected != "tau2_native":
        raise ValueError(
            f"unknown export format for raw file {raw_hash}; expected tau2 native, "
            "OpenTelemetry GenAI or Claude Code JSONL"
        )
    simulations = document["simulations"] if "simulations" in document else [document]
    tasks = {str(task.get("id")): task for task in document.get("tasks") or [] if isinstance(task, dict)}
    environment = (document.get("info") or {}).get("environment_info") or {}
    traces, rejects = [], []
    if not simulations:
        rejects.append({"trace_id": None, "sim_index": None,
                        "reason": "the file declares an empty simulations list, so it holds no run"})
    for sim_index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            rejects.append({"trace_id": None, "sim_index": sim_index,
                            "reason": f"simulation is a {type(simulation).__name__}, not an object"})
            continue
        try:
            trace = _tau2_trace(simulation, sim_index, raw_hash, environment)
        except ValidationError as exc:
            rejects.append({"trace_id": str(simulation.get("id") or f"{raw_hash[:12]}-{sim_index}"),
                            "sim_index": sim_index, "reason": _validation_reason(exc)})
            continue
        if model is not None:
            _llm_error_pass(model, trace)
        trace.hash = trace_hash(trace)
        _write_grader(simulation, tasks, trace, workdir)
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


def _tau2_trace(simulation: dict, sim_index: int, raw_hash: str, environment: dict) -> Trace:
    messages = simulation.get("messages") or []
    ptr = RawPtr(file_hash=raw_hash, sim_index=sim_index)
    # The info block is not a message, so its pointer names the section instead of a message index;
    # it is where `tools_declared` and `system_prompt` below were read from (D66).
    info = RawPtr(file_hash=raw_hash, sim_index=sim_index, section="info.environment_info")
    trace_id = str(simulation.get("id") or f"{raw_hash[:12]}-{sim_index}")
    turns, calls, pending = [], [], {}
    for msg_index, message in enumerate(messages):
        here = RawPtr(file_hash=raw_hash, sim_index=sim_index, msg_index=msg_index)
        role = message.get("role") or "assistant"
        requested = message.get("tool_calls") or []
        if role == "tool":
            waiting = pending.get(message.get("id"))
            if waiting is not None:
                _attach_result(waiting[0], message, waiting[1], here)
            turns.append(Turn(idx=msg_index, role="tool", content=_text(message.get("content")),
                              tool_call_ids=[message["id"]] if message.get("id") else [], raw_ptr=here))
            continue
        for request in requested:
            call = ToolCall(
                id=request.get("id"),
                name=request.get("name") or "",
                args=request.get("arguments") or {},
                requestor=request.get("requestor") or role,
                raw_ptr=here,
                trace_id=trace_id,
            )
            calls.append(call)
            if call.id:
                pending[call.id] = (call, message.get("timestamp"))
        turns.append(Turn(idx=msg_index, role=role, content=_text(message.get("content")),
                          tool_call_ids=[r.get("id") for r in requested if r.get("id")], raw_ptr=here))
    return Trace(
        trace_id=trace_id,
        raw_hash=raw_hash,
        ingest_version=INGEST_VERSION,
        source="tau2_native",
        turns=turns,
        tool_calls=calls,
        tools_declared=environment.get("tool_defs") if isinstance(environment.get("tool_defs"), list) else None,
        system_prompt=environment.get("policy"),
        tools_declared_ptr=info if environment.get("tool_defs") else None,
        system_prompt_ptr=info if environment.get("policy") else None,
        info_ptr=info,
        raw_ptr=ptr,
    )


def _attach_result(call: ToolCall, message: dict, asked_at: Any, ptr: Optional[RawPtr] = None) -> None:
    """Put the tool message that answered this call on the call, and say where it came from.

    `has_result` is set for every answered call, a recorded JSON null included, because `result is
    None` alone cannot tell a null answer apart from a call whose tool message was never captured
    (validate.ingest_gate reads the flag). `resolved` says the answer landed on this call.
    """
    content = message.get("content")
    call.truncated, call.visible_len, call.cut_marker = detect_truncation(content)
    flag = message.get("error")
    if flag:
        structured = flag if isinstance(flag, dict) else _structured(content)
        call.error = classify_error(content, structured, ptr=ptr)
    else:
        call.result = _parsed(content)
    call.has_result = True
    call.resolved = True
    call.result_ptr = ptr
    call.latency_ms = _latency_ms(asked_at, message.get("timestamp"))


def _structured(content: Any) -> Optional[dict]:
    """A typed error body when the source sends one, so classify_error can use the code instead of the rules."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip()[:1] == "{":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _text(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _parsed(content: Any) -> Any:
    """Tool results arrive as JSON strings in tau2; parse them so mine.py sees fields, keep text as text."""
    if isinstance(content, str) and content.strip()[:1] in ("{", "["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content  # kept verbatim; the gate reports it as not parseable
    return content


def _latency_ms(asked: Any, answered: Any) -> Optional[float]:
    try:
        start = datetime.fromisoformat(str(asked))
        end = datetime.fromisoformat(str(answered))
    except (TypeError, ValueError):
        return None
    return (end - start).total_seconds() * 1000.0


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


def _write_grader(simulation: dict, tasks: dict, trace: Trace, workdir: str | Path) -> Path:
    """Move the benchmark answer key out of the trace into its own file beside the trace."""
    fields = {key: simulation[key] for key in GRADER_FIELDS if key in simulation}
    task = tasks.get(str(simulation.get("task_id")))
    if task and "evaluation_criteria" in task:
        fields["evaluation_criteria"] = task["evaluation_criteria"]
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
    calls = errors = truncated = unresolved = unparseable = orphans = 0
    for trace in traces:
        answered = {i for turn in trace.turns if turn.role == "tool" for i in turn.tool_call_ids}
        requested = {call.id for call in trace.tool_calls if call.id}
        for call in trace.tool_calls:
            calls += 1
            errors += 1 if call.error else 0
            truncated += 1 if call.truncated else 0
            named = call.id or call.name
            if call.error is None and call.result is None and call.id not in answered:
                unresolved += 1
                failures.append(f"{trace.trace_id}: tool call {named} has no result and no error")
            elif unparsed_json(call.result):
                unparseable += 1
                failures.append(f"{trace.trace_id}: tool call {named} has a result that does not parse")
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
               "rejected": len(rejects)}
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
