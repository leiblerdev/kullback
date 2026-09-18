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
    ruling = rule_recordings(raw_hash, decision.winner, recordings, traces, rejects)
    _write_ruling(raw_hash, ruling, workdir)
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


# --- admission per recording -----------------------------------------------


# The Task-eligible share a file must keep for the stage to pass. An invented constant awaiting
# measurement on real corpora: it only says a mostly unusable file stops the build, while one bad
# recording among good ones is set aside and the rest build.
MIN_TASK_ELIGIBLE_SHARE = 0.75

TASK_ELIGIBLE = "task_eligible"
EVIDENCE_ONLY = "evidence_only"
REJECTED = "rejected"

# An explicit end marker that reads as not finished. A recording with no marker at all reads as
# complete when every call resolved: the export simply did not stamp it.
UNFINISHED_ENDS = frozenset({
    "max_steps", "max_errors", "timeout", "error", "cancelled", "truncated", "incomplete",
})


def _trace_problems(trace: Trace) -> dict:
    """The per-recording facts admission rules on: which calls never resolved, which results never
    parsed, which ids were reused while pending, which results answer no call, and the sizes."""
    by_id: dict = {}
    problems: dict = {"calls": 0, "errors": 0, "truncated": 0, "turns": len(trace.turns),
                       "unresolved": [], "unparseable": [], "reused": [], "orphans": []}
    for call in trace.tool_calls:
        problems["calls"] += 1
        problems["errors"] += 1 if call.error else 0
        problems["truncated"] += 1 if call.truncated else 0
        if call.id:
            by_id.setdefault(call.id, []).append(call)
        named = call.id or call.name
        # `resolved` is per call object, unlike `answered` which is only the set of id strings
        # that appear on some "tool" turn; a reused id can leave this exact call unresolved while
        # its id string does get answered, on a different call (D67, gate_ingest row 8).
        if not call.resolved:
            problems["unresolved"].append(named)
        elif unparsed_json(call.result):
            problems["unparseable"].append(named)
    problems["reused"] = _reused_ids(by_id)
    problems["orphans"] = _orphan_results(trace)
    return problems


def _reused_ids(by_id: dict) -> list:
    """Call ids issued again while the earlier call with that id was still pending."""
    return [call_id for call_id, group in by_id.items()
            if len(group) > 1 and any(not earlier.resolved for earlier in group[:-1])]


def _orphan_results(trace: Trace) -> list:
    """Tool results that answer no recorded call."""
    answered = {i for turn in trace.turns if turn.role == "tool" for i in turn.tool_call_ids}
    requested = {call.id for call in trace.tool_calls if call.id}
    return sorted(answered - requested)


def _standing_for(problems: dict, simulation: Any, duplicate: bool) -> tuple[str, str, list[str]]:
    """One recording's standing with its primary reason and every reason that fired.

    Rejected first (not a usable recording at all), then the evidence-only reasons in a fixed
    order, so the per-reason counts stay comparable across files. A recording with no problem
    is task-eligible: complete, every call resolved."""
    if problems["unparseable"]:
        return (REJECTED, "unparseable_result", ["unparseable_result"])
    if problems["turns"] == 0 and problems["calls"] == 0:
        return (EVIDENCE_ONLY, "missing_start", ["missing_start"])
    if duplicate:
        return (EVIDENCE_ONLY, "duplicate", ["duplicate"])
    reasons = []
    termination = simulation.get("termination_reason") if isinstance(simulation, dict) else None
    if termination in UNFINISHED_ENDS:
        reasons.append("unfinished")
    if problems["unresolved"]:
        reasons.append("unresolved_call")
    if problems["reused"]:
        reasons.append("reused_call_id")
    if problems["orphans"]:
        reasons.append("orphan_result")
    if reasons:
        return (EVIDENCE_ONLY, reasons[0], reasons)
    return (TASK_ELIGIBLE, "complete_record", ["complete_record"])


def rule_recordings(raw_hash: str, format_name: str, recordings: list, traces: list[Trace],
                    rejects: list[dict]) -> dict:
    """Rule every recording of one file to a standing, with counts per standing and per reason and
    the outcome and length mix of what was set aside, so a drift toward easy Tasks stays visible."""
    sim_of = {}
    for trace in traces:
        index = trace.raw_ptr.sim_index if trace.raw_ptr else None
        if isinstance(index, int) and 0 <= index < len(recordings):
            sim_of[id(trace)] = recordings[index]
    seen: set[str] = set()
    seen_ids: set[str] = set()
    rows = [_rule_trace_row(trace, sim_of.get(id(trace)), seen, seen_ids) for trace in traces]
    for entry in rejects:
        rows.append(_reject_row(entry))
    counts = {TASK_ELIGIBLE: 0, EVIDENCE_ONLY: 0, REJECTED: 0}
    per_reason: dict = {}
    for row in rows:
        counts[row["standing"]] += 1
        per_reason[row["reason"]] = per_reason.get(row["reason"], 0) + 1
    set_aside = [row for row in rows if row["standing"] != TASK_ELIGIBLE]
    eligible = counts[TASK_ELIGIBLE]
    total = len(rows)
    share = eligible / total if total else 1.0
    return {
        "raw_hash": raw_hash, "format": format_name, "floor": MIN_TASK_ELIGIBLE_SHARE,
        "recordings": rows, "counts": counts, "reasons": per_reason,
        "set_aside": {
            "outcomes": _mix([row["termination"] or "unstated" for row in set_aside]),
            "turns": _mix([_turn_bucket(row["turns"]) for row in set_aside]),
            "tool_calls": _mix([_call_bucket(row["tool_calls"]) for row in set_aside]),
        },
        "eligible_share": share, "passed": share >= MIN_TASK_ELIGIBLE_SHARE,
    }


def _rule_trace_row(trace: Trace, simulation: Any, seen: set[str], seen_ids: set[str]) -> dict:
    """One derived trace's ruling row: its standing with reasons, sizes, and outcome marker."""
    problems = _trace_problems(trace)
    digest = trace.hash or trace_hash(trace)
    # A duplicate is the same recording twice: the same content hash, or the same trace id
    # claimed by two simulations of one file. Pointers name the simulation, so two copies never
    # share a content hash on pointers alone; the id repeat is what catches them.
    duplicate = digest in seen or trace.trace_id in seen_ids
    seen.add(digest)
    seen_ids.add(trace.trace_id)
    standing, reason, reasons = _standing_for(problems, simulation, duplicate)
    return {
        "trace_id": trace.trace_id,
        "sim_index": trace.raw_ptr.sim_index if trace.raw_ptr else None,
        "standing": standing, "reason": reason, "reasons": reasons, "trace_hash": digest,
        "turns": problems["turns"], "tool_calls": problems["calls"],
        "termination": simulation.get("termination_reason")
        if isinstance(simulation, dict) else None,
    }


def _reject_row(entry: dict) -> dict:
    """One refused simulation's ruling row: rejected, with why the records refused it."""
    reason_text = str(entry.get("reason") or "")
    reason = "empty_file" if "empty simulations list" in reason_text else "not_a_recording"
    return {
        "trace_id": entry.get("trace_id"), "sim_index": entry.get("sim_index"),
        "standing": REJECTED, "reason": reason, "reasons": [reason], "trace_hash": None,
        "turns": 0, "tool_calls": 0, "termination": None,
    }


def _mix(values: list) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _turn_bucket(turns: int) -> str:
    if turns <= 0:
        return "0"
    if turns <= 5:
        return "1-5"
    if turns <= 20:
        return "6-20"
    return "21+"


def _call_bucket(calls: int) -> str:
    if calls <= 0:
        return "0"
    if calls <= 3:
        return "1-3"
    return "4+"


def intake_file(raw_hash: str, workdir: str | Path) -> Path:
    return Path(workdir) / "intake" / (raw_hash + ".json")


def aggregate_ruling_file(workdir: str | Path) -> Path:
    return Path(workdir) / "intake_ruling.json"


def _write_ruling(raw_hash: str, ruling: dict, workdir: str | Path) -> Path:
    target = intake_file(raw_hash, workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(target, ruling)


def read_intake_ruling(workdir: str | Path, raw_hash: str) -> dict:
    target = intake_file(raw_hash, workdir)
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}


def _update_aggregate_ruling(workdir: str | Path, raw_hash: str, ruling: dict) -> dict:
    """Fold one file's ruling into the one intake ruling, keyed by raw hash so a repeated ingest
    of the same file overwrites its own entry instead of doubling it."""
    target = aggregate_ruling_file(workdir)
    aggregate = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    files = aggregate.get("files", {})
    files[raw_hash] = ruling
    counts = {TASK_ELIGIBLE: 0, EVIDENCE_ONLY: 0, REJECTED: 0}
    per_reason: dict = {}
    for entry in files.values():
        for standing, count in (entry.get("counts") or {}).items():
            counts[standing] = counts.get(standing, 0) + count
        for reason, count in (entry.get("reasons") or {}).items():
            per_reason[reason] = per_reason.get(reason, 0) + count
    total = sum(counts.values())
    share = counts[TASK_ELIGIBLE] / total if total else 1.0
    aggregate = {"files": files, "counts": counts, "reasons": per_reason,
                 "floor": MIN_TASK_ELIGIBLE_SHARE,
                 "eligible_share": share, "passed": share >= MIN_TASK_ELIGIBLE_SHARE}
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, aggregate)
    return aggregate


def evidence_file(trace: Trace, workdir: str | Path) -> Path:
    """Where a set-aside Trace is written: the content-hash name a trace would have, in a folder
    downstream never reads, so evidence-only recordings are kept and counted but feed nothing yet."""
    return Path(workdir) / "evidence_traces" / ((trace.hash or trace_hash(trace)) + ".json")


def write_evidence(traces: list[Trace], workdir: str | Path) -> list[Path]:
    """Write set-aside Traces under workdir/evidence_traces; nothing downstream reads that folder."""
    (Path(workdir) / "evidence_traces").mkdir(parents=True, exist_ok=True)
    return [_write_json(evidence_file(trace, workdir), as_dict(trace)) for trace in traces]


# --- gate and entry point --------------------------------------------------


def gate_ingest(traces: list[Trace], workdir: str | Path, raw_hash: Optional[str] = None) -> GateResult:
    """Section 6 ingest gate: per-recording standings rule, and the stage fails only under the floor.

    Each recording is task-eligible, evidence-only, or rejected (see _standing_for); the gate keeps
    the per-problem lines below for the report, and passes when no trace is corrupt on disk and the
    task-eligible share reaches MIN_TASK_ELIGIBLE_SHARE. Downstream stages read workdir/traces,
    which ingest_file fills with task-eligible recordings only."""
    traces = list(traces)
    notes: list[str] = []
    hard: list[str] = []
    calls = errors = truncated = unresolved = unparseable = orphans = reused = 0
    for trace in traces:
        problems = _trace_problems(trace)
        calls += problems["calls"]
        errors += problems["errors"]
        truncated += problems["truncated"]
        unresolved += len(problems["unresolved"])
        unparseable += len(problems["unparseable"])
        reused += len(problems["reused"])
        orphans += len(problems["orphans"])
        for named in problems["unresolved"]:
            notes.append(f"{trace.trace_id}: tool call {named} has no result and no error")
        for named in problems["unparseable"]:
            notes.append(f"{trace.trace_id}: tool call {named} has a result that does not parse")
        for call_id in problems["reused"]:
            notes.append(
                f"{trace.trace_id}: tool call id {call_id} was issued again while the earlier "
                "call with that id was still pending"
            )
        for orphan in problems["orphans"]:
            notes.append(f"{trace.trace_id}: tool result {orphan} answers no recorded call")
        hard += _grader_failures(trace, workdir)
        if trace.hash != trace_hash(trace):
            hard.append(f"{trace.trace_id}: trace hash does not match its content")
    rejects = read_rejects(workdir, raw_hash)
    notes += [f"{r.get('trace_id') or 'file'}: rejected at ingest, {r.get('reason')}" for r in rejects]
    standings = _standings_in_scope(traces, workdir, raw_hash)
    eligible = sum(1 for standing, _ in standings if standing == TASK_ELIGIBLE)
    evidence = sum(1 for standing, _ in standings if standing == EVIDENCE_ONLY)
    rejected = sum(1 for standing, _ in standings if standing == REJECTED)
    total = len(standings)
    share = eligible / total if total else 1.0
    floor_ok = share >= MIN_TASK_ELIGIBLE_SHARE
    failures = list(hard)
    if not floor_ok:
        failures.append(
            f"task-eligible share {share:.2f} is under the floor {MIN_TASK_ELIGIBLE_SHARE:.2f}: "
            f"{eligible} of {total} recordings task-eligible "
            f"({eligible} task-eligible, {evidence} evidence-only, {rejected} rejected)"
        )
    if failures:
        failures += notes
    metrics = {"traces": len(traces), "tool_calls": calls, "errors": errors, "truncated": truncated,
               "unresolved": unresolved, "unparseable": unparseable, "orphan_results": orphans,
               "reused_pending_ids": reused, "rejected": rejected,
               "task_eligible": eligible, "evidence_only": evidence,
               "eligible_share": share, "floor": MIN_TASK_ELIGIBLE_SHARE}
    return GateResult(stage="ingest", passed=not failures, metrics=metrics, failures=failures)


def _standings_in_scope(traces: list[Trace], workdir: str | Path,
                        raw_hash: Optional[str] = None) -> list[tuple[str, str]]:
    """Standings for the given traces from the intake rulings in scope, with a content-only
    fallback for traces no ruling covers (hand-built traces carry no simulation to read)."""
    folder = Path(workdir) / "intake"
    if raw_hash is not None:
        target = intake_file(raw_hash, workdir)
        files = [target] if target.is_file() else []
    elif folder.is_dir():
        files = sorted(folder.glob("*.json"))
    else:
        files = []
    by_digest: dict = {}
    reject_rows: list = []
    for path in files:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        file_hash = body.get("raw_hash", "")
        for row in body.get("recordings", []):
            if row.get("trace_hash"):
                by_digest[(file_hash, row["trace_hash"])] = row
                by_digest[("", row["trace_hash"])] = row
            else:
                reject_rows.append(row)
    standings = []
    seen: set[str] = set()
    seen_ids: set[str] = set()
    for trace in traces:
        digest = trace.hash or trace_hash(trace)
        row = by_digest.get((trace.raw_hash, digest)) or by_digest.get(("", digest))
        if row is not None:
            standings.append((row["standing"], row["reason"]))
            continue
        problems = _trace_problems(trace)
        duplicate = digest in seen or trace.trace_id in seen_ids
        seen.add(digest)
        seen_ids.add(trace.trace_id)
        standings.append(_standing_for(problems, None, duplicate)[:2])
    standings.extend([(row["standing"], row["reason"]) for row in reject_rows])
    return standings


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


def _split_writes(traces: list[Trace], ruling: dict) -> tuple[list[Trace], list[Trace]]:
    """Task-eligible traces build downstream; the rest are written aside as evidence. Without a
    ruling every trace builds, which is the old behavior for a derive that wrote none."""
    if not ruling:
        return (list(traces), [])
    eligible_hashes = {row["trace_hash"] for row in ruling.get("recordings", [])
                       if row["standing"] == TASK_ELIGIBLE and row.get("trace_hash")}
    eligible = [trace for trace in traces if (trace.hash or trace_hash(trace)) in eligible_hashes]
    wanted = {id(trace) for trace in eligible}
    return (eligible, [trace for trace in traces if id(trace) not in wanted])


def ingest_file(path: str | Path, workdir: str | Path, model: Optional[Model] = None) -> dict:
    """Store one customer file, derive its Traces, write them, run the gate, print the counts.

    Only task-eligible recordings reach workdir/traces, which is what downstream stages read;
    evidence-only recordings are written beside them under evidence_traces and counted, and nothing
    consumes them yet."""
    raw = store_raw(path, workdir)
    traces = derive_traces(raw.raw_hash, workdir, model=model)
    ruling = read_intake_ruling(workdir, raw.raw_hash)
    eligible, set_aside = _split_writes(traces, ruling)
    write_traces(eligible, workdir)
    if set_aside:
        write_evidence(set_aside, workdir)
    if ruling:
        _update_aggregate_ruling(workdir, raw.raw_hash, ruling)
    gate = gate_ingest(traces, workdir, raw_hash=raw.raw_hash)
    summary = {
        "raw_hash": raw.raw_hash,
        "format": raw.format_detected,
        "runs": len(eligible),
        "tool_calls": gate.metrics["tool_calls"],
        "errors": gate.metrics["errors"],
        "truncated": gate.metrics["truncated"],
        "rejected": gate.metrics["rejected"],
        "task_eligible": gate.metrics["task_eligible"],
        "evidence_only": gate.metrics["evidence_only"],
        "trace_hashes": [trace.hash for trace in eligible],
        "gate": as_dict(gate),
    }
    print(
        f"ingest {raw.format_detected}: {summary['runs']} runs, {summary['tool_calls']} tool calls, "
        f"{summary['errors']} errors, {summary['truncated']} truncated, {summary['rejected']} rejected, "
        f"gate {'pass' if gate.passed else 'fail'}"
    )
    return summary
