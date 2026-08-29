"""Mines the customer's tools (ToolSig) and world (EntitySchema) out of ingested traces (D68, D70, D72, D73)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

from harness.shared.provider import Model
from harness.shared.records import (
    Column,
    EffectObservation,
    EntitySchema,
    ErrorShape,
    EvidenceStrength,
    FieldStat,
    GateResult,
    ToolSig,
    Trace,
    canonical_json,
)

READ_PREFIXES = ("get_", "find_", "list_", "search_")
WRITE_PREFIXES = ("cancel_", "modify_", "update_", "return_", "exchange_")
# A name that reads as a calculation or a handoff to a person: it answers, it does not change the world.
GENERIC_NAME = re.compile(r"^(calculate|compute|think|reflect|transfer_to_human|escalate_to_human|hand_?off_to_human)")
MIN_OBSERVED_CALLS = 3
UNKNOWN_ERROR_SHARE = 0.20  # D67: unknown above a small share on any tool is a flag on the Environment
MAX_SAMPLES = 5
MAX_VALUES = 400
MIN_COUNTER_VALUES = 5
JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool",
              "object": "dict", "array": "list", "null": "NoneType"}
# System time and counters only: a name that merely contains a date or a version is not enough (D73).
EXEMPT_TIME_NAME = re.compile(r"(^|_)(created|updated|modified)($|_)|(^|_)(at|ts|timestamp)$")
EXEMPT_COUNTER_NAME = re.compile(r"(^|_)(count|counter|seq|sequence|nonce)($|_)")
# Names that read like a date, a time or a version: business data until the values or a reviewer say otherwise.
SOFT_TIME_NAME = re.compile(r"(^|_)(time|date|version|num)($|_)")
TIMESTAMP_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


class KindProposal(NamedTuple):
    """A proposed read/write class for one tool, from a code rule or from the LLM (D68)."""
    kind: str
    confidence: str
    reason: str
    classified_by: str = "rule"


class ClassProposal(NamedTuple):
    """A proposed class for one column, from a code rule or from the LLM (D73)."""
    column_class: str
    confidence: str
    reason: str
    evidence: dict


# --- small shared helpers ----------------------------------------------------

def _parse(value: Any) -> Any:
    """A tool result as the trace stored it, decoded when it is JSON in a string."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _type_name(value: Any) -> str:
    return type(value).__name__


def _add_type(out: dict[str, list[str]], name: str, type_name: str) -> None:
    types = out.setdefault(name, [])
    if type_name not in types:
        types.append(type_name)


# A bare scalar ("-121.2", 4, true) is not a field of an object; it is the whole result. Naming it
# "value" made a one-column object indistinguishable from a tool that hands back a number, and
# every body the model wrote for such a tool obediently returned {"value": ...} to match, then
# failed replay against the real tool's bare scalar (docs/live-build.md, the calculate miss). The
# marker keeps the D72 union (types, count, first and last seen) without claiming a field name
# that was never on the wire.
SCALAR_RESULT_FIELD = "$scalar"


def _fields(obj: Any) -> dict[str, list[str]]:
    """Top-level field name to the types seen for it in one observed result or argument.

    A list result contributes every item's types, not only the first item's (D72 union). A result
    that is neither a dict nor a list carries no field name of its own; it is recorded under
    SCALAR_RESULT_FIELD so a caller can tell a bare scalar apart from a real single-field object.
    """
    if isinstance(obj, dict):
        return {str(k): [_type_name(v)] for k, v in obj.items()}
    if isinstance(obj, list):
        out: dict[str, list[str]] = {}
        for item in obj:
            if isinstance(item, dict):
                for key, value in item.items():
                    _add_type(out, "[]." + str(key), _type_name(value))
            else:
                _add_type(out, "[]", _type_name(item))
        return out
    return {SCALAR_RESULT_FIELD: [_type_name(obj)]}


def _note_field(store: dict[str, FieldStat], name: str, types: list[str], trace_id: str) -> None:
    """Union rule (D72): one more observation of one field."""
    stat = store.get(name)
    if stat is None:
        stat = store[name] = FieldStat(name=name, first_seen=trace_id)
    for type_name in types:
        if type_name not in stat.types:
            stat.types.append(type_name)
    stat.count += 1
    stat.last_seen = trace_id


def _reply_json(reply: Any) -> dict:
    """The model's reply as a dict, or an empty dict when it is not usable."""
    text = (getattr(reply, "content", None) or "").strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _ask(model: Model, system: str, payload: dict) -> Any:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, default=str)}]
    return model.query(messages)


def is_assistant_call(call: Any) -> bool:
    """Telecom's traces interleave the assistant and the simulated user's own tool calls, the user
    running a separate toolkit against its own phone (docs/cross-domain-check.md, Judgement). Only the
    assistant's calls describe the customer's Environment; a missing requestor is the assistant."""
    return (call.requestor or "assistant") == "assistant"


_is_assistant_call = is_assistant_call


def skipped_user_calls(traces: list[Trace]) -> int:
    """Tool calls whose requestor is not the assistant: never mined into a ToolSig, kind or schema."""
    return sum(1 for trace in traces for call in trace.tool_calls if not _is_assistant_call(call))


# --- tools -------------------------------------------------------------------

def _new_acc() -> dict:
    return {"calls": 0, "errors": 0, "traces": [], "args": {}, "results": {},
            "arg_calls": 0, "result_calls": 0, "errors_by_class": {}, "samples": [],
            "echoed": 0.0, "messages": 0}


def _declared_specs(traces: list[Trace]) -> dict[str, dict]:
    """The `tools` list as sent, when the traces carry one; one more source, not the contract (D72)."""
    specs: dict[str, dict] = {}
    for trace in traces:
        for entry in trace.tools_declared or []:
            spec = entry.get("function") if isinstance(entry.get("function"), dict) else entry
            if isinstance(spec, dict) and spec.get("name"):
                specs.setdefault(spec["name"], spec)
    return specs


def _accumulate(traces: list[Trace]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for trace in traces:
        for call in trace.tool_calls:
            if not _is_assistant_call(call):
                continue
            acc = stats.setdefault(call.name, _new_acc())
            acc["calls"] += 1
            if trace.trace_id not in acc["traces"]:
                acc["traces"].append(trace.trace_id)
            acc["arg_calls"] += 1
            for name, types in _fields(call.args or {}).items():
                _note_field(acc["args"], name, types, trace.trace_id)
            if call.error is not None:
                acc["errors"] += 1
                shape = acc["errors_by_class"].get(call.error.class_)
                if shape is None:
                    acc["errors_by_class"][call.error.class_] = ErrorShape(
                        class_=call.error.class_, count=1,
                        sample_payload=call.error.payload, encoding=call.error.encoding)
                else:
                    shape.count += 1
                continue
            # D95: a cut result is not an observation of the shape; it would add a bogus field and
            # make every real field optional, and the schema is what reconstruction rests on.
            if call.result is None or getattr(call, "truncated", False):
                continue
            parsed = _parse(call.result)
            acc["result_calls"] += 1
            acc["echoed"] += _echoed_share(call.args or {}, parsed)
            acc["messages"] += isinstance(parsed, str)
            for name, types in _fields(parsed).items():
                _note_field(acc["results"], name, types, trace.trace_id)
            if len(acc["samples"]) < MAX_SAMPLES:
                acc["samples"].append({"args": call.args, "result": _short(parsed)})
    return stats


def _leaves(value: Any, out: Optional[list] = None) -> list[str]:
    """Every scalar inside a value, canonically, so two nested shapes can be compared by content."""
    out = [] if out is None else out
    if isinstance(value, dict):
        for item in value.values():
            _leaves(item, out)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, out)
    elif value is not None:
        out.append(canonical_json(value))
    return out


def _echoed_share(args: dict, result: Any) -> float:
    """How much of what came back is what the call sent.

    A read answers with the world: you send an id and the world sends back everything it knows. A
    create answers with what you handed it: airline's `book_reservation` returns the reservation it
    just made out of the origin, destination, cabin, flights, passengers and payment in its own
    arguments. Measured on all three tau2 domains the two do not overlap: `book_reservation` sits at
    0.81 and every read in retail, airline and telecom sits at 0.20 or below.
    """
    values = _leaves(result)
    if not values:
        return 0.0
    sent = set(_leaves(args))
    return sum(1 for value in values if value in sent) / len(values)


def _short(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _merge_declared(sig: ToolSig, spec: Optional[dict], observed_args: dict[str, FieldStat]) -> None:
    if not spec:
        return
    sig.description = sig.description or spec.get("description")
    params = spec.get("parameters") or spec.get("input_schema") or {}
    required = set(params.get("required") or [])
    for name, prop in (params.get("properties") or {}).items():
        stat = observed_args.get(name)
        if stat is None:
            declared_type = JSON_TYPES.get(str(prop.get("type")), "str")
            stat = observed_args[name] = FieldStat(name=name, types=[declared_type], count=0)
            stat.optional = name not in required
        stat.declared = True


def _declared_out(spec: Optional[dict]) -> dict:
    """The declared output schema of a tool, under any of the names the ecosystems use (D72)."""
    if not spec:
        return {}
    out = spec.get("output_schema") or spec.get("outputSchema") or spec.get("returns") or {}
    if isinstance(out, dict) and isinstance(out.get("schema"), dict):
        out = out["schema"]
    return out if isinstance(out, dict) else {}


def _merge_declared_results(spec: Optional[dict], results: dict[str, FieldStat]) -> None:
    """D72 puts `declared` on result fields too: a declared output schema is one more source."""
    out = _declared_out(spec)
    required = set(out.get("required") or [])
    for name, prop in (out.get("properties") or {}).items():
        stat = results.get(name)
        if stat is None:
            declared_type = JSON_TYPES.get(str((prop or {}).get("type")), "str")
            stat = results[name] = FieldStat(name=name, types=[declared_type], count=0)
            stat.optional = name not in required
        stat.declared = True


def _args_schema(fields: list[FieldStat]) -> dict:
    return {
        "type": "object",
        "properties": {f.name: {"type": f.types} for f in fields},
        "required": [f.name for f in fields if not f.optional],
    }


def _build_sig(name: str, acc: dict, spec: Optional[dict], effects: list[EffectObservation]) -> ToolSig:
    args = dict(acc["args"])
    for stat in args.values():
        stat.optional = stat.count < acc["arg_calls"]
    results = dict(acc["results"])
    for stat in results.values():
        stat.optional = stat.count < acc["result_calls"]
    _merge_declared_results(spec, results)
    sig = ToolSig(
        name=name,
        result_schema=list(results.values()),
        effects_observed=effects,
        error_shapes=list(acc["errors_by_class"].values()),
        evidence=list(acc["traces"]),
        evidence_strength=EvidenceStrength(
            call_count=acc["calls"], error_count=acc["errors"], trace_count=len(acc["traces"])),
        source="observed" if acc["calls"] else "declared",
    )
    _merge_declared(sig, spec, args)
    sig.args_fields = list(args.values())
    sig.args_schema = _args_schema(sig.args_fields)
    return sig


def is_scalar_result(sig: ToolSig) -> bool:
    """True when every observed result of this tool was a bare scalar, not an object or a list.

    A tool that sometimes answers with a scalar and sometimes with an object is not this case: the
    D72 union keeps both shapes in result_schema and compile_env describes it as an object, as
    before the marker existed.
    """
    return [f.name for f in sig.result_schema] == [SCALAR_RESULT_FIELD]


def propose_kind(name: str) -> KindProposal:
    """The code rule for read, write or generic: name prefixes, with the evidence said out loud (D68)."""
    for prefix in READ_PREFIXES:
        if name.startswith(prefix):
            return KindProposal("read", "high", f"name prefix {prefix!r} reads")
    for prefix in WRITE_PREFIXES:
        if name.startswith(prefix):
            return KindProposal("write", "high", f"name prefix {prefix!r} writes")
    if GENERIC_NAME.match(name):
        return KindProposal("generic", "medium",
                            "the name reads as a calculation or a handoff to a person, which reads nothing "
                            "of the world and changes nothing in it")
    return KindProposal("read", "low", "no name rule matched, default read and unclassified (D70)")


def annotations_of(spec: Optional[dict]) -> dict:
    """The MCP annotations on a declared tool, when the customer's tools list carries them (D68)."""
    annotations = (spec or {}).get("annotations")
    return annotations if isinstance(annotations, dict) else {}


def _annotation_kind(spec: Optional[dict]) -> Optional[KindProposal]:
    """The customer's own MCP hints as a rule: what they declared about the tool beats a guess at its name."""
    annotations = annotations_of(spec)
    if annotations.get("destructiveHint") is True:
        return KindProposal("write", "medium", "the declared tool carries destructiveHint: true")
    if annotations.get("readOnlyHint") is True:
        return KindProposal("read", "medium", "the declared tool carries readOnlyHint: true")
    if annotations.get("readOnlyHint") is False:
        return KindProposal("write", "medium", "the declared tool carries readOnlyHint: false")
    return None


def kind_evidence(sig: ToolSig, samples: Optional[list] = None,
                  annotations: Optional[dict] = None) -> dict:
    """Everything code can gather about one tool, for the LLM to classify over (D68)."""
    return {
        "name": sig.name,
        "description": sig.description,
        "args_schema": sig.args_schema,
        "result_fields": [{"name": f.name, "types": f.types, "count": f.count} for f in sig.result_schema],
        "error_classes": [e.class_ for e in sig.error_shapes],
        "effects_observed": [{"trace_id": e.trace_id, "field": e.field} for e in sig.effects_observed],
        "annotations": annotations or {},
        "calls": sig.evidence_strength.call_count,
        "traces": sig.evidence_strength.trace_count,
        "samples": samples or [],
    }


KIND_SYSTEM = (
    "You classify one tool of a customer agent as read, write or generic. Answer with one JSON object: "
    '{"kind": "read|write|generic", "confidence": "low|medium|high", "reason": "one sentence"}. '
    "Say low confidence when the evidence does not settle it."
)


def classify_kind(model: Model, tool: ToolSig, evidence: dict) -> Optional[KindProposal]:
    """The LLM hook of D68. Returns None when the reply is not usable, so the code rule stands."""
    data = _reply_json(_ask(model, KIND_SYSTEM, {"tool": kind_evidence(tool), "evidence": evidence}))
    kind, confidence = data.get("kind"), data.get("confidence")
    if kind not in ("read", "write", "generic") or confidence not in ("low", "medium", "high"):
        return None
    return KindProposal(kind, confidence, str(data.get("reason") or ""), "llm")


SCHEMA_SYSTEM = (
    "You propose the result schema of a tool whose results were never observed. Answer with one JSON object: "
    '{"fields": [{"name": "field", "types": ["str"]}]}. Use only the evidence given.'
)


def _llm_result_schema(model: Model, sig: ToolSig, samples: list) -> None:
    data = _reply_json(_ask(model, SCHEMA_SYSTEM, {"tool": kind_evidence(sig, samples)}))
    fields = [f for f in (data.get("fields") or []) if isinstance(f, dict) and f.get("name")]
    if not fields:
        return
    sig.result_schema = [
        FieldStat(name=str(f["name"]), types=[str(t) for t in (f.get("types") or ["str"])], count=0)
        for f in fields
    ]
    sig.source = "llm"


CONFIDENCE_ORDER = ("low", "medium", "high")


def _stronger(a: Optional[str], b: str) -> str:
    return max([a or "low", b], key=lambda c: CONFIDENCE_ORDER.index(c) if c in CONFIDENCE_ORDER else 0)


def _observed_kind(sig: ToolSig, rule: KindProposal, acc: Optional[dict],
                   quiet: Optional[set] = None) -> Optional[KindProposal]:
    """What the recorded calls themselves say about a tool's kind, in order of how much they say.

    Three signals, none of them a verb list. A later read shows a field this call changed (D68). Or
    most of what came back is what the call sent, so the call made the thing rather than found it.
    Or the tool answered with a message instead of data, which a read does not do: the reads that
    return a bare string in these corpora all announce themselves in their names (`find_user_id_by_email`,
    `get_flight_status`), so the message signal is only read where the name rule found nothing.

    The message signal is the weakest of the three and it is the only one held to the corpus floor
    the mine gate already uses: a tool seen once or twice stays unclassified and flagged rather than
    being called a write on its answer shape alone ("flag, never synthesize", design section 6).

    The prefix lists were retail's own verb vocabulary and they miss `book_`, `send_`, `enable_` and
    `refuel_` (docs/cross-domain-check.md). These three signals recover every write the lists miss on
    airline and telecom, and they misclassify nothing on retail.
    """
    if sig.effects_observed:
        fields = ", ".join(sorted({e.field for e in sig.effects_observed})[:3])
        return KindProposal("write", "high", f"observed effect on {fields}", "observed")
    results = (acc or {}).get("result_calls") or 0
    if results and (acc["echoed"] / results) > 0.5:
        return KindProposal("write", "high",
                            f"{acc['echoed'] / results:.0%} of what came back was what the call sent, "
                            "so the call made the thing rather than found it", "observed")
    if results >= MIN_OBSERVED_CALLS and acc["messages"] == results and rule.confidence == "low" \
            and _has_id_argument(sig) and sig.name not in (quiet or set()):
        return KindProposal("write", "medium",
                            "every call answered with a message about a row it was handed rather "
                            "than with data, and no read of that row ever showed it unmoved", "observed")
    return None


def _decide_kind(sig: ToolSig, model: Optional[Model], samples: list, spec: Optional[dict] = None,
                 acc: Optional[dict] = None, quiet: Optional[set] = None) -> None:
    rule = propose_kind(sig.name)
    annotation = _annotation_kind(spec)
    if annotation is not None and rule.confidence != "high":
        rule = annotation
    elif annotation is not None and annotation.kind != rule.kind:
        rule = rule._replace(reason=f"{rule.reason}; the declared annotations disagree: {annotation.reason}")
    sig.kind, sig.kind_confidence, sig.kind_reason = rule.kind, rule.confidence, rule.reason
    sig.classified_by = "rule"
    sig.unclassified = rule.confidence == "low"
    if model is not None:
        llm = classify_kind(model, sig, {"samples": samples, "annotations": annotations_of(spec)})
        if llm is not None and llm.confidence != "low":
            sig.kind, sig.kind_confidence, sig.kind_reason = llm.kind, llm.confidence, llm.reason
            sig.classified_by = "llm"
            sig.unclassified = False
        elif llm is not None:
            sig.kind_reason = f"{sig.kind_reason}; llm low confidence: {llm.reason}"
    observed = _observed_kind(sig, rule, acc, quiet)
    if observed is not None:  # D68: what the calls show beats both the name rule and the LLM
        # Evidence that agrees with what already stands is a confirmation, not a downgrade: it must
        # never lower a confidence the annotations or the LLM had already earned.
        agrees = observed.kind == sig.kind
        sig.kind = observed.kind
        sig.kind_confidence = _stronger(sig.kind_confidence, observed.confidence) if agrees \
            else observed.confidence
        sig.kind_reason = f"{sig.kind_reason}; {observed.reason}" if agrees else observed.reason
        sig.classified_by = "observed"
        sig.unclassified = False


def mine_tools(traces: list[Trace], model: Optional[Model] = None) -> list[ToolSig]:
    """One ToolSig per tool the traces show: schemas as the union of everything observed (D72), kind per D68."""
    stats = _accumulate(traces)
    specs = _declared_specs(traces)
    for name in specs:
        stats.setdefault(name, _new_acc())
    effects = observed_effects(traces)
    quiet = quiet_tools(traces)
    sigs = []
    for name in sorted(stats):
        acc = stats[name]
        sig = _build_sig(name, acc, specs.get(name), effects.get(name, []))
        _decide_kind(sig, model, acc["samples"], specs.get(name), acc, quiet)
        if model is not None and not sig.result_schema and acc["calls"]:
            _llm_result_schema(model, sig, acc["samples"])
        sigs.append(sig)
    return sigs


def unknown_error_flags(sigs: list[ToolSig], threshold: float = UNKNOWN_ERROR_SHARE) -> list[str]:
    """D67: a tool whose errors are `unknown` above a small share is a flag on the Environment.

    The share is over that tool's own observed errors, so one unknown among two errors flags a tool
    the traces barely exercised, which is what the setup review wants to see.
    """
    flags = []
    for sig in sorted(sigs, key=lambda s: s.name):
        errors = sum(shape.count for shape in sig.error_shapes)
        unknown = sum(shape.count for shape in sig.error_shapes if shape.class_ == "unknown")
        if errors and unknown / errors > threshold:
            flags.append(f"{sig.name}: {unknown} of {errors} observed errors are unknown "
                         f"({unknown / errors:.0%}), so the Environment cannot reproduce them by class")
    return flags


def exempt_from_reruns(schema: EntitySchema, rerun_states: list[dict]) -> EntitySchema:
    """D73's correction: a column that varies across successful re-runs with the same outcome is exempt.

    `rerun_states` are the End states of re-runs that reached the same outcome, each shaped
    {table: {row_id: row}}. Observation overrides both the rule and the LLM, and the change is
    recorded with `classified_by: observed`, which is what closes out a wrong `hard` class.
    """
    if len(rerun_states) < 2:
        return schema
    varying = _varying_columns(rerun_states)
    columns = []
    for column in schema.columns:
        if (column.table, column.name) in varying and column.class_ != "exempt":
            column = column.model_copy(update={
                "class_": "exempt", "classified_by": "observed", "class_confidence": "high",
                "class_reason": "the value differs across successful re-runs with the same outcome",
            })
        columns.append(column)
    return schema.model_copy(update={"columns": columns})


def _varying_columns(rerun_states: list[dict]) -> set[tuple[str, str]]:
    """Table and column pairs whose value is not the same in every re-run that holds the row."""
    seen: dict[tuple[str, str, str], set] = {}
    for state in rerun_states:
        for table, rows in (state or {}).items():
            for row_id, row in (rows or {}).items():
                for name, value in (row or {}).items():
                    seen.setdefault((str(table), str(row_id), str(name)), set()).add(canonical_json(value))
    return {(table, name) for (table, _row, name), values in seen.items() if len(values) > 1}


# --- truncated results (D95) -------------------------------------------------

def _cut_json(text: str) -> Any:
    """The complete part of a JSON document that was cut mid-way: brackets closed, the partial pair dropped."""
    stack: list[str] = []
    in_string = escaped = False
    cut: Optional[tuple[int, tuple[str, ...]]] = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if stack:
                stack.pop()
            cut = (index + 1, tuple(stack))
        elif char == ",":
            cut = (index, tuple(stack))
    if cut is None:
        return None
    end, open_brackets = cut
    try:
        return json.loads(text[:end] + "".join(reversed(open_brackets)))
    except ValueError:
        return None


def _visible_part(call: Any) -> Any:
    """What the agent actually saw in a truncated result: parsed whole, or parsed as far as it goes."""
    parsed = _parse(call.result)
    if isinstance(parsed, (dict, list)):
        return parsed
    if isinstance(parsed, str) and parsed.strip()[:1] in ("{", "["):
        return _cut_json(parsed)
    return None


def _donor_results(call: Any, sig: ToolSig, complete_calls: list) -> list:
    """Complete results of the same tool, the calls with the same arguments first (D95)."""
    same_args = canonical_json(getattr(call, "args", None) or {})
    donors = [c for c in complete_calls
              if c is not call and not getattr(c, "truncated", False) and c.name == sig.name
              and c.result is not None]
    donors.sort(key=lambda c: canonical_json(c.args or {}) != same_args)
    return [_parse(c.result) for c in donors]


def _fill_row(row: dict, fields: list[str], donors: list[dict]) -> list[str]:
    """Fill the fields the cut removed from a donor row. An id is never borrowed: it names another entity."""
    filled = []
    for field in fields:
        if field in row or _is_id(field):
            continue
        value = next((d[field] for d in donors if isinstance(d, dict) and field in d), None)
        if value is None:
            continue
        row[field] = value
        filled.append(field)
    return filled


def reconstruct_truncated(call: Any, sig: ToolSig, complete_calls: list) -> Optional[dict]:
    """Rebuild a truncated result from the D72 result schema and complete calls to the same tool.

    D95: the reconstruction keeps the Environment running, it never claims to be what the agent saw.
    The returned dict carries the rebuilt value, the fields it had to invent and the `reconstructed`
    tag, so the caller marks the event Assisted (D49) until the customer supplies the full result.
    Everything the agent did see is kept; only the part the cut removed comes from a donor.
    """
    if not getattr(call, "truncated", False):
        return None
    visible = _visible_part(call)
    donors = _donor_results(call, sig, complete_calls)
    names = sorted(f.name for f in sig.result_schema)
    item_fields = [name[3:] for name in names if name.startswith("[].") and len(name) > 3]
    # The shape to reconstruct is this call's own visible content alone; `item_fields` (read off the
    # tool's whole schema, which may carry list-shaped fields from an unrelated call) only decides
    # which fields fill a row, never which shape a truncated dict result gets rebuilt as.
    if isinstance(visible, list):
        rows = [r for r in visible if isinstance(r, dict)]
        donor_rows = [r for d in donors if isinstance(d, list) for r in d if isinstance(r, dict)]
        if not rows and donor_rows:
            rows = [{}]
        result: Any = rows
        targets, fields, donor_pool, prefix = rows, item_fields, donor_rows, "[]."
    else:
        result = dict(visible) if isinstance(visible, dict) else {}
        targets, fields, donor_pool, prefix = [result], names, [d for d in donors if isinstance(d, dict)], ""
    filled = sorted({prefix + f for row in targets for f in _fill_row(row, fields, donor_pool)})
    return {"result": result, "reconstructed_fields": filled, "tags": ["reconstructed"],
            "assisted": True, "tool": sig.name,
            "visible_len": getattr(call, "visible_len", None),
            "cut_marker": getattr(call, "cut_marker", None)}


# --- observed effects (D68) --------------------------------------------------

def _flat(obj: Any, prefix: str = "", out: Optional[dict] = None, depth: int = 0) -> dict[str, str]:
    out = {} if out is None else out
    if depth > 3 or not isinstance(obj, (dict, list)):
        out[prefix or "value"] = canonical_json(obj)
        return out
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in items:
        _flat(value, f"{prefix}.{key}" if prefix else str(key), out, depth + 1)
    return out


def _shared_key(key: str, other: dict[str, str]) -> str:
    """The field a key belongs to, seen from the other side of the comparison.

    A leaf that became a container (null to a list, tau2's exchange_items) shows up on one side as
    `exchange_items` and on the other as `exchange_items.0`; both are the same changed field.
    """
    parts = key.split(".")
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        if prefix in other or any(k.startswith(prefix + ".") for k in other):
            return prefix
    return parts[0]


def _changed_fields(before: Any, after: Any) -> list[str]:
    left, right = _flat(before), _flat(after)
    changed = {k for k in left if k in right and left[k] != right[k]}
    for key in set(left) ^ set(right):
        changed.add(_shared_key(key, right if key in left else left))
    return sorted(changed)


def _values_at(flat: dict[str, str], field: str) -> set:
    """The canonical values under one changed field, whether it is a leaf or a container."""
    if field in flat:
        return {flat[field]}
    prefix = field + "."
    return {value for key, value in flat.items() if key.startswith(prefix)}


def _exempt_field(field: str) -> bool:
    """A field that moves on its own (a timestamp, a counter) is not evidence that anything wrote (D73)."""
    parts = [p for p in field.split(".") if not p.isdigit()]
    name = (parts[-1] if parts else field).lower()
    return bool(EXEMPT_TIME_NAME.search(name) or EXEMPT_COUNTER_NAME.search(name))


def _scalars(obj: Any, out: Optional[set] = None) -> set:
    """Every scalar in a value, in both the canonical and the plain form the other side may use."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for value in obj.values():
            _scalars(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _scalars(value, out)
    elif obj is not None:
        out.add(canonical_json(obj))  # the form _flat stores, so strings and booleans match too
        out.add(str(obj))
    return out


def _explains(call: Any, read_args: dict, changed_values: set) -> bool:
    """A call can explain a change when it names the same thing the read named, or the new value."""
    rule = propose_kind(call.name)
    if rule.kind != "write" and rule.confidence != "low":
        return False  # a tool the name rule calls a read or generic never takes credit for a write
    values = _scalars(call.args or {})
    return bool(values & _scalars(read_args)) or bool(values & changed_values)


def _credit(changed: list[str], after: dict[str, str], candidates: list) -> dict[str, Any]:
    """One actor per changed field, and none where more than one candidate could explain it (D70).

    A candidate whose arguments carry the new value explains that field alone. What is left goes to a
    single remaining candidate, preferring the ones the name rule already calls writes; where more
    than one is left, the field gets no actor and the tools stay as the rule and the LLM left them.
    """
    credited: dict[str, Any] = {}
    for field in changed:
        values = _values_at(after, field)
        named = [c for c in candidates if values & _scalars(c.args or {})]
        if len(named) == 1:
            credited[field] = named[0]
    left = [f for f in changed if f not in credited]
    if left:
        used = {id(c) for c in credited.values()}
        rest = [c for c in candidates if id(c) not in used]
        pool = [c for c in rest if propose_kind(c.name).kind == "write"] or rest
        if len(pool) == 1:
            for field in left:
                credited[field] = pool[0]
    return credited


def observed_effects(traces: list[Trace]) -> dict[str, list[EffectObservation]]:
    """A later read of the same thing shows a changed field, and one earlier call explains it (D68)."""
    out: dict[str, list[EffectObservation]] = {}
    seen: set = set()
    for trace in traces:
        # Only the assistant's own calls can be evidence of, or credited with, a change to the
        # Environment; the simulated user's own actions must not pollute what the agent is seen to
        # have written (docs/cross-domain-check.md, Judgement).
        calls = [c for c in trace.tool_calls if _is_assistant_call(c)]
        last: dict[tuple, tuple[int, Any]] = {}
        for index, call in enumerate(calls):
            if call.error is not None or call.result is None or getattr(call, "truncated", False):
                continue  # a cut result is not evidence of a change (D95)
            parsed = _parse(call.result)
            key = (call.name, canonical_json(call.args))
            previous = last.get(key)
            last[key] = (index, parsed)
            if previous is None:
                continue
            changed = [f for f in _changed_fields(previous[1], parsed) if not _exempt_field(f)]
            if not changed:
                continue
            after = _flat(parsed)
            changed_values = set().union(*(_values_at(after, field) for field in changed))
            candidates = [
                c for c in calls[previous[0] + 1: index]
                if c.name != call.name and c.error is None
                and _explains(c, call.args or {}, changed_values)
            ]
            if not candidates:
                continue
            for field, actor in sorted(_credit(changed, after, candidates).items()):
                mark = (actor.name, trace.trace_id, f"{call.name}.{field}")
                if mark in seen:
                    continue
                seen.add(mark)
                out.setdefault(actor.name, []).append(EffectObservation(
                    trace_id=trace.trace_id, field=f"{call.name}.{field}",
                    note=f"changed between two identical {call.name} calls"))
    return out


def quiet_tools(traces: list[Trace]) -> set[str]:
    """Tools that two identical reads bracketed with nothing changed in between.

    The complement of `observed_effects` and the same evidence read the other way: if the world was
    read before and after a call and it did not move, that call is evidence against a write. It is
    what stops the weakest write signal from firing on a tool the corpus has already shown to be
    quiet, and it is why a tool can be answered for without any verb list at all.
    """
    quiet: set[str] = set()
    for trace in traces:
        calls = [c for c in trace.tool_calls if _is_assistant_call(c)]
        last: dict[tuple, tuple[int, Any]] = {}
        for index, call in enumerate(calls):
            if call.error is not None or call.result is None or getattr(call, "truncated", False):
                continue
            parsed = _parse(call.result)
            key = (call.name, canonical_json(call.args))
            previous = last.get(key)
            last[key] = (index, parsed)
            if previous is None:
                continue
            if [f for f in _changed_fields(previous[1], parsed) if not _exempt_field(f)]:
                continue
            # Only a read that is about what the call named is evidence about that call, and
            # "about" means what the read was asked for, never what its answer happened to mention.
            # Airline's `send_certificate` touches a user; a reservation names its user in its body,
            # so reading a reservation twice was enough to call the certificate quiet and leave it
            # mined as a read. This mirrors `_explains`, which credits a change the same way.
            named = _scalars(call.args or {})
            quiet.update(c.name for c in calls[previous[0] + 1: index]
                         if c.name != call.name and c.error is None
                         and _scalars(c.args or {}) & named)
    return quiet


def _has_id_argument(sig: ToolSig) -> bool:
    """The tool is handed something that names a row, so its message is about that row."""
    return any(_is_id(field.name) for field in (sig.args_fields or []))


# --- the mine gate (design section 6) ----------------------------------------

def gate_tools(sigs: list[ToolSig], traces: Optional[list[Trace]] = None) -> GateResult:
    """Each ToolSig has at least three observed calls or is flagged llm; flag, never synthesize.

    `traces` is optional and only feeds the `skipped_user_calls` metric (the simulated user's own
    tool calls, mined nowhere); passing it costs nothing when there are none to skip (D66, D68).
    """
    failures = []
    for sig in sigs:
        calls = sig.evidence_strength.call_count
        if calls < MIN_OBSERVED_CALLS and sig.source != "llm":
            failures.append(f"{sig.name}: {calls} observed calls, fewer than {MIN_OBSERVED_CALLS}, not flagged llm")
    return GateResult(
        stage="mine",
        passed=not failures,
        metrics={"tools": len(sigs), "thin": len(failures),
                 "unclassified": sum(1 for s in sigs if s.unclassified),
                 "writes": sum(1 for s in sigs if s.kind == "write"),
                 "skipped_user_calls": skipped_user_calls(traces) if traces is not None else 0},
        failures=failures,
    )


# --- the world (D73) ---------------------------------------------------------

def _is_id(name: str) -> bool:
    """The name rule for an id column: the customer wrote `id` or `<entity>_id`.

    A name rule alone is retail's own convention (docs/cross-domain-check.md). It is kept because it
    is free and right whenever it fires; what the corpus finds on top of it is `id_columns`.
    """
    return name == "id" or name.endswith("_id")


# The suffixes a customer puts on an id column that is not called `_id`. Stripping one gives the
# entity the column names, which is what a table name is matched against.
ID_SUFFIXES = ("_id", "_number", "_no", "_code", "_key", "_ref", "_uuid")


def _entity_of(column: str) -> str:
    """The entity an id column names: `bill_id` is a bill, `flight_number` is a flight."""
    for suffix in ID_SUFFIXES:
        if column.endswith(suffix) and len(column) > len(suffix):
            return column[: -len(suffix)]
    return column


def _plural(token: str) -> str:
    if token.endswith("s"):
        return token
    if token.endswith("y"):
        return token[:-1] + "ies"
    return token + "s"


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ss") or token.endswith("us") or not token.endswith("s"):
        return token
    return token[:-1]


TOOL_VERBS = {"get", "find", "list", "search", "fetch", "retrieve", "read", "lookup", "load", "show",
              "details", "detail", "info", "information", "all", "by", "for", "the", "a", "my", "current"}

# A preposition ends the part of a tool name that says what the tool is about and begins the part
# that says how it is addressed: `get_bills_for_customer` is about bills, not about customers.
PREPOSITIONS = ("for", "by", "of", "from", "with", "to", "in", "at", "on")


def _noun_of(tool_name: str) -> Optional[str]:
    """The entity a tool name is about, singular, with the verb, the filler and any `for x` taken off.

    Reading the whole name filed every telecom bill under `customers`, because `customer` is a token
    of `get_bills_for_customer` and the tie-break took the first id token it recognised. What the
    tool is about is the noun run before the first preposition, and the last word of that run is the
    head of it: `search_direct_flight` is about a flight, not about a direct.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", tool_name.lower()) if t]
    head: list[str] = []
    for token in tokens:
        if token in PREPOSITIONS:
            break
        head.append(token)
    nouns = [t for t in head if t not in TOOL_VERBS]
    return _singular(nouns[-1]) if nouns else None


def _table_of(tool_name: str, row: dict, id_names: Sequence[str] = (), siblings: Sequence[dict] = ()) -> Optional[str]:
    """The entity a result row is about, from the id columns it carries and the tool's own noun.

    In order: the id whose entity is what the tool is about; the id whose values are distinct across
    the rows this one came back with, which is what an id does and a foreign key does not; the only
    id there is. A row whose only id is `id`, which is how support, CRM and ticketing APIs return
    rows (D52), takes its table from the tool name instead of being dropped.
    """
    ids = [key for key in row if isinstance(key, str) and key != "id"
           and (key.endswith("_id") and len(key) > 3 or key in id_names)]
    noun = _noun_of(tool_name)
    for key in ids:
        if _entity_of(key) == noun:
            return _plural(_entity_of(key))
    if len(ids) > 1:
        distinct = [key for key in ids if _distinct_across(key, siblings)]
        if len(distinct) == 1:
            return _plural(_entity_of(distinct[0]))
    if len(ids) == 1:
        return _plural(_entity_of(ids[0]))
    if not ids and any(key == "id" for key in row):
        return _plural(noun) if noun else None
    return None


def _distinct_across(column: str, rows: Sequence[dict]) -> bool:
    """True when every row carrying this column carries a different value for it."""
    values = [canonical_json(row[column]) for row in rows if isinstance(row, dict) and column in row]
    return len(values) > 1 and len(set(values)) == len(values)


def id_columns(traces: list[Trace]) -> set[str]:
    """Column names the corpus shows behaving like an id, whatever the customer called them.

    Two things at once, and both are needed. The column is used to address a row: some call passes
    it as an argument, which is what makes it an id rather than a value. And wherever a result came
    back with several rows carrying the column, every one of those rows had a different value for
    it, which is what an id does and what a foreign key, a status or a price does not.

    No threshold and no vocabulary, so nothing here is fitted to a domain. It is what recovers
    airline's `flights`: `flight_number` addresses a flight in 30 calls and is distinct in every
    search result, and the `_id` name rule alone never sees it, so the table is never proposed
    despite 338 calls that show its rows (docs/cross-domain-check.md).
    """
    addressed = {str(name) for trace in traces for call in trace.tool_calls
                 for name in (call.args or {}) if _is_assistant_call(call)}
    distinct: dict[str, bool] = {}
    for trace in traces:
        for call in trace.tool_calls:
            if not _is_assistant_call(call) or call.error is not None or call.result is None:
                continue
            rows = _result_rows(_parse(call.result))
            if len(rows) < 2:
                continue
            for name in addressed:
                if not any(name in row for row in rows):
                    continue
                distinct[name] = distinct.get(name, True) and _distinct_across(name, rows)
    return {name for name, ok in distinct.items() if ok} | {n for n in addressed if _is_id(n)}


def nested_rows(traces: list[Trace], id_names: Sequence[str] = ()) -> list[tuple[str, str, str, dict]]:
    """(child table, parent table, column, row) for every row stored inside another row's column.

    The signal is structural and needs no vocabulary: a column holding a dict whose values are
    dicts, each keyed by the value of its own id column. Retail's get_product_details answers with
    a product whose `variants` holds `{item_id: {item_id, price, ...}}`; get_item_details answers
    with the same rows on their own. Read only from the outside, the first live build filed them
    as a top-level `items` table, and on the customer's real database, where no such table exists,
    every lookup raised "not found" (docs/live-build.md, schema_shape). The nesting was in the
    traces the whole time; this is where it is read.
    """
    out: list[tuple[str, str, str, dict]] = []
    for trace in traces:
        for call in trace.tool_calls:
            if not _is_assistant_call(call) or call.error is not None or call.result is None:
                continue
            rows = _result_rows(_parse(call.result))
            for row in rows:
                parent = _table_of(call.name, row, id_names, rows)
                if parent is None:
                    continue
                for column, value in row.items():
                    if not isinstance(value, dict) or not value:
                        continue
                    if not all(isinstance(inner, dict) for inner in value.values()):
                        continue
                    for key, inner in value.items():
                        child = _keyed_by_own_id(str(key), inner, id_names)
                        if child and child != parent:
                            out.append((child, parent, str(column), inner))
    return out


def _keyed_by_own_id(key: str, row: dict, id_names: Sequence[str] = ()) -> Optional[str]:
    """The table a nested row belongs to when the key it sits under is its own id; else None."""
    for name, value in row.items():
        is_id = isinstance(name, str) and (name.endswith("_id") and len(name) > 3 or name in id_names)
        if is_id and isinstance(value, str) and value == key:
            return _plural(_entity_of(name))
    return None


def nested_homes(traces: list[Trace], id_names: Sequence[str] = ()) -> dict[str, dict[str, int]]:
    """Child table -> {"parent.column": rows seen there}, from `nested_rows`."""
    homes: dict[str, dict[str, int]] = {}
    for child, parent, column, _ in nested_rows(traces, id_names):
        place = homes.setdefault(child, {})
        place[f"{parent}.{column}"] = place.get(f"{parent}.{column}", 0) + 1
    return homes


def _result_rows(parsed: Any) -> list[dict]:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    return []


def _add_value(store: dict, table: str, name: str, value: Any) -> None:
    column = store.setdefault(table, {}).setdefault(name, {"count": 0, "values": []})
    column["count"] += 1
    if len(column["values"]) < MAX_VALUES:
        column["values"].append(value)


def _rows_of_db(blob: Any) -> list[dict]:
    if isinstance(blob, dict):
        return [row for row in blob.values() if isinstance(row, dict)]
    if isinstance(blob, list):
        return [row for row in blob if isinstance(row, dict)]
    return []


def _monotonic(values: list) -> bool:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return len(numbers) >= 3 and len(numbers) == len(values) and all(
        b > a for a, b in zip(numbers, numbers[1:], strict=False))


def _counter_like(values: list) -> bool:
    """Whole numbers, enough of them, only going up. Money and prices are floats and are not this."""
    numbers = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return (len(numbers) >= MIN_COUNTER_VALUES and len(numbers) == len(values)
            and all(b > a for a, b in zip(numbers, numbers[1:], strict=False)))


def propose_column_class(table: str, name: str, values: list, count: Optional[int] = None) -> ClassProposal:
    """The code rule of D73: timestamps and counters exempt, ids and enums hard, long strings semantic."""
    sample = list(values[:MAX_VALUES])
    present = [v for v in sample if v is not None]  # a nullable column is still an enum
    texts = [v for v in present if isinstance(v, str)]
    lengths = [len(t) for t in texts]
    evidence = {
        "count": len(values) if count is None else count,
        "sampled": len(sample),
        "distinct": len({canonical_json(v) for v in sample}),
        "max_len": max(lengths) if lengths else 0,
        "types": sorted({_type_name(v) for v in sample}),
        "monotonic": _monotonic(present),
    }
    lower = name.lower()
    if _is_id(name):
        return ClassProposal("hard", "high", "column name looks like an id", evidence)
    if EXEMPT_TIME_NAME.search(lower) or EXEMPT_COUNTER_NAME.search(lower):
        return ClassProposal("exempt", "high", "column name reads as a system timestamp or a counter", evidence)
    if SOFT_TIME_NAME.search(lower):
        # A birth date, a delivery date or a version is business data a Candidate must not corrupt,
        # so only the value shape may excuse it from comparison, and then only for review to confirm.
        if texts and len(texts) == len(present) and all(TIMESTAMP_VALUE.match(t) for t in texts):
            return ClassProposal("exempt", "medium", "the name reads as a time and every value is a timestamp",
                                 evidence)
        return ClassProposal("hard", "low", "the name reads like a date or a version but the values are not "
                                            "system timestamps, so it is compared until review says otherwise",
                             evidence)
    if _counter_like(present):
        return ClassProposal("exempt", "low", "whole numbers that only increase, so it may be a counter; "
                                              "low confidence so the review sees it", evidence)
    if texts and len(texts) == len(present):
        if evidence["max_len"] > 60 and evidence["distinct"] > 3:
            return ClassProposal("semantic", "medium", "long free text with many distinct values", evidence)
        if evidence["distinct"] <= 12 and evidence["max_len"] <= 40:
            return ClassProposal("hard", "high", "short strings from a small set, so an enum", evidence)
    return ClassProposal("hard", "low", "no rule matched, defaulting to hard for review", evidence)


COLUMN_SYSTEM = (
    "You verify the class of one column of a customer's world: exempt (never compared), hard (compared exactly) "
    'or semantic (compared by meaning). Answer with one JSON object: {"class": "exempt|hard|semantic", '
    '"confidence": "low|medium|high", "reason": "one sentence"}. Confirm the proposed class unless it is wrong.'
)


def classify_column(model: Model, table: str, name: str, proposal: ClassProposal,
                    samples: Optional[list] = None) -> Optional[ClassProposal]:
    """The LLM hook of D73: it sees the column, the proposed class, the evidence and samples."""
    payload = {"table": table, "column": name, "proposed_class": proposal.column_class,
               "proposed_confidence": proposal.confidence, "proposed_reason": proposal.reason,
               "evidence": proposal.evidence, "samples": _samples(samples or [])}
    data = _reply_json(_ask(model, COLUMN_SYSTEM, payload))
    column_class, confidence = data.get("class"), data.get("confidence")
    if column_class not in ("exempt", "hard", "semantic") or confidence not in ("low", "medium", "high"):
        return None
    return ClassProposal(column_class, confidence, str(data.get("reason") or ""), proposal.evidence)


def _samples(values: list) -> list:
    out = []
    for value in values[:MAX_SAMPLES]:
        out.append(_short(value, 120) if isinstance(value, (dict, list)) else value)
    return out


def _lit(char: str) -> str:
    return char if char.isalnum() or char in "_-#@" else re.escape(char)


def _collapse(atoms: list[str]) -> str:
    out = []
    for atom in atoms:
        if out and out[-1][0] == atom:
            out[-1][1] += 1
        else:
            out.append([atom, 1])
    return "".join(a if n == 1 else f"{a}{{{n}}}" for a, n in out)


def _runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for char in text:
        kind = "d" if char.isdigit() else "a" if char.isalpha() and char.isascii() else "o"
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + char)
        else:
            runs.append((kind, char))
    return runs


def _loose_shape(text: str) -> str:
    parts = []
    for kind, run in _runs(text):
        parts.append(r"\d+" if kind == "d" else "[A-Za-z]+" if kind == "a" else "".join(_lit(c) for c in run))
    return "".join(parts)


def _shape_pattern(texts: list[str]) -> Optional[str]:
    lengths = {len(t) for t in texts}
    if len(lengths) == 1:
        atoms = []
        for position in range(lengths.pop()):
            chars = {t[position] for t in texts}
            if len(chars) == 1:
                atoms.append(_lit(chars.pop()))
            elif all(c.isdigit() for c in chars):
                atoms.append(r"\d")
            elif all(c.isalpha() and c.isascii() for c in chars):
                atoms.append("[A-Za-z]")
            else:
                atoms.append(".")
        return "^" + _collapse(atoms) + "$"
    shapes = {_loose_shape(t) for t in texts}
    return "^" + shapes.pop() + "$" if len(shapes) == 1 else None


def id_pattern(values: list) -> Optional[str]:
    """A regex for an id column, from the shape its values share; None when they share nothing.

    The pattern is read off a sample and then checked against every value, because `canon.py`
    fullmatches real ids against it and a shape that first appears late must not be rejected.
    """
    texts = [v for v in values if isinstance(v, str) and v]
    if not texts:
        return None
    pattern = _shape_pattern(texts[:200])
    if pattern and all(re.fullmatch(pattern, t) for t in texts):
        return pattern
    others = sorted({c for t in texts for c in t if not (c.isalnum() and c.isascii())})
    wide = "^[A-Za-z0-9" + "".join(re.escape(c) for c in others) + "]+$"
    return wide if all(re.fullmatch(wide, t) for t in texts) else None


def mine_schema(traces: list[Trace], db_json_path: Optional[Path] = None,
                model: Optional[Model] = None) -> EntitySchema:
    """Tables and columns from observed tool results and from a given db.json, classified per D73."""
    store: dict[str, dict] = {}
    if db_json_path is not None:
        blob = json.loads(Path(db_json_path).read_text(encoding="utf-8"))
        for table, table_blob in (blob or {}).items():
            for row in _rows_of_db(table_blob):
                for name, value in row.items():
                    _add_value(store, str(table), str(name), value)
    id_names = id_columns(traces)
    for trace in traces:
        for call in trace.tool_calls:
            if not _is_assistant_call(call):
                continue
            if call.error is not None or call.result is None:
                continue
            rows = _result_rows(_parse(call.result))
            for row in rows:
                table = _table_of(call.name, row, id_names, rows)
                if table is None:
                    continue
                for name, value in row.items():
                    _add_value(store, table, str(name), value)
    # A table the corpus also stores inside another table's rows: its nested sightings are
    # sightings of its columns too, and the place seen most often is recorded as its home.
    homes: dict[str, str] = {}
    for child, places in nested_homes(traces, id_names).items():
        if child in store:
            homes[child] = max(sorted(places), key=places.get)
    for child, _parent, _column, row in nested_rows(traces, id_names):
        if child in homes:
            for name, value in row.items():
                _add_value(store, child, str(name), value)
    columns, id_patterns = [], {}
    for table in sorted(store):
        for name in sorted(store[table]):
            cell = store[table][name]
            proposal = propose_column_class(table, name, cell["values"], count=cell["count"])
            column = Column(table=table, name=name, class_=proposal.column_class,
                            class_rule=proposal.column_class, class_confidence=proposal.confidence,
                            class_reason=proposal.reason, classified_by="rule",
                            evidence=proposal.evidence, samples=_samples(cell["values"]))
            if model is not None:
                verified = classify_column(model, table, name, proposal, cell["values"])
                if verified is not None:
                    column.class_ = verified.column_class
                    column.class_confidence = verified.confidence
                    column.class_reason = verified.reason
                    column.classified_by = "llm"
            columns.append(column)
            if _is_id(name) or name in id_names:
                pattern = id_pattern(cell["values"])
                if pattern:
                    id_patterns[f"{table}.{name}"] = pattern
    return EntitySchema(tables=sorted(store), columns=columns, id_patterns=id_patterns, homes=homes)
