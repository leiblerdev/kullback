"""Mines the customer's tools (ToolSig) and world (EntitySchema) out of ingested traces (D68, D70, D72, D73)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional

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


def _fields(obj: Any) -> dict[str, list[str]]:
    """Top-level field name to the types seen for it in one observed result or argument.

    A list result contributes every item's types, not only the first item's (D72 union).
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
    return {"value": [_type_name(obj)]}


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


# --- tools -------------------------------------------------------------------

def _new_acc() -> dict:
    return {"calls": 0, "errors": 0, "traces": [], "args": {}, "results": {},
            "arg_calls": 0, "result_calls": 0, "errors_by_class": {}, "samples": []}


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
            for name, types in _fields(parsed).items():
                _note_field(acc["results"], name, types, trace.trace_id)
            if len(acc["samples"]) < MAX_SAMPLES:
                acc["samples"].append({"args": call.args, "result": _short(parsed)})
    return stats


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


def _decide_kind(sig: ToolSig, model: Optional[Model], samples: list, spec: Optional[dict] = None) -> None:
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
    if sig.effects_observed:  # D68: observed effects beat both the rule and the LLM
        fields = ", ".join(sorted({e.field for e in sig.effects_observed})[:3])
        sig.kind, sig.kind_confidence = "write", "high"
        sig.kind_reason = f"observed effect on {fields}"
        sig.classified_by = "observed"
        sig.unclassified = False


def mine_tools(traces: list[Trace], model: Optional[Model] = None) -> list[ToolSig]:
    """One ToolSig per tool the traces show: schemas as the union of everything observed (D72), kind per D68."""
    stats = _accumulate(traces)
    specs = _declared_specs(traces)
    for name in specs:
        stats.setdefault(name, _new_acc())
    effects = observed_effects(traces)
    sigs = []
    for name in sorted(stats):
        acc = stats[name]
        sig = _build_sig(name, acc, specs.get(name), effects.get(name, []))
        _decide_kind(sig, model, acc["samples"], specs.get(name))
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
    if item_fields or isinstance(visible, list):
        rows = [r for r in visible if isinstance(r, dict)] if isinstance(visible, list) else []
        donor_rows = [r for d in donors if isinstance(d, list) for r in d if isinstance(r, dict)]
        if not rows and donor_rows:
            rows = [{}]
        filled = sorted({"[]." + f for row in rows for f in _fill_row(row, item_fields, donor_rows)})
        result: Any = rows
    else:
        result = dict(visible) if isinstance(visible, dict) else {}
        filled = _fill_row(result, names, [d for d in donors if isinstance(d, dict)])
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
        last: dict[tuple, tuple[int, Any]] = {}
        for index, call in enumerate(trace.tool_calls):
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
                c for c in trace.tool_calls[previous[0] + 1: index]
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


# --- the mine gate (design section 6) ----------------------------------------

def gate_tools(sigs: list[ToolSig]) -> GateResult:
    """Each ToolSig has at least three observed calls or is flagged llm; flag, never synthesize."""
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
                 "writes": sum(1 for s in sigs if s.kind == "write")},
        failures=failures,
    )


# --- the world (D73) ---------------------------------------------------------

def _is_id(name: str) -> bool:
    return name == "id" or name.endswith("_id")


def _plural(token: str) -> str:
    if token.endswith("s"):
        return token
    if token.endswith("y"):
        return token[:-1] + "ies"
    return token + "s"


TOOL_VERBS = {"get", "find", "list", "search", "fetch", "retrieve", "read", "lookup", "load", "show",
              "details", "detail", "info", "information", "all", "by", "for", "the", "a", "my", "current"}


def _noun_of(tool_name: str) -> Optional[str]:
    """The entity a tool name is about, with the verb and the filler words taken off."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", tool_name.lower()) if t]
    nouns = [t for t in tokens if t not in TOOL_VERBS]
    return "_".join(nouns) if nouns else None


def _table_of(tool_name: str, row: dict) -> Optional[str]:
    """The entity a result row is about: an id field whose name the tool name also uses.

    A row whose only id is `id`, which is how support, CRM and ticketing APIs return rows (D52),
    takes its table from the tool name instead of being dropped.
    """
    tokens = set(re.split(r"[^a-z0-9]+", tool_name.lower()))
    ids = [key[:-3] for key in row if isinstance(key, str) and key.endswith("_id") and len(key) > 3]
    for token in ids:
        if token in tokens:
            return _plural(token)
    if len(ids) == 1:
        return _plural(ids[0])
    if not ids and any(key == "id" for key in row):
        noun = _noun_of(tool_name)
        return _plural(noun) if noun else None
    return None


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
        b > a for a, b in zip(numbers, numbers[1:]))


def _counter_like(values: list) -> bool:
    """Whole numbers, enough of them, only going up. Money and prices are floats and are not this."""
    numbers = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return (len(numbers) >= MIN_COUNTER_VALUES and len(numbers) == len(values)
            and all(b > a for a, b in zip(numbers, numbers[1:])))


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
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None or call.result is None:
                continue
            parsed = _parse(call.result)
            rows = [parsed] if isinstance(parsed, dict) else (
                [r for r in parsed if isinstance(r, dict)] if isinstance(parsed, list) else [])
            for row in rows:
                table = _table_of(call.name, row)
                if table is None:
                    continue
                for name, value in row.items():
                    _add_value(store, table, str(name), value)
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
            if _is_id(name):
                pattern = id_pattern(cell["values"])
                if pattern:
                    id_patterns[f"{table}.{name}"] = pattern
    return EntitySchema(tables=sorted(store), columns=columns, id_patterns=id_patterns)
