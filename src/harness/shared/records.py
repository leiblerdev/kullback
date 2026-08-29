"""Every record the Harness passes between modules, as Pydantic v2 models that round-trip through JSON."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

ErrorClass = Literal[
    "tool_not_found", "invalid_arguments", "permission_denied", "business_error",
    "not_found_entity", "transient", "cancelled", "unknown",
]
ClassifiedBy = Literal["code", "rule", "llm", "observed", "human"]
ToolKind = Literal["read", "write", "generic"]
Confidence = Literal["low", "medium", "high"]
ColumnClass = Literal["exempt", "hard", "semantic"]
AtomKind = Literal["required", "allowed", "forbidden", "question", "communicate", "hard"]
ProvenanceClass = Literal["user_stated", "system_derived", "user_elicited", "agent_chosen"]
EventType = Literal["model_call", "tool_call", "tool_result", "user_turn", "error", "stop"]
Route = Literal["code", "recording", "llm"]
VerdictClass = Literal["pass", "fail", "transferred_without_acting", "env_error", "not_verdicted"]
Cause = Literal["candidate", "environment", "simulated_user", "undetermined"]
SigSource = Literal["observed", "llm", "declared"]
Role = Literal["system", "assistant", "user", "tool"]


def _hashable(value: Any) -> Any:
    """What JSON cannot take, in a form that is the same in every process.

    str() was the old fallback, and a set's str() follows the interpreter's hash
    randomization, so the same set hashed differently in every run and content
    addressing broke. Anything not listed here raises rather than hash unstably.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, (set, frozenset)):
        # The elements' own canonical JSON, sorted: one order whatever the set's order.
        return sorted(json.dumps(item, sort_keys=True, default=_hashable) for item in value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"content_hash cannot hash a {type(value).__name__} stably")


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: model aliases applied, keys sorted, no spare whitespace."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json", by_alias=True)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_hashable)


def content_hash(obj: Any) -> str:
    """sha256 of the canonical JSON of a record, dict, list or scalar."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def as_dict(obj: BaseModel) -> dict:
    """JSON-ready dict of a record, with aliases (class, pass) applied."""
    return obj.model_dump(mode="json", by_alias=True)


class Record(BaseModel):
    """Base for every record: aliases work both ways so `class` and `pass` survive a round trip.

    Unknown keys are refused, not dropped: a stored file written by an older or misspelled
    schema fails at load instead of losing the field silently.
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


def _as_text_payload(payload: Any, encoding: str) -> tuple[Any, str]:
    """A verbatim payload that JSON can hold. Bytes become base64 and say so (D67)."""
    if isinstance(payload, (bytes, bytearray)):
        return base64.b64encode(bytes(payload)).decode("ascii"), "base64"
    return payload, encoding


class RawPtr(Record):
    """Where a derived field came from in the customer's stored file (D66).

    `section` names a part of the file that is not a message, so a field taken from the export's
    own `info` block (the system prompt, the declared tool list) can still cite where it came
    from; a pointer with a `section` and no `msg_index` is not inside the transcript.
    """
    file_hash: str
    sim_index: Optional[int] = None
    msg_index: Optional[int] = None
    section: Optional[str] = None

# --- raw store and traces ---

class RawFile(Record):
    """A customer file, stored byte for byte and content-hashed; never modified, never committed."""
    raw_hash: str
    path: str
    format_detected: str
    bytes: int = 0  # byte count; the content itself stays on disk at `path`


class ToolCallError(Record):
    """A tool error in the D67 taxonomy, keeping the customer's verbatim payload and encoding."""
    class_: ErrorClass = Field(alias="class")
    payload: Any = None
    encoding: str = "text"
    classified_by: ClassifiedBy = "code"
    raw_ptr: Optional[RawPtr] = None

    @model_validator(mode="after")
    def _encode_payload(self) -> "ToolCallError":
        self.payload, self.encoding = _as_text_payload(self.payload, self.encoding)
        return self


class ToolCall(Record):
    """One tool call and its result as the trace recorded it."""
    id: Optional[str] = None
    name: str
    args: dict = Field(default_factory=dict)
    result: Any = None
    requestor: str = "assistant"
    latency_ms: Optional[float] = None
    error: Optional[ToolCallError] = None
    truncated: bool = False
    visible_len: Optional[int] = None
    cut_marker: Optional[str] = None
    raw_ptr: RawPtr  # where the request for this call sits in the raw file; every derived field cites its raw (D66)
    result_ptr: Optional[RawPtr] = None  # where the tool message that answered this call sits in the raw file (D66)
    trace_id: Optional[str] = None  # which Trace recorded it, so a call knows its own Task's world (D74)
    # None means no one recorded either way, which is what lets validate.py tell a call whose tool
    # message was never captured apart from a tool that answered with a JSON null (D66).
    has_result: Optional[bool] = None  # a tool message answered this call, a recorded JSON null included
    resolved: bool = False  # the answer landed on this call; false where the trace shows none


class Turn(Record):
    """One message in a Run's transcript."""
    idx: int
    role: Role
    content: Optional[str] = None
    tool_call_ids: list[str] = Field(default_factory=list)
    raw_ptr: RawPtr  # the message this turn was derived from (D66)


class Trace(Record):
    """The captured record of one Run, derived from a RawFile, grader fields already stripped (D66)."""
    trace_id: str
    raw_hash: str
    ingest_version: str
    source: str
    turns: list[Turn] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tools_declared: Optional[list[dict]] = None
    tools_declared_ptr: Optional[RawPtr] = None
    system_prompt: Optional[str] = None
    system_prompt_ptr: Optional[RawPtr] = None
    info_ptr: Optional[RawPtr] = None  # the export's own info block, which is where the two above come from
    hash: str = ""
    raw_ptr: RawPtr  # the simulation this Trace was derived from (D66)

# --- mined signatures and schema ---

class FieldStat(Record):
    """One field of an argument or result schema, as the union of everything observed (D72)."""
    name: str
    types: list[str] = Field(default_factory=list)
    count: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    declared: bool = False
    optional: bool = True


class EffectObservation(Record):
    """Evidence that a call changed the world: a later read showed this field different (D68)."""
    trace_id: str
    field: str
    note: Optional[str] = None


class ErrorShape(Record):
    """An error class observed on one tool, with a sample payload in the customer's encoding."""
    class_: ErrorClass = Field(alias="class")
    count: int = 0
    sample_payload: Any = None
    encoding: str = "text"

    @model_validator(mode="after")
    def _encode_payload(self) -> "ErrorShape":
        self.sample_payload, self.encoding = _as_text_payload(self.sample_payload, self.encoding)
        return self


class EvidenceStrength(Record):
    """How much the traces say about one tool."""
    call_count: int = 0
    error_count: int = 0
    trace_count: int = 0


class ToolSig(Record):
    """One customer tool as mined from the traces (D68, D70, D72)."""
    name: str
    description: Optional[str] = None
    args_schema: dict = Field(default_factory=dict)
    args_fields: list[FieldStat] = Field(default_factory=list)
    result_schema: list[FieldStat] = Field(default_factory=list)
    kind: ToolKind = "read"
    kind_confidence: Confidence = "low"
    kind_reason: Optional[str] = None
    unclassified: bool = True
    effects_observed: list[EffectObservation] = Field(default_factory=list)
    error_shapes: list[ErrorShape] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = Field(default_factory=EvidenceStrength)
    source: SigSource = "observed"
    classified_by: ClassifiedBy = "rule"


class Column(Record):
    """One column of the customer's world, with the class a Verdict compares it by (D73)."""
    table: str
    name: str
    class_: ColumnClass = Field(alias="class")
    class_rule: Optional[ColumnClass] = None
    class_confidence: Confidence = "low"
    class_reason: Optional[str] = None
    classified_by: ClassifiedBy = "rule"
    evidence: dict = Field(default_factory=dict)
    samples: list[Any] = Field(default_factory=list)


class EntitySchema(Record):
    """The tables, columns and id patterns mined from the whole corpus (D73, D74)."""
    tables: list[str] = Field(default_factory=list)
    columns: list[Column] = Field(default_factory=list)
    id_patterns: dict[str, str] = Field(default_factory=dict)
    synthetic_rows: list[str] = Field(default_factory=list)
    # table -> "parent.column": where the corpus stores this table's rows inside another table's
    # row, keyed by their own id (retail's items under products.variants). A table with a home is
    # still a table, because some rows are only ever shown on their own, but the home is where a
    # tool has to look first on the customer's real database.
    homes: dict[str, str] = Field(default_factory=dict)

# --- policy and the simulated user ---

class ConstraintTests(Record):
    """One positive and one negative case per compiled constraint."""
    pos: list[dict] = Field(default_factory=list)
    neg: list[dict] = Field(default_factory=list)


class Constraint(Record):
    """A policy or system-prompt sentence as a before-write predicate, or a residual (D76)."""
    id: str
    text: str
    span: Optional[RawPtr] = None
    span_text: Optional[str] = None
    predicate_src: Optional[str] = None
    tests: ConstraintTests = Field(default_factory=ConstraintTests)
    compiled: bool = False
    judge_atom: bool = False
    rewritten_text: Optional[str] = None
    residual_reason: Optional[str] = None


class UserFact(Record):
    """A fact the recorded user gave, exact (D44)."""
    field: str
    value: Any = None
    span: Optional[RawPtr] = None


class DisclosureRule(Record):
    """A fact the Simulated user gives only when asked."""
    field: str
    on_request: bool = True
    condition: Optional[str] = None


class UserRules(Record):
    """The Simulated user for one Run: facts exact, style representative (D44, D77)."""
    facts: list[UserFact] = Field(default_factory=list)
    disclosure: list[DisclosureRule] = Field(default_factory=list)
    refusals: list[str] = Field(default_factory=list)
    walk_away: list[str] = Field(default_factory=list)
    style_sample: list[str] = Field(default_factory=list)
    incomplete_reasons: list[str] = Field(default_factory=list)


class UserBehaviour(Record):
    """Stub: how the customer's real users behave, once we mine style across Runs (D44)."""
    behaviour_id: str
    style_notes: list[str] = Field(default_factory=list)
    patience: Optional[str] = None
    verbosity: Optional[str] = None
    trace_ids: list[str] = Field(default_factory=list)

# --- environment, categories, tasks ---

class Environment(Record):
    """One built replica of the customer's world, with the D97 sub-versions."""
    env_id: str
    schema_version: str = "0"
    tools_version: str = "0"
    policy_version: str = "0"
    version: int = 1
    parent_env_id: Optional[str] = None
    assisted_tools: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class OverlayRow(Record):
    """One row pinned in the version a Task's own Runs saw (D74)."""
    table: str
    id: str
    version_hash: str
    trace_id: Optional[str] = None
    after_write: bool = False  # the sighting came after a write in its own trace (D74 merge order)


class TaskOverlay(Record):
    """A Task's Starting state: rows read before the shared db.json (D74)."""
    task_id: str
    rows: list[OverlayRow] = Field(default_factory=list)


class Category(Record):
    """The Runs whose References write through the same tool set (D83)."""
    id: str
    name: Optional[str] = None
    write_tools: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)


class Task(Record):
    """A cluster of Runs sharing one Intent, inside a Category (D83); unguarded when too small to hold out (D81)."""
    id: str
    category_id: Optional[str] = None
    run_ids: list[str] = Field(default_factory=list)
    intent: Optional[str] = None
    unguarded: bool = False
    name: Optional[str] = None
    anchor_run_ids: list[str] = Field(default_factory=list)

# --- verifier ---

class Atom(Record):
    """One check in a Task's Verifier.

    `predicate_src` is the code the Runner evaluates; `target` is the same check as structured data
    (the tool, entity, field and value the Builder derived it from), so the Builder can read an atom
    back without parsing code and the Runner never has to parse JSON.
    """
    id: str
    kind: AtomKind
    description: Optional[str] = None
    provenance: Optional[ProvenanceClass] = None
    spans: list[RawPtr] = Field(default_factory=list)
    predicate_src: Optional[str] = None
    target: dict = Field(default_factory=dict)
    judge: bool = False


class Verifier(Record):
    """The End-state check for one Task, derived from confirmed References."""
    task_id: str
    atoms: list[Atom] = Field(default_factory=list)
    verifier_version: str = "0"
    seed_run_ids: list[str] = Field(default_factory=list)

# --- runs, events, verdicts, gates ---

class Usage(Record):
    """Tokens on one model call, for budget.py. Counts are never negative: a negative one
    would make call_cost negative and could lower the spend ceiling's total."""
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)


class Cost(Record):
    """What one model call or one Run cost."""
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    usd: float = Field(default=0.0, ge=0)
    wall_ms: float = Field(default=0.0, ge=0)
    # Where usd's price came from: "models.dev" or "table" (budget.PRICES), or None for an
    # unpriced model. Set by budget.record_call; nothing else writes it.
    price_source: Optional[str] = None


class Event(Record):
    """One line of a Run's JSONL."""
    idx: int
    ts: Optional[str] = None
    type: EventType
    payload: dict = Field(default_factory=dict)
    route: Optional[Route] = None
    cache_key: Optional[str] = None
    cost: Optional[Cost] = None
    assisted: bool = False


class Run(Record):
    """One re-executed episode: events in, End state and cost out.

    The Start and End state are not fields here: `loop.py` writes them into the stop event's
    payload, which is where `verdict.py` and `verifier.load_run` read them from, so the JSONL
    footer holds nothing that is not a Run field and unknown keys are refused like everywhere else.
    """
    run_id: str
    env_id: Optional[str] = None
    trace_id: Optional[str] = None
    task_id: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    user_rules: Optional[UserRules] = None
    events: list[Event] = Field(default_factory=list)
    end_state_hash: Optional[str] = None
    termination_reason: Optional[str] = None
    cost: Cost = Field(default_factory=Cost)
    route_counts: dict[str, int] = Field(default_factory=dict)
    assisted: bool = False
    parent_run_id: Optional[str] = None


_RUN_EVENT_TYPES = frozenset({"model_call", "tool_call", "tool_result", "user_turn", "error", "stop"})


def load_run_jsonl(path: Any) -> Run:
    """Read one Run from a JSONL file: header lines, event lines and a footer all work.

    An event-typed line is an event; anything else updates the header, and a footer's own bundled
    `events` list is spliced in. Header keys the Run model does not recognize become one final
    `stop` event's payload instead of being dropped, which is where loop.py's Start and End state
    land. verdict.py calls this directly; verifier.py keeps its own copy on the Builder side of the
    D89 boundary, since the Runner and the Builder may not import one another.
    """
    file = Path(path)
    header: dict = {}
    events: list[dict] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") in _RUN_EVENT_TYPES:
            events.append(obj)
        else:
            events.extend(obj.pop("events", None) or [])
            header.update(obj)
    extra = {key: value for key, value in header.items() if key not in Run.model_fields}
    if extra:
        events.append({"type": "stop", "payload": extra})
    header = {key: value for key, value in header.items() if key in Run.model_fields}
    header.setdefault("run_id", file.stem)
    header["events"] = [dict(event, idx=event.get("idx", pos)) for pos, event in enumerate(events)]
    return Run.model_validate(header)


def plain(value: Any) -> Any:
    """A pydantic value as plain JSON data, so a state hash and an End state stay comparable.

    route.py's StateView and Router use this for the world they read and write; compile_env.py
    keeps its own copy inline, because its sandboxed subprocess runs with no import path back into
    this package (`sys.executable -I` and `env={}`).
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def disagreement_stats(rows: Iterable[dict]) -> dict:
    """Judge disagreement and abstention over one list of judge_pairs rows (D92).

    judge.py's disagreement_rate() (live, during a build) and report.py's load() (after the fact)
    both count the same rows this way, so an abstain rate or an empty-pairs rate can never drift
    between what a judge sees and what a finished report shows.
    """
    rows = list(rows)
    pairs = len(rows)
    disagreements = sum(1 for row in rows if row.get("disagreement"))
    abstains = sum(1 for row in rows if row.get("abstain"))
    return {
        "pairs": pairs,
        "disagreements": disagreements,
        "rate": (disagreements / pairs) if pairs else 0.0,
        "abstains": abstains,
        "abstain_rate": (abstains / pairs) if pairs else 0.0,
    }


class RunnerVersion(Record):
    """Content hash of the Runner files and routing config, written by freeze-runner."""
    runner_version: str
    file_hashes: dict[str, str] = Field(default_factory=dict)
    routing_config_hash: Optional[str] = None
    created_at: Optional[str] = None
    confirmed_by: Optional[str] = None


class Verdict(Record):
    """The pass or fail of one Run, on End state only (D46, D88, D97).

    The versions have no placeholder default: a Verdict that never copied its Environment's
    and Runner's versions leaves them None, which is what lets regrade refuse it (D97).
    """
    run_id: str
    env_id: Optional[str] = None
    schema_version: Optional[str] = None
    tools_version: Optional[str] = None
    policy_version: Optional[str] = None
    verifier_version: Optional[str] = None
    verdict_version: Optional[str] = None
    runner_version: Optional[str] = None
    passed: bool = Field(alias="pass")
    failing_atom: Optional[str] = None
    same_path: Optional[bool] = None
    class_: VerdictClass = Field(alias="class")
    cause: Optional[Cause] = None
    judge_used: bool = False
    environment_suspected: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pass_matches_class(self) -> "Verdict":
        """The flag and the class are two names for one outcome; a Verdict where they
        disagree is a wrong Verdict, so it is refused rather than stored."""
        if self.passed != (self.class_ == "pass"):
            raise ValueError(
                f"pass={self.passed} disagrees with class={self.class_!r}: "
                "pass is true exactly when class is 'pass'"
            )
        return self


class SetAsideLesson(Record):
    """A cross-customer lesson the Builder judged irrelevant to this customer, listed in the report (D87)."""
    id: str = ""
    pattern: str = ""
    reason: str = ""


class GateResult(Record):
    """One gate from design section 6."""
    stage: str
    passed: bool = Field(alias="pass")
    metrics: dict = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


ALL_RECORDS: tuple[type[Record], ...] = tuple(
    o for o in list(globals().values()) if isinstance(o, type) and issubclass(o, Record) and o is not Record
)
