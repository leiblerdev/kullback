"""Builds the customer's world: one shared db.json by inverse replay, a per-Task overlay, the tau2 file
shape, and each tool body with its five gates and bounded repair.

The gates run each generated tool body in a subprocess sandbox before it is trusted; see
kullback.builder.sandbox for what that sandbox does and does not close off. Nothing here calls a
model on its own: write_tool_body takes a Model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.builder import lesson as lesson_mod
from kullback.builder import mine, synth
from kullback.builder.body_skill import BODY_SKILL
from kullback.builder.mine import is_assistant_call, is_scalar_result

# The failure grouping D181's rule 6 gives every other hint; repair.py imports nothing of this
# module, so the two sit in one layer with no cycle between them.
from kullback.builder.repair import SHAPES_SHOWN, failure_shapes
from kullback.builder.sandbox import (
    DB_CLASS,
    HELPERS,
    Sandbox,
    SandboxError,
    args_text,
    gate_parses,
    id_field,
    id_pattern_for,
    key_fields,
    key_separator,
    match_table,
    parse_result,
    partial_key,
    row_key,
    run_gates,
)

# Re-exported for tests and callers that reach the gates through this module rather than through
# kullback.builder.sandbox directly; compile_env.py's own code calls only run_gates and gate_parses.
from kullback.builder.sandbox import gate_deterministic as gate_deterministic
from kullback.builder.sandbox import gate_executes_on_s0 as gate_executes_on_s0
from kullback.builder.sandbox import gate_non_trivial as gate_non_trivial
from kullback.builder.sandbox import gate_replay_fidelity as gate_replay_fidelity
from kullback.gates.confinement import ALLOWED_IMPORTS, DENIED_BUILTINS, TOOLS_CLASS, source_confinement
from kullback.gates.tool_runs import body_replay_fidelity_gate
from kullback.runner.budget import (
    CHARS_PER_TOKEN,
    CONTEXT_CAP_FRACTION,
    context_cap_tokens,
)
from kullback.runner.budget import (
    DEFAULT_CONTEXT_WINDOW as CONTEXT_WINDOW,
)
from kullback.runner.canon import (
    canonicalize as canon,  # D39: one canonicalizer, no local variant
)
from kullback.runner.records import (
    Atom,
    Column,
    EntitySchema,
    Environment,
    GateResult,
    OverlayRow,
    OverlayStep,
    Task,
    TaskOverlay,
    ToolCall,
    ToolSig,
    Trace,
    Verifier,
    as_dict,
    content_hash,
)
from kullback.runner.route import call_fingerprint  # D197: one fingerprint, written and served alike

DB_FILE = "db.json"
OVERLAY_DIR = "overlays"
PINS_FILE = "overlay_pins.json"  # what the Starting-state pinner pinned per Task, and what it could not
NODE_DIR = "tool_nodes"
OUTCOME_DETAIL_CHARS = 300  # how much of one differing call's sentence a per-call row carries (D171)
MAX_REPAIR_ATTEMPTS = 3
# D202: how many starting values one column is tried in before the pinner gives up on inverting a
# write, and how many (column, value) pairs one recorded write is tried in altogether. A boolean or
# an enum has a domain of two to a few values and settles in the first batch; a wide table with a
# numeric column is what the two caps are for, so one write can never turn into a thousand runs.
PRE_WRITE_CANDIDATE_CAP = 16
PRE_WRITE_TRIES_CAP = 64
# One sandbox job carries a whole copy of the world per candidate, so how many candidates share a
# subprocess is set by how big that world is rather than by a fixed count: a small world settles a
# whole column in one run, a large one is split so no job file grows past this.
PRE_WRITE_JOB_BYTES = 4_000_000
# D117: at most this many model calls inside one attempt's tool-use loop, so a model that keeps
# reaching for lookup_rows or test_body instead of ever submitting a body cannot spend an attempt
# for free; the last reply's content is taken as the body once the rounds run out.
MAX_TOOL_ROUNDS = 6
# CONTEXT_WINDOW and CHARS_PER_TOKEN are budget.py's own (imported above): one context-sizing
# constant per Harness, not a second copy that could drift from what budget.py actually bills.
MAX_EVIDENCE_CHARS = context_cap_tokens(CONTEXT_WINDOW, CONTEXT_CAP_FRACTION) * CHARS_PER_TOKEN
EVIDENCE_LABELS = ("initial", "failing_call", "all_failing_calls", "full_call_table")
HELD_OUT_LABEL = "held_out_failed, shown calls only"  # no shown call failed, so none can be pointed at


class OverlayConflict(ValueError):
    """Two Tasks pin the same row in different versions: a Gate failure for the tau2 export (D74)."""


@dataclass
class StartingState:
    """The shared world plus one overlay per Task, as written under the workdir."""
    db: dict
    overlays: list[TaskOverlay] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    path: Optional[Path] = None
    synthetic_rows: list[str] = field(default_factory=list)


@dataclass
class EnvBundle:
    """Everything the tau2 export needs; the Environment record holds only identity and versions."""
    environment: Environment
    schema: EntitySchema
    tools: list[ToolSig] = field(default_factory=list)
    bodies: dict = field(default_factory=dict)
    db: dict = field(default_factory=dict)
    overlays: list[TaskOverlay] = field(default_factory=list)
    overlay_values: dict = field(default_factory=dict)
    policy_text: str = ""
    tasks: list[Task] = field(default_factory=list)
    verifiers: list[Verifier] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    domain: str = "domain"
    conflicts: list[str] = field(default_factory=list)  # overlay rows the tau2 export could not keep both of


@dataclass
class ToolBuild:
    """One compiled tool: the accepted body, every attempt as a node, and whether it went assisted."""
    name: str
    body: str
    nodes: list[dict] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    assisted: bool = False
    kept_attempt: Optional[int] = None  # which attempt's body this is, when no attempt passed
    # D171: one row per recorded call of this tool, so the miss can be attributed to the Task whose
    # Trace made the call. The corpus gate keeps one fidelity number; this keeps which call it was.
    call_outcomes: list[dict] = field(default_factory=list)
    # A kept assisted body that answers every recorded call the same way while the recordings differ:
    # not a body with a bug in it, a body that never read its arguments (`hardcoded_body`).
    hardcoded: bool = False
    # Only ever set by `grade_body`: a body carried over from an earlier run of the stage that
    # answered none of the recorded calls under the world as it stands now.
    could_not_run: bool = False
    # D211: what the code-only steps read off this body's own replay, as a `lesson.Diagnosis`. Set
    # by `grade_body` alone, since it is a reading of the body a tool already has and what it is
    # for is the ask the next writer is given. It is never serialized: the lesson it renders is.
    diagnosis: Any = None

# --- reading rows out of recorded tool results ---

ROW_WALK_DEPTH = 8  # how deep a result is walked for rows; generous, and a cycle never reaches it


def _scalars(value: dict) -> dict:
    """The names a dict states a plain value under, which are the key parts it can lend a child."""
    return {str(name): item for name, item in value.items()
            if isinstance(item, (str, int, float)) and not isinstance(item, bool)}


def argument_key_parts(args: Any) -> dict:
    """Every name the call's arguments state exactly one value for, at any depth of the arguments.

    A key part a row leaves out can be read off the call, because a tool can be told which row it is
    answering by the call and not repeat it in the row (`tool_runs.row_key`). That is only ever
    sound when the call names one candidate: a call whose own arguments carry two values under a
    name (a list of objects each with its own date) names none of its rows in particular, and
    filling the part from it would give several different rows one key. So a name the arguments
    state twice in two values is dropped here, and the sighting keeps a key it is missing a part of,
    which is what `partial_key` is for.

    The walk goes to any depth for the same reason `argument_ids` does: a call names what it acts on
    wherever its own shape puts it, and a part stated once inside a nested object is still stated
    once by the call.
    """
    seen: dict[str, list] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                    seen.setdefault(str(name), []).append(item)
                else:
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(args if isinstance(args, dict) else {})
    return {name: values[0] for name, values in seen.items()
            if len({canon(item) for item in values}) == 1}


def walk_result_rows(schema: EntitySchema, result: Any, args: Optional[dict] = None,
                     depth_cap: int = ROW_WALK_DEPTH,
                     stats: Optional[dict] = None) -> tuple[list[tuple[str, str, dict, int]], int]:
    """Every row a result states at any depth, with how deep it sat, and how often the cap was hit.

    A result is a tree, and the customer's tools put rows anywhere in it: a list of routes each
    holding the stops it is made of, a settings object holding a nested object, a wrapper keyed by a
    name. Reading only the top level pinned none of those, so a search over the parts of a route
    found nothing in the world and a nested setting kept the seed world's shape, not the Task's.

    Every dict is tested with `match_table` and every list is walked. A dict that matches a table is
    a row and is still walked, because a parent row that embeds child rows is a sighting of both.
    Depth counts the rows a row sits inside and not the lists: a list is how one result holds
    several of one thing, so the elements of a list the result answers are the result's own rows
    however many lists deep they sit, and a dict inside a row is one level down. That is what
    `_observations` reads to tell a row a result states from a mention of it inside another.

    The walk carries the containers on the current path so a structure that points back at itself
    ends, and stops at `depth_cap` levels, counting each stop so the count can be read back rather
    than the rows quietly going missing.

    A nested row is named by its own scope (D207). A key part the row itself leaves out is taken
    from the dicts it sits inside, nearest first, and only then from the call's arguments, and only
    where the arguments name one candidate for it (`argument_key_parts`). Handing the same flat
    arguments to a dict at every depth is what let a part meant for the result's top level name a
    row three levels down, and a part is never taken from a sibling at the same depth, because a
    sibling is another row and not this row's scope. Where each part came from is counted into
    `stats` as nested_key_sources, so a build can see the rule working rather than be told it does.

    Two dicts of one call that compose the same key and disagree on a column are a collision
    (nested_key_collision): a key that names two rows is a finding about the shape of the result,
    counted with the depth it happened at, not an earliest sighting silently winning.
    """
    rows: list[tuple[str, str, dict, int]] = []
    path: set[int] = set()
    capped = 0
    sources: dict[str, int] = {}
    keyless = 0
    from_args = argument_key_parts(args)

    def note(table: str, row: dict, inherited: dict) -> None:
        for name in key_fields(schema, table)[1:]:
            if row.get(name) is not None:
                where = "own"
            elif inherited.get(name) is not None:
                where = "parent"
            elif from_args.get(name) is not None:
                where = "args"
            else:
                where = "missing"
            sources[where] = sources.get(where, 0) + 1

    def walk(value: Any, depth: int, inherited: dict) -> None:
        nonlocal capped, keyless
        if not isinstance(value, (dict, list, tuple)):
            return
        if depth > depth_cap:
            capped += 1
            return
        marker = id(value)
        if marker in path:
            return
        path.add(marker)
        if isinstance(value, dict):
            match = match_table(schema, value, {**from_args, **inherited})
            if match:
                rows.append((match[0], match[1], value, depth))
                note(match[0], value, inherited)
            # What this dict states is the scope of everything inside it; the call's arguments stay
            # underneath, so the nearest enclosing dict wins and the call loses to both.
            scope = {**inherited, **_scalars(value)}
            for item in value.values():
                walk(item, depth + 1, scope)
        else:
            # A list is how a result holds several of one thing, not a level of nesting: the rows of
            # a list the result answers are the result's own rows, however many lists deep it sits.
            before = len(rows)
            for item in value:
                walk(item, depth, inherited)
            if len(rows) > before:
                # A list some of whose elements are rows: an element that matched nothing carries no
                # key of its own, and its position in the list is all there is to name it by.
                keyless += sum(1 for item in value if isinstance(item, dict)
                               and not any(item is row for _, _, row, _ in rows[before:]))
        path.discard(marker)

    walk(result, 0, {})
    if stats is not None:
        for where, count in sources.items():
            stats[f"nested_key_sources.{where}"] = stats.get(f"nested_key_sources.{where}", 0) + count
        # Every row here was named by a key it composed; nothing is homed by where it sat.
        stats["homed_by.key"] = stats.get("homed_by.key", 0) + len(rows)
        if keyless:
            stats["list_elements_carrying_no_key"] = stats.get("list_elements_carrying_no_key", 0) + keyless
        for table, _key, depth in _key_collisions(rows):
            stats["nested_key_collision"] = stats.get("nested_key_collision", 0) + 1
            found = stats.setdefault("nested_key_collisions", [])
            entry = {"table": table, "depth": depth,
                     "key_class": "composite" if len(key_fields(schema, table)) > 1 else "own"}
            if entry not in found:
                found.append(entry)
    return rows, capped


def _key_collisions(rows: list[tuple[str, str, dict, int]]) -> list[tuple[str, str, int]]:
    """(table, key, deepest sighting) for a key two dicts of one result compose and disagree under.

    Two sightings of one row that say the same thing are one row seen twice, which is ordinary. Two
    that differ under one key mean the key does not name a row, and a pinner that takes the earliest
    of them is choosing between two rows by call order. That is a finding, not a value.
    """
    seen: dict[tuple[str, str], list[tuple[dict, int]]] = {}
    for table, key, row, depth in rows:
        seen.setdefault((table, key), []).append((row, depth))
    return [(table, key, max(depth for _, depth in group))
            for (table, key), group in seen.items()
            if len({canon(row) for row, _ in group}) > 1]


def extract_rows(schema: EntitySchema, result: Any, args: Optional[dict] = None) -> list[tuple[str, str, dict]]:
    """The rows a result states, at any depth (`walk_result_rows`), without their depth."""
    rows, _ = walk_result_rows(schema, result, args)
    return [(table, row_id, row) for table, row_id, row, _ in rows]


@dataclass
class PartialHome:
    """A keyless result placed on the row the call named, and the columns anything could read of it.

    `row` empty means the row is known and its values are not: the result asserts its columns in a
    shape no reader of this build turns into values, so nothing is pinned and the sighting is
    counted rather than guessed at.
    """
    table: str
    row_id: str
    row: dict


def home_partial_result(schema: EntitySchema, tool_name: str, result: Any, args: Optional[dict] = None,
                        read_result: Optional[Callable[[str, Any], Any]] = None) -> Optional[PartialHome]:
    """A keyless result placed on the row the call's own arguments named, or None (D180 homing).

    A read can answer with the fields alone and leave which row they belong to entirely to the
    call: a status for the row the call named, a settings object for the line the call named.
    Such a result matches no table, so nothing about it was ever pinned and the seed world's one
    placeholder value answered every Task. The call named the row, so the row is knowable: the
    columns the result carries are laid on the row whose id column the call passed
    (`mine.asked_for_id`, which is `_home_of`'s own first rule, narrowed by `mine._address_of` so a
    call that lists a parent's children is not filed under the parent).

    Where the result is an object it states its own columns. Where it is a scalar, the columns are
    inside the text and only a reader of that tool can say what they are (D176): `read_result` is
    that reader, answering the column values a result asserts or None. A tool with no reader pins
    nothing, because reading a value out of a sentence in this module would be a second reader
    written by hand.

    Only the result's own columns and the parts of the key are kept; the call's other arguments are
    filters and a filter is not a fact about the row. A key the call does not complete homes
    nothing, because a partial key would name a row of its own (`partial_key`).
    """
    if isinstance(result, (list, tuple)) or not isinstance(args, dict) or not args:
        return None
    stated = result if isinstance(result, dict) else None
    id_names = sorted({name for table in schema.tables for name in key_fields(schema, table)})
    candidate = dict({name: value for name, value in args.items()
                      if isinstance(value, (str, int, float)) and not isinstance(value, bool)},
                     **(stated or {}))
    column = mine.asked_for_id(tool_name, candidate, id_names, args)
    table = mine.table_for_id(column) if column else None
    if table is None or table not in set(schema.tables):
        return None
    key_row = {name: args[name] for name in key_fields(schema, table) if args.get(name) is not None}
    key_row.update({name: value for name, value in (stated or {}).items() if value is not None})
    key = row_key(schema, table, key_row, args)
    if key is None or partial_key(schema, table, key):
        return None
    if stated is None:
        read = read_result(tool_name, result) if read_result is not None else None
        stated = dict(read) if isinstance(read, dict) and read else {}
    row = dict(stated)
    if row:
        for name in key_fields(schema, table):
            if row.get(name) is None and args.get(name) is not None:
                row[name] = args[name]
    return PartialHome(table=table, row_id=key, row=row)

# --- inverse replay over the whole corpus (D33, D74) ---

@dataclass
class _Obs:
    """One sighting of one row in one trace, and whether a write had already touched it."""
    table: str
    row_id: str
    row: dict
    trace_id: str
    order: tuple
    after_write: bool
    # How deep in the result the row sat: 0 for a row the result states at its top level, more for
    # one nested inside it. Kept so the pins report can say how many rows the deeper walk found.
    depth: int = 0
    # A keyless result homed onto the row the call named (`home_partial_result`), rather than a row
    # a result stated. Kept apart from `depth` so the pins report can count the two rules apart.
    homed: bool = False
    # Which recorded call saw the row (D197). `call` fingerprints the call the way route.py does, so
    # a Run making that call again can be told it is the same read; `call_id` names the recorded call
    # itself, which is what the sandbox scores a body on. A sighting no call made carries neither.
    # `from_write` says that call was itself a write (D202): `after_write` is true of the sighting a
    # write made and of every later sighting of the same row alike, so it cannot tell the two apart,
    # and a write's recorded result is the only evidence of what a column it touched held before it.
    call: str = ""
    call_id: str = ""
    from_write: bool = False
    # The tool that answered, by name. `call` fingerprints the name together with the arguments, so
    # two reads of one row through one tool with different filters carry different fingerprints;
    # `trace_worlds` needs the tool alone, because a list projection and a detail projection of one
    # row are two tools' statements and are never compared to each other. A sighting no call made
    # (a row another requestor's prose revealed) carries no tool.
    tool: str = ""
    # A nested sighting of a row some other result states on its own, kept out of the pinning
    # (`_observations`) and kept in the record. It is still a recorded result of the Run that made
    # it, which is what a Run's own version of a row is read from (D213), so dropping it entirely
    # hid a Run whose only mention of a row was inside the result of its own write.
    shadowed: bool = False

    @property
    def partial(self) -> bool:
        """Some of a row's columns rather than the row itself, so it never replaces a fuller sighting.

        A keyless result carries only the columns it asserts, and a row nested inside another is
        that row as the thing it sits in describes it, which is rarely all of it.
        """
        return self.homed or self.depth > 0


def _observations(traces: list[Trace], schema: EntitySchema, write_tools: set[str],
                  revealed_rows: Optional[dict] = None, stats: Optional[dict] = None,
                  read_result: Optional[Callable[[str, Any], Any]] = None) -> list[_Obs]:
    """Every row sighting in corpus order; a write marks the rows it returned or named in its args.

    `revealed_rows` are the rows another requestor's own prose results revealed (builder/readers.py),
    as (table, row id) to trace to row: one sighting per trace, already walked to the version that
    trace started in, so it sits before that trace's own calls and no write has touched it. R33 is
    unchanged for everything else: only the assistant's calls describe the customer's system, and a
    row that came from another requestor is marked as that requestor's on the schema.

    A row some result states on its own is described by those sightings alone: a dict nested inside
    another row that carries the same key is a mention of that row inside something else, and what
    it carries is about the thing it sits in. A part of a booking states the price that booking paid
    and the places that booking used, and taking it for the part's own row rewrote 44 rows one
    corpus had stated plainly. So a nested sighting of a row something states on its own is marked
    `shadowed` and pins nothing; only a row nothing states on its own is pinned from one, which is
    exactly the row the world was missing.

    Marked rather than dropped (D213). A shadowed sighting is still a recorded result of the Run
    that made it, and what one Run of a Task saw of a row is read from every result that Run
    recorded, read or write, at any depth. Dropping them left a Run whose only mention of a row sat
    inside its own write's result invisible to the comparison between the Runs of one Task, so the
    Task pinned another Run's version and every call of this one was scored against it. Every caller
    that pins skips `shadowed`; the comparison between Runs does not.
    """
    out: list[_Obs] = []
    for trace_index, trace in enumerate(traces):
        written: set[str] = set()
        for (table, row_id), by_trace in sorted((revealed_rows or {}).items()):
            row = by_trace.get(trace.trace_id)
            if row:
                out.append(_Obs(table, row_id, dict(row), trace.trace_id, (trace_index, -1), False))
        for call_index, call in enumerate(trace.tool_calls):
            # A call the simulated user made through its own tools (telecom's phone tools) is not a
            # sighting of the customer's system; only the assistant's calls describe it (R33).
            if call.error is not None or not is_assistant_call(call):
                continue
            is_write = call.name in write_tools
            result = parse_result(call.result)
            rows, capped = walk_result_rows(schema, result, call.args, stats=stats)
            if stats is not None and capped:
                stats["depth_capped"] = stats.get("depth_capped", 0) + capped
            # A result that states no row of its own is still about a row when the call named one.
            homed = None if rows else home_partial_result(schema, call.name, result, call.args, read_result)
            if homed and homed.row:
                rows = [(homed.table, homed.row_id, homed.row, 0)]
            elif homed is not None and stats is not None:
                stats["unread_partial_results"] = stats.get("unread_partial_results", 0) + 1
            fingerprint = call_fingerprint(call.name, call.args or {})
            for table, row_id, row, depth in rows:
                out.append(_Obs(table, row_id, row, trace.trace_id, (trace_index, call_index),
                                is_write or row_id in written, depth, bool(homed and homed.row),
                                fingerprint, str(call.id or ""), is_write, call.name))
            if is_write:
                written |= {row_id for _, row_id, _, _ in rows}
                # A write names the rows it changes wherever its own shape puts them, and a row it
                # named one level down is as touched as one it named at the top (D207). Reading only
                # the top level left a nested row's post-write sighting looking untouched, so the
                # inverse replay could keep it as the value the world started in. A composite key is
                # composed the way `named_rows` composes it, from beside the id and then from the
                # top level, because the parts alone name no row.
                written |= {value for _, value, _ in argument_ids(call.args)}
                written |= {row_id for _, row_id in named_rows(schema, call.args or {})}
    stated = {(obs.table, obs.row_id) for obs in out if not obs.depth}
    dropped = 0
    for obs in out:
        if obs.depth and (obs.table, obs.row_id) in stated:
            obs.shadowed = True
            dropped += 1
    if stats is not None and dropped:
        stats["nested_sightings_of_a_row_already_stated"] = (
            stats.get("nested_sightings_of_a_row_already_stated", 0) + dropped)
    return out


# A column name a recording cannot produce: the tag keeps it out of reach of any real key (D39's
# rule for STRING_TAG, borrowed here for the same reason).
RECORDED_TEXT = "\x00recorded_result"


def _recorded_text(_tool: str, result: Any) -> dict:
    """A keyless result as the recording's own words, with nothing read out of it (D216).

    A read that answers a sentence and leaves which row it is about to the call states the row's
    state inside that sentence, and only a reader can say which columns are in there. `trace_worlds`
    is not allowed a reader, but it does not need one: two recordings whose sentences differ before
    either wrote disagree about the row whatever the columns turn out to be, and two whose sentences
    match agree. So the version of such a sighting is the sentence, canonicalized.
    """
    return {RECORDED_TEXT: canon(result)}


def homing_hash(schema: EntitySchema) -> str:
    """The row identity mined off the recordings, as one hash: each table with its key fields.

    This is the second and last input `cluster.split_by_world` is allowed to depend on (D216), so a
    round whose grouping moved can say whether the homing moved with it. The column classes are
    deliberately absent: they are the harness's own proposals about the corpus and no longer reach
    the split, so a schema reclassified between two rounds leaves this hash where it was.
    """
    return content_hash({table: list(key_fields(schema, table)) for table in sorted(schema.tables)})


def trace_worlds(traces: Iterable[Trace], schema: EntitySchema, write_tools: set[str]) -> dict[str, dict]:
    """Per trace, the version of every row it saw before any write touched it: the world it started in.

    Two traces that saw one row in two such versions started in different worlds. cluster.py keeps
    them in different Tasks, because a Task's overlay can pin one version only (D74), and a trace
    replayed on the other version differs on every read of that row; one corpus's single customer
    seen across 456 recordings in as many states is where this was found.

    A version is a function of the recordings and of nothing else (D216). It is the canonical form
    (`runner/canon.canonicalize`, default rules) of the recorded result projected to that row, and
    the key it is filed under is the tool that answered together with the row's identity. Nothing
    the harness proposed about the corpus reaches it: not the column classes, not the readers, not
    a cache format. The split used to be taken over the `hard` columns of the mined schema, and a
    grouping that depends on the harness's own proposals is not reproducible, so every reader or
    schema improvement regrouped the corpus and stranded Tasks a frozen list still held.

    Row identity, which recorded values name a row, is the one mined input that stays. It is itself
    a function of the recordings: the key fields come off the shapes the recordings returned, so two
    rounds over one recording set home a result the same way, and without it there is no row to
    compare a sighting to at all.

    Keying by tool is what makes the comparison honest. A tool that lists rows answers a few columns
    of each and a tool that details one answers all of them, and the two disagree by shape on every
    row in the corpus; compared to each other they split every Task. Filed apart, a trace
    contradicts another only where the same tool showed the same row differently before either
    wrote, which is the disagreement D74 is about.

    A trace's version of a row through one tool is built column by column from that tool's own
    sightings, each column taking the value of the earliest sighting that carried it, so a trace
    that read part of a row and then the whole of it states one version and not two. A result the
    call homed onto a row without stating a column of it is its own recorded sentence
    (`_recorded_text`), which is what the recording says about the row when no reader is allowed.
    """
    traces = list(traces)
    seen: dict[str, dict[tuple[str, str, str], dict]] = {}
    for obs in _observations(traces, schema, write_tools, read_result=_recorded_text):
        if obs.after_write or obs.shadowed:
            continue
        version = seen.setdefault(obs.trace_id, {}).setdefault((obs.tool, obs.table, obs.row_id), {})
        for name, value in obs.row.items():
            version.setdefault(str(name), value)
    worlds = {trace_id: {key: content_hash(canon(version)) for key, version in rows.items()}
              for trace_id, rows in seen.items()}
    for trace_id, rows in _requestor_worlds(traces, write_tools).items():
        worlds.setdefault(trace_id, {}).update(rows)
    return worlds


def _requestor_worlds(traces: Iterable[Trace], write_tools: set[str]) -> dict[str, dict]:
    """Per trace, what another requestor's own tools said about it before that requestor wrote.

    R33 keeps a requestor's calls out of the customer's world: they describe the requestor's own
    device and not the customer's system. They are recordings all the same, and two Runs whose
    recordings show one requestor's device in two states before either wrote can no more share one
    overlay than two Runs that disagree about a customer's row (D164, D74).

    That split used to come from the readers stage, out of the columns it had proposed for the
    requestor's prose, so it moved whenever a reader was written, improved or dropped. Here the
    version is the recorded result itself and the row is the requestor, which the recording names on
    every call; no proposal of the harness's is anywhere in it (D216). The requestor's own writes
    close its world, because after one of those the device is in the state the Run put it in rather
    than the state it started in.
    """
    out: dict[str, dict] = {}
    for trace in traces:
        written: set[str] = set()
        for call in trace.tool_calls:
            requestor = str(getattr(call, "requestor", "") or "")
            if call.error is not None or is_assistant_call(call) or not requestor:
                continue
            if call.name in write_tools:
                written.add(requestor)
                continue
            if requestor in written:
                continue
            out.setdefault(trace.trace_id, {}).setdefault(
                (call.name, "", requestor), content_hash(canon(parse_result(call.result))))
    return out


def build_starting_state(
    traces: Iterable[Trace],
    schema: EntitySchema,
    workdir: Path | str,
    tasks: Optional[Iterable[Task]] = None,
    tool_sigs: Optional[Iterable[ToolSig]] = None,
    synthetic: bool = True,
    grow: Optional[dict[str, int]] = None,
    grow_seed: int = 0,
    revealed_rows: Optional[dict] = None,
    revealed_assumptions: Optional[Iterable[str]] = None,
    read_result: Optional[Callable[[str, Any], Any]] = None,
    bodies: Optional[dict] = None,
    rules: Any = None,
    readers: Any = None,
    guessed_columns: Optional[Iterable[tuple[str, str]]] = None,
) -> StartingState:
    """One shared db.json for the customer, plus one TaskOverlay per Task (D33, D74).

    Inverse replay: a row's shared value is the latest sighting that no write had touched yet, so an
    observed write is undone. Where a trace shows only the post-state, that state is kept and the
    assumption is recorded. Order is the order the traces are passed in, then call order; nothing is
    keyed by wall-clock time (design section 8). Ids the traces asked for but never showed are then
    filled with tagged synthetic rows (D40), unless `synthetic` is off. `grow` names a row count per
    table to reach with rows composed from the observed ones (D107, `synth.grow`); what was added,
    the rules it followed and the checks it passed are written to synthetic.json. `revealed_rows`
    are the rows another requestor's prose results revealed (`builder/readers.py`), which take the
    same inverse replay as any other row: one sighting per trace, marked on the schema by the
    requestor that revealed them and never part of the customer's own system.
    `revealed_assumptions` are that stage's own sentences, the columns it filled from the corpus
    because no recording read them before a write, recorded here with the state's own guesses.
    `read_result` is the readers' own function (D176), asked what a scalar result asserts when the
    call named a row and the result carries no key of its own (`home_partial_result`).
    `bodies` are the tool bodies this workdir already holds, with `rules` and `readers` for the
    compare: given them, a column a Task first touches with a write is pinned from what that write
    recorded rather than left at the shared value (D202, `PreWriteInverter`). Without them, which is
    every build before the first one that compiled a body, nothing is inverted.
    `guessed_columns` are the (table, column) pairs another stage filled from the corpus rather than
    read (`readers.filled_columns`): they arrive on a revealed row looking like any other sighting,
    and a sighting nobody read is exactly what the inversion is allowed to move.
    What the pinner did is written to `overlay_pins.json` beside the overlays.
    """
    traces, workdir = list(traces), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_tools = {s.name for s in (tool_sigs or []) if s.kind == "write"}
    stats: dict = {}
    sightings = _observations(traces, schema, write_tools, revealed_rows, stats, read_result)
    # What pins: every sighting but the nested one of a row something states on its own. `sightings`
    # keeps those too, because what one Run saw is read from every result that Run recorded (D213).
    observations = [obs for obs in sightings if not obs.shadowed]
    by_row: dict[tuple[str, str], list[_Obs]] = {}
    for obs in observations:
        by_row.setdefault((obs.table, obs.row_id), []).append(obs)

    db: dict[str, dict] = {table: {} for table in sorted(schema.tables)}
    assumptions: list[str] = [str(line) for line in (revealed_assumptions or [])]
    for (table, row_id), seen in sorted(by_row.items()):
        clean = [o for o in seen if not o.after_write]
        pool = clean or seen
        # A partial sighting states the columns it carries and nothing else, so it never wins the
        # row from a sighting of the whole of it; it states its own columns on top of that row
        # where it came later, which is the same inverse replay the whole row takes.
        whole = [o for o in pool if not o.partial]
        chosen = max(whole or pool, key=lambda o: o.order)
        row = dict(chosen.row)
        for later in sorted((o for o in pool if o.partial and o.order > chosen.order),
                            key=lambda o: o.order):
            row.update(later.row)
        if not clean:
            assumptions.append(f"{table} row {row_id} was only ever seen after a write; "
                               "its post-state is kept as the starting value")
        db.setdefault(table, {})[row_id] = row
    # A row every sighting of which was partial is a row the corpus mentioned and never stated.
    in_part = {key for key, seen in by_row.items() if all(o.partial for o in seen)}

    # The constants of the world (mine.world_constants): one row of values the corpus pinned, which
    # no sighting of a row can carry because the results they came from are not rows.
    constants_table = mine.constants_table_of(schema)
    if constants_table:
        db.setdefault(constants_table, {})[mine.CONSTANTS_ROW] = mine.constants_row(schema)

    assumptions += [f"{table} row {row_id} was seen without every part of its key; it was folded "
                    f"into the {count} rows whose known key columns match and is not a row of its own"
                    for table, row_id, count in fold_partial_rows(db, schema)]
    added = add_synthetic_rows(db, schema, traces) if synthetic else []
    assumptions += [f"{table_of} row {row_id} was never shown by a trace; it is a synthetic row "
                    "shaped from the observed rows and a Run that reads it is assisted"
                    for table_of, row_id in added]
    completed = complete_partial_rows(db, schema, in_part) if synthetic else []
    added += completed
    assumptions += [f"{table_of} row {row_id} was only ever mentioned in part; the columns no "
                    "sighting of it carried are shaped from the observed rows and a Run that reads "
                    "it is assisted"
                    for table_of, row_id in completed]
    # D202 runs after the shared world is settled and inside the overlay build, because it inverts a
    # write against the Task's own world: the shared rows with that Task's pins already on them.
    inverter = (PreWriteInverter(traces, observations, schema, db, tool_sigs or [], bodies, workdir,
                                 rules=rules, readers=readers, guessed=guessed_columns)
                if bodies else None)
    overlays = _build_overlays(observations, tasks or [], workdir, assumptions, stats, inverter,
                               schema=schema, sightings=sightings)
    assumptions += [f"{table_of} row {row_id} is stored under {home}; the standalone copy was folded "
                    "into it and a Task overlay that pins it re-adds the standalone copy"
                    for table_of, row_id, home in fold_into_homes(db, schema)]
    grown_ids: list[str] = []
    if grow:
        grown = synth.grow(db, schema, dict(grow), seed=grow_seed)
        grown_ids = grown.ids
        schema.synthetic_rows = sorted(set(schema.synthetic_rows) | set(grown_ids))
        assumptions += [f"{table_of} holds {len(ids)} synthetic rows composed from the observed ones "
                        "(D107); a Run that reads one is assisted"
                        for table_of, ids in sorted(grown.added.items())]
        if not grown.checks.get("ok", False):
            assumptions.append("the synthetic rows failed a check; see synthetic.json")
        (workdir / "synthetic.json").write_text(
            json.dumps(synth.report(grown), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    path = workdir / DB_FILE
    path.write_text(json.dumps(db, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (workdir / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n", encoding="utf-8")
    return StartingState(db=db, overlays=overlays, assumptions=assumptions, path=path,
                         synthetic_rows=[row_id for _, row_id in added] + grown_ids)


def fold_partial_rows(db: dict, schema: EntitySchema) -> list[tuple[str, str, int]]:
    """A row seen without every part of its key is not a new row and not a contradiction.

    The rule. Such a sighting is folded into every row whose known key columns match it, filling
    only the columns those rows were never shown holding, so it can add what the keyed sightings
    did not show and can never overwrite what they did. Where no keyed row matches, it is kept as
    the partial sighting it is, under its own partial key, because dropping it would lose an
    observation and inventing the missing part would be a guess (D41). Returns (table, key, rows it
    was folded into) per folded row.

    Nothing to do on a table keyed by one column, which is every table until the miner names a
    composite key: a key of one part is never partial.
    """
    folded: list[tuple[str, str, int]] = []
    separator = key_separator(schema)
    for table in sorted(db):
        fields = key_fields(schema, table)
        if len(fields) < 2:
            continue
        rows = db[table]
        for key in sorted(rows):
            if not partial_key(schema, table, key):
                continue
            known = key.split(separator)
            matches = [other for other in sorted(rows) if other != key
                       and _key_agrees(other.split(separator), known)]
            for other in matches:
                for name, value in rows[key].items():
                    rows[other].setdefault(name, value)
            if matches:
                del rows[key]
                folded.append((table, key, len(matches)))
    return folded


def _key_agrees(parts: list[str], known: list[str]) -> bool:
    """Whether a full key matches every part a partial key does know."""
    return len(parts) == len(known) and all(a == b for a, b in zip(parts, known, strict=False) if b)


def fold_into_homes(db: dict, schema: EntitySchema) -> list[tuple[str, str, str]]:
    """Move a row the corpus stores inside another row (schema.homes) out of its top-level table.

    One row, one place. A standalone sighting (get_item_details) and a nested one (the same item
    under products.variants) are the same row; keeping both would let a write land in one and a
    read come from the other. The nested copy wins the position, the standalone copy contributes
    any field the nested one lacks, and a row whose parent the traces never showed stays where it
    is, because there is nowhere to put it. Returns (table, id, home) per folded row.
    """
    folded: list[tuple[str, str, str]] = []
    for table, home in sorted((schema.homes or {}).items()):
        parent, column = home.split(".", 1)
        rows = db.get(table) or {}
        parents = [r for r in (db.get(parent) or {}).values() if isinstance(r, dict)]
        for row_id in sorted(rows):
            for parent_row in parents:
                nest = parent_row.get(column)
                if isinstance(nest, dict) and isinstance(nest.get(row_id), dict):
                    for name, value in rows[row_id].items():
                        nest[row_id].setdefault(name, value)
                    del rows[row_id]
                    folded.append((table, row_id, home))
                    break
    return folded


def argument_ids(args: Any) -> list[tuple[str, str, dict]]:
    """(the name it sat under, the value, the object it sat in) for every string in a call's arguments.

    Lists and dicts are walked to any depth, because a call names the rows it acts on wherever its
    own shape puts them: at the top level under the column's name, or inside a list of objects one
    or more levels down. The object a value sat in is carried back with it so the other parts of a
    composite key can be read from beside it, which is where a call that names several rows states
    each row's own date or shift; a string inside a list keeps the name of the key the list sat
    under and the object that key belongs to.
    """
    out: list[tuple[str, str, dict]] = []

    def walk(name: Optional[str], value: Any, scope: dict) -> None:
        if isinstance(value, str):
            if name is not None:
                out.append((name, value, scope))
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(str(key), item, value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(name, item, scope)

    walk(None, args if isinstance(args, dict) else {}, {})
    return out


def referenced_ids(traces: Iterable[Trace], schema: EntitySchema) -> list[tuple[str, str]]:
    """(table, id) pairs a call that succeeded named in its arguments, by the table's own id column.

    Public: build.py's build_environment gate wires these ids into validate.environment_gate, to
    check db.json actually holds every id a trace referenced.

    Arguments are walked to any depth (`argument_ids`), because reading the top level alone missed
    the shape a write takes when it acts on several rows at once: the rows are a list of objects
    under one argument, each object naming its row by the table's own key columns, so the call owed
    the world rows nothing counted and the body refused them as not found. The name of the key a
    value sits under is what names the table, at any depth; the pattern the miner recorded is the
    guard it always was.
    """
    out: set[tuple[str, str]] = set()
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None:  # an id the customer's tool refused is not a row we owe
                continue
            out |= named_rows(schema, call.args or {})
    return sorted(out)


def named_rows(schema: EntitySchema, args: dict) -> set[tuple[str, str]]:
    """(table, key) for every row a call's own arguments name, at any depth of the arguments.

    Each part of a composite key is read from beside the id first and from the top level second, so
    a list of rows that each carry their own date completes each row's own key, and a call that
    states one date for every row it names still completes them all. A composite key the call does
    not complete names no row: a partial id would be a row of its own, which is exactly what the
    composite key exists to prevent.
    """
    keys = {table: key_fields(schema, table) for table in schema.tables}
    out: set[tuple[str, str]] = set()
    for name, value, scope in argument_ids(args):
        for table, fields in keys.items():
            if not fields or name != fields[0]:
                continue
            pattern = id_pattern_for(schema, table, fields[0])
            if pattern and not re.match(pattern, value):
                continue
            parts = [scope.get(part, args.get(part)) for part in fields[1:]]
            if any(part is None for part in parts):
                continue
            out.add((table, key_separator(schema).join([value] + [str(p) for p in parts])))
    return out


def add_synthetic_rows(db: dict, schema: EntitySchema, traces: Iterable[Trace]) -> list[tuple[str, str]]:
    """Fill ids the traces referenced but never showed, shaped from the rows they did show (D40, D41).

    Shape and values are the observed rows' own: per column the value seen most often, with the id
    column set to the referenced id. Nothing is invented beyond that recombination, so a table with
    no observed row is left empty rather than made up. The ids are tagged in
    `EntitySchema.synthetic_rows`, which is what marks a Run that reads one as assisted (D49).
    """
    added: list[tuple[str, str]] = []
    for table, row_id in referenced_ids(traces, schema):
        rows = db.get(table) or {}
        if row_id in rows or not rows:
            continue
        fields = key_fields(schema, table)
        parts = row_id.split(key_separator(schema)) if len(fields) > 1 else [row_id]
        named = dict(zip(fields, parts, strict=False)) if fields else {}
        db.setdefault(table, {})[row_id] = dict(_modal_row(rows.values()), **named)
        added.append((table, row_id))
    schema.synthetic_rows = sorted(set(schema.synthetic_rows) | {row_id for _, row_id in added})
    return added


def complete_partial_rows(db: dict, schema: EntitySchema, in_part: Iterable[tuple[str, str]]
                          ) -> list[tuple[str, str]]:
    """Fill the columns of a row the corpus only ever mentioned, from the rows it did state (D40).

    A row seen only inside another row, or only through a keyless read homed onto it, carries the
    columns that sighting was about and no others, and a body that reads one of the rest finds
    nothing there. This is the same standing as a row the traces referenced and never showed, and it
    takes the same answer: the columns it lacks are the observed rows' commonest, what was actually
    seen stands over them untouched, and the id is tagged so a Run that reads the row is assisted
    (D49). A table whose every row was mentioned in part has nothing to shape a fill from and is
    left as it is, because recombining guesses would be inventing rather than recombining.
    """
    completed: list[tuple[str, str]] = []
    mentioned = set(in_part)
    for table, row_id in sorted(mentioned):
        rows = db.get(table) or {}
        row = rows.get(row_id)
        stated = [other for key, other in sorted(rows.items())
                  if (table, key) not in mentioned and isinstance(other, dict)]
        if not isinstance(row, dict) or not stated:
            continue
        filled = dict(_modal_row(stated), **row)
        if filled != row:
            rows[row_id] = filled
            completed.append((table, row_id))
    schema.synthetic_rows = sorted(set(schema.synthetic_rows) | {row_id for _, row_id in completed})
    return completed


def _modal_row(rows: Iterable[dict]) -> dict:
    """The value seen most often per column across the observed rows, ties broken canonically."""
    columns: dict[str, dict[str, int]] = {}
    firsts: dict[str, dict[str, Any]] = {}
    for row in rows:
        for name, value in row.items():
            key = canon(value)
            counts = columns.setdefault(name, {})
            counts[key] = counts.get(key, 0) + 1
            firsts.setdefault(name, {}).setdefault(key, value)
    return {name: firsts[name][sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]]
            for name, counts in columns.items()}


def reference_run_of(task: Task) -> str:
    """The Run a Task speaks for: its first Run the Builder may build from (D213).

    A Task's Verifier is derived from the Runs the anchor left the Builder, so the Run a generated
    Candidate is judged against is one of those and never a held-out one. Which of them is a
    tie-break and not a judgement, so it is the first in the Task's own frozen order (D200), and a
    Task whose every Run was held out falls back to the first of those rather than to nothing: an
    overlay has to name one Run or a Candidate would be served a mix of two.
    """
    held = set(task.anchor_run_ids or ())
    return next((run_id for run_id in task.run_ids if run_id not in held),
                task.run_ids[0] if task.run_ids else "")


def _build_overlays(observations: list[_Obs], tasks: Iterable[Task], workdir: Path,
                    assumptions: list[str], stats: Optional[dict] = None,
                    inverter: Optional[PreWriteInverter] = None,
                    schema: Optional[EntitySchema] = None,
                    sightings: Optional[list[_Obs]] = None) -> list[TaskOverlay]:
    """A Task's rows in the version its own Runs saw, column by column, and one layer per Run (D213).

    One rule for every column, whatever shape the recording stated it in. A Task's pinned row takes
    each column from the earliest of that Task's own sightings that carried it, so a row seen once
    whole and once in part is one pinned version and not two, a column a nested sighting is the only
    one to state is pinned like any other, and a column the corpus disagrees on carries the value
    this Task's own recording read first rather than the one value the shared world holds for every
    Task. Reading only the earliest sighting whole is why a nested row and a homed partial read were
    never pinned: they arrive as sightings of some of a row's columns, and a whole-row rule has
    nowhere to put them.

    One Task can hold Runs that recorded one column in two values, and pinning the earliest of them
    served every Run the version one of them saw: the call whose own recording answered with the
    other was scored against a world it never ran on, and a body that indexed the world by a single
    direct key was read as looking up the wrong row. So the earliest sighting no longer wins the
    Task. The Task overlay holds, per row and column, only what every Run of the Task that sighted
    the row agrees on; a column its Runs disagree on is left out of the Task overlay and carried
    instead by one layer per Run, holding that Run's own earliest sighting, laid over the Task's
    rows when that Run replays. A Run that never sighted a disagreed column, and a generated
    Candidate Run, which sighted nothing at all, are served the Reference Run's layer whole
    (`reference_run_of`), never a mix of two Runs' values.

    What one Run saw is read from every result that Run recorded, read or write, at any depth
    (`sightings`, which keeps the shadowed nested sightings `observations` drops). A Run whose only
    mention of a row sits inside its own write's recorded result is a Run that saw that row, and
    leaving it out is what let the disagreement pass uncounted.

    Two Runs that read different parts of one row do not disagree, which is why the comparison is
    per column and not over the whole row. Every disagreement is recorded as an assumption and as a
    `runs_disagree` row of `overlay_pins.json`, carrying the table, the key class, the column class
    and the Run ids but no value; a Task whose Runs disagree on a key column of a row it writes is
    flagged `split_candidate`, since one Task the mining grouped may be two.

    D197's sighting sequence is a within-Run statement and is now built per Run: the nth read of a
    row in one Run is served that Run's nth sighting, and the Task's own steps are the Reference
    Run's. Built across the Runs, the sequence handed a Run's first read of a row the value another
    Run's first read had seen.

    A column whose first touch in the Task is a write has no sighting to take, so `inverter` is
    given the Task's pinned rows and reads the write's own recorded result for what the column held
    before it (D202, `PreWriteInverter`); it changes the rows in place before the overlay is split,
    so the world the gates score bodies on (`call_starting_states`) is the inverted one.

    Per Task, what was pinned and by which of these rules is counted into `overlay_pins.json`.
    """
    sightings = observations if sightings is None else sightings
    corpus: dict[tuple[str, str], dict[str, set]] = {}
    for obs in observations:
        columns = corpus.setdefault((obs.table, obs.row_id), {})
        for name, value in obs.row.items():
            columns.setdefault(str(name), set()).add(canon(value))
    overlays, pins = [], {}
    by_table: dict[str, set] = {}  # D197: which columns of which table any Task saw move
    disagreements: list[dict] = []
    splits: list[str] = []
    for task in tasks:
        members = set(task.run_ids)
        seen: dict[tuple[str, str], list[_Obs]] = {}
        for obs in sorted((o for o in observations if o.trace_id in members), key=lambda o: o.order):
            seen.setdefault((obs.table, obs.row_id), []).append(obs)
        # Every result the Task's Runs recorded, the shadowed nested ones included: what a Run saw
        # is compared from these, and only what pins a row is taken from `seen`.
        recorded: dict[tuple[str, str], list[_Obs]] = {}
        for obs in sorted((o for o in sightings if o.trace_id in members), key=lambda o: o.order):
            recorded.setdefault((obs.table, obs.row_id), []).append(obs)
        rows: dict[tuple[str, str], dict] = {}
        sources: dict[tuple[str, str], dict[str, _Obs]] = {}
        for key, group in seen.items():
            row, source = {}, {}
            for obs in group:
                for name, value in obs.row.items():
                    if str(name) not in row:
                        row[str(name)], source[str(name)] = value, obs
            rows[key], sources[key] = row, source
        # Where a sighting of the row came from, kept apart from `seen` because the inversion can
        # add a row no sighting of this Task ever stated: a write named it and never answered it.
        origins = {key: (group[0].trace_id, group[0].after_write) for key, group in seen.items()}
        inverted = (inverter.invert(task, rows, origins, sources, assumptions)
                    if inverter is not None else {})
        reference = reference_run_of(task)
        scoped = _run_scoped_rows(task, rows, recorded, reference, schema)
        for key in sorted(scoped.disagreeing):
            table, row_id = key
            classes = sorted({_column_class(schema, table, name) for name in scoped.disagreeing[key]})
            assumptions.append(
                f"task {task.id} runs disagree on {table} row {row_id}: the overlay holds what they "
                f"agree on and each run replays against its own {', '.join(classes)} value")
            disagreements.append({
                "task_id": task.id, "table": table, "key_class": _key_class(schema, table),
                "column_classes": classes, "columns": len(scoped.disagreeing[key]),
                "run_ids": sorted({obs.trace_id for obs in recorded.get(key, ())} & members),
                "split_candidate": key in scoped.split_candidates,
            })
        if scoped.split_candidates:
            splits.append(task.id)
        steps = {run_id: [step for key in sorted(seen)
                          for step in _row_steps(key[0], key[1],
                                                 [o for o in seen[key] if o.trace_id == run_id])]
                 for run_id in sorted(members)}
        task_steps = steps.get(reference) or []
        overlay = TaskOverlay(task_id=task.id, rows=[
            OverlayRow(table=t, id=i, version_hash=content_hash(scoped.agreed[(t, i)]),
                       trace_id=origins[(t, i)][0], after_write=origins[(t, i)][1])
            for t, i in sorted(rows)
        ], steps=task_steps)
        assumptions += [f"task {task.id} pins {t} row {i} from a post-write sighting"
                        for (t, i), group in sorted(seen.items()) if group[0].after_write]
        values = {content_hash(scoped.agreed[key]): scoped.agreed[key] for key in rows}
        runs: dict[str, dict] = {}
        for run_id in sorted(members):
            changed = []
            for key in sorted(scoped.per_run.get(run_id) or {}):
                row = scoped.per_run[run_id][key]
                version = content_hash(row)
                values[version] = row
                changed.append(OverlayRow(table=key[0], id=key[1], version_hash=version,
                                          trace_id=origins[key][0], after_write=origins[key][1]))
            if changed or steps.get(run_id):
                runs[run_id] = {"rows": [as_dict(row) for row in changed],
                                "steps": [as_dict(step) for step in steps.get(run_id) or ()]}
        _write_overlay(workdir, overlay, values, runs=runs, reference_run_id=reference)
        pins[task.id] = {
            "rows": len(rows),
            "rows_nested": sum(1 for key in rows if any(o.depth for o in seen.get(key, ()))),
            "columns": sum(len(row) for row in rows.values()),
            "columns_homed": sum(1 for key, source in sources.items()
                                 for name in source if source[name].homed),
            "columns_the_corpus_disagrees_on": sum(
                1 for key, row in rows.items() for name in row
                if len(corpus.get(key, {}).get(name) or ()) > 1),
            # D213: the rows and columns this Task's own Runs disagree on, which the Task overlay
            # leaves unset and one layer per Run carries. Both are zero where the Runs agree.
            "rows_run_scoped": len(scoped.disagreeing),
            "columns_run_scoped": sum(len(names) for names in scoped.disagreeing.values()),
            "runs_with_a_layer": len(runs),
            "split_candidate": bool(scoped.split_candidates),
            # D197: the columns of this Task's own rows that moved between two of its reads, and the
            # sightings a Run can be served for them. Both are zero where nothing moved.
            "columns_time_varying": len({(s.table, s.id, name)
                                         for group in steps.values() for s in group for name in s.values}),
            "sequences_served": sum(len(group) for group in steps.values()),
            **{name: inverted.get(name, 0) for name in
               ("columns_inverted", "columns_no_inverse", "inversion_candidates_tried")},
            "inversions": list(inverted.get("inversions") or ()),
        }
        for step in (s for group in steps.values() for s in group):
            by_table.setdefault(step.table, set()).update((step.id, name) for name in step.values)
        overlays.append(_with_run_scope(overlay, runs.get(reference)))
    stats = dict(stats or {})
    stats["runs_disagree"] = disagreements
    stats["split_candidates"] = sorted(splits)
    if inverter is not None:
        stats["inversion_by_tool"] = inverter.by_tool
        stats["writes_without_a_body"] = inverter.writes_without_a_body
    _write_pins(workdir, pins, stats, {t: len(v) for t, v in sorted(by_table.items())})
    return overlays


@dataclass
class _RunScoped:
    """A Task's rows split into what its Runs agree on and what each Run saw for itself (D213)."""
    agreed: dict[tuple[str, str], dict]
    per_run: dict[str, dict[tuple[str, str], dict]]
    disagreeing: dict[tuple[str, str], list[str]]
    split_candidates: set[tuple[str, str]]


def _run_scoped_rows(task: Task, rows: dict, recorded: dict, reference: str,
                     schema: Optional[EntitySchema] = None) -> _RunScoped:
    """Split each pinned row into the columns the Task's Runs agree on and one layer per Run.

    A column two Runs recorded in different values is not a fact about the Task, so the Task overlay
    does not hold it. Each Run holds its own earliest value for it, and a Run that never sighted the
    column holds the Reference Run's, because a world with a column missing is not a world any
    recording ran on. Every Run of the Task gets a layer for such a row, so serving a Run its layer
    is always serving it a whole row.

    Only a column the Task pinned can be scoped: a column no sighting of the Task carried is a
    column the shared world answers, and the overlay has nothing to lay over it either way.
    """
    agreed = {key: dict(row) for key, row in rows.items()}
    per_run: dict[str, dict[tuple[str, str], dict]] = {}
    disagreeing: dict[tuple[str, str], list[str]] = {}
    splits: set[tuple[str, str]] = set()
    for key, row in rows.items():
        group = recorded.get(key) or ()
        versions = _run_versions(group)
        names = [name for name in sorted(row)
                 if len({canon(per[name]) for per in versions.values() if name in per}) > 1]
        if not names:
            continue
        disagreeing[key] = names
        for name in names:
            agreed[key].pop(name, None)
        fallback = {name: (versions.get(reference) or {}).get(name, row[name]) for name in names}
        for run_id in task.run_ids:
            own = versions.get(run_id) or {}
            per_run.setdefault(run_id, {})[key] = dict(
                agreed[key], **{name: own.get(name, fallback[name]) for name in names})
        # A Task whose Runs part on a key column of a row one of its writes touched may be two Tasks
        # the grouping ran together: the goal writes a row the Runs do not agree is the same row.
        if any(obs.after_write for obs in group) and set(names) & set(key_fields(schema, key[0])
                                                                     if schema is not None else ()):
            splits.add(key)
    return _RunScoped(agreed=agreed, per_run=per_run, disagreeing=disagreeing, split_candidates=splits)


def _with_run_scope(overlay: TaskOverlay, scope: Optional[dict]) -> TaskOverlay:
    """The Task overlay with one Run's own rows and its own step sequence laid over it (D213)."""
    if not scope:
        return overlay
    replaced = {(str(row["table"]), str(row["id"])): row for row in scope.get("rows") or ()}
    rows = [OverlayRow.model_validate(replaced[(row.table, row.id)])
            if (row.table, row.id) in replaced else row for row in overlay.rows]
    return overlay.model_copy(update={
        "rows": rows, "steps": [OverlayStep.model_validate(step) for step in scope.get("steps") or ()]})


def _key_class(schema: Optional[EntitySchema], table: str) -> str:
    """Whether the table's rows are named by one column or by several (D207's own two classes)."""
    return "composite" if schema is not None and len(key_fields(schema, table)) > 1 else "own"


def _column_class(schema: Optional[EntitySchema], table: str, name: str) -> str:
    """The class a Verdict compares this column by (D73), or unclassified where the schema has none."""
    for column in (getattr(schema, "columns", None) or ()):
        if column.table == table and column.name == name:
            return str(column.class_)
    return "unclassified"


def _row_steps(table: str, row_id: str, sightings: list[_Obs]) -> list[OverlayStep]:
    """One row's time-varying columns, per call of one Run's own recording, in call order (D197).

    One Run's, not the Task's (D213): a sequence built across two Runs served a Run's first read of
    a row the value another Run's first read had seen, which is a statement about the order the
    corpus happens to hold the Runs in and not about the customer's system. The Task's own steps are
    its Reference Run's, and every other Run is served its own.

    A column is time-varying for this Run when two of its reads of the row hold different values
    and no recorded write touched the row between them. That is a statement about the customer's
    system and not about the recording: a status read twice an hour apart, a reading taken again,
    a queue whose position has moved. Pinning one value made
    the first read match and every later one differ, which is what a whole-Task constant can say and
    no more.

    So each sighting the Task made becomes a step carrying only the columns that moved, keyed by the
    call's fingerprint and by how many calls of that fingerprint came before it. The Router counts
    the same way, so the nth read of the row is served the nth recorded value and the last value
    holds once the sequence is spent. A row read once, or read twice in one value, gets no step and
    keeps its single pin. Walking stops at the first sighting a write had already touched: from
    there the write's own value is what a later read sees, which is how it already worked.
    """
    usable: list[_Obs] = []
    for obs in sightings:
        if obs.after_write or not obs.call:
            break
        usable.append(obs)
    moved = {name for name in {n for obs in usable for n in obs.row}
             if len({canon(obs.row[name]) for obs in usable if name in obs.row}) > 1}
    if not moved:
        return []
    steps: list[OverlayStep] = []
    counted: dict[str, int] = {}
    for obs in usable:
        index = counted.get(obs.call, 0)
        counted[obs.call] = index + 1
        values = {str(name): obs.row[name] for name in sorted(moved) if name in obs.row}
        if values:
            steps.append(OverlayStep(table=table, id=row_id, call=obs.call, index=index,
                                     call_id=obs.call_id, values=values))
    return steps


def _run_versions(sightings: Iterable[_Obs]) -> dict[str, dict[str, Any]]:
    """Per Run, the version of one row that Run's own recordings state: each column's first sighting.

    Every recorded result of that Run counts, read or write, at any depth (D213): a Run whose only
    mention of a row is inside its own write's answer has still seen that row, and comparing only
    the reads left the Runs of one Task looking as if they agreed.

    The sightings are expected in the order the recordings made them, which is the order
    `_build_overlays` groups them in, so the first value a Run states for a column is the earliest.
    """
    by_run: dict[str, dict[str, Any]] = {}
    for obs in sightings:
        per = by_run.setdefault(obs.trace_id, {})
        for name, value in obs.row.items():
            per.setdefault(str(name), value)
    return by_run


def _columns_disagreeing(sightings: list[_Obs]) -> list[str]:
    """Columns two of a Task's Runs read differently, each Run taking its own earliest value."""
    by_run = _run_versions(sightings)
    names = sorted({name for per in by_run.values() for name in per})
    return [name for name in names
            if len({canon(per[name]) for per in by_run.values() if name in per}) > 1]


def column_domains(observations: Iterable[_Obs]) -> dict[tuple[str, str], list]:
    """Every starting value the corpus ever showed a column in, most frequent first (D202).

    A column's candidate starting values are the values it was ever seen holding. A boolean gives
    two, an enum a few, a numeric or text column gives whatever the corpus holds, which is why the
    caller caps the list rather than this. Values are counted after canon, so two spellings of one
    value are one candidate, and what is handed back is the value the corpus recorded.
    """
    counted: dict[tuple[str, str], dict] = {}
    for obs in observations:
        for name, value in obs.row.items():
            slot = counted.setdefault((obs.table, str(name)), {})
            entry = slot.setdefault(canon(value), [0, value])
            entry[0] += 1
    domains: dict[tuple[str, str], list] = {}
    for key, slot in counted.items():
        ordered = sorted(slot.items(), key=lambda item: (-item[1][0], str(item[0])))
        domains[key] = [value for _, (_count, value) in ordered]
    return domains


class PreWriteInverter:
    """Pins a column a Task first touches with a write from what that write recorded (D202).

    D188 pins each of a Task's columns from the earliest of that Task's own sightings that carried
    it. A column no Run read before the first write to it has no such sighting, so the overlay
    leaves it where the shared world put it, which is a valid starting value and rarely this Task's:
    a flag toggled, a setting checked and reset, a mode switched. The write's recorded result, the
    value it left or a sentence saying what it changed, is then the only evidence there is of what
    the column held before it, and this reads that evidence.

    The rule. For each write a Task recorded, the tool's kept body is run in the sandbox on the
    Task's world as it stands. Where the body already answers the way the recording did, nothing is
    inverted: the pre-state in front of it is consistent with the write. Where it does not, the body
    is run again on that world with one column changed to one candidate starting value, and the
    first candidate under which the body's answer matches the recording column by column (the D187
    compare) is pinned, with the reason `inverted_from_write`, the write's call id and the hash of
    the body the inversion trusted. Candidates are the column's observed domain across the corpus
    plus the value the shared world holds, capped per column and per write, columns tried narrowest
    domain first because a column the corpus shows in two values is what a toggle looks like. A
    column the Task read before any write touched its row is never inverted, because a read is
    better evidence than an inversion. A column no candidate settles keeps the shared value with the
    reason `no_inverse` and is counted, so the next reading sees which tools still part.

    A column is inverted for the first write that could have touched it and never again: the writes
    after it read the state the earlier ones left, which is the replay's business and not the
    pinner's. This is code only, no model calls. If the kept body for the tool is itself wrong the
    inversion inherits its error, which is why the body hash travels with the reason: a recompile
    that moves the body moves `bodies.json`, and the Starting state declares that file, so the
    inversion re-runs. A tool with no body yet, which is every tool on a first build, is skipped and
    counted.
    """

    def __init__(self, traces: Iterable[Trace], observations: Iterable[_Obs], schema: EntitySchema,
                 db: dict, tool_sigs: Iterable[ToolSig], bodies: dict, workdir: Path | str,
                 rules: Any = None, readers: Any = None, timeout: float = 30.0,
                 guessed: Optional[Iterable[tuple[str, str]]] = None):
        observations = list(observations)
        self.guessed = {(str(table), str(name)) for table, name in (guessed or ())}
        self.schema, self.db, self.workdir = schema, db, Path(workdir)
        self.rules, self.readers, self.timeout = rules, readers, timeout
        self.sigs = {sig.name: sig for sig in tool_sigs if sig.kind == "write"}
        self.bodies = {name: body for name, body in (bodies or {}).items()
                       if name in self.sigs and str(body or "").strip()}
        self.domains = column_domains(observations)
        self.calls = self._write_calls(traces)
        self.stated: dict[str, list[tuple[str, str]]] = {}
        for obs in observations:
            if obs.call_id and obs.from_write:
                seen = self.stated.setdefault(obs.call_id, [])
                if (obs.table, obs.row_id) not in seen:
                    seen.append((obs.table, obs.row_id))
        # Row id to the tables holding it, so the row a write named in its arguments is found once
        # over the world rather than once per recorded call.
        self.homes: dict[str, list[tuple[str, str]]] = {}
        for table in sorted(self.db):
            for row_id in sorted(self.db[table]):
                self.homes.setdefault(row_id, []).append((table, row_id))
        self.batch = max(1, min(PRE_WRITE_TRIES_CAP,
                                PRE_WRITE_JOB_BYTES // max(1, len(json.dumps(db, default=str)))))
        self.by_tool: dict[str, dict[str, int]] = {}
        self.writes_without_a_body = 0
        self._sources: dict[str, str] = {}

    def _write_calls(self, traces: Iterable[Trace]) -> dict[str, list[tuple[tuple, str, ToolCall]]]:
        """Per trace, every write call it recorded, whoever made it, in the order the corpus states them.

        R33 keeps another requestor's results from stating rows of the customer's system, and that
        rule stands: nothing here reads a result as a row. What a write is read for is the pre-state
        it implies, and the Router serves every requestor's call out of the same Task world (D164),
        so a write by a simulated user constrains that world exactly as the assistant's does. Reading
        only the assistant's inverted nothing at all on the corpus this was written for, where the
        writes that part are the user's own device tools.
        """
        out: dict[str, list[tuple[tuple, str, ToolCall]]] = {}
        for trace_index, trace in enumerate(traces):
            for call_index, call in enumerate(trace.tool_calls):
                if call.error is not None:
                    continue
                if call.name in self.sigs:
                    out.setdefault(trace.trace_id, []).append(
                        ((trace_index, call_index), trace.trace_id, call))
        return out

    def invert(self, task: Task, rows: dict, origins: dict, sources: dict,
               assumptions: list[str]) -> dict:
        """Invert this Task's pre-write columns in place; the counts of what it did and did not."""
        record = {"columns_inverted": 0, "columns_no_inverse": 0, "inversion_candidates_tried": 0,
                  "inversions": []}
        entries = sorted((entry for run_id in task.run_ids for entry in self.calls.get(run_id, [])),
                         key=lambda entry: entry[0])
        if not entries:
            return record
        # A column some Run read before any write had touched its row is pinned by that read, and a
        # read is better evidence of a starting value than an inversion can be. A column another
        # stage filled from the corpus is not one of those: it arrives on a revealed row looking
        # like a sighting, and nobody read it. Whole corpora sit behind that distinction, one of
        # them a device every Task's writes touch and no recording states.
        read_pinned = {(key, name) for key, source in sources.items()
                       for name, obs in source.items()
                       if not obs.after_write and (key[0], name) not in self.guessed}
        decided: set[tuple[tuple[str, str], str]] = set()
        for _order, trace_id, call in self._failing(entries, self._world(rows)):
            world = self._world(rows)
            for key in self._targets(call, rows):
                if self._invert_one(task, call, trace_id, key, world, rows, origins, decided,
                                    read_pinned, record, assumptions):
                    break  # the write is reproduced; the rows after this one are not its doing
        return record

    def _failing(self, entries, world) -> list:
        """The Task's writes the kept bodies do not already reproduce on the world in front of them.

        Every write of one tool is asked in one sandbox run, because the subprocess is the cost here
        and a Task can hold a dozen writes. The world is the Task's as the overlay states it before
        any inversion: a write whose pre-state an earlier inversion moved is asked again through the
        `decided` set, which is the same rule that inverts a column for its first write only.
        """
        out: list = []
        for tool, group in sorted(self._by_tool(entries).items()):
            body = self.bodies.get(tool)
            if not body:
                self.writes_without_a_body += len(group)
                continue
            sandbox = Sandbox(self._source(tool), world, self.workdir / "inversion" / tool,
                              timeout=self.timeout)
            try:
                results = sandbox.run([entry[2] for entry in group])
            except SandboxError:
                continue
            out += [entry for entry, result in zip(group, results, strict=False)
                    if not self._matches(entry[2], result)]
        return sorted(out, key=lambda entry: entry[0])

    @staticmethod
    def _by_tool(entries) -> dict:
        grouped: dict[str, list] = {}
        for entry in entries:
            grouped.setdefault(entry[2].name, []).append(entry)
        return grouped

    def _invert_one(self, task, call, trace_id, key, world, rows, origins, decided, read_pinned,
                    record, assumptions) -> bool:
        table, row_id = key
        tries, considered = self._tries(key, world, decided, read_pinned)
        for name in considered:
            decided.add((key, name))
        if not tries:
            return False
        hit = None
        for start in range(0, len(tries), self.batch):
            batch = tries[start:start + self.batch]
            record["inversion_candidates_tried"] += len(batch)
            self._count(call.name, "inversion_candidates_tried", len(batch))
            hit = self._first_match(call, key, world, batch)
            if hit is not None:
                break
        if hit is None:
            record["columns_no_inverse"] += len(considered)
            self._count(call.name, "columns_no_inverse", len(considered))
            return False
        name, value = hit
        row = dict(rows.get(key) or {})
        row[name] = value
        rows[key] = row
        world.setdefault(table, {})[row_id] = row
        if key not in origins:
            origins[key] = (trace_id, True)
        record["columns_inverted"] += 1
        self._count(call.name, "columns_inverted", 1)
        record["inversions"].append({"table": table, "row": row_id, "column": name,
                                     "reason": "inverted_from_write", "tool": call.name,
                                     "call_id": call.id or "",
                                     "body": content_hash(self.bodies[call.name])[:16]})
        assumptions.append(
            f"task {task.id} never read {table} row {row_id} column {name} before write call "
            f"{call.id} changed it; the starting value is inverted from what that write recorded")
        return True

    def _tries(self, key, world, decided, read_pinned) -> tuple[list[tuple[str, Any]], list[str]]:
        """The (column, candidate) pairs to run, narrowest domain first, and the columns they cover."""
        table, row_id = key
        keys_of = set(key_fields(self.schema, table))
        names = sorted({column.name for column in self.schema.columns if column.table == table})
        order = sorted(names, key=lambda name: (len(self.domains.get((table, name)) or ()), name))
        tries: list[tuple[str, Any]] = []
        considered: list[str] = []
        for name in order:
            if name in keys_of or (key, name) in read_pinned or (key, name) in decided:
                continue
            here = canon((world.get(table, {}).get(row_id) or {}).get(name))
            values = [value for value in self._candidates(table, row_id, name)
                      if canon(value) != here]
            if not values:
                continue
            considered.append(name)
            tries += [(name, value) for value in values][:PRE_WRITE_TRIES_CAP - len(tries)]
            if len(tries) >= PRE_WRITE_TRIES_CAP:
                break
        return tries, considered

    def _candidates(self, table: str, row_id: str, name: str) -> list:
        """The column's corpus domain, most frequent first, then the value the shared world holds."""
        values = list(self.domains.get((table, name)) or ())
        shared = (self.db.get(table, {}).get(row_id) or {}).get(name)
        if shared is not None and canon(shared) not in {canon(value) for value in values}:
            values.append(shared)
        return values[:PRE_WRITE_CANDIDATE_CAP]

    def _first_match(self, call, key, world, batch):
        """Run one batch of candidate worlds and hand back the first that reproduces the recording."""
        table, row_id = key
        probes, states = [], {}
        for index, (name, value) in enumerate(batch):
            variant = dict(world)
            variant[table] = dict(world.get(table) or {})
            row = dict(variant[table].get(row_id) or {})
            row[name] = value
            variant[table][row_id] = row
            probe = call.model_copy(update={"id": f"{call.id or call.name}#d202#{index}"})
            probes.append(probe)
            states[probe.id] = variant
        sandbox = Sandbox(self._source(call.name), world, self.workdir / "inversion" / call.name,
                          timeout=self.timeout, call_states=states)
        try:
            results = sandbox.run(probes)
        except SandboxError:
            return None
        for (name, value), result in zip(batch, results, strict=False):
            if self._matches(call, result):
                return name, value
        return None

    def _matches(self, call: ToolCall, result: Optional[dict]) -> bool:
        """The D187 compare: this answer says what the recording said, column by column."""
        if result is None:
            return False
        return bool(body_replay_fidelity_gate([call], [result], self.schema, label="pre_write_inverse",
                                              rules=self.rules, readers=self.readers).passed)

    def _targets(self, call: ToolCall, rows: dict) -> list[tuple[str, str]]:
        """The rows this write could have changed, most specific evidence first.

        A write that answers rows says which rows it touched. A write that answers a sentence says
        nothing, and then the id it was given is what `_observations` already marks a row written by.
        A write that answers a sentence and takes no arguments at all, which is what a device tool
        run on the one thing in front of its requestor looks like, says only which table it is about,
        through the reader that tool has (D176); its target is then this Task's own rows of that
        table, and with no reader either, this Task's rows. Nothing outside the Task's overlay is
        ever a target, so the search is bounded by what the Task itself saw.
        """
        out = list(self.stated.get(str(call.id or ""), ()))
        for value in call.args.values():
            if not isinstance(value, str):
                continue
            for key in self.homes.get(value, ()):
                if key not in out and (key in rows or key[0] in self.db):
                    out.append(key)
        if out:
            return out
        table_of = getattr(self.readers, "table_of", None)
        table = table_of(call.name) if callable(table_of) else None
        return [key for key in sorted(rows) if table is None or key[0] == table]

    def _source(self, tool: str) -> str:
        """The one-tool module the sandbox runs, built once per tool."""
        if tool not in self._sources:
            self._sources[tool] = module_source(self.schema, [self.sigs[tool]],
                                                {tool: self.bodies[tool]})
        return self._sources[tool]

    def _world(self, rows: dict) -> dict:
        """The Task's world as the overlay states it, which is what `merge_overlays` hands the gates."""
        world = copy.deepcopy(self.db)
        for (table, row_id), row in rows.items():
            world.setdefault(table, {})[row_id] = dict(row)
        return world

    def _count(self, tool: str, name: str, amount: int) -> None:
        self.by_tool.setdefault(tool, {})[name] = self.by_tool.setdefault(tool, {}).get(name, 0) + amount


def _write_pins(workdir: Path, pins: dict, stats: dict,
                time_varying_by_table: Optional[dict] = None) -> Path:
    """What the pinner did, per Task and in total: the ruling a person reads after a build."""
    totals = {name: sum(int(row.get(name) or 0) for row in pins.values())
              for name in ("rows", "rows_nested", "columns", "columns_homed",
                           "columns_the_corpus_disagrees_on", "columns_time_varying",
                           "sequences_served", "columns_inverted", "columns_no_inverse",
                           "inversion_candidates_tried", "rows_run_scoped", "columns_run_scoped",
                           "runs_with_a_layer")}
    disagreements = list(stats.get("runs_disagree") or ())
    splits = sorted(stats.get("split_candidates") or ())
    payload = {"columns_time_varying_by_table": dict(time_varying_by_table or {}),
               # D213: every row a Task's own Runs recorded in two versions, with the classes the
               # harness already mines and the Run ids, and nothing copied out of a record.
               "runs_disagree": disagreements,
               "rows_run_scoped": totals["rows_run_scoped"],
               "columns_run_scoped": totals["columns_run_scoped"],
               "tasks_with_run_overlays": sum(1 for row in pins.values() if row.get("rows_run_scoped")),
               "tasks_split_candidate": splits,
               "depth_cap": ROW_WALK_DEPTH,
               "results_deeper_than_the_cap": int(stats.get("depth_capped") or 0),
               "unread_partial_results": int(stats.get("unread_partial_results") or 0),
               "nested_sightings_of_a_row_already_stated":
                   int(stats.get("nested_sightings_of_a_row_already_stated") or 0),
               "inversion_by_tool": dict(sorted((stats.get("inversion_by_tool") or {}).items())),
               "writes_without_a_body": int(stats.get("writes_without_a_body") or 0),
               # D207: where each part of a composed key came from, whether anything was named by
               # its position rather than by a key, and the keys that named two rows at once.
               "nested_key_sources": {where: int(stats.get(f"nested_key_sources.{where}") or 0)
                                      for where in ("own", "parent", "args", "missing")},
               "homed_by": {"key": int(stats.get("homed_by.key") or 0),
                            "position": int(stats.get("homed_by.position") or 0)},
               "list_elements_carrying_no_key": int(stats.get("list_elements_carrying_no_key") or 0),
               "nested_key_collision": int(stats.get("nested_key_collision") or 0),
               "nested_key_collisions": list(stats.get("nested_key_collisions") or ()),
               "totals": totals, "tasks": dict(sorted(pins.items()))}
    path = Path(workdir) / PINS_FILE
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_overlay(workdir: Path, overlay: TaskOverlay, values: dict,
                   runs: Optional[dict] = None, reference_run_id: str = "") -> Path:
    """One file per Task: the overlay record, one layer per Run, and the values behind both (D213).

    `overlay` holds what the Task's Runs agree on; `runs` maps a Run id to the rows and steps that
    Run replays against where they part from it, and `reference_run_id` names the layer anything
    with no Run of its own is served.
    """
    directory = Path(workdir) / OVERLAY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{overlay.task_id}.json"
    payload = {"overlay": as_dict(overlay), "values": values,
               "runs": dict(runs or {}), "reference_run_id": reference_run_id}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def load_overlay(workdir: Path | str, task_id: str,
                 run_id: Optional[str] = None) -> tuple[TaskOverlay, dict]:
    """The overlay one Run of the Task replays against, and its row values (D74, D213).

    Without a Run, or with one this Task has no layer for, which is every generated Candidate Run,
    the Reference Run's layer is served: one Run's world whole rather than a mix of two. route.py
    reads this before the shared db.json.
    """
    payload = json.loads((Path(workdir) / OVERLAY_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
    overlay = TaskOverlay.model_validate(payload["overlay"])
    runs = payload.get("runs") or {}
    scope = runs.get(run_id) if run_id else None
    if scope is None:
        scope = runs.get(str(payload.get("reference_run_id") or ""))
    return _with_run_scope(overlay, scope), payload["values"]


def load_run_overlays(workdir: Path | str) -> dict[str, dict[str, TaskOverlay]]:
    """Per Task, the overlay each of its Runs replays against (D213), for the gates' per-call worlds."""
    directory = Path(workdir) / OVERLAY_DIR
    out: dict[str, dict[str, TaskOverlay]] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        overlay = TaskOverlay.model_validate(payload["overlay"])
        out[overlay.task_id] = {run_id: _with_run_scope(overlay, scope)
                                for run_id, scope in sorted((payload.get("runs") or {}).items())}
    return out


def overlay_values(workdir: Path | str) -> dict:
    """Every overlay row value written under the workdir, keyed by version hash."""
    directory = Path(workdir) / OVERLAY_DIR
    values: dict = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        values.update(json.loads(path.read_text(encoding="utf-8"))["values"])
    return values


def merge_overlays(db: dict, overlays: Iterable[TaskOverlay], values: dict,
                   conflicts: Optional[list[str]] = None) -> dict:
    """Merge every Task overlay into the one db tau2's harness loads (D74).

    Two Tasks pinning one row in different versions is a disagreement about the world, which the
    per-Task overlays in the Runner never have to settle: each Task reads its own. The single db of
    the tau2 export has to pick one, so with `conflicts` given it keeps the version a Task saw before
    any write (over one seen after a write) and, between two of the same standing, the first Task's,
    and appends one line per conflict for the export gate. Without `conflicts` a disagreement raises,
    which is the contract the single-overlay callers rely on.
    """
    merged = copy.deepcopy(db)
    pinned: dict[tuple[str, str], tuple[str, str, bool]] = {}
    for overlay in overlays:
        for row in overlay.rows:
            seen = pinned.get((row.table, row.id))
            if seen and seen[0] != row.version_hash:
                if conflicts is None:
                    raise OverlayConflict(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row "
                                          f"{row.id} in different versions")
                conflicts.append(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row {row.id} in "
                                 f"different versions; the tau2 export keeps "
                                 f"{overlay.task_id if seen[2] and not row.after_write else seen[1]}'s")
                if not (seen[2] and not row.after_write):
                    continue  # the pinned version stands: it was seen before a write, or both were
            pinned[(row.table, row.id)] = (row.version_hash, overlay.task_id, row.after_write)
            if row.version_hash in values:
                merged.setdefault(row.table, {})[row.id] = values[row.version_hash]
    return merged

# --- rendering the tau2 file shape: code writes the signature, docstring and schema (D56) ---

_TYPES = {"str": "str", "int": "int", "float": "float", "bool": "bool", "list": "list", "dict": "dict"}

_DATA_MODEL_HEAD = '''"""Generated by the Harness from the customer's traces. Do not edit by hand."""
# Annotations stay eager: the gates exec this module without registering it in sys.modules, so a
# postponed annotation would have no module globals to resolve against.
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:  # tau2's own DB base when the export is loaded inside tau2-bench
    from tau2.environment.db import DB as _DBBase
except ImportError:
    _DBBase = BaseModel
'''

_DB_LOAD = '''
    @classmethod
    def load(cls, path):
        """Load the emitted db.json."""
        import json

        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))
'''

_TOOLKIT_SHIM = '''try:  # tau2's own toolkit base when the export is loaded inside tau2-bench
    from tau2.environment.toolkit import ToolKitBase as _ToolKitBase, ToolType, is_tool
except ImportError:
    class _ToolKitBase:
        def __init__(self, db):
            self.db = db

    class ToolType:
        READ = "read"
        WRITE = "write"
        GENERIC = "generic"

    def is_tool(tool_type):
        def decorate(func):
            func.__tool_type__ = tool_type
            return func

        return decorate
'''


def _class_name(table: str) -> str:
    """orders -> Order, payment_methods -> PaymentMethod."""
    singular = table[:-1] if table.endswith("s") and not table.endswith("ss") else table
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", singular) if p) or "Row"


def _annotation(types: Iterable[str]) -> str:
    """One observed type gives that type; a union of several gives Any (D72)."""
    kinds = {_TYPES.get(t, "Any") for t in types if t != "NoneType"}
    return kinds.pop() if len(kinds) == 1 else "Any"


def render_data_model(schema: EntitySchema) -> str:
    """One class per table plus the DB class, every column Optional[Any] so no real row is rejected.

    A mined column carries a handful of display samples, not the customer's type (D72): samples that
    are all int say nothing about the row holding 10.5, and samples that are all str say nothing
    about the row holding 94016. A narrow annotation there makes pydantic reject a real row, the
    module never loads, and every body fails gate 2 for a reason no body can fix. So the emitted
    model is as wide as the union it stands for, and the column classes in `EntitySchema` (D73), not
    the annotation, are what a Verdict compares by.
    """
    parts = [_DATA_MODEL_HEAD]
    for table in sorted(schema.tables):
        fields = []
        for name in sorted({c.name for c in schema.columns if c.table == table}):
            fields.append(f"    {name}: Optional[Any] = Field(default=None)")
        body = "\n".join(fields) or "    pass"
        parts.append(f'\n\nclass {_class_name(table)}(BaseModel):\n    """One row of {table}."""\n{body}\n')
    tables = "\n".join(f"    {t}: Dict[str, {_class_name(t)}] = Field(default_factory=dict)"
                       for t in sorted(schema.tables)) or "    pass"
    parts.append(f'\n\nclass {DB_CLASS}(_DBBase):\n    """The customer\'s world as the traces show '
                 f'it."""\n{tables}\n{_DB_LOAD}')
    return "".join(parts)


def _signature(sig: ToolSig) -> str:
    """The call signature, required arguments first, from the mined args_fields."""
    required = [f"{f.name}: {_annotation(f.types)}" for f in sig.args_fields if not f.optional]
    optional = [f"{f.name}: Optional[{_annotation(f.types)}] = None" for f in sig.args_fields if f.optional]
    return f"    def {sig.name}({', '.join(['self'] + required + optional)}) -> Any:"


def _in_docstring(text: str) -> str:
    """The customer's own words, safe to paste inside a triple-quoted string.

    A description holding a triple quote or a trailing backslash would otherwise end the docstring
    early and break gate 1 on every attempt, for a reason no model could fix (design section 7: code
    owns the docstring, so a code-owned break is code's to prevent).
    """
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _docstring(sig: ToolSig) -> str:
    """The customer's description, the argument list and the observed result fields."""
    lines = ['        """' + _in_docstring(
        sig.description or f"The customer's {sig.name} tool, as the traces show it.")]
    if sig.args_fields:
        lines += ["", "        Args:"] + [f"            {f.name}: {_annotation(f.types)}" for f in sig.args_fields]
    if is_scalar_result(sig):
        lines += ["", "        Returns:",
                  "            a bare " + _annotation(sig.result_schema[0].types) + ", not wrapped in an object"]
    elif sig.result_schema:
        lines += ["", "        Returns:", "            " + ", ".join(f.name for f in sig.result_schema)]
    return "\n".join(lines + ['        """'])


def render_tools(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict,
                 class_name: str = TOOLS_CLASS, with_imports: bool = True) -> str:
    """The toolkit class: code owns everything but the body, which the model wrote."""
    head = ""
    if with_imports:
        names = ", ".join([DB_CLASS] + [_class_name(t) for t in sorted(schema.tables)])
        head = ('"""Generated by the Harness from the customer\'s traces. Do not edit by hand."""\n'
                f"from typing import Any, Optional\n\nfrom data_model import {names}\n\n")
    parts = [head, _TOOLKIT_SHIM,
             f'\n\nclass {class_name}(_ToolKitBase):\n    """Every tool mined from the customer\'s '
             f'traces."""\n\n    def __init__(self, db) -> None:\n        super().__init__(db)\n'
             "        self.db = db\n"]
    for sig in sigs:
        body = textwrap.dedent(bodies.get(sig.name, "raise NotImplementedError")).strip("\n") or "pass"
        parts.append(f"\n    @is_tool(ToolType.{sig.kind.upper()})\n{_signature(sig)}\n{_docstring(sig)}\n"
                     + textwrap.indent(body, "        ").rstrip() + "\n")
    return "".join(parts)


def module_source(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict) -> str:
    """One self-contained module, data model plus toolkit, for the gates to execute."""
    return render_data_model(schema) + "\n\n" + render_tools(schema, sigs, bodies, with_imports=False)


def load_toolkit(source: str, db: dict, class_name: str = TOOLS_CLASS, db_class: str = DB_CLASS,
                 overlay: Optional[TaskOverlay] = None, overlay_values: Optional[dict] = None):
    """The generated module loaded in this process, with the Task's Starting state inside it.

    Given a Task's overlay, the world handed to the toolkit is the shared db with that overlay
    merged, so the March Task's code-routed call sees the March row and not the June one (D74). A
    generated body reads `self.db` and cannot do the overlay lookup itself, so a toolkit built on the
    shared db alone leaves the overlay dead for every code route; `route.py`'s StateView stays the
    lookup for the recording and stand-in routes.

    dont_inherit keeps a caller's `from __future__ import annotations` out of the generated module:
    a postponed annotation has no module globals for pydantic to resolve against. This is the loader
    the Runner's router is given; the gates use the subprocess Sandbox instead.

    `evaluate_arithmetic` is put in the namespace, not imported by the body: `ast`, `eval`, `exec`
    and `compile` stay refused, and a recorded tool that evaluates an expression string is served by
    the one code-owned evaluator instead of a parser the model writes again on every build (see
    runner/arith.py). `PROVIDED_HELPERS` is what the confinement gate reads it under, so the two
    cannot drift: a name bound here that the gate does not know is a body refused for a NameError it
    would never have raised.
    """
    if overlay is not None:
        db = merge_overlays(db, [overlay], overlay_values or {})
    refused = source_confinement(source, class_name)
    if refused:
        raise SandboxError("the generated module is not confined and would run in this process: "
                           + "; ".join(refused))
    namespace: dict = {"__name__": "generated_tools", **HELPERS}
    exec(compile(source, "<generated>", "exec", dont_inherit=True), namespace)  # noqa: S102
    return namespace[class_name](namespace[db_class].model_validate(db))

# --- the model writes the body (D56, D75) ---

# tau2 wraps every raised exception as f"Error: {e}" before it reaches the agent
# (vendor/tau2-bench, Environment.get_response), and the first live build copied that wrapper into
# seven bodies, faithfully, because nothing said it was not part of the message. The wrapper is
# the transport talking, not the customer's tool, but which wrapper a customer's transport adds
# is not ours to know in advance (D51): it is read off the corpus. A prefix that every recorded
# error shares, ending at a ": " boundary and leaving a message behind on every one of them, is
# transport; the copy shown to the model has it peeled off. `ToolCallError.payload` keeps the
# wrapper (D67 keeps the payload verbatim, with `raw_ptr` back to D66's untouched byte).


def shared_error_prefix(calls: Iterable[ToolCall]) -> str:
    """The prefix every string error payload in `calls` shares, cut at the last ": " they all have.

    Two payloads at least, or one message alone would be its own prefix. The remainder has to be
    non-empty on every payload, so a corpus whose errors are all one identical message yields the
    part before its last ": " only if that leaves something after it.
    """
    payloads = [call.error.payload for call in calls
                if call.error is not None and isinstance(call.error.payload, str)]
    if len(payloads) < 2:
        return ""
    common = payloads[0]
    for payload in payloads[1:]:
        limit = min(len(common), len(payload))
        cut = next((i for i in range(limit) if common[i] != payload[i]), limit)
        common = common[:cut]
    boundary = common.rfind(": ")
    while boundary >= 0:
        prefix = common[:boundary + 2]
        if all(len(payload) > len(prefix) for payload in payloads):
            return prefix
        boundary = common.rfind(": ", 0, boundary)
    return ""


def _display_error_payload(payload: Any, prefix: str = "") -> Any:
    """The error payload as the model should read it, with the corpus's shared prefix peeled off.

    Only a leading, exact `prefix` is peeled, and only from a str payload; a JSON payload (D67's
    `code` class) and a payload that never carried the prefix pass through untouched.
    """
    if prefix and isinstance(payload, str) and payload.startswith(prefix) and len(payload) > len(prefix):
        return payload[len(prefix):]
    return payload


_SYSTEM = ("You write the body of one Python method of a tool class rebuilt from a customer's traces. "
           "Return only the body: no signature, no fences, no explanation. The body may read and write "
           "self.db, a pydantic model with one dict per table. Each dict's values are pydantic model rows, "
           "not plain dicts: read or write a row's field by attribute, as in order.status or "
           "order.status = \"cancelled\", never with .get(...) or any other dict method. That holds for the "
           "row and stops there: what a column of a row holds is not a model. A column whose value is a "
           "list or a dict is a plain Python list or dict, so index it, read its parts by key, as in "
           "row.entries[0][\"amount\"], and write them by plain assignment, as in "
           "row.entries[0][\"amount\"] = 0; row.entries[0].amount raises AttributeError. Where a column "
           "holds one, the tables below say so and show the form that reads it. The body reads the "
           "world's tables for every id and every value it works with: an id or a value copied out "
           "of a recorded call and written into the body as a literal is refused by the "
           "memorised_values gate, however well it replays the calls you were shown. Raise ValueError "
           "with the customer's own message where the traces show an error. Where every recorded error "
           "in this corpus begins with the same transport prefix, it is shown with that prefix removed, "
           "so write the message exactly as shown and do not put a prefix of your own in front of it.")


def _example_block(calls: Iterable[ToolCall], error_prefix: Optional[str] = None) -> str:
    """The recorded calls as the model sees them: arguments, then result or error class, in full.

    Nothing is cut here. D75's third attempt is the full call table, and a node that says
    `evidence_calls: 30` has to mean the model saw thirty complete rows; the only limit is the D65
    cap, which refuses the call rather than shortening it. `error_prefix` is the corpus-wide shared
    error prefix (`shared_error_prefix`); given None it is read off these calls alone.
    """
    calls = list(calls)
    prefix = shared_error_prefix(calls) if error_prefix is None else error_prefix
    lines = []
    for call in calls:
        outcome = (f"error {call.error.class_}: {_display_error_payload(call.error.payload, prefix)!r}"
                   if call.error is not None
                   else "result " + json.dumps(parse_result(call.result), default=str))
        lines.append(f"- args {json.dumps(call.args, sort_keys=True, default=str)} -> {outcome}")
    return "\n".join(lines)


def _sample_value(column: Column) -> Any:
    """One mined sample of a column as a Python value, or None when there is none to read.

    Mining stores a nested sample as JSON text and cuts it at a length (`mine._short`), so a long
    one does not parse; a sample that does not parse is no sample, and the caller falls back to the
    form that names no field.
    """
    sample = column.samples[0] if column.samples else None
    if isinstance(sample, (list, dict)):
        return sample
    if isinstance(sample, str):
        try:
            return json.loads(sample)
        except ValueError:
            return None
    return None


def _column_access(table: str, column: Column) -> str:
    """How a column that holds a list or a dict is read, in one line, off its own mined sample.

    The data model types a nested column as `Optional[Any]`, so the value inside it is a plain
    Python list or dict however the row around it is typed. A body that took `_SYSTEM` at its word
    and read a row's field by attribute all the way down crashed on every attempt of one live
    build, at the same call, with the same AttributeError, and the retry could not help because the
    instruction that caused it was still true of the row and false of what the column held. The
    sample already beside the column says which it is and what one part is called, so the form that
    reads it is derived here rather than left to the model to infer.

    Everything is read off the schema: no name here comes from a domain.
    """
    types = (column.evidence or {}).get("types") or []
    sample = _sample_value(column)
    name = f"{table}.{column.name}"
    if "list" in types:
        element = next((v for v in sample if isinstance(v, dict)), None) if isinstance(sample, list) else None
        if element:
            key = next(iter(element))
            return (f"{name} is a list of plain dicts, not model rows; one element looks like: "
                    f"{json.dumps(element, default=str)}; read element[\"{key}\"], "
                    f"never element.{key}")
        return (f"{name} is a plain list, not a model row: index it, and read anything it holds by "
                "key, never by attribute")
    if "dict" in types:
        rows = sample if isinstance(sample, dict) else {}
        key, inner = next(((k, v) for k, v in rows.items() if isinstance(v, dict)), ("", None))
        if inner:
            keyed_by = next((f for f, v in inner.items() if v == key), "its own key")
            field = next(iter(inner))
            return (f"{name} is a dict of plain dicts keyed by {keyed_by}; read "
                    f"{column.name}[key][\"{field}\"], never {column.name}[key].{field}")
        return (f"{name} is a plain dict, not a model row: read it by key, as in "
                f"{column.name}[\"key\"], never by attribute")
    return ""


def _schema_block(schema: EntitySchema) -> str:
    """Tables and columns of the customer's world: the same for every tool in this build (D65's
    stable prefix, docs/prompt-caching.md item 1).

    A column mined as a dict gets one sample beside it, and a column that holds a list or a dict
    gets the form that reads it (`_column_access`). Mining cannot tell a customer's real table from
    a stand-in built out of a value nested inside another table's column: a tool that answers with
    one nested row makes mining propose a table for it, while the table that stores those rows
    keeps them in a column of its own. Both are real observations and nothing here decides which is
    the customer's real storage; the sample is what lets the model notice the second on its own,
    instead of trusting the proposed table alone and raising "not found" on a database that only
    fills it through the nesting. That is what happened on the first live build, nine calls out of
    nine.
    """
    lines = ["Tables on self.db:"]
    for table in sorted(schema.tables):
        columns = sorted((c for c in schema.columns if c.table == table), key=lambda c: c.name)
        names = ", ".join(c.name for c in columns) if columns else "(no columns observed)"
        lines.append(f"- {table}: {names}")
        fields = key_fields(schema, table)
        if len(fields) > 1:
            separator = key_separator(schema)
            form = separator.join("{" + name + "}" for name in fields)
            lines.append(f"    self.db.{table} is keyed by {' and '.join(fields)} together, joined "
                         f"with {separator!r}: the key of a row is f\"{form}\". One "
                         f"{fields[0]} stands for several rows here, one per "
                         f"{' and '.join(fields[1:])}, so build the whole key to reach a row and "
                         f"never look one up by {fields[0]} alone. A row's own "
                         f"{' or '.join(fields[1:])} column can be null where the customer's tool "
                         f"took the value from the call's arguments; the key is what says which row "
                         f"this is. Split a key on {separator!r} to read its parts back, and answer "
                         f"rows whose key parts match the arguments you were given.")
        if table == mine.constants_table_of(schema):
            lines.append(f"    self.db.{table} holds one row of the world's constants: every "
                         f"recorded call of the tool a column is named after answered that column's "
                         f"value, whatever it was asked. Read the row with "
                         f"next(iter(self.db.{table}.values())), never by key, and let such a tool "
                         f"answer its own column whole rather than assembling the answer out of "
                         f"other tables' rows.")
        home = (schema.homes or {}).get(table)
        if home:
            parent, column = home.split(".", 1)
            key = id_field(schema, table) or "its id"
            lines.append(f"    {table} rows are stored inside {home}, keyed by {key}: walk "
                         f"self.db.{parent}.values() and read .{column} to find one. self.db.{table} "
                         f"holds only rows the traces showed on their own and may be empty on the "
                         f"customer's real database, so look in {home} first and in self.db.{table} "
                         f"second.")
        for column in columns:
            if "dict" in (column.evidence or {}).get("types", []) and column.samples:
                lines.append(f"    {table}.{column.name} looks like: {column.samples[0]}")
            access = _column_access(table, column)
            if access:
                lines.append(f"    {access}")
    return "\n".join(lines)


def _confinement_block(denied: Iterable[str] = DENIED_BUILTINS, allowed: Iterable[str] = ALLOWED_IMPORTS) -> str:
    """The confinement gate's own rules, in words, generated from the gate's own constants.

    The first live build spent four of sixteen tools discovering these by being refused: the model
    reached for `getattr` and `__dict__` because nothing had told it not to, and each refusal cost
    an attempt to learn one rule. Written out here it costs nothing, because this text sits in the
    stable system prefix that every call of the stage reuses from the provider's cache.

    Generated rather than written so the two can never drift: if sandbox.py starts denying a name,
    the prompt says so on the next build without anyone remembering to edit it. The two lists are
    parameters defaulting to the gate's own constants, so a caller can show a reader (or a test)
    what a different gate would render without reaching into this module.

    The helper sentence sits here rather than in a per-tool message because it is the same for every
    tool of every build, so it is read out of the provider's cache and costs nothing. Build 12 spent
    four attempts on one tool watching the model write an arithmetic parser by hand, each one worse
    than the last, because nothing had said there was one already.
    """
    return ("The body is checked before it runs and is refused if it names anything outside the "
            "customer's world. It may not use: " + ", ".join(sorted(denied)) + ". It may "
            "not touch a dunder attribute (`__dict__`, `__class__`, `__globals__` and the rest). "
            "It may import only: " + ", ".join(sorted(allowed)) + ". Read fields by name "
            "(`order.status`) or by key (`self.db.orders[order_id]`), never through getattr. "
            "evaluate_arithmetic(expression) is provided in the body's namespace: it evaluates "
            "+ - * / // % ** with parentheses over a decimal and returns a Decimal, so never write "
            "a parser and never reach for eval; for example `float(evaluate_arithmetic(expression))` "
            "or round it to the places the recording shows.")


# D117: said once, only when compile_tool was asked to offer the two builder tools, so a caller
# that turns them off (or an old caller that never knew about them) gets exactly the old prompt.
_BUILDER_TOOLS_PARAGRAPH = (
    "Two tools are available while you write this body: lookup_rows(table, key=None) reads a row "
    "of the Starting state (or, with no key, that table's row count and a few sample keys), and "
    "test_body(body) runs a draft through the same gates this attempt will face, on the calls you "
    "were shown. Look a row up before guessing its shape; test the draft before you submit it. "
    "A tool call is never a submission: when a draft passes, send the body itself as your reply. "
    "The rounds are few; if they run out, the last body you tested is taken as your reply.")


def _stable_system(schema: Optional[EntitySchema] = None, tool_names: Iterable[str] = (),
                   builder_tools: bool = False, world_note: str = "") -> str:
    """`_SYSTEM` plus what every tool in this build shares: one prefix, sent unchanged on every
    call of the stage, long enough on a real customer to clear a provider's cache minimum.

    `world_note` is what a table another requestor revealed needs said about it
    (`builder/readers.py`): how to reach its one row, and the derivations of the columns nothing
    stores. It is the same bytes for every tool of a build, so it belongs in this prefix and not in
    the per-tool turn.
    """
    # The body skill (D168) sits in the prefix too: it is the same bytes for every tool of a build,
    # and it is read before the tables, which is where a body's mistakes are made.
    parts = [_SYSTEM, BODY_SKILL, _confinement_block()]
    if schema is not None:
        parts.append(_schema_block(schema))
    if world_note:
        parts.append(world_note)
    names = sorted(set(tool_names))
    if names:
        parts.append("Tools in this build: " + ", ".join(names))
    if builder_tools:
        parts.append(_BUILDER_TOOLS_PARAGRAPH)
    return "\n\n".join(parts)


def _tool_block(toolsig: ToolSig, examples: Iterable[ToolCall], error_prefix: Optional[str] = None,
                effects: str = "") -> str:
    """What differs call to call: this one tool, its signature, its recorded calls and its effects.

    `effects` (D215) is the section a write tool gets when the recording shows its calls moving rows
    their arguments never named: the columns, what each held on either side, and the formula the
    numbers imply. It sits above the recorded calls, because it is the part of the tool's behaviour
    the calls themselves do not show, and it is empty for every tool nothing was observed of, so a
    read's block is the bytes it always was.
    """
    if is_scalar_result(toolsig):
        result_line = ("Result: a bare " + _annotation(toolsig.result_schema[0].types)
                       + ", returned directly, not as {\"value\": ...} or any other wrapper object")
    else:
        result_line = "Result fields: " + ", ".join(f.name for f in toolsig.result_schema)
    parts = [f"Tool: {toolsig.name}",
             f"Description: {toolsig.description or 'not declared by the customer'}",
             "Arguments: " + ", ".join(f"{f.name} ({_annotation(f.types)})" for f in toolsig.args_fields),
             result_line]
    if effects:
        parts.append(effects)
    parts += ["Recorded calls:", _example_block(examples, error_prefix)]
    return "\n".join(parts)


def body_messages(toolsig: ToolSig, examples: Iterable[ToolCall], schema: Optional[EntitySchema] = None,
                  failure: str = "", tool_names: Iterable[str] = (),
                  error_prefix: Optional[str] = None, builder_tools: bool = False,
                  lesson: str = "", world_note: str = "", effects: str = "") -> list[dict]:
    """The whole message list one body request sends, so its size can be checked before it goes.

    The system message carries the fixed instructions plus what is the same for every tool in
    this build (the schema, the tool list): one prefix, unchanged call to call, for a provider's
    cache to reuse. The user message carries only this one tool, its recorded calls, and (for a
    one-shot request outside the repair loop) the failure of a previous attempt. `builder_tools`
    (D117) adds the one paragraph naming lookup_rows and test_body to the stable prefix; it is the
    same value on every call of one compile_tool, so the cached bytes never move mid-build.

    `lesson` is what earlier attempts at this one tool already failed on (memory.lesson_for): it
    belongs to this tool and not to the build, so it goes in the user turn and leaves the stable
    prefix alone. Empty, the messages are the same bytes they were before there was a lesson.

    `effects` (D215) belongs to this tool for the same reason and goes in the same turn: it is what
    this tool's own recorded calls were seen to change beyond their own answers.
    """
    user = _tool_block(toolsig, examples, error_prefix, effects)
    if lesson:
        user += "\n\n" + lesson
    if failure:
        user += "\n\nThe previous body failed these gates:\n" + failure
    return [{"role": "system", "content": _stable_system(schema, tool_names, builder_tools, world_note)},
            {"role": "user", "content": user}]


def _append_retry(messages: list[dict], reply_content: str, evidence: Iterable[ToolCall],
                  failure: str, error_prefix: Optional[str] = None) -> list[dict]:
    """A gate-failure retry (D75, docs/prompt-caching.md item 2): the messages so far are kept
    exactly as they were sent, so the system and first user turn stay the cached prefix; the
    model's previous reply arrives as an assistant turn and the new evidence and failure as a
    new user turn, never folded back into the first one."""
    turn = ("Recorded calls:\n" + _example_block(evidence, error_prefix)
            + "\n\nThe previous body failed these gates:\n" + failure)
    return messages + [{"role": "assistant", "content": reply_content},
                       {"role": "user", "content": turn}]


def _context_cap_error() -> tuple:
    """budget.py's D65 refusal, imported late so this module does not need budget.py to exist."""
    try:
        from kullback.runner.budget import ContextCapExceeded
        return (ContextCapExceeded,)
    except Exception:  # pragma: no cover - budget.py is always importable in this package
        return ()


def prompt_chars(messages: Iterable[dict]) -> int:
    """The size the D65 cap is checked against: the whole prompt, not the evidence block alone."""
    return sum(len(message.get("content") or "") for message in messages)


def write_tool_body(model, toolsig: ToolSig, examples: Iterable[ToolCall],
                    schema: Optional[EntitySchema] = None, failure: str = "") -> str:
    """Ask the Model for one tool body. Code owns the signature, the docstring and the schema."""
    return _body_from(model, body_messages(toolsig, examples, schema, failure))


def _body_from(model, messages: list[dict]) -> str:
    reply = model.query(messages)
    return _strip_fence(reply.content or "")


def _strip_fence(text: str) -> str:
    """Take the code out of a fenced reply and dedent it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        text = "\n".join(lines[:-1] if lines and lines[-1].strip().startswith("```") else lines)
    return textwrap.dedent(text).strip("\n")

# --- the two tools the body-writing model may call (D117) ---
#
# A body-writing model has to guess a row's shape from the schema block and the recorded calls
# alone, and has to guess whether its own draft clears the gates. lookup_rows answers the first
# guess and test_body answers the second, both without spending a repair attempt to find out.
# Neither is allowed to touch the held-out split: lookup_rows only reads db and the shown calls'
# own worlds, and test_body only ever gates the shown calls, with an empty held-out list, so
# nothing the held-out replay would have caught can leak back through either tool.

LOOKUP_ROWS_TOOL = {
    "name": "lookup_rows",
    "description": ("Read one row of the Starting state by table and key, or (with no key) a "
                    "table's row count and a few sample keys. Read-only: nothing here changes "
                    "the world. Never returns a held-out row."),
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "description": "the table to look in"},
            "key": {"type": "string", "description": "the row's id; omit to see the table's shape"},
        },
        "required": ["table"],
    },
}

TEST_BODY_TOOL = {
    "name": "test_body",
    "description": ("Run a draft body through the same gates this attempt will face, on the "
                    "calls you were shown (never the held-out ones). Returns which gate failed "
                    "and why, or that every gate passed."),
    "parameters": {
        "type": "object",
        "properties": {"body": {"type": "string", "description": "the full body to test, as Python source"}},
        "required": ["body"],
    },
}

BUILDER_TOOLS = [LOOKUP_ROWS_TOOL, TEST_BODY_TOOL]


def _find_row(schema: EntitySchema, world: dict, table: str, key: str) -> tuple[Optional[dict], Optional[str]]:
    """A row and where it sits: the table itself, or (schema.homes) the parent it is nested in."""
    row = (world.get(table) or {}).get(key)
    if isinstance(row, dict):
        return row, table
    home = (schema.homes or {}).get(table)
    if home:
        parent, column = home.split(".", 1)
        for parent_row in (world.get(parent) or {}).values():
            nest = parent_row.get(column) if isinstance(parent_row, dict) else None
            if isinstance(nest, dict) and isinstance(nest.get(key), dict):
                return nest[key], home
    return None, None


def _shown_worlds(db: dict, shown: list[ToolCall], call_states: Optional[dict]) -> list[dict]:
    """db, then every distinct world a shown call ran on (D74's per-Task overlay), never a held-out
    one: the same call_states lookup Sandbox.state_for makes, read here off the dict directly
    since no Sandbox exists yet when the model is still drafting the body."""
    seen = {id(db)}
    worlds = [db]
    for call in shown if call_states else []:
        state = call_states.get(call.id, db) if call.id else db
        if id(state) not in seen:
            seen.add(id(state))
            worlds.append(state)
    return worlds


def _lookup_rows_text(schema: EntitySchema, db: dict, shown: list[ToolCall], call_states: Optional[dict],
                      table: Optional[str] = None, key: Optional[str] = None) -> str:
    """What lookup_rows answers: a row, a table's shape, or the table list for an unknown name."""
    tables = sorted(schema.tables)
    if table not in tables:
        return f"unknown table {table!r}; tables on self.db: {', '.join(tables)}"
    if not key:
        rows = db.get(table) or {}
        return f"{table}: {len(rows)} rows; sample keys: {sorted(rows)[:3]}"
    row, location = None, None
    for world in _shown_worlds(db, shown, call_states):
        row, location = _find_row(schema, world, table, key)
        if row is not None:
            break
    if row is None:
        return f"{table} row {key!r} was not found in the Starting state or a shown call's world"
    text = json.dumps(row, sort_keys=True, default=str)
    note = ""
    if len(text) > 2000:
        text, note = text[:2000], " (truncated to 2000 characters)"
    where = f"table {table}" if location == table else f"nested inside {location} (schema.homes)"
    return f"{table} row {key} is stored in {where}{note}: {text}"


def _build_tools_impl(schema: EntitySchema, toolsig: ToolSig, shown: list[ToolCall], db: dict,
                      call_states: Optional[dict], workdir: Path, attempt: int, timeout: float,
                      rules: Any, readers: Any = None,
                      effect_values: Optional[dict] = None) -> dict[str, Callable[..., str]]:
    """lookup_rows and test_body, closed over one attempt's own evidence and probe directory.

    test_body gates on `shown` alone, with an empty held-out list: the split the repair loop keeps
    hidden from the model stays hidden from the model's own probing too, not just from the failure
    text a rejected attempt is shown.
    """
    probes = {"n": 0}

    def lookup_rows(table: Optional[str] = None, key: Optional[str] = None) -> str:
        return _lookup_rows_text(schema, db, shown, call_states, table, key)

    def test_body(body: Optional[str] = None) -> str:
        probes["n"] += 1
        source = module_source(schema, [toolsig], {toolsig.name: body or ""})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}_probe_{probes['n']}", timeout=timeout,
                          call_states=call_states)
        gates = run_gates(source, sandbox, shown, [], schema, rules,
                          probe_refusals=toolsig.kind == "write", sig=toolsig, readers=readers,
                          effect_values=effect_values)
        if all(g.passed for g in gates):
            return "passed every gate: " + ", ".join(g.stage for g in gates)
        return _failure_text(gates)

    return {"lookup_rows": lookup_rows, "test_body": test_body}


def _tool_use_record(call: Any, result_text: str) -> dict:
    """What a tool use is remembered as on the node: never the row or the body, only what was asked."""
    if call.name == "test_body":
        body = (call.arguments or {}).get("body") or ""
        arguments: dict = {"body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    else:
        arguments = {"table": (call.arguments or {}).get("table"), "key": (call.arguments or {}).get("key")}
    return {"name": call.name, "arguments": arguments, "result_chars": len(result_text)}


def _run_builder_tool(call: Any, tools_impl: dict[str, Callable[..., str]]) -> str:
    impl = tools_impl.get(call.name)
    if impl is None:
        return f"unknown tool {call.name!r}; available tools: {', '.join(sorted(tools_impl))}"
    try:
        return impl(**(call.arguments or {}))
    except TypeError as exc:
        return f"bad arguments for {call.name}: {exc}"


def _assistant_tool_turn(reply: Any) -> dict:
    """The canonical assistant turn a tool-calling reply becomes, the same shape the Runner's own
    loop and provider.py's adapters read (runner.loop._assistant_message)."""
    return {"role": "assistant", "content": reply.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls]}


def _reply_with_tools(model, messages: list[dict], tools_impl: dict[str, Callable[..., str]],
                      max_rounds: int = MAX_TOOL_ROUNDS) -> tuple[str, list[dict], str]:
    """Query the model with lookup_rows and test_body on, executing whatever it calls (D117).

    `messages` (the attempt's own system and user turns, exactly as compile_tool built them) is
    never mutated: the tool exchange runs over a local copy, so the next attempt's retry still
    appends its assistant reply and new evidence onto the plain chain `_append_retry` expects,
    never onto a transcript of tool calls. At most `max_rounds` model calls; a reply that asks for
    no tool is the body. The third value is the last body the model handed to test_body, so a
    model that tests a draft and keeps probing until the rounds run out (the sixth retail build's
    cancel_pending_order: three drafts tested, then a row lookup, and no reply) still hands the
    caller the draft it was working on rather than nothing.
    """
    working = list(messages)
    tool_uses: list[dict] = []
    reply = None
    draft = ""
    for _ in range(max_rounds):
        reply = model.query(working, tools=BUILDER_TOOLS)
        if not reply.tool_calls:
            return reply.content or "", tool_uses, draft
        working = working + [_assistant_tool_turn(reply)]
        for call in reply.tool_calls:
            result_text = _run_builder_tool(call, tools_impl)
            tool_uses.append(_tool_use_record(call, result_text))
            if call.name == "test_body" and (call.arguments or {}).get("body"):
                draft = str(call.arguments["body"])
            working = working + [{"role": "tool", "tool_call_id": call.id, "content": result_text}]
    return (reply.content or "") if reply is not None else "", tool_uses, draft

# --- the bounded repair loop (D75) ---

def split_calls(calls: Iterable[ToolCall], every: int = 3) -> tuple[list[ToolCall], list[ToolCall]]:
    """The held-out split the LLM is never shown (D51, D75): every third call, deterministically."""
    calls = list(calls)
    if len(calls) < 2:
        return calls, []
    held_out = [c for i, c in enumerate(calls) if i % every == every - 1]
    shown = [c for i, c in enumerate(calls) if i % every != every - 1]
    return (shown, held_out) if held_out else (shown[:-1], shown[-1:])


def _failing_calls(gates: list[GateResult], shown: list[ToolCall]) -> list[ToolCall]:
    """The shown calls a gate named in its failures; empty when no shown call was named."""
    named = [f for gate in gates if not gate.passed for f in gate.failures]
    return [c for c in shown if any(args_text(c) in f for f in named)]


def _held_out_only(gates: list[GateResult]) -> bool:
    """True when the held-out replay is the only gate that failed, so no shown call can be shown."""
    failed = [g for g in gates if not g.passed]
    return bool(failed) and all(g.metrics.get("split") == "held_out" for g in failed)


def _constant_evidence_note(evidence: Iterable[ToolCall]) -> str:
    """Said once, when the calls just shown already share one answer among themselves.

    Gate 4 fails a body that answers every call the same way, and the failure line reads like an
    instruction to make the answers differ. For a tool whose own recorded calls differ in nothing
    but their arguments, that is not a defect the next attempt can write its way out of; it is
    what the evidence says the tool does. Left unsaid, the model takes gate 4 at its word and
    invents a categorization the traces never showed. transfer_to_human_agents did exactly that on
    the first live build: every recorded call answered "Transfer successful" whatever the summary,
    three repair attempts each added logic to make the answers differ, and the last used a name it
    never imported and crashed on every call.
    """
    results = [canon(parse_result(call.result)) for call in evidence if call.error is None]
    if len(results) < 2 or len(set(results)) != 1:
        return ""
    return ("\nThe recorded calls above already answer every one of them the same way. If your body "
            "does too, that may be the correct behaviour; do not invent a detail the calls never "
            "showed just to make the answers differ.")


def _failure_text(gates: list[GateResult], held_out: Iterable[ToolCall] = ()) -> str:
    """What the next attempt is told, with the held-out split kept out of it (D51, D75).

    The held-out calls are the ones the model is never shown, so neither their arguments nor the
    outcome expected of them may travel back in a repair prompt. A failure on them is reported as a
    count, which tells the model its body is wrong somewhere without handing it the answers. Gates 2
    to 4 run over every call, so their failure lines are filtered by the same rule.
    """
    hidden = [args_text(call) for call in held_out]
    lines = []
    for gate in gates:
        if gate.passed:
            continue
        split = gate.metrics.get("split", "")
        kept = [] if split == "held_out" else [f for f in gate.failures if not any(h in f for h in hidden)]
        withheld = len(gate.failures) - len(kept)
        text = "; ".join(kept)
        if withheld:
            text += ("; " if text else "") + f"{withheld} more on calls you were not shown"
        lines.append(f"- gate {gate.stage} ({split}): {text}")
    lines += _import_hints(gates)
    return "\n".join(lines)


_RAISED = re.compile(r"\braised ([A-Za-z_][A-Za-z0-9_.]*): (.+)")


def exception_in(failure: str) -> str:
    """The exception a gate's failure line quotes, as `ExceptionClass: message`; "" when it quotes none.

    The executes gate writes `<tool>(<args>) raised <class>: <message>` (gates/tool_runs.py), so the
    class and the message are what is left when the call in front of them is dropped. Two attempts
    that crashed the same way then compare equal whatever calls their failures named.
    """
    match = _RAISED.search(failure)
    return f"{match.group(1)}: {match.group(2).strip()}" if match else ""


def exception_line(gates: Iterable[GateResult]) -> str:
    """The first exception the failing gates report; "" when none of them reports one."""
    for gate in gates:
        if gate.passed:
            continue
        for failure in gate.failures:
            found = exception_in(failure)
            if found:
                return found
    return ""


# Said only when the attempt that just ran raised what the attempt before it raised. One live build
# spent all four attempts at one write tool on the same AttributeError at the same call: the retry
# showed the crash each time and the model rewrote around it each time, because nothing said the
# error had not moved. This goes in the new user turn, never in the cached prefix (D65).
_SAME_ERROR = ("\nThe same error as the attempt before, at the same call. The body must change "
               "where it raised: {error}")

_UNDEFINED_NAME = re.compile(r"NameError: name '(\w+)' is not defined")


def _import_hints(gates: list[GateResult]) -> list[str]:
    """A NameError on a module the sandbox allows names the fix; say it instead of the traceback.

    transfer_to_human_agents called re.findall on the first live build and never imported re, and
    every one of its 25 replays died on the same NameError. The gate already knew which name was
    missing and that the module is on the allowed list; the retry was handing back the raw error.
    """
    names = sorted({m.group(1) for gate in gates if not gate.passed
                    for failure in gate.failures for m in _UNDEFINED_NAME.finditer(failure)})
    return [f"- `{name}` is on the allowed import list but the body never imported it; put "
            f"`import {name}` at the top of the body" for name in names if name in ALLOWED_IMPORTS]


RAISED_KINDS = ("KeyError", "AttributeError")
RAISED_SAMPLES = 3
RAISED_LINES = 2
_TABLE_READ = re.compile(r"self\.db\.([A-Za-z_][A-Za-z0-9_]*)")


def value_shape(value: str) -> str:
    """One value with its characters generalized: digits `#`, lowercase `a`, uppercase `A`, the rest itself.

    The shape of an id is what the next attempt needs (an id of this table looks like this); the id
    itself is a customer value and there is no reason to hand back three of them.
    """
    return "".join("#" if c.isdigit() else "a" if c.islower() else "A" if c.isupper() else c
                   for c in str(value))


def table_read(line: str) -> str:
    """The world table a crashing source line reads, from `self.db.<table>`; "" when it reads none."""
    found = _TABLE_READ.findall(line or "")
    return found[0] if found else ""


def missing_key(result: dict) -> str:
    """The key a KeyError names, without its quotes; "" for any other exception."""
    if result.get("error") != "KeyError":
        return ""
    message = (result.get("message") or "").strip()
    quoted = len(message) > 1 and message[0] == message[-1] and message[0] in "'\""
    return message[1:-1] if quoted else message


def table_note(state: dict, table: str, key: str) -> str:
    """What the world says about the table the body was reading, and about the key it asked for."""
    rows = state.get(table) if isinstance(state, dict) else None
    if not isinstance(rows, dict):
        return f"the world holds no table `{table}`"
    ids = sorted(str(k) for k in rows)
    note = f"`{table}` holds {len(ids)} rows"
    if key:
        note += f" and not `{key}`" if key not in rows else f", `{key}` among them"
    if ids:
        shapes = ", ".join(dict.fromkeys(value_shape(i) for i in ids[:RAISED_SAMPLES]))
        note += f"; its ids look like {shapes}"
    return note


def raised_detail(sandbox: Sandbox, calls: Iterable[ToolCall], limit: int = RAISED_LINES) -> str:
    """What the world says about a KeyError or an AttributeError the body raised, a line per crash.

    A body that reads a row nobody holds is told `KeyError: 'x'` and nothing else, so the next
    attempt guesses: it wraps the read in a try, or reaches for another table, or invents a
    fallback answer. The sandbox knows more than it was saying. The crashing source line names the
    table the body was reading (`self.db.<table>`), and the world in front of it says how many rows
    that table holds, whether the key asked for is one of them, and what the ids it does hold look
    like. That is the difference between "your read raised" and "you read the wrong table".

    Only the calls given here are named, so the caller passes the shown split and the held-out
    calls stay out of it (D51, D75). The sandbox memo makes this free: these calls have already run
    for the gate that failed.
    """
    calls = list(calls)
    try:
        results = sandbox.run(calls)
    except SandboxError:
        return ""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for call, result in zip(calls, results, strict=False):
        if result.get("ok") or result.get("error") not in RAISED_KINDS:
            continue
        line = (result.get("line") or "").strip()
        table = table_read(line)
        key = (table, result.get("error") or "")
        if not table or key in seen:
            continue
        seen.add(key)
        note = table_note(sandbox.state_for(call), table, missing_key(result))
        lines.append(f"- `{call.name}({args_text(call)})` raised {result['error']} at `{line}`: "
                     f"{note}; read the arguments that call was given, not another table.")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _evidence_for(attempt: int, shown: list[ToolCall], gates: list[GateResult]) -> list[ToolCall]:
    """Evidence grows per attempt: the failing call, then all failing calls, then the full table.

    When no shown call is named in the failures (the held-out split failed on its own, or the module
    did not parse at all), there is no failing call to point at, so the attempt gets every shown call
    rather than the first one, which passed.
    """
    if attempt == 0:
        return shown[:3]
    failing = _failing_calls(gates, shown)
    if not failing:
        return shown
    return failing[:1] if attempt == 1 else (failing if attempt == 2 else shown)


def attempt_score(gates: Iterable[GateResult]) -> tuple[int, int]:
    """How far one attempt's body got, as the key the kept body is chosen by.

    `run_gates` runs the gates in one order (parses, confined, executes_on_s0, deterministic,
    non_trivial, replay_fidelity, refuses_unknown) and stops at the first failure, so the count of
    gates a body passed is how far down that order it reached. The tie-break is how many recorded
    calls its replay actually matched, which separates two bodies that fell at the same gate.
    """
    gates = list(gates)
    passed = sum(1 for gate in gates if gate.passed)
    matched = sum(int(gate.metrics.get("success_matches") or 0)
                  for gate in gates if gate.stage == "replay_fidelity")
    return passed, matched


def body_could_not_run(gates: Iterable[GateResult], rows: Iterable[dict]) -> bool:
    """True when this body answered none of the recorded calls: it cannot be a candidate at all.

    A body kept from an earlier run of the stage is replayed under the world as it stands now, and
    the world may have moved out from under it: a table it read is gone, a column it indexed is
    renamed, and every call raises. Such a body still has a score, and on a corpus where the new
    attempt is no better the tie rule would hand the tool back to it. So it is told apart from a
    body that merely scores badly, and the tie does not save it.

    Answering nothing is read off the per-call rows and not off the executes ruling, because that
    ruling counts only the exceptions that are a fault of the module itself: a KeyError is how a
    tool refuses an id it does not hold (`CRASH_ERRORS`), and a body whose table is gone raises
    exactly that on every call. The rows say what each call actually answered (`answer_digest`
    marks a raise) and whether it matched the recording, so the reading is: every recorded call
    raised and none of them matched. Before the rows there is the module itself: one that does not
    parse, or a sandbox that could not run it, answered nothing either. A tool with no recorded
    call has nothing to have failed and is never this.
    """
    for gate in gates:
        if gate.stage == "parses" and not gate.passed:
            return True
        # The sandbox's own failure (the module did not load, or the run timed out) leaves no
        # crash count, only the message; a ruling with a count is about the body, not the module.
        if gate.stage == "executes_on_s0" and not gate.passed and "crashes" not in gate.metrics:
            return True
    answered = [row for row in rows if row.get("answer") is not None]
    return bool(answered) and all(str(row["answer"]).startswith("error:") and not row.get("replayed")
                                  for row in answered)


def _read_pair(tool: str, theirs: Any, ours: Any, readers: Any) -> tuple[Any, Any]:
    """Two prose answers of one tool as the columns its reader finds in them (D176, D187).

    The same rule `column_differences` compares under: a tool that answers with one sentence is
    read into its columns first, and where the reader finds none on either side the two sentences
    stay one value and part or do not part as a whole. Without it every relation over a prose tool
    is asked of a string, which holds no elements to be broadcast and no list to be ordered.
    """
    if readers is None or not tool or not isinstance(theirs, str) or not isinstance(ours, str):
        return theirs, ours
    if readers.table_of(tool) is None:
        return theirs, ours
    left, right = readers.read(tool, theirs), readers.read(tool, ours)
    return (left, right) if isinstance(left, dict) and isinstance(right, dict) else (theirs, ours)


def call_triples(toolsig: ToolSig, calls: Iterable[ToolCall], results: Iterable[dict],
                 outcomes: Iterable[dict], readers: Any = None) -> list[lesson_mod.Triple]:
    """One recorded call, what it answered and what the body answered, as the relation step reads it (D211).

    The triples are built here and not off `tool_call_outcomes.json`, which keeps a digest of each
    answer and not the answer: a digest says two answers differ and can say nothing about how, and
    every relation in the catalogue is a statement about how. The recorded side is the call's own
    result or the class it raised; ours is what the sandbox just answered for the same call, which
    the gates have already run and the memo already holds.
    """
    matched = {str(row.get("call_id") or ""): bool(row.get("replayed"))
               for row in outcomes if isinstance(row, dict)}
    triples: list[lesson_mod.Triple] = []
    for call, result in zip(calls, results, strict=False):
        ours = result.get("value") if isinstance(result, dict) and result.get("ok") else None
        our_error = "" if not isinstance(result, dict) or result.get("ok") else str(
            result.get("error") or "an error")
        theirs = parse_result(call.result) if call.error is None else None
        theirs, ours = _read_pair(toolsig.name, theirs, ours, readers)
        triples.append(lesson_mod.Triple(
            call_id=str(call.id or ""), args=dict(call.args or {}),
            theirs=theirs,
            ours=ours,
            their_error="" if call.error is None else str(getattr(call.error, "class_", "") or "an error"),
            our_error=our_error,
            matched=matched.get(str(call.id or ""), False)))
    return triples


def diagnose_body(toolsig: ToolSig, source: str, sandbox: Sandbox, calls: list[ToolCall],
                  shown: list[ToolCall], outcomes: list[dict], gates: list[GateResult],
                  unbeaten: int = 0, blocked: str = "", rules: Any = None, readers: Any = None) -> Any:
    """The code-only read of one body that is already there: full diff, relation, witnessed lines (D211).

    Nothing here calls a model. The answers are the sandbox's memo, already paid for by the gates
    that just ran; the one new subprocess is the traced run behind `executed_lines`, and it is
    started only for a body whose calls the sandbox reached at all, since a body that does not
    parse has no lines to have run and no answers to compare.

    The triples are the shown split's alone (D51, D75): a lesson is written into a prompt, and the
    held-out calls' arguments and outcomes may not travel there, which is the same rule
    `_failure_text` and `kept_body_hint` are filtered by. The witnessed-branch read is over every
    recorded call, because it carries no value of any of them: a line number is a fact about the
    body, and a branch reached only by a held-out call is a branch the evidence supports.
    """
    ran = any(gate.stage == "executes_on_s0" for gate in gates)
    if not calls or not ran:
        return lesson_mod.diagnose(toolsig.name, (), unbeaten=unbeaten, blocked=blocked)
    try:
        results = sandbox.run(shown)
    except SandboxError:
        return lesson_mod.diagnose(toolsig.name, (), unbeaten=unbeaten, blocked=blocked)
    triples = call_triples(toolsig, shown, results, outcomes, readers)
    carried = (lesson_mod.FAILING_SET_SHOWN if lesson_mod.stalled(unbeaten)
               else lesson_mod.EXCEPTIONS_NAMED)
    return lesson_mod.diagnose(toolsig.name, triples, source=source, function=toolsig.name,
                               executed=sandbox.executed_lines(calls), unbeaten=unbeaten,
                               blocked=blocked, rules=rules, shown=carried)


def grade_body(toolsig: ToolSig, body: str, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
               workdir: Path | str, call_states: Optional[dict] = None, rules: Any = None,
               timeout: float = 30.0, readers: Any = None,
               call_tasks: Optional[dict] = None, unbeaten: int = 0, blocked: str = "",
               effect_values: Optional[dict] = None) -> ToolBuild:
    """Run one body that already exists through the gates and the per-call replay, with no model call.

    This is `compile_tool` with the writing taken out: the same gates in the same order, the same
    per-call attribution and the same `hardcoded` reading, over a body the caller hands in. It is
    what makes a body kept from an earlier run of the stage a candidate on equal terms with the
    attempts the writer produces, since the two can only be compared under one world by the same
    ruling, and the score written on a tool's row was measured under the world of the run that
    wrote it. A stage whose inputs moved (the schema, the Starting state, the readers, the tables
    block) rewrites every body from scratch, and until now the rewrite was kept whatever it scored:
    one live round rewrote every body over a change to a world constant and took one write tool
    from 83 of its 139 recorded calls matched to 39, with nothing about that tool's own inputs
    having moved.
    """
    workdir, calls = Path(workdir), list(calls)
    build = ToolBuild(name=toolsig.name, body=body or "")
    shown, held_out = split_calls(calls)
    source = module_source(schema, [toolsig], {toolsig.name: build.body})
    sandbox = Sandbox(source, db, workdir, timeout=timeout, call_states=call_states,
                      call_tasks=call_tasks)
    build.gates = run_gates(source, sandbox, shown, held_out, schema, rules,
                            probe_refusals=toolsig.kind == "write", sig=toolsig, readers=readers,
                            effect_values=effect_values)
    build.assisted = not (build.gates and all(gate.passed for gate in build.gates))
    build.call_outcomes = (
        replay_outcomes(toolsig, build.body, calls, schema, db, workdir, call_states=call_states,
                        rules=rules, timeout=timeout, readers=readers)
        if build.assisted else
        [{"tool": toolsig.name, "call_id": call.id, "replayed": True, "detail": ""} for call in calls])
    build.could_not_run = body_could_not_run(build.gates, build.call_outcomes)
    build.hardcoded = build.assisted and hardcoded_body(calls, build.call_outcomes)
    if build.assisted:
        build.diagnosis = diagnose_body(toolsig, source, sandbox, calls, shown, build.call_outcomes,
                                        build.gates, unbeaten=unbeaten, blocked=blocked, rules=rules,
                                        readers=readers)
    return build


KEPT_BODY_HEAD = ("This tool already has a body in this build. It was replayed under the world as it "
                  "stands now and it failed as follows. You are not shown that body: a body in the "
                  "prompt is a body copied, and what is wanted here is a second, better one. Write "
                  "your own, and do not fail in these ways.")


def kept_body_hint(gates: Iterable[GateResult], held_out: Iterable[ToolCall] = (),
                   shapes_shown: int = SHAPES_SHOWN) -> str:
    """What the body already kept for this tool fails at, grouped by shape; "" when it fails at nothing.

    The hint goes where every other hint goes (the lesson in the first user turn), and it is grouped
    by shape with counts for the reason D181's rule 6 gives: a gate that ruled over sixty calls
    leaves sixty sentences, and a writer shown one of them answers one of them. The held-out split
    is filtered the way `_failure_text` filters it, so a body the writer is asked to beat still
    cannot hand it the calls it was never shown.

    `shapes_shown` is three by default and the whole failing set once the tool has stalled (D211):
    three shapes is a sample, and a writer that has answered the sample six times over and bought
    nothing needs the rest of the distribution rather than the same three again.
    """
    hidden = [args_text(call) for call in held_out]
    lines = []
    for gate in gates:
        if gate.passed:
            continue
        split = gate.metrics.get("split", "")
        kept = [] if split == "held_out" else [f for f in gate.failures if not any(h in f for h in hidden)]
        shapes = failure_shapes(kept)
        text = "; ".join(f"{shape} ({count} call{'' if count == 1 else 's'})"
                         for shape, count in shapes[:shapes_shown])
        if len(shapes) > shapes_shown:
            text += f"; {len(shapes) - shapes_shown} more shapes"
        withheld = len(gate.failures) - len(kept)
        if withheld:
            text += ("; " if text else "") + f"{withheld} more on calls you were not shown"
        lines.append(f"- gate {gate.stage}: {text}")
    return "\n".join([KEPT_BODY_HEAD] + lines) if lines else ""


def _evidence_label(attempt: int, gates: list[GateResult]) -> str:
    """The node's own name for its evidence, honest about the held-out case."""
    if attempt and _held_out_only(gates):
        return HELD_OUT_LABEL
    return EVIDENCE_LABELS[min(attempt, len(EVIDENCE_LABELS) - 1)]


def call_starting_states(db: dict, overlays: Iterable[TaskOverlay], values: dict,
                         call_tasks: dict, run_overlays: Optional[dict] = None,
                         call_runs: Optional[dict] = None) -> dict:
    """Call id to the Starting state of the Run whose trace recorded it (D74, D213).

    `call_tasks` maps a recorded call's id to its Task id; it is what carries the trace-to-Task map
    into the gates, since a `ToolCall` does not name its trace. The state is the shared world with
    that Task's overlay merged, which is the world that call actually ran on, so a corpus holding one
    row in two versions does not make a correct body look wrong.

    D213: where the Task's own Runs recorded one column in two values, the Task overlay does not
    hold that column and one layer per Run does. `call_runs` maps a call to the Run that made it and
    `run_overlays` holds those layers per Task, so a call is scored on the world its own recording
    saw and not on the version another Run of the same Task read first. Without either the Task's
    overlay answers for every call, which is what a caller that has no Run map gets.

    D197: where the Run's overlay carries a pin sequence, the call that saw a later value of a
    time-varying row runs on a world holding that value and not the first one. The scoring path and
    the replay have to serve one world or the score punishes a body for a state the replay would
    have served it. The state is keyed per call already, so the sequence needs no new key: a call a
    step names gets its own copy of that Run's world with that step's columns laid in it, and every
    other call keeps the world as it was pinned.
    """
    run_overlays = run_overlays or {}
    call_runs = call_runs or {}
    served: dict[tuple[str, str], TaskOverlay] = {}
    for overlay in overlays:
        served[(overlay.task_id, "")] = overlay
        for run_id, scoped in (run_overlays.get(overlay.task_id) or {}).items():
            served[(overlay.task_id, run_id)] = scoped
    # One world per distinct set of pinned rows, not one per Run: a Run that scopes nothing pins the
    # same rows as its Task and reads the same world, and a corpus of hundreds of Runs would
    # otherwise hold hundreds of copies of the customer's whole world at once.
    worlds: dict[tuple, dict] = {}
    states: dict[tuple[str, str], dict] = {}
    for key, overlay in served.items():
        rows = tuple(sorted((row.table, row.id, row.version_hash) for row in overlay.rows))
        if rows not in worlds:
            worlds[rows] = merge_overlays(db, [overlay], values)
        states[key] = worlds[rows]
    per_call: dict[str, dict] = {}
    for key, overlay in served.items():
        base = states.get(key)
        for step in overlay.steps:
            if base is None or not step.call_id or not step.values:
                continue
            if call_runs.get(step.call_id, "") != key[1]:
                continue  # another Run's layer holds this call's own sequence
            world = per_call.get(step.call_id)
            if world is None:
                world = per_call[step.call_id] = copy.deepcopy(base)
            row = world.setdefault(step.table, {}).get(step.id)
            world[step.table][step.id] = (dict(row, **step.values) if isinstance(row, dict)
                                          else dict(step.values))
    out: dict[str, dict] = {}
    for call_id, task_id in call_tasks.items():
        key = (task_id, call_runs.get(call_id, ""))
        if key not in states:
            key = (task_id, "")
        if key in states:
            out[call_id] = per_call.get(call_id) or states[key]
    return out


def replay_outcomes(toolsig: ToolSig, body: str, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
                    workdir: Path | str, call_states: Optional[dict] = None, rules: Any = None,
                    timeout: float = 30.0, readers: Any = None) -> list[dict]:
    """Which of these recorded calls the kept body answers the way the recording did, one row per call (D171).

    The corpus gate (`gate_replay_fidelity`) rules over all of a tool's recorded calls at once and
    leaves one fidelity number and a list of failure sentences. That is the right grain for the
    Builder, which repairs a body, and the wrong grain for a Task, which only ever makes some of
    those calls: on one live build a body replayed 239 of its 240 calls and the one miss was read as
    a fault of the tool everywhere the tool was called. So the sandbox is run once over every call
    and the same ruling, `body_replay_fidelity_gate`, is then asked about each call on its own. A
    call counted replayed here is a call the corpus gate counted matched, because it is the same
    ruling over the same result; what is added is the call's id, which is what ties a miss to the
    Trace and so to the Task that made it.

    A body that does not exist, or a module that will not run at all, misses every call: there is
    nothing to attribute a pass to. `rules` is the customer's CanonRules, the same ones the corpus
    gate compares under (D39).
    """
    calls = list(calls)
    if not calls:
        return []

    def missed(detail: str) -> list[dict]:
        return [{"tool": toolsig.name, "call_id": call.id, "replayed": False, "detail": detail}
                for call in calls]

    if not (body or "").strip():
        return missed("no body was compiled for this tool")
    source = module_source(schema, [toolsig], {toolsig.name: body})
    sandbox = Sandbox(source, db, Path(workdir) / "attribution", timeout=timeout, call_states=call_states)
    try:
        results = sandbox.run(calls)
    except SandboxError as exc:
        return missed(f"the module did not run: {exc}")
    rows: list[dict] = []
    for index, call in enumerate(calls):
        if index >= len(results):
            rows.append({"tool": toolsig.name, "call_id": call.id, "replayed": False,
                         "detail": "the sandbox returned no result for this call"})
            continue
        ruling = body_replay_fidelity_gate([call], [results[index]], schema, label="per_call", rules=rules,
                                           readers=readers)
        detail = "" if ruling.passed else (ruling.failures[0] if ruling.failures else "differs")
        rows.append({"tool": toolsig.name, "call_id": call.id, "replayed": bool(ruling.passed),
                     "detail": _clamped_detail(detail), "answer": answer_digest(results[index]),
                     "world": sandbox.state_key(call)})
    return rows


def answer_digest(result: Any) -> str:
    """One sandbox answer as a comparable key: the canonicalized value, or the class it raised.

    Values themselves are never kept here. What the digest is for is telling two answers apart, and
    a hash does that without a customer's row travelling into an artifact that is read as a summary.
    """
    if isinstance(result, dict) and not result.get("ok"):
        return f"error:{result.get('error') or 'unknown'}"
    value = result.get("value") if isinstance(result, dict) else result
    return content_hash(canon(value))


def recorded_digest(call: ToolCall) -> str:
    """The same key for what the recording answered, so the two can be compared."""
    if call.error is not None:
        return f"error:{getattr(call.error, 'class_', '') or 'unknown'}"
    return content_hash(canon(parse_result(call.result)))


HARDCODED_MARK = "# hardcoded: this body answered every recorded call alike; read what it is given"
HARDCODED_LESSON = ("the body kept answered every recorded call the same way whatever arguments and "
                    "whatever world it was given, while the recordings answered them differently; the "
                    "next body has to answer out of its arguments and the rows in front of it")


def mark_hardcoded(body: str) -> str:
    """The body as it is kept, with one comment line saying what is wrong with it.

    bodies.json holds source, and source is what the next reader of it sees: the Builder asking for
    the body back, the module the Runner loads, a person opening the file. A comment says it in all
    three without changing what the body does, which a kept body must not do.
    """
    body = body or ""
    return body if body.lstrip().startswith(HARDCODED_MARK) else f"{HARDCODED_MARK}\n{body.lstrip()}"


def hardcoded_body(calls: Iterable[ToolCall], rows: Iterable[dict]) -> bool:
    """True when the body answered differing inputs identically and the recordings did not.

    A kept assisted body is a body no attempt got through the gates, and the gates say which one it
    fell at; none of them says the body never read what it was given. One live build kept a body
    that picked the first row of a table and raised one fixed refusal on every call, and the round
    after it, and the round after that: assisted was all anyone was told, and a hint written against
    a single failing call cannot reach a body that reads nothing. Three things have to hold
    together: the recorded calls carry more than one input, the body gave them all one answer, and
    the recordings did not.

    An input is the arguments and the world the call ran on, not the arguments alone. A tool that
    takes no arguments at all is answered out of the Starting state of the Task that called it, and
    that is exactly the tool the live build kept: every call carried `{}`, the recordings answered
    two different ways, and the body answered one way whatever world it stood on. Where the
    recordings themselves answer everything the same way, that is what the tool does
    (`_constant_evidence_note`), and this stays quiet.
    """
    by_id = {row["call_id"]: row for row in rows if row.get("call_id")}
    inputs, body_answers, recorded = set(), set(), set()
    for call in calls:
        row = by_id.get(call.id) or {}
        if not call.id or row.get("answer") is None:
            continue
        inputs.add((args_text(call), str(row.get("world") or "")))
        body_answers.add(row["answer"])
        recorded.add(recorded_digest(call))
    return len(inputs) > 1 and len(body_answers) == 1 and len(recorded) > 1


def _clamped_detail(text: str) -> str:
    """One difference, short enough to sit in a Task's status row.

    A tool that returns a whole catalogue writes a failure sentence thousands of characters long,
    and the row is read by a person and by the repair verb, not by the comparer; the gate's own
    failures keep the sentence whole.
    """
    text = " ".join(text.split())
    if len(text) <= OUTCOME_DETAIL_CHARS:
        return text
    return f"{text[:OUTCOME_DETAIL_CHARS].rstrip()} [+{len(text) - OUTCOME_DETAIL_CHARS} characters]"


def compile_tool(model, toolsig: ToolSig, calls: Iterable[ToolCall], schema: EntitySchema, db: dict,
                 workdir: Path | str, max_attempts: int = MAX_REPAIR_ATTEMPTS,
                 max_evidence_chars: Optional[int] = MAX_EVIDENCE_CHARS, timeout: float = 30.0,
                 call_states: Optional[dict] = None, rules: Any = None,
                 tool_names: Iterable[str] = (), error_prefix: Optional[str] = None,
                 builder_tools: bool = True, lesson: str = "", world_note: str = "",
                 readers: Any = None, call_tasks: Optional[dict] = None,
                 effects: str = "", effect_values: Optional[dict] = None) -> ToolBuild:
    """Write one tool body, gate it, and repair it at most three times with growing evidence (D75).

    Attempt 1 sees the failing call, attempt 2 every failing call, attempt 3 the full call table, and
    every rewrite is re-checked on the shown calls and on the held-out calls separately, with the
    held-out calls never named back to the model. An attempt whose whole prompt would exceed the D65
    cap is refused, not truncated, and the cap is on by default. `call_states` (from
    `call_starting_states`) is the per-Task world each recorded call ran on; without it every call
    replays on the shared db, which is right only where the corpus shows one version of each row.
    `call_tasks` is the map behind it, call id to Task id, which is what lets gate 8 name the two
    Tasks a sensitivity pair spans (D195); without it a pair is named by the Traces its calls
    came from.
    `rules` is the customer's CanonRules (D39): given none, the gates compare under the module
    defaults and can fail a body over a difference the customer's own rules fold away. `tool_names`
    is every tool name in this build, stable across every call of this stage, so it lives in the
    system message beside the schema (docs/prompt-caching.md item 1).

    `effects` and `effect_values` are D215's two halves of one observation (`builder/effects.py`):
    the section of the prompt that says which rows this tool's recorded calls moved beyond the ones
    their arguments named, and the values those rows were left holding, which the memorised gate
    refuses a body for writing down. Both are empty for a tool nothing was observed of, and the
    prompt is then the bytes it always was.

    Attempt 0 sends the system and the first user turn; a retry (docs/prompt-caching.md item 2)
    never rewrites either: it appends the previous reply as an assistant turn and the new evidence
    and failure as a new user turn, so the first two messages of every call in the loop are the
    same bytes a cache can reuse. An attempt that raised what the attempt before it raised is told
    so in that new turn (`_SAME_ERROR`), with the exception quoted, since a rewrite that lands on
    the same crash is not a rewrite of the line that crashed.
    An attempt whose body raised a KeyError or an AttributeError also gets `raised_detail`: the
    line it crashed on, the table that line reads and what the world says about that table, since
    the bare exception says nothing a rewrite can aim at.
    Each attempt is a node dict; after the last miss the tool is marked assisted (D49) and the nodes
    are written under the workdir. When no attempt passes, the body kept is the best attempt's, not
    the last one's (`attempt_score`, `ToolBuild.kept_attempt`).

    `builder_tools` (D117, on by default) lets the model call `lookup_rows` and `test_body` while
    it drafts this attempt's reply, inside `_reply_with_tools`'s own bounded loop
    (`MAX_TOOL_ROUNDS`), before the reply it settles on is gated the same way an old, tool-less
    reply always was. Turn it off for a caller that wants the old one-call-per-attempt behaviour.

    The build carries `call_outcomes` (D171): one row per recorded call saying whether the kept body
    answered it the way the recording did, so a miss can be attributed to the Task whose Trace made
    the call and not to every Task that calls the tool.

    `lesson` is what this tool already failed on in an earlier build or an earlier recompile
    request (`memory.lesson_for`, written by `repair.record_tool_lesson`). It is carried into the
    first user turn, so it is in the prefix every retry of this attempt chain keeps; without it the
    recompile asks the same question again and the model has no way to know it was asked before.
    """
    workdir, calls = Path(workdir), list(calls)
    if error_prefix is None:  # build.py passes the corpus-wide prefix; alone, this tool's own calls
        error_prefix = shared_error_prefix(calls)
    shown, held_out = split_calls(calls)
    build, failure = ToolBuild(name=toolsig.name, body=""), ""
    skeleton = gate_parses(module_source(schema, [toolsig], {toolsig.name: "pass"}))
    messages: list[dict] = []
    reply_content = ""
    # The loop grows its evidence off the attempt that just ran, not off the best one so far, so
    # the two are kept apart: `latest` drives the next prompt, `build` holds what is kept.
    latest: list[GateResult] = []
    best: Optional[tuple[int, int]] = None
    last_error = ""  # the exception the attempt before raised, so a repeat can be named as one
    for attempt in range(max_attempts + 1):
        if not skeleton.passed:  # code owns the skeleton, so no model call can repair it
            build.gates = [skeleton]
            build.nodes.append({"attempt": attempt, "tool": toolsig.name, "evidence": "none",
                                "evidence_calls": 0, "passed": False, "refused": True,
                                "failures": [f"the code-owned skeleton does not parse: {f}"
                                             for f in skeleton.failures]})
            break
        evidence = _evidence_for(attempt, shown, latest)
        node = {"attempt": attempt, "tool": toolsig.name, "evidence": _evidence_label(attempt, latest),
                "evidence_calls": len(evidence), "passed": False, "refused": False}
        if attempt == 0:
            messages = body_messages(toolsig, evidence, schema=schema, tool_names=tool_names,
                                     error_prefix=error_prefix, builder_tools=builder_tools,
                                     lesson=lesson, world_note=world_note, effects=effects)
        else:
            messages = _append_retry(messages, reply_content, evidence, failure, error_prefix)
        # Fewer whole calls, never a shortened one. `_example_block` refuses to cut a call in
        # half and that stays true: what is dropped here is the last recorded call, entire. The
        # first live build refused `get_order_details` outright at 815,972 characters, because the
        # last-resort evidence is every shown call and this corpus has hundreds of them; a body
        # written from thirty complete calls is worth more than an attempt not taken.
        while (max_evidence_chars is not None and len(evidence) > 1
               and prompt_chars(messages) > max_evidence_chars):
            evidence = evidence[:-1]
            node["evidence_calls"] = len(evidence)
            messages = (body_messages(toolsig, evidence, schema=schema, tool_names=tool_names,
                                      error_prefix=error_prefix, builder_tools=builder_tools,
                                      lesson=lesson, world_note=world_note, effects=effects)
                        if attempt == 0
                        else _append_retry(messages[:-2], reply_content, evidence, failure, error_prefix))
        size = prompt_chars(messages)
        if max_evidence_chars is not None and size > max_evidence_chars:
            node["failures"] = [f"a prompt of {size} characters is over the cap "
                                f"of {max_evidence_chars}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        tools_impl = (_build_tools_impl(schema, toolsig, shown, db, call_states, workdir, attempt,
                                        timeout, rules, readers, effect_values)
                     if builder_tools else None)
        try:
            if builder_tools:
                reply_content, tool_uses, draft = _reply_with_tools(model, messages, tools_impl)
            else:
                reply_content, tool_uses, draft = model.query(messages).content or "", [], ""
        except _context_cap_error() as refusal:
            # `max_evidence_chars` above is not enough on its own, and the first live build is
            # what showed it. It counts the characters of the message contents; budget.py counts
            # the tokens of the JSON the request will actually carry, which is larger by the
            # envelope and by every escaped quote and newline in a tool result. A prompt of
            # 358,580 content characters was under the 320,000 cap on nothing and over the 80,000
            # token cap at 89,645. No constant reconciles two different measures, so the one that
            # is authoritative is the one the wrapper raises, and it is caught here.
            #
            # Before this it left compile_tool and killed the build: one tool with a large corpus
            # took the other thirteen with it. The stage already has a word for a tool it could not
            # write, so the refusal becomes that word: assisted (D49), and the build carries on.
            node["failures"] = [f"{refusal}; refused, not truncated"]
            build.nodes.append(dict(node, refused=True))
            break
        if tool_uses:
            node["tool_uses"] = tool_uses
        body = _strip_fence(reply_content)
        if not body and draft:
            # The rounds ran out on a model that had tested a draft and was still probing: that
            # draft is what it was submitting, so it is gated like any reply (D117).
            body, reply_content = _strip_fence(draft), draft
            node["body_from_draft"] = True
        if not body:
            # No reply and no draft: nothing to gate, and gating "pass" would blame the code-owned
            # skeleton for a reply the model never gave. The attempt fails and the next one runs
            # with that said; the sixth retail build failed whole on a refusal here, because the
            # stage gate reads an empty body as a tool the Builder could not write.
            failure = "\nno body was submitted: every round asked for a tool; send the body as your reply"
            reply_content = "(no body was submitted)"
            last_error = ""  # nothing ran, so the next attempt's error follows no error of ours
            node["failures"] = ["no body was submitted"]
            build.nodes.append(node)
            continue
        source = module_source(schema, [toolsig], {toolsig.name: body})
        sandbox = Sandbox(source, db, workdir / f"attempt_{attempt}", timeout=timeout,
                          call_states=call_states, call_tasks=call_tasks)
        gates = run_gates(source, sandbox, shown, held_out, schema, rules,
                          probe_refusals=toolsig.kind == "write", sig=toolsig, readers=readers,
                          effect_values=effect_values)
        node.update(body_hash=content_hash(body), gates=[as_dict(g) for g in gates],
                    passed=all(g.passed for g in gates))
        build.nodes.append(node)
        latest = gates
        # The last attempt is not the best attempt. A live build's fourth attempt at one write tool
        # crashed on all 67 of its calls where the third had replayed 44 of 44 and failed only the
        # refusal probe; keeping the last one cost 94 replay misses and 38 Tasks. So the body kept
        # is the one that got furthest through the gates (`attempt_score`), and a later attempt
        # replaces it only by beating it. A passing attempt ends the loop, so it is always the best.
        score = attempt_score(gates)
        if best is None or score > best:
            best, build.body, build.gates, build.kept_attempt = score, body, gates, attempt
        if node["passed"]:
            break
        failure = "\n" + _failure_text(gates, held_out)
        if any(g.stage == "non_trivial" and not g.passed for g in gates):
            failure += _constant_evidence_note(evidence)
        # Only when the sandbox has already run these calls, so the note costs no subprocess: a
        # body that never parsed has nothing for the world to say about it anyway.
        detail = raised_detail(sandbox, shown) if sandbox.cache else ""
        failure += ("\n" + detail) if detail else ""
        error = exception_line(gates)
        if error and error == last_error:
            failure += _SAME_ERROR.format(error=error)
        last_error = error
    build.assisted = not (build.gates and all(g.passed for g in build.gates))
    for record in build.nodes:
        if record["attempt"] == build.kept_attempt:
            record["kept"] = True
    # D171: a tool that cleared every gate replayed every recorded call, since `run_gates` holds the
    # shown and the held-out split to 100% each and only calls a build unassisted when both passed;
    # so its per-call rows are known without a sandbox and only an assisted tool pays for the run.
    build.call_outcomes = (
        replay_outcomes(toolsig, build.body, calls, schema, db, workdir, call_states=call_states,
                        rules=rules, timeout=timeout, readers=readers)
        if build.assisted else
        [{"tool": toolsig.name, "call_id": call.id, "replayed": True, "detail": ""} for call in calls])
    build.hardcoded = build.assisted and hardcoded_body(calls, build.call_outcomes)
    directory = workdir / NODE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{toolsig.name}.json").write_text(
        json.dumps({"tool": toolsig.name, "assisted": build.assisted,
                    "hardcoded": build.hardcoded,
                    "kept_attempt": build.kept_attempt, "nodes": build.nodes,
                    "call_outcomes": build.call_outcomes},
                   indent=2, default=str) + "\n", encoding="utf-8")
    return build

# --- emitting tau2's five files plus the sidecar (D56) ---

def _action(atom: Atom) -> dict:
    """A required atom becomes a tau2 action: the tool it names, with the value it pins as arguments.

    The Verifier's structured target is the source (`Atom.target`); an atom that spelled the call out
    as JSON in `predicate_src` still works, and one that names neither falls back to its description.
    """
    call = dict(atom.target)
    if not call and atom.predicate_src:
        try:
            parsed = json.loads(atom.predicate_src)
            call = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            call = {}
    arguments = call.get("arguments") or (
        {call["field"]: call.get("raw")} if call.get("kind") == "write_value" else {})
    return {"action_id": atom.id, "name": call.get("name") or call.get("tool") or atom.description,
            "arguments": arguments, "info": None}


def _tau2_task(task: Task, verifier: Optional[Verifier], domain: str) -> dict:
    """One entry of tasks.json; everything tau2's shape has no room for goes to the sidecar."""
    atoms = verifier.atoms if verifier else []
    return {
        "id": task.id,
        "description": {"purpose": task.intent, "relevant_policies": None, "notes": None},
        "user_scenario": {"persona": None, "instructions": {
            "task_instructions": None, "domain": domain, "reason_for_call": task.intent,
            "known_info": None, "unknown_info": None}},
        "initial_state": None,
        "evaluation_criteria": {"actions": [_action(a) for a in atoms if a.kind == "required"]},
    }


def build_environment(schema: EntitySchema, sigs: Iterable[ToolSig], bodies: dict, policy_text: str,
                      builds: Optional[dict] = None, parent_env_id: Optional[str] = None,
                      version: int = 1, files: Optional[dict] = None,
                      assisted_tools: Iterable[str] = ()) -> Environment:
    """The Environment record: identity, the D97 sub-versions, assisted tools and the flags.

    `env_id` is the hash of the five emitted files plus the three sub-versions (design section 5).
    `files` is the file name to file text map `tau2_files` returns; hand it in whenever the emitted
    world is known, because without it two worlds holding different rows, or different Tasks, share
    one env_id and a regrade cannot tell them apart. The hashes are kept on `Environment.files`.

    Flags are what the setup review has to close before the Environment is trusted: a tool whose
    errors are mostly `unknown` (D67) and a tool whose read or write class nobody confirmed (D70).

    `assisted_tools` are the names the compile_tools stage marked assisted (D49), by name because the
    stage hands its builds on as JSON; the second retail build wrote `assisted_tools: []` on an
    Environment with six assisted tools because nothing passed them in.
    """
    from kullback.builder.mine import (
        unknown_error_flags,  # builder to builder; the Runner imports neither
    )

    sigs = list(sigs)
    schema_version = content_hash(as_dict(schema))[:12]
    tools_version = content_hash({"sigs": [as_dict(s) for s in sigs], "bodies": bodies})[:12]
    policy_version = content_hash(policy_text)[:12]
    flags = unknown_error_flags(sigs)
    flags += [f"{sig.name}: read or write not confirmed, defaulted to read (D70)"
              for sig in sorted(sigs, key=lambda s: s.name) if sig.unclassified]
    file_hashes = {name: content_hash(text) for name, text in sorted((files or {}).items())}
    return Environment(
        env_id=content_hash({"schema": schema_version, "tools": tools_version,
                             "policy": policy_version, "files": file_hashes}),
        schema_version=schema_version, tools_version=tools_version, policy_version=policy_version,
        version=version, parent_env_id=parent_env_id, files=file_hashes,
        assisted_tools=sorted({name for name, build in (builds or {}).items() if build.assisted}
                              | set(assisted_tools)),
        flags=flags,
    )


def tau2_files(env: EnvBundle) -> dict:
    """The five tau2 files as text, keyed by name: what is written, and what env_id hashes.

    Overlays merge into the one db.json tau2's harness loads; a conflict between two Tasks is recorded
    on `env.conflicts` for the export gate, and the Task stays gradeable in the Runner (D74).
    """
    db = merge_overlays(env.db, env.overlays, env.overlay_values, env.conflicts) if env.overlays else env.db
    verifiers = {v.task_id: v for v in env.verifiers}
    return {
        "data_model.py": render_data_model(env.schema),
        "tools.py": render_tools(env.schema, env.tools, env.bodies),
        "db.json": json.dumps(db, sort_keys=True, indent=2) + "\n",
        "policy.md": env.policy_text,
        "tasks.json": json.dumps([_tau2_task(t, verifiers.get(t.id), env.domain) for t in env.tasks],
                                 indent=2) + "\n",
    }


def emit_tau2_shape(env: EnvBundle, workdir: Path | str, files: Optional[dict] = None) -> dict:
    """Write data_model.py, tools.py, db.json, policy.md, tasks.json and sidecar.json (D56).

    A caller that already computed `tau2_files(env)` for its own purposes (build.py's environment
    stage needs the files to size the build_environment gate) passes it in, so the render does not
    run a second time on the same inputs.
    """
    workdir = Path(workdir)
    files = tau2_files(env) if files is None else files
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in files.items():
        (workdir / name).write_text(text, encoding="utf-8")
        paths[name] = workdir / name
    sidecar = {
        "env_id": env.environment.env_id,
        "schema_version": env.environment.schema_version,
        "tools_version": env.environment.tools_version,
        "policy_version": env.environment.policy_version,
        "files": {name: content_hash(text) for name, text in files.items()},
        "assisted_tools": list(env.environment.assisted_tools),
        "assumptions": list(env.assumptions),
        "synthetic_rows": list(env.schema.synthetic_rows),
        "overlays": [as_dict(o) for o in env.overlays],
        "atoms": {v.task_id: [as_dict(a) for a in v.atoms] for v in env.verifiers},
        # Which tables another requestor's own tools revealed rather than the assistant's (R33). A
        # reader who takes db.json for the customer's system has to be able to see which rows are not.
        "revealed_by": {c.table: str((c.evidence or {}).get("revealed_by"))
                        for c in sorted(env.schema.columns, key=lambda c: (c.table, c.name))
                        if (c.evidence or {}).get("revealed_by")},
    }
    paths["sidecar.json"] = workdir / "sidecar.json"
    paths["sidecar.json"].write_text(json.dumps(sidecar, indent=2, default=str) + "\n", encoding="utf-8")
    return paths
