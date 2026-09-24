"""What one recorded call witnessed for the tool context, read off the recording alone.

A body that mints an id or stamps a time asks the tool context (`self.ctx`), and at replay the
context answers what the recording showed for that call: its new ids, in recording order, and its
time. This is the recipe that reads those values off the recorded calls, keyed per call by
`context_feed_key`. It lives in the runner so the runner's own replay can feed each call before
its body runs; the Builder's sandbox and compile step import it from here, so the recipe is one.
It also holds `holdout_values` (D220 rule 2c), the other reading of the recording both sides share.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.gates.tool_runs import id_field
from kullback.runner.records import EntitySchema, ToolCall

# Calls without a recorded position order after positioned calls, by call id, so the result never depends on input order.
MISSING_MSG_INDEX = 1 << 62


def recorded_call_parts(call: Any) -> tuple:
    """The five parts of a recorded call, for a ToolCall or a dict."""
    if isinstance(call, ToolCall):
        return (call.id, call.trace_id, call.raw_ptr, call.args or {}, call.result)
    node = call or {}
    return (node.get("id"), node.get("trace_id"), node.get("raw_ptr"), node.get("args") or {}, node.get("result"))


def context_feed_key(call: Any) -> tuple:
    """The identity of one recorded call for its tool context feed: trace, position, id.

    A call id alone does not name a call: ingestion permits an id issued again after the
    earlier call resolved, and several traces travel in one list. The triple tells those
    apart; a missing trace or id reads as "" and a missing position orders last. Two calls
    sharing the whole triple cannot be told apart and are never fed.
    """
    call_id, trace_id, raw, _, _ = recorded_call_parts(call)
    position = raw.get("msg_index") if isinstance(raw, dict) else getattr(raw, "msg_index", None)
    if not isinstance(position, int):
        position = MISSING_MSG_INDEX
    return (trace_id or "", position, call_id or "")


# A result value that reads as a calendar date or a clock time, the way created_at does.
# The text lives here alone: TIME_RE compiles it, and the Builder's tool context shim receives
# the same text by placeholder, so the two can never disagree about what a time is.
TIME_PATTERN = (r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?"
                 r"(Z|[+-]\d{2}:?\d{2})?)?$|^\d{2}:\d{2}(:\d{2})?$")
TIME_RE = re.compile(TIME_PATTERN)


def is_id_key(key: Any) -> bool:
    """A result key that names a row's id: `id`, or any key ending in `_id`."""
    name = str(key).lower()
    return name == "id" or name.endswith("_id")


def result_leaves(result: Any) -> list[tuple[str, Any]]:
    """Every (key, leaf) of a recorded result, in a deterministic order.

    Dicts walk in sorted key order so the first time-like leaf of a result is the same one on
    every invocation; lists walk in order, since reordering a list would hand a different call's
    time to this one.
    """
    out: list[tuple[str, Any]] = []

    def walk(node: Any, key: str) -> None:
        if isinstance(node, dict):
            for name in sorted(node, key=str):
                walk(node[name], str(name))
        elif isinstance(node, (list, tuple)):
            for inner in node:
                walk(inner, key)
        else:
            out.append((key, node))

    walk(result, "")
    return out


def _tables_for_id_key(schema: EntitySchema, key: str) -> list[str]:
    """The tables whose id column this result key names, by the miner's own rule.

    The rule is `tool_runs.id_field`, the same one that reads a table's id column everywhere
    else: a key it attributes to no table carries no recorded feed, and the seeded feed answers
    for that table instead. A key two tables share (a user id echoed on the row it made) is
    recorded under both, so whichever table the body asks for gets what the recording showed.
    """
    return sorted({table for table in (schema.tables or [])
                   if id_field(schema, table) == key})


def _context_id_observations(node: Any, schema: EntitySchema) -> tuple[set, set]:
    """The id sightings of one arguments or result node: (pairs, blocked).

    Pairs are (table, value) for an id key the miner's rule attributes to a table. Blocked
    values were seen under an id key no table claims, so they suppress that value for every
    table. A non id key of a result states no id and is ignored here: a note echoing the new
    id must not cost the creation its recorded id.
    """
    pairs: set = set()
    blocked: set = set()
    for key, value in result_leaves(node):
        if not isinstance(value, str) or not is_id_key(key):
            continue
        tables = _tables_for_id_key(schema, key)
        if tables:
            pairs.update((table, value) for table in tables)
        else:
            blocked.add(value)
    return pairs, blocked


def _context_arg_observations(args: Any, schema: EntitySchema) -> tuple[set, set]:
    """The id sightings of one call's arguments: (pairs, blocked).

    Like the result rule, except a value under a key that is not an id key also suppresses
    that value for every table: the body was given it, so it generated nothing.
    """
    pairs, blocked = _context_id_observations(args, schema)
    blocked.update(value for key, value in result_leaves(args)
                   if isinstance(value, str) and not is_id_key(key))
    return pairs, blocked


def _context_seen_strings(args: Any, result: Any) -> set:
    """Every string two nodes state, for the time rule's earlier observation check."""
    return ({value for _, value in result_leaves(args) if isinstance(value, str)}
            | {value for _, value in result_leaves(result) if isinstance(value, str)})


def _context_id_tables(key: Any, value: Any, schema: EntitySchema, own_pairs: set,
                       own_blocked: set, prior_pairs: set, prior_blocked: set) -> list:
    """The tables one id leaf is new for, possibly none.

    Seen under another table's key the value still feeds; seen under the same table's key, under
    a key no table claims, or among the call's own argument strings, it feeds nothing.
    """
    if not isinstance(value, str) or not is_id_key(key):
        return []
    if value in own_blocked or value in prior_blocked:
        return []
    return [table for table in _tables_for_id_key(schema, key)
            if (table, value) not in own_pairs and (table, value) not in prior_pairs]


def _context_new_ids(result: Any, schema: EntitySchema, own_pairs: set, own_blocked: set,
                     prior_pairs: set, prior_blocked: set) -> dict:
    """The result's new ids as an ordered list per table, in recording order, no duplicates."""
    found: dict = {}
    for key, value in result_leaves(result):
        for table in _context_id_tables(key, value, schema, own_pairs, own_blocked,
                                        prior_pairs, prior_blocked):
            ids = found.setdefault(table, [])
            if value not in ids:
                ids.append(value)
    return found


def _context_witnessed_time(result: Any, arg_strings: set, prior_seen: set) -> Optional[str]:
    """The result's one surviving time-like leaf, or None when zero or several survive.

    A leaf among the call's argument strings or observed earlier in the same trace is what the
    call found, not what it made. Two differing survivors are ambiguous, so they feed nothing;
    equal values are one value.
    """
    survivors: list = []
    for _, value in result_leaves(result):
        if not isinstance(value, str) or not TIME_RE.match(value.strip()):
            continue
        if value in arg_strings or value in prior_seen:
            continue
        if value not in survivors:
            survivors.append(value)
    if len(survivors) == 1:
        return survivors[0]
    return None


def _context_creation_rows(result: Any, schema: EntitySchema, own_pairs: set,
                           own_blocked: set, prior_pairs: set, prior_blocked: set) -> list:
    """The row objects directly holding a filed new id, in walk order.

    Only a row the call created speaks for the call's time: a time leaf in another row
    object of the same result is not creation evidence.
    """
    rows: list = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if any(_context_id_tables(key, value, schema, own_pairs, own_blocked,
                                      prior_pairs, prior_blocked)
                   for key, value in node.items()):
                rows.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(result)
    return rows


def _context_creation_stamp(rows: list, arg_strings: set) -> Optional[str]:
    """The one distinct time-like leaf across the created rows, or None.

    Equal times from two row objects are one survivor; zero or several feed nothing.
    """
    survivors: list = []
    for row in rows:
        for value in row.values():
            if not isinstance(value, str) or not TIME_RE.match(value.strip()) \
                    or value in arg_strings:
                continue
            if value not in survivors:
                survivors.append(value)
    if len(survivors) == 1:
        return survivors[0]
    return None


def _context_stamp_ids(rows: list, stamp: str, schema: EntitySchema, own_pairs: set,
                       own_blocked: set, prior_pairs: set, prior_blocked: set) -> list:
    """Ownership groups of the rows that supplied the surviving time.

    One group per distinct (key, value) leaf, in walk order, no duplicates: the key decides
    which tables could own the value, so equal values under different keys are never merged.
    """
    groups: list = []
    seen: set = set()
    for row in rows:
        if stamp not in {value for value in row.values() if isinstance(value, str)}:
            continue
        for key, value in row.items():
            if not isinstance(value, str) or (key, value) in seen:
                continue
            tables = _context_id_tables(key, value, schema, own_pairs, own_blocked,
                                        prior_pairs, prior_blocked)
            if tables:
                seen.add((key, value))
                groups.append({"key": key, "value": value, "tables": list(tables)})
    return groups


def _context_feed_now(result: Any, schema: EntitySchema, state: dict, own_pairs: set,
                      own_blocked: set, arg_strings: set, new_ids: dict) -> tuple:
    """The call's witnessed time, how it was evidenced, and the ids behind the evidence.

    Creation evidence wins: when the result shows a new id, the time candidates are the
    time-like leaves in the same row object, and equality with an earlier sighting or a
    stored stamp proves nothing, so only the call's own argument strings exclude a value.
    One distinct survivor feeds with "created_row" evidence and the ids of the rows that
    supplied it; otherwise the found-value rule decides over every time-like leaf, with
    neither evidence nor ids.
    """
    if new_ids:
        prior_pairs = state.get("pairs") or set()
        prior_blocked = state.get("blocked") or set()
        rows = _context_creation_rows(result, schema, own_pairs, own_blocked,
                                      prior_pairs, prior_blocked)
        stamp = _context_creation_stamp(rows, arg_strings)
        if stamp is not None:
            return stamp, "created_row", _context_stamp_ids(
                rows, stamp, schema, own_pairs, own_blocked, prior_pairs, prior_blocked)
    return _context_witnessed_time(result, arg_strings, state.get("seen") or set()), None, None


def recorded_call_context(call: ToolCall, schema: EntitySchema,
                          prior: Optional[dict] = None) -> dict:
    """The values one recorded call witnessed for the tool context: its new ids and its time.

    The new ids are the id-valued leaves of the recorded result, in recording order with no
    duplicates, filed as an ordered list under each table the miner's id rule attributes the
    key to. A top level list result reads like a dict result, so one call creating several rows
    feeds every id. An id is new only for its own table: seen under another table's key it still
    feeds, seen under the same table's key or under a key no table claims it feeds nothing. A
    value the call's own arguments state is what the body was given, so it feeds nothing.
    The time follows creation evidence where there is any, else the found-value rule (not
    in the arguments, not seen earlier in the trace, not stored at reset, one survivor); a
    feed the creation rule produced carries "now_evidence" and "now_ids" (one ownership
    group per id leaf of the rows that supplied the time: key, value, and the tables that
    key could own), so the context can judge the evidence leaf by leaf. A call whose result
    carries neither leaves both empty, and the seeded feed answers, counted.
    """
    feed: dict = {"now": None, "new_ids": {}}
    args = call.args if isinstance(call, ToolCall) else (call or {}).get("args") or {}
    result = call.result if isinstance(call, ToolCall) else (call or {}).get("result")
    if not isinstance(result, (dict, list, tuple)):
        return feed
    state = prior or {}
    own_pairs, own_blocked = _context_arg_observations(args, schema)
    feed["new_ids"] = _context_new_ids(result, schema, own_pairs, own_blocked,
                                       state.get("pairs") or set(),
                                       state.get("blocked") or set())
    arg_strings = {value for _, value in result_leaves(args) if isinstance(value, str)}
    feed["now"], evidence, now_ids = _context_feed_now(
        result, schema, state, own_pairs, own_blocked, arg_strings, feed["new_ids"])
    if evidence is not None:
        feed["now_evidence"] = evidence
        feed["now_ids"] = now_ids
    return feed


def _observe_context_call(state: dict, schema: EntitySchema, args: Any, result: Any) -> None:
    """Fold one call's arguments and result into a trace's prior observations.

    Across calls only id sightings persist: (table, value) pairs, and values under an id key
    no table claims. A plain argument string (a search query that found nothing) observed no
    entity, so it stays local to the call that stated it and never blocks a later new id.
    Every string still joins the seen strings for the time rule.
    """
    own_pairs, own_blocked = _context_id_observations(args, schema)
    res_pairs, res_blocked = _context_id_observations(result, schema)
    state["pairs"] |= own_pairs | res_pairs
    state["blocked"] |= own_blocked | res_blocked
    state["seen"] |= _context_seen_strings(args, result)


def recorded_call_contexts(calls: Iterable[ToolCall], schema: EntitySchema) -> dict[tuple, dict]:
    """One recorded feed per call, keyed by the call's feed key (trace, position, id).

    Two calls sharing a whole key cannot be told apart: neither is fed, and the seeded feed
    answers both, counted as seeded. The trace's prior observations still fold both calls in.
    """
    calls = sorted(list(calls), key=context_feed_key)
    feeds: dict[tuple, dict] = {}
    seen: set = set()
    dropped: set = set()
    observed: dict[Any, dict] = {}
    for call in calls:
        key = context_feed_key(call)
        _, trace_id, _, args, result = recorded_call_parts(call)
        if key in dropped:
            dupe = True
        elif key in seen:
            dropped.add(key)
            feeds.pop(key, None)
            dupe = True
        else:
            seen.add(key)
            dupe = False
        if not dupe:
            if trace_id is None:
                feeds[key] = recorded_call_context(call, schema)
            else:
                state = observed.setdefault(trace_id, {"pairs": set(), "blocked": set(),
                                                       "seen": set()})
                feeds[key] = recorded_call_context(call, schema, state)
        if trace_id is not None:
            state = observed.setdefault(trace_id, {"pairs": set(), "blocked": set(),
                                                   "seen": set()})
            _observe_context_call(state, schema, args, result)
    return feeds


def holdout_values(db: dict, columns: dict) -> dict[str, str]:
    """Every scalar the world holds only because a held-out Run witnessed it, as text to its column.

    The twin of `gates.tool_runs.recorded_result_values`: a body whose source spells one of these
    out did not derive it from the world it is allowed to see, whatever else its literals look like.
    A one-character value is skipped, because a flag or a digit is not a value anyone memorised.
    """
    out: dict[str, str] = {}
    for table, rows in sorted((columns or {}).items()):
        for row_id, names in sorted((rows or {}).items()):
            row = ((db or {}).get(table) or {}).get(row_id) or {}
            for name in names:
                value = row.get(name)
                if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
                    continue
                text = str(value)
                if len(text) > 1:
                    out.setdefault(text, f"{table}.{name}")
    return out
