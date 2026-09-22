"""Run end and transcript readers: how a Run ended, and whether a write call moved the world.

Moved out of kullback/user/rules.py unchanged, so the rules and these readers come apart (D214).
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

from kullback.user.rules import GAVE_UP, HANDED_OFF, STOP_REASONS, TURN_LIMIT_REASONS, USER_END_KINDS


def end_kind_of(payload: Any) -> Optional[str]:
    """The end kind one recorded user turn carries, in either shape the two users write it in.

    The rule-driven user tags the turn it ends on and the Runner copies the tags into the Run's own
    file; the agent user writes the kind under `user_end`. Both are the same fact, and a reader that
    knows only one of the two shapes silently reads every Run written in the other as a Run that
    ended in no kind at all. One function, so there is one answer.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("user_end") in USER_END_KINDS:
        return str(payload["user_end"])
    for tag in payload.get("tags") or ():
        if tag in USER_END_KINDS:
            return str(tag)
    return None


def end_of_run(run: Any) -> Optional[str]:
    """How this Run ended, in the four kinds (D210), or nothing where it ended in neither vocabulary.

    The Simulated user tags the turn it ends on, and the loop copies a turn's tags into the Run's
    own file, so the kind survives into the record without the Runner having to know about it. A
    Run the user never ended is read from the Runner's own end reason instead: the turn limit is
    `gave_up`, and a Candidate that stopped or was handed over is `handed_off`. A Run that crashed
    ended in neither vocabulary and is left unclassified rather than guessed at.
    """
    for event in reversed(list(getattr(run, "events", None) or [])):
        if getattr(event, "type", None) != "user_turn":
            continue
        kind = end_kind_of(getattr(event, "payload", None))
        if kind is not None:
            return kind
    reason = getattr(run, "termination_reason", None) or ""
    if reason in TURN_LIMIT_REASONS:
        return GAVE_UP
    return HANDED_OFF if reason in STOP_REASONS else None


def ends_by_kind(rows: Iterable[Any]) -> dict[str, int]:
    """How many Runs ended each way, over rows carrying a `user_end` (D210). Every kind is named,
    zero included, so a round that stopped ending one way says so rather than dropping the line."""
    counts = {kind: 0 for kind in USER_END_KINDS}
    for row in rows or ():
        kind = row.get("user_end") if isinstance(row, dict) else getattr(row, "user_end", None)
        if kind in counts:
            counts[kind] += 1
    return counts


class FactLookup(NamedTuple):
    """What the Starting state holds for a field; synthetic rows make the turn Assisted (D40, D77)."""
    value: Any
    synthetic: bool = False


def dict_reader(mapping: dict):
    """A Starting state reader over a plain dict, for tests and small callers."""

    class _DictReader:
        def get(self, field: str):
            return mapping.get(field)
    return _DictReader()


def _field_of(message: Any, key: str) -> str:
    if isinstance(message, dict):
        return message.get(key) or ""
    return getattr(message, key, "") or ""


def _calls_of(message: Any) -> list:
    """The tool calls one transcript message carries, whichever shape the caller builds it in."""
    calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    return list(calls or [])


def _attr(message: Any, key: str) -> Any:
    """One field of a transcript message, whichever shape it is in, with None kept as None.

    `_field_of` folds a missing field to the empty string because its callers want a name to
    compare. A marker's absence and a marker holding nothing are different things here, so this
    keeps them apart.
    """
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def write_took_effect(message: Any) -> bool:
    """Whether one write call's result shows the write took effect (D227).

    The Runner records a tool result as `result` plus, where the world refused or failed the call,
    `error`: the D67 ToolCallError carrying that refusal's `class` in the recorded taxonomy. That
    marker is the whole of the refusal reading, because it is the one field the Runner writes on
    every route it can refuse on, whatever payload the tool answered with. A call carrying it
    changed nothing, which is the reading the Verifier suite's own write_effects has always taken.

    Where the call also carries D215's write effect record, the record decides: the effect has to
    show a column of the world moved. A record naming no moved column is a write that returned
    cleanly and left the world where it was. A call carrying no record at all is answered on the
    error marker alone, so a Runner on a route that records no effect gets the reading its own
    records support rather than a goal refused for want of evidence.
    """
    if _attr(message, "error") is not None:
        return False
    effect = _attr(message, "write_effect")
    if effect is None:
        return True
    return _effect_moved_a_column(effect)


def _effect_moved_a_column(effect: Any) -> bool:
    """Whether one D215 write effect record shows a column of the world moved.

    D215 writes two records per write, and both are read here because either may be the one riding
    with the call. The evidence the Builder mined off the recording is a list of rows, one per
    column, each naming the table, the row, the column path and what that column held before and
    after the write (`builder/effects.replay_evidence`); such a column moved when its after value
    differs from its before value. The replay's own record of running those checks counts the
    columns it checked and names the ones that never reached their after value
    (`runner/replay.ScoredRouter._check_effects`); a write moved a column there when it was checked
    on at least one and failed none of them, since a failure is a column the write left where it was.

    Neither record is read for its column names, only for whether any column moved, so a record
    written in either shape answers the same question and a shape neither knows falls back to
    whether the record holds anything at all.
    """
    if isinstance(effect, dict) and _is_replay_check(effect):
        checked = _count(effect.get("effect_checks"))
        failed = _count(effect.get("effect_failures_total")) or len(effect.get("effect_failures") or ())
        return checked > 0 and failed == 0
    rows = [effect] if isinstance(effect, dict) else list(effect or ())
    return any(_column_moved(row) for row in rows)


def _is_replay_check(effect: dict) -> bool:
    """Whether this record is the replay's own count of the effect checks it ran, not one column."""
    return any(key in effect for key in ("effect_checks", "effect_failures", "effect_failures_total"))


def _column_moved(row: Any) -> bool:
    """Whether one mined effect row shows its column holding something else after the write."""
    if not isinstance(row, dict):
        return bool(row)
    if "before" not in row and "after" not in row:
        return bool(row)
    return row.get("before") != row.get("after")


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def writes_made(transcript: Any, write_tools: Iterable[str]) -> set[str]:
    """The write-kind tools this Run has made, counting only the calls whose result took effect (D227).

    One function, read by both Simulated users, so the rule-driven user's `_goal_done` and the agent
    user's `end_run` cannot disagree about whether the goal is done. Before D227 this counted the
    write-kind tools the transcript showed called, so a write the world refused still ended the
    conversation `goal_satisfied` while the same Run was graded as claiming a write it never made.

    A tool result message names the call it answered, so it is the message an effect is read from.
    An assistant message's `tool_calls` are requests and not results, and a request whose result
    never came back is not a write that happened, so a name seen only there no longer counts.
    """
    names = frozenset(write_tools or ())
    made: set[str] = set()
    if not names:
        return made
    for message in transcript or ():
        if _field_of(message, "name") in names and write_took_effect(message):
            made.add(_field_of(message, "name"))
    return made


def _flatten(row: Any) -> dict[str, list]:
    """Every leaf value seen under each key, nested dicts included, so 'zip' inside 'address' is
    found. A key seen more than once (two payment methods that both carry 'source') keeps every
    value it was seen with, instead of the first one a dict happened to iterate to, so a caller can
    tell a genuine single answer from an ambiguous one (row 11).
    """
    flat: dict[str, list] = {}
    if not isinstance(row, dict):
        return flat
    for key, value in row.items():
        if isinstance(value, dict):
            for inner_key, inner_values in _flatten(value).items():
                flat.setdefault(inner_key, []).extend(inner_values)
        elif not isinstance(value, list):
            flat.setdefault(key, []).append(value)
    return flat


def _one(values: Optional[list]) -> Any:
    """The value at one key, only when every occurrence the row carries for it agrees; otherwise the
    field is unavailable rather than answered from whichever occurrence came first."""
    if not values:
        return None
    return values[0] if len({str(v) for v in values}) == 1 else None


def _row_value(row: dict, field: str) -> Any:
    """One field of this user's row, in the customer's own column names."""
    flat = _flatten(row)
    if field == "name":
        first, last = _one(flat.get("first_name")), _one(flat.get("last_name"))
        return f"{first} {last}" if first and last else _one(flat.get("name"))
    if field == "address":
        address = row.get("address") if isinstance(row.get("address"), dict) else None
        if address:
            parts = [address.get(key) for key in ("address1", "address2", "city", "state", "zip")]
            return ", ".join(str(part) for part in parts if part)
        return _one(flat.get("address"))
    for key in {"card_last4": ("card_last4", "last_four", "last4"),
                "payment_method": ("payment_method", "source"),
                "phone": ("phone", "phone_number")}.get(field, (field,)):
        value = _one(flat.get(key))
        if value is not None:
            return value
    return None
