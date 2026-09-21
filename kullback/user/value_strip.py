from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Sequence

from kullback.runner.canon import DATE_ONLY, CanonRules, class_of
from kullback.runner.records import ColumnClass, StrippedValue, Trace


def _flat(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


# --- the strip: no value only the system knew reaches an Intent (D196) ---------------------

MIN_VALUE_CHARS = 2  # a one-character value identifies nothing and matches everywhere
CODE_TAIL = 4  # the last characters of a code, the part a person reads a reference back by
RESERVED_VALUES = frozenset(("true", "false", "null", "none"))
MONTHS = ("january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december")
# The words a stripped value leaves dangling at the end of a line ("cancel the order for").
TRAILING_WORDS = frozenset("a an the of on in for to with at from by and or".split())
STRIP_PUNCTUATION = " \t,;:.-"


def canon_rules_of(rules: Any) -> CanonRules:
    """The canon rules as the object `class_of` reads, whether the caller held one or its dict form.

    The artifact crosses the Builder's cache as plain JSON, so a stage that reads it back holds a
    dict where the stage that wrote it held the record; either one classifies a column the same way.
    """
    if isinstance(rules, CanonRules):
        return rules
    if isinstance(rules, dict):
        try:
            return CanonRules.model_validate(rules)
        except ValueError:
            return CanonRules()
    return CanonRules()


def _value_text(value: Any) -> Optional[str]:
    """One scalar as the text a line would spell it with, or nothing when it is not a value.

    A bool and a null name no row and identify nobody, so they are not values here; a number is,
    and it is spelled the way `json.dumps` spells it, which is how every other reader of a recorded
    result spells it too.
    """
    if value is None or isinstance(value, bool):
        return None
    text = value if isinstance(value, str) else _flat(value)
    text = text.strip()
    return text or None


def _walk_values(value: Any, table: Optional[str], column: Optional[str], source: str,
                 schema: Any, rules: CanonRules, out: dict[str, StrippedValue]) -> None:
    """Every leaf of a recorded argument, result or row, under the column key it sat beneath."""
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _walk_values(value[key], table, str(key), source, schema, rules, out)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _walk_values(item, table, column, source, schema, rules, out)
        return
    text = _value_text(value)
    if text is None or len(text) < MIN_VALUE_CHARS or text.lower() in RESERVED_VALUES:
        return
    out.setdefault(text, StrippedValue(
        column=column or (table or ""), table=table, source=source,
        **{"class": class_of(schema, None, column, rules) if column else rules.default_class},
    ))


def system_values(traces: Sequence[Trace], *, start_state: Any = None, schema: Any = None,
                  rules: Any = None) -> dict[str, StrippedValue]:
    """Every value the system held in these Runs, by the column it sat under and that column's class.

    The system side of a recording is what the tools were asked and what they answered: an argument
    the recorded agent passed and a field of a result it read. `start_state` is the Task's own rows
    where a caller has them; the recording's reads are the usual way the Starting state reaches an
    Intent, because a value no recorded call ever read is a value the miner never saw either.

    The class comes from `canon.class_of` under the schema's own column names, so the strip and the
    compare give one column one class (D73). The table is not looked up: a leaf key is the column
    name wherever the result nested it, and asking the schema for `<tool>.<key>` would miss every
    column the corpus stores under the table's own name.
    """
    out: dict[str, StrippedValue] = {}
    rules = canon_rules_of(rules)
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None:
                continue
            _walk_values(call.args, call.name, None, "tool_arg", schema, rules, out)
            _walk_values(call.result, call.name, None, "tool_result", schema, rules, out)
    if start_state is not None:
        _walk_values(start_state, None, None, "start_state", schema, rules, out)
    return out


def spoken_text(traces: Sequence[Trace]) -> str:
    """Everything the user side of every Run said, read as one blob.

    The leak check reads its own `said_by_user` this way (D157), so a value this says no user said
    is a value that check will call a secret: the strip is at least as strict as the audit over it.
    """
    return " ".join(turn.content or "" for trace in traces
                    for turn in trace.turns if turn.role == "user")


def _token_at(haystack: str, needle: str, start: int = 0) -> int:
    """Where this value sits in this text as a whole token, or -1; the leak check's own boundary."""
    hay, need = haystack.lower(), needle.lower()
    while True:
        at = hay.find(need, start)
        if at < 0:
            return -1
        before = hay[at - 1] if at else " "
        after = hay[at + len(need)] if at + len(need) < len(hay) else " "
        if not before.isalnum() and not after.isalnum():
            return at
        start = at + 1


def _said(spoken: str, value: str) -> bool:
    return _token_at(spoken, value) >= 0


def _month_of(value: str) -> Optional[str]:
    """The month a date names, or nothing; only a date, never a number that looks like one."""
    match = DATE_ONLY.match(value)
    if match is None or (len(value) > 10 and value[10] not in "T "):
        return None
    index = int(value[5:7])
    return MONTHS[index - 1] if 1 <= index <= 12 else None


def _is_code(value: str) -> bool:
    """Is this value a reference a person reads back by its last characters, rather than an amount?

    A code carries a digit and something else: a letter, or enough digits that no amount is spelled
    that way. A price and a count are not codes, and their last four characters say nothing.
    """
    body = value.strip("#").replace("-", "").replace("_", "")
    if len(body) <= CODE_TAIL or not any(c.isdigit() for c in body):
        return False
    if any(c.isalpha() for c in body):
        return True
    return body.isdigit()


def replacement_for(value: str, column_class: ColumnClass = "hard") -> tuple[str, str]:
    """The shape of a system-known value a user would still recognise, and the words it leaves.

    Three shapes and no more. A column the compare exempts decides nothing, so its value goes and
    leaves nothing behind. A date becomes its month, which is what a person says about a delivery
    they are waiting on. A code becomes its last characters, which is what a person reads back off a
    confirmation mail. Everything else is removed: an amount, a name and an address are either the
    user's own words, in which case they were never stripped, or they are the system's and the
    Candidate has to go and read them.
    """
    if column_class == "exempt":
        return "removed", ""
    month = _month_of(value)
    if month is not None:
        return "month", month
    if _is_code(value):
        return "last4", f"ending {value[-CODE_TAIL:]}"
    return "removed", ""


def _tidy(text: str) -> str:
    """One line again after words were taken out of it: no double space, no dangling preposition."""
    out = " ".join((text or "").split())
    out = re.sub(r"\s+([,;:.!?])", r"\1", out)
    out = re.sub(r"([,;:])(\s*[,;:])+", r"\1", out)
    out = out.strip(STRIP_PUNCTUATION)
    words = out.split()
    while words and words[-1].strip(STRIP_PUNCTUATION).lower() in TRAILING_WORDS:
        words.pop()
    return " ".join(words).strip(STRIP_PUNCTUATION)


def _protected_spans(text: str, protected: Sequence[str]) -> list[list[int]]:
    """Where words an earlier strip already put into this line sit, so this one leaves them alone."""
    spans: list[list[int]] = []
    for words in protected:
        if not words:
            continue
        at = _token_at(text, words)
        while at >= 0:
            spans.append([at, at + len(words)])
            at = _token_at(text, words, at + 1)
    return spans


def strip_intent(text: str, traces: Sequence[Trace], *, start_state: Any = None, schema: Any = None,
                 rules: Any = None,
                 protected: Sequence[str] = ()) -> tuple[str, list[StrippedValue]]:
    """The candidate line with every system-known value taken out of it, and what was taken (D196).

    A system-known value is one the recording's own tools held and no user of any Run of the Task
    said. Such a value in an Intent tells the Simulated user, and through it the Candidate, something
    no customer ever knew, which is what the D79 leak check fails a whole Task for; and it is a word
    the grounding could never evidence either (D157), so the same line was already being refused.
    Taking it out before the line is graded turns both losses into a line that still names the goal.

    The pass is code, longest value first so a code goes before the digits inside it, and it never
    reads a value it has itself written in: the words a shape leaves are frozen, and `protected`
    freezes the words an earlier strip of the same line left, so stripping twice is stripping once.
    """
    return value_strip(traces, start_state=start_state, schema=schema, rules=rules)(text, protected)


def value_strip(traces: Sequence[Trace], *, start_state: Any = None, schema: Any = None,
                rules: Any = None) -> Callable[..., tuple[str, list[StrippedValue]]]:
    """The same strip over one Task's evidence, read once and applied to many lines (D210).

    `strip_intent` grades one line per Task and can afford to walk the Task's recorded calls again
    each time. The Simulated user runs the strip over every answer of every turn of every Run, so
    what it holds is this closure: the values the system knew, sorted once, and a pass over the
    line that reads them. What is stripped and what is left is `strip_intent`'s, unchanged.
    """
    known = system_values(traces, start_state=start_state, schema=schema, rules=rules)
    spoken = spoken_text(traces)
    wanted = sorted((v for v in known if not _said(spoken, v)), key=lambda v: (-len(v), v))

    def strip(text: str, protected: Sequence[str] = ()) -> tuple[str, list[StrippedValue]]:
        return _strip_known(text, known, wanted, protected)

    return strip


def _strip_known(text: str, known: dict[str, StrippedValue], wanted: Sequence[str],
                 protected: Sequence[str]) -> tuple[str, list[StrippedValue]]:
    """One line with each wanted value taken out of it, longest first, and what was taken."""
    out, frozen = text or "", _protected_spans(text or "", protected)
    taken: list[StrippedValue] = []
    for value in wanted:
        shape, words = replacement_for(value, known[value].class_)
        found, start = False, 0
        while True:
            at = _token_at(out, value, start)
            if at < 0:
                break
            if any(begin <= at < end for begin, end in frozen):
                start = at + 1  # inside words a strip wrote; that is not the line's own value
                continue
            out = out[:at] + words + out[at + len(value):]
            shift = len(words) - len(value)
            for span in frozen:
                if span[0] >= at:
                    span[0] += shift
                    span[1] += shift
            frozen.append([at, at + len(words)])
            start, found = at + len(words), True
        if found:
            taken.append(known[value].model_copy(update={"shape": shape, "replacement": words}))
    return _tidy(out), taken
