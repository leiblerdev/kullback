"""Each item of a Verifier scored on one Run by code: True, False, or None when code cannot say.

The items are the atoms whose target kind spec/items.py writes (state, event, text, the sanity item).
A judge item is never scored here (None): the judge answers it. The reward reads these results:
0 when a gate item fails, else the weighted mean of the scored items (COMMON contract, sh-reward).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from kullback.runner.atom_context import AtomContext
from kullback.runner.canon import DIFFERENT, UNRESOLVED
from kullback.runner.expected import cell_resolution, match_context, path_value
from kullback.runner.records import Atom, Verifier

STATE_KINDS = ("row_is", "row_new", "row_keeps")
ITEM_KINDS = STATE_KINDS + ("called", "not_called", "before", "said", "nothing_else")
CONFIRM_TURN = "confirm_turn"
# A sentence that denies the value rather than states it: "no", "not", "never", "cannot" before it.
_NEGATION = re.compile(r"\b(?:no|not|never|cannot|can't|won't|isn't|aren't|don't|doesn't)\b", re.I)


def is_item(atom: Atom) -> bool:
    return (atom.target or {}).get("kind") in ITEM_KINDS


def _row_holds(target: dict, context: AtomContext) -> Optional[bool]:
    row = (context.end_state.get(target["table"]) or {}).get(target["row"])
    if row is None:
        return False
    open_ = False
    for field, value in (target.get("expect") or {}).items():
        found = cell_resolution(context, f"{target['table']}.{field}", value, path_value(row, field))
        if found == DIFFERENT:
            return False
        open_ = open_ or found == UNRESOLVED
    return None if open_ else True


def _new_holds(target: dict, context: AtomContext) -> bool:
    made = 0
    for key, moved in context.diff().items():
        table, _, row_id = key.partition(".")
        if table != target["table"] or moved["present_before"] or not moved["present_after"]:
            continue
        row = (context.end_state.get(table) or {}).get(row_id) or {}
        made += all(context.c(path_value(row, f)) == context.c(v) for f, v in (target.get("where") or {}).items())
    return made == int(target.get("count") or 1)


def _calls(context: AtomContext, tool: str, args: dict) -> list[dict]:
    return [call for call in context.calls if call["name"] == tool and not call["error"]
            and all(context.c((call["args"] or {}).get(k)) == context.c(v) for k, v in (args or {}).items())]


def _before_holds(target: dict, context: AtomContext) -> bool:
    if target["first"] == CONFIRM_TURN:
        return context.confirmed_before_first_write(target["then"])
    then = _calls(context, target["then"], {})
    if not then:
        return True
    first = _calls(context, target["first"], {})
    return bool(first) and first[0]["idx"] < then[0]["idx"]


def _stated(value: Any, texts: list[str], context: AtomContext) -> bool:
    """The value in some sentence of the texts, not in a sentence that denies it."""
    needle = context.t(value)
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            flat = context.t(sentence)
            if needle and needle in flat and not _NEGATION.search(flat.split(needle)[0]):
                return True
    return False


def _said_holds(target: dict, context: AtomContext) -> bool:
    texts = [text for _, text in context.assistant]
    return (all(_stated(v, texts, context) for v in target.get("values") or [])
            and not any(_stated(v, texts, context) for v in target.get("not_values") or []))


def _sanity_holds(verifier: Verifier, context: AtomContext) -> Optional[bool]:
    """Nothing outside the declared rows moved, in any one end state: its collateral and new-row checks."""
    if not verifier.expected:
        return None
    outcomes = []
    for state in verifier.expected:
        why = [w for w in match_context(state, context).why if w.startswith(("collateral:", "new:", "unsettled:"))]
        if not why:
            return True
        outcomes.append(None if all(w.startswith("unsettled:") for w in why) else False)
    return None if None in outcomes else False


def item_holds(atom: Atom, verifier: Verifier, context: AtomContext) -> Optional[bool]:
    """One item on one Run; None for a judge item, an unknown kind or a pair code cannot settle."""
    target = atom.target or {}
    kind = target.get("kind")
    if atom.judge or kind not in ITEM_KINDS:
        return None
    if kind in ("row_is", "row_keeps"):
        return _row_holds(target, context)
    if kind == "row_new":
        return _new_holds(target, context)
    if kind == "called":
        return bool(_calls(context, target["tool"], target.get("args") or {}))
    if kind == "not_called":
        return not _calls(context, target["tool"], target.get("args") or {})
    if kind == "before":
        return _before_holds(target, context)
    if kind == "said":
        return _said_holds(target, context)
    return _sanity_holds(verifier, context)


def score_items(verifier: Verifier, context: AtomContext) -> dict[str, Optional[bool]]:
    """Every item atom of the Verifier on this Run, by id."""
    return {atom.id: item_holds(atom, verifier, context) for atom in verifier.atoms if is_item(atom) or atom.judge}


__all__ = ["ITEM_KINDS", "is_item", "item_holds", "score_items"]
