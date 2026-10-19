"""Every write a born Task's scenario end names is gated by a differing cell, on every named column (D419).

The end's writes are the end facts (writer.with_end, D414) an item ties to a change: a row_is or row_new
citing the fact, or a called or before item on a write tool citing it. Two checks read the Spec and the
end, never the Reference (D369), and change nothing of what is reachable (D387, D414):

1. `ungated_writes`: a write fact with no gated row_is whose expected value differs from the starting cell
   and no gated row_new. A cell gated to its start value, a one_of holding it, a before or a called item
   never counts as the gate of a write: the Run that does nothing passes each.
2. `ungated_values`: a number, id or quoted value of a write fact's clause that is a value of the changed
   row (its own cells, nested ones included, such as a leg or a line item; for a new row, what its table holds
   under the keys its where names) and that no gated state item carries in its find, expect or where.

`send_back` sends both findings to the writer once, through the repair session the Examiner's rulings use
(writer.repair); what is still found after it is counted on the writer's counts (WRITES_UNGATED,
END_VALUES_UNGATED) and the Spec stands. A send back never gates a value to pass: the writer may rebut.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.spec import items as I
from kullback.spec.schema import Spec
from kullback.spec.writer import UNCHECKED_END, Ruling, Written, item_checks, repair, unchecked_end, world_of

# The writer's counts that mark a Spec, each the number still found after the one send back.
WRITES_UNGATED = "writes_ungated"
END_VALUES_UNGATED = "end_values_ungated"
# The findings behind the two counts, by fact id, and that the send back ran.
UNGATED_FINDINGS = "ungated_findings"
UNGATED_SENT_BACK = "ungated_sent_back"
# The ruling kind of the send back (writer.Ruling.kind).
RULING_KIND = "ungated_write"

WRITE_STATE = ("row_is", "row_new")

QUOTED_RE = re.compile(r"\"([^\"]+)\"|“([^”]+)”")
DIGIT_TOKEN_RE = re.compile(r"[^\s,;()\"“”]*\d[^\s,;()\"“”]*")

RULING_FIX = ("For each write named, add or fix a gated row_is on the changed row whose expected value differs "
              "from the start (or a gated row_new), citing the end fact; a before or called item is no gate of a "
              "write. For each value named, carry it in a gated state item's find, expect or where. Every value "
              "still comes from the world, the policy or the user's words. Where none can be named, change "
              "nothing and reply with one line starting \"rebut: cannot verify:\" and why.")


def _items(spec: Spec) -> list[dict]:
    return item_checks(spec)


def _cites(item: dict, fact_id: str) -> bool:
    return fact_id in (item.get("fact_ids") or [])


def _writes_tool(item: dict, write_tools: Optional[set]) -> bool:
    tool = item.get("tool") if item.get("kind") == "called" else item.get("then") if item.get("kind") == "before" \
        else None
    return bool(tool) and (write_tools is None or tool in write_tools)


def write_facts(spec: Spec, write_tools: Optional[Iterable[str]] = None) -> list[str]:
    """The end facts that name a write: cited by a row_is or row_new, or by a called or before item on a write
    tool (every tool when `write_tools` is None). A recorded Task has no end fact, so none."""
    tools = None if write_tools is None else set(write_tools)
    items = _items(spec)
    return [fact.id for fact in spec.intent.facts if I.is_end_fact(fact) and any(
        _cites(item, fact.id) and (item["kind"] in WRITE_STATE or _writes_tool(item, tools)) for item in items)]


def _differs(expected: Any, start: Any) -> bool:
    """Whether a cell expecting `expected` fails a row left at `start`."""
    if isinstance(expected, dict) and "one_of" in expected:
        return I.norm(start) not in {I.norm(member) for member in expected.get("one_of") or []}
    if isinstance(expected, dict) and "not" in expected:
        return I.norm(start) in {I.norm(member) for member in expected.get("not") or []}
    if isinstance(expected, list):
        held = start if isinstance(start, list) else [start]
        return sorted(I.norm(value) for value in expected) != sorted(I.norm(value) for value in held)
    return I.norm(expected) != I.norm(start)


def differing_gate(item: dict, world: I.World) -> bool:
    """A gated row_new, or a gated row_is with an expected cell that differs from the row it finds at the start."""
    if item.get("gate") is not True:
        return False
    if item["kind"] == "row_new":
        return int(item.get("count") or 1) >= 1
    if item["kind"] != "row_is":
        return False
    rows = [world.row(item.get("table", ""), key) for key in world.find(item.get("table", ""), item.get("find"))]
    return any(row is not None and _differs(expected, I.path_value(row, field))
               for row in rows for field, expected in (item.get("expect") or {}).items())


def ungated_writes(spec: Spec, world: I.World, write_tools: Optional[Iterable[str]] = None) -> list[dict]:
    """The write facts no differing gate cites (rule 1), each with why."""
    items = _items(spec)
    return [{"fact_id": fact_id, "reason": "no gated state item citing it expects a value other than the start"}
            for fact_id in write_facts(spec, write_tools)
            if not any(_cites(item, fact_id) and differing_gate(item, world) for item in items)]


def end_values(text: str) -> list[str]:
    """The quoted values and the tokens holding a digit that a clause names, in order, once each."""
    quoted = [first or second for first, second in QUOTED_RE.findall(text or "")]
    tokens = [token.strip(".:!?'").lstrip("$#") for token in DIGIT_TOKEN_RE.findall(QUOTED_RE.sub(" ", text or ""))]
    return list(dict.fromkeys(value.strip() for value in quoted + tokens if value.strip()))


def _keyed_leaves(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for name, inner in value.items():
            yield from _keyed_leaves(inner, str(name))
    elif isinstance(value, list):
        for inner in value:
            yield from _keyed_leaves(inner, key)
    elif value is not None:
        yield key, value


def _row_values(item: dict, world: I.World) -> set[str]:
    """The values of the changed row's columns. For a row_is, the row's own key and cells, nested leaves included
    (a leg or a line item it holds): a value another row happens to hold is not the changed row's. For a row_new,
    which does not exist yet, its where values and what the table's rows hold under the keys its where names."""
    table = item.get("table", "")
    rows = world.state.get(table) or {}
    if item["kind"] != "row_new":
        found = world.find(table, item.get("find"))
        return {I.norm(key) for key in found} | {I.norm(v) for key in found for _, v in _keyed_leaves(rows.get(key))}
    where = item.get("where") or {}
    held = {I.norm(v) for row in rows.values() for name, v in _keyed_leaves(row) if name in where}
    return {I.norm(v) for _, v in _keyed_leaves(where)} | held


def _carried(items: list[dict]) -> set[str]:
    """Every value a gated state item names in its find, expect or where (a forbidden value is not carried)."""
    out: set[str] = set()
    for item in items:
        if item.get("gate") is not True or item.get("kind") not in I.STATE_KINDS:
            continue
        expect = {field: value for field, value in (item.get("expect") or {}).items()
                  if not (isinstance(value, dict) and "not" in value)}
        for part in (item.get("find"), expect, item.get("where")):
            out |= {I.norm(value) for _, value in _keyed_leaves(part or {})}
    return out


def ungated_values(spec: Spec, world: I.World, write_tools: Optional[Iterable[str]] = None) -> list[dict]:
    """The values a write fact's clause names of the changed row that no gate carries (rule 2)."""
    items, facts = _items(spec), {fact.id: fact for fact in spec.intent.facts}
    carried, out = _carried(items), []
    for fact_id in write_facts(spec, write_tools):
        columns: set[str] = set()
        for item in items:
            if _cites(item, fact_id) and item["kind"] in I.STATE_KINDS:
                columns |= _row_values(item, world)
        out += [{"fact_id": fact_id, "value": value} for value in end_values(facts[fact_id].text)
                if I.norm(value) in columns and I.norm(value) not in carried]
    return out


def findings(spec: Spec, world: I.World, write_tools: Optional[Iterable[str]] = None) -> tuple[list, list]:
    return ungated_writes(spec, world, write_tools), ungated_values(spec, world, write_tools)


def ruling_of(spec: Spec, writes: list[dict], values: list[dict]) -> Ruling:
    """One ruling naming every finding: the facts, the items citing them (which the repair may fix), why, the fix."""
    facts = list(dict.fromkeys(row["fact_id"] for row in writes + values))
    lines = [f"end fact {row['fact_id']} names a write: {row['reason']}" for row in writes]
    lines += [f"end fact {row['fact_id']} names {row['value']!r} of the changed row, which no gated item carries"
              for row in values]
    return Ruling(check_ids=[item["id"] for item in _items(spec) if any(_cites(item, f) for f in facts)],
                  reason="; ".join(lines), item=facts[0] if facts else "", fix=RULING_FIX, kind=RULING_KIND,
                  facts=facts)


def send_back(written: Written, inputs: Any, model: Any, *, model_id: Optional[str] = None,
              ceiling_usd: Optional[float] = None) -> tuple[Written, float]:
    """(the Written after one send back of the findings, its spend); unchanged and 0 with none found.

    The repaired Spec keeps the first write's counts with the marks added: WRITES_UNGATED and END_VALUES_UNGATED
    when any is still found, the findings by fact, and the unchecked end and clashes as the repair left them.
    """
    world = world_of(inputs)
    writes, values = findings(written.spec, world, inputs.write_tools)
    if not writes and not values:
        return written, 0.0
    fixed = repair(written.spec, ruling_of(written.spec, writes, values), inputs, model, model_id=model_id,
                   ceiling_usd=ceiling_usd)
    spec = fixed.spec.model_copy(update={"version": written.spec.version})
    counts = dict(written.counts, **{UNGATED_SENT_BACK: len(writes) + len(values)})
    writes, values = findings(spec, world, inputs.write_tools)
    for key, rows in ((WRITES_UNGATED, writes), (END_VALUES_UNGATED, values)):
        counts.pop(key, None)
        if rows:
            counts[key] = len(rows)
    if writes or values:
        counts[UNGATED_FINDINGS] = writes + values
    counts.pop(UNCHECKED_END, None)
    if left := unchecked_end(spec):
        counts[UNCHECKED_END] = left
    counts["unsatisfiable"] = sorted({item_id for item_id, _ in I.contradictions(item_checks(spec))})
    return Written(spec, counts, fixed.session), float(fixed.session.get("usd") or 0.0)


__all__ = ["END_VALUES_UNGATED", "RULING_KIND", "UNGATED_FINDINGS", "UNGATED_SENT_BACK", "WRITES_UNGATED",
           "differing_gate", "end_values", "findings", "ruling_of", "send_back", "ungated_values", "ungated_writes",
           "write_facts"]
