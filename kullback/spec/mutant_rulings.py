"""Surviving mutants become the Examiner's rulings: which field or row is unchecked, never a value (D362).

After verify finds the Reference passing, its mutants (spec/mutants.py) are scored and each survivor is
read by code:

1. value, on a field a check of the Spec reads where that check rests on a fact the user volunteered (not
   an echo, not after pushback, not unfounded, not a side fact): derivation/unchecked_field on that check;
2. undone, on a row where every undone mutant built survived (no gate notices the Reference's write at
   all): derivation/unchecked_write, on a check that reads the row, else on the task;
3. extra_row or untouched: derivation/unchecked_row per table, on the task.

Any other survivor (a value on a field no volunteered fact names, an undone field on a row whose write a
gate does hold) is a note. A ruling names the table and field, never the row's value or the value to
require; it is filed in the Examiner's round with the same shape as the Examiner's own, so the writer sees
one channel and the round cap counts it. One ruling per unchecked cell, row or table per Task, ever: a
survivor already ruled on points at that ruling. The mutant report keeps a `ruling` per survivor, the
ruling number or "note".
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.runner.records import Run, Verifier, write_json
from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.mutants import CAP, SURVIVED, mutant_context, mutant_report, mutants_path
from kullback.spec.schema import Spec, load_spec

UNCHECKED_FIELD, UNCHECKED_WRITE, UNCHECKED_ROW = "unchecked_field", "unchecked_write", "unchecked_row"
NOTE = "note"


def usable_facts(spec: Spec) -> set[str]:
    """The fact ids a gate may rest on alone: volunteered, not an echo, not after pushback, not unfounded, not a
    side fact (D361: a side fact is scored, never gated)."""
    return {fact.id for fact in spec.intent.facts
            if fact.stance == "volunteered" and not (fact.echo or fact.after_pushback or fact.unfounded or fact.side)}


def reading_checks(spec: Spec, table: str, row_id: str, field: Optional[str] = None) -> list:
    """The Spec's state checks on the table and row (and field, given one), in Spec order."""
    out = []
    for check in spec.checks:
        demand = check.demand or {}
        expect = demand.get("expect") if isinstance(demand.get("expect"), dict) else None
        if demand.get("table") != table or expect is None:
            continue
        if demand.get("row") is not None and str(demand["row"]) != row_id:
            continue
        if field is None or field in expect:
            out.append(check)
    return out


def _cell(row: dict) -> dict:
    return {"run": "mutant", "operator": row["operator"], "table": row["table"], "row": row["row"],
            "field": row["field"]}


def _field_ruling(spec: Spec, row: dict) -> Optional[dict]:
    facts = usable_facts(spec)
    for check in reading_checks(spec, row["table"], str(row["row"]), row["field"]):
        named = [f for f in check.fact_ids if f in facts]
        if named:
            where = f"{row['table']}.{row['field']}"
            return {"code": UNCHECKED_FIELD, "item": check.id, "key": ("field", row["table"], row["row"], row["field"]),
                    "reason": f"A Run that writes any other value to {where}, which check {check.id} reads and fact "
                              f"{named[0]} names, still passes every gate.",
                    "fix": f"Make a gate hold {where} as fact {named[0]} says, or rebut if the user never asked for it."}
    return None


def _write_ruling(spec: Spec, table: str, row_id: str) -> dict:
    checks = reading_checks(spec, table, row_id)
    item = checks[0].id if checks else R.TASK_ITEM
    return {"code": UNCHECKED_WRITE, "item": item, "key": ("write", table, row_id),
            "reason": f"A Run that leaves out the Reference's write to a {table} row, every change on it put back, "
                      "still passes every gate.",
            "fix": f"Gate the {table} write the user asked for, or rebut if the user never asked for it."}


def _row_ruling(table: str) -> dict:
    return {"code": UNCHECKED_ROW, "item": R.TASK_ITEM, "key": ("row", table),
            "reason": f"A Run that also writes a {table} row or field outside the ones the Verifier declares still "
                      "passes every gate.",
            "fix": f"Hold the {table} rows outside the declared ones unchanged and count new rows, or rebut if the "
                   "policy allows the change."}


def candidates(spec: Spec, rows: Iterable[dict]) -> list[tuple[dict, list[dict]]]:
    """(ruling candidate, the survivor rows it covers) per unchecked cell, write or table, in report order.
    A survivor in no candidate is a note."""
    rows = list(rows)
    survived = [row for row in rows if row["outcome"] == SURVIVED]
    out: dict[tuple, tuple[dict, list[dict]]] = {}

    def add(found: Optional[dict], row: dict) -> None:
        if found is not None:
            out.setdefault(found["key"], (found, []))[1].append(row)

    for row in survived:
        if row["operator"] == "value" and row["field"] is not None:
            add(_field_ruling(spec, row), row)
        elif row["operator"] == "undone":
            same = [r for r in rows if r["operator"] == "undone" and (r["table"], r["row"]) == (row["table"], row["row"])]
            if all(r["outcome"] == SURVIVED for r in same):
                add(_write_ruling(spec, row["table"], str(row["row"])), row)
        elif row["operator"] in ("untouched", "extra_row"):
            add(_row_ruling(row["table"]), row)
    return list(out.values())


def _key_of(code: str, row: dict) -> Optional[tuple]:
    """The key a mutant ruling's verified row stands for, as `candidates` keys it."""
    if row.get("run") != "mutant":
        return None
    if code == UNCHECKED_FIELD:
        return ("field", row.get("table"), row.get("row"), row.get("field"))
    if code == UNCHECKED_WRITE:
        return ("write", row.get("table"), str(row.get("row")))
    return ("row", row.get("table"))


def _ruled_before(rulings: list[R.Ruling], found: dict) -> Optional[int]:
    """The number of an earlier mutant ruling on the same cell, write or table, else None."""
    for ruling in rulings:
        if ruling.code == found["code"] and any(_key_of(ruling.code, row) == found["key"] for row in ruling.verified):
            return ruling.number
    return None


def file_rulings(workdir: Any, spec: Spec, rows: list[dict], round_number: int = 0, bus: Any = None) -> list[int]:
    """Each candidate filed as a blocking derivation ruling in the round (0 reads the Spec's round plus one, as
    the Examiner's own rulings do), unless one stands on the same key;
    every survivor row gets `ruling`, the number it became or points at, else "note". Returns the numbers filed."""
    task_id = spec.task_id
    held = R.load_rulings(workdir, task_id)
    filed = []
    for row in rows:
        if row["outcome"] == SURVIVED:
            row["ruling"] = NOTE
    for found, covered in candidates(spec, rows):
        number = _ruled_before(held, found)
        if number is None:
            ruling = R.Ruling(task_id=task_id, number=R.next_number(workdir, task_id), round=int(round_number) or spec.round + 1,
                              item=found["item"], kind="derivation", code=found["code"], reason=found["reason"],
                              fix=found["fix"], blocking=True, verified=[_cell(row) for row in covered])
            R.save_ruling(workdir, ruling)
            held.append(ruling)
            number = ruling.number
            filed.append(number)
            if bus is not None:
                events.ruling_filed(bus, events.ruling_payload(ruling))
        for row in covered:
            row["ruling"] = number
    if filed:
        R.sync_spec(workdir, task_id)
    return filed


def mutant_rulings(workdir: Any, verifier: Verifier, reference: Optional[Run], round_number: int = 0, *,
                   judge_results: Optional[dict] = None, bus: Any = None, cap: int = CAP) -> dict:
    """The Task's mutants scored on its Reference, the survivors filed as rulings, the report written to
    exam/mutants/<task>.json. Returns the report's status and counts and the ruling numbers filed."""
    canon, tools, schema = mutant_context(workdir)
    report = mutant_report(verifier, reference, canon=canon, tools=tools, schema=schema,
                           judge_results=judge_results, cap=cap)
    spec = load_spec(workdir, verifier.task_id)
    filed = file_rulings(workdir, spec, report["mutants"], round_number, bus) \
        if report["status"] == "scored" and spec is not None else []
    write_json(mutants_path(workdir, verifier.task_id), report)
    notes = sum(row.get("ruling") == NOTE for row in report.get("mutants", []))
    return {"status": report["status"], "built": report.get("built", 0), "survived": report.get("survived", 0),
            "filed": filed, "notes": notes}


def mutants_line(out: dict) -> str:
    """The verify answer's mutant part, in one line."""
    if out["status"] != "scored":
        return f"mutants: not built ({out['status']})"
    filed = ", ".join(str(n) for n in out["filed"]) or "none"
    return (f"mutants: {out['built']} built, {out['survived']} survived; rulings filed from them: {filed}; "
            f"notes: {out['notes']}")


__all__ = ["NOTE", "UNCHECKED_FIELD", "UNCHECKED_ROW", "UNCHECKED_WRITE", "candidates", "file_rulings",
           "mutant_rulings", "mutants_line", "reading_checks", "usable_facts"]
