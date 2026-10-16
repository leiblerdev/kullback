"""Surviving mutants become the Examiner's rulings: which field or row is unchecked, never a value (D362).

At verify, the mutants of the Task's newest passing fresh Run, keyed to the ask (spec/mutants.py, D368),
are scored and each survivor is read by code:

1. value, on a field a check of the Spec reads where that check rests on an ask the user raised (not
   accepted, not a reaction, not an echo, not unfounded, not a side fact): derivation/unchecked_field on that check;
2. undone, on a row where every undone mutant built survived (no gate notices the base's write at
   all): derivation/unchecked_write, on a check that reads the row, else on the task;
3. extra_row or untouched: derivation/unchecked_row per table, on the task.

Before the mutants, every ask no gate item cites (schema.unchecked_asks, D370) is filed as
derivation/unchecked_ask on the task, one per fact, ever; it needs no passing Run.

Any other survivor (a value on a field no volunteered fact names, an undone field on a row whose write a
gate does hold) is a note. A ruling names the table and field, never the row's value or the value to
require; it is filed in the Examiner's round with the same shape as the Examiner's own, so the writer sees
one channel and the round cap counts it. One ruling per unchecked cell, row or table per Task, ever: a
survivor already ruled on points at that ruling. The mutant report keeps a `ruling` per survivor, the
ruling number or "note".
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.runner.records import Verifier, write_json
from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.mutants import CAP, SURVIVED, mutants_path, task_mutants
from kullback.spec.schema import Spec, is_reaction, load_spec, unchecked_asks

UNCHECKED_FIELD, UNCHECKED_WRITE, UNCHECKED_ROW = "unchecked_field", "unchecked_write", "unchecked_row"
NOTE = "note"


def usable_facts(spec: Spec) -> set[str]:
    """The fact ids a gate may rest on alone: raised by the user (not accepted), not a reaction (D366), not an
    echo, not unfounded, not a side fact (D361: a side fact is scored, never gated)."""
    return {fact.id for fact in spec.intent.facts if fact.stance != "accepted"
            and not (is_reaction(fact) or fact.echo or fact.unfounded or fact.side)}


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
            "reason": f"A Run that leaves out the passing Run's write to a {table} row, every change on it put back, "
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


UNCHECKED_ASK = "unchecked_ask"


def ask_rulings(workdir: Any, spec: Spec, round_number: int = 0, bus: Any = None) -> list[int]:
    """Each ask no gate item cites filed as a blocking derivation/unchecked_ask ruling on the task (D370),
    unless one already names that fact. Returns the numbers filed."""
    task_id = spec.task_id
    held = {row.get("fact_id") for ruling in R.load_rulings(workdir, task_id) if ruling.code == UNCHECKED_ASK
            for row in ruling.verified}
    filed = []
    for fact_id in unchecked_asks(spec):
        if fact_id in held:
            continue
        ruling = R.Ruling(task_id=task_id, number=R.next_number(workdir, task_id),
                          round=int(round_number) or spec.round + 1, item=R.TASK_ITEM, kind="derivation",
                          code=UNCHECKED_ASK, blocking=True, verified=[{"fact_id": fact_id}],
                          reason=f"fact {fact_id} is an ask of the user that no gate item cites",
                          fix=f"add a gate item citing {fact_id}, or file {fact_id} as a gap")
        R.save_ruling(workdir, ruling)
        filed.append(ruling.number)
        if bus is not None:
            events.ruling_filed(bus, events.ruling_payload(ruling))
    if filed:
        R.sync_spec(workdir, task_id)
    return filed


def mutant_rulings(workdir: Any, verifier: Verifier, round_number: int = 0, *, runs: Optional[list] = None,
                   bus: Any = None, cap: int = CAP) -> dict:
    """The Task's mutants scored on its newest passing fresh Run (of `runs`, else its Runs on disk), the
    survivors filed as rulings, the report written to exam/mutants/<task>.json. Returns the report's status
    and counts and the ruling numbers filed."""
    spec = load_spec(workdir, verifier.task_id)
    asks = ask_rulings(workdir, spec, round_number, bus) if spec is not None else []
    report = task_mutants(workdir, verifier, runs, cap=cap)
    filed = file_rulings(workdir, spec, report["mutants"], round_number, bus) \
        if report["status"] == "scored" and spec is not None else []
    write_json(mutants_path(workdir, verifier.task_id), report)
    notes = sum(row.get("ruling") == NOTE for row in report.get("mutants", []))
    return {"status": report["status"], "built": report.get("built", 0), "survived": report.get("survived", 0),
            "filed": filed, "notes": notes, "asks": asks}


def mutants_line(out: dict) -> str:
    """The verify answer's mutant part, in one line, after the unchecked asks filed when there are any."""
    asks = out.get("asks") or []
    head = f"unchecked asks filed: {', '.join(str(n) for n in asks)}; " if asks else ""
    if out["status"] != "scored":
        return f"{head}mutants: not built ({out['status']})"
    filed = ", ".join(str(n) for n in out["filed"]) or "none"
    return (f"{head}mutants: {out['built']} built, {out['survived']} survived; rulings filed from them: {filed}; "
            f"notes: {out['notes']}")


__all__ = ["NOTE", "UNCHECKED_ASK", "UNCHECKED_FIELD", "UNCHECKED_ROW", "UNCHECKED_WRITE", "candidates", "file_rulings",
           "ask_rulings", "mutant_rulings", "mutants_line", "reading_checks", "usable_facts"]
