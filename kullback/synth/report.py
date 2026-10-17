"""The synth report: every scenario count under its own heading, never in the build table (D372, D373).

Per source kind: born, passed, unsure, infeasible, struck and fixing; the Environment findings the
review filed; what the generator dropped, per reason; the coverage across sources; the human queue; and
what the scenario stages spent. Read off the store and the ledger; nothing is called.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from kullback.runner import budget
from kullback.runner.records import read_json
from kullback.synth import records as R

HEADING = "## synth"
STAGES = ("synth_generation", "synth_review", "synth_fix")
FINDING_KEY = "environment:scenario:"
SPREAD = (("situation_kind", "situation kind"), ("bound_table", "bound table"), ("policy_line", "policy line"))


def _table(records: list) -> dict:
    """Per source kind, how many records sit in each status."""
    table = {kind: Counter() for kind in R.SOURCE_KINDS}
    for record in records:
        table[record.source.kind][record.status] += 1
    return {kind: {status: row.get(status, 0) for status in R.STATUSES} for kind, row in table.items()}


def _findings(root: Path) -> int:
    """The Environment findings the review filed, by their key."""
    rows = read_json(root / "examiner" / "findings.json", []) or []
    return len([row for row in rows if isinstance(row, dict) and str(row.get("key") or "").startswith(FINDING_KEY)])


def _spend(root: Path) -> dict:
    stages = budget.load_totals(root)["stages"]
    return {stage: float((stages.get(stage) or {}).get("usd") or 0) for stage in STAGES}


def counts(workdir: Any) -> dict:
    """The numbers the report prints, as data."""
    root = Path(workdir)
    generated = read_json(R.store_dir(root) / R.GENERATED, {}) or {}
    return {"table": _table(R.load_scenarios(root)), "findings": _findings(root),
            "coverage": generated.get("coverage") or {}, "notes": generated.get("notes") or [],
            "human": len(read_json(R.store_dir(root) / R.HUMAN, []) or []), "spend": _spend(root)}


def _by_count(counter: dict) -> list[tuple[str, int]]:
    """A counter as printed: the largest first, ties by name, since the stored file sorts its keys."""
    return sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))


def _spread_lines(coverage: dict) -> list[str]:
    """One line per spread counter and the cells hit, where the coverage holds them."""
    spread = coverage.get("spread") or {}
    out = [f"scenarios per {name}: " + ", ".join(f"{label} {n}" for label, n in _by_count(spread.get(key) or {}))
           for key, name in SPREAD if key in spread]
    cells = coverage.get("cells")
    return out + ([f"cells hit: {cells.get('hit', 0)} of {cells.get('possible', 0)}"] if cells else [])


def lines(workdir: Any) -> list[str]:
    """The report as printed lines, under the synth heading."""
    body = counts(workdir)
    out = [HEADING, "", "| source | " + " | ".join(R.STATUSES) + " |", "| --- |" + " --- |" * len(R.STATUSES)]
    for kind, row in body["table"].items():
        out.append(f"| {kind} | " + " | ".join(str(row[status]) for status in R.STATUSES) + " |")
    coverage = body["coverage"]
    out += ["", f"findings filed: {body['findings']}", f"human queue: {body['human']}"]
    if coverage:
        out.append(f"search scenarios the recordings never show: {coverage.get('search_not_in_recordings', 0)}")
        out.append(f"ask lines nothing else covers: {coverage.get('asks_not_covered', 0)}")
        out.append(f"openings in no situation of the plan: {(coverage.get('openings_unplanned') or {}).get('count', 0)}")
        out += [f"folded into an earlier source, from {kind}: {n}"
                for kind, n in sorted((coverage.get("folded") or {}).items())]
        out += [f"dropped, {reason.replace('_', ' ')}: {n}" for reason, n in (coverage.get("dropped") or {}).items()]
        out += _spread_lines(coverage)
    out += [f"note: {note}" for note in body["notes"]]
    out.append("spend: " + ", ".join(f"{stage} {usd:.4f} USD" for stage, usd in body["spend"].items()))
    return out


__all__ = ["HEADING", "STAGES", "counts", "lines"]
