"""The scenarios report: every count under its own heading, never in the build table (D372, D373).

Per source kind: born, passed, unsure, infeasible, struck and fixing; the Environment findings the
review filed; what the miner dropped, per reason; the coverage across sources; the human queue; and
what the scenario stages spent. Read off the store and the ledger; nothing is called.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from kullback.runner import budget
from kullback.runner.records import read_json
from kullback.scenarios import records as R

HEADING = "## scenarios"
STAGES = ("scenario_mining", "scenario_review", "scenario_fix")
FINDING_KEY = "environment:scenario:"


def counts(workdir: Any) -> dict:
    """The numbers the report prints, as data."""
    root = Path(workdir)
    records = R.load_scenarios(root)
    table = {kind: Counter(record.status for record in records if record.source.kind == kind)
             for kind in R.SOURCE_KINDS}
    mined = read_json(R.store_dir(root) / R.MINED, {}) or {}
    findings = [row for row in read_json(root / "examiner" / "findings.json", []) or []
                if isinstance(row, dict) and str(row.get("key") or "").startswith(FINDING_KEY)]
    stages = budget.load_totals(root)["stages"]
    spend = {stage: float((stages.get(stage) or {}).get("usd") or 0) for stage in STAGES}
    return {"table": {kind: {status: table[kind].get(status, 0) for status in R.STATUSES} for kind in table},
            "findings": len(findings), "coverage": mined.get("coverage") or {}, "notes": mined.get("notes") or [],
            "human": len(read_json(R.store_dir(root) / R.HUMAN, []) or []), "spend": spend}


def lines(workdir: Any) -> list[str]:
    """The report as printed lines, under the scenarios heading."""
    body = counts(workdir)
    out = [HEADING, "", "| source | " + " | ".join(R.STATUSES) + " |", "| --- |" + " --- |" * len(R.STATUSES)]
    for kind, row in body["table"].items():
        out.append(f"| {kind} | " + " | ".join(str(row[status]) for status in R.STATUSES) + " |")
    coverage = body["coverage"]
    out += ["", f"findings filed: {body['findings']}", f"human queue: {body['human']}"]
    if coverage:
        out.append(f"search scenarios the recordings never show: {coverage.get('search_not_in_recordings', 0)}")
        out.append(f"ask lines nothing else covers: {coverage.get('asks_not_covered', 0)}")
        out += [f"folded into an earlier source, from {kind}: {n}"
                for kind, n in sorted((coverage.get("folded") or {}).items())]
        out += [f"dropped, {reason.replace('_', ' ')}: {n}" for reason, n in (coverage.get("dropped") or {}).items()]
    out += [f"note: {note}" for note in body["notes"]]
    out.append("spend: " + ", ".join(f"{stage} {usd:.4f} USD" for stage, usd in body["spend"].items()))
    return out


__all__ = ["HEADING", "STAGES", "counts", "lines"]
