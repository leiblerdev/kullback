"""The per-build tier report: counts per tier, the trusted count split by Reference class, one table line.

The split is kullback/spec/split.py's class of each Task's kept References, read off the benchmark's
sidecar. The sidecar is measurement only: nothing in kullback/spec/trust.py reads it, and without the
sidecar the split is None rather than a guess.

Every row carries the content hash of the code that scored it (runner/code_hash.py). The report states
the hash once at its head and warns when its rows carry more than one, or when the report it replaces in
the workdir was scored under other code: counts are comparable only under one hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kullback.runner.code_hash import SHORT
from kullback.spec.split import reference_classes
from kullback.spec.trust import TIERS, TaskTier, workdir_tiers

# The report file in the workdir, one per build.
REPORT_FILE = "tiers.json"
TABLE_HEADER = ("| Build | Corpus | Verifier from | Tasks | Trusted | Replay-only | Untrusted | Set aside "
                "| Unconfirmed | Pending | Refused "
                "| Trusted by Reference (right, wrong, mixed, unknown) | Code |")


def code_hashes(rows: list[dict]) -> list[str]:
    """The distinct code hashes the rows were scored under, sorted; a row without one adds nothing."""
    return sorted({str(row["code_hash"]) for row in rows if row.get("code_hash")})


def _hash_warnings(hashes: list[str], previous: Optional[str] = None) -> list[str]:
    warnings = []
    if len(hashes) > 1:
        warnings.append(f"rows were scored under {len(hashes)} code hashes "
                        f"({', '.join(h[:SHORT] for h in hashes)}): their counts are not comparable")
    if previous and len(hashes) == 1 and previous != hashes[0]:
        warnings.append(f"the report this one replaces was scored under code {previous[:SHORT]}, "
                        f"this one under {hashes[0][:SHORT]}: the two are not comparable")
    return warnings


def build_report(tiers: dict[str, TaskTier], *, corpus: Optional[str] = None,
                 build: Optional[str] = None, classes: Optional[dict[str, str]] = None) -> dict:
    """Counts per tier and, given the classes, the trusted Tasks by Reference class; rows keep ids only."""
    counts = {tier: 0 for tier in TIERS}
    for task_tier in tiers.values():
        counts[task_tier.tier] += 1
    split = None
    if classes is not None:
        split = {}
        for task_id, task_tier in tiers.items():
            if task_tier.tier == "trusted":
                cls = classes.get(task_id, "unknown")
                split[cls] = split.get(cls, 0) + 1
    rows = [tiers[task_id].row for task_id in sorted(tiers)]
    hashes = code_hashes(rows)
    return {"build": build, "corpus": corpus, "tasks": len(tiers),
            "code_hash": hashes[0] if len(hashes) == 1 else None, "warnings": _hash_warnings(hashes),
            "tiers": counts, "trusted_by_reference": split, "rows": rows}


def header_line(report: dict) -> str:
    """The report's head: the foundation and the one code hash its rows were scored under, or "mixed"."""
    code = report.get("code_hash")
    return f"tier report, verifier from the Spec, code {code[:SHORT] if code else 'mixed'}"


def _code_cell(report: dict) -> str:
    code = report.get("code_hash")
    return code[:SHORT] if code else ("mixed" if code_hashes(report.get("rows") or []) else "")


def table_line(report: dict) -> str:
    """One row under TABLE_HEADER, the format docs/builds/README.md keeps."""
    counts = report["tiers"]
    split = report.get("trusted_by_reference")
    by_class = ("n/a" if split is None else
                ", ".join(str(split.get(cls, 0)) for cls in ("right", "wrong", "mixed", "unknown")))
    cells = [report.get("build") or "", report.get("corpus") or "", "spec", str(report["tasks"]),
             *(str(counts[tier]) for tier in TIERS), by_class, _code_cell(report)]
    return "| " + " | ".join(cells) + " |"


def write_report(workdir: Any, *, corpus: Optional[str] = None, build: Optional[str] = None,
                 sidecar: Optional[Any] = None) -> dict:
    """The build's tier report written to workdir/tiers.json and returned."""
    report = build_report(workdir_tiers(workdir), corpus=corpus, build=build,
                          classes=reference_classes(workdir, sidecar))
    path = Path(workdir) / REPORT_FILE
    report["warnings"] = _hash_warnings(code_hashes(report["rows"]), _previous_hash(path))
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _previous_hash(path: Path) -> Optional[str]:
    """The code hash of the report already at `path`, or None when there is none or it names none."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    hashes = code_hashes([row for row in body.get("rows") or [] if isinstance(row, dict)])
    return str(body.get("code_hash") or "") or (hashes[0] if len(hashes) == 1 else None)
