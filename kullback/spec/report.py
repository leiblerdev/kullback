"""The per-build tier report: counts per tier, the trusted count split by Reference class, one table line.

The split is kullback/spec/split.py's class of each Task's kept References, read off the benchmark's
sidecar. The sidecar is measurement only: nothing in kullback/spec/trust.py reads it, and without the
sidecar the split is None rather than a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kullback.spec.split import reference_classes
from kullback.spec.trust import TIERS, TaskTier, verifier_from_of, workdir_tiers

# The report file in the workdir, one per build.
REPORT_FILE = "tiers.json"
TABLE_HEADER = ("| Build | Corpus | Verifier from | Tasks | Trusted | Replay-only | Untrusted | Set aside "
                "| Trusted by Reference (right, wrong, mixed, unknown) |")


def build_report(tiers: dict[str, TaskTier], *, verifier_from: str, corpus: Optional[str] = None,
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
    return {"build": build, "corpus": corpus, "verifier_from": verifier_from, "tasks": len(tiers),
            "tiers": counts, "trusted_by_reference": split,
            "rows": [tiers[task_id].row for task_id in sorted(tiers)]}


def table_line(report: dict) -> str:
    """One row under TABLE_HEADER, the format docs/builds/README.md keeps."""
    counts = report["tiers"]
    split = report.get("trusted_by_reference")
    by_class = ("n/a" if split is None else
                ", ".join(str(split.get(cls, 0)) for cls in ("right", "wrong", "mixed", "unknown")))
    cells = [report.get("build") or "", report.get("corpus") or "", report["verifier_from"], str(report["tasks"]),
             *(str(counts[tier]) for tier in TIERS), by_class]
    return "| " + " | ".join(cells) + " |"


def write_report(workdir: Any, *, corpus: Optional[str] = None, build: Optional[str] = None,
                 verifier_from: Optional[str] = None, sidecar: Optional[Any] = None) -> dict:
    """The build's tier report written to workdir/tiers.json and returned."""
    chosen = verifier_from or verifier_from_of(workdir)
    report = build_report(workdir_tiers(workdir, chosen), verifier_from=chosen, corpus=corpus, build=build,
                          classes=reference_classes(workdir, sidecar))
    path = Path(workdir) / REPORT_FILE
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
