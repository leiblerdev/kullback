"""The Examiner's root under the workdir: what the model session may read (D123).

`expose` copies the read surface into `exam/` at session start: the Runs as JSONL, the
JSON files by name, the task, intent and user-rule records, and one spoken file per Task
with the user turns of its References. It never copies the Builder's compiled side
(bodies, the db, the schema, `env/`, the sandbox, the overlays): the derivation's
`FORBIDDEN_INPUTS` and the old extension's `FORBIDDEN_READS` below. `Finding` is what
the session returns for each loss: the Environment file the Builder should edit and
the one line saying what should differ, never a verb.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from kullback.examiner import derive as derive_mod
from kullback.gates import PROTECTED_PATH, first_string
from kullback.gates.verifier_suite import load_run
from kullback.runner.records import ToolSig, Verifier, read_json

EXAM_DIR = "exam"
RUNS_DIR = "runs"
SPOKEN_DIR = "spoken"

# The JSON files copied by name from the workdir root into the Examiner's root.
COPIED_JSON = ("replays.json", "rerolls.json", "references.json", "task_status.json",
               "verifiers.json", "probes.json")

# The record directories copied whole from the workdir root into the Examiner's root.
COPIED_DIRS = ("tasks", "intents", "user_rules")

# Carried verbatim from kullback/examiner/extension.py (D122, D123): the old module is
# deleted by another stream, and this copy keeps the refusal identical. What the Examiner
# never reads: the Builder's compiled side (D123). A path segment equal to one of
# these, bare or under any directory, is refused whatever tool the call is for.
FORBIDDEN_READS = ("bodies.json", "env", "sandbox", "environment.json", "db.json", "schema.json", "overlays", "tools")
_SEPARATORS = re.compile(r"[/\\]")


def names_forbidden_path(value: Any) -> Optional[str]:
    """The first string in a value that names a path the Examiner never reads: the Builder's compiled side
    (a segment in FORBIDDEN_READS, D123) or the two packages no agent writes (the gates' own rule, D122)."""
    return first_string(value, lambda text: PROTECTED_PATH.search(text) is not None
                        or any(segment in FORBIDDEN_READS for segment in _SEPARATORS.split(text) if segment))


FindingKind = Literal["assisted_tool", "fidelity", "reference_disagreement", "suite",
                      "false_rejection", "environment", "other"]
FindingSource = Literal["derive", "model"]


@dataclass
class Finding:
    """One loss the Builder should answer: the file under `env/` to edit and what should differ.

    `rows` is the evidence, one record per call or Task, never a bare count (learnings 7).
    `path` is empty when no Environment file answers the loss (a Verifier of the Examiner's
    own that failed its suite). `source` says whether the code derivation or the model filed it.
    """

    task_id: Optional[str] = None
    kind: FindingKind = "other"
    text: str = ""
    rows: list[dict] = field(default_factory=list)
    path: str = ""
    change: str = ""
    source: FindingSource = "derive"

    def as_dict(self) -> dict:
        """This finding as JSON-ready data, with a stable finding id derived from its key."""
        body = {"finding_id": finding_id_of(self), "task_id": self.task_id, "kind": self.kind,
                "text": self.text, "rows": [dict(row) for row in self.rows], "path": self.path,
                "change": self.change, "source": self.source,
                "key": finding_key(self.kind, subject_of(self), self.task_id or "")}
        return body


def subject_of(finding: Finding) -> str:
    """What this finding is about: the file named, else the one-line change."""
    return finding.path or finding.change


def finding_id_of(finding: Finding) -> str:
    """A stable id for a finding, off its key, so the same loss reads as the same finding."""
    return "f-" + hashlib.sha1(finding_key(
        finding.kind, subject_of(finding), finding.task_id or "").encode()).hexdigest()[:12]


def finding_key(kind: str, subject: str = "", task_id: str = "") -> str:
    """The kind and the thing it is about, carried from kullback/examiner/findings.py (D170)."""
    return f"{kind}:{subject}:{task_id}"


def open_by_key(findings: list[Finding]) -> dict[str, str]:
    """The finding id of every finding filed so far, by key (D170)."""
    return {finding_key(row.kind, subject_of(row), row.task_id or ""): finding_id_of(row)
            for row in findings}


@dataclass
class ExamRoot:
    """What the Examiner's session works over: paths plus the state its tools read and write."""

    workdir: Path
    verifiers: dict[str, Verifier] = field(default_factory=dict)
    sigs: list[ToolSig] = field(default_factory=list)
    canon_rules: Any = None
    replays: dict = field(default_factory=dict)
    rerolls: dict = field(default_factory=dict)
    task_status: dict = field(default_factory=dict)
    probes: dict[str, list] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    bus: Any = None
    allowance_remaining: Optional[float] = None
    reroll_model: Any = None

    @property
    def exam_dir(self) -> Path:
        """The Examiner's root: the only directory its file tools are registered over."""
        return Path(self.workdir) / EXAM_DIR

    def current(self, task_id: str) -> Optional[Verifier]:
        """The Task's live Verifier: the exam proposal first, else the derived one."""
        path = self.exam_dir / "verifiers" / f"{task_id}.json"
        if path.is_file():
            try:
                return Verifier.model_validate(read_json(path))
            except ValueError:
                pass
        return self.verifiers.get(task_id)


def expose(workdir: Any) -> dict:
    """Copy the read surface into `exam/`: the Runs, the JSON files, the records, the spoken turns.

    Never a forbidden file: a source naming one is skipped, and the copy is refused outright
    when the workdir itself holds a forbidden name under the copied roots.
    """
    root = Path(workdir)
    exam = root / EXAM_DIR
    copied: list[str] = []
    spoken: list[str] = []
    for path in sorted((root / RUNS_DIR).rglob("*.jsonl")) if (root / RUNS_DIR).is_dir() else []:
        rel = path.relative_to(root)
        if names_forbidden_path(str(rel)) is not None:
            continue
        target = exam / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copied.append(str(rel))
    for name in COPIED_JSON:
        source = root / name
        if not source.is_file() or names_forbidden_path(name) is not None:
            continue
        target = exam / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        copied.append(name)
    for name in COPIED_DIRS:
        source = root / name
        if not source.is_dir() or names_forbidden_path(name) is not None:
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if names_forbidden_path(str(rel)) is not None:
                continue
            target = exam / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            copied.append(str(rel))
    spoken = write_spoken(root, exam)
    return {"copied": copied, "spoken": spoken}


def write_spoken(root: Path, exam: Path) -> list[str]:
    """One file per Task with the user turns of its References, the way derive reads them."""
    references = read_json(root / "references.json", {}) or {}
    replays = read_json(root / "replays.json", {}) or {}
    rerolls = read_json(root / "rerolls.json", {}) or {}
    task_ids = sorted({*(references or {}), *(replays or {}), *(rerolls or {})})
    written: list[str] = []
    for task_id in task_ids:
        runs = reference_runs(root, references.get(task_id) or {}, replays.get(task_id),
                              rerolls.get(task_id) or [])
        if not runs:
            continue
        target = exam / SPOKEN_DIR / f"{task_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(derive_mod.spoken_text(runs), encoding="utf-8")
        written.append(f"{SPOKEN_DIR}/{task_id}.txt")
    return written


def reference_runs(root: Path, row: dict, replay_rows: Any, reroll_rows: list) -> list:
    """The Reference Runs of one Task, joined from references.json back to the replay rows."""
    named = [str(r.get("run_id")) for r in (row.get("references") or []) if r.get("run_id")]
    by_id: dict[str, str] = {}
    if isinstance(replay_rows, dict):
        for trace_row in replay_rows.values():
            if isinstance(trace_row, dict) and trace_row.get("run_id") and trace_row.get("path"):
                by_id[str(trace_row["run_id"])] = str(trace_row["path"])
    for reroll_row in reroll_rows or []:
        if isinstance(reroll_row, dict) and reroll_row.get("run_id") and reroll_row.get("path"):
            by_id[str(reroll_row["run_id"])] = str(reroll_row["path"])
    runs = []
    for run_id in named:
        path = by_id.get(run_id)
        if not path:
            continue
        full = root / path if not Path(path).is_absolute() else Path(path)
        if not full.is_file() or names_forbidden_path(str(path)) is not None:
            continue
        try:
            runs.append(load_run(str(full)))
        except (OSError, ValueError, TypeError):
            continue
    return runs


def evidence_of(root: ExamRoot) -> dict:
    """The gate evidence for a write under the Examiner's root, in one place (D79, D201).

    The six bindings the verifiers gate reads: the replay rows, the re-roll rows, the mined
    signatures, the canon rules, the live Verifiers and the task status. Empty is None, so a
    gate with nothing to rule over is skipped the way a missing workdir file skips it. The
    trust gate needs version history the new flow does not keep, so a session root carries no
    task status and proposals are ruled through the oracle, the suite and the pool alone.
    """
    return {"replays": root.replays or None, "rerolls": root.rerolls or None,
            "sigs": root.sigs or None, "rules": root.canon_rules,
            "verifiers": list(root.verifiers.values()) or None,
            "task_status": root.task_status or None}


__all__ = ["COPIED_DIRS", "COPIED_JSON", "EXAM_DIR", "RUNS_DIR", "SPOKEN_DIR", "ExamRoot",
           "Finding", "evidence_of", "finding_id_of", "finding_key", "names_forbidden_path",
           "open_by_key", "reference_runs", "subject_of", "write_spoken", "expose", "FORBIDDEN_READS"]
