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
from kullback.examiner.plan import task_runs_of
from kullback.gates import PROTECTED_PATH, first_string
from kullback.gates.probes import version_hash, write_tools_of
from kullback.gates.verifier_suite import load_run, wrong_run
from kullback.runner.records import (
    EXAM_DIR,
    ToolSig,
    Verifier,
    VerifierHistory,
    VerifierVersion,
    as_dict,
    as_probe_pool,
    exam_history_path,
    exam_task_runs_path,
    exam_verifier_path,
    load_exam_history,
    read_json,
    write_json,
)

RUNS_DIR = "runs"
SPOKEN_DIR = "spoken"
# Where expose copies each Task's derived Verifier: read-only, never the proposal path verifiers/ (F33).
DERIVED_DIR = "derived"

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
    history: dict = field(default_factory=dict)
    probes: dict[str, list] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    bus: Any = None
    allowance_remaining: Optional[float] = None
    reroll_model: Any = None
    probe_model: Any = None
    run_probe: Any = None

    @property
    def exam_dir(self) -> Path:
        """The Examiner's root: the only directory its file tools are registered over."""
        return Path(self.workdir) / EXAM_DIR

    def current(self, task_id: str) -> Optional[Verifier]:
        """The Task's live Verifier: the exam proposal first, else the derived one."""
        path = exam_verifier_path(self.workdir, task_id)
        if path.is_file():
            try:
                return Verifier.model_validate(read_json(path))
            except ValueError:
                pass
        return self.verifiers.get(task_id)


def history_path(root: ExamRoot) -> Path:
    """Where the session keeps the per-Task Verifier histories: history.json beside the verifiers dir."""
    return exam_history_path(root.workdir)


def load_history(root: ExamRoot) -> dict[str, VerifierHistory]:
    """The per-Task Verifier histories off history.json (D127); a missing file or a bad row reads as none."""
    return load_exam_history(root.workdir)


def save_history(root: ExamRoot, history: dict) -> Path:
    """history.json beside the verifiers dir: every Task's versions in order, rejected rows kept (D127)."""
    return write_json(history_path(root), {
        task_id: as_dict(hist) if isinstance(hist, VerifierHistory) else hist
        for task_id, hist in (history or {}).items()})


def seeded_history(history: dict, task_id: str, current: Verifier, round_number: int = 0) -> VerifierHistory:
    """The Task's history, with the Verifier on disk as version 1 when it holds no version yet.

    Carried from kullback/examiner/tools.py `_seeded_history`: version 1, by "derive", the Verifier
    on disk before any repair, accepted, so the trusted gate reads the derived file as the current
    accepted version and the loosening gate has something to compare the first proposal against.
    """
    hist = history.get(task_id) or VerifierHistory(task_id=task_id)
    if not hist.versions:
        hist.versions.append(VerifierVersion(
            task_id=task_id, content_hash=version_hash(current), verifier_version=str(1),
            round=round_number, by="derive", reason="the Verifier on disk before any repair",
            accepted=True, verifier=current))
    history[task_id] = hist
    return hist


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
    copied += copy_derived(root, exam)
    spoken = write_spoken(root, exam)
    return {"copied": copied, "spoken": spoken}


def copy_derived(root: Path, exam: Path) -> list[str]:
    """Each Task's derived Verifier (workdir/verifiers/<task>.json) as exam/derived/<task>.json.

    The derived file only, never a proposal, and never into exam/verifiers/, which is the
    proposal path the gates rule on.
    """
    source = root / "verifiers"
    written: list[str] = []
    for path in sorted(source.glob("*.json")) if source.is_dir() else []:
        rel = f"{DERIVED_DIR}/{path.name}"
        if names_forbidden_path(rel) is not None:
            continue
        target = exam / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        written.append(rel)
    return written


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


def _probe_pools(root: ExamRoot) -> dict:
    """The session's probe Runs as ProbePools, so the trusted gate scores what the session holds."""
    return {task_id: as_probe_pool(task_id, runs) for task_id, runs in (root.probes or {}).items()}


def save_ruling_files(root: ExamRoot) -> None:
    """history.json and task_runs.json beside the verifiers dir, before every ruling (D201).

    The gates' loaders and the Builder's status read both through the same path functions
    (records.exam_history_path, exam_task_runs_path), so they rule over the evidence the session holds. task_runs rows are the Run dicts, as_dict of each Run.
    """
    save_history(root, root.history)
    runs = task_runs_of(root.replays, root.rerolls, workdir=root.workdir)
    write_json(exam_task_runs_path(root.workdir),
               {task_id: [as_dict(run) for run in rows] for task_id, rows in (runs or {}).items()})


def evidence_of(root: ExamRoot, task_id: Optional[str] = None) -> dict:
    """The gate evidence for a write under the Examiner's root, in one place (D79, D201).

    The eight reads of the verifiers binding the session holds: the replay rows, the re-roll
    rows, the version histories, the Runs the loosening gate compares versions over, the live
    Verifiers, the probe pools, the mined signatures and the canon rules, plus the task status
    the trusted gate reads. Empty is None, so a gate with nothing to rule over is skipped the
    way a missing workdir file skips it.

    With a `task_id`, the D79 suite's inputs for that Task join them (`reference_evidence`), so
    the suite rules on the write. A Task with no Reference Run on disk adds nothing, and the
    suite is skipped for missing evidence as before.
    """
    evidence = {"replays": root.replays or None, "rerolls": root.rerolls or None,
                "history": root.history or None, "task_runs": task_runs_of(root.replays, root.rerolls, workdir=root.workdir) or None,
                "verifiers": list(root.verifiers.values()) or None, "probes": _probe_pools(root) or None,
                "sigs": root.sigs or None, "rules": root.canon_rules,
                "task_status": root.task_status or None}
    if task_id is not None:
        evidence.update(reference_evidence(root, task_id))
    return evidence


def reference_evidence(root: ExamRoot, task_id: str) -> dict:
    """The D79 suite's inputs for one Task, off the References derive_all wrote (D79).

    The first Reference Run is the Reference, the rest are the seed Runs, the second is the
    alternate path when there is one, and the wrong Run is built from the Reference against the
    Task's Verifier in `root.verifiers` (the candidate while a proposal is ruled, which is what
    the suite tests). The loophole probe runs when the root holds a probe model and a
    `run_probe(model, verifier)`, else check 6 is not run. No Reference Run on disk is an empty
    dict, so "reference" stays absent.
    """
    workdir = Path(root.workdir)
    row = (read_json(workdir / "references.json", {}) or {}).get(task_id) or {}
    runs = reference_runs(workdir, row, (root.replays or {}).get(task_id), (root.rerolls or {}).get(task_id) or [])
    if not runs:
        return {}
    reference, verifier = runs[0], root.verifiers.get(task_id)
    try:
        wrong = wrong_run(verifier, reference, root.canon_rules) if verifier is not None else None
    except Exception:  # noqa: BLE001, a wrong Run that cannot be built leaves check 4 not run
        wrong = None
    return {"reference": reference, "seed_runs": runs[1:] or None,
            "alt_path_run": runs[1] if len(runs) > 1 else None, "wrong_run": wrong,
            "write_tools": write_tools_of(root.sigs), "probe_model": root.probe_model,
            "run_probe": root.run_probe}


__all__ = ["COPIED_DIRS", "COPIED_JSON", "DERIVED_DIR", "copy_derived", "EXAM_DIR", "RUNS_DIR", "SPOKEN_DIR", "ExamRoot",
           "Finding", "evidence_of", "finding_id_of", "finding_key", "history_path", "load_history",
           "names_forbidden_path", "open_by_key", "reference_evidence", "reference_runs", "save_history",
           "save_ruling_files", "seeded_history",
           "subject_of", "write_spoken", "expose", "FORBIDDEN_READS"]
