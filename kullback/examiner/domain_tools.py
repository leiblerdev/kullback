"""The Examiner's domain tools over its root: propose_verifier, probe, finding, reroll.

`propose_verifier` writes the next Verifier version off the current one; the D79 suite,
the pool and the loosening gate rule on the write through `gate_writes`, never here
(D79). `probe` scores a hand-written Run against the current Verifier and keeps it in
the pool. `finding` files what is wrong on the Builder's side with the file to edit
and what should differ, never a verb. `reroll` buys frontier Runs through the Runner
tool, refused at or below the remaining allowance (D112, D133). Every result carries
rows, never a count alone.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, RetryableToolError
from kullback.examiner.exam_files import (
    ExamRoot,
    Finding,
    evidence_of,
    finding_key,
    open_by_key,
    save_ruling_files,
    seeded_history,
)
from kullback.gates.bindings import rulings_for
from kullback.gates.hook import ruling_line
from kullback.gates.probes import version_hash, write_tools_of
from kullback.gates.verifier_suite import D79_STAGES, HELPERS_SRC, check_run, make_atom
from kullback.runner import tool as runner_tool
from kullback.runner.records import Event, Run, Verifier, VerifierVersion, as_dict, write_json

PRODUCED_VERIFIERS = ["verifiers"]
PRODUCED_PROBES = ["probes"]
PRODUCED_FINDINGS = ["findings"]
PRODUCED_REROLLS = ["rerolls"]


def render(result: BaseModel) -> str:
    """The summary line; everything else stays in details for the transcript."""
    return getattr(result, "summary", None) or getattr(result, "text", "") or ""


class ProposeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="Why the current version is wrong and what the new one changes.")
    drop: list[str] = Field(default_factory=list, description="Atom ids to remove.")
    add: list[dict] = Field(default_factory=list, description="Atoms to add: id, kind and payload.")


class ProposeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    content_hash: str
    verifier_version: str
    atoms: int = 0
    rulings: list[dict] = Field(default_factory=list, description="One record per ruling in the "
                                "shape attach_ruling uses: name, accepted, line, rows.")
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_VERIFIERS))


class ProbeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task whose Verifier the probe attacks.")
    bug_class: str = Field(default="", description="What the probe changes, in one phrase.")
    note: str = Field(default="", description="Why the Task is not done.")
    events: list[dict] = Field(description="The probe's events: type and payload each.")
    termination_reason: str = Field(default="success", description="How the Run ended.")
    base_run_id: Optional[str] = Field(default=None, description="The Run the probe was edited from.")


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    probe_id: str
    task_id: str
    scored_pass: bool
    failing_atom: Optional[str] = None
    pool_size: int
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_PROBES))


class FindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = Field(default=None)
    kind: str = Field(description="What the finding is about: assisted_tool, fidelity, "
                                  "reference_disagreement, suite, false_rejection, environment, other.")
    text: str = Field(description="What is wrong, in sentences.")
    rows: list[dict] = Field(default_factory=list, description="The evidence, one record per call or Task.")
    path: str = Field(default="", description="The Environment file the Builder should edit, or empty.")
    change: str = Field(default="", description="One line saying what should differ.")


class FindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    finding_id: str
    finding: dict = Field(default_factory=dict)
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_FINDINGS))


class RerollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    count: int = Field(default=1, ge=1, le=10, description="How many frontier Runs to buy (D112, D133).")


class RerollResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    runs: list[dict] = Field(default_factory=list)
    spent_usd: float = 0.0
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_REROLLS))


FINDING_KINDS = ("assisted_tool", "fidelity", "reference_disagreement", "suite",
                 "false_rejection", "environment", "other")

# Carried from kullback/examiner/tools.py: the atom shape the repair tool takes. An atom is
# an object with `id`, `kind` and `payload`, an object naming the atom's target.
ATOM_SHAPE = ("an atom is an object with `id`, `kind` (required, allowed, question, communicate, hard) "
              "and `payload`, an object naming the atom's target")
_SHAPE_ASK = ("Your proposal was refused for the shape of one atom, not for what it says. Call "
              "propose_verifier again with the same reason and that atom written as: {shape}")


def payload_shape(atoms: Iterable[Any], kind: Any) -> dict:
    """The payload an atom of this kind carries in the Verifier being proposed from."""
    rows = [atom for atom in atoms or () if getattr(atom, "target", None)]
    for atom in rows:
        if getattr(atom, "kind", None) == kind:
            return dict(atom.target)
    return dict(rows[0].target) if rows else {}


def corrected_atom(row: dict, atoms: Iterable[Any] = ()) -> str:
    """The row the model sent, written out in the shape the tool takes."""
    body = dict(row)
    body.pop("payload", None)
    body.pop("target", None)
    example = {"id": body.pop("id", None) or "atom-1", "kind": body.pop("kind", None) or "required",
               "payload": payload_shape(atoms, row.get("kind"))}
    example.update({key: value for key, value in body.items() if key != "predicate_src"})
    return json.dumps(example, sort_keys=True, default=str, ensure_ascii=False)

def _atom_of(row: dict, atoms: Iterable[Any] = ()):
    """One `add` row as an Atom, carried from kullback/examiner/tools.py (`_atom_of`)."""
    body = dict(row)
    missing = [name for name in ("id", "kind") if not body.get(name)]
    if missing:
        raise RetryableToolError(
            f"the atom {row!r} names no {', '.join(missing)}: {ATOM_SHAPE}. The same atom with "
            f"every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    atom_id, kind = body.pop("id"), body.pop("kind")
    payload = body.pop("payload", None) or body.pop("target", None) or {}
    if not isinstance(payload, dict):
        raise RetryableToolError(
            f"the atom {atom_id!r} carries its payload as a {type(payload).__name__}, not an object: "
            f"{ATOM_SHAPE}. The same atom with every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    return make_atom(atom_id, kind, payload, helpers=HELPERS_SRC, **body)


def _candidate_atoms(current: Verifier, drop: Iterable[str], add: Iterable[Any]) -> list:
    """The atoms of the next version, carried from kullback/examiner/tools.py (`_repair`)."""
    dropped = set(drop)
    added = [row if hasattr(row, "kind") else _atom_of(row, current.atoms) for row in add]
    atoms = [a for a in current.atoms if a.id not in dropped] + added
    if not atoms:
        raise ValueError("a proposal cannot leave the Verifier without atoms")
    return atoms


def _events(rows: list[dict]) -> list[Event]:
    """Probe event rows as Events, carried from kullback/examiner/tools.py."""
    out = []
    for number, row in enumerate(rows):
        body = dict(row)
        body.setdefault("idx", number)
        out.append(Event.model_validate(body))
    return out


def _current(root: ExamRoot, task_id: str) -> Verifier:
    verifier = root.current(task_id)
    if verifier is None:
        raise LookupError(f"task {task_id} has no live Verifier; derive first")
    return verifier


def _next_version(current: Verifier) -> str:
    try:
        return str(int(current.verifier_version or 0) + 1)
    except (TypeError, ValueError):
        return f"{current.verifier_version or 1}+1"


def _ruling_records(rulings: Iterable[Any]) -> list[dict]:
    """One record per ruling in the shape attach_ruling uses: name, accepted, line, rows."""
    return [{"name": ruling.stage, "accepted": bool(ruling.passed), "line": ruling_line(ruling),
             "rows": [dict(row) for row in (ruling.rows or [])]} for ruling in rulings]


def _keep_suite_status(root: ExamRoot, task_id: str, records: list[dict]) -> None:
    """An accepted proposal's D79 suite as the Task's status row, in the exposed exam/task_status.json.

    Never the workdir's file: that one is derive_all's. A ruling with no suite leaves the row as it was.
    """
    ran = {record["name"]: record["accepted"] for record in records if record["name"] in D79_STAGES}
    if not ran:
        return
    root.task_status[task_id] = {**(root.task_status.get(task_id) or {}), "verifier_passed": all(ran.values()),
                                 "checks": {D79_STAGES[stage]: passed for stage, passed in ran.items()},
                                 "not_run": [stage for stage in D79_STAGES if stage not in ran]}
    write_json(root.exam_dir / "task_status.json", root.task_status)


def _refuse_proposal(task_id: str, version: str, records: list[dict]) -> RetryableToolError:
    """A refused proposal as a retryable error: every ruling line, and the rows behind each failure."""
    lines = "\n".join(record["line"] for record in records) or "no gate ruled on the proposal"
    rows = "; ".join(f"{record['name']}: {record['rows']!r}"
                     for record in records if not record["accepted"])
    return RetryableToolError(
        f"proposal version {version} for task {task_id} refused, the previous file stands:\n"
        f"{lines}" + (f"\nrows behind the failures: {rows}" if rows else ""),
        ask=("Read the rows behind the failures and propose again with atoms answering them, "
             "or file a finding naming the Environment file and what differs after two refusals."))


def _propose_verifier(root: ExamRoot):
    """One proposal as a transaction (D201): ruled on the write, kept only when every ruling passed.

    The candidate joins the Task's history as an accepted trial version before the ruling, so
    the trusted gate reads it as the current accepted version and the loosening gate compares
    it against the version before it (D127). A refused proposal keeps its row, rewritten as
    rejected with the gates that rejected it: the history is the record, never dropped (D201).
    """
    async def propose_verifier(args: ProposeArgs) -> ProposeResult:
        current = _current(root, args.task_id)
        candidate = current.model_copy(deep=True, update={
            "atoms": _candidate_atoms(current, args.drop, args.add),
            "verifier_version": _next_version(current)})
        digest = version_hash(candidate)
        hist = seeded_history(root.history, args.task_id, current)
        hist.versions.append(VerifierVersion(
            task_id=args.task_id, content_hash=digest, verifier_version=str(len(hist.versions) + 1),
            parent_hash=next((v.content_hash for v in reversed(hist.versions) if v.accepted), None),
            by="repair", reason=args.reason, accepted=True, verifier=candidate))
        root.history[args.task_id] = hist
        target = root.exam_dir / "verifiers" / f"{args.task_id}.json"
        previous = target.read_bytes() if target.is_file() else None
        previous_verifier = root.verifiers.get(args.task_id)
        root.verifiers[args.task_id] = candidate
        write_json(target, as_dict(candidate))
        save_ruling_files(root)
        records = _ruling_records(rulings_for(
            root.exam_dir, f"verifiers/{args.task_id}.json", root.workdir,
            evidence=evidence_of(root, task_id=args.task_id)))
        if records and all(record["accepted"] for record in records):
            _keep_suite_status(root, args.task_id, records)
            summary = (f"proposal version {candidate.verifier_version} for task {args.task_id} "
                       f"({digest[:12]}) accepted: " + "; ".join(record["line"] for record in records))
            return ProposeResult(summary=summary, task_id=args.task_id, content_hash=digest,
                                verifier_version=candidate.verifier_version,
                                atoms=len(candidate.atoms), rulings=records)
        if previous_verifier is None:
            root.verifiers.pop(args.task_id, None)
        else:
            root.verifiers[args.task_id] = previous_verifier
        hist.versions[-1] = hist.versions[-1].model_copy(
            update={"accepted": False,
                    "rejected_by": [record["name"] for record in records if not record["accepted"]]})
        root.history[args.task_id] = hist
        save_ruling_files(root)
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(previous)
        raise _refuse_proposal(args.task_id, candidate.verifier_version, records)

    return propose_verifier


def _probe(root: ExamRoot):
    async def probe(args: ProbeArgs) -> ProbeResult:
        verifier = _current(root, args.task_id)
        pool = root.probes.setdefault(args.task_id, [])
        probe_id = f"probe-{args.task_id}-{len(pool) + 1}"
        run = Run(run_id=probe_id, task_id=args.task_id, model="probe:examiner",
                  events=_events(args.events), termination_reason=args.termination_reason,
                  parent_run_id=args.base_run_id)
        write_tools = write_tools_of(root.sigs)
        passed, failing = check_run(verifier, run, root.canon_rules, write_tools=write_tools)
        pool.append(run)
        target = root.exam_dir / "probes" / args.task_id / f"{len(pool)}.json"
        write_json(target, as_dict(run))
        verdict = ("accepted: it scores a pass, so the Verifier is loose"
                   if passed else f"rejected at {failing}")
        summary = (f"probe {probe_id} ({args.bug_class or 'no class'}) on task {args.task_id}: "
                   f"{verdict}; pool holds {len(pool)}")
        return ProbeResult(summary=summary, probe_id=probe_id, task_id=args.task_id,
                           scored_pass=bool(passed),
                           failing_atom=failing if not passed else None, pool_size=len(pool))

    return probe


def _finding(root: ExamRoot):
    async def finding(args: FindingArgs) -> FindingResult:
        kind = (args.kind or "").strip()
        if kind not in FINDING_KINDS:
            raise ValueError(f"{args.kind!r} is not a finding kind. The kinds are: "
                             f"{', '.join(FINDING_KINDS)}.")
        record = Finding(task_id=args.task_id, kind=kind, text=args.text, rows=list(args.rows),
                         path=args.path, change=args.change, source="model")
        key = finding_key(kind, record.path or record.change, args.task_id or "")
        existing = open_by_key(root.findings).get(key)
        if existing is not None:
            raise ValueError(f"{existing} already says this ({kind}"
                             + (f" in {args.path}" if args.path else "")
                             + (f" on task {args.task_id}" if args.task_id else "")
                             + "); it is filed and the Builder has not answered it yet. Act on it, "
                               "or file a finding that says something else.")
        root.findings.append(record)
        body = record.as_dict()
        if root.bus is not None:
            root.bus.publish_custom("finding", body)
        summary = f"finding {body['finding_id']} ({kind}) filed" + \
            (f" on task {args.task_id}" if args.task_id else "") + \
            (f", naming {args.path}" if args.path else "")
        return FindingResult(summary=summary, finding_id=body["finding_id"], finding=body)

    return finding


def _reroll(root: ExamRoot):
    async def reroll(args: RerollArgs) -> RerollResult:
        if root.allowance_remaining is not None and root.allowance_remaining <= 0:
            raise RuntimeError(f"the allowance is spent ({root.allowance_remaining:.2f} left)")
        if root.reroll_model is None:
            raise ValueError("no reroll model was given to this examination; "
                             "file a finding naming the Task instead")
        reports = runner_tool.reroll(root.workdir, args.task_id, root.reroll_model,
                                     count=args.count, workdir=root.workdir)
        rows = []
        spent = 0.0
        for report in reports or []:
            body = report.as_dict() if hasattr(report, "as_dict") else dict(report)
            spend = body.get("spend") or {}
            price = spend.get("usd", 0.0) if isinstance(spend, dict) else 0.0
            try:
                spent += float(price)
            except (TypeError, ValueError):
                pass
            rows.append({"run_id": body.get("run_id"), "verdict": body.get("verdict"),
                         "termination_reason": body.get("termination_reason"), "spend": spend})
        if root.allowance_remaining is not None:
            root.allowance_remaining -= spent
        summary = (f"reroll of task {args.task_id}: {len(rows)} Runs, "
                   f"{sum(1 for r in rows if r['termination_reason'] == 'success')} finished, "
                   f"{spent:.4f} USD")
        return RerollResult(summary=summary, task_id=args.task_id, runs=rows, spent_usd=spent)

    return reroll


def domain_tools(root: ExamRoot) -> list[AgentTool]:
    """The four domain tools over one ExamRoot."""
    return [
        AgentTool("propose_verifier", "Propose a new Verifier version by dropping and adding atoms; "
                  "the D79 suite, the pool and the loosening gate rule on the write.",
                  ProposeArgs, ProposeResult, _propose_verifier(root), render=render),
        AgentTool("probe", "Score a hand-written Run against a Task's current Verifier and keep it "
                  "in the Task's pool.",
                  ProbeArgs, ProbeResult, _probe(root), render=render),
        AgentTool("finding", "File what is wrong on the Builder's side: the Environment file to edit "
                  "in `path` and the one line saying what should differ in `change`, never a verb.",
                  FindingArgs, FindingResult, _finding(root), render=render),
        AgentTool("reroll", "Buy more frontier Runs of a Task through the Runner (D112, D133).",
                  RerollArgs, RerollResult, _reroll(root), render=render),
    ]


def tool_names() -> tuple[str, ...]:
    """The domain tool names, so tests pin the Builder-called surface."""
    return ("propose_verifier", "probe", "finding", "reroll")


__all__ = ["FindingArgs", "FindingResult", "ProbeArgs", "ProbeResult", "ProposeArgs",
           "ProposeResult", "RerollArgs", "RerollResult", "domain_tools", "render", "tool_names"]
