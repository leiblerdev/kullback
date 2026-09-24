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

import hashlib
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
from kullback.gates import Ruling
from kullback.gates.bindings import ALT_PATH_STAGE, WAIVED_ROW, rows_for, rulings_for
from kullback.gates.hook import ruling_line
from kullback.gates.loosening import false_rejection_gate, loosening_gate
from kullback.gates.probes import version_hash, write_tools_of
from kullback.gates.trust import trusted_gate
from kullback.gates.verifier_suite import (
    D79_STAGES,
    HELPERS_SRC,
    NO_ATOM_CHECKED,
    check_run,
    make_atom,
    not_run_reason,
    validate_verifier,
)
from kullback.runner import budget
from kullback.runner import tool as runner_tool
from kullback.runner.records import Atom, Event, GateResult, Run, Verifier, VerifierVersion, as_dict, write_json
from kullback.runner.target import SCORED_KINDS, atom_payload

PRODUCED_VERIFIERS = ["verifiers"]
PRODUCED_PROBES = ["probes"]
PRODUCED_FINDINGS = ["findings"]
PRODUCED_REROLLS = ["rerolls"]


def render(result: BaseModel) -> str:
    """The summary line; everything else stays in details for the transcript."""
    return getattr(result, "summary", None) or getattr(result, "text", "") or ""


# The atoms check_run scores, in the words the tool description and the refusal both use (F51).
SCORABLE = ("check_run scores an atom whose payload kind is one of " + ", ".join(SCORED_KINDS)
            + "; a `hard` atom needs `predicate_src`, Python source defining "
            "check(pre_state, write_call, transcript) that returns True when a call keeps the rule")


class ProposeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="Why the current version is wrong and what the new one changes.")
    drop: list[str] = Field(default_factory=list, description="Atom ids to remove.")
    add: list[dict] = Field(default_factory=list, description="Atoms to add: id, kind and payload. "
                            + SCORABLE + "; any other atom is refused.")


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


# The fields propose_verifier sets itself on an atom: a row's own copy is taken out first (F30).
SET_BY_TOOL = ("id", "kind", "payload", "target", "predicate_src")
# The fields a row may carry beside them, passed through to the Atom as they are.
ROW_FIELDS = tuple(name for name in Atom.model_fields if name not in SET_BY_TOOL)


def corrected_atom(row: dict, atoms: Iterable[Any] = ()) -> str:
    """The row the model sent, written out in the shape the tool takes."""
    example = {"id": row.get("id") or "atom-1", "kind": row.get("kind") or "required",
               "payload": payload_shape(atoms, row.get("kind"))}
    example.update({key: value for key, value in row.items() if key in ROW_FIELDS})
    return json.dumps(example, sort_keys=True, default=str, ensure_ascii=False)


def _atom_of(row: dict, atoms: Iterable[Any] = ()):
    """One `add` row as an Atom, carried from kullback/examiner/tools.py (`_atom_of`)."""
    missing = [name for name in ("id", "kind") if not row.get(name)]
    if missing:
        raise RetryableToolError(
            f"the atom {row!r} names no {', '.join(missing)}: {ATOM_SHAPE}. The same atom with "
            f"every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    unknown = sorted(key for key in row if key not in SET_BY_TOOL and key not in ROW_FIELDS)
    if unknown:
        raise RetryableToolError(
            f"the atom {row['id']!r} carries fields an atom does not have: {', '.join(unknown)}. "
            f"{ATOM_SHAPE}. The same atom with every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    payload = row.get("payload") or row.get("target") or {}
    if not isinstance(payload, dict):
        raise RetryableToolError(
            f"the atom {row['id']!r} carries its payload as a {type(payload).__name__}, not an object: "
            f"{ATOM_SHAPE}. The same atom with every field it needs: {corrected_atom(row, atoms)}",
            ask=_SHAPE_ASK.format(shape=corrected_atom(row, atoms)))
    if row.get("predicate_src"):
        # F51: the rule the model wrote is the Hard atom's check; make_atom wraps it off the payload.
        payload = {"kind": "hard", **payload, "predicate_src": row["predicate_src"]}
    fields = {key: value for key, value in row.items() if key in ROW_FIELDS}
    atom = make_atom(row["id"], row["kind"], payload, helpers=HELPERS_SRC, **fields)
    if not scored(atom):
        raise RetryableToolError(
            f"the atom {row['id']!r} is not scored, so no Run could fail it: {SCORABLE}.",
            ask="Propose again with every added atom one check_run scores, or drop it.")
    return atom


def scored(atom: Atom) -> bool:
    """Whether check_run knows this atom's payload, or the judge answers it (F51).

    An empty payload or a payload kind check_run has no branch for would pass every Run, and a Hard
    atom with no rule answers nothing.
    """
    payload = atom_payload(atom)
    if atom.judge:
        return True
    if not payload:
        return False
    if atom.kind == "hard":
        return bool(payload.get("predicate_src"))
    return payload.get("kind") in SCORED_KINDS


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


def _record(ruling: Any, held: Optional[str] = None) -> dict:
    """One ruling in the shape attach_ruling uses: name, accepted, line, rows.

    A check that did not run has `accepted` None and says why in `not_run` (F45): it is not a
    failure of the proposal. `held` is that reason for a gate that says so in its metrics.
    """
    rows = [dict(row) for row in (ruling.rows or [])]
    why = held if held is not None else (None if ruling.passed else not_run_reason(ruling.failures))
    if why is None:
        return {"name": ruling.stage, "accepted": bool(ruling.passed), "line": ruling_line(ruling), "rows": rows}
    return {"name": ruling.stage, "accepted": None, "not_run": why, "line": f"{ruling.stage} not run ({why})",
            "rows": rows}


def _ruling_records(rulings: Iterable[Any]) -> list[dict]:
    return [_record(ruling) for ruling in rulings]


def _gate_record(result: GateResult, verifier: Verifier) -> dict:
    """A gate this tool ran itself, as a record; a trusted ruling that holds the Task only for
    checks nobody could run is not run, with the gate's reason (F45)."""
    ruling = Ruling(stage=result.stage, passed=bool(result.passed), failures=list(result.failures),
                    rows=rows_for(result, {"verifier": verifier}))
    metrics = result.metrics or {}
    untrusted = metrics.get("untrusted") or {}
    held = None
    if result.stage == "trusted" and untrusted and set(untrusted) <= set(metrics.get("not_run_only") or ()):
        held = "; ".join(f"task {task_id}: {reason}" for task_id, reason in sorted(untrusted.items()))
    return _record(ruling, held)


def _suite_row(records: list[dict], base: dict) -> Optional[dict]:
    """The status row the D79 checks among these records give the Task, in the derivation's shape.

    A check that did not run is `not_run` with its reason and not passed, so the Task stays
    untrusted (F45). A waived alt-path check (D199), by its ruling's row or by the Task's row when
    the check did not run, is the check not passed and the Verifier passed.
    """
    suite = [record for record in records if record["name"] in D79_STAGES]
    if not suite:
        return None
    ran = {record["name"]: record["accepted"] for record in suite if record["accepted"] is not None}
    held = {record["name"]: record["not_run"] for record in suite if record["accepted"] is None}
    waived = any(record["name"] == ALT_PATH_STAGE and record["accepted"]
                 and any(WAIVED_ROW in row for row in record["rows"]) for record in suite) \
        or bool(base.get("second_path_waived") and ALT_PATH_STAGE in held)
    passed = all(ran.values()) and set(held) <= ({ALT_PATH_STAGE} if waived else set())
    if waived:
        ran[ALT_PATH_STAGE] = False
    return {"verifier_passed": passed, "second_path_waived": waived,
            "checks": {D79_STAGES[stage]: bool(ran.get(stage)) for stage in D79_STAGES if stage in ran or stage in held},
            "not_run": [stage for stage in D79_STAGES if stage not in ran],
            "not_run_reasons": {D79_STAGES[stage]: why for stage, why in held.items()}}


def _rule(root: ExamRoot, task_id: str, candidate: Verifier) -> list[dict]:
    """Every ruling on the proposal, judged by its own Task only (F47).

    The binding stops at its first refusal, and a check that did not run reads as one there. That
    is no refusal of the proposal (F45), so when the ruling stopped on one, the D79 checks run again
    here in full and the gates after them rule on the write, over the same evidence.
    """
    evidence = evidence_of(root, task_id=task_id)
    records = _ruling_records(rulings_for(root.exam_dir, f"verifiers/{task_id}.json", root.workdir,
                                          evidence=evidence))
    if not records or records[-1]["accepted"] is not None or records[-1]["name"] not in D79_STAGES \
            or evidence.get("reference") is None:
        return records
    out = [record for record in records if record["name"] not in D79_STAGES]
    results = validate_verifier(candidate, evidence["reference"], None, evidence.get("wrong_run"),
                                evidence.get("alt_path_run"), canon=evidence.get("rules"),
                                write_tools=evidence.get("write_tools"), seed_runs=evidence.get("seed_runs"),
                                model=evidence.get("probe_model"), run_probe=evidence.get("run_probe"))
    for result in results:
        out.append(_gate_record(result, candidate))
        if out[-1]["accepted"] is False:
            return out
    base = dict(evidence.get("task_status") or {})
    base[task_id] = {**(base.get(task_id) or {}), **(_suite_row(out, base.get(task_id) or {}) or {})}
    runs, replays, rerolls = evidence.get("task_runs") or {}, evidence.get("replays") or {}, evidence.get("rerolls") or {}
    rules, sigs = evidence.get("rules"), root.sigs or []
    gates = (lambda: loosening_gate(evidence.get("history") or {}, runs, replays, rerolls, rules, sigs),
             lambda: false_rejection_gate(evidence["verifiers"], runs, replays, rerolls, rules, sigs,
                                          task_status=evidence.get("task_status")),
             lambda: trusted_gate(base, evidence["verifiers"], evidence.get("probes") or {},
                                  evidence.get("history") or {}, {}, runs, replays, rerolls, rules, sigs))
    for run_gate in gates:
        out.append(_gate_record(run_gate(), candidate))
        if out[-1]["accepted"] is False:
            break
    return out


def _keep_suite_status(root: ExamRoot, task_id: str, records: list[dict]) -> None:
    """An accepted proposal's D79 suite as the Task's status row, in the exposed exam/task_status.json.

    Never the workdir's file: that one is derive_all's. A ruling with no suite leaves the row as it was.
    """
    base = root.task_status.get(task_id) or {}
    row = _suite_row(records, base)
    if row is None:
        return
    root.task_status[task_id] = {**base, **row}
    write_json(root.exam_dir / "task_status.json", root.task_status)


# The D79 checks a Run should fail or pass, where the atom the rows name is the one to change.
_ATOM_CHECKS = {"verifier_empty_run": "the atom the empty Run passed on",
                "verifier_alt_path": "the atom that turned the second path away"}

# Atom kinds an empty Run passes whatever the Task: none of them demands that the Run did or said something (F37).
_EMPTY_PASSING_KINDS = {"allowed", "forbidden", "hard"}
_EMPTY_RUN_ADVICE = ("every atom here passes an empty Run: add a question or communicate atom for what the "
                     "Reference said, or a required write")

# How many refusals of one Task a session takes before the tool says to stop proposing for it (F32).
REFUSALS_PER_TASK = 3
_STOP = " Stop proposing for this Task in this session."


def _atoms_named(records: list[dict]) -> str:
    """The atom ids the empty-run and alt-path failures turn on, off their rows, when the rows carry them."""
    lines = []
    for record in records:
        if record["accepted"] is not False or record["name"] not in _ATOM_CHECKS:
            continue
        for row in record["rows"]:
            if isinstance(row, dict) and row.get("atom"):
                text = f" ({row['text']})" if row.get("text") else ""
                lines.append(f"{record['name']}: {_ATOM_CHECKS[record['name']]} is {row['atom']}{text}")
        kinds = {row["kind"] for row in record["rows"] if isinstance(row, dict) and row.get("kind")}
        none_checked = any(isinstance(row, dict) and row.get("failure") == NO_ATOM_CHECKED for row in record["rows"])
        if record["name"] == "verifier_empty_run" and (none_checked or (kinds and kinds <= _EMPTY_PASSING_KINDS)):
            lines.append(_EMPTY_RUN_ADVICE)
    return "\n".join(lines)


def _refuse_proposal(task_id: str, version: str, records: list[dict], stop: bool = False) -> RetryableToolError:
    """A refused proposal as a retryable error: every ruling line, and the rows behind each failure."""
    lines = "\n".join(record["line"] for record in records) or "no gate ruled on the proposal"
    rows = "; ".join(f"{record['name']}: {record['rows']!r}"
                     for record in records if record["accepted"] is False)
    atoms = _atoms_named(records)
    return RetryableToolError(
        f"proposal version {version} for task {task_id} refused, the previous file stands:\n"
        f"{lines}" + (f"\nrows behind the failures: {rows}" if rows else "")
        + (f"\n{atoms}" if atoms else "") + (_STOP if stop else ""),
        ask=("Read the rows behind the failures and propose again with atoms answering them, "
             "or file a finding naming the Environment file and what differs after two refusals."))


def atoms_hash(atoms: Iterable[Any]) -> str:
    """The content hash of a candidate's atoms, so a proposal refused once is known again (F32)."""
    body = json.dumps([as_dict(atom) for atom in atoms], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def _propose_verifier(root: ExamRoot):
    """One proposal as a transaction (D201): ruled on the write, kept only when every ruling passed.

    The candidate joins the Task's history as an accepted trial version before the ruling, so
    the trusted gate reads it as the current accepted version and the loosening gate compares
    it against the version before it (D127). A refused proposal keeps its row, rewritten as
    rejected with the gates that rejected it: the history is the record, never dropped (D201).
    """
    # Per Task in this session: each refused candidate's atoms hash, with its version and the gates
    # that refused it, and how many refusals the Task has had (F32).
    refused: dict[str, dict[str, tuple[str, list[str]]]] = {}
    refusals: dict[str, int] = {}

    async def propose_verifier(args: ProposeArgs) -> ProposeResult:
        current = _current(root, args.task_id)
        candidate = current.model_copy(deep=True, update={
            "atoms": _candidate_atoms(current, args.drop, args.add),
            "verifier_version": _next_version(current)})
        atoms_digest = atoms_hash(candidate.atoms)
        if atoms_digest == atoms_hash(current.atoms):
            # F37: a proposal that changes nothing is answered without the suite, and counts as a refusal.
            refusals[args.task_id] = refusals.get(args.task_id, 0) + 1
            raise RetryableToolError(
                f"the proposal changes nothing: version {current.verifier_version} already holds these "
                f"atoms; drop or add an atom, or move on to another Task"
                + (_STOP if refusals[args.task_id] > REFUSALS_PER_TASK else ""))
        seen = refused.setdefault(args.task_id, {}).get(atoms_digest)
        if seen is not None:
            refusals[args.task_id] = refusals.get(args.task_id, 0) + 1
            version, gates = seen
            raise RetryableToolError(
                f"already refused at version {version} for {', '.join(gates) or 'no gate'}: change the "
                f"atoms or move on to another Task"
                + (_STOP if refusals[args.task_id] > REFUSALS_PER_TASK else ""))
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
        records = _rule(root, args.task_id, candidate)
        # F45: a check that did not run is no refusal; only a ruling that ran and failed refuses.
        if records and all(record["accepted"] is not False for record in records):
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
        gates = [record["name"] for record in records if record["accepted"] is False]
        hist.versions[-1] = hist.versions[-1].model_copy(update={"accepted": False, "rejected_by": gates})
        root.history[args.task_id] = hist
        save_ruling_files(root)
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(previous)
        refused[args.task_id][atoms_digest] = (candidate.verifier_version, gates)
        refusals[args.task_id] = refusals.get(args.task_id, 0) + 1
        raise _refuse_proposal(args.task_id, candidate.verifier_version, records,
                               stop=refusals[args.task_id] > REFUSALS_PER_TASK)

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


def _reroll_row(report: Any) -> dict:
    """One re-rolled Run as the row the tool answers."""
    body = report.as_dict() if hasattr(report, "as_dict") else dict(report)
    return {"run_id": body.get("run_id"), "verdict": body.get("verdict"),
            "termination_reason": body.get("termination_reason"), "spend": body.get("spend") or {}}


def _ledger_usd(workdir: Any) -> float:
    """What budget.json says the build has spent, where every priced Run call lands (F29)."""
    return float(budget.load_totals(workdir)["total"].get("usd") or 0.0)


def _reroll(root: ExamRoot):
    from kullback.examiner.runners import _priced  # imported here: this closure is the only user

    async def reroll(args: RerollArgs) -> RerollResult:
        if root.allowance_remaining is not None and root.allowance_remaining <= 0:
            raise RuntimeError(f"the allowance is spent ({root.allowance_remaining:.2f} left)")
        if root.reroll_model is None:
            raise ValueError("no reroll model was given to this examination; "
                             "file a finding naming the Task instead")
        # Priced under the runner stage like every other Run, so the re-roll counts (F29).
        model = _priced(root.reroll_model, root.workdir)
        rows = []
        spent = 0.0
        stopped = False
        # One Run at a time, the allowance checked before each and deducted after each, so a
        # batch never spends past what is left.
        for seed in range(args.count):
            if root.allowance_remaining is not None and root.allowance_remaining <= 0:
                stopped = True
                break
            before = _ledger_usd(root.workdir)
            reports = runner_tool.reroll(root.workdir, args.task_id, model, count=1,
                                         workdir=root.workdir, first_seed=seed)
            rows.extend(_reroll_row(report) for report in reports or [])
            price = max(_ledger_usd(root.workdir) - before, 0.0)
            spent += price
            if root.allowance_remaining is not None:
                root.allowance_remaining -= price
        summary = (f"reroll of task {args.task_id}: {len(rows)} Runs, "
                   f"{sum(1 for r in rows if r['termination_reason'] == 'success')} finished, "
                   f"{spent:.4f} USD")
        if stopped:
            summary += f"; stopped after {len(rows)} of {args.count} Runs: the allowance is spent"
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


__all__ = ["REFUSALS_PER_TASK", "ROW_FIELDS", "SET_BY_TOOL", "FindingArgs", "FindingResult", "ProbeArgs", "ProbeResult", "ProposeArgs",
           "ProposeResult", "RerollArgs", "RerollResult", "atoms_hash", "domain_tools", "render", "tool_names"]
