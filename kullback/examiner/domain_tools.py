"""The Examiner's domain tools over its root: edit_verifier, probe, finding, reroll.

`edit_verifier` writes the next Verifier version off the current one (edit, then drop and
add; `propose_verifier` is its alias for one release); the D79 suite,
the pool and the loosening gate rule on the write through `gate_writes`, never here
(D79). `probe` scores a hand-written Run against the current Verifier and keeps it in
the pool. `finding` files what is wrong on the Builder's side with the file to edit
and what should differ, never a verb. `reroll` buys frontier Runs through the Runner
tool, refused at or below the remaining allowance (D112, D133). Every result carries
rows, never a count alone.

The Builder's notes (one per Task under the workdir's notes/, a reason from NOTE_REASONS and one
sentence) are read here too: an accepted edit, a probe, or a finding with `note_ruling` on the
Task rules the note, and the ruling is written next to it for the Builder's next round.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

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
    elide_agent_prose,
    make_atom,
    not_run_reason,
    validate_verifier,
)
from kullback.runner import budget
from kullback.runner import tool as runner_tool
from kullback.runner.records import (
    Atom,
    Event,
    GateResult,
    Run,
    Verifier,
    VerifierVersion,
    as_dict,
    read_json,
    write_json,
)
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
    edit: list[dict] = Field(default_factory=list, description="Atoms to change in place: each names "
                             "an existing atom `id` and only the fields that change; `payload` keys "
                             "replace the same keys of the atom's payload, and `kind` stays. Applied "
                             "before drop and add.")
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
    note_ruling: Optional[Literal["builder_right", "builder_wrong"]] = Field(
        default=None, description="Set to rule on the Builder's open note on this Task: whether the "
                                  "Builder is right, with the why in text.")


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
              "edit_verifier again with the same reason and that atom written as: {shape}")


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
    # D292: an atom the model wrote pins the recorded agent's prose no more than a derived one does.
    atom = elide_agent_prose(make_atom(row["id"], row["kind"], payload, helpers=HELPERS_SRC, **fields))
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


def _edited_atom(atom: Atom, row: dict) -> Atom:
    """One existing atom with the fields an `edit` row changes; its id and kind stay."""
    if row.get("kind") not in (None, atom.kind):
        raise RetryableToolError(
            f"the edit of atom {atom.id!r} changes its kind from {atom.kind} to {row['kind']}; an edit "
            "keeps the kind: drop the atom and add one of the new kind instead",
            ask="Edit again without kind, or drop and add the atom.")
    unknown = sorted(key for key in row if key not in SET_BY_TOOL and key not in ROW_FIELDS)
    if unknown:
        raise RetryableToolError(
            f"the edit of atom {atom.id!r} carries fields an atom does not have: {', '.join(unknown)}. "
            f"{ATOM_SHAPE}", ask="Edit again naming only the id and the fields that change.")
    changes = row.get("payload") or row.get("target") or {}
    if not isinstance(changes, dict):
        raise RetryableToolError(
            f"the edit of atom {atom.id!r} carries its payload as a {type(changes).__name__}, not an "
            f"object: {ATOM_SHAPE}", ask="Edit again with payload as an object of the keys that change.")
    fields = {name: getattr(atom, name) for name in ROW_FIELDS}
    fields.update({key: value for key, value in row.items() if key in ROW_FIELDS})
    return _atom_of({"id": atom.id, "kind": atom.kind, "payload": {**atom_payload(atom), **changes},
                     **({"predicate_src": row["predicate_src"]} if row.get("predicate_src") else {}),
                     **fields}, ())


def _edited_atoms(atoms: list, edit: Iterable[dict]) -> list:
    """The atoms with each `edit` row applied in place, an unknown atom id refused by name."""
    by_id = {atom.id: number for number, atom in enumerate(atoms)}
    out = list(atoms)
    for row in edit:
        atom_id = row.get("id") if isinstance(row, dict) else None
        if atom_id not in by_id:
            raise RetryableToolError(
                f"the edit names atom {atom_id!r}, which the current Verifier does not hold; its atoms "
                f"are {', '.join(sorted(by_id)) or 'none'}",
                ask="Edit again naming one of the current atom ids, or add the atom instead.")
        out[by_id[atom_id]] = _edited_atom(out[by_id[atom_id]], row)
    return out


def _candidate_atoms(current: Verifier, drop: Iterable[str], add: Iterable[Any],
                     edit: Iterable[dict] = ()) -> list:
    """The atoms of the next version: edits first, then drop and add (`_repair`, carried)."""
    dropped = set(drop)
    added = [row if hasattr(row, "kind") else _atom_of(row, current.atoms) for row in add]
    atoms = [a for a in _edited_atoms(list(current.atoms), edit) if a.id not in dropped] + added
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


def _ran_and_held(suite: list[dict]) -> tuple[dict, dict]:
    """The suite's checks that ran, with whether each passed, and those that did not, with why."""
    ran = {record["name"]: record["accepted"] for record in suite if record["accepted"] is not None}
    held = {record["name"]: record["not_run"] for record in suite if record["accepted"] is None}
    return ran, held


def _alt_path_waived(suite: list[dict], base: dict, held: dict) -> bool:
    """Whether the alt-path check is waived (D199): by its passed ruling's row, or by the Task's
    row when the check did not run."""
    return any(record["name"] == ALT_PATH_STAGE and record["accepted"]
               and any(WAIVED_ROW in row for row in record["rows"]) for record in suite) \
        or bool(base.get("second_path_waived") and ALT_PATH_STAGE in held)


def _suite_row(records: list[dict], base: dict) -> Optional[dict]:
    """The status row the D79 checks among these records give the Task, in the derivation's shape.

    A check that did not run is `not_run` with its reason and not passed, so the Task stays
    untrusted (F45). A waived alt-path check (D199), by its ruling's row or by the Task's row when
    the check did not run, is the check not passed and the Verifier passed.
    """
    suite = [record for record in records if record["name"] in D79_STAGES]
    if not suite:
        return None
    ran, held = _ran_and_held(suite)
    waived = _alt_path_waived(suite, base, held)
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
    _merge_bought(root, task_id)
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
    if _append_until_refused(out, results, candidate):
        return out
    gates = _write_gates(root, evidence, _suite_status(evidence, task_id, out))
    _append_until_refused(out, (run_gate() for run_gate in gates), candidate)
    return out


def _append_until_refused(out: list[dict], results: Iterable[GateResult], candidate: Verifier) -> bool:
    """Each gate result as a record onto `out`, in order, up to the first that ran and failed;
    whether one did. `results` is read lazily, so no gate after that refusal runs."""
    for result in results:
        out.append(_gate_record(result, candidate))
        if out[-1]["accepted"] is False:
            return True
    return False


def _suite_status(evidence: dict, task_id: str, out: list[dict]) -> dict:
    """The task status the trusted gate reads, with the Task's row carrying the suite just run."""
    base = dict(evidence.get("task_status") or {})
    base[task_id] = {**(base.get(task_id) or {}), **(_suite_row(out, base.get(task_id) or {}) or {})}
    return base


def _write_gates(root: ExamRoot, evidence: dict, base: dict) -> tuple[Callable[[], GateResult], ...]:
    """The gates after the D79 suite that rule on the write, in order, each run only when called."""
    runs, replays, rerolls = evidence.get("task_runs") or {}, evidence.get("replays") or {}, evidence.get("rerolls") or {}
    rules, sigs = evidence.get("rules"), root.sigs or []
    return (lambda: loosening_gate(evidence.get("history") or {}, runs, replays, rerolls, rules, sigs),
            lambda: false_rejection_gate(evidence["verifiers"], runs, replays, rerolls, rules, sigs,
                                         task_status=evidence.get("task_status")),
            lambda: trusted_gate(base, evidence["verifiers"], evidence.get("probes") or {},
                                 evidence.get("history") or {}, {}, runs, replays, rerolls, rules, sigs,
                                 workdir=root.workdir))


def _merge_bought(root: ExamRoot, task_id: str) -> None:
    """Every second path Run bought for the Task joins its re-roll rows before a ruling.

    The derivation keeps those rows in its own re-roll file, not the workdir's, so a ruling that
    read only the workdir's rows never saw a second path and left that check not run.
    """
    from kullback.examiner.plan import merged_rerolls
    from kullback.examiner.stage import second_path_rows

    bought = second_path_rows(Path(root.workdir), task_id)
    if bought:
        root.rerolls = merged_rerolls(root.rerolls or {}, {task_id: bought})


def _status_row(root: ExamRoot, task_id: str, records: list[dict], version: str) -> Optional[dict]:
    """The Task's status row after an accepted ruling: the suite it passed and the version it holds."""
    row = _suite_row(records, root.task_status.get(task_id) or {})
    return None if row is None else {**row, "verifier_version": version}


def _keep_suite_status(root: ExamRoot, task_id: str, row: Optional[dict]) -> None:
    """An accepted proposal's suite and version as the Task's status row, in both status files.

    The exposed exam/task_status.json the session reads, and the workdir's task_status.json the
    Builder's status reads, the rest of each row kept. A ruling with no suite leaves the rows as they were.
    """
    if row is None:
        return
    root.task_status[task_id] = {**(root.task_status.get(task_id) or {}), **row}
    write_json(root.exam_dir / "task_status.json", root.task_status)
    path = Path(root.workdir) / "task_status.json"
    status = read_json(path, {}) if path.is_file() else {}
    status = status if isinstance(status, dict) else {}
    status[task_id] = {**(status.get(task_id) or {}), **row}
    write_json(path, status)


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


def _needs_no_ruling(current: Verifier, atoms_digest: str,
                     refused_here: dict[str, tuple[str, list[str]]]) -> Optional[str]:
    """Why a candidate is answered without the suite, else None: its atoms are the current ones
    (F37), or the same atoms were refused before in this session (F32)."""
    if atoms_digest == atoms_hash(current.atoms):
        return (f"the proposal changes nothing: version {current.verifier_version} already holds these "
                f"atoms; edit, drop or add an atom, or move on to another Task")
    seen = refused_here.get(atoms_digest)
    if seen is None:
        return None
    version, gates = seen
    return (f"already refused at version {version} for {', '.join(gates) or 'no gate'}: change the "
            f"atoms or move on to another Task")


def _trial_history(root: ExamRoot, task_id: str, current: Verifier, candidate: Verifier, digest: str,
                   reason: str) -> Any:
    """The Task's history with the candidate joined as an accepted trial version (D127), kept on the root."""
    hist = seeded_history(root.history, task_id, current)
    hist.versions.append(VerifierVersion(
        task_id=task_id, content_hash=digest, verifier_version=str(len(hist.versions) + 1),
        parent_hash=next((v.content_hash for v in reversed(hist.versions) if v.accepted), None),
        by="repair", reason=reason, accepted=True, verifier=candidate))
    root.history[task_id] = hist
    return hist


def _status_snapshot(root: ExamRoot, task_id: str) -> tuple[tuple[Path, ...], list[Optional[bytes]], Optional[dict]]:
    """Both status files' bytes and the Task's in-memory status row, before a transaction."""
    status_files = (root.exam_dir / "task_status.json", Path(root.workdir) / "task_status.json")
    previous_status = [path.read_bytes() if path.is_file() else None for path in status_files]
    return status_files, previous_status, root.task_status.get(task_id)


def _keep_status_or_restore(root: ExamRoot, task_id: str, records: list[dict], version: str,
                            before: tuple[tuple[Path, ...], list[Optional[bytes]], Optional[dict]],
                            ) -> tuple[list[dict], bool]:
    """The accepted suite and version written as the Task's status rows, and whether they were.

    A write that fails puts both status files and the row back as `before` held them and adds a
    failed status_write record, which refuses the proposal.
    """
    try:
        _keep_suite_status(root, task_id, _status_row(root, task_id, records, version))
    except (OSError, ValueError, TypeError):
        status_files, previous_status, previous_row = before
        for path, body in zip(status_files, previous_status, strict=True):
            _restore(path, body)
        if previous_row is None:
            root.task_status.pop(task_id, None)
        else:
            root.task_status[task_id] = previous_row
        return records + [{"name": "status_write", "accepted": False,
                           "line": "status_write failed: the status rows could not be written",
                           "rows": []}], False
    return records, True


def _accepted_result(root: ExamRoot, args: ProposeArgs, candidate: Verifier, digest: str,
                     records: list[dict], alias: bool) -> ProposeResult:
    """An accepted proposal's result; the Builder's open note on the Task is ruled agrees."""
    summary = (f"proposal version {candidate.verifier_version} for task {args.task_id} "
               f"({digest[:12]}) accepted: " + "; ".join(record["line"] for record in records))
    ruled = rule_note(root.workdir, args.task_id, "edit_verifier", "agrees",
                      f"version {candidate.verifier_version} accepted: {args.reason}")
    if ruled is not None:
        summary += f"; the Builder's note on task {args.task_id} is ruled: agrees"
    if alias:
        summary += f"; {DEPRECATED_ALIAS}"
    return ProposeResult(summary=summary, task_id=args.task_id, content_hash=digest,
                         verifier_version=candidate.verifier_version,
                         atoms=len(candidate.atoms), rulings=records)


def _roll_back(root: ExamRoot, task_id: str, previous_verifier: Optional[Verifier], hist: Any,
               records: list[dict]) -> list[str]:
    """A refused proposal undone in memory and in the ruling files: the previous Verifier back,
    the trial version kept as rejected by the gates that refused it (D201); those gates."""
    if previous_verifier is None:
        root.verifiers.pop(task_id, None)
    else:
        root.verifiers[task_id] = previous_verifier
    gates = [record["name"] for record in records if record["accepted"] is False]
    hist.versions[-1] = hist.versions[-1].model_copy(update={"accepted": False, "rejected_by": gates})
    root.history[task_id] = hist
    save_ruling_files(root)
    return gates


def _propose_verifier(root: ExamRoot, alias: bool = False):
    """One proposal as a transaction (D201): ruled on the write, kept only when every ruling passed.

    The candidate joins the Task's history as an accepted trial version before the ruling, so
    the trusted gate reads it as the current accepted version and the loosening gate compares
    it against the version before it (D127). A refused proposal keeps its row, rewritten as
    rejected with the gates that rejected it: the history is the record, never dropped (D201).
    An accepted one writes the Task's status rows in the same transaction as the Verifier file:
    a status write that fails restores the file and both status files. `alias` marks the old
    name, whose result carries one deprecation line.
    """
    # Per Task in this session: each refused candidate's atoms hash, with its version and the gates
    # that refused it, and how many refusals the Task has had (F32).
    refused: dict[str, dict[str, tuple[str, list[str]]]] = {}
    refusals: dict[str, int] = {}

    async def edit_verifier(args: ProposeArgs) -> ProposeResult:
        current = _current(root, known_task(root.workdir, args.task_id))
        candidate = current.model_copy(deep=True, update={
            "atoms": _candidate_atoms(current, args.drop, args.add, args.edit),
            "verifier_version": _next_version(current)})
        atoms_digest = atoms_hash(candidate.atoms)
        why = _needs_no_ruling(current, atoms_digest, refused.setdefault(args.task_id, {}))
        if why is not None:
            # F37, F32: answered without the suite, and counted as a refusal.
            refusals[args.task_id] = refusals.get(args.task_id, 0) + 1
            raise RetryableToolError(why + (_STOP if refusals[args.task_id] > REFUSALS_PER_TASK else ""))
        digest = version_hash(candidate)
        hist = _trial_history(root, args.task_id, current, candidate, digest, args.reason)
        target = task_path(root.workdir, args.task_id, root.exam_dir / "verifiers")
        previous = target.read_bytes() if target.is_file() else None
        previous_verifier = root.verifiers.get(args.task_id)
        before = _status_snapshot(root, args.task_id)
        root.verifiers[args.task_id] = candidate
        write_json(target, as_dict(candidate))
        save_ruling_files(root)
        records = _rule(root, args.task_id, candidate)
        # F45: a check that did not run is no refusal; only a ruling that ran and failed refuses.
        accepted = bool(records) and all(record["accepted"] is not False for record in records)
        if accepted:
            records, accepted = _keep_status_or_restore(root, args.task_id, records,
                                                        candidate.verifier_version, before)
        if accepted:
            return _accepted_result(root, args, candidate, digest, records, alias)
        gates = _roll_back(root, args.task_id, previous_verifier, hist, records)
        _restore(target, previous)
        refused[args.task_id][atoms_digest] = (candidate.verifier_version, gates)
        refusals[args.task_id] = refusals.get(args.task_id, 0) + 1
        raise _refuse_proposal(args.task_id, candidate.verifier_version, records,
                               stop=refusals[args.task_id] > REFUSALS_PER_TASK)

    return edit_verifier


def _restore(path: Path, body: Optional[bytes]) -> None:
    """A file back to the bytes it held before a transaction, or gone when it held none."""
    if body is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(body)


# The line the old tool name's result carries for the one release it stays (Finding B).
DEPRECATED_ALIAS = "propose_verifier is deprecated and goes after this release: call edit_verifier"


# --- the Builder's notes: one per Task, ruled by the Examiner (D280) ---

#: Where the Builder's notes live under the workdir, one file per Task, the ruling next to it.
NOTES_DIR = "notes"
RULING_SUFFIX = ".ruling.json"
#: Why a Task is not verifiable as written: the only reasons a note may give.
NOTE_REASONS = ("outcome_not_in_state", "intent_contradicts_reference", "fact_unavailable_to_user",
                "needs_action_record")
NoteReason = Literal["outcome_not_in_state", "intent_contradicts_reference", "fact_unavailable_to_user",
                     "needs_action_record"]
NOTE_SENTENCE_CHARS = 300
# What a note's sentence may not carry: structure or check source, which is an atom or Verifier text.
_NOTE_FORBIDDEN = ("{", "}", "[", "]", "predicate", "atom", "def ", "lambda", "return ")


def task_ids_of(workdir: Any) -> list[str]:
    """The workdir's Task ids, the stems of tasks/<id>.json: the one set a model-named id is checked against."""
    return [path.stem for path in sorted((Path(workdir) / "tasks").glob("*.json")) if path.name != "tasks.json"]


def known_task(workdir: Any, task_id: Any) -> str:
    """The id when it is one of the workdir's Tasks, else a ValueError naming it unknown.

    Membership, never a path check: an id with a separator, `..` or an absolute path is not a Task.
    """
    if not isinstance(task_id, str) or task_id not in task_ids_of(workdir):
        raise ValueError(f"unknown Task {task_id!r}: it is not one of the Tasks under tasks/")
    return task_id


def task_path(workdir: Any, task_id: Any, folder: Any, suffix: str = ".json") -> Path:
    """Where a Task's file lives under `folder`, resolved only for a known Task id."""
    return Path(folder) / f"{known_task(workdir, task_id)}{suffix}"


def notes_dir(workdir: Any) -> Path:
    return Path(workdir) / NOTES_DIR


def note_path(workdir: Any, task_id: str) -> Path:
    return notes_dir(workdir) / f"{task_id}.json"


def ruling_path(workdir: Any, task_id: str) -> Path:
    return notes_dir(workdir) / f"{task_id}{RULING_SUFFIX}"


def note_refusal(reason: Any, sentence: Any) -> Optional[str]:
    """Why a note is refused, or None: a reason off the fixed list and one plain sentence only."""
    if reason not in NOTE_REASONS:
        return f"the reason {reason!r} is not one of {', '.join(NOTE_REASONS)}"
    text = str(sentence or "").strip()
    if not text:
        return "the note carries no sentence"
    if "\n" in text or len(text) > NOTE_SENTENCE_CHARS:
        return f"the note is one sentence on one line, at most {NOTE_SENTENCE_CHARS} characters"
    lowered = text.lower()
    carried = [mark for mark in _NOTE_FORBIDDEN if mark in lowered]
    if carried:
        return (f"the note carries {', '.join(repr(mark) for mark in carried)}: a note says why the Task "
                "is not verifiable as written, never an atom or Verifier text")
    return None


def note_hash(note: dict) -> str:
    body = json.dumps({key: note.get(key) for key in ("task_id", "reason", "sentence")}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def write_note(workdir: Any, task_id: str, reason: str, sentence: str) -> dict:
    """The Builder's note on one Task, written over any earlier note; refused as a ValueError."""
    why = note_refusal(reason, sentence)
    if why is not None:
        raise ValueError(why)
    note = {"task_id": task_id, "reason": reason, "sentence": str(sentence).strip()}
    note["note_hash"] = note_hash(note)
    write_json(task_path(workdir, task_id, notes_dir(workdir)), note)
    return note


def _read_dict(path: Path) -> Optional[dict]:
    try:
        body = read_json(path, None)
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def read_note(workdir: Any, task_id: str) -> Optional[dict]:
    return _read_dict(note_path(workdir, task_id))


def read_ruling(workdir: Any, task_id: str) -> Optional[dict]:
    """The Examiner's ruling on the Task's note as it stands now; a ruling on an older note is none."""
    note, ruling = read_note(workdir, task_id), _read_dict(ruling_path(workdir, task_id))
    if note is None or ruling is None or ruling.get("note_hash") != note.get("note_hash"):
        return None
    return ruling


def notes_of(workdir: Any) -> dict[str, tuple[dict, Optional[dict]]]:
    """Every note by Task with its ruling, None while the note is open."""
    folder = notes_dir(workdir)
    out: dict[str, tuple[dict, Optional[dict]]] = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name.endswith(RULING_SUFFIX):
            continue
        note = _read_dict(path)
        if note is not None:
            out[path.stem] = (note, read_ruling(workdir, path.stem))
    return out


def open_note(workdir: Any, task_id: str) -> Optional[dict]:
    """The Task's note while no ruling answers it, else None."""
    note = read_note(workdir, task_id)
    return note if note is not None and read_ruling(workdir, task_id) is None else None


def rule_note(workdir: Any, task_id: Optional[str], move: str, verdict: str, text: str) -> Optional[dict]:
    """Rule on the Task's open note with one of the Examiner's moves, written next to the note.

    Returns the ruling, or None when the Task has no open note.
    """
    note = open_note(workdir, task_id) if task_id else None
    if note is None:
        return None
    ruling = {"task_id": task_id, "note_hash": note["note_hash"], "move": move, "verdict": verdict,
              "text": text}
    write_json(task_path(workdir, task_id, notes_dir(workdir), RULING_SUFFIX), ruling)
    return ruling


def note_line(task_id: str, note: dict, ruling: Optional[dict]) -> str:
    """One note as a line of an opening message: the reason, the sentence, and its ruling or open."""
    head = f"{task_id}: {note.get('reason')}: {note.get('sentence')}"
    if ruling is None:
        return f"{head} (open)"
    return f"{head} (ruled by {ruling.get('move')}, {ruling.get('verdict')}: {ruling.get('text')})"


def _probe(root: ExamRoot):
    async def probe(args: ProbeArgs) -> ProbeResult:
        verifier = _current(root, known_task(root.workdir, args.task_id))
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
        if rule_note(root.workdir, args.task_id, "probe", "builder_right" if passed else "builder_wrong",
                     f"{verdict}" + (f"; {args.note}" if args.note else "")) is not None:
            summary += f"; the Builder's note on task {args.task_id} is ruled by this probe"
        return ProbeResult(summary=summary, probe_id=probe_id, task_id=args.task_id,
                           scored_pass=bool(passed),
                           failing_atom=failing if not passed else None, pool_size=len(pool))

    return probe


def _on_task(task_id: Optional[str]) -> str:
    """The ` on task <id>` clause a finding's message carries, empty when it names no Task."""
    return f" on task {task_id}" if task_id else ""


def _check_finding(root: ExamRoot, args: FindingArgs) -> str:
    """The finding's kind once its kind, Task and note ruling hold; raises naming what does not."""
    kind = (args.kind or "").strip()
    if kind not in FINDING_KINDS:
        raise ValueError(f"{args.kind!r} is not a finding kind. The kinds are: "
                         f"{', '.join(FINDING_KINDS)}.")
    if args.task_id is not None:
        known_task(root.workdir, args.task_id)
    if args.note_ruling is not None and open_note(root.workdir, args.task_id or "") is None:
        raise ValueError(f"task {args.task_id} has no open note from the Builder to rule on; "
                         "file the finding without note_ruling")
    return kind


def _refuse_repeat(root: ExamRoot, record: Finding, args: FindingArgs) -> None:
    """Raise when an open finding already says what `record` says."""
    key = finding_key(record.kind, record.path or record.change, args.task_id or "")
    existing = open_by_key(root.findings).get(key)
    if existing is not None:
        raise ValueError(f"{existing} already says this ({record.kind}"
                         + (f" in {args.path}" if args.path else "")
                         + _on_task(args.task_id)
                         + "); it is filed and the Builder has not answered it yet. Act on it, "
                           "or file a finding that says something else.")


def _finding(root: ExamRoot):
    async def finding(args: FindingArgs) -> FindingResult:
        kind = _check_finding(root, args)
        record = Finding(task_id=args.task_id, kind=kind, text=args.text, rows=list(args.rows),
                         path=args.path, change=args.change, source="model")
        _refuse_repeat(root, record, args)
        root.findings.append(record)
        body = record.as_dict()
        if root.bus is not None:
            root.bus.publish_custom("finding", body)
        summary = (f"finding {body['finding_id']} ({kind}) filed" + _on_task(args.task_id)
                   + (f", naming {args.path}" if args.path else ""))
        if args.note_ruling is not None:
            rule_note(root.workdir, args.task_id, "finding", args.note_ruling, args.text)
            summary += f"; the Builder's note on task {args.task_id} is ruled: {args.note_ruling}"
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
        known_task(root.workdir, args.task_id)
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
            # The user the derivation's re-rolls meet, so the Run is answered after the instruction.
            extra = ({"make_user": root.reroll_user(args.task_id)}
                     if root.reroll_user is not None else {})
            reports = runner_tool.reroll(root.workdir, args.task_id, model, count=1,
                                         workdir=root.workdir, first_seed=seed, **extra)
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
    """The four domain tools over one ExamRoot, and propose_verifier as edit_verifier's alias for one release."""
    return [
        AgentTool("edit_verifier", "Write a new Verifier version: edit atoms in place (a change to an "
                  "existing atom), then drop and add; the D79 suite, the pool and the loosening gate "
                  "rule on the write.",
                  ProposeArgs, ProposeResult, _propose_verifier(root), render=render),
        AgentTool("probe", "Score a hand-written Run against a Task's current Verifier and keep it "
                  "in the Task's pool.",
                  ProbeArgs, ProbeResult, _probe(root), render=render),
        AgentTool("finding", "File what is wrong on the Builder's side: the Environment file to edit "
                  "in `path` and the one line saying what should differ in `change`, never a verb.",
                  FindingArgs, FindingResult, _finding(root), render=render),
        AgentTool("reroll", "Buy more frontier Runs of a Task through the Runner (D112, D133).",
                  RerollArgs, RerollResult, _reroll(root), render=render),
        AgentTool("propose_verifier", "Deprecated alias of edit_verifier, kept for one release.",
                  ProposeArgs, ProposeResult, _propose_verifier(root, alias=True), render=render),
    ]


def tool_names() -> tuple[str, ...]:
    """The domain tool names, so tests pin the Builder-called surface."""
    return ("edit_verifier", "probe", "finding", "reroll", "propose_verifier")


__all__ = ["DEPRECATED_ALIAS", "NOTES_DIR", "NOTE_REASONS", "note_line", "note_refusal", "notes_of",
           "known_task", "open_note", "read_note", "read_ruling", "rule_note", "ruling_path", "task_ids_of", "task_path", "write_note", "REFUSALS_PER_TASK", "ROW_FIELDS", "SET_BY_TOOL", "FindingArgs", "FindingResult", "ProbeArgs", "ProbeResult", "ProposeArgs",
           "ProposeResult", "RerollArgs", "RerollResult", "atoms_hash", "domain_tools", "render", "tool_names"]
