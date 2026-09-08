"""The harness loosening an over-strict Verifier by itself, code only, gated as any repair (D205).

D133 measures how many held-out Runs a Verifier's required atoms wrongly reject, D185 cut the pool
down to the Runs that reached the Reference and D194 made the number a step in the trusted ruling.
Three readers, no actor: the only thing that could relax a Verifier was the Examiner's `repair`, a
verb the model has to choose, and over one live build it never chose it once while 24 false-rejection
findings closed over 23 Tasks with the fraction at 1.0 in every round.

An atom that rejects a Run which reached the Reference's End state, while the End state it guards is
satisfied, is over-specific by definition: the Task's own answer was reached and the atom turned the
Run away for how it got there or for a literal the End state does not require. That is a class and
not a Task, so the relaxation is written by kind and never by name:

  a path atom is dropped, because it pins the order the recording happened to take (a question the
  Reference asked and this Run did not have to);
  a read atom is dropped, because an atom over a tool that writes nothing asks the Candidate to
  repeat a lookup rather than to reach a state;
  a write atom naming one row is relaxed to the End-state predicate it implies, the shape the row's
  own column keeps (D190's shape atom) for a value, and a wrote-any over the same tool for a row;
  an answer atom keeps its shape check and loses its literal, which is D190's reported fact: same
  value, same predicate, and no Run rejected for it.

Nothing here decides whether the relaxed Verifier stands. The proposal goes through the very path
the `repair` tool uses, so the D79 suite, the probe pool and the loosening gate rule on it (D127,
D133), and a proposal they reject leaves the Task exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from kullback.examiner.derive import REPORTED_COMMUNICATE, shape_atom
from kullback.gates.loosening import discarded_runs, false_rejection, legitimate_runs, over_strict
from kullback.gates.probes import write_tools_of
from kullback.gates.verifier_suite import atom_payload, check_run, make_atom, names_no_row
from kullback.runner.records import Atom, Run, Verifier

# How many atoms one automatic proposal may relax. Two is the whole of what one version is allowed to
# change: a version that rewrites more than that is no longer a reading of one over-specific atom, it
# is a new Verifier, and the D79 suite would be ruling on something no evidence produced.
MAX_RELAXATIONS = 2
# How many rounds a Task may be loosened automatically before the harness stops and leaves it to the
# Examiner. Two rounds of at most two relaxations is the stopping rule D133 borrows from the
# hacker-fixer paper: a Task still over-strict after that is not one more rewrite away, and the
# finding names the atoms so a model can read what code could not relax.
MAX_ROUNDS = 2
# What the reason on a rejected proposal says, so a reader can tell a Task the gates turned down from
# a Task nothing was proposed for.
REJECTED = "auto_loosen_rejected"
# The relaxation each atom gets, by the word the counts report it under. `path` and `read` are drops;
# `write` and `write_value` are relaxations to an End-state predicate; `answer` keeps the predicate
# and stops rejecting on it.
KINDS = ("path", "read", "write", "write_value", "answer")


@dataclass
class Relaxation:
    """One atom relaxed: what it was, what replaces it (nothing, for a drop), and the word for it."""
    atom_id: str
    kind: str
    replacement: Optional[Atom] = None


@dataclass
class Proposal:
    """One Task's automatic proposal: the atom ids to drop, the atoms to add, and why."""
    task_id: str
    run_id: str
    relaxations: list[Relaxation] = field(default_factory=list)

    @property
    def drop(self) -> list[str]:
        return [r.atom_id for r in self.relaxations]

    @property
    def add(self) -> list[Atom]:
        return [r.replacement for r in self.relaxations if r.replacement is not None]

    @property
    def kinds(self) -> list[str]:
        return [r.kind for r in self.relaxations]

    @property
    def reason(self) -> str:
        named = ", ".join(f"{r.atom_id} ({r.kind})" for r in self.relaxations)
        return (f"the required atoms reject Run {self.run_id}, which reached the Reference End state, "
                f"so they are over-specific: {named}")


def required_atoms(verifier: Verifier) -> Verifier:
    """The Verifier without its Hard atoms, which is what the false-rejection number is measured over:
    a Run that broke the policy is rightly rejected and is not evidence of over-strictness."""
    return verifier.model_copy(update={"atoms": [a for a in verifier.atoms if a.kind != "hard"]})


def relax(atom: Atom, write_tools: Iterable[str]) -> Optional[Relaxation]:
    """How this atom is relaxed, or None when no rule here can relax it.

    The dispatch is on the atom's payload kind and never on a tool, a column or a value, so the same
    rule reads an atom of any Environment. An atom nothing here can relax (a write cap, a forbidden
    write, a Hard constraint) stops the proposal rather than being guessed at: the Examiner is handed
    the finding instead, which is what the model is for.
    """
    payload = atom_payload(atom)
    kind = payload.get("kind")
    tool = payload.get("tool")
    if kind in ("write", "write_value") and tool and tool not in set(write_tools or ()):
        # An atom over a tool that writes nothing demands a lookup, not an End state: the Candidate
        # that reached the same End state by reading elsewhere did the Task.
        return Relaxation(atom.id, "read")
    if kind == "question":
        return Relaxation(atom.id, "path")
    if kind == "write_value":
        if not (tool and payload.get("field") and payload.get("id_field")):
            return None
        return Relaxation(atom.id, "write_value",
                          shape_atom(atom.id, tool, payload["field"], payload["id_field"], write_tools))
    if kind == "write":
        if names_no_row(payload):
            return None  # already a wrote-any; there is nothing left to take off it
        # `at` stays: it points at the call the atom was read from, which is what the provenance
        # check and the synthesized wrong and unfinished Runs use, and no scorer reads it as a demand.
        relaxed = {key: value for key, value in payload.items()
                   if key not in ("entity", "entity_raw", "id_field")}
        return Relaxation(atom.id, "write",
                          make_atom(atom.id, atom.kind, relaxed, description=atom.description,
                                    provenance=atom.provenance, spans=list(atom.spans)))
    if kind == "communicate":
        # D190's reported fact: the same value and the same predicate, so the Verdict and the report
        # still say whether the Run stated it, and the kind is no longer one a scorer reads as a
        # demand, so no Run is rejected for the literal.
        reported = make_atom(atom.id, "allowed", dict(payload), description=atom.description,
                             spans=list(atom.spans))
        return Relaxation(atom.id, "answer",
                          reported.model_copy(update={"target": dict(payload, kind=REPORTED_COMMUNICATE)}))
    return None


def apply(verifier: Verifier, relaxations: Iterable[Relaxation]) -> Verifier:
    """The Verifier with these relaxations in it, each atom replaced where it stood."""
    rows = {r.atom_id: r for r in relaxations}
    atoms: list[Atom] = []
    for atom in verifier.atoms:
        row = rows.get(atom.id)
        if row is None:
            atoms.append(atom)
        elif row.replacement is not None:
            atoms.append(row.replacement)
    return verifier.model_copy(deep=True, update={"atoms": atoms})


def proposal_for(verifier: Verifier, run: Run, canon_rules: Any, write_tools: Iterable[str],
                 limit: int = MAX_RELAXATIONS) -> Optional[Proposal]:
    """The relaxations that let this held-out Run through, or None when there are none to make.

    The rejecting atoms are found one at a time rather than all at once: the scorer names the first
    atom that turns the Run away, and which atom comes second depends on what the first relaxation
    left, so relaxing them together would be relaxing atoms that no longer reject anything. The walk
    stops at `limit` whether or not the Run passes by then, and a Task that needs more comes back
    next round under the round cap; a walk that relaxes nothing is no proposal at all.
    """
    working = verifier
    made: list[Relaxation] = []
    for _ in range(limit):
        passed, failing = check_run(required_atoms(working), run, canon_rules, write_tools=write_tools)
        if passed:
            break
        atom = next((a for a in working.atoms if a.id == failing), None)
        if atom is None:
            # The Run was turned away by a write no atom covers (`extra_write:<tool>`), which asks for
            # an atom to be added and not for one to be relaxed. That is the Examiner's repair.
            break
        relaxation = relax(atom, write_tools)
        if relaxation is None:
            break
        made.append(relaxation)
        working = apply(working, [relaxation])
    if not made:
        return None
    return Proposal(task_id=verifier.task_id, run_id=run.run_id, relaxations=made)


# --- which Tasks are over-strict, and how often one has been loosened already ----------------

def over_strict_rows(store: dict) -> dict[str, dict]:
    """The false-rejection row of every Task the gate rules over_strict, read over the same pool the
    trusted ruling and the finding read (D185, D194): the held-out Runs that reached the Reference."""
    write_tools = write_tools_of(store.get("sigs") or [])
    legitimate = legitimate_runs(store.get("replays") or {}, store.get("rerolls") or {},
                                 discarded_runs(store.get("task_status") or {}))
    canon_rules = store.get("canon_rules")
    task_runs = store.get("task_runs") or {}
    rows: dict[str, dict] = {}
    for verifier in sorted(store.get("verifiers") or [], key=lambda v: v.task_id):
        row = false_rejection(verifier, task_runs.get(verifier.task_id, []),
                              legitimate.get(verifier.task_id, set()), canon_rules, write_tools)
        if over_strict(row):
            rows[verifier.task_id] = row
    return rows


def attempts(rows: Iterable[dict], task_id: str) -> list[dict]:
    """Every automatic proposal made for one Task, in the order they were made."""
    return [row for row in rows or () if isinstance(row, dict) and row.get("task_id") == task_id]


def may_propose(rows: Iterable[dict], task_id: str, round_number: int) -> bool:
    """Is this Task still open to an automatic proposal? At most one per round and at most
    `MAX_ROUNDS` rounds, after which the finding is the Examiner's to answer."""
    made = attempts(rows, task_id)
    if any(row.get("round") == round_number for row in made):
        return False
    return len({row.get("round") for row in made}) < MAX_ROUNDS


def round_counts(rows: Iterable[dict], round_number: int) -> dict:
    """What the automatic loosening did this round, in the four counts D205 asks to be read."""
    made = [row for row in rows or () if isinstance(row, dict) and row.get("round") == round_number]
    by_kind = {kind: 0 for kind in KINDS}
    for row in made:
        for kind in row.get("kinds") or ():
            by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
    return {"auto_loosen_proposed": len(made),
            "auto_loosen_accepted": sum(1 for row in made if row.get("accepted")),
            "auto_loosen_rejected": sum(1 for row in made if not row.get("accepted")),
            "relaxations_by_kind": by_kind}
