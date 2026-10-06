"""The Spec answers a review: a finding's atoms and text edits applied by code, then the gates (D320).

The Examiner files one review per Task with the edit to apply (D317); it writes nothing. Its atoms
and text edits come here, never through the Builder. An atoms edit drops and adds atoms on the
Task's Verifier; it is kept on the Spec (`atom_edits`) so a later rewrite keeps it. An added atom is
a fact or a question only: a write is a cell of the expected end state, never an atom (D320). A text
edit replaces its verbatim `where` once in an Intent or Task file of the workdir, and in the Spec's
Intent facts where they hold it, so the checks re-ground against the new words. Nothing here calls a
model: the expected end states are derived again by code from the same Reference.

Each call is one round, counted on the Spec (`round`). The Verifier is rewritten and the trust gates
rule on it (`spec/trust.py`, the grounded, can-fail and Reference-replay gates); a Task still failing
at the ceiling is set aside as pending, never refused by the Examiner alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from kullback.runner.canon import load_rules
from kullback.runner.records import read_json
from kullback.spec.router import ROUNDS_CAP
from kullback.spec.schema import Spec, intent_text, load_spec, save_spec
from kullback.spec.writer import WRITE_PREDICATE, load_inputs, with_gaps, write_verifier

PENDING = "pending"
SPEC_EDITS = ("atoms", "text")
TEXT_DIRS = ("intents", "tasks")
# The atom payloads a review may add: facts told and questions asked, nothing a write decides.
ADDABLE = ("communicate", "question")
# The gates a repaired Verifier must pass: the ones no fresh Run is needed for.
REPAIR_GATES = ("grounded", "can_fail", "reference_replay")


def task_of_edit(edit: dict, finding_task: Any = None) -> str:
    """The Task an edit is about: its own task_id, else the finding's, else the file it names."""
    path = str(edit.get("path") or "")
    return str(edit.get("task_id") or finding_task or (Path(path).stem if path else ""))


def _check_add(atom: Any) -> str:
    """Why an added atom is refused, or empty when it stands."""
    if not isinstance(atom, dict) or not atom.get("id") or not isinstance(atom.get("payload"), dict):
        return "an added atom names id, kind and payload"
    if atom.get("predicate_src") or atom.get("predicate"):
        return WRITE_PREDICATE
    if atom["payload"].get("kind") not in ADDABLE:
        return f"{WRITE_PREDICATE}: an added atom is one of {', '.join(ADDABLE)}"
    return ""


def _text_edit(workdir: Path, spec: Spec, edit: dict) -> tuple[Spec, str]:
    """The text edit applied once to its file and to the Intent facts holding it; or why it was refused."""
    path, where, replace = str(edit.get("path") or ""), str(edit.get("where") or ""), str(edit.get("replace") or "")
    parts = path.split("/")
    if parts[0] not in TEXT_DIRS or ".." in parts or not where:
        return spec, f"a text edit names a file under {' or '.join(TEXT_DIRS)}/ and where"
    target = Path(workdir) / path
    try:
        body = target.read_text(encoding="utf-8")
    except OSError:
        return spec, f"{path} does not exist"
    if body.count(where) != 1:
        return spec, f"found {body.count(where)} times in {path}; a text edit matches once"
    target.write_text(body.replace(where, replace), encoding="utf-8")
    facts = [fact.model_copy(update={"text": fact.text.replace(where, replace)}) for fact in spec.intent.facts]
    intent = spec.intent.model_copy(update={"facts": facts, "text": intent_text(facts)})
    return spec.model_copy(update={"intent": intent}), ""


def _atoms_edit(spec: Spec, edit: dict) -> tuple[Spec, str]:
    refused = next((why for why in map(_check_add, edit.get("add") or []) if why), "")
    if refused:
        return spec, refused
    if not (edit.get("drop") or edit.get("add")):
        return spec, "an atoms edit drops or adds at least one atom"
    kept = {"drop": [str(a) for a in edit.get("drop") or []], "add": list(edit.get("add") or []),
            "why": str(edit.get("why") or "")}
    return spec.model_copy(update={"atom_edits": [*spec.atom_edits, kept]}), ""


def gate_row(workdir: Path, spec: Spec) -> dict:
    """The trust gates on the Task's written Verifier, the same code `kullback report` runs."""
    from kullback.gates.probes import write_tools_of
    from kullback.spec.actions import workdir_action_tools
    from kullback.spec.trust import faithful_references, load_spec_verifier, runs_on_disk, tier_of_task
    from kullback.spec.writer_tools import policy_sections_of

    root = Path(workdir)
    verifier = load_spec_verifier(root, spec.task_id)
    if verifier is None:
        return {"tier": "untrusted", "gates": {}, "failing": "no verifier file"}
    sigs = read_json(root / "tool_sigs.json", None) or []
    tools = write_tools_of(sigs.get("sigs", []) if isinstance(sigs, dict) else sigs) - workdir_action_tools(root)
    tier = tier_of_task(spec, verifier, runs_on_disk(root).get(spec.task_id, []),
                        canon=load_rules(root / "canon-rules.json"), write_tools=tools,
                        policy_sections=policy_sections_of(root),
                        references=faithful_references(root).get(spec.task_id, ()))
    return tier.row


def gates_pass(row: dict) -> bool:
    return all((row.get("gates") or {}).get(name) for name in REPAIR_GATES)


def apply_review(workdir: Any, task_id: str, edits: Iterable[dict], *, ceiling: int = ROUNDS_CAP) -> dict:
    """One review round on one Task: the edits applied, the Verifier rewritten, the gates' answer.

    Returns the counts (applied, refused with why), the gate row and whether the Task is now pending.
    A Task without a Spec, or already set aside, is left as it is.
    """
    root = Path(workdir)
    spec = load_spec(root, task_id)
    if spec is None or spec.set_aside:
        return {"task_id": task_id, "applied": 0, "refused": [], "skipped": "set aside" if spec else "no Spec"}
    applied, refused = 0, []
    for edit in edits:
        if edit.get("kind") == "atoms":
            spec, why = _atoms_edit(spec, edit)
        elif edit.get("kind") == "text":
            spec, why = _text_edit(root, spec, edit)
        else:
            continue
        applied += not why
        refused += [why] if why else []
    inputs = load_inputs(root, task_id, spec.intent)
    spec = with_gaps(spec, inputs).model_copy(update={"version": spec.version + 1, "round": spec.round + 1})
    write_verifier(spec, root, inputs)
    spec = load_spec(root, task_id)
    row = gate_row(root, spec)
    passed = gates_pass(row)
    if not passed and spec.round >= ceiling:
        spec = spec.model_copy(update={"set_aside": PENDING})
        save_spec(root, spec)
    return {"task_id": task_id, "applied": applied, "refused": refused, "round": spec.round, "passed": passed,
            "failing": row.get("failing"), "pending": spec.set_aside == PENDING}


def route_findings(workdir: Any, findings: list[dict]) -> tuple[list[dict], list[dict]]:
    """The findings with their atoms and text edits taken out and applied by the Spec, and each Task's round.

    A finding left with no edit and none to begin with is kept as it was; one whose every edit went
    to the Spec is answered and dropped from what the Builder reads.
    """
    by_task: dict[str, list[dict]] = {}
    kept: list[dict] = []
    for finding in findings:
        edits = [e for e in finding.get("edits") or [] if isinstance(e, dict)]
        spec_edits = [e for e in edits if e.get("kind") in SPEC_EDITS]
        for edit in spec_edits:
            by_task.setdefault(task_of_edit(edit, finding.get("task_id")), []).append(edit)
        if spec_edits and len(spec_edits) == len(edits):
            continue
        kept.append(dict(finding, edits=[e for e in edits if e.get("kind") not in SPEC_EDITS]))
    rounds = [apply_review(workdir, task_id, edits) for task_id, edits in sorted(by_task.items()) if task_id]
    return kept, rounds


def rounds_line(rounds: list[dict]) -> str:
    """The one line the Builder reads about the Spec's repairs."""
    if not rounds:
        return ""
    edits = sum(r.get("applied", 0) for r in rounds)
    passed = sum(1 for r in rounds if r.get("passed"))
    pending = sum(1 for r in rounds if r.get("pending"))
    return (f"spec repairs: {edits} edits on {len(rounds)} Tasks, {passed} pass the gates, "
            f"{pending} pending at the ceiling")


__all__ = ["ADDABLE", "PENDING", "REPAIR_GATES", "SPEC_EDITS", "apply_review", "gate_row", "gates_pass",
           "route_findings", "rounds_line", "task_of_edit"]
