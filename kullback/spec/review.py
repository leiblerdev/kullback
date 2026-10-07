"""The Spec answers a review: a finding's atoms and text edits applied by code, then the gates (D320).

The Examiner files one review per Task with the edit to apply (D317); it writes nothing. Its atoms
and text edits come here, never through the Builder. An atoms edit drops and adds atoms on the
Task's Verifier; it is kept on the Spec (`atom_edits`) so a later rewrite keeps it. An added atom is
a fact or a question only: a write is a cell of the expected end state, never an atom (D320). A text
edit replaces its verbatim `where` once in an Intent or Task file of the workdir, and in the Spec's
Intent facts where they hold it, so the checks re-ground against the new words. Nothing here calls a
model: the expected end states are derived again by code from the same Reference.

A cell edit allows or drops a cell of every expected end state, a conduct edit adds or removes a
conduct rule; both are kept on the Spec (`end_state_edits`) and applied after the Verifier is written,
since the end states are derived by code. A reference edit promotes nothing: it opens a ruling the
founder or a later stage answers, so the Task cannot be trusted meanwhile.

Each call is one round, counted on the Spec (`round`). The Verifier is rewritten and the trust gates
rule on it (`spec/trust.py`, the grounded, can-fail and Reference-replay gates); a Task still failing
at the ceiling is set aside as pending, never refused by the Examiner alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.canon import load_rules
from kullback.runner.records import Conduct, ValueSource, Verifier, read_json, write_json
from kullback.spec.router import ROUNDS_CAP
from kullback.spec.schema import Spec, intent_text, load_spec, save_spec, spec_verifier_path
from kullback.spec.writer import (
    ATOM_DEMANDS,
    WRITE_PREDICATE,
    load_inputs,
    runner_verifier_path,
    with_gaps,
    write_verifier,
)

PENDING = "pending"
END_STATE_EDITS = ("cell", "conduct")
SPEC_EDITS = ("atoms", "text", *END_STATE_EDITS, "reference")
CELL_ACTIONS = ("allow", "drop")
CONDUCT_ACTIONS = ("add", "remove")
CONDUCT_KINDS = ("handoff", "refusal", "confirm_before_write")
# The reviews row a reference edit leaves: an open ruling until the founder or a later stage acts.
REFERENCE_PROPOSED = "reference_proposed"
# The Examiner's reviews file (examiner/domain_tools.py REVIEWS_FILE); the Spec sits below the Examiner.
REVIEWS_FILE = "examiner/reviews.json"
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


def atom_checks(spec: Spec, verifier: Optional[Verifier]) -> dict[str, str]:
    """Each atom id of the Verifier with the check that made it (compile names atoms i<n>, n the index
    of the compiled check), or "review" for an atom a review added."""
    if verifier is None:
        return {}
    checks = spec.checks
    if verifier.expected:
        checks = [check for check in checks if check.demand.get("demand") in ATOM_DEMANDS]
    out: dict[str, str] = {}
    for atom in verifier.atoms:
        number = re.match(r"i(\d+)", atom.id)
        index = int(number.group(1)) if number else -1
        out[atom.id] = checks[index].id if 0 <= index < len(checks) else "review"
    return out


def _atoms_edit(spec: Spec, edit: dict, atom_ids: list[str]) -> tuple[Spec, str]:
    refused = next((why for why in map(_check_add, edit.get("add") or []) if why), "")
    if refused:
        return spec, refused
    if not (edit.get("drop") or edit.get("add")):
        return spec, "an atoms edit drops or adds at least one atom"
    drops = [str(a) for a in edit.get("drop") or []]
    unknown = next((a for a in drops if a not in atom_ids), None)
    if unknown is not None:
        return spec, f"drop names no atom of the Verifier: {unknown}; atoms are {', '.join(atom_ids) or 'none'}"
    adds = len(edit.get("add") or [])
    if 2 * len(set(drops)) > len(atom_ids) + adds:  # its adds count, so one wrong atom is replaced in one edit
        return spec, f"drops {len(set(drops))} of {len(atom_ids)} atoms and adds {adds}"
    kept = {"drop": [str(a) for a in edit.get("drop") or []], "add": list(edit.get("add") or []),
            "why": str(edit.get("why") or "")}
    return spec.model_copy(update={"atom_edits": [*spec.atom_edits, kept]}), ""


def _matches(cell: Any, edit: dict) -> bool:
    return all(edit.get(key) in (None, "", getattr(cell, key)) for key in ("table", "row_id", "field"))


def _check_end_state_edit(verifier: Optional[Verifier], edit: dict, tables: set[str]) -> str:
    """Why a cell or conduct edit is refused on this Verifier, or empty when it stands."""
    if verifier is None or not verifier.expected:
        return "the Verifier has no expected end state to edit"
    if edit["kind"] == "cell":
        if edit.get("action") not in CELL_ACTIONS or not edit.get("table"):
            return f"a cell edit names table and action, one of {', '.join(CELL_ACTIONS)}"
        if edit["table"] not in tables:
            return f"the Verifier has no table {edit['table']}; tables are {', '.join(sorted(tables))}"
        cells = [c for state in verifier.expected for c in state.cells]
        if edit["action"] == "drop" and not any(_matches(c, edit) for c in cells):
            return f"no cell of the Verifier matches {_where(edit)}"
        return ""
    if edit.get("action") not in CONDUCT_ACTIONS or edit.get("conduct") not in CONDUCT_KINDS:
        return (f"a conduct edit names action ({', '.join(CONDUCT_ACTIONS)}) and kind "
                f"({', '.join(CONDUCT_KINDS)})")
    if edit["action"] == "remove" and not any(_is_conduct(c, edit) for c in verifier.conduct):
        return f"the Verifier has no {edit['conduct']} conduct" + (f" on {edit['tool']}" if edit.get("tool") else "")
    return ""


def _where(edit: dict) -> str:
    return ".".join(str(edit[key]) for key in ("table", "row_id", "field") if edit.get(key))


def _is_conduct(conduct: Conduct, edit: dict) -> bool:
    return conduct.kind == edit["conduct"] and edit.get("tool") in (None, "", conduct.tool)


def _edited_end_states(verifier: Verifier, edit: dict) -> Verifier:
    """The Verifier with one kept cell or conduct edit applied; one that no longer matches changes nothing."""
    if edit["kind"] == "cell":
        if edit["action"] == "allow":
            rule = {key: edit[key] for key in ("table", "row_id", "field") if edit.get(key)}
            states = [s.model_copy(update={"allowed": [*s.allowed, rule]}) for s in verifier.expected]
        else:
            states = [s.model_copy(update={"cells": [c for c in s.cells if not _matches(c, edit)]})
                      for s in verifier.expected]
        return verifier.model_copy(update={"expected": states})
    if edit["action"] == "remove":
        return verifier.model_copy(update={"conduct": [c for c in verifier.conduct if not _is_conduct(c, edit)]})
    # ValueSource.kind has no examiner kind: the review's why is the clause, the finding its pointer.
    source = ValueSource(kind="policy", ptr={"clause": str(edit.get("why") or ""), "finding": edit.get("finding")})
    added = Conduct(kind=edit["conduct"], tool=edit.get("tool") or None, source=source)
    return verifier.model_copy(update={"conduct": [*verifier.conduct, added]})


def _tables(root: Path, verifier: Optional[Verifier]) -> set[str]:
    """The tables a cell edit may name: the Verifier's own and the schema's."""
    columns = (read_json(root / "schema.json", None) or {}).get("columns") if (root / "schema.json").is_file() else None
    tables = set(columns) if isinstance(columns, dict) else set()
    if verifier is not None:
        tables |= {c.table for s in verifier.expected for c in s.cells}
        tables |= {f.table for f in verifier.forbidden if f.table}
    return tables


def apply_end_state_edits(workdir: Path, spec: Spec) -> Optional[Verifier]:
    """The Spec's kept cell and conduct edits on its written Verifier, both copies rewritten, version bumped."""
    from kullback.spec.trust import load_spec_verifier

    verifier = load_spec_verifier(workdir, spec.task_id)
    if verifier is None or not spec.end_state_edits:
        return verifier
    for edit in spec.end_state_edits:
        verifier = _edited_end_states(verifier, edit)
    verifier = verifier.model_copy(update={"verifier_version": f"{verifier.verifier_version}.e{len(spec.end_state_edits)}"})
    text = json.dumps(verifier.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    for path in (spec_verifier_path(workdir, spec.task_id), runner_verifier_path(workdir, spec.task_id)):
        path.write_text(text, encoding="utf-8")
    return verifier


def _reference_edit(root: Path, spec: Spec, edit: dict) -> tuple[Spec, str]:
    """A proposed Reference recorded as an open ruling; nothing is promoted by code."""
    from kullback.spec.trust import runs_on_disk

    run_id = str(edit.get("run_id") or "")
    if run_id not in {run.run_id for run in runs_on_disk(root).get(spec.task_id, [])}:
        return spec, f"a reference edit names a Run of task {spec.task_id} on disk; {run_id or 'none'} is not one"
    reviews = read_json(root / REVIEWS_FILE, {}) or {}
    reviews[spec.task_id] = {"task_id": spec.task_id, "outcome": REFERENCE_PROPOSED, "run_id": run_id,
                             "reason": str(edit.get("why") or ""), "finding": edit.get("finding")}
    write_json(root / REVIEWS_FILE, reviews)
    return spec.model_copy(update={"rulings_open": spec.rulings_open + 1}), ""


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
    from kullback.spec.trust import load_spec_verifier

    applied, refused = 0, []
    verifier = load_spec_verifier(root, task_id)
    atom_ids = [atom.id for atom in verifier.atoms] if verifier is not None else []
    tables = _tables(root, verifier)
    for edit in edits:
        if edit.get("kind") == "atoms":
            spec, why = _atoms_edit(spec, edit, atom_ids)
        elif edit.get("kind") == "text":
            spec, why = _text_edit(root, spec, edit)
        elif edit.get("kind") in END_STATE_EDITS:
            why = _check_end_state_edit(verifier, edit, tables)
            if not why:
                spec = spec.model_copy(update={"end_state_edits": [*spec.end_state_edits, dict(edit)]})
                verifier = _edited_end_states(verifier, edit)
        elif edit.get("kind") == "reference":
            spec, why = _reference_edit(root, spec, edit)
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
            by_task.setdefault(task_of_edit(edit, finding.get("task_id")), []).append(
                dict(edit, finding=edit.get("finding") or finding.get("finding_id")))
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


__all__ = ["ADDABLE", "END_STATE_EDITS", "PENDING", "REFERENCE_PROPOSED", "REPAIR_GATES", "SPEC_EDITS",
           "apply_end_state_edits", "apply_review", "atom_checks", "gate_row", "gates_pass", "route_findings", "rounds_line", "task_of_edit"]
