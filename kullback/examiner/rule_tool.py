"""The Examiner's ruling tool: `rule` files a ruling with a target (DESIGN.md rule 3, D320).

The Examiner judges only. It reads the Spec of a Task (facts with stance, checks with because and
tier), each Run's verdict rows against the Spec's checks, and the Runs, and files a ruling that
says what is wrong: a Run, a check, the Intent, or the Environment. Code moves on the ruling
(kullback/spec/router.py); the Examiner holds no pen over the Verifier.

A ruling argues from evidence: its reason must quote, verbatim and at least MIN_QUOTE characters, a
because of the Task's Spec or a turn of the Run it names. Code checks the quote and refuses a
ruling without one. The ruling is written whole to exam/rulings/<task>/<n>.json and published as
ruling.filed with ids, counts and the reason with each quote replaced by where it came from.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.examiner.exam_files import ExamRoot
from kullback.gates.probes import write_tools_of
from kullback.runner.records import EXAM_DIR, Run, Verifier, read_json, run_path, write_json
from kullback.runner.target import atom_payload, canon_fn, check_run, load_run
from kullback.spec import compile as compile_mod
from kullback.spec import events
from kullback.spec.ground import MIN_QUOTE
from kullback.spec.schema import Spec, load_spec

RULINGS_DIR = "rulings"
VIEW_DIR = "spec_view"
REFUSED_FILE = "refused.jsonl"
#: The evidence a quote may come from: a because of the Spec, or a turn of the named Run.
BECAUSE, TURN = "because", "turn"
# Where a quoted span ends: sentence ends, quote marks, colons, semicolons and brackets.
_SPAN_ENDS = r"[.!?\n\"\u201c\u201d:;()\[\]]+"


class RuleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    target: events.Target = Field(description="What is wrong: run, check, intent or environment.")
    reason: str = Field(description="Why, quoting verbatim a because of the Task's Spec or a turn of the named Run.")
    code: events.RulingCode = Field(default="other", description="The kind of reason: " + ", ".join(events.RULING_CODES))
    check_ids: list[str] = Field(default_factory=list, description="The checks the ruling is about.")
    run_id: Optional[str] = Field(default=None, description="The Run the ruling is about or quotes.")
    excerpt: Optional[str] = Field(default=None, description="The quoted evidence, verbatim, when it is long.")


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    number: int
    path: str
    quoted: list[str] = Field(default_factory=list)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def turn_texts(run: Run) -> list[str]:
    """The Run's turns in order: what the user said and what the agent said, tool traffic left out."""
    out = []
    for event in run.events:
        payload = event.payload or {}
        if event.type == "user_turn":
            out.append(str(payload.get("text") or payload.get("content") or ""))
        elif event.type == "model_call":
            reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else payload
            out.append(str(reply.get("content") or ""))
    return [text for text in out if text.strip()]


def run_rows(root: ExamRoot, task_id: str) -> dict[str, str]:
    """The Task's Runs by id, each with its stored path: the replay rows, then the re-roll rows."""
    rows = list(((root.replays or {}).get(task_id) or {}).values()) + list((root.rerolls or {}).get(task_id) or [])
    return {str(row["run_id"]): str(row["path"]) for row in rows
            if isinstance(row, dict) and row.get("run_id") and row.get("path")}


def open_run(root: ExamRoot, task_id: str, run_id: str) -> Optional[Run]:
    """One Run of the Task from the Examiner's copy, else the workdir's; None when it is not the Task's."""
    stored = run_rows(root, task_id).get(str(run_id))
    if stored is None:
        return None
    for base in (root.exam_dir, Path(root.workdir)):
        path = run_path(base, stored)
        if path.is_file():
            try:
                return load_run(path)
            except (OSError, ValueError):
                return None
    return None


def evidence(spec: Optional[Spec], run: Optional[Run]) -> list[tuple[str, str]]:
    """Every text a reason may quote, each with where it came from."""
    out = [(f"{BECAUSE} {check.id}", check.because) for check in (spec.checks if spec else ())]
    out += [(f"{TURN} {index}", text) for index, text in enumerate(turn_texts(run) if run else ())]
    return out


def quoted_spans(text: str, sources: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """The spans of `text` (split at sentence ends, quote marks, colons and brackets) found verbatim in a source."""
    flat = [(label, _norm(body)) for label, body in sources]
    found = []
    for span in re.split(_SPAN_ENDS, str(text or "")):
        key = _norm(span.strip("' "))
        if len(key) < MIN_QUOTE:
            continue
        label = next((label for label, body in flat if key in body), None)
        if label is not None:
            found.append((span.strip("' "), label))
    return found


def rulings_dir(workdir: Any, task_id: str) -> Path:
    return Path(workdir) / EXAM_DIR / RULINGS_DIR / task_id


def next_number(workdir: Any, task_id: str) -> int:
    folder = rulings_dir(workdir, task_id)
    numbers = [int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()] if folder.is_dir() else []
    return max(numbers, default=0) + 1


def _refuse(workdir: Any, args: "RuleArgs", why: str) -> None:
    """Count the refusal where a report reads it (the code, the ids, a hash of the reason), then raise it."""
    path = Path(workdir) / EXAM_DIR / RULINGS_DIR / REFUSED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"task_id": args.task_id, "code": why, "target": args.target, "check_ids": list(args.check_ids),
           "run_id": args.run_id, "reason_sha256": hashlib.sha256(args.reason.encode("utf-8")).hexdigest()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    raise ValueError(REFUSALS[why])


REFUSALS = {
    "no_spec": "the Task has no Spec to rule on; rule on a Task whose Spec is listed in the opening",
    "unknown_run": "run_id is not a Run of this Task; name one of the Runs the opening lists",
    "no_run": "a ruling on a Run names it in run_id",
    "unknown_check": "a check id is not on the Task's Spec; name checks from its spec view",
    "no_check": "a ruling on a check names it in check_ids",
    events.QUOTE_MISSING: ("the reason quotes no evidence: copy verbatim, in quotation marks and at least %d characters, "
                 "a because of the Spec or a turn of the Run in run_id" % MIN_QUOTE),
    "bad_excerpt": "the excerpt is not verbatim in a because of the Spec or a turn of the Run in run_id",
}


def _checked(root: ExamRoot, args: RuleArgs) -> tuple[Spec, Optional[Run], list[tuple[str, str]]]:
    """The Spec, the named Run and the quoted spans, or a counted refusal."""
    task_id, workdir = args.task_id, root.workdir
    spec = load_spec(workdir, task_id)
    if spec is None:
        _refuse(workdir, args, "no_spec")
    run = None
    if args.run_id is not None:
        run = open_run(root, task_id, args.run_id)
        if run is None:
            _refuse(workdir, args, "unknown_run")
    elif args.target == "run":
        _refuse(workdir, args, "no_run")
    known = {check.id for check in spec.checks}
    if any(check_id not in known for check_id in args.check_ids):
        _refuse(workdir, args, "unknown_check")
    if args.target == "check" and not args.check_ids:
        _refuse(workdir, args, "no_check")
    sources = evidence(spec, run)
    if args.excerpt and not quoted_spans(args.excerpt, sources):
        _refuse(workdir, args, "bad_excerpt")
    spans = quoted_spans(args.reason, sources)
    if not spans:
        _refuse(workdir, args, events.QUOTE_MISSING)
    return spec, run, spans


def file_ruling(root: ExamRoot, args: RuleArgs) -> dict:
    """Write the ruling whole under exam/rulings/<task>/<n>.json and publish ruling.filed."""
    spec, _, spans = _checked(root, args)
    number = next_number(root.workdir, args.task_id)
    labels = [label for _, label in spans]
    payload = events.ruling_payload(args.task_id, args.target, args.check_ids, args.run_id, args.code, labels,
                                    args.excerpt, spec.round, number)
    body = dict(payload, code=args.code, reason=args.reason, excerpt=args.excerpt, quoted=labels)
    path = rulings_dir(root.workdir, args.task_id) / f"{number}.json"
    write_json(path, body)
    if root.bus is not None:
        events.ruling_filed(root.bus, payload)
    return dict(body, path=path.relative_to(root.exam_dir).as_posix())


def _rule(root: ExamRoot):
    async def rule(args: RuleArgs) -> RuleResult:
        body = file_ruling(root, args)
        summary = (f"ruling {body['number']} on task {args.task_id} filed: target {args.target}, "
                   f"quoting {', '.join(body['quoted'])}; code routes it")
        return RuleResult(summary=summary, task_id=args.task_id, number=body["number"], path=body["path"],
                          quoted=body["quoted"])

    return rule


# --- the view: the Spec and each Run's verdict rows against its checks ---


def _check_atoms(check: Any, write_tools: set, fn: Any) -> list:
    build = compile_mod.BUILDERS.get(str(check.demand.get("demand")))
    return (build(check.id, check.kind, check.demand, write_tools, fn) or []) if build else []


def _expected(atoms: list) -> list[dict]:
    rows = []
    for atom in atoms:
        payload = atom_payload(atom)
        if "raw" in payload or "value" in payload:
            rows.append({"atom": atom.id, "field": payload.get("field"),
                         "expected": payload.get("raw", payload.get("value"))})
    return rows


def verdict_rows(spec: Spec, run: Run, canon: Any, write_tools: set) -> dict:
    """Which checks the Run fails and the values they expect; a check no builder compiles says so."""
    fn = canon_fn(canon)
    failed, unscored = [], []
    for check in spec.checks:
        atoms = _check_atoms(check, write_tools, fn)
        if not atoms:
            unscored.append(check.id)
            continue
        ok, atom_id = check_run(Verifier(task_id=spec.task_id, atoms=atoms), run, canon, write_tools=write_tools)
        if not ok and atom_id in {atom.id for atom in atoms}:
            failed.append({"check": check.id, "tier": check.tier, "atom": atom_id, "values": _expected(atoms)})
    return {"run_id": run.run_id, "termination_reason": run.termination_reason, "failed": failed,
            "unscored": unscored, "passed": not failed and not unscored}


def spec_view(root: ExamRoot, task_id: str) -> Optional[dict]:
    """The Task's view: the Spec's facts and checks, the Verifier's gates, then each Run's verdict rows."""
    spec = load_spec(root.workdir, task_id)
    if spec is None:
        return None
    tools = write_tools_of(root.sigs)
    runs = []
    for run_id, stored in sorted(run_rows(root, task_id).items()):
        run = open_run(root, task_id, run_id)
        if run is not None:
            runs.append(dict(verdict_rows(spec, run, root.canon_rules, tools), path=stored))
    return {"task_id": task_id, "round": spec.round, "version": spec.version,
            "facts": [{"id": f.id, "stance": f.stance, "text": f.text} for f in spec.intent.facts],
            "checks": [{"id": c.id, "kind": c.kind, "tier": c.tier, "because": c.because,
                        "demand": c.demand, "fact_ids": c.fact_ids} for c in spec.checks],
            "gaps": spec.gaps, "end_state": spec.end_state, **_gates_view(root, task_id), "runs": runs}


def _gates_view(root: ExamRoot, task_id: str) -> dict:
    """The Spec Verifier's gates as the review reads them: each expected cell with its source (D320)."""
    from kullback.spec.trust import load_spec_verifier

    verifier = load_spec_verifier(root.workdir, task_id)
    if verifier is None:
        return {"expected": [], "forbidden": [], "conduct": []}
    return {name: [item.model_dump(mode="json") for item in getattr(verifier, name)]
            for name in ("expected", "forbidden", "conduct")}


def write_views(root: ExamRoot, task_ids: Iterable[str]) -> dict[str, str]:
    """Write each Task's view under exam/spec_view/; returns the root-relative path per Task with a Spec."""
    out = {}
    for task_id in task_ids:
        view = spec_view(root, task_id)
        if view is not None:
            path = root.exam_dir / VIEW_DIR / f"{task_id}.json"
            write_json(path, view)
            out[task_id] = path.relative_to(root.exam_dir).as_posix()
    return out


def rule_tool(root: ExamRoot) -> AgentTool:
    from kullback.examiner.domain_tools import render

    return AgentTool("rule", "File one ruling on a Task: target run, check, intent or environment, with a reason "
                     "that quotes verbatim a because of its Spec or a turn of the Run in run_id. Code routes it.",
                     RuleArgs, RuleResult, _rule(root), render=render)


def reject_tool(root: ExamRoot) -> AgentTool:
    """reject_reference in intent mode: a ruling on the Reference Run, so one path excludes it (the router)."""
    from kullback.examiner import reference_check as RC
    from kullback.examiner.domain_tools import render

    async def reject(args: RC.RejectReferenceArgs) -> RC.RejectReferenceResult:
        body = file_ruling(root, RuleArgs(task_id=args.task_id, target="run", run_id=args.run_id, reason=args.why,
                                          code="reference_wrong"))
        refs = sorted(reference_ids_of(root.workdir, args.task_id))
        summary = (f"ruling {body['number']} on task {args.task_id} filed against Run {args.run_id}; "
                   f"the References are now {', '.join(refs) or 'none'}")
        return RC.RejectReferenceResult(summary=summary, task_id=args.task_id, excluded=[args.run_id],
                                        references=refs, pooled=not refs)

    return AgentTool("reject_reference", "Reject a Reference Run the Spec says is wrong: files rule(target=run) "
                     "on it, so the why must quote a because of the Spec or a turn of that Run.",
                     RC.RejectReferenceArgs, RC.RejectReferenceResult, reject, render=render)


def reference_ids_of(workdir: Any, task_id: str) -> set[str]:
    row = (read_json(Path(workdir) / "references.json", {}) or {}).get(task_id) or {}
    return {str(r.get("run_id")) for r in row.get("references") or [] if isinstance(r, dict)}


def read_rulings(workdir: Any, task_id: str) -> list[dict]:
    """The Task's rulings in number order."""
    folder = rulings_dir(workdir, task_id)
    paths = sorted((p for p in folder.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)) \
        if folder.is_dir() else []
    return [read_json(p, {}) for p in paths]


__all__ = ["REFUSALS", "RULINGS_DIR", "VIEW_DIR", "RuleArgs", "RuleResult", "evidence", "file_ruling",
           "open_run", "quoted_spans", "read_rulings", "reference_ids_of", "reject_tool", "rule_tool", "run_rows", "spec_view",
           "turn_texts", "verdict_rows", "write_views"]
