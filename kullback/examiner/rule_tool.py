"""The Examiner's ruling tools: rule, close, verify and the world lookups (DESIGN.md rule 3, D320, D331).

The Examiner gives feedback and never edits. It reads the Intent, the policy and the Spec (items with
their provenance) and files rulings: one item, a kind and code (spec/rulings.py RULING_KINDS), the reason
in one sentence, the fix the writer should make, blocking or note. The writer applies or rebuts each in
its next round; `close` then closes the ruling or keeps it open, with why.

`verify` runs the constructed Runs (do nothing, one stray write) and the Reference's end state through
the Task's Verifier by code and keeps the rows, so a code ruling can cite them. `lookup_rows` and
`search_rows` are the writer's own two lookups over the Task's Starting state: the Examiner checks every
anchor and value with the same world the writer had.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.records import EXAM_DIR, Run, Verifier, read_json, run_path, write_json
from kullback.runner.target import load_run
from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.schema import Spec, load_spec

RULINGS_DIR = R.RULINGS_DIR
REVIEW_DIR = "review"
VIEW_DIR = f"{REVIEW_DIR}/views"
POLICY_FILE = f"{REVIEW_DIR}/policy.md"
TOOLS_FILE = f"{REVIEW_DIR}/tools.json"
REFUSED_FILE = "refused.jsonl"


class RuleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    item: str = Field(description="The id the ruling is about: a check, an atom, an Intent fact, or task.")
    kind: R.RulingKind = Field(description="derivation, judge, code or scope.")
    code: str = Field(description="How, one of the kind's codes: " + "; ".join(
        f"{kind}: {', '.join(codes)}" for kind, codes in R.RULING_KINDS.items()))
    reason: str = Field(description="What is wrong, in one sentence.")
    fix: str = Field(description="What the writer should do, in one sentence.")
    blocking: bool = Field(description="True when the Task must not be trusted until this is fixed; false for a note.")
    wrong_side: Optional[str] = Field(default=None, description="For fails_reference: reference or verifier.")


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    number: int
    path: str


class CloseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    number: int
    keep_open: bool = Field(description="True when the writer's answer does not settle the ruling.")
    why: str = Field(description="Why, in one sentence.")


class CloseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


class VerifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    rows: list[dict] = Field(default_factory=list)


class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    table: str
    key: str = ""


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    table: str
    field: str
    value: str


class TextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


# --- Runs, as other readers of the Examiner's root open them ---


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


def reference_ids_of(workdir: Any, task_id: str) -> set[str]:
    row = (read_json(Path(workdir) / "references.json", {}) or {}).get(task_id) or {}
    return {str(r.get("run_id")) for r in row.get("references") or [] if isinstance(r, dict)}


def read_rulings(workdir: Any, task_id: str) -> list[dict]:
    """The Task's rulings in number order, as dicts."""
    return [r.model_dump(mode="json") for r in R.load_rulings(workdir, task_id)]


# --- filing ---


def _refuse(workdir: Any, task_id: str, code: str, why: str, text: str = "") -> None:
    """Count the refusal where a report reads it (the code, the Task, a hash of the text), then raise it."""
    path = Path(workdir) / EXAM_DIR / RULINGS_DIR / REFUSED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"task_id": task_id, "code": code, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    raise ValueError(why)


def _spec_and_verifier(workdir: Any, task_id: str) -> tuple[Optional[Spec], Optional[Verifier]]:
    from kullback.spec.trust import load_spec_verifier

    return load_spec(workdir, task_id), load_spec_verifier(workdir, task_id)


def checked_ruling(root: ExamRoot, args: RuleArgs) -> R.Ruling:
    """The ruling as it will be filed, or a counted refusal naming what is missing."""
    workdir, task_id = root.workdir, args.task_id
    spec, verifier = _spec_and_verifier(workdir, task_id)
    if spec is None:
        _refuse(workdir, task_id, "no_spec", "the Task has no Spec to rule on; rule on a Task the opening lists")
    why = R.code_refusal(args.kind, args.code)
    if why:
        _refuse(workdir, task_id, "bad_code", why)
    known = R.item_ids(spec, verifier)
    if args.item not in known:
        _refuse(workdir, task_id, "unknown_item", f"item {args.item!r} is not on the Task; items are {', '.join(known)}")
    for text, what in ((args.reason, "reason"), (args.fix, "fix")):
        why = R.sentence_refusal(text, what)
        if why:
            _refuse(workdir, task_id, f"bad_{what}", why, text)
    if args.code == "fails_reference" and args.wrong_side not in R.WRONG_SIDES:
        _refuse(workdir, task_id, "no_side", "a fails_reference ruling says which side is wrong: wrong_side "
                "reference or verifier")
    verified = list(root.verified.get(task_id) or [])
    if args.code in R.VERIFY_CODES and not verified:
        _refuse(workdir, task_id, "not_verified", f"call verify on task {task_id} first: a {args.code} ruling "
                "cites what the Verifier did")
    # A wrong Reference leaves the Verifier standing: the ruling is a note for the witness flag, never blocking.
    blocking = bool(args.blocking) and args.wrong_side != "reference"
    return R.Ruling(task_id=task_id, number=R.next_number(workdir, task_id), round=root.round or spec.round + 1,
                    item=args.item, kind=args.kind, code=args.code, reason=" ".join(args.reason.split()),
                    fix=" ".join(args.fix.split()), blocking=blocking, wrong_side=args.wrong_side,
                    verified=verified if args.kind == "code" else [])


def file_ruling(root: ExamRoot, args: RuleArgs) -> dict:
    """Write the ruling under exam/rulings/<task>/<n>.json, count it on the Spec, publish ruling.filed."""
    ruling = checked_ruling(root, args)
    path = R.save_ruling(root.workdir, ruling)
    R.sync_spec(root.workdir, ruling.task_id)
    if root.bus is not None:
        events.ruling_filed(root.bus, events.ruling_payload(ruling))
    return dict(ruling.model_dump(mode="json"), path=path.relative_to(root.exam_dir).as_posix())


def _rule(root: ExamRoot):
    async def rule(args: RuleArgs) -> RuleResult:
        body = file_ruling(root, args)
        side = " (the Reference is ruled wrong: a note)" if body.get("wrong_side") == "reference" else ""
        summary = (f"ruling {body['number']} on task {args.task_id} filed: {args.kind}/{args.code} on "
                   f"{args.item}, {'blocking' if body['blocking'] else 'note'}{side}; the writer answers it next round")
        return RuleResult(summary=summary, task_id=args.task_id, number=body["number"], path=body["path"])

    return rule


def _close(root: ExamRoot):
    async def close(args: CloseArgs) -> CloseResult:
        why = R.sentence_refusal(args.why, "why")
        if why:
            _refuse(root.workdir, args.task_id, "bad_why", why, args.why)
        try:
            ruling = R.close(root.workdir, args.task_id, args.number, args.keep_open, args.why, root.round)
        except (OSError, ValueError) as exc:
            _refuse(root.workdir, args.task_id, "bad_close", str(exc) or f"no ruling {args.number} on task {args.task_id}")
        R.sync_spec(root.workdir, args.task_id)
        if root.bus is not None:
            events.ruling_closed(root.bus, args.task_id, args.number, args.keep_open)
        state = "kept open" if args.keep_open else "closed"
        return CloseResult(summary=f"ruling {ruling.number} on task {args.task_id} {state}")

    return close


# --- verify: the constructed Runs and the Reference through the Verifier, by code ---


def _reference_run(workdir: Any, task_id: str) -> Optional[Run]:
    """The Task's first faithful Reference replay on disk, else None."""
    from kullback.spec.trust import faithful_references, runs_on_disk

    wanted = set(faithful_references(workdir).get(task_id) or ())
    return next((run for run in runs_on_disk(workdir).get(task_id, []) if run.run_id in wanted), None)


def verify_rows(workdir: Any, task_id: str, verifier: Verifier, reference: Optional[Run]) -> list[dict]:
    """What the Verifier says of each constructed Run and of the Reference.

    A constructed Run (do nothing, one stray write on the Reference, spec/trust.py) should fail; the
    Reference passing says the Verifier is satisfiable. A Run the Verdict cannot read is a row with its error.
    """
    from kullback.gates.probes import write_tools_of
    from kullback.runner.canon import load_rules
    from kullback.runner.records import ToolSig
    from kullback.spec.canfail import judge_run
    from kullback.spec.trust import constructed_runs

    if reference is None:
        return [{"run": "reference", "error": "no faithful Reference on disk"}]
    sigs = read_json(Path(workdir) / "tool_sigs.json", None) or []
    sigs = sigs.get("sigs", []) if isinstance(sigs, dict) else sigs
    tools = write_tools_of([ToolSig.model_validate(s) for s in sigs])
    canon = load_rules(Path(workdir) / "canon-rules.json")
    schema = read_json(Path(workdir) / "schema.json", None)
    rows = []
    try:
        built = constructed_runs(verifier, [reference], [reference.run_id], schema=schema, canon=canon)
        for name, run in built.items():
            if run is not None:
                rows.append({"run": name, "verifier_fails_it": not judge_run(verifier, run, canon, tools)[0]})
    except Exception as exc:  # a constructed Run the Verdict cannot read says the code does not run
        rows.append({"run": "constructed", "error": type(exc).__name__})
    try:
        passed, failing = judge_run(verifier, reference, canon, tools)
        rows.append({"run": "reference", "verifier_passes_it": bool(passed),
                     "failing": None if passed else str(getattr(failing, "id", failing))})
    except Exception as exc:
        rows.append({"run": "reference", "error": type(exc).__name__})
    return rows


def verify_line(rows: list[dict]) -> str:
    """The verify rows in one line: each kind of Run with how many the Verifier failed, the Reference's word."""
    parts, kinds = [], {}
    for row in rows:
        if "error" in row:
            parts.append(f"{row['run']}: {row['error']}")
        elif row["run"] == "reference":
            parts.append("reference: " + ("passes" if row["verifier_passes_it"] else f"fails on {row['failing']}"))
        else:
            failed, total = kinds.get(row["run"], (0, 0))
            kinds[row["run"]] = (failed + bool(row["verifier_fails_it"]), total + 1)
    lines = [f"{run}: the Verifier fails {failed} of {total}" for run, (failed, total) in kinds.items()]
    return "; ".join(lines + parts)


def _verify(root: ExamRoot):
    async def verify(args: VerifyArgs) -> VerifyResult:
        from kullback.spec.trust import load_spec_verifier

        verifier = load_spec_verifier(root.workdir, args.task_id)
        if verifier is None:
            raise ValueError(f"task {args.task_id} has no Verifier to run")
        rows = verify_rows(root.workdir, args.task_id, verifier, _reference_run(root.workdir, args.task_id))
        root.verified[args.task_id] = rows
        return VerifyResult(summary=verify_line(rows), rows=rows)

    return verify


# --- the writer's own world lookups, per Task ---


def _state(root: ExamRoot, task_id: str) -> dict:
    """The Task's Starting state as the writer read it, loaded once per session."""
    if task_id not in root.states:
        from kullback.spec.writer import load_inputs

        root.states[task_id] = load_inputs(Path(root.workdir), task_id).state
    return root.states[task_id]


def _lookups(root: ExamRoot) -> list[AgentTool]:
    from kullback.examiner.domain_tools import render
    from kullback.spec.writer_tools import READ_TOOLS, read_tools

    described = {tool["name"]: tool["description"] for tool in READ_TOOLS}

    async def lookup_rows(args: LookupArgs) -> TextResult:
        return TextResult(summary=read_tools(_state(root, args.task_id))["lookup_rows"](args.table, args.key))

    async def search_rows(args: SearchArgs) -> TextResult:
        found = read_tools(_state(root, args.task_id))["search_rows"](args.table, args.field, args.value)
        return TextResult(summary=found)

    return [AgentTool("lookup_rows", described["lookup_rows"] + " Names the Task.", LookupArgs, TextResult,
                      lookup_rows, render=render),
            AgentTool("search_rows", described["search_rows"] + " Names the Task.", SearchArgs, TextResult,
                      search_rows, render=render)]


# --- the view: what the Examiner reads of a Task ---


def _item(check: Any) -> dict:
    """One check of the Spec with its provenance and, when the record carries them, gate and weight."""
    row = {"id": check.id, "kind": check.kind, "tier": check.tier, "demand": check.demand,
           "because": check.because, "fact_ids": check.fact_ids}
    for name in ("gate", "weight", "policy_line"):
        if getattr(check, name, None) is not None:
            row[name] = getattr(check, name)
    return row


def spec_view(root: ExamRoot, task_id: str) -> Optional[dict]:
    """The Task as the Examiner reads it: the Intent facts, the Spec's items with provenance, the
    Verifier's items, the facts no item covers, and every ruling so far with the writer's answer."""
    spec, verifier = _spec_and_verifier(root.workdir, task_id)
    if spec is None:
        return None
    covered = {fact_id for check in spec.checks for fact_id in check.fact_ids}
    view = {"task_id": task_id, "round": spec.round, "version": spec.version,
            "facts": [{"id": f.id, "stance": f.stance, "text": f.text} for f in spec.intent.facts],
            "items": [_item(check) for check in spec.checks],
            "facts_without_item": [f.id for f in spec.intent.facts if f.id not in covered],
            "rulings": read_rulings(root.workdir, task_id)}
    if verifier is not None:
        view["verifier"] = {"atoms": [{"id": a.id, "kind": a.kind, "judge": a.judge, "description": a.description,
                                       "target": a.target} for a in verifier.atoms],
                            **{name: [item.model_dump(mode="json") for item in getattr(verifier, name)]
                               for name in ("expected", "forbidden", "conduct")}}
    return view


def write_views(root: ExamRoot, task_ids: Iterable[str]) -> dict[str, str]:
    """Each Task's view, the policy and the tool list under exam/review/; the root-relative view paths."""
    from kullback.spec.writer import load_inputs

    out: dict[str, str] = {}
    for task_id in task_ids:
        view = spec_view(root, task_id)
        if view is None:
            continue
        path = root.exam_dir / VIEW_DIR / f"{task_id}.json"
        write_json(path, view)
        out[task_id] = path.relative_to(root.exam_dir).as_posix()
        if not (root.exam_dir / POLICY_FILE).is_file():
            try:
                inputs = load_inputs(Path(root.workdir), task_id)
                policy, tools = inputs.policy_text, inputs.tools
            except Exception:  # a workdir without a built world still gets its views
                policy, tools = "", []
            (root.exam_dir / POLICY_FILE).write_text(policy or "no policy recorded", encoding="utf-8")
            write_json(root.exam_dir / TOOLS_FILE, tools)
    return out


def rule_tools(root: ExamRoot) -> list[AgentTool]:
    """rule, close, verify, lookup_rows and search_rows over one root."""
    from kullback.examiner.domain_tools import render

    return [
        AgentTool("rule", "File one ruling on one item of a Task: kind and code, the reason and the fix in one "
                  "sentence each, blocking or note. You never edit; the writer applies or rebuts it.",
                  RuleArgs, RuleResult, _rule(root), render=render),
        AgentTool("close", "Close a ruling the writer answered, or keep it open, with why in one sentence.",
                  CloseArgs, CloseResult, _close(root), render=render),
        AgentTool("verify", "Run the constructed Runs (do nothing, one stray write, an expected cell undone) and the "
                  "Reference's end state through the Task's Verifier, by code. A code ruling cites it.",
                  VerifyArgs, VerifyResult, _verify(root), render=render),
        *_lookups(root),
    ]


RULE_TOOL_NAMES = ("rule", "close", "verify", "lookup_rows", "search_rows")

__all__ = ["POLICY_FILE", "REFUSED_FILE", "REVIEW_DIR", "RULE_TOOL_NAMES", "RULINGS_DIR", "TOOLS_FILE", "VIEW_DIR",
           "CloseArgs", "RuleArgs", "RuleResult", "VerifyArgs", "checked_ruling", "file_ruling", "open_run",
           "read_rulings", "reference_ids_of", "rule_tools", "run_rows", "spec_view", "turn_texts", "verify_rows",
           "write_views"]
