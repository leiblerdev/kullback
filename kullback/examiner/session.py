"""The Examiner as an extension on the core, called by the Builder (D79, D111, D122-D124, D128, D230).

`examine` derives one Verifier per Task from its References through the suite by code,
files what the records already say as findings with rows, then runs one model session
over the Examiner's root when a model is given. It returns the findings and writes them
to `findings.json` at once, never a round late (learnings 7).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.agent.base_tools import register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig, prompt_block
from kullback.agent.extensions import ExtensionAPI, load_extensions, refuse_paths
from kullback.agent.harness import AgentHarness
from kullback.agent.session import SessionStore
from kullback.examiner import findings as findings_mod
from kullback.examiner import prompt as prompt_mod
from kullback.examiner import runners as runners_mod
from kullback.examiner import stage as stage_mod
from kullback.examiner.domain_tools import domain_tools
from kullback.examiner.exam_files import (
    DERIVED_DIR,
    ExamRoot,
    Finding,
    evidence_of,
    expose,
    load_history,
    names_forbidden_path,
    save_history,
    seeded_history,
)
from kullback.gates import names_protected_path
from kullback.gates.hook import PATH_KEYS, WRITE_TOOLS, gate_writes
from kullback.gates.ledger import GateLedger
from kullback.gates.loosening import finished_run_ids
from kullback.gates.verifier_suite import load_run
from kullback.runner import budget
from kullback.runner.canon import rules_of
from kullback.runner.records import Constraint, Task, ToolSig, UserRules, Verifier, read_json, run_path, write_json

BASE_ONLY = ("read", "write", "edit", "grep", "find", "ls", "web_search")
WRITABLE_DIRS = prompt_mod.WRITABLE_DIRS
EXAMINE_MESSAGE = "Examine."
# The Examiner's turn cap: reading Runs, replays and Verifiers before filing takes tens of turns, and
# at 8 every live session stopped while still reading (F17).
EXAMINE_MAX_TURNS = 40
# The harness's own words when a session stops on its cap (agent/loop.py).
CAP_STOP = "stopped after max_turns="
HINT_CHARS = 200
# How many Run paths one Task's rulings line names before the count of the rest.
RUN_PATHS_SHOWN = 5
# How many left-out Tasks the note names before the count of the rest.
LEFT_OUT_SHOWN = 10
NO_FINISHED_RUN = "no finished Run"
NO_CONFIRMED_REFERENCE = "no confirmed Reference"


def _written_path(arguments: Any) -> Optional[str]:
    """Where a write or edit lands, first path-like argument wins."""
    if not isinstance(arguments, dict):
        return None
    for key in PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _refuse_forbidden_paths() -> Callable:
    """The carried read refusal, except for the filing tool: a finding names the
    Environment file the Builder should edit, it never reads it (D123)."""
    inner = refuse_paths(
        names_forbidden_path,
        "which the Examiner never reads: a tool body, the Starting state, the schema, "
        "the Environment, the sandbox or the overlays (D123)",
        "examiner_reads_only_its_surface")

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) == "finding":
            return None
        return inner(call)

    refuse.hook_name = "examiner_reads_only_its_surface"  # type: ignore[attr-defined]
    return refuse


def _refuse_outside_roots() -> Callable:
    """A tool_call hook refusing a write or edit outside verifiers/ and probes/."""

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) not in WRITE_TOOLS:
            return None
        rel = (_written_path(call.arguments or {}) or "").replace("\\", "/").lstrip("/")
        if (rel == "verifiers" or rel.startswith("verifiers/") or rel == "probes"
                or rel.startswith("probes/")):
            return None
        raise PermissionError(
            f"the Examiner writes only under verifiers/ and probes/, not {rel!r}")

    refuse.hook_name = "examiner_writes_only_proposals"  # type: ignore[attr-defined]
    return refuse


def _result_protection(api: ExtensionAPI) -> Callable:
    """A tool_result hook (D124): a result carrying a failed ruling stays protected.

    The entry is guarded until the next write of the same path, so a failed ruling cannot
    be compacted away before the proposal that answers it.
    """
    guarded: dict[str, str] = {}

    def protect(call: Any, result: Any) -> Any:
        if result is None or getattr(result, "is_error", False):
            return None
        details = getattr(result, "details", None) or {}
        path = _written_path(getattr(call, "arguments", None) or {})
        if getattr(call, "name", None) in WRITE_TOOLS and path in guarded:
            api.unprotect([guarded.pop(path)])
            return None
        rulings = details.get("rulings") or []
        if not any(isinstance(r, dict) and not r.get("accepted", True) for r in rulings):
            return None
        if path is None:
            return None
        entry = api.entry_id_for(call.id)
        if entry is None:
            return None
        api.protect([entry], f"unacted ruling on {path}")
        guarded[path] = entry
        return None

    protect.hook_name = "examiner_result_protection"  # type: ignore[attr-defined]
    return protect


def rulings_line(root: ExamRoot, task_ids: Optional[Iterable[str]] = None) -> str:
    """One line per Task saying what it has, with the paths of its Runs and its Verifier.

    Paths are relative to the Examiner's root, so the session opens them instead of searching.
    `task_ids` limits the lines to the Tasks the session examines.
    """
    lines = []
    wanted = set(task_ids) if task_ids is not None else None
    for task_id in sorted({*root.replays, *root.rerolls, *root.verifiers}):
        if wanted is not None and task_id not in wanted:
            continue
        row = root.task_status.get(task_id) or {}
        parts = ["reference confirmed" if row.get("reference_confirmed") else "no reference",
                 "verifier passed" if row.get("verifier_passed") else "verifier not passed",
                 _runs_part(root, task_id), _verifier_part(root, task_id)]
        lines.append(f"{task_id}: " + "; ".join(parts))
    return "\n".join(lines) if lines else "no Tasks"


def _runs_part(root: ExamRoot, task_id: str) -> str:
    """The Task's Runs as paths under the Examiner's root, the ones its copy holds."""
    rows = list(((root.replays or {}).get(task_id) or {}).values())
    rows += list((root.rerolls or {}).get(task_id) or [])
    paths: list[str] = []
    for row in rows:
        stored = row.get("path") if isinstance(row, dict) else None
        rel = _relative_to(run_path(root.workdir, stored), Path(root.workdir)) if stored else None
        if rel and rel not in paths and (root.exam_dir / rel).is_file():
            paths.append(rel)
    if not paths:
        return "runs: none copied"
    rest = len(paths) - RUN_PATHS_SHOWN
    return "runs: " + ", ".join(paths[:RUN_PATHS_SHOWN]) + (f" and {rest} more" if rest > 0 else "")


def _verifier_part(root: ExamRoot, task_id: str) -> str:
    """The Task's derived Verifier, its proposal and its user rules, as paths under the root (F33)."""
    derived = f"{DERIVED_DIR}/{task_id}.json"
    rel = f"verifiers/{task_id}.json"
    parts = [f"derived verifier: {derived}" if (root.exam_dir / derived).is_file()
             else "derived verifier: none copied"]
    parts.append(f"proposal: {rel}" if (root.exam_dir / rel).is_file()
                 else f"proposal: {rel} once you first propose one")
    trace_id = reference_trace_id(root, task_id)
    parts.append(f"user rules: user_rules/{trace_id}.json" if trace_id
                 else "user rules: no user rules until the Reference is confirmed")
    return "; ".join(parts)


def reference_trace_id(root: ExamRoot, task_id: str) -> Optional[str]:
    """The confirmed replay's trace id off the Task's replay rows, the choice the Runner's rules make.

    Rows in id order, confirmed ones only; the first whose rules sit in the root wins, else the first.
    """
    rows = (root.replays or {}).get(task_id) or {}
    confirmed = [str(row["trace_id"]) for _, row in sorted(rows.items())
                 if isinstance(row, dict) and row.get("confirmed") and row.get("trace_id")]
    held = [trace for trace in confirmed if (root.exam_dir / "user_rules" / f"{trace}.json").is_file()]
    return (held or confirmed or [None])[0]


def _relative_to(path: Path, base: Path) -> Optional[str]:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return None


def root_listing(folder: Path) -> list[str]:
    """The entries directly under a root, directories with a trailing slash, sorted."""
    if not folder.is_dir():
        return []
    return sorted(f"{p.name}/" if p.is_dir() else p.name for p in folder.iterdir())


def examiner_extension(root: ExamRoot,
                       task_ids: Optional[Iterable[str]] = None) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over exam/, domain tools, prompt, hooks.

    The prompt names the root as "." and lists what it holds, computed once here (F18, F25).
    """

    def setup(api: ExtensionAPI) -> None:
        register_base_tools(api, root.exam_dir, only=BASE_ONLY)
        for tool in domain_tools(root):
            api.register_tool(tool)
        for name, text in prompt_mod.sections(rulings_line(root, task_ids), root_listing(root.exam_dir)):
            api.add_prompt_section(f"examiner_{name}", prompt_block(name, text))
        api.tool_call(refuse_paths(
            names_protected_path, "under the gates or the Runner, which no agent writes (D122)",
            "examiner_protected_paths"))
        api.tool_call(_refuse_forbidden_paths())
        api.tool_call(_refuse_outside_roots())
        api.tool_result(gate_writes(root=root.exam_dir, workdir=root.workdir,
                                    evidence=evidence_of(root)))
        api.tool_result(_result_protection(api))

    return setup


def select_for_session(task_ids: Iterable[str], replays: dict, rerolls: dict,
                       status: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """Which Tasks the model session sees, and the rest with why they were left out (F24).

    A Task is examined when it has a finished Run and a confirmed Reference; a session over
    anything else spends and has nothing to read.
    """
    selected: list[str] = []
    left_out: list[tuple[str, str]] = []
    for task_id in sorted(set(task_ids)):
        if not finished_run_ids(task_id, replays, rerolls):
            left_out.append((task_id, NO_FINISHED_RUN))
        elif not (status.get(task_id) or {}).get("reference_confirmed"):
            left_out.append((task_id, NO_CONFIRMED_REFERENCE))
        else:
            selected.append(task_id)
    return selected, left_out


def left_out_finding(left_out: list[tuple[str, str]], examined: int) -> Finding:
    """The note naming the Tasks the session did not see and why, one row each (F24)."""
    named = ", ".join(f"{task_id} ({reason})" for task_id, reason in left_out[:LEFT_OUT_SHOWN])
    rest = len(left_out) - LEFT_OUT_SHOWN
    named += f" and {rest} more" if rest > 0 else ""
    opened = (f"the Examiner session read {examined} Tasks" if examined
              else "no Examiner session opened")
    return Finding(kind="other", source="derive",
                   text=f"{len(left_out)} Tasks left out of the Examiner session: {named}",
                   rows=[{"task_id": task_id, "left_out": reason} for task_id, reason in left_out],
                   change=(f"{opened}; {len(left_out)} Tasks left out: {named}; replay or run them "
                           "until a Run finishes on a confirmed Reference, then examine again"))


def load_store(workdir: Any) -> dict:
    """The derivation's store off the workdir files, read with read_json (D123)."""
    root = Path(workdir)
    tasks = _load_tasks(root)
    sigs = _load_records(root / "tool_sigs.json", ToolSig, [])
    constraints = _load_records(root / "constraints.json", Constraint, [])
    store = {"tasks": tasks, "sigs": sigs, "constraints": constraints,
             "canon_rules": read_json(root / "canon-rules.json", {}) or {},
             "replays": read_json(root / "replays.json", {}) or {},
             "rerolls": read_json(root / "rerolls.json", {}) or {},
             "intents": _load_keyed(root / "intents", "intents.json"),
             "user_rules": _load_user_rules(root),
             "traces": [], "assisted_tools": [],
             "tool_fidelity": read_json(root / "tool_fidelity.json", {}) or {}}
    missing = stage_mod.missing_inputs(store)
    if missing:
        raise ValueError(f"the workdir holds nothing to derive from: missing {', '.join(missing)}")
    return store


def _load_user_rules(root: Path) -> dict:
    """The user rules by run id as records, skipping what does not parse.

    The derivation's leak gate scores the rules as a record; raw dicts off disk would fail
    there rather than here, so invalid entries are left out the way invalid tasks are.
    """
    out: dict[str, Any] = {}
    for key, body in _load_keyed(root / "user_rules", "user_rules.json").items():
        try:
            out[key] = UserRules.model_validate(body)
        except ValueError:
            continue
    return out


class _FileAnchor:
    """anchor.json as the derivation takes it: held-out Runs never seed a Reference (D81)."""

    def __init__(self, body: dict):
        self._held_out = body.get("held_out", {}) or {}

    def seed_runs(self, task_id: str, run_ids: Any) -> list:
        """What may be built from: every Run of the Task except its anchor."""
        held = set(self._held_out.get(task_id, []))
        return [run_id for run_id in run_ids if run_id not in held]


def _load_anchor(workdir: Any) -> Optional[_FileAnchor]:
    """The workdir's anchor when one was chosen, else nothing to exclude (D81)."""
    try:
        body = read_json(Path(workdir) / "anchor.json", None)
    except (OSError, ValueError):
        return None
    return _FileAnchor(body) if isinstance(body, dict) else None


def _load_tasks(root: Path) -> list[Task]:
    out = []
    folder = root / "tasks"
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                out.append(Task.model_validate(read_json(path)))
            except ValueError:
                continue
    if not out:
        raw = read_json(root / "tasks.json", {}) or {}
        for row in raw.get("tasks", []) if isinstance(raw, dict) else []:
            try:
                out.append(Task.model_validate(row))
            except ValueError:
                continue
    return out


def _load_records(path: Path, model: Any, default: Any) -> Any:
    raw = read_json(path, None)
    if raw is None:
        return default
    rows = raw if isinstance(raw, list) else [raw]
    out = []
    for row in rows:
        try:
            out.append(model.model_validate(row))
        except ValueError:
            continue
    return out


def _load_keyed(folder: Path, single: str) -> dict:
    out: dict[str, Any] = {}
    root = folder.parent
    raw = read_json(root / single, None)
    if isinstance(raw, dict):
        out.update(raw)
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            body = read_json(path, None)
            if isinstance(body, dict):
                out[body.get("task_id") or body.get("trace_id") or path.stem] = body
    return out


def task_runs_of(replays: dict, rerolls: dict, *, workdir: Any) -> dict[str, list]:
    """Every Run on disk per Task, loaded tolerantly: a missing file is skipped (D133).

    A row's path is resolved against the workdir (records.run_path)."""
    out: dict[str, list] = {}
    for task_id, rows in sorted((replays or {}).items()):
        for _, row in sorted((rows or {}).items()):
            _append_run(out, task_id, row, workdir)
    for task_id, rows in sorted((rerolls or {}).items()):
        for row in rows or []:
            _append_run(out, task_id, row, workdir)
    return out


def _append_run(out: dict, task_id: str, row: Any, workdir: Any) -> None:
    stored = row.get("path") if isinstance(row, dict) else None
    path = run_path(workdir, stored) if stored else None
    if path is None or not path.is_file():
        return
    try:
        run = load_run(str(path))
    except (OSError, ValueError, TypeError):
        return
    if any(r.run_id == run.run_id for r in out.get(task_id, [])):
        return
    out.setdefault(task_id, []).append(run)


def derive_findings(workdir: Any, store: dict) -> list[Finding]:
    """What the records already say, as Findings with rows and source derive (D79, D133, D171)."""
    root = Path(workdir)
    status = read_json(root / "task_status.json", {}) or {}
    fidelity = read_json(root / "tool_fidelity.json", {}) or {}
    replays = read_json(root / "replays.json", {}) or {}
    rerolls = read_json(root / "rerolls.json", {}) or {}
    references = read_json(root / "references.json", {}) or {}
    rows = findings_mod.assisted_tool_rows(status, fidelity)
    rows += findings_mod.suite_rows(status)
    rows += findings_mod.fidelity_rows(status, fidelity, replays)
    rows += findings_mod.disagreement_rows(status, references)
    rows += findings_mod.false_rejection_rows({
        "verifiers": [Verifier.model_validate(read_json(path))
                      for path in sorted((root / "verifiers").glob("*.json"))] if (root / "verifiers").is_dir() else [],
        "sigs": store.get("sigs") or [], "replays": replays, "rerolls": rerolls,
        "task_status": status, "canon_rules": rules_of({"canon_rules": store.get("canon_rules")}),
        "task_runs": task_runs_of(replays, rerolls, workdir=root)})
    return [finding_from_row(row) for row in rows]


def finding_from_row(row: dict) -> Finding:
    """One rule row as a Finding: the env file and what differs, never a verb (D123)."""
    task_ids = [str(t) for t in row.get("task_ids") or []]
    task_id = row.get("task_id") or (task_ids[0] if len(task_ids) == 1 else None)
    kind = str(row.get("kind") or "other")
    tool = row.get("tool")
    hint = _one_line(str(row.get("hint") or ""))
    if kind in ("fidelity", "assisted_tool") and tool:
        path = f"env/tools/{tool}.py"
        change = hint or f"the body of {tool} answers its recorded calls"
    else:
        path, change = "", hint
    return Finding(task_id=task_id, kind=kind, text=str(row.get("text") or ""),
                   rows=_evidence_rows(kind, row, task_ids), path=path, change=change,
                   source="derive")


def _evidence_rows(kind: str, row: dict, task_ids: list[str]) -> list[dict]:
    if kind == "suite":
        check = str(row.get("key") or "").split(":")[1] if ":" in str(row.get("key") or "") else ""
        return [{"task_id": task_id, "check": check, "passed": False} for task_id in task_ids]
    if kind in ("fidelity", "assisted_tool"):
        return [{"task_id": task_id, "tool": row.get("tool"),
                 "detail": _one_line(str(row.get("hint") or ""))} for task_id in task_ids]
    if kind == "false_rejection":
        return [{"task_id": task_id, "run_id": row.get("run_id")} for task_id in task_ids]
    return [{"task_id": task_id, "reason": _one_line(str(row.get("hint") or ""))}
            for task_id in task_ids]


def _one_line(text: str, limit: int = HINT_CHARS) -> str:
    return " ".join(text.split())[:limit]


def examine(workdir: Any, *, task_ids: Optional[Iterable[str]] = None, model: Any,
            judge_model: Any = None, probe_model: Any = None, reroll_model: Any = None,
            allowance_usd: Optional[float] = None, session_path: Any = None,
            subscribers: Iterable[Callable] = (), max_turns: int = EXAMINE_MAX_TURNS) -> list[Finding]:
    """Derive the Verifiers by code, file what the records say, then run one model session.

    With `model=None` the session is code only: the derivation and its findings. With a model,
    one message over the Examiner's root, capped at `max_turns`; the Builder decides when to
    call again. A session stopped on the cap adds one finding saying so, so the Builder reads the
    rest as partial. Returns every finding, derive-filed and model-filed, and writes findings.json.
    The session seeds one Verifier history per Task at version 1 from the derived files, so the
    trusted and loosening gates rule on proposals (D127).

    The probe, re-roll and variant runners come from the runner tool over the workdir (D120):
    without a re-roll model the derivation stays code only and buys no Runs.

    The session sees only the Tasks with a finished Run and a confirmed Reference; one finding
    of kind other names the rest and why, and when none is left no session opens at all (F24).
    """
    root = Path(workdir)
    task_ids = list(task_ids) if task_ids is not None else None
    expose(workdir)
    store = load_store(workdir)
    anchor = _load_anchor(root)
    ctx = stage_mod.ExamContext(root, GateLedger(root), anchor=anchor)
    runners = runners_mod.runners_for(root, reroll_model=reroll_model, anchor=anchor)
    stage_mod.derive_all(ctx, store, probe_model=probe_model, judge_model=judge_model,
                         run_probe=runners["run_probe"],
                         run_rerolls=runners["run_rerolls"] if reroll_model is not None else None,
                         run_variant=runners["run_variant"], round_number=0)
    findings = derive_findings(workdir, store)
    if task_ids is not None:
        wanted = set(task_ids)
        findings = [f for f in findings if f.task_id in wanted or
                    any(str(r.get("task_id")) in wanted for r in f.rows)]
    write_json(root / "findings.json", [f.as_dict() for f in findings])
    if model is None:
        return findings
    candidates = task_ids if task_ids is not None else {
        *(t.id for t in store.get("tasks") or []), *(store.get("replays") or {}),
        *(store.get("rerolls") or {})}
    status = read_json(root / "task_status.json", {}) or {}
    selected, left_out = select_for_session(candidates, store.get("replays") or {},
                                            store.get("rerolls") or {},
                                            status if isinstance(status, dict) else {})
    note = [left_out_finding(left_out, len(selected))] if left_out else []
    if not selected:
        out = findings + note
        write_json(root / "findings.json", [f.as_dict() for f in out])
        return out
    exam_root = _exam_root(workdir, store, findings, reroll_model=reroll_model,
                           probe_model=probe_model, run_probe=runners["run_probe"],
                           allowance_usd=allowance_usd)
    harness = AgentHarness(model=model, max_turns=max_turns,
                           session=SessionStore.load(session_path) if session_path is not None else None,
                           context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
                           bus=Bus(root / "bus.jsonl", agent="examiner"))
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    harness.subscribe(budget.subscriber(root, "examiner", getattr(model, "name", None)))
    load_extensions(harness, [examiner_extension(exam_root, selected)])

    capped: list[bool] = []

    async def go() -> None:
        async for event in harness.prompt(EXAMINE_MESSAGE):
            if getattr(event, "type", None) == "turn_end" and _stopped_on_cap(event.message):
                capped.append(True)

    asyncio.run(go())
    out = list(exam_root.findings) + note + ([turns_ran_out(max_turns)] if capped else [])
    write_json(root / "findings.json", [f.as_dict() for f in out])
    return out


def _stopped_on_cap(message: Any) -> bool:
    """Did this turn end because the harness reached the session's turn cap?"""
    return str(getattr(message, "error_message", None) or "").startswith(CAP_STOP)


def turns_ran_out(max_turns: int) -> Finding:
    """The note a session stopped on its cap files: the findings beside it are partial (F17)."""
    return Finding(kind="other", source="derive",
                   text=f"the Examiner ran out of turns at {max_turns} before it finished reading",
                   rows=[{"max_turns": max_turns}],
                   change=(f"the Examiner ran out of turns at {max_turns}, so these findings are "
                           "partial; call examine again to continue"))


def _exam_root(workdir: Any, store: dict, findings: list[Finding], reroll_model: Any = None,
               probe_model: Any = None, run_probe: Any = None,
               allowance_usd: Optional[float] = None) -> ExamRoot:
    """The session root: live Verifiers, signatures, rules, rows, task status and version history.

    Task status is what derive_all wrote to task_status.json in this same examine call, and the
    history is the exam history.json with every live Verifier seeded as version 1 by derive and
    saved. The trusted gate reads each derived file as the current accepted version from that
    seed, and the loosening gate compares each proposal against the version before it (D127).
    The probe model and the probe runner examine built let the suite run its loophole check on
    each proposal (D79).
    """
    root = Path(workdir)
    verifiers: dict[str, Verifier] = {}
    if (root / "verifiers").is_dir():
        for path in sorted((root / "verifiers").glob("*.json")):
            try:
                verifier = Verifier.model_validate(read_json(path))
            except ValueError:
                continue
            verifiers[verifier.task_id] = verifier
    status = read_json(root / "task_status.json", None)
    exam_root = ExamRoot(workdir=root, verifiers=verifiers, sigs=list(store.get("sigs") or []),
                         canon_rules=rules_of({"canon_rules": store.get("canon_rules")}),
                         replays=store.get("replays") or {}, rerolls=store.get("rerolls") or {},
                         task_status=status if isinstance(status, dict) else {},
                         findings=list(findings), reroll_model=reroll_model,
                         probe_model=probe_model, run_probe=run_probe,
                         allowance_remaining=allowance_usd)
    history = load_history(exam_root)
    for task_id, verifier in verifiers.items():
        seeded_history(history, task_id, verifier)
    exam_root.history = history
    save_history(exam_root, history)
    return exam_root
__all__ = ["BASE_ONLY", "EXAMINE_MESSAGE", "derive_findings", "examine", "examiner_extension",
           "finding_from_row", "left_out_finding", "load_store", "root_listing", "rulings_line",
           "select_for_session", "task_runs_of", "turns_ran_out"]
