"""The Examiner as an extension on the core, called by the Builder (D79, D111, D122-D124, D128, D230).

`examine` derives one Verifier per Task from its References through the suite by code,
files what the records already say as findings with rows, then runs one model session
over the Examiner's root when a model is given. It returns the findings and writes them
to `findings.json` at once, never a round late (learnings 7).
"""

from __future__ import annotations

import asyncio
import os
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
from kullback.examiner.domain_tools import domain_tools, note_line, notes_of
from kullback.examiner.exam_files import (
    DERIVED_DIR,
    SPOKEN_DIR,
    ExamRoot,
    Finding,
    expose,
    load_history,
    names_forbidden_path,
    save_history,
    seeded_history,
)
from kullback.examiner.plan import merged_rerolls
from kullback.gates import names_protected_path
from kullback.gates.hook import WRITE_TOOLS
from kullback.gates.ledger import GateLedger
from kullback.gates.loosening import finished_run_ids
from kullback.gates.verifier_suite import load_run
from kullback.runner import budget
from kullback.runner.canon import load_rules, rules_of
from kullback.runner.records import Constraint, Task, ToolSig, UserRules, Verifier, read_json, run_path, write_json

BASE_ONLY = ("read", "grep", "find", "ls", "inspect", "web_search")
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
# How many Tasks the derivation derives at once when the caller names no number: every picked Task
# is derived in one call, so one Task at a time left most of a build underived (F40, F55).
MAX_DERIVE_WORKERS = 8
NOT_DERIVED_SHOWN = 10
NO_FINISHED_RUN = "no finished Run"
NO_CONFIRMED_REFERENCE = "no confirmed Reference"


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


EXAMINER_WRITES = ("the Examiner writes Verifiers through edit_verifier and probes through probe, "
                   "not with write or edit")


def _refuse_generic_writes() -> Callable:
    """A tool_call hook refusing write and edit everywhere, so a model that asks for them hears why.

    The Examiner has no write and no edit: a generic write under verifiers/ would land before the
    gates ruled and stay live when they refused it; propose_verifier and probe run the gates first.
    """

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) in WRITE_TOOLS:
            raise PermissionError(EXAMINER_WRITES)
        return None

    refuse.hook_name = "examiner_writes_through_its_tools"  # type: ignore[attr-defined]
    return refuse


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
    parts.append(_spoken_part(root, task_id))
    return "; ".join(parts)


def _spoken_part(root: ExamRoot, task_id: str) -> str:
    """The Task's spoken file under its name on disk, extension and all, so no read guesses it (F35)."""
    folder = root.exam_dir / SPOKEN_DIR
    names = sorted(p.name for p in folder.glob(f"{task_id}.*") if p.is_file() and p.stem == task_id) \
        if folder.is_dir() else []
    return f"spoken: {SPOKEN_DIR}/{names[0]}" if names else "spoken: none"


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


def session_opening(root: ExamRoot, task_ids: Optional[Iterable[str]] = None) -> str:
    """The Examiner's opening message: the ask, what its root holds, the rulings to answer (F25),
    and the Builder's notes on these Tasks, each open or with its ruling."""
    return prompt_mod.opening(EXAMINE_MESSAGE, rulings_line(root, task_ids), root_listing(root.exam_dir),
                              notes_line(root, task_ids))


def notes_line(root: ExamRoot, task_ids: Optional[Iterable[str]] = None) -> str:
    """One line per Builder note on the Tasks the session examines, open or with its ruling."""
    wanted = set(task_ids) if task_ids is not None else None
    return "\n".join(note_line(task_id, note, ruling) for task_id, (note, ruling) in sorted(notes_of(root.workdir).items())
                     if wanted is None or task_id in wanted)


def examiner_extension(root: ExamRoot,
                       task_ids: Optional[Iterable[str]] = None) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over exam/, domain tools, prompt, hooks.

    The prompt names the root as "." (F18); what the root holds and the rulings (F25) are this
    session's own facts, so `examine` says them in the opening message (`session_opening`).
    """

    def setup(api: ExtensionAPI) -> None:
        register_base_tools(api, root.exam_dir, only=BASE_ONLY)
        for tool in domain_tools(root):
            api.register_tool(tool)
        for name, text in prompt_mod.sections():
            api.add_prompt_section(f"examiner_{name}", prompt_block(name, text))
        api.tool_call(refuse_paths(
            names_protected_path, "under the gates or the Runner, which no agent writes (D122)",
            "examiner_protected_paths"))
        api.tool_call(_refuse_forbidden_paths())
        api.tool_call(_refuse_generic_writes())

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


def derive_pick(workdir: Any, store: dict, task_ids: Optional[list[str]]) -> list[str]:
    """The Tasks this examine call is about, in id order (F40).

    With task_ids, those that name a Task. Without, every Task not derived yet: it has no status
    row because no derivation has read it, or its Verifier file is absent and either its row holds
    a confirmed Reference or replays.json now holds a confirmed replay for it, read the way
    select_for_session reads one, so a Task left unconfirmed is revisited once evidence confirms it.
    A row a stop cut short is pending whatever else it holds, so a stop never parks a Task for good."""
    known = sorted(task.id for task in store.get("tasks") or [])
    if task_ids is not None:
        wanted = set(task_ids)
        return [task_id for task_id in known if task_id in wanted]
    root = Path(workdir)
    status = read_json(root / "task_status.json", {}) or {}
    status = status if isinstance(status, dict) else {}
    replays = store.get("replays") or {}

    def pending(task_id: str) -> bool:
        if task_id not in status:
            return True
        if (status[task_id] or {}).get("stopped"):
            return True
        if (root / "verifiers" / f"{task_id}.json").is_file():
            return False
        return bool((status[task_id] or {}).get("reference_confirmed")
                    or finished_run_ids(task_id, replays, {}))

    return [task_id for task_id in known if pending(task_id)]


def default_workers() -> int:
    """How many Tasks derive at once when the caller names no number: the cores, at most eight."""
    return max(1, min(MAX_DERIVE_WORKERS, os.cpu_count() or 1))


def not_derived_finding(later: list[str], limit: Optional[int] = None) -> Finding:
    """The note naming the Tasks past a limit the caller asked for, one row each (F40).

    examine derives every Task it picks unless the caller passes `limit`; only then are Tasks
    left for the next call, and this note says which.
    """
    named = ", ".join(later[:NOT_DERIVED_SHOWN])
    rest = len(later) - NOT_DERIVED_SHOWN
    named += f" and {rest} more" if rest > 0 else ""
    text = f"{len(later)} confirmed Tasks not derived yet: {named}; call examine again"
    why = f"this examine call was limited to {limit} Tasks" if limit is not None else "a limit was set"
    return Finding(kind="other", source="derive", text=text,
                   rows=[{"task_id": task_id, "not_derived": "limit"} for task_id in later],
                   change=f"{why}; {text}")


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
             "canon_rules": load_rules(root / "canon-rules.json"),
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
            subscribers: Iterable[Callable] = (), max_turns: int = EXAMINE_MAX_TURNS,
            workers: Optional[int] = None, limit: Optional[int] = None,
            should_stop: Callable[[], bool] = stage_mod.never_stop) -> list[Finding]:
    """Derive the Verifiers by code, file what the records say, then run one model session.

    With `model=None` the session is code only: the derivation and its findings. With a model,
    one message over the Examiner's root, capped at `max_turns`; the Builder decides when to
    call again. A session stopped on the cap adds one finding saying so, so the Builder reads the
    rest as partial. Returns every finding, derive-filed and model-filed, and writes findings.json.
    The session seeds one Verifier history per Task at version 1 from the derived files, so the
    trusted and loosening gates rule on proposals (D127).

    The probe, re-roll and variant runners come from the runner tool over the workdir (D120):
    without a re-roll model the derivation stays code only and buys no Runs.

    Every Task derive_pick returns is derived in this call, `workers` at a time (default: the
    cores, at most eight); `limit`, when given, derives only the first that many and files one
    note naming the rest (F40, F55). The exam view is copied after the derivation, so the
    session reads this call's Verifiers and task status, never the last call's (F52).

    The session sees only the Tasks with a finished Run and a confirmed Reference; one finding
    of kind other names the rest and why, and when none is left no session opens at all (F24).

    `should_stop` is a person's stop from the Builder's screen, asked at the safe points only:
    before each Task's derivation starts, before each re-roll Run is bought, and on every event of
    the Examiner session, which it cancels at its next step. A stopped call opens no session it had
    not opened yet, files one note saying how far it got, and writes findings.json whole.
    """
    root = Path(workdir)
    task_ids = list(task_ids) if task_ids is not None else None
    store = load_store(workdir)
    anchor = _load_anchor(root)
    ctx = stage_mod.ExamContext(root, GateLedger(root), anchor=anchor)
    runners = runners_mod.runners_for(root, reroll_model=reroll_model, anchor=anchor)
    picked = derive_pick(root, store, task_ids)
    now, later = (picked, []) if limit is None else (picked[:limit], picked[limit:])
    derived = len(now)
    if now:
        result = stage_mod.derive_all(
            ctx, store, probe_model=probe_model, judge_model=judge_model,
            run_probe=runners["run_probe"],
            run_rerolls=runners["run_rerolls"] if reroll_model is not None else None,
            run_variant=runners["run_variant"], round_number=0, only=now,
            workers=workers if workers is not None else default_workers(), should_stop=should_stop)
        derived = result["derived"]
    expose(workdir)
    findings = derive_findings(workdir, store) + ([not_derived_finding(later, limit)] if later else [])
    if task_ids is not None:
        wanted = set(task_ids)
        findings = [f for f in findings if f.task_id in wanted or
                    any(str(r.get("task_id")) in wanted for r in f.rows)]
    if should_stop():
        findings = findings + [stopped_finding(derived, len(now), session="not opened")]
    write_json(root / "findings.json", [f.as_dict() for f in findings])
    if model is None or should_stop():
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
    exam_root.should_stop = should_stop
    harness = AgentHarness(model=model, max_turns=max_turns,
                           session=SessionStore.load(session_path) if session_path is not None else None,
                           context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
                           bus=Bus(root / "bus.jsonl", agent="examiner"))
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    harness.subscribe(budget.subscriber(root, "examiner", getattr(model, "name", None)))
    cancelled = _cancel_on_stop(harness, should_stop)
    load_extensions(harness, [examiner_extension(exam_root, selected)])
    cap_notes = _run_session(harness, session_opening(exam_root, selected), max_turns, selected)
    stop_note = [stopped_finding(derived, len(now), session="cancelled")] if cancelled else []
    out = list(exam_root.findings) + note + cap_notes + stop_note
    write_json(root / "findings.json", [f.as_dict() for f in out])
    return out


def _cancel_on_stop(harness: AgentHarness, should_stop: Callable[[], bool]) -> list[bool]:
    """Cancel the Examiner session on its first event after a stop; the list says whether it did.

    The loop reads its cancel token between steps, so the session ends at its next step and the
    tool already running finishes. A subscriber is enough: the events come on the session's own
    loop, so no thread watches the flag.
    """
    cancelled: list[bool] = []

    def watch(event: Any) -> None:
        if not cancelled and should_stop():
            cancelled.append(True)
            harness.cancel()

    harness.subscribe(watch)
    return cancelled


def stopped_finding(derived: int, tasks: int, *, session: str) -> Finding:
    """The note a stopped examine files: how far the derivation got and what became of the session.

    Its one row carries `stopped`, so the Builder's examine tool leads its result with it. The Tasks
    not reached are left as they were, open, for the next call.
    """
    text = f"examine stopped early: {derived} of {tasks} Tasks derived, Examiner session {session}"
    return Finding(kind="other", source="derive", text=text,
                   rows=[{"stopped": True, "derived": derived, "tasks": tasks, "session": session}],
                   change=f"{text}; call examine again to continue")


def _run_session(harness: AgentHarness, opening_message: str, max_turns: int,
                 selected: list[str]) -> list[Finding]:
    """Play the session on its opening message; the one cap note when it stopped on the turn cap, else none."""
    capped: list[bool] = []

    async def go() -> None:
        async for event in harness.prompt(opening_message):
            if getattr(event, "type", None) == "turn_end" and _stopped_on_cap(event.message):
                capped.append(True)

    asyncio.run(go())
    if not capped:
        return []
    return [turns_ran_out(max_turns, task_id=_last_task(harness.messages, selected),
                          refusal=_last_refusal(harness.messages))]


def _stopped_on_cap(message: Any) -> bool:
    """Did this turn end because the harness reached the session's turn cap?"""
    return str(getattr(message, "error_message", None) or "").startswith(CAP_STOP)


def turns_ran_out(max_turns: int, task_id: Optional[str] = None,
                  refusal: Optional[str] = None) -> Finding:
    """The note a session stopped on its cap files: the findings beside it are partial (F17).

    It names the Task the session was on and the last refusal it read, so the Builder sees where
    the turns went (F36).
    """
    return Finding(kind="other", source="derive", task_id=task_id,
                   text=f"the Examiner ran out of turns at {max_turns} before it finished reading",
                   rows=[{"max_turns": max_turns}],
                   change=(f"the Examiner ran out of turns at {max_turns}, so these findings are "
                           "partial; call examine again to continue"
                           + (f"; last refusal: {refusal}" if refusal else "")))


# The tools whose task_id says which Task a session is working on (F36).
TASK_TOOLS = ("edit_verifier", "propose_verifier", "probe")


def _last_task(messages: Iterable[Any], selected: list[str]) -> Optional[str]:
    """The task_id of the session's last propose or probe call, else the first selected Task."""
    last = None
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.name in TASK_TOOLS and call.arguments.get("task_id"):
                last = str(call.arguments["task_id"])
    return last or (selected[0] if selected else None)


def _last_refusal(messages: Iterable[Any]) -> Optional[str]:
    """The first line of the last tool result the session read as an error, if any."""
    last = None
    for message in messages:
        if getattr(message, "role", None) == "tool" and message.is_error and message.content.strip():
            last = message.content.strip().splitlines()[0]
    return last


def _exam_root(workdir: Any, store: dict, findings: list[Finding], reroll_model: Any = None,
               probe_model: Any = None, run_probe: Any = None,
               allowance_usd: Optional[float] = None) -> ExamRoot:
    """The session root: live Verifiers, signatures, rules, rows, task status and version history.

    Task status is what derive_all wrote to task_status.json in this same examine call, and the
    re-roll rows join the derivation's own (the second path Runs it bought) to the workdir's, and the
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
                         replays=store.get("replays") or {},
                         rerolls=merged_rerolls(store.get("rerolls") or {},
                                                read_json(stage_mod.extra_rerolls_path(root), {}) or {}),
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
__all__ = ["BASE_ONLY", "EXAMINE_MESSAGE", "derive_findings", "examine", "examiner_extension", "session_opening",
           "finding_from_row", "left_out_finding", "load_store", "notes_line", "root_listing", "rulings_line",
           "select_for_session", "stopped_finding", "task_runs_of", "turns_ran_out"]
