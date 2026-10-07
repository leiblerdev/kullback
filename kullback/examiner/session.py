"""The Examiner as an extension on the core, called by the Builder (D79, D111, D122-D124, D230, D320).

`examine` chooses each Task's References by code (the Reference stage, which writes no Verifier:
the Spec writes every Verifier), files what the records already say as findings with rows, then
runs one review session over the Examiner's root when a model is given. The session reads, files
rulings on the Specs and findings on the Environment; it edits nothing (D331). It returns the findings
and writes them to `findings.json` at once, never a round late (learnings 7).

`review_specs` is one Examiner round of the review loop (spec/rounds.py) alone: the Spec views, the
ruling tools, no Reference stage and no Environment findings.
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
from kullback.examiner import stage as stage_mod
from kullback.examiner.domain_tools import TOOL_NAMES, domain_tools, note_line, notes_of
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
NO_SPEC = "no Spec"


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


EXAMINER_WRITES = ("the Examiner writes nothing: it files rulings the writer applies or rebuts, and findings "
                   "for the Builder (D320, D331)")


def _refuse_generic_writes() -> Callable:
    """A tool_call hook refusing write and edit everywhere, so a model that asks for them hears why.

    The Examiner has no write and no edit: it reads and reviews, and the Spec writes (D320).
    """

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) in WRITE_TOOLS:
            raise PermissionError(EXAMINER_WRITES)
        return None

    refuse.hook_name = "examiner_writes_through_its_tools"  # type: ignore[attr-defined]
    return refuse


def unsourced_cells(verifier: Optional[Verifier]) -> list[str]:
    """The cells of the Verifier's expected end states with no source for the value or for the row."""
    if verifier is None:
        return []
    return sorted({".".join(str(part) for part in (cell.table, cell.row_id, cell.field) if part is not None)
                   for state in verifier.expected for cell in state.cells
                   if cell.source is None or cell.row_source is None})


def _refuse_no_finding_on_unsourced(root: ExamRoot) -> Callable:
    """A tool_call hook refusing no_finding on a Task whose Verifier holds an unsourced cell: the review
    has something to say there (40 no-finding reviews sat on such Tasks)."""

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) != "no_finding":
            return None
        task_id = str((getattr(call, "arguments", None) or {}).get("task_id") or "")
        cells = unsourced_cells(root.current(task_id))
        if cells:
            raise PermissionError(
                f"task {task_id} has unsourced cells: {', '.join(cells)}; file a ruling on the item instead")
        return None

    refuse.hook_name = "examiner_no_finding_on_unsourced"  # type: ignore[attr-defined]
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
    """The Task's Verifier (the Spec's) and its user rules, as paths under the root (F33)."""
    derived = f"{DERIVED_DIR}/{task_id}.json"
    parts = [f"verifier: {derived}" if (root.exam_dir / derived).is_file() else "verifier: none copied"]
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


def examiner_extension(root: ExamRoot, task_ids: Optional[Iterable[str]] = None) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over exam/, domain tools, prompt, hooks.

    The prompt names the root as "." (F18); what the root holds and the rulings (F25) are this
    session's own facts, so `examine` says them in the opening message (`session_opening`).
    """

    def setup(api: ExtensionAPI) -> None:
        register_base_tools(api, root.exam_dir, only=BASE_ONLY)
        for tool in domain_tools(root):
            api.register_tool(tool)
        for name, text in prompt_mod.sections(TOOL_NAMES):
            api.add_prompt_section(f"examiner_{name}", prompt_block(name, text))
        api.tool_call(refuse_paths(
            names_protected_path, "under the gates or the Runner, which no agent writes (D122)",
            "examiner_protected_paths"))
        api.tool_call(_refuse_forbidden_paths())
        api.tool_call(_refuse_generic_writes())
        api.tool_call(_refuse_no_finding_on_unsourced(root))

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

    With task_ids, those that name a Task. Without, every Task the Reference stage has not settled: it
    has no status row because no call has read it, or references.json keeps no Reference for it and
    either its row holds a confirmed Reference or replays.json now holds a confirmed replay for it,
    read the way select_for_session reads one, so a Task left unconfirmed is revisited once evidence
    confirms it."""
    known = sorted(task.id for task in store.get("tasks") or [])
    if task_ids is not None:
        wanted = set(task_ids)
        return [task_id for task_id in known if task_id in wanted]
    root = Path(workdir)
    status = read_json(root / "task_status.json", {}) or {}
    status = status if isinstance(status, dict) else {}
    replays = store.get("replays") or {}
    references = read_json(root / "references.json", {}) or {}

    def pending(task_id: str) -> bool:
        if task_id not in status:
            return True
        if ((references.get(task_id) or {}) if isinstance(references, dict) else {}).get("references"):
            return False
        return bool((status[task_id] or {}).get("reference_confirmed")
                    or finished_run_ids(task_id, replays, {}))

    return [task_id for task_id in known if pending(task_id)]


def spec_owned(store: dict) -> list[str]:
    """Every Task of the store: the Spec writes each Verifier, so the Reference stage writes none (D320)."""
    return [task.id for task in store.get("tasks") or []]


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
    hint = _one_line(str(row.get("change") or ""))
    if kind in ("fidelity", "assisted_tool") and tool:
        path = f"env/tools/{tool}.py"
        change = hint or f"the body of {tool} answers its recorded calls"
    else:
        path, change = "", hint
    return Finding(task_id=task_id, kind=kind, text=str(row.get("text") or ""),
                   rows=_evidence_rows(kind, row, task_ids), path=path, change=change,
                   source="derive", edits=list(row.get("edits") or []))


def _evidence_rows(kind: str, row: dict, task_ids: list[str]) -> list[dict]:
    if kind == "suite":
        check = str(row.get("key") or "").split(":")[1] if ":" in str(row.get("key") or "") else ""
        return [{"task_id": task_id, "check": check, "passed": False} for task_id in task_ids]
    if kind in ("fidelity", "assisted_tool"):
        return [{"task_id": task_id, "tool": row.get("tool"),
                 "detail": _one_line(str(row.get("change") or ""))} for task_id in task_ids]
    if kind == "false_rejection":
        return [{"task_id": task_id, "run_id": row.get("run_id")} for task_id in task_ids]
    return [{"task_id": task_id, "reason": _one_line(str(row.get("change") or ""))}
            for task_id in task_ids]


def _one_line(text: str, limit: int = HINT_CHARS) -> str:
    return " ".join(text.split())[:limit]


def examine(workdir: Any, *, task_ids: Optional[Iterable[str]] = None, model: Any,
            judge_model: Any = None, allowance_usd: Optional[float] = None, session_path: Any = None,
            subscribers: Iterable[Callable] = (), max_turns: int = EXAMINE_MAX_TURNS,
            workers: Optional[int] = None, limit: Optional[int] = None) -> list[Finding]:
    """Choose References by code, file what the records say, then run one review session.

    With `model=None` the call is code only: the Reference stage and its findings. The Reference
    stage writes references.json, task_status.json and constraints_check.json and no Verifier: the
    Spec writes every Verifier (D320), so every Task's verifiers/ file is kept as the Spec left it.
    It runs no probe, buys no Run and synthesises no variant.

    With a model, one review session over the Examiner's root, capped at `max_turns`, over the Tasks
    with a Spec; one finding of kind other names the rest. The session holds the ruling tools,
    finding, no_finding and check_reference; the writer answers its rulings in the review loop. A session stopped on the cap adds one finding saying so. Returns every finding and
    writes findings.json.

    Every Task the pick returns goes through the Reference stage in this call, `workers` at a time;
    `limit`, when given, takes only the first that many and files one note naming the rest (F40).
    """
    root = Path(workdir)
    task_ids = list(task_ids) if task_ids is not None else None
    store = load_store(workdir)
    anchor = _load_anchor(root)
    ctx = stage_mod.ExamContext(root, GateLedger(root), anchor=anchor, kept=spec_owned(store))
    picked = derive_pick(root, store, task_ids)
    now, later = (picked, []) if limit is None else (picked[:limit], picked[limit:])
    if now:
        stage_mod.derive_all(ctx, store, judge_model=judge_model, round_number=0, only=now,
                             workers=workers if workers is not None else default_workers())
    expose(workdir)
    findings = derive_findings(workdir, store) + ([not_derived_finding(later, limit)] if later else [])
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
    selected, left_out = _session_tasks(candidates, root, store)
    note = [left_out_finding(left_out, len(selected))] if left_out else []
    if not selected:
        out = findings + note
        write_json(root / "findings.json", [f.as_dict() for f in out])
        return out
    exam_root = _exam_root(workdir, store, findings, allowance_usd=allowance_usd)
    harness = AgentHarness(model=model, max_turns=max_turns,
                           session=SessionStore.load(session_path) if session_path is not None else None,
                           context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
                           bus=_exam_bus(exam_root))
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    harness.subscribe(budget.subscriber(root, "examiner", getattr(model, "name", None)))
    load_extensions(harness, [examiner_extension(exam_root, selected)])
    opening_message, detach = _opening(exam_root, selected, model, allowance_usd)
    try:
        cap_notes = _run_session(harness, opening_message, max_turns, selected)
    finally:
        detach()
    out = list(exam_root.findings) + note + cap_notes
    write_json(root / "findings.json", [f.as_dict() for f in out])
    return out


def _exam_bus(exam_root: ExamRoot) -> Bus:
    """The one bus of the session: the harness's events and the rule tool's rulings on one log."""
    exam_root.bus = Bus(Path(exam_root.workdir) / "bus.jsonl", agent="examiner")
    return exam_root.bus


def _session_tasks(candidates: Iterable[str], root: Path, store: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """The Tasks the session reviews: a finished Run, a confirmed Reference and a Spec (F24, D320);
    the rest left out with why."""
    from kullback.spec.schema import spec_path

    status = read_json(root / "task_status.json", {}) or {}
    ready, left_out = select_for_session(candidates, store.get("replays") or {},
                                         store.get("rerolls") or {}, status)
    return ([t for t in ready if spec_path(root, t).is_file()],
            left_out + [(t, NO_SPEC) for t in ready if not spec_path(root, t).is_file()])


def intent_line(root: ExamRoot, task_id: str, view: Optional[str]) -> str:
    """One Task's line: its spec view, its rulings so far, its Runs and its spoken file, as root paths."""
    return (f"{task_id}: spec view: {view or 'none'}; {rulings_part(root.workdir, task_id)}; "
            f"{_runs_part(root, task_id)}; {_spoken_part(root, task_id)}")


def rulings_part(workdir: Any, task_id: str) -> str:
    """The Task's rulings in a few words: how many, how many open, and the writer's answers to close or keep."""
    from kullback.spec.rulings import load_rulings

    rulings = load_rulings(workdir, task_id)
    if not rulings:
        return "rulings: none"
    answered = [f"{r.number} {r.answer['action']}" for r in rulings if r.status == "open" and r.answer]
    unruled = [str(r.number) for r in rulings
               if r.status == "open" and r.code == "fails_reference" and r.wrong_side is None]
    text = f"rulings: {len(rulings)}, {sum(r.status == 'open' for r in rulings)} open"
    text += f"; answered, to close or keep open: {', '.join(answered)}" if answered else ""
    return text + (f"; the Reference fails the Verifier, rule which side is wrong: {', '.join(unruled)}"
                   if unruled else "")


def _opening(exam_root: ExamRoot, selected: list[str], model: Any = None,
             ceiling_usd: Optional[float] = None) -> tuple[str, Callable[[], None]]:
    """The opening message with the spec views written. Nothing moves on a ruling inside the session:
    the writer answers rulings in its own round (spec/rounds.py, D331)."""
    from kullback.examiner import rule_tool

    views = rule_tool.write_views(exam_root, selected)
    lines = "\n".join(intent_line(exam_root, task_id, views.get(task_id)) for task_id in selected)
    text = prompt_mod.opening(EXAMINE_MESSAGE, lines, root_listing(exam_root.exam_dir),
                              notes_line(exam_root, selected), exam_root.round or 1)
    return text, lambda: None


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
TASK_TOOLS = ("finding", "no_finding", "rule", "close", "verify")


def _last_task(messages: Iterable[Any], selected: list[str]) -> Optional[str]:
    """The task_id of the session's last review call, else the first selected Task."""
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


def _exam_root(workdir: Any, store: dict, findings: list[Finding],
               allowance_usd: Optional[float] = None) -> ExamRoot:
    """The session root: live Verifiers, signatures, rules, rows, task status and version history.

    Task status is what the Reference stage wrote to task_status.json in this same examine call; the
    re-roll rows join the workdir's own. The root holds what the Examiner reads; it runs nothing.
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
                         findings=list(findings), allowance_remaining=allowance_usd)
    history = load_history(exam_root)
    for task_id, verifier in verifiers.items():
        seeded_history(history, task_id, verifier)
    exam_root.history = history
    save_history(exam_root, history)
    return exam_root
REVIEW_MESSAGE = "Examine these Specs."


def review_extension(root: ExamRoot) -> Callable[[ExtensionAPI], None]:
    """One review round's setup: base tools over exam/, the ruling tools, the prompt, the read and write hooks."""
    from kullback.examiner.rule_tool import rule_tools

    def setup(api: ExtensionAPI) -> None:
        register_base_tools(api, root.exam_dir, only=BASE_ONLY)
        for tool in rule_tools(root):
            api.register_tool(tool)
        for name, text in prompt_mod.sections(prompt_mod.REVIEW_TOOLS):
            api.add_prompt_section(f"examiner_{name}", prompt_block(name, text))
        api.tool_call(refuse_paths(
            names_protected_path, "under the gates or the Runner, which no agent writes (D122)",
            "examiner_protected_paths"))
        api.tool_call(_refuse_forbidden_paths())
        api.tool_call(_refuse_generic_writes())

    return setup


def review_specs(workdir: Any, task_ids: Iterable[str], *, model: Any, round_number: int,
                 max_turns: Optional[int] = None, subscribers: Iterable[Callable] = (),
                 session_path: Any = None) -> dict:
    """One Examiner round over the Tasks with a Spec: views written, rulings filed or closed, Specs synced.

    Returns the Tasks examined, whether the session stopped on its turn cap, and the rulings this round
    filed, by kind, code and blocking (spec/rulings.py counts).
    """
    from kullback.examiner.rule_tool import write_views
    from kullback.spec.rulings import counts, load_rulings, sync_spec
    from kullback.spec.schema import spec_path

    root = Path(workdir)
    selected = sorted(t for t in set(task_ids) if spec_path(root, t).is_file())
    if not selected:
        return {"round": round_number, "tasks": [], "capped": False, "counts": counts(())}
    exam_root = ExamRoot(workdir=root, round=round_number,
                         replays=read_json(root / "replays.json", {}) or {},
                         rerolls=read_json(root / "rerolls.json", {}) or {})
    exam_root.bus = Bus(root / "bus.jsonl", agent="examiner")
    turns = max_turns or 8 * len(selected) + 10
    harness = AgentHarness(model=model, max_turns=turns,
                           session=SessionStore.load(session_path) if session_path is not None else None,
                           context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
                           bus=exam_root.bus)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    harness.subscribe(budget.subscriber(root, "examiner", getattr(model, "name", None)))
    load_extensions(harness, [review_extension(exam_root)])
    views = write_views(exam_root, selected)
    lines = "\n".join(f"{t}: view {views.get(t, 'none')}; {rulings_part(root, t)}" for t in selected)
    entries = [f"review/{entry}" for entry in root_listing(exam_root.exam_dir / "review")]
    text = prompt_mod.opening(REVIEW_MESSAGE, lines, entries, round_number=round_number)
    capped = _run_session(harness, text, turns, selected)
    filed = [r for t in selected for r in load_rulings(root, t) if r.round == round_number]
    for task_id in selected:
        sync_spec(root, task_id, round_number)
    return {"round": round_number, "tasks": selected, "capped": bool(capped), "counts": counts(filed)}


__all__ = ["BASE_ONLY", "EXAMINE_MESSAGE", "REVIEW_MESSAGE", "review_extension", "review_specs", "rulings_part", "derive_findings", "examine", "examiner_extension", "session_opening",
           "finding_from_row", "left_out_finding", "load_store", "notes_line", "root_listing", "rulings_line",
           "select_for_session", "spec_owned", "task_runs_of", "turns_ran_out"]


def examine_rounds(workdir: Any, task_ids: Iterable[str], *, model: Any, writer_model: Any = None,
                   rounds: Optional[int] = None, ceiling_usd: Optional[float] = None) -> dict:
    """The review loop over the Tasks with the Examiner's round wired in (spec/rounds.py, D331)."""
    from kullback.spec.rounds import run_rounds
    from kullback.spec.router import ROUNDS_CAP

    return run_rounds(workdir, task_ids, examine=review_specs, model=model, writer_model=writer_model,
                      rounds=rounds or ROUNDS_CAP, ceiling_usd=ceiling_usd)
