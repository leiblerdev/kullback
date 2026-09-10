"""One workdir as the report reads it: every record off disk, nothing decided."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from kullback import claims, difficulty, round_snapshot
from kullback.examiner import lifecycle
from kullback.report.data import ReportData, ScorecardItem, TaskCoverage
from kullback.report.numbers import _by_pair, _percent, assisted_share_from_runs
from kullback.report.pipeline import environment_gate, stage_statuses
from kullback.runner.records import (
    Constraint,
    Environment,
    GateResult,
    RoundRecord,
    Run,
    SetAsideLesson,
    Task,
    TaskOverlay,
    ToolSig,
    Verdict,
    Verifier,
    disagreement_stats,
)

EVENT_TYPES = ("model_call", "tool_call", "tool_result", "user_turn", "error", "stop")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path, unread: Optional[list] = None) -> list[dict]:
    """One JSONL file as rows. A line that does not parse is named, never raised on."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines() if path.is_file() else [], 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            _note(unread, f"{path.name} line {number}: not JSON")
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _relative(folder: Path, path: Path) -> str:
    """The file as the workdir names it, so a reader can go and open it."""
    try:
        return str(path.relative_to(folder.parent))
    except ValueError:
        return path.name


def _note(unread: Optional[list], text: str) -> None:
    if unread is not None and text not in unread:
        unread.append(text)


def _records(folder: Path, model: type, unread: Optional[list] = None) -> list:
    """Every record of one kind under a folder.

    A writer may keep the record beside other data in the same file (compile_env.py writes an
    overlay as {"overlay": ..., "values": ...}), so a file that is not the record itself is searched
    one level down for it. A file that holds neither is skipped and, where every file in the folder
    is meant to be this record, named in `unread` so the counts are never quietly short.
    """
    out = []
    for path in sorted(folder.rglob("*.json")) if folder.is_dir() else []:
        body = _json(path)
        if not isinstance(body, dict):
            _note(unread, f"{_relative(folder, path)}: not a JSON object")
            continue
        for candidate in [body] + [v for v in body.values() if isinstance(v, dict)]:
            try:
                out.append(model.model_validate(candidate))
                break
            except ValidationError:
                continue
        else:
            _note(unread, f"{_relative(folder, path)}: not a {model.__name__} this report can read")
    return out


def run_from_jsonl(path: Path) -> Optional[Run]:
    """One stored Run as loop.py writes it: header lines, one line per event, a footer (D90).

    report.py may not import the Runner (design section 4 item 18), so the shape is read again here;
    tests/test_report.py asserts a Run the loop actually wrote still loads.
    """
    head: dict = {}
    events: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            continue
        if obj.get("type") in EVENT_TYPES:
            events.append(obj)
        else:
            events.extend(obj.pop("events", []) or [])
            head.update(obj)
    head = {k: v for k, v in head.items() if k in Run.model_fields}
    head.setdefault("run_id", path.stem)
    head["events"] = [dict(event, idx=event.get("idx", index)) for index, event in enumerate(events)]
    # loop.py marks the Run assisted the moment one event is; the footer does not repeat the flag.
    head["assisted"] = bool(head.get("assisted") or any(event.get("assisted") for event in events))
    try:
        return Run.model_validate(head)
    except ValidationError:
        return None


def load_runs(folder: Path, unread: Optional[list] = None) -> list[Run]:
    """Every stored Run under `runs/`: the JSONL the loop writes, and a Run stored as one JSON file."""
    out: dict[str, Run] = {}
    for path in sorted(folder.rglob("*.jsonl")) if folder.is_dir() else []:
        run = run_from_jsonl(path)
        if run is None:
            _note(unread, f"{_relative(folder, path)}: not a Run JSONL this report can read")
            continue
        out[run.run_id] = run
    for run in _records(folder, Run):
        out.setdefault(run.run_id, run)
    return [out[run_id] for run_id in sorted(out)]


def _rate_row(name: str, part: dict) -> ScorecardItem:
    matched, total = part.get("matched", 0), part.get("total", 0)
    note = f"{matched} of {total} matched"
    if part.get("explained_misses"):
        note += f", {part['explained_misses']} misses explained"
    if part.get("unexplained"):
        note += f", {part['unexplained']} unexplained"
    return ScorecardItem(name=name, raw=_percent(part.get("raw")), explained=_percent(part.get("explained")), note=note)


def scorecard_rows(body: Any) -> list[ScorecardItem]:
    """The scorecard as rows, from a list of ScorecardItem or from gates/scorecard.py's own dict (D62, D80).

    gates/scorecard.py returns one nested dict per build, which is the only scorecard any module produces,
    so the report flattens that shape rather than asking for a second writer.
    """
    if isinstance(body, list):
        out = []
        for item in body:
            try:
                out.append(ScorecardItem.model_validate(item))
            except ValidationError:
                continue
        return out
    if not isinstance(body, dict):
        return []
    rows: list[ScorecardItem] = []
    fidelity = body.get("tool_fidelity") or {}
    for kind in ("success", "error"):
        part = fidelity.get(kind)
        if isinstance(part, dict):
            rows.append(_rate_row(f"replay fidelity, {kind} calls", part))
    for key, name in (("verdict_agreement", "verdict agreement"), ("user_fact_consistency", "user fact consistency")):
        part = body.get(key)
        if isinstance(part, dict):
            rows.append(_rate_row(name, part))
    coverage = body.get("task_coverage")
    if isinstance(coverage, dict):
        rows.append(
            ScorecardItem(
                name="task coverage",
                raw=f"{coverage.get('tasks_covered', 0)} of {coverage.get('tasks_total', 0)} Tasks",
                explained=_percent(coverage.get("run_weighted")),
                note="the explained column is the Run-weighted share (D96)",
            )
        )
    return rows


def _list_of(path: Path, model: type) -> list:
    return _list_of_bodies(_json(path), model)


def _list_of_bodies(body: Any, model: type) -> list:
    """A JSON list as records, skipping anything in it that is not one."""
    out = []
    for item in body if isinstance(body, list) else []:
        try:
            out.append(model.model_validate(item))
        except ValidationError:
            continue
    return out


def _difficulty_body(root: Path) -> dict:
    """difficulty.json as the last round left it (D209); a workdir without one reads as no record."""
    body = _json(root / difficulty.FILE_NAME)
    return body if isinstance(body, dict) else {}


def _rounds_of(path: Path, unread: Optional[list] = None) -> list[RoundRecord]:
    """rounds.json as records: a file that is there and is not a list of RoundRecord is named, since a
    round that did not load would leave the trusted count and the exit unsaid."""
    if not path.is_file():
        return []
    body = _json(path)
    rows = body if isinstance(body, list) else None
    out: list[RoundRecord] = []
    for item in rows or []:
        try:
            out.append(RoundRecord.model_validate(item))
        except ValidationError:
            rows = None
            break
    if rows is None:
        _note(unread, f"{path.name}: not a list of RoundRecord this report can read")
        return []
    return out


def coverage_rows(tasks: list[Task], uncovered: dict[str, str]) -> list[TaskCoverage]:
    """The D96 rows from a Task list and the first failing reason per uncovered Task.

    The rule itself is gates/scorecard.py's `task_coverage`, so the scorecard and the report cannot drift
    apart; cli.py applies it and hands the reasons here, which keeps the Runner out of this module.
    """
    return [
        TaskCoverage(
            task_id=task.id,
            covered=task.id not in uncovered,
            reason=uncovered.get(task.id),
            run_count=len(task.run_ids),
        )
        for task in tasks
    ]


def load_tool_sigs(workdir: Any) -> list[ToolSig]:
    """The mined ToolSigs of a build, from the tool_sigs.json one stage writes.

    cli.py reads them too, so a Verdict knows which tools write (extra-write and entity-count checks)
    and which are still flagged (D70), and the report and the Verdict cannot disagree about it.
    """
    root = Path(workdir)
    return _list_of(root / "tool_sigs.json", ToolSig)


SYNTHETIC_INDEX = ("synthetic", "index.json")
"""Where the generated Tasks keep their index (D224). Named here rather than imported: the report
reads records and never reaches into the Builder (design section 4 item 18)."""


DOMAIN_ARCHETYPES = ("domain", "archetypes.json")


DOMAIN_GAPS = ("domain", "gaps.json")


SHAPED_INDEX = ("synthetic", "shaped.json")
"""Where the domain reading and the shaped Tasks keep their records (D225). Named here rather than
imported, for the reason SYNTHETIC_INDEX is: the report reads records and never reaches into the
Builder (design section 4 item 18)."""


def _domain_body(root: Path, parts: tuple) -> dict:
    """One of D225's records as the last reading left it, or nothing where none has run."""
    body = _json(root.joinpath(*parts))
    return body if isinstance(body, dict) else {}


def _synthetic_body(root: Path) -> dict:
    """synthetic/index.json as the last request left it, or nothing where none has run (D224)."""
    body = _json(root.joinpath(*SYNTHETIC_INDEX))
    return body if isinstance(body, dict) else {}


def load(workdir: Any) -> ReportData:
    """Read every record the report shows from one workdir. Missing files mean a shorter report, not an error."""
    root = Path(workdir)
    unread: list[str] = []
    env_body = _json(root / "environment.json")
    environment = Environment.model_validate(env_body) if isinstance(env_body, dict) else None
    fidelity_body = _json(root / "tool_fidelity.json")
    state = _json(root / "pipeline" / "state.json")
    state = state if isinstance(state, dict) else {}
    stopped = state.get("stopped") if isinstance(state.get("stopped"), dict) else {}
    budget = _json(root / "budget.json")
    stages = stage_statuses(state, budget if isinstance(budget, dict) else {}, stopped)
    config = _json(root / "report_config.json") or {}
    pairs = _jsonl(root / "judge_pairs.jsonl", unread)
    tasks = _records(root / "tasks", Task, unread)
    runs = load_runs(root / "runs", unread)
    gates = _list_of(root / "gates.json", GateResult) + _list_of_bodies(state.get("gates"), GateResult)
    trusted = next((g for g in reversed(_list_of(root / "gates.json", GateResult)) if g.stage == "trusted"), None)
    status = str(state.get("status", "complete"))
    snapshot = round_snapshot.read_snapshot(root)
    data = ReportData(
        title=config.get("title") or ("Run batch report" if config.get("kind") == "batch" else "Harness build report"),
        kind=config.get("kind", "build"),
        built=environment is not None and status not in ("failed", "stopped"),
        stopped_reason=state.get("stopped_reason") or stopped.get("reason") or config.get("stopped_reason"),
        stopped=stopped,
        records_not_read=unread,
        environment=environment,
        gates=gates,
        scorecard=scorecard_rows(_json(root / "scorecard.json")),
        stages=stages,
        tasks=tasks,
        # D208: the live ones only. A workdir an older build left holds a file per Task that ever
        # had a Reference, and a report that counted one whose Reference has since been withdrawn
        # would say the harness stands behind a check it has retired.
        verifiers=lifecycle.live(
            _records(root / "verifiers", Verifier, unread), _json(root / "task_status.json") or {}
        ),
        tool_sigs=load_tool_sigs(root),
        runs=runs,
        verdicts=_records(root / "verdicts", Verdict, unread),
        overlays=_records(root / "overlays", TaskOverlay),
        policy_items=_list_of(root / "constraints.json", Constraint),
        policy_exercised=(_json(root / "policy_coverage.json") or {}).get("exercised", []),
        task_coverage=_list_of(root / "coverage.json", TaskCoverage),
        frontier_models=config.get("frontier_models", []),
        assisted_share=config.get("assisted_share") or assisted_share_from_runs(runs),
        judge_disagreement=dict(disagreement_stats(pairs), by_pair=_by_pair(pairs)),
        judge_models=config.get("judge_models") or {},
        audit_rate=config.get("audit_rate"),
        disagreement_queue=_jsonl(root / "disagreement_queue.jsonl", unread),
        tasks_aside=_jsonl(root / "tasks_aside.jsonl", unread),
        lessons_set_aside=_list_of(root / "lessons_set_aside.json", SetAsideLesson),
        tool_fidelity=fidelity_body if isinstance(fidelity_body, dict) else {},
        user_fidelity=_json(root / "user_fidelity.json")
        if isinstance(_json(root / "user_fidelity.json"), dict)
        else {},
        rounds=_rounds_of(root / "rounds.json", unread),
        trusted=trusted,
        difficulty=_difficulty_body(root),
        synthetic=_synthetic_body(root),
        claims=claims.read_records(root),
        domain=_domain_body(root, DOMAIN_ARCHETYPES),
        domain_gaps=list((_domain_body(root, DOMAIN_GAPS).get("gaps") or [])),
        shaped=_domain_body(root, SHAPED_INDEX),
        findings=_jsonl(root / "repairs" / "repair_record_finding.jsonl", unread),
        # D218 rule 4: the last closed round's table, and the drift of the live files from it.
        snapshot=snapshot or {},
        drift=round_snapshot.drift(
            snapshot, task_status=_json(root / "task_status.json") or {}, replays=_json(root / "replays.json") or {}
        ),
    )
    gate = environment_gate(data)
    if gate is not None and not gate.passed:
        # design section 6: the build Environment gate is what says the Environment was built.
        data.built = False
    return data
