"""The commands of the Harness: ingest, build, freeze-runner, run, verdict, regrade, report, status,
difficulty, export, publish and fetch, each reading and writing records under one workdir with no
hidden state."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from kullback import difficulty, round_snapshot
from kullback.ai.provider import DEFAULT_MODEL
from kullback.report import coverage_rows, load, load_tool_sigs, write_report
from kullback.runner import feed, heartbeat
from kullback.runner.records import (
    EntitySchema,
    Environment,
    RunnerVersion,
    Task,
    Verifier,
    as_dict,
    content_hash,
)

app = typer.Typer(add_completion=False,
                  help="Build an Environment from customer traces, re-run it, grade it, report it.")

WORKDIR = typer.Option(Path("."), "--workdir", "-w", help="Directory every record is read from and written to.")
JUDGE_MODEL = typer.Option(None, "--judge-model",
                          help="Model id for the two agentic judges, as provider/model. Without it, judge atoms "
                               "are left unevaluated and a failure keeps no cause.")
SECOND_JUDGE_MODEL = typer.Option(None, "--second-judge-model",
                                 help="Model id for the second judge, as provider/model (D160). Without it the "
                                      "second judge is --judge-model's own model under a second persona (D97).")
BASE_URL = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model.")


def _entry(path: str, name: str):
    """One function of another Harness module, imported only when a command runs, or one clear message.

    A module that fails to import raises its own traceback: masking that as "not available yet"
    would report a broken module as an unwritten one.
    """
    function = getattr(importlib.import_module(path), name, None)
    if function is None:
        typer.echo(f"{path}.{name} is not available yet")
        raise typer.Exit(2)
    return function


def _write(path: Path, body: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load(path: Path, model: type):
    body = json.loads(path.read_text(encoding="utf-8"))
    return model.model_validate(body)


def _tasks(workdir: Path, task_id: Optional[str]) -> list[Task]:
    folder = Path(workdir) / "tasks"
    tasks = [_load(p, Task) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []
    return [task for task in tasks if task_id is None or task.id == task_id]


def _schema(workdir: Path):
    """The mined EntitySchema, which is what drops the exempt columns from the diff a predicate sees (D39, D73).

    Without it a forbidden atom over `diff()` fires on a column the customer marked exempt, so a
    Verdict from the CLI disagrees with the same Verdict computed in the build. A build that has not
    written `schema.json` yet is scored without it and told so.
    """
    path = Path(workdir) / "schema.json"
    return _load(path, EntitySchema) if path.is_file() else None


def _judges(model_id: Optional[str], base_url: Optional[str] = None,
            second_model_id: Optional[str] = None):
    """The two agentic judges of D92, or None when the caller named no judge model.

    Both are constructed here and never inside the Runner: `verdict.py` takes judge answers as data
    and calls no model itself (D76, build brief rule 2). With `--second-judge-model` the second judge
    is a judge of its own on that model (D160), which is what makes a disagreement a disagreement
    between two models rather than between two personas; without it the second judge is the same
    adapter under judge.py's own second persona, the D97 default. Either way each judge is named
    `<provider/model>:<persona letter>`, so every ruling says which model it ran on.
    """
    if not model_id:
        return None
    build_judge = _entry("kullback.runner.judge", "AgenticJudge")
    name = _entry("kullback.runner.judge", "judge_name")
    first = build_judge(_live_model(model_id, base_url), name=name(model_id, "a"))
    if second_model_id:
        return first, build_judge(_live_model(second_model_id, base_url), name=name(second_model_id, "b"))
    return first, _entry("kullback.runner.judge", "third_judge")(first, name=name(model_id, "b"))


def _judged_atoms(verifier: Verifier, paths: list, judges, workdir: Path) -> dict:
    """{run_id: {atom_id: JudgeResult}} for a Verifier with judge atoms, which is verdict.py's shape (D76).

    Without this the judge atoms of a Verifier are never answered, so every Run carrying one is
    "not verdicted, Verifier immature" whatever the Candidate did.
    """
    if judges is None or not any(getattr(atom, "judge", False) for atom in verifier.atoms):
        return {}
    load_run = _entry("kullback.runner.verdict", "load_run")
    answer = _entry("kullback.runner.judge", "judge_atom_results")
    out = {}
    for path in paths:
        run = load_run(path)
        out[run.run_id] = answer(verifier, run, judges[0], judges[1],
                                 workdir=workdir, run_id=run.run_id)
    return out


def _drop_uncaused(out_dir: Path, run_id: str) -> None:
    """Take away the Verdict written before the judge named the cause, so one Run keeps one Verdict.

    Both sit under the same versions and the same folder, so a report reading it would otherwise
    count the Run twice; the Verdict that names a cause is the one that stands.
    """
    for path in sorted(Path(out_dir).glob(f"{run_id}.*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if body.get("run_id") == run_id and body.get("cause") is None:
            path.unlink()


def _name_causes(verdicts: list, paths: list, judges, workdir: Path, out_dir: Path, rescore) -> int:
    """Ask the judges to name the cause of every failed Run that code left unmarked (D88).

    Code marks the Run and the judge names the cause; neither is computed in `verdict.py`. The Run is
    scored a second time with the answer, which lands under its own cache key because `cause_result`
    is one of the Verdict's inputs, so nothing has to be refreshed by hand.
    """
    if judges is None:
        return 0
    name = _entry("kullback.runner.judge", "judge_cause_result")
    load_run = _entry("kullback.runner.verdict", "load_run")
    by_id = {load_run(path).run_id: path for path in paths}
    reference = load_run(paths[0]) if paths else None
    named = 0
    for record in verdicts:
        if record.passed or record.cause is not None or "cause_pending_judge" not in record.notes:
            continue
        path = by_id.get(record.run_id)
        if path is None:
            continue
        result = name(load_run(path), reference, judges[0], judges[1],
                      workdir=workdir, run_id=record.run_id)
        rescore(path, cause_result=result)
        _drop_uncaused(out_dir, record.run_id)
        named += 1
    return named


def _rescorer(score_one, verifier: Verifier, canon, out_dir: Path, judge_version, common: dict):
    """The callback `_name_causes` scores one Run with, once a judge has named its cause (D88)."""

    def rescore(path: Path, **extra):
        return score_one(path, verifier, canon, out_dir=out_dir,
                         judge_version=judge_version, **common, **extra)

    return rescore

def _warn_unfrozen_runner() -> None:
    """One stderr line saying the Verdicts carry no Runner version, and the command that freezes one."""
    typer.echo("warning: the Verdicts carry no Runner version, run `kullback freeze-runner`", err=True)


def _score(workdir: Path, task_id: Optional[str], what: str, use_queue: bool = False,
           judge_model: Optional[str] = None, base_url: Optional[str] = None,
           second_judge_model: Optional[str] = None) -> None:
    """Score stored Runs against their Task's Verifier. Nothing is re-executed; the version cache makes a repeat free.

    With `--judge-model` the judge atoms of each Verifier are answered before the Verdict and the
    cause of each unexplained failure after it (D76, D88). Without one, a judge atom stays
    unevaluated and a failure keeps `cause_pending_judge`; both are said out loud rather than
    silently passing. `--second-judge-model` puts a second model on the other side of every question
    (D160); without it the second judge is the same model under a second persona.
    """
    score = _entry("kullback.runner.regrade", "regrade")
    score_one = _entry("kullback.runner.regrade", "regrade_run")
    regrade_gate = _entry("kullback.gates.artifacts", "regrade_gate")
    judge_version = _entry("kullback.runner.judge", "JUDGE_VERSION") if judge_model else None
    judges = _judges(judge_model, base_url, second_judge_model)
    load_rules = _entry("kullback.runner.canon", "load_rules")
    canon_rules = load_rules(Path(workdir) / "canon-rules.json")
    canon = canon_rules
    env_path, version_path = Path(workdir) / "environment.json", Path(workdir) / "runner_version.json"
    environment = _load(env_path, Environment) if env_path.is_file() else None
    version = _load(version_path, RunnerVersion).runner_version if version_path.is_file() else None
    if not version_path.is_file():
        _warn_unfrozen_runner()
    schema = _schema(workdir)
    if schema is None:
        typer.echo("no EntitySchema on disk (schema.json): exempt columns are not dropped from the diff (D73)")
    sigs = load_tool_sigs(workdir)  # without these the extra-write and D70 checks never fire
    write_tools = {sig.name for sig in sigs if sig.kind == "write"}
    flagged_tools = {sig.name for sig in sigs if sig.unclassified}
    if sigs and not write_tools:
        typer.echo("no write tools among the mined ToolSigs: side effects are not checked")
    elif not sigs:
        typer.echo("no ToolSigs on disk: side effects are not checked")
    queued = set(_entry("kullback.runner.canon", "queued_regrades")(workdir)) if use_queue else set()
    scored = 0
    for task in _tasks(workdir, task_id):
        path = Path(workdir) / "verifiers" / f"{task.id}.json"
        if not path.is_file():
            typer.echo(f"task {task.id}: no Verifier yet, not scored")
            continue
        verifier, folder = _load(path, Verifier), Path(workdir) / "runs" / task.id
        paths = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
        if not paths:
            typer.echo(f"task {task.id}: no stored Runs")
            continue
        out_dir = Path(workdir) / "verdicts" / task.id
        common = dict(environment=environment, runner_version=version, schema=schema,
                      write_tools=write_tools or None, flagged_tools=flagged_tools,
                      rules=canon_rules)
        verdicts = score(paths, verifier, canon, out_dir=out_dir,
                         judge_results=_judged_atoms(verifier, paths, judges, Path(workdir)),
                         judge_version=judge_version,
                         queue_dir=Path(workdir) if use_queue else None, **common)
        gated = regrade_gate(verdicts)
        # getattr, not gated.passed: a test double that stands in for regrade_gate (fake_modules)
        # answers neither attribute, and a stand-in gate must not read as a refusal (D97).
        if not getattr(gated, "passed", True):
            for failure in getattr(gated, "failures", []):
                typer.echo(f"task {task.id}: refused, {failure}")
            continue
        named = _name_causes(verdicts, paths, judges, Path(workdir), out_dir,
                             _rescorer(score_one, verifier, canon, out_dir, judge_version, common))
        scored += 1
        forced = len([p for p in paths if p.stem in queued])
        cached = len(paths) - forced
        tail = (f", {forced} re-scored from the regrade queue (D84) and {cached} served from the version cache"
                if use_queue else "")
        tail += f", {named} failures given a cause by the judges (D88)" if named else ""
        typer.echo(f"task {task.id}: {what} {len(paths)} Runs{tail}")
    if not scored:
        typer.echo(f"no Task matched {task_id}: nothing was scored" if task_id
                   else "no Task with a Verifier and stored Runs: nothing was scored")
        raise typer.Exit(1)


def runner_version(routing_config: Optional[Path] = None) -> RunnerVersion:
    """The content hash of every file in runner/ and the routing config, as one RunnerVersion record.

    runner/boundary.py computes it, so freeze-runner and the gate that checks a Run can never
    disagree; the gates package is hashed beside it as `gates_version` (D122).
    """
    compute = _entry("kullback.runner.boundary", "runner_version")

    config = Path(routing_config).read_text(encoding="utf-8") if routing_config else None
    return compute(Path(__file__).parent, config,
                   created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


@app.command()
def ingest(files: list[Path] = typer.Argument(..., help="The customer's export files."),  # noqa: B008
          workdir: Path = WORKDIR,
          intake_floor: Optional[float] = typer.Option(
              None, "--intake-floor",
              help="Task-eligible share each file must keep (default: the intake floor); "
                   "refused outside [0, 1]."),
          dry_run: bool = typer.Option(
              False, "--dry-run",
              help="Say whether each file would ingest and what it would give, without writing anything.")):
    """Store the customer's files byte for byte and derive Traces from them (D66)."""
    if intake_floor is not None and not 0.0 <= intake_floor <= 1.0:
        raise typer.BadParameter("--intake-floor must lie within [0, 1]")
    if dry_run:
        _ingest_dry_run(files, workdir, intake_floor)
        return
    ingest_file = _entry("kullback.builder.ingest", "ingest_file")
    gate_error = _entry("kullback.builder.ingest", "IntakeGateError")
    summaries = []
    for path in files:
        try:
            if intake_floor is None:
                summary = ingest_file(path, workdir)
            else:
                summary = ingest_file(path, workdir, intake_floor=intake_floor)
        except gate_error as exc:
            typer.echo(str(exc))
            raise typer.Exit(1) from None
        summaries.append(summary)
        typer.echo(_ingest_row(str(path), summary["format"], _detect_confidence(path),
                              summary["runs"], summary["gate"]["metrics"]["eligible_share"],
                              summary["gate"]["metrics"]["floor"],
                              _ingested_tools(Path(workdir), summary["raw_hash"]), "ready"))
    _write(Path(workdir) / "ingest_summary.json", summaries)


def _ingest_dry_run(files: list[Path], workdir: Path, intake_floor: Optional[float]) -> None:
    """One dry-run row per file: whether it would ingest and what it would give."""
    dry = _entry("kullback.builder.ingest", "dry_run")
    for path in files:
        result = dry(path, workdir, intake_floor=intake_floor)
        if result["passed"]:
            status = "ready"
        else:
            first = (result["failures"] or result["reasons"] or ["no reason given"])[0]
            status = f"not read: {first}"
        typer.echo(_ingest_row(str(path), result["format"], result["confidence"], result["traces"],
                              result["eligible_share"], result["floor"], result["tools"], status))


def _ingest_row(path: str, format: str, confidence: float, traces: int, share: float,
                floor: float, tools: int, status: str) -> str:
    """One ingest line: the file, what it is, and whether it is ready to build on."""
    return (f"{path} format {format} confidence {confidence:.2f} traces {traces} "
            f"eligible {share:.2f} vs {floor:.2f} tools {tools} {status}")


def _detect_confidence(path: Path) -> float:
    """The winning adapter's vote on a file, for the row after a successful ingest.

    Confidence is a display concern, so it is re-voted here instead of riding the summary.
    """
    decode = _entry("kullback.builder.ingest", "_decode")
    detect = _entry("kullback.builder.sources", "detect_format")
    document, jsonl = decode(Path(path).read_bytes())
    decision = detect(document, jsonl)
    if decision.winner == "unknown":
        return 0.0
    vote = decision.votes.get(decision.winner, (0.0, []))[0]
    return float(vote) if isinstance(vote, (int, float)) and vote > 0 else 0.0


def _ingested_tools(workdir: Path, raw_hash: str) -> int:
    """How many distinct tool names one ingested file's traces call, eligible and set aside."""
    names: set[str] = set()
    for folder in ("traces", "evidence_traces"):
        root = workdir / folder
        if not root.is_dir():
            continue
        for cand in root.glob("*.json"):
            try:
                body = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(body, dict) or body.get("raw_hash") != raw_hash:
                continue
            for call in body.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("name"):
                    names.add(call["name"])
    return len(names)


sources_app = typer.Typer(add_completion=False,
                          help="Shape a file and check a reader for a format the harness "
                               "does not map yet.")
app.add_typer(sources_app, name="sources")


@sources_app.command("shape")
def sources_shape(file: Path = typer.Argument(..., help="The customer's export file."),  # noqa: B008
                  as_json: bool = typer.Option(False, "--json",
                                               help="Print the summary as JSON.")):
    """Print the structure-only summary of a file, keeping no customer string."""
    shape_summary = _entry("kullback.builder.sources.shape", "shape_summary")
    summary = shape_summary(file)
    if as_json:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    for line in _shape_lines(summary):
        typer.echo(line)


def _shape_lines(summary: dict) -> list[str]:
    """One line per key path: the types, the count, length ranges, hints, kept words."""
    lines = [f"{summary.get('file')} jsonl={summary.get('jsonl')} records={summary.get('records')}"]
    for path, info in sorted(summary.get("paths", {}).items()):
        line = f"{path}: {','.join(info.get('types', []))} x{info.get('count', 0)}"
        if "min_len" in info:
            line += f" len {info['min_len']}..{info['max_len']}"
        for hint in info.get("hints", []):
            line += f" {hint}"
        if "values" in info:
            line += f" ={'|'.join(info['values'])}"
        lines.append(line)
    return lines


@sources_app.command("check")
def sources_check(file: Path = typer.Argument(..., help="The raw file the reader was written for."),  # noqa: B008
                  reader: Path = typer.Option(..., "--reader",  # noqa: B008
                                              help="The workdir reader file defining ADAPTER.")):
    """Check a reader against a file, printing the problems or passes."""
    load = _entry("kullback.builder.sources.workdir_readers", "adapter_from_path")
    check = _entry("kullback.builder.sources.workdir_readers", "check_reader")
    try:
        adapter = load(reader)
    except (ValueError, ImportError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    problems = check(adapter, file)
    if problems:
        for problem in problems:
            typer.echo(problem)
        raise typer.Exit(1)
    typer.echo("passes")


def _rescue_candidates(workdir: Path, raw_hash: Optional[str]) -> list:
    """Every eligible or set-aside trace whose sidecar carries a task reference.

    Eligible traces live under traces/, set-aside ones under evidence_traces/;
    both keep a grader sidecar under the same content-hash name, which is where
    the adapter's task_ref and instruction were written at ingest. A trace with
    no sidecar, an unreadable one, or one with no task id is outside this round
    and is skipped, never counted.
    """
    candidates = []
    for folder in ("traces", "evidence_traces"):
        folder_path = Path(workdir) / folder
        if not folder_path.is_dir():
            continue
        for trace_path in sorted(folder_path.glob("*.json")):
            found = _rescue_ref(Path(workdir) / "grader" / f"{trace_path.stem}.json",
                                trace_path.stem, raw_hash)
            if found is not None:
                candidates.append((trace_path.stem, found[0], found[1]))
    return candidates


def _rescue_ref(sidecar: Path, trace_hash: str, raw_hash: Optional[str]):
    """One grader sidecar as (task id, recorded instruction), or None outside this round.

    None covers every way a trace falls outside the rescue: no sidecar, an
    unreadable one, one belonging to another hash, the wrong file under
    --raw-hash, or fields with no task id. Skipped traces are never counted.
    """
    if not sidecar.is_file():
        return None
    try:
        body = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("trace_hash", trace_hash) != trace_hash:
        return None
    if raw_hash is not None and body.get("raw_hash") != raw_hash:
        return None
    fields = body.get("fields") or {}
    if not isinstance(fields, dict):
        return None
    ref = fields.get("task_ref") or {}
    if not isinstance(ref, dict) or not isinstance(ref.get("id"), str):
        return None
    recorded = fields.get("instruction")
    return (ref["id"], recorded if isinstance(recorded, str) else None)


@app.command()
def rescue(
    workdir: Path = WORKDIR,
    registry: Path = typer.Option(..., "--registry", help="Local registry: a directory of task "  # noqa: B008
                                             "directories, or a JSON file of rows."),
    raw_hash: Optional[str] = typer.Option(None, "--raw-hash", help="Only the recordings of this "  # noqa: B008
                                                           "ingested file."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the counts without writing."),
):
    """Attach task definitions to the recordings that name a task id (D263).

    For every eligible or set-aside recording whose sidecar carries a task_ref,
    the id is looked up in the registry. When found and its instruction is the
    recording's own, the definition lands under task_defs/ beside a strength
    label; when found with a different instruction nothing is written, and when
    absent nothing is either. Nothing else in the workdir changes: no recording
    changes standing here.
    """
    from kullback.builder import registry as registry_mod

    try:
        table = registry_mod.open_registry(registry)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(2) from None
    attached = mismatched = missing = 0
    strengths = {"content": 0, "existence": 0, "none": 0}
    for trace_hash, task_id, recorded in _rescue_candidates(Path(workdir), raw_hash):
        try:
            definition = table.lookup(task_id)
        except ValueError:
            definition = None
        if definition is None:
            missing += 1
            continue
        if not registry_mod.instruction_matches(definition, recorded):
            mismatched += 1
            continue
        strength = registry_mod.verifier_strength(definition)
        strengths[strength] += 1
        attached += 1
        if not dry_run:
            body = registry_mod.definition_dict(definition)
            body["verifier_strength"] = strength
            body["matched"] = True
            _write(Path(workdir) / "task_defs" / f"{trace_hash}.json", body)
    looked = attached + mismatched + missing
    line = (f"rescue: looked up {looked}, attached {attached}, mismatched {mismatched}, "
            f"missing {missing} (content {strengths['content']}, "
            f"existence {strengths['existence']}, none {strengths['none']})")
    typer.echo(line + (" (dry run: nothing written)" if dry_run else ""))


def _load_keys() -> None:
    """Keys the screen reads from the environment: exported env, then .env, then remembered.

    The screen and its guards read os.environ before any model call, so both entries that
    open it load every store first, in the same precedence live_model uses. Values never print."""
    _entry("kullback.ai.provider", "load_dotenv")()
    _entry("kullback.ai.credentials", "load_credentials")()


def _live_model(model_id: str, base_url: Optional[str]):
    """One live adapter, or the refusal in words. provider.live_model is the single place the
    live-call flag is ever set, so the screen and the CLI refuse for the same reason."""
    try:
        return _entry("kullback.ai.provider", "live_model")(model_id, base_url)
    except RuntimeError as error:
        typer.echo(str(error))
        raise typer.Exit(2) from None

@app.command()
def login(
    name: Optional[str] = typer.Argument(None, help="Key variable to remember, as OPENAI_API_KEY."),
    list_keys: bool = typer.Option(False, "--list", help="Print the remembered key names, never values."),
):
    """Remember a provider key for every later launch, or list what is remembered.

    The value is read with getpass so it never lands in shell history, and it is stored
    in ~/.kullback/auth.json at mode 0600, never in a workdir or a package. Names are
    printed, values never."""
    from kullback.ai import credentials

    if list_keys:
        for stored in credentials.stored_names():
            typer.echo(stored)
        if not credentials.stored_names():
            typer.echo("no remembered keys")
        return
    if not name:
        typer.echo("kullback login takes a key name, as `kullback login OPENAI_API_KEY`, or --list")
        raise typer.Exit(2)
    import getpass

    try:
        secret = getpass.getpass(f"{name}: ")
    except (EOFError, KeyboardInterrupt):
        typer.echo("")
        raise typer.Exit(1) from None
    if not secret:
        typer.echo("empty key: nothing remembered")
        raise typer.Exit(1)
    try:
        path = credentials.remember(name, secret)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    typer.echo(f"remembered {name} in {path}")


@app.command()
def logout(name: str = typer.Argument(..., help="Key variable to forget, as OPENAI_API_KEY.")):
    """Forget a remembered provider key. The live key store is ~/.kullback/auth.json."""
    from kullback.ai import credentials

    try:
        removed = credentials.forget(name)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    typer.echo(f"forgot remembered {name}" if removed else f"no remembered key {name}")


@app.command()
def build(
    workdir: Path = WORKDIR,
    model: str = typer.Option(DEFAULT_MODEL, "--model",
                              help=f"Builder model id, as provider/model; the default is {DEFAULT_MODEL}."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model",
                                              help="Model id for the judges (D160); the default is --model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    files: Optional[list[Path]] = typer.Option(None, "--file", help="Customer export to ingest first."),  # noqa: B008
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Per-build spend ceiling (D86)."),
):
    """Run the autonomous Builder session over the ingested Traces and write the Environment.

    The Builder is an extension on the agent core: it ingests, derives the world, reads and edits
    its Environment with the base tools, calls examine when the Environment is ready, acts on the
    findings, and stops when every Task is trusted or refused, when the spend ceiling is reached,
    or when it states why it cannot go further. `--model` drives the session and, unless named
    otherwise, the Examiner, the judges, the loophole probe and the re-rolls; `--judge-model`
    puts the judges on a model of their own.
    """
    adapter = _live_model(model, base_url)
    judge_adapter = _live_model(judge_model, base_url) if judge_model else None
    # The screen lists running builds from these heartbeats; the pid tells it who is alive. The
    # pulse keeps beating while the build runs so a screen watching from another directory sees
    # the spend move, rather than a stale $0.0000 until the build is over.
    # The feed is this build's story, opened here and appended to as it goes: /watch reads it to
    # show the calls as they happen instead of a board that only moves when the session ends.
    feed.start(workdir, model=model, ceiling_usd=ceiling_usd)
    pulse = heartbeat.pulse(workdir, model, "running")
    try:
        result = _entry("kullback.builder.session", "build")(
            workdir, adapter, files=list(files or []), ceiling_usd=ceiling_usd,
            subscribers=[_session_subscriber(workdir)], judge_model=judge_adapter,
            probe_model=adapter, reroll_model=adapter)
    except Exception:
        pulse.stop()
        heartbeat.beat(workdir, model, "failed")
        raise
    pulse.stop()
    # A session that ended on a provider error failed, even though nothing was raised.
    failed = isinstance(result, dict) and result.get("stopped") == "error"
    heartbeat.beat(workdir, model, "failed" if failed else "done")
    typer.echo(_counts_line(workdir))
    typer.echo(json.dumps(result, indent=2, default=str))
    if failed:
        raise typer.Exit(code=1)


def _session_subscriber(workdir: Path):
    """The subscriber the build hands the session: a line per tool result as each lands.

    After every examine result the workdir is checkpointed under the next number, so status
    and report keep the drift section the rounds used to feed.
    """
    seen = {"examine": 0}

    def handle(event: Any) -> None:
        if getattr(event, "type", None) != "tool_execution_end":
            return
        result = event.result
        content = str(getattr(result, "content", "") or "")
        lines = content.splitlines()
        typer.echo(f"{event.tool_name}: {lines[0] if lines else ''}")
        for ruling in (getattr(result, "details", None) or {}).get("rulings") or ():
            mark = "accepted" if ruling.get("accepted") else "rejected"
            typer.echo(f"  ruling {ruling.get('name')}: {mark} ({ruling.get('line')})")
        if event.tool_name == "examine":
            seen["examine"] += 1
            round_snapshot.checkpoint(workdir, seen["examine"])

    return handle


def _counts_line(workdir: Path) -> str:
    """The workdir counts as one line: trusted, refused and fidelity off the gate rulings."""
    from kullback import live_counts

    counts = live_counts.workdir_counts(workdir, max_age=0.0)
    return (f"trusted {counts['trusted']}, refused {counts['refused']}, "
            f"fidelity {counts['fidelity']}/{counts['tasks']}")


def _verifier_dicts(root: Path) -> list:
    """The stored Verifiers as dicts; a workdir with none yet reads as empty."""
    from kullback import live_counts

    return live_counts.verifier_dicts(root)


def _refusal_dicts(root: Path) -> dict:
    """The stored refusals keyed by Task; a workdir with none yet reads as empty."""
    from kullback import live_counts

    return live_counts.refusal_dicts(root)


@app.command("freeze-runner")
def freeze_runner(
    workdir: Path = WORKDIR,
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
    routing_config: Optional[Path] = typer.Option(None, "--routing-config", help="Routing config to hash in."),  # noqa: B008
    by: str = typer.Option("unknown", "--by", help="Who confirmed the freeze."),
):
    """Write the RunnerVersion that every later Verdict carries, after a person confirms it."""
    version = runner_version(routing_config)
    typer.echo(f"runner version {version.runner_version} over {len(version.file_hashes)} files")
    typer.echo(f"gates version {version.gates_version} over {len(version.gates_file_hashes)} files")
    if not yes and not typer.confirm("write this RunnerVersion?"):
        typer.echo("not frozen")
        raise typer.Exit(1)
    version.confirmed_by = by if not yes else f"{by} (--yes)"
    path = _write(Path(workdir) / "runner_version.json", as_dict(version))
    typer.echo(str(path))


def _run_task_ids(workdir: Path, selector: str) -> list[str]:
    """The Task ids `--tasks` names: trusted, all, or comma separated ids, in on-disk order.

    Trusted is the live ruling the hub publishes from, so a graded Run always has a checked
    Verifier behind it.
    """
    folder = Path(workdir) / "tasks"
    on_disk = sorted(path.stem for path in folder.glob("*.json")) if folder.is_dir() else []
    on_disk = [task_id for task_id in on_disk if task_id != "tasks.json"]
    if selector == "all":
        if not on_disk:
            typer.echo("no Tasks with a file: nothing was run")
            raise typer.Exit(1)
        return on_disk
    if selector == "trusted":
        trusted = set(_entry("kullback.hub.package", "trusted_ids")(workdir))
        picked = [task_id for task_id in on_disk if task_id in trusted]
        if not picked:
            typer.echo("no trusted Tasks: nothing was run")
            raise typer.Exit(1)
        return picked
    wanted = [part.strip() for part in selector.split(",") if part.strip()]
    if not wanted:
        typer.echo("--tasks names no Tasks")
        raise typer.Exit(2)
    unknown = sorted(set(wanted) - set(on_disk))
    if unknown:
        typer.echo(f"no Task named {', '.join(unknown)}")
        raise typer.Exit(2)
    return [task_id for task_id in on_disk if task_id in set(wanted)]


def _run_scoring(workdir: Path) -> dict:
    """The verdict path's inputs for Runs played by `run --tasks`, loaded once for every Task.

    The same regrade `_score` calls, over the Run files just played rather than the per-Task
    folders, so each Task is scored as soon as its Runs land.
    """
    load_rules = _entry("kullback.runner.canon", "load_rules")
    canon = load_rules(Path(workdir) / "canon-rules.json")
    env_path, version_path = Path(workdir) / "environment.json", Path(workdir) / "runner_version.json"
    if not version_path.is_file():
        _warn_unfrozen_runner()
    sigs = load_tool_sigs(workdir)
    return {
        "canon": canon,
        "environment": _load(env_path, Environment) if env_path.is_file() else None,
        "runner_version": _load(version_path, RunnerVersion).runner_version if version_path.is_file() else None,
        "schema": _schema(workdir),
        "write_tools": {sig.name for sig in sigs if sig.kind == "write"} or None,
        "flagged_tools": {sig.name for sig in sigs if sig.unclassified},
    }


def _score_played_runs(workdir: Path, task_id: str, played: list, scoring: dict):
    """Score just-played Runs through regrade and answer pass per Run, or the refusal in words.

    `played` is (run id, Run file) in play order; the Verdicts land where `verdict` writes them.
    """
    score = _entry("kullback.runner.regrade", "regrade")
    gate = _entry("kullback.gates.artifacts", "regrade_gate")
    verifier = _load(Path(workdir) / "verifiers" / f"{task_id}.json", Verifier)
    verdicts = score([path for _, path in played], verifier, scoring["canon"],
                     out_dir=Path(workdir) / "verdicts" / task_id, judge_results={},
                     environment=scoring["environment"], runner_version=scoring["runner_version"],
                     schema=scoring["schema"], write_tools=scoring["write_tools"],
                     flagged_tools=scoring["flagged_tools"], rules=scoring["canon"])
    failures = getattr(gate(verdicts), "failures", [])
    if failures:
        return None, "; ".join(str(failure) for failure in failures)
    passed = {verdict.run_id: bool(verdict.passed) for verdict in verdicts}
    return [passed.get(run_id, False) for run_id, _ in played], None


def _played_runs(workdir: Path, paths: set) -> list:
    """(run id, Run file) in filename order for Run files the verdict path can open."""
    load_run = _entry("kullback.runner.verdict", "load_run")
    return [(load_run(path).run_id, path) for path in sorted(paths, key=lambda path: path.name)]


def _run_totals(rows: list, count: int, planned: int, spend_usd: float, ceiling_usd,
                ceiling_stopped: bool) -> dict:
    """One dict under the per-Task lines: pass@1, pass^k, Runs done, spend, the Task that failed most."""
    scored = [row for row in rows if row["runs"] and not row["refused"] and not row["no_verifier"]]
    rates = [row["passes"] / row["runs"] for row in scored]
    full = [row for row in scored if row["runs"] == count]
    failing = max(scored, key=lambda row: row["runs"] - row["passes"], default=None)
    if failing is not None and failing["runs"] - failing["passes"] == 0:
        failing = None
    return {
        "pass_at_1": sum(rates) / len(rates) if rates else None,
        "pass_k": (sum(1 for row in full if row["passes"] == row["runs"]) / len(full)
                   if full and count > 1 else None),
        "k": count,
        "runs_done": sum(row["runs"] for row in rows),
        "runs_planned": planned,
        "spend_usd": spend_usd,
        "failing_most": failing["task"] if failing else None,
        "failing_most_fails": failing["runs"] - failing["passes"] if failing else 0,
        "ceiling_stopped": ceiling_stopped,
        "ceiling_usd": ceiling_usd,
    }


def _rate_or_na(value) -> str:
    """One rate with two decimals, or n/a where no scored Run stands behind it."""
    return "n/a" if value is None else f"{value:.2f}"


def _echo_run_totals(totals: dict, count: int) -> None:
    """The pass table's last lines: pass@1, pass^k, Runs done, spend, the Task that failed most."""
    head = f"pass@1 {_rate_or_na(totals['pass_at_1'])}"
    if count > 1:
        head += f" pass^{count} {_rate_or_na(totals['pass_k'])}"
    typer.echo(head)
    failing = (f"{totals['failing_most']} ({totals['failing_most_fails']})"
               if totals["failing_most"] else "none")
    typer.echo(f"runs {totals['runs_done']}/{totals['runs_planned']} "
               f"spend ${totals['spend_usd']:.2f} failing most: {failing}")


@app.command()
def run(
    workdir: Path = WORKDIR,
    task: Optional[str] = typer.Option(None, "--task", help="Task id to run."),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Tasks to run: trusted, all, or comma "
                                                              "separated ids."),
    model: str = typer.Option(..., "--model", help="Candidate model id, as provider/model."),
    count: int = typer.Option(1, "--count", help="Runs per Task."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Spend ceiling (D86): "
                                                                            "the run stops where it stands."),
    as_json: bool = typer.Option(False, "--json", help="Print the rows and the totals as one JSON document."),
):
    """Run a Candidate against the built Environment and write one JSONL per Run.

    With --task the command answers as it always did: the Run reports as JSON. With --tasks it
    runs every named Task --count times, scores each Task as its Runs land, and prints one line
    per Task plus the pass table underneath.
    """
    if tasks is not None and task is not None:
        typer.echo("run takes --task or --tasks, not both")
        raise typer.Exit(2)
    if tasks is None:
        if task is None:
            typer.echo("run takes --task or --tasks")
            raise typer.Exit(2)
        reports = _entry("kullback.runner.tool", "reroll")(
            environment_dir=workdir, task_id=task, model=_live_model(model, base_url), count=count,
            workdir=workdir)
        typer.echo(json.dumps([report.as_dict() for report in reports], default=str))
        return
    task_ids = _run_task_ids(workdir, tasks)
    trusted_only = tasks.strip() == "trusted"
    if trusted_only and not as_json:
        typer.echo("only trusted Tasks are graded by a checked Verifier")
    budget = importlib.import_module("kullback.runner.budget")
    candidate = _live_model(model, base_url)
    ceiling, ceiling_stopped = None, False
    if ceiling_usd is not None:
        try:
            ceiling = budget.Ceiling.from_totals(workdir, ceiling_usd)
        except budget.BudgetExceeded:
            ceiling_stopped = True
    # The Candidate is priced into the workdir ledger like every other model call, but never
    # memoized: a Candidate's answer has to be a fresh sample, and a memoized one would turn a
    # sample into a replay. It is never context capped either: it runs under production settings.
    candidate = budget.BudgetedModel(candidate, stage="run", workdir=workdir, model_id=model,
                                     ceiling=ceiling, cap_context=False)
    scoring = _run_scoring(workdir)
    reroll = _entry("kullback.runner.tool", "reroll")
    rows: list = []
    for task_id in task_ids:
        if ceiling_stopped:
            break
        if not (Path(workdir) / "verifiers" / f"{task_id}.json").is_file():
            if not as_json:
                typer.echo(f"{task_id} no Verifier yet, not scored")
            rows.append({"task": task_id, "runs": 0, "passes": 0, "outcomes": [],
                         "refused": False, "no_verifier": True})
            continue
        runs_dir = Path(workdir) / "runs"
        before = set(runs_dir.glob("*.jsonl")) if runs_dir.is_dir() else set()
        try:
            reports = reroll(environment_dir=workdir, task_id=task_id, model=candidate, count=count,
                             workdir=workdir)
            played = [(report.run_id, Path(workdir) / report.path) for report in reports]
        except budget.BudgetExceeded:
            ceiling_stopped = True
            after = set(runs_dir.glob("*.jsonl")) if runs_dir.is_dir() else set()
            played = _played_runs(workdir, after - before)
        if played:
            outcomes, refusal = _score_played_runs(workdir, task_id, played, scoring)
        else:
            outcomes, refusal = [], None
        if refusal:
            row = {"task": task_id, "runs": len(played), "passes": 0, "outcomes": [],
                   "refused": True, "no_verifier": False}
            line = f"{task_id} refused, {refusal}"
        elif not played:
            row = {"task": task_id, "runs": 0, "passes": 0, "outcomes": [],
                   "refused": False, "no_verifier": False}
            line = f"{task_id} no stored Runs"
        else:
            words = ["pass" if passed else "fail" for passed in outcomes]
            row = {"task": task_id, "runs": len(words), "passes": sum(outcomes),
                   "outcomes": words, "refused": False, "no_verifier": False}
            line = f"{task_id} {' '.join(words)}"
        rows.append(row)
        if not as_json:
            typer.echo(line)
    spend_usd = float(budget.load_totals(workdir)["total"]["usd"])
    totals = _run_totals(rows, count, len(task_ids) * count, spend_usd, ceiling_usd, ceiling_stopped)
    if as_json:
        typer.echo(json.dumps({"rows": rows, "totals": totals, "trusted_only": trusted_only},
                               default=str))
        return
    if ceiling_stopped:
        typer.echo(f"ceiling ${ceiling_usd:g} reached, stopped where it stood")
    _echo_run_totals(totals, count)


@app.command("solve-rate")
def solve_rate(
    workdir: Path = WORKDIR,
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Task ids to run, comma separated; "
                                                                 "default is every Task with a file."),
    model: str = typer.Option(..., "--model", help="Policy model id, as provider/model."),
    runs: int = typer.Option(1, "--runs", help="Runs per Task, each under its own drawn seed."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Spend ceiling (D86): "
                                                                               "the stage stops where it stands."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Where the Runs and the table go; "  # noqa: B008
                                                                  "default is episodes/ under the directory."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
):
    """Run a policy model over Tasks through the reset and step interface and print the solve-rate table (G1).

    A measuring instrument, not a trainer: no-signal Runs (not verdicted, environment failures)
    stand outside the rate and inside the count.
    """
    package = _entry("kullback.runner.world.environment", "BuiltEnvironment")
    stage = _entry("kullback.runner.world.solve", "solve_rate")
    lines = _entry("kullback.runner.world.solve", "markdown_table")
    env = package(workdir)
    wanted = [part.strip() for part in tasks.split(",")] if tasks else None
    task_ids = [task_id for task_id in env.task_ids() if wanted is None or task_id in wanted]
    if wanted:
        unknown = sorted(set(wanted) - set(task_ids))
        if unknown:
            typer.echo(f"no Task named {', '.join(unknown)}")
            raise typer.Exit(2)
    table = stage(env, task_ids, _live_model(model, base_url), runs_per_task=runs,
                  ceiling_usd=ceiling_usd,
                  outdir=Path(outdir) if outdir is not None else Path(workdir) / "episodes")
    for line in lines(table):
        typer.echo(line)
    typer.echo(f"estimated spend {table['spent_usd']:.4f} USD" +
               (" (ceiling reached, stopped where it stood)" if table["ceiling_stopped"] else ""))
    if table.get("unpriced_calls"):
        typer.echo(f"{table['unpriced_calls']} calls are unpriced; the spend estimate is incomplete")


def _format_consistency(value: Optional[float]) -> str:
    """One consistency number as four decimals, or n/a where no law was checked."""
    return "n/a" if value is None else f"{value:.4f}"


def _consistency_line(task_id: str, checked: int, broken: int, consistency: Optional[float],
                      requestors: Optional[dict] = None) -> str:
    """One Task's law counts, its Environment consistency, and each tool's requestor as one line."""
    line = f"{task_id}: checked {checked} broken {broken} consistency {_format_consistency(consistency)}"
    if requestors:
        sides = ", ".join(f"{name} as {requestors[name]}" for name in sorted(requestors))
        return f"{line}, {sides}"
    return line


def _mean_consistency(rows: dict) -> Optional[float]:
    """The mean consistency over the Tasks that checked anything, else nothing."""
    values = [row["consistency"] for row in rows.values() if row["consistency"] is not None]
    return sum(values) / len(values) if values else None


def _echo_gaps(task_id: str, gaps: dict) -> dict:
    """One line per unfit tool, returned for the JSON where any exist."""
    for name in sorted(gaps):
        typer.echo(f"task {task_id}: unfit tool {name}, {gaps[name]}")
    return {task_id: gaps} if gaps else {}


def _finish_consistency(rows: dict, skipped: dict, unfit: dict, out: Optional[Path],
                        laws_version: int, task: Optional[str]) -> None:
    """The final line, the JSON beside it, and the exit: 3 on a skipped request or none completed."""
    mean = _mean_consistency(rows)
    violated = sum(1 for row in rows.values() if row["broken"])
    typer.echo(f"mean consistency {_format_consistency(mean)} over {len(rows)} Tasks, "
               f"{violated} with violations, {len(skipped)} skipped")
    if out is not None:
        body = {"tasks": rows, "mean_consistency": mean, "tasks_with_violations": violated,
                "laws_version": laws_version, "skipped": skipped, "unfit": unfit}
        Path(out).write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    if task is not None and task in skipped:
        raise typer.Exit(3)
    if not rows:
        raise typer.Exit(3)


@app.command("consistency")
def consistency(
    workdir: Path = WORKDIR,
    task: Optional[str] = typer.Option(None, "--task", help="Check one Task instead of every Task."),  # noqa: B008
    sequences: int = typer.Option(20, "--sequences", help="Call sequences generated per Task."),
    max_length: int = typer.Option(5, "--max-length", help="Longest generated call sequence."),
    seed: int = typer.Option(0, "--seed", help="Seed the sequences are drawn under."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the JSON; nothing is written without it."),  # noqa: B008
):
    """Check the laws that hold for any tool over each built Environment, with no model (D258).

    One line per Task names its checked and broken law counts and its Environment
    consistency, one line per skipped Task names why it was skipped, and one line
    per unfit tool names why no sequence may check it. The final line gives the
    mean consistency over completed Tasks, how many broke a law and how many
    were skipped. The workdir is read in place and never written into; --out
    writes the same numbers as JSON beside the laws version. --sequences takes
    1 or more. A Task that checked no law is skipped as no law checked, and a
    Task whose world fails to load or check is skipped under its error's class
    name while the rest still run. A skipped request, or no completed Task at
    all, exits 3.
    """
    from kullback.runner.world.loading import EnvironmentError

    if sequences < 1 or max_length < 1:
        raise typer.BadParameter("--sequences takes 1 or more and --max-length takes 1 or more")
    check = _entry("kullback.laws", "check_laws")
    build = _entry("kullback.runner.world.environment", "BuiltEnvironment")
    make_world = _entry("kullback.runner.world.law_world", "EnvironmentLawWorld")
    laws_module = importlib.import_module("kullback.runner.world.law_world")
    try:
        env = build(workdir)
        ids = env.task_ids()
    except EnvironmentError as exc:
        typer.echo(f"no built Environment under {workdir}: {exc}")
        raise typer.Exit(1) from None
    if task is not None:
        if task not in ids:
            typer.echo(f"no Task named {task}")
            raise typer.Exit(2)
        ids = [task]
    rows: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    unfit: dict[str, dict] = {}

    def _one(task_id: str):
        """One Task's row, its unfit tools and its skip reason; a skip keeps the unfit tools."""
        try:
            world = make_world(env, task_id, seed=seed)
            gaps = world.unfit()
            sides = {info.name: info.requestor for info in world.tools()}
            report = check(world, seed=seed, sequences=sequences, max_length=max_length)
        except Exception as exc:
            return None, {}, f"{type(exc).__name__}: {exc}"
        if not sum(report.checked.values()):
            return None, gaps, "no law checked"
        row = {"checked": sum(report.checked.values()), "broken": sum(report.broken.values()),
               "consistency": report.consistency, "requestors": sides}
        return row, gaps, None

    for task_id in ids:
        row, gaps, reason = _one(task_id)
        if reason is not None:
            skipped[task_id] = reason
            unfit.update(_echo_gaps(task_id, gaps))
            typer.echo(f"task {task_id}: skipped, {reason}")
            continue
        rows[task_id] = row
        unfit.update(_echo_gaps(task_id, gaps))
        typer.echo(_consistency_line(task_id, row["checked"], row["broken"], row["consistency"],
                                     row["requestors"]))
    _finish_consistency(rows, skipped, unfit, out, laws_module.LAWS_VERSION, task)


@app.command()
def verdict(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id."),
            judge_model: Optional[str] = JUDGE_MODEL, base_url: Optional[str] = BASE_URL,
            second_judge_model: Optional[str] = SECOND_JUDGE_MODEL):
    """Score the stored Runs of one Task, or of every Task, on their End state."""
    _score(Path(workdir), task, "scored", judge_model=judge_model, base_url=base_url,
           second_judge_model=second_judge_model)


@app.command()
def regrade(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id."),
            judge_model: Optional[str] = JUDGE_MODEL, base_url: Optional[str] = BASE_URL,
            second_judge_model: Optional[str] = SECOND_JUDGE_MODEL):
    """Re-score stored Runs against the current Environment and Verifier versions, without re-executing them.

    A Run whose equivalence entry a person overturned is in canon.py's regrade queue (D84): its
    versions have not moved, so only the queue makes it score again.
    """
    _score(Path(workdir), task, "regraded", use_queue=True, judge_model=judge_model, base_url=base_url,
           second_judge_model=second_judge_model)


def _coverage_runs(runs: list) -> list:
    """The Runs as D96 counts them: a replay stands for the Trace it replays.

    A Task's `run_ids` are the ids of the customer's Traces (cluster.py), while a replayed Run keeps
    its own id and carries the Trace's id as `trace_id`. Without this every replayed Task reads
    "Run <trace> was not replayed" while the Run is on disk.
    """
    out = list(runs)
    for run in runs:
        if run.trace_id and run.trace_id != run.run_id:
            out.append(run.model_copy(update={"run_id": run.trace_id}))
    return out


@app.command()
def report(
    workdir: Path = WORKDIR,
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the Markdown."),  # noqa: B008
    batch: bool = typer.Option(False, "--batch", help="Report one Run batch instead of a build."),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="Only the Runs of this model, which is what names a batch."),
):
    """Write the Markdown report: the Environment first, then the numbers per Task, then a suggestion (D85)."""
    data = load(workdir)
    if model:
        # A batch is the Runs of one Candidate model; without this filter a batch report counts
        # every stored Verdict of every batch (design section 4 item 18).
        data.runs = [run for run in data.runs if run.model == model]
        keep = {run.run_id for run in data.runs}
        data.verdicts = [v for v in data.verdicts if v.run_id in keep]
    if not data.task_coverage and data.tasks:
        # D96's two headline numbers: covered Tasks over the frozen list, and the same Run-weighted.
        # The rule lives in gates/scorecard.py so the scorecard and the report count the same thing.
        task_coverage = _entry("kullback.gates.scorecard", "task_coverage")
        status_path = Path(workdir) / "task_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        computed = task_coverage(data.tasks, _coverage_runs(data.runs), status)
        data.task_coverage = coverage_rows(
            data.tasks, {row["task_id"]: row["reason"] for row in computed["uncovered"]})
    if batch or model:
        data.kind = "batch"
        data.title = "Run batch report" + (f" for {model}" if model else "")
    for name in data.records_not_read:
        typer.echo(f"not read, so it is not counted: {name}")
    target = Path(out) if out else Path(workdir) / "report.md"
    typer.echo(str(write_report(data, target.parent, target.name)))


@app.command("status")
def status(
    workdir: Path = WORKDIR,
    round_number: Optional[int] = typer.Option(None, "--round", help="Which closed round to read."),
    named: int = typer.Option(3, "--named", help="How many drifted Task ids to name."),
):
    """Read one closed round's Task table and say how far the live files have moved from it (D218).

    The table is what the round ruled, in one pass, and nothing rewrites it. task_status.json is the
    live file and goes on moving under a loosening, a re-derive or a cache recompute, which is right
    and is exactly what made the numbers unreadable: a reader joining the two could not tell a
    harness regression from that movement. So this prints the round it read, the counts that round
    ruled, and the Tasks whose live status now disagrees with it, at the stage each moved.
    """
    root = Path(workdir)
    snapshot = round_snapshot.read_snapshot(root, round_number)
    report = round_snapshot.drift(snapshot, task_status=_json_at(root, "task_status.json"),
                                  replays=_json_at(root, "replays.json"), named=named)
    typer.echo(round_snapshot.drift_line(report))
    counts = dict((snapshot or {}).get("counts") or {})
    for name in ("tasks", "fidelity", "reference", "verifier_passed", "trusted", "refused"):
        if name in counts:
            typer.echo(f"{name}: {counts[name]}")
    for row in report.get("first") or ():
        typer.echo(f"moved: {row['task_id']} at {row['stage']}")


@app.command("tasks")
def tasks(
    workdir: Path = WORKDIR,
    status: Optional[str] = typer.Option(None, "--status",
                                         help="Show only Tasks standing in this word: open, refused, trusted or drifted."),
    as_json: bool = typer.Option(False, "--json", help="Print the rows as JSON."),
    task: Optional[str] = typer.Option(None, "--task", help="Print one Task's detail instead of the rows."),
):
    """List every Task with why it stands where it stands: status, replay, suite, probes, drift, last finding."""
    from kullback import live_counts

    root = Path(workdir)
    if task is not None:
        try:
            detail = live_counts.task_detail(root, task)
        except KeyError:
            typer.echo(f"no such task: {task}")
            raise typer.Exit(1) from None
        if as_json:
            typer.echo(json.dumps(detail, indent=2, sort_keys=True, default=str))
        else:
            for line in _task_detail_lines(detail):
                typer.echo(line)
        return
    rows = live_counts.task_rows(root)
    if status is not None:
        if status == "drifted":
            rows = [row for row in rows if row["drift"] is not None]
        elif status in ("open", "refused", "trusted"):
            rows = [row for row in rows if row["status"] == status]
        else:
            typer.echo("status is open, refused, trusted or drifted")
            raise typer.Exit(2)
    if as_json:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return
    typer.echo("id status replay suite probes drift last finding")
    for row in rows:
        typer.echo(_task_row_line(row))


def _task_row_line(row: dict) -> str:
    """One Task's row as text: every cell, with a dash where the workdir holds nothing."""
    finding = " ".join(part for part in (row["last_finding_kind"], row["last_finding_path"]) if part)
    cells = [row["task_id"], row["status"], row["replay"] or "-", row["suite"] or "-",
             row["probes"] or "-", row["drift"] or "-", finding or "-"]
    return " ".join(cells)


def _task_detail_lines(detail: dict) -> list[str]:
    """One Task's detail as text: the row, the gate that failed, findings, atoms and replay calls."""
    finding = " ".join(part for part in (detail["last_finding_kind"], detail["last_finding_path"]) if part)
    lines = [f"task {detail['task_id']}", f"status: {detail['status']}",
             f"replay: {detail['replay'] or '-'}", f"suite: {detail['suite'] or '-'}",
             f"probes: {detail['probes'] or '-'}", f"drift: {detail['drift'] or '-'}",
             f"last finding: {finding or '-'}", f"failing gate: {detail['failing_gate'] or 'none'}"]
    for row in detail["open_findings"]:
        lines.append("finding: " + " ".join(part for part in (row["kind"], row["path"]) if part))
    atoms = detail["atom_counts"]
    lines.append("atoms: " + (", ".join(f"{kind} {count}" for kind, count in sorted(atoms.items()))
                              if atoms else "none"))
    matched, total = detail["replay_matched"], detail["replay_total"]
    lines.append(f"replay calls: {(matched if matched is not None else '-')}/"
                 f"{(total if total is not None else '-')} matched/total")
    return lines


def _json_at(root: Path, name: str) -> dict:
    """One JSON record of a workdir, or nothing where the file is missing or half-written."""
    from kullback import live_counts

    return live_counts.json_at(root, name)


@app.command("judge-smoke")
def judge_smoke(
    model: str = typer.Option(..., "--model", help="Candidate judge model id, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url",
                                           help="Endpoint for an OpenAI-compatible model."),
):
    """Ask one model two invented equivalence pairs and print resolved or refused per pair (D222).

    A relaunch names a judge model beside the build model, and the only thing it has to know first
    is whether that model returns a verdict on a semantic pair at all. This is that question in one
    call: two pairs of an invented column, one the same and one not, with the route each took.
    """
    build_judge = _entry("kullback.runner.judge", "AgenticJudge")
    name = _entry("kullback.runner.judge", "judge_name")
    judge = build_judge(_live_model(model, base_url), name=name(model, "a"))
    rows = _entry("kullback.runner.judge", "smoke")(judge)
    for line in _entry("kullback.runner.judge", "smoke_lines")(rows):
        typer.echo(line)


def _buckets(pairs: Optional[list[str]]) -> dict[str, int]:
    """`--bucket w1t2p1=20` as {bucket: count}; a name no bucket is spelled with is refused."""
    from kullback.graph import bands

    out: dict[str, int] = {}
    for pair in pairs or []:
        name, sep, count = pair.partition("=")
        if not sep or not count.isdigit() or bands(name)[0] < 0:
            typer.echo(f"not a bucket and a count: {pair}")
            raise typer.Exit(2)
        out[name] = int(count)
    return out


def _shaped_lines(body: dict) -> list[str]:
    """What a shaping request made, per realism rung and per source (D225)."""
    if body.get("refused"):
        return [f"refused: {body['refused']}"]
    counts = dict(body.get("counts") or {})
    fell = dict(counts.get("fell") or {})
    lines = [f"archetypes read {counts.get('archetypes_read', 0)}, "
             f"Tasks shaped {counts.get('tasks_shaped', 0)}", "",
             "| rung | fell |", "| --- | --- |"]
    lines += [f"| {rung} | {fell.get(rung, 0)} |" for rung in fell]
    lines += ["", "| source | Tasks |", "| --- | --- |"]
    lines += [f"| {row.get('source')} | {row.get('tasks')} |"
              for row in counts.get("per_source") or []] or ["| none |  |"]
    return lines


def _synthesis_lines(body: dict) -> list[str]:
    """What a synthesis request generated, as the lines both commands print."""
    counts = dict(body.get("counts") or {})
    rows = body.get("tasks") or []
    lines = [f"walks tried {counts.get('walks_tried', 0)}, refused {counts.get('walks_refused', 0)}, "
             f"crashed {counts.get('walks_crashed', 0)}, unbound {counts.get('walks_unbound', 0)}",
             "", "| bucket asked | bucket reached | Tasks | suite passed | mean pool |",
             "| --- | --- | --- | --- | --- |"]
    grouped: dict = {}
    for row in rows:
        key = (str(row.get("bucket_requested") or ""), str(row.get("bucket") or ""))
        held = grouped.setdefault(key, {"tasks": 0, "passed": 0, "pool": 0})
        held["tasks"] += 1
        held["passed"] += 1 if row.get("suite_passed") else 0
        held["pool"] += int(row.get("pool") or 0)
    for (asked, reached), held in sorted(grouped.items()):
        lines.append(f"| {asked} | {reached} | {held['tasks']} | {held['passed']} | "
                     f"{held['pool'] / held['tasks'] if held['tasks'] else 0:.1f} |")
    if not grouped:
        lines.append("| none |  |  |  |  |")
    return lines


@app.command("difficulty")
def difficulty_table(
    workdir: Path = WORKDIR,
    write: bool = typer.Option(True, "--write/--no-write",
                               help="Rewrite difficulty.json from what the workdir holds."),
    fill: Optional[list[str]] = typer.Option(None, "--fill",  # noqa: B008
                                             help="Generate synthetic Tasks into a bucket, as "
                                                  "bucket=count; repeatable (D224)."),
    seed: Optional[str] = typer.Option(None, "--seed", help="Seed the walks are drawn under (D212)."),
):
    """Print the difficulty buckets of a finished build: Tasks, trusted and solve rate per bucket (D209).

    `--fill` walks the mined dependency graph for the bucket asked for and reports what came out
    (D224); what it generates is stored apart and is never added to the trusted column above.
    """
    body = (difficulty.refresh(workdir) if write else difficulty.compute(workdir))
    for line in difficulty.markdown_table(body.get("buckets") or [], len(body.get("no_record") or {})):
        typer.echo(line)
    targets = _buckets(fill)
    if not targets:
        return
    made = _entry("kullback.synthesise", "synthesise")(workdir, targets, seed=seed)
    typer.echo("")
    typer.echo("Synthetic Tasks, generated and counted apart from everything above (D224):")
    for line in _synthesis_lines(made):
        typer.echo(line)


@app.command()
def synthesise(
    workdir: Path = WORKDIR,
    bucket: Optional[list[str]] = typer.Option(None, "--bucket",  # noqa: B008
                                               help="A difficulty bucket and how many Tasks to "
                                                    "generate into it, as bucket=count; repeatable."),
    seed: Optional[str] = typer.Option(None, "--seed", help="Seed the walks are drawn under (D212)."),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="Model id for the Intent writer, as provider/model. "
                                             "Without it the Intent is written by code from the walk."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd",
                                                help="Spend ceiling for the Intent writer (D86)."),
    archetype: Optional[list[str]] = typer.Option(None, "--archetype",  # noqa: B008
                                                  help="Shape walks to one archetype the domain "
                                                       "reading extracted; repeatable (D225)."),
    from_domain: bool = typer.Option(False, "--from-domain",
                                     help="Shape walks to every mapped archetype the domain "
                                          "reading extracted (D225)."),
    per_archetype: int = typer.Option(1, "--per-archetype", help="How many Tasks per archetype."),
):
    """Walk the mined tool-call graph, run the walks in the rebuilt world and store them apart (D224).

    Nothing generated here enters tasks.json, replay fidelity, a confirmed Reference or the trusted
    count: it lands under synthetic/ in the workdir and is reported under its own heading.

    `--from-domain` and `--archetype` shape the walks to what the domain's own public material
    attests (D225): the walk must visit the tools the archetype's expected effects were mapped onto,
    and every Task it makes climbs the realism bar, whose fallen are counted per rung.
    """
    if archetype or from_domain:
        writer = _live_model(model, base_url) if model else None
        if writer is not None:
            writer = _wrapped_model(writer, "synthetic_intent", Path(workdir), ceiling_usd,
                                    model_id=model)
        made = _entry("kullback.synthesise", "shape")(
            workdir, archetype_ids=list(archetype or []), per_archetype=per_archetype, seed=seed,
            model=writer, judges=[writer, writer] if writer is not None else ())
        for line in _shaped_lines(made):
            typer.echo(line)
        return
    targets = _buckets(bucket)
    if not targets:
        typer.echo("nothing asked for: pass --bucket <bucket>=<count>")
        raise typer.Exit(2)
    # Every model call the harness makes is priced into budget.json and refused past the ceiling
    # (D65, D86); the Intent writer is no exception because it is the only model call here.
    writer = _live_model(model, base_url) if model else None
    if writer is not None:
        writer = _wrapped_model(writer, "synthetic_intent", Path(workdir), ceiling_usd,
                                model_id=model)
    body = _entry("kullback.synthesise", "synthesise")(workdir, targets, seed=seed, model=writer)
    for line in _synthesis_lines(body):
        typer.echo(line)


def _wrapped_model(model: Any, stage: str, workdir: Path, ceiling_usd: Optional[float],
                   model_id: Optional[str] = None) -> Any:
    """A model priced into budget.json and refused past the ceiling, memoized on disk (D65, D86).

    What the Builder's stage wrapper did for its own models, beside the Builder: the memo
    keeps a byte-identical request from reaching the network twice, and the ceiling resumes from
    budget.json so a stopped command does not start at zero.
    """
    budget = importlib.import_module("kullback.runner.budget")
    provider = importlib.import_module("kullback.ai.provider")
    if model is None:
        return None
    name = model_id or getattr(model, "name", None) or "model"
    ceiling = budget.Ceiling.from_totals(workdir, ceiling_usd) if ceiling_usd is not None else None
    if ceiling is not None:
        ceiling.require_priced(name)
    inner = provider.MemoModel(model, workdir)
    cache_key = f"kullback-{content_hash(str(Path(workdir).resolve()))[:12]}-{stage}"
    return budget.BudgetedModel(inner, stage=stage, workdir=workdir, model_id=name,
                                ceiling=ceiling, prompt_cache_key=cache_key)


budget_app = typer.Typer(add_completion=False,
                         help="What a build spent, read from its ledger and feed; nothing is called.")
app.add_typer(budget_app, name="budget")


@budget_app.command("cache")
def budget_cache(workdir: Path = WORKDIR,
                 as_json: bool = typer.Option(False, "--json", help="Print the view as JSON.")):
    """Per stage: calls, the share of prompt tokens read from the cache, writes on new and on already
    sent prefixes, conversations that grew and still read nothing, and the ten largest uncached calls."""
    cache_view = importlib.import_module("kullback.runner.cache_view")
    view = cache_view.cache_view(workdir)
    if as_json:
        typer.echo(json.dumps(view, indent=2))
        return
    for line in cache_view.render(view):
        typer.echo(line)


@budget_app.command("reprice")
def budget_reprice(workdir: Path = WORKDIR,
                   model: Optional[str] = typer.Option(None, "--model",
                                                       help="Price every call under this provider/model id "
                                                            "instead of the id its feed line recorded.")):
    """Price the calls in the feed again and rewrite the dollars in budget.json; nothing is called."""
    budget = importlib.import_module("kullback.runner.budget")
    total = budget.reprice(workdir, model_id=model)["total"]
    typer.echo(f"{int(total['calls'])} calls, ${total['usd']:,.4f}, "
               f"{int(total['unpriced_calls'])} unpriced; {budget.cache_line(total)}")


user_app = typer.Typer(add_completion=False,
                       help="The Simulated user: how close its turns are to the recorded ones, and how "
                            "often it runs out of scenario (D214).")
app.add_typer(user_app, name="user")


@user_app.command("fidelity")
def user_fidelity(
    workdir: Path = WORKDIR,
    agent_model: Optional[str] = typer.Option(None, "--agent-model",
                                              help="Model id for the agent user, as provider/model. Without it "
                                                   "only the rule-driven baseline is scored, which costs nothing."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    task: Optional[str] = typer.Option(None, "--task", help="Score one Task instead of every Task."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Score only the first N Tasks, in id order."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd",
                                                help="Stop the scoring when the agent user has spent this much (D86)."),
    write: bool = typer.Option(True, "--write/--no-write", help="Rewrite user_fidelity.json."),
):
    """Score the Simulated user against the recorded turns, per Task and per corpus (D214 rule 5)."""
    fidelity = importlib.import_module("kullback.user.fidelity")
    budget = importlib.import_module("kullback.runner.budget")
    ceiling = budget.Ceiling(usd=ceiling_usd) if ceiling_usd else None
    make_agent = _agent_user_factory(workdir, agent_model, base_url, ceiling) if agent_model else None
    wanted = [task] if task else fidelity.task_ids(workdir, limit=limit)
    try:
        out = fidelity.score_workdir(workdir, make_agent=make_agent, write=write, tasks=wanted)
    except budget.BudgetExceeded as stop:
        typer.echo(f"stopped on the ceiling: {stop}")
        raise typer.Exit(1) from None
    for line in fidelity.markdown_table(out["body"]):
        typer.echo(line)
    if ceiling is not None:
        typer.echo(f"spend: ${ceiling.spent:.4f} of ${ceiling.usd:.2f}")


def _agent_user_factory(workdir: Path, model_id: str, base_url: Optional[str], ceiling: Any = None):
    """A callable the score uses to build one Task's agent user, on a live model (D214 rule 3).

    The model is wrapped the way a build wraps its own (D86), so every turn this scoring pays for is
    priced, charged against the ceiling and stopped at it, rather than counted after the fact.
    """
    budget = importlib.import_module("kullback.runner.budget")
    model = _live_model(model_id, base_url)
    if ceiling is not None:
        model = budget.BudgetedModel(model, stage="user_fidelity", workdir=workdir,
                                     model_id=model_id, ceiling=ceiling, cap_context=True)
    factory = importlib.import_module("kullback.user.factory")

    def make(ctx, fallback, record_values):
        return factory.build_user(workdir, ctx.task_id, model, factory.PURPOSE_SCORE, ctx=ctx,
                                  fallback=fallback, record_values=record_values)

    return make


@user_app.command("dry-run")
def user_dry_run(workdir: Path = WORKDIR):
    """How the Simulated user ended this build's Runs, and how often its scenario ran out (D210)."""
    fidelity = importlib.import_module("kullback.user.fidelity")
    counts = fidelity.dry_run_counts(workdir)
    typer.echo(f"{counts['runs']} Run(s) over {counts['tasks']} Task(s); "
               f"{counts['classified']} carry an end kind")
    for kind, count in (counts.get("ends") or {}).items():
        typer.echo(f"  {kind}: {count}")
    if counts["classified"] < counts["runs"]:
        typer.echo("  Runs with no end kind were written before the kinds existed; their "
                   "termination reasons are " + ", ".join(
                       f"{name} {n}" for name, n in (counts.get("termination_reasons") or {}).items()))


CORPUS = typer.Option(None, "--corpus", help="Name of the corpus the traces came from, for the manifest "
                                             "and the card.")
CORPUS_LICENSE = typer.Option(None, "--corpus-license", help="Licence of that corpus, as an SPDX id; the "
                                                             "card states it and the front matter indexes it.")
CORPUS_URL = typer.Option(None, "--corpus-url", help="Where that corpus came from.")


def _echo_manifest(manifest: dict) -> None:
    """The numbers a person needs to see before a package leaves the machine."""
    fidelity = manifest.get("replay_fidelity") or {}
    scan = manifest.get("leak_scan") or {}
    typer.echo(f"round {manifest.get('round')}, {manifest.get('tasks_total', 0)} Tasks, "
               f"replay fidelity {_rate(fidelity.get('tasks_rate'))} over Tasks and "
               f"{_rate(fidelity.get('runs_rate'))} over Runs, "
               f"{manifest.get('reference_confirmed', 0)} References confirmed, "
               f"{manifest.get('verifier_derived', 0)} Verifiers derived, "
               f"{manifest.get('trusted', 0)} trusted")
    typer.echo(f"leak scan: {scan.get('leaks', 0)} recorded strings and {scan.get('value_echoes', 0)} "
               f"value echoes over {scan.get('files_scanned', 0)} graded files, of "
               f"{scan.get('values_checked', 0)} strings checked against {scan.get('corpus_strings', 0)} "
               f"the corpus holds ({scan.get('strict_env_only_unaccounted', 0)} against env/ alone)")
    typer.echo(f"content hash {manifest.get('content_hash')}")


def _rate(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{float(value):.1%}"


@app.command()
def export(
    workdir: Path = WORKDIR,
    out: Path = typer.Option(..., "--out", help="Directory the package is written to."),  # noqa: B008
    name: Optional[str] = typer.Option(None, "--name", help="Name of the Environment, which is also its "
                                                            "domain tag; environment.json carries none."),
    corpus: Optional[str] = CORPUS,
    corpus_license: Optional[str] = CORPUS_LICENSE,
    corpus_url: Optional[str] = CORPUS_URL,
    preview: bool = typer.Option(False, "--preview", help="Mark the package as below the fidelity bar."),
):
    """Write a self-contained Environment package: the rebuilt world, the Task list, the Verifiers, a manifest.

    Nothing of the recordings the world was rebuilt from goes in, and a leak scan over the customer's
    export refuses the package if anything repeats a string only a recording could have said (D221).
    """
    build = _entry("kullback.hub.package", "export")
    manifest = build(workdir, out, name=name, corpus=corpus, corpus_license=corpus_license,
                     corpus_url=corpus_url, preview=preview)
    _echo_manifest(manifest)
    typer.echo(str(Path(out) / "manifest.json"))


def _echo_checklist(workdir: Path, repo: str) -> str:
    """Print the publish checklist rows and their readiness, and answer the readiness."""
    checklist = _entry("kullback.hub.checklist", "publish_checklist")(workdir, repo)
    readiness = _entry("kullback.hub.checklist", "readiness")(checklist)
    for row in checklist:
        typer.echo(f"[{'x' if row.done else ' '}] {row.name}: {row.detail}")
    typer.echo(readiness)
    return readiness


@app.command()
def publish(
    workdir: Path = WORKDIR,
    repo: str = typer.Option(..., "--repo", help="Dataset repository, as organisation/name."),
    name: Optional[str] = typer.Option(None, "--name", help="Name of the Environment; the default is the "
                                                            "last segment of --repo."),
    corpus: Optional[str] = CORPUS,
    corpus_license: Optional[str] = CORPUS_LICENSE,
    corpus_url: Optional[str] = CORPUS_URL,
    preview: bool = typer.Option(False, "--preview", help="Publish below the fidelity bar, with a banner "
                                                          "on the card saying so."),
    keep: Optional[Path] = typer.Option(None, "--keep", help="Keep the staged package here instead of a "  # noqa: B008
                                                             "temporary directory."),
    check: bool = typer.Option(False, "--check", help="Print the release checklist and upload nothing."),
):
    """Export the Environment, write its card and upload it as one commit, tagged with its round (D221).

    A release needs replay fidelity at or above 0.90 over Tasks; below that only --preview is
    allowed. Publishing again writes a new commit on the same repository and rewrites the card's
    numbers; older rounds stay reachable by their tags. With --check the command prints the
    checklist and uploads nothing.
    """
    readiness = _echo_checklist(workdir, repo)
    if check:
        if readiness == "not ready":
            raise typer.Exit(1)
        return
    push = _entry("kullback.hub.publish", "publish")
    hosted, manifest = push(workdir, repo, name=name, preview=preview, corpus=corpus,
                            corpus_license=corpus_license, corpus_url=corpus_url, keep=keep)
    _echo_manifest(manifest)
    typer.echo(f"{manifest.get('status', 'preview')} at {hosted.url}, tag {hosted.tag}")


@app.command()
def fetch(
    repo: str = typer.Argument(..., help="Dataset repository, as organisation/name."),
    out: Path = typer.Option(..., "--out", help="Directory the Environment is laid out in."),  # noqa: B008
    revision: Optional[str] = typer.Option(None, "--revision", help="Tag, branch or commit; the default "
                                                                     "is the newest."),
):
    """Download a published Environment, verify it against its content hash, and lay it out as a workdir.

    What lands is what `kullback run --workdir <out>` takes: the world, the Task list, the Verifiers
    and the manifest, with no builder state. A package that does not verify is not laid out.
    """
    pull = _entry("kullback.hub.publish", "fetch")
    manifest = pull(repo, out, revision=revision)
    _echo_manifest(manifest)
    missing = manifest.get("missing_run_inputs") or []
    if missing:
        typer.echo("this package cannot be run as a workdir: it is missing " + ", ".join(missing))
    typer.echo(f"{out}")


@app.command("read-domain")
def read_domain(
    workdir: Path = WORKDIR,
    source: Optional[list[str]] = typer.Option(None, "--source",  # noqa: B008
                                               help="A public page to read; repeatable (D225)."),
    sources_file: Optional[Path] = typer.Option(None, "--sources",  # noqa: B008
                                                help="A file of source URLs, one per line."),
    domain_line: Optional[str] = typer.Option(None, "--domain",
                                              help="One line describing the business; the model "
                                                   "names the public pages worth reading for it."),
    depth: int = typer.Option(1, "--depth", help="How far links on the same host are followed."),
    search: Optional[str] = typer.Option(None, "--search",
                                         help="A search query; runs only where a search key is set "
                                              "in KULLBACK_SEARCH_KEY, whose value is never printed."),
    corpus_url: Optional[str] = typer.Option(None, "--corpus-url",
                                             help="The source corpus's own repository or paper. "
                                                  "Nothing under it is ever read."),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude",  # noqa: B008
                                                help="A URL nothing under may be read; repeatable."),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="Model id for the reader and the mapper, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Spend ceiling (D86)."),
):
    """Read a domain's public material into task archetypes and coverage gaps (D225).

    What is fetched is kept only in a content addressed cache under the workdir, never in the
    package and never in git. Anything under the source corpus's own repository or paper is refused
    and counted, whoever proposed it: a benchmark that reads its own published task list back in has
    attested nothing.
    """
    from kullback import domain as domain_mod

    urls = list(source or [])
    if sources_file:
        urls += [line.strip() for line in Path(sources_file).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
    if not urls and not domain_line and not search:
        typer.echo("nothing to read: pass --source, --sources, --domain or --search")
        raise typer.Exit(2)
    reader = _live_model(model, base_url) if model else None
    if reader is not None:
        reader = _wrapped_model(reader, "domain_reading", Path(workdir), ceiling_usd,
                                model_id=model)
    body = domain_mod.read(workdir, sources=urls, depth=depth, corpus_url=corpus_url,
                           exclude=list(exclude or []), reader=reader, mapper=reader, judge=reader,
                           description=str(domain_line or ""), search_query=str(search or ""))
    counts = dict(body.get("counts") or {})
    for note in (body.get("sources") or {}).get("notes") or []:
        typer.echo(note)
    for row in (body.get("sources") or {}).get("refused") or []:
        typer.echo(f"refused, {row.get('reason')}: {row.get('url')}")
    for name in ("sources_read", "sources_named", "sources_refused", "archetypes_extracted",
                 "archetypes_copied", "archetypes_contaminated", "archetypes_folded",
                 "archetypes_kept", "archetypes_mapped", "archetype_gaps"):
        typer.echo(f"{name}: {counts.get(name, 0)}")
    typer.echo("")
    typer.echo("| source | archetypes | mapped |")
    typer.echo("| --- | --- | --- |")
    for row in domain_mod.per_source(body):
        typer.echo(f"| {row['source']} | {row['archetypes']} | {row['mapped']} |")




STEER_KINDS = ("nudge", "tell", "stop")
STEER_OUTCOMES = {"queued": "queued", "cancel_asked": "cancel asked", "refused": "refused"}


@app.command()
def steer(
    workdir: Path = typer.Argument(..., help="The workdir of the live build to steer."),  # noqa: B008
    kind: str = typer.Argument(..., help="nudge (before the next model turn), tell (when the run would "
                                        "stop) or stop (at the next step)."),
    text: str = typer.Argument("", help="What to tell the Builder; stop takes none."),
    timeout: float = typer.Option(30.0, "--timeout", help="Seconds to wait for the build's answer."),
    session: str = typer.Option("", "--session", help="The session to steer (its pid) when several "
                                "builds share the workdir; names the only live build itself."),
):
    """Steer a live build from any process: the request goes on the workdir's bus and the build answers.

    The build, wherever it was started, reads the request, queues it on its harness and writes an
    ack beside it; this command prints that ack. No ack within the timeout means no live build is
    reading that bus. The request names the live build's session, so a second build on the same
    workdir never acts on it; with several live and no session named, this refuses and lists them.
    """
    if kind not in STEER_KINDS:
        typer.echo(f"steer takes nudge, tell or stop, not {kind}")
        raise typer.Exit(2)
    steer_mod = importlib.import_module("kullback.agent.steer")
    from kullback.tui import live_heartbeats

    try:
        target = steer_mod.target_session(live_heartbeats(workdir), session)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    request_id = steer_mod.request(workdir, kind, text, sender="kullback steer", session=target)
    ack = steer_mod.wait_for_ack(workdir, request_id, timeout)
    if ack is None:
        typer.echo(f"No live build answered in {workdir} within {timeout:g} seconds, so nothing was "
                   "steered; `kullback tui` lists the running builds under /sessions.")
        raise typer.Exit(1)
    typer.echo(f"{kind} {STEER_OUTCOMES[ack.outcome]}" + (f": {ack.reason}" if ack.reason else ""))
    if ack.outcome == "refused":
        raise typer.Exit(1)

@app.command()
def events(
    workdir: Path = WORKDIR,
    follow: bool = typer.Option(False, "--follow", help="Keep printing as the bus grows, until ctrl-c."),
    since: int = typer.Option(0, "--since", help="Print only the records after this seq."),
    as_json: bool = typer.Option(False, "--json", help="Print each bus record as its JSON line."),
):
    """Print a build's event stream off workdir/bus.jsonl, one readable line per event or raw JSON.

    The readable lines are the screen's transcript, so a script, a second dashboard or a pipe into
    jq reads the same story the screen shows. Reading never touches the build.
    """
    bus_cls = _entry("kullback.agent.bus", "Bus")
    bus = bus_cls(workdir / "bus.jsonl")
    transcript = None if as_json else _entry("kullback.tui", "Transcript")(
        on_line=lambda line, _style: typer.echo(line))
    records = bus.tail(since) if follow else bus.replay(since)
    try:
        for record in records:
            if transcript is None:
                typer.echo(record.model_dump_json())
            else:
                transcript.event(record.event)
    except KeyboardInterrupt:
        pass


@app.command()
def attach(
    workdir: Path = typer.Argument(Path("."), help="The workdir of the build to follow."),  # noqa: B008
):
    """Open the screen on a workdir and follow its live build at once, wherever it was started.

    With no live build there, the screen says so and shows the build's status instead.
    """
    _load_keys()
    _entry("kullback.tui", "loop")(workdir=_default_workdir(workdir), attach=True)


@app.command()
def tui(
    workdir: Path = WORKDIR,
    model: Optional[str] = typer.Option(None, "--model", help="Builder model id, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Per-build spend ceiling (D86)."),
):
    """Open the kullback screen: one build, its stages, its gates and its spend, while it runs."""
    _load_keys()
    _entry("kullback.tui", "loop")(workdir=_default_workdir(workdir), model=model, base_url=base_url,
                                       ceiling_usd=ceiling_usd)


# A directory counts as a build's workdir when it holds any record a build writes. The screen
# answers from files, so opening on an empty directory only ever says 'no build yet'.
WORKDIR_RECORDS = ("budget.json", "rounds.json", "gates.json", "tasks.json", "db.json",
                   "pipeline")


def _default_workdir(workdir: Path) -> Path:
    """The workdir the screen opens: the flag when given, else ./work when it holds a build,
    else the current directory. `kullback tui` with no flag opens the build in front of you."""
    if str(workdir) != ".":
        return workdir
    candidate = Path("work")
    if candidate.is_dir() and any((candidate / record).exists() for record in WORKDIR_RECORDS):
        return candidate
    return workdir


if __name__ == "__main__":
    app()
