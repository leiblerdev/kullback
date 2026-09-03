"""The seven commands of the Harness: ingest, build, freeze-runner, run, verdict, regrade and report, each
reading and writing records under one workdir with no hidden state."""

from __future__ import annotations

import contextlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from kullback.report import coverage_rows, load, load_tool_sigs, write_report
from kullback.runner import heartbeat
from kullback.runner.records import (
    EntitySchema,
    Environment,
    RunnerVersion,
    Task,
    Verifier,
    as_dict,
)

app = typer.Typer(add_completion=False,
                  help="Build an Environment from customer traces, re-run it, grade it, report it.")

WORKDIR = typer.Option(Path("."), "--workdir", "-w", help="Directory every record is read from and written to.")
JUDGE_MODEL = typer.Option(None, "--judge-model",
                          help="Model id for the two agentic judges, as provider/model. Without it, judge atoms "
                               "are left unevaluated and a failure keeps no cause.")
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


def _judges(model_id: Optional[str], base_url: Optional[str] = None):
    """The two agentic judges of D92, or None when the caller named no judge model.

    Both are constructed here and never inside the Runner: `verdict.py` takes judge answers as data
    and calls no model itself (D76, build brief rule 2). The second judge is the same adapter under
    judge.py's own second persona, which is the D97 default; a second model id would be better and is
    what `--judge-model` should grow when a customer has two providers configured.
    """
    if not model_id:
        return None
    first = _entry("kullback.runner.judge", "AgenticJudge")(_live_model(model_id, base_url),
                                                           name=f"{model_id}:a")
    return first, _entry("kullback.runner.judge", "third_judge")(first)


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


def _rescorer(score_one, verifier: Verifier, canon_value, out_dir: Path, judge_version, common: dict):
    """The callback `_name_causes` scores one Run with, once a judge has named its cause (D88)."""

    def rescore(path: Path, **extra):
        return score_one(path, verifier, canon_value, out_dir=out_dir,
                         judge_version=judge_version, **common, **extra)

    return rescore


def _score(workdir: Path, task_id: Optional[str], what: str, use_queue: bool = False,
           judge_model: Optional[str] = None, base_url: Optional[str] = None) -> None:
    """Score stored Runs against their Task's Verifier. Nothing is re-executed; the version cache makes a repeat free.

    With `--judge-model` the judge atoms of each Verifier are answered before the Verdict and the
    cause of each unexplained failure after it (D76, D88). Without one, a judge atom stays
    unevaluated and a failure keeps `cause_pending_judge`; both are said out loud rather than
    silently passing.
    """
    score = _entry("kullback.runner.regrade", "regrade")
    score_one = _entry("kullback.runner.regrade", "regrade_run")
    regrade_gate = _entry("kullback.gates.artifacts", "regrade_gate")
    judge_version = _entry("kullback.runner.judge", "JUDGE_VERSION") if judge_model else None
    judges = _judges(judge_model, base_url)
    canon_value = _entry("kullback.runner.canon", "canon_value")
    env_path, version_path = Path(workdir) / "environment.json", Path(workdir) / "runner_version.json"
    environment = _load(env_path, Environment) if env_path.is_file() else None
    version = _load(version_path, RunnerVersion).runner_version if version_path.is_file() else None
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
                      write_tools=write_tools or None, flagged_tools=flagged_tools)
        verdicts = score(paths, verifier, canon_value, out_dir=out_dir,
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
                             _rescorer(score_one, verifier, canon_value, out_dir, judge_version, common))
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
          workdir: Path = WORKDIR):
    """Store the customer's files byte for byte and derive Traces from them (D66)."""
    ingest_file = _entry("kullback.builder.ingest", "ingest_file")
    summaries = [ingest_file(path, workdir) for path in files]
    _write(Path(workdir) / "ingest_summary.json", summaries)


def _live_model(model_id: str, base_url: Optional[str]):
    """One live adapter, or the refusal in words. provider.live_model is the single place the
    live-call flag is ever set, so the screen and the CLI refuse for the same reason."""
    try:
        return _entry("kullback.ai.provider", "live_model")(model_id, base_url)
    except RuntimeError as error:
        typer.echo(str(error))
        raise typer.Exit(2) from None


@app.command()
def build(
    workdir: Path = WORKDIR,
    iterate: bool = typer.Option(False, "--iterate", help="Resume the content-addressed build and keep improving."),
    model: Optional[str] = typer.Option(None, "--model", help="Builder model id, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    files: Optional[list[Path]] = typer.Option(None, "--file", help="Customer export to ingest first."),  # noqa: B008
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Per-build spend ceiling (D86)."),
    grow: Optional[list[str]] = typer.Option(None, "--grow",  # noqa: B008
                                             help="Grow a table to this many rows with synthetic ones, "
                                                  "as table=count; repeatable (D107)."),
    grow_seed: int = typer.Option(0, "--grow-seed", help="Seed for the synthetic rows."),
    probe_limit: Optional[int] = typer.Option(None, "--probe-limit",
                                              help="Loophole probes per build (D79 check 6); default every Task."),
    rerolls: int = typer.Option(3, "--rerolls", help="Frontier re-rolls per Task beside its recordings (D112); "
                                                    "0 turns them off."),
    # 8 is the default the CLI sets (D118); build() itself defaults to 1 so a scripted model in a
    # test still answers in the order it was given. The number lives here, not in parallel.py.
    workers: int = typer.Option(8, "--workers", help="Model calls in flight at once across tools, policy "
                                                     "sentences, Intents and re-rolls (D118); 1 runs them in a line."),
    target: str = typer.Option("environment", "--target", help="What to build: environment (everything), "
                                                                 "or one stage or artifact by name."),
    agent: bool = typer.Option(False, "--agent", help="Let the --model drive both sessions, the Builder's and "
                                                        "the Examiner's; without it code issues the tool calls."),
    stall_rounds: int = typer.Option(1, "--stall-rounds", help="Rounds that move no gate count before the loop "
                                                              "exits stalled (D126)."),
    allowance_usd: Optional[float] = typer.Option(None, "--allowance-usd",
                                                  help="Per-agent spend allowance per round; the default is "
                                                       "each agent's own round-1 spend from round 2 on (D123)."),
):
    """Run the Builder and the Examiner in rounds over the ingested Traces and write the Environment.

    Both agents are extensions on the agent core, driven in turns on one stream (D128): the Builder
    builds the target, the Examiner derives and examines the Verifiers, and the round ends with the
    counts the gates report. By default code issues the tool calls, so the build is deterministic and
    byte-identical offline; `--agent` hands both sessions to the model.
    """
    adapter = _live_model(model, base_url) if model else None
    if agent and adapter is None:
        raise typer.BadParameter("--agent needs --model: a model has to drive the session")
    search = _entry("kullback.builder.search", "search_for")(workdir)  # None unless live is on or a memo exists
    # The screen lists running builds from these heartbeats; the pid tells it who is alive.
    heartbeat.beat(workdir, model, "running")
    try:
        # The provider owns an http client when it made one; close it on the way out rather than at exit.
        with contextlib.closing(search) if search is not None else contextlib.nullcontext():
            result = _entry("kullback.rounds", "run_rounds")(
                workdir=workdir, iterate=iterate, model=adapter, files=list(files or []),
                ceiling_usd=ceiling_usd, grow=_grow_targets(grow), grow_seed=grow_seed,
                probe_limit=probe_limit, rerolls=rerolls, search=search, workers=workers, target=target,
                agent_model=adapter if agent else None, stall_rounds=stall_rounds,
                allowance_usd=allowance_usd, subscribers=[_echo_round])
    except Exception:
        heartbeat.beat(workdir, model, "failed")
        raise
    heartbeat.beat(workdir, model, "failed" if result.get("failed") else "done",
                   exit=result.get("exit"))
    if isinstance(result, dict) and result.get("exit"):
        count = len(result.get("rounds") or [])
        typer.echo(f"exit: {result['exit']} after {count} round{'' if count == 1 else 's'}")
    typer.echo(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get("failed"):
        # A stalled exit on a broken agent contract is a failed run, not a quiet one: CI and
        # scripts must see it. Plain stalled (the gates, no progress) still exits zero.
        raise typer.Exit(1)


def _round_line(counts: dict) -> str:
    """One round's counts as one line, every number a gate's (D126)."""
    compactions = counts.get("fallback_compactions") or {}
    spend = counts.get("spend") or {}
    return (f"fidelity {counts.get('fidelity', 0)}/{counts.get('tasks', 0)} tasks, "
            f"trusted {counts.get('trusted', 0)}, refused {counts.get('refused_count', 0)}, "
            f"assisted runs {counts.get('assisted_runs', 0)}, probes passing {counts.get('probes_passing', 0)}, "
            f"compactions builder {compactions.get('builder', 0)} examiner {compactions.get('examiner', 0)}, "
            f"spend ${float(spend.get('total') or 0.0):.4f}")


def _echo_round(event: Any) -> None:
    """The subscriber the build hands the driver: a line per round as each round ends."""
    if getattr(event, "type", None) == "round_end":
        typer.echo(f"round {event.round}: {_round_line(dict(event.counts or {}))}")


def _grow_targets(pairs: Optional[list[str]]) -> dict[str, int]:
    """`--grow users=500 --grow orders=1000` as {table: count}; a malformed pair is refused."""
    out: dict[str, int] = {}
    for pair in pairs or []:
        table, sep, count = pair.partition("=")
        if not sep or not table or not count.isdigit():
            raise typer.BadParameter(f"--grow takes table=count, got {pair!r}")
        out[table.strip()] = int(count)
    return out


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


@app.command()
def run(
    workdir: Path = WORKDIR,
    task: str = typer.Option(..., "--task", help="Task id to run."),
    model: str = typer.Option(..., "--model", help="Candidate model id, as provider/model."),
    count: int = typer.Option(1, "--count", help="Runs per Task."),
    seed: int = typer.Option(0, "--seed", help="First seed; the batch counts up from it."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
):
    """Run a Candidate against the built Environment and write one JSONL per Run."""
    result = _entry("kullback.builder.build", "run_batch")(
        workdir=workdir, task_id=task, model=_live_model(model, base_url), count=count, seed=seed)
    typer.echo(json.dumps(result, default=str))


@app.command()
def verdict(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id."),
            judge_model: Optional[str] = JUDGE_MODEL, base_url: Optional[str] = BASE_URL):
    """Score the stored Runs of one Task, or of every Task, on their End state."""
    _score(Path(workdir), task, "scored", judge_model=judge_model, base_url=base_url)


@app.command()
def regrade(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id."),
            judge_model: Optional[str] = JUDGE_MODEL, base_url: Optional[str] = BASE_URL):
    """Re-score stored Runs against the current Environment and Verifier versions, without re-executing them.

    A Run whose equivalence entry a person overturned is in canon.py's regrade queue (D84): its
    versions have not moved, so only the queue makes it score again.
    """
    _score(Path(workdir), task, "regraded", use_queue=True, judge_model=judge_model, base_url=base_url)


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


@app.command()
def tui(
    workdir: Path = WORKDIR,
    model: Optional[str] = typer.Option(None, "--model", help="Builder model id, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Per-build spend ceiling (D86)."),
):
    """Open the kullback screen: one build, its stages, its gates and its spend, while it runs."""
    _entry("kullback.tui", "loop")(workdir=workdir, model=model, base_url=base_url, ceiling_usd=ceiling_usd)


if __name__ == "__main__":
    app()
