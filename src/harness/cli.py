"""The seven commands of the Harness: ingest, build, freeze-runner, run, verdict, regrade and report, each reading and writing records under one workdir with no hidden state."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from harness.shared.records import Environment, RunnerVersion, Task, Verifier, as_dict
from harness.shared.report import coverage_rows, load, load_tool_sigs, write_report

app = typer.Typer(add_completion=False, help="Build an Environment from customer traces, re-run it, grade it, report it.")

WORKDIR = typer.Option(Path("."), "--workdir", "-w", help="Directory every record is read from and written to.")
RUNNER_FILES = ("loop.py", "route.py", "verdict.py")  # what freeze-runner hashes (D89)


def _entry(path: str, name: str):
    """One function of another Harness module, imported only when a command runs, or one clear message."""
    try:
        function = getattr(importlib.import_module(path), name, None)
    except Exception as error:
        function, name = None, f"{name} ({error})"
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


def _score(workdir: Path, task_id: Optional[str], what: str) -> None:
    """Score stored Runs against their Task's Verifier. Nothing is re-executed; the version cache makes a repeat free."""
    score = _entry("harness.runner.regrade", "regrade")
    canon_value = _entry("harness.shared.canon", "canon_value")
    env_path, version_path = Path(workdir) / "environment.json", Path(workdir) / "runner_version.json"
    environment = _load(env_path, Environment) if env_path.is_file() else None
    version = _load(version_path, RunnerVersion).runner_version if version_path.is_file() else None
    sigs = load_tool_sigs(workdir)  # without these the extra-write and D70 checks never fire
    write_tools = {sig.name for sig in sigs if sig.kind == "write"}
    flagged_tools = {sig.name for sig in sigs if sig.unclassified}
    if sigs and not write_tools:
        typer.echo("no write tools among the mined ToolSigs: side effects are not checked")
    elif not sigs:
        typer.echo("no ToolSigs on disk: side effects are not checked")
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
        score(paths, verifier, canon_value, out_dir=Path(workdir) / "verdicts" / task.id,
              environment=environment, runner_version=version,
              write_tools=write_tools or None, flagged_tools=flagged_tools)
        typer.echo(f"task {task.id}: {what} {len(paths)} Runs")


def runner_version(routing_config: Optional[Path] = None) -> RunnerVersion:
    """The content hash of loop.py, route.py, verdict.py and the routing config, as one RunnerVersion record.

    validate.py computes it, so freeze-runner and the gate that checks a Run can never disagree.
    """
    from harness.runner.validate import runner_version as compute

    config = Path(routing_config).read_text(encoding="utf-8") if routing_config else None
    return compute(Path(__file__).parent, config,
                   created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


@app.command()
def ingest(files: list[Path] = typer.Argument(..., help="The customer's export files."), workdir: Path = WORKDIR):
    """Store the customer's files byte for byte and derive Traces from them (D66)."""
    ingest_file = _entry("harness.builder.ingest", "ingest_file")
    summaries = [ingest_file(path, workdir) for path in files]
    _write(Path(workdir) / "ingest_summary.json", summaries)


@app.command()
def build(
    workdir: Path = WORKDIR,
    iterate: bool = typer.Option(False, "--iterate", help="Resume the content-addressed build and keep improving."),
):
    """Run the Builder pipeline over the ingested Traces and write the Environment."""
    result = _entry("harness.runner.pipeline", "build")(workdir=workdir, iterate=iterate)
    typer.echo(json.dumps(result, default=str))


@app.command("freeze-runner")
def freeze_runner(
    workdir: Path = WORKDIR,
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
    routing_config: Optional[Path] = typer.Option(None, "--routing-config", help="Routing config to hash in."),
    by: str = typer.Option("unknown", "--by", help="Who confirmed the freeze."),
):
    """Write the RunnerVersion that every later Verdict carries, after a person confirms it."""
    version = runner_version(routing_config)
    typer.echo(f"runner version {version.runner_version} over {len(version.file_hashes)} files")
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
):
    """Run a Candidate against the built Environment and write one JSONL per Run."""
    result = _entry("harness.runner.pipeline", "run_batch")(
        workdir=workdir, task_id=task, model=model, count=count, seed=seed)
    typer.echo(json.dumps(result, default=str))


@app.command()
def verdict(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id.")):
    """Score the stored Runs of one Task, or of every Task, on their End state."""
    _score(Path(workdir), task, "scored")


@app.command()
def regrade(workdir: Path = WORKDIR, task: Optional[str] = typer.Option(None, "--task", help="One Task id.")):
    """Re-score stored Runs against the current Environment and Verifier versions, without re-executing them."""
    _score(Path(workdir), task, "regraded")


@app.command()
def report(
    workdir: Path = WORKDIR,
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the Markdown."),
    batch: bool = typer.Option(False, "--batch", help="Report one Run batch instead of a build."),
):
    """Write the Markdown report: the Environment first, then the numbers per Task, then a suggestion (D85)."""
    data = load(workdir)
    if not data.task_coverage and data.tasks:
        # D96's two headline numbers: covered Tasks over the frozen list, and the same Run-weighted.
        # The rule lives in validate.py so the scorecard and the report count the same thing.
        task_coverage = _entry("harness.runner.validate", "task_coverage")
        status = json.loads((Path(workdir) / "task_status.json").read_text(encoding="utf-8")) \
            if (Path(workdir) / "task_status.json").is_file() else {}
        computed = task_coverage(data.tasks, data.runs, status)
        data.task_coverage = coverage_rows(
            data.tasks, {row["task_id"]: row["reason"] for row in computed["uncovered"]})
    if batch:
        data.kind = "batch"
        data.title = "Run batch report"
    target = Path(out) if out else Path(workdir) / "report.md"
    typer.echo(str(write_report(data, target.parent, target.name)))


if __name__ == "__main__":
    app()
