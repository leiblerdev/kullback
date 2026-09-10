"""The commands of the Harness: ingest, build, freeze-runner, run, verdict, regrade, report, status,
difficulty, export, publish and fetch, each reading and writing records under one workdir with no
hidden state."""

from __future__ import annotations

import contextlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from kullback import difficulty, round_snapshot, rounds
from kullback.report import coverage_rows, load, load_tool_sigs, write_report
from kullback.runner import feed, heartbeat
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


def _rescorer(score_one, verifier: Verifier, canon_value, out_dir: Path, judge_version, common: dict):
    """The callback `_name_causes` scores one Run with, once a judge has named its cause (D88)."""

    def rescore(path: Path, **extra):
        return score_one(path, verifier, canon_value, out_dir=out_dir,
                         judge_version=judge_version, **common, **extra)

    return rescore


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
    judge_model: Optional[str] = typer.Option(None, "--judge-model",
                                              help="Model id for the build's judge, as provider/model (D160); "
                                                   "the default is --model."),
    second_judge_model: Optional[str] = typer.Option(None, "--second-judge-model",
                                                     help="Model id for the second judge, as provider/model "
                                                          "(D160); the default is the judge's own model under "
                                                          "a second persona."),
    judge_agent: bool = typer.Option(False, "--judge-agent",
                                     help="The reference judge is an agent with a bounded look over the Task; "
                                          "default is the one-shot judge."),
    user_model: Optional[str] = typer.Option(None, "--user-model",
                                             help="Model id for the Simulated user, as provider/model (D232); "
                                                  "without it the rule-driven user drives every Run."),
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
    fidelity_stall: int = typer.Option(3, "--fidelity-stall", help="Rounds without a rise in fidelity before the "
                                                                  "loop exits stalled, whatever the other counts "
                                                                  "do; 0 turns it off (D169)."),
    max_rounds: int = typer.Option(12, "--max-rounds", help="Rounds after which the loop exits max_rounds; "
                                                            "0 is no cap (D169)."),
    allowance_usd: Optional[float] = typer.Option(None, "--allowance-usd",
                                                  help="Per-agent spend allowance per round; the default is "
                                                       "each agent's own round-1 spend from round 2 on (D123)."),
):
    """Run the Builder and the Examiner in rounds over the ingested Traces and write the Environment.

    Both agents are extensions on the agent core, driven in turns on one stream (D128): the Builder
    builds the target, the Examiner derives and examines the Verifiers, and the round ends with the
    counts the gates report. By default code issues the tool calls, so the build is deterministic and
    byte-identical offline; `--agent` hands both sessions to the model. The judge is a model of its
    own when `--judge-model` names one (D160), so the model that writes the Environment need not be
    the one that rules on it. The judge that settles a Task whose Runs disagree is the one-shot judge
    unless `--judge-agent` asks for the one with a bounded look, which reads before it rules and
    costs References (D185). The Simulated user is rule-driven unless `--user-model` names the
    model that drives it, in which case the agent user drives the Tasks it beats the rules on (D232).
    """
    adapter = _live_model(model, base_url) if model else None
    if agent and adapter is None:
        raise typer.BadParameter("--agent needs --model: a model has to drive the session")
    if second_judge_model and not (judge_model or model):
        raise typer.BadParameter("--second-judge-model needs a first judge: name --judge-model or --model")
    judge_adapter = _live_model(judge_model, base_url) if judge_model else None
    second_judge_adapter = _live_model(second_judge_model, base_url) if second_judge_model else None
    # The Simulated user needs no Builder model beside it, so --user-model without --model is
    # allowed (D232). Without a Builder model there are no re-rolls, so the user model pays only
    # for the offline driver scoring.
    user_adapter = _live_model(user_model, base_url) if user_model else None
    search = _entry("kullback.builder.search", "search_for")(workdir)  # None unless live is on or a memo exists
    # The screen lists running builds from these heartbeats; the pid tells it who is alive. The
    # pulse keeps beating while the build runs so a screen watching from another directory sees
    # the spend move, rather than a stale $0.0000 until the build is over.
    # The feed is this build's story, opened here and appended to as it goes: /watch reads it to
    # show the calls as they happen instead of a board that only moves when a stage ends.
    feed.start(workdir, model=model, target=target, ceiling_usd=ceiling_usd)
    pulse = heartbeat.pulse(workdir, model, "running")
    try:
        # The provider owns an http client when it made one; close it on the way out rather than at exit.
        with contextlib.closing(search) if search is not None else contextlib.nullcontext():
            result = _entry("kullback.rounds", "run_rounds")(
                workdir=workdir, iterate=iterate, model=adapter, judge_model=judge_adapter,
                second_judge_model=second_judge_adapter, judge_agent=judge_agent,
                user_agent_model=user_adapter, files=list(files or []),
                ceiling_usd=ceiling_usd, grow=_grow_targets(grow), grow_seed=grow_seed,
                probe_limit=probe_limit, rerolls=rerolls, search=search, workers=workers, target=target,
                agent_model=adapter if agent else None, stall_rounds=stall_rounds,
                fidelity_stall=fidelity_stall, max_rounds=max_rounds,
                allowance_usd=allowance_usd, subscribers=[_echo_round])
    except Exception:
        pulse.stop()
        heartbeat.beat(workdir, model, "failed")
        raise
    pulse.stop()
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


def _regrouped(counts: dict) -> str:
    """What the round says about a grouping that no longer matches the frozen one (D216).

    Nothing at all on the ordinary round, because the split of one recording set reproduces and a
    line that said so every round would only be noise. Where it did move, the line names the input
    that moved it, which is the whole of the finding: a corpus that grew regrouped for a reason, and
    the harness raises rather than print anything else here.
    """
    moved = str(counts.get("tasks_grouping_moved") or "")
    return f" regrouped ({moved} moved)" if moved else ""


def _ended_by(counts: dict) -> str:
    """How a round ended where a beat raised in it (D231), and nothing at all where none did.

    The counts beside it are what the round measured before the raise, which is a real reading and
    not a default; this is what says the round stopped short of the rest of its work.
    """
    row = counts.get(rounds.BEAT_ERROR) or {}
    beat = str(row.get("beat") or "")
    if not beat:
        return ""
    kind = str(row.get("kind") or "")
    return f"ended by {beat} error" + (f" ({kind})" if kind else "") + ", "


def _round_line(counts: dict) -> str:
    """One round's counts as one line, every number a gate's (D126)."""
    compactions = counts.get("fallback_compactions") or {}
    spend = counts.get("spend") or {}
    return (_ended_by(counts)
            + f"fidelity {counts.get('fidelity', 0)}/{counts.get('tasks', 0)} tasks, "
            f"trusted {counts.get('trusted', 0)}, refused {counts.get('refused_count', 0)}, "
            f"assisted runs {counts.get('assisted_runs', 0)}, probes passing {counts.get('probes_passing', 0)}, "
            f"tasks frozen {counts.get('tasks_frozen', 0)} added {counts.get('tasks_added', 0)} "
            f"(${float(counts.get('tasks_added_cost') or 0.0):.4f}) "
            f"frozen only {counts.get('tasks_frozen_only', 0)} cleared {counts.get('tasks_cleared', 0)}"
            f"{_regrouped(counts)}, "
            f"compactions builder {compactions.get('builder', 0)} examiner {compactions.get('examiner', 0)}, "
            f"spend ${float(spend.get('total') or 0.0):.4f}, cache saved ${float(spend.get('cache_saved') or 0.0):.4f}, "
            f"buckets: {difficulty.round_summary(counts.get('buckets') or [])}"
            + _synthetic_line(counts))


def _synthetic_line(counts: dict) -> str:
    """What the round's synthetic store holds, appended to the line under its own names (D224).

    A round that generated nothing says nothing, so a reader never mistakes a build that was never
    asked for synthetic Tasks for one that asked and got none.
    """
    if not counts.get("synthetic_tasks"):
        return ""
    return (f", synthetic {counts.get('synthetic_tasks', 0)} verified "
            f"{counts.get('synthetic_verified', 0)}")


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


def _json_at(root: Path, name: str) -> dict:
    """One JSON record of a workdir, or nothing where the file is missing or half-written."""
    path = root / name
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
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
            writer = _entry("kullback.builder.build", "_wrap")(
                writer, "synthetic_intent", Path(workdir),
                _entry("kullback.builder.build", "_ceiling")(Path(workdir), ceiling_usd),
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
        writer = _entry("kullback.builder.build", "_wrap")(
            writer, "synthetic_intent", Path(workdir),
            _entry("kullback.builder.build", "_ceiling")(Path(workdir), ceiling_usd), model_id=model)
    body = _entry("kullback.synthesise", "synthesise")(workdir, targets, seed=seed, model=writer)
    for line in _synthesis_lines(body):
        typer.echo(line)


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
    agent_mod = importlib.import_module("kullback.user.agent")
    fidelity = importlib.import_module("kullback.user.fidelity")
    budget = importlib.import_module("kullback.runner.budget")
    model = _live_model(model_id, base_url)
    if ceiling is not None:
        model = budget.BudgetedModel(model, stage="user_fidelity", workdir=workdir,
                                     model_id=model_id, ceiling=ceiling, cap_context=True)
    writes = fidelity.write_tools_of(workdir)
    vocab = fidelity.vocabulary_of(workdir)

    def make(ctx, fallback, record_values):
        return agent_mod.AgentUser(ctx, fallback, model, vocab=vocab, write_tools=writes,
                                   record_values=record_values)

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
):
    """Export the Environment, write its card and upload it as one commit, tagged with its round (D221).

    A release needs replay fidelity at or above 0.90 over Tasks; below that only --preview is
    allowed. Publishing again writes a new commit on the same repository and rewrites the card's
    numbers; older rounds stay reachable by their tags.
    """
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
        reader = _entry("kullback.builder.build", "_wrap")(
            reader, "domain_reading", Path(workdir),
            _entry("kullback.builder.build", "_ceiling")(Path(workdir), ceiling_usd), model_id=model)
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




@app.command()
def tui(
    workdir: Path = WORKDIR,
    model: Optional[str] = typer.Option(None, "--model", help="Builder model id, as provider/model."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible model."),
    ceiling_usd: Optional[float] = typer.Option(None, "--ceiling-usd", help="Per-build spend ceiling (D86)."),
):
    """Open the kullback screen: one build, its stages, its gates and its spend, while it runs."""
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
