"""The seven commands of cli.py: every one reads and writes records under a --workdir, and freeze-runner asks first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kullback import cli
from kullback.runner.records import Environment, RunnerVersion, Task, Verdict, Verifier, as_dict

runner = CliRunner()


@pytest.fixture
def fake_modules(monkeypatch):
    """Replace the lazy entry-point lookup with a recorder, for the commands whose forwarding is the
    only thing on the wire.

    Everything a real run leaves on disk is asserted on the real path instead (verdict, regrade and
    report below). What is left is `build` and `run`, whose real work is a whole pipeline or a live
    adapter, and the three verdict kwargs (write_tools, flagged_tools, schema) that the stored
    Verdict does not carry. cli._score reads the regrade gate through getattr for this stub's sake.
    `run_rounds` answers the way the driver does: two rounds ended through the subscribers the
    command handed it (ROUND_COUNTS), then the dict with the rounds and the exit.
    """
    from kullback.agent.events import RoundEnd

    calls: dict[str, list] = {}

    def run_rounds(**kwargs):
        records = []
        for n, counts in enumerate(ROUND_COUNTS, start=1):
            exit_ = "stalled" if n == len(ROUND_COUNTS) else None
            for subscriber in kwargs.get("subscribers", ()):
                subscriber(RoundEnd(round=n, counts=counts, exit=exit_))
            records.append({"round": n, "counts": counts, "exit": exit_})
        return {"status": "complete", "rounds": records, "exit": "stalled", "trusted": ["t1"], "refused": {}}

    def entry(path: str, name: str):
        def fn(*args, **kwargs):
            calls.setdefault(f"{path}.{name}", []).append({"args": args, "kwargs": kwargs})
            # search_for returns a provider the command closes, or None when there is nothing to
            # search with; the stub answers None, which is the case a build without live or a memo hits.
            if name == "search_for":
                return None
            return run_rounds(**kwargs) if name == "run_rounds" else {"ok": True}
        return fn

    monkeypatch.setattr(cli, "_entry", entry)
    return calls


ROUND_COUNTS = [
    {"fidelity": 1, "tasks": 2, "trusted": 0, "refused_count": 0, "assisted_runs": 3, "probes_passing": 0,
     "fallback_compactions": {"builder": 0, "examiner": 0}, "spend": {"builder": 0.0, "examiner": 0.0, "total": 0.0}},
    {"fidelity": 2, "tasks": 2, "trusted": 1, "refused_count": 1, "assisted_runs": 3, "probes_passing": 4,
     "fallback_compactions": {"builder": 1, "examiner": 0}, "spend": {"builder": 0.5, "examiner": 0.75, "total": 1.25}},
]


def invoke(*args, **kwargs):
    return runner.invoke(cli.app, list(args), **kwargs)


# --- the command list -------------------------------------------------------

def test_help_lists_every_command():
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("ingest", "build", "freeze-runner", "run", "verdict", "regrade", "report"):
        assert command in result.output


def test_every_command_takes_a_workdir():
    for command in ("ingest", "build", "freeze-runner", "run", "verdict", "regrade", "report"):
        result = invoke(command, "--help")
        assert result.exit_code == 0, command
        assert "--workdir" in result.output, command


# --- ingest and build -------------------------------------------------------

def test_ingest_passes_each_file_to_the_ingest_module(tmp_path, workdir, fake_modules):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for path in (first, second):
        path.write_text("{}", encoding="utf-8")
    result = invoke("ingest", str(first), str(second), "--workdir", str(workdir))
    assert result.exit_code == 0
    calls = fake_modules["kullback.builder.ingest.ingest_file"]
    assert [call["args"][0] for call in calls] == [first, second]


@pytest.mark.parametrize("extra_args,iterate", [(["--iterate"], True), ([], False)])
def test_build_passes_iterate_through_to_the_builder(workdir, fake_modules, extra_args, iterate):
    result = invoke("build", "--workdir", str(workdir), *extra_args)
    assert result.exit_code == 0
    call = fake_modules["kullback.rounds.run_rounds"][0]
    assert call["kwargs"]["iterate"] is iterate


def test_build_is_driven_by_code_unless_agent_is_asked_for(workdir, fake_modules):
    """`kullback build` issues build(target) itself, so the offline build stays deterministic; the
    model drives only under --agent, and --agent without a model has nothing to drive with."""
    assert invoke("build", "--workdir", str(workdir), "--target", "cluster").exit_code == 0
    kwargs = fake_modules["kullback.rounds.run_rounds"][0]["kwargs"]
    assert kwargs["agent_model"] is None and kwargs["model"] is None and kwargs["target"] == "cluster"
    assert kwargs["workers"] == 8
    assert invoke("build", "--workdir", str(workdir)).exit_code == 0
    assert fake_modules["kullback.rounds.run_rounds"][1]["kwargs"]["target"] == "environment"
    refused = invoke("build", "--workdir", str(workdir), "--agent")
    assert refused.exit_code != 0 and "--agent needs --model" in refused.output


def test_a_missing_entry_point_is_one_clear_message(capsys):
    """A command whose module has no such function says so once and exits 2, never a traceback."""
    import typer

    with pytest.raises(typer.Exit) as stopped:
        cli._entry("kullback.builder.pipeline", "no_such_entry_point")
    assert stopped.value.exit_code == 2
    assert "kullback.builder.pipeline.no_such_entry_point is not available yet" in capsys.readouterr().out


def test_build_and_run_reach_a_function_that_actually_exists():
    """cli build and cli run are the only path to a build; a missing entry point makes both dead ends.

    The wiring lives in `kullback.builder.build`, not in `runner/pipeline.py`: assembling the stage
    graph means naming every Builder module, and the Runner never imports the Builder (design
    section 3, build brief rule 7, D89). `pipeline.py` stays the stage runner the graph runs on.
    """
    import inspect

    from kullback import rounds
    from kullback.builder import agent as agent_module
    from kullback.builder import build as build_module

    assert callable(getattr(build_module, "build", None)), "build.build is the whole graph, what run_builder drives"
    assert callable(getattr(agent_module, "run_builder", None)), "agent.run_builder is the Builder's beat"
    assert callable(getattr(rounds, "run_rounds", None)), "rounds.run_rounds is what cli build calls"
    assert callable(getattr(build_module, "run_batch", None)), "build.run_batch is what cli run calls"
    build_args = inspect.signature(build_module.build).parameters
    assert {"workdir", "iterate"} <= set(build_args)
    round_args = inspect.signature(rounds.run_rounds).parameters
    assert {"workdir", "iterate", "agent_model", "stall_rounds", "allowance_usd", "subscribers"} <= set(round_args)
    run_args = inspect.signature(build_module.run_batch).parameters
    assert {"workdir", "task_id", "model", "count", "seed"} <= set(run_args)


def test_build_prints_one_line_per_round_and_the_exit_before_the_json(workdir, fake_modules):
    result = invoke("build", "--workdir", str(workdir))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == ("round 1: fidelity 1/2 tasks, trusted 0, refused 0, assisted runs 3, probes passing 0, "
                        "compactions builder 0 examiner 0, spend $0.0000")
    assert lines[1] == ("round 2: fidelity 2/2 tasks, trusted 1, refused 1, assisted runs 3, probes passing 4, "
                        "compactions builder 1 examiner 0, spend $1.2500")
    assert lines[2] == "exit: stalled after 2 rounds"
    body = json.loads("\n".join(lines[3:]))
    assert body["exit"] == "stalled" and [r["round"] for r in body["rounds"]] == [1, 2]


def test_stall_rounds_and_allowance_reach_the_driver(workdir, fake_modules):
    assert invoke("build", "--workdir", str(workdir)).exit_code == 0
    kwargs = fake_modules["kullback.rounds.run_rounds"][0]["kwargs"]
    assert kwargs["stall_rounds"] == 1 and kwargs["allowance_usd"] is None
    assert invoke("build", "--workdir", str(workdir), "--stall-rounds", "3", "--allowance-usd", "0.5").exit_code == 0
    kwargs = fake_modules["kullback.rounds.run_rounds"][1]["kwargs"]
    assert kwargs["stall_rounds"] == 3 and kwargs["allowance_usd"] == 0.5


def test_agent_drives_both_agents_with_the_one_model(workdir, fake_modules):
    """--agent hands the one --model to both sessions: the driver gets it as `model` for the
    Builder's stages and as `agent_model` for the two harnesses; without --agent, agent_model is None."""
    result = invoke("build", "--workdir", str(workdir), "--model", "some/model", "--agent")
    assert result.exit_code == 0, result.output
    kwargs = fake_modules["kullback.rounds.run_rounds"][0]["kwargs"]
    assert kwargs["agent_model"] is kwargs["model"] and kwargs["model"] is not None
    assert invoke("build", "--workdir", str(workdir), "--model", "some/model").exit_code == 0
    kwargs = fake_modules["kullback.rounds.run_rounds"][1]["kwargs"]
    assert kwargs["agent_model"] is None and kwargs["model"] is not None


# --- freeze-runner ----------------------------------------------------------

def version_file(workdir: Path) -> Path:
    return Path(workdir) / "runner_version.json"


def test_freeze_runner_writes_after_a_yes(workdir):
    result = invoke("freeze-runner", "--workdir", str(workdir), input="y\n")
    assert result.exit_code == 0
    body = json.loads(version_file(workdir).read_text(encoding="utf-8"))
    assert body["runner_version"]
    assert set(body["file_hashes"]) >= {"loop.py", "route.py", "verdict.py"}
    # The gates package is hashed beside the Runner, not into it (D122).
    assert body["gates_version"] and body["gates_version"] != body["runner_version"]
    assert set(body["gates_file_hashes"]) >= {"__init__.py", "artifacts.py", "verifier_suite.py"}
    assert "gates version" in result.output


def test_freeze_runner_writes_nothing_on_a_no(workdir):
    result = invoke("freeze-runner", "--workdir", str(workdir), input="n\n")
    assert result.exit_code != 0
    assert not version_file(workdir).exists()
    assert "not frozen" in result.output


def test_yes_skips_the_question(workdir):
    result = invoke("freeze-runner", "--workdir", str(workdir), "--yes")
    assert result.exit_code == 0
    assert version_file(workdir).exists()


def test_the_runner_version_is_the_hash_of_the_runner_files(workdir):
    invoke("freeze-runner", "--workdir", str(workdir), "--yes")
    first = json.loads(version_file(workdir).read_text(encoding="utf-8"))
    second = cli.runner_version()
    assert second.runner_version == first["runner_version"]
    assert second.gates_version == first["gates_version"]


def test_a_routing_config_changes_the_version(workdir, tmp_path):
    config = tmp_path / "routing.json"
    config.write_text('{"llm_standin": false}', encoding="utf-8")
    plain = cli.runner_version()
    with_config = cli.runner_version(config)
    assert with_config.runner_version != plain.runner_version
    assert with_config.routing_config_hash


# --- run, verdict, regrade --------------------------------------------------

def seed_task(workdir: Path, task_id: str = "t1", run_id: str = "r1") -> None:
    """One Task with its Verifier and one stored Run, the shape verdict and regrade read."""
    (workdir / "tasks").mkdir(exist_ok=True)
    (workdir / "tasks" / f"{task_id}.json").write_text(
        json.dumps(as_dict(Task(id=task_id, run_ids=[run_id]))), encoding="utf-8")
    (workdir / "verifiers").mkdir(exist_ok=True)
    (workdir / "verifiers" / f"{task_id}.json").write_text(
        json.dumps(as_dict(Verifier(task_id=task_id, verifier_version="v1"))), encoding="utf-8")
    (workdir / "runs" / task_id).mkdir(parents=True, exist_ok=True)
    (workdir / "runs" / task_id / f"{run_id}.jsonl").write_text(
        json.dumps({"idx": 0, "type": "stop", "payload": {"run_id": run_id}}) + "\n", encoding="utf-8")
    (workdir / "environment.json").write_text(json.dumps(as_dict(Environment(env_id="env-1"))), encoding="utf-8")
    # regrade_gate (D97) refuses a Verdict with no runner_version, so the real regrade path (the
    # one test below that does not fake cli._entry) needs a frozen RunnerVersion on disk too.
    (workdir / "runner_version.json").write_text(
        json.dumps(as_dict(RunnerVersion(runner_version="rv-1"))), encoding="utf-8")


def seed_runs(workdir: Path, runs: list) -> None:
    """One stored Run JSONL and one Verdict per (run id, model, pass), the way loop.py and regrade write them."""
    (workdir / "runs" / "t1").mkdir(parents=True, exist_ok=True)
    (workdir / "verdicts" / "t1").mkdir(parents=True, exist_ok=True)
    for run_id, model, passed in runs:
        (workdir / "runs" / "t1" / f"{run_id}.jsonl").write_text(
            json.dumps({"idx": 0, "type": "stop", "payload": {"reason": "user_stop"}}) + "\n"
            + json.dumps({"run_id": run_id, "task_id": "t1", "env_id": "env-1", "model": model}) + "\n",
            encoding="utf-8")
        verdict = Verdict(run_id=run_id, env_id="env-1",
                          **{"pass": passed, "class": "pass" if passed else "fail"})
        (workdir / "verdicts" / "t1" / f"{run_id}.json").write_text(
            json.dumps(as_dict(verdict)), encoding="utf-8")


def test_run_asks_the_pipeline_for_a_batch(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("run", "--workdir", str(workdir), "--task", "t1", "--model", "candidate-model", "--count", "2")
    assert result.exit_code == 0
    call = fake_modules["kullback.builder.build.run_batch"][0]
    assert call["kwargs"]["task_id"] == "t1"
    assert call["kwargs"]["count"] == 2


def stored_verdicts(workdir: Path, task_id: str = "t1") -> list[dict]:
    folder = workdir / "verdicts" / task_id
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def test_verdict_scores_the_stored_runs_of_a_task(workdir):
    seed_task(workdir)
    result = invoke("verdict", "--workdir", str(workdir), "--task", "t1")
    assert result.exit_code == 0
    assert "task t1: scored 1 Runs" in result.output
    assert [body["run_id"] for body in stored_verdicts(workdir)] == ["r1"]


def test_verdict_without_a_task_scores_every_task(workdir):
    seed_task(workdir)
    seed_task(workdir, "t2", "r2")
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    assert [body["run_id"] for body in stored_verdicts(workdir, "t1")] == ["r1"]
    assert [body["run_id"] for body in stored_verdicts(workdir, "t2")] == ["r2"]


def test_verdict_says_so_when_a_task_has_no_verifier(workdir):
    (workdir / "tasks").mkdir()
    (workdir / "tasks" / "t2.json").write_text(json.dumps(as_dict(Task(id="t2"))), encoding="utf-8")
    result = invoke("verdict", "--workdir", str(workdir))
    assert "no Verifier" in result.output


def test_regrade_re_scores_without_re_executing(workdir):
    """A Verdict is written and the stored Run is not touched: nothing was executed again."""
    seed_task(workdir)
    run = workdir / "runs" / "t1" / "r1.jsonl"
    before = run.read_bytes()
    result = invoke("regrade", "--workdir", str(workdir))
    assert result.exit_code == 0
    assert [body["run_id"] for body in stored_verdicts(workdir)] == ["r1"]
    assert run.read_bytes() == before


# --- report -----------------------------------------------------------------

def test_report_writes_markdown_under_the_workdir(workdir):
    seed_task(workdir)
    result = invoke("report", "--workdir", str(workdir))
    assert result.exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "## Environment" in text
    assert str(workdir / "report.md") in result.output


def test_report_takes_an_output_path(workdir, tmp_path):
    out = tmp_path / "elsewhere" / "build.md"
    assert invoke("report", "--workdir", str(workdir), "--out", str(out)).exit_code == 0
    assert out.is_file()


def test_report_names_a_run_batch(workdir):
    result = invoke("report", "--workdir", str(workdir), "--batch")
    assert result.exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# Run batch report"
    assert "These are the numbers for one Run batch" in text


def test_a_batch_report_counts_only_the_runs_of_that_model(workdir):
    """Design section 4 item 18: one report per Run batch, so a batch cannot count every other batch."""
    seed_task(workdir)
    seed_runs(workdir, [("r1", "cand/y", True), ("r2", "frontier/x", True)])
    result = invoke("report", "--workdir", str(workdir), "--model", "cand/y")
    assert result.exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# Run batch report for cand/y"
    assert "- Runs graded: 1" in text
    assert "  - r1: pass" in text
    assert "r2" not in text.split("### Task t1", 1)[1]


# --- the ToolSigs a Verdict needs (D70, side effects) -----------------------

def seed_tool_sigs(workdir: Path) -> None:
    from kullback.runner.records import ToolSig

    (workdir / "tool_sigs.json").write_text(json.dumps([
        as_dict(ToolSig(name="cancel_order", kind="write", unclassified=False)),
        as_dict(ToolSig(name="get_order_details", kind="read", unclassified=False)),
        as_dict(ToolSig(name="mystery_tool", kind="read", unclassified=True)),
    ]), encoding="utf-8")


def test_verdict_passes_the_write_tools_and_the_flagged_tools(workdir, fake_modules):
    """Without these the extra-write, entity-count and D70 checks never fire from the CLI."""
    seed_task(workdir)
    seed_tool_sigs(workdir)
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    call = fake_modules["kullback.runner.regrade.regrade"][0]
    assert call["kwargs"]["write_tools"] == {"cancel_order"}
    assert call["kwargs"]["flagged_tools"] == {"mystery_tool"}


def test_verdict_says_when_neither_tool_sigs_nor_a_schema_are_on_disk(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("verdict", "--workdir", str(workdir))
    assert "side effects are not checked" in result.output
    assert "no EntitySchema on disk" in result.output
    kwargs = fake_modules["kullback.runner.regrade.regrade"][0]["kwargs"]
    assert kwargs["write_tools"] is None and kwargs["schema"] is None


def test_the_report_computes_task_coverage_when_no_stage_wrote_it(workdir):
    """D96's rule lives in validate.py; the report states the reason it gives, not a blank."""
    seed_task(workdir)
    assert invoke("report", "--workdir", str(workdir)).exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert "Task coverage: 0 of 1 Tasks covered" in text
    assert "t1: not covered, no Reference confirmation is recorded for this Task (D57, D93)" in text


# --- house rules ------------------------------------------------------------

def test_no_em_dashes_in_the_source():
    source = (Path(__file__).resolve().parents[1] / "kullback" / "cli.py").read_text(encoding="utf-8")
    assert "\u2014" not in source and "\u2013" not in source


def test_the_verdict_carries_the_environment_and_runner_versions(workdir):
    seed_task(workdir)
    invoke("freeze-runner", "--workdir", str(workdir), "--yes")
    frozen = json.loads((workdir / "runner_version.json").read_text(encoding="utf-8"))["runner_version"]
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    body = stored_verdicts(workdir)[0]
    assert body["env_id"] == "env-1"
    assert body["runner_version"] == frozen


# --- what the Verdict is actually given (D39, D73, D84) ---------------------

def test_verdict_passes_the_entity_schema_so_exempt_columns_are_dropped(workdir, fake_modules):
    """D73: without the schema a forbidden atom over diff() fires on a column the customer exempted."""
    from kullback.runner.records import Column, EntitySchema

    seed_task(workdir)
    schema = EntitySchema(tables=["orders"], columns=[
        Column(table="orders", name="updated_at", **{"class": "exempt"})])
    (workdir / "schema.json").write_text(json.dumps(as_dict(schema)), encoding="utf-8")
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    call = fake_modules["kullback.runner.regrade.regrade"][0]
    assert [c.name for c in call["kwargs"]["schema"].columns] == ["updated_at"]


def test_regrade_reads_the_queue_and_verdict_leaves_it_alone(workdir):
    """D84: only regrade re-scores a Run whose equivalence entry a person overturned."""
    from kullback.runner import canon

    seed_task(workdir)
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    canon._append(workdir / canon.USES_FILE, {"run_id": "r1", "key": "k1", "task_id": "t1"})
    assert canon.queue_regrade(workdir, "k1", "overturned by a reviewer") == ["r1"]

    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    assert canon.queued_regrades(workdir) == ["r1"], "verdict must not consume the regrade queue"
    assert invoke("regrade", "--workdir", str(workdir)).exit_code == 0
    assert canon.queued_regrades(workdir) == []


def test_regrade_re_scores_a_queued_run_and_empties_the_queue(workdir):
    """The whole D84 path with the real regrade: queue a Run, regrade, the stale Verdict is gone."""
    from kullback.runner import canon

    seed_task(workdir)
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    stored = sorted((workdir / "verdicts" / "t1").glob("*.json"))
    assert len(stored) == 1
    stored[0].write_text(json.dumps(dict(json.loads(stored[0].read_text(encoding="utf-8")),
                                         notes=["stale"])), encoding="utf-8")

    canon._append(workdir / canon.USES_FILE, {"run_id": "r1", "key": "k1", "task_id": "t1"})
    assert canon.queue_regrade(workdir, "k1", "overturned by a reviewer") == ["r1"]
    result = invoke("regrade", "--workdir", str(workdir))
    assert result.exit_code == 0
    assert "1 re-scored from the regrade queue" in result.output
    assert canon.queued_regrades(workdir) == []
    assert json.loads(stored[0].read_text(encoding="utf-8"))["notes"] != ["stale"]


def test_a_verdict_missing_its_runner_version_is_refused_not_counted(workdir):
    """D97: regrade_gate refuses a Verdict that never copied its Runner version (row 18).

    The regrade cache has already written the Verdict to disk by the time the gate sees it (D97's
    own cache path), so refusal is what happens next: the Task is not counted as scored and the
    command exits non-zero, rather than the file being retracted.
    """
    seed_task(workdir)
    (workdir / "runner_version.json").unlink()
    result = invoke("verdict", "--workdir", str(workdir))
    assert result.exit_code == 1
    assert "refused" in result.output
    assert "runner_version is not on the Verdict" in result.output
    assert "nothing was scored" in result.output


def test_nothing_scored_is_a_failure_not_a_silent_success(workdir):
    """An empty workdir or a Task id that matches nothing must not read as a clean run."""
    empty = invoke("verdict", "--workdir", str(workdir))
    assert empty.exit_code == 1
    assert "nothing was scored" in empty.output
    seed_task(workdir)
    typo = invoke("verdict", "--workdir", str(workdir), "--task", "t-does-not-exist")
    assert typo.exit_code == 1
    assert "no Task matched t-does-not-exist" in typo.output
    assert invoke("regrade", "--workdir", str(workdir), "--task", "t-does-not-exist").exit_code == 1


def test_the_report_echoes_the_records_it_could_not_read(workdir):
    seed_task(workdir)
    (workdir / "verdicts" / "t1").mkdir(parents=True)
    (workdir / "verdicts" / "t1" / "r1.json").write_text("{not json", encoding="utf-8")
    result = invoke("report", "--workdir", str(workdir))
    assert result.exit_code == 0
    assert "not read, so it is not counted: verdicts/t1/r1.json" in result.output


def test_a_replayed_run_covers_the_trace_it_replays(workdir):
    """D96: a Task's run_ids are Trace ids, so a replay covers the Trace it carries as trace_id."""
    seed_task(workdir)
    (workdir / "runs" / "t1" / "r1.jsonl").unlink()
    (workdir / "runs" / "t1" / "replay-1.jsonl").write_text(
        json.dumps({"idx": 0, "type": "stop", "payload": {"reason": "user_stop"}}) + "\n"
        + json.dumps({"run_id": "replay-1", "task_id": "t1", "trace_id": "r1", "model": "cand/y"}) + "\n",
        encoding="utf-8")
    (workdir / "task_status.json").write_text(
        json.dumps({"t1": {"reference_confirmed": True, "verifier_passed": True}}), encoding="utf-8")
    assert invoke("report", "--workdir", str(workdir)).exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert "Task coverage: 1 of 1 Tasks covered" in text
    assert "not covered" not in text


# --- the judges the CLI puts between a Run and its Verdict (D76, D88) -------

def _judge(name: str, verdict: str):
    """One agentic judge on a scripted model: a tool check, then its JSON answer."""
    from kullback.ai.provider import TestModel
    from kullback.runner.judge import AgenticJudge

    def read_order(order_id: str = "o1") -> dict:
        """Read one order out of the End state."""
        return {"order_id": order_id, "status": "cancelled"}

    replies = [
        {"tool_calls": [{"id": "c1", "name": "read_order", "arguments": {}}]},
        {"content": json.dumps({"verdict": verdict, "cited_spans": [f"by {name}"],
                                "sub_answers": [{"question": "q", "answer": "yes", "cited_span": "s"}]})},
    ]
    return AgenticJudge(TestModel(replies, name=name, loop=True), {"read_order": read_order}, name=name)


def _run_jsonl(folder: Path, run_id: str, wrote: bool) -> Path:
    lines = [{"run_id": run_id, "env_id": "e1", "task_id": "t1"},
             {"idx": 0, "type": "user_turn", "payload": {"content": "stop order W123"}}]
    if wrote:
        lines += [{"idx": 1, "type": "tool_call",
                   "payload": {"id": "c5", "name": "cancel_pending_order", "args": {"order_id": "W123"}}},
                  {"idx": 2, "type": "tool_result", "payload": {"id": "c5", "result": {"ok": True}}}]
    lines.append({"idx": 9, "type": "stop", "payload": {"termination_reason": "done"}})
    path = folder / f"{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _judge_verifier() -> Verifier:
    from kullback.runner.records import Atom

    return Verifier(task_id="t1", atoms=[
        Atom(id="a_write", kind="required",
             predicate_src='wrote("cancel_pending_order", order_id="W123")'),
        Atom(id="a_polite", kind="required", judge=True, description="the tone matched the policy"),
    ])


def test_the_judge_atoms_of_a_verifier_are_answered_before_the_verdict(tmp_path):
    """Without this wiring a judge atom is never answered, so every Run carrying one is not verdicted."""
    verifier = _judge_verifier()
    paths = [_run_jsonl(tmp_path, "r1", True), _run_jsonl(tmp_path, "r2", False)]
    answers = cli._judged_atoms(verifier, paths, (_judge("a", "pass"), _judge("b", "pass")), tmp_path)
    assert set(answers) == {"r1", "r2"}
    assert list(answers["r1"]) == ["a_polite"]  # the code atom is not asked of a judge
    assert answers["r1"]["a_polite"].verdict == "pass"


def test_no_judge_model_means_no_judge_results_and_no_model_call(tmp_path):
    verifier = _judge_verifier()
    paths = [_run_jsonl(tmp_path, "r1", True)]
    assert cli._judged_atoms(verifier, paths, None, tmp_path) == {}
    assert cli._name_causes([], paths, None, tmp_path, tmp_path, None) == 0


def test_a_failure_code_left_unmarked_gets_a_cause_and_one_verdict_file(tmp_path):
    """D88: code marks the Run, the judge names the cause, and the Run keeps one Verdict, not two."""
    from kullback.runner.regrade import regrade, regrade_run

    verifier = _judge_verifier()
    paths = [_run_jsonl(tmp_path, "r1", True), _run_jsonl(tmp_path, "r2", False)]
    out_dir = tmp_path / "verdicts"
    common = {"write_tools": {"cancel_pending_order"}}
    answers = cli._judged_atoms(verifier, paths, (_judge("a", "pass"), _judge("b", "pass")), tmp_path)
    verdicts = regrade(paths, verifier, None, out_dir=out_dir, judge_results=answers,
                       judge_version="1", **common)
    assert [v.cause for v in verdicts] == [None, None]
    assert "cause_pending_judge" in dict((v.run_id, v.notes) for v in verdicts)["r2"]

    named = cli._name_causes(
        verdicts, paths, (_judge("a2", "candidate"), _judge("b2", "candidate")), tmp_path, out_dir,
        lambda path, **extra: regrade_run(path, verifier, None, out_dir=out_dir,
                                          judge_version="1", **common, **extra))
    assert named == 1
    stored: dict[str, list] = {}
    for path in sorted(out_dir.glob("*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        stored.setdefault(body["run_id"], []).append(body.get("cause"))
    assert stored == {"r1": [None], "r2": ["candidate"]}


# --- build --grow table=count (D107) ---

def test_grow_targets_parse_table_equals_count_and_refuse_anything_else():
    assert cli._grow_targets(["users=500", "orders=1000"]) == {"users": 500, "orders": 1000}
    assert cli._grow_targets(None) == {}
    for bad in ("users", "users=", "=5", "users=many"):
        with pytest.raises(Exception, match="table=count"):
            cli._grow_targets([bad])


def test_build_passes_grow_through(workdir, fake_modules):
    result = runner.invoke(cli.app, ["build", "--workdir", str(workdir), "--grow", "users=500",
                                     "--grow", "orders=1000", "--grow-seed", "7"])
    assert result.exit_code == 0, result.output
    kwargs = fake_modules["kullback.rounds.run_rounds"][0]["kwargs"]
    assert kwargs["grow"] == {"users": 500, "orders": 1000} and kwargs["grow_seed"] == 7
