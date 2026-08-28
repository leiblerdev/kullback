"""The seven commands of cli.py: every one reads and writes records under a --workdir, and freeze-runner asks first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness import cli
from harness.shared.records import Environment, Task, Verifier, as_dict

runner = CliRunner()


@pytest.fixture
def fake_modules(monkeypatch):
    """Replace the lazy entry-point lookup, so these tests do not depend on the other agents' modules."""
    calls: dict[str, list] = {}

    def entry(path: str, name: str):
        def fn(*args, **kwargs):
            calls.setdefault(f"{path}.{name}", []).append({"args": args, "kwargs": kwargs})
            return {"ok": True}
        return fn

    monkeypatch.setattr(cli, "_entry", entry)
    return calls


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
    first = tmp_path / "a.json"
    first.write_text("{}", encoding="utf-8")
    result = invoke("ingest", str(first), "--workdir", str(workdir))
    assert result.exit_code == 0
    assert len(fake_modules["harness.builder.ingest.ingest_file"]) == 1


def test_build_accepts_iterate(workdir, fake_modules):
    result = invoke("build", "--workdir", str(workdir), "--iterate")
    assert result.exit_code == 0
    call = fake_modules["harness.runner.pipeline.build"][0]
    assert call["kwargs"]["iterate"] is True


def test_build_without_iterate_is_still_a_build(workdir, fake_modules):
    assert invoke("build", "--workdir", str(workdir)).exit_code == 0
    assert fake_modules["harness.runner.pipeline.build"][0]["kwargs"]["iterate"] is False


def test_a_missing_entry_point_is_one_clear_message(workdir):
    result = invoke("build", "--workdir", str(workdir))
    if result.exit_code != 0:
        assert "not available yet" in result.output


# --- freeze-runner ----------------------------------------------------------

def version_file(workdir: Path) -> Path:
    return Path(workdir) / "runner_version.json"


def test_freeze_runner_writes_after_a_yes(workdir):
    result = invoke("freeze-runner", "--workdir", str(workdir), input="y\n")
    assert result.exit_code == 0
    body = json.loads(version_file(workdir).read_text(encoding="utf-8"))
    assert body["runner_version"]
    assert set(body["file_hashes"]) >= {"loop.py", "route.py", "verdict.py"}


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


def test_a_routing_config_changes_the_version(workdir, tmp_path):
    config = tmp_path / "routing.json"
    config.write_text('{"llm_standin": false}', encoding="utf-8")
    plain = cli.runner_version()
    with_config = cli.runner_version(config)
    assert with_config.runner_version != plain.runner_version
    assert with_config.routing_config_hash


# --- run, verdict, regrade --------------------------------------------------

def seed_task(workdir: Path) -> None:
    (workdir / "tasks").mkdir(exist_ok=True)
    (workdir / "tasks" / "t1.json").write_text(json.dumps(as_dict(Task(id="t1", run_ids=["r1"]))), encoding="utf-8")
    (workdir / "verifiers").mkdir(exist_ok=True)
    (workdir / "verifiers" / "t1.json").write_text(
        json.dumps(as_dict(Verifier(task_id="t1", verifier_version="v1"))), encoding="utf-8")
    (workdir / "runs" / "t1").mkdir(parents=True, exist_ok=True)
    (workdir / "runs" / "t1" / "r1.jsonl").write_text(
        json.dumps({"idx": 0, "type": "stop", "payload": {"run_id": "r1"}}) + "\n", encoding="utf-8")
    (workdir / "environment.json").write_text(json.dumps(as_dict(Environment(env_id="env-1"))), encoding="utf-8")


def test_run_asks_the_pipeline_for_a_batch(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("run", "--workdir", str(workdir), "--task", "t1", "--model", "candidate-model", "--count", "2")
    assert result.exit_code == 0
    call = fake_modules["harness.runner.pipeline.run_batch"][0]
    assert call["kwargs"]["task_id"] == "t1"
    assert call["kwargs"]["count"] == 2


def test_verdict_scores_the_stored_runs_of_a_task(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("verdict", "--workdir", str(workdir), "--task", "t1")
    assert result.exit_code == 0
    call = fake_modules["harness.runner.regrade.regrade"][0]
    assert [Path(p).name for p in call["args"][0]] == ["r1.jsonl"]
    assert call["args"][1].task_id == "t1"


def test_verdict_without_a_task_scores_every_task(workdir, fake_modules):
    seed_task(workdir)
    assert invoke("verdict", "--workdir", str(workdir)).exit_code == 0
    assert len(fake_modules["harness.runner.regrade.regrade"]) == 1


def test_verdict_says_so_when_a_task_has_no_verifier(workdir, fake_modules):
    (workdir / "tasks").mkdir()
    (workdir / "tasks" / "t2.json").write_text(json.dumps(as_dict(Task(id="t2"))), encoding="utf-8")
    result = invoke("verdict", "--workdir", str(workdir))
    assert "no Verifier" in result.output


def test_regrade_re_scores_without_re_executing(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("regrade", "--workdir", str(workdir))
    assert result.exit_code == 0
    assert fake_modules["harness.runner.regrade.regrade"]


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
    invoke("report", "--workdir", str(workdir), "--batch")
    assert "Run batch report" in (workdir / "report.md").read_text(encoding="utf-8")


# --- the ToolSigs a Verdict needs (D70, side effects) -----------------------

def seed_tool_sigs(workdir: Path) -> None:
    from harness.shared.records import ToolSig

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
    call = fake_modules["harness.runner.regrade.regrade"][0]
    assert call["kwargs"]["write_tools"] == {"cancel_order"}
    assert call["kwargs"]["flagged_tools"] == {"mystery_tool"}


def test_verdict_says_so_when_there_are_no_tool_sigs(workdir, fake_modules):
    seed_task(workdir)
    result = invoke("verdict", "--workdir", str(workdir))
    assert "side effects are not checked" in result.output
    assert fake_modules["harness.runner.regrade.regrade"][0]["kwargs"]["write_tools"] is None


def test_the_report_computes_task_coverage_when_no_stage_wrote_it(workdir):
    seed_task(workdir)
    assert invoke("report", "--workdir", str(workdir)).exit_code == 0
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert "Task coverage: 0 of 1 Tasks covered" in text
    assert "t1: not covered" in text


# --- house rules ------------------------------------------------------------

def test_no_em_dashes_in_the_source():
    source = (Path(__file__).resolve().parents[1] / "src" / "harness" / "cli.py").read_text(encoding="utf-8")
    assert "—" not in source and "–" not in source


def test_the_verdict_carries_the_environment_and_runner_versions(workdir, fake_modules):
    seed_task(workdir)
    invoke("freeze-runner", "--workdir", str(workdir), "--yes")
    invoke("verdict", "--workdir", str(workdir))
    call = fake_modules["harness.runner.regrade.regrade"][0]
    assert call["kwargs"]["environment"].env_id == "env-1"
    assert call["kwargs"]["runner_version"]
