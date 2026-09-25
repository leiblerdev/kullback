"""One live count of a workdir, and one row per Task, for every frontend.

The domain is invented (a library that lends copies, from the gate fixtures); what is under test
is the bookkeeping: the data count matches the counts line, a second count reads nothing, rows
carry no recorded values, synthetic Tasks are marked, the command filters, and the detail names
the failing gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gates.examiner_fixtures import SIGS, TASK, base, pool, probe, replay_row, reroll_row, status
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, write_events_jsonl, wrong_run
from kullback import cli, live_counts, round_snapshot
from kullback.gates.verifier_suite import D79_STAGES
from kullback.runner.records import Task, as_dict

runner = CliRunner()

# A recorded value planted in three files below; no row may ever carry it.
MARKER = "zz-marker-never-shown-9"

# A real D79 check name, the same one the fixture fails for t2.
FAILING_CHECK = sorted(D79_STAGES.values())[0]


def _seed(workdir: Path, tmp_path: Path) -> dict:
    """A workdir with a passing Task, a failing one, a refused one and a synthetic one, all invented."""
    scratch = tmp_path / "derive"
    scratch.mkdir()
    verifier = base(scratch)
    checks = {name: True for name in D79_STAGES.values()}
    live_status = {TASK: status(checks=dict(checks)),
                   "t2": status(verifier_passed=False, reference_confirmed=False,
                                checks={**checks, FAILING_CHECK: False},
                                reason="the suite did not pass"),
                   "t3": status(verifier_passed=False, reference_confirmed=False)}
    (workdir / "task_status.json").write_text(json.dumps(live_status), encoding="utf-8")
    body = as_dict(verifier)
    body["atoms"][0]["description"] = f"derived from a recording that says {MARKER}"
    (workdir / "verifiers").mkdir()
    (workdir / "verifiers" / f"{TASK}.json").write_text(json.dumps(body), encoding="utf-8")
    replay_t1 = dict(replay_row("tr1", True, run_id="ref"),
                     counts={"calls": 14, "writes": 6, "writes_matched": 6, "reads": 8,
                             "reads_same": 7, "reads_cosmetic": 1, "reads_both_refused": 0},
                     recording={"messages": [f"a user once wrote {MARKER}"]})
    replays = {TASK: {"tr1": replay_t1}, "t2": {"tr2": replay_row("tr2", False)}}
    (workdir / "replays.json").write_text(json.dumps(replays), encoding="utf-8")
    rerolls = {TASK: [reroll_row("rr2"), reroll_row("alt")], "t2": [reroll_row("reroll-t2-0", "max_steps")]}
    (workdir / "rerolls.json").write_text(json.dumps(rerolls), encoding="utf-8")
    (workdir / "tool_sigs.json").write_text(
        json.dumps({"sigs": [as_dict(sig) for sig in SIGS]}), encoding="utf-8")
    (workdir / "tasks").mkdir()
    for task_id in (TASK, "t2", "t3"):
        (workdir / "tasks" / f"{task_id}.json").write_text(
            json.dumps(as_dict(Task(id=task_id))), encoding="utf-8")
    runs_dir = workdir / "runs" / TASK
    runs_dir.mkdir(parents=True)
    for run in (reference_run(), alt_path_run(), other_reason_run()):
        write_events_jsonl(run, runs_dir / f"{run.run_id}.jsonl")
    (workdir / "refusals").mkdir()
    (workdir / "refusals" / "t3.json").write_text(
        json.dumps({"reason": "the corpus never settles on an End state", "admitted": True}),
        encoding="utf-8")
    (workdir / "probes" / TASK).mkdir(parents=True)
    (workdir / "probes" / TASK / "pool.json").write_text(json.dumps(as_dict(pool(
        probe("p-pass", reference_run(), verifier), probe("p-fail", wrong_run(), verifier)))), encoding="utf-8")
    (workdir / "findings.json").write_text(json.dumps([{
        "finding_id": "f-1", "task_id": "t2", "kind": "suite",
        "text": f"the suite failed where the recording says {MARKER}",
        "rows": [], "path": "env/tools/lend.py", "change": "", "source": "derive",
        "key": "suite:env/tools/lend.py:t2"}]), encoding="utf-8")
    (workdir / "gates.json").write_text(json.dumps([
        {"stage": "replay_reference", "pass": True, "failures": [], "metrics": {}},
        {"stage": "compile_tools.replay_fidelity", "pass": False,
         "failures": ["one recorded call replays differently"], "metrics": {}}]), encoding="utf-8")
    synthetic = workdir / "synthetic" / "tasks"
    synthetic.mkdir(parents=True)
    (synthetic / "s1.json").write_text(json.dumps(
        {"synthetic": True, "task": as_dict(Task(id="s1"))}), encoding="utf-8")
    snapshot_status = {TASK: status(checks=dict(checks)),
                       "t2": status(checks=dict(checks))}
    snapshot_replays = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")},
                        "t2": {"tr2": replay_row("tr2", True, run_id="ref")}}
    round_snapshot.write_snapshot(
        workdir, 1, round_snapshot.task_rows(1, task_status=snapshot_status,
                                             replays=snapshot_replays, trusted=None))
    return {"checks": checks}


def test_workdir_counts_match_the_counts_line_on_the_fixture_workdir(workdir, tmp_path):
    _seed(workdir, tmp_path)
    counts = live_counts.workdir_counts(workdir, max_age=0.0)
    assert set(counts) == {"tasks", "fidelity", "trusted", "refused", "open", "drifted", "read_at"}
    assert counts["tasks"] == 3 and counts["fidelity"] == 1 and counts["refused"] == 1
    assert counts["open"] == 2 and counts["drifted"] == 1
    assert cli._counts_line(workdir) == (
        f"trusted {counts['trusted']}, refused {counts['refused']}, "
        f"fidelity {counts['fidelity']}/{counts['tasks']}")


def test_a_second_count_with_no_file_changed_reads_nothing(workdir, tmp_path):
    _seed(workdir, tmp_path)
    opens: list[str] = []

    def counting_reader(path, default=None):
        opens.append(str(path))
        return live_counts._read_json(path, default)

    first = live_counts.workdir_counts(workdir, reader=counting_reader)
    assert opens, "the first count reads the workdir files"
    opens.clear()
    second = live_counts.workdir_counts(workdir, reader=counting_reader)
    assert second == first and opens == []
    third = live_counts.workdir_counts(workdir, max_age=0.0, reader=counting_reader)
    assert third == first and opens == []


def test_task_detail_names_the_failing_gate(workdir, tmp_path):
    _seed(workdir, tmp_path)
    detail = live_counts.task_detail(workdir, "t2")
    assert detail["failing_gate"] == FAILING_CHECK
    assert detail["open_findings"] == [{"kind": "suite", "path": "env/tools/lend.py"}]
    assert detail["atom_counts"] is None and (detail["replay_matched"], detail["replay_total"]) == (None, None)
    good = live_counts.task_detail(workdir, TASK)
    atoms = json.loads((workdir / "verifiers" / f"{TASK}.json").read_text(encoding="utf-8"))["atoms"]
    assert sum(good["atom_counts"].values()) == len(atoms) and len(atoms) > 0
    assert (good["replay_matched"], good["replay_total"]) == (14, 14)
    result = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--task", "t2"])
    assert result.exit_code == 0, result.output
    assert f"failing gate: {FAILING_CHECK}" in result.output
    assert runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--task", "missing"]).exit_code == 1


def test_task_rows_mark_synthetic_tasks(workdir, tmp_path):
    _seed(workdir, tmp_path)
    by_id = {row["task_id"]: row for row in live_counts.task_rows(workdir)}
    assert by_id["s1"]["synthetic"] is True
    assert all(not by_id[task_id]["synthetic"] for task_id in (TASK, "t2", "t3"))
    assert by_id["t2"]["drift"] == "fidelity"
    assert (by_id["t2"]["last_finding_kind"], by_id["t2"]["last_finding_path"]) == ("suite", "env/tools/lend.py")
    assert by_id["t2"]["replay"] == "unconfirmed" and by_id["t2"]["suite"] == "fail"
    assert by_id["t3"]["status"] == "refused" and by_id["t3"]["replay"] is None
    assert by_id[TASK]["probes"] == "1/2"


def test_tasks_command_filters_by_status(workdir, tmp_path):
    _seed(workdir, tmp_path)
    result = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir)])
    assert result.exit_code == 0, result.output
    assert "id status replay suite probes drift last finding" in result.output
    refused = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--status", "refused"])
    assert refused.exit_code == 0, refused.output
    assert "t3" in refused.output and TASK not in refused.output.split()
    drifted = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--status", "drifted"])
    assert drifted.exit_code == 0, drifted.output
    assert "t2" in drifted.output and "t3" not in drifted.output.split()
    opened = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--status", "open"])
    assert opened.exit_code == 0, opened.output
    assert "t3" not in opened.output.split()
    assert runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--status", "bogus"]).exit_code == 2
    as_json = runner.invoke(cli.app, ["tasks", "--workdir", str(workdir), "--json"])
    assert as_json.exit_code == 0, as_json.output
    assert {row["task_id"] for row in json.loads(as_json.output)} == {TASK, "t2", "t3", "s1"}

