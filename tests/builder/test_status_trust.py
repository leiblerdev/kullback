"""The status tool rules trust over what the Examiner wrote (F22).

The Examiner keeps its version histories, task runs and probe pools under exam/ and the
Builder's status reads them from there, so a Task whose derived Verifier the history accepts
is trusted once the D79 suite passed, and otherwise the status names the reason that holds.
"""

from __future__ import annotations

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.gates.probes import version_hash
from kullback.runner.records import (
    Verifier,
    as_dict,
    exam_history_path,
    exam_probe_path,
    exam_verifier_path,
    load_exam_history,
    load_run_jsonl,
    read_json,
    write_json,
)
from tests.builder.session_fixtures import reply
from tests.builder.test_domain_tools import _run, _workdir

TASK = "widget_task"


def _examined(tmp_path):
    """The invented workdir after one replay and one examine: the Examiner derived and accepted a Verifier."""
    root = _workdir(tmp_path)
    model = TestModel([reply("Nothing to file.")], loop=True)
    tools = {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model)}
    replayed = _run(tools["replay"], {"task_id": TASK})
    assert not replayed.is_error, replayed.content
    examined = _run(tools["examine"], {"task_ids": [TASK]})
    assert not examined.is_error, examined.content
    return root, tools


def _state(tools):
    result = _run(tools["status"], {})
    assert not result.is_error, result.content
    return {row["task_id"]: row for row in result.details["tasks"]}[TASK]


def test_a_task_whose_history_accepts_its_derived_verifier_is_trusted_when_the_suite_passed(tmp_path):
    root, tools = _examined(tmp_path)
    verifier = Verifier.model_validate(read_json(root / "verifiers" / f"{TASK}.json"))
    history = load_exam_history(root)[TASK]
    assert exam_history_path(root).is_file()
    assert [v.content_hash for v in history.versions if v.accepted][-1] == version_hash(verifier)
    assert read_json(root / "task_status.json")[TASK]["verifier_passed"] is True
    assert _state(tools) == {"task_id": TASK, "state": "trusted", "reason": ""}


def test_a_task_whose_suite_failed_is_open_with_the_failing_check_named(tmp_path):
    root, tools = _examined(tmp_path)
    status = read_json(root / "task_status.json")
    status[TASK]["verifier_passed"] = False
    status[TASK]["checks"]["mutation_flips"] = False
    write_json(root / "task_status.json", status)
    row = _state(tools)
    assert row["state"] == "open"
    assert row["reason"] == "the D79 suite did not pass: mutation_flips failed"


def test_an_examiner_proposal_the_history_never_accepted_is_named_by_its_version(tmp_path):
    root, tools = _examined(tmp_path)
    verifier = Verifier.model_validate(read_json(root / "verifiers" / f"{TASK}.json"))
    proposal = verifier.model_copy(update={"verifier_version": "proposal"})
    write_json(exam_verifier_path(root, TASK), as_dict(proposal))
    row = _state(tools)
    assert row["state"] == "open"
    assert row["reason"] == f"version {version_hash(proposal)} is not an accepted version"


def test_a_probe_the_examiner_wrote_that_scores_a_pass_keeps_the_task_open(tmp_path):
    root, tools = _examined(tmp_path)
    reference = load_run_jsonl(root / "exam" / "runs" / TASK / "replay-rec1.jsonl")
    probe = reference.model_copy(update={"run_id": "probe-widget_task-1"})
    write_json(exam_probe_path(root, TASK, 1), as_dict(probe))
    row = _state(tools)
    assert row["state"] == "open"
    assert row["reason"] == "probe probe-widget_task-1 scores a pass"


def test_the_round_snapshot_and_the_status_agree_on_a_task_whose_seeds_are_another_tasks_runs(tmp_path):
    """D281: one trusted ruling, so the round's trusted count and the status name the same Tasks."""
    from kullback.round_snapshot import counts_of, snapshot_rows

    root, tools = _examined(tmp_path)
    assert _state(tools)["state"] == "trusted" and counts_of(snapshot_rows(root, 1))["trusted"] == 1
    verifier = Verifier.model_validate(read_json(root / "verifiers" / f"{TASK}.json"))
    seed = verifier.seed_run_ids[0]
    path = root / "runs" / TASK / f"{seed}.jsonl"
    assert load_run_jsonl(path, task_id=TASK).run_id == seed
    path.write_text(path.read_text(encoding="utf-8").replace(f'"task_id": "{TASK}"', '"task_id": "another_task"'),
                    encoding="utf-8")
    row = _state(tools)
    assert row["state"] == "open" and seed in row["reason"] and "not Runs of this Task" in row["reason"]
    [snapshot] = [r for r in snapshot_rows(root, 2) if r["task_id"] == TASK]
    assert snapshot["trusted"] is False and snapshot["trusted_reason"] == row["reason"]


def _cased_workdir(tmp_path, rules):
    """A workdir whose Verifier was derived under case-keeping rules, its seeds on disk, a held-out Run beside them."""
    from gates.examiner_fixtures import SIGS, history, replay_row, reroll_row, status, version
    from gates.verifier_fixtures import TASK as FIXTURE_TASK
    from gates.verifier_fixtures import alt_path_run, derive, other_reason_run, reference_run, write_events_jsonl
    from kullback.runner.canon import save_rules
    from kullback.runner.records import exam_task_runs_path

    verifier = derive(tmp_path, reruns=[alt_path_run()], canon=rules)
    task_id = FIXTURE_TASK.id
    write_json(tmp_path / "tasks" / f"{task_id}.json", as_dict(FIXTURE_TASK))
    write_json(tmp_path / "verifiers" / f"{task_id}.json", as_dict(verifier))
    write_json(tmp_path / "task_status.json", {task_id: status()})
    write_json(tmp_path / "replays.json", {task_id: {"tr1": replay_row("tr1", True, run_id="ref")}})
    write_json(tmp_path / "rerolls.json", {task_id: [reroll_row("alt"), reroll_row("rr2")]})
    write_json(tmp_path / "tool_sigs.json", [as_dict(sig) for sig in SIGS])
    write_json(exam_history_path(tmp_path), {k: as_dict(v) for k, v in history(version(verifier)).items()})
    write_json(exam_task_runs_path(tmp_path),
               {task_id: [as_dict(run) for run in (reference_run(), alt_path_run(), other_reason_run())]})
    for seed in (reference_run(), alt_path_run()):
        (tmp_path / "runs" / task_id).mkdir(parents=True, exist_ok=True)
        write_events_jsonl(seed, tmp_path / "runs" / task_id / f"{seed.run_id}.jsonl")
    save_rules(rules, tmp_path / "canon-rules.json")
    return tmp_path, task_id


def test_status_scores_held_out_runs_under_the_environments_rules_so_an_id_they_keep_cased_is_not_rejected(tmp_path):
    from kullback.runner.canon import CanonRules

    root, task_id = _cased_workdir(tmp_path, CanonRules(lowercase=False))
    rows = {row["task_id"]: row for row in domain_tools_mod.status_of(root)["tasks"]}
    assert rows[task_id] == {"task_id": task_id, "state": "trusted", "reason": ""}
    (root / "canon-rules.json").unlink()
    rows = {row["task_id"]: row for row in domain_tools_mod.status_of(root)["tasks"]}
    assert rows[task_id]["state"] == "open" and rows[task_id]["reason"].startswith("false_rejection")
