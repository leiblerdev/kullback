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
