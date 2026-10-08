"""Every refusal is admitted only when no frontier Run finished; a re-picked Task stays untrusted until derived again."""

from __future__ import annotations

import pytest

from gates.examiner_fixtures import SIGS, TASK, base, history, pool, probe, replay_row, reroll_row, status, version
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, wrong_run
from kullback.examiner import reference_check as RC
from kullback.gates import trust as T
from kullback.runner.canon import CanonRules
from kullback.runner.records import read_json, write_json

FINISHED = ({TASK: {"tr1": replay_row("tr1", True)}}, {TASK: [reroll_row("reroll-t1-0", "success")]})


@pytest.mark.parametrize("reason", ["no frontier Run reaches the End state", "outcome_not_in_state",
                                    "reference_wrong", ""])
def test_a_refusal_with_a_finished_run_is_rejected_whatever_its_reason(reason):
    ruling = T.refuse_gate({TASK: {"task_id": TASK, "reason": reason}}, *FINISHED)
    assert not ruling.passed and ruling.metrics == {"refused": [], "rejected": [TASK]}


def _trusted_world(tmp_path, task_status: dict) -> dict:
    verifier = base(tmp_path)
    return dict(task_status=task_status, verifiers=[verifier],
                probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                history=history(version(verifier)), refusals={},
                task_runs={TASK: [reference_run(), alt_path_run(), other_reason_run()]},
                replays={TASK: {"tr1": replay_row("tr1", True, run_id="ref")}},
                rerolls={TASK: [reroll_row("rr2", "success"), reroll_row("alt", "success")]},
                canon_rules=CanonRules(), sigs=SIGS)


def test_a_trusted_task_is_not_trusted_after_its_reference_is_rejected(tmp_path):
    workdir = tmp_path / "work"
    write_json(workdir / "task_status.json", {TASK: status()})
    write_json(workdir / "references.json", {TASK: {
        "references": [{"run_id": "ref"}], "failed": {"alt": "survivor B was not chosen: A stood up better"},
        "groups": [{"label": "A", "runs": ["ref"]}, {"label": "B", "runs": ["alt"]}]}})
    assert T.trusted_gate(**_trusted_world(tmp_path, read_json(workdir / "task_status.json"))).metrics[
        "trusted"] == [TASK]
    done = RC.reject_reference(workdir, TASK, "ref", "the End state is not what the Intent asks")
    assert done["references"] == ["alt"] and not done["pooled"]
    ruling = T.trusted_gate(**_trusted_world(tmp_path, read_json(workdir / "task_status.json")))
    assert ruling.metrics["trusted"] == []
    assert ruling.metrics["untrusted"] == {TASK: RC.REPICKED}
