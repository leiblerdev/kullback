"""The round_end counts come from gate rulings and the three exits are decided by code (D126)."""

from __future__ import annotations

import pytest

from gates.examiner_fixtures import SIGS, TASK, base, history, pool, probe, replay_row, reroll_row, status, version
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, wrong_run
from kullback.gates import round_end as R
from kullback.runner.records import RoundRecord, as_dict

D126_COUNTS = ("fidelity", "trusted", "refused", "assisted_runs", "probes_passing")


def _world(tmp_path, **update):
    verifier = base(tmp_path)
    world = dict(task_status={TASK: status(), "t2": status(verifier_passed=False, reference_confirmed=False)},
                 verifiers=[verifier], probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                 history=history(version(verifier)), refusals={},
                 task_runs={TASK: [reference_run(), alt_path_run(), other_reason_run()]},
                 replays={TASK: {"tr1": replay_row("tr1", True, run_id="ref")}, "t2": {"tr2": replay_row("tr2", False)}},
                 rerolls={TASK: [reroll_row("rr2"), reroll_row("alt")], "t2": [reroll_row("reroll-t2-0", "max_steps")]},
                 canon_rules=None, sigs=SIGS)
    world.update(update)
    return world


def test_round_counts_name_every_count_d126_lists_and_each_comes_from_a_gate_ruling(tmp_path):
    recorded = []
    counts = R.round_counts(**_world(tmp_path), record=recorded.append)
    for key in D126_COUNTS + ("tasks", "tasks_with_reference", "trusted_ids", "refused_count", "false_rejection",
                              "unfinished"):
        assert key in counts, key
    assert [r.stage for r in recorded] == ["replay_reference", "trusted"]
    assert counts["trusted"] == len(recorded[1].metrics["trusted"]) == 1 and counts["trusted_ids"] == [TASK]
    assert counts["fidelity"] == 2 - len(recorded[0].failures) == 1
    assert counts["probes_passing"] == recorded[1].metrics["probes_passing"] == 0
    assert counts["refused"] == recorded[1].metrics["refused"] == {} and counts["refused_count"] == 0
    # Every legitimate Run seeded this version, so nothing is held out and the number is None, not 0.
    assert counts["false_rejection"] == {TASK: None}
    assert counts["tasks"] == 2 and counts["tasks_with_reference"] == 1 and counts["unfinished"] == []
    # The stalled exit watches exactly D126's five counts, refused as its count and assisted Runs included.
    assert R.GATE_COUNTS == ("fidelity", "trusted", "refused_count", "assisted_runs", "probes_passing")
    assert set(R.GATE_COUNTS) <= set(counts)


def test_fidelity_counts_tasks_with_a_confirmed_replay(tmp_path):
    world = _world(tmp_path)
    assert R.round_counts(**world)["fidelity"] == 1
    world["replays"]["t2"]["tr3"] = replay_row("tr3", True)
    assert R.round_counts(**world)["fidelity"] == 2
    world["replays"] = {}
    assert R.round_counts(**world)["fidelity"] == 0


def test_assisted_runs_counts_runs_whose_record_says_assisted(tmp_path):
    world = _world(tmp_path)
    assert R.round_counts(**world)["assisted_runs"] == 0
    helped = other_reason_run().model_copy(update={"assisted": True})
    world["task_runs"] = {TASK: [reference_run(), helped], "t2": [alt_path_run().model_copy(update={"assisted": True})]}
    assert R.round_counts(**world)["assisted_runs"] == 2


def test_done_is_the_state_every_task_with_a_reference_is_trusted_and_clears_fidelity_or_is_refused_and_no_probe_passes(tmp_path):
    counts = R.round_counts(**_world(tmp_path))
    assert counts["unfinished"] == [] and R.done(counts)
    # A refused Task with a Reference counts as finished.
    world = _world(tmp_path, task_status={TASK: status(), "t3": status(verifier_passed=False)},
                   refusals={"t3": {"task_id": "t3", "reason": "unreachable"}})
    counts = R.round_counts(**world)
    assert counts["refused"] == {"t3": "unreachable"} and counts["unfinished"] == [] and R.done(counts)
    # A probe that scores a pass is not done, whatever the Tasks say.
    assert not R.done(dict(counts, probes_passing=1))
    assert not R.done({"unfinished": ["t3"], "probes_passing": 0})


def test_a_task_with_a_reference_and_no_trusted_verifier_is_unfinished_and_not_done(tmp_path):
    verifier = base(tmp_path)
    world = _world(tmp_path, probes={TASK: pool(probe("probe-t1-1", reference_run(), verifier))})
    counts = R.round_counts(**world)
    assert counts["unfinished"] == [TASK] and counts["probes_passing"] == 1 and not R.done(counts)
    world = _world(tmp_path, task_status={TASK: status(verifier_passed=False)})
    counts = R.round_counts(**world)
    assert counts["unfinished"] == [TASK] and counts["trusted"] == 0 and not R.done(counts)
    # Trusted but not clearing fidelity is unfinished too.
    world = _world(tmp_path, replays={TASK: {"tr1": replay_row("tr1", False, run_id="ref")}})
    counts = R.round_counts(**world)
    assert counts["unfinished"] == [TASK] and counts["fidelity"] == 0


def test_a_workdir_with_no_task_with_a_reference_is_done_after_its_first_round(tmp_path):
    world = _world(tmp_path, task_status={TASK: status(verifier_passed=False, reference_confirmed=False),
                                          "t2": status(verifier_passed=False, reference_confirmed=False)},
                   verifiers=[], probes={}, history={})
    counts = R.round_counts(**world)
    assert counts["tasks_with_reference"] == 0 and counts["unfinished"] == [] and counts["trusted"] == 0
    assert R.done(counts)
    assert R.exit_for([counts], 1, ceiling_reached=False, exhausted=[False]) == "done"


def _round(**counts) -> dict:
    return dict({"fidelity": 1, "trusted": 1, "refused_count": 0, "assisted_runs": 0, "probes_passing": 0,
                 "unfinished": ["t9"]}, **counts)


def test_stalled_needs_stall_rounds_consecutive_rounds_with_no_gate_count_moving():
    same = _round()
    assert R.stalled([same, same], 1)
    assert not R.stalled([same, same], 2)
    assert R.stalled([same, same, same], 2)
    assert not R.stalled([_round(trusted=0), same, same], 3)
    assert not R.stalled([_round(trusted=0), same], 1)
    # The records the driver keeps are accepted as they are, and so is their dict form.
    assert R.stalled([RoundRecord(round=1, counts=same), as_dict(RoundRecord(round=2, counts=same))], 1)


@pytest.mark.parametrize("key", R.GATE_COUNTS)
def test_a_count_moving_down_is_movement_too_whichever_gate_count_it_is(key):
    assert not R.stalled([_round(**{key: 2}), _round(**{key: 1})], 1)
    assert not R.stalled([_round(**{key: 1}), _round(**{key: 2})], 1)
    assert R.stalled([_round(**{key: 2}), _round(**{key: 2})], 1)


def test_the_first_round_can_never_be_stalled():
    assert not R.stalled([_round()], 1)
    assert not R.stalled([], 1)
    assert R.exit_for([_round()], 1, ceiling_reached=False, exhausted=[False]) is None


def test_ceiling_when_the_build_ceiling_was_reached():
    assert R.exit_for([_round()], 1, ceiling_reached=True, exhausted=[]) == "ceiling"
    assert R.exit_for([_round(), _round()], 1, ceiling_reached=False, exhausted=[False, False]) == "stalled"


def test_ceiling_when_the_allowance_was_exhausted_two_rounds_in_a_row_and_not_after_one():
    assert R.exit_for([_round()], 1, ceiling_reached=False, exhausted=[True]) is None
    assert R.exit_for([_round(), _round(trusted=2)], 1, ceiling_reached=False, exhausted=[True, True]) == "ceiling"
    assert R.exit_for([_round(), _round(trusted=2)], 1, ceiling_reached=False, exhausted=[True, False]) is None
    assert R.exit_for([_round(), _round(trusted=2), _round(trusted=3)], 1, ceiling_reached=False,
                      exhausted=[True, False, True]) is None


def test_ceiling_wins_over_done_and_done_wins_over_stalled():
    finished = _round(unfinished=[])
    assert R.done(finished)
    assert R.exit_for([finished, finished], 1, ceiling_reached=True, exhausted=[]) == "ceiling"
    assert R.exit_for([_round(), finished], 1, ceiling_reached=False, exhausted=[]) == "done"
    assert R.exit_for([finished, finished], 1, ceiling_reached=False, exhausted=[]) == "done"
    assert R.exit_for([_round(), _round()], 1, ceiling_reached=False, exhausted=[]) == "stalled"
