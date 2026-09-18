"""G31: the round stop rule reads rates over the Task list and always says why it stopped.

Skipped until the founder re-freezes kullback/gates with docs/frozen-patches/stop-rule.patch.
"""

import random

import pytest

from gates.examiner_fixtures import SIGS, TASK, base, history, pool, probe, replay_row, reroll_row, status, version
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, wrong_run
from kullback.gates import round_end as R

pytestmark = pytest.mark.skipif(not hasattr(R, "exit_reason"),
                                reason="stop-rule patch waits for the founder's re-freeze")


def _goal_round(fidelity_tasks, trusted_tasks, tasks, refused=0, unfinished=(), probes=0):
    return {"fidelity": fidelity_tasks, "trusted": trusted_tasks, "refused_count": refused,
            "assisted_runs": 0, "probes_passing": probes, "unfinished": list(unfinished), "tasks": tasks,
            "fidelity_rate": fidelity_tasks / tasks if tasks else 1.0,
            "trusted_share": trusted_tasks / tasks if tasks else 1.0}


def test_a_round_where_every_task_was_refused_exits_refused_not_done():
    """Claim 2 fixed: refusal clears the unfinished list, but the goal is unmet, so the stop is
    named for what happened instead of reading as done."""
    last = _goal_round(0, 0, 3, refused=3, unfinished=[])
    assert R.done(last), "the literal D126 state still reads refused Tasks as finished"
    assert R.exit_for([last], 1, ceiling_reached=False, exhausted=[False]) == "refused"
    reason = R.exit_reason([last], 1, ceiling_reached=False, exhausted=[False])
    assert "refused" in reason and "3" in reason


def test_a_finish_with_half_the_list_refused_is_not_done_either():
    last = _goal_round(2, 2, 4, refused=2, unfinished=[])
    assert R.done(last)
    assert R.exit_for([last], 1, ceiling_reached=False, exhausted=[False]) == "refused"


def test_the_done_exit_needs_the_goal_not_just_nothing_left():
    full = _goal_round(3, 3, 3, unfinished=[])
    assert R.goal_met(full)
    assert R.exit_for([full], 1, ceiling_reached=False, exhausted=[False]) == "done"
    short = _goal_round(1, 1, 4, unfinished=[])
    assert not R.goal_met(short)
    assert R.exit_for([short], 1, ceiling_reached=False, exhausted=[False]) is None


def test_a_growing_task_list_with_a_falling_rate_is_flat():
    """Claim 3 fixed: fidelity 1 of 2, then 2 of 5, then 2 of 8 reads as a rise in absolute
    counts while the rate falls from one half to one quarter."""
    growing = [_goal_round(1, 0, 2, unfinished=["n1"]), _goal_round(2, 0, 5, unfinished=["n1"]),
               _goal_round(2, 0, 8, unfinished=["n1"])]
    assert not R.fidelity_flat(growing, 2), "absolute counts still read the growth as progress"
    assert R.fidelity_rate_flat(growing, 2)
    assert R.exit_for(growing[-1:], 9, ceiling_reached=False, exhausted=[], all_rounds=growing,
                      fidelity_stall=2) == "stalled"
    reason = R.exit_reason(growing[-1:], 9, ceiling_reached=False, exhausted=[], all_rounds=growing,
                           fidelity_stall=2)
    assert "stalled" in reason and "2" in reason


def test_legacy_counts_without_a_task_list_decide_as_before():
    bare = {"fidelity": 1, "trusted": 1, "refused_count": 0, "assisted_runs": 0, "probes_passing": 0,
            "unfinished": []}
    assert R.goal_met(bare)
    assert R.exit_for([bare], 1, ceiling_reached=False, exhausted=[False]) == "done"
    assert R.fidelity_rate(bare) is None and R.trusted_share(bare) is None


def test_round_counts_reports_rates_over_this_rounds_task_list(tmp_path):
    verifier = base(tmp_path)
    world = dict(task_status={TASK: status(), "t2": status(verifier_passed=False, reference_confirmed=False)},
                 verifiers=[verifier], probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                 history=history(version(verifier)), refusals={},
                 task_runs={TASK: [reference_run(), alt_path_run(), other_reason_run()]},
                 replays={TASK: {"tr1": replay_row("tr1", True, run_id="ref")},
                          "t2": {"tr2": replay_row("tr2", False)}},
                 rerolls={TASK: [reroll_row("rr2"), reroll_row("alt")],
                          "t2": [reroll_row("reroll-t2-0", "max_steps")]},
                 canon_rules=None, sigs=SIGS)
    counts = R.round_counts(**world)
    assert counts["fidelity_rate"] == pytest.approx(0.5) and counts["trusted_share"] == pytest.approx(0.5)
    assert counts["fidelity_tasks"] == 1 and counts["trusted_tasks"] == 1 and counts["tasks"] == 2


def test_every_stop_and_every_non_stop_carries_a_reason_with_numbers():
    rng = random.Random(7)
    for _ in range(300):
        seq = []
        for _ in range(rng.randint(1, 4)):
            tasks = rng.randint(0, 6)
            fid = rng.randint(0, tasks)
            tru = rng.randint(0, tasks)
            refused = rng.randint(0, tasks - tru) if tasks else 0
            seq.append({"fidelity": fid, "trusted": tru, "refused_count": refused,
                        "assisted_runs": rng.randint(0, 3), "probes_passing": rng.randint(0, 1),
                        "unfinished": ["n%s" % i for i in range(rng.randint(0, tasks))],
                        "tasks": tasks, "fidelity_rate": fid / tasks if tasks else 1.0,
                        "trusted_share": tru / tasks if tasks else 1.0})
        stall = rng.randint(1, 3)
        ceiling = rng.random() < 0.2
        exhausted = [rng.random() < 0.3 for _ in seq]
        flat = rng.choice([None, 1, 2])
        cap = rng.choice([None, 1, 5])
        exit = R.exit_for(seq, stall, ceiling_reached=ceiling, exhausted=exhausted, all_rounds=seq,
                          fidelity_stall=flat, max_rounds=cap)
        reason = R.exit_reason(seq, stall, ceiling_reached=ceiling, exhausted=exhausted, all_rounds=seq,
                               fidelity_stall=flat, max_rounds=cap)
        assert reason and any(char.isdigit() for char in reason), seq
        if exit is None:
            assert "no exit" in reason, seq
        else:
            assert exit in reason, (exit, seq)
