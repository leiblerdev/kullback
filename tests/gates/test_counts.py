"""The workdir counts come from gate rulings (D126)."""

from __future__ import annotations

from gates.examiner_fixtures import SIGS, TASK, base, history, pool, probe, replay_row, reroll_row, status, version
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, wrong_run
from kullback.gates import counts
from kullback.runner.canon import CanonRules

D126_COUNTS = ("fidelity", "trusted", "refused", "assisted_runs", "probes_passing")


def _world(tmp_path, **update):
    verifier = base(tmp_path)
    world = dict(task_status={TASK: status(), "t2": status(verifier_passed=False, reference_confirmed=False)},
                 verifiers=[verifier], probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                 history=history(version(verifier)), refusals={},
                 task_runs={TASK: [reference_run(), alt_path_run(), other_reason_run()]},
                 replays={TASK: {"tr1": replay_row("tr1", True, run_id="ref")}, "t2": {"tr2": replay_row("tr2", False)}},
                 rerolls={TASK: [reroll_row("rr2"), reroll_row("alt")], "t2": [reroll_row("reroll-t2-0", "max_steps")]},
                 canon_rules=CanonRules(), sigs=SIGS)
    world.update(update)
    return world


def test_round_counts_name_every_count_d126_lists_and_each_comes_from_a_gate_ruling(tmp_path):
    recorded = []
    result = counts.round_counts(**_world(tmp_path), record=recorded.append)
    for key in D126_COUNTS + ("tasks", "tasks_with_reference", "trusted_ids", "refused_count", "false_rejection",
                              "unfinished"):
        assert key in result, key
    assert [r.stage for r in recorded] == ["replay_reference", "trusted"]
    assert result["trusted"] == len(recorded[1].metrics["trusted"]) == 1 and result["trusted_ids"] == [TASK]
    assert result["fidelity"] == 2 - len(recorded[0].failures) == 1
    assert result["probes_passing"] == recorded[1].metrics["probes_passing"] == 0
    assert result["refused"] == recorded[1].metrics["refused"] == {} and result["refused_count"] == 0
    # Every legitimate Run seeded this version, so nothing is held out and the number is None, not 0.
    assert result["false_rejection"] == {TASK: None}
    # t2 has no Reference yet: it is the loop's remaining work, so it is unfinished (D172).
    assert result["tasks"] == 2 and result["tasks_with_reference"] == 1 and result["unfinished"] == ["t2"]
    # The watch list is exactly D126's five counts, refused as its count and assisted Runs included.
    assert counts.GATE_COUNTS == ("fidelity", "trusted", "refused_count", "assisted_runs", "probes_passing")
    assert set(counts.GATE_COUNTS) <= set(result)


def test_fidelity_counts_tasks_with_a_confirmed_replay(tmp_path):
    world = _world(tmp_path)
    assert counts.round_counts(**world)["fidelity"] == 1
    world["replays"]["t2"]["tr3"] = replay_row("tr3", True)
    assert counts.round_counts(**world)["fidelity"] == 2
    world["replays"] = {}
    assert counts.round_counts(**world)["fidelity"] == 0


def test_assisted_runs_counts_runs_whose_record_says_assisted(tmp_path):
    world = _world(tmp_path)
    assert counts.round_counts(**world)["assisted_runs"] == 0
    helped = other_reason_run().model_copy(update={"assisted": True})
    world["task_runs"] = {TASK: [reference_run(), helped], "t2": [alt_path_run().model_copy(update={"assisted": True})]}
    assert counts.round_counts(**world)["assisted_runs"] == 2


def test_the_round_counts_say_what_the_strip_took_out_and_what_the_leak_check_still_caught(tmp_path):
    """D196: three numbers, so a reader can tell a strip that covers the corpus from one that covers
    the lines someone happened to look at. A round with no Intents at all counts zero of each."""
    world = _world(tmp_path)
    assert counts.round_counts(**world)["intents_stripped"] == 0
    world["task_status"]["t2"] = dict(world["task_status"]["t2"], leak_columns=["orders.total"])
    world["intents"] = {
        TASK: {"task_id": TASK, "text": "renew the loan", "grounded": True,
               "stripped": [{"column": "loans.due_date", "class": "hard", "shape": "month"},
                            {"column": "loans.barcode", "class": "hard", "shape": "last4"}]},
        "t2": {"task_id": "t2", "text": "pay the fine", "grounded": True, "stripped": []},
    }
    result = counts.round_counts(**world)
    assert result["intents_stripped"] == 1 and result["values_stripped"] == 2
    assert result["leak_misses"] == 1
