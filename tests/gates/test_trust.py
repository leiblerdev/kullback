"""The refuse ruling reads re-rolls already paid for (D128), and a trusted Verifier is the composition of the
suite, the pool, the history and the refusals (D126)."""

from __future__ import annotations

from gates.examiner_fixtures import (
    SIGS,
    TASK,
    base,
    history,
    loosen,
    pool,
    probe,
    replay_row,
    reroll_row,
    status,
    tighten,
    version,
)
from gates.verifier_fixtures import alt_path_run, other_reason_run, reference_run, wrong_run
from kullback.gates import trust as T
from kullback.gates.probes import version_hash

REFUSAL = {TASK: {"task_id": TASK, "reason": "no frontier Run reaches the End state", "round": 1}}
NOTHING_FINISHED = ({TASK: {"tr1": replay_row("tr1", False)}}, {TASK: [reroll_row("reroll-t1-0", "max_steps")]})


def test_a_refusal_is_admitted_only_when_no_frontier_run_of_the_task_finished():
    replays, rerolls = NOTHING_FINISHED
    ruling = T.refuse_gate(REFUSAL, replays, rerolls)
    assert ruling.passed and ruling.stage == "refuse"
    assert ruling.metrics == {"refused": [TASK], "rejected": []}
    assert T.finished_runs(TASK, replays, rerolls) == []
    assert T.refuse_gate({}, {}, {}).passed


def test_a_refusal_of_a_task_with_a_confirmed_replay_is_rejected_and_names_the_run():
    replays = {TASK: {"tr1": replay_row("tr1", True)}}
    ruling = T.refuse_gate(REFUSAL, replays, {})
    assert not ruling.passed
    assert ruling.failures == ["task t1: the frontier finished it: replay-tr1"]
    assert ruling.metrics == {"refused": [], "rejected": [TASK]}


def test_a_refusal_of_a_task_whose_reroll_finished_is_rejected_and_names_the_reroll():
    rerolls = {TASK: [reroll_row("reroll-t1-0", "max_steps"), reroll_row("reroll-t1-2", "user_stop")]}
    ruling = T.refuse_gate(REFUSAL, {TASK: {"tr1": replay_row("tr1", False)}}, rerolls)
    assert not ruling.passed
    assert ruling.failures == ["task t1: the frontier finished it: reroll-t1-2"]
    assert T.finished_runs(TASK, {TASK: {"tr1": replay_row("tr1", True)}}, rerolls) == ["replay-tr1", "reroll-t1-2"]


def _world(tmp_path, verifier=None):
    verifier = verifier if verifier is not None else base(tmp_path)
    replays = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    rerolls = {TASK: [reroll_row("rr2", "success"), reroll_row("alt", "success")]}
    runs = {TASK: [reference_run(), alt_path_run(), other_reason_run()]}
    return dict(task_status={TASK: status()}, verifiers=[verifier],
                probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                history=history(version(verifier)), refusals={}, task_runs=runs, replays=replays, rerolls=rerolls,
                canon_rules=None, sigs=SIGS)


def test_a_trusted_verifier_passed_the_suite_rejects_every_probe_is_an_accepted_version_and_its_task_is_not_refused(tmp_path):
    ruling = T.trusted_gate(**_world(tmp_path))
    assert ruling.passed and ruling.stage == "trusted"
    assert ruling.metrics["trusted"] == [TASK] and ruling.metrics["untrusted"] == {}
    assert ruling.metrics["probes_passing"] == 0 and ruling.metrics["refused"] == {}
    assert T.trusted_gate({}, [], {}, {}, {}, {}, {}, {}, None, []).passed


def test_a_verifier_with_a_passing_probe_is_not_trusted_and_the_reason_names_the_probe(tmp_path):
    world = _world(tmp_path)
    verifier = world["verifiers"][0]
    world["probes"] = {TASK: pool(probe("probe-t1-1", wrong_run(), verifier), probe("probe-t1-2", reference_run(), verifier))}
    ruling = T.trusted_gate(**world)
    assert not ruling.passed
    assert ruling.failures == ["task t1: probe probe-t1-2 scores a pass"]
    assert ruling.metrics["untrusted"] == {TASK: "probe probe-t1-2 scores a pass"} and ruling.metrics["probes_passing"] == 1


def test_a_verifier_that_failed_the_suite_is_not_trusted(tmp_path):
    world = _world(tmp_path)
    world["task_status"] = {TASK: status(verifier_passed=False)}
    ruling = T.trusted_gate(**world)
    assert ruling.failures == ["task t1: the D79 suite did not pass"] and ruling.metrics["trusted"] == []
    assert ruling.metrics["checks_not_run"] == {}


def test_a_task_with_one_reference_says_the_second_path_check_was_not_run_rather_than_failed(tmp_path):
    """The rule is unchanged: a Task the suite refused is untrusted whatever the reason. What the
    ruling says changes, because a check with no input asks for more Runs of the Task and a check
    that failed asks for a looser Verifier, and until D173 both read the same."""
    world = _world(tmp_path)
    world["task_status"] = {TASK: status(verifier_passed=False, not_run=["verifier_alt_path"],
                                         checks={"second_path_passes": False, "oracle_passes": True})}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == []
    assert ruling.metrics["checks_not_run"] == {TASK: ["second_path_passes"]}
    assert ruling.metrics["untrusted"] == {
        TASK: "the D79 suite did not pass: second_path_passes not run (one Reference, so there is no "
              "second path to score)"}
    # A check that really failed is still reported as a failure, beside the one nobody could run.
    world["task_status"] = {TASK: status(verifier_passed=False, not_run=["verifier_alt_path"],
                                         checks={"second_path_passes": False, "mutation_flips": False})}
    assert T.trusted_gate(**world).metrics["untrusted"][TASK].endswith("mutation_flips failed")


def test_a_suite_failure_names_every_failing_check_and_every_check_with_no_input(tmp_path):
    """A reader groups the Tasks that stop at the suite by what stopped them, so the reason names
    each check in the suite's own order and says why the ones with no input had none."""
    world = _world(tmp_path)
    checks = {name: True for name in T.D79_STAGES.values()}
    checks["mutation_flips"] = checks["plausible_wrong_fails"] = checks["loophole_probe_fails"] = False
    world["task_status"] = {TASK: status(verifier_passed=False, checks=checks,
                                         not_run=["verifier_loophole"],
                                         not_run_reasons={"loophole_probe_fails": "no model"})}
    assert T.trusted_gate(**world).metrics["untrusted"] == {
        TASK: "the D79 suite did not pass: plausible_wrong_fails failed, "
              "loophole_probe_fails not run (no model), mutation_flips failed"}


def test_a_verifier_that_is_not_the_last_accepted_version_is_not_trusted(tmp_path):
    """The file on disk has to be the history's last accepted row: a version the loosening gate rejected,
    or one never entered in the history, is not a version anyone accepted."""
    world = _world(tmp_path)
    plain = world["verifiers"][0]
    rejected = loosen(plain)
    world["verifiers"] = [rejected]
    world["history"] = history(version(plain), version(rejected, 2, by="repair", accepted=False, rejected_by=["probe_pool"]))
    ruling = T.trusted_gate(**world)
    assert ruling.failures == [f"task t1: version {version_hash(rejected)} is not an accepted version"]
    world["history"] = {}
    assert not T.trusted_gate(**world).passed


def test_a_version_the_loosening_gate_rejects_is_not_trusted_whatever_its_accepted_flag_says(tmp_path):
    """The accepted flag is a row in a file; trust runs the loosening gate again over the history cut
    at the current version. The plain version newly passes rr2 against the strict one before it;
    with rr2 unfinished that is loosening past the frontier, with rr2 finished it is the pool growing."""
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    world = _world(tmp_path, plain)
    world["history"] = history(version(strict), version(plain, 2, by="repair", parent=strict))
    world["rerolls"] = {TASK: [reroll_row("rr2", "max_steps"), reroll_row("alt", "success")]}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == []
    assert ruling.failures == [f"task t1: version {version_hash(plain)} loosens past the frontier: task t1: version "
                               f"{version_hash(plain)} newly passes rr2, which is not the Reference, a frontier "
                               "re-roll or a production Run"]
    world["rerolls"] = {TASK: [reroll_row("rr2", "success"), reroll_row("alt", "success")]}
    assert T.trusted_gate(**world).metrics["trusted"] == [TASK]
    # A rejected attempt after the current version is not what the file holds and does not count against it.
    world["history"] = history(version(strict), version(plain, 2, by="repair", parent=strict),
                               version(loosen(plain), 3, by="repair", accepted=False, rejected_by=["probe_pool"]))
    assert T.trusted_gate(**world).metrics["trusted"] == [TASK]


def test_a_verifier_of_a_refused_task_is_not_counted_as_trusted(tmp_path):
    world = _world(tmp_path)
    world["replays"], world["rerolls"] = NOTHING_FINISHED
    world["refusals"] = REFUSAL
    ruling = T.trusted_gate(**world)
    assert ruling.failures == ["task t1: the Task is refused"]
    assert ruling.metrics["trusted"] == [] and ruling.metrics["refused"] == {TASK: REFUSAL[TASK]["reason"]}
    # A refusal the gate rejected (the frontier finished the Task) does not make the Task refused.
    world["replays"], world["rerolls"] = _world(tmp_path)["replays"], _world(tmp_path)["rerolls"]
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == [TASK] and ruling.metrics["refused"] == {}


def test_a_false_rejection_under_the_threshold_is_trusted_and_the_ruling_carries_the_fraction_and_the_pool_size(tmp_path):
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    ruling = T.trusted_gate(**_world(tmp_path, strict))
    # rr2 gives another reason and is rejected; alt is held out and passes: one in two.
    assert ruling.metrics["false_rejection"] == {TASK: 0.5}
    assert ruling.metrics["false_rejection_pool"] == {TASK: 2}
    assert ruling.metrics["false_rejection_ruling"] == {TASK: "0.50 of 2 held-out Runs"}
    assert ruling.passed and ruling.metrics["trusted"] == [TASK], \
        "a Verifier that recognises some path other than its seeds is a check of the Task"


def test_a_verifier_that_rejects_every_held_out_run_is_not_trusted_and_the_reason_names_the_false_rejection(tmp_path):
    """D194: the number was measured from the start and never read, so a Verifier the false-rejection
    gate calls over-strict was trusted anyway. Rejecting every Run that reached the Reference makes it
    a check of one path, not of the Task."""
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    world = _world(tmp_path, strict)
    # Only the Run giving another reason is left in the pool, and the strict version rejects it.
    world["rerolls"] = {TASK: [reroll_row("rr2", "success"), reroll_row("alt", "max_steps")]}
    ruling = T.trusted_gate(**world)
    assert not ruling.passed and ruling.metrics["trusted"] == []
    assert ruling.metrics["false_rejection"] == {TASK: 1.0} and ruling.metrics["false_rejection_pool"] == {TASK: 1}
    assert ruling.failures == [
        "task t1: false_rejection 1.00 of 1 held-out Runs: the required atoms reject every held-out Run that "
        "reached the Reference, so the Verifier checks one path and not the Task"]
    assert ruling.metrics["untrusted"][TASK].startswith("false_rejection 1.00 of 1 held-out Runs")


def test_a_task_with_nothing_held_out_says_no_pool_and_is_trusted_on_the_other_gates(tmp_path):
    """An empty sample is not a rate and not a failure: no held-out Run reached the Reference, so the
    step has nothing to rule on and the Task stands or falls on the checks that do."""
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    world = _world(tmp_path, strict)
    world["rerolls"] = {}
    world["replays"] = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["false_rejection"] == {TASK: None} and ruling.metrics["false_rejection_pool"] == {TASK: 0}
    assert ruling.metrics["false_rejection_ruling"] == {TASK: "no_pool"}
    assert ruling.passed and ruling.metrics["trusted"] == [TASK]
    # The other gates still rule: the same empty pool does not carry a Task the suite turned away.
    world["task_status"] = {TASK: status(verifier_passed=False)}
    denied = T.trusted_gate(**world)
    assert denied.metrics["trusted"] == [] and denied.metrics["false_rejection_ruling"] == {TASK: "no_pool"}
