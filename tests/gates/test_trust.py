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


def test_the_trusted_ruling_carries_the_false_rejection_number_per_task(tmp_path):
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    world = _world(tmp_path, strict)
    ruling = T.trusted_gate(**world)
    # rr2 gives another reason and is rejected; alt is held out and passes: one in two.
    assert ruling.metrics["false_rejection"] == {TASK: 0.5}
    assert ruling.metrics["trusted"] == [TASK], "over-strict is reported next to trusted, not hidden by it"
    world["rerolls"] = {}
    world["replays"] = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    assert T.trusted_gate(**world).metrics["false_rejection"] == {TASK: None}
