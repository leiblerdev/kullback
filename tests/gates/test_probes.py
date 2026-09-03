"""The probe pool is monotone (D127) and a Task closes to new probes after three rejected ones (D133)."""

from __future__ import annotations

import json

from gates.examiner_fixtures import SIGS, TASK, base, loosen, pool, probe, tighten
from gates.verifier_fixtures import extra_write_run, other_reason_run, reference_run, wrong_run
from kullback.gates import probes as P
from kullback.runner.records import ProbePool, Verifier, as_dict, content_hash


def test_a_probe_the_current_verifier_rejects_leaves_the_pool_gate_green(tmp_path):
    verifier = base(tmp_path)
    ruling = P.probe_pool_gate([verifier], {TASK: pool(probe("probe-t1-1", wrong_run(), verifier))}, None, SIGS)
    assert ruling.passed and ruling.stage == "probe_pool"
    assert ruling.metrics == {"probes": 1, "tasks_probed": 1, "passing": 0, "passing_ids": {}}


def test_a_probe_the_current_verifier_passes_fails_the_pool_gate_and_names_the_probe_its_bug_class_and_the_version_hash(tmp_path):
    verifier = base(tmp_path)
    attack = probe("probe-t1-2", reference_run(), verifier, bug_class="extra_field_acceptance")
    ruling = P.probe_pool_gate([verifier], {TASK: pool(probe("probe-t1-1", wrong_run(), verifier), attack)}, None, SIGS)
    assert not ruling.passed
    assert ruling.failures == [f"task t1: probe probe-t1-2 (extra_field_acceptance) scores a pass on version "
                               f"{P.version_hash(verifier)}"]
    assert ruling.metrics["passing"] == 1 and ruling.metrics["passing_ids"] == {"t1": ["probe-t1-2"]}


def test_every_probe_ever_written_is_scored_against_a_new_version_so_a_repair_cannot_drop_the_probe_that_found_it(tmp_path):
    """The probe gives another reason for the cancel; version 1 required the reason and rejected it, and
    the record says so. A version 2 that stops requiring the reason passes it, and the gate says so
    because it rescores the whole pool, never reads the recorded score."""
    first = tighten(base(tmp_path))
    found = probe("probe-t1-1", other_reason_run(), first, bug_class="loose_answer_extraction")
    assert found.scored_pass is False and found.verifier_hash == P.version_hash(first)
    second = base(tmp_path)
    ruling = P.probe_pool_gate([second], {TASK: pool(found)}, None, SIGS)
    assert not ruling.passed
    assert ruling.failures[0].startswith("task t1: probe probe-t1-1 (loose_answer_extraction) scores a pass on version "
                                         + P.version_hash(second))
    # The same pool against the first version is still green: the probe found a hole in the second.
    assert P.probe_pool_gate([first], {TASK: pool(found)}, None, SIGS).passed


def test_probing_a_task_is_admitted_until_three_consecutive_probes_were_already_rejected(tmp_path):
    verifier = base(tmp_path)
    rejected = [probe(f"probe-t1-{n}", wrong_run(), verifier) for n in range(1, 4)]
    open_pool = {TASK: pool(*rejected[:2])}
    assert P.probe_admission_gate(open_pool, [verifier]).passed
    assert P.consecutive_failed(open_pool[TASK], P.version_hash(verifier)) == 2
    closed = P.probe_admission_gate({TASK: pool(*rejected)}, [verifier])
    assert not closed.passed
    assert closed.failures == [f"task t1: probing stopped, the last 3 probes were already rejected by version "
                               f"{P.version_hash(verifier)}"]
    assert closed.metrics == {"closed": ["t1"], "stop": P.PROBE_STOP} and P.PROBE_STOP == 3


def test_a_probe_that_scored_a_pass_between_two_rejections_resets_the_consecutive_count(tmp_path):
    verifier = base(tmp_path)
    rows = [probe("probe-t1-1", wrong_run(), verifier), probe("probe-t1-2", wrong_run(), verifier),
            probe("probe-t1-3", reference_run(), verifier), probe("probe-t1-4", wrong_run(), verifier)]
    assert rows[2].scored_pass is True
    assert P.consecutive_failed(pool(*rows), P.version_hash(verifier)) == 1
    assert P.probe_admission_gate({TASK: pool(*rows)}, [verifier]).passed
    # Without the passing probe in between the tail is three and the Task is closed.
    assert not P.probe_admission_gate({TASK: pool(rows[0], rows[1], rows[3])}, [verifier]).passed


def test_the_consecutive_count_is_over_probes_written_against_the_current_version_only(tmp_path):
    first = base(tmp_path)
    second = loosen(first)
    against_first = [probe(f"probe-t1-{n}", wrong_run(), first) for n in range(1, 4)]
    against_second = [probe("probe-t1-4", wrong_run(), second)]
    # Three rejected against the first version closed it; an accepted repair is a new version and a new tail.
    assert not P.probe_admission_gate({TASK: pool(*against_first)}, [first]).passed
    assert P.consecutive_failed(pool(*against_first), P.version_hash(second)) == 0
    assert P.consecutive_failed(pool(*against_first, *against_second), P.version_hash(second)) == 1
    assert P.probe_admission_gate({TASK: pool(*against_first, *against_second)}, [second]).passed


def test_the_two_gates_over_no_probes_and_no_verifiers_pass_with_zero_counts():
    empty = P.probe_pool_gate([], {}, None, [])
    assert empty.passed and empty.metrics == {"probes": 0, "tasks_probed": 0, "passing": 0, "passing_ids": {}}
    admission = P.probe_admission_gate({}, [])
    assert admission.passed and admission.metrics == {"closed": [], "stop": 3}
    # A pool for a Task with no current Verifier is counted and not scored.
    orphan = P.probe_pool_gate([], {TASK: pool(probe("probe-t1-1", wrong_run(), Verifier(task_id=TASK),
                                                     scored_pass=False))}, None, SIGS)
    assert orphan.passed and orphan.metrics["probes"] == 1 and orphan.metrics["tasks_probed"] == 1


def test_a_probe_pool_round_trips_through_json_with_its_runs_and_hashes_by_content(tmp_path):
    verifier = base(tmp_path)
    original = pool(probe("probe-t1-1", wrong_run(), verifier, bug_class="loose_answer_extraction"),
                    probe("probe-t1-2", extra_write_run(), verifier, bug_class="extra_field_acceptance"))
    payload = json.loads(json.dumps(as_dict(original)))
    again = ProbePool.model_validate(payload)
    assert again == original
    assert content_hash(again) == content_hash(original)
    assert again.probes[1].run.events[-1].payload == original.probes[1].run.events[-1].payload
    assert P.probe_scores(verifier, again, None, {"cancel_pending_order"}) == {"probe-t1-1": False, "probe-t1-2": False}
