"""Check 6 and the trusted gate read a passing probe as a loophole only when it skipped a step every seed took first."""

from __future__ import annotations

from gates.examiner_fixtures import SIGS, TASK, history, pool, probe, replay_row, reroll_row, status, version
from gates.verifier_fixtures import (
    alt_path_run,
    derive,
    empty_run,
    make_run,
    other_reason_run,
    reference_events,
    reference_run,
)
from kullback.gates import trust as T
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules


def check(tmp_path, test_model, run, seeds=None):
    verifier = derive(tmp_path)
    seeds = [reference_run()] if seeds is None else seeds
    return S.loophole_probe(verifier, test_model, run_probe=lambda model, ver: run, canon=CanonRules(),
                            seed_runs=seeds)


# The read every seed makes before writing, named off the fixture rather than spelled out here.
READ = next(e["payload"]["name"] for e in reference_events() if e["payload"].get("id") == "c0"
            and e["type"] == "tool_call")


def skipping_the_read():
    """The seeds' write and answer, without the read every seed made before writing."""
    return make_run("probe", [e for e in reference_events() if e["payload"].get("id") != "c0"])


def test_a_probe_that_passed_and_skipped_nothing_is_a_legitimate_path_and_passes_the_check(tmp_path, test_model):
    gate = check(tmp_path, test_model, make_run("probe", reference_events()))
    assert gate.passed is True
    assert gate.metrics["probe_passed"] is True and gate.metrics["legitimate_path"] is True
    assert gate.metrics["skipped_calls"] == []


def test_a_probe_that_passed_and_skipped_a_call_every_seed_made_fails_the_check(tmp_path, test_model):
    gate = check(tmp_path, test_model, skipping_the_read())
    assert gate.passed is False
    assert gate.metrics["skipped_calls"] == [READ]
    assert f"skipped {READ}" in gate.failures[0]


def test_a_call_only_some_seeds_made_is_not_a_step_the_probe_can_skip(tmp_path, test_model):
    gate = check(tmp_path, test_model, skipping_the_read(), seeds=[reference_run(), skipping_the_read()])
    assert gate.passed is True


def test_a_probe_that_failed_passes_the_check_as_before(tmp_path, test_model):
    gate = check(tmp_path, test_model, empty_run())
    assert gate.passed is True
    assert gate.metrics["probe_passed"] is False and gate.metrics["legitimate_path"] is False


def test_a_passing_probe_with_no_seed_to_compare_still_fails_the_check(tmp_path, test_model):
    """Nothing says it skipped nothing, so the old reading stands."""
    gate = check(tmp_path, test_model, make_run("probe", reference_events()), seeds=[])
    assert gate.passed is False


# --- the trusted gate reads a passing probe the same way ----------------------

def _trusted(tmp_path, probe_run):
    verifier = derive(tmp_path)
    return T.trusted_gate(
        {TASK: status()}, [verifier], {TASK: pool(probe("probe-t1-1", probe_run, verifier))},
        history(version(verifier)), {},
        {TASK: [reference_run(), alt_path_run(), other_reason_run()]},
        {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}},
        {TASK: [reroll_row("rr2", "success"), reroll_row("alt", "success")]}, CanonRules(), SIGS)


def test_a_passing_probe_that_skipped_nothing_leaves_the_task_trusted(tmp_path):
    ruling = _trusted(tmp_path, make_run("probe", reference_events()))
    assert ruling.metrics["trusted"] == ["t1"] and ruling.metrics["probes_passing"] == 0


def test_a_passing_probe_that_skipped_a_seed_call_withholds_trust(tmp_path):
    ruling = _trusted(tmp_path, skipping_the_read())
    assert ruling.metrics["trusted"] == []
    assert ruling.metrics["untrusted"] == {"t1": "probe probe-t1-1 scores a pass"}
