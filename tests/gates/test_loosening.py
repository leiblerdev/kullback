"""Loosening is one-directional over a pool the Runner's records build (D127, D133), and the false-rejection
number is measured per Task over the held-out legitimate Runs."""

from __future__ import annotations

from gates.examiner_fixtures import SIGS, TASK, base, history, loosen, replay_row, reroll_row, tighten, version
from gates.verifier_fixtures import WRITE_TOOLS, alt_path_run, extra_write_run, other_reason_run, reference_run
from kullback.gates import loosening as L
from kullback.gates.probes import version_hash
from kullback.gates.verifier_suite import check_run
from kullback.runner.records import Atom

REPLAYS = {TASK: {"tr1": replay_row("tr1", True, run_id="ref"), "tr2": replay_row("tr2", False, run_id="replay-tr2")}}
REROLLS = {TASK: [reroll_row("rr2", "success"), reroll_row("reroll-t1-1", "max_steps")]}


def test_the_legitimate_pool_is_the_confirmed_replays_and_the_finished_rerolls_and_nothing_else():
    assert L.legitimate_runs(REPLAYS, REROLLS) == {TASK: {"ref", "rr2"}}
    assert L.legitimate_runs({}, {}) == {}
    # A Task with rows and nothing finished is present with an empty pool, so a count can name it.
    assert L.legitimate_runs({"t9": {"x": replay_row("x", False)}}, {}) == {"t9": set()}


def test_the_pool_grows_with_rerolls_of_a_later_round_merged_from_the_examiners_rows():
    merged = {TASK: REROLLS[TASK] + [reroll_row("reroll-r2-t1-0", "user_stop"), reroll_row("reroll-r2-t1-1", "error")]}
    assert L.legitimate_runs(REPLAYS, merged)[TASK] == {"ref", "rr2", "reroll-r2-t1-0"}


def test_a_reroll_that_died_unfinished_is_not_in_the_pool():
    assert "reroll-t1-1" not in L.legitimate_runs(REPLAYS, REROLLS)[TASK]
    assert L.legitimate_runs({}, {TASK: [reroll_row("r", "max_steps"), reroll_row("s", "error")]}) == {TASK: set()}


def test_a_new_version_may_newly_pass_the_reference_or_a_frontier_reroll(tmp_path):
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    runs = {TASK: [reference_run(), alt_path_run(), other_reason_run()]}
    assert L.newly_passed(strict, plain, runs[TASK], None, WRITE_TOOLS) == ["rr2"]
    ruling = L.loosening_gate(history(version(strict), version(plain, 2, by="repair", parent=strict)),
                              runs, REPLAYS, REROLLS, None, SIGS)
    assert ruling.passed, ruling.failures
    assert ruling.metrics == {"tasks": 1, "compared": 1, "loosened": {}, "legitimate": {TASK: 2}}


def test_a_new_version_that_newly_passes_any_other_run_is_rejected_and_the_run_is_named(tmp_path):
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    runs = {TASK: [reference_run(), alt_path_run(), other_reason_run()]}
    unfinished = {TASK: [reroll_row("rr2", "max_steps")]}
    ruling = L.loosening_gate(history(version(strict), version(plain, 2, by="repair", accepted=False)),
                              runs, REPLAYS, unfinished, None, SIGS)
    assert not ruling.passed
    assert ruling.failures == [f"task t1: version {version_hash(plain)} newly passes rr2, which is not the Reference, "
                               "a frontier re-roll or a production Run"]
    assert ruling.metrics["loosened"] == {TASK: ["rr2"]} and ruling.metrics["legitimate"] == {TASK: 1}


def test_a_version_that_only_tightens_passes_the_gate(tmp_path):
    plain, strict = base(tmp_path), tighten(base(tmp_path))
    runs = {TASK: [reference_run(), alt_path_run(), other_reason_run(), extra_write_run()]}
    ruling = L.loosening_gate(history(version(plain), version(strict, 2, by="repair")), runs, {}, {}, None, SIGS)
    assert ruling.passed and ruling.metrics["compared"] == 1 and ruling.metrics["loosened"] == {}


def test_the_first_version_has_nothing_to_loosen_from_and_passes(tmp_path):
    plain = base(tmp_path)
    runs = {TASK: [reference_run(), extra_write_run()]}
    ruling = L.loosening_gate(history(version(plain)), runs, {}, {}, None, SIGS)
    assert ruling.passed and ruling.metrics["compared"] == 0
    # A rejected first attempt followed by a second is still a first accepted version to come.
    ruling = L.loosening_gate(history(version(plain, accepted=False, rejected_by=["verifier_oracle"]),
                                      version(loosen(plain), 2)), runs, {}, {}, None, SIGS)
    assert ruling.passed and ruling.metrics["compared"] == 0
    assert L.loosening_gate({}, {}, {}, {}, None, []).passed


def test_the_previous_version_is_the_last_accepted_one_not_a_rejected_attempt(tmp_path):
    """Version 1 is strict; a rejected attempt loosened it; the newest attempt loosens the same way.
    Measured against the rejected attempt nothing is newly passed; measured against the last accepted
    version, rr2 is, and rr2 is not in the pool, so the ruling fails."""
    strict, plain = tighten(base(tmp_path)), base(tmp_path)
    runs = {TASK: [reference_run(), other_reason_run()]}
    rows = history(version(strict), version(plain, 2, by="repair", accepted=False, rejected_by=["loosening"]),
                   version(plain, 3, by="repair", accepted=False))
    ruling = L.loosening_gate(rows, runs, {}, {}, None, SIGS)
    assert not ruling.passed and ruling.metrics["loosened"] == {TASK: ["rr2"]}
    assert L.accepted_versions(rows[TASK]) == [rows[TASK].versions[0]]


def test_false_rejection_counts_the_held_out_legitimate_runs_the_required_atoms_fail(tmp_path):
    strict = tighten(base(tmp_path))
    assert strict.seed_run_ids == ["ref", "alt", "rr2"]
    strict = strict.model_copy(update={"seed_run_ids": ["ref"]})
    runs = [reference_run(), alt_path_run(), other_reason_run()]
    row = L.false_rejection(strict, runs, {"ref", "alt", "rr2"}, None, WRITE_TOOLS)
    assert row == {"held_out": 2, "rejected": 1, "fraction": 0.5, "rejected_ids": ["rr2"]}


def test_a_seed_run_is_never_held_out(tmp_path):
    strict = tighten(base(tmp_path))
    runs = [reference_run(), alt_path_run(), other_reason_run()]
    # Every legitimate Run is a seed of this version, so nothing is held out, whatever it would score.
    row = L.false_rejection(strict, runs, {"ref", "alt", "rr2"}, None, WRITE_TOOLS)
    assert row["held_out"] == 0 and row["rejected_ids"] == []
    # A Run that is legitimate and not a seed is held out; one that is not legitimate is not.
    row = L.false_rejection(strict.model_copy(update={"seed_run_ids": ["ref", "alt"]}), runs, {"ref", "rr2"}, None,
                            WRITE_TOOLS)
    assert row["held_out"] == 1 and row["rejected_ids"] == ["rr2"]


def test_a_held_out_frontier_run_rejected_only_by_a_hard_constraint_is_not_a_false_rejection(tmp_path):
    """A Hard constraint is the policy over the Run: a frontier Run that broke it is rightly rejected,
    so the number counts what the required atoms fail and leaves the Hard atoms out."""
    plain = base(tmp_path).model_copy(update={"seed_run_ids": ["ref"]})
    policy = Atom(id="hard.never", kind="hard", predicate_src="def check():\n    return False\n")
    strict_policy = plain.model_copy(update={"atoms": plain.atoms + [policy]})
    runs = [reference_run(), alt_path_run()]
    assert check_run(strict_policy, alt_path_run(), None, write_tools=WRITE_TOOLS) == (False, "hard.never")
    row = L.false_rejection(strict_policy, runs, {"ref", "alt"}, None, WRITE_TOOLS)
    assert row == {"held_out": 1, "rejected": 0, "fraction": 0.0, "rejected_ids": []}
    # The required atoms still count: the tightened reason rejects rr2 with or without the policy atom.
    tight = tighten(strict_policy)
    row = L.false_rejection(tight, runs + [other_reason_run()], {"ref", "alt", "rr2"}, None, WRITE_TOOLS)
    assert row["rejected_ids"] == ["rr2"] and row["fraction"] == 0.5


def test_false_rejection_with_no_held_out_runs_is_none_not_zero(tmp_path):
    plain = base(tmp_path)
    assert L.false_rejection(plain, [reference_run()], set(), None, WRITE_TOOLS)["fraction"] is None
    assert L.false_rejection(plain, [], {"ref"}, None, WRITE_TOOLS) == {"held_out": 0, "rejected": 0,
                                                                          "fraction": None, "rejected_ids": []}


def test_a_verifier_that_rejects_every_held_out_frontier_run_fails_the_false_rejection_gate_and_one_that_passes_some_does_not(tmp_path):
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    single_path = L.false_rejection_gate([strict], {TASK: [reference_run(), other_reason_run()]}, REPLAYS, REROLLS,
                                         None, SIGS)
    assert not single_path.passed
    assert single_path.failures == ["task t1: the required atoms reject every held-out frontier Run"]
    assert single_path.metrics["per_task"][TASK]["fraction"] == 1.0
    some = L.false_rejection_gate([strict], {TASK: [reference_run(), alt_path_run(), other_reason_run()]},
                                  {TASK: dict(REPLAYS[TASK], tr3=replay_row("tr3", True, run_id="alt"))}, REROLLS,
                                  None, SIGS)
    assert some.passed and some.metrics["per_task"][TASK]["fraction"] == 0.5
    assert some.metrics["per_task"][TASK]["rejected_ids"] == ["rr2"]
    assert L.false_rejection_gate([], {}, {}, {}, None, []).passed
