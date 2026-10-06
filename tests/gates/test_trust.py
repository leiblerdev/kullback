"""The refuse ruling reads re-rolls already paid for (D128), and a trusted Verifier is the composition of the
suite, the pool, the history and the refusals (D126)."""

from __future__ import annotations

import pytest

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
from gates.verifier_fixtures import (
    alt_path_run,
    make_run,
    other_reason_run,
    reference_events,
    reference_run,
    write_events_jsonl,
    wrong_run,
)
from kullback.gates import trust as T
from kullback.gates.probes import version_hash
from kullback.round_snapshot import counts_of
from kullback.runner.canon import CanonRules
from kullback.runner.records import Column, Conduct, EndState, EntitySchema, ExpectedCell, ValueSource, Verifier

REFUSAL = {TASK: {"task_id": TASK, "reason": "no frontier Run reaches the End state", "round": 1}}
NOTHING_FINISHED = ({TASK: {"tr1": replay_row("tr1", False)}}, {TASK: [reroll_row("reroll-t1-0", "max_steps")]})


def test_a_refusal_is_admitted_only_when_no_frontier_run_of_the_task_finished():
    replays, rerolls = NOTHING_FINISHED
    ruling = T.refuse_gate(REFUSAL, replays, rerolls)
    assert ruling.passed and ruling.stage == "refuse"
    assert ruling.metrics == {"refused": [TASK], "rejected": []}
    assert T.finished_runs(TASK, replays, rerolls) == []
    assert T.refuse_gate({}, {}, {}).passed


FINISHED_REROLLS = {TASK: [reroll_row("reroll-t1-0", "max_steps"), reroll_row("reroll-t1-2", "user_stop")]}


@pytest.mark.parametrize("replays, rerolls, finished", [
    ({TASK: {"tr1": replay_row("tr1", True)}}, {}, "replay-tr1"),
    ({TASK: {"tr1": replay_row("tr1", False)}}, FINISHED_REROLLS, "reroll-t1-2"),
], ids=["confirmed_replay", "finished_reroll"])
def test_a_refusal_of_a_task_the_frontier_finished_is_rejected_and_names_the_run_that_finished(
        replays, rerolls, finished):
    ruling = T.refuse_gate(REFUSAL, replays, rerolls)
    assert not ruling.passed
    assert ruling.failures == [f"task t1: the frontier finished it: {finished}"]
    assert ruling.metrics == {"refused": [], "rejected": [TASK]}
    assert T.finished_runs(TASK, {TASK: {"tr1": replay_row("tr1", True)}}, FINISHED_REROLLS) == [
        "replay-tr1", "reroll-t1-2"]


def _world(tmp_path, verifier=None):
    verifier = verifier if verifier is not None else base(tmp_path)
    replays = {TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}
    rerolls = {TASK: [reroll_row("rr2", "success"), reroll_row("alt", "success")]}
    runs = {TASK: [reference_run(), alt_path_run(), other_reason_run()]}
    return dict(task_status={TASK: status()}, verifiers=[verifier],
                probes={TASK: pool(probe("probe-t1-1", wrong_run(), verifier))},
                history=history(version(verifier)), refusals={}, task_runs=runs, replays=replays, rerolls=rerolls,
                canon_rules=CanonRules(), sigs=SIGS)


def test_a_verifier_meeting_every_trust_condition_is_trusted(tmp_path):
    ruling = T.trusted_gate(**_world(tmp_path))
    assert ruling.passed and ruling.stage == "trusted"
    assert ruling.metrics["trusted"] == [TASK] and ruling.metrics["untrusted"] == {}
    assert ruling.metrics["probes_passing"] == 0 and ruling.metrics["refused"] == {}
    assert T.trusted_gate({}, [], {}, {}, {}, {}, {}, {}, None, []).passed


def _skipping_the_read():
    """The seeds' write and answer without the read every seed made before writing."""
    return make_run("skip", [e for e in reference_events() if e["payload"].get("id") != "c0"])


def test_a_verifier_with_a_passing_probe_that_skipped_a_step_is_not_trusted_and_the_reason_names_the_probe(tmp_path):
    world = _world(tmp_path)
    verifier = world["verifiers"][0]
    world["probes"] = {TASK: pool(probe("probe-t1-1", wrong_run(), verifier),
                                  probe("probe-t1-2", _skipping_the_read(), verifier))}
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


def test_a_suite_failure_names_each_failing_check_and_says_why_a_check_had_no_input(tmp_path):
    """The rule is unchanged: a Task the suite refused is untrusted whatever the reason. What the
    ruling says changes, because a check with no input asks for more Runs of the Task and a check
    that failed asks for a looser Verifier, and until D173 both read the same. A reader groups the
    Tasks that stop at the suite by what stopped them, so the reason names each check in the
    suite's own order and says why the ones with no input had none."""
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


def test_a_task_held_only_by_checks_that_did_not_run_is_untrusted_and_named_in_not_run_only(tmp_path):
    """F45: every check that ran passed, so the Task is held by the checks nobody could run. It stays
    untrusted with each one named and why, and `not_run_only` lets a proposal ruling tell that apart
    from a Task whose atoms something failed."""
    world = _world(tmp_path)
    checks = {name: True for name in T.D79_STAGES.values()}
    checks["second_path_passes"] = False
    world["task_status"] = {TASK: status(verifier_passed=False, checks=checks, not_run=["verifier_alt_path"])}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == [] and ruling.metrics["not_run_only"] == [TASK]
    assert ruling.metrics["untrusted"] == {
        TASK: "the D79 suite did not pass: second_path_passes not run (one Reference, so there is no "
              "second path to score)"}
    checks["mutation_flips"] = False
    world["task_status"] = {TASK: status(verifier_passed=False, checks=checks, not_run=["verifier_alt_path"])}
    assert T.trusted_gate(**world).metrics["not_run_only"] == [], "a check that ran and failed holds it"


def test_a_task_held_by_not_run_checks_is_still_ruled_on_its_other_steps_first(tmp_path):
    world = _world(tmp_path)
    verifier = world["verifiers"][0]
    world["probes"] = {TASK: pool(probe("probe-t1-2", _skipping_the_read(), verifier))}
    world["task_status"] = {TASK: status(verifier_passed=False, checks={"second_path_passes": False},
                                         not_run=["verifier_alt_path"])}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["untrusted"] == {TASK: "probe probe-t1-2 scores a pass"}
    assert ruling.metrics["not_run_only"] == []


def _seeded_world(tmp_path, foreign: tuple = ()):
    """The trusted world with its seed Runs on disk under a workdir, the named seeds played by another Task."""
    world = _world(tmp_path)
    workdir = tmp_path / "work"
    folder = workdir / "runs" / TASK
    folder.mkdir(parents=True)
    rows = {}
    for run in (reference_run(), alt_path_run(), other_reason_run()):
        owner = "t_other" if run.run_id in foreign else TASK
        write_events_jsonl(run.model_copy(update={"task_id": owner}), folder / f"{run.run_id}.jsonl")
        rows[run.run_id] = f"runs/{TASK}/{run.run_id}.jsonl"
    world["replays"] = {TASK: {"tr1": replay_row("tr1", True, run_id="ref", path=rows["ref"])}}
    world["rerolls"] = {TASK: [reroll_row(run_id, "success", path=rows[run_id]) for run_id in ("rr2", "alt")]}
    return dict(world, workdir=workdir)


def test_a_verifier_seeded_from_its_own_task_runs_passes_the_provenance_step(tmp_path):
    world = _seeded_world(tmp_path)
    assert set(world["verifiers"][0].seed_run_ids) >= {"ref", "alt"}
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == [TASK] and ruling.metrics["untrusted_seeds"] == {}


def test_a_verifier_seeded_from_another_tasks_run_fails_the_trusted_gate_naming_the_seed(tmp_path):
    """D281: a seed file that the loader attributes to another Task withdraws trust; the history stays."""
    world = _seeded_world(tmp_path, foreign=("alt",))
    kept = world["history"][TASK].model_copy(deep=True)
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == [] and ruling.metrics["untrusted_seeds"] == {TASK: ["alt"]}
    assert ruling.failures == [f"task t1: version {version_hash(world['verifiers'][0])} was derived from seeds "
                               "that are not Runs of this Task: alt"]
    assert world["history"][TASK] == kept


def test_a_seed_no_row_or_runs_folder_can_find_is_not_attributable(tmp_path):
    world = _seeded_world(tmp_path)
    (world["workdir"] / "runs" / TASK / "alt.jsonl").unlink()
    assert T.trusted_gate(**world).metrics["untrusted_seeds"] == {TASK: ["alt"]}


# --- D319: trust is decided by code on the expected end states ---

START = {"orders": {"W1": {"status": "pending", "note": "none"}}}
END = {"orders": {"W1": {"status": "cancelled", "note": "none"}}}
CODE_CHECKS = {"plausible_wrong_fails": True, "leak_check_clean": True}


def _turn(text):
    return {"type": "user_turn", "payload": {"text": text}}


def _replay(start=START, end=END, run_id="ref"):
    """A faithful replay: the user asks and confirms, the agent writes, the world moves from start to end."""
    return make_run(run_id, [
        _turn("Please cancel order W1."),
        _turn("Yes, go ahead."),
        {"type": "tool_call", "payload": {"id": "c1", "name": "cancel_order", "args": {"order_id": "W1"}}},
        {"type": "tool_result", "payload": {"id": "c1", "result": {"order_id": "W1", "status": "cancelled"}}},
        {"type": "stop", "payload": {"start_state": start, "end_state": end}},
    ])


def _cell(field="status", value="cancelled", sourced=True, table="orders"):
    said = ValueSource(kind="user_turn", ptr={"turn": 0})
    return ExpectedCell(table=table, row_id="W1", field=field, value=value, row_source=said,
                        source=said if sourced else None)


def _code_world(cells, *, conduct=(), run=None, checks=CODE_CHECKS, schema=None):
    """A Verifier with expected end states and no atoms, its faithful replay, and no probe pool."""
    verifier = Verifier(task_id=TASK, seed_run_ids=["ref"], expected=[EndState(cells=list(cells))],
                        conduct=list(conduct))
    return dict(task_status={TASK: status(checks=dict(checks))}, verifiers=[verifier], probes={}, history={},
                refusals={}, task_runs={TASK: [run or _replay()]},
                replays={TASK: {"tr1": replay_row("tr1", True, run_id="ref")}}, rerolls={},
                canon_rules=CanonRules(), sigs=SIGS, schema=schema)


def test_a_verifier_that_passes_every_code_gate_is_trusted_with_no_probe_pool_present():
    ruling = T.trusted_gate(**_code_world([_cell()]))
    assert ruling.metrics["trusted"] == [TASK] and ruling.metrics["trust_tier"] == {TASK: "trusted"}
    assert ruling.metrics["unsupported_cells"] == {TASK: 0} and ruling.metrics["legacy_only"] == []


def test_a_verifier_with_one_unsupported_cell_is_unconfirmed_not_trusted_and_names_the_cell():
    ruling = T.trusted_gate(**_code_world([_cell(sourced=False)]))
    assert ruling.metrics["trusted"] == [] and ruling.metrics["trust_tier"] == {TASK: "unconfirmed"}
    assert ruling.metrics["untrusted"] == {TASK: "unconfirmed: unsupported cell orders.W1.status"}
    assert ruling.metrics["unsupported_cells"] == {TASK: 1}


def test_a_reference_run_whose_match_is_unsettled_leaves_the_task_unconfirmed_naming_the_column_kind():
    schema = EntitySchema(tables=["orders"], columns=[Column(table="orders", name="note", **{"class": "semantic"})])
    replay = _replay(end={"orders": {"W1": {"status": "cancelled", "note": "item arrived broken"}}})
    cells = [_cell(), _cell(field="note", value="the item came damaged")]
    ruling = T.trusted_gate(**_code_world(cells, run=replay, schema=schema))
    assert ruling.metrics["trust_tier"] == {TASK: "unconfirmed"}
    assert ruling.metrics["untrusted"] == {TASK: "unconfirmed: unsettled orders.note on faithful replay ref"}


def test_a_task_with_an_empty_expected_diff_and_one_conduct_is_not_trusted_when_the_empty_run_passes_it():
    """The confirmation rule holds vacuously where nothing is written, so the empty Run passes (D295)."""
    still = _replay(end=START)
    rule = Conduct(kind="confirm_before_write", tool="cancel_order", source=ValueSource(kind="policy"))
    world = _code_world([], conduct=[rule], run=still)
    world["verifiers"][0].expected[0].cells.clear()
    ruling = T.trusted_gate(**world)
    assert ruling.metrics["trusted"] == [] and ruling.metrics["trust_tier"] == {TASK: "pending"}
    assert ruling.metrics["untrusted"] == {TASK: "pending: the empty Run passes"}


def test_a_contradicted_cell_flags_the_row_and_does_not_change_the_tier():
    start = {"orders": {"W1": {"status": "pending", "note": "none"}, "W2": {"status": "returned", "note": "none"}}}
    end = {"orders": {"W1": {"status": "cancelled", "note": "none"}, "W2": {"status": "returned", "note": "none"}}}
    replay = make_run("ref", [_turn("Please mark W1 returned."), _turn("Yes, go ahead."),
                              {"type": "tool_call", "payload": {"id": "c1", "name": "cancel_order",
                                                                "args": {"order_id": "W1"}}},
                              {"type": "stop", "payload": {"start_state": start, "end_state": end}}])
    world = _code_world([_cell()], run=replay)
    ruling = T.code_ruling(world["verifiers"][0], [replay], world["task_status"][TASK], CanonRules(),
                           T.write_tools_of(SIGS))
    assert ruling.tier == "trusted" and ruling.contradicts == ["orders.W1.status"]
    gated = T.trusted_gate(**world)
    assert gated.metrics["trust_tier"] == {TASK: "trusted"} and gated.metrics["consistency_flags"] == {TASK: 1}


def test_a_reference_write_no_user_turn_asked_for_is_flagged_by_its_tool_and_does_not_gate():
    replay = make_run("ref", [_turn("I have a problem with an order."),
                              {"type": "tool_call", "payload": {"id": "c1", "name": "cancel_pending_order",
                                                                "args": {"order_id": "W1"}}},
                              {"type": "stop", "payload": {"start_state": START, "end_state": END}}])
    world = _code_world([_cell()], run=replay)
    ruling = T.code_ruling(world["verifiers"][0], [replay], world["task_status"][TASK], CanonRules(),
                           T.write_tools_of(SIGS))
    assert ruling.unasked == ["cancel_pending_order"] and ruling.tier == "trusted"


def test_a_workdir_with_specs_reports_the_spec_tier_through_the_workdir_ruling(tmp_path):
    spec_tiers = {"a": ("trusted", {"failing": None}),
                  "b": ("unconfirmed", {"failing": "unsupported cell orders.W1.status", "contradicts": ["x"]}),
                  "c": ("replay_only", {"failing": "independent_pass"})}
    ruling = T.workdir_trusted_ruling(tmp_path, spec_tiers=spec_tiers)
    assert ruling.metrics["trusted"] == ["a"]
    assert ruling.metrics["trust_tier"] == {"a": "trusted", "b": "unconfirmed", "c": "replay_only"}
    assert ruling.metrics["untrusted"] == {"b": "unconfirmed: unsupported cell orders.W1.status",
                                           "c": "replay_only: independent_pass"}
    assert ruling.metrics["consistency_flags"] == {"a": 0, "b": 1, "c": 0}


def test_the_status_table_shows_trusted_unconfirmed_refused_and_the_legacy_count_on_one_row():
    rows = [{"task_id": "a", "trust_tier": "trusted", "trusted": True, "legacy_trusted": True},
            {"task_id": "b", "trust_tier": "unconfirmed", "legacy_trusted": True,
             "false_rejection": 0.5, "false_rejection_pool": 4},
            {"task_id": "c", "trust_tier": "refused", "refused": True},
            {"task_id": "d", "trust_tier": "pending", "legacy_trusted": True, "consistency_flags": 2}]
    line = T.trust_row(counts_of(rows))
    assert "\n" not in line
    assert line == ("trusted 1 | unconfirmed 1 | refused 1 | pending 1 | legacy rule trusted 3 | "
                    "valid other solutions failing 2 of 4 held-out Runs | consistency flags 1")
    ruling = T.trusted_gate(**_code_world([_cell(sourced=False)]))
    assert T.tier_counts(ruling.metrics)["unconfirmed"] == 1
