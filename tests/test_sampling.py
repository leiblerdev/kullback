"""Keyed draws (D212): the same id and salt draw the same value, and no other id is an input.

The domain is invented: Tasks of a lending desk, Runs named after them. What is under test is the
draw, not the domain, so the ids are the only thing that matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import sampling
from kullback.builder.pipeline import ANCHOR_KIND, choose_anchor, load_anchor
from kullback.runner.records import Task

SALT = "loan-desk-salt"
TASKS = ("open-facility", "raise-limit", "close-account", "restate-terms", "waive-fee")


@pytest.fixture()
def workdir(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


def _tasks(ids, runs=10):
    return [Task(id=task_id, run_ids=[f"{task_id}-r{i}" for i in range(runs)]) for task_id in ids]


# --- the draw itself --------------------------------------------------------

def test_the_same_kind_id_and_salt_draw_the_same_value_on_every_invocation():
    first = [sampling.sample_key("reroll", task_id, SALT) for task_id in TASKS]
    again = [sampling.sample_key("reroll", task_id, SALT) for task_id in TASKS]
    assert first == again
    assert all(0 <= key < sampling.KEY_SPACE for key in first)


def test_a_draw_of_one_kind_is_not_the_draw_of_another_for_the_same_id():
    assert sampling.sample_key("reroll", "open-facility", SALT) != sampling.sample_key(
        "probe_slot", "open-facility", SALT)


def test_a_different_salt_draws_a_different_value_for_the_same_id():
    assert sampling.sample_key("reroll", "open-facility", SALT) != sampling.sample_key(
        "reroll", "open-facility", "another-desk-salt")


def test_an_id_carrying_the_separator_is_not_the_same_draw_as_the_pair_it_looks_like():
    assert sampling.sample_key("reroll", "facility:1", SALT) != sampling.sample_key("reroll:facility", "1", SALT)


def test_a_seed_is_a_non_negative_thirty_two_bit_value_of_the_same_draw():
    seeds = [sampling.sample_seed("run_seed", f"{task_id}-r0", SALT) for task_id in TASKS]
    assert all(0 <= seed < sampling.SEED_SPACE for seed in seeds)
    assert seeds == [sampling.sample_seed("run_seed", f"{task_id}-r0", SALT) for task_id in TASKS]


# --- independence -----------------------------------------------------------

def test_adding_an_id_leaves_every_other_ids_draw_unchanged():
    before = {task_id: sampling.sample_key("reroll", task_id, SALT) for task_id in TASKS}
    grown = TASKS + ("novate-loan",)
    after = {task_id: sampling.sample_key("reroll", task_id, SALT) for task_id in grown}
    assert all(after[task_id] == before[task_id] for task_id in TASKS)


def test_the_order_a_budget_goes_out_in_does_not_move_when_an_id_is_added_or_dropped():
    order = sampling.keyed_order("probe_slot", TASKS, SALT)
    grown = sampling.keyed_order("probe_slot", TASKS + ("novate-loan",), SALT)
    assert [task_id for task_id in grown if task_id in TASKS] == order
    dropped = sampling.keyed_order("probe_slot", [t for t in TASKS if t != order[0]], SALT)
    assert dropped == order[1:]


def test_the_order_is_the_ids_own_and_not_the_order_they_were_handed_over_in():
    assert sampling.keyed_order("probe_slot", TASKS, SALT) == sampling.keyed_order(
        "probe_slot", tuple(reversed(TASKS)), SALT)


def test_a_share_holds_the_membership_of_every_other_id_when_one_id_joins():
    runs = [f"r{i}" for i in range(20)]
    before = set(sampling.keyed_share("anchor", runs, SALT, 0.20))
    after = set(sampling.keyed_share("anchor", runs + ["r20"], SALT, 0.20))
    assert (after - {"r20"}) == before


def test_a_share_with_a_floor_and_a_ceiling_gives_exactly_the_count_asked_for():
    runs = [f"r{i}" for i in range(10)]
    chosen = sampling.keyed_share("anchor", runs, SALT, 0.20, minimum=2, maximum=2)
    assert len(chosen) == 2
    assert chosen == sampling.keyed_share("anchor", runs, SALT, 0.20, minimum=2, maximum=2)
    assert set(chosen) <= set(runs)


def test_a_count_that_grows_keeps_the_draws_the_earlier_count_took():
    early = [sampling.sample_seed("run_seed", f"open-facility-{i}", SALT) for i in range(3)]
    grown = [sampling.sample_seed("run_seed", f"open-facility-{i}", SALT) for i in range(6)]
    assert grown[:3] == early


def test_a_run_discarded_and_made_again_under_its_own_id_draws_the_seed_it_had():
    once = sampling.sample_seed("run_seed", "reroll-open-facility-1", SALT)
    assert sampling.sample_seed("run_seed", "reroll-open-facility-1", SALT) == once


# --- the salt ---------------------------------------------------------------

def test_a_workdir_stores_its_salt_once_and_a_resume_reads_the_stored_one_back(workdir, monkeypatch):
    first = sampling.build_salt(workdir)
    assert json.loads(sampling.salt_path(workdir).read_text(encoding="utf-8"))["salt"] == first
    monkeypatch.setenv(sampling.SALT_ENV, "a-salt-asked-for-later")
    assert sampling.build_salt(workdir) == first, "a resume draws what the first launch drew"


def test_a_fresh_workdir_takes_the_asked_for_salt_and_two_fresh_workdirs_agree_without_one(tmp_path, monkeypatch):
    monkeypatch.setenv(sampling.SALT_ENV, "a-salt-asked-for")
    asked = tmp_path / "asked"
    assert sampling.build_salt(asked) == "a-salt-asked-for"
    monkeypatch.delenv(sampling.SALT_ENV)
    left, right = tmp_path / "left", tmp_path / "right"
    assert sampling.build_salt(left) == sampling.build_salt(right) == sampling.DEFAULT_SALT


def test_a_workdir_that_has_not_drawn_yet_reads_back_no_salt_and_labels_none(workdir):
    assert sampling.read_salt(workdir) is None
    assert sampling.salt_label(None) == ""
    assert len(sampling.salt_label(SALT)) == 8


# --- what a round drew -----------------------------------------------------

def test_a_round_is_told_how_many_draws_of_each_kind_it_took_since_the_round_before():
    baseline = sampling.draws_by_kind()
    sampling.sample_key("reroll", "open-facility", SALT)
    sampling.sample_key("reroll", "raise-limit", SALT)
    sampling.sample_seed("run_seed", "open-facility-0", SALT)
    assert sampling.draws_since(baseline) == {"reroll": 2, "run_seed": 1}
    assert sampling.draws_since(baseline) == {"reroll": 2, "run_seed": 1}, "reading is not taking"
    assert sampling.draws_since(sampling.draws_by_kind()) == {}


def test_a_loop_built_after_a_draw_does_not_report_that_draw_as_its_own(tmp_path):
    """Greptile P1 (PR 26): the counter is process-global, so a second Loop starts from what it finds.

    Without this the first round of any Loop but the first in a process reports the draws an
    earlier build or an earlier Loop took.
    """
    from kullback import rounds
    from kullback.builder import agent as builder_agent
    from kullback.builder.build import BuildPlan

    sampling.sample_key("reroll", "open-facility", SALT)
    plan = BuildPlan(workdir=tmp_path / "work")
    loop = rounds.Loop(plan=plan, builder=builder_agent.build_harness(plan))
    assert sampling.draws_since(loop.draws_seen) == {}
    sampling.sample_key("reroll", "raise-limit", SALT)
    assert sampling.draws_since(loop.draws_seen) == {"reroll": 1}


# --- the anchor, which is the one draw with a workdir behind it -------------

def test_the_anchor_holds_out_the_same_runs_on_two_invocations_over_one_task_list(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    tasks = _tasks(TASKS)
    assert choose_anchor(tasks, left).held_out == choose_anchor(tasks, right).held_out


def test_a_task_added_to_a_fresh_build_moves_no_other_tasks_held_out_runs(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    was = choose_anchor(_tasks(TASKS), before).held_out
    now = choose_anchor(_tasks(TASKS + ("novate-loan",)), after).held_out
    assert {task_id: now[task_id] for task_id in TASKS} == was


def test_a_run_added_to_a_task_leaves_the_other_runs_membership_to_their_own_keys(tmp_path):
    ten, eleven = tmp_path / "ten", tmp_path / "eleven"
    ten.mkdir()
    eleven.mkdir()
    task = Task(id="open-facility", run_ids=[f"open-facility-r{i}" for i in range(10)])
    grown = Task(id="open-facility", run_ids=[f"open-facility-r{i}" for i in range(11)])
    was = set(choose_anchor([task], ten).anchor_runs("open-facility"))
    now = set(choose_anchor([grown], eleven).anchor_runs("open-facility"))
    # The count is D81's share of a longer list, so it may take one more Run; what it must not do is
    # drop a Run whose key still ranks inside the share.
    assert was <= now or now <= was


def test_the_anchor_records_the_salt_it_was_keyed_under_and_reads_it_back(workdir):
    anchor = choose_anchor(_tasks(TASKS), workdir)
    assert anchor.salt == sampling.read_salt(workdir)
    assert load_anchor(workdir).salt == anchor.salt


def test_an_anchor_stored_before_the_salt_existed_loads_and_keeps_its_own_rule(workdir):
    (Path(workdir) / "anchor.json").write_text(json.dumps({
        "held_out": {"open-facility": ["open-facility-r3"]}, "unguarded": [],
        "share": 0.2, "min_runs": 3, "seed": 20260827}), encoding="utf-8")
    stored = load_anchor(workdir)
    assert stored.salt == ""
    assert "salt" not in stored.to_dict(), "an anchor drawn before D212 hashes as it always did"
    grown = choose_anchor(_tasks(TASKS), workdir)
    assert grown.anchor_runs("open-facility") == ["open-facility-r3"]
    assert grown.salt == "", "a Task appended to that anchor is drawn under the rule that anchor used"


def test_the_anchor_kind_is_the_one_name_every_held_out_draw_goes_through():
    assert ANCHOR_KIND == "anchor"
    runs = ["open-facility-r0", "open-facility-r1", "open-facility-r2"]
    assert sampling.keyed_share(ANCHOR_KIND, runs, SALT, 0.2, minimum=1, maximum=1) == sampling.keyed_share(
        "anchor", runs, SALT, 0.2, minimum=1, maximum=1)
