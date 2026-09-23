"""Keyed draws (D212): the same id and salt draw the same value, and no other id is an input.

The domain is invented: Tasks of a lending desk, Runs named after them. What is under test is the
draw, not the domain, so the ids are the only thing that matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback import sampling
from kullback.builder import world_tools
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
    seeds = [sampling.sample_seed("run_seed", f"{task_id}-r0", SALT) for task_id in TASKS]
    assert all(0 <= seed < sampling.SEED_SPACE for seed in seeds)
    assert seeds == [sampling.sample_seed("run_seed", f"{task_id}-r0", SALT) for task_id in TASKS]


def test_a_draw_of_one_kind_or_one_salt_is_not_the_draw_of_another_for_the_same_id():
    assert sampling.sample_key("reroll", "open-facility", SALT) != sampling.sample_key(
        "probe_slot", "open-facility", SALT)
    assert sampling.sample_key("reroll", "open-facility", SALT) != sampling.sample_key(
        "reroll", "open-facility", "another-desk-salt")


def test_an_id_carrying_the_separator_is_not_the_same_draw_as_the_pair_it_looks_like():
    assert sampling.sample_key("reroll", "facility:1", SALT) != sampling.sample_key("reroll:facility", "1", SALT)


# --- independence -----------------------------------------------------------

def test_adding_an_id_leaves_every_other_ids_draw_unchanged():
    before = {task_id: sampling.sample_key("reroll", task_id, SALT) for task_id in TASKS}
    grown = TASKS + ("novate-loan",)
    after = {task_id: sampling.sample_key("reroll", task_id, SALT) for task_id in grown}
    assert all(after[task_id] == before[task_id] for task_id in TASKS)
    early = [sampling.sample_seed("run_seed", f"open-facility-{i}", SALT) for i in range(3)]
    grown_count = [sampling.sample_seed("run_seed", f"open-facility-{i}", SALT) for i in range(6)]
    assert grown_count[:3] == early


def test_the_order_a_budget_goes_out_in_does_not_move_when_an_id_is_added_or_dropped():
    order = sampling.keyed_order("probe_slot", TASKS, SALT)
    grown = sampling.keyed_order("probe_slot", TASKS + ("novate-loan",), SALT)
    assert [task_id for task_id in grown if task_id in TASKS] == order
    dropped = sampling.keyed_order("probe_slot", [t for t in TASKS if t != order[0]], SALT)
    assert dropped == order[1:]
    assert order == sampling.keyed_order("probe_slot", tuple(reversed(TASKS)), SALT)


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


# --- what was drawn --------------------------------------------------------

def test_draws_since_a_baseline_count_each_kind_and_reading_takes_nothing():
    baseline = sampling.draws_by_kind()
    sampling.sample_key("reroll", "open-facility", SALT)
    sampling.sample_key("reroll", "raise-limit", SALT)
    sampling.sample_seed("run_seed", "open-facility-0", SALT)
    assert sampling.draws_since(baseline) == {"reroll": 2, "run_seed": 1}
    assert sampling.draws_since(baseline) == {"reroll": 2, "run_seed": 1}, "reading is not taking"
    assert sampling.draws_since(sampling.draws_by_kind()) == {}


# --- the anchor, which is the one draw with a workdir behind it -------------

def test_the_anchor_holds_out_the_same_runs_on_two_invocations_over_one_task_list(workdir):
    tasks = _tasks(TASKS)
    first = world_tools.draw_anchor(tasks, workdir)
    second = world_tools.draw_anchor(tasks, workdir)
    assert first["held_out"] == second["held_out"]


def test_a_task_added_to_a_fresh_build_moves_no_other_tasks_held_out_runs(workdir):
    was = world_tools.draw_anchor(_tasks(TASKS), workdir)["held_out"]
    now = world_tools.draw_anchor(_tasks(TASKS + ("novate-loan",)), workdir)["held_out"]
    assert {task_id: now[task_id] for task_id in TASKS} == was


def test_a_run_added_to_a_task_keeps_the_held_out_runs_the_anchor_already_had(workdir):
    task = Task(id="open-facility", run_ids=[f"open-facility-r{i}" for i in range(10)])
    grown = Task(id="open-facility", run_ids=[f"open-facility-r{i}" for i in range(11)])
    was = world_tools.draw_anchor([task], workdir)["held_out"]["open-facility"]
    now = world_tools.draw_anchor([grown], workdir)["held_out"]["open-facility"]
    assert was == now, "a stored anchor keeps the Runs it held out"


def test_the_anchor_records_the_salt_it_was_keyed_under_and_reads_it_back(workdir):
    anchor = world_tools.draw_anchor(_tasks(TASKS), workdir)
    assert anchor["salt"] == sampling.read_salt(workdir)
    body = json.loads((workdir / "anchor.json").read_text(encoding="utf-8"))
    assert body.get("salt", "") == anchor["salt"]
    assert body["held_out"] == anchor["held_out"]


def test_an_anchor_stored_before_the_salt_existed_loads_and_keeps_its_own_rule(workdir):
    (Path(workdir) / "anchor.json").write_text(json.dumps({
        "held_out": {"open-facility": ["open-facility-r3"]}, "unguarded": [],
        "share": 0.2, "min_runs": 3, "seed": 20260827}), encoding="utf-8")
    grown = world_tools.draw_anchor(_tasks(TASKS), workdir)
    assert grown["held_out"]["open-facility"] == ["open-facility-r3"]
