"""The router moves each ruling on the bus once, by its target, and sets a Task aside past the round cap."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from kullback.agent.bus import Bus
from kullback.spec import events
from kullback.spec import router as R
from kullback.spec.schema import load_spec, save_spec
from tests.spec.fixtures import spec


def _ruling(bus, target, number=1, task_id="t1", **extra):
    payload = events.ruling_payload(task_id, target, extra.get("check_ids", ["c1"]), extra.get("run_id"),
                                    "check_unpassable", ["because c1"], None, 0, number)
    return events.ruling_filed(bus, payload)


@pytest.fixture
def no_writer(monkeypatch):
    """The Spec writer has not landed: importing it fails."""
    monkeypatch.setitem(sys.modules, R.WRITER_MODULE, None)


def _setup(tmp_path, **callees):
    save_spec(tmp_path, spec())
    bus = Bus(tmp_path / "bus.jsonl", agent="test")
    router = R.Router(tmp_path, bus, **callees)
    router.attach()
    return bus, router


def _actions(router):
    return [(a["target"], a["action"]) for a in router.state["actions"]]


def test_a_ruling_on_a_run_rerolls_the_task_through_the_runner_callee(tmp_path):
    calls = []
    bus, router = _setup(tmp_path, reroll=lambda task_id, count: calls.append((task_id, count)))
    _ruling(bus, "run", run_id="r2")
    assert calls == [("t1", R.REROLL_COUNT)]
    assert _actions(router) == [("run", "reroll")]
    assert load_spec(tmp_path, "t1").round == 1 and load_spec(tmp_path, "t1").rulings_open == 0


def test_a_ruling_on_a_reference_run_rejects_it_before_the_reroll(tmp_path):
    rejected = []
    bus, router = _setup(tmp_path, reject_reference=lambda w, t, r, why: rejected.append((t, r)),
                         reference_ids_of=lambda w, t: {"ref"})
    _ruling(bus, "run", run_id="ref")
    assert rejected == [("t1", "ref")]
    assert _actions(router) == [("run", "reference_rejected_and_reroll_queued")]
    assert router.state["queue"][0]["move"] == "reroll" and load_spec(tmp_path, "t1").rulings_open == 1


def test_without_the_injected_exclusion_a_ruling_on_a_run_rerolls_only_and_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=R.__name__):
        bus, router = _setup(tmp_path)
        _ruling(bus, "run", run_id="ref")
    assert _actions(router) == [("run", "reroll_queued")]
    assert any("no Reference exclusion" in record.getMessage() for record in caplog.records)


def test_a_ruling_on_a_check_without_a_writer_is_deferred_and_kept_as_pending_repair(tmp_path, no_writer, caplog):
    bus, router = _setup(tmp_path)
    with caplog.at_level(logging.WARNING, logger=R.__name__):
        _ruling(bus, "check")
    saved = load_spec(tmp_path, "t1")
    assert saved.pending_repair == [{"seq": 1, "number": 1, "target": "check", "check_ids": ["c1"]}]
    assert saved.rulings_open == 1 and _actions(router) == [("check", "pending_repair")]
    deferred = [r.event.payload for r in bus.replay() if getattr(r.event, "name", "") == events.SPEC_REPAIR_DEFERRED]
    assert deferred == [{"task_id": "t1", "number": 1}]
    assert any("pending_repair" in record.getMessage() for record in caplog.records)


def test_the_writer_entry_gets_the_ruling_model_and_ceiling_and_its_repair_closes_the_ruling(tmp_path, monkeypatch):
    bus = Bus(tmp_path / "bus.jsonl", agent="test")
    calls = []

    def repair_for_ruling(workdir, task_id, ruling, model, ceiling_usd):
        calls.append((task_id, ruling["number"], model, ceiling_usd))
        events.spec_repaired(bus, task_id, 2, ruling["check_ids"])
        return load_spec(workdir, task_id)

    monkeypatch.setitem(sys.modules, R.WRITER_MODULE, types.SimpleNamespace(repair_for_ruling=repair_for_ruling))
    save_spec(tmp_path, spec())
    router = R.Router(tmp_path, bus, model="m", ceiling_usd=1.5)
    router.attach()
    _ruling(bus, "intent")
    assert calls == [("t1", 1, "m", 1.5)] and _actions(router) == [("intent", "repair")]
    assert load_spec(tmp_path, "t1").rulings_open == 0 and load_spec(tmp_path, "t1").round == 1


def test_set_aside_is_the_mark_or_the_round_cap_with_a_ruling_open():
    assert R.is_set_aside(spec().model_copy(update={"set_aside": R.NO_AGREEMENT}))
    assert R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 1}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 0}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP - 1, "rulings_open": 1}))


def test_a_ruling_on_the_environment_goes_to_the_builder_note(tmp_path):
    notes = []
    bus, router = _setup(tmp_path, note=lambda task_id, ruling: notes.append((task_id, ruling["number"])))
    _ruling(bus, "environment")
    assert notes == [("t1", 1)] and _actions(router) == [("environment", "builder_note")]


def test_a_ruling_after_the_round_cap_sets_the_task_aside_once(tmp_path):
    bus, router = _setup(tmp_path, reroll=lambda task_id, count: None)
    for number in range(1, R.ROUNDS_CAP + 3):
        _ruling(bus, "run", number=number, run_id="r2")
    assert [a for _, a in _actions(router)] == ["reroll"] * R.ROUNDS_CAP + ["set_aside", "ignored_set_aside"]
    aside = [r.event.payload for r in bus.replay() if getattr(r.event, "name", "") == events.TASK_SET_ASIDE]
    assert aside == [{"task_id": "t1", "reason": R.NO_AGREEMENT, "rounds": R.ROUNDS_CAP}]
    assert load_spec(tmp_path, "t1").set_aside == R.NO_AGREEMENT and R.is_set_aside(load_spec(tmp_path, "t1"))


def test_replaying_the_bus_handles_each_ruling_once(tmp_path):
    calls = []
    save_spec(tmp_path, spec())
    bus = Bus(tmp_path / "bus.jsonl", agent="test")
    _ruling(bus, "run", run_id="r2")
    first = R.Router(tmp_path, bus, reroll=lambda task_id, count: calls.append(task_id))
    first.attach()
    assert first.catch_up() == 0
    again = R.Router(tmp_path, Bus(tmp_path / "bus.jsonl", agent="other"),
                     reroll=lambda task_id, count: calls.append(task_id))
    again.attach()
    assert calls == ["t1"] and R.load_state(tmp_path)["last_seq"] == 1


def test_a_task_without_a_spec_counts_its_rounds_in_the_router_state(tmp_path):
    bus, router = _setup(tmp_path, reroll=lambda task_id, count: None)
    _ruling(bus, "run", task_id="t2", run_id="r2")
    assert R.load_state(tmp_path)["rounds"] == {"t2": 1}
