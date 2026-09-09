"""A derived artefact is retired when the source it was derived from is withdrawn (D208).

The world is a lending library: one write tool that puts a reserve on a copy of a title and one read
tool that reports what a copy is doing. Two Tasks are derived over the same Runs, one of them loses
its Reference in a later round, and the question is what happens to the Verifier that Reference
produced: it stops being scored, it says on the Task's row why, and the Task that kept its Reference
is untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examiner.worlds import World
from gates.verifier_fixtures import make_run, write_events_jsonl
from kullback.examiner import agent as examiner_agent
from kullback.examiner import lifecycle, loosen
from kullback.gates.loosening import false_rejection_gate
from kullback.gates.trust import trusted_gate
from kullback.runner.records import Run, Task, ToolSig, Verifier

WITHDRAWN, KEPT = "t1", "t2"
RESERVE = "reserve_copy"
RECORD = "copy_record"
SIGS = [ToolSig(name=RESERVE, kind="write", kind_confidence="high", unclassified=False),
        ToolSig(name=RECORD, kind="read", kind_confidence="high", unclassified=False)]
COPY = {"copy_id": "C7", "branch": "north", "state": "on the shelf"}


# --- the world -------------------------------------------------------------------------

def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return _ev("user_turn", content=text)


def says(text: str) -> dict:
    return _ev("model_call", reply={"content": text})


def call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return _ev("tool_call", id=cid, name=name, args=args, kind=kind)


def result(data, cid: str = "c1") -> dict:
    return _ev("tool_result", id=cid, result=data)


def _asked_then_reserved(run_id: str, task_id: str, opening: str = "Put a reserve on the almanac.") -> Run:
    """The recorded path: the branch is confirmed with the reader, then the reserve is placed."""
    return make_run(run_id, [
        user(opening),
        call(RECORD, {"copy_id": "C7"}, kind="read", cid="c0"),
        result(COPY, cid="c0"),
        says("Shall I reserve the north branch copy?"),
        user("yes"),
        call(RESERVE, {"copy_id": "C7", "branch": "north"}, cid="c1"),
        result({"copy_id": "C7", "state": "reserved"}, cid="c1"),
        says("Copy C7 at the north branch is reserved for you."),
    ], task_id=task_id)


def _reserved_without_asking(run_id: str, task_id: str) -> Run:
    """The same End state by another path, which is what the held-out pool holds."""
    return make_run(run_id, [
        user("Put a reserve on the almanac."),
        call(RECORD, {"copy_id": "C7"}, kind="read", cid="c0"),
        result(COPY, cid="c0"),
        call(RESERVE, {"copy_id": "C7", "branch": "north"}, cid="c1"),
        result({"copy_id": "C7", "state": "reserved"}, cid="c1"),
        says("Copy C7 at the north branch is reserved for you."),
    ], task_id=task_id)


def _reserved_nothing(run_id: str = "probe") -> Run:
    return make_run(run_id, [user("Put a reserve on the almanac."), says("I have noted it down.")],
                    task_id=WITHDRAWN)


def library(root: Path) -> World:
    """Two Tasks over the same Runs: a confirmed recording each and one finished re-run each."""
    workdir = root / "world"
    paths: dict[str, str] = {}
    for task_id in (WITHDRAWN, KEPT):
        folder = workdir / "runs" / task_id
        folder.mkdir(parents=True)
        for run_id, run in {"ref": _asked_then_reserved("ref", task_id),
                            "alt": _asked_then_reserved("alt", task_id, "Reserve the almanac for me please."),
                            "direct": _reserved_without_asking("direct", task_id)}.items():
            paths[f"{task_id}/{run_id}"] = write_events_jsonl(run, folder / f"{run_id}.jsonl")
    inputs = {
        "tasks": [Task(id=task_id, intent="reserve the copy of the title the reader named",
                       run_ids=["ref"]) for task_id in (WITHDRAWN, KEPT)],
        "sigs": list(SIGS), "constraints": [], "canon_rules": {},
        "replays": {task_id: {"ref": {"trace_id": "ref", "run_id": "ref", "confirmed": True,
                                      "path": paths[f"{task_id}/ref"], "reasons": []}}
                    for task_id in (WITHDRAWN, KEPT)},
        "rerolls": {task_id: [{"run_id": "alt", "path": paths[f"{task_id}/alt"],
                               "termination_reason": "success"}]
                    for task_id in (WITHDRAWN, KEPT)},
        "intents": {}, "user_rules": {}, "traces": [], "assisted_tools": [],
    }
    return World(workdir=workdir, inputs=inputs, paths=paths)


def _probe_runner(run: Run):
    def run_probe(model, verifier):
        return run
    return run_probe


def _derive(world: World, round_number: int = 0) -> None:
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(),
                                run_probe=_probe_runner(_reserved_nothing()), round=round_number)


def _hold_out_the_other_path(world: World) -> None:
    """The Run that reached the same End state another way, in every Task's legitimate pool.

    It goes into the Examiner's own re-roll file, which is what a later round's frontier looks like:
    the derivation never saw it, so it is nobody's Reference and the pool holds it as held out.
    """
    plan = _plan(world)
    plan.extra_rerolls = {task_id: [{"run_id": "direct", "path": world.paths[f"{task_id}/direct"],
                                     "termination_reason": "success"}]
                          for task_id in (WITHDRAWN, KEPT)}
    plan.write_state()


def _withdraw(world: World) -> None:
    """The Reference of one Task goes: its recording no longer replays and no re-run finished."""
    world.inputs["replays"][WITHDRAWN]["ref"]["confirmed"] = False
    world.inputs["replays"][WITHDRAWN]["ref"]["reasons"] = ["the recorded call replays otherwise"]
    world.inputs["rerolls"][WITHDRAWN] = []


def _plan(world: World):
    return world.plan(probe_model=object(), run_probe=_probe_runner(_reserved_nothing()))


def _verifier_file(world: World, task_id: str) -> Path:
    return world.workdir / "verifiers" / f"{task_id}.json"


@pytest.fixture
def derived(tmp_path) -> World:
    world = library(tmp_path)
    _derive(world)
    _hold_out_the_other_path(world)
    return world


# --- what the rule is, over a status row and an artefact --------------------------------

def test_a_verifier_is_live_only_while_its_task_holds_the_reference_it_was_derived_from():
    verifier = Verifier(task_id=WITHDRAWN, atoms=[], seed_run_ids=["r1", "r2"])
    held = {WITHDRAWN: {"reference_confirmed": True, "reference_run_ids": ["r1", "r2"]}}
    assert lifecycle.retirement(verifier, held[WITHDRAWN]) is None
    assert lifecycle.live([verifier], held) == [verifier]


def test_a_verifier_whose_task_holds_no_reference_this_round_is_retired_as_withdrawn():
    verifier = Verifier(task_id=WITHDRAWN, atoms=[], seed_run_ids=["r1"])
    rows = {WITHDRAWN: {"reference_confirmed": False, "reason": "the recordings disagree"}}
    live, retired = lifecycle.partition([verifier], rows)
    assert live == [] and [row["reason"] for row in retired] == [lifecycle.REFERENCE_WITHDRAWN]
    assert retired[0]["source_run_ids"] == ["r1"]


def test_a_verifier_derived_over_a_run_the_reference_no_longer_holds_is_retired_as_rederived():
    """A survivor choice keeps the Task's Reference but makes it another group of Runs; a Verifier
    derived over a Run that group has dropped is a check on an End state nobody chose."""
    verifier = Verifier(task_id=WITHDRAWN, atoms=[], seed_run_ids=["r1", "r2"])
    rows = {WITHDRAWN: {"reference_confirmed": True, "reference_run_ids": ["r3"]}}
    live, retired = lifecycle.partition([verifier], rows)
    assert live == [] and retired[0]["reason"] == lifecycle.REFERENCE_REDERIVED
    assert "r1" in retired[0]["text"] and "r2" in retired[0]["text"]


def test_a_reference_that_has_grown_a_run_is_the_same_source_and_retires_nothing():
    """More evidence behind the same Reference is not a different Reference: the next derivation
    reads the extra Run, and nothing on disk is stale in the meantime."""
    verifier = Verifier(task_id=WITHDRAWN, atoms=[], seed_run_ids=["r1"])
    rows = {WITHDRAWN: {"reference_confirmed": True, "reference_run_ids": ["r1", "r2"]}}
    assert lifecycle.live([verifier], rows) == [verifier]


def test_a_row_that_does_not_name_its_reference_is_judged_on_confirmation_alone():
    """A status row written before the Run ids were recorded says nothing about which Reference is
    held, so the identity half of the rule cannot be asked of it and only the other half is."""
    verifier = Verifier(task_id=WITHDRAWN, atoms=[], seed_run_ids=["r1"])
    assert lifecycle.live([verifier], {WITHDRAWN: {"reference_confirmed": True}}) == [verifier]
    assert lifecycle.live([verifier], {WITHDRAWN: {"reference_confirmed": False}}) == []


# --- what a derivation does with the artefacts of a withdrawn Reference ------------------

def test_the_derivation_writes_the_run_ids_of_the_reference_each_task_holds(derived):
    plan = _plan(derived)
    for task_id in (WITHDRAWN, KEPT):
        row = plan.store["task_status"][task_id]
        assert row["reference_confirmed"] and row["reference_run_ids"] == ["ref", "alt"]
        assert plan.current(task_id).seed_run_ids == row["reference_run_ids"]


def test_withdrawing_the_reference_retires_the_verifier_and_no_gate_scores_it(derived):
    before = loosen.over_strict_rows(_plan(derived).store)
    assert WITHDRAWN in before, "the held-out Run is rejected while the Verifier is live"
    _withdraw(derived)
    _derive(derived, round_number=1)
    plan = _plan(derived)
    assert not _verifier_file(derived, WITHDRAWN).is_file(), "the file leaves verifiers/"
    assert [v.task_id for v in plan.store["verifiers"]] == [KEPT]
    assert WITHDRAWN not in loosen.over_strict_rows(plan.store)
    ruling = false_rejection_gate(plan.store["verifiers"], plan.store["task_runs"], plan.store["replays"],
                                  plan.store["rerolls"], plan.store["canon_rules"], plan.store["sigs"],
                                  plan.store["task_status"])
    assert WITHDRAWN not in ruling.metrics["per_task"], "the false-rejection pool holds nothing for it"
    trusted = trusted_gate(plan.store["task_status"], plan.store["verifiers"], plan.store["probes"],
                           plan.store["history"], plan.store["refusals"], plan.store["task_runs"],
                           plan.store["replays"], plan.store["rerolls"], plan.store["canon_rules"],
                           plan.store["sigs"])
    assert WITHDRAWN not in trusted.metrics["untrusted"] and WITHDRAWN not in trusted.metrics["trusted"]


def test_the_task_reads_verifier_retired_with_the_reason_rather_than_over_strict(derived):
    _withdraw(derived)
    _derive(derived, round_number=1)
    row = lifecycle.retired_row(_plan(derived).store["task_status"][WITHDRAWN])
    assert row is not None and row["reason"] == lifecycle.REFERENCE_WITHDRAWN and row["round"] == 1
    assert row["source_run_ids"] == ["ref", "alt"]
    assert lifecycle.counts(lifecycle.retired_in_round(_plan(derived).store["task_status"], 1)) == {
        "verifiers_retired": 1,
        "verifiers_retired_by_reason": {lifecycle.REFERENCE_WITHDRAWN: 1, lifecycle.REFERENCE_REDERIVED: 0}}


def test_the_retired_verifier_stays_in_history_under_the_reason_it_was_retired_for(derived):
    was = _plan(derived).current(WITHDRAWN)
    _withdraw(derived)
    _derive(derived, round_number=1)
    versions = _plan(derived).store["history"][WITHDRAWN].versions
    last = versions[-1]
    assert last.accepted is False and last.rejected_by == [lifecycle.REFERENCE_WITHDRAWN]
    assert last.reason.startswith(lifecycle.REFERENCE_WITHDRAWN) and last.round == 1
    assert last.verifier.atoms == was.atoms, "the version it was is still readable"


def test_a_task_whose_reference_is_confirmed_again_derives_a_verifier_afresh(derived):
    _withdraw(derived)
    _derive(derived, round_number=1)
    derived.inputs["replays"][WITHDRAWN]["ref"]["confirmed"] = True
    derived.inputs["replays"][WITHDRAWN]["ref"]["reasons"] = []
    derived.inputs["rerolls"][WITHDRAWN] = [{"run_id": "alt", "path": derived.paths[f"{WITHDRAWN}/alt"],
                                             "termination_reason": "success"}]
    _derive(derived, round_number=2)
    _hold_out_the_other_path(derived)
    plan = _plan(derived)
    assert _verifier_file(derived, WITHDRAWN).is_file()
    assert plan.current(WITHDRAWN) is not None, "the derivation ran again, nothing was revived"
    reasons = [v.reason for v in plan.store["history"][WITHDRAWN].versions]
    assert any(reason.startswith(lifecycle.REFERENCE_WITHDRAWN) for reason in reasons), "history keeps it"


def test_the_task_that_kept_its_reference_is_untouched(derived):
    was = _plan(derived).current(KEPT)
    _withdraw(derived)
    _derive(derived, round_number=1)
    plan = _plan(derived)
    assert _verifier_file(derived, KEPT).is_file() and plan.current(KEPT) is not None
    assert plan.current(KEPT).seed_run_ids == was.seed_run_ids, "the same Reference, so the same lineage"
    assert lifecycle.retired_row(plan.store["task_status"][KEPT]) is None
    ruling = false_rejection_gate(plan.store["verifiers"], plan.store["task_runs"], plan.store["replays"],
                                 plan.store["rerolls"], plan.store["canon_rules"], plan.store["sigs"],
                                 plan.store["task_status"])
    assert KEPT in ruling.metrics["per_task"], "its held-out Run is still measured"
