"""The harness loosening an over-strict Verifier by itself, gated as any repair (D205).

The world is a greenhouse: one write tool that schedules a watering for a zone and one read tool
that reports what a zone got. The Reference asks the grower to confirm the length before it writes;
a held-out Run of a later round reaches the same End state without asking, and the question atom
that turns it away is the class of over-specific atom this step relaxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examiner.worlds import World, drive
from gates.verifier_fixtures import make_run, write_events_jsonl
from kullback.examiner import agent as examiner_agent
from kullback.examiner import loosen
from kullback.gates.verifier_suite import atom_payload, check_run, make_atom
from kullback.runner.records import Run, Task, ToolSig, Verifier, as_dict

TASK = "t1"
SCHEDULE = "set_zone_schedule"
READING = "zone_reading"
WRITE_TOOLS = {SCHEDULE}
SIGS = [ToolSig(name=SCHEDULE, kind="write", kind_confidence="high", unclassified=False),
        ToolSig(name=READING, kind="read", kind_confidence="high", unclassified=False)]
ZONE = {"zone_id": "Z4", "state": "idle", "litres": 12}


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


def _asked_then_wrote(run_id: str, zone: str = "Z4", opening: str = "Water zone Z4 this evening.") -> Run:
    """The recorded path: the grower is asked to confirm the length, says yes, then the write lands."""
    return make_run(run_id, [
        user(opening),
        call(READING, {"zone_id": zone}, kind="read", cid="c0"),
        result(ZONE, cid="c0"),
        says("Shall I schedule it for 20 minutes?"),
        user("yes"),
        call(SCHEDULE, {"zone_id": zone, "minutes": 20}, cid="c1"),
        result({"zone_id": zone, "state": "scheduled"}, cid="c1"),
        says(f"Zone {zone} is scheduled for 20 minutes this evening."),
    ], task_id=TASK)


def _wrote_without_asking(run_id: str, zone: str = "Z4") -> Run:
    """The same End state by another path: no confirmation, the same write."""
    return make_run(run_id, [
        user("Water zone Z4 this evening."),
        call(READING, {"zone_id": zone}, kind="read", cid="c0"),
        result(ZONE, cid="c0"),
        call(SCHEDULE, {"zone_id": zone, "minutes": 20}, cid="c1"),
        result({"zone_id": zone, "state": "scheduled"}, cid="c1"),
        says(f"Zone {zone} is scheduled for 20 minutes this evening."),
    ], task_id=TASK)


def _wrote_nothing(run_id: str = "probe") -> Run:
    return make_run(run_id, [
        user("Water zone Z4 this evening."),
        says("I have made a note of it."),
    ], task_id=TASK)


def greenhouse(root: Path) -> World:
    """The Reference and one re-run of it on disk, with the replay and re-roll rows the Runner writes."""
    workdir = root / "world"
    folder = workdir / "runs" / TASK
    folder.mkdir(parents=True)
    runs = {"ref": _asked_then_wrote("ref"),
            "alt": _asked_then_wrote("alt", opening="Please start this evening's watering for zone Z4."),
            "direct": _wrote_without_asking("direct")}
    paths = {run_id: write_events_jsonl(run, folder / f"{run_id}.jsonl") for run_id, run in runs.items()}
    inputs = {
        "tasks": [Task(id=TASK, intent="schedule this evening's watering for the zone the grower named",
                       run_ids=["ref"])],
        "sigs": list(SIGS), "constraints": [], "canon_rules": {},
        "replays": {TASK: {"ref": {"trace_id": "ref", "run_id": "ref", "confirmed": True,
                                   "path": paths["ref"], "reasons": []}}},
        "rerolls": {TASK: [{"run_id": "alt", "path": paths["alt"], "termination_reason": "success"}]},
        "intents": {}, "user_rules": {}, "traces": [], "assisted_tools": [],
    }
    return World(workdir=workdir, inputs=inputs, paths=paths)


def _probe_runner(run: Run):
    def run_probe(model, verifier):
        return run
    return run_probe


@pytest.fixture
def derived(tmp_path) -> World:
    world = greenhouse(tmp_path)
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(),
                                run_probe=_probe_runner(_wrote_nothing()))
    return world


def _plan(world: World, **kwargs):
    plan = world.plan(probe_model=object(), run_probe=_probe_runner(_wrote_nothing()), **kwargs)
    return plan, examiner_agent.examiner_harness(plan)


def _held_out(plan, world: World, run_id: str = "direct") -> None:
    """One finished re-roll of a later round, in the legitimate pool and not among the Verifier's seeds."""
    plan.extra_rerolls[TASK] = [{"run_id": run_id, "path": world.paths[run_id],
                                 "termination_reason": "success"}]
    plan.write_state()
    plan.load_state()


def _scores(verifier: Verifier, run: Run) -> bool:
    return check_run(verifier, run, None, write_tools=WRITE_TOOLS)[0]


def _rows(world: World) -> list[dict]:
    return json.loads((world.workdir / "examiner" / "auto_loosen.json").read_text(encoding="utf-8"))


# --- what a relaxation is, by kind ------------------------------------------------------

def test_a_question_atom_that_rejects_a_held_out_run_which_reached_the_reference_is_dropped(derived):
    plan, _ = _plan(derived)
    verifier, run = plan.current(TASK), _wrote_without_asking("direct")
    asking = [a.id for a in verifier.atoms if atom_payload(a).get("kind") == "question"]
    assert asking, "the derivation read the confirmation the Reference asked for as an atom"
    assert _scores(verifier, run) is False, "the question atom turns the held-out Run away"
    proposal = loosen.proposal_for(verifier, run, None, WRITE_TOOLS)
    assert proposal is not None and proposal.kinds == ["path"] and proposal.drop == asking[:1]
    assert proposal.add == [], "a path atom is dropped, not rewritten"
    assert _scores(loosen.apply(verifier, proposal.relaxations), run) is True


def test_a_write_atom_naming_one_row_is_relaxed_to_a_wrote_any_over_the_same_tool(derived):
    plan, _ = _plan(derived)
    verifier = plan.current(TASK)
    atom = next(a for a in verifier.atoms if atom_payload(a).get("kind") == "write")
    assert atom_payload(atom).get("entity"), "the derived atom names the row the Reference wrote"
    relaxed = loosen.relax(atom, WRITE_TOOLS)
    assert relaxed is not None and relaxed.kind == "write"
    payload = atom_payload(relaxed.replacement)
    assert payload["tool"] == SCHEDULE and not payload.get("entity") and not payload.get("id_field")
    assert relaxed.replacement.predicate_src.strip() == f"wrote({SCHEDULE!r})"
    # The End state it implies: any write by that tool, whichever row it acted on.
    other_row = verifier.model_copy(update={"atoms": [relaxed.replacement]})
    assert _scores(other_row, _wrote_without_asking("elsewhere", zone="Z9")) is True
    assert _scores(other_row, _wrote_nothing()) is False


def test_a_write_value_atom_is_relaxed_to_the_end_state_predicate_the_row_column_implies(derived):
    plan, _ = _plan(derived)
    verifier = plan.current(TASK)
    atom = next(a for a in verifier.atoms if atom_payload(a).get("kind") == "write_value")
    relaxed = loosen.relax(atom, WRITE_TOOLS)
    assert relaxed is not None and relaxed.kind == "write_value"
    payload = atom_payload(relaxed.replacement)
    assert payload["kind"] == "hard" and payload["derived_as"] == "shape"
    assert "value" not in payload and "raw" not in payload, "the literal is gone; the column keeps its shape"


def test_an_answer_atom_keeps_its_predicate_and_stops_rejecting_the_run_that_worded_it_otherwise():
    atom = make_atom("c0", "communicate", {"kind": "communicate", "value": "12", "text": "12 litres"})
    relaxed = loosen.relax(atom, WRITE_TOOLS)
    assert relaxed is not None and relaxed.kind == "answer"
    assert relaxed.replacement.predicate_src == atom.predicate_src, "the shape check is kept"
    assert atom_payload(relaxed.replacement)["kind"] == "communicate_reported"
    verifier = Verifier(task_id=TASK, atoms=[relaxed.replacement])
    silent = make_run("quiet", [user("how much water?"), says("The zone had what it needed.")], task_id=TASK)
    assert _scores(verifier, silent) is True, "no Run is rejected for the literal any more"
    assert _scores(Verifier(task_id=TASK, atoms=[atom]), silent) is False


def test_an_atom_no_rule_can_relax_stops_the_proposal_rather_than_being_guessed_at(derived):
    """The write cap and the compiled policy are not path atoms and not row literals; the step makes
    no proposal for them and the Task keeps its finding for the Examiner."""
    cap = next(a for a in derived_atoms(derived) if atom_payload(a).get("kind") == "entity_count")
    assert loosen.relax(cap, WRITE_TOOLS) is None


def derived_atoms(world: World):
    plan, _ = _plan(world)
    return plan.current(TASK).atoms


# --- the whole step, through the gates that rule on a repair -----------------------------

def test_the_derivation_loosens_an_over_strict_verifier_itself_and_the_gates_accept_it(derived):
    plan, harness = _plan(derived, round=2)
    _held_out(plan, derived)
    before = plan.current(TASK)
    assert _scores(before, _wrote_without_asking("direct")) is False
    result = drive(harness, "derive", {"target": "all"})
    assert result.is_error is False, result.content
    rows = _rows(derived)
    assert [(r["task_id"], r["kinds"], r["accepted"]) for r in rows] == [(TASK, ["path"], True)]
    after = plan.current(TASK)
    assert _scores(after, _wrote_without_asking("direct")) is True
    assert len(after.atoms) == len(before.atoms) - 1
    history = json.loads((derived.workdir / "examiner" / "history" / f"{TASK}.json").read_text(encoding="utf-8"))
    assert [(v["by"], v["accepted"]) for v in history["versions"]] == [("derive", True), ("auto_loosen", True)]
    assert "over-specific" in history["versions"][-1]["reason"]
    assert loosen.round_counts(rows, 2) == {
        "auto_loosen_proposed": 1, "auto_loosen_accepted": 1, "auto_loosen_rejected": 0,
        "relaxations_by_kind": {"path": 1, "read": 0, "write": 0, "write_value": 0, "answer": 0}}


def test_a_proposal_the_gates_turn_down_leaves_the_verifier_as_it_was_and_names_the_check(derived):
    """A probe already in the pool scores a pass against the relaxed Verifier, so the pool gate
    refuses the proposal; nothing on disk moves and the row says which check turned it down."""
    plan, harness = _plan(derived, round=2)
    _held_out(plan, derived)
    scored = drive(harness, "probe", {"task_id": TASK, "bug_class": "visible-test overfitting",
                                      "events": [dict(e) for e in _events(_wrote_without_asking("p1"))],
                                      "termination_reason": "success"})
    assert scored.is_error is False and scored.details["scored_pass"] is False
    before = plan.current(TASK)
    result = drive(harness, "derive", {"target": "all"}, call_id="d2")
    assert result.is_error is False, result.content
    row = _rows(derived)[-1]
    assert row["accepted"] is False and row["rejected_by"] == ["probe_pool"]
    assert row["reason"].startswith("auto_loosen_rejected") and "probe_pool" in row["reason"]
    assert plan.current(TASK).atoms == before.atoms, "the Verifier on disk is the one the gates accepted"
    assert loosen.round_counts(_rows(derived), 2)["auto_loosen_rejected"] == 1


def _events(run: Run) -> list[dict]:
    return [as_dict(event) for event in run.events]


def _told(run_id: str, answer: str) -> Run:
    """A read-only Run of a second Task: the grower is told what the zone drank, nothing is written."""
    return make_run(run_id, [
        user("How many litres did zone Z4 get yesterday?"),
        call(READING, {"zone_id": "Z4"}, kind="read", cid="c0"),
        result({"zone_id": "Z4", "litres": 12}, cid="c0"),
        says(answer),
    ], task_id=TASK)


@pytest.fixture
def read_only(tmp_path) -> World:
    """A Task whose whole Verifier is the facts its answer states, so relaxing them leaves nothing
    a Run can fail."""
    workdir = tmp_path / "world"
    folder = workdir / "runs" / TASK
    folder.mkdir(parents=True)
    runs = {"ref": _told("ref", "Zone Z4 received 12 litres yesterday."),
            "alt": _told("alt", "Yesterday zone Z4 took 12 litres."),
            "vague": _told("vague", "The zone got its full allocation yesterday.")}
    paths = {run_id: write_events_jsonl(run, folder / f"{run_id}.jsonl") for run_id, run in runs.items()}
    inputs = {
        "tasks": [Task(id=TASK, intent="say how many litres the zone got yesterday", run_ids=["ref"])],
        "sigs": list(SIGS), "constraints": [], "canon_rules": {},
        "replays": {TASK: {"ref": {"trace_id": "ref", "run_id": "ref", "confirmed": True,
                                   "path": paths["ref"], "reasons": []}}},
        "rerolls": {TASK: [{"run_id": "alt", "path": paths["alt"], "termination_reason": "success"}]},
        "intents": {}, "user_rules": {}, "traces": [], "assisted_tools": [],
    }
    world = World(workdir=workdir, inputs=inputs, paths=paths)
    examiner_agent.run_examiner(workdir, inputs=inputs, probe_model=object(),
                                run_probe=_probe_runner(_wrote_nothing()))
    return world


def test_a_proposal_that_leaves_nothing_falsifiable_is_rejected_and_the_mutation_check_is_named(read_only):
    """Relaxing the only atoms a Run can fail leaves a Verifier no mutation can flip, which is what
    the D79 suite exists to refuse; the step proposes it anyway and the gates decide, as designed."""
    plan, harness = _plan(read_only, round=2)
    _held_out(plan, read_only, run_id="vague")
    before = plan.current(TASK)
    assert drive(harness, "derive", {"target": "all"}).is_error is False
    row = _rows(read_only)[-1]
    assert row["accepted"] is False and row["kinds"] == ["answer", "answer"]
    assert "verifier_mutation" in row["rejected_by"] and row["reason"].startswith("auto_loosen_rejected")
    assert plan.current(TASK).atoms == before.atoms


# --- the caps ---------------------------------------------------------------------------

def test_one_task_gets_one_automatic_proposal_a_round_and_two_rounds_at_most():
    rows: list[dict] = []
    assert loosen.may_propose(rows, TASK, 1) is True
    rows.append({"task_id": TASK, "round": 1, "accepted": False, "kinds": ["path"]})
    assert loosen.may_propose(rows, TASK, 1) is False, "one proposal per Task per round"
    assert loosen.may_propose(rows, TASK, 2) is True
    rows.append({"task_id": TASK, "round": 2, "accepted": False, "kinds": ["write"]})
    assert loosen.may_propose(rows, TASK, 3) is False, "two rounds, then it is the Examiner's"
    assert loosen.may_propose(rows, "t2", 3) is True, "the cap is per Task"


def test_no_proposal_relaxes_more_than_two_atoms_at_once(derived):
    plan, _ = _plan(derived)
    verifier = plan.current(TASK)
    run = make_run("far", [user("Water zone Z4 this evening."),
                           call(SCHEDULE, {"zone_id": "Z9", "minutes": 5}, cid="c1"),
                           result({"zone_id": "Z9", "state": "scheduled"}, cid="c1"),
                           says("Done.")], task_id=TASK)
    proposal = loosen.proposal_for(verifier, run, None, WRITE_TOOLS)
    assert proposal is not None and len(proposal.relaxations) <= loosen.MAX_RELAXATIONS
    assert loosen.proposal_for(verifier, run, None, WRITE_TOOLS, limit=1) is not None


def test_a_task_the_step_cannot_relax_gets_no_proposal_and_nothing_is_written(derived):
    plan, harness = _plan(derived, round=2)
    result = drive(harness, "derive", {"target": "all"})
    assert result.is_error is False and _rows(derived) == [], "nothing was over-strict to loosen"
