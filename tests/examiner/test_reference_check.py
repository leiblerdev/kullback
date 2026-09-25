"""The Examiner checks a Task's Reference in one view and rejects it when its End state is wrong."""

from __future__ import annotations

import asyncio

from examiner.worlds import make_world
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.examiner import domain_tools as D
from kullback.examiner import reference as R
from kullback.examiner import reference_check as RC
from kullback.examiner.exam_files import ExamRoot
from kullback.gates import trust as T
from kullback.runner.canon import CanonRules
from kullback.runner.records import read_json, write_json

READ_TOOL = next(sig.name for sig in SIGS if sig.kind == "read")


def _root(world) -> ExamRoot:
    write_json(world.workdir / "replays.json", world.inputs["replays"])
    write_json(world.workdir / "rerolls.json", world.inputs["rerolls"])
    return ExamRoot(workdir=world.workdir, sigs=list(SIGS), replays=world.inputs["replays"],
                    rerolls=world.inputs["rerolls"], canon_rules=CanonRules())


def _row(groups: dict[str, list[str]], references: list[str], failed: dict | None = None) -> dict:
    """A references.json row the way derive_all writes it: groups by label, recordings, failures."""
    runs = [run_id for members in groups.values() for run_id in members]
    return {"references": [{"run_id": r, "kind": R.RECORDING} for r in references],
            "recordings": [{"run_id": r, "kind": R.RECORDING} for r in runs],
            "groups": [{"label": label, "runs": members} for label, members in groups.items()],
            "failed": dict(failed or {}), "reason": None}


def _tool(root: ExamRoot, name: str):
    [tool] = [t for t in D.domain_tools(root) if t.name == name]
    return tool


def test_rejecting_one_of_two_recordings_repicks_the_other_as_reference(tmp_path):
    world = make_world(tmp_path)
    root = _root(world)
    write_json(world.workdir / "references.json", {"t1": _row(
        {"A": ["ref"], "B": ["rr2"]}, ["ref"], {"rr2": "survivor B was not chosen: A stood up better"})})
    result = asyncio.run(_tool(root, "reject_reference").execute(
        RC.RejectReferenceArgs(task_id="t1", run_id="ref", why="the End state is not what the Intent asks")))
    assert result.excluded == ["ref"] and result.references == ["rr2"] and not result.refused
    row = read_json(world.workdir / "references.json")["t1"]
    assert [r["run_id"] for r in row["references"]] == ["rr2"]
    assert row["failed"]["ref"].startswith("rejected:")
    [excluded] = read_json(RC.exclusions_path(world.workdir))["t1"]
    assert excluded["run_id"] == "ref" and excluded["why"]
    assert not (world.workdir / "refusals" / "t1.json").exists()


def test_reference_choice_skips_an_excluded_recording_and_picks_among_the_rest():
    first = R.Recording(run_id="a", path="", end_state=(("w", "e1", ()),))
    second = R.Recording(run_id="b", path="", end_state=(("w", "e2", ()),))
    out = R.confirm([first, second], excluded={"a": "wrote the wrong entity"})
    assert [r.run_id for r in out.references] == ["b"]
    assert out.failed == {"a": "rejected: wrote the wrong entity"}


def test_rejecting_the_only_recording_refuses_the_task_and_the_refusal_stands_though_a_run_finished(tmp_path):
    world = make_world(tmp_path)
    root = _root(world)
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref", "alt"]}, ["ref", "alt"])})
    result = asyncio.run(_tool(root, "reject_reference").execute(
        RC.RejectReferenceArgs(task_id="t1", run_id="ref", why="the user asked for no change")))
    assert sorted(result.excluded) == ["alt", "ref"], "every Run of the rejected End state goes"
    assert result.references == [] and result.refused
    refusal = read_json(world.workdir / "refusals" / "t1.json")
    assert refusal["reason"] == RC.REFERENCE_WRONG
    ruling = T.refuse_gate({"t1": refusal}, world.inputs["replays"], world.inputs["rerolls"])
    assert T.finished_runs("t1", world.inputs["replays"], world.inputs["rerolls"])
    assert ruling.passed and ruling.metrics["refused"] == ["t1"]


def test_a_builder_right_ruling_on_an_outcome_note_excludes_the_reference(tmp_path):
    world = make_world(tmp_path)
    write_json(world.workdir / "references.json", {"t1": _row(
        {"A": ["ref"], "B": ["rr2"]}, ["ref"], {"rr2": "survivor B was not chosen: A stood up better"})})
    D.write_note(world.workdir, "t1", "intent_contradicts_reference", "The user asked for another reason.")
    ruling = D.rule_note(world.workdir, "t1", "finding", "builder_right", "the Reference wrote another reason")
    assert ruling["reference_rejected"]["excluded"] == ["ref"]
    assert RC.excluded_runs(world.workdir, "t1") == {"ref": "the Reference wrote another reason"}
    assert [r["run_id"] for r in read_json(world.workdir / "references.json")["t1"]["references"]] == ["rr2"]
    assert D.read_ruling(world.workdir, "t1")["verdict"] == "builder_right", "the ruling file is kept"


def test_a_builder_wrong_ruling_leaves_the_reference_in_place(tmp_path):
    world = make_world(tmp_path)
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref"]}, ["ref"])})
    D.write_note(world.workdir, "t1", "outcome_not_in_state", "The outcome is not visible in the state.")
    ruling = D.rule_note(world.workdir, "t1", "probe", "builder_wrong", "the probe was rejected")
    assert "reference_rejected" not in ruling
    assert RC.excluded_runs(world.workdir, "t1") == {}


def test_check_reference_on_a_reference_that_wrote_nothing_says_so(tmp_path):
    world = make_world(tmp_path)
    quiet = VF.make_run("quiet", [VF.user("Please look this up."),
                                  VF.call(READ_TOOL, {"id": "1"}, kind="read", cid="c0"),
                                  VF.result({"state": "open"}, cid="c0"),
                                  VF.assistant("I looked it up; nothing else to do.")])
    path = VF.write_events_jsonl(quiet, world.workdir / "runs" / "t1" / "quiet.jsonl")
    world.inputs["replays"]["t1"]["quiet"] = {"trace_id": "quiet", "run_id": "quiet", "confirmed": True,
                                              "path": path, "reasons": []}
    root = _root(world)
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["quiet"]}, ["quiet"])})
    result = asyncio.run(_tool(root, "check_reference").execute(RC.CheckReferenceArgs(task_id="t1")))
    view = result.view
    assert view["reference"] == "quiet" and view["wrote_nothing"] and view["end_equals_start"]
    assert view["writes"] == [] and view["recordings_agree"]
    assert view["rerolls"] == {RC.REACHED: [], RC.WROTE_ELSE: ["alt"], RC.WROTE_NOTHING: []}
    assert view["intent"] == VF.TASK.intent
    assert "wrote nothing" in result.summary


def test_check_reference_lists_each_write_with_the_reads_before_it(tmp_path):
    world = make_world(tmp_path)
    root = _root(world)
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref", "alt"]}, ["ref", "alt"])})
    view = RC.reference_view(root, "t1")
    [write] = view["writes"]
    assert write["reads_before"] == [READ_TOOL] and not view["wrote_nothing"]
    assert view["rerolls"][RC.REACHED] == ["alt"]
