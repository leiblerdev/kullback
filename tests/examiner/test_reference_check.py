"""The Examiner checks a Task's Reference in one view and rejects it when its End state is wrong."""

from __future__ import annotations

import asyncio

from examiner.worlds import make_world, probe_runner_over
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.examiner import domain_tools as D
from kullback.examiner import reference as R
from kullback.examiner import reference_check as RC
from kullback.examiner import stage
from kullback.examiner.exam_files import ExamRoot
from kullback.gates import trust as T
from kullback.gates.ledger import GateLedger
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
    result = RC.reject_reference(world.workdir, "t1", "ref", "the End state is not what the Intent asks", root=root)
    assert result["excluded"] == ["ref"] and result["references"] == ["rr2"] and not result["pooled"]
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


def test_rejecting_the_only_recording_pools_the_task_and_writes_no_refusal(tmp_path):
    world = make_world(tmp_path)
    root = _root(world)
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref", "alt"]}, ["ref", "alt"])})
    result = RC.reject_reference(world.workdir, "t1", "ref", "the user asked for no change", root=root)
    assert sorted(result["excluded"]) == ["alt", "ref"], "every Run of the rejected End state goes"
    assert result["references"] == [] and result["pooled"]
    assert read_json(R.pool_path(world.workdir)) == {
        "t1": {"why": "the user asked for no change", "excluded": ["alt", "ref"]}}
    assert not (world.workdir / "refusals" / "t1.json").exists()


def test_a_pooled_task_is_untrusted_for_want_of_a_reference(tmp_path):
    world = make_world(tmp_path)
    (tmp_path / "v").mkdir()
    verifier = VF.derive(tmp_path / "v")
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref", "alt"]}, ["ref", "alt"])})
    done = RC.reject_reference(world.workdir, "t1", "ref", "the user asked for no change")
    assert done["pooled"]
    ruling = T.trusted_gate({"t1": {"verifier_passed": True, "references": 1}}, [verifier],
                            {}, {}, {}, {}, world.inputs["replays"], world.inputs["rerolls"],
                            CanonRules(), list(SIGS), workdir=world.workdir)
    assert ruling.metrics["trusted"] == []
    assert ruling.metrics["untrusted"] == {"t1": T.NEEDS_REFERENCE}


def _pool_t1(world) -> None:
    """Reject the one End state the world's Runs share, so t1 waits in the pool."""
    write_json(world.workdir / "references.json", {"t1": _row({"A": ["ref", "alt"]}, ["ref", "alt"])})
    done = RC.reject_reference(world.workdir, "t1", "ref", "the user asked for no change")
    assert done["pooled"]


def test_a_new_finished_run_with_another_end_state_takes_the_task_out_of_the_pool(tmp_path):
    world = make_world(tmp_path, rerolls=("alt",))
    _pool_t1(world)
    world.inputs["rerolls"]["t1"].append({
        "run_id": "rr2", "path": world.paths["rr2"],
        "termination_reason": VF.other_reason_run().termination_reason})
    _derive(world)
    assert _references(world) == ["rr2"]
    assert R.pool(world.workdir) == {}


def test_a_new_finished_run_with_an_excluded_end_state_leaves_the_task_in_the_pool(tmp_path):
    world = make_world(tmp_path, rerolls=("alt",))
    _pool_t1(world)
    rerun = VF.alt_path_run().model_copy(update={"run_id": "alt2"})
    path = VF.write_events_jsonl(rerun, world.workdir / "runs" / "t1" / "alt2.jsonl")
    [alt_row] = [row for row in world.inputs["rerolls"]["t1"] if row["run_id"] == "alt"]
    world.inputs["rerolls"]["t1"].append(dict(alt_row, run_id="alt2", path=path))
    _derive(world)
    assert _references(world) == []
    assert R.pooled_tasks(world.workdir) == ["t1"]


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


def _derive(world) -> None:
    ctx = stage.ExamContext(world.workdir, GateLedger(world.workdir))
    stage.derive_all(ctx, world.inputs, probe_model=object(), run_probe=probe_runner_over())


def _references(world) -> list[str]:
    return [r["run_id"] for r in read_json(world.workdir / "references.json")["t1"]["references"]]


def test_a_re_derive_after_a_rejection_picks_the_other_recording_and_not_the_cached_answer(tmp_path):
    world = make_world(tmp_path, rerolls=("rr2",))
    _derive(world)
    assert _references(world) == [], "two End states and no judge: no Reference yet"
    RC.reject_reference(world.workdir, "t1", "ref", "the reason written is not the one the user gave")
    _derive(world)
    assert _references(world) == ["rr2"]
    assert read_json(world.workdir / "references.json")["t1"]["failed"]["ref"].startswith("rejected:")


def _repicked_session(tmp_path):
    """A session whose builder_right ruling on an outcome note re-picked the Reference to ref and alt."""
    world = make_world(tmp_path)
    (tmp_path / "v").mkdir()
    root = _root(world)
    root.verifiers = {"t1": VF.derive(tmp_path / "v")}
    root.probe_model, root.run_probe = object(), probe_runner_over()
    root.task_status = {"t1": {"verifier_passed": True, "references": 1}}
    write_json(world.workdir / "task_status.json", {"t1": {"verifier_passed": True, "references": 1}})
    write_json(world.workdir / "references.json", {"t1": _row(
        {"A": ["earlier"], "B": ["ref", "alt"]}, ["earlier"],
        {"ref": "survivor B was not chosen: A stood up better", "alt": "survivor B was not chosen: A stood up better"})})
    D.write_note(world.workdir, "t1", "outcome_not_in_state", "The Reference wrote one entity more than asked.")
    asyncio.run(_tool(root, "finding").execute(D.FindingArgs(
        task_id="t1", kind="reference_disagreement", text="the Reference wrote one entity more",
        note_ruling="builder_right")))
    assert _references(world) == ["ref", "alt"]
    return world, root


def _status_rows(world) -> list[dict]:
    return [read_json(world.workdir / "exam" / "task_status.json")["t1"],
            read_json(world.workdir / "task_status.json")["t1"]]


def test_a_rejection_with_no_edit_after_it_leaves_the_task_marked_and_untrusted(tmp_path):
    world, root = _repicked_session(tmp_path)
    assert all(row[RC.DERIVE_AGAIN] == RC.REPICKED and row["verifier_passed"] is False
               for row in _status_rows(world))
    assert root.task_status["t1"][RC.DERIVE_AGAIN] == RC.REPICKED
    ruling = T.trusted_gate(root.task_status, list(root.verifiers.values()), {}, {}, {}, {}, root.replays,
                            root.rerolls, root.canon_rules, root.sigs)
    assert ruling.metrics["untrusted"] == {"t1": RC.REPICKED}


