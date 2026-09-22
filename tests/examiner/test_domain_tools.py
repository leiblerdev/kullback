"""The Examiner's domain tools write proposals, probes and findings over its root."""

from __future__ import annotations

import asyncio

import pytest

from examiner.worlds import events_of, make_world, probe_runner_over
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.agent.bus import Bus
from kullback.agent.tools import RetryableToolError
from kullback.examiner import domain_tools as D
from kullback.examiner.exam_files import ExamRoot
from kullback.gates.verifier_suite import D79_STAGES
from kullback.runner.records import as_dict, read_json, write_json


def _root(tmp_path, **kwargs):
    world = make_world(tmp_path / "w")
    workdir = world.workdir
    (tmp_path / "v").mkdir(parents=True, exist_ok=True)
    verifier = VF.derive(tmp_path / "v")
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])
    keyword = dict(workdir=workdir, verifiers={"t1": verifier}, sigs=list(SIGS),
                   replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                   bus=Bus(tmp_path / "bus.jsonl", agent="examiner"))
    keyword.update(kwargs)
    return ExamRoot(**keyword), world, verifier


def _with_reference(world) -> None:
    """The Task's Reference as derive_all leaves it: a references.json row naming the replayed Run and a second path."""
    write_json(world.workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}, {"run_id": "alt"}]}})


def _run(coro):
    return asyncio.run(coro)


def test_accepted_proposal_returns_rulings_and_keeps_the_file(tmp_path):
    root, world, current = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    keep = current.atoms[0].id
    add = {"id": "atom-new", "kind": current.atoms[0].kind,
           "payload": dict(current.atoms[0].target)}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[keep], add=[add])))
    assert result.rulings, "the gates rule on the write"
    assert all(record["accepted"] for record in result.rulings)
    assert all(set(record) == {"name", "accepted", "line", "rows"} for record in result.rulings)
    body = read_json(world.workdir / "exam" / "verifiers" / "t1.json")
    ids = [a["id"] for a in body["atoms"]]
    assert keep not in ids and "atom-new" in ids
    assert body["verifier_version"] != current.verifier_version
    assert result.content_hash and result.verifier_version == body["verifier_version"]
    assert root.current("t1").verifier_version == body["verifier_version"]


def test_propose_verifier_refuses_an_atom_of_the_wrong_shape(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(Exception, match="payload"):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="x",
                                        add=[{"id": "a", "kind": "required", "payload": "nope"}])))
def test_refused_proposal_restores_the_previous_file_and_names_rows(tmp_path):
    root, world, current = _root(tmp_path)
    write_json(world.workdir / "exam" / "verifiers" / "t1.json", as_dict(current))
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    drop = [a.id for a in current.atoms if a.id != "w0.reason"]
    before = (world.workdir / "exam" / "verifiers" / "t1.json").read_bytes()
    with pytest.raises(Exception) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop, add=[])))
    message = str(excinfo.value)
    assert "verifier_empty_run" in message, message
    assert "rows" in message, message
    assert (world.workdir / "exam" / "verifiers" / "t1.json").read_bytes() == before
    assert root.current("t1").verifier_version == current.verifier_version

def test_probe_scores_a_run_and_keeps_it_in_the_pool(tmp_path):
    root, world, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "probe"]
    bad = _run(tool.execute(D.ProbeArgs(task_id="t1", bug_class="wrong order",
                                        events=events_of(VF.wrong_run()))))
    assert bad.scored_pass is False and bad.failing_atom
    assert "rejected" in bad.summary
    good = _run(tool.execute(D.ProbeArgs(task_id="t1", bug_class="reference replay",
                                         events=events_of(VF.reference_run()))))
    assert good.scored_pass is True
    assert "accepted" in good.summary
    assert (world.workdir / "exam" / "probes" / "t1" / "1.json").is_file()
    assert (world.workdir / "exam" / "probes" / "t1" / "2.json").is_file()
    assert good.pool_size == 2


def test_finding_files_publishes_and_refuses_a_duplicate_with_its_id(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "finding"]
    first = _run(tool.execute(D.FindingArgs(task_id="t1", kind="fidelity", text="replays differ",
                                            rows=[{"run_id": "ref"}], path="env/tools/x.py",
                                            change="answer the recorded call")))
    lines = (tmp_path / "bus.jsonl").read_text().splitlines()
    assert any(first.finding_id in line for line in lines)
    with pytest.raises(ValueError, match=first.finding_id):
        _run(tool.execute(D.FindingArgs(task_id="t1", kind="fidelity", text="replays differ again",
                                        rows=[{"run_id": "ref"}], path="env/tools/x.py",
                                        change="answer the recorded call")))
    assert first.finding["kind"] == "fidelity"


def test_finding_refuses_an_unknown_kind(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "finding"]
    with pytest.raises(ValueError, match="not a finding kind"):
        _run(tool.execute(D.FindingArgs(kind="replay_reference", text="x")))


def test_reroll_is_refused_when_the_allowance_is_spent(tmp_path):
    root, _, _ = _root(tmp_path, allowance_remaining=0.0)
    [tool] = [t for t in D.domain_tools(root) if t.name == "reroll"]
    with pytest.raises(RuntimeError, match="allowance is spent"):
        _run(tool.execute(D.RerollArgs(task_id="t1", count=1)))


def test_reroll_without_a_model_is_refused_before_touching_the_runner(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "reroll"]
    with pytest.raises(ValueError, match="no reroll model was given"):
        _run(tool.execute(D.RerollArgs(task_id="t1", count=1)))


def test_domain_tools_are_exactly_the_four_named_in_the_prompt(tmp_path):
    root, _, _ = _root(tmp_path)
    assert [t.name for t in D.domain_tools(root)] == list(D.tool_names())

def test_accepted_proposal_appends_version_2_with_parent_hash_of_version_1(tmp_path):
    root, world, current = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    keep = current.atoms[0].id
    add = {"id": "atom-new", "kind": current.atoms[0].kind,
           "payload": dict(current.atoms[0].target)}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[keep], add=[add])))
    assert all(record["accepted"] for record in result.rulings)
    versions = root.history["t1"].versions
    assert [(v.verifier_version, v.accepted, v.by) for v in versions] == [("1", True, "derive"),
                                                                          ("2", True, "repair")]
    assert versions[1].parent_hash == versions[0].content_hash
    assert versions[1].reason == "tighten" and versions[1].content_hash == result.content_hash


def test_refused_proposal_keeps_a_rejected_row_naming_the_gate(tmp_path):
    root, world, current = _root(tmp_path)
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    drop = [a.id for a in current.atoms if a.id != "w0.reason"]
    with pytest.raises(RetryableToolError):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop, add=[])))
    versions = root.history["t1"].versions
    assert len(versions) == 2, "a rejected row is never dropped"
    assert versions[1].accepted is False and versions[1].rejected_by == ["verifier_empty_run"]
    written = read_json(world.workdir / "exam" / "history.json")
    assert written["t1"]["versions"][-1]["accepted"] is False
    assert root.current("t1").verifier_version == current.verifier_version


def test_accepted_proposal_rules_trusted_when_status_and_history_are_present(tmp_path):
    root, world, current = _root(
        tmp_path, task_status={"t1": {"verifier_passed": True,
                                      "checks": {name: True for name in D79_STAGES.values()}}})
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    keep = current.atoms[0].id
    add = {"id": "atom-new", "kind": current.atoms[0].kind,
           "payload": dict(current.atoms[0].target)}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[keep], add=[add])))
    assert "trusted" in [record["name"] for record in result.rulings]
    assert all(record["accepted"] for record in result.rulings)
    verifiers_dir = world.workdir / "exam" / "verifiers"
    assert (verifiers_dir.parent / "history.json").is_file()
    assert (verifiers_dir.parent / "task_runs.json").is_file()


def test_accepted_proposal_on_a_task_with_a_reference_carries_the_suite_and_keeps_its_status(tmp_path):
    root, world, current = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(),
                                 task_status={"t1": {"verifier_passed": False, "references": 1}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[], add=[])))
    names = [record["name"] for record in result.rulings]
    assert [name for name in names if name in D79_STAGES] == list(D79_STAGES)
    assert "trusted" in names and all(record["accepted"] for record in result.rulings)
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is True and row["not_run"] == []
    assert row["checks"] == {name: True for name in D79_STAGES.values()}
    assert row["references"] == 1, "the rest of the derivation's row stands"
    assert not (world.workdir / "task_status.json").exists(), "the workdir's file is derive_all's"


def test_proposal_on_a_task_with_a_reference_is_refused_while_the_loophole_probe_cannot_run(tmp_path):
    root, world, _ = _root(tmp_path, task_status={"t1": {"verifier_passed": True}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(RetryableToolError, match="verifier_loophole"):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[], add=[])))
    assert root.history["t1"].versions[-1].rejected_by == ["verifier_loophole"]
    assert root.task_status["t1"] == {"verifier_passed": True}, "a refusal changes no status"
    assert not (world.workdir / "exam" / "task_status.json").exists()


def test_proposal_on_a_task_with_a_reference_is_refused_when_the_loophole_probe_passes(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(VF.reference_run()),
                           task_status={"t1": {"verifier_passed": True}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[], add=[])))
    assert "verifier_loophole fail" in str(excinfo.value)
    assert "reached the End state and scored pass" in str(excinfo.value)
    assert root.history["t1"].versions[-1].rejected_by == ["verifier_loophole"]
    assert not (world.workdir / "exam" / "task_status.json").exists(), "a refusal changes no status"
