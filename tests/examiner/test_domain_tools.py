"""The Examiner's domain tools write proposals, probes and findings over its root."""

from __future__ import annotations

import asyncio
import json

import pytest

from examiner.worlds import events_of, make_world
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.agent.bus import Bus
from kullback.examiner import domain_tools as D
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.records import as_dict, read_json


def _root(tmp_path, **kwargs):
    world = make_world(tmp_path / "w")
    (tmp_path / "v").mkdir(parents=True, exist_ok=True)
    verifier = VF.derive(tmp_path / "v")
    keyword = dict(workdir=tmp_path, verifiers={"t1": verifier}, sigs=list(SIGS),
                   bus=Bus(tmp_path / "bus.jsonl", agent="examiner"))
    keyword.update(kwargs)
    return ExamRoot(**keyword), world, verifier

def _run(coro):
    return asyncio.run(coro)


def test_propose_verifier_writes_next_version_off_current(tmp_path):
    root, _, current = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    keep = current.atoms[0].id
    add = {"id": "atom-new", "kind": current.atoms[0].kind,
           "payload": dict(current.atoms[0].target)}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[keep], add=[add])))
    body = read_json(tmp_path / "exam" / "verifiers" / "t1.json")
    ids = [a["id"] for a in body["atoms"]]
    assert keep not in ids and "atom-new" in ids
    assert body["verifier_version"] != current.verifier_version
    assert result.content_hash and result.verifier_version == body["verifier_version"]


def test_propose_verifier_refuses_an_atom_of_the_wrong_shape(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(Exception, match="payload"):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="x",
                                        add=[{"id": "a", "kind": "required", "payload": "nope"}])))


def test_probe_scores_a_run_and_keeps_it_in_the_pool(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "probe"]
    bad = _run(tool.execute(D.ProbeArgs(task_id="t1", bug_class="wrong order",
                                        events=events_of(VF.wrong_run()))))
    assert bad.scored_pass is False and bad.failing_atom
    assert "rejected" in bad.summary
    good = _run(tool.execute(D.ProbeArgs(task_id="t1", bug_class="reference replay",
                                         events=events_of(VF.reference_run()))))
    assert good.scored_pass is True
    assert "accepted" in good.summary
    assert (tmp_path / "exam" / "probes" / "t1" / "1.json").is_file()
    assert (tmp_path / "exam" / "probes" / "t1" / "2.json").is_file()
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


def test_domain_tools_are_the_four_builder_called_verbs(tmp_path):
    root, _, _ = _root(tmp_path)
    assert [t.name for t in D.domain_tools(root)] == list(D.tool_names())
    assert json.loads(json.dumps([as_dict(v) for v in root.verifiers.values()]))
