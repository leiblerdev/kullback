"""The Examiner reads and reviews: no tool of its session writes a Verifier or runs a Run (D320)."""

from __future__ import annotations

import asyncio

import pytest

from examiner.worlds import make_world
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import domain_tools as D
from kullback.examiner import session as S
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.canon import CanonRules
from kullback.spec.schema import save_spec
from tests.spec.fixtures import spec as make_spec

# What a tool that writes a Verifier or plays a Run would be called, old names included.
WRITERS_AND_RUNNERS = {"write", "edit", "bash", "edit_verifier", "propose_verifier", "try_atoms", "probe",
                       "reroll", "run", "replay"}


def _root(tmp_path) -> ExamRoot:
    return ExamRoot(workdir=make_world(tmp_path / "w").workdir, canon_rules=CanonRules())


def test_the_examiner_session_holds_no_tool_that_writes_a_verifier_or_runs_a_run(tmp_path):
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.examiner_extension(_root(tmp_path))])
    names = set(harness.registry.names())
    assert not names & WRITERS_AND_RUNNERS
    assert set(D.TOOL_NAMES) <= names
    for tool in D.domain_tools(_root(tmp_path / "again")):
        fields = set(tool.args_model.model_fields)
        assert not fields & {"events", "count", "atoms", "verifier"}, (tool.name, fields)


def test_a_review_that_finds_nothing_is_filed_as_no_finding_with_its_reason(tmp_path):
    root = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "no_finding"]
    result = asyncio.run(tool.execute(D.NoFindingArgs(task_id="t1",
                                                      reason="every cell has a source the user said")))
    assert result.review == {"task_id": "t1", "outcome": D.NO_FINDING,
                             "reason": "every cell has a source the user said"}
    assert D.reviews_of(root.workdir) == {"t1": result.review}
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(tool.execute(D.NoFindingArgs(task_id="t1", reason="fine")))


def test_the_review_session_holds_the_finding_tool_for_the_builder(tmp_path):
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.review_extension(_root(tmp_path))])
    assert harness.registry.get("finding") is not None


def test_the_rounds_examiner_can_file_a_finding_for_the_builder(tmp_path, monkeypatch):
    import asyncio

    from kullback.spec import rounds as RD
    from tests.spec.test_roles import _written

    root, _ = _written(tmp_path)

    def file_one(harness, opening, turns, selected):
        tool = harness.registry.get("finding")
        asyncio.run(tool.execute(D.FindingArgs(kind="environment", task_id="t1", text="a tool body answers wrong",
                                               path="env/tools/update_item.py", change="answer the recorded calls")))
        return []

    monkeypatch.setattr(S, "_run_session", file_one)
    monkeypatch.setattr(RD, "_ask_confirm", lambda *args, **kwargs: {"asked": False, "added": 0})
    monkeypatch.setattr(RD, "_route_reference_fails", lambda *args, **kwargs: {})
    out = S.examine_rounds(root, ["t1"], model=TestModel([]), writer_model=TestModel([]))
    assert out["findings"] and out["findings"][0]["kind"] == "environment"
    assert out["findings"][0]["task_id"] == "t1"


def _specced_world(tmp_path, task_ids=("t1", "t2")):
    """The hand built world with one Spec file per named Task for the review round."""
    root = make_world(tmp_path / "w", tasks=len(task_ids)).workdir
    for task_id in task_ids:
        save_spec(root, make_spec().model_copy(update={"task_id": task_id}))
    return root


def _opening_of(call):
    """The review opening of one model call: the user message carrying the round ask."""
    for message in call["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        if S.REVIEW_MESSAGE in str(content or ""):
            return str(content)
    raise AssertionError("no review opening in the model call")


def test_each_task_gets_its_own_review_session(tmp_path):
    root = _specced_world(tmp_path)
    model = TestModel(["Nothing to rule here."], loop=True)
    out = S.review_specs(root, ["t2", "t1"], model=model, round_number=3)
    assert out["tasks"] == ["t1", "t2"] and out["examined"] == ["t1", "t2"]
    assert out["capped"] is False and out["capped_tasks"] == []
    assert len(model.calls) == 2
    first, second = (_opening_of(call) for call in model.calls)
    assert "t1" in first and "t2" not in first
    assert "t2" in second and "t1" not in second


def test_a_capped_session_names_its_task_and_the_others_still_run(tmp_path):
    root = _specced_world(tmp_path)
    filing = ModelReply(tool_calls=[ToolCallRequest(id="c1", name="finding", arguments={
        "kind": "environment", "task_id": "t1", "text": "a tool body answers wrong",
        "path": "env/tools/update_item.py", "change": "answer the recorded calls"})])
    model = TestModel([filing, "Nothing to rule here."])
    out = S.review_specs(root, ["t1", "t2"], model=model, round_number=1, max_turns=1)
    assert out["capped"] is True and out["capped_tasks"] == ["t1"]
    assert out["examined"] == ["t1", "t2"]
    assert [finding["task_id"] for finding in out["findings"]] == ["t1"]
    assert "t2" in _opening_of(model.calls[1]) and len(model.calls) == 2


def test_a_task_session_gets_twenty_turns(tmp_path, monkeypatch):
    root = _specced_world(tmp_path, task_ids=("t1",))
    budgets = []
    monkeypatch.setattr(S, "_run_session", lambda harness, text, turns, selected: budgets.append(turns) or [])
    S.review_specs(root, ["t1"], model=TestModel([]), round_number=1)
    assert S.REVIEW_TURNS_PER_TASK == 20 and budgets == [20]
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [S.review_extension(_root(tmp_path / "again"))])
    assert harness.registry.get("find_tools") is not None
