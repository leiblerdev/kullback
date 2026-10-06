"""The Examiner reads and reviews: no tool of its session writes a Verifier or runs a Run (D320)."""

from __future__ import annotations

import asyncio

import pytest

from examiner.worlds import make_world
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import TestModel
from kullback.examiner import domain_tools as D
from kullback.examiner import session as S
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.canon import CanonRules

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
