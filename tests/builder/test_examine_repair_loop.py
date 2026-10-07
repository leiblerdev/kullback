"""The Builder's examine tool reads body edits and one line on the rulings; it never edits a Spec (D331).

The findings are the examine result the Builder's `examine_fn` seam answers with; the workdir is
tests/spec/test_roles.py's, its Verifier written from a Spec with one state item and one fact told.
"""

from __future__ import annotations

import asyncio
import json

from kullback.builder import domain_tools as D
from kullback.runner.records import Verifier
from kullback.spec import writer as W
from kullback.spec.schema import load_spec
from tests.spec.fixtures import check
from tests.spec.test_roles import _workdir, inputs, spec

TOLD = check("c2", demand={"demand": "say", "text": "slot seven"}, because="move item A1 to slot seven")
BODY = {"kind": "body", "path": "env/tools/update_item.py", "call_id": "c1", "column": "slot",
        "recorded": "seven", "replayed": "two", "why": "the body drops the slot"}


def _written(root):
    """The workdir with the Spec's Verifier written, and the Task's Intent file the text edit names."""
    _workdir(root)
    W.write_verifier(spec(checks=[TOLD]), root, inputs())
    (root / "intents").mkdir()
    (root / "intents" / "t1.json").write_text(json.dumps({"task_id": "t1", "text": load_spec(root, "t1").intent.text}))
    return root


def _verifier(root) -> Verifier:
    return Verifier.model_validate_json(W.runner_verifier_path(root, "t1").read_text())


def _examine(root, findings):
    """The Builder's examine tool over a prepared Examiner result: its rendered text and its record."""
    tools = {tool.name: tool for tool in D.domain_tools(
        workdir=root, examine_fn=lambda workdir, task_ids: {"summary": "3 findings", "findings": findings})}
    result = asyncio.run(tools["examine"].run({"task_ids": ["t1"]}))
    assert not result.is_error, result.content
    return result


def test_the_builder_reads_body_edits_and_one_line_counting_the_rulings_and_the_verifier_is_untouched(tmp_path):
    from kullback.examiner.exam_files import ExamRoot
    from kullback.examiner.rule_tool import RuleArgs, file_ruling

    root = _written(tmp_path)
    before = _verifier(root)
    file_ruling(ExamRoot(workdir=root), RuleArgs(
        task_id="t1", item="c2", kind="judge", code="unanswerable", blocking=True,
        reason="The final answer may never mention the slot at all.", fix="Name the turn the slot is told in."))
    result = _examine(root, [{"finding_id": "e1", "task_id": "t1", "kind": "fidelity", "edits": [BODY]}])
    assert "spec rulings: 1 filed, 1 open and blocking" in result.content
    assert _verifier(root) == before and load_spec(root, "t1").rulings_open == 1
