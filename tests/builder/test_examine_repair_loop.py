"""An Examiner review reaches a rewritten Verifier through the Builder's examine tool, by code (D320).

The findings are the examine result the Builder's `examine_fn` seam answers with; the workdir is
tests/spec/test_roles.py's, its Verifier written from a Spec with one write and one fact told.
"""

from __future__ import annotations

import asyncio
import json

from kullback.builder import domain_tools as D
from kullback.runner.records import Verifier
from kullback.spec import review as RV
from kullback.spec import writer as W
from kullback.spec.schema import load_spec
from tests.spec.fixtures import check, spec
from tests.spec.test_roles import _workdir
from tests.spec.test_writer import inputs

TOLD = check("c2", demand={"demand": "say", "text": "slot seven"}, because="move item A1 to slot seven")
HEARD = {"id": "said_item", "kind": "required", "payload": {"kind": "communicate", "text": "Item A1",
                                                             "id": "said_item"}}
BODY = {"kind": "body", "path": "env/tools/update_item.py", "call_id": "c1", "column": "slot",
        "recorded": "seven", "replayed": "two", "why": "the body drops the slot"}


def _written(root):
    """The workdir with the Spec's Verifier written, and the Task's Intent file the text edit names."""
    _workdir(root)
    W.write_verifier(spec(checks=[check(), TOLD]), root, inputs())
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


def test_an_atoms_and_a_text_finding_rewrite_the_verifier_and_the_builder_reads_body_edits_and_one_line(tmp_path):
    root = _written(tmp_path)
    [told] = [atom.id for atom in _verifier(root).atoms]
    atoms = {"kind": "atoms", "drop": [told], "add": [HEARD], "why": "the user asked to hear the item"}
    text = {"kind": "text", "path": "intents/t1.json", "where": "urgent", "replace": "pressing"}
    result = _examine(root, [{"task_id": "t1", "kind": "other", "text": "atoms", "edits": [atoms]},
                             {"task_id": "t1", "kind": "other", "text": "words", "edits": [text]},
                             {"task_id": "t1", "kind": "fidelity", "text": "body", "edits": [BODY]}])

    assert [finding["edits"] for finding in result.details["findings"]] == [[BODY]]
    saved = load_spec(root, "t1")
    row = RV.gate_row(root, saved)
    assert RV.rounds_line([{"applied": 2, "passed": RV.gates_pass(row), "pending": False}]) in result.content
    assert saved.round == 1 and saved.atom_edits == [{"drop": [told], "add": [HEARD], "why": atoms["why"]}]
    verifier = _verifier(root)
    assert verifier.verifier_version == "spec1"
    assert {atom.id for atom in verifier.atoms} == {"said_item"}
    assert "pressing" in (root / "intents" / "t1.json").read_text() and "urgent" not in saved.intent.text
    # The fixture's Reference replay carries no fidelity, so the sourced gate is the first to fail.
    assert next(gate for gate, passed in row["gates"].items() if not passed) == "sourced"
    assert row["failing"] == "no faithful replay of a kept Reference"


def test_an_atoms_finding_adding_a_write_predicate_is_refused_and_the_verifier_is_unchanged(tmp_path):
    root = _written(tmp_path)
    before = _verifier(root)
    coded = dict(HEARD, id="coded", predicate_src="lambda run: run.wrote('A1')")
    result = _examine(root, [{"task_id": "t1", "kind": "other", "text": "atoms",
                              "edits": [{"kind": "atoms", "add": [coded], "why": "check the write"}]}])
    assert result.details["findings"] == []
    assert "spec repairs: 0 edits on 1 Tasks" in result.content
    assert load_spec(root, "t1").atom_edits == []
    after = _verifier(root)
    assert after.atoms == before.atoms and after.expected == before.expected
    out = RV.apply_review(root, "t1", [{"kind": "atoms", "add": [coded]}])
    assert out["refused"] == [W.WRITE_PREDICATE]
