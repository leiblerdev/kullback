"""try_atoms rules on an edit the way edit_verifier does and writes nothing."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from examiner.worlds import make_world, probe_runner_over
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.agent.bus import Bus
from kullback.agent.tools import RetryableToolError
from kullback.examiner import domain_tools as D
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.canon import CanonRules
from kullback.runner.records import as_dict, write_json

# A write cap no Run reaches: it changes the atoms and none of the rulings.
_WIDE_CAP = {"id": "wide_cap", "kind": "allowed", "payload": {"kind": "entity_count", "count": 1000}}


def _root(tmp_path, **kwargs):
    world = make_world(tmp_path / "w")
    workdir = world.workdir
    (tmp_path / "v").mkdir(parents=True, exist_ok=True)
    verifier = VF.derive(tmp_path / "v")
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])
    write_json(workdir / "exam" / "verifiers" / "t1.json", as_dict(verifier))
    write_json(workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}, {"run_id": "alt"}]}})
    keyword = dict(workdir=workdir, verifiers={"t1": verifier}, sigs=list(SIGS),
                   replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                   bus=Bus(tmp_path / "bus.jsonl", agent="examiner"), canon_rules=CanonRules())
    keyword.update(kwargs)
    root = ExamRoot(**keyword)
    tools = {tool.name: tool for tool in D.domain_tools(root)}
    return root, world, verifier, tools


def _run(coro):
    return asyncio.run(coro)


def _tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _loosening(verifier) -> D.ProposeArgs:
    """An edit that leaves only an atom an empty Run passes, which edit_verifier refuses."""
    return D.ProposeArgs(task_id="t1", reason="loosen", drop=[a.id for a in verifier.atoms if a.id != "w0.reason"])


def test_a_dry_run_of_an_edit_edit_verifier_refuses_returns_the_same_failing_ruling(tmp_path):
    _, _, verifier, tools = _root(tmp_path)
    tried = _run(tools["try_atoms"].execute(_loosening(verifier)))
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tools["edit_verifier"].execute(_loosening(verifier)))
    assert tried.accepted is False
    failed = [record for record in tried.rulings if record["accepted"] is False]
    assert [record["name"] for record in failed] == ["verifier_empty_run"]
    assert failed[0]["line"] in str(excinfo.value)


def test_a_dry_run_names_the_added_atom_its_failing_ruling_turns_on(tmp_path):
    _, _, _, tools = _root(tmp_path)
    unsaid = {"id": "c_unsaid", "kind": "communicate",
              "payload": {"kind": "communicate", "value": '"999"', "text": "999", "field": "total"}}
    tried = _run(tools["try_atoms"].execute(D.ProposeArgs(task_id="t1", reason="say it", add=[unsaid])))
    assert tried.accepted is False
    assert tried.failing_atoms == ["c_unsaid"] and "c_unsaid" in tried.summary, tried.summary


def test_a_dry_run_of_an_edit_that_would_pass_returns_accepted(tmp_path):
    _, _, _, tools = _root(tmp_path, task_status={"t1": {"verifier_passed": True}})
    tried = _run(tools["try_atoms"].execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP])))
    assert tried.accepted is True, tried.summary
    assert tried.rulings and not tried.failing_atoms


def test_a_dry_run_whose_only_open_check_is_the_loophole_probe_is_not_decided(tmp_path):
    _, _, _, tools = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(VF.reference_run()),
                           task_status={"t1": {"verifier_passed": True}})
    tried = _run(tools["try_atoms"].execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP])))
    loophole = next(record for record in tried.rulings if record["name"] == "verifier_loophole")
    assert loophole["accepted"] is None and loophole["not_run"] == D.DRY_RUN_NOT_RUN
    assert tried.accepted == "not decided"


def test_a_dry_run_leaves_the_workdir_byte_identical_and_the_root_as_it_was(tmp_path):
    root, world, verifier, tools = _root(tmp_path)
    before = _tree_hash(world.workdir)
    history, rerolls = dict(root.history), root.rerolls
    _run(tools["try_atoms"].execute(_loosening(verifier)))
    _run(tools["try_atoms"].execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP])))
    assert _tree_hash(world.workdir) == before
    assert root.verifiers["t1"] is verifier and root.history == history and root.rerolls is rerolls


def test_dry_runs_do_not_count_toward_the_refusals_that_say_stop(tmp_path):
    _, _, verifier, tools = _root(tmp_path)
    for _ in range(D.REFUSALS_PER_TASK + 1):
        _run(tools["try_atoms"].execute(_loosening(verifier)))
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tools["edit_verifier"].execute(_loosening(verifier)))
    message = str(excinfo.value)
    assert not message.startswith("already refused"), "the suite ran: the dry runs were no refusal"
    assert "Stop proposing" not in message and "try_atoms" in message
