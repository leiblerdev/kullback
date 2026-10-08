"""The Examiner's domain tools file findings, reviews and notes over its root; none writes a Verifier (D320)."""

from __future__ import annotations

import asyncio

import pytest

from examiner.worlds import make_world
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.agent.bus import Bus
from kullback.examiner import domain_tools as D
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.canon import CanonRules
from kullback.runner.records import read_json, write_json


def _root(tmp_path, **kwargs):
    world = make_world(tmp_path / "w")
    workdir = world.workdir
    (tmp_path / "v").mkdir(parents=True, exist_ok=True)
    verifier = VF.derive(tmp_path / "v")
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])
    keyword = dict(workdir=workdir, verifiers={"t1": verifier}, sigs=list(SIGS),
                   replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                   bus=Bus(tmp_path / "bus.jsonl", agent="examiner"), canon_rules=CanonRules())
    keyword.update(kwargs)
    return ExamRoot(**keyword), world, verifier


def _with_reference(world) -> None:
    """The Task's Reference as derive_all leaves it: a references.json row naming the replayed Run and a second path."""
    write_json(world.workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}, {"run_id": "alt"}]}})


# A proposal that changes the atoms and none of the rulings: a write cap no Run reaches, which
# no check can mutate. A proposal of no change is refused before the suite (F37).


def _run(coro):
    return asyncio.run(coro)


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


def test_a_note_carrying_an_atom_or_verifier_text_is_refused(tmp_path):
    with pytest.raises(ValueError, match="never an atom"):
        D.write_note(tmp_path, "t1", "outcome_not_in_state", '{"id": "w0", "kind": "required"}')
    with pytest.raises(ValueError, match="never an atom"):
        D.write_note(tmp_path, "t1", "outcome_not_in_state", "add an atom demanding the count")
    with pytest.raises(ValueError, match="not one of"):
        D.write_note(tmp_path, "t1", "some other reason", "the outcome is only said")
    assert not (tmp_path / D.NOTES_DIR).exists()


def test_the_examiners_tools_refuse_an_id_that_is_not_a_task_and_write_nothing(tmp_path):
    root, world, _ = _root(tmp_path)
    aimed = world.workdir / "replays.json"
    before = aimed.read_bytes()
    tools = {t.name: t for t in D.domain_tools(root)}
    calls = [tools["no_finding"].execute(D.NoFindingArgs(task_id="../../replays", reason="nothing is wrong here")),
             tools["finding"].execute(D.FindingArgs(task_id="../replays", kind="other", text="x"))]
    for call in calls:
        with pytest.raises(ValueError, match="unknown Task"):
            _run(call)
    with pytest.raises(ValueError, match="unknown Task"):
        D.write_note(world.workdir, "../replays", "outcome_not_in_state", "the outcome is only said.")
    assert aimed.read_bytes() == before
    assert not (world.workdir / D.NOTES_DIR).exists() and not root.findings
    assert D.known_task(world.workdir, "t1") == "t1"


def test_a_finding_with_note_ruling_rules_an_open_note(tmp_path):
    root, world, _ = _root(tmp_path)
    D.write_note(world.workdir, "t1", "fact_unavailable_to_user", "the user never learns the code.")
    assert D.open_note(world.workdir, "t1") is not None, "a new note is open again"
    [finding] = [t for t in D.domain_tools(root) if t.name == "finding"]
    _run(finding.execute(D.FindingArgs(task_id="t1", kind="other", text="the code is in the first turn",
                                       note_ruling="builder_wrong")))
    ruling = read_json(D.ruling_path(world.workdir, "t1"))
    assert ruling["move"] == "finding" and ruling["verdict"] == "builder_wrong"
    with pytest.raises(ValueError, match="no open note"):
        _run(finding.execute(D.FindingArgs(task_id="t1", kind="other", text="again",
                                           note_ruling="builder_right")))


# --- every refusal says why: the failing run, its differing call, the whole atom ---


def _intent_in_exam(root, text: str) -> None:
    (root.exam_dir / "intents").mkdir(parents=True, exist_ok=True)
    (root.exam_dir / "intents" / "t1.json").write_text(text, encoding="utf-8")


def test_a_spec_edit_on_a_finding_is_refused_naming_rulings_and_a_short_body_edit_is_refused(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "finding"]
    for kind in ("atoms", "text", "cell", "conduct", "reference"):
        with pytest.raises(ValueError, match="is a ruling"):
            _run(tool.execute(D.FindingArgs(task_id="t1", kind="other", text="x", edits=[
                {"kind": kind, "task_id": "t1", "drop": ["i0"], "why": "too tight"}])))
    with pytest.raises(ValueError, match="a body edit names"):
        _run(tool.execute(D.FindingArgs(task_id="t1", kind="other", text="x", edits=[
            {"kind": "body", "path": "env/tools/a.py", "call_id": "c1", "why": "w"}])))
    assert not root.findings
