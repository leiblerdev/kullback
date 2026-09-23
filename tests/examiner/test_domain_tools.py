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
from kullback.examiner import prompt as P
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


# A proposal that changes the atoms and none of the rulings: a write cap no Run reaches, which
# no check can mutate. A proposal of no change is refused before the suite (F37).
_WIDE_CAP_PROPOSAL = {"id": "wide_cap", "kind": "allowed", "payload": {"kind": "entity_count", "count": 1000}}


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
    versions = root.history["t1"].versions
    assert [(v.verifier_version, v.accepted, v.by) for v in versions] == [("1", True, "derive"),
                                                                          ("2", True, "repair")]
    assert versions[1].parent_hash == versions[0].content_hash
    assert versions[1].reason == "tighten" and versions[1].content_hash == result.content_hash


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
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop, add=[])))
    message = str(excinfo.value)
    assert "verifier_empty_run" in message, message
    assert "rows" in message, message
    assert (world.workdir / "exam" / "verifiers" / "t1.json").read_bytes() == before
    assert root.current("t1").verifier_version == current.verifier_version
    versions = root.history["t1"].versions
    assert len(versions) == 2, "a rejected row is never dropped"
    assert versions[1].accepted is False and versions[1].rejected_by == ["verifier_empty_run"]
    written = read_json(world.workdir / "exam" / "history.json")
    assert written["t1"]["versions"][-1]["accepted"] is False

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
    names = [t.name for t in D.domain_tools(root)]
    assert names == list(D.tool_names())
    assert set(names) == {"propose_verifier", "probe", "finding", "reroll"}
    text = P.tools_section()
    for name in names:
        assert name in text, name


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
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    names = [record["name"] for record in result.rulings]
    assert [name for name in names if name in D79_STAGES] == list(D79_STAGES)
    assert "trusted" in names and all(record["accepted"] for record in result.rulings)
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is True and row["not_run"] == []
    assert row["checks"] == {name: True for name in D79_STAGES.values()}
    assert row["references"] == 1, "the rest of the derivation's row stands"
    assert not (world.workdir / "task_status.json").exists(), "the workdir's file is derive_all's"


def test_proposal_on_a_single_reference_task_with_a_waived_row_is_accepted_and_keeps_the_waiver(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(),
                           task_status={"t1": {"verifier_passed": True, "second_path_waived": True}})
    write_json(world.workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}]}})
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    assert "trusted" in [record["name"] for record in result.rulings]
    assert all(record["accepted"] for record in result.rulings)
    alt = next(record for record in result.rulings if record["name"] == "verifier_alt_path")
    assert alt["rows"][0]["second_path_waived"], "the ruling says the pass is the D199 waiver"
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is True and row["second_path_waived"] is True
    assert row["checks"] == {name: name != "second_path_passes" for name in D79_STAGES.values()}


def test_proposal_on_a_task_with_a_reference_is_refused_while_the_loophole_probe_cannot_run(tmp_path):
    root, world, _ = _root(tmp_path, task_status={"t1": {"verifier_passed": True}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(RetryableToolError, match="verifier_loophole"):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    assert root.history["t1"].versions[-1].rejected_by == ["verifier_loophole"]
    assert root.task_status["t1"] == {"verifier_passed": True}, "a refusal changes no status"
    assert not (world.workdir / "exam" / "task_status.json").exists()


def test_proposal_on_a_task_with_a_reference_is_refused_when_the_loophole_probe_passes(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(VF.reference_run()),
                           task_status={"t1": {"verifier_passed": True}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    assert "verifier_loophole fail" in str(excinfo.value)
    assert "reached the End state and scored pass" in str(excinfo.value)
    assert root.history["t1"].versions[-1].rejected_by == ["verifier_loophole"]
    assert not (world.workdir / "exam" / "task_status.json").exists(), "a refusal changes no status"


def test_an_added_row_carrying_its_own_predicate_src_is_taken_not_crashed_on(tmp_path):
    root, world, current = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    add = {"id": "atom-new", "kind": current.atoms[0].kind, "payload": dict(current.atoms[0].target),
           "predicate_src": "lambda state: True", "description": "kept"}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[add])))
    body = read_json(world.workdir / "exam" / "verifiers" / "t1.json")
    [atom] = [a for a in body["atoms"] if a["id"] == "atom-new"]
    assert atom["predicate_src"] != "lambda state: True", "the tool sets the predicate from the payload"
    assert atom["description"] == "kept"
    assert result.verifier_version == body["verifier_version"]


def test_an_added_row_with_an_unknown_field_is_refused_naming_it_and_the_corrected_atom(tmp_path):
    root, _, current = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    add = {"id": "atom-new", "kind": current.atoms[0].kind, "payload": dict(current.atoms[0].target),
           "weight": 2}
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[add])))
    message = str(excinfo.value)
    assert "weight" in message.split("The same atom")[0], message
    assert '"id": "atom-new"' in message and '"weight"' not in message.split("The same atom")[1], message


def _refused_loosening(tmp_path):
    root, world, current = _root(tmp_path)
    write_json(world.workdir / "exam" / "verifiers" / "t1.json", as_dict(current))
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    drop = [a.id for a in current.atoms if a.id != "w0.reason"]
    return root, world, tool, drop


def test_a_repeated_refused_proposal_is_answered_without_running_the_suite(tmp_path):
    root, world, tool, drop = _refused_loosening(tmp_path)
    with pytest.raises(RetryableToolError):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop)))
    rows = len(root.history["t1"].versions)
    written = read_json(world.workdir / "exam" / "history.json")
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen again", drop=drop)))
    message = str(excinfo.value)
    assert message.startswith("already refused at version"), message
    assert "verifier_empty_run" in message and "move on to another Task" in message, message
    assert len(root.history["t1"].versions) == rows, "the suite did not run, so no history row was added"
    assert read_json(world.workdir / "exam" / "history.json") == written


def test_the_fourth_refusal_of_one_task_says_stop_proposing_for_it(tmp_path):
    root, _, tool, drop = _refused_loosening(tmp_path)
    messages = []
    for _ in range(D.REFUSALS_PER_TASK + 1):
        with pytest.raises(RetryableToolError) as excinfo:
            _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop)))
        messages.append(str(excinfo.value))
    assert not any("Stop proposing" in message for message in messages[:-1]), messages
    assert "Stop proposing for this Task in this session" in messages[-1]


def test_a_refusal_names_the_atom_its_empty_run_rows_carry():
    records = [{"name": "verifier_empty_run", "accepted": False, "line": "verifier_empty_run fail",
                "rows": [{"check": "verifier_empty_run", "atom": "w0.reason", "text": "reason is given"}]},
               {"name": "verifier_alt_path", "accepted": False, "line": "verifier_alt_path fail",
                "rows": [{"check": "verifier_alt_path", "failure": "not run"}]}]
    message = str(D._refuse_proposal("t1", "2", records))
    assert "verifier_empty_run: the atom the empty Run passed on is w0.reason (reason is given)" in message
    assert "verifier_alt_path: the atom" not in message, "no atom in the rows, none named"


def _history_proposal(world):
    path = world.workdir / "exam" / "history.json"
    return read_json(path) if path.is_file() else None


def test_a_proposal_that_changes_nothing_is_refused_without_the_suite_and_leaves_the_history(tmp_path):
    root, world, current = _root(tmp_path)
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    before, rows = _history_proposal(world), dict(root.history)
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="again", drop=[], add=[])))
    assert str(excinfo.value) == (f"the proposal changes nothing: version {current.verifier_version} already "
                                  "holds these atoms; drop or add an atom, or move on to another Task")
    assert _history_proposal(world) == before and root.history == rows, "no history row for no change"
    assert not (world.workdir / "exam" / "verifiers" / "t1.json").exists(), "nothing was written"


def test_a_proposal_that_changes_nothing_counts_toward_the_refusals_that_say_stop(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    messages = []
    for _ in range(D.REFUSALS_PER_TASK + 1):
        with pytest.raises(RetryableToolError) as excinfo:
            _run(tool.execute(D.ProposeArgs(task_id="t1", reason="again")))
        messages.append(str(excinfo.value))
    assert not any("Stop proposing" in message for message in messages[:-1]), messages
    assert messages[-1].endswith("Stop proposing for this Task in this session."), messages[-1]


def _empty_run_records_proposal(verifier, empty) -> list[dict]:
    from kullback.gates.bindings import rows_for
    from kullback.gates.verifier_suite import validate_verifier

    gates = validate_verifier(verifier, VF.reference_run(), empty)
    return [{"name": g.stage, "accepted": g.passed, "line": f"{g.stage} {'pass' if g.passed else 'fail'}",
             "rows": rows_for(g, {})} for g in gates if g.stage == "verifier_empty_run"]


def test_a_refusal_on_an_empty_run_of_allowed_atoms_names_them_and_says_what_to_add():
    from kullback.gates.verifier_suite import make_atom
    from kullback.runner.records import Verifier

    verifier = Verifier(task_id="t1", atoms=[make_atom("cap", "allowed", {"kind": "entity_count", "count": 5})])
    message = str(D._refuse_proposal("t1", "2", _empty_run_records_proposal(verifier, VF.empty_run())))
    assert "verifier_empty_run: the atom the empty Run passed on is cap" in message, message
    assert ("every atom here passes an empty Run: add a question or communicate atom for what the "
            "Reference said, or a required write") in message, message


def test_a_refusal_on_an_empty_run_that_passes_a_communicate_atom_names_it_and_gives_no_advice():
    from kullback.examiner.derive import derive_verifier
    from kullback.gates.verifier_suite import atom_payload

    answering = VF.make_run("answering", [
        VF.user("What is the status of order #W123?"),
        VF.call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        VF.result(VF.ORDER, cid="c0"), VF.assistant("Order #W123 is pending."), VF.user("Thanks, bye.")])
    derived = derive_verifier(VF.TASK, answering, [], None, write_tools=VF.WRITE_TOOLS)
    [said] = [a for a in derived.atoms if atom_payload(a).get("kind") == "communicate"]
    verifier = derived.model_copy(update={"atoms": [said]})
    message = str(D._refuse_proposal("t1", "2", _empty_run_records_proposal(verifier, answering)))
    assert f"verifier_empty_run: the atom the empty Run passed on is {said.id}" in message, message
    assert "every atom here passes an empty Run" not in message, message


def test_a_reroll_from_the_tool_lands_its_model_calls_in_the_runner_stage(tmp_path):
    """The Examiner's own reroll tool prices its Runs like the derivation's runners (F29)."""
    from kullback.runner import budget
    from tests.examiner.test_runners import _exam_env, _priced_script

    workdir = _exam_env(tmp_path / "env")
    model = _priced_script()
    root = ExamRoot(workdir=workdir, reroll_model=model)
    [tool] = [t for t in D.domain_tools(root) if t.name == "reroll"]
    _run(tool.execute(D.RerollArgs(task_id="widget_task", count=1)))
    stage = budget.load_totals(workdir)["stages"]["runner"]
    assert stage["calls"] == len(model.calls) > 0 and stage["usd"] > 0
