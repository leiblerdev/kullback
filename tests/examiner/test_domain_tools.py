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
from kullback.examiner import stage
from kullback.examiner.exam_files import ExamRoot
from kullback.gates.verifier_suite import ALT_PATH_NOT_RUN, D79_STAGES
from kullback.runner.canon import CanonRules
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
                   bus=Bus(tmp_path / "bus.jsonl", agent="examiner"), canon_rules=CanonRules())
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


def test_domain_tools_are_the_four_named_in_the_prompt_and_the_deprecated_alias(tmp_path):
    root, _, _ = _root(tmp_path)
    names = [t.name for t in D.domain_tools(root)]
    assert names == list(D.tool_names())
    assert set(names) == {"edit_verifier", "probe", "finding", "reroll", "propose_verifier"}
    text = P.tools_section()
    for name in names[:-1]:
        assert name in text, name
    assert "propose_verifier" not in text, "the prompt teaches only the new name"


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


def test_an_accepted_proposal_writes_the_new_suite_and_version_into_the_status_row_the_builder_reads(tmp_path):
    root, world, current = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(),
                                 task_status={"t1": {"verifier_passed": False, "references": 1}})
    _with_reference(world)
    write_json(world.workdir / "task_status.json", {"t1": {"verifier_passed": False, "references": 1,
                                                           "reference_confirmed": True}})
    [tool] = [t for t in D.domain_tools(root) if t.name == "edit_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    names = [record["name"] for record in result.rulings]
    assert [name for name in names if name in D79_STAGES] == list(D79_STAGES)
    assert "trusted" in names and all(record["accepted"] for record in result.rulings)
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is True and row["not_run"] == []
    assert row["checks"] == {name: True for name in D79_STAGES.values()}
    assert row["references"] == 1, "the rest of the derivation's row stands"
    assert row["verifier_version"] == result.verifier_version
    on_disk = read_json(world.workdir / "task_status.json")["t1"]
    assert on_disk["verifier_passed"] is True and on_disk["verifier_version"] == result.verifier_version
    assert on_disk["checks"] == row["checks"] and on_disk["reference_confirmed"] is True


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


def test_proposal_whose_loophole_probe_cannot_run_lands_and_the_task_stays_untrusted(tmp_path):
    """F45: a check nobody could run is no refusal of the proposal. The gates after it still rule,
    the file is kept, and the Task's row holds it untrusted with the check named and why."""
    root, world, _ = _root(tmp_path, task_status={"t1": {"verifier_passed": True}})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    by_name = {record["name"]: record for record in result.rulings}
    assert [name for name in by_name if name in D79_STAGES] == list(D79_STAGES), "every check ruled"
    assert {"loosening", "false_rejection", "trusted"} <= set(by_name), "the gates after the suite ruled"
    assert by_name["verifier_loophole"]["accepted"] is None
    assert by_name["verifier_loophole"]["not_run"].startswith("no model")
    assert by_name["trusted"]["accepted"] is None
    assert "loophole_probe_fails not run (no model" in by_name["trusted"]["line"]
    assert not any(record["accepted"] is False for record in result.rulings)
    assert root.history["t1"].versions[-1].accepted is True
    assert (world.workdir / "exam" / "verifiers" / "t1.json").is_file()
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is False and row["not_run"] == ["verifier_loophole"]
    assert row["not_run_reasons"]["loophole_probe_fails"].startswith("no model")


def test_proposal_on_a_single_reference_task_is_accepted_and_reported_not_trusted_for_the_second_path(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over(),
                           task_status={"t1": {"verifier_passed": False}})
    write_json(world.workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}]}})
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    by_name = {record["name"]: record for record in result.rulings}
    assert by_name["verifier_alt_path"]["accepted"] is None
    assert by_name["verifier_alt_path"]["not_run"] == ALT_PATH_NOT_RUN
    assert by_name["trusted"]["accepted"] is None
    assert f"second_path_passes not run ({ALT_PATH_NOT_RUN})" in by_name["trusted"]["line"]
    assert "accepted" in result.summary
    row = read_json(world.workdir / "exam" / "task_status.json")["t1"]
    assert row["verifier_passed"] is False and row["second_path_waived"] is False
    assert row["not_run"] == ["verifier_alt_path"]


def test_proposal_on_one_task_is_not_refused_for_another_tasks_untrusted_verifier(tmp_path):
    """F47: a proposal is judged by its own Task only. Task t2 is untrusted (a probe in its pool scores
    a pass), and a proposal on t1 passes the trusted and false_rejection gates."""
    root, _, current = _root(tmp_path, task_status={
        task_id: {"verifier_passed": True, "checks": {name: True for name in D79_STAGES.values()}}
        for task_id in ("t1", "t2")})
    other = current.model_copy(update={"task_id": "t2"})
    root.verifiers["t2"] = other
    root.probes["t2"] = [VF.reference_run().model_copy(update={"task_id": "t2", "run_id": "probe-t2-1"})]
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    add = {"id": "atom-new", "kind": current.atoms[0].kind, "payload": dict(current.atoms[0].target)}
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", drop=[current.atoms[0].id],
                                             add=[add])))
    by_name = {record["name"]: record for record in result.rulings}
    assert by_name["trusted"]["accepted"] is True and by_name["false_rejection"]["accepted"] is True
    assert "t2" not in " ".join(record["line"] for record in result.rulings)


def test_proposal_adding_an_atom_check_run_does_not_score_is_refused_before_any_gate(tmp_path):
    """F51: an atom whose payload kind check_run has no branch for, or whose payload is empty, would
    pass every Run; the proposal is refused with the kinds that are scored, and no version is written."""
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    for row in ({"id": "r0", "kind": "required", "payload": {"kind": "read", "tool": "look_up"}},
                {"id": "r1", "kind": "required", "payload": {}},
                {"id": "h0", "kind": "hard", "payload": {"kind": "hard"}}):
        with pytest.raises(RetryableToolError) as excinfo:
            _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[row])))
        assert f"the atom {row['id']!r} is not scored" in str(excinfo.value)
        assert "write, write_value, entity_count, question, communicate" in str(excinfo.value)
        assert "predicate_src" in str(excinfo.value)
    assert "t1" not in root.history, "refused before any gate, so no version is written"


def test_a_hard_atom_proposed_with_predicate_src_is_scored_by_the_suite(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over())
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    rule = "def check(pre_state, write_call, transcript):\n    return False\n"
    with pytest.raises(RetryableToolError) as excinfo:
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="a rule no call keeps",
                                        add=[{"id": "h0", "kind": "hard", "predicate_src": rule}])))
    assert "verifier_oracle fail" in str(excinfo.value)
    assert root.history["t1"].versions[-1].rejected_by == ["verifier_oracle"]


def test_the_propose_tool_describes_the_atoms_check_run_scores():
    description = D.ProposeArgs.model_fields["add"].description
    assert "write, write_value, entity_count, question, communicate" in description
    assert "`hard` atom needs `predicate_src`" in description


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
                                  "holds these atoms; edit, drop or add an atom, or move on to another Task")
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

    gates = validate_verifier(verifier, VF.reference_run(), empty, canon=CanonRules())
    return [{"name": g.stage, "accepted": g.passed, "line": f"{g.stage} {'pass' if g.passed else 'fail'}",
             "rows": rows_for(g, {})} for g in gates if g.stage == "verifier_empty_run"]


def test_a_refusal_on_an_empty_run_of_allowed_atoms_names_none_of_them_and_says_what_to_add():
    from kullback.gates.verifier_suite import make_atom
    from kullback.runner.records import Verifier

    verifier = Verifier(task_id="t1", atoms=[make_atom("cap", "allowed", {"kind": "entity_count", "count": 5})])
    message = str(D._refuse_proposal("t1", "2", _empty_run_records_proposal(verifier, VF.empty_run())))
    assert "the atom the empty Run passed on is cap" not in message, message
    assert "no atom of the Verifier is one a Run is checked on" in message, message
    assert ("every atom here passes an empty Run: add a question or communicate atom for what the "
            "Reference said, or a required write") in message, message


def test_a_refusal_on_an_empty_run_that_passes_a_communicate_atom_names_it_and_gives_no_advice():
    from kullback.examiner.derive import derive_verifier
    from kullback.gates.verifier_suite import atom_payload

    answering = VF.make_run("answering", [
        VF.user("What is the status of order #W123?"),
        VF.call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        VF.result(VF.ORDER, cid="c0"), VF.assistant("Order #W123 is pending."), VF.user("Thanks, bye.")])
    derived = derive_verifier(VF.TASK, answering, [], CanonRules(), write_tools=VF.WRITE_TOOLS)
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
    root = ExamRoot(workdir=workdir, reroll_model=model, canon_rules=CanonRules())
    [tool] = [t for t in D.domain_tools(root) if t.name == "reroll"]
    _run(tool.execute(D.RerollArgs(task_id="widget_task", count=1)))
    stage = budget.load_totals(workdir)["stages"]["runner"]
    assert stage["calls"] == len(model.calls) > 0 and stage["usd"] > 0


def test_an_allowance_that_covers_one_run_buys_one_run_of_a_count_of_three(tmp_path):
    from tests.examiner.test_runners import _exam_env, _priced_script

    priced = ExamRoot(workdir=_exam_env(tmp_path / "priced"), reroll_model=_priced_script(), canon_rules=CanonRules())
    [measure] = [t for t in D.domain_tools(priced) if t.name == "reroll"]
    one_run = _run(measure.execute(D.RerollArgs(task_id="widget_task", count=1))).spent_usd
    assert one_run > 0

    root = ExamRoot(workdir=_exam_env(tmp_path / "env"), reroll_model=_priced_script(),
                    allowance_remaining=one_run, canon_rules=CanonRules())
    [tool] = [t for t in D.domain_tools(root) if t.name == "reroll"]
    result = _run(tool.execute(D.RerollArgs(task_id="widget_task", count=3)))
    assert len(result.runs) == 1
    assert result.spent_usd == pytest.approx(one_run)
    assert "stopped after 1 of 3 Runs: the allowance is spent" in result.summary
    assert root.allowance_remaining <= 0


def test_a_refused_proposal_leaves_the_status_row_the_builder_reads_as_it_was(tmp_path):
    root, world, current = _root(tmp_path)
    _with_reference(world)
    write_json(world.workdir / "task_status.json", {"t1": {"verifier_passed": False}})
    before = (world.workdir / "task_status.json").read_bytes()
    [tool] = [t for t in D.domain_tools(root) if t.name == "edit_verifier"]
    drop = [a.id for a in current.atoms if a.id != "w0.reason"]
    with pytest.raises(RetryableToolError):
        _run(tool.execute(D.ProposeArgs(task_id="t1", reason="loosen", drop=drop)))
    assert (world.workdir / "task_status.json").read_bytes() == before


def test_the_ruling_reads_a_second_path_run_the_derivation_bought_for_the_task(tmp_path):
    root, world, _ = _root(tmp_path, probe_model=object(), run_probe=probe_runner_over())
    rows = dict(world.inputs["rerolls"])
    alt = [row for row in rows["t1"] if row.get("run_id") == "alt"]
    assert alt, "the world holds a second path Run"
    root.rerolls = {"t1": [row for row in rows["t1"] if row.get("run_id") != "alt"]}
    write_json(stage.extra_rerolls_path(world.workdir),
               {"t1": [{**row, "reason": stage.SECOND_PATH_REASON, "batch": 1} for row in alt]})
    _with_reference(world)
    [tool] = [t for t in D.domain_tools(root) if t.name == "edit_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    [alt_path] = [r for r in result.rulings if r["name"] == "verifier_alt_path"]
    assert alt_path["accepted"] is True, alt_path
    assert "alt" in [row.get("run_id") for row in root.rerolls["t1"]]


def _edit_tool(root):
    [tool] = [t for t in D.domain_tools(root) if t.name == "edit_verifier"]
    return tool


def test_an_edit_of_one_payload_field_writes_a_new_version_with_the_other_atoms_untouched(tmp_path):
    root, world, current = _root(tmp_path)
    count = next(a for a in current.atoms if (a.target or {}).get("kind") == "entity_count")
    others = [as_dict(a) for a in current.atoms if a.id != count.id]
    result = _run(_edit_tool(root).execute(D.ProposeArgs(
        task_id="t1", reason="widen the cap", edit=[{"id": count.id, "payload": {"count": 1000}}])))
    body = read_json(world.workdir / "exam" / "verifiers" / "t1.json")
    assert body["verifier_version"] == result.verifier_version != current.verifier_version
    [edited] = [a for a in body["atoms"] if a["id"] == count.id]
    assert edited["target"] == {**count.target, "count": 1000} and edited["kind"] == count.kind
    assert [a for a in body["atoms"] if a["id"] != count.id] == others
    assert [a["id"] for a in body["atoms"]] == [a.id for a in current.atoms], "the order stands"
    assert len(root.history["t1"].versions) == 2, "one new version per call"


def test_an_edit_of_an_unknown_atom_id_is_refused_by_name(tmp_path):
    root, world, _ = _root(tmp_path)
    with pytest.raises(RetryableToolError, match="'no-such-atom'"):
        _run(_edit_tool(root).execute(D.ProposeArgs(
            task_id="t1", reason="x", edit=[{"id": "no-such-atom", "payload": {"count": 2}}])))
    assert not (world.workdir / "exam" / "verifiers" / "t1.json").exists()


def test_an_edit_that_changes_an_atoms_kind_is_refused(tmp_path):
    root, _, current = _root(tmp_path)
    atom = current.atoms[0]
    other = "hard" if atom.kind != "hard" else "required"
    with pytest.raises(RetryableToolError, match="keeps the kind"):
        _run(_edit_tool(root).execute(D.ProposeArgs(task_id="t1", reason="x",
                                                    edit=[{"id": atom.id, "kind": other}])))


def test_the_old_tool_name_still_works_and_its_result_says_it_is_deprecated(tmp_path):
    root, _, _ = _root(tmp_path)
    [tool] = [t for t in D.domain_tools(root) if t.name == "propose_verifier"]
    result = _run(tool.execute(D.ProposeArgs(task_id="t1", reason="tighten", add=[_WIDE_CAP_PROPOSAL])))
    assert result.summary.endswith(D.DEPRECATED_ALIAS)


def test_a_note_carrying_an_atom_or_verifier_text_is_refused(tmp_path):
    with pytest.raises(ValueError, match="never an atom"):
        D.write_note(tmp_path, "t1", "outcome_not_in_state", '{"id": "w0", "kind": "required"}')
    with pytest.raises(ValueError, match="never an atom"):
        D.write_note(tmp_path, "t1", "outcome_not_in_state", "add an atom demanding the count")
    with pytest.raises(ValueError, match="not one of"):
        D.write_note(tmp_path, "t1", "some other reason", "the outcome is only said")
    assert not (tmp_path / D.NOTES_DIR).exists()


def test_an_accepted_edit_rules_the_open_note_next_to_it(tmp_path):
    root, world, _ = _root(tmp_path)
    D.write_note(world.workdir, "t1", "outcome_not_in_state", "the outcome is only said to the user.")
    assert D.open_note(world.workdir, "t1") is not None
    result = _run(_edit_tool(root).execute(D.ProposeArgs(task_id="t1", reason="agree",
                                                         add=[_WIDE_CAP_PROPOSAL])))
    assert "note on task t1 is ruled" in result.summary
    assert D.open_note(world.workdir, "t1") is None
    ruling = read_json(D.ruling_path(world.workdir, "t1"))
    assert ruling["move"] == "edit_verifier" and ruling["verdict"] == "agrees"


def test_a_probe_and_a_finding_with_note_ruling_each_rule_an_open_note(tmp_path):
    root, world, _ = _root(tmp_path)
    [probe] = [t for t in D.domain_tools(root) if t.name == "probe"]
    D.write_note(world.workdir, "t1", "needs_action_record", "the hand off leaves no row to check.")
    _run(probe.execute(D.ProbeArgs(task_id="t1", events=[])))
    assert read_json(D.ruling_path(world.workdir, "t1"))["move"] == "probe"
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
