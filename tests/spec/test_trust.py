import json

import pytest

from kullback.runner.records import EndState, Event, ExpectedCell, Forbidden, ValueSource, Verifier
from kullback.spec import trust as T
from kullback.spec.compile import compile_demands, compile_spec
from kullback.spec.events import TASK_SET_ASIDE
from kullback.spec.router import ROUNDS_CAP
from kullback.spec.schema import save_spec
from tests.spec.fixtures import WRITE, WRITE_TOOLS, check, fn, reference_run
from tests.spec.fixtures import spec as base_spec

REPLAY_ID = "replay-rec1"


def spec(checks=None, gaps=()):
    """The shared Spec with its write naming the row by its canonical id, the key check_run scores on."""
    written = check(demand=dict(WRITE, entity=fn(WRITE["entity"])))
    return base_spec(checks=[written] if checks is None else checks, gaps=gaps)


START = {"items": {"A1": {"slot": "two"}, "B2": {"slot": "four"}}}
END = {"items": {"A1": {"slot": "seven"}, "B2": {"slot": "four"}}}
CODE_CHECKS = {"checks": {"plausible_wrong_fails": True, "leak_check_clean": True}}


def _cell(sourced=True):
    said = ValueSource(kind="user_turn", ptr={"turn": 0})
    return ExpectedCell(table="items", row_id="A1", field="slot", value="seven", row_source=said,
                        source=said if sourced else None)


def _verifier(the_spec=None, sourced=True) -> Verifier:
    """The compiled Verifier with one expected end state: A1 moved to slot seven, said by the user."""
    verifier = compile_spec(the_spec or spec(), WRITE_TOOLS, fn).verifier
    at = next(e.idx for e in reference_run().events if e.type == "tool_call" and e.payload["name"] == "update_item")
    return verifier.model_copy(update={"atoms": [a.model_copy(update={"target": dict(a.target, at=at)})
                                                 for a in verifier.atoms],
                                       "expected": [EndState(cells=[_cell(sourced)])]})


def _stopped(run, end=END):
    stop = Event(idx=len(run.events), type="stop", payload={"start_state": START, "end_state": end})
    return run.model_copy(update={"events": run.events + [stop]})


def _replay(run_id=REPLAY_ID, trace_id="rec1"):
    return _stopped(reference_run()).model_copy(update={"run_id": run_id, "model": "recorded", "trace_id": trace_id})


def _fresh(run_id="fresh-1", right=True):
    run = reference_run().model_copy(update={"run_id": run_id, "model": "some/model"})
    if right:
        return _stopped(run)
    return _stopped(run.model_copy(update={"events": [e for e in run.events if e.idx < 3 or e.idx == 5]}), START)


def _tier(runs, the_spec=None, rulings=(), verifier=None, references=(REPLAY_ID,)):
    the_spec = the_spec or spec(gaps=["f2"])
    return T.tier_of_task(the_spec, verifier or _verifier(the_spec), runs, rulings, canon=fn,
                          write_tools=WRITE_TOOLS, references=references)


def test_a_verifier_both_constructed_runs_fail_with_no_ruling_open_is_trusted_and_names_the_runs_scored():
    tier, row = _tier([_replay(), _fresh()])
    assert tier == "trusted" and row["reason"] is None and all(row["gates"].values())
    assert set(row["gates"]) == {"do_nothing_fails", "stray_write_fails", "no_open_ruling"}
    assert row["runs_scored"] == ["fresh-1", REPLAY_ID]
    assert row["reference_passes"] is True and row["solvable"] is True and row["fresh_passed"] == ["fresh-1"]


def test_no_fresh_pass_leaves_trust_alone_and_sets_solvable_false_where_a_fresh_run_failed():
    tier, row = _tier([_replay(), _fresh(right=False)])
    assert tier == "trusted" and row["solvable"] is False and row["fresh_runs"] == 1
    assert row["reference_passes"] is True and row["fresh_passed"] == []


def test_no_fresh_run_on_disk_leaves_solvable_unscored_and_never_false():
    tier, row = _tier([_replay()])
    assert tier == "trusted" and row["solvable"] is None and row["fresh_runs"] == 0


def test_a_failing_recording_that_is_not_the_kept_reference_moves_no_flag():
    failing = _fresh(REPLAY_ID, right=False).model_copy(update={"model": "recorded"})
    tier, row = _tier([failing, _replay("replay-rec2", "rec2"), _fresh()], references=("replay-rec2",))
    assert tier == "trusted" and row["reference_passes"] is True and row["reference"]["runs"] == ["replay-rec2"]


def test_a_spec_with_no_check_or_no_intent_fact_is_untrusted_for_no_intent():
    assert _tier([_replay(), _fresh()], the_spec=spec(checks=[])).row["reason"] == "no_intent"
    no_facts = spec(gaps=["f2"]).model_copy(update={"intent": spec().intent.model_copy(update={"facts": []})})
    assert _tier([_replay(), _fresh()], the_spec=no_facts).row["reason"] == "no_intent"


def test_grounding_is_no_gate_any_more_so_an_unlisted_fact_does_not_withhold_trust():
    assert _tier([_replay(), _fresh()], the_spec=spec()).tier == "trusted"


def test_a_verifier_that_checks_nothing_is_untrusted_for_no_intent():
    tier, row = _tier([_replay(), _fresh()], verifier=Verifier(task_id="t1"))
    assert tier == "untrusted" and row["reason"] == "no_intent"


def test_a_verifier_the_do_nothing_run_passes_is_untrusted_for_a_constructed_run_passed():
    unmoved = EndState(cells=[], allowed=[{"table": "items", "row_id": "A1"}])
    atomless = Verifier(task_id="t1", expected=[unmoved])
    tier, row = _tier([_replay(), _fresh()], verifier=atomless)
    assert tier == "untrusted" and row["reason"] == "constructed_run_passed"
    assert row["constructed"]["do_nothing"]["failed"] is False and row["constructed"]["stray_write"]["failed"] is True


def test_a_verifier_with_no_end_state_lets_the_stray_write_pass_and_is_untrusted():
    atoms_only = _verifier().model_copy(update={"expected": []})
    tier, row = _tier([_replay(), _fresh()], verifier=atoms_only)
    assert tier == "untrusted" and row["reason"] == "constructed_run_passed"
    assert row["constructed"]["do_nothing"]["failed"] is True and row["constructed"]["stray_write"]["failed"] is False


def test_the_stray_write_lands_on_a_passing_run_outside_the_declared_rows_and_fails_at_the_end_state():
    tier, row = _tier([_replay(), _fresh()])
    stray = row["constructed"]["stray_write"]
    assert stray["built"] and stray["failed"] and stray["run_id"] == f"{REPLAY_ID}.stray_write"
    assert stray["atom"].startswith("gate:sanity:")
    run = T.stray_write_run(_verifier(), _replay(), canon=fn)
    assert run.events[-1].payload["end_state"]["items"]["B2"] == {"slot": "four (stray)"}
    assert run.events[-1].payload["end_state"]["items"]["A1"] == END["items"]["A1"]


def test_a_stray_write_failing_at_a_cell_or_on_the_do_nothing_run_left_the_sanity_check_untested():
    tier, row = _tier([_fresh(right=False)], references=())
    stray = row["constructed"]["stray_write"]
    assert stray["untested"] is True and stray["failed"] is False and stray["atom"].startswith("gate:expected:cell")
    assert tier == "untrusted" and row["reason"] == "constructed_run_passed" and row["detail"] == "stray_untested"


def test_a_stray_write_on_a_passing_base_failing_at_the_sanity_check_is_tested():
    stray = _tier([_replay(), _fresh()])[1]["constructed"]["stray_write"]
    assert stray["untested"] is False and stray["failed"] is True and T.at_sanity(stray["atom"])


def test_no_stray_write_is_built_when_every_row_is_declared_and_trust_is_withheld():
    declared = _verifier().model_copy(update={"expected": [EndState(cells=[_cell()],
                                                                    allowed=[{"table": "items"}])]})
    assert T.stray_write_run(declared, _replay(), canon=fn) is None
    tier, row = _tier([_replay(), _fresh()], verifier=declared)
    assert tier == "untrusted" and row["reason"] == "constructed_run_passed"
    assert row["constructed"]["stray_write"] == {"built": False, "failed": False, "atom": None, "run_id": None}


def test_a_refusal_verifier_is_not_asked_to_fail_the_do_nothing_run():
    said = ValueSource(kind="policy", ptr={})
    refusal = Verifier(task_id="t1", expected=[EndState()],
                       forbidden=[Forbidden(kind="write", tool="update_item", table="items", row_id="A1", source=said)])
    tier, row = _tier([_replay(), _fresh()], verifier=refusal)
    assert "do_nothing" not in row["constructed"] and row["constructed"]["stray_write"]["failed"] is True
    assert tier == "trusted" and row["reference_passes"] is False


@pytest.mark.parametrize("update, rulings, tier", [
    ({"rulings_open": 1}, (), "untrusted"),
    ({"rulings_open": 1, "round": ROUNDS_CAP}, (), "untrusted"),
    ({"round": ROUNDS_CAP}, (), "trusted"),
    ({"set_aside": "no agreement"}, (), "untrusted"),
    ({}, [{"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "t1"}}}], "untrusted"),
    ({}, [{"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "other"}}}], "trusted"),
], ids=["open_ruling", "cap_rounds_open", "cap_rounds_agreed", "marked_on_spec", "bus_event", "other_task_event"])
def test_an_open_ruling_or_a_set_aside_withholds_trust_for_open_ruling(update, rulings, tier):
    the_spec = spec(gaps=["f2"]).model_copy(update=update)
    got, row = _tier([_replay(), _fresh()], the_spec=the_spec, rulings=rulings)
    assert got == tier and row["reason"] == (None if tier == "trusted" else "open_ruling")
    assert row["gates"]["no_open_ruling"] is (tier == "trusted")


def test_a_run_that_names_no_model_is_neither_a_replay_nor_fresh():
    unnamed = _fresh().model_copy(update={"model": None})
    assert T.is_replay(_replay()) and not T.is_fresh(_replay())
    assert T.is_fresh(_fresh()) and not T.is_replay(_fresh())
    assert not T.is_replay(unnamed) and not T.is_fresh(unnamed)


def _write_verifier(root, verifier, task_id="t1"):
    path = T.spec_verifier_path(root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verifier.model_dump_json())


def _write_run(folder, run):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run.run_id}.jsonl").write_text(run.model_dump_json() + "\n")


def _intent_workdir(root, runs, verifier=None):
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"}]))
    (root / "task_status.json").write_text(json.dumps({"t1": CODE_CHECKS, "t2": {}}))
    _hold_reference(root)
    for run in runs:
        _write_run(root / "runs", run)
    save_spec(root, spec(gaps=["f2"]))
    _write_verifier(root, verifier or _verifier(spec(gaps=["f2"])))


def test_a_workdir_rules_every_task_from_its_spec_and_a_task_without_one_is_untrusted(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh(right=False)])
    tiers = T.workdir_tiers(tmp_path)
    assert tiers["t2"].tier == "untrusted" and tiers["t2"].row["reason"] == "no_intent"
    assert tiers["t1"].row["runs_scored"] == ["fresh-1", REPLAY_ID]
    assert tiers["t1"].row["reference_passes"] is True and tiers["t1"].row["solvable"] is False


def test_the_written_verifier_file_is_what_the_gates_score_so_an_atom_beyond_the_checks_counts(tmp_path):
    compiled = _verifier(spec(gaps=["f2"]))
    unsaid = {"kind": "required", "demand": "say", "text": "a phrase no Run states",
              "because": "move item A1 to slot seven"}
    extra = compile_demands([unsaid], spec().intent.text, (), WRITE_TOOLS, fn).verifier.atoms
    written = compiled.model_copy(update={"atoms": compiled.atoms + [a.model_copy(update={"id": "extra"})
                                                                     for a in extra]})
    _intent_workdir(tmp_path, [_replay(), _fresh()], verifier=written)
    row = T.intent_tiers(tmp_path)["t1"].row
    assert row["reference_passes"] is False and row["fresh_passed"] == []
    assert _tier([_replay(), _fresh()], verifier=compiled).row["fresh_passed"] == ["fresh-1"]


def test_intent_tiers_score_the_run_files_under_runs_at_any_depth_and_skip_a_file_that_does_not_load(tmp_path):
    _intent_workdir(tmp_path, [_replay()])
    _write_run(tmp_path / "runs" / "nested", _fresh())
    (tmp_path / "runs" / "torn.jsonl").write_text("{not json\n")
    assert not (tmp_path / "exam" / "task_runs.json").exists()
    row = T.intent_tiers(tmp_path)["t1"].row
    assert row["runs_scored"] == ["fresh-1", REPLAY_ID] and row["fresh_passed"] == ["fresh-1"]


def test_intent_tiers_score_without_the_tools_the_schema_records_as_actions(tmp_path):
    fresh = _fresh()
    action = [Event(idx=6, type="tool_call", payload={"id": "c9", "name": "close_case", "args": {}}),
              Event(idx=7, type="tool_result", payload={"id": "c9", "result": "done"})]
    _intent_workdir(tmp_path, [fresh.model_copy(update={"events": fresh.events + action})])
    (tmp_path / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"},
                                                         {"name": "close_case", "kind": "write"}]))
    (tmp_path / "schema.json").write_text(json.dumps({"columns": [
        {"table": "actions", "name": "tool", "class": "hard", "evidence": {"actions_of": ["close_case"]}}]}))
    assert T.intent_tiers(tmp_path)["t1"].row["fresh_passed"] == ["fresh-1"]


def test_a_task_with_a_spec_and_no_verifier_file_is_untrusted_for_no_intent(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    T.spec_verifier_path(tmp_path, "t1").unlink()
    tier, row = T.intent_tiers(tmp_path)["t1"]
    assert tier == "untrusted" and row["reason"] == "no_intent" and row["why"] == "no verifier file"
    assert row["runs_scored"] == []


def test_a_set_aside_task_with_no_verifier_file_is_no_intent_when_empty_else_open_ruling(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    save_spec(tmp_path, spec(checks=[]).model_copy(update={"set_aside": "empty_spec"}))
    T.spec_verifier_path(tmp_path, "t1").unlink()
    (tmp_path / "bus.jsonl").write_text(json.dumps(
        {"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "t2"}}}) + "\n")
    save_spec(tmp_path, spec(gaps=["f2"]).model_copy(update={"task_id": "t2"}))
    tiers = T.intent_tiers(tmp_path)
    assert {t: (tiers[t].tier, tiers[t].row["reason"], tiers[t].row["why"]) for t in tiers} == \
        {"t1": ("untrusted", "no_intent", "empty_spec"), "t2": ("untrusted", "open_ruling", "no verifier file")}
    assert tiers["t1"].row["runs_scored"] == []


def test_tiers_are_computed_with_the_routers_state_file_present_and_it_is_no_task(tmp_path):
    from kullback.runner.records import write_json
    from kullback.spec.router import state_path

    _intent_workdir(tmp_path, [_replay(), _fresh()])
    write_json(state_path(tmp_path), {"last_seq": 3, "rounds": {"t1": 1}, "queue": [], "actions": []})
    tiers = T.intent_tiers(tmp_path)
    assert sorted(tiers) == ["t1", "t2"]
    assert tiers["t1"].row["runs_scored"] == ["fresh-1", REPLAY_ID]


def test_a_stored_task_sample_narrows_the_tier_report_to_it(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    T.store_tasks(tmp_path, ["t1"])
    tiers = T.intent_tiers(tmp_path)
    assert sorted(tiers) == ["t1"]
    assert tiers["t1"].row["runs_scored"] == ["fresh-1", REPLAY_ID]


def test_without_a_stored_sample_the_tier_report_covers_every_task_and_names_the_reason(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    tiers = T.intent_tiers(tmp_path)
    assert sorted(tiers) == ["t1", "t2"]
    assert all("reason" in tier.row for tier in tiers.values())
    assert tiers["t2"].row["reason"] == "no_intent"


def _recorded_elsewhere(on_reply_only=False):
    """A recording played back under a run id that carries no replay prefix."""
    run = _replay().model_copy(update={"run_id": "synth-rec1"})
    if not on_reply_only:
        return run
    events = [e.model_copy(update={"payload": {"reply": dict(e.payload["reply"], model="recorded")}})
              if e.type == "model_call" else e for e in run.events]
    return run.model_copy(update={"model": None, "events": events})


@pytest.mark.parametrize("on_reply_only", [False, True], ids=["run_model", "reply_model"])
def test_a_recorded_model_run_under_any_id_is_a_replay_and_never_makes_the_task_solvable(on_reply_only):
    tier, row = _tier([_recorded_elsewhere(on_reply_only), _fresh(right=False)], references=("synth-rec1",))
    assert row["solvable"] is False and row["reference_passes"] is True and row["fresh_passed"] == []


def _hold_reference(root, fidelity=1.0, confirmed=True, kept=True):
    """The harness's own record of rec1: kept as the Task's Reference, and its replay's fidelity."""
    kept_rows = [{"kind": "recording", "trace_id": "rec1", "run_id": REPLAY_ID}] if kept else []
    (root / "references.json").write_text(json.dumps({"t1": {"references": kept_rows}}))
    (root / "replays.json").write_text(json.dumps({"env": {"rec1": {
        "task_id": "t1", "trace_id": "rec1", "run_id": REPLAY_ID, "fidelity": fidelity, "confirmed": confirmed}}}))


def _failing_replay():
    return _fresh(REPLAY_ID, right=False).model_copy(update={"model": "recorded", "trace_id": "rec1"})


def test_a_verifier_failing_a_faithful_kept_reference_flags_it_and_hands_the_examiner_a_code_ruling(tmp_path):
    _intent_workdir(tmp_path, [_failing_replay(), _fresh()])
    _hold_reference(tmp_path)
    tiers = T.intent_tiers(tmp_path)
    tier, row = tiers["t1"]
    verifier = T.load_spec_verifier(tmp_path, "t1")
    assert tier == "trusted" and row["reference_passes"] is False
    assert row["reference"]["failed"] == REPLAY_ID
    assert row["reference"]["atom"] == f"gate:expected:cell:{verifier.expected[0].cells[0].table}.A1.slot"
    assert T.examiner_rulings(tiers) == [{"task_id": "t1", "kind": "reference_fails", "run_id": REPLAY_ID,
                                          "atom": row["reference"]["atom"], "blocking": False}]


@pytest.mark.parametrize("held", [{"fidelity": 0.8}, {"confirmed": False}, {"kept": False}],
                         ids=["unfaithful", "unconfirmed", "not_kept"])
def test_a_failed_replay_that_is_not_a_faithful_kept_reference_sets_no_flag(tmp_path, held):
    _intent_workdir(tmp_path, [_failing_replay(), _fresh()])
    _hold_reference(tmp_path, **held)
    tiers = T.intent_tiers(tmp_path)
    assert tiers["t1"].row["reference_passes"] is None and T.examiner_rulings(tiers) == []


# --- D322: one trust ruling, the code gates inside the Spec's tiers ---

def test_an_unsupported_cell_is_no_trust_gate_any_more_sourced_lives_in_the_writers_tool():
    assert _tier([_replay(), _fresh()], verifier=_verifier(spec(gaps=["f2"]), sourced=False)).tier == "trusted"


def test_a_verifier_passing_every_gate_is_trusted_with_no_probe_pool(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    assert not (tmp_path / "probes").exists()
    tier, row = T.intent_tiers(tmp_path)["t1"]
    assert tier == "trusted" and row["reason"] is None and all(row["gates"].values())


def test_the_workdir_ruling_reads_the_spec_tiers_reasons_and_flags_where_the_workdir_has_specs(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    metrics = T.workdir_ruling(tmp_path).metrics
    assert metrics["trust_tier"] == {"t1": "trusted", "t2": "untrusted"}
    assert metrics["trust_reason"] == {"t1": None, "t2": "no_intent"}
    assert metrics["trusted"] == ["t1"] and metrics["untrusted"] == {"t2": "no_intent: no spec"}
    assert metrics["reference_passes"] == {"t1": True, "t2": None} and metrics["solvable"] == {"t1": True, "t2": None}
