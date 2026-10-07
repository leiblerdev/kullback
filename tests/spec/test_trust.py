import json

import pytest

from kullback.runner.records import EndState, Event, ExpectedCell, ValueSource, Verifier
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
                          write_tools=WRITE_TOOLS, references=references, status=CODE_CHECKS)


def test_a_grounded_spec_that_can_fail_with_a_fresh_pass_and_no_ruling_is_trusted_and_names_the_runs_scored():
    tier, row = _tier([_replay(), _fresh()])
    assert tier == "trusted" and row["failing"] is None and all(row["gates"].values())
    assert row["runs_scored"] == ["fresh-1", REPLAY_ID]
    assert row["fresh_passed"] == ["fresh-1"] and row["replay"] == {REPLAY_ID: True}


def test_a_replay_pass_without_a_fresh_pass_is_replay_only_and_the_replay_verdict_is_evidence_not_the_condition():
    tier, row = _tier([_replay(), _fresh(right=False)])
    assert tier == "replay_only" and row["failing"] == "no fresh Run passes" and row["fresh_runs"] == 1
    assert row["replay"] == {REPLAY_ID: True} and row["fresh_passed"] == []


def test_a_replay_pass_with_no_fresh_run_on_disk_is_replay_only_as_not_run_never_as_failed():
    tier, row = _tier([_replay()])
    assert tier == "replay_only" and row["failing"] == "not run: no fresh Run on disk" and row["fresh_runs"] == 0


def test_a_failing_recording_that_is_not_the_kept_reference_with_a_fresh_pass_is_trusted():
    failing = _fresh(REPLAY_ID, right=False).model_copy(update={"model": "recorded"})
    tier, row = _tier([failing, _replay("replay-rec2", "rec2"), _fresh()], references=("replay-rec2",))
    assert tier == "trusted" and row["replay"] == {REPLAY_ID: False, "replay-rec2": True}


def test_a_check_whose_because_quotes_nothing_is_untrusted_at_the_first_gate():
    the_spec = spec(checks=[check(because="it seemed right")], gaps=["f2"])
    tier, row = _tier([_replay(), _fresh()], the_spec=the_spec)
    assert tier == "untrusted" and row["failing"] == "grounded" and row["grounding"]["refusals"] == 1


def test_an_unlisted_fact_without_a_check_is_not_grounded():
    tier, row = _tier([_replay(), _fresh()], the_spec=spec())
    assert tier == "untrusted" and row["failing"] == "grounded"


def test_a_verifier_that_demands_nothing_cannot_fail_and_is_untrusted():
    tier, row = _tier([_replay(), _fresh()], verifier=Verifier(task_id="t1"))
    assert tier == "unconfirmed" and row["failing"] == "no expected end state"
    assert row["gates"]["can_fail"] is False


@pytest.mark.parametrize("update, rulings, tier", [
    ({"rulings_open": 1}, (), "untrusted"),
    ({"rulings_open": 1, "round": ROUNDS_CAP}, (), "set_aside"),
    ({"round": ROUNDS_CAP}, (), "trusted"),
    ({"set_aside": "no agreement"}, (), "set_aside"),
    ({}, [{"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "t1"}}}], "set_aside"),
    ({}, [{"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "other"}}}], "trusted"),
], ids=["open_ruling", "cap_rounds_open", "cap_rounds_agreed", "marked_on_spec", "bus_event", "other_task_event"])
def test_an_open_ruling_withholds_trust_and_the_routers_rule_or_a_bus_event_sets_the_task_aside(update, rulings, tier):
    the_spec = spec(gaps=["f2"]).model_copy(update=update)
    got, row = _tier([_replay(), _fresh()], the_spec=the_spec, rulings=rulings)
    assert got == tier
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
    assert tiers["t2"].tier == "untrusted" and tiers["t2"].row["failing"] == "grounded"
    assert tiers["t1"].row["runs_scored"] == ["fresh-1", REPLAY_ID]
    assert tiers["t1"].row["replay"] == {REPLAY_ID: True}


def test_the_written_verifier_file_is_what_the_gates_score_so_an_atom_beyond_the_checks_counts(tmp_path):
    compiled = _verifier(spec(gaps=["f2"]))
    unsaid = {"kind": "required", "demand": "say", "text": "a phrase no Run states",
              "because": "move item A1 to slot seven"}
    extra = compile_demands([unsaid], spec().intent.text, (), WRITE_TOOLS, fn).verifier.atoms
    written = compiled.model_copy(update={"atoms": compiled.atoms + [a.model_copy(update={"id": "extra"})
                                                                     for a in extra]})
    _intent_workdir(tmp_path, [_replay(), _fresh()], verifier=written)
    row = T.intent_tiers(tmp_path)["t1"].row
    assert row["replay"] == {REPLAY_ID: False} and row["fresh_passed"] == []
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


def test_a_task_with_a_spec_and_no_verifier_file_is_untrusted_for_want_of_the_file(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    T.spec_verifier_path(tmp_path, "t1").unlink()
    tier, row = T.intent_tiers(tmp_path)["t1"]
    assert tier == "untrusted" and row["reason"] == "no verifier file" and row["runs_scored"] == []


def test_a_set_aside_task_with_no_verifier_file_is_set_aside_with_its_reason_not_untrusted(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    save_spec(tmp_path, spec(gaps=["f2"]).model_copy(update={"set_aside": "empty_spec"}))
    T.spec_verifier_path(tmp_path, "t1").unlink()
    (tmp_path / "bus.jsonl").write_text(json.dumps(
        {"event": {"type": "custom", "name": TASK_SET_ASIDE, "payload": {"task_id": "t2"}}}) + "\n")
    save_spec(tmp_path, spec(gaps=["f2"]).model_copy(update={"task_id": "t2"}))
    tiers = T.intent_tiers(tmp_path)
    assert {t: (tiers[t].tier, tiers[t].row["reason"], tiers[t].row["set_aside"]) for t in tiers} == \
        {"t1": ("set_aside", "empty_spec", True), "t2": ("set_aside", "set aside", True)}
    assert tiers["t1"].row["runs_scored"] == [] and tiers["t1"].row["failing"] is None


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


def test_without_a_stored_sample_the_tier_report_covers_every_task_and_names_the_failing_gate(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    tiers = T.intent_tiers(tmp_path)
    assert sorted(tiers) == ["t1", "t2"]
    assert all("failing" in tier.row for tier in tiers.values())
    assert tiers["t2"].row["failing"] == "grounded"


def _recorded_elsewhere(on_reply_only=False):
    """A recording played back under a run id that carries no replay prefix."""
    run = _replay().model_copy(update={"run_id": "synth-rec1"})
    if not on_reply_only:
        return run
    events = [e.model_copy(update={"payload": {"reply": dict(e.payload["reply"], model="recorded")}})
              if e.type == "model_call" else e for e in run.events]
    return run.model_copy(update={"model": None, "events": events})


@pytest.mark.parametrize("on_reply_only", [False, True], ids=["run_model", "reply_model"])
def test_a_recorded_model_run_under_any_id_is_a_replay_and_never_an_independent_pass(on_reply_only):
    tier, row = _tier([_recorded_elsewhere(on_reply_only), _fresh(right=False)], references=("synth-rec1",))
    assert tier == "replay_only" and row["failing"] == "no fresh Run passes"
    assert row["replay"] == {"synth-rec1": True} and row["fresh_passed"] == []


def _hold_reference(root, fidelity=1.0, confirmed=True, kept=True):
    """The harness's own record of rec1: kept as the Task's Reference, and its replay's fidelity."""
    kept_rows = [{"kind": "recording", "trace_id": "rec1", "run_id": REPLAY_ID}] if kept else []
    (root / "references.json").write_text(json.dumps({"t1": {"references": kept_rows}}))
    (root / "replays.json").write_text(json.dumps({"env": {"rec1": {
        "task_id": "t1", "trace_id": "rec1", "run_id": REPLAY_ID, "fidelity": fidelity, "confirmed": confirmed}}}))


def _failing_replay():
    return _fresh(REPLAY_ID, right=False).model_copy(update={"model": "recorded", "trace_id": "rec1"})


def test_a_verifier_that_fails_a_faithful_replay_of_a_kept_reference_is_pending_and_names_the_cell(tmp_path):
    _intent_workdir(tmp_path, [_failing_replay(), _fresh()])
    _hold_reference(tmp_path)
    tier, row = T.intent_tiers(tmp_path)["t1"]
    verifier = T.load_spec_verifier(tmp_path, "t1")
    assert tier == "pending" and row["failing"].startswith("Verifier defect, ruled back to the Spec")
    assert row["gates"]["reference_replay"] is False
    assert row["reference_replay"]["failed"] == REPLAY_ID
    assert row["reference_replay"]["atom"] == f"gate:expected:cell:{verifier.expected[0].cells[0].table}.A1.slot"


@pytest.mark.parametrize("held", [{"fidelity": 0.8}, {"confirmed": False}, {"kept": False}],
                         ids=["unfaithful", "unconfirmed", "not_kept"])
def test_a_failed_replay_that_is_not_a_faithful_kept_reference_fails_no_gate_and_leaves_trust_pending(tmp_path, held):
    _intent_workdir(tmp_path, [_failing_replay(), _fresh()])
    _hold_reference(tmp_path, **held)
    tier, row = T.intent_tiers(tmp_path)["t1"]
    assert row["gates"]["reference_replay"] is True and row["reference_replay"]["failed"] is None
    assert tier == "pending" and row["failing"] == "no faithful replay of a kept Reference"


# --- D322: one trust ruling, the code gates inside the Spec's tiers ---

def test_a_verifier_with_one_unsupported_cell_is_unconfirmed_naming_the_cell_and_never_trusted():
    tier, row = _tier([_replay(), _fresh()], verifier=_verifier(spec(gaps=["f2"]), sourced=False))
    assert tier == "unconfirmed" and row["failing"] == "unsupported cell items.A1.slot"
    assert row["gates"]["sourced"] is False and row["sourced"]["unsupported"] == 1


def test_a_verifier_passing_every_gate_is_trusted_with_no_probe_pool(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    assert not (tmp_path / "probes").exists()
    tier, row = T.intent_tiers(tmp_path)["t1"]
    assert tier == "trusted" and row["failing"] is None and all(row["gates"].values())
    assert row["contradicts"] == [] and row["unasked"] == []


def test_the_workdir_ruling_reads_the_spec_tiers_where_the_workdir_has_specs(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh()])
    ruling = T.workdir_ruling(tmp_path)
    assert ruling.metrics["trust_tier"] == {"t1": "trusted", "t2": "untrusted"}
    assert ruling.metrics["trusted"] == ["t1"] and ruling.metrics["untrusted"] == {"t2": "untrusted: grounded"}


# --- the writer's disagreement with the Reference (D327): a flag on the row, never a gate -------------------

ACTIONS_SCHEMA = {"key_separator": "|", "columns": [
    {"table": "actions", "name": "tool", "evidence": {"actions_of": ["hand_over"]}}]}


def _write(check_id, tool="update_item", entity="b2"):
    return check(check_id, demand=dict(WRITE, tool=tool, entity=entity))


def _reference(end=END, calls=()):
    run = _replay()
    extra = [Event(idx=100 + i, type="tool_call", payload={"id": f"x{i}", "name": name, "args": {}})
             for i, name in enumerate(calls)]
    return _stopped(run.model_copy(update={"events": run.events[:-1] + extra}), end)


def test_the_writer_and_a_reference_that_wrote_the_demanded_row_agree_and_the_trusted_row_carries_no_flag():
    tier, row = _tier([_replay(), _fresh()])
    assert tier == "trusted" and row["writer_disagrees"] == []


def test_a_required_write_the_reference_never_called_is_flagged_by_its_check_id_and_does_not_gate():
    the_spec = spec(gaps=["f2"]).model_copy(update={"checks": [*spec().checks, _write("c2", tool="remove_item")]})
    assert T.writer_disagreement(the_spec, _reference()) == ["c2: no call to remove_item"]
    tier, row = _tier([_replay(), _fresh()], the_spec=the_spec, verifier=_verifier())
    assert tier == "trusted" and row["writer_disagrees"] == ["c2: no call to remove_item"]


def test_a_required_write_on_a_row_the_reference_left_unchanged_is_flagged_and_an_allowed_one_is_not():
    required = spec().model_copy(update={"checks": [*spec().checks, _write("c2")]})
    allowed = spec().model_copy(update={"checks": [*spec().checks, check("c2", kind="allowed",
                                                                         demand=dict(WRITE, entity="b2"))]})
    assert T.writer_disagreement(required, _reference()) == ["c2: its row is in no diffed row"]
    assert T.writer_disagreement(allowed, _reference()) == []


def test_a_diffed_row_no_write_demand_names_is_not_flagged():
    end = {"items": {"A1": {"slot": "seven"}, "B2": {"slot": "nine"}}}
    assert T.writer_disagreement(spec(), _reference(end)) == []


def test_action_table_rows_never_count_and_an_action_demand_is_read_by_its_call_alone():
    end = dict(END, actions={"0": {"tool": "hand_over"}})
    handed = spec().model_copy(update={"checks": [*spec().checks, _write("c2", tool="hand_over", entity="desk")]})
    assert T.writer_disagreement(handed, _reference(end, calls=["hand_over"]), ACTIONS_SCHEMA) == []
    assert T.writer_disagreement(handed, _reference(end), ACTIONS_SCHEMA) == ["c2: no call to hand_over"]


def test_without_a_reference_the_writer_cannot_disagree():
    assert T.writer_disagreement(spec(), None) == []
