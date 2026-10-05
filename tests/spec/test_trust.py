import json

import pytest

from gates.examiner_fixtures import TASK
from kullback.gates import trust as gates_trust
from kullback.runner.records import Verifier
from kullback.spec import trust as T
from kullback.spec.compile import compile_demands, compile_spec
from kullback.spec.events import TASK_SET_ASIDE
from kullback.spec.router import ROUNDS_CAP
from kullback.spec.schema import save_spec
from tests.gates.test_trust import _world
from tests.spec.fixtures import WRITE, WRITE_TOOLS, check, fn, reference_run
from tests.spec.fixtures import spec as base_spec

REPLAY_ID = "replay-rec1"


def spec(checks=None, gaps=()):
    """The shared Spec with its write naming the row by its canonical id, the key check_run scores on."""
    written = check(demand=dict(WRITE, entity=fn(WRITE["entity"])))
    return base_spec(checks=[written] if checks is None else checks, gaps=gaps)


def _verifier(the_spec=None) -> Verifier:
    verifier = compile_spec(the_spec or spec(), WRITE_TOOLS, fn).verifier
    at = next(e.idx for e in reference_run().events if e.type == "tool_call" and e.payload["name"] == "update_item")
    return verifier.model_copy(update={"atoms": [a.model_copy(update={"target": dict(a.target, at=at)})
                                                 for a in verifier.atoms]})


def _replay():
    return reference_run().model_copy(update={"run_id": REPLAY_ID, "model": "recorded", "trace_id": "rec1"})


def _fresh(run_id="fresh-1", right=True):
    run = reference_run().model_copy(update={"run_id": run_id, "model": "some/model"})
    if right:
        return run
    return run.model_copy(update={"events": [e for e in run.events if e.idx < 3 or e.idx == 5]})


def _tier(runs, the_spec=None, rulings=(), verifier=None):
    the_spec = the_spec or spec(gaps=["f2"])
    return T.tier_of_task(the_spec, verifier or _verifier(the_spec), runs, rulings, canon=fn,
                          write_tools=WRITE_TOOLS, prefix="replay-")


def test_a_grounded_spec_that_can_fail_with_a_fresh_pass_and_no_ruling_is_trusted_and_names_the_runs_scored():
    tier, row = _tier([_replay(), _fresh()])
    assert tier == "trusted" and row["failing"] is None and all(row["gates"].values())
    assert row["runs_scored"] == ["fresh-1", REPLAY_ID]
    assert row["fresh_passed"] == ["fresh-1"] and row["replay"] == {REPLAY_ID: True}


def test_a_replay_pass_without_a_fresh_pass_is_replay_only_and_the_replay_verdict_is_evidence_not_the_condition():
    tier, row = _tier([_replay(), _fresh(right=False)])
    assert tier == "replay_only" and row["failing"] == "independent_pass"
    assert row["replay"] == {REPLAY_ID: True} and row["fresh_passed"] == []


def test_a_failing_recording_with_a_fresh_pass_is_trusted():
    tier, row = _tier([_fresh("replay-rec1", right=False), _fresh()])
    assert tier == "trusted" and row["replay"] == {REPLAY_ID: False}


def test_a_check_whose_because_quotes_nothing_is_untrusted_at_the_first_gate():
    the_spec = spec(checks=[check(because="it seemed right")], gaps=["f2"])
    tier, row = _tier([_replay(), _fresh()], the_spec=the_spec)
    assert tier == "untrusted" and row["failing"] == "grounded" and row["grounding"]["refusals"] == 1


def test_an_unlisted_fact_without_a_check_is_not_grounded():
    tier, row = _tier([_replay(), _fresh()], the_spec=spec())
    assert tier == "untrusted" and row["failing"] == "grounded"


def test_a_verifier_that_demands_nothing_cannot_fail_and_is_untrusted():
    tier, row = _tier([_replay(), _fresh()], verifier=Verifier(task_id="t1"))
    assert tier == "untrusted" and row["failing"] == "can_fail"


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


def test_the_replay_prefix_is_read_off_the_runners_replay_index_and_without_one_the_recorded_model_decides():
    index = {"t1": {"rec1": {"run_id": "replay-rec1", "trace_id": "rec1"}},
             "t2": {"rec2": {"run_id": "replay-rec2", "trace_id": "rec2"}, "x": {"run_id": "x", "trace_id": ""}}}
    assert T.replay_prefix(index) == "replay-"
    assert T.replay_prefix({}) is None
    assert T.is_replay(_replay(), None) and not T.is_replay(_fresh(), None)


def test_the_runs_path_is_the_gates_trusted_ruling_task_for_task(tmp_path):
    world = _world(tmp_path)
    tiers = T.runs_tiers(**world)
    ruling = gates_trust.trusted_gate(**world)
    assert [t for t, v in tiers.items() if v.tier == "trusted"] == ruling.metrics["trusted"] == [TASK]
    world["task_status"] = {TASK: {"verifier_passed": False}}
    tiers = T.runs_tiers(**world)
    assert tiers[TASK].tier == "untrusted"
    assert tiers[TASK].row["reason"] == gates_trust.trusted_gate(**world).metrics["untrusted"][TASK]


def test_a_workdir_under_runs_is_the_gates_workdir_ruling(tmp_path):
    ruling = gates_trust.workdir_trusted_ruling(tmp_path)
    assert T.workdir_tiers(tmp_path) == T.tiers_of_ruling(ruling)


def test_the_build_manifest_keeps_the_foundation_and_whatever_else_it_holds(tmp_path):
    assert T.verifier_from_of(tmp_path) == "runs" and T.stored_verifier_from(tmp_path) is None
    (tmp_path / T.BUILD_MANIFEST).write_text(json.dumps({"other": 1}))
    T.store_verifier_from(tmp_path, "intent")
    assert json.loads((tmp_path / T.BUILD_MANIFEST).read_text()) == {"other": 1, "verifier_from": "intent"}
    assert T.verifier_from_of(tmp_path) == "intent"
    with pytest.raises(ValueError):
        T.store_verifier_from(tmp_path, "neither")


def _write_verifier(root, verifier, task_id="t1"):
    path = T.spec_verifier_path(root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verifier.model_dump_json())


def _write_run(folder, run):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run.run_id}.jsonl").write_text(run.model_dump_json() + "\n")


def _intent_workdir(root, runs, verifier=None):
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"}]))
    (root / "task_status.json").write_text(json.dumps({"t1": {}, "t2": {}}))
    (root / "replays.json").write_text(json.dumps({"t1": {"rec1": {"run_id": REPLAY_ID, "trace_id": "rec1"}}}))
    for run in runs:
        _write_run(root / "runs", run)
    save_spec(root, spec(gaps=["f2"]))
    _write_verifier(root, verifier or _verifier(spec(gaps=["f2"])))


def test_a_workdir_under_intent_rules_every_task_from_its_spec_and_a_task_without_one_is_untrusted(tmp_path):
    _intent_workdir(tmp_path, [_replay(), _fresh(right=False)])
    T.store_verifier_from(tmp_path, "intent")
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
