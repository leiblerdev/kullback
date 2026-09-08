"""D201: a repair lands only where it fixed its target and cost nothing else.

The domain is a seed nursery: trays of seedlings, a tool that lists them and a tool that waters one.
"""

from __future__ import annotations

import json

from kullback.builder import repair, transaction


def _write(workdir, name, body):
    path = workdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _nursery(workdir, differing=()):
    """A build with two Tasks, two tools, and every light green unless a call is named as differing."""
    _write(workdir, "tasks_frozen.json", ["task_sow", "task_water"])
    _write(workdir, "tool_fidelity.json", {
        "tasks": {
            "task_sow": {"list_trays": {"replayed": 2, "differing": 0, "reasons": []}},
            "task_water": {"list_trays": {"replayed": 1, "differing": 0, "reasons": []},
                           "water_tray": {"replayed": 3, "differing": 0, "reasons": []}},
        },
        "tools": {},
    })
    for task, tool in differing:
        fidelity = json.loads((workdir / "tool_fidelity.json").read_text(encoding="utf-8"))
        fidelity["tasks"][task][tool]["differing"] = 1
        _write(workdir, "tool_fidelity.json", fidelity)
    _write(workdir, "task_status.json", {
        "task_sow": {"reference_confirmed": True, "verifier_passed": True},
        "task_water": {"reference_confirmed": True, "verifier_passed": True},
    })
    _write(workdir, "gates.json", [{"stage": "trusted", "pass": True,
                                    "metrics": {"trusted": ["task_sow", "task_water"]}, "failures": []}])
    _write(workdir, "tool_builds.json", {"water_tray": {"assisted": True, "score": [4, 10]},
                                         "list_trays": {"assisted": False, "score": [7, 3]}})
    _write(workdir, "bodies.json", {"water_tray": "def water_tray(): return 1"})


def _apply(workdir, kind, target, change, round_no=1):
    """Open a transaction, let `change` move the artifacts, close it, and record the request."""
    txn = transaction.open_transaction(workdir, kind, target, round_no=round_no)
    change()
    ruling = transaction.close(txn)
    repair.record_request(workdir, kind, target, {"arguments": {}, **ruling.as_row()}, round_no=round_no)
    return ruling


def test_a_repair_that_fixes_its_target_and_breaks_nothing_lands(tmp_path):
    _nursery(tmp_path)
    _write(tmp_path, "intents/task_sow.json", {"grounded": False, "reason": "no Run says it"})

    def rewrite():
        _write(tmp_path, "intents/task_sow.json", {"grounded": True, "text": "sow a tray"})

    ruling = _apply(tmp_path, "repair_intent", "task_sow", rewrite)
    assert ruling.status == transaction.ACCEPTED and ruling.accepted
    assert ruling.moved and ruling.broke == [] and ruling.scope == 1 and ruling.scoped
    assert "accepted: fixed task_sow" in ruling.line
    kept = json.loads((tmp_path / "intents" / "task_sow.json").read_text(encoding="utf-8"))
    assert kept["grounded"] is True


def test_a_repair_that_fixes_its_target_and_breaks_another_task_is_reverted(tmp_path):
    """The ruling names the Task it broke and the light it lost, which is the lesson the next one reads."""
    _nursery(tmp_path)

    def recompile():
        builds = json.loads((tmp_path / "tool_builds.json").read_text(encoding="utf-8"))
        builds["water_tray"] = {"assisted": False, "score": [7, 12]}
        _write(tmp_path, "tool_builds.json", builds)
        fidelity = json.loads((tmp_path / "tool_fidelity.json").read_text(encoding="utf-8"))
        fidelity["tasks"]["task_water"]["water_tray"]["differing"] = 2
        _write(tmp_path, "tool_fidelity.json", fidelity)
        _write(tmp_path, "bodies.json", {"water_tray": "def water_tray(): return 2"})

    ruling = _apply(tmp_path, "repair_recompile", "water_tray", recompile)
    assert ruling.status == transaction.REVERTED_REGRESSION and not ruling.accepted
    assert ruling.moved and ruling.broke == ["task_water replay"]
    assert "reverted: fixed water_tray, broke 1: task_water replay" in ruling.line
    # Put back byte for byte: the body, the row it was ruled on, and the fidelity it cost.
    assert json.loads((tmp_path / "bodies.json").read_text(encoding="utf-8")) == \
        {"water_tray": "def water_tray(): return 1"}
    builds = json.loads((tmp_path / "tool_builds.json").read_text(encoding="utf-8"))
    assert builds["water_tray"]["assisted"] is True
    fidelity = json.loads((tmp_path / "tool_fidelity.json").read_text(encoding="utf-8"))
    assert fidelity["tasks"]["task_water"]["water_tray"]["differing"] == 0


def test_a_repair_that_fixes_nothing_is_reverted_for_no_effect(tmp_path):
    _nursery(tmp_path)
    ruling = _apply(tmp_path, "repair_recompile", "water_tray", lambda: None)
    assert ruling.status == transaction.REVERTED_NO_EFFECT and not ruling.moved and ruling.broke == []
    assert "reverted: no effect" in ruling.line


def test_a_repair_that_breaks_a_task_without_fixing_its_target_says_which_it_was(tmp_path):
    _nursery(tmp_path)

    def recompile():
        fidelity = json.loads((tmp_path / "tool_fidelity.json").read_text(encoding="utf-8"))
        fidelity["tasks"]["task_water"]["water_tray"]["differing"] = 1
        _write(tmp_path, "tool_fidelity.json", fidelity)

    ruling = _apply(tmp_path, "repair_recompile", "water_tray", recompile)
    assert ruling.status == transaction.REVERTED_REGRESSION and not ruling.moved
    assert "reverted: did not fix water_tray, broke 1" in ruling.line


def test_the_set_holds_only_the_tasks_whose_calls_the_repaired_tool_answers(tmp_path):
    _nursery(tmp_path)
    scope, scoped = transaction.scope_of(tmp_path, "repair_recompile", "water_tray")
    assert scope == ["task_water"] and scoped
    scope, scoped = transaction.scope_of(tmp_path, "repair_recompile", "list_trays")
    assert scope == ["task_sow", "task_water"] and scoped
    scope, scoped = transaction.scope_of(tmp_path, "repair_intent", "task_sow")
    assert scope == ["task_sow"] and scoped


def test_a_kind_whose_set_cannot_be_worked_out_is_held_against_the_whole_corpus(tmp_path):
    _nursery(tmp_path)
    scope, scoped = transaction.scope_of(tmp_path, "repair_grow", "trays")
    assert scope == ["task_sow", "task_water"] and not scoped


def test_the_fourth_whole_corpus_repair_of_a_round_waits_for_round_end(tmp_path):
    """Three repairs a round may hold against every Task; the fourth applies and says it is unchecked."""
    _nursery(tmp_path)
    _write(tmp_path, "db.json", {"trays": {"tray_1": {"seedlings": 4}}})

    def grow():
        db = json.loads((tmp_path / "db.json").read_text(encoding="utf-8"))
        db["trays"][f"tray_{len(db['trays']) + 1}"] = {"seedlings": 9}
        _write(tmp_path, "db.json", db)

    for _ in range(transaction.WHOLE_CORPUS_LIMIT):
        assert _apply(tmp_path, "repair_grow", "trays", grow).status == transaction.ACCEPTED
    ruling = _apply(tmp_path, "repair_grow", "trays", grow, round_no=1)
    assert ruling.status == transaction.DEFERRED
    assert "waits for the round's own gates at round end" in ruling.line
    # A later round starts the count again, because the round's own gates have ruled since.
    assert _apply(tmp_path, "repair_grow", "trays", grow, round_no=2).status != transaction.DEFERRED


def test_the_body_scorer_rules_through_the_same_function(tmp_path):
    """D184's keep or beat is this rule with the tool's own recorded calls as its set."""
    kept = [{"call_id": "c1", "replayed": True}, {"call_id": "c2", "replayed": True},
            {"call_id": "c3", "replayed": False}]
    better = [{"call_id": "c1", "replayed": True}, {"call_id": "c2", "replayed": True},
              {"call_id": "c3", "replayed": True}]
    traded = [{"call_id": "c1", "replayed": True}, {"call_id": "c2", "replayed": False},
              {"call_id": "c3", "replayed": True}]
    lands = transaction.rule("compile_tools.body", "water_tray", transaction.improved([6, 2], [7, 3]),
                             transaction.call_lights(kept), transaction.call_lights(better))
    assert lands.accepted and lands.scope == 3
    # Two calls gained and one lost is not an improvement, whatever the score says.
    cost = transaction.rule("compile_tools.body", "water_tray", transaction.improved([6, 2], [7, 3]),
                            transaction.call_lights(kept), transaction.call_lights(traded))
    assert cost.status == transaction.REVERTED_REGRESSION and cost.broke == ["c2 replay"]
    tie = transaction.rule("compile_tools.body", "water_tray", transaction.improved([7, 3], [7, 3]),
                           transaction.call_lights(kept), transaction.call_lights(kept))
    assert tie.status == transaction.REVERTED_NO_EFFECT


def test_a_light_no_gate_has_ruled_on_is_never_counted_as_lost(tmp_path):
    """A repair answers for the lights that were on when it started, not for gates that never ran."""
    _write(tmp_path, "tasks_frozen.json", ["task_sow"])
    before = transaction.task_lights(tmp_path, ["task_sow"])
    assert before["task_sow"] == {"replay": None, "reference": None, "suite": None, "trusted": None}
    _write(tmp_path, "task_status.json", {"task_sow": {"reference_confirmed": False,
                                                       "verifier_passed": False}})
    after = transaction.task_lights(tmp_path, ["task_sow"])
    assert transaction.regressions(before, after) == []


def test_the_round_counts_and_the_report_line_read_off_the_recorded_rows(tmp_path):
    _nursery(tmp_path)
    _write(tmp_path, "intents/task_sow.json", {"grounded": False})
    _apply(tmp_path, "repair_intent", "task_sow",
           lambda: _write(tmp_path, "intents/task_sow.json", {"grounded": True, "text": "sow"}))
    _apply(tmp_path, "repair_recompile", "water_tray", lambda: None)
    rows = transaction.recorded_repairs(tmp_path, 1)
    counts = transaction.round_outcomes(rows)
    assert counts == {"repairs_accepted": 1, "repairs_reverted_regression": 0,
                      "repairs_reverted_no_effect": 1, "repairs_deferred": 0}
    said = transaction.reverted_by_kind(rows)
    assert said == "reverted repairs: repair_recompile 1 (1 for no effect)"


def test_a_revert_removes_a_file_the_repair_created(tmp_path):
    """"As it was" for a file that did not exist is that it does not exist."""
    _nursery(tmp_path)
    (tmp_path / "tool_call_outcomes.json").unlink(missing_ok=True)
    _apply(tmp_path, "repair_recompile", "water_tray",
           lambda: _write(tmp_path, "tool_call_outcomes.json", {"water_tray": []}))
    assert not (tmp_path / "tool_call_outcomes.json").exists()


def test_the_stages_own_ruling_is_stamped_and_not_put_back(tmp_path):
    """The score pair is the lesson the next hint is written against, so a revert marks it rather
    than taking it away; without it a reverted recompile would say nothing at all."""
    _nursery(tmp_path)

    def recompile():
        _write(tmp_path, "kept_bodies.json",
               {"water_tray": {"outcome": "beaten", "attempt_score": [7, 12], "kept_score": [4, 10]}})
        fidelity = json.loads((tmp_path / "tool_fidelity.json").read_text(encoding="utf-8"))
        fidelity["tasks"]["task_water"]["water_tray"]["differing"] = 1
        _write(tmp_path, "tool_fidelity.json", fidelity)
        builds = json.loads((tmp_path / "tool_builds.json").read_text(encoding="utf-8"))
        builds["water_tray"] = {"assisted": False, "score": [7, 12]}
        _write(tmp_path, "tool_builds.json", builds)

    ruling = _apply(tmp_path, "repair_recompile", "water_tray", recompile)
    assert ruling.status == transaction.REVERTED_REGRESSION
    row = json.loads((tmp_path / "kept_bodies.json").read_text(encoding="utf-8"))["water_tray"]
    assert row["attempt_score"] == [7, 12] and row["reverted"] == transaction.REVERTED_REGRESSION
    assert row["reverted_tasks"] == ["task_water replay"]
    assert "the repair was then put back because it cost task_water replay" in \
        repair.recompile_ruling(tmp_path, "water_tray")
