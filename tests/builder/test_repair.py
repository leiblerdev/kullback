"""Phase 6 repair verbs, ratchet, and tool lessons."""

from __future__ import annotations

import asyncio
import json

from kullback.agent.messages import ToolCall
from kullback.builder import repair


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_verbs_record_requests(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    assert sorted(tools) == ["repair_escalate", "repair_grow", "repair_recompile", "repair_record_finding",
                              "repair_refuse_task", "repair_rewrite_skill"]
    out = _run(tools["repair_recompile"], {"name": "get_order"})
    assert not out.is_error and "repair_recompile get_order" in out.content
    out = _run(tools["repair_grow"], {"table": "orders", "count": 50})
    assert not out.is_error
    out = _run(tools["repair_rewrite_skill"], {"name": "compile", "content": "# compile skill"})
    assert not out.is_error and (tmp_path / "skills" / "compile" / "SKILL.md").is_file()
    out = _run(tools["repair_refuse_task"], {"task_id": "t1", "reason": "no frontier run finishes"})
    assert not out.is_error
    out = _run(tools["repair_escalate"], {"task_id": "t2", "queue": "review"})
    assert not out.is_error
    assert (tmp_path / "repairs" / "repair_recompile.jsonl").is_file()


def _requests(workdir, verb) -> list[dict]:
    lines = (workdir / "repairs" / f"{verb}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_a_request_records_the_round_it_was_made_in(tmp_path):
    """The request carries a timestamp and rounds.json carries none, so the round number is the only
    thing that places a repair against the rulings that round left (gates_by_round.json)."""
    at_round = {"n": 1}
    tools = {t.name: t for t in repair.repair_tools(tmp_path, round_of=lambda: at_round["n"])}
    _run(tools["repair_recompile"], {"name": "get_order"})
    at_round["n"] = 3
    _run(tools["repair_recompile"], {"name": "get_order"})
    _run(tools["repair_escalate"], {"task_id": "t1", "queue": "review"})
    assert [row["round"] for row in _requests(tmp_path, "repair_recompile")] == [1, 3]
    assert [row["round"] for row in _requests(tmp_path, "repair_escalate")] == [3]


def test_a_request_made_outside_a_round_driver_is_round_one(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    _run(tools["repair_refuse_task"], {"task_id": "t1", "reason": "no frontier run finishes"})
    row = _requests(tmp_path, "repair_refuse_task")[0]
    assert row["round"] == 1 and row["target"] == "t1" and row["at"] > 0


def test_an_acting_verb_records_the_round_the_build_plan_is_in(tmp_path):
    """The two acting verbs are registered from builder/tools.py against the plan, and the round
    moves under a session that was registered once, so the plan is asked at call time."""
    from kullback.builder.build import BuildPlan
    from kullback.builder.tools import repair_verb_tools
    plan = BuildPlan(workdir=tmp_path, max_attempts=0)
    tools = {t.name: t for t in repair_verb_tools(plan)}
    plan.round = 4
    _run(tools["repair_recompile"], {"name": "get_order", "hint": "it never imported decimal"})
    _run(tools["repair_escalate"], {"task_id": "t1"})
    assert [row["round"] for row in _requests(tmp_path, "repair_recompile")] == [4]
    assert _requests(tmp_path, "repair_recompile")[0]["lesson"] == "it never imported decimal"
    assert [row["round"] for row in _requests(tmp_path, "repair_escalate")] == [4]


def test_verbs_reject_bad_args(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    out = _run(tools["repair_grow"], {"table": "orders", "count": 0})
    assert out.is_error


def test_repair_verbs_do_not_collide_with_builder_tools(tmp_path):
    from kullback.builder.build import BuildPlan
    from kullback.builder.tools import builder_tools
    builder_names = {t.name for t in builder_tools(BuildPlan(workdir=tmp_path))}
    repair_names = {t.name for t in repair.repair_tools(tmp_path)}
    assert not (builder_names & repair_names)


# --- what one repair did to its own target -------------------------------------

def _intent_file(workdir, task_id: str, **fields) -> None:
    path = workdir / "intents" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task_id, **fields}), encoding="utf-8")


def test_a_targets_hash_moves_only_when_that_targets_own_artifact_moves(tmp_path):
    """A round-wide fingerprint says `intents` changed and cannot say which Task it was, so a round
    that wrote six Intents again and left a seventh refused reads as seven repairs that worked."""
    _intent_file(tmp_path, "task_a", text="renew the loan", grounded=True)
    _intent_file(tmp_path, "task_b", text="find a branch", grounded=True)
    before = repair.target_hash(tmp_path, "repair_intent", "task_a")
    _intent_file(tmp_path, "task_b", text="find the nearest branch", grounded=True)
    assert repair.target_hash(tmp_path, "repair_intent", "task_a") == before
    _intent_file(tmp_path, "task_a", text="renew both loans", grounded=True)
    assert repair.target_hash(tmp_path, "repair_intent", "task_a") != before


def test_a_change_carries_the_hash_either_side_of_the_call_and_whether_the_target_moved(tmp_path):
    before = repair.target_hash(tmp_path, "repair_recompile", "renew_loan")
    (tmp_path / "bodies.json").write_text(json.dumps({"renew_loan": "def renew_loan(db): return db"}),
                                          encoding="utf-8")
    change = repair.change_of(tmp_path, "repair_recompile", "renew_loan", before)
    assert change == {"changed": True, "hash_before": before,
                      "hash_after": repair.target_hash(tmp_path, "repair_recompile", "renew_loan")}
    assert repair.change_of(tmp_path, "repair_recompile", "renew_loan",
                            change["hash_after"])["changed"] is False


def test_a_request_records_whether_the_call_changed_the_targets_own_artifact(tmp_path):
    """The verbs of this module only record, so nothing moves between the two hashes they take."""
    (tmp_path / "bodies.json").write_text(json.dumps({"renew_loan": "def renew_loan(db): return db"}),
                                          encoding="utf-8")
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    _run(tools["repair_recompile"], {"name": "renew_loan"})
    row = _requests(tmp_path, "repair_recompile")[0]
    assert row["changed"] is False and row["hash_before"] == row["hash_after"] is not None


def test_a_deciding_verb_owns_no_artifact_so_its_request_can_only_read_as_unchanged(tmp_path):
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    _run(tools["repair_refuse_task"], {"task_id": "task_a", "reason": "no frontier Run finishes it"})
    row = _requests(tmp_path, "repair_refuse_task")[0]
    assert row["changed"] is False and row["hash_before"] is None and row["hash_after"] is None


# --- the target's own ruling, off the artifact the stage just wrote -------------

def test_an_intent_that_grounded_reads_back_as_grounded_with_its_own_text(tmp_path):
    _intent_file(tmp_path, "task_a", text="renew the loan on the overdue book", grounded=True)
    args = repair.IntentRepairArgs(task_id="task_a")
    assert repair.target_ruling(tmp_path, "repair_intent", args) == (
        'repair_intent task_a: grounded: "renew the loan on the overdue book"')


def test_an_intent_still_refused_reads_back_with_the_reason_its_own_record_carries(tmp_path):
    _intent_file(tmp_path, "task_a", grounded=False, reason="noun phrases with no span: the late fee")
    assert repair.intent_ruling(tmp_path, "task_a") == (
        "repair_intent task_a: still refused: noun phrases with no span: the late fee")
    assert repair.intent_ruling(tmp_path, "task_none") == (
        f"repair_intent task_none: still refused: {repair.NO_INTENT}")


def test_a_tool_that_cleared_the_gates_says_so_and_an_assisted_one_carries_every_failing_shape(tmp_path):
    """A tool that ends assisted leaves the compile_tools stage gate passing, so the gate-wide line
    said `compile_tools pass` while the body the mechanic asked for was never written."""
    (tmp_path / "tool_builds.json").write_text(json.dumps({
        "renew_loan": {"assisted": False, "nodes": [{"attempt": 0, "gates": [
            {"stage": "parses", "pass": True, "failures": []}]}]},
        "find_branch": {"assisted": True, "nodes": [
            {"attempt": 0, "gates": [{"stage": "parses", "pass": True, "failures": []}]},
            {"attempt": 1, "gates": [
                {"stage": "parses", "pass": True, "failures": []},
                {"stage": "executes_on_s0", "pass": False,
                 "failures": ["find_branch({}) raised NameError: name 'decimal' is not defined",
                              "find_branch({'city': 'x'}) raised the same"]}]}]},
        "no_reply": {"assisted": True, "nodes": [{"attempt": 0, "failures": ["no body was submitted"]}]},
    }), encoding="utf-8")
    assert repair.recompile_ruling(tmp_path, "renew_loan") == "repair_recompile renew_loan: cleared the gates"
    assert repair.recompile_ruling(tmp_path, "find_branch") == (
        "repair_recompile find_branch: still assisted: executes_on_s0: 2 calls failed in 2 shapes: "
        "find_branch({}) raised NameError: name 'decimal' is not defined (1 call); "
        "find_branch({'city': 'x'}) raised the same (1 call)")
    assert repair.recompile_ruling(tmp_path, "no_reply") == (
        "repair_recompile no_reply: still assisted: no body was submitted")
    assert repair.recompile_ruling(tmp_path, "nothing_compiled") == (
        f"repair_recompile nothing_compiled: {repair.NO_ATTEMPT}")


def test_calls_that_failed_the_same_way_are_one_shape_with_a_count_and_only_three_are_shown(tmp_path):
    """A gate that ruled over sixty calls left sixty sentences and the mechanic saw one of them, so
    its hint answered one example and the next recompile met the rest."""
    same = [f"find_branch({{'city': 'c-{i}'}}) answered nothing, the recording answered a branch"
            for i in range(60)]
    others = [f"find_branch({{'city': 'c-{i}'}}) raised KeyError: 'c-{i}'" for i in range(4)]
    (tmp_path / "tool_builds.json").write_text(json.dumps({
        "find_branch": {"assisted": True, "nodes": [{"attempt": 0, "gates": [
            {"stage": "replay_fidelity", "pass": False, "failures": same + others}]}]}}),
        encoding="utf-8")
    ruling = repair.recompile_ruling(tmp_path, "find_branch")
    assert "64 calls failed in 2 shapes" in ruling
    assert "(60 calls)" in ruling and "(4 calls)" in ruling


def test_only_the_first_three_shapes_are_shown_and_the_rest_are_counted(tmp_path):
    ways = ["answered nothing", "answered another branch", "raised", "answered a list", "timed out"]
    failures = [f"find_branch({{'city': 'c-{i}'}}) {way}" for i, way in enumerate(ways)]
    (tmp_path / "tool_builds.json").write_text(json.dumps({
        "find_branch": {"assisted": True, "nodes": [{"attempt": 0, "gates": [
            {"stage": "replay_fidelity", "pass": False, "failures": failures}]}]}}),
        encoding="utf-8")
    ruling = repair.recompile_ruling(tmp_path, "find_branch")
    assert ruling.count(" call)") == 3 and ruling.endswith("; 2 more shapes")


def test_a_grown_table_says_how_many_rows_it_holds_against_the_count_that_was_asked_for(tmp_path):
    (tmp_path / "db.json").write_text(json.dumps({"branches": {"b1": {}, "b2": {}}}), encoding="utf-8")
    assert repair.grow_ruling(tmp_path, "branches", 2) == "repair_grow branches: 2 rows, the 2 asked for"
    assert repair.grow_ruling(tmp_path, "branches", 5) == (
        "repair_grow branches: 2 rows, short of the 5 asked for")
    assert repair.grow_ruling(tmp_path, "loans", 5) == (
        "repair_grow loans: the Starting state holds no table of that name")


def test_a_verb_that_decides_something_has_no_target_ruling_to_open_with(tmp_path):
    args = repair.RefuseTaskArgs(task_id="task_a", reason="no frontier Run finishes it")
    assert repair.target_ruling(tmp_path, "repair_refuse_task", args) == ""


def test_ratchet_never_replaces_pass_with_fail():
    prior = {"bodies": {"a": "old-good", "b": "old-b"}}
    new = {"bodies": {"a": "new-bad", "b": "new-b"}}
    out = repair.ratchet_bodies(prior, new, {"a": False, "b": True})
    assert out["bodies"]["a"] == "old-good"
    assert out["bodies"]["b"] == "new-b"


def test_ratchet_keeps_tools_only_the_prior_has():
    prior = {"bodies": {"kept": "old-good", "a": "old-a"}}
    new = {"bodies": {"a": "new-a"}}
    out = repair.ratchet_bodies(prior, new, {"a": True})
    assert out["bodies"]["kept"] == "old-good"
    assert out["bodies"]["a"] == "new-a"


def test_ratchet_hook_restores_prior(tmp_path):
    (tmp_path / "bodies.json").write_text(json.dumps({"bodies": {"calc": "old-good"}}))
    hook = repair.ratchet_hook(tmp_path)
    call = ToolCall(id="c1", name="compile_tool", arguments={"name": "calc"})
    from kullback.agent.tools import ToolResult
    failed = ToolResult(content="compile_tool calc: failed",
                        details={"produced": ["bodies"], "payload": {},
                                 "stage_gates": [{"stage": "executes", "passed": False}]})
    out = hook(call, failed)
    assert out is not None and out.details["ratchet_restored"] == "calc"
    assert out.details["ratchet_restored_body"] == "old-good"


def test_tool_lesson_round_trip(tmp_path):
    assert repair.lesson_for_tool(tmp_path, "calc") == ""
    repair.record_tool_lesson(tmp_path, "calc", ["executes: import decimal missing"])
    text = repair.lesson_for_tool(tmp_path, "calc")
    assert "decimal" in text and "do not repeat" in text


CRASH = "get_member(member_id='m-1') raised AttributeError: 'dict' object has no attribute 'copy_id'"


def _tool_builds(tmp_path, failures):
    (tmp_path / "tool_builds.json").write_text(json.dumps(
        {"get_member": {"assisted": True, "nodes": [
            {"attempt": 0, "gates": [{"stage": "parses", "pass": True, "failures": []}]},
            {"attempt": 1, "gates": [{"stage": "parses", "pass": True, "failures": []},
                                     {"stage": "executes_on_s0", "pass": False, "failures": failures}]}]}}),
        encoding="utf-8")


def test_a_lesson_carries_the_exception_the_last_attempt_raised_beside_the_hint(tmp_path):
    """The hint is what the mechanic thinks went wrong; the gate says what the body actually raised.
    Without the second, the compiler is asked again and has no line to avoid."""
    _tool_builds(tmp_path, [CRASH])
    repair.record_tool_lesson(tmp_path, "get_member", ["read the loans column by key"])
    text = repair.lesson_for_tool(tmp_path, "get_member")
    assert "read the loans column by key" in text
    assert "AttributeError: 'dict' object has no attribute 'copy_id'" in text


def test_a_hint_that_already_quotes_the_exception_does_not_carry_it_twice(tmp_path):
    _tool_builds(tmp_path, [CRASH])
    hint = "it raised AttributeError: 'dict' object has no attribute 'copy_id'; read it by key"
    repair.record_tool_lesson(tmp_path, "get_member", [hint])
    text = repair.lesson_for_tool(tmp_path, "get_member")
    assert text.count("no attribute 'copy_id'") == 1


def test_a_lesson_for_a_memorising_body_says_to_write_the_lookup_over_the_worlds_tables(tmp_path):
    """D162: the gate's failures name the literals of one attempt; the lesson belongs to the tool
    and says the one thing every such failure is repaired by."""
    (tmp_path / "tool_builds.json").write_text(json.dumps(
        {"get_member": {"assisted": True, "nodes": [{"attempt": 0, "gates": [
            {"stage": "compile_tools.memorised_values", "pass": False,
             "failures": ["get_member: the literal 'M0042' is a row id of members in the Starting state"]}]}]}}),
        encoding="utf-8")
    repair.record_tool_lesson(tmp_path, "get_member", ["read the member by the argument"])
    text = repair.lesson_for_tool(tmp_path, "get_member")
    assert "read the member by the argument" in text
    assert "memorised recorded ids" in text and "world's tables" in text


def test_a_tool_whose_latest_body_memorised_nothing_leaves_no_such_lesson(tmp_path):
    _tool_builds(tmp_path, [CRASH])
    assert repair.memorised_values_lesson(tmp_path, "get_member") == ""


def test_the_ratchet_does_not_keep_a_prior_body_a_gate_of_this_build_refuses():
    """A body cleared the gates of the build it was written in. A build that adds a gate (D162) can
    hold a prior body no gate accepts today, and ratcheting onto it is a repair that never lands."""
    prior = {"bodies": {"a": "old-memorising", "b": "old-b"}}
    new = {"bodies": {"a": "new-bad", "b": "new-b"}}
    out = repair.ratchet_bodies(prior, new, {"a": False, "b": True}, refused=["a"])
    assert out["bodies"]["a"] == "new-bad"
    assert out["bodies"]["b"] == "new-b"


def test_the_ratchet_hook_restores_nothing_for_a_tool_whose_prior_body_is_refused(tmp_path):
    (tmp_path / "bodies.json").write_text(json.dumps({"bodies": {"calc": "old-memorising"}}))
    hook = repair.ratchet_hook(tmp_path, refused=["calc"])
    call = ToolCall(id="c1", name="compile_tool", arguments={"name": "calc"})
    from kullback.agent.tools import ToolResult
    failed = ToolResult(content="compile_tool calc: failed",
                        details={"produced": ["bodies"], "payload": {},
                                 "stage_gates": [{"stage": "compile_tools.memorised_values",
                                                  "passed": False}]})
    assert hook(call, failed) is None


def test_a_workdir_with_no_tool_builds_records_the_hint_alone(tmp_path):
    assert repair.gate_exception_line(tmp_path, "get_member") == ""
    repair.record_tool_lesson(tmp_path, "get_member", ["read the loans column by key"])
    assert "raised" not in repair.lesson_for_tool(tmp_path, "get_member")


def test_a_finding_records_the_tool_the_leaf_and_the_calls_it_rests_on_and_moves_nothing(tmp_path):
    """D155: the recording is the standard, so the body reproduces what several recorded calls agree
    on, and the finding is how the customer reads what was reproduced."""
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    out = _run(tools["repair_record_finding"], {
        "tool": "update_booking", "evidence": ["call_12", "call_40"],
        "finding": "rooms[*].rate: on a call that changes several rooms the recording writes the last "
                   "new room's rate on every changed row; the body reproduces it"})
    assert not out.is_error and "no gate moves" in out.content
    row = _requests(tmp_path, "repair_record_finding")[0]
    assert row["target"] == "update_booking" and row["changed"] is False
    assert row["arguments"]["evidence"] == ["call_12", "call_40"]
    assert row["arguments"]["finding"].startswith("rooms[*].rate:")


def test_the_same_task_refused_twice_for_the_same_reason_is_refused_a_third_time(tmp_path):
    """D181 stopped the Examiner repairing one Verifier against one check for ever; the Builder's
    refusal had no such stop. One live build made 17 of them, had 0 admitted, and two rounds later
    refused four Tasks again with the reason word for word. Refusing moves nothing, so two
    identical requests are all the information there is."""
    tools = {t.name: t for t in repair.repair_tools(tmp_path)}
    ask = {"task_id": "task_dock", "reason": "no frontier Run docks the bike"}
    assert not _run(tools["repair_refuse_task"], ask).is_error
    assert not _run(tools["repair_refuse_task"], dict(ask)).is_error
    third = _run(tools["repair_refuse_task"], dict(ask))
    assert third.is_error and "refused 2 times for this reason" in third.content
    assert "repair_escalate" in third.content, "and what buys something instead"
    assert repair.refused_twice(tmp_path) == ["task_dock"]
    # A different reason is a different thing to say, and so is a different Task.
    assert not _run(tools["repair_refuse_task"],
                    {"task_id": "task_dock", "reason": "its recordings disagree"}).is_error
    assert not _run(tools["repair_refuse_task"], {"task_id": "task_rack", **{"reason": ask["reason"]}}).is_error
    rows = _requests(tmp_path, "repair_refuse_task")
    assert [row.get("blocked") for row in rows] == [None, None, True, None, None]
    assert repair.refuse_repeats(tmp_path) == 2, "the second identical ask and the blocked third"
    assert repair.refuse_repeats(tmp_path, round_no=99) == 0, "narrowed to a round that made none"


# --- D191: the repair ruling says what the stage decided, not only what the released body scored ---


def _kept_bodies(tmp_path, **rows):
    (tmp_path / "kept_bodies.json").write_text(json.dumps(rows), encoding="utf-8")


def _builds_rows(tmp_path, **rows):
    (tmp_path / "tool_builds.json").write_text(json.dumps(rows), encoding="utf-8")


def test_a_recompile_whose_attempt_lost_says_so_beside_the_released_bodys_ruling(tmp_path):
    """`cleared the gates` is a reading of the body the stage released, and the released body is the
    one that was already there whenever the attempt did not beat it. Read alone it says a repair
    worked, over rounds in which the repair changed nothing at all."""
    _builds_rows(tmp_path, renew_loan={"assisted": False})
    _kept_bodies(tmp_path, renew_loan={"outcome": "kept", "attempt_score": [6, 12],
                                       "kept_score": [6, 104], "unbeaten": 1,
                                       "from_replay": 0, "evidence_calls": 216})

    ruling = repair.recompile_ruling(tmp_path, "renew_loan")

    assert ruling == ("repair_recompile renew_loan: cleared the gates (the attempt scored [6, 12] "
                      "against the kept body's [6, 104] (gates passed, calls matched), so the body "
                      "already there stands and this recompile changed nothing)")


def test_a_recompile_whose_attempt_won_says_it_was_released(tmp_path):
    _builds_rows(tmp_path, renew_loan={"assisted": False})
    _kept_bodies(tmp_path, renew_loan={"outcome": "beaten", "attempt_score": [7, 40],
                                       "kept_score": [6, 10], "unbeaten": 0})

    assert repair.recompile_ruling(tmp_path, "renew_loan") == (
        "repair_recompile renew_loan: cleared the gates (the attempt scored [7, 40] against the "
        "kept body's [6, 10] (gates passed, calls matched) and was released)")


def test_a_still_assisted_tool_carries_the_score_pair_before_the_failure_the_next_hint_answers(tmp_path):
    _builds_rows(tmp_path, find_branch={"assisted": True, "nodes": [{"attempt": 0, "gates": [
        {"stage": "replay_fidelity", "pass": False, "failures": ["find_branch({}) answered nothing"]}]}]})
    _kept_bodies(tmp_path, find_branch={"outcome": "kept", "attempt_score": [6, 0],
                                        "kept_score": [6, 10], "unbeaten": 1})

    ruling = repair.recompile_ruling(tmp_path, "find_branch")

    assert ruling.startswith("repair_recompile find_branch: still assisted (the attempt scored [6, 0]")
    assert ruling.endswith(": replay_fidelity: find_branch({}) answered nothing")


def test_the_evidence_a_replay_failure_put_back_is_counted_in_the_ruling(tmp_path):
    _builds_rows(tmp_path, renew_loan={"assisted": False})
    _kept_bodies(tmp_path, renew_loan={"outcome": "kept", "attempt_score": [6, 9], "kept_score": [6, 9],
                                       "unbeaten": 1, "from_replay": 7, "evidence_calls": 43})

    assert ("7 of the 43 evidence calls are from_replay, put back because the Reference replay "
            "failed on them") in repair.recompile_ruling(tmp_path, "renew_loan")


def test_a_tool_stalled_for_two_recompiles_says_so_and_one_recompile_does_not(tmp_path):
    assert repair.stalled_note({"unbeaten": 1}) == ""
    assert repair.stalled_note({}) == ""
    assert "stalled: 2 recompiles in a row scored no higher" in repair.stalled_note({"unbeaten": 2})
    assert "stalled: 5 recompiles in a row scored no higher" in repair.stalled_note({"unbeaten": 5})


def test_a_workdir_with_no_kept_body_ruling_leaves_the_line_as_it_was(tmp_path):
    """The first run of the stage for a tool has no body to have beaten, so there is no pair to
    name and the ruling is the one it always was."""
    _builds_rows(tmp_path, renew_loan={"assisted": False})
    assert repair.recompile_ruling(tmp_path, "renew_loan") == "repair_recompile renew_loan: cleared the gates"


def test_a_kept_body_that_could_not_run_says_the_attempt_took_the_tool(tmp_path):
    _builds_rows(tmp_path, renew_loan={"assisted": False})
    _kept_bodies(tmp_path, renew_loan={"outcome": "could_not_run", "attempt_score": [6, 3],
                                       "kept_score": [6, 3], "unbeaten": 0})

    ruling = repair.recompile_ruling(tmp_path, "renew_loan")

    assert "answered no call at all under this world, so the attempt was released" in ruling
