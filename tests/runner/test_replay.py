"""replay.py turns a Trace into the Reference Run the Verifier consumes, scored call by call (D108)."""

from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest

from kullback.builder import effects as effects_mod
from kullback.examiner import derive as verifier_mod
from kullback.gates import verifier_suite as suite
from kullback.runner import replay
from kullback.runner.records import Task, ToolCallError, Trace, Turn
from runner.replay_fixtures import PTR, Toolkit, call, do_replay, events, router, sigs, trace, world  # noqa: F401


def test_the_replay_writes_the_reference_run_the_loop_would(tmp_path):
    out = do_replay(tmp_path)
    assert out.confirmed, out.reasons
    assert out.termination_reason == "user_stop"
    run = suite.load_run(out.path)
    assert run.trace_id == "tr1" and run.task_id == "t1" and run.env_id == "env1" and run.model == "recorded"
    assert [e.type for e in run.events] == [
        "model_call", "user_turn", "model_call", "tool_call", "tool_result", "model_call", "tool_call",
        "tool_result", "model_call", "user_turn", "stop"]
    assert run.events[1].payload["text"] == "Please cancel order 123."
    assert run.events[3].payload == {"id": "c1", "name": "get_order_details", "args": {"order_id": "123"}}
    assert run.events[4].route == "code"
    assert run.events[-1].payload["end_state"]["orders"]["123"]["status"] == "cancelled"
    assert out.counts["writes"] == 1 and out.counts["writes_matched"] == 1
    assert out.counts["reads"] == 1 and out.counts["reads_same"] == 1


def test_a_write_whose_effect_differs_does_not_confirm(tmp_path):
    class Misspelt(Toolkit):
        cancelled = "canceled"

    out = do_replay(tmp_path, Misspelt)
    assert not out.confirmed
    assert out.reasons == ["cancel_order write: differs (value)"]
    assert out.counts["writes_matched"] == 0


def test_a_cosmetic_read_difference_still_confirms(tmp_path):
    class Floaty(Toolkit):
        total_as = float

    out = do_replay(tmp_path, Floaty)
    assert out.confirmed, out.reasons
    assert out.counts["reads_cosmetic"] == 1 and out.counts["reads_same"] == 0


def test_a_tool_that_refuses_where_the_real_one_answered_is_a_semantic_miss(tmp_path):
    class Broken(Toolkit):
        def get_order_details(self, order_id):
            raise KeyError(order_id)

    out = do_replay(tmp_path, Broken)
    assert not out.confirmed
    assert out.reasons == ["get_order_details read: ours_refused (error)"]


def test_compare_call_names_every_way_two_answers_part():
    same = call("x", "t", {}, {"a": 1})
    assert replay.compare_call(same, {"a": 1}, None) == replay.SAME
    assert replay.compare_call(same, '{"a": 1}', None) == replay.SAME  # a JSON string is its value
    assert replay.compare_call(same, {"a": 1.0}, None) == replay.COSMETIC
    assert replay.compare_call(same, {"a": 2}, None) == replay.DIFFERS
    assert replay.compare_call(same, None, ToolCallError(**{"class": "not_found_entity"})) == replay.OURS_REFUSED
    refused = call("x", "t", {}, None, error=ToolCallError(**{"class": "not_found_entity"}))
    assert replay.compare_call(refused, {"a": 1}, None) == replay.THEIRS_REFUSED
    assert replay.compare_call(refused, None, ToolCallError(**{"class": "unknown"})) == replay.BOTH_REFUSED


def _checks(out, verdict: str) -> list[dict]:
    return [check for check in out.checks if check["verdict"] == verdict]


def test_a_parted_call_alone_records_its_keys_and_both_answers_whole(tmp_path):
    """The two preview fields keep their 160 characters; the difference record is what a report
    reads the cause off, so it says the types, the keys and the answers themselves. The record is
    written for the calls that parted and for those only, so a replay that agreed all the way writes
    the same bytes it always did."""
    agreed = do_replay(tmp_path / "agreed")
    assert agreed.confirmed and agreed.checks
    assert all("difference" not in check for check in agreed.checks)
    assert all(set(check) == {"tool", "kind", "requestor", "verdict", "verdict_route", "route", "call_id", "ours", "recorded"}
               for check in agreed.checks)

    class Misspelt(Toolkit):
        cancelled = "canceled"

    out = do_replay(tmp_path / "parted", Misspelt)
    parted = _checks(out, replay.DIFFERS)
    assert [check["tool"] for check in parted] == ["cancel_order"]
    difference = parted[0]["difference"]
    assert difference["keys_changed"] == ["status"]
    assert difference["keys_only_ours"] == [] and difference["keys_only_theirs"] == []
    assert difference["type_mismatch"] is False and difference["ours_type"] == "dict"
    assert not difference["ours_truncated"] and not difference["theirs_truncated"]
    assert json.loads(difference["ours"])["status"] == "canceled"
    assert json.loads(difference["theirs"])["status"] == "cancelled"
    assert difference["ours_errored"] is False and difference["ours_error"] == ""
    assert difference["leaf"] == 'status: ours "canceled", recorded "cancelled"', "the leaf, not only the key"

    # 160 characters is not always enough to reach the message inside a refusal, and the message is
    # the whole of what says whose error it was.
    class Broken(Toolkit):
        def get_order_details(self, order_id):
            raise KeyError("x" * 300)

    refused = do_replay(tmp_path / "refused", Broken)
    difference = _checks(refused, replay.OURS_REFUSED)[0]["difference"]
    assert difference["ours_errored"] is True and difference["theirs_errored"] is False
    assert "x" * 300 in difference["ours_error"] and difference["theirs_error"] == ""
    assert difference["ours_type"] == "error" and difference["type_mismatch"] is True


def test_an_answer_too_long_to_keep_is_marked_as_cut_with_the_keys_still_named(tmp_path):
    class Wordy(Toolkit):
        def get_order_details(self, order_id):
            return {"id": order_id, "essay": "x" * 6000}

    out = do_replay(tmp_path, Wordy)
    difference = _checks(out, replay.DIFFERS)[0]["difference"]
    assert difference["ours_truncated"] is True and len(difference["ours"]) == replay.DIFFERENCE_LIMIT
    assert difference["theirs_truncated"] is False
    assert difference["keys_only_ours"] == ["essay"] and difference["keys_only_theirs"] == ["status", "total"]


def test_a_tool_the_user_called_is_routed_under_the_users_name(tmp_path):
    out = do_replay(tmp_path, user_call=True)
    assert out.confirmed, out.reasons
    run = suite.load_run(out.path)
    own = [e for e in run.events if e.type == "tool_call" and e.payload.get("requestor") == "user"]
    assert len(own) == 1 and own[0].payload["name"] == "check_balance"
    assert out.counts["calls"] == 3 and out.counts["reads_same"] == 2


def test_the_verifier_derives_from_the_replayed_run(tmp_path):
    """The whole point: a Run on disk the Verifier stage can read (D91), with the write as an atom."""
    out = do_replay(tmp_path)
    task = Task(id="t1", run_ids=["tr1"])
    verifier = verifier_mod.derive_verifier(task, out.path, write_tools={"cancel_order"})
    writes = [a for a in verifier.atoms if a.target.get("kind") == "write"]
    assert [a.kind for a in writes] == ["required"] and writes[0].target["tool"] == "cancel_order"
    gates = {g.stage: g.passed for g in suite.validate_verifier(
        verifier, out.path, write_tools={"cancel_order"})}
    assert gates["verifier_oracle"] and gates["verifier_empty_run"]



# --- D187: the scoring path is handed what knows the schema's column classes ---

def _kiln_comparer():
    """The comparer the build hands the Runner, over an invented kiln with one exempt column."""
    from kullback.gates.tool_runs import ReplayComparer
    from kullback.runner.records import Column, EntitySchema

    schema = EntitySchema(tables=["firings"], id_patterns={"firings.firing_id": r"^F\d+$"}, columns=[
        Column(table="firings", name="firing_id", **{"class": "hard"}),
        Column(table="firings", name="peak_c", **{"class": "hard"}),
        Column(table="firings", name="logged_at", **{"class": "exempt"})])
    return ReplayComparer(schema)


@pytest.mark.parametrize(
    "ours, verdict, notes",
    [
        ({"firing_id": "F9", "peak_c": 1230, "logged_at": "2031-07-04T18:00:00"}, replay.COSMETIC,
         ["exempt:logged_at"]),
        ({"firing_id": "F9", "peak_c": 900, "logged_at": "2031-01-01T00:00:00"}, replay.DIFFERS,
         ["peak_c: ours 900, recorded 1230"]),
    ],
    ids=["exempt_column_is_forgiven", "hard_column_parts_and_is_named"],
)
def test_a_replayed_call_parts_by_the_column_classes_the_schema_names(ours, verdict, notes):
    recorded = call("x", "t", {}, {"firing_id": "F9", "peak_c": 1230, "logged_at": "2031-01-01T00:00:00"})
    assert replay.compare_call(recorded, ours, None) == replay.DIFFERS
    assert replay.compare_call(recorded, ours, None, comparer=_kiln_comparer()) == verdict
    assert replay.compare_call_notes(recorded, ours, None, comparer=_kiln_comparer()) == (verdict, notes)


# --- D204: a run of consecutive turns of one role is one logical turn ---

def _replay_of(tmp_path, turns, calls=()):
    """One invented Trace replayed over the order world, so a test can shape its turn order."""
    recording = Trace(trace_id="tr1", raw_hash="r" * 64, ingest_version="1", source="test",
                      turns=turns, tool_calls=list(calls), raw_ptr=PTR, system_prompt="You are the agent.")
    return replay.replay_trace(recording, router(), workdir=tmp_path / "runs" / "t1", task_id="t1",
                               env_id="env1", write_tools={"cancel_order"})


def _turn(idx, role, content=None, call_ids=()):
    return Turn(idx=idx, role=role, content=content, tool_call_ids=list(call_ids), raw_ptr=PTR)


def test_a_run_of_turns_of_one_role_is_one_turn_and_the_other_role_is_asked_once(tmp_path):
    """The loop asks the user once, so a recording where the user speaks twice has to hand both over
    at once or the cursor stalls on the second and every later turn is a gap. An assistant turn that
    called nothing is not one the loop comes back to, so a second assistant turn beside it is
    absorbed rather than left for an ask that never comes."""
    out = _replay_of(tmp_path / "users", [
        _turn(0, "assistant", "How can I help?"),
        _turn(1, "user", "Cancel order 123."),
        _turn(2, "user", "It is the delivered one."),
        _turn(3, "assistant", None, ["c2"]),
        _turn(4, "tool", '{"id": "123"}', ["c2"]),
        _turn(5, "assistant", "Order 123 is cancelled."),
        _turn(6, "user", "Thanks."),
    ], [call("c2", "cancel_order", {"order_id": "123", "reason": "requested"},
             {"id": "123", "status": "cancelled", "total": 25})])
    assert out.confirmed, out.reasons
    assert out.counts["gaps"] == 0
    assert out.counts["absorbed_user_runs"] == 1 and out.counts["absorbed_turns"] == 1
    run = suite.load_run(out.path)
    spoken = [e.payload["text"] for e in run.events if e.type == "user_turn"]
    assert spoken == ["Cancel order 123.\nIt is the delivered one.", "Thanks."]
    assert out.counts["writes"] == 1 and out.counts["writes_matched"] == 1, "the write after the run replays"

    out = _replay_of(tmp_path / "assistants", [
        _turn(0, "assistant", "Hello."),
        _turn(1, "assistant", "How can I help?"),
        _turn(2, "user", "Nothing, thanks."),
    ])
    assert out.confirmed, out.reasons
    assert out.counts["absorbed_model_runs"] == 1 and out.counts["gaps"] == 0
    run = suite.load_run(out.path)
    said = [e.payload["reply"]["content"] for e in run.events if e.type == "model_call"]
    assert said[0] == "Hello.\nHow can I help?"
    assert len([e for e in run.events if e.type == "user_turn"]) == 1


def test_a_tool_a_user_turn_called_inside_a_run_is_routed_compared_and_named_a_user_call(tmp_path):
    """The user's own call is part of what the assistant answers next, so it has to reach the
    Environment before the next assistant ask, and it is scored like any other recorded call."""
    out = _replay_of(tmp_path / "same", [
        _turn(0, "assistant", "How can I help?"),
        _turn(1, "user", "What is my balance?", ["u1"]),
        _turn(2, "tool", '{"balance": 10}', ["u1"]),
        _turn(3, "user", "Then cancel order 123."),
        _turn(4, "assistant", None, ["c2"]),
        _turn(5, "tool", '{"id": "123"}', ["c2"]),
        _turn(6, "assistant", "Order 123 is cancelled."),
        _turn(7, "user", "Thanks."),
    ], [call("u1", "check_balance", {}, {"balance": 10}, requestor="user"),
        call("c2", "cancel_order", {"order_id": "123", "reason": "requested"},
             {"id": "123", "status": "cancelled", "total": 25})])
    assert out.confirmed, out.reasons
    assert out.counts["absorbed_user_runs"] == 1 and out.counts["gaps"] == 0
    own = [c for c in out.checks if c["requestor"] == "user"]
    assert [(c["tool"], c["verdict"]) for c in own] == [("check_balance", replay.SAME)]
    run = suite.load_run(out.path)
    called = [e.payload["name"] for e in run.events if e.type == "tool_call"]
    assert called == ["check_balance", "cancel_order"], "the user's call is routed in recorded order"

    parted = _replay_of(tmp_path / "parted", [
        _turn(0, "assistant", "How can I help?"),
        _turn(1, "user", "What is my balance?", ["u1"]),
        _turn(2, "tool", '{"balance": 99}', ["u1"]),
        _turn(3, "user", "Thanks."),
        _turn(4, "assistant", "Anything else?"),
    ], [call("u1", "check_balance", {}, {"balance": 99}, requestor="user")])
    assert parted.reasons == ["check_balance user_call: differs (value)"]


def test_only_a_role_standing_where_the_other_was_due_counts_a_gap_and_then_resyncs(tmp_path):
    """A recorded user turn where an assistant turn was due is a real mismatch, and it is counted
    once: the cursor takes the user run on the next ask instead of stalling on it for good.
    Assistant turns separated by the tool turns of their own calls are not a run: the loop asks
    the model again after each, so absorbing them would swallow the ask the user is owed."""
    out = _replay_of(tmp_path / "gap", [
        _turn(0, "assistant", None, ["c1"]),
        _turn(1, "tool", '{"id": "123"}', ["c1"]),
        _turn(2, "user", "Any update?"),
        _turn(3, "assistant", "It is delivered."),
        _turn(4, "user", "Thanks."),
    ], [call("c1", "get_order_details", {"order_id": "123"},
             {"id": "123", "status": "delivered", "total": 25})])
    assert out.counts["gaps"] == 1
    assert out.reasons == ["1 turn(s) out of order"]
    assert out.counts["absorbed_user_runs"] == 0 and out.counts["absorbed_model_runs"] == 0
    run = suite.load_run(out.path)
    assert [e.payload["text"] for e in run.events if e.type == "user_turn"] == ["Any update?", "Thanks."]

    alternating = _replay_of(tmp_path / "alternating", list(trace().turns), trace().tool_calls)
    assert alternating.confirmed and alternating.reasons == []
    assert alternating.counts["absorbed_user_runs"] == 0 and alternating.counts["absorbed_model_runs"] == 0
    assert alternating.counts["absorbed_turns"] == 0 and alternating.counts["gaps"] == 0
    assert alternating.counts["writes"] == 1 and alternating.counts["reads"] == 1


# --- D217: a cosmetic verdict says which way it was reached, and states are not cosmetic ---

def _kiln_state_comparer(judge=None):
    """A comparer over an invented kiln whose report is prose and whose states are named."""
    from kullback.gates.tool_runs import ReplayComparer
    from kullback.runner.records import Column, EntitySchema

    schema = EntitySchema(tables=["firings"], id_patterns={"firings.firing_id": r"^F\d+$"}, columns=[
        Column(table="firings", name="firing_id", **{"class": "hard"}),
        Column(table="firings", name="damper", **{"class": "hard"},
               vocabulary=["closed", "open", "vented"]),
        Column(table="firings", name="firing_report", **{"class": "semantic"}),
        Column(table="firings", name="logged_at", **{"class": "exempt"})])
    return ReplayComparer(schema, judge=judge)


_EQUIVALENT = lambda column, ours, theirs: {"verdict": "equivalent"}  # noqa: E731
_NOT_EQUIVALENT = lambda column, ours, theirs: {"verdict": "not_equivalent"}  # noqa: E731
_PROSE = {"firing_id": "F9", "firing_report": "the damper is open"}
_PROSE_OURS = {"firing_id": "F9", "firing_report": "open damper, that is"}


@pytest.mark.parametrize(
    "recorded, ours, comparer, verdict, route",
    [
        ({"a": 1}, {"a": 1}, None, replay.SAME, replay.BY_BYTES),
        ({"a": 1}, {"a": 1.0}, None, replay.COSMETIC, replay.BY_CANONICAL),
        ({"firing_id": "F9", "firing_report": "damper open"}, {"firing_report": "  damper   open ", "firing_id": "F9"},
         _kiln_state_comparer, replay.COSMETIC, replay.BY_CANONICAL),
        ({"firing_id": "F9", "logged_at": "2031-01-01T00:00:00"}, {"firing_id": "F9", "logged_at": "2031-07-04T18:00:00"},
         _kiln_state_comparer, replay.COSMETIC, replay.BY_EXEMPT),
        (_PROSE, _PROSE_OURS, lambda: _kiln_state_comparer(_EQUIVALENT), replay.COSMETIC, replay.BY_JUDGE),
        (_PROSE, _PROSE_OURS, lambda: _kiln_state_comparer(_NOT_EQUIVALENT), replay.DIFFERS, replay.BY_JUDGE),
        ({"firing_id": "F9", "logged_at": "2031-01-01T00:00:00"}, {"firing_id": "F9"},
         _kiln_state_comparer, replay.DIFFERS, replay.BY_PRESENCE),
    ],
    ids=["same_bytes", "same_canonical_form", "whitespace_and_key_order", "forgiven_column", "judged_equivalent",
         "judged_different_not_by_the_columns", "value_only_one_side_answers"],
)
def test_each_agreement_names_the_route_that_decided_it(recorded, ours, comparer, verdict, route):
    extra = {"comparer": comparer()} if comparer else {}
    got_verdict, _notes, got_route = replay.compare_call_route(call("x", "t", {}, recorded), ours, None, **extra)
    assert (got_verdict, got_route) == (verdict, route)


def test_a_semantic_column_nobody_settled_parts_and_says_so_rather_than_agreeing():
    """D219: an unresolved pair is not agreement. With no judge and no table nothing settled these
    two sentences, so the check parts under its own route instead of counting as a replayed call."""
    prose = call("x", "t", {}, {"firing_id": "F9", "firing_report": "the damper is open"})
    verdict, notes, route = replay.compare_call_route(
        prose, {"firing_id": "F9", "firing_report": "open damper, that is"}, None,
        comparer=_kiln_state_comparer())
    assert verdict == replay.DIFFERS and route == replay.BY_UNRESOLVED
    assert notes[0].startswith("unresolved:firing_report")


def test_an_answer_naming_a_different_state_parts_and_names_the_states_on_each_side():
    recorded = call("x", "t", {}, {"firing_id": "F9", "firing_report": "damper open, kiln loaded"})
    ours = {"firing_id": "F9", "firing_report": "damper closed, kiln loaded"}
    verdict, notes, route = replay.compare_call_route(recorded, ours, None,
                                                      comparer=_kiln_state_comparer())
    assert verdict == replay.DIFFERS and route == replay.BY_TOKEN_SET
    assert notes == ["token_set:firing_report: ours [closed], recorded [open]"]


def test_the_replay_counts_how_much_agreement_rested_on_each_route(tmp_path):
    class Floaty(Toolkit):
        total_as = float

    out = do_replay(tmp_path, Floaty)
    assert out.confirmed, out.reasons
    assert out.counts["cosmetic_by_canonical"] == 1
    assert out.counts["cosmetic_by_exempt"] == 0 and out.counts["cosmetic_by_judge"] == 0
    assert out.counts["differs_by_token_set"] == 0 and out.counts["differs_by_presence"] == 0
    assert all(check["verdict_route"] for check in out.checks)


STRAY = "\x00unattributed"


class _WorldRouter:
    def __init__(self, state):
        self._world = state

    def world(self):
        return self._world


def _effect_row(unattributed, after="cancelled"):
    return {"tool": "cancel_order", "table": "orders", "row": "123", "path": "status",
            "before": "delivered", "after": after, "named": True, "ambiguous": False,
            "unattributed": unattributed, "formulas": []}


def _effect_world():
    return {"orders": {"123": {"id": "123", "status": "cancelled", "total": 25}}}


def _check_effects(effects):
    scored = replay.ScoredRouter(_WorldRouter(_effect_world()), deque(), write_tools={"cancel_order"},
                                 effects=effects)
    recorded = call("c2", "cancel_order", {"order_id": "123"},
                    {"id": "123", "status": "cancelled", "total": 25})
    check = {"tool": "cancel_order", "kind": "write", "call_id": "c2"}
    scored._check_effects(check, recorded, SimpleNamespace(error=None))
    return scored, check


def test_an_unattributed_row_is_counted_apart_and_never_fails_its_call(tmp_path):
    """Only a row that is attributed to the call, by a flag that is exactly True, can fail it; an
    unattributed miss leaves no stale row for downstream blame."""
    cases = [
        # rows, effect_checks, failures, unattributed, stale
        ([_effect_row(True, after="shipped")], 0, None, 1, 0),
        ([_effect_row(False, after="shipped")], 1, 1, None, None),
        ([_effect_row(False, after="shipped"), _effect_row(True, after="shipped")], 1, 1, 1, None),
        ([_effect_row("yes", after="shipped")], None, 1, None, None),
    ]
    for rows, checks, failures, unattributed, stale in cases:
        scored, check = _check_effects({"c2": rows})
        if checks is not None:
            assert check["effect_checks"] == checks
        if failures is None:
            assert check.get("effect_failures") is None
        else:
            assert len(check["effect_failures"]) == failures
        assert check.get("effect_failures_total") == failures
        assert check.get("unattributed_effects") == unattributed
        if stale is not None:
            assert len(scored.stale) == stale

    out = replay.replay_trace(trace(), router(), workdir=tmp_path / "runs" / "t8", task_id="t8",
                              env_id="env1", write_tools={"cancel_order"},
                              effects={"c2": [_effect_row(False)],
                                       STRAY: [_effect_row(True, after="shipped")]})
    assert out.confirmed, out.reasons
    assert out.counts["effect_checks"] == 1
    assert out.counts["effect_failures"] == 0
    assert out.counts["effect_unattributed"] == 1


def test_a_stray_bucket_colliding_with_a_call_id_still_fails_nothing(tmp_path):
    cid = effects_mod.UNATTRIBUTED
    effect = effects_mod.WriteEffect(tool="cancel_order", call_id=cid, trace_id="tr1", columns=[
        effects_mod.EffectColumn(table="orders", row_id="123", path="status", before="delivered",
                                 after="shipped", named=True, checked=True, unattributed=False),
        effects_mod.EffectColumn(table="orders", row_id="123", path="total", before=25,
                                 after=26, named=True, checked=False, unattributed=True)])
    assert effects_mod._stray_key({cid: [effect]}) == cid + "\x00"
    evidence = effects_mod.replay_evidence({cid: [effect]})
    assert evidence[cid] and all(row.get("unattributed") is not True for row in evidence[cid])
    assert evidence[cid + "\x00"] and all(row.get("unattributed") is True
                                             for row in evidence[cid + "\x00"])
    evidence[cid].append(_effect_row(True, after="shipped"))
    evidence[cid].append(_effect_row("yes", after="shipped"))
    recorded = call(cid, "cancel_order", {"order_id": "123"},
                    {"id": "123", "status": "cancelled", "total": 25})
    scored = replay.ScoredRouter(_WorldRouter(_effect_world()), deque(), write_tools={"cancel_order"},
                                 effects=evidence)
    check = {"tool": "cancel_order", "kind": "write", "call_id": cid}
    scored._check_effects(check, recorded, SimpleNamespace(error=None))
    assert check["effect_checks"] == 2
    assert check["effect_failures_total"] == 2
    assert check["unattributed_effects"] == 1
    assert len(scored.stale) == 2
    assert replay._count_unattributed_effects(evidence) == 2

    # the same collision replayed: the stray is scored apart and the later read is blamed downstream
    colliding = Trace(trace_id="tr9", raw_hash="r" * 64, ingest_version="1", source="test", turns=[
        Turn(idx=0, role="assistant", content="Hi! How can I help you today?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Please cancel order 123.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"id": "123"}', tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=[cid], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"id": "123"}', tool_call_ids=[cid], raw_ptr=PTR),
        Turn(idx=6, role="assistant", content=None, tool_call_ids=["c3"], raw_ptr=PTR),
        Turn(idx=7, role="tool", content='{"id": "123"}', tool_call_ids=["c3"], raw_ptr=PTR),
        Turn(idx=8, role="assistant", content="Order 123 is cancelled. Anything else?", raw_ptr=PTR),
        Turn(idx=9, role="user", content="No, thanks. ###STOP###", raw_ptr=PTR)],
        tool_calls=[
            call("c1", "get_order_details", {"order_id": "123"},
                 {"id": "123", "status": "delivered", "total": 25}),
            call(cid, "cancel_order", {"order_id": "123", "reason": "requested"},
                 {"id": "123", "status": "cancelled", "total": 25}),
            call("c3", "get_order_details", {"order_id": "123"},
                 {"id": "123", "status": "shipped", "total": 25})],
        raw_ptr=PTR, system_prompt="You are the agent.")
    effect = effects_mod.WriteEffect(tool="cancel_order", call_id=cid, trace_id="tr9", columns=[
        effects_mod.EffectColumn(table="orders", row_id="123", path="status", before="delivered",
                                 after="shipped", named=True, checked=True, unattributed=False),
        effects_mod.EffectColumn(table="orders", row_id="123", path="total", before=25,
                                 after=26, named=True, checked=False, unattributed=True)])
    evidence = effects_mod.replay_evidence({cid: [effect]})
    out = replay.replay_trace(colliding, router(), workdir=tmp_path / "runs" / "t9", task_id="t9",
                              env_id="env1", write_tools={"cancel_order"}, effects=evidence)
    assert not out.confirmed
    assert out.counts["gaps"] == 0
    assert out.counts["effect_checks"] == 1
    assert out.counts["effect_failures"] == 1
    assert out.counts["effect_unattributed"] == 1
    assert out.counts["effects_downstream"] == 1
    assert "cancel_order write: effect orders.status the column did not move" in out.reasons
    reread = [check for check in out.checks if check.get("call_id") == "c3"]
    assert len(reread) == 1 and reread[0].get("downstream_of") == cid
