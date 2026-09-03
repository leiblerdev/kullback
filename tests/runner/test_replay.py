"""replay.py turns a Trace into the Reference Run the Verifier consumes, scored call by call (D108)."""

from __future__ import annotations

from kullback.examiner import derive as verifier_mod
from kullback.gates import verifier_suite as suite
from kullback.runner import replay
from kullback.runner.records import Task, ToolCallError
from runner.replay_fixtures import Toolkit, call, do_replay, events, sigs, trace, world  # noqa: F401


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
    assert out.reasons == ["cancel_order write: differs"]
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
    assert out.reasons == ["get_order_details read: ours_refused"]


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

