"""replay.py turns a Trace into the Reference Run the Verifier consumes, scored call by call (D108)."""

from __future__ import annotations

import json

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


def _checks(out, verdict: str) -> list[dict]:
    return [check for check in out.checks if check["verdict"] == verdict]


def test_a_check_that_agreed_carries_no_difference_record(tmp_path):
    """The record is written for the calls that parted and for those only, so a replay that agreed
    all the way writes the same bytes it always did."""
    out = do_replay(tmp_path)
    assert out.confirmed and out.checks
    assert all("difference" not in check for check in out.checks)
    assert all(set(check) == {"tool", "kind", "verdict", "route", "call_id", "ours", "recorded"}
               for check in out.checks)


def test_a_differing_answer_records_which_keys_parted_and_both_answers_whole(tmp_path):
    """The two preview fields keep their 160 characters; the difference record is what a report
    reads the cause off, so it says the types, the keys and the answers themselves."""
    class Misspelt(Toolkit):
        cancelled = "canceled"

    out = do_replay(tmp_path, Misspelt)
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


def test_a_refusal_on_one_side_records_the_error_message_in_full(tmp_path):
    """160 characters is not always enough to reach the message inside a refusal, and the message is
    the whole of what says whose error it was."""
    class Broken(Toolkit):
        def get_order_details(self, order_id):
            raise KeyError("x" * 300)

    out = do_replay(tmp_path, Broken)
    difference = _checks(out, replay.OURS_REFUSED)[0]["difference"]
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


def test_a_column_the_schema_marks_exempt_does_not_part_a_replayed_call():
    recorded = call("x", "t", {}, {"firing_id": "F9", "peak_c": 1230, "logged_at": "2031-01-01T00:00:00"})
    ours = {"firing_id": "F9", "peak_c": 1230, "logged_at": "2031-07-04T18:00:00"}
    assert replay.compare_call(recorded, ours, None) == replay.DIFFERS
    assert replay.compare_call(recorded, ours, None, comparer=_kiln_comparer()) == replay.COSMETIC


def test_a_hard_column_still_parts_a_replayed_call_and_the_note_names_it():
    recorded = call("x", "t", {}, {"firing_id": "F9", "peak_c": 1230, "logged_at": "2031-01-01T00:00:00"})
    ours = {"firing_id": "F9", "peak_c": 900, "logged_at": "2031-01-01T00:00:00"}
    verdict, notes = replay.compare_call_notes(recorded, ours, None, comparer=_kiln_comparer())
    assert verdict == replay.DIFFERS
    assert notes == ["peak_c: ours 900, recorded 1230"]
