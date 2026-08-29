"""replay.py turns a Trace into the Reference Run the Verifier consumes, scored call by call (D108)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from harness.builder import build as build_module
from harness.builder import verifier as verifier_mod
from harness.runner import replay
from harness.runner.route import Router
from harness.shared.records import RawPtr, Task, ToolCall, ToolCallError, ToolSig, Trace, Turn

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


class Order(BaseModel):
    id: str
    status: str
    total: int = 0


class DB(BaseModel):
    orders: dict[str, Order] = {}


class Toolkit:
    """A toolkit in the shape compile_env.load_toolkit returns."""

    cancelled = "cancelled"
    total_as = int

    def __init__(self, db: dict):
        self.db = DB.model_validate(db)

    def get_order_details(self, order_id):
        row = self.db.orders.get(order_id)
        if row is None:
            raise KeyError(order_id)
        out = row.model_dump()
        out["total"] = self.total_as(out["total"])
        return out

    def cancel_order(self, order_id, reason="requested"):
        self.db.orders[order_id].status = self.cancelled
        return self.db.orders[order_id].model_dump()

    def check_balance(self):
        return {"balance": 10}


def world() -> dict:
    return {"orders": {"123": {"id": "123", "status": "delivered", "total": 25}}}


def sigs() -> list[ToolSig]:
    return [ToolSig(name="get_order_details", kind="read"), ToolSig(name="cancel_order", kind="write"),
            ToolSig(name="check_balance", kind="read")]


def _call(call_id, name, args, result, error=None, requestor="assistant") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, error=error, requestor=requestor,
                    raw_ptr=PTR, trace_id="tr1")


def trace(user_call: bool = False) -> Trace:
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help you today?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Please cancel order 123.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"id": "123"}', tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=["c2"], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"id": "123"}', tool_call_ids=["c2"], raw_ptr=PTR),
        Turn(idx=6, role="assistant", content="Order 123 is cancelled. Anything else?", raw_ptr=PTR),
        Turn(idx=7, role="user", content="No, thanks. ###STOP###", tool_call_ids=["c3"] if user_call else [],
             raw_ptr=PTR),
    ]
    calls = [
        _call("c1", "get_order_details", {"order_id": "123"}, {"id": "123", "status": "delivered", "total": 25}),
        _call("c2", "cancel_order", {"order_id": "123", "reason": "requested"},
              {"id": "123", "status": "cancelled", "total": 25}),
    ]
    if user_call:
        calls.append(_call("c3", "check_balance", {}, {"balance": 10}, requestor="user"))
    return Trace(trace_id="tr1", raw_hash="r" * 64, ingest_version="1", source="test", turns=turns,
                 tool_calls=calls, raw_ptr=PTR, system_prompt="You are the agent.")


def router(toolkit_class=Toolkit) -> Router:
    return Router(env_tools_module=toolkit_class(world()), starting_state=world(), tool_sigs=sigs())


def do_replay(tmp_path: Path, toolkit_class=Toolkit, user_call: bool = False) -> replay.Replay:
    return replay.replay_trace(trace(user_call), router(toolkit_class), workdir=tmp_path / "runs" / "t1",
                               task_id="t1", env_id="env1", write_tools={"cancel_order"})


def events(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if '"type"' in line]


def test_the_replay_writes_the_reference_run_the_loop_would(tmp_path):
    out = do_replay(tmp_path)
    assert out.confirmed, out.reasons
    assert out.termination_reason == "user_stop"
    run = verifier_mod.load_run(out.path)
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
    same = _call("x", "t", {}, {"a": 1})
    assert replay.compare_call(same, {"a": 1}, None) == replay.SAME
    assert replay.compare_call(same, '{"a": 1}', None) == replay.SAME  # a JSON string is its value
    assert replay.compare_call(same, {"a": 1.0}, None) == replay.COSMETIC
    assert replay.compare_call(same, {"a": 2}, None) == replay.DIFFERS
    assert replay.compare_call(same, None, ToolCallError(**{"class": "not_found_entity"})) == replay.OURS_REFUSED
    refused = _call("x", "t", {}, None, error=ToolCallError(**{"class": "not_found_entity"}))
    assert replay.compare_call(refused, {"a": 1}, None) == replay.THEIRS_REFUSED
    assert replay.compare_call(refused, None, ToolCallError(**{"class": "unknown"})) == replay.BOTH_REFUSED


def test_a_tool_the_user_called_is_routed_under_the_users_name(tmp_path):
    out = do_replay(tmp_path, user_call=True)
    assert out.confirmed, out.reasons
    run = verifier_mod.load_run(out.path)
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
    gates = {g.stage: g.passed for g in verifier_mod.validate_verifier(
        verifier, out.path, write_tools={"cancel_order"})}
    assert gates["verifier_oracle"] and gates["verifier_empty_run"]


def test_summarize_counts_traces_tasks_and_calls_and_names_the_common_miss(tmp_path):
    good = do_replay(tmp_path / "a").as_dict()
    bad = do_replay(tmp_path / "b", type("Misspelt", (Toolkit,), {"cancelled": "canceled"})).as_dict()
    summary = replay.summarize({"t1": {"tr1": good}, "t2": {"tr1": bad, "tr2": bad}})
    assert summary["traces"] == 3 and summary["confirmed"] == 1
    assert summary["tasks"] == 2 and summary["tasks_confirmed"] == 1
    assert summary["writes"] == 3 and summary["writes_matched"] == 1
    assert replay.unconfirmed_reason({"tr1": bad, "tr2": bad}) == "cancel_order write: differs"
    assert replay.unconfirmed_reason({}) == "no Trace of the Task was replayed"


def test_every_stage_hashes_the_modules_it_delegates_to():
    """R42 for every stage, not only compile_tools: the first live build was served a schema mined before D106."""
    from harness.builder import cluster, compile_env, mine, synth, user_sim
    from harness.runner import replay as replay_mod
    assert build_module._mine_stage().code_version.endswith(build_module._module_hash(mine))
    assert build_module._module_hash(cluster) in build_module._cluster_stage().code_version
    assert build_module._module_hash(user_sim) in build_module._user_rules_stage().code_version
    assert build_module._module_hash(replay_mod) in build_module._replay_stage().code_version
    grown = build_module._state_stage({"users": 10}, 0).code_version
    assert build_module._module_hash(compile_env) in grown and build_module._module_hash(synth) in grown
    assert grown != build_module._state_stage({"users": 20}, 0).code_version
    assert grown == build_module._state_stage({"users": 10}, 0).code_version


# --- the D79 checks the build could not run before (wrong Run by code, second path, probe) ---

def test_the_wrong_run_aims_every_required_write_at_another_entity(tmp_path):
    out = do_replay(tmp_path)
    task = Task(id="t1", run_ids=["tr1"])
    verifier = verifier_mod.derive_verifier(task, out.path, write_tools={"cancel_order"})
    wrong = verifier_mod.wrong_run(verifier, out.path)
    assert wrong is not None and wrong.run_id.endswith(".wrong")
    call = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call.payload["args"]["order_id"] != "123"
    passed, failing = verifier_mod.check_run(verifier, wrong, write_tools={"cancel_order"})
    assert passed is False and failing == "w0"
    # the Reference itself is untouched
    assert verifier_mod.check_run(verifier, out.path, write_tools={"cancel_order"})[0] is True


def test_the_wrong_run_prefers_an_id_the_reference_showed(tmp_path):
    class TwoOrders(Toolkit):
        def get_order_details(self, order_id):
            return {"id": order_id, "status": "delivered", "total": 25, "other": {"order_id": "456"}}

    out = do_replay(tmp_path, TwoOrders)
    verifier = verifier_mod.derive_verifier(Task(id="t1", run_ids=["tr1"]), out.path, write_tools={"cancel_order"})
    wrong = verifier_mod.wrong_run(verifier, out.path)
    call = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call.payload["args"]["order_id"] == "456"


def test_a_verifier_that_requires_nothing_has_no_wrong_run(tmp_path):
    from harness.shared.records import Verifier
    out = do_replay(tmp_path)
    assert verifier_mod.wrong_run(Verifier(task_id="t1", atoms=[]), out.path) is None


def test_the_suite_runs_the_wrong_run_and_the_second_path(tmp_path):
    first = do_replay(tmp_path / "a")
    second = do_replay(tmp_path / "b")
    verifier = verifier_mod.derive_verifier(Task(id="t1", run_ids=["tr1"]), first.path, [second.path],
                                            write_tools={"cancel_order"})
    gates = {g.stage: g for g in verifier_mod.validate_verifier(
        verifier, first.path, write_tools={"cancel_order"}, wrong_run=verifier_mod.wrong_run(verifier, first.path),
        alt_path_run=second.path)}
    assert gates["verifier_wrong_run"].passed and gates["verifier_alt_path"].passed
    assert gates["verifier_loophole"].metrics.get("skipped")  # no model, so not known to be tight


def test_tool_definitions_speak_json_schema():
    sig = ToolSig(name="find_user_id_by_email", kind="read",
                  args_schema={"properties": {"email": {"type": ["str"]}, "n": {"type": ["int", "NoneType"]}},
                               "required": ["email"], "type": "object"})
    [definition] = build_module._tool_definitions([sig])
    assert definition["name"] == "find_user_id_by_email"
    assert definition["parameters"]["properties"]["email"]["type"] == "string"
    assert definition["parameters"]["properties"]["n"]["type"] == ["integer", "null"]
    assert definition["parameters"]["required"] == ["email"]
