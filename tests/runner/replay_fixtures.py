"""The replayed Trace every test of a Reference Run starts from: one cancel_order Task in a
tiny order world, its Toolkit, and `do_replay` writing the Run to disk.

It lives beside the replay tests because `replay.replay_trace` is what builds it, and the
gates and Builder tests that rule on a replayed Run import it from here rather than each
standing up a second copy of the same world.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from kullback.runner import replay
from kullback.runner.records import RawPtr, ToolCall, ToolSig, Trace, Turn
from kullback.runner.route import Router

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


def call(call_id, name, args, result, error=None, requestor="assistant") -> ToolCall:
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
        call("c1", "get_order_details", {"order_id": "123"}, {"id": "123", "status": "delivered", "total": 25}),
        call("c2", "cancel_order", {"order_id": "123", "reason": "requested"},
              {"id": "123", "status": "cancelled", "total": 25}),
    ]
    if user_call:
        calls.append(call("c3", "check_balance", {}, {"balance": 10}, requestor="user"))
    return Trace(trace_id="tr1", raw_hash="r" * 64, ingest_version="1", source="test", turns=turns,
                 tool_calls=calls, raw_ptr=PTR, system_prompt="You are the agent.")


def router(toolkit_class=Toolkit) -> Router:
    return Router(env_tools_module=toolkit_class(world()), starting_state=world(), tool_sigs=sigs())


def do_replay(tmp_path: Path, toolkit_class=Toolkit, user_call: bool = False) -> replay.Replay:
    return replay.replay_trace(trace(user_call), router(toolkit_class), workdir=tmp_path / "runs" / "t1",
                               task_id="t1", env_id="env1", write_tools={"cancel_order"})


def events(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if '"type"' in line]
