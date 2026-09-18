"""Every tool call is a transaction, and a crashed body is its own outcome (G27).

These tests need the body-transactions frozen patch applied. Without it they skip: the
Router takes no snapshot and knows no body fault, so every assertion here would fail for the
wrong reason. Run with the patch: `git apply docs/frozen-patches/body-transactions.patch`.
"""

from __future__ import annotations

import json
import types
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from kullback.runner import loop as loop_mod
from kullback.runner import route as route_mod
from kullback.runner.records import ErrorClass, ErrorShape, ToolSig, Verifier, as_dict
from kullback.runner.route import Router
from kullback.runner.state import StateView
from kullback.runner.verdict import _env_marks, verdict

HAS_PATCH = hasattr(route_mod, "_snapshot_world") and "body_fault" in get_args(ErrorClass)
pytestmark = pytest.mark.skipif(not HAS_PATCH, reason="needs the body-transactions frozen patch applied")


class Gadget(BaseModel):
    id: str
    status: str
    total: int = 0


class GadgetDB(BaseModel):
    gadgets: dict[str, Gadget] = {}
    spares: dict[str, Gadget] = {}


class GadgetToolkit:
    """A generated toolkit: bodies that read and write self.db rows by attribute."""

    def __init__(self, db: dict):
        self.db = GadgetDB.model_validate(db)

    def read_gadget(self, gadget_id):
        return self.db.gadgets[gadget_id].model_dump()

    def shelve_two_then_crash(self, gadget_id, spare_id):
        self.db.gadgets[gadget_id].status = "shelved"
        self.db.spares[spare_id].status = "taken"
        raise RuntimeError(f"boom on {gadget_id}")

    def shelve_gadget(self, gadget_id):
        self.db.gadgets[gadget_id].status = "shelved"
        return self.db.gadgets[gadget_id].model_dump()


def fault_module() -> types.ModuleType:
    """Module tools with a dict db, writing both the toolkit world and the StateView."""

    def read_gadget(state, gadget_id):
        row = state.row("gadgets", gadget_id)
        if row is None:
            raise KeyError(gadget_id)
        return dict(row)

    def relocate(state, gadget_id, spare_id):
        module.db["gadgets"][gadget_id] = dict(module.db["gadgets"][gadget_id], status="moved")
        state.put("gadgets", gadget_id, {"status": "moved"})
        module.db["spares"][spare_id] = dict(module.db["spares"][spare_id], status="taken")
        state.put("spares", spare_id, {"status": "taken"})
        raise RuntimeError(f"boom on {gadget_id}")

    def refuse(state, gadget_id):
        state.put("gadgets", gadget_id, {"status": "moved"})
        module.db["gadgets"][gadget_id] = dict(module.db["gadgets"][gadget_id], status="moved")
        raise ValueError("no such gadget")

    def crash_attr(state, gadget_id):  # noqa: ARG001 - the args shape is the tool surface
        raise AttributeError(f"no attribute on {gadget_id}")

    module = types.ModuleType("fault_tools")
    module.db = {
        "gadgets": {"g1": {"id": "g1", "status": "new"}},
        "spares": {"s1": {"id": "s1", "status": "new"}},
    }
    module.read_gadget = read_gadget
    module.relocate = relocate
    module.refuse = refuse
    module.crash_attr = crash_attr
    return module


def shared_world() -> dict:
    return {"gadgets": {"g1": {"id": "g1", "status": "new"}},
            "spares": {"s1": {"id": "s1", "status": "new"}}}


def sigs() -> list[ToolSig]:
    return [
        ToolSig(
            name="read_gadget",
            error_shapes=[
                ErrorShape(class_="not_found_entity", count=1,
                           sample_payload={"error": "missing"}, encoding="json"),
            ],
        ),
        ToolSig(name="relocate"),
        ToolSig(name="refuse"),
        ToolSig(name="crash_attr"),
    ]


def make_router(module: Any = None, **kwargs: Any) -> Router:
    state = kwargs.pop("state", None) or StateView(shared=json.loads(json.dumps(shared_world())))
    kwargs.setdefault("env_tools_module", module if module is not None else fault_module())
    kwargs.setdefault("tool_sigs", sigs())
    return Router(starting_state=state, **kwargs)


def test_crash_rolls_back_both_stores():
    router = make_router()
    before_db = json.loads(json.dumps(router.tools.db))
    before_shared = json.loads(json.dumps(router.state.shared))
    out = router.route("relocate", {"gadget_id": "g1", "spare_id": "s1"})
    assert out.error is not None
    assert router.tools.db == before_db
    assert router.state.shared == before_shared


def test_later_call_sees_the_unmoved_world():
    router = make_router()
    router.route("relocate", {"gadget_id": "g1", "spare_id": "s1"})
    out = router.route("read_gadget", {"gadget_id": "g1"})
    assert out.error is None
    assert out.result == {"id": "g1", "status": "new"}


def test_pydantic_toolkit_rolls_back_and_keeps_row_classes():
    toolkit = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}},
                             "spares": {"s1": {"id": "s1", "status": "new"}}})
    router = make_router(module=toolkit, tool_sigs=[ToolSig(name="shelve_two_then_crash"),
                                                    ToolSig(name="read_gadget"),
                                                    ToolSig(name="shelve_gadget")])
    out = router.route("shelve_two_then_crash", {"gadget_id": "g1", "spare_id": "s1"})
    assert out.error is not None and out.error.class_ == "body_fault"
    assert toolkit.db.gadgets["g1"].status == "new"
    assert toolkit.db.spares["s1"].status == "new"
    assert isinstance(toolkit.db.gadgets["g1"], Gadget)
    follow = router.route("read_gadget", {"gadget_id": "g1"})
    assert follow.result == {"id": "g1", "status": "new", "total": 0}


def test_large_world_rolls_back():
    count = 3000
    db = {"gadgets": {f"g{i}": {"id": f"g{i}", "status": "new"} for i in range(count)}}
    module = fault_module()
    module.db = dict(db)
    router = make_router(module=module, state=StateView(shared=json.loads(json.dumps(db))),
                         tool_sigs=[ToolSig(name="relocate")])

    def relocate(state, gadget_id, spare_id):  # noqa: ARG001
        state.put("gadgets", "g0", {"status": "moved"})
        module.db["gadgets"]["g0"] = {"id": "g0", "status": "moved"}
        raise RuntimeError("boom")

    module.relocate = relocate
    out = router.route("relocate", {"gadget_id": "g0", "spare_id": "s0"})
    assert out.error is not None
    assert module.db["gadgets"]["g0"] == {"id": "g0", "status": "new"}
    assert router.state.shared["gadgets"]["g0"] == {"id": "g0", "status": "new"}


def test_value_error_reaches_candidate_as_today_and_changes_nothing():
    router = make_router()
    out = router.route("refuse", {"gadget_id": "g1"})
    assert out.route == "code"
    assert out.result == "ValueError: no such gadget"
    assert as_dict(out.error) == {"class": "business_error", "payload": "ValueError: no such gadget",
                                  "encoding": "text", "classified_by": "code", "raw_ptr": None}
    assert out.overlay_miss is None
    assert router.tools.db["gadgets"]["g1"] == {"id": "g1", "status": "new"}
    assert router.state.shared["gadgets"]["g1"] == {"id": "g1", "status": "new"}


def test_key_error_mapping_is_unchanged():
    router = make_router()
    out = router.route("read_gadget", {"gadget_id": "zzz"})
    assert out.error is not None and out.error.class_ == "not_found_entity"
    assert out.result == {"error": "missing"}


def test_attribute_error_is_a_body_fault():
    router = make_router()
    out = router.route("crash_attr", {"gadget_id": "g1"})
    assert out.route == "code"
    assert out.result == "unavailable"
    assert "no attribute" not in json.dumps(out.result)
    assert out.error is not None and out.error.class_ == "body_fault"
    assert out.error.payload == {"tool": "crash_attr", "fault": "AttributeError"}
    assert out.overlay_miss == [{"body_fault": "crash_attr", "fault": "AttributeError"}]
    assert router.tools.db["gadgets"]["g1"] == {"id": "g1", "status": "new"}


def test_body_fault_uses_the_customer_json_encoding():
    router = make_router()
    out = router.route("relocate", {"gadget_id": "g1", "spare_id": "s1"})
    assert out.error is not None and out.error.class_ == "body_fault"
    shaped = make_router(tool_sigs=[ToolSig(
        name="relocate",
        error_shapes=[ErrorShape(class_="not_found_entity", count=1,
                                 sample_payload={"error": "missing"}, encoding="json")],
    )]).route("relocate", {"gadget_id": "g1", "spare_id": "s1"})
    assert shaped.result == {"error": "unavailable", "class": "body_fault"}
    assert "boom" not in json.dumps(shaped.result)


def test_body_fault_marks_the_environment_not_the_candidate():
    router = make_router()
    state = loop_mod.new_run_state("r1")
    call = types.SimpleNamespace(id="c1", name="crash_attr", arguments={"gadget_id": "g1"})
    loop_mod._tool_call(state, call, router)
    result_event = next(event for event in state.run.events if event.type == "tool_result")
    assert result_event.payload["overlay_miss"] == [{"body_fault": "crash_attr",
                                                     "fault": "AttributeError"}]
    assert result_event.payload["error"]["class"] == "body_fault"
    assert "env_mark:overlay_miss" in _env_marks(state.run, ())
    done = verdict(state.run, Verifier(task_id="t", atoms=[]))
    assert done.passed is True


def test_successful_write_is_byte_identical_to_a_direct_call():
    toolkit = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
    router = make_router(module=toolkit, tool_sigs=[ToolSig(name="shelve_gadget")])
    expected = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
    direct = expected.shelve_gadget(gadget_id="g1")
    out = router.route("shelve_gadget", {"gadget_id": "g1"})
    assert out.error is None and out.route == "code"
    assert out.result == direct
    assert toolkit.db.gadgets["g1"].model_dump() == expected.db.gadgets["g1"].model_dump()
