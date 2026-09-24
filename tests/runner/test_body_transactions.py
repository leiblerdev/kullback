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


class PickyRefusal(ValueError):
    """A deliberate looking refusal that is not exactly ValueError: a fault, not an answer."""


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

    def validate_then_crash(self, gadget_id):
        self.db.gadgets[gadget_id].status = "shelved"
        Gadget.model_validate({"id": gadget_id, "status": {"nested": "dict"}})
        return None

    def replace_table_then_crash(self, gadget_id):  # noqa: ARG001
        self.db.gadgets = {gadget_id: {"id": gadget_id, "status": "wrecked"}}
        raise RuntimeError("wrecked")

    def drop_table_then_crash(self):
        del self.db.gadgets
        raise RuntimeError("dropped")


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

    def refuse_subclass(state, gadget_id):
        state.put("gadgets", gadget_id, {"status": "moved"})
        raise PickyRefusal("not this one")

    def decode_crash(state, gadget_id):  # noqa: ARG001
        raise json.JSONDecodeError("bad json", "doc", 0)

    def stamp_then_refuse(state, gadget_id):  # noqa: ARG001
        import datetime
        state.put("meta", "m1", {"seen_at": datetime.datetime(2026, 9, 18, 12, 0, 0)})
        raise ValueError("no")

    def tag_then_refuse(state, gadget_id):  # noqa: ARG001
        state.shared["gadgets"][gadget_id]["tags"].add("planted")
        raise ValueError("no")

    def meddle_then_crash(state, gadget_id):  # noqa: ARG001
        state.add({"gadgets": {"g9": {"id": "g9", "status": "planted"}}}, None)
        state.overlay_misses.append({"table": "gadgets", "id": "g9"})
        module.db["planted"] = {"g9": {"id": "g9"}}
        raise RuntimeError("meddled")

    module = types.ModuleType("fault_tools")
    module.db = {
        "gadgets": {"g1": {"id": "g1", "status": "new"}},
        "spares": {"s1": {"id": "s1", "status": "new"}},
    }
    module.read_gadget = read_gadget
    module.relocate = relocate
    module.refuse = refuse
    module.crash_attr = crash_attr
    module.refuse_subclass = refuse_subclass
    module.decode_crash = decode_crash
    module.meddle_then_crash = meddle_then_crash
    module.stamp_then_refuse = stamp_then_refuse
    module.tag_then_refuse = tag_then_refuse
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
        ToolSig(name="meddle_then_crash"),
        ToolSig(name="refuse_subclass"),
        ToolSig(name="decode_crash"),
        ToolSig(name="stamp_then_refuse"),
        ToolSig(name="tag_then_refuse"),
    ]


def make_router(module: Any = None, **kwargs: Any) -> Router:
    state = kwargs.pop("state", None) or StateView(shared=json.loads(json.dumps(shared_world())))
    kwargs.setdefault("env_tools_module", module if module is not None else fault_module())
    kwargs.setdefault("tool_sigs", sigs())
    return Router(starting_state=state, **kwargs)


def test_a_crashing_body_rolls_back_the_world_and_the_overlay():
    """Both stores, the overlay pins and the miss list roll back, a later call sees the unmoved
    world, and a replaced or dropped table is rebuilt with its row classes."""
    router = make_router()
    before_db = json.loads(json.dumps(router.tools.db))
    before_shared = json.loads(json.dumps(router.state.shared))
    out = router.route("relocate", {"gadget_id": "g1", "spare_id": "s1"})
    assert out.error is not None
    assert router.tools.db == before_db
    assert router.state.shared == before_shared
    out = router.route("read_gadget", {"gadget_id": "g1"})
    assert out.error is None
    assert out.result == {"id": "g1", "status": "new"}

    router = make_router()
    before_overlay = json.loads(json.dumps(router.state.overlay))
    before_misses = list(router.state.overlay_misses)
    before_shared = json.loads(json.dumps(router.state.shared))
    out = router.route("meddle_then_crash", {"gadget_id": "g1"})
    assert out.error is not None and out.error.class_ == "body_fault"
    assert router.state.overlay == before_overlay
    assert router.state.overlay_misses == before_misses
    assert router.state.shared == before_shared
    assert "planted" not in router.tools.db
    assert "g9" not in router.state.shared.get("gadgets", {})

    for tool, args in (("replace_table_then_crash", {"gadget_id": "g1"}), ("drop_table_then_crash", {})):
        toolkit = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
        router = make_router(module=toolkit, tool_sigs=[ToolSig(name=tool), ToolSig(name="read_gadget")])
        out = router.route(tool, args)
        assert out.error is not None and out.error.class_ == "body_fault"
        assert toolkit.db.gadgets["g1"].status == "new"
        assert isinstance(toolkit.db.gadgets["g1"], Gadget)
        assert router.route("read_gadget", {"gadget_id": "g1"}).result["status"] == "new"


def test_a_rolled_back_pydantic_world_keeps_its_row_classes():
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


def test_a_value_error_reaches_the_candidate_and_changes_nothing():
    """Exactly ValueError is the customer's refusal; a KeyError still maps to the mined error shape."""
    router = make_router()
    out = router.route("refuse", {"gadget_id": "g1"})
    assert out.route == "code"
    assert out.result == "ValueError: no such gadget"
    assert as_dict(out.error) == {"class": "business_error", "payload": "ValueError: no such gadget",
                                  "encoding": "text", "classified_by": "code", "raw_ptr": None}
    assert out.overlay_miss is None
    assert router.tools.db["gadgets"]["g1"] == {"id": "g1", "status": "new"}
    assert router.state.shared["gadgets"]["g1"] == {"id": "g1", "status": "new"}
    out = router.route("read_gadget", {"gadget_id": "zzz"})
    assert out.error is not None and out.error.class_ == "not_found_entity"
    assert out.result == {"error": "missing"}


def test_an_attribute_error_is_a_body_fault_not_an_answer():
    """A ValueError subclass (a refusal subclass, a decode error, a validation failure) is a fault too,
    and its writes roll back."""
    router = make_router()
    out = router.route("crash_attr", {"gadget_id": "g1"})
    assert out.route == "code"
    assert out.result == "unavailable"
    assert "no attribute" not in json.dumps(out.result)
    assert out.error is not None and out.error.class_ == "body_fault"
    # F57: the record names the tool and the fault, and keeps the message and the body line that raised.
    assert {key: out.error.payload[key] for key in ("tool", "fault")} == {"tool": "crash_attr", "fault": "AttributeError"}
    assert out.error.payload["message"] == "AttributeError: no attribute on g1"
    assert isinstance(out.error.payload["line"], int)
    assert out.overlay_miss == [{"body_fault": "crash_attr", "fault": "AttributeError"}]
    assert router.tools.db["gadgets"]["g1"] == {"id": "g1", "status": "new"}
    for name, fault in (("refuse_subclass", "PickyRefusal"), ("decode_crash", "JSONDecodeError")):
        router = make_router()
        out = router.route(name, {"gadget_id": "g1"})
        assert out.error is not None and out.error.class_ == "body_fault"
        assert out.result == "unavailable"
        assert {key: out.error.payload[key] for key in ("tool", "fault")} == {"tool": name, "fault": fault}
        assert out.error.payload["message"].startswith(f"{fault}: ")
        assert isinstance(out.error.payload["line"], int)
        assert router.state.shared["gadgets"]["g1"] == {"id": "g1", "status": "new"}
    toolkit = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
    router = make_router(module=toolkit, tool_sigs=[ToolSig(name="validate_then_crash"),
                                                    ToolSig(name="read_gadget")])
    out = router.route("validate_then_crash", {"gadget_id": "g1"})
    assert out.error is not None and out.error.class_ == "body_fault"
    fault = out.error.payload
    assert {key: fault[key] for key in ("tool", "fault")} == {"tool": "validate_then_crash", "fault": "ValidationError"}
    assert fault["message"].startswith("ValidationError: ")
    # The line is the body's own, not the library frame the error was raised in.
    assert fault["code"] == 'Gadget.model_validate({"id": gadget_id, "status": {"nested": "dict"}})'
    assert toolkit.db.gadgets["g1"].status == "new"
    assert router.route("read_gadget", {"gadget_id": "g1"}).result["status"] == "new"


def test_a_body_fault_is_encoded_like_the_customers_errors():
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


def test_rollback_restores_values_json_cannot_carry():
    """The deep copy fallback restores values JSON cannot carry and shares nothing with the live world."""
    import datetime
    first = datetime.datetime(2026, 1, 1, 0, 0, 0)
    router = make_router(state=StateView(shared={**shared_world(),
                                                 "meta": {"m1": {"seen_at": first}}}))
    out = router.route("stamp_then_refuse", {"gadget_id": "g1"})
    assert out.error is not None and out.error.class_ == "business_error"
    assert router.state.shared["meta"]["m1"] == {"seen_at": first}
    router = make_router(state=StateView(shared={**shared_world(),
                                                 "gadgets": {"g1": {"id": "g1", "status": "new",
                                                                         "tags": {"shelf"}}}}))
    out = router.route("tag_then_refuse", {"gadget_id": "g1"})
    assert out.error is not None and out.error.class_ == "business_error"
    assert router.state.shared["gadgets"]["g1"]["tags"] == {"shelf"}


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


def test_a_successful_write_matches_a_direct_call_byte_for_byte():
    toolkit = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
    router = make_router(module=toolkit, tool_sigs=[ToolSig(name="shelve_gadget")])
    expected = GadgetToolkit({"gadgets": {"g1": {"id": "g1", "status": "new"}}, "spares": {}})
    direct = expected.shelve_gadget(gadget_id="g1")
    out = router.route("shelve_gadget", {"gadget_id": "g1"})
    assert out.error is None and out.route == "code"
    assert out.result == direct
    assert toolkit.db.gadgets["g1"].model_dump() == expected.db.gadgets["g1"].model_dump()


@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
def test_model_snapshot_fallback_restores_database_without_serializing(monkeypatch, failure):
    toolkit = GadgetToolkit(shared_world())
    original_db = toolkit.db
    def refuse_serialization(self, *args, **kwargs):
        raise ValueError("serializer unavailable")
    monkeypatch.setattr(GadgetDB, "model_dump_json", refuse_serialization)
    def mutate_then_fail(gadget_id):
        toolkit.db.gadgets[gadget_id].status = "changed"
        toolkit.db.gadgets["added"] = Gadget(id="added", status="changed")
        del toolkit.db.spares["s1"]
        raise failure("failed")
    toolkit.mutate_then_fail = mutate_then_fail
    router = make_router(module=toolkit, tool_sigs=[ToolSig(name="mutate_then_fail"), ToolSig(name="read_gadget")])
    out = router.route("mutate_then_fail", {"gadget_id": "g1"})
    assert out.error is not None
    assert out.error.class_ == ("business_error" if failure is ValueError else "body_fault")
    assert toolkit.db is original_db
    assert toolkit.db.gadgets["g1"].status == "new"
    assert "added" not in toolkit.db.gadgets
    assert toolkit.db.spares["s1"].status == "new"
    assert isinstance(toolkit.db.gadgets["g1"], Gadget)
    assert router.route("read_gadget", {"gadget_id": "g1"}).result == {"id": "g1", "status": "new", "total": 0}
