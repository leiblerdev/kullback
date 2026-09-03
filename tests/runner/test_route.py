"""Route order (code, recording, stand-in), the D45 error encoding and the D74 overlay lookup."""

from __future__ import annotations

import json
import types

import pytest
from pydantic import BaseModel

from kullback.ai.provider import TestModel
from kullback.runner.records import ErrorShape, OverlayRow, TaskOverlay, ToolSig
from kullback.runner.route import Router, canonical_args, recording, recording_key
from kullback.runner.state import StateView


def tools_module() -> types.ModuleType:
    """A stand-in for the compiled tools.py: functions taking the state view first."""
    module = types.ModuleType("tools_module")

    def get_order_details(state, order_id):
        row = state.row("orders", order_id)
        if row is None:
            raise KeyError(order_id)
        return row

    def cancel_order(state, order_id, reason="requested"):
        row = dict(state.row("orders", order_id) or {})
        row["status"] = "cancelled"
        row["reason"] = reason
        state.shared.setdefault("orders", {})[str(order_id)] = row
        return row

    module.get_order_details = get_order_details
    module.cancel_order = cancel_order
    return module


SHARED = {"orders": {"123": {"id": "123", "status": "delivered", "total": 25}}}


def shared() -> dict:
    return json.loads(json.dumps(SHARED))


class Order(BaseModel):
    id: str
    status: str
    total: int = 0


class DB(BaseModel):
    orders: dict[str, Order] = {}


class Toolkit:
    """The shape compile_env.load_toolkit returns: bodies that read and write self.db (D56)."""

    def __init__(self, db: dict):
        self.db = DB.model_validate(db)

    def get_order_details(self, order_id):
        row = self.db.orders.get(order_id)
        if row is None:
            raise KeyError(order_id)
        return row.model_dump()

    def cancel_order(self, order_id, reason="requested"):
        self.db.orders[order_id].status = "cancelled"
        return self.db.orders[order_id].model_dump()

    def get_db_hash(self):
        """A public helper the way tau2's ToolKitBase has one; it is not one of the customer's tools."""
        return "hash"


def pinned_overlay() -> tuple[TaskOverlay, dict]:
    """The Task's own row: order 123 as its Runs saw it, still pending (D74)."""
    overlay = TaskOverlay(task_id="t1", rows=[OverlayRow(table="orders", id="123", version_hash="v1")])
    return overlay, {"v1": {"id": "123", "status": "pending", "total": 25}}


def sigs() -> list[ToolSig]:
    """The mined tool surface; the first one's observed errors came back as JSON, so ours must too (D45)."""
    return [
        ToolSig(
            name="get_order_details",
            error_shapes=[
                ErrorShape(
                    class_="not_found_entity", count=2, sample_payload={"error": "no such order"}, encoding="json"
                ),
            ],
        ),
        ToolSig(name="cancel_order"),
    ]


def make_router(**kwargs) -> Router:
    state = kwargs.pop("state", None) or StateView(shared=json.loads(json.dumps(SHARED)))
    kwargs.setdefault("env_tools_module", tools_module())
    kwargs.setdefault("tool_sigs", sigs())
    return Router(starting_state=state, **kwargs)


# --- code route ---

def test_code_route_calls_the_tool():
    router = make_router()
    out = router.route("get_order_details", {"order_id": "123"})
    assert out.route == "code"
    assert out.error is None
    assert out.result["status"] == "delivered"
    assert out.assisted is False


def test_overlay_is_read_before_the_shared_world():
    """D74: the Task's own rows win over the shared db.json, on both kinds of tool."""
    overlay, rows = pinned_overlay()
    state = StateView(shared=shared(), overlay=overlay, overlay_rows=rows)
    out = make_router(state=state).route("get_order_details", {"order_id": "123"})
    assert out.result["status"] == "pending"
    assert make_router().route("get_order_details", {"order_id": "123"}).result["status"] == "delivered"

    # the compiled toolkit reads its own db, so the overlay has to reach that db as well
    toolkit = Router(env_tools_module=Toolkit(shared()), starting_state=shared(),
                     overlay=overlay, overlay_rows=rows, tool_sigs=sigs())
    assert toolkit.route("get_order_details", {"order_id": "123"}).result["status"] == "pending"
    assert Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs()).route(
        "get_order_details", {"order_id": "123"}).result["status"] == "delivered"


def test_the_start_state_of_a_toolkit_run_carries_the_overlay():
    """D74, D46: the Starting state a Verdict compares against is the overlay over the shared world."""
    overlay, rows = pinned_overlay()
    router = Router(env_tools_module=Toolkit(shared()), starting_state=shared(),
                    overlay=overlay, overlay_rows=rows, tool_sigs=sigs())
    assert router.start_world["orders"]["123"]["status"] == "pending"
    assert router.world()["orders"]["123"]["status"] == "pending"


def test_the_start_state_is_not_moved_by_a_tool_that_edits_the_row_it_was_handed():
    """D46: the Start state is a snapshot; a write moves the End state and leaves the Start alone."""
    module = types.ModuleType("in_place_tools")

    def cancel_in_place(state, order_id):
        row = state.row("orders", order_id)
        row["status"] = "cancelled"  # a body that edits the row it was handed
        return row

    module.cancel_in_place = cancel_in_place
    router = Router(env_tools_module=module, starting_state=shared(),
                    tool_sigs=[ToolSig(name="cancel_in_place")])
    router.route("cancel_in_place", {"order_id": "123"})
    assert router.world()["orders"]["123"]["status"] == "cancelled"
    assert router.start_world["orders"]["123"]["status"] == "delivered"


def test_a_write_to_an_overlaid_row_reaches_the_end_state():
    """D74: the overlay is laid over the world, so it cannot shadow a write made after it."""
    overlay, rows = pinned_overlay()
    state = StateView(shared=shared(), overlay=overlay, overlay_rows=rows)
    router = make_router(state=state)
    assert router.route("get_order_details", {"order_id": "123"}).result["status"] == "pending"
    router.route("cancel_order", {"order_id": "123"})
    assert router.route("get_order_details", {"order_id": "123"}).result["status"] == "cancelled"
    assert router.world()["orders"]["123"]["status"] == "cancelled"


def test_the_state_view_write_path_lands_in_the_world():
    """A tool body writes one row through the view, and the End state and the next read both see it."""
    state = StateView(shared=shared())
    state.put("orders", "123", {"status": "shipped"})
    assert state.row("orders", "123") == {"id": "123", "status": "shipped", "total": 25}


def test_an_overlay_row_with_no_stored_value_is_recorded_as_a_miss():
    """D74, D88: a missing overlay value is an env mark on the Run, never a silent fallback."""
    overlay = TaskOverlay(task_id="t1", rows=[OverlayRow(table="orders", id="123", version_hash="gone")])
    router = make_router(state=StateView(shared=shared(), overlay=overlay, overlay_rows={}))
    assert router.state.overlay_misses == [{"table": "orders", "id": "123", "version_hash": "gone"}]
    out = router.route("get_order_details", {"order_id": "123"})
    assert out.overlay_miss == router.state.overlay_misses


def test_a_task_whose_overlay_is_whole_carries_no_miss():
    overlay, rows = pinned_overlay()
    router = make_router(state=StateView(shared=shared(), overlay=overlay, overlay_rows=rows))
    assert router.state.overlay_misses == []
    assert router.route("get_order_details", {"order_id": "123"}).overlay_miss is None


def test_an_overlay_given_beside_a_state_view_is_not_dropped():
    """D74: a caller may hand the view and the Task's overlay separately."""
    overlay, rows = pinned_overlay()
    router = Router(env_tools_module=tools_module(), starting_state=StateView(shared=shared()),
                    overlay=overlay, overlay_rows=rows, tool_sigs=sigs())
    assert router.route("get_order_details", {"order_id": "123"}).result["status"] == "pending"


def test_a_public_helper_on_the_toolkit_is_not_a_tool():
    """D45: a hallucinated call to a helper method has no effect, like any tool that does not exist."""
    router = Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs())
    out = router.route("get_db_hash", {})
    assert out.error is not None
    assert out.error.class_ == "tool_not_found"


def test_a_write_tool_changes_the_state_hash():
    router = make_router()
    before = router.state_hash()
    router.route("cancel_order", {"order_id": "123"})
    assert router.state_hash() != before


# --- bad calls, answered in the customer's encoding (D45) ---

def test_unknown_tool_is_answered_not_raised():
    out = make_router().route("teleport_order", {"order_id": "123"})
    assert out.error is not None
    assert out.error.class_ == "tool_not_found"
    assert out.route == "code"
    assert out.assisted is False


def test_invalid_arguments_keep_the_customers_encoding():
    out = make_router().route("get_order_details", {"nope": 1})
    assert out.error.class_ == "invalid_arguments"
    assert out.error.encoding == "json"
    assert isinstance(out.result, dict)


def test_a_missing_entity_is_classed_not_found_entity_by_code():
    out = make_router().route("get_order_details", {"order_id": "999"})
    assert out.error.class_ == "not_found_entity"
    assert out.error.classified_by == "code"


def test_text_encoding_when_the_traces_show_text_errors():
    router = make_router(tool_sigs=[ToolSig(name="get_order_details", error_shapes=[
        ErrorShape(class_="not_found_entity", sample_payload="no such order", encoding="text"),
    ])])
    out = router.route("get_order_details", {"order_id": "999"})
    assert out.error.encoding == "text"
    assert isinstance(out.result, str)


def test_a_python_exception_is_answered_in_the_words_the_corpus_shows_for_its_class():
    """Build 8: 155 unknown-id lookups came back as `KeyError: '#W...'` beside recordings that said 'Order not found'."""
    router = make_router(tool_sigs=[ToolSig(name="get_order_details", error_shapes=[
        ErrorShape(class_="not_found_entity", sample_payload="Error: Order not found", encoding="text"),
    ])])
    out = router.route("get_order_details", {"order_id": "999"})
    assert out.result == "Error: Order not found"
    assert out.error.payload == "Error: Order not found"
    assert out.error.class_ == "not_found_entity"


def test_a_json_corpus_error_is_answered_as_the_corpus_shows_it():
    out = make_router().route("get_order_details", {"order_id": "999"})
    assert out.result == {"error": "no such order"}


def test_a_class_the_corpus_never_showed_keeps_the_exception_text():
    out = make_router().route("get_order_details", {"nope": 1})
    assert out.error.class_ == "invalid_arguments"
    assert "TypeError" in out.result["error"]


def test_an_error_the_body_raised_in_its_own_words_keeps_them():
    class Toolkit:
        def __init__(self, db):
            self.db = db

        def cancel_order(self, order_id):
            raise ValueError("Order is not pending")

    router = make_router(env_tools_module=Toolkit({}), tool_sigs=[ToolSig(name="cancel_order", error_shapes=[
        ErrorShape(class_="business_error", sample_payload="Error: Item not found", encoding="text"),
    ])])
    out = router.route("cancel_order", {"order_id": "123"})
    assert out.error.class_ == "business_error"
    assert out.result == "ValueError: Order is not pending"


def test_encoding_falls_back_to_text_when_no_errors_were_observed():
    out = make_router(tool_sigs=[ToolSig(name="get_order_details")]).route("get_order_details", {"order_id": "999"})
    assert out.error.encoding == "text"


# --- recording route ---

def test_recording_is_an_exact_lookup_on_tool_args_and_pre_state_hash():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "a@b.com"}, state.hash(), {"name": "Ada"})
    router = make_router(state=state, recordings=[entry])
    out = router.route("lookup_user", {"email": "a@b.com"})
    assert out.route == "recording"
    assert out.result == {"name": "Ada"}
    assert out.assisted is False


def test_a_stale_recording_cannot_answer_a_changed_state():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "a@b.com"}, state.hash(), {"name": "Ada"})
    router = make_router(state=state, recordings=[entry])
    router.route("cancel_order", {"order_id": "123"})
    out = router.route("lookup_user", {"email": "a@b.com"})
    assert out.route != "recording"
    assert out.error.class_ == "tool_not_found"


def test_recording_matches_on_canonical_args():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "a@b.com", "n": 25}, state.hash(), "ok")
    router = make_router(state=state, recordings=[entry])
    out = router.route("lookup_user", {"email": " a@b.com ", "n": 25.0})
    assert out.route == "recording"


def test_a_recorded_error_comes_back_as_an_error():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "x"}, state.hash(), None,
                      error={"class": "business_error", "payload": "closed account", "encoding": "text"})
    out = make_router(state=state, recordings=[entry]).route("lookup_user", {"email": "x"})
    assert out.route == "recording"
    assert out.error.class_ == "business_error"


def test_a_write_answered_from_the_recording_lands_in_the_world():
    """A recorded write changes the world, so a later read is not stale and the End state shows it."""
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("refund_order", {"order_id": "123"}, state.hash(), {"status": "refunded"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded", "total": 25}}})
    router = make_router(state=state, recordings=[entry])
    before = router.world()["orders"]["123"]["status"]
    out = router.route("refund_order", {"order_id": "123"})
    assert out.route == "recording"
    assert before == "delivered"
    assert router.world()["orders"]["123"]["status"] == "refunded"
    assert router.route("get_order_details", {"order_id": "123"}).result["status"] == "refunded"


def test_a_recorded_write_is_visible_to_the_next_toolkit_read():
    """The same rule on the real path: a compiled body reads self.db, so the write has to land there."""
    router = Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs())
    entry = recording("refund_order", {"order_id": "123"}, router.state_hash(), {"status": "refunded"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded", "total": 25}}})
    router.recordings = {recording_key("refund_order", {"order_id": "123"}, router.state_hash()): entry}
    assert router.route("refund_order", {"order_id": "123"}).route == "recording"
    assert router.world()["orders"]["123"]["status"] == "refunded"
    assert router.route("get_order_details", {"order_id": "123"}).result["status"] == "refunded"


def test_a_code_write_after_a_recorded_write_is_the_one_the_end_state_shows():
    """D46: the End state is the world as it stands, so the later write wins, not the recorded one."""
    router = Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs())
    entry = recording("refund_order", {"order_id": "123"}, router.state_hash(), {"status": "refunded"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded", "total": 25}}})
    router.recordings = {recording_key("refund_order", {"order_id": "123"}, router.state_hash()): entry}
    assert router.route("refund_order", {"order_id": "123"}).route == "recording"
    assert router.route("cancel_order", {"order_id": "123"}).result["status"] == "cancelled"
    assert router.world()["orders"]["123"]["status"] == "cancelled"


def test_a_recorded_error_writes_nothing():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("refund_order", {"order_id": "123"}, state.hash(), None,
                      error={"class": "business_error", "payload": "already shipped", "encoding": "text"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded"}}})
    router = make_router(state=state, recordings=[entry])
    router.route("refund_order", {"order_id": "123"})
    assert router.world()["orders"]["123"]["status"] == "delivered"


def test_a_recording_without_a_pre_state_hash_never_matches():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    router = make_router(state=state, recordings=[{"tool": "lookup_user", "args": {}, "result": "ok"}])
    assert router.unkeyed_recordings == 1
    assert router.route("lookup_user", {}).route != "recording"


def test_recordings_may_be_given_already_keyed():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    key = recording_key("lookup_user", {"email": "a@b.com"}, state.hash())
    router = make_router(state=state, recordings={key: {"result": "ok"}})
    assert router.route("lookup_user", {"email": "a@b.com"}).result == "ok"


# --- stand-in route ---

def test_the_stand_in_answers_last_and_marks_the_run_assisted():
    """D49: an LLM stand-in is the last route and every Run that uses it is Assisted."""
    model = TestModel(['{"name": "Ada"}'])
    out = make_router(stand_in_model=model).route("lookup_user", {"email": "a@b.com"})
    assert out.route == "llm"
    assert out.assisted is True
    assert out.result == {"name": "Ada"}
    assert len(model.calls) == 1


def test_the_stand_in_is_not_asked_when_code_answers():
    model = TestModel(["never used"])
    out = make_router(stand_in_model=model).route("get_order_details", {"order_id": "123"})
    assert out.route == "code"
    assert model.calls == []


def test_a_non_json_stand_in_reply_comes_back_as_text():
    out = make_router(stand_in_model=TestModel(["the order is pending"])).route("lookup_user", {})
    assert out.result == "the order is pending"
    assert out.assisted is True


# --- canonical args helper ---

@pytest.mark.parametrize("a,b", [
    ({"n": 25}, {"n": 25.0}),
    ({"s": "abc"}, {"s": " abc "}),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
])
def test_canonical_args_agrees_on_the_same_call(a, b):
    assert canonical_args(a) == canonical_args(b)


def test_canonical_args_keeps_real_differences():
    assert canonical_args({"n": 25}) != canonical_args({"n": 26})
    # True == 1 in Python, so the difference has to show in the key, which is where it counts
    assert recording_key("t", {"ok": True}, "s") != recording_key("t", {"ok": 1}, "s")


def test_the_recording_key_folds_the_way_the_customers_rules_fold():
    """D39: keyed under the module defaults, a call the customer's rules call the same is missed."""
    from kullback.runner.canon import CanonRules

    rules = CanonRules(id_patterns={"order_id": r"#?W\d+"}, id_strip_chars="#")
    assert recording_key("t", {"order_id": "#W123"}, "s") != recording_key("t", {"order_id": "W123"}, "s")
    assert recording_key("t", {"order_id": "#W123"}, "s", rules) == \
        recording_key("t", {"order_id": "W123"}, "s", rules)


# --- a synthetic row read through a tool makes the Run assisted (D40, D49, D107) ---

def test_a_code_route_that_returns_a_synthetic_row_is_assisted():
    router = make_router(synthetic_rows=["123"])
    out = router.route("get_order_details", {"order_id": "123"})
    assert out.route == "code" and out.error is None
    assert out.assisted is True


def test_a_code_route_that_touches_no_synthetic_row_is_not_assisted():
    router = make_router(synthetic_rows=["999"])
    assert router.route("get_order_details", {"order_id": "123"}).assisted is False
    assert make_router().route("get_order_details", {"order_id": "123"}).assisted is False
