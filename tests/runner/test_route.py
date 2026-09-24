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
    # D46: the Starting state a Verdict compares against is the overlay over the shared world
    assert toolkit.start_world["orders"]["123"]["status"] == "pending"
    assert toolkit.world()["orders"]["123"]["status"] == "pending"


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
    # the view's own write path: one row put through it reads back merged
    view = StateView(shared=shared())
    view.put("orders", "123", {"status": "shipped"})
    assert view.row("orders", "123") == {"id": "123", "status": "shipped", "total": 25}


def test_only_an_overlay_row_with_no_stored_value_is_recorded_as_a_miss():
    """D74, D88: a missing overlay value is an env mark on the Run, never a silent fallback."""
    overlay = TaskOverlay(task_id="t1", rows=[OverlayRow(table="orders", id="123", version_hash="gone")])
    router = make_router(state=StateView(shared=shared(), overlay=overlay, overlay_rows={}))
    assert router.state.overlay_misses == [{"table": "orders", "id": "123", "version_hash": "gone"}]
    out = router.route("get_order_details", {"order_id": "123"})
    assert out.overlay_miss == router.state.overlay_misses
    overlay, rows = pinned_overlay()
    whole = make_router(state=StateView(shared=shared(), overlay=overlay, overlay_rows=rows))
    assert whole.state.overlay_misses == []
    assert whole.route("get_order_details", {"order_id": "123"}).overlay_miss is None


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


def test_a_caller_the_tool_never_answered_is_refused_before_anything_runs_and_its_own_callers_are_answered():
    """D164: the recording answered `cancel_order` for the assistant and never for the simulated
    user, so a call from the user is refused in the same class, before any code or real world runs;
    a tool of the user's own toolkit answers the user."""
    router = make_router(tool_sigs=[ToolSig(name="get_order_details"),
                                    ToolSig(name="cancel_order", callers=["assistant"],
                                            refused_callers=["user"])])
    before = router.state_hash()
    out = router.route("cancel_order", {"order_id": "123"}, requestor="user")
    assert out.error is not None and out.error.class_ == "tool_not_found"
    assert "for the user" in str(out.error.payload)
    assert router.state_hash() == before
    assert router.world()["orders"]["123"]["status"] == "delivered"

    own = make_router(tool_sigs=[ToolSig(name="get_order_details", callers=["user"])])
    answered = own.route("get_order_details", {"order_id": "123"}, requestor="user")
    assert answered.error is None and answered.route == "code"
    assert answered.result["status"] == "delivered"
    assert own.route("get_order_details", {"order_id": "123"}).error.class_ == "tool_not_found"

    from test_real_tools import FakeWorld

    from kullback.runner.real_tools import RealTool
    world = FakeWorld()
    real = Router(env_tools_module=real_shell_module(),
                  starting_state=StateView(shared={}),
                  tool_sigs=[ToolSig(name="shell", callers=["user"])],
                  real_tools={"shell": RealTool("shell", lambda: world)})
    refused = real.route("shell", {"commands": [{"keystrokes": "ls"}]}, requestor="assistant")
    assert refused.error is not None and refused.error.class_ == "tool_not_found"
    assert world.reset_count == 0


def test_invalid_arguments_keep_the_customers_encoding():
    out = make_router().route("get_order_details", {"nope": 1})
    assert out.error.class_ == "invalid_arguments"
    assert out.error.encoding == "json"
    assert isinstance(out.result, dict)


def test_a_missing_entity_is_classed_not_found_entity_by_code():
    out = make_router().route("get_order_details", {"order_id": "999"})
    assert out.error.class_ == "not_found_entity"
    assert out.error.classified_by == "code"


def test_an_exception_is_answered_in_the_corpus_words_for_its_class():
    """Build 8: 155 unknown-id lookups came back as `KeyError: '#W...'` beside recordings that said 'Order not found'.
    A JSON corpus error comes back as JSON, a class the corpus never showed keeps the exception text, and
    an error the body raised in its own words keeps them."""
    router = make_router(tool_sigs=[ToolSig(name="get_order_details", error_shapes=[
        ErrorShape(class_="not_found_entity", sample_payload="Error: Order not found", encoding="text"),
    ])])
    out = router.route("get_order_details", {"order_id": "999"})
    assert out.result == "Error: Order not found"
    assert out.error.payload == "Error: Order not found"
    assert out.error.class_ == "not_found_entity"

    assert make_router().route("get_order_details", {"order_id": "999"}).result == {"error": "no such order"}

    unshown = make_router().route("get_order_details", {"nope": 1})
    assert unshown.error.class_ == "invalid_arguments"
    assert "TypeError" in unshown.result["error"]

    class RaisingToolkit:
        def __init__(self, db):
            self.db = db

        def cancel_order(self, order_id):
            raise ValueError("Order is not pending")

    raising = make_router(env_tools_module=RaisingToolkit({}), tool_sigs=[ToolSig(name="cancel_order", error_shapes=[
        ErrorShape(class_="business_error", sample_payload="Error: Item not found", encoding="text"),
    ])])
    own_words = raising.route("cancel_order", {"order_id": "123"})
    assert own_words.error.class_ == "business_error"
    assert own_words.result == "ValueError: Order is not pending"


def test_arguments_the_signature_cannot_take_are_invalid_arguments_and_the_body_never_runs():
    """F57: the signature is bound before the body runs, so a bad call is the caller's and the world is untouched."""
    router = make_router()
    out = router.route("cancel_order", {"order_id": "123", "colour": "red"})
    assert out.error.class_ == "invalid_arguments"
    assert "colour" in out.result
    assert router.state.row("orders", "123")["status"] == "delivered"


def test_a_type_error_inside_the_body_is_a_body_fault_that_keeps_its_message_and_line():
    """F57: a TypeError the body raises after binding is the body's bug, not a bad call; the record says where."""

    class AddingToolkit:
        def __init__(self, db):
            self.db = db

        def refund(self, order_id):
            total = "25"
            return total + 1

    out = make_router(env_tools_module=AddingToolkit({}), tool_sigs=[ToolSig(name="refund")]).route(
        "refund", {"order_id": "123"})
    assert out.error.class_ == "body_fault"
    assert out.result == "unavailable"
    assert out.error.payload["fault"] == "TypeError"
    assert out.error.payload["message"].startswith("TypeError: can only concatenate str")
    assert out.error.payload["code"] == "return total + 1"
    assert isinstance(out.error.payload["line"], int)


def test_the_error_encoding_follows_the_traces_and_falls_back_to_text():
    out = make_router(tool_sigs=[ToolSig(name="get_order_details")]).route("get_order_details", {"order_id": "999"})
    assert out.error.encoding == "text"
    router = make_router(tool_sigs=[ToolSig(name="get_order_details", error_shapes=[
        ErrorShape(class_="not_found_entity", sample_payload="no such order", encoding="text"),
    ])])
    shown = router.route("get_order_details", {"order_id": "999"})
    assert shown.error.encoding == "text"
    assert isinstance(shown.result, str)


# --- recording route ---

def test_recording_is_an_exact_lookup_on_tool_args_and_pre_state_hash():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "a@b.com"}, state.hash(), {"name": "Ada"})
    router = make_router(state=state, recordings=[entry])
    out = router.route("lookup_user", {"email": "a@b.com"})
    assert out.route == "recording"
    assert out.result == {"name": "Ada"}
    assert out.assisted is False
    canon = recording("lookup_user", {"email": "a@b.com", "n": 25}, state.hash(), "ok")
    canonical = make_router(state=state, recordings=[canon]).route("lookup_user", {"email": " a@b.com ", "n": 25.0})
    assert canonical.route == "recording"


def test_a_stale_recording_cannot_answer_a_changed_state():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "a@b.com"}, state.hash(), {"name": "Ada"})
    router = make_router(state=state, recordings=[entry])
    router.route("cancel_order", {"order_id": "123"})
    out = router.route("lookup_user", {"email": "a@b.com"})
    assert out.route != "recording"
    assert out.error.class_ == "tool_not_found"
    unkeyed = make_router(state=StateView(shared=shared()), recordings=[{"tool": "lookup_user", "args": {}, "result": "ok"}])
    assert unkeyed.unkeyed_recordings == 1
    assert unkeyed.route("lookup_user", {}).route != "recording"


def test_a_recorded_error_comes_back_as_an_error_and_writes_nothing():
    state = StateView(shared=json.loads(json.dumps(SHARED)))
    entry = recording("lookup_user", {"email": "x"}, state.hash(), None,
                      error={"class": "business_error", "payload": "closed account", "encoding": "text"})
    out = make_router(state=state, recordings=[entry]).route("lookup_user", {"email": "x"})
    assert out.route == "recording"
    assert out.error.class_ == "business_error"
    entry = recording("refund_order", {"order_id": "123"}, state.hash(), None,
                      error={"class": "business_error", "payload": "already shipped", "encoding": "text"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded"}}})
    router = make_router(state=StateView(shared=shared()), recordings=[entry])
    router.route("refund_order", {"order_id": "123"})
    assert router.world()["orders"]["123"]["status"] == "delivered"


def test_a_write_answered_from_the_recording_lands_in_the_world():
    """A recorded write changes the world, so a later read is not stale and the End state shows it,
    on the state view and on a compiled toolkit that reads self.db."""
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

    toolkit = Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs())
    entry = recording("refund_order", {"order_id": "123"}, toolkit.state_hash(), {"status": "refunded"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded", "total": 25}}})
    toolkit.recordings = {recording_key("refund_order", {"order_id": "123"}, toolkit.state_hash()): entry}
    assert toolkit.route("refund_order", {"order_id": "123"}).route == "recording"
    assert toolkit.world()["orders"]["123"]["status"] == "refunded"
    assert toolkit.route("get_order_details", {"order_id": "123"}).result["status"] == "refunded"


def test_a_code_write_after_a_recorded_write_is_the_one_the_end_state_shows():
    """D46: the End state is the world as it stands, so the later write wins, not the recorded one."""
    router = Router(env_tools_module=Toolkit(shared()), starting_state=shared(), tool_sigs=sigs())
    entry = recording("refund_order", {"order_id": "123"}, router.state_hash(), {"status": "refunded"},
                      writes={"orders": {"123": {"id": "123", "status": "refunded", "total": 25}}})
    router.recordings = {recording_key("refund_order", {"order_id": "123"}, router.state_hash()): entry}
    assert router.route("refund_order", {"order_id": "123"}).route == "recording"
    assert router.route("cancel_order", {"order_id": "123"}).result["status"] == "cancelled"
    assert router.world()["orders"]["123"]["status"] == "cancelled"


# --- stand-in route ---

def test_the_stand_in_answers_last_only_and_marks_the_run_assisted():
    """D49: an LLM stand-in is the last route and every Run that uses it is Assisted."""
    model = TestModel(['{"name": "Ada"}'])
    out = make_router(stand_in_model=model).route("lookup_user", {"email": "a@b.com"})
    assert out.route == "llm"
    assert out.assisted is True
    assert out.result == {"name": "Ada"}
    assert len(model.calls) == 1
    unused = TestModel(["never used"])
    coded = make_router(stand_in_model=unused).route("get_order_details", {"order_id": "123"})
    assert coded.route == "code"
    assert unused.calls == []


def test_a_non_json_stand_in_reply_comes_back_as_text():
    out = make_router(stand_in_model=TestModel(["the order is pending"])).route("lookup_user", {})
    assert out.result == "the order is pending"
    assert out.assisted is True


# --- canonical args helper ---

@pytest.mark.parametrize("a,b,same", [
    ({"n": 25}, {"n": 25.0}, True),
    ({"s": "abc"}, {"s": " abc "}, True),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
    ({"n": 25}, {"n": 26}, False),
    # True == 1 in Python, so the difference has to show in the key, which is where it counts
    ({"ok": True}, {"ok": 1}, False),
])
def test_canonical_args_agrees_on_the_same_call_and_keeps_real_differences(a, b, same):
    assert (canonical_args(a) == canonical_args(b)) is same
    assert (recording_key("t", a, "s") == recording_key("t", b, "s")) is same


def test_the_recording_key_folds_the_way_the_customers_rules_fold():
    """D39: keyed under the module defaults, a call the customer's rules call the same is missed."""
    from kullback.runner.canon import CanonRules

    rules = CanonRules(id_patterns={"order_id": r"#?W\d+"}, id_strip_chars="#")
    assert recording_key("t", {"order_id": "#W123"}, "s") != recording_key("t", {"order_id": "W123"}, "s")
    assert recording_key("t", {"order_id": "#W123"}, "s", rules) == \
        recording_key("t", {"order_id": "W123"}, "s", rules)


# --- a synthetic row read through a tool makes the Run assisted (D40, D49, D107) ---

def test_only_a_code_route_that_returns_a_synthetic_row_is_assisted():
    router = make_router(synthetic_rows=["123"])
    out = router.route("get_order_details", {"order_id": "123"})
    assert out.route == "code" and out.error is None
    assert out.assisted is True
    assert make_router(synthetic_rows=["999"]).route("get_order_details", {"order_id": "123"}).assisted is False
    assert make_router().route("get_order_details", {"order_id": "123"}).assisted is False


# --- a Real tool runs for real before any imitation (D262) ---

def real_shell_module():
    """A compiled toolkit that also carries a shell body, which the real route must beat."""
    module = types.ModuleType("real_shell_module")

    def shell(commands):
        return "compiled"

    module.shell = shell
    return module


def test_a_real_tool_answers_before_a_compiled_body_of_the_same_name():
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    world = FakeWorld([FakeReceipt(stdout=b"real-out")])
    router = Router(env_tools_module=real_shell_module(),
                    starting_state=StateView(shared={}),
                    real_tools={"shell": RealTool("shell", lambda: world)})
    out = router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert out.route == "real" and out.error is None
    assert out.result == "real-out"
    assert world.commands == ["ls"]


def test_a_real_call_leaves_the_state_hash_unchanged():
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    world = FakeWorld([FakeReceipt(stdout=b"out")])
    router = Router(env_tools_module=real_shell_module(),
                    starting_state=StateView(shared=json.loads(json.dumps(SHARED))),
                    real_tools={"shell": RealTool("shell", lambda: world)})
    before = router.state_hash()
    router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert router.state_hash() == before


def test_a_routed_world_error_ends_the_run_against_the_environment():
    """The outcome field decides, never the message text: a novel failure class still ends the Run."""
    from test_real_tools import FakeWorld, UnresolvedError

    from kullback.runner.real_tools import RealTool
    from kullback.runner.route import CANNOT_ANSWER_REASON
    world = FakeWorld()
    world.step_error = UnresolvedError("creation is unresolved")
    router = Router(env_tools_module=real_shell_module(),
                    starting_state=StateView(shared={}),
                    tool_sigs=[ToolSig(name="shell", error_shapes=[
                        ErrorShape(class_="unknown", count=1,
                                   sample_payload={"error": "nope"}, encoding="json")])],
                    real_tools={"shell": RealTool("shell", lambda: world)})
    out = router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert out.route == "cannot_answer"
    assert out.error is not None and out.error.class_ == "cannot_answer"
    assert out.error.payload["reason"] == CANNOT_ANSWER_REASON
    assert out.error.payload["tool"] == "shell"
    assert out.error.payload["world_error_class"] == "unknown"
    assert "UnresolvedError" in (out.error.payload["world_error"] or "")
    world.step_error = None
    again = router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert again.error is None

    from kullback.runner.real_tools import RealOutcome

    class StubReal:
        def call(self, args):
            return RealOutcome(None, "mystery_class", "mystery words", [], world_answered=False)

    novel = Router(env_tools_module=real_shell_module(),
                   starting_state=StateView(shared={}),
                   real_tools={"shell": StubReal()})
    unmatched = novel.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert unmatched.route == "cannot_answer"
    assert unmatched.error is not None and unmatched.error.class_ == "cannot_answer"
    assert unmatched.error.payload["reason"] == CANNOT_ANSWER_REASON
    assert unmatched.error.payload["tool"] == "shell"
    assert unmatched.error.payload["world_error_class"] == "mystery_class"
    assert unmatched.error.payload["world_error"] == "mystery words"


def test_refuse_stand_in_leaves_real_tools_answering():
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    from kullback.runner.route import refuse_stand_in
    world = FakeWorld([FakeReceipt(stdout=b"real-out")])
    router = Router(env_tools_module=real_shell_module(),
                    starting_state=StateView(shared={}),
                    real_tools={"shell": RealTool("shell", lambda: world)})
    refuse_stand_in(router)
    out = router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert out.route == "real" and out.result == "real-out"


def test_close_real_closes_every_opened_world_reports_failures_and_opens_nothing_new():
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    first, second = FakeWorld([FakeReceipt(stdout=b"a")]), FakeWorld()
    made_second = []
    router = Router(starting_state=StateView(shared={}), real_tools={
        "one": RealTool("one", lambda: first),
        "two": RealTool("two", lambda: made_second.append(second) or second)})
    router.route("one", {"commands": [{"keystrokes": "ls"}]})
    router.close_real()
    assert first.closed is True
    assert made_second == []

    from test_real_tools import shell_batch
    failing = FakeWorld([FakeReceipt(stdout=b"a")])
    failing.close_error = RuntimeError("cannot remove")
    closing = FakeWorld([FakeReceipt(stdout=b"b")])
    both = Router(starting_state=StateView(shared={}), real_tools={
        "a": RealTool("a", lambda: failing), "b": RealTool("b", lambda: closing)})
    both.route("a", shell_batch("ls"))
    both.route("b", shell_batch("ls"))
    assert both.close_real() == [("a", "RuntimeError: cannot remove")]
    assert closing.closed is True


def test_a_listed_tool_with_a_real_route_does_not_end_the_run():
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    world = FakeWorld([FakeReceipt(stdout=b"real-out")])
    router = Router(starting_state=StateView(shared={}),
                    tool_sigs=[ToolSig(name="shell", callers=["assistant"])],
                    real_tools={"shell": RealTool("shell", lambda: world)})
    out = router.route("shell", {"commands": [{"keystrokes": "ls"}]})
    assert out.route == "real" and out.result == "real-out"


def test_last_real_receipts_carries_the_exit_codes_and_starts_empty():
    from test_real_tools import FakeReceipt, FakeWorld, shell_batch

    from kullback.runner.real_tools import RealTool
    world = FakeWorld([FakeReceipt(stdout=b"one"),
                       FakeReceipt(stdout=b"two", exit_code=1)])
    router = Router(starting_state=StateView(shared={}), real_tools={
        "shell": RealTool("shell", lambda: world),
        "other": RealTool("other", lambda: FakeWorld())})
    assert router.last_real_receipts("shell") == []
    out = router.route("shell", shell_batch("one", "two"))
    assert out.error is None and isinstance(out.result, str)
    assert [receipt.exit_code for receipt in router.last_real_receipts("shell")] == [0, 1]
    assert router.last_real_receipts("other") == []


def test_close_real_exports_under_the_routers_limit_and_the_kept_export_obeys_it_too():
    from test_real_tools import FakeReceipt, FakeWorld, shell_batch

    from kullback.runner.real_tools import RealTool
    from kullback.runner.route import REAL_EXPORT_LIMIT_BYTES
    world = FakeWorld([FakeReceipt(stdout=b"a")], export=b"tar-here")
    router = Router(starting_state=StateView(shared={}),
                    real_tools={"shell": RealTool("shell", lambda: world)})
    assert router.real_export_limit == REAL_EXPORT_LIMIT_BYTES == 64 * 1024 * 1024
    router.real_export_limit = 1234
    router.route("shell", shell_batch("ls"))
    router.close_real()
    assert world.export_limit == 1234

    from kullback.runner.real_tools import ExportLimitError
    kept = FakeWorld([FakeReceipt(stdout=b"a")], export=b"12345678")
    router = Router(starting_state=StateView(shared={}),
                    real_tools={"shell": RealTool("shell", lambda: kept)})
    router.route("shell", shell_batch("ls"))
    with pytest.raises(ExportLimitError) as live:
        router.real_end_state("shell", 4)
    router.close_real()
    assert router.real_end_state("shell", 64) == b"12345678"
    with pytest.raises(ExportLimitError) as refused:
        router.real_end_state("shell", 4)
    assert str(refused.value) == str(live.value)


def test_a_real_world_timeout_ends_routing_as_cannot_answer():
    """A world that could not run the call ends the Run, naming the tool and the reason."""
    from test_real_tools import FakeReceipt, FakeWorld

    from kullback.runner.real_tools import RealTool
    from kullback.runner.route import CANNOT_ANSWER_REASON
    world = FakeWorld([FakeReceipt(stdout=b"part", timed_out=True)])
    router = Router(env_tools_module=real_shell_module(),
                    starting_state=StateView(shared={}),
                    real_tools={"shell": RealTool("shell", lambda: world)})
    out = router.route("shell", {"commands": [{"keystrokes": "sleep 99"}]})
    assert out.route == "cannot_answer"
    assert out.result is None
    assert out.error is not None and out.error.class_ == "cannot_answer"
    assert out.error.payload["reason"] == CANNOT_ANSWER_REASON
    assert out.error.payload["tool"] == "shell"
    assert out.error.payload["world_error_class"] == "transient"