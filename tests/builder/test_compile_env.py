"""Tests for compile_env: inverse replay, per-Task overlays, the tau2 shape, the five gates and the repair loop."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import tokenize

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sandbox_mod
from kullback.builder import tools as builder_tools
from kullback.gates.confinement import PROVIDED_HELPERS, gate_confined
from kullback.runner.records import (
    Atom,
    Column,
    EntitySchema,
    Environment,
    FieldStat,
    Task,
    ToolCall,
    ToolCallError,
    ToolSig,
    Trace,
    Verifier,
    as_dict,
)

# --- hand-written tool bodies, standing in for what a model would write ---

CORRECT_BODY = """
if order_id not in self.db.orders:
    raise ValueError("Order not found")
return self.db.orders[order_id]
"""

WRONG_BODY = """
return {"order_id": order_id, "status": "delivered"}
"""

CONSTANT_BODY = """
return {"order_id": "#W0000000", "status": "delivered"}
"""

BROKEN_BODY = """
return self.db.orders[order_id
"""

CRASHING_BODY = """
return self.db.warehouses[order_id]
"""

SLOW_BODY = """
while True:
    pass
"""

CANCEL_BODY = """
if order_id not in self.db.orders:
    raise ValueError("Order not found")
order = self.db.orders[order_id]
if order.status != "pending":
    raise ValueError("Non-pending order cannot be cancelled")
order.status = "cancelled"
order.cancel_reason = reason
return order
"""

FORGING_BODY = """
import json
import os
import sys

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"results": [{"ok": True, "value": {"order_id": order_id, "status": "delivered"}}]}, handle)
os._exit(0)
"""


def _schema_for(db, exempt=(), semantic=("address",)):
    """A schema over the rows of one small db, with the column classes the test cares about."""
    columns = []
    for table, rows in db.items():
        for name in sorted({key for row in rows.values() for key in row}):
            kind = "exempt" if name in exempt else ("semantic" if name in semantic else "hard")
            columns.append(Column(table=table, name=name, **{"class": kind}, classified_by="rule"))
    return EntitySchema(
        tables=sorted(db),
        columns=columns,
        id_patterns={"orders": r"^#W\d+$", "users": r"^[a-z]+_[a-z]+_\d+$", "products": r"^\d+$"},
    )


# --- fixtures built from tau2's own retail db ---


@pytest.fixture(scope="session")
def retail_db(tau2_retail_dir):
    return json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample(retail_db):
    """Three orders with different statuses, plus one user and one product, kept small on purpose."""
    orders = retail_db["orders"]
    picked = {}
    for status in ("pending", "delivered", "processed"):
        oid = next(k for k in sorted(orders) if orders[k]["status"] == status)
        picked[status] = orders[oid]
    user_id = picked["pending"]["user_id"]
    product_id = picked["pending"]["items"][0]["product_id"]
    return {
        "orders": {o["order_id"]: o for o in picked.values()},
        "users": {user_id: retail_db["users"][user_id]},
        "products": {product_id: retail_db["products"][product_id]},
        "by_status": picked,
    }


@pytest.fixture(scope="session")
def db0(sample):
    return {"orders": sample["orders"], "users": sample["users"], "products": sample["products"]}


@pytest.fixture(scope="session")
def schema(db0):
    columns = []
    for table, rows in db0.items():
        names = sorted({key for row in rows.values() for key in row})
        for name in names:
            kind = "semantic" if name == "address" else "hard"
            columns.append(Column(table=table, name=name, **{"class": kind}, classified_by="rule"))
    return EntitySchema(
        tables=sorted(db0),
        columns=columns,
        id_patterns={
            "orders": r"^#W\d+$",
            "users": r"^[a-z]+_[a-z]+_\d+$",
            "products": r"^\d+$",
        },
    )


@pytest.fixture(scope="session")
def sigs():
    return [
        ToolSig(
            name="get_order_details",
            description="Get the status and details of an order.",
            args_fields=[FieldStat(name="order_id", types=["str"], optional=False)],
            kind="read",
            unclassified=False,
        ),
        ToolSig(name="cancel_pending_order", kind="write", unclassified=False),
    ]


def _call(name, args, result=None, error=None, idx=0):
    return ToolCall(id=f"c{idx}", name=name, args=args, result=result, error=error, raw_ptr=PTR)


def _trace(trace_id, calls):
    return Trace(
        trace_id=trace_id,
        raw_hash="rawhash",
        ingest_version="1",
        source="tau2",
        tool_calls=calls,
        raw_ptr=PTR,
    )


@pytest.fixture
def order_calls(sample):
    """Recorded get_order_details calls: three successes and one not-found error."""
    calls = []
    for idx, order in enumerate(sample["orders"].values()):
        calls.append(_call("get_order_details", {"order_id": order["order_id"]}, result=order, idx=idx))
    calls.append(
        _call(
            "get_order_details",
            {"order_id": "#W0000000"},
            error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"),
            idx=9,
        )
    )
    return calls


# --- inverse replay over the whole corpus (D33, D74) ---


def test_inverse_replay_undoes_an_observed_write(sample, schema, sigs, workdir):
    pending = sample["by_status"]["pending"]
    oid = pending["order_id"]
    post = dict(pending, status="cancelled", cancel_reason="no longer needed")
    trace = _trace(
        "A",
        [
            _call("get_order_details", {"order_id": oid}, result=pending, idx=0),
            _call("cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}, result=post, idx=1),
        ],
    )
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["orders"][oid]["status"] == "pending"
    assert json.loads((workdir / ce.DB_FILE).read_text(encoding="utf-8")) == state.db


def test_post_state_only_is_kept_and_the_assumption_is_recorded(sample, schema, sigs, workdir):
    pending = sample["by_status"]["pending"]
    oid = pending["order_id"]
    post = dict(pending, status="cancelled")
    trace = _trace("B", [_call("cancel_pending_order", {"order_id": oid}, result=post, idx=0)])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["orders"][oid]["status"] == "cancelled"
    assert any(oid in note for note in state.assumptions)


def test_latest_observation_wins_in_the_shared_rows(sample, schema, sigs, workdir):
    order = sample["by_status"]["delivered"]
    oid = order["order_id"]
    first = _trace("A", [_call("get_order_details", {"order_id": oid}, result=dict(order, status="processed"))])
    second = _trace("B", [_call("get_order_details", {"order_id": oid}, result=dict(order, status="delivered"))])
    state = ce.build_starting_state([first, second], schema, workdir, tool_sigs=sigs)
    assert state.db["orders"][oid]["status"] == "delivered"


def test_a_json_string_result_is_read_as_a_row(sample, schema, sigs, workdir):
    order = sample["by_status"]["processed"]
    trace = _trace("A", [_call("get_order_details", {"order_id": order["order_id"]}, result=json.dumps(order))])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["orders"][order["order_id"]]["status"] == "processed"


def test_the_shared_db_is_the_same_bytes_on_two_runs(sample, schema, sigs, tmp_path):
    order = sample["by_status"]["pending"]
    traces = [_trace("A", [_call("get_order_details", {"order_id": order["order_id"]}, result=order)])]
    outputs = []
    for name in ("one", "two"):
        work = tmp_path / name
        work.mkdir()
        ce.build_starting_state(traces, schema, work, tool_sigs=sigs)
        outputs.append((work / ce.DB_FILE).read_bytes())
    assert outputs[0] == outputs[1]


# --- per-Task overlays (D74) ---


def test_each_task_overlay_pins_the_version_its_runs_saw(sample, schema, sigs, workdir):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    cancelled = dict(order, status="cancelled")
    march = _trace("A", [_call("get_order_details", {"order_id": oid}, result=order)])
    june = _trace("C", [_call("get_order_details", {"order_id": oid}, result=cancelled)])
    tasks = [Task(id="t_cancel", run_ids=["A"]), Task(id="t_after", run_ids=["C"])]
    state = ce.build_starting_state([march, june], schema, workdir, tasks=tasks, tool_sigs=sigs)

    assert {o.task_id for o in state.overlays} == {"t_cancel", "t_after"}
    cancel_overlay, cancel_values = ce.load_overlay(workdir, "t_cancel")
    after_overlay, after_values = ce.load_overlay(workdir, "t_after")
    # The order's own row, and the rows the order mentions inside itself (D188 walks a result to
    # any depth), the order first because the rows are sorted by table.
    assert cancel_overlay.rows[0].table == "orders" and cancel_overlay.rows[0].id == oid
    assert {r.table for r in cancel_overlay.rows} == {"orders", "products"}
    assert cancel_values[cancel_overlay.rows[0].version_hash]["status"] == "pending"
    assert after_values[after_overlay.rows[0].version_hash]["status"] == "cancelled"
    assert cancel_overlay.rows[0].version_hash != after_overlay.rows[0].version_hash
    assert cancel_overlay.rows[0].trace_id == "A"


def test_an_overlay_row_carries_the_pre_write_version(sample, schema, sigs, workdir):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    post = dict(order, status="cancelled")
    trace = _trace(
        "A",
        [
            _call("get_order_details", {"order_id": oid}, result=order, idx=0),
            _call("cancel_pending_order", {"order_id": oid}, result=post, idx=1),
        ],
    )
    tasks = [Task(id="t1", run_ids=["A"])]
    ce.build_starting_state([trace], schema, workdir, tasks=tasks, tool_sigs=sigs)
    overlay, values = ce.load_overlay(workdir, "t1")
    assert values[overlay.rows[0].version_hash]["status"] == "pending"


# --- the model writes the body, code writes the signature, docstring and schema ---


def test_write_tool_body_asks_the_model_and_returns_the_body(make_test_model, sigs, order_calls):
    model = make_test_model(["```python\n" + CORRECT_BODY.strip() + "\n```"])
    body = ce.write_tool_body(model, sigs[0], order_calls)
    assert body.strip().startswith("if order_id not in self.db.orders")
    assert "```" not in body
    prompt = model.calls[0]["messages"][-1]["content"]
    assert "get_order_details" in prompt
    assert order_calls[0].args["order_id"] in prompt


def test_render_tools_writes_the_signature_and_the_docstring(schema, sigs):
    source = ce.render_tools(schema, sigs[:1], {"get_order_details": CORRECT_BODY})
    assert "def get_order_details(self, order_id: str)" in source
    assert "Get the status and details of an order." in source
    assert "        if order_id not in self.db.orders:" in source
    compile(source, "<rendered>", "exec")


def test_render_data_model_has_one_class_per_table_and_a_db(schema):
    source = ce.render_data_model(schema)
    for name in ("class Order(", "class User(", "class Product(", f"class {ce.DB_CLASS}("):
        assert name in source
    compile(source, "<rendered>", "exec")


# --- the five gates ---


def _sandbox(schema, sigs, db0, workdir, body, timeout=30):
    source = ce.module_source(schema, sigs[:1], {"get_order_details": body})
    return source, ce.Sandbox(source, db0, workdir, timeout=timeout)


def test_gate_parses_fails_on_a_body_that_is_not_python(schema, sigs, db0, workdir):
    source = ce.module_source(schema, sigs[:1], {"get_order_details": BROKEN_BODY})
    result = ce.gate_parses(source)
    assert result.passed is False
    assert result.stage == "parses"
    assert result.failures


def test_a_correct_body_passes_all_five_gates(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CORRECT_BODY)
    shown, held_out = ce.split_calls(order_calls)
    gates = ce.run_gates(source, box, shown, held_out, schema)
    # "confined" is the gate that stands in for the deferred sandbox: the same module is exec'd
    # in the Runner's process by load_toolkit, so a body that reaches past the customer's world
    # is refused before it ever executes anywhere.
    assert [g.stage for g in gates][:6] == [
        "parses", "confined", "compile_tools.memorised_values", "executes_on_s0", "deterministic",
        "non_trivial"]
    assert all(g.passed for g in gates), [g.failures for g in gates if not g.passed]


def test_a_crashing_body_fails_the_executes_gate(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CRASHING_BODY)
    result = ce.gate_executes_on_s0(box, order_calls[:1])
    assert result.passed is False
    assert result.stage == "executes_on_s0"


def test_a_constant_body_fails_the_non_trivial_gate(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CONSTANT_BODY)
    result = ce.gate_non_trivial(box, order_calls[:3])
    assert result.passed is False
    assert result.stage == "non_trivial"


def test_a_correct_body_is_deterministic_across_two_runs(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CORRECT_BODY)
    result = ce.gate_deterministic(box, order_calls[:1])
    assert result.passed is True
    assert result.metrics["calls"] == 1


def test_replay_fidelity_reports_success_and_error_separately(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CORRECT_BODY)
    result = ce.gate_replay_fidelity(box, order_calls, schema)
    assert result.passed is True
    assert result.metrics["success_calls"] == 3
    assert result.metrics["error_calls"] == 1
    assert result.metrics["success_fidelity"] == 1.0
    assert result.metrics["error_fidelity"] == 1.0


def test_replay_fidelity_fails_a_wrong_body_on_hard_columns_and_on_errors(
    schema, sigs, db0, workdir, order_calls
):
    source, box = _sandbox(schema, sigs, db0, workdir, WRONG_BODY)
    result = ce.gate_replay_fidelity(box, order_calls, schema)
    assert result.passed is False
    assert result.metrics["success_fidelity"] < 1.0
    assert result.metrics["error_fidelity"] == 0.0
    assert result.failures


def test_the_sandbox_kills_a_body_that_does_not_stop(schema, sigs, db0, workdir, order_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, SLOW_BODY, timeout=2)
    result = ce.gate_executes_on_s0(box, order_calls[:1])
    assert result.passed is False
    assert any("timeout" in failure for failure in result.failures)


def test_split_calls_puts_every_call_in_exactly_one_of_the_shown_and_held_out_splits_and_leaves_neither_empty(order_calls):
    shown, held_out = ce.split_calls(order_calls)
    assert shown and held_out
    assert len(shown) + len(held_out) == len(order_calls)
    assert not set(id(c) for c in shown) & set(id(c) for c in held_out)


# --- the repair loop (D75) ---


def test_the_repair_loop_accepts_a_correct_rewrite(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model([WRONG_BODY, CORRECT_BODY])
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)
    assert build.assisted is False
    assert len(build.nodes) == 2
    assert build.nodes[0]["evidence"] == "initial"
    assert build.nodes[1]["evidence"] == "failing_call"
    assert build.nodes[-1]["passed"] is True
    assert build.body.strip() == CORRECT_BODY.strip()


def test_the_repair_loop_grows_the_evidence_and_marks_the_tool_assisted(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model([WRONG_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)
    assert build.assisted is True
    assert [node["evidence"] for node in build.nodes] == list(ce.EVIDENCE_LABELS)
    assert len(model.calls) == 4
    assert all(node["passed"] is False for node in build.nodes)
    assert len(model.calls[3]["messages"][-1]["content"]) > len(model.calls[1]["messages"][-1]["content"])
    written = json.loads((workdir / ce.NODE_DIR / "get_order_details.json").read_text(encoding="utf-8"))
    assert len(written["nodes"]) == 4


def test_a_retry_keeps_the_first_two_messages_byte_identical(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """docs/prompt-caching.md item 2: the system message and the first user turn are the cached
    prefix; a retry must append, never rewrite them."""
    model = make_test_model([WRONG_BODY] * 4)
    ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)
    assert len(model.calls) == 4
    first_two = model.calls[0]["messages"][:2]
    for call in model.calls[1:]:
        assert call["messages"][:2] == first_two
        assert len(call["messages"]) > 2


def test_a_later_attempt_that_crashes_does_not_replace_an_earlier_one_that_replays(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """A live build's fourth attempt at one write tool crashed on all 67 of its calls where the
    third had replayed 44 of 44 and failed only the refusal probe. The last attempt was the one
    kept, which cost 94 replay misses and 38 Tasks; the best attempt is kept now."""
    model = make_test_model([WRONG_BODY] + [CRASHING_BODY] * 3)
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)
    assert build.assisted is True and all(not node["passed"] for node in build.nodes)
    assert build.kept_attempt == 0
    assert build.body.strip() == WRONG_BODY.strip()
    assert [gate.stage for gate in build.gates if gate.passed][:4] == [
        "parses", "confined", "compile_tools.memorised_values", "executes_on_s0"]
    written = json.loads((workdir / ce.NODE_DIR / "get_order_details.json").read_text(encoding="utf-8"))
    assert written["kept_attempt"] == 0
    assert [node["attempt"] for node in written["nodes"] if node.get("kept")] == [0]


def test_the_kept_attempt_is_the_one_that_got_furthest_through_the_gates():
    """The gates run in one order and stop at the first failure, so how many passed is how far the
    body got; two bodies that fell at the same gate are separated by what their replay matched."""
    from kullback.runner.records import GateResult

    def gates(passed: int, matches: int) -> list:
        names = ["parses", "confined", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity"]
        return [GateResult(stage=name, **{"pass": True},
                           metrics={"success_matches": matches} if name == "replay_fidelity" else {})
                for name in names[:passed]]

    assert ce.attempt_score(gates(6, 44)) > ce.attempt_score(gates(2, 0))
    assert ce.attempt_score(gates(6, 44)) > ce.attempt_score(gates(6, 12))
    assert ce.attempt_score([]) == (0, 0)


# --- an attempt that raises what the attempt before raised is told so ---

LOANS_BY_ATTRIBUTE = """
member = self.db.members[member_id]
return {"member_id": member_id, "copy_id": member.loans[0].copy_id}
"""

LOANS_BY_ANOTHER_ATTRIBUTE = """
member = self.db.members[member_id]
return {"member_id": member_id, "due": member.loans[0].due}
"""


def _library_world():
    """A lending library, its schema, the tool that reads one member and four recorded calls."""
    db = {"members": {f"m-{i}": {"member_id": f"m-{i}",
                                 "loans": [{"copy_id": f"c-{i}", "due": "2026-01-02"}]}
                      for i in range(1, 5)},
          "titles": {}}
    sig = ToolSig(name="get_member", description="Read one member and their loans.",
                  args_fields=[FieldStat(name="member_id", types=["str"], optional=False)],
                  kind="read", unclassified=False)
    calls = [_call("get_member", {"member_id": member_id},
                   result={"member_id": member_id, "copy_id": row["loans"][0]["copy_id"]}, idx=i)
             for i, (member_id, row) in enumerate(sorted(db["members"].items()))]
    return db, _library_schema(), sig, calls


MEMBER_BY_A_KEY_NOBODY_HOLDS = """
return self.db.members[member_id + "-x"]
"""

WRONG_MEMBER_BODY = """
member = self.db.members[member_id]
return {"member_id": member.member_id}
"""


def test_a_body_that_raises_on_a_missing_row_is_told_the_table_and_that_the_id_is_absent(
    make_test_model, workdir
):
    """A bare KeyError says nothing about what the body was reading, so the next attempt guesses.
    The crashing line names the table and the world in front of it answers for the id."""
    db, schema, sig, calls = _library_world()
    model = make_test_model([MEMBER_BY_A_KEY_NOBODY_HOLDS] * 4)
    ce.compile_tool(model, sig, calls, schema, db, workdir)
    retry = model.calls[1]["messages"][-1]["content"]
    assert "raised KeyError at `return self.db.members[member_id + \"-x\"]`" in retry
    assert "`members` holds 4 rows and not `m-1-x`" in retry
    assert "its ids look like a-#" in retry, retry
    assert "m-2" not in retry.split("holds 4 rows")[1], "the other ids are shapes, not values"


def test_a_body_that_does_not_raise_gets_no_table_note(make_test_model, workdir):
    db, schema, sig, calls = _library_world()
    model = make_test_model([WRONG_MEMBER_BODY] * 4)
    ce.compile_tool(model, sig, calls, schema, db, workdir)
    retry = model.calls[1]["messages"][-1]["content"]
    assert "holds" not in retry and "look like" not in retry


CONSTANT_MEMBER_BODY = """
return {"member_id": "none", "copy_id": "none"}
"""


def test_a_body_that_answers_one_constant_to_calls_with_different_arguments_is_marked_hardcoded(
    make_test_model, workdir
):
    """One live build kept a body that picked the first row and raised one fixed refusal on every
    call, round after round: assisted was all anyone was told, and no hint reaches a body with no
    argument in it."""
    db, schema, sig, calls = _library_world()
    model = make_test_model([CONSTANT_MEMBER_BODY] * 4)
    build = ce.compile_tool(model, sig, calls, schema, db, workdir)
    assert build.assisted is True and build.hardcoded is True
    written = json.loads((workdir / ce.NODE_DIR / "get_member.json").read_text(encoding="utf-8"))
    assert written["hardcoded"] is True
    assert ce.mark_hardcoded(build.body).splitlines()[0] == ce.HARDCODED_MARK
    assert ce.mark_hardcoded(ce.mark_hardcoded(build.body)).count(ce.HARDCODED_MARK) == 1


def test_a_body_that_answers_per_argument_is_assisted_but_not_hardcoded(make_test_model, workdir):
    db, schema, sig, calls = _library_world()
    model = make_test_model([WRONG_MEMBER_BODY] * 4)
    build = ce.compile_tool(model, sig, calls, schema, db, workdir)
    assert build.assisted is True and build.hardcoded is False


def test_a_tool_whose_own_recordings_answer_alike_is_not_called_hardcoded():
    """Where the recorded calls share one answer, one answer is what the tool does."""
    calls = [_call("close_ticket", {"ticket_id": f"t-{i}"}, result={"status": "closed"}, idx=i)
             for i in range(3)]
    rows = [{"call_id": call.id, "answer": "one", "world": "w"} for call in calls]
    assert ce.hardcoded_body(calls, rows) is False
    differing = [_call("close_ticket", {"ticket_id": f"t-{i}"}, result={"status": f"s-{i}"}, idx=i)
                 for i in range(3)]
    rows = [{"call_id": c.id, "answer": "one", "world": "w"} for c in differing]
    assert ce.hardcoded_body(differing, rows) is True
    assert ce.hardcoded_body(differing[:1], rows[:1]) is False


def test_a_tool_with_no_arguments_is_hardcoded_when_it_answers_two_worlds_the_same_way():
    """The live build's own case: every recorded call carried no arguments at all, the recordings
    answered two ways out of the Task's own world, and the body answered one way on both."""
    calls = [_call("current_balance", {}, result={"balance": f"{i}"}, idx=i) for i in range(4)]
    worlds = [{"call_id": call.id, "answer": "one", "world": f"w-{i}"} for i, call in enumerate(calls)]
    assert ce.hardcoded_body(calls, worlds) is True
    one_world = [dict(row, world="w") for row in worlds]
    assert ce.hardcoded_body(calls, one_world) is False, "one input, so one answer says nothing"


def test_the_shape_of_an_id_generalizes_its_characters_and_keeps_the_rest():
    assert ce.value_shape("m-14") == "a-##"
    assert ce.value_shape("AB_9") == "AA_#"
    assert ce.table_read("row = self.db.members[member_id]") == "members"
    assert ce.table_read("return {}") == ""


def test_an_attempt_that_raises_what_the_attempt_before_raised_is_told_so_with_the_exception(
    make_test_model, workdir
):
    """One write tool of a live build crashed on all four attempts with the same AttributeError at
    the same call. Each retry showed the crash and none said the error had not moved."""
    db, schema, sig, calls = _library_world()
    model = make_test_model([LOANS_BY_ATTRIBUTE] * 4)
    build = ce.compile_tool(model, sig, calls, schema, db, workdir)
    assert build.assisted is True and len(model.calls) == 4
    third_turn = model.calls[2]["messages"][-1]["content"]
    assert "The same error as the attempt before, at the same call" in third_turn
    assert "AttributeError: 'dict' object has no attribute 'copy_id'" in third_turn
    assert model.calls[2]["messages"][:2] == model.calls[0]["messages"][:2]


def test_an_attempt_that_raises_a_different_exception_is_not_told_it_is_the_same_error(
    make_test_model, workdir
):
    db, schema, sig, calls = _library_world()
    model = make_test_model([LOANS_BY_ATTRIBUTE, LOANS_BY_ANOTHER_ATTRIBUTE,
                             LOANS_BY_ATTRIBUTE, LOANS_BY_ANOTHER_ATTRIBUTE])
    ce.compile_tool(model, sig, calls, schema, db, workdir)
    assert len(model.calls) == 4
    for call in model.calls[1:]:
        assert "The same error as the attempt before" not in call["messages"][-1]["content"]


def test_the_exception_a_failing_gate_quotes_is_read_off_its_failure_line():
    from kullback.runner.records import GateResult

    crash = GateResult(stage="executes_on_s0", **{"pass": False}, failures=[
        "get_member(member_id='m-1') raised AttributeError: 'dict' object has no attribute 'copy_id'"])
    green = GateResult(stage="parses", **{"pass": True})
    assert ce.exception_line([green, crash]) == "AttributeError: 'dict' object has no attribute 'copy_id'"
    assert ce.exception_line([green]) == ""
    assert ce.exception_in("get_member(member_id='m-1') answered differently on a second run") == ""


def test_a_recompile_hint_reaches_the_compiler_prompt_for_that_tool(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """A hint nothing reads changes nothing. The lesson `repair.record_tool_lesson` wrote is what
    the next attempt at this one tool is shown; without it the live build asked for the same
    recompile seven times and was handed the same body every time."""
    from kullback.builder import repair

    repair.record_tool_lesson(workdir, sigs[0].name, ["executes: the body raised KeyError on every call"])
    model = make_test_model([CORRECT_BODY])
    ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir,
                    lesson=repair.lesson_for_tool(workdir, sigs[0].name))
    system, user = model.calls[0]["messages"][0]["content"], model.calls[0]["messages"][-1]["content"]
    assert "raised KeyError on every call" in user
    assert f"Past gate failures for {sigs[0].name}" in user
    assert "KeyError" not in system, "one tool's lesson must not move the prefix every tool shares"


def test_with_no_lesson_the_compile_prompt_is_unchanged(schema, sigs, order_calls):
    """The offline builds are deterministic byte for byte, so a tool with no lesson has to send
    exactly the messages it sent before there was such a thing as a lesson."""
    assert (ce.body_messages(sigs[0], order_calls[:3], schema=schema, lesson="")
            == ce.body_messages(sigs[0], order_calls[:3], schema=schema))


def test_an_attempt_over_the_evidence_cap_is_refused_not_truncated(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model([WRONG_BODY] * 4)
    build = ce.compile_tool(
        model, sigs[0], order_calls, schema, db0, workdir, max_evidence_chars=10
    )
    assert build.assisted is True
    assert build.nodes[-1]["refused"] is True
    assert model.calls == []


# --- the tau2 file shape (D56) ---


def _bundle(schema, sigs, db0, overlays=(), tasks=(), verifiers=()):
    return ce.EnvBundle(
        environment=Environment(env_id="env1", assisted_tools=["cancel_pending_order"]),
        schema=schema,
        tools=sigs[:1],
        bodies={"get_order_details": CORRECT_BODY},
        db=db0,
        overlays=list(overlays),
        policy_text="# Retail policy\n\nAuthenticate the user first.\n",
        tasks=list(tasks),
        verifiers=list(verifiers),
        assumptions=["order #W0000000 was only seen after a write"],
    )


def test_emit_tau2_shape_writes_the_five_files_and_a_sidecar(schema, sigs, db0, workdir):
    task = Task(id="t1", intent="cancel a pending order", run_ids=["A"])
    verifier = Verifier(
        task_id="t1",
        atoms=[
            Atom(
                id="t1_0",
                kind="required",
                description="cancel the order",
                predicate_src=json.dumps({"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}),
            ),
            Atom(id="t1_1", kind="allowed", description="offer a gift card"),
        ],
    )
    paths = ce.emit_tau2_shape(_bundle(schema, sigs, db0, tasks=[task], verifiers=[verifier]), workdir)
    assert set(paths) == {"data_model.py", "tools.py", "db.json", "policy.md", "tasks.json", "sidecar.json"}
    for path in paths.values():
        assert path.exists()
    compile(paths["data_model.py"].read_text(encoding="utf-8"), "data_model.py", "exec")
    compile(paths["tools.py"].read_text(encoding="utf-8"), "tools.py", "exec")

    tasks_json = json.loads(paths["tasks.json"].read_text(encoding="utf-8"))
    assert tasks_json[0]["id"] == "t1"
    assert tasks_json[0]["evaluation_criteria"]["actions"][0]["name"] == "cancel_pending_order"

    sidecar = json.loads(paths["sidecar.json"].read_text(encoding="utf-8"))
    assert sidecar["assisted_tools"] == ["cancel_pending_order"]
    assert sidecar["assumptions"]
    assert sidecar["atoms"]["t1"][1]["kind"] == "allowed"
    assert set(sidecar["files"]) == {"data_model.py", "tools.py", "db.json", "policy.md", "tasks.json"}


def test_emit_merges_overlays_into_one_db(sample, schema, sigs, db0, workdir):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    cancelled = dict(order, status="cancelled")
    trace = _trace("A", [_call("get_order_details", {"order_id": oid}, result=cancelled)])
    state = ce.build_starting_state(
        [trace], schema, workdir, tasks=[Task(id="t1", run_ids=["A"])], tool_sigs=sigs
    )
    bundle = _bundle(schema, sigs, db0, overlays=state.overlays)
    bundle.overlay_values = ce.overlay_values(workdir)
    paths = ce.emit_tau2_shape(bundle, workdir / "out")
    merged = json.loads(paths["db.json"].read_text(encoding="utf-8"))
    assert merged["orders"][oid]["status"] == "cancelled"


def test_emit_records_two_tasks_that_pin_the_same_row_differently(sample, schema, sigs, db0, workdir):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    cancelled = dict(order, status="cancelled")
    traces = [
        _trace("A", [_call("get_order_details", {"order_id": oid}, result=order)]),
        _trace("C", [_call("get_order_details", {"order_id": oid}, result=cancelled)]),
    ]
    tasks = [Task(id="t1", run_ids=["A"]), Task(id="t2", run_ids=["C"])]
    state = ce.build_starting_state(traces, schema, workdir, tasks=tasks, tool_sigs=sigs)
    bundle = _bundle(schema, sigs, db0, overlays=state.overlays)
    bundle.overlay_values = ce.overlay_values(workdir)
    ce.emit_tau2_shape(bundle, workdir / "out")
    # D74: the export records the conflict and keeps one version; the Runner keeps both overlays.
    assert len(bundle.conflicts) == 1 and oid in bundle.conflicts[0]
    exported = json.loads((workdir / "out" / "db.json").read_text())
    assert exported["orders"][oid]["status"] == order["status"]
    with pytest.raises(ce.OverlayConflict):
        ce.merge_overlays(db0, state.overlays, bundle.overlay_values)


def test_the_export_prefers_the_version_seen_before_a_write(tmp_path):
    from kullback.runner.records import OverlayRow, TaskOverlay
    late = TaskOverlay(task_id="t1", rows=[OverlayRow(table="orders", id="o1", version_hash="v_after",
                                                       after_write=True)])
    early = TaskOverlay(task_id="t2", rows=[OverlayRow(table="orders", id="o1", version_hash="v_before")])
    values = {"v_after": {"status": "cancelled"}, "v_before": {"status": "pending"}}
    conflicts: list[str] = []
    merged = ce.merge_overlays({"orders": {}}, [late, early], values, conflicts)
    assert merged["orders"]["o1"]["status"] == "pending"
    assert conflicts == ["tasks t1 and t2 pin orders row o1 in different versions; the tau2 export keeps t2's"]


# --- the Environment record, its sub-versions and its open flags (D67, D70, D97) ---

def test_build_environment_carries_the_sub_versions_and_the_flags():
    from kullback.builder.compile_env import build_environment
    from kullback.runner.records import Column, EntitySchema, ErrorShape, ToolSig

    schema = EntitySchema(tables=["orders"], columns=[Column(table="orders", name="status", class_="hard")])
    sigs = [
        ToolSig(name="cancel_order", kind="write", unclassified=False,
                error_shapes=[ErrorShape(class_="unknown", count=8), ErrorShape(class_="business_error", count=2)]),
        ToolSig(name="mystery_tool", unclassified=True),
    ]
    env = build_environment(schema, sigs, {"cancel_order": "pass"}, "the policy text")
    assert env.env_id and env.schema_version and env.tools_version and env.policy_version
    assert len(env.flags) == 2
    assert any(f.startswith("cancel_order: 8 of 10 observed errors are unknown") for f in env.flags)
    assert any("mystery_tool: read or write not confirmed" in f for f in env.flags)


def test_a_changed_policy_only_moves_the_policy_version():
    from kullback.builder.compile_env import build_environment
    from kullback.runner.records import EntitySchema

    schema = EntitySchema(tables=[], columns=[])
    first = build_environment(schema, [], {}, "one")
    second = build_environment(schema, [], {}, "two")
    assert first.policy_version != second.policy_version
    assert first.schema_version == second.schema_version
    assert first.tools_version == second.tools_version
    assert first.env_id != second.env_id


def test_the_evidence_cap_is_on_by_default_and_is_forty_percent_of_the_window(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """D65 in numbers, not in the formula that defines it, and on by default in compile_tool."""
    assert ce.CONTEXT_WINDOW == 200_000
    assert ce.MAX_EVIDENCE_CHARS == 320_000  # 40 percent of 200,000 tokens, 4 characters a token
    assert inspect.signature(ce.compile_tool).parameters["max_evidence_chars"].default == 320_000

    # and it is a cap on the prompt that is sent, not on the evidence block alone
    wordy = sigs[0].model_copy(update={"description": "x" * 5000})
    cap = len(ce._example_block(order_calls[:3])) + 100
    model = make_test_model([CORRECT_BODY] * 4)
    build = ce.compile_tool(model, wordy, order_calls, schema, db0, workdir, max_evidence_chars=cap)
    assert model.calls == [], "a prompt of over 5,000 characters went out under a cap of a few hundred"
    assert build.nodes[-1]["refused"] is True
    assert build.assisted is True


# --- every recorded call replays on its own Starting state (D74, design section 6 gate 2) ---


@pytest.fixture
def cancel_sig():
    return ToolSig(
        name="cancel_pending_order",
        description="Cancel a pending order.",
        args_fields=[
            FieldStat(name="order_id", types=["str"], optional=False),
            FieldStat(name="reason", types=["str"], optional=False),
        ],
        kind="write",
        unclassified=False,
    )


@pytest.fixture
def cancel_world(sample):
    """One pending order with room for a cancel reason, and the schema over it."""
    order = dict(sample["by_status"]["pending"], cancel_reason=None)
    db = {"orders": {order["order_id"]: order}, "users": {}, "products": {}}
    return db, _schema_for(db), order


def test_a_write_tool_passes_all_five_gates_on_calls_two_runs_recorded(cancel_world, cancel_sig, workdir):
    """Two Runs of one Task each cancel the same pending order; both start from S0, so both replay."""
    db, schema, order = cancel_world
    oid = order["order_id"]
    first = dict(order, status="cancelled", cancel_reason="no longer needed")
    second = dict(order, status="cancelled", cancel_reason="ordered by mistake")
    calls = [
        _call(cancel_sig.name, {"order_id": oid, "reason": "no longer needed"}, result=first, idx=0),
        _call(cancel_sig.name, {"order_id": oid, "reason": "ordered by mistake"}, result=second, idx=1),
        _call(
            cancel_sig.name,
            {"order_id": "#W0000000", "reason": "no longer needed"},
            error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"),
            idx=2,
        ),
    ]
    source = ce.module_source(schema, [cancel_sig], {cancel_sig.name: CANCEL_BODY})
    box = ce.Sandbox(source, db, workdir)
    shown, held_out = ce.split_calls(calls)
    gates = ce.run_gates(source, box, shown, held_out, schema)
    assert all(g.passed for g in gates), [g.failures for g in gates if not g.passed]
    fidelity = [g for g in gates if g.stage == "replay_fidelity"]
    # D219: a half with no call of its kind is unmeasured, not perfect. The held-out split here is
    # the one recorded error call, so its success half has nothing in it and says so.
    for ruling in fidelity:
        measured = [half for half in ("success", "error")
                    if ruling.metrics[f"{half}_fidelity"] is not None]
        assert measured, ruling.metrics
        assert all(ruling.metrics[f"{half}_fidelity"] == 1.0 for half in measured)
        assert ruling.metrics["not_measured"] == [half for half in ("success", "error")
                                                  if half not in measured]


def test_the_same_write_call_twice_answers_from_the_starting_state_both_times(
    cancel_world, cancel_sig, workdir
):
    """The memo may not hand the second call the world the first call left behind."""
    db, schema, order = cancel_world
    args = {"order_id": order["order_id"], "reason": "no longer needed"}
    calls = [_call(cancel_sig.name, args, idx=0), _call(cancel_sig.name, args, idx=1)]
    source = ce.module_source(schema, [cancel_sig], {cancel_sig.name: CANCEL_BODY})
    results = ce.Sandbox(source, db, workdir).run(calls)
    assert [r["ok"] for r in results] == [True, True], results
    assert [r["value"]["status"] for r in results] == ["cancelled", "cancelled"]


def test_a_row_the_corpus_shows_in_two_versions_replays_on_each_tasks_own_world(
    make_test_model, sample, sigs, workdir
):
    """The March Task saw the order pending, the June Task saw it cancelled; one body serves both."""
    pending = sample["by_status"]["pending"]
    oid = pending["order_id"]
    other = sample["by_status"]["delivered"]
    db = {"orders": {oid: pending, other["order_id"]: other}, "users": {}, "products": {}}
    schema = _schema_for(db)
    march = _trace("A", [
        _call("get_order_details", {"order_id": oid}, result=pending, idx=0),
        _call("get_order_details", {"order_id": other["order_id"]}, result=other, idx=2),
    ])
    june = _trace("C", [_call("get_order_details", {"order_id": oid},
                              result=dict(pending, status="cancelled"), idx=1)])
    tasks = [Task(id="tA", run_ids=["A"]), Task(id="tC", run_ids=["C"])]
    state = ce.build_starting_state([march, june], schema, workdir, tasks=tasks, tool_sigs=sigs)
    assert state.db["orders"][oid]["status"] == "cancelled"  # latest observation wins in the shared world

    calls = [
        march.tool_calls[0],
        june.tool_calls[0],
        march.tool_calls[1],
        _call("get_order_details", {"order_id": "#W0000000"},
              error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"), idx=3),
    ]
    states = ce.call_starting_states(
        state.db, state.overlays, ce.overlay_values(workdir), {"c0": "tA", "c1": "tC", "c2": "tA"}
    )
    model = make_test_model([CORRECT_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], calls, schema, state.db, workdir / "tool",
                            call_states=states)
    assert build.assisted is False, [g.failures for g in build.gates if not g.passed]
    assert len(model.calls) == 1


# --- a recorded invalid_arguments call is an answer, not a crash (D67) ---


def test_a_recorded_invalid_arguments_call_does_not_fail_the_executes_gate(sample, sigs, workdir):
    order = sample["by_status"]["pending"]
    db = {"orders": {order["order_id"]: order}, "users": {}, "products": {}}
    schema = _schema_for(db)
    calls = [
        _call("get_order_details", {"order_id": order["order_id"]}, result=order, idx=0),
        _call("get_order_details", {"order": order["order_id"]},
              error=ToolCallError(**{"class": "invalid_arguments"},
                                  payload="unexpected keyword argument 'order'"), idx=1),
    ]
    source = ce.module_source(schema, sigs[:1], {"get_order_details": CORRECT_BODY})
    box = ce.Sandbox(source, db, workdir)
    executes = ce.gate_executes_on_s0(box, calls)
    assert executes.passed is True, executes.failures
    replay = ce.gate_replay_fidelity(box, calls, schema)
    assert replay.passed is True, replay.failures
    assert replay.metrics["error_matches"] == 1


# --- a mined column never rejects a real row (D72) ---


def test_a_float_row_and_an_int_row_both_load_under_mined_string_and_int_samples(sigs, workdir):
    db = {"orders": {"#W1": {"order_id": "#W1", "amount": 10}, "#W2": {"order_id": "#W2", "amount": 10.5}},
          "users": {"a_b_1": {"user_id": "a_b_1", "zip": "94016"}, "c_d_2": {"user_id": "c_d_2", "zip": 94016}}}
    schema = EntitySchema(
        tables=["orders", "users"],
        columns=[
            Column(table="orders", name="order_id", class_="hard", samples=["#W1"]),
            Column(table="orders", name="amount", class_="hard", samples=[10, 20]),
            Column(table="users", name="user_id", class_="hard", samples=["a_b_1"]),
            Column(table="users", name="zip", class_="hard", samples=["94016", "10001"]),
        ],
        id_patterns={"orders": r"^#W\d+$"},
    )
    assert "Optional[Any]" in ce.render_data_model(schema)
    calls = [_call("get_order_details", {"order_id": "#W2"}, result=db["orders"]["#W2"], idx=0)]
    source = ce.module_source(schema, sigs[:1], {"get_order_details": CORRECT_BODY})
    gate = ce.gate_executes_on_s0(ce.Sandbox(source, db, workdir), calls)
    assert gate.passed is True, gate.failures


# --- the sandbox refuses a result file it did not write ---


def test_a_body_that_writes_the_result_file_itself_is_refused(sample, sigs, workdir):
    order = sample["by_status"]["pending"]
    db = {"orders": {order["order_id"]: order}, "users": {}, "products": {}}
    schema = _schema_for(db)
    calls = [_call("get_order_details", {"order_id": f"#W000000{i}"}, idx=i) for i in range(3)]
    source = ce.module_source(schema, sigs[:1], {"get_order_details": FORGING_BODY})
    box = ce.Sandbox(source, db, workdir)
    with pytest.raises(ce.SandboxError) as caught:
        box.run(calls)
    assert "nonce" in str(caught.value)
    gate = ce.gate_non_trivial(box, calls)
    assert gate.passed is False


# --- the evidence block is not cut (D75 attempt 3, D65) ---


def test_the_full_call_table_attempt_shows_every_shown_call_uncut(
    make_test_model, sample, sigs, db0, schema, workdir
):
    order = sample["by_status"]["pending"]
    orders = [dict(order, order_id=f"#W{i:07d}") for i in range(30)]
    db = {"orders": {o["order_id"]: o for o in orders}, "users": {}, "products": {}}
    calls = [_call("get_order_details", {"order_id": o["order_id"]}, result=o, idx=i)
             for i, o in enumerate(orders)]
    shown, _ = ce.split_calls(calls)
    model = make_test_model([CONSTANT_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], calls, _schema_for(db), db, workdir)
    last = model.calls[-1]["messages"][-1]["content"]
    rows = [line for line in last.splitlines() if line.startswith("- args")]
    assert len(rows) == len(shown) == build.nodes[-1]["evidence_calls"]
    assert json.dumps(orders[0], default=str)[-40:] in last, "a recorded result was cut short"


# --- column classes decide inside a list result too (D73, D84) ---


def test_an_exempt_column_cannot_fail_a_replay_inside_a_list_result(sample, workdir):
    order = sample["by_status"]["pending"]
    now = dict(order, last_seen="2026-06-01T00:00:00Z")
    then = dict(order, last_seen="2026-03-01T00:00:00Z")
    db = {"orders": {order["order_id"]: now}, "users": {}, "products": {}}
    schema = _schema_for(db, exempt=("last_seen",))
    sig = ToolSig(name="list_orders", kind="read", unclassified=False,
                  args_fields=[FieldStat(name="user_id", types=["str"], optional=False)])
    calls = [_call(sig.name, {"user_id": order["user_id"]}, result=[then], idx=0)]
    body = "return [o for o in self.db.orders.values() if o.user_id == user_id]"
    source = ce.module_source(schema, [sig], {sig.name: body})
    gate = ce.gate_replay_fidelity(ce.Sandbox(source, db, workdir), calls, schema)
    assert gate.passed is True, gate.failures


def test_a_hard_column_inside_a_list_result_still_fails_the_replay(sample, workdir):
    order = sample["by_status"]["pending"]
    db = {"orders": {order["order_id"]: dict(order, status="cancelled")}, "users": {}, "products": {}}
    schema = _schema_for(db)
    sig = ToolSig(name="list_orders", kind="read", unclassified=False,
                  args_fields=[FieldStat(name="user_id", types=["str"], optional=False)])
    calls = [_call(sig.name, {"user_id": order["user_id"]}, result=[order], idx=0)]
    body = "return [o for o in self.db.orders.values() if o.user_id == user_id]"
    source = ce.module_source(schema, [sig], {sig.name: body})
    gate = ce.gate_replay_fidelity(ce.Sandbox(source, db, workdir), calls, schema)
    assert gate.passed is False
    assert any("status" in failure for failure in gate.failures)


# --- two Runs of one Task that disagree are recorded, not pinned silently (D74) ---


def test_two_runs_of_one_task_that_saw_different_versions_are_recorded_as_an_assumption(
    sample, schema, sigs, workdir
):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    traces = [
        _trace("A", [_call("get_order_details", {"order_id": oid}, result=order, idx=0)]),
        _trace("B", [_call("get_order_details", {"order_id": oid},
                           result=dict(order, status="cancelled"), idx=1)]),
    ]
    state = ce.build_starting_state(traces, schema, workdir, tasks=[Task(id="t", run_ids=["A", "B"])],
                                    tool_sigs=sigs)
    overlay, values = ce.load_overlay(workdir, "t")
    assert values[overlay.rows[0].version_hash]["status"] == "pending"
    assert any("t" in note and oid in note and "disagree" in note for note in state.assumptions)


def test_two_runs_of_one_task_that_agree_record_nothing(sample, schema, sigs, workdir):
    order = sample["by_status"]["pending"]
    traces = [
        _trace("A", [_call("get_order_details", {"order_id": order["order_id"]}, result=order, idx=0)]),
        _trace("B", [_call("get_order_details", {"order_id": order["order_id"]}, result=order, idx=1)]),
    ]
    state = ce.build_starting_state(traces, schema, workdir, tasks=[Task(id="t", run_ids=["A", "B"])],
                                    tool_sigs=sigs)
    assert not [note for note in state.assumptions if "disagree" in note]


# --- gate 3 checks every call, not the first two ---


def test_a_body_that_is_nondeterministic_on_a_later_call_fails_the_deterministic_gate(
    sample, sigs, workdir
):
    orders = list(sample["orders"].values())
    db = {"orders": {o["order_id"]: o for o in orders}, "users": {}, "products": {}}
    schema = _schema_for(db)
    third = orders[2]["status"]
    body = (f"import random\nif self.db.orders[order_id].status == {third!r}:\n"
            "    return {'order_id': order_id, 'status': str(random.random())}\n"
            "return self.db.orders[order_id]\n")
    calls = [_call("get_order_details", {"order_id": o["order_id"]}, result=o, idx=i)
             for i, o in enumerate(orders)]
    source = ce.module_source(schema, sigs[:1], {"get_order_details": body})
    gates = ce.run_gates(source, ce.Sandbox(source, db, workdir), calls, [], schema)
    deterministic = next(g for g in gates if g.stage == "deterministic")
    assert deterministic.passed is False
    assert deterministic.metrics["calls"] == 3


# --- env_id moves when the world moves (design section 5) ---


def test_env_id_and_files_follow_the_five_emitted_files():
    schema = EntitySchema(tables=["orders"], columns=[Column(table="orders", name="status", class_="hard")])
    one = {"db.json": '{"orders": {"#W1": {"status": "pending"}}}\n', "policy.md": "p"}
    two = {"db.json": '{"orders": {"#W1": {"status": "cancelled"}}}\n', "policy.md": "p"}
    first = ce.build_environment(schema, [], {}, "policy", files=one)
    second = ce.build_environment(schema, [], {}, "policy", files=two)
    assert first.env_id != second.env_id, "two worlds with different rows shared one env_id"
    assert set(first.files) == {"db.json", "policy.md"}
    assert first.files["policy.md"] == second.files["policy.md"]
    assert first.schema_version == second.schema_version


def test_tau2_files_are_what_emit_writes_and_what_env_id_can_hash(schema, sigs, db0, workdir):
    bundle = _bundle(schema, sigs, db0)
    files = ce.tau2_files(bundle)
    assert set(files) == {"data_model.py", "tools.py", "db.json", "policy.md", "tasks.json"}
    paths = ce.emit_tau2_shape(bundle, workdir / "out")
    for name, text in files.items():
        assert paths[name].read_text(encoding="utf-8") == text


# --- ids the traces referenced but never showed get a tagged synthetic row (D40, D41) ---


def test_an_id_the_traces_only_referenced_gets_a_synthetic_row_shaped_from_the_observed_rows(
    sample, retail_db, sigs, workdir
):
    known = next(iter(sample["users"]))
    unseen = next(k for k in sorted(retail_db["users"]) if k != known)
    schema = _schema_for({"users": sample["users"], "orders": sample["orders"], "products": {}})
    trace = _trace("A", [
        _call("get_user_details", {"user_id": known}, result=sample["users"][known], idx=0),
        _call("list_orders", {"user_id": unseen}, result=[], idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert unseen in state.db["users"], "a read of a referenced id has nothing to answer it"
    assert set(state.db["users"][unseen]) == set(sample["users"][known])
    assert state.db["users"][unseen]["user_id"] == unseen
    assert state.synthetic_rows == [unseen]
    assert unseen in schema.synthetic_rows
    assert any(unseen in note and "synthetic" in note for note in state.assumptions)


def test_an_id_only_a_failed_call_named_is_not_invented(sample, schema, sigs, workdir):
    order = sample["by_status"]["pending"]
    trace = _trace("A", [
        _call("get_order_details", {"order_id": order["order_id"]}, result=order, idx=0),
        _call("get_order_details", {"order_id": "#W0000000"},
              error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"), idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert "#W0000000" not in state.db["orders"]
    assert state.synthetic_rows == []


# --- a description the customer wrote cannot break the code-owned skeleton ---


def test_a_description_holding_a_triple_quote_still_parses(schema, sigs):
    sig = sigs[0].model_copy(update={"description": 'Get """the""" order, path C:\\orders\\'})
    source = ce.module_source(schema, [sig], {sig.name: CORRECT_BODY})
    assert ce.gate_parses(source).passed is True
    assert "the" in source


# --- the toolkit the Router calls holds the Task's own world (D74) ---


def test_load_toolkit_puts_the_tasks_overlay_inside_the_toolkit(sample, schema, sigs, workdir):
    order = sample["by_status"]["pending"]
    oid = order["order_id"]
    traces = [
        _trace("A", [_call("get_order_details", {"order_id": oid}, result=order, idx=0)]),
        _trace("B", [_call("get_order_details", {"order_id": oid},
                           result=dict(order, status="cancelled"), idx=1)]),
    ]
    state = ce.build_starting_state(traces, schema, workdir, tasks=[Task(id="tA", run_ids=["A"])],
                                    tool_sigs=sigs)
    assert state.db["orders"][oid]["status"] == "cancelled"
    overlay, values = ce.load_overlay(workdir, "tA")
    source = ce.module_source(schema, sigs[:1], {"get_order_details": CORRECT_BODY})
    toolkit = ce.load_toolkit(source, state.db, overlay=overlay, overlay_values=values)
    assert toolkit.get_order_details(oid).status == "pending"
    assert ce.load_toolkit(source, state.db).get_order_details(oid).status == "cancelled"


# --- the held-out split stays hidden through the whole repair loop (D51, D75) ---


def _held_out_body(refused_status):
    """A body that reads the world for every id and gets only the held-out call wrong.

    It refuses the one status the shown calls never carry, so the shown split replays whole and the
    hidden call does not. It names no id of its own: a body that memorised the shown ids is refused
    by the memorised_values gate before any call replays (D162), and this fixture is about the
    held-out split staying hidden, not about memorisation.
    """
    return (f"order = self.db.orders.get(order_id)\n"
            f"if order is None or order.status == {refused_status!r}:\n"
            "    raise ValueError('Order not found')\n"
            "return order\n")


@pytest.fixture
def split_world(sample):
    """Four recorded calls whose held-out one is a read the body will get wrong."""
    orders = list(sample["orders"].values())
    db = {"orders": {o["order_id"]: o for o in orders}, "users": {}, "products": {}}
    calls = [_call("get_order_details", {"order_id": o["order_id"]}, result=o, idx=i)
             for i, o in enumerate(orders)]
    calls.append(_call("get_order_details", {"order_id": "#W0000000"},
                       error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"),
                       idx=9))
    return db, _schema_for(db), calls


def test_no_held_out_call_reaches_the_model_through_the_repair_prompts(
    make_test_model, split_world, sigs, workdir
):
    db, schema, calls = split_world
    shown, held_out = ce.split_calls(calls)
    body = _held_out_body(held_out[0].result["status"])
    model = make_test_model([body] * 4)
    build = ce.compile_tool(model, sigs[0], calls, schema, db, workdir)

    assert build.assisted is True
    held_ids = [c.args["order_id"] for c in held_out]
    prompts = [c["messages"][-1]["content"] for c in model.calls]
    assert len(prompts) == 4
    leaked = [i for i, p in enumerate(prompts) for h in held_ids if h in p]
    assert leaked == [], f"held-out arguments reached the model in prompts {leaked}"
    assert "not shown" in prompts[1], "the model was not told that a hidden call failed"


def test_the_node_says_so_when_only_the_held_out_split_failed(
    make_test_model, split_world, sigs, workdir
):
    db, schema, calls = split_world
    shown, held_out = ce.split_calls(calls)
    body = _held_out_body(held_out[0].result["status"])
    model = make_test_model([body] * 4)
    build = ce.compile_tool(model, sigs[0], calls, schema, db, workdir)
    assert build.nodes[0]["evidence"] == "initial"
    assert [node["evidence"] for node in build.nodes[1:]] == [ce.HELD_OUT_LABEL] * 3
    assert build.nodes[1]["evidence_calls"] == len(shown)


def test_the_repair_loop_grows_the_evidence_until_the_cap_refuses_an_attempt(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """D75's grow-then-refuse path: attempt 0 fits, a later attempt does not and is refused whole."""
    shown, _ = ce.split_calls(order_calls)
    cap = ce.prompt_chars(ce.body_messages(sigs[0], shown[:3], schema=schema)) + 50
    model = make_test_model([WRONG_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir, max_evidence_chars=cap)
    assert model.calls, "attempt 0 fitted under the cap and should have been sent"
    assert all(ce.prompt_chars(call["messages"]) <= cap for call in model.calls)
    assert build.nodes[-1]["refused"] is True
    assert len(build.nodes) > 1, "the loop refused before it ever grew the evidence"
    assert build.assisted is True


# --- confinement: the static check that stands in for the deferred sandbox ---

def test_a_body_that_reaches_outside_the_world_is_refused_before_it_runs(schema, sigs, db0, workdir):
    """load_toolkit exec's the module in the Runner's own process, so the body is checked first."""
    body = "import os\nopen('/tmp/x', 'w').write('hi')\nreturn self.db.orders.get(order_id)\n"
    source = ce.module_source(schema, sigs[:1], {sigs[0].name: body})
    assert ce.source_confinement(source) == [
        f"{sigs[0].name} imports os", f"{sigs[0].name} uses open"]
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(source, db0)


def test_a_body_that_walks_the_class_tree_is_refused(schema, sigs, db0, workdir):
    """().__class__.__base__.__subclasses__() is the escape a restricted namespace does not stop."""
    body = ("cls = ().__class__.__base__.__subclasses__()\n"
            "return {'n': len(cls)}\n")
    source = ce.module_source(schema, sigs[:1], {sigs[0].name: body})
    assert any("touches __class__" in line for line in ce.source_confinement(source))
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(source, db0)


def test_a_body_that_walks_the_class_tree_through_getattr_is_refused(schema, sigs, db0, workdir):
    """The same escape as above, spelled through getattr so the literal dunder attribute never
    appears; DENIED_BUILTINS has to name getattr, setattr and delattr, not just the dunder."""
    body = ("cls = getattr(getattr(getattr((), '__class__'), '__base__'), '__subclasses__')()\n"
            "return {'n': len(cls)}\n")
    source = ce.module_source(schema, sigs[:1], {sigs[0].name: body})
    assert any("uses getattr" in line for line in ce.source_confinement(source))
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(source, db0)


def test_the_code_owned_skeleton_is_confined_as_it_stands(schema, sigs, db0):
    """DomainDB.load opens the emitted db.json; that is code, not a model's body, and stays allowed."""
    source = ce.module_source(schema, sigs, {s.name: "return None" for s in sigs})
    assert ce.source_confinement(source) == []
    assert ce.load_toolkit(source, db0) is not None


def test_a_user_requestor_call_is_not_a_row_sighting(sample, schema):
    """Telecom's simulated user calls its own phone tools inside the trace (R33); those results
    describe the user's device, not the customer's system, so they never enter the Starting state."""
    order = next(iter(sample["orders"].values()))
    seen = ToolCall(id="c1", name="get_order_details", args={"order_id": order["order_id"]}, result=order,
                    raw_ptr=PTR)
    user_side = ToolCall(id="c2", name="check_status_bar", args={},
                         result={**order, "status": "on the phone"}, requestor="user", raw_ptr=PTR)
    observations = ce._observations([_trace("t1", [seen, user_side])], schema, set())
    assert observations
    assert all(o.order[1] == 0 for o in observations)


def test_the_mined_id_pattern_actually_guards_referenced_ids():
    """A pattern the miner recorded must reject an argument that is not an id of that table.

    `mine_schema` keys its patterns `table.column` and this guard used to look them up by the table
    alone, so on every mined schema the lookup missed, the pattern came back None and the guard let
    every argument through. The regression this test holds is that the guard does something at all.
    """
    schema = EntitySchema(
        tables=["orders"],
        columns=[Column(table="orders", name="order_id", **{"class": "hard"}, classified_by="rule")],
        id_patterns={"orders.order_id": r"^#W\d+$"},
    )
    traces = [_trace("t1", [
        _call("get_order_details", {"order_id": "#W123"}, result='{"order_id": "#W123"}', idx=0),
        _call("get_order_details", {"order_id": "not-an-order"}, result="{}", idx=1),
    ])]
    assert ce.referenced_ids(traces, schema) == [("orders", "#W123")]


def test_the_id_pattern_lookup_reads_a_hand_written_schema_too():
    """A schema written by hand keys the pattern by the table; both keys must resolve."""
    from kullback.builder.sandbox import id_pattern_for

    by_column = EntitySchema(tables=["orders"], id_patterns={"orders.order_id": r"^#W\d+$"})
    by_table = EntitySchema(tables=["orders"], id_patterns={"orders": r"^#W\d+$"})
    assert id_pattern_for(by_column, "orders", "order_id") == r"^#W\d+$"
    assert id_pattern_for(by_table, "orders", "order_id") == r"^#W\d+$"
    assert id_pattern_for(by_column, "orders", "user_id") is None


def test_a_tool_the_model_wrapper_refuses_on_size_does_not_take_the_build_with_it(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """The live case: budget.py counts the request, compile_env counts the contents, and only the
    first one is authoritative. A refusal from the wrapper has to land as an assisted tool."""
    from kullback.runner.budget import ContextCapExceeded

    class Refusing:
        name = "refusing"
        calls = 0

        def query(self, messages, tools=None, config=None):
            Refusing.calls += 1
            raise ContextCapExceeded(89_645, 80_000, 200_000)

    build = ce.compile_tool(Refusing(), sigs[0], order_calls, schema, db0, workdir)
    assert Refusing.calls == 1, "it should not keep retrying a prompt that is too big"
    assert build.nodes[-1]["refused"] is True
    assert build.assisted is True
    assert "refused, not truncated" in build.nodes[-1]["failures"][0]


def test_a_refusal_on_size_is_the_only_exception_compile_tool_swallows(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    class Broken:
        name = "broken"

        def query(self, messages, tools=None, config=None):
            raise RuntimeError("the provider is down")

    with pytest.raises(RuntimeError, match="the provider is down"):
        ce.compile_tool(Broken(), sigs[0], order_calls, schema, db0, workdir)


def test_a_relative_workdir_still_finds_the_runner_it_just_wrote(tmp_path, monkeypatch, db0, sigs, schema, order_calls):
    """The sandbox subprocess runs with cwd inside its own directory, so a relative path handed in
    would be resolved against that directory a second time and every path would double. The first
    live build failed all sixteen tools this way; every test until now passed an absolute tmp_path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work").mkdir()
    source = ce.module_source(schema, sigs[:1], {"get_order_details": "return {}"})
    box = ce.Sandbox(source, db0, "work")
    assert box.dir.is_absolute() and box.runner.is_file()
    box.run([order_calls[0]])


def test_the_prompt_states_the_rules_the_gate_will_enforce(schema, sigs):
    """Four of sixteen tools were refused on the first live build for reaching for names nothing
    had told the model about. The rules are in the stable prefix, so saying them is free."""
    from kullback.gates.confinement import ALLOWED_IMPORTS, DENIED_BUILTINS

    system = ce.body_messages(sigs[0], [], schema=schema)[0]["content"]
    for name in DENIED_BUILTINS:
        assert name in system, f"the gate denies {name} and the prompt never says so"
    for name in ALLOWED_IMPORTS:
        assert name in system
    assert "dunder" in system


def test_the_rules_in_the_prompt_are_generated_from_the_gate_not_copied():
    """A hand-written copy drifts the first time the gate changes; a generated one cannot."""
    assert "summon_a_demon" in ce._confinement_block(denied=frozenset({"summon_a_demon"}))


# --- the code-owned helpers both loaders bind (runner/arith.py) ---
#
# An invented world of its own, because what is under test is a body calling a function nobody
# imported, not any customer's tools.

QUOTES_SCHEMA = EntitySchema(
    tables=["quotes"],
    columns=[Column(table="quotes", name="quote_id", **{"class": "hard"}, classified_by="rule"),
             Column(table="quotes", name="formula", **{"class": "hard"}, classified_by="rule")],
    id_patterns={"quotes": r"^q\d+$"},
)
QUOTES_SIG = ToolSig(
    name="settle_quote",
    description="Work out what the quote's formula comes to.",
    args_fields=[FieldStat(name="quote_id", types=["str"], optional=False)],
    kind="read",
    unclassified=False,
)
QUOTES_DB = {"quotes": {"q1": {"quote_id": "q1", "formula": "(18.25 * 3) - (7.5 + 2.25)"}}}
ARITH_BODY = ("quote = self.db.quotes[quote_id]\n"
              "return {'quote_id': quote_id, 'total': float(evaluate_arithmetic(quote.formula))}\n")


def _quotes_source(body: str) -> str:
    return ce.module_source(QUOTES_SCHEMA, [QUOTES_SIG], {QUOTES_SIG.name: body})


def test_a_body_that_evaluates_an_expression_passes_the_gate_and_runs_in_the_runners_process():
    """Nothing imports the helper and nothing binds it in the module, so the gate has to know it."""
    source = _quotes_source(ARITH_BODY)
    assert ce.source_confinement(source) == []
    assert gate_confined(source).passed is True
    toolkit = ce.load_toolkit(source, QUOTES_DB)
    assert toolkit.settle_quote("q1") == {"quote_id": "q1", "total": 45.0}


def test_the_same_body_gives_the_same_answer_in_the_sandbox_subprocess(workdir):
    """The subprocess is started with the environment cleared and cannot import the harness, so it
    carries the evaluator's own bytes; a body that agrees here and disagrees there would pass the
    gates and then answer differently on a Run."""
    source = _quotes_source(ARITH_BODY)
    box = ce.Sandbox(source, QUOTES_DB, workdir)
    call = ToolCall(id="c1", name="settle_quote", args={"quote_id": "q1"}, raw_ptr=PTR)
    assert box.run([call]) == [{"ok": True, "value": {"quote_id": "q1", "total": 45.0}}]


def test_the_helper_is_the_only_name_the_loaders_add_and_the_gate_knows_that_name():
    """One list, read by the gate and by whatever binds the functions, so neither can grow alone."""
    assert set(sandbox_mod.HELPERS) == set(PROVIDED_HELPERS)
    assert PROVIDED_HELPERS == {"evaluate_arithmetic"}


def test_a_body_that_reaches_for_eval_or_for_ast_is_refused_as_it_was_before():
    """The helper is not a loosening: the names it exists to replace stay refused."""
    with_eval = _quotes_source("return eval(self.db.quotes[quote_id].formula)\n")
    assert "settle_quote uses eval" in ce.source_confinement(with_eval)
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(with_eval, QUOTES_DB)
    with_ast = _quotes_source("import ast\nreturn ast.parse(self.db.quotes[quote_id].formula)\n")
    assert "settle_quote imports ast" in ce.source_confinement(with_ast)
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(with_ast, QUOTES_DB)


def test_the_prompt_names_the_helper_so_no_body_has_to_write_a_parser():
    """Four attempts of build 12 went on a hand-written parser nobody had told the model about."""
    block = ce._confinement_block()
    for name in PROVIDED_HELPERS:
        assert name in block, f"the loaders bind {name} and the prompt never says so"
    assert "never write a parser" in block
    assert "evaluate_arithmetic" in ce.body_messages(QUOTES_SIG, [], schema=QUOTES_SCHEMA)[0]["content"]


def test_evidence_shrinks_by_whole_calls_rather_than_giving_up(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """The live build refused `get_order_details` at 815,972 characters because the last-resort
    evidence is every shown call. Dropping whole calls keeps the attempt; no call is ever cut."""
    model = make_test_model([CORRECT_BODY] * 4)
    one_call = len(ce._example_block(order_calls[:1]))
    cap = len(ce.body_messages(sigs[0], [], schema=schema)[0]["content"]) + 4 * one_call
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir,
                            max_evidence_chars=cap)
    assert model.calls, "the attempt was taken rather than refused"
    assert build.nodes[0]["refused"] is False
    assert 1 <= build.nodes[0]["evidence_calls"] <= 3
    # what went out is whole calls: the block ends where a call ends, never mid row.
    sent = model.calls[0]["messages"][-1]["content"]
    assert not sent.rstrip().endswith(",")


def test_a_single_call_that_cannot_fit_is_still_refused_not_cut(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model([CORRECT_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir, max_evidence_chars=10)
    assert model.calls == []
    assert build.nodes[-1]["refused"] is True
    assert build.assisted is True


def test_a_scalar_result_tool_is_described_as_a_bare_value_not_an_object():
    """calculate's traces record a bare number ("-121.2"), never {"value": ...}. The prompt has to
    say so, or the model reaches for the one-field-object shape it uses for any other tool whose
    result_schema lists one name, and every replay against the real tool fails."""
    from kullback.builder.mine import SCALAR_RESULT_FIELD

    calc = ToolSig(name="calculate", description="Calculate the result of an expression.",
                   args_fields=[FieldStat(name="expression", types=["str"], optional=False)],
                   result_schema=[FieldStat(name=SCALAR_RESULT_FIELD, types=["str"], count=3, optional=False)],
                   kind="generic", unclassified=False)
    block = ce._tool_block(calc, [_call("calculate", {"expression": "1 + 1"}, result="2")])
    assert "Result fields:" not in block
    assert 'Result: a bare str, returned directly, not as {"value": ...}' in block
    assert "a bare str, not wrapped in an object" in ce._docstring(calc)


def test_the_prompt_says_rows_are_pydantic_models_not_dicts(schema, sigs):
    """modify_pending_order_payment called order.get("payment_history") on the live build and
    crashed with AttributeError. The prompt said self.db holds one dict per table and never said
    what is inside the dict."""
    system = ce.body_messages(sigs[0], [], schema=schema)[0]["content"]
    assert "pydantic model rows" in system
    assert ".get(" in system  # named as what not to do


def test_the_schema_block_shows_a_sample_of_a_dict_shaped_column():
    """products.variants and a mined "items" table carried the same rows on the live corpus, and
    nothing in the block said so beyond the bare column name."""
    schema = EntitySchema(tables=["items", "products"], columns=[
        Column(table="items", name="item_id", **{"class": "hard"}, classified_by="rule"),
        Column(table="products", name="variants", **{"class": "hard"}, classified_by="rule",
               evidence={"types": ["dict"]},
               samples=['{"9612497925":{"item_id":"9612497925","price":50.88}}'])])
    block = ce._schema_block(schema)
    assert 'products.variants looks like: {"9612497925":{"item_id":"9612497925","price":50.88}}' in block


# --- what a column holds is a plain list or dict, and the prompt says so ---


def _library_schema():
    """A lending library: a member's loans are a list of dicts, a title's copies a dict of dicts."""
    return EntitySchema(
        tables=["members", "titles"],
        columns=[
            Column(table="members", name="member_id", **{"class": "hard"}, classified_by="rule",
                   evidence={"types": ["str"]}, samples=["m-1"]),
            Column(table="members", name="loans", **{"class": "hard"}, classified_by="rule",
                   evidence={"types": ["list"]},
                   samples=['[{"copy_id": "c-1", "due": "2026-01-02"}]']),
            Column(table="titles", name="copies", **{"class": "hard"}, classified_by="rule",
                   evidence={"types": ["dict"]},
                   samples=['{"c-1": {"shelf": "A3", "copy_id": "c-1"}}']),
        ],
        id_patterns={"members": r"^m-\d+$", "titles": r"^t-\d+$"},
    )


def test_the_schema_block_names_the_access_form_of_a_list_column_and_of_a_dict_column():
    """The data model types every column Optional[Any], so a column holding a list holds a plain
    list whatever the row around it is. One write tool crashed on all four of its compile attempts
    reading such a column by attribute, as the system rule told it to."""
    block = ce._schema_block(_library_schema())
    assert "members.loans is a list of plain dicts, not model rows" in block
    assert '{"copy_id": "c-1", "due": "2026-01-02"}' in block
    assert 'read element["copy_id"], never element.copy_id' in block
    assert "titles.copies is a dict of plain dicts keyed by copy_id" in block
    assert 'read copies[key]["shelf"], never copies[key].shelf' in block


def test_the_access_form_names_no_field_when_the_mined_sample_does_not_parse():
    """Mining cuts a long nested sample at a length, so the text beside the column is not always
    JSON. What it is stays true, so the line is still said, with no field named."""
    schema = EntitySchema(tables=["members"], columns=[
        Column(table="members", name="loans", **{"class": "hard"}, classified_by="rule",
               evidence={"types": ["list"]}, samples=['[{"copy_id": "c-1", "du...'])])
    block = ce._schema_block(schema)
    assert "members.loans is a plain list, not a model row" in block
    assert "never by attribute" in block


def test_the_system_rule_says_a_row_is_a_model_and_that_what_a_column_holds_is_not():
    """Both halves, or the first one alone sends a body to read a list of dicts by attribute."""
    sig = ToolSig(name="get_member", kind="read", unclassified=False,
                  args_fields=[FieldStat(name="member_id", types=["str"], optional=False)])
    system = ce.body_messages(sig, [])[0]["content"]
    assert "pydantic model rows" in system and ".get(" in system
    assert "read or write a row's field by attribute" in system
    assert "is a plain Python list or dict" in system
    assert "write them by plain assignment" in system


def test_the_stable_prefix_is_the_same_bytes_for_two_tools_of_one_build():
    """D65: the access forms live in the schema block, which is the prefix every tool of a build
    shares, so adding them may not make one tool's prompt differ from another's."""
    schema = _library_schema()
    names = ["get_member", "return_copy"]
    reader = ToolSig(name="get_member", kind="read", unclassified=False,
                     args_fields=[FieldStat(name="member_id", types=["str"], optional=False)])
    writer = ToolSig(name="return_copy", kind="write", unclassified=False,
                     args_fields=[FieldStat(name="copy_id", types=["str"], optional=False)])
    first = ce.body_messages(reader, [_call("get_member", {"member_id": "m-1"}, result={})],
                             schema=schema, tool_names=names)[0]["content"]
    second = ce.body_messages(writer, [_call("return_copy", {"copy_id": "c-1"}, result={})],
                              schema=schema, tool_names=names)[0]["content"]
    assert first == second
    assert "members.loans is a list of plain dicts, not model rows" in first


def test_the_schema_block_stays_quiet_for_a_column_with_no_dict_samples():
    schema = EntitySchema(tables=["orders"], columns=[
        Column(table="orders", name="status", **{"class": "hard"}, classified_by="rule",
               evidence={"types": ["str"]}, samples=["pending"])])
    assert "looks like:" not in ce._schema_block(schema)


def test_constant_evidence_note_speaks_only_when_the_recorded_calls_already_agree(order_calls):
    same = [_call("transfer_to_human_agents", {"summary": s}, result="Transfer successful", idx=i)
            for i, s in enumerate("ab")]
    assert "already answer every one of them the same way" in ce._constant_evidence_note(same)
    assert ce._constant_evidence_note(order_calls) == ""
    assert ce._constant_evidence_note(same[:1]) == ""


def test_a_body_that_is_constant_like_the_recorded_tool_passes_gate_four_first_time(
    make_test_model, workdir
):
    """transfer_to_human_agents's own recorded calls all answer "Transfer successful". A body that
    also answers that way is right, and is not sent back to invent detail (second retail build)."""
    schema = EntitySchema(tables=[], columns=[])
    sig = ToolSig(name="transfer_to_human_agents", kind="generic", unclassified=False,
                  args_fields=[FieldStat(name="summary", types=["str"], optional=False)])
    calls = [_call("transfer_to_human_agents", {"summary": f"issue {i}"},
                   result="Transfer successful", idx=i) for i in range(4)]
    model = make_test_model(['return "Transfer successful"'] * 4)
    build = ce.compile_tool(model, sig, calls, schema, {}, workdir)
    assert len(model.calls) == 1 and build.assisted is False
    gate_four = next(g for g in build.gates if g.stage == "non_trivial")
    assert gate_four.passed is True and gate_four.metrics["recorded_constant"] is True


def test_the_example_block_peels_the_transport_error_prefix_off_the_payload():
    """tau2 wraps every raised exception as "Error: {e}" before the agent sees it. A model that
    copies the payload verbatim raises ValueError("Error: User not found") where the real tool
    raises ValueError("User not found"); seven bodies did on the first live build."""
    calls = [_call("find_user_id_by_email", {"email": "a@example.com"},
                   error=ToolCallError(**{"class": "not_found_entity"}, payload="Error: User not found")),
             _call("get_order_details", {"order_id": "#W0000000"},
                   error=ToolCallError(**{"class": "not_found_entity"}, payload="Error: Order not found"),
                   idx=1)]
    block = ce._example_block(calls)
    assert "'User not found'" in block and "'Order not found'" in block
    assert "Error: User not found" not in block


def test_a_single_recorded_error_is_shown_whole_because_one_message_is_its_own_prefix():
    calls = [_call("find_user_id_by_email", {"email": "a@example.com"},
                   error=ToolCallError(**{"class": "not_found_entity"}, payload="Error: User not found"))]
    assert "'Error: User not found'" in ce._example_block(calls)
    assert "'User not found'" in ce._example_block(calls, error_prefix="Error: ")


@pytest.mark.parametrize("payloads, expected", [
    (["Error: User not found", "Error: Order not found"], "Error: "),
    (["Error: User not found", "Error: User not found"], "Error: "),
    (["ValueError: Error: a", "ValueError: Error: b"], "ValueError: Error: "),
    (["User not found", "Order not found"], ""),
    (["Error: User not found"], ""),
    (["Error: ", "Error: x"], ""),
    (["Error: x", {"message": "Error: y"}], ""),
])
def test_the_shared_error_prefix_is_read_off_the_corpus_not_written_in(payloads, expected):
    """D51: which wrapper a customer's transport adds is not known in advance. The rule is the
    prefix every recorded error shares, cut at a ': ' boundary that leaves a message behind."""
    calls = [_call("t", {"i": i}, error=ToolCallError(**{"class": "unknown"}, payload=payload), idx=i)
             for i, payload in enumerate(payloads)]
    assert ce.shared_error_prefix(calls) == expected


def test_a_repair_prompt_names_an_allowed_module_the_body_forgot_to_import():
    """transfer_to_human_agents used re.findall and never imported re: 25 of 25 replays died on
    the same NameError, and the retry handed back the traceback instead of the one-line fix."""
    from kullback.runner.gate_support import gate
    gates = [gate("executes", ["call 0: NameError: name 're' is not defined",
                               "call 1: NameError: name 'requests' is not defined"])]
    text = ce._failure_text(gates)
    assert "`re` is on the allowed import list but the body never imported it" in text
    assert "`requests` is on the allowed import list" not in text
    assert ce._failure_text([gate("executes", [])]) == ""


def _retail_nesting_traces():
    variants = {"1906487464": {"item_id": "1906487464", "price": 102.02, "available": True},
                "2820119811": {"item_id": "2820119811", "price": 94.68, "available": False}}
    product = {"name": "Tea Kettle", "product_id": "9832717871", "variants": variants}
    item = {"item_id": "1906487464", "price": 102.02, "available": True,
            "options": {"capacity": "2 liters"}}
    return [_trace("t1", [_call("get_product_details", {"product_id": "9832717871"}, result=product),
                          _call("get_item_details", {"item_id": "1906487464"}, result=item, idx=1)])]


def test_mining_reads_a_nested_home_off_the_traces():
    """items sat under products.variants in every get_product_details result of the first live
    build, keyed by item_id, and the schema still said items was a top-level table: nine calls
    out of nine raised 'not found' on the real database (docs/live-build.md, schema_shape)."""
    from kullback.builder import mine
    schema = mine.mine_schema(_retail_nesting_traces())
    assert schema.homes == {"items": "products.variants"}
    assert "items" in schema.tables and "products" in schema.tables
    names = {c.name for c in schema.columns if c.table == "items"}
    assert {"item_id", "price", "available", "options"} <= names


def test_a_nested_collection_never_shown_on_its_own_stays_a_column_of_its_parent():
    from kullback.builder import mine
    traces = _retail_nesting_traces()
    traces[0].tool_calls = traces[0].tool_calls[:1]
    schema = mine.mine_schema(traces)
    assert schema.homes == {}
    assert "items" not in schema.tables


def test_the_schema_block_says_where_a_table_with_a_home_is_stored():
    from kullback.builder import mine
    block = ce._schema_block(mine.mine_schema(_retail_nesting_traces()))
    assert "items rows are stored inside products.variants, keyed by item_id" in block
    assert "self.db.products.values()" in block and ".variants" in block
    assert "may be empty on the customer's real database" in block


def test_the_starting_state_folds_a_standalone_row_into_its_home(workdir):
    from kullback.builder import mine
    traces = _retail_nesting_traces()
    schema = mine.mine_schema(traces)
    state = ce.build_starting_state(traces, schema, workdir, synthetic=False)
    assert "1906487464" not in state.db["items"]
    nested = state.db["products"]["9832717871"]["variants"]["1906487464"]
    assert nested["options"] == {"capacity": "2 liters"}, "the standalone copy's extra field is kept"
    assert nested["price"] == 102.02
    assert any("is stored under products.variants" in a for a in state.assumptions)


def test_a_row_whose_parent_was_never_shown_stays_in_its_table():
    db = {"items": {"x1": {"item_id": "x1"}, "y1": {"item_id": "y1"}},
          "products": {"p": {"product_id": "p", "variants": {"x1": {"item_id": "x1", "price": 1}}}}}
    schema = EntitySchema(tables=["items", "products"], homes={"items": "products.variants"})
    assert ce.fold_into_homes(db, schema) == [("items", "x1", "products.variants")]
    assert list(db["items"]) == ["y1"]


def test_the_example_block_leaves_a_payload_with_no_transport_prefix_alone():
    calls = [_call("get_order_details", {"order_id": "#W0000000"},
                   error=ToolCallError(**{"class": "not_found_entity"}, payload="Order not found"))]
    assert "'Order not found'" in ce._example_block(calls)


def test_the_example_block_does_not_peel_a_prefix_out_of_a_json_payload():
    """D67's code-classified errors carry a structured payload; the prefix is a string artifact."""
    calls = [_call("find_user_id_by_email", {"email": "a@example.com"},
                   error=ToolCallError(**{"class": "not_found_entity"},
                                       payload={"message": "Error: User not found"}))]
    assert "{'message': 'Error: User not found'}" in ce._example_block(calls)


def test_the_system_prompt_says_the_transport_prefix_is_already_removed():
    assert "same transport prefix" in ce._SYSTEM and "prefix removed" in ce._SYSTEM


# --- the Starting state grows on request, and the ids are tagged (D40, D107) ---

def _twelve_user_traces():
    first = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Eli", "Ivy", "Max", "Uma", "Kai", "Lea", "Tom"]
    last = ["Chen", "Diaz", "Khan", "Lee", "Moss", "Nair", "Ortiz", "Park", "Quinn", "Reyes", "Sato", "Voss"]
    traces = []
    for n, (f, l_) in enumerate(zip(first, last, strict=True)):
        uid = f"{f.lower()}_{l_.lower()}_{1000 + n}"
        row = {"user_id": uid, "name": {"first_name": f, "last_name": l_},
               "email": f"{f.lower()}.{l_.lower()}{2000 + n}@example.com", "tier": ["gold", "basic"][n % 2]}
        traces.append(_trace(f"u{n}", [_call("get_user_details", {"user_id": uid}, result=row, idx=n)]))
    return traces


def test_the_starting_state_grows_to_the_asked_size_and_tags_every_grown_row(tmp_path):
    from kullback.builder import mine
    traces = _twelve_user_traces()
    schema = mine.mine_schema(traces)
    state = ce.build_starting_state(traces, schema, tmp_path, grow={"users": 20}, grow_seed=1)
    assert len(state.db["users"]) == 20
    assert len(state.synthetic_rows) == 8
    assert set(state.synthetic_rows) <= set(state.db["users"]) and set(state.synthetic_rows) <= set(schema.synthetic_rows)
    assert any("8 synthetic rows" in note and "D107" in note for note in state.assumptions)
    written = json.loads((tmp_path / "synthetic.json").read_text())
    assert written["added"] == {"users": 8} and written["checks"]["ok"] is True
    grown = state.db["users"][state.synthetic_rows[0]]
    first, last = grown["name"]["first_name"], grown["name"]["last_name"]
    assert grown["user_id"].startswith(f"{first.lower()}_{last.lower()}_")
    assert grown["email"].startswith(f"{first.lower()}.{last.lower()}")


def test_without_grow_nothing_is_grown_and_no_synthetic_file_is_written(tmp_path):
    from kullback.builder import mine
    traces = _twelve_user_traces()
    state = ce.build_starting_state(traces, mine.mine_schema(traces), tmp_path)
    assert len(state.db["users"]) == 12 and state.synthetic_rows == []
    assert not (tmp_path / "synthetic.json").exists()


# --- gate 4 against the recording, gate 6 the refusal probe ---

STRICT_CANCEL_BODY = """
if order_id not in self.db.orders:
    raise ValueError("Order not found")
order = self.db.orders[order_id]
order.status = "cancelled"
return order
"""

PERMISSIVE_CANCEL_BODY = """
order = self.db.orders.get(order_id)
if order is None:
    return {"order_id": order_id, "status": "cancelled"}
order.status = "cancelled"
return order
"""


def _cancel_calls(sample):
    pending = sample["by_status"]["pending"]
    return [_call("cancel_pending_order", {"order_id": pending["order_id"], "reason": "no longer needed"},
                  result=dict(pending, status="cancelled"), idx=idx) for idx in range(2)]


CANCEL_SIG = ToolSig(name="cancel_pending_order", kind="write", unclassified=False,
                     args_fields=[FieldStat(name="order_id", types=["str"], optional=False),
                                  FieldStat(name="reason", types=["str"], optional=False)])


def _both_tools(schema, sigs, db0, workdir, cancel_body):
    source = ce.module_source(schema, [sigs[0], CANCEL_SIG],
                              {"get_order_details": CORRECT_BODY, "cancel_pending_order": cancel_body})
    return source, ce.Sandbox(source, db0, workdir)


def test_a_constant_body_passes_non_trivial_when_the_recorded_tool_was_constant(schema, sigs, db0, workdir, order_calls):
    """The recording is the standard: a tool that answered every call the same way is faithfully constant."""
    constant = [c.model_copy(update={"result": {"order_id": "#W0000000", "status": "delivered"}})
                for c in order_calls[:3]]
    source, box = _sandbox(schema, sigs, db0, workdir, CONSTANT_BODY)
    result = ce.gate_non_trivial(box, constant)
    assert result.passed is True and result.metrics["recorded_constant"] is True
    assert result.metrics["recorded_answers"] == 1 and result.metrics["arg_sets"] == 3


# --- gate 7: a body may not memorise the recordings (D162) ---


MEMORISING_BODY = """
statuses = {"#W1006327": "pending", "#W2611340": "processed"}
return {"order_id": order_id, "status": statuses[order_id]}
"""


def test_the_compile_stage_retries_a_memorising_body_and_its_red_light_asks_for_a_recompile(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """Build 13's tool that replaces items on an order held three recorded item ids in a dict, so
    every other id raised KeyError: 126 of the 150 replay calls that differed. A body like it is
    refused before a subprocess runs, every attempt, and the red light names the verb that owns it."""
    model = make_test_model([MEMORISING_BODY] * 4)
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)

    assert len(model.calls) == 4, "the stage retried the refused body up to max_attempts"
    assert build.assisted is True
    ruling = next(g for g in build.gates if g.stage == "compile_tools.memorised_values")
    assert ruling.passed is False
    assert any("'#W1006327'" in f for f in ruling.failures)
    assert all("does not execute" not in f for f in ruling.failures), "refused before it ran anywhere"
    assert [g.stage for g in build.gates][-1] == "compile_tools.memorised_values", \
        "no later gate ran on a body the memorised_values gate refused"
    failure_shown = model.calls[1]["messages"][-1]["content"]
    assert "gate compile_tools.memorised_values" in failure_shown

    (workdir / "gates.json").write_text(json.dumps([as_dict(g) for g in build.gates]), encoding="utf-8")
    lights = [light for light in builder_tools.red_lights(workdir)
              if light.stage == "compile_tools.memorised_values"]
    assert lights, "a refused body leaves no red light"
    assert {light.verb for light in lights} == {"repair_recompile"}
    assert {light.target for light in lights} == {"get_order_details"}


def test_reference_args_are_the_arguments_whose_values_the_world_holds(schema, sigs, db0, workdir, sample):
    source, box = _both_tools(schema, sigs, db0, workdir, STRICT_CANCEL_BODY)
    assert list(sandbox_mod.reference_args(box, _cancel_calls(sample))) == ["order_id"]  # reason is free text


def test_the_refusal_probe_fails_a_body_that_writes_on_an_id_nobody_holds(schema, sigs, db0, workdir, sample):
    calls = _cancel_calls(sample)
    for body, expected in ((STRICT_CANCEL_BODY, True), (PERMISSIVE_CANCEL_BODY, False)):
        source, box = _both_tools(schema, sigs, db0, workdir, body)
        result = sandbox_mod.gate_refuses_unknown(box, calls)
        assert result.passed is expected, body
        assert result.metrics["reference_args"] == 1
    assert "accepted order_id=" in result.failures[0]


def test_the_refusal_probe_runs_on_write_tools_only(schema, sigs, db0, workdir, sample):
    calls = _cancel_calls(sample)
    source, box = _both_tools(schema, sigs, db0, workdir, PERMISSIVE_CANCEL_BODY)
    assert "refuses_unknown" not in [g.stage for g in ce.run_gates(source, box, calls, [], schema)]
    probed = ce.run_gates(source, box, calls, [], schema, probe_refusals=True)
    assert probed[-1].stage == "refuses_unknown" and probed[-1].passed is False


def test_build_environment_lists_the_assisted_tools_it_is_told(schema, sigs):
    env = ce.build_environment(schema, sigs, {s.name: "pass" for s in sigs}, "policy",
                               assisted_tools=["cancel_pending_order"])
    assert env.assisted_tools == ["cancel_pending_order"]


# --- the builder tools: lookup_rows and test_body, in a bounded loop (D117) ---


def _tool_call_reply(call_id, name, arguments):
    return {"content": None, "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def test_a_lookup_rows_call_returns_the_row_and_the_body_still_gets_gated(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    order_id = order_calls[0].args["order_id"]
    model = make_test_model([
        _tool_call_reply("tc1", "lookup_rows", {"table": "orders", "key": order_id}),
        CORRECT_BODY,
    ])
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)

    tool_messages = [m for call in model.calls for m in call["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    result_text = tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "tc1"
    assert order_id in result_text and "status" in result_text

    assert build.assisted is False
    assert build.body.strip() == CORRECT_BODY.strip()
    assert build.nodes[0]["tool_uses"] == [
        {"name": "lookup_rows", "arguments": {"table": "orders", "key": order_id},
         "result_chars": len(result_text)}
    ]
    # two rounds of the tool loop, one model call each
    assert len(model.calls) == 2
    assert model.calls[0]["tools"] == ce.BUILDER_TOOLS


def test_a_test_body_call_returns_the_gate_failure_and_never_names_a_held_out_call(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    shown, held_out = ce.split_calls(order_calls)
    hidden_id = held_out[0].args["order_id"]
    model = make_test_model([
        _tool_call_reply("tc1", "test_body", {"body": WRONG_BODY}),
        CORRECT_BODY,
    ])
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)

    tool_messages = [m for call in model.calls for m in call["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    result_text = tool_messages[0]["content"]
    assert "gate" in result_text
    assert hidden_id not in result_text, "a held-out call's argument reached the model through test_body"

    assert build.assisted is False
    assert build.body.strip() == CORRECT_BODY.strip()
    tool_use = build.nodes[0]["tool_uses"][0]
    assert tool_use["name"] == "test_body"
    assert tool_use["arguments"] == {"body_sha256": hashlib.sha256(WRONG_BODY.encode("utf-8")).hexdigest()}
    assert "body" not in tool_use["arguments"]


def test_the_tool_rounds_cap_ends_the_attempt_with_no_body_submitted(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model(
        [_tool_call_reply("tc", "lookup_rows", {"table": "orders"})], loop=True
    )
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)

    # Every attempt gets its rounds, is told no body arrived, and the tool ends assisted with no
    # body: the stage gate then names it, as it would any tool the Builder could not write.
    attempts = ce.MAX_REPAIR_ATTEMPTS + 1  # the first attempt plus the repairs (D75)
    assert len(model.calls) == ce.MAX_TOOL_ROUNDS * attempts
    assert len(build.nodes) == attempts
    assert build.assisted is True and build.body == ""
    assert all(n["failures"] == ["no body was submitted"] and not n.get("refused") for n in build.nodes)
    assert len(build.nodes[-1]["tool_uses"]) == ce.MAX_TOOL_ROUNDS
    retry = model.calls[ce.MAX_TOOL_ROUNDS]["messages"]
    assert retry[-2]["content"] == "(no body was submitted)" and "send the body as your reply" in retry[-1]["content"]


def test_the_last_tested_draft_is_the_body_when_the_rounds_run_out(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """The sixth retail build: three drafts tested, a row looked up, the rounds gone, no reply."""
    model = make_test_model(
        [_tool_call_reply("t1", "test_body", {"body": WRONG_BODY}),
         _tool_call_reply("t2", "test_body", {"body": CORRECT_BODY})]
        + [_tool_call_reply("l", "lookup_rows", {"table": "orders"})] * (ce.MAX_TOOL_ROUNDS - 2)
    )
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)

    assert len(model.calls) == ce.MAX_TOOL_ROUNDS and len(build.nodes) == 1
    assert build.nodes[0]["body_from_draft"] is True and build.nodes[0]["passed"] is True
    assert build.assisted is False and build.body.strip() == CORRECT_BODY.strip()


def test_builder_tools_false_sends_no_tools_and_keeps_the_old_behaviour(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    model = make_test_model([CORRECT_BODY])
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir, builder_tools=False)

    assert len(model.calls) == 1
    assert not model.calls[0]["tools"]
    assert build.body.strip() == CORRECT_BODY.strip()
    assert build.assisted is False
    assert "tool_uses" not in build.nodes[0]


# --- per-call replay outcomes, the per-Task grain of the fidelity ruling (D171) ---

LIBRARY_DB = {
    "loans": {
        "L1": {"loan_id": "L1", "member_id": "m_ada_1", "due_on": "2026-01-05"},
        "L2": {"loan_id": "L2", "member_id": "m_bo_2", "due_on": "2026-02-11"},
        "L3": {"loan_id": "L3", "member_id": "m_cy_3", "due_on": "2026-03-19"},
    }
}
LIBRARY_SCHEMA = EntitySchema(
    tables=["loans"],
    columns=[Column(table="loans", name=name, **{"class": "hard"}, classified_by="rule")
             for name in ("loan_id", "member_id", "due_on")],
    id_patterns={"loans": r"^L\d+$"},
)
LIBRARY_SIG = ToolSig(
    name="get_loan_details",
    description="Get one loan by its id.",
    args_fields=[FieldStat(name="loan_id", types=["str"], optional=False)],
    kind="read",
    unclassified=False,
)
# Right on every loan but one, which is the shape a corpus-wide fidelity number hides: the tool is
# assisted, and only the Task whose Trace asked for that loan is answered differently.
ONE_LOAN_WRONG = """
loan = self.db.loans[loan_id]
if loan_id == "L2":
    return {"loan_id": loan.loan_id, "member_id": loan.member_id, "due_on": "2099-12-31"}
return loan
"""


def _loan_calls():
    return [_call("get_loan_details", {"loan_id": loan_id}, result=row, idx=index)
            for index, (loan_id, row) in enumerate(sorted(LIBRARY_DB["loans"].items()))]


def test_replay_outcomes_names_the_one_recorded_call_a_body_answers_differently(workdir):
    rows = ce.replay_outcomes(LIBRARY_SIG, ONE_LOAN_WRONG, _loan_calls(), LIBRARY_SCHEMA, LIBRARY_DB, workdir)
    assert [(row["call_id"], row["replayed"]) for row in rows] == [("c0", True), ("c1", False), ("c2", True)]
    assert "due_on" in rows[1]["detail"] and rows[0]["detail"] == ""


def test_a_call_replay_outcomes_counts_replayed_is_a_call_the_corpus_gate_counts_matched(workdir, tmp_path):
    """The two grains are the same ruling: what the per-call rows say has to add up to the gate's
    own success count, or a Task could be cleared by a call the corpus gate failed."""
    calls = _loan_calls()
    source = ce.module_source(LIBRARY_SCHEMA, [LIBRARY_SIG], {LIBRARY_SIG.name: ONE_LOAN_WRONG})
    box = ce.Sandbox(source, LIBRARY_DB, tmp_path / "corpus")
    corpus = ce.gate_replay_fidelity(box, calls, LIBRARY_SCHEMA)
    rows = ce.replay_outcomes(LIBRARY_SIG, ONE_LOAN_WRONG, calls, LIBRARY_SCHEMA, LIBRARY_DB, workdir)
    assert corpus.passed is False
    assert sum(1 for row in rows if row["replayed"]) == corpus.metrics["success_matches"]


def test_a_tool_with_no_body_replays_none_of_its_recorded_calls(workdir):
    rows = ce.replay_outcomes(LIBRARY_SIG, "", _loan_calls(), LIBRARY_SCHEMA, LIBRARY_DB, workdir)
    assert [row["replayed"] for row in rows] == [False, False, False]
    assert all("no body" in row["detail"] for row in rows)


def test_a_compiled_tool_carries_one_call_outcome_per_recorded_call(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """D171: every build carries the rows, and a build that cleared the gates replayed every call."""
    build = ce.compile_tool(make_test_model([CORRECT_BODY]), sigs[0], order_calls, schema, db0, workdir)
    assert build.assisted is False
    assert [row["call_id"] for row in build.call_outcomes] == [c.id for c in order_calls]
    assert all(row["replayed"] for row in build.call_outcomes)
    written = json.loads((workdir / ce.NODE_DIR / "get_order_details.json").read_text(encoding="utf-8"))
    assert written["call_outcomes"] == build.call_outcomes


# --- rows whose identity is more than one column (composite row keys) ---------

def dock_schema(**extra) -> EntitySchema:
    """An invented `docks` table let out per shift, keyed by dock_id and shift together."""
    names = {"dock_id": "hard", "shift": "hard", "seats": "hard", "zone": "hard", "updated_at": "exempt"}
    return EntitySchema(
        tables=["docks"],
        columns=[Column(table="docks", name=name, **{"class": kind}, classified_by="rule")
                 for name, kind in sorted(names.items())],
        id_patterns={"docks.dock_id": r"^dock_\d$"},
        composite_keys={"docks": ["dock_id", "shift"]},
        **extra,
    )


def a_dock(dock_id: str, seats: int, shift=None) -> dict:
    return {"dock_id": dock_id, "shift": shift, "seats": seats, "zone": "north"}


def test_a_composite_key_keeps_both_versions_of_one_id(workdir):
    schema = dock_schema()
    trace = _trace("A", [
        _call("list_docks", {"shift": "early"}, result=[a_dock("dock_1", 4)], idx=0),
        _call("list_docks", {"shift": "late"}, result=[a_dock("dock_1", 1)], idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[], synthetic=False)
    assert sorted(state.db["docks"]) == ["dock_1|early", "dock_1|late"]
    assert state.db["docks"]["dock_1|early"]["seats"] == 4
    assert state.db["docks"]["dock_1|late"]["seats"] == 1


def test_a_key_column_the_row_leaves_null_is_taken_from_the_call_that_returned_it():
    schema = dock_schema()
    row = a_dock("dock_1", 4)
    assert ce.match_table(schema, row, {"shift": "early"}) == ("docks", "dock_1|early")
    assert ce.match_table(schema, row) == ("docks", "dock_1|")
    assert ce.match_table(schema, dict(row, shift="late"), {"shift": "early"}) == ("docks", "dock_1|late")


def test_a_single_key_table_is_keyed_exactly_as_it_was(schema, sample):
    order = sample["by_status"]["pending"]
    assert ce.match_table(schema, order, {"anything": "at all"}) == ("orders", order["order_id"])


def test_a_partial_sighting_folds_into_the_rows_whose_known_key_columns_match(workdir):
    """It is not a new row and not a contradiction: it fills only what the keyed rows never showed."""
    schema = dock_schema()
    trace = _trace("A", [
        _call("list_docks", {"shift": "early"}, result=[a_dock("dock_1", 4)], idx=0),
        _call("list_docks", {"shift": "late"}, result=[a_dock("dock_1", 1)], idx=1),
        _call("find_dock", {"dock_id": "dock_1"}, result={"dock_id": "dock_1", "zone": "south",
                                                          "seats": 9, "berth": "outer"}, idx=2),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[], synthetic=False)
    assert sorted(state.db["docks"]) == ["dock_1|early", "dock_1|late"]
    assert state.db["docks"]["dock_1|early"]["seats"] == 4  # what the keyed sighting showed stands
    assert state.db["docks"]["dock_1|early"]["berth"] == "outer"  # what it never showed is filled
    assert any("without every part of its key" in line for line in state.assumptions)


def test_a_partial_sighting_no_keyed_row_matches_is_kept_as_the_partial_sighting_it_is(workdir):
    schema = dock_schema()
    trace = _trace("A", [
        _call("list_docks", {"shift": "early"}, result=[a_dock("dock_1", 4)], idx=0),
        _call("find_dock", {"dock_id": "dock_2"}, result={"dock_id": "dock_2", "seats": 9}, idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[], synthetic=False)
    assert sorted(state.db["docks"]) == ["dock_1|early", "dock_2|"]


def test_an_id_a_call_named_without_every_key_column_owes_no_synthetic_row(workdir):
    schema = dock_schema()
    trace = _trace("A", [
        _call("list_docks", {"shift": "early"}, result=[a_dock("dock_1", 4)], idx=0),
        _call("find_dock", {"dock_id": "dock_9"}, result=None, idx=1),
        _call("find_dock", {"dock_id": "dock_8", "shift": "late"}, result=None, idx=2),
    ])
    assert ce.referenced_ids([trace], schema) == [("docks", "dock_8|late")]
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[])
    assert sorted(state.db["docks"]) == ["dock_1|early", "dock_8|late"]
    assert state.db["docks"]["dock_8|late"]["shift"] == "late"  # the key's own parts, not the modal row
    assert state.synthetic_rows == ["dock_8|late"]


def test_the_tables_block_tells_the_body_writer_how_to_form_a_composite_key():
    block = ce._schema_block(dock_schema())
    assert 'f"{dock_id}|{shift}"' in block
    assert "never look one up by dock_id alone" in block


def test_the_tables_block_says_nothing_about_keys_for_a_single_key_table(schema):
    assert "keyed by" not in ce._schema_block(schema)


# --- ids named below the top level of a call's arguments ---------------------


def _berth_schema() -> EntitySchema:
    """Two invented tables, one of them let out per shift so its key is two columns (D179)."""
    names = {"docks": ["dock_id", "shift", "seats"], "vessels": ["vessel_id", "hull"]}
    return EntitySchema(
        tables=sorted(names),
        columns=[Column(table=table, name=name, **{"class": "hard"}, classified_by="rule")
                 for table, columns in sorted(names.items()) for name in columns],
        id_patterns={"docks.dock_id": r"^dock_\d+$", "vessels.vessel_id": r"^VSL\d+$"},
        composite_keys={"docks": ["dock_id", "shift"]},
    )


def test_an_id_two_levels_down_in_a_list_of_dicts_is_a_referenced_id():
    schema = _berth_schema()
    trace = _trace("A", [_call("book_berths", {"booking": {"legs": [{"vessel_id": "VSL4"},
                                                                   {"vessel_id": "VSL7"}]}},
                               result={"booked": 2}, idx=0)])
    assert ce.referenced_ids([trace], schema) == [("vessels", "VSL4"), ("vessels", "VSL7")]


def test_a_nested_row_completes_its_composite_key_from_the_object_it_sits_in():
    """Each object in the list names its own shift, so the two rows are two keys, not one."""
    schema = _berth_schema()
    trace = _trace("A", [_call("book_berths", {"legs": [{"dock_id": "dock_4", "shift": "early"},
                                                        {"dock_id": "dock_4", "shift": "late"}]},
                               result={"booked": 2}, idx=0)])
    assert ce.referenced_ids([trace], schema) == [("docks", "dock_4|early"), ("docks", "dock_4|late")]


def test_a_nested_row_takes_the_key_part_the_call_states_once_at_the_top_level():
    schema = _berth_schema()
    trace = _trace("A", [_call("book_berths", {"shift": "late", "legs": [{"dock_id": "dock_4"}]},
                               result={"booked": 1}, idx=0)])
    assert ce.referenced_ids([trace], schema) == [("docks", "dock_4|late")]


def test_a_nested_value_the_id_pattern_rejects_owes_no_row():
    schema = _berth_schema()
    trace = _trace("A", [_call("book_berths", {"legs": [{"vessel_id": "not-a-vessel"}]}, result={}, idx=0)])
    assert ce.referenced_ids([trace], schema) == []


def test_a_nested_id_owes_a_synthetic_row_shaped_like_the_rows_the_traces_showed(workdir):
    schema = _berth_schema()
    trace = _trace("A", [
        _call("find_vessel", {"vessel_id": "VSL1"}, result={"vessel_id": "VSL1", "hull": "steel"}, idx=0),
        _call("book_berths", {"legs": [{"vessel_id": "VSL9"}]}, result={"booked": 1}, idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=[])
    assert sorted(state.db["vessels"]) == ["VSL1", "VSL9"]
    assert state.db["vessels"]["VSL9"]["hull"] == "steel"  # the modal row's shape, the id its own
    assert state.synthetic_rows == ["VSL9"]


# --- a typographic character in generated code (a live build stalled a round on one) ---

# Written as escapes so no editor or lint pass can quietly alter the characters under test.
DASH = "\u2014"
QUOTE = "\u201c"
UNQUOTE = "\u201d"
ELLIPSIS = "\u2026"
PLUS_MINUS = "\u00b1"


def _parses(body: str) -> bool:
    try:
        ast.parse(body)
    except SyntaxError:
        return False
    return True


def test_a_dash_in_code_is_replaced_and_the_same_dash_in_a_string_is_kept():
    body = f'def f(x):\n    note = "keep {DASH} inside strings"\n    return x {DASH} 1  # trailing {DASH}\n'
    fixed, note = ce.sanitize_body(body)
    assert _parses(fixed)
    assert f'"keep {DASH} inside strings"' in fixed
    assert "return x - 1  # trailing -" in fixed
    assert "replaced typographic characters outside strings" in note


def test_a_dash_inside_an_f_string_expression_is_replaced_and_the_printed_text_is_kept():
    """The expression inside a replacement field is code, and a confusable there fails the parse
    exactly as one on a line of its own does. What the f-string prints around it is not."""
    body = f'def f(value):\n    return f"keep {DASH} here {{value {DASH} 1}}"\n'
    fixed, note = ce.sanitize_body(body)
    assert _parses(fixed)
    assert fixed == f'def f(value):\n    return f"keep {DASH} here {{value - 1}}"\n'
    assert note is not None


def test_an_f_string_keeps_its_format_spec_its_conversion_and_its_nested_string():
    """Everything an f-string prints stays as written: the spec after the colon, the text after a
    conversion, the doubled braces and a string nested inside a replacement field."""
    for printed in [f'f"{{v:{DASH}>10}}"', f'f"{{v!r}} {DASH} x"', f'f"{{{{{DASH}}}}} x"',
                    f"f\"{{d['a{DASH}b']}}\""]:
        body = f"def f(v, d):\n    return {printed}\n"
        assert ce.sanitize_body(body) == (body, None), printed


def test_a_clean_body_is_returned_untouched_and_unremarked():
    body = "def f(x):\n    return x + 1\n"
    assert ce.sanitize_body(body) == (body, None)


def test_a_non_ascii_character_with_no_replacement_is_named_rather_than_guessed_at():
    body = f"def f(x):\n    return x {PLUS_MINUS} 1\n"
    fixed, note = ce.sanitize_body(body)
    assert fixed == body
    assert "U+00B1" in note


def test_a_body_the_tokenizer_refuses_is_returned_unchanged_and_unremarked():
    """A body that does not tokenize has nothing to stand on, and the compile gate already tells the
    model where it does not parse."""
    assert ce.sanitize_body(f"def f(x):\n  return (x {DASH} 1\n") == (
        f"def f(x):\n  return (x {DASH} 1\n", None)


def test_a_smart_quote_becomes_a_straight_quote_and_an_ellipsis_becomes_three_dots():
    fixed, _ = ce.sanitize_body(f"def f():\n    return 1  # {QUOTE}a{UNQUOTE} and {ELLIPSIS}\n")
    assert fixed == 'def f():\n    return 1  # "a" and ...\n'


def test_the_expression_halves_of_an_f_string_are_found_without_help_from_the_tokenizer():
    """Up to Python 3.11 the tokenizer hands a whole f-string back as one string token, so what the
    field runs has to be told apart from what the string prints here. From 3.12 the tokenizer does
    it, and both readings have to agree."""
    token = f"f\"keep {DASH} here {{d['a{DASH}b'] {DASH} 1}} {{w:>{{n}}}}\""
    runs = [token[begin:end] for begin, end in ce._fstring_expression_spans(token)]
    assert runs == ["d[", f"] {DASH} 1", "w", "n"]
    assert ce._fstring_expression_spans(f'"plain {DASH} string"') == []


def test_a_whole_f_string_arriving_as_one_token_protects_only_what_it_prints():
    """The reading Python 3.11 produces, spelled out as tokens so it is checked on every version."""
    line = f'x = f"keep {DASH} here {{value {DASH} 1}}"\n'
    tokens = [tokenize.TokenInfo(tokenize.STRING, line[4:-1], (1, 4), (1, len(line) - 1), line)]
    printed = ce._printed_positions(tokens, [line])
    assert (1, line.index(DASH)) in printed
    assert (1, line.index(DASH, line.index("value"))) not in printed


def test_the_confinement_prompt_asks_for_ascii_only():
    assert "ASCII only" in ce._confinement_block()


def test_a_sanitized_body_is_recorded_on_the_node_and_shown_to_the_next_attempt(
    make_test_model, schema, sigs, db0, workdir, order_calls
):
    """The change is free and deterministic, so it never costs an attempt, but the attempt is told it
    happened, so the model stops writing the character."""
    poisoned = WRONG_BODY.replace("return {", f"# a note {DASH} in a comment\nreturn {{")
    model = make_test_model([poisoned, CORRECT_BODY])
    build = ce.compile_tool(model, sigs[0], order_calls, schema, db0, workdir)
    assert build.nodes[0]["sanitized"].startswith("replaced typographic characters outside strings")
    assert "sanitized" not in build.nodes[1]
    assert build.nodes[0]["passed"] is False  # sanitizing repairs the character, not the body
    assert DASH in model.calls[1]["messages"][-1]["content"]
    assert build.body.strip() == CORRECT_BODY.strip()


# --- mined names are text, not identifiers (a review walked past the gate through one) ---

def test_a_column_name_carrying_python_source_is_refused_before_it_is_written():
    """Mining reads column names off JSON keys in the customer's traces. A key with a newline and a
    statement in it used to render as a live class-body statement that `load_toolkit` executed with
    every gate passing, because the confinement gate only reads the tool methods."""
    schema = EntitySchema(tables=["berths"], columns=[
        Column(table="berths", name="status", **{"class": "hard"}),
        Column(table="berths", name="ok = 1\n    import os", **{"class": "hard"})])
    assert ce.unsafe_names(schema) == [
        "column 'ok = 1\\n    import os' of berths is not a Python name"]
    with pytest.raises(ValueError) as raised:
        ce.render_data_model(schema)
    assert "cannot be written into the generated module" in str(raised.value)


def test_a_name_that_is_a_keyword_or_carries_a_space_is_refused_by_name():
    """The benign twin of the same defect: these used to fail the parse gate in the code-owned
    skeleton, where no body could repair it and the message named no cause."""
    schema = EntitySchema(tables=["berth slots"], columns=[
        Column(table="berth slots", name="class", **{"class": "hard"})])
    assert ce.unsafe_names(schema) == ["table 'berth slots' is not a Python name",
                                       "column 'class' of berth slots is not a Python name"]


def test_a_tool_or_argument_name_that_is_not_a_python_name_is_refused_too():
    schema = EntitySchema(tables=["berths"],
                          columns=[Column(table="berths", name="status", **{"class": "hard"})])
    sig = ToolSig(name="find berth", kind="read",
                  args_fields=[FieldStat(name="berth id", types=["str"])], result_schema=[])
    assert ce.unsafe_names(schema, [sig]) == ["tool 'find berth' is not a Python name",
                                              "argument 'berth id' of find berth is not a Python name"]
    with pytest.raises(ValueError):
        ce.render_tools(schema, [sig], {})


def test_the_names_an_ordinary_mined_schema_carries_are_written_without_complaint():
    schema = EntitySchema(tables=["berths", "vessels"], columns=[
        Column(table="berths", name="status", **{"class": "hard"}),
        Column(table="vessels", name="hull", **{"class": "semantic"})])
    sig = ToolSig(name="find_berth", kind="read",
                  args_fields=[FieldStat(name="berth_id", types=["str"])], result_schema=[])
    assert ce.unsafe_names(schema, [sig]) == []
    assert "class Berth(BaseModel):" in ce.render_data_model(schema)


def test_an_argument_named_like_the_receiver_or_mined_twice_is_refused():
    """Greptile on PR 45: `self` is a Python name, and the skeleton has already written it. A tool
    whose mined argument is called `self` used to render `def find_berth(self, self)`, a SyntaxError
    in the code-owned half of the module that no body could repair."""
    schema = EntitySchema(tables=["berths"],
                          columns=[Column(table="berths", name="status", **{"class": "hard"})])
    receiver = ToolSig(name="find_berth", kind="read",
                       args_fields=[FieldStat(name="self", types=["str"])], result_schema=[])
    assert ce.unsafe_names(schema, [receiver]) == [
        "argument 'self' of find_berth is a name the signature already carries"]
    with pytest.raises(ValueError):
        ce.render_tools(schema, [receiver], {})
    twice = ToolSig(name="find_berth", kind="read",
                    args_fields=[FieldStat(name="berth_id", types=["str"]),
                                 FieldStat(name="berth_id", types=["int"], optional=True)],
                    result_schema=[])
    assert ce.unsafe_names(schema, [twice]) == [
        "argument 'berth_id' of find_berth is a name the signature already carries"]
