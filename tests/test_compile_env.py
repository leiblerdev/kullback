"""Tests for compile_env: inverse replay, per-Task overlays, the tau2 shape, the five gates and the repair loop."""

from __future__ import annotations

import inspect
import json

import pytest

from conftest import PTR
from harness.builder import compile_env as ce
from harness.shared.records import (
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
    assert [(r.table, r.id) for r in cancel_overlay.rows] == [("orders", oid)]
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
    assert [g.stage for g in gates][:5] == [
        "parses", "confined", "executes_on_s0", "deterministic", "non_trivial"]
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


def test_split_calls_holds_some_calls_back(order_calls):
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


def test_emit_refuses_two_tasks_that_pin_the_same_row_differently(sample, schema, sigs, db0, workdir):
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
    with pytest.raises(ce.OverlayConflict) as caught:
        ce.emit_tau2_shape(bundle, workdir / "out")
    assert oid in str(caught.value)


# --- the Environment record, its sub-versions and its open flags (D67, D70, D97) ---

def test_build_environment_carries_the_sub_versions_and_the_flags():
    from harness.builder.compile_env import build_environment
    from harness.shared.records import Column, EntitySchema, ErrorShape, ToolSig

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
    from harness.builder.compile_env import build_environment
    from harness.shared.records import EntitySchema

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
    assert all(g.metrics["success_fidelity"] == 1.0 for g in fidelity)


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
    third = orders[2]["order_id"]
    body = (f"import random\nif order_id == {third!r}:\n"
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


def _held_out_body(shown_ids):
    return ("if order_id in %r:\n    return self.db.orders[order_id]\n"
            "raise ValueError('Order not found')\n" % (shown_ids,))


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
    body = _held_out_body([c.args["order_id"] for c in shown if c.error is None])
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
    body = _held_out_body([c.args["order_id"] for c in shown if c.error is None])
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
    from harness.builder.sandbox import id_pattern_for

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
    from harness.shared.budget import ContextCapExceeded

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
    from harness.builder.sandbox import ALLOWED_IMPORTS, DENIED_BUILTINS

    system = ce.body_messages(sigs[0], [], schema=schema)[0]["content"]
    for name in DENIED_BUILTINS:
        assert name in system, f"the gate denies {name} and the prompt never says so"
    for name in ALLOWED_IMPORTS:
        assert name in system
    assert "dunder" in system


def test_the_rules_in_the_prompt_are_generated_from_the_gate_not_copied(monkeypatch):
    """A hand-written copy drifts the first time the gate changes; a generated one cannot."""
    monkeypatch.setattr(ce, "DENIED_BUILTINS", frozenset({"summon_a_demon"}))
    assert "summon_a_demon" in ce._confinement_block()


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
    from harness.builder.mine import SCALAR_RESULT_FIELD

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


def test_a_repair_prompt_says_the_evidence_already_agrees_when_gate_four_fails_on_it(
    make_test_model, workdir
):
    """transfer_to_human_agents's own recorded calls all answer "Transfer successful". A body that
    also answers that way should not be told to invent detail."""
    schema = EntitySchema(tables=[], columns=[])
    sig = ToolSig(name="transfer_to_human_agents", kind="generic", unclassified=False,
                  args_fields=[FieldStat(name="summary", types=["str"], optional=False)])
    calls = [_call("transfer_to_human_agents", {"summary": f"issue {i}"},
                   result="Transfer successful", idx=i) for i in range(4)]
    model = make_test_model(['return "Transfer successful"'] * 4)
    ce.compile_tool(model, sig, calls, schema, {}, workdir)
    assert len(model.calls) >= 2, "gate 4 fails a constant body, so a retry happens"
    assert "already answer every one of them the same way" in model.calls[1]["messages"][-1]["content"]


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
    from harness.runner.validate import gate
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
    from harness.builder import mine
    schema = mine.mine_schema(_retail_nesting_traces())
    assert schema.homes == {"items": "products.variants"}
    assert "items" in schema.tables and "products" in schema.tables
    names = {c.name for c in schema.columns if c.table == "items"}
    assert {"item_id", "price", "available", "options"} <= names


def test_a_nested_collection_never_shown_on_its_own_stays_a_column_of_its_parent():
    from harness.builder import mine
    traces = _retail_nesting_traces()
    traces[0].tool_calls = traces[0].tool_calls[:1]
    schema = mine.mine_schema(traces)
    assert schema.homes == {}
    assert "items" not in schema.tables


def test_the_schema_block_says_where_a_table_with_a_home_is_stored():
    from harness.builder import mine
    block = ce._schema_block(mine.mine_schema(_retail_nesting_traces()))
    assert "items rows are stored inside products.variants, keyed by item_id" in block
    assert "self.db.products.values()" in block and ".variants" in block
    assert "may be empty on the customer's real database" in block


def test_the_starting_state_folds_a_standalone_row_into_its_home(workdir):
    from harness.builder import mine
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
    from harness.builder import mine
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
    from harness.builder import mine
    traces = _twelve_user_traces()
    state = ce.build_starting_state(traces, mine.mine_schema(traces), tmp_path)
    assert len(state.db["users"]) == 12 and state.synthetic_rows == []
    assert not (tmp_path / "synthetic.json").exists()
