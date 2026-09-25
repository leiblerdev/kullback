"""Tests for compile_env: inverse replay, per-Task overlays, the tau2 shape and the five gates."""

from __future__ import annotations

import ast
import json
import tokenize

import pytest

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sandbox_mod
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
)

# --- hand-written tool bodies, standing in for what a model would write ---

CORRECT_BODY = """
if loan_id not in self.db.loans:
    raise ValueError("Loan not found")
return self.db.loans[loan_id]
"""

WRONG_BODY = """
return {"loan_id": loan_id, "status": "delivered"}
"""

CONSTANT_BODY = """
return {"loan_id": "loan_9999", "status": "delivered"}
"""

BROKEN_BODY = """
return self.db.loans[loan_id
"""

CRASHING_BODY = """
return self.db.sheds[loan_id]
"""

SLOW_BODY = """
while True:
    pass
"""

CANCEL_BODY = """
if loan_id not in self.db.loans:
    raise ValueError("Loan not found")
loan = self.db.loans[loan_id]
if loan.status != "pending":
    raise ValueError("Non-pending loan cannot be cancelled")
loan.status = "cancelled"
loan.cancel_reason = reason
return loan
"""

FORGING_BODY = """
import json
import os
import sys

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"results": [{"ok": True, "value": {"loan_id": loan_id, "status": "delivered"}}]}, handle)
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
        id_patterns={"loans": r"^loan_\d+$", "members": r"^member_\d+$", "copies": r"^copy_\d+$"},
    )


# --- fixtures built from an invented library db, in the shape of world_a ---


@pytest.fixture(scope="session")
def sample():
    """Three loans with different statuses, plus one member and one copy, kept small on purpose."""
    loans = {}
    for loan_id, status in (("loan_01", "pending"), ("loan_02", "delivered"), ("loan_03", "processed")):
        loans[loan_id] = {
            "loan_id": loan_id,
            "member_id": "member_01",
            "copies": [{"copy_id": "copy_01", "title": "Riverside Walks"}],
            "status": status,
            "address": "elm street",
            "amount": 10,
        }
    members = {"member_01": {"member_id": "member_01", "name": "Ava Chen", "zip": "11201"}}
    copies = {"copy_01": {"copy_id": "copy_01", "title": "Riverside Walks"}}
    return {
        "loans": loans,
        "members": members,
        "copies": copies,
        "by_status": {row["status"]: row for row in loans.values()},
    }


@pytest.fixture(scope="session")
def db0(sample):
    return {"loans": sample["loans"], "members": sample["members"], "copies": sample["copies"]}


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
            "loans": r"^loan_\d+$",
            "members": r"^member_\d+$",
            "copies": r"^copy_\d+$",
        },
    )


@pytest.fixture(scope="session")
def sigs():
    return [
        ToolSig(
            name="get_loan_details",
            description="Get the status and details of a loan.",
            args_fields=[FieldStat(name="loan_id", types=["str"], optional=False)],
            kind="read",
            unclassified=False,
        ),
        ToolSig(name="update_loan", kind="write", unclassified=False),
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
def loan_calls(sample):
    """Recorded get_loan_details calls: three successes and one not-found error."""
    calls = []
    for idx, loan in enumerate(sample["loans"].values()):
        calls.append(_call("get_loan_details", {"loan_id": loan["loan_id"]}, result=loan, idx=idx))
    calls.append(
        _call(
            "get_loan_details",
            {"loan_id": "loan_9999"},
            error=ToolCallError(**{"class": "not_found_entity"}, payload="Loan not found"),
            idx=9,
        )
    )
    return calls


# --- inverse replay over the whole corpus (D33, D74) ---


def test_inverse_replay_undoes_an_observed_write(sample, schema, sigs, workdir):
    pending = sample["by_status"]["pending"]
    oid = pending["loan_id"]
    post = dict(pending, status="cancelled", cancel_reason="no longer needed")
    trace = _trace(
        "A",
        [
            _call("get_loan_details", {"loan_id": oid}, result=pending, idx=0),
            _call("update_loan", {"loan_id": oid, "reason": "no longer needed"}, result=post, idx=1),
        ],
    )
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["loans"][oid]["status"] == "pending"
    assert json.loads((workdir / ce.DB_FILE).read_text(encoding="utf-8")) == state.db


def test_post_state_only_is_kept_and_the_assumption_is_recorded(sample, schema, sigs, workdir):
    pending = sample["by_status"]["pending"]
    oid = pending["loan_id"]
    post = dict(pending, status="cancelled")
    trace = _trace("B", [_call("update_loan", {"loan_id": oid}, result=post, idx=0)])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["loans"][oid]["status"] == "cancelled"
    assert any(oid in note for note in state.assumptions)


def test_latest_observation_wins_in_the_shared_rows(sample, schema, sigs, workdir):
    loan = sample["by_status"]["delivered"]
    oid = loan["loan_id"]
    first = _trace("A", [_call("get_loan_details", {"loan_id": oid}, result=dict(loan, status="processed"))])
    second = _trace("B", [_call("get_loan_details", {"loan_id": oid}, result=dict(loan, status="delivered"))])
    state = ce.build_starting_state([first, second], schema, workdir, tool_sigs=sigs)
    assert state.db["loans"][oid]["status"] == "delivered"


def test_a_json_string_result_is_read_as_a_row(sample, schema, sigs, workdir):
    loan = sample["by_status"]["processed"]
    trace = _trace("A", [_call("get_loan_details", {"loan_id": loan["loan_id"]}, result=json.dumps(loan))])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert state.db["loans"][loan["loan_id"]]["status"] == "processed"


def test_the_shared_db_is_the_same_bytes_on_two_runs(sample, schema, sigs, tmp_path):
    loan = sample["by_status"]["pending"]
    traces = [_trace("A", [_call("get_loan_details", {"loan_id": loan["loan_id"]}, result=loan)])]
    outputs = []
    for name in ("one", "two"):
        work = tmp_path / name
        work.mkdir()
        ce.build_starting_state(traces, schema, work, tool_sigs=sigs)
        outputs.append((work / ce.DB_FILE).read_bytes())
    assert outputs[0] == outputs[1]


# --- per-Task overlays (D74) ---


def test_each_task_overlay_pins_the_version_its_runs_saw(sample, schema, sigs, workdir):
    loan = sample["by_status"]["pending"]
    oid = loan["loan_id"]
    cancelled = dict(loan, status="cancelled")
    march = _trace("A", [_call("get_loan_details", {"loan_id": oid}, result=loan)])
    june = _trace("C", [_call("get_loan_details", {"loan_id": oid}, result=cancelled)])
    tasks = [Task(id="t_cancel", run_ids=["A"]), Task(id="t_after", run_ids=["C"])]
    state = ce.build_starting_state([march, june], schema, workdir, tasks=tasks, tool_sigs=sigs)

    assert {o.task_id for o in state.overlays} == {"t_cancel", "t_after"}
    cancel_overlay, cancel_values = ce.load_overlay(workdir, "t_cancel")
    after_overlay, after_values = ce.load_overlay(workdir, "t_after")
    # The loan's own row, and the rows the loan mentions inside itself (D188 walks a result to
    # any depth); rows are sorted by table, so the copies row comes before the loan row.
    cancel_by_table = {r.table: r for r in cancel_overlay.rows}
    after_by_table = {r.table: r for r in after_overlay.rows}
    assert set(cancel_by_table) == {"loans", "copies"}
    assert cancel_by_table["loans"].id == oid
    assert cancel_by_table["copies"].id == "copy_01"
    assert cancel_values[cancel_by_table["loans"].version_hash]["status"] == "pending"
    assert after_values[after_by_table["loans"].version_hash]["status"] == "cancelled"
    assert cancel_by_table["loans"].version_hash != after_by_table["loans"].version_hash
    assert cancel_by_table["loans"].trace_id == "A"


def test_an_overlay_row_carries_the_pre_write_version(sample, schema, sigs, workdir):
    loan = sample["by_status"]["pending"]
    oid = loan["loan_id"]
    post = dict(loan, status="cancelled")
    trace = _trace(
        "A",
        [
            _call("get_loan_details", {"loan_id": oid}, result=loan, idx=0),
            _call("update_loan", {"loan_id": oid}, result=post, idx=1),
        ],
    )
    tasks = [Task(id="t1", run_ids=["A"])]
    ce.build_starting_state([trace], schema, workdir, tasks=tasks, tool_sigs=sigs)
    overlay, values = ce.load_overlay(workdir, "t1")
    loan_row = next(r for r in overlay.rows if r.table == "loans")
    assert values[loan_row.version_hash]["status"] == "pending"


# --- the model writes the body, code writes the signature, docstring and schema ---


def test_render_tools_writes_the_signature_and_the_docstring(schema, sigs):
    source = ce.render_tools(schema, sigs[:1], {"get_loan_details": CORRECT_BODY})
    assert "def get_loan_details(self, loan_id: str)" in source
    assert "Get the status and details of a loan." in source
    assert "        if loan_id not in self.db.loans:" in source
    compile(source, "<rendered>", "exec")


def test_render_data_model_has_one_class_per_table_and_a_db(schema):
    source = ce.render_data_model(schema)
    for name in ("class Loan(", "class Member(", "class Copie(", f"class {ce.DB_CLASS}("):
        assert name in source
    compile(source, "<rendered>", "exec")


# --- the five gates ---


def _sandbox(schema, sigs, db0, workdir, body, timeout=30):
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": body})
    return source, ce.Sandbox(source, db0, workdir, timeout=timeout)


def test_gate_parses_fails_on_a_body_that_is_not_python(schema, sigs, db0, workdir):
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": BROKEN_BODY})
    result = ce.gate_parses(source)
    assert result.passed is False
    assert result.stage == "parses"
    assert result.failures


def test_a_correct_body_passes_all_five_gates(schema, sigs, db0, workdir, loan_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CORRECT_BODY)
    shown, held_out = ce.split_calls(loan_calls)
    gates = ce.run_gates(source, box, shown, held_out, schema)
    # "confined" is the gate that stands in for the deferred sandbox: the same module is exec'd
    # in the Runner's process by load_toolkit, so a body that reaches past the customer's world
    # is refused before it ever executes anywhere.
    assert [g.stage for g in gates][:6] == [
        "parses", "confined", "compile_tools.memorised_values", "executes_on_s0", "deterministic",
        "non_trivial"]
    assert all(g.passed for g in gates), [g.failures for g in gates if not g.passed]


def test_a_crashing_body_fails_the_executes_gate(schema, sigs, db0, workdir, loan_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CRASHING_BODY)
    result = ce.gate_executes_on_s0(box, loan_calls[:1])
    assert result.passed is False
    assert result.stage == "executes_on_s0"


def test_a_constant_body_fails_the_non_trivial_gate(schema, sigs, db0, workdir, loan_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CONSTANT_BODY)
    result = ce.gate_non_trivial(box, loan_calls[:3])
    assert result.passed is False
    assert result.stage == "non_trivial"


def test_replay_fidelity_reports_success_and_error_separately(schema, sigs, db0, workdir, loan_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, CORRECT_BODY)
    result = ce.gate_replay_fidelity(box, loan_calls, schema)
    assert result.passed is True
    assert result.metrics["success_calls"] == 3
    assert result.metrics["error_calls"] == 1
    assert result.metrics["success_fidelity"] == 1.0
    assert result.metrics["error_fidelity"] == 1.0


def test_replay_fidelity_fails_a_wrong_body_on_hard_columns_and_on_errors(
    schema, sigs, db0, workdir, loan_calls
):
    source, box = _sandbox(schema, sigs, db0, workdir, WRONG_BODY)
    result = ce.gate_replay_fidelity(box, loan_calls, schema)
    assert result.passed is False
    assert result.metrics["success_fidelity"] < 1.0
    assert result.metrics["error_fidelity"] == 0.0
    assert result.failures


def test_the_sandbox_kills_a_body_that_does_not_stop(schema, sigs, db0, workdir, loan_calls):
    source, box = _sandbox(schema, sigs, db0, workdir, SLOW_BODY, timeout=2)
    result = ce.gate_executes_on_s0(box, loan_calls[:1])
    assert result.passed is False
    assert any("timeout" in failure for failure in result.failures)


def test_split_calls_puts_every_call_in_exactly_one_of_the_shown_and_held_out_splits_and_leaves_neither_empty(loan_calls):
    shown, held_out = ce.split_calls(loan_calls)
    assert shown and held_out
    assert len(shown) + len(held_out) == len(loan_calls)
    assert not set(id(c) for c in shown) & set(id(c) for c in held_out)


# --- hold-out by argument shape (D250) ---


# Copies only the keys the unadorned wares carry, so a held-out ware with a mark misses that key.
COPY_WARE_BODY = """
return ware
"""


def _ware_calls():
    """Two argument shapes: unadorned wares, and wares that also carry a mark."""
    plain = [
        {"kind": "cup", "glaze": "ash"},
        {"kind": "bowl", "glaze": "salt"},
        {"kind": "plate", "glaze": "slip"},
    ]
    marked = [
        {"kind": "vase", "glaze": "ash", "mark": "incised"},
        {"kind": "jar", "glaze": "salt", "mark": "stamped"},
    ]
    calls = [_call("describe_ware", {"ware": ware}, result=ware, idx=i) for i, ware in enumerate(plain)]
    calls += [_call("describe_ware", {"ware": ware}, result=ware, idx=10 + i)
              for i, ware in enumerate(marked)]
    return calls


def _leaf_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _leaf_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaf_strings(item)
    elif value is not None:
        yield str(value)


def test_split_calls_holds_out_a_whole_argument_shape():
    calls = _ware_calls()
    shown, held_out = ce.split_calls(calls)
    shown_shapes = {ce.arg_shape(c.args) for c in shown}
    held_shapes = {ce.arg_shape(c.args) for c in held_out}
    assert shown and held_out
    assert len(shown) + len(held_out) == len(calls)
    assert shown_shapes and held_shapes and shown_shapes.isdisjoint(held_shapes)
    assert ce.no_shape_holdout(calls) is False
    one_filter = ce.arg_shape({"q": "cups", "filters": ["kind"]})
    same_shape = ce.arg_shape({"q": "bowls", "filters": ["glaze"]})
    two_filters = ce.arg_shape({"q": "vases", "filters": ["kind", "glaze"]})
    assert one_filter == same_shape != two_filters
    plain = ce.arg_shape({"ware": {"glaze": "ash", "kind": "cup"}})
    other_plain = ce.arg_shape({"ware": {"glaze": "salt", "kind": "bowl"}})
    marked = ce.arg_shape({"ware": {"glaze": "ash", "kind": "vase", "mark": "incised"}})
    assert plain == other_plain != marked


def test_split_calls_falls_back_to_every_third_when_every_call_shares_one_shape(loan_calls):
    shown, held_out = ce.split_calls(loan_calls)
    by_index_shown = [c for i, c in enumerate(loan_calls) if i % 3 != 2]
    by_index_held = [c for i, c in enumerate(loan_calls) if i % 3 == 2]
    assert [c.id for c in shown] == [c.id for c in by_index_shown]
    assert [c.id for c in held_out] == [c.id for c in by_index_held]
    assert ce.no_shape_holdout(loan_calls) is True


def test_the_shape_of_an_id_generalizes_its_characters_and_keeps_the_rest():
    assert ce.value_shape("m-14") == "a-##"
    assert ce.value_shape("AB_9") == "AA_#"
    assert ce.table_read("row = self.db.members[member_id]") == "members"
    assert ce.table_read("return {}") == ""


# --- the tau2 file shape (D56) ---


def _bundle(schema, sigs, db0, overlays=(), tasks=(), verifiers=()):
    return ce.EnvBundle(
        environment=Environment(env_id="env1", assisted_tools=["update_loan"]),
        schema=schema,
        tools=sigs[:1],
        bodies={"get_loan_details": CORRECT_BODY},
        db=db0,
        overlays=list(overlays),
        policy_text="# Library policy\n\nAuthenticate the member first.\n",
        tasks=list(tasks),
        verifiers=list(verifiers),
        assumptions=["loan loan_9999 was only seen after a write"],
    )


def test_emit_tau2_shape_writes_the_five_files_and_a_sidecar(schema, sigs, db0, workdir):
    task = Task(id="t1", intent="cancel a pending loan", run_ids=["A"])
    verifier = Verifier(
        task_id="t1",
        atoms=[
            Atom(
                id="t1_0",
                kind="required",
                description="cancel the loan",
                predicate_src=json.dumps({"name": "update_loan", "arguments": {"loan_id": "#W1"}}),
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
    assert tasks_json[0]["evaluation_criteria"]["actions"][0]["name"] == "update_loan"

    sidecar = json.loads(paths["sidecar.json"].read_text(encoding="utf-8"))
    assert sidecar["assisted_tools"] == ["update_loan"]
    assert sidecar["assumptions"]
    assert sidecar["atoms"]["t1"][1]["kind"] == "allowed"
    assert set(sidecar["files"]) == {"data_model.py", "tools.py", "db.json", "policy.md", "tasks.json"}


def test_emit_merges_overlays_into_one_db(sample, schema, sigs, db0, workdir):
    loan = sample["by_status"]["pending"]
    oid = loan["loan_id"]
    cancelled = dict(loan, status="cancelled")
    trace = _trace("A", [_call("get_loan_details", {"loan_id": oid}, result=cancelled)])
    state = ce.build_starting_state(
        [trace], schema, workdir, tasks=[Task(id="t1", run_ids=["A"])], tool_sigs=sigs
    )
    bundle = _bundle(schema, sigs, db0, overlays=state.overlays)
    bundle.overlay_values = ce.overlay_values(workdir)
    paths = ce.emit_tau2_shape(bundle, workdir / "out")
    merged = json.loads(paths["db.json"].read_text(encoding="utf-8"))
    assert merged["loans"][oid]["status"] == "cancelled"


def test_emit_records_two_tasks_that_pin_the_same_row_differently(sample, schema, sigs, db0, workdir):
    loan = sample["by_status"]["pending"]
    oid = loan["loan_id"]
    cancelled = dict(loan, status="cancelled")
    traces = [
        _trace("A", [_call("get_loan_details", {"loan_id": oid}, result=loan)]),
        _trace("C", [_call("get_loan_details", {"loan_id": oid}, result=cancelled)]),
    ]
    tasks = [Task(id="t1", run_ids=["A"]), Task(id="t2", run_ids=["C"])]
    state = ce.build_starting_state(traces, schema, workdir, tasks=tasks, tool_sigs=sigs)
    bundle = _bundle(schema, sigs, db0, overlays=state.overlays)
    bundle.overlay_values = ce.overlay_values(workdir)
    ce.emit_tau2_shape(bundle, workdir / "out")
    # D74: the export records the conflict and keeps one version; the Runner keeps both overlays.
    assert len(bundle.conflicts) == 1 and oid in bundle.conflicts[0]
    exported = json.loads((workdir / "out" / "db.json").read_text())
    assert exported["loans"][oid]["status"] == loan["status"]
    with pytest.raises(ce.OverlayConflict):
        ce.merge_overlays(db0, state.overlays, bundle.overlay_values)


def test_the_export_prefers_the_version_seen_before_a_write(tmp_path):
    from kullback.runner.records import OverlayRow, TaskOverlay
    late = TaskOverlay(task_id="t1", rows=[OverlayRow(table="loans", id="o1", version_hash="v_after",
                                                       after_write=True)])
    early = TaskOverlay(task_id="t2", rows=[OverlayRow(table="loans", id="o1", version_hash="v_before")])
    values = {"v_after": {"status": "cancelled"}, "v_before": {"status": "pending"}}
    conflicts: list[str] = []
    merged = ce.merge_overlays({"loans": {}}, [late, early], values, conflicts)
    assert merged["loans"]["o1"]["status"] == "pending"
    assert conflicts == ["tasks t1 and t2 pin loans row o1 in different versions; the benchmark export keeps t2's"]


# --- the Environment record, its sub-versions and its open flags (D67, D70, D97) ---

def test_build_environment_carries_the_sub_versions_the_flags_and_the_assisted_tools(schema, sigs):
    env = ce.build_environment(schema, sigs, {s.name: "pass" for s in sigs}, "policy",
                               assisted_tools=["update_loan"])
    assert env.assisted_tools == ["update_loan"]
    from kullback.builder.compile_env import build_environment
    from kullback.runner.records import Column, EntitySchema, ErrorShape, ToolSig

    schema = EntitySchema(tables=["loans"], columns=[Column(table="loans", name="status", class_="hard")])
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


# --- every recorded call replays on its own Starting state (D74, design section 6 gate 2) ---


@pytest.fixture
def cancel_sig():
    return ToolSig(
        name="update_loan",
        description="Cancel a pending loan.",
        args_fields=[
            FieldStat(name="loan_id", types=["str"], optional=False),
            FieldStat(name="reason", types=["str"], optional=False),
        ],
        kind="write",
        unclassified=False,
    )


@pytest.fixture
def cancel_world(sample):
    """One pending loan with room for a cancel reason, and the schema over it."""
    loan = dict(sample["by_status"]["pending"], cancel_reason=None)
    db = {"loans": {loan["loan_id"]: loan}, "members": {}, "copies": {}}
    return db, _schema_for(db), loan


def test_a_write_tool_passes_all_five_gates_on_calls_two_runs_recorded(cancel_world, cancel_sig, workdir):
    """Two Runs of one Task each cancel the same pending loan; both start from S0, so both replay."""
    db, schema, loan = cancel_world
    oid = loan["loan_id"]
    first = dict(loan, status="cancelled", cancel_reason="no longer needed")
    second = dict(loan, status="cancelled", cancel_reason="ordered by mistake")
    calls = [
        _call(cancel_sig.name, {"loan_id": oid, "reason": "no longer needed"}, result=first, idx=0),
        _call(cancel_sig.name, {"loan_id": oid, "reason": "ordered by mistake"}, result=second, idx=1),
        _call(
            cancel_sig.name,
            {"loan_id": "loan_9999", "reason": "no longer needed"},
            error=ToolCallError(**{"class": "not_found_entity"}, payload="Loan not found"),
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
    db, schema, loan = cancel_world
    args = {"loan_id": loan["loan_id"], "reason": "no longer needed"}
    calls = [_call(cancel_sig.name, args, idx=0), _call(cancel_sig.name, args, idx=1)]
    source = ce.module_source(schema, [cancel_sig], {cancel_sig.name: CANCEL_BODY})
    results = ce.Sandbox(source, db, workdir).run(calls)
    assert [r["ok"] for r in results] == [True, True], results
    assert [r["value"]["status"] for r in results] == ["cancelled", "cancelled"]


def test_a_row_the_corpus_shows_in_two_versions_replays_on_each_tasks_own_world(
    sample, sigs, workdir
):
    """The March Task saw the loan pending, the June Task saw it cancelled; one body serves both."""
    pending = sample["by_status"]["pending"]
    oid = pending["loan_id"]
    other = sample["by_status"]["delivered"]
    db = {"loans": {oid: pending, other["loan_id"]: other}, "members": {}, "copies": {}}
    schema = _schema_for(db)
    march = _trace("A", [
        _call("get_loan_details", {"loan_id": oid}, result=pending, idx=0),
        _call("get_loan_details", {"loan_id": other["loan_id"]}, result=other, idx=2),
    ])
    june = _trace("C", [_call("get_loan_details", {"loan_id": oid},
                              result=dict(pending, status="cancelled"), idx=1)])
    tasks = [Task(id="tA", run_ids=["A"]), Task(id="tC", run_ids=["C"])]
    state = ce.build_starting_state([march, june], schema, workdir, tasks=tasks, tool_sigs=sigs)
    assert state.db["loans"][oid]["status"] == "cancelled"  # latest observation wins in the shared world

    calls = [
        march.tool_calls[0],
        june.tool_calls[0],
        march.tool_calls[1],
        _call("get_loan_details", {"loan_id": "loan_9999"},
              error=ToolCallError(**{"class": "not_found_entity"}, payload="Loan not found"), idx=3),
    ]
    states = ce.call_starting_states(
        state.db, state.overlays, ce.overlay_values(workdir), {"c0": "tA", "c1": "tC", "c2": "tA"}
    )
    rows = ce.replay_outcomes(sigs[0], CORRECT_BODY, calls, schema, state.db, workdir / "tool",
                              call_states=states)
    assert [row["replayed"] for row in rows] == [True] * len(calls), rows


# --- a recorded invalid_arguments call is an answer, not a crash (D67) ---


def test_a_recorded_invalid_arguments_call_does_not_fail_the_executes_gate(sample, sigs, workdir):
    loan = sample["by_status"]["pending"]
    db = {"loans": {loan["loan_id"]: loan}, "members": {}, "copies": {}}
    schema = _schema_for(db)
    calls = [
        _call("get_loan_details", {"loan_id": loan["loan_id"]}, result=loan, idx=0),
        _call("get_loan_details", {"loan": loan["loan_id"]},
              error=ToolCallError(**{"class": "invalid_arguments"},
                                  payload="unexpected keyword argument 'loan'"), idx=1),
    ]
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": CORRECT_BODY})
    box = ce.Sandbox(source, db, workdir)
    executes = ce.gate_executes_on_s0(box, calls)
    assert executes.passed is True, executes.failures
    replay = ce.gate_replay_fidelity(box, calls, schema)
    assert replay.passed is True, replay.failures
    assert replay.metrics["error_matches"] == 1


# --- a mined column never rejects a real row (D72) ---


def test_a_float_row_and_an_int_row_both_load_under_mined_string_and_int_samples(sigs, workdir):
    db = {"loans": {"loan_01": {"loan_id": "loan_01", "amount": 10},
               "loan_02": {"loan_id": "loan_02", "amount": 10.5}},
          "members": {"member_01": {"member_id": "member_01", "zip": "11201"},
                      "member_02": {"member_id": "member_02", "zip": 11202}}}
    schema = EntitySchema(
        tables=["loans", "members"],
        columns=[
            Column(table="loans", name="loan_id", class_="hard", samples=["loan_01"]),
            Column(table="loans", name="amount", class_="hard", samples=[10, 20]),
            Column(table="members", name="member_id", class_="hard", samples=["member_01"]),
            Column(table="members", name="zip", class_="hard", samples=["11201", "11202"]),
        ],
        id_patterns={"loans": r"^loan_\d+$"},
    )
    assert "Optional[Any]" in ce.render_data_model(schema)
    calls = [_call("get_loan_details", {"loan_id": "loan_02"}, result=db["loans"]["loan_02"], idx=0)]
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": CORRECT_BODY})
    gate = ce.gate_executes_on_s0(ce.Sandbox(source, db, workdir), calls)
    assert gate.passed is True, gate.failures


# --- the sandbox refuses a result file it did not write ---


def test_a_body_that_writes_the_result_file_itself_is_refused(sample, sigs, workdir):
    loan = sample["by_status"]["pending"]
    db = {"loans": {loan["loan_id"]: loan}, "members": {}, "copies": {}}
    schema = _schema_for(db)
    calls = [_call("get_loan_details", {"loan_id": f"loan_9900{i}"}, idx=i) for i in range(3)]
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": FORGING_BODY})
    box = ce.Sandbox(source, db, workdir)
    with pytest.raises(ce.SandboxError) as caught:
        box.run(calls)
    assert "nonce" in str(caught.value)
    gate = ce.gate_non_trivial(box, calls)
    assert gate.passed is False


# --- column classes decide inside a list result too (D73, D84) ---


def test_inside_a_list_result_a_hard_column_fails_a_replay_and_an_exempt_one_cannot(sample, workdir):
    loan = sample["by_status"]["pending"]
    now = dict(loan, last_seen="2026-06-01T00:00:00Z")
    then = dict(loan, last_seen="2026-03-01T00:00:00Z")
    db = {"loans": {loan["loan_id"]: now}, "members": {}, "copies": {}}
    schema = _schema_for(db, exempt=("last_seen",))
    sig = ToolSig(name="list_loans", kind="read", unclassified=False,
                  args_fields=[FieldStat(name="member_id", types=["str"], optional=False)])
    calls = [_call(sig.name, {"member_id": loan["member_id"]}, result=[then], idx=0)]
    body = "return [o for o in self.db.loans.values() if o.member_id == member_id]"
    source = ce.module_source(schema, [sig], {sig.name: body})
    gate = ce.gate_replay_fidelity(ce.Sandbox(source, db, workdir), calls, schema)
    assert gate.passed is True, gate.failures
    # the same list replay with a hard column that moved fails
    hard = workdir.parent / "hard"
    hard.mkdir()
    loan = sample["by_status"]["pending"]
    db = {"loans": {loan["loan_id"]: dict(loan, status="cancelled")}, "members": {}, "copies": {}}
    schema = _schema_for(db)
    sig = ToolSig(name="list_loans", kind="read", unclassified=False,
                  args_fields=[FieldStat(name="member_id", types=["str"], optional=False)])
    calls = [_call(sig.name, {"member_id": loan["member_id"]}, result=[loan], idx=0)]
    body = "return [o for o in self.db.loans.values() if o.member_id == member_id]"
    source = ce.module_source(schema, [sig], {sig.name: body})
    gate = ce.gate_replay_fidelity(ce.Sandbox(source, db, hard), calls, schema)
    assert gate.passed is False
    assert any("status" in failure for failure in gate.failures)


# --- two Runs of one Task that disagree are recorded, not pinned silently (D74) ---


# --- gate 3 checks every call, not the first two ---


def test_a_correct_body_passes_the_deterministic_gate_and_one_nondeterministic_on_a_later_call_fails(sample, sigs, workdir, schema, db0, loan_calls):
    correct = workdir.parent / "correct"
    correct.mkdir()
    source, box = _sandbox(schema, sigs, db0, correct, CORRECT_BODY)
    result = ce.gate_deterministic(box, loan_calls[:1])
    assert result.passed is True
    assert result.metrics["calls"] == 1
    loans = list(sample["loans"].values())
    db = {"loans": {o["loan_id"]: o for o in loans}, "members": {}, "copies": {}}
    schema = _schema_for(db)
    third = loans[2]["status"]
    # G26 took `random` out of the imports a body may use, so the staged dice is the object's
    # own identity: it answers alike inside one call and differs across the gate's two runs.
    body = (f"if self.db.loans[loan_id].status == {third!r}:\n"
            "    return {'loan_id': loan_id, 'status': str(id(loan_id))}\n"
            "return self.db.loans[loan_id]\n")
    calls = [_call("get_loan_details", {"loan_id": o["loan_id"]}, result=o, idx=i)
             for i, o in enumerate(loans)]
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": body})
    gates = ce.run_gates(source, ce.Sandbox(source, db, workdir), calls, [], schema)
    deterministic = next(g for g in gates if g.stage == "deterministic")
    assert deterministic.passed is False
    assert deterministic.metrics["calls"] == 3


# --- env_id moves when the world moves (design section 5) ---


def test_tau2_files_are_what_emit_writes_and_what_env_id_can_hash(schema, sigs, db0, workdir):
    bundle = _bundle(schema, sigs, db0)
    files = ce.tau2_files(bundle)
    assert set(files) == {"data_model.py", "tools.py", "db.json", "policy.md", "tasks.json"}
    paths = ce.emit_tau2_shape(bundle, workdir / "out")
    for name, text in files.items():
        assert paths[name].read_text(encoding="utf-8") == text
    schema = EntitySchema(tables=["loans"], columns=[Column(table="loans", name="status", class_="hard")])
    one = {"db.json": '{"loans": {"loan_01": {"status": "pending"}}}\n', "policy.md": "p"}
    two = {"db.json": '{"loans": {"loan_01": {"status": "cancelled"}}}\n', "policy.md": "p"}
    first = ce.build_environment(schema, [], {}, "policy", files=one)
    second = ce.build_environment(schema, [], {}, "policy", files=two)
    assert first.env_id != second.env_id, "two worlds with different rows shared one env_id"
    assert set(first.files) == {"db.json", "policy.md"}
    assert first.files["policy.md"] == second.files["policy.md"]
    assert first.schema_version == second.schema_version


# --- ids the traces referenced but never showed get a tagged synthetic row (D40, D41) ---


def test_an_id_the_traces_only_referenced_gets_a_synthetic_row_shaped_from_the_observed_rows(
    sample, sigs, workdir
):
    known = next(iter(sample["members"]))
    unseen = "member_09"
    schema = _schema_for({"members": sample["members"], "loans": sample["loans"], "copies": {}})
    trace = _trace("A", [
        _call("get_member_details", {"member_id": known}, result=sample["members"][known], idx=0),
        _call("list_loans", {"member_id": unseen}, result=[], idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert unseen in state.db["members"], "a read of a referenced id has nothing to answer it"
    assert set(state.db["members"][unseen]) == set(sample["members"][known])
    assert state.db["members"][unseen]["member_id"] == unseen
    assert state.synthetic_rows == [unseen]
    assert unseen in schema.synthetic_rows
    assert any(unseen in note and "synthetic" in note for note in state.assumptions)


def test_an_id_only_a_failed_call_named_is_not_invented(sample, schema, sigs, workdir):
    loan = sample["by_status"]["pending"]
    trace = _trace("A", [
        _call("get_loan_details", {"loan_id": loan["loan_id"]}, result=loan, idx=0),
        _call("get_loan_details", {"loan_id": "loan_9999"},
              error=ToolCallError(**{"class": "not_found_entity"}, payload="Loan not found"), idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=sigs)
    assert "loan_9999" not in state.db["loans"]
    assert state.synthetic_rows == []


# --- a description the customer wrote cannot break the code-owned skeleton ---


def test_a_description_holding_a_triple_quote_still_parses(schema, sigs):
    sig = sigs[0].model_copy(update={"description": 'Get """the""" loan, path C:\\loans\\'})
    source = ce.module_source(schema, [sig], {sig.name: CORRECT_BODY})
    assert ce.gate_parses(source).passed is True
    assert "the" in source


# --- the toolkit the Router calls holds the Task's own world (D74) ---


def test_load_toolkit_puts_the_tasks_overlay_inside_the_toolkit(sample, schema, sigs, workdir):
    loan = sample["by_status"]["pending"]
    oid = loan["loan_id"]
    traces = [
        _trace("A", [_call("get_loan_details", {"loan_id": oid}, result=loan, idx=0)]),
        _trace("B", [_call("get_loan_details", {"loan_id": oid},
                           result=dict(loan, status="cancelled"), idx=1)]),
    ]
    state = ce.build_starting_state(traces, schema, workdir, tasks=[Task(id="tA", run_ids=["A"])],
                                    tool_sigs=sigs)
    assert state.db["loans"][oid]["status"] == "cancelled"
    overlay, values = ce.load_overlay(workdir, "tA")
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": CORRECT_BODY})
    toolkit = ce.load_toolkit(source, state.db, overlay=overlay, overlay_values=values)
    assert toolkit.get_loan_details(oid).status == "pending"
    assert ce.load_toolkit(source, state.db).get_loan_details(oid).status == "cancelled"


# --- confinement: the static check that stands in for the deferred sandbox ---

def test_a_body_that_reaches_outside_the_world_is_refused_before_it_runs(schema, sigs, db0, workdir):
    """load_toolkit exec's the module in the Runner's own process, so the body is checked first."""
    body = "import os\nopen('/tmp/x', 'w').write('hi')\nreturn self.db.loans.get(loan_id)\n"
    source = ce.module_source(schema, sigs[:1], {sigs[0].name: body})
    assert ce.source_confinement(source) == [
        f"{sigs[0].name} imports os", f"{sigs[0].name} uses open"]
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(source, db0)
    with_eval = _quotes_source("return eval(self.db.quotes[quote_id].formula)\n")
    assert "settle_quote uses eval" in ce.source_confinement(with_eval)
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(with_eval, QUOTES_DB)
    with_ast = _quotes_source("import ast\nreturn ast.parse(self.db.quotes[quote_id].formula)\n")
    assert "settle_quote imports ast" in ce.source_confinement(with_ast)
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(with_ast, QUOTES_DB)


def test_a_body_that_walks_the_class_tree_is_refused(schema, sigs, db0, workdir):
    """().__class__.__base__.__subclasses__() is the escape a restricted namespace does not stop."""
    body = ("cls = ().__class__.__base__.__subclasses__()\n"
            "return {'n': len(cls)}\n")
    source = ce.module_source(schema, sigs[:1], {sigs[0].name: body})
    assert any("touches __class__" in line for line in ce.source_confinement(source))
    with pytest.raises(ce.SandboxError):
        ce.load_toolkit(source, db0)
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
    """A corpus whose simulated user calls its own phone tools does so inside the trace (R33); those
    results describe the user's device, not the customer's system, so they never enter the Starting state."""
    loan = next(iter(sample["loans"].values()))
    seen = ToolCall(id="c1", name="get_loan_details", args={"loan_id": loan["loan_id"]}, result=loan,
                    raw_ptr=PTR)
    user_side = ToolCall(id="c2", name="check_status_bar", args={},
                         result={**loan, "status": "on the phone"}, requestor="user", raw_ptr=PTR)
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
        tables=["loans"],
        columns=[Column(table="loans", name="loan_id", **{"class": "hard"}, classified_by="rule")],
        id_patterns={"loans.loan_id": r"^loan_\d+$"},
    )
    traces = [_trace("t1", [
        _call("get_loan_details", {"loan_id": "loan_42"}, result='{"loan_id": "loan_42"}', idx=0),
        _call("get_loan_details", {"loan_id": "not-a-loan"}, result="{}", idx=1),
    ])]
    assert ce.referenced_ids(traces, schema) == [("loans", "loan_42")]
    from kullback.builder.sandbox import id_pattern_for

    by_column = EntitySchema(tables=["loans"], id_patterns={"loans.loan_id": r"^loan_\d+$"})
    by_table = EntitySchema(tables=["loans"], id_patterns={"loans": r"^loan_\d+$"})
    assert id_pattern_for(by_column, "loans", "loan_id") == r"^loan_\d+$"
    assert id_pattern_for(by_table, "loans", "loan_id") == r"^loan_\d+$"
    assert id_pattern_for(by_column, "loans", "member_id") is None


def test_a_relative_workdir_still_finds_the_runner_it_just_wrote(tmp_path, monkeypatch, db0, sigs, schema, loan_calls):
    """The sandbox subprocess runs with cwd inside its own directory, so a relative path handed in
    would be resolved against that directory a second time and every path would double. A relative
    workdir once failed every tool this way; every test until now passed an absolute tmp_path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work").mkdir()
    source = ce.module_source(schema, sigs[:1], {"get_loan_details": "return {}"})
    box = ce.Sandbox(source, db0, "work")
    assert box.dir.is_absolute() and box.runner.is_file()
    box.run([loan_calls[0]])


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


# --- what a column holds is a plain list or dict, and the prompt says so ---


def _nursery_nesting_traces():
    """Copies nested under their house, in the shape of world_b: one house read, one copy read."""
    copies = {"copy_11": {"copy_id": "copy_11", "count": 6, "available": True},
              "copy_22": {"copy_id": "copy_22", "count": 2, "available": False}}
    house = {"name": "North House", "house_id": "house_7", "copies": copies}
    copy = {"copy_id": "copy_11", "count": 6, "available": True,
            "light": "full sun"}
    return [_trace("t1", [_call("get_house_details", {"house_id": "house_7"}, result=house),
                          _call("get_copy_details", {"copy_id": "copy_11"}, result=copy, idx=1)])]


def test_mining_reads_a_nested_home_off_the_traces():
    """copies sat under houses.copies in every get_house_details result, keyed by copy_id,
    and the schema still said copies was a top-level table: reads answered from the wrong
    home miss, so the Starting state folds them into the home the traces showed."""
    from kullback.builder import mine
    schema = mine.mine_schema(_nursery_nesting_traces())
    assert schema.homes == {"copies": "houses.copies"}
    assert "copies" in schema.tables and "houses" in schema.tables
    names = {c.name for c in schema.columns if c.table == "copies"}
    assert {"copy_id", "count", "available", "light"} <= names


def test_a_nested_collection_never_shown_on_its_own_stays_a_column_of_its_parent():
    from kullback.builder import mine
    traces = _nursery_nesting_traces()
    traces[0].tool_calls = traces[0].tool_calls[:1]
    schema = mine.mine_schema(traces)
    assert schema.homes == {}
    assert "copies" not in schema.tables


def test_the_starting_state_folds_a_standalone_row_into_its_home(workdir):
    from kullback.builder import mine
    traces = _nursery_nesting_traces()
    schema = mine.mine_schema(traces)
    state = ce.build_starting_state(traces, schema, workdir, synthetic=False)
    assert "copy_11" not in state.db["copies"]
    nested = state.db["houses"]["house_7"]["copies"]["copy_11"]
    assert nested["light"] == "full sun", "the standalone copy's extra field is kept"
    assert nested["count"] == 6
    assert any("is stored under houses.copies" in a for a in state.assumptions)


def test_a_row_whose_parent_was_never_shown_stays_in_its_table():
    db = {"copies": {"c1": {"copy_id": "c1"}, "c2": {"copy_id": "c2"}},
          "houses": {"h": {"house_id": "h", "copies": {"c1": {"copy_id": "c1", "count": 1}}}}}
    schema = EntitySchema(tables=["copies", "houses"], homes={"copies": "houses.copies"})
    assert ce.fold_into_homes(db, schema) == [("copies", "c1", "houses.copies")]
    assert list(db["copies"]) == ["c2"]


# --- the Starting state grows on request, and the ids are tagged (D40, D107) ---

def _twelve_member_traces():
    first = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Eli", "Ivy", "Max", "Uma", "Kai", "Lea", "Tom"]
    last = ["Chen", "Diaz", "Khan", "Lee", "Moss", "Nair", "Ortiz", "Park", "Quinn", "Reyes", "Sato", "Voss"]
    traces = []
    for n, (f, l_) in enumerate(zip(first, last, strict=True)):
        uid = f"{f.lower()}_{l_.lower()}_{1000 + n}"
        row = {"member_id": uid, "name": {"first_name": f, "last_name": l_},
               "email": f"{f.lower()}.{l_.lower()}{2000 + n}@example.com", "tier": ["gold", "basic"][n % 2]}
        traces.append(_trace(f"u{n}", [_call("get_member_details", {"member_id": uid}, result=row, idx=n)]))
    return traces


def test_the_starting_state_grows_only_when_asked_and_tags_every_grown_row(tmp_path):
    from kullback.builder import mine
    traces = _twelve_member_traces()
    schema = mine.mine_schema(traces)
    state = ce.build_starting_state(traces, schema, tmp_path, grow={"members": 20}, grow_seed=1)
    assert len(state.db["members"]) == 20
    assert len(state.synthetic_rows) == 8
    assert set(state.synthetic_rows) <= set(state.db["members"]) and set(state.synthetic_rows) <= set(schema.synthetic_rows)
    assert any("8 synthetic rows" in note and "D107" in note for note in state.assumptions)
    written = json.loads((tmp_path / "synthetic.json").read_text())
    assert written["added"] == {"members": 8} and written["checks"]["ok"] is True
    grown = state.db["members"][state.synthetic_rows[0]]
    first, last = grown["name"]["first_name"], grown["name"]["last_name"]
    assert grown["member_id"].startswith(f"{first.lower()}_{last.lower()}_")
    assert grown["email"].startswith(f"{first.lower()}.{last.lower()}")
    # without grow nothing is grown and no synthetic file is written
    ungrown = tmp_path / "ungrown"
    ungrown.mkdir()
    ungrown_traces = _twelve_member_traces()
    state = ce.build_starting_state(ungrown_traces, mine.mine_schema(ungrown_traces), ungrown)
    assert len(state.db["members"]) == 12 and state.synthetic_rows == []
    assert not (ungrown / "synthetic.json").exists()


# --- gate 4 against the recording, gate 6 the refusal probe ---

STRICT_CANCEL_BODY = """
if loan_id not in self.db.loans:
    raise ValueError("Loan not found")
loan = self.db.loans[loan_id]
loan.status = "cancelled"
return loan
"""

PERMISSIVE_CANCEL_BODY = """
loan = self.db.loans.get(loan_id)
if loan is None:
    return {"loan_id": loan_id, "status": "cancelled"}
loan.status = "cancelled"
return loan
"""


def _cancel_calls(sample):
    pending = sample["by_status"]["pending"]
    return [_call("update_loan", {"loan_id": pending["loan_id"], "reason": "no longer needed"},
                  result=dict(pending, status="cancelled"), idx=idx) for idx in range(2)]


CANCEL_SIG = ToolSig(name="update_loan", kind="write", unclassified=False,
                     args_fields=[FieldStat(name="loan_id", types=["str"], optional=False),
                                  FieldStat(name="reason", types=["str"], optional=False)])


def _both_tools(schema, sigs, db0, workdir, cancel_body):
    source = ce.module_source(schema, [sigs[0], CANCEL_SIG],
                              {"get_loan_details": CORRECT_BODY, "update_loan": cancel_body})
    return source, ce.Sandbox(source, db0, workdir)


def test_a_constant_body_passes_non_trivial_when_the_recorded_tool_was_constant(schema, sigs, db0, workdir, loan_calls):
    """The recording is the standard: a tool that answered every call the same way is faithfully constant."""
    constant = [c.model_copy(update={"result": {"loan_id": "loan_9999", "status": "delivered"}})
                for c in loan_calls[:3]]
    source, box = _sandbox(schema, sigs, db0, workdir, CONSTANT_BODY)
    result = ce.gate_non_trivial(box, constant)
    assert result.passed is True and result.metrics["recorded_constant"] is True
    assert result.metrics["recorded_answers"] == 1 and result.metrics["arg_sets"] == 3


# --- gate 7: a body may not memorise the recordings (D162) ---


MEMORISING_BODY = """
statuses = {"loan_101": "pending", "loan_202": "processed"}
return {"loan_id": loan_id, "status": statuses[loan_id]}
"""


def test_the_refusal_probe_fails_a_body_that_writes_on_an_id_nobody_holds(schema, sigs, db0, workdir, sample):
    calls = _cancel_calls(sample)
    for body, expected in ((STRICT_CANCEL_BODY, True), (PERMISSIVE_CANCEL_BODY, False)):
        source, box = _both_tools(schema, sigs, db0, workdir, body)
        result = sandbox_mod.gate_refuses_unknown(box, calls)
        assert result.passed is expected, body
        assert result.metrics["reference_args"] == 1
    assert "accepted loan_id=" in result.failures[0]
    source, box = _both_tools(schema, sigs, db0, workdir, STRICT_CANCEL_BODY)
    assert list(sandbox_mod.reference_args(box, _cancel_calls(sample))) == ["loan_id"]  # reason is free text


def test_the_refusal_probe_runs_on_write_tools_only(schema, sigs, db0, workdir, sample):
    calls = _cancel_calls(sample)
    source, box = _both_tools(schema, sigs, db0, workdir, PERMISSIVE_CANCEL_BODY)
    assert "refuses_unknown" not in [g.stage for g in ce.run_gates(source, box, calls, [], schema)]
    probed = ce.run_gates(source, box, calls, [], schema, probe_refusals=True)
    assert probed[-1].stage == "refuses_unknown" and probed[-1].passed is False


# --- the builder tools: lookup_rows and test_body, in a bounded loop (D117) ---


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


def test_a_composite_key_keeps_both_versions_of_one_id_and_a_single_key_table_is_keyed_as_before(workdir, schema, sample):
    loan = sample["by_status"]["pending"]
    assert ce.match_table(schema, loan, {"anything": "at all"}) == ("loans", loan["loan_id"])
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
    body = "def f(x):\n    return x + 1\n"
    assert ce.sanitize_body(body) == (body, None)
    assert ce.sanitize_body(f"def f(x):\n  return (x {DASH} 1\n") == (
        f"def f(x):\n  return (x {DASH} 1\n", None)
    fixed, _ = ce.sanitize_body(f"def f():\n    return 1  # {QUOTE}a{UNQUOTE} and {ELLIPSIS}\n")
    assert fixed == 'def f():\n    return 1  # "a" and ...\n'


def test_a_dash_inside_an_f_string_expression_is_replaced_and_the_printed_text_is_kept():
    """The expression inside a replacement field is code, and a confusable there fails the parse
    exactly as one on a line of its own does. What the f-string prints around it is not."""
    body = f'def f(value):\n    return f"keep {DASH} here {{value {DASH} 1}}"\n'
    fixed, note = ce.sanitize_body(body)
    assert _parses(fixed)
    assert fixed == f'def f(value):\n    return f"keep {DASH} here {{value - 1}}"\n'
    assert note is not None
    for printed in [f'f"{{v:{DASH}>10}}"', f'f"{{v!r}} {DASH} x"', f'f"{{{{{DASH}}}}} x"',
                    f"f\"{{d['a{DASH}b']}}\""]:
        body = f"def f(v, d):\n    return {printed}\n"
        assert ce.sanitize_body(body) == (body, None), printed
    token = f"f\"keep {DASH} here {{d['a{DASH}b'] {DASH} 1}} {{w:>{{n}}}}\""
    runs = [token[begin:end] for begin, end in ce._fstring_expression_spans(token)]
    assert runs == ["d[", f"] {DASH} 1", "w", "n"]
    assert ce._fstring_expression_spans(f'"plain {DASH} string"') == []
    line = f'x = f"keep {DASH} here {{value {DASH} 1}}"\n'
    tokens = [tokenize.TokenInfo(tokenize.STRING, line[4:-1], (1, 4), (1, len(line) - 1), line)]
    printed = ce._printed_positions(tokens, [line])
    assert (1, line.index(DASH)) in printed
    assert (1, line.index(DASH, line.index("value"))) not in printed


def test_a_non_ascii_character_with_no_replacement_is_named_rather_than_guessed_at():
    body = f"def f(x):\n    return x {PLUS_MINUS} 1\n"
    fixed, note = ce.sanitize_body(body)
    assert fixed == body
    assert "U+00B1" in note


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
    schema = EntitySchema(tables=["berth slots"], columns=[
        Column(table="berth slots", name="class", **{"class": "hard"})])
    assert ce.unsafe_names(schema) == ["table 'berth slots' is not a Python name",
                                       "column 'class' of berth slots is not a Python name"]
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


# --- D246: the compile_tools envelope is frozen per build ---


# --- a row a recording created is not a starting fact (D290) -----------------


def _mooring_schema() -> EntitySchema:
    """An invented harbour: a skipper reads their moorings and a write reserves a new one."""
    names = {"skippers": ["skipper_id", "name"], "moorings": ["mooring_id", "skipper_id", "berth"]}
    return EntitySchema(
        tables=sorted(names),
        columns=[Column(table=table, name=name, **{"class": "hard"}, classified_by="rule")
                 for table, columns in sorted(names.items()) for name in columns],
        id_patterns={"skippers.skipper_id": r"^SK\d+$", "moorings.mooring_id": r"^M\d+$"},
    )


MOORING_SIGS = [ToolSig(name="find_skipper", kind="read", unclassified=False),
                ToolSig(name="find_mooring", kind="read", unclassified=False),
                ToolSig(name="reserve_mooring", kind="write", unclassified=False)]


def _reserving_trace(trace_id, skipper="SK1"):
    """A Trace that reads its skipper, then reserves mooring M1 and reads it back: M1 is its own."""
    return _trace(trace_id, [
        _call("find_skipper", {"skipper_id": skipper}, result={"skipper_id": skipper, "name": "Ada"}, idx=0),
        _call("reserve_mooring", {"skipper_id": skipper, "berth": "north"},
              result={"mooring_id": "M1", "skipper_id": skipper, "berth": "north"}, idx=1),
        _call("find_mooring", {"mooring_id": "M1"},
              result={"mooring_id": "M1", "skipper_id": skipper, "berth": "north"}, idx=2),
    ])


def test_a_row_a_write_created_is_absent_from_the_shared_db_and_the_overlay(workdir):
    schema, trace = _mooring_schema(), _reserving_trace("A")
    state = ce.build_starting_state([trace], schema, workdir, tasks=[Task(id="t_reserve", run_ids=["A"])],
                                    tool_sigs=MOORING_SIGS)
    assert "M1" not in state.db["moorings"]
    assert state.synthetic_rows == []  # the later read names what the Trace made, so nothing is filled
    overlay, _ = ce.load_overlay(workdir, "t_reserve")
    assert [(row.table, row.id) for row in overlay.rows] == [("skippers", "SK1")]
    assert "moorings row M1 was created by A, reserve_mooring: absent at start" in state.assumptions
    feed = ce.recorded_call_contexts(trace.tool_calls, schema)[sandbox_mod.context_feed_key(trace.tool_calls[1])]
    assert feed["new_ids"] == {"moorings": ["M1"]}


def test_a_row_read_before_any_write_stays_in_the_shared_db(workdir):
    schema = _mooring_schema()
    trace = _trace("A", [
        _call("find_mooring", {"mooring_id": "M4"}, result={"mooring_id": "M4", "skipper_id": "SK1", "berth": "east"},
              idx=0),
        _call("reserve_mooring", {"skipper_id": "SK1", "berth": "east"},
              result={"mooring_id": "M4", "skipper_id": "SK1", "berth": "west"}, idx=1),
    ])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=MOORING_SIGS, synthetic=False)
    assert state.db["moorings"]["M4"]["berth"] == "east"
    assert not any("absent at start" in line for line in state.assumptions)


def test_a_row_another_trace_read_without_creating_it_is_in_that_tasks_overlay_only(workdir):
    schema = _mooring_schema()
    reader = _trace("B", [_call("find_mooring", {"mooring_id": "M1"},
                                result={"mooring_id": "M1", "skipper_id": "SK2", "berth": "south"}, idx=0)])
    tasks = [Task(id="t_reserve", run_ids=["A"]), Task(id="t_read", run_ids=["B"])]
    state = ce.build_starting_state([_reserving_trace("A"), reader], schema, workdir, tasks=tasks,
                                    tool_sigs=MOORING_SIGS)
    # The creator's replay starts without M1: the shared world lacks it and so does its overlay.
    assert "M1" not in state.db["moorings"]
    assert "M1" not in json.loads((workdir / ce.DB_FILE).read_text())["moorings"]
    reserve_overlay, _ = ce.load_overlay(workdir, "t_reserve")
    assert ("moorings", "M1") not in {(row.table, row.id) for row in reserve_overlay.rows}
    # The reader is served M1 by its own overlay, with the values that Trace read.
    read_overlay, read_values = ce.load_overlay(workdir, "t_read")
    [pinned] = [row for row in read_overlay.rows if (row.table, row.id) == ("moorings", "M1")]
    assert read_values[pinned.version_hash] == {"mooring_id": "M1", "skipper_id": "SK2", "berth": "south"}
    assert "M1" not in state.synthetic_rows
    assert ("moorings row M1 was created by A, reserve_mooring: absent at start; absent from the shared "
            "world; in the overlay of t_read which read it") in state.assumptions


def test_a_row_a_write_answers_nested_inside_the_row_it_changed_is_not_a_creation(workdir):
    schema = _mooring_schema()
    trace = _trace("A", [_call("reserve_mooring", {"skipper_id": "SK3", "berth": "north"},
                               result={"skipper_id": "SK3", "name": "Bo",
                                       "moorings": [{"mooring_id": "M7", "skipper_id": "SK3", "berth": "north"}]},
                               idx=0)])
    state = ce.build_starting_state([trace], schema, workdir, tool_sigs=MOORING_SIGS, synthetic=False)
    assert "M7" in state.db["moorings"]
    assert not any("absent at start" in line for line in state.assumptions)
