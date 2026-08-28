"""Tests for the section 6 gates, the D80 scorecard, the D89 import boundary and the RunnerVersion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.runner.validate import (
    MISS_REASONS,
    audit_gate,
    candidate_runs_gate,
    compile_tools_gates,
    deterministic_gate,
    environment_gate,
    executes_gate,
    import_boundary_check,
    ingest_gate,
    leak_gate,
    mine_gate,
    non_trivial_gate,
    oracle_replay_gate,
    parses_gate,
    policy_gate,
    regrade_gate,
    replay_fidelity_gate,
    runner_version,
    scorecard,
    setup_review_gate,
    user_rules_gate,
    verdict_golden_gate,
    verifier_gate,
)
from harness.shared.records import (
    Constraint,
    ConstraintTests,
    DisclosureRule,
    Environment,
    EvidenceStrength,
    GateResult,
    Task,
    ToolSig,
    Trace,
    UserFact,
    UserRules,
    Verdict,
    as_dict,
)


# --- ingest ---

def a_trace(**kw) -> Trace:
    base = dict(trace_id="t1", raw_hash="r1", ingest_version="1", source="tau2", hash="h1")
    base.update(kw)
    return Trace(**base)


def test_ingest_gate_passes_on_a_clean_trace():
    trace = a_trace(tool_calls=[{"name": "get_order", "args": {}, "result": {"id": "W1"}}])
    out = ingest_gate([trace])
    assert isinstance(out, GateResult)
    assert out.stage == "ingest"
    assert out.passed is True
    assert out.metrics["traces"] == 1
    assert out.metrics["tool_calls"] == 1


def test_ingest_gate_fails_a_call_with_neither_result_nor_error():
    trace = a_trace(tool_calls=[{"name": "get_order", "args": {}}])
    out = ingest_gate([trace])
    assert out.passed is False
    assert any("get_order" in f for f in out.failures)


def test_ingest_gate_fails_on_a_grader_field():
    trace = a_trace(tool_calls=[{"name": "t", "args": {"reward_info": 1}, "result": 1}])
    out = ingest_gate([trace])
    assert out.passed is False
    assert any("reward_info" in f for f in out.failures)


def test_ingest_gate_fails_when_the_hash_moved_between_two_passes():
    trace = a_trace(tool_calls=[{"name": "t", "args": {}, "result": 1}])
    assert ingest_gate([trace], second_pass={"t1": "h1"}).passed is True
    out = ingest_gate([trace], second_pass={"t1": "h2"})
    assert out.passed is False
    assert any("hash" in f for f in out.failures)


def test_ingest_gate_fails_a_trace_with_no_hash():
    out = ingest_gate([a_trace(hash="")])
    assert out.passed is False


# --- mine ---

def a_sig(name="get_order", calls=3, **kw) -> ToolSig:
    base = dict(
        name=name,
        args_schema={"properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        evidence_strength=EvidenceStrength(call_count=calls),
    )
    base.update(kw)
    return ToolSig(**base)


def test_mine_gate_passes_with_three_calls_and_valid_args():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {"order_id": "W1"}}])
    assert out.stage == "mine"
    assert out.passed is True


def test_mine_gate_fails_a_thin_signature_that_is_not_flagged_llm():
    out = mine_gate([a_sig(calls=2)])
    assert out.passed is False
    assert any("get_order" in f and "2" in f for f in out.failures)


def test_mine_gate_allows_a_thin_signature_flagged_llm():
    assert mine_gate([a_sig(calls=1, source="llm")]).passed is True


def test_mine_gate_fails_a_recorded_arg_outside_the_schema():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {"order_id": "W1", "colour": "red"}}])
    assert out.passed is False
    assert any("colour" in f for f in out.failures)


def test_mine_gate_fails_a_missing_required_arg():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {}}])
    assert out.passed is False
    assert any("order_id" in f for f in out.failures)


# --- the five compile-tool gates, in order ---

def test_parses_gate():
    assert parses_gate({"a": "def a(x):\n    return x\n"}).passed is True
    bad = parses_gate({"a": "def a(x)\n    return x\n"})
    assert bad.passed is False
    assert any("a" in f for f in bad.failures)


def test_executes_gate():
    assert executes_gate({"a": {"ok": True}}).passed is True
    out = executes_gate({"a": {"ok": False, "error": "KeyError: orders"}})
    assert out.passed is False
    assert any("KeyError" in f for f in out.failures)


def test_deterministic_gate_uses_canonicalization():
    assert deterministic_gate({"a": [{"total": 25}, {"total": 25.0}]}).passed is True
    assert deterministic_gate({"a": [{"total": 25}, {"total": 26}]}).passed is False


def test_non_trivial_gate_rejects_a_constant_tool():
    assert non_trivial_gate({"a": [{"x": 1}, {"x": 2}]}).passed is True
    out = non_trivial_gate({"a": [{"x": 1}, {"x": 1}]})
    assert out.passed is False
    assert any("constant" in f for f in out.failures)


def test_non_trivial_gate_needs_two_samples():
    assert non_trivial_gate({"a": [{"x": 1}]}).passed is False


def test_replay_fidelity_reports_success_and_error_separately():
    calls = [
        {"tool": "get_order", "expected": {"id": "W1", "total": 25}, "actual": {"id": "W1", "total": 25.0},
         "held_out": True},
        {"tool": "get_order", "expected": {"id": "W2"}, "actual": {"id": "W3"},
         "held_out": True, "reason": "our_bug"},
        {"tool": "get_order", "expected_error": {"class": "not_found_entity"},
         "actual_error": {"class": "not_found_entity"}, "held_out": True},
        {"tool": "get_order", "expected_error": {"class": "not_found_entity"},
         "actual_error": {"class": "business_error"}, "held_out": True},
    ]
    out = replay_fidelity_gate(calls)
    assert out.passed is False
    success = out.metrics["success"]
    error = out.metrics["error"]
    assert (success["total"], success["matched"]) == (2, 1)
    assert (error["total"], error["matched"]) == (2, 1)
    assert success["raw"] == 0.5
    assert success["explained"] == 1.0
    assert error["unexplained"] == 1
    assert out.metrics["per_tool"]["get_order"]["success"]["total"] == 2


def test_replay_fidelity_fails_an_explained_miss_on_a_recorded_call():
    calls = [{"tool": "t", "expected": 1, "actual": 2, "held_out": False, "reason": "our_bug"}]
    out = replay_fidelity_gate(calls)
    assert out.passed is False
    assert any("recorded" in f for f in out.failures)


def test_compile_tools_gates_stop_at_the_first_failure():
    evidence = {
        "sources": {"a": "def a(x)\n    return x\n"},
        "outcomes": {"a": {"ok": True}},
        "runs": {"a": [1, 1]},
        "outputs": {"a": [1, 2]},
        "calls": [],
    }
    out = compile_tools_gates(evidence)
    assert [g.stage for g in out] == ["compile_tools.parses"]
    assert out[-1].passed is False

    evidence["sources"] = {"a": "def a(x):\n    return x\n"}
    out = compile_tools_gates(evidence)
    assert [g.stage for g in out] == [
        "compile_tools.parses",
        "compile_tools.executes",
        "compile_tools.deterministic",
        "compile_tools.non_trivial",
        "compile_tools.replay_fidelity",
    ]
    assert all(g.passed for g in out)


# --- policy ---

def a_constraint(**kw) -> Constraint:
    base = dict(
        id="c1",
        text="never cancel a delivered order",
        compiled=True,
        predicate_src="def check(case):\n    return case['status'] != 'delivered'\n",
        tests=ConstraintTests(pos=[{"status": "pending"}], neg=[{"status": "delivered"}]),
    )
    base.update(kw)
    return Constraint(**base)


def test_policy_gate_runs_the_positive_and_negative_cases():
    out = policy_gate([a_constraint()])
    assert out.stage == "compile_policy"
    assert out.passed is True
    assert out.metrics["compiled"] == 1


def test_policy_gate_fails_when_a_negative_case_is_allowed():
    bad = a_constraint(predicate_src="def check(case):\n    return True\n")
    out = policy_gate([bad])
    assert out.passed is False
    assert any("neg" in f for f in out.failures)


def test_policy_gate_counts_residuals_without_failing():
    out = policy_gate([a_constraint(), Constraint(id="c2", text="be nice", residual_reason="not checkable")])
    assert out.passed is True
    assert out.metrics["residual"] == 1


def test_policy_gate_fails_a_reference_that_violates_policy():
    out = policy_gate([a_constraint()], reference_violations=[{"run_id": "r1", "constraint_id": "c1"}])
    assert out.passed is False
    assert any("r1" in f for f in out.failures)


def test_policy_gate_takes_an_evaluator():
    calls = []

    def evaluate(constraint, case):
        calls.append(case)
        return case["status"] != "delivered"

    assert policy_gate([a_constraint()], evaluate=evaluate).passed is True
    assert len(calls) == 2


# --- environment ---

def a_env(**kw) -> Environment:
    files = {name: "h" for name in ("data_model.py", "tools.py", "db.json", "policy.md", "tasks.json")}
    base = dict(env_id="e1", files=files)
    base.update(kw)
    return Environment(**base)


def test_environment_gate_passes_the_tau2_shape():
    out = environment_gate(a_env(), referenced_ids=["W1"], db_ids=["W1", "W2"])
    assert out.stage == "build_environment"
    assert out.passed is True


def test_environment_gate_fails_a_missing_file():
    env = a_env(files={"db.json": "h"})
    out = environment_gate(env)
    assert out.passed is False
    assert any("tools.py" in f for f in out.failures)


def test_environment_gate_fails_an_id_the_traces_reference_but_the_db_lacks():
    out = environment_gate(a_env(), referenced_ids=["W1", "W9"], db_ids=["W1"])
    assert out.passed is False
    assert any("W9" in f for f in out.failures)


def test_environment_gate_reads_ids_from_a_files_dir(tmp_path: Path):
    (tmp_path / "db.json").write_text(json.dumps({"orders": [{"order_id": "W1"}]}), encoding="utf-8")
    for name in ("data_model.py", "tools.py", "policy.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "tasks.json").write_text("[]", encoding="utf-8")
    assert environment_gate(a_env(), files_dir=tmp_path, referenced_ids=["W1"]).passed is True
    assert environment_gate(a_env(), files_dir=tmp_path, referenced_ids=["W7"]).passed is False


def test_environment_gate_fails_an_untagged_synthetic_row():
    out = environment_gate(a_env(), synthetic_rows=["row1"])
    assert out.passed is False
    assert any("row1" in f for f in out.failures)
    assert environment_gate(a_env(), synthetic_rows=[{"id": "row1", "synthetic": True}]).passed is True


# --- simulated user rules ---

def a_rules(**kw) -> UserRules:
    base = dict(
        facts=[UserFact(field="zip", value="94105")],
        disclosure=[DisclosureRule(field="zip")],
        refusals=["will not share the card number"],
    )
    base.update(kw)
    return UserRules(**base)


def test_user_rules_gate_passes():
    out = user_rules_gate(a_rules(), asked_fields=["zip"], trace_refusals=["will not share the card number"])
    assert out.stage == "build_user_rules"
    assert out.passed is True


def test_user_rules_gate_fails_a_missing_disclosure_rule():
    out = user_rules_gate(a_rules(), asked_fields=["zip", "email"])
    assert out.passed is False
    assert any("email" in f for f in out.failures)
    assert out.metrics["incomplete_reasons"] == out.failures


def test_user_rules_gate_fails_a_missing_refusal_branch():
    out = user_rules_gate(a_rules(refusals=[]), trace_refusals=["no card number"])
    assert out.passed is False
    assert any("refusal" in f for f in out.failures)


def test_user_rules_gate_checks_fact_consistency_on_rerun():
    assert user_rules_gate(a_rules(), rerun_facts=[{"zip": "94105"}]).passed is True
    out = user_rules_gate(a_rules(), rerun_facts=[{"zip": "10001"}])
    assert out.passed is False
    assert any("zip" in f for f in out.failures)


# --- gate A, oracle replay ---

def test_oracle_replay_gate_splits_seed_and_held_out():
    replays = [
        {"run_id": "r1", "held_out": False, "writes": [{"expected": {"total": 25}, "actual": {"total": 25.0}}]},
        {"run_id": "r2", "held_out": True, "writes": [{"expected": {"total": 25}, "actual": {"total": 26}}]},
    ]
    out = oracle_replay_gate(replays)
    assert out.stage == "gate_a_oracle_replay"
    assert out.passed is False
    assert out.metrics["seed"]["writes"] == 1
    assert out.metrics["seed"]["matched"] == 1
    assert out.metrics["held_out"]["matched"] == 0
    assert any("r2" in f for f in out.failures)


def test_oracle_replay_gate_fails_a_semantic_read_mismatch():
    replays = [{"run_id": "r1", "held_out": False, "writes": [],
                "semantic_reads": [{"expected": "blue shirt", "actual": "red shirt"}]}]
    out = oracle_replay_gate(replays)
    assert out.passed is False
    assert out.metrics["seed"]["semantic_mismatches"] == 1


# --- verifier suite (D79) and the leak check ---

def test_verifier_gate_needs_every_d79_check():
    checks = {
        "provenance_spans": True, "oracle_passes": True, "empty_fails": True,
        "plausible_wrong_fails": True, "second_path_passes": True,
        "loophole_probe_fails": True, "leak_check_clean": True, "mutation_flips": True,
    }
    assert verifier_gate(checks).passed is True
    checks.pop("mutation_flips")
    out = verifier_gate(checks)
    assert out.passed is False
    assert any("mutation_flips" in f and "not run" in f for f in out.failures)
    checks["mutation_flips"] = False
    assert any("failed" in f for f in verifier_gate(checks).failures)


def test_leak_gate_finds_a_verifier_constant_in_the_intent():
    assert leak_gate(["cancel the pending order"], ["W0000000"]).passed is True
    out = leak_gate(["cancel order W0000000"], ["W0000000"])
    assert out.passed is False
    assert any("W0000000" in f for f in out.failures)


# --- setup review, candidate runs, verdict goldens, audit, regrade ---

def test_setup_review_gate():
    assert setup_review_gate(["t1"], ["t1", "t2"]).passed is True
    out = setup_review_gate(["t1", "t3"], ["t1"])
    assert out.passed is False
    assert any("t3" in f for f in out.failures)


def a_run(run_id="r1", task_id="t1", seed=1, stopped=True, **kw) -> dict:
    events = [{"idx": 0, "type": "tool_call", "payload": {"name": "x"}}]
    if stopped:
        events.append({"idx": 1, "type": "stop", "payload": {"reason": "done"}})
    run = {"run_id": run_id, "task_id": task_id, "seed": seed, "events": events}
    run.update(kw)
    return run


def test_candidate_runs_gate_wants_k_complete_runs_with_seeds():
    runs = [a_run("r1", seed=1), a_run("r2", seed=2)]
    assert candidate_runs_gate(runs, k=2).passed is True
    out = candidate_runs_gate(runs, k=3)
    assert out.passed is False
    assert any("t1" in f for f in out.failures)


def test_candidate_runs_gate_fails_an_incomplete_jsonl_and_a_missing_seed():
    out = candidate_runs_gate([a_run("r1", stopped=False), a_run("r2", seed=None)], k=1)
    assert out.passed is False
    assert any("r1" in f and "stop" in f for f in out.failures)
    assert any("r2" in f and "seed" in f for f in out.failures)


def test_verdict_golden_gate():
    checks = {"oracle_passes": True, "empty_fails": True, "plausible_wrong_fails": True, "two_orders_pass": True}
    assert verdict_golden_gate(checks).passed is True
    checks["empty_fails"] = False
    assert verdict_golden_gate(checks).passed is False


def test_audit_gate_needs_a_sample_per_task_and_a_published_agreement():
    assert audit_gate({"t1": 2}, ["t1"], agreement=0.9).passed is True
    assert audit_gate({"t1": 2}, ["t1", "t2"], agreement=0.9).passed is False
    out = audit_gate({"t1": 2}, ["t1"])
    assert out.passed is False
    assert any("agreement" in f for f in out.failures)


def a_verdict(**kw) -> Verdict:
    base = dict(
        run_id="r1", env_id="e1", schema_version="1", tools_version="1", policy_version="1",
        verifier_version="1", verdict_version="1", runner_version="rv1", passed=True, class_="pass",
    )
    base.update(kw)
    return Verdict(**base)


def test_regrade_gate_wants_every_version_on_every_verdict():
    assert regrade_gate([a_verdict()]).passed is True
    out = regrade_gate([a_verdict(runner_version=None)])
    assert out.passed is False
    assert any("runner_version" in f for f in out.failures)


# --- D89 import boundary ---

def test_import_boundary_check_passes_on_this_repo():
    import harness

    root = Path(harness.__file__).resolve().parent
    out = import_boundary_check(root)
    assert out.stage == "import_boundary"
    assert out.passed is True, out.failures


def test_import_boundary_check_accepts_the_src_directory_too():
    import harness

    src = Path(harness.__file__).resolve().parents[1]
    assert import_boundary_check(src).passed is True


def _tree(root: Path, runner_src: str, shared_src: str = "x = 1\n") -> Path:
    for part in ("runner", "shared", "builder"):
        (root / part).mkdir(parents=True)
        (root / part / "__init__.py").write_text("", encoding="utf-8")
    (root / "runner" / "loop.py").write_text(runner_src, encoding="utf-8")
    (root / "shared" / "canon.py").write_text(shared_src, encoding="utf-8")
    (root / "builder" / "verifier.py").write_text("x = 1\n", encoding="utf-8")
    return root


def test_import_boundary_check_catches_a_top_level_builder_import(tmp_path: Path):
    root = _tree(tmp_path / "harness", "from harness.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("loop.py" in f and "harness.builder" in f for f in out.failures)


def test_import_boundary_check_catches_a_lazy_builder_import(tmp_path: Path):
    src = "def go():\n    import harness.builder.verifier as v\n    return v\n"
    out = import_boundary_check(_tree(tmp_path / "harness", src))
    assert out.passed is False


def test_import_boundary_check_catches_importlib_by_string(tmp_path: Path):
    src = "import importlib\n\n\ndef go():\n    return importlib.import_module('harness.builder.verifier')\n"
    out = import_boundary_check(_tree(tmp_path / "harness", src))
    assert out.passed is False
    assert any("import_module" in f for f in out.failures)


def test_import_boundary_check_catches_builder_reaching_runner_through_shared(tmp_path: Path):
    root = _tree(tmp_path / "harness", "x = 1\n", shared_src="from harness.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("canon.py" in f for f in out.failures)


def test_import_boundary_check_allows_shared_imports_from_runner(tmp_path: Path):
    root = _tree(tmp_path / "harness", "from harness.shared.records import Run\n")
    assert import_boundary_check(root).passed is True


# --- RunnerVersion ---

def _runner_tree(root: Path, loop_body="a = 1\n") -> Path:
    (root / "runner").mkdir(parents=True)
    (root / "runner" / "loop.py").write_text(loop_body, encoding="utf-8")
    (root / "runner" / "route.py").write_text("b = 2\n", encoding="utf-8")
    (root / "runner" / "verdict.py").write_text("c = 3\n", encoding="utf-8")
    return root


def test_runner_version_hashes_the_three_files_and_the_routing_config(tmp_path: Path):
    root = _runner_tree(tmp_path / "harness")
    out = runner_version(root, routing_config={"order": ["code", "recording", "llm"]})
    assert set(out.file_hashes) == {"loop.py", "route.py", "verdict.py"}
    assert all(len(h) == 64 for h in out.file_hashes.values())
    assert out.routing_config_hash is not None
    assert len(out.runner_version) == 64
    assert runner_version(root, routing_config={"order": ["code", "recording", "llm"]}).runner_version == \
        out.runner_version


def test_runner_version_moves_when_a_runner_file_changes(tmp_path: Path):
    root = _runner_tree(tmp_path / "harness")
    before = runner_version(root).runner_version
    (root / "runner" / "loop.py").write_text("a = 2\n", encoding="utf-8")
    assert runner_version(root).runner_version != before


def test_runner_version_moves_when_the_routing_config_changes(tmp_path: Path):
    root = _runner_tree(tmp_path / "harness")
    a = runner_version(root, routing_config={"order": ["code"]}).runner_version
    b = runner_version(root, routing_config={"order": ["code", "recording"]}).runner_version
    assert a != b


def test_runner_version_marks_a_file_that_is_not_there_yet(tmp_path: Path):
    root = _runner_tree(tmp_path / "harness")
    (root / "runner" / "verdict.py").unlink()
    out = runner_version(root)
    assert out.file_hashes["verdict.py"] == "missing"


def test_runner_version_on_this_repo():
    import harness

    out = runner_version(Path(harness.__file__).resolve().parent)
    assert len(out.file_hashes["loop.py"]) == 64
    assert len(out.file_hashes["route.py"]) == 64


# --- the D62 / D80 / D96 scorecard ---

@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    """A small build directory: two Tasks, three Runs, one of everything the scorecard reads."""
    root = tmp_path / "build"
    root.mkdir()

    def write(name, payload):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")

    write("tasks.json", [
        as_dict(Task(id="t1", run_ids=["r1", "r2"], intent="cancel an order")),
        as_dict(Task(id="t2", run_ids=["r3"], intent="exchange an item")),
    ])
    write("runs.json", [a_run("r1", "t1"), a_run("r2", "t1"), a_run("r3", "t2")])
    write("task_status.json", {"t1": {"reference_confirmed": True, "verifier_passed": True},
                               "t2": {"reference_confirmed": True, "verifier_passed": True}})
    write("held_out_calls.json", [
        {"tool": "get_order", "expected": {"id": "W1"}, "actual": {"id": "W1"}, "held_out": True},
        {"tool": "get_order", "expected": {"id": "W2"}, "actual": {"id": "W3"}, "held_out": True,
         "reason": "our_bug"},
        {"tool": "get_order", "expected_error": {"class": "not_found_entity"},
         "actual_error": {"class": "not_found_entity"}, "held_out": True},
    ])
    write("user_facts.json", [
        {"run_id": "r1", "field": "zip", "expected": "94105", "observed": "94105"},
        {"run_id": "r2", "field": "zip", "expected": "94105", "observed": "10001", "reason": "ambiguous"},
    ])
    write("verdicts.json", [
        {"run_id": "r1", "pass": True, "class": "pass"},
        {"run_id": "r2", "pass": False, "class": "fail"},
        {"run_id": "r3", "pass": True, "class": "pass"},
    ])
    write("reference_verdicts.json", [
        {"run_id": "r1", "pass": True},
        {"run_id": "r2", "pass": True, "reason": "reference_bug"},
        {"run_id": "r3", "pass": True},
    ])
    write("policy.json", {"rules": 40, "compiled": 6, "residual": 2})
    return root


def test_scorecard_tool_fidelity_success_and_error_side_by_side(build_dir: Path):
    card = scorecard(build_dir)
    fidelity = card["tool_fidelity"]
    assert fidelity["success"]["total"] == 2
    assert fidelity["success"]["raw"] == 0.5
    assert fidelity["success"]["explained"] == 1.0
    assert fidelity["error"]["raw"] == 1.0
    assert fidelity["per_tool"]["get_order"]["success"]["matched"] == 1


def test_scorecard_task_coverage_plain_and_run_weighted(build_dir: Path):
    card = scorecard(build_dir)
    coverage = card["task_coverage"]
    assert coverage["tasks_total"] == 2
    assert coverage["tasks_covered"] == 2
    assert coverage["tasks"] == 1.0
    assert coverage["runs_total"] == 3
    assert coverage["run_weighted"] == 1.0
    assert coverage["uncovered"] == []


def test_scorecard_uncovered_task_carries_the_first_failing_reason(build_dir: Path):
    runs = json.loads((build_dir / "runs.json").read_text())
    runs[1]["events"][0]["assisted"] = True
    (build_dir / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert coverage["tasks_covered"] == 1
    assert coverage["tasks"] == 0.5
    assert coverage["run_weighted"] == 0.3333
    assert coverage["uncovered"][0]["task_id"] == "t1"
    assert "assisted" in coverage["uncovered"][0]["reason"]
    assert "r2" in coverage["uncovered"][0]["reason"]


@pytest.mark.parametrize(
    "patch,word",
    [
        ({"events": [{"idx": 0, "type": "user_turn", "payload": {"fact_unavailable": "zip"}}]}, "fact_unavailable"),
        ({"events": [{"idx": 0, "type": "tool_call", "payload": {"name": "x", "overlay_miss": True}}]}, "overlay_miss"),
        ({"events": [{"idx": 0, "type": "tool_result", "payload": {"reconstructed": True}}]}, "reconstructed"),
    ],
)
def test_scorecard_coverage_reasons_cover_d96(build_dir: Path, patch, word):
    runs = json.loads((build_dir / "runs.json").read_text())
    runs[0].update(patch)
    (build_dir / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert word in coverage["uncovered"][0]["reason"]


def test_scorecard_coverage_needs_a_confirmed_reference_and_a_passing_verifier(build_dir: Path):
    (build_dir / "task_status.json").write_text(
        json.dumps({"t1": {"reference_confirmed": False, "verifier_passed": True},
                    "t2": {"reference_confirmed": True, "verifier_passed": False}}), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert coverage["tasks_covered"] == 0
    reasons = {u["task_id"]: u["reason"] for u in coverage["uncovered"]}
    assert "Reference" in reasons["t1"]
    assert "Verifier" in reasons["t2"]


def test_scorecard_counts_a_run_that_was_never_replayed_as_uncovered(build_dir: Path):
    runs = json.loads((build_dir / "runs.json").read_text())
    (build_dir / "runs.json").write_text(json.dumps(runs[1:]), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert "r1" in coverage["uncovered"][0]["reason"]


def test_scorecard_user_fact_consistency(build_dir: Path):
    facts = scorecard(build_dir)["user_fact_consistency"]
    assert facts["total"] == 2
    assert facts["matched"] == 1
    assert facts["raw"] == 0.5
    assert facts["explained"] == 1.0
    assert facts["by_reason"] == {"ambiguous": 1}


def test_scorecard_verdict_agreement_with_reasons(build_dir: Path):
    agreement = scorecard(build_dir)["verdict_agreement"]
    assert agreement["total"] == 3
    assert agreement["matched"] == 2
    assert agreement["raw"] == 0.6667
    assert agreement["explained"] == 1.0
    assert agreement["by_reason"] == {"reference_bug": 1}
    miss = agreement["misses"][0]
    assert miss["run_id"] == "r2"
    assert miss["ours"] is False
    assert miss["reference"] is True


def test_scorecard_takes_a_supplied_reference_verdict_set(build_dir: Path):
    reference = [{"run_id": "r1", "pass": False, "reason": "our_bug"},
                 {"run_id": "r2", "pass": False},
                 {"run_id": "r3", "pass": True}]
    agreement = scorecard(build_dir, reference_verdicts=reference)["verdict_agreement"]
    assert agreement["matched"] == 2
    assert agreement["by_reason"] == {"our_bug": 1}


def test_scorecard_gate_fails_on_a_miss_with_no_reason(build_dir: Path):
    assert scorecard(build_dir)["gate"]["pass"] is True
    reference = json.loads((build_dir / "reference_verdicts.json").read_text())
    reference[1].pop("reason")
    (build_dir / "reference_verdicts.json").write_text(json.dumps(reference), encoding="utf-8")
    card = scorecard(build_dir)
    assert card["gate"]["pass"] is False
    assert any("r2" in f for f in card["gate"]["failures"])
    assert card["verdict_agreement"]["unexplained"] == 1


def test_scorecard_reads_a_miss_reason_from_the_verdict_notes(build_dir: Path):
    reference = json.loads((build_dir / "reference_verdicts.json").read_text())
    reference[1].pop("reason")
    (build_dir / "reference_verdicts.json").write_text(json.dumps(reference), encoding="utf-8")
    verdicts = json.loads((build_dir / "verdicts.json").read_text())
    verdicts[1]["notes"] = ["miss_reason:our_bug"]
    (build_dir / "verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")
    card = scorecard(build_dir)
    assert card["gate"]["pass"] is True
    assert card["verdict_agreement"]["by_reason"] == {"our_bug": 1}


def test_scorecard_rejects_a_reason_outside_the_vocabulary(build_dir: Path):
    reference = json.loads((build_dir / "reference_verdicts.json").read_text())
    reference[1]["reason"] = "because"
    (build_dir / "reference_verdicts.json").write_text(json.dumps(reference), encoding="utf-8")
    card = scorecard(build_dir)
    assert card["gate"]["pass"] is False
    assert set(MISS_REASONS) == {"our_bug", "reference_bug", "ambiguous"}


def test_scorecard_passes_policy_coverage_through(build_dir: Path):
    assert scorecard(build_dir)["policy_coverage"] == {"rules": 40, "compiled": 6, "residual": 2}


def test_scorecard_on_an_empty_build_dir_reports_nothing_rather_than_a_hundred_percent(tmp_path: Path):
    card = scorecard(tmp_path)
    assert card["tool_fidelity"]["success"]["raw"] is None
    assert card["task_coverage"]["tasks"] is None
    assert card["verdict_agreement"]["raw"] is None
    assert card["gate"]["pass"] is True


def test_scorecard_is_json_serializable(build_dir: Path):
    json.dumps(scorecard(build_dir))
