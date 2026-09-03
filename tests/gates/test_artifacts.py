"""Tests for the artifact gates of kullback.gates.artifacts: ingest to regrade, one ruling record each."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PTR
from kullback.gates.artifacts import (
    audit_gate,
    budget_gate,
    candidate_runs_gate,
    compile_tools_gates,
    deterministic_gate,
    environment_gate,
    executes_gate,
    ingest_gate,
    leak_gate,
    mine_gate,
    non_trivial_gate,
    parses_gate,
    policy_gate,
    regrade_gate,
    setup_review_gate,
    user_rules_gate,
    verdict_golden_gate,
    verifier_gate,
)
from kullback.runner.records import (
    Constraint,
    ConstraintTests,
    DisclosureRule,
    Environment,
    EvidenceStrength,
    GateResult,
    ToolSig,
    Trace,
    UserFact,
    UserRules,
    Verdict,
)

# --- ingest ---

def a_trace(**kw) -> Trace:
    base = dict(trace_id="t1", raw_hash="r1", ingest_version="1", source="tau2", hash="h1",
                raw_ptr=PTR)
    base.update(kw)
    calls = [c if not isinstance(c, dict) else {"raw_ptr": PTR, **c} for c in base.get("tool_calls") or []]
    if calls:
        base["tool_calls"] = calls
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


def test_ingest_gate_takes_a_recorded_null_result_as_a_result():
    """A tool that answers with JSON null answered; only a call with nothing recorded fails.

    The flag is read off the call, so this holds for the dict form today and for the `ToolCall`
    record once `records.py` carries `has_result` (see needs_from_others).
    """
    def trace(call):
        return {"trace_id": "t1", "hash": "h1", "tool_calls": [call]}

    assert ingest_gate([trace({"name": "delete_x", "result": None, "has_result": True})]).passed is True
    out = ingest_gate([trace({"name": "delete_x", "has_result": False})])
    assert out.passed is False
    assert any("delete_x" in f for f in out.failures)
    assert ingest_gate([trace({"name": "delete_x"})]).passed is False, \
        "a call with no flag and nothing recorded still fails"


def test_ingest_gate_fails_on_a_grader_field_and_reads_no_customer_result_as_one():
    """D66 is about the benchmark's own sidecar keys; a customer's tool answering `trial` is data."""
    trace = a_trace(tool_calls=[{"name": "t", "args": {"reward_info": 1}, "result": 1}])
    out = ingest_gate([trace])
    assert out.passed is False
    assert any("reward_info" in f for f in out.failures)
    customer = a_trace(tool_calls=[{"name": "get_sub", "args": {}, "result": {"trial": True}}])
    assert ingest_gate([customer]).passed is True


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


def test_mine_gate_believes_a_call_count_of_zero_over_the_evidence_list():
    """A ToolSig that counted no calls is thin, whatever trace ids it lists."""
    out = mine_gate([a_sig(calls=0, evidence=["a", "b", "c"])])
    assert out.passed is False
    assert any("get_order" in f and "0 observed calls" in f for f in out.failures)
    assert mine_gate([a_sig(calls=3, evidence=[])]).passed is True


def test_mine_gate_falls_back_to_the_evidence_only_when_nothing_counted_the_calls():
    """A ToolSig record always carries a count, so the fallback is for the dict form alone."""
    assert mine_gate([{"name": "get_order", "evidence": ["a", "b", "c"]}]).passed is True
    assert mine_gate([{"name": "get_order", "evidence": ["a", "b"]}]).passed is False
    assert mine_gate([ToolSig(name="get_order", evidence=["a", "b", "c"])]).passed is False, \
        "a record that counted no calls is thin, whatever it lists as evidence"


def test_mine_gate_fails_a_recorded_arg_outside_the_schema():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {"order_id": "W1", "colour": "red"}}])
    assert out.passed is False
    assert any("colour" in f for f in out.failures)


def test_mine_gate_fails_a_missing_required_arg():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {}}])
    assert out.passed is False
    assert any("order_id" in f for f in out.failures)


# --- the five compile-tool gates, in order ---

def test_the_parses_gate_names_the_tool_whose_body_does_not_parse():
    assert parses_gate({"a": "def a(x):\n    return x\n"}).passed is True
    bad = parses_gate({"a": "def a(x)\n    return x\n"})
    assert bad.passed is False
    assert any("a" in f for f in bad.failures)


def test_the_executes_gate_carries_the_error_of_a_tool_that_did_not_run():
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
        predicate_src="def check(pre_state, write_call, transcript):\n    return pre_state['status'] != 'delivered'\n",
        tests=ConstraintTests(pos=[{"pre_state": {"status": "pending"}}], neg=[{"pre_state": {"status": "delivered"}}]),
    )
    base.update(kw)
    return Constraint(**base)


def test_policy_gate_runs_the_positive_and_negative_cases():
    out = policy_gate([a_constraint()])
    assert out.stage == "compile_policy"
    assert out.passed is True
    assert out.metrics["compiled"] == 1


def test_the_gate_runs_a_predicate_the_way_the_verdict_does_with_the_state_the_write_and_the_transcript():
    """Build 8: the gate handed the case as one argument and every compiled constraint failed with a TypeError."""
    reads_all_three = a_constraint(
        predicate_src=("def check(pre_state, write_call, transcript):\n"
                       "    return write_call.get('tool') != 'cancel_order' or pre_state.get('status') == 'pending'\n"),
        tests=ConstraintTests(
            pos=[{"pre_state": {"status": "pending"}, "write_call": {"tool": "cancel_order"}, "transcript": []},
                 {"pre_state": {"status": "delivered"}, "write_call": {"tool": "get_order"}, "transcript": []}],
            neg=[{"pre_state": {"status": "delivered"}, "write_call": {"tool": "cancel_order"}, "transcript": []}],
        ),
    )
    assert policy_gate([reads_all_three]).passed is True
    one_argument = a_constraint(predicate_src="def check(case):\n    return True\n")
    out = policy_gate([one_argument])
    assert out.passed is False
    assert any("TypeError" in f for f in out.failures)


def test_policy_gate_fails_when_a_negative_case_is_allowed():
    bad = a_constraint(predicate_src="def check(pre_state, write_call, transcript):\n    return True\n")
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


def test_a_predicate_that_reaches_for_the_builtins_mapping_takes_nothing_from_the_next_predicate():
    """Naming `__builtins__` reaches every allowed builtin as data to edit, so the static check
    refuses the predicate, and the exec that does run gets its own copy of the mapping: either way
    the constraint checked next still has the allowlist it was certified against."""
    reaching = a_constraint(id="c1", predicate_src="def check(pre_state, write_call, transcript):\n"
                                                 "    __builtins__.clear()\n    return True\n",
                            tests=ConstraintTests(pos=[{"pre_state": {"status": "pending"}}]))
    counts = a_constraint(id="c2", predicate_src="def check(pre_state, write_call, transcript):\n"
                                               "    return len(pre_state) > 0\n",
                          tests=ConstraintTests(pos=[{"pre_state": {"status": "pending"}}]))
    out = policy_gate([reaching, counts])
    assert out.passed is False
    assert any("c1" in f and "not confined" in f for f in out.failures)
    assert [f for f in out.failures if f.startswith("c2")] == []
    assert policy_gate([counts]).passed is True


def test_policy_gate_runs_every_pos_and_neg_case_through_the_evaluator_it_is_given():
    calls = []

    def evaluate(constraint, case):
        calls.append(case)
        return case["pre_state"]["status"] != "delivered"

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


def test_user_rules_gate_passes_when_every_asked_field_has_a_disclosure_rule_and_every_trace_refusal_is_kept():
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


def test_a_gate_compares_under_the_customers_canon_rules():
    """A gate that compared under the module defaults could differ from the Verdict (D39, D84)."""
    from kullback.runner.canon import CanonRules

    rounded = CanonRules(number_precision=0)
    assert deterministic_gate({"a": [{"total": 25.4}, {"total": 25.0}]}).passed is False
    assert deterministic_gate({"a": [{"total": 25.4}, {"total": 25.0}]}, canon_rules=rounded).passed is True


# --- verifier suite (D79) and the leak check ---

def test_verifier_gate_needs_every_d79_check():
    checks = {
        "provenance_spans": True, "oracle_passes": True, "empty_fails": True,
        "plausible_wrong_fails": True, "unsolved_state_fails": True, "second_path_passes": True,
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


def test_leak_gate_matches_whole_tokens_not_substrings():
    """A constant of 25 is not a leak in "250" or in the id "order_25", and 25 on its own still is."""
    assert leak_gate(["ship 250 units"], [25]).passed is True
    assert leak_gate(["order_25 shipped"], [25]).passed is True
    assert leak_gate(["ship 25 units"], [25]).passed is False
    assert leak_gate(["cancel the order"], ["order"]).passed is False
    assert leak_gate(["reorder the item"], ["order"]).passed is True


def test_leak_gate_names_the_constants_it_was_too_short_to_use():
    out = leak_gate(["cancel the order"], ["e", "", "W1"])
    assert out.passed is True
    assert out.metrics["skipped"] == ["e", ""]
    assert out.metrics["constants"] == 3


# --- setup review, candidate runs, verdict goldens, audit, regrade ---

def test_setup_review_gate_names_the_task_nobody_reviewed():
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


def test_verdict_golden_gate_fails_when_any_golden_check_fails():
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


def test_the_predicate_builtins_cover_everything_policy_certifies_at_build_time():
    """The same pin test_verdict.py runs for verdict.py's atom gate, for the policy gate's constraint check.

    Both import runner/confinement.py's one allowlist now, but this test stays here so a future
    change that gives either module its own copy again is caught on both sides of the split.
    """
    import re

    from kullback.builder import policy
    from kullback.runner.confinement import SAFE_BUILTIN_NAMES

    block = re.search(r"_ALLOWED = \(\n(.*?)\n\)", policy._RUNNER_SRC, re.S).group(1)
    allowed = set(re.findall(r'"([A-Za-z_]+)"', block))
    assert allowed
    assert allowed <= set(SAFE_BUILTIN_NAMES)

# --- an unpriced Candidate call is a gate failure, not a number in the report (D65, D85) ---

def test_the_budget_gate_fails_a_candidate_batch_with_unpriced_calls():
    out = budget_gate({"stages": {"candidate": {"calls": 10, "unpriced_calls": 3}}})
    assert out.stage == "budget"
    assert out.passed is False
    assert out.metrics["unpriced_calls"] == 3
    assert "3 of 10" in out.failures[0]


def test_the_budget_gate_passes_when_every_call_was_priced():
    out = budget_gate({"stages": {"candidate": {"calls": 10, "unpriced_calls": 0}}})
    assert out.passed is True and out.metrics["calls"] == 10


def test_the_budget_gate_passes_a_batch_that_made_no_call():
    assert budget_gate({"stages": {}}).passed is True

def test_environment_gate_finds_a_row_keyed_by_a_column_not_named_id(tmp_path):
    """Airline's flights are keyed by flight_number; the row is there, and the gate has to see it."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    for name in ("data_model.py", "tools.py", "policy.md"):
        (env_dir / name).write_text("", encoding="utf-8")
    (env_dir / "tasks.json").write_text("[]", encoding="utf-8")
    (env_dir / "db.json").write_text(json.dumps({"flights": {"HAT006": {"flight_number": "HAT006", "origin": "SFO"}}}),
                                     encoding="utf-8")
    gate = environment_gate({"files": {}}, files_dir=env_dir, referenced_ids=["HAT006", "HAT999"])
    assert gate.passed is False
    assert gate.failures == ["id HAT999 is referenced by the traces and is not in db.json"]
