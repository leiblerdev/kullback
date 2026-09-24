"""Tests for the artifact gates of kullback.gates.artifacts: ingest to regrade, one ruling record each."""

from __future__ import annotations

import json

from conftest import PTR
from kullback.gates.artifacts import (
    budget_gate,
    environment_gate,
    ingest_gate,
    leak_gate,
    mine_gate,
    policy_gate,
    regrade_gate,
    user_rules_gate,
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


def test_ingest_gate_passes_a_clean_trace_and_fails_a_call_with_neither_result_nor_error():
    clean = a_trace(tool_calls=[{"name": "get_order", "args": {}, "result": {"id": "W1"}}])
    passed = ingest_gate([clean])
    assert isinstance(passed, GateResult)
    assert passed.stage == "ingest"
    assert passed.passed is True
    assert passed.metrics["traces"] == 1
    assert passed.metrics["tool_calls"] == 1
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


def test_ingest_gate_fails_a_trace_with_no_hash_or_a_hash_that_moved_between_two_passes():
    trace = a_trace(tool_calls=[{"name": "t", "args": {}, "result": 1}])
    assert ingest_gate([trace], second_pass={"t1": "h1"}).passed is True
    out = ingest_gate([trace], second_pass={"t1": "h2"})
    assert out.passed is False
    assert any("hash" in f for f in out.failures)
    assert ingest_gate([a_trace(hash="")]).passed is False


# --- mine ---

def a_sig(name="get_order", calls=3, **kw) -> ToolSig:
    base = dict(
        name=name,
        args_schema={"properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        evidence_strength=EvidenceStrength(call_count=calls),
    )
    base.update(kw)
    return ToolSig(**base)


def test_mine_gate_fails_a_thin_signature_unless_it_is_flagged_llm():
    passed = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {"order_id": "W1"}}])
    assert passed.stage == "mine"
    assert passed.passed is True
    out = mine_gate([a_sig(calls=2)])
    assert out.passed is False
    assert any("get_order" in f and "2" in f for f in out.failures)
    assert mine_gate([a_sig(calls=1, source="llm")]).passed is True


def test_mine_gate_believes_a_call_count_of_zero_over_the_evidence_list():
    """A ToolSig that counted no calls is thin, whatever trace ids it lists."""
    out = mine_gate([a_sig(calls=0, evidence=["a", "b", "c"])])
    assert out.passed is False
    assert any("get_order" in f and "0 observed calls" in f for f in out.failures)
    assert mine_gate([a_sig(calls=3, evidence=[])]).passed is True


def test_mine_gate_fails_a_recorded_arg_outside_the_schema_or_a_missing_required_arg():
    out = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {"order_id": "W1", "colour": "red"}}])
    assert out.passed is False
    assert any("colour" in f for f in out.failures)
    missing = mine_gate([a_sig()], calls=[{"name": "get_order", "args": {}}])
    assert missing.passed is False
    assert any("order_id" in f for f in missing.failures)


def test_mine_gate_reports_the_unknown_tool_names_and_does_not_fail_on_them():
    """D164: a name the recording refused on every call, and a name that is no identifier, are
    counted and named. Having no such tool is the Environment being right, not a gate failure."""
    unknown = [{"name": "renew_every_loan", "calls": 4, "requestors": ["assistant"],
                "reason": "refused on every call"},
               {"name": "$LOAN_ACTION", "calls": 1, "requestors": ["assistant"], "reason": "not a tool name"}]
    out = mine_gate([a_sig()], unknown=unknown)
    assert out.passed is True
    assert out.metrics["unknown_tools"] == 2
    assert out.metrics["unknown_tool_names"] == ["renew_every_loan", "$LOAN_ACTION"]
    assert mine_gate([a_sig()]).metrics["unknown_tools"] == 0


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


def test_the_gate_runs_a_predicate_that_calls_the_transcript_helpers_the_compiler_puts_in_scope():
    """Build 10: the gate execed the rule on its own, so `user_confirmed(transcript)`, which the
    compiler's prompt says is already in scope and its sandbox defines, raised NameError and three
    more constraints failed the gate for a reason that was ours and not the model's."""
    asks_first = a_constraint(
        predicate_src=("def check(pre_state, write_call, transcript):\n"
                       "    return write_call.get('name') != 'cancel_order' or user_confirmed(transcript)\n"),
        tests=ConstraintTests(
            pos=[{"pre_state": {}, "write_call": {"name": "cancel_order"},
                  "transcript": [{"role": "assistant", "content": "Shall I cancel order #W1?"},
                                 {"role": "user", "content": "yes please"}]}],
            neg=[{"pre_state": {}, "write_call": {"name": "cancel_order"},
                  "transcript": [{"role": "assistant", "content": "Shall I cancel order #W1?"},
                                 {"role": "user", "content": "no, leave it"}]}],
        ),
    )
    assert policy_gate([asks_first]).passed is True


def test_a_rule_that_defines_no_predicate_of_its_own_is_never_answered_by_a_transcript_helper():
    """The helpers are pasted in beside the rule, so the fallback that looks for the predicate has
    to skip them: a constraint with no function is a defect and not whatever helper was last defined."""
    empty = a_constraint(predicate_src="allowed = True\n")
    out = policy_gate([empty])
    assert out.passed is False
    assert any("defines no function" in f for f in out.failures)


def test_the_gate_and_the_compilers_own_sandbox_rule_the_same_way_on_the_same_constraint():
    """The stage compiles a constraint by running its cases in a subprocess and the gate runs the
    same cases in this process. Two runners for one predicate is where builds 8 and 10 drifted, so
    the two read one source assembly and one argument list."""
    from kullback.builder.policy import run_constraint_tests

    holds = a_constraint(
        predicate_src=("def check(pre_state, write_call, transcript):\n"
                       "    return called_before(transcript, 'get_order_details')\n"),
        tests=ConstraintTests(
            pos=[{"pre_state": {}, "write_call": {"name": "cancel_order"},
                  "transcript": [{"role": "assistant", "content": None,
                                  "tool_calls": [{"name": "get_order_details", "arguments": {}}]}]}],
            neg=[{"pre_state": {}, "write_call": {"name": "cancel_order"}, "transcript": []}],
        ),
    )
    breaks = a_constraint(predicate_src="def check(pre_state, write_call, transcript):\n    return True\n")
    for constraint in (holds, breaks):
        assert policy_gate([constraint]).passed is run_constraint_tests(constraint).passed


def test_policy_gate_passes_the_positive_cases_and_fails_when_a_negative_case_is_allowed():
    passed = policy_gate([a_constraint()])
    assert passed.stage == "compile_policy"
    assert passed.passed is True
    assert passed.metrics["compiled"] == 1
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


# --- environment ---

def a_env(**kw) -> Environment:
    files = {name: "h" for name in ("data_model.py", "tools.py", "db.json", "policy.md", "tasks.json")}
    base = dict(env_id="e1", files=files)
    base.update(kw)
    return Environment(**base)


def test_environment_gate_passes_the_full_file_set_and_fails_a_missing_file():
    passed = environment_gate(a_env(), referenced_ids=["W1"], db_ids=["W1", "W2"])
    assert passed.stage == "build_environment"
    assert passed.passed is True
    env = a_env(files={"db.json": "h"})
    out = environment_gate(env)
    assert out.passed is False
    assert any("tools.py" in f for f in out.failures)


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


def test_user_rules_gate_passes_a_complete_rule_set_and_fails_a_missing_disclosure_rule():
    passed = user_rules_gate(a_rules(), asked_fields=["zip"], trace_refusals=["will not share the card number"])
    assert passed.stage == "build_user_rules"
    assert passed.passed is True
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


# --- verifier suite (D79) and the leak check ---

def test_verifier_gate_needs_every_d79_check():
    checks = {
        "provenance_spans": True, "oracle_passes": True, "empty_fails": True,
        "plausible_wrong_fails": True, "unsolved_state_fails": True, "second_path_passes": True,
        "loophole_probe_fails": True, "leak_check_clean": True, "mutation_flips": True,
        "verifier_specific": True,
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


# --- regrade ---

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

def test_the_budget_gate_passes_a_priced_batch_and_fails_a_candidate_batch_with_unpriced_calls():
    priced = budget_gate({"stages": {"candidate": {"calls": 10, "unpriced_calls": 0}}})
    assert priced.passed is True and priced.metrics["calls"] == 10
    out = budget_gate({"stages": {"candidate": {"calls": 10, "unpriced_calls": 3}}})
    assert out.stage == "budget"
    assert out.passed is False
    assert out.metrics["unpriced_calls"] == 3
    assert "3 of 10" in out.failures[0]


def test_environment_gate_finds_a_referenced_row_under_any_key_and_fails_the_id_the_db_lacks(tmp_path):
    """Airline's flights are keyed by flight_number; the row is there, and the gate has to see it."""
    out = environment_gate(a_env(), referenced_ids=["W1", "W9"], db_ids=["W1"])
    assert out.passed is False
    assert any("W9" in f for f in out.failures)
    ids_dir = tmp_path / "ids"
    ids_dir.mkdir()
    (ids_dir / "db.json").write_text(json.dumps({"orders": [{"order_id": "W1"}]}), encoding="utf-8")
    for name in ("data_model.py", "tools.py", "policy.md"):
        (ids_dir / name).write_text("x", encoding="utf-8")
    (ids_dir / "tasks.json").write_text("[]", encoding="utf-8")
    assert environment_gate(a_env(), files_dir=ids_dir, referenced_ids=["W1"]).passed is True
    assert environment_gate(a_env(), files_dir=ids_dir, referenced_ids=["W7"]).passed is False
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
