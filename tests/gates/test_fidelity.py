"""Tests for kullback.gates.fidelity: the per-tool bar (D80), the per-Task Reference ruling (D108) and Gate A."""

from __future__ import annotations

from kullback.gates.fidelity import (
    oracle_replay_gate,
    reference_replay_gate,
    replay_fidelity_gate,
    summarize,
    unconfirmed_reason,
)
from runner.replay_fixtures import Toolkit, do_replay

# --- the per-tool bar (compile_tools gate 5, D80) ---

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


# --- gate A, oracle replay ---

def test_gate_a_fails_a_held_out_write_mismatch_and_names_the_run():
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


def test_the_fidelity_gates_compare_under_the_customers_rules_and_equivalence_table_like_the_verdict():
    """A gate that compared under the module defaults could differ from the Verdict (D39, D84): Gate A
    reads a semantic mismatch the customer's table calls cosmetic as a match, and the per-tool gate
    the scorecard reads compares numbers under the customer's canon rules."""
    from kullback.runner.canon import CanonRules, EquivalenceTable, canon_value, put

    replays = [{"run_id": "r1", "held_out": False, "writes": [],
                "semantic_reads": [{"column": "orders.note", "expected": "blue shirt",
                                    "actual": "navy shirt"}]}]
    table = EquivalenceTable()
    put(table, "orders.note", canon_value("blue shirt"), canon_value("navy shirt"), True,
        classified_by="human")
    assert oracle_replay_gate(replays).passed is False
    assert oracle_replay_gate(replays, equivalence=table).passed is True
    calls = [{"tool": "get_order", "expected": {"total": 25.4}, "actual": {"total": 25.0},
              "held_out": False}]
    assert replay_fidelity_gate(calls).passed is False
    assert replay_fidelity_gate(calls, canon_rules=CanonRules(number_precision=0)).passed is True


# --- the per-Task ruling over the replays (D108) ---

def _replay(confirmed: bool, *reasons: str, writes: int = 1, matched: int = 1) -> dict:
    return {"confirmed": confirmed, "reasons": list(reasons),
            "counts": {"writes": writes, "writes_matched": matched, "reads": 2, "reads_semantic": 0,
                       "reads_cosmetic": 1, "unmade": 0}}


def test_unconfirmed_reason_is_the_most_common_first_reason():
    a = _replay(False, "cancel_order write: differs", "get_order read: differs")
    b = _replay(False, "cancel_order write: differs")
    c = _replay(False, "replay crashed: KeyError")
    assert unconfirmed_reason({"tr1": a, "tr2": b, "tr3": c}) == "cancel_order write: differs"
    assert unconfirmed_reason({}) == "no Trace of the Task was replayed"
    assert unconfirmed_reason({"tr1": _replay(False)}) == "no Trace of the Task was replayed"


def test_reference_replay_gate_fails_only_the_task_no_trace_confirms():
    """Section 6: a Task none of whose Traces reach their End state is rejected for that Task, with
    the reason its replays agree on; a Task one Trace confirms is fine whatever the others did. No
    Task replayed is no Task rejected; the stage's own gate says how many Tasks there were."""
    replays = {
        "t1": {"tr1": _replay(True), "tr2": _replay(False, "get_order read: differs")},
        "t2": {"tr1": _replay(False, "cancel_order write: differs", matched=0)},
        "t3": {},
    }
    out = reference_replay_gate(replays)
    assert out.stage == "replay_reference"
    assert out.passed is False
    assert out.failures == ["task t2: cancel_order write: differs",
                            "task t3: no Trace of the Task was replayed"]
    assert out.metrics["tasks"] == 3 and out.metrics["tasks_confirmed"] == 1
    assert out.metrics == summarize(replays)
    assert reference_replay_gate({"t1": {"tr1": _replay(True)}}).passed is True
    empty = reference_replay_gate({})
    assert empty.passed is True and empty.metrics["tasks"] == 0 and empty.metrics["traces"] == 0


# --- summarize over real replays (D108) ---

def test_summarize_counts_traces_tasks_and_calls_and_names_the_common_miss(tmp_path):
    good = do_replay(tmp_path / "a").as_dict()
    bad = do_replay(tmp_path / "b", type("Misspelt", (Toolkit,), {"cancelled": "canceled"})).as_dict()
    summary = summarize({"t1": {"tr1": good}, "t2": {"tr1": bad, "tr2": bad}})
    assert summary["traces"] == 3 and summary["confirmed"] == 1
    assert summary["tasks"] == 2 and summary["tasks_confirmed"] == 1
    assert summary["writes"] == 3 and summary["writes_matched"] == 1
    assert unconfirmed_reason({"tr1": bad, "tr2": bad}) == "cancel_order write: differs (value)"
    assert unconfirmed_reason({}) == "no Trace of the Task was replayed"
    built_good, built_bad = _replay(True), _replay(False, "cancel_order write: differs", matched=0)
    built = summarize({"t1": {"tr1": built_good}, "t2": {"tr1": built_bad, "tr2": built_bad}})
    assert built["traces"] == 3 and built["confirmed"] == 1
    assert built["tasks"] == 2 and built["tasks_confirmed"] == 1
    assert built["writes"] == 3 and built["writes_matched"] == 1
    assert built["reads"] == 6 and built["reads_cosmetic"] == 3
