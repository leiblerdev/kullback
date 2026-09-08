"""Tests for kullback.gates.scorecard: the D62 scorecard as D80 and D96 leave it, and its one gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.gates.scorecard import FROZEN_TASKS_NAME, freeze_tasks, scorecard, task_coverage
from kullback.runner.gate_support import MISS_REASONS
from kullback.runner.records import Task, as_dict


def a_run(run_id="r1", task_id="t1", seed=1, stopped=True, **kw) -> dict:
    events = [{"idx": 0, "type": "tool_call", "payload": {"name": "x"}}]
    if stopped:
        events.append({"idx": 1, "type": "stop", "payload": {"reason": "done"}})
    run = {"run_id": run_id, "task_id": task_id, "seed": seed, "events": events}
    run.update(kw)
    return run


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


def test_scorecard_coverage_needs_a_status_entry_at_all(build_dir: Path):
    """No entry means nothing confirmed the Reference and nothing ran the D79 suite (D96)."""
    (build_dir / "task_status.json").unlink()
    coverage = scorecard(build_dir)["task_coverage"]
    assert coverage["tasks_covered"] == 0
    assert coverage["tasks"] == 0.0
    reasons = {u["task_id"]: u["reason"] for u in coverage["uncovered"]}
    assert "no Reference confirmation" in reasons["t1"]

    (build_dir / "task_status.json").write_text(
        json.dumps({"t1": {"reference_confirmed": True, "verifier_passed": True},
                    "t2": {"reference_confirmed": True}}), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert coverage["tasks_covered"] == 1
    assert "no D79 result" in coverage["uncovered"][0]["reason"]


def test_scorecard_coverage_says_which_d96_reasons_nothing_measures(build_dir: Path):
    """overlay_miss and reconstructed are read here; only the first has a producer today."""
    coverage = scorecard(build_dir)["task_coverage"]
    assert coverage["reasons_not_measured"] == ["reconstructed", "truncated"]


def test_task_coverage_counts_a_task_with_no_runs_as_uncovered():
    coverage = task_coverage([Task(id="empty", run_ids=[])], [],
                             {"empty": {"reference_confirmed": True, "verifier_passed": True}})
    assert coverage["tasks_covered"] == 0
    assert coverage["uncovered"][0]["reason"] == "Task has no Runs"


def test_task_coverage_reads_the_llm_route_as_assisted_even_with_the_flag_lost():
    """D49 defence in depth: the events show the stand-in whether or not the flag survived."""
    status = {"t1": {"reference_confirmed": True, "verifier_passed": True}}
    run = a_run("r1", "t1")
    run["events"][0]["route"] = "llm"
    coverage = task_coverage([Task(id="t1", run_ids=["r1"])], [run], status)
    assert coverage["tasks_covered"] == 0
    assert "assisted" in coverage["uncovered"][0]["reason"]

    counted = a_run("r2", "t1", route_counts={"code": 1, "llm": 2})
    coverage = task_coverage([Task(id="t1", run_ids=["r2"])], [counted], status)
    assert coverage["tasks_covered"] == 0
    assert "assisted" in coverage["uncovered"][0]["reason"]


def test_the_frozen_task_list_is_written_once_and_holds_the_denominator(build_dir: Path):
    """D96: a Task split after the freeze cannot raise coverage."""
    tasks = json.loads((build_dir / "tasks.json").read_text())
    assert freeze_tasks(build_dir, tasks) == ["t1", "t2"]
    before = scorecard(build_dir)["task_coverage"]
    assert (before["tasks_total"], before["tasks_covered"]) == (2, 2)

    split = [as_dict(Task(id="t1a", run_ids=["r1"])), as_dict(Task(id="t1b", run_ids=["r2"])),
             as_dict(Task(id="t2", run_ids=["r3"]))]
    (build_dir / "tasks.json").write_text(json.dumps(split), encoding="utf-8")
    assert freeze_tasks(build_dir, split) == ["t1", "t2"], "the frozen list is not rewritten"
    after = scorecard(build_dir)["task_coverage"]
    assert after["tasks_total"] == 2
    assert after["tasks_covered"] == 1, "t1 left the build, so it is uncovered, not gone"
    assert after["added_later"] == ["t1a", "t1b"]
    reasons = {u["task_id"]: u["reason"] for u in after["uncovered"]}
    assert "frozen list" in reasons["t1"]
    assert json.loads((build_dir / FROZEN_TASKS_NAME).read_text())["task_ids"] == ["t1", "t2"]


def test_scorecard_counts_a_reference_run_we_never_verdicted_as_a_miss(build_dir: Path):
    """D80: a Run we failed to verdict is an unexplained miss, not a smaller denominator."""
    verdicts = json.loads((build_dir / "verdicts.json").read_text())
    (build_dir / "verdicts.json").write_text(json.dumps(verdicts[:1]), encoding="utf-8")
    card = scorecard(build_dir)
    agreement = card["verdict_agreement"]
    assert agreement["total"] == 3
    assert agreement["matched"] == 1
    assert agreement["raw"] == 0.3333
    assert agreement["unexplained"] == 1, "r2 carries reference_bug, r3 carries nothing"
    assert card["gate"]["pass"] is False
    assert any("r3" in f and "we produced none" in f for f in card["gate"]["failures"])


def test_a_task_set_aside_as_not_gradeable_leaves_the_agreement_denominator(build_dir: Path):
    """D93: a Task with a disputed Reference is listed, not counted against us."""
    verdicts = json.loads((build_dir / "verdicts.json").read_text())
    (build_dir / "verdicts.json").write_text(json.dumps(verdicts[:2]), encoding="utf-8")
    (build_dir / "not_gradeable.json").write_text(json.dumps(["r3"]), encoding="utf-8")
    card = scorecard(build_dir)
    assert card["verdict_agreement"]["total"] == 2
    assert card["gate"]["pass"] is True


def test_scorecard_counts_a_run_that_was_never_replayed_as_uncovered(build_dir: Path):
    runs = json.loads((build_dir / "runs.json").read_text())
    (build_dir / "runs.json").write_text(json.dumps(runs[1:]), encoding="utf-8")
    coverage = scorecard(build_dir)["task_coverage"]
    assert "r1" in coverage["uncovered"][0]["reason"]


def test_scorecard_user_fact_consistency_reports_raw_and_explained_rates_with_the_miss_by_reason(build_dir: Path):
    facts = scorecard(build_dir)["user_fact_consistency"]
    assert facts["total"] == 2
    assert facts["matched"] == 1
    assert facts["raw"] == 0.5
    assert facts["explained"] == 1.0
    assert facts["by_reason"] == {"ambiguous": 1}


def test_scorecard_verdict_agreement_explains_a_miss_by_its_reason_and_lists_the_miss(build_dir: Path):
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
    agreement = card["verdict_agreement"]
    assert agreement["unexplained"] == 1, "a reason outside MISS_REASONS explains nothing"
    assert agreement["by_reason"] == {}
    assert agreement["explained"] == agreement["raw"]
    assert "because" not in json.dumps(agreement["by_reason"])
    assert set(MISS_REASONS) == {"our_bug", "reference_bug", "ambiguous"}


def test_scorecard_passes_policy_coverage_through(build_dir: Path):
    """R22 item 10 keeps its own line: read as written, and never folded into the gate."""
    card = scorecard(build_dir)
    assert card["policy_coverage"] == {"rules": 40, "compiled": 6, "residual": 2}
    assert "policy" not in json.dumps(card["gate"]["metrics"])
    (build_dir / "policy.json").unlink()
    assert scorecard(build_dir)["policy_coverage"] == {}
    assert scorecard(build_dir)["gate"]["pass"] is True


def test_scorecard_on_an_empty_build_dir_reports_nothing_rather_than_a_hundred_percent(tmp_path: Path):
    card = scorecard(tmp_path)
    assert card["tool_fidelity"]["success"]["raw"] is None
    assert card["task_coverage"]["tasks"] is None
    assert card["verdict_agreement"]["raw"] is None
    assert card["gate"]["pass"] is True


def test_scorecard_is_json_serializable(build_dir: Path):
    card = scorecard(build_dir)
    assert json.loads(json.dumps(card)) == card


def test_the_scorecard_gate_is_never_green_over_nothing(build_dir: Path):
    status = {t: {"reference_confirmed": False, "verifier_passed": False, "reason": "x"} for t in ("t1", "t2")}
    (build_dir / "task_status.json").write_text(json.dumps(status), encoding="utf-8")
    card = scorecard(build_dir)
    assert card["task_coverage"]["tasks_covered"] == 0
    assert card["gate"]["pass"] is False
    assert any("no Task is gradeable" in f for f in card["gate"]["failures"])


