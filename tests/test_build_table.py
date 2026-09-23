"""`scripts/build_table.py`: the D139 table over a workdir built offline. No model, no live build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_table as B  # noqa: E402
from env_fidelity import CAUSE_OWNER  # noqa: E402


def rows_of(text: str, section: str) -> list[str]:
    """The table rows under one heading of the report, without its header and rule lines."""
    body = text.split(f"# {section}\n", 1)[1] if section else text
    body = body.split("\n#", 1)[0]
    return [line for line in body.splitlines() if line.startswith("|") and not line.startswith("|---")][1:]


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


# --- the headline -------------------------------------------------------------

def test_every_row_a_workdir_cannot_answer_names_the_record_it_would_need(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    text = B.render(B.Build(empty))
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(text, "")}
    assert values["Tasks covered (D96)"] == "n/a (needs scorecard.json task_coverage)"
    assert values["Dollars"] == "n/a (needs budget.json total)"
    assert "replay_reference" in values["Writes replaying exactly"]
    assert "`gates.json`" in text and "`scorecard.json`" in text


def test_the_mechanic_row_counts_the_builders_tool_calls_by_name(tmp_path):
    workdir = tmp_path / "driven"
    (workdir / "builder").mkdir(parents=True)
    session = [{"type": "session_info", "id": "e1"},
               {"type": "message", "id": "e2", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "name": "build", "arguments": {}},
                               {"id": "c2", "name": "repair_recompile", "arguments": {}}],
                "usage": {"input": 900, "cache_read": 100}}},
               {"type": "message", "id": "e3", "message": {"role": "assistant", "content": "done",
                "tool_calls": [{"id": "c3", "name": "build", "arguments": {}}],
                "usage": {"input": 300, "cache_read": 0}}}]
    (workdir / "builder" / "session.jsonl").write_text(
        "\n".join(json.dumps(row) for row in session) + "\n", encoding="utf-8")
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(B.render(B.Build(workdir)), "")}
    assert values["What the mechanic called"] == "3 over 2 model turns: build 2, repair_recompile 1"
    assert "the largest input a turn recorded is 1,000 tokens" in values["Peak context fill, Builder"]


# --- per tool -----------------------------------------------------------------
def test_a_cause_is_one_env_fidelity_already_names_or_the_word_for_a_preview_it_cannot_read():
    read = {"verdict": "ours_refused", "ours": '{"payload": "NameError: name \'decimal\' is not defined"}',
            "recorded": '"27.4"'}
    assert B.check_cause(read) == "missing_import"
    assert B.check_cause({"verdict": "same", "ours": "1", "recorded": "1"}) == "none"
    assert B.check_cause({"verdict": "both_refused", "ours": '{"payload": "Error: no"}',
                          "recorded": '{"payload": "no"}'}) == "none"
    assert B.check_cause({"verdict": "differs", "ours": '{"value": 3}', "recorded": "3"}) == "result_shape"
    cut = {"verdict": "differs", "ours": '{"a": "' + "x" * 150 + "...", "recorded": '{"a": "y"}'}
    assert B.check_cause(cut) == B.UNREADABLE, "a build recorded before the difference record"
    assert B.UNREADABLE not in CAUSE_OWNER, "the cause a preview hides is this file's word, not env_fidelity's"


def test_a_cut_preview_is_read_off_the_difference_the_check_recorded_instead():
    """runner/replay.py keeps the two answers of a differing call, so the cause of a long answer is
    the one env_fidelity would give the whole answer, not the word for an unreadable preview. An
    answer too long even for the difference is named off its keys and types."""
    cut = {"verdict": "differs", "ours": '{"a": "' + "x" * 150 + "...", "recorded": '{"a": "y"}',
           "difference": {"ours": '{"value": 3}', "ours_truncated": False, "ours_type": "dict",
                          "theirs": "3", "theirs_truncated": False, "theirs_type": "int",
                          "type_mismatch": True}}
    assert B.check_cause(cut) == "result_shape"
    refused = {"verdict": "ours_refused", "ours": '{"payload": "NameError: ' + "y" * 140 + '...',
               "recorded": '"27.4"',
               "difference": {"ours_errored": True, "ours_error": "NameError: name 'decimal' is not defined"}}
    assert B.check_cause(refused) == "missing_import"
    both_cut = {"verdict": "differs", "ours": '{"a": "' + "x" * 150 + "...", "recorded": '{"a": "y"}',
                "difference": {"ours_truncated": True, "theirs_truncated": True, "type_mismatch": False,
                               "keys_only_ours": ["extra"], "keys_only_theirs": [], "keys_changed": []}}
    assert B.check_cause(both_cut) == "result_shape"
    both_cut["difference"] = {"ours_truncated": True, "theirs_truncated": True, "type_mismatch": False,
                              "keys_only_ours": [], "keys_only_theirs": [], "keys_changed": ["status"]}
    assert B.check_cause(both_cut) == "value"
    both_cut["difference"] = {"ours_truncated": True, "theirs_truncated": True, "type_mismatch": False,
                              "keys_only_ours": [], "keys_only_theirs": [], "keys_changed": []}
    assert B.check_cause(both_cut) == B.UNREADABLE, "nothing in the record names this one"


def test_a_refused_differently_row_and_a_body_fault_row_each_get_their_cause_and_their_column(tmp_path):
    """F56, F57: neither is agreement, so each is named like the four other misses and counted in a column."""
    workdir = tmp_path / "refusals"
    workdir.mkdir()
    fault = {"tool": "refund", "fault": "NameError", "message": "NameError: name 'decimal' is not defined",
             "line": 4, "code": "return decimal.Decimal(total)"}
    checks = [
        {"tool": "cancel", "kind": "write", "verdict": "refused_differently",
         "ours": '{"payload": "ValueError: Payment amount does not add up"}',
         "recorded": '{"payload": "Error: Order is not pending"}',
         "difference": {"ours_error": "ValueError: Payment amount does not add up",
                        "theirs_error": "Error: Order is not pending"}},
        {"tool": "refund", "kind": "write", "verdict": "body_fault",
         "ours": json.dumps({"class_": "body_fault", "payload": fault}),
         "recorded": '{"payload": "Error: Order is not pending"}',
         "difference": {"ours_error": str(fault), "theirs_error": "Error: Order is not pending"}},
    ]
    assert [B.check_cause(check) for check in checks] == ["error_message", "missing_import"]
    (workdir / "replays.json").write_text(json.dumps({"t1": {"tr1": {"checks": checks}}}), encoding="utf-8")
    text = B.render(B.Build(workdir))
    section = text.split("# Per tool, over the replayed calls\n", 1)[1]
    header = cells(next(line for line in section.splitlines() if line.startswith("|")))
    rows = {cells(row)[0]: dict(zip(header, cells(row), strict=True)) for row in rows_of(text, "Per tool, over the replayed calls")}
    assert rows["`cancel`"]["refused differently"] == "1" and rows["`cancel`"]["body faults"] == "0"
    assert rows["`cancel`"]["agrees"] == "0" and rows["`cancel`"]["top cause"] == "error_message"
    assert rows["`refund`"]["body faults"] == "1" and rows["`refund`"]["refused differently"] == "0"
    assert rows["`refund`"]["agrees"] == "0" and rows["`refund`"]["top cause"] == "missing_import"


def test_a_task_whose_verifier_the_suite_refused_carries_the_checks_that_failed_and_its_atoms(tmp_path):
    workdir = tmp_path / "graded"
    (workdir / "verifiers").mkdir(parents=True)
    (workdir / "tasks_frozen.json").write_text(json.dumps({"task_ids": ["task_a"]}), encoding="utf-8")
    (workdir / "task_status.json").write_text(json.dumps({"task_a": {
        "reference_confirmed": True, "verifier_passed": False, "not_run": ["verifier_alt_path"],
        "checks": {"oracle_passes": True, "empty_fails": False, "mutation_flips": False}}}), encoding="utf-8")
    (workdir / "verifiers" / "task_a.json").write_text(json.dumps({"task_id": "task_a", "atoms": [
        {"id": "w0", "kind": "required", "description": "the Run makes at most 0 write calls"},
        {"id": "w1", "kind": "allowed", "description": "amount is 50"}]}), encoding="utf-8")
    row = cells(rows_of(B.render(B.Build(workdir)), "Per Task")[0])
    assert row[1] == "yes" and row[2] == "no"
    assert row[3] == "empty_fails, mutation_flips" and row[4] == "verifier_alt_path"
    assert row[5] == "1 required of 2: the Run makes at most 0 write calls"


# --- per failing Run ----------------------------------------------------------

@pytest.mark.parametrize("lines, rule, side", [
    ([{"type": "error", "payload": {"class": "env_error",
       "message": "ProviderError: openai/x: HTTP 400: Invalid 'messages': empty array."}}],
     "provider_error", "ours"),
    ([{"type": "tool_result", "payload": {"error": {"class": "business_error",
       "payload": "NameError: name 'decimal' is not defined"}}},
      {"run_id": "reroll-task_a-0", "termination_reason": "user_stop"}],
     "body_exception", "ours"),
    ([{"type": "tool_result", "payload": {"error": {"class": "not_found_entity",
       "payload": "ValueError: 'User not found'"}}},
      {"run_id": "reroll-task_a-0", "termination_reason": "max_turns"}],
     "env_refused_the_candidate", "the model's"),
    ([{"type": "user_turn", "payload": {"tags": ["fact_unavailable"]}},
      {"run_id": "reroll-task_a-0", "termination_reason": "max_turns"}],
     "simulated_user_had_no_answer", "ours"),
])
def test_the_first_rule_that_matches_a_runs_own_record_decides_whose_error_it_was(tmp_path, lines, rule, side):
    workdir = tmp_path / rule
    (workdir / "runs" / "task_a").mkdir(parents=True)
    (workdir / "runs" / "task_a" / "reroll-task_a-0.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    row = cells(rows_of(B.render(B.Build(workdir)), "Per failing Run: the model's error or ours")[-1])
    assert row[0] == "`reroll-task_a-0`" and row[1] == "reroll"
    assert row[2] == side and row[3] == rule
    assert row[5] == "`runs/task_a/reroll-task_a-0.jsonl`"


def test_a_run_that_ended_the_way_the_recording_did_is_not_a_failing_run(tmp_path):
    workdir = tmp_path / "finished"
    (workdir / "runs" / "task_a").mkdir(parents=True)
    (workdir / "runs" / "task_a" / "reroll-task_a-0.jsonl").write_text(
        json.dumps({"run_id": "reroll-task_a-0", "termination_reason": "user_stop"}) + "\n", encoding="utf-8")
    assert "No Run failed" in B.render(B.Build(workdir)).split("## Per failing Run", 1)[1]


# --- the shape of the output --------------------------------------------------

def test_a_workdir_that_is_not_a_directory_is_one_clear_message(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        B.main([str(tmp_path / "no-such-build")])
    assert "is not a directory" in str(stopped.value)
