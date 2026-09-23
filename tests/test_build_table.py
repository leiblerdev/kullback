"""`scripts/build_table.py`: the D139 table over a workdir built offline. No model, no live build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_table as B  # noqa: E402
from env_fidelity import CAUSE_OWNER  # noqa: E402

# What the offline build over tests/fixtures/tau2_retail_small.json leaves behind: three Tasks, none
# covered, two of the three Traces confirming their Reference. A number here that moves is a change
# in the build, not in the table, and the table is what says so.
TASKS = 3
CONFIRMED = "2 of 3"
WRITES = "3 of 3"
READS = "1 of 19"
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
def test_the_peak_context_fill_is_the_largest_a_round_recorded_over_the_rounds(tmp_path):
    workdir = tmp_path / "filled"
    workdir.mkdir()
    (workdir / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"context_fill": {"builder": 0.41, "examiner": 0.12},
                                "turns": {"builder": 3, "examiner": 1, "total": 4}}},
        {"round": 2, "counts": {"context_fill": {"builder": 0.62, "examiner": 0.08},
                                "turns": {"builder": 2, "examiner": 1, "total": 3}}}]), encoding="utf-8")
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(B.render(B.Build(workdir)), "")}
    assert values["Peak context fill, Builder"] == "62.0% of the window at a turn end, over 5 turns in 2 rounds"
    assert values["Peak context fill, Examiner"] == "12.0% of the window at a turn end, over 2 turns in 2 rounds"
def test_the_build_duration_names_the_record_when_no_round_kept_a_clock(tmp_path):
    workdir = tmp_path / "clockless"
    workdir.mkdir()
    (workdir / "rounds.json").write_text(json.dumps([{"round": 1, "counts": {"fidelity": 1}}]), encoding="utf-8")
    value = {cells(row)[0]: cells(row)[1] for row in rows_of(B.render(B.Build(workdir)), "")}["Build duration"]
    assert value == ("n/a (needs rounds.json counts.started_at and counts.ended_at; pipeline/state.json "
                     "records the stage statuses and no clock)")
def test_the_mechanic_row_counts_the_tool_calls_in_the_builders_session_by_verb(tmp_path):
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


# --- the gates ----------------------------------------------------------------
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
    the one env_fidelity would give the whole answer, not the word for an unreadable preview."""
    cut = {"verdict": "differs", "ours": '{"a": "' + "x" * 150 + "...", "recorded": '{"a": "y"}',
           "difference": {"ours": '{"value": 3}', "ours_truncated": False, "ours_type": "dict",
                          "theirs": "3", "theirs_truncated": False, "theirs_type": "int",
                          "type_mismatch": True}}
    assert B.check_cause(cut) == "result_shape"
    refused = {"verdict": "ours_refused", "ours": '{"payload": "NameError: ' + "y" * 140 + '...',
               "recorded": '"27.4"',
               "difference": {"ours_errored": True, "ours_error": "NameError: name 'decimal' is not defined"}}
    assert B.check_cause(refused) == "missing_import"


def test_an_answer_too_long_even_for_the_difference_is_named_off_its_keys_and_types():
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


def test_the_round_row_says_which_of_the_gate_counts_moved_what_it_spent_and_the_turns_it_took(tmp_path):
    workdir = tmp_path / "rounds"
    workdir.mkdir()
    (workdir / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"fidelity": 1, "trusted": 0, "refused_count": 0, "assisted_runs": 0,
                                "probes_passing": 0, "spend": {"builder": 0.5, "examiner": 0.25, "total": 0.75, "cache_saved": 0.3},
                                "turns": {"builder": 6, "examiner": 2, "total": 8},
                                "artifacts": "aaaaaaaaaaaa", "artifacts_changed": []},
         "exit": None},
        {"round": 2, "counts": {"fidelity": 1, "trusted": 2, "refused_count": 0, "assisted_runs": 0,
                                "probes_passing": 0, "spend": {"builder": 0.1, "examiner": 0.0, "total": 0.1},
                                "turns": {"builder": 3, "examiner": 1, "total": 4},
                                "artifacts": "bbbbbbbbbbbb", "artifacts_changed": ["bodies", "intents"]},
         "exit": "stalled"}]), encoding="utf-8")
    first, second = [cells(row) for row in rows_of(B.render(B.Build(workdir)), "Per round")]
    assert first[2] == "first round" and first[6] == "the round did not exit"
    assert first[4] == "builder $0.50, examiner $0.25, total $0.75, cache saved $0.30"
    assert first[5] == "builder 6, examiner 2, total 8"
    assert second[2] == "trusted 2 from 0" and second[6] == "stalled"
    assert second[4] == "builder $0.10, examiner $0.00, total $0.10"
    assert second[5] == "builder 3, examiner 1, total 4"


def test_the_round_row_names_the_artifacts_the_round_rewrote(tmp_path):
    """The artifacts column is the other half of whether a round did anything: a round that rewrote
    a tool body and an Intent says so even where no gate count moved (`rounds.round_moved`)."""
    workdir = tmp_path / "artifacts"
    workdir.mkdir()
    (workdir / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"fidelity": 1, "artifacts_changed": []}, "exit": None},
        {"round": 2, "counts": {"fidelity": 1, "artifacts_changed": ["bodies", "intents"],
                                "moved": True}, "exit": None},
        {"round": 3, "counts": {"fidelity": 1, "artifacts_changed": [], "moved": False},
         "exit": "stalled"}]), encoding="utf-8")
    rows = [cells(row) for row in rows_of(B.render(B.Build(workdir)), "Per round")]
    assert [row[3] for row in rows] == ["first round", "bodies, intents", "none"]
    assert rows[1][2] == "none", "no gate count moved in the round that rewrote the two artifacts"


def test_a_round_that_kept_no_turn_count_names_the_record_it_would_need(tmp_path):
    workdir = tmp_path / "turnless"
    (workdir / "builder").mkdir(parents=True)
    (workdir / "builder" / "session.jsonl").write_text(json.dumps(
        {"type": "message", "id": "e1", "message": {"role": "assistant", "content": "done"}}) + "\n",
        encoding="utf-8")
    (workdir / "rounds.json").write_text(json.dumps(
        [{"round": 1, "counts": {"fidelity": 1}},
         {"round": 2, "counts": {"fidelity": 1}}]), encoding="utf-8")
    first, second = [cells(row) for row in rows_of(B.render(B.Build(workdir)), "Per round")]
    assert first[4] == "n/a (needs rounds.json counts.spend)"
    assert first[5] == "n/a (needs rounds.json counts.turns)"
    assert second[3] == "n/a (needs rounds.json counts.artifacts_changed)", \
        "a build from before the fingerprint names the record it would need"
def _repaired(workdir: Path) -> Path:
    """A workdir with a repair in round 2, one in round 3, and the rulings each round left."""
    (workdir / "repairs").mkdir(parents=True)
    (workdir / "repairs" / "repair_recompile.jsonl").write_text(
        json.dumps({"verb": "repair_recompile", "target": "calculate", "at": 1.0, "round": 2}) + "\n",
        encoding="utf-8")
    (workdir / "repairs" / "repair_escalate.jsonl").write_text(
        json.dumps({"verb": "repair_escalate", "target": "task_a", "at": 2.0, "round": 3}) + "\n",
        encoding="utf-8")
    (workdir / "gates_by_round.json").write_text(json.dumps([
        {"round": 1, "rulings": [{"stage": "replay_fidelity", "pass": False}, {"stage": "intent", "pass": False}]},
        {"round": 2, "rulings": [{"stage": "replay_fidelity", "pass": True}, {"stage": "intent", "pass": False}]},
        {"round": 3, "rulings": [{"stage": "replay_fidelity", "pass": True}, {"stage": "intent", "pass": False}]},
    ]), encoding="utf-8")
    return workdir


def test_a_repair_is_listed_with_the_rulings_before_and_after_its_own_round(tmp_path):
    text = B.render(B.Build(_repaired(tmp_path / "repaired")))
    escalate, recompile = [cells(row) for row in rows_of(text, "Per repair verb")]
    assert recompile[:3] == ["`repair_recompile`", "`calculate`", "2"]
    assert recompile[3] == "red: intent, replay_fidelity" and recompile[4] == "red: intent"
    assert recompile[5] == "replay_fidelity"
    assert escalate[3] == "red: intent" and escalate[4] == "red: intent" and escalate[5] == "none"


def test_the_headline_counts_the_repairs_whose_round_turned_a_red_gate_green(tmp_path):
    values = {cells(row)[0]: cells(row)[1]
              for row in rows_of(B.render(B.Build(_repaired(tmp_path / "repaired"))), "")}
    assert values["Repairs requested"] == "2 requested: repair_escalate 1, repair_recompile 1"
    assert values["Repairs that turned a red gate green"] == \
        "1 of 2: repair_recompile on calculate (round 2, replay_fidelity went red to green)"
    assert values["Repairs that did not"] == "1 of 2: repair_escalate on task_a (round 3)"


def test_a_repair_request_with_no_round_names_the_record_that_would_place_it(tmp_path):
    workdir = tmp_path / "unplaced"
    (workdir / "repairs").mkdir(parents=True)
    (workdir / "repairs" / "repair_recompile.jsonl").write_text(
        json.dumps({"verb": "repair_recompile", "target": "calculate", "at": 1.0}) + "\n", encoding="utf-8")
    text = B.render(B.Build(workdir))
    row = cells(rows_of(text, "Per repair verb")[0])
    assert row[0] == "`repair_recompile`" and row[1] == "`calculate`"
    assert row[2] == "n/a (needs repairs/*.jsonl round)"
    assert row[3].startswith("n/a (needs") and row[4] == row[3]
    values = {cells(line)[0]: cells(line)[1] for line in rows_of(text, "")}
    assert values["Repairs requested"] == "1 requested: repair_recompile 1"
    assert values["Repairs that turned a red gate green"] == \
        "0 of 1; 1 not placed: n/a (needs " + B.REPAIR_GAP + ")"


def test_a_repair_of_round_one_has_no_earlier_ruling_to_be_read_against(tmp_path):
    """A repair acts inside its round, and round 1 has no round before it, so nothing in the records
    says what the gate said before it: that is a gap, not a repair that moved nothing."""
    workdir = _repaired(tmp_path / "first-round")
    (workdir / "repairs" / "repair_grow.jsonl").write_text(
        json.dumps({"verb": "repair_grow", "target": "orders", "at": 0.5, "round": 1}) + "\n",
        encoding="utf-8")
    text = B.render(B.Build(workdir))
    grow = [cells(row) for row in rows_of(text, "Per repair verb") if cells(row)[0] == "`repair_grow`"][0]
    assert grow[2] == "1" and grow[3] == "no round ran before this one"
    assert grow[4] == "red: intent, replay_fidelity" and grow[5] == "n/a (needs gates_by_round.json)"
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(text, "")}
    assert values["Repairs that turned a red gate green"].endswith(
        "; 1 not placed: n/a (needs " + B.REPAIR_GAP + ")")
    assert values["Repairs that did not"].startswith("1 of 3: repair_escalate on task_a (round 3)")
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
