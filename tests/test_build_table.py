"""`scripts/build_table.py`: the D139 table over a workdir built offline. No model, no live build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from builder.test_build import Bodies
from kullback import rounds

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


@pytest.fixture(scope="module")
def built(tmp_path_factory, request) -> Path:
    """One whole offline build, driven by code, the way tests/test_rounds.py drives one."""
    workdir = tmp_path_factory.mktemp("table")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    rounds.run_rounds(workdir, model=Bodies(), files=[fixture], max_attempts=0)
    return workdir


@pytest.fixture(scope="module")
def printed(built: Path) -> str:
    return B.render(B.Build(built))


def rows_of(text: str, section: str) -> list[str]:
    """The table rows under one heading of the report, without its header and rule lines."""
    body = text.split(f"# {section}\n", 1)[1] if section else text
    body = body.split("\n#", 1)[0]
    return [line for line in body.splitlines() if line.startswith("|") and not line.startswith("|---")][1:]


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


# --- the headline -------------------------------------------------------------

def test_the_headline_is_the_tasks_covered_out_of_the_frozen_task_list(printed: str, built: Path):
    frozen = json.loads((built / "tasks_frozen.json").read_text(encoding="utf-8"))["task_ids"]
    assert len(frozen) == TASKS
    assert f"**Tasks covered: 0 of {TASKS} frozen Tasks (0.0%), 0 of {TASKS} Runs (0.0%)**" in printed


def test_the_headline_rows_are_the_same_ones_every_build_in_the_same_order(printed: str):
    names = [cells(row)[0] for row in rows_of(printed, "")]
    assert names == ["Tasks covered (D96)", "Traces confirming their Reference",
                     "Writes replaying exactly", "Reads differing in substance", "Gates green",
                     "What the mechanic called", "Repairs requested",
                     "Repairs that turned a red gate green", "Repairs that did not", "Dollars",
                     "Model calls", "Model call wall time", "Build duration",
                     "Peak context fill, Builder", "Peak context fill, Examiner"]


def test_the_replay_rows_say_what_the_stages_own_ruling_says(printed: str, built: Path):
    ruling = [row for row in json.loads((built / "gates.json").read_text(encoding="utf-8"))
              if row["stage"] == "replay_reference"][-1]
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(printed, "")}
    assert values["Traces confirming their Reference"].startswith(CONFIRMED)
    assert values["Writes replaying exactly"].startswith(WRITES)
    assert values["Reads differing in substance"].startswith(READS)
    assert ruling["metrics"]["confirmed"] == 2 and ruling["metrics"]["traces"] == TASKS
    assert ruling["metrics"]["writes_matched"] == 3 and ruling["metrics"]["reads_semantic"] == 1
    assert {cells(row)[2] for row in rows_of(printed, "")} >= {"`gates.json replay_reference metrics`"}


def test_every_row_a_workdir_cannot_answer_names_the_record_it_would_need(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    text = B.render(B.Build(empty))
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(text, "")}
    assert values["Tasks covered (D96)"] == "n/a (needs scorecard.json task_coverage)"
    assert values["Dollars"] == "n/a (needs budget.json total)"
    assert "replay_reference" in values["Writes replaying exactly"]
    assert "`gates.json`" in text and "`scorecard.json`" in text


def test_the_peak_context_fill_is_n_a_because_no_stage_writes_the_fill_into_the_workdir(printed: str):
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(printed, "")}
    for agent in ("Builder", "Examiner"):
        assert values[f"Peak context fill, {agent}"].startswith("n/a (needs")
        assert "ContextStats.fill_at_turn_end" in values[f"Peak context fill, {agent}"]


def test_the_mechanic_row_says_no_model_turn_was_recorded_under_the_code_driver(printed: str):
    values = {cells(row)[0]: cells(row)[1] for row in rows_of(printed, "")}
    assert values["What the mechanic called"].startswith("0 over 0 model turns")
    assert values["Repairs requested"].startswith("0;")


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

def test_every_gate_in_the_ledger_is_named_with_its_ruling_and_how_many_rulings_were_green(printed: str, built: Path):
    ledger = json.loads((built / "gates.json").read_text(encoding="utf-8"))
    named = {cells(row)[0].strip("`"): cells(row) for row in rows_of(printed, "Gates")}
    assert set(named) == {row["stage"] for row in ledger}
    for stage, row in named.items():
        rulings = [entry for entry in ledger if entry["stage"] == stage]
        green = sum(1 for entry in rulings if entry["pass"])
        assert row[2] == f"{green} of {len(rulings)}"
        assert row[1] == ("green" if green == len(rulings) else "red")
    assert named["replay_reference"][1] == "red", "no Task on the fixture confirms every Trace"


# --- per tool -----------------------------------------------------------------

def test_the_per_tool_rows_count_every_replayed_call_of_that_tool(printed: str, built: Path):
    replays = json.loads((built / "replays.json").read_text(encoding="utf-8"))
    calls: dict[str, int] = {}
    for traces in replays.values():
        for replay in traces.values():
            for check in replay["checks"]:
                calls[check["tool"]] = calls.get(check["tool"], 0) + 1
    printed_calls = {cells(row)[0].strip("`"): int(cells(row)[2].replace(",", ""))
                     for row in rows_of(printed, "Per tool, over the replayed calls")}
    assert printed_calls == calls


def test_a_cause_is_one_env_fidelity_already_names_or_the_word_for_a_preview_it_cannot_read():
    read = {"verdict": "ours_refused", "ours": '{"payload": "NameError: name \'decimal\' is not defined"}',
            "recorded": '"27.4"'}
    assert B.check_cause(read) == "missing_import"
    assert B.check_cause({"verdict": "same", "ours": "1", "recorded": "1"}) == "none"
    assert B.check_cause({"verdict": "both_refused", "ours": '{"payload": "Error: no"}',
                          "recorded": '{"payload": "no"}'}) == "none"
    assert B.check_cause({"verdict": "differs", "ours": '{"value": 3}', "recorded": "3"}) == "result_shape"
    cut = {"verdict": "differs", "ours": '{"a": "' + "x" * 150 + "...", "recorded": '{"a": "y"}'}
    assert B.check_cause(cut) == B.UNREADABLE
    assert B.UNREADABLE not in CAUSE_OWNER, "the cause a preview hides is this file's word, not env_fidelity's"


def test_the_cause_table_names_the_builder_stage_that_owns_each_miss(printed: str):
    rows = [cells(row) for row in rows_of(printed, "Where the misses come from")]
    assert rows, "the fixture's list_all_product_types read differs from its recording"
    for cause, _calls, _share, _tools, owner, where in rows:
        if cause in CAUSE_OWNER:
            assert owner == CAUSE_OWNER[cause]
        else:
            assert cause == B.UNREADABLE and "160 character preview" in owner
        assert where == "`replays.json`"


# --- per Task, per round, per repair ------------------------------------------

def test_every_frozen_task_has_a_row_saying_how_far_it_got(printed: str, built: Path):
    frozen = json.loads((built / "tasks_frozen.json").read_text(encoding="utf-8"))["task_ids"]
    rows = rows_of(printed, "Per Task")
    assert {cells(row)[0].strip("`") for row in rows} == set(frozen)
    for row in rows:
        assert cells(row)[1] == "no" and cells(row)[2] == "no", "no fixture Task confirms a Reference"
        assert cells(row)[7] == f"`task_status.json {cells(row)[0].strip('`')}`"


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


def test_the_round_row_says_which_of_the_gate_counts_moved_and_what_the_round_spent(tmp_path):
    workdir = tmp_path / "rounds"
    workdir.mkdir()
    (workdir / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"fidelity": 1, "trusted": 0, "refused_count": 0, "assisted_runs": 0,
                                "probes_passing": 0, "spend": {"builder": 0.5, "examiner": 0.25, "total": 0.75}},
         "exit": None},
        {"round": 2, "counts": {"fidelity": 1, "trusted": 2, "refused_count": 0, "assisted_runs": 0,
                                "probes_passing": 0, "spend": {"builder": 0.1, "examiner": 0.0, "total": 0.1}},
         "exit": "stalled"}]), encoding="utf-8")
    first, second = [cells(row) for row in rows_of(B.render(B.Build(workdir)), "Per round")]
    assert first[2] == "first round" and first[5] == "the round did not exit"
    assert first[3] == "builder $0.50, examiner $0.25, total $0.75"
    assert second[2] == "trusted 2 from 0" and second[5] == "stalled"


def test_a_round_that_moved_no_count_says_so(printed: str):
    row = cells(rows_of(printed, "Per round")[0])
    assert row[0] == "1" and row[5] == "done"
    assert row[4] == "0, no model turn recorded"


def test_a_repair_request_is_listed_with_the_ruling_the_records_cannot_place(tmp_path):
    workdir = tmp_path / "repaired"
    (workdir / "repairs").mkdir(parents=True)
    (workdir / "repairs" / "repair_recompile.jsonl").write_text(
        json.dumps({"verb": "repair_recompile", "target": "calculate", "at": 1.0}) + "\n", encoding="utf-8")
    text = B.render(B.Build(workdir))
    row = cells(rows_of(text, "Per repair verb")[0])
    assert row[0] == "`repair_recompile`" and row[1] == "`calculate`"
    assert row[2].startswith("n/a (needs") and row[3] == row[2]
    values = {cells(line)[0]: cells(line)[1] for line in rows_of(text, "")}
    assert values["Repairs requested"] == "1 requested: repair_recompile 1"


def test_no_repair_verb_leaves_the_section_saying_the_code_driver_files_none(printed: str):
    assert "No repair verb was called" in printed.split("## Per repair verb", 1)[1]


# --- per failing Run ----------------------------------------------------------

def test_a_failing_run_is_ours_or_the_models_by_the_rule_that_matched_it(printed: str, built: Path):
    rows = [cells(row) for row in rows_of(printed, "Per failing Run: the model's error or ours")]
    listed = [row for row in rows if row[0].startswith("`replay-")]
    assert listed, "the fixture's third Trace does not replay its reads the way it recorded them"
    for row in listed:
        assert row[2] == "ours" and row[3] == "answer_differs"
        assert row[4].endswith("read: differs") or row[4].endswith("write: differs")
        assert row[5].startswith("`runs/")


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

def test_the_report_is_markdown_with_the_detail_under_the_headline(printed: str):
    assert printed.startswith("# Build table: ")
    for section in ("## Gates", "## Per tool, over the replayed calls", "## Per Task", "## Per round",
                    "## Per repair verb", "## Per failing Run: the model's error or ours"):
        assert f"\n{section}\n" in printed
    assert printed.endswith("\n")
    assert "—" not in printed and "–" not in printed


def test_a_long_detail_table_is_capped_and_says_how_many_rows_it_did_not_print(built: Path):
    build = B.Build(built)
    short = B.render(build, limit=1)
    assert "2 more Tasks not listed; pass `--limit 0` to print every row." in short
    assert len(rows_of(short, "Per Task")) == 1
    assert len(rows_of(B.render(build, limit=0), "Per Task")) == TASKS


def test_the_table_reads_the_workdir_and_never_writes_into_it(built: Path):
    before = {path: path.stat().st_mtime_ns for path in sorted(built.rglob("*")) if path.is_file()}
    B.render(B.Build(built))
    after = {path: path.stat().st_mtime_ns for path in sorted(built.rglob("*")) if path.is_file()}
    assert before == after


def test_the_script_prints_the_table_for_the_workdir_it_is_given(built: Path, capsys):
    assert B.main([str(built)]) == 0
    assert capsys.readouterr().out == B.render(B.Build(built))


def test_a_workdir_that_is_not_a_directory_is_one_clear_message(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        B.main([str(tmp_path / "no-such-build")])
    assert "is not a directory" in str(stopped.value)
