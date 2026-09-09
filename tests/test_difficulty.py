"""The difficulty record, the buckets and the two places they are reported (D209).

The world is an invented greenhouse: benches hold pots, a pot is watered and a bench is read. Every
number here is a count or a rate off records the harness already writes, so nothing in this file
calls a model and nothing in it names a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from kullback import cli, difficulty, report
from kullback.examiner import variants
from kullback.runner.records import Atom, Event, Run, Verifier, as_dict, write_json

WATER = "water_pot"
MOVE = "move_pot"
READ = "read_bench"
WRITE_TOOLS = (WATER, MOVE)


def _write_atom(atom_id: str, tool: str = WATER, entity: str = "p1") -> Atom:
    return Atom(id=atom_id, kind="required", target={"kind": "write", "tool": tool, "entity": entity})


def _value_atom(atom_id: str) -> Atom:
    return Atom(id=atom_id, kind="required",
                target={"kind": "write_value", "tool": WATER, "entity": "p1", "field": "litres"})


def _question_atom(atom_id: str) -> Atom:
    return Atom(id=atom_id, kind="question", target={"kind": "question", "key": "litres"})


def _verifier(*atoms: Atom, task_id: str = "task_1", seeds=("run-a",)) -> Verifier:
    return Verifier(task_id=task_id, atoms=list(atoms), seed_run_ids=list(seeds))


def _events(*calls: tuple[str, dict, dict]) -> list[Event]:
    out = [Event(idx=0, type="user_turn", payload={"content": "please water the pots"})]
    for number, (name, args, result) in enumerate(calls):
        call_id = f"c{number}"
        out.append(Event(idx=len(out), type="tool_call",
                         payload={"id": call_id, "name": name, "args": args, "requestor": "assistant"}))
        out.append(Event(idx=len(out), type="tool_result", payload={"id": call_id, "result": result}))
    return out


def _run(*calls: tuple[str, dict, dict], run_id: str = "run-a") -> Run:
    return Run(run_id=run_id, task_id="task_1", termination_reason="success", events=list(_events(*calls)))


def _calls(*calls: tuple[str, dict, dict]) -> list[dict]:
    return variants.call_results(_run(*calls))


READ_B1 = (READ, {"bench_id": "b1"}, {"bench_id": "b1", "pots": [{"pot_id": "p1"}]})
READ_B7 = (READ, {"bench_id": "b7"}, {"bench_id": "b7", "pots": [{"pot_id": "p7"}]})
WATER_P1 = (WATER, {"pot_id": "p1", "litres": 0.5}, {"pot_id": "p1", "watered": True})
MOVE_P9 = (MOVE, {"pot_id": "p9", "bench_id": "b2"}, {"pot_id": "p9", "moved": True})


# --- the record ----------------------------------------------------------------

def test_a_record_counts_the_writes_the_tools_the_paths_and_the_questions_of_one_task():
    verifier = _verifier(_write_atom("w0"), _value_atom("w0.litres"), _question_atom("q.litres"))
    record = difficulty.record_for(verifier, calls=_calls(READ_B1, WATER_P1, MOVE_P9),
                                   write_tools=WRITE_TOOLS, status_row={"references": 3},
                                   false_rejection=0.25)
    assert record["task_id"] == "task_1"
    assert record["write_count"] == 1, "the entity-level write atom counts once, its value atom not again"
    assert record["tools_touched"] == 3
    assert record["question_atoms"] == 1
    assert record["second_paths"] == 2, "three References are the Reference and two other paths"
    assert record["held_out_solve_rate"] == 0.75
    assert record["bucket"] == "w1t3+p2+"


def test_branching_depth_counts_every_rewrite_the_second_path_rule_finds_and_is_not_capped():
    calls = _calls(READ_B1, READ_B7, WATER_P1, MOVE_P9)
    depth = difficulty.branching_depth(calls, WRITE_TOOLS)
    assert depth > difficulty.record_for(_verifier(), calls=calls[:2], write_tools=WRITE_TOOLS)["branching_depth"]
    assert depth >= 4, "a longer path offers more rewrites, and none of them is thrown away for budget"


def test_a_single_call_path_has_no_rewrite_and_no_second_path():
    record = difficulty.record_for(_verifier(_write_atom("w0")), calls=_calls(WATER_P1),
                                   write_tools=WRITE_TOOLS, status_row={"references": 1})
    assert record["branching_depth"] == 0
    assert record["second_paths"] == 0
    assert record["bucket"] == "w1t1p1"


def test_a_task_that_held_nothing_out_reports_no_solve_rate_rather_than_a_perfect_one():
    record = difficulty.record_for(_verifier(), calls=[], false_rejection=None)
    assert record["held_out_solve_rate"] is None
    assert difficulty.solve_rate(None) is None
    assert difficulty.solve_rate(0.0) == 1.0, "an empty pool is not a rate of one, but a full pass is"


def test_a_task_no_judge_row_names_reports_no_agreement_rather_than_a_perfect_one():
    rows = [{"item_id": "task_1", "disagreement": True}, {"item_id": "task_1", "disagreement": False}]
    assert difficulty.judge_agreement(rows, "task_1") == 0.5
    assert difficulty.judge_agreement(rows, "task_2") is None
    assert difficulty.judge_agreement([], "task_1") is None


# --- the buckets ---------------------------------------------------------------

def test_the_bucket_bands_stop_counting_where_the_caps_say_and_never_move_with_a_corpus():
    assert difficulty.bucket_key(0, 1, 1) == "w0t1p1"
    assert difficulty.bucket_key(2, 2, 1) == "w2t2p1"
    assert difficulty.bucket_key(3, 3, 2) == "w3+t3+p2+"
    assert difficulty.bucket_key(9, 40, 7) == "w3+t3+p2+", "everything above a cap is one band"
    assert difficulty.bucket_key(2, 3, 2) != difficulty.bucket_key(3, 3, 2), "the last band below a cap is its own"


def test_a_bucket_row_counts_tasks_and_trusted_and_means_the_rate_over_the_tasks_that_have_one():
    records = [
        {"task_id": "a", "bucket": "w1t2p1", "held_out_solve_rate": 1.0},
        {"task_id": "b", "bucket": "w1t2p1", "held_out_solve_rate": 0.5},
        {"task_id": "c", "bucket": "w1t2p1", "held_out_solve_rate": None},
        {"task_id": "d", "bucket": "w0t1p1", "held_out_solve_rate": None},
    ]
    rows = difficulty.bucket_rows(records, trusted_ids=["a", "d"])
    assert [row["bucket"] for row in rows] == ["w0t1p1", "w1t2p1"]
    assert rows[1] == {"bucket": "w1t2p1", "tasks": 3, "trusted": 1, "solve_rate": 0.75, "rated": 2}
    assert rows[0]["solve_rate"] is None, "a bucket where nothing was held out has no rate"


# --- where the buckets are reported --------------------------------------------

def test_the_round_line_carries_trusted_over_tasks_for_every_bucket_that_holds_a_task():
    line = cli._round_line({"trusted": 4, "tasks": 6, "buckets": [
        {"bucket": "w0t1p1", "tasks": 2, "trusted": 1},
        {"bucket": "w1t2p2+", "tasks": 4, "trusted": 3},
    ]})
    assert "buckets: w0t1p1 1/2 w1t2p2+ 3/4" in line
    assert "buckets: none" in cli._round_line({"trusted": 0, "tasks": 0})


def test_the_report_table_names_every_bucket_and_says_which_tasks_carry_no_record():
    data = report.ReportData(difficulty={
        "buckets": [{"bucket": "w1t2p2+", "tasks": 4, "trusted": 3, "solve_rate": 0.5, "rated": 2}],
        "no_record": {"task_9": difficulty.NO_VERIFIER},
    })
    lines = report._difficulty_table(data)
    assert "### Difficulty buckets" in lines
    assert "| w1t2p2+ | 4 | 3 | 50% over 2 Tasks |" in lines
    assert any("1 Tasks carry no difficulty record" in line for line in lines)


def test_a_bucket_with_no_pool_says_so_in_the_table_rather_than_showing_a_rate():
    lines = difficulty.markdown_table([{"bucket": "w0t1p1", "tasks": 1, "trusted": 0,
                                        "solve_rate": None, "rated": 0}])
    assert "| w0t1p1 | 1 | 0 | no held-out pool |" in lines


def test_a_build_with_no_difficulty_file_says_so_rather_than_showing_an_empty_table():
    lines = report._difficulty_table(report.ReportData())
    assert any("No difficulty record was written" in line for line in lines)


# --- over a workdir ------------------------------------------------------------

def _workdir(tmp_path: Path) -> Path:
    """A workdir holding one Task with a Verifier and its Reference Run, and one Task with neither."""
    run = _run(READ_B1, WATER_P1, MOVE_P9)
    path = tmp_path / "runs" / "task_1" / "run-a.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_dict(run)) + "\n", encoding="utf-8")
    write_json(tmp_path / "verifiers" / "task_1.json",
               as_dict(_verifier(_write_atom("w0"), _question_atom("q.litres"))))
    write_json(tmp_path / "task_status.json", {
        "task_1": {"reference_confirmed": True, "references": 2},
        "task_2": {"reference_confirmed": False, "reason": "no seed recording replayed"},
    })
    write_json(tmp_path / "references.json", {"task_1": {"references": [{"run_id": "run-a"}]}})
    write_json(tmp_path / "replays.json",
               {"task_1": {"t-a": {"run_id": "run-a", "path": "runs/task_1/run-a.jsonl", "confirmed": True}}})
    write_json(tmp_path / "tool_sigs.json",
               [{"name": name, "kind": "write"} for name in WRITE_TOOLS] + [{"name": READ, "kind": "read"}])
    write_json(tmp_path / "rounds.json",
               [{"round": 1, "counts": {"trusted_ids": ["task_1"], "false_rejection": {"task_1": 0.5}}}])
    return tmp_path


def test_a_workdir_reads_back_a_record_per_task_with_a_verifier_and_names_the_rest(tmp_path):
    body = difficulty.compute(_workdir(tmp_path))
    assert [record["task_id"] for record in body["tasks"]] == ["task_1"]
    assert body["tasks"][0]["tools_touched"] == 3
    assert body["tasks"][0]["held_out_solve_rate"] == 0.5
    assert body["no_record"] == {"task_2": difficulty.NO_VERIFIER}
    assert body["buckets"] == [{"bucket": "w1t3+p2+", "tasks": 1, "trusted": 1,
                                "solve_rate": 0.5, "rated": 1}]


def test_the_reason_a_task_carries_no_record_is_the_harness_own_words_and_never_the_status_rows(tmp_path):
    workdir = _workdir(tmp_path)
    body = difficulty.compute(workdir)
    assert "no seed recording replayed" not in json.dumps(body), \
        "the status row quotes the customer's world back and this file is read into reports"


def test_a_verifier_whose_reference_run_is_gone_carries_no_record_rather_than_an_empty_one(tmp_path):
    workdir = _workdir(tmp_path)
    (workdir / "runs" / "task_1" / "run-a.jsonl").unlink()
    body = difficulty.compute(workdir)
    assert body["tasks"] == []
    assert body["no_record"]["task_1"] == difficulty.NO_REFERENCE_RUN


def test_the_record_is_written_beside_the_build_and_read_back_the_same(tmp_path):
    workdir = _workdir(tmp_path)
    written = difficulty.refresh(workdir)
    assert (workdir / difficulty.FILE_NAME).is_file()
    assert difficulty.read_records(workdir) == written
    assert written["format"] == difficulty.FORMAT
    assert difficulty.read_records(tmp_path / "empty")["buckets"] == []
