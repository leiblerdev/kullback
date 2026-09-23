"""How many replayed calls rest on a value only a held-out Run witnessed (D220 rule 2c).

Carried from the deleted build module's tests: two Runs of one Task, one of them held out.
"""

from __future__ import annotations

from kullback.builder.replay_evidence import HeldOutRuns, holdout_answers
from kullback.runner.records import write_json

ANCHOR = HeldOutRuns({"task_ferry": ["run_held"]})


def test_a_call_answered_out_of_a_held_out_only_value_is_counted_per_tool_and_per_task():
    replays = {"task_ferry": {
        "run_held": {"confirmed": True,
                     "counts": {"answered_from_holdout": 1, "holdout_columns": ["crossings.berth"]},
                     "checks": [{"tool": "read_crossing", "answered_from_holdout": ["crossings.berth"]}]},
        "run_seed": {"confirmed": True, "counts": {"answered_from_holdout": 0}, "checks": [
            {"tool": "read_crossing"}]}}}

    found = holdout_answers(replays, ANCHOR)

    assert found["totals"] == {"calls": 1, "held_out_calls": 1, "runs": 1, "held_out_runs": 1}
    assert found["tools"] == {"read_crossing": {"calls": 1, "held_out_calls": 1}}
    assert found["columns"] == ["crossings.berth"]


def test_a_corpus_no_value_of_which_is_held_out_only_counts_nothing():
    replays = {"task_ferry": {"run_seed": {"confirmed": True, "counts": {}, "checks": []}}}

    assert holdout_answers(replays, ANCHOR)["totals"]["calls"] == 0


def test_the_runners_per_call_rows_are_counted_like_the_replays_own_checks():
    replays = {"task_ferry": {"run_held": {
        "confirmed": True, "counts": {"answered_from_holdout": 1},
        "calls": [{"tool": "read_crossing", "answered_from_holdout": ["crossings.berth"]}]}}}

    assert holdout_answers(replays, ANCHOR)["tools"] == {
        "read_crossing": {"calls": 1, "held_out_calls": 1}}


def test_the_anchor_is_read_off_the_workdir_and_holds_nothing_out_without_one(tmp_path):
    assert not HeldOutRuns.of(tmp_path).is_held_out("run_held")
    write_json(tmp_path / "anchor.json", {"held_out": {"task_ferry": ["run_held"]}})
    anchor = HeldOutRuns.of(tmp_path)
    assert anchor.is_held_out("run_held") and not anchor.is_held_out("run_seed")
