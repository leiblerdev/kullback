"""Behaviour tests for kullback.trainer.observe."""

from __future__ import annotations

import json

from kullback.trainer import observe


def rate_line(task_id, step, k, passes, fails, started, ended, stop="done"):
    return json.dumps({
        "format": 1, "train_run_id": "r1", "task_id": task_id,
        "task_origin": "recorded", "model_id": "m", "checkpoint": "c",
        "step": step, "source": "training", "k": k, "passes": passes,
        "fails": fails, "interrupted": 0, "no_signal": {},
        "stop_reasons": {stop: k}, "user_driver": "rule", "sampling": {},
        "seeds": list(range(k)), "env_hash": {}, "run_files": [],
        "started_at": started, "ended_at": ended,
    })


def write_run(folder, *, state=None, log=None, rates=(), metrics=()):
    folder.mkdir(parents=True, exist_ok=True)
    if state is not None:
        (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if log is not None:
        (folder / "run.log").write_text(log, encoding="utf-8")
    if rates:
        (folder / "solve_rates.jsonl").write_text("\n".join(rates) + "\n",
                                                  encoding="utf-8")
    if metrics:
        (folder / "metrics.jsonl").write_text("\n".join(metrics) + "\n",
                                              encoding="utf-8")


def test_runs_lists_sorted_run_ids(tmp_path):
    for name in ("b", "a"):
        folder = tmp_path / "training" / name
        folder.mkdir(parents=True)
        (folder / "state.json").write_text("{}", encoding="utf-8")
    assert observe.runs(tmp_path) == ["a", "b"]


def test_runs_missing_training_dir_gives_empty(tmp_path):
    assert observe.runs(tmp_path) == []


def test_training_status_reads_state_tail_errors_and_rate(tmp_path):
    folder = tmp_path / "training" / "r1"
    log = "\n".join([f"line {i}" for i in range(50)]
                    + ["boom Error here", "CUDA out of memory on device"])
    write_run(folder, state={"status": "running", "step": 7, "checkpoint": "c9",
                             "updated_at": 12.5}, log=log,
              rates=[rate_line("t1", 1, 4, 2, 2, 0.0, 60.0),
                     rate_line("t2", 2, 4, 4, 0, 60.0, 120.0)])
    status = observe.training_status(tmp_path, "r1")
    assert (status["status"], status["step"], status["checkpoint"]) == ("running", 7, "c9")
    assert status["updated_at"] == 12.5
    assert len(status["log_tail"]) == 40
    assert status["log_tail"][-1] == "CUDA out of memory on device"
    assert status["errors"] == ["boom Error here", "CUDA out of memory on device"]
    assert status["rollouts_per_minute"] == 8 / 2.0


def test_training_status_missing_files_give_nones_and_empty(tmp_path):
    status = observe.training_status(tmp_path, "ghost")
    assert status["status"] is None and status["step"] is None
    assert status["checkpoint"] is None and status["updated_at"] is None
    assert status["log_tail"] == [] and status["errors"] == []
    assert status["rollouts_per_minute"] is None


def test_training_status_single_rate_line_gives_no_rate(tmp_path):
    folder = tmp_path / "training" / "r1"
    write_run(folder, rates=[rate_line("t1", 1, 4, 2, 2, 0.0, 60.0)])
    assert observe.training_status(tmp_path, "r1")["rollouts_per_minute"] is None


def test_metrics_bands_spread_pass_estimates_stops_and_trend(tmp_path):
    folder = tmp_path / "training" / "r1"
    write_run(folder,
              rates=[rate_line("t1", 1, 4, 2, 2, 0.0, 10.0),
                     rate_line("t2", 1, 4, 4, 0, 0.0, 10.0),
                     rate_line("t1", 2, 4, 2, 2, 20.0, 30.0),
                     rate_line("t2", 2, 4, 4, 0, 20.0, 30.0)],
              metrics=[json.dumps({"step": i, "entropy": 1.0 + i, "kl": 0.5})
                       for i in range(5)])
    got = observe.metrics(tmp_path, "r1", window=2)
    assert got["bands"]["tasks"] == 2
    assert got["decision"]["action"] in ("ask", "ask_harder", "none",
                                         "report_no_signal")
    assert got["zero_spread_share"] == 0.5
    assert got["pass_at_4"] == ((1 - 1 / 70) + 1.0) / 2
    assert got["pass_hat_4"] == ((1 / 70) + 1.0) / 2
    assert got["stop_reasons"] == {"done": 16}
    assert got["trend"]["entropy"] == {"first": 1.0, "last": 5.0, "mean": 3.0}
    assert got["trend"]["kl"] == {"first": 0.5, "last": 0.5, "mean": 0.5}


def test_metrics_missing_files_give_empty_shapes(tmp_path):
    got = observe.metrics(tmp_path, "ghost")
    assert got["bands"]["tasks"] == 0
    assert got["zero_spread_share"] is None
    assert got["pass_at_4"] is None and got["pass_hat_4"] is None
    assert got["stop_reasons"] == {} and got["trend"] == {}


def test_metrics_tasks_below_n4_leave_pass_estimates_none(tmp_path):
    folder = tmp_path / "training" / "r1"
    write_run(folder, rates=[rate_line("t1", 1, 1, 1, 0, 0.0, 10.0),
                             rate_line("t1", 2, 1, 1, 0, 20.0, 30.0)])
    got = observe.metrics(tmp_path, "r1", window=2)
    assert got["pass_at_4"] is None and got["pass_hat_4"] is None


def test_task_pool_counts_and_lists_unseen_ids(tmp_path):
    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    pool = observe.task_pool(root, since=["widget_task"])
    assert (pool["total"], pool["with_verifier"]) == (1, 1)
    assert pool["agent_driven"] == 0 and pool["new"] == []


def test_task_pool_new_ids_are_those_not_in_since(tmp_path):
    import json as _json

    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    write_env(root, task_id="second_task")
    (root / "user_fidelity.json").write_text(_json.dumps(
        {"format": 1, "tasks": [{"task_id": "second_task", "drives": True}]}),
        encoding="utf-8")
    pool = observe.task_pool(root, since=["widget_task"])
    assert pool["total"] == 2 and pool["agent_driven"] == 1
    assert pool["with_verifier"] == 2 and pool["new"] == ["second_task"]


def ask_line(train_run_id, ask_id):
    return json.dumps({"seq": 1, "recorded_at": 0.0, "agent": "t",
                       "event": {"type": "custom", "name": "task_ask",
                                 "payload": {"train_run_id": train_run_id,
                                             "ask_id": ask_id}}})


def answered_line(ask_id):
    return json.dumps({"seq": 2, "recorded_at": 1.0, "agent": "s",
                       "event": {"type": "custom", "name": "task_ask_answered",
                                 "payload": {"ask_id": ask_id}}})


def test_synth_status_counts_asks_answers_and_open_ids(tmp_path):
    bus = tmp_path / "bus.jsonl"
    bus.write_text(ask_line("r1", "a1") + "\n" + ask_line("r1", "a2") + "\n"
                   + answered_line("a1") + "\n" + ask_line("r9", "z9") + "\n",
                   encoding="utf-8")
    assert observe.synth_status(bus, "r1") == {"asked": 2, "answered": 1,
                                               "open_ask_ids": ["a2"]}


def test_synth_status_missing_file_gives_zeros(tmp_path):
    assert observe.synth_status(tmp_path / "absent.jsonl", "r1") == {
        "asked": 0, "answered": 0, "open_ask_ids": []}


def test_metrics_zero_spread_counts_tasks_not_group_records(tmp_path):
    folder = tmp_path / "training" / "r1"
    write_run(folder, rates=[
        rate_line("t1", 1, 4, 2, 2, 0.0, 10.0),
        rate_line("t1", 1, 4, 2, 2, 10.0, 20.0),
        rate_line("t1", 1, 4, 2, 2, 20.0, 30.0),
        rate_line("t2", 1, 4, 4, 0, 0.0, 10.0),
    ])
    got = observe.metrics(tmp_path, "r1")
    assert got["bands"]["tasks"] == 2
    assert got["zero_spread_share"] == 0.5


def test_runs_skips_dirs_without_a_run_shape(tmp_path):
    root = tmp_path / "training"
    (root / "r1").mkdir(parents=True)
    (root / "r1" / "state.json").write_text("{}", encoding="utf-8")
    (root / "r2").mkdir(parents=True)
    (root / "r2" / "solve_rates.jsonl").write_text("", encoding="utf-8")
    for name in ("smoke", "scripts", "skills", "empty"):
        (root / name).mkdir(parents=True)
    assert observe.runs(tmp_path) == ["r1", "r2"]
