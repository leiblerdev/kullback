"""Behaviour tests for kullback.curriculum."""

import pytest
from pydantic import ValidationError

from kullback.curriculum import (
    SolveRate,
    TaskAsk,
    TaskView,
    append,
    decide,
    fold,
    pass_at_k,
    pass_hat_k,
    read,
    window_summary,
)


def make_rate(**over):
    base = {
        "train_run_id": "run1",
        "task_id": "t1",
        "task_origin": "recorded",
        "birth_level": "easy",
        "model_id": "m1",
        "checkpoint": "ckpt0",
        "step": 0,
        "source": "training",
        "k": 8,
        "passes": 3,
        "fails": 5,
        "interrupted": 0,
        "no_signal": {},
        "stop_reasons": {},
        "user_driver": "rule",
        "sampling": {},
        "seeds": [1, 2],
        "env_hash": {"runner": "a"},
        "run_files": [],
        "started_at": 1.0,
        "ended_at": 2.0,
    }
    base.update(over)
    return SolveRate(**base)


def test_k_validator_accepts_split_counts():
    r = make_rate(k=8, passes=2, fails=3, interrupted=1, no_signal={"stall": 2})
    assert r.passes + r.fails + r.interrupted + sum(r.no_signal.values()) == 8


@pytest.mark.parametrize(
    "counts",
    [
        {"passes": 4, "fails": 5, "interrupted": 0, "no_signal": {}},
        {"passes": 2, "fails": 5, "interrupted": 0, "no_signal": {}},
        {"passes": 3, "fails": 4, "interrupted": 0, "no_signal": {"stall": 2}},
    ],
)
def test_k_validator_rejects_counts_off_k(counts):
    with pytest.raises(ValidationError):
        make_rate(k=8, **counts)


@pytest.mark.parametrize(
    "counts,band,rate",
    [
        ({"passes": 0, "fails": 0, "no_signal": {"stall": 8}}, "no_signal", None),
        ({"passes": 0, "fails": 8, "no_signal": {}}, "all_fail", 0.0),
        ({"passes": 8, "fails": 0, "no_signal": {}}, "all_pass", 1.0),
        ({"passes": 3, "fails": 5, "no_signal": {}}, "mixed", 0.375),
    ],
)
def test_band_labels_follow_scored_counts(counts, band, rate):
    r = make_rate(k=8, interrupted=0, **counts)
    assert r.band == band
    assert r.scored == counts["passes"] + counts["fails"]
    assert r.solve_rate == pytest.approx(rate) if rate is not None else r.solve_rate is None


def test_pass_at_k_hand_value_n8_c2_k4():
    assert pass_at_k(8, 2, 4) == pytest.approx(11 / 14)


def test_pass_hat_k_hand_value_n8_c2_k4():
    assert pass_hat_k(8, 2, 4) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "n,c,k,want_at,want_hat",
    [
        (8, 0, 4, 0.0, 0.0),
        (8, 8, 4, 1.0, 1.0),
        (5, 2, 2, 1 - 3 / 10, 1 / 10),
    ],
)
def test_pass_estimates_match_table(n, c, k, want_at, want_hat):
    assert pass_at_k(n, c, k) == pytest.approx(want_at)
    assert pass_hat_k(n, c, k) == pytest.approx(want_hat)


def test_pass_estimates_return_none_when_n_below_k():
    assert pass_at_k(3, 1, 4) is None
    assert pass_hat_k(3, 1, 4) is None


def test_fold_keeps_only_last_n_checkpoints_and_pools():
    recs = [
        make_rate(task_id="t1", step=0, ended_at=1.0, passes=8, fails=0),
        make_rate(task_id="t1", step=1, ended_at=2.0, passes=0, fails=8),
        make_rate(task_id="t1", step=2, ended_at=3.0, passes=4, fails=4),
        make_rate(task_id="t2", step=2, ended_at=3.5, passes=8, fails=0),
    ]
    views = {v.task_id: v for v in fold(recs, last_checkpoints=1)}
    assert set(views) == {"t1", "t2"}
    assert (views["t1"].passes, views["t1"].fails) == (4, 4)
    assert views["t1"].band == "mixed"
    assert views["t1"].runs == 8
    assert views["t1"].solve_rate == pytest.approx(0.5)
    assert views["t2"].band == "all_pass"


def test_fold_pools_across_kept_checkpoints():
    recs = [
        make_rate(task_id="t1", step=0, ended_at=1.0, passes=8, fails=0),
        make_rate(task_id="t1", step=1, ended_at=2.0, passes=0, fails=8),
    ]
    (view,) = fold(recs, last_checkpoints=2)
    assert (view.passes, view.fails, view.runs) == (8, 8, 16)
    assert view.band == "mixed"


def test_fold_treats_step_none_records_as_own_checkpoints():
    recs = [
        make_rate(task_id="t1", step=None, ended_at=1.0, passes=8, fails=0),
        make_rate(task_id="t1", step=None, ended_at=2.0, passes=0, fails=8),
        make_rate(task_id="t1", step=None, ended_at=3.0, passes=4, fails=4),
    ]
    (view,) = fold(recs, last_checkpoints=1)
    assert (view.passes, view.fails) == (4, 4)


def view(task_id, band, level=None, runs=8, no_signal=0):
    scored = {"mixed": (3, 5), "all_pass": (8, 0), "all_fail": (0, 8), "no_signal": (0, 0)}[band]
    return TaskView(
        task_id=task_id,
        birth_level=level,
        runs=runs,
        passes=scored[0],
        fails=scored[1],
        no_signal=no_signal,
        solve_rate=(scored[0] / (scored[0] + scored[1]) if scored[0] + scored[1] else None),
        band=band,
    )


def test_window_summary_counts_tasks_and_shares():
    views = [view("t1", "mixed", "easy"), view("t2", "all_pass", "easy"), view("t3", "no_signal", "easy")]
    s = window_summary(views)
    assert (s["tasks"], s["mixed"], s["all_pass"], s["all_fail"], s["no_signal"]) == (3, 1, 1, 0, 1)
    assert s["mixed_share"] == pytest.approx(0.5)
    assert s["by_level"]["easy"]["mixed_share"] == pytest.approx(0.5)
    assert s["by_level"]["easy"]["tasks"] == 3


def test_window_summary_groups_none_birth_level():
    views = [view("t1", "mixed"), view("t2", "all_fail", "hard")]
    s = window_summary(views)
    assert s["by_level"]["none"]["tasks"] == 1
    assert s["by_level"]["hard"]["all_fail"] == 1
    assert s["mixed_share"] == pytest.approx(0.5)


def test_window_summary_no_signal_share_is_run_based():
    views = [
        view("t1", "mixed", "easy", runs=8, no_signal=0),
        view("t2", "no_signal", "easy", runs=8, no_signal=8),
    ]
    s = window_summary(views)
    assert s["no_signal_share"] == pytest.approx(0.5)
    assert s["mixed_share"] == pytest.approx(1.0)


def test_window_summary_empty_pool_gives_none_shares():
    s = window_summary([])
    assert s["tasks"] == 0
    assert s["mixed_share"] is None
    assert s["no_signal_share"] is None
    assert s["by_level"] == {}


def test_decide_reports_no_signal_first():
    s = window_summary(
        [
            view("t1", "no_signal", "easy", runs=8, no_signal=8),
            view("t2", "all_pass", "easy"),
        ]
    )
    d = decide(s)
    assert d["action"] == "report_no_signal"
    assert d["reason"].endswith(".")
    assert "percent" in d["reason"]


def test_decide_asks_harder_above_saturation():
    views = [view(f"e{i}", "all_pass", "easy") for i in range(4)] + [view("e9", "mixed", "easy")]
    d = decide(window_summary(views))
    assert d["action"] == "ask_harder"
    assert d["level"] == "medium"
    assert d["count"] == 4


def test_decide_picks_level_with_most_mixed():
    views = [view("e1", "mixed", "easy")] + [view(f"e{i}", "all_fail", "easy") for i in range(5)]
    views += [view("m1", "mixed", "medium"), view("m2", "mixed", "medium")]
    views += [view(f"m{i}", "all_fail", "medium") for i in range(6)]
    d = decide(window_summary(views), saturated_bar=0.99)
    assert d["action"] == "ask"
    assert d["level"] == "medium"


def test_ask_count_brings_mixed_share_to_the_bar_once_new_tasks_join():
    views = [view(f"t{i}", "all_fail") for i in range(10)]
    d = decide(window_summary(views))
    assert d["action"] == "ask"
    assert d["count"] == 5
    assert (0 + d["count"]) / (10 + d["count"]) >= 0.30


def test_decide_hard_saturation_means_no_ask():
    views = [view(f"h{i}", "all_pass", "hard") for i in range(4)] + [view("h9", "mixed", "hard")]
    d = decide(window_summary(views))
    assert d["action"] == "none"


def test_decide_saturation_beats_thin_band():
    views = [view(f"e{i}", "all_pass", "easy") for i in range(9)] + [view("m1", "all_fail", "medium")]
    s = window_summary(views)
    assert s["mixed_share"] == 0.0
    d = decide(s)
    assert d["action"] == "ask_harder"


def test_decide_no_signal_beats_saturation():
    views = [view(f"e{i}", "all_pass", "easy") for i in range(8)]
    views += [view(f"n{i}", "no_signal", "easy", runs=8, no_signal=8) for i in range(8)]
    s = window_summary(views)
    d = decide(s)
    assert d["action"] == "report_no_signal"


def test_decide_asks_easiest_fallback_when_all_fail():
    views = [view("t1", "all_fail"), view("t2", "all_fail")]
    d = decide(window_summary(views))
    assert d["action"] == "ask"
    assert d["level"] == "easy"
    assert 1 <= d["count"] <= 60


def test_decide_asks_medium_fallback_otherwise():
    views = [view("t1", "all_pass"), view("t2", "all_pass")]
    s = window_summary(views)
    d = decide(s, saturated_bar=0.99)
    assert d["action"] == "ask"
    assert d["level"] == "medium"
    assert 1 <= d["count"] <= 60


def test_decide_rests_when_band_is_healthy():
    views = [view("t1", "mixed"), view("t2", "mixed"), view("t3", "all_fail"), view("t4", "all_pass")]
    d = decide(window_summary(views))
    assert d["action"] == "none"
    assert d["reason"].endswith(".")


def test_ask_id_is_stable_and_hex():
    a = TaskAsk.ask_id_for("run1", 5, "easy")
    assert a == TaskAsk.ask_id_for("run1", 5, "easy")
    assert len(a) == 16
    int(a, 16)
    assert TaskAsk.ask_id_for("run1", 6, "easy") != a
    assert TaskAsk.ask_id_for("run1", 5, "hard") != a


def test_ask_rejects_bad_count_and_extra_exemplars():
    with pytest.raises(ValidationError):
        TaskAsk(
            ask_id="x",
            train_run_id="r",
            model_id="m",
            checkpoint="c",
            step=1,
            level="easy",
            count=0,
            evidence={},
            exemplars=[],
        )
    with pytest.raises(ValidationError):
        TaskAsk(
            ask_id="x",
            train_run_id="r",
            model_id="m",
            checkpoint="c",
            step=1,
            level="easy",
            count=61,
            evidence={},
            exemplars=[],
        )
    with pytest.raises(ValidationError):
        TaskAsk(
            ask_id="x",
            train_run_id="r",
            model_id="m",
            checkpoint="c",
            step=1,
            level="easy",
            count=2,
            evidence={},
            exemplars=["a", "b", "c", "d", "e", "f"],
        )


def test_append_then_read_round_trip(tmp_path):
    path = tmp_path / "rates.jsonl"
    r = make_rate()
    append(path, r)
    append(path, make_rate(task_id="t2", step=1, ended_at=3.0))
    got = read(path)
    assert [g.task_id for g in got] == ["t1", "t2"]
    assert got[0].solve_rate == pytest.approx(0.375)


def test_read_missing_path_returns_empty(tmp_path):
    assert read(tmp_path / "absent.jsonl") == []
