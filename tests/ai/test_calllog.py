"""The call log (kullback/ai/calllog.py, callrollup.py): a row per live call, its dollars, the cache counters."""

from __future__ import annotations

import random

import httpx
import pytest

from kullback.ai import calllog
from kullback.ai import calllog as calls
from kullback.ai import provider as pv
from kullback.ai.usage import Usage


@pytest.fixture(autouse=True)
def no_call_log():
    """No test leaves a call sink or its context behind for the next."""
    yield
    calls.install(None)
    calls.set_default()


def _price(usage, _model):
    return (usage.input * 2.0 + usage.cache_read * 0.2 + usage.output * 10.0) / 1_000_000


def _each(items, fn, workers):
    import threading

    threads = [threading.Thread(target=fn, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _interactions(handler) -> pv.GoogleInteractionsModel:
    pv.enable_live_calls_from_env({pv.LIVE_ENV_VAR: "1"})
    return pv.GoogleInteractionsModel("google/gemini-4-argon", api_key="k", env={},
                                      client=httpx.Client(transport=httpx.MockTransport(handler)),
                                      sleep=lambda _s: None, rng=random.Random(0))


ANSWERED = {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
            "usage": {"total_input_tokens": 5000, "total_cached_tokens": 3000, "total_output_tokens": 10,
                      "total_thought_tokens": 40}}


def test_a_live_call_is_logged_by_the_provider_with_its_context_tokens_price_and_http_attempts(tmp_path):
    sent = []

    def handler(request):
        sent.append(1)
        if len(sent) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=ANSWERED)

    log = calls.begin(tmp_path, "checks", run_id="exp-1", cost=_price)
    with calls.labelled("review", 2, group="run-a"):
        _interactions(handler).query([{"role": "system", "content": "secret system words"},
                                      {"role": "user", "content": "private user words"}])
    (row,) = calls.read(tmp_path)
    assert row == log.rows[0]
    assert (row["run_id"], row["stage"], row["label"], row["attempt"], row["group"]) == ("exp-1", "checks", "review", 2,
                                                                                           "run-a")
    assert (row["model"], row["http_attempts"]) == ("google/gemini-4-argon", 2)
    assert (row["input"], row["cache_read"], row["output"], row["reasoning"]) == (2000, 3000, 50, 40)
    assert row["usd_actual"] == round(_price(Usage(input=2000, cache_read=3000, output=50, reasoning=40), ""), 8) > 0
    assert row["failed_statuses"] == [429] and row["temperature_sent"] is None and row["effort_sent"] is None
    text = (tmp_path / calls.CALLS_FILE).read_text()
    assert "secret" not in text and "private" not in text


def test_a_failed_live_call_is_logged_with_its_error_and_still_raises(tmp_path):
    calls.begin(tmp_path, "exam")
    model = _interactions(lambda request: httpx.Response(400, json={"error": {"message": "bad request"}}))
    with pytest.raises(pv.ProviderError):
        model.query([{"role": "user", "content": "hi"}])
    (row,) = calls.read(tmp_path)
    assert row["stage"] == "exam" and row["label"] == calls.DEFAULT_LABEL and row["error"]


def test_with_no_log_begun_a_live_call_writes_nothing(tmp_path):
    _interactions(lambda request: httpx.Response(200, json=ANSWERED)).query([{"role": "user", "content": "hi"}])
    assert calllog.installed() is None and not list(tmp_path.iterdir())


def _record(messages, cache_read, total=6000, provider="google", group=None):
    return {"context": {"stage": "s", "label": "review", "group": group}, "model": "m", "provider": provider,
            "system": "S" * 100, "messages": messages, "tools": None, "wall_ms": 0.0,
            "usage": Usage(input=total - cache_read, cache_read=cache_read, output=1)}


def test_a_call_that_repeats_a_finished_calls_steps_should_read_them_and_a_shortfall_is_a_warned_miss(tmp_path):
    warned, clock = [], iter(range(100))
    log = calls.Log(tmp_path, cost=lambda *_: 0.0, clock=lambda: float(next(clock)), warn=warned.append)
    conversation = {"role": "user", "content": "C" * 4000}
    first = [conversation, {"role": "assistant", "content": "Read."}, {"role": "user", "content": "rule one"}]
    log(_record(first, 0))
    log(_record(first[:2] + [{"role": "user", "content": "rule two"}], 0))
    log(_record(first[:2] + [{"role": "user", "content": "rule three"}], 3900))
    rows = log.rows
    assert rows[0]["expected_cached"] == 0 and not rows[0]["miss"]
    assert rows[1]["expected_cached"] == rows[2]["expected_cached"] == min(round(6000 * 4110 / 4118), 6000 - 2049)
    assert rows[1]["miss"] and not rows[2]["miss"] and log.misses == 1
    assert len(warned) == 1 and "expected 3951" in warned[0]


def test_a_shared_prefix_another_call_refreshed_while_this_one_was_in_flight_is_still_expected(tmp_path):
    log = calls.Log(tmp_path, cost=lambda *_: 0.0, clock=lambda: 100.0, warn=lambda _l: None)
    conversation = {"role": "user", "content": "C" * 4000}
    record = _record([conversation, {"role": "assistant", "content": "Read."}, {"role": "user", "content": "one"}], 0)
    log(dict(record, wall_ms=1000.0))  # finished at 100, so the steps are cached from 100 on
    log.clock = lambda: 130.0
    log(dict(record, wall_ms=1000.0))  # the same steps finish again at 130
    log.clock = lambda: 131.0
    late = _record([conversation, {"role": "assistant", "content": "Read."}, {"role": "user", "content": "two"}], 3900)
    log(dict(late, wall_ms=20000.0))  # sent at 111, after the first finish and before the refresh
    assert log.rows[-1]["expected_cached"] > 0 and not log.rows[-1]["miss"]


def test_a_shared_prefix_under_the_providers_minimum_is_not_expected_from_the_cache(tmp_path):
    log = calls.Log(tmp_path, cost=lambda *_: 0.0, warn=lambda _l: None)
    shared = [{"role": "user", "content": "C" * 2600}, {"role": "assistant", "content": "Read."}]
    for rule in ("one", "two"):
        log(_record(shared + [{"role": "user", "content": rule}], 0, total=2900))
    assert log.rows[1]["expected_cached"] == 0 and not log.rows[1]["miss"]  # 851 would be under the 1024 minimum


def test_a_provider_with_no_known_cache_rule_gets_no_expectation_but_its_shared_text_is_counted(tmp_path):
    log = calls.Log(tmp_path, cost=lambda *_: 0.0, warn=lambda _l: None)
    log(_record([{"role": "user", "content": "same words, then rule one"}], 0, total=100, provider="other"))
    log(_record([{"role": "user", "content": "same words, then rule two"}], 0, total=100, provider="other"))
    assert "expected_cached" not in log.rows[1]
    assert log.rows[1]["uncacheable_shared"] > 0  # the shared words sit inside one step, past every boundary


def test_the_rollup_reports_actual_over_expected_misses_and_the_cached_share_of_first_and_later_calls():
    rows = [{"stage": "checks", "group": "r1", "input": 5000, "cache_read": 0, "expected_cached": 0},
            {"stage": "checks", "group": "r1", "input": 2000, "cache_read": 3000, "expected_cached": 3000},
            {"stage": "checks", "group": "r1", "input": 3000, "cache_read": 1000, "expected_cached": 3000,
             "miss": True, "attempt": 2},
            {"stage": "exam", "input": 10, "output": 3, "error": "ProviderError"}]
    out = calls.rollup(rows)
    checks = out["checks"]
    assert (checks["calls"], checks["misses"], checks["retries"]) == (3, 1, 1)
    assert checks["actual_over_expected"] == round(4000 / 6000, 4)
    assert checks["cached_share_first"] == 0.0 and checks["cached_share_later"] == round(4000 / 9000, 4)
    assert out["exam"]["errors"] == 1
    lines = calls.rollup_lines(rows)
    assert any(line.startswith("cache checks: actual/expected 66.67% (target 95%") for line in lines)


def test_a_stage_names_the_calls_its_pool_threads_send_and_a_label_stays_on_its_own_thread(tmp_path):
    log = calls.Log(tmp_path, cost=lambda *_: 0.0, warn=lambda _l: None)
    calllog.install(log)

    def send(i):
        with calls.labelled(f"session {i}"):
            calllog.emit("m", "other", "", [{"role": "user", "content": str(i)}], None, Usage(input=1))

    with calls.stage("review"):
        _each(range(4), send, 4)
    assert sorted((r["stage"], r["label"]) for r in log.rows) == [("review", f"session {i}") for i in range(4)]




def test_with_no_log_installed_and_the_calls_folder_set_the_first_call_installs_the_log_there(tmp_path, monkeypatch):
    calls.install(None)
    monkeypatch.setenv(calls.CALLS_DIR_VAR, str(tmp_path))
    calls.emit("free/model-a", "free", "", [{"role": "user", "content": "hi"}], None, usage=Usage(input=10, output=2))
    (row,) = calls.read(tmp_path)
    assert (row["model"], row["input"], row["output"]) == ("free/model-a", 10, 2)
    calls.install(None)


def test_the_rollup_adds_up_per_step_and_in_total_with_the_wall_time_percentiles():
    rows = [{"stage": "s", "label": label, "input": 10, "output": 1, "wall_ms": wall, "usd_actual": 0.5}
            for label, wall in [("a", 100), ("a", 300), ("b", 200)]]
    steps = calls.rollup(rows, "step")
    assert set(steps) == {"s/a", "s/b"} and steps["s/a"]["calls"] == 2
    total = calls.rollup(rows, "total")["total"]
    assert (total["calls"], total["input"], total["usd_actual"], total["p50_ms"], total["p90_ms"]) == (
        3, 30, 1.5, 200, 300)
    assert calls.rollup_lines(rows)[-1].startswith("run total: 3 calls")


def test_the_gateways_cost_header_is_the_estimate_and_the_upstreams_billed_cost_is_what_was_paid():
    usage = Usage(input=1000, output=100)
    assert calls.dollars(usage, "paid/model-a", "paid", 0.02, None) == {
        "usd_gateway": 0.02, "usd_billed": None, "usd_actual": 0.02, "price_source": "gateway"}
    assert calls.dollars(usage, "paid/model-a", "paid", 0.02, 0.015)["usd_actual"] == 0.015
    assert calls.dollars(usage, "paid/model-a", "paid", 0.02, 0.015)["price_source"] == "billed"


def test_a_subscription_route_pays_nothing_while_the_gateways_list_price_stays_on_the_row():
    out = calls.dollars(Usage(input=1000, output=100), "cc/model-a", "cc", 0.04, None)
    assert (out["usd_actual"], out["usd_gateway"], out["price_source"]) == (0.0, 0.04, "free")


def test_a_paid_route_the_gateway_reports_as_free_is_priced_from_the_local_table_and_flagged():
    model_id = next(iter(calls.LOCAL_PRICES))
    route = calls.route_of(model_id)
    price = calls.LOCAL_PRICES[model_id]
    out = calls.dollars(Usage(input=1_000_000, output=0), model_id, route, 0.0, None)
    assert (out["usd_gateway"], out["usd_actual"], out["price_source"]) == (price["input"], price["input"], "local")


def test_the_call_log_package_imports_nothing_from_kullback_outside_its_own_package():
    import re
    from pathlib import Path

    folder = Path(calls.__file__).parent
    for path in folder.glob("*.py"):
        for name in re.findall(r"^\s*(?:from|import)\s+(kullback\.\w+)", path.read_text(encoding="utf-8"), re.M):
            assert name == "kullback.ai", (path.name, name)


def test_a_price_from_the_gateways_pricing_table_stands_in_when_its_cost_header_gave_none():
    out = calls.dollars(Usage(input=1000, output=100), "paid/model-a", "paid", None, None, priced=0.003)
    assert (out["usd_gateway"], out["usd_actual"], out["price_source"]) == (0.003, 0.003, "pricing")
