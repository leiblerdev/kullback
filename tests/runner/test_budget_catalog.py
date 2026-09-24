"""A call is priced under the catalog row of the id it was sent with: exact, then without the
profile, then the vendor's own row, then the hand table. A small fake catalog, never the network."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from kullback.ai.model_limits import catalog_candidates, split_vendor, without_profile
from kullback.runner import budget, feed
from kullback.runner.records import Cost, Event, Usage

CATALOG = {
    "amazon-bedrock": {"id": "amazon-bedrock", "models": {
        "global.a-vendor.big-1": {"cost": {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.0},
                                  "limit": {"context": 900_000}},
        "us.a-vendor.big-1": {"cost": {"input": 4.4, "output": 22.0, "cache_read": 0.44, "cache_write": 5.5},
                              "limit": {"context": 900_000}},
        "global.a-vendor.small-1": {"cost": {"input": 1.0, "output": 5.0}},
        "a-vendor.small-1": {"cost": {"input": 1.1, "output": 5.5}},
    }},
    "a-vendor": {"id": "a-vendor", "models": {
        "big-1": {"cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
                  "limit": {"context": 800_000}},
    }},
}
MILLION = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)


@pytest.fixture
def catalog(tmp_path):
    """conftest points budget at tmp_path / models.dev.json with the catalog unloaded."""
    (tmp_path / "models.dev.json").write_text(json.dumps(
        {"fetched_at": datetime.now(timezone.utc).isoformat(), "catalog": CATALOG}), encoding="utf-8")


def test_the_candidates_run_exact_then_without_profile_then_the_vendor_row():
    assert catalog_candidates("bedrock/us.a-vendor.big-1") == [
        ("exact", "amazon-bedrock/us.a-vendor.big-1"),
        ("without-profile", "amazon-bedrock/a-vendor.big-1"),
        ("vendor", "a-vendor/big-1"),
    ]
    assert catalog_candidates("a-host/quick-1") == [("exact", "a-host/quick-1")]
    assert without_profile("bedrock/global.a-vendor.big-1") == "bedrock/a-vendor.big-1"
    assert split_vendor("global.a-vendor.big-1.5") == ("a-vendor", "big-1.5")
    assert split_vendor("quick-1") is None


def test_each_profile_prices_from_its_own_row_and_an_unlisted_one_from_the_vendor_row(catalog):
    assert budget.call_cost(MILLION, "bedrock/global.a-vendor.big-1") == pytest.approx(29.4)
    assert budget.call_cost(MILLION, "bedrock/us.a-vendor.big-1") == pytest.approx(32.34)
    assert budget.call_cost(MILLION, "bedrock/eu.a-vendor.big-1") == pytest.approx(22.05)
    assert budget.price_source("bedrock/global.a-vendor.big-1") == "models.dev"
    assert budget.price_source("bedrock/eu.a-vendor.big-1") == "models.dev:vendor"
    for model in ("bedrock/global.a-vendor.big-1", "bedrock/us.a-vendor.big-1", "bedrock/eu.a-vendor.big-1"):
        assert budget.is_priced(model)
    assert budget.window_for("bedrock/global.a-vendor.big-1") == 900_000
    assert budget.window_for("bedrock/eu.a-vendor.big-1") == 800_000


def test_a_recorded_call_is_priced_and_its_feed_line_names_the_id_it_was_sent_with(catalog, tmp_path):
    """The endpoint echoes a stripped name; the ledger prices and records the one that was sent."""
    workdir = tmp_path / "w"
    price = budget.subscriber(workdir, "builder", "bedrock/global.a-vendor.big-1")
    from kullback.agent.events import MessageEndEvent
    from kullback.ai.messages import AssistantMessage

    price(MessageEndEvent(message=AssistantMessage(content="ok", usage=Usage(input=1000, output=10),
                                                   model="big-1")))
    total = budget.load_totals(workdir)["total"]
    assert total["usd"] > 0 and total["unpriced_calls"] == 0 and total["models_dev_calls"] == 1
    rows = [json.loads(line) for line in feed.path_for(workdir).read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == ["bedrock/global.a-vendor.big-1"]


def test_reprice_recovers_the_cost_of_a_round_that_ran_unpriced(catalog, tmp_path):
    workdir = tmp_path / "w"
    for model in ("bedrock/stripped-1", "bedrock/stripped-1"):
        cost = Cost(provider="bedrock", model=model.split("/")[1], usage=Usage(input=1_000_000, output=100_000))
        budget.record_call(Event(idx=0, type="model_call", cost=cost), stage="solve", workdir=workdir)
    before = budget.load_totals(workdir)["total"]
    assert (before["usd"], before["unpriced_calls"]) == (0, 2)

    after = budget.reprice(workdir, model_id="bedrock/global.a-vendor.big-1")
    assert after["total"]["usd"] == pytest.approx(2 * (4.0 + 2.0))
    assert after["total"]["unpriced_calls"] == 0 and after["total"]["calls"] == 2
    assert after["stages"]["solve"]["usd"] == after["total"]["usd"]
    assert budget.load_totals(workdir) == after


SESSION_CALLS = (
    Usage(input=1_000, output=200_000, cache_read=3_000_000, cache_write=400_000, cache_write_1h=300_000),
    Usage(input=50_000, output=10_000, cache_read=900_000, cache_write=20_000),
    Usage(),
    Usage(input=2_000, output=70_000, cache_read=1_500_000, cache_write=90_000, cache_write_1h=90_000),
)


def _record(workdir, usage, stage="solve"):
    cost = Cost(provider="bedrock", model="global.a-vendor.big-1", usage=usage)
    budget.record_call(Event(idx=0, type="model_call", cost=cost), stage=stage, workdir=workdir)


def test_reprice_of_a_workdir_built_across_two_sessions_prices_every_call_not_only_the_last_session(
        catalog, tmp_path):
    """feed.start truncates the feed on each session; the ledger's counters keep every call (D291)."""
    workdir = tmp_path / "w"
    feed.start(workdir)
    for usage in SESSION_CALLS[:3]:
        _record(workdir, usage)
    feed.start(workdir)
    _record(workdir, SESSION_CALLS[3])

    after = budget.reprice(workdir, model_id="bedrock/global.a-vendor.big-1")
    every_call = sum(budget.call_cost(usage, "bedrock/global.a-vendor.big-1") for usage in SESSION_CALLS)
    saved = sum(budget.cache_effect(usage, "bedrock/global.a-vendor.big-1") for usage in SESSION_CALLS)
    assert after["total"]["usd"] == pytest.approx(every_call)
    assert after["total"]["cache_saved_usd"] == pytest.approx(saved)
    assert after["total"]["calls"] == 4 and after["total"]["models_dev_calls"] == 4
    assert budget.reprice(workdir)["total"]["usd"] == pytest.approx(every_call)


def test_reprice_of_a_single_session_workdir_gives_the_numbers_its_feed_lines_price_to(catalog, tmp_path):
    """Guard: when the feed holds every call, pricing the counters changes nothing (D291)."""
    workdir = tmp_path / "w"
    feed.start(workdir)
    for stage, usage in zip(("solve", "solve", "check", "check"), SESSION_CALLS, strict=True):
        _record(workdir, usage, stage)
    rows = [row for row in feed.read_since(workdir)[0] if row["kind"] == "model_call"]
    by_line: dict[str, float] = {}
    for row in rows:
        usage = Usage(**{field: row.get(field, 0) for field in ("input", "output", "cache_read", "cache_write", "cache_write_1h")})
        by_line[row["stage"]] = by_line.get(row["stage"], 0.0) + budget.call_cost(usage, "bedrock/us.a-vendor.big-1")

    after = budget.reprice(workdir, model_id="bedrock/us.a-vendor.big-1")
    assert {stage: bucket["usd"] for stage, bucket in after["stages"].items()} == pytest.approx(by_line)
    assert after["total"]["usd"] == pytest.approx(sum(by_line.values()))


def test_a_bare_bedrock_id_is_priced_under_the_global_row_it_is_sent_on(catalog, tmp_path):
    """The adapter puts a bare id on the `global.` profile, so the ledger prices that row, not the bare one."""
    from kullback.agent.events import MessageEndEvent
    from kullback.ai.messages import AssistantMessage

    workdir = tmp_path / "w"
    price = budget.subscriber(workdir, "builder", "bedrock/a-vendor.small-1")
    price(MessageEndEvent(message=AssistantMessage(content="ok", usage=Usage(input=1_000_000), model="small-1")))
    assert budget.load_totals(workdir)["total"]["usd"] == pytest.approx(1.0)
    rows = [json.loads(line) for line in feed.path_for(workdir).read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == ["bedrock/global.a-vendor.small-1"]


def test_reprice_of_a_build_resumed_on_another_model_prices_each_session_at_its_own_model(catalog, tmp_path):
    """The ledger keeps each stage's calls per model, so the earlier session is not priced at the later rate."""
    workdir = tmp_path / "w"
    first, second = Usage(input=1_000_000, output=100_000), Usage(input=500_000, output=50_000)
    feed.start(workdir)
    budget.record_call(Event(idx=0, type="model_call", cost=Cost(
        provider="bedrock", model="global.a-vendor.big-1", usage=first)), stage="solve", workdir=workdir)
    feed.start(workdir)
    budget.record_call(Event(idx=0, type="model_call", cost=Cost(
        provider="bedrock", model="global.a-vendor.small-1", usage=second)), stage="solve", workdir=workdir)

    after = budget.reprice(workdir)
    expected = (budget.call_cost(first, "bedrock/global.a-vendor.big-1")
                + budget.call_cost(second, "bedrock/global.a-vendor.small-1"))
    assert after["total"]["usd"] == pytest.approx(expected)
    assert after["stages"]["solve"]["usd"] == pytest.approx(expected)
    assert after["total"]["calls"] == 2 and after["total"]["models_dev_calls"] == 2


def test_reprice_of_a_ledger_without_a_per_model_split_prices_as_before(catalog, tmp_path):
    """A budget.json written before the split prices its feed lines and the rest under the last model."""
    workdir = tmp_path / "w"
    feed.start(workdir)
    for usage in SESSION_CALLS:
        _record(workdir, usage)
    totals = budget.load_totals(workdir)
    del totals["stages"]["solve"]["models"]
    budget.save_totals(workdir, totals)

    after = budget.reprice(workdir)
    assert after["total"]["usd"] == pytest.approx(
        sum(budget.call_cost(usage, "bedrock/global.a-vendor.big-1") for usage in SESSION_CALLS))


def test_reprice_keeps_a_direct_charge_the_ceiling_recorded_without_tokens(catalog, tmp_path):
    """A charge made through Ceiling.add has dollars and no tokens; reprice carries it over."""
    workdir = tmp_path / "w"
    feed.start(workdir)
    _record(workdir, SESSION_CALLS[0])
    budget.Ceiling(usd=100.0, workdir=workdir).add(2.5, stage="solve", item="a")

    after = budget.reprice(workdir)
    called = budget.call_cost(SESSION_CALLS[0], "bedrock/global.a-vendor.big-1")
    assert after["stages"]["solve"]["usd"] == pytest.approx(called + 2.5)
    assert after["total"]["usd"] == pytest.approx(called + 2.5)
    assert after["total"]["calls"] == 2 and after["total"]["models_dev_calls"] == 1
