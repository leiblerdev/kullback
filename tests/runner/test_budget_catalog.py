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
