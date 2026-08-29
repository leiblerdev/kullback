"""Tests for the models.dev live price source (D116): catalog parsing, the on-disk snapshot,
and budget.py preferring it over the hand-kept table."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from harness.shared import budget, pricing


def transport_of(handler):
    """An httpx client whose every request is answered by handler; nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def write_snapshot(path, catalog, fetched_at=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {
        "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat(),
        "catalog": catalog,
    }
    path.write_text(json.dumps(wrapped), encoding="utf-8")
    return wrapped


CATALOG = {
    "openai": {
        "models": {
            "gpt-5.6-luna": {
                "cost": {"input": 0.2, "output": 1.2, "cache_read": 0.02, "cache_write": 0.25}
            },
            # Deliberately different from budget.PRICES["openai/gpt-4.1-mini"] (0.40/1.60/0.10),
            # so a test that reads this value proves the snapshot won, not that the two agree.
            "gpt-4.1-mini": {
                "cost": {"input": 0.35, "output": 1.55, "cache_read": 0.09}
            },
        }
    },
    "anthropic": {
        "models": {
            "claude-opus-5": {
                "cost": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}
            }
        }
    },
}


# --- price_from_catalog ---


def test_price_from_catalog_reads_the_full_provider_slash_model_id():
    price = pricing.price_from_catalog(CATALOG, "openai/gpt-5.6-luna")
    assert price == {"input": 0.2, "output": 1.2, "cache_read": 0.02, "cache_write": 0.25}


def test_price_from_catalog_missing_cache_write_defaults_to_zero():
    price = pricing.price_from_catalog(CATALOG, "openai/gpt-4.1-mini")
    assert price["cache_write"] == 0.0


def test_price_from_catalog_missing_cache_read_defaults_to_input():
    catalog = {"x": {"models": {"m": {"cost": {"input": 3.0, "output": 9.0}}}}}
    price = pricing.price_from_catalog(catalog, "x/m")
    assert price["cache_read"] == 3.0
    assert price["cache_write"] == 0.0


def test_price_from_catalog_matches_the_bare_wire_id_across_providers():
    assert pricing.price_from_catalog(CATALOG, "gpt-5.6-luna") == pricing.price_from_catalog(
        CATALOG, "openai/gpt-5.6-luna"
    )


def test_price_from_catalog_unknown_model_is_none():
    assert pricing.price_from_catalog(CATALOG, "openai/does-not-exist") is None
    assert pricing.price_from_catalog(CATALOG, "nobody/nothing") is None


def test_price_from_catalog_handles_a_missing_or_empty_catalog():
    assert pricing.price_from_catalog(None, "openai/gpt-5.6-luna") is None
    assert pricing.price_from_catalog({}, "openai/gpt-5.6-luna") is None
    assert pricing.price_from_catalog(CATALOG, None) is None


def test_price_from_catalog_ignores_experimental_modes():
    """Only the model's own top-level cost is read; experimental.modes.*.cost is not a fallback."""
    catalog = {
        "openai": {
            "models": {
                "m": {
                    "cost": {"input": 1.0, "output": 2.0},
                    "experimental": {"modes": {"fast": {"cost": {"input": 0.1, "output": 0.2}}}},
                }
            }
        }
    }
    price = pricing.price_from_catalog(catalog, "openai/m")
    assert price["input"] == 1.0 and price["output"] == 2.0


# --- refresh: live off ---


def test_refresh_live_off_with_no_snapshot_returns_none_and_touches_no_network(tmp_path):
    def handler(request):
        raise AssertionError("refresh must not reach the network while live is off")

    result = pricing.refresh(
        client=transport_of(handler), path=tmp_path / "models.dev.json", env={}
    )
    assert result is None


def test_refresh_live_off_reads_the_existing_snapshot_regardless_of_age(tmp_path):
    def handler(request):
        raise AssertionError("refresh must not reach the network while live is off")

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG, fetched_at=datetime.now(timezone.utc) - timedelta(days=365))
    result = pricing.refresh(client=transport_of(handler), path=path, env={})
    assert result == CATALOG


# --- refresh: live on ---


def test_refresh_live_on_fetches_when_there_is_no_snapshot(tmp_path):
    def handler(request):
        assert str(request.url) == pricing.MODELS_DEV_URL
        return httpx.Response(200, json=CATALOG)

    path = tmp_path / "models.dev.json"
    result = pricing.refresh(
        client=transport_of(handler), path=path, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert result == CATALOG
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["catalog"] == CATALOG
    assert "fetched_at" in stored


def test_refresh_live_on_does_not_refetch_a_fresh_snapshot(tmp_path):
    def handler(request):
        raise AssertionError("a fresh snapshot must not be refetched")

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    result = pricing.refresh(
        client=transport_of(handler), path=path, max_age_days=7, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert result == CATALOG


def test_refresh_live_on_refetches_a_stale_snapshot(tmp_path):
    newer_catalog = {"openai": {"models": {"gpt-5.6-luna": {"cost": {"input": 0.1, "output": 0.5}}}}}

    def handler(request):
        return httpx.Response(200, json=newer_catalog)

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG, fetched_at=datetime.now(timezone.utc) - timedelta(days=30))
    result = pricing.refresh(
        client=transport_of(handler), path=path, max_age_days=7, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert result == newer_catalog
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["catalog"] == newer_catalog


def test_refresh_network_failure_falls_back_to_the_existing_snapshot(tmp_path):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG, fetched_at=datetime.now(timezone.utc) - timedelta(days=30))
    result = pricing.refresh(
        client=transport_of(handler), path=path, max_age_days=7, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert result == CATALOG, "a network error must fall back to the old snapshot, not raise"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["catalog"] == CATALOG, "a failed fetch must not touch the file on disk"


def test_refresh_network_failure_with_no_snapshot_returns_none(tmp_path):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    path = tmp_path / "models.dev.json"
    result = pricing.refresh(
        client=transport_of(handler), path=path, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert result is None


# --- budget.price_for and budget.price_source: models.dev first, then the table ---


def test_budget_price_for_prefers_models_dev_over_the_table(monkeypatch, tmp_path):
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    monkeypatch.setattr(budget, "_SNAPSHOT_PATH", path)
    monkeypatch.setattr(budget, "_CATALOG_LOADED", False)
    monkeypatch.setattr(budget, "_CATALOG", None)

    price = budget.price_for("openai/gpt-4.1-mini")
    assert price == {"input": 0.35, "output": 1.55, "cache_read": 0.09, "cache_write": 0.0}
    assert budget.price_source("openai/gpt-4.1-mini") == "models.dev"


def test_budget_price_for_falls_back_to_the_table_when_models_dev_has_no_row(monkeypatch, tmp_path):
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)  # CATALOG has no entry for claude-haiku-4-5
    monkeypatch.setattr(budget, "_SNAPSHOT_PATH", path)
    monkeypatch.setattr(budget, "_CATALOG_LOADED", False)
    monkeypatch.setattr(budget, "_CATALOG", None)

    assert budget.price_for("anthropic/claude-haiku-4-5") == budget.PRICES["anthropic/claude-haiku-4-5"]
    assert budget.price_source("anthropic/claude-haiku-4-5") == "table"


def test_budget_price_source_is_none_for_a_model_priced_by_neither_source():
    assert budget.price_for("openai/mystery") is None
    assert budget.price_source("openai/mystery") is None


def test_budget_price_for_reads_no_snapshot_by_default_in_tests():
    """isolated_price_catalog (conftest) points _SNAPSHOT_PATH at an empty tmp dir, so with no
    snapshot written, every model still comes from the table alone."""
    assert budget.price_source("anthropic/claude-opus-5") == "table"
    assert budget.price_for("anthropic/claude-opus-5") == budget.PRICES["anthropic/claude-opus-5"]


def test_record_call_writes_price_source_onto_the_event_and_the_totals(monkeypatch, tmp_path, workdir):
    from harness.shared.records import Cost, Event, Usage

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    monkeypatch.setattr(budget, "_SNAPSHOT_PATH", path)
    monkeypatch.setattr(budget, "_CATALOG_LOADED", False)
    monkeypatch.setattr(budget, "_CATALOG", None)

    event = Event(
        idx=0,
        type="model_call",
        cost=Cost(provider="openai", model="openai/gpt-4.1-mini", usage=Usage(input=1_000_000), wall_ms=1.0),
    )
    out = budget.record_call(event, stage="mine", workdir=workdir)
    assert out.cost.price_source == "models.dev"
    assert out.cost.usd == pytest.approx(0.35)

    totals = budget.load_totals(workdir)
    assert totals["stages"]["mine"]["models_dev_calls"] == 1
    assert totals["total"]["models_dev_calls"] == 1
