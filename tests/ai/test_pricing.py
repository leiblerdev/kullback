"""Tests for the models.dev live price source (D116): catalog parsing, the on-disk snapshot,
and budget.py preferring it over the hand-kept table."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from kullback.ai import pricing
from kullback.runner import budget


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


def from_models_dev(result):
    """What refresh returned minus the local provider registry it lays over every catalog.

    refresh answers with both, always; these tests are about the models.dev half, so the local
    half is taken off here rather than restated in every assertion.
    """
    local = pricing.local_providers()
    return {name: entry for name, entry in (result or {}).items() if name not in local}


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
    assert from_models_dev(result) == {}


def test_refresh_live_off_reads_the_existing_snapshot_regardless_of_age(tmp_path):
    def handler(request):
        raise AssertionError("refresh must not reach the network while live is off")

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG, fetched_at=datetime.now(timezone.utc) - timedelta(days=365))
    result = pricing.refresh(client=transport_of(handler), path=path, env={})
    assert from_models_dev(result) == CATALOG


# --- refresh: live on ---


def test_refresh_live_on_fetches_when_there_is_no_snapshot(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if str(request.url) != pricing.MODELS_DEV_URL:
            return httpx.Response(404, json={})  # a provider price list this test does not stub
        return httpx.Response(200, json=CATALOG)

    path = tmp_path / "models.dev.json"
    result = pricing.refresh(
        client=transport_of(handler), path=path, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert pricing.MODELS_DEV_URL in seen
    assert from_models_dev(result) == CATALOG
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["catalog"] == CATALOG, "the snapshot stays what models.dev said, overlay apart"
    assert "fetched_at" in stored


def test_refresh_live_on_does_not_refetch_a_fresh_snapshot(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404, json={})  # a provider price list this test does not stub

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    result = pricing.refresh(
        client=transport_of(handler), path=path, max_age_days=7, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert pricing.MODELS_DEV_URL not in seen, "a fresh snapshot must not be refetched"
    assert from_models_dev(result) == CATALOG


def test_refresh_live_on_refetches_a_stale_snapshot(tmp_path):
    newer_catalog = {"openai": {"models": {"gpt-5.6-luna": {"cost": {"input": 0.1, "output": 0.5}}}}}

    def handler(request):
        return httpx.Response(200, json=newer_catalog)

    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG, fetched_at=datetime.now(timezone.utc) - timedelta(days=30))
    result = pricing.refresh(
        client=transport_of(handler), path=path, max_age_days=7, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert from_models_dev(result) == newer_catalog
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
    assert from_models_dev(result) == CATALOG, \
        "a network error must fall back to the old snapshot, not raise"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["catalog"] == CATALOG, "a failed fetch must not touch the file on disk"


def test_refresh_network_failure_with_no_snapshot_leaves_only_the_local_registry(tmp_path):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    path = tmp_path / "models.dev.json"
    result = pricing.refresh(
        client=transport_of(handler), path=path, env={pricing.LIVE_ENV_VAR: "1"}
    )
    assert from_models_dev(result) == {}
    assert set(result) == set(pricing.local_providers())


# --- budget.price_for and budget.price_source: models.dev first, then the table ---


@pytest.mark.parametrize("model_id,price,source", [
    ("openai/gpt-4.1-mini",
     {"input": 0.35, "output": 1.55, "cache_read": 0.09, "cache_write": 0.0}, "models.dev"),
    # CATALOG has no entry for claude-haiku-4-5, so the hand-kept table answers instead.
    ("anthropic/claude-haiku-4-5", budget.PRICES["anthropic/claude-haiku-4-5"], "table"),
])
def test_budget_prices_from_the_snapshot_when_it_has_a_row_and_from_the_table_otherwise(
        tmp_path, model_id, price, source):
    """conftest's isolated_price_catalog already points budget at tmp_path / models.dev.json with
    the catalogue unloaded, so writing the snapshot there is the whole setup."""
    write_snapshot(tmp_path / "models.dev.json", CATALOG)

    assert budget.price_for(model_id) == price
    assert budget.price_source(model_id) == source


def test_budget_price_source_is_none_for_a_model_priced_by_neither_source():
    assert budget.price_for("openai/mystery") is None
    assert budget.price_source("openai/mystery") is None


def test_budget_price_for_reads_no_snapshot_by_default_in_tests():
    """isolated_price_catalog (conftest) points _SNAPSHOT_PATH at an empty tmp dir, so with no
    snapshot written, every model still comes from the table alone."""
    assert budget.price_source("anthropic/claude-opus-5") == "table"
    assert budget.price_for("anthropic/claude-opus-5") == budget.PRICES["anthropic/claude-opus-5"]


def test_record_call_writes_price_source_onto_the_event_and_the_totals(tmp_path, workdir):
    from kullback.runner.records import Cost, Event, Usage

    write_snapshot(tmp_path / "models.dev.json", CATALOG)

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


def test_model_adapter_prefers_the_model_row_over_the_provider_entry():
    from kullback.ai import pricing

    catalog = {"opencode-go": {"npm": "@ai-sdk/openai-compatible",
                               "models": {"minimax-m3": {"provider": {"npm": "@ai-sdk/anthropic"}},
                                          "kimi-k3": {}}}}
    assert pricing.model_adapter_for(catalog, "opencode-go/minimax-m3") == "@ai-sdk/anthropic"
    assert pricing.model_adapter_for(catalog, "opencode-go/kimi-k3") == "@ai-sdk/openai-compatible"
    assert pricing.model_adapter_for(catalog, "opencode-go/absent") == "@ai-sdk/openai-compatible"
    assert pricing.model_adapter_for(catalog, "nope/none") == ""
    assert pricing.model_adapter_for(None, "opencode-go/kimi-k3") == ""


def test_a_bare_wire_id_several_providers_price_differently_is_unpriced():
    """models.dev carries 'gpt-5.6-luna' under OpenAI and under a dozen resellers at their own
    rates. Taking whichever came first billed a real build at five times OpenAI's price, so a
    disagreement is no price at all; the ledger counts that as an unpriced call."""
    catalog = {
        "openai": {"models": {"gpt-5.6-luna": {"cost": {"input": 0.2, "output": 1.2}}}},
        "a-reseller": {"models": {"gpt-5.6-luna": {"cost": {"input": 1.0, "output": 6.0}}}},
    }
    assert pricing.price_from_catalog(catalog, "gpt-5.6-luna") is None
    assert pricing.price_from_catalog(catalog, "openai/gpt-5.6-luna")["input"] == 0.2
    assert pricing.price_from_catalog(catalog, "a-reseller/gpt-5.6-luna")["input"] == 1.0


def test_a_bare_wire_id_every_provider_prices_the_same_still_prices():
    catalog = {
        "openai": {"models": {"gpt-5.6-luna": {"cost": {"input": 0.2, "output": 1.2}}}},
        "a-mirror": {"models": {"gpt-5.6-luna": {"cost": {"input": 0.2, "output": 1.2}}}},
    }
    assert pricing.price_from_catalog(catalog, "gpt-5.6-luna")["input"] == 0.2


# --- nested wire ids, the shape a reseller mostly speaks ---


NESTED_CATALOG = {
    "a-gateway": {
        "id": "a-gateway", "npm": "@openrouter/ai-sdk-provider",
        "api": "https://gateway.invalid/api/v1", "env": ["A_GATEWAY_API_KEY"],
        "models": {
            "a-lab/small-3": {"limit": {"context": 1_000_000, "output": 64_000},
                              "cost": {"input": 0.03, "output": 0.13, "cache_read": 0.006}},
        },
    },
}


def test_a_wire_id_with_a_slash_in_it_prices_and_sizes_from_its_own_row():
    """A gateway names its models 'lab/model', so the id has two slashes and only the first one
    parts the provider from the wire id."""
    price = pricing.price_from_catalog(NESTED_CATALOG, "a-gateway/a-lab/small-3")
    assert price == {"input": 0.03, "output": 0.13, "cache_read": 0.006, "cache_write": 0.0}
    assert pricing.window_from_catalog(NESTED_CATALOG, "a-gateway/a-lab/small-3") == 1_000_000
    assert pricing.endpoint_from_catalog(NESTED_CATALOG, "a-gateway/a-lab/small-3").openai_shaped


# --- the local provider registry ---


LOCAL_FILE = {
    "a-host": {
        "id": "a-host", "npm": "@ai-sdk/openai-compatible", "api": "https://a-host.invalid/v1",
        "env": ["A_HOST_API_KEY"],
        "models": {"quick-1": {"limit": {"context": 32_000}, "cost": {"input": 1.0, "output": 2.0}}},
    },
}


def write_local(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / pricing.LOCAL_PROVIDERS_NAME).write_text(json.dumps(entries), encoding="utf-8")


def test_a_provider_only_the_local_registry_names_is_reachable_priced_and_sized(tmp_path):
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    write_local(path, LOCAL_FILE)
    catalog = pricing.refresh(path=path, env={})
    endpoint = pricing.endpoint_from_catalog(catalog, "a-host/quick-1")
    assert endpoint.base_url == "https://a-host.invalid/v1"
    assert endpoint.key_env_var == "A_HOST_API_KEY" and endpoint.openai_shaped
    assert pricing.price_from_catalog(catalog, "a-host/quick-1")["output"] == 2.0
    assert pricing.window_from_catalog(catalog, "a-host/quick-1") == 32_000
    assert catalog["openai"] == CATALOG["openai"], "the overlay adds providers, it does not drop any"


def test_the_local_registry_can_add_one_model_row_to_a_provider_models_dev_already_names(tmp_path):
    """The failure this is for: the registry names the provider and the vendor serves a wire id the
    snapshot has no row for, so a real Run was billed nothing and failed the budget gate. One row
    in the local file prices it, and the provider's host, key and other models still come from
    models.dev."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, {"a-vendor": {"id": "a-vendor", "npm": "@ai-sdk/openai-compatible",
                                       "api": "https://a-vendor.invalid", "env": ["A_VENDOR_API_KEY"],
                                       "models": {"old-1": {"cost": {"input": 9.0, "output": 9.0}}}}})
    write_local(path, {"a-vendor": {"models": {"new-1": {"limit": {"context": 500_000},
                                                         "cost": {"input": 0.3, "output": 1.2}}}}})
    catalog = pricing.refresh(path=path, env={})
    assert pricing.price_from_catalog(catalog, "a-vendor/new-1")["input"] == 0.3
    assert pricing.price_from_catalog(catalog, "a-vendor/old-1")["input"] == 9.0
    assert pricing.endpoint_from_catalog(catalog, "a-vendor/new-1").base_url == "https://a-vendor.invalid"
    assert pricing.endpoint_from_catalog(catalog, "a-vendor/new-1").key_env_var == "A_VENDOR_API_KEY"


def test_a_row_correcting_one_number_keeps_everything_else_the_catalog_knows(tmp_path):
    """A file that moves a price says the price and stops. Restating the window to keep it would be
    a second place for it to go stale, and dropping it would leave the model priced and unsized,
    which is a Run refused for a limit nobody meant to remove."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, {"a-vendor": {"id": "a-vendor", "npm": "@ai-sdk/openai-compatible",
                                       "api": "https://a-vendor.invalid", "env": ["A_VENDOR_API_KEY"],
                                       "models": {"old-1": {"name": "Old One",
                                                            "limit": {"context": 200_000},
                                                            "cost": {"input": 9.0, "output": 9.0}}}}})
    write_local(path, {"a-vendor": {"models": {"old-1": {"cost": {"input": 3.0, "output": 6.0}}}}})
    catalog = pricing.refresh(path=path, env={})
    assert pricing.price_from_catalog(catalog, "a-vendor/old-1")["input"] == 3.0
    assert pricing.window_from_catalog(catalog, "a-vendor/old-1") == 200_000
    assert catalog["a-vendor"]["models"]["old-1"]["name"] == "Old One"


def test_a_row_correcting_one_rate_keeps_the_rates_beside_it(tmp_path):
    """The same rule inside the cost mapping. A file that says the input rate moved says that and
    nothing else, and a model left with an input and no output is a model the budget gate refuses,
    which is a Run lost to a number nobody meant to remove."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, {"a-vendor": {"id": "a-vendor", "npm": "@ai-sdk/openai-compatible",
                                       "api": "https://a-vendor.invalid", "env": ["A_VENDOR_API_KEY"],
                                       "models": {"old-1": {"limit": {"context": 200_000, "output": 8_000},
                                                            "cost": {"input": 9.0, "output": 18.0,
                                                                     "cache_read": 0.9}}}}})
    write_local(path, {"a-vendor": {"models": {"old-1": {"cost": {"input": 3.0},
                                                         "limit": {"context": 400_000}}}}})
    catalog = pricing.refresh(path=path, env={})
    assert pricing.price_from_catalog(catalog, "a-vendor/old-1") == {
        "input": 3.0, "output": 18.0, "cache_read": 0.9, "cache_write": 0.0}
    assert pricing.window_from_catalog(catalog, "a-vendor/old-1") == 400_000
    assert catalog["a-vendor"]["models"]["old-1"]["limit"]["output"] == 8_000


def test_a_model_row_that_names_nothing_changes_nothing(tmp_path):
    """The narrow end of the same rule: an override row with no fields in it says nothing, so the
    price and the window underneath stand rather than being wiped by an empty row."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, {"a-vendor": {"id": "a-vendor", "npm": "@ai-sdk/openai-compatible",
                                       "api": "https://a-vendor.invalid", "env": ["A_VENDOR_API_KEY"],
                                       "models": {"old-1": {"limit": {"context": 200_000},
                                                            "cost": {"input": 9.0, "output": 9.0}}}}})
    write_local(path, {"a-vendor": {"models": {"old-1": {}}}})
    catalog = pricing.refresh(path=path, env={})
    assert pricing.price_from_catalog(catalog, "a-vendor/old-1")["input"] == 9.0
    assert pricing.window_from_catalog(catalog, "a-vendor/old-1") == 200_000


def test_a_local_registry_file_that_cannot_be_read_is_ignored_not_raised_on(tmp_path):
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    (tmp_path / pricing.LOCAL_PROVIDERS_NAME).write_text("{ not json", encoding="utf-8")
    catalog = pricing.refresh(path=path, env={})
    assert catalog["openai"] == CATALOG["openai"]
    assert set(pricing.local_providers(path)) == set(pricing.BUILTIN_LOCAL_PROVIDERS)


def test_a_provider_row_whose_models_is_a_list_is_left_out_and_the_rows_beside_it_still_price(tmp_path):
    """A misshapen row must not take every model call down. Its shape is wrong, not one of its
    fields, so the whole row is left out rather than half-applied, and the file's other providers
    are read as if it were not there."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    write_local(path, {"b-host": {"api": "https://b-host.invalid/v1", "env": ["B_HOST_API_KEY"],
                                  "npm": "@ai-sdk/openai-compatible",
                                  "models": ["swift-2", "swift-3"]},
                       **LOCAL_FILE})
    catalog = pricing.refresh(path=path, env={})
    assert "b-host" not in pricing.local_providers(path)
    assert pricing.price_from_catalog(catalog, "a-host/quick-1")["output"] == 2.0
    assert pricing.endpoint_from_catalog(catalog, "a-host/quick-1").key_env_var == "A_HOST_API_KEY"
    assert catalog["openai"] == CATALOG["openai"]


def test_a_provider_row_whose_models_is_a_string_is_left_out_of_the_registry(tmp_path):
    path = tmp_path / "models.dev.json"
    write_local(path, {"c-host": {"api": "https://c-host.invalid/v1", "models": "swift-2"}})
    registry = pricing.local_providers(path)
    assert "c-host" not in registry
    assert set(registry) == set(pricing.BUILTIN_LOCAL_PROVIDERS), "the built-in rows still answer"


def test_a_model_row_that_is_a_string_is_dropped_and_its_well_formed_siblings_still_price(tmp_path):
    """One bad row inside a models mapping is narrower than a bad models field: only that wire id
    goes, and the provider it sits in keeps answering for the ids that are written properly."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    write_local(path, {"d-host": {"id": "d-host", "npm": "@ai-sdk/openai-compatible",
                                  "api": "https://d-host.invalid/v1", "env": ["D_HOST_API_KEY"],
                                  "models": {"swift-2": "0.5 in, 1.5 out",
                                             "swift-3": {"limit": {"context": 64_000},
                                                         "cost": {"input": 0.5, "output": 1.5}}}}})
    catalog = pricing.refresh(path=path, env={})
    assert pricing.price_from_catalog(catalog, "d-host/swift-2") is None
    assert pricing.price_from_catalog(catalog, "d-host/swift-3")["output"] == 1.5
    assert pricing.window_from_catalog(catalog, "d-host/swift-3") == 64_000
    assert pricing.endpoint_from_catalog(catalog, "d-host/swift-3").base_url == "https://d-host.invalid/v1"


# --- a provider row that names its own price list ---


PRICE_LIST_URL = "https://e-host.invalid/v1/models"

LIVE_PRICED = {
    "e-host": {
        "id": "e-host", "npm": "@ai-sdk/openai-compatible", "api": "https://e-host.invalid/v1",
        "env": ["E_HOST_API_KEY"],
        "prices": {"url": PRICE_LIST_URL, "field": "rates"},
        "models": {"brisk-4": {"limit": {"context": 128_000},
                               "cost": {"input": 1.0, "output": 2.0, "cache_read": 0.1,
                                        "cache_write": 1.0}}},
    },
}

WRITTEN_RATES = {"input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 1.0}


def price_list_transport(listing, seen=None, status=200):
    """The provider's price list, stubbed. models.dev and every other host answer an empty body, so
    a test says what one listing serves and nothing leaves the machine."""
    def handler(request):
        if seen is not None:
            seen.append(request)
        if str(request.url) != PRICE_LIST_URL:
            return httpx.Response(200, json={})
        if listing is None:
            raise httpx.ConnectError("the price list is unreachable", request=request)
        return httpx.Response(status, json=listing)
    return transport_of(handler)


LIVE_ENV = {pricing.LIVE_ENV_VAR: "1", "E_HOST_API_KEY": "e-host-test-key"}


def refresh_with(listing, path, seen=None, status=200, env=None, remember=False):
    """One refresh against a stubbed price list. The process-wide reading of that list is dropped
    first, so each test says for itself what the vendor serves; remember=True keeps it, which is
    what a test of the reading being kept needs."""
    if not remember:
        pricing.forget_live_prices()
    write_snapshot(path, CATALOG)
    write_local(path, LIVE_PRICED)
    return pricing.refresh(client=price_list_transport(listing, seen, status), path=path,
                           env=dict(LIVE_ENV) if env is None else env)


def test_a_row_that_names_a_price_list_is_billed_at_the_rates_that_list_serves_now(tmp_path):
    """A written rate is stale the day the vendor moves one, and one gateway here moved by nearly a
    factor of two inside a day, so the row's own rates give way to the list it names."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25,
                                                    "cache_read": 0.05, "cache_write": 0.5}}]}
    catalog = refresh_with(listing, tmp_path / "models.dev.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == {
        "input": 0.5, "output": 1.25, "cache_read": 0.05, "cache_write": 0.5}
    assert pricing.window_from_catalog(catalog, "e-host/brisk-4") == 128_000, \
        "the list prices the model, it does not restate the rest of the row"


def test_a_price_list_that_does_not_carry_the_named_field_leaves_the_written_rates_standing(tmp_path):
    """A vendor that renames the field, or lists a model it does not price, must leave the ledger
    with a rate rather than with none."""
    listing = {"data": [{"id": "brisk-4", "cost_per_token": {"input": 0.5, "output": 1.25}}]}
    catalog = refresh_with(listing, tmp_path / "models.dev.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_a_price_list_that_cannot_be_reached_leaves_the_written_rates_standing(tmp_path):
    catalog = refresh_with(None, tmp_path / "models.dev.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_a_price_list_that_answers_an_error_leaves_the_written_rates_standing(tmp_path):
    listing = {"error": "no key"}
    catalog = refresh_with(listing, tmp_path / "models.dev.json", status=503)
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_a_rate_that_does_not_read_as_a_number_leaves_the_written_rates_standing(tmp_path):
    """A misread rate is worse than a dated one: it bills every call of the Run wrong."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": "half a cent", "output": 1.25}},
                        {"id": "brisk-5", "rates": {"input": -1.0, "output": 1.25}}]}
    catalog = refresh_with(listing, tmp_path / "models.dev.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_the_price_list_is_asked_for_with_the_key_the_row_names_and_only_when_live_is_on(tmp_path):
    """The list is behind the provider's own key, and reaching for it is a call off the machine, so
    it happens under the one switch every other call is under."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}}]}
    seen = []
    refresh_with(listing, tmp_path / "models.dev.json", seen=seen)
    asked = [request for request in seen if str(request.url) == PRICE_LIST_URL]
    assert len(asked) == 1
    assert asked[0].headers["authorization"] == f"Bearer {LIVE_ENV['E_HOST_API_KEY']}"

    off = []
    catalog = refresh_with(listing, tmp_path / "off.json", seen=off, env={})
    assert [request for request in off if str(request.url) == PRICE_LIST_URL] == []
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_no_key_for_a_provider_means_its_price_list_is_never_asked_for(tmp_path):
    """Nobody can call a provider they hold no key for, so its rates price nothing, and a machine
    that only ever calls one vendor must not be reaching out to another's host to ask."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}}]}
    seen = []
    catalog = refresh_with(listing, tmp_path / "models.dev.json", seen=seen,
                           env={pricing.LIVE_ENV_VAR: "1"})
    assert [request for request in seen if str(request.url) == PRICE_LIST_URL] == []
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES


def test_two_listed_spellings_of_one_model_that_disagree_leave_the_written_rate_standing(tmp_path):
    """A gateway lists a model under its own name and the lab's, and both name the same row here.
    Agreeing rates can be read from either; disagreeing ones say nothing about what the wallet is
    billed, so the row keeps the rate written beside it."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}},
                        {"id": "a-lab/brisk-4", "rates": {"input": 0.9, "output": 3.0}}]}
    catalog = refresh_with(listing, tmp_path / "models.dev.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4") == WRITTEN_RATES

    agreeing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}},
                         {"id": "a-lab/brisk-4", "rates": {"input": 0.5, "output": 1.25}}]}
    catalog = refresh_with(agreeing, tmp_path / "agreed.json")
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4")["input"] == 0.5


def test_a_price_list_is_read_once_a_process_and_not_once_a_lookup(tmp_path):
    """refresh runs wherever an id is resolved, not only where a call is billed, so asking the
    vendor on every lookup would be a request per Run for a number that moves in days."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}}]}
    seen = []
    path = tmp_path / "models.dev.json"
    refresh_with(listing, path, seen=seen)
    catalog = refresh_with(listing, path, seen=seen, remember=True)
    assert len([r for r in seen if str(r.url) == PRICE_LIST_URL]) == 1
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4")["input"] == 0.5, \
        "the reading that was kept is still laid over the row"

    pricing.forget_live_prices()
    moved = {"data": [{"id": "brisk-4", "rates": {"input": 0.7, "output": 1.4}}]}
    catalog = refresh_with(moved, path, seen=seen, remember=True)
    assert pricing.price_from_catalog(catalog, "e-host/brisk-4")["input"] == 0.7, \
        "and forgetting it asks the vendor again"


def test_a_price_list_cannot_add_a_model_the_row_does_not_offer(tmp_path):
    """The written row says what the Harness offers. A list that prices two hundred other models
    changes the price of the one it names and adds nothing, so nothing is offered unsized."""
    listing = {"data": [{"id": "brisk-4", "rates": {"input": 0.5, "output": 1.25}},
                        {"id": "brisk-9", "rates": {"input": 0.7, "output": 1.4}}]}
    catalog = refresh_with(listing, tmp_path / "models.dev.json")
    assert set(catalog["e-host"]["models"]) == {"brisk-4"}
    assert pricing.price_from_catalog(catalog, "e-host/brisk-9") is None


def test_the_built_in_gateway_row_names_its_price_list_and_carries_a_fallback_rate():
    """The row that went stale inside a day: it says where its prices live, and what to bill at
    when that list cannot be read."""
    row = pricing.BUILTIN_LOCAL_PROVIDERS["cheaperinference"]
    assert row["prices"]["url"].startswith(row["api"])
    assert row["prices"]["field"]
    for model in row["models"].values():
        assert model["cost"]["input"] > 0 and model["cost"]["output"] > 0


def test_a_row_whose_models_mapping_names_nothing_readable_is_left_out_whole(tmp_path):
    """An empty mapping, or one whose every row is misshapen, leaves nothing to lay over. Landing
    the top-level fields alone would move the host and leave the prices where they were, which is
    the same half-applied row a models field of the wrong shape would have been."""
    path = tmp_path / "models.dev.json"
    name = next(iter(pricing.BUILTIN_LOCAL_PROVIDERS))
    for models in ({}, {"quick-1": "1 in, 2 out"}):
        write_local(path, {name: {"api": "https://elsewhere.invalid/v1", "models": models}})
        assert pricing.local_providers(path)[name] == pricing.BUILTIN_LOCAL_PROVIDERS[name]


def test_a_built_in_provider_given_a_misshapen_models_field_keeps_the_rows_it_shipped_with(tmp_path):
    """The overlay row is dropped whole, so the provider underneath is untouched: the built-in
    host and prices stand rather than a half-merged mixture of the two."""
    path = tmp_path / "models.dev.json"
    name = next(iter(pricing.BUILTIN_LOCAL_PROVIDERS))
    write_local(path, {name: {"api": "https://elsewhere.invalid/v1", "models": ["not", "an", "object"]}})
    row = pricing.local_providers(path)[name]
    assert row == pricing.BUILTIN_LOCAL_PROVIDERS[name]


def test_the_local_file_wins_over_a_built_in_row_of_the_same_name(tmp_path):
    path = tmp_path / "models.dev.json"
    name = next(iter(pricing.BUILTIN_LOCAL_PROVIDERS))
    write_local(path, {name: {"api": "https://elsewhere.invalid/v1"}})
    assert pricing.local_providers(path)[name]["api"] == "https://elsewhere.invalid/v1"
    assert pricing.local_providers(path)[name]["models"], "its model rows survive the override"


def test_a_wire_id_priced_only_by_the_overlay_is_billed_and_an_unpriced_one_still_is_not(tmp_path):
    """The gate that refused the Run must keep refusing: an overlay row prices its own model and
    nothing else, so a model neither source names stays unpriced."""
    path = tmp_path / "models.dev.json"
    write_snapshot(path, CATALOG)
    write_local(path, LOCAL_FILE)
    budget._SNAPSHOT_PATH = path
    budget._CATALOG_LOADED = False
    try:
        assert budget.is_priced("a-host/quick-1") is True
        assert budget.price_source("a-host/quick-1") == "models.dev"
        assert budget.is_priced("a-host/absent-9") is False
        with pytest.raises(budget.UnpricedModel):
            budget.Ceiling(usd=1.0).require_priced("a-host/absent-9")
    finally:
        budget._CATALOG_LOADED = False


def test_every_built_in_local_provider_row_carries_what_the_one_lookup_needs():
    """A row read off a vendor's docs is only useful if it answers all four questions the single
    lookup asks: where to post, which variable holds the key, what a call costs, what fits."""
    for name, entry in pricing.BUILTIN_LOCAL_PROVIDERS.items():
        models = entry.get("models") or {}
        assert models, f"{name} names no model"
        for wire, row in models.items():
            assert row["cost"]["input"] >= 0 and row["cost"]["output"] >= 0
            if "api" in entry:
                assert entry["npm"] in pricing.OPENAI_SHAPED, f"{name} is served in another shape"
                assert entry["env"] and entry["env"][0].endswith("_API_KEY")
                assert row["limit"]["context"] > 0, f"{name}/{wire} names no context window"


# --- a gateway that answers under a name it was not called by ---


ECHOING_CATALOG = {
    "a-gateway": {
        "id": "a-gateway", "npm": "@ai-sdk/openai-compatible", "api": "https://gateway.invalid/v1",
        "env": ["A_GATEWAY_API_KEY"],
        "models": {"quick-1": {"limit": {"context": 400_000},
                               "cost": {"input": 0.06, "output": 0.2, "cache_read": 0.012}}},
    },
}


def test_a_model_the_gateway_answers_under_an_upstream_name_prices_from_its_own_row():
    """The ledger keys a call on the id the endpoint echoed, and a gateway asked for 'quick-1' can
    answer 'a-lab/quick-1'. Without this the call is billed nothing and fails the budget gate."""
    price = pricing.price_from_catalog(ECHOING_CATALOG, "a-gateway/a-lab/quick-1")
    assert price["input"] == 0.06
    assert pricing.window_from_catalog(ECHOING_CATALOG, "a-gateway/a-lab/quick-1") == 400_000


def test_a_bare_name_also_finds_the_row_a_provider_lists_under_a_nested_one():
    catalog = {"a-gateway": {"models": {"a-lab/quick-1": {"cost": {"input": 0.06, "output": 0.2}}}}}
    assert pricing.price_from_catalog(catalog, "a-gateway/quick-1")["input"] == 0.06


def test_two_rows_of_one_provider_ending_the_same_way_are_no_price_at_all():
    """The same rule the cross-provider lookup follows: an ambiguous match is worse than a missing
    one, because a wrong price bills a build for something it did not spend."""
    catalog = {"a-gateway": {"models": {
        "a-lab/quick-1": {"cost": {"input": 0.06, "output": 0.2}},
        "b-lab/quick-1": {"cost": {"input": 6.0, "output": 20.0}},
    }}}
    assert pricing.price_from_catalog(catalog, "a-gateway/quick-1") is None
    assert pricing.price_from_catalog(catalog, "a-gateway/a-lab/quick-1")["input"] == 0.06
