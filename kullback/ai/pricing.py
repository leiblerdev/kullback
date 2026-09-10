"""The live price source: models.dev's catalog, snapshotted to disk (D116).

budget.py's hand-kept PRICES table is the offline fallback; this module is what keeps prices
current without a person re-checking a vendor's page by hand. The one way this module leaves
the machine is the same switch every other adapter uses: kullback.ai.provider.LIVE_ENV_VAR.
With that off (the default, and always off in tests), refresh() only ever reads the snapshot
already on disk, or returns None when there is none.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Optional

import httpx

# LIVE_ENV_VAR is re-exported: a caller reading prices names the switch through this module
# rather than reaching past it into the adapters.
from kullback.ai.provider import LIVE_ENV_VAR, live_calls_requested  # noqa: F401

MODELS_DEV_URL = "https://models.dev/api.json"

# A snapshot older than this is refetched, but only when live calls are on.
DEFAULT_MAX_AGE_DAYS = 7


def snapshot_path(path: Optional[str | Path] = None) -> Path:
    """Where the catalog snapshot lives on disk. Overridable so tests never touch the real one."""
    if path is not None:
        return Path(path)
    return Path.home() / ".cache" / "harness" / "models.dev.json"


# The one mechanism for a provider models.dev does not list. A file beside the snapshot, in the
# same entry shape as a models.dev provider row, is laid over the catalog before anyone reads it,
# so the host, the key variable, the prices and the context windows all come out of the single
# lookup that already exists: model_for resolves the id, budget.py prices the call, the D65 cap
# reads the window, and no stage learns a second place to look.
#
# Shape, the catalog's own, so a row can be copied either way:
#
#   {"cheaperinference": {"id": "cheaperinference", "name": "...", "npm": "@ai-sdk/openai-compatible",
#                         "api": "https://...", "env": ["..._API_KEY"],
#                         "models": {"<wire id>": {"limit": {"context": 1, "output": 1},
#                                                  "cost": {"input": 0.0, "output": 0.0,
#                                                           "cache_read": 0.0}}}}}
#
# A provider named in both wins on its own top-level fields, and its model rows are merged into the
# catalog's row by row, so one missing model can be added without restating the provider.
#
# A row may also name the provider's own price list, which is the only source that cannot go stale:
#
#   "prices": {"url": "https://.../v1/models", "field": "pricing"}
#
# url is the provider's model listing, field is the key each listed model carries its rates under,
# in this catalog's own cost shape and unit (input, output, cache_read, cache_write, USD per 1M
# tokens). refresh reads that listing when live calls are on and a key for the provider is held,
# and lays the rates it finds over the model rows the row already names, matched by the same lookup
# that resolves a call. It adds no models: the written row says which models the Harness offers and
# what they cost when the listing cannot be read, so a vendor that renames a field, answers an
# error, or is unreachable leaves the prices below standing rather than leaving a call unpriced.
LOCAL_PROVIDERS_NAME = "providers.local.json"

# The key of the row naming a provider's live price list, and the model listing's own list of rows.
PRICES_FIELD = "prices"
LISTING_ROWS_FIELD = "data"

# Providers the Harness ships knowing about because their docs were read, kept in code so a fresh
# machine has them without a file to write, and overridable by the file above. CheaperInference is
# the first: an OpenAI-compatible gateway, "The API is served from https://api.cheaperinference.com/v1.
# Authenticate with `Authorization: Bearer ci_live_...`", integrations that read the key from
# CHEAPER_INFERENCE_API_KEY, and "@ai-sdk/openai-compatible" as the SDK its own docs name
# (platform.cheaperinference.com/llms.txt and /docs, read 2026-09-10). The model row is copied from
# that day's GET /v1/models, which prices every model it lists: the rates the gateway charges, which
# is what the wallet is billed, rather than the list rates it discounts from.
#
# That listing is also where the prices come from now, because the gateway's docs say a rate moves
# whenever the upstream it buys from moves, and a written number is stale the day after it is read:
# the first copy of this row, taken 2026-09-10, already priced input at 0.060426 against the
# 0.105 the same endpoint served hours later, which is the ledger under-billing by a factor of
# nearly two. The cost below is the listing's reading at 2026-09-10 12:40 UTC and is a fallback,
# not the price: refresh replaces it from GET /v1/models whenever live calls are on and that call
# comes back, and falls back to it when the call fails or the listing prices nothing.
BUILTIN_LOCAL_PROVIDERS: dict[str, dict] = {
    "cheaperinference": {
        "id": "cheaperinference",
        "name": "CheaperInference",
        "npm": "@ai-sdk/openai-compatible",
        "api": "https://api.cheaperinference.com/v1",
        "env": ["CHEAPER_INFERENCE_API_KEY"],
        "doc": "https://platform.cheaperinference.com/docs",
        "prices": {"url": "https://api.cheaperinference.com/v1/models", "field": "pricing"},
        "models": {
            "glm-5.3-flash": {
                "id": "glm-5.3-flash",
                "name": "GLM 5.3 Flash",
                "reasoning": True,
                "tool_call": True,
                "limit": {"context": 1_048_576, "output": 131_072},
                # Fallback rates, per 1M tokens, read 2026-09-10. Live rates win over them.
                "cost": {"input": 0.105, "output": 0.35, "cache_read": 0.01275,
                         "cache_write": 0.105},
            },
        },
    },
    # A provider models.dev does list, carrying the one model row it does not. models.dev names
    # deepseek-v4-flash; the live endpoint serves deepseek-flash, and only that id and
    # deepseek-v4-pro (GET https://api.deepseek.com/models, 2026-09-10), so a Run on the id the
    # vendor actually serves was billed nothing and failed the budget gate as an unpriced call.
    # The row is the vendor's own published price per 1M tokens: cache miss 0.30, cache hit 0.006,
    # output 1.20, context 1M (api-docs.deepseek.com/quick_start/pricing, read 2026-09-10). Those
    # are the peak rates; the page halves them off peak, and a ledger that must never under-bill
    # takes the higher of the two. The same page charges nothing to write a cache, which is what
    # the 0.0 cache_write says, and announces that from 2026-09-14 deepseek-v4-pro is routed to
    # V4.1 Flash and billed as V4.1 Flash, so a Run on that id after the 14th is billed at these
    # rates and not at the pro rates the snapshot carries. Prices checked 2026-09-10. There is no
    # prices url here: DeepSeek publishes its rates on that page and not in its model listing, so
    # this row is read by hand and dated rather than refreshed. Only the model row is given, so the
    # host, the key variable and every other model of the provider keep coming from models.dev.
    "deepseek": {
        "models": {
            "deepseek-flash": {
                "id": "deepseek-flash",
                "name": "DeepSeek Flash",
                "reasoning": True,
                "tool_call": True,
                "limit": {"context": 1_000_000, "output": 384_000},
                "cost": {"input": 0.30, "output": 1.20, "cache_read": 0.006, "cache_write": 0.0},
            },
        },
    },
}


def local_providers_path(path: Optional[str | Path] = None) -> Path:
    """Where the local provider registry lives: beside the snapshot, whichever one is in use."""
    return snapshot_path(path).with_name(LOCAL_PROVIDERS_NAME)


def local_providers(path: Optional[str | Path] = None) -> dict[str, dict]:
    """The providers models.dev does not list: the built-in rows, then the file's, which win.

    An unreadable or misshapen file is ignored rather than raised on: a broken side file must not
    take every model call down with it, and the built-in rows still answer.
    """
    merged = {name: copy.deepcopy(entry) for name, entry in BUILTIN_LOCAL_PROVIDERS.items()}
    file_path = local_providers_path(path)
    if not file_path.is_file():
        return merged
    try:
        stored = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return merged
    entries = stored.get("providers") if isinstance(stored, dict) and "providers" in stored else stored
    if not isinstance(entries, dict):
        return merged
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        usable = _usable_entry(entry)
        if usable is not None:
            merged[name] = _merge_provider(merged.get(name), usable)
    return merged


def _usable_entry(entry: dict) -> Optional[dict]:
    """One row of the file with what cannot be read dropped, or None when the row cannot be read.

    models has to be a mapping of wire id to row. A models field of any other shape is not a row
    with one bad field, it is a row written to a different shape, so the whole row is left out:
    landing half of it would point a provider at a new host while its prices still came from the
    row underneath. Every other provider in the file is unaffected. A models field that names
    nothing readable, empty or every row of it misshapen, goes the same way and for the same
    reason: there is nothing left to lay over, and the top-level fields alone would move the host
    and leave the prices where they were. A single misshapen row beside good ones is narrower, and
    only that wire id is dropped.

    A row with no models field at all is not this case. It says nothing about models and is meant
    to: it overrides a host or a key variable and leaves the rows underneath alone.
    """
    models = entry.get("models")
    if models is None:
        return entry
    if not isinstance(models, dict):
        return None
    rows = {wire_id: row for wire_id, row in models.items() if isinstance(row, dict)}
    if not rows:
        return None
    kept = {key: value for key, value in entry.items() if key != "models"}
    kept["models"] = rows
    return kept


def _laid_over(under: Any, over: Any) -> Any:
    """One value laid over another: mappings merged key by key, all the way down, anything else
    replaced.

    This is what lets a file correct one number. {"cost": {"input": 3.0}} moves the input rate and
    leaves the output rate, the cache rates, the window and the name where the catalog had them.
    Restating them to keep them would be a second place for them to go stale, and losing them would
    leave the model priced and unsized, or half priced, which the budget gate reads as a call it
    must refuse. A rate that went to nothing is written as 0.0 rather than left out, so no override
    ever needs to remove a key.

    A value only lands where it is written to the shape of the one underneath: a mapping over a
    mapping, a number over a number, a string over a string, a list over a list. A cost or a limit
    given as a string, at the mapping or at the rate inside it, is not a correction anyone can
    read: taking it would leave the model unpriced, which the budget gate reads as a call it must
    refuse, and unsized, which sends the context cap to the generic limit. The written value is
    the better answer of the two, so it stands, and the fields beside the unreadable one still
    land. A key with nothing underneath is new rather than wrong, so it is taken as written.
    """
    if isinstance(under, dict):
        if not isinstance(over, dict):
            return under
        merged = dict(under)
        for key, value in over.items():
            merged[key] = _laid_over(merged.get(key), value)
        return merged
    if under is None or _same_shape(under, over):
        return over
    return under


def _same_shape(under: Any, over: Any) -> bool:
    """Whether one value is written to the shape of another. Whole numbers and fractions count as
    one shape, since a rate of 3 and a rate of 3.0 are the same rate written two ways, and true
    and false are their own shape rather than the numbers Python also reads them as."""
    if isinstance(under, bool) or isinstance(over, bool):
        return isinstance(under, bool) and isinstance(over, bool)
    if isinstance(under, (int, float)) and isinstance(over, (int, float)):
        return True
    return isinstance(over, type(under))


def _merge_provider(under: Optional[dict], over: dict) -> dict:
    """One provider row laid over another, key by key, with the model rows merged by wire id.

    Either side can carry a models field of the wrong shape (the catalog underneath is whatever
    models.dev last served), and a merge that raises would take every model call down, so a
    models field that is not a mapping is passed over rather than read.
    """
    if not isinstance(under, dict):
        return copy.deepcopy(over)
    under_fields = copy.deepcopy(under)
    over_fields = copy.deepcopy(over)
    under_models = under_fields.pop("models", None)
    over_models = over_fields.pop("models", None)
    merged = _laid_over(under_fields, over_fields)
    models = under_models if isinstance(under_models, dict) else {}
    if isinstance(over_models, dict):
        for wire_id, row in over_models.items():
            models[wire_id] = _laid_over(models.get(wire_id), row)
    if models:
        merged["models"] = models
    return merged


def overlay_local(catalog: Optional[dict], path: Optional[str | Path] = None) -> Optional[dict]:
    """The catalog with the local provider registry laid over it, or the local rows alone.

    Returns None only when neither side names a provider, which is what a caller already reads as
    "nothing on disk yet".
    """
    local = local_providers(path)
    if not local:
        return catalog
    merged = dict(catalog or {})
    for name, entry in local.items():
        merged[name] = _merge_provider(merged.get(name), entry)
    return merged or None


def _read_snapshot(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict) or "catalog" not in stored:
        return None
    return stored


def _snapshot_age_days(stored: dict) -> Optional[float]:
    fetched_at = stored.get("fetched_at")
    if not isinstance(fetched_at, str):
        return None
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched
    return age.total_seconds() / 86400.0


def _get_json(client: Any, url: str, headers: Optional[dict[str, str]] = None) -> Any:
    """One GET, parsed. Any failure, network or parsing, returns None rather than raising.

    A header here can carry a provider key. It is never returned, stored or written anywhere: this
    module has no log for a key to reach, and the failure path answers None and not the request.
    """
    try:
        owns_client = client is None
        http_client = client if client is not None else httpx.Client()
        try:
            response = http_client.get(url, timeout=30.0, headers=headers or {})
            response.raise_for_status()
            return response.json()
        finally:
            if owns_client:
                http_client.close()
    except Exception:
        return None


def _fetch(client: Any) -> Optional[dict]:
    """One GET of the models.dev catalog, or None when it cannot be read."""
    catalog = _get_json(client, MODELS_DEV_URL)
    return catalog if isinstance(catalog, dict) else None


def _key_for(entry: dict, env: Optional[dict[str, str]]) -> tuple[str, str]:
    """The variable this row says its key lives in, and what holds it. Either can be empty."""
    names = entry.get("env")
    name = names[0] if isinstance(names, list) and names and isinstance(names[0], str) else ""
    values = os.environ if env is None else env
    return name, (str(values.get(name) or "") if name else "")


def _cost_numbers(cost: Any) -> Optional[dict[str, float]]:
    """One listed rate read as a cost row, or None when it is not one.

    input and output have to be there, since a row missing either prices nothing, and every value
    has to be a number that is not negative: a rate below zero is a field read wrong, not a
    discount, and a misread rate is worse than a dated one.
    """
    if not isinstance(cost, dict) or "input" not in cost or "output" not in cost:
        return None
    numbers: dict[str, float] = {}
    for key in ("input", "output", "cache_read", "cache_write"):
        if key not in cost:
            continue
        try:
            value = float(cost[key])
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        numbers[key] = value
    return numbers


def _listed_prices(listing: Any, field: str) -> dict[str, dict[str, float]]:
    """The rates a provider's model listing carries, by wire id.

    The listing is the OpenAI one every gateway here serves, {"data": [{"id": ..., "<field>": ...}]},
    and a bare list of rows is read the same way. A row that prices nothing is passed over rather
    than answered for, so one unreadable row does not cost the others their live rates.
    """
    rows = listing.get(LISTING_ROWS_FIELD) if isinstance(listing, dict) else listing
    if not isinstance(rows, list):
        return {}
    prices: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        wire_id = row.get("id")
        cost = _cost_numbers(row.get(field))
        if isinstance(wire_id, str) and wire_id and cost is not None:
            prices[wire_id] = cost
    return prices


# One reading of each price list per process. refresh runs wherever an id is resolved and not only
# where a call is billed, so a request per lookup would be a request per Run for a number that
# moves in days, and the ledger already reads its prices once and keeps them. A reading that came
# back with nothing is kept too: asking a host that is down again on the next lookup costs the
# timeout again and answers the same nothing.
_LIVE_PRICES_SEEN: dict[tuple[str, str], dict[str, dict[str, float]]] = {}


def forget_live_prices() -> None:
    """Drop this process's reading of every price list, so the next refresh asks the vendors again."""
    _LIVE_PRICES_SEEN.clear()


def _overlay_live_prices(catalog: Optional[dict], client: Any, env: Optional[dict[str, str]]) -> Optional[dict]:
    """Every provider row that names its own price list, repriced from that list.

    A written rate is stale the day the vendor moves it, and one gateway here moved by a factor of
    nearly two inside a day, so a row says where its prices live and this reads them. The row's own
    rates are the fallback and never the other way round: the call happens only when live calls are
    on, and an unreachable listing, an error, a renamed field or a rate that does not read as a
    number all leave the written rates standing. No model is added, so the listing can only change
    the price of a model the row already offers.

    A row whose key variable holds nothing is passed over without a request. Nobody can call a
    provider they hold no key for, so its rates price nothing, and a machine that only ever calls
    one vendor should not be reaching out to another's host to ask.
    """
    if not catalog or not live_calls_requested(env):
        return catalog
    for entry in catalog.values():
        source = entry.get(PRICES_FIELD) if isinstance(entry, dict) else None
        url = source.get("url") if isinstance(source, dict) else None
        field = source.get("field") if isinstance(source, dict) else None
        if not isinstance(url, str) or not url or not isinstance(field, str) or not field:
            continue
        key_var, key = _key_for(entry, env)
        if key_var and not key:
            continue
        headers = {"Authorization": f"Bearer {key}"} if key else None
        models = entry.get("models")
        listed = _LIVE_PRICES_SEEN.get((url, field))
        if listed is None:
            listed = _listed_prices(_get_json(client, url, headers), field)
            _LIVE_PRICES_SEEN[(url, field)] = listed
        for row_key, cost in _by_row_key(models, listed).items():
            models[row_key]["cost"] = cost
    return catalog


def _by_row_key(models: Any, listed: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Listed rates keyed by the model row each one names, with the disagreements left out.

    A gateway lists one model under two spellings, its own and the lab's, and both resolve to the
    same row here. When the two rates agree it does not matter which is read; when they disagree
    there is no telling which the wallet will be billed at, so the row keeps the rate written
    beside it, which is the rule the cross-provider price lookup already follows.
    """
    priced: dict[str, dict[str, float]] = {}
    disagreed: set[str] = set()
    for wire_id, cost in listed.items():
        row_key = model_row_key(models, wire_id)
        if row_key is None:
            continue
        if row_key in priced and priced[row_key] != cost:
            disagreed.add(row_key)
        priced[row_key] = cost
    return {row_key: cost for row_key, cost in priced.items() if row_key not in disagreed}


def refresh(
    client: Any = None,
    path: Optional[str | Path] = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    env: Optional[dict[str, str]] = None,
) -> Optional[dict]:
    """The models.dev catalog: fetched fresh when live calls are allowed and the snapshot is
    missing or older than max_age_days, otherwise read from the existing snapshot. A network
    error or a missing snapshot never raises; it falls back to whatever is already on disk, or
    to None when there is nothing to fall back to.

    The local provider registry is laid over whatever comes back, always and last, so a provider
    models.dev does not list is answered here rather than by a second lookup somewhere else. A row
    of that registry naming its own price list is then repriced from it, so a rate the vendor moved
    today is what the ledger bills at. Neither overlay is written into the snapshot: the snapshot
    stays what models.dev said.
    """
    snap_path = snapshot_path(path)
    stored = _read_snapshot(snap_path)
    catalog = stored.get("catalog") if stored is not None else None
    if live_calls_requested(env):
        age = _snapshot_age_days(stored) if stored else None
        stale = stored is None or age is None or age > max_age_days
        if stale:
            fetched = _fetch(client)
            if fetched is not None:
                wrapped = {"fetched_at": datetime.now(timezone.utc).isoformat(), "catalog": fetched}
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                snap_path.write_text(json.dumps(wrapped), encoding="utf-8")
                catalog = fetched
    return _overlay_live_prices(overlay_local(catalog, path), client, env)


# The npm adapters models.dev names for providers that speak the OpenAI request shape. A provider
# it lists under any other adapter (Google, Cohere, Vertex) takes a different body, so the registry
# says so rather than posting chat completions at it and reading the 400.
#
# @openrouter/ai-sdk-provider is here because OpenRouter's own reference says its endpoint is the
# OpenAI chat one: "POST /api/v1/chat/completions", "Authorization: Bearer <OPENROUTER_API_KEY>",
# a body of messages, tools and tool_choice, and choices[].message back
# (openrouter.ai/docs/api-reference/overview, read 2026-09-10). The npm package differs from
# @ai-sdk/openai-compatible only in the SDK layer above the wire, and it is the wire this table is
# about. A provider is added here after its docs are read, never because its name is familiar.
OPENAI_SHAPED = ("@ai-sdk/openai-compatible", "@ai-sdk/openai", "@openrouter/ai-sdk-provider")


class Endpoint(NamedTuple):
    """How to reach one provider: its host, the environment variable holding its key, and whether
    the request shape is one this Harness builds."""
    provider: str
    base_url: str
    key_env_var: str
    openai_shaped: bool
    adapter: str


def endpoint_from_catalog(catalog: Optional[dict], model_id: Optional[str]) -> Optional[Endpoint]:
    """The endpoint models.dev lists for a 'provider/model' id, or None when it lists no such
    provider or the provider names no host (Google and Vertex are reached through their own SDKs).

    This is the registry side of the snapshot the prices already come from: one catalog answers
    what a model costs, how much context it takes, and where to send the call.
    """
    provider, _, _wire = str(model_id or "").partition("/")
    entry = (catalog or {}).get(provider)
    if not isinstance(entry, dict):
        return None
    base_url = entry.get("api")
    if not isinstance(base_url, str) or not base_url:
        return None
    keys = entry.get("env")
    key_env_var = keys[0] if isinstance(keys, list) and keys and isinstance(keys[0], str) else ""
    adapter = str(entry.get("npm") or "")
    return Endpoint(provider, base_url, key_env_var, adapter in OPENAI_SHAPED, adapter)


def model_adapter_for(catalog: Optional[dict], model_id: Optional[str]) -> str:
    """The npm adapter for one MODEL: its own row's provider.npm wins over the provider entry's.

    The provider-level field cannot say that minimax-m3 rides opencode-go but speaks the
    Anthropic shape; the model row can, and does. Empty when the catalog names neither."""
    provider, _, wire = str(model_id or "").partition("/")
    entry = (catalog or {}).get(provider)
    if not isinstance(entry, dict):
        return ""
    models = entry.get("models")
    row = models.get(wire) if isinstance(models, dict) else None
    if isinstance(row, dict):
        override = row.get("provider")
        if isinstance(override, dict):
            npm = override.get("npm")
            if isinstance(npm, str) and npm:
                return npm
    npm = entry.get("npm")
    return npm if isinstance(npm, str) else ""


def model_row(provider_entry: Any, wire_id: str) -> Optional[dict]:
    """One provider's row for a wire id: its own key, else the one whose last segment matches.

    A gateway does not always answer under the name it was called by. One prices glm-5.3-flash and
    answers z-ai/glm-5.3-flash, and the ledger keys a call on the id the endpoint echoed, so the
    row has to be found from either spelling or the call is billed nothing. The match is inside one
    provider's own catalog, on the last slash-separated segment, and only when exactly one row
    matches: two rows ending the same way is a disagreement, and a disagreement is no price, which
    is the rule the cross-provider lookup below already follows.
    """
    models = provider_entry.get("models") if isinstance(provider_entry, dict) else None
    key = model_row_key(models, wire_id)
    return models.get(key) if key is not None else None


def model_row_key(models: Any, wire_id: str) -> Optional[str]:
    """Which key of a provider's models a wire id names, under the rule model_row is written to.

    The key rather than the row, for a caller that has to write back into the mapping and cannot
    tell two rows apart by value.
    """
    if not isinstance(models, dict) or not wire_id:
        return None
    if isinstance(models.get(wire_id), dict):
        return wire_id
    tail = wire_id.rpartition("/")[2]
    found = [key for key, candidate in models.items()
             if isinstance(candidate, dict) and key.rpartition("/")[2] == tail]
    return found[0] if len(found) == 1 else None


def window_from_catalog(catalog: Optional[dict], model_id: Optional[str]) -> Optional[int]:
    """The context window models.dev lists for a model, for the D65 cap."""
    provider, _, wire_id = str(model_id or "").partition("/")
    entry = (catalog or {}).get(provider) if wire_id else None
    if not isinstance(entry, dict):
        return None
    model_entry = model_row(entry, wire_id)
    limit = model_entry.get("limit") if isinstance(model_entry, dict) else None
    context = limit.get("context") if isinstance(limit, dict) else None
    try:
        return int(context) if context else None
    except (TypeError, ValueError):
        return None


def _price_from_provider(provider_entry: Any, wire_id: str) -> Optional[dict[str, float]]:
    model_entry = model_row(provider_entry, wire_id)
    if model_entry is None:
        return None
    cost = model_entry.get("cost")
    if not isinstance(cost, dict) or "input" not in cost or "output" not in cost:
        return None
    try:
        input_price = float(cost["input"])
        output_price = float(cost["output"])
        cache_read = float(cost["cache_read"]) if "cache_read" in cost else input_price
        cache_write = float(cost["cache_write"]) if "cache_write" in cost else 0.0
    except (TypeError, ValueError):
        return None
    return {"input": input_price, "output": output_price, "cache_read": cache_read, "cache_write": cache_write}


def price_from_catalog(catalog: Optional[dict], model_id: Optional[str]) -> Optional[dict[str, float]]:
    """Prices for a full 'provider/model' id, or for the wire id alone, from a fetched catalog.

    models.dev's own prices are per 1M tokens, the same unit budget.PRICES uses (checked by
    hand: anthropic/claude-opus-5 comes back {"input": 5, "output": 25, "cache_read": 0.5,
    "cache_write": 6.25}, matching PRICES exactly), so nothing here converts.

    A missing cache_write is 0.0 (OpenAI charges nothing to write a cache) unless the catalog
    says otherwise; a missing cache_read is billed at the input price. Modes under
    "experimental.modes" are ignored: only the model's own top-level "cost" is read.
    """
    if not catalog or not model_id:
        return None
    if "/" in model_id:
        provider_id, _, wire_id = model_id.partition("/")
        price = _price_from_provider(catalog.get(provider_id), wire_id)
        if price is not None:
            return price
    # A bare wire id is only a price when the catalog agrees on one. Resellers list the frontier
    # models under their own names at their own rates ('gpt-5.6-luna' appears under ten of them,
    # five times OpenAI's price), and taking whichever the scan met first made the harness report
    # a build costing five times what it cost. Disagreement is unpriced, which the ledger already
    # counts as unpriced_calls, because a wrong price is worse than a known missing one.
    found = [price for provider_entry in catalog.values()
             if (price := _price_from_provider(provider_entry, model_id)) is not None]
    if found and all(price == found[0] for price in found):
        return found[0]
    return None
