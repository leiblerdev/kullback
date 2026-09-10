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
LOCAL_PROVIDERS_NAME = "providers.local.json"

# Providers the Harness ships knowing about because their docs were read, kept in code so a fresh
# machine has them without a file to write, and overridable by the file above. CheaperInference is
# the first: an OpenAI-compatible gateway, "The API is served from https://api.cheaperinference.com/v1.
# Authenticate with `Authorization: Bearer ci_live_...`", integrations that read the key from
# CHEAPER_INFERENCE_API_KEY, and "@ai-sdk/openai-compatible" as the SDK its own docs name
# (platform.cheaperinference.com/llms.txt and /docs, read 2026-09-10). The model row is copied from
# that day's GET /v1/models, which prices every model it lists: the rates the gateway charges, which
# is what the wallet is billed, rather than the list rates it discounts from.
BUILTIN_LOCAL_PROVIDERS: dict[str, dict] = {
    "cheaperinference": {
        "id": "cheaperinference",
        "name": "CheaperInference",
        "npm": "@ai-sdk/openai-compatible",
        "api": "https://api.cheaperinference.com/v1",
        "env": ["CHEAPER_INFERENCE_API_KEY"],
        "doc": "https://platform.cheaperinference.com/docs",
        "models": {
            "glm-5.3-flash": {
                "id": "glm-5.3-flash",
                "name": "GLM 5.3 Flash",
                "reasoning": True,
                "tool_call": True,
                "limit": {"context": 1_048_576, "output": 131_072},
                "cost": {"input": 0.060426, "output": 0.201421, "cache_read": 0.012085,
                         "cache_write": 0.060426},
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
    # takes the higher of the two. Only the model row is given, so the host, the key variable and
    # every other model of the provider keep coming from models.dev.
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
    row underneath. Every other provider in the file is unaffected. A single model row of the
    wrong shape is narrower, and only that wire id is dropped.
    """
    models = entry.get("models")
    if models is None:
        return entry
    if not isinstance(models, dict):
        return None
    rows = {wire_id: row for wire_id, row in models.items() if isinstance(row, dict)}
    kept = {key: value for key, value in entry.items() if key != "models"}
    if rows:
        kept["models"] = rows
    return kept


def _merge_provider(under: Optional[dict], over: dict) -> dict:
    """One provider row laid over another: top-level fields replaced, model rows merged by id.

    Either side can carry a models field of the wrong shape (the catalog underneath is whatever
    models.dev last served), and a merge that raises would take every model call down, so a
    models field that is not a mapping is passed over rather than read.
    """
    if not isinstance(under, dict):
        return copy.deepcopy(over)
    merged = {**copy.deepcopy(under), **copy.deepcopy(over)}
    under_models = copy.deepcopy(under.get("models"))
    over_models = copy.deepcopy(over.get("models"))
    models = under_models if isinstance(under_models, dict) else {}
    if isinstance(over_models, dict):
        models.update(over_models)
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


def _fetch(client: Any) -> Optional[dict]:
    """One GET of the catalog. Any failure, network or parsing, returns None rather than raising."""
    try:
        owns_client = client is None
        http_client = client if client is not None else httpx.Client()
        try:
            response = http_client.get(MODELS_DEV_URL, timeout=30.0)
            response.raise_for_status()
            catalog = response.json()
        finally:
            if owns_client:
                http_client.close()
    except Exception:
        return None
    if not isinstance(catalog, dict):
        return None
    return catalog


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
    models.dev does not list is answered here rather than by a second lookup somewhere else. The
    overlay is never written into the snapshot: the snapshot stays what models.dev said.
    """
    snap_path = snapshot_path(path)
    stored = _read_snapshot(snap_path)
    if live_calls_requested(env):
        age = _snapshot_age_days(stored) if stored else None
        stale = stored is None or age is None or age > max_age_days
        if stale:
            catalog = _fetch(client)
            if catalog is not None:
                wrapped = {"fetched_at": datetime.now(timezone.utc).isoformat(), "catalog": catalog}
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                snap_path.write_text(json.dumps(wrapped), encoding="utf-8")
                return overlay_local(catalog, path)
    return overlay_local(stored.get("catalog") if stored is not None else None, path)


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
    if not isinstance(models, dict) or not wire_id:
        return None
    row = models.get(wire_id)
    if isinstance(row, dict):
        return row
    tail = wire_id.rpartition("/")[2]
    found = [candidate for key, candidate in models.items()
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
