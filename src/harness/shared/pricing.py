"""The live price source: models.dev's catalog, snapshotted to disk (D116).

budget.py's hand-kept PRICES table is the offline fallback; this module is what keeps prices
current without a person re-checking a vendor's page by hand. The one way this module leaves
the machine is the same switch every other adapter uses: harness.shared.provider.LIVE_ENV_VAR.
With that off (the default, and always off in tests), refresh() only ever reads the snapshot
already on disk, or returns None when there is none.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from harness.shared.provider import LIVE_ENV_VAR

MODELS_DEV_URL = "https://models.dev/api.json"

# models.dev's own prices are per 1M tokens, the same unit budget.PRICES uses (checked by
# hand: anthropic/claude-opus-5 comes back {"input": 5, "output": 25, "cache_read": 0.5,
# "cache_write": 6.25}, matching PRICES exactly), so price_from_catalog does no conversion.
DEFAULT_MAX_AGE_DAYS = 7


def snapshot_path(path: Optional[str | Path] = None) -> Path:
    """Where the catalog snapshot lives on disk. Overridable so tests never touch the real one."""
    if path is not None:
        return Path(path)
    return Path.home() / ".cache" / "harness" / "models.dev.json"


def _is_live(env: Optional[dict[str, str]]) -> bool:
    values = os.environ if env is None else env
    return str(values.get(LIVE_ENV_VAR, "")).strip().lower() in ("1", "true", "yes", "on")


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
    """
    snap_path = snapshot_path(path)
    stored = _read_snapshot(snap_path)
    if _is_live(env):
        age = _snapshot_age_days(stored) if stored else None
        stale = stored is None or age is None or age > max_age_days
        if stale:
            catalog = _fetch(client)
            if catalog is not None:
                wrapped = {"fetched_at": datetime.now(timezone.utc).isoformat(), "catalog": catalog}
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                snap_path.write_text(json.dumps(wrapped), encoding="utf-8")
                return catalog
    if stored is not None:
        return stored.get("catalog")
    return None


def _price_from_provider(provider_entry: Any, wire_id: str) -> Optional[dict[str, float]]:
    if not isinstance(provider_entry, dict):
        return None
    models = provider_entry.get("models")
    if not isinstance(models, dict):
        return None
    model_entry = models.get(wire_id)
    if not isinstance(model_entry, dict):
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
    for provider_entry in catalog.values():
        price = _price_from_provider(provider_entry, model_id)
        if price is not None:
            return price
    return None
