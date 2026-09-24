"""Cost and token accounting: per call, per stage, per build, plus the D65 context cap and the D86 spend ceiling."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from kullback.ai import cache as cache_module
from kullback.ai import pricing as pricing_module
from kullback.ai.model_limits import catalog_candidates, without_profile
from kullback.ai.provider import Model, ModelConfig, ModelReply
from kullback.runner import feed
from kullback.runner.records import Cost, Event, Usage

# --- price table -------------------------------------------------------------
# Data, not code. US dollars per 1M tokens. cache_read is what a cache hit costs,
# cache_write what writing the cache costs.
#
# UPDATE ME: this table is now the offline fallback, not the only source. price_for() asks
# kullback.ai.pricing's models.dev snapshot first, since a vendor's list price changes
# without warning; this table is what prices a call when there is no snapshot (no network
# allowed, or nothing fetched yet) or when models.dev has no row for the model. Keep it
# checked by hand on the date below anyway, and add an entry for every model a build
# actually calls: a model priced by neither source is not priced at zero quietly, its calls
# are counted under unpriced_calls in the totals file and the report shows that count.
PRICES_CHECKED = "2026-09-23"
PRICES_NOTE = (
    "list prices per 1M tokens, checked by hand on "
    + PRICES_CHECKED
    + "; update me before trusting a build's cost, and add the models you call"
)
PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25},
    "anthropic/claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    "anthropic/claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
    # First-party Opus 5.5: the claude-api reference, model-migration.md line 2021 (4 / 20, cache
    # reads 0.20, five-minute cache writes 5.00).
    "anthropic/claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0},
    # Claude in Amazon Bedrock (provider.BedrockAnthropicModel). AWS's price list for us-east-2
    # (pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/
    # us-east-2/index.json, published 2026-09-22, read 2026-09-24), the "Standard, Global" rows,
    # cache_write at the five-minute TTL. The same list prices the in-region rows ten percent
    # higher. models.dev lists each profile as its own row under amazon-bedrock, and price_for asks
    # it for the exact id first; these rows are the offline fallback, looked up without the profile
    # (without_profile), so a us. or eu. call priced from here is priced a little low.
    "bedrock/anthropic.claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0},
    "bedrock/anthropic.claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25},
    "bedrock/anthropic.claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    "bedrock/anthropic.claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
    # GPT-6 Sol on Bedrock's Chat Completions path (provider.BedrockOpenAIModel): models.dev,
    # amazon-bedrock/global.openai.gpt-6-sol (read 2026-09-24), the tier under 272,000 prompt tokens
    # (4 and 15 above it, which prompt_tier_limit keeps the D65 cap out of).
    "bedrock/openai.gpt-6-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5,
                                 "prompt_tier_limit": 272_000},
    # The first slice's Candidates (design section 11 step 6). OpenAI has no cache-write
    # charge, so cache_write is 0 and cached input is billed at cache_read.
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_write": 0.0},
    "openai/o4-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.275, "cache_write": 0.0},
    # OpenCode Go's Muse Spark 1.3 speaks the Responses API and models.dev does not list it
    # yet, so the fallback table carries the Go docs' price: without a row the spend ceiling
    # cannot see the model at all.
    "opencode-go/muse-spark-1.3-contributor": {"input": 0.10, "output": 0.20, "cache_read": 0.002,
                                                "cache_write": 0.0},
    # The harness default model until 2026-09-24, still a choice. OpenRouter's catalogue
    # (api/v1/models, read 2026-09-23), per-token prices times 1M, the tier below 272,000 prompt
    # tokens; a longer prompt is billed at twice the input and one and a half times the output,
    # so prompt_tier_limit keeps the D65 cap inside the priced tier (window_for).
    "openai/gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125,
                          "prompt_tier_limit": 272_000},
    # models.dev's snapshot lacks it. OpenRouter's catalogue (api/v1/models, read 2026-09-23),
    # per-token prices times 1M, the tier below 272,000 prompt tokens (prompt_tier_limit).
    "openai/gpt-6-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5,
                         "prompt_tier_limit": 272_000},
}

# What fits in one call, per model, for the D65 cap. A model with no row uses the default.
DEFAULT_CONTEXT_WINDOW = 200_000
CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-5": 1_000_000,
    "anthropic/claude-sonnet-5": 1_000_000,
    "anthropic/claude-haiku-4-5": 200_000,
    "anthropic/claude-opus-5-5": 1_000_000,
    # The same models on Bedrock; Haiku 4.5 keeps its 200,000 there too (models.dev, amazon-bedrock).
    "bedrock/anthropic.claude-opus-5-5": 1_000_000,
    "bedrock/anthropic.claude-opus-5": 1_000_000,
    "bedrock/anthropic.claude-sonnet-5": 1_000_000,
    "bedrock/anthropic.claude-haiku-4-5": 200_000,
    # models.dev, amazon-bedrock/global.openai.gpt-6-sol, limit.context (read 2026-09-24).
    "bedrock/openai.gpt-6-sol": 1_050_000,
    "openai/gpt-4.1-mini": 1_000_000,
    "openai/o4-mini": 200_000,
    # Measured, not read off a page: a 500,000 token prompt was accepted, so this is a floor and
    # the real window is at least this. It is here because the 200,000 default put the D65 cap at
    # 80,000 tokens and refused a compile_tools prompt the model would have taken.
    "openai/gpt-5.6-luna": 400_000,
    "openai/gpt-5.6-sol": 400_000,
    "openai/gpt-5.6-terra": 400_000,
    # The harness default model: context_length in OpenRouter's catalogue (api/v1/models, read
    # 2026-09-23).
    "openai/gpt-6-luna": 1_050_000,
    # context_length in OpenRouter's catalogue (api/v1/models, read 2026-09-23).
    "openai/gpt-6-sol": 1_050_000,
}

TOTALS_NAME = "budget.json"
# One ledger, one lock (D118): the Builder's stages call the model from a few threads at once,
# and record_call is a read-modify-write of budget.json. Process-wide rather than per model,
# since every wrapper in a build writes the same file.
_LEDGER_LOCK = threading.RLock()
# memo_hits: how many of "calls" were served from provider.MemoModel's on-disk memo rather than
# the network. A hit's usage is zeroed at the source, so it already costs 0.00 in the other
# fields; this is only the count, so a stage's cache effectiveness is a number, not a feeling.
# models_dev_calls: how many priced calls got their price from the models.dev snapshot rather
# than the PRICES table below, so a stale table is visible in the totals file, not just guessed.
# cache_saved_usd: what the provider's prompt cache took off the bill, net: the cache-read tokens
# at the input rate minus what they cost at the cache rate, less the premium cache writes carry
# over plain input where a vendor charges one. usd is what was paid; usd + cache_saved_usd is
# what the same calls would have cost with no cache. A memo hit is a count only: nothing was
# sent, so nothing is known about what it would have cost.
BUCKET_FIELDS = ("calls", "input", "output", "cache_read", "cache_write", "usd", "wall_ms",
                 "unpriced_calls", "memo_hits", "models_dev_calls", "cache_saved_usd")
CONTEXT_CAP_FRACTION = 0.40
# Tokens are estimated from characters before a call, because the count endpoint is itself a
# call. Four characters per token is the usual English ratio and errs on the low side.
CHARS_PER_TOKEN = 4


class BudgetError(RuntimeError):
    """Base for the two refusals this module makes."""


class ContextCapExceeded(BudgetError):
    """A Builder call whose prompt is over the D65 share of the window; refused, not compacted."""

    def __init__(self, tokens: int, cap: int, window: int):
        self.tokens = tokens
        self.cap = cap
        self.window = window
        super().__init__(
            f"call refused: {tokens} prompt tokens is over the cap of {cap} "
            f"({int(round(cap / window * 100))} percent of a {window} token window)"
        )


class UnpricedModel(BudgetError):
    """A ceilinged build asked for a model the price table does not have.

    Its calls would cost 0.00, so the ceiling could never be reached and the build could
    spend without limit while reporting nothing spent. Refused at the start instead.
    """

    def __init__(self, model_id: Optional[str]):
        self.model_id = model_id
        super().__init__(
            f"no price for {model_id!r}, so a spend ceiling cannot hold; "
            f"add it to budget.PRICES (checked {PRICES_CHECKED}) or run without a ceiling"
        )


class BudgetExceeded(BudgetError):
    """The per-build spend ceiling (D86): stop where you are, report as is, ask before continuing."""

    def __init__(self, stage: str, item: str, spent: float, ceiling_usd: float, estimate_to_finish: float):
        self.stage = stage
        self.item = item
        self.spent = spent
        self.ceiling_usd = ceiling_usd
        self.estimate_to_finish = estimate_to_finish
        super().__init__(
            f"spend ceiling reached in stage {stage} on {item}: "
            f"spent {spent:.4f} USD of {ceiling_usd:.4f}, about {estimate_to_finish:.4f} USD left to finish"
        )


def _lookup(table: dict[str, Any], model_id: Optional[str]) -> Optional[Any]:
    """A table keyed by full 'provider/model' id, matched by the full id or by the wire id alone."""
    if not model_id:
        return None
    if model_id in table:
        return table[model_id]
    for key, value in table.items():
        if key.split("/", 1)[-1] == model_id:
            return value
    return None


# The models.dev snapshot, loaded and cached once per process: price_for is on the hot path
# of every call and must not re-read a file each time. _SNAPSHOT_PATH is None for the real
# default (kullback.ai.pricing.snapshot_path()); tests monkeypatch it to a tmp path so no
# test ever reads or writes the real ~/.cache/harness/models.dev.json.
_SNAPSHOT_PATH: Optional[Path] = None
_CATALOG_LOADED = False
_CATALOG: Optional[dict] = None


def _price_catalog() -> Optional[dict]:
    """models.dev's catalog for this process. No network unless live calls are allowed: with
    them off (the default, and always off in tests), this only ever reads a snapshot already
    on disk, if any."""
    global _CATALOG_LOADED, _CATALOG
    if not _CATALOG_LOADED:
        from kullback.ai import provider as provider_module
        env = os.environ if provider_module.ALLOW_MODEL_REQUESTS else {}
        _CATALOG = pricing_module.refresh(path=_SNAPSHOT_PATH, env=env)
        _CATALOG_LOADED = True
    return _CATALOG


# What price_source says for a catalog price; a fallback step is appended (`models.dev:vendor`).
MODELS_DEV_SOURCE = "models.dev"


def _price_from_models_dev(model_id: Optional[str]) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """The first catalog candidate (model_limits.catalog_candidates) that prices the model, and
    the step that answered."""
    catalog = _price_catalog()
    for step, candidate in catalog_candidates(model_id):
        price = pricing_module.price_from_catalog(catalog, candidate)
        if price is not None:
            return price, step
    return None, None


def _price_from_table(model_id: Optional[str]) -> Optional[dict[str, float]]:
    return _lookup(PRICES, model_id) or _lookup(PRICES, without_profile(model_id))


def _resolve_price(model_id: Optional[str]) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """A price and where it came from, in the one order every lookup follows.

    models.dev first, since a vendor's list price changes without warning: the exact wire id under
    the provider's catalog name (a Bedrock profile row carries its own price, the us. one ten
    percent over the global. one), then the id without its profile, then the vendor's own row.
    PRICES is the offline fallback for a model or a machine without a snapshot.
    """
    price, step = _price_from_models_dev(model_id)
    if price is not None:
        return price, MODELS_DEV_SOURCE if step == "exact" else f"{MODELS_DEV_SOURCE}:{step}"
    price = _price_from_table(model_id)
    return (price, "table") if price is not None else (None, None)


def price_for(model_id: Optional[str]) -> Optional[dict[str, float]]:
    """Prices for a full 'provider/model' id, or for the wire id alone (see _resolve_price)."""
    return _resolve_price(model_id)[0]


def price_source(model_id: Optional[str]) -> Optional[str]:
    """Which lookup answered price_for: "models.dev", "models.dev:without-profile",
    "models.dev:vendor" or "table". None when nothing prices the model."""
    return _resolve_price(model_id)[1]


def is_priced(model_id: Optional[str]) -> bool:
    return price_for(model_id) is not None


def _window_from_models_dev(model_id: Optional[str]) -> Optional[int]:
    catalog = _price_catalog()
    for _step, candidate in catalog_candidates(model_id):
        window = pricing_module.window_from_catalog(catalog, candidate)
        if window:
            return window
    return None


def window_for(model_id: Optional[str]) -> int:
    """The context window of a model, for the D65 cap.

    The hand table first, since two of its rows were measured against a live endpoint rather than
    read off a page, then models.dev in the order prices follow (_resolve_price), then the default.
    A model the registry knows no longer takes the 200,000 default and a cap four fifths smaller
    than the one it could have had.

    A price row that carries `prompt_tier_limit` is priced only for prompts under that many
    tokens, so the window is cut to the limit over CONTEXT_CAP_FRACTION: the D65 cap never lets
    a prompt past the priced tier, where the ledger would bill it at the lower rate.
    """
    window = (_lookup(CONTEXT_WINDOWS, model_id) or _lookup(CONTEXT_WINDOWS, without_profile(model_id))
              or _window_from_models_dev(model_id)
              or DEFAULT_CONTEXT_WINDOW)
    tier_limit = (_price_from_table(model_id) or {}).get("prompt_tier_limit")
    if tier_limit:
        return min(window, int(tier_limit / CONTEXT_CAP_FRACTION))
    return window


def priced_model_id(cost: Any) -> Optional[str]:
    """The id a call's price is looked up under: 'provider/model' whenever the provider is known.

    `Cost.model` is `reply.model`, the wire id the endpoint echoes back ('gpt-5.6-luna'), and a
    wire id on its own is not a model: models.dev carries that same name under a dozen resellers
    at their own prices, and the first one a scan meets billed a real build at five times
    OpenAI's rate. The provider is recorded on the Cost beside the wire id, so it is used.
    """
    model = getattr(cost, "model", None)
    provider = getattr(cost, "provider", None)
    if model and provider and "/" not in model:
        return f"{provider}/{model}"
    return model


def sent_wire_id(model_id: Optional[str], echoed: Optional[str]) -> Optional[str]:
    """The wire id a call is priced and recorded under: the one it was sent with.

    An endpoint echoes a shorter name than it was called by (Bedrock answers
    `global.anthropic.claude-opus-5-5` with `claude-opus-5-5`), and the profile is what picks the
    catalog row, so the configured id wins whenever it names its provider; the echo is used only
    when nothing else names the model.
    """
    if model_id and "/" in model_id:
        return model_id.split("/", 1)[1]
    return echoed or model_id


# A one-hour cache write costs twice the input price where a five-minute one costs one and a
# quarter times (prompt-caching.md, Economics; cost-optimization.md). A price row may carry its own
# `cache_write_1h`; a row without one is priced at this multiple of its input rate.
CACHE_WRITE_1H_INPUT_MULTIPLE = 2.0


def write_1h_rate(price: dict[str, float]) -> float:
    """The price per 1M tokens of a one-hour cache write under one price row."""
    return float(price.get("cache_write_1h", price["input"] * CACHE_WRITE_1H_INPUT_MULTIPLE))


def _write_cost(usage: Usage, price: dict[str, float]) -> float:
    """The cache writes of one call at their rates: five-minute at cache_write, one-hour at its own."""
    return ((usage.cache_write - usage.cache_write_1h) * price["cache_write"]
            + usage.cache_write_1h * write_1h_rate(price))


def call_cost(usage: Usage, model_id: Optional[str]) -> float:
    """What one call cost.

    Each count is billed at its own rate, with no arithmetic between them: Usage.input is the
    input neither read from nor written to the cache everywhere in the Harness, which is what
    Anthropic reports directly and what the OpenAI adapter subtracts down to. Subtracting the
    cached counts from input again under-billed every cached call. The one-hour part of the
    writes is billed at the one-hour rate.
    """
    price = price_for(model_id)
    if price is None:
        return 0.0
    return (
        usage.input * price["input"]
        + usage.output * price["output"]
        + usage.cache_read * price["cache_read"]
        + _write_cost(usage, price)
    ) / 1_000_000


def cache_effect(usage: Usage, model_id: Optional[str]) -> float:
    """What the cache changed on this call's bill, in dollars, positive when it saved money.

    Cache-read tokens are billed at the cache rate instead of the input rate; cache-write tokens
    are billed at the write rate, which some vendors set above input, and a one-hour write at its
    own higher rate. The two together are the cache's effect, so a call that wrote a cache nobody
    read shows as a cost, not a saving. An unpriced model has no rates, so its effect is 0.0.
    """
    price = price_for(model_id)
    if price is None:
        return 0.0
    return (
        usage.cache_read * (price["input"] - price["cache_read"])
        - (_write_cost(usage, price) - usage.cache_write * price["input"])
    ) / 1_000_000


def cache_line(bucket: dict) -> str:
    """One line a report prints beside the dollars: what the cache did to them."""
    saved = float(bucket.get("cache_saved_usd") or 0.0)
    paid = float(bucket.get("usd") or 0.0)
    reads = int(bucket.get("cache_read") or 0)
    hits = int(bucket.get("memo_hits") or 0)
    return (f"the cache saved ${saved:,.4f}: {reads:,} cache-read tokens at the cache rate, "
            f"{hits:,} memo hits sent nothing; without it ${paid + saved:,.4f}")


def empty_bucket() -> dict[str, float]:
    return {field: 0 for field in BUCKET_FIELDS}


def empty_totals() -> dict:
    return {"stages": {}, "total": empty_bucket()}


def totals_path(workdir: str | Path) -> Path:
    return Path(workdir) / TOTALS_NAME


def load_totals(workdir: str | Path) -> dict:
    """The build's ledger. Every bucket is filled out over an empty one, so a file written
    before a field existed still loads and resumes instead of raising a KeyError."""
    path = totals_path(workdir)
    if not path.exists():
        return empty_totals()
    stored = json.loads(path.read_text(encoding="utf-8"))
    totals = {"stages": {}, "total": {**empty_bucket(), **(stored.get("total") or {})}}
    for stage, bucket in (stored.get("stages") or {}).items():
        totals["stages"][stage] = {**empty_bucket(), **(bucket or {})}
    return totals


def save_totals(workdir: str | Path, totals: dict) -> Path:
    path = totals_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_call(
    event: Event,
    stage: str,
    workdir: str | Path,
    ceiling: Optional["Ceiling"] = None,
    item: str = "",
    items_left: int = 0,
    memo_hit: bool = False,
    request: Optional[dict] = None,
) -> Event:
    """Price one model-call Event, write the cost back onto it, and add it to the stage and build totals.

    With a ceiling, the same write charges it: the file is the one ledger, so the ceiling's
    spend and the report's cost so far cannot drift apart. `memo_hit` only adds to the count in
    `memo_hits`; a hit already carries zeroed usage, so it prices at 0.00 through the usual fields.
    `request` is the call's cache fingerprint (kullback.ai.cache.fingerprint); it rides the feed line
    so a miss on an unchanged prefix is a row anyone can count (`kullback budget cache`).
    """
    if event.cost is None:
        return event
    usage = event.cost.usage
    priced_id = priced_model_id(event.cost)
    event.cost.usd = call_cost(usage, priced_id)
    source = price_source(priced_id)
    event.cost.price_source = source
    saved = cache_effect(usage, priced_id)
    with _LEDGER_LOCK:
        return _record_locked(event, usage, source, stage, workdir, ceiling, item, items_left, memo_hit,
                              saved, request)


def _record_locked(event: Event, usage: Usage, source: Optional[str], stage: str, workdir: str | Path,
                   ceiling: Optional["Ceiling"], item: str, items_left: int, memo_hit: bool,
                   saved: float = 0.0, request: Optional[dict] = None) -> Event:
    totals = load_totals(workdir)
    bucket = totals["stages"].setdefault(stage, empty_bucket())
    for target in (bucket, totals["total"]):
        target["calls"] += 1
        target["input"] += usage.input
        target["output"] += usage.output
        target["cache_read"] += usage.cache_read
        target["cache_write"] += usage.cache_write
        target["usd"] += event.cost.usd
        target["cache_saved_usd"] += saved
        target["wall_ms"] += event.cost.wall_ms
        if source is None:
            target["unpriced_calls"] += 1
        elif source.startswith(MODELS_DEV_SOURCE):
            target["models_dev_calls"] += 1
        if memo_hit:
            target["memo_hits"] += 1
    save_totals(workdir, totals)
    # The same numbers, once as state and once as story: the ledger says what the build has spent,
    # the feed says what it just did. Written here, under the ledger lock, so the feed's order is
    # the order the calls were recorded in and a watcher can never read a call the ledger has not
    # counted. feed.append never raises (see feed.py).
    cache_fields = {"cache_write": usage.cache_write, **_request_fields(request, memo_hit)}
    if usage.cache_write_1h:
        cache_fields["cache_write_1h"] = usage.cache_write_1h
    feed.append(workdir, "model_call", stage=stage, model=priced_model_id(event.cost),
                input=usage.input, output=usage.output, cache_read=usage.cache_read,
                usd=event.cost.usd, wall_ms=event.cost.wall_ms, item=item or None,
                calls=totals["total"]["calls"], spend_usd=totals["total"]["usd"],
                cache_saved_usd=totals["total"]["cache_saved_usd"], **cache_fields)
    if ceiling is not None:
        ceiling.charge_recorded(totals, stage, item or str(event.idx), items_left)
    return event


# The ledger fields a price decides; reprice rewrites these and leaves counts and tokens alone.
PRICED_FIELDS = ("usd", "cache_saved_usd", "unpriced_calls", "models_dev_calls")


def reprice(workdir: str | Path, model_id: Optional[str] = None) -> dict:
    """Price a workdir's calls again from its feed and write the result into budget.json.

    Each model_call line carries its tokens and the id it was recorded under, so a round that ran
    unpriced (a price row missing, a catalog name wrong) has its cost recovered after the fact.
    `model_id` prices every line under one id instead, for a feed whose lines recorded a stripped id
    no catalog row answers to. Only the priced fields change; a stage with no feed line keeps its.
    """
    rows, _ = feed.read_since(workdir)
    priced: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("kind") != "model_call":
            continue
        usage = Usage(input=int(row.get("input") or 0), output=int(row.get("output") or 0),
                      cache_read=int(row.get("cache_read") or 0), cache_write=int(row.get("cache_write") or 0),
                      cache_write_1h=int(row.get("cache_write_1h") or 0))
        name = model_id or row.get("model")
        source = price_source(name)
        bucket = priced.setdefault(str(row.get("stage") or ""), dict.fromkeys(PRICED_FIELDS, 0))
        bucket["usd"] += call_cost(usage, name)
        bucket["cache_saved_usd"] += cache_effect(usage, name)
        bucket["unpriced_calls"] += source is None
        bucket["models_dev_calls"] += bool(source and source.startswith(MODELS_DEV_SOURCE))
    with _LEDGER_LOCK:
        totals = load_totals(workdir)
        for stage, fields in priced.items():
            totals["stages"].setdefault(stage, empty_bucket()).update(fields)
        for field in PRICED_FIELDS:
            totals["total"][field] = sum(bucket[field] for bucket in totals["stages"].values())
        save_totals(workdir, totals)
    return totals


# The fingerprint keys a feed line carries, as kullback.ai.cache.fingerprint names them.
REQUEST_FIELDS = ("prefix_hash", "history_hash", "parent_hash", "n_messages")


def _request_fields(request: Optional[dict], memo_hit: bool) -> dict:
    """The fingerprint as feed fields; a memo hit is marked, since it sent nothing to cache."""
    fields = {key: request[key] for key in REQUEST_FIELDS if request and request.get(key) is not None}
    if memo_hit:
        fields["memo_hit"] = True
    return fields


def subscriber(workdir: str | Path, stage: str, model_id: Optional[str],
               ceiling: Optional[Ceiling] = None) -> Any:
    """A harness subscriber that prices every assistant message the core streams into budget.json.

    The agent core streams its model and never calls BudgetedModel.query, so its calls are priced
    here, off the events: one assistant message_end with reported usage is one call, built into the
    same model_call Event BudgetedModel builds and recorded under `stage`. turn_end carries the same
    message again and is ignored, so no call is counted twice. The runner sits below the agent core,
    so the event is read by its shape, not its class. With a ceiling, record_call charges it; the
    stop itself is the caller's guard, since a raise here would end the run from inside its stream.
    """
    name = model_id or "model"
    provider = name.split("/", 1)[0] if "/" in name else None
    calls = 0

    def price(event: Any) -> None:
        nonlocal calls
        if getattr(event, "type", None) != "message_end":
            return
        message = getattr(event, "message", None)
        usage = getattr(message, "usage", None)
        if getattr(message, "role", None) != "assistant" or not isinstance(usage, Usage):
            return
        if usage.input + usage.output + usage.cache_read + usage.cache_write <= 0:
            return
        with _LEDGER_LOCK:
            calls += 1
            idx = calls - 1
        record = Event(idx=idx, type="model_call",
                       cost=Cost(provider=provider, model=sent_wire_id(model_id, getattr(message, "model", None)) or name,
                                 usage=usage, wall_ms=0.0))
        try:
            record_call(record, stage, workdir, ceiling=ceiling, item=name,
                        request=getattr(event, "request", None))
        except BudgetExceeded:
            return

    return price


def estimate_tokens(*parts: Any) -> int:
    """Prompt size before the call, from the characters the request will carry."""
    chars = 0
    for part in parts:
        if part is None:
            continue
        chars += len(part if isinstance(part, str) else json.dumps(part, default=str, ensure_ascii=False))
    return chars // CHARS_PER_TOKEN


def context_cap_tokens(window: int, fraction: float = CONTEXT_CAP_FRACTION) -> int:
    """The D65 cap in tokens for one window."""
    return int(window * fraction)


def check_context_cap(messages_tokens: int, window: int, fraction: float = CONTEXT_CAP_FRACTION) -> None:
    """D65: a Builder call over the cap is refused, never compacted. Returns None when the call may go."""
    cap = context_cap_tokens(window, fraction)
    if messages_tokens > cap:
        raise ContextCapExceeded(messages_tokens, cap, window)


class Ceiling:
    """The per-build spend ceiling (D86). Charges are recorded first, then the build stops.

    With a workdir the ceiling writes its spend into the same budget.json the calls are
    recorded in, so a resumed build sees what the stopped one had already paid for.
    """

    def __init__(self, usd: float, spent: float = 0.0, workdir: Optional[str | Path] = None):
        self.usd = float(usd)
        self.spent = float(spent)
        self.workdir = Path(workdir) if workdir is not None else None
        self.stage_spend: dict[str, float] = {}
        self.stage_charges: dict[str, int] = {}
        if self.spent >= self.usd:
            # Resuming on a ceiling that is already reached buys one more item before
            # stopping again, which is money spent for nothing. Refuse before that.
            raise BudgetExceeded(
                stage="start", item="(none)", spent=self.spent,
                ceiling_usd=self.usd, estimate_to_finish=0.0,
            )

    @classmethod
    def from_totals(cls, workdir: str | Path, usd: float) -> "Ceiling":
        """Resume a stopped build: what is already paid for is already spent."""
        totals = load_totals(workdir)
        ceiling = cls(usd=usd, spent=float(totals["total"]["usd"]), workdir=workdir)
        for stage, bucket in totals["stages"].items():
            ceiling.stage_spend[stage] = float(bucket["usd"])
            ceiling.stage_charges[stage] = int(bucket["calls"])
        return ceiling

    def require_priced(self, model_id: Optional[str]) -> None:
        """A model this ceiling cannot price is refused before the build calls it."""
        if not is_priced(model_id):
            raise UnpricedModel(model_id)

    @property
    def remaining(self) -> float:
        return max(0.0, self.usd - self.spent)

    def estimate_to_finish(self, stage: str, items_left: int) -> float:
        """From the per-stage numbers already recorded: mean cost per item times the items left."""
        charges = self.stage_charges.get(stage, 0)
        if not items_left or not charges:
            return 0.0
        return self.stage_spend.get(stage, 0.0) / charges * items_left

    def report(self, stage: str, item: str, items_left: int = 0) -> dict:
        """What the report needs when the build stops: where it was, what it spent, what is left."""
        return {
            "stage": stage,
            "item": item,
            "spent": self.spent,
            "ceiling_usd": self.usd,
            "remaining": self.remaining,
            "items_left": items_left,
            "estimate_to_finish": self.estimate_to_finish(stage, items_left),
            "stages": dict(self.stage_spend),
        }

    def add(self, usd: float, stage: str, item: str, items_left: int = 0) -> float:
        """Charge one item, then stop the build if the ceiling is reached. Returns what remains."""
        with _LEDGER_LOCK:
            self._refuse_if_reached(stage, item, items_left)
            self.spent += float(usd)
            self.stage_spend[stage] = self.stage_spend.get(stage, 0.0) + float(usd)
            self.stage_charges[stage] = self.stage_charges.get(stage, 0) + 1
            self._write_spend(stage, float(usd))
            self._refuse_if_reached(stage, item, items_left)
            return self.remaining

    def charge_recorded(self, totals: dict, stage: str, item: str, items_left: int = 0) -> float:
        """Take the spend from the ledger record_call just wrote, rather than counting twice.

        Spent is refreshed from the ledger before the refuse check, not after: a caller that
        charges past an already-reached ceiling still has that charge reflected in spent and in
        report(), instead of both freezing at the value from before this call.
        """
        self.spent = float(totals["total"]["usd"])
        for name, bucket in totals["stages"].items():
            self.stage_spend[name] = float(bucket["usd"])
            self.stage_charges[name] = int(bucket["calls"])
        self._refuse_if_reached(stage, item, items_left)
        return self.remaining

    def _refuse_if_reached(self, stage: str, item: str, items_left: int) -> None:
        """Nothing is charged past a ceiling that is already reached.

        Checked both before a charge, so the next item is refused, and after one, so the charge
        that breached the ceiling is the last the build makes.
        """
        if self.spent >= self.usd:
            raise BudgetExceeded(
                stage=stage, item=item, spent=self.spent, ceiling_usd=self.usd,
                estimate_to_finish=self.estimate_to_finish(stage, items_left),
            )

    def _write_spend(self, stage: str, usd: float) -> None:
        """One ledger: a charge that did not come through record_call still lands in the file."""
        if self.workdir is None:
            return
        totals = load_totals(self.workdir)
        bucket = totals["stages"].setdefault(stage, empty_bucket())
        for target in (bucket, totals["total"]):
            target["calls"] += 1
            target["usd"] += usd
        save_totals(self.workdir, totals)


class BudgetedModel(Model):
    """Every model call the Harness makes, priced, capped and charged, at the one seam it crosses.

    Wrapping the Model is what makes the accounting unmissable: a module that takes a Model
    cannot forget to call record_call, and a module that builds too large a prompt is refused
    here rather than by each module remembering the D65 cap for itself.
    """

    def __init__(
        self,
        inner: Model,
        stage: str,
        workdir: str | Path,
        model_id: Optional[str] = None,
        ceiling: Optional[Ceiling] = None,
        window: Optional[int] = None,
        fraction: float = CONTEXT_CAP_FRACTION,
        cap_context: bool = True,
        prompt_cache_key: Optional[str] = None,
    ):
        self.inner = inner
        self.name = getattr(inner, "name", "model")
        self.stage = stage
        self.workdir = Path(workdir)
        self.model_id = model_id or self.name
        self.ceiling = ceiling
        self.window = window or window_for(self.model_id)
        self.fraction = fraction
        # The Candidate is tested under the production setting and is never capped (D65);
        # only the Builder's own calls are.
        self.cap_context = cap_context
        # OpenAI routes by this key when the caller's own config does not already set one
        # (build.py sets one per build and stage); the Anthropic adapter ignores it.
        self.prompt_cache_key = prompt_cache_key
        # How long this stage's cache points live when the caller's config names no TTL (cache.py).
        self.cache_ttl = cache_module.ttl_for(stage)
        self.calls = 0
        if ceiling is not None:
            ceiling.require_priced(self.model_id)

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        if self.cap_context:
            check_context_cap(estimate_tokens(messages, tools), self.window, self.fraction)
        if self.ceiling is not None:
            # Refuse before the live call once the ceiling is already reached; without this, a
            # ceiling breached by one call keeps letting every call after it run to completion
            # before the after-the-fact check fires.
            self.ceiling._refuse_if_reached(self.stage, self.model_id, 0)
        if self.prompt_cache_key and (config is None or config.prompt_cache_key is None):
            config = (config or ModelConfig()).model_copy(update={"prompt_cache_key": self.prompt_cache_key})
        if self.cache_ttl != cache_module.DEFAULT_CACHE_TTL and (config is None or config.cache_ttl is None):
            config = (config or ModelConfig()).model_copy(update={"cache_ttl": self.cache_ttl})
        request = cache_module.fingerprint(messages, tools)
        started = time.monotonic()
        reply = self.inner.query(messages, tools=tools, config=config)
        wall_ms = (time.monotonic() - started) * 1000.0
        # provider.MemoModel, when this wraps one, marks a served hit here (per thread, D118);
        # anything else (a plain adapter, TestModel, RecordedModel) has no such attribute and
        # counts as a miss.
        memo_hit = bool(getattr(self.inner, "last_hit", False))
        with _LEDGER_LOCK:
            self.calls += 1
            idx = self.calls - 1
        event = Event(
            idx=idx,
            type="model_call",
            cost=Cost(
                provider=self.model_id.split("/", 1)[0] if "/" in self.model_id else None,
                model=sent_wire_id(self.model_id, reply.model),
                usage=reply.usage,
                wall_ms=wall_ms,
            ),
        )
        record_call(event, self.stage, self.workdir, ceiling=self.ceiling, item=self.model_id,
                   memo_hit=memo_hit, request=request)
        return reply
