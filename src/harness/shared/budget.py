"""Cost and token accounting: per call, per stage, per build, plus the D65 context cap and the D86 spend ceiling."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from harness.shared.provider import Model, ModelConfig, ModelReply
from harness.shared.records import Cost, Event, Usage

# --- price table -------------------------------------------------------------
# Data, not code. US dollars per 1M tokens. cache_read is what a cache hit costs,
# cache_write what writing the cache costs.
#
# UPDATE ME: these are list prices checked by hand on the date below. Vendors change
# them without warning, so re-check before trusting a build's cost, and add an entry
# for every model a build actually calls. A model with no entry is not priced at zero
# quietly: its calls are counted under unpriced_calls in the totals file and the
# report shows that count.
PRICES_CHECKED = "2026-08-28"
PRICES_NOTE = (
    "list prices per 1M tokens, checked by hand on "
    + PRICES_CHECKED
    + "; update me before trusting a build's cost, and add the models you call"
)
PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25},
    "anthropic/claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    "anthropic/claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
    # The first slice's Candidates (design section 11 step 6). OpenAI has no cache-write
    # charge, so cache_write is 0 and cached input is billed at cache_read.
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_write": 0.0},
    "openai/o4-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.275, "cache_write": 0.0},
}

# What fits in one call, per model, for the D65 cap. A model with no row uses the default.
DEFAULT_CONTEXT_WINDOW = 200_000
CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-5": 1_000_000,
    "anthropic/claude-sonnet-5": 1_000_000,
    "anthropic/claude-haiku-4-5": 200_000,
    "openai/gpt-4.1-mini": 1_000_000,
    "openai/o4-mini": 200_000,
}

TOTALS_NAME = "budget.json"
BUCKET_FIELDS = ("calls", "input", "output", "cache_read", "cache_write", "usd", "wall_ms", "unpriced_calls")
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


def price_for(model_id: Optional[str]) -> Optional[dict[str, float]]:
    """Prices for a full 'provider/model' id, or for the wire id alone."""
    if not model_id:
        return None
    if model_id in PRICES:
        return PRICES[model_id]
    for key, price in PRICES.items():
        if key.split("/", 1)[-1] == model_id:
            return price
    return None


def is_priced(model_id: Optional[str]) -> bool:
    return price_for(model_id) is not None


def window_for(model_id: Optional[str]) -> int:
    """The context window of a model, for the D65 cap."""
    if model_id:
        if model_id in CONTEXT_WINDOWS:
            return CONTEXT_WINDOWS[model_id]
        for key, window in CONTEXT_WINDOWS.items():
            if key.split("/", 1)[-1] == model_id:
                return window
    return DEFAULT_CONTEXT_WINDOW


def call_cost(usage: Usage, model_id: Optional[str]) -> float:
    """What one call cost.

    Each of the four counts is billed at its own rate, with no arithmetic between them:
    Usage.input is the uncached input everywhere in the Harness, which is what Anthropic
    reports directly and what the OpenAI adapter subtracts down to. Subtracting the cached
    counts from input again under-billed every cached call.
    """
    price = price_for(model_id)
    if price is None:
        return 0.0
    return (
        usage.input * price["input"]
        + usage.output * price["output"]
        + usage.cache_read * price["cache_read"]
        + usage.cache_write * price["cache_write"]
    ) / 1_000_000


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
) -> Event:
    """Price one model-call Event, write the cost back onto it, and add it to the stage and build totals.

    With a ceiling, the same write charges it: the file is the one ledger, so the ceiling's
    spend and the report's cost so far cannot drift apart.
    """
    if event.cost is None:
        return event
    usage = event.cost.usage
    event.cost.usd = call_cost(usage, event.cost.model)
    totals = load_totals(workdir)
    bucket = totals["stages"].setdefault(stage, empty_bucket())
    for target in (bucket, totals["total"]):
        target["calls"] += 1
        target["input"] += usage.input
        target["output"] += usage.output
        target["cache_read"] += usage.cache_read
        target["cache_write"] += usage.cache_write
        target["usd"] += event.cost.usd
        target["wall_ms"] += event.cost.wall_ms
        if not is_priced(event.cost.model):
            target["unpriced_calls"] += 1
    save_totals(workdir, totals)
    if ceiling is not None:
        ceiling.charge_recorded(totals, stage, item or str(event.idx), items_left)
    return event


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
        self._refuse_if_reached(stage, item, items_left)
        self.spent += float(usd)
        self.stage_spend[stage] = self.stage_spend.get(stage, 0.0) + float(usd)
        self.stage_charges[stage] = self.stage_charges.get(stage, 0) + 1
        self._write_spend(stage, float(usd))
        self._stop_if_reached(stage, item, items_left)
        return self.remaining

    def charge_recorded(self, totals: dict, stage: str, item: str, items_left: int = 0) -> float:
        """Take the spend from the ledger record_call just wrote, rather than counting twice."""
        self._refuse_if_reached(stage, item, items_left)
        self.spent = float(totals["total"]["usd"])
        for name, bucket in totals["stages"].items():
            self.stage_spend[name] = float(bucket["usd"])
            self.stage_charges[name] = int(bucket["calls"])
        self._stop_if_reached(stage, item, items_left)
        return self.remaining

    def _refuse_if_reached(self, stage: str, item: str, items_left: int) -> None:
        """Nothing is charged past a ceiling that is already reached."""
        if self.spent >= self.usd:
            raise BudgetExceeded(
                stage=stage, item=item, spent=self.spent, ceiling_usd=self.usd,
                estimate_to_finish=self.estimate_to_finish(stage, items_left),
            )

    def _stop_if_reached(self, stage: str, item: str, items_left: int) -> None:
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
        started = time.monotonic()
        reply = self.inner.query(messages, tools=tools, config=config)
        wall_ms = (time.monotonic() - started) * 1000.0
        self.calls += 1
        event = Event(
            idx=self.calls - 1,
            type="model_call",
            cost=Cost(
                provider=self.model_id.split("/", 1)[0] if "/" in self.model_id else None,
                model=reply.model or self.model_id,
                usage=reply.usage,
                wall_ms=wall_ms,
            ),
        )
        record_call(event, self.stage, self.workdir, ceiling=self.ceiling, item=self.model_id)
        return reply
