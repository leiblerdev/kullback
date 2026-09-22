"""How much context one model takes, which is what decides when the core compacts.

Mirrors tau_ai/model_limits.py. The numbers come from the models.dev snapshot `pricing.py` already
keeps, so a window and a price are read from one file and no stage learns a second place to look.
Compaction at 40 percent of the window (the overhaul's rule, ADR-0011) reads
`effective_context_window` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

# What a model is assumed to take when the snapshot lists no window for it. Deliberately small:
# compacting earlier than needed costs a summary, compacting later than the model allows costs the
# whole turn.
DEFAULT_CONTEXT_WINDOW = 128_000
# The share of the window the core fills before it compacts.
COMPACTION_SHARE_PERCENT = 40


@dataclass(frozen=True)
class ModelLimits:
    """One model's limits on the surface serving it."""

    context_window: int
    max_output_tokens: Optional[int] = None
    effective_context_window_percent: int = 100

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window is a positive number of tokens")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens is a positive number of tokens")
        if not 1 <= self.effective_context_window_percent <= 100:
            raise ValueError("effective_context_window_percent is between 1 and 100")

    @property
    def effective_context_window(self) -> int:
        """The usable window after the surface's own headroom."""
        return max(1, self.context_window * self.effective_context_window_percent // 100)

    @property
    def compaction_token_limit(self) -> int:
        """The filled context at which the core compacts: 40 percent of the usable window."""
        return max(1, self.effective_context_window * COMPACTION_SHARE_PERCENT // 100)


@runtime_checkable
class ModelLimitsProvider(Protocol):
    """A provider that can be asked what one model's limits are on its surface."""

    def discover_model_limits(self, model: str) -> Optional[ModelLimits]:
        ...


def limits_for(model_id: Optional[str], *, path: Optional[str] = None,
               env: Optional[dict[str, str]] = None) -> ModelLimits:
    """The limits the snapshot lists for a model id, falling back to the generic window.

    `pricing` is imported here rather than at the top because it reads the live-call switch off
    `provider`, the same deferred import budget.py makes, for the same reason.
    """
    from kullback.ai import pricing

    catalog = pricing.refresh(path=path, env=env)
    window = pricing.window_from_catalog(catalog, model_id)
    return ModelLimits(context_window=window or DEFAULT_CONTEXT_WINDOW)
