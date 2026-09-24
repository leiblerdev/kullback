"""How much context one model takes, which is what decides when the core compacts.

Mirrors tau_ai/model_limits.py. The numbers come from the models.dev snapshot `pricing.py` already
keeps, so a window and a price are read from one file and no stage learns a second place to look.
Compaction at 40 percent of the window (the overhaul's rule, ADR-0011) reads
`effective_context_window` and nothing else.
"""

from __future__ import annotations

import re
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


@dataclass(frozen=True)
class RequestRules:
    """What one Messages API model refuses in a request body, so the handle never sends it.

    Keyed by capability, not by provider: the same model answers the same way on the first-party
    API and on a cloud that serves the Messages API shape (Bedrock's included), so a rule is a
    property of the model and one handle applies it wherever the model lives.
    """

    # `thinking: {type: "disabled"}` and `{type: "enabled", budget_tokens: N}` are a 400; only
    # adaptive thinking (or no thinking field) is accepted, and effort is the control.
    thinking_always_on: bool = False
    # `tool_choice` `any` or a named tool is a 400; `auto` and `none` still work.
    forced_tool_choice: bool = True
    # `temperature`, `top_p` and `top_k` are a 400.
    sampling: bool = True


# The model-migration guide in the claude-api reference (.claude/herdr/overhaul-0922/reference):
# sampling parameters stop being accepted at Opus 4.7 ("Sampling parameters", line 613) and the
# later sections carry that forward; thinking is always on for Fable 5 and later (line 1322)
# and for Opus 5.5 (line 1870); forced tool choice is refused on Fable 5.1, Mythos 5.1 (line 1557)
# and Opus 5.5 (line 1905). The next model is a row here, not a branch in an adapter.
_ALWAYS_ON = RequestRules(thinking_always_on=True, forced_tool_choice=False, sampling=False)
REQUEST_RULES: dict[str, RequestRules] = {
    "claude-opus-5-5": _ALWAYS_ON,
    "claude-fable-5-1": _ALWAYS_ON,
    "claude-mythos-5-1": _ALWAYS_ON,
    "claude-fable-5": RequestRules(thinking_always_on=True, sampling=False),
    "claude-opus-5": RequestRules(sampling=False),
    "claude-opus-4-8": RequestRules(sampling=False),
    "claude-opus-4-7": RequestRules(sampling=False),
    "claude-sonnet-5": RequestRules(sampling=False),
}
DEFAULT_REQUEST_RULES = RequestRules()


def request_rules_for(wire_id: Optional[str]) -> RequestRules:
    """The rules for a wire id, with any gateway path or cloud prefix taken off first.

    `anthropic.claude-opus-5-5` (Bedrock) and `us.anthropic.claude-opus-5-5` (a cross-region
    profile) name the same model as `claude-opus-5-5`: a Claude id carries no dot, so the part
    after the last slash and the last dot is the model.
    """
    name = str(wire_id or "").rpartition("/")[2].rpartition(".")[2]
    return REQUEST_RULES.get(name, DEFAULT_REQUEST_RULES)


# An AWS inference profile names a model with a geography in front of Bedrock's own id:
# `us.anthropic.claude-opus-5-5` and `global.anthropic.claude-opus-5-5` both answer as
# `claude-opus-5-5` (live probe, us-east-2, 2026-09-24). The geography is a rule on the prefix,
# not a list of models, so one price row and one rules row serve every profile of a model.
_PROFILE_PREFIX = re.compile(r"^(?:us|eu|apac|ap|jp|au|us-gov|global)\.(?=[^.]+\.)")


def without_profile(model_id: Optional[str]) -> Optional[str]:
    """The id with any leading inference profile geography taken off its wire part.

    `bedrock/us.anthropic.claude-opus-5-5` becomes `bedrock/anthropic.claude-opus-5-5`; an id with
    no profile, or no Bedrock-shaped `vendor.model` after the geography, comes back unchanged.
    """
    if not model_id:
        return model_id
    head, slash, wire = model_id.rpartition("/")
    return head + slash + _PROFILE_PREFIX.sub("", wire)



# Our provider name where models.dev spells it another way. The catalog keys Bedrock's rows by the
# full wire id, profile included (`global.anthropic.claude-opus-5-5` and `us.anthropic.claude-opus-5-5`
# are two rows at two prices), under `amazon-bedrock`.
PROVIDER_CATALOG_NAMES: dict[str, str] = {"bedrock": "amazon-bedrock"}


def split_vendor(wire_id: Optional[str]) -> Optional[tuple[str, str]]:
    """A cloud id's vendor and model, `global.a-vendor.model-1` to (`a-vendor`, `model-1`).

    The profile geography comes off first; what is left is `vendor.model`, split at the first dot,
    since a model name may carry dots of its own. None when the id has no vendor segment.
    """
    bare = without_profile(str(wire_id or "").rpartition("/")[2]) or ""
    vendor, dot, model = bare.partition(".")
    return (vendor, model) if dot and vendor and model else None


def catalog_candidates(model_id: Optional[str]) -> list[tuple[str, str]]:
    """The catalog ids a model is looked up under, in order, each with the name of its step.

    `exact`: the wire id as sent, under the provider's models.dev name. `without-profile`: the same
    with the inference profile geography taken off. `vendor`: for a provider that renames itself in
    the catalog (a cloud serving other vendors' models), the vendor's own row for the model alone.
    Whoever asks (a price, a window) takes the first candidate that answers.
    """
    if not model_id or "/" not in model_id:
        return [("exact", model_id)] if model_id else []
    provider, _, wire = model_id.partition("/")
    listed = PROVIDER_CATALOG_NAMES.get(provider, provider)
    found: list[tuple[str, str]] = [("exact", f"{listed}/{wire}")]
    stripped = without_profile(wire) or wire
    if stripped != wire:
        found.append(("without-profile", f"{listed}/{stripped}"))
    vendor = split_vendor(wire) if provider in PROVIDER_CATALOG_NAMES else None
    if vendor is not None:
        found.append(("vendor", f"{vendor[0]}/{vendor[1]}"))
    return found
