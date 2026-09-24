"""The token count one model call returns, the one record the provider layer owns.

It sits in its own module rather than in provider.py because runner/records.py imports it (Cost
carries a Usage) and verdict.py must have no import path to a provider (D76): a module that holds
one pydantic class and no client keeps that true while ai still imports nothing of ours (D121).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class Usage(BaseModel):
    """Tokens on one model call, for budget.py. Counts are never negative: a negative one
    would make call_cost negative and could lower the spend ceiling's total.

    `reasoning` is a part of `output`, not an addition to it: providers that report reasoning
    tokens count them inside the completion total, so call_cost prices output as before and a
    record with the count costs the same as one without it. A count above output is refused,
    not clamped: it means the response was malformed, and quietly rewriting a provider's
    numbers would hide that. (The provider boundary maps an odd reported count to zero before
    a record is ever built; see provider._reasoning_share. Reaching this refusal therefore
    means a stored record is malformed.) Zero means not reported, not none: a provider that reports no
    count leaves zero, and no reader may treat zero as proof no reasoning happened.

    A zero count is omitted from the stored form: model_dump and model_dump_json drop the
    key, so a record that carries no count is byte-identical to one written before the field
    existed. Readers must use attribute access, which always yields zero, never key access on
    a dumped dict. A nonzero count is stored as usual and round-trips.

    A record in the sense of runner/records.py (same config: aliases both ways, unknown keys
    refused); records.py re-exports it and lists it in ALL_RECORDS.
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)
    # The part of cache_write written with the one-hour TTL, which is billed at its own, higher
    # rate (budget.call_cost). A part of cache_write, never an addition to it, and omitted from the
    # stored form at zero like reasoning, so a record written before it existed reads the same.
    cache_write_1h: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _reasoning_within_output(self) -> "Usage":
        if self.reasoning > self.output:
            raise ValueError(
                f"reasoning tokens ({self.reasoning}) are a part of output "
                f"({self.output}), so they cannot exceed it"
            )
        if self.cache_write_1h > self.cache_write:
            raise ValueError(
                f"one-hour cache writes ({self.cache_write_1h}) are a part of cache_write "
                f"({self.cache_write}), so they cannot exceed it"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_unreported_reasoning(self, handler):
        data = handler(self)
        if self.reasoning == 0:
            data.pop("reasoning", None)
        if self.cache_write_1h == 0:
            data.pop("cache_write_1h", None)
        return data


def reasoning_of(usage: Any) -> int:
    """The reasoning count a provider usage payload reports, or zero when it carries none.

    One reader for every shape the adapters parse: a flat reasoning key on a stored reply,
    the chat shape's completion_tokens_details.reasoning_tokens, and the Responses shape's
    output_tokens_details.reasoning_tokens. This reads the reported number only; whether it is
    a share of output is the boundary's call in `reasoning_share` below. Zero here means the
    provider did not report it, not that no reasoning happened.
    """
    if not isinstance(usage, dict):
        return 0
    direct = usage.get("reasoning", 0) or 0
    chat = usage.get("completion_tokens_details") or {}
    answered = usage.get("output_tokens_details") or {}
    if not isinstance(chat, dict):
        chat = {}
    if not isinstance(answered, dict):
        answered = {}
    return int(direct or chat.get("reasoning_tokens", 0) or answered.get("reasoning_tokens", 0) or 0)


def reasoning_share(usage: Any, output: int) -> int:
    """The reported count as a share of output, or zero when it is not one.

    The boundary is tolerant where the record is strict: some routed providers report reasoning
    outside the completion total, and a reply the build already paid for must not die over a
    telemetry field. A count that is negative or above the reported output is recorded as zero,
    which means "not reported as a part of output", and is never rewritten into a different
    nonzero number. The oddity stays visible on the reply's `raw` payload beside it, which is
    the one way this module surfaces something non-fatal (there is no log line and no warning
    here). `Usage` itself still refuses such a record, so a stored file carrying one fails at
    load instead of loading wrong.
    """
    reported = reasoning_of(usage)
    if reported < 0 or reported > output:
        return 0
    return reported


def usage_from_openai_chat(usage: Any) -> Usage:
    """One chat completions usage payload as ours.

    `Usage.input` means the input neither read from nor written to the cache everywhere in the
    Harness, which is what Anthropic already reports (its three counts add up to the prompt).
    OpenAI's prompt_tokens includes the cached ones and, on endpoints that charge to write the cache,
    the written ones too, so both come off here and budget.py bills each count at its own rate with
    no arithmetic. Leaving the written tokens inside input billed them twice, once at the input rate
    and once at the write rate (smoke 8: every one of 1726 writes was at most its call's input).
    """
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    cached = int(details.get("cached_tokens", 0) or 0)
    written = int(details.get("cache_write_tokens", 0) or 0)
    output = int(usage.get("completion_tokens", 0) or 0)
    return Usage(
        input=max(0, int(usage.get("prompt_tokens", 0) or 0) - cached - written),
        output=output,
        cache_read=cached,
        cache_write=written,
        reasoning=reasoning_share(usage, output),
    )


def usage_from_openai_responses(usage: Any) -> Usage:
    """One Responses API usage payload as ours, on the same convention: cached and written come off input."""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    cached = int(details.get("cached_tokens", 0) or 0)
    written = int(details.get("cache_write_tokens", 0) or 0)
    output = int(usage.get("output_tokens", 0) or 0)
    return Usage(
        input=max(0, int(usage.get("input_tokens", 0) or 0) - cached - written),
        output=output,
        cache_read=cached,
        cache_write=written,
        reasoning=reasoning_share(usage, output),
    )


def usage_from_anthropic(usage: Any) -> Usage:
    """One Messages API usage payload as ours. Its input count is already the uncached one.

    The Messages API reports no separate reasoning count today (thinking tokens sit inside
    output_tokens), so the reasoning field reads zero until the payload carries one.
    """
    usage = usage if isinstance(usage, dict) else {}
    output = int(usage.get("output_tokens", 0) or 0)
    written = int(usage.get("cache_creation_input_tokens", 0) or 0)
    # The write split by TTL, when the payload carries it; the one-hour part is billed higher.
    split = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    return Usage(
        input=int(usage.get("input_tokens", 0) or 0),
        output=output,
        cache_read=int(usage.get("cache_read_input_tokens", 0) or 0),
        cache_write=written,
        reasoning=reasoning_share(usage, output),
        cache_write_1h=min(written, int(split.get("ephemeral_1h_input_tokens", 0) or 0)),
    )
