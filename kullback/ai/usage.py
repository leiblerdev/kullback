"""The token count one model call returns, the one record the provider layer owns.

It sits in its own module rather than in provider.py because runner/records.py imports it (Cost
carries a Usage) and verdict.py must have no import path to a provider (D76): a module that holds
one pydantic class and no client keeps that true while ai still imports nothing of ours (D121).
"""

from __future__ import annotations

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

    @model_validator(mode="after")
    def _reasoning_within_output(self) -> "Usage":
        if self.reasoning > self.output:
            raise ValueError(
                f"reasoning tokens ({self.reasoning}) are a part of output "
                f"({self.output}), so they cannot exceed it"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_unreported_reasoning(self, handler):
        data = handler(self)
        if self.reasoning == 0:
            data.pop("reasoning", None)
        return data
