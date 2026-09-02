"""The token count one model call returns, the one record the provider layer owns.

It sits in its own module rather than in provider.py because runner/records.py imports it (Cost
carries a Usage) and verdict.py must have no import path to a provider (D76): a module that holds
one pydantic class and no client keeps that true while ai still imports nothing of ours (D121).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Tokens on one model call, for budget.py. Counts are never negative: a negative one
    would make call_cost negative and could lower the spend ceiling's total.

    A record in the sense of runner/records.py (same config: aliases both ways, unknown keys
    refused); records.py re-exports it and lists it in ALL_RECORDS.
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
