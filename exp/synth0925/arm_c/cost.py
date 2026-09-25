"""Priced live models for arm C, with a running USD total and a hard ceiling.

Only the two briefed model roles exist here. Prices come from the harness
table so the ceiling sees the same numbers the harness would charge.
"""
from __future__ import annotations

import os
from pathlib import Path

DOTENV = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env")

SYNTH_ID = "opencode-go/muse-spark-1.3-contributor"
JUDGE_ID = "bedrock/global.anthropic.claude-opus-5-5"


def _ensure_env() -> None:
    from kullback.ai import provider as P

    P.load_dotenv(DOTENV)
    P.enable_live_calls_from_env()
    if not os.environ.get("HARNESS_ALLOW_MODEL_REQUESTS"):
        os.environ["HARNESS_ALLOW_MODEL_REQUESTS"] = "1"
        P.enable_live_calls_from_env()


def _price(model_id: str) -> dict:
    from kullback.runner import budget as B

    row = B.price_for(model_id) or {}
    return {"input": float(row.get("input", 0.0)), "output": float(row.get("output", 0.0)),
            "cache_read": float(row.get("cache_read", 0.0)),
            "cache_write": float(row.get("cache_write", 0.0))}


class Priced:
    """One live model with manual USD accounting. No state leaves this object."""

    def __init__(self, model_id: str, ceiling_usd: float) -> None:
        _ensure_env()
        from kullback.ai import provider as P

        self.model_id = model_id
        self.ceiling = float(ceiling_usd)
        self.inner = P.live_model(model_id)
        self.usd = 0.0
        self.calls = 0
        self.price = _price(model_id)

    def _refuse_if_broke(self, need: float = 0.0) -> None:
        if self.usd + need >= self.ceiling:
            raise RuntimeError(f"spend ceiling reached: {self.usd:.4f} USD")

    def cost_of(self, reply: object) -> float:
        use = getattr(reply, "usage", None)
        if use is None:
            return 0.0
        def get(name):
            return float(getattr(use, name, 0) or 0)
        return (get("input") * self.price["input"] + get("output") * self.price["output"]
                + get("cache_read") * self.price["cache_read"]
                + get("cache_write") * self.price["cache_write"]) / 1_000_000.0

    def query(self, messages: list, tools: list | None = None, share: "Share | None" = None) -> object:
        self._refuse_if_broke()
        reply = self.inner.query(messages, tools=tools)
        cost = self.cost_of(reply)
        self.usd += cost
        self.calls += 1
        if share is not None:
            share.add(self.model_id, cost)
        return reply


class Share:
    """Split ledger: model spend vs search spend, kept in one place."""

    def __init__(self, ceiling_usd: float) -> None:
        self.ceiling = float(ceiling_usd)
        self.model_usd = 0.0
        self.search_usd = 0.0
        self.model_calls = 0
        self.searches = 0
        self.fetches = 0

    def add(self, model_id: str, cost: float) -> None:
        self.model_usd += cost
        self.model_calls += 1

    def total(self) -> float:
        return self.model_usd + self.search_usd

    def refuse_if_broke(self) -> None:
        if self.total() >= self.ceiling:
            raise RuntimeError(f"spend ceiling reached: {self.total():.4f} USD")
