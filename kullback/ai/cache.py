"""Prompt caching, written once for every adapter: the request fingerprint, the TTL, the break points.

Every provider the Harness talks to caches the same thing, a prefix that is byte identical from one
call to the next in the order tools, system, messages. OpenAI-shaped endpoints cache it on their
own; Anthropic, on its own API and on Bedrock, caches only up to a block marked with
`cache_control`, at most four marks per request, and the legacy Bedrock stack refuses a top-level
`cache_control` field with a 400. So the marks go on blocks only, and never more than four.

The fingerprint is what makes a miss visible without guessing. Three short hashes per call:

- `prefix_hash`: the tools and the leading system messages, the part every call of a role shares;
- `history_hash`: the messages after them, up to the last break point, which is the last message;
- `parent_hash`: the same messages up to the last assistant turn, which is what the request that
  produced that turn sent. When a call's `parent_hash` is an earlier call's `history_hash` under the
  same `prefix_hash`, the conversation grew by appending and the provider had that prefix to read.

The hash keeps key order as the dict holds it, the way the adapters put it on the wire: a key that
moves is a changed prefix to the provider, so it has to be a changed hash here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional, Sequence

# How long a written prefix lives. "5m" is the provider default and is sent as a bare ephemeral
# mark, byte identical to what the adapters sent before the setting existed; "1h" writes at twice
# the input price instead of one and a quarter times, so it pays only where the same prefix is
# asked again after more than five minutes (prompt-caching.md, Choosing the TTL).
CACHE_TTLS = ("5m", "1h")
DEFAULT_CACHE_TTL = "5m"
# The TTL per stage when the caller names none. Measured on smoke 8 (the report of ov-cache): the
# runner stage never went five minutes without a call, while the Builder session waited on its own
# long tools (examine, run) for 6 to 47 minutes between turns, six times in one build, and each
# wait re-sent a prompt of 75,000 to 130,000 tokens. The Examiner waited past five minutes once or
# three times on prompts a quarter of the size, which does not pay back the doubled write.
STAGE_CACHE_TTL: dict[str, str] = {"builder_session": "1h"}
# Anthropic's limit on cache_control marks in one request, counted over tools, system and messages.
MAX_CACHE_POINTS = 4
# Sixteen hex digits: enough to tell two prefixes of one build apart, short enough for every feed line.
HASH_CHARS = 16


def ttl_for(stage: Optional[str]) -> str:
    """The TTL a stage's calls write with when the caller named none."""
    return STAGE_CACHE_TTL.get(stage or "", DEFAULT_CACHE_TTL)


def cache_control(ttl: Optional[str] = None) -> dict:
    """The mark one block carries. The default TTL is the bare mark, so a 5m request is unchanged."""
    if ttl is None or ttl == DEFAULT_CACHE_TTL:
        return {"type": "ephemeral"}
    if ttl not in CACHE_TTLS:
        raise ValueError(f"cache TTL is one of {', '.join(CACHE_TTLS)}, not {ttl!r}")
    return {"type": "ephemeral", "ttl": ttl}


def _digest(part: Any) -> str:
    blob = json.dumps(part, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:HASH_CHARS]


def _plain(message: Any) -> Any:
    """A message as data: a pydantic message is dumped, a dict is taken as it is."""
    dump = getattr(message, "model_dump", None)
    return dump(mode="json") if callable(dump) else message


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def fingerprint(messages: Sequence[Any], tools: Optional[Iterable[Any]] = None) -> dict:
    """The three hashes of one request, and how many messages it carried (see the module docstring).

    `messages` is the canonical chat list a Model is asked with, system messages included; `tools`
    is the tool list as the caller passes it. A request with no assistant turn has no parent.
    """
    plain = [_plain(message) for message in messages or ()]
    lead = 0
    while lead < len(plain) and _role(plain[lead]) == "system":
        lead += 1
    system, history = plain[:lead], plain[lead:]
    last_assistant = max((i for i, m in enumerate(history) if _role(m) == "assistant"), default=None)
    return {
        "prefix_hash": _digest({"tools": list(tools or ()), "system": system}),
        "history_hash": _digest(history),
        "parent_hash": _digest(history[:last_assistant]) if last_assistant is not None else None,
        "n_messages": len(plain),
    }


def count_cache_points(body: Any) -> int:
    """How many cache_control marks a request body carries, wherever they sit."""
    if isinstance(body, dict):
        return int("cache_control" in body) + sum(count_cache_points(value) for value in body.values())
    if isinstance(body, list):
        return sum(count_cache_points(value) for value in body)
    return 0


__all__ = ["CACHE_TTLS", "DEFAULT_CACHE_TTL", "HASH_CHARS", "MAX_CACHE_POINTS", "STAGE_CACHE_TTL",
           "cache_control", "count_cache_points", "fingerprint", "ttl_for"]
