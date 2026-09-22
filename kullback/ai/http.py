"""The HTTP plumbing every adapter shares: the read budget, the client, and what one post carries.

Mirrors tau_ai/http.py, with the timeout helpers and the request fingerprints that were in
provider.py moved here verbatim. Nothing here knows a provider; provider.py and the streaming
adapters both read it.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, NamedTuple, Optional

import httpx

# The provider's own id for one exchange, under whichever header that provider sends it.
REQUEST_ID_HEADERS = ("x-request-id", "request-id", "cf-ray")
# The read budget for one model call. A code-generating answer at the size a build asks for can
# take minutes, while a dead host should still fail fast, so the default is generous and only
# establishing contact stays short. One variable overrides it per process.
MODEL_TIMEOUT_ENV_VAR = "KULLBACK_MODEL_TIMEOUT_S"
DEFAULT_READ_TIMEOUT_S = 300.0
CONNECT_TIMEOUT_S = 10.0


def model_read_timeout_s(env: dict[str, str], explicit: Optional[float] = None) -> float:
    """The read timeout in seconds: an explicit argument wins, then the environment, then 300 s.

    A set variable that is not a positive finite number is an error naming the variable,
    never a silent default. An empty value counts as set, so it raises too.
    """
    if explicit is not None:
        return float(explicit)
    raw = env.get(MODEL_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_READ_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{MODEL_TIMEOUT_ENV_VAR} must be a number of seconds, got {raw!r}"
        ) from None
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(
            f"{MODEL_TIMEOUT_ENV_VAR} must be a positive number of seconds, got {raw!r}"
        )
    return value


def request_timeout(read_s: float) -> httpx.Timeout:
    """The split budget one request carries: slow answers get the read budget, a dead host does not."""
    return httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=read_s, write=read_s, pool=CONNECT_TIMEOUT_S)


def timeout_note(exc: Exception, connect_s: float, read_s: float) -> str:
    """Name the budget an expired timeout spent: read for a slow answer, connect or pool for a dead host."""
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
        return f" (read timeout {read_s:g}s)"
    if isinstance(exc, httpx.PoolTimeout):
        return f" (pool timeout {connect_s:g}s)"
    if isinstance(exc, httpx.ConnectTimeout):
        return f" (connect timeout {connect_s:g}s)"
    return ""


def create_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """The one place an async client is built, so a streaming provider takes a timeout and no more."""
    return httpx.AsyncClient(**kwargs)


def json_body(request: Any) -> dict:
    """The JSON a request carries; used by the adapters' tests and by error reporting."""
    try:
        return json.loads(request.content or b"{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def body_hash(part: Any) -> str:
    """sha256 of one part of a request body, so a renderer can check it rebuilt the same prompt.

    `records.content_hash` does the same job for the Harness' records, and this is not it: `ai`
    imports nothing of ours (D121), so the hash the adapters take is written here. It is a
    fingerprint of what was sent, compared with itself, never with a record's hash.
    """
    return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def request_id_of(headers: Any) -> Optional[str]:
    """The provider's id for one exchange: the first of the known headers that is there."""
    get = getattr(headers, "get", None)
    if get is None:
        return None
    for key in REQUEST_ID_HEADERS:
        value = get(key)
        if value:
            return str(value)
    return None


class Posted(NamedTuple):
    """One finished post: the JSON it answered with, and how it answered.

    `post` returns the JSON alone, as it always did; `query` takes this so the Exchange it records
    can say how many attempts the retry loop made and which request the provider logged.
    """

    data: dict
    status: Optional[int] = None
    attempts: int = 0
    headers: Any = None
