"""The errors a provider answers with, and the safe detail a message may carry.

Moved here verbatim from provider.py so the adapters, the retry loop and the streaming providers
all name one set of errors (tau_ai/http_errors.py is the module this mirrors). `error_text` was
`_error_text` there; the name is public now because two adapter modules read it.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# The words providers use when the prompt did not fit. Matched on the error body, lowercased.
CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "context_length",
    "prompt is too long",
    "too long for",
    "maximum context",
    "reduce the length",
)

# A provider body can be a whole HTML page; a message carries the front of it and no more.
MAX_ERROR_DETAIL_LENGTH = 1000


class ProviderError(RuntimeError):
    """A provider said no. Carries the status, the body and how many attempts were made.

    `attempts` is 1 for an error raised without a retry loop behind it (a missing key, a body the
    provider refused once); the retry loop sets the real count on the error it raises, which is what
    a Run's error event reports (D159).
    """

    def __init__(self, message: str, status: Optional[int] = None, body: Any = None,
                 attempts: int = 1):
        self.status = status
        self.body = body
        self.attempts = attempts
        super().__init__(message)


class ContextOverflowError(ProviderError):
    """The prompt did not fit. Never retried: the same prompt will not fit next time either."""


class RetryExhausted(ProviderError):
    """Every attempt failed on a retryable error."""


def error_text(body: Any) -> str:
    """The message a provider's error body carries, whatever shape it came in."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        return str(body.get("message") or body)
    return str(body)


def is_context_overflow(body: Any) -> bool:
    """Whether an error body says the prompt did not fit."""
    return any(marker in error_text(body).lower() for marker in CONTEXT_OVERFLOW_MARKERS)


def provider_http_error_message(
    *,
    provider_name: str,
    status_code: int,
    body: str,
    model: Optional[str] = None,
) -> str:
    """An actionable, secret-free message for one failed HTTP exchange.

    The streaming providers have no response object to hand to `error_for`: they read a body off
    the wire and say what it was. A key never reaches this, because only the response body does.
    """
    prefix = f"{provider_name} request failed with status {status_code}"
    if model:
        prefix = f"{prefix} for model {model}"
    detail = error_text(parsed_body(body)).strip()[:MAX_ERROR_DETAIL_LENGTH]
    return f"{prefix}: {detail}" if detail else prefix


def parsed_body(body: str) -> Any:
    """A response body as the object it carries, or as the text it is when it carries none."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body
    return parsed if isinstance(parsed, dict) else body
