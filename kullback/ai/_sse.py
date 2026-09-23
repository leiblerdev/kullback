"""The streaming POST every adapter shares: retry, cancel, and one SSE line at a time.

tau keeps this envelope inside each adapter; Kullback has one copy, because the only difference
between its endpoints is how a chunk is read, and that is the `StreamParser` an adapter supplies.
The loop yields what the parser made of a chunk the moment the chunk arrives, so a consumer sees a
delta while the rest of the answer is still on the wire.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Optional, Protocol

import httpx

from kullback.ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseStart,
    ProviderRetry,
)
from kullback.ai.http import CONNECT_TIMEOUT_S, create_async_client, request_id_of, timeout_note
from kullback.ai.http_errors import provider_http_error_message
from kullback.ai.retry import (
    CancelSignal,
    retry_delay_seconds,
    retryable_status,
    wait_for_retry,
)


class StreamParser(Protocol):
    """One endpoint's reading of the SSE chunks, driven by the envelope below."""

    # True once any model output has been reported: a stream that dropped after saying something
    # is not tried again, because the answer would be produced twice.
    emitted_content: bool
    # True when the parser already reported a terminal error and finalize must not be called.
    fatal: bool

    def feed(self, chunk: str) -> tuple[list[ProviderEvent], bool]:
        """One SSE data payload, as events and whether the stream is over."""
        ...

    def finalize(self) -> list[ProviderEvent]:
        """The trailing tool calls and the response end, once the lines have run out."""
        ...


def parse_sse_line(line: str) -> Optional[str]:
    """The payload of one `data:` line, or None for the framing around it."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def loads_object(value: str) -> Optional[dict]:
    """One SSE payload as the object it carries, or None when it is not one."""
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def stream_sse_events(
    *,
    client: Callable[[], httpx.AsyncClient],
    url: str,
    headers: dict,
    payload: dict,
    parser_factory: Callable[[], StreamParser],
    parser_name: str,
    model: Optional[str] = None,
    max_retries: int = 2,
    max_retry_delay_seconds: float = 1.0,
    read_timeout_s: float = 300.0,
    signal: Optional[CancelSignal] = None,
) -> AsyncIterator[ProviderEvent]:
    """Post once and report what comes back, trying a transient failure again from the start.

    A retry is only offered while the parser has said nothing: once a delta has been yielded the
    consumer has seen part of an answer, and a second attempt would give it a second one.
    """

    async def iterator() -> AsyncIterator[ProviderEvent]:
        attempt = 0
        while True:
            parser = parser_factory()
            try:
                async with client().stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        if attempt < max_retries and retryable_status(response.status_code):
                            delay = retry_delay_seconds(attempt, max_delay_seconds=max_retry_delay_seconds)
                            yield _retry_event(
                                attempt=attempt,
                                max_retries=max_retries,
                                delay_seconds=delay,
                                reason=f"HTTP {response.status_code}",
                                data={"status": response.status_code, "body": body},
                            )
                            attempt += 1
                            if not await wait_for_retry(delay, signal=signal):
                                return
                            continue
                        yield ProviderErrorEvent(
                            message=provider_http_error_message(
                                provider_name=parser_name,
                                status_code=response.status_code,
                                body=body,
                                model=model,
                            ),
                            data={"status": response.status_code, "body": body, "attempts": attempt + 1},
                        )
                        return

                    yield ProviderResponseStart(model=model, request_id=request_id_of(response.headers))
                    async for line in response.aiter_lines():
                        if signal is not None and signal.is_cancelled():
                            return
                        chunk = parse_sse_line(line)
                        if chunk is None:
                            continue
                        events, done = parser.feed(chunk)
                        for event in events:
                            yield event
                        if done:
                            break
                    if parser.fatal:
                        return
                    for event in parser.finalize():
                        yield event
                    return
            except httpx.HTTPError as exc:
                if not parser.emitted_content and attempt < max_retries:
                    delay = retry_delay_seconds(attempt, max_delay_seconds=max_retry_delay_seconds)
                    yield _retry_event(
                        attempt=attempt,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        reason="network error",
                        data={"error": str(exc), "error_type": type(exc).__name__},
                    )
                    attempt += 1
                    if not await wait_for_retry(delay, signal=signal):
                        return
                    continue
                note = timeout_note(exc, CONNECT_TIMEOUT_S, read_timeout_s)
                yield ProviderErrorEvent(
                    message=f"{parser_name}: {exc}{note}",
                    data={"attempts": attempt + 1},
                )
                return

    return iterator()


def _retry_event(
    *,
    attempt: int,
    max_retries: int,
    delay_seconds: float,
    reason: str,
    data: Optional[dict[str, Any]] = None,
) -> ProviderRetry:
    next_attempt = attempt + 2
    max_attempts = max_retries + 1
    suffix = f" in {delay_seconds:g}s" if delay_seconds else ""
    return ProviderRetry(
        attempt=next_attempt,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        message=f"trying the provider again, {next_attempt} of {max_attempts}, after {reason}{suffix}",
        data=data,
    )


def async_client(timeout_s: float) -> httpx.AsyncClient:
    """The client a streaming provider makes when the caller passed none."""
    return create_async_client(timeout=httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=timeout_s,
                                                    write=timeout_s, pool=CONNECT_TIMEOUT_S))
