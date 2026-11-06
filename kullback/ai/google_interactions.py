"""Google's Interactions API, streamed.

As in anthropic.py the body comes from the synchronous handle in provider.py
(GoogleInteractionsModel), so the steps, the tools and the generation config are written once; this
module reads the answer off the wire.

The stream (recorded live 2026-10-28) is interaction.created, a status update, then per step a
step.start carrying the step's type (and a function call's id and name), step.delta events filling
it (text, arguments_delta, thought_summary, thought_signature), and a step.stop; then
interaction.completed with the status and the usage, and a closing [DONE]. A thought step is
rebuilt whole, signature and summary, because the next request of a tool loop has to send it back.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Sequence

import httpx

from kullback.ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEnd,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderToolCall,
)
from kullback.ai._sse import StreamFacts, async_client, loads_object, stream_sse_events
from kullback.ai.events import StreamEvent
from kullback.ai.messages import AssistantMessage, Message, ToolCall, to_wire
from kullback.ai.provider import (
    INTERACTION_FAILED_STATUSES,
    INTERACTION_STOP_REASONS,
    CancellationToken,
    ModelConfig,
    require_live_calls_enabled,
)
from kullback.ai.stream import canonicalize_provider_stream
from kullback.ai.streamlog import logged_stream
from kullback.ai.usage import Usage, usage_from_interactions

# The statuses an interaction ends in without an answer to read.


class GoogleInteractionsProvider:
    """The Interactions API behind the provider contract, reading SSE as it arrives."""

    def __init__(self, handle: Any, *, client: Optional[httpx.AsyncClient] = None,
                 max_retries: Optional[int] = None):
        self.handle = handle
        self._client = client
        self._owns_client = client is None
        self.max_retries = handle.retry.attempts - 1 if max_retries is None else max_retries

    @property
    def name(self) -> str:
        return self.handle.name

    async def aclose(self) -> None:
        """Close the client, when this provider is the one that made it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def stream_response(
        self,
        *,
        model: str = "",
        system: str = "",
        messages: Sequence[Message] = (),
        tools: Sequence[dict] = (),
        signal: Optional[CancellationToken] = None,
        session_id: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ) -> AsyncIterator[StreamEvent]:
        """One response as canonical events. Nothing is posted until the first event is asked for."""
        return canonicalize_provider_stream(
            self.stream_provider_events(
                system=system, messages=messages, tools=tools, signal=signal, config=config,
            ),
            model=model or self.handle.name,
            # Steps arrive in order, each started and stopped, so a channel switch closes the one it left.
            independent_channels=False,
        )

    def stream_provider_events(
        self,
        *,
        system: str = "",
        messages: Sequence[Message] = (),
        tools: Sequence[dict] = (),
        signal: Optional[CancellationToken] = None,
        config: Optional[ModelConfig] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """The endpoint's own report, before anything is canonicalised."""
        require_live_calls_enabled()
        settings, wire, tool_list = config or ModelConfig(), to_wire(messages, system or None), list(tools) or None
        payload = dict(self.handle.build_body(wire, tool_list, settings))
        payload["stream"] = True
        content = self.handle.encode_body(payload)
        facts = StreamFacts()
        events = stream_sse_events(
            client=self._get_client,
            url=self.handle.base_url + self.handle.path,
            headers=self.handle.headers(content),
            payload=payload,
            content=content,
            parser_factory=InteractionsStreamParser,
            parser_name=self.handle.name,
            model=self.handle.name,
            max_retries=self.max_retries,
            read_timeout_s=self.handle.timeout,
            signal=signal,
            retry_policy=self.handle.retry,
            rng=self.handle.rng,
            rate_bucket=self.handle.rate_bucket,
            facts=facts,
        )
        return logged_stream(self.handle, events, wire, tool_list, settings, facts)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            require_live_calls_enabled()
            self._client = async_client(self.handle.timeout)
        return self._client


class InteractionsStreamParser:
    """The Interactions API SSE events: one step started, filled and stopped at a time."""

    def __init__(self) -> None:
        self.emitted_content = False
        self.fatal = False
        self._text: list[str] = []
        self._thinking: list[str] = []
        # Thought and function_call steps by index, as they will be sent back or called.
        self._steps: dict[int, dict] = {}
        self._arguments: dict[int, list[str]] = {}
        self._status: Optional[str] = None
        self._usage = Usage()

    def feed(self, chunk: str) -> tuple[list[ProviderEvent], bool]:
        if chunk == "[DONE]":
            return [], True
        data = loads_object(chunk)
        if data is None:
            self.fatal = True
            return [ProviderErrorEvent(message="the endpoint sent a chunk that is not JSON")], True
        kind = data.get("event_type")
        if kind == "step.start":
            self._step_start(data)
        elif kind == "step.delta":
            return self._step_delta(data), False
        elif kind == "interaction.completed":
            return self._completed(data)
        elif kind == "error" or "error" in data:
            return self._stream_error(data, data.get("error"))
        return [], False

    def _step_start(self, data: dict) -> None:
        """A thought step kept to send back, or a function call's id and name."""
        step = data.get("step")
        if isinstance(step, dict) and step.get("type") in ("thought", "function_call"):
            self._steps[_step_index(data)] = dict(step)
            if step.get("type") == "function_call":
                self.emitted_content = True

    def _step_delta(self, data: dict) -> list[ProviderEvent]:
        delta = data.get("delta")
        if not isinstance(delta, dict):
            return []
        index, kind = _step_index(data), delta.get("type")
        if kind == "text":
            return self._text_delta(str(delta.get("text") or ""))
        if kind == "thought_signature":
            step = self._steps.setdefault(index, {"type": "thought"})
            step["signature"] = str(step.get("signature") or "") + str(delta.get("signature") or "")
        elif kind == "thought_summary":
            return self._summary_delta(index, delta.get("content"))
        elif kind == "arguments_delta":
            self._arguments.setdefault(index, []).append(str(delta.get("arguments") or ""))
            self.emitted_content = True
        return []

    def _text_delta(self, text: str) -> list[ProviderEvent]:
        if not text:
            return []
        self.emitted_content = True
        self._text.append(text)
        return [ProviderTextDelta(delta=text)]

    def _summary_delta(self, index: int, content: Any) -> list[ProviderEvent]:
        """A summary grows its thought step for the next request, and is reported as thinking."""
        text = str(content.get("text") or "") if isinstance(content, dict) else ""
        if not text:
            return []
        summary = self._steps.setdefault(index, {"type": "thought"}).setdefault("summary", [])
        if summary:
            summary[-1]["text"] += text
        else:
            summary.append({"type": "text", "text": text})
        self.emitted_content = True
        self._thinking.append(text)
        return [ProviderThinkingDelta(delta=text)]

    def _completed(self, data: dict) -> tuple[list[ProviderEvent], bool]:
        interaction = data.get("interaction") if isinstance(data.get("interaction"), dict) else {}
        self._status = interaction.get("status")
        self._usage = usage_from_interactions(interaction.get("usage"))
        if self._status in INTERACTION_FAILED_STATUSES:
            return self._stream_error(data, interaction.get("error"))
        return [], True

    def _stream_error(self, data: dict, error: Any) -> tuple[list[ProviderEvent], bool]:
        self.fatal = True
        message = error.get("message") if isinstance(error, dict) else None
        return [ProviderErrorEvent(message=str(message or "the provider ended the stream with an error"),
                                   data={"event": data})], True

    def finalize(self) -> list[ProviderEvent]:
        calls = [self._call(index, step) for index, step in sorted(self._steps.items())
                 if step.get("type") == "function_call"]
        events: list[ProviderEvent] = [ProviderToolCall(tool_call=call) for call in calls]
        events.append(
            ProviderResponseEnd(
                message=AssistantMessage(
                    content="".join(self._text) or None,
                    thinking="".join(self._thinking) or None,
                    tool_calls=calls,
                    usage=self._usage,
                    # An unsigned thought step (a stream cut before its signature) would be refused
                    # if sent back, so only signed ones are kept.
                    thinking_blocks=[step for _, step in sorted(self._steps.items())
                                     if step.get("type") == "thought" and step.get("signature")] or None,
                ),
                finish_reason=INTERACTION_STOP_REASONS.get(self._status or "", self._status),
            )
        )
        return events

    def _call(self, index: int, step: dict) -> ToolCall:
        """A function call from its start and its argument deltas; the start's arguments when none came."""
        text = "".join(self._arguments.get(index, []))
        parsed = loads_object(text) if text else (step.get("arguments") if isinstance(step.get("arguments"), dict) else {})
        return ToolCall(
            # Sent back unchanged as the result's call_id, so the id is not cleaned.
            id=str(step.get("id") or f"call_{index}"),
            name=str(step.get("name") or ""),
            arguments=parsed if parsed is not None else {"_raw": text},
        )


def _step_index(data: dict) -> int:
    """The step an event is about; a missing index is step 0."""
    return int(data.get("index", 0) or 0)
