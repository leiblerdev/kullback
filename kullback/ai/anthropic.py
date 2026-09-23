"""Anthropic's Messages API, streamed.

Mirrors tau_ai/anthropic.py. As in openai_compatible.py the body comes from the synchronous handle
in provider.py, so the cache points, the system split and the tool shaping are written once; this
module reads the answer off the wire.
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
from kullback.ai._sse import async_client, loads_object, stream_sse_events
from kullback.ai.events import StreamEvent
from kullback.ai.messages import AssistantMessage, Message, ToolCall, to_wire
from kullback.ai.provider import CancellationToken, ModelConfig, require_live_calls_enabled
from kullback.ai.stream import canonicalize_provider_stream
from kullback.ai.tool_call_ids import clean_tool_call_id
from kullback.ai.usage import Usage, usage_from_anthropic


class AnthropicProvider:
    """The Messages API behind the provider contract, reading SSE as it arrives."""

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
            # The Messages API sends its content blocks in order and closes each one, so a channel
            # switch closes the block it left.
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
        payload = dict(
            self.handle.build_body(to_wire(messages, system or None), list(tools) or None,
                                   config or ModelConfig())
        )
        payload["stream"] = True
        return stream_sse_events(
            client=self._get_client,
            url=self.handle.base_url + self.handle.path,
            headers=self.handle.headers(),
            payload=payload,
            parser_factory=MessagesStreamParser,
            parser_name=self.handle.name,
            model=self.handle.name,
            max_retries=self.max_retries,
            read_timeout_s=self.handle.timeout,
            signal=signal,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            require_live_calls_enabled()
            self._client = async_client(self.handle.timeout)
        return self._client


class MessagesStreamParser:
    """The Messages API SSE events: one content block opened, filled and stopped at a time."""

    def __init__(self) -> None:
        self.emitted_content = False
        self.fatal = False
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._builders: dict[int, _ToolUseBuilder] = {}
        self._finish_reason: Optional[str] = None
        self._usage = Usage()

    def feed(self, chunk: str) -> tuple[list[ProviderEvent], bool]:
        data = loads_object(chunk)
        if data is None:
            self.fatal = True
            return [ProviderErrorEvent(message="the endpoint sent a chunk that is not JSON")], True
        kind = data.get("type")

        if kind == "message_start":
            message = data.get("message")
            if isinstance(message, dict):
                self._usage = usage_from_anthropic(message.get("usage"))
        elif kind == "content_block_start":
            block = data.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                builder = self._builders.setdefault(int(data.get("index", 0) or 0), _ToolUseBuilder())
                builder.id = str(block.get("id") or "")
                builder.name = str(block.get("name") or "")
                self.emitted_content = True
        elif kind == "content_block_delta":
            return self._block_delta(data)
        elif kind == "message_delta":
            delta = data.get("delta")
            if isinstance(delta, dict):
                self._finish_reason = delta.get("stop_reason") or self._finish_reason
            # message_delta carries the output count, which message_start could not know yet.
            self._usage = _with_output(self._usage, data.get("usage"))
        elif kind == "message_stop":
            return [], True
        elif kind == "error":
            self.fatal = True
            error = data.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            return [ProviderErrorEvent(message=str(message or "the provider ended the stream with an error"),
                                       data={"event": data})], True
        return [], False

    def _block_delta(self, data: dict) -> tuple[list[ProviderEvent], bool]:
        delta = data.get("delta")
        if not isinstance(delta, dict):
            return [], False
        kind = delta.get("type")
        if kind == "text_delta":
            text = str(delta.get("text") or "")
            if text:
                self.emitted_content = True
                self._text.append(text)
                return [ProviderTextDelta(delta=text)], False
        elif kind == "thinking_delta":
            thinking = str(delta.get("thinking") or "")
            if thinking:
                self.emitted_content = True
                self._thinking.append(thinking)
                return [ProviderThinkingDelta(delta=thinking)], False
        elif kind == "input_json_delta":
            builder = self._builders.setdefault(int(data.get("index", 0) or 0), _ToolUseBuilder())
            builder.arguments.append(str(delta.get("partial_json") or ""))
            self.emitted_content = True
        return [], False

    def finalize(self) -> list[ProviderEvent]:
        calls = [builder.build(index) for index, builder in sorted(self._builders.items())]
        events: list[ProviderEvent] = [ProviderToolCall(tool_call=call) for call in calls]
        events.append(
            ProviderResponseEnd(
                message=AssistantMessage(
                    content="".join(self._text) or None,
                    thinking="".join(self._thinking) or None,
                    tool_calls=calls,
                    usage=self._usage,
                ),
                finish_reason=self._finish_reason,
            )
        )
        return events


class _ToolUseBuilder:
    """One tool_use block growing across input_json_delta chunks."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []

    def build(self, index: int) -> ToolCall:
        text = "".join(self.arguments)
        parsed = loads_object(text) if text else {}
        return ToolCall(
            id=clean_tool_call_id(self.id) if self.id else f"call_{index}",
            name=self.name,
            arguments=parsed if parsed is not None else {"_raw": text},
        )


def _with_output(usage: Usage, reported: Any) -> Usage:
    """The running usage with the counts a later event revised, leaving the rest where it was."""
    if not isinstance(reported, dict):
        return usage
    merged = usage.model_dump()
    later = usage_from_anthropic(reported).model_dump()
    for key, value in later.items():
        if value:
            merged[key] = value
    merged["reasoning"] = min(merged.get("reasoning", 0), merged.get("output", 0))
    return Usage.model_validate(merged)
