"""The OpenAI-shaped endpoints, streamed: chat completions and the Responses API.

Mirrors tau_ai/openai_compatible.py. The body is built by the synchronous handle in provider.py, so
there is one request shaping for the blocking call and the streamed one and a fix to either reaches
both; only the transport and the reading of the answer are here.

Kullback reaches openrouter, the cheaper inference hosts, OpenCode Go and any local server through
this module, because they all speak one of these two shapes. Which shape a model takes is the
handle's business (provider.RESPONSES_API_MODELS), not this module's.
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
from kullback.ai.http_errors import ProviderError
from kullback.ai.messages import AssistantMessage, Message, ToolCall, to_wire
from kullback.ai.provider import (
    CancellationToken,
    ModelConfig,
    OpenAIResponsesModel,
    require_live_calls_enabled,
)
from kullback.ai.stream import canonicalize_provider_stream
from kullback.ai.tool_call_ids import clean_tool_call_id
from kullback.ai.usage import Usage, usage_from_openai_chat, usage_from_openai_responses

# The reasoning field names a chat endpoint puts a thinking delta under, in the order they are read.
THINKING_DELTA_KEYS = ("reasoning_content", "reasoning", "thinking")
# The Responses API says the same thing under several event names.
RESPONSES_TEXT_EVENTS = ("response.output_text.delta", "response.refusal.delta")
RESPONSES_THINKING_EVENTS = ("response.reasoning_summary_text.delta", "response.reasoning_text.delta")


class OpenAICompatibleProvider:
    """One OpenAI-shaped endpoint behind the provider contract, reading SSE as it arrives.

    It holds the handle rather than a config of its own: the handle already resolved the base URL,
    the key variable, the headers a host asks for and the read budget, and it builds the body.
    """

    def __init__(self, handle: Any, *, client: Optional[httpx.AsyncClient] = None,
                 max_retries: Optional[int] = None):
        self.handle = handle
        self.responses = isinstance(handle, OpenAIResponsesModel)
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
                system=system, messages=messages, tools=tools, signal=signal,
                session_id=session_id, config=config,
            ),
            model=model or self.handle.name,
            # Chat completions interleaves its text and reasoning channels freely; the Responses
            # API sends its items in order, so a channel switch there really does close a block.
            independent_channels=not self.responses,
        )

    def stream_provider_events(
        self,
        *,
        system: str = "",
        messages: Sequence[Message] = (),
        tools: Sequence[dict] = (),
        signal: Optional[CancellationToken] = None,
        session_id: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """The endpoint's own report, before anything is canonicalised."""
        require_live_calls_enabled()
        settings = (config or ModelConfig()).model_copy(deep=True)
        if session_id and not settings.prompt_cache_key:
            settings.prompt_cache_key = session_id
        wire, tool_list = to_wire(messages, system or None), list(tools) or None
        return self._stream_learning_shape(wire, tool_list, settings, signal)

    async def _stream_learning_shape(self, wire: list, tools: Optional[list], settings: ModelConfig,
                                     signal: Optional[CancellationToken]) -> AsyncIterator[ProviderEvent]:
        """The stream, posted once more when a 400 names a shape field the handle can adjust.

        The same one-retry rule as HttpModel.query: the handle keeps what it learned, so the next
        call never pays the 400.
        """
        payload = self._payload(wire, tools, settings)
        async for event in self._post_stream(payload, signal):
            if isinstance(event, ProviderErrorEvent) and (event.data or {}).get("status") == 400:
                error = ProviderError(event.message, status=400, body=(event.data or {}).get("body"))
                retry = self._payload(wire, tools, settings) if self.handle.learn_shape(error, settings) else payload
                if retry != payload:
                    async for again in self._post_stream(retry, signal):
                        yield again
                    return
            yield event

    def _payload(self, wire: list, tools: Optional[list], settings: ModelConfig) -> dict:
        payload = dict(self.handle.build_body(wire, tools, settings))
        payload["stream"] = True
        if not self.responses:
            # Without this the final chunk carries no usage at all, and a Run would record a call
            # that cost nothing.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _post_stream(self, payload: dict, signal: Optional[CancellationToken]) -> AsyncIterator[ProviderEvent]:
        return stream_sse_events(
            client=self._get_client,
            url=self.handle.base_url + self.handle.path,
            headers=self.handle.headers(),
            payload=payload,
            parser_factory=ResponsesStreamParser if self.responses else ChatStreamParser,
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


class ChatStreamParser:
    """The `/chat/completions` SSE chunks."""

    def __init__(self) -> None:
        self.emitted_content = False
        self.fatal = False
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._builders: dict[int, _ToolCallBuilder] = {}
        self._finish_reason: Optional[str] = None
        self._usage: Optional[Usage] = None

    def feed(self, chunk: str) -> tuple[list[ProviderEvent], bool]:
        if chunk == "[DONE]":
            return [], True
        data = loads_object(chunk)
        if data is None:
            self.fatal = True
            return [ProviderErrorEvent(message="the endpoint sent a chunk that is not JSON")], True

        # The usage chunk stream_options asks for carries usage at the top level and often no
        # choices at all; some gateways hang it on the choice instead.
        if isinstance(data.get("usage"), dict):
            self._usage = usage_from_openai_chat(data["usage"])
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if choice is None:
            return [], False
        if not isinstance(data.get("usage"), dict) and isinstance(choice.get("usage"), dict):
            self._usage = usage_from_openai_chat(choice["usage"])
        self._finish_reason = choice.get("finish_reason") or self._finish_reason

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return [], False
        events: list[ProviderEvent] = []
        text = delta.get("content")
        if isinstance(text, str) and text:
            self.emitted_content = True
            self._text.append(text)
            events.append(ProviderTextDelta(delta=text))
        thinking = _thinking_delta(delta)
        if thinking:
            self.emitted_content = True
            self._thinking.append(thinking)
            events.append(ProviderThinkingDelta(delta=thinking))
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            self.emitted_content = True
            self._builders.setdefault(int(call.get("index", 0) or 0), _ToolCallBuilder()).add(call)
        return events, False

    def finalize(self) -> list[ProviderEvent]:
        calls = [builder.build(index) for index, builder in sorted(self._builders.items())]
        events: list[ProviderEvent] = [ProviderToolCall(tool_call=call) for call in calls]
        events.append(
            ProviderResponseEnd(
                message=AssistantMessage(
                    content="".join(self._text) or None,
                    thinking="".join(self._thinking) or None,
                    tool_calls=calls,
                    usage=self._usage or Usage(),
                ),
                finish_reason=self._finish_reason,
            )
        )
        return events


class ResponsesStreamParser:
    """The `/responses` SSE events, which name themselves rather than sending bare chunks."""

    def __init__(self) -> None:
        self.emitted_content = False
        self.fatal = False
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._builders: dict[str, _ToolCallBuilder] = {}
        self._order: list[str] = []
        self._status: Optional[str] = None
        self._usage: Optional[Usage] = None

    def feed(self, chunk: str) -> tuple[list[ProviderEvent], bool]:
        # This endpoint has no [DONE] sentinel: it ends on one of the terminal events below.
        data = loads_object(chunk)
        if data is None:
            return [], False
        kind = data.get("type")
        if not isinstance(kind, str):
            return [], False

        if kind in RESPONSES_TEXT_EVENTS:
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                self.emitted_content = True
                self._text.append(delta)
                return [ProviderTextDelta(delta=delta)], False
        elif kind in RESPONSES_THINKING_EVENTS:
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                self.emitted_content = True
                self._thinking.append(delta)
                return [ProviderThinkingDelta(delta=delta)], False
        elif kind in ("response.output_item.added", "response.output_item.done"):
            self._register(data.get("item"))
        elif kind == "response.function_call_arguments.delta":
            builder = self._builder(data.get("item_id"))
            if builder is not None:
                builder.arguments.append(str(data.get("delta") or ""))
                self.emitted_content = True
        elif kind == "response.function_call_arguments.done":
            builder = self._builder(data.get("item_id"))
            if builder is not None and isinstance(data.get("arguments"), str):
                builder.arguments = [data["arguments"]]
        elif kind in ("response.completed", "response.incomplete"):
            response = data.get("response") or {}
            self._status = (response.get("status") if isinstance(response, dict) else None) or "completed"
            if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                self._usage = usage_from_openai_responses(response["usage"])
            return [], True
        elif kind in ("response.failed", "error"):
            self.fatal = True
            return [ProviderErrorEvent(message=_responses_failure(data), data={"event": data})], True
        return [], False

    def finalize(self) -> list[ProviderEvent]:
        calls = [self._builders[item_id].build(index) for index, item_id in enumerate(self._order)]
        events: list[ProviderEvent] = [ProviderToolCall(tool_call=call) for call in calls]
        events.append(
            ProviderResponseEnd(
                message=AssistantMessage(
                    content="".join(self._text) or None,
                    thinking="".join(self._thinking) or None,
                    tool_calls=calls,
                    usage=self._usage or Usage(),
                ),
                finish_reason=self._status,
            )
        )
        return events

    def _register(self, item: Any) -> None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        item_id = str(item.get("id") or item.get("call_id") or "")
        if not item_id:
            return
        builder = self._builders.get(item_id)
        if builder is None:
            builder = self._builders[item_id] = _ToolCallBuilder()
            self._order.append(item_id)
        builder.id = str(item.get("call_id") or item.get("id") or builder.id)
        builder.name = str(item.get("name") or builder.name)
        if isinstance(item.get("arguments"), str) and item["arguments"]:
            builder.arguments = [item["arguments"]]
        self.emitted_content = True

    def _builder(self, item_id: Any) -> Optional["_ToolCallBuilder"]:
        if not isinstance(item_id, str) or not item_id:
            return None
        builder = self._builders.get(item_id)
        if builder is None:
            builder = self._builders[item_id] = _ToolCallBuilder()
            self._order.append(item_id)
        return builder


class _ToolCallBuilder:
    """One tool call growing across chunks. The arguments are parsed once, when they are whole."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []

    def add(self, delta: dict) -> None:
        call_id = delta.get("id")
        if isinstance(call_id, str) and call_id:
            self.id = call_id
        function = delta.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str) and name:
            self.name = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments.append(arguments)

    def build(self, index: int) -> ToolCall:
        text = "".join(self.arguments)
        parsed = loads_object(text) if text else {}
        # Arguments that are not JSON are handed on as they came: a tool call the model meant to
        # make is better read as an unparsable argument than dropped from the turn.
        return ToolCall(
            id=clean_tool_call_id(self.id) if self.id else f"call_{index}",
            name=self.name,
            arguments=parsed if parsed is not None else {"_raw": text},
        )


def _thinking_delta(delta: dict) -> str:
    for key in THINKING_DELTA_KEYS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _responses_failure(data: dict) -> str:
    for holder in (data.get("response"), data):
        if isinstance(holder, dict):
            error = holder.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
    message = data.get("message")
    return str(message) if message else "the Responses API ended without an answer"
