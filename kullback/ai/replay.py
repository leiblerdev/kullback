"""The replay seam: a stored reply answered through the same contract a live endpoint answers.

tau has nothing like this, and Kullback cannot lose it: deterministic replay is what makes a Run
reproducible and what lets a build be re-read without paying for it again. MemoModel, RecordedModel
and TestModel are handles that already hold a whole reply; `ReplayProvider` puts one behind
`ModelProvider`, reporting it as the adapter report a live stream would have made, so the agent core
sees one interface for live and replayed models and the canonical events are the same ones.

The blocking `query` runs in a worker thread, as `stream.stream` does, so a memo read off disk does
not hold the loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional, Sequence

from kullback.ai._provider_events import ProviderErrorEvent, ProviderEvent
from kullback.ai.events import StreamEvent
from kullback.ai.messages import Message, to_wire
from kullback.ai.provider import CancellationToken, Model, ModelConfig
from kullback.ai.stream import canonicalize_provider_stream, reply_events


class ReplayProvider:
    """One handle behind the provider contract. Live or replayed, the caller sees one interface."""

    def __init__(self, model: Model):
        self.model = model

    @property
    def name(self) -> str:
        return getattr(self.model, "name", "model")

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
        """One stored reply as canonical events, in the order a live stream would have said them."""
        return canonicalize_provider_stream(
            self.stream_provider_events(
                system=system, messages=messages, tools=tools, signal=signal, config=config,
            ),
            model=model or self.name,
        )

    async def stream_provider_events(
        self,
        *,
        system: str = "",
        messages: Sequence[Message] = (),
        tools: Sequence[dict] = (),
        signal: Optional[CancellationToken] = None,
        config: Optional[ModelConfig] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """The report a stored reply makes: a start, one delta per block, an end."""
        wire = to_wire(messages, system or None)
        tool_list = list(tools) or None
        prefix = f"call_{len(messages)}"
        try:
            reply = await asyncio.to_thread(self.model.query, wire, tool_list, config)
        # CancelledError is a BaseException on the Python this package requires, so a cancellation
        # passes through here: that is the caller stopping the run, not a provider failure.
        except Exception as exc:  # noqa: BLE001 - the provider is an isolation boundary
            yield ProviderErrorEvent(message=f"{type(exc).__name__}: {exc}")
            return
        if signal is not None and signal.is_cancelled():
            return
        async for event in reply_events(reply, call_id_prefix=prefix):
            yield event


def provider_of(model: Any) -> Any:
    """A handle as a provider, and a provider as itself. One call wherever either may arrive."""
    if hasattr(model, "stream_response"):
        return model
    return ReplayProvider(model)
