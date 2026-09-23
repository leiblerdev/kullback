"""The provider contract the loop streams through: tau_agent/provider.py's `ModelProvider`.

The loop knows a provider, not a model: something that answers `stream_response` with a stream of
assistant message events (`kullback.ai.stream`, which already has tau's event shape). That is the
seam a real SSE provider lands behind without the loop changing.

`ModelStream` is temporary. It adapts today's `kullback.ai.Model` (one `query` per turn, the
stream assembled from the reply) to the protocol, so the Builder, the Examiner, the user and the
Runner keep handing a `Model` to a harness while stream 2b builds the provider layer in
`kullback.ai`. When that lands, `provider_for` returns the real provider and this class goes.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol, Sequence, runtime_checkable

from kullback.agent.messages import Message
from kullback.ai.provider import Model, ModelConfig
from kullback.ai.stream import StreamEvent, stream

__all__ = ["CancellationToken", "ModelProvider", "ModelStream", "provider_for"]


@runtime_checkable
class CancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        """Whether the stream in flight should stop."""
        ...


@runtime_checkable
class ModelProvider(Protocol):
    """One model response as a stream of assistant message events."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Message],
        tools: Optional[Sequence[dict]] = None,
        config: Optional[ModelConfig] = None,
        signal: Optional[CancellationToken] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one assistant response. `session_id` is a routing hint a provider may ignore."""
        ...


class ModelStream:
    """The temporary adapter from `kullback.ai.Model` to `ModelProvider`; see the module docstring."""

    def __init__(self, model: Model):
        self.model = model
        self.name = getattr(model, "name", None) or type(model).__name__

    def stream_response(
        self,
        *,
        model: str = "",
        system: str = "",
        messages: Sequence[Message] = (),
        tools: Optional[Sequence[dict]] = None,
        config: Optional[ModelConfig] = None,
        signal: Optional[CancellationToken] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        # `model`, `signal` and `session_id` are the protocol's; today's Model carries its own id,
        # the loop checks cancellation between steps, and no adapter routes by session.
        del model, signal, session_id
        return stream(self.model, list(messages), tools=tools, system=system or None, config=config)


def provider_for(model: Model | ModelProvider) -> ModelProvider:
    """The provider to stream through: a provider is passed through, a Model is adapted."""
    if hasattr(model, "stream_response"):
        return model  # type: ignore[return-value]
    return ModelStream(model)  # type: ignore[arg-type]
