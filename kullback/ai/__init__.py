"""The provider layer: one contract over every model, the streams it answers in, pricing, usage.

Mirrors tau_ai. `ModelProvider` is the contract and `stream_response` is the whole of it: a
response arrives as the canonical events of `events.py`, whether it came token by token off a live
endpoint or whole from a memo, a recording or a script. Imports nothing of ours outside this
package (D121).

`Model` and `ModelReply` below are the older synchronous handle, kept importable while the builder,
the examiner, the user and the runner still call `model.query`; `replay.ReplayProvider` is how one
is used behind the contract in the meantime.
"""

# ruff: noqa: F401 - this module is the package's facade and re-exports on purpose

from kullback.ai.anthropic import AnthropicProvider
from kullback.ai.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from kullback.ai.http_errors import ContextOverflowError, ProviderError, RetryExhausted
from kullback.ai.messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from kullback.ai.model_catalog import CatalogModel, ModelCatalog, catalog_from_snapshot
from kullback.ai.model_limits import ModelLimits, limits_for
from kullback.ai.openai_compatible import OpenAICompatibleProvider
from kullback.ai.provider import (
    CancellationToken,
    MemoModel,
    Model,
    ModelConfig,
    ModelProvider,
    ModelReply,
    RecordedModel,
    TestModel,
    live_model,
    live_provider,
    model_for,
    provider_for,
)
from kullback.ai.replay import ReplayProvider, provider_of
from kullback.ai.stream import canonicalize_provider_stream, stream
from kullback.ai.usage import Usage

__all__ = [name for name in globals() if not name.startswith("_")]
