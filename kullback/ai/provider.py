"""The provider contract every model answers through, and the model handle the Harness holds.

Two things live here, mirroring tau_ai/provider.py plus what Kullback still needs beside it.

`ModelProvider` is the contract: `stream_response` yields the canonical events of `events.py` as
the provider produces them. The streaming adapters (`openai_compatible.py`, `anthropic.py`) and the
replay seam (`replay.py`) implement it, and the agent core sees one interface for live and replayed
models.

`Model`, `ModelReply` and the adapters under them are the older synchronous handle: one blocking
`query` that returns a whole reply. It is temporary. Every caller listed in the overhaul brief
(builder, examiner, user, runner) still holds a `Model`, so the handle stays until they take a
`ModelProvider` instead; `replay.ReplayProvider` is how a handle is used behind the new contract in
the meantime.

The three offline models tests are allowed to use (TestModel, RecordedModel, MemoModel) are handles
and live here with the live adapters, because `ADAPTERS`, `model_for` and the live-call switch are
read off this module by name from elsewhere in the Harness.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional, Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

# ruff: noqa: F401 - the imports below that this module does not itself use are the re-exports
# named in the note under them: the rest of the Harness spells them kullback.ai.provider.X.
from kullback.ai import calllog, warmup
from kullback.ai.cache import CACHE_TTLS, MAX_CACHE_POINTS, cache_control, count_cache_points
from kullback.ai.events import StreamEvent
from kullback.ai.http import (
    CONNECT_TIMEOUT_S,
    DEFAULT_READ_TIMEOUT_S,
    MODEL_TIMEOUT_ENV_VAR,
    REQUEST_ID_HEADERS,
    Posted,
    body_hash,
    json_body,
    model_read_timeout_s,
    request_id_of,
    request_timeout,
    timeout_note,
)
from kullback.ai.http_errors import (
    CONTEXT_OVERFLOW_MARKERS,
    ContextOverflowError,
    ModelMismatch,
    ProviderError,
    RetryExhausted,
    error_text,
    is_context_overflow,
)
from kullback.ai.limits import Bucket, RateSpec, bucket_for, note_cooldown
from kullback.ai.messages import Message
from kullback.ai.model_limits import RequestRules, request_rules_for, split_vendor
from kullback.ai.retry import (
    RetryPolicy,
    backoff_delay,
    rate_limit_delay,
    retry_after_seconds,
    retryable_status,
)
from kullback.ai.tool_call_ids import ID_ALLOWED, ID_DIGEST_LEN, clean_tool_call_id
from kullback.ai.usage import (
    Usage,
    hidden_thinking,
    reasoning_of,
    reasoning_share,
    usage_from_anthropic,
    usage_from_interactions,
    usage_from_openai_chat,
    usage_from_openai_responses,
)

# The names below were defined here before the modules above were carved out of this one. They
# stay importable from `kullback.ai.provider` because the rest of the Harness spells them that way.
_error_text = error_text
_retryable_status = retryable_status
_reasoning_share = reasoning_share
_reasoning_of = reasoning_of

# Tests never call a real model. Only a real adapter checks this flag; TestModel, RecordedModel
# and MemoModel ignore it: the first two never leave the machine and the third only forwards to
# the model it wraps.
ALLOW_MODEL_REQUESTS = False

# The harness default for every role: Opus 5.5 on Bedrock's global profile (founder decision 2026-09-24).
DEFAULT_MODEL = "bedrock/global.anthropic.claude-opus-5-5"

# The one way to turn live calls on: a person exports this before running the CLI. There is
# no flag a module can set by accident, and the default above stays False.
LIVE_ENV_VAR = "HARNESS_ALLOW_MODEL_REQUESTS"


class ToolCallRequest(BaseModel):
    """A tool call the model asked for."""

    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    name: str
    arguments: dict = Field(default_factory=dict)


class Exchange(BaseModel):
    """One provider exchange as it went over the wire: what was sent, how long, how many tries.

    Recorded so a stored Run says what was asked of the provider and not only what came back: the
    sampling a renderer has to reproduce, fingerprints of the prompt it has to rebuild, and the
    timing, attempts, read timeout in force, and provider request id an incident is read from (D159).
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: Optional[str] = None
    endpoint: Optional[str] = None  # base_url + path, never a key
    wire_id: Optional[str] = None
    sampling: dict = Field(default_factory=dict)  # the request body minus messages, tools, input, system
    n_messages: int = 0
    messages_hash: Optional[str] = None
    n_tools: int = 0
    tools_hash: Optional[str] = None
    wall_ms: float = 0.0
    attempts: int = 0
    status: Optional[int] = None
    request_id: Optional[str] = None
    read_timeout_s: Optional[float] = None  # the read budget in force on this call, in seconds


class ModelReply(BaseModel):
    """What one model call returned."""

    model_config = ConfigDict(populate_by_name=True)

    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: Optional[str] = None
    stop_reason: Optional[str] = None
    raw: Optional[dict] = None
    # What the adapter sent to get this reply. None on the three offline models: TestModel and
    # RecordedModel never touch a wire, and a MemoModel hit carries the original call's exchange.
    exchange: Optional[Exchange] = None
    # What the model reasoned before it answered, when the endpoint reports a reasoning summary.
    # Kept for a reader and never sent back: replaying reasoning we did not produce would be
    # fabrication, so no adapter echoes it. Omitted from a dump when absent, by the same rule as
    # thinking_blocks. Readers must use attribute access, never key access on a dumped dict.
    thinking: Optional[str] = None
    # The Messages API's signed thinking blocks, exactly as they came, for the next request of the
    # conversation to send back unchanged (see AnthropicModel.parse_reply). A reply without them
    # omits the key when dumped, so a memo or recording written before the field is unchanged.
    thinking_blocks: Optional[list[dict]] = None

    @model_serializer(mode="wrap")
    def _omit_absent_thinking(self, handler):
        data = handler(self)
        if self.thinking is None:
            data.pop("thinking", None)
        if self.thinking_blocks is None:
            data.pop("thinking_blocks", None)
        return data


TEMPERATURE = 1.0  # every model call outside the frozen runner runs at 1; 0 buys no determinism on a reasoning model


class ModelConfig(BaseModel):
    """Call parameters. The Runner passes the recorded values; there is no output clamp here.

    The three reasoning fields are typed rather than left to `extra`, because an extra field
    is silently dropped by every adapter and a Run recorded with thinking on would then be
    replayed with thinking off.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: list[str] = Field(default_factory=list)
    # Anthropic: {"type": "adaptive"} on the current models. budget_tokens is rejected on
    # Opus 5 and Sonnet 5, so it is not offered here.
    thinking: Optional[dict] = None
    # Anthropic: output_config.effort, one of low, medium, high, xhigh, max.
    effort: Optional[str] = None
    # OpenAI: reasoning_effort on the reasoning models.
    reasoning_effort: Optional[str] = None
    # OpenAI: prompt_cache_key, so the provider routes calls that share a prefix to the same
    # cache. Set once per build and stage (build.py); the Anthropic adapter ignores it, since it
    # caches by cache_control points instead (cache_system, cache_last_two below).
    prompt_cache_key: Optional[str] = None
    # Anthropic: how long a cache point lives, "5m" (the default, a bare mark) or "1h" (cache.py
    # says when each pays). OpenAI-shaped endpoints cache on their own and take no such field, so
    # their adapters ignore it. budget.BudgetedModel fills it from the stage when it is not set.
    cache_ttl: Optional[str] = None
    # Logprobs, for a training renderer that needs them (D159). OpenAI chat takes both fields, the
    # Responses API takes top_logprobs and an include entry, and Anthropic offers neither, so its
    # adapter ignores both. Whatever comes back lands in the reply's `raw` and nowhere else.
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # Whether the endpoint may, must or must not answer with a tool call: auto, required, none
    # (D222). Every chat and Responses endpoint carries the field under this name and Anthropic
    # carries the same three answers under its own words, so the caller states one thing and each
    # adapter spells it. It is sent only with tools, because a choice with nothing to choose from
    # is refused. A gateway that does not know the field refuses the whole request, which is why
    # the caller that forces a call also has a path for the refusal.
    tool_choice: Optional[str] = None

    @field_validator("tool_choice")
    @classmethod
    def _known_tool_choice(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in TOOL_CHOICES:
            raise ValueError(f"tool_choice is one of {', '.join(TOOL_CHOICES)}, not {value!r}")
        return value

    @field_validator("cache_ttl")
    @classmethod
    def _known_cache_ttl(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in CACHE_TTLS:
            raise ValueError(f"cache_ttl is one of {', '.join(CACHE_TTLS)}, not {value!r}")
        return value


# What a caller may ask of the endpoint about tool calling. `required` is the one D222 forces.
TOOL_CHOICES = ("auto", "required", "none")

# Anthropic spells the same three answers its own way on the Messages API.
ANTHROPIC_TOOL_CHOICE = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}


@runtime_checkable
class CancellationToken(Protocol):
    """Anything that can say the Run behind a stream was cancelled."""

    def is_cancelled(self) -> bool:
        ...


@runtime_checkable
class ModelProvider(Protocol):
    """The contract every model answers through: one response, streamed as canonical events.

    `stream_response` returns the iterator without awaiting anything, so the caller decides when
    the first byte is asked for. It never raises for a provider failure: a refusal, a dead host or
    an exhausted retry arrives as a StreamError event, because the loop has to record it in the
    transcript rather than unwind through it.
    """

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict],
        signal: Optional[CancellationToken] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        ...


class Model:
    """The interface every model goes through. Code that needs a model takes one; it never builds one."""

    name: str = "model"

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        raise NotImplementedError


def require_live_calls_enabled() -> None:
    """Every real adapter calls this before touching the network."""
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError(
            "live model requests are off (kullback.ai.provider.ALLOW_MODEL_REQUESTS is False); "
            f"use TestModel or RecordedModel, or export {LIVE_ENV_VAR}=1 for a live Candidate Run"
        )


def load_dotenv(path: Path = Path(".env"), env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file into the environment, without overriding what is set.

    Blank lines and # comments are skipped, an optional `export ` prefix and matching quotes are
    stripped. Returns the variables that were added. A missing file adds nothing.
    """
    values = os.environ if env is None else env
    added: dict[str, str] = {}
    if not Path(path).is_file():
        return added
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in values:
            values[key] = added[key] = value
    return added


def live_calls_requested(env: Optional[dict[str, str]] = None) -> bool:
    """Whether the environment asks for live calls. Reads the switch, never sets it."""
    values = os.environ if env is None else env
    return str(values.get(LIVE_ENV_VAR, "")).strip().lower() in ("1", "true", "yes", "on")


def enable_live_calls_from_env(env: Optional[dict[str, str]] = None) -> bool:
    """Turn live calls on when a person asked for it in the environment. Returns the new state.

    Called from `live_model` below and nowhere else in the package, so a module cannot reach the
    network by importing its way there.
    """
    global ALLOW_MODEL_REQUESTS
    ALLOW_MODEL_REQUESTS = live_calls_requested(env)
    return ALLOW_MODEL_REQUESTS


class TestModel(Model):
    """A scripted model: hand it the replies it should give, in order."""

    __test__ = False  # pytest must not collect this as a test class

    def __init__(self, replies: Optional[Iterable[Any]] = None, name: str = "test", loop: bool = False):
        self.name = name
        self.loop = loop
        self.replies: list[ModelReply] = [_as_reply(r) for r in (replies or [])]
        self.calls: list[dict] = []
        self.index = 0

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "config": config})
        if not self.replies:
            # An empty reply here would let a module that must not call the model call it and
            # still pass its test. A test that wants a silent model scripts one, with loop=True.
            raise IndexError("TestModel was given no replies, so this call was not expected")
        if self.index >= len(self.replies):
            if not self.loop:
                raise IndexError(f"TestModel ran out of replies after {self.index} calls")
            self.index = 0
        reply = self.replies[self.index]
        self.index += 1
        return reply.model_copy(deep=True)


class RecordedModel(Model):
    """Replays the assistant messages of a stored Run, in order, from its JSONL file."""

    def __init__(self, run_jsonl_path: str | Path, name: str = "recorded"):
        self.name = name
        self.path = Path(run_jsonl_path)
        self.replies: list[ModelReply] = _read_assistant_replies(self.path)
        self.calls: list[dict] = []
        self.index = 0

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "config": config})
        if self.index >= len(self.replies):
            raise IndexError(
                f"RecordedModel ran out of recorded assistant messages after {self.index} calls "
                f"({self.path})"
            )
        reply = self.replies[self.index]
        self.index += 1
        return reply.model_copy(deep=True)


class MemoModel(Model):
    """A content-addressed, on-disk memo of model replies, so a repeat request never reaches the network.

    Key: sha256 of the model id, the normalized messages, the tools and the config, as sorted-key
    JSON, so a byte-identical request always finds the same file regardless of process. Value: the
    reply, stored under `<workdir>/model_cache/<hash>.json`. A hit is returned with its usage
    zeroed, so budget.py's BudgetedModel (which wraps this) prices it at zero; `hits` and `calls`
    count on the wrapper itself, and `last_hit` is the per-call marker budget.py reads into its own
    `memo_hits` bucket, since records.py's Record base forbids an unlisted extra field on a stored
    reply.

    Never wrap a Candidate's live model in this (build.py's run_batch does not): a Candidate's
    answer has to be a fresh sample, and a memoized one would turn a sample into a replay.
    """

    CACHE_DIR = "model_cache"

    def __init__(self, inner: Model, workdir: str | Path):
        self.inner = inner
        self.name = getattr(inner, "name", "model")
        self.dir = Path(workdir) / self.CACHE_DIR
        self.calls = 0
        self.hits = 0
        # D118: the Builder queries one MemoModel from several threads, and budget.py reads
        # last_hit right after its own call returns, so the flag is per thread, not per instance.
        self._local = threading.local()
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}

    @property
    def last_hit(self) -> bool:
        return bool(getattr(self._local, "hit", False))

    @last_hit.setter
    def last_hit(self, value: bool) -> None:
        self._local.hit = bool(value)

    def _key(self, messages: list[dict], tools: Optional[list[dict]], config: Optional[ModelConfig]) -> str:
        settings = (config or ModelConfig()).model_dump(mode="json")
        # An unset cache TTL is left out, so the key of every request stored before the field
        # existed is the key it had: the TTL is how long the provider keeps a prefix, not the ask.
        if settings.get("cache_ttl") is None:
            settings.pop("cache_ttl", None)
        payload = {
            "model": self.name,
            "messages": normalize_messages(messages),
            "tools": tools or [],
            "config": settings,
        }
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        key = self._key(messages, tools, config)
        with self._lock:
            self.calls += 1
            # One request in flight per key (D118): a second thread asking the same thing waits
            # for the first answer and reads it as a hit, rather than paying for it twice.
            gate = self._inflight.setdefault(key, threading.Lock())
        with gate:
            path = self._path(key)
            if path.is_file():
                with self._lock:
                    self.hits += 1
                self.last_hit = True
                reply = ModelReply.model_validate(json.loads(path.read_text(encoding="utf-8")))
                reply.usage = Usage()  # a hit costs nothing; the stored usage is kept on disk, not here
                return reply
            self.last_hit = False
            reply = self.inner.query(messages, tools=tools, config=config)
            self.dir.mkdir(parents=True, exist_ok=True)
            # Written whole under a temporary name and renamed, so a reader never sees half a reply.
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            tmp.write_text(reply.model_dump_json(), encoding="utf-8")
            tmp.replace(path)
            return reply


def _as_reply(item: Any) -> ModelReply:
    """Accept a ModelReply, a dict in reply shape, or a bare string of content."""
    if isinstance(item, ModelReply):
        return item
    if isinstance(item, str):
        return ModelReply(content=item)
    if isinstance(item, dict):
        return _reply_from_dict(item)
    raise TypeError(f"cannot make a ModelReply from {type(item).__name__}")


def _content_and_calls(data: dict) -> tuple[Optional[str], list[dict]]:
    """Content as text and the tool calls, whether the message is a string or a block list.

    Anthropic and Claude Code JSONL record an assistant message as a list of blocks, so a
    stored Run in that shape has to replay too.
    """
    content = data.get("content")
    calls = list(data.get("tool_calls") or [])
    if not isinstance(content, list):
        return content, calls
    text: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text.append(str(block))
        elif block.get("type") == "tool_use":
            calls.append({"id": block.get("id"), "name": block.get("name"), "arguments": block.get("input")})
        elif block.get("type") in TEXT_BLOCK_TYPES:
            text.append(str(block.get("text") or block.get("thinking") or ""))
    return ("".join(text) or None), calls


def _reply_from_dict(data: dict) -> ModelReply:
    """One assistant message or reply payload, in trace shape or reply shape, as a ModelReply."""
    if data.get("object") == "interaction":
        # A stored Interactions API payload is read the way the live adapter reads it, so a
        # recorded reply is priced and replayed the same as the call that made it.
        return interactions_reply(data, data.get("model"))
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    content, calls = _content_and_calls(data)
    output = int(usage.get("output", usage.get("completion_tokens", 0)) or 0)
    # A stored chat payload may carry thinking only in total_tokens; bill it as usage_from_openai_chat does.
    hidden = 0 if "output" in usage else hidden_thinking(usage, "prompt_tokens", "completion_tokens")
    reasoning = max(_reasoning_share(usage, output), hidden)
    output += hidden
    return ModelReply(
        content=content,
        thinking=data.get("thinking"),
        thinking_blocks=copy.deepcopy(data.get("thinking_blocks"))
        if isinstance(data.get("thinking_blocks"), list)
        else None,
        tool_calls=[
            ToolCallRequest(
                id=c.get("id"),
                name=c.get("name") or (c.get("function") or {}).get("name", ""),
                arguments=_arguments_of(c),
            )
            for c in calls
        ],
        usage=usage_from_interactions(usage) if "total_input_tokens" in usage else Usage(
            input=int(usage.get("input", usage.get("prompt_tokens", 0)) or 0),
            output=output,
            cache_read=int(usage.get("cache_read", 0) or 0),
            cache_write=int(usage.get("cache_write", 0) or 0),
            reasoning=reasoning,
        ),
        model=data.get("model"),
        stop_reason=data.get("stop_reason") or data.get("finish_reason"),
    )


def _arguments_of(call: dict) -> dict:
    args = call.get("arguments")
    if args is None:
        args = (call.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
    return args if isinstance(args, dict) else {}


def _read_assistant_replies(path: Path) -> list[ModelReply]:
    """Read a Run JSONL: model_call events, or plain assistant messages, both in file order."""
    replies: list[ModelReply] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("type") == "model_call":
                payload = data.get("payload") or {}
                replies.append(_reply_from_dict(payload.get("reply") or payload))
            elif data.get("role") == "assistant":
                replies.append(_reply_from_dict(data))
    return replies


# --- Extension point: real provider adapters ---
# Anthropic, OpenAI and OpenAI-compatible adapters, keeping the Model interface above
# unchanged (D97, design section 4 item 20): model id as provider/model, wire id and base
# URL per model, keys from the environment, retry on 5xx, network errors and rate limits
# only, prompt-cache points, usage on every reply for budget.py. Every adapter calls
# require_live_calls_enabled() first.


ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
CACHE_CONTROL = cache_control()
TEXT_BLOCK_TYPES = ("text", "reasoning", "thinking")
# The Messages API's reasoning blocks, which a later request sends back byte for byte.
THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")
# A wire assistant message may carry the reply's thinking blocks under this key; only the Messages
# API shape reads it, and the OpenAI shapes' key whitelist drops it.
THINKING_BLOCKS_KEY = "thinking_blocks"
# Said in the system prompt when a caller asks for a tool call from a model that refuses a forced
# tool_choice (model_limits.RequestRules.forced_tool_choice): `auto` goes on the wire instead.
FORCED_TOOL_LINE = "Answer this turn with a tool call."
# The keys of a request body that are the prompt itself rather than the sampling: each adapter
# builds a different body, so the sampling is what is left after these come out, never a list of
# the fields worth keeping (a field nobody enumerated is a field a Run would not record).
PROMPT_BODY_KEYS = ("messages", "tools", "input", "system", "instructions", "system_instruction")
# The Responses API returns logprobs only for what the include list asks for.
RESPONSES_LOGPROBS_INCLUDE = "message.output_text.logprobs"
# An empty user turn cannot be dropped (that would end the request on an assistant message,
# which the current Anthropic API rejects: prefill is gone on 4.6 and later), and it cannot be
# sent empty either, so it goes as one placeholder block.
EMPTY_USER_PLACEHOLDER = "(no content)"
def split_model_id(model_id: str) -> tuple[str, str]:
    """'anthropic/claude-opus-5' into the provider and the wire id sent on the wire."""
    provider, _, wire_id = model_id.partition("/")
    if not wire_id:
        raise ValueError(f"model id must be 'provider/model', got {model_id!r}")
    return provider, wire_id


def substitute_env(text: str, env: dict[str, str]) -> str:
    """Fill ${VAR} from the environment. A missing variable is an error, not an empty string."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in env:
            raise KeyError(f"{name} is not set, and {text!r} needs it")
        return env[name]

    return ENV_VAR.sub(replace, text)


def strip_unpaired_surrogates(text: str) -> str:
    """Drop lone surrogates; some providers reject a body that carries them."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def strip_surrogates_deep(value: Any) -> Any:
    if isinstance(value, str):
        return strip_unpaired_surrogates(value)
    if isinstance(value, dict):
        return {k: strip_surrogates_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_surrogates_deep(v) for v in value]
    return value


def _is_empty_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return not block
    if block.get("type") in THINKING_BLOCK_TYPES and (block.get("signature") or block.get("data")):
        # A signed thinking block comes back with empty text under the default display and still
        # has to be echoed; only the signature (or the redacted payload) says it is real.
        return False
    if block.get("type") in TEXT_BLOCK_TYPES:
        text = block.get("text") or block.get("thinking") or ""
        return not str(text).strip()
    return False


def _clean_ids(block: dict) -> dict:
    for key in ("id", "tool_use_id", "tool_call_id"):
        if key in block and block[key] is not None:
            block[key] = clean_tool_call_id(block[key])
    return block


def normalize_messages(messages: list[dict]) -> list[dict]:
    """The three defensive fixes: no lone surrogates, clean tool ids, no empty messages or reasoning parts.

    Only an empty assistant message is dropped. An empty user message becomes a placeholder
    block instead, because dropping the last one would leave the request ending on an
    assistant turn, which the current API reads as a prefill and rejects.
    """
    out: list[dict] = []
    for message in strip_surrogates_deep(copy.deepcopy(messages)):
        message = _clean_ids(dict(message))
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                _clean_ids(b) if isinstance(b, dict) else b for b in content if not _is_empty_block(b)
            ]
            empty = not message["content"]
        else:
            empty = not str(content or "").strip()
        for call in message.get("tool_calls") or []:
            _clean_ids(call)
        if empty and not message.get("tool_calls"):
            if message.get("role") == "assistant":
                continue
            message["content"] = [{"type": "text", "text": EMPTY_USER_PLACEHOLDER}]
        out.append(message)
    return out


def cache_tools(tools: list[dict], ttl: Optional[str] = None) -> list[dict]:
    """A cache point on the last tool: tools render first, so this caches the whole tool list.

    Tools and system usually change at different rates (a role's tools are fixed for a build, a
    session's system prompt can differ per session), so each gets its own point (prompt-caching.md,
    Placement patterns); a call whose system changed still reads the tools.
    """
    out = copy.deepcopy(tools)
    if out:
        out[-1] = {**out[-1], "cache_control": cache_control(ttl)}
    return out


def cache_system(system: Any, ttl: Optional[str] = None) -> list[dict]:
    """A cache point on the system prompt: it is the same on every call of a build."""
    blocks = [{"type": "text", "text": system}] if isinstance(system, str) else copy.deepcopy(system or [])
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": cache_control(ttl)}
    return blocks


def cache_last_two(messages: list[dict], ttl: Optional[str] = None) -> list[dict]:
    """Cache points on the last two non-system messages, so a growing conversation reuses its prefix.

    The last one is where the next call's read starts; the one before it is where this call reads
    from, when the previous call wrote its point there. A block that cannot carry a mark (a thinking
    block, an empty text) is skipped for the one before it in the same message.
    """
    out = copy.deepcopy(messages)
    marked = 0
    for message in reversed(out):
        if marked >= 2 or message.get("role") == "system":
            continue
        blocks = message.get("content")
        if not isinstance(blocks, list):
            continue
        # A thinking block cannot carry a cache point, and an empty text block is refused with one,
        # so the point goes on the last block that can.
        last = next((i for i in range(len(blocks) - 1, -1, -1) if isinstance(blocks[i], dict)
                     and blocks[i].get("type") not in THINKING_BLOCK_TYPES and not _is_empty_block(blocks[i])),
                    None)
        if last is not None:
            blocks[last] = {**blocks[last], "cache_control": cache_control(ttl)}
            marked += 1
    return out


class HttpModel(Model):
    """Shared plumbing for the HTTP adapters: ids, keys, the retry loop, one httpx client."""

    key_env_var = ""
    default_base_url = ""
    path = "/"
    key_required = True

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        wire_id: Optional[str] = None,
        client: Any = None,
        retry: Optional[RetryPolicy] = None,
        env: Optional[dict[str, str]] = None,
        sleep: Any = None,
        rng: Optional[random.Random] = None,
        timeout: Optional[float] = None,
    ):
        self.name = model_id
        self.provider, derived = split_model_id(model_id)
        self.wire_id = wire_id or derived
        self.env = dict(os.environ) if env is None else dict(env)
        self.base_url = substitute_env(base_url or self.default_base_url, self.env).rstrip("/")
        self.api_key = api_key or self.env.get(self.key_env_var)
        self.retry = retry or RetryPolicy()
        # The read budget, resolved once here: explicit argument, then the environment,
        # then 300 s. Reads and writes share it; only establishing contact stays short, so
        # the post below carries a split timeout object rather than one number.
        self.timeout = model_read_timeout_s(self.env, timeout)
        self.request_timeout = request_timeout(self.timeout)
        self.sleep = sleep or time.sleep
        self.rng = rng or random.Random()
        self._client = client
        self._client_lock = threading.Lock()
        # Request-shape fields an endpoint's 400 taught this instance (see `shape_adjustment`).
        self.shape_fixes: set[str] = set()

    @functools.cached_property
    def rate_bucket(self) -> Optional[Bucket]:
        """The shared limiter for this handle's provider key, or None when the catalogue names
        no rate for it: every request takes a slot from it first. Read on the first request
        and cached after, so construction never fetches. The bucket is process-wide, read off
        the same snapshot model_for resolves the host from. A provider gains a limiter by
        gaining a rate row, with no branch per provider here. The lookups below already
        tolerate anything unreadable, so no limiter, not a failed call, is what comes back."""
        from kullback.ai import pricing

        spec = pricing.rate_from_catalog(
            pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=self.env), self.name)
        if spec is None:
            return None
        host = urlparse(self.base_url).hostname or self.base_url
        fingerprint = hashlib.sha256(self.api_key.encode()).hexdigest()[:12] if self.api_key else ""
        return bucket_for(provider=self.provider, host=host, key=fingerprint,
                          key_name=self.key_env_var or "", model=self.name, spec=spec)

    @classmethod
    def for_model(cls, model_id: str) -> type:
        """The adapter class that serves this id; a provider serving several shapes picks here."""
        return cls

    @classmethod
    def sent_wire_id(cls, wire_id: str) -> str:
        """The id this adapter puts on the wire for the part after `provider/`; most send it as is."""
        return wire_id

    @classmethod
    def credential_vars(cls) -> tuple[tuple[str, ...], ...]:
        """The environment variables that authenticate this adapter: any one group, fully set, is enough.

        One place for the names, so the missing-key error and the TUI's /login status say the same
        thing. Most adapters read one key; an adapter that signs requests reads several.
        """
        return ((cls.key_env_var,),) if cls.key_env_var else ()

    def has_credentials(self) -> bool:
        return bool(self.api_key)

    def missing_key_message(self) -> str:
        return f"no API key for {self.name}; set {self.key_env_var} or pass api_key"

    def encode_body(self, body: dict) -> bytes:
        """The exact bytes posted: httpx's own JSON encoding, spelled here so a signer sees them, with sorted keys so
        two calls that send the same tools and messages send the same bytes (a provider caches a byte prefix)."""
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                          sort_keys=True).encode("utf-8")

    def client(self) -> Any:
        # httpx.Client is safe to share across threads; creating it is the one step that is
        # not, so the first caller makes it and the rest wait (D118).
        with self._client_lock:
            if self._client is None:
                require_live_calls_enabled()
                self._client = httpx.Client()
            return self._client

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        require_live_calls_enabled()
        if self.key_required and not self.has_credentials():
            raise ProviderError(self.missing_key_message())
        config = config or ModelConfig()
        body = self.build_body(messages, tools, config)
        with warmup.gate(self.name, messages, tools), calllog.attempts() as retried:
            return self._query(messages, tools, config, body, retried)

    def _query(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig, body: dict,
               retried: list) -> ModelReply:
        started, sent_at = time.monotonic(), time.time()
        posted = reply = None
        spent = 0  # the HTTP requests a first post made before a shape retry
        try:
            try:
                posted = self._post_full(body)
            except ProviderError as error:
                # One silent retry when the 400 named a shape field this instance can adjust; the
                # adjustment stays, so the next call never pays the 400.
                retry = self.build_body(messages, tools, config) if self.learn_shape(error, config) else body
                if retry == body:
                    raise
                body, spent = retry, error.attempts
                posted = self._post_full(body)
                posted = posted._replace(attempts=posted.attempts + spent)
            wall_ms, finished_at = (time.monotonic() - started) * 1000.0, time.time()
            reply = self.parse_reply(posted.data)
            self.check_reply(posted)
        except Exception as error:
            # A reply refused after it was paid for (check_reply) is logged with its usage.
            attempts = posted.attempts if posted is not None else spent + (getattr(error, "attempts", None) or 1)
            if posted is None and isinstance(error, ProviderError):
                error.attempts = attempts  # a Run's error event says the same count
            times = (sent_at, time.time())  # the meta below can wait on the price table
            self._log(messages, tools, reply.usage if reply is not None else None, attempts,
                      (time.monotonic() - started) * 1000.0, f"{type(error).__name__}: {str(error)[:200]}",
                      self._row_meta(posted, body, config, retried), times)
            raise
        reply.exchange = self.exchange_of(body, posted, wall_ms)
        self._log(messages, tools, reply.usage, posted.attempts, wall_ms, None,
                  self._row_meta(posted, body, config, retried), (sent_at, finished_at))
        return reply

    def _row_meta(self, posted: Optional[Posted], body: dict, config: ModelConfig, retried: list) -> dict:
        """The row's extras, or none when they cannot be built: the log never costs a call its answer or its error."""
        try:
            return self._meta(posted, body, config, retried)
        except Exception:  # noqa: BLE001
            return {}

    def _meta(self, posted: Optional[Posted], body: dict, config: ModelConfig, retried: list) -> dict:
        """The row's extras: the temperature the body carried and the effort asked (what was sent, which a gateway
        may still drop), the status of every failed HTTP attempt (a 429 a retry hid included), and the reply's own (reply_meta)."""
        meta = {"temperature_sent": body.get("temperature"), "effort_sent": config.reasoning_effort or config.effort}
        if warmup.waited_ms():
            meta["warmup_wait_ms"] = warmup.waited_ms()  # held back while an earlier call warmed the shared prefix
        if retried:
            meta["failed_statuses"] = list(retried)
        return {**meta, **self.reply_meta(posted)}

    def check_reply(self, posted: Posted) -> None:
        """Raise when the answer is not one this call may use (a gateway that answered with another model)."""

    def names_a_model(self, posted: Posted) -> bool:
        """Whether the answer so far says which model gave it; a stream learns it part way (streamlog.py)."""
        return False

    def reply_meta(self, posted: Optional[Posted]) -> dict:
        """What the call log keeps from the reply besides its usage (a gateway's request id); none by default."""
        return {}

    def _log(self, messages: list[dict], tools: Optional[list[dict]], usage: Any, attempts: int, wall_ms: float,
             error: Optional[str] = None, meta: Optional[dict] = None,
             times: Optional[tuple[float, float]] = None) -> None:
        """The call handed to the process's call sink (calllog.py): the one place every live call is logged."""
        system = "\n\n".join(_text_of(m) for m in messages if m.get("role") == "system")
        rest = [m for m in messages if m.get("role") != "system"]
        calllog.emit(self.name, self.provider, system, rest, tools, usage, attempts, wall_ms, error, meta, times)

    def exchange_of(self, body: dict, posted: Posted, wall_ms: float) -> Exchange:
        """What this call sent and how it went, from the body itself and the response's own headers.

        The prompt is fingerprinted rather than copied: the Run already carries the transcript, and
        a count and a hash are what a renderer needs to say it rebuilt the same request (D159).
        """
        prompt = body.get("messages") or body.get("input") or []
        tools = body.get("tools") or []
        return Exchange(
            provider=self.provider,
            endpoint=self.base_url + self.path,
            wire_id=self.wire_id,
            sampling={key: value for key, value in body.items() if key not in PROMPT_BODY_KEYS},
            n_messages=len(prompt),
            messages_hash=body_hash(prompt),
            n_tools=len(tools),
            tools_hash=body_hash(tools),
            wall_ms=wall_ms,
            attempts=posted.attempts,
            status=posted.status,
            request_id=request_id_of(posted.headers),
            read_timeout_s=self.timeout,
        )

    def post(self, body: dict) -> dict:
        """The JSON one post answered with. The retry rules live in `_post_full` below."""
        return self._post_full(body).data

    def _post_full(self, body: dict) -> Posted:
        # The gate sits here, on the network path itself, not only on query(): a caller that
        # builds a body and posts it must not reach the transport while live calls are off.
        require_live_calls_enabled()
        url = self.base_url + self.path
        content = self.encode_body(body)
        bucket = self.rate_bucket
        for attempt in range(1, self.retry.attempts + 1):
            last_attempt = attempt == self.retry.attempts
            # One slot per request: taken just before the post, handed back with the
            # response read, so a retry counts against the minute cap and waits the
            # shared cooldown like any other caller, and a backoff sleep holds no slot.
            try:
                with bucket.slot() if bucket is not None else nullcontext():
                    # After the slot, per attempt: a signed request carries its time, and a
                    # queue before it must not send a signature the endpoint stopped accepting.
                    headers = self.headers(content)
                    response = self.client().post(url, headers=headers, content=content, timeout=self.request_timeout)
            except httpx.HTTPError as exc:
                calllog.retried(type(exc).__name__)
                if last_attempt:
                    # A timeout names the budget that was in force, so the log line that lands
                    # says whether the answer was slow or the host was down.
                    suffix = timeout_note(exc, CONNECT_TIMEOUT_S, self.timeout)
                    raise RetryExhausted(f"{self.name}: {self.retry.attempts} attempts failed: {exc}{suffix}",
                                         attempts=attempt) from exc
                self.sleep(backoff_delay(attempt, self.retry, self.rng))
                continue
            if response.status_code < 400:
                try:
                    return Posted(response.json(), response.status_code, attempt, response.headers)
                except ValueError as exc:
                    # A 2xx that is not JSON is a proxy or gateway page, not an answer. It is
                    # a transport fault, so it retries like one instead of escaping as a
                    # JSONDecodeError from the middle of query().
                    error = ProviderError(
                        f"{self.name}: HTTP {response.status_code} body is not JSON: {response.text[:200]}",
                        status=response.status_code,
                        body=response.text,
                    )
                    if last_attempt:
                        raise RetryExhausted(
                            f"{self.name}: {self.retry.attempts} attempts failed: {error}",
                            status=response.status_code,
                            body=response.text,
                            attempts=attempt,
                        ) from exc
                    self.sleep(backoff_delay(attempt, self.retry, self.rng))
                    continue
            error = self.error_for(response)
            error.attempts = attempt
            calllog.retried(response.status_code)
            if isinstance(error, ContextOverflowError) or not retryable_status(response.status_code):
                raise error
            note_cooldown(bucket, response.status_code,
                          rate_limit_delay(response.headers, attempt, self.retry, self.rng))
            if last_attempt:
                raise RetryExhausted(
                    f"{self.name}: {self.retry.attempts} attempts failed: {error}",
                    status=response.status_code,
                    body=error.body,
                    attempts=attempt,
                ) from error
            wait = retry_after_seconds(response.headers)
            if wait is not None and wait > self.retry.max_retry_after_s:
                raise RetryExhausted(
                    f"{self.name}: provider asked for {wait:.0f}s, over the "
                    f"{self.retry.max_retry_after_s:.0f}s this build will wait: {error}",
                    status=response.status_code,
                    body=error.body,
                    attempts=attempt,
                ) from error
            self.sleep(backoff_delay(attempt, self.retry, self.rng) if wait is None else wait)
        raise RetryExhausted(f"{self.name}: no attempts were made", attempts=0)

    def learn_shape(self, error: ProviderError, config: ModelConfig) -> bool:
        """Whether this error taught a request-shape adjustment. Only the OpenAI shape learns."""
        return False

    def error_for(self, response: Any) -> ProviderError:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        text = f"{self.name}: HTTP {response.status_code}: {error_text(body)}"
        if is_context_overflow(body):
            return ContextOverflowError(text, status=response.status_code, body=body)
        return ProviderError(text, status=response.status_code, body=body)

    def headers(self, body: Optional[bytes] = None) -> dict:
        """The headers of one request. `body` is the exact bytes that will be posted, for an adapter
        that signs them; the others ignore it."""
        raise NotImplementedError

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        raise NotImplementedError

    def parse_reply(self, data: dict) -> ModelReply:
        raise NotImplementedError


def _collect_anthropic_thinking(block: dict, thinking: list[dict], thinking_text: list[str]) -> bool:
    """File one thinking block. True when the block was thinking and is now filed."""
    if block.get("type") not in THINKING_BLOCK_TYPES:
        return False
    # Kept whole, signature and all: the next request of a tool loop sends them back
    # unchanged (preserved thinking), and an edited block is a 400. The readable
    # text goes on the reply beside them, so a reader sees the summary.
    if block.get("signature") or block.get("data"):
        thinking.append(copy.deepcopy(block))
    summary = block.get("thinking")
    if isinstance(summary, str):
        thinking_text.append(summary or "")
    return True


class AnthropicModel(HttpModel):
    """Anthropic Messages API: system split out, cache points, tool_use blocks."""

    key_env_var = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com"
    path = "/v1/messages"
    api_version = "2023-06-01"
    default_max_tokens = 4096

    def headers(self, body: Optional[bytes] = None) -> dict:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        # What this model refuses, read from one table by capability (model_limits.REQUEST_RULES),
        # so the same model is asked the same way on every host that serves it.
        rules = request_rules_for(self.wire_id)
        system, converted = _to_anthropic(messages)
        ttl = config.cache_ttl
        # The four cache points, in render order: tools, system, then the last two messages. They
        # sit on blocks only: a top-level cache_control field is refused by the legacy Bedrock stack.
        body: dict[str, Any] = {
            "model": self.wire_id,
            "max_tokens": config.max_tokens or self.default_max_tokens,
            "messages": cache_last_two(normalize_messages(converted), ttl),
        }
        if system:
            body["system"] = cache_system(system, ttl)
        if tools:
            body["tools"] = cache_tools([_anthropic_tool(t) for t in tools], ttl)
            _put_anthropic_tool_choice(body, config.tool_choice, rules)
        if config.temperature is not None and rules.sampling:
            body["temperature"] = config.temperature
        if config.stop:
            body["stop_sequences"] = list(config.stop)
        # Reasoning branch one of four: Anthropic takes thinking as its own block and the
        # depth as output_config.effort. budget_tokens is not sent: the current models reject it.
        thinking = _anthropic_thinking(config.thinking, rules)
        if thinking:
            body["thinking"] = thinking
        if config.effort:
            body["output_config"] = {"effort": config.effort}
        # config.logprobs and config.top_logprobs are deliberately not sent: the Messages API has
        # no logprobs field, and a field it does not know makes it refuse the whole request.
        points = count_cache_points(body)
        if points > MAX_CACHE_POINTS:  # pragma: no cover - the placement above makes at most four
            raise ValueError(f"{points} cache points in one request; the Messages API takes {MAX_CACHE_POINTS}")
        return body

    def parse_reply(self, data: dict) -> ModelReply:
        text: list[str] = []
        calls: list[ToolCallRequest] = []
        thinking: list[dict] = []
        thinking_text: list[str] = []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if _collect_anthropic_thinking(block, thinking, thinking_text):
                continue
            if block.get("type") == "text":
                text.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCallRequest(
                        id=clean_tool_call_id(block.get("id")),
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )
        return ModelReply(
            content="".join(text) or None,
            thinking="".join(thinking_text) or None,
            tool_calls=calls,
            usage=usage_from_anthropic(data.get("usage")),
            model=data.get("model") or self.wire_id,
            stop_reason=data.get("stop_reason"),
            raw=data,
            thinking_blocks=thinking or None,
        )


# Claude in Amazon Bedrock: the Messages API on AWS. The anthropic SDK's
# src/anthropic/lib/bedrock/_mantle.py gave the bearer variables (line 38), the bearer header
# (line 234) and anthropic-version (line 242). The host and the ids are the founder's live probes
# of 2026-09-24 in us-east-2: bedrock-mantle.<region>.api.aws answers 404 "The model does not
# exist" for every id, while bedrock-runtime.<region>.amazonaws.com/anthropic/v1/messages answers
# 200 with the same headers, and only for an inference profile id (us. or global.): the bare
# anthropic.<model> is refused, on-demand throughput is not supported for it.
BEDROCK_SERVICE = "bedrock"
# Global profiles are billed at the list price, a regional one about ten percent over it (the
# founder, 2026-09-24), so a bare id goes out on the global profile in every region.
BEDROCK_DEFAULT_PROFILE = "global."
BEDROCK_DEFAULT_REGION = "us-east-2"
BEDROCK_BEARER_VARS = ("AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AWS_API_KEY")
BEDROCK_KEY_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
BEDROCK_SESSION_VAR = "AWS_SESSION_TOKEN"


def bedrock_region(env: dict[str, str]) -> str:
    """AWS_REGION, then AWS_DEFAULT_REGION (the SDK's order), else the Harness default."""
    return env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or BEDROCK_DEFAULT_REGION


# A bare Bedrock model id, `vendor.model` with no profile in front: one dot, no ARN.
_BARE_BEDROCK_ID = re.compile(r"[a-z0-9-]+\.[a-z0-9:-]+")


def bedrock_wire_id(wire_id: str) -> str:
    """The id Bedrock is sent: a profile id or an ARN unchanged, a bare id on the global profile."""
    return BEDROCK_DEFAULT_PROFILE + wire_id if _BARE_BEDROCK_ID.fullmatch(wire_id) else wire_id


def _put_anthropic_tool_choice(body: dict[str, Any], choice: Optional[str], rules: RequestRules) -> None:
    """Set the Messages API tool_choice, moving a forced choice into the prompt where it is refused."""
    if choice == "required" and not rules.forced_tool_choice:
        # A forced choice is a 400 on this model: `auto` goes on the wire and the demand
        # moves into the prompt, after the cache point so the cached system stays reusable.
        choice = "auto"
        body["system"] = list(body.get("system") or []) + [{"type": "text", "text": FORCED_TOOL_LINE}]
    if choice:
        body["tool_choice"] = dict(ANTHROPIC_TOOL_CHOICE[choice])


def _anthropic_thinking(requested: Optional[dict], rules: RequestRules) -> dict:
    """The thinking block to send, empty when the model's rules leave the field out.

    Adaptive thinking asks for the summarized display, so the thinking text comes back
    readable instead of encrypted: without it the block carries only a signature.
    """
    thinking = dict(requested or {})
    if rules.thinking_always_on:
        # Disabled or budgeted thinking is a 400 here; leaving the field out is adaptive.
        thinking.pop("budget_tokens", None)
        if thinking.get("type") == "adaptive":
            thinking.setdefault("display", "summarized")
        elif not thinking:
            thinking = {"type": "adaptive", "display": "summarized"}
        else:
            thinking = {}
    elif thinking.get("type") == "adaptive":
        thinking.setdefault("display", "summarized")
    return thinking


class BedrockAuth:
    """What every vendor on Bedrock shares: the runtime host of a region, the wire id, the keys.

    Mixed in front of a vendor's own adapter, which keeps its body, its parsing and its stream. The
    vendor's route on the host is `bedrock_route`; the wire id is whatever follows `bedrock/`, a
    profile id sent as is and a bare `vendor.model` put on the `global.` profile, which is the one
    served on demand. Two ways in, in the SDK's order for keys read from the environment: a Bedrock
    API key (AWS_BEARER_TOKEN_BEDROCK) goes as a bearer token, else the access key pair signs every
    request with SigV4 over the exact body bytes. Keys come from the environment or .env only: no
    profile files, no instance metadata, no STS.
    """

    key_env_var = BEDROCK_BEARER_VARS[0]
    bedrock_route = ""

    def __init__(self, model_id: str, base_url: Optional[str] = None, **kwargs):
        env = dict(os.environ) if kwargs.get("env") is None else dict(kwargs["env"])
        self.region = bedrock_region(env)
        default = f"https://bedrock-runtime.{self.region}.amazonaws.com/{self.bedrock_route}"
        super().__init__(model_id, base_url=base_url or default, **kwargs)
        self.wire_id = self.sent_wire_id(self.wire_id)
        self.api_key = self.api_key or next((self.env[v] for v in BEDROCK_BEARER_VARS if self.env.get(v)), None)
        self.access_key = self.env.get(BEDROCK_KEY_VARS[0]) or None
        self.secret_key = self.env.get(BEDROCK_KEY_VARS[1]) or None
        self.session_token = self.env.get(BEDROCK_SESSION_VAR) or None

    @classmethod
    def sent_wire_id(cls, wire_id: str) -> str:
        return bedrock_wire_id(wire_id)

    @classmethod
    def credential_vars(cls) -> tuple[tuple[str, ...], ...]:
        return ((BEDROCK_BEARER_VARS[0],), BEDROCK_KEY_VARS)

    def has_credentials(self) -> bool:
        return bool(self.api_key or (self.access_key and self.secret_key))

    def missing_key_message(self) -> str:
        return (f"no AWS credentials for {self.name}; set {BEDROCK_BEARER_VARS[0]}, or "
                f"{' and '.join(BEDROCK_KEY_VARS)} (plus {BEDROCK_SESSION_VAR} for temporary keys), "
                f"and AWS_REGION if not {BEDROCK_DEFAULT_REGION}")

    def authorized(self, headers: dict, body: Optional[bytes]) -> dict:
        """The vendor's headers with the bearer token on, or signed with SigV4 over the body."""
        if self.api_key:
            return {**headers, "authorization": f"Bearer {self.api_key}"}
        if not (self.access_key and self.secret_key):
            raise ProviderError(self.missing_key_message())
        from kullback.ai import sigv4

        return sigv4.sign("POST", self.base_url + self.path, headers, body or b"",
                          access_key=self.access_key, secret_key=self.secret_key,
                          session_token=self.session_token, region=self.region, service=BEDROCK_SERVICE)


class BedrockAnthropicModel(BedrockAuth, AnthropicModel):
    """Claude through Amazon Bedrock's Messages endpoint: the Anthropic body, AWS authentication.

    The body, the cache points and the model rules are AnthropicModel's unchanged; the host, the
    wire id and the keys are BedrockAuth's. The adapter `bedrock/` ids resolve to (ADAPTERS): a
    model of another vendor is handed to that vendor's Bedrock adapter (for_model).
    """

    bedrock_route = "anthropic"
    path = "/v1/messages"

    @classmethod
    def for_model(cls, model_id: str) -> type:
        """The Bedrock adapter for this id's vendor: OpenAI's models speak Chat Completions there."""
        vendor = split_vendor(split_model_id(model_id)[1])
        return BEDROCK_VENDOR_ADAPTERS.get(vendor[0] if vendor else "", cls)

    def headers(self, body: Optional[bytes] = None) -> dict:
        return self.authorized({"anthropic-version": self.api_version, "content-type": "application/json"}, body)


# gpt-<major> or o<digit>, after an optional gateway prefix such as 'openai/'.
_REASONING_WIRE = re.compile(r"(?:.*/)?(?:gpt-(\d+)|o\d)(?![a-z])")


def reasoning_family(wire_id: str) -> bool:
    """Whether a wire id has the reasoning-family shape: gpt-<major> with major 5 or above, or o<digit>."""
    match = _REASONING_WIRE.match(wire_id.lower())
    return bool(match) and (match.group(1) is None or int(match.group(1)) >= 5)


def shape_adjustment(error_text: str) -> Optional[str]:
    """The request-shape field a 400's text asks to change, or None.

    'reasoning_effort': send it as 'none' with tools. 'max_completion_tokens': cap under that name.
    'temperature': drop it.
    """
    text = (error_text or "").lower()
    if "reasoning_effort" in text and ("tool" in text or "'none'" in text or '"none"' in text):
        return "reasoning_effort"
    if "max_completion_tokens" in text:
        return "max_completion_tokens"
    if "temperature" in text:
        return "temperature"
    return None


class OpenAIModel(HttpModel):
    """OpenAI chat completions: messages as given, tool calls with JSON string arguments."""

    key_env_var = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"
    path = "/chat/completions"

    def headers(self, body: Optional[bytes] = None) -> dict:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        body: dict[str, Any] = {
            "model": self.wire_id,
            "messages": [_openai_message(m) for m in strip_surrogates_deep(copy.deepcopy(messages))],
        }
        if tools:
            body["tools"] = [_openai_tool(t) for t in tools]
            if config.tool_choice:
                body["tool_choice"] = config.tool_choice
        if config.max_tokens is not None:
            body[self.token_cap_field()] = config.max_tokens
        if config.temperature is not None and not self._reasoning_family() and "temperature" not in self.shape_fixes:
            body["temperature"] = config.temperature
        if config.seed is not None:
            body["seed"] = config.seed
        if config.stop:
            body["stop"] = list(config.stop)
        if config.prompt_cache_key:
            body["prompt_cache_key"] = config.prompt_cache_key
        if config.logprobs is not None:
            body["logprobs"] = config.logprobs
        if config.top_logprobs is not None:
            body["top_logprobs"] = config.top_logprobs
        body.update(self.reasoning_fields(config))
        wants_none = self._reasoning_family() or "reasoning_effort" in self.shape_fixes
        if tools and wants_none and "reasoning_effort" not in body:
            # Found live: gpt-5.6-luna answers a tool call with HTTP 400 saying function tools and
            # reasoning_effort cannot be combined on /v1/chat/completions unless the effort is
            # 'none'. The endpoint applies a default effort we never sent, so not sending one is
            # not enough; it has to be turned off by name. A caller that did ask for an effort
            # keeps it and gets the 400, because silently downgrading what they asked for would be
            # a worse answer than the error.
            body["reasoning_effort"] = "none"
        return body

    def _reasoning_family(self) -> bool:
        """The OpenAI models that took a different request shape from gpt-4 onwards.

        Found live, not read: gpt-5.6-luna answers `max_tokens` with HTTP 400 telling us to send
        `max_completion_tokens`, and the same families refuse any temperature but the default. The
        test is a shape of the wire id (gpt-<major> with major 5 or above, or o<digit>) rather than
        a list of names: a prefix list went stale the week after it was written. `learn_shape` is the
        safety net when the guess is wrong.
        """
        return reasoning_family(self.wire_id or "")

    def token_cap_field(self) -> str:
        learned = "max_completion_tokens" in self.shape_fixes
        return "max_completion_tokens" if self._reasoning_family() or learned else "max_tokens"

    def learn_shape(self, error: ProviderError, config: ModelConfig) -> bool:
        """Keep the adjustment a 400 names. A caller who asked for an effort keeps it and the 400."""
        if getattr(error, "status", None) != 400:
            return False
        field = shape_adjustment(str(error))
        if field is None or field in self.shape_fixes:
            return False
        if field == "reasoning_effort" and config.reasoning_effort:
            return False
        self.shape_fixes.add(field)
        return True

    def reasoning_fields(self, config: ModelConfig) -> dict:
        """Reasoning branch two of four: OpenAI takes one reasoning_effort field."""
        return {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}

    def parse_reply(self, data: dict) -> ModelReply:
        choices = data.get("choices") or [{}]
        message = (choices[0] or {}).get("message") or {}
        return ModelReply(
            content=message.get("content"),
            tool_calls=[
                ToolCallRequest(
                    id=clean_tool_call_id(call.get("id")),
                    name=call.get("name") or (call.get("function") or {}).get("name") or "",
                    arguments=_arguments_of(call),
                )
                for call in (message.get("tool_calls") or [])
            ],
            usage=usage_from_openai_chat(data.get("usage")),
            model=data.get("model") or self.wire_id,
            stop_reason=(choices[0] or {}).get("finish_reason"),
            raw=data,
        )


class BedrockOpenAIModel(BedrockAuth, OpenAIModel):
    """OpenAI's models through Amazon Bedrock: the Chat Completions body and stream, AWS authentication.

    Bedrock serves them at `/openai/v1/chat/completions` on the same runtime host, in OpenAI's own
    shape: choices, tool calls with JSON string arguments, usage with prompt_tokens_details carrying
    cached_tokens and cache_write_tokens (reviewer's live probe, us-east-2, 2026-09-24). So the body,
    the parsing and the stream are OpenAIModel's; the host, the wire id and the keys are BedrockAuth's.
    """

    bedrock_route = "openai/v1"

    def headers(self, body: Optional[bytes] = None) -> dict:
        return self.authorized({"content-type": "application/json"}, body)

    def _reasoning_family(self) -> bool:
        """The shape test on the model alone: `<profile>.openai.gpt-<n>` is a gpt-<n>."""
        vendor = split_vendor(self.wire_id)
        return reasoning_family(vendor[1] if vendor else self.wire_id or "")

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ) -> ModelReply:
        """A call with tools or a reasoning effort goes to the Responses route; any other stays on chat.

        Bedrock's Chat Completions route refuses function tools together with a reasoning effort
        (HTTP 400, live pilot 2026-10-05); its Responses route on the same host takes both.
        """
        config = config or ModelConfig()
        if tools or config.effort or config.reasoning_effort:
            return self.responses().query(messages, tools, config)
        return super().query(messages, tools, config)

    def responses(self) -> "BedrockOpenAIResponsesModel":
        """The Responses sibling of this adapter: same host, wire id, keys, client and retry rules."""
        sibling = getattr(self, "_responses", None)
        if sibling is None:
            sibling = BedrockOpenAIResponsesModel(
                self.name, api_key=self.api_key, base_url=self.base_url, client=self._client, retry=self.retry,
                env=self.env, sleep=self.sleep, rng=self.rng, timeout=self.timeout)
            self._responses = sibling
        return sibling


# The Bedrock adapter per vendor segment of the wire id; a vendor with no row speaks the Messages API.
# The openai row routes per call: Chat Completions, or Responses when the call carries tools or an effort.
BEDROCK_VENDOR_ADAPTERS: dict[str, type] = {"openai": BedrockOpenAIModel}


# One session id per process for OpenCode's prompt-cache optimization. Stable across the whole
# run on purpose: the same id on every call is what lets the gateway cache, and a fresh id per
# call would look like the abusive traffic the Go docs ask clients not to generate.
_OPENCODE_SESSION = uuid.uuid4().hex


def opencode_headers(base_url: str, headers: dict) -> dict:
    """Identify this client on OpenCode hosts, and only there.

    Go asks clients to identify themselves (no broad user agents) and to send
    `x-opencode-session`; without both, gateway traffic looks abusive and keys get blocked.
    Every adapter that posts to an opencode.ai host calls this; no other provider ever sees
    these headers."""
    if "opencode.ai" in urlparse(base_url).netloc:
        headers["user-agent"] = "kullback"
        headers["x-opencode-session"] = _OPENCODE_SESSION
    return headers


def key_var_for_provider(provider: str) -> str:
    """The variable a provider's key is read from when nothing else names one: PROVIDER_API_KEY.

    One rule, spelled once: the TUI's /login menu shows the person typing the same name this
    reads, so what they are asked to set is what the adapter later looks for. A provider with an
    adapter of its own answers with that adapter's variable (Bedrock's is AWS_BEARER_TOKEN_BEDROCK).
    """
    adapter = ADAPTERS.get(provider)
    if adapter is not None and adapter.key_env_var:
        return adapter.key_env_var
    return f"{provider.upper().replace('-', '_')}_API_KEY" if provider else ""


class OpenAICompatibleModel(OpenAIModel):
    """A local or self-hosted endpoint that speaks the OpenAI shape. Base URL required, key optional."""

    key_required = False

    def __init__(self, model_id: str, base_url: str, key_env_var: Optional[str] = None, **kwargs):
        # A host reached by --base-url is not OpenAI, so it must not inherit OPENAI_API_KEY from
        # OpenAIModel: one person's key would go to another vendor's gateway, and the call that
        # should have failed for want of a key would fail as a rejected one instead. The key is
        # read from PROVIDER_API_KEY, derived from the id, and stays optional, because a local
        # server needs none.
        self.key_env_var = key_var_for_provider(split_model_id(model_id)[0]) \
            if key_env_var is None else key_env_var
        super().__init__(model_id, base_url=base_url, **kwargs)

    def headers(self, body: Optional[bytes] = None) -> dict:
        return opencode_headers(self.base_url, super().headers(body))

    def reasoning_fields(self, config: ModelConfig) -> dict:
        """Reasoning branch three of four: a local endpoint gets none of it. Servers that do
        not know the field reject the whole request, and there is no effort table to guess from."""
        return {}

    def parse_reply(self, data: dict) -> ModelReply:
        """The reply, named by the id this Harness asked under rather than the one echoed back.

        An id is only a model together with its provider. A gateway answers under the upstream name
        it routed to: asked for one flash model it answered 'z-ai/<the same model>', and a name with
        no provider on it is carried by thirty resellers at thirty different rates, so the ledger
        priced the call at nothing and the budget gate failed the Run. The endpoint's own name is
        still in `raw`, and what went on the wire is still on the Exchange, so this loses nothing
        and makes the one field a price is looked up under a name that can be looked up.
        """
        reply = super().parse_reply(data)
        reply.model = self.name
        return reply


class RegistryModel(OpenAICompatibleModel):
    """A provider the models.dev registry names: its host and its key variable, the OpenAI shape.

    This is how `opencode-go/kimi-k3` or `groq/llama-3.3-70b` runs without anyone writing an
    adapter for it. The registry answers where to send the call and which variable holds the key;
    the body is OpenAI's, because the registry is only asked for providers that speak that shape.
    Reasoning fields stay off for the reason the local endpoint leaves them off: a gateway that
    does not know a field refuses the whole request, and the registry lists no effort table.
    """

    def __init__(self, model_id: str, base_url: str, key_env_var: str = "", **kwargs):
        # A provider the registry names a key variable for cannot be reached without that key, so
        # the call is refused by name rather than sent unauthenticated. When it names none, the
        # PROVIDER_API_KEY rule below still finds a key if the person set one.
        self.key_required = bool(key_env_var)
        super().__init__(model_id, base_url=base_url, key_env_var=key_env_var or None, **kwargs)


# The local gateway every call goes through when KULLBACK_ROUTE=omniroute (the direct keys were revoked
# 2026-11-05): OpenAI chat completions on OMNIROUTE_BASE_URL, keyed by OMNIROUTE_API_KEY as a Bearer token.
ROUTE_VAR = "KULLBACK_ROUTE"
OMNIROUTE = "omniroute"
OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
OMNIROUTE_KEY_VAR = "OMNIROUTE_API_KEY"
# Sent on every request. Compression rewrites the prompt (it breaks the provider's prefix cache and changes what
# the model reads), the semantic cache would answer for the model, and memory or skills injection adds text we did
# not send. Each is turned off by the header the gateway documents for it.
OMNIROUTE_HEADERS = {"x-omniroute-compression": "off", "X-OmniRoute-No-Cache": "true",
                     "x-omniroute-no-memory": "true"}
# Our model id to the id the gateway serves it under (its GET /v1/models, probed 2026-11-05); an id not listed
# goes as it is.
OMNIROUTE_IDS: dict[str, str] = {"google/gemini-4-argon": "gemini/gemini-4-argon"}
# The reply headers that say who answered; the gateway falls back to another model on some errors (a 402) by
# default, and an answer from another model must never pass as the one asked.
ANSWERED_MODEL, ANSWERED_PROVIDER = "x-omniroute-model", "x-omniroute-provider"
FALLBACK_HEADER = "x-omniroute-emergency-fallback"
# The id the gateway's call log keys a call by (its correlationId); on an error reply only x-request-id comes back.
# The gateway's own x-omniroute-request-id matches nothing in its log.
CORRELATION_HEADERS = ("x-correlation-id", "x-request-id")
LATENCY_HEADER = "x-omniroute-latency-ms"  # the gateway's own time for a non-stream call
PRICING_PATH = "/api/pricing"  # {route: {model: USD per 1M tokens: input, cached, cache_creation, output}}
# Routes whose provider caches only where the request marks it (Anthropic cache_control, which the gateway keeps
# as sent); the other routes cache a shared prefix on their own and get no marks.
MARKED_ROUTES = ("cc", "bedrock")
# The shortest prefix a mark may close: Claude caches nothing under 512 tokens (Opus 5.5 and Sonnet 5.5; older
# models more). Prompts here run 2.4 characters a token, so 4 a token is a safe floor.
MIN_CACHED_TOKENS, CHARS_PER_TOKEN_FLOOR = 512, 4
MIN_CACHED_CHARS = MIN_CACHED_TOKENS * CHARS_PER_TOKEN_FLOOR
# Routes that cache by a key the request names rather than by marks: a constant key per stage (from the call log's
# run id and stage) in the body's prompt_cache_key and the x-grok-conv-id header, so the calls of one stage land on
# the same cache. Other routes ignore the field, so it is sent only here.
CACHE_KEY_ROUTES = ("cu/grok-",)
CONV_ID_HEADER = "x-grok-conv-id"
# Routes whose Claude models give the thinking text back when asked: with an effort set, it is asked back
# summarized. The gateway's chat completions translator takes this body field for it and removes it before the
# provider sees the request. The subscription route returns no thinking text either way (probe 2026-11-05).
CLAUDE_ROUTES = ("bedrock/us.anthropic.",)
SUMMARIZED_THINKING = {"_omnirouteCopilotReasoningSummary": "summarized"}
# The gateway's own list price of the call (cache-aware); it reads 0 on a stream and on a model with no pricing
# row, which is no price, so the call is then priced from its pricing table (PRICING_PATH) where it has a row.
COST_HEADER = "x-omniroute-response-cost"
NOT_ALLOWED = 403  # the gateway key's allowlist refuses the id
# The gateway's own caps on its subscription routes (live 2026-11-05): calls in flight per route, and 60 requests a
# minute per account. Extra calls queue at the gateway (up to 300 s, depth 50); holding to the same numbers here
# keeps that queue a backstop, not the normal path. One bucket per route, shared by every model on it.
ROUTE_CAPS: dict[str, RateSpec] = {"cc": RateSpec(requests_per_minute=60, in_flight=2),
                                   "cx": RateSpec(requests_per_minute=60, in_flight=2),
                                   "cu": RateSpec(requests_per_minute=60, in_flight=3)}


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


_PRICING: dict[str, dict] = {}
_PRICING_LOCK = threading.Lock()


def pricing_usd(table: dict, wire_id: str, usage: Any) -> Optional[float]:
    """The usage priced from the gateway's pricing table (USD per 1M tokens; reasoning is inside output), or None
    when the table has no row for the route and model or a rate in the row is not a finite number."""
    route, _, model = wire_id.partition("/")
    row = (table.get(route) or {}).get(model) if isinstance(table, dict) else None
    if not isinstance(row, dict) or usage is None:
        return None
    rate = lambda key, default=0.0: float(row.get(key, default) or 0.0)  # noqa: E731
    try:
        usd = (usage.input * rate("input") + usage.cache_read * rate("cached", row.get("input", 0))
               + usage.cache_write * rate("cache_creation", row.get("input", 0)) + usage.output * rate("output")) / 1e6
    except (TypeError, ValueError):
        return None
    return usd if math.isfinite(usd) else None


def _route(wire_id: str) -> str:
    return wire_id.split("/", 1)[0] if "/" in wire_id else ""


def _stage_tag() -> str:
    """The current call's "run_id.stage" (either alone when the other is unset; empty outside both)."""
    context = calllog.current()
    return ".".join(str(context[k]) for k in ("run_id", "stage") if context.get(k))


def marked_steps(messages: list[dict], tools_chars: int = 0) -> list[int]:
    """The messages that carry a cache mark (at most MAX_CACHE_POINTS with the system): the last message, and the
    last two assistant messages before it, where an append-only prompt's next call (the next question on the same
    conversation, the next turn of a loop) shares everything up to. A mark whose prefix (tools, then the messages up
    to and including it) is under the provider's cacheable minimum (MIN_CACHED_CHARS) is left off: it would cache
    nothing and use up a checkpoint."""
    if not messages:
        return []
    last = len(messages) - 1
    earlier = [i for i in range(last) if messages[i].get("role") == "assistant"][-(MAX_CACHE_POINTS - 2):]
    systems = [i for i, m in enumerate(messages) if m.get("role") == "system"][:1]
    sizes, total = [], tools_chars
    for message in messages:
        total += len(_text_of(message))
        sizes.append(total)
    return sorted(i for i in set(systems + earlier + [last]) if sizes[i] >= MIN_CACHED_CHARS)


def _mark(message: dict, ttl: Optional[str] = None) -> dict:
    """The message with an ephemeral cache mark, of the lifetime asked, on its last content block."""
    content = message.get("content")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list) and content:
        blocks = [dict(block) for block in content]
    else:
        return message
    blocks[-1] = {**blocks[-1], "cache_control": cache_control(ttl)}
    return {**message, "content": blocks}


class OmniRouteModel(OpenAICompatibleModel):
    """Any model id reached through the OmniRoute gateway, named by our id so the call is logged under its own provider
    and priced from the gateway's reply (calllog.dollars); only the wire id is the gateway's.

    The reasoning effort is passed through (the gateway forwards it to the provider's thinking mode), cached
    input is read from usage.prompt_tokens_details.cached_tokens and the thinking from
    completion_tokens_details, as every chat completions reply is read (usage_from_openai_chat)."""

    key_required = True

    def __init__(self, model_id: str, base_url: Optional[str] = None, **kwargs):
        env = kwargs.get("env")
        values = os.environ if env is None else env
        super().__init__(model_id, base_url=base_url or values.get("OMNIROUTE_BASE_URL") or OMNIROUTE_BASE_URL,
                         key_env_var=OMNIROUTE_KEY_VAR,
                         wire_id=kwargs.pop("wire_id", None) or OMNIROUTE_IDS.get(model_id, model_id), **kwargs)

    def headers(self, body: Optional[bytes] = None) -> dict:
        """The Bearer key, the headers that keep prompt and answer ours, and the session tag the gateway's
        call_logs keep (<run id>.<stage>, from the call log's context), so its rows match ours."""
        out = {**super().headers(body), **OMNIROUTE_HEADERS}
        tag = _stage_tag()
        if tag:
            out["X-OmniRoute-Session-Id"] = tag
        if self.wire_id.startswith(CACHE_KEY_ROUTES) and self.cache_key():
            out[CONV_ID_HEADER] = self.cache_key()
        return out

    def cache_key(self) -> Optional[str]:
        """The stage's constant cache key: a UUID made from the run id and stage, None outside a stage."""
        tag = _stage_tag()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, tag)) if tag else None

    @functools.cached_property
    def rate_bucket(self) -> Optional[Bucket]:
        """The route's shared limiter (ROUTE_CAPS) on a capped route, else the catalogue's, as for any model."""
        route = _route(self.wire_id)
        spec = ROUTE_CAPS.get(route)
        if spec is None:
            return super().rate_bucket
        host = urlparse(self.base_url).hostname or self.base_url
        fingerprint = hashlib.sha256(self.api_key.encode()).hexdigest()[:12] if self.api_key else ""
        return bucket_for(provider=self.provider, host=host, key=fingerprint, key_name=self.key_env_var or "",
                          model=route, spec=spec)

    def check_reply(self, posted: Posted) -> None:
        """The reply came from the model asked: the gateway's model and provider headers name it, and it did
        not fall back. Anything else is an error, never a silent answer."""
        headers = posted.headers or {}
        provider, _, model = self.wire_id.rpartition("/")
        body_model = posted.data.get("model") if isinstance(posted.data, dict) else None
        # A reply with no model header (a stream) is read by the body's bare model name, under the route asked.
        answered = (headers.get(ANSWERED_PROVIDER) or (provider or None), headers.get(ANSWERED_MODEL) or body_model)
        if headers.get(FALLBACK_HEADER) or answered != (provider or None, model):
            raise ModelMismatch(f"{self.name}: the gateway answered with {answered[0]}/{answered[1]}"
                                f"{' (emergency fallback)' if headers.get(FALLBACK_HEADER) else ''}, "
                                f"not the model asked ({self.wire_id})")

    def names_a_model(self, posted: Posted) -> bool:
        headers = posted.headers or {}
        return bool(headers.get(ANSWERED_MODEL) or headers.get(FALLBACK_HEADER) or posted.data.get("model"))

    def reply_meta(self, posted: Optional[Posted]) -> dict:
        """What the row keeps from the gateway: the route, the model and provider it says answered, the id its call
        log keys the call by (correlation id), the session tag we sent, its own latency, its list price (the cost
        header, else the usage priced from its pricing table), the upstream's billed cost when the usage carries one,
        and reasoning null when the usage reports no thinking count (unknown, never 0)."""
        tag = _stage_tag()
        meta: dict = {"route": _route(self.wire_id) or None, "session_tag": tag or None}
        if posted is None:
            return meta
        headers = posted.headers or {}
        meta.update(answered_model=headers.get(ANSWERED_MODEL), answered_provider=headers.get(ANSWERED_PROVIDER),
                    request_id=next((headers.get(h) for h in CORRELATION_HEADERS if headers.get(h)), None),
                    gateway_ms=_number(headers.get(LATENCY_HEADER)), usd_gateway=_number(headers.get(COST_HEADER)))
        usage = (posted.data or {}).get("usage") if isinstance(posted.data, dict) else None
        if usage is not None:
            meta.update(self._usage_meta(usage, priced=not meta["usd_gateway"]))
        return meta

    def _usage_meta(self, usage: dict, priced: bool) -> dict:
        """From the reply's usage: the billed cost, the list price from the pricing table when the cost header is
        absent or 0 (it reads 0 on a stream, and on a model with no pricing row, which then stays no price), and
        reasoning null when no thinking count is reported."""
        meta = {"usd_billed": _number(usage.get("cost"))}
        if priced and calllog.enabled():  # the table is a blocking read, for a row nobody writes when logging is off
            meta["usd_priced"] = pricing_usd(self.pricing(), self.wire_id, usage_from_openai_chat(usage))
        if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") is None:
            meta["reasoning"] = None
        return meta

    def pricing(self) -> dict:
        """The gateway's pricing table, read once per gateway and process; {} when it cannot be read."""
        root = self.base_url.rsplit("/v1", 1)[0]
        with _PRICING_LOCK:
            if root not in _PRICING:
                try:
                    response = self.client().get(root + PRICING_PATH, headers=self.headers(), timeout=10.0)
                    data = response.json() if response.status_code == 200 else {}
                    _PRICING[root] = data if isinstance(data, dict) else {}
                except (httpx.HTTPError, ValueError):
                    _PRICING[root] = {}
            return _PRICING[root]

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        """The chat completions body, with cache marks on a route that needs them (MARKED_ROUTES) and the thinking
        asked back summarized on a Claude route with an effort set."""
        body = super().build_body(messages, tools, config)
        if self.wire_id.startswith(CACHE_KEY_ROUTES) and self.cache_key():
            body["prompt_cache_key"] = self.cache_key()
        if _route(self.wire_id) in MARKED_ROUTES:
            tools_chars = len(json.dumps(body.get("tools"), sort_keys=True)) if body.get("tools") else 0
            marked = set(marked_steps(body["messages"], tools_chars))
            body["messages"] = [_mark(m, config.cache_ttl) if i in marked else m for i, m in enumerate(body["messages"])]
        if (config.reasoning_effort or config.effort) and self.wire_id.startswith(CLAUDE_ROUTES):
            body.update(SUMMARIZED_THINKING)
        return body

    def error_for(self, response: Any) -> ProviderError:
        """A 403 names the id the gateway key does not allow; never a reason to try another model."""
        error = super().error_for(response)
        if response.status_code == NOT_ALLOWED:
            return ProviderError(f"{self.wire_id} is not allowed by the gateway key ({error})",
                                 status=error.status, body=error.body)
        return error

    def reasoning_fields(self, config: ModelConfig) -> dict:
        effort = config.reasoning_effort or config.effort
        return {"reasoning_effort": effort} if effort else {}


def routed(env: Optional[dict] = None) -> bool:
    """Whether every model goes through the OmniRoute gateway (KULLBACK_ROUTE=omniroute)."""
    values = os.environ if env is None else env
    return (values.get(ROUTE_VAR) or "").strip().lower() == OMNIROUTE


class OpenAIResponsesModel(HttpModel):
    """OpenAI's Responses API: input items in, output items out, one round trip per query.

    Built for the OpenCode Go models the docs serve through /v1/responses (Muse Spark 1.3).
    The Harness above never sees the difference: query() takes messages and tools and returns
    a ModelReply, and the agent loop re-queries with the tool results, exactly as on chat.
    Reasoning items are read, never echoed: replaying encrypted reasoning we did not produce
    would be fabrication, so follow-up turns carry the text and the tool calls, not the blob.
    The body stays minimal for the reason the chat adapters stay minimal: a gateway that does
    not know a field refuses the whole request."""

    path = "/responses"
    key_required = False

    def __init__(self, model_id: str, base_url: str, key_env_var: str = "", **kwargs):
        self.key_env_var = key_env_var
        self.key_required = bool(key_env_var)
        super().__init__(model_id, base_url=base_url, **kwargs)
        # Whether this model reasons comes from its catalogue row, read on the first request
        # and cached after: construction never fetches, so an explicit endpoint pays nothing.
        # A plain chat model behind a gateway may refuse the reasoning field, so the row gates it.
        self.sends_reasoning: Optional[bool] = None
        self._reasoning_known = False

    def headers(self, body: Optional[bytes] = None) -> dict:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return opencode_headers(self.base_url, headers)


    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        body: dict[str, Any] = {
            "model": self.wire_id,
            "input": _responses_input(strip_surrogates_deep(copy.deepcopy(messages))),
        }
        if tools:
            body["tools"] = [_responses_tool(t) for t in tools]
            if config.tool_choice:
                body["tool_choice"] = config.tool_choice
        if config.max_tokens is not None:
            body["max_output_tokens"] = config.max_tokens
        # Reasoning branch four of four: the effort comes from the config and a summary is
        # always asked for, so the reasoning text is kept like the summarized thinking.
        if not self._reasoning_known:
            self.sends_reasoning = _catalogue_reasoning(self.provider, self.wire_id, self.env)
            self._reasoning_known = True
        reasoning = _responses_reasoning(config, self.sends_reasoning)
        if reasoning is not None:
            body["reasoning"] = reasoning
        if config.logprobs or config.top_logprobs is not None:
            if config.top_logprobs is not None:
                body["top_logprobs"] = config.top_logprobs
            include = list(body.get("include") or [])
            if RESPONSES_LOGPROBS_INCLUDE not in include:
                include.append(RESPONSES_LOGPROBS_INCLUDE)
            body["include"] = include
        return body

    def parse_reply(self, data: dict) -> ModelReply:
        if data.get("status") not in (None, "completed", "in_progress"):
            error = data.get("error") or {}
            raise ProviderError(
                f"{self.name}: the Responses API ended as {data.get('status')}: "
                f"{error.get('message') or error or 'no reason given'}"
            )
        output = data.get("output") or []
        texts = _responses_text_of(output, "message", "content", "output_text")
        # The summary text is kept; the encrypted blob never is (read, never echo).
        thinking = _responses_text_of(output, "reasoning", "summary", "summary_text")
        calls: list[ToolCallRequest] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                calls.append(
                    ToolCallRequest(
                        id=clean_tool_call_id(item.get("call_id") or item.get("id")),
                        name=item.get("name") or "",
                        arguments=_arguments_of({"arguments": item.get("arguments")}),
                    )
                )
        return ModelReply(
            content="".join(texts) or None,
            thinking="".join(thinking) or None,
            tool_calls=calls,
            usage=usage_from_openai_responses(data.get("usage")),
            model=data.get("model") or self.wire_id,
            stop_reason=data.get("status"),
            raw=data,
        )

def _catalogue_reasoning(provider: str, wire_id: str, env: dict) -> Optional[bool]:
    """What the catalogue row says about reasoning: True, False, or None when no row says.

    The same lookup price_from_catalog reads from, through the snapshot model_for already
    uses: the provider entry, then the row for the wire id. Anything but a boolean reads
    as no row, so a row that predates the field takes the cautious path below.
    """
    from kullback.ai import pricing
    entry = (pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=env) or {}).get(provider)
    row = pricing.model_row(entry, wire_id)
    verdict = row.get("reasoning") if isinstance(row, dict) else None
    return verdict if isinstance(verdict, bool) else None


def _responses_reasoning(config: ModelConfig, sends: Optional[bool]) -> Optional[dict[str, Any]]:
    """The reasoning field for one Responses request, or None when this model gets none.

    A model whose row says it reasons always asks for a summary; one whose row says it does
    not sends no reasoning field. With no row the field goes only when the config sets an
    effort, since a plain chat model behind a gateway may refuse it.
    """
    if sends is False:
        return None
    effort = config.reasoning_effort or config.effort
    if sends is None and not effort:
        return None
    reasoning: dict[str, Any] = {"summary": "auto"}
    if effort:
        reasoning["effort"] = effort
    return reasoning



def _responses_text_of(output: Any, item_type: str, parts_key: str, text_type: str) -> list[str]:
    """Text parts of one Responses output kind: message content or reasoning summaries."""
    texts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict) or item.get("type") != item_type:
            continue
        for part in item.get(parts_key) or []:
            if isinstance(part, dict) and part.get("type") == text_type:
                texts.append(part.get("text") or "")
    return texts


class BedrockOpenAIResponsesModel(BedrockAuth, OpenAIResponsesModel):
    """OpenAI's models through Amazon Bedrock's `/openai/v1/responses`: the Responses body, AWS authentication.

    The body and the parsing are OpenAIResponsesModel's; the host, the wire id and the keys are
    BedrockAuth's. BedrockOpenAIModel hands it every call that carries tools or a reasoning effort.
    """

    bedrock_route = "openai/v1"
    path = "/responses"

    def __init__(self, model_id: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_id, base_url=base_url, **kwargs)
        # Bedrock never answers without credentials, so a missing key is refused before the call.
        self.key_required = True

    def headers(self, body: Optional[bytes] = None) -> dict:
        return self.authorized({"content-type": "application/json"}, body)


def _responses_tool(tool: dict) -> dict:
    """One function tool in the Responses shape, which matches the chat shape field for field."""
    out = _openai_tool(tool)
    function = out.get("function") or {}
    return {"type": "function", "name": function.get("name", ""),
            "description": function.get("description") or "",
            "parameters": function.get("parameters") or {"type": "object"}}


def _responses_input(messages: list[dict]) -> list[dict]:
    """History into Responses input items: text stays text, tool traffic becomes call items.

    An assistant turn that called tools is replayed as its function_call items (so the model
    sees what it did) plus its text, if any; tool results become function_call_output items
    against the same call ids, which is what keeps a multi-turn tool loop coherent."""
    items: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": clean_tool_call_id(message.get("tool_call_id")),
                          "output": _responses_text(message.get("content"))})
            continue
        calls = message.get("tool_calls") or []
        text = _responses_text(message.get("content"))
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            args = function.get("arguments", call.get("arguments"))
            items.append({"type": "function_call",
                          "call_id": clean_tool_call_id(call.get("id")),
                          "name": function.get("name", call.get("name", "")),
                          "arguments": args if isinstance(args, str) else json.dumps(args or {}, sort_keys=True)})
        if text or not calls:
            items.append({"role": role if role in ("user", "assistant", "system", "developer") else "user",
                          "content": text})
    return items


def _responses_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_responses_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("output_text") or "")
    return str(content)


# Gemini through Google's own Interactions API (ai.google.dev/api/interactions, read 2026-10-28):
# "Generally Available and recommended for all new projects". It replaced the OpenAI-compatible
# endpoint here because that one reports no cached tokens at all and thinking only inside
# total_tokens, while this one reports both under their own names (usage_from_interactions).
# The conversation is a list of typed steps, sent stateless (store false) and in full each turn.
GOOGLE_TOOL_CHOICE = {"auto": "auto", "required": "any", "none": "none"}
# An interaction's status as the stop words the loop reads (stream.normalize_stop_reason).
INTERACTION_STOP_REASONS = {"completed": "stop", "requires_action": "tool_calls", "incomplete": "length"}
INTERACTION_FAILED_STATUSES = ("failed", "cancelled")
# The base the old OpenAI-compatible route was reached on ends in this; the same host serves the
# Interactions API one level up, so a GOOGLE_BASE_URL still set to the old base keeps working.
GOOGLE_COMPAT_SUFFIX = "/openai"


class GoogleInteractionsModel(HttpModel):
    """Google's Interactions API: system_instruction, typed input steps, thought steps echoed back.

    The thought steps of a reply are kept whole, signature and all, as its thinking_blocks, and
    lead that assistant turn on the next request: the endpoint refuses a tool loop whose history
    drops them (HTTP 400, probed 2026-10-28). Temperature is not in the documented
    generation_config but the endpoint accepted it on the same probe, so it is sent when set.
    """

    key_env_var = "GOOGLE_API_KEY"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    path = "/interactions"
    alternate_key_var = "GEMINI_API_KEY"

    def __init__(self, model_id: str, base_url: Optional[str] = None, env: Optional[dict[str, str]] = None,
                 **kwargs):
        values = os.environ if env is None else env
        base = (base_url or values.get(base_url_var_for_provider("google")) or self.default_base_url).rstrip("/")
        super().__init__(model_id, base_url=base.removesuffix(GOOGLE_COMPAT_SUFFIX), env=env, **kwargs)
        self.api_key = self.api_key or self.env.get(self.alternate_key_var)

    @classmethod
    def credential_vars(cls) -> tuple[tuple[str, ...], ...]:
        return ((cls.key_env_var,), (cls.alternate_key_var,))

    def headers(self, body: Optional[bytes] = None) -> dict:
        return {"x-goog-api-key": self.api_key or "", "content-type": "application/json"}

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        system, steps = _to_interactions(messages)
        body: dict[str, Any] = {"model": self.wire_id, "store": False, "input": steps}
        if system:
            body["system_instruction"] = system
        if tools:
            body["tools"] = [_interactions_tool(t) for t in tools]
        generation: dict[str, Any] = {}
        if config.max_tokens:
            # Thought tokens count against this cap, so a small one can end "incomplete" unanswered.
            generation["max_output_tokens"] = config.max_tokens
        if config.temperature is not None:
            generation["temperature"] = config.temperature
        if config.stop:
            generation["stop_sequences"] = list(config.stop)
        if config.seed is not None:
            generation["seed"] = config.seed
        if tools and config.tool_choice:
            generation["tool_choice"] = GOOGLE_TOOL_CHOICE[config.tool_choice]
        # Reasoning branch: one thinking_level; unset leaves the depth to the model.
        level = config.effort or config.reasoning_effort
        if level:
            generation["thinking_level"] = level
        if generation:
            body["generation_config"] = generation
        return body

    def parse_reply(self, data: dict) -> ModelReply:
        return interactions_reply(data, data.get("model") or self.wire_id)


def interactions_reply(data: dict, model: Optional[str]) -> ModelReply:
    """One Interactions API reply as ours: its output text, its calls, its thought steps verbatim. An interaction
    that failed or was cancelled is an error with the provider's message, never an answer."""
    if data.get("status") in INTERACTION_FAILED_STATUSES:
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise ProviderError(f"{model}: the interaction {data['status']}: {message or 'the provider gave no reason'}",
                            body=data)
    text: list[str] = []
    summaries: list[str] = []
    calls: list[ToolCallRequest] = []
    thoughts: list[dict] = []
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        kind = step.get("type")
        if kind == "thought":
            thoughts.append(copy.deepcopy(step))
            summaries.extend(_interactions_text(step.get("summary")))
        elif kind == "model_output":
            text.extend(_interactions_text(step.get("content")))
        elif kind == "function_call":
            calls.append(ToolCallRequest(id=step.get("id"), name=step.get("name") or "",
                                         arguments=_arguments_of(step)))
    status = data.get("status")
    return ModelReply(
        content="".join(text) or None,
        thinking="".join(summaries) or None,
        tool_calls=calls,
        usage=usage_from_interactions(data.get("usage")),
        model=model,
        stop_reason=INTERACTION_STOP_REASONS.get(status, status),
        raw=data,
        thinking_blocks=thoughts or None,
    )


def _interactions_text(parts: Any) -> list[str]:
    """The text parts of a step's content or summary list."""
    return [str(part.get("text") or "") for part in parts or [] if isinstance(part, dict) and part.get("type") == "text"]


def _interactions_tool(tool: dict) -> dict:
    """Our OpenAI-shaped tool as the flat function tool the Interactions API takes."""
    function = _openai_tool(tool)["function"]
    return {"type": "function", "name": function.get("name", ""), "description": function.get("description") or "",
            "parameters": function.get("parameters") or {"type": "object"}}


def _to_interactions(messages: list[dict]) -> tuple[str, list[dict]]:
    """Our wire messages as the system instruction and the input steps.

    An assistant turn is its own thought steps first, unchanged, then its text, then one
    function_call per tool call; a tool message is a function_result naming the call it answers.
    A thought block another provider wrote is not a step here, so only type "thought" is echoed.
    """
    system: list[str] = []
    steps: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system.append(_text_of(message))
        elif role == "tool":
            step = {"type": "function_result", "call_id": message.get("tool_call_id") or message.get("id"),
                    "name": message.get("name") or "", "result": [{"type": "text", "text": _text_of(message)}]}
            if message.get("is_error"):
                step["is_error"] = True
            steps.append(step)
        elif role == "assistant":
            steps.extend(_interactions_assistant_steps(message))
        else:
            steps.append({"type": "user_input", "content": [{"type": "text", "text": _text_of(message)}]})
    return "\n\n".join(system), steps


def _interactions_assistant_steps(message: dict) -> list[dict]:
    """One assistant turn as steps: its thought steps unchanged, its text, then its function calls."""
    steps = [copy.deepcopy(block) for block in message.get(THINKING_BLOCKS_KEY) or []
             if isinstance(block, dict) and block.get("type") == "thought"]
    text = _text_of(message)
    if text:
        steps.append({"type": "model_output", "content": [{"type": "text", "text": text}]})
    for call in message.get("tool_calls") or []:
        steps.append({"type": "function_call", "id": call.get("id"),
                      "name": call.get("name") or (call.get("function") or {}).get("name") or "",
                      "arguments": _arguments_of(call)})
    return steps


ADAPTERS: dict[str, type] = {"anthropic": AnthropicModel, "openai": OpenAIModel, "bedrock": BedrockAnthropicModel,
                             "google": GoogleInteractionsModel}

# The registry snapshot unknown providers are resolved against. None is the real default
# (kullback.ai.pricing.snapshot_path()); tests point it at a tmp file, so no test reads the real one.
REGISTRY_SNAPSHOT_PATH: Optional[str] = None


def registry_endpoint(model_id: str, env: Optional[dict[str, str]] = None) -> Any:
    """What the models.dev snapshot says about this id's provider, or None when it says nothing.

    Imported inside the function because `pricing` reads the live switch from this module: the same
    deferred import `budget.py` makes, for the same reason. No network unless live calls are
    already on, and then the snapshot is refetched at most once a week.
    """
    from kullback.ai import pricing

    catalog = pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=env)
    return pricing.endpoint_from_catalog(catalog, model_id)


def sent_model_id(model_id: str) -> str:
    """The 'provider/model' id as its adapter sends it, so a call is priced under what went out.

    A Bedrock id with no profile goes out on `global.`, and the catalog prices the bare row and the
    `global.` row differently; a provider with no adapter of its own sends the id as given.
    """
    provider, wire = split_model_id(model_id)
    adapter = ADAPTERS.get(provider)
    if adapter is None or not wire:
        return model_id
    return f"{provider}/{adapter.for_model(model_id).sent_wire_id(wire)}"


# The host of a provider the registry names none for, used when its PROVIDER_BASE_URL is unset too.
# Empty since Google moved to an adapter of its own (GoogleInteractionsModel); the next such provider is one row.
KNOWN_HOSTS: dict[str, str] = {}


def base_url_var_for_provider(provider: str) -> str:
    """The variable a provider's host is read from when the registry names none: PROVIDER_BASE_URL,
    spelled by the same rule as the key variable."""
    return f"{provider.upper().replace('-', '_')}_BASE_URL"


def _model_from_base_url_variable(model_id: str, provider: str, **kwargs) -> Model:
    """A provider the snapshot names no host for, reached on the host its PROVIDER_BASE_URL names,
    else on its entry in KNOWN_HOSTS.

    The key is then required under PROVIDER_API_KEY: a host set by hand for a named provider is a
    hosted one, so a missing key is refused by name rather than sent unauthenticated.
    """
    values = os.environ if kwargs.get("env") is None else kwargs["env"]
    variable = base_url_var_for_provider(provider)
    base_url = values.get(variable) or KNOWN_HOSTS.get(provider)
    if not base_url:
        raise ValueError(
            f"{model_id} has no adapter of its own and the models.dev snapshot names no host for "
            f"{provider!r}; pass base_url, set {variable}, or refresh the snapshot with live calls on"
        )
    return RegistryModel(model_id, base_url=base_url, key_env_var=key_var_for_provider(provider), **kwargs)


def model_for(model_id: str, base_url: Optional[str] = None, **kwargs) -> Model:
    """The one place a live adapter is built, from the 'provider/model' id.

    With KULLBACK_ROUTE=omniroute every model goes through the OmniRoute gateway (OmniRouteModel). Otherwise
    three ways to reach a model, in order: an adapter of its own, the base URL the caller passed,
    or the host the models.dev registry lists for that provider. The last is what puts a provider
    nobody wrote code for (OpenCode Go, Groq, DeepSeek, OpenRouter) one `--model` away, from the
    same snapshot `budget.py` prices the call from. A provider the registry does not list, or lists
    behind a request shape this Harness does not build, still needs a base URL: passed, or for a
    provider the registry names no host for, read from PROVIDER_BASE_URL.
    """
    if routed(kwargs.get("env")):
        return OmniRouteModel(model_id, base_url=base_url, **kwargs)
    provider, _ = split_model_id(model_id)
    adapter = ADAPTERS.get(provider)
    if adapter is not None:
        return adapter.for_model(model_id)(model_id, base_url=base_url, **kwargs)
    if base_url:
        from kullback.ai import pricing

        catalog = pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=kwargs.get("env"))
        if pricing.speaks_responses(catalog, model_id):
            # An explicit endpoint never changes the wire shape: a Responses model speaks
            # Responses wherever it lives, so this check sits before the chat branch.
            # An unreadable catalogue reads as chat.
            return OpenAIResponsesModel(model_id, base_url=base_url, **kwargs)
        return OpenAICompatibleModel(model_id, base_url=base_url, **kwargs)
    endpoint = registry_endpoint(model_id, env=kwargs.get("env"))
    if endpoint is None:
        return _model_from_base_url_variable(model_id, provider, **kwargs)
    from kullback.ai import pricing

    catalog = pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=kwargs.get("env"))
    if pricing.speaks_responses(catalog, model_id):
        return OpenAIResponsesModel(model_id, base_url=endpoint.base_url,
                                     key_env_var=endpoint.key_env_var, **kwargs)
    per_model = pricing.model_adapter_for(catalog, model_id)
    shape = per_model or endpoint.adapter
    if shape not in pricing.OPENAI_SHAPED:
        raise ValueError(
            f"models.dev serves {model_id} through {shape}, which is not a request shape this "
            f"Harness builds; pass base_url for an endpoint that is"
        )
    return RegistryModel(model_id, base_url=endpoint.base_url, key_env_var=endpoint.key_env_var, **kwargs)


def provider_for(model_id: str, base_url: Optional[str] = None, **kwargs) -> Any:
    """The streaming provider for a model id: the same three ways in as `model_for`.

    The handle `model_for` built decides the shape, and the provider that reads that shape wraps it,
    so a model reached through the registry streams exactly as one with an adapter of its own does.
    The adapter modules are imported here rather than at the top because they import this one: the
    request shaping they stream is the handle's, written once.
    """
    from kullback.ai.anthropic import AnthropicProvider
    from kullback.ai.google_interactions import GoogleInteractionsProvider
    from kullback.ai.openai_compatible import OpenAICompatibleProvider

    handle = model_for(model_id, base_url, **kwargs)
    if isinstance(handle, AnthropicModel):
        return AnthropicProvider(handle)
    if isinstance(handle, GoogleInteractionsModel):
        return GoogleInteractionsProvider(handle)
    if isinstance(handle, HttpModel):
        return OpenAICompatibleProvider(handle)
    from kullback.ai.replay import ReplayProvider

    return ReplayProvider(handle)


def live_model(model_id: str, base_url: Optional[str] = None, **kwargs) -> Model:
    """One live adapter, after the environment has said live calls are allowed.

    Keys come from the environment or from a .env file in the current directory (read here, never
    overriding exported values). Both frontends go through this, so there is one refusal and one
    place the flag is ever set.
    """
    load_dotenv()
    if not enable_live_calls_from_env():
        raise RuntimeError(
            f"live model requests are off; put {LIVE_ENV_VAR}=1 in .env or export it")
    return model_for(model_id, base_url, **kwargs)


def live_provider(model_id: str, base_url: Optional[str] = None, **kwargs) -> Any:
    """One live streaming provider, after the environment has said live calls are allowed."""
    load_dotenv()
    if not enable_live_calls_from_env():
        raise RuntimeError(
            f"live model requests are off; put {LIVE_ENV_VAR}=1 in .env or export it")
    return provider_for(model_id, base_url, **kwargs)


def _anthropic_tool(tool: dict) -> dict:
    schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object"}
    return {"name": tool.get("name", ""), "description": tool.get("description") or "", "input_schema": schema}


def _openai_tool(tool: dict) -> dict:
    if "function" in tool:
        return tool
    schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object"}
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description") or "",
            "parameters": schema,
        },
    }


# The keys a chat message is allowed to carry on the wire. A transcript is the harness's own record
# as well as the model's input, and a stage that needs to write something beside a message writes it
# on the message: the Runner marks a refused tool result so the Simulated user can read it (D227).
# The wire shape is a whitelist so no such mark can reach a provider and be rejected there; a key
# the API learns is added here once, rather than every writer having to know what the wire holds.
#
# The reasoning keys are a table of the fields a provider's own thinking mode puts on an assistant
# message and asks to see again on the next request of a tool-call round. DeepSeek: "for requests
# carrying the tools parameter, the reasoning_content must be fully passed back to the API in all
# subsequent requests ... If your code does not correctly pass back reasoning_content, the API will
# return a 400 error" (api-docs.deepseek.com/guides/thinking_mode). OpenRouter says the same of
# `reasoning` and `reasoning_details` under "Preserving Reasoning"
# (openrouter.ai/docs/use-cases/reasoning-tokens). Both read 2026-09-10. Passing them through costs
# an endpoint that does not use them nothing, because only a reply that carried the field can put it
# on a message, and it is a table rather than a branch so the next provider is one row.
REASONING_ECHO_KEYS = frozenset(("reasoning_content", "reasoning", "reasoning_details"))
OPENAI_MESSAGE_KEYS = frozenset(
    ("role", "content", "name", "tool_calls", "tool_call_id", "refusal")) | REASONING_ECHO_KEYS


def _openai_message(message: dict) -> dict:
    """Clean ids, drop what the wire does not carry, and put our tool calls into the wire shape."""
    out = _clean_ids({key: value for key, value in message.items() if key in OPENAI_MESSAGE_KEYS})
    if isinstance(out.get("content"), dict):
        out["content"] = _text_of(out)  # JSON, not a Python repr; same rule as _text_of
    calls = out.get("tool_calls")
    if not calls:
        # the loop writes every assistant message with a tool_calls list; the API rejects an
        # empty one ("Invalid 'messages[4].tool_calls': empty array"), so a plain reply goes without
        out.pop("tool_calls", None)
    else:
        out["tool_calls"] = [
            call
            if "function" in call
            else {
                "id": clean_tool_call_id(call.get("id")),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments") or {}, sort_keys=True),
                },
            }
            for call in calls
        ]
    return out


def _text_of(message: dict) -> str:
    """One message's content as text.

    A structured tool result goes as JSON, not as Python's repr: repr writes single quotes
    and True and None, which is text production never showed the model (D65).
    """
    content = message.get("content")
    if isinstance(content, list):
        if all(isinstance(b, dict) and "type" in b for b in content) and content:
            return "".join(b.get("text", "") for b in content)
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content or "")


def _anthropic_blocks(message: dict) -> list[dict]:
    """One user or assistant message's content blocks: its thinking, its content, then its tool calls."""
    content = message.get("content")
    # The reply's own thinking blocks lead its turn, unchanged: the Messages API reads them
    # before the text and tool calls they preceded, and refuses an edited one.
    thinking = [copy.deepcopy(b) for b in message.get(THINKING_BLOCKS_KEY) or [] if isinstance(b, dict)]
    if isinstance(content, list) and all(isinstance(b, dict) and "type" in b for b in content):
        blocks = thinking + list(content)
    elif content:
        blocks = thinking + [{"type": "text", "text": _text_of(message)}]
    else:
        blocks = thinking
    for call in message.get("tool_calls") or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": call.get("name") or (call.get("function") or {}).get("name") or "",
                "input": _arguments_of(call),
            }
        )
    return blocks


def _to_anthropic(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Our canonical messages into Anthropic's: system pulled out, tool calls and results as blocks.

    Every tool_result answering one assistant turn has to land in a single following user
    message, one block each: two tool calls in a turn followed by two tool messages must become
    one user message with two tool_result blocks, not two user messages back to back, which the
    Anthropic Messages API rejects.
    """
    system: list[dict] = []
    out: list[dict] = []
    in_tool_group = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            system.append({"type": "text", "text": _text_of(message)})
            in_tool_group = False
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or message.get("id"),
                "content": _text_of(message),
            }
            if in_tool_group:
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
                in_tool_group = True
            continue
        in_tool_group = False
        blocks = _anthropic_blocks(message)
        out.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    return system, out
