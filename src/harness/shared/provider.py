"""One interface for model calls, with the two offline models tests are allowed to use."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from harness.shared.records import Usage

# Tests never call a real model. Only a real adapter checks this flag; TestModel and
# RecordedModel ignore it because neither leaves the machine.
ALLOW_MODEL_REQUESTS = False

# The one way to turn live calls on: a person exports this before running the CLI. There is
# no flag a module can set by accident, and the default above stays False.
LIVE_ENV_VAR = "HARNESS_ALLOW_MODEL_REQUESTS"


class ToolCallRequest(BaseModel):
    """A tool call the model asked for."""

    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    name: str
    arguments: dict = Field(default_factory=dict)


class ModelReply(BaseModel):
    """What one model call returned."""

    model_config = ConfigDict(populate_by_name=True)

    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: Optional[str] = None
    stop_reason: Optional[str] = None
    raw: Optional[dict] = None


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
            "live model requests are off (harness.shared.provider.ALLOW_MODEL_REQUESTS is False); "
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


def enable_live_calls_from_env(env: Optional[dict[str, str]] = None) -> bool:
    """Turn live calls on when a person asked for it in the environment. Returns the new state.

    Called once by the CLI's run command. Nothing else in the package sets the flag, so a
    module cannot reach the network by importing its way there.
    """
    global ALLOW_MODEL_REQUESTS
    values = os.environ if env is None else env
    ALLOW_MODEL_REQUESTS = str(values.get(LIVE_ENV_VAR, "")).strip().lower() in ("1", "true", "yes", "on")
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
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    content, calls = _content_and_calls(data)
    return ModelReply(
        content=content,
        tool_calls=[
            ToolCallRequest(
                id=c.get("id"),
                name=c.get("name") or c.get("function", {}).get("name", ""),
                arguments=_arguments_of(c),
            )
            for c in calls
        ],
        usage=Usage(
            input=int(usage.get("input", usage.get("prompt_tokens", 0)) or 0),
            output=int(usage.get("output", usage.get("completion_tokens", 0)) or 0),
            cache_read=int(usage.get("cache_read", 0) or 0),
            cache_write=int(usage.get("cache_write", 0) or 0),
        ),
        model=data.get("model"),
        stop_reason=data.get("stop_reason") or data.get("finish_reason"),
    )


def _arguments_of(call: dict) -> dict:
    args = call.get("arguments")
    if args is None:
        args = call.get("function", {}).get("arguments")
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


class ProviderError(RuntimeError):
    """A provider said no. Carries the status and the body so the caller can see what it said."""

    def __init__(self, message: str, status: Optional[int] = None, body: Any = None):
        self.status = status
        self.body = body
        super().__init__(message)


class ContextOverflowError(ProviderError):
    """The prompt did not fit. Never retried: the same prompt will not fit next time either."""


class RetryExhausted(ProviderError):
    """Every attempt failed on a retryable error."""


class RetryPolicy(BaseModel):
    """Five attempts, exponential backoff with jitter, nothing clever."""

    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: float = 0.25
    # A provider may ask for a wait longer than the build is willing to sit still for; past
    # this the call gives up rather than blocking the build for hours.
    max_retry_after_s: float = 120.0


CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "context_length",
    "prompt is too long",
    "too long for",
    "maximum context",
    "reduce the length",
)
ID_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")
ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
CACHE_CONTROL = {"type": "ephemeral"}
TEXT_BLOCK_TYPES = ("text", "reasoning", "thinking")
ID_DIGEST_LEN = 8
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


def clean_tool_call_id(call_id: Any) -> str:
    """Tool call ids are restricted to [a-zA-Z0-9_-]; other characters become underscores.

    Replacing characters can collide: 'a:b' and 'a_b' both cleaned to 'a_b', and a tool_result
    could then be paired with the wrong tool_use in the same message. When anything was
    replaced, a short digest of the original id is appended so distinct ids stay distinct.
    """
    original = str(call_id or "")
    cleaned = ID_ALLOWED.sub("_", original)
    if not cleaned:
        return "id"
    if cleaned == original:
        return cleaned
    return f"{cleaned}_{hashlib.sha256(original.encode('utf-8')).hexdigest()[:ID_DIGEST_LEN]}"


def _is_empty_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return not block
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
            message["content"] = [_clean_ids(b) if isinstance(b, dict) else b for b in content if not _is_empty_block(b)]
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


def cache_system(system: Any) -> list[dict]:
    """A cache point on the system prompt: it is the same on every call of a build."""
    blocks = [{"type": "text", "text": system}] if isinstance(system, str) else copy.deepcopy(system or [])
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}
    return blocks


def cache_last_two(messages: list[dict]) -> list[dict]:
    """Cache points on the last two non-system messages, so a growing conversation reuses its prefix."""
    out = copy.deepcopy(messages)
    marked = 0
    for message in reversed(out):
        if marked >= 2 or message.get("role") == "system":
            continue
        blocks = message.get("content")
        if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict):
            blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}
            marked += 1
    return out


def json_body(request: Any) -> dict:
    """The JSON a request carries; used by the adapters' tests and by error reporting."""
    try:
        return json.loads(request.content or b"{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def retry_after_seconds(headers: Any, now: Optional[float] = None) -> Optional[float]:
    """Retry-After as seconds, whether the provider sent a count or an HTTP date."""
    raw = None
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            raw = headers[key]
            break
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    stamp = parsedate_to_datetime(str(raw))
    if stamp is None:
        return None
    reference = time.time() if now is None else now
    return max(0.0, stamp.timestamp() - reference)


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Exponential backoff with jitter, capped."""
    delay = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
    return delay + rng.random() * policy.jitter * delay


def _retryable_status(status: int) -> bool:
    """Only rate limits and server faults. A 400 is our bug and retrying it wastes money."""
    return status == 429 or status >= 500


def _error_text(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        return str(body.get("message") or body)
    return str(body)


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
        timeout: float = 60.0,
    ):
        self.name = model_id
        self.provider, derived = split_model_id(model_id)
        self.wire_id = wire_id or derived
        self.env = dict(os.environ) if env is None else dict(env)
        self.base_url = substitute_env(base_url or self.default_base_url, self.env).rstrip("/")
        self.api_key = api_key or self.env.get(self.key_env_var)
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self.sleep = sleep or time.sleep
        self.rng = rng or random.Random()
        self._client = client

    def client(self) -> Any:
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
        if self.key_required and not self.api_key:
            raise ProviderError(f"no API key for {self.name}; set {self.key_env_var} or pass api_key")
        body = self.build_body(messages, tools, config or ModelConfig())
        data = self.post(body)
        return self.parse_reply(data)

    def post(self, body: dict) -> dict:
        # The gate sits here, on the network path itself, not only on query(): a caller that
        # builds a body and posts it must not reach the transport while live calls are off.
        require_live_calls_enabled()
        url = self.base_url + self.path
        headers = self.headers()
        for attempt in range(1, self.retry.attempts + 1):
            last_attempt = attempt == self.retry.attempts
            try:
                response = self.client().post(url, headers=headers, json=body, timeout=self.timeout)
            except httpx.HTTPError as exc:
                if last_attempt:
                    raise RetryExhausted(f"{self.name}: {self.retry.attempts} attempts failed: {exc}") from exc
                self.sleep(backoff_delay(attempt, self.retry, self.rng))
                continue
            if response.status_code < 400:
                try:
                    return response.json()
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
                        ) from exc
                    self.sleep(backoff_delay(attempt, self.retry, self.rng))
                    continue
            error = self.error_for(response)
            if isinstance(error, ContextOverflowError) or not _retryable_status(response.status_code):
                raise error
            if last_attempt:
                raise RetryExhausted(
                    f"{self.name}: {self.retry.attempts} attempts failed: {error}",
                    status=response.status_code,
                    body=error.body,
                ) from error
            wait = retry_after_seconds(response.headers)
            if wait is not None and wait > self.retry.max_retry_after_s:
                raise RetryExhausted(
                    f"{self.name}: provider asked for {wait:.0f}s, over the "
                    f"{self.retry.max_retry_after_s:.0f}s this build will wait: {error}",
                    status=response.status_code,
                    body=error.body,
                ) from error
            self.sleep(backoff_delay(attempt, self.retry, self.rng) if wait is None else wait)
        raise RetryExhausted(f"{self.name}: no attempts were made")

    def error_for(self, response: Any) -> ProviderError:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        message = _error_text(body).lower()
        text = f"{self.name}: HTTP {response.status_code}: {_error_text(body)}"
        if any(marker in message for marker in CONTEXT_OVERFLOW_MARKERS):
            return ContextOverflowError(text, status=response.status_code, body=body)
        return ProviderError(text, status=response.status_code, body=body)

    def headers(self) -> dict:
        raise NotImplementedError

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        raise NotImplementedError

    def parse_reply(self, data: dict) -> ModelReply:
        raise NotImplementedError


class AnthropicModel(HttpModel):
    """Anthropic Messages API: system split out, cache points, tool_use blocks."""

    key_env_var = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com"
    path = "/v1/messages"
    api_version = "2023-06-01"
    default_max_tokens = 4096

    def headers(self) -> dict:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def build_body(self, messages: list[dict], tools: Optional[list[dict]], config: ModelConfig) -> dict:
        system, converted = _to_anthropic(messages)
        body: dict[str, Any] = {
            "model": self.wire_id,
            "max_tokens": config.max_tokens or self.default_max_tokens,
            "messages": cache_last_two(normalize_messages(converted)),
        }
        if system:
            body["system"] = cache_system(system)
        if tools:
            body["tools"] = [_anthropic_tool(t) for t in tools]
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.stop:
            body["stop_sequences"] = list(config.stop)
        # Reasoning branch one of three: Anthropic takes thinking as its own block and the
        # depth as output_config.effort. budget_tokens is not sent: the current models reject it.
        if config.thinking:
            body["thinking"] = dict(config.thinking)
        if config.effort:
            body["output_config"] = {"effort": config.effort}
        return body

    def parse_reply(self, data: dict) -> ModelReply:
        text: list[str] = []
        calls: list[ToolCallRequest] = []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
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
        usage = data.get("usage") or {}
        return ModelReply(
            content="".join(text) or None,
            tool_calls=calls,
            usage=Usage(
                input=int(usage.get("input_tokens", 0) or 0),
                output=int(usage.get("output_tokens", 0) or 0),
                cache_read=int(usage.get("cache_read_input_tokens", 0) or 0),
                cache_write=int(usage.get("cache_creation_input_tokens", 0) or 0),
            ),
            model=data.get("model") or self.wire_id,
            stop_reason=data.get("stop_reason"),
            raw=data,
        )


class OpenAIModel(HttpModel):
    """OpenAI chat completions: messages as given, tool calls with JSON string arguments."""

    key_env_var = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"
    path = "/chat/completions"

    def headers(self) -> dict:
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
        if config.max_tokens is not None:
            body["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.seed is not None:
            body["seed"] = config.seed
        if config.stop:
            body["stop"] = list(config.stop)
        body.update(self.reasoning_fields(config))
        return body

    def reasoning_fields(self, config: ModelConfig) -> dict:
        """Reasoning branch two of three: OpenAI takes one reasoning_effort field."""
        return {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}

    def parse_reply(self, data: dict) -> ModelReply:
        choices = data.get("choices") or [{}]
        message = (choices[0] or {}).get("message") or {}
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        # Usage.input means uncached input everywhere in the Harness, which is what Anthropic
        # already reports. OpenAI's prompt_tokens includes the cached ones, so subtract here,
        # at the adapter, and let budget.py bill each count at its own rate with no arithmetic.
        cached = int(details.get("cached_tokens", 0) or 0)
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
            usage=Usage(
                input=max(0, int(usage.get("prompt_tokens", 0) or 0) - cached),
                output=int(usage.get("completion_tokens", 0) or 0),
                cache_read=cached,
            ),
            model=data.get("model") or self.wire_id,
            stop_reason=(choices[0] or {}).get("finish_reason"),
            raw=data,
        )


class OpenAICompatibleModel(OpenAIModel):
    """A local or self-hosted endpoint that speaks the OpenAI shape. Base URL required, key optional."""

    key_required = False

    def __init__(self, model_id: str, base_url: str, **kwargs):
        super().__init__(model_id, base_url=base_url, **kwargs)

    def reasoning_fields(self, config: ModelConfig) -> dict:
        """Reasoning branch three of three: a local endpoint gets none of it. Servers that do
        not know the field reject the whole request, and there is no effort table to guess from."""
        return {}


ADAPTERS: dict[str, type] = {"anthropic": AnthropicModel, "openai": OpenAIModel}


def model_for(model_id: str, base_url: Optional[str] = None, **kwargs) -> Model:
    """The one place a live adapter is built, from the 'provider/model' id.

    A provider with no adapter of its own is treated as an OpenAI-compatible endpoint, which
    needs a base URL: a local model has no default host to guess.
    """
    provider, _ = split_model_id(model_id)
    adapter = ADAPTERS.get(provider)
    if adapter is not None:
        return adapter(model_id, base_url=base_url, **kwargs)
    if not base_url:
        raise ValueError(f"{model_id} is not anthropic or openai, so it needs a base_url")
    return OpenAICompatibleModel(model_id, base_url=base_url, **kwargs)


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


def _openai_message(message: dict) -> dict:
    """Clean ids, and put our canonical tool calls into the wire shape."""
    out = _clean_ids(dict(message))
    if isinstance(out.get("content"), dict):
        out["content"] = _text_of(out)  # JSON, not a Python repr; same rule as _text_of
    calls = out.get("tool_calls")
    if calls:
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


def _to_anthropic(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Our canonical messages into Anthropic's: system pulled out, tool calls and results as blocks."""
    system: list[dict] = []
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system.append({"type": "text", "text": _text_of(message)})
            continue
        if role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or message.get("id"),
                            "content": _text_of(message),
                        }
                    ],
                }
            )
            continue
        content = message.get("content")
        if isinstance(content, list) and all(isinstance(b, dict) and "type" in b for b in content):
            blocks = list(content)
        elif content:
            blocks = [{"type": "text", "text": _text_of(message)}]
        else:
            blocks = []
        for call in message.get("tool_calls") or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id"),
                    "name": call.get("name") or (call.get("function") or {}).get("name") or "",
                    "input": _arguments_of(call),
                }
            )
        out.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    return system, out
