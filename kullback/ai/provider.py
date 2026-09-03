"""One interface for model calls, with the three offline models tests are allowed to use."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.usage import Usage

# Tests never call a real model. Only a real adapter checks this flag; TestModel, RecordedModel
# and MemoModel ignore it: the first two never leave the machine and the third only forwards to
# the model it wraps.
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
    # OpenAI: prompt_cache_key, so the provider routes calls that share a prefix to the same
    # cache. Set once per build and stage (build.py); the Anthropic adapter ignores it, since it
    # caches by cache_control points instead (cache_system, cache_last_two below).
    prompt_cache_key: Optional[str] = None


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
        payload = {
            "model": self.name,
            "messages": normalize_messages(messages),
            "tools": tools or [],
            "config": (config or ModelConfig()).model_dump(mode="json"),
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
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    content, calls = _content_and_calls(data)
    return ModelReply(
        content=content,
        tool_calls=[
            ToolCallRequest(
                id=c.get("id"),
                name=c.get("name") or (c.get("function") or {}).get("name", ""),
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
        self._client_lock = threading.Lock()

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
            body[self.token_cap_field()] = config.max_tokens
        if config.temperature is not None and not self._reasoning_family():
            body["temperature"] = config.temperature
        if config.seed is not None:
            body["seed"] = config.seed
        if config.stop:
            body["stop"] = list(config.stop)
        if config.prompt_cache_key:
            body["prompt_cache_key"] = config.prompt_cache_key
        body.update(self.reasoning_fields(config))
        if tools and self._reasoning_family() and "reasoning_effort" not in body:
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
        test is on the wire id rather than a list of model names, because a list would be stale the
        week after it was written and the failure mode is a whole build dying on its first call.
        """
        wire = (self.wire_id or "").lower()
        return wire.startswith(("gpt-5", "o1", "o3", "o4"))

    def token_cap_field(self) -> str:
        return "max_completion_tokens" if self._reasoning_family() else "max_tokens"

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


class OpenAICompatibleModel(OpenAIModel):
    """A local or self-hosted endpoint that speaks the OpenAI shape. Base URL required, key optional."""

    key_required = False

    def __init__(self, model_id: str, base_url: str, **kwargs):
        super().__init__(model_id, base_url=base_url, **kwargs)

    def headers(self) -> dict:
        return opencode_headers(self.base_url, super().headers())

    def reasoning_fields(self, config: ModelConfig) -> dict:
        """Reasoning branch three of three: a local endpoint gets none of it. Servers that do
        not know the field reject the whole request, and there is no effort table to guess from."""
        return {}


class RegistryModel(OpenAICompatibleModel):
    """A provider the models.dev registry names: its host and its key variable, the OpenAI shape.

    This is how `opencode-go/kimi-k3` or `groq/llama-3.3-70b` runs without anyone writing an
    adapter for it. The registry answers where to send the call and which variable holds the key;
    the body is OpenAI's, because the registry is only asked for providers that speak that shape.
    Reasoning fields stay off for the reason the local endpoint leaves them off: a gateway that
    does not know a field refuses the whole request, and the registry lists no effort table.
    """

    def __init__(self, model_id: str, base_url: str, key_env_var: str = "", **kwargs):
        self.key_env_var = key_env_var
        self.key_required = bool(key_env_var)
        super().__init__(model_id, base_url=base_url, **kwargs)


# Models OpenCode serves through the Responses API (/v1/responses) rather than chat completions,
# from its Go docs' Endpoints table. The models.dev snapshot carries no per-model shape field
# (and does not list 1.3 at all yet), so the docs are the source of truth here. Delete an entry
# when the snapshot carries that model with a shape the resolver can read; never add one the
# docs' table does not name. gpt-5.6-luna is deliberately absent: it answers chat bodies live.
RESPONSES_API_MODELS = frozenset({"opencode-go/muse-spark-1.3-contributor"})


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

    def headers(self) -> dict:
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
        if config.max_tokens is not None:
            body["max_output_tokens"] = config.max_tokens
        return body

    def parse_reply(self, data: dict) -> ModelReply:
        if data.get("status") not in (None, "completed", "in_progress"):
            error = data.get("error") or {}
            raise ProviderError(
                f"{self.name}: the Responses API ended as {data.get('status')}: "
                f"{error.get('message') or error or 'no reason given'}"
            )
        texts: list[str] = []
        calls: list[ToolCallRequest] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        texts.append(part.get("text") or "")
            elif kind == "function_call":
                calls.append(
                    ToolCallRequest(
                        id=clean_tool_call_id(item.get("call_id") or item.get("id")),
                        name=item.get("name") or "",
                        arguments=_arguments_of({"arguments": item.get("arguments")}),
                    )
                )
        usage = data.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        # Same convention as the chat adapter: Usage.input means uncached input everywhere,
        # so the cached tokens come off here, at the adapter, and budget.py bills plain counts.
        cached = int(details.get("cached_tokens", 0) or 0)
        return ModelReply(
            content="".join(texts) or None,
            tool_calls=calls,
            usage=Usage(
                input=max(0, int(usage.get("input_tokens", 0) or 0) - cached),
                output=int(usage.get("output_tokens", 0) or 0),
                cache_read=cached,
            ),
            model=data.get("model") or self.wire_id,
            stop_reason=data.get("status"),
            raw=data,
        )


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
                          "arguments": args if isinstance(args, str) else json.dumps(args or {})})
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


ADAPTERS: dict[str, type] = {"anthropic": AnthropicModel, "openai": OpenAIModel}

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


def model_for(model_id: str, base_url: Optional[str] = None, **kwargs) -> Model:
    """The one place a live adapter is built, from the 'provider/model' id.

    Three ways to reach a model, in order: an adapter of its own, the base URL the caller passed,
    or the host the models.dev registry lists for that provider. The last is what puts a provider
    nobody wrote code for (OpenCode Go, Groq, DeepSeek, OpenRouter) one `--model` away, from the
    same snapshot `budget.py` prices the call from. A provider the registry does not list, or lists
    behind a request shape this Harness does not build, still needs a base URL.
    """
    provider, _ = split_model_id(model_id)
    adapter = ADAPTERS.get(provider)
    if adapter is not None:
        return adapter(model_id, base_url=base_url, **kwargs)
    if base_url:
        return OpenAICompatibleModel(model_id, base_url=base_url, **kwargs)
    endpoint = registry_endpoint(model_id, env=kwargs.get("env"))
    if endpoint is None:
        raise ValueError(
            f"{model_id} has no adapter of its own and the models.dev snapshot names no host for "
            f"{provider!r}; pass base_url, or refresh the snapshot with live calls on"
        )
    if model_id in RESPONSES_API_MODELS:
        return OpenAIResponsesModel(model_id, base_url=endpoint.base_url,
                                     key_env_var=endpoint.key_env_var, **kwargs)
    from kullback.ai import pricing

    catalog = pricing.refresh(path=REGISTRY_SNAPSHOT_PATH, env=kwargs.get("env"))
    per_model = pricing.model_adapter_for(catalog, model_id)
    shape = per_model or endpoint.adapter
    if shape not in pricing.OPENAI_SHAPED:
        raise ValueError(
            f"models.dev serves {model_id} through {shape}, which is not a request shape this "
            f"Harness builds; pass base_url for an endpoint that is"
        )
    return RegistryModel(model_id, base_url=endpoint.base_url, key_env_var=endpoint.key_env_var, **kwargs)


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
