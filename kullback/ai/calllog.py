"""The call log: one JSONL row per live model call, written by the provider layer for any code that calls a model.

Every live call passes HttpModel.query (the agent loop's ModelStream reaches it too), which hands it to `emit`, so
a caller cannot forget to log a call and needs to import nothing above kullback/ai. This module holds:

- the sink, one per process: `begin(folder, stage)` installs a `Log` writing <folder>/calls.jsonl; with no sink
  installed and KULLBACK_CALLS_DIR set, the first call installs one there; with neither, nothing is written;
- the context a row carries (run id, stage, label, attempt, group, route_switch): a process-wide default a
  command sets once (`begin`, `stage`), under a per-thread override (`context`, `labelled`) a worker sets around
  its own calls. A pool thread that copies no context reads the process default;
- the HTTP status of every failed attempt of a call (`attempts`, `retried`), so a 429 a retry hid still shows;
- the row (`Log.row`): the time, the context, the model asked and the model and provider the gateway reported,
  the route, the tokens (input is the uncached part, cache_read the cached part, output includes thinking,
  reasoning is the thinking share, null where the provider does not report it), the dollars (`dollars`: the
  gateway's list price, the upstream's billed cost, the actual cost, 0 on a subscription route, and where the
  price came from), the HTTP attempts and failed statuses, the wall time and the gateway's own latency, the
  error, the gateway's correlation id (its call log's key) and the session tag sent, the temperature and effort
  sent, and the characters sent. No message text and no key is written;
- the cache counters. A call that should hit the cache and does not is a bug. A provider in CACHE_TAIL caches
  implicitly at content-step boundaries (measured on Gemini, directly and through the gateway: a request whose
  leading steps an earlier finished request sent reads them from cache, all but the last TAIL tokens, and nothing
  under CACHE_MIN). So a call's expected cached tokens are min(shared step prefix, total input - TAIL), or 0
  under CACHE_MIN, where the shared step prefix is the longest run of whole leading steps (tools and system
  first, then the messages, consecutive same-side messages merged as the provider merges them) that a call
  finished before this one was sent, within TTL_S. A cache_read under MISS_SHARE of a nonzero expectation is a
  miss: a warning line and a counter. `uncacheable_shared` counts tokens shared with an earlier call past the
  last shared boundary, inside one step: shared, but where no cache can use them. Shares are scaled from
  characters.

A sink error never fails the call it logs. The rollup of a log is callrollup.py.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from kullback.ai.callrollup import CALLS_FILE, PIN, TOKENS, read, rollup, rollup_lines  # noqa: F401 (one import)

CALLS_DIR_VAR = "KULLBACK_CALLS_DIR"
RUN_ID_VAR = "KULLBACK_RUN_ID"
DEFAULT_LABEL = "call"
DEFAULT_STAGE = "call"
# Subscription routes: a call there costs nothing per token, so its usd is 0 and its list price is kept apart.
FREE_ROUTES = ("cc", "cx", "cu")
# Prices come from the gateway (its X-OmniRoute-Response-Cost header, from its own cache-aware price table, and
# the upstream's billed usage.cost where one is returned). Its table has no row for these routes (it reports 0),
# so a paid call there is priced here, per 1M tokens, and flagged price_source=local (checked 2026-11-05).
LOCAL_PRICED_ROUTES = ("bedrock",)
LOCAL_PRICES: dict[str, dict[str, float]] = {
    "bedrock/us.anthropic.claude-sonnet-5-5": {"input": 2.2, "output": 11.0, "cache_read": 0.11, "cache_write": 2.75},
    "bedrock/us.anthropic.claude-opus-5-5": {"input": 4.4, "output": 22.0, "cache_read": 0.22, "cache_write": 5.5},
    "bedrock/us.openai.gpt-6.1-sol": {"input": 2.2, "output": 11.0, "cache_read": 0.11, "cache_write": 2.75},
    "bedrock/us.xai.grok-4.7": {"input": 2.2, "output": 6.6, "cache_read": 0.55, "cache_write": 0.0},
}
# Providers with implicit step-prefix caching: the tokens a hit leaves uncached, and the fewest tokens it caches
# (measured 2026-11-05: a 2049-token tail on every hit; under 1024 expected, 9 of 103 calls read anything).
CACHE_TAIL = {"google": 2049}
CACHE_MIN = {"google": 1024}
TTL_S = 600.0  # how long a finished request's step prefixes count as cached
MISS_SHARE = 0.9  # a call whose cache_read is under this share of what it should have read is a miss
WARN_FIRST, WARN_EVERY = 10, 100  # miss warnings: the first ones, then one in every WARN_EVERY

_SINK: Optional[Callable[[dict], None]] = None
_DEFAULT: dict[str, Any] = {}
_LOCK = threading.Lock()
_OVERRIDE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("calllog_context", default=None)
_RETRIED: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("calllog_retried", default=None)


# --- the sink and the context ---


def install(sink: Optional[Callable[[dict], None]]) -> None:
    """The process's one sink (None removes it)."""
    global _SINK
    _SINK = sink


def installed() -> Optional[Callable[[dict], None]]:
    """The sink installed now, or None."""
    return _SINK


def set_default(**fields: Any) -> None:
    """The context every call in this process carries unless a thread overrides it."""
    with _LOCK:
        _DEFAULT.clear()
        _DEFAULT.update(fields)


def current() -> dict[str, Any]:
    """The context a call sent now carries: the process default under this thread's overrides."""
    with _LOCK:
        out = dict(_DEFAULT)
    out.update(_OVERRIDE.get() or {})
    return out


@contextmanager
def context(**fields: Any) -> Iterator[None]:
    """Calls sent inside the block carry these fields over the default (stage, label, attempt, group, ...)."""
    token = _OVERRIDE.set({**(_OVERRIDE.get() or {}), **fields})
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


@contextmanager
def labelled(label: str, attempt: int = 1, **fields: Any) -> Iterator[None]:
    """Every call this thread sends inside the block is labelled `label`, attempt `attempt` (and carries `fields`)."""
    with context(label=label, attempt=attempt, **fields):
        yield


@contextmanager
def attempts() -> Iterator[list]:
    """The statuses `retried` notes inside the block (one call's HTTP attempts), in order."""
    token = _RETRIED.set([])
    try:
        yield _RETRIED.get()
    finally:
        _RETRIED.reset(token)


def retried(status: Any) -> None:
    """One attempt of the call in flight failed with `status` (an HTTP code, or the transport error's name)."""
    seen = _RETRIED.get()
    if seen is not None:
        seen.append(status)


def emit(model: str, provider: str, system: str, messages: list[dict], tools: Any, usage: Any = None,
         attempts: int = 1, wall_ms: float = 0.0, error: Optional[str] = None, meta: Optional[dict] = None) -> None:
    """One call handed to the sink with its context; nothing without a sink, and a sink error is swallowed."""
    sink = _SINK or _from_env()
    if sink is None:
        return
    record = {"context": current(), "model": model, "provider": provider, "system": system, "messages": messages,
              "tools": tools, "usage": usage, "attempts": attempts, "wall_ms": wall_ms, "error": error,
              "meta": meta or {}}
    try:
        sink(record)
    except Exception:  # noqa: BLE001 - the log never fails the call it logs
        pass


def _from_env() -> Optional[Callable[[dict], None]]:
    """With no sink and KULLBACK_CALLS_DIR set, the log there, installed at the first call."""
    folder = os.environ.get(CALLS_DIR_VAR)
    if not folder:
        return None
    with _LOCK:
        stage_now = _DEFAULT.get("stage")
    return begin(folder, stage_now or DEFAULT_STAGE, keep_context=True)


def begin(folder: Path | str, stage: str, run_id: Optional[str] = None, keep_context: bool = False,
          **kwargs: Any) -> "Log":
    """Install the call log of one command: every live call from here on is a row in <folder>/calls.jsonl. The run
    id is `run_id`, else KULLBACK_RUN_ID, else one made here."""
    log = Log(folder, **kwargs)
    before = current() if keep_context else {}
    run_id = run_id or before.get("run_id") or os.environ.get(RUN_ID_VAR) or f"{stage}-{uuid.uuid4().hex[:8]}"
    set_default(**dict(before, stage=stage, run_id=run_id))
    install(log)
    return log


@contextmanager
def stage(name: str, folder: Optional[Path | str] = None) -> Iterator[None]:
    """Calls sent inside the block, on any thread, carry stage `name`; with no log installed and a folder given,
    one is installed for the block. Stage tools run one at a time (a workdir lock), so the process default holds."""
    before = current()
    fresh = installed() is None and folder is not None
    if fresh:
        begin(folder, name, before.get("run_id"))
    set_default(**dict(before, stage=name))
    try:
        yield
    finally:
        set_default(**before)
        if fresh:
            install(None)


def now() -> str:
    """The time a row carries: UTC, ISO 8601 to the millisecond, so rows compare as strings."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- dollars ---


def route_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else ""


def _local_usd(usage: Any, model_id: str) -> Optional[float]:
    row = LOCAL_PRICES.get(model_id)
    if row is None or usage is None:
        return None
    return (usage.input * row["input"] + usage.output * row["output"] + usage.cache_read * row["cache_read"]
            + usage.cache_write * row["cache_write"]) / 1_000_000


def dollars(usage: Any, model_id: str, route: str, gateway: Optional[float], billed: Optional[float],
            priced: Optional[float] = None) -> dict:
    """usd_gateway: the gateway's cache-aware list price of the call. Its cost header when above 0 (it reads 0 on a
    stream and on a model with no pricing row, which is no price, not free), else `priced` (the usage times the
    gateway's own pricing table), else our LOCAL_PRICES on a local-priced route, flagged. usd_billed: the cost the
    upstream billed, when it says. usd_actual: 0 on a subscription route, else billed, else usd_gateway."""
    source = "gateway" if gateway else None
    if not gateway and priced is not None:
        gateway, source = priced, "pricing"
    if not gateway and route in LOCAL_PRICED_ROUTES:
        local = _local_usd(usage, model_id)
        if local is not None:
            gateway, source = local, "local"
    actual = 0.0 if route in FREE_ROUTES else (billed if billed is not None else gateway or 0.0)
    if billed is not None and route not in FREE_ROUTES:
        source = "billed"
    return {"usd_gateway": _round(gateway), "usd_billed": _round(billed), "usd_actual": _round(actual),
            "price_source": "free" if route in FREE_ROUTES else source}


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 8)


# --- the log ---


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str) if value else ""


def steps(system: str, messages: Iterable[dict], tools: Any) -> list[str]:
    """The request as the provider's content steps: tools and system text first, then the messages, consecutive
    messages of one side (a tool result is the user's side) merged into one step."""
    out = [_text(tools) + "\n" + (system or "")]
    side = None
    for message in messages or []:
        role = "user" if message.get("role") in ("user", "tool") else str(message.get("role"))
        text = _text(message.get("content")) + _text(message.get("tool_calls"))
        if role == side:
            out[-1] += "\n" + text
        else:
            out.append(role + ":" + text)
            side = role
    return out


def _hashes(parts: list[str]) -> list[str]:
    out, h = [], hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "surrogatepass") + b"\x00")
        out.append(h.copy().hexdigest())
    return out


def _common(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class _Prefixes:
    """The step prefixes finished calls sent, with the first and last time each finished, to say what a new call
    should read from cache."""

    def __init__(self, ttl_s: float = TTL_S):
        self.ttl_s, self.done, self.next_step = ttl_s, {}, {}

    def shared(self, parts: list[str], sent_at: float) -> tuple[int, int]:
        """(characters of the leading whole steps a call finished before `sent_at` sent, characters this call
        shares with an earlier one inside the next step: shared, but past the last boundary)."""
        hashes = _hashes(parts)
        whole = 0
        for k, h in enumerate(hashes):
            first, last = self.done.get(h, (None, None))
            if first is None or first > sent_at or sent_at - last > self.ttl_s:
                break
            whole = k + 1
        chars = sum(len(p) for p in parts[:whole])
        inside = 0
        if whole < len(parts):
            seen = self.next_step.get(hashes[whole - 1] if whole else "")
            inside = _common(seen, parts[whole]) if seen else 0
        return chars, inside

    def add(self, parts: list[str], finished_at: float) -> None:
        hashes = _hashes(parts)
        for k, h in enumerate(hashes):
            # The first finish says from when the prefix was in the cache, the last keeps it fresh: keeping only
            # the latest let a busy shared step (the system prompt) look unsent to every call already in flight.
            first, _last = self.done.get(h, (finished_at, finished_at))
            self.done[h] = (min(first, finished_at), finished_at)
            if k + 1 < len(parts):
                self.next_step[h] = parts[k + 1]
        self.next_step[""] = parts[0]


def _identity(ctx: dict, record: dict, model: str, route: str) -> dict:
    """Which call a row is: time, run, stage, label, attempt, group, model, provider and route."""
    return {"time": now(), "run_id": ctx.get("run_id"), "stage": ctx.get("stage") or "",
            "label": ctx.get("label") or DEFAULT_LABEL, "attempt": int(ctx.get("attempt") or 1),
            "group": ctx.get("group"), "model": model, "provider": record.get("provider"), "route": route}


def _failure(ctx: dict, record: dict) -> dict:
    """The call's error, and the later route of the same model it went on after a switch, when either happened."""
    out = {"error": record.get("error"), "route_switch": ctx.get("route_switch")}
    return {k: v for k, v in out.items() if v}


class Log:
    """The sink of one command: rows appended to <folder>/calls.jsonl under a lock, and kept in `rows`. `cost`, when
    given, stands in for the gateway's estimate (a test's fixed price, or a route outside the gateway)."""

    def __init__(self, folder: Path | str, cost: Optional[Callable[[Any, str], float]] = None,
                 clock: Callable[[], float] = time.time, warn: Optional[Callable[[str], None]] = None):
        self.path = Path(folder) / CALLS_FILE
        self.cost, self.clock = cost, clock
        self.warn = warn or (lambda line: print(line, file=sys.stderr, flush=True))
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.prefixes = _Prefixes()
        self.misses = 0

    def __call__(self, record: dict) -> None:
        finished = self.clock()
        sent = finished - float(record.get("wall_ms") or 0.0) / 1000.0
        parts = steps(record.get("system") or "", record.get("messages") or [], record.get("tools"))
        row = self.row(record, parts)
        with self.lock:
            shared, inside = self.prefixes.shared(parts, sent)
            if record.get("usage") is not None:
                self.prefixes.add(parts, finished)
            self.expect(row, record, parts, shared, inside)
            self.append(row)

    def append(self, row: dict) -> None:
        """One row kept and appended to calls.jsonl (the caller holds the lock)."""
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def pin(self, role: str, model: str, source: str, routes: Iterable[str] = ()) -> dict:
        """A row that is no call: the model this run pinned for a role, its routes, and where the choice came from."""
        now_ctx = current()
        row = {"time": now(), "kind": PIN, "run_id": now_ctx.get("run_id"), "stage": now_ctx.get("stage") or "",
               "role": role, "model": model, "source": source, "routes": list(routes)}
        with self.lock:
            self.append(row)
        return row

    def row(self, record: dict, parts: list[str]) -> dict:
        """The row of one call (see the module text); the reply's own fields (meta) go in last, so a reasoning the
        provider did not report is null there."""
        ctx, usage, model = record.get("context") or {}, record.get("usage"), str(record.get("model") or "")
        meta = dict(record.get("meta") or {})
        route = meta.pop("route", None) or route_of(model) or str(record.get("provider") or "")
        gateway, billed = meta.pop("usd_gateway", None), meta.pop("usd_billed", None)
        priced = meta.pop("usd_priced", None)
        row = {**_identity(ctx, record, model, route), **{k: int(getattr(usage, k, 0) or 0) for k in TOKENS},
               "http_attempts": int(record.get("attempts") or 1), "wall_ms": round(float(record.get("wall_ms") or 0)),
               "system_chars": len(parts[0]), "rest_chars": sum(len(p) for p in parts[1:])}
        if self.cost is not None:
            gateway = float(self.cost(usage, model)) if usage is not None else 0.0
        row.update(dollars(usage, model, route, gateway, billed, priced))
        row.update(_failure(ctx, record))
        row.update(meta)
        return row

    def expect(self, row: dict, record: dict, parts: list[str], shared: int, inside: int) -> None:
        """The cache counters of a row: uncacheable shared tokens, and where the provider's rule is known, the
        expected cached tokens and whether the call missed them (warned)."""
        total = row["input"] + row["cache_read"] + row["cache_write"]
        per_char = total / (sum(len(p) for p in parts) or 1)
        row["uncacheable_shared"] = round(inside * per_char)
        tail = CACHE_TAIL.get(str(record.get("provider") or ""))
        if tail is None or record.get("usage") is None:
            return
        expected = max(0, min(round(shared * per_char), total - tail))
        if expected < CACHE_MIN.get(str(record.get("provider") or ""), 0):
            expected = 0
        row["expected_cached"] = expected
        row["miss"] = bool(expected and row["cache_read"] < MISS_SHARE * expected)
        if row["miss"]:
            self.misses += 1
            if self.misses <= WARN_FIRST or self.misses % WARN_EVERY == 0:
                self.warn(f"cache miss #{self.misses}: stage {row['stage']} label {row['label']} group "
                          f"{row['group']}: expected {expected} cached tokens, read {row['cache_read']}")
