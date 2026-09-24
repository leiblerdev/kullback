"""What the provider's prompt cache did for one build, per stage, read off the feed (`kullback budget cache`).

Read only. Every priced call is a `model_call` line in events.jsonl (budget.record_call writes it);
since ov-cache each line also carries `cache_write` and the request's cache fingerprint
(kullback.ai.cache.fingerprint). Per stage the view counts:

- the share of prompt tokens served from the cache;
- cache writes on a prefix the stage had not sent before (the first write of a prefix, expected);
- cache writes on a prefix the stage had already sent, with nothing read (the prefix was there to
  read and was written again: expiry when the gap was past the TTL, else an invalidator below the
  prefix hash or a concurrent start);
- conversations that grew by appending (the call's parent is an earlier call's history under the
  same prefix) and still read nothing;
- the ten largest uncached calls.

A feed written before the fingerprint existed has no hashes, so the two prefix counts print as not
known, and a line with no `cache_write` gets it back from its dollars at the price the ledger used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.ai.cache import DEFAULT_CACHE_TTL, ttl_for
from kullback.runner import budget, feed

TOP_UNCACHED = 10
TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}


def model_calls(workdir: Any) -> list[dict]:
    """The feed's model_call lines, in the order they were written; a torn line is skipped."""
    path = feed.path_for(workdir)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("kind") == "model_call":
            rows.append(row)
    return rows


def _derived_write(row: dict) -> Optional[int]:
    """A line with no cache_write: the write that makes its dollars add up at the ledger's price.

    The ledger priced each call as input, output, cache_read and cache_write at their rates, so the
    write is what is left of the dollars over the other three. None where the model is unpriced or
    its write rate is zero, since nothing can be recovered then.
    """
    price = budget.price_for(row.get("model"))
    if not price or not price.get("cache_write"):
        return None
    rest = (int(row.get("input") or 0) * price["input"] + int(row.get("output") or 0) * price["output"]
            + int(row.get("cache_read") or 0) * price["cache_read"])
    return max(0, int(round((float(row.get("usd") or 0.0) * 1_000_000 - rest) / price["cache_write"])))


def counts(row: dict) -> dict:
    """One call's disjoint token counts: uncached input, cache read, cache write, and the prompt.

    A line from the current ledger holds three disjoint counts. A line from before it held them
    only for the Messages API; the chat adapters kept the written tokens inside input then (usage.py
    now takes them out), so for those the write is taken back out of input here.
    """
    read = int(row.get("cache_read") or 0)
    if "cache_write" in row:
        uncached, write, derived = int(row.get("input") or 0), int(row.get("cache_write") or 0), False
    else:
        write, derived = _derived_write(row) or 0, True
        uncached = int(row.get("input") or 0)
        if not str(row.get("model") or "").startswith("anthropic/"):
            uncached = max(0, uncached - write)
    return {"input": uncached, "cache_read": read, "cache_write": write,
            "prompt": uncached + read + write, "derived_write": derived}


def _tally_counts(view: dict, row: dict, c: dict) -> None:
    """Add one call's token counts, memo hit and write kind to the stage's view."""
    view["derived_write"] = view["derived_write"] or c["derived_write"]
    for key in ("prompt", "cache_read", "cache_write", "input"):
        view[key] += c[key]
    view["memo_hits"] += int(bool(row.get("memo_hit")))
    if c["cache_write"] and c["cache_read"]:
        view["tail_writes"] += 1
    elif c["cache_write"]:
        view["cold_writes"] += 1


def _tally_prefix(view: dict, row: dict, c: dict, at: float, ttl: float,
                  last_seen: dict[str, float], histories: set[tuple[str, str]]) -> None:
    """Add one hashed call's prefix writes and unread appends, then remember its prefix and history."""
    prefix = row["prefix_hash"]
    seen_at = last_seen.get(prefix)
    if c["cache_write"] and seen_at is None:
        view["writes_new_prefix"] += 1
    elif c["cache_write"] and not c["cache_read"]:
        view["writes_seen_prefix"] += 1
        view["writes_seen_prefix_past_ttl"] += int(at - seen_at > ttl)
    parent = row.get("parent_hash")
    if parent and (prefix, parent) in histories and not c["cache_read"] and not row.get("memo_hit"):
        view["appended_but_unread"] += 1
    last_seen[prefix] = at
    if row.get("history_hash"):
        histories.add((prefix, row["history_hash"]))


def stage_view(rows: Iterable[dict], stage: str) -> dict:
    """The cache counts of one stage over its calls, in feed order."""
    rows = list(rows)
    ttl = TTL_SECONDS.get(ttl_for(stage), TTL_SECONDS[DEFAULT_CACHE_TTL])
    hashed = any(row.get("prefix_hash") for row in rows)
    last_seen: dict[str, float] = {}
    histories: set[tuple[str, str]] = set()
    view = {"stage": stage, "calls": len(rows), "prompt": 0, "cache_read": 0, "cache_write": 0, "input": 0,
            "cold_writes": 0, "tail_writes": 0, "writes_new_prefix": 0, "writes_seen_prefix": 0,
            "writes_seen_prefix_past_ttl": 0, "appended_but_unread": 0, "memo_hits": 0,
            "hashed": hashed, "derived_write": False, "largest_uncached": []}
    calls = []
    for row in rows:
        c = counts(row)
        _tally_counts(view, row, c)
        prefix = row.get("prefix_hash")
        at = float(row.get("at") or 0.0)
        if prefix:
            _tally_prefix(view, row, c, at, ttl, last_seen, histories)
        calls.append({"at": at, "prefix_hash": prefix, "n_messages": row.get("n_messages"), **c})
    view["read_share"] = view["cache_read"] / view["prompt"] if view["prompt"] else 0.0
    view["largest_uncached"] = sorted(calls, key=lambda c: (-c["input"], c["at"]))[:TOP_UNCACHED]
    return view


def cache_view(workdir: Any) -> dict:
    """Every stage's view, stages in the order their first call was written."""
    by_stage: dict[str, list[dict]] = {}
    for row in model_calls(workdir):
        by_stage.setdefault(str(row.get("stage") or "?"), []).append(row)
    return {"workdir": str(Path(workdir)), "stages": [stage_view(rows, stage) for stage, rows in by_stage.items()]}


def _known(view: dict, key: str) -> str:
    return f"{view[key]:,}" if view["hashed"] else "not known"


def render(view: dict) -> list[str]:
    """The view as the lines the command prints."""
    lines = [f"prompt cache by stage in {view['workdir']}"]
    if not view["stages"]:
        return lines + ["no model_call lines in the feed"]
    header = ("stage", "calls", "read share", "cold writes", "tail writes", "write, new prefix",
              "write, seen prefix (past TTL)", "appended, unread", "memo hits")
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in view["stages"]:
        seen = (f"{s['writes_seen_prefix']:,} ({s['writes_seen_prefix_past_ttl']:,})" if s["hashed"]
                else "not known")
        lines.append("| " + " | ".join([
            s["stage"], f"{s['calls']:,}", f"{s['read_share']:.1%}", f"{s['cold_writes']:,}",
            f"{s['tail_writes']:,}", _known(s, "writes_new_prefix"), seen, _known(s, "appended_but_unread"),
            f"{s['memo_hits']:,}"]) + " |")
    for s in view["stages"]:
        note = " (cache_write recovered from the dollars)" if s["derived_write"] else ""
        lines += ["", f"{s['stage']}: prompt {s['prompt']:,} tokens, read {s['cache_read']:,}, "
                      f"written {s['cache_write']:,}, uncached {s['input']:,}{note}",
                  f"ten largest uncached calls in {s['stage']}:",
                  "| at | uncached | read | written | prompt | messages | prefix |", "|---|---|---|---|---|---|---|"]
        for c in s["largest_uncached"]:
            lines.append(f"| {c['at']:.0f} | {c['input']:,} | {c['cache_read']:,} | {c['cache_write']:,} | "
                         f"{c['prompt']:,} | {c['n_messages'] if c['n_messages'] is not None else '-'} | "
                         f"{c['prefix_hash'] or '-'} |")
    return lines


__all__ = ["cache_view", "counts", "model_calls", "render", "stage_view"]
