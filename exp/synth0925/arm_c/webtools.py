"""Web search and fetch for arm C: one search endpoint, two fetch paths.

Reads no names from code: the caller hands over query text learned from the
workdir, and this module only transports it. Every destination passes the
harness fetch guard before a socket opens. Search itself is free of charge;
both vendors run on free tiers, so search spend stays zero and is counted
in calls, not dollars.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"
FIRECRAWL_SCRAPE = "https://api.firecrawl.dev/v1/scrape"

# Host classes that never carry customer voice: code forges and paper
# archives. Excluded as classes, not as named sources.
_BLOCKED_SUFFIXES = ("github.com", "arxiv.org", "huggingface.co")


def guard_ok(url: str) -> str:
    """Empty string where a fetch may go, else the reason in words."""
    from kullback import domain as D

    try:
        refused = D.destination_refusal(url)
    except Exception:
        return "guard failed"
    if refused:
        return str(refused)
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    for suffix in _BLOCKED_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return "not customer voice"
    return ""


def tinyfish_search(query: str, key: str, limit: int = 8) -> list:
    params = urllib.parse.urlencode({"query": query, "page": 0})
    req = urllib.request.Request(f"{SEARCH_URL}?{params}", headers={"X-API-Key": key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8", "replace"))
    out = []
    for row in (body.get("results") or [])[:limit]:
        url = str(row.get("url") or "")
        if not url:
            continue
        out.append({"title": str(row.get("title") or ""), "url": url,
                    "snippet": str(row.get("snippet") or ""),
                    "site": str(row.get("site_name") or "")})
    return out


def tinyfish_fetch(urls: list, key: str) -> tuple:
    """Returns (pages, errors). Pages carry url, title and text only."""
    ok = [u for u in urls if not guard_ok(u)]
    refused = [u for u in urls if guard_ok(u)]
    pages, errors = [], [{"url": u, "error": guard_ok(u)} for u in refused]
    for i in range(0, len(ok), 10):
        batch = ok[i:i + 10]
        payload = json.dumps({"urls": batch, "format": "markdown"}).encode()
        req = urllib.request.Request(FETCH_URL, data=payload, method="POST",
                                     headers={"X-API-Key": key,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as error:
            errors += [{"url": u, "error": type(error).__name__} for u in batch]
            continue
        for row in body.get("results") or []:
            text = row.get("text") or ""
            pages.append({"url": str(row.get("url") or ""),
                          "title": str(row.get("title") or ""),
                          "text": str(text)[:6000] if isinstance(text, str) else ""})
        for row in body.get("errors") or []:
            errors.append({"url": str(row.get("url") or ""),
                           "error": str(row.get("error") or "fetch_error")})
    return pages, errors


def firecrawl_fetch(url: str) -> dict:
    """One keyless scrape attempt. Free tier, no key sent. Returns page or error."""
    reason = guard_ok(url)
    if reason:
        return {"url": url, "error": reason}
    payload = json.dumps({"url": url, "formats": ["markdown"]}).encode()
    req = urllib.request.Request(FIRECRAWL_SCRAPE, data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as error:
        return {"url": url, "error": type(error).__name__}
    data = body.get("data") or {}
    text = data.get("markdown") or ""
    return {"url": url, "title": str(data.get("title") or ""),
            "text": str(text)[:6000] if isinstance(text, str) else "",
            "error": "" if text else "empty_content"}


def tinyfish_key() -> str:
    from kullback.ai import provider as P

    from .cost import DOTENV

    P.load_dotenv(DOTENV)
    return os.environ.get("TINYFISH_API_KEY", "")
