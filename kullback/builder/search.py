"""Web search and page fetch for the Builder (D115).

The Builder reads a customer's domain off the traces, the schema and the signatures. The web is a
second source for one thing only: the words people use in that domain, which a vocabulary read off a
few hundred traces never covers (an airline agent asks for a "booking reference" where the tool
argument says `reservation_id`). Nothing here decides what a fact is; `builder/vocabulary.py` does,
and it takes the web's words as wording, never as evidence.

Two providers, tried in order: TinyFish (search and fetch, free with a key) and Firecrawl keyless
(no key, 1,000 credits a month; refused from some networks, which the chain then reports). Every answer is
memoized under `<workdir>/web_cache/<hash>.json` keyed by the request, so a repeat build reads the
same pages and a test never needs the network. Live requests are off unless the same switch that
turns model calls on is set (`provider.LIVE_ENV_VAR`): one way for a process to leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import httpx

from kullback.ai.provider import LIVE_ENV_VAR
from kullback.runner.records import Record

TINYFISH_KEY_VAR = "TINYFISH_API_KEY"
TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
TIMEOUT_S = 60.0
MAX_URLS_PER_FETCH = 10  # TinyFish's limit; Firecrawl scrapes one URL per call anyway
MAX_PAGE_CHARS = 20_000  # what the memo keeps of a page; a vocabulary prompt reads far less


class SearchError(RuntimeError):
    """No provider could answer; the message names every provider and what it said."""


class Hit(Record):
    """One search result."""
    title: str = ""
    url: str
    snippet: str = ""
    site: str = ""


class Page(Record):
    """One fetched page as markdown, or the reason it could not be read."""
    url: str
    title: str = ""
    text: str = ""
    error: Optional[str] = None


class Search(Protocol):
    name: str

    def search(self, query: str, limit: int = 8) -> list[Hit]: ...

    def fetch(self, urls: Iterable[str]) -> list[Page]: ...


def _client(client: Optional[httpx.Client]) -> tuple[httpx.Client, bool]:
    """The client to use and whether this provider made it; only a client it made is its to close."""
    return (client, False) if client is not None else (httpx.Client(timeout=TIMEOUT_S), True)


class _Closes:
    """Closes the connection pool this provider opened, and leaves a caller's own client alone."""

    client: httpx.Client
    owns_client: bool

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _raise_for(name: str, response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise SearchError(f"{name}: HTTP {response.status_code}: {response.text[:300]}")


class TinyFish(_Closes):
    """TinyFish search (GET, `query`) and fetch (POST, up to ten `urls`), `X-API-Key` on both."""

    name = "tinyfish"

    def __init__(self, api_key: str, client: Optional[httpx.Client] = None):
        self.api_key = api_key
        self.client, self.owns_client = _client(client)

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        response = self.client.get(TINYFISH_SEARCH_URL, params={"query": query}, headers=self._headers())
        _raise_for(self.name, response)
        rows = (response.json() or {}).get("results") or []
        return [Hit(title=r.get("title") or "", url=r["url"], snippet=r.get("snippet") or "",
                    site=r.get("site_name") or "") for r in rows if r.get("url")][:limit]

    def fetch(self, urls: Iterable[str]) -> list[Page]:
        pages: list[Page] = []
        batch = list(urls)
        for start in range(0, len(batch), MAX_URLS_PER_FETCH):
            chunk = batch[start:start + MAX_URLS_PER_FETCH]
            response = self.client.post(TINYFISH_FETCH_URL, json={"urls": chunk, "format": "markdown"},
                                        headers={**self._headers(), "Content-Type": "application/json"})
            _raise_for(self.name, response)
            body = response.json() or {}
            by_url = {r.get("url"): r for r in body.get("results") or []}
            errors = {e.get("url"): e.get("error") for e in body.get("errors") or []}
            for url in chunk:
                row = by_url.get(url)
                if row is None:
                    pages.append(Page(url=url, error=errors.get(url) or "no result"))
                    continue
                text = row.get("text")
                text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
                pages.append(Page(url=url, title=row.get("title") or "", text=text[:MAX_PAGE_CHARS]))
        return pages


class Firecrawl(_Closes):
    """Firecrawl v2 search and scrape, keyless: no Authorization header, the free monthly credits."""

    name = "firecrawl"

    def __init__(self, client: Optional[httpx.Client] = None):
        self.client, self.owns_client = _client(client)

    @staticmethod
    def _headers() -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _post(self, url: str, payload: dict) -> dict:
        response = self.client.post(url, json=payload, headers=self._headers())
        _raise_for(self.name, response)
        body = response.json() or {}
        if not body.get("success", True):
            raise SearchError(f"{self.name}: {body.get('error') or 'no success'}")
        return body

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        body = self._post(FIRECRAWL_SEARCH_URL, {"query": query[:500], "limit": limit})
        rows = ((body.get("data") or {}).get("web") if isinstance(body.get("data"), dict)
                else body.get("data")) or []
        return [Hit(title=r.get("title") or "", url=r["url"], snippet=r.get("description") or "",
                    site=httpx.URL(r["url"]).host or "") for r in rows if r.get("url")][:limit]

    def fetch(self, urls: Iterable[str]) -> list[Page]:
        pages: list[Page] = []
        for url in urls:
            try:
                body = self._post(FIRECRAWL_SCRAPE_URL, {"url": url, "formats": ["markdown"], "onlyMainContent": True})
            except SearchError as exc:
                pages.append(Page(url=url, error=str(exc)))
                continue
            data = body.get("data") or {}
            pages.append(Page(url=url, title=(data.get("metadata") or {}).get("title") or "",
                              text=(data.get("markdown") or "")[:MAX_PAGE_CHARS]))
        return pages


class Chain:
    """The providers in order; the first that answers wins, and a failure of all is one SearchError."""

    name = "chain"

    def __init__(self, providers: list[Any]):
        self.providers = list(providers)
        self.answered_by: Optional[str] = None

    def _first(self, call: str, *args: Any) -> Any:
        errors: list[str] = []
        for provider in self.providers:
            try:
                out = getattr(provider, call)(*args)
            except (SearchError, httpx.HTTPError, ValueError) as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            self.answered_by = provider.name
            return out
        raise SearchError("; ".join(errors) if errors else "no search provider is configured")

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        return self._first("search", query, limit)

    def fetch(self, urls: Iterable[str]) -> list[Page]:
        return self._first("fetch", list(urls))

    def close(self) -> None:
        for provider in self.providers:
            closing = getattr(provider, "close", None)
            if callable(closing):
                closing()

    def __enter__(self) -> "Chain":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class MemoSearch:
    """A content-addressed memo of answers under `<workdir>/web_cache`, so a repeat build reads
    the same pages and never leaves the machine for a request it has already made."""

    CACHE_DIR = "web_cache"
    name = "memo"

    def __init__(self, inner: Any, workdir: str | Path):
        self.inner = inner
        self.dir = Path(workdir) / self.CACHE_DIR
        self.calls = 0
        self.hits = 0

    def _path(self, kind: str, payload: Any) -> Path:
        blob = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, ensure_ascii=False)
        return self.dir / f"{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.json"

    def _memo(self, kind: str, payload: Any, produce: Any, model: Any) -> list:
        self.calls += 1
        path = self._path(kind, payload)
        if path.is_file():
            self.hits += 1
            stored = json.loads(path.read_text(encoding="utf-8"))
            return [model.model_validate(row) for row in stored["rows"]]
        rows = produce()
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind, "payload": payload, "provider": getattr(self.inner, "answered_by",
                                    None) or getattr(self.inner, "name", None),
                                    "rows": [row.model_dump(mode="json") for row in rows]},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        return rows

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        return self._memo("search", {"query": query, "limit": limit}, lambda: self.inner.search(query, limit), Hit)

    def fetch(self, urls: Iterable[str]) -> list[Page]:
        batch = list(urls)
        return self._memo("fetch", {"urls": batch}, lambda: self.inner.fetch(batch), Page)

    def close(self) -> None:
        closing = getattr(self.inner, "close", None)
        if callable(closing):
            closing()

    def __enter__(self) -> "MemoSearch":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class TestSearch:
    """A scripted search for tests: hits by query substring, pages by URL; nothing leaves the machine."""

    name = "test"

    def __init__(self, hits: Optional[dict[str, list[dict]]] = None, pages: Optional[dict[str, str]] = None):
        self.hits = hits or {}
        self.pages = pages or {}
        self.queries: list[str] = []
        self.fetched: list[str] = []

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        self.queries.append(query)
        rows = [row for key, rows in self.hits.items() if key.lower() in query.lower() for row in rows]
        return [Hit.model_validate(row) for row in rows][:limit]

    def fetch(self, urls: Iterable[str]) -> list[Page]:
        out = []
        for url in urls:
            self.fetched.append(url)
            text = self.pages.get(url)
            out.append(Page(url=url, text=text) if text is not None else Page(url=url, error="not scripted"))
        return out


def live_allowed(env: Optional[dict] = None) -> bool:
    return (env if env is not None else os.environ).get(LIVE_ENV_VAR, "") == "1"


def providers_from_env(env: Optional[dict] = None) -> list[Any]:
    """TinyFish when its key is set, then Firecrawl keyless."""
    values = env if env is not None else os.environ
    out: list[Any] = []
    if values.get(TINYFISH_KEY_VAR):
        out.append(TinyFish(values[TINYFISH_KEY_VAR]))
    out.append(Firecrawl())
    return out


def search_for(workdir: str | Path, env: Optional[dict] = None) -> Optional[MemoSearch]:
    """The Builder's search for this workdir: memoized over the provider chain, or None when live
    requests are off. A memo hit is served whether live is on or not, which is what lets a
    repeat build in CI read the pages the first build read."""
    values = env if env is not None else os.environ
    if not live_allowed(values):
        memo = MemoSearch(Chain([]), workdir)
        return memo if memo.dir.is_dir() else None
    return MemoSearch(Chain(providers_from_env(values)), workdir)
