"""The scenario layer's seam onto the harness's one web reader (D374, D115).

Search and page fetch go through kullback/builder/search.py only: TinyFish when its key is set, then
Firecrawl keyless, in one chain, every answer memoized under <workdir>/web_cache keyed by the request,
and nothing live unless the switch that turns model calls on is set. `reader` is that reader for a
workdir, or None where nothing is live and no memo exists. `callables` turns it into the two shapes
the miner and the review take: a query in and result URLs out, and a URL in and page text out, so a
fetched page still lands in the D225 crawl's cache and a span's byte range cites it as before. A
failure is a note, never a raise; a key's value is never read here, only its variable named.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from kullback.ai.provider import LIVE_ENV_VAR
from kullback.builder import search as web

# How many result URLs one query keeps.
RESULTS = 10
OFF_NOTE = (f"search off: {LIVE_ENV_VAR} is not set; the chain reads {web.TINYFISH_KEY_VAR} first, "
            "then Firecrawl keyless")

Search = Callable[[str], list[str]]
Fetch = Callable[[str], str]


def reader(workdir: Any, env: Optional[dict] = None) -> Optional[Any]:
    """The harness's reader for this workdir: memoized over the chain, or None when live is off and no
    memo exists (a memo hit is served with live off)."""
    return web.search_for(workdir, env)


def _note(notes: list, text: str) -> None:
    if text not in notes:
        notes.append(text)


def callables(engine: Any, notes: Optional[list] = None, limit: int = RESULTS) -> tuple[Search, Fetch]:
    """The search and fetch callables over one reader; what fails lands in `notes` and reads as nothing."""
    notes = [] if notes is None else notes

    def search(query: str) -> list[str]:
        try:
            hits = engine.search(str(query), int(limit))
        except web.SearchError as error:
            _note(notes, f"search failed: {error}")
            return []
        out: list[str] = []
        for hit in hits:
            if hit.url.startswith(("http://", "https://")) and hit.url not in out:
                out.append(hit.url)
        return out

    def fetch(url: str) -> str:
        try:
            pages = engine.fetch([str(url)])
        except web.SearchError as error:
            _note(notes, f"page not read: {url}: {error}")
            return ""
        page = pages[0] if pages else None
        if page is None or page.error:
            _note(notes, f"page not read: {url}: {page.error if page is not None else 'no answer'}")
            return ""
        return page.text

    return search, fetch


__all__ = ["OFF_NOTE", "RESULTS", "callables", "reader"]
