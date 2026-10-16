"""The scenario layer's seam onto the harness's one web reader (D374, D115) and the shared web tools (D375).

Search and page fetch go through kullback/builder/search.py only: TinyFish when its key is set, then
Firecrawl keyless, in one chain, every answer memoized under <workdir>/web_cache keyed by the request,
and nothing live unless the switch that turns model calls on is set. `reader` is that reader for a
workdir, or None where nothing is live and no memo exists. The roles never wrap it in tools of their
own: they register kullback.agent.web_tools with `Crawled` over the reader as the search. `Crawled` has
the D115 shape and reads every page through the D225 crawl (the corpus refusal, the exclude list, the
content addressed cache), so the text the agent reads is the cached text a span's byte range is checked
against, and the cache key of every URL read lands in `keys`. A failure is a note or a Page error, never
a raise; a key's value is never read here, only its variable named.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback import domain
from kullback.ai.provider import LIVE_ENV_VAR
from kullback.builder import search as web

OFF_NOTE = (f"search off: {LIVE_ENV_VAR} is not set; the chain reads {web.TINYFISH_KEY_VAR} first, "
            "then Firecrawl keyless")


def reader(workdir: Any, env: Optional[dict] = None) -> Optional[Any]:
    """The harness's reader for this workdir: memoized over the chain, or None when live is off and no
    memo exists (a memo hit is served with live off)."""
    return web.search_for(workdir, env)


def _note(notes: list, text: str) -> None:
    if text not in notes:
        notes.append(text)


class Crawled:
    """The D115 shape over a reader, each page read through the crawl and filed in its cache."""

    def __init__(self, engine: Any, workdir: Any, *, keys: Optional[dict] = None, corpus_url: Optional[str] = None,
                 exclude: Iterable[str] = (), notes: Optional[list] = None) -> None:
        self.engine = engine
        self.workdir = workdir
        self.keys = {} if keys is None else keys
        self.corpus_url = corpus_url
        self.exclude = tuple(exclude)
        self.notes = [] if notes is None else notes
        self.name = getattr(engine, "name", "reader")
        self._titles: dict[str, str] = {}

    def search(self, query: str, limit: int) -> list:
        try:
            return list(self.engine.search(str(query), int(limit)))
        except web.SearchError as error:
            _note(self.notes, f"search failed: {error}")
            return []

    def _html(self, url: str) -> str:
        """The reader's fetch as the crawl's page fetch: the text, or a raise the crawl counts as unread."""
        pages = self.engine.fetch([url])
        page = pages[0] if pages else None
        if page is None or page.error:
            raise web.SearchError(page.error if page is not None else "no answer")
        self._titles[url] = page.title
        return page.text

    def fetch(self, urls: list[str]) -> list:
        out = []
        for url in urls:
            crawled = domain.crawl(self.workdir, [{"url": url, "depth": 0, "via": domain.SEARCHED}],
                                   fetch=self._html, depth=0, corpus_url=self.corpus_url, exclude=self.exclude,
                                   limit=1)
            if crawled["pages"]:
                key = crawled["pages"][0]["key"]
                self.keys[url] = key
                out.append(web.Page(url=url, title=self._titles.get(url, ""),
                                    text=domain.cached_text(self.workdir, key)))
                continue
            reason = (crawled["refused"] or [{"reason": domain.UNREACHABLE}])[0]["reason"]
            _note(self.notes, f"page not read: {url}: {reason}")
            out.append(web.Page(url=url, error=reason))
        return out


__all__ = ["OFF_NOTE", "Crawled", "reader"]
