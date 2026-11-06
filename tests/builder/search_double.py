"""A scripted search for tests: hits by query substring, pages by URL; nothing leaves the machine."""

from __future__ import annotations

from typing import Iterable, Optional

from kullback.builder.search import Hit, Page


class ScriptedSearch:
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
