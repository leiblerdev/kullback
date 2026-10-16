"""The search reader: one provider behind an environment URL and key (D374).

A query string goes in and a list of result URLs comes out. It is the `search` callable
`domain.search_sources` already expects, so the key is read once there, its variable named in any note
and its value never printed or stored. The URL comes from KULLBACK_SEARCH_URL and defaults to one
public endpoint; the reader posts the query as JSON with the key in a header and reads the links of
the organic results. The endpoint is checked as a public destination before a socket is opened.

Answers are cached under the workdir by the query's hash, so a review that searches the same words
twice reads the cache and a rerun of the miner asks nothing again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from kullback import domain
from kullback.runner.records import read_json, write_json

URL_VAR = "KULLBACK_SEARCH_URL"
DEFAULT_URL = "https://google.serper.dev/search"
KEY_HEADER = "X-API-KEY"
CACHE = ("cache", "search")
# How many result URLs one query keeps.
RESULTS = 10

Post = Callable[[str, dict, dict], Any]


def endpoint(env: Optional[dict] = None) -> str:
    values = os.environ if env is None else env
    return str(values.get(URL_VAR) or DEFAULT_URL)


def http_post(url: str, headers: dict, body: dict, timeout: float = 20.0) -> Any:
    """One JSON POST over the network: the only socket this module opens, to a public destination only."""
    from urllib.request import Request, urlopen  # imported here: nothing offline pays for it

    why = domain.destination_refusal(url)
    if why:
        raise domain.DestinationRefused(why)
    request = Request(str(url), data=json.dumps(body).encode("utf-8"),  # noqa: S310
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as answer:  # noqa: S310
        return json.loads(answer.read().decode("utf-8", errors="replace"))


def result_urls(answer: Any, limit: int = RESULTS) -> list[str]:
    """The links of the organic results, in order, each once."""
    rows = (answer or {}).get("organic") if isinstance(answer, dict) else None
    out: list[str] = []
    for row in rows or ():
        url = str((row or {}).get("link") or (row or {}).get("url") or "") if isinstance(row, dict) else ""
        if url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out[:int(limit)]


def cache_path(workdir: Any, query: str) -> Path:
    digest = hashlib.sha256(str(query).strip().lower().encode("utf-8")).hexdigest()[:32]
    return Path(workdir).joinpath(*CACHE) / f"{digest}.json"


def reader(workdir: Any = None, *, post: Optional[Post] = None, env: Optional[dict] = None,
           limit: int = RESULTS) -> Callable[[str, str], list[str]]:
    """The `search(query, key)` callable: the cached answer where there is one, else one POST."""
    url = endpoint(env)
    send = post or http_post

    def search(query: str, key: str) -> list[str]:
        path = cache_path(workdir, query) if workdir is not None else None
        held = read_json(path, None) if path is not None else None
        if isinstance(held, dict) and isinstance(held.get("urls"), list):
            return [str(found) for found in held["urls"]]
        urls = result_urls(send(url, {KEY_HEADER: str(key)}, {"q": str(query)}), limit)
        if path is not None:
            write_json(path, {"query": str(query), "urls": urls})
        return urls

    return search


def run(query: str, workdir: Any = None, *, search: Optional[Callable[[str, str], list[str]]] = None,
        env: Optional[dict] = None) -> tuple[list[str], str]:
    """One query through `domain.search_sources`: the URLs, and a note where nothing ran."""
    return domain.search_sources(query, search=search or reader(workdir, env=env), env=env)


__all__ = ["DEFAULT_URL", "KEY_HEADER", "URL_VAR", "endpoint", "http_post", "reader", "result_urls", "run"]
