"""shared/search.py: two providers, one chain, a memo on disk, and no network unless live is on (D115)."""

from __future__ import annotations

import json

import httpx
import pytest

from kullback.ai.provider import LIVE_ENV_VAR
from kullback.builder import search as sx
from kullback.builder.search import TestSearch as ScriptedSearch


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


TINYFISH_SEARCH = {"query": "q", "results": [
    {"position": 1, "site_name": "docs.example", "title": "Booking help", "snippet": "your booking reference",
     "url": "https://docs.example/booking"},
    {"position": 2, "site_name": "blog.example", "title": "Other", "snippet": "", "url": "https://blog.example/x"},
], "total_results": 2, "page": 1}


def test_tinyfish_search_and_fetch_take_the_key_header_and_the_documented_shapes():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.search.tinyfish.ai":
            assert request.method == "GET" and request.url.params["query"] == "airline booking reference"
            return httpx.Response(200, json=TINYFISH_SEARCH)
        assert request.method == "POST" and json.loads(request.content)["urls"] == ["https://docs.example/booking"]
        return httpx.Response(200, json={"results": [{"url": "https://docs.example/booking", "title": "Booking help",
                                                       "text": "# Booking\nGive the agent your booking reference."}],
                                         "errors": []})

    tiny = sx.TinyFish("k1", client=_client(handler))
    hits = tiny.search("airline booking reference", limit=1)
    assert [h.url for h in hits] == ["https://docs.example/booking"] and hits[0].site == "docs.example"
    pages = tiny.fetch([h.url for h in hits])
    assert pages[0].text.startswith("# Booking") and pages[0].error is None
    assert all(r.headers["X-API-Key"] == "k1" for r in seen)


def test_tinyfish_reports_a_page_it_could_not_read_by_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "errors": [{"url": "https://x/y", "error": "timeout"}]})

    pages = sx.TinyFish("k", client=_client(handler)).fetch(["https://x/y"])
    assert pages[0].error == "timeout" and pages[0].text == ""


def test_firecrawl_is_keyless_and_reads_the_v2_shapes():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"success": True, "data": {"web": [
                {"title": "T", "description": "D", "url": "https://a.example/p"}]}})
        return httpx.Response(200, json={"success": True, "data": {"markdown": "# P\nbooking reference",
                                                                    "metadata": {"title": "P"}}})

    fire = sx.Firecrawl(client=_client(handler))
    hits = fire.search("anything")
    assert hits[0].url == "https://a.example/p" and hits[0].site == "a.example"
    pages = fire.fetch([hits[0].url])
    assert pages[0].title == "P" and "booking reference" in pages[0].text
    assert all("Authorization" not in r.headers for r in seen)


def test_firecrawl_refusing_keyless_is_a_search_error_with_its_words():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "error": "your IP address looks suspicious"})

    with pytest.raises(sx.SearchError, match="looks suspicious"):
        sx.Firecrawl(client=_client(handler)).search("q")


def test_the_chain_falls_through_to_the_next_provider_and_names_every_failure():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    def fine(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "data": {"web": [{"url": "https://ok.example/"}]}})

    chain = sx.Chain([sx.TinyFish("k", client=_client(failing)), sx.Firecrawl(client=_client(fine))])
    assert [h.url for h in chain.search("q")] == ["https://ok.example/"] and chain.answered_by == "firecrawl"
    dead = sx.Chain([sx.TinyFish("k", client=_client(failing)), sx.Firecrawl(client=_client(failing))])
    with pytest.raises(sx.SearchError, match="tinyfish: .*HTTP 500.*; firecrawl: .*HTTP 500"):
        dead.search("q")
    with pytest.raises(sx.SearchError, match="no search provider"):
        sx.Chain([]).search("q")


def test_the_memo_answers_a_repeat_from_disk_and_records_the_provider(tmp_path):
    inner = ScriptedSearch(hits={"booking": [{"url": "https://a/", "title": "A"}]}, pages={"https://a/": "text"})
    memo = sx.MemoSearch(inner, tmp_path)
    first = memo.search("airline booking")
    again = memo.search("airline booking")
    assert first == again and inner.queries == ["airline booking"] and memo.hits == 1
    pages = memo.fetch(["https://a/"])
    assert pages[0].text == "text" and memo.fetch(["https://a/"]) == pages and inner.fetched == ["https://a/"]
    stored = [json.loads(p.read_text()) for p in (tmp_path / "web_cache").glob("*.json")]
    assert {s["kind"] for s in stored} == {"search", "fetch"} and all(s["provider"] == "test" for s in stored)


def test_live_off_means_no_provider_but_the_memo_still_answers(tmp_path):
    assert sx.search_for(tmp_path, env={}) is None
    (tmp_path / "web_cache").mkdir()
    memo = sx.search_for(tmp_path, env={})
    assert memo is not None and memo.inner.providers == []
    with pytest.raises(sx.SearchError):
        memo.search("q")


def test_live_on_builds_tinyfish_then_firecrawl_keyless():
    both = sx.providers_from_env({LIVE_ENV_VAR: "1", sx.TINYFISH_KEY_VAR: "k"})
    assert [p.name for p in both] == ["tinyfish", "firecrawl"]
    assert [p.name for p in sx.providers_from_env({LIVE_ENV_VAR: "1"})] == ["firecrawl"]
    assert sx.live_allowed({LIVE_ENV_VAR: "1"}) and not sx.live_allowed({})
