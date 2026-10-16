"""The scenario layer's seam onto the harness's one web reader (D374, D375), on builder.search.TestSearch only."""

from __future__ import annotations

from pathlib import Path

from kullback import domain
from kullback.synth import search as S
from tests.synth.fixture import URL, web


def test_the_adapter_passes_a_search_through_and_a_repeat_is_served_from_the_memo(tmp_path):
    engine, inner = web(tmp_path)
    assert [hit.url for hit in S.Crawled(engine, tmp_path).search("move a sealed widget", 5)] == [URL]
    again = S.Crawled(S.reader(tmp_path, env={}), tmp_path)
    assert [hit.url for hit in again.search("move a sealed widget", 5)] == [URL]
    assert inner.queries == ["move a sealed widget"]


def test_a_page_is_read_through_the_crawl_and_its_text_is_the_cached_text_under_the_key_recorded(tmp_path):
    engine, inner = web(tmp_path)
    keys: dict = {}
    [page] = S.Crawled(engine, tmp_path, keys=keys).fetch([URL])
    assert page.error is None and "sealed" in page.text and "<p>" not in page.text
    assert page.text == domain.cached_text(tmp_path, keys[URL]) and inner.fetched == [URL]


def test_a_refused_or_unreadable_page_is_a_page_error_and_a_note_never_a_raise(tmp_path):
    engine, inner = web(tmp_path, pages={})
    notes: list = []
    [corpus] = S.Crawled(engine, tmp_path, corpus_url=URL, notes=notes).fetch([URL])
    assert corpus.error == domain.CORPUS_URL and inner.fetched == []
    [unread] = S.Crawled(engine, tmp_path, notes=notes).fetch([URL])
    assert unread.error == domain.UNREACHABLE and unread.text == "" and any(URL in note for note in notes)


def test_no_module_of_the_layer_holds_a_second_reader_or_its_key_variables():
    folder = Path(S.__file__).parent
    texts = [path.read_text(encoding="utf-8").lower() for path in folder.glob("*.py")]
    assert texts and not [text for text in texts if "serper" in text or "kullback_search" in text]
