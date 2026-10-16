"""The scenario layer's seam onto the harness's one web reader (D374), on builder.search.TestSearch only."""

from __future__ import annotations

from pathlib import Path

from kullback.synth import search as S
from tests.synth.fixture import URL, web


def test_the_seam_reads_urls_and_page_text_through_the_memo_and_serves_a_repeat_from_it(tmp_path):
    engine, inner = web(tmp_path)
    search, fetch = S.callables(engine)
    assert search("move a sealed widget") == [URL] and "sealed" in fetch(URL)
    again, again_fetch = S.callables(S.reader(tmp_path, env={}))
    assert again("move a sealed widget") == [URL] and "sealed" in again_fetch(URL)
    assert inner.queries == ["move a sealed widget"] and inner.fetched == [URL]


def test_a_page_the_provider_cannot_read_is_empty_text_and_a_note(tmp_path):
    engine, _ = web(tmp_path, pages={})
    notes: list = []
    _, fetch = S.callables(engine, notes)
    assert fetch(URL) == "" and any(URL in note for note in notes)


def test_no_module_of_the_layer_holds_a_second_reader_or_its_key_variables():
    folder = Path(S.__file__).parent
    texts = [path.read_text(encoding="utf-8").lower() for path in folder.glob("*.py")]
    assert texts and not [text for text in texts if "serper" in text or "kullback_search" in text]
