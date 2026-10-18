"""`scripts/check_doc_links.py`: every relative link in the docs, README.md and AGENTS.md resolves."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_doc_links as C  # noqa: E402


def test_every_relative_link_in_the_docs_resolves_to_a_file_or_heading():
    assert list(C.broken_links(ROOT)) == []


def test_a_missing_file_and_a_missing_heading_are_reported_and_fenced_links_are_not(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "b.md").write_text("# Two words\n", encoding="utf-8")
    (docs / "a.md").write_text(
        "# A\n[ok](b.md#two-words) [self](#a) [web](https://x.org)\n"
        "[gone](missing.md) [bad](b.md#nope)\n```\n[fenced](nowhere.md)\n```\n",
        encoding="utf-8",
    )
    assert list(C.broken_links(tmp_path)) == [
        "docs/a.md: missing.md (no such file)",
        "docs/a.md: b.md#nope (no such heading)",
    ]
