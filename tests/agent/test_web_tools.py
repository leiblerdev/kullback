"""The shared web tools: search and read_page over an injected search, refusals, search off, on_call, imports."""

from __future__ import annotations

import ast
import asyncio
from functools import partial
from pathlib import Path

import pytest

from kullback.agent import web_tools
from kullback.agent.extensions import ExtensionAPI
from kullback.agent.harness import AgentHarness
from kullback.agent.web_tools import OFF, TOOL_NAMES, numbered_lines, register_web_tools
from kullback.ai.provider import TestModel
from kullback.builder.search import MemoSearch
from kullback.builder.search import TestSearch as ScriptedSearch
from kullback.domain import destination_refusal

PAGE_URL = "https://example.com/help/refunds"
HITS = {"refund": [{"title": "Refunds", "url": PAGE_URL, "snippet": "how refunds work", "site": "example.com"},
                   {"title": "Returns", "url": "https://example.org/returns", "snippet": "returns", "site": "example.org"}]}
PAGE_TEXT = "Refunds\nA refund takes five days.\nContact support for help."


def public(host):
    return ["93.184.215.14"]


# The domain module's check over a scripted resolver: every name answers with one public address, no network.
REFUSE = partial(destination_refusal, resolve=public)


def setup(tmp_path, **kwargs):
    double = ScriptedSearch(hits=HITS, pages={PAGE_URL: PAGE_TEXT})
    harness = AgentHarness(TestModel(["done"]))
    names = register_web_tools(ExtensionAPI(harness), search=MemoSearch(double, tmp_path), refuse=REFUSE, **kwargs)
    return harness.registry, double, names


def call(registry, name, arguments):
    return asyncio.run(registry.get(name).run(arguments))


def test_both_tools_register_and_are_listed_by_name(tmp_path):
    registry, _, names = setup(tmp_path)
    assert names == list(TOOL_NAMES) == ["search", "read_page"]
    assert sorted(registry.names()) == ["read_page", "search"]
    assert all("Example:" in registry.get(name).description for name in names)


def test_search_rows_come_from_the_double_and_a_repeat_query_is_a_memo_hit(tmp_path):
    registry, double, _ = setup(tmp_path)
    first = call(registry, "search", {"query": "refund time", "limit": 5})
    assert not first.is_error
    assert [row["url"] for row in first.details["rows"]] == [PAGE_URL, "https://example.org/returns"]
    assert first.details["rows"][0] == {"title": "Refunds", "url": PAGE_URL, "snippet": "how refunds work",
                                        "site": "example.com"}
    assert first.details["summary"].startswith("2 hits")
    second = call(registry, "search", {"query": "refund time", "limit": 5})
    assert second.details == first.details
    assert double.queries == ["refund time"]


def test_search_limit_above_ten_is_refused_as_invalid_arguments(tmp_path):
    registry, double, _ = setup(tmp_path)
    result = call(registry, "search", {"query": "refund", "limit": 11})
    assert result.is_error and "limit" in result.content
    assert double.queries == []


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://localhost:8080/", "http://10.0.0.5/",
                                 "file:///etc/hosts"])
def test_read_page_refuses_a_private_or_loopback_destination_with_a_summary_and_no_rows(tmp_path, url):
    registry, double, _ = setup(tmp_path)
    result = call(registry, "read_page", {"url": url})
    assert not result.is_error
    assert result.details["rows"] == []
    assert result.details["summary"].startswith("read_page refused")
    assert double.fetched == []


def test_read_page_gives_numbered_lines_with_title_and_length(tmp_path):
    registry, double, _ = setup(tmp_path)
    result = call(registry, "read_page", {"url": PAGE_URL})
    assert result.details["rows"] == [{"line": 1, "text": "Refunds"}, {"line": 2, "text": "A refund takes five days."},
                                      {"line": 3, "text": "Contact support for help."}]
    assert f"{len(PAGE_TEXT)} characters" in result.details["summary"]
    assert double.fetched == [PAGE_URL]


def test_read_page_caps_lines_at_max_chars_and_cuts_the_line_that_does_not_fit(tmp_path):
    registry, _, _ = setup(tmp_path)
    result = call(registry, "read_page", {"url": PAGE_URL, "max_chars": 20})
    assert result.details["rows"] == [{"line": 1, "text": "Refunds"}, {"line": 2, "text": "A refund tak"}]
    assert "showing lines 1 to 2 of 3" in result.details["summary"]
    assert "line 2 cut to 12 of 25 characters" in result.details["summary"]


def test_read_page_names_the_next_line_while_more_remain(tmp_path):
    registry, _, _ = setup(tmp_path)
    first = call(registry, "read_page", {"url": PAGE_URL, "max_chars": 8})
    assert first.details["rows"] == [{"line": 1, "text": "Refunds"}]
    assert first.details["summary"].endswith("call again with from_line=2")
    whole = call(registry, "read_page", {"url": PAGE_URL})
    assert "call again" not in whole.details["summary"]


def test_a_middle_window_keeps_the_line_numbers_of_the_whole_page():
    text = "\n".join(f"line {number}" for number in range(1, 11))
    assert numbered_lines(text, 13, from_line=4) == [{"line": 4, "text": "line 4"}, {"line": 5, "text": "line 5"}]


def test_a_long_line_is_cut_and_the_next_line_is_read_on_the_following_call():
    text = "short\n" + "x" * 50 + "\nafter the long line"
    first = numbered_lines(text, 20)
    assert first == [{"line": 1, "text": "short"}, {"line": 2, "text": "x" * 14}]
    assert numbered_lines(text, 20, from_line=first[-1]["line"] + 1) == [{"line": 3, "text": "after the long line"}]


def test_read_page_from_a_line_past_the_end_shows_no_rows_and_says_so(tmp_path):
    registry, _, _ = setup(tmp_path)
    result = call(registry, "read_page", {"url": PAGE_URL, "from_line": 9})
    assert result.details["rows"] == []
    assert "from_line=9 is past the end" in result.details["summary"]


def test_a_single_line_longer_than_max_chars_is_cut_not_dropped():
    assert numbered_lines("abcdefghij", 4) == [{"line": 1, "text": "abcd"}]


def test_read_page_says_the_page_error_when_the_page_could_not_be_read(tmp_path):
    registry, _, _ = setup(tmp_path)
    result = call(registry, "read_page", {"url": "https://example.com/missing"})
    assert result.details["rows"] == []
    assert "error: not scripted" in result.details["summary"]


def test_with_no_search_both_tools_register_and_answer_that_search_is_off(tmp_path):
    harness = AgentHarness(TestModel(["done"]))
    assert register_web_tools(ExtensionAPI(harness), search=None, refuse=REFUSE) == ["search", "read_page"]
    for name, arguments in (("search", {"query": "refund"}), ("read_page", {"url": PAGE_URL})):
        result = call(harness.registry, name, arguments)
        assert not result.is_error and result.details == {"summary": OFF, "rows": []}
    assert "=" not in OFF


def test_read_page_asks_the_injected_refuser_before_any_fetch(tmp_path):
    asked = []

    def refuse(url):
        asked.append(url)
        return "the scripted refuser says no"

    double = ScriptedSearch(pages={PAGE_URL: PAGE_TEXT})
    harness = AgentHarness(TestModel(["done"]))
    register_web_tools(ExtensionAPI(harness), search=MemoSearch(double, tmp_path), refuse=refuse)
    result = call(harness.registry, "read_page", {"url": PAGE_URL})
    assert asked == [PAGE_URL] and double.fetched == []
    assert result.details["rows"] == [] and "the scripted refuser says no" in result.details["summary"]


def test_the_module_imports_nothing_above_the_agent_layer():
    tree = ast.parse(Path(web_tools.__file__).read_text(encoding="utf-8"))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    harness_names = [name for name in names if name.split(".")[0] == "kullback"]
    assert harness_names and all(name.startswith(("kullback.agent", "kullback.ai")) for name in harness_names)


def test_on_call_sees_every_call_and_never_changes_the_result(tmp_path):
    seen = []

    def keep(name, args, result):
        seen.append((name, args, result.summary, len(result.rows)))
        result.rows.clear()

    registry, _, _ = setup(tmp_path, on_call=keep)
    searched = call(registry, "search", {"query": "refund"})
    read = call(registry, "read_page", {"url": PAGE_URL})
    refused = call(registry, "read_page", {"url": "http://127.0.0.1/"})
    assert [(name, rows) for name, _, _, rows in seen] == [("search", 2), ("read_page", 3), ("read_page", 0)]
    assert seen[0][1] == {"query": "refund", "limit": 8}
    assert seen[1][1] == {"url": PAGE_URL, "max_chars": 6000, "from_line": 1}
    assert len(searched.details["rows"]) == 2 and len(read.details["rows"]) == 3
    assert refused.details["summary"] == seen[2][2]
