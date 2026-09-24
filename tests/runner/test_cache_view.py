"""`kullback budget cache`: what the prompt cache did per stage, read off a build's feed."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kullback import cli
from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from kullback.builder.domain_tools import reader_model
from kullback.runner import budget, cache_view, feed

MODEL = "vendor/cached"


@pytest.fixture(autouse=True)
def priced(monkeypatch):
    monkeypatch.setitem(budget.PRICES, MODEL, {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.0})


def ask(model, system, turns):
    messages = [{"role": "system", "content": system}]
    for turn in range(turns):
        messages.append({"role": "user", "content": f"ask {turn}"})
        model.query(list(messages), tools=[{"name": "lookup", "description": "d", "input_schema": {}}])
        messages.append({"role": "assistant", "content": f"answer {turn}"})


def stage(view, name):
    return next(s for s in view["stages"] if s["stage"] == name)


def test_a_conversation_that_reads_its_prefix_shows_a_high_read_share_and_one_new_prefix_write(workdir):
    replies = [ModelReply(content="a", usage=Usage(input=5, cache_write=1000)),
               ModelReply(content="b", usage=Usage(input=5, cache_read=1000, cache_write=40)),
               ModelReply(content="c", usage=Usage(input=5, cache_read=1040, cache_write=40))]
    ask(budget.BudgetedModel(TestModel(replies), "examiner", workdir, model_id=MODEL), "a fixed system", 3)
    view = stage(cache_view.cache_view(workdir), "examiner")
    assert (view["calls"], view["cache_read"], view["cache_write"], view["input"]) == (3, 2040, 1080, 15)
    assert view["read_share"] == pytest.approx(2040 / (2040 + 1080 + 15))
    assert (view["writes_new_prefix"], view["writes_seen_prefix"], view["appended_but_unread"]) == (1, 0, 0)
    assert (view["cold_writes"], view["tail_writes"]) == (1, 2)


def test_a_prefix_written_again_with_nothing_read_is_counted_and_so_is_the_append_that_missed(workdir):
    replies = [ModelReply(content="a", usage=Usage(cache_write=1000)),
               ModelReply(content="b", usage=Usage(cache_write=1040))]
    ask(budget.BudgetedModel(TestModel(replies), "examiner", workdir, model_id=MODEL), "a fixed system", 2)
    view = stage(cache_view.cache_view(workdir), "examiner")
    assert (view["writes_new_prefix"], view["writes_seen_prefix"], view["appended_but_unread"]) == (1, 1, 1)
    assert view["writes_seen_prefix_past_ttl"] == 0


def test_a_rewrite_after_a_gap_past_the_stage_ttl_is_told_apart_from_one_inside_it(workdir):
    rows = [{"kind": "model_call", "stage": "examiner", "model": MODEL, "at": at, "input": 0, "output": 1,
             "cache_read": 0, "cache_write": 900, "usd": 0.0, "prefix_hash": "p", "history_hash": f"h{at}"}
            for at in (0.0, 100.0, 1000.0)]
    feed.path_for(workdir).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    view = stage(cache_view.cache_view(workdir), "examiner")
    assert (view["writes_seen_prefix"], view["writes_seen_prefix_past_ttl"]) == (2, 1)


def test_a_feed_from_before_the_fingerprint_gets_its_write_back_from_the_dollars_and_no_prefix_counts(workdir):
    # The old ledger billed a chat call's written tokens twice: inside input, and again at the write rate.
    usd = (600 * 4.0 + 10 * 20.0 + 2000 * 0.4 + 500 * 5.0) / 1_000_000
    row = {"kind": "model_call", "stage": "runner", "model": MODEL, "at": 1.0, "input": 600, "output": 10,
           "cache_read": 2000, "usd": usd}
    feed.path_for(workdir).write_text(json.dumps(row) + "\n", encoding="utf-8")
    view = stage(cache_view.cache_view(workdir), "runner")
    assert (view["input"], view["cache_write"], view["cache_read"], view["derived_write"]) == (100, 500, 2000, True)
    assert view["hashed"] is False
    assert "not known" in "\n".join(cache_view.render(cache_view.cache_view(workdir)))


def test_the_largest_uncached_calls_are_listed_largest_first_and_at_most_ten(workdir):
    replies = [ModelReply(content=str(n), usage=Usage(input=n * 10)) for n in range(1, 13)]
    model = budget.BudgetedModel(TestModel(replies), "runner", workdir, model_id=MODEL)
    for n in range(12):
        model.query([{"role": "user", "content": f"q{n}"}])
    largest = stage(cache_view.cache_view(workdir), "runner")["largest_uncached"]
    assert [c["input"] for c in largest] == [n * 10 for n in range(12, 2, -1)]


def test_a_reader_asked_the_same_thing_twice_is_served_from_the_memo_and_counted_as_a_hit(workdir):
    inner = TestModel([ModelReply(content="proposal", usage=Usage(input=300, output=20))])
    first = reader_model(inner, workdir)
    first.query([{"role": "user", "content": "read this recording"}])
    reader_model(inner, workdir).query([{"role": "user", "content": "read this recording"}])
    view = stage(cache_view.cache_view(workdir), "readers")
    assert (view["calls"], view["memo_hits"]) == (2, 1)
    assert len(inner.calls) == 1
    assert budget.load_totals(workdir)["stages"]["readers"]["memo_hits"] == 1


def test_the_budget_cache_command_prints_one_row_per_stage_and_reads_nothing_else(workdir):
    ask(budget.BudgetedModel(TestModel([ModelReply(content="a", usage=Usage(cache_write=900))]), "examiner",
                             workdir, model_id=MODEL), "a fixed system", 1)
    before = sorted(p.name for p in workdir.iterdir())
    result = CliRunner().invoke(cli.app, ["budget", "cache", "--workdir", str(workdir)])
    assert result.exit_code == 0, result.output
    assert "| examiner | 1 |" in result.output
    assert sorted(p.name for p in workdir.iterdir()) == before
    as_json = CliRunner().invoke(cli.app, ["budget", "cache", "--workdir", str(workdir), "--json"])
    assert json.loads(as_json.output)["stages"][0]["stage"] == "examiner"


def test_the_budget_cache_command_on_an_empty_workdir_says_there_is_nothing_to_show(workdir):
    result = CliRunner().invoke(cli.app, ["budget", "cache", "--workdir", str(workdir)])
    assert result.exit_code == 0 and "no model_call lines" in result.output
