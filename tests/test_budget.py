"""Tests for cost accounting, the 40 percent context cap (D65) and the spend ceiling (D86)."""

from __future__ import annotations

import json

import pytest

from harness.shared import budget
from harness.shared.records import Cost, Event, Usage


def call_event(idx=0, model="anthropic/claude-opus-5", **usage):
    return Event(
        idx=idx,
        type="model_call",
        cost=Cost(provider=model.split("/")[0], model=model, usage=Usage(**usage), wall_ms=120.0),
    )


# --- the price table ---


def test_price_table_entries_have_the_four_prices():
    assert budget.PRICES
    for name, price in budget.PRICES.items():
        assert set(price) == {"input", "output", "cache_read", "cache_write"}, name
        assert all(value >= 0 for value in price.values()), name
        assert price["input"] > 0 and price["output"] > price["input"], name
        assert price["cache_read"] < price["input"], name


def test_the_anthropic_rows_carry_the_published_numbers():
    """Values, not shape: a typo here bills every call wrong and nothing else would catch it."""
    assert budget.PRICES["anthropic/claude-opus-5"] == {
        "input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25
    }
    assert budget.PRICES["anthropic/claude-sonnet-5"]["input"] == 2.0
    assert budget.PRICES["anthropic/claude-haiku-4-5"]["output"] == 5.0
    for name in ("anthropic/claude-opus-5", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5"):
        price = budget.PRICES[name]
        # Anthropic's published ratios: a cache read is a tenth of input, a write 1.25 times it.
        assert price["cache_read"] == pytest.approx(price["input"] / 10), name
        assert price["cache_write"] == pytest.approx(price["input"] * 1.25), name


def test_price_table_says_when_it_was_last_checked():
    """The date is what tells a reader whether to trust a build's cost, so it has to be a date."""
    import datetime

    checked = datetime.date.fromisoformat(budget.PRICES_CHECKED)
    assert checked <= datetime.date.today()
    assert budget.PRICES_CHECKED in budget.PRICES_NOTE
    assert "update" in budget.PRICES_NOTE.lower()


def test_price_lookup_takes_the_full_id_or_the_wire_id():
    assert budget.price_for("anthropic/claude-opus-5") == budget.price_for("claude-opus-5")
    assert budget.price_for("openai/gpt-does-not-exist") is None


# --- per call cost ---


def test_cost_of_one_million_input_tokens():
    price = budget.PRICES["anthropic/claude-opus-5"]
    usd = budget.call_cost(Usage(input=1_000_000), "anthropic/claude-opus-5")
    assert usd == pytest.approx(price["input"])


def test_each_token_count_is_billed_once_at_its_own_rate():
    """Usage.input is uncached input already (Anthropic reports it that way and the OpenAI
    adapter subtracts down to it), so subtracting the cached counts again under-billed."""
    price = budget.PRICES["anthropic/claude-opus-5"]
    usage = Usage(input=1_000_000, cache_read=400_000, cache_write=100_000)
    expected = price["input"] + 0.4 * price["cache_read"] + 0.1 * price["cache_write"]
    assert budget.call_cost(usage, "anthropic/claude-opus-5") == pytest.approx(expected)


def test_a_mostly_cached_call_still_bills_the_uncached_input():
    """The provider fixture's own numbers: 10 uncached, 7 read from cache, 2 written."""
    price = budget.PRICES["anthropic/claude-opus-5"]
    usd = budget.call_cost(Usage(input=10, cache_read=7, cache_write=2), "anthropic/claude-opus-5")
    expected = (10 * price["input"] + 7 * price["cache_read"] + 2 * price["cache_write"]) / 1e6
    assert usd == pytest.approx(expected)
    assert usd > budget.call_cost(Usage(cache_read=7, cache_write=2), "anthropic/claude-opus-5")


def test_an_unpriced_model_costs_zero_and_is_counted_as_unpriced():
    assert budget.call_cost(Usage(input=10), "openai/mystery") == 0.0
    assert budget.is_priced("openai/mystery") is False


# --- record_call and the totals file ---


def test_record_call_fills_the_cost_on_the_event(workdir):
    event = call_event(input=1_000_000, output=0)
    out = budget.record_call(event, stage="mine", workdir=workdir)
    assert out.cost.usd == pytest.approx(budget.PRICES["anthropic/claude-opus-5"]["input"])


def test_record_call_writes_per_stage_and_per_build_totals(workdir):
    budget.record_call(call_event(input=1_000_000), stage="mine", workdir=workdir)
    budget.record_call(call_event(idx=1, input=1_000_000), stage="mine", workdir=workdir)
    budget.record_call(call_event(idx=2, output=1_000_000), stage="compile_tools", workdir=workdir)

    totals = json.loads((workdir / budget.TOTALS_NAME).read_text(encoding="utf-8"))
    assert totals["stages"]["mine"]["calls"] == 2
    assert totals["stages"]["mine"]["input"] == 2_000_000
    assert totals["stages"]["compile_tools"]["output"] == 1_000_000
    assert totals["total"]["calls"] == 3
    assert totals["total"]["usd"] == pytest.approx(
        totals["stages"]["mine"]["usd"] + totals["stages"]["compile_tools"]["usd"]
    )


def test_totals_record_wall_time_and_unpriced_calls(workdir):
    budget.record_call(call_event(model="openai/mystery", input=100), stage="mine", workdir=workdir)
    totals = budget.load_totals(workdir)
    assert totals["stages"]["mine"]["unpriced_calls"] == 1
    assert totals["stages"]["mine"]["wall_ms"] == pytest.approx(120.0)


def test_record_call_leaves_an_event_without_usage_alone(workdir):
    event = Event(idx=0, type="stop")
    out = budget.record_call(event, stage="mine", workdir=workdir)
    assert out.cost is None
    assert budget.load_totals(workdir)["total"]["calls"] == 0


# --- the 40 percent context cap (D65) ---


def test_context_cap_allows_a_call_under_forty_percent():
    budget.check_context_cap(39_000, window=100_000)


def test_context_cap_allows_a_call_at_exactly_forty_percent():
    budget.check_context_cap(40_000, window=100_000)


def test_context_cap_refuses_a_call_over_forty_percent():
    with pytest.raises(budget.ContextCapExceeded) as excinfo:
        budget.check_context_cap(40_001, window=100_000)
    message = str(excinfo.value)
    assert "40001" in message and "40000" in message


def test_context_cap_fraction_is_configurable():
    budget.check_context_cap(500, window=1_000, fraction=0.5)
    with pytest.raises(budget.ContextCapExceeded):
        budget.check_context_cap(501, window=1_000, fraction=0.5)


def test_context_cap_tokens():
    assert budget.context_cap_tokens(200_000) == 80_000


# --- the spend ceiling (D86) ---


def test_ceiling_lets_spending_through_under_the_limit():
    ceiling = budget.Ceiling(usd=1.0)
    assert ceiling.add(0.25, stage="compile_tools", item="get_order") == pytest.approx(0.75)
    assert ceiling.spent == pytest.approx(0.25)
    assert ceiling.remaining == pytest.approx(0.75)
    assert ceiling.add(0.25, stage="compile_tools", item="get_user") == pytest.approx(0.5)
    assert ceiling.stage_spend["compile_tools"] == pytest.approx(0.5)
    assert ceiling.stage_charges["compile_tools"] == 2


def test_ceiling_raises_with_stage_item_spent_and_estimate():
    ceiling = budget.Ceiling(usd=1.0)
    ceiling.add(0.4, stage="compile_tools", item="tool_a", items_left=3)
    with pytest.raises(budget.BudgetExceeded) as excinfo:
        ceiling.add(0.7, stage="compile_tools", item="tool_b", items_left=2)
    error = excinfo.value
    assert error.stage == "compile_tools"
    assert error.item == "tool_b"
    assert error.spent == pytest.approx(1.1)
    assert error.ceiling_usd == pytest.approx(1.0)
    assert error.estimate_to_finish == pytest.approx(1.1)
    assert "compile_tools" in str(error) and "tool_b" in str(error)


def test_ceiling_charges_the_call_before_it_stops():
    ceiling = budget.Ceiling(usd=0.5)
    with pytest.raises(budget.BudgetExceeded):
        ceiling.add(0.6, stage="mine", item="t")
    assert ceiling.spent == pytest.approx(0.6)
    assert ceiling.remaining == 0.0


def test_ceiling_estimate_is_zero_with_nothing_left():
    ceiling = budget.Ceiling(usd=0.1)
    with pytest.raises(budget.BudgetExceeded) as excinfo:
        ceiling.add(0.2, stage="mine", item="t", items_left=0)
    assert excinfo.value.estimate_to_finish == 0.0


def test_ceiling_resumes_from_a_totals_file(workdir):
    budget.record_call(call_event(input=1_000_000), stage="mine", workdir=workdir)
    ceiling = budget.Ceiling.from_totals(workdir, usd=100.0)
    assert ceiling.spent == pytest.approx(budget.PRICES["anthropic/claude-opus-5"]["input"])


def test_ceiling_report_names_what_was_done_and_what_remains():
    ceiling = budget.Ceiling(usd=1.0)
    ceiling.add(0.4, stage="mine", item="a", items_left=1)
    report = ceiling.report(stage="mine", item="b", items_left=1)
    assert report["spent"] == pytest.approx(0.4)
    assert report["ceiling_usd"] == pytest.approx(1.0)
    assert report["stage"] == "mine"
    assert report["item"] == "b"
    assert report["estimate_to_finish"] == pytest.approx(0.4)


def test_budget_does_not_import_the_builder():
    source = (budget.__file__ or "")
    assert source.endswith("budget.py")
    text = open(source, encoding="utf-8").read()
    assert "harness.builder" not in text


# --- an unpriced model, and the price table's coverage ---


def test_the_first_slices_candidate_models_are_priced():
    """Design section 11 step 6 runs gpt-4.1-mini and o4-mini; unpriced, their Runs cost 0.00."""
    for model_id in ("openai/gpt-4.1-mini", "openai/o4-mini"):
        price = budget.price_for(model_id)
        assert price is not None, model_id
        assert price["input"] > 0 and price["output"] > 0


def test_a_ceilinged_build_refuses_a_model_it_cannot_price():
    """Unpriced calls cost 0.00, so a ceiling over them can never be reached: refuse instead."""
    ceiling = budget.Ceiling(usd=1.0)
    with pytest.raises(budget.UnpricedModel) as excinfo:
        ceiling.require_priced("openai/mystery")
    assert "mystery" in str(excinfo.value)
    ceiling.require_priced("anthropic/claude-opus-5")


def test_a_budgeted_model_on_an_unpriced_model_refuses_before_the_first_call(workdir, make_test_model):
    inner = make_test_model(["hi"], name="openai/mystery")
    with pytest.raises(budget.UnpricedModel):
        budget.BudgetedModel(inner, stage="runs", workdir=workdir, ceiling=budget.Ceiling(usd=0.01))
    assert inner.calls == []


# --- one ledger, and a resume that cannot pay twice ---


def test_a_reached_ceiling_cannot_be_resumed_without_a_new_one(workdir):
    budget.record_call(call_event(input=1_000_000), stage="mine", workdir=workdir)
    spent = budget.load_totals(workdir)["total"]["usd"]
    with pytest.raises(budget.BudgetExceeded):
        budget.Ceiling.from_totals(workdir, usd=spent)
    resumed = budget.Ceiling.from_totals(workdir, usd=spent + 1.0)
    assert resumed.spent == pytest.approx(spent)


def test_nothing_is_charged_past_a_ceiling_that_is_already_reached():
    ceiling = budget.Ceiling(usd=0.5)
    with pytest.raises(budget.BudgetExceeded):
        ceiling.add(0.6, stage="mine", item="a")
    assert ceiling.spent == pytest.approx(0.6)
    with pytest.raises(budget.BudgetExceeded):
        ceiling.add(0.4, stage="mine", item="b")
    assert ceiling.spent == pytest.approx(0.6), "a second item was paid for past the ceiling"


def test_ceiling_spend_lands_in_the_same_file_a_resume_reads(workdir):
    ceiling = budget.Ceiling(usd=1.0, workdir=workdir)
    ceiling.add(0.9, stage="mine", item="a")
    assert budget.Ceiling.from_totals(workdir, usd=2.0).spent == pytest.approx(0.9)


def test_record_call_charges_the_ceiling_from_the_ledger_without_counting_twice(workdir):
    ceiling = budget.Ceiling(usd=100.0, workdir=workdir)
    budget.record_call(call_event(input=1_000_000), stage="mine", workdir=workdir, ceiling=ceiling)
    expected = budget.PRICES["anthropic/claude-opus-5"]["input"]
    assert ceiling.spent == pytest.approx(expected)
    assert budget.load_totals(workdir)["total"]["usd"] == pytest.approx(expected)


def test_charge_recorded_refreshes_spent_before_refusing(workdir):
    """Once the ceiling has already been breached, a further charge still lands in the ledger
    (say, from a caller that does not check before charging); spent must reflect it rather than
    freezing at the value from the call that first breached the ceiling."""
    ceiling = budget.Ceiling(usd=12.0, workdir=workdir)
    for idx in range(2):
        budget.record_call(call_event(idx=idx, input=1_000_000), stage="mine", workdir=workdir, ceiling=ceiling)
    with pytest.raises(budget.BudgetExceeded):
        budget.record_call(call_event(idx=2, input=1_000_000), stage="mine", workdir=workdir, ceiling=ceiling)
    assert ceiling.spent == pytest.approx(15.0)
    with pytest.raises(budget.BudgetExceeded):
        budget.record_call(call_event(idx=3, input=1_000_000), stage="mine", workdir=workdir, ceiling=ceiling)
    assert ceiling.spent == pytest.approx(20.0), "spent must reflect the ledger even when charge_recorded refuses"
    assert budget.load_totals(workdir)["total"]["usd"] == pytest.approx(20.0)


def test_record_call_stops_the_build_when_the_ledger_passes_the_ceiling(workdir):
    ceiling = budget.Ceiling(usd=1.0, workdir=workdir)
    with pytest.raises(budget.BudgetExceeded) as excinfo:
        budget.record_call(call_event(input=1_000_000), stage="mine", workdir=workdir, ceiling=ceiling)
    assert excinfo.value.stage == "mine"
    assert budget.load_totals(workdir)["total"]["usd"] > 1.0, "the call that was made is still recorded"


def test_a_totals_file_written_before_a_field_existed_still_resumes(workdir):
    """An older budget.json has no unpriced_calls; record_call must not raise a KeyError on it."""
    old = {"calls": 1, "input": 1, "output": 0, "cache_read": 0, "cache_write": 0, "usd": 0.1, "wall_ms": 1}
    (workdir / budget.TOTALS_NAME).write_text(
        json.dumps({"stages": {"mine": dict(old)}, "total": dict(old)}), encoding="utf-8"
    )
    budget.record_call(call_event(model="openai/mystery", input=1), stage="mine", workdir=workdir)
    totals = budget.load_totals(workdir)
    assert totals["stages"]["mine"]["unpriced_calls"] == 1
    assert totals["total"]["calls"] == 2


# --- the seam: every call priced, recorded and capped (D65, D86) ---


def test_every_call_through_the_wrapper_is_priced_and_written_to_the_ledger(workdir, make_test_model):
    from harness.shared.provider import ModelReply

    inner = make_test_model(
        [ModelReply(content="ok", model="claude-opus-5", usage={"input": 1_000_000, "output": 0})]
    )
    model = budget.BudgetedModel(inner, stage="mine", workdir=workdir, model_id="anthropic/claude-opus-5")
    assert model.query([{"role": "user", "content": "hi"}]).content == "ok"

    totals = budget.load_totals(workdir)
    assert totals["stages"]["mine"]["calls"] == 1
    assert totals["stages"]["mine"]["input"] == 1_000_000
    assert totals["total"]["usd"] == pytest.approx(budget.PRICES["anthropic/claude-opus-5"]["input"])
    assert (workdir / budget.TOTALS_NAME).is_file()


def test_the_wrapper_charges_the_ceiling_and_stops_the_build(workdir, make_test_model):
    from harness.shared.provider import ModelReply

    inner = make_test_model(
        [ModelReply(model="claude-opus-5", usage={"input": 1_000_000})], loop=True
    )
    ceiling = budget.Ceiling(usd=12.0, workdir=workdir)
    model = budget.BudgetedModel(
        inner, stage="mine", workdir=workdir, model_id="anthropic/claude-opus-5", ceiling=ceiling
    )
    model.query([{"role": "user", "content": "hi"}])
    model.query([{"role": "user", "content": "hi"}])
    with pytest.raises(budget.BudgetExceeded):
        model.query([{"role": "user", "content": "hi"}])
    assert ceiling.spent == pytest.approx(15.0)


def test_the_wrapper_refuses_before_the_live_call_once_the_ceiling_is_reached(workdir, make_test_model):
    """Once the ceiling is breached, the live model must not be called again: the refusal has to
    come before self.inner.query, not only from the after-the-fact record_call check."""
    from harness.shared.provider import ModelReply

    inner = make_test_model(
        [ModelReply(model="claude-opus-5", usage={"input": 1_000_000})], loop=True
    )
    ceiling = budget.Ceiling(usd=12.0, workdir=workdir)
    model = budget.BudgetedModel(
        inner, stage="mine", workdir=workdir, model_id="anthropic/claude-opus-5", ceiling=ceiling
    )
    model.query([{"role": "user", "content": "hi"}])
    model.query([{"role": "user", "content": "hi"}])
    with pytest.raises(budget.BudgetExceeded):
        model.query([{"role": "user", "content": "hi"}])  # this one breaches the ceiling
    calls_after_breach = len(inner.calls)
    with pytest.raises(budget.BudgetExceeded):
        model.query([{"role": "user", "content": "hi"}])
    assert len(inner.calls) == calls_after_breach, "the live model ran again after the ceiling was reached"


def test_the_wrapper_refuses_a_builder_prompt_over_the_cap_before_calling(workdir, make_test_model):
    """D65: the call is refused, not compacted, and the model never sees it."""
    inner = make_test_model(["never reached"], loop=True)
    model = budget.BudgetedModel(
        inner, stage="mine", workdir=workdir, model_id="anthropic/claude-haiku-4-5"
    )
    cap_chars = budget.context_cap_tokens(200_000) * budget.CHARS_PER_TOKEN
    with pytest.raises(budget.ContextCapExceeded):
        model.query([{"role": "user", "content": "x" * (cap_chars + 100)}])
    assert inner.calls == []
    assert budget.load_totals(workdir)["total"]["calls"] == 0
    # A prompt under the cap goes through.
    model.query([{"role": "user", "content": "small"}])
    assert len(inner.calls) == 1


def test_the_candidate_is_not_capped(workdir, make_test_model):
    """D65: the Candidate is tested under the production setting; only Builder calls are capped."""
    inner = make_test_model(["ok"], loop=True)
    model = budget.BudgetedModel(
        inner, stage="runs", workdir=workdir, model_id="anthropic/claude-haiku-4-5", cap_context=False
    )
    big = "x" * (budget.context_cap_tokens(200_000) * budget.CHARS_PER_TOKEN + 100)
    assert model.query([{"role": "user", "content": big}]).content == "ok"


def test_a_memomodel_hit_is_counted_in_the_memo_hits_bucket_and_priced_at_zero(workdir):
    """docs/prompt-caching.md item 3: a hit still counts as a call, but adds to memo_hits and
    prices at zero because MemoModel already zeroed its usage."""
    from harness.shared.provider import MemoModel, ModelReply, TestModel

    inner = TestModel([ModelReply(content="ok", model="claude-opus-5", usage={"input": 1_000_000})])
    memo = MemoModel(inner, workdir)
    model = budget.BudgetedModel(memo, stage="mine", workdir=workdir, model_id="anthropic/claude-opus-5")

    model.query([{"role": "user", "content": "hi"}])
    model.query([{"role": "user", "content": "hi"}])  # same request: a memo hit

    totals = budget.load_totals(workdir)
    assert totals["stages"]["mine"]["calls"] == 2
    assert totals["stages"]["mine"]["memo_hits"] == 1
    assert totals["total"]["usd"] == pytest.approx(budget.PRICES["anthropic/claude-opus-5"]["input"])


def test_a_plain_model_never_adds_to_memo_hits(workdir, make_test_model):
    inner = make_test_model(["ok"], loop=True)
    model = budget.BudgetedModel(inner, stage="mine", workdir=workdir, model_id="anthropic/claude-opus-5")
    model.query([{"role": "user", "content": "hi"}])
    assert budget.load_totals(workdir)["stages"]["mine"]["memo_hits"] == 0


def test_the_wrapper_sets_the_prompt_cache_key_when_the_caller_left_it_unset(workdir, make_test_model):
    inner = make_test_model(["ok"])
    model = budget.BudgetedModel(
        inner, stage="mine", workdir=workdir, model_id="openai/gpt-4o-mini", prompt_cache_key="kullback-abc-mine"
    )
    model.query([{"role": "user", "content": "hi"}])
    assert inner.calls[0]["config"].prompt_cache_key == "kullback-abc-mine"


def test_the_wrapper_does_not_override_a_prompt_cache_key_the_caller_already_set(workdir, make_test_model):
    from harness.shared.provider import ModelConfig

    inner = make_test_model(["ok"])
    model = budget.BudgetedModel(
        inner, stage="mine", workdir=workdir, model_id="openai/gpt-4o-mini", prompt_cache_key="kullback-abc-mine"
    )
    model.query([{"role": "user", "content": "hi"}], config=ModelConfig(prompt_cache_key="caller-set"))
    assert inner.calls[0]["config"].prompt_cache_key == "caller-set"


def test_token_estimate_counts_the_tools_as_well_as_the_messages():
    messages = [{"role": "user", "content": "a" * 400}]
    assert budget.estimate_tokens(messages) >= 100
    assert budget.estimate_tokens(messages, [{"name": "get_order"}]) > budget.estimate_tokens(messages)
    assert budget.estimate_tokens(None) == 0


def test_the_window_comes_from_the_model_id():
    assert budget.window_for("anthropic/claude-haiku-4-5") == 200_000
    assert budget.window_for("claude-opus-5") == 1_000_000
    assert budget.window_for("something/unknown") == budget.DEFAULT_CONTEXT_WINDOW
