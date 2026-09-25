"""The shipped model tables by their real names: prices, windows, rules and shapes.

Unit tests elsewhere use neutral models registered into these tables, so nothing there can go
stale when a vendor ships a new name. This tier pins the real names instead: it iterates the
shipped tables (never a hand-kept list) and asserts the behaviour the units used to pin. Fast
and offline: no keys, no network, no snapshot needed.
"""

from __future__ import annotations

import pytest

from kullback.ai import provider as pv
from kullback.ai.model_limits import REQUEST_RULES, request_rules_for
from kullback.runner import budget
from kullback.tui import LOGIN_DEFAULT_MODELS


@pytest.mark.smoke
def test_the_shipped_default_model_is_priced_windowed_and_has_request_rules():
    """The model the builds run on bills under the table, caps at its measured window, and
    shapes requests with its own rules row rather than the default."""
    assert budget.is_priced(pv.DEFAULT_MODEL)
    assert budget.window_for(pv.DEFAULT_MODEL) == 1_000_000
    rules = request_rules_for(pv.DEFAULT_MODEL.split("/")[1])
    assert rules is REQUEST_RULES["claude-opus-5-5"]
    assert rules.thinking_always_on and not rules.forced_tool_choice and not rules.sampling


@pytest.mark.smoke
def test_every_request_rules_row_resolves_under_every_host_prefix():
    """One rules row serves every profile of a model: the bare name, the Bedrock vendor id,
    the cross-region profiles, and the full provider id all answer with the same row."""
    for name, rules in REQUEST_RULES.items():
        assert request_rules_for(name) is rules
        assert request_rules_for(f"anthropic.{name}") is rules
        assert request_rules_for(f"us.anthropic.{name}") is rules
        assert request_rules_for(f"global.anthropic.{name}") is rules
        assert request_rules_for(f"bedrock/global.anthropic.{name}") is rules


def _chat_body(model_id: str) -> dict:
    """The body one call would send, without sending it."""
    model = pv.OpenAIModel(model_id=model_id, api_key="k", env={})
    return model.build_body([{"role": "user", "content": "hi"}], None, pv.ModelConfig(max_tokens=2000))


@pytest.mark.smoke
def test_the_shipped_reasoning_family_routes_to_the_reasoning_request_shape():
    """gpt-5 and later (including the shipped OpenAI login default) and the o-series take the
    reasoning shape; gpt-4.1 and gpt-4o keep the older shape."""
    in_shape = ["openai/gpt-5.6-luna", "openai/o1", "openai/o3-mini", "openai/o4-mini",
                LOGIN_DEFAULT_MODELS["openai"]]
    for model_id in in_shape:
        assert pv.reasoning_family(model_id.split("/", 1)[1])
        body = _chat_body(model_id)
        assert body["max_completion_tokens"] == 2000 and "max_tokens" not in body
    for model_id in ("openai/gpt-4.1-mini", "openai/gpt-4o"):
        assert not pv.reasoning_family(model_id.split("/", 1)[1])
        body = _chat_body(model_id)
        assert body["max_tokens"] == 2000 and "max_completion_tokens" not in body


@pytest.mark.smoke
def test_every_responses_table_id_resolves_to_the_responses_adapter_and_others_do_not():
    """Each id the Responses table names speaks /v1/responses wherever it lives; a chat model
    with an adapter of its own and a registry model given an explicit host do not."""
    assert pv.RESPONSES_API_MODELS, "an empty table would let this test pass while pinning nothing"
    for model_id in pv.RESPONSES_API_MODELS:
        model = pv.model_for(model_id, base_url="http://127.0.0.1:8080/v1", env={})
        assert isinstance(model, pv.OpenAIResponsesModel)
    assert type(pv.model_for("openai/gpt-6-luna", env={})) is pv.OpenAIModel
    other = pv.model_for(LOGIN_DEFAULT_MODELS["opencode-go"],
                         base_url="http://127.0.0.1:8080/v1", env={})
    assert type(other) is pv.OpenAICompatibleModel


@pytest.mark.smoke
def test_every_login_default_names_a_priced_model():
    """Each provider the /login menu starts at names a model the ledger can bill, so a first
    try of a provider never lands on an unpriced id. The openrouter default is the one id only
    the registry snapshot prices (the TUI names it the cheapest tool-capable snapshot row), so
    no offline catalog answers it: it is pinned by id here instead, and priced wherever a
    snapshot is read."""
    for provider, model in LOGIN_DEFAULT_MODELS.items():
        if provider == "openrouter":
            assert model == "openrouter/qwen/qwen3.7-flash"
        else:
            assert budget.is_priced(model), provider
