"""The catalogue decides the wire shape: a row override means Responses, nothing else does."""

from __future__ import annotations

import json

from kullback.ai import provider as pv


def snapshot(tmp_path, monkeypatch, catalog):
    """A models.dev snapshot on disk, and the provider module pointed at it."""
    path = tmp_path / "models.dev.json"
    path.write_text(json.dumps({"fetched_at": "2026-09-01T00:00:00+00:00", "catalog": catalog}),
                    encoding="utf-8")
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(path))
    return path


CATALOG = {
    "gw": {"id": "gw", "name": "Gateway", "npm": "@ai-sdk/openai-compatible",
           "api": "https://gw.invalid/v1", "env": ["GW_API_KEY"],
           "models": {"a": {"provider": {"npm": "@ai-sdk/openai"}},
                      "b": {}}},
}


def test_a_row_marked_responses_resolves_to_the_responses_adapter(tmp_path, monkeypatch):
    """The row's own npm override is the switch: no name in code takes part."""
    snapshot(tmp_path, monkeypatch, CATALOG)
    model = pv.model_for("gw/a", env={"GW_API_KEY": "sk-gw"})
    assert isinstance(model, pv.OpenAIResponsesModel)
    assert model.base_url == "https://gw.invalid/v1"
    assert model.wire_id == "a"
    assert model.api_key == "sk-gw"


def test_a_responses_row_keeps_its_shape_on_an_explicit_endpoint(tmp_path, monkeypatch):
    """An explicit endpoint never changes the wire shape; the row still decides."""
    snapshot(tmp_path, monkeypatch, CATALOG)
    model = pv.model_for("gw/a", base_url="http://127.0.0.1:8080/v1", env={})
    assert isinstance(model, pv.OpenAIResponsesModel)
    assert model.base_url == "http://127.0.0.1:8080/v1"


def test_a_row_without_an_override_stays_on_chat(tmp_path, monkeypatch):
    """No row override is the chat shape, through the registry and on an explicit endpoint."""
    snapshot(tmp_path, monkeypatch, CATALOG)
    assert isinstance(pv.model_for("gw/b", env={}), pv.RegistryModel)
    explicit = pv.model_for("gw/b", base_url="http://127.0.0.1:8080/v1", env={})
    assert type(explicit) is pv.OpenAICompatibleModel


def test_a_provider_level_npm_alone_does_not_switch_shape(tmp_path, monkeypatch):
    """The same npm string on the provider entry still means chat; only the row switches."""
    catalog = {"gw": {"id": "gw", "name": "Gateway", "npm": "@ai-sdk/openai",
                      "api": "https://gw.invalid/v1", "env": ["GW_API_KEY"],
                      "models": {"b": {}}}}
    snapshot(tmp_path, monkeypatch, catalog)
    assert isinstance(pv.model_for("gw/b", env={}), pv.RegistryModel)


def test_an_unreadable_catalogue_reads_as_chat(tmp_path, monkeypatch):
    """No snapshot on disk is chat on an explicit endpoint, never a refusal."""
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(tmp_path / "absent.json"))
    model = pv.model_for("gw/a", base_url="http://127.0.0.1:8080/v1", env={})
    assert type(model) is pv.OpenAICompatibleModel
