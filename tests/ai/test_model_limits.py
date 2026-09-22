"""The window a model takes, and the point at which the core compacts."""

from __future__ import annotations

import json

import pytest

from kullback.ai import model_catalog, model_limits
from kullback.ai.model_limits import ModelLimits


def snapshot(tmp_path, providers):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"fetched_at": "2026-09-22T00:00:00+00:00",
                                "catalog": providers}), encoding="utf-8")
    return str(path)


def test_the_core_compacts_at_two_fifths_of_the_usable_window():
    limits = ModelLimits(context_window=200_000)
    assert limits.effective_context_window == 200_000
    assert limits.compaction_token_limit == 80_000


def test_a_surface_that_keeps_headroom_shrinks_both_numbers():
    limits = ModelLimits(context_window=200_000, effective_context_window_percent=90)
    assert limits.effective_context_window == 180_000
    assert limits.compaction_token_limit == 72_000


@pytest.mark.parametrize(
    "kwargs",
    [{"context_window": 0}, {"context_window": 10, "max_output_tokens": 0},
     {"context_window": 10, "effective_context_window_percent": 0}],
)
def test_limits_that_cannot_be_true_are_refused(kwargs):
    with pytest.raises(ValueError):
        ModelLimits(**kwargs)


def test_a_model_the_snapshot_sizes_takes_the_window_it_lists(tmp_path):
    path = snapshot(tmp_path, {"p": {"models": {"m": {"limit": {"context": 500_000}}}}})
    assert model_limits.limits_for("p/m", path=path).context_window == 500_000


def test_a_model_the_snapshot_does_not_size_takes_the_generic_window(tmp_path):
    path = snapshot(tmp_path, {"p": {"models": {"m": {}}}})
    limits = model_limits.limits_for("p/m", path=path)
    assert limits.context_window == model_limits.DEFAULT_CONTEXT_WINDOW


def test_the_catalog_names_every_model_by_provider_and_wire_id(tmp_path):
    path = snapshot(tmp_path, {
        "p": {"models": {"m": {"name": "M", "limit": {"context": 1000}}}},
        "q": {"models": {"n": {}}},
    })
    catalog = model_catalog.catalog_from_snapshot(path=path)
    # The locally registered providers are laid over the snapshot, so the listing holds them too.
    listed_ids = [model.id for model in catalog.models]
    assert {"p/m", "q/n"} <= set(listed_ids)
    assert listed_ids == sorted(listed_ids)
    listed = catalog.model("p/m")
    assert listed.name == "M" and listed.limits.context_window == 1000
    assert catalog.model("q/n").limits is None
    assert catalog.model("nothing/here") is None
