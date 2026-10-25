"""What models an account can reach, and what each of them takes.

Mirrors tau_ai/model_catalog.py. The inventory comes from the models.dev snapshot `pricing.py`
keeps, so this module adds a shape a caller can read rather than a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from kullback.ai.model_limits import ModelLimits


@dataclass(frozen=True)
class CatalogModel:
    """One model a provider advertises."""

    id: str
    name: Optional[str] = None
    limits: Optional[ModelLimits] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a model id is never empty")


@dataclass(frozen=True)
class ModelCatalog:
    """Every model an account can reach, newest snapshot wins."""

    models: tuple[CatalogModel, ...] = field(default_factory=tuple)

    def model(self, model_id: str) -> Optional[CatalogModel]:
        """One advertised model by its exact id."""
        return next((model for model in self.models if model.id == model_id), None)


@runtime_checkable
class ModelCatalogProvider(Protocol):
    """A provider that can be asked what it serves."""

    def discover_models(self) -> ModelCatalog:
        ...


def catalog_from_snapshot(*, path: Optional[str] = None,
                          env: Optional[dict[str, str]] = None) -> ModelCatalog:
    """The snapshot's providers and models as a catalog, ids spelled 'provider/model'."""
    from kullback.ai import pricing

    catalog = pricing.refresh(path=path, env=env)
    if not isinstance(catalog, dict):
        return ModelCatalog()
    models: list[CatalogModel] = []
    for provider_id, entry in sorted(catalog.items()):
        rows = entry.get("models") if isinstance(entry, dict) else None
        if not isinstance(rows, dict):
            continue
        for wire_id, row in sorted(rows.items()):
            model_id = f"{provider_id}/{wire_id}"
            window = pricing.window_from_catalog(catalog, model_id)
            models.append(
                CatalogModel(
                    id=model_id,
                    name=(row.get("name") if isinstance(row, dict) else None),
                    limits=ModelLimits(context_window=window) if window else None,
                )
            )
    return ModelCatalog(models=tuple(models))
