"""Detector for the OpenTelemetry GenAI shape: named here so the mapper has a home later."""

from __future__ import annotations

from typing import Any, Iterator


class OtelGenaiDetector:
    """Votes on the GenAI span shape; mapping it is still on todo."""

    name = "otel_genai"
    maps = False
    display = "OpenTelemetry GenAI"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            heads = [item for item in document[:20] if isinstance(item, dict)]
            if any(_looks_otel(item) for item in heads):
                return (0.9, ["list records carry GenAI span names or attributes"])
            return (0.0, ["no list record carries a GenAI span name or attribute"])
        if isinstance(document, dict):
            if "resourceSpans" in document or "resource_spans" in document:
                return (0.9, ["has a resource spans block"])
            if _looks_otel(document):
                return (0.9, ["carries GenAI span names or attributes"])
            return (0.0, ["no resource spans block and no GenAI span marker"])
        return (0.0, ["not a parsed object, so no shape to read"])

    def recordings(self, document: Any) -> Iterator[Any]:
        raise NotImplementedError(self.unmapped_message())

    def to_trace(self, recording: Any, ctx: Any) -> Any:
        raise NotImplementedError(self.unmapped_message())

    def environment(self, document: Any) -> dict:
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        return {}

    def unmapped_message(self) -> str:
        return (f"{self.display} ingest ({self.name}) is not written yet; only one export "
                "is mapped so far. format_detect names the format so the mapper has a home here.")


def _looks_otel(item: dict) -> bool:
    if str(item.get("name", "")).startswith("gen_ai."):
        return True
    attributes = item.get("attributes")
    return isinstance(attributes, dict) and any(str(k).startswith("gen_ai.") for k in attributes)
