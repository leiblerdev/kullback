"""Detector for the Claude Code JSONL shape: named here so the mapper has a home later."""

from __future__ import annotations

from typing import Any, Iterator


class ClaudeCodeJsonlDetector:
    """Votes on line-delimited assistant transcript rows; mapping it is still on todo."""

    name = "claude_code_jsonl"
    maps = False
    display = "Claude Code JSONL"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            if jsonl and any(_looks_claude_code(item) for item in document[:20]
                             if isinstance(item, dict)):
                return (0.8, ["line-delimited rows carry transcript fields"])
            return (0.0, ["no transcript row marker, or the file is one JSON document, not lines"])
        return (0.0, ["not a record list, so not line-delimited rows"])

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


def _looks_claude_code(item: dict) -> bool:
    if item.get("type") not in ("user", "assistant", "system", "summary", "result"):
        return False
    return any(key in item for key in ("message", "content", "uuid", "sessionId"))
