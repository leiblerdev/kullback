"""One intake seam: every trace format enters through a registered adapter.

An adapter is a small object with three jobs: vote on whether a parsed
payload is its format (positive evidence only), yield the payload's
recordings one by one, and map one recording to a Trace. ingest.py asks
every registered adapter and takes the strongest vote; the two detector
adapters below the mapper name formats no mapper reads yet. Two small
helpers ride along so ingest.py never reads a format layout directly:
environment (the export-level block a recording's tools and prompt come
from) and sidecar (the answer key that leaves the trace for its own file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol


@dataclass
class MapContext:
    """What to_trace needs beyond the recording itself: where it sits in
    the stored file, the export-level block around it, and the ingest code
    hash the Trace carries (D66)."""

    raw_hash: str
    index: int
    environment: dict
    ingest_version: str


class SourceAdapter(Protocol):
    """The three jobs plus the two export-block helpers. Detectors set
    maps to False and raise from recordings and to_trace."""

    name: str
    maps: bool
    display: str

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        """Vote with positive evidence: (confidence above zero, reasons).

        Zero means no evidence either way, never a veto; the seam turns a
        tie or an all-zero ballot into "unknown" with the reasons in words.
        """
        ...  # pragma: no cover - the protocol states the shape only

    def recordings(self, document: Any) -> Iterator[Any]:
        """Yield the payload's recordings one by one."""
        ...  # pragma: no cover - the protocol states the shape only

    def to_trace(self, recording: Any, ctx: MapContext) -> Any:
        """Map one recording to a Trace."""
        ...  # pragma: no cover - the protocol states the shape only

    def environment(self, document: Any) -> dict:
        """The export-level block around the recordings (tools, prompt)."""
        ...  # pragma: no cover - the protocol states the shape only

    def sidecar(self, recording: Any, document: Any) -> dict:
        """The answer key that leaves the trace for its own file."""
        ...  # pragma: no cover - the protocol states the shape only


@dataclass
class FormatDecision:
    """One ballot: the winning adapter name, or "unknown", with why."""

    winner: str
    votes: dict[str, tuple[float, list[str]]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


_REGISTRY: dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> SourceAdapter:
    """Add one adapter; every registered adapter votes on each payload."""
    _REGISTRY[adapter.name] = adapter
    return adapter


def unregister(name: str) -> None:
    """Remove one adapter; tests use it to retire their toy formats."""
    _REGISTRY.pop(name, None)


def registered() -> list[SourceAdapter]:
    """Every registered adapter, in registration order."""
    return list(_REGISTRY.values())


def by_name(name: str) -> Optional[SourceAdapter]:
    """The adapter behind a format name, or None for "unknown"."""
    return _REGISTRY.get(name)


def display_names() -> list[str]:
    """Every registered format's display name, for error messages."""
    return [adapter.display for adapter in _REGISTRY.values()]


def detect_format(document: Any, jsonl: bool = False) -> FormatDecision:
    """Ask every registered adapter and take the strongest positive vote.

    A tie or an all-zero ballot is "unknown" with every reason in words,
    so an ambiguous payload is refused instead of mislabelled. A payload
    that declares its own format is trusted before any guess from shape.
    """
    votes = {adapter.name: adapter.detect(document, jsonl) for adapter in _REGISTRY.values()}
    declared = _declared_name(document)
    if declared is not None and declared in _REGISTRY:
        confidence, shape_reasons = votes[declared]
        reasons = [f"the payload declares its format as {declared}, so the declaration is trusted "
                   "before any guess from shape"]
        if confidence > 0:
            reasons = reasons + [f"its shape agrees ({'; '.join(shape_reasons)})"]
        return FormatDecision(winner=declared, votes=votes, reasons=reasons)
    positives = [(name, vote[0]) for name, vote in votes.items() if vote[0] > 0]
    if not positives:
        reasons = [f"{name} found no positive evidence ({'; '.join(found) or 'no reason given'})"
                   for name, (_, found) in votes.items()]
        return FormatDecision(winner="unknown", votes=votes, reasons=reasons)
    top = max(score for _, score in positives)
    winners = [name for name, score in positives if score == top]
    if len(winners) > 1:
        tied = ", ".join(sorted(winners))
        detail = "; ".join(f"{name}: {'; '.join(votes[name][1])}" for name in sorted(winners))
        return FormatDecision(
            winner="unknown", votes=votes,
            reasons=[f"a tie for strongest vote between {tied}, so the payload is refused as ambiguous",
                     detail],
        )
    winner = winners[0]
    return FormatDecision(winner=winner, votes=votes, reasons=list(votes[winner][1]))


def _declared_name(document: Any) -> Optional[str]:
    """The format name a payload declares for itself, if it names one.

    A dict carries it under format, source_format or span_kind; a list of
    records carries it when every record with the key agrees on the value.
    Anything else (no key, disagreement, not a string) declares nothing.
    """
    if isinstance(document, dict):
        for key in ("format", "source_format", "span_kind"):
            value = document.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if isinstance(document, list):
        seen: set[str] = set()
        for item in document[:20]:
            if not isinstance(item, dict):
                continue
            for key in ("format", "source_format", "span_kind"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    seen.add(value)
        if len(seen) == 1:
            return next(iter(seen))
    return None


from kullback.builder.sources.claude_code_jsonl import ClaudeCodeJsonlDetector  # noqa: E402
from kullback.builder.sources.otel_genai import OtelGenaiDetector  # noqa: E402
from kullback.builder.sources.tau2_native import Tau2NativeAdapter  # noqa: E402

register(Tau2NativeAdapter())
register(OtelGenaiDetector())
register(ClaudeCodeJsonlDetector())
