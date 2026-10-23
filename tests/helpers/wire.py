"""What a provider sends on the wire, faked: server-sent event bodies, a chat stream, a transport, a catalogue."""

from __future__ import annotations

import json

import httpx

from kullback.ai import provider as pv

CHAT_CHUNKS = [
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "Hel"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {"content": "lo"}}]}',
    '{"id": "r1", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}],'
    ' "usage": {"prompt_tokens": 12, "completion_tokens": 3}}',
    "[DONE]",
]


def sse(chunks):
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode("utf-8")


def transport_of(handler):
    """An httpx client whose every request is answered by handler; nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def registry_snapshot(tmp_path, monkeypatch, catalog):
    """A models.dev snapshot on disk, and the provider module pointed at it."""
    path = tmp_path / "models.dev.json"
    path.write_text(json.dumps({"fetched_at": "2026-09-01T00:00:00+00:00", "catalog": catalog}),
                    encoding="utf-8")
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(path))
    return path
