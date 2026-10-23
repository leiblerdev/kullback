"""The scripted policy the trainer's tests play: it replays the invented rename turn by turn."""

from __future__ import annotations

from tests.episode.invented import SCRIPTED_RENAME


def rename_policy(messages, tools):
    """Replay the scripted rename: the turn count picks the scripted line."""
    n = sum(1 for message in messages if message.get("role") == "assistant")
    step = SCRIPTED_RENAME[min(n, len(SCRIPTED_RENAME) - 1)]
    return {"content": step.get("content"),
            "tool_calls": [dict(call) for call in step.get("tool_calls", [])]}
