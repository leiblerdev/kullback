"""Invented terminal-agent recordings, as the registry rows carry them."""

from __future__ import annotations

import json


def rescue_turns(instruction: str) -> list[dict]:
    """An invented recording whose first user turn carries the instruction between headers."""
    return [
        {"role": "user",
         "content": ("Invented preamble.\n\nTask Description:\n" + instruction
                     + "\n\nCurrent terminal state:\n\ninvented-ready")},
        {"role": "assistant", "content": json.dumps({"commands": [
            {"keystrokes": "invented-solo"}]})},
        {"role": "user", "content": "New Terminal Output:\n\ninvented solo output"},
    ]
