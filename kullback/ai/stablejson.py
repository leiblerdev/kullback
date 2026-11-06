"""Deterministic JSON text for model prompts: sorted keys, fixed separators, so equal content is byte-identical."""

import json
from typing import Any


def dumps(value: Any, indent: int | None = None) -> str:
    """Serialise `value` the same way every time, so calls that share content share bytes and the prefix cache can hit."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=indent,
                      separators=(",", ":") if indent is None else (",", ": "), default=str)
