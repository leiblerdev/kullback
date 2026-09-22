"""The JSON-like aliases tool arguments, tool results and event payloads are written in.

tau_agent/types.py, with `Union` spellings rather than PEP 695 `type` statements because this
package still supports Python 3.11 (pyproject's requires-python).
"""

from __future__ import annotations

from typing import Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, list["JSONValue"], dict[str, "JSONValue"]]
JSONObject = dict[str, JSONValue]

__all__ = ["JSONObject", "JSONPrimitive", "JSONValue"]
