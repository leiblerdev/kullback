"""Tool call ids a provider will accept, without letting two distinct ids become one.

Mirrors tau_ai/tool_call_ids.py; the rule is provider.py's, moved here verbatim.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

ID_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")
ID_DIGEST_LEN = 8


def clean_tool_call_id(call_id: Any) -> str:
    """Tool call ids are restricted to [a-zA-Z0-9_-]; other characters become underscores.

    Replacing characters can collide: 'a:b' and 'a_b' both cleaned to 'a_b', and a tool_result
    could then be paired with the wrong tool_use in the same message. When anything was
    replaced, a short digest of the original id is appended so distinct ids stay distinct.
    """
    original = str(call_id or "")
    cleaned = ID_ALLOWED.sub("_", original)
    if not cleaned:
        return "id"
    if cleaned == original:
        return cleaned
    return f"{cleaned}_{hashlib.sha256(original.encode('utf-8')).hexdigest()[:ID_DIGEST_LEN]}"
