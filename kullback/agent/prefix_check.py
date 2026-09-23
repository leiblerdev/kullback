"""Whether one conversation's requests grow by appending, and nothing else (G24).

Providers cache a prompt up to the first changed byte, so an agent whose every request extends
the one before it reuses what it already sent. This is that property with no constant and no
provider in it: each request must be a byte-prefix extension of the one before, checked here over
the request log. What the check compares is the wire shape the provider layer sends (the system
prompt first, then the messages in order), canonicalized so two serializations of the same
request compare equal.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence


def _blob(request: Sequence[Any]) -> bytes:
    """One request as bytes: each message canonicalized, joined on a byte JSON never emits."""
    return b"\x00".join(
        json.dumps(message, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":"), default=str).encode("utf-8")
        for message in request or ())


def first_prefix_break(requests: Sequence[Sequence[Any]]) -> Optional[int]:
    """The turn where the requests stop growing by appending, or None where they never do.

    The answer names the turn: index i means request i is not a byte-prefix extension of request
    i minus one, so the head moved or history was rewritten on that turn. None means every
    request extends the previous one. A request identical to the one before counts as an
    extension, vacuously.
    """
    blobs = [_blob(request) for request in requests or ()]
    for index in range(1, len(blobs)):
        if not blobs[index].startswith(blobs[index - 1]):
            return index
    return None
