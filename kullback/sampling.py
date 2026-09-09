"""One keyed draw for every sample the harness takes per Task or per Run (D212).

A sample drawn from a count or a position moves when the count moves. Two builds of one
Environment rarely hold the same number of Runs: Tasks drift, re-rolls are discarded and made
again, a round is resumed part way. When the draw is `the first n in list order` or
`a counter that advances per attempt`, adding one Task or one Run reshuffles the draws of the
Tasks around it, and the delta between the two builds mixes the code change with the sampling
change. The loop reads those deltas every day, so the sampling has to hold still.

Every draw here is a function of three things: the kind of draw, the id it is drawn for (a Task
id, a Run id, a Task id with the attempt index appended) and the build salt. Nothing about the
size of the list, the position of the id in it or how many draws came before is an input, so a
draw for one id is unchanged by every other id, and the same id under the same salt draws the
same value on every invocation and on every resume.

The salt is stored in the workdir the first time anything draws, and read back after that, so a
resumed build draws what its first launch drew. It defaults to one constant, which is what makes
two workdirs of the same Environment comparable; a caller that wants a second, independent draw
over the same ids sets `KULLBACK_SAMPLE_SALT` before the workdir is made.

`draws_by_kind` and `draws_since` hand the round record what was drawn, by kind, so a round says
how much sampling it did beside what it derived.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

SALT_NAME = "sample_salt.json"
SALT_ENV = "KULLBACK_SAMPLE_SALT"
# The default is a constant and not a clock, a path or a run id: two workdirs built from the same
# Environment draw the same samples unless someone asks for otherwise, which is the whole point.
DEFAULT_SALT = "kullback-sample-salt-1"
KEY_BITS = 64
KEY_SPACE = 1 << KEY_BITS
SEED_SPACE = 1 << 31  # what a Run record carries as its seed: a non-negative 32-bit int

_LOCK = threading.Lock()
_DRAWS: Counter = Counter()


def sample_key(kind: str, ident: Any, salt: str) -> int:
    """A stable 64-bit value for one draw, from the kind, the id and the build salt.

    The three are hashed as a JSON list, so no separator can be smuggled out of an id: a Task
    called `a` with attempt `1:2` and a Task called `a:1` with attempt `2` are different draws.
    """
    body = json.dumps([str(kind), str(ident), str(salt)], separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).digest()
    with _LOCK:
        _DRAWS[str(kind)] += 1
    return int.from_bytes(digest[:8], "big")


def sample_unit(kind: str, ident: Any, salt: str) -> float:
    """The same draw as a number in [0, 1), which is what a share is a threshold on."""
    return sample_key(kind, ident, salt) / KEY_SPACE


def sample_seed(kind: str, ident: Any, salt: str) -> int:
    """The same draw as a non-negative 32-bit seed, for a record that carries one.

    A Run discarded and made again under its own id draws this seed again, so the re-made Run is
    the one it replaced rather than the next number off a counter.
    """
    return sample_key(kind, ident, salt) % SEED_SPACE


def keyed_order(kind: str, idents: Iterable[Any], salt: str) -> list[str]:
    """The ids ordered by their key, ties broken by the id itself.

    Where a bounded budget has to be handed out and there is not enough for everyone, this is the
    order it goes out in: the position an id held in the caller's list is not an input, so an id
    added to the list does not push the ids after it out of the budget, it takes the place its own
    key earns.
    """
    return [ident for _, ident in _ranked(kind, idents, salt)]


def _ranked(kind: str, idents: Iterable[Any], salt: str) -> list[tuple[int, str]]:
    """(key, id) for every id, lowest key first, ties broken by the id; one draw per id."""
    keys = {str(i): sample_key(kind, str(i), salt) for i in idents}
    return sorted(((key, ident) for ident, key in keys.items()), key=lambda pair: (pair[0], pair[1]))


def keyed_share(kind: str, idents: Iterable[Any], salt: str, share: float,
                minimum: int = 0, maximum: Optional[int] = None) -> list[str]:
    """The ids whose key falls under `share`, held between `minimum` and `maximum` by key order.

    Membership is a threshold on each id's own key and not a position in a shuffled list, so an id
    added to or taken out of the list leaves every other id's membership alone. The floor and the
    ceiling are the caller's guarantees, not a draw: they are filled and trimmed from the lowest
    keys, which is the same order the threshold itself picks in. A caller that owes an exact count
    passes it as both, and gets the count's worth of lowest keys, which is still a function of the
    ids alone.
    """
    ranked = _ranked(kind, idents, salt)
    chosen = [ident for key, ident in ranked if key / KEY_SPACE < share]
    if maximum is not None and len(chosen) > maximum:
        chosen = chosen[:max(0, maximum)]
    if len(chosen) < minimum:
        picked = set(chosen)
        chosen = chosen + [ident for _, ident in ranked if ident not in picked][:minimum - len(chosen)]
    return sorted(chosen)


# --- the build salt ---------------------------------------------------------

def salt_path(workdir: Any) -> Path:
    return Path(workdir) / SALT_NAME


def read_salt(workdir: Any) -> Optional[str]:
    """The salt this workdir already stored, or None when nothing has drawn in it yet."""
    try:
        body = json.loads(salt_path(workdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    salt = body.get("salt") if isinstance(body, dict) else None
    return str(salt) if salt else None


def build_salt(workdir: Any) -> str:
    """This build's salt: the stored one, else the asked-for one written down now.

    Written once and read back after that, so a resumed build and a build re-entered with
    `--iterate` draw what the first launch drew. A workdir copied somewhere else carries its salt
    with it, which is what lets a measurement on a copy answer for the original.
    """
    stored = read_salt(workdir)
    if stored is not None:
        return stored
    salt = os.environ.get(SALT_ENV) or DEFAULT_SALT
    path = salt_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"salt": salt}, indent=2, sort_keys=True), encoding="utf-8")
    return salt


def salt_label(salt: Optional[str]) -> str:
    """The eight characters of a salt's digest a round record prints, so two builds can be told apart."""
    if not salt:
        return ""
    return hashlib.sha256(str(salt).encode("utf-8")).hexdigest()[:8]


# --- what a round drew ------------------------------------------------------

def draws_by_kind() -> dict[str, int]:
    """How many draws of each kind this process has taken, cumulative and never reset here.

    Reading is not taking: a caller that wants one round's share of it keeps the totals it saw at
    the round's start and asks `draws_since` for the difference. A destructive read would empty the
    counter for whoever asked second, and a round's counts are assembled more than once.
    """
    with _LOCK:
        return {kind: int(count) for kind, count in sorted(_DRAWS.items())}


def draws_since(baseline: Optional[dict[str, int]]) -> dict[str, int]:
    """The draws taken since a `draws_by_kind` snapshot, kinds with nothing new left out."""
    was = baseline or {}
    return {kind: count - int(was.get(kind, 0)) for kind, count in draws_by_kind().items()
            if count - int(was.get(kind, 0)) > 0}


__all__ = ["DEFAULT_SALT", "KEY_BITS", "KEY_SPACE", "SALT_ENV", "SALT_NAME", "SEED_SPACE",
           "build_salt", "draws_by_kind", "draws_since", "keyed_order", "keyed_share", "read_salt",
           "salt_label", "salt_path", "sample_key", "sample_seed", "sample_unit"]
