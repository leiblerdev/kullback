"""One content hash of the code that executes and scores a Run, read off the files' bytes at import.

The hash covers every `.py` file under the packages named in SCORING_PACKAGES (the Runner, the
gates and the Spec), walked and sorted by path, so a file moving in or out of a package moves it
the same way a changed byte inside one does. It replaces the freeze: a fix to any of these
packages lands as an ordinary commit, and every Run, Verdict and tier row carries the hash of
the code that made it, so two numbers are comparable exactly when their hashes match.

This module reads files and imports nothing of the packages it hashes, so the Runner's import
boundary (runner/boundary.py) holds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Union

# Package roots under kullback/, hashed whole; no file inside them is named here.
SCORING_PACKAGES = ("runner", "gates", "spec")
# How many hex characters a report prints; the stored hash is always whole.
SHORT = 12


def code_hash(root: Union[str, Path, None] = None, packages: Iterable[str] = SCORING_PACKAGES) -> str:
    """sha256 over the relative path and bytes of every .py file under each package of `root`.

    `root` is the directory that holds the packages (the installed `kullback/` by default).
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in sorted(packages):
        directory = base / package
        for path in sorted(directory.rglob("*.py")) if directory.is_dir() else ():
            digest.update(path.relative_to(base).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


# The hash of the code this process imported, taken once when the module loads.
CODE_HASH = code_hash()
