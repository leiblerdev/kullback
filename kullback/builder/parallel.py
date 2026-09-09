"""The Builder's name for the worker pool, which moved to the Runner when the Examiner needed it too.

The pool was the Builder's alone until the Examiner's derivation ran per Task on threads (D163), and
the two packages may not import each other (the layers contract in pyproject.toml), so `each` lives in
`kullback.runner.parallel` and the Builder's callers keep the name they had.
"""

from __future__ import annotations

from kullback.runner.parallel import each

__all__ = ["each"]
