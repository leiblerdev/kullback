"""What certifies the Runner itself: its import boundary and its version (design section 7).

Every accept-or-reject check over a build artifact moved to `kullback.gates` in phase 3 (D122):
the artifact gates to `gates/artifacts.py`, the fidelity bar to `gates/fidelity.py`, the two
confinement gates to `gates/confinement.py`, the D79 suite to `gates/verifier_suite.py` and the
scorecard to `gates/scorecard.py`. What stays under this name is the pair of checks that rule on
the Runner rather than on the Builder's or the Examiner's work, both defined in `runner/boundary.py`
and re-exported here so `kullback.runner.validate` keeps naming them.
"""

from __future__ import annotations

from kullback.runner.boundary import import_boundary_check, runner_version

__all__ = ["import_boundary_check", "runner_version"]
