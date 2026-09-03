"""Who may reach into the agent core (D121, D123, D130).

Phase 2 wrote this as "nothing outside `kullback/agent/` imports the new code", because nothing
used the core yet. Phase 4 put the Builder on it and phase 5 will put the Examiner there, which is
the layout D121 and D123 decided, so the assertion is now the one that still holds: only the two
application packages and the frontends reach into the agent core, and `runner/` and `gates/` never
do. The import-linter contract in `pyproject.toml` says the same over real imports; this test reads
the source so a failure names the file and the module rather than a layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "kullback"
CORE = ("kullback.agent", "kullback.ai.stream", "kullback.ai.messages")
# The stream and its message shapes are the core's own; the applications sit on the core (D121,
# D123); the frontends consume the event stream and call the applications' entrypoints (D129).
MAY_IMPORT = ("kullback/ai/stream.py", "kullback/builder/", "kullback/examiner/", "kullback/rounds.py",
              "kullback/cli.py", "kullback/tui/")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_the_applications_and_the_frontends_import_the_agent_core():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if rel.startswith("kullback/agent/") or rel.startswith(MAY_IMPORT):
            continue
        hits = [m for m in imported_modules(path) if m.startswith(CORE)]
        if hits:
            offenders.append((rel, hits))
    assert offenders == [], offenders


def test_the_runner_and_the_gates_never_import_the_agent_core():
    offenders = []
    for package in ("kullback/runner", "kullback/gates"):
        scanned = sorted((PACKAGE.parent / package).rglob("*.py"))
        # a package that moved or was renamed must fail here, not silently scan nothing
        assert scanned, package
        for path in scanned:
            rel = path.relative_to(PACKAGE.parent).as_posix()
            hits = [m for m in imported_modules(path) if m.startswith(CORE)]
            if hits:
                offenders.append((rel, hits))
    assert offenders == [], offenders
