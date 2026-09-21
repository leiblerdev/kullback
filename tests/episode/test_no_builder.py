"""The interface stands below the Builder: no import of it, ever (G1).

The importlinter contract in pyproject.toml proves the same over the package graph; this test
fails fast inside the episode suite with the offending line.
"""

from __future__ import annotations

from pathlib import Path


def test_episode_sources_import_no_builder_module():
    root = Path(__file__).resolve().parents[2] / "kullback" / "episode"
    offenders = []
    for path in sorted(root.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if "kullback.builder" in stripped or "kullback.build" in stripped:
                offenders.append(f"{path.name}:{number}: {stripped}")
    assert offenders == []
