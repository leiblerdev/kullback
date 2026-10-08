"""The Spec sits below the Examiner and the Builder: it never imports either at module level."""

from __future__ import annotations

import ast
from pathlib import Path

import kullback.spec

EXPECTED: set = set()


def test_spec_imports_nothing_from_the_examiner_or_the_builder():
    found = set()
    for path in sorted(Path(kullback.spec.__file__).parent.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.startswith(("kullback.examiner", "kullback.builder")):
                    found.add((path.name, name))
    assert found == EXPECTED
