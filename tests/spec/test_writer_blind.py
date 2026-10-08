"""The Spec writer is blind to the Reference: no module of the Spec package derives a Verifier from a Run.

Source tests: they read the package's import lines, so a Reference path cannot come back unnoticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "kullback" / "spec"
# The writer's path: what writes, compiles and stages a Spec and its Verifier.
WRITER_PATH = ("writer.py", "writer_tools.py", "items.py", "compile.py", "stage.py", "intent.py",
               "intent_tools.py", "schema.py")


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def test_no_module_of_the_spec_package_imports_the_reference_derivation():
    offenders = sorted(path.name for path in SPEC.glob("*.py")
                       if any(name.startswith("kullback.runner.expected") for name in _imports(path)))
    assert offenders == []


def test_the_writers_path_never_loads_a_reference_run():
    for name in WRITER_PATH:
        imports = _imports(SPEC / name)
        assert not any(i.startswith("kullback.spec.witness") for i in imports), name
        assert "references.json" not in (SPEC / name).read_text(encoding="utf-8"), name


def test_no_module_on_the_writers_path_imports_the_reference_gates_or_the_end_state_matcher():
    for name in WRITER_PATH:
        imports = _imports(SPEC / name)
        assert not any(i.startswith(("kullback.spec.end_state", "kullback.runner.expected")) for i in imports), name
