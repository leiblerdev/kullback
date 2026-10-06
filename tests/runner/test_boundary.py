"""Tests for runner/boundary.py: the D89 and D91 import scan."""

from __future__ import annotations

from pathlib import Path

import pytest

from kullback.runner.boundary import import_boundary_check

# --- D89 import boundary ---

def test_the_repo_obeys_the_runner_import_boundary():
    """The live check passes on the package directory and on the repo root above it."""
    import kullback

    root = Path(kullback.__file__).resolve().parent
    out = import_boundary_check(root)
    assert out.stage == "import_boundary"
    assert out.passed is True, out.failures
    assert import_boundary_check(root.parent).passed is True


def _tree(root: Path, runner_src: str, ai_src: str = "x = 1\n") -> Path:
    for part in ("runner", "ai", "builder", "examiner"):
        (root / part).mkdir(parents=True)
        (root / part / "__init__.py").write_text("", encoding="utf-8")
    (root / "runner" / "loop.py").write_text(runner_src, encoding="utf-8")
    (root / "ai" / "provider.py").write_text(ai_src, encoding="utf-8")
    (root / "derive.py").write_text("x = 1\n", encoding="utf-8")
    return root


def test_a_runner_import_of_the_builder_is_caught(tmp_path: Path):
    """D89 in both directions: the Runner or the ai package importing the Builder, and the
    derivation reaching into the Runner past records and canon (D91, D121, D123)."""
    root = _tree(tmp_path / "kullback", "from kullback.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("loop.py" in f and "kullback.builder" in f for f in out.failures)

    root = _tree(tmp_path / "ai_side" / "kullback", "x = 1\n", ai_src="from kullback.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("provider.py" in f for f in out.failures)

    root = _tree(tmp_path / "derive_side" / "kullback", "x = 1\n")
    (root / "derive.py").write_text("from kullback.runner.loop import run\n", encoding="utf-8")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("kullback/derive.py" in f and "kullback.runner.loop" in f for f in out.failures)
    (root / "derive.py").write_text(
        "from kullback.runner.records import Run\nfrom kullback.gates import verifier_suite\n", encoding="utf-8")
    assert import_boundary_check(root).passed is True


@pytest.mark.parametrize(
    "src, fragment",
    [
        ("def go():\n    import kullback.builder.verifier as v\n    return v\n", "kullback.builder"),
        (
            "import importlib\n\n\ndef go():\n"
            "    return importlib.import_module('kullback.builder.verifier')\n",
            "import_module",
        ),
        ("from importlib import import_module as grab\n\n\ndef go():\n    return grab('kullback.builder')\n", None),
        ("def go():\n    return __import__('kullback', fromlist=['builder'])\n", None),
        ("import importlib\n\n\ndef go():\n    return importlib.import_module('kullback' + '.builder')\n", None),
        ("import importlib\n\n\ndef go(p):\n    return importlib.import_module(f'kullback.{p}')\n", None),
        ("import importlib\n\n\ndef go():\n    return getattr(importlib, 'import_module')('kullback.builder')\n", None),
        ("import importlib\n\n\ndef go():\n    return importlib.import_module(name='kullback.builder')\n", None),
        ("def go():\n    exec('from kullback.builder import mine')\n", None),
        ("import sys\n\n\ndef go():\n    return sys.modules['kullback.builder.mine']\n", None),
        ("from importlib.util import spec_from_file_location\n\n\ndef go(p):\n"
         "    return spec_from_file_location('m', p)\n", None),
        ("import runpy\n\n\ndef go():\n    return runpy.run_module('kullback.builder.mine')\n", None),
        ("import pkgutil\n\n\ndef go():\n    return pkgutil.resolve_name('kullback.builder.mine:go')\n", None),
    ],
)
def test_import_boundary_check_catches_every_way_around_the_import_statement(
    tmp_path: Path, src: str, fragment
):
    """D89 cannot be read off an aliased, built or exec'd module name, so the primitives are refused."""
    out = import_boundary_check(_tree(tmp_path / "kullback", src))
    assert out.passed is False
    assert any("loop.py" in f and (fragment or "") in f for f in out.failures), out.failures


def test_import_boundary_check_leaves_the_predicate_exec_alone_and_lists_it(tmp_path: Path):
    """verdict.py and the policy gate run model-written predicates; that is not an import."""
    src = "def go(source, env):\n    exec(compile(source, '<atom>', 'exec'), env)\n    return env\n"
    out = import_boundary_check(_tree(tmp_path / "kullback", src))
    assert out.passed is True
    assert any("loop.py" in site and "exec" in site for site in out.metrics["dynamic_code_sites"])


def test_import_boundary_check_fails_a_file_that_does_not_parse_rather_than_raising(tmp_path: Path):
    out = import_boundary_check(_tree(tmp_path / "kullback", "def go(:\n"))
    assert out.passed is False
    assert any("loop.py" in f and "does not parse" in f for f in out.failures)


def test_import_boundary_check_allows_ai_imports_from_runner(tmp_path: Path):
    root = _tree(tmp_path / "kullback", "from kullback.ai.provider import Model\n")
    assert import_boundary_check(root).passed is True
