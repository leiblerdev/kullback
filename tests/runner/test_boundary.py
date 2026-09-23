"""Tests for runner/boundary.py: the D89 and D91 import scan, the RunnerVersion and the gates hash beside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from kullback.runner.boundary import import_boundary_check, runner_version

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
    (root / "examiner" / "derive.py").write_text("x = 1\n", encoding="utf-8")
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
    (root / "examiner" / "derive.py").write_text("from kullback.runner.loop import run\n", encoding="utf-8")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("examiner/derive.py" in f and "kullback.runner.loop" in f for f in out.failures)
    (root / "examiner" / "derive.py").write_text(
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


# --- RunnerVersion ---

def _runner_tree(root: Path, loop_body="a = 1\n") -> Path:
    (root / "runner").mkdir(parents=True)
    (root / "runner" / "loop.py").write_text(loop_body, encoding="utf-8")
    (root / "runner" / "route.py").write_text("b = 2\n", encoding="utf-8")
    (root / "runner" / "verdict.py").write_text("c = 3\n", encoding="utf-8")
    return root


def test_runner_version_hashes_every_runner_file_and_the_routing_config(tmp_path: Path):
    """The hash covers every Runner file and the routing config; on the real tree every file is
    hashed as it is on disk (D61: Runner frozen)."""
    import kullback
    from kullback.runner.records import content_hash

    root = _runner_tree(tmp_path / "kullback")
    out = runner_version(root, routing_config={"order": ["code", "recording", "llm"]})
    assert set(out.file_hashes) == {"loop.py", "route.py", "verdict.py"}
    assert all(len(h) == 64 for h in out.file_hashes.values())
    assert out.routing_config_hash is not None
    assert len(out.runner_version) == 64
    assert runner_version(root, routing_config={"order": ["code", "recording", "llm"]}).runner_version == \
        out.runner_version

    real = Path(kullback.__file__).resolve().parent
    out = runner_version(real)
    for name in ("loop.py", "route.py", "verdict.py"):
        assert out.file_hashes[name] == content_hash((real / "runner" / name).read_text(encoding="utf-8"))
    assert out.runner_version == runner_version(real).runner_version
    assert out.runner_version != runner_version(real, routing_config={"order": ["code"]}).runner_version


def test_runner_version_follows_the_files_under_runner_and_the_routing_config(tmp_path: Path):
    """RUNNER_FILES went away with the move (D121, D130): the hash is over whatever is under runner/.

    A file that is not there is not hashed, not marked "missing", and a new file joins the hash
    unasked; both move the version, and so does a changed routing config.
    """
    root = _runner_tree(tmp_path / "kullback")
    a = runner_version(root, routing_config={"order": ["code"]}).runner_version
    b = runner_version(root, routing_config={"order": ["code", "recording"]}).runner_version
    assert a != b
    before = runner_version(root).runner_version
    (root / "runner" / "verdict.py").unlink()
    removed = runner_version(root)
    assert "verdict.py" not in removed.file_hashes
    assert set(removed.file_hashes) == {"loop.py", "route.py"}
    assert removed.runner_version != before
    (root / "runner" / "gate_support.py").write_text("d = 4\n", encoding="utf-8")
    added = runner_version(root)
    assert set(added.file_hashes) == {"loop.py", "route.py", "gate_support.py"}
    assert added.runner_version not in (before, removed.runner_version)


# --- the gates hash beside the Runner's (D122) ---

def _gated_tree(root: Path) -> Path:
    _runner_tree(root)
    (root / "gates").mkdir()
    (root / "gates" / "__init__.py").write_text("", encoding="utf-8")
    (root / "gates" / "artifacts.py").write_text("g = 1\n", encoding="utf-8")
    return root


def test_the_gates_package_is_hashed_beside_the_runner_and_never_into_it(tmp_path: Path):
    """A gate that changes does not change what executes or grades a Run, so the Runner's hash
    holds still and the gates hash moves; a regrade can name both. A tree with no gates package
    (a RunnerVersion frozen before phase 3) records None, and the real package is hashed file by
    file as it is on disk."""
    import kullback
    from kullback.runner.records import content_hash

    root = _gated_tree(tmp_path / "kullback")
    out = runner_version(root)
    assert set(out.gates_file_hashes) == {"__init__.py", "artifacts.py"}
    assert out.gates_version and len(out.gates_version) == 64
    assert out.gates_version != out.runner_version
    (root / "gates" / "artifacts.py").write_text("g = 2\n", encoding="utf-8")
    moved = runner_version(root)
    assert moved.gates_version != out.gates_version
    assert moved.runner_version == out.runner_version
    (root / "runner" / "loop.py").write_text("a = 9\n", encoding="utf-8")
    runner_moved = runner_version(root)
    assert runner_moved.runner_version != out.runner_version
    assert runner_moved.gates_version == moved.gates_version

    bare = runner_version(_runner_tree(tmp_path / "bare" / "kullback"))
    assert bare.gates_version is None
    assert bare.gates_file_hashes == {}

    real = Path(kullback.__file__).resolve().parent
    out = runner_version(real)
    on_disk = sorted(p.relative_to(real / "gates").as_posix() for p in (real / "gates").rglob("*.py"))
    assert sorted(out.gates_file_hashes) == on_disk
    for name in ("__init__.py", "artifacts.py", "verifier_suite.py", "fidelity.py", "confinement.py"):
        assert out.gates_file_hashes[name] == content_hash((real / "gates" / name).read_text(encoding="utf-8"))
    assert out.gates_version == content_hash({"files": out.gates_file_hashes})
