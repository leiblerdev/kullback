"""Tests for runner/boundary.py: the D89 and D91 import scan, the RunnerVersion and the gates hash beside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from kullback.runner.boundary import import_boundary_check, runner_version

# --- D89 import boundary ---

def test_import_boundary_check_passes_on_this_repo():
    import kullback

    root = Path(kullback.__file__).resolve().parent
    out = import_boundary_check(root)
    assert out.stage == "import_boundary"
    assert out.passed is True, out.failures


def test_import_boundary_check_accepts_the_repo_root_too():
    import kullback

    repo_root = Path(kullback.__file__).resolve().parents[1]
    assert import_boundary_check(repo_root).passed is True


def _tree(root: Path, runner_src: str, ai_src: str = "x = 1\n") -> Path:
    for part in ("runner", "ai", "builder"):
        (root / part).mkdir(parents=True)
        (root / part / "__init__.py").write_text("", encoding="utf-8")
    (root / "runner" / "loop.py").write_text(runner_src, encoding="utf-8")
    (root / "ai" / "provider.py").write_text(ai_src, encoding="utf-8")
    (root / "builder" / "verifier.py").write_text("x = 1\n", encoding="utf-8")
    return root


def test_import_boundary_check_catches_a_top_level_builder_import(tmp_path: Path):
    root = _tree(tmp_path / "kullback", "from kullback.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("loop.py" in f and "kullback.builder" in f for f in out.failures)


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


def test_import_boundary_check_catches_the_verifier_reaching_into_the_runner(tmp_path: Path):
    """D91's other direction, which D89 says the same test covers.

    records and canon moved into runner/ from the dissolved shared/ package (D121) and stay
    readable to verifier.py, the same allowance it always had; every other runner/ module is still
    off limits.
    """
    root = _tree(tmp_path / "kullback", "x = 1\n")
    (root / "builder" / "verifier.py").write_text("from kullback.runner.loop import run\n", encoding="utf-8")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("verifier.py" in f and "kullback.runner.loop" in f for f in out.failures)
    (root / "builder" / "verifier.py").write_text(
        "from kullback.runner.records import Run\nfrom kullback.builder import policy\n", encoding="utf-8")
    assert import_boundary_check(root).passed is True


def test_import_boundary_check_fails_a_file_that_does_not_parse_rather_than_raising(tmp_path: Path):
    out = import_boundary_check(_tree(tmp_path / "kullback", "def go(:\n"))
    assert out.passed is False
    assert any("loop.py" in f and "does not parse" in f for f in out.failures)


def test_import_boundary_check_catches_the_ai_package_importing_the_builder(tmp_path: Path):
    root = _tree(tmp_path / "kullback", "x = 1\n", ai_src="from kullback.builder import mine\n")
    out = import_boundary_check(root)
    assert out.passed is False
    assert any("provider.py" in f for f in out.failures)


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
    root = _runner_tree(tmp_path / "kullback")
    out = runner_version(root, routing_config={"order": ["code", "recording", "llm"]})
    assert set(out.file_hashes) == {"loop.py", "route.py", "verdict.py"}
    assert all(len(h) == 64 for h in out.file_hashes.values())
    assert out.routing_config_hash is not None
    assert len(out.runner_version) == 64
    assert runner_version(root, routing_config={"order": ["code", "recording", "llm"]}).runner_version == \
        out.runner_version


def test_runner_version_moves_when_the_routing_config_changes(tmp_path: Path):
    root = _runner_tree(tmp_path / "kullback")
    a = runner_version(root, routing_config={"order": ["code"]}).runner_version
    b = runner_version(root, routing_config={"order": ["code", "recording"]}).runner_version
    assert a != b


def test_runner_version_follows_the_files_under_runner(tmp_path: Path):
    """RUNNER_FILES went away with the move (D121, D130): the hash is over whatever is under runner/.

    A file that is not there is not hashed, not marked "missing", and a new file joins the hash
    unasked; both move the version.
    """
    root = _runner_tree(tmp_path / "kullback")
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


def test_runner_version_hashes_the_real_runner_files_as_they_are_on_disk():
    """Every real Runner file, hashed as it is on disk (D61: Runner frozen)."""
    import kullback
    from kullback.runner.records import content_hash

    root = Path(kullback.__file__).resolve().parent
    out = runner_version(root)
    for name in ("loop.py", "route.py", "verdict.py"):
        assert out.file_hashes[name] == content_hash((root / "runner" / name).read_text(encoding="utf-8"))
    assert out.runner_version == runner_version(root).runner_version
    assert out.runner_version != runner_version(root, routing_config={"order": ["code"]}).runner_version




# --- the gates hash beside the Runner's (D122) ---

def _gated_tree(root: Path) -> Path:
    _runner_tree(root)
    (root / "gates").mkdir()
    (root / "gates" / "__init__.py").write_text("", encoding="utf-8")
    (root / "gates" / "artifacts.py").write_text("g = 1\n", encoding="utf-8")
    return root


def test_the_gates_package_is_hashed_beside_the_runner_and_never_into_it(tmp_path: Path):
    """A gate that changes does not change what executes or grades a Run, so the Runner's hash
    holds still and the gates hash moves; a regrade can name both."""
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


def test_a_tree_with_no_gates_package_records_no_gates_version(tmp_path: Path):
    """A RunnerVersion frozen before phase 3 has no gates hash, and the record says so with None."""
    out = runner_version(_runner_tree(tmp_path / "kullback"))
    assert out.gates_version is None
    assert out.gates_file_hashes == {}


def test_gates_version_hashes_every_file_under_the_real_gates_package():
    """Every .py under kullback/gates/, hashed as it is on disk, the same way the Runner is."""
    import kullback
    from kullback.runner.records import content_hash

    root = Path(kullback.__file__).resolve().parent
    out = runner_version(root)
    on_disk = sorted(p.relative_to(root / "gates").as_posix() for p in (root / "gates").rglob("*.py"))
    assert sorted(out.gates_file_hashes) == on_disk
    for name in ("__init__.py", "artifacts.py", "verifier_suite.py", "fidelity.py", "confinement.py"):
        assert out.gates_file_hashes[name] == content_hash((root / "gates" / name).read_text(encoding="utf-8"))
    assert out.gates_version == content_hash({"files": out.gates_file_hashes})
