"""runner/code_hash.py: one hash of the code that executes and scores a Run, from the files' bytes."""

from __future__ import annotations

from pathlib import Path

import kullback
from kullback.runner import code_hash as C
from kullback.runner import tool


def _tree(root: Path) -> Path:
    for package, name, body in (("runner", "loop.py", "a = 1\n"), ("gates", "trust.py", "b = 2\n"),
                                ("spec", "report.py", "c = 3\n"), ("builder", "mine.py", "d = 4\n")):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / name).write_text(body, encoding="utf-8")
    return root


def test_a_changed_byte_in_any_scoring_package_changes_the_hash_and_other_packages_do_not(tmp_path):
    root = _tree(tmp_path / "kullback")
    before = C.code_hash(root)
    assert C.code_hash(root) == before and len(before) == 64
    for package, name in (("runner", "loop.py"), ("gates", "trust.py"), ("spec", "report.py")):
        path = root / package / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        assert C.code_hash(root) != before, package
        path.write_bytes(original)
    assert C.code_hash(root) == before
    (root / "builder" / "mine.py").write_text("d = 5\n", encoding="utf-8")
    assert C.code_hash(root) == before


def test_a_file_moving_in_or_out_of_a_package_moves_the_hash(tmp_path):
    root = _tree(tmp_path / "kullback")
    before = C.code_hash(root)
    (root / "gates" / "new.py").write_text("", encoding="utf-8")
    added = C.code_hash(root)
    (root / "gates" / "new.py").rename(root / "gates" / "renamed.py")
    assert len({before, added, C.code_hash(root)}) == 3


def test_the_hash_taken_at_import_is_the_installed_trees_and_runs_carry_it():
    assert C.CODE_HASH == C.code_hash(Path(kullback.__file__).resolve().parent)
    assert tool.version() == C.CODE_HASH
