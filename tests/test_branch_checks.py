"""Tests for scripts/branch_checks.py over throwaway git repositories."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "branch_checks.py"

BANNED = "ret" + "ail"
DASH = "\u2014"


def make_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    return path


def write_file(repo, relpath, content):
    target = Path(repo) / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def commit_paths(repo, message, relpaths, body=None):
    subprocess.run(["git", "add", "--", *relpaths], cwd=repo, check=True)
    cmd = ["git", "commit", "-q", "-m", message]
    if body is not None:
        cmd += ["-m", body]
    subprocess.run(cmd, cwd=repo, check=True)
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def run_checks(repo, base):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), base, "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return out.returncode, out.stdout.splitlines()


def test_added_dash_character_fails_no_dashes(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "notes.txt", "plain line\n")
    base = commit_paths(repo, "add notes", ["notes.txt"])
    write_file(repo, "notes.txt", f"plain line\nwith dash {DASH} here\n")
    commit_paths(repo, "add a dash", ["notes.txt"])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert "FAIL no-dashes: notes.txt:2 contains a dash character" in lines


def test_added_banned_word_in_harness_fails_no_corpus_names(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "kullback/toolbox.py", "VALUE = 1\n")
    base = commit_paths(repo, "add toolbox", ["kullback/toolbox.py"])
    write_file(repo, "kullback/toolbox.py", f'VALUE = 1\nDOMAIN = "{BANNED}"\n')
    commit_paths(repo, "name a domain", ["kullback/toolbox.py"])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert f"FAIL no-corpus-names: kullback/toolbox.py:2 names {BANNED!r}" in lines


def add_secret_file(repo):
    write_file(repo, ".env", "SECRET = 1\n")
    commit_paths(repo, "add secret file", [".env"])
    return ".env"


def rename_into_raw(repo):
    (Path(repo) / "data" / "raw").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", "notes.txt", "data/raw/leaked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "move into raw"], cwd=repo, check=True)
    return "data/raw/leaked.txt"


@pytest.mark.parametrize("change", [add_secret_file, rename_into_raw])
def test_a_file_added_or_renamed_onto_a_forbidden_path_fails_no_forbidden_paths(tmp_path, change):
    repo = make_repo(tmp_path)
    write_file(repo, "notes.txt", "plain line\n")
    base = commit_paths(repo, "add notes", ["notes.txt"])
    path = change(repo)
    code, lines = run_checks(repo, base)
    assert code == 1
    assert f"FAIL no-forbidden-paths: {path} is on a forbidden path" in lines


def test_attribution_trailer_fails_no_attribution_trailers(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "notes.txt", "plain line\n")
    base = commit_paths(repo, "add notes", ["notes.txt"])
    write_file(repo, "notes.txt", "plain line\nanother line\n")
    commit_paths(
        repo,
        "extend notes",
        ["notes.txt"],
        body="Co-Authored-By: Some One <some@example.com>",
    )
    code, lines = run_checks(repo, base)
    assert code == 1
    failures = [line for line in lines if line.startswith("FAIL no-attribution-trailers: ")]
    assert len(failures) == 1
    assert "carries Co-Authored-By" in failures[0]


@pytest.mark.parametrize(
    ("tree", "path"),
    [("kullback/runner", "kullback/runner/engine.py"), ("kullback/gates", "kullback/gates/policy.py")],
)
def test_changed_frozen_file_fails_frozen_trees(tmp_path, tree, path):
    repo = make_repo(tmp_path)
    write_file(repo, path, "VALUE = 1\n")
    base = commit_paths(repo, "add frozen file", [path])
    write_file(repo, path, "VALUE = 2\n")
    commit_paths(repo, "tweak frozen file", [path])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert f"FAIL frozen-trees: {tree} differs between {base} and HEAD" in lines


def test_removed_decision_log_line_fails_append_only(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "docs/decision-log.md", "## Log\n\n- First entry kept.\n- Second entry kept.\n")
    base = commit_paths(repo, "add log", ["docs/decision-log.md"])
    write_file(repo, "docs/decision-log.md", "## Log\n\n- First entry kept.\n")
    commit_paths(repo, "trim log", ["docs/decision-log.md"])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert (
        "FAIL decision-log-append-only: docs/decision-log.md removes line: - Second entry kept."
        in lines
    )


def test_risen_function_under_ceiling_passes_complexity_ceiling(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "calc.py", "def score(n):\n    if n > 1:\n        return 1\n    if n > 2:\n        return 2\n    return 0\n")
    base = commit_paths(repo, "add scorer", ["calc.py"])
    write_file(
        repo,
        "calc.py",
        "def score(n):\n    if n > 1:\n        return 1\n    if n > 2:\n        return 2\n"
        "    if n > 3:\n        return 3\n    if n > 4:\n        return 4\n    return 0\n",
    )
    commit_paths(repo, "grow scorer", ["calc.py"])
    code, lines = run_checks(repo, base)
    assert code == 0
    assert "ok complexity-ceiling" in lines
    assert "  complexity-table calc.py score base=3 head=5" in lines


def test_risen_function_above_ceiling_fails_complexity_ceiling(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "calc.py", f"def score(x):\n{big_body(14)}    return -1\n")
    base = commit_paths(repo, "add scorer", ["calc.py"])
    write_file(repo, "calc.py", f"def score(x):\n{big_body(16)}    return -1\n")
    commit_paths(repo, "grow scorer", ["calc.py"])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert "FAIL complexity-ceiling: calc.py score rose from 15 to 17" in lines


def test_same_method_name_in_two_classes_is_tracked_per_class(tmp_path):
    repo = make_repo(tmp_path)
    write_file(
        repo,
        "kullback/shapes.py",
        "class Circle:\n    def run(self, n):\n        if n:\n            return 1\n        return 0\n"
        "\n\nclass Square:\n    def run(self, n):\n        return 0\n",
    )
    base = commit_paths(repo, "add shapes", ["kullback/shapes.py"])
    branches = "".join(f"        if x == {i}:\n            return {i}\n" for i in range(15))
    write_file(
        repo,
        "kullback/shapes.py",
        "class Circle:\n    def run(self, n):\n"
        f"{branches}        return 0\n\n\nclass Square:\n    def run(self, n):\n        return 0\n",
    )
    commit_paths(repo, "grow circle", ["kullback/shapes.py"])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert "FAIL complexity-ceiling: kullback/shapes.py Circle.run rose from 2 to 16" in lines


def test_long_subject_fails_subject_length(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "notes.txt", "plain line\n")
    base = commit_paths(repo, "add notes", ["notes.txt"])
    write_file(repo, "notes.txt", "plain line\nanother line\n")
    commit_paths(repo, "x" * 73, ["notes.txt"])
    code, lines = run_checks(repo, base)
    assert code == 1
    failures = [line for line in lines if line.startswith("FAIL subject-length: ")]
    assert len(failures) == 1
    assert "subject is 73 characters" in failures[0]


@pytest.mark.parametrize(
    ("path", "trailer"),
    [("README.md", "Allow: readme"), ("docs/adr/0001-choice.md", "Allow: adr")],
)
def test_a_change_the_allow_trailer_names_passes_frozen_trees(tmp_path, path, trailer):
    repo = make_repo(tmp_path)
    write_file(repo, path, "Hello\n")
    base = commit_paths(repo, "add file", [path])
    write_file(repo, path, "Hello again\n")
    commit_paths(repo, "update file", [path], body=trailer)
    code, lines = run_checks(repo, base)
    assert "ok frozen-trees" in lines
    assert code == 0


def test_branch_behind_moved_base_still_passes(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "docs/decision-log.md", "## Log\n\n- First entry kept.\n")
    write_file(repo, "kullback/toolbox.py", "def double(n):\n    if n:\n        return 2 * n\n    return 0\n")
    commit_paths(repo, "add base files", ["docs/decision-log.md", "kullback/toolbox.py"])
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "branch", "mainline"], cwd=repo, check=True)
    write_file(
        repo,
        "kullback/toolbox.py",
        "def double(n):\n    if n:\n        return 2 * n\n    return 0\n\n\ndef triple(n):\n    if n:\n        return 3 * n\n    return 0\n",
    )
    commit_paths(repo, "add a small function", ["kullback/toolbox.py"])
    subprocess.run(["git", "checkout", "-q", "mainline"], cwd=repo, check=True)
    write_file(repo, "docs/decision-log.md", "## Log\n\n- First entry kept.\n- Main appended this line.\n")
    commit_paths(repo, "main moves on", ["docs/decision-log.md"])
    subprocess.run(["git", "checkout", "-q", branch], cwd=repo, check=True)
    raw = subprocess.run(
        ["git", "diff", "mainline", "HEAD", "--", "docs/decision-log.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "-- Main appended this line." in raw.stdout
    code, lines = run_checks(repo, "mainline")
    assert code == 0
    assert not any(line.startswith("FAIL ") for line in lines)


def test_unresolvable_base_fails_setup(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "notes.txt", "plain line\n")
    commit_paths(repo, "add notes", ["notes.txt"])
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "origin/nonexistent", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1
    assert "FAIL setup: origin/nonexistent does not resolve" in out.stdout.splitlines()


def test_configured_import_contract_passes(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "pack/__init__.py", "")
    write_file(repo, "pack/one/__init__.py", "")
    write_file(repo, "pack/two/__init__.py", "")
    write_file(repo, "pack/one/a.py", "VALUE = 1\n")
    write_file(repo, "pack/two/b.py", "VALUE = 2\n")
    write_file(
        repo,
        "pyproject.toml",
        '[tool.importlinter]\nroot_package = "pack"\n\n'
        "[[tool.importlinter.contracts]]\nname = \"layers hold\"\ntype = \"layers\"\n"
        'containers = ["pack"]\nlayers = ["one", "two"]\n',
    )
    base = commit_paths(
        repo,
        "add layered pack",
        ["pack/__init__.py", "pack/one/__init__.py", "pack/two/__init__.py",
         "pack/one/a.py", "pack/two/b.py", "pyproject.toml"],
    )
    write_file(repo, "pack/one/a.py", "VALUE = 1\nEXTRA = 2\n")
    commit_paths(repo, "extend pack", ["pack/one/a.py"])
    code, lines = run_checks(repo, base)
    assert code == 0
    assert "ok import-contract" in lines


def test_clean_change_passes_every_check(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "README.md", "Hello\n")
    write_file(repo, "docs/decision-log.md", "## Log\n\n- First entry kept.\n")
    write_file(repo, "kullback/toolbox.py", "def double(n):\n    if n:\n        return 2 * n\n    return 0\n")
    base = commit_paths(
        repo, "add base files", ["README.md", "docs/decision-log.md", "kullback/toolbox.py"]
    )
    write_file(
        repo,
        "kullback/toolbox.py",
        "def double(n):\n    if n:\n        return 2 * n\n    return 0\n\n\ndef triple(n):\n    if n:\n        return 3 * n\n    return 0\n",
    )
    commit_paths(repo, "add a small function", ["kullback/toolbox.py"])
    code, lines = run_checks(repo, base)
    assert code == 0
    assert all(
        line.startswith("ok ") or line.startswith("skip ") or line.startswith("  complexity-table ")
        for line in lines
    )
    assert not any(line.startswith("FAIL ") for line in lines)
    assert "  complexity-table kullback/toolbox.py triple base=- head=2" in lines


def big_body(count):
    branches = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(count))
    return branches


@pytest.mark.parametrize("add_created", [False, True])
def test_a_moved_function_over_ceiling_is_not_counted_as_new(tmp_path, add_created):
    repo = make_repo(tmp_path)
    body = big_body(15)
    write_file(repo, "kullback/oldmod.py", f"def evaluate(x):\n{body}    return -1\n")
    base = commit_paths(repo, "add old module", ["kullback/oldmod.py"])
    subprocess.run(["git", "rm", "-q", "kullback/oldmod.py"], cwd=repo, check=True)
    created = f"\n\ndef judge(x):\n{big_body(15)}    return -1\n" if add_created else ""
    write_file(repo, "kullback/newmod.py", f"def evaluate(x):\n{body}    return -1\n{created}")
    subprocess.run(["git", "add", "--", "kullback/newmod.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "move function"], cwd=repo, check=True)
    code, lines = run_checks(repo, base)
    failures = [line for line in lines if line.startswith("FAIL complexity-ceiling: ")]
    if add_created:
        assert code == 1
        assert len(failures) == 1
        assert "judge is new at 16" in failures[0]
        assert "evaluate" not in failures[0]
    else:
        assert code == 0
        assert "ok complexity-ceiling" in lines


def test_moved_function_grown_fails_naming_both_files(tmp_path):
    repo = make_repo(tmp_path)
    write_file(repo, "kullback/oldmod.py", f"def evaluate(x):\n{big_body(15)}    return -1\n")
    base = commit_paths(repo, "add old module", ["kullback/oldmod.py"])
    subprocess.run(["git", "rm", "-q", "kullback/oldmod.py"], cwd=repo, check=True)
    write_file(repo, "kullback/newmod.py", f"def evaluate(x):\n{big_body(16)}    return -1\n")
    subprocess.run(["git", "add", "--", "kullback/newmod.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "move and grow"], cwd=repo, check=True)
    code, lines = run_checks(repo, base)
    assert code == 1
    failures = [line for line in lines if line.startswith("FAIL complexity-ceiling: ")]
    assert len(failures) == 1
    assert "kullback/newmod.py" in failures[0]
    assert "evaluate" in failures[0]
    assert "kullback/oldmod.py" in failures[0]
    assert "16" in failures[0]
    assert "17" in failures[0]


@pytest.mark.parametrize(
    ("old_path", "old_text", "new_path"),
    [
        ("notes.txt", "plain line\n", "big.py"),
        (
            "kullback/oldmod.py",
            "def double(n):\n    if n:\n        return 2 * n\n    return 0\n",
            "kullback/newmod.py",
        ),
    ],
)
def test_genuinely_new_function_over_ceiling_still_fails(tmp_path, old_path, old_text, new_path):
    repo = make_repo(tmp_path)
    write_file(repo, old_path, old_text)
    base = commit_paths(repo, "add old file", [old_path])
    write_file(repo, new_path, f"def evaluate(x):\n{big_body(15)}    return -1\n")
    commit_paths(repo, "add new function", [new_path])
    code, lines = run_checks(repo, base)
    assert code == 1
    assert f"FAIL complexity-ceiling: {new_path} evaluate is new at 16 (ceiling 15)" in lines

