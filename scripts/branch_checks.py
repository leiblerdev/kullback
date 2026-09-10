"""Mechanical branch checks for worker PRs, reviewers and CI.

Usage: python scripts/branch_checks.py [base] [head]
Defaults: base is origin/main, head is HEAD. Exit 0 when every check
passes, 1 otherwise. Prints one line per check: "ok <name>" or
"FAIL <name>: <reason>". The complexity check also prints a table of
touched functions with their before and after values.
"""

import argparse
import ast
import difflib
import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Names the harness must stay independent of: corpus and customer
# domains that must never appear in new harness code or tests.
FORBIDDEN_NAMES = ("tau2", "tau-bench", "retail", "airline", "telecom")

FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN_NAMES) + r")\b",
    re.IGNORECASE,
)

FROZEN_PATHS = (
    "kullback/runner",
    "kullback/gates",
    "docs/adr",
    "docs/founder-words.md",
    "README.md",
)

MAX_NEW_COMPLEXITY = 15
MAX_SUBJECT_LEN = 72


def run_git(root, *args):
    out = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return out


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print("FAIL setup: not inside a git repository")
        raise SystemExit(1)
    return Path(out.stdout.strip())


def commit_bodies(root, base, head):
    out = run_git(root, "log", "--format=%B%x1e%x1e", f"{base}..{head}")
    if out.returncode != 0:
        return None
    return out.stdout


def allow_lines_present(root, base, head):
    bodies = commit_bodies(root, base, head)
    if bodies is None:
        return set()
    found = set()
    for line in bodies.splitlines():
        text = line.strip()
        if text == "Allow: readme":
            found.add("readme")
        elif text == "Allow: adr":
            found.add("adr")
    return found


def check_frozen_trees(root, base, head):
    """Frozen trees stay byte identical unless an allow line permits it."""
    allowed = allow_lines_present(root, base, head)
    bad = []
    for path in FROZEN_PATHS:
        out = run_git(root, "diff", "--quiet", base, head, "--", path)
        if out.returncode == 0:
            continue
        if path == "README.md" and "readme" in allowed:
            continue
        if path == "docs/adr" and "adr" in allowed:
            continue
        bad.append(f"{path} differs between {base} and {head}")
    if bad:
        return "; ".join(bad)
    return None


def check_decision_log_append_only(root, base, head):
    """The decision log gains lines but never loses them."""
    out = run_git(root, "diff", base, head, "--", "docs/decision-log.md")
    removed = [
        line[1:]
        for line in out.stdout.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    if removed:
        return "docs/decision-log.md removes line: " + removed[0][:120]
    return None


def added_lines_with_numbers(root, base, head):
    """Yield (path, new line number, text) for every added diff line."""
    out = run_git(root, "diff", base, head, "--")
    current = None
    new_no = 0
    for raw in out.stdout.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[len("+++ b/"):]
            continue
        if raw.startswith("+++ "):
            current = None
            continue
        if raw.startswith("@@ "):
            match = re.search(r"\+(\d+)", raw)
            new_no = int(match.group(1)) if match else 0
            continue
        if raw.startswith("\\"):
            continue
        if current is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            yield current, new_no, raw[1:]
            new_no += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue
        else:
            new_no += 1


def in_scope(paths, path):
    return any(path == p or path.startswith(p + "/") for p in paths)


def check_no_dashes(root, base, head):
    """Added lines carry no em dash or en dash."""
    bad = [
        f"{path}:{no} contains a dash character"
        for path, no, text in added_lines_with_numbers(root, base, head)
        if "\u2014" in text or "\u2013" in text
    ]
    if bad:
        return "; ".join(bad)
    return None


def check_no_corpus_names(root, base, head):
    """Added harness and test lines name no outside corpus or customer."""
    bad = [
        f"{path}:{no} names {FORBIDDEN_RE.search(text).group(0)!r}"
        for path, no, text in added_lines_with_numbers(root, base, head)
        if in_scope(("kullback", "tests"), path) and FORBIDDEN_RE.search(text)
    ]
    if bad:
        return "; ".join(bad)
    return None


def is_forbidden_path(path):
    if path == ".claude" or path.startswith(".claude/"):
        return True
    if any(fnmatch.fnmatch(part, ".work-*") for part in path.split("/")):
        return True
    if path == "vendor" or path.startswith("vendor/"):
        return True
    if path == "data/raw" or path.startswith("data/raw/"):
        return True
    name = path.rsplit("/", 1)[-1]
    return name in (".env", ".DS_Store", "review.md", "docs-cleanup-plan.md")


def check_no_forbidden_paths(root, base, head):
    """No added file lives on a forbidden path."""
    out = run_git(root, "diff", "--name-status", base, head, "--")
    bad = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status, names = parts[0], parts[1:]
        if status.startswith("R") or status.startswith("C"):
            names = names[-1:]
        for name in names:
            if status[0] in "ACR" and is_forbidden_path(name):
                bad.append(f"{name} is on a forbidden path")
    if bad:
        return "; ".join(bad)
    return None


def check_no_attribution_trailers(root, base, head):
    """Commit messages carry no attribution trailer but a session line may."""
    out = run_git(root, "log", "--format=%H %s", f"{base}..{head}")
    bad = []
    for line in out.stdout.splitlines():
        sha, _, subject = line.partition(" ")
        body = run_git(root, "log", "--format=%B", "-n", "1", sha)
        if "Co-Authored-By" in body.stdout:
            bad.append(f"{sha[:12]} {subject[:60]} carries Co-Authored-By")
    if bad:
        return "; ".join(bad)
    return None


def complexity_of(node):
    score = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ExceptHandler,
                ast.With,
                ast.AsyncWith,
                ast.Assert,
                ast.IfExp,
            ),
        ):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
        elif isinstance(
            child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            for gen in child.generators:
                score += 1 + len(gen.ifs)
        elif isinstance(child, ast.match_case):
            score += 1
    return score


def function_details(source):
    """Map qualified function name to (complexity, body hash, body dump)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}" if prefix else child.name
                dump = ast.dump(child, annotate_fields=False, include_attributes=False)
                digest = hashlib.sha256(dump.encode("utf-8")).hexdigest()
                found[name] = (complexity_of(child), digest, dump)
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def function_complexities(source):
    return {name: item[0] for name, item in function_details(source).items()}


def base_moved_index(root, base):
    """Map function name to [(file, complexity, body hash, body dump)] on base."""
    out = run_git(root, "ls-tree", "-r", "--name-only", base, "--", "kullback", "scripts")
    if out.returncode != 0:
        return {}
    index = {}
    for path in out.stdout.splitlines():
        if not path.endswith(".py"):
            continue
        src = show_file(root, base, path)
        if src is None:
            continue
        for name, (value, digest, dump) in function_details(src).items():
            index.setdefault(name, []).append((path, value, digest, dump))
    return index


def pick_moved_candidate(candidates, head_digest, head_dump, head_value):
    same = [item for item in candidates if item[2] == head_digest]
    if same:
        return sorted(same, key=lambda item: item[0])[0]
    scored = []
    for item in candidates:
        ratio = difflib.SequenceMatcher(None, head_dump, item[3]).ratio()
        scored.append((ratio, -abs(item[1] - head_value), item[0], item))
    scored.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return scored[0][3]


def show_file(root, rev, path):
    out = run_git(root, "show", f"{rev}:{path}")
    if out.returncode != 0:
        return None
    return out.stdout


def touched_python_files(root, base, head):
    out = run_git(root, "diff", "--name-status", base, head, "--")
    files = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status, names = parts[0], parts[1:]
        if status.startswith("D"):
            continue
        current = names[-1] if names else ""
        old = names[0] if status.startswith("R") and len(names) > 1 else current
        if current.endswith(".py"):
            files.append((old, current))
    return files


def moved_or_new_reason(name, current, after, head_info, moved_index):
    """Pass verbatim moves, flag grown moves, flag new functions."""
    candidates = moved_index.get(name, [])
    if not candidates:
        return f"{current} {name} is new at {after} (ceiling {MAX_NEW_COMPLEXITY})"
    picked = pick_moved_candidate(candidates, head_info[1], head_info[2], after)
    old_file, old_value = picked[0], picked[1]
    if old_value >= after:
        return None
    return f"{current} {name} moved from {old_file} and rose from {old_value} to {after}"


def rose_reason(name, old, current, before, after):
    """Name both files when a rename carries a complexity rise."""
    if old != current:
        return f"{current} {name} moved from {old} and rose from {before} to {after}"
    return f"{current} {name} rose from {before} to {after}"


def dash_text(value):
    """Render a complexity value for the table."""
    if value is None:
        return "-"
    return str(value)


def file_ceiling_entries(old, current, base_funcs, head_infos, box):
    """Table lines and failure reasons for one touched file."""
    head_funcs = {name: item[0] for name, item in head_infos.items()}
    lines = []
    reasons = []
    for name in sorted(set(head_funcs) | set(base_funcs)):
        before = base_funcs.get(name)
        after = head_funcs.get(name)
        lines.append(
            f"  complexity-table {current} {name} "
            f"base={dash_text(before)} head={dash_text(after)}"
        )
        if after is None:
            continue
        if before is None:
            if after > MAX_NEW_COMPLEXITY:
                if box["index"] is None:
                    box["index"] = base_moved_index(box["root"], box["base"])
                reason = moved_or_new_reason(name, current, after, head_infos[name], box["index"])
                if reason is not None:
                    reasons.append(reason)
        elif after > before:
            reasons.append(rose_reason(name, old, current, before, after))
    return lines, reasons


def check_complexity_ceiling(root, base, head):
    """Changed functions keep their ceiling and new ones stay at 15 or under."""
    table = []
    bad = []
    box = {"root": root, "base": base, "index": None}
    for old, current in touched_python_files(root, base, head):
        head_src = show_file(root, head, current)
        if head_src is None:
            continue
        base_src = show_file(root, base, old)
        head_infos = function_details(head_src)
        if base_src is None:
            base_funcs = {}
        else:
            base_funcs = function_complexities(base_src)
        lines, reasons = file_ceiling_entries(old, current, base_funcs, head_infos, box)
        table.extend(lines)
        bad.extend(reasons)
    for line in table:
        print(line)
    if bad:
        return "; ".join(bad)
    return None


def check_subject_length(root, base, head):
    """Every commit subject fits in 72 characters."""
    out = run_git(root, "log", "--format=%H %s", f"{base}..{head}")
    bad = []
    for line in out.stdout.splitlines():
        sha, _, subject = line.partition(" ")
        if len(subject) > MAX_SUBJECT_LEN:
            bad.append(f"{sha[:12]} subject is {len(subject)} characters")
    if bad:
        return "; ".join(bad)
    return None


def check_import_contract(root, base, head):
    """The import contract holds when the project configures it."""
    pyproject = root / "pyproject.toml"
    try:
        text = pyproject.read_text()
    except OSError:
        return "skip"
    if "[tool.importlinter]" not in text:
        return "skip"
    binary = Path(sys.prefix) / "bin" / "lint-imports"
    command = str(binary) if binary.exists() else "lint-imports"
    if command == "lint-imports" and shutil.which("lint-imports") is None:
        return "lint-imports is not installed"
    out = subprocess.run(
        [command],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        tail = (out.stdout + out.stderr).strip().splitlines()
        detail = tail[-1][:160] if tail else "contract broken"
        return detail
    return None


CHECKS = (
    ("frozen-trees", check_frozen_trees),
    ("decision-log-append-only", check_decision_log_append_only),
    ("no-dashes", check_no_dashes),
    ("no-corpus-names", check_no_corpus_names),
    ("no-forbidden-paths", check_no_forbidden_paths),
    ("no-attribution-trailers", check_no_attribution_trailers),
    ("complexity-ceiling", check_complexity_ceiling),
    ("subject-length", check_subject_length),
    ("import-contract", check_import_contract),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mechanical branch checks.")
    parser.add_argument("base", nargs="?", default="origin/main")
    parser.add_argument("head", nargs="?", default="HEAD")
    args = parser.parse_args(argv)
    root = repo_root()
    for rev in (args.base, args.head):
        probe = run_git(root, "rev-parse", "--verify", rev + "^{commit}")
        if probe.returncode != 0:
            print(f"FAIL setup: {rev} does not resolve")
            return 1
    merged = run_git(root, "merge-base", args.base, args.head)
    if merged.returncode != 0 or not merged.stdout.strip():
        print(f"FAIL setup: no merge base between {args.base} and {args.head}")
        return 1
    base = merged.stdout.strip()
    failed = False
    for name, func in CHECKS:
        reason = func(root, base, args.head)
        if reason == "skip":
            print("skip import contract (not configured)")
        elif reason is None:
            print(f"ok {name}")
        else:
            print(f"FAIL {name}: {reason}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
