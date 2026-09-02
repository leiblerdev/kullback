"""The D89 and D91 import boundary as a static AST scan, and the RunnerVersion that scan
certifies (design section 7)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional, Union

from kullback.runner.gate_support import gate
from kullback.runner.records import GateResult, RunnerVersion, content_hash

# The two directories that must never reach the Builder, whatever the model does with a tool call:
# ai/ (the provider layer runner/ depends on) and runner/ (which now also holds the records,
# canonicalizer, confinement and budget code that used to live in the dissolved shared/ package,
# D121). A dynamic import is a failure on its own: an aliased `import_module`, a module name built
# by concatenation and an `exec` string all read the same to a static scan, so the primitives are
# refused rather than their arguments inspected.
SCANNED_PACKAGES = ("runner", "ai")
# records and canon moved into runner/ from shared/ with nothing else about them changing (D121):
# verifier.py may still read them, the same allowance it had when they sat outside the Runner
# entirely. Every other runner/ module stays off limits under D91.
RUNNER_DATA_MODULES = ("records", "canon")
DYNAMIC_IMPORT_MODULES = ("importlib", "runpy", "pkgutil")
DYNAMIC_IMPORT_CALLS = (
    "import_module", "__import__", "spec_from_file_location", "module_from_spec",
    "run_module", "run_path", "resolve_name", "get_loader", "find_loader",
)


def _package_root(src_root: Union[str, Path]) -> Path:
    """`src_root` is either the `kullback` package directory itself or its parent."""
    root = Path(src_root)
    return root / "kullback" if (root / "kullback").is_dir() else root


def import_boundary_check(src_root: Union[str, Path]) -> GateResult:
    """Both directions of the D89 and D91 boundary, over runner/, ai/ and builder/verifier.py.

    Sites that run code from a value this scan cannot read (the Verifier atoms, the policy
    predicates) are listed in the metrics, not failed.
    """
    root = _package_root(src_root)
    failures, files, sites = [], 0, []
    for part in SCANNED_PACKAGES:
        directory = root / part
        for path in sorted(directory.rglob("*.py")) if directory.is_dir() else ():
            files += 1
            found, seen = _import_failures(path, part)
            failures += found
            sites += seen
    verifier = root / "builder" / "verifier.py"
    if verifier.is_file():
        files += 1
        failures += _verifier_failures(verifier)
    return gate("import_boundary", failures, files=files, dynamic_code_sites=sites)


def _import_failures(path: Path, part: str) -> tuple[list[str], list[str]]:
    where = f"{part}/{path.name}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, ValueError) as exc:
        return [f"{where} does not parse, so the D89 boundary cannot be checked on it: {exc}"], []
    except OSError as exc:
        return [f"{where} cannot be read, so the D89 boundary cannot be checked on it: {exc}"], []
    return _boundary_failures(tree, where)


def _boundary_failures(tree: ast.AST, where: str, inside: str = "") -> tuple[list[str], list[str]]:
    out: list[str] = []
    sites: list[str] = []
    for node in ast.walk(tree):
        for name, how in _imported_names(node):
            out += _boundary_line(where + inside, name, how)
        if isinstance(node, ast.Name) and node.id == "__import__":
            out.append(f"{where}{inside} uses __import__; nothing here reaches the module system at "
                       "runtime, because the D89 boundary cannot be read off such a call")
        if isinstance(node, ast.Attribute) and node.attr == "modules" and \
                isinstance(node.value, ast.Name) and node.value.id == "sys":
            out.append(f"{where}{inside} reaches sys.modules; nothing here reaches the module system "
                       "at runtime (D89)")
        if not isinstance(node, ast.Call):
            continue
        callee = _callee(node)
        if callee in DYNAMIC_IMPORT_CALLS:
            out.append(f"{where}{inside} calls {callee}; nothing here imports by name at runtime, "
                       "whatever module the call names (D89)")
            out += [line for value in _string_args(node) for line in _boundary_line(where + inside, value, "call")]
        elif callee in ("exec", "eval"):
            source = node.args[0] if node.args else None
            if isinstance(source, ast.Constant) and isinstance(source.value, str):
                out += _exec_failures(source.value, where, inside)
            else:
                sites.append(f"{where}: {callee} on a value this scan cannot read, line {node.lineno}")
    return sorted(set(out)), sites


def _exec_failures(source: str, where: str, inside: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _boundary_failures(tree, where, inside + " (inside an exec string)")[0]


def _boundary_line(where: str, name: str, how: str) -> list[str]:
    verb = "imports" if how == "import" else "names"
    if _is_builder(name):
        return [f"{where} {verb} {name}; the Runner never imports the Builder (D89)"]
    if _is_verifier(name):
        return [f"{where} {verb} {name}; nothing here reads a Verifier file (D89)"]
    if how == "import" and name.lstrip(".").split(".")[0] in DYNAMIC_IMPORT_MODULES:
        return [f"{where} imports {name}; nothing here reaches the module system at runtime, which "
                "is how an import of the Builder would step around this check (D89)"]
    return []


def _verifier_failures(path: Path) -> list[str]:
    """D91's other direction: verifier.py talks to the Runner through records and cli, never its internals."""
    where = "builder/verifier.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, ValueError, OSError) as exc:
        return [f"{where} cannot be parsed, so the D91 boundary cannot be checked on it: {exc}"]
    out = []
    for node in ast.walk(tree):
        for name, _how in _imported_names(node):
            if _is_runner_internal(name):
                out.append(f"{where} imports {name}; verifier.py asks the Runner for Runs through cli "
                           "and reads records back, it never imports Runner internals (D91)")
    return sorted(set(out))


def _callee(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _string_args(node: ast.Call) -> list[str]:
    values = list(node.args) + [keyword.value for keyword in node.keywords]
    return [v.value for v in values if isinstance(v, ast.Constant) and isinstance(v.value, str)]


def _imported_names(node: ast.AST) -> list[tuple]:
    if isinstance(node, ast.Import):
        return [(alias.name, "import") for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = "." * (node.level or 0) + (node.module or "")
        return [(base, "import")] + [(f"{base}.{alias.name}", "import") for alias in node.names]
    return []


def _is_builder(name: str) -> bool:
    parts = name.lstrip(".").split(".")
    return parts[0] == "builder" or parts[:2] == ["kullback", "builder"]


def _is_verifier(name: str) -> bool:
    return name.lstrip(".").split(".")[-1] == "verifier"


def _is_runner_internal(name: str) -> bool:
    """True for a Runner execution module (loop, route, verdict, ...), false for a bare package
    name or one of the data modules records/canon carried in from shared/."""
    parts = name.lstrip(".").split(".")
    if parts[0] == "runner":
        tail = parts[1:]
    elif parts[:2] == ["kullback", "runner"]:
        tail = parts[2:]
    else:
        return False
    return bool(tail) and tail[0] not in RUNNER_DATA_MODULES


def _package_hashes(directory: Path) -> dict[str, str]:
    """Every .py file under a package, keyed by its path relative to the package, in sorted order."""
    hashes = {}
    if directory.is_dir():
        for path in sorted(directory.rglob("*.py")):
            hashes[path.relative_to(directory).as_posix()] = content_hash(path.read_text(encoding="utf-8"))
    return hashes


def runner_version(src_root: Union[str, Path], routing_config: Any = None,
                   created_at: Optional[str] = None, confirmed_by: Optional[str] = None) -> RunnerVersion:
    """The content hash of every .py file in kullback/runner/ and the routing config, written by freeze-runner.

    Every file under runner/ is hashed, sorted by its path relative to runner/, rather than a
    hand-kept list of names (D121, the Runner's hash is a package): a file that moves in or out of
    the package moves the hash the same way a line changing inside one of the old three files
    (loop.py, route.py, verdict.py) always did. The gates package is hashed the same way into
    `gates_version`, recorded beside the Runner's hash and never folded into it (D122): a gate
    that changes does not change what executes or grades a Run, and a regrade that names both can
    say which gates accepted an artifact and under which Runner it was graded.
    """
    root = _package_root(src_root)
    hashes = _package_hashes(root / "runner")
    gate_hashes = _package_hashes(root / "gates")
    config_hash = content_hash(routing_config) if routing_config is not None else None
    return RunnerVersion(
        runner_version=content_hash({"files": hashes, "routing_config": config_hash}),
        file_hashes=hashes, routing_config_hash=config_hash,
        gates_version=content_hash({"files": gate_hashes}) if gate_hashes else None,
        gates_file_hashes=gate_hashes,
        created_at=created_at, confirmed_by=confirmed_by,
    )
