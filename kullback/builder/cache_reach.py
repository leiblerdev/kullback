"""The import closure of a builder module as one hash (G29).

A caller that caches work keyed on code (the world build in `world_tools.py`) hashes the
modules it delegates to whole through `closure_hash`, so an edit anywhere in what a hashed
module imports moves the hash and no key is served entries from before the change. An import
over-approximates reach (an import is not a call), so counts from this are the upper bound.
"""

from __future__ import annotations

import ast
from pathlib import Path

BUILD_PATH = Path(__file__).with_name("world_tools.py")

FIRST_PARTY = "kullback."

def _repo_root() -> Path:
    return BUILD_PATH.resolve().parent.parent.parent


def _resolve_import(package: str, name: str) -> str:
    """The module a `from package import name` binding points at.

    A member that is itself a submodule resolves to the submodule; anything else
    (a function, class or constant defined in the package) resolves to the package.
    """
    candidate = f"{package}.{name}" if package else name
    if package and _is_submodule(candidate):
        return candidate
    return package


def _dir_entries(path: Path) -> set[str]:
    """Directory entries with exact case: the filesystem may ignore case, the walk must not."""
    import os

    try:
        return set(os.listdir(path))
    except OSError:
        return set()


def _is_submodule(dotted: str) -> bool:
    parts = dotted.split(".")
    base = _repo_root() / parts[0]
    for part in parts[1:]:
        entries = _dir_entries(base)
        if f"{part}.py" in entries:
            base = base / f"{part}.py"
        elif part in entries and "__init__.py" in _dir_entries(base / part):
            base = base / part
        else:
            return False
    return True

EXEMPT: frozenset[str] = frozenset()
# The exemption list is empty on purpose: nothing cleared the bar. Price tables and
# usage accounting were considered and refused, since prices feed ceiling stops and
# usage shapes reach record serialization, and neither is provably output free. The
# confinement and gate modules decide what compiles and what passes. If in doubt,
# do not exempt: the founder rules on additions with the blast radius table.

def _dotted_file(dotted: str) -> Path | None:
    """The file behind a dotted first-party module: the module file or the package init."""
    parts = dotted.split(".")
    if len(parts) == 1:
        init = _repo_root() / parts[0] / "__init__.py"
        return init if init.is_file() else None
    root = _repo_root() / parts[0]
    path = root.joinpath(*parts[1:])
    if (path.with_suffix(".py")).is_file():
        return path.with_suffix(".py")
    if path.is_dir() and (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _package_inits(dotted: str) -> dict[str, Path]:
    """The existing `__init__` files above a module, dotted name to path."""
    parts, root, out = dotted.split("."), _repo_root() / dotted.split(".")[0], {}
    for depth in range(1, len(parts)):
        init = root.joinpath(*parts[1:depth], "__init__.py")
        if init.is_file():
            out[".".join([*parts[:depth], "__init__"])] = init
    return out


def _guard_type_checking(test: ast.AST) -> bool | None:
    """Whether an `if` test is a TYPE_CHECKING guard: True, inverted, or not one."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _guard_type_checking(test.operand)
        return None if inner is None else not inner
    if any(isinstance(n, ast.Name) and n.id == "TYPE_CHECKING" for n in ast.walk(test)):
        return True
    return None


def _live_imports(node: ast.AST) -> list[ast.AST]:
    """Import nodes that execute at runtime: everything except TYPE_CHECKING bodies."""
    found: list[ast.AST] = []

    def visit(item: ast.AST, guarded: bool) -> None:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            if not guarded:
                found.append(item)
            return
        if isinstance(item, ast.If):
            polarity = _guard_type_checking(item.test)
            if polarity is True:
                for stmt in item.orelse:
                    visit(stmt, guarded)
                return
            if polarity is False:
                for stmt in item.body:
                    visit(stmt, guarded)
                return
        for child in ast.iter_child_nodes(item):
            visit(child, guarded)

    visit(node, False)
    return found


def _relative_base(level: int, package: str) -> str:
    """The dotted base a relative import resolves against: the current package, up level by one."""
    base = package
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    return base


def _resolve_from_import(node: ast.ImportFrom, package: str) -> set[str]:
    """The first-party modules one `from` import binds, relative or absolute."""
    level = node.level or 0
    base = _relative_base(level, package) if level else (node.module or "")
    out = set()
    for item in node.names:
        if item.name == "*":
            if base.startswith(FIRST_PARTY):
                out.add(base)
            continue
        target = f"{base}.{item.name}" if base else item.name
        if target.startswith(FIRST_PARTY):
            out.add(_resolve_import(base, item.name))
    return out


def _imported_modules(path: Path, dotted: str) -> set[str]:
    """The first-party modules one file imports, at any level, minus type only guards."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    package = dotted.rpartition(".")[0] or dotted
    out = set()
    for node in _live_imports(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name.startswith(FIRST_PARTY):
                    out.add(item.name)
        elif isinstance(node, ast.ImportFrom):
            out |= _resolve_from_import(node, package)
    return out


_CLOSURE: dict[str, frozenset[str]] = {}


def import_closure(dotted: str) -> frozenset[str]:
    """Every other first-party module one module statically imports, transitively.

    Whole files, function level imports included, `if TYPE_CHECKING:` bodies skipped.
    Memoised per process; callers sort for determinism. An import over-approximates
    reach (an import is not a call), so counts from this are the upper bound.
    """
    if dotted in _CLOSURE:
        return _CLOSURE[dotted]
    seen, frontier = set(), {dotted}
    while frontier:
        current = sorted(frontier)[0]
        frontier.discard(current)
        if current in seen:
            continue
        seen.add(current)
        path = _dotted_file(current)
        if path is not None:
            frontier |= _imported_modules(path, current) - seen
    seen.discard(dotted)
    _CLOSURE[dotted] = frozenset(seen)
    return _CLOSURE[dotted]


def closure_files(dotted: str) -> dict[str, str]:
    """The module, its closure and every package init above each of them, name to bytes."""
    names = sorted({dotted} | set(import_closure(dotted)))
    out: dict[str, str] = {}
    for name in names:
        path = _dotted_file(name)
        if path is not None:
            out[name] = path.read_bytes().decode("utf-8", errors="replace")
        for init_name, init_path in _package_inits(name).items():
            out.setdefault(init_name, init_path.read_bytes().decode("utf-8", errors="replace"))
    return out


def closure_hash(module: object, exempt: frozenset[str] = EXEMPT) -> str:
    """The module and its first-party import closure as one hash, minus exemptions.

    A change to anything the module can call through its imports moves the hash, so a
    stage key built from it cannot be served entries from before the change. Modules
    with no resolvable file fall back to their repr, the old behaviour.

    The module's own name is hashed beside the closure. Two modules that import each
    other have the same closure, so without the name they would carry the same hash and
    a stage key could no longer say which of them it delegates to.
    """
    from kullback.runner.records import content_hash

    name = getattr(module, "__name__", "") or ""
    if not name.startswith(FIRST_PARTY) or _dotted_file(name) is None:
        path = Path(getattr(module, "__file__", "") or "")
        return content_hash(path.read_bytes() if path.is_file() else repr(module))
    import hashlib

    files = {key: value for key, value in closure_files(name).items() if key not in exempt}
    digest = {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in sorted(files.items())}
    return content_hash({"module": name, "closure": digest})
