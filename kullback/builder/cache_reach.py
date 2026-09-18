"""Which first-party modules each builder stage delegates to, and what its key hashes.

G29: a stage that calls into a module its cache key does not hash keeps a stale cache entry
when that module changes. This module walks each stage function with ast and reports the
reachable first-party modules beside the modules the stage version hashes, so the test can
require that nothing reachable is missing.

What the walk counts, and what it honestly misses:
- Only names in executable positions count. Annotations are skipped (the file uses
  `from __future__ import annotations`), as are comments and docstrings.
- Names bound in the stage closure (its arguments, loop variables, the model and search
  objects handed to the factory) are values, not code, and are skipped.
- The walk follows transitively through module level functions defined in build.py itself
  (the helpers a stage body calls) and through the methods of classes the factory
  instantiates (the semantic judging the replay stage closes over). Calls into other
  modules are covered by hashing those modules whole; the walk stops at the package
  boundary and never follows into another module's own imports.
- The gate closure is walked with the run closure. A gate re-runs on every attempt,
  including a cache hit, so a gate only callee cannot serve stale outputs; hashing it
  all the same is the over counting G29 accepts as visible cost.
- Factory time code (the version expression itself and the helpers it calls to build the
  key) runs in the current process when the graph is built. A change there computes a
  different key and misses the cache rather than hitting a stale entry, so those
  references are not reachability.
- Dynamic dispatch stays invisible: getattr with a computed name, registries keyed by
  string, and lazy imports the walk cannot resolve. The test pins known delegations per
  stage so a silent miss fails loudly instead of passing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

BUILD_PATH = Path(__file__).with_name("build.py")

STAGE_FACTORIES: dict[str, str] = {
    "ingest": "_ingest_stage",
    "mine": "_mine_stage",
    "readers": "_readers_stage",
    "cluster": "_cluster_stage",
    "canon_rules": "_canon_stage",
    "starting_state": "_state_stage",
    "compile_tools": "_tools_stage",
    "compile_policy": "_policy_stage",
    "judge_lessons": "_lessons_stage",
    "vocabulary": "_vocabulary_stage",
    "user_rules": "_user_rules_stage",
    "build_environment": "_environment_stage",
    "replay_reference": "_replay_stage",
    "intent": "_intent_stage",
    "rerolls": "_rerolls_stage",
}

FIRST_PARTY = "kullback."


@dataclass
class Reachability:
    """What one stage can execute: first-party modules, and own file helpers by name."""

    modules: set[str] = field(default_factory=set)
    helpers: set[str] = field(default_factory=set)


def _repo_root() -> Path:
    return BUILD_PATH.resolve().parent.parent.parent


def _is_submodule(dotted: str) -> bool:
    parts = dotted.split(".")
    base = _repo_root() / parts[0]
    for part in parts[1:]:
        if (base / f"{part}.py").is_file() or (base / part / "__init__.py").is_file():
            base = base / part if (base / part).is_dir() else base
            continue
        return False
    return True


def _resolve_import(package: str, name: str) -> str:
    """The module a `from package import name` binding points at.

    A member that is itself a submodule resolves to the submodule; anything else
    (a function, class or constant defined in the package) resolves to the package.
    """
    candidate = f"{package}.{name}" if package else name
    if package and _is_submodule(candidate):
        return candidate
    return package


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names bound by module level kullback imports, to dotted modules."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name.startswith(FIRST_PARTY):
                    out[(item.asname or item.name).split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(FIRST_PARTY):
            for item in node.names:
                out[item.asname or item.name] = _resolve_import(node.module or "", item.name)
    return out


def _own_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _own_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


class _FreeNames(ast.NodeVisitor):
    """Free names under a node: loaded but bound in no enclosing function scope.

    Modules reached through imports (at any level, including function bodies) land in
    `modules`; names of module level build.py functions land in `helpers`; names of
    module level classes land in `classes`. Attribute roots resolve the same way, so
    `memory.lesson_for` reaches the memory module through the `memory` root.
    """

    def __init__(self, aliases: dict[str, str], own: dict[str, str]) -> None:
        self.aliases = dict(aliases)
        self.own = own
        self.bound: list[set[str]] = [set()]
        self.modules: set[str] = set()
        self.helpers: set[str] = set()
        self.classes: set[str] = set()

    def _is_bound(self, name: str) -> bool:
        return any(name in scope for scope in self.bound)

    def _bind(self, target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.bound[-1].add(node.id)
            elif isinstance(node, ast.arg):
                self.bound[-1].add(node.arg)

    def _note(self, name: str) -> None:
        if self._is_bound(name):
            return
        if name in self.aliases:
            self.modules.add(self.aliases[name])
        elif name in self.own:
            if self.own[name] == "def":
                self.helpers.add(name)
            else:
                self.classes.add(name)

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.name.startswith(FIRST_PARTY):
                self.modules.add(item.name)
                self.bound[-1].add((item.asname or item.name).split(".")[0])
            else:
                self.bound[-1].add((item.asname or item.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for item in node.names:
            self.bound[-1].add(item.asname or item.name)
        if (node.module or "").startswith(FIRST_PARTY):
            for item in node.names:
                resolved = _resolve_import(node.module or "", item.name)
                self.modules.add(resolved)
                self.aliases[item.asname or item.name] = resolved

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._note(node.id)
        else:
            self.bound[-1].add(node.id)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in (*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]):
            self.visit(default)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.bound[-1].add(node.name)
        self.bound.append(set())
        for arg in (*node.args.args, *node.args.kwonlyargs):
            self.bound[-1].add(arg.arg)
        for extra in (node.args.vararg, node.args.kwarg):
            if extra is not None:
                self.bound[-1].add(extra.arg)
        for stmt in node.body:
            self.visit(stmt)
        self.bound.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.bound.append(set())
        for arg in (*node.args.args, *node.args.kwonlyargs):
            self.bound[-1].add(arg.arg)
        self.visit(node.body)
        self.bound.pop()

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        self.bound.append(set())
        for gen in node.generators:
            self.visit(gen.iter)
            self._bind(gen.target)
            for condition in gen.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.bound.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.bound[-1].add(node.target.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._bind(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind(node.target)


def _free_under(node: ast.AST, aliases: dict[str, str], own: dict[str, str]) -> _FreeNames:
    found = _FreeNames(aliases, own)
    found.visit(node)
    return found


def _build_tree() -> ast.Module:
    return ast.parse(BUILD_PATH.read_text(encoding="utf-8"))


def _factory(tree: ast.Module, factory: str) -> ast.FunctionDef:
    node = _own_functions(tree).get(factory)
    if node is None:
        raise KeyError(f"no factory {factory} in {BUILD_PATH.name}")
    return node


def _closures(factory_node: ast.FunctionDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in factory_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ("run", "gate")]


def _method_bodies(tree: ast.Module, cls: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    classes = _own_classes(tree)
    if cls not in classes:
        return []
    return [n for n in classes[cls].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _own_map(tree: ast.Module) -> dict[str, str]:
    own = {name: "def" for name in _own_functions(tree)}
    own.update({name: "class" for name in _own_classes(tree)})
    return own


@dataclass
class _Walk:
    """The moving state of one reachability walk: aliases learned, names queued."""

    tree: ast.Module
    aliases: dict[str, str]
    own: dict[str, str]
    out: Reachability = field(default_factory=Reachability)
    pending: set[str] = field(default_factory=set)

    def absorb(self, found: _FreeNames) -> None:
        self.out.modules |= found.modules
        for key, value in found.aliases.items():
            self.aliases.setdefault(key, value)
        self.pending |= found.helpers
        for cls in found.classes:
            for method in _method_bodies(self.tree, cls):
                self.pending.add(f"{cls}.{method.name}")

    def drain(self) -> None:
        done: set[str] = set()
        while self.pending - done:
            name = sorted(self.pending - done)[0]
            done.add(name)
            body = _helper_body(self.tree, name)
            if body is not None:
                self.absorb(_free_under(body, self.aliases, self.own))
        self.out.helpers |= set(done)


def reachable(factory: str, tree: ast.Module | None = None) -> Reachability:
    """The first-party modules and own file helpers one stage can execute.

    Starts at the stage run and gate closures, then follows transitively through
    module level helpers of build.py and through methods of classes the factory
    instantiates.
    """
    tree = _build_tree() if tree is None else tree
    walk = _Walk(tree, _module_aliases(tree), _own_map(tree))
    factory_node = _factory(tree, factory)
    closures = _closures(factory_node)
    if not any(n.name == "run" for n in closures):
        raise KeyError(f"factory {factory} defines no run closure")
    for closure in closures:
        walk.absorb(_free_under(closure, walk.aliases, walk.own))
        for stmt in factory_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for cls in _free_under(stmt, walk.aliases, walk.own).classes:
                for method in _method_bodies(tree, cls):
                    walk.pending.add(f"{cls}.{method.name}")
        walk.drain()
    return walk.out


def _helper_body(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if "." in name:
        cls, _, method = name.partition(".")
        for candidate in _method_bodies(tree, cls):
            if candidate.name == method:
                return candidate
        return None
    return _own_functions(tree).get(name)


def _version_expression(factory_node: ast.FunctionDef) -> ast.AST | None:
    """The expression behind the factory code_version keyword, with `version` resolved."""
    assigned: dict[str, ast.AST] = {}
    for stmt in factory_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assigned[stmt.targets[0].id] = stmt.value
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            for key in stmt.value.keywords:
                if key.arg == "code_version":
                    return _resolve_key(assigned, key.value)
    return None


def _resolve_key(assigned: dict[str, ast.AST], value: ast.AST) -> ast.AST:
    """The full run key expression: a name, or the full run branch of a narrowed ternary."""
    if isinstance(value, ast.Name) and value.id in assigned:
        return assigned[value.id]
    if isinstance(value, ast.IfExp):
        for branch in (value.body, value.orelse):
            if isinstance(branch, ast.Name) and branch.id in assigned:
                return assigned[branch.id]
    return value


def _hash_call(tree: ast.Module, node: ast.Call, aliases: dict[str, str], own: dict[str, str], out: Reachability) -> None:
    """Credit one call inside a key expression: a module hash, a version list, or a key builder."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "_module_hash" and node.args:
        out.modules.add(_call_module(node.args[0], aliases))
    elif isinstance(func, ast.Name) and func.id == "_version":
        _hash_version_call(node, aliases, own, out)
    elif isinstance(func, ast.Name) and func.id in own and own[func.id] == "def":
        out.helpers |= _committed_helpers(tree, func.id, aliases, own)
    elif isinstance(func, ast.Attribute) and func.attr == "_fn_identity":
        _hash_identity_call(node, own, out)


def _hash_version_call(node: ast.Call, aliases: dict[str, str], own: dict[str, str], out: Reachability) -> None:
    """Credit a `_version` call: the modules after the name and function, and the helpers list."""
    for arg in node.args[2:]:
        out.modules.add(_call_module(arg, aliases))
    for key in node.keywords:
        if key.arg == "helpers":
            out.helpers |= _helper_names(key.value, own)


def _hash_identity_call(node: ast.Call, own: dict[str, str], out: Reachability) -> None:
    """Credit helpers named directly in an `_fn_identity` call."""
    for arg in node.args:
        for child in ast.walk(arg):
            if isinstance(child, ast.Name) and child.id in own and own[child.id] == "def":
                out.helpers.add(child.id)


def hashed(factory: str, tree: ast.Module | None = None) -> Reachability:
    """The modules and helpers the factory code_version expression commits to.

    Reads the `_version` and `_module_hash` arguments, the `helpers` list, helpers
    named in `_fn_identity` calls, and the helpers a factory time key builder such
    as `_evidence_version` hashes into its own output.
    """
    tree = _build_tree() if tree is None else tree
    aliases = _module_aliases(tree)
    own = _own_map(tree)
    out = Reachability()
    expression = _version_expression(_factory(tree, factory))
    if expression is None:
        return out
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            _hash_call(tree, node, aliases, own, out)
    out.modules.discard("")
    return out


def _call_module(arg: ast.AST, aliases: dict[str, str]) -> str:
    node: ast.AST | None = arg
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id, "")
    return ""


def _helper_names(value: ast.AST, own: dict[str, str]) -> set[str]:
    out = set()
    if isinstance(value, (ast.List, ast.Tuple)):
        for elt in value.elts:
            ref = _helper_ref(elt, own)
            if ref:
                out.add(ref)
    return out


def _helper_ref(elt: ast.AST, own: dict[str, str]) -> str:
    """An own file helper named by a `helpers` entry: a function, or a class method."""
    if isinstance(elt, ast.Name) and elt.id in own and own[elt.id] == "def":
        return elt.id
    node, parts = elt, []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in own and own[node.id] == "class":
        dotted = node.id + "." + ".".join(reversed(parts))
        return dotted[: -len(".fget")] if dotted.endswith(".fget") else dotted
    return ""


def _committed_helpers(tree: ast.Module, helper: str, aliases: dict[str, str], own: dict[str, str]) -> set[str]:
    """The own file helpers a factory time key builder hashes into its output."""
    node = _own_functions(tree).get(helper)
    if node is None:
        return set()
    return set(_free_under(node, aliases, own).helpers)


def evidence_commit(tree: ast.Module | None = None) -> set[str]:
    """The helpers `_evidence_version` hashes into the compile tools key."""
    tree = _build_tree() if tree is None else tree
    return _committed_helpers(tree, "_evidence_version", _module_aliases(tree),
                              {**{n: "def" for n in _own_functions(tree)},
                               **{n: "class" for n in _own_classes(tree)}})


def missing(factory: str, tree: ast.Module | None = None) -> Reachability:
    """Reachable modules and helpers the factory key does not hash."""
    tree = _build_tree() if tree is None else tree
    reached, hashed_sets = reachable(factory, tree), hashed(factory, tree)
    return Reachability(modules=reached.modules - hashed_sets.modules,
                        helpers=reached.helpers - hashed_sets.helpers)
