"""The confinement gates: a model-written tool body and a model-written constraint predicate are
refused before either runs anywhere in this process (design section 4, section 7).

Two surfaces, one idea. A constraint predicate (or a Verifier atom) is a function over its own
case and may import nothing, so `runner/confinement.py`'s `confine` is its whole check, and the
gate here (`predicate_confinement`) is that primitive stated as a ruling; verdict.py's atom gate
runs the same primitive, which is why it stays in `runner/` and is not repeated. A tool body is a
method of the generated toolkit and legitimately imports `datetime`, `pydantic` or `re`, so its
surface is an import allowlist, a denied-builtin list and a dunder rule of its own
(`source_confinement`, `gate_confined`), checked only on the tool methods because the data model,
the toolkit shim and `DomainDB.load` are code-owned bytes no model wrote.

Both are name checks, not proofs, and every caller states them as such: the subprocess sandbox in
`builder/sandbox.py` reduces the blast radius of a body that gets through, and a real sandbox for
model-written code is still on todo.md. No Run is executed here and no model is called (D122).
"""

from __future__ import annotations

import ast
import builtins
import importlib
import re
import types
from typing import Iterator, Optional

from kullback.runner.confinement import DENIED_ATTRS, DENIED_NAMES, confine
from kullback.runner.records import GateResult

TOOLS_CLASS = "DomainTools"
# What a generated module may name. The skeleton is code-owned and imports the first four; the rest
# are what a tool body plausibly needs to compute a value. `os`, `sys`, `subprocess`, `socket`,
# `pathlib`, `importlib` and everything else are not on it.
ALLOWED_IMPORTS = frozenset({"typing", "pydantic", "tau2", "data_model", "datetime", "decimal",
                             "math", "json", "re", "copy", "collections", "itertools", "functools",
                             "string", "statistics", "random", "time", "uuid"})
# Everything a predicate may not name, and three more a tool body may not: `exit` and `quit` stop the
# Runner's process, and `memoryview` hands out the bytes behind an object the body was given.
DENIED_BUILTINS = DENIED_NAMES | frozenset({"exit", "quit", "memoryview"})
# The dunders the code-owned skeleton itself writes; every other one is an object walk.
ALLOWED_DUNDERS = frozenset({"__init__", "__tool_type__", "__doc__", "__name__"})
# What a generated class may call while it is being created: pydantic's field declarations and the
# empty containers their defaults are built from.
DECLARATION_CALLS = frozenset({"Field", "ConfigDict", "dict", "list", "set", "tuple"})
# A dunder spelled inside a string constant, which no attribute node reports.
DUNDER_IN_TEXT = re.compile(r"__\w+__")
# A ruling names the first few offences; the body's author reads them, not a ledger.
MAX_NAMED_FAILURES = 5


# --- a constraint predicate ---

def predicate_confinement(source: str) -> list[str]:
    """Everything a model-written constraint predicate names that reaches outside its own case.

    Restricting `__builtins__` is not enough on its own: `check.__globals__` hands the predicate its
    caller's globals and `().__class__.__base__.__subclasses__()` walks every loaded class, so a
    predicate could read or write anything the process can. policy.py certifies a predicate the same
    way when it compiles one, but a Constraint can reach the policy gate from disk, so it is
    certified again there. runner/confinement.py holds the actual check, the same one
    runner/verdict.py's atom gate uses, so the two cannot drift.
    """
    return confine(source)


def predicate_confinement_gate(source: str, constraint_id: str = "?") -> GateResult:
    """`predicate_confinement` as a ruling: the predicate may run only when nothing is named."""
    failures = [f"{constraint_id}: {line}" for line in predicate_confinement(source)]
    return GateResult(stage="compile_policy.confined", passed=not failures,
                      metrics={"chars": len(source)}, failures=failures[:MAX_NAMED_FAILURES])


# --- a tool body ---

def source_confinement(source: str, class_name: str = TOOLS_CLASS) -> list[str]:
    """Everything a model-written tool body names that reaches outside the customer's world.

    The Builder's gates run a body in the subprocess sandbox, but `load_toolkit` executes the same
    module in the Runner's own process, where a body that opens a file, imports `os` or walks
    `().__class__.__base__.__subclasses__()` runs with the Runner's rights. A real sandbox for
    model-written tool code is deferred (design section 4, "Deliberately absent"), so this is the
    static check that stands in for it: an import outside the allowlist, a denied builtin and a
    dunder attribute are refused before the module is executed anywhere. It is a name check, not a
    proof; it is stated as one.

    Only the tool methods are checked. The data model, the toolkit shim and `DomainDB.load` are
    code-owned: they are the same bytes for every customer and no model wrote them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"does not parse: {exc.msg}"]
    bad: list[str] = module_shape(tree)
    # The imports the generated module makes at the top are in scope inside every body, so a body
    # can read `json.codecs` without importing json itself. The module's own aliases are collected
    # once and handed to each body.
    aliases = _module_aliases_at_top(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name == "__init__":
                continue
            bad += [f"{member.name} {line}" for line in _body_confinement(member, aliases)]
    return sorted(set(bad))


def module_shape(tree: ast.AST) -> list[str]:
    """Everything in the module that is not a shape the Harness itself writes.

    The name checks below read the tool methods, because those are what a model wrote. That left a
    hole a review walked through: the module also carries mined table and column names, and a name
    carrying a newline and a statement becomes a statement, at class-body or module level, where no
    method is looked at. `compile_env.unsafe_names` stops that at the point the source is written;
    this is the same rule stated where the source is about to be executed, so a module that arrives
    by any other path is refused too.

    The shapes the Harness writes are few: imports, class definitions, the two try/except blocks
    that prefer tau2's own bases, annotated fields, methods and docstrings. Anything else, an
    expression statement, a bare call, a loop, an assignment outside those blocks, is not something
    this codebase emits, and is refused rather than executed.
    """
    bad: list[str] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) or _is_docstring(node):
            pass
        elif isinstance(node, ast.Try):
            bad += _import_fallback_shape(node)
        else:
            bad.append(f"module runs {_shape_of(node)} outside any class")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if not _is_class_member(member):
                    bad.append(f"{node.name} declares {_shape_of(member)} in its class body")
                elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                    bad += [f"{node.name} computes a field: {line}" for line in _field_shape(member)]
    return sorted(set(bad))


def _field_shape(node: ast.AST) -> list[str]:
    """A field of a generated class is a declaration, never a computation.

    The module carries no `from __future__ import annotations`, on purpose, so both halves run when
    the class is created: the annotation and the default. The Harness writes only
    `name: Optional[Any] = Field(default=None)` and the dict of tables, so anything that is not a
    literal, a name, a subscript of those or a call to Field is refused.
    """
    parts = [getattr(node, "annotation", None), node.value]
    return [f"{ast.unparse(part)[:60]} is not a declaration"
            for part in parts if part is not None and not _declaration(part)]


def _declaration(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) or isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _declaration(node.value)
    if isinstance(node, ast.Subscript):
        return _declaration(node.value) and _declaration(node.slice)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_declaration(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(_declaration(item) for item in list(node.keys) + list(node.values) if item is not None)
    if isinstance(node, ast.Call):
        named = node.func.id if isinstance(node.func, ast.Name) else None
        return named in DECLARATION_CALLS \
            and all(_declaration(arg) for arg in node.args) \
            and all(_declaration(kw.value) for kw in node.keywords)
    return False


def _import_fallback_shape(node: ast.Try) -> list[str]:
    """The `try: from tau2... except ImportError:` blocks the skeleton opens with, and nothing else."""
    inside = list(node.body) + [stmt for handler in node.handlers for stmt in handler.body] \
        + list(node.orelse) + list(node.finalbody)
    allowed = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.Assign, ast.AnnAssign, ast.Pass)
    return [f"module runs {_shape_of(stmt)} in a try block"
            for stmt in inside if not isinstance(stmt, allowed)]


def _is_class_member(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign, ast.Assign,
                             ast.ClassDef, ast.Pass)) or _is_docstring(node)


def _is_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)


def _shape_of(node: ast.AST) -> str:
    return type(node).__name__


def _module_aliases_at_top(tree: ast.AST) -> dict[str, str]:
    """The module's own top-level imports, which every body can name."""
    top = [node for node in getattr(tree, "body", []) if isinstance(node, (ast.Import, ast.ImportFrom))]
    out: dict[str, str] = {}
    for node in top:
        out.update(_module_aliases(node))
    return out


def _body_confinement(function: ast.AST, aliases: Optional[dict[str, str]] = None) -> list[str]:
    out: list[str] = []
    aliases = dict(aliases or {})
    aliases.update(_module_aliases(function))
    docstring = _docstring_node(function)
    for node in ast.walk(function):
        for name in _imported(node):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"imports {name}")
        if isinstance(node, ast.Attribute):
            out += _attribute_confinement(node, aliases)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not docstring:
            # A dunder written as text, not as an attribute node: "{0.__class__}".format(x) walks
            # the object at runtime and the attribute rule above never sees it. `format` is refused
            # a few lines up, so this is the second lock on the same door: nothing in a tool body
            # has a reason to spell a dunder inside a string.
            out += [f"names {name} inside a string" for name in _dunders_in(node.value)]
        elif isinstance(node, ast.Name) and node.id in DENIED_BUILTINS:
            out.append(f"uses {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # A tool body keeps its state in the customer's database and in its own locals. A name
            # it declares global or nonlocal is state that outlives the call and is shared with
            # every other call, which is the module-level mutation the Runner replays badly and
            # the sandbox cannot see. Refused by name, which also settles where such a name would
            # have been resolved from.
            out += [f"declares {'global' if isinstance(node, ast.Global) else 'nonlocal'} {name}"
                    for name in node.names]
    return out


def _attribute_confinement(node: ast.Attribute, aliases: dict[str, str]) -> list[str]:
    """What one attribute access reaches: a dunder, a private name, a format walk, or a module.

    The last is the one the first live builds never showed and a review found: `uuid`, `random`,
    `statistics`, `typing` and `collections` are all on the import allowlist and every one of them
    re-exports another module as a plain attribute, so `uuid.os` hands a body the operating system
    through a name the allowlist blesses. Refusing the shape rather than the name closes the family:
    an attribute of an allowed module that is itself a module is refused, whatever it is called.
    """
    if node.attr.startswith("__"):
        return [] if node.attr in ALLOWED_DUNDERS else [f"touches {node.attr}"]
    if node.attr.startswith("_"):
        # The predicate check has refused single-underscore attributes since it was written. A tool
        # body reads the customer's fields, which are public names, so it loses nothing by the rule.
        return [f"touches {node.attr}"]
    if node.attr in DENIED_ATTRS:
        return [f"uses {node.attr}"]
    module = _module_behind(node, aliases)
    if module is not None:
        return [f"reaches the {node.attr} module through {module}"]
    return []


def _module_behind(node: ast.Attribute, aliases: dict[str, str]) -> Optional[str]:
    """The import this attribute reads another module out of, or None.

    Only names bound by an import in this module are resolved, and only names on the allowlist are
    imported to ask. A module the check cannot import (`tau2`, `data_model`, a missing extra)
    answers None rather than refusing a body over the checker's own environment.
    """
    if not isinstance(node.value, ast.Name):
        return None
    imported = aliases.get(node.value.id)
    if imported is None or imported.split(".")[0] not in ALLOWED_IMPORTS:
        return None
    try:
        module = importlib.import_module(imported)
    except Exception:
        return None
    return imported if isinstance(getattr(module, node.attr, None), types.ModuleType) else None


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Every name an import binds here, mapped to the module it names."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = alias.name.split(".")[0] \
                    if alias.asname is None else alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def _docstring_node(function: ast.AST) -> Optional[ast.AST]:
    body = getattr(function, "body", None) or []
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        return first.value
    return None


def _dunders_in(text: str) -> list[str]:
    return sorted({name for name in DUNDER_IN_TEXT.findall(text) if name not in ALLOWED_DUNDERS})


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


# --- a name the body loads that nothing binds ---

def unbound_names(source: str, class_name: str = TOOLS_CLASS) -> list[str]:
    """Every name a tool method loads that neither the method, the module nor Python binds.

    Such a name is a NameError waiting for the first call that reaches its line. The gates that run
    the recorded calls only see the lines those calls reach: build 8's `calculate` passed them and
    then raised `NameError: decimal` on 49 live results and 33 replayed reads, from a branch the
    shown calls never took. A module on the allowed list is named with the one-line fix.

    Scopes are read the way Python reads them. A name bound inside a nested function, lambda, class
    body or comprehension is not bound in the method around it, so a method that binds `total` only
    inside a comprehension and then returns `total` is still a NameError; counting a child scope's
    bindings as the parent's would let the gate pass the shape it exists to catch.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the parses gate rules on that
    known = set(dir(builtins)) | {"__builtins__"} | _module_bindings(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for name in sorted(_unbound_in_scope(member, known)):
                hint = f"; put `import {name}` at the top of the body" if name in ALLOWED_IMPORTS else ""
                out.append(f"{member.name} names {name}, which nothing binds{hint}")
    return out


# A scope of its own: what it binds is invisible to the code around it.
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _module_bindings(tree: ast.Module) -> set[str]:
    """Names bound at module level, the classes and functions named but not entered.

    A top-level statement's own body counts, since a try is no scope of its own: the generated
    skeleton binds `ToolType` and `is_tool` from tau2 inside a try and falls back to definitions of
    its own in the except. What a method assigns does not count, or every name any body binds would
    read as bound in every other body.
    """
    return _scope_bindings(tree)


def _unbound_in_scope(scope: ast.AST, outer: set[str]) -> set[str]:
    """The names this scope, and the scopes inside it, load without anything binding them."""
    bound = outer | _scope_bindings(scope)
    own = list(_own_nodes(scope))
    unbound = {node.id for node in own
               if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound}
    # A class body is not a closure scope: a method inside it does not see the names the class body
    # bound, only the names around the class. Passing `bound` down here would accept a method that
    # loads a class attribute unqualified, which raises NameError the moment it is called.
    inherited = outer if isinstance(scope, ast.ClassDef) else bound
    for child in own:
        if isinstance(child, SCOPES):
            unbound |= _unbound_in_scope(child, inherited)
    return unbound


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node of this scope, the nested scopes named but not entered."""
    queue = list(ast.iter_child_nodes(scope))
    while queue:
        node = queue.pop(0)
        yield node
        if not isinstance(node, SCOPES):
            queue += list(ast.iter_child_nodes(node))


def _scope_bindings(scope: ast.AST) -> set[str]:
    """What this one scope binds: parameters, assignment, loop, with and except targets,
    comprehension variables, its own imports, and the functions and classes it defines by name.

    A `global x` or `nonlocal x` declaration binds nothing by itself: it says where a later
    assignment lands. A body that declares one and then only loads the name still raises NameError,
    so the declaration is not counted and the load is reported. An assignment after it is counted
    the way any other Store name is.
    """
    bound: set[str] = set()
    for node in _own_nodes(scope):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def gate_confined(source: str, class_name: str = TOOLS_CLASS) -> GateResult:
    """Every name a tool body loads is one it may name and one something binds.

    Two static rules over the same names, both about what the body may say before it runs anywhere:
    nothing reaches past the customer's world (`source_confinement`), and nothing is loaded that
    neither the body, the module nor Python binds (`unbound_names`). The gate keeps one ruling
    because the fix is the same shape either way: the offending name, and what to do about it.
    """
    failures = source_confinement(source, class_name) + unbound_names(source, class_name)
    return GateResult(stage="confined", passed=not failures,
                      metrics={"chars": len(source), "unbound": len(unbound_names(source, class_name))},
                      failures=failures[:MAX_NAMED_FAILURES])
