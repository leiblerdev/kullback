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
from typing import Iterator

from kullback.runner.confinement import DENIED_NAMES, confine
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
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name == "__init__":
                continue
            bad += [f"{member.name} {line}" for line in _body_confinement(member)]
    return sorted(set(bad))


def _body_confinement(function: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(function):
        for name in _imported(node):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"imports {name}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") \
                and node.attr not in ALLOWED_DUNDERS:
            out.append(f"touches {node.attr}")
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
