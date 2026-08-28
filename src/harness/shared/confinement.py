"""The one AST-based check that certifies a model-written or model-checked predicate cannot reach
outside its own case, shared by verdict.py's atoms and validate.py's constraints."""

from __future__ import annotations

import ast
import builtins

# The same names policy.py certifies for a compiled predicate at build time, so a predicate that
# passed its positive and negative test cannot NameError here and be skipped as a broken atom or
# constraint. tests pin every copy of this list, wherever it is imported, as a superset of
# policy.py's own allowed set.
SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
    "max", "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "KeyError", "TypeError", "ValueError",
)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
# The build-time gate policy.py runs, repeated here because a predicate can reach this check from
# disk or from a Builder that edited itself (D69), so the only gate that counts is this one.
DENIED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr",
})
DENIED_ATTRS = frozenset({"format", "format_map"})  # "{0.__class__}".format(x) is an attribute walk


def confine(source: str) -> list[str]:
    """Everything a predicate names that would reach outside its own case, or a parse error.

    Restricting `__builtins__` is not enough on its own: `check.__globals__` hands the predicate
    its caller's globals, and `().__class__.__base__.__subclasses__()` walks every loaded class, so
    an unconfined predicate could read or write anything the process can. This is a name check, not
    a proof, and every caller states it as one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"does not parse: {error.msg}"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad.append("imports a module")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENIED_ATTRS:
                bad.append(f"touches {node.attr}")
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            bad.append(f"uses {node.id}")
    return sorted(set(bad))
