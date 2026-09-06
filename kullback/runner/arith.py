"""One code-owned arithmetic evaluator, handed to every model-written tool body by name.

A customer's recorded tool sometimes takes an arithmetic expression as a string and answers with
its value. A generated body cannot compute that the way a person would: `eval`, `exec`, `compile`
and `ast` are refused by the confinement gate, for the reason gates/confinement.py states, so the
model hand-writes a parser instead. On build 12 it wrote four of them for one tool, each one
splitting the string on spaces, none of them able to read a parenthesis, and the tool stayed
assisted. A parser is not what the customer's tool is about, and it is the same parser every time,
so it is ours to write once: `evaluate_arithmetic` is bound into the generated module's namespace
by both loaders (builder/compile_env.py's `load_toolkit` and builder/sandbox.py's subprocess
runner), and a body calls it by name without importing anything.

What it is not: a calculator language. There are no names, no calls, no attributes, no comparisons
and no lists, because none of those are arithmetic and every one of them is a way out of a
restricted namespace. The whole grammar is a number, `+ - * / // % **`, parentheses and a leading
sign, over `decimal.Decimal` so a money column is exact to the cent. Everything else raises
ValueError naming the kind of thing it refused, in the words a body's author would use.

Bounded rather than timed: a wall clock in a thread would be a second thing to get wrong, so the
caps do the work instead. The text is capped, an exponent is capped and checked against the size of
its base before the power is computed (`9 ** 9 ** 9` is refused, not attempted), and every
intermediate value is capped by digit count. The context is fixed at 34 digits of precision, so the
same expression has the same answer on every machine and in every process.

One thing to know before reading an answer: `//` here is decimal's, which truncates toward zero,
so it differs from Python's own `//` on negative operands (`-7 // 2` is -3, not -4).
"""

from __future__ import annotations

import ast
import decimal

# The text a body may hand over. Long enough for any expression a recorded call carries, short
# enough that parsing it is never the expensive part.
MAX_EXPRESSION_CHARS = 2000
# An exponent this far from zero, and no further. Checked against the base before the power runs.
MAX_EXPONENT = 64
# How wide any value in the expression may be, counted in digits either side of the point.
MAX_RESULT_DIGITS = 512
# Fixed, so the answer does not depend on whatever context the calling process happens to be in.
PRECISION = 34

_CONTEXT = decimal.Context(prec=PRECISION, Emax=999999, Emin=-999999,
                           traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])

_BINARY_OPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left ** right,
}
_DIVIDING = (ast.Div, ast.FloorDiv, ast.Mod)
# The marker `_value` pushes to say "this node's operands are done, put them together".
_COMBINE = object()

# The operators that are not arithmetic, each in the words a reader would say it in.
_OPERATOR_WORDS = {
    ast.BitOr: "bitwise or", ast.BitAnd: "bitwise and", ast.BitXor: "bitwise exclusive or",
    ast.LShift: "a left shift", ast.RShift: "a right shift", ast.MatMult: "matrix multiplication",
    ast.Not: "not", ast.Invert: "bitwise inversion",
}
# The kinds of expression that are not arithmetic, likewise. Plural, so one sentence fits them all.
_NODE_WORDS = {
    ast.Name: "names", ast.Call: "calls", ast.Attribute: "attributes", ast.Subscript: "subscripts",
    ast.Compare: "comparisons", ast.BoolOp: "boolean operators", ast.IfExp: "conditional expressions",
    ast.Lambda: "lambdas", ast.List: "lists", ast.Tuple: "tuples", ast.Dict: "dictionaries",
    ast.Set: "sets", ast.ListComp: "comprehensions", ast.SetComp: "comprehensions",
    ast.DictComp: "comprehensions", ast.GeneratorExp: "generator expressions",
    ast.JoinedStr: "formatted strings", ast.FormattedValue: "formatted strings",
    ast.Starred: "starred values", ast.NamedExpr: "assignments", ast.Slice: "slices",
    ast.Await: "await expressions", ast.Yield: "yield expressions", ast.YieldFrom: "yield expressions",
}
# What a literal that is not a number is, in the same words.
_CONSTANT_WORDS = {
    bool: "a true or false value", str: "a text value", bytes: "a bytes value",
    type(None): "a missing value", complex: "a complex number", type(...): "an ellipsis",
}

_GRAMMAR = "an arithmetic expression is numbers, + - * / // % ** and parentheses"


def evaluate_arithmetic(expression: str) -> decimal.Decimal:
    """The value of an arithmetic expression, exact to the digit, or ValueError saying what is wrong.

    `evaluate_arithmetic("(2459.74 * 2) - (2291.87 + 2520.52)")` is `Decimal("107.09")`, not a float
    that is nearly that, because every literal enters as `Decimal(str(value))` and the arithmetic is
    decimal throughout. Anything that is not arithmetic is refused by name before it is computed, and
    nothing in the expression can reach a name, a module or an attribute, because the grammar has no
    way to say one.
    """
    if not isinstance(expression, str):
        raise ValueError(f"{_GRAMMAR}, given as text, not as a value of another kind")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"the expression is {len(expression)} characters long and at most "
                         f"{MAX_EXPRESSION_CHARS} are allowed")
    try:
        # Stripped, because a recorded expression carries whatever whitespace the trace had around
        # it and `ast.parse` in eval mode reads a leading space as an indent and refuses the line.
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"the expression does not parse: {exc.msg}") from exc
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ValueError("the expression cannot be read: it is nested too deeply, or holds a "
                         "character that is not text") from exc
    with decimal.localcontext(_CONTEXT):
        try:
            return _value(tree.body)
        except decimal.DivisionByZero as exc:
            raise ValueError("division by zero is not allowed") from exc
        except decimal.Overflow as exc:
            raise ValueError("the result is too large to compute") from exc
        except decimal.DecimalException as exc:
            raise ValueError("the expression asks for a value arithmetic does not have") from exc


def _value(root: ast.AST) -> decimal.Decimal:
    """The tree's value, walked with a stack of its own rather than by recursion.

    A flat sum of five hundred numbers is 1996 characters, well inside the cap, and left-nests five
    hundred BinOps deep. Evaluated recursively that is a RecursionError on a legitimate expression,
    and whose stack it overflows depends on how deep the caller already was, which is exactly the
    kind of answer that differs between the Runner's process and the sandbox subprocess. With a list
    for a stack the only limit is the character cap.

    Each node is pushed twice: once to be read, and once as a `(_COMBINE, node)` marker that runs
    after its operands are on `values`. A node kind outside the grammar is refused when it is read,
    and an operator outside the grammar when its node is read, so nothing is computed first.
    """
    stack: list = [root]
    values: list[decimal.Decimal] = []
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            _combine(node[1], values)
        elif isinstance(node, ast.Constant):
            values.append(_constant(node.value))
        elif isinstance(node, ast.UnaryOp):
            _check_unary(node)
            stack += [(_COMBINE, node), node.operand]
        elif isinstance(node, ast.BinOp):
            _check_binary(node)
            stack += [(_COMBINE, node), node.right, node.left]
        else:
            words = _NODE_WORDS.get(type(node), "expressions of this kind")
            raise ValueError(f"{words} are not allowed: {_GRAMMAR}")
    return values.pop()


def _constant(value: object) -> decimal.Decimal:
    """A literal number as an exact Decimal.

    `type(value) is int` rather than isinstance, because True is an int and `True * 3` is not
    arithmetic anyone wrote on purpose. A float enters through its own repr, so 2459.74 is the
    2459.74 the recording shows and not the binary value nearest to it.

    A literal is bounded like every other value, and a float literal python has already rounded to
    infinity (`1e400`) is refused here: past this point an infinity spreads through the expression
    without tripping a single decimal trap, and the caller gets `Infinity` back as if it were an
    answer.
    """
    if type(value) is int or type(value) is float:
        number = decimal.Decimal(str(value))
        if not number.is_finite():
            raise ValueError("that number is too large to write as a decimal, so it is refused")
        return _bounded(number)
    words = _CONSTANT_WORDS.get(type(value), "a value of this kind")
    raise ValueError(f"{words} is not allowed: {_GRAMMAR}")


def _check_unary(node: ast.UnaryOp) -> None:
    """A sign is arithmetic; `not` and `~` are not."""
    if not isinstance(node.op, (ast.UAdd, ast.USub)):
        words = _OPERATOR_WORDS.get(type(node.op), "this operator")
        raise ValueError(f"{words} is not allowed: {_GRAMMAR}")


def _check_binary(node: ast.BinOp) -> None:
    """The seven operators of the grammar, and nothing shaped like one."""
    if type(node.op) not in _BINARY_OPS:
        words = _OPERATOR_WORDS.get(type(node.op), "this operator")
        raise ValueError(f"{words} is not allowed: {_GRAMMAR}")


def _combine(node: ast.AST, values: list) -> None:
    """Put this node's operands together, now that `_value` has left them on `values`."""
    if isinstance(node, ast.UnaryOp):
        value = values.pop()
        values.append(+value if isinstance(node.op, ast.UAdd) else -value)
        return
    right, left = values.pop(), values.pop()
    if isinstance(node.op, _DIVIDING) and right == 0:
        # Named here rather than left to the decimal traps, so dividing, floor dividing and taking a
        # remainder by zero all read back the same plain sentence.
        raise ValueError("division by zero is not allowed")
    if isinstance(node.op, ast.Pow):
        _check_power(left, right)
    values.append(_bounded(_BINARY_OPS[type(node.op)](left, right)))


def _check_power(base: decimal.Decimal, exponent: decimal.Decimal) -> None:
    """Refuse a power before computing it, on the exponent and on how wide the base is.

    Both halves are needed. The exponent cap alone still allows a hundred-digit literal raised to
    the 64th, and the digit cap alone would only notice `9 ** 9 ** 9` after asking decimal for a
    number with 370 million digits. Checked here, the outer power of `9 ** 9 ** 9` is refused on an
    exponent of 387420489 and nothing large is ever built.
    """
    if not exponent.is_finite() or abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"an exponent may be at most {MAX_EXPONENT} away from zero, and this one "
                         "is further, so the power is refused before it is computed")
    digits = max(base.adjusted() + 1, 1) if base.is_finite() else 1
    if digits * max(int(abs(exponent)), 1) > MAX_RESULT_DIGITS:
        raise ValueError(f"this power would run to more than {MAX_RESULT_DIGITS} digits, so it is "
                         "refused before it is computed")


def _bounded(value: decimal.Decimal) -> decimal.Decimal:
    """The value, or ValueError when it has grown past the digit cap either side of the point."""
    if value.is_finite() and value != 0 and abs(value.adjusted()) > MAX_RESULT_DIGITS:
        raise ValueError(f"the result runs to more than {MAX_RESULT_DIGITS} digits, which is further "
                         "than this evaluator will go")
    return value
