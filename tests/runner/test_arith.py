"""The code-owned arithmetic evaluator a tool body is handed by name (kullback.runner.arith)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from kullback.runner.arith import (
    MAX_EXPONENT,
    MAX_EXPRESSION_CHARS,
    MAX_RESULT_DIGITS,
    evaluate_arithmetic,
)

# --- the arithmetic a recorded tool actually asks for ---

def test_parentheses_beat_precedence_and_precedence_beats_left_to_right():
    assert evaluate_arithmetic("2 + 3 * 4") == Decimal("14")
    assert evaluate_arithmetic("(2 + 3) * 4") == Decimal("20")
    assert evaluate_arithmetic("100 - (20 - 5)") == Decimal("85")
    assert evaluate_arithmetic("2 ** 3 ** 2") == Decimal("512"), "a power groups to the right"
    assert evaluate_arithmetic("  \n 42 \t ") == Decimal("42")
    assert evaluate_arithmetic("(" * 40 + "1 + 1" + ")" * 40) == Decimal("2")


def test_a_money_expression_is_exact_to_the_cent_where_floats_would_drift():
    """The shape build 12 could not compute: a difference of two sums, read off a recording.

    Every literal enters as Decimal(str(value)), so the answer is the one a person adding the
    column would write, not the float nearest to it.
    """
    assert evaluate_arithmetic("(2459.74 * 2) - (2291.87 + 2520.52)") == Decimal("107.09")
    assert evaluate_arithmetic("0.1 + 0.2") == Decimal("0.3")
    assert float(evaluate_arithmetic("0.1 + 0.2")) != 0.1 + 0.2


def test_every_operator_the_grammar_names_computes():
    assert evaluate_arithmetic("7 / 2") == Decimal("3.5")
    assert evaluate_arithmetic("7 // 2") == Decimal("3")
    assert evaluate_arithmetic("7 % 2") == Decimal("1")
    assert evaluate_arithmetic("2 ** 10") == Decimal("1024")
    assert evaluate_arithmetic("-5 + 2") == Decimal("-3")
    assert evaluate_arithmetic("- -5") == Decimal("5")
    assert evaluate_arithmetic("+3.5") == Decimal("3.5")
    assert evaluate_arithmetic("4 * -2.5") == Decimal("-10")


def test_floor_division_truncates_toward_zero_the_way_decimal_does():
    """Decimal's `//` is not Python's on a negative operand, and a body reading the answer back
    should find that written down rather than discover it on a live call."""
    assert evaluate_arithmetic("-7 // 2") == Decimal("-3")
    assert -7 // 2 == -4


def test_the_answer_does_not_depend_on_the_calling_process_decimal_context():
    """A fixed context of its own, so two Runs of the same body agree digit for digit."""
    import decimal

    with decimal.localcontext() as ctx:
        ctx.prec = 3
        assert evaluate_arithmetic("1 / 3") == Decimal("0.3333333333333333333333333333333333")


# --- what is not arithmetic is refused by name, before anything runs ---

@pytest.mark.parametrize("expression, words", [
    ("total", "names"),
    ("abs(-1)", "calls"),
    ("(2).numerator", "attributes"),
    ("prices[0]", "subscripts"),
    ("1 < 2", "comparisons"),
    ("1 and 2", "boolean operators"),
    ("[1, 2]", "lists"),
    ("1 if 2 else 3", "conditional expressions"),
    ("f'{1}'", "formatted strings"),
    ("6 & 3", "bitwise and"),
    ("1 << 4", "a left shift"),
    ("~1", "bitwise inversion"),
    ("'12'", "a text value"),
    ("True + 1", "a true or false value"),
    ("None", "a missing value"),
    ("2 +", "does not parse"),
    ("", "does not parse"),
    ("total = 2 + 2", "does not parse"),
    (None, "given as text"),
    (12, "given as text"),
    ("1e400", "too large to write as a decimal"),
])
def test_anything_outside_the_arithmetic_grammar_is_refused_with_its_kind_named(expression, words):
    """Text that does not parse is refused rather than raising a syntax error, a value that is not
    text never reaches the parser, and a float literal Python reads as infinity is refused rather
    than answered as a Decimal that is not a number at all."""
    with pytest.raises(ValueError, match=words):
        evaluate_arithmetic(expression)


# --- the caps, which are what stands in for a timeout ---

def test_an_overlong_expression_is_refused_before_it_is_parsed():
    with pytest.raises(ValueError, match="at most 2000 are allowed"):
        evaluate_arithmetic("1 + " * (MAX_EXPRESSION_CHARS // 2))
    assert evaluate_arithmetic("1" + " + 1" * ((MAX_EXPRESSION_CHARS - 1) // 4)) > 0


def test_a_power_too_large_to_compute_is_refused_before_computing():
    """9 ** 9 ** 9 has 370 million digits. The inner power is small and computes; the outer one is
    refused on an exponent of 387420489, so nothing large is ever asked for. The exponent cap alone
    is not enough: a hundred-digit base to the sixtieth is still enormous, so width is capped too."""
    with pytest.raises(ValueError, match="refused before it is computed"):
        evaluate_arithmetic("9 ** 9 ** 9")
    with pytest.raises(ValueError, match=f"at most {MAX_EXPONENT} away from zero"):
        evaluate_arithmetic(f"2 ** {MAX_EXPONENT + 1}")
    assert evaluate_arithmetic(f"2 ** {MAX_EXPONENT}") == Decimal(2) ** MAX_EXPONENT
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic("9" * 100 + " ** 60")
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic("1" * (MAX_RESULT_DIGITS + 2))
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic(f"(10 ** {MAX_EXPONENT}) * " * 9 + "10")


def test_dividing_taking_a_remainder_or_floor_dividing_by_zero_all_read_the_same_way():
    for expression in ("1 / 0", "1 // 0", "1 % 0", "5 / (3 - 3)"):
        with pytest.raises(ValueError, match="division by zero is not allowed"):
            evaluate_arithmetic(expression)
    with pytest.raises(ValueError, match="does not have"):
        evaluate_arithmetic("(0 - 2) ** 0.5")

