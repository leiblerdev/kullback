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


def test_a_money_expression_is_exact_to_the_cent_where_floats_would_drift():
    """The shape build 12 could not compute: a difference of two sums, read off a recording.

    Every literal enters as Decimal(str(value)), so the answer is the one a person adding the
    column would write, not the float nearest to it.
    """
    assert evaluate_arithmetic("(2459.74 * 2) - (2291.87 + 2520.52)") == Decimal("107.09")
    assert evaluate_arithmetic("0.1 + 0.2") == Decimal("0.3")
    assert float(evaluate_arithmetic("0.1 + 0.2")) != 0.1 + 0.2


def test_a_leading_minus_or_plus_is_read_as_a_sign():
    assert evaluate_arithmetic("-5 + 2") == Decimal("-3")
    assert evaluate_arithmetic("- -5") == Decimal("5")
    assert evaluate_arithmetic("+3.5") == Decimal("3.5")
    assert evaluate_arithmetic("4 * -2.5") == Decimal("-10")


def test_every_operator_the_grammar_names_computes():
    assert evaluate_arithmetic("7 / 2") == Decimal("3.5")
    assert evaluate_arithmetic("7 // 2") == Decimal("3")
    assert evaluate_arithmetic("7 % 2") == Decimal("1")
    assert evaluate_arithmetic("2 ** 10") == Decimal("1024")


def test_floor_division_truncates_toward_zero_the_way_decimal_does():
    """Decimal's `//` is not Python's on a negative operand, and a body reading the answer back
    should find that written down rather than discover it on a live call."""
    assert evaluate_arithmetic("-7 // 2") == Decimal("-3")
    assert -7 // 2 == -4


def test_whitespace_and_deep_parentheses_around_one_number_are_read():
    assert evaluate_arithmetic("  \n 42 \t ") == Decimal("42")
    assert evaluate_arithmetic("(" * 40 + "1 + 1" + ")" * 40) == Decimal("2")


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
])
def test_an_expression_that_is_not_arithmetic_is_refused_and_the_kind_is_named(expression, words):
    with pytest.raises(ValueError, match=words):
        evaluate_arithmetic(expression)


def test_an_operator_outside_the_grammar_is_refused_and_named():
    with pytest.raises(ValueError, match="bitwise and"):
        evaluate_arithmetic("6 & 3")
    with pytest.raises(ValueError, match="a left shift"):
        evaluate_arithmetic("1 << 4")
    with pytest.raises(ValueError, match="bitwise inversion"):
        evaluate_arithmetic("~1")


def test_a_literal_that_is_not_a_number_is_refused_and_named():
    with pytest.raises(ValueError, match="a text value"):
        evaluate_arithmetic("'12'")
    with pytest.raises(ValueError, match="a true or false value"):
        evaluate_arithmetic("True + 1")
    with pytest.raises(ValueError, match="a missing value"):
        evaluate_arithmetic("None")


def test_text_that_is_not_an_expression_at_all_is_refused_rather_than_raising_a_syntax_error():
    with pytest.raises(ValueError, match="does not parse"):
        evaluate_arithmetic("2 +")
    with pytest.raises(ValueError, match="does not parse"):
        evaluate_arithmetic("")
    with pytest.raises(ValueError, match="does not parse"):
        evaluate_arithmetic("total = 2 + 2")


def test_something_that_is_not_text_is_refused_rather_than_reaching_the_parser():
    with pytest.raises(ValueError, match="given as text"):
        evaluate_arithmetic(None)
    with pytest.raises(ValueError, match="given as text"):
        evaluate_arithmetic(12)


# --- the caps, which are what stands in for a timeout ---

def test_an_overlong_expression_is_refused_before_it_is_parsed():
    with pytest.raises(ValueError, match="at most 2000 are allowed"):
        evaluate_arithmetic("1 + " * (MAX_EXPRESSION_CHARS // 2))
    assert evaluate_arithmetic("1" + " + 1" * ((MAX_EXPRESSION_CHARS - 1) // 4)) > 0


def test_a_tower_of_powers_is_refused_on_its_exponent_and_never_computed():
    """9 ** 9 ** 9 has 370 million digits. The inner power is small and computes; the outer one is
    refused on an exponent of 387420489, so nothing large is ever asked for."""
    with pytest.raises(ValueError, match="refused before it is computed"):
        evaluate_arithmetic("9 ** 9 ** 9")
    with pytest.raises(ValueError, match=f"at most {MAX_EXPONENT} away from zero"):
        evaluate_arithmetic(f"2 ** {MAX_EXPONENT + 1}")
    assert evaluate_arithmetic(f"2 ** {MAX_EXPONENT}") == Decimal(2) ** MAX_EXPONENT


def test_a_power_of_a_wide_base_is_refused_on_the_width_even_where_the_exponent_is_small():
    """The exponent cap alone is not enough: a hundred-digit base to the sixtieth is still enormous."""
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic("9" * 100 + " ** 60")


def test_a_value_that_grows_past_the_digit_cap_is_refused():
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic("1" * (MAX_RESULT_DIGITS + 2))
    with pytest.raises(ValueError, match=f"more than {MAX_RESULT_DIGITS} digits"):
        evaluate_arithmetic(f"(10 ** {MAX_EXPONENT}) * " * 9 + "10")


def test_a_float_literal_python_reads_as_infinity_is_refused_rather_than_answered():
    """Without this, `1e400 - 1e400` comes back as a Decimal that is not a number at all."""
    with pytest.raises(ValueError, match="too large to write as a decimal"):
        evaluate_arithmetic("1e400")


def test_dividing_taking_a_remainder_or_floor_dividing_by_zero_all_read_the_same_way():
    for expression in ("1 / 0", "1 // 0", "1 % 0", "5 / (3 - 3)"):
        with pytest.raises(ValueError, match="division by zero is not allowed"):
            evaluate_arithmetic(expression)


def test_a_power_arithmetic_has_no_answer_for_is_a_refusal_not_a_crash():
    with pytest.raises(ValueError, match="does not have"):
        evaluate_arithmetic("(0 - 2) ** 0.5")
