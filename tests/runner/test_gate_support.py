"""Tests for runner/gate_support.py: the field readers every gate is built from."""

from __future__ import annotations

# --- the field readers every gate is built from ------------------------------
# gate_support._passed and gate_support._get decide what a gate sees. Both accept a record and the
# plain dict that record becomes in JSON, and mutation testing found only one of the three spellings
# under test: a mutant that dropped the `passed` fallback and one that flipped the missing-field
# default both survived the whole suite.


def test_passed_reads_all_three_spellings_of_the_same_field():
    """`Verdict.passed` carries the alias `pass`, so a record, a dump by alias and a dump by field
    name are three shapes of one Verdict and a gate must read the same answer off each."""
    from kullback.runner.gate_support import _passed
    from kullback.runner.records import Verdict

    verdict = Verdict(run_id="r1", **{"pass": True, "class": "pass"})
    assert _passed(verdict) is True
    assert _passed(verdict.model_dump(by_alias=True)) is True
    assert _passed(verdict.model_dump()) is True

    failed = Verdict(run_id="r2", **{"pass": False, "class": "fail"})
    assert _passed(failed) is False
    assert _passed(failed.model_dump(by_alias=True)) is False
    assert _passed(failed.model_dump()) is False


def test_passed_is_false_when_nothing_says_it_passed():
    """A value that carries no verdict at all has not passed. The default must not be True: a gate
    reading an object it does not understand would then count it as a pass."""
    from kullback.runner.gate_support import _passed

    class Nothing:
        pass

    assert _passed(Nothing()) is False
    assert _passed({}) is False
    assert _passed({"other": True}) is False


def test_get_reads_a_trailing_underscore_field_off_its_json_name():
    """`class_` is written `class` in JSON, which is the only reason `_get` strips the underscore."""
    from kullback.runner.gate_support import _get
    from kullback.runner.records import Verdict

    verdict = Verdict(run_id="r1", **{"pass": False, "class": "fail"})
    assert _get(verdict, "class_") == "fail"
    assert _get(verdict.model_dump(by_alias=True), "class_") == "fail"
    assert _get({"class_": "fail"}, "class_") == "fail"
    assert _get({}, "class_", "missing") == "missing"


def test_share_of_an_empty_sample_is_not_a_hundred_percent():
    """A rate over nothing has no value, and reporting 1.0 there would read as a perfect score."""
    from kullback.runner.gate_support import _share

    assert _share(0, 0) is None
    assert _share(0, 4) == 0.0
    assert _share(4, 4) == 1.0

