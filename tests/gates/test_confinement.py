"""Tests for kullback.gates.confinement: a predicate and a tool body are refused before they run anywhere."""

from __future__ import annotations

from kullback.gates.artifacts import policy_gate
from kullback.gates.confinement import (
    MAX_NAMED_FAILURES,
    gate_confined,
    predicate_confinement,
    predicate_confinement_gate,
    source_confinement,
)
from kullback.runner.records import Constraint, ConstraintTests


def a_constraint(**kw) -> Constraint:
    base = dict(
        id="c1",
        text="never cancel a delivered order",
        compiled=True,
        predicate_src="def check(case):\n    return case['status'] != 'delivered'\n",
        tests=ConstraintTests(pos=[{"status": "pending"}], neg=[{"status": "delivered"}]),
    )
    base.update(kw)
    return Constraint(**base)


# --- the Runner runs no model-written predicate it has not certified (D89, design section 7) ---

def test_a_predicate_that_walks_out_of_its_case_is_refused_before_it_runs():
    """A restricted __builtins__ alone is not confinement: subclasses() reaches every loaded class."""
    escape = ("def check(case):\n"
              "    return [c for c in ().__class__.__base__.__subclasses__()\n"
              "            if c.__name__ == 'catch_warnings'] != []\n")
    refused = predicate_confinement(escape)
    assert refused, "the escape was certified"
    assert any("touches __" in line for line in refused)
    out = policy_gate([a_constraint(predicate_src=escape,
                                    tests=ConstraintTests(pos=[{"status": "pending"}]))])
    assert out.passed is False
    assert any("not confined" in f for f in out.failures)


def test_the_predicate_gate_is_the_same_ruling_as_a_record():
    importing = "import os\n\n\ndef check(case):\n    return True\n"
    assert predicate_confinement(importing) == ["imports a module"]
    assert predicate_confinement(a_constraint().predicate_src) == []
    out = predicate_confinement_gate(importing, constraint_id="c9")
    assert out.stage == "compile_policy.confined"
    assert out.passed is False
    assert out.metrics == {"chars": len(importing)}
    assert out.failures == ["c9: imports a module"]
    assert predicate_confinement_gate(a_constraint().predicate_src).passed is True


# --- a generated tool body, before it runs anywhere (compile_tools gate 0) ---

def _module(body: str) -> str:
    return ("import json\n\n\nclass DomainDB:\n    pass\n\n\nclass DomainTools:\n"
            "    def __init__(self, db):\n        self.db = db\n\n"
            "    def get_order(self, order_id):\n" + body)


def test_a_clean_tool_body_is_confined():
    source = _module("        return self.db.orders[order_id].__doc__ or {'id': order_id}\n")
    assert source_confinement(source) == []
    out = gate_confined(source)
    assert out.stage == "confined" and out.passed is True
    assert out.metrics == {"chars": len(source)}


def test_a_tool_body_that_reaches_outside_the_world_is_refused_by_name():
    source = _module("        import os\n        return getattr(os, 'system')(order_id)\n")
    out = gate_confined(source)
    assert out.passed is False
    assert out.failures == ["get_order imports os", "get_order uses getattr"]


def test_only_the_tool_methods_are_checked():
    """The skeleton around the methods is code-owned, so an import there is not the model's doing."""
    source = "import os\n\n\nclass DomainTools:\n    def __init__(self, db):\n        self.db = os\n"
    assert gate_confined(source).passed is True
    assert gate_confined(source, class_name="Other").passed is True


def test_a_body_that_does_not_parse_fails_the_gate_rather_than_raising():
    out = gate_confined("class DomainTools:\n    def get_order(self, x)\n        return x\n")
    assert out.passed is False
    assert any("does not parse" in f for f in out.failures)


def test_the_named_failures_are_capped_and_the_ruling_is_not():
    methods = "".join(f"    def t{i}(self):\n        import os\n        return os\n\n" for i in range(8))
    source = "class DomainTools:\n" + methods
    out = gate_confined(source)
    assert out.passed is False
    assert len(source_confinement(source)) == 8
    assert len(out.failures) == MAX_NAMED_FAILURES
