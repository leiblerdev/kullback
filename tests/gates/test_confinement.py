"""Tests for kullback.gates.confinement: a predicate and a tool body are refused before they run anywhere."""

from __future__ import annotations

from kullback.gates.artifacts import policy_gate
from kullback.gates.confinement import (
    MAX_NAMED_FAILURES,
    gate_confined,
    predicate_confinement,
    predicate_confinement_gate,
    source_confinement,
    unbound_names,
)
from kullback.runner.records import Constraint, ConstraintTests


def a_constraint(**kw) -> Constraint:
    base = dict(
        id="c1",
        text="never cancel a delivered order",
        compiled=True,
        predicate_src="def check(pre_state, write_call, transcript):\n    return pre_state['status'] != 'delivered'\n",
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
    assert out.metrics == {"chars": len(source), "unbound": 0}


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


# --- a name the body loads that nothing binds ---

def test_a_body_that_names_a_module_it_never_imported_is_refused_before_a_call_reaches_the_line():
    """Build 8: `calculate` raised NameError: decimal on 49 live results and 33 replayed reads, past a
    compile gate whose recorded calls never reached the line."""
    source = _module("        if order_id == 'x':\n            return decimal.Decimal(1)\n        return {'id': order_id}\n")
    assert unbound_names(source) == [
        "get_order names decimal, which nothing binds; put `import decimal` at the top of the body"]
    out = gate_confined(source)
    assert out.stage == "confined" and out.passed is False
    assert out.metrics == {"chars": len(source), "unbound": 1}
    assert out.failures == unbound_names(source)


def test_a_name_nothing_could_bind_is_refused_without_an_import_hint():
    source = _module("        return helper(order_id)\n")
    assert unbound_names(source) == ["get_order names helper, which nothing binds"]


def test_names_the_body_the_module_or_python_bind_are_not_unbound():
    body = ("        rows = [r for r in self.db.orders if r]\n"
            "        total = sum(len(json.dumps(r)) for r in rows)\n"
            "        for i, row in enumerate(rows):\n"
            "            pass\n"
            "        try:\n"
            "            import math\n"
            "        except KeyError as exc:\n"
            "            return str(exc)\n"
            "        with self.db.session() as handle:\n"
            "            pass\n"
            "        return {'n': total, 'i': i, 'row': row, 'pi': math.pi, 'h': handle, 'db': DomainDB}\n")
    assert unbound_names(_module(body)) == []
    assert gate_confined(_module(body)).passed is True


def test_only_the_tool_methods_are_checked_for_unbound_names():
    source = "class Other:\n    def f(self):\n        return nothing\n"
    assert unbound_names(source) == []
    assert unbound_names(source, class_name="Other") == ["f names nothing, which nothing binds"]


def test_a_module_that_does_not_parse_has_no_unbound_names_because_the_parses_gate_rules_on_it():
    assert unbound_names("def broken(:\n") == []


def test_a_name_bound_only_inside_a_nested_scope_is_still_unbound_around_it():
    """Greptile on PR 4: reading a child scope's bindings as the parent's would pass the very shape
    the check exists to catch."""
    nested_function = _module("        def helper():\n"
                              "            total = 1\n"
                              "            return total\n"
                              "        return helper() + total\n")
    assert unbound_names(nested_function) == ["get_order names total, which nothing binds"]
    comprehension = _module("        rows = [item for item in self.db.orders]\n"
                            "        return {'rows': rows, 'last': item}\n")
    assert unbound_names(comprehension) == ["get_order names item, which nothing binds"]
    lambda_arg = _module("        pick = lambda row: row\n        return pick(order_id) or row\n")
    assert unbound_names(lambda_arg) == ["get_order names row, which nothing binds"]


def test_a_nested_scope_sees_what_the_method_around_it_bound():
    source = _module("        prefix = str(order_id)\n"
                     "        return [prefix + str(r) for r in self.db.orders]\n")
    assert unbound_names(source) == []


def test_an_unbound_name_inside_a_nested_scope_is_reported_against_the_method():
    source = _module("        return [decimal.Decimal(r) for r in self.db.orders]\n")
    assert unbound_names(source) == [
        "get_order names decimal, which nothing binds; put `import decimal` at the top of the body"]


def test_a_global_declaration_without_an_assignment_does_not_bind_the_name():
    """`global counter` says where an assignment would land, not that anything bound it."""
    source = _module("        global counter\n        return counter + 1\n")
    assert unbound_names(source) == ["get_order names counter, which nothing binds"]


def test_a_body_that_declares_a_name_global_or_nonlocal_is_refused_by_name():
    """Greptile on PR 4: a nested `global` resolves at module scope, not against the method's
    locals. A tool body has no business keeping state that outlives its call, so the declaration
    itself is the failure and the resolution question never arises."""
    module_state = _module("        global counter\n        counter = 1\n        return counter\n")
    assert source_confinement(module_state) == ["get_order declares global counter"]
    assert gate_confined(module_state).passed is False
    nested = _module("        total = 0\n"
                     "        def helper():\n"
                     "            global total\n"
                     "            return total\n"
                     "        return helper()\n")
    assert source_confinement(nested) == ["get_order declares global total"]


def test_a_method_of_a_nested_class_does_not_see_what_the_class_body_bound():
    """Greptile on PR 4: Python resolves a class body's names in the class body alone, so a method
    that loads one unqualified raises NameError however plainly it reads."""
    source = _module("        class Row:\n"
                     "            kind = 'order'\n"
                     "            def label(self):\n"
                     "                return kind\n"
                     "        return Row().label()\n")
    assert unbound_names(source) == ["get_order names kind, which nothing binds"]
    qualified = _module("        class Row:\n"
                        "            kind = 'order'\n"
                        "            def label(self):\n"
                        "                return Row.kind\n"
                        "        return Row().label()\n")
    assert unbound_names(qualified) == []

