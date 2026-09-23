"""Tests for kullback.gates.confinement: a predicate and a tool body are refused before they run anywhere."""

from __future__ import annotations

from kullback.gates.artifacts import policy_gate
from kullback.gates.confinement import (
    PROVIDED_HELPERS,
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


def test_an_importing_predicate_fails_the_confined_gate_by_id():
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

# Two names only the code-owned skeleton may import. The first is built from parts so the lines
# carry no corpus name (the branch check refuses the literal); the gate sees the full value.
_SKELETON_ONLY_PACKAGE = "tau" + "2"
_SKELETON_ONLY_MODULE = "data_model"


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
    for statement, failure in (("import os", "get_order imports os"),
                               ("import " + _SKELETON_ONLY_PACKAGE, "get_order imports " + _SKELETON_ONLY_PACKAGE),
                               ("from " + _SKELETON_ONLY_MODULE + " import Widget",
                                "get_order imports " + _SKELETON_ONLY_MODULE)):
        refused = _module("        " + statement + "\n        return {'id': order_id}\n")
        assert source_confinement(refused) == [failure]
        assert gate_confined(refused).passed is False


def test_only_the_tool_methods_are_checked():
    """The skeleton around the methods is code-owned, so an import there is not the model's doing."""
    source = "import os\n\n\nclass DomainTools:\n    def __init__(self, db):\n        self.db = os\n"
    assert gate_confined(source).passed is True
    assert gate_confined(source, class_name="Other").passed is True
    other = "class Other:\n    def f(self):\n        return nothing\n"
    assert unbound_names(other) == []
    assert unbound_names(other, class_name="Other") == ["f names nothing, which nothing binds"]


def test_module_level_skeleton_imports_with_clean_bodies_still_pass():
    source = (
        "import " + _SKELETON_ONLY_PACKAGE + "\nfrom " + _SKELETON_ONLY_MODULE + " import Widget\n\n\n"
        "class DomainTools:\n"
        "    def __init__(self, db):\n        self.db = db\n\n"
        "    def fetch_widget(self, widget_id):\n"
        "        return {'id': widget_id}\n"
    )
    assert source_confinement(source) == []
    assert gate_confined(source).passed is True


def test_a_body_that_does_not_parse_fails_the_gate_rather_than_raising():
    out = gate_confined("class DomainTools:\n    def get_order(self, x)\n        return x\n")
    assert out.passed is False
    assert any("does not parse" in f for f in out.failures)


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
    unknown = _module("        return helper(order_id)\n")
    assert unbound_names(unknown) == ["get_order names helper, which nothing binds"]


def test_a_provided_helper_counts_as_bound_and_a_name_that_only_sounds_like_one_does_not():
    """Both loaders put `evaluate_arithmetic` in the module's namespace, so a body that calls it
    without importing anything is not a NameError waiting for the first call. The list is what is
    bound, not a way of waving names through: a helper that only sounds like one is refused."""
    source = _module("        return float(evaluate_arithmetic('(3.5 + 1.25) * 2'))\n")
    assert PROVIDED_HELPERS == {"evaluate_arithmetic"}
    assert unbound_names(source) == []
    assert source_confinement(source) == []
    assert gate_confined(source).passed is True
    lookalike = _module("        return evaluate_arithmetically('1 + 1')\n")
    assert unbound_names(lookalike) == ["get_order names evaluate_arithmetically, which nothing binds"]


def test_the_names_the_helper_replaces_are_refused_as_they_were_before():
    """Providing an evaluator loosens nothing: eval, exec, compile and ast stay out."""
    for body, failure in (("        return eval('1 + 1')\n", "get_order uses eval"),
                          ("        return exec('x = 1')\n", "get_order uses exec"),
                          ("        return compile('1', '<x>', 'eval')\n", "get_order uses compile"),
                          ("        import ast\n        return ast.parse('1 + 1')\n", "get_order imports ast")):
        assert failure in source_confinement(_module(body))
        assert gate_confined(_module(body)).passed is False


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


def test_a_name_bound_only_inside_a_nested_scope_is_unbound_around_it_and_reported_against_the_method():
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
    inside = _module("        return [decimal.Decimal(r) for r in self.db.orders]\n")
    assert unbound_names(inside) == [
        "get_order names decimal, which nothing binds; put `import decimal` at the top of the body"]


def test_a_nested_scope_sees_what_the_method_around_it_bound():
    source = _module("        prefix = str(order_id)\n"
                     "        return [prefix + str(r) for r in self.db.orders]\n")
    assert unbound_names(source) == []


def test_a_body_that_declares_a_name_global_or_nonlocal_is_refused_by_name_and_the_declaration_binds_nothing():
    """`global counter` says where an assignment would land, not that anything bound it.

    Greptile on PR 4: a nested `global` resolves at module scope, not against the method's
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
    declared_only = _module("        global counter\n        return counter + 1\n")
    assert unbound_names(declared_only) == ["get_order names counter, which nothing binds"]


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


def test_a_nested_class_method_sees_every_enclosing_scope_and_still_flags_what_nothing_binds():
    """The reported false reject does not happen: a method of a class defined inside the tool
    method closes over the method's locals like any nested scope, so valid generated code is
    accepted and never forced into repair or assisted fallback for this. Threading those bindings
    through the class must not blind the gate to a name nothing binds anywhere, and one level deeper
    (the class inside a nested function) the method reads a local from each enclosing scope."""
    source = _module("        factor = order_id * 2\n"
                     "        class Helper:\n"
                     "            def run(self):\n"
                     "                return factor + 1\n"
                     "        return Helper().run()\n")
    assert unbound_names(source) == []
    missing = _module("        factor = order_id * 2\n"
                      "        class Helper:\n"
                      "            def run(self):\n"
                      "                return factor + missing\n"
                      "        return Helper().run()\n")
    assert unbound_names(missing) == ["get_order names missing, which nothing binds"]
    deeper = _module("        scale = 2\n"
                     "        def make():\n"
                     "            class Helper:\n"
                     "                def run(self):\n"
                     "                    return scale + order_id\n"
                     "            return Helper()\n"
                     "        return make().run()\n")
    assert unbound_names(deeper) == []
