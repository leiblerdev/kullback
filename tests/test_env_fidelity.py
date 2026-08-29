"""`scripts/env_fidelity.py`: the verdict per call and the cause behind a miss. No model, no tau2."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import env_fidelity as F  # noqa: E402


def test_verdict_keeps_error_and_result_apart():
    assert F.verdict({"error": "ValueError: x"}, {"result": 1}, None) == "only_ours_errored"
    assert F.verdict({"result": 1}, {"error": "ValueError: x"}, None) == "only_theirs_errored"
    assert F.verdict({"result": 1, "changed": True}, {"result": 1, "changed": False}, None) == "effect_differs"
    assert F.verdict({"result": 1, "changed": True}, {"result": 1, "changed": True}, None) == "same"


def test_both_errors_compare_the_message_not_the_prefix():
    assert F.verdict({"error": "ValueError: User not found"},
                     {"error": "ValueError: User not found"}, None) == "both_error"
    assert F.verdict({"error": "ValueError: Error: User not found"},
                     {"error": "ValueError: User not found"}, None) == "both_error_other_message"


@pytest.mark.parametrize("word, mine, real, expected", [
    ("not_confined", {}, {}, "confinement"),
    ("same", {}, {}, "none"),
    ("both_error", {}, {}, "none"),
    ("both_error_other_message", {"error": "ValueError: Error: User not found"},
     {"error": "ValueError: User not found"}, "error_prefix"),
    ("both_error_other_message", {"error": "ValueError: No such user"},
     {"error": "ValueError: User not found"}, "error_message"),
    ("only_ours_errored", {"error": "NameError: name 're' is not defined"}, {"result": "ok"},
     "missing_import"),
    ("only_ours_errored", {"error": "AttributeError: 'Order' object has no attribute 'get'"},
     {"result": {}}, "row_access"),
    ("only_ours_errored", {"error": "ValueError: Item not found"}, {"result": {"item_id": "1"}},
     "schema_shape"),
    ("only_ours_errored", {"error": "ZeroDivisionError: division by zero"}, {"result": 1},
     "body_error"),
    ("only_theirs_errored", {"result": 1}, {"error": "ValueError: x"}, "real_errored"),
    ("result_differs", {"result": {"value": -121.2}}, {"result": -121.2}, "result_shape"),
    ("result_differs", {"result": {"value": -121.2}}, {"result": "-121.2"}, "result_shape"),
    ("result_differs", {"result": 24.24}, {"result": "24.24"}, "result_shape"),
    ("result_differs", {"result": {"value": 1}}, {"result": {"value": 2}}, "value"),
    ("result_differs", {"result": {"total": 3}}, {"result": {"sum": 3}}, "result_shape"),
    ("result_differs", {"result": {"total": 3}}, {"result": {"total": 4}}, "value"),
    ("effect_differs", {}, {}, "effect"),
])
def test_cause_names_the_owner(word, mine, real, expected):
    assert F.cause(word, mine, real) == expected
    assert expected == "none" or expected in F.CAUSE_OWNER


def test_report_splits_by_cause():
    result = {"domain": "retail", "tools": 2, "assisted": [], "calls_scored": 3, "calls_recorded": 5,
              "agreement": 0.3333, "agreement_all": 0.2,
              "totals": {"same": 1, "result_differs": 2, "not_confined": 2},
              "by_cause": {"confinement": {"calls": 2, "tools": {"b": 2}},
                           "result_shape": {"calls": 2, "tools": {"a": 2}}},
              "per_tool": {"a": {"calls": 3, "assisted": False, "same": 1, "result_differs": 2},
                           "b": {"calls": 2, "assisted": False, "not_confined": 2, "why": "b uses getattr"}},
              "examples": [{"tool": "a", "args": {}, "verdict": "result_differs",
                            "cause": "result_shape", "ours": "{'value': 1}", "real": "1"}]}
    text = F.report(result)
    assert "Where the misses come from" in text
    assert "| confinement | 2 | 40.0% | `b` 2 |" in text
    assert "| result_shape | 2 | 40.0% | `a` 2 |" in text
    assert "refused tools counted as misses: **20.0%**" in text
    assert "(result_differs, result_shape)" in text
