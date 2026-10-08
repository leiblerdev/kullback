from kullback.runner.records import Verifier
from kullback.spec.canfail import can_fail
from kullback.spec.compile import compile_spec
from tests.spec.fixtures import WRITE_TOOLS, fn, reference_run, spec


def _write_verifier() -> Verifier:
    verifier = compile_spec(spec(), WRITE_TOOLS, fn).verifier
    at = next(e.idx for e in reference_run().events if e.type == "tool_call" and e.payload["name"] == "update_item")
    atoms = [a.model_copy(update={"target": dict(a.target, at=at)}) for a in verifier.atoms]
    return verifier.model_copy(update={"atoms": atoms})


def test_a_verifier_with_one_required_write_fails_the_empty_and_the_swapped_value_run():
    result = can_fail(_write_verifier(), reference_run(), canon=fn, write_tools=WRITE_TOOLS)
    rows = {row["stage"]: row for row in result.rows}
    assert result.passed, result.rows
    assert rows["verifier_empty_run"]["passed"]
    assert rows["verifier_wrong_run"]["passed"] and rows["verifier_wrong_run"]["swaps"] > 0


def test_an_empty_verifier_cannot_fail():
    result = can_fail(Verifier(task_id="t1"), reference_run(), canon=fn, write_tools=WRITE_TOOLS)
    assert not result.passed
    assert not {row["stage"]: row for row in result.rows}["verifier_empty_run"]["passed"]
