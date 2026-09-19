import pytest

from kullback.runner import target
from kullback.runner.records import Atom, Run, Verifier
from kullback.runner.verdict import verdict
from tests.runner.test_verdict import CANCEL, oracle_lines


@pytest.mark.parametrize("matches", [True, False])
def test_structured_forbidden_write_agrees_between_gate_and_verdict(matches):
    run = Run(run_id="forbidden-proof", task_id="t1", events=oracle_lines()[1:])
    fn = target.canon_fn(None)
    effects = target.write_effects(run, {CANCEL}, fn)
    effect = next(iter(effects.values()))
    atom = Atom(id="ban", kind="forbidden", target={
        "kind": "write", "tool": effect["tool"],
        "entity": effect["entity"] if matches else "unobserved_entity",
    })
    verifier = Verifier(task_id="t1", atoms=[
        Atom(id="allow", kind="allowed", target={"kind": "write", "tool": CANCEL}),
        atom,
    ])
    assert target.atom_holds(atom, run, fn, {CANCEL}) is matches
    assert target.check_run(verifier, run, write_tools={CANCEL}) == (not matches, "ban" if matches else None)
    result = verdict(run, verifier, write_tools={CANCEL})
    assert result.passed is not matches
    assert result.class_ == ("fail" if matches else "pass")
    assert result.failing_atom == ("ban" if matches else None)
