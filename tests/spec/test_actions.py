"""A tool the world records as an action is no write: demanded as a call that happened, never counted as a write.

The invented world of tests/spec/fixtures.py gains one action tool, `close_case`, whose calls the schema
records as rows of an actions table. Verdicts go through the real check_run.
"""

from __future__ import annotations

from kullback.runner.records import Event, Run
from kullback.runner.target import check_run
from kullback.spec.actions import action_tools_of, workdir_action_tools
from kullback.spec.compile import compile_spec
from tests.spec.fixtures import WRITE_TOOLS, check, fn, reference_run, spec

ACTION = "close_case"
ACTIONS = {ACTION}
ALL_WRITES = set(WRITE_TOOLS) | ACTIONS
SCHEMA = {"tables": ["actions", "items"], "columns": [
    {"table": "actions", "name": "tool", "evidence": {"actions_of": [ACTION]}},
    {"table": "items", "name": "slot", "evidence": {}}]}
NO_WRITE = {"demand": "no_write"}
ON_ACTION = {"demand": "write", "tool": ACTION}


def _with_calls(*names: str) -> Run:
    """A Run that reads, then makes each named call once, each answered without an error."""
    events = [Event(idx=0, type="user_turn", payload={"content": "Please move item A1 to slot seven."}),
              Event(idx=1, type="tool_call", payload={"id": "r", "name": "list_items", "args": {}}),
              Event(idx=2, type="tool_result", payload={"id": "r", "result": []})]
    for n, name in enumerate(names):
        events += [Event(idx=3 + 2 * n, type="tool_call", payload={"id": f"c{n}", "name": name, "args": {}}),
                   Event(idx=4 + 2 * n, type="tool_result", payload={"id": f"c{n}", "result": "done"})]
    return Run(run_id="r", task_id="t1", termination_reason="success", events=events)


def _passes(checks, run, actions=ACTIONS) -> bool:
    verifier = compile_spec(spec(checks), ALL_WRITES, fn, action_tools=actions).verifier
    return check_run(verifier, run, canon=fn, write_tools=ALL_WRITES - set(actions))[0]


def test_the_action_tools_are_the_ones_the_schema_records_as_rows_of_actions(tmp_path):
    assert action_tools_of(SCHEMA) == ACTIONS
    assert workdir_action_tools(tmp_path) == set()


def test_a_run_whose_only_call_beyond_reads_is_an_allowed_action_passes_a_no_write_spec():
    checks = [check("c0", "required", NO_WRITE), check("c1", "allowed", ON_ACTION)]
    assert _passes(checks, _with_calls(ACTION))


def test_a_required_action_fails_the_run_that_never_calls_it_and_passes_the_one_that_does():
    checks = [check("c0", "required", ON_ACTION)]
    assert not _passes(checks, _with_calls())
    assert _passes(checks, _with_calls(ACTION))


def test_a_forbidden_action_fails_a_run_that_calls_it():
    for demand, kind in ((ON_ACTION, "forbidden"), ({"demand": "no_write", "tool": ACTION}, "required")):
        checks = [check("c0", kind, demand)]
        assert not _passes(checks, _with_calls(ACTION))
        assert _passes(checks, _with_calls())


def test_a_real_write_still_fails_the_no_write_atom_beside_an_action():
    checks = [check("c0", "required", NO_WRITE)]
    assert not _passes(checks, reference_run())


def test_a_required_action_is_no_extra_write_to_a_scorer_that_still_counts_it_as_a_write():
    verifier = compile_spec(spec([check("c0", "required", ON_ACTION)]), ALL_WRITES, fn,
                            action_tools=ACTIONS).verifier
    assert check_run(verifier, _with_calls(ACTION), canon=fn, write_tools=ALL_WRITES)[0]
