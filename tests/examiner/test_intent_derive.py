"""Tests for examiner/intent_derive.py: the Verifier writer, must side.

The writer sees the request, the policy, the tool schemas and reads of the
Starting state. Every check carries a because the grounding code accepts, and
the scorer rules on the atoms unchanged.
"""

from __future__ import annotations

import json

from kullback.ai.provider import TestModel
from kullback.examiner import intent_derive as W
from kullback.runner.canon import CanonRules

CANON = CanonRules()


def _write(task_id: str, intent: object, policy: object, tools: object,
           world: object, model: object, stats: object = None, attempts: int = 3) -> object:
    return W.write_verifier(task_id, intent, policy, tools, world, model,
                            canon=CANON, stats=stats, max_attempts=attempts)  # type: ignore[arg-type]


def _checks(*checks: dict) -> str:
    return json.dumps({"checks": list(checks)})


def _write_check(**fields: object) -> dict:
    check = {"kind": "required", "tool": "fetch", "entity": "record",
             "id_field": "code", "field": "state", "value": "ready"}
    check.update(fields)
    return check


def _world() -> dict:
    return {"record": {"K7": {"code": "K7", "state": "ready"}}}


def _tools(**bodies: object) -> list[dict]:
    tools = [{"name": "fetch", "kind": "read", "description": "read a record",
              "args_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
              "body": bodies.get("fetch", lambda code: {"code": code, "state": "ready"})},
             {"name": "amend", "kind": "write", "description": "change a record",
              "args_schema": {"type": "object", "properties": {"code": {"type": "string"}}}}]
    return tools


def test_a_check_quoting_the_request_is_kept_with_its_because():
    """The writer's grounded check lands in the Verifier carrying its because."""
    model = TestModel([_checks(_write_check(because_quote="mark the record ready"))])
    stats: dict = {}
    verifier = _write("t1", "please mark the record ready", None,
                                _tools(), _world(), model, stats=stats)
    assert [atom.kind for atom in verifier.atoms] == ["required", "required"]
    assert verifier.atoms[0].target["kind"] == "write"
    assert verifier.atoms[1].target["value"] is not None
    assert "mark the record ready" in verifier.atoms[1].description
    assert stats["kept_checks"] == 1 and stats["refused"] == 0


def test_a_quote_outside_the_request_is_refused_then_rewritten():
    """A refused check goes back to the session; the best attempt wins."""
    bad = _write_check(because_quote="paint it blue")
    good = _write_check(because_quote="mark the record ready")
    fixed = _write_check(because_quote="mark the record ready", value="ready")
    model = TestModel([_checks(bad, good), _checks(fixed, good)])
    stats: dict = {}
    verifier = _write("t1", "please mark the record ready", None,
                                _tools(), _world(), model, stats=stats)
    assert stats["attempts"] == 2 and stats["refused"] == 0
    assert stats["kept_checks"] == 2
    assert len(verifier.atoms) == 4


def test_a_value_no_read_returned_is_refused():
    """A value neither read nor said has no grounding and is refused."""
    model = TestModel([_checks(_write_check(because_quote="mark the record ready",
                                            value="purple"))])
    stats: dict = {}
    verifier = _write("t1", "please mark the record ready", None,
                                _tools(), _world(), model, stats=stats, attempts=1)
    assert verifier.atoms == [] and stats["kept_checks"] == 0
    assert stats["refused"] == 1


def test_a_value_only_a_read_holds_is_listed_allowed_not_required():
    """A required check may not pick a row the request never names; allowed it is kept."""
    demanded = _write_check(because_quote="mark the record", value="ready")
    listed = _write_check(kind="allowed", because_quote="mark the record", value="ready")
    model = TestModel([
        {"content": "", "tool_calls": [{"id": "c1", "name": "fetch", "arguments": {"code": "K7"}}]},
        _checks(demanded),
        _checks(listed),
    ])
    stats: dict = {}
    verifier = _write("t1", "please mark the record", None,
                                _tools(), _world(), model, stats=stats)
    assert stats["reads"] == 1 and stats["attempts"] == 2 and stats["kept_checks"] == 1
    assert verifier.atoms[1].kind == "allowed" and verifier.atoms[1].target["raw"] == "ready"
    assert verifier.atoms[1].provenance == "system_derived"


def test_a_required_row_the_request_never_names_is_refused():
    """The demand scope covers the row as well as the value."""
    check = _write_check(entity="ledger", because_quote="mark the record ready", value="ready")
    model = TestModel([_checks(check)])
    stats: dict = {}
    verifier = _write("t1", "please mark the record ready", None,
                                _tools(), _world(), model, stats=stats, attempts=1)
    assert verifier.atoms == [] and stats["refused"] == 1


def test_a_value_the_request_holds_is_user_stated():
    """A value from the user's own words needs no read."""
    model = TestModel([_checks(_write_check(because_quote="mark the record ready",
                                            value="ready"))])
    verifier = _write("t1", "please mark the record ready", None,
                                _tools(), _world(), model)
    assert verifier.atoms[1].provenance == "user_stated"


def test_an_unknown_policy_rule_is_refused():
    """A because naming no policy rule is refused, even with a good value."""
    policy = {"ask-first": "ask before changing anything"}
    model = TestModel([
        _checks(_write_check(because_rule="no-such-rule")),
        _checks(_write_check(because_rule="ask-first")),
    ])
    stats: dict = {}
    verifier = _write("t1", "please mark the record ready", policy,
                                _tools(), _world(), model, stats=stats)
    assert stats["attempts"] == 2 and stats["kept_checks"] == 1
    assert "ask-first" in verifier.atoms[0].description


def test_alternative_end_states_become_allowed_checks():
    """Several acceptable values are allowed checks, not one required literal."""
    model = TestModel([_checks(_write_check(because_quote="mark the record ready",
                                            value="ready", alternatives=["steady"]))])
    world = {"record": {"K7": {"code": "K7", "state": "ready", "alt": "steady"}}}
    tools = _tools(fetch=lambda code: {"code": code, "state": "ready", "alt": "steady"})
    verifier = _write("t1", "mark the record ready, steady is fine", None,
                                tools, world, model)
    kinds = [(atom.id, atom.kind) for atom in verifier.atoms]
    assert ("w0", "required") in kinds
    assert ("w0.state", "required") in kinds
    assert ("w0.state.alt1", "allowed") in kinds


def test_a_question_names_what_the_policy_asks_for():
    """A question check carries the policy rule that asks for it."""
    policy = {"ask-first": "ask before changing anything"}
    check = {"kind": "question", "tool": "fetch", "field": "code",
             "because_rule": "ask-first"}
    model = TestModel([_checks(check)])
    verifier = _write("t1", "change the record", policy,
                                _tools(), _world(), model)
    assert verifier.atoms[0].target == {"kind": "question", "key": "field:code",
                                        "tool": "fetch", "field": "code"}


def test_a_communicate_check_splits_into_stated_tokens():
    """One message fact becomes one atom per token the scorer reads."""
    check = {"kind": "communicate", "text": "record K7",
             "because_quote": "record K7 ready"}
    model = TestModel([_checks(check)])
    verifier = _write("t1", "is record K7 ready?", None,
                                _tools(), _world(), model)
    assert [atom.target["text"] for atom in verifier.atoms] == ["K7"]
    assert all(atom.kind == "communicate" for atom in verifier.atoms)


def test_a_fact_without_a_user_question_is_refused():
    """No question in the request means no message fact is demanded."""
    check = {"kind": "communicate", "text": "record K7",
             "because_quote": "record K7 ready"}
    model = TestModel([_checks(check)])
    stats: dict = {}
    verifier = _write("t1", "mark record K7 ready", None,
                                _tools(), _world(), model, stats=stats, attempts=1)
    assert verifier.atoms == [] and stats["refused"] == 1


def test_a_failing_body_leaves_the_starting_state_whole():
    """A body that raises reads as a failure and changes nothing."""
    world = _world()
    calls = []

    def fetch(state: object, code: str) -> dict:
        calls.append(code)
        if len(calls) == 1:
            state.put("record", "K7", {"state": "changed"})  # type: ignore[union-attr]
            raise ValueError("gone")
        return dict(state.row("record", code))  # type: ignore[union-attr]

    model = TestModel([
        {"content": "", "tool_calls": [{"id": "c1", "name": "fetch", "arguments": {"code": "K7"}},
                                       {"id": "c2", "name": "fetch", "arguments": {"code": "K7"}}]},
        _checks(_write_check(because_quote="mark the record ready")),
    ])
    _write("t1", "please mark the record ready", None,
                     _tools(fetch=fetch), world, model)
    first, second = (message["content"] for message in model.calls[1]["messages"][3:5])
    assert "ValueError" in first
    assert json.loads(second)["state"] == "ready"


def test_the_session_only_calls_reads_and_names_no_other_episode():
    """Write tools are described so checks can name them, never called."""
    seen: dict = {}

    def fetch(code: str) -> dict:
        seen["called"] = True
        return {"code": code}

    tools = _tools(fetch=fetch)
    model = TestModel([_checks(_write_check(because_quote="mark the record ready"))])
    _write("t1", "please mark the record ready", None, tools, _world(), model)
    call = model.calls[0]
    tool_names = [tool["name"] for tool in call["tools"]]
    assert tool_names == ["fetch"]
    assert "called" not in seen
    prompt = call["messages"][0]["content"] + call["messages"][1]["content"]
    for banned in ("reference", "reroll", "re-run", "re-roll"):
        assert banned not in prompt.lower()
    assert "amend" in call["messages"][1]["content"]

def test_a_state_first_body_reads_the_copied_world():
    """Bodies taking the state read the copy the way the Runner passes it."""

    def fetch(state: object, code: str) -> dict:
        row = state.row("record", code)  # type: ignore[union-attr]
        return {"code": code, "state": row["state"]}

    model = TestModel([
        {"content": "", "tool_calls": [{"id": "c1", "name": "fetch", "arguments": {"code": "K7"}}]},
        _checks(_write_check(kind="allowed", because_quote="mark the record")),
    ])
    stats: dict = {}
    _write("t1", "please mark the record", None,
                     _tools(fetch=fetch), _world(), model, stats=stats)
    assert stats["reads"] == 1 and stats["kept_checks"] == 1


def test_tiers_come_from_the_kind_only():
    """vf-add1 pin: the tier mapping every atom kind lands on."""
    assert W.tier_of({"kind": "required"}) == "critical"
    assert W.tier_of({"kind": "forbidden"}) == "sanity"
    assert W.tier_of({"kind": "hard"}) == "sanity"
    assert W.tier_of({"kind": "question"}) == "important"
    assert W.tier_of({"kind": "communicate"}) == "important"
    assert W.tier_of({"kind": "allowed"}) == "important"


def test_the_tokeniser_matches_the_intent_tokeniser():
    """The vendored copy reads words the way builder/intent.py does."""
    from kullback.builder.intent import _tokens as intent_tokens

    samples = ["Cancel order #W1 please", "$17.99 price difference", "credit_card_1234",
               "Change wireless earbuds", "e-mail gift-card", "PayPal/Venmo refund",
               "Ship 1,000 units by 12:30", "The user's 2nd request! Really?"]
    for sample in samples:
        assert W._intent_tokens(sample) == intent_tokens(sample), sample


def test_mined_schemas_reach_the_model_in_provider_shape():
    """A miner-spelled type rides to the model as draft 2020-12."""
    entry = W._Entry("fetch", args_schema={"type": "object", "properties": {
        "expression": {"type": ["str"]}}, "required": ["expression"]})
    assert W._input_schema(entry) == {"type": "object", "properties": {
        "expression": {"type": "string"}}, "required": ["expression"]}
    bare = W._Entry("fetch", args_schema=None)
    assert W._input_schema(bare) == {"type": "object", "properties": {}}


def test_the_schema_shape_matches_the_toolkit_loader():
    """The vendored copy normalises the way runner/world/loading.py does."""
    from kullback.runner.world.loading import _json_schema as loader_schema

    samples = [{"properties": {"expression": {"type": ["str"]}}, "required": ["expression"],
                "type": "object"},
               {"type": ["int", "NoneType"]}, {"type": "dict", "properties": {}},
               [{"type": "list"}, "plain", 3, None]]
    for sample in samples:
        assert W._json_schema(sample) == loader_schema(sample), sample


def test_a_fenced_answer_still_parses():
    """Fences around the JSON are presentation, not a refusal."""
    fenced = "```json\n" + _checks(_write_check(because_quote="mark the record ready")) + "\n```"
    model = TestModel([fenced])
    stats: dict = {}
    _write("t1", "please mark the record ready", None,
           _tools(), _world(), model, stats=stats, attempts=1)
    assert stats["kept_checks"] == 1 and stats["refused"] == 0
