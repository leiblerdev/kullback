from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kullback import sampling
from kullback.runner.records import DisclosureRule, RawPtr, Task, ToolCall, ToolCallError, Trace, UserFact, UserRules
from kullback.user.population import Population, RecordingStanding, build_population, pick_user

WRITE_TOOL = "set_address"
WRITE_TOOLS = [WRITE_TOOL]
SALT = "test-salt-1"
REF_ARGS = {"city": "Springfield", "primary": True}


def _ptr():
    return RawPtr(file_hash="aa")


def _ok_call(name=WRITE_TOOL, args=None, result="done"):
    base = {"city": "Springfield", "primary": True}
    if args is not None:
        base = dict(args)
    return ToolCall(
        name=name,
        args=base,
        result=result,
        has_result=True,
        resolved=True,
        truncated=False,
        raw_ptr=_ptr(),
    )


def _failed_call(name=WRITE_TOOL, args=None, error_class="business_error"):
    base = {"city": "Springfield", "primary": True}
    if args is not None:
        base = dict(args)
    return ToolCall(
        name=name,
        args=base,
        result=None,
        error=ToolCallError(class_=error_class, payload="refused", classified_by="code"),
        has_result=True,
        resolved=True,
        truncated=False,
        raw_ptr=_ptr(),
    )


def _trace(tid, calls):
    return Trace(
        trace_id=tid,
        raw_hash="h",
        ingest_version="v1",
        source="s",
        turns=[],
        tool_calls=list(calls),
        raw_ptr=_ptr(),
    )


def _task(run_ids, tid="t1"):
    return Task(id=tid, run_ids=list(run_ids))


def _st(e=True, c=True, r=True):
    return RecordingStanding(task_eligible=e, complete=c, replay_confirmed=r)


def _rules():
    return UserRules()


def _withheld_map(pop):
    return {w.recording_id: tuple(w.reasons) for w in pop.withheld}


def _build(calls_by_id, *, run_ids=None, standings=None, rules=None, held=(), tools=WRITE_TOOLS):
    traces = {rid: _trace(rid, calls) for rid, calls in calls_by_id.items()}
    return build_population(
        _task(run_ids if run_ids is not None else list(calls_by_id)),
        traces,
        reference_id="ref",
        standings=standings if standings is not None else {k: _st() for k in traces},
        rules=rules if rules is not None else {k: _rules() for k in traces},
        held_out_ids=set(held),
        write_tools=list(tools),
    )


def test_matching_recordings_join_the_population_and_others_carry_reasons():
    calls = {
        "ref": [_ok_call(args=REF_ARGS)],
        "ok": [_ok_call(args=dict(REF_ARGS))],
        "mismatch": [_ok_call(args={"city": "Shelbyville", "primary": True})],
        "held": [_ok_call(args=dict(REF_ARGS))],
        "unconf": [_ok_call(args=dict(REF_ARGS))],
        "nullok": [
            ToolCall(
                name=WRITE_TOOL,
                args=dict(REF_ARGS),
                result=None,
                has_result=True,
                resolved=True,
                truncated=False,
                raw_ptr=_ptr(),
            )
        ],
        "retry": [_failed_call(args=dict(REF_ARGS)), _ok_call(args=dict(REF_ARGS))],
        "reader": [
            _ok_call(args=dict(REF_ARGS)),
            _failed_call(name="get_address", args={"city": "Springfield"}),
        ],
        "slowreader": [
            _ok_call(args=dict(REF_ARGS)),
            _failed_call(name="get_address", args={"city": "Springfield"}, error_class="transient"),
        ],
        "extra": [
            _ok_call(args=dict(REF_ARGS)),
            _failed_call(name="set_email", args={"email": "a@b.c"}, error_class="invalid_arguments"),
        ],
    }
    standings = {k: _st() for k in calls}
    standings["unconf"] = _st(e=True, c=True, r=False)
    pop = _build(calls, standings=standings, held={"held"}, tools=[WRITE_TOOL, "set_email"])
    assert list(pop.eligible_ids) == ["extra", "nullok", "ok", "reader", "ref", "retry", "slowreader"]
    assert pop.support_count == 7
    reasons = _withheld_map(pop)
    assert "write_mismatch" in reasons["mismatch"]
    assert "held_out" in reasons["held"]
    assert "unconfirmed" in reasons["unconf"]
    assert "retry" not in reasons
    assert set(pop.eligible_ids) | set(reasons) == set(calls)


def _cut_call():
    return ToolCall(
        name=WRITE_TOOL,
        args=dict(REF_ARGS),
        result="done",
        has_result=True,
        resolved=True,
        truncated=True,
        raw_ptr=_ptr(),
    )


def _silent_call():
    return ToolCall(
        name=WRITE_TOOL,
        args=dict(REF_ARGS),
        result=None,
        has_result=None,
        resolved=False,
        truncated=False,
        raw_ptr=_ptr(),
    )


def _broken_call():
    return ToolCall.model_construct(
        name=WRITE_TOOL,
        args=dict(REF_ARGS),
        result=None,
        error="boom",
        has_result=True,
        resolved=True,
        truncated=False,
        raw_ptr=_ptr(),
    )


def _withheld_row(
    case_id, cand_calls, want, *, exact=False, ref_calls=None, standing=None, rules=True, held=(), tools=WRITE_TOOLS
):
    standing = {} if standing is None else standing
    return pytest.param(case_id, cand_calls, want, exact, ref_calls, standing, rules, held, tools, id=case_id)


@pytest.mark.parametrize(
    ("cand_id", "cand_calls", "want", "exact", "ref_calls", "standing_kw", "with_rules", "held", "tools"),
    [
        _withheld_row("ghost", [_ok_call(args=REF_ARGS)], ("unknown_standing",), exact=True, standing=False),
        _withheld_row("held", [_ok_call(args=REF_ARGS)], ("held_out",), held=("held",)),
        _withheld_row("norules", [_ok_call(args=REF_ARGS)], ("missing_rules",), rules=False),
        _withheld_row("cut", [_cut_call()], ("truncated",)),
        _withheld_row(
            "unconfirmed", [_ok_call(args=REF_ARGS)], ("unconfirmed",), exact=True, standing={"r": False}
        ),
        _withheld_row(
            "incomplete", [_ok_call(args=REF_ARGS)], ("incomplete",), exact=True, standing={"c": False}
        ),
        _withheld_row("silent", [_silent_call()], ("unanswered",)),
        _withheld_row(
            "bool_int",
            [_ok_call(args={"flag": 1})],
            ("write_mismatch",),
            ref_calls=[_ok_call(args={"flag": True})],
        ),
        _withheld_row(
            "reordered",
            [_ok_call(name="write_b", args={"n": 2}), _ok_call(name="write_a", args={"n": 1})],
            ("write_mismatch",),
            ref_calls=[_ok_call(name="write_a", args={"n": 1}), _ok_call(name="write_b", args={"n": 2})],
            tools=["write_a", "write_b"],
        ),
        _withheld_row("failed", [_failed_call(args=REF_ARGS)], ("write_mismatch", f"write_failed:{WRITE_TOOL}")),
        _withheld_row("broken", [_broken_call()], (f"malformed_error:{WRITE_TOOL}",)),
        _withheld_row(
            "extra",
            [
                _ok_call(args=REF_ARGS),
                _failed_call(name="set_email", args={"email": "a@b.c"}, error_class="transient"),
            ],
            ("write_indeterminate:set_email",),
            tools=[WRITE_TOOL, "set_email"],
        ),
        _withheld_row(
            "retry",
            [_failed_call(args=REF_ARGS, error_class="transient"), _ok_call(args=REF_ARGS)],
            (f"write_indeterminate:{WRITE_TOOL}",),
        ),
    ],
)
def test_a_candidate_is_withheld_for_exactly_one_named_cause(
    cand_id, cand_calls, want, exact, ref_calls, standing_kw, with_rules, held, tools
):
    calls = {"ref": ref_calls or [_ok_call(args=REF_ARGS)], cand_id: cand_calls}
    rules = {"ref": _rules()}
    if with_rules:
        rules[cand_id] = _rules()
    standings = {"ref": _st()}
    if standing_kw is not False:
        standings[cand_id] = _st(**standing_kw)
    pop = _build(calls, standings=standings, rules=rules, held=held, tools=tools)
    assert list(pop.eligible_ids) == ["ref"]
    reasons = _withheld_map(pop)[cand_id]
    if exact:
        assert reasons == want
    else:
        for reason in want:
            assert reason in reasons


def _reference_inputs(case):
    traces = {"ref": _trace("ref", [_ok_call()]), "ok": _trace("ok", [_ok_call()])}
    standings = {"ref": _st(), "ok": _st()}
    rules = {k: _rules() for k in traces}
    task = _task(["ref", "ok"])
    ref_id = "ref"
    held = set()
    if case == "outside_task":
        task = _task(["a", "b"])
        traces = {"a": _trace("a", [_ok_call()]), "b": _trace("b", [_ok_call()])}
        standings = {"a": _st(), "b": _st()}
        rules = {k: _rules() for k in traces}
        ref_id = "zzz"
    elif case == "held_out":
        held = {"ref"}
    elif case == "bad_standing":
        standings = {"ref": _st(e=False), "ok": _st()}
    elif case == "missing_rules":
        rules = {"ok": _rules()}
    elif case == "missing_trace":
        traces = {"ok": _trace("ok", [_ok_call()])}
    elif case == "indeterminate_write":
        traces["ref"] = _trace(
            "ref",
            [
                _ok_call(args=REF_ARGS),
                _failed_call(args={"city": "Shelbyville", "primary": True}, error_class="unknown"),
            ],
        )
    return task, traces, standings, rules, held, ref_id


@pytest.mark.parametrize(
    ("case", "match"),
    [
        pytest.param("outside_task", None, id="outside_task"),
        pytest.param("held_out", None, id="held_out"),
        pytest.param("bad_standing", None, id="bad_standing"),
        pytest.param("missing_rules", None, id="missing_rules"),
        pytest.param("missing_trace", None, id="missing_trace"),
        pytest.param("indeterminate_write", "write_indeterminate", id="indeterminate_write"),
    ],
)
def test_a_broken_reference_raises(case, match):
    task, traces, standings, rules, held, ref_id = _reference_inputs(case)
    with pytest.raises(ValueError, match=match):
        build_population(
            task,
            traces,
            reference_id=ref_id,
            standings=standings,
            rules=rules,
            held_out_ids=held,
            write_tools=list(WRITE_TOOLS),
        )


def _mini_population(run_order, trace_order):
    task = _task(list(run_order))
    all_traces = {
        "ref": _trace("ref", [_ok_call(args=REF_ARGS)]),
        "b": _trace("b", [_ok_call(args=dict(REF_ARGS))]),
        "c": _trace("c", [_ok_call(args={"city": "Shelbyville", "primary": True})]),
    }
    all_standings = {"ref": _st(), "b": _st(), "c": _st()}
    all_rules = {k: _rules() for k in all_traces}
    traces = {k: all_traces[k] for k in trace_order}
    standings = {k: all_standings[k] for k in trace_order}
    rules = {k: all_rules[k] for k in trace_order}
    return build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )


def _pop_for_rules(rules_map):
    calls = {"ref": [_ok_call(args=REF_ARGS)], "ok": [_ok_call(args=dict(REF_ARGS))]}
    return _build(calls, rules=dict(rules_map))


def test_the_pick_is_repeatable_whatever_the_input_order():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    pop_b = _mini_population(["c", "b", "ref"], ["c", "b", "ref"])
    assert pop_a.fingerprint == pop_b.fingerprint
    assert list(pop_a.eligible_ids) == list(pop_b.eligible_ids)
    sel_a = pick_user(pop_a, seed=7, salt=SALT)
    sel_b = pick_user(pop_b, seed=7, salt=SALT)
    assert sel_a.chosen_id == sel_b.chosen_id
    assert sel_a.population_fingerprint == pop_a.fingerprint
    rules_a = {
        "ref": UserRules(
            facts=[UserFact(field="zip", value="19122", context="stated")],
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["email"],
            walk_away=["closing"],
        ),
        "ok": UserRules(
            facts=[UserFact(field="zip", value="19122", context="stated")],
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["email"],
            walk_away=["closing"],
        ),
    }
    reordered = {
        "walk_away": ["closing"],
        "refusals": ["email"],
        "disclosure": [DisclosureRule(**{"on_request": True, "field": "zip"})],
        "facts": [UserFact(**{"context": "stated", "value": "19122", "field": "zip"})],
    }
    rules_b = {"ok": UserRules(**reordered), "ref": UserRules(**reordered)}
    assert _pop_for_rules(rules_a).fingerprint == _pop_for_rules(rules_b).fingerprint


def _ident(pop, seed):
    return json.dumps(
        [pop.task_id, seed, pop.fingerprint],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _pool_changed():
    pop_b = _build(
        {
            "ref": [_ok_call(args=REF_ARGS)],
            "b": [_ok_call(args={"city": "Ogdenville", "primary": True})],
            "c": [_ok_call(args={"city": "Shelbyville", "primary": True})],
        }
    )
    return _mini_population(["ref", "b", "c"], ["ref", "b", "c"]).fingerprint, pop_b.fingerprint


def _unrelated_record_added():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    pop_b = _build(
        {
            "ref": [_ok_call(args=REF_ARGS)],
            "b": [_ok_call(args=dict(REF_ARGS))],
            "c": [_ok_call(args={"city": "Shelbyville", "primary": True})],
            "ghost": [_ok_call(args=dict(REF_ARGS))],
        },
        run_ids=["ref", "b", "c"],
    )
    assert pick_user(pop_a, seed=3, salt=SALT).chosen_id == pick_user(pop_b, seed=3, salt=SALT).chosen_id
    return pop_a.fingerprint, pop_b.fingerprint


def _inputs_mutated_after_build():
    run_ids = ["ref", "ok"]
    task = _task(run_ids)
    traces = {
        "ref": _trace("ref", [_ok_call(args=REF_ARGS)]),
        "ok": _trace("ok", [_ok_call(args=dict(REF_ARGS))]),
    }
    standings = {"ref": _st(), "ok": _st()}
    rules = {k: _rules() for k in traces}
    tools = list(WRITE_TOOLS)
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=tools,
    )
    before_eligible = list(pop.eligible_ids)
    before_fp = pop.fingerprint
    run_ids.append("ghost")
    task.run_ids.append("ghost")
    traces["ghost"] = _trace("ghost", [_ok_call(args=dict(REF_ARGS))])
    standings["ghost"] = _st()
    rules["ghost"] = _rules()
    tools.append("extra_tool")
    assert list(pop.eligible_ids) == before_eligible
    assert isinstance(pop.eligible_ids, tuple)
    with pytest.raises(ValidationError):
        pop.support_count = 999
    return before_fp, pop.fingerprint


def _rule_fact_value_changed():
    base = {
        "ref": UserRules(facts=[UserFact(field="city", value="Springfield")]),
        "ok": UserRules(facts=[UserFact(field="city", value="Springfield")]),
    }
    mutated = {
        "ref": UserRules(facts=[UserFact(field="city", value="Springfield")]),
        "ok": UserRules(facts=[UserFact(field="city", value="Shelbyville")]),
    }
    pop_a = _pop_for_rules(base)
    pop_b = _pop_for_rules(mutated)
    assert list(pop_a.eligible_ids) == list(pop_b.eligible_ids) == ["ok", "ref"]
    assert _ident(pop_b, 7) != _ident(pop_a, 7)
    key_a = sampling.sample_key("user_population", _ident(pop_a, 7), SALT)
    key_b = sampling.sample_key("user_population", _ident(pop_b, 7), SALT)
    assert key_b != key_a
    return pop_a.fingerprint, pop_b.fingerprint


def _rules_with(on_request=True, refusal="email", walk_away="closing"):
    return UserRules(
        disclosure=[DisclosureRule(field="zip", on_request=on_request)],
        refusals=[refusal],
        walk_away=[walk_away],
    )


def _rule_behaviour_changed(change):
    def build():
        pop_a = _pop_for_rules({"ref": _rules_with(), "ok": _rules_with()})
        pop_b = _pop_for_rules({"ref": _rules_with(), "ok": _rules_with(**change)})
        assert list(pop_b.eligible_ids) == ["ok", "ref"]
        return pop_a.fingerprint, pop_b.fingerprint

    return build


def _rules_mutated_after_build():
    ok_rules = UserRules(facts=[UserFact(field="city", value="Springfield")])
    rules_map = {"ref": UserRules(facts=[UserFact(field="city", value="Springfield")]), "ok": ok_rules}
    pop = _pop_for_rules(rules_map)
    before = pop.fingerprint
    ok_rules.facts.append(UserFact(field="city", value="Shelbyville"))
    assert pop.fingerprint == before
    assert _pop_for_rules(rules_map).fingerprint != before
    return before, pop.fingerprint


def _membership_changed():
    pop_ok = _build({"ref": [_ok_call(args=REF_ARGS)], "cand": [_ok_call(args=dict(REF_ARGS))]})
    pop_bad = _build({"ref": [_ok_call(args=REF_ARGS)], "cand": [_failed_call(args=dict(REF_ARGS))]})
    assert list(pop_ok.eligible_ids) == ["cand", "ref"]
    assert list(pop_bad.eligible_ids) == ["ref"]
    return pop_ok.fingerprint, pop_bad.fingerprint


@pytest.mark.parametrize(
    ("fingerprints", "moves"),
    [
        pytest.param(_pool_changed, True, id="pool_content"),
        pytest.param(_unrelated_record_added, False, id="unrelated_record"),
        pytest.param(_inputs_mutated_after_build, False, id="inputs_mutated_after_build"),
        pytest.param(_rule_fact_value_changed, True, id="rule_fact_value"),
        pytest.param(_rule_behaviour_changed({"on_request": False}), True, id="rule_disclosure"),
        pytest.param(_rule_behaviour_changed({"refusal": "phone"}), True, id="rule_refusal"),
        pytest.param(_rule_behaviour_changed({"walk_away": "other"}), True, id="rule_walk_away"),
        pytest.param(_rules_mutated_after_build, False, id="rules_mutated_after_build"),
        pytest.param(_membership_changed, True, id="membership"),
    ],
)
def test_the_fingerprint_moves_only_with_what_the_pool_holds(fingerprints, moves):
    before, after = fingerprints()
    assert (before != after) is moves


def test_empty_population_pick_none_without_draw():
    pop = Population(
        task_id="t1",
        reference_id="ref",
        eligible_ids=(),
        withheld=(),
        support_count=0,
        fingerprint="abc",
    )
    before = sampling.draws_by_kind()
    sel = pick_user(pop, seed=0, salt=SALT)
    assert sel.chosen_id is None
    assert sel.task_id == "t1"
    assert sel.seed == 0
    assert sel.population_fingerprint == "abc"
    assert sampling.draws_since(before) == {}


def test_seed_validation():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    with pytest.raises(TypeError):
        pick_user(pop_a, seed=True, salt=SALT)
    with pytest.raises(ValueError):
        pick_user(pop_a, seed=-1, salt=SALT)
    with pytest.raises(TypeError):
        pick_user(pop_a, seed="3", salt=SALT)
    sel = pick_user(pop_a, seed=0, salt=SALT)
    assert sel.chosen_id in set(pop_a.eligible_ids)
    assert sel.seed == 0


def test_draw_matches_keyed_primitive():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    for seed, salt in [(11, "pin-salt"), (0, "sält-Ω-日本語")]:
        key = sampling.sample_key("user_population", _ident(pop_a, seed), salt)
        want = sorted(pop_a.eligible_ids)[key % len(pop_a.eligible_ids)]
        assert pick_user(pop_a, seed=seed, salt=salt).chosen_id == want


def test_each_pick_counts_one_keyed_draw_and_repeats_exactly():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    before = sampling.draws_by_kind()
    first = pick_user(pop_a, seed=9, salt=SALT)
    assert first.chosen_id is not None
    assert sampling.draws_since(before) == {"user_population": 1}
    second = pick_user(pop_a, seed=9, salt=SALT)
    assert first == second
    assert sampling.draws_since(before) == {"user_population": 2}
