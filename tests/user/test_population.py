from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kullback import sampling
from kullback.runner.records import DisclosureRule, RawPtr, Task, ToolCall, ToolCallError, Trace, UserFact, UserRules
from kullback.user.population import Population, RecordingStanding, build_population, pick_user

WRITE_TOOL = "set_address"
WRITE_TOOLS = [WRITE_TOOL]
SALT = "test-salt-1"


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


def test_eligible_support_and_reasons():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "ok", "mismatch", "held", "unconf"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "ok": _trace("ok", [_ok_call(args=dict(ref_args))]),
        "mismatch": _trace("mismatch", [_ok_call(args={"city": "Shelbyville", "primary": True})]),
        "held": _trace("held", [_ok_call(args=dict(ref_args))]),
        "unconf": _trace("unconf", [_ok_call(args=dict(ref_args))]),
    }
    standings = {
        "ref": _st(),
        "ok": _st(),
        "mismatch": _st(),
        "held": _st(),
        "unconf": _st(e=True, c=True, r=False),
    }
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids={"held"},
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ok", "ref"]
    assert pop.support_count == 2
    reasons = _withheld_map(pop)
    assert "write_mismatch" in reasons["mismatch"]
    assert "held_out" in reasons["held"]
    assert "unconfirmed" in reasons["unconf"]
    assert set(pop.eligible_ids) | set(reasons) == set(task.run_ids)


def test_missing_standing_withheld():
    task = _task(["ref", "ghost"])
    traces = {
        "ref": _trace("ref", [_ok_call()]),
        "ghost": _trace("ghost", [_ok_call()]),
    }
    standings = {"ref": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert _withheld_map(pop)["ghost"] == ("unknown_standing",)


def test_held_out_exclusion():
    task = _task(["ref", "held"])
    traces = {"ref": _trace("ref", [_ok_call()]), "held": _trace("held", [_ok_call()])}
    standings = {"ref": _st(), "held": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=["held"],
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "held_out" in _withheld_map(pop)["held"]


def test_reference_not_in_task_raises():
    task = _task(["a", "b"])
    traces = {"a": _trace("a", [_ok_call()]), "b": _trace("b", [_ok_call()])}
    standings = {"a": _st(), "b": _st()}
    rules = {k: _rules() for k in traces}
    with pytest.raises(ValueError):
        build_population(
            task,
            traces,
            reference_id="zzz",
            standings=standings,
            rules=rules,
            held_out_ids=set(),
            write_tools=list(WRITE_TOOLS),
        )


def test_reference_held_out_raises():
    task = _task(["ref", "ok"])
    traces = {"ref": _trace("ref", [_ok_call()]), "ok": _trace("ok", [_ok_call()])}
    standings = {"ref": _st(), "ok": _st()}
    rules = {k: _rules() for k in traces}
    with pytest.raises(ValueError):
        build_population(
            task,
            traces,
            reference_id="ref",
            standings=standings,
            rules=rules,
            held_out_ids={"ref"},
            write_tools=list(WRITE_TOOLS),
        )


def test_reference_standing_raises():
    task = _task(["ref", "ok"])
    traces = {"ref": _trace("ref", [_ok_call()]), "ok": _trace("ok", [_ok_call()])}
    standings = {"ref": _st(e=False), "ok": _st()}
    rules = {k: _rules() for k in traces}
    with pytest.raises(ValueError):
        build_population(
            task,
            traces,
            reference_id="ref",
            standings=standings,
            rules=rules,
            held_out_ids=set(),
            write_tools=list(WRITE_TOOLS),
        )


def test_reference_missing_trace_or_rules_raises():
    task = _task(["ref", "ok"])
    traces = {"ref": _trace("ref", [_ok_call()]), "ok": _trace("ok", [_ok_call()])}
    standings = {"ref": _st(), "ok": _st()}
    rules = {"ok": _rules()}
    with pytest.raises(ValueError):
        build_population(
            task,
            traces,
            reference_id="ref",
            standings=standings,
            rules=rules,
            held_out_ids=set(),
            write_tools=list(WRITE_TOOLS),
        )
    traces_only_ok = {"ok": _trace("ok", [_ok_call()])}
    rules_full = {"ref": _rules(), "ok": _rules()}
    with pytest.raises(ValueError):
        build_population(
            task,
            traces_only_ok,
            reference_id="ref",
            standings=standings,
            rules=rules_full,
            held_out_ids=set(),
            write_tools=list(WRITE_TOOLS),
        )


def test_complete_vs_confirmed():
    task = _task(["ref", "unconfirmed", "incomplete"])
    ref_args = {"city": "Springfield", "primary": True}
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "unconfirmed": _trace("unconfirmed", [_ok_call(args=dict(ref_args))]),
        "incomplete": _trace("incomplete", [_ok_call(args=dict(ref_args))]),
    }
    standings = {
        "ref": _st(),
        "unconfirmed": _st(e=True, c=True, r=False),
        "incomplete": _st(e=True, c=False, r=True),
    }
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    reasons = _withheld_map(pop)
    assert reasons["unconfirmed"] == ("unconfirmed",)
    assert reasons["incomplete"] == ("incomplete",)


def test_truncated_withheld():
    task = _task(["ref", "cut"])
    good = _ok_call()
    cut = ToolCall(
        name=WRITE_TOOL,
        args={"city": "Springfield", "primary": True},
        result="done",
        has_result=True,
        resolved=True,
        truncated=True,
        raw_ptr=_ptr(),
    )
    traces = {"ref": _trace("ref", [good]), "cut": _trace("cut", [cut])}
    standings = {"ref": _st(), "cut": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "truncated" in _withheld_map(pop)["cut"]


def test_unanswered_vs_null_success():
    task = _task(["ref", "silent", "nullok"])
    ref_args = {"city": "Springfield", "primary": True}
    silent = ToolCall(
        name=WRITE_TOOL,
        args=dict(ref_args),
        result=None,
        has_result=None,
        resolved=False,
        truncated=False,
        raw_ptr=_ptr(),
    )
    nullok = ToolCall(
        name=WRITE_TOOL,
        args=dict(ref_args),
        result=None,
        has_result=True,
        resolved=True,
        truncated=False,
        raw_ptr=_ptr(),
    )
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "silent": _trace("silent", [silent]),
        "nullok": _trace("nullok", [nullok]),
    }
    standings = {"ref": _st(), "silent": _st(), "nullok": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["nullok", "ref"]
    assert "unanswered" in _withheld_map(pop)["silent"]


def test_bool_int_mismatch():
    task = _task(["ref", "candidate"])
    traces = {
        "ref": _trace("ref", [_ok_call(args={"flag": True})]),
        "candidate": _trace("candidate", [_ok_call(args={"flag": 1})]),
    }
    standings = {"ref": _st(), "candidate": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "write_mismatch" in _withheld_map(pop)["candidate"]


def test_reordered_writes_mismatch():
    tools = ["write_a", "write_b"]
    task = _task(["ref", "candidate"])
    traces = {
        "ref": _trace(
            "ref",
            [
                _ok_call(name="write_a", args={"n": 1}),
                _ok_call(name="write_b", args={"n": 2}),
            ],
        ),
        "candidate": _trace(
            "candidate",
            [
                _ok_call(name="write_b", args={"n": 2}),
                _ok_call(name="write_a", args={"n": 1}),
            ],
        ),
    }
    standings = {"ref": _st(), "candidate": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(tools),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "write_mismatch" in _withheld_map(pop)["candidate"]


def test_missing_rules_withheld():
    task = _task(["ref", "norules"])
    traces = {"ref": _trace("ref", [_ok_call()]), "norules": _trace("norules", [_ok_call()])}
    standings = {"ref": _st(), "norules": _st()}
    rules = {"ref": _rules()}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "missing_rules" in _withheld_map(pop)["norules"]


def test_nonbool_standing_rejected():
    with pytest.raises(ValidationError):
        RecordingStanding(task_eligible=1, complete=True, replay_confirmed=True)
    with pytest.raises(ValidationError):
        RecordingStanding(task_eligible=True, complete="yes", replay_confirmed=True)
    with pytest.raises(ValidationError):
        RecordingStanding(task_eligible=True, complete=True, replay_confirmed=0)


def _mini_population(run_order, trace_order):
    task = _task(list(run_order))
    ref_args = {"city": "Springfield", "primary": True}
    all_traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "b": _trace("b", [_ok_call(args=dict(ref_args))]),
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


def test_repeatability_after_permutation():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    pop_b = _mini_population(["c", "b", "ref"], ["c", "b", "ref"])
    assert pop_a.fingerprint == pop_b.fingerprint
    assert list(pop_a.eligible_ids) == list(pop_b.eligible_ids)
    sel_a = pick_user(pop_a, seed=7, salt=SALT)
    sel_b = pick_user(pop_b, seed=7, salt=SALT)
    assert sel_a.chosen_id == sel_b.chosen_id
    assert sel_a.population_fingerprint == pop_a.fingerprint


def test_unrelated_records_no_effect():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    task = _task(["ref", "b", "c"])
    ref_args = {"city": "Springfield", "primary": True}
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "b": _trace("b", [_ok_call(args=dict(ref_args))]),
        "c": _trace("c", [_ok_call(args={"city": "Shelbyville", "primary": True})]),
        "ghost": _trace("ghost", [_ok_call(args=dict(ref_args))]),
    }
    standings = {"ref": _st(), "b": _st(), "c": _st(), "ghost": _st()}
    rules = {k: _rules() for k in traces}
    pop_b = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert pop_b.fingerprint == pop_a.fingerprint
    sel_a = pick_user(pop_a, seed=3, salt=SALT)
    sel_b = pick_user(pop_b, seed=3, salt=SALT)
    assert sel_a.chosen_id == sel_b.chosen_id


def test_pool_change_changes_fingerprint():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    task = _task(["ref", "b", "c"])
    traces = {
        "ref": _trace("ref", [_ok_call(args={"city": "Springfield", "primary": True})]),
        "b": _trace("b", [_ok_call(args={"city": "Ogdenville", "primary": True})]),
        "c": _trace("c", [_ok_call(args={"city": "Shelbyville", "primary": True})]),
    }
    standings = {"ref": _st(), "b": _st(), "c": _st()}
    rules = {k: _rules() for k in traces}
    pop_b = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert pop_b.fingerprint != pop_a.fingerprint


def test_empty_population_pick_none():
    pop = Population(
        task_id="t1",
        reference_id="ref",
        eligible_ids=(),
        withheld=(),
        support_count=0,
        fingerprint="abc",
    )
    sel = pick_user(pop, seed=0, salt=SALT)
    assert sel.chosen_id is None
    assert sel.task_id == "t1"
    assert sel.seed == 0
    assert sel.population_fingerprint == "abc"


def test_mutation_isolation():
    ref_args = {"city": "Springfield", "primary": True}
    run_ids = ["ref", "ok"]
    task = _task(run_ids)
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "ok": _trace("ok", [_ok_call(args=dict(ref_args))]),
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
    traces["ghost"] = _trace("ghost", [_ok_call(args=dict(ref_args))])
    standings["ghost"] = _st()
    rules["ghost"] = _rules()
    tools.append("extra_tool")
    assert list(pop.eligible_ids) == before_eligible
    assert pop.fingerprint == before_fp
    assert isinstance(pop.eligible_ids, tuple)
    with pytest.raises(ValidationError):
        pop.support_count = 999


def test_no_builder_imports():
    text = Path("kullback/user/population.py").read_text(encoding="utf-8")
    assert "kullback.builder" not in text
    assert "kullback.examiner" not in text
    assert "kullback.rounds" not in text
    assert "user_population" in text


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
        ident = json.dumps(
            [pop_a.task_id, seed, pop_a.fingerprint],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        key = sampling.sample_key("user_population", ident, salt)
        want = sorted(pop_a.eligible_ids)[key % len(pop_a.eligible_ids)]
        assert pick_user(pop_a, seed=seed, salt=salt).chosen_id == want


def test_nonempty_pick_counts_one_draw():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    before = sampling.draws_by_kind()
    sel = pick_user(pop_a, seed=5, salt=SALT)
    assert sel.chosen_id is not None
    assert sampling.draws_since(before) == {"user_population": 1}


def test_empty_pick_counts_no_draw():
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
    assert sampling.draws_since(before) == {}


def test_repeated_draws_identical_and_counted():
    pop_a = _mini_population(["ref", "b", "c"], ["ref", "b", "c"])
    before = sampling.draws_by_kind()
    first = pick_user(pop_a, seed=9, salt=SALT)
    second = pick_user(pop_a, seed=9, salt=SALT)
    assert first == second
    assert sampling.draws_since(before) == {"user_population": 2}


def _two_run_inputs(rules_map):
    task = _task(["ref", "ok"])
    ref_args = {"city": "Springfield", "primary": True}
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "ok": _trace("ok", [_ok_call(args=dict(ref_args))]),
    }
    standings = {"ref": _st(), "ok": _st()}
    return task, traces, standings


def _pop_for_rules(rules_map):
    task, traces, standings = _two_run_inputs(rules_map)
    return build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=dict(rules_map),
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )


def _ident(pop, seed):
    return json.dumps(
        [pop.task_id, seed, pop.fingerprint],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_rules_fact_value_mutation_changes_fingerprint():
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
    assert pop_b.fingerprint != pop_a.fingerprint
    assert _ident(pop_b, 7) != _ident(pop_a, 7)
    key_a = sampling.sample_key("user_population", _ident(pop_a, 7), SALT)
    key_b = sampling.sample_key("user_population", _ident(pop_b, 7), SALT)
    assert key_b != key_a


def test_rules_disclosure_refusal_walkaway_mutation_changes_fingerprint():
    base = {
        "ref": UserRules(
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["email"],
            walk_away=["closing"],
        ),
        "ok": UserRules(
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["email"],
            walk_away=["closing"],
        ),
    }
    pop_a = _pop_for_rules(base)
    for mutated_ok in [
        UserRules(
            disclosure=[DisclosureRule(field="zip", on_request=False)],
            refusals=["email"],
            walk_away=["closing"],
        ),
        UserRules(
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["phone"],
            walk_away=["closing"],
        ),
        UserRules(
            disclosure=[DisclosureRule(field="zip", on_request=True)],
            refusals=["email"],
            walk_away=["other"],
        ),
    ]:
        mutated = {"ref": base["ref"], "ok": mutated_ok}
        pop_b = _pop_for_rules(mutated)
        assert list(pop_b.eligible_ids) == ["ok", "ref"]
        assert pop_b.fingerprint != pop_a.fingerprint


def test_equal_rules_content_stable_across_build_and_key_order():
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
    rules_b = {
        "ok": UserRules(
            **{
                "walk_away": ["closing"],
                "refusals": ["email"],
                "disclosure": [DisclosureRule(**{"on_request": True, "field": "zip"})],
                "facts": [UserFact(**{"context": "stated", "value": "19122", "field": "zip"})],
            }
        ),
        "ref": UserRules(
            **{
                "walk_away": ["closing"],
                "refusals": ["email"],
                "disclosure": [DisclosureRule(**{"on_request": True, "field": "zip"})],
                "facts": [UserFact(**{"context": "stated", "value": "19122", "field": "zip"})],
            }
        ),
    }
    assert _pop_for_rules(rules_a).fingerprint == _pop_for_rules(rules_b).fingerprint


def test_post_build_rule_mutation_does_not_alter_fingerprint():
    ok_rules = UserRules(facts=[UserFact(field="city", value="Springfield")])
    rules_map = {
        "ref": UserRules(facts=[UserFact(field="city", value="Springfield")]),
        "ok": ok_rules,
    }
    pop = _pop_for_rules(rules_map)
    before = pop.fingerprint
    ok_rules.facts.append(UserFact(field="city", value="Shelbyville"))
    assert pop.fingerprint == before
    assert _pop_for_rules(rules_map).fingerprint != before


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


def test_failed_required_write_withheld():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "failed"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "failed": _trace("failed", [_failed_call(args=dict(ref_args))]),
    }
    standings = {"ref": _st(), "failed": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    reasons = _withheld_map(pop)["failed"]
    assert "write_mismatch" in reasons
    assert f"write_failed:{WRITE_TOOL}" in reasons


def test_retry_after_failure_stays_eligible():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "retry"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "retry": _trace(
            "retry",
            [_failed_call(args=dict(ref_args)), _ok_call(args=dict(ref_args))],
        ),
    }
    standings = {"ref": _st(), "retry": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref", "retry"]
    assert "retry" not in _withheld_map(pop)


def test_failed_read_stays_eligible():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "reader"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "reader": _trace(
            "reader",
            [
                _ok_call(args=dict(ref_args)),
                _failed_call(name="get_address", args={"city": "Springfield"}),
            ],
        ),
    }
    standings = {"ref": _st(), "reader": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["reader", "ref"]


def test_fingerprint_moves_with_membership():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "cand"])
    ok_traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "cand": _trace("cand", [_ok_call(args=dict(ref_args))]),
    }
    bad_traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "cand": _trace("cand", [_failed_call(args=dict(ref_args))]),
    }
    standings = {"ref": _st(), "cand": _st()}
    rules = {k: _rules() for k in ok_traces}
    pop_ok = build_population(
        task,
        ok_traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    pop_bad = build_population(
        task,
        bad_traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop_ok.eligible_ids) == ["cand", "ref"]
    assert list(pop_bad.eligible_ids) == ["ref"]
    assert pop_bad.fingerprint != pop_ok.fingerprint


def test_malformed_error_withheld():
    ref_args = {"city": "Springfield", "primary": True}
    broken = ToolCall.model_construct(
        name=WRITE_TOOL,
        args=dict(ref_args),
        result=None,
        error="boom",
        has_result=True,
        resolved=True,
        truncated=False,
        raw_ptr=_ptr(),
    )
    task = _task(["ref", "broken"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "broken": _trace("broken", [broken]),
    }
    standings = {"ref": _st(), "broken": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert f"malformed_error:{WRITE_TOOL}" in _withheld_map(pop)["broken"]


def test_indeterminate_extra_write_withheld():
    ref_args = {"city": "Springfield", "primary": True}
    tools = [WRITE_TOOL, "set_email"]
    task = _task(["ref", "extra"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "extra": _trace(
            "extra",
            [
                _ok_call(args=dict(ref_args)),
                _failed_call(name="set_email", args={"email": "a@b.c"}, error_class="transient"),
            ],
        ),
    }
    standings = {"ref": _st(), "extra": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(tools),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert "write_indeterminate:set_email" in _withheld_map(pop)["extra"]


def test_indeterminate_retry_still_withheld():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "retry"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "retry": _trace(
            "retry",
            [
                _failed_call(args=dict(ref_args), error_class="transient"),
                _ok_call(args=dict(ref_args)),
            ],
        ),
    }
    standings = {"ref": _st(), "retry": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["ref"]
    assert f"write_indeterminate:{WRITE_TOOL}" in _withheld_map(pop)["retry"]


def test_deterministic_extra_rejection_stays_eligible():
    ref_args = {"city": "Springfield", "primary": True}
    tools = [WRITE_TOOL, "set_email"]
    task = _task(["extra", "ref"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "extra": _trace(
            "extra",
            [
                _ok_call(args=dict(ref_args)),
                _failed_call(
                    name="set_email", args={"email": "a@b.c"}, error_class="invalid_arguments"
                ),
            ],
        ),
    }
    standings = {"ref": _st(), "extra": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(tools),
    )
    assert list(pop.eligible_ids) == ["extra", "ref"]


def test_reference_indeterminate_write_raises():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["ref", "ok"])
    traces = {
        "ref": _trace(
            "ref",
            [
                _ok_call(args=ref_args),
                _failed_call(args={"city": "Shelbyville", "primary": True}, error_class="unknown"),
            ],
        ),
        "ok": _trace("ok", [_ok_call(args=dict(ref_args))]),
    }
    standings = {"ref": _st(), "ok": _st()}
    rules = {k: _rules() for k in traces}
    with pytest.raises(ValueError, match="write_indeterminate"):
        build_population(
            task,
            traces,
            reference_id="ref",
            standings=standings,
            rules=rules,
            held_out_ids=set(),
            write_tools=list(WRITE_TOOLS),
        )


def test_indeterminate_read_stays_eligible():
    ref_args = {"city": "Springfield", "primary": True}
    task = _task(["reader", "ref"])
    traces = {
        "ref": _trace("ref", [_ok_call(args=ref_args)]),
        "reader": _trace(
            "reader",
            [
                _ok_call(args=dict(ref_args)),
                _failed_call(name="get_address", args={"city": "Springfield"}, error_class="transient"),
            ],
        ),
    }
    standings = {"ref": _st(), "reader": _st()}
    rules = {k: _rules() for k in traces}
    pop = build_population(
        task,
        traces,
        reference_id="ref",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=list(WRITE_TOOLS),
    )
    assert list(pop.eligible_ids) == ["reader", "ref"]
