"""The one factory builds the same driver down every call path (G5).

The domain is invented: a garden centre that delivers plants, with a delivery slot, a plot number
and a membership tier. Nothing here names any customer's world.
"""

from __future__ import annotations

import pytest

from kullback.ai.provider import TestModel
from kullback.builder import intent as intent_mod
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, UserRules
from kullback.user import context as context_mod
from kullback.user import factory as factory_mod
from kullback.user import fidelity as fidelity_mod
from kullback.user import rules as rules_mod
from kullback.user.agent import AgentUser
from kullback.user.vocabulary import GENERIC_FIELDS, FieldSpec, Vocabulary

PTR = RawPtr(file_hash="testfile", sim_index=0)

VOCAB = Vocabulary(domain="garden", fields=[
    *[f.model_copy(deep=True) for f in GENERIC_FIELDS],
    FieldSpec(field="plot_id", kind="reference", pattern=r"\bPLOT-\d{4}\b",
              cues=[r"\bplot (?:id|number)\b", r"\byour plot\b"], aliases=["plot id"]),
    FieldSpec(field="delivery_slot", kind="value", pattern=r"\b(?:0[1-9]|1[0-9]|2[0-3]):00\b",
              cues=[r"\bdelivery slot\b", r"\bwhat time\b"], aliases=["delivery slot"]),
])


def turn(idx: int, role: str, content: str) -> Turn:
    return Turn(idx=idx, role=role, content=content, tool_call_ids=[], raw_ptr=PTR)


def recorded_trace() -> Trace:
    return Trace(trace_id="t1", raw_hash="h", ingest_version="v", source="test", raw_ptr=PTR,
                 turns=[
                     turn(0, "user", "Hello, please could you move my plant delivery to a later slot."),
                     turn(1, "assistant", "Of course. Could you provide your plot number?"),
                     turn(2, "user", "Thanks, my plot number is PLOT-4471."),
                     turn(3, "assistant", "Thank you. What delivery slot would you like?"),
                     turn(4, "user", "Please make it 16:00."),
                     turn(5, "assistant", "Done, your delivery is moved. Anything else?"),
                     turn(6, "user", "No, that is all. Thank you."),
                 ],
                 tool_calls=[ToolCall(id="c1", name="move_delivery",
                                      args={"plot_id": "PLOT-4471", "slot": "16:00"},
                                      result={"plot_id": "PLOT-4471", "slot": "16:00",
                                              "courier_ref": "CR-90881"},
                                      raw_ptr=PTR)])


@pytest.fixture
def workdir(tmp_path):
    """An invented workdir with one Task, the way a build leaves it on disk."""
    from kullback.runner.records import write_json
    trace = recorded_trace()
    rules = rules_mod.derive_user_rules(trace, VOCAB, writes=["move_delivery"])
    write_json(tmp_path / "vocabulary.json", VOCAB.model_dump(mode="json"))
    write_json(tmp_path / "tool_sigs.json", [{"name": "move_delivery", "kind": "write"}])
    write_json(tmp_path / "replays.json",
               {"task_1": {"t1": {"trace_id": trace.trace_id, "confirmed": True}}})
    write_json(tmp_path / "traces" / "abc.json", trace.model_dump(mode="json"))
    write_json(tmp_path / "user_rules" / f"{trace.trace_id}.json", rules.model_dump(mode="json"))
    return tmp_path


def config_of(user: AgentUser) -> dict:
    """The configuration every call path must agree on, as comparable values."""
    return {
        "vocab": user.vocab.model_dump(mode="json"),
        "write_tools": sorted(user.write_tools),
        "goal_writes": sorted(user.protocol.goal_writes)
        if user.protocol.goal_writes is not None else None,
        "record_values": dict(user.guards.record_values or {}),
        "trace_id": user.trace.trace_id if user.trace is not None else None,
        "facts": [(f.field, str(f.value)) for f in user.ctx.facts],
        "goal": user.ctx.goal,
        "grounding": list(user.guards.grounding.values),
        "has_strip": user.guards.strip is not None,
    }


def test_the_three_call_paths_build_users_with_equal_configuration(workdir):
    """Rounds, CLI and build paths meet the same inputs on one invented Task (G5)."""
    model = TestModel([], loop=True)
    index = fidelity_mod.trace_index(workdir)
    trace = index["t1"]
    writes = ["move_delivery"]

    minimal = factory_mod.build_user(workdir, "task_1", model, factory_mod.PURPOSE_SCORE)
    assert isinstance(minimal, AgentUser)

    ctx, user_rules, _trace, record = next(iter(
        fidelity_mod.contexts_of(workdir).values()))
    fallback = rules_mod.SimulatedUser(user_rules, vocab=fidelity_mod.vocabulary_of(workdir))
    scorer = factory_mod.build_user(workdir, "task_1", model, factory_mod.PURPOSE_SCORE,
                                    ctx=ctx, fallback=fallback, record_values=record)
    assert isinstance(scorer, AgentUser)

    live_ctx = context_mod.curate("task_1", user_rules, trace, vocab=VOCAB, write_tools=writes,
                                  record_fields=sorted(record), lessons=[])
    live_floor = rules_mod.SimulatedUser(user_rules, vocab=VOCAB)
    live_strip = intent_mod.value_strip([trace], schema=None, rules=None)
    live = factory_mod.build_user(
        workdir, "task_1", model, factory_mod.PURPOSE_RUN, ctx=live_ctx, fallback=live_floor,
        vocab=VOCAB, write_tools=writes, goal_writes=rules_mod.goal_write_set(trace, writes),
        answer_strip=live_strip, record_values=record, trace=trace)
    assert isinstance(live, AgentUser)

    assert config_of(scorer) == config_of(minimal)
    assert config_of(live) == config_of(minimal)
    for probe in ("My plot number is PLOT-4471.", "Hello."):
        assert live.guards.strip(probe) == minimal.guards.strip(probe)


def test_the_factory_wires_the_builds_own_strip_over_the_tasks_evidence(workdir):
    """The scorer's user strips what the build's strip strips, on the same Task (G5)."""
    from kullback.builder import intent as intent_mod
    model = TestModel([], loop=True)
    user = factory_mod.build_user(workdir, "task_1", model, factory_mod.PURPOSE_SCORE)
    assert isinstance(user, AgentUser)
    index = fidelity_mod.trace_index(workdir)
    direct = intent_mod.value_strip([index["t1"]], schema=None, rules=None)
    probes = ["My plot number is PLOT-4471.", "The courier reference is CR-90881.", "Hello."]
    for probe in probes:
        assert user.guards.strip is not None
        assert user.guards.strip(probe) == direct(probe)


def test_an_examination_with_no_reference_is_nothing_to_examine(workdir):
    assert factory_mod.build_user(workdir, "missing", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE) is None


def test_a_run_with_no_reference_and_no_live_inputs_is_an_error(workdir):
    with pytest.raises(ValueError):
        factory_mod.build_user(workdir, "missing", TestModel([], loop=True),
                               factory_mod.PURPOSE_RUN)


def test_an_unknown_purpose_is_an_error(workdir):
    with pytest.raises(ValueError):
        factory_mod.build_user(workdir, "task_1", TestModel([], loop=True), "examine")


def test_a_rules_floor_is_not_an_agent_user(workdir):
    """The factory builds the driver over the floor; the floor alone is not one."""
    rules: UserRules = fidelity_mod.rules_for(workdir, "t1")
    floor = rules_mod.SimulatedUser(rules, vocab=VOCAB)
    assert not isinstance(floor, AgentUser)
    user = factory_mod.build_user(workdir, "task_1", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE, fallback=floor)
    assert isinstance(user, AgentUser)
    assert user.fallback is floor


def test_full_live_inputs_never_touch_the_disk(workdir):
    """A caller that passes everything gets no disk read and no missing-file error (P2)."""
    model = TestModel([], loop=True)
    index = fidelity_mod.trace_index(workdir)
    trace = index["t1"]
    user_rules = fidelity_mod.rules_for(workdir, "t1")
    record = context_mod.mine_record_values(trace)
    ctx = context_mod.curate("task_1", user_rules, trace, vocab=VOCAB,
                             write_tools=["move_delivery"], record_fields=sorted(record))
    user = factory_mod.build_user(
        workdir / "nowhere", "task_1", model, factory_mod.PURPOSE_RUN, ctx=ctx,
        fallback=rules_mod.SimulatedUser(user_rules, vocab=VOCAB), vocab=VOCAB,
        write_tools=["move_delivery"], goal_writes={"move_delivery"}, answer_strip=None,
        record_values=record, trace=trace)
    assert isinstance(user, AgentUser)
