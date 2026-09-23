"""The one factory builds the same driver down every call path (G5).

The domain is invented: a garden centre that delivers plants, with a delivery slot, a plot number
and a membership tier. Nothing here names any customer's world.
"""

from __future__ import annotations

import pytest

from kullback.ai.provider import TestModel
from kullback.builder import intent as intent_mod
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn
from kullback.user import context as context_mod
from kullback.user import factory as factory_mod
from kullback.user import fidelity as fidelity_mod
from kullback.user import rules as rules_mod
from kullback.user import simulated as simulated_mod
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
    """An invented workdir with one Task, written through the writers the build uses."""
    from kullback.runner import canon as canon_mod
    from kullback.runner.canon import CanonRules
    from kullback.runner.records import (
        EntitySchema,
        ToolSig,
        as_dict,
        write_json,
    )
    trace = recorded_trace()
    rules = rules_mod.derive_user_rules(trace, VOCAB, writes=["move_delivery"])
    write_json(tmp_path / "vocabulary.json", as_dict(VOCAB))
    write_json(tmp_path / "tool_sigs.json",
               [as_dict(ToolSig(name="move_delivery", kind="write"))])
    write_json(tmp_path / "replays.json",
               {"task_1": {"t1": {"trace_id": trace.trace_id, "confirmed": True}}})
    write_json(tmp_path / "traces" / "abc.json", as_dict(trace))
    write_json(tmp_path / "user_rules" / f"{trace.trace_id}.json", as_dict(rules))
    write_json(tmp_path / "schema.json", as_dict(EntitySchema()))
    canon_mod.save_rules(CanonRules(), tmp_path / "canon-rules.json")
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
    fallback = simulated_mod.SimulatedUser(user_rules, vocab=fidelity_mod.vocabulary_of(workdir))
    scorer = factory_mod.build_user(workdir, "task_1", model, factory_mod.PURPOSE_SCORE,
                                    ctx=ctx, fallback=fallback, record_values=record)
    assert isinstance(scorer, AgentUser)

    live_ctx = context_mod.curate("task_1", user_rules, trace, vocab=VOCAB, write_tools=writes,
                                  record_fields=sorted(record), lessons=[])
    live_floor = simulated_mod.SimulatedUser(user_rules, vocab=VOCAB)
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


def test_an_examination_with_no_reference_is_nothing_to_examine(workdir):
    assert factory_mod.build_user(workdir, "missing", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE) is None


@pytest.mark.parametrize("task_id,purpose", [
    ("task_1", "examine"),
    ("missing", factory_mod.PURPOSE_RUN),
], ids=["unknown-purpose", "run-with-no-reference-and-no-live-inputs"])
def test_an_unknown_purpose_or_a_run_with_nothing_to_drive_is_an_error(workdir, task_id, purpose):
    with pytest.raises(ValueError):
        factory_mod.build_user(workdir, task_id, TestModel([], loop=True), purpose)


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
        fallback=simulated_mod.SimulatedUser(user_rules, vocab=VOCAB), vocab=VOCAB,
        write_tools=["move_delivery"], goal_writes={"move_delivery"}, answer_strip=None,
        record_values=record, trace=trace)
    assert isinstance(user, AgentUser)


def test_the_disk_fallback_carries_write_tools_goal_writes_and_strip(workdir):
    """The disk fallback ends and strips like the dict's own values, not unchecked."""
    user = factory_mod.build_user(workdir, "task_1", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE)
    assert isinstance(user, AgentUser)
    fallback = user.fallback
    assert sorted(fallback.write_tools) == sorted(user.write_tools)
    assert fallback.goal_writes is not None
    assert sorted(fallback.goal_writes) == sorted(user.protocol.goal_writes)
    assert fallback.answer_strip is not None
    for probe in ("My plot number is PLOT-4471.", "The courier reference is CR-90881.",
                  "Hello."):
        assert fallback.answer_strip(probe) == user.guards.strip(probe)


def test_the_disk_derivation_lists_the_tasks_own_recordings_as_members(workdir):
    """Members off disk are the Task's own recordings, in a fixed order."""
    from kullback.builder import intent as intent_mod
    from kullback.builder import user_sim as user_sim_mod
    from kullback.runner import canon as canon_mod
    from kullback.runner.records import EntitySchema, read_json

    index = fidelity_mod.trace_index(workdir)
    disk_members = factory_mod._members_of(workdir, "task_1", index)
    assert [t.trace_id for t in disk_members] == ["t1"]

    writes = fidelity_mod.write_tools_of(workdir)
    assert writes == ["move_delivery"]
    reference = index["t1"]
    assert user_sim_mod.goal_write_set is rules_mod.goal_write_set
    user = factory_mod.build_user(workdir, "task_1", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE)
    assert user.protocol.goal_writes == user_sim_mod.goal_write_set(reference, writes)
    assert user.trace is not None and user.trace.trace_id == reference.trace_id

    schema = EntitySchema.model_validate(read_json(workdir / "schema.json", {}))
    live_strip = intent_mod.value_strip(disk_members, schema=schema,
                                        rules=canon_mod.load_rules(workdir / "canon-rules.json"))
    direct = intent_mod.value_strip([reference], schema=None, rules=None)
    assert user.guards.strip is not None
    for probe in ("My plot number is PLOT-4471.", "The courier reference is CR-90881.",
                  "Hello."):
        assert user.guards.strip(probe) == live_strip(probe)
        assert user.guards.strip(probe) == direct(probe)


def test_the_built_user_carries_the_workdirs_bus_unless_the_caller_passes_one(workdir):
    """No bus passed means the workdir's own, named for this agent; a caller's own is kept."""
    from kullback.agent.bus import Bus
    user = factory_mod.build_user(workdir, "task_1", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE)
    assert user.bus is not None
    assert user.bus.path == workdir / "bus.jsonl"
    assert user.bus.agent == "user"
    assert user.harness().bus is user.bus

    bus = Bus(workdir / "custom.jsonl", "user")
    user = factory_mod.build_user(workdir, "task_1", TestModel([], loop=True),
                                  factory_mod.PURPOSE_SCORE, bus=bus)
    assert user.bus is bus
    assert user.harness().bus is bus
