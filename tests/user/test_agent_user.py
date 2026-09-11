"""The Simulated user as an agent of the harness: its curated context, its guards, its score, its
lessons and the driver choice (D214).

The domain is invented: a garden centre that delivers plants, with a delivery slot, a plot number
and a membership tier. Nothing here names any customer's world.
"""

from __future__ import annotations

import pytest

from kullback.ai.provider import TestModel
from kullback.runner.records import (
    DisclosureRule,
    RawPtr,
    ToolCall,
    Trace,
    Turn,
    UserFact,
    UserRules,
)
from kullback.user import context as context_mod
from kullback.user import fidelity as fidelity_mod
from kullback.user import guards as guards_mod
from kullback.user import lesson as lesson_mod
from kullback.user import rules as rules_mod
from kullback.user.agent import AgentUser
from kullback.user.tools import Toolbox, user_tools
from kullback.user.vocabulary import GENERIC_FIELDS, FieldSpec, Vocabulary

# The shared conftest defines this too, but two test folders both name a module `conftest` and
# which one answers `from conftest import ...` depends on the order pytest put them on the path.
PTR = RawPtr(file_hash="testfile", sim_index=0)

VOCAB = Vocabulary(domain="garden", fields=[
    *[f.model_copy(deep=True) for f in GENERIC_FIELDS],
    FieldSpec(field="plot_id", kind="reference", pattern=r"\bPLOT-\d{4}\b",
              cues=[r"\bplot (?:id|number)\b", r"\byour plot\b"], aliases=["plot id"]),
    FieldSpec(field="delivery_slot", kind="value", pattern=r"\b(?:0[1-9]|1[0-9]|2[0-3]):00\b",
              cues=[r"\bdelivery slot\b", r"\bwhat time\b"], aliases=["delivery slot"]),
])


def turn(idx: int, role: str, content: str, calls: list[str] = ()) -> Turn:
    return Turn(idx=idx, role=role, content=content, tool_call_ids=list(calls), raw_ptr=PTR)


def trace_of(turns: list[Turn], calls: list[ToolCall] = ()) -> Trace:
    return Trace(trace_id="t1", raw_hash="h", ingest_version="v", source="test",
                 turns=turns, tool_calls=list(calls), raw_ptr=PTR)


@pytest.fixture
def recorded() -> Trace:
    """Three user turns: the goal, a plot id when asked, a slot when asked, then the close."""
    return trace_of([
        turn(0, "user", "Hello, please could you move my plant delivery to a later slot."),
        turn(1, "assistant", "Of course. Could you provide your plot number?"),
        turn(2, "user", "Thanks, my plot number is PLOT-4471."),
        turn(3, "assistant", "Thank you. What delivery slot would you like?"),
        turn(4, "user", "Please make it 16:00."),
        turn(5, "assistant", "Done, your delivery is moved. Anything else?"),
        turn(6, "user", "No, that is all. Thank you."),
    ], [ToolCall(id="c1", name="move_delivery", args={"plot_id": "PLOT-4471", "slot": "16:00"},
                 result={"plot_id": "PLOT-4471", "slot": "16:00", "courier_ref": "CR-90881"},
                 raw_ptr=PTR)])


@pytest.fixture
def rules(recorded) -> UserRules:
    return rules_mod.derive_user_rules(recorded, VOCAB, writes=["move_delivery"])


@pytest.fixture
def ctx(recorded, rules) -> context_mod.TaskContext:
    record = context_mod.mine_record_values(recorded)
    return context_mod.curate("task_1", rules, recorded, vocab=VOCAB, write_tools=["move_delivery"],
                              record_fields=sorted(record))


# --- the package's own shape ---------------------------------------------------------------

def test_the_user_package_imports_neither_of_the_two_agents_that_drive_it():
    """D214: the Builder derives the vocabulary and drives the Runs, so the user may not depend back."""
    import ast
    import pathlib

    import kullback.user as package
    allowed = {"kullback.agent", "kullback.ai", "kullback.gates", "kullback.runner", "kullback.user"}
    for path in pathlib.Path(package.__path__[0]).glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [alias.name for alias in getattr(node, "names", ())] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("kullback"):
                    root = ".".join(name.split(".")[:2])
                    assert root in allowed, f"{path.name} imports {name}"


def test_the_rule_driven_user_is_still_reachable_under_the_name_it_moved_from():
    """The re-export the move left behind: every D210 caller keeps working."""
    from kullback.builder import user_sim
    assert user_sim.SimulatedUser is rules_mod.SimulatedUser
    assert user_sim.GOAL_SATISFIED == rules_mod.GOAL_SATISFIED


# --- what is curated ------------------------------------------------------------------------

def test_a_persona_is_mined_from_the_user_turns_as_counted_classes_and_carries_no_quote(recorded, rules):
    persona = context_mod.mine_persona(recorded, rules)
    assert persona.turns == 4
    assert persona.tone == context_mod.COURTEOUS
    assert persona.ends == context_mod.CLOSING_LINE
    body = persona.model_dump_json()
    assert "PLOT-4471" not in body and "plant delivery" not in body


def test_the_values_only_the_world_returned_are_mined_as_record_facts(recorded):
    """A value a tool returned that no user turn ever said is the world's, not the user's (D210)."""
    record = context_mod.mine_record_values(recorded)
    assert record.get("courier_ref") == "CR-90881"
    assert "plot_id" not in record  # the user said it, so it is askable


def test_consult_is_absent_where_the_recording_shows_the_user_looking_nothing_up(ctx):
    assert ctx.consultations == []
    names = [tool.name for tool in user_tools(Toolbox(ctx))]
    assert "consult" not in names
    assert names == ["my_facts", "my_goal", "what_i_said", "end_run"]


def test_consult_appears_where_the_user_says_a_value_only_the_worlds_records_hold():
    """The mined class: the user reads out a courier reference nobody in the conversation gave them."""
    looked_up = trace_of([
        turn(0, "user", "Hello, I need my plant delivery moved."),
        turn(1, "assistant", "Could you tell me which delivery this is about?"),
        turn(2, "user", "One moment, let me check the paperwork. The courier reference is CR-90881."),
    ], [ToolCall(id="c1", name="find_delivery", args={"plot_id": "PLOT-8812"},
                 result={"courier_ref": "CR-90881", "slot": "09:00"}, raw_ptr=PTR)])
    assert context_mod.mine_consultations(looked_up) == ["courier_ref"]
    assert "courier_ref" not in context_mod.mine_record_values(looked_up)


def test_the_reference_write_arguments_are_the_values_this_user_chooses(ctx):
    assert ctx.choices["slot"] == "16:00"


def test_every_curated_item_is_a_section_under_a_stable_tag_and_an_empty_one_is_absent(ctx):
    names = [section.name for section in context_mod.sections(ctx)]
    assert context_mod.GOAL_TAG in names and context_mod.PERSONA_TAG in names
    assert context_mod.LESSONS_TAG not in names  # no lesson yet, so no blank section
    assert set(names) <= set(context_mod.SECTION_TAGS)


# --- the guards ------------------------------------------------------------------------------

def guards_for(ctx, record=None, strip=None) -> guards_mod.Guards:
    return guards_mod.Guards(guards_mod.grounding_for(ctx.askable(), ctx.grounding()),
                             record_values=record, strip=strip)


def test_a_number_this_user_never_said_is_dropped_and_counted(ctx):
    outcome = guards_for(ctx).check("My plot number is PLOT-9999.")
    assert outcome.dropped and outcome.reason == guards_mod.INVENTED


def test_a_value_this_user_did_say_goes_through(ctx):
    outcome = guards_for(ctx).check("My plot number is PLOT-4471.")
    assert not outcome.dropped and "PLOT-4471" in outcome.text


def test_a_record_fact_is_replaced_by_the_sentence_that_points_at_it(ctx, recorded):
    record = context_mod.mine_record_values(recorded)
    outcome = guards_for(ctx, record=record).check("The courier reference is CR-90881.")
    assert not outcome.dropped
    assert "CR-90881" not in outcome.text
    assert guards_mod.RECORD_FACT in outcome.changed


def test_the_strip_takes_a_sentence_this_user_could_never_have_said(ctx):
    def strip(text):
        return (text, ["a value only the system knew"]) if "PLOT-4471" in text else (text, [])

    outcome = guards_for(ctx, strip=strip).check("My plot number is PLOT-4471.")
    assert outcome.dropped and outcome.reason == guards_mod.STRIPPED


def test_volunteering_past_the_recorded_point_is_cut(ctx):
    guards = guards_for(ctx)
    outcome = guards.check("My plot number is PLOT-4471. My slot is 16:00.", facts_allowed=1)
    assert not outcome.dropped
    assert guards_mod.OVER_DISCLOSED in outcome.changed
    assert "16:00" not in outcome.text


def test_the_disclosure_allowance_is_counted_in_facts_and_not_in_questions(recorded, rules):
    """A Candidate that asks many questions cannot farm facts: the limit is what the recorded user
    had said by its own turn number, whatever it was asked to get there."""
    assert context_mod.volunteer_allowance(recorded, rules, 1) == 0
    assert context_mod.volunteer_allowance(recorded, rules, 2) == 1
    assert context_mod.volunteer_allowance(recorded, rules, 3) == 2


# --- the end protocol -------------------------------------------------------------------------

def test_the_agent_may_only_request_an_end_and_code_decides_it(ctx):
    box = Toolbox(ctx)
    out = box.end(rules_mod.GOAL_SATISFIED)
    assert out.requested == rules_mod.GOAL_SATISFIED and out.accepted is False
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    assert protocol.kind("Anything else?", said_anything=True, had_nothing=False, made=set(),
                         requested=rules_mod.GOAL_SATISFIED) == rules_mod.HANDED_OFF
    assert protocol.kind("Anything else?", said_anything=True, had_nothing=False,
                         made={"move_delivery"}) == rules_mod.GOAL_SATISFIED


def test_a_candidate_that_twice_asks_what_nobody_told_this_user_runs_the_scenario_out():
    # Communicate-only: a write Task with implied writes still unmade is the sibling below (D251).
    protocol = guards_mod.EndProtocol()
    assert protocol.kind("What is your tier?", said_anything=True, had_nothing=True) is None
    assert protocol.kind("And your tier?", said_anything=True,
                         had_nothing=True) == rules_mod.SCENARIO_EXHAUSTED


# --- the agent user ---------------------------------------------------------------------------

def agent_for(ctx, rules, replies, recorded=None, record=None) -> AgentUser:
    fallback = rules_mod.SimulatedUser(rules, vocab=VOCAB)
    return AgentUser(ctx, fallback, TestModel(replies, loop=True), vocab=VOCAB,
                     write_tools=["move_delivery"], goal_writes=["move_delivery"],
                     record_values=record, trace=recorded)


def test_the_model_drives_the_turn_when_every_guard_passes(ctx, rules, recorded):
    user = agent_for(ctx, rules, ["Sure, it is PLOT-4471."], recorded)
    said = user.reply([{"role": "assistant", "content": "Could you provide your plot number?"}])
    assert "PLOT-4471" in said
    assert user.counts["agent_user_turns"] == 1 and user.counts["fallback_turns"] == 0


def test_the_rule_driven_user_answers_the_beat_a_guard_dropped(ctx, rules, recorded):
    user = agent_for(ctx, rules, ["Sure, it is PLOT-1111."], recorded)
    said = user.reply([{"role": "assistant", "content": "Could you provide your plot number?"}])
    assert "PLOT-1111" not in said
    assert user.counts["fallback_turns"] == 1
    assert user.guards.counts[guards_mod.INVENTED] == 1
    assert user.events[-1].payload["agent_turn_dropped"] == guards_mod.INVENTED


def test_a_run_with_no_user_model_is_the_rule_driven_user_it_always_was(ctx, rules, recorded):
    user = AgentUser(ctx, rules_mod.SimulatedUser(rules, vocab=VOCAB), None, vocab=VOCAB,
                     trace=recorded)
    said = user.reply([{"role": "assistant", "content": "Could you provide your plot number?"}])
    assert "PLOT-4471" in said and user.counts["agent_user_turns"] == 0


# --- the fidelity score -----------------------------------------------------------------------

def test_the_score_is_one_for_the_recorded_users_own_turns(recorded, rules):
    facts = [f for f in rules.facts if f.field not in rules_mod.SPOKEN_FIELDS]
    turns = fidelity_mod.recorded_turns(recorded)
    scores = [fidelity_mod.score_turn(t.content, t.content, facts,
                                      ended=(t is turns[-1]), recorded_ended=(t is turns[-1]))
              for t in turns]
    assert fidelity_mod.score_of(scores) == 1.0


def test_a_turn_that_misses_a_fact_scores_below_the_recorded_one(recorded, rules):
    facts = [f for f in rules.facts if f.field not in rules_mod.SPOKEN_FIELDS]
    missed = fidelity_mod.score_turn("I do not have that to hand.",
                                     "Thanks, my plot number is PLOT-4471.", facts)
    assert missed.score < 1.0 and missed.missed == ["plot_id"]


def test_a_turn_that_reads_out_a_record_value_loses_the_record_reading(recorded, rules):
    facts = [f for f in rules.facts if f.field not in rules_mod.SPOKEN_FIELDS]
    record = context_mod.mine_record_values(recorded)
    leaked = fidelity_mod.score_turn("The courier reference is CR-90881.", "Please make it 16:00.",
                                     facts, record_values=record)
    assert leaked.record_clean == 0.0 and leaked.record_spoken == ["courier_ref"]


def test_the_rule_driven_user_is_scored_with_no_model_call(recorded, rules):
    score = fidelity_mod.score_driver(rules_mod.SimulatedUser(rules, vocab=VOCAB), recorded,
                                      [f for f in rules.facts], task_id="task_1")
    assert 0.0 <= score.score <= 1.0 and score.turns > 0


# --- the lesson loop ---------------------------------------------------------------------------

def lesson(agent: float, rules_score: float, round: int, key: str = "k") -> lesson_mod.Lesson:
    return lesson_mod.Lesson(task_id="task_1", round=round, key=key, agent=agent, rules=rules_score,
                             missed=["plot_id"])


def test_a_lesson_names_the_missed_fact_and_lands_in_the_next_context(ctx):
    lines = lesson_mod.lines_for([lesson(0.4, 0.6, 1)], "task_1")
    assert any("plot_id" in line for line in lines)
    next_ctx = ctx.model_copy(update={"lessons": lines})
    section = next(s for s in context_mod.sections(next_ctx) if s.name == context_mod.LESSONS_TAG)
    assert "plot_id" in section.text


def test_a_task_the_agent_never_beats_stops_paying_after_three_rounds(ctx):
    history = [lesson(0.4, 0.6, n) for n in (1, 2)]
    assert not lesson_mod.stalled(history, "task_1")
    history.append(lesson(0.4, 0.6, 3))
    assert lesson_mod.stalled(history, "task_1")
    assert not lesson_mod.drives(history, "task_1", 0.9, 0.6)


def test_a_task_whose_facts_changed_starts_paying_again(ctx):
    history = [lesson(0.4, 0.6, n) for n in (1, 2, 3)]
    assert lesson_mod.stalled(history, "task_1", "k")
    assert not lesson_mod.stalled(history, "task_1", "a-new-key")


def test_the_agent_drives_a_task_only_where_it_beats_the_rules():
    assert lesson_mod.drives([], "task_1", 0.8, 0.6)
    assert not lesson_mod.drives([], "task_1", 0.6, 0.8)
    assert not lesson_mod.drives([], "task_1", None, 0.8)


# --- what the table says ------------------------------------------------------------------------

def test_the_corpus_table_names_both_drivers_and_where_the_agent_won():
    rows = [{"task_id": "task_1", "rules": 0.5, "agent": 0.9},
            {"task_id": "task_2", "rules": 1.0, "agent": 0.4}]
    body = {"format": fidelity_mod.FORMAT, "tasks": rows, "summary": fidelity_mod.summarise(rows)}
    lines = fidelity_mod.markdown_table(body)
    assert any(line.startswith("| rules |") for line in lines)
    assert any(line.startswith("| agent |") for line in lines)
    assert "agent beats rules on 1 of 2" in lines[-1]


def test_a_facts_tool_answers_only_what_was_asked(ctx):
    box = Toolbox(ctx)
    assert "PLOT-4471" in box.facts(["plot_id"])
    assert "16:00" not in box.facts(["plot_id"])


def test_the_user_never_holds_a_tool_that_reads_the_world(ctx):
    from kullback.user.extension import names_world
    assert names_world({"path": "db.json"}) == "db.json"
    assert names_world({"field": "plot_id"}) is None


def test_a_disclosure_rule_the_recording_made_is_what_the_persona_counted(recorded, rules):
    volunteered = [r for r in rules.disclosure if isinstance(r, DisclosureRule) and not r.on_request]
    assert context_mod.mine_persona(recorded, rules).volunteered == len(volunteered)


def test_a_fact_the_user_gave_is_askable_and_a_value_only_the_world_holds_is_not(rules):
    assert rules_mod.fact_class(rules, "plot_id", "PLOT-4471") == rules_mod.ASKABLE
    assert rules_mod.fact_class(rules, "courier_ref", "CR-90881") == rules_mod.RECORD


def test_a_fact_carried_by_a_turn_is_read_the_same_way_for_both_drivers():
    facts = [UserFact(field="plot_id", value="PLOT-4471", span=PTR, context="my plot is PLOT-4471")]
    assert fidelity_mod.facts_carried("It is PLOT-4471.", facts) == ["plot_id"]
    assert fidelity_mod.facts_carried("it is plot-4471.", facts) == ["plot_id"]  # a value, not a spelling
    assert fidelity_mod.facts_carried("It is PLOT-4472.", facts) == []


def test_the_cli_scores_a_workdirs_tasks_and_prints_the_per_corpus_table(tmp_path, recorded, rules):
    """`kullback user fidelity` off the records a build leaves, with no model and so no spend."""
    from typer.testing import CliRunner

    from kullback import cli
    from kullback.runner.records import write_json
    write_json(tmp_path / "vocabulary.json", VOCAB.model_dump(mode="json"))
    write_json(tmp_path / "tool_sigs.json", [{"name": "move_delivery", "kind": "write"}])
    write_json(tmp_path / "replays.json",
               {"task_1": {"r1": {"trace_id": recorded.trace_id, "confirmed": True}}})
    write_json(tmp_path / "traces" / "abc.json", recorded.model_dump(mode="json"))
    write_json(tmp_path / "user_rules" / f"{recorded.trace_id}.json", rules.model_dump(mode="json"))

    result = CliRunner().invoke(cli.app, ["user", "fidelity", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "| rules |" in result.output and "| agent |" in result.output
    written = fidelity_mod.load_scores(tmp_path)
    assert [row["task_id"] for row in written["tasks"]] == ["task_1"]


def test_the_cli_says_how_the_simulated_user_ended_a_builds_runs(tmp_path):
    from typer.testing import CliRunner

    from kullback import cli
    from kullback.runner.records import write_json
    write_json(tmp_path / "runs" / "task_1" / "rerolls.json",
               {"rolls": [{"user_end": rules_mod.GOAL_SATISFIED},
                          {"user_end": rules_mod.SCENARIO_EXHAUSTED}]})
    result = CliRunner().invoke(cli.app, ["user", "dry-run", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert rules_mod.SCENARIO_EXHAUSTED in result.output


# --- D227: both users read one function, and the refused-write count is a round count ------------


DELIVERY_MOVED = {"role": "tool", "tool_call_id": "c1", "name": "move_delivery", "content": "{}"}
DELIVERY_REFUSED = {**DELIVERY_MOVED,
                    "content": "that slot is full",
                    "error": {"class": "business_error", "payload": "that slot is full"}}


def test_the_agent_users_end_and_the_rules_read_writes_through_one_function():
    assert guards_mod.writes_made is not rules_mod.writes_made
    transcript = [{"role": "user", "content": "Hi."}, DELIVERY_REFUSED]
    assert guards_mod.writes_made(transcript, ["move_delivery"]) == \
        rules_mod.writes_made(transcript, ["move_delivery"]) == set()


def test_the_end_protocol_does_not_satisfy_a_goal_on_a_write_the_world_refused():
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    made = guards_mod.writes_made([{"role": "user", "content": "Hi."}, DELIVERY_REFUSED],
                                  ["move_delivery"])
    assert protocol.goal_done(made) is False
    assert protocol.kind("Anything else?", said_anything=True, had_nothing=False,
                         made=made) == rules_mod.HANDED_OFF


def test_the_end_protocol_satisfies_a_goal_on_a_write_that_took_effect():
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    made = guards_mod.writes_made([{"role": "user", "content": "Hi."}, DELIVERY_MOVED],
                                  ["move_delivery"])
    assert protocol.goal_done(made) is True
    assert protocol.kind("Anything else?", said_anything=True, had_nothing=False,
                         made=made) == rules_mod.GOAL_SATISFIED


def test_the_end_protocol_does_not_satisfy_a_goal_on_a_write_whose_effect_record_moved_nothing():
    """D215's record rides with the call, so a write that answered cleanly and left every column
    where it was is not a write the agent user may end its Run on."""
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    unmoved = {**DELIVERY_MOVED,
               "write_effect": [{"table": "deliveries", "row": "D77", "path": "slot",
                                 "before": "morning", "after": "morning"}]}
    made = guards_mod.writes_made([{"role": "user", "content": "Hi."}, unmoved], ["move_delivery"])
    assert protocol.goal_done(made) is False


def test_the_end_protocol_satisfies_a_goal_on_a_write_whose_effect_record_moved_a_column():
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    moved = {**DELIVERY_MOVED,
             "write_effect": [{"table": "deliveries", "row": "D77", "path": "slot",
                               "before": "morning", "after": "evening"}]}
    made = guards_mod.writes_made([{"role": "user", "content": "Hi."}, moved], ["move_delivery"])
    assert protocol.goal_done(made) is True


def _run_file(path, events):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


def test_a_run_the_user_ended_on_a_refused_write_is_counted_for_the_round(tmp_path):
    refused = [{"type": "tool_result", "payload": {"name": "move_delivery",
                                                   "error": {"class": "business_error"}}},
               {"type": "user_turn", "payload": {"user_end": rules_mod.GOAL_SATISFIED}}]
    moved = [{"type": "tool_result", "payload": {"name": "move_delivery"}},
             {"type": "user_turn", "payload": {"user_end": rules_mod.GOAL_SATISFIED}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", refused)
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-1.jsonl", moved)
    counts = fidelity_mod.refused_write_ends(tmp_path, ["move_delivery"])
    assert counts[fidelity_mod.REFUSED_WRITE_ENDS] == 1
    assert counts["runs_read"] == 2 and counts["runs_with_a_write"] == 2


def test_a_run_handed_off_over_a_refused_write_is_not_counted_as_a_goal_the_user_closed(tmp_path):
    """The count is of the one end kind the old rule reached over a refusal. A Run the Candidate
    closed ended the way it would have ended anyway, so counting it would overstate the change."""
    handed = [{"type": "tool_result", "payload": {"name": "move_delivery",
                                                  "error": {"class": "business_error"}}},
              {"type": "user_turn", "payload": {"user_end": rules_mod.HANDED_OFF}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", handed)
    counts = fidelity_mod.refused_write_ends(tmp_path, ["move_delivery"])
    assert counts[fidelity_mod.REFUSED_WRITE_ENDS] == 0
    assert counts["runs_with_a_write"] == 1 and counts["runs_with_no_end_kind"] == 0


def test_a_run_whose_end_kind_is_a_tag_is_read_the_same_as_one_that_names_it(tmp_path):
    """The rule-driven user tags the turn it ends on and the Runner copies the tags into the file;
    the agent user writes the kind under its own name. One reader, so neither shape is missed."""
    tagged = [{"type": "tool_result", "payload": {"name": "move_delivery",
                                                  "error": {"class": "business_error"}}},
              {"type": "user_turn", "payload": {"tags": [rules_mod.GOAL_SATISFIED]}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", tagged)
    counts = fidelity_mod.refused_write_ends(tmp_path, ["move_delivery"])
    assert counts[fidelity_mod.REFUSED_WRITE_ENDS] == 1 and counts["runs_with_no_end_kind"] == 0


def test_a_run_written_before_the_end_kinds_existed_is_counted_as_unclassified(tmp_path):
    old = [{"type": "tool_result", "payload": {"name": "move_delivery",
                                               "error": {"class": "business_error"}}},
           {"type": "stop", "payload": {"reason": "user_stop"}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", old)
    counts = fidelity_mod.refused_write_ends(tmp_path, ["move_delivery"])
    assert counts[fidelity_mod.REFUSED_WRITE_ENDS] == 0
    assert counts["runs_with_no_end_kind"] == 1


def test_the_end_kinds_are_counted_under_the_user_whose_turn_ended_the_run(tmp_path):
    """D231: a round could only say how many Runs ended each way over both users at once, so the
    one number D214 exists to move, how often a user runs out of scenario, could not be read per
    driver at all. The turn that ends a Run names its own driver, and that is what the split reads."""
    agent_ended = [{"type": "user_turn", "payload": {"driver": "agent", "text": "Any news?"}},
                   {"type": "user_turn", "payload": {"driver": "agent",
                                                     "user_end": rules_mod.GOAL_SATISFIED}}]
    rules_ended = [{"type": "user_turn", "payload": {"driver": "agent", "text": "Any news?"}},
                   {"type": "user_turn", "payload": {"driver": "rules",
                                                     "user_end": rules_mod.SCENARIO_EXHAUSTED}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", agent_ended)
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-1.jsonl", rules_ended)
    split = fidelity_mod.ends_by_driver(tmp_path)
    assert split["agent"][rules_mod.GOAL_SATISFIED] == 1
    assert split["rules"][rules_mod.SCENARIO_EXHAUSTED] == 1
    assert split["agent"][rules_mod.SCENARIO_EXHAUSTED] == 0, "every kind is named for a driver that spoke"


def test_a_run_the_rule_driven_user_ended_alone_needs_no_driver_written_on_its_turn(tmp_path):
    """The rule-driven user writes no driver on its own turns, so an unnamed one is its own; a Run
    that ended in no kind at all is in neither driver's count rather than guessed at."""
    tagged = [{"type": "user_turn", "payload": {"tags": [rules_mod.HANDED_OFF]}}]
    unclassified = [{"type": "user_turn", "payload": {"text": "Thanks."}}]
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-0.jsonl", tagged)
    _run_file(tmp_path / "runs" / "task_1" / "reroll-task_1-1.jsonl", unclassified)
    split = fidelity_mod.ends_by_driver(tmp_path)
    assert list(split) == ["rules"] and split["rules"][rules_mod.HANDED_OFF] == 1
    assert sum(split["rules"].values()) == 1
    assert fidelity_mod.ends_by_driver(tmp_path / "nowhere") == {}


# --- D251: goal_satisfied waits for the required write -----------------------------------------


def ask(text: str) -> list[dict]:
    return [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": text}]


def kiln_user(**kwargs) -> rules_mod.SimulatedUser:
    """An invented pottery studio: the customer wants a kiln firing booked."""
    facts = [UserFact(field=rules_mod.GOAL, value="Please book a kiln firing.", span=PTR)]
    return rules_mod.SimulatedUser(UserRules(facts=facts), vocab=VOCAB, **kwargs)


def test_a_write_task_with_no_write_in_the_transcript_is_not_goal_satisfied():
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    assert protocol.goal_done(set()) is False
    assert protocol.kind("What is your plot number?", said_anything=True, had_nothing=False,
                         made=set()) is None
    user = kiln_user(write_tools=["book_kiln"], goal_writes=["book_kiln"])
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("I have noted that down."))
    assert user.done is False and user.end_reason is None


def test_an_empty_goal_writes_on_a_write_task_does_not_satisfy_until_the_write_took_effect():
    """The vacuous fallback D210 pinned: empty goal_writes is a subset of empty made."""
    protocol = guards_mod.EndProtocol(goal_writes=[], write_tools=["move_delivery"])
    assert protocol.goal_done(set()) is False
    assert protocol.goal_done({"move_delivery"}) is True
    assert rules_mod.goal_writes_done(set(), None, ["move_delivery"]) is False
    assert rules_mod.goal_writes_done({"move_delivery"}, None, ["move_delivery"]) is True


def test_a_write_task_out_of_answers_is_not_scenario_exhausted_while_the_write_is_unmade():
    """Sibling of test_a_candidate_that_twice_asks_what_nobody_told_this_user_runs_the_scenario_out.
    That test encoded exhaustion on a write Task; D251 keeps it for communicate-only Tasks."""
    protocol = guards_mod.EndProtocol(goal_writes=["move_delivery"], write_tools=["move_delivery"])
    assert protocol.kind("What is your tier?", said_anything=True, had_nothing=True) is None
    assert protocol.kind("And your tier?", said_anything=True, had_nothing=True) is None
    user = kiln_user(write_tools=["book_kiln"], goal_writes=["book_kiln"])
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("What is your membership tier?"))
    user.reply(ask("Could you give me your account code?"))
    assert user.done is False
    assert user.end_reason is None


def test_a_communicate_only_task_with_an_empty_write_set_may_still_exhaust():
    protocol = guards_mod.EndProtocol()
    assert protocol.kind("What is your tier?", said_anything=True, had_nothing=True) is None
    assert protocol.kind("And your tier?", said_anything=True,
                         had_nothing=True) == rules_mod.SCENARIO_EXHAUSTED
    user = kiln_user()
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("What is your membership tier?"))
    user.reply(ask("Could you give me your account code?"))
    assert user.done is True
    assert user.end_reason == rules_mod.SCENARIO_EXHAUSTED


def test_both_drivers_read_the_implied_writes_through_one_function():
    assert rules_mod.implied_writes([], ["book_kiln"]) == frozenset()
    assert rules_mod.implied_writes(["book_kiln"], ["book_kiln", "pay_glaze"]) == frozenset({"book_kiln"})
    assert rules_mod.implied_writes([], []) == frozenset()
    protocol = guards_mod.EndProtocol(goal_writes=[], write_tools=["book_kiln"])
    assert protocol.goal_done(set()) is False
    assert rules_mod.goal_writes_done(set(), [], ["book_kiln"]) is False
    assert rules_mod.goal_writes_done({"book_kiln"}, [], ["book_kiln"]) is True
    # write_tools is the environment's capability set, not each required write.
    assert rules_mod.goal_writes_done({"book_kiln"}, [], ["book_kiln", "pay_glaze"]) is True
    assert rules_mod.goal_writes_done({"book_kiln"}, ["book_kiln", "pay_glaze"],
                                      ["book_kiln", "pay_glaze"]) is False
    user = kiln_user(write_tools=["book_kiln"], goal_writes=[])
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("I have noted that down."))
    assert user.end_reason != rules_mod.GOAL_SATISFIED
    assert user.done is False
