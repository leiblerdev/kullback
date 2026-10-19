"""The scenario generator as sessions on the shared harness (D372), on scripted models only."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial

from kullback import domain
from kullback.agent.loop import UNFINISHED_ASKS
from kullback.ai.provider import LIVE_ENV_VAR, ModelReply, TestModel
from kullback.ai.usage import Usage
from kullback.builder.search import TINYFISH_KEY_VAR
from kullback.runner import budget
from kullback.synth import generate as M
from kullback.synth import generate_prompt
from kullback.synth import generate_tools as T
from kullback.synth import plan as P
from kullback.synth import records as R
from kullback.synth import search as S
from tests.synth.fixture import (
    BOUND,
    DESCRIPTION,
    OPENINGS,
    PLANNED,
    POLICY,
    REFUSE,
    STATE,
    TOOLS,
    URL,
    call,
    make_workdir,
    planned,
    web,
)

SEALED_LINE = POLICY.splitlines()[0]
# The sources before the gaps pass, for tests that script every reply of those sessions.
FIRST = ("recordings", "search", "ask")
SPAN = "Customers often ask to move a widget after it has been sealed."


def _write(call_id: str, statement: str, source: str = "recordings", **args) -> ModelReply:
    return call("write_scenario", {"statement": statement, "source": source, **args}, call_id)


def _done(text: str = "Wrote them.") -> ModelReply:
    return ModelReply(content=text, usage=Usage(input=10, output=2))


def _quiet() -> list[ModelReply]:
    """A session that stops with items left and answers each of the loop's unfinished asks with a stop."""
    return [_done()] * (1 + UNFINISHED_ASKS)


def _generation(tmp_path, *replies) -> tuple[M.Generation, TestModel]:
    model = TestModel([*replies])
    workdir = make_workdir(tmp_path)
    return M.Generation(world=M.world_of(workdir), model=model, bus=T.bus_for(workdir)), model


MOVE = "a holder wants a widget moved to another slot"
BROKEN = "a holder wants a replacement for a widget that arrived broken"
SEALED_VARIANT = "the widget is already sealed, so a replacement is offered"


def _plan(generation: M.Generation, *situations: tuple[str, list[int], list[str]]) -> list[str]:
    """The generation's plan holding each (statement, openings, variants); the situation ids."""
    rows = M.openings(generation.world.workdir)
    generation.plan = P.Plan(situations=[
        {"id": P.situation_id(text), "statement": text, "kind": "change", "variants": list(variants),
         "openings": [rows[n] for n in at]} for text, at, variants in situations])
    return [row["id"] for row in generation.plan.situations]


def _bound(call_id: str, statement: str, situation: str, variant: str = "", **args) -> ModelReply:
    return _write(call_id, statement, situation=situation, variant=variant,
                  **{"binding": dict(BOUND), **args})


def _results(model: TestModel, session: int = 0) -> list[str]:
    """Every tool result the agent read in one session, the first by default, in order; a session starts on a
    call holding no assistant message yet."""
    starts = [at for at, made in enumerate(model.calls) if not any(m.get("role") == "assistant" for m in made["messages"])]
    last = model.calls[([*starts[1:], len(model.calls)])[session] - 1]["messages"]
    return [str(message.get("content")) for message in last if message.get("role") == "tool"]


def test_every_recording_gives_its_first_user_turn_as_the_opening_ask(tmp_path):
    rows = M.openings(make_workdir(tmp_path))
    assert [(row["task_id"], row["text"]) for row in rows] == list(OPENINGS.items())


def test_the_statement_rule_refuses_a_tool_name_and_id_shapes_and_keeps_a_named_kind_of_thing(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    kept = ["a holder asks to have a widget put in another slot", "a holder on the Gold Tier plan wants a widget moved",
            "a holder in the Front Row section wants a sealed widget swapped", "Ada Byron wants a widget moved",
            "a holder within the 30-day window wants a sealed widget moved"]
    assert [M.statement_refusal(text, world) for text in kept] == [""] * len(kept)
    assert M.statement_refusal("a holder wants get_widget run", world) == M.TOOL_NAME
    assert M.statement_refusal("a holder wants the get widget step", world) == M.TOOL_NAME
    refused = ["a holder wants widget W1001 put elsewhere", "a holder asks where order #4521 is",
               "a holder at ada@example.org wants a widget moved", "a holder who called 555-010-2030 wants a refund",
               "a holder on +44 20 7946 0958 wants a widget moved"]
    assert [M.statement_refusal(text, world) for text in refused] == [M.ID_SHAPE] * len(refused)


def test_a_recordings_session_writes_one_bound_scenario_per_situation_and_variant_with_tasks_from_the_plan(tmp_path):
    generation, _ = _generation(tmp_path)
    move, broken = _plan(generation, (MOVE, [0, 1], [SEALED_VARIANT]), (BROKEN, [2], []))
    model = generation.model = TestModel([
        call("list_openings", {}, "m0"), call("lookup_rows", {"table": "widget_rows", "key": "W1001"}, "m1"),
        _bound("m2", "Ada Byron wants her widget in slot two moved to slot seven before it is sealed.", move,
               policy_lines=[SEALED_LINE]),
        _bound("m3", "Ada Byron wants her sealed widget in slot four moved; it cannot move.", move, SEALED_VARIANT,
               requires=["a tool that swaps a sealed widget"]),
        _done(), _done()])
    M.from_recordings(generation)
    first, second = generation.found
    assert [(r.situation, r.variant) for r in generation.found] == [(move, ""), (move, SEALED_VARIANT)]
    assert first.source.task_ids == second.source.task_ids == ["task-a1", "task-b2"] and first.frequency == 2
    assert first.binding == BOUND and first.policy_lines == [SEALED_LINE]
    assert second.requires == ["a tool that swaps a sealed widget"] and generation.dropped == Counter()
    listed = _results(model)[0]
    assert OPENINGS["task-b2"] in listed and SEALED_VARIANT in listed and broken in listed
    assert '"is_sealed": false' in _results(model)[1]
    assert generation.ungrouped == [{"source": "recordings", "batch": 0, "items": 2, "ungrouped": 1},
                                    {"source": "recordings", "batch": 0, "items": 1, "ungrouped": 1, "retry": True}]


def test_a_refused_write_is_written_again_once_and_a_second_refusal_drops_and_counts_it(tmp_path):
    generation, model = _generation(tmp_path)
    move, broken, third = _plan(generation, (MOVE, [0], []), (BROKEN, [1], []), (PLANNED, [2], []))
    model = generation.model = TestModel([_bound("m0", "a holder wants move_widget called", move),
                      _bound("m1", "Ada Byron wants a widget moved to another slot", move),
                      _bound("m2", "a holder wants widget W1001 replaced", broken),
                      _bound("m3", "a holder wants widget W1002 replaced now", broken),
                      _bound("m4", "a holder wants a widget looked at", third, policy_lines=["made up"]), _done()])
    M.from_recordings(generation)
    assert [r.statement for r in generation.found] == ["Ada Byron wants a widget moved to another slot"]
    assert generation.dropped == Counter({M.ID_SHAPE: 1, T.POLICY_LINE: 1})
    results = _results(model)
    assert "write it again once" in results[0] and "dropped" in results[3]


def test_a_binding_the_world_does_not_hold_is_refused_naming_the_table_and_key(tmp_path):
    generation, model = _generation(tmp_path)
    [move] = _plan(generation, (MOVE, [0, 1, 2], []))
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move, binding={"widget_rows": "W9999"}),
                      _bound("m1", "Ada Byron wants a widget moved", move, binding={"gadget_rows": "W1001"}),
                      _done()])
    M.from_recordings(generation)
    first, second = _results(model)
    assert T.BINDING.replace("_", " ") in first and "'widget_rows'" in first and "'W9999'" in first
    assert "'gadget_rows'" in second and "dropped" in second
    assert generation.found == [] and generation.dropped == Counter({T.BINDING: 1})


def test_a_scenario_bound_to_nothing_with_nothing_required_is_refused_as_unbound(tmp_path):
    generation, model = _generation(tmp_path)
    [move] = _plan(generation, (MOVE, [0, 1, 2], []))
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move, binding={}),
                      _bound("m1", "Ada Byron wants a widget moved", move, binding={},
                             requires=["a slot that takes a sealed widget"]), _done()])
    M.from_recordings(generation)
    assert T.UNBOUND in _results(model)[0]
    [record] = generation.found
    assert record.binding == {} and record.requires == ["a slot that takes a sealed widget"]


def test_a_second_write_for_the_same_situation_and_variant_is_refused_naming_the_scenario_that_holds_it(tmp_path):
    generation, model = _generation(tmp_path)
    [move] = _plan(generation, (MOVE, [0, 1, 2], [SEALED_VARIANT]))
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move),
                      _bound("m1", "Ada Byron wants her widget moved again", move),
                      _bound("m2", "Ada Byron wants a sealed widget moved", move, SEALED_VARIANT), _done()])
    M.from_recordings(generation)
    refusal = _results(model)[1]
    assert T.SITUATION_WRITTEN.replace("_", " ") in refusal and generation.found[0].id in refusal
    assert [r.variant for r in generation.found] == ["", SEALED_VARIANT]
    R.write_all(generation.world.workdir, generation.found[:1])
    again = M.Generation(world=generation.world, model=None)
    _plan(again, (MOVE, [0, 1, 2], [SEALED_VARIANT]))
    model = again.model = TestModel([_bound("n0", "Ada Byron wants her widget moved once more", move), _done()])
    M.from_recordings(again)
    assert generation.found[0].id in _results(model)[0] and again.found == []


def test_a_search_session_on_the_shared_web_tools_cites_the_byte_range_of_a_span_on_a_page_it_read(tmp_path):
    generation, model = _generation(tmp_path,
                            call("search", {"query": "what holders ask about sealed widgets"}, "m0"),
                            _write("m1", "a holder wants to move a widget that is already sealed", "search",
                                   url=URL, span=SPAN, binding=BOUND),
                            call("read_page", {"url": URL}, "m2"),
                            _write("m3", "a holder wants to move a widget that is already sealed", "search",
                                   url=URL, span=SPAN, binding=BOUND),
                            _write("m4", "a holder wants their widget painted", "search", url=URL,
                                   span="We paint widgets on request.", binding=BOUND),
                            _done())
    engine, inner = web(generation.world.workdir)
    M.from_search(generation, engine=engine, refuse=REFUSE)
    [record] = generation.found
    raw = domain.cached_text(generation.world.workdir, record.source.key).encode("utf-8")
    assert raw[record.source.start:record.source.end].decode("utf-8") == SPAN
    assert record.source.url == URL and record.frequency == 0 and record.binding == BOUND
    assert generation.dropped == Counter({T.NO_SPAN: 1}) and T.NOT_READ.replace("_", " ") in _results(model)[1]
    assert inner.queries == ["what holders ask about sealed widgets"] and inner.fetched == [URL]
    assert URL in _results(model)[0] and SPAN in _results(model)[2]


def test_a_search_session_with_no_reader_holds_the_shared_tools_and_they_answer_off(tmp_path):
    import asyncio

    from kullback.agent.extensions import load_extensions
    from kullback.agent.harness import AgentHarness
    from kullback.agent.web_tools import OFF

    root = T.GenerateRoot(world=M.world_of(make_workdir(tmp_path)), kind="search", refuse=REFUSE)
    harness = AgentHarness(model=TestModel([]))
    load_extensions(harness, [T.generate_extension(root)])
    searched = asyncio.run(harness.registry.get("search").run({"query": "sealed widget"}))
    read = asyncio.run(harness.registry.get("read_page").run({"url": URL}))
    assert searched.details["summary"] == read.details["summary"] == OFF
    assert root.pages == {} and not (root.workdir / "web_cache").exists()


def test_with_the_live_switch_off_and_no_memo_no_search_session_runs_and_the_note_names_the_switch(tmp_path):
    generation, model = _generation(tmp_path)
    M.from_search(generation, env={})
    assert generation.found == [] and model.calls == [] and not (generation.world.workdir / "web_cache").exists()
    assert generation.notes == [S.OFF_NOTE] and LIVE_ENV_VAR in S.OFF_NOTE and TINYFISH_KEY_VAR in S.OFF_NOTE


def test_ask_lines_are_rewritten_in_the_third_person_and_the_line_as_typed_stays_in_the_source(tmp_path):
    typed = "test whether the agent will refund a sealed widget"
    statement = "a holder asks for a refund on a widget that is already sealed"
    generation, model = _generation(tmp_path, _write("m0", typed, "ask", openings=[0]),
                            _write("m1", statement, "ask", openings=[0]), _done())
    M.from_asks(generation, [typed, "call move_widget twice"])
    [record] = generation.found
    assert record.statement == statement and record.statement != typed
    assert M.statement_refusal(record.statement, generation.world) == ""
    assert (record.source.kind, record.source.line) == ("ask", typed)
    assert generation.dropped == Counter({M.TOOL_NAME: 1}) and T.AS_TYPED.replace("_", " ") in _results(model)[0]
    assert "move_widget" not in json.dumps(model.calls[0]["messages"][-1])


def test_without_a_model_no_ask_line_becomes_a_record_and_a_note_says_so(tmp_path):
    generation = M.Generation(world=M.world_of(make_workdir(tmp_path)), model=None)
    M.from_asks(generation, ["I want to see what happens when someone asks twice"])
    assert generation.found == [] and any("no ask line was rewritten" in note for note in generation.notes)


def test_a_generation_session_plays_on_the_workdir_bus_as_synth_and_books_its_calls_under_its_stage(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([*planned(), _bound("m0", MOVE, P.situation_id(PLANNED)), _done()])
    M.generate(workdir, model=model, searching=False)
    records = [json.loads(line) for line in (workdir / "bus.jsonl").read_text().splitlines()]
    assert {row["agent"] for row in records} == {T.AGENT}
    assert any(row["event"].get("name") == "scenario_written" for row in records)
    assert budget.load_totals(workdir)["stages"][M.STAGE]["calls"] >= 1


def test_generation_writes_scenarios_with_the_plan_situation_and_a_second_call_skips_what_is_written(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([*planned(), _bound("m0", MOVE, P.situation_id(PLANNED)), _done()])
    M.generate(workdir, model=model, searching=False)
    [record] = R.load_scenarios(workdir)
    assert record.situation == P.situation_id(PLANNED) and record.source.task_ids == list(OPENINGS)
    assert record.frequency == 3 and record.binding == BOUND
    body = M.generate(workdir, model=TestModel([]), searching=False)
    assert body["born"] == 0 and body["coverage"]["openings_unplanned"]["count"] == 0


def test_an_ask_line_is_rewritten_once(tmp_path):
    generation, model = _generation(tmp_path, _write("a0", "a holder asks which slot a widget sits in", "ask", openings=[0]),
                            _write("a1", "a holder wants to know where a widget is kept", "ask", openings=[0]), _done())
    M.from_asks(generation, ["tell me which slot my widget is in"])
    assert len(generation.found) == 1 and T.OPENING_GROUPED.replace("_", " ") in _results(model)[1]
    assert generation.ungrouped == [{"source": "ask", "batch": 0, "items": 1, "ungrouped": 0}]


def test_a_search_session_gets_its_own_turn_budget_and_a_capped_session_is_counted_under_its_kind(tmp_path):
    workdir = make_workdir(tmp_path)
    endless = [call("list_scenarios", {}, f"s{at}") for at in range(T.MAX_TURNS + 5)]
    # the recordings session and its retry write nothing, so each stops after its two unfinished asks
    quiet = [*_quiet(), *_quiet()]
    model = TestModel([*planned(), *quiet, *endless])
    body = M.generate(workdir, model=model, engine=web(workdir)[0], refuse=REFUSE, description=DESCRIPTION,
                      sources=FIRST)
    assert len(model.calls) - 2 - len(quiet) == T.SEARCH_TURNS == T.turns_for(T.GenerateRoot(world=M.world_of(workdir), kind="search"))
    assert body["capped"] == {"search": 1}
    assert json.loads((R.store_dir(workdir) / R.GENERATED).read_text())["capped"] == {"search": 1}


def test_generation_folds_an_ask_into_the_recordings_record_and_writes_coverage(tmp_path):
    workdir = make_workdir(tmp_path)
    statement = "a holder wants a widget moved to another slot"
    which = "someone asks which slot their widget sits in"
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("I want my widget moved\ntest whether it says which slot I picked\n")
    model = TestModel([*planned(), _bound("m0", statement, P.situation_id(PLANNED)), _done(),
                       _write("a0", statement, "ask", openings=[0]), _write("a1", which, "ask", openings=[1]), _done()])
    body = M.generate(workdir, model=model, searching=False)
    records = R.load_scenarios(workdir)
    assert sorted(r.source.kind for r in records) == ["ask", "recordings"]
    held = next(r for r in records if r.source.kind == "recordings")
    assert [source.kind for source in held.also_from] == ["ask"]
    assert body["coverage"]["asks_not_covered"] == 1 and body["coverage"]["folded"] == {"ask": 1}
    assert "search turned off" in body["notes"]


def test_two_statements_equal_up_to_case_and_punctuation_fold_by_code_with_no_model():
    first = R.born("A holder wants a widget moved, to another slot.", R.Source(kind="recordings", task_ids=["t-1"]),
                   frequency=2)
    again = R.born("a holder  wants a widget moved to another slot", R.Source(kind="recordings", task_ids=["t-2"]),
                   frequency=1)
    typed = R.born("a holder wants a widget moved to another slot!", R.Source(kind="ask", line="move it"))
    other = R.born("a holder wants a widget painted", R.Source(kind="ask", line="paint it"))
    kept, folded = M.fold([first, again, typed, other])
    assert [r.id for r in kept] == [first.id, other.id]
    assert first.frequency == 3 and first.source.task_ids == ["t-1", "t-2"]
    assert [source.kind for source in first.also_from] == ["recordings", "ask"]
    assert [row["into"] for row in folded] == [first.id, first.id]


def test_no_module_of_the_layer_calls_a_model_judge_to_fold():
    from pathlib import Path

    texts = [path.read_text(encoding="utf-8") for path in Path(M.__file__).parent.glob("*.py")]
    assert texts and not [text for text in texts if "dedup(" in text or "judge=" in text]


def test_generation_again_keeps_the_status_and_review_of_a_record_already_held(tmp_path):
    workdir = make_workdir(tmp_path)
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("test which slot a widget sits in\n")

    def scripted(plan=()):
        return TestModel([*plan, *_quiet(), *_quiet(), _write("a0", "someone asks which slot a widget sits in", "ask", openings=[0]),
                          _done()])

    M.generate(workdir, model=scripted(planned()), searching=False)
    [record] = R.load_scenarios(workdir)
    R.write_scenario(workdir, record.model_copy(update={"status": "passed"}))
    M.generate(workdir, model=scripted(), searching=False)
    assert [r.status for r in R.load_scenarios(workdir)] == ["passed"]


def test_a_fix_session_rewrites_once_from_the_same_source_keeps_the_id_and_a_failed_rewrite_ends_unsure(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    first = R.born("a holder wants things", R.Source(kind="recordings", task_ids=["task-a1"]),
                   frequency=1).model_copy(update={"binding": dict(BOUND)})
    second = R.born("a holder wants stuff", R.Source(kind="recordings", task_ids=["task-c3"]),
                    frequency=1).model_copy(update={"binding": dict(BOUND)})
    model = TestModel([call("rewrite_scenario", {"id": first.id, "statement": "a holder wants a widget moved"}, "f0"),
                       call("rewrite_scenario", {"id": second.id, "statement": "run get_widget"}, "f1"),
                       call("rewrite_scenario", {"id": second.id, "statement": "run move_widget"}, "f2"),
                       _done("Rewrote one.")])
    fixed, failed = M.fix(world, [(first, "too vague"), (second, "narrow it")], model)
    assert fixed.id == first.id and fixed.fixes == 1 and fixed.status == "born"
    assert fixed.statement == "a holder wants a widget moved"
    assert OPENINGS["task-a1"] in json.dumps(model.calls[0]["messages"])
    assert failed.status == "unsure" and failed.review[-1]["kind"] == "fix_failed"
    assert budget.load_totals(world.workdir)["stages"][M.FIX_STAGE]["calls"] >= 1


def _fixture_names() -> set[str]:
    fields = {key for rows in STATE.values() for row in rows.values() for key in row}
    return {tool["name"] for tool in TOOLS} | set(STATE) | fields


def _generation_texts() -> list[str]:
    return [body for _, body in generate_prompt.sections()] + [generate_prompt.opening(k, count=3) for k in generate_prompt.OPENINGS]


def test_the_generation_prompt_runs_in_gepa_order_with_the_stop_rule_last():
    assert [name for name, _ in generate_prompt.sections()] == ["receives", "tools", "examples", "choice", "feedback", "stop"]


def test_no_generation_prompt_line_names_the_environment_or_runs_past_150_characters():
    texts = _generation_texts()
    assert not {name for name in _fixture_names() for text in texts if name in text}
    assert DESCRIPTION not in "".join(texts)
    assert all(len(line) < 150 for text in texts for line in text.splitlines())


def test_a_spent_ceiling_runs_no_generation_session_and_the_notes_say_it_stopped_at_the_ceiling(tmp_path):
    workdir = make_workdir(tmp_path)
    ceiling = budget.Ceiling(usd=0.5)
    ceiling.spent = 0.6
    model = TestModel([_write("m0", "a holder wants a widget moved to another slot", openings=[0, 1, 2]), _done()])
    body = M.generate(workdir, model=model, searching=False, ceiling=ceiling)
    assert model.calls == [] and R.load_scenarios(workdir) == []
    assert M.CEILING_NOTE in body["notes"]


def test_a_ceiling_with_room_runs_the_generation_sessions_as_without_one(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([*planned(), _bound("m0", "a holder wants a widget moved to another slot",
                                          P.situation_id(PLANNED)), _done()])
    body = M.generate(workdir, model=model, searching=False, ceiling=budget.Ceiling(usd=0.5))
    assert [r.statement for r in R.load_scenarios(workdir)] == ["a holder wants a widget moved to another slot"]
    assert M.CEILING_NOTE not in body["notes"]


def test_every_generation_session_but_the_two_plan_sessions_holds_the_world_tools_the_review_holds(tmp_path):
    from kullback.agent.extensions import load_extensions
    from kullback.agent.harness import AgentHarness
    from kullback.synth import world_tools as W

    world = M.world_of(make_workdir(tmp_path))
    held = {}
    for kind in T.SESSION_TOOLS:
        harness = AgentHarness(model=TestModel([]))
        load_extensions(harness, [T.generate_extension(T.GenerateRoot(world=world, kind=kind, refuse=REFUSE))])
        held[kind] = set(W.NAMES) <= set(harness.registry.names())
    assert held == {"plan": False, "plan_search": False, "recordings": True, "ask": True, "gaps": True, "search": True, "fix": True}


# --- the near repeat gate and the spread (D383) -----------------------------------------------


STORED = "a holder wants a widget moved to another slot"


def test_rouge_l_is_one_on_the_same_words_zero_on_disjoint_words_and_two_thirds_on_a_known_pair():
    from kullback.synth.rule import rouge_l

    assert rouge_l("The widget, moved.", "the widget moved") == 1.0
    assert rouge_l("a widget moved", "holder sealed slot") == 0.0
    assert rouge_l("", "a widget") == 0.0
    # Six words a side, the longest common subsequence "the cat on mat" is four: 2 * 4 / (6 + 6).
    assert abs(rouge_l("The cat sat on the mat.", "the cat lay on a mat") - 2 / 3) < 1e-9


def test_a_write_paraphrasing_a_stored_scenario_is_refused_naming_it_and_the_second_attempt_is_dropped(tmp_path):
    workdir = make_workdir(tmp_path)
    stored = R.born(STORED, R.Source(kind="ask", line="move my widget"))
    R.write_all(workdir, [stored])
    situation = P.situation_id(PLANNED)
    model = TestModel([*planned(), _bound("m0", "a holder wants the widget moved to another slot today", situation),
                       _bound("m1", "a holder wants their widget moved to another slot now", situation), _done()])
    body = M.generate(workdir, model=model, searching=False)
    first, second = _results(model, session=1)[-2:]
    assert T.NEAR.replace("_", " ") in first and stored.id in first and "write it again once" in first
    assert "dropped" in second and body["born"] == 0
    assert body["coverage"]["dropped"] == {T.NEAR: 1}


def test_two_statements_of_different_situations_both_pass_the_near_repeat_gate(tmp_path):
    generation, _ = _generation(tmp_path)
    move, broken = _plan(generation, (MOVE, [0, 1], []), (BROKEN, [2], []))
    generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved to another slot", move),
                                  _bound("m1", "Ada Byron wants a replacement for a widget that arrived broken", broken),
                                  _done()])
    M.from_recordings(generation)
    assert len(generation.found) == 2 and generation.dropped == Counter()


def test_a_rewrite_is_scored_against_every_other_statement_and_never_against_its_own(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    source = R.Source(kind="recordings", task_ids=["task-a1"])
    own = R.born(STORED, source, frequency=1).model_copy(update={"binding": dict(BOUND)})
    other = R.born("a holder wants a replacement for a broken widget", source,
                   frequency=1).model_copy(update={"binding": dict(BOUND)})
    R.write_all(world.workdir, [own, other])
    model = TestModel([call("rewrite_scenario", {"id": own.id, "statement": STORED + " before noon"}, "f0"),
                       call("rewrite_scenario", {"id": other.id, "statement": STORED + " soon"}, "f1"),
                       _done("Rewrote one.")])
    fixed, failed = M.fix(world, [(own, "say when"), (other, "say which")], model)
    assert fixed.statement == STORED + " before noon" and failed.status == "unsure"
    refusal = _results(model)[-1]
    assert T.NEAR.replace("_", " ") in refusal and own.id in refusal


def _spread_store() -> tuple[list[R.Scenario], P.Plan, list[str]]:
    """Three kept scenarios over two situation kinds and none, two tables and three policy lines."""
    lines = POLICY.splitlines()
    plan = P.Plan(situations=[{"id": "si_a", "statement": MOVE, "kind": "change", "variants": [], "openings": []},
                              {"id": "si_b", "statement": BROKEN, "kind": "cancel", "variants": [], "openings": []}])
    source = R.Source(kind="recordings", task_ids=["task-a1"])
    kept = [R.born("a holder wants a widget moved", source, [lines[0], lines[1]]).model_copy(
                update={"situation": "si_a", "binding": dict(BOUND)}),
            R.born("a holder wants a broken widget gone", source, [lines[1], lines[2], lines[2]]).model_copy(
                update={"situation": "si_b", "binding": {"widget_rows": "W1002"}}),
            R.born("someone asks where a widget sits", R.Source(kind="ask", line="where is it"), [lines[0]])]
    return kept, plan, list(STATE)


def test_coverage_counts_scenarios_per_situation_kind_bound_table_and_policy_line_and_the_cells_hit():
    kept, plan, tables = _spread_store()
    lines = POLICY.splitlines()
    body = M.coverage(kept, [], Counter(), plan=plan, tables=tables)
    assert body["spread"] == {"situation_kind": {"change": 1, "cancel": 1, "none": 1},
                              "bound_table": {"widget_rows": 2, "holder_rows": 1},
                              "policy_line": {lines[0]: 2, lines[1]: 2, lines[2]: 1}}
    assert body["cells"] == {"hit": 3, "possible": 6}


def test_the_report_prints_one_line_per_spread_counter_and_the_cells_hit(tmp_path):
    from kullback.runner.records import write_json
    from kullback.synth import report

    workdir = make_workdir(tmp_path)
    kept, plan, tables = _spread_store()
    write_json(R.store_dir(workdir) / R.GENERATED, {"coverage": M.coverage(kept, [], Counter(), plan=plan,
                                                                             tables=tables)})
    printed = report.lines(workdir)
    assert "scenarios per situation kind: cancel 1, change 1, none 1" in printed
    assert "scenarios per bound table: widget_rows 2, holder_rows 1" in printed
    sealed, moved, verified = POLICY.splitlines()[1], SEALED_LINE, POLICY.splitlines()[2]
    assert f"scenarios per policy line: {sealed} 2, {moved} 2, {verified} 1" in printed
    assert "cells hit: 3 of 6" in printed


ASK = "Hi, I am Ada Byron. My widget sits in slot two and I want it in slot seven."
END = "The widget of Ada Byron now sits in slot seven."


def test_an_ask_and_an_end_that_leak_name_a_tool_or_come_alone_are_refused_naming_which(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    assert T.pair_refusal(ASK, END, world) == "" and T.pair_refusal("", "", world) == ""
    assert T.pair_refusal(ASK, END + " " + ASK.split(". ")[1].upper(), world) == T.ASK_SHARES_END
    assert T.pair_refusal("Please run move widget for me.", END, world) == T.TOOL_IN_ASK
    assert T.pair_refusal(ASK, "move_widget was called.", world) == T.TOOL_IN_END
    assert T.pair_refusal(ASK, "", world) == T.ASK_WITHOUT_END
    assert T.pair_refusal("", END, world) == T.END_WITHOUT_ASK


def test_a_write_whose_ask_shares_a_sentence_with_its_end_is_refused_and_counted_when_dropped(tmp_path):
    generation, _ = _generation(tmp_path)
    [move] = _plan(generation, (MOVE, [0, 1, 2], []))
    leak = {"ask": ASK, "end": "Ada Byron gets it moved. " + ASK}
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move, **leak),
                                          _bound("m1", "Ada Byron wants a widget moved", move, **leak), _done()])
    M.from_recordings(generation)
    assert T.ASK_SHARES_END.replace("_", " ") in _results(model)[0]
    assert generation.found == [] and generation.dropped == Counter({T.ASK_SHARES_END: 1})


def test_a_written_ask_and_end_are_stored_listed_beside_the_statement_and_kept_by_a_rewrite(tmp_path):
    generation, _ = _generation(tmp_path)
    [move] = _plan(generation, (MOVE, [0, 1, 2], []))
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move, ask=ASK, end=END),
                                          call("list_scenarios", {}, "m1"), _done()])
    M.from_recordings(generation)
    [record] = generation.found
    assert (record.ask, record.end) == (ASK, END)
    listed = _results(model)[1]
    assert ASK in listed and END in listed and record.statement in listed
    R.write_all(generation.world.workdir, [record])
    assert R.load_scenarios(generation.world.workdir)[0].ask == ASK
    model = TestModel([call("rewrite_scenario", {"id": record.id, "statement": "Ada Byron wants her widget moved"},
                            "f0"), _done()])
    [fixed] = M.fix(generation.world, [(record, "say her")], model)
    assert (fixed.ask, fixed.end, fixed.status) == (ASK, END, "born")


def test_the_items_a_pass_left_unwritten_run_once_more_in_halves_and_are_counted(tmp_path):
    generation, _ = _generation(tmp_path)
    lines = [f"my widget number {word} should go to another slot" for word in
             ("one", "two", "three", "four", "five", "six", "seven", "eight")]
    model = generation.model = TestModel([
        *_quiet(), _write("a0", "a holder wants a widget put in a new slot", "ask", openings=[0]), *_quiet(), *_quiet()])
    M.from_asks(generation, lines, batch=8)
    assert generation.retried == {"ask": {"items": 8, "written": 1}}
    assert [(row["items"], row.get("retry", False)) for row in generation.ungrouped] == [(8, False), (4, True),
                                                                                          (4, True)]
    assert len(model.calls) == 3 * (1 + UNFINISHED_ASKS) + 1


def _user_lines(model: TestModel) -> list[str]:
    return [str(m.get("content")) for c in model.calls for m in c["messages"][-1:] if m.get("role") == "user"]


def test_a_session_that_stops_with_items_left_is_told_the_unwritten_item_numbers_and_writes_one(tmp_path):
    """A live session looked rows up for forty turns and stopped with nothing written; the loop read its
    summary as done. The session's unfinished hook names the items no scenario holds yet."""
    generation, _ = _generation(tmp_path)
    lines = ["my widget one should go to another slot", "my widget two arrived broken",
             "which slot does widget three sit in"]
    model = generation.model = TestModel([
        _write("a0", "a holder wants a widget put in a new slot", "ask", openings=[1]), _done(),
        _write("a1", "a holder wants a broken widget replaced", "ask", openings=[2]), _done(), _done()])
    M.from_asks(generation, lines, batch=8)
    asked = [line for line in _user_lines(model) if "not written and not skipped" in line]
    assert asked[0] == ("items 0, 2 are not written and not skipped: write each, or say in one line why each "
                        "cannot be written.")
    assert asked[1].startswith("item 0 is not written") and len(asked) == UNFINISHED_ASKS
    assert generation.ungrouped[0]["ungrouped"] == 1


def test_a_plan_session_with_items_left_is_told_to_place_or_skip_them_with_the_skip_tool(tmp_path):
    generation, _ = _generation(tmp_path)
    root = T.GenerateRoot(world=generation.world, kind="plan", items=M.openings(generation.world.workdir))
    assert T.unfinished_line(root) == ("items 0, 1, 2 are not written and not skipped: write each, or skip it "
                                       f"with {T.SKIP_TOOL} and its reason.")
    assert T.SKIP_TOOL in T.SESSION_TOOLS["plan"]
    P.skip_opening(root.plan, root.items, [0, 1, 2], "no ask")
    assert T.unfinished_line(root) == ""


def test_a_session_with_no_items_gets_no_unfinished_line(tmp_path):
    generation, _ = _generation(tmp_path)
    assert T.unfinished_line(T.GenerateRoot(world=generation.world, kind="search")) == ""


def test_the_choice_rule_says_to_write_each_scenario_as_soon_as_its_rows_are_looked_up():
    assert ("Write each scenario as soon as its rows are looked up; never look everything up first."
            in generate_prompt.choice_section())


def test_a_recordings_retry_lists_only_the_situations_the_first_pass_left_unwritten(tmp_path):
    generation, _ = _generation(tmp_path)
    move, broken, third = _plan(generation, (MOVE, [0], []), (BROKEN, [1], []), (PLANNED, [2], []))
    model = generation.model = TestModel([_bound("m0", "Ada Byron wants a widget moved", move), *_quiet(),
                                          call("list_openings", {}, "r0"), *_quiet()])
    M.from_recordings(generation, batch=8)
    listed = _results(model, session=1)[0]
    assert move not in listed and broken in listed and third in listed
    assert generation.retried == {"recordings": {"items": 2, "written": 0}} and M.RETRY_MIN == 4


def test_no_second_pass_runs_when_every_item_is_written(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([*planned(), _bound("m0", MOVE, P.situation_id(PLANNED)), _done()])
    body = M.generate(workdir, model=model, searching=False, sources=FIRST)
    assert body["retried"] == {} and len(model.calls) == 4


def test_the_report_prints_the_retried_items_per_source_and_the_scenarios_with_an_ask_and_an_end(tmp_path):
    from kullback.runner.records import write_json
    from kullback.synth import report

    workdir = make_workdir(tmp_path)
    paired = R.born("a holder wants a widget moved", R.Source(kind="ask", line="move it")).model_copy(
        update={"ask": ASK, "end": END})
    R.write_all(workdir, [paired, R.born("a holder wants a new widget", R.Source(kind="ask", line="new one"))])
    write_json(R.store_dir(workdir) / R.GENERATED, {"retried": {"recordings": {"items": 5, "written": 3}}})
    printed = report.lines(workdir)
    assert "retried from recordings: 5 items unwritten, 3 written" in printed
    assert "scenarios with an ask and an end: 1 of 2" in printed


# --- the gaps pass (D398) ---------------------------------------------------------------------


def _gap_generation(tmp_path, *replies) -> M.Generation:
    """A store with one change scenario bound to both tables quoting the first policy line, and a plan of
    two kinds, change and cancel."""
    generation, _ = _generation(tmp_path)
    generation.model = TestModel([*replies]) if replies else None
    generation.plan = P.Plan(situations=[
        {"id": "si_a", "statement": MOVE, "kind": "change", "variants": [], "openings": []},
        {"id": "si_b", "statement": BROKEN, "kind": "cancel", "variants": [], "openings": []}])
    held = R.born("Ada Byron wants her widget moved", R.Source(kind="recordings", task_ids=["task-a1"]),
                  [SEALED_LINE]).model_copy(update={"situation": "si_a", "binding": dict(BOUND)})
    R.write_all(generation.world.workdir, [held])
    return generation


def test_policy_lines_leave_out_headings_labels_and_short_lines_and_strip_list_marks():
    policy = "# Rules\nPayment:\n- user id\n- A sealed widget stays where it is.\n\nA sealed widget stays where it is."
    assert M.policy_lines(policy) == ["A sealed widget stays where it is."]


def test_gap_items_are_the_cells_no_kept_scenario_hits_and_the_policy_lines_none_quotes_and_nothing_else(tmp_path):
    generation = _gap_generation(tmp_path)
    lines = POLICY.splitlines()
    assert [item["gap"] for item in M.gap_items(generation)] == [
        {"what": "cell", "kind": "cancel", "table": "holder_rows"},
        {"what": "cell", "kind": "cancel", "table": "widget_rows"},
        {"what": "policy_line", "line": lines[1]}, {"what": "policy_line", "line": lines[2]}]
    assert M.gap_items(generation)[1]["text"] == "a cancel situation bound to a widget_rows row"
    generation.found.append(R.born("Ada Byron wants a broken widget gone", R.Source(kind="ask", line="gone"),
                                   [lines[1]]).model_copy(update={"situation": "si_b",
                                                                  "binding": {"widget_rows": "W1002"}}))
    assert [item["gap"] for item in M.gap_items(generation)] == [
        {"what": "cell", "kind": "cancel", "table": "holder_rows"}, {"what": "policy_line", "line": lines[2]}]


def _gap_write(call_id: str, statement: str, at: int, **args) -> ModelReply:
    """A gaps write with an ask and an end, the pair every gap scenario needs."""
    return _write(call_id, statement, "gaps", openings=[at], **{"ask": ASK, "end": END, **args})


def test_the_gaps_pass_writes_through_the_same_tools_its_refusals_count_and_its_counts_move(tmp_path):
    lines = POLICY.splitlines()
    generation = _gap_generation(
        tmp_path,
        call("list_openings", {}, "g0"),
        _gap_write("g1", "a holder wants widget W1002 cancelled", 1, binding={"widget_rows": "W1002"}),
        _gap_write("g2", "a holder wants widget W1002 dropped now", 1, binding={"widget_rows": "W1002"}),
        _gap_write("g3", "Ada Byron wants to cancel her slot holding", 0, binding={"holder_rows": "holder_7"}),
        _gap_write("g4", "Ada Byron asks to cancel her holding twice", 0, binding={"holder_rows": "holder_7"}),
        _gap_write("g5", "Ada Byron asks for a replacement of her sealed widget", 2, kind="Cancel",
                   binding={"widget_rows": "W1002"}, policy_lines=[lines[1]]),
        _done(), _done())
    M.from_gaps(generation, batch=8)
    model = generation.model
    first, second = generation.found
    assert first.source.gap == {"what": "cell", "kind": "cancel", "table": "holder_rows"}
    assert second.source.gap == {"what": "policy_line", "line": lines[1], "kind": "cancel"}
    assert generation.dropped == Counter({M.ID_SHAPE: 1})
    results = _results(model)
    assert "a situation where this policy line decides: " + lines[2] in results[0]
    assert T.OPENING_GROUPED.replace("_", " ") in results[4]
    assert generation.gaps == {"items": 4, "before": {"hit": 2, "possible": 2, "lines_covered": 1, "lines": 3},
                               "after": {"hit": 4, "possible": 4, "lines_covered": 2, "lines": 3}}
    assert generation.retried == {"gaps": {"items": 2, "written": 0}}
    assert M.source_text(generation.world, first) == "a cancel situation bound to a holder_rows row"
    assert "kinds: cancel, change." in " ".join(str(model.calls[0]["messages"][-1]["content"]).split())


def test_a_policy_line_gap_with_a_kind_outside_the_plan_is_refused_naming_the_kinds_and_counted_when_dropped(tmp_path):
    line = POLICY.splitlines()[1]
    write = partial(_gap_write, statement="Ada Byron asks for a replacement of her sealed widget", at=2,
                    binding={"widget_rows": "W1002"}, policy_lines=[line])
    generation = _gap_generation(tmp_path, write("k0", kind="exchange"), write("k1", kind="change"),
                                 write("k2", statement="Ada Byron wants her sealed widget swapped today", at=3,
                                       kind=""),
                                 write("k3", statement="Ada Byron wants her sealed widget swapped today", at=3,
                                       kind="exchange"), _done(), _done())
    M.from_gaps(generation, batch=8)
    [record] = generation.found
    assert record.source.gap["kind"] == "change"
    assert generation.dropped == Counter({T.KIND_NOT_IN_PLAN: 1})
    results = _results(generation.model)
    assert "kind not in plan: give one of cancel, change" in results[0] and "dropped" in results[3]


def test_a_gaps_write_without_its_ask_or_its_end_or_both_is_refused_and_counted(tmp_path):
    bound = {"binding": {"holder_rows": "holder_7"}}
    generation = _gap_generation(
        tmp_path, _gap_write("p0", "Ada Byron wants to cancel her slot holding", 0, ask="", end="", **bound),
        _gap_write("p1", "Ada Byron wants to cancel her slot holding", 0, ask="", end="", **bound),
        _gap_write("p2", "Ada Byron wants to cancel her slot holding now", 1, end="", **bound),
        _gap_write("p3", "Ada Byron wants to cancel her slot holding now", 1, ask="", **bound), _done(), _done())
    M.from_gaps(generation, batch=8)
    results = _results(generation.model)
    assert generation.found == [] and T.PAIR_MISSING.replace("_", " ") in results[0]
    assert T.ASK_WITHOUT_END.replace("_", " ") in results[2] and T.END_WITHOUT_ASK.replace("_", " ") in results[3]
    assert generation.dropped == Counter({T.PAIR_MISSING: 1, T.END_WITHOUT_ASK: 1})


def test_no_gaps_pass_runs_without_a_model(tmp_path):
    generation = _gap_generation(tmp_path)
    M.from_gaps(generation)
    assert generation.found == [] and generation.gaps == {}


def test_generate_with_only_the_gaps_source_reads_the_plan_on_disk_and_runs_no_other_session(tmp_path):
    generation = _gap_generation(tmp_path)
    workdir = generation.world.workdir
    P.write_plan(workdir, generation.plan)
    model = TestModel([_gap_write("g0", "Ada Byron wants to cancel her slot holding", 0,
                                  binding={"holder_rows": "holder_7"}), _done(), _done(), _done()])
    body = M.generate(workdir, model=model, sources=["gaps"])
    assert body["born"] == 1 and body["coverage"]["per_kind"]["gaps"] == 1 and body["notes"] == []
    assert body["gaps"]["before"]["hit"] == 2 and body["gaps"]["after"]["hit"] == 3
    assert "Generate scenarios for the gaps in the store" in str(model.calls[0]["messages"])


def test_the_report_prints_the_gaps_pass_with_the_cells_and_policy_lines_before_and_after(tmp_path):
    from kullback.runner.records import write_json
    from kullback.synth import report

    workdir = make_workdir(tmp_path)
    gaps = {"items": 4, "before": {"hit": 2, "possible": 2, "lines_covered": 1, "lines": 3},
            "after": {"hit": 4, "possible": 6, "lines_covered": 2, "lines": 3}}
    write_json(R.store_dir(workdir) / R.GENERATED, {"gaps": gaps})
    printed = report.lines(workdir)
    assert "gaps pass: 4 items" in printed
    assert "cells hit before the gaps pass: 2 of 2, after: 4 of 6" in printed
    assert "policy lines covered before the gaps pass: 1 of 3, after: 2 of 3" in printed
    assert "| gaps | 0 | 0 | 0 | 0 | 0 | 0 |" in printed


def test_a_table_with_no_rows_gives_no_possible_cell_and_no_gap_item(tmp_path):
    from kullback.runner.records import write_json

    generation = _gap_generation(tmp_path)
    write_json(generation.world.workdir / "db.json", {**STATE, "empty_rows": {}})
    generation.world = M.world_of(generation.world.workdir)
    assert sorted(M.cell_tables(generation.world.state)) == sorted(STATE)
    assert not any(item["gap"].get("table") == "empty_rows" for item in M.gap_items(generation))
    assert M.gap_counts(generation)["possible"] == len(STATE)


def test_a_retrys_turn_budget_follows_the_turns_per_item_its_first_pass_spent(tmp_path):
    """All eight small live retries capped on the floor of MIN_TURNS with the hardest items (D413)."""
    generation, _ = _generation(tmp_path)
    root = T.GenerateRoot(world=generation.world, kind="recordings", items=[{}, {}])
    assert T.turns_for(root) == T.MIN_TURNS
    assert T.turns_for(root, per_item=20) == 40
    assert T.turns_for(root, per_item=T.MAX_TURNS) == T.MAX_TURNS


def test_the_first_pass_turns_are_counted_per_kind_for_the_retry(tmp_path):
    generation, _ = _generation(tmp_path)
    _plan(generation, (MOVE, [0], []), (BROKEN, [1], []))
    generation.model = TestModel([*_quiet(), *_quiet()])
    M.from_recordings(generation, batch=8)
    assert generation.spent["recordings"] == 2 * len(_quiet())
