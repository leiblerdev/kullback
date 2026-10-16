"""The synthetic generator's miner as sessions on the shared harness (D372), on scripted models only."""

from __future__ import annotations

import json
from collections import Counter

from kullback import domain
from kullback.ai.provider import LIVE_ENV_VAR, ModelReply, TestModel
from kullback.ai.usage import Usage
from kullback.builder.search import TINYFISH_KEY_VAR
from kullback.runner import budget
from kullback.synth import mine_prompt
from kullback.synth import mine_tools as T
from kullback.synth import miner as M
from kullback.synth import records as R
from kullback.synth import search as S
from tests.synth.fixture import DESCRIPTION, OPENINGS, POLICY, STATE, TOOLS, URL, call, make_workdir, web

SEALED_LINE = POLICY.splitlines()[0]
SPAN = "Customers often ask to move a widget after it has been sealed."


def _write(call_id: str, statement: str, source: str = "recordings", **args) -> ModelReply:
    return call("write_scenario", {"statement": statement, "source": source, **args}, call_id)


def _done(text: str = "Wrote them.") -> ModelReply:
    return ModelReply(content=text, usage=Usage(input=10, output=2))


def _mining(tmp_path, *replies) -> tuple[M.Mining, TestModel]:
    model = TestModel([*replies])
    workdir = make_workdir(tmp_path)
    return M.Mining(world=M.world_of(workdir), model=model, bus=T.bus_for(workdir)), model


def _results(model: TestModel) -> list[str]:
    """Every tool result the agent read, in order."""
    last = model.calls[-1]["messages"]
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


def test_a_recordings_session_groups_openings_with_their_frequency_and_policy_lines(tmp_path):
    mining, model = _mining(tmp_path,
                            call("list_openings", {}, "m0"),
                            _write("m1", "a holder wants a widget moved to another slot", openings=[0, 1],
                                   policy_lines=[SEALED_LINE]),
                            _write("m2", "a holder wants a replacement for a widget that arrived broken", openings=[2]),
                            _done())
    M.from_recordings(mining)
    assert [(r.frequency, r.source.task_ids) for r in mining.found] == [(2, ["task-a1", "task-b2"]), (1, ["task-c3"])]
    assert mining.found[0].policy_lines == [SEALED_LINE] and mining.dropped == Counter()
    assert OPENINGS["task-b2"] in _results(model)[0]


def test_a_refused_write_is_written_again_once_and_a_second_refusal_drops_and_counts_it(tmp_path):
    mining, model = _mining(tmp_path,
                            _write("m0", "a holder wants move_widget called", openings=[0]),
                            _write("m1", "a holder wants a widget moved to another slot", openings=[0]),
                            _write("m2", "a holder wants widget W1001 moved", openings=[1]),
                            _write("m3", "a holder wants widget W1002 moved now", openings=[1]),
                            _write("m4", "a holder wants a widget moved", openings=[2], policy_lines=["made up"]),
                            _done())
    M.from_recordings(mining)
    assert [r.statement for r in mining.found] == ["a holder wants a widget moved to another slot"]
    assert mining.dropped == Counter({M.ID_SHAPE: 1, T.POLICY_LINE: 1})
    results = _results(model)
    assert "write it again once" in results[0] and "dropped" in results[3]


def test_a_search_session_cites_the_byte_range_of_a_span_on_a_page_it_read(tmp_path):
    mining, model = _mining(tmp_path,
                            call("search", {"query": "what holders ask about sealed widgets"}, "m0"),
                            _write("m1", "a holder wants to move a widget that is already sealed", "search",
                                   url=URL, span=SPAN),
                            call("read_page", {"url": URL}, "m2"),
                            _write("m3", "a holder wants to move a widget that is already sealed", "search",
                                   url=URL, span=SPAN),
                            _write("m4", "a holder wants their widget painted", "search", url=URL,
                                   span="We paint widgets on request."),
                            _done())
    engine, inner = web(mining.world.workdir)
    M.from_search(mining, engine=engine)
    [record] = mining.found
    raw = domain.cached_text(mining.world.workdir, record.source.key).encode("utf-8")
    assert raw[record.source.start:record.source.end].decode("utf-8") == SPAN
    assert record.source.url == URL and record.frequency == 0
    assert mining.dropped == Counter({T.NO_SPAN: 1}) and T.NOT_READ.replace("_", " ") in _results(model)[1]
    assert inner.queries == ["what holders ask about sealed widgets"] and inner.fetched == [URL]


def test_with_the_live_switch_off_and_no_memo_no_search_session_runs_and_the_note_names_the_switch(tmp_path):
    mining, model = _mining(tmp_path)
    M.from_search(mining, env={})
    assert mining.found == [] and model.calls == [] and not (mining.world.workdir / "web_cache").exists()
    assert mining.notes == [S.OFF_NOTE] and LIVE_ENV_VAR in S.OFF_NOTE and TINYFISH_KEY_VAR in S.OFF_NOTE


def test_ask_lines_are_rewritten_in_the_third_person_and_the_line_as_typed_stays_in_the_source(tmp_path):
    typed = "test whether the agent will refund a sealed widget"
    statement = "a holder asks for a refund on a widget that is already sealed"
    mining, model = _mining(tmp_path, _write("m0", typed, "ask", openings=[0]),
                            _write("m1", statement, "ask", openings=[0]), _done())
    M.from_asks(mining, [typed, "call move_widget twice"])
    [record] = mining.found
    assert record.statement == statement and record.statement != typed
    assert M.statement_refusal(record.statement, mining.world) == ""
    assert (record.source.kind, record.source.line) == ("ask", typed)
    assert mining.dropped == Counter({M.TOOL_NAME: 1}) and T.AS_TYPED.replace("_", " ") in _results(model)[0]
    assert "move_widget" not in json.dumps(model.calls[0]["messages"][-1])


def test_without_a_model_no_ask_line_becomes_a_record_and_a_note_says_so(tmp_path):
    mining = M.Mining(world=M.world_of(make_workdir(tmp_path)), model=None)
    M.from_asks(mining, ["I want to see what happens when someone asks twice"])
    assert mining.found == [] and any("no ask line was rewritten" in note for note in mining.notes)


def test_a_mining_session_plays_on_the_workdir_bus_as_synth_and_books_its_calls_under_its_stage(tmp_path):
    workdir = make_workdir(tmp_path)
    model = TestModel([_write("m0", "a holder wants a widget moved to another slot", openings=[0, 1, 2]), _done()])
    M.mine(workdir, model=model, searching=False)
    records = [json.loads(line) for line in (workdir / "bus.jsonl").read_text().splitlines()]
    assert {row["agent"] for row in records} == {T.AGENT}
    assert any(row["event"].get("name") == "scenario_written" for row in records)
    assert budget.load_totals(workdir)["stages"][M.STAGE]["calls"] >= 1


def test_mining_folds_an_ask_into_the_recordings_record_and_writes_coverage(tmp_path):
    workdir = make_workdir(tmp_path)
    statement = "a holder wants a widget moved to another slot"
    which = "someone asks which slot their widget sits in"
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("I want my widget moved\ntest whether it says which slot I picked\n")
    model = TestModel([_write("m0", statement, openings=[0, 1]), _done(),
                       _write("a0", statement, "ask", openings=[0]), _write("a1", which, "ask", openings=[1]), _done()])
    body = M.mine(workdir, model=model, searching=False)
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


def test_mining_again_keeps_the_status_and_review_of_a_record_already_held(tmp_path):
    workdir = make_workdir(tmp_path)
    (workdir / R.DIR).mkdir()
    (workdir / R.DIR / R.ASKS).write_text("test which slot a widget sits in\n")

    def scripted():
        return TestModel([_done(), _write("a0", "someone asks which slot a widget sits in", "ask", openings=[0]),
                          _done()])

    M.mine(workdir, model=scripted(), searching=False)
    [record] = R.load_scenarios(workdir)
    R.write_scenario(workdir, record.model_copy(update={"status": "passed"}))
    M.mine(workdir, model=scripted(), searching=False)
    assert [r.status for r in R.load_scenarios(workdir)] == ["passed"]


def test_a_fix_session_rewrites_once_from_the_same_source_keeps_the_id_and_a_failed_rewrite_ends_unsure(tmp_path):
    world = M.world_of(make_workdir(tmp_path))
    first = R.born("a holder wants things", R.Source(kind="recordings", task_ids=["task-a1"]), frequency=1)
    second = R.born("a holder wants stuff", R.Source(kind="recordings", task_ids=["task-c3"]), frequency=1)
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


def test_the_mining_prompt_runs_in_gepa_order_and_names_nothing_of_the_environment():
    sections = mine_prompt.sections()
    assert [name for name, _ in sections] == ["receives", "tools", "examples", "choice", "feedback", "stop"]
    texts = [body for _, body in sections] + [mine_prompt.opening(kind, count=3) for kind in mine_prompt.OPENINGS]
    names = {tool["name"] for tool in TOOLS} | set(STATE) | {f for rows in STATE.values() for row in rows.values()
                                                              for f in row}
    assert not {name for name in names for text in texts if name in text}
    assert DESCRIPTION not in "".join(texts)
    assert all(len(line) < 150 for text in texts for line in text.splitlines())
