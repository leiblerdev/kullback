"""Situations from the web and a target count of scenarios (D410), on scripted models only."""

from __future__ import annotations

import json

from kullback import domain
from kullback.ai.provider import TestModel
from kullback.synth import generate as M
from kullback.synth import generate_tools as T
from kullback.synth import plan as P
from kullback.synth import records as R
from kullback.synth import report
from tests.synth.fixture import BOUND, DESCRIPTION, PLANNED, REFUSE, URL, call, make_workdir, web
from tests.synth.test_generate import MOVE, SPAN, _bound, _done, _results, _write

SEARCHED = "a holder asks to move a widget that is already sealed"
SEALED_ASK = "Ada Byron asks to move her widget that is already sealed to slot one"


def _situation(statement: str, kind: str = "change", call_id: str = "w0") -> object:
    return call("write_situation", {"statement": statement, "kind": kind, "url": URL, "span": SPAN}, call_id)


def _planned_store(tmp_path, *kinds: str) -> tuple:
    """A workdir whose plan on disk holds one situation per kind given, the first holding every opening."""
    workdir = make_workdir(tmp_path)
    rows = M.openings(workdir)
    plan = P.Plan(situations=[{"id": P.situation_id(f"{PLANNED} {at}" if at else PLANNED),
                               "statement": f"{PLANNED} {at}" if at else PLANNED, "kind": kind, "variants": [],
                               "openings": rows if at == 0 else []} for at, kind in enumerate(kinds)])
    P.write_plan(workdir, plan)
    return workdir, plan


def _searching(workdir, model) -> M.Generation:
    generation = M.Generation(world=M.world_of(workdir, DESCRIPTION), model=model, bus=T.bus_for(workdir))
    generation.plan = P.load_plan(workdir)
    return generation


def test_a_plan_search_session_writes_situations_on_a_page_span_with_no_openings_into_the_plan_on_disk(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change")
    model = TestModel([call("search", {"query": "what holders ask"}, "s0"), call("read_page", {"url": URL}, "s1"),
                       _situation(SEARCHED), _done()])
    generation = _searching(workdir, model)
    assert M.plan_search(generation, 1, engine=web(workdir)[0], refuse=REFUSE) == 1
    [_, made] = P.load_plan(workdir).situations
    raw = domain.cached_text(workdir, made["source"]["key"]).encode("utf-8")
    assert raw[made["source"]["start"]:made["source"]["end"]].decode("utf-8") == SPAN == made["span"]
    assert made["source"]["kind"] == "search" and made["source"]["url"] == URL
    assert made["openings"] == [] and made["variants"] == [] and made["kind"] == "change"


def test_a_situation_nearly_repeating_one_of_the_plan_or_outside_its_kinds_or_on_an_unread_page_is_refused(tmp_path):
    workdir, plan = _planned_store(tmp_path, "change")
    model = TestModel([_situation(SEARCHED), call("read_page", {"url": URL}, "s1"), _situation(SEARCHED, "refund", "w1"),
                       _situation(f"{PLANNED} today", call_id="w2"), _done()])
    generation = _searching(workdir, model)
    assert M.plan_search(generation, 3, engine=web(workdir)[0], refuse=REFUSE) == 0
    results = _results(model)
    assert T.NOT_READ.replace("_", " ") in results[0]
    assert P.KIND.replace("_", " ") in results[2] and "give one of change" in results[2]
    assert P.NEAR.replace("_", " ") in results[3] and plan.situations[0]["id"] in results[3]
    assert len(P.load_plan(workdir).situations) == 1


def test_the_plan_search_opening_asks_for_a_count_and_names_the_thinnest_kinds_first(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change", "change", "cancel", "question", "question", "question")
    model = TestModel([_done()])
    M.plan_search(_searching(workdir, model), 4, engine=web(workdir)[0], refuse=REFUSE)
    opening = " ".join(str(model.calls[0]["messages"][-1]["content"]).split())
    assert "Add 4 situations" in opening
    assert "cancel 1, change 2, question 3; thinnest: cancel, change, question" in opening


def test_a_scenario_of_a_search_born_situation_carries_its_span_and_lists_the_span_beside_it(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change")
    searcher = TestModel([call("read_page", {"url": URL}, "s1"), _situation(SEARCHED), _done()])
    generation = _searching(workdir, searcher)
    M.plan_search(generation, 1, engine=web(workdir)[0], refuse=REFUSE)
    born = P.situation_id(SEARCHED)
    generation.model = writer = TestModel([call("list_openings", {}, "l0"), _bound("b0", SEALED_ASK, born), _done()])
    generation.plan.situations = generation.plan.situations[1:]
    M.from_recordings(generation)
    [record] = generation.found
    assert record.situation == born and record.source.kind == "search" and record.frequency == 0
    held = generation.plan.situations[0]["source"]
    assert (record.source.url, record.source.start, record.source.end) == (URL, held["start"], held["end"])
    assert "page_span" in _results(writer)[0] and SPAN in _results(writer)[0]


def test_a_search_scenario_takes_a_kind_from_the_plan_and_counts_under_it(tmp_path):
    workdir, plan = _planned_store(tmp_path, "change")
    model = TestModel([call("read_page", {"url": URL}, "s1"),
                       _write("s2", SEARCHED, "search", url=URL, span=SPAN, binding=BOUND),
                       _write("s3", SEARCHED, "search", url=URL, span=SPAN, binding=BOUND, kind="change"), _done()])
    generation = _searching(workdir, model)
    M.from_search(generation, engine=web(workdir)[0], refuse=REFUSE)
    [record] = generation.found
    assert T.KIND_NOT_IN_PLAN.replace("_", " ") in _results(model)[1] and "give one of change" in _results(model)[1]
    assert record.kind == "change" and "change" in str(model.calls[0]["messages"])
    assert M.spread([record], plan, ["widget_rows"])[0]["situation_kind"] == {"change": 1}


def _round_model(*more) -> TestModel:
    """Round 1: the recordings session writes the planned situation's scenario, the loose search writes none."""
    return TestModel([_bound("m0", MOVE, P.situation_id(PLANNED)), _done(), _done(), *more])


def test_a_target_runs_rounds_of_situations_from_the_web_and_their_scenarios_until_the_store_keeps_it(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change")
    model = _round_model(call("read_page", {"url": URL}, "s1"), _situation(SEARCHED), _done(),
                         _bound("b0", SEALED_ASK, P.situation_id(SEARCHED)), _done())
    body = M.generate(workdir, model=model, engine=web(workdir)[0], refuse=REFUSE, description=DESCRIPTION,
                      sources=["recordings", "search"], target=2)
    assert body["stop"] == "target" and body["target"] == 2 and body["born"] == 2
    assert [row["born"] for row in body["rounds"]] == [
        {"recordings": 1, "search": 0, "ask": 0, "gaps": 0}, {"recordings": 0, "search": 1, "ask": 0, "gaps": 0}]
    assert [(row["situations"], row["kept"]) for row in body["rounds"]] == [(0, 1), (1, 2)]
    assert "Add 1 situations" in json.dumps([c["messages"] for c in model.calls])
    assert body["diversity"]["kept"] == 2 and body["diversity"]["situations_per_kind"] == {"change": 2}
    assert body["diversity"]["rouge_l_median"] is not None
    assert json.loads((R.store_dir(workdir) / R.GENERATED).read_text())["rounds"] == body["rounds"]


def test_a_round_adding_under_five_percent_of_the_target_stops_the_rounds_and_says_so(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change")
    body = M.generate(workdir, model=_round_model(), engine=web(workdir)[0], refuse=REFUSE,
                      description=DESCRIPTION, sources=["recordings", "search"], target=100)
    assert body["stop"] == "slow" and len(body["rounds"]) == 1 and body["born"] == 1


def test_without_a_target_generate_writes_no_rounds(tmp_path):
    workdir, _ = _planned_store(tmp_path, "change")
    body = M.generate(workdir, model=_round_model(), engine=web(workdir)[0], refuse=REFUSE,
                      description=DESCRIPTION, sources=["recordings", "search"])
    assert "rounds" not in body and "target" not in body


def test_the_report_prints_one_line_per_round_the_stop_reason_and_the_diversity(tmp_path):
    from kullback.runner.records import write_json

    workdir = make_workdir(tmp_path)
    born = {"recordings": 3, "search": 0, "ask": 0, "gaps": 1}
    write_json(R.store_dir(workdir) / R.GENERATED, {
        "target": 10, "stop": "slow",
        "rounds": [{"round": 1, "situations": 0, "born": born, "kept": 4},
                   {"round": 2, "situations": 2, "born": {**born, "recordings": 0, "search": 1, "gaps": 0}, "kept": 5}],
        "diversity": {"kept": 5, "rouge_l_median": 0.2, "situation_kind": {"change": 3, "cancel": 2},
                      "cells": {"hit": 3, "possible": 4}, "situations_per_kind": {"cancel": 1, "change": 2}}})
    printed = report.lines(workdir)
    assert "round 1: situations from the web 0, born recordings 3, search 0, ask 0, gaps 1, kept 4" in printed
    assert "round 2: situations from the web 2, born recordings 0, search 1, ask 0, gaps 0, kept 5" in printed
    assert "target 10, stopped: a round added under 5 percent of the target" in printed
    assert "diversity: pairwise ROUGE-L median 0.2 over 5 kept, cells hit 3 of 4" in printed
    assert "scenarios per kind: change 3, cancel 2" in printed and "situations per kind: change 2, cancel 1" in printed
