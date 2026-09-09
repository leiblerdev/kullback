"""The dependency graph mined from recordings, and the walks taken over it (D224)."""

from __future__ import annotations

from kullback import graph


def _run(*calls: dict) -> list[dict]:
    return [dict(call, error=call.get("error")) for call in calls]


LOOKUP = {"name": "find_kite", "args": {"colour": "teal"}, "result": {"kite_id": "k1"}}
DETAILS = {"name": "get_kite", "args": {"kite_id": "k1"},
           "result": {"kite_id": "k1", "spool_id": "s7", "length": "40"}}
RETIE = {"name": "retie_kite", "args": {"kite_id": "k1", "length": "60"}, "result": {"kite_id": "k1"}}


def test_graph_holds_only_the_edges_the_recordings_showed():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE)], write_tools=["retie_kite"])
    pairs = {(edge["from"], edge["to"]) for edge in body["edges"]}
    assert ("find_kite", "get_kite") in pairs
    assert ("get_kite", "retie_kite") in pairs
    # A tool no recording of this world ever called is in no edge, so no walk may ever reach it.
    assert not [edge for edge in body["edges"] if "check_spool" in (edge["from"], edge["to"])]
    assert "check_spool" not in {node["name"] for node in body["nodes"]}


def test_an_edge_carries_the_argument_it_fed_and_the_column_it_came_from():
    body = graph.mine([_run(LOOKUP, DETAILS)], write_tools=[])
    edge = next(e for e in body["edges"] if e["to"] == "get_kite" and e["kind"] == graph.VALUE_EDGE)
    assert edge["arg"] == "kite_id" and edge["column"] == "kite_id"


def test_an_edge_weight_counts_the_runs_that_showed_it_and_not_the_calls():
    once = graph.mine([_run(LOOKUP, DETAILS)], write_tools=[])
    twice = graph.mine([_run(LOOKUP, DETAILS), _run(LOOKUP, DETAILS)], write_tools=[])
    assert _weight(once, "find_kite", "get_kite") == 1
    assert _weight(twice, "find_kite", "get_kite") == 2
    # One Run that took the same step twice still shows the step once.
    repeated = graph.mine([_run(LOOKUP, DETAILS, DETAILS)], write_tools=[])
    assert _weight(repeated, "find_kite", "get_kite") == 1


def _weight(body: dict, source: str, target: str) -> int:
    return max([int(edge["weight"]) for edge in body["edges"]
                if edge["from"] == source and edge["to"] == target] or [0])


def test_a_read_and_a_write_on_one_row_are_joined_even_where_no_value_was_carried():
    silent = {"name": "check_spool", "args": {"spool_id": "s7"}, "result": "ok"}
    write = {"name": "retie_kite", "args": {"spool_id": "s7", "length": "60"}, "result": {"ok": True}}
    body = graph.mine([_run(silent, write)], write_tools=["retie_kite"])
    kinds = {edge["kind"] for edge in body["edges"] if edge["from"] == "check_spool"}
    assert graph.ROW_EDGE in kinds


def test_a_call_the_world_refused_carries_nothing_forward():
    failed = dict(LOOKUP, error={"message": "no such kite"})
    body = graph.mine([_run(failed, DETAILS)], write_tools=[])
    assert not [edge for edge in body["edges"] if edge["from"] == "find_kite"]


def test_a_keyed_collection_collapses_so_an_edge_can_bind_to_another_row():
    answer = {"name": "get_kite", "args": {"kite_id": "k1"},
              "result": {"spools": {"s7": {"id": "s7", "turns": "3"}}}}
    used = {"name": "retie_kite", "args": {"spool_id": "s7"}, "result": {}}
    body = graph.mine([_run(answer, used)], write_tools=["retie_kite"])
    edge = next(e for e in body["edges"] if e["kind"] == graph.VALUE_EDGE)
    assert "s7" not in edge["column"] and edge["column"].startswith("spools{}")


def test_a_walk_holds_the_writes_and_the_tools_its_band_asks_for():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE)], write_tools=["retie_kite"])
    steps = graph.walk(body, "w1t3p1", "seed-1")
    writes, tools = graph.reached(steps, ["retie_kite"])
    assert writes == 1 and tools == 3


def test_a_walk_of_a_band_asking_for_no_write_makes_none():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE)], write_tools=["retie_kite"])
    steps = graph.walk(body, "w0t2p1", "seed-1")
    writes, _tools = graph.reached(steps, ["retie_kite"])
    assert writes == 0


def test_a_walk_starts_where_a_recorded_run_started():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE)], write_tools=["retie_kite"])
    starts = {name for name, _count in graph.start_names(body)}
    assert graph.walk(body, "w1t2p1", "seed-2")[0]["tool"] in starts


def test_one_seed_draws_one_walk_however_often_it_is_asked():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE),
                       _run(LOOKUP, RETIE)], write_tools=["retie_kite"])
    first = graph.walk(body, "w1t2p1", "seed-3")
    assert [step["tool"] for step in first] == [step["tool"] for step in graph.walk(body, "w1t2p1", "seed-3")]


def test_two_seeds_may_draw_two_walks():
    body = graph.mine([_run(LOOKUP, DETAILS, RETIE)] * 3 + [_run(LOOKUP, RETIE)] * 3,
                      write_tools=["retie_kite"])
    walks = {tuple(step["tool"] for step in graph.walk(body, "w1t2p1", f"seed-{n}")) for n in range(12)}
    assert len(walks) > 1


def test_a_bucket_nobody_spells_that_way_asks_for_nothing():
    body = graph.mine([_run(LOOKUP, DETAILS)], write_tools=[])
    assert graph.bands("nonsense") == (-1, -1, -1)
    assert graph.walk(body, "nonsense", "seed-4") is None


def test_a_graph_with_no_recording_offers_no_walk():
    assert graph.walk(graph.mine([], write_tools=[]), "w1t1p1", "seed-5") is None
