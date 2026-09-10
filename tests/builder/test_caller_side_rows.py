"""Caller-side state homed to a row (D233).

The domain is invented, the way every Builder test's is: an apiary whose keeper runs tools of
its own against the hives beside the assistant, answering in rows rather than in sentences. A
second requestor with prose results and a third with no mined table cover the sentence and the
fallback paths. Nothing in this file comes from a customer's corpus.
"""

from __future__ import annotations

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder.cluster import split_by_world, world_split_reason
from kullback.runner.records import (
    Column,
    EntitySchema,
    Task,
    ToolCall,
    ToolSig,
    Trace,
    Turn,
)

KEEPER = "keeper"
CARETAKER = "caretaker"
GUEST = "guest"


def _column(table, name, class_, revealed_by=None):
    return Column(table=table, name=name, **{"class": class_}, classified_by="rule",
                  evidence={"revealed_by": revealed_by} if revealed_by else {})


def hive_schema():
    """One table the keeper's own tools revealed, with one column of each class."""
    return EntitySchema(
        tables=["hives"],
        columns=[
            _column("hives", "brood", "hard", KEEPER),
            _column("hives", "hum", "exempt", KEEPER),
            _column("hives", "scent", "semantic", KEEPER),
        ],
    )


def full_schema():
    schema = hive_schema()
    schema.tables = sorted([*schema.tables, "greenhouse"])
    schema.columns = sorted([*schema.columns,
                             _column("greenhouse", "vent", "hard", CARETAKER),
                             _column("greenhouse", "warmth", "exempt", CARETAKER)],
                            key=lambda c: (c.table, c.name))
    return schema


def bare_schema():
    """No mined table for any requestor: every caller-side sighting takes the fallback."""
    return EntitySchema(
        tables=["plots"],
        columns=[Column(table="plots", name="plot_id", **{"class": "hard"}, classified_by="rule")],
        id_patterns={"plot_id": r"^p\d+$"},
    )


def call(name, result, requestor=KEEPER, args=None, cid="c0"):
    return ToolCall(id=cid, name=name, args=dict(args or {}), result=result,
                    requestor=requestor, raw_ptr=PTR, has_result=True, resolved=True)


def trace(trace_id, calls, said="the hives sound restless, please check them"):
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 turns=[Turn(idx=0, role="user", content=said, raw_ptr=PTR)], tool_calls=calls)


def keeper_trace(trace_id, row, cid="c0"):
    return trace(trace_id, [call("check_hive", dict(row), cid=cid)])


def test_a_requestor_sighting_becomes_a_row_in_the_mined_table():
    """The keeper's state is filed under the mined table with its columns, not under the requestor."""
    traces = [keeper_trace("keep_a", {"brood": "capped", "hum": "low", "scent": "sweet"})]
    worlds = ce.trace_worlds(traces, hive_schema(), set())
    keys = set(worlds["keep_a"])
    assert ("check_hive", "hives", KEEPER, "brood") in keys
    assert not any(key[1] == "" for key in keys), "no opaque fallback key where a table is mined"
    assert not any(key[3] in ("hum", "scent") for key in keys if len(key) == 4)


def test_the_row_is_built_column_by_column_with_the_earliest_sighting_winning():
    """Two reads through one tool state one version, the way a customer row does."""
    traces = [trace("keep_a", [call("check_hive", {"brood": "capped"}, cid="c0"),
                               call("check_hive", {"brood": "open", "hum": "low"}, cid="c1")])]
    worlds = ce.trace_worlds(traces, hive_schema(), set())
    assert set(worlds["keep_a"]) == {("check_hive", "hives", KEEPER, "brood")}
    assert worlds["keep_a"][("check_hive", "hives", KEEPER, "brood")][1] == "hard"


def test_the_row_pins_into_the_overlay_and_a_read_body_reads_it(tmp_path):
    """Once homed, the keeper's row pins like any other row and a body can serve it."""
    row = {"brood": "capped", "hum": "low", "scent": "sweet"}
    traces = [keeper_trace("keep_a", row)]
    sigs = [ToolSig(name="check_hive", kind="read")]
    state = ce.build_starting_state(
        traces, hive_schema(), tmp_path, tasks=[Task(id="t", run_ids=["keep_a"])],
        tool_sigs=sigs, synthetic=False,
        revealed_rows={("hives", KEEPER): {"keep_a": dict(row)}})
    assert state.db["hives"][KEEPER] == row
    overlay, values = ce.load_overlay(tmp_path, "t")
    assert ("hives", KEEPER) in {(row.table, row.id) for row in overlay.rows}
    body = ("row = next(iter(self.db.hives.values()))\n"
            "return {'brood': row.brood}\n")
    source = ce.module_source(hive_schema(), sigs, {"check_hive": body})
    toolkit = ce.load_toolkit(source, state.db, overlay=overlay, overlay_values=values)
    assert toolkit.check_hive() == {"brood": "capped"}


def test_runs_differing_only_in_exempt_or_semantic_columns_share_a_task():
    """A cosmetic or exempt part of the sentence no longer splits Tasks."""
    traces = [keeper_trace("keep_a", {"brood": "capped", "hum": "low", "scent": "sweet"}),
              keeper_trace("keep_b", {"brood": "capped", "hum": "high", "scent": "sharp"})]
    worlds = ce.trace_worlds(traces, hive_schema(), set())
    assert worlds["keep_a"] == worlds["keep_b"]
    assert len(split_by_world(traces, worlds)) == 1


def test_a_hard_difference_splits_with_a_reason_naming_the_column_class():
    traces = [keeper_trace("keep_a", {"brood": "capped", "hum": "low", "scent": "sweet"}),
              keeper_trace("keep_c", {"brood": "open", "hum": "low", "scent": "sweet"})]
    worlds = ce.trace_worlds(traces, hive_schema(), set())
    assert [[t.trace_id for t in part] for part in split_by_world(traces, worlds)] == [
        ["keep_a"], ["keep_c"]]
    reason = world_split_reason(["keep_a", "keep_c"], worlds)
    assert "hard" in reason and "brood" in reason
    assert "hives" in reason and "check_hive" in reason


def test_a_requestor_write_closes_its_world():
    """What the keeper's device showed after its own write is not the state it started in."""
    traces = [trace("keep_a", [call("check_hive", {"brood": "capped"}, cid="c0"),
                               call("smoke_hive", {"brood": "calm"}, cid="c1"),
                               call("check_hive", {"brood": "open"}, cid="c2")])]
    worlds = ce.trace_worlds(traces, hive_schema(), {"smoke_hive"})
    assert set(worlds["keep_a"]) == {("check_hive", "hives", KEEPER, "brood")}
    before = ce.trace_worlds([keeper_trace("keep_a", {"brood": "capped"})], hive_schema(), set())
    assert worlds["keep_a"] == before["keep_a"]


def test_a_prose_sentence_is_still_the_version_where_no_reader_may_read():
    """D216 holds for prose: two sentences that differ split, on the opaque sentence key.

    This is the fallback the pre-D233 split always took; a prose sighting states its columns
    inside the sentence and no reader may read them in the split, so it keeps that key even
    where the schema mines a table for its requestor.
    """
    traces = [trace("care_a", [call("check_vent", "Vent: CLOSED", requestor=CARETAKER)], said="cold"),
              trace("care_b", [call("check_vent", "Vent: OPEN", requestor=CARETAKER)], said="cold")]
    worlds = ce.trace_worlds(traces, full_schema(), set())
    assert set(worlds["care_a"]) == {("check_vent", "", CARETAKER)}
    assert len(split_by_world(traces, worlds)) == 2


def test_a_requestor_with_no_mined_table_falls_back_to_the_sentence():
    """The old opaque key and the old split stand where the schema mines no table."""
    traces = [trace("guest_a", [call("check_bench", "Bench: free", requestor=GUEST)], said="sit"),
              trace("guest_b", [call("check_bench", "Bench: taken", requestor=GUEST)], said="sit")]
    worlds = ce.trace_worlds(traces, bare_schema(), set())
    assert set(worlds["guest_a"]) == {("check_bench", "", GUEST)}
    assert len(split_by_world(traces, worlds)) == 2
    reason = world_split_reason(["guest_a", "guest_b"], worlds)
    assert GUEST in reason and "whole recorded sentence" in reason


def test_the_homing_moves_with_a_revealed_reclassification_and_nowhere_else():
    """The revealed classes are a grouping input; a customer reclassification is not (D216)."""
    schema = hive_schema()
    before = ce.homing_hash(schema)
    schema.columns[0].class_ = "exempt"
    assert ce.homing_hash(schema) != before
    customer = bare_schema()
    assert ce.homing_hash(customer) == ce.homing_hash(bare_schema())
