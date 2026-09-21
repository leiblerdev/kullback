import json

import pytest

from kullback.builder.mine import mine_schema
from kullback.runner.records import RawPtr, ToolCall, Trace


@pytest.mark.parametrize("observed_elsewhere", [False, True])
@pytest.mark.parametrize("address_returns_row", [False, True])
def test_observed_identifier_support_does_not_cross_table_boundaries(tmp_path, observed_elsewhere, address_returns_row):
    ptr = RawPtr(file_hash='invented')
    calls = [
        ToolCall(name='get_berth', args={'berth_tag': 'T1'}, result={'berth_tag': 'T1', 'colour': 'red'} if address_returns_row else 'red', raw_ptr=ptr),
        ToolCall(name='list_berths', args={}, result=[{'berth_tag': 'T1', 'colour': 'red'}, {'berth_tag': 'T2', 'colour': 'blue'}], raw_ptr=ptr),
    ]
    if observed_elsewhere:
        calls.append(ToolCall(name='get_note', args={'note_id': 'N1'}, result={'note_id': 'N1', 'berth_tag': 'a narrative description rather than an addressable entity'}, raw_ptr=ptr))
    trace = Trace(trace_id='invented-trace', raw_hash='invented', ingest_version='test', source='synthetic', tool_calls=calls, raw_ptr=ptr)
    snapshot = tmp_path / 'world.json'
    snapshot.write_text(json.dumps({'notes': {'N1': {'note_id': 'N1', 'berth_tag': 'a narrative description rather than an addressable entity'}}}))
    schema = mine_schema([trace], db_json_path=snapshot)
    observed = next(c for c in schema.columns if c.table == 'berth_tags' and c.name == 'berth_tag')
    unrelated = next(c for c in schema.columns if c.table == 'notes' and c.name == 'berth_tag')
    assert observed.evidence['id_basis'] == 'observed'
    assert unrelated.evidence.get('id_basis') != 'observed'
    assert not unrelated.evidence.get('id_support')
    assert 'notes.berth_tag' not in schema.id_patterns


def test_a_filter_argument_does_not_inherit_another_tables_identifier_evidence():
    ptr = RawPtr(file_hash="invented")
    calls = [
        ToolCall(name="get_berth", args={"berth_tag": "T1"}, result="red", raw_ptr=ptr),
        ToolCall(name="list_berths", args={}, result=[
            {"berth_tag": "T1", "colour": "red"},
            {"berth_tag": "T2", "colour": "blue"},
        ], raw_ptr=ptr),
        ToolCall(name="search_notes", args={"berth_tag": "T1"},
                 result={"note_id": "N1", "berth_tag": "T1"}, raw_ptr=ptr),
    ]
    trace = Trace(trace_id="filter-proof", raw_hash="invented", ingest_version="test",
                  source="synthetic", tool_calls=calls, raw_ptr=ptr)
    schema = mine_schema([trace])
    source = next(c for c in schema.columns if c.table == "berth_tags" and c.name == "berth_tag")
    unrelated = next(c for c in schema.columns if c.table == "notes" and c.name == "berth_tag")
    assert source.evidence["id_basis"] == "observed"
    assert unrelated.evidence.get("id_basis") != "observed"
    assert not unrelated.evidence.get("id_support")
    assert "notes.berth_tag" not in schema.id_patterns
