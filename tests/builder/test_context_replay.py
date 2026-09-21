from collections import deque
from types import SimpleNamespace

import pytest

from kullback.builder import build
from kullback.builder import compile_env as ce
from kullback.runner import replay
from kullback.runner.records import Column, EntitySchema, FieldStat, RawPtr, ToolCall, ToolSig
from kullback.runner.route import Router

SCHEMA = EntitySchema(tables=['loans'], columns=[Column(table='loans', name='loan_id', **{'class': 'hard'}, classified_by='rule'), Column(table='loans', name='opened', **{'class': 'hard'}, classified_by='rule')], id_patterns={'loans.loan_id': r'^L\d+$'})
SIGS = [ToolSig(name=name, args_fields=[FieldStat(name='patron', types=['str'], optional=False)], kind='write', unclassified=False) for name in ('open_plain', 'open_context')]
BODIES = {'open_plain': "return {'loan_id': 'L101', 'opened': '2024-03-04T05:06:07'}", 'open_context': "return {'loan_id': self.ctx.new_id('loans'), 'opened': self.ctx.now()}"}

def toolkit():
    return ce.load_toolkit(ce.module_source(SCHEMA, SIGS, BODIES), {'loans': {'L100': {'loan_id': 'L100', 'opened': '2024-01-02T10:00:00'}}})

def call(name, id_, result):
    return ToolCall(id=id_, name=name, args={'patron': 'ann'}, result=result, raw_ptr=RawPtr(file_hash='invented'))

def test_two_new_ids_before_insertion_are_unique():
    tools = toolkit()
    assert tools.ctx.new_id('loans') != tools.ctx.new_id('loans')

def test_one_recorded_call_does_not_reuse_its_id_twice():
    tools = toolkit()
    tools.ctx.feed_call({'new_ids': {'loans': ['L101']}, 'now': None})
    assert tools.ctx.new_id('loans') != tools.ctx.new_id('loans')

@pytest.mark.skipif(not hasattr(replay.ScoredRouter, "_feed_context"), reason="needs tool-context replay patch")
def test_replay_feeds_each_recorded_call_even_if_previous_body_ignored_context():
    first = call('open_plain', 'c1', {'loan_id': 'L101', 'opened': '2024-03-04T05:06:07'})
    second = call('open_context', 'c2', {'loan_id': 'L102', 'opened': '2024-03-05T05:06:07'})
    tools = toolkit()
    inner = Router(env_tools_module=tools, starting_state={}, tool_sigs=SIGS)
    kwargs = build._replay_context(tools, SimpleNamespace(tool_calls=[first, second]), SCHEMA)
    scored = replay.ScoredRouter(inner, deque([first, second]), **kwargs)
    scored.route(first.name, first.args)
    assert scored.route(second.name, second.args).result == second.result


def test_consuming_a_call_feed_does_not_mutate_the_saved_evidence():
    tools = toolkit()
    feed = {'new_ids': {'loans': ['L101']}, 'now': None}
    tools.ctx.feed_call(feed)
    assert tools.ctx.new_id('loans') == 'L101'
    assert feed == {'new_ids': {'loans': ['L101']}, 'now': None}


def test_replay_refuses_a_wrong_update_body_its_recorded_old_id():
    schema = SCHEMA
    update_sig = ToolSig(name='touch_loan', args_fields=[FieldStat(name='loan_id', types=['str'], optional=False)], kind='write', unclassified=False)
    wrong = {'touch_loan': "loan_id = self.ctx.new_id('loans')\nopened = self.ctx.now()\nself.db.loans[loan_id] = {'loan_id': loan_id, 'opened': opened}\nreturn {'loan_id': loan_id, 'opened': opened}\n"}
    tools = ce.load_toolkit(ce.module_source(schema, [update_sig], wrong), {'loans': {'L100': {'loan_id': 'L100', 'opened': '2024-01-02T10:00:00'}}})
    recorded = ToolCall(id='u1', name='touch_loan', args={'loan_id': 'L100'}, result={'loan_id': 'L100', 'opened': '2024-01-02T10:00:00'}, raw_ptr=RawPtr(file_hash='invented'))
    inner = Router(env_tools_module=tools, starting_state={}, tool_sigs=[update_sig])
    kwargs = build._replay_context(tools, SimpleNamespace(tool_calls=[recorded]), schema)
    scored = replay.ScoredRouter(inner, deque([recorded]), write_tools={'touch_loan'}, **kwargs)
    scored.route('touch_loan', {'loan_id': 'L100'})
    assert scored.checks[0]['verdict'] == replay.DIFFERS
