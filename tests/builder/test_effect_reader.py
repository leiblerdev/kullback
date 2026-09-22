"""Effect evidence reads prose results through the same reader the comparer uses.

The world here is an invented ledger of accounts. A read answers in a sentence naming an
account and the balance it carries, and a write answers in a sentence carrying no column at
all. A reader that parses the balance out of a sentence turns those answers into sightings
of the row; without it they state nothing, so the only sightings are the Starting state the
run began in and the one dict read at the end, and the column that moved between them is
filed without an owner.

That shape is the whole point. Nothing about it is any customer's, and every mechanism
under test here is about the shape and not about the names: which results become sightings,
which span a write is credited for, and whether the column it moved is checked or left
unattributed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from conftest import PTR
from kullback.builder import build as build_module
from kullback.builder import effects
from kullback.runner.records import Column, EntitySchema, ToolCall, Trace, Turn

READ = "get_account"
WRITE = "adjust_account"
AUDIT = "note_observation"

START = {"accounts": {"AC01": {"account_id": "AC01", "balance": 100}}}


def schema() -> EntitySchema:
    columns = [Column(table="accounts", name=name, **{"class": "hard"})
               for name in ("account_id", "balance")]
    return EntitySchema(tables=["accounts"], columns=columns,
                        id_patterns={"accounts.account_id": r"^AC\d{2}$"})


def sentence_reader(tool: str, result: Any) -> Any:
    """The balance a sentence answer states, or nothing where it states none."""
    if not isinstance(result, str):
        return None
    found = re.search(r"balance (\d+)", result)
    return {"balance": int(found.group(1))} if found else None


def call(call_id: str, name: str, args: dict, result) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, requestor="assistant",
                    raw_ptr=PTR, trace_id="tr1")


def sentence_trace() -> Trace:
    calls = [
        call("c1", AUDIT, {"note": "audit"}, "Noted."),
        call("c2", READ, {"account_id": "AC01"}, "Account AC01 carries balance 100."),
        call("c3", WRITE, {"delta": -18}, "Adjusted by minus 18."),
        call("c4", READ, {"account_id": "AC01"}, "Account AC01 carries balance 82."),
        call("c5", READ, {"account_id": "AC01"}, {"account_id": "AC01", "balance": 82}),
    ]
    turns = [Turn(idx=0, role="assistant", content="How can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content="Move the balance.", raw_ptr=PTR)]
    for one in calls:
        turns.append(Turn(idx=len(turns), role="assistant", content=None,
                          tool_call_ids=[one.id], raw_ptr=PTR))
        turns.append(Turn(idx=len(turns), role="tool", content=json.dumps(one.result),
                          tool_call_ids=[one.id], raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="assistant", content="Done.", raw_ptr=PTR))
    turns.append(Turn(idx=len(turns), role="user", content="Thanks. ###STOP###", raw_ptr=PTR))
    return Trace(trace_id="tr1", raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR, system_prompt="You are the agent.")


def balance_columns(seen) -> list:
    return [column for rows in seen.values() for effect in rows for column in effect.columns
            if column.table == "accounts" and column.path == "balance"]


def test_a_sentence_answer_naming_a_row_and_column_is_a_checked_effect_with_the_reader():
    seen = effects.observe_effects([sentence_trace()], schema(), {WRITE, AUDIT}, db=dict(START),
                                   read_result=sentence_reader)
    columns = [column for column in balance_columns(seen) if column.checked]
    assert len(columns) == 1
    assert (columns[0].before, columns[0].after) == (100, 82)
    assert columns[0].unattributed is False
    assert effects.counts(seen)["effect_columns_checked"] == 1


def test_the_same_sentence_answers_leave_the_column_unattributed_without_the_reader():
    seen = effects.observe_effects([sentence_trace()], schema(), {WRITE, AUDIT}, db=dict(START))
    columns = balance_columns(seen)
    assert len(columns) == 1
    assert (columns[0].before, columns[0].after) == (100, 82)
    assert columns[0].checked is False
    assert columns[0].unattributed is True
    assert effects.counts(seen)["effect_columns_checked"] == 0


def test_the_compile_stage_reads_its_effect_evidence_through_the_stage_reader(
        monkeypatch, tmp_path):
    """The compile stage builds its reader from its own inputs and hands it to the evidence."""
    built = {}

    def fake_result_reader(artifact, traces, workdir):
        built["artifact"] = artifact
        return sentence_reader

    monkeypatch.setattr(build_module.readers, "result_reader", fake_result_reader)
    seen = build_module.compile_stage_effects([sentence_trace()], schema(), {WRITE, AUDIT},
                                              db=dict(START), worlds=None,
                                              readers_artifact={"note": "test"}, workdir=tmp_path)
    assert built["artifact"] == {"note": "test"}
    columns = [column for column in balance_columns(seen) if column.checked]
    assert len(columns) == 1
    assert (columns[0].before, columns[0].after) == (100, 82)


def test_the_replay_stage_reads_each_task_through_the_one_stage_reader(monkeypatch):
    """The replay path takes the reader the stage built once, and builds none of its own."""
    def no_reader(artifact, traces, workdir):
        raise AssertionError("the replay path builds its reader once per stage, not per task")

    monkeypatch.setattr(build_module.readers, "result_reader", no_reader)
    seen = build_module.replay_stage_effects([sentence_trace()], schema(), {WRITE, AUDIT},
                                             db=dict(START), effect_reader=sentence_reader)
    columns = [column for column in balance_columns(seen) if column.checked]
    assert len(columns) == 1
    assert (columns[0].before, columns[0].after) == (100, 82)
    bare = build_module.replay_stage_effects([sentence_trace()], schema(), {WRITE, AUDIT},
                                             db=dict(START), effect_reader=None)
    assert effects.counts(bare)["effect_columns_checked"] == 0
