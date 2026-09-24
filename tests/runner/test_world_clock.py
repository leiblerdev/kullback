"""D283: a body reads time from the world clock, the recording's time for the Task, never the machine's.

The domain is invented: a notebook where opening a note stamps it. Nothing names a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.builder import env_files
from kullback.builder.compile_env import module_source
from kullback.gates.confinement import gate_confined
from kullback.runner import tool
from kullback.runner.records import EntitySchema, RawPtr, ToolCall, ToolSig, Trace, Turn, write_json
from kullback.runner.world import BuiltEnvironment, clock
from kullback.runner.world.loading import SandboxError, load_toolkit
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)
STAMP = "2023-03-04T09:30:00"
OPEN_BODY = ('note_id = "n" + str(len(self.db.notes) + 1)\n'
             "row = Note(note_id=note_id, title=title, opened_at=self.ctx.now())\n"
             "self.db.notes[note_id] = row\n"
             "return row.model_dump()")
SCHEMA = {"tables": ["notes"],
          "columns": [{"table": "notes", "name": "note_id", "class": "hard"},
                      {"table": "notes", "name": "title", "class": "hard"},
                      {"table": "notes", "name": "opened_at", "class": "hard"}]}
SIGS = [{"name": "read_note", "kind": "read", "unclassified": False,
         "args_fields": [{"name": "note_id", "types": ["str"]}]},
        {"name": "open_note", "kind": "write", "unclassified": False,
         "args_fields": [{"name": "title", "types": ["str"]}]}]


def _call(cid: str, name: str, args: dict, result: object) -> ToolCall:
    return ToolCall(id=cid, name=name, args=args, result=result, raw_ptr=PTR, trace_id="rec1")


def _trace(calls: list[ToolCall]) -> Trace:
    turns = [Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content="Open a note called groceries.", raw_ptr=PTR)]
    for call in calls:
        turns += [Turn(idx=len(turns), role="assistant", tool_call_ids=[call.id], raw_ptr=PTR),
                  Turn(idx=len(turns) + 1, role="tool", content=json.dumps(call.result),
                       tool_call_ids=[call.id], raw_ptr=PTR)]
    turns += [Turn(idx=len(turns), role="assistant", content="Done.", raw_ptr=PTR),
              Turn(idx=len(turns) + 1, role="user", content="Thanks.", raw_ptr=PTR)]
    return Trace(trace_id="rec1", raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _opening_trace() -> Trace:
    """Read the note the world already holds, then open a new one, stamped at the Task's time."""
    return _trace([
        _call("c1", "read_note", {"note_id": "n1"},
              {"note_id": "n1", "title": "old", "opened_at": STAMP}),
        _call("c2", "open_note", {"title": "groceries"},
              {"note_id": "n2", "title": "groceries", "opened_at": STAMP}),
    ])


def _notebook(root: Path) -> Path:
    """A workdir whose toolkit is rendered by the Builder's own compile step, and whose start
    state already holds a row carrying the Task's time, the way a world with one clock does."""
    write_env(root, with_verifier=False, with_rules=False)
    write_json(root / "schema.json", SCHEMA)
    write_json(root / "tool_sigs.json", SIGS)
    write_json(root / "db.json", {"notes": {"n1": {"note_id": "n1", "title": "old", "opened_at": STAMP}}})
    write_json(root / "bodies.json", {"read_note": "return self.db.notes[note_id].model_dump()",
                                      "open_note": OPEN_BODY})
    write_json(root / "traces" / "rec1.json", _opening_trace().model_dump(mode="json", by_alias=True))
    env_files.explode(root)
    env_files.regenerate(root)
    return root


def test_a_body_that_stamps_now_replays_the_recorded_stamp_even_where_the_start_state_holds_it(tmp_path):
    root = _notebook(tmp_path / "env")
    report = tool.replay(root, "widget_task", workdir=tmp_path / "work", trace_id="rec1")
    assert report.confirmed, report.reasons
    opened = next(call for call in report.calls if call.tool == "open_note")
    assert STAMP in opened.ours and opened.first_differing_column is None


def test_the_task_clock_is_the_moment_its_first_row_creating_write_stamped():
    assert clock.trace_clock(_opening_trace(), {"open_note"}) == STAMP


def test_a_trace_whose_writes_create_no_row_or_stamp_only_a_bare_date_sets_no_clock():
    updates_only = _trace([_call("c1", "open_note", {"title": "x", "note_id": "n1"},
                                 {"note_id": "n1", "title": "x", "opened_at": STAMP})])
    bare_date = _trace([_call("c1", "open_note", {"title": "x"},
                              {"note_id": "n2", "title": "x", "opened_at": "2023-03-04"})])
    assert clock.trace_clock(updates_only, {"open_note"}) is None
    assert clock.trace_clock(bare_date, {"open_note"}) is None
    assert clock.task_clock([None, bare_date, _opening_trace()], {"open_note"}) == STAMP


def test_start_context_reseeds_and_then_serves_the_world_clock(tmp_path):
    root = _notebook(tmp_path / "env")
    toolkit = load_toolkit(BuiltEnvironment(root).toolkit_source(), {"notes": {}})
    clock.start_context(toolkit, 7, STAMP)
    assert toolkit.open_note("groceries")["opened_at"] == STAMP
    clock.start_context(toolkit, 7, None)
    assert toolkit.open_note("again")["opened_at"] != STAMP  # no clock: the seeded draw answers


def _module(body: str) -> str:
    sigs = [ToolSig.model_validate(SIGS[1])]
    return module_source(EntitySchema.model_validate(SCHEMA), sigs, {"open_note": body})


@pytest.mark.parametrize("body, lines", [
    ("import datetime\nreturn datetime.datetime.now().isoformat()",
     [(1, "import datetime"), (2, "return datetime.datetime.now().isoformat()")]),
    ("from time import time as moment\nreturn moment()", [(1, "from time import time as moment")]),
    ("stamp = title\nwhen = stamp.today()\nreturn when", [(2, "when = stamp.today()")]),
])
def test_a_body_that_reads_the_machine_clock_is_refused_at_the_gate_naming_its_line(body, lines):
    ruling = gate_confined(_module(body))
    assert not ruling.passed
    named = [failure for failure in ruling.failures if "machine's clock" in failure]
    assert named == [f"open_note body line {line} reads the machine's clock (`{text}`); "
                     f"read the world clock as self.ctx.now()" for line, text in lines]


def test_the_runner_refuses_to_load_a_body_that_reads_the_machine_clock():
    with pytest.raises(SandboxError, match=r"open_note body line 2 reads the machine's clock"):
        load_toolkit(_module("stamp = title\nwhen = stamp.utcnow()\nreturn when"), {"notes": {}})


def test_reading_the_world_clock_is_not_a_machine_clock_read():
    assert clock.wall_clock_reads(_module(OPEN_BODY), "DomainTools") == []
