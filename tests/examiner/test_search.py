"""`search` over the Examiner's records: one phrase across the Traces, the Runs, the Intents, the
task status and the Verifiers, a count per kind and one line per match, and the paths it refuses.

The world here is invented (a left-luggage counter with lockers) and shares nothing with any
corpus: what is tested is the shape of a search, not any customer's words.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PTR
from examiner.worlds import drive
from kullback.examiner import agent as examiner_agent
from kullback.examiner import extension as ext
from kullback.examiner import tools as tools_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.runner.records import (
    Atom,
    Intent,
    IntentSpan,
    Task,
    ToolCall,
    Trace,
    Turn,
    Verifier,
    as_dict,
    write_json,
)

PHRASE = "priority handling"
FIRST, SECOND = "task_locker_1", "task_locker_2"


def _turn(idx: int, role: str, content: str) -> Turn:
    return Turn(idx=idx, role=role, content=content, raw_ptr=PTR)


def _trace(trace_id: str, said: str, size: str) -> Trace:
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                 turns=[_turn(0, "user", said), _turn(1, "assistant", f"Opening the {size} locker.")],
                 tool_calls=[ToolCall(name="find_locker", args={"size": size}, raw_ptr=PTR,
                                      result={"locker_id": "L12", "size": size}),
                             ToolCall(name="open_locker", args={"locker_id": "L12"}, raw_ptr=PTR,
                                      result={"status": "open"})])


def _run_file(folder: Path, run_id: str, said: str) -> str:
    """One Run as the Runner writes it: the Run's own fields on the first line, then one event a line."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.jsonl"
    rows = [{"run_id": run_id, "termination_reason": "success"},
            {"idx": 0, "type": "user_turn", "payload": {"content": said}},
            {"idx": 1, "type": "tool_call", "payload": {"id": "c0", "name": "find_locker", "args": {"size": "small"}}},
            {"idx": 2, "type": "tool_result", "payload": {"id": "c0", "name": "find_locker",
                                                          "result": {"locker_id": "L12", "size": "small"}}}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def _world(tmp_path: Path) -> ExaminerPlan:
    """Two Tasks whose Intents claim a phrase no Trace and no Run of either one says."""
    workdir = tmp_path / "counter"
    workdir.mkdir()
    inputs = {
        "tasks": [Task(id=FIRST, run_ids=["trace_a"]), Task(id=SECOND, run_ids=["trace_b"])],
        "traces": [_trace("trace_a", "I need the small locker by the west door.", "small"),
                   _trace("trace_b", "The large locker, please.", "large")],
        "intents": {
            FIRST: as_dict(Intent(task_id=FIRST, text=f"open the small locker with {PHRASE}", grounded=False,
                                  ungrounded_phrases=[PHRASE], run_coverage={"small locker": ["trace_a"]},
                                  spans=[IntentSpan(phrase="small locker", trace_id="trace_a",
                                                    source="user_utterance",
                                                    text="I need the small locker by the west door.")])),
            SECOND: as_dict(Intent(task_id=SECOND, text=f"open the large locker with {PHRASE}", grounded=False,
                                   ungrounded_phrases=[PHRASE])),
        },
        "replays": {
            FIRST: {"trace_a": {"trace_id": "trace_a", "run_id": "replay_a", "confirmed": True,
                                "path": _run_file(workdir / "runs" / FIRST, "replay_a",
                                                  "I need the small locker by the west door.")}},
            SECOND: {"trace_b": {"trace_id": "trace_b", "run_id": "replay_b", "confirmed": True,
                                 "path": _run_file(workdir / "runs" / SECOND, "replay_b",
                                                   "The large locker, please.")}},
        },
        "rerolls": {},
    }
    write_json(workdir / "task_status.json",
               {FIRST: {"reference_confirmed": True, "verifier_passed": False,
                        "reason": f"the Intent says {PHRASE} and no Run does"},
                SECOND: {"reference_confirmed": False, "verifier_passed": False, "reason": "no Reference"}})
    write_json(workdir / "verifiers" / f"{FIRST}.json",
               as_dict(Verifier(task_id=FIRST, atoms=[Atom(id="w0", kind="required",
                                                           description="the small locker is open",
                                                           target={"tool": "open_locker", "locker_id": "L12"})])))
    return ExaminerPlan(workdir=workdir, inputs=inputs)


def _search(plan: ExaminerPlan, **arguments):
    return drive(examiner_agent.examiner_harness(plan), "search", arguments)


def test_a_search_counts_every_match_per_kind_and_names_the_record_the_task_and_the_field(tmp_path):
    result = _search(_world(tmp_path), text=PHRASE)
    assert result.is_error is False
    details = result.details
    # the phrase is in both Intents and in the reason of one status row; no Trace and no Run says it
    assert details["counts"] == {"trace": 0, "run": 0, "intent": 4, "task_status": 1, "verifier": 0}
    assert details["matches"] == 5 and details["shown"] == 5
    lines = details["text"].splitlines()
    assert lines[0] == (f"search '{PHRASE}' over trace, run, intent, task_status, verifier: 5 matches "
                        "(trace 0, run 0, intent 4, task_status 1, verifier 0), showing 5")
    assert f"intent {FIRST}: text: open the small locker with {PHRASE}" in lines
    assert f"intent {FIRST}: ungrounded_phrase: {PHRASE}" in lines
    assert f"task_status {FIRST}: reason: the Intent says {PHRASE} and no Run does" in lines
    assert result.content == details["text"], "the rendered result is the search itself"


def test_a_search_reads_the_traces_and_the_runs_and_says_which_turn_or_event_matched(tmp_path):
    plan = _world(tmp_path)
    details = _search(plan, text="small locker").details
    assert details["counts"]["trace"] == 2 and details["counts"]["run"] == 1
    lines = details["text"].splitlines()
    assert (f"trace trace_a (task {FIRST}): turn 0 user: I need the small locker by the west door."
            in lines)
    assert any(line.startswith(f"run replay_a (task {FIRST}): event 0 user_turn: ") and "small locker" in line
               for line in lines)
    # a tool call and its result are two events, so both Runs match twice, and the Run's own
    # fields are a line of the file like any other
    events = _search(plan, text="find_locker", kinds=["run"]).details
    assert events["counts"] == {"run": 4}
    assert all(" event " in line for line in events["text"].splitlines()[1:])
    assert _search(plan, text="replay_a", kinds=["run"]).details["text"].splitlines()[1].endswith(
        f"run fields: {json.dumps({'run_id': 'replay_a', 'termination_reason': 'success'})}")


def test_a_search_narrows_by_kind_by_task_and_by_tool_and_reads_a_regular_expression_when_asked(tmp_path):
    plan = _world(tmp_path)
    assert _search(plan, text=PHRASE, kinds=["intent"]).details["counts"] == {"intent": 4}
    assert _search(plan, text=PHRASE, task_id=SECOND).details["counts"]["intent"] == 2
    # only a Trace's tool call and a Run's event belong to a tool
    by_tool = _search(plan, text="locker_id", tool="open_locker").details
    assert by_tool["counts"] == {"trace": 2, "run": 0, "intent": 0, "task_status": 0, "verifier": 0}
    assert by_tool["text"].splitlines()[1].startswith(
        f"trace trace_a (task {FIRST}): tool call open_locker args: ")
    # a substring is taken literally; the same text is a pattern only when the call says so
    assert _search(plan, text="small.locker", kinds=["trace"]).details["matches"] == 0
    assert _search(plan, text="small.locker", kinds=["trace"], regex=True).details["counts"]["trace"] == 2
    assert _search(plan, text="[", kinds=["trace"]).details["matches"] == 0
    broken = _search(plan, text="[", kinds=["trace"], regex=True)
    assert broken.is_error and "is not a regular expression" in broken.content


def test_a_search_counts_every_match_but_answers_at_most_the_limit(tmp_path):
    details = _search(_world(tmp_path), text="locker", limit=3).details
    assert details["matches"] > 3 and details["shown"] == 3
    assert details["text"].splitlines()[0].endswith(f"{details['matches']} matches "
                                                    "(trace 8, run 6, intent 3, task_status 0, verifier 1), "
                                                    "showing 3")
    assert len(details["text"].splitlines()) == 4


def test_a_search_that_names_a_path_the_examiner_never_reads_is_blocked_before_it_runs(tmp_path):
    plan = _world(tmp_path)
    harness = examiner_agent.examiner_harness(plan)
    for value in ("bodies.json", "build/env/tables.json", "kullback/runner/loop.py"):
        result = drive(harness, "search", {"text": value})
        assert result.is_error and "blocked by examiner_reads_only_its_surface" in result.content, value
        assert value in result.content


def test_a_match_is_cut_to_about_one_line_with_the_match_inside_what_is_left(tmp_path):
    workdir = tmp_path / "wide"
    workdir.mkdir()
    padding = "z" * 400
    inputs = {"tasks": [Task(id=FIRST, run_ids=["trace_a"])],
              "traces": [Trace(trace_id="trace_a", raw_hash="h", ingest_version="1", source="test", raw_ptr=PTR,
                               turns=[_turn(0, "user", f"{padding} {PHRASE} {padding}")])]}
    plan = ExaminerPlan(workdir=workdir, inputs=inputs)
    line = _search(plan, text=PHRASE).details["text"].splitlines()[1]
    excerpt = line.split(": ", 2)[2]
    assert PHRASE in excerpt and len(excerpt) == tools_mod.SEARCH_LINE + len("...") * 2
    assert excerpt.startswith("...") and excerpt.endswith("...")


def test_the_search_tool_is_registered_beside_read_and_its_schema_names_the_kinds(tmp_path):
    harness = examiner_agent.examiner_harness(_world(tmp_path))
    schema = harness.registry.get("search").schema()["input_schema"]
    assert set(schema["properties"]) == {"text", "regex", "kinds", "tool", "task_id", "limit"}
    assert schema["required"] == ["text"]
    assert set(tools_mod.SEARCH_KINDS) == set(schema["properties"]["kinds"]["items"]["enum"])
    # the prompt teaches the tool and the finding it grounds
    assert "search(text=" in ext.TOOLS and "search(text=" in ext.EXAMPLES
