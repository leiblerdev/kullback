"""The runner's replay feeds each recorded call's tool context before its body runs (D290).

The world is an invented plant nursery: plots that hold a crop, and one tool that plants a new
plot under an id the tool context mints. Nothing here names a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from kullback.builder import env_files
from kullback.runner import tool
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, write_json
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)
TASK = "plot_task"
PLANT_BODY = ('plot_id = self.ctx.new_id("plots")\n'
              "row = Plot(plot_id=plot_id, crop=crop)\n"
              "self.db.plots[plot_id] = row\n"
              "return row.model_dump()")
SCHEMA = {"tables": ["plots"],
          "columns": [{"table": "plots", "name": "plot_id", "class": "hard"},
                      {"table": "plots", "name": "crop", "class": "hard"}],
          "id_patterns": {"plots.plot_id": r"^p\d+$"}}
SIGS = [{"name": "plant_plot", "kind": "write", "unclassified": False,
         "args_fields": [{"name": "crop", "types": ["str"]}]}]


def _trace(trace_id: str, results: list) -> Trace:
    """One Run that plants a plot per recorded result, each call answered what `results` says."""
    calls = [ToolCall(id=f"{trace_id}-c{i}", name="plant_plot", args={"crop": "kale"},
                      result=result, raw_ptr=PTR, trace_id=trace_id)
             for i, result in enumerate(results)]
    turns = [Turn(idx=0, role="assistant", content="Nursery desk, how can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content="Plant some kale for me.", raw_ptr=PTR)]
    for call in calls:
        turns += [Turn(idx=len(turns), role="assistant", tool_call_ids=[call.id], raw_ptr=PTR),
                  Turn(idx=len(turns) + 1, role="tool", content=json.dumps(call.result),
                       tool_call_ids=[call.id], raw_ptr=PTR)]
    turns += [Turn(idx=len(turns), role="assistant", content="Planted.", raw_ptr=PTR),
              Turn(idx=len(turns) + 1, role="user", content="Thanks.", raw_ptr=PTR)]
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _nursery(root: Path, traces: list[Trace]) -> Path:
    """A workdir whose toolkit the Builder's own compile step rendered, with the given Runs."""
    write_env(root, task_id=TASK, with_verifier=False, with_rules=False)
    write_json(root / "schema.json", SCHEMA)
    write_json(root / "tool_sigs.json", SIGS)
    write_json(root / "db.json", {"plots": {"p1": {"plot_id": "p1", "crop": "leek"}}})
    write_json(root / "bodies.json", {"plant_plot": PLANT_BODY})
    for trace in traces:
        write_json(root / "traces" / f"{trace.trace_id}.json",
                   trace.model_dump(mode="json", by_alias=True))
    write_json(root / "tasks" / f"{TASK}.json",
               {"id": TASK, "run_ids": [trace.trace_id for trace in traces], "intent": "plant kale"})
    env_files.explode(root)
    env_files.regenerate(root)
    return root


def test_two_runs_that_created_the_same_id_each_replay_it_from_the_recording(tmp_path):
    planted = {"plot_id": "p7", "crop": "kale"}
    root = _nursery(tmp_path / "env", [_trace("rec1", [planted]), _trace("rec2", [planted])])
    reports = tool.replay_all(root, TASK, workdir=tmp_path / "work")
    assert [report.trace_id for report in reports] == ["rec1", "rec2"]
    for report in reports:
        assert report.confirmed, report.reasons
        assert report.fidelity == 1.0
        assert [json.loads(call.ours)["plot_id"] for call in report.calls] == ["p7"]


def test_a_call_the_recording_feeds_nothing_still_replays_on_the_seeded_context(tmp_path):
    """The second call's recorded answer names no id, so its body mints one and the Run goes on."""
    root = _nursery(tmp_path / "env", [_trace("rec1", [{"plot_id": "p7", "crop": "kale"}, "planted"])])
    [report] = tool.replay_all(root, TASK, workdir=tmp_path / "work")
    fed, unfed = report.calls
    assert (fed.verdict, json.loads(fed.ours)["plot_id"]) == ("same", "p7")
    assert unfed.verdict == "differs"
    assert json.loads(unfed.ours)["plot_id"] not in ("p1", "p7")
