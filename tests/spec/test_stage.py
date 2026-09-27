"""The Spec stage writes each named Task's Spec and Verifier once, tells the bus, and survives a failing Task."""

from __future__ import annotations

import json

from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner.records import Usage
from kullback.spec import stage
from kullback.spec.schema import load_spec, spec_path
from kullback.spec.writer import runner_verifier_path, spec_verifier_path

STATE = {"items": {"A1": {"item_id": "A1", "slot": "two"}}}
POLICY = "# Store policy\n\n## Moving items\n\nMove an item only to a free slot.\n"
USER_LINE = "Please move item A1 to slot seven today."
FACT = {"text": "move item A1 to slot seven", "recording": "rec-a", "turn": 1, "stance": "volunteered"}
DEMAND = {"id": "d0", "kind": "required", "demand": "write", "tool": "update_item", "entity": "A1",
          "id_field": "item_id", "values": {"slot": "seven"}, "fact_ids": ["f1"],
          "because": "move item A1 to slot seven"}


def _trace(task_id: str) -> dict:
    turns = [{"idx": 0, "role": "assistant", "content": "How can I help?", "tool_call_ids": [],
              "raw_ptr": {"file_hash": "h"}},
             {"idx": 1, "role": "user", "content": USER_LINE, "tool_call_ids": [], "raw_ptr": {"file_hash": "h"}}]
    return {"trace_id": f"rec-{task_id}", "raw_hash": "h", "ingest_version": "v", "source": "s", "turns": turns,
            "tool_calls": [], "raw_ptr": {"file_hash": "h"}}


def _workdir(root, task_ids=("t1",)):
    """A minimal Environment on disk: the recordings the miner reads and the files the writer reads."""
    for name in ("tasks", "traces", "overlays", "env"):
        (root / name).mkdir()
    for task_id in task_ids:
        (root / "tasks" / f"{task_id}.json").write_text(json.dumps({"id": task_id, "run_ids": ["rec-a"]}))
        (root / "overlays" / f"{task_id}.json").write_text(
            json.dumps({"overlay": {"task_id": task_id, "rows": []}, "values": {}}))
    (root / "traces" / "rec-a.json").write_text(json.dumps(_trace("t1") | {"trace_id": "rec-a"}))
    (root / "db.json").write_text(json.dumps(STATE))
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"},
                                                     {"name": "list_items", "kind": "read"}]))
    (root / "env" / "policy.md").write_text(POLICY)
    return root


def _call(name: str, arguments: dict) -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id=f"{name}-1", name=name, arguments=arguments)],
                      usage=Usage(input=100, output=20))


def _one_task_replies() -> list[ModelReply]:
    """Mining: read the recording, keep one fact, stop. Writing: one demand grounded on that fact."""
    return [_call("user_turns", {"recording": "rec-a"}), _call("add_facts", {"facts": [FACT]}),
            ModelReply(content="1 fact kept.", usage=Usage(input=100, output=5)),
            ModelReply(content=json.dumps({"demands": [DEMAND]}), usage=Usage(input=300, output=50))]


def _events(root) -> list[str]:
    return [json.loads(line)["event"]["name"] for line in (root / "bus.jsonl").read_text().splitlines()]


def test_write_specs_writes_the_spec_both_verifiers_and_publishes_spec_written(tmp_path):
    root = _workdir(tmp_path)
    counts = stage.write_specs(root, ["t1"], TestModel(_one_task_replies(), name="anthropic/claude-opus-5-5"))
    assert {k: counts[k] for k in ("written", "skipped_existing", "failed")} == \
        {"written": 1, "skipped_existing": 0, "failed": 0}
    spec = load_spec(root, "t1")
    assert [check.id for check in spec.checks] == ["c0"] and spec.version == 1
    assert spec_verifier_path(root, "t1").read_text() == runner_verifier_path(root, "t1").read_text()
    assert _events(root) == ["spec.written"]
    ledger = json.loads((root / "budget.json").read_text())["stages"]["spec"]
    assert ledger["calls"] == 4 and counts["spent_usd"] > 0


def test_write_specs_skips_a_task_whose_spec_and_verifier_are_on_disk(tmp_path):
    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_one_task_replies()))
    counts = stage.write_specs(root, ["t1"], TestModel([ModelReply(content="unused")]),
                               bus=Bus(root / "bus.jsonl", agent="spec"))
    assert (counts["written"], counts["skipped_existing"]) == (0, 1)
    assert _events(root) == ["spec.written"]


def test_write_specs_records_a_failing_task_and_goes_on_to_the_next(tmp_path):
    root = _workdir(tmp_path, task_ids=("t1", "t2"))
    model = TestModel(_one_task_replies())
    counts = stage.write_specs(root, ["t0", "t1"], model)
    assert (counts["written"], counts["failed"]) == (1, 1)
    assert set(counts["failures"]) == {"t0"}
    assert not spec_path(root, "t0").exists() and spec_path(root, "t1").is_file()
