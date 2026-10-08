"""The Spec stage writes each named Task's Spec and Verifier once, tells the bus, and survives a failing Task."""

from __future__ import annotations

import json

from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner import budget
from kullback.runner.records import Usage
from kullback.spec import stage
from kullback.spec.schema import FactSource, IntentFact, SpecIntent, intent_text, load_spec, spec_path
from kullback.spec.writer import runner_verifier_path, spec_verifier_path

STATE = {"items": {"A1": {"item_id": "A1", "slot": "two"}}}
POLICY = "# Store policy\n\n## Moving items\n\nMove an item only to a free slot.\n"
USER_LINE = "Please move item A1 to slot seven today."
FACT = {"text": "move item A1 to slot seven", "recording": "rec-a", "turn": 1, "stance": "volunteered"}
ITEM = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "seven"},
        "fact_ids": ["f1"]}


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


DONE = ModelReply(content="done", usage=Usage(input=300, output=5))


def _items(*items: dict) -> list[ModelReply]:
    """One writer session: add the items, then say done."""
    return [_call("add_items", {"items": list(items)}), DONE]


def _one_task_replies() -> list[ModelReply]:
    """Mining: read the recording, keep one fact, stop. Writing: one state item resting on that fact."""
    return [_call("user_turns", {"recording": "rec-a"}), _call("add_facts", {"facts": [FACT]}),
            ModelReply(content="1 fact kept.", usage=Usage(input=100, output=5))] + _items(ITEM)


EMPTY = DONE


def _events(root) -> list[str]:
    return [json.loads(line)["event"]["name"] for line in (root / "bus.jsonl").read_text().splitlines()]


def test_write_specs_writes_the_spec_both_verifiers_and_publishes_spec_written(tmp_path, monkeypatch):
    monkeypatch.setitem(budget.PRICES, "test/model",
                        {"input": 10.0, "output": 10.0, "cache_read": 0.0, "cache_write": 0.0})
    root = _workdir(tmp_path)
    counts = stage.write_specs(root, ["t1"], TestModel(_one_task_replies(), name="test/model"))
    assert {k: counts[k] for k in ("written", "skipped_existing", "failed")} == \
        {"written": 1, "skipped_existing": 0, "failed": 0}
    spec = load_spec(root, "t1")
    assert [check.id for check in spec.checks] == ["s1"] and spec.version == 1
    assert {k: spec.writer[k] for k in ("offered", "kept", "attempts", "empty_first", "sent_back")} == \
        {"offered": 1, "kept": 1, "attempts": 1, "empty_first": False, "sent_back": 0}
    assert spec.writer["usd"] > 0 and spec.writer["read_rounds"] == 1 and "first_attempt" not in spec.writer
    assert spec_verifier_path(root, "t1").read_text() == runner_verifier_path(root, "t1").read_text()
    assert _events(root) == ["spec.written"]
    ledger = json.loads((root / "budget.json").read_text())["stages"]["spec"]
    assert ledger["calls"] == 5 and counts["spent_usd"] > 0


def test_write_specs_skips_a_task_whose_spec_and_verifier_are_on_disk(tmp_path):
    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_one_task_replies()))
    counts = stage.write_specs(root, ["t1"], TestModel([ModelReply(content="unused")]),
                               bus=Bus(root / "bus.jsonl", agent="spec"))
    assert (counts["written"], counts["skipped_existing"]) == (0, 1)
    assert _events(root) == ["spec.written"]


def test_a_spec_of_the_older_demand_grammar_is_rewritten_as_items_with_its_sanity_item(tmp_path):
    from kullback.runner.records import Verifier
    from kullback.spec.schema import Check, save_spec

    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_one_task_replies()))
    item_spec = load_spec(root, "t1")
    older = Check(id="c1", kind="required", demand={"demand": "write", "tool": "update_item"},
                  because="move item A1 to slot seven", tier="critical", fact_ids=["f1"])
    save_spec(root, item_spec.model_copy(update={"checks": [older]}))
    assert stage.older_grammar(load_spec(root, "t1")) and not stage.written_already(root, "t1")
    counts = stage.write_specs(root, ["t1"], TestModel(_one_task_replies()))
    assert (counts["written"], counts["rewritten_older"]) == (1, 1)
    verifier = Verifier.model_validate_json(spec_verifier_path(root, "t1").read_text())
    assert [check.id for check in load_spec(root, "t1").checks] == ["s1"]
    assert "sanity" in {atom.id for atom in verifier.atoms}


def test_write_specs_records_a_failing_task_and_goes_on_to_the_next(tmp_path):
    root = _workdir(tmp_path, task_ids=("t1", "t2"))
    model = TestModel(_one_task_replies())
    counts = stage.write_specs(root, ["t0", "t1"], model)
    assert (counts["written"], counts["failed"]) == (1, 1)
    assert set(counts["failures"]) == {"t0"}
    assert not spec_path(root, "t0").exists() and spec_path(root, "t1").is_file()


def test_an_empty_first_spec_is_written_once_more_and_the_second_stands(tmp_path):
    root = _workdir(tmp_path)
    replies = _one_task_replies()
    counts = stage.write_specs(root, ["t1"], TestModel(replies[:3] + [EMPTY] + replies[3:]))
    assert {k: counts[k] for k in ("written", "empty_seen", "retried", "set_aside_empty")} == \
        {"written": 1, "empty_seen": 1, "retried": 1, "set_aside_empty": 0}
    spec = load_spec(root, "t1")
    assert spec.set_aside is None and (spec.writer["attempts"], spec.writer["empty_first"]) == (2, True)
    first = spec.writer["first_attempt"]
    assert (first["offered"], first["kept"]) == (0, 0) and spec.writer["kept"] == 1
    assert _events(root) == ["spec.written"]


def test_an_empty_spec_twice_sets_the_task_aside_with_no_verifier_and_a_resume_skips_it(tmp_path):
    root = _workdir(tmp_path)
    counts = stage.write_specs(root, ["t1"], TestModel(_one_task_replies()[:3] + [EMPTY, EMPTY]))
    assert {k: counts[k] for k in ("written", "empty_seen", "retried", "set_aside_empty")} == \
        {"written": 0, "empty_seen": 1, "retried": 1, "set_aside_empty": 1}
    spec = load_spec(root, "t1")
    assert spec.set_aside == "empty_spec" and spec.writer["empty"] is True
    assert not spec_verifier_path(root, "t1").exists() and not runner_verifier_path(root, "t1").exists()
    (line,) = (root / "bus.jsonl").read_text().splitlines()
    assert json.loads(line)["event"]["payload"] == {"task_id": "t1", "reason": "empty_spec", "rounds": 0}
    again = stage.write_specs(root, ["t1"], TestModel([ModelReply(content="unused")]))
    assert again["skipped_existing"] == 1


def test_write_from_intent_writes_the_spec_from_a_given_intent_without_mining(tmp_path):
    root = _workdir(tmp_path)
    facts = [IntentFact(id="f1", text=FACT["text"], stance="volunteered", source=FactSource(recording="rec-a", turn=1))]
    intent = SpecIntent(task_id="t1", facts=facts, text=intent_text(facts))
    model = TestModel(_items(ITEM))
    spent = stage.write_from_intent(root, "t1", intent, model, 1.0, Bus(root / "bus.jsonl", agent="spec"))
    assert len(model.calls) == 2 and spent == 0.0
    assert load_spec(root, "t1").intent == intent and spec_verifier_path(root, "t1").is_file()


def test_a_spec_no_run_can_pass_goes_back_once_and_still_unpassable_is_set_aside_with_no_verifier(tmp_path):
    root = _workdir(tmp_path)
    clash = _items({"id": "e1", "kind": "called", "tool": "update_item", "fact_ids": ["f1"]},
                   {"id": "e2", "kind": "not_called", "tool": "update_item", "fact_ids": ["f1"]})
    model = TestModel(_one_task_replies()[:3] + clash + [DONE])
    counts = stage.write_specs(root, ["t1"], model)
    assert (counts["written"], counts["set_aside_unsatisfiable"], len(model.calls)) == (0, 1, 6)
    spec = load_spec(root, "t1")
    assert spec.set_aside == "unsatisfiable_spec" and spec.writer["unsatisfiable"] == ["e2"]
    assert not spec_verifier_path(root, "t1").exists() and not runner_verifier_path(root, "t1").exists()
    (line,) = (root / "bus.jsonl").read_text().splitlines()
    assert json.loads(line)["event"]["payload"] == {"task_id": "t1", "reason": "unsatisfiable_spec", "rounds": 0}
