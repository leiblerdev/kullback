"""The stage's end (D439): a Task no right Run can pass ends with no Verifier, a passable one keeps it."""

from __future__ import annotations

import json

from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import Event, Run, Usage
from kullback.spec import stage
from kullback.spec.router import no_verifier_why
from kullback.spec.schema import load_spec
from kullback.spec.writer import runner_verifier_path, spec_verifier_path
from tests.spec.test_stage import DONE, _call, _items, _trace, _workdir

POLICY = ("# Store policy\n\n## Moving items\n\nMove an item only to a free slot with update_item.\n"
          "- The user must provide their member code before any move.\n")
MEMBERS = {"M1": {"member_code": "mc-77", "name": "Lee"}}
START = {"items": {"A1": {"item_id": "A1", "slot": "two"}}, "members": MEMBERS}
SIGS = [{"name": "update_item", "kind": "write",
         "args_fields": [{"name": name} for name in ("item_id", "slot", "member_code")]},
        {"name": "list_items", "kind": "read"}]
SAID = "Please move item A1 to slot seven today; my member code is mc-77."
MOVE = {"text": "move item A1 to slot seven", "recording": "rec-a", "turn": 1, "stance": "volunteered", "kind": "do"}
CODE = {"text": "my member code is mc-77", "recording": "rec-a", "turn": 1, "stance": "volunteered",
        "kind": "detail"}
CALLED = {"id": "c1", "kind": "called", "tool": "update_item", "args": {"item_id": "A1", "slot": "seven"},
          "fact_ids": ["f1"], "gate": True, "policy_line": "Move an item only to a free slot with update_item."}
MOVED = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "seven"},
         "fact_ids": ["f1"]}


def _environment(root, facts, *items):
    """A workdir whose one Task's Spec and Verifier the stage wrote from the facts and the items."""
    _workdir(root)
    (root / "env" / "policy.md").write_text(POLICY)
    (root / "db.json").write_text(json.dumps(START))
    (root / "tool_sigs.json").write_text(json.dumps(SIGS))
    trace = _trace("t1") | {"trace_id": "rec-a"}
    trace["turns"][1]["content"] = SAID
    (root / "traces" / "rec-a.json").write_text(json.dumps(trace))
    mined = [_call("user_turns", {"recording": "rec-a"}), _call("add_facts", {"facts": list(facts)}),
             ModelReply(content="kept.", usage=Usage(input=100, output=5))]
    model = TestModel([*mined, *_items(*items), DONE])  # the writer's one send-back, where it asks one
    counts = stage.write_specs(root, ["t1"], model)
    assert counts["written"] == 1, counts
    return root


def _run(run_id, end_slot):
    end = {"items": {"A1": {"item_id": "A1", "slot": end_slot}}, "members": MEMBERS}
    stop = Event(idx=0, type="stop", payload={"start_state": START, "end_state": end})
    return Run(run_id=run_id, task_id="t1", events=[stop], model="recorded", trace_id="rec-a",
               termination_reason="done")


def _references(root, *runs):
    """Each Run kept as a Reference of t1 and replayed faithfully, on disk where the build leaves them."""
    (root / "references.json").write_text(json.dumps(
        {"t1": {"references": [{"kind": "recording", "trace_id": run.run_id} for run in runs]}}))
    (root / "replays.json").write_text(json.dumps({"env": {run.run_id: {
        "task_id": "t1", "trace_id": run.run_id, "run_id": run.run_id, "fidelity": 1, "confirmed": True}
        for run in runs}}))
    (root / "runs").mkdir(exist_ok=True)
    for run in runs:
        (root / "runs" / f"{run.run_id}.jsonl").write_text(run.model_dump_json() + "\n")


def _end(root):
    return stage.end_unpassable(root, ["t1"], bus=Bus(root / "bus.jsonl", agent="spec"))


def test_a_gated_write_whose_policy_asks_a_value_no_rendered_fact_gives_ends_with_no_verifier(tmp_path):
    root = _environment(tmp_path, [MOVE], CALLED)
    counts = _end(root)
    assert (counts["set_aside_precondition_unrendered"], counts["set_aside_no_reference_holds"]) == (1, 0)
    spec = load_spec(root, "t1")
    assert spec.set_aside == "precondition_unrendered" and no_verifier_why(spec) == "precondition_unrendered"
    assert (spec.writer["ended"]["tool"], spec.writer["ended"]["field"]) == ("update_item", "member_code")
    assert not spec_verifier_path(root, "t1").exists() and not runner_verifier_path(root, "t1").exists()
    payload = json.loads((root / "bus.jsonl").read_text().splitlines()[-1])["event"]["payload"]
    assert payload["reason"] == "precondition_unrendered"


def test_a_gated_write_whose_asked_value_a_rendered_fact_gives_keeps_its_verifier(tmp_path):
    root = _environment(tmp_path, [MOVE, CODE], CALLED)
    counts = _end(root)
    assert (counts["checked"], counts["set_aside_precondition_unrendered"]) == (1, 0)
    assert load_spec(root, "t1").set_aside is None and spec_verifier_path(root, "t1").is_file()


def test_a_task_whose_references_all_fail_a_gate_ends_with_no_verifier(tmp_path):
    root = _environment(tmp_path, [MOVE], MOVED)
    _references(root, _run("ref-1", "two"), _run("ref-2", "nine"))
    counts = _end(root)
    assert (counts["set_aside_no_reference_holds"], counts["set_aside_precondition_unrendered"]) == (1, 0)
    spec = load_spec(root, "t1")
    assert spec.set_aside == "no_reference_holds" and spec.writer["ended"]["references"] == ["ref-1", "ref-2"]
    assert not spec_verifier_path(root, "t1").exists()


def test_a_task_one_of_whose_references_holds_every_gate_keeps_its_verifier(tmp_path):
    root = _environment(tmp_path, [MOVE], MOVED)
    _references(root, _run("ref-1", "two"), _run("ref-2", "seven"))
    counts = _end(root)
    assert counts["set_aside_no_reference_holds"] == 0 and counts["ended"] == {}
    assert load_spec(root, "t1").set_aside is None and spec_verifier_path(root, "t1").is_file()


def test_a_task_with_no_reference_keeps_its_verifier_whatever_its_gates(tmp_path):
    root = _environment(tmp_path, [MOVE], MOVED)
    assert _end(root)["ended"] == {} and spec_verifier_path(root, "t1").is_file()


def test_a_policy_line_binds_a_field_only_where_it_says_the_user_must_give_it():
    fields = {"member_code", "slot"}
    assert stage.given_fields("The user must provide their member code.", fields) == ["member_code"]
    assert stage.given_fields("The agent must first obtain the member code from the user.", fields) == ["member_code"]
    assert stage.given_fields("The agent may look up the member code.", fields) == []
    assert stage.given_fields("Each member has a member code and a slot.", fields) == []
    assert stage.given_fields("If the slot changes, the user needs to provide a member code.", fields) == ["member_code"]


def test_a_policy_line_speaks_of_a_tool_by_its_name_or_by_its_heading():
    policy = "# Moving items\nThe user must provide a code.\n# Returns\nSay the refund.\nUse move_item twice.\n"
    assert stage.tool_lines(policy, "move_item") == ["The user must provide a code.", "Use move_item twice."]


def test_a_tool_without_learned_effects_reads_the_columns_of_the_table_its_argument_names(tmp_path):
    from kullback.spec.writer import load_inputs

    root = _environment(tmp_path, [MOVE], MOVED)
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "move_item", "kind": "write",
                                                      "args_fields": [{"name": "item_id"}]}]))
    (root / "db.json").write_text(json.dumps({"items": {"A1": {"item_id": "A1", "owner": "M1"}}, "members": MEMBERS}))
    inputs = load_inputs(root, "t1", load_spec(root, "t1").intent)
    assert stage.tool_fields(inputs, "move_item") == {"item_id", "owner"}
