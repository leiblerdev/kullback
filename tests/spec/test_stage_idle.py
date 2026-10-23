"""D441: a Verifier every code gate of which an idle Run holds ships only with a gated judge or said item."""

from __future__ import annotations

import json

from kullback.ai.provider import ModelReply
from kullback.runner.records import Usage
from kullback.spec import stage
from kullback.spec.router import no_verifier_why
from kullback.spec.schema import load_spec
from kullback.spec.writer import runner_verifier_path, spec_verifier_path
from tests.spec.test_stage import DONE, _call
from tests.spec.test_stage_end import CODE, MOVE, MOVED, _end, written

KEPT = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "two"},
        "fact_ids": ["f1"]}
DECLINED = {"id": "j1", "kind": "judge", "question": "Did the agent decline to move the item?",
            "policy_line": "Move an item only to a free slot with update_item.",
            "anchor": {"answer": "No: move an item only to a free slot with update_item."}}
CANNOT = ModelReply(content="rebut: cannot verify: the right end writes nothing", usage=Usage(input=100, output=5))


def test_a_verifier_whose_idle_run_fails_a_code_gate_is_neither_sent_back_nor_ended(tmp_path):
    counts, model = written(tmp_path, [MOVE, CODE], [MOVED])
    assert counts[stage.IDLE_SENT_BACK] == 0 and len(model.calls) == 6
    assert stage.IDLE_SENT_BACK not in load_spec(tmp_path, "t1").writer
    assert _end(tmp_path)["ended"] == {} and spec_verifier_path(tmp_path, "t1").is_file()


def test_a_no_write_verifier_with_a_gated_judge_item_keeps_its_verifier(tmp_path):
    counts, model = written(tmp_path, [MOVE, CODE], [KEPT, DECLINED])
    assert counts[stage.IDLE_SENT_BACK] == 0 and len(model.calls) == 6
    ended = _end(tmp_path)
    assert ended["set_aside_idle_run_passes"] == 0 and spec_verifier_path(tmp_path, "t1").is_file()


def test_a_no_write_verifier_of_code_gates_only_goes_back_once_and_unchanged_ends_with_no_verifier(tmp_path):
    counts, model = written(tmp_path, [MOVE, CODE], [KEPT], CANNOT)
    assert counts[stage.IDLE_SENT_BACK] == 1 and len(model.calls) == 7, "one send back, through the repair session"
    ask = model.calls[-1]["messages"][-1]["content"]
    assert "makes no call and says nothing holds every code gate" in ask and "s1" in ask
    spec = load_spec(tmp_path, "t1")
    assert "sanity" in spec.writer[stage.IDLE_SENT_BACK] and spec.set_aside is None
    ended = _end(tmp_path)
    assert (ended["set_aside_idle_run_passes"], ended["ended"]) == (1, {"t1": "idle_run_passes"})
    spec = load_spec(tmp_path, "t1")
    assert spec.set_aside == "idle_run_passes" and no_verifier_why(spec) == "idle_run_passes"
    assert "sanity" in spec.writer["ended"]["held"]
    assert not spec_verifier_path(tmp_path, "t1").exists() and not runner_verifier_path(tmp_path, "t1").exists()
    payload = json.loads((tmp_path / "bus.jsonl").read_text().splitlines()[-1])["event"]["payload"]
    assert payload["reason"] == "idle_run_passes"


def test_a_send_back_answered_with_a_gated_judge_item_keeps_the_verifier(tmp_path):
    counts, _ = written(tmp_path, [MOVE, CODE], [KEPT], _call("add_items", {"items": [DECLINED]}), DONE)
    assert counts[stage.IDLE_SENT_BACK] == 1
    assert _end(tmp_path)["set_aside_idle_run_passes"] == 0 and spec_verifier_path(tmp_path, "t1").is_file()


def test_a_gated_said_item_naming_a_value_counts_as_told_and_one_naming_none_does_not():
    from kullback.runner.records import Atom

    said = Atom(id="t1", kind="allowed", description="d", target={"kind": "said", "values": ["seven"]}, gate=True)
    assert stage._told_gate(said)
    assert not stage._told_gate(said.model_copy(update={"target": {"kind": "said", "values": [], "not_values": ["x"]}}))
    assert not stage._told_gate(said.model_copy(update={"gate": False}))
