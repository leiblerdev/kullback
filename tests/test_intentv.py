"""Offline tests for the Intent-first Verifier experiment script.

No live calls, no workdir writes: pure helpers plus the existing scorer on a
synthetic run. What is asserted is the contract the experiment rests on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))

from intentv import (  # noqa: E402
    compile_verifier,
    parse_demands,
    task_class,
    valid_because,
)

INTENT = "Please change my default address to my daughter's place in Chicago."
POLICY = ["disclosure:zip:volunteered"]


def test_task_class_names_right_wrong_mixed_unknown():
    assert task_class(["a"], {"a": 1}) == "right"
    assert task_class(["a", "b"], {"a": 0, "b": 0}) == "wrong"
    assert task_class(["a", "b"], {"a": 1, "b": 0}) == "mixed"
    assert task_class(["a"], {}) == "unknown"


def test_because_accepts_an_intent_quote():
    demand = {"because": "change my default address to my daughter's place"}
    assert valid_because(demand, INTENT, POLICY) is True


def test_because_accepts_a_policy_rule_name():
    assert valid_because({"because": "policy disclosure:zip allows this"}, INTENT, POLICY) is True


def test_because_rejects_empty_and_unrelated():
    assert valid_because({"because": ""}, INTENT, POLICY) is False
    assert valid_because({"because": "it seems sensible"}, INTENT, POLICY) is False
    assert valid_because({}, INTENT, POLICY) is False


def test_parse_demands_reads_the_object_out_of_prose():
    demands, error = parse_demands('here is the result {"demands": [{"id": "d0"}]} done')
    assert error == "" and demands == [{"id": "d0"}]


def test_parse_demands_names_prose_without_json():
    demands, error = parse_demands("no json here")
    assert demands == [] and error


def _fn():
    from kullback.runner.canon import canon_value

    return canon_value


def test_compile_write_demand_becomes_scored_atoms():
    demands = [{"id": "d0", "kind": "required", "demand": "write",
                "tool": "modify_user_address", "id_field": "user_id", "entity": "u1",
                "values": {"city": "springfield"},
                "because": "change my default address to my daughter's place"}]
    verifier, invalid = compile_verifier(demands, INTENT, POLICY, {"modify_user_address"}, _fn())
    assert invalid == 0
    kinds = sorted((a.kind, a.target.get("kind")) for a in verifier.atoms)
    assert kinds == [("required", "write"), ("required", "write_value")]
    assert all(a.predicate_src for a in verifier.atoms)


def test_compile_rejects_unknown_tool_and_bad_kind():
    demands = [{"id": "d0", "kind": "required", "demand": "write",
                "tool": "invented_tool", "id_field": "user_id", "entity": "u1",
                "values": {}, "because": "change my default address to my daughter's place"},
               {"id": "d1", "kind": "wish", "demand": "write",
                "because": "change my default address to my daughter's place"}]
    verifier, invalid = compile_verifier(demands, INTENT, POLICY, {"modify_user_address"}, _fn())
    assert invalid == 2 and verifier.atoms == []


def _run_dict():
    return {
        "run_id": "r1", "task_id": "t1",
        "events": [
            {"idx": 0, "type": "tool_call",
             "payload": {"id": "c1", "name": "modify_user_address",
                         "args": {"user_id": "u1", "city": "springfield"}}},
            {"idx": 1, "type": "tool_result",
             "payload": {"id": "c1", "result": {"user_id": "u1"}}},
            {"idx": 2, "type": "model_call",
             "payload": {"reply": {"content": "done for user u1", "tool_calls": []}}},
            {"idx": 3, "type": "stop",
             "payload": {"start_state": {}, "end_state": {},
                         "termination_reason": "user_stop"}},
        ],
    }


def test_existing_scorer_passes_a_run_meeting_intent_demands():
    from kullback.runner.verdict import verdict

    demands = [{"id": "d0", "kind": "required", "demand": "write",
                "tool": "modify_user_address", "id_field": "user_id", "entity": "u1",
                "values": {"user_id": "u1"},
                "because": "change my default address to my daughter's place"},
               {"id": "d1", "kind": "communicate", "demand": "say", "text": "u1",
                "because": "change my default address to my daughter's place"}]
    verifier, invalid = compile_verifier(demands, INTENT, POLICY, {"modify_user_address"}, _fn())
    assert invalid == 0
    result = verdict(_run_dict(), verifier, write_tools={"modify_user_address"})
    assert result.passed is True


def test_existing_scorer_fails_a_run_missing_a_demanded_write():
    from kullback.runner.verdict import verdict

    demands = [{"id": "d0", "kind": "required", "demand": "write",
                "tool": "modify_user_address", "id_field": "user_id", "entity": "u1",
                "values": {"user_id": "u1"},
                "because": "change my default address to my daughter's place"}]
    verifier, _ = compile_verifier(demands, INTENT, POLICY, {"modify_user_address"}, _fn())
    run = _run_dict()
    run["events"][0]["payload"]["name"] = "get_user_details"
    result = verdict(run, verifier, write_tools={"modify_user_address"})
    assert result.passed is False

def test_steering_leak_counts_values_past_the_opening_turn(tmp_path):
    from intentv import steering_leak

    (tmp_path / "exam").mkdir()
    (tmp_path / "runs" / "t1").mkdir(parents=True)
    (tmp_path / "exam" / "references.json").write_text(
        '{"t1": {"references": [{"run_id": "r1"}]}}', encoding="utf-8")
    events = [
        {"idx": 0, "type": "user_turn", "payload": {"text": "Where is my order"}},
        {"idx": 1, "type": "tool_result", "payload": {"result": {"tacking": "1Z999"}}},
        {"idx": 2, "type": "user_turn", "payload": {"text": "Thanks, tracking 1Z999 helps"}},
    ]
    (tmp_path / "runs" / "t1" / "r1.jsonl").write_text(
        "\n".join(['{"run_id": "r1", "task_id": "t1"}']
                   + [__import__("json").dumps(e) for e in events]), encoding="utf-8")
    demands = [{"because": "Second user turn. tracking 1Z999 helps. Noted."}]
    leak = steering_leak(tmp_path, "t1", demands, None)
    assert leak["steered_values"] >= 1
    assert leak["steered_demand_becauses"] == 1
    assert leak["user_turns"] == 2

def test_because_accepts_a_policy_section_name():
    from intentv import valid_because

    demand = {"because": "per section:cancel pending order, the run may cancel"}
def test_policy_sections_names_recorded_headers(tmp_path):
    import json
    from intentv import corpus_policy, policy_sections

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "r.json").write_text(
        json.dumps({"simulations": [{"policy": "# Cancel pending order\nbody\n## Modify items\nmore"}]}),
        encoding="utf-8")
    assert corpus_policy(tmp_path) != ""
    assert policy_sections(tmp_path) == ["section:cancel pending order", "section:modify items"]

def test_writer_prompts_differ_only_by_revision():
    import json
    from intentv import SYSTEM_PROMPT, SYSTEM_PROMPT_R0, writer_messages

    def body(**kwargs):
        content = writer_messages("intent", ["disclosure:zip:volunteered"],
                                  [{"name": "t"}], **kwargs)[0]["content"]
        return content, json.loads(content[content.index('{"intent"'):])

    r0_content, r0 = body(prompt="r0")
    r1_content, r1 = body(policy_text="policy words")
    assert "policy_text" not in r0 and r0["policy"] == ["disclosure:zip:volunteered"]
    assert r1["policy_text"] == "policy words"
    assert r0_content.startswith(SYSTEM_PROMPT_R0)
    assert r1_content.startswith(SYSTEM_PROMPT)
    assert SYSTEM_PROMPT_R0 != SYSTEM_PROMPT
