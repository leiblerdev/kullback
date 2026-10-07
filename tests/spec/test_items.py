"""Items: code keeps only values it finds, compiles them to end states and gates, and scores each on a Run.

The world, the policy and the facts are invented: two items in slots, one user asking to move one.
"""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import match_context
from kullback.runner.items import score_items
from kullback.runner.records import Event, Run, Verifier
from kullback.spec import items as I
from kullback.spec.schema import FactSource, IntentFact


def _matches(end_states, context):
    """Any one end state matching passes; else an unsettled one is None; else False."""
    oks = [match_context(state, context).ok for state in end_states]
    return True if True in oks else (None if None in oks else False)


STATE = {"items": {"A1": {"item_id": "A1", "slot": "two", "label": "red"},
                   "B2": {"item_id": "B2", "slot": "four", "label": "red"}},
         "notes": {}}
POLICY = "# Store policy\n\n## Moving items\n\nMove an item only to a free slot, after the user says yes.\n"
FACTS = [IntentFact(id="f1", text="Please move item A1 to slot seven.", stance="volunteered",
                    source=FactSource(recording="rec1", turn=1), witnesses=["rec2"]),
         IntentFact(id="f2", text="Maybe leave a note.", stance="accepted", source=FactSource(recording="rec1", turn=3))]
TOOLS = [{"name": "update_item", "kind": "write", "args": ["item_id", "slot"]},
         {"name": "add_note", "kind": "write", "args": ["text"]},
         {"name": "list_items", "kind": "read", "args": []}]
KNOWN = {fact.id: fact for fact in FACTS}
MOVE = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "seven"},
        "fact_ids": ["f1"]}


def world() -> I.World:
    return I.World(STATE, POLICY, FACTS, TOOLS)


def unwitnessed(fact) -> bool:
    return fact.stance == "accepted" and not fact.witnesses


def keep(item: dict) -> tuple[dict, str]:
    return world().check(item, KNOWN, unwitnessed)


def test_a_state_item_keeps_its_resolved_row_its_gate_and_its_provenance():
    kept, why = keep(MOVE)
    assert why == "" and kept["row"] == "A1" and kept["gate"] is True and kept["weight"] == 1.0
    assert kept["fact_ids"] == ["f1"] and kept["policy_line"] is None


def test_a_value_found_nowhere_is_refused_and_the_reason_says_find_it():
    _, why = keep(dict(MOVE, expect={"slot": "nine"}))
    assert "not in the world, the policy or a user turn" in why


def test_values_are_sourced_from_a_user_turn_then_the_policy_then_the_world():
    w = world()
    assert w.source("seven").kind == "user_turn" and w.source("seven").ptr["fact"] == "f1"
    assert w.source("free slot").kind == "policy"
    assert w.source("four").kind == "world" and w.source("four").ptr == {"table": "items", "row": "B2"}
    assert w.source("7") is None


def test_a_find_must_name_exactly_one_row_and_a_field_must_exist():
    assert "matches 2 rows" in keep(dict(MOVE, find={"label": "red"}))[1]
    assert "matches 0 rows" in keep(dict(MOVE, find={"item_id": "Z9"}))[1]
    assert "has no field 'colour'" in keep(dict(MOVE, expect={"colour": "red"}))[1]


def test_provenance_is_required_a_policy_line_must_be_verbatim_and_unwitnessed_facts_alone_are_refused():
    assert "fact_ids" in keep(dict(MOVE, fact_ids=[]))[1]
    assert "verbatim" in keep(dict(MOVE, fact_ids=[], policy_line="move anything anywhere at all"))[1]
    assert "unwitnessed" in keep(dict(MOVE, fact_ids=["f2"]))[1]
    kept, why = keep(dict(MOVE, fact_ids=[], policy_line="Move an item only to a free slot"))
    assert why == "" and kept["policy_line"] == "Move an item only to a free slot"


def test_value_sets_event_text_and_judge_items_are_checked_by_their_own_rules():
    assert keep(dict(MOVE, expect={"slot": {"one_of": ["seven", "four"]}}))[1] == ""
    assert "value set" in keep(dict(MOVE, expect={"slot": {"any": ["seven"]}}))[1]
    assert keep({"id": "e1", "kind": "before", "first": "confirm_turn", "then": "update_item", "fact_ids": ["f1"]})[1] == ""
    assert "no tool" in keep({"id": "e2", "kind": "called", "tool": "teleport", "fact_ids": ["f1"]})[1]
    assert "takes item_id, slot" in keep({"id": "e3", "kind": "called", "tool": "update_item",
                                          "args": {"colour": "red"}, "fact_ids": ["f1"]})[1]
    assert "sentence" in keep({"id": "t1", "kind": "said", "values": ["move item A1 to slot seven now please"],
                               "fact_ids": ["f1"]})[1]
    assert "yes or no" in keep({"id": "j0", "kind": "judge", "question": "Say where A1 is.",
                                "anchor": {"answer": "x"}, "fact_ids": ["f1"]})[1]


def test_a_judge_anchor_cites_the_one_row_it_was_found_in_and_its_answer_states_that_rows_fields():
    judge = {"id": "j1", "kind": "judge", "question": "Did the agent say where B2 is?", "fact_ids": ["f1"],
             "anchor": {"answer": "B2 is in slot four", "row": {"table": "items", "find": {"item_id": "B2"},
                                                                 "fields": ["slot"]}}}
    kept, why = keep(judge)
    assert why == "" and kept["gate"] is False and kept["evidence"] == ["final_answer"]
    assert kept["anchor"]["row"] == {"table": "items", "find": {"item_id": "B2"}, "key": "B2",
                                     "fields": {"slot": "four"}}
    wrong_row = dict(judge, anchor=dict(judge["anchor"], row=dict(judge["anchor"]["row"], find={"item_id": "A1"})))
    assert "does not state items row A1 slot" in keep(wrong_row)[1]
    assert "anchor.row is" in keep(dict(judge, anchor={"answer": "slot four"}))[1]
    assert "matches 2 rows" in keep(dict(judge, anchor=dict(judge["anchor"], row={
        "table": "items", "find": {"label": "red"}, "fields": ["slot"]})))[1]
    refusal = {"id": "j2", "kind": "judge", "question": "Did the agent decline?",
               "policy_line": "Move an item only to a free slot",
               "anchor": {"answer": "No: move an item only to a free slot."}}
    assert keep(refusal)[1] == ""
    for ref in ("after_call:list_items", "assistant:2", "assistant_turns"):
        assert keep(dict(judge, evidence=[ref]))[1] == ""
    assert "evidence" in keep(dict(judge, evidence=["the third turn"]))[1]
    assert "evidence" in keep(dict(judge, evidence=["after_call:teleport"]))[1]


def _kept(*items: dict) -> list[dict]:
    out = []
    for item in items:
        kept, why = keep(item)
        assert why == "", why
        out.append(kept)
    return out


def _run(end: dict, calls=(), said="Done.", user=("Yes please.",)) -> Run:
    events = [Event(idx=0, type="user_turn", payload={"content": "Please move item A1 to slot seven."})]
    for text in user:
        events.append(Event(idx=len(events), type="user_turn", payload={"content": text}))
    for n, (name, args) in enumerate(calls):
        events.append(Event(idx=len(events), type="tool_call", payload={"id": f"c{n}", "name": name, "args": args,
                                                                       "kind": "write"}))
        events.append(Event(idx=len(events), type="tool_result", payload={"id": f"c{n}", "result": {}}))
    events.append(Event(idx=len(events), type="model_call", payload={"reply": {"content": said}}))
    events.append(Event(idx=len(events), type="stop", payload={"start_state": STATE, "end_state": end}))
    return Run(run_id="r", task_id="t1", events=events)


def _verifier(items: list[dict]) -> Verifier:
    states = I.end_states_of(items, world())
    conduct, forbidden = I.event_gates(items, KNOWN, {"update_item", "add_note"})
    return Verifier(task_id="t1", atoms=[I.atom_of(i) for i in items] + [I.sanity_atom(states)],
                    expected=states, conduct=conduct, forbidden=forbidden)


def _moved(slot="seven", **extra) -> dict:
    end = {"items": {**STATE["items"], "A1": dict(STATE["items"]["A1"], slot=slot)}, "notes": {}}
    for table, rows in extra.items():
        end[table] = {**end[table], **rows}
    return end


def test_the_sanity_item_fails_a_stray_write_and_holds_on_the_declared_row():
    verifier = _verifier(_kept(MOVE))
    right = AtomContext(_run(_moved(), [("update_item", {"item_id": "A1", "slot": "seven"})]))
    stray = AtomContext(_run(_moved(items={"B2": dict(STATE["items"]["B2"], label="blue")})))
    assert _matches(verifier.expected, right) and score_items(verifier, right) == {"s1": True, "sanity": True}
    assert _matches(verifier.expected, stray) is False
    assert score_items(verifier, stray) == {"s1": True, "sanity": False}
    nothing = AtomContext(_run(STATE))
    assert score_items(verifier, nothing) == {"s1": False, "sanity": True}


def test_with_no_state_item_the_one_end_state_has_no_cell_and_any_change_fails_it():
    verifier = _verifier([])
    assert len(verifier.expected) == 1 and verifier.expected[0].cells == []
    assert _matches(verifier.expected, AtomContext(_run(STATE)))
    assert _matches(verifier.expected, AtomContext(_run(_moved()))) is False


def test_alternative_end_states_pass_on_any_one_and_value_sets_and_new_rows_are_matched():
    seven = dict(MOVE, id="s1", alt=0)
    four = dict(MOVE, id="s2", alt=1, expect={"slot": "four"})
    note = {"id": "n1", "kind": "row_new", "table": "notes", "where": {"item": "A1"}, "count": 1, "fact_ids": ["f1"]}
    verifier = _verifier(_kept(seven, four, note))
    assert len(verifier.expected) == 2 and all(s.new_rows for s in verifier.expected)
    with_note = {"N-9": {"item": "A1", "text": "moved"}}
    assert _matches(verifier.expected, AtomContext(_run(_moved("four", notes=with_note))))
    assert _matches(verifier.expected, AtomContext(_run(_moved("four")))) is False, "the note is missing"
    two = {"N-9": {"item": "A1"}, "N-10": {"item": "A1"}}
    assert _matches(verifier.expected, AtomContext(_run(_moved("seven", notes=two)))) is False
    banned = _verifier(_kept(dict(MOVE, expect={"slot": {"not": ["two", "four"]}})))
    assert _matches(banned.expected, AtomContext(_run(_moved("seven"))))
    assert _matches(banned.expected, AtomContext(_run(STATE))) is False


def test_event_items_become_conduct_and_forbidden_calls_and_are_scored_on_the_event_log():
    confirm = {"id": "e1", "kind": "before", "first": "confirm_turn", "then": "update_item",
               "policy_line": "after the user says yes"}
    never = {"id": "e2", "kind": "not_called", "tool": "add_note", "fact_ids": ["f1"]}
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    verifier = _verifier(_kept(MOVE, confirm, never, told))
    assert [(c.kind, c.tool) for c in verifier.conduct] == [("confirm_before_write", "update_item")]
    assert [(f.kind, f.tool) for f in verifier.forbidden] == [("write", "add_note")]
    calls = [("update_item", {"item_id": "A1", "slot": "seven"})]
    good = score_items(verifier, AtomContext(_run(_moved(), calls, said="A1 is now in slot seven.")))
    assert good == {"s1": True, "e1": True, "e2": True, "t1": True, "sanity": True}
    rushed = score_items(verifier, AtomContext(_run(_moved(), calls + [("add_note", {"text": "x"})], user=(),
                                                    said="I could not use slot seven.")))
    assert rushed["e1"] is False and rushed["e2"] is False and rushed["t1"] is False


def test_contradictions_name_a_tool_both_needed_and_forbidden_and_a_cell_expected_twice():
    called = {"id": "e1", "kind": "called", "tool": "add_note", "fact_ids": ["f1"]}
    never = {"id": "e2", "kind": "not_called", "tool": "add_note", "fact_ids": ["f1"]}
    other = dict(MOVE, id="s2", expect={"slot": "four"})
    found = I.contradictions(_kept(MOVE, other, called, never))
    assert [item_id for item_id, _ in found] == ["e2", "s2"]


def test_counts_split_items_by_kind_gate_and_provenance():
    items = _kept(MOVE, {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]})
    counts = I.counts_of(items + [I.sanity_item([])])
    assert counts["by_kind"] == {"nothing_else": 1, "row_is": 1, "said": 1}
    assert (counts["gate"], counts["scored"], counts["fact_backed"]) == (2, 1, 2)


def test_a_list_value_matches_its_members_in_any_order_and_a_free_field_may_move_unchecked():
    tagged = dict(MOVE, expect={"slot": "seven", "tags": ["four", "seven"]})
    assert "scalars found in the world" in keep(dict(MOVE, expect={"slot": [{"a": 1}]}))[1]
    assert "not in the world" in keep(dict(MOVE, expect={"slot": ["seven", "nine"]}))[1]
    STATE["items"]["A1"]["tags"] = []
    try:
        verifier = _verifier(_kept(tagged, dict(MOVE, id="s2", find={"item_id": "B2"}, expect={}, free=["label"])))
        b2 = dict(STATE["items"]["B2"], label="computed by a tool")
        moved = {"items": {"A1": dict(STATE["items"]["A1"], slot="seven", tags=["seven", "four"]), "B2": b2},
                 "notes": {}}
        assert _matches(verifier.expected, AtomContext(_run(moved)))
        assert score_items(verifier, AtomContext(_run(moved)))["sanity"] is True
        short = {**moved, "items": {**moved["items"], "A1": dict(moved["items"]["A1"], tags=["seven"])}}
        assert _matches(verifier.expected, AtomContext(_run(short))) is False
        assert "is in expect and in free" in keep(dict(MOVE, free=["slot"]))[1]
    finally:
        del STATE["items"]["A1"]["tags"]
