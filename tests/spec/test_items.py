"""Items: code keeps only values it finds, compiles them to end states and gates, and scores each on a Run.

The world, the policy and the facts are invented: two items in slots, one user asking to move one.
"""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import match_context
from kullback.runner.records import Event, Run, Verifier
from kullback.runner.verdict import verdict
from kullback.spec import items as I
from kullback.spec.schema import FactSource, IntentFact


def _matches(end_states, context):
    """Any one end state matching passes; else an unsettled one is None; else False."""
    oks = [match_context(state, context).ok for state in end_states]
    return True if True in oks else (None if None in oks else False)


def _holds(verifier, run):
    """Each item of the Verdict on this Run: event and judge items by id, cells as state:table.row.field."""
    return {item.id: item.holds for item in verdict(run, verifier).items}


CELL = "state:items.A1.slot"


STATE = {"items": {"A1": {"item_id": "A1", "slot": "two", "label": "red"},
                   "B2": {"item_id": "B2", "slot": "four", "label": "red"}},
         "notes": {}}
POLICY = ("# Store policy\n\n## Moving items\n\nMove an item only to a free slot, after the user says yes.\n"
          "Moves go through update_item, notes through add_note, and nothing else does either.\n")
# The policy line naming the write tools (D414): a called gate stands only where its policy line names the tool.
PATH_LINE = "Moves go through update_item, notes through add_note, and nothing else does either."
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


def test_a_cell_path_meeting_a_list_without_an_index_is_refused_naming_the_path_and_the_list():
    state = {"items": dict(STATE["items"], A1=dict(STATE["items"]["A1"], parts=[{"slot": "two"}, {"slot": "four"}]))}
    w = I.World(state, POLICY, FACTS, TOOLS)
    _, why = w.check(dict(MOVE, expect={"parts.slot": "seven"}), KNOWN, unwitnessed)
    assert "'parts.slot' cannot be read" in why and "'parts' is a list of 2" in why and "parts.<index>.slot" in why
    assert w.check(dict(MOVE, expect={"parts.1.slot": "seven"}), KNOWN, unwitnessed)[1] == ""
    _, why = w.check({"id": "k1", "kind": "row_keeps", "table": "items", "find": {"item_id": "A1"},
                      "fields": ["parts.slot"], "fact_ids": ["f1"]}, KNOWN, unwitnessed)
    assert "cannot be read" in why


def test_a_cell_path_past_a_lists_end_or_to_a_key_the_row_lacks_is_kept_for_a_tool_may_add_it():
    assert I.unreadable_path({"parts": [{"slot": "two"}]}, "parts.3.slot") == ""
    assert I.unreadable_path({"parts": {"p1": {"slot": "two"}}}, "parts.p9.slot") == ""


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
    assert why == "" and kept["gate"] is False and kept["evidence"] == ["assistant_turns"]
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
    assert keep(refusal)[1] == "" and keep(dict(refusal, gate=False))[0]["gate"] is True, "a refusal item gates"
    assert keep(dict(judge, evidence=["assistant_turns"]))[1] == ""
    for ref in ("after_call:list_items", "assistant:2", "final_answer"):
        assert "evidence_why" in keep(dict(judge, evidence=[ref]))[1], "a narrower window needs its reason"
        kept, why = keep(dict(judge, evidence=[ref], evidence_why="the slot is read by the lookup"))
        assert why == "" and kept["evidence_why"] == "the slot is read by the lookup"
    assert "evidence" in keep(dict(judge, evidence=["the third turn"]))[1]
    assert "evidence" in keep(dict(judge, evidence=["after_call:teleport"]))[1]


def _kept(*items: dict) -> list[dict]:
    """Each item kept, a gated called item citing the policy line that names its tool."""
    out = []
    for item in items:
        named = item.get("kind") == "called" and item.get("gate", True) and not item.get("policy_line")
        kept, why = keep(dict(item, policy_line=PATH_LINE) if named else item)
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
    return Verifier(task_id="t1", atoms=[I.atom_of(i) for i in items] + [I.sanity_atom(states)], expected=states)


def _moved(slot="seven", **extra) -> dict:
    end = {"items": {**STATE["items"], "A1": dict(STATE["items"]["A1"], slot=slot)}, "notes": {}}
    for table, rows in extra.items():
        end[table] = {**end[table], **rows}
    return end


def test_the_sanity_item_fails_a_stray_write_and_holds_on_the_declared_row():
    verifier = _verifier(_kept(MOVE))
    right = _run(_moved(), [("update_item", {"item_id": "A1", "slot": "seven"})])
    stray = _run(_moved(items={"B2": dict(STATE["items"]["B2"], label="blue")}))
    assert _matches(verifier.expected, AtomContext(right)) and _holds(verifier, right) == {CELL: True, "sanity": True}
    assert _matches(verifier.expected, AtomContext(stray)) is False
    assert _holds(verifier, stray) == {CELL: True, "sanity": False}
    assert _holds(verifier, _run(STATE)) == {CELL: False, "sanity": True}


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
    missing = _holds(verifier, _run(_moved("four")))
    assert missing["new_rows:notes:1"] is False  # the count is its own gated item now, not sanity
    assert verdict(_run(_moved("four")), verifier).passed is False
    two = {"N-9": {"item": "A1"}, "N-10": {"item": "A1"}}
    assert _holds(verifier, _run(_moved("seven", notes=two)))["new_rows:notes:1"] is False
    banned = _verifier(_kept(dict(MOVE, expect={"slot": {"not": ["two", "four"]}})))
    assert _matches(banned.expected, AtomContext(_run(_moved("seven"))))
    assert _matches(banned.expected, AtomContext(_run(STATE))) is False


def test_an_alternative_whose_new_row_count_holds_beats_an_earlier_tie():
    move0 = dict(MOVE, id="s1", alt=0)
    move1 = dict(MOVE, id="s2", alt=1)
    two = {"id": "n1", "kind": "row_new", "table": "notes", "where": {"item": "A1"}, "count": 2,
           "fact_ids": ["f1"], "alt": 0}
    one = {"id": "n2", "kind": "row_new", "table": "notes", "where": {"item": "A1"}, "count": 1,
           "fact_ids": ["f1"], "alt": 1}
    verifier = _verifier(_kept(move0, move1, two, one))
    assert len(verifier.expected) == 2
    run = _run(_moved("seven", notes={"N-9": {"item": "A1", "text": "moved"}}))
    assert verdict(run, verifier).passed is True


def test_event_items_are_decided_by_the_verdict_off_the_event_log_not_compiled_to_conduct():
    confirm = {"id": "e1", "kind": "before", "first": "confirm_turn", "then": "update_item",
               "policy_line": "after the user says yes"}
    never = {"id": "e2", "kind": "not_called", "tool": "add_note", "fact_ids": ["f1"]}
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    verifier = _verifier(_kept(MOVE, confirm, never, told))
    assert verifier.conduct == [] and verifier.forbidden == []
    calls = [("update_item", {"item_id": "A1", "slot": "seven"})]
    good = _holds(verifier, _run(_moved(), calls, said="A1 is now in slot seven."))
    assert good == {CELL: True, "e1": True, "e2": True, "t1": True, "sanity": True}
    rushed = _holds(verifier, _run(_moved(), calls + [("add_note", {"text": "x"})], user=(), said="Nothing done."))
    assert rushed["e1"] is False and rushed["e2"] is False and rushed["t1"] is False


def test_a_gated_called_item_with_arguments_fails_a_run_that_called_the_tool_with_other_arguments():
    called = {"id": "c1", "kind": "called", "tool": "update_item", "args": {"item_id": "A1", "slot": "seven"},
              "fact_ids": ["f1"]}
    verifier = _verifier(_kept(called))
    assert verifier.atoms[0].kind == "allowed" and verifier.atoms[0].gate is True
    other = verdict(_run(STATE, [("update_item", {"item_id": "B2", "slot": "seven"})]), verifier)
    assert other.passed is False and other.failing_atom == "gate:called:c1" and other.score == 0.0
    assert verdict(_run(STATE, [("update_item", {"item_id": "A1", "slot": "seven"})]), verifier).passed is True


def test_a_gated_not_called_item_with_arguments_fails_only_on_a_matching_call():
    never = {"id": "n1", "kind": "not_called", "tool": "update_item", "args": {"item_id": "B2"}, "fact_ids": ["f1"]}
    verifier = _verifier(_kept(never))
    assert verdict(_run(STATE, [("update_item", {"item_id": "A1", "slot": "four"})]), verifier).passed is True
    hit = verdict(_run(STATE, [("update_item", {"item_id": "B2", "slot": "four"})]), verifier)
    assert hit.passed is False and hit.failing_atom == "gate:not_called:n1"


def test_a_scored_event_item_lowers_the_score_and_never_fails_the_run():
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"], "gate": False}
    result = verdict(_run(STATE, said="Done."), _verifier(_kept(told)))
    assert result.passed is True and 0.0 < result.score < 1.0


def test_contradictions_name_a_tool_both_needed_and_forbidden_and_a_cell_expected_twice():
    called = {"id": "e1", "kind": "called", "tool": "add_note", "fact_ids": ["f1"]}
    never = {"id": "e2", "kind": "not_called", "tool": "add_note", "fact_ids": ["f1"]}
    other = dict(MOVE, id="s2", expect={"slot": "four"})
    found = I.contradictions(_kept(MOVE, other, called, never))
    assert [item_id for item_id, _ in found] == ["e2", "s2"]


def test_two_gated_called_items_on_one_tool_with_different_args_contradict_and_only_those():
    def called(item_id, tool="update_item", **extra):
        return dict({"id": item_id, "kind": "called", "tool": tool, "fact_ids": ["f1"]}, **extra)
    a1, b2 = {"item_id": "A1"}, {"item_id": "B2"}
    found = I.contradictions(_kept(called("c1", args=a1), called("c2", args=b2)))
    assert found == [("c2", "called update_item with other args beside c1; gate the end state row instead, "
                             "or one called item with one_of")]
    assert I.contradictions(_kept(called("c1"), called("c2", args=b2))) == []
    assert I.contradictions(_kept(called("c1", args=a1), called("c2", tool="add_note", args={"text": "B2"}))) == []
    assert I.contradictions(_kept(called("c1", args=a1), called("c2", args=b2, gate=False))) == []
    assert [item_id for item_id, _ in I.contradictions(_kept(called("c1", args=a1), called("c2", args=dict(a1))))] \
        == ["c2"]


def test_two_gated_calls_of_one_tool_contradict_only_when_they_write_one_row_two_ways():
    """D440: with a World, calls on other rows, lookups and calls that make their own row each stand."""
    def called(name, tool="update_item", **args):
        return {"id": name, "kind": "called", "tool": tool, "gate": True, "args": args}
    at = I.World(STATE, POLICY, FACTS, TOOLS)
    assert I.contradictions([called("c1", item_id="A1", slot="seven"), called("c2", item_id="B2", slot="eight")],
                            at) == []
    found = I.contradictions([called("c1", item_id="A1", slot="seven"), called("c2", item_id="A1", slot="eight")], at)
    assert [item_id for item_id, _ in found] == ["c2"] and "one row two ways" in found[0][1]
    assert I.contradictions([called("c1", "list_items", item_id="A1"), called("c2", "list_items", item_id="B2")],
                            at) == []
    made = I.World(STATE, POLICY, FACTS, TOOLS, effects={"update_item": {"makes": ["items"]}})
    assert I.contradictions([called("c1", item_id="A1", slot="seven"), called("c2", item_id="A1", slot="eight")],
                            made) == []


def test_a_call_targets_the_rows_its_arguments_name_in_the_tables_it_writes():
    effects = {"update_item": {"writes": [{"table": "items", "field": "slot"}], "columns": []}}
    at = I.World(STATE, POLICY, FACTS, TOOLS, effects=effects)
    assert at.target_rows("update_item", {"item_id": "B2", "slot": "four"}) == {("items", "B2")}
    assert at.target_rows("list_items", {"item_id": "B2"}) == set()
    assert at.target_rows("update_item", {"item_id": "Z9"}) == set()


def test_a_freeze_on_a_field_the_specs_own_write_expects_on_that_row_contradicts_naming_the_write():
    frozen = {"id": "k1", "kind": "row_keeps", "table": "items", "find": {"item_id": "A1"}, "fields": ["slot"],
              "fact_ids": ["f1"]}
    found = I.contradictions(_kept(MOVE, frozen))
    assert ("k1", "row_keeps holds slot of that row, which s1 changes; drop the freeze on that field or the write") \
        in found and {item_id for item_id, _ in found} == {"k1"}


def test_a_freeze_on_another_field_or_another_row_stands_beside_the_write():
    label = {"id": "k1", "kind": "row_keeps", "table": "items", "find": {"item_id": "A1"}, "fields": ["label"],
             "fact_ids": ["f1"]}
    other = dict(label, id="k2", find={"item_id": "B2"}, fields=["slot"])
    assert I.contradictions(_kept(MOVE, label, other)) == []


def _event(item_id, kind, **extra):
    return dict({"id": item_id, "kind": kind, "tool": "update_item", "fact_ids": ["f1"]}, **extra)


def test_a_gated_not_called_matching_a_gated_called_items_args_contradicts_and_nothing_else_does():
    call = _event("c1", "called", args={"item_id": "A1", "slot": "seven"})
    never = _event("n1", "not_called", args={"item_id": "A1"})
    assert I.contradictions(_kept(call, never)) == [
        ("n1", "gated not_called update_item beside gated called c1; keep the one the ask calls for")]
    assert [item_id for item_id, _ in I.contradictions(_kept(call, _event("n1", "not_called")))] == ["n1"]
    assert I.contradictions(_kept(call, _event("n1", "not_called", args={"item_id": "A1"}, gate=False))) == []


def test_a_gated_not_called_with_other_args_stands_beside_the_called_item():
    call = _event("c1", "called", args={"item_id": "A1", "slot": "seven"})
    assert I.contradictions(_kept(call, _event("n1", "not_called", args={"item_id": "B2"}))) == []
    assert I.contradictions(_kept(_event("c1", "called"), _event("n1", "not_called", args={"item_id": "B2"}))) == []


def test_an_event_item_description_names_its_args_and_reads_as_before_without_them():
    call = _verifier(_kept(_event("c1", "called", args={"item_id": "A1", "slot": "seven"}))).atoms[0]
    assert call.description == "update_item is called with item_id A1, slot seven"
    never = _verifier(_kept(_event("n1", "not_called"))).atoms[0]
    assert never.description == "update_item is not called"


def test_a_repeated_gated_event_item_is_reported_once_and_the_first_is_kept():
    never = _event("n1", "not_called", args={"item_id": "B2"})
    found = I.contradictions(_kept(never, dict(never, id="n2"), _event("n3", "not_called", args={"item_id": "A1"})))
    assert found == [("n2", "not_called update_item repeats n1 with the same args; remove this one")]


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
        verifier = _verifier(_kept(tagged, dict(MOVE, id="s2", find={"item_id": "B2"}, expect={"slot": "four"},
                                                       free={"label": "a tool recolours it"})))
        b2 = dict(STATE["items"]["B2"], label="computed by a tool")
        moved = {"items": {"A1": dict(STATE["items"]["A1"], slot="seven", tags=["seven", "four"]), "B2": b2},
                 "notes": {}}
        assert _matches(verifier.expected, AtomContext(_run(moved)))
        assert _holds(verifier, _run(moved))["sanity"] is True
        short = {**moved, "items": {**moved["items"], "A1": dict(moved["items"]["A1"], tags=["seven"])}}
        assert _matches(verifier.expected, AtomContext(_run(short))) is False
        assert "is in expect and in free" in keep(dict(MOVE, free={"slot": "a tool sets it"}))[1]
    finally:
        del STATE["items"]["A1"]["tags"]


def test_a_free_field_needs_a_one_line_reason_and_the_reason_rides_on_the_end_state():
    for free in (["label"], {"label": ""}, {"label": "two\nlines"}):
        assert "free is {field: one line" in keep(dict(MOVE, free=free))[1]
    kept, why = keep(dict(MOVE, free={"label": "a tool recolours it"}))
    assert why == "" and kept["free"] == {"label": "a tool recolours it"}
    allowed = I.end_states_of([kept], world())[0].allowed
    assert {"table": "items", "row_id": "A1", "field": "label", "why": "free: s1: a tool recolours it"} in allowed
    assert I.counts_of([kept])["free_fields"] == 1


def test_a_row_is_whose_expect_is_empty_or_only_its_find_key_is_refused():
    assert "checks nothing" in keep(dict(MOVE, expect={}, free={"label": "a tool sets it"}))[1]
    assert "checks nothing" in keep(dict(MOVE, expect={"item_id": "A1"}))[1]


def test_a_free_field_holding_a_value_an_intent_fact_names_is_refused():
    named = IntentFact(id="f3", text="Please put item A1 in slot four.", stance="volunteered",
                       source=FactSource(recording="rec1", turn=1), witnesses=["rec2"])
    told = I.World(STATE, POLICY, FACTS + [named], TOOLS)
    item = dict(MOVE, expect={"label": "red"}, free={"slot": "a tool picks the slot"})
    assert "fact f3 names a value" in told.check(item, KNOWN, unwitnessed)[1]
    assert world().check(item, KNOWN, unwitnessed)[1] == "", "no fact names a slot the world holds"


def test_a_gate_whose_value_only_an_echo_fact_gives_is_weighed_and_a_scored_item_may_cite_it():
    """Two turns: the agent proposes slot 9, the user says yes. The fact is echo by code."""
    from kullback.spec.intent_tools import UserTurn, is_echo
    turn = UserTurn(index=1, text="Yes, that works.", agent_before="I can move A1 to slot 9 for you, ok?")
    assert is_echo("agrees to move A1 to slot 9", "accepted", turn)
    assert is_echo("wants A1 in slot 9", "volunteered", turn)
    assert not is_echo("wants A1 moved", "volunteered", UserTurn(index=1, text="Move A1 please.", agent_before="Hi."))
    echo = IntentFact(id="f3", text="agrees to move A1 to slot 9", stance="accepted", witnesses=["rec2"],
                      source=FactSource(recording="rec1", turn=1), echo=True)
    known = {**KNOWN, "f3": echo}
    item = {"id": "s9", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "9"},
            "fact_ids": ["f3"]}
    gate_world = I.World(STATE, POLICY, [*FACTS, echo], TOOLS)
    kept, why = gate_world.check(item, known, unwitnessed)
    assert kept["gate"] is False and kept["weighed"] == I.ECHO_WEIGHED and "rests only on echo facts" in why
    kept, why = gate_world.check({**item, "gate": False}, known, unwitnessed)
    assert why == "" and kept["gate"] is False


def test_a_stored_intent_is_marked_echo_from_its_own_turns():
    from kullback.spec.intent_tools import TaskRecordings, UserTurn, mark_echo
    found = TaskRecordings(task_id="t", recordings={"rec1": [
        UserTurn(index=1, text="Please move item A1 to slot seven.", agent_before="Hello."),
        UserTurn(index=3, text="Sure.", agent_before="Shall I leave a note?")]})
    assert [fact.echo for fact in mark_echo(FACTS, found)] == [False, True]


def test_an_anchor_answer_is_looked_up_never_computed():
    """A number the cited row and the policy do not hold is the writer's arithmetic: refused."""
    judge = {"id": "j1", "kind": "judge", "question": "Did the agent give both slots?", "fact_ids": ["f1"],
             "anchor": {"answer": "B2 is in slot four, 6 slots in all", "row": {
                 "table": "items", "find": {"item_id": "B2"}, "fields": ["slot"]}}}
    kept, why = keep(judge)
    assert kept is None and "states 6" in why and "computed" in why
    assert keep(dict(judge, anchor=dict(judge["anchor"], answer="B2 is in slot four")))[1] == ""
    counted = I.counts_of([{"kind": "judge", "anchor": {"row": None}}, {"kind": "judge", "anchor": {"row": {"t": 1}}}])
    assert counted["anchors_without_lookup"] == 1


def test_a_hand_off_item_gates_only_when_its_policy_line_demands_the_hand_off():
    policy = POLICY + "\nTransfer the user to a person when the slot is taken.\n"
    tools = [*TOOLS, {"name": "transfer_to_person", "kind": "write", "args": []}]
    hand_off = {"id": "h1", "kind": "called", "tool": "transfer_to_person", "gate": True, "fact_ids": ["f1"]}
    kept, why = I.World(STATE, policy, FACTS, tools).check(hand_off, KNOWN, unwitnessed)
    assert why == "" and kept["gate"] is False, "no policy line: weighed"
    grounded = dict(hand_off, policy_line="Transfer the user to a person when the slot is taken.")
    kept, why = I.World(STATE, policy, FACTS, tools).check(grounded, KNOWN, unwitnessed)
    assert why == "" and kept["gate"] is True
    move = {"id": "c1", "kind": "called", "tool": "update_item", "gate": True, "fact_ids": ["f1"],
            "policy_line": PATH_LINE}
    assert I.World(STATE, policy, FACTS, tools).check(move, KNOWN, unwitnessed)[0]["gate"] is True


def test_after_pushback_and_a_witnessed_yes_are_evidence_and_no_longer_weigh_a_gate():
    """D366: the pushback bar is gone. A fact said after pushback, or accepted and volunteered elsewhere,
    gates; a reaction (condition agent) is what weighs a gate."""
    pushed = IntentFact(id="f3", text="Cancel the A1 order instead.", stance="volunteered",
                        source=FactSource(recording="rec1", turn=3), after_pushback=True)
    agreed = IntentFact(id="f4", text="Maybe leave a note.", stance="accepted",
                        source=FactSource(recording="rec1", turn=3), witnesses=["rec2"])
    reaction = pushed.model_copy(update={"id": "f5", "stance": "fallback", "condition": "agent"})
    known = {**KNOWN, "f3": pushed, "f4": agreed, "f5": reaction}
    world = I.World(STATE, POLICY, FACTS + [pushed, agreed], TOOLS)

    def weighed(fact_ids, **extra):
        return world.check(dict(MOVE, id="s9", fact_ids=fact_ids, **extra), known, unwitnessed)
    assert weighed(["f3"]) == (weighed(["f3"])[0], "") and weighed(["f3"])[0]["gate"] is True
    assert weighed(["f4"])[0]["gate"] is True
    kept, why = weighed(["f1", "f5"], policy_line="Move an item only to a free slot")
    assert kept["gate"] is False and kept["weighed"] == I.AGENT_WEIGHED and "reaction" in why


def test_the_counts_report_one_end_state_per_branch():
    branch = dict(MOVE, id="s2", alt=1, expect={"slot": "four"})
    assert I.counts_of([MOVE, I.sanity_item([])])["end_states"] == 1
    assert I.counts_of([MOVE, branch, I.sanity_item([])])["end_states"] == 2
    assert I.counts_of([I.sanity_item([])])["end_states"] == 1


def test_row_new_carries_its_gate_and_weight_into_new_rows():
    [state] = I.end_states_of([{"id": "n1", "kind": "row_new", "table": "notes",
                                "where": {"text": "hi"}, "count": 1, "gate": False, "weight": 2.0}],
                              world())
    assert state.new_rows == [{"table": "notes", "where": {"text": "hi"}, "count": 1, "item": "n1",
                               "gate": False, "weight": 2.0}]


def test_a_dict_cell_whose_first_value_is_a_scalar_resolves_its_source_instead_of_raising():
    priced = IntentFact(id="f9", text="Set the price to 7.5 please.", stance="volunteered",
                        source=FactSource(recording="rec1", turn=5))
    told = I.World(STATE, POLICY, FACTS + [priced], TOOLS)
    assert I._cell_source({"amount": 7.5}, told) == told.source(7.5) != None  # noqa: E711
    assert I._cell_source({"amount": 7.5}, world()) is None
    assert I._cell_source({"slot": "seven"}, world()) == world().source("seven")


def test_a_dict_cell_whose_first_value_is_a_list_resolves_to_the_source_of_its_last_value():
    assert I._cell_source({"slots": ["seven", "red"]}, world()) == I.ValueSource(
        kind="world", ptr={"table": "items", "row": "A1"})
    assert I._cell_source({"slots": ["red", "nowhere"]}, world()) is None
    assert I._cell_source({"slots": [None]}, world()) == I.ValueSource(kind="world", ptr={"removed": True})


def test_a_plain_list_or_plain_scalar_cell_resolves_as_before():
    assert I._cell_source(["red", "seven"], world()) == world().source("seven")
    assert I._cell_source(["seven", "nowhere"], world()) is None
    assert I._cell_source("two", world()) == I.ValueSource(kind="world", ptr={"table": "items", "row": "A1"})
    assert I._cell_source(None, world()) == I.ValueSource(kind="world", ptr={"removed": True})
