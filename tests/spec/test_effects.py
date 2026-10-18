"""Learned tool effects free what a write tool always moves; recorded call shapes refuse impossible call items.

The world is invented: two items in slots, each with a history list the move tool appends to, and a
shelf count the move lowers on whichever shelf it touches.
"""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import collateral
from kullback.runner.records import Event, Run, ToolCall, Verifier
from kullback.runner.verdict import verdict
from kullback.spec import effects as E
from kullback.spec import items as I
from kullback.spec.schema import FactSource, IntentFact

STATE = {"items": {"A1": {"item_id": "A1", "slot": "two", "label": "red", "history": [{"at": "one"}]},
                   "B2": {"item_id": "B2", "slot": "four", "label": "red", "history": []}},
         "shelves": {"S1": {"shelf_id": "S1", "free": 3}}, "notes": {}}
FACTS = [IntentFact(id="f1", text="Please move item A1 to slot seven, paying with card K9.", stance="volunteered",
                    source=FactSource(recording="rec1", turn=1), witnesses=["rec2"])]
TOOLS = [{"name": "move_item", "kind": "write", "args": ["item_id", "slot"]},
         {"name": "pay", "kind": "write", "args": ["item_id", "payments", "from_slot", "to_slot"]}]
KNOWN = {fact.id: fact for fact in FACTS}
MOVE = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "seven"},
        "fact_ids": ["f1"]}


def _moved_diff(row: str, before: dict, slot: str) -> dict:
    """What one recorded move did: the slot set, a history entry appended, a shelf count lowered."""
    after = dict(before, slot=slot, history=before["history"] + [{"at": slot}])
    return {"items": {row: {"before": before, "after": after}},
            "shelves": {"S1": {"before": {"shelf_id": "S1", "free": 3}, "after": {"shelf_id": "S1", "free": 2}}}}


# Two recorded moves on two rows: both moved slot and history and the shelf count; only one moved label.
EFFECTS = {"move_item": E.always([
    _moved_diff("A1", STATE["items"]["A1"], "five"),
    dict(_moved_diff("B2", STATE["items"]["B2"], "six"),
         items={"B2": {"before": STATE["items"]["B2"],
                       "after": dict(STATE["items"]["B2"], slot="six", label="blue", history=[{"at": "six"}])}})])}

CALLS = [ToolCall(id=f"k{n}", name="pay", raw_ptr={"file_hash": "h", "sim_index": n},
                  args={"item_id": item, "from_slot": "two", "to_slot": to,
                        "payments": [{"card": card, "amount": amount}]})
         for n, (item, to, card, amount) in enumerate([("A1", "five", "K1", 10), ("B2", "six", "K2", 20),
                                                       ("A1", "four", "K3", 30)])]
SIGNATURES = E.signatures(CALLS, key_columns=["card"])


def world() -> I.World:
    return I.World(STATE, "", FACTS, TOOLS, signatures=SIGNATURES)


def _kept(*items: dict) -> list[dict]:
    out = []
    for item in items:
        kept, why = world().check(item, KNOWN)
        assert why == "", why
        out.append(kept)
    return out


def _end(items_rows: dict, free: int = 2) -> dict:
    return {"items": {**STATE["items"], **items_rows}, "shelves": {"S1": {"shelf_id": "S1", "free": free}},
            "notes": {}}


def _run(end: dict, calls: list = ()) -> Run:
    events = [Event(idx=0, type="user_turn", payload={"content": "move it"})]
    for n, (name, args) in enumerate(calls):
        events.append(Event(idx=len(events), type="tool_call", payload={"id": f"c{n}", "name": name, "args": args}))
        events.append(Event(idx=len(events), type="tool_result", payload={"id": f"c{n}", "result": {}}))
    events.append(Event(idx=len(events), type="stop", payload={"start_state": STATE, "end_state": end}))
    return Run(run_id="r", task_id="t1", events=events)


def _verifier(items: list[dict]) -> Verifier:
    states = I.end_states_of(items, world(), effects=EFFECTS)
    return Verifier(task_id="t1", atoms=[I.atom_of(i) for i in items] + [I.sanity_atom(states)], expected=states)


def _sanity(items: list[dict], run: Run):
    [state] = I.end_states_of(items, world(), effects=EFFECTS)
    return collateral(state, AtomContext(run)).ok, state


A1_MOVED = dict(STATE["items"]["A1"], slot="seven", history=[{"at": "one"}, {"at": "seven"}])


def test_a_write_tools_always_changed_field_is_free_on_the_rows_it_writes():
    learned = EFFECTS["move_item"]
    assert [(c["table"], c["column"], c["patterns"]) for c in learned["columns"]] == [
        ("items", "history", ["history.N"]), ("items", "slot", ["slot"]), ("shelves", "free", ["free"])]
    ok, state = _sanity(_kept(MOVE), _run(_end({"A1": A1_MOVED})))
    assert ok is True
    rules = [rule for rule in state.allowed if rule["why"] == "free: tool effect: move_item"]
    assert {"table": "items", "row_id": "A1", "field": "history", "why": "free: tool effect: move_item"} in rules
    assert {"table": "shelves", "field": "free", "why": "free: tool effect: move_item"} in rules
    # Without the learned effects the same right Run fails the sanity gate on the appended history.
    [frozen] = I.end_states_of(_kept(MOVE), world())
    assert collateral(frozen, AtomContext(_run(_end({"A1": A1_MOVED})))).ok is False


def test_a_cell_on_an_effect_field_still_checks_that_field():
    # The writer pins the shelf count the move always lowers: the cell decides it, no free rule is added.
    pinned = {"id": "s2", "kind": "row_is", "table": "shelves", "find": {"shelf_id": "S1"}, "expect": {"free": 3},
              "fact_ids": ["f1"]}
    verifier = _verifier(_kept(MOVE, pinned))
    assert not any(rule.get("table") == "shelves" for rule in verifier.expected[0].allowed)
    assert verdict(_run(_end({"A1": A1_MOVED}, free=2)), verifier).passed is False
    assert verdict(_run(_end({"A1": A1_MOVED}, free=3)), verifier).passed is True


def test_the_sanity_gate_still_fails_a_change_outside_the_tool_effects():
    relabelled = dict(A1_MOVED, label="blue")
    assert _sanity(_kept(MOVE), _run(_end({"A1": relabelled})))[0] is False
    other_row = dict(STATE["items"]["B2"], history=[{"at": "seven"}])
    assert _sanity(_kept(MOVE), _run(_end({"A1": A1_MOVED, "B2": other_row})))[0] is False
    assert _sanity(_kept(MOVE), _run({**_end({"A1": A1_MOVED}), "notes": {"n1": {"text": "x"}}}))[0] is False


def _called(**args) -> dict:
    """A weighed called item: no policy line names the tool, so its gate would be weighed (D410)."""
    return {"id": "c1", "kind": "called", "tool": "pay", "args": args, "fact_ids": ["f1"], "gate": False}


def test_a_called_item_with_an_arg_shape_no_recording_shows_is_refused():
    _, bare = world().check(_called(item_id="A1", payments=["K9"]), KNOWN)
    assert bare == "pay payments: every recorded call passes objects with keys amount, card; " \
                   "write each entry as such an object"
    _, typed = world().check(_called(item_id="A1", payments={"card": "K9", "amount": 5}), KNOWN)
    assert typed == "pay payments: every recorded call passes list, not dict"
    _, same = world().check(_called(item_id="A1", from_slot="two", to_slot="two"), KNOWN)
    assert same == "pay: no recorded call gives from_slot and to_slot the same value"
    assert SIGNATURES["pay"]["never_equal"] == [["from_slot", "item_id"], ["from_slot", "to_slot"],
                                                ["item_id", "to_slot"]]


def test_a_list_of_objects_arg_matches_on_its_key_field():
    # The amount is the Run's own arithmetic, found nowhere; the card is the key field and is sourced.
    item = _called(item_id="A1", payments=[{"card": "K9", "amount": 12.5}])
    kept, why = world().check(item, KNOWN)
    assert why == "" and kept["args"]["payments"] == [{"card": "K9", "amount": 12.5}]
    _, unknown = world().check(_called(item_id="A1", payments=[{"card": "Q4", "amount": 12.5}]), KNOWN)
    assert unknown.startswith("payments.card: value \"Q4\" is not in the world")
    _, short = world().check(_called(item_id="A1", payments=[{"card": "K9"}]), KNOWN)
    assert short == "payments: each object carries the keys amount, card"
    verifier = _verifier([kept])
    paid = [("pay", {"item_id": "A1", "payments": [{"amount": 12.5, "card": "K9"}]})]
    assert verdict(_run(STATE, paid), verifier).passed is True
