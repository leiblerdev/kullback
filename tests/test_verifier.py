"""Tests for builder/verifier.py: derivation from re-runs on disk, the D79 checks, and the tau2 export."""

from __future__ import annotations

import json
from pathlib import Path

from harness.builder import verifier as V
from harness.shared.canon import canon_value
from harness.shared.records import (
    Atom,
    Constraint,
    Event,
    Run,
    Task,
    UserRules,
    Verifier,
    as_dict,
)

WRITE_TOOLS = {"cancel_pending_order"}
ORDER = {"order_id": "#W123", "status": "pending", "total": 150.0}
TASK = Task(id="t1", intent="cancel the pending order and record the reason")


# --- tiny hand-built Runs -------------------------------------------------

def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return _ev("user_turn", content=text)


def assistant(text: str) -> dict:
    return _ev("model_call", reply={"content": text})


def call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return _ev("tool_call", id=cid, name=name, args=args, kind=kind)


def result(data, cid: str = "c1") -> dict:
    return _ev("tool_result", id=cid, result=data)


def make_run(run_id: str, events: list[dict], task_id: str = "t1", termination_reason: str = "success") -> Run:
    return Run(
        run_id=run_id,
        task_id=task_id,
        termination_reason=termination_reason,
        events=[Event(idx=i, type=e["type"], payload=e["payload"]) for i, e in enumerate(events)],
    )


def reference_events(reason: str = "no longer needed",
                     final: str = "Your order #W123 is cancelled and 150.0 is refunded.") -> list[dict]:
    return [
        user("Please cancel my order #W123."),
        assistant("Sure. Why do you want to cancel it?"),
        user(reason),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": reason}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant(final),
    ]


def reference_run() -> Run:
    return make_run("ref", reference_events())


def alt_path_run() -> Run:
    """Same writes, different reads and different wording (D46 different path)."""
    return make_run("alt", [
        user("Please cancel my order #W123."),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Happy to help. May I ask the reason?"),
        user("no longer needed"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Done: #W123 is cancelled, 150.0 goes back to your card."),
    ])


def other_reason_run() -> Run:
    return make_run("rr2", reference_events(reason="changed my mind"))


def extra_write_run() -> Run:
    """A successful re-run that writes one entity more than the Reference did."""
    return make_run("extra", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W888", "reason": "no longer needed"}, cid="c2"),
        result({"order_id": "#W888", "status": "cancelled"}, cid="c2"),
    ])


def failed_run() -> Run:
    return make_run("bad", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c1"),
        result({"error": "not found"}, cid="c1"),
        assistant("I could not find that order."),
    ], termination_reason="max_steps")


def wrong_run() -> Run:
    """Plausible but wrong: the wrong entity, everything else in place."""
    return make_run("wrong", [
        user("Please cancel my order #W123."),
        assistant("Sure. Why do you want to cancel it?"),
        user("no longer needed"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W999", "status": "cancelled"}, cid="c1"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])


def empty_run() -> Run:
    return make_run("empty", [], termination_reason="max_steps")


def write_events_jsonl(run: Run, path: Path) -> str:
    """A Run as one header line plus one line per event."""
    head = as_dict(run)
    events = head.pop("events")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(head) + "\n")
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return str(path)


def write_run_json(run: Run, path: Path) -> str:
    """A whole Run as a single JSON line."""
    path.write_text(json.dumps(as_dict(run)) + "\n", encoding="utf-8")
    return str(path)


def derive(tmp_path: Path, reruns=None, **kwargs) -> Verifier:
    reruns = reruns if reruns is not None else [alt_path_run(), other_reason_run()]
    paths = [write_events_jsonl(r, tmp_path / f"{r.run_id}.jsonl") for r in reruns]
    kwargs.setdefault("write_tools", WRITE_TOOLS)
    return V.derive_verifier(TASK, reference_run(), paths, None, **kwargs)


def atom_by_id(verifier: Verifier, atom_id: str) -> Atom:
    found = [a for a in verifier.atoms if a.id == atom_id]
    assert found, f"no atom {atom_id} in {[a.id for a in verifier.atoms]}"
    return found[0]


def payloads(verifier: Verifier) -> list[dict]:
    return [V.atom_payload(a) for a in verifier.atoms]


# --- reading re-runs from disk (D91) --------------------------------------

def test_load_run_reads_header_plus_event_lines(tmp_path):
    path = Path(write_events_jsonl(reference_run(), tmp_path / "ref.jsonl"))
    run = V.load_run(path)
    assert run.run_id == "ref"
    assert len(run.events) == 8
    assert run.events[0].type == "user_turn"


def test_load_run_reads_a_whole_run_on_one_line(tmp_path):
    path = Path(write_run_json(alt_path_run(), tmp_path / "alt.jsonl"))
    run = V.load_run(path)
    assert run.run_id == "alt"
    assert len(run.events) == 8


def test_module_never_imports_the_runner_and_never_runs_anything(tmp_path):
    """D91: the Builder talks to the Runner through Run records on disk, in one direction only."""
    import inspect

    source = Path(V.__file__).read_text(encoding="utf-8")
    assert "harness.runner" not in source
    assert "from harness import runner" not in source
    assert "subprocess" not in source and "os.system" not in source
    # Nothing in the module takes a way to execute a Run; re-runs arrive as paths the Runner wrote.
    taken = set(inspect.signature(V.derive_verifier).parameters)
    assert "rerun_paths" in taken
    assert not taken & {"model", "runner", "environment", "loop", "router"}
    # The one check that needs a Run nobody has written yet asks the caller for it (D91).
    assert "run_probe" in set(inspect.signature(V.loophole_probe).parameters)


def test_load_run_keeps_the_state_the_loop_footer_carries(tmp_path):
    """loop.py writes Start and End state on a footer line; a Hard rule reads them off the stop event."""
    path = tmp_path / "footer.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in as_dict(reference_run())["events"]:
            handle.write(json.dumps(event) + "\n")
        handle.write(json.dumps({"run_id": "delivered", "task_id": "t1", "termination_reason": "user_stop",
                                 "start_state": {"orders": {"#W123": {"status": "delivered"}}},
                                 "end_state": {"orders": {"#W123": {"status": "cancelled"}}}}) + "\n")
    run = V.load_run(path)
    assert run.run_id == "delivered"
    assert run.termination_reason == "user_stop"
    assert V._start_state(run) == {"orders": {"#W123": {"status": "delivered"}}}
    assert [e.type for e in run.events][-1] == "stop"


def test_a_state_reading_hard_rule_fails_the_run_it_cannot_hold_on(tmp_path):
    """The Starting state comes off the footer, so 'never cancel a delivered order' can be checked."""
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    order = pre_state['orders'][write_call['arguments']['order_id']]\n"
                                     "    return order['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])

    def with_state(status: str) -> Path:
        path = tmp_path / f"{status}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in as_dict(reference_run())["events"]:
                handle.write(json.dumps(event) + "\n")
            handle.write(json.dumps({"run_id": status, "task_id": "t1", "termination_reason": "user_stop",
                                     "start_state": {"orders": {"#W123": {"status": status}}}}) + "\n")
        return path

    assert V.check_run(verifier, with_state("pending")) == (True, None)
    assert V.check_run(verifier, with_state("delivered")) == (False, "hard.k1")


def test_a_hard_rule_that_raises_is_a_failure_and_not_a_silent_pass(tmp_path):
    """A predicate that blows up decided nothing, so the oracle check has to see it (D79 check 2)."""
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return pre_state['orders']['#W123']['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    assert V.check_run(verifier, reference_run()) == (False, "hard.k1")  # the Run carries no state
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run())}
    assert gates["verifier_oracle"].passed is False


def test_a_hard_rule_cannot_walk_dunders_out_of_the_sandbox(tmp_path):
    """A Hard constraint that never imports anything can still reach the process through
    ().__class__.__base__.__subclasses__(); _hard_holds has to refuse it before exec, the same way
    runner/verdict.py's gate() refuses it, or the walk runs for real and the Run scores a pass."""
    import subprocess  # make sure Popen is a loaded subclass of object for the walk to find

    assert subprocess.Popen  # imported, not used directly: the escape below finds it by name
    marker = tmp_path / "escaped.txt"
    escape = (
        "def check(pre_state, write_call, transcript):\n"
        "    for cls in ().__class__.__base__.__subclasses__():\n"
        "        if cls.__name__ == 'Popen':\n"
        f"            cls(['touch', {str(marker)!r}]).wait()\n"
        "            return True\n"
        "    return False\n"
    )
    rule = Constraint(id="k1", text="escape probe", compiled=True, predicate_src=escape)
    verifier = derive(tmp_path, constraints=[rule])
    atom = atom_by_id(verifier, "hard.k1")
    assert V._hard_gate(atom.predicate_src)  # the walk is caught by the dunder-attribute gate
    assert V._hard_holds(atom, reference_run(), WRITE_TOOLS) is False
    assert not marker.exists(), "the escape payload ran outside the sandbox"
    assert V.check_run(verifier, reference_run()) == (False, "hard.k1")


def test_a_benign_hard_rule_still_evaluates_once_gated(tmp_path):
    """The gate has to let an ordinary policy predicate through, not just refuse bad ones."""
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    order = pre_state['orders'][write_call['arguments']['order_id']]\n"
                                     "    return order['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    assert V._hard_gate(rule.predicate_src) == []
    path = tmp_path / "pending.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in as_dict(reference_run())["events"]:
            handle.write(json.dumps(event) + "\n")
        handle.write(json.dumps({"run_id": "pending", "task_id": "t1", "termination_reason": "user_stop",
                                 "start_state": {"orders": {"#W123": {"status": "pending"}}}}) + "\n")
    assert V.check_run(verifier, path) == (True, None)


# --- write-set diff, agreement, provenance --------------------------------

def test_write_present_in_every_successful_rerun_is_required(tmp_path):
    verifier = derive(tmp_path)
    presence = [a for a in verifier.atoms if V.atom_payload(a).get("kind") == "write"]
    assert len(presence) == 1
    assert presence[0].kind == "required"
    assert V.atom_payload(presence[0])["entity"] == canon_value("#W123")  # canon.py by default (D39)


def test_user_stated_value_is_required_and_elicited_value_is_allowed(tmp_path):
    verifier = derive(tmp_path)
    order = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "order_id"][0]
    reason = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "reason"][0]
    assert order.provenance == "user_stated"
    assert order.kind == "required"
    assert reason.provenance == "user_elicited"
    assert reason.kind == "allowed"


def test_system_derived_and_agent_chosen_provenance(tmp_path):
    run = make_run("p", [
        user("Refund my order please."),
        call("get_order_details", {"order_id": "#W555"}, kind="read", cid="c0"),
        result({"order_id": "#W555", "total": 42.5}, cid="c0"),
        call("cancel_pending_order", {"order_id": "#W555", "reason": "goodwill"}, cid="c1"),
        result({"status": "cancelled"}, cid="c1"),
        assistant("Refunded 42.5."),
    ])
    verifier = V.derive_verifier(TASK, run, [], None, write_tools=WRITE_TOOLS)
    order = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "order_id"][0]
    reason = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "reason"][0]
    assert order.provenance == "system_derived"
    assert order.kind == "required"
    assert reason.provenance == "agent_chosen"
    assert reason.kind == "allowed"


def test_value_in_some_successful_reruns_is_allowed(tmp_path):
    verifier = derive(tmp_path)
    reason = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "reason"][0]
    assert reason.kind == "allowed"
    assert V.atom_payload(reason)["raw"] == "no longer needed"


def test_a_write_only_a_failed_rerun_made_is_not_an_atom(tmp_path):
    """D43: present in every successful re-run is required, in some allowed, in none not an atom."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    entities = {V.atom_payload(a)["entity"] for a in verifier.atoms if V.atom_payload(a)["kind"] == "write"}
    assert entities == {canon_value("#W123")}
    assert [a.id for a in verifier.atoms if a.kind == "forbidden"] == []


def test_every_atom_span_points_at_the_event_holding_its_value(tmp_path):
    """A span is evidence, so it has to name the Run and the turn the value actually sits in (D66)."""
    verifier = derive(tmp_path)
    runs = {r.run_id: r for r in [reference_run(), alt_path_run(), other_reason_run()]}
    checked = 0
    for atom in verifier.atoms:
        payload = V.atom_payload(atom)
        if payload["kind"] not in ("write", "write_value", "question", "communicate"):
            continue
        assert atom.spans, atom.id
        span = atom.spans[0]
        run = runs[span.file_hash]
        event = next(e for e in run.events if e.idx == span.msg_index)
        text = json.dumps(event.payload, default=str)
        wanted = payload.get("raw", payload.get("text", payload.get("entity")))
        if payload["kind"] == "write":
            assert event.type == "tool_call" and event.payload["name"] == payload["tool"]
        elif payload["kind"] == "question":
            assert event.type == "user_turn"
        else:
            assert str(wanted) in text or canon_value(wanted) in canon_value(text), (atom.id, text)
        checked += 1
    assert checked == len([a for a in verifier.atoms if V.atom_payload(a)["kind"] != "entity_count"])


def test_a_write_whose_result_carried_an_error_is_not_an_effect(tmp_path):
    """A call that failed changed nothing (D67), so it is not a write the next Run has to repeat."""
    ref = make_run("ref", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W124", "reason": "customer request"}, cid="c1"),
        _ev("tool_result", id="c1", result=None,
            error={"class": "not_found_entity", "payload": "no such order"}),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c2"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c2"),
        assistant("Cancelled."),
    ])
    verifier = V.derive_verifier(TASK, ref, [], None, write_tools=WRITE_TOOLS)
    writes = [V.atom_payload(a)["entity"] for a in verifier.atoms if V.atom_payload(a)["kind"] == "write"]
    assert writes == [canon_value("#W123")]
    assert V.atom_payload(atom_by_id(verifier, "entity_count"))["count"] == 1
    good = make_run("cand", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c2"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c2"),
        assistant("Cancelled."),
    ])
    assert V.check_run(verifier, good) == (True, None)


def test_the_entity_is_the_scalar_id_and_not_a_list_of_item_ids(tmp_path):
    """tau2's exchange tools take `item_ids` too; two Runs listing them in another order are one write."""
    def events(items):
        return [
            user("Exchange items 1001 and 2002 in order #W1."),
            call("exchange_delivered_order_items",
                 {"order_id": "#W1", "item_ids": items, "new_item_ids": ["3003"]}, cid="c1"),
            result({"ok": True}, cid="c1"),
            assistant("Done."),
        ]
    rerun = make_run("rr", events(["2002", "1001"]))
    verifier = V.derive_verifier(TASK, make_run("ref", events(["1001", "2002"])),
                                 [write_events_jsonl(rerun, tmp_path / "rr.jsonl")], None,
                                 write_tools={"exchange_delivered_order_items"})
    writes = [(a.kind, V.atom_payload(a)["entity"]) for a in verifier.atoms
              if V.atom_payload(a)["kind"] == "write"]
    assert writes == [("required", canon_value("#W1"))]
    assert V.check_run(verifier, empty_run())[0] is False


def test_a_user_stated_amount_in_another_spelling_is_still_user_stated(tmp_path):
    """D42: the user said the value, whatever the spelling; the canonicalizer decides, not the letters."""
    def events(amount):
        return [
            user("Please refund $150 on my order #W123."),
            call("refund_order", {"order_id": "#W123", "amount": amount}, cid="c1"),
            result({"ok": True}, cid="c1"),
            assistant("Refunded."),
        ]
    verifier = V.derive_verifier(TASK, make_run("ref", events(150.0)), [], None,
                                 write_tools={"refund_order"})
    amount = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "amount"][0]
    assert amount.provenance == "user_stated"
    assert amount.kind == "required"
    assert V.check_run(verifier, make_run("cand", events(10)))[0] is False


def test_a_user_side_write_keeps_its_requestor(tmp_path):
    """D71: a write the logs record as the user's own says so on the atom and in the tau2 export."""
    ref = make_run("ref", [
        user("I will pay the bill myself now."),
        _ev("tool_call", id="u1", name="pay_bill", args={"bill_id": "B1"}, kind="write", requestor="user"),
        result({"paid": True}, cid="u1"),
        assistant("Thanks, I see the payment."),
    ])
    verifier = V.derive_verifier(TASK, ref, [], None)
    write = [a for a in verifier.atoms if V.atom_payload(a)["kind"] == "write"][0]
    assert V.atom_payload(write)["requestor"] == "user"
    assert V.export_tau2_actions(verifier)[0]["requestor"] == "user"


def test_write_tools_can_come_from_the_event_marking(tmp_path):
    verifier = V.derive_verifier(TASK, reference_run(), [], None)
    written = [V.atom_payload(a)["entity"] for a in verifier.atoms if V.atom_payload(a).get("kind") == "write"]
    assert written == [canon_value("#W123")]


def test_successful_run_ids_decides_which_reruns_agree(tmp_path):
    reruns = [alt_path_run(), other_reason_run()]
    paths = [write_events_jsonl(r, tmp_path / f"{r.run_id}.jsonl") for r in reruns]
    verifier = V.derive_verifier(TASK, reference_run(), paths, None, write_tools=WRITE_TOOLS,
                                 successful_run_ids=["alt"])
    assert verifier.seed_run_ids == ["ref", "alt"]


class LowerCanon:
    """A stand-in for canon.py: the one rule that two spellings of an id are the same value (D39)."""

    def canonical(self, value):
        return value.lower() if isinstance(value, str) else value


def test_the_canonicalizer_makes_two_spellings_of_an_id_agree(tmp_path):
    shouty = make_run("shouty", [
        user("Please cancel my order #W123."),
        assistant("Sure. Why do you want to cancel it?"),
        user("no longer needed"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": "#w123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    paths = [write_events_jsonl(shouty, tmp_path / "shouty.jsonl")]
    plain = V.derive_verifier(TASK, reference_run(), paths, lambda value: value, write_tools=WRITE_TOOLS)
    canoned = V.derive_verifier(TASK, reference_run(), paths, LowerCanon(), write_tools=WRITE_TOOLS)
    assert len([a for a in plain.atoms if V.atom_payload(a).get("kind") == "write"]) == 2
    assert len([a for a in canoned.atoms if V.atom_payload(a).get("kind") == "write"]) == 1
    order = [a for a in canoned.atoms if V.atom_payload(a).get("field") == "order_id"][0]
    assert order.kind == "required"


# --- questions, communicate facts, entity count, hard constraints ---------

def test_question_asked_becomes_an_atom(tmp_path):
    verifier = derive(tmp_path)
    questions = [a for a in verifier.atoms if a.kind == "question"]
    assert [V.atom_payload(a)["key"] for a in questions] == ["field:reason"]


def test_a_run_that_skips_the_required_question_fails(tmp_path):
    verifier = derive(tmp_path)
    silent = make_run("silent", [
        user("Please cancel my order #W123."),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    passed, failing = V.check_run(verifier, silent)
    assert passed is False
    assert failing == "q.field:reason"


def test_communicate_facts_agreed_across_reruns(tmp_path):
    verifier = derive(tmp_path)
    stated = sorted(V.atom_payload(a)["text"] for a in verifier.atoms if a.kind == "communicate")
    assert stated == ["#W123", "150.0"]


def test_a_fact_missing_from_one_rerun_is_not_a_communicate_atom(tmp_path):
    quiet = make_run("quiet", reference_events(final="Your order #W123 is cancelled."))
    verifier = derive(tmp_path, reruns=[quiet])
    stated = sorted(V.atom_payload(a)["text"] for a in verifier.atoms if a.kind == "communicate")
    assert stated == ["#W123"]


def test_entity_count_atom_caps_side_effects(tmp_path):
    verifier = derive(tmp_path)
    count = atom_by_id(verifier, "entity_count")
    assert count.kind == "required"
    assert V.atom_payload(count)["count"] == 1
    noisy = make_run("noisy", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W777", "reason": "oops"}, cid="c2"),
        result({"status": "cancelled"}, cid="c2"),
    ])
    passed, failing = V.check_run(verifier, noisy)
    assert passed is False
    assert failing in ("entity_count", "w1")


def test_the_write_cap_counts_calls_so_the_same_entity_twice_fails(tmp_path):
    """The cap and the Runner's `writes_count()` have to count the same thing or they disagree."""
    verifier = derive(tmp_path)
    twice = make_run("twice", reference_events()[:-1] + [
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c2"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c2"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    assert atom_by_id(verifier, "entity_count").predicate_src == "writes_count() <= 1"
    assert V.check_run(verifier, twice) == (False, "entity_count")


def test_a_rerun_that_ran_to_the_end_without_a_simulated_user_is_successful(tmp_path):
    """loop.py stops such a Run with `agent_stop`; reading that as a failure loses a good re-run."""
    assert V._successful(make_run("r", [], termination_reason="agent_stop"), None) is True
    assert V._successful(make_run("r", [], termination_reason="max_turns"), None) is False
    assert V._successful(make_run("r", [], termination_reason="agent_stop"), ["other"]) is False
    stopped = make_run("stopped", reference_events(), termination_reason="agent_stop")
    verifier = derive(tmp_path, reruns=[stopped])
    assert verifier.seed_run_ids == ["ref", "stopped"]


def test_compiled_hard_constraints_become_atoms_and_residuals_do_not(tmp_path):
    compiled = Constraint(id="k1", text="never cancel a delivered order", predicate_src="def check(): return True",
                          compiled=True)
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    residual = Constraint(id="k3", text="use good judgement", residual_reason="not checkable")
    verifier = derive(tmp_path, constraints=[compiled, judged, residual])
    hard = [a.id for a in verifier.atoms if a.kind == "hard"]
    assert hard == ["hard.k1", "hard.k2"]
    assert atom_by_id(verifier, "hard.k2").judge is True


def test_check_run_fails_a_run_that_breaks_a_hard_constraint(tmp_path):
    """A Hard constraint is a gate, so the D79 checks have to see it fail, not skip it."""
    never = Constraint(
        id="k1", text="never cancel without a prior user confirmation", compiled=True,
        predicate_src="def check(pre_state, write_call, transcript):\n    return user_confirmed(transcript)\n")
    always = Constraint(id="k2", text="always allowed", compiled=True,
                        predicate_src="def check(pre_state, write_call, transcript):\n    return True\n")
    unconfirmed = derive(tmp_path, constraints=[never])
    passed, failing = V.check_run(unconfirmed, reference_run())
    assert passed is False
    assert failing == "hard.k1"
    satisfied = derive(tmp_path, constraints=[always])
    assert V.check_run(satisfied, reference_run()) == (True, None)


def test_a_judge_atom_is_never_answered_by_code(tmp_path):
    """D76: judge.py answers it, so the atom carries no predicate and no scorer may decide it."""
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    verifier = derive(tmp_path, constraints=[judged])
    atom = atom_by_id(verifier, "hard.k2")
    assert atom.judge is True
    assert not atom.predicate_src  # nothing for verdict.py to evaluate
    assert V._hard_holds(atom, reference_run(), WRITE_TOOLS) is None
    # It is not silently satisfied either: a rude Run passes only because code did not decide.
    assert V.check_run(verifier, reference_run()) == (True, None)
    assert V.check_run(verifier, make_run("rude", reference_events(final="No."))) [1] != "hard.k2"


def test_a_hard_rule_sees_the_transcript_before_the_write_and_nothing_after(tmp_path):
    """D43 case 3: a confirmation that arrived after the write did not precede it."""
    never = Constraint(
        id="k1", text="never cancel without a prior user confirmation", compiled=True,
        predicate_src="def check(pre_state, write_call, transcript):\n    return user_confirmed(transcript)\n")
    confirmed = make_run("ref", [
        user("Please cancel my order #W123."),
        assistant("Shall I cancel #W123?"),
        user("yes"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Cancelled."),
    ])
    verifier = V.derive_verifier(TASK, confirmed, [], None, write_tools=WRITE_TOOLS, constraints=[never])
    late = make_run("late", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("I cancelled #W123. Was that what you wanted?"),
        user("yes"),
        assistant("Great."),
    ])
    hard = atom_by_id(verifier, "hard.k1")
    assert V._hard_holds(hard, confirmed, WRITE_TOOLS) is True
    assert V._hard_holds(hard, late, WRITE_TOOLS) is False
    assert V.check_run(Verifier(task_id="t1", atoms=[hard]), late) == (False, "hard.k1")


def test_a_hard_rule_is_asked_about_a_tool_the_seed_runs_never_used(tmp_path):
    """D94: 'never call delete_order' is worth nothing if only the Reference's own tools are judged."""
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    hard_only = Verifier(task_id="t1", atoms=[atom_by_id(verifier, "hard.k1")])
    deleted = make_run("bad", reference_events()[:-1] + [
        call("delete_order", {"order_id": "#W123"}, cid="c9"),
        result({"deleted": True}, cid="c9"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    assert V.check_run(hard_only, deleted) == (False, "hard.k1")
    assert V.check_run(hard_only, reference_run()) == (True, None)


def test_a_hard_rule_is_not_asked_about_a_tool_the_seed_runs_only_read_with(tmp_path):
    """A before-write predicate judges writes and unknown tools, not the reads it was never written for."""
    rule = Constraint(id="k1", text="never touch a delivered order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['arguments'].get('order_id') != '#W123'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    reading = make_run("reading", [
        user("What is the status of #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("It is pending."),
    ])
    hard_only = Verifier(task_id="t1", atoms=[atom_by_id(verifier, "hard.k1")])
    assert V.check_run(hard_only, reading) == (True, None)
    assert V.check_run(hard_only, reference_run()) == (False, "hard.k1")


def test_verifier_records_its_seed_runs_and_round_trips(tmp_path):
    verifier = derive(tmp_path)
    assert verifier.task_id == "t1"
    assert verifier.seed_run_ids == ["ref", "alt", "rr2"]
    again = Verifier.model_validate(json.loads(json.dumps(as_dict(verifier))))
    assert again == verifier


# --- the local scorer -----------------------------------------------------

def test_oracle_passes_empty_fails_alt_path_passes(tmp_path):
    verifier = derive(tmp_path)
    assert V.check_run(verifier, reference_run())[0] is True
    assert V.check_run(verifier, alt_path_run())[0] is True
    assert V.check_run(verifier, empty_run())[0] is False
    assert V.check_run(verifier, wrong_run())[0] is False


def test_a_different_wording_of_an_allowed_value_passes(tmp_path):
    verifier = derive(tmp_path)
    reworded = make_run("reworded", reference_events(reason="I found it cheaper elsewhere"))
    assert V.check_run(verifier, reworded)[0] is True


def test_a_write_no_successful_run_made_still_fails_on_the_write_cap(tmp_path):
    """The stray write of a failed re-run is not an atom (D43); the cap is what stops a Candidate."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    over = make_run("over", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c2"),
        result({"status": "cancelled"}, cid="c2"),
    ])
    passed, failing = V.check_run(verifier, over)
    assert passed is False
    assert failing == "entity_count"


def test_check_run_needs_write_tools_to_catch_an_extra_write_on_an_uncovered_tool(tmp_path):
    """build.py:347 always supplies the mined write_tools in production; no call in this file did
    until this test, so the extra-write safety net (_extra_write) had no coverage of that shape."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), extra_write_run()])
    sneaky = make_run("sneaky", reference_events() + [
        call("refund_order", {"order_id": "#W123", "amount": 25.0}, kind="read", cid="c2"),
        result({"ok": True}, cid="c2"),
    ])
    # With no write_tools, the extra call is invisible: nothing marked it as a write, so it slips
    # through as a full pass.
    assert V.check_run(verifier, sneaky) == (True, None)
    # The same Run, scored the way build.py actually scores it, is caught.
    assert V.check_run(verifier, sneaky, write_tools={"cancel_pending_order", "refund_order"}) == (
        False, "extra_write:refund_order")


# --- D79 validation -------------------------------------------------------

def test_validate_verifier_all_checks_pass(tmp_path, test_model):
    verifier = derive(tmp_path)
    gates = V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()])
    assert [g.stage for g in gates] == [
        "verifier_provenance_spans", "verifier_oracle", "verifier_empty_run",
        "verifier_wrong_run", "verifier_unfinished_run", "verifier_alt_path", "verifier_loophole",
        "verifier_leak", "verifier_mutation",
    ]
    assert all(g.passed for g in gates), [(g.stage, g.failures) for g in gates]


def test_the_unfinished_run_stops_just_before_the_required_write_and_must_fail(tmp_path):
    """D119: the Reference cut one step short (GLM 5.3's unsolved-state check) scores no pass."""
    verifier = derive(tmp_path)
    unfinished = V.unfinished_run(verifier, reference_run())
    assert unfinished.run_id == "ref.unfinished" and unfinished.termination_reason == "max_turns"
    assert [e.type for e in unfinished.events][-2:] == ["tool_call", "tool_result"]  # the read, not the cancel
    assert not any(e.type == "tool_call" and e.payload.get("name") == "cancel_pending_order"
                   for e in unfinished.events)
    assert V.check_run(verifier, unfinished)[0] is False
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run())}
    assert gates["verifier_unfinished_run"].passed is True


def test_a_verifier_that_rewards_an_unfinished_run_is_caught(tmp_path):
    verifier = derive(tmp_path)
    # Only the write cap survives: a Verifier that checks the agent wrote nothing extra and never
    # that the cancellation happened, which a Run that stopped short satisfies.
    hollow = verifier.model_copy(update={"atoms": [a for a in verifier.atoms
                                                    if V.atom_payload(a).get("kind") == "entity_count"]})
    gates = {g.stage: g for g in V.validate_verifier(hollow, reference_run())}
    assert gates["verifier_unfinished_run"].metrics["run_passed"] is True
    assert gates["verifier_unfinished_run"].passed is False


def test_a_reference_with_nothing_to_cut_has_no_unfinished_run(tmp_path):
    verifier = derive(tmp_path)
    assert V.unfinished_run(verifier, empty_run()) is None
    assert V.unfinished_run(verifier, make_run("talk", [user("hi"), assistant("hello")])).events == [
        e for e in make_run("talk", [user("hi"), assistant("hello")]).events if e.type == "user_turn"]


def test_the_suite_reports_every_d79_check_by_the_name_validate_py_wants(tmp_path, test_model):
    """validate.py counts a check it was not told about as a failure, so the names have to line up."""
    from harness.runner import (
        validate,  # the mapping is the contract between the two; nothing imports it
    )

    verifier = derive(tmp_path)
    gates = V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()])
    results = V.d79_results(gates)
    assert sorted(results) == sorted(validate.D79_CHECKS)
    assert validate.verifier_gate(results).passed is True


def test_a_check_with_no_run_is_reported_as_not_run(tmp_path):
    """A Verifier nobody tried to break is not a validated Verifier; missing input is a failure."""
    from harness.runner import validate

    verifier = derive(tmp_path)
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run())}
    assert gates["verifier_wrong_run"].passed is False
    assert gates["verifier_alt_path"].passed is False
    assert gates["verifier_loophole"].passed is False
    assert all("not run" in " ".join(gates[stage].failures)
               for stage in ("verifier_wrong_run", "verifier_alt_path", "verifier_loophole"))
    assert validate.verifier_gate(V.d79_results(gates.values())).passed is False


def test_a_hollow_verifier_fails_the_suite_with_no_runs_supplied(tmp_path):
    """The empty Run needs no Runner, so check 3 always runs and an atomless Verifier cannot clear it."""
    hollow = Verifier(task_id="t1", atoms=[])
    gates = {g.stage: g for g in V.validate_verifier(hollow, reference_run())}
    assert gates["verifier_empty_run"].passed is False
    assert gates["verifier_empty_run"].metrics["run_passed"] is True
    assert any(not g.passed for g in gates.values())


def test_validate_flags_a_verifier_an_empty_run_can_pass(tmp_path):
    hollow = Verifier(task_id="t1", atoms=[])
    gates = {g.stage: g for g in V.validate_verifier(hollow, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None)}
    assert gates["verifier_empty_run"].passed is False
    assert gates["verifier_wrong_run"].passed is False


def test_validate_flags_a_provenance_atom_without_a_span(tmp_path):
    verifier = derive(tmp_path)
    for atom in verifier.atoms:
        if atom.provenance == "user_stated":
            atom.spans = []
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None)}
    assert gates["verifier_provenance_spans"].passed is False
    assert gates["verifier_provenance_spans"].failures


def test_validate_flags_a_span_that_points_at_the_wrong_turn(tmp_path):
    verifier = derive(tmp_path)
    for atom in verifier.atoms:
        if atom.provenance == "user_stated" and atom.spans:
            atom.spans[0].msg_index = 3
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None)}
    assert gates["verifier_provenance_spans"].passed is False


def test_the_spans_gate_reads_a_rerun_span_in_the_run_it_names(tmp_path):
    """An atom only a re-run holds carries that re-run's span; resolving it elsewhere is a false hit."""
    extra = make_run("extra", [
        user("Please cancel my orders #W123 and #W888."),
        assistant("Sure. Why do you want to cancel them?"),
        user("moving abroad"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "moving abroad"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        call("cancel_pending_order", {"order_id": "#W888", "reason": "moving abroad"}, cid="c2"),
        result({"order_id": "#W888", "status": "cancelled"}, cid="c2"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    verifier = derive(tmp_path, reruns=[extra])
    w888 = [a for a in verifier.atoms
            if V.atom_payload(a)["kind"] == "write_value" and V.atom_payload(a)["field"] == "reason"
            and str(V.atom_payload(a)["entity"]).lower().endswith("w888")][0]
    assert w888.spans[0].file_hash == "extra"
    with_seed = {g.stage: g for g in V.validate_verifier(verifier, reference_run(), seed_runs=[extra])}
    assert with_seed["verifier_provenance_spans"].passed is True
    alone = {g.stage: g for g in V.validate_verifier(verifier, reference_run())}
    assert alone["verifier_provenance_spans"].passed is False
    assert "not supplied" in " ".join(alone["verifier_provenance_spans"].failures)


def test_the_mutation_check_flags_an_atom_the_reference_cannot_fail(tmp_path):
    """D79 check 8: an atom whose mutation changes nothing is an atom that is not being checked."""
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    talking = make_run("talking", [user("Hello?"), assistant("Hello, how can I help?")])
    verifier = V.derive_verifier(TASK, talking, [], None, write_tools=WRITE_TOOLS, constraints=[rule])
    gate = [g for g in V.validate_verifier(verifier, talking) if g.stage == "verifier_mutation"][0]
    assert gate.passed is False  # nothing was written, so the rule was never asked anything
    assert "hard.k1" in " ".join(gate.failures)
    lively = [g for g in V.validate_verifier(derive(tmp_path, constraints=[rule]), reference_run())
              if g.stage == "verifier_mutation"][0]
    assert lively.passed is True
    assert lively.metrics["atoms_mutated"] >= 4


def test_leak_check_finds_a_number_the_verifier_read_off_a_tool_result(tmp_path):
    """Check 7 greps for constants only the Verifier should know, whether or not they are strings."""
    def events(amount):
        return [
            user("Please refund my order #W123."),
            call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
            result(ORDER, cid="c0"),
            call("refund_order", {"order_id": "#W123", "amount": amount}, cid="c1"),
            result({"ok": True}, cid="c1"),
            assistant("Refunded."),
        ]
    verifier = V.derive_verifier(TASK, make_run("ref", events(150.0)), [], None,
                                 write_tools={"refund_order"})
    amount = [a for a in verifier.atoms if V.atom_payload(a).get("field") == "amount"][0]
    assert amount.provenance == "system_derived"
    gates = {g.stage: g for g in V.validate_verifier(verifier, make_run("ref", events(150.0)),
                                                     intent_text="refund exactly 150.0 on #W123",
                                                     user_rules=UserRules())}
    assert gates["verifier_leak"].passed is False
    assert "150.0" in " ".join(gates["verifier_leak"].failures)


def test_leak_check_finds_a_verifier_constant_in_the_intent(tmp_path):
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in V.validate_verifier(
        verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
        intent_text="cancel #W123 and refund exactly 150.0", user_rules=UserRules())}
    assert gates["verifier_leak"].passed is False
    assert "150.0" in " ".join(gates["verifier_leak"].failures)


def test_leak_check_finds_a_verifier_constant_in_the_user_rules(tmp_path):
    verifier = derive(tmp_path)
    rules = UserRules(refusals=["do not accept less than 150.0"])
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=rules)}
    assert gates["verifier_leak"].passed is False


def test_the_user_own_words_are_not_a_leak(tmp_path):
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in V.validate_verifier(
        verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
        intent_text="cancel order #W123 and record the reason", user_rules=UserRules())}
    assert gates["verifier_leak"].passed is True


def test_a_loophole_probe_that_did_not_run_is_not_a_pass(tmp_path):
    """D79 check 6 skipped is 'we do not know', which the suite has to say out loud."""
    verifier = derive(tmp_path)
    gate = V.loophole_probe(verifier, None)
    assert gate.stage == "verifier_loophole"
    assert gate.metrics["skipped"] is True
    assert gate.passed is False
    assert "not run" in " ".join(gate.failures)
    gates = {g.stage: g for g in V.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None)}
    assert gates["verifier_loophole"].passed is False


def test_loophole_probe_uses_the_run_it_is_given(tmp_path, test_model):
    verifier = derive(tmp_path)
    gate = V.loophole_probe(verifier, test_model, run_probe=lambda model, ver: empty_run())
    assert gate.passed is True
    gate = V.loophole_probe(verifier, test_model, run_probe=lambda model, ver: reference_run())
    assert gate.passed is False


# --- tau2 export ----------------------------------------------------------

def test_export_tau2_actions_shape(tmp_path):
    actions = V.export_tau2_actions(derive(tmp_path))
    assert actions == [{
        "action_id": "t1_0",
        "requestor": "assistant",
        "name": "cancel_pending_order",
        "arguments": {"order_id": "#W123", "reason": "no longer needed"},
        "info": None,
    }]


def test_export_skips_forbidden_writes(tmp_path):
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    actions = V.export_tau2_actions(verifier)
    assert [a["arguments"]["order_id"] for a in actions] == ["#W123"]


def test_export_can_be_limited_to_required_writes(tmp_path):
    verifier = derive(tmp_path, reruns=[alt_path_run(), extra_write_run()])
    assert len(V.export_tau2_actions(verifier)) == 2
    required_only = V.export_tau2_actions(verifier, include_allowed=False)
    assert [a["arguments"]["order_id"] for a in required_only] == ["#W123"]


def test_a_fact_stated_before_the_farewell_is_a_communicate_fact():
    run = make_run("r", [
        user("What is the status of order #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Order #W123 is pending."),
        user("Thanks, that is all."),
        assistant("You are welcome, goodbye."),
    ])
    said = V.communicate_values(run, V._canon_fn(None))
    assert [v["text"] for v in said.values()] == ["#W123"]
