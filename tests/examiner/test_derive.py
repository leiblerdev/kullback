"""Tests for examiner/derive.py: derivation from re-runs on disk and the tau2 export.

The module moved here from builder/verifier.py in phase 5 (D123) with its behaviour unchanged, which
the hash pin at the top holds. Reading a Run, scoring one against a Verifier and the D79 suite live in
kullback.gates.verifier_suite (phase 3, D122) and are tested in tests/gates/test_verifier_suite.py; the
Runs both files use come from tests/gates/verifier_fixtures.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from gates.verifier_fixtures import (
    ORDER,
    TASK,
    WRITE_TOOLS,
    _ev,
    alt_path_run,
    assistant,
    atom_by_id,
    call,
    derive,
    empty_run,
    extra_write_run,
    failed_run,
    make_run,
    other_reason_run,
    reference_events,
    reference_run,
    result,
    user,
    write_events_jsonl,
)
from kullback.examiner import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules, canon_value
from kullback.runner.records import Constraint, Task, Verifier, as_dict, content_hash

# The derivation over verifier_fixtures.derive(tmp_path): seven atoms, seeds ref, alt and rr2. It was
# pinned at the commit the phase 5 move started from (a40812c, 77d497a99ac9e8c1) to hold that the
# move changed no byte of the artifact (D130). It has moved twice since: when a stated fact the
# request never asked about stopped being a demand, and when the write cap moved to the end of the
# atom list so the rule on a cap of 0 could see what else the Verifier demands. A change to this
# value is a change to every Verifier of every build, so it is moved deliberately or not at all.
DERIVATION_HASH = "ab18f441cf7e5e3790ca3f78b67ea2e7a7ffa06879c66318156d607d39537fa3"


def test_the_derivation_never_imports_the_builder_the_runner_internals_or_anything_that_runs(tmp_path):
    """D91 and D123: the derivation reads Run records the Runner wrote and the suite's vocabulary, nothing else.

    canon and records sit in kullback.runner with the rest of the Runner package (D121); this module
    may read them, so the check names the internal modules that stay out of reach rather than the
    package prefix. The Builder is off limits the other way: the Examiner is the one writer of
    Verifiers and never reads what the Builder compiled.
    """
    import inspect
    import re

    source = Path(V.__file__).read_text(encoding="utf-8")
    assert re.search(r"kullback\.runner\.(?!records\b|canon\b)\w", source) is None
    assert "from kullback import runner" not in source
    assert "kullback.builder" not in source
    assert "from kullback import builder" not in source
    assert "subprocess" not in source and "os.system" not in source
    # Nothing in the module takes a way to execute a Run; re-runs arrive as paths the Runner wrote.
    taken = set(inspect.signature(V.derive_verifier).parameters)
    assert "rerun_paths" in taken
    assert not taken & {"model", "runner", "environment", "loop", "router"}
    # The one check that needs a Run nobody has written yet asks the caller for it (D91).
    assert "run_probe" in set(inspect.signature(S.loophole_probe).parameters)
    assert not [line for line in source.splitlines()
                if line.startswith(("import kullback.gates", "from kullback.gates"))
                and "verifier_suite" not in line], "derive.py reads the suite and nothing else in gates"


def test_the_derivation_over_the_fixture_runs_hashes_to_the_pinned_value(tmp_path):
    """The same Runs derive the same atoms, byte for byte, until a decision moves the pin (D130)."""
    verifier = derive(tmp_path)
    assert len(verifier.atoms) == 7 and verifier.seed_run_ids == ["ref", "alt", "rr2"]
    assert content_hash(as_dict(verifier)) == DERIVATION_HASH


# --- write-set diff, agreement, provenance --------------------------------

def test_write_present_in_every_successful_rerun_is_required(tmp_path):
    verifier = derive(tmp_path)
    presence = [a for a in verifier.atoms if S.atom_payload(a).get("kind") == "write"]
    assert len(presence) == 1
    assert presence[0].kind == "required"
    assert S.atom_payload(presence[0])["entity"] == canon_value("#W123")  # canon.py by default (D39)


def test_user_stated_value_is_required_and_elicited_value_is_allowed(tmp_path):
    verifier = derive(tmp_path)
    order = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "order_id"][0]
    reason = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "reason"][0]
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
    order = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "order_id"][0]
    reason = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "reason"][0]
    assert order.provenance == "system_derived"
    assert order.kind == "required"
    assert reason.provenance == "agent_chosen"
    assert reason.kind == "allowed"


def test_value_in_some_successful_reruns_is_allowed(tmp_path):
    verifier = derive(tmp_path)
    reason = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "reason"][0]
    assert reason.kind == "allowed"
    assert S.atom_payload(reason)["raw"] == "no longer needed"


def test_a_write_only_a_failed_rerun_made_is_not_an_atom(tmp_path):
    """D43: present in every successful re-run is required, in some allowed, in none not an atom."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    entities = {S.atom_payload(a)["entity"] for a in verifier.atoms if S.atom_payload(a)["kind"] == "write"}
    assert entities == {canon_value("#W123")}
    assert [a.id for a in verifier.atoms if a.kind == "forbidden"] == []


def test_every_atom_span_points_at_the_event_holding_its_value(tmp_path):
    """A span is evidence, so it has to name the Run and the turn the value actually sits in (D66)."""
    verifier = derive(tmp_path)
    runs = {r.run_id: r for r in [reference_run(), alt_path_run(), other_reason_run()]}
    checked = 0
    for atom in verifier.atoms:
        payload = S.atom_payload(atom)
        if payload["kind"] not in ("write", "write_value", "question", "communicate",
                                   V.REPORTED_COMMUNICATE):
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
    assert checked == len([a for a in verifier.atoms if S.atom_payload(a)["kind"] != "entity_count"])


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
    writes = [S.atom_payload(a)["entity"] for a in verifier.atoms if S.atom_payload(a)["kind"] == "write"]
    assert writes == [canon_value("#W123")]
    assert S.atom_payload(atom_by_id(verifier, "entity_count"))["count"] == 1
    good = make_run("cand", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c2"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c2"),
        assistant("Cancelled."),
    ])
    assert S.check_run(verifier, good) == (True, None)


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
    writes = [(a.kind, S.atom_payload(a)["entity"]) for a in verifier.atoms
              if S.atom_payload(a)["kind"] == "write"]
    assert writes == [("required", canon_value("#W1"))]
    assert S.check_run(verifier, empty_run())[0] is False


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
    amount = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "amount"][0]
    assert amount.provenance == "user_stated"
    assert amount.kind == "required"
    assert S.check_run(verifier, make_run("cand", events(10)))[0] is False


def test_a_user_side_write_keeps_its_requestor(tmp_path):
    """D71: a write the logs record as the user's own says so on the atom and in the tau2 export."""
    ref = make_run("ref", [
        user("I will pay the bill myself now."),
        _ev("tool_call", id="u1", name="pay_bill", args={"bill_id": "B1"}, kind="write", requestor="user"),
        result({"paid": True}, cid="u1"),
        assistant("Thanks, I see the payment."),
    ])
    verifier = V.derive_verifier(TASK, ref, [], None)
    write = [a for a in verifier.atoms if S.atom_payload(a)["kind"] == "write"][0]
    assert S.atom_payload(write)["requestor"] == "user"
    assert V.export_tau2_actions(verifier)[0]["requestor"] == "user"


def test_write_tools_can_come_from_the_event_marking(tmp_path):
    verifier = V.derive_verifier(TASK, reference_run(), [], None)
    written = [S.atom_payload(a)["entity"] for a in verifier.atoms if S.atom_payload(a).get("kind") == "write"]
    assert written == [canon_value("#W123")]


def test_successful_run_ids_decides_which_reruns_agree(tmp_path):
    reruns = [alt_path_run(), other_reason_run()]
    paths = [write_events_jsonl(r, tmp_path / f"{r.run_id}.jsonl") for r in reruns]
    verifier = V.derive_verifier(TASK, reference_run(), paths, None, write_tools=WRITE_TOOLS,
                                 successful_run_ids=["alt"])
    assert verifier.seed_run_ids == ["ref", "alt"]


def test_the_canonicalizer_makes_two_spellings_of_an_id_agree(tmp_path):
    """D39: the shipped canonicalizer under the customer's own learned rules, not a stand-in for it."""
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
    rules = CanonRules(id_patterns={"orders.order_id": r"#[Ww]\d+"})
    plain = V.derive_verifier(TASK, reference_run(), paths, lambda value: value, write_tools=WRITE_TOOLS)
    canoned = V.derive_verifier(TASK, reference_run(), paths, rules, write_tools=WRITE_TOOLS)
    assert len([a for a in plain.atoms if S.atom_payload(a).get("kind") == "write"]) == 2
    assert len([a for a in canoned.atoms if S.atom_payload(a).get("kind") == "write"]) == 1
    order = [a for a in canoned.atoms if S.atom_payload(a).get("field") == "order_id"][0]
    assert order.kind == "required"


# --- questions, communicate facts, entity count, hard constraints ---------

def test_question_asked_becomes_an_atom(tmp_path):
    verifier = derive(tmp_path)
    questions = [a for a in verifier.atoms if a.kind == "question"]
    assert [S.atom_payload(a)["key"] for a in questions] == ["field:reason"]


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
    passed, failing = S.check_run(verifier, silent)
    assert passed is False
    assert failing == "q.field:reason"


def test_communicate_facts_agreed_across_reruns(tmp_path):
    """Both facts become atoms; which of them the answer must repeat is the request's business."""
    verifier = derive(tmp_path)
    stated = sorted(S.atom_payload(a)["text"] for a in verifier.atoms
                    if S.atom_payload(a)["kind"] in ("communicate", V.REPORTED_COMMUNICATE))
    assert stated == ["#W123", "150.0"]


def test_a_fact_missing_from_one_rerun_is_not_a_communicate_atom(tmp_path):
    quiet = make_run("quiet", reference_events(final="Your order #W123 is cancelled."))
    verifier = derive(tmp_path, reruns=[quiet])
    stated = sorted(S.atom_payload(a)["text"] for a in verifier.atoms if a.kind == "communicate")
    assert stated == ["#W123"]


def test_entity_count_atom_caps_side_effects(tmp_path):
    verifier = derive(tmp_path)
    count = atom_by_id(verifier, "entity_count")
    assert count.kind == "required"
    assert S.atom_payload(count)["count"] == 1
    noisy = make_run("noisy", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W777", "reason": "oops"}, cid="c2"),
        result({"status": "cancelled"}, cid="c2"),
    ])
    passed, failing = S.check_run(verifier, noisy)
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
    assert S.check_run(verifier, twice) == (False, "entity_count")


def test_a_rerun_that_ran_to_the_end_without_a_simulated_user_is_successful(tmp_path):
    """loop.py stops such a Run with `agent_stop`; reading that as a failure loses a good re-run."""
    assert V.successful(make_run("r", [], termination_reason="agent_stop"), None) is True
    assert V.successful(make_run("r", [], termination_reason="max_turns"), None) is False
    assert V.successful(make_run("r", [], termination_reason="agent_stop"), ["other"]) is False
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
    passed, failing = S.check_run(unconfirmed, reference_run())
    assert passed is False
    assert failing == "hard.k1"
    satisfied = derive(tmp_path, constraints=[always])
    assert S.check_run(satisfied, reference_run()) == (True, None)


def test_a_judge_atom_is_never_answered_by_code(tmp_path):
    """D76: judge.py answers it, so the atom carries no predicate and no scorer may decide it."""
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    verifier = derive(tmp_path, constraints=[judged])
    atom = atom_by_id(verifier, "hard.k2")
    assert atom.judge is True
    assert not atom.predicate_src  # nothing for verdict.py to evaluate
    assert S.hard_holds(atom, reference_run(), WRITE_TOOLS) is None
    # It is not silently satisfied either: a rude Run passes only because code did not decide.
    assert S.check_run(verifier, reference_run()) == (True, None)
    assert S.check_run(verifier, make_run("rude", reference_events(final="No."))) [1] != "hard.k2"


def test_verifier_records_its_seed_runs_and_round_trips(tmp_path):
    verifier = derive(tmp_path)
    assert verifier.task_id == "t1"
    assert verifier.seed_run_ids == ["ref", "alt", "rr2"]
    again = Verifier.model_validate(json.loads(json.dumps(as_dict(verifier))))
    assert again == verifier


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




# --- which stated facts the answer must repeat (D43) ----------------------
#
# An invented domain: a repair shop whose agent looks a pump up and answers the caller. The
# recorded answer states three facts read from the world, and the request asks about one of them.

PUMP = {"part_id": "P-2044", "warranty_months": 36, "list_price": 249.0}
SHOP_TOOLS = {"open_repair_ticket"}


def pump_run(run_id: str = "shop-ref",
             final: str = "Pump P-2044 has 36 warranty months left and lists at 249.0.",
             asked: str = "How long is the warranty on pump P-2044?") -> object:
    return make_run(run_id, [
        user(asked),
        call("get_part_record", {"part_id": "P-2044"}, kind="read", cid="r0"),
        result(PUMP, cid="r0"),
        assistant(final),
    ], task_id="shop")


def pump_verifier(intent_text: str) -> Verifier:
    return V.derive_verifier(Task(id="shop", intent=intent_text), pump_run(), [], None,
                             write_tools=SHOP_TOOLS)


def stated(verifier: Verifier, kind: str) -> list[str]:
    return sorted(S.atom_payload(a)["text"] for a in verifier.atoms if S.atom_payload(a)["kind"] == kind)


def fact_atom(verifier: Verifier, text: str):
    return [a for a in verifier.atoms if S.atom_payload(a).get("text") == text][0]


def test_a_fact_the_caller_named_must_be_stated_back(tmp_path):
    verifier = pump_verifier("tell the caller about the pump they asked about")
    assert stated(verifier, "communicate") == ["P-2044"]
    assert fact_atom(verifier, "P-2044").kind == "communicate"


def test_a_fact_the_request_never_asked_about_is_reported_and_rejects_no_run(tmp_path):
    """The recorded agent volunteered the price; a Run that answers without it has still done the job."""
    verifier = pump_verifier("tell the caller about the pump they asked about")
    assert stated(verifier, V.REPORTED_COMMUNICATE) == ["249.0", "36"]
    quieter = pump_run("shop-alt", final="Your pump P-2044 is still under warranty.")
    assert S.check_run(verifier, quieter, write_tools=SHOP_TOOLS) == (True, None)


def test_a_fact_the_request_names_by_its_field_must_be_stated_back(tmp_path):
    """The caller asked in words, not in numbers: the field the fact was read from is the link."""
    verifier = pump_verifier("tell the caller how many warranty months the pump has left")
    assert stated(verifier, "communicate") == ["36", "P-2044"]
    assert stated(verifier, V.REPORTED_COMMUNICATE) == ["249.0"]
    silent = pump_run("shop-alt", final="Your pump P-2044 is still under warranty.")
    assert S.check_run(verifier, silent, write_tools=SHOP_TOOLS)[0] is False


def test_a_reported_fact_keeps_the_value_the_span_and_a_predicate_that_can_answer(tmp_path):
    """Reported is not dropped: the Verdict can still say whether the Run stated the fact."""
    verifier = pump_verifier("tell the caller about the pump they asked about")
    price = fact_atom(verifier, "249.0")
    assert price.kind == "allowed" and price.provenance == "system_derived"
    assert price.predicate_src and "249.0" in price.predicate_src
    assert price.spans and price.spans[0].msg_index == 2
    assert "rejects no Run" in (price.description or "")


# --- the write cap on a Task whose Reference wrote nothing -----------------

def cap_atoms(verifier: Verifier) -> list[str]:
    return [a.id for a in verifier.atoms if S.atom_payload(a)["kind"] == "entity_count"]


def test_a_reference_that_wrote_nothing_caps_writes_at_zero(tmp_path):
    """Nothing else of this Task wrote, so "do not write" is what its Runs say."""
    verifier = pump_verifier("tell the caller how many warranty months the pump has left")
    assert cap_atoms(verifier) == ["entity_count"]
    assert S.atom_payload(atom_by_id(verifier, "entity_count"))["count"] == 0


def test_a_cap_of_zero_is_not_written_when_another_run_of_the_task_wrote_and_nothing_else_is_asked(tmp_path):
    """The Task's own Runs contradict the cap, and the request named no fact to keep it for."""
    vague = pump_run(asked="Can you look into my pump for me?")
    verifier = V.derive_verifier(Task(id="shop", intent="look into the pump the caller asked about"),
                                 vague, [], None, write_tools=SHOP_TOOLS, writes_elsewhere=True)
    assert cap_atoms(verifier) == []


def test_a_verifier_that_would_ask_nothing_at_all_keeps_the_facts_the_reference_stated(tmp_path):
    """An empty Run has to fail: with no write and no fact named, what was said is the only evidence."""
    vague = pump_run(asked="Can you look into my pump for me?")
    verifier = V.derive_verifier(Task(id="shop", intent="look into the pump the caller asked about"),
                                 vague, [], None, write_tools=SHOP_TOOLS)
    assert stated(verifier, V.REPORTED_COMMUNICATE) == []
    assert stated(verifier, "communicate") == ["249.0", "36", "P-2044"]
    assert "asks nothing else" in (fact_atom(verifier, "249.0").description or "")
    silent = pump_run("shop-alt", asked="Can you look into my pump for me?", final="I have looked.")
    assert S.check_run(verifier, silent, write_tools=SHOP_TOOLS)[0] is False


def test_the_cap_stays_when_the_verifier_asks_for_something_else(tmp_path):
    """It is only the Verifier that is nothing but the cap that the writing Run contradicts."""
    verifier = V.derive_verifier(Task(id="shop", intent="tell the caller how many warranty months are left"),
                                 pump_run(), [], None, write_tools=SHOP_TOOLS, writes_elsewhere=True)
    assert cap_atoms(verifier) == ["entity_count"]
    assert [a.id for a in verifier.atoms if a.kind == "communicate"] == ["c1", "c2"]
